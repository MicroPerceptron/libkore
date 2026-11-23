//! The **watch plane**: a filesystem pair's change channel (`res/EVENT.md`).
//!
//! A client that shows a directory — the Files explorer — needs to learn when
//! what it shows went stale without asking. The plane is the
//! service→client direction of one pair, as the window pair's event channel is
//! the compositor→client one: an event channel formatted after the
//! request/reply rings in the pair's **own** window, so it rides the lease
//! both sides already hold and needs no capability of its own.
//!
//! - **State is level.** [`FsWatchState`] holds, per slot, the tag the slot
//!   was armed with and a wrapping count of changes since; the service
//!   republishes it whole. Several changes between two service passes advance
//!   a count once — a client lists again, it does not replay.
//! - **The wake is conditional.** The service commits after the request (or
//!   teardown) that changed a watched directory and owes the client a notice
//!   only if it had parked on the plane; a client that is running pays the
//!   service nothing.
//! - **No authority, no names.** A watch is armed on a directory handle the
//!   client holds; the service revalidates that handle's route before every
//!   count and ends the watch the moment it stops proving. The plane says
//!   *that* a listing may differ, never *what* it holds — the client lists
//!   through its own handle.
//!
//! The record ring is reserved ([`FS_WATCH_RECORDS`], never pushed) so a later
//! per-entry change record lands without moving the state or the offset.

use kore_abi::svc::{FS_WATCH_RECORDS, FS_WATCH_SLOTS, FsWatchRecord, FsWatchState};
use kore_ring::event::{EventSink, EventSource};
use kore_ring::spsc::RingError;
use kore_rt::svc::{REP_CAP, REQ_CAP, SvcProtocol};

use crate::DagfsProtocol;

/// The producer half: the service's.
pub type WatchSource<'a> = EventSource<'a, FsWatchState, FsWatchRecord>;
/// The consumer half: the client's.
pub type WatchSink<'a> = EventSink<'a, FsWatchState, FsWatchRecord>;

/// Where the plane begins in a pair window: right after the canonical
/// request/reply rings (both 64-byte rounded, so the plane is line-aligned).
pub const PLANE_AT: usize = kore_rt::svc::window_bytes::<DagfsProtocol>(
    <DagfsProtocol as SvcProtocol>::REQUEST_CAPACITY,
    <DagfsProtocol as SvcProtocol>::RESPONSE_CAPACITY,
);

/// Bytes the plane occupies.
pub const PLANE_BYTES: usize =
    kore_ring::event::region_bytes::<FsWatchState, FsWatchRecord>(FS_WATCH_RECORDS);

/// The smallest pair window that carries the plane.
pub const WINDOW_BYTES: usize = PLANE_AT + PLANE_BYTES;

// Whoever sizes a filesystem pair window asserts it holds `WINDOW_BYTES`
// against its own constant (exec's `FS_CONTROL_BYTES`); the formatter refuses
// a smaller window at founding. Line alignment is the layout's own invariant.
const _: () = assert!(PLANE_AT % 64 == 0);
const _: () = assert!(
    <DagfsProtocol as SvcProtocol>::REQUEST_CAPACITY == REQ_CAP
        && <DagfsProtocol as SvcProtocol>::RESPONSE_CAPACITY == REP_CAP
);

/// Format a whole filesystem pair window — the rings, then the plane — before
/// either side maps it. The protocol's [`SvcProtocol::init_pair_window`].
pub fn init_pair_window(window: &mut [u8], req_cap: u32, rep_cap: u32) -> Result<(), RingError> {
    kore_rt::svc::init_window::<DagfsProtocol>(window, req_cap, rep_cap)?;
    let at = kore_rt::svc::window_bytes::<DagfsProtocol>(req_cap, rep_cap);
    let plane = window.get_mut(at..).ok_or(RingError::TooSmall)?;
    kore_ring::event::init::<FsWatchState, FsWatchRecord>(plane, FS_WATCH_RECORDS)
}

/// The service's half over its mapping of the pair window's tail (`aux` is
/// what follows the rings, as `attach_peer_aux` hands it over).
pub fn source(aux: &mut [u8]) -> Result<WatchSource<'_>, RingError> {
    kore_ring::event::source(aux.get_mut(..PLANE_BYTES).ok_or(RingError::TooSmall)?)
}

/// The client's half over its mapping of the same tail.
pub fn sink(aux: &mut [u8]) -> Result<WatchSink<'_>, RingError> {
    kore_ring::event::sink(aux.get_mut(..PLANE_BYTES).ok_or(RingError::TooSmall)?)
}

/// What one slot's watch did since a [`Cursor`] last looked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Seen {
    /// Nothing the cursor has not seen.
    Same,
    /// The directory changed: list it again.
    Changed,
    /// The watch ended (its route stopped proving, or the directory went):
    /// what was shown cannot be refreshed through this watch.
    Ended,
}

/// A client's view of one slot: the tag it armed and the last count it acted
/// on. A slot is read only under the cursor's own tag, so the counts the
/// slot carried for an earlier arming — or for nobody — never read as news.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cursor {
    slot: u8,
    tag: u32,
    seen: u32,
    ended: bool,
}

impl Cursor {
    /// A cursor for `slot` armed with `tag`; `None` for a slot past the plane
    /// or the zero tag (the plane's "disarmed").
    pub const fn new(slot: u8, tag: u32) -> Option<Self> {
        if slot as usize >= FS_WATCH_SLOTS || tag == 0 {
            return None;
        }
        Some(Self {
            slot,
            tag,
            seen: 0,
            ended: false,
        })
    }

    pub const fn slot(&self) -> u8 {
        self.slot
    }

    pub const fn tag(&self) -> u32 {
        self.tag
    }

    /// Compare `state` with what this cursor last acted on, and take it as
    /// seen. A state still showing another tag in the slot (the arming has
    /// not landed, or the slot was re-armed by someone else's request on
    /// this pair) is [`Seen::Same`]: nothing of this watch is in it yet.
    pub fn observe(&mut self, state: &FsWatchState) -> Seen {
        let slot = usize::from(self.slot);
        if state.tag[slot] != self.tag {
            return Seen::Same;
        }
        if state.ended & (1 << slot) != 0 {
            if self.ended {
                return Seen::Same;
            }
            self.ended = true;
            return Seen::Ended;
        }
        let changes = state.changes[slot];
        if changes == self.seen {
            return Seen::Same;
        }
        self.seen = changes;
        Seen::Changed
    }
}

/// The plane's state as a producer edits it — the service's one batch verb
/// over a slot. Kept beside the layout so both halves read one definition.
pub mod state {
    use super::{FS_WATCH_SLOTS, FsWatchState};

    /// Arm `slot` under `tag`: no changes yet, not ended.
    pub fn arm(state: &mut FsWatchState, slot: usize, tag: u32) {
        if slot < FS_WATCH_SLOTS {
            state.tag[slot] = tag;
            state.changes[slot] = 0;
            state.ended &= !(1 << slot);
        }
    }

    /// Disarm `slot`.
    pub fn disarm(state: &mut FsWatchState, slot: usize) {
        if slot < FS_WATCH_SLOTS {
            state.tag[slot] = 0;
            state.changes[slot] = 0;
            state.ended &= !(1 << slot);
        }
    }

    /// One more change under `slot`'s arming.
    pub fn changed(state: &mut FsWatchState, slot: usize) {
        if slot < FS_WATCH_SLOTS {
            state.changes[slot] = state.changes[slot].wrapping_add(1);
        }
    }

    /// `slot`'s watch ended; its tag stays, so the client that armed it
    /// learns that it did.
    pub fn end(state: &mut FsWatchState, slot: usize) {
        if slot < FS_WATCH_SLOTS {
            state.ended |= 1 << slot;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec;

    #[repr(align(64))]
    struct Window([u8; WINDOW_BYTES]);

    #[test]
    fn the_plane_follows_the_rings_and_both_halves_attach() {
        let mut window = Window([0; WINDOW_BYTES]);
        init_pair_window(&mut window.0, REQ_CAP, REP_CAP).unwrap();
        // The rings are exactly where every existing client expects them.
        let (pair, aux) = window.0.split_at_mut(PLANE_AT);
        assert!(kore_rt::svc::client_rings::<DagfsProtocol>(pair, REQ_CAP, REP_CAP).is_ok());
        let base = aux.as_mut_ptr();
        // SAFETY: the two halves alias the plane only through the channel
        // protocol, as the cross-space form does over two mappings.
        let alias = unsafe { core::slice::from_raw_parts_mut(base, PLANE_BYTES) };
        let mut source = source(aux).unwrap();
        let sink = sink(alias).unwrap();
        assert_eq!(sink.state(), Some(FsWatchState::default()));
        source.state(|s| state::arm(s, 1, 7));
        assert!(!source.commit(), "nobody parked: no notice owed");
        let seen = sink.seq();
        assert!(sink.park(seen));
        source.state(|s| state::changed(s, 1));
        assert!(source.commit(), "a parked client is owed its notice");
        assert_eq!(sink.state().unwrap().changes[1], 1);
    }

    #[test]
    fn a_window_too_small_for_the_plane_is_refused() {
        let mut short = vec![0u8; PLANE_AT + 64 + 64];
        let at = short.as_ptr().align_offset(64);
        let window = &mut short[at..at + PLANE_AT];
        assert_eq!(init_pair_window(window, REQ_CAP, REP_CAP), Err(RingError::TooSmall));
    }

    #[test]
    fn a_cursor_reads_only_its_own_arming() {
        let mut cursor = Cursor::new(2, 9).unwrap();
        let mut s = FsWatchState::default();
        // The slot's previous watch left counts: not this cursor's news.
        state::arm(&mut s, 2, 8);
        state::changed(&mut s, 2);
        assert_eq!(cursor.observe(&s), Seen::Same);
        state::arm(&mut s, 2, 9);
        assert_eq!(cursor.observe(&s), Seen::Same);
        state::changed(&mut s, 2);
        state::changed(&mut s, 2);
        assert_eq!(cursor.observe(&s), Seen::Changed, "two changes coalesce");
        assert_eq!(cursor.observe(&s), Seen::Same);
        state::end(&mut s, 2);
        assert_eq!(cursor.observe(&s), Seen::Ended);
        assert_eq!(cursor.observe(&s), Seen::Same, "an ending is reported once");
        assert!(Cursor::new(FS_WATCH_SLOTS as u8, 1).is_none());
        assert!(Cursor::new(0, 0).is_none());
    }
}
