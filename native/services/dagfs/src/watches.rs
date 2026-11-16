//! Directory watches: the service half of the watch plane
//! (`kore_dagfs::watch`, `res/DAGFS.md` §4).
//!
//! A session arms up to [`FS_WATCH_SLOTS`] watches, each on a directory
//! handle it holds. After every unit of service work each armed directory's
//! change epoch ([`Namespace::change_epoch`]) is read **through the handle's
//! route** — revalidated as every operation's is — and the plane republished
//! when one moved. Only a client parked on its plane is owed a wake; the
//! serving host sends each one notice (`SystemService::take_owed_wakes`).
//!
//! Nothing here is global. A peer's [`Plane`] lives in its serving slot's own
//! record, beside the slot's window and staged replacement, allocated with it
//! and wiped with it; the service keeps only [`WatchIndex`] — which slots have
//! a watch armed, and which owe a wake — so a settle visits exactly the slots
//! that need it, however large the slot space grows. A watch on the dual
//! router's system volume is recorded in the same plane with the store it
//! lives in, so the one plane a client holds carries both.
//!
//! [`Namespace::change_epoch`]: crate::namespace::Namespace::change_epoch

use kore_abi::svc::{FS_WATCH_SLOTS, FsWatchState};
use kore_dagfs::watch::{self, WatchSource, state};
use kore_rt::svc::PeerId;

use crate::{
    ObjectId,
    authority::{Handle, SessionId},
    bounds::PAIR_SESSIONS,
    error::{FsError, FsResult},
    router::Store,
};

/// One armed slot.
#[derive(Clone, Copy, Debug)]
struct Watch {
    store: Store,
    handle: Handle,
    dir: ObjectId,
    /// The epoch the plane last reported for this watch.
    seen: u64,
}

/// One recyclable peer's watch plane and the watches published into it.
pub(crate) struct Plane {
    peer: PeerId,
    source: WatchSource<'static>,
    slots: [Option<Watch>; FS_WATCH_SLOTS],
    /// The plane's state changed since its last commit.
    dirty: bool,
}

impl Plane {
    /// `peer`'s plane over its pair window's tail (`aux`, as the host hands it
    /// to `attach_peer_aux`), or `None` for a tail without a formatted plane.
    /// Starts from nothing armed, whatever the window last carried.
    pub(crate) fn adopt(peer: PeerId, aux: &'static mut [u8]) -> Option<Self> {
        let mut source = watch::source(aux).ok()?;
        if source.replace_state(FsWatchState::default()) {
            let _ = source.commit();
        }
        Some(Self {
            peer,
            source,
            slots: [None; FS_WATCH_SLOTS],
            dirty: false,
        })
    }

    pub(crate) const fn peer(&self) -> PeerId {
        self.peer
    }

    /// Whether any slot is armed.
    pub(crate) fn armed(&self) -> bool {
        self.slots.iter().any(Option::is_some)
    }

    fn edit(&mut self, edit: impl FnOnce(&mut FsWatchState)) {
        self.source.state(edit);
        self.dirty = true;
    }

    /// Publish a batch: `true` when its commit found the client parked, and
    /// so owes it one notice.
    fn commit(&mut self) -> bool {
        core::mem::take(&mut self.dirty) && self.source.commit()
    }

    /// Arm `slot` under `tag` on the directory `dir` at epoch `seen` (its
    /// baseline: only a later change counts), replacing what the slot held.
    /// Answers whether a parked client is owed a wake.
    pub(crate) fn arm(
        &mut self,
        slot: u8,
        tag: u32,
        store: Store,
        handle: Handle,
        (dir, seen): (ObjectId, u64),
    ) -> FsResult<bool> {
        let slot = usize::from(slot);
        if slot >= FS_WATCH_SLOTS || tag == 0 {
            return Err(FsError::Range);
        }
        self.slots[slot] = Some(Watch {
            store,
            handle,
            dir,
            seen,
        });
        self.edit(|s| state::arm(s, slot, tag));
        Ok(self.commit())
    }

    /// Disarm `slot`, armed or not. Answers whether a parked client is owed
    /// a wake.
    pub(crate) fn disarm(&mut self, slot: u8) -> FsResult<bool> {
        let slot = usize::from(slot);
        if slot >= FS_WATCH_SLOTS {
            return Err(FsError::Range);
        }
        self.slots[slot] = None;
        self.edit(|s| state::disarm(s, slot));
        Ok(self.commit())
    }

    /// Read every armed watch's epoch through `epoch` — the engine of its
    /// store, revalidating its route — and publish what moved. A watch whose
    /// route no longer proves ends; one whose directory was mid-publish
    /// (`Busy`) counts as changed and keeps its baseline, so the next reading
    /// reports the change again rather than losing it. Answers whether a
    /// parked client is owed a wake.
    pub(crate) fn settle(
        &mut self,
        session: SessionId,
        mut epoch: impl FnMut(Store, SessionId, Handle, ObjectId) -> FsResult<u64>,
    ) -> bool {
        for slot in 0..FS_WATCH_SLOTS {
            let Some(watch) = self.slots[slot].as_mut() else {
                continue;
            };
            match epoch(watch.store, session, watch.handle, watch.dir) {
                Ok(now) if now == watch.seen => {}
                Ok(now) => {
                    watch.seen = now;
                    self.edit(|s| state::changed(s, slot));
                }
                Err(FsError::Busy) => self.edit(|s| state::changed(s, slot)),
                Err(_) => {
                    self.slots[slot] = None;
                    self.edit(|s| state::end(s, slot));
                }
            }
        }
        self.commit()
    }
}

/// Words of a set over every pair session's slot.
const WORDS: usize = PAIR_SESSIONS.div_ceil(64);

/// A set of pair-session slots: one bit each, scanned word by word.
#[derive(Clone, Copy, Debug, Default)]
struct SlotSet([u64; WORDS]);

impl SlotSet {
    fn insert(&mut self, slot: usize) {
        if let Some(word) = self.0.get_mut(slot / 64) {
            *word |= 1 << (slot % 64);
        }
    }

    fn remove(&mut self, slot: usize) {
        if let Some(word) = self.0.get_mut(slot / 64) {
            *word &= !(1 << (slot % 64));
        }
    }

    fn is_empty(&self) -> bool {
        self.0.iter().all(|word| *word == 0)
    }

    /// The members, lowest first, as of the call.
    fn members(&self) -> impl Iterator<Item = usize> + use<> {
        let words = self.0;
        (0..WORDS).flat_map(move |at| {
            let mut word = words[at];
            core::iter::from_fn(move || {
                (word != 0).then(|| {
                    let bit = word.trailing_zeros() as usize;
                    word &= word - 1;
                    at * 64 + bit
                })
            })
        })
    }
}

/// Which slots' planes have a watch armed, and which owe their client a wake.
/// Derived from the planes and kept beside them: a settle and a wake
/// collection cost the slots they name, and nothing when the sets are empty.
#[derive(Debug, Default)]
pub(crate) struct WatchIndex {
    armed: SlotSet,
    owed: SlotSet,
}

impl WatchIndex {
    pub(crate) const fn new() -> Self {
        Self {
            armed: SlotSet([0; WORDS]),
            owed: SlotSet([0; WORDS]),
        }
    }

    /// Record what an edit of `slot`'s plane left: whether a watch is still
    /// armed there, and whether its client is owed a wake.
    pub(crate) fn note(&mut self, slot: usize, armed: bool, owed: bool) {
        if armed {
            self.armed.insert(slot);
        } else {
            self.armed.remove(slot);
        }
        if owed {
            self.owed.insert(slot);
        }
    }

    /// Forget `slot`: its plane is gone.
    pub(crate) fn forget(&mut self, slot: usize) {
        self.armed.remove(slot);
        self.owed.remove(slot);
    }

    /// The slots with a watch armed.
    pub(crate) fn armed(&self) -> impl Iterator<Item = usize> + use<> {
        self.armed.members()
    }

    pub(crate) fn idle(&self) -> bool {
        self.armed.is_empty()
    }

    /// Hand `out` each slot owing a wake, lowest first, taking it off the set;
    /// `out` answers `false` when its batch is full, and the rest stay owed.
    pub(crate) fn take_owed(&mut self, mut out: impl FnMut(usize) -> bool) {
        for slot in self.owed.members() {
            if !out(slot) {
                return;
            }
            self.owed.remove(slot);
        }
    }

    #[cfg(test)]
    pub(crate) fn armed_count(&self) -> usize {
        self.armed.members().count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_slot_set_names_its_members_lowest_first_across_words() {
        let mut set = SlotSet::default();
        for slot in [PAIR_SESSIONS - 1, 3, 64, 65, 0] {
            set.insert(slot);
        }
        set.insert(PAIR_SESSIONS); // past the space: ignored
        let members: std::vec::Vec<usize> = set.members().collect();
        assert_eq!(members, [0, 3, 64, 65, PAIR_SESSIONS - 1]);
        set.remove(64);
        set.remove(0);
        assert_eq!(set.members().collect::<std::vec::Vec<_>>(), [3, 65, PAIR_SESSIONS - 1]);
        assert!(!set.is_empty());
        for slot in [3, 65, PAIR_SESSIONS - 1] {
            set.remove(slot);
        }
        assert!(set.is_empty());
    }

    #[test]
    fn owed_wakes_are_taken_once_and_a_full_batch_keeps_the_rest() {
        let mut index = WatchIndex::new();
        for slot in [2, 7, 9] {
            index.note(slot, true, true);
        }
        let mut taken = std::vec::Vec::new();
        index.take_owed(|slot| {
            if taken.len() == 2 {
                return false;
            }
            taken.push(slot);
            true
        });
        assert_eq!(taken, [2, 7]);
        taken.clear();
        index.take_owed(|slot| {
            taken.push(slot);
            true
        });
        assert_eq!(taken, [9], "the batch's leftover, and nothing twice");
        index.forget(9);
        assert_eq!(index.armed().collect::<std::vec::Vec<_>>(), [2, 7]);
        index.note(2, false, false);
        index.note(7, false, false);
        assert!(index.idle());
    }
}
