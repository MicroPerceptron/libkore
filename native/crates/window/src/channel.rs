//! The window pair's **event channel** — the compositor→client direction
//! as `res/EVENT.md` models it: level state read from memory, edge
//! records on the ring, one conditional wake.
//!
//! The channel replaces the plain event ring in the pair window's tail
//! *and* the per-frame status/configure round trips: everything a client
//! used to ask the compositor per frame (`Status`, `Configure`
//! re-validation) is a field of [`WindowState`], republished by the
//! compositor when it changes and read by the client at memory cost.
//! The [`Event`](crate::event::Event) records ride the channel's ring
//! unchanged — they are the edges (an input, a close request, a tab
//! activation) that must not coalesce, while the state is the level a
//! late reader may observe already-current.
//!
//! Direction discipline (SPSC, like every channel): the compositor is
//! the one producer, the client the one consumer.  The client→compositor
//! direction keeps its existing planes — the commit ring for frames, the
//! request ring for commands — whose wake is the pair's request kick.

use kore_ring::event::{EventSink, EventSource};
use kore_ring::spsc::RingError;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::configure::{Bounds, Configure};
use crate::event::{EVENT_CAP, EventWire};

/// The window protocol's level state — one seqlock'd block the
/// compositor republishes whole.
///
/// Field semantics:
///
/// - `configure_serial` / `window` / `bounds_*` describe the latest client
///   aperture size contract: a client re-validates its serial from memory,
///   and treats a change exactly as it treated
///   [`Event::Configure`](crate::event::Event::Configure) — the record still
///   arrives as the edge. A compositor-only move keeps the serial and may
///   leave `bounds_x` / `bounds_y` at that size epoch's origin; clients do not
///   need screen coordinates and therefore receive no wake for pure motion.
///   [`FOCUSED`](Self::FOCUSED) always reflects the current focus level.
/// - `surface` names the backing whose books follow: the peer's **active**
///   surface, `0` while none is.  `presented` and `refused_commits` are
///   that surface's engine books (monotone per surface).  A client whose
///   current backing is not yet active reads its books as fresh
///   (`presented = 0`, `refused = 0`), which is exactly what a surface
///   with no consumed commit has.
/// - [`VISIBLE`](Self::VISIBLE) is clear while nothing of the window can
///   reach the screen — minimized, closing, or fully occluded under a
///   higher window's opaque core: a client may stop producing frames and
///   park on the channel until it flips back.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable,
)]
pub struct WindowState {
    pub configure_serial: u64,
    pub window: u64,
    pub surface: u64,
    pub presented: u64,
    pub refused_commits: u64,
    pub bounds_x: u16,
    pub bounds_y: u16,
    pub bounds_w: u16,
    pub bounds_h: u16,
    pub flags: u32,
    pub reserved: u32,
}

impl WindowState {
    /// The window holds input focus.
    pub const FOCUSED: u32 = 1 << 0;
    /// Some pixel of the window can reach the screen; while clear, a
    /// client may pause frame production and park on the channel.
    pub const VISIBLE: u32 = 1 << 1;
    /// Resolved desktop appearance. Absence preserves the dark fallback.
    pub const LIGHT: u32 = 1 << 2;

    pub fn color_scheme(&self) -> crate::appearance::ColorScheme {
        if self.flags & Self::LIGHT != 0 {
            crate::appearance::ColorScheme::Light
        } else {
            crate::appearance::ColorScheme::Dark
        }
    }

    /// The state's [`Configure`] view, `None` before the first publish
    /// (serial 0 is never a valid configuration).
    pub fn configure(&self) -> Option<Configure> {
        (self.configure_serial != 0).then_some(Configure {
            window: self.window,
            serial: self.configure_serial,
            bounds: Bounds {
                x: self.bounds_x,
                y: self.bounds_y,
                width: self.bounds_w,
                height: self.bounds_h,
            },
            focused: self.flags & Self::FOCUSED != 0,
        })
    }

    /// Record `config` as the current configuration.
    pub fn set_configure(&mut self, config: &Configure) {
        self.configure_serial = config.serial;
        self.window = config.window;
        self.bounds_x = config.bounds.x;
        self.bounds_y = config.bounds.y;
        self.bounds_w = config.bounds.width;
        self.bounds_h = config.bounds.height;
        if config.focused {
            self.flags |= Self::FOCUSED;
        } else {
            self.flags &= !Self::FOCUSED;
        }
    }
}

/// The compositor's half.
pub type WindowChannelSource<'a> = EventSource<'a, WindowState, EventWire>;
/// The client's half.
pub type WindowChannelSink<'a> = EventSink<'a, WindowState, EventWire>;

/// Bytes the channel occupies in the pair window's tail.
pub const fn region_bytes() -> usize {
    kore_ring::event::region_bytes::<WindowState, EventWire>(EVENT_CAP)
}

/// Format `window` as an empty channel (the pair-window init path).
pub fn init(window: &mut [u8]) -> Result<(), RingError> {
    kore_ring::event::init::<WindowState, EventWire>(window, EVENT_CAP)
}

/// Attach the compositor's half over its mapping of the pair tail.
pub fn source(window: &mut [u8]) -> Result<WindowChannelSource<'_>, RingError> {
    kore_ring::event::source(window)
}

/// Attach the client's half over its mapping of the pair tail.
pub fn sink(window: &mut [u8]) -> Result<WindowChannelSink<'_>, RingError> {
    kore_ring::event::sink(window)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerocopy::FromZeros;

    #[test]
    fn the_state_round_trips_a_configuration() {
        let mut state = WindowState::new_zeroed();
        assert_eq!(state.configure(), None, "serial 0 is not a configuration");
        let config = Configure {
            window: 7,
            serial: 3,
            bounds: Bounds {
                x: 22,
                y: 8,
                width: 640,
                height: 400,
            },
            focused: true,
        };
        state.set_configure(&config);
        state.flags |= WindowState::VISIBLE;
        assert_eq!(state.configure(), Some(config));
        state.set_configure(&Configure {
            focused: false,
            ..config
        });
        assert_eq!(
            state.flags,
            WindowState::VISIBLE,
            "focus clears without disturbing visibility"
        );
    }

    /// The whole pair — request/reply rings, commit plane, terminal
    /// output, and this channel — fits the frozen pair lease.
    #[test]
    fn the_pair_window_still_fits_its_lease() {
        let pair = kore_rt::svc::window_bytes::<crate::control::WindowWire>(
            kore_rt::svc::REQ_CAP,
            kore_rt::svc::REP_CAP,
        );
        let commits = kore_ring::spsc::region_bytes::<kore_abi::svc::CommitWire>(crate::COMMIT_CAP);
        let output = kore_ring::spsc::region_bytes::<kore_abi::svc::OutChunk>(
            crate::wire::TERMINAL_OUTPUT_CAP,
        );
        let total = pair + commits + output + region_bytes();
        assert!(
            total <= crate::wire::SURFACE_PAIR_BYTES,
            "pair layout overflows the lease: {total:#x} > {:#x}",
            crate::wire::SURFACE_PAIR_BYTES
        );
    }
}
