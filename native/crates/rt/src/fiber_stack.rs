//! **Guarded fiber stacks** — whole stacks for runtimes that switch onto
//! them with the lane seam ([`crate::lane::switch`]): Wasmtime's async
//! fibers (`crates/wasm-platform`), which park a guest's entire call stack
//! while a host future is pending.
//!
//! The lane tier's own fibers (`lane::fiber`) are 64 KiB quarters in a
//! guardless window, safe by the yield audit their bodies promise. A foreign
//! runtime promises no such thing — a guest's stack depth is bounded by the
//! engine's own limit, and host code below it by nothing at all — so each
//! stack here is a **thread stack's twin**: a guarded strip off its own
//! [`StripPlan`], mapped by one `map_region` session with the guard page in
//! the strip's reserved granule, and handed back by one revoke. It is the
//! exact lifecycle [`thread::join`](crate::thread::join) runs for a worker
//! stack, minus the thread.
//!
//! A stack is **thread-bound**: its `Region` capability lives in the arena
//! of the thread that mapped it, so [`GuardedStack`] is `!Send`. A runtime
//! that demands `Send` carries it in [`Pinned`](crate::pinned::Pinned).

use core::marker::PhantomData;

use crate::stack_arena::{GRANULE, StripPlan};

/// The window (`res/SPMD.md` §13.4's precedent): a pure VA reservation
/// directly above the lane fiber window, proven disjoint from every thread
/// band at compile time beside [`BANDS`](crate::thread::BANDS).
pub(crate) const WINDOW_BASE: u64 = 0x7fff_e200_0000;

/// The window's granule capacity: 32 MiB of reservation, no byte mapped by
/// declaring it. Wasmtime's default async stack is 2 MiB, so fifteen live
/// fibers fit with their guards — each call parks one.
pub(crate) const WINDOW_GRANULES: u32 = 8192;

/// **The process's guarded-stack plan.** Carved by [`GuardedStack::new`],
/// released by its drop; task-wide like every band.
// Carved by the metal arm only; host builds keep it so the plan's
// compile-time geometry asserts run in `make test` too.
#[cfg_attr(not(target_os = "none"), allow(dead_code))]
static PLAN: StripPlan<{ WINDOW_GRANULES as usize / 64 }> =
    StripPlan::new(WINDOW_BASE, WINDOW_GRANULES);

/// One mapped, guarded stack. Dropping it unmaps the pages and frees the
/// strip — on the thread that mapped it (the type is `!Send`).
#[must_use = "a dropped stack is unmapped at once"]
pub struct GuardedStack {
    strip: Option<crate::stack_arena::Strip>,
    #[cfg_attr(not(target_os = "none"), allow(dead_code))]
    region: kore_abi::session::SlotRef,
    _thread_bound: PhantomData<*mut ()>,
}

impl GuardedStack {
    /// Map a fresh zeroed stack of at least `bytes` (rounded up to whole
    /// granules) with an unmapped guard page below it. `None` when the
    /// window is full, the thread has no slot to hold the region, or the
    /// kernel refuses the mapping — and always off the kore target.
    pub fn new(bytes: usize) -> Option<Self> {
        let granules = u32::try_from(bytes.div_ceil(GRANULE)).ok()?.max(1);
        Self::map(granules)
    }

    /// First usable byte.
    pub fn base(&self) -> u64 {
        self.strip.as_ref().map_or(0, |strip| strip.base())
    }

    /// One past the last usable byte — the stack's top (16-aligned: a
    /// granule boundary).
    pub fn top(&self) -> u64 {
        self.strip.as_ref().map_or(0, |strip| strip.end())
    }

    /// The unmapped guard page directly below [`base`](Self::base).
    pub fn guard(&self) -> core::ops::Range<u64> {
        let base = self.base();
        base.saturating_sub(GRANULE as u64)..base
    }

    #[cfg(target_os = "none")]
    fn map(granules: u32) -> Option<Self> {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};
        use kore_abi::cap::Rights;

        let strip = PLAN.carve_guarded(granules)?;
        let Some((self_space, region)) = crate::entry::geometry::self_space_slot()
            .zip(crate::entry::geometry::acquire_cap_slot())
        else {
            PLAN.release(strip);
            return None;
        };
        let mut b = SessionBuilder::new();
        let mapped = b
            .map_region(
                self_space,
                Rights::ACCESS | Rights::MODIFY | Rights::REVOKE,
                strip.base(),
                u64::from(strip.bytes()),
                1,
                0,
                region,
            )
            .is_ok()
            && TrapTransport::new()
                .commit_now(&b)
                .is_ok_and(|done| done.status() == kore_abi::session::Status::Ok);
        if !mapped {
            crate::entry::geometry::release_cap_slot(region);
            PLAN.release(strip);
            return None;
        }
        Some(Self {
            strip: Some(strip),
            region,
            _thread_bound: PhantomData,
        })
    }

    #[cfg(not(target_os = "none"))]
    fn map(_granules: u32) -> Option<Self> {
        None
    }
}

impl Drop for GuardedStack {
    #[cfg(target_os = "none")]
    fn drop(&mut self) {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        // The revoke IS the unmap. Only once it lands are the slot and the
        // strip reusable; a refused revoke keeps both claimed, so no later
        // stack is ever mapped over pages this one still holds.
        let mut b = SessionBuilder::new();
        let unmapped = b.revoke(self.region).is_ok()
            && TrapTransport::new()
                .commit_now(&b)
                .is_ok_and(|done| done.status() == kore_abi::session::Status::Ok);
        if unmapped {
            crate::entry::geometry::release_cap_slot(self.region);
            if let Some(strip) = self.strip.take() {
                PLAN.release(strip);
            }
        }
    }

    #[cfg(not(target_os = "none"))]
    fn drop(&mut self) {}
}
