//! **The fiber switch seam** (`res/SPMD.md` §13.3 rule 1) — the
//! cooperative stackful context switch the escape hatch stands on,
//! behind the hint-pattern contract: per-arch impls, one cfg'd alias,
//! an honest floor everywhere a sequence is not proven on its metal.
//!
//! A suspend pushes the FULL callee-saved set onto the fiber's own
//! strip and the saved SP is the *sole* root — that stated discipline
//! is what makes one packed word a complete quad TCB.  This is ring-3
//! machinery: a userland cooperative switch point has dead caller-saved
//! registers (vector state included) and saves none of them; the
//! kernel's own switch path is untouched.

/// The contract: seed a fresh fiber frame, and swap stacks.
pub(crate) trait FiberSwitch {
    /// Whether this target has a proven switch sequence.
    fn available() -> bool;
    /// Lay a synthetic suspended frame at `stack_top` so the first
    /// resume enters `entry(arg)`; returns the parked SP.
    ///
    /// # Safety
    /// `stack_top` must be the 16-aligned top of writable, exclusively
    /// owned stack memory with room for the frame below it; `entry`
    /// must never return.
    unsafe fn seed(stack_top: u64, entry: extern "C" fn(usize) -> !, arg: usize) -> u64;
    /// Save the callee-saved set + SP into `*save`, adopt `to_sp`, and
    /// continue wherever that stack was suspended.  Returns when some
    /// later switch resumes `*save`.
    ///
    /// # Safety
    /// `save` must be a live word this fiber's owner re-adopts later;
    /// `to_sp` must be a SP previously produced by [`seed`](Self::seed)
    /// or by this function's save half, on a live exclusively-owned
    /// stack.
    unsafe fn switch(save: *mut u64, to_sp: u64);
}

/// **No switch** — the honest floor: `available() == false`, and the
/// deck's provisioning refuses rather than pretending.  Selected off
/// the metal x86-64 target; the aarch64/riscv sequences are mechanical
/// siblings behind this same seam, landing when a boot can prove them
/// (the `no mock world` rule).
pub(crate) struct NoSwitch;

impl FiberSwitch for NoSwitch {
    fn available() -> bool {
        false
    }

    unsafe fn seed(_stack_top: u64, _entry: extern "C" fn(usize) -> !, _arg: usize) -> u64 {
        0
    }

    unsafe fn switch(_save: *mut u64, _to_sp: u64) {}
}

#[cfg(all(target_os = "none", target_arch = "x86_64"))]
mod x86_64;

/// This target's switch.
#[cfg(all(target_os = "none", target_arch = "x86_64"))]
pub(crate) use x86_64::X64Switch as Switch;

/// This target's switch — the floor.
#[cfg(not(all(target_os = "none", target_arch = "x86_64")))]
pub(crate) use NoSwitch as Switch;
