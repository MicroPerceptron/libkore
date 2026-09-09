//! AArch64 address monitor — the **exclusive monitor** plus `WFE`.
//!
//! The oldest and cheapest of the three. `LDXR` arms this core's exclusive
//! monitor on the watched word; a store to that line by any observer clears
//! the monitor, and clearing it raises an event that ends `WFE`. This is the
//! spin-lock idiom the architecture was designed around, not a late addition.
//!
//! # Nothing to detect
//!
//! `LDXR`, `WFE`, `SEV`, `SEVL`, and `CLREX` are **ARMv8.0 baseline** and
//! execute at EL0. There is no feature bit and no ID register, so nothing has
//! to be *probed* — [`available`](AddressMonitor::available) is a compile-time
//! constant here, which no other target manages. The one thing the kernel must
//! still do is *ungate* the EL0 `WFE` trap (`SCTLR_EL1.nTWE`) — a fixed
//! precondition rather than a detection, met at bring-up; see
//! [`available`](AddressMonitor::available).
//!
//! The timed variant `WFET` (FEAT_WFxT, ARMv8.7) takes a deadline operand, but
//! selecting it requires an EL1 feature probe plus a founder-published counter
//! contract. Neither is available to this runtime yet. Baseline `WFE` remains
//! implemented as the eventual fast tier, but the capped engine declines it
//! until that independent hard bound exists.
//!
//! # What bounds the wait
//!
//! Baseline `WFE` has none. It is *permitted* to complete for any reason or
//! none, but is not *required* to. In ordinary operation it normally ends on:
//!
//! 1. **The event being waited for** — the waker's store clears the exclusive
//!    monitor, which raises the event. This is the intended path.
//! 2. **Any unmasked interrupt**, including the scheduler tick.
//! 3. Permitted spurious completion.
//!
//! Those are progress sources, not a deadline: neither an absent peer store
//! nor a later scheduler tick proves return within the `Duration` supplied to
//! [`wait_on`](super::wait_on). Consequently
//! [`AddressMonitor::deadline_after`] returns `None` and the generic engine
//! spins. The arm/sleep/disarm implementation stays in place so a future
//! bounded AArch64 encoding can enable the fast tier without changing any
//! caller.
//!
//! # `SEVL` is deliberately absent
//!
//! The canonical ARM spin-wait opens with `SEVL` so the *first* `WFE` falls
//! straight through, because that idiom sleeps before it has armed anything.
//! This seam's engine always arms and re-checks before sleeping, so priming
//! the event register would only guarantee the first sleep does nothing.

use core::time::Duration;

use super::AddressMonitor;

/// **Whether the kernel guarantees EL0 `WFE` does not trap.**
///
/// One constant so the switch is a single, reviewable edit rather than logic
/// spread across the impl. `true` because aarch64 per-CPU bring-up
/// unconditionally sets `SCTLR_EL1.nTWE` on the BSP and every AP —
/// `ungate_el0_wfe`, folded into `init_extstate`
/// (`src/platform/arch/arm64/mod.rs`) — so an EL0 `WFE` executes natively on
/// every PE this kernel brings up rather than trapping to EL1.
const NTWE_GUARANTEED: bool = true;

/// The AArch64 address monitor.
pub struct A64Monitor;

impl AddressMonitor for A64Monitor {
    /// **No operand.** The wait is ended by the event, an interrupt, or the
    /// implementation's discretion — none of which the caller names.
    type Deadline = ();

    fn name() -> &'static str {
        "wfe"
    }

    /// **`true` — the kernel guarantees `nTWE`.**
    ///
    /// The instructions are ARMv8.0 baseline, so nothing about the *hardware*
    /// gates this backend; the one precondition is the *platform* one, and the
    /// kernel now meets it. An EL0 `WFE` that would suspend traps to EL1 unless
    /// `SCTLR_EL1.nTWE` is set, and EL0 cannot read `SCTLR_EL1` to check — so
    /// this can only ever be a promise the kernel makes. It does: aarch64
    /// per-CPU bring-up sets `nTWE` on the BSP and every AP before either
    /// enters EL0 (`ungate_el0_wfe`, in `src/platform/arch/arm64/mod.rs`).
    /// This is the `CR4.TSD` precedent on x86, cleared by per-CPU chrono
    /// bring-up so `clock` can rely on `RDTSC` at EL0 without probing for it.
    ///
    /// Were the bit *not* guaranteed, answering `true` would be exactly the
    /// fail-open this trait's contract forbids — and not theoretically:
    /// `wait_on` is on the futex's pre-block path, which the input relay's
    /// `settle_note` runs on every boot, so a clear `nTWE` would turn every
    /// startup into an EL0 trap. The kernel's unconditional set is what earns
    /// the `true`.
    fn available() -> bool {
        NTWE_GUARANTEED
    }

    /// Baseline `WFE` cannot express the caller's hard cap. An interrupt or a
    /// spurious completion may end it early, but neither is guaranteed within
    /// `cap`; decline the monitor so the arch-neutral engine uses its bounded
    /// spin floor instead of turning an optimization into a possible stall.
    fn deadline_after(_cap: Duration, _counter_hz: u64) -> Option<()> {
        None
    }

    /// # Safety
    /// Per the trait: `addr` names a live, readable, aligned word. `LDXR`
    /// additionally requires natural alignment, which an `AtomicU32` has.
    unsafe fn arm(addr: *const u32) {
        // SAFETY: a word-aligned load-exclusive from a live address. The
        // destination is discarded — the value is re-read by the caller
        // through the atomic; what is wanted here is the side effect of
        // arming this core's exclusive monitor on the line.
        unsafe {
            core::arch::asm!(
                "ldxr {tmp:w}, [{addr}]",
                tmp = out(reg) _,
                addr = in(reg) addr,
                options(nostack),
            );
        }
    }

    /// # Safety
    /// Per the trait: must follow [`arm`](Self::arm) on this core.
    unsafe fn sleep(_deadline: ()) {
        // SAFETY: a hint instruction with no memory operand. Executable at
        // EL0 given `SCTLR_EL1.nTWE`; see `available`.
        unsafe {
            core::arch::asm!("wfe", options(nomem, nostack));
        }
    }

    /// Baseline `WFE` carries no deadline, so a pause it cannot bound is one
    /// it declines — `available` already gates this to unreachable, and the
    /// engine spins instead. `WFET` would change that when a published counter
    /// contract exists to bound it against.
    unsafe fn pause(_deadline: ()) {}

    /// **`CLREX`** — and this is the target the trait's `disarm` exists for.
    ///
    /// `LDXR` opens an exclusive sequence, not merely a watch. Abandoning it
    /// without `CLREX` leaves this core's local monitor set, and a later
    /// unrelated `STXR` could then observe a reservation this code made and
    /// never intended to complete. x86 needs nothing here; AArch64 is why the
    /// method is required rather than defaulted to nothing.
    ///
    /// # Safety
    /// Per the trait; `CLREX` is unconditionally safe to execute.
    unsafe fn disarm() {
        // SAFETY: clears the local exclusive monitor; no operands, no memory.
        unsafe {
            core::arch::asm!("clrex", options(nomem, nostack));
        }
    }
}
