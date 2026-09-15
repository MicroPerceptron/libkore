//! RISC-V address monitor — the **reservation set** plus `WRS.STO`.
//!
//! The same shape as the aarch64 backend, one ISA over: `LR.W` arms this hart's
//! reservation set on the watched word, a store to that set by any hart clears
//! the reservation, and a cleared reservation ends the wait. Load-reserved and
//! wait-on-reservation are the pieces RISC-V's own spin-wait was built from, so
//! this is the architecture's intended idiom rather than a bolted-on primitive.
//!
//! # Zawrs is an extension, so this one *is* detected
//!
//! This is where RISC-V parts from aarch64. `LR.W`/`SC.W` are baseline — the
//! `A` extension, present in `rv64gc` — but the *wait*, `WRS.NTO`/`WRS.STO`,
//! lives in **Zawrs**, a separate extension a hart need not implement. So
//! unlike aarch64's `WFE`, whose presence is an ARMv8.0 constant, `WRS`
//! presence is a runtime fact that has to be detected — and U-mode cannot
//! detect it. The bit is in `misa` / the ISA string, readable only at S/M-mode,
//! so a process can no more discover Zawrs on its own than an x86 process can
//! skip `CPUID`.
//!
//! The kernel does not yet publish that fact. RISC-V is aspirational in this
//! tree — there is no board it boots on and no feature channel a `hint` caller
//! could read — so [`available`](WrsMonitor::available) fails **closed** (see
//! [`ZAWRS_AVAILABLE`]) and every call site spins, exactly as it did before
//! this backend existed. The sequences below are written, assembled, and ready;
//! flipping one constant makes them live the day the kernel grows Zawrs
//! detection, the same one-line switch as aarch64's `nTWE` constant was.
//!
//! # Why `WRS.STO`, not `WRS.NTO`
//!
//! Two reasons, pointing the same way.
//!
//! 1. **It is capped.** `WRS.STO` stalls only for an implementation-defined
//!    *short* timeout before returning on its own; `WRS.NTO` has no bound at
//!    all. This seam offers only capped waits, and `WRS.STO`'s architectural
//!    timeout is the riscv entry in that table — the counterpart of aarch64's
//!    licence-to-wake-spuriously, needing no operand from the caller.
//! 2. **It needs no privileged enable.** `WRS.NTO` traps to a higher privilege
//!    when `mstatus.TW` (or `hstatus.VTW`) is set and it overruns — the direct
//!    analogue of an EL0 `WFE` trapping without `nTWE`. `WRS.STO` is **exempt
//!    from `TW`**: because it always terminates within its short timeout, the
//!    architecture never traps it, so it runs at U-mode with nothing for the
//!    kernel to ungate. That is why this backend, unlike aarch64's, ships with
//!    **no kernel change at all** — only the detection it is parked behind.
//!
//! `WRS.NTO`'s uncapped form is the better choice for a single-word loop *on
//! validated silicon*, exactly as on x86 and aarch64, and unavailable here for
//! the same reason: nothing in this tree can execute it yet.
//!
//! # `CLREX` has no counterpart, and needs none
//!
//! aarch64's `disarm` retires an abandoned `LDXR` with `CLREX`, because a stray
//! exclusive monitor can be observed by a later unrelated `STXR`. RISC-V has no
//! `CLREX` and wants none: a reservation is not an open exclusive *sequence*,
//! it is a set that any store clears, that any trap or context switch clears,
//! and that a later `SC` without its own `LR` simply fails against. An
//! abandoned `LR.W` reservation is therefore inert — the next legitimate
//! `LR`/`SC` pair re-establishes its own — so [`disarm`](WrsMonitor::disarm) is
//! the genuine no-op the trait permits, stated rather than defaulted.

use core::time::Duration;

use super::AddressMonitor;

/// **Whether the kernel has told userland this hart implements Zawrs.**
///
/// One constant, so the switch is a single reviewable edit rather than logic
/// spread across the impl — the shape aarch64's `NTWE_GUARANTEED` uses. `false`
/// because nothing in this tree can answer the question yet: Zawrs presence
/// lives in `misa`, which U-mode cannot read, and the kernel publishes no
/// feature channel for it (RISC-V boots on no board here). Fail **closed** — a
/// wrong `true` would take an illegal-instruction trap on the first `wrs.sto`,
/// not degrade to a spin.
///
/// This is *not* aarch64's story of a trap the kernel must clear: `WRS.STO`
/// needs no privileged enable — it is exempt from `mstatus.TW` (see the module
/// docs). The only thing between this backend and `true` is a kernel-published
/// *detection*, so this flips to a real runtime check — never a bare `true` —
/// the day one exists.
const ZAWRS_AVAILABLE: bool = false;

/// The RISC-V address monitor.
pub struct WrsMonitor;

impl AddressMonitor for WrsMonitor {
    /// **No operand.** `WRS.STO`'s bound is the architecture's short timeout,
    /// not a deadline the caller supplies — as on aarch64.
    type Deadline = ();

    /// `wrs` once detected, `none` until then.
    ///
    /// Gated on availability like x86's `name`, not unconditional like
    /// aarch64's: riscv's mechanism is one the machine *may* lack, so reporting
    /// `wrs` while it is parked would claim a fast path the runtime cannot take.
    /// A status row reading `none` is the honest description of a hart whose
    /// Zawrs the kernel has not vouched for.
    fn name() -> &'static str {
        if ZAWRS_AVAILABLE { "wrs" } else { "none" }
    }

    /// Whether this hart can wait on a reservation — a detected fact, parked
    /// `false`; see [`ZAWRS_AVAILABLE`] and the module docs.
    fn available() -> bool {
        ZAWRS_AVAILABLE
    }

    /// Nothing to compute: `WRS.STO` carries its own bound, so any cap is
    /// expressible without a clock — `Some(())`, as on aarch64. Never reached
    /// while [`available`](Self::available) is parked `false`, but the honest
    /// answer is the architectural timeout, not "cannot bound".
    fn deadline_after(_cap: Duration, _counter_hz: u64) -> Option<()> {
        Some(())
    }

    /// # Safety
    /// Per the trait: `addr` names a live, readable word. `LR.W` additionally
    /// requires natural (4-byte) alignment, which an `AtomicU32` has; a
    /// misaligned load-reserved raises an exception rather than wrapping.
    unsafe fn arm(addr: *const u32) {
        // SAFETY: a word-aligned load-reserved from a live address. The loaded
        // value is discarded — the caller re-reads through the atomic; what is
        // wanted is the side effect of registering this hart's reservation set
        // on the line. `LR.W` reads memory, so no `nomem`.
        unsafe {
            core::arch::asm!(
                "lr.w {tmp}, ({addr})",
                tmp = out(reg) _,
                addr = in(reg) addr,
                options(nostack),
            );
        }
    }

    /// # Safety
    /// Per the trait: must follow [`arm`](Self::arm) on this hart with no
    /// intervening store or `SC` that could clear the reservation.
    unsafe fn sleep(_deadline: ()) {
        // SAFETY: a wait hint with no memory operand — it stalls on the
        // reservation-set state, it does not access memory, hence `nomem`.
        // `wrs.sto` is Zawrs, absent from this crate's `rv64gc` feature set, so
        // the extension is enabled for this one instruction with a scoped
        // `.option arch` push/pop rather than a hand-encoded word — the same
        // mnemonic-not-bytes discipline the x86 backend keeps. Runs at U-mode
        // unconditionally: `WRS.STO` is exempt from `mstatus.TW`.
        unsafe {
            core::arch::asm!(
                ".option push",
                ".option arch, +zawrs",
                "wrs.sto",
                ".option pop",
                options(nomem, nostack),
            );
        }
    }

    /// `WRS.STO`'s timeout is implementation-defined rather than a deadline
    /// this crate sets, so a *timed* pause cannot be expressed here. The engine
    /// spins instead, which is always correct.
    unsafe fn pause(_deadline: ()) {}

    /// **Nothing to do on RISC-V** — and, unlike x86's empty `disarm`, that is
    /// a property of the reservation model rather than a happy accident.
    ///
    /// A reservation is not an exclusive sequence a hart must close: any store
    /// to the set clears it, any trap or context switch clears it, and a later
    /// `SC` with no matching `LR` fails against it. An abandoned `LR.W` is
    /// therefore inert — there is no `CLREX` to issue, and nothing a stray
    /// reservation can corrupt, since correct code never relies on one
    /// persisting. Contrast aarch64, where this method earns its keep.
    ///
    /// # Safety
    /// Per the trait; a no-op is unconditionally safe.
    unsafe fn disarm() {}
}
