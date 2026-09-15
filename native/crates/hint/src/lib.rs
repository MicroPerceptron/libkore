#![no_std]

//! **Sleep the core on a cache line** — the hardware address monitor.
//!
//! Every other wait in this runtime needs *authority* to deliver: a doorbell
//! notice rides a held `ChanSend`, an interrupt park rides a route, a join
//! rides a task handle. This one needs none. The waiter arms its core's
//! address monitor on a word, and the wake is performed by the silicon when
//! **any** agent stores to that line — no capability, no notice, no kernel.
//!
//! That is the whole reason this exists beside [`futex`](crate::futex). A
//! futex wake must reach the sleeping thread's doorbell, which needs a send
//! end held in the waker's own arena; where no such end exists the wake cannot
//! be delivered at all. A monitored wait has no such requirement, because
//! there is nothing to deliver — the store *is* the event.
//!
//! # What this is not
//!
//! **Not a park.** `UMWAIT` and `WFET` clock-gate the core and save dynamic
//! power; neither lowers voltage and — the part that decides where this may be
//! used — **neither releases the core to the scheduler**. A monitored wait
//! occupies its core for the whole wait. So:
//!
//! - **microseconds** (a lock handoff, a barrier) — monitor. The wake is
//!   immediate on the store and costs no syscall at either end.
//! - **milliseconds and up** — [`futex`](crate::futex)'s doorbell park, which
//!   actually deschedules the thread and frees the core for other work.
//!
//! Re-arming a monitor in a loop is *not* cadence polling — a store wakes the
//! core immediately, where sleep-and-recheck only notices at the next tick.
//! The reason not to hold a monitor for a long wait is core occupancy, not
//! latency.
//!
//! # Every wait here is capped, and that is a safety property
//!
//! Where the bound comes from differs per target, and only x86 pays for it
//! with an operand:
//!
//! - **x86** — an explicit deadline. `UMWAIT` and `MWAITX` genuinely sleep
//!   until theirs, so without one they would not return.
//! - **aarch64** — baseline `WFE` has no deadline operand. Its permission to
//!   return spuriously is not a promise that it will do so, and an eventual
//!   scheduler interrupt does not enforce a caller's microsecond cap. The
//!   backend therefore declines the monitored tier until a hard bound (for
//!   example `WFET` plus a published counter contract) is available.
//! - **riscv** — `WRS.STO`'s implementation-defined short timeout.
//!
//! The instructions also admit a genuinely uncapped form (`WRS.NTO`, or
//! `UMWAIT` with a deadline centuries out), and on validated silicon it is the
//! better choice for a loop guarding a single word: the monitor covers every
//! wake source, so a timeout only wakes the core to learn nothing.
//!
//! This module does not offer it. None of these instructions can be exercised
//! by anything we can currently boot: QEMU's TCG decodes no user-mode monitor
//! instruction at all — not `UMWAIT`, not `TPAUSE`, not `UMONITOR`, not AMD's
//! `MWAITX` — and the x86 boot gate runs `-cpu EPYC-v4`, which advertises none
//! of them. Real silicon is the only validation, so the monitor path ships
//! unexercised by construction.
//! Under that constraint the cap is what separates two very different bugs: a
//! monitor that silently never fires costs a slightly slower spin when the
//! wait is capped, and wedges the machine forever when it is not. Same defect,
//! and only one of them is survivable.
//!
//! So the cap is structural rather than advisory — [`wait_on`] takes a
//! `Duration`, not an `Option<Duration>`, and there is no second entry point.
//! When the sequences have been proved on real silicon, an uncapped form can
//! be added deliberately; until then its absence is the point.
//!
//! # The caller still re-checks
//!
//! A return is never a promise that the word changed. The monitor may fire on
//! a store to a *neighbouring* word that shares the line, on an interrupt, or
//! on the cap. [`wait_on`] reports only whether the core actually slept, which
//! is telemetry — the word is the truth, exactly as the doorbell's cookie is a
//! hint rather than a filter.

#[cfg(test)]
extern crate std;

use core::sync::atomic::{AtomicU32, Ordering};
use core::time::Duration;

/// How a [`wait_on`] turn was spent.
///
/// Reported for telemetry and tests, never as a synchronization fact: both
/// answers mean "look at the word again". A caller that branches on this for
/// correctness has misread the module.
///
/// `#[non_exhaustive]`: `UMWAIT` reports in `CF` whether the cap or a store
/// ended the sleep, and once a real impl lands that distinction may be worth
/// surfacing. Callers carry a wildcard arm so adding it is not a break.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum WaitEnd {
    /// The core slept on the watched line and something ended the sleep — a
    /// store, an interrupt, or the cap. Which one is deliberately not
    /// reported: only `UMWAIT` can distinguish them cheaply, and a caller that
    /// could act on the difference would be trusting the wake instead of the
    /// word.
    Monitored,
    /// No monitor was used — this turn was an ordinary spin hint. Either the
    /// machine has no such instruction, the deadline could not be computed, or
    /// the word moved between arming and sleeping.
    Spun,
}

/// **The per-arch address monitor** — the one place this mechanism varies.
///
/// Static methods over a unit type, like [`PowerControl`] and the clock's
/// `UserCounter`: there is no per-instance state, and the machine's monitor is
/// not a thing a caller can hold two of.
///
/// [`PowerControl`]: https://example.invalid
pub trait AddressMonitor {
    /// This architecture's deadline encoding.
    ///
    /// An associated type because the targets disagree completely and the
    /// engine must not know: aarch64's baseline `WFE` and riscv's `WRS.STO`
    /// take **no operand at all**, so their encoding is `()` when the backend
    /// can prove an independent bound (and `deadline_after` returns `None`
    /// when it cannot). x86 disagrees *with itself* — Intel's `UMWAIT` takes
    /// an absolute TSC deadline in `EDX:EAX` while AMD's `MWAITX` takes a
    /// relative max-wait in `EBX`. Since those two are chosen at **runtime**
    /// by vendor rather than by `cfg`, an x86 implementation is expected to
    /// carry the relative form and let its Intel path add the current counter
    /// at sleep time, which is also the more honest cap: the arm and the
    /// re-check happen between computing the bound and sleeping on it, and a
    /// relative bound does not silently spend that interval.
    ///
    /// Pushing all of this into the contract keeps [`wait_on`] arch-neutral
    /// instead of carrying a `u64` that means four different things.
    type Deadline: Copy;

    /// **What this machine's monitor is**, for a human reading a status row.
    ///
    /// The mechanism's own short name — `waitpkg`, `monitorx`, `wfet`, `wrs`
    /// — or `none`. Part of the contract rather than an x86 detail because
    /// every target's answer is worth reporting and none of them is inferable
    /// from the arch alone: x86 has two possible answers on one architecture,
    /// and every target has `none` as a real one.
    ///
    /// This is the only visibility that costs nothing to obtain. Detection is
    /// an ambient, unprivileged read on every target, so any process can ask
    /// without a service, a wire field, or an ABI change — which matters when
    /// the alternative is growing a frozen struct to answer "did this machine
    /// take the fast path at all".
    fn name() -> &'static str;

    /// Whether this machine can actually monitor an address.
    ///
    /// A **runtime** question, not a compile-time one, and on x86 a
    /// per-*vendor* one: `WAITPKG` (`UMONITOR`/`UMWAIT`) is Intel's and absent
    /// on AMD, which has the differently-encoded `MONITORX`/`MWAITX` instead —
    /// both usable from ring 3, neither a substitute for the other.
    /// `Zawrs` is rare. aarch64 is the exception that needs no detection at
    /// all: `WFE`/`SEV`/`SEVL` are ARMv8.0 baseline, so its answer is a
    /// constant. Where detection *is* needed it must be asked once and cached
    /// — this is read on the contended path, and `CPUID` is a serializing
    /// instruction that becomes a VM exit under virtualization.
    ///
    /// **Must fail closed**, and that is not decoration here. No emulator
    /// implements any of these instructions — QEMU's TCG decodes none of
    /// `UMWAIT`, `TPAUSE`, `UMONITOR`, or `MWAITX` — so a wrong `true` does
    /// not degrade, it takes an undefined-opcode fault on every machine we can
    /// currently boot. A wrong `false` costs a spin.
    fn available() -> bool;

    /// Convert a relative cap into this architecture's deadline encoding, or
    /// `None` when it cannot be expressed.
    ///
    /// `None` is a real answer rather than a failure: x86 needs the TSC
    /// frequency to turn nanoseconds into cycles, and a machine whose clock
    /// page is uncalibrated cannot bound a wait at all. The engine then spins,
    /// which is always correct.
    fn deadline_after(cap: Duration, counter_hz: u64) -> Option<Self::Deadline>;

    /// Arm this core's address monitor on the line holding `addr`.
    ///
    /// Arming alone does not sleep. The caller **must** re-read the watched
    /// word after this and skip [`sleep`](Self::sleep) if it has already
    /// moved — otherwise a store landing between the caller's first read and
    /// this arm is missed, and the core sleeps on a condition that already
    /// happened. That is the same arm-before-check discipline the futex uses
    /// for its doorbell post, and it is the caller's half of it.
    ///
    /// # Safety
    /// `addr` must point at a live, readable, naturally aligned word for the
    /// whole wait. The monitor watches the enclosing cache line, so a store to
    /// an unrelated neighbour in that line is an ordinary spurious wake.
    unsafe fn arm(addr: *const u32);

    /// Sleep until a store to the armed line, an interrupt, or `deadline`.
    ///
    /// **Consumes the arm.** An implementation must leave this core with no
    /// monitor outstanding when it returns, so the only path that needs
    /// [`disarm`](Self::disarm) is the one that never sleeps.
    ///
    /// # Safety
    /// Must follow an [`arm`](Self::arm) on this core with no intervening
    /// operation that could clear the monitor. Calling it unarmed is not a
    /// memory-safety fault on any of the three targets, but it is a
    /// contract violation that degenerates to an unbounded stall on an
    /// architecture whose wait has no independent bound.
    unsafe fn sleep(deadline: Self::Deadline);

    /// **Pause until `deadline` with nothing watched** — the backend half of
    /// [`pause_until`].
    ///
    /// Distinct from [`arm`](Self::arm) + [`sleep`](Self::sleep) because the
    /// targets genuinely differ: Intel's `TPAUSE` needs no monitor at all,
    /// while AMD's `MWAITX` requires a prior `MONITORX` and so must arm a
    /// private line. A backend that can only express the armed form arms one
    /// nothing ever stores to.
    ///
    /// # Safety
    /// Callers must have checked [`available`](Self::available). The
    /// implementation is responsible for any arming its own instruction
    /// requires.
    unsafe fn pause(deadline: Self::Deadline);

    /// **Abandon an arm without sleeping.**
    ///
    /// The engine arms *before* its final re-read of the word, so that a store
    /// landing in that window is not missed. When the re-read shows the word
    /// already moved, the wait is over before it began — and the monitor armed
    /// a moment ago is still outstanding. This retires it.
    ///
    /// A required method rather than one defaulted to nothing, because two of
    /// the three targets genuinely need nothing and the third genuinely does:
    /// x86's monitor is consumed by the next `UMONITOR` or any store to the
    /// line, and riscv's reservation likewise, but AArch64's `LDAXR` opens an
    /// exclusive sequence that architecturally wants `CLREX` to abandon — and
    /// leaving it open lets an unrelated later `STXR` on this core observe a
    /// reservation it did not make. Defaulting this to a no-op would make that
    /// the silent case rather than a decision each implementation states.
    ///
    /// # Safety
    /// Must only be called on a core this implementation armed and has not
    /// slept on.
    unsafe fn disarm();
}

/// **The engine**: arm, re-check, sleep — capped, with a spin as the floor.
///
/// Generic over the monitor so the ordering above can be tested on a host that
/// has none of these instructions, which is the only place it *can* be tested
/// (see the module docs on why the boot gates cannot execute this path).
///
/// Returns as soon as the word is observed to differ from `expected`, without
/// sleeping — a caller that has already got what it wanted must not be put to
/// sleep waiting for it.
pub fn wait_on_with<M: AddressMonitor>(
    word: &AtomicU32,
    expected: u32,
    cap: Duration,
    counter_hz: u64,
) -> WaitEnd {
    if !M::available() {
        core::hint::spin_loop();
        return WaitEnd::Spun;
    }
    let Some(deadline) = M::deadline_after(cap, counter_hz) else {
        // No usable clock: a wait we cannot bound is one we decline to take.
        core::hint::spin_loop();
        return WaitEnd::Spun;
    };

    // SAFETY: `word` is a live aligned `AtomicU32` for the borrow's life.
    unsafe { M::arm(word.as_ptr().cast_const()) };

    // **Re-check between arming and sleeping.** The store this waiter is
    // waiting for may have landed before the monitor was armed; sleeping now
    // would wait for an event that has already passed. `SeqCst` to match the
    // futex registry's discipline — a waiter's check and a waker's store must
    // not be reorderable past one another.
    if word.load(Ordering::SeqCst) != expected {
        // Armed a moment ago and now not sleeping: retire the monitor rather
        // than leaving an exclusive sequence open across the return.
        // SAFETY: armed immediately above on this core, and not slept on.
        unsafe { M::disarm() };
        return WaitEnd::Spun;
    }

    // SAFETY: armed immediately above on this core, nothing between.
    unsafe { M::sleep(deadline) };
    WaitEnd::Monitored
}

/// [`wait_on_with`] over this target's monitor.
///
/// The one call sites use. Spins when the machine has no monitor, so it is
/// always a correct replacement for a bare `core::hint::spin_loop()` and never
/// a behaviour change a caller has to reason about.
#[inline]
pub fn wait_on(word: &AtomicU32, expected: u32, cap: Duration, counter_hz: u64) -> WaitEnd {
    wait_on_with::<Monitor>(word, expected, cap, counter_hz)
}

/// **A timed pause with no watched word** — the deadline half of the seam,
/// without the monitor half.
///
/// `wait_on` exists to wait for a *store*; the cap is a safety bound on it.
/// This is the opposite shape: nothing will store, and the deadline is the
/// whole point. AML `Sleep` is the motivating caller — a firmware-requested
/// delay with no waker anywhere.
///
/// **Why this is not `wait_on` on a dead word.** That works, but it arms the
/// core's monitor to do it, and there is exactly **one monitor per logical
/// processor**: an arm here retires whatever the kernel's idle path armed, and
/// vice versa. Intel can avoid the question entirely — `TPAUSE` takes only a
/// deadline and arms nothing. AMD cannot: `MWAITX` requires a prior
/// `MONITORX`, so that path still arms, on a per-caller dummy line whose
/// address is never stored to. The asymmetry is real hardware, so it is
/// modelled rather than hidden.
///
/// Returns [`WaitEnd::Spun`] and executes one `spin_loop` where the machine
/// has no monitor or the caller supplied no usable rate, exactly as
/// [`wait_on`] does, so this is always a correct replacement for a bare
/// `core::hint::spin_loop()` inside a deadline loop.
#[inline]
pub fn pause_until(cap: Duration, counter_hz: u64) -> WaitEnd {
    pause_until_with::<Monitor>(cap, counter_hz)
}

/// [`pause_until`] over an explicit monitor, for host tests.
pub fn pause_until_with<M: AddressMonitor>(cap: Duration, counter_hz: u64) -> WaitEnd {
    if !M::available() {
        core::hint::spin_loop();
        return WaitEnd::Spun;
    }
    let Some(deadline) = M::deadline_after(cap, counter_hz) else {
        core::hint::spin_loop();
        return WaitEnd::Spun;
    };
    // SAFETY: the backend is available (checked above) and `pause` is
    // documented to arm whatever its target requires — nothing on Intel,
    // a private never-stored line on AMD.
    unsafe { M::pause(deadline) };
    WaitEnd::Monitored
}

/// **What this machine's address monitor is** — `waitpkg`, `monitorx`,
/// `wfet`, `wrs`, or `none`.
///
/// Ambient and unprivileged on every target, so a diagnostic can print it
/// without asking a service anything.
#[inline]
pub fn name() -> &'static str {
    Monitor::name()
}

/// Whether this machine can monitor an address at all.
///
/// Lets a caller pick a strategy up front rather than discovering it inside a
/// loop, the courtesy [`random::available`](crate::random::available) extends.
#[inline]
pub fn available() -> bool {
    Monitor::available()
}

/// **No monitor** — the honest floor.
///
/// Selected wherever a real one is not yet wired: off the metal targets, and
/// on any architecture whose sequences have not landed. [`wait_on`] then spins,
/// which is exactly what the call sites did before this module existed, so
/// adopting it early costs nothing and regresses nothing.
///
/// Deliberately not a "spin impl" of the monitor contract: it reports
/// `available() == false` rather than pretending to arm and sleep. A contract
/// that can be satisfied by doing nothing is one that cannot be tested.
pub struct NoMonitor;

impl AddressMonitor for NoMonitor {
    type Deadline = ();

    #[inline]
    fn name() -> &'static str {
        "none"
    }

    #[inline]
    fn available() -> bool {
        false
    }

    #[inline]
    fn deadline_after(_cap: Duration, _counter_hz: u64) -> Option<()> {
        None
    }

    #[inline]
    unsafe fn arm(_addr: *const u32) {}

    #[inline]
    unsafe fn sleep(_deadline: ()) {}

    /// Never reached: `available` is `false`, so both engines spin first.
    #[inline]
    unsafe fn pause(_deadline: ()) {}

    #[inline]
    unsafe fn disarm() {}
}

// ── Selection ───────────────────────────────────────────────────────────
//
// One cfg'd alias per target, and the only `#[cfg]` in this subsystem: every
// call site names `Monitor` and no consumer carries an arch branch. The
// per-arch sequences landed behind these names one architecture at a time;
// all three metal targets now have their own, and every non-metal build keeps
// `NoMonitor` and spins — so nothing ever waited on all three arriving together.

#[cfg(all(target_os = "none", target_arch = "x86_64"))]
mod x86_64;

#[cfg(all(target_os = "none", target_arch = "aarch64"))]
mod aarch64;

#[cfg(all(target_os = "none", target_arch = "riscv64"))]
mod riscv64;

/// This target's address monitor.
///
/// Gated on `target_os = "none"` as well as the arch, matching how `clock`
/// selects its counters: a host build keeps [`NoMonitor`] even on x86 silicon
/// that has the instructions, because the host suite must not pass for a
/// reason the metal target never reproduces — and because the calibration
/// these deadlines are computed from is a founder-published page that exists
/// only there.
#[cfg(all(target_os = "none", target_arch = "x86_64"))]
pub use x86_64::X86Monitor as Monitor;

/// This target's address monitor — the exclusive monitor plus `WFE`.
#[cfg(all(target_os = "none", target_arch = "aarch64"))]
pub use aarch64::A64Monitor as Monitor;

/// This target's address monitor — the reservation set plus `WRS.STO`.
#[cfg(all(target_os = "none", target_arch = "riscv64"))]
pub use riscv64::WrsMonitor as Monitor;

/// This target's address monitor — none.
///
/// The host floor. All three metal arches now have a backend of their own
/// above; this row is what every off-metal build keeps, because the sequences
/// are gated on `target_os = "none"` so the host suite exercises the engine
/// over a mock rather than a monitor it could never reproduce. Here every call
/// site spins, exactly as it always did.
#[cfg(not(all(
    target_os = "none",
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64"
    )
)))]
pub use NoMonitor as Monitor;

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::AtomicUsize;

    /// A monitor that records what the engine asked of it, so the ordering —
    /// the part that is genuinely easy to get wrong and impossible to observe
    /// on real silicon here — is checked where it can be.
    struct Mock;

    static ARMED: AtomicUsize = AtomicUsize::new(0);
    static SLEPT: AtomicUsize = AtomicUsize::new(0);
    static PAUSED: AtomicUsize = AtomicUsize::new(0);
    static DISARMED: AtomicUsize = AtomicUsize::new(0);
    static AVAILABLE: AtomicUsize = AtomicUsize::new(1);
    static DEADLINE_OK: AtomicUsize = AtomicUsize::new(1);

    /// The mock's counters are one shared set, and the test harness is
    /// parallel — without serialization, one test's [`reset`] lands
    /// between another's act and its assert (seen in the wild as
    /// `ARMED == 0` on a path that provably armed).  Every test below
    /// takes this guard first; a poisoned lock is fine to reuse, since
    /// [`reset`] restores the world regardless of how the loser died.
    fn serialize() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let guard = LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        guard
    }

    fn reset() {
        ARMED.store(0, Ordering::SeqCst);
        SLEPT.store(0, Ordering::SeqCst);
        DISARMED.store(0, Ordering::SeqCst);
        AVAILABLE.store(1, Ordering::SeqCst);
        DEADLINE_OK.store(1, Ordering::SeqCst);
    }

    impl AddressMonitor for Mock {
        type Deadline = u64;
        fn name() -> &'static str {
            "mock"
        }
        fn available() -> bool {
            AVAILABLE.load(Ordering::SeqCst) == 1
        }
        fn deadline_after(_cap: Duration, _counter_hz: u64) -> Option<u64> {
            (DEADLINE_OK.load(Ordering::SeqCst) == 1).then_some(42)
        }
        unsafe fn arm(_addr: *const u32) {
            ARMED.fetch_add(1, Ordering::SeqCst);
        }
        unsafe fn sleep(_deadline: u64) {
            SLEPT.fetch_add(1, Ordering::SeqCst);
        }
        unsafe fn disarm() {
            DISARMED.fetch_add(1, Ordering::SeqCst);
        }

        unsafe fn pause(_deadline: u64) {
            PAUSED.fetch_add(1, Ordering::SeqCst);
        }
    }

    const CAP: Duration = Duration::from_micros(1);
    /// A plausible counter rate for the mock backend; its `deadline_after`
    /// ignores the value, so only the non-zero-ness is meaningful here.
    const HZ: u64 = 1_000_000_000;

    #[test]
    fn an_unchanged_word_arms_then_sleeps() {
        let _guard = serialize();
        let word = AtomicU32::new(7);
        assert_eq!(wait_on_with::<Mock>(&word, 7, CAP, HZ), WaitEnd::Monitored);
        assert_eq!(ARMED.load(Ordering::SeqCst), 1);
        assert_eq!(SLEPT.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_word_that_already_moved_arms_but_never_sleeps() {
        let _guard = serialize();
        // The race the arm-then-recheck order exists for: the store landed
        // before the monitor was armed, so sleeping would wait for an event
        // that has already happened.
        let word = AtomicU32::new(9);
        assert_eq!(wait_on_with::<Mock>(&word, 7, CAP, HZ), WaitEnd::Spun);
        assert_eq!(ARMED.load(Ordering::SeqCst), 1, "must still arm first");
        assert_eq!(
            SLEPT.load(Ordering::SeqCst),
            0,
            "slept on a condition that had already passed"
        );
    }

    #[test]
    fn no_monitor_spins_without_arming() {
        let _guard = serialize();
        AVAILABLE.store(0, Ordering::SeqCst);
        let word = AtomicU32::new(7);
        assert_eq!(wait_on_with::<Mock>(&word, 7, CAP, HZ), WaitEnd::Spun);
        assert_eq!(ARMED.load(Ordering::SeqCst), 0);
        assert_eq!(SLEPT.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn an_available_but_unbounded_monitor_is_declined_rather_than_taken() {
        let _guard = serialize();
        // This is baseline AArch64's exact posture: WFE exists, but cannot
        // express the caller's cap. The engine must degrade to a spin rather
        // than becoming the uncapped sleep this module refuses to offer.
        DEADLINE_OK.store(0, Ordering::SeqCst);
        let word = AtomicU32::new(7);
        assert_eq!(wait_on_with::<Mock>(&word, 7, CAP, HZ), WaitEnd::Spun);
        assert_eq!(ARMED.load(Ordering::SeqCst), 0, "armed without a bound");
        assert_eq!(SLEPT.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn abandoning_an_arm_retires_the_monitor() {
        let _guard = serialize();
        // The engine arms BEFORE its final re-read, so a store landing in that
        // window is not missed. When the re-read shows the word already moved,
        // the monitor armed a moment ago is still outstanding — and on AArch64
        // an open exclusive sequence left across the return can be observed by
        // an unrelated later `STXR` on this core.
        let word = AtomicU32::new(9);
        assert_eq!(wait_on_with::<Mock>(&word, 7, CAP, HZ), WaitEnd::Spun);
        assert_eq!(ARMED.load(Ordering::SeqCst), 1);
        assert_eq!(SLEPT.load(Ordering::SeqCst), 0);
        assert_eq!(
            DISARMED.load(Ordering::SeqCst),
            1,
            "abandoned an armed monitor without retiring it"
        );
    }

    #[test]
    fn sleeping_consumes_the_arm_without_a_separate_disarm() {
        let _guard = serialize();
        // The contract puts the obligation in one place per impl: `sleep`
        // leaves no monitor outstanding, so the engine must not also disarm —
        // a second retire on a consumed monitor is at best wasted work and at
        // worst clears a reservation someone else just made.
        let word = AtomicU32::new(7);
        assert_eq!(wait_on_with::<Mock>(&word, 7, CAP, HZ), WaitEnd::Monitored);
        assert_eq!(SLEPT.load(Ordering::SeqCst), 1);
        assert_eq!(
            DISARMED.load(Ordering::SeqCst),
            0,
            "disarmed a monitor that sleeping had already consumed"
        );
    }

    #[test]
    fn a_refused_wait_never_arms_so_never_needs_retiring() {
        let _guard = serialize();
        AVAILABLE.store(0, Ordering::SeqCst);
        let word = AtomicU32::new(7);
        assert_eq!(wait_on_with::<Mock>(&word, 7, CAP, HZ), WaitEnd::Spun);
        assert_eq!(
            DISARMED.load(Ordering::SeqCst),
            0,
            "retired an unarmed core"
        );
    }

    #[test]
    fn every_monitor_names_itself() {
        // A status row that said "unknown" would be the one answer that helps
        // nobody: the whole point is telling a machine with the fast path from
        // one without.
        assert_eq!(NoMonitor::name(), "none");
        assert!(!name().is_empty());
    }

    #[test]
    fn the_fallback_monitor_reports_itself_absent() {
        // `NoMonitor` must not pretend: an impl that satisfies the contract by
        // doing nothing would make every test above pass vacuously.
        assert!(!NoMonitor::available());
        assert!(NoMonitor::deadline_after(CAP, HZ).is_none());
    }

    #[test]
    fn the_selected_monitor_is_always_a_correct_spin_replacement() {
        // Whatever `Monitor` resolves to, `wait_on` must be safe to drop in
        // wherever a bare `spin_loop()` stood.
        let word = AtomicU32::new(3);
        let end = wait_on(&word, 3, CAP, HZ);
        assert!(matches!(end, WaitEnd::Monitored | WaitEnd::Spun));
        // And so must the word-less form, which is the AML-sleep shape.
        assert!(matches!(
            pause_until(CAP, HZ),
            WaitEnd::Monitored | WaitEnd::Spun
        ));
        // A caller with no calibration declines the monitored tier rather
        // than taking an unbounded wait.
        assert_eq!(pause_until(CAP, 0), WaitEnd::Spun);
    }
}
