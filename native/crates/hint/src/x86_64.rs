//! x86-64 address monitors — **two vendors, two instruction sets**.
//!
//! Intel and AMD share nothing here but the idea. Intel has `UMONITOR` +
//! `UMWAIT` behind `WAITPKG`; AMD has `MONITORX` + `MWAITX` behind its own
//! feature bit — differently encoded, timeout in a different register, and
//! expressed *relatively* rather than as an absolute deadline. Neither decodes
//! on the other's silicon.
//!
//! Both are usable from **ring 3**, which is the only reason either is a
//! userland mechanism at all: the older `MONITOR`/`MWAIT` pair is CPL-0.
//! (ForkUnion's header claims `MWAITX` needs ring 0. It does not — AMD
//! introduced the pair with Excavator specifically for unprivileged use, and
//! inheriting that claim would have silently excluded every AMD machine.)
//!
//! So the vendor choice is a **runtime** one *inside* one `target_arch`, and
//! cannot be a `cfg` alias like the arch choice above it. It is a probed,
//! cached branch — see [`Kind`].
//!
//! # Mnemonics, not hand-encoded bytes
//!
//! All four instructions assemble by name with this tree's pinned toolchain,
//! with no `target_feature` gate on the inline-asm path. Prior art hand-encodes
//! them (`.byte 0xf3, 0x0f, 0xae, 0xf0` and friends) because it must also
//! satisfy MSVC, which has no inline assembly, and because naming the feature
//! in a `target()` attribute breaks Apple's clang. Neither constraint applies
//! here, and a hand-encoded byte string is a thing that can be silently wrong
//! in a way `umonitor rax` cannot.

use core::time::Duration;

use raw_cpuid::CpuId;

use core::sync::atomic::{AtomicU8, Ordering};

use super::AddressMonitor;

/// Which instruction pair this machine has, decided once by CPUID.
///
/// Not an `Option<bool>`: "has a monitor" is not one question on x86, and
/// collapsing the two vendors into a boolean is how the arm and the sleep
/// would end up disagreeing about which encoding to emit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// `UMONITOR` + `UMWAIT` — `CPUID.07H.0:ECX[5]`.
    Waitpkg,
    /// `MONITORX` + `MWAITX` — `CPUID.8000_0001H:ECX[29]`.
    MonitorX,
    /// Neither — and the case every emulator reports, since QEMU's TCG decodes
    /// no user-mode monitor instruction at all.
    None,
}

/// The cached answer: `0` unasked, otherwise `Kind as u8 + 1`.
///
/// The **whole** answer, not just whether there is one. Caching a bool and
/// re-deriving the vendor looked equivalent and is not: `kind` is called by
/// `available`, `name`, `arm`, and `sleep`, so a single `wait_on` turn would
/// issue `CPUID` up to three times — a serializing instruction, and a VM exit
/// under virtualization — on the exact path this seam exists to make cheap.
///
/// Racy and idempotent: two threads may both detect on first use and write the
/// same answer. It is a machine constant, so `Relaxed` is enough and no lock is
/// wanted on a path whose whole purpose is avoiding contention.
static CACHED: AtomicU8 = AtomicU8::new(0);

/// This machine's monitor, or [`Kind::None`], asked once.
///
/// **Fail-closed by construction**: every path that is not a positive feature
/// bit answers `None`. That matters more than it reads — no emulator
/// implements these instructions, so a wrong positive does not degrade to a
/// slow path, it takes `#UD` on every machine CI can boot.
fn kind() -> Kind {
    match CACHED.load(Ordering::Relaxed) {
        0 => {
            let found = detect();
            CACHED.store(found as u8 + 1, Ordering::Relaxed);
            found
        }
        cached => match cached - 1 {
            x if x == Kind::Waitpkg as u8 => Kind::Waitpkg,
            x if x == Kind::MonitorX as u8 => Kind::MonitorX,
            _ => Kind::None,
        },
    }
}

/// The raw CPUID question, asked through the crate the kernel already reads
/// its feature bits with rather than hand-decoded here — two different leaves,
/// and getting either wrong is a fault rather than a wrong answer.
///
/// `WAITPKG` is tested first: a machine reporting both is Intel silicon whose
/// `MONITORX` bit is a hypervisor's invention rather than a real AMD part.
fn detect() -> Kind {
    let cpuid = CpuId::new();
    if cpuid
        .get_extended_feature_info()
        .is_some_and(|f| f.has_waitpkg())
    {
        return Kind::Waitpkg;
    }
    if cpuid
        .get_extended_processor_and_feature_identifiers()
        .is_some_and(|f| f.has_monitorx_mwaitx())
    {
        return Kind::MonitorX;
    }
    Kind::None
}

/// `UMWAIT`'s control word: **C0.1**, the shallow state.
///
/// Bit 0 set selects C0.1 — shallower and faster-waking than C0.2. The waits
/// this seam serves are lock handoffs and barriers measured in hundreds of
/// nanoseconds, where the deeper state's wake latency would be a large
/// fraction of the wait it is supposed to shorten. Neither state lowers
/// voltage; both only clock-gate.
const UMWAIT_C01: u32 = 1;

/// `MWAITX`'s `ECX`: enable the timeout (bit 1).
///
/// Without it `MWAITX` waits with no bound, which is exactly the uncapped form
/// this module refuses to offer on hardware it cannot validate.
const MWAITX_TIMER_ENABLE: u32 = 1 << 1;

/// `MWAITX`'s `EAX` hints — none.
const MWAITX_NO_HINTS: u32 = 0;

/// The x86 address monitor, over whichever pair this machine has.
pub struct X86Monitor;

impl AddressMonitor for X86Monitor {
    /// **Relative** counter cycles, not an absolute deadline.
    ///
    /// AMD's `MWAITX` takes a relative max-wait natively, so carrying the
    /// relative form lets the Intel path add the current counter at sleep time
    /// rather than forcing AMD to subtract it back off. It is also the more
    /// honest cap: the arm and the re-check happen between computing the bound
    /// and sleeping on it, and a relative bound does not silently spend that
    /// interval.
    type Deadline = u64;

    fn name() -> &'static str {
        match kind() {
            Kind::Waitpkg => "waitpkg",
            Kind::MonitorX => "monitorx",
            Kind::None => "none",
        }
    }

    fn available() -> bool {
        kind() != Kind::None
    }

    /// Nanoseconds to counter cycles, through the founder-published
    /// calibration.
    ///
    /// `counter_hz` *is* the TSC frequency on this target, supplied by the
    /// caller's own clock — userland's calibrated page, the kernel's TSC
    /// calibration. Passing it rather than reaching for one keeps this crate
    /// usable from both sides of the privilege wall. `0` means the caller has
    /// no calibration: a wait we cannot bound is one we decline to take.
    ///
    /// The arithmetic stays in `u128` **end to end**, including the input.
    /// `nanos · tick_hz` overflows `u64` for any cap past a few seconds on a
    /// GHz-class part, and a wrapped bound is a deadline in the past — a
    /// monitor that returns instantly, forever, which reads as a working spin
    /// rather than a bug. `Duration::as_nanos` is already `u128`, so narrowing
    /// it first would reintroduce that wrap at the one point the widening was
    /// meant to remove it.
    fn deadline_after(cap: Duration, counter_hz: u64) -> Option<u64> {
        if counter_hz == 0 {
            return None;
        }
        let hz = counter_hz;
        let cycles = cap
            .as_nanos()
            .saturating_mul(u128::from(hz))
            / 1_000_000_000u128;
        Some(u64::try_from(cycles).unwrap_or(u64::MAX))
    }

    /// # Safety
    /// Per the trait: `addr` names a live, readable, aligned word.
    unsafe fn arm(addr: *const u32) {
        match kind() {
            // SAFETY: `WAITPKG` is present (checked above), and `UMONITOR`
            // only reads the address to arm this core's monitor on its line.
            Kind::Waitpkg => unsafe {
                core::arch::asm!("umonitor {addr}", addr = in(reg) addr, options(nostack));
            },
            // SAFETY: `MONITORX` is present. Its operands are implicit —
            // `rAX` the address, `ECX` extensions, `EDX` hints — and zero is
            // the defined "no extension / no hint" encoding for both.
            Kind::MonitorX => unsafe {
                core::arch::asm!(
                    "monitorx",
                    in("rax") addr,
                    in("ecx") 0u32,
                    in("edx") 0u32,
                    options(nostack),
                );
            },
            // Never armed, so never slept on: `wait_on` checks `available`
            // first, and this arm is unreachable through it.
            Kind::None => {}
        }
    }

    /// # Safety
    /// Per the trait: must follow [`arm`](Self::arm) on this core.
    unsafe fn sleep(deadline: u64) {
        match kind() {
            Kind::Waitpkg => {
                // `UMWAIT` wants an ABSOLUTE deadline in `EDX:EAX`, so the
                // relative bound becomes one here — as late as possible, so
                // the arm and the caller's re-check are not charged against it.
                // Saturating rather than wrapping: a wrapped sum is a deadline
                // in the past, i.e. a wait that silently never waits.
                let at = read_tsc().saturating_add(deadline);
                // SAFETY: `WAITPKG` is present and this core was just armed.
                unsafe {
                    core::arch::asm!(
                        "umwait {ctrl:e}",
                        ctrl = in(reg) UMWAIT_C01,
                        in("eax") at as u32,
                        in("edx") (at >> 32) as u32,
                        options(nostack),
                    );
                }
            }
            Kind::MonitorX => {
                // `MWAITX` takes the RELATIVE bound directly, in `EBX`, which
                // is 32 bits — so a cap beyond `u32::MAX` cycles is clamped
                // rather than truncated. Clamping shortens a wait; truncating
                // would randomise it.
                let ticks = u32::try_from(deadline).unwrap_or(u32::MAX);
                // `EBX` cannot be named as an operand — LLVM reserves `rbx`
                // for its own use — so the timeout is swapped in around the
                // instruction and swapped straight back out. The same dance
                // every `CPUID` wrapper on this target performs, and the
                // reason `MWAITX` is the one instruction here that is not a
                // single line.
                //
                // SAFETY: `MONITORX` is present and this core was just armed.
                // Both `xchg`es are register-only; `rbx` is restored before
                // the block ends, so LLVM's invariant holds across it.
                unsafe {
                    core::arch::asm!(
                        "xchg {tmp}, rbx",
                        "mwaitx",
                        "xchg {tmp}, rbx",
                        tmp = inout(reg) u64::from(ticks) => _,
                        in("eax") MWAITX_NO_HINTS,
                        in("ecx") MWAITX_TIMER_ENABLE,
                        options(nostack),
                    );
                }
            }
            Kind::None => {}
        }
    }

    /// # Safety
    /// Per the trait: `available` was checked.
    unsafe fn pause(deadline: u64) {
        match kind() {
            // `TPAUSE` takes ONLY a deadline — no `UMONITOR`, so this leaves
            // the core's single monitor untouched and cannot retire an arm the
            // kernel's idle path is relying on. This is the whole reason the
            // trait carries `pause` separately from `arm` + `sleep`.
            Kind::Waitpkg => {
                let at = read_tsc().saturating_add(deadline);
                // SAFETY: `WAITPKG` is present (checked by the caller).
                unsafe {
                    core::arch::asm!(
                        "tpause {ctrl:e}",
                        ctrl = in(reg) UMWAIT_C01,
                        in("eax") at as u32,
                        in("edx") (at >> 32) as u32,
                        options(nostack),
                    );
                }
            }
            // AMD has no `TPAUSE`. `MWAITX` requires a prior `MONITORX`, so
            // arm a line this crate owns and nothing ever stores to: the wake
            // can then only be the timer. Static rather than per-CPU because a
            // spurious cross-core wake here costs one extra loop iteration in
            // the caller's deadline check, and never a missed deadline.
            Kind::MonitorX => {
                static PAUSE_LINE: core::sync::atomic::AtomicU32 =
                    core::sync::atomic::AtomicU32::new(0);
                // SAFETY: `MONITORX` is present; the line is a live static.
                unsafe {
                    Self::arm(PAUSE_LINE.as_ptr().cast_const());
                    Self::sleep(deadline);
                }
            }
            Kind::None => {}
        }
    }

    /// **Nothing to do on x86.**
    ///
    /// Neither vendor has an instruction that abandons an armed monitor, and
    /// neither needs one: the arm is a hardware watch on a line, not an
    /// exclusive reservation the core must later account for. It is retired by
    /// the next `UMONITOR`/`MONITORX`, by any store to the line, or simply by
    /// never being slept on. AArch64 is the target this method exists for.
    ///
    /// # Safety
    /// Per the trait; trivially satisfied.
    unsafe fn disarm() {}
}

/// The time-stamp counter — the same counter the clock page calibrates.
#[inline]
fn read_tsc() -> u64 {
    // SAFETY: `RDTSC` is unprivileged unless `CR4.TSD` is set, and the kernel
    // clears it before user dispatch — the precondition `clock::x86_64` already
    // relies on for every ordinary time read.
    unsafe { core::arch::x86_64::_rdtsc() }
}
