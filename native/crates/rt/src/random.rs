//! Hardware entropy for userland — the seed nothing else can manufacture.
//!
//! A generator is not a source. `rand`-style PRNGs are deterministic
//! functions of their seed, so a program that wants unpredictable bytes needs
//! *something* outside itself to start from; on a machine with no OS behind
//! it, that something is the CPU's own entropy instruction. This module is
//! that one step, and everything else — stretching a seed, a userland
//! CSPRNG, `std`'s `hashmap_random_keys` — sits on top of it.
//!
//! **No trap.** On x86-64 and aarch64 the entropy instruction is executable
//! at ring 3 / EL0, so a fill is a few instructions and never enters the
//! kernel — the same posture as the zero-trap clock. The kernel keeps its
//! own [`EntropySource`] for its own use; this is not a session verb, and
//! deliberately so: a per-fill trap would make `HashMap` construction a
//! syscall.
//!
//! [`EntropySource`]: https://example.invalid
//!
//! # Per-arch sources, and the one that has none
//!
//! - **x86-64** — `RDRAND`, gated on `CPUID.01H:ECX[30]` because executing it
//!   without the feature is `#UD`, a fault rather than a refusal. Intel
//!   specifies a bounded retry: the DRNG can report "not ready" under heavy
//!   multi-core demand, and a caller that retries forever turns a busy RNG
//!   into a hang.
//! - **aarch64** — `RNDR`, read through its raw system-register encoding so
//!   the assembler needs no `+rand` in the target spec. Failure is reported
//!   in `PSTATE.NZCV` (Z set, result zeroed), and is retried on the same
//!   bounded budget. `FEAT_RNG` is **optional** in the architecture and
//!   genuinely absent on shipping parts, and EL0 cannot read the ID
//!   register that says so — so the answer is read from the kernel-stamped
//!   machine block on this task's boot page. Still no trap: that page is
//!   already mapped and immutable.
//! - **riscv64** — **none, and this refuses.** The `seed` CSR is the ISA's
//!   only entropy source, and it is not readable from U-mode unless
//!   `mseccfg.{s,u}seed` permits it; kore does not grant that. Sourcing
//!   entropy there needs a kernel verb that does not exist yet, so callers
//!   get [`EntropyUnavailable`] rather than a number that looks random and
//!   is not. That refusal is the whole point: a fabricated seed is worse
//!   than an absent one, because it silently weakens everything derived
//!   from it.

/// The machine has no entropy source this thread may read.
///
/// Not "try again later" — the source is absent, and a caller that needs
/// unpredictability must refuse the work rather than continue with a
/// substitute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntropyUnavailable;

/// Retries before a fill gives up.
///
/// Intel's DRNG guidance for `RDRAND`; reused for `RNDR`, whose failure mode
/// is the same shape (a shared generator reporting "no value this cycle").
/// Bounded because the alternative is an unkillable spin inside what callers
/// treat as a cheap function.
///
/// Only the arches that actually draw have a budget to spend; riscv64 has no
/// source to retry.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const RETRIES: usize = 10;

/// Fill `buf` with hardware entropy.
///
/// Every byte comes from the CPU's generator; no state is kept between
/// calls, so this is a *source*, not a stream — a caller wanting many bytes
/// cheaply should seed a generator from it once.
pub fn fill(buf: &mut [u8]) -> Result<(), EntropyUnavailable> {
    let mut written = 0;
    while written < buf.len() {
        let word = imp::word()?.to_ne_bytes();
        let take = core::cmp::min(word.len(), buf.len() - written);
        buf[written..written + take].copy_from_slice(&word[..take]);
        written += take;
    }
    Ok(())
}

/// One 64-bit draw — the shape `hashmap_random_keys` and seeding want.
pub fn u64() -> Result<u64, EntropyUnavailable> {
    imp::word()
}

/// Whether this machine exposes an entropy source to userland at all.
///
/// Lets a caller decide *before* doing work whose result would have to be
/// thrown away, rather than discovering the refusal at the end.
pub fn available() -> bool {
    imp::available()
}

#[cfg(target_arch = "x86_64")]
mod imp {
    use super::{EntropyUnavailable, RETRIES};
    use core::arch::x86_64::{__cpuid, _rdrand64_step};

    /// `CPUID.01H:ECX[30]` — the RDRAND feature bit.
    ///
    /// Checked rather than assumed: `RDRAND` on a CPU without it is `#UD`,
    /// which faults the task instead of returning a refusal we could report.
    pub(super) fn available() -> bool {
        // Leaf 1 is architectural on every x86-64 part.
        __cpuid(1).ecx & (1 << 30) != 0
    }

    pub(super) fn word() -> Result<u64, EntropyUnavailable> {
        if !available() {
            return Err(EntropyUnavailable);
        }
        // SAFETY: the feature bit above is exactly the precondition for
        // executing RDRAND.
        unsafe { draw() }
    }

    #[target_feature(enable = "rdrand")]
    unsafe fn draw() -> Result<u64, EntropyUnavailable> {
        let mut value = 0u64;
        for _ in 0..RETRIES {
            // Carry set (`1`) means the draw is valid; the intrinsic is
            // safe to call now that the caller established the feature.
            if _rdrand64_step(&mut value) == 1 {
                return Ok(value);
            }
        }
        // Carry clear for the whole budget: the generator is saturated or
        // broken.  Either way this thread has no entropy right now, and
        // saying so beats returning the zeroed `value`.
        Err(EntropyUnavailable)
    }
}

#[cfg(target_arch = "aarch64")]
mod imp {
    use super::{EntropyUnavailable, RETRIES};

    /// Whether this part implements `FEAT_RNG`.
    ///
    /// Unlike x86 there is no EL0-readable feature bit — `ID_AA64ISAR0_EL1`
    /// is EL1-only — and executing `RNDR` to find out is an
    /// undefined-instruction trap, which kills the task instead of
    /// returning the refusal a caller could handle.  So the kernel, which
    /// can read the register, stamps the answer into this task's boot page
    /// (`MACHINE_FLAG_CPU_ENTROPY`), and this is a read of already-mapped
    /// memory — no round trip, no trap.
    ///
    /// Fails **closed**: a task with no boot page, or one founded by a
    /// kernel that predates the flag, reports unavailable.  Refusing on a
    /// machine that could have served is a caller's inconvenience;
    /// executing on a machine that cannot is its death.
    pub(super) fn available() -> bool {
        crate::entry::thread_context()
            .and_then(|rt| rt.machine())
            .is_some_and(|block| block.cpu_entropy())
    }

    pub(super) fn word() -> Result<u64, EntropyUnavailable> {
        if !available() {
            return Err(EntropyUnavailable);
        }
        for _ in 0..RETRIES {
            let value: u64;
            let ok: u64;
            // SAFETY: `S3_3_C2_C4_0` is RNDR, readable at EL0 on a
            // FEAT_RNG part.  The raw encoding avoids needing `+rand` in
            // the target spec.  RNDR reports failure in PSTATE.NZCV (Z
            // set) and zeroes its result, so the flag read must follow the
            // MRS with nothing between that could clobber NZCV.
            unsafe {
                core::arch::asm!(
                    "mrs {value}, S3_3_C2_C4_0",
                    "cset {ok}, ne",
                    value = out(reg) value,
                    ok = out(reg) ok,
                    options(nostack),
                );
            }
            if ok != 0 {
                return Ok(value);
            }
        }
        Err(EntropyUnavailable)
    }
}

#[cfg(target_arch = "riscv64")]
mod imp {
    use super::EntropyUnavailable;

    /// The `seed` CSR is not readable from U-mode under kore's
    /// `mseccfg`, and there is no kernel verb for entropy yet.
    pub(super) fn available() -> bool {
        false
    }

    pub(super) fn word() -> Result<u64, EntropyUnavailable> {
        Err(EntropyUnavailable)
    }
}

#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "riscv64"
)))]
mod imp {
    use super::EntropyUnavailable;

    pub(super) fn available() -> bool {
        false
    }

    pub(super) fn word() -> Result<u64, EntropyUnavailable> {
        Err(EntropyUnavailable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host suite runs on x86-64, so this exercises the real
    /// instruction rather than a model of it — rare enough for arch code to
    /// be worth stating.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn a_fill_succeeds_and_is_not_a_constant() {
        assert!(available(), "the host suite's x86-64 CPU has RDRAND");
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        fill(&mut a).expect("hardware entropy");
        fill(&mut b).expect("hardware entropy");
        assert_ne!(a, b, "two draws returned identical bytes");
        assert_ne!(a, [0u8; 32], "a draw returned the zeroed failure value");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn a_short_fill_writes_every_byte_and_no_more() {
        // The tail path: a length that is not a multiple of the draw width
        // must fill exactly `len` bytes.
        let mut buf = [0u8; 13];
        fill(&mut buf).expect("hardware entropy");
        let mut guard = [0u8; 16];
        guard[..13].copy_from_slice(&buf);
        assert_eq!(&guard[13..], &[0, 0, 0], "wrote past the requested length");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn an_empty_fill_is_ok_and_draws_nothing() {
        fill(&mut []).expect("an empty fill needs no entropy");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn a_word_is_not_repeatable() {
        let (x, y) = (u64().expect("entropy"), u64().expect("entropy"));
        assert_ne!(x, y, "two draws returned the same word");
    }
}
