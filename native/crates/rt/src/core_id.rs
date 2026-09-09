//! **Which core is running this thread, by the silicon's own name** — an
//! ambient, unprivileged identity read for placement diagnostics.
//!
//! The G6 pin decides placement in the kernel and witnesses it on serial
//! (`spmd-pin: cpu N class P`), which a production terminal cannot read — a
//! metal box with no serial hookup has no way to see whether its workers
//! actually landed pinned.  This is the userland half of that truth: the
//! executing core's hardware identity, read in a few instructions with no
//! kernel crossing (the zero-trap posture of [`clock`](crate::clock) and
//! [`random`](crate::random)), so a tool can print where its threads *ran*
//! next to the numbers that depend on it.
//!
//! # Equality only
//!
//! The value is the architecture's own name for the core (x86: the x2APIC
//! id), **not** a kore cpu id, not dense, not portable across machines.
//! Two reads are comparable for equality — same core or not — and that is
//! the entire contract: distinctness across workers and stability across
//! time.  Nothing here maps identity to class or index; that mapping is
//! the kernel's, behind the catalog.
//!
//! # Per-arch sources, and the two that refuse
//!
//! - **x86-64** — `CPUID`, executable at CPL3 by design.  Leaf `0BH`
//!   subleaf 0 returns the running logical processor's full 32-bit x2APIC
//!   id in EDX; where that leaf is absent, the 8-bit initial APIC id from
//!   leaf `01H`:EBX\[31:24\] is the pre-x2APIC identity.
//! - **aarch64** — **none, and this refuses.**  The identity register
//!   (`MPIDR_EL1`) is EL1-only; EL0 has no architectural self-identity
//!   read.  Reporting `None` beats inventing one — a fabricated identity
//!   would satisfy every diagnostic while witnessing nothing.
//! - **riscv64** — **none**, same shape: `mhartid` is M-mode.
//!
//! # Not for the timed window
//!
//! `CPUID` is a serializing instruction.  Sample *around* a measured
//! region — never inside it — or the probe becomes the perturbation the
//! measurement exists to see past.

/// The executing core's hardware identity, or `None` where this
/// architecture gives userland no self-identity read.
///
/// Comparable for equality only (see the module docs).  The thread may be
/// rescheduled the instant this returns; a caller wanting "where did this
/// stretch of work run" samples at both ends and compares.
#[inline]
pub fn read() -> Option<u32> {
    imp::read()
}

/// Whether this machine exposes a core identity to userland at all —
/// the courtesy [`random::available`](crate::random::available) extends,
/// so a report can say "unknown" once rather than discovering it per
/// worker.
#[inline]
pub fn available() -> bool {
    imp::read().is_some()
}

#[cfg(target_arch = "x86_64")]
mod imp {
    use core::arch::x86_64::{__cpuid, __cpuid_count};

    pub(super) fn read() -> Option<u32> {
        // Leaf 0BH is gated on the max basic leaf: an out-of-range CPUID
        // is not a fault, but its zeroed answer would be indistinguishable
        // from core 0 — the one failure mode a diagnostic must not have.
        if __cpuid(0).eax >= 0xB {
            let topo = __cpuid_count(0xB, 0);
            // SDM: a zeroed EBX at subleaf 0 means the leaf is not
            // actually implemented on this part.
            if topo.ebx != 0 {
                return Some(topo.edx);
            }
        }
        // The initial APIC id — 8 bits, architectural since leaf 1 has
        // existed, and still a truthful per-core name on parts this old.
        Some((__cpuid(1).ebx >> 24) & 0xFF)
    }
}

#[cfg(target_arch = "aarch64")]
mod imp {
    /// `MPIDR_EL1` is EL1-only and EL0 has no aliased read of it; until a
    /// kernel-stamped page carries placement, aarch64 userland honestly
    /// does not know where it runs.
    pub(super) fn read() -> Option<u32> {
        None
    }
}

#[cfg(target_arch = "riscv64")]
mod imp {
    /// `mhartid` is M-mode; U-mode has no self-identity CSR.
    pub(super) fn read() -> Option<u32> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host suite runs on x86-64 (Rosetta included, which emulates
    /// CPUID): the read must produce *an* identity there.  The value and
    /// its stability are deliberately unasserted — the host scheduler may
    /// migrate this thread between any two reads.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn x86_reports_an_identity() {
        assert!(read().is_some());
        assert!(available());
    }

    /// The refusing arches must refuse, not fabricate.
    #[test]
    #[cfg(not(target_arch = "x86_64"))]
    fn refusal_is_honest() {
        assert!(read().is_none());
        assert!(!available());
    }
}
