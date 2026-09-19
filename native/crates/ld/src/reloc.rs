//! The relocation seam — one contract trait, three per-ISA data impls
//! (`res/IMAGE.md` §7; the doctrine-§3 shape klink's `#[cfg]` modules
//! are retired for).
//!
//! Under the pinned flag regime (`-z now -z relro -z text
//! -Bsymbolic-functions -fno-plt`, all-PIC, no ifuncs, LE/IE TLS) an
//! image may carry exactly **five relocation semantics** ([`RelocKind`]).
//! Everything else is either on the **reject list** (a hard, *named*
//! refusal — these are the relocations the toolchain regime promises
//! never to emit, so their presence marks a foreign or hostile image) or
//! **unknown** (refused likewise).  The engine ([`crate::apply`]) is
//! generic over [`RelocArch`]; a new ISA is a new impl, never a branch.
//!
//! Every `R_*` number below is copied from its ABI supplement — the
//! x86-64 psABI, ARM AAELF64, and the RISC-V psABI — not inferred.

/// The five eager semantics the engine computes (`res/IMAGE.md` §7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelocKind {
    /// `B + A` — base plus addend (the static-PIE self-reloc class, and
    /// the bulk of every PIE's `.rela.dyn`).
    Relative,
    /// `S` — a GOT data slot (symbol value; addend ignored per spec).
    GlobDat,
    /// `S` — a PLT slot.  Identical to [`GlobDat`](Self::GlobDat) under
    /// `-z now` (eager: no resolver, no lazy stub), kept distinct so a
    /// mis-tagged image still classifies faithfully.
    JumpSlot,
    /// `S + A` — an absolute 64-bit data reference.
    Abs64,
    /// `tpoff(S) + A` — an initial-exec TLS slot: the thread-pointer-
    /// relative offset of a symbol in the static TLS block
    /// ([`TlsRules`] gives the per-arch layout).
    TpRel,
}

/// Why a relocation type refused ([`RelocArch::classify`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelocRefusal {
    /// On the reject list: a relocation the flag regime never emits —
    /// `COPY` (breaks sharing), `IRELATIVE` (ifuncs are banned),
    /// dynamic-TLS (`DTPMOD`/`DTPOFF`/`TLSDESC` — the load set is
    /// closed, IE/LE only).  Presence marks a foreign/hostile image.
    Rejected(u32),
    /// Not a relocation this ABI defines for the eager set — refused
    /// identically (defense in depth), but distinguishable in
    /// diagnostics.
    Unknown(u32),
}

/// Static-TLS placement rules — the per-arch facts the IE/LE layout
/// builder needs ([`crate::apply`]).  Variant vocabulary per Drepper's
/// TLS paper as pinned by each ABI supplement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TlsRules {
    /// Variant 1: TLS blocks sit **above** the thread pointer, after a
    /// `tcb`-byte reserved header (aarch64: 16; riscv64: 0).  The first
    /// block starts at `tp + max(tcb, align)`-rounded placement and
    /// offsets are positive.
    V1 { tcb: u64 },
    /// Variant 2 (x86_64): TLS blocks sit **below** the thread pointer
    /// (offsets negative) and `tp` (`%fs:0`) holds the TCB self-pointer.
    V2,
}

/// The per-ISA relocation contract: classification tables + TLS layout
/// facts.  Implementations are pure data — the engine owns all logic.
pub trait RelocArch {
    /// `e_machine` this impl serves (cross-checked against the image).
    const MACHINE: u16;
    /// The ISA's static-TLS placement rules.
    const TLS: TlsRules;
    /// Classify a raw `r_type` into an eager semantic, or refuse.
    fn classify(r_type: u32) -> Result<RelocKind, RelocRefusal>;
}

// ============================================================================
// x86_64 — System V AMD64 psABI, table 4.10 "Relocation Types"
// ============================================================================

/// x86-64 relocation data (`EM_X86_64 = 62`).
pub struct X86_64;

impl RelocArch for X86_64 {
    const MACHINE: u16 = 62;
    /// Variant 2: blocks below `tp`, `%fs:0` = self-pointer.
    const TLS: TlsRules = TlsRules::V2;

    fn classify(r_type: u32) -> Result<RelocKind, RelocRefusal> {
        match r_type {
            8 => Ok(RelocKind::Relative), // R_X86_64_RELATIVE
            6 => Ok(RelocKind::GlobDat),  // R_X86_64_GLOB_DAT
            7 => Ok(RelocKind::JumpSlot), // R_X86_64_JUMP_SLOT
            1 => Ok(RelocKind::Abs64),    // R_X86_64_64
            18 => Ok(RelocKind::TpRel),   // R_X86_64_TPOFF64
            // The reject list (psABI numbers):
            //   5 = COPY, 16 = DTPMOD64, 17 = DTPOFF64, 36 = TLSDESC,
            //   37 = IRELATIVE.
            5 | 16 | 17 | 36 | 37 => Err(RelocRefusal::Rejected(r_type)),
            other => Err(RelocRefusal::Unknown(other)),
        }
    }
}

// ============================================================================
// aarch64 — ARM AAELF64 (IHI 0056), §4.6 dynamic relocations
// ============================================================================

/// AArch64 relocation data (`EM_AARCH64 = 183`).
pub struct AArch64;

impl RelocArch for AArch64 {
    const MACHINE: u16 = 183;
    /// Variant 1 with a 16-byte reserved TCB at `tp`; the first block is
    /// placed at `tp + 16` rounded up to the segment alignment.
    const TLS: TlsRules = TlsRules::V1 { tcb: 16 };

    fn classify(r_type: u32) -> Result<RelocKind, RelocRefusal> {
        match r_type {
            1027 => Ok(RelocKind::Relative), // R_AARCH64_RELATIVE
            1025 => Ok(RelocKind::GlobDat),  // R_AARCH64_GLOB_DAT
            1026 => Ok(RelocKind::JumpSlot), // R_AARCH64_JUMP_SLOT
            257 => Ok(RelocKind::Abs64),     // R_AARCH64_ABS64
            1030 => Ok(RelocKind::TpRel),    // R_AARCH64_TLS_TPREL64
            // Reject list (AAELF64 numbers):
            //   1024 = COPY, 1028 = TLS_DTPMOD64, 1029 = TLS_DTPREL64,
            //   1031 = TLSDESC, 1032 = IRELATIVE.
            1024 | 1028 | 1029 | 1031 | 1032 => Err(RelocRefusal::Rejected(r_type)),
            other => Err(RelocRefusal::Unknown(other)),
        }
    }
}

// ============================================================================
// riscv64 — RISC-V psABI, "Relocations" table
// ============================================================================

/// RV64 relocation data (`EM_RISCV = 243`).
pub struct Riscv64;

impl RelocArch for Riscv64 {
    const MACHINE: u16 = 243;
    /// Variant 1 with **zero** reserved TCB: `tp` points one past the
    /// TCB end, so the first block starts at `tp + 0` (the psABI's
    /// "tp contains the address one past the end of the TCB").  The
    /// `0x800` DTPREL bias applies only to dynamic-model relocs, which
    /// the reject list bans — no bias appears in the IE/LE path.
    const TLS: TlsRules = TlsRules::V1 { tcb: 0 };

    fn classify(r_type: u32) -> Result<RelocKind, RelocRefusal> {
        match r_type {
            3 => Ok(RelocKind::Relative), // R_RISCV_RELATIVE
            // RISC-V has no GLOB_DAT: GOT data slots are plain
            // R_RISCV_64 (= S + A, addend honored).  Classified Abs64 so
            // the engine computes it faithfully.
            2 => Ok(RelocKind::Abs64),    // R_RISCV_64
            5 => Ok(RelocKind::JumpSlot), // R_RISCV_JUMP_SLOT
            11 => Ok(RelocKind::TpRel),   // R_RISCV_TLS_TPREL64
            // Reject list (psABI numbers):
            //   4 = COPY, 7 = TLS_DTPMOD64, 9 = TLS_DTPREL64,
            //   12 = TLSDESC, 58 = IRELATIVE.
            4 | 7 | 9 | 12 | 58 => Err(RelocRefusal::Rejected(r_type)),
            other => Err(RelocRefusal::Unknown(other)),
        }
    }
}

// ============================================================================
// Selection — the seam's other half
// ============================================================================

#[cfg(target_arch = "aarch64")]
pub use AArch64 as HostArch;
#[cfg(target_arch = "riscv64")]
pub use Riscv64 as HostArch;
/// The [`RelocArch`] for the ISA this build **executes on**.
///
/// The three impls above are the contract's data; this alias is its
/// *selection*, and it is the only `#[cfg(target_arch)]` the seam is allowed
/// to carry.  Every site that loads an image meant for the running machine —
/// the kernel's founding check, the exec service's founder, the `kore-ld`
/// stub — names `HostArch`, so adding an ISA is one `impl` plus one row here
/// rather than an edit at each call site.  (Sites that deliberately serve
/// *any* ISA — the image service's cross-arch checker, which dispatches on the
/// image's own `e_machine` — keep naming the concretes; that is a different
/// question and rightly asks it explicitly.)
///
/// Naming `X86_64` directly in a would-be-neutral site is how the aarch64
/// boot came to refuse every one of its own service images with
/// `Parse(Machine)`: the images were correct, the loader was asking for 62.
#[cfg(target_arch = "x86_64")]
pub use X86_64 as HostArch;

#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "riscv64"
)))]
compile_error!(
    "kore-ld has no RelocArch impl for this target: add one beside X86_64 / \
     AArch64 / Riscv64 and give it a HostArch row"
);

#[cfg(test)]
mod tests {
    use super::*;

    /// The host alias resolves to the impl whose `MACHINE` this build's own
    /// ELFs carry — the property every `HostArch` call site depends on.
    #[test]
    fn host_alias_matches_this_build_s_machine() {
        let expected: u16 = if cfg!(target_arch = "x86_64") {
            62
        } else if cfg!(target_arch = "aarch64") {
            183
        } else {
            243
        };
        assert_eq!(HostArch::MACHINE, expected);
    }

    /// The spec tables, pinned: a wrong number here is a security bug
    /// (a reject-list reloc classified eager, or vice versa).
    #[test]
    fn x86_64_table_matches_psabi() {
        assert_eq!(X86_64::classify(8), Ok(RelocKind::Relative));
        assert_eq!(X86_64::classify(6), Ok(RelocKind::GlobDat));
        assert_eq!(X86_64::classify(7), Ok(RelocKind::JumpSlot));
        assert_eq!(X86_64::classify(1), Ok(RelocKind::Abs64));
        assert_eq!(X86_64::classify(18), Ok(RelocKind::TpRel));
        for rejected in [5, 16, 17, 36, 37] {
            assert_eq!(
                X86_64::classify(rejected),
                Err(RelocRefusal::Rejected(rejected))
            );
        }
        // PC32 (2) is a link-time reloc — never eager; refuses as unknown.
        assert_eq!(X86_64::classify(2), Err(RelocRefusal::Unknown(2)));
        assert_eq!(X86_64::MACHINE, 62);
        assert_eq!(X86_64::TLS, TlsRules::V2);
    }

    #[test]
    fn aarch64_table_matches_aaelf64() {
        assert_eq!(AArch64::classify(1027), Ok(RelocKind::Relative));
        assert_eq!(AArch64::classify(1025), Ok(RelocKind::GlobDat));
        assert_eq!(AArch64::classify(1026), Ok(RelocKind::JumpSlot));
        assert_eq!(AArch64::classify(257), Ok(RelocKind::Abs64));
        assert_eq!(AArch64::classify(1030), Ok(RelocKind::TpRel));
        for rejected in [1024, 1028, 1029, 1031, 1032] {
            assert_eq!(
                AArch64::classify(rejected),
                Err(RelocRefusal::Rejected(rejected))
            );
        }
        assert_eq!(AArch64::MACHINE, 183);
        assert_eq!(AArch64::TLS, TlsRules::V1 { tcb: 16 });
    }

    #[test]
    fn riscv64_table_matches_psabi() {
        assert_eq!(Riscv64::classify(3), Ok(RelocKind::Relative));
        assert_eq!(Riscv64::classify(2), Ok(RelocKind::Abs64));
        assert_eq!(Riscv64::classify(5), Ok(RelocKind::JumpSlot));
        assert_eq!(Riscv64::classify(11), Ok(RelocKind::TpRel));
        for rejected in [4, 7, 9, 12, 58] {
            assert_eq!(
                Riscv64::classify(rejected),
                Err(RelocRefusal::Rejected(rejected))
            );
        }
        assert_eq!(Riscv64::MACHINE, 243);
        assert_eq!(Riscv64::TLS, TlsRules::V1 { tcb: 0 });
    }
}
