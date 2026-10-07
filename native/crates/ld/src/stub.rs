//! The linker stub's **decisions**, separated from its entry.
//!
//! `user/runtime/ld-stub` is the binary whose validated PT_LOADs the parent shares inside
//! `LAYOUT.stub` before entering its ELF entry in every two-phase founding
//! (`res/IMAGE.md` §4, phase 2).
//! Its `_start` is a handful of instructions — read a block, jump — but
//! the reading is a gate: an unstamped page, a library with no entry, and
//! a founding that passed no boot page must all refuse rather than jump
//! somewhere arbitrary, in a context with no console, no channel and no
//! runtime to report from.
//!
//! That gate lives here rather than in the binary because **`user/` is a
//! separate workspace, excluded from the root one on purpose** — `make
//! test` runs `test-kore test-crates` and reaches none of it (userland
//! behaviour is proven by the QEMU boot gates instead). A `#[cfg(test)]`
//! module inside `user/runtime/ld-stub` would never run, so the logic that decides
//! whether to jump would be the one part of the stub nothing exercises.
//! `kore-ld` is already in the root workspace and already owns image
//! shapes, so the decision is testable here and the binary keeps only the
//! `unsafe` it cannot avoid.

use kore_abi::image::{IMAGE_ABI, LdBlock, kind};
use kore_abi::task::BootBlock;

/// Byte offset of the `.kore.ld` block within the boot page.
///
/// Mirrors `kore_rt::entry::LD_BLOCK_AT`. Restated rather than imported
/// because the stub cannot depend on `kore-rt` — it runs *before* the
/// runtime it hands control to — and `kore-ld` must not either.
/// [`ld_block_offset_matches_the_runtime_layout`] pins the two together.
///
/// [`ld_block_offset_matches_the_runtime_layout`]: self::tests
pub const LD_BLOCK_AT: usize = core::mem::size_of::<BootBlock>();

/// The exe's entry VA from a parent-written `.kore.ld` block — a VA the
/// stub may jump to **directly**, or `None` meaning *park*.
///
/// Three refusals, all of them "do not jump":
///
/// - no [`IMAGE_ABI`] stamp — an unwritten boot page reads all-zero, so
///   the stamp separates "the parent wrote this" from "nobody did";
/// - a kind whose addresses are not absolute ([`kind::is_absolute`]) —
///   an [`EXECUTABLE`](kind::EXECUTABLE) descriptor is image-relative and
///   its entry means nothing until a load base is chosen, which the stub
///   does not do until it maps the image itself.  Reading the kind is
///   what keeps that a refusal instead of a jump to an offset;
/// - a zero entry — a library, which has nothing to transfer to.
pub fn exe_entry(ld: &LdBlock) -> Option<u64> {
    if ld.abi.get() != IMAGE_ABI {
        return None;
    }
    if !kind::is_absolute(ld.exe.kind) {
        return None;
    }
    match ld.exe.entry.get() {
        0 => None,
        entry => Some(entry),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kore_abi::image::ModuleDescriptor;

    /// The stub reads `.kore.ld` at an offset it restates; a drift from
    /// the runtime's layout would have it decode the *geometry* block as
    /// a link block and jump to whatever that produced.
    #[test]
    fn ld_block_offset_matches_the_runtime_layout() {
        assert_eq!(
            LD_BLOCK_AT,
            core::mem::size_of::<BootBlock>(),
            "`.kore.ld` sits immediately past `.kore.boot`"
        );
    }

    /// An unwritten boot page is all-zero. Refusing it is the difference
    /// between parking and jumping to address zero.
    #[test]
    fn an_unstamped_block_refuses() {
        let blank = LdBlock {
            abi: 0.into(),
            flags: 0.into(),
            reserved: 0.into(),
            exe_name: kore_abi::image::ImageName::empty(),
            exe: ModuleDescriptor::empty(),
        };
        assert_eq!(exe_entry(&blank), None);
    }

    /// A stamped block whose descriptor is a library names no entry.
    /// Distinct from a failed read, and equally not jumpable.
    #[test]
    fn a_stamped_library_has_no_entry() {
        let exe = ModuleDescriptor {
            kind: kind::STATIC_EXECUTABLE,
            entry: 0.into(),
            ..ModuleDescriptor::empty()
        };
        assert_eq!(
            exe_entry(&LdBlock::new(kore_abi::image::ImageName::empty(), exe)),
            None
        );
    }

    /// **An image-relative descriptor refuses.**  A PIE executable's
    /// entry is an offset, not an address; jumping to it would land at a
    /// low address that happens to be mapped by something else.  The
    /// descriptor says which convention it uses, so the stub can tell —
    /// it cannot see the image, only the block.
    #[test]
    fn a_relocatable_executable_refuses_its_relative_entry() {
        let exe = ModuleDescriptor {
            kind: kind::EXECUTABLE,
            entry: 0x1190.into(),
            ..ModuleDescriptor::empty()
        };
        assert_eq!(
            exe_entry(&LdBlock::new(kore_abi::image::ImageName::empty(), exe)),
            None,
            "image-relative entries need a load base the stub has not chosen"
        );
    }

    /// An unknown kind is refused, not assumed absolute — the fail-closed
    /// direction for a vocabulary that grows.
    #[test]
    fn an_unknown_kind_refuses() {
        let exe = ModuleDescriptor {
            kind: 0xff,
            entry: 0x4000_2240.into(),
            ..ModuleDescriptor::empty()
        };
        assert_eq!(
            exe_entry(&LdBlock::new(kore_abi::image::ImageName::empty(), exe)),
            None
        );
    }

    /// The happy path yields exactly the entry the parent recorded.
    #[test]
    fn a_stamped_executable_yields_its_entry() {
        let exe = ModuleDescriptor {
            kind: kind::STATIC_EXECUTABLE,
            entry: 0x4000_2240.into(),
            ..ModuleDescriptor::empty()
        };
        assert_eq!(
            exe_entry(&LdBlock::new(kore_abi::image::ImageName::empty(), exe)),
            Some(0x4000_2240)
        );
    }
}
