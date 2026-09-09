//! The SMMUv3 device-MMU format plugin — the first consumer of the
//! [`pagetable`](crate::common::pagetable) seam.
//!
//! Pure-value spec encoders ([`descriptor`]) ported from the kernel's proven
//! platform SMMUv3, plus the **new** stage-1 page-table format ([`stage1`]) a
//! device-embedded SMMUv3 uses to translate a device VA → IOVA.
//! [`Smmuv3DeviceContext`] ties the three pieces — stream-table entry, context
//! descriptor, and the built stage-1 table root — into the "program one
//! per-context translation" shape, all in IOVA terms (no physical address).
//!
//! # Stage-1 only, deliberately
//!
//! This library serves **device-private** MMUs: the block sitting inside an
//! accelerator, translating the addresses that device's own engines emit into
//! IOVAs. The second stage — IOVA → physical, and the nesting a hypervisor
//! needs — happens *downstream*, at the platform IOMMU, which is a different
//! piece of silicon with a different driver
//! (`platform::arch::arm64::iommu::smmuv3` on the kernel side, which retains
//! the full stage-2 surface).
//!
//! So the stage-2 encoders this module was seeded with — `StreamEntry::stage2`
//! / `nested`, the VTCR builder, `TLBI_S12_VMALL` / `TLBI_S2_IPA` — were
//! removed rather than carried: nothing consumed them, a device-embedded SMMU
//! has no stage-2 to program, and keeping them invited a driver to configure a
//! translation stage its hardware does not implement. Recover them from git
//! history if a device ever turns up that genuinely nests internally.
//!
//! What the device axis *did* lack, and now has, is
//! [`Command::tlbi_va_range`] / [`Command::tlbi_va_sweep`]: stage-2 had range
//! invalidation and stage-1 did not, which is backwards for this library's
//! purpose. Devices retire *buffers*, not pages.
//!
//! Fault handling (PRI, stall/resume) is kept and is *more* relevant here than
//! on a CPU-side IOMMU — a device-embedded SMMU stalling its engine on a
//! translation fault is the normal recovery path, not an exceptional one.

mod descriptor;
mod stage1;

pub use descriptor::{
    COMMAND_ENTRY_BYTES, CONTEXT_ENTRY_BYTES, Command, ContextDescriptor, EVENT_ENTRY_BYTES,
    MAX_LINEAR_CD_BITS, PRI_ENTRY_BYTES, STREAM_ENTRY_BYTES, SmmuV3Fault, SmmuV3FaultReason,
    SmmuV3PriEntry, StreamEntry,
};
pub use stage1::{
    STAGE1_MAIR, SmmuV3AddressWidth, SmmuV3OutputAddressSize, SmmuV3Stage1Flags,
    SmmuV3Stage1Meta39, SmmuV3Stage1Meta48, SmmuV3Stage1Pte, SmmuV3Stage1Table39,
    SmmuV3Stage1Table48,
};

use kore_abi::device::IovaAddr;

/// A response to a device page-request (PRI) or a stalled fault resume.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageResponse {
    /// Translation succeeded — retry the access.
    Success,
    /// The request was invalid — do not retry.
    Invalid,
    /// A permanent failure — abort the access.
    Failure,
}

/// Assembles the per-context SMMUv3 programming for a built stage-1 table.
///
/// Pure data: it produces the descriptor **values** (context descriptor +
/// stream-table entry) a driver writes into its CD / stream tables. The MMIO
/// writes, barriers, and CFGI/TLBI invalidation choreography stay in the
/// consuming driver — this layer never touches hardware.
#[derive(Clone, Copy, Debug)]
pub struct Smmuv3DeviceContext {
    /// Input (device VA) geometry.
    pub width: SmmuV3AddressWidth,
    /// Output-address size the implementation supports.
    pub output: SmmuV3OutputAddressSize,
    /// ASID tagging this context's stage-1 TLB entries.
    pub asid: u16,
}

impl Smmuv3DeviceContext {
    /// A 4 KiB-granule context over the given geometry and ASID.
    #[inline]
    pub const fn new(
        width: SmmuV3AddressWidth,
        output: SmmuV3OutputAddressSize,
        asid: u16,
    ) -> Self {
        Self {
            width,
            output,
            asid,
        }
    }

    /// The context descriptor for a stage-1 table rooted at `root` (the walker's
    /// [`root()`](crate::common::pagetable::DeviceWalker::root)). Carries
    /// [`STAGE1_MAIR`] so the leaves' AttrIndx values resolve, TG0 = 0 (4 KiB),
    /// and T0SZ / IPS from the configured geometry.
    #[inline]
    pub const fn context_descriptor(&self, root: IovaAddr, stall: bool) -> ContextDescriptor {
        ContextDescriptor::stage1(
            root.0,
            self.asid,
            stall,
            STAGE1_MAIR,
            self.width.t0sz(),
            0,
            self.output.ps_encoding(),
        )
    }

    /// The stream-table entry pointing at a context-descriptor table at
    /// `cd_table` (`cd_bits` = substream-id bits; 0 for a single context).
    ///
    /// `cd_bits` is clamped to [`MAX_LINEAR_CD_BITS`] — this layer only builds
    /// linear CD tables, and an over-wide S1CDMax would let the SMMU walk past
    /// the table a driver actually allocated (the same clamp the kernel's
    /// SMMUv3 unit applies at its stream-entry callsite).
    #[inline]
    pub const fn stream_entry(
        &self,
        cd_table: IovaAddr,
        cd_bits: u8,
        ats: bool,
        stall: bool,
    ) -> StreamEntry {
        let cd_bits = if cd_bits > MAX_LINEAR_CD_BITS {
            MAX_LINEAR_CD_BITS
        } else {
            cd_bits
        };
        StreamEntry::stage1(cd_table.0, cd_bits, ats, stall, false, false)
    }
}

#[cfg(test)]
mod tests {
    use alloc::boxed::Box;
    use alloc::vec::Vec;

    use kore_abi::device::IovaAddr;

    use super::{
        Smmuv3DeviceContext,
        stage1::{SmmuV3Stage1Flags, SmmuV3Stage1Meta48, SmmuV3Stage1Pte, SmmuV3Stage1Table48},
    };
    use crate::common::flags::{AccessFlags, PageSize};
    use crate::common::pagetable::{
        DevVirtAddr, DeviceRange, DeviceWalkerView, MmuError, MmuResult, TableSource,
    };

    /// A host [`TableSource`]: hands out page-aligned synthetic IOVAs
    /// `((idx + 1) << 12)` and owns leaked 4 KiB frames so `resolve` maps an
    /// IOVA back to its window and leaf/table addresses round-trip through the
    /// format's low-12-bit masking.
    struct TestTables {
        frames: Vec<*mut [u64; 512]>,
        output_bias: u64,
        table_bytes: usize,
    }

    impl TestTables {
        fn new() -> Self {
            Self {
                frames: Vec::new(),
                output_bias: 0,
                table_bytes: 0x1000,
            }
        }

        fn with_geometry(output_bias: u64, table_bytes: usize) -> Self {
            Self {
                frames: Vec::new(),
                output_bias,
                table_bytes,
            }
        }
    }

    impl TableSource for TestTables {
        type Addr = IovaAddr;

        fn alloc_table(&mut self) -> MmuResult<IovaAddr> {
            let frame = Box::into_raw(Box::new([0u64; 512]));
            let idx = self.frames.len();
            self.frames.push(frame);
            Ok(IovaAddr((((idx as u64) + 1) << 12) + self.output_bias))
        }

        fn resolve(&self, addr: IovaAddr) -> Option<*mut u8> {
            let raw = addr.0.checked_sub(self.output_bias)?;
            let idx = ((raw >> 12).checked_sub(1)?) as usize;
            self.frames.get(idx).map(|&frame| frame as *mut u8)
        }

        fn release(&mut self, _addr: IovaAddr) {
            // Frames are bulk-freed on drop.
        }

        fn table_bytes(&self) -> usize {
            self.table_bytes
        }
    }

    impl Drop for TestTables {
        fn drop(&mut self) {
            for &frame in &self.frames {
                // SAFETY: each pointer came from `Box::into_raw` and is freed once.
                unsafe { drop(Box::from_raw(frame)) };
            }
        }
    }

    #[test]
    fn map_then_translate_returns_iova_leaf_and_size() {
        let mut src = TestTables::new();
        let mut pt = SmmuV3Stage1Table48::create(&mut src).unwrap();
        let flags = SmmuV3Stage1Flags::new(AccessFlags::READ | AccessFlags::WRITE);

        // 2 MiB leaf: the leaf holds the IOVA, never a physical address.
        pt.map(
            &mut src,
            DevVirtAddr(0x20_0000),
            IovaAddr(0x4020_0000),
            PageSize::Size2M,
            flags,
        )
        .unwrap();
        assert_eq!(
            pt.translate(&src, DevVirtAddr(0x20_0000)),
            Some((IovaAddr(0x4020_0000), PageSize::Size2M))
        );

        // 4 KiB leaf elsewhere.
        pt.map(
            &mut src,
            DevVirtAddr(0x1000),
            IovaAddr(0x9000),
            PageSize::Size4K,
            flags,
        )
        .unwrap();
        assert_eq!(
            pt.translate(&src, DevVirtAddr(0x1000)),
            Some((IovaAddr(0x9000), PageSize::Size4K))
        );

        assert_eq!(pt.translate(&src, DevVirtAddr(0x8000)), None);
    }

    #[test]
    fn double_map_rejected_and_unmap_clears() {
        let mut src = TestTables::new();
        let mut pt = SmmuV3Stage1Table48::create(&mut src).unwrap();
        let flags = SmmuV3Stage1Flags::new(AccessFlags::READ);

        pt.map(
            &mut src,
            DevVirtAddr(0x1000),
            IovaAddr(0x1000),
            PageSize::Size4K,
            flags,
        )
        .unwrap();
        assert_eq!(
            pt.map(
                &mut src,
                DevVirtAddr(0x1000),
                IovaAddr(0x2000),
                PageSize::Size4K,
                flags
            ),
            Err(MmuError::AlreadyMapped)
        );

        pt.unmap(&mut src, DevVirtAddr(0x1000), PageSize::Size4K)
            .unwrap();
        assert_eq!(pt.translate(&src, DevVirtAddr(0x1000)), None);
    }

    #[test]
    fn map_range_splits_into_largest_leaves_then_tears_down() {
        let mut src = TestTables::new();
        let mut pt = SmmuV3Stage1Table48::create(&mut src).unwrap();
        let flags = SmmuV3Stage1Flags::new(AccessFlags::READ | AccessFlags::WRITE);

        // 2 MiB + 4 KiB from a 2 MiB-aligned base → one 2 MiB then one 4 KiB.
        pt.map_range(
            &mut src,
            DevVirtAddr(0x20_0000),
            IovaAddr(0x20_0000),
            0x20_1000,
            flags,
        )
        .unwrap();
        assert_eq!(
            pt.translate(&src, DevVirtAddr(0x20_0000)).unwrap().1,
            PageSize::Size2M
        );
        assert_eq!(
            pt.translate(&src, DevVirtAddr(0x40_0000)).unwrap().1,
            PageSize::Size4K
        );

        // teardown walks and releases every table frame without panicking; the
        // test source bulk-frees on drop, so this exercises the walk + release.
        let frame_count = src.frames.len();
        assert!(frame_count >= 2, "expected root + at least one level table");
        pt.teardown(&mut src).unwrap();
    }

    #[test]
    fn unknown_table_frame_is_a_typed_refusal() {
        // A reconstructed root view carries observation but no mutation or
        // teardown authority. An unknown frame is a typed refusal / clean miss.
        let src = TestTables::new();
        let pt = DeviceWalkerView::<SmmuV3Stage1Meta48, SmmuV3Stage1Pte>::from_root(IovaAddr(
            0x0dea_d000,
        ));
        assert_eq!(
            pt.query(&src, DevVirtAddr(0x1000)),
            Err(MmuError::UnknownTableFrame)
        );
        assert_eq!(pt.translate(&src, DevVirtAddr(0x1000)), None);
    }

    #[test]
    fn malformed_table_geometry_is_refused_before_a_frame_is_touched() {
        let mut wrong_size = TestTables::with_geometry(0, 0x2000);
        assert!(matches!(
            SmmuV3Stage1Table48::create(&mut wrong_size),
            Err(MmuError::InvalidTableGeometry)
        ));

        let mut misaligned = TestTables::with_geometry(1, 0x1000);
        assert!(matches!(
            SmmuV3Stage1Table48::create(&mut misaligned),
            Err(MmuError::InvalidTableGeometry)
        ));

        let mut out_of_range = TestTables::with_geometry(1 << 48, 0x1000);
        assert!(matches!(
            SmmuV3Stage1Table48::create(&mut out_of_range),
            Err(MmuError::InvalidTableGeometry)
        ));
    }

    #[test]
    fn query_names_the_exact_leaf_and_translate_applies_the_offset() {
        let mut src = TestTables::new();
        let mut pt = SmmuV3Stage1Table48::create(&mut src).unwrap();
        let flags = SmmuV3Stage1Flags::new(AccessFlags::READ);
        pt.map(
            &mut src,
            DevVirtAddr(0x20_0000),
            IovaAddr(0x80_0000),
            PageSize::Size2M,
            flags,
        )
        .unwrap();

        let at = DevVirtAddr(0x20_4321);
        let mapping = pt.query(&src, at).unwrap();
        assert_eq!(mapping.range.start(), DevVirtAddr(0x20_0000));
        assert_eq!(mapping.range.end(), DevVirtAddr(0x40_0000));
        assert_eq!(mapping.output, IovaAddr(0x80_0000));
        assert_eq!(mapping.translate(at), Some(IovaAddr(0x80_4321)));
        assert_eq!(
            pt.translate(&src, at),
            Some((IovaAddr(0x80_4321), PageSize::Size2M))
        );
    }

    #[test]
    fn protect_returns_the_old_mapping_and_updates_only_flags() {
        let mut src = TestTables::new();
        let mut pt = SmmuV3Stage1Table48::create(&mut src).unwrap();
        let rw = SmmuV3Stage1Flags::new(AccessFlags::READ | AccessFlags::WRITE);
        let ro = SmmuV3Stage1Flags::new(AccessFlags::READ);
        pt.map(
            &mut src,
            DevVirtAddr(0x1000),
            IovaAddr(0x9000),
            PageSize::Size4K,
            rw,
        )
        .unwrap();

        let old = pt
            .protect(&mut src, DevVirtAddr(0x1000), PageSize::Size4K, ro)
            .unwrap();
        assert_eq!(old.output, IovaAddr(0x9000));
        assert!(old.flags.access().contains(AccessFlags::WRITE));
        let current = pt.query(&src, DevVirtAddr(0x1000)).unwrap();
        assert_eq!(current.output, IovaAddr(0x9000));
        assert!(!current.flags.access().contains(AccessFlags::WRITE));
    }

    #[test]
    fn rejected_rights_never_allocate_or_modify_a_mapping() {
        let mut src = TestTables::new();
        let mut pt = SmmuV3Stage1Table48::create(&mut src).unwrap();
        let root_only = src.frames.len();
        let invalid = SmmuV3Stage1Flags::new(AccessFlags::WRITE);

        assert_eq!(
            pt.map(
                &mut src,
                DevVirtAddr(0x1000),
                IovaAddr(0x9000),
                PageSize::Size4K,
                invalid,
            ),
            Err(MmuError::UnsupportedAccess)
        );
        assert_eq!(src.frames.len(), root_only);
        assert_eq!(pt.translate(&src, DevVirtAddr(0x1000)), None);

        let valid = SmmuV3Stage1Flags::new(AccessFlags::READ);
        pt.map(
            &mut src,
            DevVirtAddr(0x1000),
            IovaAddr(0x9000),
            PageSize::Size4K,
            valid,
        )
        .unwrap();
        assert_eq!(
            pt.protect(&mut src, DevVirtAddr(0x1000), PageSize::Size4K, invalid,),
            Err(MmuError::UnsupportedAccess)
        );
        assert_eq!(
            pt.query(&src, DevVirtAddr(0x1000)).unwrap().flags.access(),
            AccessFlags::READ
        );
    }

    #[test]
    fn scattered_extents_form_one_virtual_span_without_erasing_provenance() {
        let mut src = TestTables::new();
        let mut pt = SmmuV3Stage1Table48::create(&mut src).unwrap();
        let segments = [
            DeviceRange::from_start_len(IovaAddr(0x10_0000), 0x2000).unwrap(),
            DeviceRange::from_start_len(IovaAddr(0x40_0000), 0x1000).unwrap(),
        ];
        let mapped = pt
            .map_scattered(
                &mut src,
                DevVirtAddr(0x80_0000),
                &segments,
                SmmuV3Stage1Flags::new(AccessFlags::READ),
                Some(PageSize::Size4K),
            )
            .unwrap();
        assert_eq!(mapped, 0x3000);
        assert_eq!(
            pt.translate(&src, DevVirtAddr(0x80_1000)).unwrap().0,
            IovaAddr(0x10_1000)
        );
        assert_eq!(
            pt.translate(&src, DevVirtAddr(0x80_2000)).unwrap().0,
            IovaAddr(0x40_0000)
        );
    }

    #[test]
    fn range_and_exact_leaf_shape_fail_closed() {
        let mut src = TestTables::new();
        let mut pt = SmmuV3Stage1Table48::create(&mut src).unwrap();
        let flags = SmmuV3Stage1Flags::new(AccessFlags::READ);
        assert_eq!(
            pt.map_range(
                &mut src,
                DevVirtAddr(u64::MAX - 0xfff),
                IovaAddr(0x1000),
                0x2000,
                flags
            ),
            Err(MmuError::AddressOverflow)
        );
        pt.map(
            &mut src,
            DevVirtAddr(0x20_0000),
            IovaAddr(0x40_0000),
            PageSize::Size2M,
            flags,
        )
        .unwrap();
        assert_eq!(
            pt.unmap(&mut src, DevVirtAddr(0x20_1000), PageSize::Size4K),
            Err(MmuError::InvalidMappingShape)
        );
    }

    #[test]
    fn exact_range_walk_and_unmap_advance_across_mixed_leaf_sizes() {
        let mut src = TestTables::new();
        let mut pt = SmmuV3Stage1Table48::create(&mut src).unwrap();
        let flags = SmmuV3Stage1Flags::new(AccessFlags::READ);
        let range = DeviceRange::from_start_len(DevVirtAddr(0x20_0000), 0x20_1000).unwrap();
        pt.map_range(
            &mut src,
            range.start(),
            IovaAddr(0x40_0000),
            range.len(),
            flags,
        )
        .unwrap();

        let mut sizes = Vec::new();
        assert_eq!(
            pt.visit_range(&src, range, |mapping| sizes.push(mapping.size)),
            Ok(range.len())
        );
        assert_eq!(sizes, [PageSize::Size2M, PageSize::Size4K]);
        assert_eq!(pt.unmap_range(&mut src, range), Ok(range.len()));
        assert_eq!(pt.query(&src, range.start()), Err(MmuError::NotMapped));
        assert_eq!(
            pt.query(&src, DevVirtAddr(0x40_0000)),
            Err(MmuError::NotMapped)
        );
    }

    #[test]
    fn protect_range_preflights_the_whole_shape_before_granting_access() {
        let mut src = TestTables::new();
        let mut pt = SmmuV3Stage1Table48::create(&mut src).unwrap();
        let ro = SmmuV3Stage1Flags::new(AccessFlags::READ);
        let rw = SmmuV3Stage1Flags::new(AccessFlags::READ | AccessFlags::WRITE);
        pt.map(
            &mut src,
            DevVirtAddr(0x1000),
            IovaAddr(0x10_0000),
            PageSize::Size4K,
            ro,
        )
        .unwrap();
        pt.map(
            &mut src,
            DevVirtAddr(0x3000),
            IovaAddr(0x30_0000),
            PageSize::Size4K,
            ro,
        )
        .unwrap();

        let with_hole = DeviceRange::from_start_len(DevVirtAddr(0x1000), 0x3000).unwrap();
        assert_eq!(
            pt.protect_range(&mut src, with_hole, rw),
            Err(MmuError::NotMapped)
        );
        assert!(
            !pt.query(&src, DevVirtAddr(0x1000))
                .unwrap()
                .flags
                .access()
                .contains(AccessFlags::WRITE),
            "the validated prefix must not be changed when a later hole refuses"
        );

        let first = DeviceRange::from_start_len(DevVirtAddr(0x1000), 0x1000).unwrap();
        assert_eq!(pt.protect_range(&mut src, first, rw), Ok(first.len()));
        assert!(
            pt.query(&src, DevVirtAddr(0x1000))
                .unwrap()
                .flags
                .access()
                .contains(AccessFlags::WRITE)
        );
    }

    #[test]
    fn context_descriptor_carries_mair_and_geometry() {
        let ctx = Smmuv3DeviceContext::new(
            super::SmmuV3AddressWidth::Bits48,
            super::SmmuV3OutputAddressSize::Bits48,
            0x1234,
        );
        let cd = ctx.context_descriptor(IovaAddr(0x5000), false);
        // TTB0 in word 1, MAIR in word 3, ASID in word 0 bits [63:48], T0SZ=16.
        assert_eq!(cd.0[1] & 0x000f_ffff_ffff_fff0, 0x5000);
        assert_eq!(cd.0[3], super::STAGE1_MAIR);
        assert_eq!(cd.0[0] >> 48, 0x1234);
        assert_eq!(cd.0[0] & 0x3f, 16);
    }
}
