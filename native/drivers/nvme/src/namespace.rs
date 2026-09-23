//! The **namespace** — `spdk_nvme_ns`: an addressable block store on the
//! controller (nsid + geometry), and the builder for its I/O commands.
//!
//! Kept a **pure command builder** (SPDK-shaped): a namespace holds
//! metadata and constructs [`NvmeSqe`]s; the [`Qpair`](crate::Qpair) is the
//! I/O engine that submits and polls them.  Data rides **caller DMA
//! buffers** (zero-copy — the device reads/writes the frames the caller
//! owns), addressed by PRP at the controller's **configured page size**
//! ([`MemPage`], carried here so every transfer's math matches `CC.MPS`).

use kore_abi::device::IovaAddr;

use crate::commands::{DsmRange, NvmeSqe};
use crate::prp::{self, ListPage, MemPage};
use crate::{Error, Result};

// Re-assert the DSM table shape where the builder is (16 B × count).
const _: () = assert!(core::mem::size_of::<DsmRange>() == 16);

/// NVM Read/Write CDW12.NLB is a 16-bit zero-based field: one command can
/// encode at most 65,536 logical blocks, independent of MDTS and PRP reach.
pub const MAX_NLB_BLOCKS: u32 = u16::MAX as u32 + 1;

/// One namespace on a controller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Namespace {
    /// Namespace id (1-based).
    pub nsid: u32,
    /// Logical block size in bytes.
    pub block_size: u32,
    /// Size in logical blocks.
    pub block_count: u64,
    /// The controller's configured memory page (PRP math).
    pub page: MemPage,
    /// The controller's MDTS in bytes (0 = unlimited) — transfers beyond
    /// it refuse with [`Error::ExceedsMdts`] instead of reaching the
    /// device as an invalid command.
    pub mdts_bytes: u64,
}

impl Namespace {
    /// A namespace from its identify-derived geometry + the controller's
    /// configured page + MDTS.
    pub const fn new(
        nsid: u32,
        block_size: u32,
        block_count: u64,
        page: MemPage,
        mdts_bytes: u64,
    ) -> Self {
        Self {
            nsid,
            block_size,
            block_count,
            page,
            mdts_bytes,
        }
    }

    /// Byte length of `blocks` logical blocks.
    pub const fn bytes(&self, blocks: u32) -> u64 {
        blocks as u64 * self.block_size as u64
    }

    /// Build a **Read** command: `blocks` logical blocks from `lba` into
    /// the caller's DMA buffer at `data_iova` (one contiguous device
    /// window — the [`Dma`](crate::seam::Dma) contract), addressed by PRP.
    /// A transfer spanning more than two pages needs `list` (a PRP-list
    /// page); the range is bounds-checked against the namespace.
    pub fn read(
        &self,
        lba: u64,
        blocks: u32,
        data_iova: IovaAddr,
        list: Option<ListPage<'_>>,
    ) -> Result<NvmeSqe> {
        self.check_range(lba, blocks)?;
        let p = prp::build(data_iova, self.bytes(blocks), self.page, list)?;
        Ok(NvmeSqe::read(self.nsid, lba, (blocks - 1) as u16, p))
    }

    /// Build a **Write** command (mirror of [`read`](Self::read)).
    pub fn write(
        &self,
        lba: u64,
        blocks: u32,
        data_iova: IovaAddr,
        list: Option<ListPage<'_>>,
    ) -> Result<NvmeSqe> {
        self.check_range(lba, blocks)?;
        let p = prp::build(data_iova, self.bytes(blocks), self.page, list)?;
        Ok(NvmeSqe::write(self.nsid, lba, (blocks - 1) as u16, p))
    }

    /// Build a **Flush** command.
    pub fn flush(&self) -> NvmeSqe {
        NvmeSqe::flush(self.nsid)
    }

    /// Build a **Deallocate** (Dataset Management, the discard/TRIM path
    /// a swap backend rides): the `count`-entry [`DsmRange`] table sits in
    /// the caller's DMA buffer at `ranges_iova` (`count` is 1-based here,
    /// ≤ 256).
    pub fn deallocate(&self, ranges_iova: IovaAddr, count: u16) -> Result<NvmeSqe> {
        if count == 0 || count > 256 {
            return Err(Error::Qpair(crate::qpair::QpairError::BadDepth));
        }
        Ok(NvmeSqe::dsm_deallocate(
            self.nsid,
            (count - 1) as u8,
            ranges_iova,
        ))
    }

    /// `blocks ≥ 1`, `lba + blocks ≤ block_count`, and within MDTS.
    fn check_range(&self, lba: u64, blocks: u32) -> Result<()> {
        let ok = blocks >= 1
            && lba
                .checked_add(blocks as u64)
                .is_some_and(|end| end <= self.block_count);
        if !ok {
            return Err(Error::Qpair(crate::qpair::QpairError::BadDepth));
        }
        if blocks > MAX_NLB_BLOCKS {
            return Err(Error::Geometry);
        }
        if self.mdts_bytes != 0 && self.bytes(blocks) > self.mdts_bytes {
            return Err(Error::ExceedsMdts);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prp::PrpError;

    const PG: MemPage = MemPage::MIN;

    #[test]
    fn read_builds_a_single_prp_command_within_range() {
        let ns = Namespace::new(1, 4096, 0x1000, PG, 0);
        let cmd = ns.read(0, 1, IovaAddr(0xdead_0000), None).unwrap();
        assert_eq!(cmd.opcode(), crate::commands::IoOpcode::Read as u8);
        assert_eq!(cmd.nsid, 1);
        assert_eq!(cmd.prp1, 0xdead_0000);
        assert_eq!(cmd.prp2, 0, "one 4 KiB block is a single PRP");
        assert_eq!(cmd.cdw12, 0, "NLB is 0-based (1 block ⇒ 0)");
    }

    #[test]
    fn out_of_range_and_zero_length_refuse() {
        // 4 KiB blocks so a 1-block read stays a single PRP (isolates the
        // range check from PRP sizing).
        let ns = Namespace::new(1, 4096, 100, PG, 0);
        assert!(
            ns.read(100, 1, IovaAddr(0x1000), None).is_err(),
            "first LBA past the end"
        );
        assert!(
            ns.read(99, 2, IovaAddr(0x1000), None).is_err(),
            "spills past the end"
        );
        assert!(
            ns.read(0, 0, IovaAddr(0x1000), None).is_err(),
            "zero blocks"
        );
        assert!(
            ns.read(99, 1, IovaAddr(0x1000), None).is_ok(),
            "the last block fits"
        );
    }

    #[test]
    fn a_large_transfer_needs_a_list() {
        // 4 KiB blocks, 4 blocks = 4 pages ⇒ needs a PRP list.
        let ns = Namespace::new(1, 4096, 0x1000, PG, 0);
        assert!(matches!(
            ns.read(0, 4, IovaAddr(0x1000), None),
            Err(Error::Prp(PrpError::ListNeeded))
        ));
    }

    #[test]
    fn a_larger_configured_page_relaxes_the_list_requirement() {
        // The same 4-block (16 KiB) read fits two 8 KiB pages ⇒ no list —
        // the namespace's page (from CC.MPS) governs, not a constant.
        let ns = Namespace::new(1, 4096, 0x1000, MemPage::from_mps(1), 0);
        let cmd = ns.read(0, 4, IovaAddr(0x4000), None).unwrap();
        assert_eq!(cmd.prp2, 0x6000, "PRP2-direct at 8 KiB pages");
    }

    #[test]
    fn nlb_encoding_limit_is_independent_of_namespace_and_mdts() {
        let ns = Namespace::new(1, 512, u64::MAX, PG, 0);
        assert!(ns.check_range(0, MAX_NLB_BLOCKS).is_ok());
        assert_eq!(ns.check_range(0, MAX_NLB_BLOCKS + 1), Err(Error::Geometry));
    }
}
