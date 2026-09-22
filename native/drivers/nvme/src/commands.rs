//! NVMe command + completion + identify structures — **spec facts** (NVMe
//! base spec §4, §5), ported from `os0/user/dev/block/nvme/commands.rs`.
//! The wire layouts are fixed by the spec; the `size_of` asserts guard
//! them.

use core::mem;

use kore_abi::device::IovaAddr;

use crate::prp::Prp;

/// Admin command opcodes (spec §5).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdminOpcode {
    /// Delete I/O Submission Queue.
    DeleteIoSq = 0x00,
    /// Create I/O Submission Queue.
    CreateIoSq = 0x01,
    /// Get Log Page.
    GetLogPage = 0x02,
    /// Delete I/O Completion Queue.
    DeleteIoCq = 0x04,
    /// Create I/O Completion Queue.
    CreateIoCq = 0x05,
    /// Identify.
    Identify = 0x06,
    /// Set Features.
    SetFeatures = 0x09,
}

/// NVM I/O command opcodes (spec §6).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IoOpcode {
    /// Flush.
    Flush = 0x00,
    /// Write.
    Write = 0x01,
    /// Read.
    Read = 0x02,
    /// Dataset Management (deallocate/TRIM — the swap `Discard` backend).
    DatasetManagement = 0x09,
}

/// A 64-byte NVMe submission queue entry (spec Figure — SQE).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NvmeSqe {
    /// CDW0: opcode / fused / PSDT / CID.
    pub cdw0: u32,
    /// Namespace ID.
    pub nsid: u32,
    pub cdw2: u32,
    pub cdw3: u32,
    /// Metadata pointer.
    pub mptr: u64,
    /// PRP entry 1.
    pub prp1: u64,
    /// PRP entry 2 (or PRP-list pointer).
    pub prp2: u64,
    pub cdw10: u32,
    pub cdw11: u32,
    pub cdw12: u32,
    pub cdw13: u32,
    pub cdw14: u32,
    pub cdw15: u32,
}

const _: () = assert!(mem::size_of::<NvmeSqe>() == 64);

impl NvmeSqe {
    /// A zeroed command.
    pub const fn zeroed() -> Self {
        Self {
            cdw0: 0,
            nsid: 0,
            cdw2: 0,
            cdw3: 0,
            mptr: 0,
            prp1: 0,
            prp2: 0,
            cdw10: 0,
            cdw11: 0,
            cdw12: 0,
            cdw13: 0,
            cdw14: 0,
            cdw15: 0,
        }
    }

    /// Stamp the command id into CDW0 (bits 31:16).
    pub fn set_cid(&mut self, cid: u16) {
        self.cdw0 = (self.cdw0 & 0x0000_FFFF) | ((cid as u32) << 16);
    }

    /// The command id.
    pub fn cid(&self) -> u16 {
        (self.cdw0 >> 16) as u16
    }

    /// The opcode (CDW0 bits 7:0).
    pub fn opcode(&self) -> u8 {
        (self.cdw0 & 0xFF) as u8
    }

    // ── Admin commands ──────────────────────────────────────────────────
    // Builders speak typed [`IovaAddr`]; the raw `u64` exists only in the
    // wire fields they fill (the one place the spec fixes the layout).

    /// Identify Controller (CNS = 1).
    pub fn identify_controller(dma: IovaAddr) -> Self {
        let mut c = Self::zeroed();
        c.cdw0 = AdminOpcode::Identify as u32;
        c.prp1 = dma.0;
        c.cdw10 = 1;
        c
    }

    /// Identify Namespace (CNS = 0).
    pub fn identify_namespace(nsid: u32, dma: IovaAddr) -> Self {
        let mut c = Self::zeroed();
        c.cdw0 = AdminOpcode::Identify as u32;
        c.nsid = nsid;
        c.prp1 = dma.0;
        c.cdw10 = 0;
        c
    }

    /// Set Features — Number of Queues (FID 0x07).  `nsq`/`ncq` are
    /// 0-based (pass 3 to request 4).
    pub fn set_num_queues(nsq: u16, ncq: u16) -> Self {
        let mut c = Self::zeroed();
        c.cdw0 = AdminOpcode::SetFeatures as u32;
        c.cdw10 = 0x07;
        c.cdw11 = ((ncq as u32) << 16) | (nsq as u32);
        c
    }

    /// Set Features — Host Memory Buffer (FID 0x0D), enabling it.
    ///
    /// `hsize_pages` is the buffer's total size and `list` the IOVA of the
    /// [`HmbDescriptor`] table with `entries` rows; both sizes are in
    /// **`CC.MPS` units**, not the 4 KiB units Identify reports in.  The
    /// table's address must be 16-byte aligned (CDW13 bits 3:0 are
    /// reserved).  Memory Return (CDW11 bit 1) is never set: this core
    /// allocates a fresh buffer every bring-up and hands nothing back.
    pub fn enable_host_memory_buffer(hsize_pages: u32, list: IovaAddr, entries: u32) -> Self {
        let mut c = Self::zeroed();
        c.cdw0 = AdminOpcode::SetFeatures as u32;
        c.cdw10 = 0x0D;
        c.cdw11 = 1; // EHM
        c.cdw12 = hsize_pages;
        c.cdw13 = list.0 as u32;
        c.cdw14 = (list.0 >> 32) as u32;
        c.cdw15 = entries;
        c
    }

    /// Set Features — Host Memory Buffer (FID 0x0D), disabling it: the
    /// controller stops using the buffer, after which its memory may be
    /// released.  Every other field is zero, as the spec asks for a disable.
    pub fn disable_host_memory_buffer() -> Self {
        let mut c = Self::zeroed();
        c.cdw0 = AdminOpcode::SetFeatures as u32;
        c.cdw10 = 0x0D;
        c
    }

    /// Create I/O Completion Queue (physically contiguous).
    pub fn create_io_cq(qid: u16, size: u16, dma: IovaAddr, vector: u16, ien: bool) -> Self {
        let mut c = Self::zeroed();
        c.cdw0 = AdminOpcode::CreateIoCq as u32;
        c.prp1 = dma.0;
        c.cdw10 = (((size - 1) as u32) << 16) | (qid as u32);
        c.cdw11 = ((vector as u32) << 16) | (if ien { 2 } else { 0 }) | 1;
        c
    }

    /// Create I/O Submission Queue (physically contiguous, bound to `cqid`).
    pub fn create_io_sq(qid: u16, size: u16, dma: IovaAddr, cqid: u16) -> Self {
        let mut c = Self::zeroed();
        c.cdw0 = AdminOpcode::CreateIoSq as u32;
        c.prp1 = dma.0;
        c.cdw10 = (((size - 1) as u32) << 16) | (qid as u32);
        c.cdw11 = ((cqid as u32) << 16) | 1;
        c
    }

    /// Delete I/O Submission Queue.
    pub fn delete_io_sq(qid: u16) -> Self {
        let mut c = Self::zeroed();
        c.cdw0 = AdminOpcode::DeleteIoSq as u32;
        c.cdw10 = qid as u32;
        c
    }

    /// Delete I/O Completion Queue.
    pub fn delete_io_cq(qid: u16) -> Self {
        let mut c = Self::zeroed();
        c.cdw0 = AdminOpcode::DeleteIoCq as u32;
        c.cdw10 = qid as u32;
        c
    }

    // ── I/O commands ────────────────────────────────────────────────────

    /// Read `nlb + 1` logical blocks from `lba` (NLB is 0-based).
    pub fn read(nsid: u32, lba: u64, nlb: u16, prp: Prp) -> Self {
        let mut c = Self::zeroed();
        c.cdw0 = IoOpcode::Read as u32;
        c.nsid = nsid;
        c.prp1 = prp.prp1.0;
        c.prp2 = prp.prp2.0;
        c.cdw10 = lba as u32;
        c.cdw11 = (lba >> 32) as u32;
        c.cdw12 = nlb as u32;
        c
    }

    /// Write `nlb + 1` logical blocks to `lba` (NLB is 0-based).
    pub fn write(nsid: u32, lba: u64, nlb: u16, prp: Prp) -> Self {
        let mut c = Self::zeroed();
        c.cdw0 = IoOpcode::Write as u32;
        c.nsid = nsid;
        c.prp1 = prp.prp1.0;
        c.prp2 = prp.prp2.0;
        c.cdw10 = lba as u32;
        c.cdw11 = (lba >> 32) as u32;
        c.cdw12 = nlb as u32;
        c
    }

    /// Flush the namespace.
    pub fn flush(nsid: u32) -> Self {
        let mut c = Self::zeroed();
        c.cdw0 = IoOpcode::Flush as u32;
        c.nsid = nsid;
        c
    }

    /// Dataset Management — **deallocate** (`AD` set) the `nr + 1` ranges
    /// in the [`DsmRange`] table at `ranges` (0-based count, ≤ 255).  The
    /// discard path a swap backend rides.
    pub fn dsm_deallocate(nsid: u32, nr: u8, ranges: IovaAddr) -> Self {
        let mut c = Self::zeroed();
        c.cdw0 = IoOpcode::DatasetManagement as u32;
        c.nsid = nsid;
        c.prp1 = ranges.0;
        c.cdw10 = nr as u32;
        c.cdw11 = 1 << 2; // AD — Attribute: Deallocate
        c
    }

    /// Get Log Page — SMART / Health Information (LID 0x02), the full
    /// 512-byte page into `dma`.  `NUMD` is in dwords, 0-based.
    pub fn get_log_smart(dma: IovaAddr) -> Self {
        let mut c = Self::zeroed();
        c.cdw0 = AdminOpcode::GetLogPage as u32;
        c.nsid = 0xFFFF_FFFF; // controller-scoped
        c.prp1 = dma.0;
        let numd = (512 / 4) - 1u32; // 0-based dword count
        c.cdw10 = (numd << 16) | 0x02;
        c
    }
}

/// One Dataset Management range descriptor (16 bytes, spec Figure).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DsmRange {
    /// Context attributes (0 for a plain deallocate).
    pub cattr: u32,
    /// Length in logical blocks (1-based).
    pub nlb: u32,
    /// Starting LBA.
    pub slba: u64,
}

const _: () = assert!(mem::size_of::<DsmRange>() == 16);

/// One Host Memory Buffer descriptor (16 bytes, spec Figure): a memory
/// page aligned chunk the controller may read and write for as long as the
/// buffer is enabled.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct HmbDescriptor {
    /// Buffer address — device-visible, `CC.MPS`-aligned.
    pub badd: u64,
    /// Buffer size in `CC.MPS` units.
    pub bsize: u32,
    _reserved: u32,
}

const _: () = assert!(mem::size_of::<HmbDescriptor>() == 16);

impl HmbDescriptor {
    /// A chunk of `pages` memory pages at `badd`.
    pub const fn new(badd: IovaAddr, pages: u32) -> Self {
        Self {
            badd: badd.0,
            bsize: pages,
            _reserved: 0,
        }
    }
}

impl DsmRange {
    /// A deallocate range: `count` blocks from `lba`.
    pub const fn new(lba: u64, count: u32) -> Self {
        Self {
            cattr: 0,
            nlb: count,
            slba: lba,
        }
    }
}

/// SMART / Health Information log (LID 0x02, 512 bytes, spec Figure).
/// Only the fields the driver reads are named; offsets are asserted.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SmartLog {
    /// Critical warning bits.
    pub critical_warning: u8,
    /// Composite temperature, Kelvin — **unaligned** at offset 1 (the
    /// spec's layout), so byte-array + [`temp_kelvin`](Self::temp_kelvin).
    pub composite_temp: [u8; 2],
    /// Available spare (%).
    pub avail_spare: u8,
    /// Available spare threshold (%).
    pub spare_threshold: u8,
    /// Percentage of rated device life used.
    pub percentage_used: u8,
    _pad6: [u8; 26],
    /// Data units read (thousands of 512-byte units, 128-bit LE).
    pub data_units_read: [u8; 16],
    /// Data units written.
    pub data_units_written: [u8; 16],
    /// Host read/write commands + controller busy time (unread here).
    _pad64: [u8; 48],
    /// Power cycles (128-bit LE).
    pub power_cycles: [u8; 16],
    /// Power-on hours (128-bit LE).
    pub power_on_hours: [u8; 16],
    /// Unsafe shutdowns (128-bit LE).
    pub unsafe_shutdowns: [u8; 16],
    /// Media and data-integrity errors (128-bit LE).
    pub media_errors: [u8; 16],
    _tail: [u8; 336],
}

const _: () = assert!(mem::size_of::<SmartLog>() == 512);
const _: () = assert!(mem::offset_of!(SmartLog, composite_temp) == 1);
const _: () = assert!(mem::offset_of!(SmartLog, percentage_used) == 5);
const _: () = assert!(mem::offset_of!(SmartLog, data_units_read) == 32);
const _: () = assert!(mem::offset_of!(SmartLog, power_cycles) == 112);

impl SmartLog {
    /// Composite temperature, Kelvin.
    pub fn temp_kelvin(&self) -> u16 {
        u16::from_le_bytes(self.composite_temp)
    }

    /// Composite temperature in °C (the log stores Kelvin).
    pub fn temp_celsius(&self) -> i32 {
        self.temp_kelvin() as i32 - 273
    }

    /// The low 64 bits of a 128-bit LE counter field.
    pub fn counter(field: &[u8; 16]) -> u64 {
        u64::from_le_bytes(field[..8].try_into().unwrap())
    }
}

/// A 16-byte NVMe completion queue entry.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NvmeCqe {
    /// Command-specific result (DW0).
    pub result: u32,
    pub rsvd: u32,
    /// SQ head pointer.
    pub sq_head: u16,
    /// SQ identifier.
    pub sq_id: u16,
    /// Command identifier.
    pub cid: u16,
    /// Status field (P, SC, SCT, …).
    pub status: u16,
}

const _: () = assert!(mem::size_of::<NvmeCqe>() == 16);

impl NvmeCqe {
    /// The phase (P) bit — toggles each time the CQ wraps.
    #[inline]
    pub fn phase(&self) -> bool {
        self.status & 1 != 0
    }

    /// Whether the command succeeded (SC = SCT = 0).
    #[inline]
    pub fn is_success(&self) -> bool {
        (self.status >> 1) & 0x3FFF == 0
    }

    /// Status Code.
    #[inline]
    pub fn status_code(&self) -> u8 {
        ((self.status >> 1) & 0xFF) as u8
    }

    /// Status Code Type.
    #[inline]
    pub fn status_code_type(&self) -> u8 {
        ((self.status >> 9) & 0x7) as u8
    }
}

/// Identify Controller data (4096 B, spec Figure).  Only the fields the
/// core reads are named; the rest is a single reserved tail, keeping the
/// struct exact (the size assert guards the layout).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IdentifyController {
    /// PCI vendor id.
    pub vid: u16,
    /// PCI subsystem vendor id.
    pub ssvid: u16,
    /// Serial number.
    pub sn: [u8; 20],
    /// Model number.
    pub mn: [u8; 40],
    /// Firmware revision.
    pub fr: [u8; 8],
    /// rab + ieee[3] + cmic — 5 bytes to `mdts` at offset 77.
    _pad_rab_to_mdts: [u8; 5],
    /// Maximum Data Transfer Size (`2^mdts` pages; 0 = unlimited).
    pub mdts: u8,
    /// cntlid..hmpre — to `hmpre` at offset 272.
    _pad_mdts_to_hmpre: [u8; 194],
    /// Host Memory Buffer Preferred Size, in **4 KiB units** (0 = the
    /// controller does not use a host memory buffer).  A DRAM-less
    /// controller keeps its mapping tables here; without one it pages them
    /// from NAND on every access.
    pub hmpre: u32,
    /// Host Memory Buffer Minimum Size, in 4 KiB units: a smaller buffer
    /// may be refused.
    pub hmmin: u32,
    /// tnvmcap..nn — to `nn` at offset 516.
    _pad_hmmin_to_nn: [u8; 236],
    /// Number of Namespaces.
    pub nn: u32,
    /// Optional NVM Command Support.
    pub oncs: u16,
    /// fuses..hmminds — to `hmminds` at offset 528.
    _pad_oncs_to_hmminds: [u8; 6],
    /// Host Memory Buffer Minimum Descriptor Entry Size, in 4 KiB units
    /// (0 = no constraint): every descriptor must cover at least this.
    pub hmminds: u32,
    /// Host Memory Maximum Descriptor Entries (0 = no limit).
    pub hmmaxd: u16,
    _tail: [u8; 3562],
}

const _: () = assert!(mem::size_of::<IdentifyController>() == 4096);
// Pin the offsets the padding-collapsed fields depend on (spec Figure).
const _: () = assert!(mem::offset_of!(IdentifyController, mdts) == 77);
const _: () = assert!(mem::offset_of!(IdentifyController, hmpre) == 272);
const _: () = assert!(mem::offset_of!(IdentifyController, hmmin) == 276);
const _: () = assert!(mem::offset_of!(IdentifyController, nn) == 516);
const _: () = assert!(mem::offset_of!(IdentifyController, oncs) == 520);
const _: () = assert!(mem::offset_of!(IdentifyController, hmminds) == 528);
const _: () = assert!(mem::offset_of!(IdentifyController, hmmaxd) == 532);

/// One LBA format descriptor.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LbaFormat {
    /// Metadata size.
    pub ms: u16,
    /// LBA data size as a power of two (`2^lbads` bytes).
    pub lbads: u8,
    /// Relative performance.
    pub rp: u8,
}

const _: () = assert!(mem::size_of::<LbaFormat>() == 4);

/// Identify Namespace data (4096 B, spec Figure).  As above, only the
/// read fields are named.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IdentifyNamespace {
    /// Namespace size, in logical blocks.
    pub nsze: u64,
    /// Namespace capacity, in logical blocks.
    pub ncap: u64,
    /// Namespace utilization.
    pub nuse: u64,
    /// Namespace features.
    pub nsfeat: u8,
    /// Number of LBA formats (0-based).
    pub nlbaf: u8,
    /// Formatted LBA Size. Bits 3:0 are the low format-index bits and
    /// bits 6:5 are the high format-index bits; bit 4 selects separate vs.
    /// extended metadata and is not part of the index.
    pub flbas: u8,
    /// Metadata Capabilities.
    pub mc: u8,
    /// End-to-end Data Protection Capabilities.
    pub dpc: u8,
    /// End-to-end Data Protection Type Settings.
    pub dps: u8,
    /// Namespace Multi-path I/O and Namespace Sharing Capabilities.
    pub nmic: u8,
    /// Reservation Capabilities.
    pub rescap: u8,
    /// Format Progress Indicator.
    pub fpi: u8,
    /// Deallocate Logical Block Features.
    pub dlfeat: u8,
    /// Namespace Atomic Write Unit Normal (0-based).
    pub nawun: u16,
    /// Namespace Atomic Write Unit Power Fail (0-based).
    pub nawupf: u16,
    /// Namespace Atomic Compare & Write Unit (0-based).
    pub nacwu: u16,
    /// Namespace Atomic Boundary Size Normal (0-based).
    pub nabsn: u16,
    /// Namespace Atomic Boundary Offset.
    pub nabo: u16,
    /// Namespace Atomic Boundary Size Power Fail (0-based).
    pub nabspf: u16,
    /// Namespace Optimal I/O Boundary, in logical blocks (0 = none).
    pub noiob: u16,
    /// Namespace NVM Capacity.
    pub nvmcap: [u8; 16],
    /// Namespace Preferred Write Granularity (0-based when supported).
    pub npwg: u16,
    /// Namespace Preferred Write Alignment (0-based when supported).
    pub npwa: u16,
    /// Namespace Preferred Deallocate Granularity (0-based when supported).
    pub npdg: u16,
    /// Namespace Preferred Deallocate Alignment (0-based when supported).
    pub npda: u16,
    /// Namespace Optimal Write Size (0-based when supported).
    pub nows: u16,
    /// mssrl..lbaf — to the first LBA format descriptor at offset 128.
    _pad_mssrl_to_lbaf: [u8; 54],
    /// LBA format descriptors.
    pub lbaf: [LbaFormat; 64],
    _tail: [u8; 3712],
}

const _: () = assert!(mem::size_of::<IdentifyNamespace>() == 4096);
const _: () = assert!(mem::offset_of!(IdentifyNamespace, flbas) == 26);
const _: () = assert!(mem::offset_of!(IdentifyNamespace, nawun) == 34);
const _: () = assert!(mem::offset_of!(IdentifyNamespace, noiob) == 46);
const _: () = assert!(mem::offset_of!(IdentifyNamespace, npwg) == 64);
const _: () = assert!(mem::offset_of!(IdentifyNamespace, nows) == 72);
const _: () = assert!(mem::offset_of!(IdentifyNamespace, lbaf) == 128);

impl IdentifyNamespace {
    /// The active LBA format selected by the split six-bit FLBAS index.
    /// Device-controlled indices outside both `NLBAF` and the descriptor
    /// table refuse instead of indexing an unrelated descriptor.
    pub fn active_lba_format(&self) -> Option<&LbaFormat> {
        let high = if self.nlbaf > 16 {
            (self.flbas >> 1) & 0x30
        } else {
            0
        };
        let index = ((self.flbas & 0x0f) | high) as usize;
        (index <= self.nlbaf as usize)
            .then(|| self.lbaf.get(index))
            .flatten()
    }

    /// Metadata bytes attached to each active logical block. The current
    /// data path carries no metadata buffer, so callers must refuse a
    /// non-zero result rather than issue a short transfer.
    pub fn metadata_size(&self) -> u16 {
        self.active_lba_format().map_or(0, |format| format.ms)
    }

    /// Logical block size in bytes — `0` when the descriptor's LBADS
    /// cannot encode a `u32` size (a shift past the width).  Identify
    /// data is device-controlled input: a hostile or broken descriptor
    /// must decode to a sentinel the caller validates, never to a panic
    /// (`kore_nvme::swap::blocks_per_page` is that validation for the
    /// swap backend; `0` fails it like every other unusable size).
    pub fn block_size(&self) -> u32 {
        self.active_lba_format()
            .filter(|format| format.lbads >= 9)
            .and_then(|format| 1u32.checked_shl(format.lbads as u32))
            .unwrap_or(0)
    }

    /// Performance fields the generic block seam can normalize. NVMe
    /// `NPW*`/`NOWS` fields are 0-based and defined only when NSFEAT.OPTPERF
    /// is set. `NOIOB` is an independent 1-based block count, with zero
    /// meaning that the namespace reports no optimal boundary.
    /// Every LBA data size this namespace can be formatted to, as a mask:
    /// bit `k` set means `512 << k` bytes is offered by a format with no
    /// metadata (the only kind this core can drive).  Reads the whole
    /// descriptor table, not just the active entry, so an operator can see
    /// that a 512-byte namespace *could* be a 4 KiB one before deciding on
    /// the destructive Format that would make it so.
    pub fn supported_block_sizes(&self) -> u32 {
        let count = (usize::from(self.nlbaf) + 1).min(self.lbaf.len());
        self.lbaf[..count]
            .iter()
            .filter(|format| format.ms == 0 && (9..=9 + 31).contains(&format.lbads))
            .fold(0u32, |mask, format| mask | (1u32 << (format.lbads - 9)))
    }

    pub fn io_performance(&self) -> (u32, u32, u32, u32) {
        let (granularity, alignment, optimal) = if self.nsfeat & (1 << 4) != 0 {
            (
                u32::from(self.npwg) + 1,
                u32::from(self.npwa) + 1,
                // A zero NOWS decodes to one LBA but carries no useful bulk
                // sizing advice. QEMU advertises this with OPTPERF set;
                // propagating it caps every filesystem write at one sector.
                // Match Linux's NVMe normalization: retain real hints, but
                // report this degenerate hint as absent. MDTS is independent.
                if self.nows == 0 {
                    0
                } else {
                    u32::from(self.nows) + 1
                },
            )
        } else {
            (0, 0, 0)
        };
        (granularity, alignment, optimal, u32::from(self.noiob))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqe_cid_and_opcode_pack_into_cdw0() {
        let prp = Prp {
            prp1: IovaAddr(0xdead_0000),
            prp2: IovaAddr(0),
        };
        let mut c = NvmeSqe::read(1, 0x1000, 7, prp);
        assert_eq!(c.opcode(), IoOpcode::Read as u8);
        c.set_cid(0xABCD);
        assert_eq!(c.cid(), 0xABCD);
        assert_eq!(
            c.opcode(),
            IoOpcode::Read as u8,
            "cid does not clobber opcode"
        );
        assert_eq!(c.cdw12, 7, "NLB is 0-based (7 ⇒ 8 blocks)");
        assert_eq!(c.cdw10, 0x1000, "LBA low");
    }

    #[test]
    fn cqe_status_decoding() {
        // Success, phase set.
        let ok = NvmeCqe {
            result: 0,
            rsvd: 0,
            sq_head: 1,
            sq_id: 0,
            cid: 5,
            status: 0x0001,
        };
        assert!(ok.phase());
        assert!(ok.is_success());

        // Status Code 0x02 (SCT 0), phase clear ⇒ failure.
        let err = NvmeCqe {
            status: 0x0004,
            ..ok
        };
        assert!(!err.phase());
        assert!(!err.is_success());
        assert_eq!(err.status_code(), 0x02);
    }

    #[test]
    fn create_cq_encodes_qid_size_vector_and_ien() {
        let c = NvmeSqe::create_io_cq(1, 64, IovaAddr(0xcafe_0000), 3, true);
        assert_eq!(c.opcode(), AdminOpcode::CreateIoCq as u8);
        assert_eq!(c.prp1, 0xcafe_0000);
        assert_eq!(c.cdw10, (63 << 16) | 1, "QSIZE 0-based, QID 1");
        assert_eq!(c.cdw11, (3 << 16) | 2 | 1, "IV=3, IEN, PC");
    }

    #[test]
    fn namespace_block_size_reads_active_format() {
        let mut ns: IdentifyNamespace = unsafe { core::mem::zeroed() };
        ns.nsze = 0x10_0000;
        ns.nlbaf = 1;
        ns.flbas = 1; // select lbaf[1]
        ns.lbaf[1].lbads = 12; // 4096-byte blocks
        assert_eq!(ns.block_size(), 4096);
        assert_eq!(ns.nsze, 0x10_0000);
    }

    #[test]
    fn namespace_format_index_uses_both_flbas_fragments() {
        let mut ns: IdentifyNamespace = unsafe { core::mem::zeroed() };
        ns.nlbaf = 63;
        // Index 0b10_0011 (35): high bits in FLBAS 6:5, low bits in 3:0.
        ns.flbas = (0b10 << 5) | 0b0011;
        ns.lbaf[35].lbads = 13;
        assert_eq!(ns.block_size(), 8192);
    }

    #[test]
    fn legacy_format_count_ignores_reserved_flbas_high_bits() {
        let mut ns: IdentifyNamespace = unsafe { core::mem::zeroed() };
        ns.nlbaf = 1;
        ns.flbas = (0b11 << 5) | 1;
        ns.lbaf[1].lbads = 12;
        assert_eq!(ns.block_size(), 4096);
    }

    #[test]
    fn malformed_namespace_geometry_refuses_without_panicking() {
        let mut ns: IdentifyNamespace = unsafe { core::mem::zeroed() };
        ns.nlbaf = 0;
        ns.flbas = 1;
        ns.lbaf[1].lbads = 12;
        assert_eq!(ns.block_size(), 0, "FLBAS may not exceed NLBAF");

        ns.flbas = 0;
        ns.lbaf[0].lbads = 8;
        assert_eq!(ns.block_size(), 0, "NVMe LBA data is at least 512 bytes");
        ns.lbaf[0].lbads = u8::MAX;
        assert_eq!(ns.block_size(), 0, "hostile shifts remain checked");
    }

    #[test]
    fn supported_block_sizes_read_the_whole_descriptor_table() {
        let mut ns: IdentifyNamespace = unsafe { core::mem::zeroed() };
        ns.nlbaf = 2; // three formats
        ns.lbaf[0].lbads = 9; // 512 B
        ns.lbaf[1].lbads = 12; // 4 KiB
        ns.lbaf[2].lbads = 12; // 4 KiB + metadata: not drivable here
        ns.lbaf[2].ms = 8;
        ns.lbaf[3].lbads = 13; // past NLBAF: not offered
        assert_eq!(ns.supported_block_sizes(), (1 << 0) | (1 << 3));
        ns.lbaf[0].lbads = 8; // sub-512 is not an NVMe LBA
        assert_eq!(ns.supported_block_sizes(), 1 << 3);
        // A hostile count clamps to the table rather than indexing past it,
        // so the fourth descriptor now legitimately counts.
        ns.nlbaf = u8::MAX;
        assert_eq!(ns.supported_block_sizes(), (1 << 3) | (1 << 4));
    }

    #[test]
    fn io_performance_normalizes_zero_based_fields() {
        let mut ns: IdentifyNamespace = unsafe { core::mem::zeroed() };
        ns.nsfeat = 1 << 4;
        ns.npwg = 7;
        ns.npwa = 3;
        ns.nows = 31;
        ns.noiob = 256;
        assert_eq!(ns.io_performance(), (8, 4, 32, 256));

        ns.nsfeat = 0;
        assert_eq!(ns.io_performance(), (0, 0, 0, 256));
        ns.nsfeat = 1 << 5; // reserved, not OPTPERF
        assert_eq!(ns.io_performance(), (0, 0, 0, 256));
    }

    #[test]
    fn single_lba_optimal_hint_does_not_throttle_bulk_transfers() {
        let mut ns: IdentifyNamespace = unsafe { core::mem::zeroed() };
        ns.nsfeat = 1 << 4;
        // QEMU's default: valid optimization fields, all zero-based fields
        // zero. Keep its one-block granularity/alignment, not a transfer cap.
        assert_eq!(ns.io_performance(), (1, 1, 0, 0));

        // Suppressing the degenerate size must neither suppress the other
        // hints nor wrap the largest valid zero-based optimal size.
        ns.npwg = 7;
        ns.npwa = 3;
        ns.noiob = 256;
        assert_eq!(ns.io_performance(), (8, 4, 0, 256));
        ns.nows = 1;
        assert_eq!(ns.io_performance(), (8, 4, 2, 256));
        ns.nows = u16::MAX;
        assert_eq!(ns.io_performance(), (8, 4, 65_536, 256));
    }
}
