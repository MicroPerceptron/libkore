//! `MockNvmeDevice` — a host-memory NVMe controller, the driver core's
//! test backend (the D2 `virtio-accel-mock` lesson: prove the semantics on
//! the host, prove the *seam* on hardware).  It implements [`Regs`] and
//! hands out a [`DmaPool`]; a doorbell write drives its command engine
//! (read the new SQEs, process, publish CQEs with the phase bit), so the
//! **entire core** — bring-up, identify, qpair create, read — runs with no
//! hardware.
//!
//! The trick that makes it simple: a mock IOVA **is** the host pointer
//! (there is no IOMMU to translate), so the device reads SQEs / writes
//! CQEs / fills identify buffers by casting the IOVAs the driver
//! programmed straight back to pointers — the same frames the driver's CPU
//! window sees.

use alloc::alloc::{Layout, alloc_zeroed};
use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::{Cell, RefCell};

use kore_abi::device::IovaAddr;

use crate::commands::{
    AdminOpcode, HmbDescriptor, IdentifyController, IdentifyNamespace, NvmeCqe, NvmeSqe,
};
use crate::regs::{
    CC_EN, CC_SHN_MASK, CSTS_RDY, CSTS_SHST_COMPLETE, CSTS_SHST_MASK, REG_ACQ, REG_AQA, REG_ASQ,
    REG_CAP, REG_CC, REG_CSTS, REG_DOORBELL_BASE, REG_VS,
};
use crate::seam::{Dma, DmaPool};

/// The mock's advertised `CAP`: MQES 0x3F (64 entries), CQR (bit 16),
/// DSTRD 0 — enough for a 32-deep admin queue and 4-byte doorbell stride.
const MOCK_CAP: u64 = 0x0000_0000_0001_003F;
/// The mock's advertised version (NVMe 1.4.0).
const MOCK_VS: u32 = 0x0001_0400;
/// The mock namespace's logical block size.
pub const MOCK_BLOCK_SIZE: u32 = 4096;
/// The mock namespace's block count.
pub const MOCK_BLOCK_COUNT: u64 = 0x10_0000; // 4 GiB at 4 KiB blocks
/// Blocks the backing disk actually holds (I/O past it fails with LBA
/// Out of Range — deliberately smaller than the advertised namespace).
pub const MOCK_DISK_BLOCKS: u32 = 64;

// ── The DMA pool: page-aligned host memory, iova == cpu pointer ──────────

/// A mock DMA buffer over leaked, page-aligned host memory.
pub struct MockBuf {
    ptr: *mut u8,
    len: usize,
}

impl Dma for MockBuf {
    fn cpu(&self) -> *mut u8 {
        self.ptr
    }
    fn iova(&self) -> IovaAddr {
        // The mock's device "domain" is host memory: iova == pointer.
        IovaAddr(self.ptr as u64)
    }
    fn len(&self) -> usize {
        self.len
    }
}

/// Hands out page-aligned zeroed host buffers (leaked — valid for the test
/// process, so a returned `MockBuf` never dangles).
#[derive(Default)]
pub struct MockPool;

impl DmaPool for MockPool {
    type Buf = MockBuf;
    fn alloc(&self, bytes: usize) -> Option<MockBuf> {
        self.alloc_aligned(bytes, 4096)
    }

    fn alloc_aligned(&self, bytes: usize, alignment: usize) -> Option<MockBuf> {
        let alignment = alignment.max(4096);
        if !alignment.is_power_of_two() {
            return None;
        }
        let len = bytes.next_multiple_of(4096).max(4096);
        let layout = Layout::from_size_align(len, alignment).ok()?;
        // SAFETY: non-zero, well-formed layout; leaked for the process.
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return None;
        }
        Some(MockBuf { ptr, len })
    }
}

// ── The device: register file + command engine ──────────────────────────

/// A created I/O queue the device knows about — its ring IOVAs, depth,
/// and the device-side head/tail/phase the I/O completion path advances.
#[derive(Clone, Copy, Default)]
pub(crate) struct IoQueueState {
    pub(crate) sq_iova: u64,
    pub(crate) cq_iova: u64,
    pub(crate) depth: u16,
    pub(crate) sq_head: u16,
    pub(crate) cq_tail: u16,
    pub(crate) cq_phase: bool,
}

/// The mock controller's mutable state.
pub(crate) struct DeviceState {
    cap: u64,
    cc: u32,
    csts: u32,
    admin_depth: u16,
    asq_iova: u64,
    acq_iova: u64,
    admin_sq_head: u16,
    admin_cq_tail: u16,
    admin_cq_phase: bool,
    pub(crate) io_queues: Vec<IoQueueState>,
    /// I/O SQ-tail MMIO writes (admin and CQ doorbells excluded).
    io_sq_doorbells: u64,
    /// I/O CQ-head MMIO writes.
    io_cq_doorbells: u64,
    /// A backing "disk" the Read/Write path moves bytes to/from.
    pub(crate) disk: Box<[u8]>,
    /// Deallocated ranges (test probe for the DSM path): `(slba, nlb)`.
    pub(crate) deallocated: Vec<(u64, u32)>,
    /// The LBA format Identify Namespace advertises (`2^lbads`-byte
    /// blocks) — a test knob; the I/O engine itself always moves
    /// [`MOCK_BLOCK_SIZE`] blocks.
    lbads: u8,
    /// I/O completions to withhold (test knob): the command executes, its
    /// CQE waits for [`MockNvmeDevice::release_swallowed`] — a device
    /// whose completion arrives after the driver stopped waiting.
    swallow: u32,
    /// The withheld completions: `(queue index, cid, status code)`.
    held: Vec<(usize, u16, u8)>,
    /// Host Memory Buffer facts Identify advertises (test knob), in the
    /// spec's 4 KiB units: preferred, minimum, and the descriptor ceiling.
    hmpre: u32,
    hmmin: u32,
    hmmaxd: u16,
    /// The buffer the driver enabled: `(HSIZE in pages, descriptor count)`.
    hmb: Option<(u32, u32)>,
    /// Set Features 0x0D disables accepted.
    hmb_disables: u32,
}

impl DeviceState {
    fn new() -> Self {
        // A small backing disk, LBA-addressed, seeded so reads are checkable.
        let mut disk = alloc::vec![0u8; MOCK_DISK_BLOCKS as usize * MOCK_BLOCK_SIZE as usize]
            .into_boxed_slice();
        for (i, b) in disk.iter_mut().enumerate() {
            *b = (i as u32).wrapping_mul(2654435761).to_le_bytes()[0];
        }
        Self {
            cap: MOCK_CAP,
            cc: 0,
            csts: 0,
            admin_depth: 0,
            asq_iova: 0,
            acq_iova: 0,
            admin_sq_head: 0,
            admin_cq_tail: 0,
            admin_cq_phase: true,
            io_queues: Vec::new(),
            io_sq_doorbells: 0,
            io_cq_doorbells: 0,
            disk,
            deallocated: Vec::new(),
            lbads: MOCK_BLOCK_SIZE.trailing_zeros() as u8,
            swallow: 0,
            held: Vec::new(),
            hmpre: 0,
            hmmin: 0,
            hmmaxd: 0,
            hmb: None,
            hmb_disables: 0,
        }
    }

    /// Publish one admin completion (status is the raw SC<<1; the phase
    /// bit is OR'd in).
    fn complete_admin(&mut self, cid: u16, sc: u8, result: u32) {
        let cqe = NvmeCqe {
            result,
            rsvd: 0,
            sq_head: self.admin_sq_head,
            sq_id: 0,
            cid,
            status: ((sc as u16) << 1) | (self.admin_cq_phase as u16),
        };
        // SAFETY: acq_iova is a host pointer to `admin_depth * 16` bytes;
        // cq_tail < depth.
        unsafe {
            let slot = (self.acq_iova as *mut NvmeCqe).add(self.admin_cq_tail as usize);
            slot.write_volatile(cqe);
        }
        self.admin_cq_tail = (self.admin_cq_tail + 1) % self.admin_depth;
        if self.admin_cq_tail == 0 {
            self.admin_cq_phase = !self.admin_cq_phase;
        }
    }

    /// Process the admin SQ up to `new_tail` (called on the admin SQ
    /// doorbell write).
    fn process_admin(&mut self, new_tail: u16) {
        while self.admin_sq_head != new_tail {
            // SAFETY: asq_iova is a host pointer to `admin_depth * 64`
            // bytes; sq_head < depth.
            let sqe: NvmeSqe = unsafe {
                (self.asq_iova as *const NvmeSqe)
                    .add(self.admin_sq_head as usize)
                    .read_volatile()
            };
            let (sc, result) = self.dispatch_admin(&sqe);
            self.complete_admin(sqe.cid(), sc, result);
            self.admin_sq_head = (self.admin_sq_head + 1) % self.admin_depth;
        }
    }

    /// Execute one admin command, returning `(status_code, result_dw0)`.
    fn dispatch_admin(&mut self, sqe: &NvmeSqe) -> (u8, u32) {
        match sqe.opcode() {
            x if x == AdminOpcode::Identify as u8 => {
                match sqe.cdw10 & 0xFF {
                    1 => self.write_identify_controller(sqe.prp1),
                    0 => self.write_identify_namespace(sqe.prp1),
                    _ => return (0x02, 0), // Invalid Field
                }
                (0, 0)
            }
            x if x == AdminOpcode::GetLogPage as u8 => {
                // SMART/Health (LID 0x02): fixed, checkable values.
                if sqe.cdw10 & 0xFF == 0x02 {
                    // SAFETY: prp1 is a host pointer to a ≥512-byte buffer.
                    let log = unsafe { &mut *(sqe.prp1 as *mut crate::commands::SmartLog) };
                    *log = unsafe { core::mem::zeroed() };
                    log.composite_temp = (273u16 + 36).to_le_bytes(); // 36 °C
                    log.percentage_used = 3;
                    log.data_units_read[..8].copy_from_slice(&1234u64.to_le_bytes());
                    log.power_cycles[..8].copy_from_slice(&7u64.to_le_bytes());
                    (0, 0)
                } else {
                    (0x02, 0) // Invalid Field
                }
            }
            x if x == AdminOpcode::SetFeatures as u8 => {
                match sqe.cdw10 & 0xFF {
                    // Number of Queues: echo the request back (allocated == asked).
                    0x07 => (0, sqe.cdw11),
                    // Host Memory Buffer: check the shape the spec fixes, then
                    // record what the driver handed over.
                    0x0D => {
                        if sqe.cdw11 & 1 == 0 {
                            self.hmb = None;
                            self.hmb_disables += 1;
                            (0, 0)
                        } else {
                            let list = u64::from(sqe.cdw13) | (u64::from(sqe.cdw14) << 32);
                            let entries = sqe.cdw15;
                            let too_many = self.hmmaxd != 0 && entries > u32::from(self.hmmaxd);
                            if list & 0xF != 0 || entries == 0 || too_many {
                                (0x02, 0) // Invalid Field
                            } else {
                                // Sum the table the way a controller would: MPS
                                // pages per descriptor, at the host pointer the
                                // mock's IOVAs are.
                                let mps = 1u64 << (12 + ((self.cc >> 7) & 0xF));
                                let mut pages = 0u64;
                                for i in 0..entries as usize {
                                    // SAFETY: the driver handed a host pointer to
                                    // `entries` descriptors it owns.
                                    let d = unsafe { &*((list as *const HmbDescriptor).add(i)) };
                                    if d.badd % mps != 0 || d.bsize == 0 {
                                        pages = u64::MAX;
                                        break;
                                    }
                                    pages += u64::from(d.bsize);
                                }
                                let hmmin_pages = (u64::from(self.hmmin) << 12) / mps;
                                if pages == u64::MAX
                                    || pages != u64::from(sqe.cdw12)
                                    || pages < hmmin_pages
                                {
                                    (0x02, 0)
                                } else {
                                    self.hmb = Some((sqe.cdw12, entries));
                                    (0, 0)
                                }
                            }
                        }
                    }
                    _ => (0, 0),
                }
            }
            x if x == AdminOpcode::CreateIoCq as u8 => {
                let qid = (sqe.cdw10 & 0xFFFF) as u16;
                let depth = ((sqe.cdw10 >> 16) & 0xFFFF) as u16 + 1;
                self.ensure_io_queue(qid).cq_iova = sqe.prp1;
                self.ensure_io_queue(qid).depth = depth;
                (0, 0)
            }
            x if x == AdminOpcode::CreateIoSq as u8 => {
                let qid = (sqe.cdw10 & 0xFFFF) as u16;
                self.ensure_io_queue(qid).sq_iova = sqe.prp1;
                (0, 0)
            }
            x if x == AdminOpcode::DeleteIoSq as u8 || x == AdminOpcode::DeleteIoCq as u8 => (0, 0),
            _ => (0x01, 0), // Invalid Opcode
        }
    }

    /// Process the I/O SQ of queue `qid` up to `new_tail` (called on its SQ
    /// doorbell write): read each command, move disk bytes per its PRP,
    /// and publish a completion on its CQ (or hold it back while the
    /// swallow knob is armed — the work is done either way).
    fn process_io(&mut self, qid: u16, new_tail: u16) {
        let idx = qid as usize - 1;
        if idx >= self.io_queues.len() {
            return;
        }
        loop {
            let q = self.io_queues[idx];
            if q.depth == 0 || q.sq_head == new_tail {
                break;
            }
            // SAFETY: sq_iova is a host pointer to `depth * 64` bytes;
            // sq_head < depth.
            let sqe: NvmeSqe = unsafe {
                (q.sq_iova as *const NvmeSqe)
                    .add(q.sq_head as usize)
                    .read_volatile()
            };
            let sc = self.dispatch_io(&sqe);

            // The command is consumed whether or not its CQE is withheld.
            let e = &mut self.io_queues[idx];
            e.sq_head = (e.sq_head + 1) % e.depth;

            if self.swallow > 0 {
                self.swallow -= 1;
                self.held.push((idx, sqe.cid(), sc));
            } else {
                self.publish_io_cqe(idx, sqe.cid(), sc);
            }
        }
    }

    /// Publish one I/O completion on queue `idx`'s CQ.
    fn publish_io_cqe(&mut self, idx: usize, cid: u16, sc: u8) {
        let q = self.io_queues[idx];
        let cqe = NvmeCqe {
            result: 0,
            rsvd: 0,
            sq_head: q.sq_head,
            sq_id: idx as u16 + 1,
            cid,
            status: ((sc as u16) << 1) | (q.cq_phase as u16),
        };
        // SAFETY: cq_iova is a host pointer to `depth * 16` bytes;
        // cq_tail < depth.
        unsafe {
            (q.cq_iova as *mut NvmeCqe)
                .add(q.cq_tail as usize)
                .write_volatile(cqe);
        }
        let e = &mut self.io_queues[idx];
        e.cq_tail = (e.cq_tail + 1) % e.depth;
        if e.cq_tail == 0 {
            e.cq_phase = !e.cq_phase;
        }
    }

    /// Execute one I/O command, returning its status code.  Read/Write
    /// honor the command's **PRP** exactly (the driver's `prp::build`
    /// output is validated here), moving `blocks` from `disk[lba..]`.
    /// Read/Write geometry from CDW10..12 — **valid only for those
    /// opcodes** (CDW meanings are per-command: DSM's CDW10 is a range
    /// count and CDW11 its attributes — decoding those as an LBA once
    /// produced a phantom out-of-range refusal here).
    fn rw_geometry(&self, sqe: &NvmeSqe) -> Option<(usize, usize)> {
        let lba = sqe.cdw10 as u64 | ((sqe.cdw11 as u64) << 32);
        let blocks = (sqe.cdw12 & 0xFFFF) + 1;
        let size = blocks as usize * MOCK_BLOCK_SIZE as usize;
        let disk_off = (lba as usize).checked_mul(MOCK_BLOCK_SIZE as usize)?;
        (disk_off + size <= self.disk.len()).then_some((disk_off, size))
    }

    fn dispatch_io(&mut self, sqe: &NvmeSqe) -> u8 {
        use crate::commands::IoOpcode;
        match sqe.opcode() {
            x if x == IoOpcode::Read as u8 => {
                let Some((disk_off, size)) = self.rw_geometry(sqe) else {
                    return 0x80; // LBA Out of Range (NVM command set)
                };
                self.prp_walk(sqe.prp1, sqe.prp2, size, |page, poff, n| {
                    // disk → device page.
                    let src = &self.disk[disk_off + poff..disk_off + poff + n];
                    // SAFETY: `page` is a device (host) buffer span inside
                    // the configured NVMe memory page.
                    unsafe { core::ptr::copy_nonoverlapping(src.as_ptr(), page, n) };
                });
                0
            }
            x if x == IoOpcode::Write as u8 => {
                let Some((disk_off, size)) = self.rw_geometry(sqe) else {
                    return 0x80;
                };
                // Snapshot the pages first (borrow discipline), then apply.
                let mut staged = alloc::vec![0u8; size];
                self.prp_walk(sqe.prp1, sqe.prp2, size, |page, poff, n| {
                    // SAFETY: as above.
                    unsafe { core::ptr::copy_nonoverlapping(page, staged[poff..].as_mut_ptr(), n) };
                });
                self.disk[disk_off..disk_off + size].copy_from_slice(&staged);
                0
            }
            x if x == IoOpcode::Flush as u8 => 0,
            x if x == IoOpcode::DatasetManagement as u8 => {
                // Deallocate: read the (nr + 1)-entry DsmRange table via
                // PRP1 and record each range (zeroing the blocks — the
                // deterministic deallocate the spec's DLFEAT=001 offers).
                let nr = (sqe.cdw10 & 0xFF) as usize + 1;
                for i in 0..nr {
                    // SAFETY: prp1 is a host pointer to the range table.
                    let r: crate::commands::DsmRange = unsafe {
                        (sqe.prp1 as *const crate::commands::DsmRange)
                            .add(i)
                            .read_volatile()
                    };
                    let off = r.slba as usize * MOCK_BLOCK_SIZE as usize;
                    let len = r.nlb as usize * MOCK_BLOCK_SIZE as usize;
                    if off + len <= self.disk.len() {
                        self.disk[off..off + len].fill(0);
                    }
                    self.deallocated.push((r.slba, r.nlb));
                }
                0
            }
            _ => 0x01,
        }
    }

    /// Walk the PRP-described device pages of a `size`-byte transfer,
    /// calling `f(page_ptr, transfer_offset, page_bytes)` per page — the
    /// mock's reading of `prp1` / `prp2` (direct or list).  The driver's
    /// PRP1 may start at an offset within the configured memory page.
    fn prp_walk(
        &self,
        prp1: u64,
        prp2: u64,
        size: usize,
        mut f: impl FnMut(*mut u8, usize, usize),
    ) {
        let page = 1usize << (12 + ((self.cc >> 7) & 0x0f));
        let mut off = 0;
        // Page 0 via PRP1.
        let first_remainder = page - (prp1 as usize & (page - 1));
        let n0 = size.min(first_remainder);
        f(prp1 as *mut u8, 0, n0);
        off += n0;
        if off >= size {
            return;
        }
        let remaining_pages = (size - off).div_ceil(page);
        if remaining_pages == 1 {
            // PRP2 is the second page directly.
            f(prp2 as *mut u8, off, size - off);
            return;
        }
        // PRP2 points at a list of page addresses.
        for i in 0..remaining_pages {
            // SAFETY: prp2 is a host pointer to a list page of `>= remaining_pages` u64s.
            let address = unsafe { *(prp2 as *const u64).add(i) };
            let n = (size - off).min(page);
            f(address as *mut u8, off, n);
            off += n;
        }
    }

    fn ensure_io_queue(&mut self, qid: u16) -> &mut IoQueueState {
        let idx = qid as usize - 1;
        if idx >= self.io_queues.len() {
            self.io_queues.resize(
                idx + 1,
                IoQueueState {
                    cq_phase: true,
                    ..Default::default()
                },
            );
        }
        &mut self.io_queues[idx]
    }

    fn write_identify_controller(&self, prp1: u64) {
        // SAFETY: prp1 is a host pointer to the 4 KiB identify buffer.
        let idc = unsafe { &mut *(prp1 as *mut IdentifyController) };
        *idc = unsafe { core::mem::zeroed() };
        idc.vid = 0x1B36; // QEMU
        idc.mdts = 5; // 4096 << 5 = 128 KiB
        idc.nn = 1;
        // Identify Controller ONCS bit 2: Dataset Management supported.
        idc.oncs = 1 << 2;
        idc.hmpre = self.hmpre;
        idc.hmmin = self.hmmin;
        idc.hmmaxd = self.hmmaxd;
        idc.mn[..5].copy_from_slice(b"kmock");
    }

    fn write_identify_namespace(&self, prp1: u64) {
        // SAFETY: as above.
        let idn = unsafe { &mut *(prp1 as *mut IdentifyNamespace) };
        *idn = unsafe { core::mem::zeroed() };
        idn.nsze = MOCK_BLOCK_COUNT;
        idn.ncap = MOCK_BLOCK_COUNT;
        idn.flbas = 0; // lbaf[0]
        idn.lbaf[0].lbads = self.lbads; // 12 ⇒ 4096, unless the knob turned
    }
}

/// The mock controller — clone it to hand `Controller::bring_up` the
/// register seam while keeping a handle to inspect device state.
#[derive(Clone)]
pub struct MockNvmeDevice {
    state: Rc<RefCell<DeviceState>>,
}

impl MockNvmeDevice {
    /// A fresh mock controller (disabled, admin queue unprogrammed).
    pub fn new() -> Self {
        Self {
            state: Rc::new(RefCell::new(DeviceState::new())),
        }
    }

    /// Advertise one exact minimum/maximum NVMe memory-page encoding.
    pub fn with_mps(self, mps: u8) -> Option<Self> {
        if mps > 15 {
            return None;
        }
        let mut state = self.state.borrow_mut();
        state.cap &= !(0xff << 48);
        state.cap |= u64::from(mps) << 48;
        state.cap |= u64::from(mps) << 52;
        drop(state);
        Some(self)
    }

    /// A fresh DMA pool (independent of the device — buffers are plain
    /// host memory the device reads by IOVA).
    pub fn pool(&self) -> MockPool {
        MockPool
    }

    /// How many I/O queues the driver has created (test probe).
    pub fn io_queue_count(&self) -> usize {
        self.state.borrow().io_queues.len()
    }

    /// I/O submission doorbells observed by the mock controller.
    pub fn io_sq_doorbells(&self) -> u64 {
        self.state.borrow().io_sq_doorbells
    }

    /// I/O completion-head doorbells observed by the mock controller.
    pub fn io_cq_doorbells(&self) -> u64 {
        self.state.borrow().io_cq_doorbells
    }

    /// A copy of the backing disk at `lba` for `blocks` — the expected
    /// bytes a read returns (test oracle).
    pub fn disk_snapshot(&self, lba: u64, blocks: u32) -> Vec<u8> {
        let off = lba as usize * MOCK_BLOCK_SIZE as usize;
        let len = blocks as usize * MOCK_BLOCK_SIZE as usize;
        self.state.borrow().disk[off..off + len].to_vec()
    }

    /// Ranges the driver deallocated via DSM (test probe).
    pub fn deallocated(&self) -> Vec<(u64, u32)> {
        self.state.borrow().deallocated.clone()
    }

    /// Overwrite the backing disk at `lba` (test fixture writer — how a
    /// GPT lands on the mock disk for discovery tests).
    pub fn disk_write(&self, lba: u64, bytes: &[u8]) {
        let off = lba as usize * MOCK_BLOCK_SIZE as usize;
        self.state.borrow_mut().disk[off..off + bytes.len()].copy_from_slice(bytes);
    }

    /// Advertise `2^lbads`-byte blocks from the next Identify Namespace
    /// on (test knob — geometry-validation coverage; the I/O engine is
    /// untouched).
    pub fn set_lbads(&self, lbads: u8) {
        self.state.borrow_mut().lbads = lbads;
    }

    /// Advertise a Host Memory Buffer appetite from the next Identify
    /// Controller on, in the spec's 4 KiB units (test knob — a DRAM-less
    /// controller).
    pub fn set_hmb(&self, preferred_units: u32, min_units: u32, max_descriptors: u16) {
        let mut state = self.state.borrow_mut();
        state.hmpre = preferred_units;
        state.hmmin = min_units;
        state.hmmaxd = max_descriptors;
    }

    /// The host memory buffer the driver enabled — `(HSIZE in memory
    /// pages, descriptor count)` — or `None` (test probe).
    pub fn hmb(&self) -> Option<(u32, u32)> {
        self.state.borrow().hmb
    }

    /// Set Features 0x0D disables the controller accepted (test probe).
    pub fn hmb_disables(&self) -> u32 {
        self.state.borrow().hmb_disables
    }

    /// Withhold the next `n` I/O completions: each command executes, but
    /// its CQE arrives only at [`release_swallowed`](Self::release_swallowed)
    /// — the late-completing device the driver's timeout discipline must
    /// survive (test knob).
    pub fn swallow_completions(&self, n: u32) {
        self.state.borrow_mut().swallow += n;
    }

    /// Publish every withheld completion — the late arrivals land now.
    pub fn release_swallowed(&self) {
        let mut s = self.state.borrow_mut();
        let held: Vec<(usize, u16, u8)> = core::mem::take(&mut s.held);
        for (idx, cid, sc) in held {
            s.publish_io_cqe(idx, cid, sc);
        }
    }
}

impl Default for MockNvmeDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::seam::Regs for MockNvmeDevice {
    fn read32(&self, off: usize) -> u32 {
        let s = self.state.borrow();
        match off {
            REG_CAP => s.cap as u32,
            x if x == REG_CAP + 4 => (s.cap >> 32) as u32,
            REG_VS => MOCK_VS,
            REG_CC => s.cc,
            REG_CSTS => s.csts,
            _ => 0,
        }
    }

    fn write32(&self, off: usize, val: u32) {
        let mut s = self.state.borrow_mut();
        match off {
            REG_CC => {
                s.cc = val;
                if val & CC_EN != 0 {
                    s.csts |= CSTS_RDY;
                } else {
                    s.csts &= !CSTS_RDY;
                }
                // Shutdown notification: the mock has nothing to flush, so
                // any SHN value reports shutdown complete immediately (and
                // clearing SHN clears the report).
                if val & CC_SHN_MASK != 0 {
                    s.csts = (s.csts & !CSTS_SHST_MASK) | CSTS_SHST_COMPLETE;
                } else {
                    s.csts &= !CSTS_SHST_MASK;
                }
            }
            REG_AQA => {
                // Admin CQ size (0-based) in 27:16; SQ size in 15:0.
                s.admin_depth = ((val >> 16) & 0xFFF) as u16 + 1;
            }
            REG_ASQ => s.asq_iova = (s.asq_iova & !0xFFFF_FFFF) | val as u64,
            x if x == REG_ASQ + 4 => s.asq_iova = (s.asq_iova & 0xFFFF_FFFF) | ((val as u64) << 32),
            REG_ACQ => s.acq_iova = (s.acq_iova & !0xFFFF_FFFF) | val as u64,
            x if x == REG_ACQ + 4 => s.acq_iova = (s.acq_iova & 0xFFFF_FFFF) | ((val as u64) << 32),
            // Doorbells (0x1000+): decode (qid, SQ-vs-CQ) from the offset
            // at the mock's 4-byte stride (DSTRD 0).  An SQ-tail write
            // drives that queue's engine; a CQ-head write is informational
            // (the driver's bookkeeping — the mock needs nothing).
            off if off >= REG_DOORBELL_BASE => {
                let dbl = (off - REG_DOORBELL_BASE) / 4;
                let qid = (dbl / 2) as u16;
                let is_sq = dbl % 2 == 0;
                if is_sq {
                    if qid == 0 {
                        s.process_admin(val as u16);
                    } else {
                        s.io_sq_doorbells = s.io_sq_doorbells.saturating_add(1);
                        s.process_io(qid, val as u16);
                    }
                } else if qid != 0 {
                    s.io_cq_doorbells = s.io_cq_doorbells.saturating_add(1);
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Controller;

    /// The N1a thesis: the whole bring-up + identify sequence runs over a
    /// host mock — no hardware — and reads back the device's advertised
    /// geometry.
    #[test]
    fn bring_up_and_identify_over_the_mock() {
        let dev = MockNvmeDevice::new();
        let mut ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {})
            .expect("controller comes up over the mock");

        assert_eq!(ctrl.caps().max_queue_entries(), 64);
        assert_eq!(ctrl.info().version, MOCK_VS);

        let info = ctrl.identify(16, || {}).expect("identify");
        assert_eq!(info.num_namespaces, 1);
        assert_eq!(info.mdts_bytes, 4096 << 5, "MDTS = 128 KiB");
        assert_eq!(info.block_size, MOCK_BLOCK_SIZE);
        assert_eq!(info.block_count, MOCK_BLOCK_COUNT);
    }

    #[test]
    fn larger_memory_page_aligns_every_queue_and_scales_mdts() {
        let dev = MockNvmeDevice::new().with_mps(1).unwrap();
        let mut ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {})
            .expect("8 KiB controller comes up");
        assert_eq!(ctrl.page().bytes(), 8192);
        {
            let state = dev.state.borrow();
            assert_eq!(state.asq_iova & 0x1fff, 0);
            assert_eq!(state.acq_iova & 0x1fff, 0);
        }

        let info = ctrl.identify(16, || {}).expect("identify at 8 KiB MPS");
        assert_eq!(info.mdts_bytes, 8192 << 5);
        assert_eq!(ctrl.set_num_queues(1, 16, || {}), Ok((1, 1)));
        let qpair = ctrl
            .create_io_qpair(1, 8, None, 16, || {})
            .expect("I/O queues align to CC.MPS");
        assert_eq!(qpair.sq_iova().0 & 0x1fff, 0);
        assert_eq!(qpair.cq_iova().0 & 0x1fff, 0);
    }

    /// Preparation is the DMA-publication cut: it must stop a previously
    /// enabled controller before allocating either admin queue, program both
    /// queue pointers while halted, and leave `CC.EN` clear until `start`.
    #[test]
    fn prepare_programs_admin_queues_while_halted_then_start_enables() {
        struct HaltCheckedPool(MockNvmeDevice);

        impl DmaPool for HaltCheckedPool {
            type Buf = MockBuf;

            fn alloc(&self, bytes: usize) -> Option<Self::Buf> {
                use crate::seam::Regs as _;
                assert_eq!(
                    self.0.read32(REG_CC) & CC_EN,
                    0,
                    "CC.EN must be clear before allocating controller DMA"
                );
                assert_eq!(
                    self.0.read32(REG_CSTS) & CSTS_RDY,
                    0,
                    "the controller must acknowledge !RDY before allocation"
                );
                MockPool.alloc(bytes)
            }
        }

        use crate::seam::Regs as _;
        let dev = MockNvmeDevice::new();
        dev.write32(REG_CC, CC_EN);
        assert_ne!(dev.read32(REG_CSTS) & CSTS_RDY, 0, "fixture starts live");

        let prepared = Controller::prepare(dev.clone(), HaltCheckedPool(dev.clone()), 16, || {})
            .expect("prepare");

        {
            let state = dev.state.borrow();
            assert_eq!(state.cc & CC_EN, 0, "prepare does not enable");
            assert_eq!(state.csts & CSTS_RDY, 0, "prepare remains halted");
            assert_eq!(state.admin_depth, crate::controller::ADMIN_DEPTH);
            assert_ne!(state.asq_iova, 0, "admin SQ programmed while halted");
            assert_ne!(state.acq_iova, 0, "admin CQ programmed while halted");
        }

        let ctrl = prepared.start(16, || {}).expect("start");
        assert_ne!(dev.read32(REG_CC) & CC_EN, 0, "start sets CC.EN");
        assert_ne!(dev.read32(REG_CSTS) & CSTS_RDY, 0, "start observes RDY");
        assert_eq!(ctrl.info().version, MOCK_VS);
    }

    /// The quiesce path: an orderly shutdown notifies `SHN`, observes
    /// shutdown complete, disables, and observes `!RDY` — then the seam
    /// comes back out whole ([`Controller::into_seam`]), which is what the
    /// class binding's `unbind` restores into the common resource set.
    #[test]
    fn shutdown_completes_and_releases_the_seam() {
        let dev = MockNvmeDevice::new();
        let mut ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {})
            .expect("controller comes up over the mock");
        ctrl.shutdown(16, || {}).expect("orderly shutdown");
        use crate::seam::Regs as _;
        let csts = dev.read32(crate::regs::REG_CSTS);
        assert_eq!(csts & CSTS_RDY, 0, "disabled after shutdown");
        let (regs, _pool) = ctrl.into_seam();
        assert_eq!(
            regs.read32(crate::regs::REG_VS),
            MOCK_VS,
            "the released register seam still decodes"
        );
    }

    /// The N1b thesis: create an I/O qpair and **read a real LBA** over
    /// the mock, zero-copy — the bytes land in the caller's DMA buffer and
    /// match the device's backing disk, with the PRP the driver built
    /// honored by the device.
    #[test]
    fn create_io_qpair_and_read_an_lba() {
        let dev = MockNvmeDevice::new();
        let pool = dev.pool();
        let mut ctrl = Controller::bring_up(dev.clone(), pool, 16, || {}).unwrap();
        ctrl.identify(16, || {}).unwrap();
        let ns = ctrl.namespace();
        assert_eq!(ns.block_size, MOCK_BLOCK_SIZE);

        ctrl.set_num_queues(4, 16, || {}).unwrap();
        let mut qp = ctrl
            .create_io_qpair(1, 64, Some(0), 16, || {})
            .expect("I/O qpair");
        assert_eq!(dev.io_queue_count(), 1, "the device registered the queue");

        // Read one block (LBA 5) into a caller DMA buffer — single PRP.
        let buf = ctrl.pool().alloc(ns.block_size as usize).unwrap();
        let cmd = ns.read(5, 1, buf.iova(), None).unwrap();
        let cid = qp.submit_one(cmd, 0xF00D, ctrl.regs()).unwrap();

        // Poll to completion (the mock completed synchronously on the ring).
        let mut done = None;
        for _ in 0..8 {
            if let Some(c) = qp.poll(ctrl.regs()) {
                done = Some(c);
                break;
            }
        }
        let c = done.expect("completion arrives");
        assert_eq!(c.cid, cid);
        assert_eq!(c.cookie, 0xF00D, "cookie correlation");
        assert!(c.is_success());

        // The bytes are the device's disk at LBA 5 — zero-copy into `buf`.
        let disk = dev.disk_snapshot(5, 1);
        // SAFETY: `buf` is `block_size` bytes the device just wrote.
        let got = unsafe { core::slice::from_raw_parts(buf.cpu(), ns.block_size as usize) };
        assert_eq!(got, &disk[..], "read returned the backing bytes");
    }

    /// A two-block read spans two pages (PRP1 + PRP2 direct); a write
    /// round-trips through the same PRP path — proving `prp::build` against
    /// a real device consumer.
    #[test]
    fn two_page_read_and_write_round_trip_via_prp() {
        let dev = MockNvmeDevice::new();
        let mut ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).unwrap();
        ctrl.identify(16, || {}).unwrap();
        let ns = ctrl.namespace();
        let mut qp = ctrl.create_io_qpair(1, 64, Some(0), 16, || {}).unwrap();

        // Read 2 blocks (8 KiB ⇒ PRP1 + PRP2-direct) at LBA 10.
        let buf = ctrl.pool().alloc(ns.bytes(2) as usize).unwrap();
        let cmd = ns.read(10, 2, buf.iova(), None).unwrap();
        assert_ne!(cmd.prp2, 0, "two pages use PRP2 directly");
        qp.submit_one(cmd, 0, ctrl.regs()).unwrap();
        while qp.poll(ctrl.regs()).is_none() {}
        let got = unsafe { core::slice::from_raw_parts(buf.cpu(), ns.bytes(2) as usize) };
        assert_eq!(got, &dev.disk_snapshot(10, 2)[..]);

        // Write a pattern to LBA 20, then read it back.
        let wbuf = ctrl.pool().alloc(ns.block_size as usize).unwrap();
        unsafe { core::ptr::write_bytes(wbuf.cpu(), 0x5A, ns.block_size as usize) };
        let wcmd = ns.write(20, 1, wbuf.iova(), None).unwrap();
        qp.submit_one(wcmd, 0, ctrl.regs()).unwrap();
        while qp.poll(ctrl.regs()).is_none() {}
        assert!(
            dev.disk_snapshot(20, 1).iter().all(|&b| b == 0x5A),
            "write reached the disk"
        );
    }

    /// **Breadth on the wire**: a PRP-list read (8 pages through the SIMD
    /// iota + the device's list walk), batched submissions with ONE
    /// doorbell ring, a second qpair interleaved (the per-core shape),
    /// a device-surfaced error (LBA beyond the disk), and the honest
    /// counters recording all of it.
    #[test]
    fn prp_list_io_batched_doorbell_two_qpairs_and_counters() {
        use crate::prp::ListPage;

        let dev = MockNvmeDevice::new();
        let mut ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).unwrap();
        ctrl.identify(16, || {}).unwrap();
        let ns = ctrl.namespace();
        let mut qp1 = ctrl.create_io_qpair(1, 64, Some(0), 16, || {}).unwrap();
        let mut qp2 = ctrl.create_io_qpair(2, 64, Some(1), 16, || {}).unwrap();
        assert_eq!(dev.io_queue_count(), 2, "two hardware queues live");

        // An 8-page (32 KiB) read via a PRP list on qp1.
        let data = ctrl.pool().alloc(8 * 4096).unwrap();
        let listpage = ctrl.pool().alloc(4096).unwrap();
        let mut entries = [0u64; 512];
        let cmd = ns
            .read(
                8,
                8,
                data.iova(),
                Some(ListPage {
                    iova: listpage.iova(),
                    entries: &mut entries,
                }),
            )
            .unwrap();
        // The driver-built list is in driver memory; the DEVICE reads it
        // from the list page — copy the entries into the DMA page (the
        // driver-task flow writes them there directly).
        unsafe {
            core::ptr::copy_nonoverlapping(entries.as_ptr(), listpage.cpu() as *mut u64, 512)
        };

        // Batch on qp1 (no doorbell yet), plus a 1-block read on qp2.
        qp1.submit(cmd, 0xAA).unwrap();
        let small = ctrl.pool().alloc(4096).unwrap();
        qp2.submit_one(
            ns.read(3, 1, small.iova(), None).unwrap(),
            0xBB,
            ctrl.regs(),
        )
        .unwrap();
        qp1.ring(ctrl.regs()); // ONE ring for the batch

        while qp1.poll(ctrl.regs()).is_none() {}
        while qp2.poll(ctrl.regs()).is_none() {}

        let got = unsafe { core::slice::from_raw_parts(data.cpu(), 8 * 4096) };
        assert_eq!(
            got,
            &dev.disk_snapshot(8, 8)[..],
            "8 pages via the PRP list"
        );
        let got2 = unsafe { core::slice::from_raw_parts(small.cpu(), 4096) };
        assert_eq!(got2, &dev.disk_snapshot(3, 1)[..], "qp2 interleaved");

        // A read past the disk end surfaces the device's error status.
        let bad = ns.read(ns.block_count - 1, 1, small.iova(), None).unwrap();
        qp1.submit_one(bad, 0, ctrl.regs()).unwrap();
        let c = loop {
            if let Some(c) = qp1.poll(ctrl.regs()) {
                break c;
            }
        };
        assert!(!c.is_success(), "mock disk is smaller than the namespace");

        let (s1, s2) = (qp1.stats(), qp2.stats());
        assert_eq!((s1.submitted, s1.completed, s1.errors), (2, 2, 1));
        assert_eq!((s2.submitted, s2.completed, s2.errors), (1, 1, 0));
    }

    /// **DSM deallocate + SMART**: the discard path a swap backend rides
    /// (ranges reach the device; blocks read back zeroed), and the health
    /// log decodes.
    #[test]
    fn deallocate_and_smart_log() {
        use crate::commands::DsmRange;

        let dev = MockNvmeDevice::new();
        let mut ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).unwrap();
        ctrl.identify(16, || {}).unwrap();
        let ns = ctrl.namespace();
        let mut qp = ctrl.create_io_qpair(1, 64, Some(0), 16, || {}).unwrap();

        // Deallocate LBAs 4..6 (two ranges).
        let table = ctrl.pool().alloc(4096).unwrap();
        let ranges = [DsmRange::new(4, 1), DsmRange::new(5, 2)];
        unsafe {
            core::ptr::copy_nonoverlapping(
                ranges.as_ptr(),
                table.cpu() as *mut DsmRange,
                ranges.len(),
            )
        };
        let cmd = ns.deallocate(table.iova(), 2).unwrap();
        qp.submit_one(cmd, 0, ctrl.regs()).unwrap();
        while qp.poll(ctrl.regs()).is_none() {}
        assert_eq!(dev.deallocated(), alloc::vec![(4, 1), (5, 2)]);
        assert!(
            dev.disk_snapshot(5, 2).iter().all(|&b| b == 0),
            "deallocated blocks read back zeroed"
        );

        // SMART decodes through the typed log.
        let log = ctrl.smart(16, || {}).unwrap();
        assert_eq!(log.temp_celsius(), 36);
        assert_eq!(log.percentage_used, 3);
        assert_eq!(
            crate::commands::SmartLog::counter(&log.data_units_read),
            1234
        );
        assert_eq!(crate::commands::SmartLog::counter(&log.power_cycles), 7);
    }

    /// The MDTS guard refuses an oversized transfer before it reaches the
    /// device (the mock advertises MDTS = 128 KiB).
    #[test]
    fn mdts_guard_refuses_oversized_transfers() {
        let dev = MockNvmeDevice::new();
        let mut ctrl = Controller::bring_up(dev.clone(), dev.pool(), 16, || {}).unwrap();
        ctrl.identify(16, || {}).unwrap();
        let ns = ctrl.namespace();
        assert_eq!(ns.mdts_bytes, 4096 << 5, "128 KiB advertised");
        // 33 blocks × 4 KiB = 132 KiB > 128 KiB.
        assert!(matches!(
            ns.read(0, 33, IovaAddr(0x1000), None),
            Err(crate::Error::ExceedsMdts)
        ));
    }

    /// Bring-up actually drives the register protocol: a controller that
    /// never asserts RDY (a mock that ignores CC.EN) times out rather than
    /// hanging or falsely succeeding.
    #[test]
    fn a_controller_that_never_readies_times_out() {
        struct DeadRegs {
            cc: Rc<Cell<u32>>,
        }
        impl crate::seam::Regs for DeadRegs {
            fn read32(&self, off: usize) -> u32 {
                match off {
                    REG_CAP => MOCK_CAP as u32,
                    x if x == REG_CAP + 4 => (MOCK_CAP >> 32) as u32,
                    REG_VS => MOCK_VS,
                    REG_CC => self.cc.get(),
                    _ => 0, // CSTS always 0 ⇒ never ready
                }
            }
            fn write32(&self, off: usize, value: u32) {
                if off == REG_CC {
                    self.cc.set(value);
                }
            }
        }
        let cc = Rc::new(Cell::new(0));
        let r = Controller::bring_up(DeadRegs { cc: cc.clone() }, MockPool, 8, || {});
        assert!(matches!(r, Err(crate::Error::NotReady)));
        assert_eq!(
            cc.get() & CC_EN,
            0,
            "a failed start best-effort disables before releasing its queues"
        );
    }

    /// A capability host must retain both seams across a refused start so it
    /// can restore BAR/pool custody before disabling bus mastering and
    /// retiring mappings. The typed error makes dropping either seam before
    /// that ownership decision impossible inside `PreparedController::start`.
    #[test]
    fn a_refused_start_returns_the_prepared_seams() {
        struct HeldRegs {
            cc: Rc<Cell<u32>>,
            dropped: Rc<Cell<bool>>,
        }

        impl Drop for HeldRegs {
            fn drop(&mut self) {
                self.dropped.set(true);
            }
        }

        impl crate::seam::Regs for HeldRegs {
            fn read32(&self, off: usize) -> u32 {
                match off {
                    REG_CAP => MOCK_CAP as u32,
                    x if x == REG_CAP + 4 => (MOCK_CAP >> 32) as u32,
                    REG_VS => MOCK_VS,
                    REG_CC => self.cc.get(),
                    _ => 0,
                }
            }

            fn write32(&self, off: usize, value: u32) {
                if off == REG_CC {
                    self.cc.set(value);
                }
            }
        }

        struct HeldPool {
            dropped: Rc<Cell<bool>>,
        }

        impl Drop for HeldPool {
            fn drop(&mut self) {
                self.dropped.set(true);
            }
        }

        impl DmaPool for HeldPool {
            type Buf = MockBuf;

            fn alloc(&self, bytes: usize) -> Option<Self::Buf> {
                MockPool.alloc(bytes)
            }
        }

        let cc = Rc::new(Cell::new(0));
        let regs_dropped = Rc::new(Cell::new(false));
        let pool_dropped = Rc::new(Cell::new(false));
        let prepared = Controller::prepare(
            HeldRegs {
                cc: cc.clone(),
                dropped: regs_dropped.clone(),
            },
            HeldPool {
                dropped: pool_dropped.clone(),
            },
            8,
            || {},
        )
        .expect("preparation succeeds while CSTS is halted");

        let failure = match prepared.start(8, || {}) {
            Ok(_) => panic!("a controller that never sets RDY cannot start"),
            Err(failure) => failure,
        };
        assert_eq!(failure.error(), crate::Error::NotReady);
        assert_eq!(cc.get() & CC_EN, 0, "refused start rolls CC.EN back");
        assert!(!regs_dropped.get(), "the register seam rides back");
        assert!(!pool_dropped.get(), "the pool seam rides back");

        let (error, regs, pool) = failure.into_seam();
        assert_eq!(error, crate::Error::NotReady);
        assert!(!regs_dropped.get(), "recovery transfers register custody");
        assert!(!pool_dropped.get(), "recovery transfers pool custody");
        drop(regs);
        drop(pool);
        assert!(regs_dropped.get());
        assert!(pool_dropped.get());
    }
}
