//! **`MockE1000e`** — an 82574L in process memory, the host proof for the
//! whole core (the `MockNvmeDevice` lesson: prove the semantics on the
//! host, prove the seam on hardware).
//!
//! It is a [`Regs`] register file with **side effects**: writing `CTRL.RST`
//! runs the reset vector the bring-up waits on; writing `TDT` walks the
//! transmit ring from `TDH`, copies each frame onto the wire, and writes
//! `DD` back; writing `RDT` (or enabling the receiver) delivers queued
//! frames into the receive ring with `DD|EOP`; and every cause goes through
//! the same interrupt model QEMU's `e1000e` implements in MSI-X mode —
//! `ICR` is **not** cleared by a read while `IMS` is non-zero, and
//! delivering a message auto-clears the `EIAC` causes from `ICR` *and*
//! `IMS` — so a driver that forgets to re-arm fails here, not on the gate.
//!
//! DMA is process memory: [`MockPool`] (the substrate's
//! [`host_double`](kore_driver::host_double)) hands out page-aligned leaked
//! allocations with `iova == cpu pointer`, so the mock dereferences the
//! addresses the driver programmed directly. It refuses an extended
//! descriptor: this core is legacy-only, and the mock is where that is
//! checked.

use alloc::collections::VecDeque;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::ptr;

use kore_driver::seam::{Dma, Regs};

use crate::desc::{DESC_BYTES, RxDesc, TxDesc};
use crate::regs::REGS_SPAN;
use crate::regs::i82574::*;

/// The station address the mock's "NVM" loads.
pub const MOCK_MAC: [u8; 6] = [0x52, 0x54, 0x00, 0x12, 0x34, 0x56];

// ── The DMA pool: the substrate's host double, iova == cpu pointer ───────

/// The pool the tests draw from — `kore_driver::host_double`'s, one copy
/// for every driver crate's mock rather than one per crate.
pub use kore_driver::host_double::HostPool as MockPool;

// ── The device ───────────────────────────────────────────────────────────

const WORDS: usize = REGS_SPAN / 4;
const RX_BUF: usize = 2048;

struct State {
    regs: Vec<u32>,
    /// Frames transmitted, in order.
    wire: Vec<Vec<u8>>,
    /// Frames waiting for a receive descriptor, with the error byte each
    /// should be written back with.
    inbox: VecDeque<(Vec<u8>, u8)>,
    loopback: bool,
    link_up: bool,
    hold_reset: bool,
    nvm_never_loads: bool,
    no_station_address: bool,
    /// Transmitted descriptors are not written back until released — for
    /// the ring-full test.
    hold_tx_writeback: bool,
    held_tx: Vec<u64>,
    /// The MSI-X table entry each delivered message named.
    notifies: Vec<u8>,
    resets: u32,
    tdt_writes: u32,
}

/// The fake controller.
pub struct MockE1000e {
    st: RefCell<State>,
}

impl MockE1000e {
    /// A powered, link-up controller before any driver touched it.
    pub fn new() -> Self {
        let this = Self {
            st: RefCell::new(State {
                regs: alloc::vec![0; WORDS],
                wire: Vec::new(),
                inbox: VecDeque::new(),
                loopback: false,
                link_up: true,
                hold_reset: false,
                nvm_never_loads: false,
                no_station_address: false,
                hold_tx_writeback: false,
                held_tx: Vec::new(),
                notifies: Vec::new(),
                resets: 0,
                tdt_writes: 0,
            }),
        };
        this.st.borrow_mut().reset_vector();
        this.st.borrow_mut().resets = 0;
        this
    }

    /// Transmitted frames come back as received ones.
    pub fn set_loopback(&self, on: bool) {
        self.st.borrow_mut().loopback = on;
    }
    /// Raise or drop the link; raises `LSC`.
    pub fn set_link(&self, up: bool) {
        let mut st = self.st.borrow_mut();
        st.link_up = up;
        let status = st.regs[E1000_STATUS as usize / 4];
        st.regs[E1000_STATUS as usize / 4] = if up {
            status | E1000_STATUS_LU | E1000_STATUS_SPEED_1000
        } else {
            status & !(E1000_STATUS_LU | E1000_STATUS_SPEED_MASK)
        };
        st.raise(E1000_ICR_LSC);
    }
    /// `CTRL.RST` never self-clears.
    pub fn hold_reset(&self) {
        self.st.borrow_mut().hold_reset = true;
    }
    /// `EECD.AUTO_RD` never sets.
    pub fn nvm_never_loads(&self) {
        self.st.borrow_mut().nvm_never_loads = true;
    }
    /// The receive-address entry comes up without `AV`.
    pub fn no_station_address(&self) {
        self.st.borrow_mut().no_station_address = true;
    }
    /// Transmitted descriptors stay un-written-back until released.
    pub fn hold_tx_writeback(&self) {
        self.st.borrow_mut().hold_tx_writeback = true;
    }
    /// Write back everything held.
    pub fn release_tx_writeback(&self) {
        let mut st = self.st.borrow_mut();
        st.hold_tx_writeback = false;
        let held = core::mem::take(&mut st.held_tx);
        for addr in held {
            // SAFETY: a descriptor address the driver programmed, in the
            // pool's leaked memory.
            unsafe { ptr::write_volatile((addr as *mut u8).add(12), E1000_TXD_STAT_DD as u8) };
        }
        st.raise(E1000_ICR_TXDW | E1000_ICR_TXQ0);
    }
    /// Raise one cause as the controller would — an overrun, say — without
    /// a frame behind it.
    pub fn raise_cause(&self, cause: u32) {
        self.st.borrow_mut().raise(cause);
    }
    /// A frame arrives on the wire.
    pub fn inject_rx(&self, bytes: &[u8]) {
        self.inject_rx_with_errors(bytes, 0);
    }
    /// A frame arrives, to be written back with `errors`.
    pub fn inject_rx_with_errors(&self, bytes: &[u8], errors: u8) {
        let mut st = self.st.borrow_mut();
        st.inbox.push_back((bytes.to_vec(), errors));
        st.deliver();
    }
    /// Everything transmitted so far.
    pub fn transmitted(&self) -> Vec<Vec<u8>> {
        self.st.borrow().wire.clone()
    }
    /// The MSI-X entries of every message delivered so far.
    pub fn notifies(&self) -> Vec<u8> {
        self.st.borrow().notifies.clone()
    }
    /// A register, read without side effects.
    pub fn reg(&self, off: usize) -> u32 {
        self.st.borrow().regs[off / 4]
    }
    /// Resets performed.
    pub fn resets(&self) -> u32 {
        self.st.borrow().resets
    }
    /// `TDT` writes seen.
    pub fn tdt_writes(&self) -> u32 {
        self.st.borrow().tdt_writes
    }
    /// Transmit descriptor `slot`, as the device sees it.
    pub fn tx_desc(&self, slot: u16) -> TxDesc {
        let st = self.st.borrow();
        let base = st.ring_base(e1000_tdbal(0), e1000_tdbah(0));
        // SAFETY: the driver programmed the ring base at a leaked pool buffer.
        unsafe { ptr::read_volatile((base as *const TxDesc).add(slot as usize)) }
    }
    /// Receive descriptor `slot`, as the device sees it.
    pub fn rx_desc(&self, slot: u16) -> RxDesc {
        let st = self.st.borrow();
        let base = st.ring_base(e1000_rdbal(0), e1000_rdbah(0));
        // SAFETY: as above.
        unsafe { ptr::read_volatile((base as *const RxDesc).add(slot as usize)) }
    }
    /// Where receive buffer `slot` should be, given the pool.
    pub fn rx_desc_addr_expected(&self, pool: &impl Dma, slot: u16) -> u64 {
        pool.iova().0 + slot as u64 * RX_BUF as u64
    }
}

impl Default for MockE1000e {
    fn default() -> Self {
        Self::new()
    }
}

impl State {
    fn w(&self, off: u32) -> u32 {
        self.regs[off as usize / 4]
    }
    fn set(&mut self, off: u32, v: u32) {
        self.regs[off as usize / 4] = v;
    }

    fn reset_vector(&mut self) {
        self.resets += 1;
        let hold = self.hold_reset;
        self.regs.iter_mut().for_each(|w| *w = 0);
        let mut status = E1000_STATUS_FD;
        if self.link_up {
            status |= E1000_STATUS_LU | E1000_STATUS_SPEED_1000;
        }
        self.set(E1000_STATUS, status);
        let mac = MOCK_MAC;
        self.set(
            e1000_ral(0),
            u32::from_le_bytes([mac[0], mac[1], mac[2], mac[3]]),
        );
        let rah = u32::from(u16::from_le_bytes([mac[4], mac[5]]));
        self.set(
            e1000_rah(0),
            if self.no_station_address {
                rah
            } else {
                rah | E1000_RAH_AV
            },
        );
        let mut eecd = E1000_EECD_PRES;
        if !self.nvm_never_loads {
            eecd |= E1000_EECD_AUTO_RD;
        }
        self.set(E1000_EECD, eecd);
        if hold {
            self.set(E1000_CTRL, E1000_CTRL_RST);
        }
    }

    fn ring_base(&self, lo: u32, hi: u32) -> u64 {
        u64::from(self.w(lo)) | (u64::from(self.w(hi)) << 32)
    }

    /// The transmit engine: walk `[TDH, TDT)`.
    fn transmit(&mut self) {
        if self.w(E1000_TCTL) & E1000_TCTL_EN == 0 {
            return;
        }
        let base = self.ring_base(e1000_tdbal(0), e1000_tdbah(0));
        let n = (self.w(e1000_tdlen(0)) as usize / DESC_BYTES) as u32;
        let mut head = self.w(e1000_tdh(0));
        let tail = self.w(e1000_tdt(0));
        let mut sent = false;
        while head != tail {
            let at = base + u64::from(head) * DESC_BYTES as u64;
            // SAFETY: a descriptor address the driver programmed, inside a
            // leaked pool buffer; iova == cpu pointer.
            let desc = unsafe { ptr::read_volatile(at as *const TxDesc) };
            assert_eq!(
                desc.cmd & E1000_TXD_CMD_DEXT as u8,
                0,
                "this core speaks legacy descriptors only"
            );
            // SAFETY: the frame address is the client's slot in a leaked
            // window; the length is what the driver declared.
            let bytes = unsafe {
                core::slice::from_raw_parts(desc.addr as *const u8, desc.length as usize)
            }
            .to_vec();
            if self.loopback {
                self.inbox.push_back((bytes.clone(), 0));
            }
            self.wire.push(bytes);
            if desc.cmd & E1000_TXD_CMD_RS as u8 != 0 {
                if self.hold_tx_writeback {
                    self.held_tx.push(at);
                } else {
                    // SAFETY: as above; the status byte at +12.
                    unsafe {
                        ptr::write_volatile((at as *mut u8).add(12), E1000_TXD_STAT_DD as u8)
                    };
                }
            }
            head = (head + 1) % n;
            sent = true;
        }
        self.set(e1000_tdh(0), head);
        if sent && !self.hold_tx_writeback {
            self.raise(E1000_ICR_TXDW | E1000_ICR_TXQ0);
        }
        if self.loopback {
            self.deliver();
        }
    }

    /// The receive engine: fill `[RDH, RDT)` from the inbox.
    fn deliver(&mut self) {
        if self.w(E1000_RCTL) & E1000_RCTL_EN == 0 {
            return;
        }
        let base = self.ring_base(e1000_rdbal(0), e1000_rdbah(0));
        let n = (self.w(e1000_rdlen(0)) as usize / DESC_BYTES) as u32;
        if n == 0 {
            return;
        }
        let mut head = self.w(e1000_rdh(0));
        let tail = self.w(e1000_rdt(0));
        let mut delivered = false;
        while head != tail {
            let Some((frame, errors)) = self.inbox.pop_front() else {
                break;
            };
            let at = base + u64::from(head) * DESC_BYTES as u64;
            // SAFETY: a descriptor the driver posted, in leaked pool memory.
            let mut desc = unsafe { ptr::read_volatile(at as *const RxDesc) };
            let len = frame.len().min(RX_BUF);
            // SAFETY: the buffer the driver pointed this descriptor at, of
            // `RX_BUF` bytes, in leaked pool memory.
            unsafe { ptr::copy_nonoverlapping(frame.as_ptr(), desc.addr as *mut u8, len) };
            desc.length = len as u16;
            desc.status = E1000_RXD_STAT_DD as u8 | E1000_RXD_STAT_EOP as u8;
            desc.errors = errors;
            // SAFETY: as above.
            unsafe { ptr::write_volatile(at as *mut RxDesc, desc) };
            head = (head + 1) % n;
            delivered = true;
        }
        self.set(e1000_rdh(0), head);
        if delivered {
            self.raise(E1000_ICR_RXT0 | E1000_ICR_RXQ0);
        }
    }

    /// QEMU's MSI-X cause model: set the cause, deliver a message per
    /// enabled queue/other cause through `IVAR`, auto-clear the `EIAC`
    /// causes from `ICR` and `IMS`.
    fn raise(&mut self, cause: u32) {
        let icr = self.w(E1000_ICR) | cause;
        self.set(E1000_ICR, icr);
        // QEMU's `E1000_ICR_OTHER_CAUSES` is LSC | RXO | MDAC | SRPD | ACK |
        // MNG: the legacy timer causes (`RXT0`, `TXDW`) are *not* folded
        // into `OTHER`, so a receive delivers exactly one queue message.
        const OTHER_CAUSES: u32 = E1000_ICR_LSC | E1000_ICR_RXO;
        if self.w(E1000_ICR) & self.w(E1000_IMS) & OTHER_CAUSES != 0 {
            let icr = self.w(E1000_ICR) | E1000_ICR_OTHER;
            self.set(E1000_ICR, icr);
        }
        let ivar = self.w(E1000_IVAR);
        for (bit, shift) in [
            (E1000_ICR_RXQ0, E1000_IVAR_RXQ0_SHIFT),
            (E1000_ICR_TXQ0, E1000_IVAR_TXQ0_SHIFT),
            (E1000_ICR_OTHER, E1000_IVAR_OTHER_SHIFT),
        ] {
            if self.w(E1000_ICR) & self.w(E1000_IMS) & bit == 0 {
                continue;
            }
            let field = (ivar >> shift) & 0xf;
            if field & E1000_IVAR_INT_ALLOC_VALID == 0 {
                continue;
            }
            self.notifies.push((field & 0x7) as u8);
            let eiac = self.w(E1000_EIAC) & bit;
            let icr = self.w(E1000_ICR) & !eiac;
            self.set(E1000_ICR, icr);
            if self.w(E1000_CTRL_EXT) & E1000_CTRL_EXT_IAME == 0 {
                let ims = self.w(E1000_IMS) & !eiac;
                self.set(E1000_IMS, ims);
            }
        }
    }
}

impl Regs for MockE1000e {
    fn read32(&self, off: usize) -> u32 {
        let mut st = self.st.borrow_mut();
        let off = off as u32;
        let v = st.w(off);
        if off == E1000_ICR && st.w(E1000_IMS) == 0 {
            // Read-to-clear only while fully masked (QEMU's MSI-X rule).
            st.set(E1000_ICR, 0);
        }
        v
    }

    fn write32(&self, off: usize, val: u32) {
        let mut st = self.st.borrow_mut();
        let off = off as u32;
        match off {
            E1000_CTRL => {
                if val & E1000_CTRL_RST != 0 {
                    st.reset_vector();
                } else {
                    st.set(E1000_CTRL, val);
                }
            }
            E1000_ICR => {
                let icr = st.w(E1000_ICR) & !val;
                st.set(E1000_ICR, icr);
            }
            E1000_IMS => {
                let ims = st.w(E1000_IMS) | val;
                st.set(E1000_IMS, ims);
                st.raise(0);
            }
            E1000_IMC => {
                let ims = st.w(E1000_IMS) & !val;
                st.set(E1000_IMS, ims);
            }
            E1000_RCTL => {
                st.set(E1000_RCTL, val);
                st.deliver();
            }
            _ if off == e1000_tdt(0) => {
                st.tdt_writes += 1;
                st.set(off, val);
                st.transmit();
            }
            _ if off == e1000_rdt(0) => {
                st.set(off, val);
                st.deliver();
            }
            _ => st.set(off, val),
        }
    }
}
