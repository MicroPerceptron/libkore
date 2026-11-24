//! **`MockRtw89`** — an RTL8922AE in process memory, the host proof for
//! the whole ladder (the `MockNvmeDevice` lesson: prove the semantics on
//! the host, prove the seam on hardware).
//!
//! A byte-addressed register file with **side effects**, so the narrow
//! accesses the part needs (`R_BE_UDM1 + 1`, `R_BE_H2CREG_CTRL`, the
//! scoreboard byte) land where the dword views see them: the power
//! sequence's polls answer once their enables are written; the crystal
//! interface records every command and completes it; the efuse window
//! serves a physical map the test built; the DLE reports ready when its
//! engines are enabled; the firmware-command ring is walked on every
//! doorbell — each descriptor dereferenced through the host double's
//! `iova == cpu pointer` — and the download's status field follows what
//! the header packet declared; the register mailbox answers `GET_FEATURE`
//! with a `PHY_CAP` once the firmware is "running". Each rung has a
//! fault switch that makes exactly that rung refuse.
//!
//! DMA goes through [`IommuPool`]: the substrate's host double allocates
//! the memory, but the device-visible address is a translation this
//! module hands out from a 36-bit-addressable window and resolves back
//! when the mock dereferences a descriptor — a host heap pointer is
//! 47 bits wide, which the part's own reach check rightly refuses, and
//! an address the driver never leased is an IOMMU fault here as on
//! metal.

use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::{Cell, RefCell};

use kore_abi::device::IovaAddr;
use kore_driver::host_double::{HostBuf, HostPool};
use kore_driver::seam::{Dma, DmaPool, Regs, RegsNarrow};

use crate::desc::{FWCMD_DESC_BYTES, H2C_HDR_BYTES, TX_BD_BYTES};
use crate::efuse::PHYSICAL_SIZE;
use crate::mailbox::PhyCap;
use crate::regs::be::*;
use crate::regs::{REGS_SPAN, TXCH_FWCMD, TxChannelRegs, encode, get};

/// One leased window: where the device sees it, where the host has it.
#[derive(Clone, Copy, Debug)]
struct Mapping {
    iova: u64,
    cpu: usize,
    len: usize,
}

/// The translation table the pool fills and the device resolves through.
type Table = Rc<RefCell<Vec<Mapping>>>;

/// The first device address handed out: 2 GiB, inside the 36-bit reach
/// and above anything a real BAR would alias.
const IOVA_BASE: u64 = 0x8000_0000;

/// A buffer from [`IommuPool`]: host memory with a translated address.
pub struct IommuBuf {
    inner: HostBuf,
    iova: u64,
}

impl Dma for IommuBuf {
    fn cpu(&self) -> *mut u8 {
        self.inner.cpu()
    }
    fn iova(&self) -> IovaAddr {
        IovaAddr(self.iova)
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
}

/// The pool the tests draw from: the substrate's host double behind a
/// bump-allocated device address space the mock translates.
#[derive(Clone)]
pub struct IommuPool {
    inner: HostPool,
    table: Table,
    next: Rc<Cell<u64>>,
}

impl DmaPool for IommuPool {
    type Buf = IommuBuf;
    fn alloc(&self, bytes: usize) -> Option<IommuBuf> {
        let inner = self.inner.alloc(bytes)?;
        let iova = self.next.get();
        self.next.set(iova + inner.len() as u64);
        self.table.borrow_mut().push(Mapping {
            iova,
            cpu: inner.cpu() as usize,
            len: inner.len(),
        });
        Some(IommuBuf { inner, iova })
    }
}

/// Bytes of "efuse" the mock serves: the physical map plus the
/// secure-boot bytes at `0x1580` and room after them.
const EFUSE_BYTES: usize = 0x1800;
const _: () =
    assert!(EFUSE_BYTES > EFUSE_SEC_BE_START as usize + 4 && EFUSE_BYTES >= PHYSICAL_SIZE);

/// One recorded write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Write {
    /// 1, 2 or 4.
    pub width: u8,
    /// Byte offset.
    pub off: u32,
    /// The value written.
    pub val: u32,
}

/// The fault switches.
#[derive(Clone, Copy, Debug, Default)]
pub struct Faults {
    /// Every read answers all-ones.
    pub dead: bool,
    /// `RDY_SYSPWR` never rises.
    pub power_never_ready: bool,
    /// The efuse window never reports a word ready.
    pub efuse_never_ready: bool,
    /// The DLE engines never report ready.
    pub dle_never_ready: bool,
    /// Every DMA channel reads busy.
    pub dma_busy: bool,
    /// `H2C_PATH_RDY` never rises.
    pub h2c_path_never: bool,
    /// The WCPU answers the download with this status code instead of
    /// ready (4 = checksum fail).
    pub reject_status: Option<u8>,
    /// BB-MCU never clears its download bit even though WLAN looks ready.
    pub baseband_never_ready: bool,
    /// A normal-MAC handshake register whose completion never arrives.
    pub normal_wait_stuck: Option<u32>,
    /// RF-v2 HWSI path that never becomes idle.
    pub rf_busy_path: Option<u8>,
    /// The final analog write remains busy, even across a direct RF write.
    pub rf_write_stuck: bool,
    /// Firmware boots, but the normal RF command descriptor never retires.
    pub rf_command_stall: bool,
    pub mac_command_stall: bool,
    pub command_done_silent: bool,
    pub command_reject: Option<u8>,
    pub command_wrong_identity: bool,
    pub c2h_bad_tag: bool,
    pub c2h_bad_len: bool,
    /// Override the final TX hardware index despite firmware readiness.
    pub final_tx_index: Option<u16>,
    /// The mailbox never replies.
    pub mailbox_silent: bool,
    /// The mailbox replies with function 1 (`ERR_RPT`).
    pub mailbox_wrong_func: bool,
}

struct State {
    mem: Vec<u8>,
    efuse: Vec<u8>,
    table: Table,
    faults: Faults,
    trace: Vec<Write>,
    xtal: Vec<(u8, u8, u8)>,
    powered: bool,
    power_ons: u32,
    power_offs: u32,
    /// The firmware-command ring's hardware index.
    ch12_rp: u16,
    header_packet: Option<Vec<u8>>,
    expected_payload: usize,
    suit_received: usize,
    received: Vec<u8>,
    rf_commands: Vec<Vec<u8>>,
    mac_commands: Vec<Vec<u8>>,
    rx_tag: u16,
    fw_running: bool,
    phy_cap: PhyCap,
}

/// The fake radio.
pub struct MockRtw89 {
    st: RefCell<State>,
}

/// The `PHY_CAP` the mock's firmware answers.
pub const MOCK_PHY_CAP: PhyCap = PhyCap {
    rx_nss: 2,
    bw: 4,
    tx_nss: 2,
    prot: 0x0F,
    nic: 1,
    wl_func: 0x1F,
    hw_type: 0,
    ant_tx_num: 2,
    ant_rx_num: 2,
    band_sel: 0x07,
};

impl MockRtw89 {
    /// A cold radio whose efuse is `efuse` (the physical map, then the
    /// secure-boot bytes at `0x1580`).
    pub fn new(efuse: &[u8], faults: Faults) -> Self {
        let mut map = alloc::vec![0xFFu8; EFUSE_BYTES];
        map[..efuse.len()].copy_from_slice(efuse);
        let mut mem = alloc::vec![0u8; REGS_SPAN.max(crate::mac::BASEBAND_REGS_SPAN)];
        // The autoload flag: bit 5 of the halfword at 0x000A.
        mem[R_BE_SYS_WL_EFUSE_CTRL as usize] |= B_BE_AUTOLOAD_SUS as u8;
        // The A-cut is the silicon on Gigatron's first sample? No: `rev
        // 01` — cut B. `R_AX_SYS_CFG1[15:12]` = 1.
        mem[R_AX_SYS_CFG1 as usize + 1] = 0x10;
        Self {
            st: RefCell::new(State {
                mem,
                efuse: map,
                table: Rc::new(RefCell::new(Vec::new())),
                faults,
                trace: Vec::new(),
                xtal: Vec::new(),
                powered: false,
                power_ons: 0,
                power_offs: 0,
                ch12_rp: 0,
                header_packet: None,
                expected_payload: 0,
                suit_received: 0,
                received: Vec::new(),
                rf_commands: Vec::new(),
                mac_commands: Vec::new(),
                rx_tag: 1,
                fw_running: false,
                phy_cap: MOCK_PHY_CAP,
            }),
        }
    }

    /// A pool whose buffers this device can reach.
    pub fn pool(&self) -> IommuPool {
        IommuPool {
            inner: HostPool,
            table: Rc::clone(&self.st.borrow().table),
            next: Rc::new(Cell::new(IOVA_BASE)),
        }
    }

    /// A radio already powered (a warm hand-over).
    pub fn set_powered(&self) {
        self.st.borrow_mut().set_powered(true);
    }

    pub fn powered(&self) -> bool {
        self.st.borrow().powered
    }
    pub fn power_ons(&self) -> u32 {
        self.st.borrow().power_ons
    }
    pub fn power_offs(&self) -> u32 {
        self.st.borrow().power_offs
    }
    pub fn xtal_writes(&self) -> Vec<(u8, u8, u8)> {
        self.st.borrow().xtal.clone()
    }
    pub fn trace(&self) -> Vec<Write> {
        self.st.borrow().trace.clone()
    }
    pub fn header_packet(&self) -> Option<Vec<u8>> {
        self.st.borrow().header_packet.clone()
    }
    pub fn received(&self) -> Vec<u8> {
        self.st.borrow().received.clone()
    }
    pub fn mac_commands(&self) -> Vec<Vec<u8>> {
        self.st.borrow().mac_commands.clone()
    }

    pub fn rf_commands(&self) -> Vec<Vec<u8>> {
        self.st.borrow().rf_commands.clone()
    }
    pub fn fw_running(&self) -> bool {
        self.st.borrow().fw_running
    }
    /// The register file's dword at `off`.
    pub fn word(&self, off: u32) -> u32 {
        self.st.borrow().w(off)
    }
    pub fn byte(&self, off: u32) -> u8 {
        self.st.borrow().mem[off as usize]
    }
}

impl State {
    /// Resolve a device address the driver programmed, or fault.
    fn translate(&self, iova: u64, len: usize) -> *const u8 {
        let table = self.table.borrow();
        let m = table
            .iter()
            .find(|m| iova >= m.iova && iova + len as u64 <= m.iova + m.len as u64)
            .unwrap_or_else(|| panic!("IOMMU fault: {len} bytes at {iova:#x} were never leased"));
        (m.cpu + (iova - m.iova) as usize) as *const u8
    }

    fn w(&self, off: u32) -> u32 {
        let o = off as usize;
        u32::from_le_bytes([
            self.mem[o],
            self.mem[o + 1],
            self.mem[o + 2],
            self.mem[o + 3],
        ])
    }

    fn set(&mut self, off: u32, val: u32) {
        let o = off as usize;
        self.mem[o..o + 4].copy_from_slice(&val.to_le_bytes());
    }

    fn set_powered(&mut self, on: bool) {
        self.powered = on;
        let state = if on {
            encode(MAC_AX_SYS_ACT, B_BE_WHOLE_SYS_PWR_STE_MASK) | encode(1, B_BE_WLMAC_PWR_STE_MASK)
        } else {
            0
        };
        self.set(R_BE_IC_PWR_STATE, state);
        if on {
            self.power_ons += 1;
        } else {
            self.power_offs += 1;
            self.fw_running = false;
        }
    }

    /// The side effects of a register's new value.
    fn effect(&mut self, off: u32) {
        match off {
            R_BE_SYS_PW_CTRL => {
                let v = self.w(off);
                if v & B_BE_APFN_ONMAC != 0 {
                    self.set(off, v & !B_BE_APFN_ONMAC);
                    self.set_powered(true);
                }
                if v & B_BE_APFM_OFFMAC != 0 {
                    self.set(off, self.w(off) & !B_BE_APFM_OFFMAC);
                    self.set_powered(false);
                }
            }
            R_BE_HCI_OPT_CTRL => {
                let v = self.w(off);
                let mut st = v & !(B_BE_HAXIDMA_IO_ST
                    | B_BE_HAXIDMA_BACKUP_RESTORE_ST
                    | B_BE_HCI_WLAN_IO_ST);
                if v & B_BE_HAXIDMA_IO_EN != 0 {
                    st |= B_BE_HAXIDMA_IO_ST;
                }
                if v & B_BE_HCI_WLAN_IO_EN != 0 {
                    st |= B_BE_HCI_WLAN_IO_ST;
                }
                self.set(off, st);
            }
            R_BE_WLAN_XTAL_SI_CTRL => {
                let v = self.w(off);
                if v & B_BE_WL_XTAL_SI_CMD_POLL != 0 {
                    if get(v, B_BE_WL_XTAL_SI_MODE_MASK) == XTAL_SI_NORMAL_WRITE {
                        self.xtal.push((
                            get(v, B_BE_WL_XTAL_SI_ADDR_MASK) as u8,
                            get(v, B_BE_WL_XTAL_SI_DATA_MASK) as u8,
                            get(v, B_BE_WL_XTAL_SI_BITMASK_MASK) as u8,
                        ));
                    }
                    self.set(off, v & !B_BE_WL_XTAL_SI_CMD_POLL);
                }
            }
            R_BE_EFUSE_CTRL => {
                let v = self.w(off);
                if v & B_BE_EF_RDY == 0 && !self.faults.efuse_never_ready {
                    let at = get(v, B_BE_EF_ADDR_MASK) as usize;
                    let word = u32::from_le_bytes([
                        self.efuse[at],
                        self.efuse[at + 1],
                        self.efuse[at + 2],
                        self.efuse[at + 3],
                    ]);
                    self.set(R_BE_EFUSE_CTRL_1_V1, word);
                    self.set(off, v | B_BE_EF_RDY);
                }
            }
            R_BE_DMAC_FUNC_EN => {
                let v = self.w(off);
                let ready = if v & (B_BE_DLE_WDE_EN | B_BE_DLE_PLE_EN)
                    == (B_BE_DLE_WDE_EN | B_BE_DLE_PLE_EN)
                    && !self.faults.dle_never_ready
                {
                    0x3
                } else {
                    0
                };
                self.set(R_AX_WDE_INI_STATUS, ready);
                self.set(R_AX_PLE_INI_STATUS, ready);
                if get(self.w(R_BE_WDE_PKTBUF_CFG), B_BE_WDE_FREE_PAGE_NUM_MASK) != 0 {
                    if let Some(r @ (R_AX_WDE_INI_STATUS | R_AX_PLE_INI_STATUS)) =
                        self.faults.normal_wait_stuck
                    {
                        self.set(r, 0);
                    }
                }
            }
            0x22ae0 | 0x22be0 if self.faults.rf_write_stuck => {
                self.set(if off == 0x22ae0 { 0x22c24 } else { 0x22d24 }, 1 << 29);
            }
            R_BE_SS_CTRL => {
                let v = self.w(off);
                if v & B_BE_SS_EN != 0 && self.faults.normal_wait_stuck != Some(off) {
                    self.set(off, v | B_BE_SS_INIT_DONE);
                }
            }
            R_BE_MLO_INIT_CTL => {
                let v = self.w(off);
                if v & B_BE_MLO_TABLE_REINIT != 0 {
                    self.set(off, v & !B_BE_MLO_TABLE_INIT_DONE);
                } else if self.faults.normal_wait_stuck != Some(off) {
                    self.set(off, v | B_BE_MLO_TABLE_INIT_DONE);
                }
            }
            R_BE_ADDR_CAM_CTRL => {
                if self.faults.normal_wait_stuck != Some(off) {
                    self.set(off, self.w(off) & !B_BE_ADDR_CAM_CLR);
                }
            }
            R_BE_RESPBA_CAM_CTRL => {
                if self.faults.normal_wait_stuck != Some(off) {
                    self.set(off, self.w(off) & !B_BE_BACAM_RST_MASK);
                }
            }
            R_BE_HAXI_INIT_CFG1 => {
                let v = self.w(off);
                if v & B_BE_SET_BDRAM_BOUND != 0 {
                    self.set(off, v & !B_BE_SET_BDRAM_BOUND);
                }
            }
            R_BE_PLATFORM_ENABLE => {
                let v = self.w(off);
                if v & B_BE_WCPU_EN != 0 && v & B_BE_HOLD_AFTER_RESET == 0 {
                    // The CPU released into its ROM: the H2C path opens,
                    // any earlier download is forgotten.
                    self.header_packet = None;
                    self.received.clear();
                    self.expected_payload = 0;
                    self.fw_running = false;
                    if !self.faults.h2c_path_never {
                        let ctrl = self.w(R_BE_WCPU_FW_CTRL);
                        self.set(R_BE_WCPU_FW_CTRL, ctrl | B_BE_H2C_PATH_RDY);
                    }
                }
            }
            R_BE_H2CREG_CTRL => {
                if self.mem[off as usize] & B_AX_H2CREG_TRIGGER as u8 != 0 {
                    self.mailbox();
                }
            }
            _ => {}
        }
    }

    /// The firmware side of the register mailbox.
    fn mailbox(&mut self) {
        if !self.fw_running || self.faults.mailbox_silent {
            return;
        }
        let w0 = self.w(R_BE_H2CREG_DATA0);
        let func = get(w0, RTW89_H2CREG_HDR_FUNC_MASK);
        // The firmware consumed the command.
        self.mem[R_BE_H2CREG_CTRL as usize] = 0;
        if func != RTW89_FWCMD_H2CREG_FUNC_GET_FEATURE {
            return;
        }
        let c = self.phy_cap;
        let reply_func = if self.faults.mailbox_wrong_func {
            1
        } else {
            RTW89_FWCMD_C2HREG_FUNC_PHY_CAP
        };
        let words = [
            encode(reply_func, RTW89_C2HREG_HDR_FUNC_MASK)
                | encode(4, RTW89_C2HREG_HDR_LEN_MASK)
                | encode(c.rx_nss as u32, RTW89_C2HREG_PHYCAP_W0_RX_NSS)
                | encode(c.bw as u32, RTW89_C2HREG_PHYCAP_W0_BW),
            encode(c.tx_nss as u32, RTW89_C2HREG_PHYCAP_W1_TX_NSS)
                | encode(c.prot as u32, RTW89_C2HREG_PHYCAP_W1_PROT)
                | encode(c.nic as u32, RTW89_C2HREG_PHYCAP_W1_NIC)
                | encode(c.wl_func as u32, RTW89_C2HREG_PHYCAP_W1_WL_FUNC),
            encode(c.hw_type as u32, RTW89_C2HREG_PHYCAP_W2_HW_TYPE),
            encode(c.ant_tx_num as u32, RTW89_C2HREG_PHYCAP_W3_ANT_TX_NUM)
                | encode(c.ant_rx_num as u32, RTW89_C2HREG_PHYCAP_W3_ANT_RX_NUM)
                | encode(c.band_sel as u32, RTW89_C2HREG_PHYCAP_W3_BAND_SEL),
        ];
        for (r, w) in [
            R_BE_C2HREG_DATA0,
            R_BE_C2HREG_DATA1,
            R_BE_C2HREG_DATA2,
            R_BE_C2HREG_DATA3,
        ]
        .into_iter()
        .zip(words)
        {
            self.set(r, w);
        }
        self.mem[R_BE_C2HREG_CTRL as usize] = 1;
    }

    /// The doorbell on channel 12: walk the descriptors from the hardware
    /// index to the new host index, consuming each packet.
    fn kick_ch12(&mut self, host_idx: u16) {
        if self.w(R_BE_HAXI_INIT_CFG1) & B_BE_TXDMA_EN == 0 {
            return; // the engine is off: nothing moves
        }
        if self.fw_running && self.faults.rf_command_stall {
            return;
        }
        let regs = TxChannelRegs::of(TXCH_FWCMD);
        let len =
            self.mem[regs.num as usize] as u16 | (self.mem[regs.num as usize + 1] as u16) << 8;
        let base = self.w(regs.desa_l) as u64 | (self.w(regs.desa_l + 4) as u64) << 32;
        while self.ch12_rp != host_idx {
            let bd_at = base + u64::from(self.ch12_rp) * TX_BD_BYTES as u64;
            // SAFETY: `translate` proved the range lies in a buffer the
            // driver leased, which lives for the mock's lifetime.
            let bd = unsafe {
                core::slice::from_raw_parts(self.translate(bd_at, TX_BD_BYTES), TX_BD_BYTES)
            };
            let length = u16::from_le_bytes([bd[0], bd[1]]) as usize;
            let opt = u16::from_le_bytes([bd[2], bd[3]]);
            let dma = u32::from_le_bytes([bd[4], bd[5], bd[6], bd[7]]) as u64
                | ((opt as u32 & RTW89_PCI_TXBD_OPT_DMA_HI) as u64 >> 6) << 32;
            assert!(
                opt as u32 & RTW89_PCI_TXBD_OPT_LS != 0,
                "channel 12 never chains"
            );
            // SAFETY: as above, for the packet the descriptor names.
            let pkt = unsafe { core::slice::from_raw_parts(self.translate(dma, length), length) };
            self.consume(pkt);
            if self.faults.mac_command_stall
                && pkt[FWCMD_DESC_BYTES] & 3 == 1
                && matches!(pkt[FWCMD_DESC_BYTES] >> 2, 9 | 12)
            {
                return;
            }
            self.ch12_rp = (self.ch12_rp + 1) % len;
        }
        let idx = encode(host_idx as u32, TXBD_HOST_IDX_MASK)
            | encode(self.ch12_rp as u32, TXBD_HW_IDX_MASK);
        self.set(regs.idx, idx);
    }

    /// Emit through the actual posted RXQ0 descriptors, with tag last.
    fn c2h(&mut self, func: u32, ack: u32) {
        let r = crate::regs::RXQ0;
        let idx = self.w(r.idx);
        let slot = get(idx, TXBD_HW_IDX_MASK) as u16;
        let host = get(idx, TXBD_HOST_IDX_MASK) as u16;
        assert_ne!(slot, host, "mock RX ring full");
        let base = u64::from(self.w(r.desa_l)) | u64::from(self.w(r.desa_l + 4)) << 32;
        // SAFETY: the mock IOMMU validates both the posted descriptor and buffer.
        let bd = unsafe {
            core::slice::from_raw_parts(self.translate(base + u64::from(slot) * 8, 8), 8)
        };
        let size = u16::from_le_bytes([bd[0], bd[1]]) as usize;
        let opt = u16::from_le_bytes([bd[2], bd[3]]) as u32;
        let dma = u64::from(u32::from_le_bytes(bd[4..8].try_into().unwrap()))
            | u64::from((opt & RTW89_PCI_RXBD_OPT_DMA_HI) >> 6) << 32;
        assert!(size >= 40);
        let out = self.translate(dma, 40) as *mut u8;
        let tag = if self.faults.c2h_bad_tag && func == 1 {
            0
        } else {
            self.rx_tag
        };
        let words = [
            (if self.faults.c2h_bad_len && func == 1 {
                39
            } else {
                40
            }) | RTW89_PCI_RXBD_LS
                | u32::from(tag) << 16,
            12 | RTW89_CORE_RX_TYPE_C2H << 24,
            0,
            0,
            0,
            0,
            0,
            1 | func << 8,
            12,
            ack,
        ];
        for i in (0..10).rev() {
            // SAFETY: the translated range holds all ten little-endian words.
            unsafe {
                core::ptr::copy_nonoverlapping(words[i].to_le_bytes().as_ptr(), out.add(i * 4), 4);
            }
        }
        self.rx_tag = if self.rx_tag == 0x1fff {
            1
        } else {
            self.rx_tag + 1
        };
        self.set(
            r.idx,
            (idx & TXBD_HOST_IDX_MASK) | u32::from((slot + 1) % crate::nic::RX_RING) << 16,
        );
    }

    /// One firmware-command packet: the descriptor, then a header packet
    /// or a firmware chunk.
    fn consume(&mut self, pkt: &[u8]) {
        let dword0 = u32::from_le_bytes([pkt[0], pkt[1], pkt[2], pkt[3]]);
        let pkt_len = get(dword0, BE_RXD_RPKT_LEN_MASK) as usize;
        let kind = get(dword0, BE_RXD_RPKT_TYPE_MASK);
        assert_eq!(
            pkt.len(),
            FWCMD_DESC_BYTES + pkt_len,
            "the descriptor's length is the packet's"
        );
        let body = &pkt[FWCMD_DESC_BYTES..];
        match kind {
            RTW89_CORE_RX_TYPE_H2C => {
                let hdr0 = u32::from_le_bytes([body[0], body[1], body[2], body[3]]);
                let hdr1 = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
                assert_eq!(get(hdr1, H2C_HDR_TOTAL_LEN) as usize, body.len());
                if get(hdr0, H2C_HDR_CAT) == 2 && matches!(get(hdr0, H2C_HDR_CLASS), 8 | 9) {
                    self.rf_commands.push(body.to_vec());
                }
                if self.fw_running {
                    let cat = get(hdr0, H2C_HDR_CAT);
                    let class = get(hdr0, H2C_HDR_CLASS);
                    let func = get(hdr0, H2C_HDR_FUNC);
                    let seq = get(hdr0, H2C_HDR_H2C_SEQ);
                    let identity = cat | class << 2 | func << 8;
                    if hdr1 & H2C_HDR_REC_ACK != 0 {
                        self.c2h(0, identity | seq << 16);
                    }
                    if cat == 1 && matches!(class, 9 | 12) {
                        self.mac_commands.push(body.to_vec());
                        if !self.faults.command_done_silent {
                            let identity = identity
                                ^ if self.faults.command_wrong_identity {
                                    0x100
                                } else {
                                    0
                                };
                            let status = u32::from(self.faults.command_reject.unwrap_or(0));
                            self.c2h(1, identity | status << 16 | seq << 24);
                        }
                    }
                }
                if get(hdr0, H2C_HDR_CLASS) == H2C_CL_MAC_FWDL
                    && get(hdr0, H2C_HDR_FUNC) == H2C_FUNC_MAC_FWHDR_DL
                {
                    let header = body[H2C_HDR_BYTES..].to_vec();
                    self.expected_payload = expected_payload(&header);
                    self.suit_received = 0;
                    self.header_packet = Some(header);
                    let ctrl = self.w(R_BE_WCPU_FW_CTRL);
                    self.set(R_BE_WCPU_FW_CTRL, ctrl | B_BE_DLFW_PATH_RDY);
                }
            }
            RTW89_CORE_RX_TYPE_FWDL => {
                assert!(self.header_packet.is_some(), "a chunk before the header");
                assert!(body.len() <= FWDL_SECTION_PER_PKT_LEN as usize);
                self.received.extend_from_slice(body);
                self.suit_received += body.len();
                if self.suit_received == self.expected_payload {
                    let ctrl = self.w(R_BE_WCPU_FW_CTRL);
                    let ctrl = match self.faults.reject_status {
                        Some(st) => {
                            encode(st as u32, B_BE_WCPU_FWDL_STATUS_MASK)
                                | (ctrl & !B_BE_WCPU_FWDL_STATUS_MASK)
                        }
                        None => {
                            let completed = if ctrl & B_BE_WLANCPU_FWDL_EN != 0 {
                                B_BE_WLANCPU_FWDL_EN
                            } else {
                                B_BE_BBMCU0_FWDL_EN
                            };
                            let remaining = if completed == B_BE_BBMCU0_FWDL_EN
                                && self.faults.baseband_never_ready
                            {
                                ctrl
                            } else {
                                ctrl & !completed
                            };
                            self.fw_running =
                                remaining & (B_BE_WLANCPU_FWDL_EN | B_BE_BBMCU0_FWDL_EN) == 0;
                            let status = if self.fw_running || self.faults.baseband_never_ready {
                                3
                            } else {
                                0
                            };
                            encode(status, B_BE_WCPU_FWDL_STATUS_MASK)
                                | (remaining & !B_BE_WCPU_FWDL_STATUS_MASK)
                        }
                    };
                    self.set(R_BE_WCPU_FW_CTRL, ctrl);
                }
            }
            other => panic!("unknown fwcmd packet type {other}"),
        }
    }
}

/// What the WCPU's ROM expects to receive after a header packet: the sum
/// of the table's section sizes (plus the checksum trailer where
/// flagged) — the header version decides where the table starts.
fn expected_payload(header: &[u8]) -> usize {
    let w = |i: usize| u32::from_le_bytes([header[i], header[i + 1], header[i + 2], header[i + 3]]);
    let hdr_ver = get(w(12), FW_HDR_W3_HDR_VER);
    let (base, num) = match hdr_ver {
        0 => (32, get(w(24), FW_HDR_W6_SEC_NUM)),
        _ => (48, get(w(24), FW_HDR_V1_W6_SEC_NUM)),
    };
    assert_eq!(
        get(w(28), FW_HDR_W7_PART_SIZE),
        FWDL_SECTION_PER_PKT_LEN,
        "the part size is stamped"
    );
    let mut total = 0;
    for i in 0..num as usize {
        let w1 = w(base + i * 16 + 4);
        total += get(w1, FWSECTION_HDR_W1_SEC_SIZE) as usize;
        if w1 & FWSECTION_HDR_W1_CHECKSUM != 0 {
            total += FWDL_SECTION_CHKSUM_LEN as usize;
        }
    }
    total
}

impl Regs for MockRtw89 {
    fn read32(&self, off: usize) -> u32 {
        let st = self.st.borrow();
        if st.faults.dead {
            return u32::MAX;
        }
        let off = off as u32;
        let mut v = st.w(off);
        match off {
            R_BE_SYS_PW_CTRL if !st.faults.power_never_ready => v |= B_BE_RDY_SYSPWR,
            R_BE_HAXI_DMA_BUSY1 if st.faults.dma_busy => v = 0x7FFF | (0x3 << 16),
            _ => {}
        }
        if (off == 0x22c24 && st.faults.rf_busy_path == Some(0))
            || (off == 0x22d24 && st.faults.rf_busy_path == Some(1))
        {
            v |= 1 << 29;
        }
        if off == TxChannelRegs::of(TXCH_FWCMD).idx && st.fw_running {
            if let Some(index) = st.faults.final_tx_index {
                v = crate::regs::replace(v, TXBD_HW_IDX_MASK, index.into());
            }
        }
        v
    }

    fn write32(&self, off: usize, val: u32) {
        let mut st = self.st.borrow_mut();
        let off = off as u32;
        st.trace.push(Write { width: 4, off, val });
        st.set(off, val);
        st.effect(off);
    }
}

impl RegsNarrow for MockRtw89 {
    fn read8(&self, off: usize) -> u8 {
        let st = self.st.borrow();
        if st.faults.dead {
            return u8::MAX;
        }
        st.mem[off]
    }

    fn write8(&self, off: usize, val: u8) {
        let mut st = self.st.borrow_mut();
        st.trace.push(Write {
            width: 1,
            off: off as u32,
            val: val as u32,
        });
        st.mem[off] = val;
        st.effect(off as u32 & !3);
        st.effect(off as u32);
    }

    fn read16(&self, off: usize) -> u16 {
        let st = self.st.borrow();
        if st.faults.dead {
            return u16::MAX;
        }
        u16::from_le_bytes([st.mem[off], st.mem[off + 1]])
    }

    fn write16(&self, off: usize, val: u16) {
        let mut st = self.st.borrow_mut();
        st.trace.push(Write {
            width: 2,
            off: off as u32,
            val: val as u32,
        });
        st.mem[off..off + 2].copy_from_slice(&val.to_le_bytes());
        let off = off as u32;
        if off == TxChannelRegs::of(TXCH_FWCMD).idx {
            st.kick_ch12(val);
        } else {
            st.effect(off & !3);
        }
    }
}
