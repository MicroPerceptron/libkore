//! **The firmware download** — the WCPU released into its boot ROM, the
//! header packet and every section pushed through the firmware-command
//! ring (channel 12), and the status field that says the image was
//! accepted (`__rtw89_fw_download`, fw.c:1795-1843, with `include_bb =
//! false`: the WLAN CPU only, the probe-time posture that D1 is).
//!
//! Channel 12 is the one ring D1 drives, and it is driven here rather
//! than through the substrate's `SubRing`, because its contract is not a
//! ring's: the buffer *is* the packet (no write-descriptor page, dossier
//! §F.2), completion is the hardware index in the same register the
//! doorbell is written to, and the only consumer is this download. The
//! [`FwcmdQueue`] owns that arithmetic; a later H2C over DMA reuses it.

use core::sync::atomic::{Ordering, fence};

use kore_driver::seam::{Dma, DmaWrite, RegsNarrow};

use crate::Waits;
use crate::desc::{FWCMD_DESC_BYTES, FwcmdKind, H2C_HDR_BYTES, fwcmd_desc, h2c_header, tx_bd};
use crate::fw::{HEADER_PACKET_MAX, Plan};
use crate::regs::be::*;
use crate::regs::{SECURE_BOOT_MALLOC_INFO_8922A, TXCH_FWCMD, TxChannelRegs, get, reg};

/// Bytes one staging slot holds: the 24-byte descriptor, the 8-byte
/// command header and a payload of at most `FWDL_SECTION_PER_PKT_LEN`
/// (2020) — 2052 bytes, rounded to the next power of two so slot
/// arithmetic is a shift and a header packet (at most
/// [`HEADER_PACKET_MAX`] + 32) fits with room.
pub const SLOT_BYTES: usize = 4096;
const _: () =
    assert!(FWCMD_DESC_BYTES + H2C_HDR_BYTES + FWDL_SECTION_PER_PKT_LEN as usize <= SLOT_BYTES);
const _: () = assert!(FWCMD_DESC_BYTES + H2C_HDR_BYTES + HEADER_PACKET_MAX <= SLOT_BYTES);
/// Staging slots: packets that may be in flight before the oldest slot is
/// reused. A 1.4 MB image is ~700 packets; 32 in flight keeps the WCPU's
/// consumer ahead of the host's producer without a per-packet round trip,
/// and 128 KiB of the row's 2 MiB pool is the price.
pub const SLOTS: usize = 32;
/// Bytes the staging area occupies.
pub const STAGING_BYTES: usize = SLOTS * SLOT_BYTES;

/// Nanoseconds the hardware may take to consume the oldest in-flight
/// packet before the download is declared stalled. The WCPU's ROM pulls
/// a 2 KiB packet in microseconds; a second of silence is a channel that
/// is not being read.
const SLOT_BUDGET_NS: u64 = 1_000_000_000;
/// `rtw89_fwdl_check_path_ready_be` (mac_be.c:573-575): 1 s.
const PATH_READY_NS: u64 = 1_000_000_000;
/// `FWDL_WAIT_CNT` probes of 1 µs (fw.c:112-114): 0.4 s.
const STATUS_NS: u64 = FWDL_WAIT_CNT as u64 * 1_000;
/// The `mdelay(5)` between the download and the FreeRTOS-ready check
/// (fw.c:1830).
const SETTLE_NS: u64 = 5_000_000;
/// `rtw89_fw_download` retries the whole download this many times
/// (fw.c:1851).
const ATTEMPTS: u32 = 5;
/// The hardware status codes of `R_BE_WCPU_FW_CTRL[29:26]`
/// (`fwdl_status_map`, mac_be.c:519-530).
const STATUS_INIT_RDY: u32 = 3;
const STATUS_CHECKSUM_FAIL: u32 = 4;
const STATUS_SECURITY_FAIL_A: u32 = 5;
const STATUS_SECURITY_FAIL_B: u32 = 6;
const STATUS_CV_NOT_MATCH: u32 = 7;

/// Why the download did not complete.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FwdlError {
    /// `H2C_PATH_RDY` never rose after the CPU was released.
    H2cPathNotReady,
    /// `DLFW_PATH_RDY` never rose after the header packet.
    DlfwPathNotReady,
    /// The hardware stopped consuming channel 12.
    Stalled,
    /// The WCPU rejected the image's checksum.
    ChecksumFail,
    /// The WCPU rejected the image's signature.
    SecurityFail,
    /// The image is for another cut.
    CvNotMatch,
    /// Another status code, carried.
    Unexpected(u8),
    /// The status never reached ready within the budget.
    NotReady,
}

/// Retained channel cursor; moving it preserves the live ring across phases.
/// Buffers remain owned by the radio. Only a consumed queue produces this state.
pub(crate) struct FwcmdState {
    posted: u64,
    consumed: u64,
    hw_idx: u16,
}

impl FwcmdState {
    pub(crate) fn retired(&self, ring_len: u16) -> bool {
        self.posted == self.consumed && u64::from(self.hw_idx) == self.posted % u64::from(ring_len)
    }
}

/// The firmware-command ring and its staging slots.
///
/// `bds` holds channel 12's buffer descriptors at `bd_at`, `ring_len` of
/// them; `staging` holds [`SLOTS`] packet buffers. Both are the core's
/// allocations from the row's pool, so `iova` is the device's view of
/// the same bytes.
pub struct FwcmdQueue<'a, B: Dma> {
    bds: &'a B,
    bd_at: usize,
    ring_len: u16,
    staging: &'a B,
    /// Descriptors posted, ever. The host index is `posted % ring_len`,
    /// the slot `posted % SLOTS`.
    posted: u64,
    /// Descriptors the hardware has passed, ever.
    consumed: u64,
    /// The hardware index last read (`bd_ring->rp`).
    hw_idx: u16,
}

impl<'a, B: Dma> FwcmdQueue<'a, B> {
    /// Over a freshly reset ring (both indices zero).
    pub fn new(bds: &'a B, bd_at: usize, ring_len: u16, staging: &'a B) -> Self {
        debug_assert!(staging.len() >= STAGING_BYTES);
        Self {
            bds,
            bd_at,
            ring_len,
            staging,
            posted: 0,
            consumed: 0,
            hw_idx: 0,
        }
    }

    pub(crate) fn into_state(self) -> FwcmdState {
        FwcmdState {
            posted: self.posted,
            consumed: self.consumed,
            hw_idx: self.hw_idx,
        }
    }

    /// Fold the hardware index into the consumed count
    /// (`rtw89_pci_dma_recalc`, pci.c:61-82, the transmit branch).
    fn reclaim<R: RegsNarrow>(&mut self, regs: &R) -> bool {
        let cur = get(
            regs.read32(reg(TxChannelRegs::of(TXCH_FWCMD).idx)),
            TXBD_HW_IDX_MASK,
        ) as u16;
        if cur >= self.ring_len {
            return false;
        }
        let advanced = if cur >= self.hw_idx {
            cur - self.hw_idx
        } else {
            self.ring_len - (self.hw_idx - cur)
        };
        if u64::from(advanced) > self.posted - self.consumed {
            return false;
        }
        self.consumed += u64::from(advanced);
        self.hw_idx = cur;
        true
    }

    /// Wait until a staging slot is free.
    fn wait_slot<R: RegsNarrow>(&mut self, regs: &R, waits: &Waits) -> bool {
        let mut budget = waits.budget(SLOT_BUDGET_NS);
        loop {
            if !self.reclaim(regs) {
                return false;
            }
            if self.posted - self.consumed < SLOTS as u64 {
                return true;
            }
            if !budget.may_wait() {
                return self.reclaim(regs) && self.posted - self.consumed < SLOTS as u64;
            }
            (waits.pause)();
        }
    }

    /// Post one packet: the descriptor, an optional command header, the
    /// payload, and — under secure boot — the key over the payload's tail
    /// (`__rtw89_fw_download_main`, fw.c:1648-1659), then the doorbell.
    pub fn post<R: RegsNarrow>(
        &mut self,
        regs: &R,
        waits: &Waits,
        kind: FwcmdKind,
        header: Option<[u8; H2C_HDR_BYTES]>,
        payload: &[u8],
        key: Option<&[u8]>,
    ) -> Result<(), FwdlError> {
        if !self.wait_slot(regs, waits) {
            return Err(FwdlError::Stalled);
        }
        let slot = (self.posted % SLOTS as u64) as usize;
        let at = slot * SLOT_BYTES;
        let hdr_len = header.map_or(0, |_| H2C_HDR_BYTES);
        let pkt_len = hdr_len + payload.len();
        let total = FWCMD_DESC_BYTES + pkt_len;
        debug_assert!(total <= SLOT_BYTES);
        let staging = self.staging;
        let mut off = at;
        staging
            .write_bytes_at(off, &fwcmd_desc(pkt_len as u16, kind))
            .expect("a slot holds a packet");
        off += FWCMD_DESC_BYTES;
        if let Some(h) = header {
            staging
                .write_bytes_at(off, &h)
                .expect("a slot holds a packet");
            off += H2C_HDR_BYTES;
        }
        staging
            .write_bytes_at(off, payload)
            .expect("a slot holds a packet");
        if let Some(key) = key {
            staging
                .write_bytes_at(off + payload.len() - key.len(), key)
                .expect("the key fits the payload");
        }
        let bd_index = (self.posted % u64::from(self.ring_len)) as usize;
        let iova = staging.iova().0 + at as u64;
        self.bds
            .write_bytes_at(
                self.bd_at + bd_index * crate::desc::TX_BD_BYTES,
                &tx_bd(total as u16, iova),
            )
            .expect("the ring holds its descriptors");
        self.posted += 1;
        // The packet and its descriptor are visible before the doorbell.
        fence(Ordering::Release);
        regs.write16(
            reg(TxChannelRegs::of(TXCH_FWCMD).idx),
            (self.posted % u64::from(self.ring_len)) as u16,
        );
        Ok(())
    }

    /// Packets posted so far (for the host tests).
    pub fn posted(&self) -> u64 {
        self.posted
    }

    /// One bounded observation; lets a control wait drain RX concurrently.
    pub(crate) fn poll_retired<R: RegsNarrow>(&mut self, regs: &R) -> Result<bool, FwdlError> {
        if !self.reclaim(regs) {
            return Err(FwdlError::Stalled);
        }
        Ok(self.consumed == self.posted)
    }

    /// Firmware readiness is not TX descriptor retirement. Before changing
    /// the DLE partition, require the final hardware consumer index as well.
    /// Invalid or regressing indices refuse without releasing staging.
    pub fn retire<R: RegsNarrow>(&mut self, regs: &R, waits: &Waits) -> Result<(), FwdlError> {
        let mut budget = waits.budget(SLOT_BUDGET_NS);
        loop {
            if !self.reclaim(regs) {
                return Err(FwdlError::Stalled);
            }
            if self.consumed == self.posted {
                return Ok(());
            }
            if !budget.may_wait() {
                return Err(FwdlError::Stalled);
            }
            (waits.pause)();
        }
    }
}

/// `rtw89_mac_disable_cpu_be` (mac_be.c:427-446).
fn disable_cpu<R: RegsNarrow>(regs: &R) {
    let v = regs.read32(reg(R_BE_PLATFORM_ENABLE));
    regs.write32(reg(R_BE_PLATFORM_ENABLE), v & !B_BE_WCPU_EN);
    let v = regs.read32(reg(R_BE_PLATFORM_ENABLE));
    regs.write32(reg(R_BE_PLATFORM_ENABLE), v | B_BE_HOLD_AFTER_RESET);
    let v = regs.read32(reg(R_BE_PLATFORM_ENABLE));
    regs.write32(reg(R_BE_PLATFORM_ENABLE), v | B_BE_WCPU_EN);
    let v = regs.read32(reg(R_BE_WCPU_FW_CTRL));
    regs.write32(reg(R_BE_WCPU_FW_CTRL), v & B_BE_RUN_ENV_MASK);
    let v = regs.read32(reg(R_BE_DCPU_PLATFORM_ENABLE));
    regs.write32(reg(R_BE_DCPU_PLATFORM_ENABLE), v | B_BE_DCPU_PLATFORM_EN);
    regs.write32(reg(R_BE_UDM0), 0);
    regs.write32(reg(R_BE_HALT_C2H), 0);
    regs.write32(reg(R_BE_UDM2), 0);
}

/// `rtw89_mac_fwdl_enable_wcpu_be(boot_reason = 0, dlfw = true,
/// include_bb = false)` (mac_be.c:448-516): the WLAN CPU's download
/// enable, then `wcpu_on` — the debug registers cleared, the halt
/// interrupt acknowledged, the CPU clock on, the watchdog wakes off, the
/// boot reason zero, and the CPU released.
fn enable_wcpu<R: RegsNarrow>(regs: &R, include_bb: bool) {
    let v = regs.read32(reg(R_BE_WCPU_FW_CTRL));
    let enables = B_BE_WLANCPU_FWDL_EN | if include_bb { B_BE_BBMCU0_FWDL_EN } else { 0 };
    regs.write32(reg(R_BE_WCPU_FW_CTRL), v | enables);
    // The three "AON L2 debug register not empty" reads are warnings
    // upstream; the writes that follow clear them either way.
    regs.write32(reg(R_BE_UDM1), 0);
    regs.write32(reg(R_BE_UDM2), 0);
    regs.write32(reg(R_BE_HALT_H2C), 0);
    regs.write32(reg(R_BE_HALT_C2H), 0);
    regs.write32(reg(R_BE_HALT_H2C_CTRL), 0);
    regs.write32(reg(R_BE_HALT_C2H_CTRL), 0);
    let _ = regs.read32(reg(R_BE_HISR0));
    regs.write32(reg(R_BE_HISR0), B_BE_HALT_C2H_INT);
    let v = regs.read32(reg(R_BE_SYS_CLK_CTRL));
    regs.write32(reg(R_BE_SYS_CLK_CTRL), v | B_BE_CPU_CLK_EN);
    let v = regs.read32(reg(R_BE_SYS_CFG5));
    regs.write32(
        reg(R_BE_SYS_CFG5),
        v & !(B_BE_WDT_WAKE_PCIE_EN | B_BE_WDT_WAKE_USB_EN),
    );
    let v = regs.read32(reg(R_BE_WCPU_FW_CTRL));
    regs.write32(
        reg(R_BE_WCPU_FW_CTRL),
        v & !(B_BE_WDT_PLT_RST_EN | B_BE_WCPU_ROM_CUT_GET),
    );
    let v = regs.read16(reg(R_BE_BOOT_REASON));
    regs.write16(reg(R_BE_BOOT_REASON), v & !(B_BE_BOOT_REASON_MASK as u16));
    let v = regs.read32(reg(R_BE_PLATFORM_ENABLE));
    regs.write32(reg(R_BE_PLATFORM_ENABLE), v & !B_BE_WCPU_EN);
    let v = regs.read32(reg(R_BE_PLATFORM_ENABLE));
    regs.write32(reg(R_BE_PLATFORM_ENABLE), v & !B_BE_HOLD_AFTER_RESET);
    let v = regs.read32(reg(R_BE_PLATFORM_ENABLE));
    regs.write32(reg(R_BE_PLATFORM_ENABLE), v | B_BE_WCPU_EN);
}

/// The status field's verdict when the ready poll expired
/// (`rtw89_fw_check_rdy`, fw.c:116-132).
fn status_error(ctrl: u32) -> FwdlError {
    match get(ctrl, B_BE_WCPU_FWDL_STATUS_MASK) {
        STATUS_CHECKSUM_FAIL => FwdlError::ChecksumFail,
        STATUS_SECURITY_FAIL_A | STATUS_SECURITY_FAIL_B => FwdlError::SecurityFail,
        STATUS_CV_NOT_MATCH => FwdlError::CvNotMatch,
        st => FwdlError::Unexpected(st as u8),
    }
}

/// `rtw89_fw_download_suit` (fw.c:1759-1793) + `rtw89_fw_download_main`'s
/// ready check for the WLAN CPU.
fn download_suit<R: RegsNarrow, B: Dma>(
    regs: &R,
    waits: &Waits,
    plan: &Plan<'_>,
    q: &mut FwcmdQueue<'_, B>,
    baseband: bool,
) -> Result<(), FwdlError> {
    if !baseband {
        regs.write32(
            reg(R_BE_SECURE_BOOT_MALLOC_INFO),
            SECURE_BOOT_MALLOC_INFO_8922A,
        );
    }
    if !waits.poll_set(regs, R_BE_WCPU_FW_CTRL, B_BE_H2C_PATH_RDY, PATH_READY_NS) {
        return Err(FwdlError::H2cPathNotReady);
    }
    // The header packet (`rtw89_fw_download_hdr`, fw.c:1586-1609): an H2C
    // of class FWDL, function FWHDR_DL, sequence 0, no acknowledgements.
    let mut hdr = [0u8; HEADER_PACKET_MAX];
    let len = plan
        .header_packet(&mut hdr)
        .expect("a verified plan's header fits its maximum");
    let h2c = h2c_header(
        H2C_CAT_MAC,
        H2C_CL_MAC_FWDL,
        H2C_FUNC_MAC_FWHDR_DL,
        0,
        len as u16,
        false,
        false,
    );
    q.post(regs, waits, FwcmdKind::H2c, Some(h2c), &hdr[..len], None)?;
    if !waits.poll_set(regs, R_BE_WCPU_FW_CTRL, B_BE_DLFW_PATH_RDY, PATH_READY_NS) {
        return Err(FwdlError::DlfwPathNotReady);
    }
    regs.write32(reg(R_BE_HALT_H2C_CTRL), 0);
    regs.write32(reg(R_BE_HALT_C2H_CTRL), 0);

    // The sections (`__rtw89_fw_download_main`, fw.c:1611-1673).
    for section in plan.sections() {
        if section.ignore {
            continue;
        }
        let key = section.copies_key();
        let mut rest = &section.data[..section.push_len()];
        while !rest.is_empty() {
            let n = rest.len().min(FWDL_SECTION_PER_PKT_LEN as usize);
            let (chunk, tail) = rest.split_at(n);
            q.post(regs, waits, FwcmdKind::Fwdl, None, chunk, key)?;
            rest = tail;
        }
    }

    // `RTW89_FWDL_CHECK_WCPU_FWDL_DONE`: the download-enable bit dropped,
    // or the status already at ready (`fwdl_get_status_be`).
    let done = |ctrl: u32| {
        if baseband {
            // A WLAN-ready status must never stand in for the BB-MCU ack.
            ctrl & B_BE_BBMCU0_FWDL_EN == 0
        } else {
            ctrl & B_BE_WLANCPU_FWDL_EN == 0
                || get(ctrl, B_BE_WCPU_FWDL_STATUS_MASK) == STATUS_INIT_RDY
        }
    };
    if !waits.poll_until(regs, R_BE_WCPU_FW_CTRL, STATUS_NS, done) {
        return Err(status_error(regs.read32(reg(R_BE_WCPU_FW_CTRL))));
    }
    Ok(())
}

/// `rtw89_fw_download(RTW89_FW_NORMAL, include_bb = false)`: up to five
/// attempts, each releasing the CPU afresh, pushing the image, settling,
/// and waiting for FreeRTOS to report ready in the status field.
pub fn download<R: RegsNarrow, B: Dma>(
    regs: &R,
    waits: &Waits,
    plan: &Plan<'_>,
    q: &mut FwcmdQueue<'_, B>,
) -> Result<(), FwdlError> {
    download_images(regs, waits, plan, None, q)
}

/// Full 8922A firmware download: enable BB-MCU0 with WCPU, send the WLAN
/// suit first and then the cut-selected BB suit, checking each CPU's own
/// download bit before the final FreeRTOS ready verdict (fw.c:1797–1843).
pub fn download_with_baseband<R: RegsNarrow, B: Dma>(
    regs: &R,
    waits: &Waits,
    wlan: &Plan<'_>,
    baseband: &Plan<'_>,
    q: &mut FwcmdQueue<'_, B>,
) -> Result<(), FwdlError> {
    download_images(regs, waits, wlan, Some(baseband), q)
}

fn download_images<R: RegsNarrow, B: Dma>(
    regs: &R,
    waits: &Waits,
    plan: &Plan<'_>,
    baseband: Option<&Plan<'_>>,
    q: &mut FwcmdQueue<'_, B>,
) -> Result<(), FwdlError> {
    let mut last = FwdlError::NotReady;
    for _ in 0..ATTEMPTS {
        disable_cpu(regs);
        enable_wcpu(regs, baseband.is_some());
        if let Err(e) = download_suit(regs, waits, plan, q, false) {
            last = e;
            continue;
        }
        if let Some(bb) = baseband {
            if let Err(e) = download_suit(regs, waits, bb, q, true) {
                last = e;
                continue;
            }
        }
        waits.delay(SETTLE_NS);
        let ready = |ctrl: u32| get(ctrl, B_BE_WCPU_FWDL_STATUS_MASK) == STATUS_INIT_RDY;
        if waits.poll_until(regs, R_BE_WCPU_FW_CTRL, STATUS_NS, ready) {
            return Ok(());
        }
        last = status_error(regs.read32(reg(R_BE_WCPU_FW_CTRL)));
    }
    Err(last)
}
