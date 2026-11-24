//! **The bring-up ladder.**
//!
//! Two affine states, the e1000e / NVMe template: [`Prepared`] is the
//! radio powered on, its efuse read, the firmware verified and planned,
//! the rings carved and pointed at, with **the DMA engines still
//! disabled** — the state a capability host reaches before the device may
//! master memory (`DmaPrepared`); [`Rtw89`] is the running radio,
//! firmware loaded and answering. The core stores no register window;
//! every method takes `regs`, because in the row the BAR lives in the
//! device's resource set.
//!
//! # The carve
//!
//! Four allocations from the row's 2 MiB pool, page-aligned:
//!
//! | buffer | bytes | what |
//! |---|---|---|
//! | transmit BD rings | 13 × 4 KiB | one page per channel, 256 × 8-byte descriptors each; only channel 12 is ever posted, the twelve data channels stay stopped but programmed, as the engine expects every ring to exist |
//! | receive BD rings | 2 × 4 KiB | RXQ0 and RPQ0, [`RX_RING`] descriptors each |
//! | receive buffers | 2 × 8 × 12 KiB | one buffer per receive descriptor, upstream's 11 498-byte maximum rounded to pages |
//! | staging | 32 × 4 KiB | the firmware-command packets in flight |
//!
//! 316 KiB in all. Upstream posts 256 receive buffers per ring; D1 posts
//! eight, because nothing drains them yet — the witness rides the register
//! mailbox — and a ring the firmware fills is a descriptor-unavailable
//! event, not a fault.

use core::sync::atomic::{Ordering, fence};

use kore_driver::seam::{Dma, DmaPool, DmaWrite, RegsNarrow};

use crate::desc::{RX_BD_BYTES, addressable, rx_bd};
use crate::efuse::{RfEfuse, read_rf, read_secure_boot};
use crate::fw::{Digest, FirmwareImage, Plan, SecureBoot, Version};
use crate::fwdl::{FwcmdQueue, STAGING_BYTES};
use crate::mac::{
    RingSetup, dmac_pre_init, pci_pre_deinit, pci_pre_init, power_off, power_on, trxdma_enable,
};
use crate::mailbox::{Counters, PhyCap, get_phy_cap};
use crate::product::Product;
use crate::regs::be::*;
use crate::regs::{RX_CHANNELS, TX_CHANNELS, TXCH_FWCMD, controller_present, cut_version};
use crate::{Error, Result, Waits};

/// Transmit descriptors per channel: `RTW89_PCI_TXBD_NUM_MAX` (pci.h:1051),
/// what the 16-bit `num` registers are programmed with upstream, so the
/// hardware index arithmetic (12-bit fields) is exercised the same way.
pub const TX_RING: u16 = RTW89_PCI_TXBD_NUM_MAX as u16;
/// Receive descriptors per queue at D1 (see the module docs).
pub const RX_RING: u16 = 8;
/// Bytes per receive buffer: `RTW89_PCI_RX_BUF_SIZE` = 11454 + 40 + 4
/// (pci.h:1057: the largest frame, a long descriptor, the info word)
/// rounded up to the page.
pub const RX_BUF_BYTES: usize = (11454 + 40 + 4usize).next_multiple_of(PAGE);
/// The carve's granule: the pool's guarantee, and what keeps every ring
/// inside one page.
const PAGE: usize = 4096;
const _: () = assert!(TX_RING as usize * crate::desc::TX_BD_BYTES <= PAGE);
const _: () = assert!(RX_RING as usize * RX_BD_BYTES <= PAGE);
/// Bytes the transmit BD rings occupy.
const TX_BDS_BYTES: usize = TX_CHANNELS * PAGE;
/// Bytes the receive BD rings occupy.
const RX_BDS_BYTES: usize = RX_CHANNELS * PAGE;
/// Bytes the receive buffers occupy.
const RX_BUFS_BYTES: usize = RX_CHANNELS * RX_RING as usize * RX_BUF_BYTES;
const _: () = assert!(TX_BDS_BYTES + RX_BDS_BYTES + RX_BUFS_BYTES + STAGING_BYTES <= 0x20_0000 / 4);

/// The radio powered on, its efuse read, the firmware planned, the rings
/// carved and pointed at, both DMA engines disabled.
pub struct Prepared<P: DmaPool> {
    product: Product,
    cut: u8,
    secure: SecureBoot,
    efuse: RfEfuse,
    plan: Plan<'static>,
    baseband: Option<Plan<'static>>,
    phy: Option<crate::fw::phy::PhyTables<'static>>,
    tx_bds: P::Buf,
    rx_bds: P::Buf,
    rx_bufs: P::Buf,
    staging: P::Buf,
}

/// Draw one buffer and prove the part can address it.
fn draw<P: DmaPool>(pool: &P, bytes: usize) -> Result<P::Buf> {
    let buf = pool.alloc(bytes).ok_or(Error::DmaAlloc)?;
    if !addressable(buf.iova().0, buf.len() as u64) {
        return Err(Error::DmaAlloc);
    }
    Ok(buf)
}

impl<P: DmaPool> Prepared<P> {
    /// **Bring the radio to the prepared state.** Each step is a distinct
    /// refusal; after the MAC is powered, a refusal powers it off again
    /// before returning, so a re-founding starts from a cold part.
    ///
    /// `bar_len` is BAR2's mapped length; `firmware` the whole leased boot
    /// module (page-rounded, sliced to `size` here) or `None` on a boot
    /// without one; `digest` and `size` the compiled manifest citation;
    /// `waits` the wait tier's hooks.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare<R: RegsNarrow>(
        regs: &R,
        pool: &P,
        product: Product,
        bar_len: usize,
        firmware: Option<&'static [u8]>,
        digest: &Digest,
        size: usize,
        waits: Waits,
    ) -> Result<Self> {
        Self::prepare_images(
            regs, pool, product, bar_len, firmware, digest, size, waits, false,
        )
    }

    /// Prepare the cut-selected WLAN and BB-MCU0 suits. Missing or malformed
    /// elements refuse before power-on; both plans precede DMA activation.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_with_baseband<R: RegsNarrow>(
        regs: &R,
        pool: &P,
        product: Product,
        bar_len: usize,
        firmware: Option<&'static [u8]>,
        digest: &Digest,
        size: usize,
        waits: Waits,
    ) -> Result<Self> {
        Self::prepare_images(
            regs, pool, product, bar_len, firmware, digest, size, waits, true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_images<R: RegsNarrow>(
        regs: &R,
        pool: &P,
        product: Product,
        bar_len: usize,
        firmware: Option<&'static [u8]>,
        digest: &Digest,
        size: usize,
        waits: Waits,
        include_bb: bool,
    ) -> Result<Self> {
        // 1. Does the window decode a controller at all?
        if (include_bb && bar_len < crate::mac::BASEBAND_REGS_SPAN)
            || !controller_present(regs, bar_len)
        {
            return Err(Error::NotPresent);
        }
        let cut = cut_version(regs);

        // 2–3. The trust decision, before the radio is touched: the blob
        //      is the pinned one and carries an image for this cut.
        let module = firmware.ok_or(Error::NoFirmware)?;
        let blob = module.get(..size).ok_or(Error::BadFirmware)?;
        let image = FirmwareImage::verified(blob, digest).map_err(|_| Error::BadFirmware)?;
        let suit = image.suit(cut).map_err(|_| Error::BadFirmware)?;

        let bb_suit = if include_bb {
            Some(image.bbmcu0(cut).map_err(|_| Error::BadFirmware)?)
        } else {
            None
        };

        // 4. Power.
        power_on(regs, &waits, cut).map_err(|_| Error::PowerOn)?;
        let powered = |e: Error| {
            let _ = power_off(regs, &waits, cut);
            e
        };

        // 5. The efuse: secure boot (what the plan's key selection needs)
        //    and the RF block (the regulatory root).
        let secure = read_secure_boot(regs, &waits, cut).map_err(|_| powered(Error::Efuse))?;
        let efuse = read_rf(regs, &waits, cut).map_err(|_| powered(Error::Efuse))?;

        // 3, again: the header, now that the key selection is known.
        let plan = suit
            .plan(&secure)
            .map_err(|_| powered(Error::BadFirmware))?;

        let baseband = bb_suit
            .map(|s| s.plan(&secure))
            .transpose()
            .map_err(|_| powered(Error::BadFirmware))?;

        // Validate every selected/unselected PHY record before any DMA
        // allocation, so malformed tails cannot leave a programmed prefix.
        let phy = if include_bb {
            Some(
                image
                    .phy_tables(efuse.rfe_type, cut)
                    .map_err(|_| powered(Error::BadFirmware))?,
            )
        } else {
            None
        };

        // 6. The carve.
        let tx_bds = draw(pool, TX_BDS_BYTES).map_err(powered)?;
        let rx_bds = draw(pool, RX_BDS_BYTES).map_err(powered)?;
        let rx_bufs = draw(pool, RX_BUFS_BYTES).map_err(powered)?;
        let staging = draw(pool, STAGING_BYTES).map_err(powered)?;
        // Every receive descriptor points at its buffer; the rings are
        // zero (the pool zeroes), which is what a fresh ring must read.
        for q in 0..RX_CHANNELS {
            for slot in 0..RX_RING as usize {
                let buf = (q * RX_RING as usize + slot) * RX_BUF_BYTES;
                let iova = rx_bufs.iova().0 + buf as u64;
                rx_bds
                    .write_bytes_at(
                        q * PAGE + slot * RX_BD_BYTES,
                        &rx_bd(RX_BUF_BYTES as u16, iova),
                    )
                    .expect("the ring page holds its descriptors");
            }
        }
        let rings = RingSetup {
            tx: core::array::from_fn(|ch| tx_bds.iova().0 + (ch * PAGE) as u64),
            tx_len: TX_RING,
            rx: core::array::from_fn(|q| rx_bds.iova().0 + (q * PAGE) as u64),
            rx_len: RX_RING,
        };

        // 7. The engine's pre-init, up to the enable.
        if include_bb {
            crate::mac::baseband_pre_init(regs, &waits);
        }
        dmac_pre_init(regs, &waits).map_err(|_| powered(Error::DmaInit))?;
        pci_pre_init(regs, &waits, cut, &rings).map_err(|_| powered(Error::DmaInit))?;

        Ok(Self {
            product,
            cut,
            secure,
            efuse,
            plan,
            baseband,
            phy,
            tx_bds,
            rx_bds,
            rx_bufs,
            staging,
        })
    }

    /// The regulatory root the efuse holds.
    pub fn efuse(&self) -> RfEfuse {
        self.efuse
    }

    /// **Start**: enable both DMA engines, download the firmware, and ask
    /// it the first question. `Ok` is firmware alive.
    pub fn start<R: RegsNarrow>(self, regs: &R, waits: Waits) -> Result<Rtw89<P>> {
        let powered = |e: Error| {
            let _ = power_off(regs, &waits, self.cut);
            e
        };
        // Every descriptor written in `prepare` is visible before the
        // engines may fetch.
        fence(Ordering::Release);
        trxdma_enable(regs);

        // 8. The download.
        let mut queue = FwcmdQueue::new(&self.tx_bds, TXCH_FWCMD * PAGE, TX_RING, &self.staging);
        match &self.baseband {
            Some(bb) => {
                crate::fwdl::download_with_baseband(regs, &waits, &self.plan, bb, &mut queue)
            }
            None => crate::fwdl::download(regs, &waits, &self.plan, &mut queue),
        }
        .map_err(|_| powered(Error::FwDownload))?;
        let packets = queue.posted();

        if self.baseband.is_some() {
            queue
                .retire(regs, &waits)
                .map_err(|_| powered(Error::FwDownload))?;
            crate::mac::normal::init(regs, &waits, self.cut)
                .map_err(|_| powered(Error::MacInit))?;
        }

        let mut control = crate::control::Control::new();
        if let Some(phy) = &self.phy {
            crate::phy::initialize(
                regs,
                &waits,
                self.cut,
                phy,
                &mut queue,
                &mut control,
                &self.rx_bufs,
            )
            .map_err(|_| powered(Error::PhyInit))?;
        }

        if self.baseband.is_some() {
            control
                .initialize(regs, &waits, &mut queue, &self.rx_bufs)
                .map_err(|_| powered(Error::MacCommand))?;
        }
        let fwcmd = queue.into_state();

        // 9. The witness: the register mailbox's `GET_FEATURE`, answered
        //    with `PHY_CAP` by the firmware just loaded. The counters
        //    start from zero after a download (fw.c:1823-1826).
        let mut counters = Counters::default();
        let phy_cap =
            get_phy_cap(regs, &waits, &mut counters).map_err(|_| powered(Error::NoAck))?;

        Ok(Rtw89 {
            product: self.product,
            cut: self.cut,
            secure: self.secure,
            efuse: self.efuse,
            version: self.plan.version(),
            header_version: self.plan.header_version(),
            packets,
            normal_mac: self.baseband.is_some(),
            gain: self.phy.map(|p| p.gain()),
            control,
            fwcmd,
            phy_cap,
            counters,
            tx_bds: self.tx_bds,
            rx_bds: self.rx_bds,
            rx_bufs: self.rx_bufs,
            staging: self.staging,
        })
    }
}

/// The running radio: firmware loaded and answering.
pub struct Rtw89<P: DmaPool> {
    product: Product,
    cut: u8,
    secure: SecureBoot,
    efuse: RfEfuse,
    version: Version,
    header_version: u8,
    packets: u64,
    normal_mac: bool,
    gain: Option<crate::fw::phy::GainTable<'static>>,
    control: crate::control::Control,
    fwcmd: crate::fwdl::FwcmdState,
    phy_cap: PhyCap,
    counters: Counters,
    /// The carve, owned for the engine's lifetime: the device fetches
    /// from these until `quiesce`, and nothing on the host reads them
    /// until D2's frame path does.
    #[cfg_attr(not(test), expect(dead_code, reason = "seam: #388 D2 reads the rings"))]
    tx_bds: P::Buf,
    #[cfg_attr(not(test), expect(dead_code, reason = "seam: #388 D2 reads the rings"))]
    rx_bds: P::Buf,
    #[cfg_attr(not(test), expect(dead_code, reason = "seam: #388 D2 reads the rings"))]
    rx_bufs: P::Buf,
    #[cfg_attr(not(test), expect(dead_code, reason = "seam: #388 D2 reads the rings"))]
    staging: P::Buf,
}

impl<P: DmaPool> Rtw89<P> {
    /// The part.
    pub fn product(&self) -> Product {
        self.product
    }

    /// The cut (`R_AX_SYS_CFG1[15:12]`).
    pub fn cut(&self) -> u8 {
        self.cut
    }

    /// What the efuse said about secure boot.
    pub fn secure_boot(&self) -> SecureBoot {
        self.secure
    }

    /// The regulatory root.
    pub fn efuse(&self) -> RfEfuse {
        self.efuse
    }

    /// The firmware image's version words.
    pub fn firmware_version(&self) -> Version {
        self.version
    }

    /// The container's inner header version the download used (dossier
    /// open question 2, answered by the blob).
    pub fn firmware_header_version(&self) -> u8 {
        self.header_version
    }

    /// Packets the download posted on channel 12 (across attempts).
    pub fn download_packets(&self) -> u64 {
        self.packets
    }

    /// Normal DMAC/CMAC0 configuration completed before the alive reply.
    /// This is a bring-up witness, not PHY or frame-path readiness.
    pub fn normal_mac_ready(&self) -> bool {
        self.normal_mac
    }

    /// BB/RF programming and command TX retirement completed before the
    /// final WLAN reply. This does not prove calibration or BB-MCU liveness.
    pub fn phy_configured(&self) -> bool {
        self.gain.is_some()
    }

    /// All startup MAC feature commands returned successful done acknowledgements.
    pub fn mac_features_ready(&self) -> bool {
        self.control.features_ready() && self.fwcmd.retired(TX_RING)
    }

    /// Borrowed BE gain metadata for later per-channel calibration lookup.
    pub fn gain_table(&self) -> Option<crate::fw::phy::GainTable<'static>> {
        self.gain
    }

    /// What the firmware answered to `GET_FEATURE`.
    pub fn phy_cap(&self) -> PhyCap {
        self.phy_cap
    }

    /// The mailbox counters, for a later exchange.
    pub fn counters(&mut self) -> &mut Counters {
        &mut self.counters
    }

    /// **Quiesce**: stop both engines, clear every index, power the MAC
    /// off. After `Ok` the device masters nothing, so the DMA and BAR
    /// authority above it can be retired without racing a transfer.
    pub fn quiesce<R: RegsNarrow>(&mut self, regs: &R, waits: Waits) -> Result<()> {
        pci_pre_deinit(regs, RX_RING);
        power_off(regs, &waits, self.cut).map_err(|_| Error::PowerOn)
    }

    /// The carve, for the host tests' inspection.
    #[cfg(test)]
    pub(crate) fn buffers(&self) -> [&P::Buf; 4] {
        [&self.tx_bds, &self.rx_bds, &self.rx_bufs, &self.staging]
    }
}

#[cfg(test)]
mod tests {
    use alloc::boxed::Box;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::cell::Cell;

    use kore_driver::seam::DmaPool;

    use super::*;
    use crate::desc::TX_BD_BYTES;
    use crate::efuse::build::rf_map;
    use crate::fw::build::{Sec, digest_of, image, mfw, mss_pool};
    use crate::mailbox::PhyCap;
    use crate::mock::{Faults, IommuBuf, IommuPool, MOCK_PHY_CAP, MockRtw89, Write};
    use crate::regs::{RPQ0, RXQ0, TX_STOP_DATA, TX_STOP_UNUSED, TxChannelRegs};

    const BAR_LEN: usize = 1 << 20;

    fn waits() -> Waits {
        Waits {
            clock: || None,
            pause: || {},
        }
    }

    // Literal two-suit v0 fixture: MFW table + WLAN, then a BB element.
    // Fixed backing keeps the firmware borrow static without a new allocator.
    const DUAL: [u8; 384] = {
        let mut b = [0; 384];
        b[0] = 0xff;
        b[1] = 1;
        b[16] = 1;
        b[17] = 1;
        b[20] = 32;
        b[24] = 64;
        b[100] = 64;
        b[120] = 1;
        let mut n = 0;
        while n < 2 {
            let at = if n == 0 { 32 } else { 128 };
            b[at + 4] = n as u8 + 1;
            b[at + 25] = 1;
            b[at + 33] = 0x10;
            b[at + 36] = 16;
            let mut i = 0;
            while i < 16 {
                b[at + 48 + i] = n as u8 + 0xa0;
                i += 1;
            }
            n += 1;
        }
        // BB (including a bypass), gain metadata, then RF A in slot 1 and
        // RF B in slot 0. Each RF program has analog and direct writes.
        let mut t = 0;
        while t < 4 {
            let at = 192 + t * 48;
            b[at] = t as u8 + 2;
            b[at + 4] = if t == 1 { 8 } else { 16 };
            let pairs: [u32; 4] = match t {
                0 => [0x1000, 42, 0x1004, 0xbabecafe],
                1 => [0, 0, 0, 0],
                _ => [0x12, 0x54321, 0x10012, 0x12343 + t as u32],
            };
            b[at + 24] = if t == 2 { 1 } else { 0 };
            let mut i = 0;
            while i < (b[at + 4] / 4) as usize {
                let bytes = pairs[i].to_le_bytes();
                let mut j = 0;
                while j < 4 {
                    b[at + 32 + i * 4 + j] = bytes[j];
                    j += 1;
                }
                i += 1;
            }
            t += 1;
        }
        b
    };

    fn bring_dual(dev: &MockRtw89) -> Result<Rtw89<IommuPool>> {
        Prepared::prepare_with_baseband(
            dev,
            &dev.pool(),
            Product::Rtl8922ae,
            BAR_LEN,
            Some(&DUAL),
            &digest_of(&DUAL),
            DUAL.len(),
            waits(),
        )?
        .start(dev, waits())
    }

    #[test]
    fn normal_mac_owns_the_scc_partition_but_opens_no_frame_or_interrupt_path() {
        let dev = MockRtw89::new(&efuse(false), Faults::default());
        let mut radio = bring_dual(&dev).unwrap();
        assert!(radio.normal_mac_ready());
        // Literal register readbacks independently check the packed fields.
        assert_eq!(
            crate::regs::get(dev.word(R_BE_WDE_PKTBUF_CFG), B_BE_WDE_FREE_PAGE_NUM_MASK),
            3328
        );
        assert_eq!(
            crate::regs::get(dev.word(R_BE_PLE_PKTBUF_CFG), B_BE_PLE_FREE_PAGE_NUM_MASK),
            2688
        );
        assert_eq!(dev.word(R_BE_WDE_QTA0_CFG), 0x0ce6_0ce6);
        assert_eq!(dev.word(R_BE_WDE_QTA4_CFG), 0x0014_0014);
        assert_eq!(dev.word(R_BE_PLE_QTA0_CFG + 6 * 4), 0x063b_0124);
        assert_eq!(
            crate::regs::get(
                dev.word(R_BE_TXPKTCTL_MPDUINFO_CFG),
                B_BE_MPDUINFO_PKTID_MASK
            ),
            2688
        );
        assert_eq!(
            crate::regs::get(
                dev.word(R_BE_RESP_CSI_RESERVED_PAGE),
                B_BE_CSI_RESERVED_START_PAGE_MASK
            ),
            2690
        );
        assert_eq!(
            crate::regs::get(
                dev.word(R_BE_RESP_CSI_RESERVED_PAGE),
                B_BE_CSI_RESERVED_PAGE_NUM_MASK
            ),
            107
        );
        for offset in [R_BE_MGNT_FLTR, R_BE_CTRL_FLTR, R_BE_DATA_FLTR] {
            assert_eq!(dev.word(offset), 0);
        }
        assert_eq!(
            dev.word(R_BE_HAXI_DMA_STOP1) & (TX_STOP_DATA | TX_STOP_UNUSED),
            TX_STOP_DATA | TX_STOP_UNUSED
        );
        assert_ne!(dev.word(R_BE_FEN_RST_ENABLE) & B_BE_BOOT_RDY0, 0);
        assert_eq!(dev.word(R_BE_HAXI_HIMR00), 0);
        let trace = dev.trace();
        let csi = trace
            .iter()
            .position(|w| w.off == R_BE_RESP_CSI_RESERVED_PAGE)
            .unwrap();
        let mailbox = trace
            .iter()
            .position(|w| w.off == R_BE_H2CREG_DATA0)
            .unwrap();
        assert!(
            csi < mailbox,
            "the alive reply must follow normal MAC initialization"
        );
        assert!(
            trace
                .iter()
                .any(|w| w.off == R_BE_DLK_PROTECT_CTL && w.width == 2)
        );
        assert!(
            trace
                .iter()
                .filter(|w| w.off == R_BE_DLK_PROTECT_CTL)
                .all(|w| w.width == 2)
        );
        assert_eq!(radio.download_packets(), 4);
        radio.quiesce(&dev, waits()).unwrap();
    }

    #[test]
    fn phy_execution_uses_the_phy_aperture_and_physical_rf_classes() {
        let dev = MockRtw89::new(&efuse(false), Faults::default());
        let mut radio = bring_dual(&dev).unwrap();
        assert!(radio.phy_configured());
        assert_eq!(radio.gain_table().unwrap().len(), 1);
        assert_eq!(dev.word(0x21000), 42);
        assert!(!dev.trace().iter().any(|w| w.off == 0x21004)); // bypass
        assert_eq!(dev.word(0x22be0), 0x05432112); // analog B
        assert_eq!(dev.word(0x22ae0), 0x05432112); // analog A
        assert_eq!(dev.word(0x2f048) & 0xfffff, 0x12346);
        assert_eq!(dev.word(0x2e048) & 0xfffff, 0x12345);
        let cmds = dev.rf_commands();
        assert_eq!(cmds.len(), 2);
        for (i, class) in [9, 8].into_iter().enumerate() {
            let h0 = u32::from_le_bytes(cmds[i][..4].try_into().unwrap());
            let h1 = u32::from_le_bytes(cmds[i][4..8].try_into().unwrap());
            assert_eq!(crate::regs::get(h0, H2C_HDR_CAT), 2);
            assert_eq!(crate::regs::get(h0, H2C_HDR_CLASS), class);
            assert_eq!(crate::regs::get(h0, H2C_HDR_FUNC), 0);
            assert_eq!(crate::regs::get(h0, H2C_HDR_H2C_SEQ), i as u32);
            assert_eq!(h1 & H2C_HDR_REC_ACK != 0, i == 0);
            assert_eq!(h1 & H2C_HDR_DONE_ACK, 0);
            assert_eq!(cmds[i][8..], (0x01212346u32 - i as u32).to_le_bytes());
        }
        let trace = dev.trace();
        let bb = trace.iter().position(|w| w.off == 0x21000).unwrap();
        let ready = trace
            .iter()
            .position(|w| w.off == R_BE_FEN_RST_ENABLE && w.val & B_BE_BOOT_RDY0 != 0)
            .unwrap();
        let rf = trace.iter().position(|w| w.off == 0x22be0).unwrap();
        let mailbox = trace
            .iter()
            .position(|w| w.off == R_BE_H2CREG_DATA0)
            .unwrap();
        assert!(bb < ready && ready < rf && rf < mailbox);
        assert_eq!(radio.download_packets(), 4); // RF H2Cs aren't download packets
        radio.quiesce(&dev, waits()).unwrap();
    }

    #[test]
    fn rf_notifications_keep_the_final_partial_page_and_path_sequence() {
        const LARGE: [u8; 4384] = {
            let mut b = [0; 4384];
            let mut i = 0;
            while i < 320 {
                b[i] = DUAL[i];
                i += 1;
            }
            let size = (501u32 * 8).to_le_bytes();
            i = 0;
            while i < 4 {
                b[292 + i] = size[i];
                i += 1;
            }
            i = 0;
            while i < 501 {
                let a = (0x10000u32 + (i as u32 & 255)).to_le_bytes();
                let v = (i as u32).to_le_bytes();
                let mut j = 0;
                while j < 4 {
                    b[320 + i * 8 + j] = a[j];
                    b[324 + i * 8 + j] = v[j];
                    j += 1;
                }
                i += 1;
            }
            i = 0;
            while i < 48 {
                b[4336 + i] = DUAL[336 + i];
                i += 1;
            }
            b
        };
        let dev = MockRtw89::new(&efuse(false), Faults::default());
        let prepared = Prepared::prepare_with_baseband(
            &dev,
            &dev.pool(),
            Product::Rtl8922ae,
            BAR_LEN,
            Some(&LARGE),
            &digest_of(&LARGE),
            LARGE.len(),
            waits(),
        )
        .unwrap();
        let mut radio = prepared.start(&dev, waits()).unwrap();
        let cmds = dev.rf_commands();
        assert_eq!(cmds.len(), 3);
        assert_eq!(cmds[1].len(), 2008);
        assert_eq!(cmds[2].len(), 12);
        for (i, page) in [0, 1].into_iter().enumerate() {
            let h = u32::from_le_bytes(cmds[i + 1][..4].try_into().unwrap());
            assert_eq!(crate::regs::get(h, H2C_HDR_CLASS), 8);
            assert_eq!(crate::regs::get(h, H2C_HDR_FUNC), page);
            assert_eq!(crate::regs::get(h, H2C_HDR_H2C_SEQ), i as u32 + 1);
        }
        assert_eq!(cmds[2][8..], 0x0f4001f4u32.to_le_bytes());
        radio.quiesce(&dev, waits()).unwrap();
    }

    #[test]
    fn phy_busy_and_unretired_commands_refuse_without_publishing_a_radio() {
        for faults in [
            Faults {
                rf_busy_path: Some(0),
                ..Default::default()
            },
            Faults {
                rf_busy_path: Some(1),
                ..Default::default()
            },
            Faults {
                rf_write_stuck: true,
                ..Default::default()
            },
            Faults {
                rf_command_stall: true,
                ..Default::default()
            },
        ] {
            let dev = MockRtw89::new(&efuse(false), faults);
            assert!(matches!(bring_dual(&dev), Err(Error::PhyInit)));
            assert!(!dev.powered());
            assert!(!dev.trace().iter().any(|w| w.off == R_BE_H2CREG_DATA0));
        }
    }

    #[test]
    fn mac_features_require_done_ack_and_retirement_before_the_alive_reply() {
        let dev = MockRtw89::new(&efuse(false), Faults::default());
        let mut radio = bring_dual(&dev).unwrap();
        assert!(radio.mac_features_ready());
        assert_eq!(radio.control.next_sequence(), 5);
        assert!(radio.fwcmd.retired(TX_RING));
        let cmds = dev.mac_commands();
        assert_eq!(cmds.len(), 3);
        for (i, (class, func, payload)) in [
            (12, 2, &0x00000001u32.to_le_bytes()[..]),
            (12, 2, &0x01004001u32.to_le_bytes()[..]),
            (9, 0x14, &[9, 0, 0, 0, 0x5e, 0, 0, 0][..]),
        ]
        .into_iter()
        .enumerate()
        {
            let h0 = u32::from_le_bytes(cmds[i][..4].try_into().unwrap());
            let h1 = u32::from_le_bytes(cmds[i][4..8].try_into().unwrap());
            assert_eq!(crate::regs::get(h0, H2C_HDR_CAT), 1);
            assert_eq!(crate::regs::get(h0, H2C_HDR_CLASS), class);
            assert_eq!(crate::regs::get(h0, H2C_HDR_FUNC), func);
            assert_eq!(crate::regs::get(h0, H2C_HDR_H2C_SEQ), i as u32 + 2);
            assert_ne!(h1 & H2C_HDR_DONE_ACK, 0);
            assert_eq!(h1 & H2C_HDR_REC_ACK != 0, i == 2);
            assert_eq!(&cmds[i][8..], payload);
        }
        let trace = dev.trace();
        let recycled = trace
            .iter()
            .rposition(|w| w.off == RXQ0.idx && w.width == 2)
            .unwrap();
        let mailbox = trace
            .iter()
            .position(|w| w.off == R_BE_H2CREG_DATA0)
            .unwrap();
        assert!(recycled < mailbox);
        assert_eq!(dev.word(R_BE_HAXI_DMA_STOP1) & TX_STOP_DATA, TX_STOP_DATA);
        assert_eq!(dev.word(R_BE_HAXI_HIMR00), 0);
        radio.quiesce(&dev, waits()).unwrap();
    }

    #[test]
    fn mac_command_failures_power_off_without_publishing_the_final_reply() {
        for faults in [
            Faults {
                command_done_silent: true,
                ..Default::default()
            },
            Faults {
                command_reject: Some(7),
                ..Default::default()
            },
            Faults {
                command_wrong_identity: true,
                ..Default::default()
            },
            Faults {
                c2h_bad_tag: true,
                ..Default::default()
            },
            Faults {
                c2h_bad_len: true,
                ..Default::default()
            },
            Faults {
                mac_command_stall: true,
                ..Default::default()
            },
        ] {
            let dev = MockRtw89::new(&efuse(false), faults);
            assert!(matches!(bring_dual(&dev), Err(Error::MacCommand)));
            assert!(!dev.powered());
            assert!(!dev.trace().iter().any(|w| w.off == R_BE_H2CREG_DATA0));
            assert_eq!(dev.mac_commands().len(), 1);
        }
    }

    #[test]
    fn malformed_phy_tail_refuses_before_dma_or_phy_writes() {
        const BAD: [u8; 384] = {
            let mut b = DUAL;
            b[336 + 32 + 2] = 2;
            b
        }; // invalid RF address
        let dev = MockRtw89::new(&efuse(false), Faults::default());
        let result = Prepared::prepare_with_baseband(
            &dev,
            &dev.pool(),
            Product::Rtl8922ae,
            BAR_LEN,
            Some(&BAD),
            &digest_of(&BAD),
            BAD.len(),
            waits(),
        );
        assert!(matches!(result, Err(Error::BadFirmware)));
        assert!(!dev.powered());
        assert!(!dev.trace().iter().any(|w| w.off >= 0x20000));
        assert!(dev.received().is_empty());
    }

    #[test]
    fn each_normal_mac_timeout_refuses_and_powers_off_before_the_alive_reply() {
        for register in [
            R_AX_WDE_INI_STATUS,
            R_AX_PLE_INI_STATUS,
            R_BE_SS_CTRL,
            R_BE_MLO_INIT_CTL,
            R_BE_ADDR_CAM_CTRL,
            R_BE_RESPBA_CAM_CTRL,
        ] {
            let dev = MockRtw89::new(
                &efuse(false),
                Faults {
                    normal_wait_stuck: Some(register),
                    ..Default::default()
                },
            );
            assert!(
                matches!(bring_dual(&dev), Err(Error::MacInit)),
                "register {register:x}"
            );
            assert!(!dev.powered());
            assert!(dev.power_offs() > 0);
            assert!(!dev.trace().iter().any(|w| w.off == R_BE_H2CREG_DATA0));
        }
    }

    #[test]
    fn firmware_ready_does_not_allow_repartitioning_before_final_tx_retirement() {
        // A stalled valid index, an out-of-ring index, and a consumer which
        // advanced beyond the four submitted descriptors are all refused.
        for index in [3, TX_RING, 5] {
            let dev = MockRtw89::new(
                &efuse(false),
                Faults {
                    final_tx_index: Some(index),
                    ..Default::default()
                },
            );
            assert!(matches!(bring_dual(&dev), Err(Error::FwDownload)));
            assert!(
                !dev.trace()
                    .iter()
                    .any(|w| w.off == R_BE_WDE_QTA0_CFG && w.val != 0)
            );
            assert!(!dev.powered());
        }
    }

    #[test]
    fn baseband_preparation_downloads_both_suits_and_requires_its_own_ack() {
        let short = MockRtw89::new(&efuse(false), Faults::default());
        assert!(matches!(
            Prepared::prepare_with_baseband(
                &short,
                &short.pool(),
                Product::Rtl8922ae,
                crate::mac::BASEBAND_REGS_SPAN - 1,
                Some(&DUAL),
                &digest_of(&DUAL),
                DUAL.len(),
                waits()
            ),
            Err(Error::NotPresent)
        ));
        assert_eq!(short.power_ons(), 0);
        for stalled in [false, true] {
            let dev = MockRtw89::new(
                &efuse(false),
                Faults {
                    baseband_never_ready: stalled,
                    ..Default::default()
                },
            );
            let pool = dev.pool();
            let digest = digest_of(&DUAL);
            let prepared = Prepared::prepare_with_baseband(
                &dev,
                &pool,
                Product::Rtl8922ae,
                BAR_LEN,
                Some(&DUAL),
                &digest,
                DUAL.len(),
                waits(),
            )
            .unwrap();
            let result = prepared.start(&dev, waits());
            let trace = dev.trace();
            let memory = trace
                .iter()
                .position(|w| w.off == 0x842c && w.val & 0xffff == 0x7ff9)
                .unwrap();
            let final_cr = trace
                .iter()
                .position(|w| w.off == 0x36820 && w.val == 0xffff_ffff)
                .unwrap();
            let enable = trace
                .iter()
                .position(|w| w.off == R_BE_WCPU_FW_CTRL && w.val & B_BE_BBMCU0_FWDL_EN != 0)
                .unwrap();
            assert!(memory < final_cr && final_cr < enable);
            assert!(
                trace
                    .iter()
                    .any(|w| w.off == 0x36808 && w.val == 0x7654_3210)
            );
            if stalled {
                assert!(matches!(result, Err(Error::FwDownload)));
            } else {
                let mut radio = result.unwrap();
                assert_eq!(radio.download_packets(), 4);
                let mut expected = [0xa0; 32];
                expected[16..].fill(0xa1);
                assert_eq!(dev.received().as_slice(), &expected);
                radio.quiesce(&dev, waits()).unwrap();
            }
        }
    }

    /// A version-1 container: a 3000-byte section, a formatted security
    /// section with two keys, and a 5000-byte section (three packets).
    fn container() -> (&'static [u8], Digest) {
        let pool = mss_pool(&[[0xA1; 512], [0xB2; 512]], 8, &[0b0000_0110]);
        let mut sec_content = vec![0x5Cu8; 600];
        // `sb_sel_ver` at 58 and `key_sign_len` at 60 overlap by design:
        // the selector 0x08000100 carries 0x0800 in its top half, i.e. a
        // 512-byte key.
        sec_content[58..62].copy_from_slice(&0x0800_0100u32.to_le_bytes());
        let secs = [
            Sec::plain(0x1000, (0..3000u32).map(|i| i as u8).collect()),
            Sec {
                dl_addr: 0x2000,
                kind: FWDL_SECURITY_SECTION_TYPE as u8,
                data: sec_content,
                checksum: true,
                w2: FORMATTED_MSSC,
                trailer: pool,
            },
            Sec::plain(0x3000, (0..5000u32).map(|i| (i * 7) as u8).collect()),
        ];
        let version = Version {
            major: 0,
            minor: 35,
            sub: 54,
            idx: 1,
            commit_id: 0xC0FFEE,
        };
        let img = image(1, version, false, &secs);
        let blob: &'static [u8] = Box::leak(mfw(&[(0, 1, 0, img)]).into_boxed_slice());
        (blob, digest_of(blob))
    }

    /// The efuse: the RF facts, and secure boot off or on.
    fn efuse(secure: bool) -> Vec<u8> {
        let mut map = rf_map(*b"US", 0x2A, 3).to_vec();
        map.resize(EFUSE_SEC_BE_START as usize + 4, 0xFF);
        if secure {
            // Selector index 0 (bit 0 and its mirror clear, nibble 0 /
            // 0xF) → `sb_sel_mgn` 0x08000100; device type WINLIN (0xC);
            // customer index bits all set → customer 0; key number
            // 0xF - 0xE = 1.
            let sel: u16 = 0xF0 << 8 | 0b0111_1110;
            let at = EFUSE_SEC_BE_START as usize;
            map[at] = 0xFC;
            map[at + 1] = 0x4E;
            map[at + 2] = sel as u8;
            map[at + 3] = (sel >> 8) as u8;
        }
        map
    }

    /// What the download must have pushed: every kept section, the key
    /// over its last packet's tail where secure boot copies one.
    fn expected_stream(blob: &'static [u8], digest: &Digest, secure: &SecureBoot) -> Vec<u8> {
        let plan = FirmwareImage::verified(blob, digest)
            .unwrap()
            .suit(1)
            .unwrap()
            .plan(secure)
            .unwrap();
        let mut out = Vec::new();
        for s in plan.sections().iter().filter(|s| !s.ignore) {
            let mut data = s.data[..s.push_len()].to_vec();
            if let Some(key) = s.copies_key() {
                let at = data.len() - key.len();
                data[at..].copy_from_slice(key);
            }
            out.extend_from_slice(&data);
        }
        out
    }

    fn bring_up(
        dev: &MockRtw89,
        firmware: Option<&'static [u8]>,
        digest: &Digest,
        size: usize,
    ) -> Result<Rtw89<IommuPool>> {
        let pool = dev.pool();
        let prepared = Prepared::prepare(
            dev,
            &pool,
            Product::Rtl8922ae,
            BAR_LEN,
            firmware,
            digest,
            size,
            waits(),
        )?;
        prepared.start(dev, waits())
    }

    #[test]
    fn the_ladder_reaches_firmware_alive_and_records_the_facts() {
        let (blob, digest) = container();
        let dev = MockRtw89::new(&efuse(false), Faults::default());
        let radio = bring_up(&dev, Some(blob), &digest, blob.len()).expect("firmware alive");

        // The facts.
        assert_eq!(radio.cut(), 1);
        assert_eq!(radio.efuse().country_code, *b"US");
        assert_eq!(radio.efuse().rfe_type, 3);
        assert!(radio.efuse().autoload);
        assert_eq!(radio.secure_boot(), SecureBoot::OFF);
        assert_eq!(radio.firmware_version().minor, 35);
        assert_eq!(radio.firmware_header_version(), 1);
        assert_eq!(radio.phy_cap(), MOCK_PHY_CAP);
        assert!(dev.fw_running());
        assert_eq!(dev.power_ons(), 1);
        assert_eq!(dev.power_offs(), 0);

        // The power sequence: the dossier's 15 crystal writes (rows
        // 21-37 less the two pad-power register writes) in order, and
        // the three enable values.
        let xtal = dev.xtal_writes();
        assert_eq!(xtal.len(), 15);
        assert_eq!(xtal[0], (0xE0, 0x02, 0x02));
        assert_eq!(xtal[14], (0xE1, 0x40, 0x60));
        let trace = dev.trace();
        let has = |w: Write| trace.contains(&w);
        assert!(has(Write {
            width: 4,
            off: R_BE_DMAC_FUNC_EN,
            val: 0x7FFF_4F80
        }));
        assert!(has(Write {
            width: 4,
            off: R_BE_CMAC_SHARE_FUNC_EN,
            val: 0x4000_0007
        }));
        assert!(has(Write {
            width: 4,
            off: R_BE_CMAC_FUNC_EN,
            val: 0x7000_01FF
        }));
        assert!(has(Write {
            width: 1,
            off: R_AX_SCOREBOARD + 3,
            val: 0x81
        }));
        // The A-die patch ran (cut B): the 8-bit and 32-bit views of the
        // PMC register both appear.
        assert!(has(Write {
            width: 4,
            off: R_BE_PMC_DBG_CTRL2,
            val: B_BE_SYSON_DIS_PMCR_BE_WRMSK
        }));

        // The DLE carve for download: WCPU quota 6/6, H2C 256/256.
        assert!(has(Write {
            width: 4,
            off: R_BE_WDE_QTA1_CFG,
            val: 6 | 6 << 16
        }));
        assert!(has(Write {
            width: 4,
            off: R_BE_PLE_QTA0_CFG + 12,
            val: 256 | 256 << 16
        }));
        assert_eq!(dev.word(R_BE_PLE_PKTBUF_CFG) & 0x7F00, 26 << 8);
        assert_eq!(dev.word(R_BE_CH_PAGE_CTRL), 0, "the upstream truncation");
        assert_eq!(
            dev.word(R_BE_HCI_FC_CTRL) & (B_BE_HCI_FC_EN | B_BE_HCI_FC_CH12_EN),
            B_BE_HCI_FC_CH12_EN
        );

        // The rings: every channel's length and base, CH12 open and the
        // data channels stopped, both receive queues full.
        let [tx_bds, rx_bds, rx_bufs, staging] = radio.buffers();
        for ch in 0..TX_CHANNELS {
            let r = TxChannelRegs::of(ch);
            assert_eq!(dev.word(r.num) & 0xFFFF, u32::from(TX_RING));
            let base = dev.word(r.desa_l) as u64 | (dev.word(r.desa_l + 4) as u64) << 32;
            assert_eq!(base, tx_bds.iova().0 + (ch * PAGE) as u64);
        }
        for (q, r) in [RXQ0, RPQ0].iter().enumerate() {
            assert_eq!(dev.word(r.num) & 0xFFFF, u32::from(RX_RING));
            assert_eq!(
                dev.word(r.desa_l) as u64,
                rx_bds.iova().0 + (q * PAGE) as u64
            );
            assert_eq!(dev.word(r.idx) & 0xFFFF, u32::from(RX_RING - 1));
        }
        let stop = dev.word(R_BE_HAXI_DMA_STOP1);
        assert_eq!(stop & B_BE_STOP_CH12, 0);
        assert_eq!(
            stop & (TX_STOP_DATA | TX_STOP_UNUSED),
            TX_STOP_DATA | TX_STOP_UNUSED
        );
        assert_eq!(stop & B_BE_STOP_WPDMA, B_BE_STOP_WPDMA);
        let cfg1 = dev.word(R_BE_HAXI_INIT_CFG1);
        assert_eq!(
            cfg1 & (B_BE_TXDMA_EN | B_BE_RXDMA_EN),
            B_BE_TXDMA_EN | B_BE_RXDMA_EN
        );
        assert_eq!(cfg1 & B_BE_STOP_AXI_MST, 0);
        assert_eq!(cfg1 & B_BE_MAX_TXDMA_MASK, 2);
        assert_eq!(dev.word(R_BE_HAXI_EXP_CTRL_V1) & B_BE_MAX_TAG_NUM_MASK, 7);
        assert_eq!(dev.word(R_BE_WP_ADDR_H_SEL12_15_V1), 0x0F0E_0D0C);
        // The first receive descriptor names the first buffer.
        let bd = unsafe { core::slice::from_raw_parts(rx_bds.cpu(), 8) };
        assert_eq!(u16::from_le_bytes([bd[0], bd[1]]) as usize, RX_BUF_BYTES);
        assert_eq!(
            u32::from_le_bytes([bd[4], bd[5], bd[6], bd[7]]) as u64,
            rx_bufs.iova().0 & 0xFFFF_FFFF
        );
        assert!(staging.len() >= STAGING_BYTES);

        // The download: the header packet the plan builds, then every
        // kept section, in 2020-byte packets.
        let plan = FirmwareImage::verified(blob, &digest)
            .unwrap()
            .suit(1)
            .unwrap()
            .plan(&SecureBoot::OFF)
            .unwrap();
        let mut hdr = [0u8; crate::fw::HEADER_PACKET_MAX];
        let n = plan.header_packet(&mut hdr).unwrap();
        assert_eq!(dev.header_packet().as_deref(), Some(&hdr[..n]));
        assert_eq!(
            dev.received(),
            expected_stream(blob, &digest, &SecureBoot::OFF)
        );
        assert_eq!(radio.download_packets(), 1 + 2 + 1 + 3);
        assert!(has(Write {
            width: 4,
            off: R_BE_SECURE_BOOT_MALLOC_INFO,
            val: 0x2024_8000
        }));
        assert_eq!(dev.word(R_BE_WCPU_FW_CTRL) & B_BE_WLANCPU_FWDL_EN, 0);
        // The mailbox: one command, one reply, the counter nibbles.
        assert_eq!(dev.byte(R_BE_UDM1 + 1), 0x11);
        assert_eq!(dev.byte(R_BE_C2HREG_CTRL), 0);
        assert_eq!(
            dev.word(R_BE_WL_BT_PWR_CTRL) & B_BE_BT_DISN_EN,
            B_BE_BT_DISN_EN
        );

        // Quiesce: engines off, indices cleared, the MAC powered down.
        let mut radio = radio;
        radio.quiesce(&dev, waits()).expect("quiesced");
        assert!(!dev.powered());
        assert_eq!(dev.power_offs(), 1);
        assert_eq!(
            dev.word(R_BE_HAXI_INIT_CFG1) & (B_BE_TXDMA_EN | B_BE_RXDMA_EN),
            0
        );
        assert_eq!(dev.word(R_BE_HCI_OPT_CTRL) & BIT_WAKE_CTRL_V1, 0);
        assert_eq!(
            dev.word(R_BE_SYS_PW_CTRL) & B_BE_APFM_SWLPS,
            B_BE_APFM_SWLPS
        );
        assert_eq!(dev.xtal_writes().len(), 15 + 12);
    }

    #[test]
    fn rung_1_a_dead_or_short_window_is_not_present() {
        let (blob, digest) = container();
        let dead = MockRtw89::new(
            &efuse(false),
            Faults {
                dead: true,
                ..Faults::default()
            },
        );
        assert_eq!(
            bring_up(&dead, Some(blob), &digest, blob.len()).err(),
            Some(Error::NotPresent)
        );
        let dev = MockRtw89::new(&efuse(false), Faults::default());
        let short = Prepared::prepare(
            &dev,
            &dev.pool(),
            Product::Rtl8922ae,
            0x1_0000,
            Some(blob),
            &digest,
            blob.len(),
            waits(),
        );
        assert_eq!(short.err(), Some(Error::NotPresent));
        assert_eq!(dev.power_ons(), 0);
    }

    #[test]
    fn rungs_2_and_3_refuse_the_firmware_before_the_radio_is_touched() {
        let (blob, digest) = container();
        let dev = MockRtw89::new(&efuse(false), Faults::default());
        assert_eq!(
            bring_up(&dev, None, &digest, blob.len()).err(),
            Some(Error::NoFirmware)
        );
        assert_eq!(
            bring_up(&dev, Some(blob), &Digest([0; 32]), blob.len()).err(),
            Some(Error::BadFirmware)
        );
        assert_eq!(
            bring_up(&dev, Some(blob), &digest, blob.len() + 1).err(),
            Some(Error::BadFirmware)
        );
        assert_eq!(dev.trace().len(), 0, "no write reached the radio");
        // No image for the cut: the wrapper's only entry is cut 2.
        let img = image(1, Version::default(), false, &[Sec::plain(0, vec![1; 8])]);
        let cut2: &'static [u8] = Box::leak(mfw(&[(2, 1, 0, img)]).into_boxed_slice());
        assert_eq!(
            bring_up(&dev, Some(cut2), &digest_of(cut2), cut2.len()).err(),
            Some(Error::BadFirmware)
        );
        assert_eq!(dev.power_ons(), 0);
        // A header that does not parse is found after the efuse, and the
        // radio is powered off again.
        let mut lying = image(1, Version::default(), false, &[Sec::plain(0, vec![1; 8])]);
        lying.push(0);
        let lying: &'static [u8] = Box::leak(mfw(&[(0, 1, 0, lying)]).into_boxed_slice());
        assert_eq!(
            bring_up(&dev, Some(lying), &digest_of(lying), lying.len()).err(),
            Some(Error::BadFirmware)
        );
        assert_eq!((dev.power_ons(), dev.power_offs()), (1, 1));
        assert!(!dev.powered());
    }

    #[test]
    fn rung_4_power_on_retries_once_through_off_then_refuses() {
        let (blob, digest) = container();
        let dev = MockRtw89::new(
            &efuse(false),
            Faults {
                power_never_ready: true,
                ..Faults::default()
            },
        );
        assert_eq!(
            bring_up(&dev, Some(blob), &digest, blob.len()).err(),
            Some(Error::PowerOn)
        );
        // Two on attempts (each dying at step 6, before the MAC enable)
        // and one off between them: step 3 of each on and the off's
        // `0xA1B2` all write the LPS control register.
        let writes = dev
            .trace()
            .iter()
            .filter(|w| w.off == R_BE_WLLPS_CTRL)
            .count();
        assert_eq!(writes, 3);
        assert_eq!(dev.power_offs(), 1);
        assert_eq!(dev.power_ons(), 0);
    }

    #[test]
    fn a_warm_radio_is_powered_off_before_it_is_powered_on() {
        let (blob, digest) = container();
        let dev = MockRtw89::new(&efuse(false), Faults::default());
        dev.set_powered();
        let radio = bring_up(&dev, Some(blob), &digest, blob.len()).expect("firmware alive");
        assert_eq!(dev.power_offs(), 1);
        assert_eq!(dev.power_ons(), 1 + 1);
        assert!(dev.powered());
        drop(radio);
    }

    #[test]
    fn rung_5_an_efuse_that_never_answers_refuses_and_powers_off() {
        let (blob, digest) = container();
        let dev = MockRtw89::new(
            &efuse(false),
            Faults {
                efuse_never_ready: true,
                ..Faults::default()
            },
        );
        assert_eq!(
            bring_up(&dev, Some(blob), &digest, blob.len()).err(),
            Some(Error::Efuse)
        );
        assert!(!dev.powered());
        // The power cut was disabled again on the way out.
        assert_eq!(
            dev.byte(R_BE_PMC_DBG_CTRL2) & B_BE_SYSON_DIS_PMCR_BE_WRMSK as u8,
            0
        );
        assert_eq!(dev.word(R_BE_EFUSE_CTRL_2_V1) & B_BE_EF_BURST, 0);
    }

    /// A pool that refuses after `n` allocations.
    struct Limited {
        inner: IommuPool,
        left: Cell<usize>,
    }

    impl DmaPool for Limited {
        type Buf = IommuBuf;
        fn alloc(&self, bytes: usize) -> Option<Self::Buf> {
            if self.left.get() == 0 {
                return None;
            }
            self.left.set(self.left.get() - 1);
            self.inner.alloc(bytes)
        }
    }

    #[test]
    fn rung_6_an_exhausted_pool_refuses_and_powers_off() {
        let (blob, digest) = container();
        for n in 0..4 {
            let dev = MockRtw89::new(&efuse(false), Faults::default());
            let pool = Limited {
                inner: dev.pool(),
                left: Cell::new(n),
            };
            let r = Prepared::prepare(
                &dev,
                &pool,
                Product::Rtl8922ae,
                BAR_LEN,
                Some(blob),
                &digest,
                blob.len(),
                waits(),
            );
            assert_eq!(r.err(), Some(Error::DmaAlloc), "{n} allocations");
            assert!(!dev.powered());
        }
        let dev = MockRtw89::new(&efuse(false), Faults::default());
        let pool = Limited {
            inner: dev.pool(),
            left: Cell::new(4),
        };
        assert!(
            Prepared::prepare(
                &dev,
                &pool,
                Product::Rtl8922ae,
                BAR_LEN,
                Some(blob),
                &digest,
                blob.len(),
                waits()
            )
            .is_ok()
        );
    }

    #[test]
    fn rung_7_the_engine_pre_init_refuses_on_dle_or_busy_channels() {
        let (blob, digest) = container();
        for faults in [
            Faults {
                dle_never_ready: true,
                ..Faults::default()
            },
            Faults {
                dma_busy: true,
                ..Faults::default()
            },
        ] {
            let dev = MockRtw89::new(&efuse(false), faults);
            assert_eq!(
                bring_up(&dev, Some(blob), &digest, blob.len()).err(),
                Some(Error::DmaInit)
            );
            assert!(!dev.powered());
        }
    }

    #[test]
    fn rung_8_a_rejected_image_is_retried_five_times_then_refused() {
        let (blob, digest) = container();
        let dev = MockRtw89::new(
            &efuse(false),
            Faults {
                reject_status: Some(4),
                ..Faults::default()
            },
        );
        assert_eq!(
            bring_up(&dev, Some(blob), &digest, blob.len()).err(),
            Some(Error::FwDownload)
        );
        let primes = dev
            .trace()
            .iter()
            .filter(|w| w.off == R_BE_SECURE_BOOT_MALLOC_INFO)
            .count();
        assert_eq!(primes, 5);
        assert!(!dev.powered());
        let dev = MockRtw89::new(
            &efuse(false),
            Faults {
                h2c_path_never: true,
                ..Faults::default()
            },
        );
        assert_eq!(
            bring_up(&dev, Some(blob), &digest, blob.len()).err(),
            Some(Error::FwDownload)
        );
        assert!(dev.header_packet().is_none());
    }

    #[test]
    fn rung_9_a_silent_or_wrong_mailbox_is_no_ack() {
        let (blob, digest) = container();
        for faults in [
            Faults {
                mailbox_silent: true,
                ..Faults::default()
            },
            Faults {
                mailbox_wrong_func: true,
                ..Faults::default()
            },
        ] {
            let dev = MockRtw89::new(&efuse(false), faults);
            assert_eq!(
                bring_up(&dev, Some(blob), &digest, blob.len()).err(),
                Some(Error::NoAck)
            );
            assert!(dev.received().len() > 0, "the download completed first");
            assert!(!dev.powered());
        }
    }

    #[test]
    fn secure_boot_copies_the_selected_key_into_the_sections_last_packet() {
        let (blob, digest) = container();
        let dev = MockRtw89::new(&efuse(true), Faults::default());
        let radio = bring_up(&dev, Some(blob), &digest, blob.len()).expect("firmware alive");
        let secure = radio.secure_boot();
        assert!(secure.secure_boot);
        assert_eq!(secure.mss_key_num, 1);
        assert_eq!(secure.mss_dev_type, 0);
        let stream = expected_stream(blob, &digest, &secure);
        assert_eq!(dev.received(), stream);
        // The security section (600 bytes and its 8-byte checksum
        // trailer) goes out in one packet with the selected 512-byte key
        // over its tail — the first key, because the remap table's lower
        // bits below the selected slot are clear.
        assert!(stream[3000..3058].iter().all(|&b| b == 0x5C));
        assert!(stream[3096..3608].iter().all(|&b| b == 0xA1));
        assert_eq!(stream[3608], 0);
        let _ = PhyCap::default();
        let _ = TX_BD_BYTES;
    }
}
