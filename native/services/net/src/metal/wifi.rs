//! **The Wi-Fi composition** — the RTL8922AE class binding over
//! `kore-rtw89`, and what its bound radio offers the row's generic device
//! host.
//!
//! `kore-rtw89` owns every register sequence and `service` owns every
//! frame decision; this module owns neither. It realizes a claimed function
//! into the driver seams (BAR2, one DMA pool, one MSI-family route) and
//! carries the pinned firmware into the bring-up ladder. Everything here is
//! exercised without hardware: the ladder by `kore-rtw89`'s mock, the class
//! binding by the driver host's own tests. The witness it earns is
//! metal-only, because no QEMU profile carries the part.
//!
//! The row's BAR window is a `CapRegs`, which implements the seam's
//! `RegsNarrow` contract the core takes: the part's byte- and
//! halfword-wide control registers are accessed at their own width, never
//! synthesized from a dword read-modify-write.
//!
//! **D1 only.** A bound radio has firmware loaded and answering, and
//! nothing else: no vif, no frame. [`NicBound::backend`] therefore answers
//! `None` and every verb refuses `NotReady`, which is the honest report
//! until D2 gives the row a frame path over this part.

use kore_driver::host::{
    CapPool, ClassQuiesce, DeviceBinding, DeviceResources, DmaActive, DmaPrepared, InterruptBlock,
    InterruptPolicy, PciBindingContract,
};
use kore_rtw89::fw::Digest;
use kore_rtw89::mailbox::PhyCap;
use kore_rtw89::{Error as Rtw89Error, Prepared, Product, Rtw89, Waits};

use super::{NetDeviceHost, NicBound, NoFrames};
use crate::service::{RadioFirmware, RadioPhy, RadioReport};

/// The BAR the RTL8922AE's registers live in. BAR0 is 256 bytes of legacy
/// port I/O the kernel never turns into a resource; BAR2 is the 1 MiB
/// register window every access goes through (`lspci` on Gigatron, and
/// upstream `rtw89/pci.c`'s `bar_id = 2`).
pub const REGS_BAR: u8 = 2;

/// The wait tier's hooks the row hands the core: the boot clock and the
/// 4 µs relaxation every other metal core probes with.
const WAITS: Waits = Waits {
    clock: kore_driver::host::clock_now_ns,
    pause: kore_driver::host::relax,
};

/// The radio powered on, its efuse read, the firmware verified and the
/// rings allocated — the device may not master memory yet. The generic
/// host owns the only transition from here.
pub struct Rtw89NetPrepared {
    radio: Prepared<CapPool>,
}

/// The bound owner: the running radio (firmware loaded and answering) and
/// the capability resource set.
pub struct Rtw89NetBound {
    radio: Rtw89<CapPool>,
    resources: DeviceResources<DmaActive>,
}

/// What the claim decided before spending the pool: the generation the
/// probed identity selected, and the firmware the boot carried.
///
/// `firmware` is the whole leased `kore-rtw89-firmware` module, exactly as
/// the founder mapped it (page-rounded, so longer than the blob), or `None`
/// on a boot with no module — which the ladder reports as its own rung
/// rather than the row inventing a blob. The digest and size are the
/// compiled citation of `drivers/net/rtw89/firmware/manifest.toml`; the
/// manifest is the trust decision.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rtw89Config {
    /// The part the claimed function's PCI identity selected.
    pub product: Product,
    /// The firmware module's bytes, when the boot carried one.
    pub firmware: Option<&'static [u8]>,
}

/// The class declaration: BAR2, one interrupt block, DMA.
///
/// `Required(SINGLE)` with the row's `RouteCapable` posture: the block's
/// `PreferMsiX` source admits the part's plain MSI capability through the
/// same remapped route the 82574L's MSI-X takes, and a radio that could
/// only poll would congest the whole machine for a frame that may never
/// come.
pub struct Rtw89NetBinding;

impl PciBindingContract for Rtw89NetBinding {
    const BARS: kore_rt::service::PciBarMask = kore_rt::service::PciBarMask::BAR2;
    const INTERRUPTS: InterruptPolicy = InterruptPolicy::Required(InterruptBlock::SINGLE);
}

impl DeviceBinding for Rtw89NetBinding {
    type Pci = Self;
    type Config = Rtw89Config;
    type Prepared = Rtw89NetPrepared;
    type Bound = Rtw89NetBound;
    type Error = Rtw89Error;

    fn prepare(
        resources: &mut DeviceResources<DmaPrepared>,
        config: Self::Config,
    ) -> Result<Self::Prepared, Rtw89Error> {
        let regs = resources.bar(REGS_BAR).ok_or(Rtw89Error::NotPresent)?;
        let pool = resources.pool().ok_or(Rtw89Error::DmaAlloc)?;
        let radio = Prepared::prepare_with_baseband(
            regs,
            pool,
            config.product,
            regs.len(),
            config.firmware,
            &FW_DIGEST,
            FW_SIZE,
            WAITS,
        )?;
        Ok(Rtw89NetPrepared { radio })
    }

    fn start(
        prepared: Self::Prepared,
        resources: DeviceResources<DmaActive>,
    ) -> Result<Self::Bound, Rtw89Error> {
        let regs = resources.bar(REGS_BAR).ok_or(Rtw89Error::NotPresent)?;
        let radio = prepared.radio.start(regs, WAITS)?;
        Ok(Rtw89NetBound { radio, resources })
    }

    fn quiesce(bound: &mut Self::Bound) -> Result<ClassQuiesce, Rtw89Error> {
        let regs = bound
            .resources
            .bar(REGS_BAR)
            .ok_or(Rtw89Error::NotPresent)?;
        bound.radio.quiesce(regs, WAITS)?;
        Ok(ClassQuiesce::Stopped)
    }

    fn unbind(bound: Self::Bound) -> DeviceResources<DmaActive> {
        let Rtw89NetBound { radio, resources } = bound;
        drop(radio);
        resources
    }

    /// The core's rung rides
    /// [`IntakeError::ClassBindRefused`](kore_driver::host::IntakeError::ClassBindRefused)
    /// out to the entry ladder, which notes it as
    /// [`WIFI_BRING_UP_BASE`](crate::note::WIFI_BRING_UP_BASE)` + rung`.
    fn refusal_code(error: &Self::Error) -> u8 {
        error.rung()
    }
}

impl NicBound for Rtw89NetBound {
    type Backend<'a> = NoFrames;

    /// No frame path at D1: the radio is alive, not serving. The row reports
    /// `NotReady` on every verb, the same posture as a controller with no
    /// adaptor, and the founding witness says what *was* proven.
    fn backend(&mut self) -> Option<NoFrames> {
        None
    }

    fn resources(&self) -> &DeviceResources<DmaActive> {
        &self.resources
    }

    /// A bound radio has firmware alive by construction — every rung below
    /// that is a refusal the intake carried out — so it earns no note.
    fn founding_note(&self) -> u64 {
        0
    }

    /// No sub-address: the function is the device.
    fn detail(&self) -> u8 {
        0
    }
}

impl Rtw89NetBound {
    /// What the firmware answered to the alive handshake: the PHY's shape
    /// as it sees it. The entry derives the founding witness from it — a
    /// radio whose firmware names no stream proved nothing worth a bit —
    /// and D2's capability advertisement is built from it.
    pub fn phy_cap(&self) -> PhyCap {
        self.radio.phy_cap()
    }

    /// Everything the ladder read on the way to firmware-alive, in the
    /// service's own backend-free vocabulary — what the row answers
    /// [`NetRequest::Radio`](crate::service::NetRequest::Radio) with. A
    /// straight transcription: no field is derived, so the report is exactly
    /// the evidence, and a reader decodes the codes against the same
    /// upstream definitions `kore-rtw89` cites.
    pub fn radio_report(&self) -> RadioReport {
        let version = self.radio.firmware_version();
        let efuse = self.radio.efuse();
        let secure = self.radio.secure_boot();
        let cap = self.radio.phy_cap();
        RadioReport {
            cut: self.radio.cut(),
            firmware: RadioFirmware {
                header: self.radio.firmware_header_version(),
                major: version.major,
                minor: version.minor,
                sub: version.sub,
                idx: version.idx,
                commit: version.commit_id,
            },
            country: efuse.country_code,
            xtal_cap: efuse.xtal_cap,
            rfe_type: efuse.rfe_type,
            efuse_autoload: efuse.autoload,
            normal_mac: self.radio.normal_mac_ready(),
            phy_configured: self.radio.phy_configured(),
            mac_features: self.radio.mac_features_ready(),
            secure_boot: secure.secure_boot,
            sb_sel_mgn: secure.sb_sel_mgn,
            mss_dev_type: secure.mss_dev_type,
            mss_cust_idx: secure.mss_cust_idx,
            mss_key_num: secure.mss_key_num,
            phy: RadioPhy {
                rx_nss: cap.rx_nss,
                bw: cap.bw,
                tx_nss: cap.tx_nss,
                prot: cap.prot,
                nic: cap.nic,
                wl_func: cap.wl_func,
                hw_type: cap.hw_type,
                ant_tx: cap.ant_tx_num,
                ant_rx: cap.ant_rx_num,
                band_sel: cap.band_sel,
            },
            download_packets: u32::try_from(self.radio.download_packets()).unwrap_or(u32::MAX),
        }
    }
}

/// The frame engine's host over the RTL8922AE: the generic device host,
/// instantiated.
pub type Rtw89Host = NetDeviceHost<Rtw89NetBinding>;

/// The pinned blob's exact byte length, the compiled citation of
/// `drivers/net/rtw89/firmware/manifest.toml` (`[firmware.rtw8922a_fw]
/// size`). The mapped module is page-rounded; the ladder slices the blob to
/// this length before the digest check, so a short module refuses cleanly.
pub const FW_SIZE: usize = kore_rtw89::fw::PINNED_SIZE;
/// …and its BLAKE3 digest (`blake3` in the same entry). The manifest is
/// the trust decision; a blob that hashes differently refuses at rung 3
/// before a byte of it is interpreted and before the radio is powered.
pub const FW_DIGEST: Digest = kore_rtw89::fw::PINNED_DIGEST;

#[cfg(test)]
mod tests {
    use super::*;

    /// The intake code is the only rung map: the entry ladder notes
    /// `WIFI_BRING_UP_BASE + code` straight off `ClassBindRefused`, so the
    /// code must be the core's own dense ladder, starting at one.
    #[test]
    fn the_refusal_code_is_the_cores_rung_dense_from_one() {
        let ladder = [
            Rtw89Error::NotPresent,
            Rtw89Error::NoFirmware,
            Rtw89Error::BadFirmware,
            Rtw89Error::PowerOn,
            Rtw89Error::Efuse,
            Rtw89Error::DmaAlloc,
            Rtw89Error::DmaInit,
            Rtw89Error::FwDownload,
            Rtw89Error::NoAck,
            Rtw89Error::MacInit,
            Rtw89Error::PhyInit,
            Rtw89Error::MacCommand,
        ];
        for (i, error) in ladder.iter().enumerate() {
            assert_eq!(usize::from(Rtw89NetBinding::refusal_code(error)), i + 1);
        }
        let top = crate::note::WIFI_BRING_UP_BASE
            + u64::from(Rtw89NetBinding::refusal_code(&Rtw89Error::MacCommand));
        assert!(
            top < 64,
            "the whole band stays below the note's detail bits"
        );
        assert_eq!(top, 60, "the decoder ring in kore-abi ends at 60");
        assert!(
            crate::note::WIFI_BRING_UP_BASE > crate::note::PCI_BRING_UP_BASE + 15,
            "the PCI band keeps its fifteen rungs of room"
        );
    }

    #[test]
    fn the_binding_names_bar2_and_one_msi_family_route() {
        assert_eq!(
            <Rtw89NetBinding as PciBindingContract>::BARS,
            kore_rt::service::PciBarMask::BAR2
        );
        assert_eq!(
            <Rtw89NetBinding as PciBindingContract>::INTERRUPTS,
            InterruptPolicy::Required(InterruptBlock::SINGLE)
        );
        assert_eq!(REGS_BAR, 2);
    }
}
