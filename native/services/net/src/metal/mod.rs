//! **The production composition** — the row's endowment, the row itself,
//! the generic device host every bus shares, and the per-host compositions
//! under it: [`pci`] (the 82574L class binding over `kore-e1000e`),
//! [`wifi`] (the RTL8922AE class binding over `kore-rtw89`) and [`usb`]
//! (the xHCI class binding over `kore-usbnet`). The frame engine is
//! `service`'s and every device sequence is a driver core's; this module
//! owns the wiring between them and the kernel.
//!
//! Which hosts a boot gets is decided **once, at the claim**
//! (`entry::claim_controller`): the wired PCI NIC carries frames if it is
//! there; otherwise the Wi-Fi function is bound for its facts and an
//! Ethernet adaptor behind an xHCI controller is looked for beside it,
//! since a radio at #388 D1 lends no frames and must not shadow a dongle
//! that does. [`MetalHost`] is the one place the three meet, and it only
//! delegates — the engine is generic over [`NetHost`] and never learns
//! which bus it is driving.
//!
//! [`NetDeviceHost::adopt_window`] is the whole untestable surface of this
//! row, and it is deliberately a few dozen lines rather than a datapath.
//! Everything else the row does is exercised without hardware: the frame
//! engine by `service`'s host tests, each bring-up ladder by its core's
//! mock, the class bindings by the driver host's own.

use kore_abi::device::IovaAddr;
use kore_abi::session::SlotRef;
use kore_driver::host::{
    BoundDevice, CapBuf, DeviceBinding, DeviceResources, DmaActive, PciBindingContract,
    PciClaimPosture, pci_selector_for_binding,
};
use kore_driver::host::{CapPool, CapRegs};
use kore_driver::net::{
    LinkState, MacAddr, NicBackend, NicCounters, NicHost, PacketFilter, Steering, TxCompletion,
    TxFrame,
    window::{WINDOW_BYTES, WINDOW_LEASE_BYTES},
};
use kore_e1000e::E1000eNic;
use kore_rt::cap::{ReturnedCapability, StashedLease, verbs};
use kore_usbnet::device::EcmNic;

use crate::service::{AdoptedWindow, NetHost, RadioReport, ShareFault};

pub mod mt7925;
pub mod pci;
pub mod usb;
pub mod wifi;

pub use mt7925::*;
pub use pci::*;
pub use usb::*;
pub use wifi::*;

/// The rights a vif-window transfer must carry.
///
/// `adopt_window` checks the same rights that clients use to transfer the
/// window.
pub use crate::client::WINDOW_RIGHTS;

// ── the device host ────────────────────────────────────────────────────

/// What a bound network device offers the row, whatever bus it came in on.
///
/// The class binding realizes the claim; this is the face the row reads
/// afterwards. Selection between bindings is a type instantiation of
/// [`NetDeviceHost`] at the claim, once — the engine never matches on it.
pub trait NicBound {
    /// The backend this device lends, borrowing from `self`.
    type Backend<'a>: NicBackend
    where
        Self: 'a;

    /// Borrow the frame backend for one drive cycle; `None` when the device
    /// cannot serve one (no adaptor behind the controller, a resource gone).
    fn backend(&mut self) -> Option<Self::Backend<'_>>;

    /// The capability resource set: the pool vif windows are carved from,
    /// the device token they are installed into, the held route.
    fn resources(&self) -> &DeviceResources<DmaActive>;

    /// The readiness note this device earns at founding: `0` when the
    /// datapath is live, otherwise the [`crate::note`] rung that says why
    /// not — a fact only the bring-up walk can still tell.
    fn founding_note(&self) -> u64;

    /// What rides the note's detail bits: the USB root port an adaptor
    /// answered on; `0` where the bus has no such sub-address.
    fn detail(&self) -> u8;
}

/// The frame engine's host: whoever owns the controller.
///
/// A machine with no adaptor — or no controller to claim — still constructs
/// one. `backend()` then answers `None` and every verb refuses
/// [`NetRefusal::NotReady`](crate::NetRefusal::NotReady), which is a report
/// rather than a failure.
pub struct NetDeviceHost<B: DeviceBinding>
where
    B::Bound: NicBound,
{
    device: Option<BoundDevice<B>>,
}

impl<B: DeviceBinding> NetDeviceHost<B>
where
    B::Bound: NicBound,
{
    /// Host a claimed and bound controller.
    pub fn bound(device: BoundDevice<B>) -> Self {
        Self {
            device: Some(device),
        }
    }

    /// The claim-less machine's posture.
    pub fn idle() -> Self {
        Self { device: None }
    }

    /// Move the bound controller out for the terminal teardown (#809 §L).
    /// The host keeps serving; every device-plane path already tolerates an
    /// absent controller (the controller-less machine's posture).
    pub fn take_device(&mut self) -> Option<BoundDevice<B>> {
        self.device.take()
    }

    /// The bound controller's class state, while one is hosted.
    pub fn parts(&self) -> Option<&B::Bound> {
        self.device.as_ref().map(BoundDevice::parts)
    }

    /// The readiness note the founding claim earned; `0` is a live datapath.
    /// A claim-less host has nothing to report here — its note is the
    /// claim's own.
    pub fn founding_note(&self) -> u64 {
        self.device
            .as_ref()
            .map_or(0, |d| d.parts().founding_note())
    }

    /// The bound device's detail bits; `0` when there is none.
    pub fn detail(&self) -> u8 {
        self.device.as_ref().map_or(0, |d| d.parts().detail())
    }

    /// Whether this row holds a realized interrupt route — the condition the
    /// boot gate's `remapped-msi-irq` witness is *derived* from, so the row
    /// arms its parks only when it can honestly earn it.
    pub fn has_route(&self) -> bool {
        self.device
            .as_ref()
            .is_some_and(|d| d.parts().resources().route().is_some())
    }

    /// Borrow the device token for the share.
    fn device_token(&self) -> Result<&kore_rt::cap::DeviceToken, ShareFault> {
        self.device
            .as_ref()
            .and_then(|d| d.parts().resources().device())
            .ok_or(ShareFault::Unsupported)
    }

    /// **Carve a vif window out of this row's own DMA pool** — the founding
    /// self-test's window, and nothing else's.
    ///
    /// It reports the pool's address rather than the engine's proposal
    /// because the pool picked it: a buffer is device-visible the moment it
    /// is allocated, and pretending otherwise would mean installing a second
    /// mapping over frames that already have one.
    pub fn carve_own_window(&mut self) -> Result<AdoptedWindow<VifLease>, ShareFault> {
        use kore_driver::seam::{Dma, DmaPool};

        let bound = self.device.as_ref().ok_or(ShareFault::Unsupported)?;
        let pool = bound
            .parts()
            .resources()
            .pool()
            .ok_or(ShareFault::Unsupported)?;
        let buf = pool.alloc(WINDOW_BYTES).ok_or(ShareFault::NoView)?;
        let (cpu, iova) = (buf.cpu(), buf.iova());
        // SAFETY: the pool hands out `WINDOW_BYTES` of zeroed, page-aligned,
        // device-visible memory owned by the returned buffer, which the lease
        // below keeps for the row's life.
        let bytes: &'static mut [u8] =
            unsafe { core::slice::from_raw_parts_mut(cpu, WINDOW_BYTES) };
        Ok(AdoptedWindow {
            bytes,
            iova,
            lease: VifLease::Own(buf),
        })
    }
}

impl<B: DeviceBinding> NicHost for NetDeviceHost<B>
where
    B::Bound: NicBound,
{
    type Backend<'a>
        = <B::Bound as NicBound>::Backend<'a>
    where
        Self: 'a;

    fn backend(&mut self) -> Option<Self::Backend<'_>> {
        self.device.as_mut()?.parts_mut().backend()
    }
}

/// What keeps one adopted vif window alive.
pub enum VifLease {
    /// A client's transferred lease.
    ///
    /// Both the CPU leaf and the device leaf are **children** of it, so
    /// retiring this one owner reaps both — which is what makes
    /// [`VifMapping::retire`](kore_driver::net::VifMapping::retire)'s refusal
    /// a real safety property rather than bookkeeping: while a transfer is
    /// outstanding the lease is unreachable, so the window cannot be torn out
    /// from under the controller.
    Client(StashedLease),
    /// A window carved from this row's **own** DMA pool — the founding
    /// self-test's, and the only vif window that is not a client's.
    ///
    /// A pool buffer is already both CPU-visible and device-visible, so it
    /// needs no share and has no leaf to reap. The pool is a bump allocator
    /// that never frees, so the buffer is spent for the row's life; one
    /// window, once per boot, is what a self-test that proves the whole
    /// datapath should cost.
    Own(CapBuf),
}

impl<B: DeviceBinding> NetHost for NetDeviceHost<B>
where
    B::Bound: NicBound,
{
    type Lease = VifLease;

    /// The bound class's own declaration — the same constant
    /// [`intake`](kore_driver::host) placed this row's DMA pool from, so the
    /// pool and the vif arena are derived from one fact rather than two.
    fn dma_address_bits(&self) -> u32 {
        B::Pci::DMA_ADDRESS_BITS
    }

    /// The live pool's admitted capacity, which includes anything
    /// `grow_dma` has added since founding — so the arena clears what the
    /// pool *is*, not what it was endowed with. `0` before a device binds.
    fn dma_pool_bytes(&self) -> u64 {
        self.device
            .as_ref()
            .and_then(|device| device.parts().resources().pool())
            .map_or(0, CapPool::capacity_bytes)
    }

    /// **The one step host tests cannot execute.**
    ///
    /// Three moves, in the order [`verbs::dma_window`] proved for the
    /// driver's own pool:
    ///
    /// 1. Retype the delivery as a lease, which re-checks the kernel-
    ///    authenticated species and rights at this ownership boundary — a
    ///    wrong-kind delivery is dropped, which revokes it.
    /// 2. Map the CPU half read/write. This service reads transmit slots to
    ///    validate frame headers and writes receive slots.
    /// 3. Install the same frames into the controller's IOMMU domain at
    ///    `iova`, **read-only**: the controller fetches transmit frames and
    ///    must never be able to write into a client's memory through a leaf
    ///    the client did not ask for.
    ///
    /// The device sees the whole window rather than just the transmit slots,
    /// because a held-lease install has no sub-offset. That exposes nothing:
    /// it is the same client's own memory, read-only, and the only address
    /// this service ever posts is a transmit slot's.
    fn adopt_window(
        &mut self,
        returned: ReturnedCapability,
        iova: IovaAddr,
    ) -> Result<AdoptedWindow<VifLease>, ShareFault> {
        // Establish the target before spending anything: a host with no
        // device refuses here, and the delivery drops (revoked) rather than
        // being mapped into a task that has nowhere to point it.
        if self.device_token().is_err() {
            return Err(ShareFault::Unsupported);
        }
        let mut lease = returned
            .try_into_stashed_lease(WINDOW_RIGHTS)
            .map_err(|_| ShareFault::NoTransfer)?;
        let Some(slot) = lease.slot() else {
            return Err(ShareFault::NoTransfer);
        };

        let Some(window) = verbs::map_held_rw(slot) else {
            let _ = lease.try_retire();
            return Err(ShareFault::NoView);
        };
        if window.len() < WINDOW_LEASE_BYTES {
            let _ = lease.try_retire();
            return Err(ShareFault::NoView);
        }

        let published = match self.device_token() {
            Ok(device) => {
                verbs::share_window(slot, device, iova.0, WINDOW_LEASE_BYTES as u64, false).is_ok()
            }
            Err(_) => false,
        };
        if !published {
            let _ = lease.try_retire();
            return Err(ShareFault::NoDeviceLeaf);
        }

        // SAFETY: `map_held_rw` installed `[base, +len)` R/W-USER in THIS
        // task's own space over the client's window backing, page-aligned and
        // live until the lease above is retired — which cannot happen while
        // the vif holds it. The client writes transmit slots and reads receive
        // ones from its own mapping of the same frames; the two sides touch
        // disjoint subregions and hand slots over through the ring protocol,
        // the same discipline every pair window rests on.
        let bytes: &'static mut [u8] =
            unsafe { core::slice::from_raw_parts_mut(window.cast::<u8>().as_ptr(), window.len()) };
        Ok(AdoptedWindow {
            bytes: &mut bytes[..WINDOW_BYTES],
            // Installed exactly where the engine proposed: this host owns the
            // domain and honours the plan's stride.
            iova,
            lease: VifLease::Client(lease),
        })
    }

    /// Retire the lease, which reaps both leaves with it.
    fn release_window(&mut self, lease: VifLease) {
        release_vif_lease(lease);
    }

    fn device(&self) -> Option<SlotRef> {
        Some(self.device.as_ref()?.parts().resources().device()?.slot())
    }
}

/// Retire a vif lease, which reaps both leaves with it.
fn release_vif_lease(lease: VifLease) {
    match lease {
        // A refused retire leaves a live lineage this row can no longer
        // name. It is bounded — one window — and preferable to the
        // alternative: forcing closure while the kernel still says the
        // record is reachable.
        VifLease::Client(mut lease) => {
            let _ = lease.try_retire();
        }
        // Nothing to retire: the pool never frees.
        VifLease::Own(_) => {}
    }
}

// ── the metal host ─────────────────────────────────────────────────────

/// The host this boot claimed — decided **once**, at the claim, by the
/// entry ladder's rung order, and never re-decided.
///
/// Two axes, because a machine can have two network devices that answer
/// different questions: the **frame host** carries frames (the wired PCI
/// NIC, or an Ethernet adaptor behind an xHCI controller — whichever the
/// ladder bound first), and the **radio** is the Wi-Fi function, bound for
/// its facts and its firmware-alive proof and lending no frames until #388
/// D2. A laptop with the RTL8922AE and a USB Ethernet adaptor plugged in
/// hosts both: `status net` reads the radio *and* the link, and the
/// self-test proves the frame path the dongle lends. Before this was a
/// composition the bound radio shadowed the adaptor — the ladder stopped
/// at the first device that bound, frames or not.
///
/// A struct over enums rather than a trait object because the engine is
/// generic over [`NetHost`] and the backends' associated types differ;
/// every method here delegates.
pub struct MetalHost {
    frames: FrameHost,
    radio: Option<RadioHost>,
}

/// The Wi-Fi function that was bound, and **how far it got**.
///
/// Two parts, two standings, and the enum exists so the difference cannot be
/// lost: the RTL8922AE binds at D1 with firmware alive, the MT7925 binds at
/// D0 having only answered with its identity. A single `Option<…Host>` would
/// make "a radio is bound" mean two different things depending on which part
/// the machine has, and `net-wifi-fw-alive` is derived from exactly that
/// question.
///
/// Neither part has a frame path, so both lend `NoFrames` and every verb
/// refuses `NotReady`.
pub enum RadioHost {
    /// The RTL8922AE at D1 — firmware loaded and answering. No machine in
    /// the lab binds it since Gigatron left (`res/NETWORK_RESTART.md` §2).
    Rtw89(Rtw89Host),
    /// The MT7925 at D0 — identity read back through the L1 remap window and
    /// ownership taken. **Not** firmware-alive.
    Mt7925(Mt7925Host),
}

impl RadioHost {
    /// Whether this part has firmware alive — the `net-wifi-fw-alive`
    /// question, and the reason this is an enum. **D0 is not alive.**
    pub fn firmware_alive(&self) -> bool {
        match self {
            Self::Rtw89(_) => true,
            Self::Mt7925(_) => false,
        }
    }

    /// The MT7925's composed revision, when that is the bound part — what
    /// the `net-mt7925-d0` witness prints.
    pub fn mt7925_revision(&self) -> Option<u32> {
        match self {
            Self::Mt7925(host) => host.parts().map(Mt7925NetBound::revision),
            Self::Rtw89(_) => None,
        }
    }

    /// Whether the MT7925's download ring was programmed and read back —
    /// the `net-mt7925-d1-ring` witness. `None` when the bound part is not
    /// an MT7925.
    pub fn mt7925_ring_ready(&self) -> Option<bool> {
        match self {
            Self::Mt7925(host) => host.parts().map(Mt7925NetBound::ring_ready),
            Self::Rtw89(_) => None,
        }
    }

    /// Whether the MT7925 answered an MCU command — the
    /// `net-mt7925-d1-mcu` witness.
    pub fn mt7925_mcu_answered(&self) -> Option<bool> {
        match self {
            Self::Mt7925(host) => host.parts().map(Mt7925NetBound::mcu_answered),
            Self::Rtw89(_) => None,
        }
    }

    /// The first rung the MT7925's D1 ladder refused at; zero if it
    /// refused nowhere.
    pub fn mt7925_bring_up_rung(&self) -> Option<u8> {
        match self {
            Self::Mt7925(host) => host.parts().map(Mt7925NetBound::bring_up_rung),
            Self::Rtw89(_) => None,
        }
    }

    /// Whether the boot carried both MT7925 blobs and both are the pinned
    /// bytes — the `net-mt7925-fw-pinned` witness.
    pub fn mt7925_firmware_pinned(&self) -> Option<bool> {
        match self {
            Self::Mt7925(host) => host
                .parts()
                .map(|p| Mt7925NetBound::firmware(p).is_pinned()),
            Self::Rtw89(_) => None,
        }
    }

    /// What C2b found: the patch crossed, the part already had it, or the
    /// rung was not driven.
    pub fn mt7925_patch(&self) -> Option<kore_mt7925::c2::PatchOutcome> {
        match self {
            Self::Mt7925(host) => host.parts().and_then(Mt7925NetBound::patch),
            Self::Rtw89(_) => None,
        }
    }

    /// What C2c sent, or `None` when the rung was not driven.
    pub fn mt7925_ram(&self) -> Option<kore_mt7925::c2::RamOutcome> {
        match self {
            Self::Mt7925(host) => host.parts().and_then(Mt7925NetBound::ram),
            Self::Rtw89(_) => None,
        }
    }

    /// `MT_CONN_ON_MISC` when this boot watched the N9 come up.
    pub fn mt7925_fw_alive(&self) -> Option<u32> {
        match self {
            Self::Mt7925(host) => host.parts().and_then(Mt7925NetBound::fw_alive),
            Self::Rtw89(_) => None,
        }
    }

    /// What the part said about itself at M1 — `None` when the rung was
    /// not driven, or refused.
    pub fn mt7925_phy(&self) -> Option<kore_mt7925::d2::PhyReport> {
        match self {
            Self::Mt7925(host) => host.parts().and_then(Mt7925NetBound::phy),
            Self::Rtw89(_) => None,
        }
    }

    /// One byte of evidence for that rung.
    pub fn mt7925_bring_up_detail(&self) -> Option<u8> {
        match self {
            Self::Mt7925(host) => host.parts().map(Mt7925NetBound::bring_up_detail),
            Self::Rtw89(_) => None,
        }
    }

    /// Four bytes of evidence for that rung — the note's bits 16..=47.
    pub fn mt7925_bring_up_evidence(&self) -> Option<u32> {
        match self {
            Self::Mt7925(host) => host.parts().map(Mt7925NetBound::bring_up_evidence),
            Self::Rtw89(_) => None,
        }
    }

    /// Whether this part holds a realized interrupt route.
    pub fn has_route(&self) -> bool {
        match self {
            Self::Rtw89(host) => host.has_route(),
            Self::Mt7925(host) => host.has_route(),
        }
    }
}

/// The device that carries frames, when one was bound.
pub enum FrameHost {
    /// The 82574L, over `kore-e1000e`.
    Pci(E1000eHost),
    /// An Ethernet adaptor behind an xHCI controller, over `kore-usbnet`.
    /// The controller may be bound with nothing plugged into it, in which
    /// case its `backend()` is `None` and the row serves the same posture
    /// as a machine with no frame host at all.
    Usb(EcmHost),
    /// No frame host: the claim-less machine, or a radio-only one.
    None,
}

impl MetalHost {
    /// The claim-less machine's posture: no backend, every verb refuses.
    pub const IDLE: Self = Self {
        frames: FrameHost::None,
        radio: None,
    };

    /// Compose the frame host the ladder bound with the radio it bound.
    pub fn new(frames: FrameHost, radio: Option<RadioHost>) -> Self {
        Self { frames, radio }
    }

    /// Whether the PCI host is the one carrying frames — what the
    /// `net-pci-frame-rt` witness is derived from, once the self-test has
    /// earned frame-rt.
    pub fn is_pci(&self) -> bool {
        matches!(self.frames, FrameHost::Pci(_))
    }

    /// What the radio's firmware answered at the alive handshake, on a
    /// host that bound one. The entry derives the Wi-Fi witness from it.
    pub fn wifi_phy_cap(&self) -> Option<kore_rtw89::mailbox::PhyCap> {
        match self.radio.as_ref()? {
            RadioHost::Rtw89(host) => host.parts().map(Rtw89NetBound::phy_cap),
            // A PHY capability is something firmware answers with. D0 has no
            // firmware, so there is nothing to report and nothing is invented.
            RadioHost::Mt7925(_) => None,
        }
    }

    /// The MT7925's revision, when that part is bound — the `net-mt7925-d0`
    /// witness. Separate from [`is_wifi`](Self::is_wifi) because D0 and D1
    /// are different claims and must not share a witness.
    pub fn mt7925_revision(&self) -> Option<u32> {
        self.radio.as_ref()?.mt7925_revision()
    }

    /// Whether the MT7925's download ring was accepted — the
    /// `net-mt7925-d1-ring` witness. A separate claim from
    /// [`mt7925_revision`](Self::mt7925_revision): D0 is the part
    /// answering, D1a is the engine taking a ring, and a boot can honestly
    /// have the first without the second.
    pub fn mt7925_ring_ready(&self) -> bool {
        self.radio
            .as_ref()
            .and_then(RadioHost::mt7925_ring_ready)
            .unwrap_or(false)
    }

    /// Whether the MT7925 answered an MCU command — the first rung the
    /// part must *do* something for, rather than accept a write.
    pub fn mt7925_mcu_answered(&self) -> bool {
        self.radio
            .as_ref()
            .and_then(RadioHost::mt7925_mcu_answered)
            .unwrap_or(false)
    }

    /// The first rung the MT7925's D1 ladder refused at; zero if it
    /// refused nowhere, or if no MT7925 is bound.
    ///
    /// **Read from `self.radio`, not `self.frames`.** That distinction is
    /// the whole bug this exists to fix: `founding_note` consults frames
    /// only, so a note written onto the bound part never reached the row.
    pub fn mt7925_bring_up_rung(&self) -> u8 {
        self.radio
            .as_ref()
            .and_then(RadioHost::mt7925_bring_up_rung)
            .unwrap_or(0)
    }

    /// One byte of evidence for that rung; zero when there is none.
    pub fn mt7925_bring_up_detail(&self) -> u8 {
        self.radio
            .as_ref()
            .and_then(RadioHost::mt7925_bring_up_detail)
            .unwrap_or(0)
    }

    /// Four bytes of evidence for that rung; zero when there are none.
    ///
    /// Read from `self.radio` for the same reason
    /// [`mt7925_bring_up_rung`](Self::mt7925_bring_up_rung) is.
    pub fn mt7925_bring_up_evidence(&self) -> u32 {
        self.radio
            .as_ref()
            .and_then(RadioHost::mt7925_bring_up_evidence)
            .unwrap_or(0)
    }

    /// Whether the boot carried both MT7925 blobs and both are pinned.
    pub fn mt7925_firmware_pinned(&self) -> bool {
        self.radio
            .as_ref()
            .and_then(RadioHost::mt7925_firmware_pinned)
            .unwrap_or(false)
    }

    /// What C2b found; `None` when the rung was not driven this boot.
    pub fn mt7925_patch(&self) -> Option<kore_mt7925::c2::PatchOutcome> {
        self.radio.as_ref().and_then(RadioHost::mt7925_patch)
    }

    /// What C2c sent; `None` when the rung was not driven this boot.
    pub fn mt7925_ram(&self) -> Option<kore_mt7925::c2::RamOutcome> {
        self.radio.as_ref().and_then(RadioHost::mt7925_ram)
    }

    /// `MT_CONN_ON_MISC` when this boot watched the N9 come up; `None`
    /// when it did not.
    ///
    /// **Not the same question as [`Self::is_wifi`].** That asks whether a
    /// radio bound, which for the rtw89 lineage implies running firmware
    /// because every rung below the bind refuses. The MT7925 binds at D0
    /// and climbs afterwards, so for this part the question has to be
    /// asked separately — and answered by the part's own register rather
    /// than inferred from the bind.
    pub fn mt7925_fw_alive(&self) -> Option<u32> {
        self.radio.as_ref().and_then(RadioHost::mt7925_fw_alive)
    }

    /// What the part said about itself at M1.
    ///
    /// **Read from `self.radio`, not `self.frames`**, for the reason
    /// [`mt7925_bring_up_rung`](Self::mt7925_bring_up_rung) records: a
    /// report written onto the bound part never reaches the row through
    /// `frames`.
    pub fn mt7925_phy(&self) -> Option<kore_mt7925::d2::PhyReport> {
        self.radio.as_ref().and_then(RadioHost::mt7925_phy)
    }

    /// Whether a radio is bound — what the `net-wifi-fw-alive` witness is
    /// derived from: a bound radio *is* the firmware-alive proof, because
    /// every rung below it refuses the bind.
    pub fn is_wifi(&self) -> bool {
        self.radio.as_ref().is_some_and(RadioHost::firmware_alive)
    }

    /// The readiness note the frame host's founding claim earned; `0` is a
    /// live datapath, and also the radio-only and claim-less postures —
    /// what those lack is said by the witness bits and `status net`, not by
    /// a rung.
    pub fn founding_note(&self) -> u64 {
        match &self.frames {
            FrameHost::Pci(host) => host.founding_note(),
            FrameHost::Usb(host) => host.founding_note(),
            FrameHost::None => 0,
        }
    }

    /// The frame host's detail bits; `0` when there is none.
    pub fn detail(&self) -> u8 {
        match &self.frames {
            FrameHost::Pci(host) => host.detail(),
            FrameHost::Usb(host) => host.detail(),
            FrameHost::None => 0,
        }
    }

    /// Whether this row holds a realized interrupt route on any of its
    /// devices — the parks arm every route the task holds.
    pub fn has_route(&self) -> bool {
        let frames = match &self.frames {
            FrameHost::Pci(host) => host.has_route(),
            FrameHost::Usb(host) => host.has_route(),
            FrameHost::None => false,
        };
        frames || self.radio.as_ref().is_some_and(RadioHost::has_route)
    }

    /// Carve the founding self-test's window out of the frame host's pool.
    pub fn carve_own_window(&mut self) -> Result<AdoptedWindow<VifLease>, ShareFault> {
        match &mut self.frames {
            FrameHost::Pci(host) => host.carve_own_window(),
            FrameHost::Usb(host) => host.carve_own_window(),
            FrameHost::None => Err(ShareFault::Unsupported),
        }
    }

    /// The terminal path (#809 §L): move every bound device out and tear it
    /// down, folding each outcome into the row's sweep. The host keeps
    /// serving in the controller-less posture.
    pub fn teardown_into(&mut self, sweep: &mut kore_rt::service::QuiesceSweep) {
        match &mut self.frames {
            FrameHost::Pci(host) => {
                if let Some(device) = host.take_device() {
                    device.teardown().record(sweep);
                }
            }
            FrameHost::Usb(host) => {
                if let Some(device) = host.take_device() {
                    device.teardown().record(sweep);
                }
            }
            FrameHost::None => {}
        }
        match self.radio.as_mut() {
            Some(RadioHost::Rtw89(host)) => {
                if let Some(device) = host.take_device() {
                    device.teardown().record(sweep);
                }
            }
            Some(RadioHost::Mt7925(host)) => {
                if let Some(device) = host.take_device() {
                    device.teardown().record(sweep);
                }
            }
            None => {}
        }
    }
}

/// **No frame path.** The backend type of a device that is bound but lends
/// no frames — the RTL8922AE at #388 D1, firmware alive and nothing above
/// it. Uninhabited, so a `Some(NoFrames)` cannot be constructed and the
/// host's `backend()` is `None` by type rather than by convention; every
/// method body is the empty match that proves it. D2 replaces it with the
/// radio's real backend, the way `E1000eNic` is the 82574L's.
pub enum NoFrames {}

impl NicBackend for NoFrames {
    type Error = MetalError;

    fn mac(&self) -> MacAddr {
        match *self {}
    }

    fn link(&self) -> LinkState {
        match *self {}
    }

    fn steering(&self) -> Steering {
        match *self {}
    }

    fn counters(&self) -> NicCounters {
        match *self {}
    }

    fn set_packet_filter(&mut self, _filter: PacketFilter) -> Result<(), MetalError> {
        match *self {}
    }

    fn transmit(&mut self, _frame: TxFrame<'_>) -> Result<(), MetalError> {
        match *self {}
    }

    fn reap_tx(&mut self, _out: &mut [TxCompletion]) -> usize {
        match *self {}
    }

    fn poll_rx(&mut self, _f: &mut dyn FnMut(&[u8])) -> bool {
        match *self {}
    }

    fn refill_rx(&mut self) -> Result<(), MetalError> {
        match *self {}
    }
}

/// The backend the metal host lends: one drive cycle over whichever core
/// the claim selected. Every method delegates; the error is widened into
/// [`MetalError`] so the engine sees one vocabulary.
pub enum MetalBackend<'a> {
    /// The 82574L's.
    Pci(E1000eNic<'a, CapRegs, CapPool>),
    /// The CDC Ethernet adaptor's.
    Usb(EcmNic<'a, CapRegs, CapPool>),
}

/// A backend refusal, in the vocabulary of whichever core raised it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetalError {
    /// `kore-e1000e`'s.
    Pci(kore_e1000e::Error),
    /// `kore-usbnet`'s.
    Usb(kore_usbnet::Error),
}

impl NicBackend for MetalBackend<'_> {
    type Error = MetalError;

    fn mac(&self) -> MacAddr {
        match self {
            Self::Pci(nic) => nic.mac(),
            Self::Usb(nic) => nic.mac(),
        }
    }

    fn link(&self) -> LinkState {
        match self {
            Self::Pci(nic) => nic.link(),
            Self::Usb(nic) => nic.link(),
        }
    }

    fn steering(&self) -> Steering {
        match self {
            Self::Pci(nic) => nic.steering(),
            Self::Usb(nic) => nic.steering(),
        }
    }

    fn counters(&self) -> NicCounters {
        match self {
            Self::Pci(nic) => nic.counters(),
            Self::Usb(nic) => nic.counters(),
        }
    }

    fn set_packet_filter(&mut self, filter: PacketFilter) -> Result<(), MetalError> {
        match self {
            Self::Pci(nic) => nic.set_packet_filter(filter).map_err(MetalError::Pci),
            Self::Usb(nic) => nic.set_packet_filter(filter).map_err(MetalError::Usb),
        }
    }

    fn transmit(&mut self, frame: TxFrame<'_>) -> Result<(), MetalError> {
        match self {
            Self::Pci(nic) => nic.transmit(frame).map_err(MetalError::Pci),
            Self::Usb(nic) => nic.transmit(frame).map_err(MetalError::Usb),
        }
    }

    fn service(&mut self, now_ns: Option<u64>) {
        match self {
            Self::Pci(nic) => nic.service(now_ns),
            Self::Usb(nic) => nic.service(now_ns),
        }
    }

    fn service_deadline(&self) -> Option<u64> {
        match self {
            Self::Pci(nic) => nic.service_deadline(),
            Self::Usb(nic) => nic.service_deadline(),
        }
    }

    fn flush_tx(&mut self) {
        match self {
            Self::Pci(nic) => nic.flush_tx(),
            Self::Usb(nic) => nic.flush_tx(),
        }
    }

    fn reap_tx(&mut self, out: &mut [TxCompletion]) -> usize {
        match self {
            Self::Pci(nic) => nic.reap_tx(out),
            Self::Usb(nic) => nic.reap_tx(out),
        }
    }

    fn poll_rx(&mut self, f: &mut dyn FnMut(&[u8])) -> bool {
        match self {
            Self::Pci(nic) => nic.poll_rx(f),
            Self::Usb(nic) => nic.poll_rx(f),
        }
    }

    fn refill_rx(&mut self) -> Result<(), MetalError> {
        match self {
            Self::Pci(nic) => nic.refill_rx().map_err(MetalError::Pci),
            Self::Usb(nic) => nic.refill_rx().map_err(MetalError::Usb),
        }
    }
}

impl NicHost for MetalHost {
    type Backend<'a> = MetalBackend<'a>;

    /// The frame host's backend. The radio lends none at D1 (`NoFrames`
    /// is uninhabited), so it is not consulted here at all.
    fn backend(&mut self) -> Option<MetalBackend<'_>> {
        match &mut self.frames {
            FrameHost::Pci(host) => host.backend().map(MetalBackend::Pci),
            FrameHost::Usb(host) => host.backend().map(MetalBackend::Usb),
            FrameHost::None => None,
        }
    }
}

impl NetHost for MetalHost {
    type Lease = VifLease;

    /// **The reach of whichever device a vif window would be installed
    /// into**, which is a runtime fact: each `FrameHost` arm is a different
    /// binding with its own
    /// [`DMA_ADDRESS_BITS`](kore_driver::host::PciBindingContract::DMA_ADDRESS_BITS),
    /// and nothing about the row's *type* says which one it founded.
    ///
    /// The frame host first, because [`adopt_window`](Self::adopt_window)
    /// installs into exactly that device's domain. Then the **radio**, which
    /// is not decoration: `MetalHost::founding_note` once read `self.frames`
    /// on a row whose part binds into `self.radio`, and three flashes
    /// reported nothing while every test passed (`res/NETWORK_C2_TO_PING.md`
    /// §4). A radio-only row cannot adopt a window today — `FrameHost::None`
    /// refuses below — but when the MT7925 lends a frame path it will be a
    /// 32-bit master, and this answers 32 for it *now* rather than
    /// defaulting to 64 and waiting to be noticed.
    fn dma_address_bits(&self) -> u32 {
        match (&self.frames, &self.radio) {
            (FrameHost::Pci(host), _) => host.dma_address_bits(),
            (FrameHost::Usb(host), _) => host.dma_address_bits(),
            (FrameHost::None, Some(RadioHost::Rtw89(host))) => host.dma_address_bits(),
            (FrameHost::None, Some(RadioHost::Mt7925(host))) => host.dma_address_bits(),
            (FrameHost::None, None) => 64,
        }
    }

    /// The same device's pool span, chosen by the same rule and for the same
    /// reason — the two must describe one device, or the arena would be
    /// placed clear of a pool that is not the one it shares a domain with.
    fn dma_pool_bytes(&self) -> u64 {
        match (&self.frames, &self.radio) {
            (FrameHost::Pci(host), _) => host.dma_pool_bytes(),
            (FrameHost::Usb(host), _) => host.dma_pool_bytes(),
            (FrameHost::None, Some(RadioHost::Rtw89(host))) => host.dma_pool_bytes(),
            (FrameHost::None, Some(RadioHost::Mt7925(host))) => host.dma_pool_bytes(),
            (FrameHost::None, None) => 0,
        }
    }

    fn adopt_window(
        &mut self,
        returned: ReturnedCapability,
        iova: IovaAddr,
    ) -> Result<AdoptedWindow<VifLease>, ShareFault> {
        match &mut self.frames {
            FrameHost::Pci(host) => host.adopt_window(returned, iova),
            FrameHost::Usb(host) => host.adopt_window(returned, iova),
            // The delivery drops here, which revokes it: nothing to map it
            // into.
            FrameHost::None => Err(ShareFault::Unsupported),
        }
    }

    fn release_window(&mut self, lease: VifLease) {
        release_vif_lease(lease);
    }

    fn device(&self) -> Option<SlotRef> {
        match &self.frames {
            FrameHost::Pci(host) => NetHost::device(host),
            FrameHost::Usb(host) => NetHost::device(host),
            FrameHost::None => None,
        }
    }

    /// The bound radio's report; a bound one *is* the firmware-alive proof,
    /// so its report exists exactly when the witness does, whatever carries
    /// the frames beside it.
    fn radio(&self) -> Option<RadioReport> {
        match self.radio.as_ref()? {
            RadioHost::Rtw89(host) => host.parts().map(Rtw89NetBound::radio_report),
            // The report's every field is something the firmware or the
            // efuse answered. D0 read neither, and a report of zeroes would
            // be indistinguishable from a radio that answered zeroes.
            RadioHost::Mt7925(_) => None,
        }
    }
}

// ── the row ────────────────────────────────────────────────────────────

/// The row's role bundle: two DMA supplies (the frame host's and the
/// radio's — a supply is affine per bound device, and the two may be bound
/// together), a channel space, and the **optional** Wi-Fi firmware byte
/// store — nothing else.
///
/// **No `DEVICE`.** The controller is *claimed* from the device broker at
/// startup (`res/DEVICE_SERVICE.md` §10, the swap-nvme posture), so the
/// founder never has to choose which xHC this row drives — which is the whole
/// reason the kernel no longer reads CAPLENGTH and PORTSC through a
/// candidate's BAR to make that decision for anyone. The DMA pool stays
/// founded because it is class-agnostic: sizing it needs no driver knowledge.
///
/// The firmware store (`role::FIRMWARE_STORE`, coordinate 2 — the
/// `kore-rtw89-firmware` boot module `make fetch-realtek-firmware` stages) is
/// the ivpu row's posture exactly: `RoleSlot::indexed_optional`, so a boot
/// with no module resolves `None` and the row founds without its Wi-Fi rung,
/// while a present-but-unclaimable handle still fails the bundle — optional
/// never means unchecked.
pub struct NetEndowment {
    /// The DMA supply the frame host's window opens over.
    pub pool: kore_rt::cap::SupplyToken,
    /// The DMA supply the Wi-Fi function is bound over — its own, so a
    /// radio bound beside an Ethernet adaptor takes nothing from the frame
    /// host's, and a supply spent on one device is never asked to back two.
    pub radio_pool: kore_rt::cap::SupplyToken,
    /// The channel space `Signal` mints receive doorbells from.
    pub channels: kore_rt::cap::ChannelSpaceToken,
    /// The Realtek RTL8922AE firmware byte store when the boot carries the
    /// module — a whole-lease boot module claimed like the image archive.
    /// `None` is the ordinary un-provisioned posture (every QEMU boot).
    pub firmware: Option<kore_rt::cap::ImageStoreToken>,
    /// The MT7925's ROM patch, when the boot carries it.
    pub mt7925_patch: Option<kore_rt::cap::ImageStoreToken>,
    /// The MT7925's N9 RAM code, when the boot carries it.
    pub mt7925_ram: Option<kore_rt::cap::ImageStoreToken>,
}

/// The `role::FIRMWARE_STORE` coordinate of the Wi-Fi firmware module — the
/// third entry of the kernel's `FIRMWARE_MODULES` (`kore-rtw89-firmware`),
/// cited by `services/net/boot.toml`'s `FIRMWARE_STORE@2`.
pub const WIFI_FIRMWARE_INDEX: u16 = 2;

/// The `role::FIRMWARE_STORE` coordinates of the MT7925's **two** blobs —
/// the sixth and seventh entries of the kernel's `FIRMWARE_MODULES`
/// (`kore-mt7925-patch-firmware`, `kore-mt7925-ram-firmware`), cited by
/// `services/net/boot.toml`'s `FIRMWARE_STORE@5` and `@6`.
///
/// Two coordinates because the part takes two images and a lease is whole
/// module: a ROM patch the MCU runs from its own memory, and the N9 RAM
/// code it runs after. Neither is useful without the other, and they are
/// still endowed separately, because a boot that carries one and not the
/// other is a true state the row should be able to report rather than
/// conflate with carrying neither.
pub const MT7925_PATCH_FIRMWARE_INDEX: u16 = 5;
/// See [`MT7925_PATCH_FIRMWARE_INDEX`].
pub const MT7925_RAM_FIRMWARE_INDEX: u16 = 6;

/// The `role::DMA_POOL` coordinate of the radio's supply, cited by
/// `services/net/boot.toml`'s `DMA_POOL@2`. Index 2, not 1: the founder
/// stages `(DMA_POOL, 1)` as the LARGE size class the NPU rows take, and
/// every other index as the ordinary 2 MiB leaf — which the radio's 316 KiB
/// of rings and firmware staging fit inside with room to spare.
pub const RADIO_POOL_INDEX: u16 = 2;

impl kore_rt::service::Bundle for NetEndowment {
    const PROFILE: kore_abi::task::Profile = kore_abi::task::Profile::new(
        "driver",
        &[
            kore_abi::task::RoleSlot::at(kore_abi::task::role::DMA_POOL),
            kore_abi::task::RoleSlot::indexed(kore_abi::task::role::DMA_POOL, RADIO_POOL_INDEX),
            kore_abi::task::RoleSlot::at(kore_abi::task::role::CHANNEL_SPACE),
            kore_abi::task::RoleSlot::indexed_optional(
                kore_abi::task::role::FIRMWARE_STORE,
                WIFI_FIRMWARE_INDEX,
            ),
            kore_abi::task::RoleSlot::indexed_optional(
                kore_abi::task::role::FIRMWARE_STORE,
                MT7925_PATCH_FIRMWARE_INDEX,
            ),
            kore_abi::task::RoleSlot::indexed_optional(
                kore_abi::task::role::FIRMWARE_STORE,
                MT7925_RAM_FIRMWARE_INDEX,
            ),
        ],
    );

    fn resolve(boot: &kore_abi::task::Bootstrap<'_>) -> Result<Self, kore_rt::service::BindError> {
        use kore_abi::task::role;
        use kore_rt::service::BindError;

        let pool_slot = boot
            .handle(role::DMA_POOL, 0)
            .ok_or(BindError::MissingRole {
                role: role::DMA_POOL,
                index: 0,
            })?;
        let pool = verbs::claim_supply(pool_slot).ok_or(BindError::BadHandle {
            role: role::DMA_POOL,
            index: 0,
        })?;
        let radio_pool_slot =
            boot.handle(role::DMA_POOL, RADIO_POOL_INDEX)
                .ok_or(BindError::MissingRole {
                    role: role::DMA_POOL,
                    index: RADIO_POOL_INDEX,
                })?;
        let radio_pool = verbs::claim_supply(radio_pool_slot).ok_or(BindError::BadHandle {
            role: role::DMA_POOL,
            index: RADIO_POOL_INDEX,
        })?;
        let chan_slot = boot
            .handle(role::CHANNEL_SPACE, 0)
            .ok_or(BindError::MissingRole {
                role: role::CHANNEL_SPACE,
                index: 0,
            })?;
        let channels = verbs::claim_channel_space(chan_slot).ok_or(BindError::BadHandle {
            role: role::CHANNEL_SPACE,
            index: 0,
        })?;
        // Firmware is OPTIONAL: a vacant coordinate is `None` (the boot
        // shipped no module), while a PRESENT-but-unclaimable handle still
        // fails the bundle.
        let firmware = match boot.handle(role::FIRMWARE_STORE, WIFI_FIRMWARE_INDEX) {
            Some(fw_slot) => Some(verbs::claim_image_store(fw_slot).ok_or(
                BindError::BadHandle {
                    role: role::FIRMWARE_STORE,
                    index: WIFI_FIRMWARE_INDEX,
                },
            )?),
            None => None,
        };
        // The same posture for each MT7925 blob, and one `claim` apiece
        // rather than a loop: the coordinate has to appear in the refusal,
        // and a loop that lost it would report "some firmware store" on a
        // machine with no serial port.
        let mut claim_at = |index: u16| -> Result<Option<kore_rt::cap::ImageStoreToken>, BindError> {
            match boot.handle(role::FIRMWARE_STORE, index) {
                Some(slot) => Ok(Some(verbs::claim_image_store(slot).ok_or(
                    BindError::BadHandle {
                        role: role::FIRMWARE_STORE,
                        index,
                    },
                )?)),
                None => Ok(None),
            }
        };
        let mt7925_patch = claim_at(MT7925_PATCH_FIRMWARE_INDEX)?;
        let mt7925_ram = claim_at(MT7925_RAM_FIRMWARE_INDEX)?;
        Ok(Self {
            pool,
            radio_pool,
            channels,
            firmware,
            mt7925_patch,
            mt7925_ram,
        })
    }
}

/// The frame **row**: the founded supplies, the claimed controller when the
/// machine has one, and the frame service over it.
///
/// A machine with no adaptor still founds this row and serves — every request
/// answers [`NetRefusal::NotReady`](crate::NetRefusal::NotReady) and the
/// readiness note names why ([`crate::note`]). That is the settled
/// degraded-but-serving posture, and it is what keeps a missing dongle from
/// being a dead boot.
pub struct NetRow {
    pool: Option<kore_rt::cap::SupplyToken>,
    /// The radio's own supply, until the entry ladder binds the Wi-Fi
    /// function over it.
    radio_pool: Option<kore_rt::cap::SupplyToken>,
    channels: Option<kore_rt::cap::ChannelSpaceToken>,
    /// The Wi-Fi firmware store, until the entry ladder maps it for the
    /// claim; `None` on a boot that carried no module.
    firmware: Option<kore_rt::cap::ImageStoreToken>,
    /// The MT7925's two stores, on the same terms. Held separately because
    /// a boot carrying one and not the other is a true state the row
    /// reports rather than conflates with carrying neither.
    mt7925_patch: Option<kore_rt::cap::ImageStoreToken>,
    /// See [`Self::mt7925_patch`].
    mt7925_ram: Option<kore_rt::cap::ImageStoreToken>,
    /// The frame service, adopted by the entry ladder. `None` only before it
    /// runs — unreachable once the row serves.
    host: Option<crate::NetService<MetalHost>>,
    /// Requests refused before a host was adopted: counted, never silent.
    unhosted_refusals: u64,
}

impl NetRow {
    /// Take the founded DMA supply, once.
    pub fn take_pool(&mut self) -> Option<kore_rt::cap::SupplyToken> {
        self.pool.take()
    }

    /// Take the radio's DMA supply, once.
    pub fn take_radio_pool(&mut self) -> Option<kore_rt::cap::SupplyToken> {
        self.radio_pool.take()
    }

    /// Take the founded channel space, once.
    pub fn take_channels(&mut self) -> Option<kore_rt::cap::ChannelSpaceToken> {
        self.channels.take()
    }

    /// Move the Wi-Fi firmware store out for the claim; `None` when the boot
    /// carried no module, and after it was taken.
    pub fn take_firmware(&mut self) -> Option<kore_rt::cap::ImageStoreToken> {
        self.firmware.take()
    }

    /// Take the MT7925's ROM patch store, once.
    pub fn take_mt7925_patch(&mut self) -> Option<kore_rt::cap::ImageStoreToken> {
        self.mt7925_patch.take()
    }

    /// Take the MT7925's N9 RAM code store, once.
    pub fn take_mt7925_ram(&mut self) -> Option<kore_rt::cap::ImageStoreToken> {
        self.mt7925_ram.take()
    }

    /// Adopt the frame service the ladder built.
    pub fn adopt(&mut self, host: crate::NetService<MetalHost>) {
        self.host = Some(host);
    }

    /// The frame service, mutably.
    pub fn service_mut(&mut self) -> Option<&mut crate::NetService<MetalHost>> {
        self.host.as_mut()
    }

    /// The frame service.
    pub fn service(&self) -> Option<&crate::NetService<MetalHost>> {
        self.host.as_ref()
    }

    /// Requests that arrived before a host existed.
    pub fn unhosted_refusals(&self) -> u64 {
        self.unhosted_refusals
    }
}

impl kore_rt::service::SystemService for NetRow {
    /// **A minted face is leaving** — retire the authority it held.
    ///
    /// The host has closed its mappings and endpoints; this closes the vifs
    /// and the doorbell, which are the row's. Returning `false` keeps the
    /// cell closing and retries from the serve loop, so a window lease still
    /// draining a transfer is never retired under it.
    fn peer_closed(&mut self, via: u16, _reason: kore_rt::service::PeerClose) -> bool {
        self.host
            .as_mut()
            .is_none_or(|host| host.close_peer_vifs(via))
    }

    /// The runtime's verdict on an admission this row asked for, handed to
    /// the mint that asked. A verdict arriving before the host is adopted is
    /// impossible — nothing can have asked — and is refused rather than
    /// silently dropped.
    fn peer_admitted(
        &mut self,
        result: Result<kore_rt::svc::PeerId, kore_rt::svc::PeerAdmissionRefusal>,
    ) -> bool {
        self.host
            .as_mut()
            .is_some_and(|host| host.peer_admitted(result))
    }

    /// The terminal path (#809 §L): halt and tear down the hosted controller,
    /// whichever bus it came in on.
    async fn quiesce_devices(&mut self, _deadline_ns: u64) -> kore_rt::service::QuiesceSweep {
        let mut sweep = kore_rt::service::QuiesceSweep::NONE;
        if let Some(host) = self.host.as_mut() {
            host.host_mut().teardown_into(&mut sweep);
        }
        sweep
    }
    const NAME: &'static str = "kore-svc-net";
    type Endowment = NetEndowment;

    /// Exec sends a ring lease plus channel endpoints when admitting a peer.
    /// This shares the host's return mechanism with VIF window transfers.
    /// The peer assembler checks each component's authority. `adopt_window`
    /// requires a `MemLease` with `WINDOW_RIGHTS` before mapping it for CPU
    /// access or deriving the controller's read-only DMA leaf.
    const RETURN_MODE: kore_rt::service::ReturnMode = kore_rt::service::ReturnMode::RetainedAny;

    /// The two hosts this row can claim, **neither required**. Index 0 is
    /// the xHCI class (`entry::SELECTOR_XHCI`), index 1 the 82574L
    /// (`entry::SELECTOR_PCI`); the entry ladder tries PCI first and the
    /// order here is the manifest's, cited by index and never re-typed.
    ///
    /// `required: false` is the whole degraded posture in one field: a machine
    /// whose firmware describes no second controller and no 82574L — or whose
    /// only xHC the input router already took — founds this row anyway, and
    /// it reports.
    ///
    /// The PCI selector names the **exact** identity rather than the Ethernet
    /// class, because the broker cannot un-claim: a class-wide match would
    /// hand this row a NIC the core does not drive and quarantine it for the
    /// boot. `Product::from_pci` is the same fact, re-checked at the claim.
    const DEVICES: &'static [kore_rt::service::DeviceRequest] = &[
        kore_rt::service::DeviceRequest {
            selector: kore_rt::service::DeviceSelector::Pci(pci_selector_for_binding::<
                XhciNetBinding,
            >(
                0x0c,
                0x03,
                0x30,
                kore_rt::service::PciIdentitySelector::ANY,
                PciClaimPosture::RouteCapable,
            )),
            required: false,
        },
        kore_rt::service::DeviceRequest {
            selector: kore_rt::service::DeviceSelector::Pci(pci_selector_for_binding::<
                E1000eNetBinding,
            >(
                0x02,
                0x00,
                0x00,
                kore_rt::service::PciIdentitySelector::exact(
                    kore_e1000e::PCI_VENDOR_INTEL,
                    kore_e1000e::PCI_DEVICE_82574L,
                ),
                PciClaimPosture::RouteCapable,
            )),
            required: false,
        },
        // The Wi-Fi function: class 02:80 (network controller, other), the
        // exact RTL8922AE identity, BAR2 and one MSI-family route.
        kore_rt::service::DeviceRequest {
            selector: kore_rt::service::DeviceSelector::Pci(pci_selector_for_binding::<
                Rtw89NetBinding,
            >(
                0x02,
                0x80,
                0x00,
                kore_rt::service::PciIdentitySelector::exact(
                    kore_rtw89::PCI_VENDOR_REALTEK,
                    kore_rtw89::PCI_DEVICE_RTL8922AE,
                ),
                PciClaimPosture::RouteCapable,
            )),
            required: false,
        },
        // The OTHER Wi-Fi function: the same class 02:80, the exact MT7925
        // identity, BAR0 and one MSI-family route.
        //
        // Its own request rather than a widening of the one above, because
        // the two parts share nothing but a class: the Realtek reads its
        // registers through BAR2 and the MediaTek through BAR0, and
        // `pci_selector_for_binding` derives that demand from the binding.
        // A selector loose enough to admit both would hand each part the
        // other's window.
        kore_rt::service::DeviceRequest {
            selector: kore_rt::service::DeviceSelector::Pci(pci_selector_for_binding::<
                Mt7925NetBinding,
            >(
                0x02,
                0x80,
                0x00,
                kore_rt::service::PciIdentitySelector::exact(
                    kore_mt7925::VENDOR_MEDIATEK,
                    kore_mt7925::DEVICE_MT7925,
                ),
                PciClaimPosture::RouteCapable,
            )),
            required: false,
        },
    ];

    const DEVICE_CLASSES: &'static [kore_rt::service::DeviceClass] = &[
        kore_driver::host::device_class::<XhciNetBinding>(),
        kore_driver::host::device_class::<E1000eNetBinding>(),
        kore_driver::host::device_class::<Rtw89NetBinding>(),
        kore_driver::host::device_class::<Mt7925NetBinding>(),
    ];

    fn bind(endowment: NetEndowment) -> Self {
        Self {
            pool: Some(endowment.pool),
            radio_pool: Some(endowment.radio_pool),
            channels: Some(endowment.channels),
            firmware: endowment.firmware,
            mt7925_patch: endowment.mt7925_patch,
            mt7925_ram: endowment.mt7925_ram,
            host: None,
            unhosted_refusals: 0,
        }
    }

    /// Never reached in production — [`serve_async`](Self::serve_async)
    /// overrides it — but the contract requires it and a row that answered
    /// nothing here would be a trap for the next reader.
    fn serve(
        &mut self,
        req: crate::NetRequest,
        sv: &mut kore_rt::service::Serve<'_, crate::NetResponse>,
    ) {
        match self.host.as_mut() {
            Some(host) => host.serve_step(req, sv),
            None => {
                self.unhosted_refusals = self.unhosted_refusals.saturating_add(1);
                sv.reply(crate::NetResponse::Refused(crate::NetRefusal::NotReady));
            }
        }
    }

    /// The row's real serve path.
    ///
    /// Overridden because three verbs mint a flow and installing the filter
    /// that enforces it is `async` at the seam. Those verbs implement
    /// `ServeVerb::serve_async` and inherit the synchronous default, which
    /// answers `None` — so on the sync step every one of them is refused
    /// `Unsupported`, while decoding perfectly off the wire. Present,
    /// reachable, and dead.
    ///
    /// Nothing is awaited across another service: the install is a function
    /// call into this row's own vif table. The seam is `async` because it
    /// also serves a host double, and keeping one shape for both is what
    /// makes a host test and the founded row answer identically.
    async fn serve_async(
        &mut self,
        req: crate::NetRequest,
        sv: &mut kore_rt::service::Serve<'_, crate::NetResponse>,
    ) {
        match self.host.as_mut() {
            Some(host) => host.serve_step_async(req, sv).await,
            None => {
                self.unhosted_refusals = self.unhosted_refusals.saturating_add(1);
                sv.reply(crate::NetResponse::Refused(crate::NetRefusal::NotReady));
            }
        }
    }

    fn grants_dropped(&mut self, response: &mut crate::NetResponse, emitted: usize) {
        if let Some(host) = self.host.as_mut() {
            host.grants_unwound(response, emitted);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kore_rt::service::{DeviceSelector, PciInterruptRequirement, SystemService};

    #[test]
    fn the_return_post_can_receive_the_channels_in_a_peer_bundle() {
        assert!(matches!(
            NetRow::RETURN_MODE,
            kore_rt::service::ReturnMode::RetainedAny
        ));
    }

    /// The row derives CPU and device leaves under the same window lease and
    /// revokes the lease to remove both when the vif closes.
    #[test]
    fn the_vif_window_lease_can_map_and_retire_both_leaves() {
        assert!(
            crate::client::WINDOW_RIGHTS.contains(kore_abi::cap::Rights::DERIVE),
            "the row parents a CPU leaf and a device leaf from this one lease",
        );
        assert!(
            crate::client::WINDOW_RIGHTS.contains(kore_abi::cap::Rights::REVOKE),
            "the row tears down both leaves by revoking the sublease",
        );
    }

    #[test]
    fn the_xhci_request_stays_at_index_zero_unchanged() {
        let DeviceSelector::Pci(xhci) = NetRow::DEVICES[0].selector else {
            panic!("the xHCI request is a PCI selector");
        };
        assert_eq!((xhci.class, xhci.subclass, xhci.progif), (0x0c, 0x03, 0x30));
        assert_eq!(xhci.identity, kore_rt::service::PciIdentitySelector::ANY);
        assert!(!NetRow::DEVICES[0].required);
        assert_eq!(NetRow::DEVICE_CLASSES.len(), NetRow::DEVICES.len());
    }

    #[test]
    fn the_pci_selector_names_the_82574l_route_capable_with_dma() {
        let DeviceSelector::Pci(nic) = NetRow::DEVICES[1].selector else {
            panic!("the NIC request is a PCI selector");
        };
        assert_eq!((nic.class, nic.subclass, nic.progif), (0x02, 0x00, 0x00));
        assert_eq!(
            nic.identity.exact_pair(),
            Some((
                kore_e1000e::PCI_VENDOR_INTEL,
                kore_e1000e::PCI_DEVICE_82574L
            )),
            "exact, because the broker cannot un-claim"
        );
        assert!(nic.dma);
        assert_eq!(nic.bars, kore_rt::service::PciBarMask::BAR0);
        assert!(
            !matches!(nic.interrupts, PciInterruptRequirement::None),
            "a route-capable claim asks for the interrupt carrier"
        );
        assert!(!NetRow::DEVICES[1].required);
    }

    #[test]
    fn the_wifi_selector_names_the_rtl8922ae_on_bar2_route_capable_with_dma() {
        let DeviceSelector::Pci(radio) = NetRow::DEVICES[2].selector else {
            panic!("the Wi-Fi request is a PCI selector");
        };
        assert_eq!(
            (radio.class, radio.subclass, radio.progif),
            (0x02, 0x80, 0x00)
        );
        assert_eq!(
            radio.identity.exact_pair(),
            Some((
                kore_rtw89::PCI_VENDOR_REALTEK,
                kore_rtw89::PCI_DEVICE_RTL8922AE
            )),
            "exact, because the broker cannot un-claim"
        );
        assert!(radio.dma);
        assert_eq!(
            radio.bars,
            kore_rt::service::PciBarMask::BAR2,
            "BAR0 is legacy port I/O the kernel never maps; the registers are BAR2"
        );
        assert!(
            !matches!(radio.interrupts, PciInterruptRequirement::None),
            "a route-capable claim asks for the interrupt carrier"
        );
        assert!(
            matches!(radio.interrupts, PciInterruptRequirement::MsiFamily),
            "the part has plain MSI and no MSI-X: only the MSI-family requirement can match it"
        );
        assert!(!NetRow::DEVICES[2].required);
        // Four classes: xHCI, the 82574L, the RTL8922AE and the MT7925.
        // The MediaTek part takes a selector of its own rather than widening
        // this one — see `mt7925::tests::the_row_asks_for_this_exact_part`,
        // and note that this test's own `exact_pair` assertion above is what
        // makes a widening impossible to do quietly.
        assert_eq!(NetRow::DEVICE_CLASSES.len(), 4);
        assert_eq!(NetRow::DEVICES.len(), 4);
    }

    #[test]
    fn a_metal_host_with_nothing_claimed_lends_nothing() {
        let mut idle = MetalHost::IDLE;
        assert!(!idle.is_pci());
        assert!(!idle.is_wifi());
        assert!(idle.backend().is_none());
        assert_eq!(idle.founding_note(), 0);
        assert_eq!(idle.detail(), 0);
        assert!(!idle.has_route());
        assert!(matches!(
            idle.carve_own_window(),
            Err(ShareFault::Unsupported)
        ));
        assert!(NetHost::device(&idle).is_none());
        let mut sweep = kore_rt::service::QuiesceSweep::NONE;
        idle.teardown_into(&mut sweep);
        assert_eq!(sweep, kore_rt::service::QuiesceSweep::NONE);
    }

    #[test]
    fn an_idle_host_lends_no_backend_and_refuses_windows() {
        let mut idle = EcmHost::idle();
        assert!(idle.backend().is_none());
        assert_eq!(
            idle.founding_note(),
            0,
            "the claim's note is the claim's own"
        );
        assert_eq!(idle.detail(), 0);
        assert!(!idle.has_route(), "no route to arm parks on");
        assert!(matches!(
            idle.carve_own_window(),
            Err(ShareFault::Unsupported)
        ));
        assert!(NetHost::device(&idle).is_none());
        assert!(idle.take_device().is_none());
    }
}
