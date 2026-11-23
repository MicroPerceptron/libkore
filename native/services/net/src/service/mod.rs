//! **The frame engine** — vif policy, the classifier, and the transmit
//! discipline, over one lending [`NetHost`].
//!
//! Everything here is a pure function of *(host, requests, time)*: the vif
//! table and its generations, the per-pair policy and ownership ledgers, the
//! receive classifier, and the per-frame transmit gates. Nothing in this
//! module maps memory or speaks to the kernel — the single step that does is
//! [`NetHost::adopt_window`], which the production row implements and host
//! tests substitute, so the whole engine is exercised with no hardware and no
//! kernel at all.
//!
//! # Why the engine is generic over a *host* and not a backend
//!
//! A [`NicBackend`] is a short-lived borrow of hardware someone else owns: a
//! USB Ethernet backend borrows the controller, its registers, and the
//! device's state together. It cannot be stored, so `NetService<B: NicBackend>`
//! — the shape `AccelService<A: Accelerator>` uses — does not typecheck, and a
//! service that *owned* the controller would make every other device on it
//! unreachable. [`NicHost`] lends one per drive cycle; [`NetHost`] adds the
//! other half of device ownership, which is what a window mapped into *that
//! device's* address space is kept alive by.
//!
//! # Authority comes from the wiring
//!
//! A client never states a policy. It states a *request*, which is checked
//! against the [`VifPolicy`] its **arrival pair** carries
//! ([`Serve::via`](kore_rt::service::Serve::via) — host-stamped from the
//! pair's own endpoint, never a wire field). That is what makes a forged
//! `EtherTypeMatch::Any` unforgeable rather than merely refused: there is no
//! field on the wire that could carry the power, only one that could ask for
//! it. Vif *ownership* is keyed on the same stamp, so a second face — when a
//! minted one exists — cannot reach the first's vifs even by guessing a
//! [`VifId`].

use kore_abi::device::IovaAddr;
use kore_abi::session::{Notice, SlotRef};
use kore_driver::net::flow::{IpProto, Ipv4Addr};
use kore_driver::net::window::{
    FRAME_AREA_BYTES, MAX_RX_FANOUT, RX_AREA_AT, RX_DEPTH, RX_RING_AT, SLOT_BYTES, TX_AREA_AT,
    TX_DEPTH, WINDOW_BYTES, rx_frame_at, rx_slot_of, tx_frame_at,
};
use kore_driver::net::{
    EtherType, FRAME_MAX, FRAME_MIN, FrameDesc, FrameFilter, FrameHead, LinkState, MAX_VIFS,
    MacAddr, NicBackend, NicCounters, NicHost, PacketFilter, PolicyRefusal, TxCompletion, TxCookie,
    TxStatus, VifId, VifMapping, VifPolicy, VifRefusal, VifTable,
};
use kore_inet::flow::FlowCap;
use kore_inet::flow::Powers;
use kore_inet::row::{Identity, InetRow, OpenGrant, OpenRefusal};
use kore_inet::{FlowInstall, InstallRefusal};
use kore_ring::spsc;
use kore_rt::cap::{CHANNEL_ENDPOINT_RIGHTS, ChanSendToken, ChannelRoot, ChannelSpaceToken, verbs};
use kore_rt::chan::ChanTx;
use kore_rt::service::{GrantArg, Serve};
use kore_rt::svc::{PeerBundlePart, PeerId, ServeTable};

pub mod peers;
use kore_rt::transport::Transport;

use crate::arp::Neighbor;
use crate::verbs::NetVerbs;
use crate::{COOKIE_NET_LINK, COOKIE_NET_RX};

/// Serving pairs the policy and ownership tables are sized for — the
/// runtime's fixed ceiling, so a `via` stamp is always a lawful index.
pub const PAIRS: usize = kore_rt::svc::MAX_SERVE_PAIRS;

/// Transmits one vif may have in flight at once.
///
/// The transmit ring is popped **on completion**, not on submit — head
/// advancing is what tells the client its slot is free, and the device reads
/// that slot long after the descriptor was read. So a pipelined vif looks
/// past its own head ([`spsc::Consumer::peek_at`]) by up to this many
/// records. One would be correct and slow: a single 1514-byte frame per bulk
/// round trip.
pub const MAX_TX_INFLIGHT: u32 = 8;

/// Receive frames one [`drive`](NetService::drive) sweep will take from the
/// backend before returning. Bounded so one busy link cannot starve the serve
/// loop.
pub const RX_BATCH: u32 = 64;

/// Transmit completions reaped per sweep.
const TX_REAP: usize = 16;

/// The device-address plan's stride: vif *ordinal* `n` is proposed at
/// `base + n * WINDOW_BYTES`, where `base` is
/// [`vif_iova_base`](kore_driver::host::vif_iova_base) of the hosted
/// device's own reach and pool span.
///
/// **Not a constant, and that is the fix.** This was `VIF_IOVA_BASE =
/// 0x2_0000_0000` — 8 GiB — guarded only by `> DMA_IOVA_BASE`. That guard
/// was written when every device's pool sat at
/// [`DMA_IOVA_BASE`](kore_driver::host::DMA_IOVA_BASE); it says nothing
/// about a device that cannot *reach* 8 GiB. The MT7925 is a 32-bit DMA
/// master (`res/MT7925_EVIDENCE.md` §15), so its pool is placed at
/// [`NARROW_DMA_IOVA_BASE`](kore_driver::host::NARROW_DMA_IOVA_BASE) and an
/// 8 GiB vif window is an address its WFDMA descriptors cannot carry — it
/// would be truncated into whatever lives at the low 32 bits, silently.
/// The placement now derives from the device, and refuses rather than
/// truncates; see `kore_driver::host::vif_iova_base` for both relations it
/// checks.
///
/// This assertion is the engine's half: whatever base the placement
/// returns, the arena it reserved has to hold every stride this engine
/// hands out.
const _: () = assert!(
    (MAX_VIFS * WINDOW_BYTES) as u64 <= kore_driver::host::VIF_ARENA_BYTES,
    "the vif arena must hold every ordinal's window"
);

// ── the host seam ──────────────────────────────────────────────────────

/// Why a client's window could not be made device-visible.
///
/// The rungs of the one production step host tests cannot execute, kept
/// separate so a metal failure says *which* — a row that founds and silently
/// never transmits is the failure this enum exists to prevent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShareFault {
    /// This host cannot adopt windows at all — a test double, or a row
    /// founded with no device.
    Unsupported,
    /// Nothing was delivered on the return leg, or the record was not a
    /// transferable region.
    NoTransfer,
    /// The transferred region would not map into this task.
    NoView,
    /// The kernel refused the device share (`Derive × MemLease → LeasedIova`).
    NoDeviceLeaf,
}

/// One adopted vif window, as the host hands it back.
///
/// The device address is **returned**, not dictated: the engine proposes a
/// stride out of its own address plan, and the host says where the window
/// actually landed. A host that installs where it was asked returns the
/// proposal unchanged; one whose window came out of a pool that picks its own
/// addresses returns the pool's. Either way the mapping is built from the
/// address the device will really use, rather than from the one someone
/// intended it to use.
pub struct AdoptedWindow<L> {
    /// The whole window, CPU-visible to this task.
    pub bytes: &'static mut [u8],
    /// Where the device sees its first byte.
    pub iova: IovaAddr,
    /// What keeps both mappings alive.
    pub lease: L,
}

/// **What the frame engine needs of whoever owns the device.**
///
/// [`NicHost`] says it can lend a backend; this adds the other half of device
/// ownership — mapping a client's window into that same device's address
/// space, and naming what keeps that mapping alive. They are one trait
/// because they are one fact: only the holder of the device can do either.
///
/// Both methods default to a refusal, so a test double implements *nothing*
/// beyond `NicHost` and the engine's whole vif lifecycle stays reachable
/// through [`NetService::open_vif`]'s adopt closure.
pub trait NetHost: NicHost {
    /// What keeps one adopted vif window alive.
    ///
    /// Retired only through [`VifMapping::retire`], which refuses while a
    /// transfer is outstanding — so the window cannot be unmapped while the
    /// device might still be reading it.
    type Lease;

    /// **The one step host tests cannot execute.** Adopt the window this
    /// request transferred: map it into this task, and install the same
    /// backing into the hosted device's address space at `iova`.
    ///
    /// `iova` is the engine's *proposal* out of its address plan; the returned
    /// [`AdoptedWindow`] says where the window actually landed.
    fn adopt_window(
        &mut self,
        returned: kore_rt::cap::ReturnedCapability,
        iova: IovaAddr,
    ) -> Result<AdoptedWindow<Self::Lease>, ShareFault> {
        let _ = (returned, iova);
        Err(ShareFault::Unsupported)
    }

    /// Retire a lease whose vif has closed and whose transfers have all
    /// completed. Dropping must be safe; a row whose lease is a slot rather
    /// than an affine owner unmaps here.
    fn release_window(&mut self, lease: Self::Lease) {
        let _ = lease;
    }

    /// **How many address bits the hosted device's DMA master can emit** —
    /// its binding's
    /// [`DMA_ADDRESS_BITS`](kore_driver::host::PciBindingContract::DMA_ADDRESS_BITS),
    /// surfaced here because the engine's address plan is a function of it.
    ///
    /// A method rather than an associated const because
    /// [`MetalHost`](crate::metal::MetalHost) dispatches over a PCI or a USB
    /// frame host chosen at founding — the width is a fact about the device
    /// that row actually bound, not about the row's type.
    ///
    /// 64 by default: a test double and a device-less row have nothing to
    /// truncate, and the default is the one every binding in the tree but
    /// the MT7925 declares.
    fn dma_address_bits(&self) -> u32 {
        64
    }

    /// **The span of the hosted device's own DMA pool arena**, the thing a
    /// vif window must not land on top of.
    ///
    /// Measured from the live pool
    /// ([`CapPool::capacity_bytes`](kore_driver::host::CapPool::capacity_bytes)),
    /// never declared: the founder endows a `(DMA_POOL, _)` role with a
    /// 2 MiB leaf and a `(DMA_POOL, 1)` role with a 128 MiB one, and a row
    /// may have grown past either. Reading it is also what keeps the figure
    /// from becoming a second hand-maintained copy of the kernel's
    /// `DRIVER_DMA_POOL_BYTES`, which the kernel deliberately does not
    /// export — `Cargo.toml` states in so many words that no driver crate is
    /// linked into the kernel, so there is no seam to share a constant
    /// across and the number has to be observed instead.
    ///
    /// `0` means no pool is placed — the degraded row, and every host test
    /// double.
    fn dma_pool_bytes(&self) -> u64 {
        0
    }

    /// The device's session slot, for diagnostics. `None` in the degraded
    /// posture.
    fn device(&self) -> Option<SlotRef> {
        None
    }

    /// The bound radio's facts, for [`NetRequest::Radio`]. `None` on a host
    /// with no radio — every Ethernet host, and the default — which the
    /// service reports as `Unsupported` rather than inventing a report.
    fn radio(&self) -> Option<RadioReport> {
        None
    }
}

// ── the vocabulary ─────────────────────────────────────────────────────

/// How vif addresses are assigned.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AddressMode {
    /// Every vif answers to the device's own address; vifs separate by
    /// protocol alone.
    ///
    /// **The default, and the conservative one.** It works on every adaptor
    /// and every upstream network, because nothing new ever appears on the
    /// wire. Per-vif addressing needs a device that will *receive* for a
    /// derived address and an upstream that will *carry* frames sourced from
    /// it — two facts no emulator can establish.
    #[default]
    Shared,
    /// Each vif gets a locally-administered address derived from the
    /// device's. Requires a promiscuous device packet filter, which
    /// [`NetService`] sets for itself when this mode is live.
    PerVif,
}

/// What a client learns about the device and its own authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NetInfo {
    /// Whether the link has carrier.
    pub link_up: bool,
    /// Negotiated rate in megabits per second; 0 when unknown.
    pub link_mbps: u32,
    /// The device's own hardware address.
    pub mac: MacAddr,
    /// The largest frame this service carries.
    pub mtu: u16,
    /// Bytes a client must supply for one vif window.
    pub window_bytes: u32,
    /// The device's counters, read live: interrupts serviced, link changes,
    /// receive overruns and errors.
    pub counters: NicCounters,
    /// Wake notices the transport refused — a wake lost, never a frame.
    pub dropped_notices: u32,
}

/// One vif's counters, as the wire carries them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NetStats {
    /// Frames delivered into this vif's receive ring.
    pub rx_frames: u64,
    /// Frames transmitted from it.
    pub tx_frames: u64,
    /// Frames that did not make it, for any reason.
    pub dropped: u64,
}

/// One vif's counters in full — the service-side view, which separates the
/// reasons a frame did not make it. The wire sums them; a diagnostic should
/// not have to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VifCounters {
    /// Frames delivered into the receive ring.
    pub rx_frames: u64,
    /// Frames the hardware reported sent.
    pub tx_frames: u64,
    /// Receive frames dropped because the vif's ring was full — the KINP
    /// posture: the slow consumer loses its own frames, never everyone's.
    pub rx_dropped: u64,
    /// Transmit descriptors **refused**: an impossible length, an EtherType
    /// outside this vif's own filter, or a source address it may not use.
    /// Refused, never clamped or rewritten.
    pub tx_refused: u64,
    /// Transmits the device reported failed.
    pub tx_failed: u64,
}

impl VifCounters {
    /// The wire view: one `dropped` number, every reason folded in.
    pub fn wire(&self) -> NetStats {
        NetStats {
            rx_frames: self.rx_frames,
            tx_frames: self.tx_frames,
            dropped: self
                .rx_dropped
                .saturating_add(self.tx_refused)
                .saturating_add(self.tx_failed),
        }
    }
}

/// The frame service's request vocabulary.
///
/// Deliberately **not** `Open`/`Bind`/`Send`/`Recv`: there is no send verb
/// and no receive verb at all. Frames move on the vif's own ring, which is
/// the point — a control call per frame would make the queue pair
/// decorative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NetRequest {
    /// Link, address, and the window geometry this pair must supply.
    Query,
    /// Open a vif over the window riding this request's return leg.
    OpenVif {
        /// The filter asked for — checked against the arrival pair's policy,
        /// never taken as given.
        filter: FrameFilter,
        /// The founder-published return-landing correlation value naming the
        /// transfer. Carries no authority itself.
        announced: SlotRef,
    },
    /// Replace the receive filter on a vif this pair already holds.
    ///
    /// The verb an arbiter one layer up needs: `services/inet` mints a flow
    /// capability and this is how that capability stops being a description
    /// and becomes the predicate the classifier runs. `OpenVif` takes a
    /// filter once, at open; without this, a filter could never narrow to a
    /// flow that did not exist yet when the vif was opened.
    ///
    /// Re-authorized in full — holding a vif is not authority to widen what
    /// it accepts.
    SetFilter {
        /// The vif to re-filter. Ownership is re-checked, like every op that
        /// names one.
        vif: VifId,
        /// The filter asked for — checked against the arrival pair's policy,
        /// never taken as given.
        filter: FrameFilter,
    },
    /// Close a vif; every identifier issued for it stops resolving at once.
    CloseVif {
        /// The vif to close.
        vif: VifId,
    },
    /// Mint this pair's receive doorbell.
    Signal,
    /// One vif's counters.
    Stats {
        /// The vif to report.
        vif: VifId,
    },
    /// The bound radio's facts — what its firmware and efuse said at
    /// bring-up. Answered whether or not the radio lends a frame path (at
    /// #388 D1 it does not), so a machine with no serial console can still
    /// read the regulatory root and the PHY shape off the row. Refused
    /// `Unsupported` on a host with no radio.
    Radio,
    // ── the flow program ──────────────────────────────────────────────
    //
    // The arbitrating verbs. On this vocabulary rather than their own
    // because a row serves one protocol, and the arbitration had to come
    // to the table it enforces on — see `NetService::inet`.
    // ── the peer program (`res/PEER_LINK.md` P1) ──────────────────────
    //
    // The row's own mint. A frame client cannot be founder-wired — a
    // chained driver row may be neither a `shell_client` nor a declared
    // provider, both refused at infrastructure build — so a client face on
    // this row exists only because these verbs make one.
    /// Contribute one component of a peer bundle.
    ///
    /// The bundle is assembled across several of these, one component each;
    /// the one that completes it mints a serving pair. Only a FIXED pair may
    /// ask — a minted face cannot mint another.
    AdmitPeer {
        /// Which component this request carries.
        part: PeerBundlePart,
        /// The return-landing correlation value naming the transfer. Carries
        /// no authority itself, exactly as at [`OpenVif`](Self::OpenVif).
        announced: SlotRef,
    },
    /// Collect the verdict on the bundle most recently completed.
    ///
    /// Separate from [`AdmitPeer`](Self::AdmitPeer) because the mint is the
    /// runtime's and completes after the reply to the last component.
    TakePeerAdmission,
    /// Close the minted pair this request arrives on.
    ///
    /// Only a minted face may ask: a fixed pair is the founder's and is not
    /// this row's to retire.
    ClosePeer,
    /// Retire one exact minted generation, from the administration side.
    RetirePeer {
        /// The generation to retire. Exact: a stale identity names a cell
        /// whose occupant has already changed and must not take its
        /// successor down.
        peer: PeerId,
    },
    /// Ask whether that generation's cell is free again.
    ConfirmRetired {
        /// The generation asked about.
        peer: PeerId,
    },
    /// Report the machine's address, prefix, gateway and resolver.
    ///
    /// Distinct from [`Query`](Self::Query), which answers for a vif and
    /// the link under it. This answers for the station.
    InetQuery,
    /// Give the machine its address.
    Configure {
        /// The station address.
        addr: Ipv4Addr,
        /// Its prefix length, 1..=32.
        prefix_len: u8,
        /// The default gateway, if there is one.
        gateway: Option<Ipv4Addr>,
    },
    /// Mint a flow to one peer, binding a local port.
    OpenFlow {
        /// The vif the flow's filter is installed on.
        vif: VifId,
        /// The transport.
        proto: IpProto,
        /// The local port, or `None` to draw an ephemeral one.
        local_port: Option<u16>,
        /// The peer this flow is with.
        peer: (Ipv4Addr, u16),
    },
    /// Mint a listening flow: a local endpoint, any peer.
    OpenListener {
        /// The vif the flow's filter is installed on.
        vif: VifId,
        /// The transport.
        proto: IpProto,
        /// The local port.
        local_port: u16,
    },
    /// Release a vif's flows and the ports they hold.
    CloseFlow {
        /// The vif whose flows go.
        vif: VifId,
    },
}

/// The frame service's reply vocabulary. Data only — the one authority that
/// ever leaves is the doorbell recv riding [`NetResponse::Signaled`]'s grant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NetResponse {
    /// The device's facts.
    Info(NetInfo),
    /// A vif was opened.
    Opened {
        /// Its identity — index and generation.
        vif: VifId,
        /// The address it answers to.
        mac: MacAddr,
    },
    /// A vif's receive filter was replaced. From this reply onward the
    /// classifier runs the new predicate; the old one admits nothing further.
    Filtered {
        /// The identity it was replaced on.
        vif: VifId,
    },
    /// A vif was closed. Its identifier no longer resolves, whether or not
    /// its mapping has finished retiring.
    Closed {
        /// The identity that was closed.
        vif: VifId,
    },
    /// The receive doorbell was minted; one grant rides along.
    Signaled,
    /// One vif's counters.
    Stats {
        /// Whose.
        vif: VifId,
        /// The counters.
        stats: NetStats,
    },
    /// The bound radio's facts.
    Radio(RadioReport),
    // ── the peer program ──────────────────────────────────────────────
    /// The component was taken into the bundle. Not yet a face: the mint
    /// happens when the bundle completes, and its verdict is
    /// [`TakePeerAdmission`](NetRequest::TakePeerAdmission)'s.
    PeerAccepted,
    /// The bundle completed and a serving pair was minted.
    PeerAdmitted {
        /// The identity requests off the minted pair are stamped with.
        peer: PeerId,
    },
    /// The minted pair is retiring.
    PeerClosing,
    /// That generation's cell is free; nothing of it remains.
    PeerRetired,
    // ── the flow program ──────────────────────────────────────────────
    /// The station's identity, as the row holds it.
    Identity {
        /// `None` before anything configured one.
        identity: Option<IdentityReport>,
    },
    /// A flow was minted.
    Flow {
        /// The transport.
        proto: IpProto,
        /// The endpoint it bound.
        local: (Ipv4Addr, u16),
        /// The peer, or `None` for a listener — which pins no peer, and
        /// says so in the capability rather than in a flag beside it.
        peer: Option<(Ipv4Addr, u16)>,
    },
    /// A vif's flows were released.
    FlowClosed {
        /// How many ports came back.
        released: u16,
    },

    /// Refused, by name.
    Refused(NetRefusal),
}

/// The bound radio's facts, as the bring-up ladder read them.
///
/// Plain values, deliberately: the frame service's vocabulary imports no
/// hardware type, so a client decodes each code against the upstream
/// definition the driver core cites rather than against a type it cannot
/// name. The service interprets none of it either — it is a report, not a
/// capability advertisement (that is D2's, built from the same facts).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RadioReport {
    /// The chip cut.
    pub cut: u8,
    /// The firmware the row verified against its pin and downloaded.
    pub firmware: RadioFirmware,
    /// The efuse country code: two ASCII letters, or `0xFF 0xFF` when
    /// unprogrammed. The regulatory root D2 keys on.
    pub country: [u8; 2],
    /// The efuse crystal trim.
    pub xtal_cap: u8,
    /// The efuse RF front-end type the firmware's tables are indexed by.
    pub rfe_type: u8,
    /// Whether the MAC's own autoload from the efuse completed.
    pub efuse_autoload: bool,
    /// Normal DMAC/CMAC0 configuration completed before the alive reply.
    pub normal_mac: bool,
    /// BB/RF register setup completed; calibration and PHY liveness are separate.
    pub phy_configured: bool,
    /// Startup MAC feature commands returned successful firmware done acknowledgements.
    pub mac_features: bool,
    /// Whether the WCPU verifies the image against a key.
    pub secure_boot: bool,
    /// Secure boot: the key-management selector the image section matched.
    pub sb_sel_mgn: u32,
    /// Secure boot: the key-management device type, in index form.
    pub mss_dev_type: u8,
    /// Secure boot: the customer index.
    pub mss_cust_idx: u8,
    /// Secure boot: the key number.
    pub mss_key_num: u8,
    /// What the firmware answered to the alive handshake.
    pub phy: RadioPhy,
    /// Packets the download posted on the firmware-command channel, across
    /// attempts (saturating: one download is on the order of a thousand).
    pub download_packets: u32,
}

/// The downloaded firmware's identity: the container's inner header version
/// and the image's own version words.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RadioFirmware {
    /// The container's inner header version the download used.
    pub header: u8,
    /// Version word 1, bits `[7:0]`.
    pub major: u8,
    /// Version word 1, bits `[15:8]`.
    pub minor: u8,
    /// Version word 1, bits `[23:16]`.
    pub sub: u8,
    /// Version word 1, bits `[31:24]`.
    pub idx: u8,
    /// Version word 2: the commit identifier.
    pub commit: u32,
}

/// The PHY's shape as the firmware reported it (`PHY_CAP`), every field the
/// firmware's own code — decoded by a reader, never by this service.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RadioPhy {
    /// Receive spatial streams.
    pub rx_nss: u8,
    /// The widest bandwidth code.
    pub bw: u8,
    /// Transmit spatial streams.
    pub tx_nss: u8,
    /// Protocol capability code.
    pub prot: u8,
    /// NIC type code.
    pub nic: u8,
    /// WLAN function bits.
    pub wl_func: u8,
    /// Hardware type code.
    pub hw_type: u8,
    /// Transmit antennas.
    pub ant_tx: u8,
    /// Receive antennas.
    pub ant_rx: u8,
    /// Band selection code.
    pub band_sel: u8,
}

/// Why a request was refused. Every refusal names the power that was missing
/// or the identity that did not resolve; none of them is a clamp.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NetRefusal {
    /// The verb is not available in this posture.
    Unsupported,
    /// No device: the row founded degraded, and says so rather than pretending.
    NotReady,
    /// An EtherType outside this pair's grant — including a forged `Any`.
    PolicyEtherType,
    /// A multicast group this pair may not join.
    PolicyMulticast,
    /// Broadcast without the grant.
    PolicyBroadcast,
    /// Promiscuous receive without the grant.
    PolicyPromiscuous,
    /// More vifs than this pair may hold.
    PolicyVifs,
    /// The pair may only bind a restricted flow, and the request bound none.
    PolicyFlowRequired,
    /// An IP protocol outside this pair's flow grant.
    PolicyFlowProto,
    /// A local endpoint outside this pair's flow grant.
    PolicyFlowLocal,
    /// A peer endpoint outside this pair's flow grant.
    PolicyFlowPeer,
    /// The identifier names no live vif.
    NoVif,
    /// The identifier is from before a close — a use-after-close, named
    /// distinctly so the answer says so.
    StaleVif,
    /// The vif belongs to another pair.
    NotOwner,
    /// The window was absent, malformed, or not a well-formed ring pair.
    BadWindow,
    /// The vif table is full.
    Exhausted,
    /// A transfer is still outstanding.
    Busy,
    /// The window could not be made device-visible.
    NoMapping,
    /// No channel space to mint a doorbell from.
    NoChannel,
    /// A refusal the service could not attribute.
    Internal,
    /// The arrival pair may not assemble peer bundles.
    PeerDenied,
    /// The component offered is not one this bundle can take.
    BadPeerBundle,

    // ── the flow program ──────────────────────────────────────────────
    /// The machine has no address, so no flow can be meaningful yet.
    NoIdentity,
    /// The address offered is not one a station may hold.
    NotAStation,
    /// The prefix length is outside 1..=32.
    BadPrefix,
    /// The gateway is not reachable on the prefix given.
    GatewayOffLink,
    /// Another vif holds that protocol and port.
    PortTaken,
    /// Below 1024 without the grant that permits it.
    PortPrivileged,
    /// Port zero is the wire's wildcard and never a binding.
    PortReserved,
    /// Every ephemeral port is spoken for.
    NoEphemeral,
    /// The port table is full.
    PortsExhausted,
    /// No route to that peer from this station.
    Unreachable,
    /// That vif already carries a flow; one vif holds one filter.
    VifAlreadyBound,
    /// This vif has spent its socket budget.
    SocketsExhausted,
    /// The vif went away between the mint and the install.
    FrameLost,
    /// Policy refused the filter the mint would have installed.
    FlowPolicy,
}

impl NetRefusal {
    /// The wire code.
    pub fn code(self) -> u8 {
        use kore_abi::svc::NetRepWire as W;
        match self {
            Self::Unsupported => W::REFUSE_UNSUPPORTED,
            Self::NotReady => W::REFUSE_NOT_READY,
            Self::PolicyEtherType => W::REFUSE_POLICY_ETHERTYPE,
            Self::PolicyMulticast => W::REFUSE_POLICY_MULTICAST,
            Self::PolicyBroadcast => W::REFUSE_POLICY_BROADCAST,
            Self::PolicyPromiscuous => W::REFUSE_POLICY_PROMISCUOUS,
            Self::PolicyVifs => W::REFUSE_POLICY_VIFS,
            Self::PolicyFlowRequired => W::REFUSE_POLICY_FLOW_REQUIRED,
            Self::PolicyFlowProto => W::REFUSE_POLICY_FLOW_PROTO,
            Self::PolicyFlowLocal => W::REFUSE_POLICY_FLOW_LOCAL,
            Self::PolicyFlowPeer => W::REFUSE_POLICY_FLOW_PEER,
            Self::NoVif => W::REFUSE_NO_VIF,
            Self::StaleVif => W::REFUSE_STALE_VIF,
            Self::NotOwner => W::REFUSE_NOT_OWNER,
            Self::BadWindow => W::REFUSE_BAD_WINDOW,
            Self::Exhausted => W::REFUSE_EXHAUSTED,
            Self::Busy => W::REFUSE_BUSY,
            Self::NoMapping => W::REFUSE_NO_MAPPING,
            Self::NoChannel => W::REFUSE_NO_CHANNEL,
            Self::Internal => W::REFUSE_INTERNAL,
            Self::NoIdentity => W::REFUSE_NO_IDENTITY,
            Self::NotAStation => W::REFUSE_NOT_A_STATION,
            Self::BadPrefix => W::REFUSE_BAD_PREFIX,
            Self::GatewayOffLink => W::REFUSE_GATEWAY_OFF_LINK,
            Self::PortTaken => W::REFUSE_PORT_TAKEN,
            Self::PortPrivileged => W::REFUSE_PORT_PRIVILEGED,
            Self::PortReserved => W::REFUSE_PORT_RESERVED,
            Self::NoEphemeral => W::REFUSE_NO_EPHEMERAL,
            Self::PortsExhausted => W::REFUSE_PORTS_EXHAUSTED,
            Self::Unreachable => W::REFUSE_UNREACHABLE,
            Self::VifAlreadyBound => W::REFUSE_VIF_ALREADY_BOUND,
            Self::SocketsExhausted => W::REFUSE_SOCKETS_EXHAUSTED,
            Self::FrameLost => W::REFUSE_FRAME_LOST,
            Self::PeerDenied => W::REFUSE_PEER_DENIED,
            Self::BadPeerBundle => W::REFUSE_BAD_PEER_BUNDLE,
            Self::FlowPolicy => W::REFUSE_FLOW_POLICY,
        }
    }

    /// The wire code's inverse; `None` for a code this version does not know
    /// — refused rather than folded into a generic failure.
    pub fn from_code(code: u8) -> Option<Self> {
        use kore_abi::svc::NetRepWire as W;
        Some(match code {
            W::REFUSE_UNSUPPORTED => Self::Unsupported,
            W::REFUSE_NOT_READY => Self::NotReady,
            W::REFUSE_POLICY_ETHERTYPE => Self::PolicyEtherType,
            W::REFUSE_POLICY_MULTICAST => Self::PolicyMulticast,
            W::REFUSE_POLICY_BROADCAST => Self::PolicyBroadcast,
            W::REFUSE_POLICY_PROMISCUOUS => Self::PolicyPromiscuous,
            W::REFUSE_POLICY_VIFS => Self::PolicyVifs,
            W::REFUSE_POLICY_FLOW_REQUIRED => Self::PolicyFlowRequired,
            W::REFUSE_POLICY_FLOW_PROTO => Self::PolicyFlowProto,
            W::REFUSE_POLICY_FLOW_LOCAL => Self::PolicyFlowLocal,
            W::REFUSE_POLICY_FLOW_PEER => Self::PolicyFlowPeer,
            W::REFUSE_NO_VIF => Self::NoVif,
            W::REFUSE_STALE_VIF => Self::StaleVif,
            W::REFUSE_NOT_OWNER => Self::NotOwner,
            W::REFUSE_BAD_WINDOW => Self::BadWindow,
            W::REFUSE_EXHAUSTED => Self::Exhausted,
            W::REFUSE_BUSY => Self::Busy,
            W::REFUSE_NO_MAPPING => Self::NoMapping,
            W::REFUSE_NO_CHANNEL => Self::NoChannel,
            W::REFUSE_INTERNAL => Self::Internal,
            W::REFUSE_NO_IDENTITY => Self::NoIdentity,
            W::REFUSE_NOT_A_STATION => Self::NotAStation,
            W::REFUSE_BAD_PREFIX => Self::BadPrefix,
            W::REFUSE_GATEWAY_OFF_LINK => Self::GatewayOffLink,
            W::REFUSE_PORT_TAKEN => Self::PortTaken,
            W::REFUSE_PORT_PRIVILEGED => Self::PortPrivileged,
            W::REFUSE_PORT_RESERVED => Self::PortReserved,
            W::REFUSE_NO_EPHEMERAL => Self::NoEphemeral,
            W::REFUSE_PORTS_EXHAUSTED => Self::PortsExhausted,
            W::REFUSE_UNREACHABLE => Self::Unreachable,
            W::REFUSE_VIF_ALREADY_BOUND => Self::VifAlreadyBound,
            W::REFUSE_SOCKETS_EXHAUSTED => Self::SocketsExhausted,
            W::REFUSE_FRAME_LOST => Self::FrameLost,
            W::REFUSE_PEER_DENIED => Self::PeerDenied,
            W::REFUSE_BAD_PEER_BUNDLE => Self::BadPeerBundle,
            W::REFUSE_FLOW_POLICY => Self::FlowPolicy,
            _ => return None,
        })
    }
}

impl From<PolicyRefusal> for NetRefusal {
    fn from(refusal: PolicyRefusal) -> Self {
        match refusal {
            PolicyRefusal::EtherType => Self::PolicyEtherType,
            PolicyRefusal::Multicast => Self::PolicyMulticast,
            PolicyRefusal::Broadcast => Self::PolicyBroadcast,
            PolicyRefusal::Promiscuous => Self::PolicyPromiscuous,
            PolicyRefusal::SourceMac => Self::PolicyEtherType,
            PolicyRefusal::TooManyVifs => Self::PolicyVifs,
            PolicyRefusal::FlowRequired => Self::PolicyFlowRequired,
            PolicyRefusal::FlowProto => Self::PolicyFlowProto,
            PolicyRefusal::FlowLocal => Self::PolicyFlowLocal,
            PolicyRefusal::FlowPeer => Self::PolicyFlowPeer,
        }
    }
}

impl From<kore_inet::row::OpenRefusal> for NetRefusal {
    fn from(refusal: kore_inet::row::OpenRefusal) -> Self {
        use kore_inet::row::{BindRefusal as B, OpenRefusal as R};
        match refusal {
            R::NoIdentity => Self::NoIdentity,
            R::Unreachable => Self::Unreachable,
            R::VifAlreadyBound => Self::VifAlreadyBound,
            R::StationAlreadyRaw => Self::VifAlreadyBound,
            R::RawL2NotGranted => Self::PolicyEtherType,
            R::Port(B::Taken) => Self::PortTaken,
            R::Port(B::Privileged) => Self::PortPrivileged,
            R::Port(B::Reserved) => Self::PortReserved,
            R::Port(B::NoEphemeral) => Self::NoEphemeral,
            R::Port(B::Exhausted) => Self::PortsExhausted,
            R::Install(i) => Self::from(i),
        }
    }
}

impl From<kore_inet::InstallRefusal> for NetRefusal {
    /// The install half, on its own: a close consults the same seam without
    /// an [`OpenRefusal`](kore_inet::row::OpenRefusal) to carry it.
    ///
    /// `NoFrameFace` cannot occur from this composition -- the install is a
    /// function call -- but the seam still names it, so it maps to the
    /// generic fault rather than being unreachable-asserted.
    fn from(refusal: kore_inet::InstallRefusal) -> Self {
        use kore_inet::InstallRefusal as I;
        match refusal {
            I::NoVif => Self::NoVif,
            I::NotOwner => Self::NotOwner,
            I::Lost => Self::FrameLost,
            I::Policy(p) => Self::from(p),
            I::NoFrameFace => Self::Internal,
        }
    }
}

impl From<kore_inet::row::IdentityRefusal> for NetRefusal {
    fn from(refusal: kore_inet::row::IdentityRefusal) -> Self {
        use kore_inet::row::IdentityRefusal as R;
        match refusal {
            R::NotAStation => Self::NotAStation,
            R::BadPrefix => Self::BadPrefix,
            R::GatewayOffLink => Self::GatewayOffLink,
        }
    }
}

impl From<VifRefusal> for NetRefusal {
    fn from(refusal: VifRefusal) -> Self {
        match refusal {
            VifRefusal::OutOfRange | VifRefusal::Closed => Self::NoVif,
            VifRefusal::Stale => Self::StaleVif,
        }
    }
}

/// What one [`drive`](NetService::drive) sweep moved.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DriveReport {
    /// Frames handed to the hardware.
    pub tx_submitted: u32,
    /// Transmit completions reaped.
    pub tx_completed: u32,
    /// Receive frames taken off the wire (**not** the number of ring
    /// writes — one frame may fan out to several vifs).
    pub rx_polled: u32,
    /// Ring writes the classifier made.
    pub rx_delivered: u32,
    /// Copies the classifier could not make: a full vif ring, or fan-out past
    /// [`MAX_RX_FANOUT`].
    pub rx_dropped: u32,
    /// Transmit descriptors refused by the per-frame gates.
    pub tx_refused: u32,
    /// Bitmask of serving pairs whose vifs received something — one wake
    /// notice per set bit, after the sweep.
    pub wake: u16,
    /// The link state differs from the last sweep's — one link notice per
    /// pair holding a doorbell, after the sweep. An edge derived from the
    /// live state, so a cause the device coalesced or the driver missed
    /// cannot leave the service believing the wrong state.
    pub link_changed: bool,
    /// Earliest link-management alarm, not evidence of frame activity.
    pub service_deadline: Option<u64>,
}

impl DriveReport {
    /// Whether the sweep did anything at all.
    pub fn is_idle(&self) -> bool {
        Self {
            service_deadline: None,
            ..*self
        } == Self::default()
    }
}

// ── the transmit cookie ────────────────────────────────────────────────

/// Pack a vif identity and a transmit slot into the backend's correlation
/// token.
///
/// The generation rides along on purpose: a completion for a vif that was
/// closed mid-flight is caught by **the same generation check that protects
/// everything else**, rather than crediting the slot's next tenant. One
/// mechanism, two jobs.
fn pack_cookie(vif: VifId, slot: u32) -> TxCookie {
    TxCookie(((vif.index as u64) << 48) | ((vif.generation as u64) << 32) | slot as u64)
}

/// [`pack_cookie`]'s inverse.
fn unpack_cookie(cookie: TxCookie) -> (VifId, u32) {
    (
        VifId {
            index: (cookie.0 >> 48) as u16,
            generation: (cookie.0 >> 32) as u16,
        },
        cookie.0 as u32,
    )
}

// ── per-vif and per-pair state ─────────────────────────────────────────

/// One live vif: its authority, its rings, its window, and its mapping.
struct Vif<L> {
    /// The pair that opened it — re-checked on **every** op that names it,
    /// so a second face cannot reach this vif by guessing its identity.
    via: u16,
    /// The policy-approved filter. Not what was asked for: what was granted.
    filter: FrameFilter,
    /// The scoped ARP exchange installed alongside a connected IP flow.
    neighbor: Option<Neighbor>,
    /// The address this vif answers to.
    mac: MacAddr,
    /// Its ordinal in the device address plan — freed on close.
    ordinal: u8,
    /// The frame area, past both rings: transmit slots then receive slots.
    frames: &'static mut [u8],
    /// The client's transmit ring. Popped **on completion**, never on submit.
    tx: spsc::Consumer<'static, FrameDesc>,
    /// This vif's receive ring.
    rx: spsc::Producer<'static, FrameDesc>,
    /// The device mapping over the transmit slots.
    mapping: VifMapping<L>,
    /// Transmit records popped so far — this consumer's own cursor, mirrored
    /// because [`spsc::Consumer`] publishes only the ring's head and the slot
    /// a record occupies is *derived* from its position, never read out of a
    /// field the client controls.
    tx_popped: u64,
    /// Receive pushes so far — the same derivation on the other side
    /// (doctrine §4: slot ownership is derived from ring position).
    rx_count: u64,
    /// Counters.
    counters: VifCounters,
}

/// A vif whose identity is already dead but whose mapping the device may
/// still be reading.
///
/// Close is two-phase for exactly one reason: the security half must not
/// wait. The generation bumps at once — the client's identifier stops
/// resolving on the same request — while the device leaf is retired later,
/// once the controller has finished with the frames it was handed.
struct Closing<L> {
    /// The identity it was closed under, so a late completion still finds it.
    id: VifId,
    /// Its ordinal, released when the mapping retires.
    ordinal: u8,
    /// The mapping, refusing to retire while transfers are outstanding.
    mapping: VifMapping<L>,
}

/// A minted receive doorbell: the **root** is the revocation pivot, the
/// **send** is this service's lane. The mint's own recv retires immediately —
/// the client's is a fresh derive staged in the reply's grant session.
struct SignalLane {
    root: ChannelRoot,
    send: ChanSendToken,
}

impl SignalLane {
    /// Revoke the lane: root first (that is the revocation), then the send's
    /// bookkeeping retire.
    fn retire(self) -> Result<(), Self> {
        let Self { root, send } = self;
        match root.try_retire() {
            Ok(_) => {
                let _ = send.try_retire();
                Ok(())
            }
            Err(root) => Err(Self { root, send }),
        }
    }
}

/// One serving pair's session state.
struct PairSession {
    /// Its receive doorbell, if it has asked for one.
    signal: Option<SignalLane>,
    /// For a MINTED cell, the exact arrival occupying it; `None` for a fixed
    /// pair and for a cell whose occupant has gone.
    ///
    /// This is what makes retirement answerable: `RetirePeer` acts only on
    /// the generation that is actually here, and `ConfirmRetired` is true
    /// precisely when it is not.
    arrival: Option<u16>,
}

impl PairSession {
    const EMPTY: Self = Self {
        signal: None,
        arrival: None,
    };
}

// ── the service ────────────────────────────────────────────────────────

/// The L2 frame service — one device behind the vif protocol.
///
/// A machine with no adaptor still constructs one: every verb answers
/// [`NetRefusal::NotReady`] and the row's readiness beacon says why. A driver
/// row that refuses to found because its hardware is absent takes the boot
/// down with it; this one reports.
pub struct NetService<H: NetHost> {
    host: H,
    /// The rendezvous mint receive doorbells are rooted in; `None` hosts a
    /// poll-only service.
    channels: Option<ChannelSpaceToken>,
    /// **Authority, per serving pair.** Indexed by
    /// [`Serve::via`](kore_rt::service::Serve::via) — the host-stamped pair
    /// identity, which is why this is a fact about the wiring and not about
    /// any payload. One entry is live today; the shape is what a minted
    /// second face plugs into without reshaping the engine.
    policy: [VifPolicy; PAIRS],
    sessions: [PairSession; PAIRS],
    vifs: VifTable<Vif<H::Lease>>,
    /// Vifs whose identity is dead and whose mapping is still draining.
    closing: [Option<Closing<H::Lease>>; MAX_VIFS],
    /// The device-address plan: which strides of the arena
    /// [`vif_iova_base`](kore_driver::host::vif_iova_base) placed are spent.
    /// Deliberately separate from the vif table's own slot allocation — the
    /// two happen to move together, and nothing here relies on it.
    ordinals: [bool; MAX_VIFS],
    /// **The arbitrating half of the network program.**
    ///
    /// The machine's address and the port space — the two things that must
    /// be exactly one party — and the mint that turns a request into a
    /// flow. It lives here rather than in a row of its own because a flow
    /// is only authority once a `FrameFilter` enforces it, and the table
    /// those filters go on is three fields up.
    ///
    /// `res/INET_MIGRATION.md` §5.7a is why: reaching this table from
    /// another row needs a founder face onto a *chained* provider, which
    /// `declared_client_topology_supported` refuses at compile time and
    /// which the core chain ladder has no position for. Rather than add a
    /// founder mechanism for one caller, the caller moved. The engine is
    /// still a library (`crates/inet`), so this is a composition and not a
    /// merge of concerns.
    inet: InetRow,
    address_mode: AddressMode,
    /// Which pair's `Signal` the in-flight grant session belongs to.
    last_signal_via: Option<u16>,
    served: u64,
    refused: u64,
    dropped_notices: u64,
    /// Completions whose cookie matched no live or closing vif — a number
    /// that should stay zero, and a diagnostic if it does not.
    orphan_completions: u64,
    /// The link as the last sweep saw it — the level an edge is derived
    /// against.
    link: LinkState,
    /// Edges [`drive`](Self::drive) derived, ever.
    link_edges: u64,
    /// **The row's own mint** — the only way a frame client obtains a face,
    /// both founder routes being refused at infrastructure build. See
    /// [`peers`].
    peers: peers::Peers,
}

impl<H: NetHost> NetService<H> {
    /// Host `host` behind the protocol, with `policy[k]` the authority of
    /// serving pair `k`.
    pub fn host(
        mut host: H,
        channels: Option<ChannelSpaceToken>,
        policy: [VifPolicy; PAIRS],
    ) -> Self {
        // The level the first sweep's edge is derived against: what the
        // device says now, so a link that was up at founding is not
        // announced as having come up on the first sweep.
        let link = host.backend().map(|b| b.link()).unwrap_or_default();
        Self {
            host,
            channels,
            policy,
            sessions: [const { PairSession::EMPTY }; PAIRS],
            vifs: VifTable::new(),
            closing: [const { None }; MAX_VIFS],
            ordinals: [false; MAX_VIFS],
            inet: InetRow::new(),
            address_mode: AddressMode::Shared,
            last_signal_via: None,
            served: 0,
            refused: 0,
            dropped_notices: 0,
            orphan_completions: 0,
            link,
            link_edges: 0,
            peers: peers::Peers::new(),
        }
    }

    /// **Reserve this row's transient geometry for minted pair windows.**
    ///
    /// Geometry reserves address space only — each admitted client supplies
    /// its own backing — so this hands the mint a base and nothing else.
    /// Until it is called, peer admission refuses
    /// [`NetRefusal::Unsupported`] rather than inventing an address.
    pub fn configure_peer_map(&mut self, base: u64) {
        self.peers.map = Some(peers::peer_map(base));
    }

    /// **Retire everything a departing minted face owned.**
    ///
    /// The domain half of peer teardown: the host closes its mappings and
    /// endpoints, and this closes the authority. Without it a face's vifs
    /// outlive it — the classifier keeps writing frames into rings nobody
    /// reads, the window leases are never released, and the vif table fills
    /// over a boot's worth of launches.
    ///
    /// Keyed on the full `via`, so only *this* occupant's vifs go: a
    /// successor in the same cell has a different arrival and owns none of
    /// them.
    ///
    /// Returns whether the face now holds nothing — `false` keeps the host's
    /// slot closing and retries from the serve loop, which is what a lease
    /// still draining a transfer needs.
    pub fn close_peer_vifs(&mut self, via: u16) -> bool {
        // Collected before anything closes, because closing mutates the
        // table being walked. `MAX_VIFS` slots for at most `MAX_VIFS` vifs,
        // so the array cannot overflow.
        let mut doomed = [const { None }; MAX_VIFS];
        let mut found = 0;
        for (id, vif) in self.vifs.iter() {
            if vif.via == via {
                doomed[found] = Some(id);
                found += 1;
            }
        }
        for id in doomed.into_iter().flatten() {
            let _ = self.close_vif(via, id);
        }
        // The doorbell is the cell's, not the vif's, so it is retired beside
        // them — a stale lane would otherwise ring for the next occupant.
        if let Some(session) = pair_index(via).and_then(|cell| self.sessions.get_mut(cell))
            && let Some(lane) = session.signal.take()
            && let Err(lane) = lane.retire()
        {
            session.signal = Some(lane);
            return false;
        }
        // The cell is free only once nothing of this occupant remains, so
        // the record is cleared last — `ConfirmRetired` reads it.
        let done = !self.vifs.iter().any(|(_, vif)| vif.via == via);
        if done
            && let Some(session) = pair_index(via).and_then(|cell| self.sessions.get_mut(cell))
            && session.arrival == Some(via)
        {
            session.arrival = None;
        }
        done
    }

    /// Whether `peer` is the generation presently occupying its cell.
    ///
    /// Exact, because a stale identity names a cell whose occupant has
    /// already changed: acting on it would retire that occupant's successor.
    pub fn peer_is_current(&self, peer: PeerId) -> bool {
        self.sessions
            .get(peer.slot())
            .is_some_and(|session| session.arrival == Some(peer.arrival()))
    }

    /// Serve one peer verb. `None` for a verb that is not the mint's.
    ///
    /// The two retirement verbs are answered here rather than in
    /// [`peers`] because both turn on the cell roster this type owns: whether
    /// the generation named is the one actually seated.
    pub fn peer_arm(
        &mut self,
        req: &NetRequest,
        sv: &mut Serve<'_, NetResponse>,
    ) -> Option<NetResponse> {
        match req {
            NetRequest::RetirePeer { peer } => {
                // Admin-only and EXACT: a stale identity names a cell whose
                // occupant has already changed, and retiring on it would
                // take that occupant's successor down.
                if !crate::wire::is_admin_pair(sv.via()) || !self.peer_is_current(*peer) {
                    return Some(NetResponse::Refused(NetRefusal::PeerDenied));
                }
                sv.retire_peer(*peer, kore_rt::service::PeerClose::Orderly);
                Some(NetResponse::PeerClosing)
            }
            NetRequest::ConfirmRetired { peer } => {
                if !crate::wire::is_admin_pair(sv.via()) {
                    return Some(NetResponse::Refused(NetRefusal::PeerDenied));
                }
                // Retired precisely when the cell no longer holds it. While
                // it still does the teardown is in flight, which is `Busy`
                // and not a refusal to try again.
                if self.peer_is_current(*peer) {
                    Some(NetResponse::Refused(NetRefusal::Busy))
                } else {
                    Some(NetResponse::PeerRetired)
                }
            }
            _ => self.peers.handle(req, sv),
        }
    }

    /// Record the runtime's verdict on an admission this row requested.
    ///
    /// `accepted` is this row's own judgement on the identity the host
    /// chose: a peer seated in a cell a FIXED pair already occupies is
    /// refused, because `via` would otherwise name two different faces.
    pub fn peer_admitted(
        &mut self,
        result: Result<PeerId, kore_rt::svc::PeerAdmissionRefusal>,
    ) -> bool {
        // The **slot**, not the arrival: an arrival is
        // `generation * SLOT_CAPACITY + slot`, so every live arrival is
        // numerically past the fixed bank and testing it would accept every
        // collision. The cell a face occupies is what `via` indexes.
        let accepted = result
            .as_ref()
            .is_ok_and(|peer| peer.slot() >= crate::wire::NET_SERVE_PAIRS);
        if accepted && let Ok(peer) = result.as_ref() {
            // The cell records the exact generation it now holds, so a
            // retirement naming a stale one cannot take this occupant down.
            if let Some(session) = self.sessions.get_mut(peer.slot()) {
                session.arrival = Some(peer.arrival());
            }
        }
        self.peers.admitted(result, accepted)
    }

    /// The hosted device owner, mutably — the row's hardware door (interrupt
    /// acknowledgement, notification drains).
    pub fn host_mut(&mut self) -> &mut H {
        &mut self.host
    }

    /// Whether a backend is reachable right now.
    pub fn ready(&mut self) -> bool {
        self.host.backend().is_some()
    }

    /// This pair's authority, as configured at founding.
    pub fn policy_of(&self, via: u16) -> Option<&VifPolicy> {
        self.policy.get(pair_index(via)?)
    }

    /// Choose how vif addresses are assigned. See [`AddressMode`] — the
    /// default is the one that works everywhere.
    pub fn set_address_mode(&mut self, mode: AddressMode) {
        self.address_mode = mode;
    }

    /// Live vifs.
    pub fn open_vifs(&self) -> usize {
        self.vifs.len()
    }

    /// Vifs whose identity is dead and whose mapping is still draining.
    pub fn closing_vifs(&self) -> usize {
        self.closing.iter().flatten().count()
    }

    /// Completions that matched no vif at all. Zero on a healthy service.
    pub fn orphan_completions(&self) -> u64 {
        self.orphan_completions
    }

    /// Requests answered, and how many of those were refusals.
    pub fn tally(&self) -> (u64, u64) {
        (self.served, self.refused)
    }

    /// Wake notices the transport would not take.
    pub fn dropped_notices(&self) -> u64 {
        self.dropped_notices
    }

    /// Link edges the sweep derived, ever — a flapping cable's count, as
    /// this service saw it (the device's own count rides `Query`).
    pub fn link_edges(&self) -> u64 {
        self.link_edges
    }

    /// One vif's full counters — the service-side view.
    pub fn counters(&self, vif: VifId) -> Result<VifCounters, NetRefusal> {
        Ok(self.vifs.get(vif)?.counters)
    }

    /// One vif's granted filter — what it actually received, not what it
    /// asked for.
    pub fn filter_of(&self, vif: VifId) -> Result<FrameFilter, NetRefusal> {
        Ok(self.vifs.get(vif)?.filter)
    }

    /// The address one vif answers to.
    pub fn mac_of(&self, vif: VifId) -> Result<MacAddr, NetRefusal> {
        Ok(self.vifs.get(vif)?.mac)
    }

    // ── the facts ──────────────────────────────────────────────────────

    /// The device's facts, or `None` in the degraded posture.
    pub fn info(&mut self) -> Option<NetInfo> {
        let backend = self.host.backend()?;
        let link = backend.link();
        Some(NetInfo {
            link_up: link.is_up(),
            link_mbps: match link {
                LinkState::Up { mbps } => mbps,
                LinkState::Down => 0,
            },
            mac: backend.mac(),
            mtu: FRAME_MAX as u16,
            window_bytes: WINDOW_BYTES as u32,
            counters: backend.counters(),
            dropped_notices: u32::try_from(self.dropped_notices).unwrap_or(u32::MAX),
        })
    }

    // ── opening a vif ──────────────────────────────────────────────────

    /// **Open one vif** — the whole lifecycle, with the single untestable
    /// step supplied as `adopt`.
    ///
    /// Order is the point:
    ///
    /// 1. Policy first, so a refusal costs **nothing** — no address reserved,
    ///    no window mapped, no device leaf installed. A client that asks for
    ///    a power it does not have cannot make the service spend anything.
    /// 2. Reserve the device address window.
    /// 3. `adopt` — map the client's window and install it into the device's
    ///    address space. Production does the kernel work here; a host test
    ///    hands over a leaked buffer.
    /// 4. Attach the rings, which **validate**: the client formatted them, so
    ///    a malformed or hostile window is refused here rather than trusted.
    /// 5. Install, and re-derive the device packet filter.
    ///
    /// Every failure after step 2 unwinds what the earlier steps spent.
    pub fn open_vif(
        &mut self,
        via: u16,
        request: FrameFilter,
        adopt: impl FnOnce(&mut H, IovaAddr) -> Result<AdoptedWindow<H::Lease>, ShareFault>,
    ) -> Result<(VifId, MacAddr), NetRefusal> {
        let device_mac = self.host.backend().ok_or(NetRefusal::NotReady)?.mac();
        let filter = self.authorize(via, &request)?;

        // Before anything is spent: a device that cannot reach a vif arena
        // is refused here, rather than handed a proposal it would truncate.
        let base = self.arena_base().ok_or(NetRefusal::NoMapping)?;
        let (ordinal, proposed) = self.reserve_ordinal(base).ok_or(NetRefusal::Exhausted)?;
        let AdoptedWindow {
            bytes: window,
            iova,
            lease,
        } = match adopt(&mut self.host, proposed) {
            Ok(adopted) => adopted,
            Err(_) => {
                self.ordinals[ordinal as usize] = false;
                return Err(NetRefusal::NoMapping);
            }
        };

        let mac = match self.address_mode {
            AddressMode::Shared => device_mac,
            AddressMode::PerVif => device_mac.derive_local(ordinal),
        };
        match Self::assemble(via, filter, mac, ordinal, window, lease, iova) {
            Ok(vif) => match self.vifs.open(vif) {
                Some(id) => {
                    self.refresh_packet_filter();
                    Ok((id, mac))
                }
                None => {
                    // Unreachable while `MAX_VIFS` bounds both, but the
                    // window is already spent here — unwind rather than leak.
                    self.ordinals[ordinal as usize] = false;
                    Err(NetRefusal::Exhausted)
                }
            },
            Err((refusal, lease)) => {
                self.ordinals[ordinal as usize] = false;
                self.host.release_window(lease);
                Err(refusal)
            }
        }
    }

    /// Split an adopted window into its rings and frame area, attaching each
    /// ring the way the *other* side is not: the client produces transmit and
    /// consumes receive, so this service consumes transmit and produces
    /// receive.
    ///
    /// Attaching is where a hostile window dies: [`spsc::consumer`] and
    /// [`spsc::producer`] check the magic, the ABI, and the record size, and
    /// refuse a window that is not the ring pair the protocol says it is.
    #[allow(clippy::type_complexity)]
    fn assemble(
        via: u16,
        filter: FrameFilter,
        mac: MacAddr,
        ordinal: u8,
        window: &'static mut [u8],
        lease: H::Lease,
        iova: IovaAddr,
    ) -> Result<Vif<H::Lease>, (NetRefusal, H::Lease)> {
        if window.len() < WINDOW_BYTES {
            return Err((NetRefusal::BadWindow, lease));
        }
        let (rings, frames) = window.split_at_mut(TX_AREA_AT);
        let (tx_ring, rx_ring) = rings.split_at_mut(RX_RING_AT);
        let Ok(tx) = spsc::consumer::<FrameDesc>(tx_ring) else {
            return Err((NetRefusal::BadWindow, lease));
        };
        let Ok(rx) = spsc::producer::<FrameDesc>(rx_ring) else {
            return Err((NetRefusal::BadWindow, lease));
        };
        if tx.capacity() != TX_DEPTH || rx.capacity() != RX_DEPTH {
            return Err((NetRefusal::BadWindow, lease));
        }
        let frames = &mut frames[..FRAME_AREA_BYTES];
        Ok(Vif {
            via,
            filter,
            neighbor: None,
            mac,
            ordinal,
            frames,
            tx,
            rx,
            mapping: VifMapping::new(lease, iova),
            tx_popped: 0,
            rx_count: 0,
            counters: VifCounters::default(),
        })
    }

    /// Where this host's vif arena sits in its device's address space.
    ///
    /// Derived per call rather than stored: the two facts it rests on — the
    /// device's reach and its pool's span — are the host's, and a row binds
    /// its device after the engine is constructed. `None` is a device whose
    /// master cannot reach any arena this host knows how to place, which
    /// [`open_vif`](Self::open_vif) reports as `NoMapping` rather than
    /// proposing addresses the device would truncate.
    fn arena_base(&self) -> Option<u64> {
        kore_driver::host::vif_iova_base(self.host.dma_address_bits(), self.host.dma_pool_bytes())
    }

    /// Reserve one stride of the device address plan, and propose its address.
    ///
    /// The stride is what bounds concurrent vifs and keeps two of them from
    /// being *offered* the same device address; where each window actually
    /// lands is the host's answer ([`AdoptedWindow`]), because the driver owns
    /// its IOMMU domain and this engine does not.
    ///
    /// `base` comes from [`arena_base`](Self::arena_base), which has already
    /// proved the whole arena reachable and clear of the pool — so every
    /// stride below is too, by the const assertion above.
    fn reserve_ordinal(&mut self, base: u64) -> Option<(u8, IovaAddr)> {
        let ordinal = self.ordinals.iter().position(|spent| !spent)?;
        self.ordinals[ordinal] = true;
        Some((
            ordinal as u8,
            IovaAddr(base + (ordinal * WINDOW_BYTES) as u64),
        ))
    }

    /// Put the device in the receive mode this vif set needs.
    ///
    /// **Explicit, because CDC Ethernet makes it a device mode, not a vif
    /// one.** Per-vif addresses are only *receivable* if the device is
    /// admitting frames that are not addressed to its own station — so a
    /// service that assumed per-vif hardware filtering would silently receive
    /// nothing for its second vif. The classifier is authoritative either
    /// way; this only decides what reaches it.
    fn refresh_packet_filter(&mut self) {
        let wants_all = matches!(self.address_mode, AddressMode::PerVif)
            || self.vifs.iter().any(|(_, vif)| vif.filter.promiscuous);
        let filter = if wants_all {
            PacketFilter::PROMISCUOUS
        } else {
            PacketFilter::DIRECTED
        };
        if let Some(mut backend) = self.host.backend() {
            let _ = backend.set_packet_filter(filter);
        }
    }

    // ── closing a vif ──────────────────────────────────────────────────

    /// Close one vif.
    ///
    /// The identity dies **now** — [`VifTable::close`] bumps the generation,
    /// so the client's identifier stops resolving on this very request, and a
    /// stale producer cannot reach the slot's next tenant. The mapping is
    /// retired here if it can be, and parked to drain if it cannot: a vif is
    /// never closed out from under a transfer the controller is still reading.
    pub fn close_vif(&mut self, via: u16, id: VifId) -> Result<VifId, NetRefusal> {
        let vif = self.vifs.get(id)?;
        if vif.via != via {
            return Err(NetRefusal::NotOwner);
        }
        // Ownership is settled on the line above, so the ports this vif holds
        // may be freed. Without this a bind/close cycle retained an entry per
        // bind and the 64-slot table refused new binds after enough of them —
        // the vif was gone and its ports were not.
        self.inet.release_ports(id);
        let vif = self.vifs.close(id)?;
        let Vif {
            ordinal, mapping, ..
        } = vif;
        match mapping.retire() {
            Ok(lease) => {
                self.host.release_window(lease);
                self.ordinals[ordinal as usize] = false;
            }
            Err((mapping, _)) => {
                let parked = Closing {
                    id,
                    ordinal,
                    mapping,
                };
                match self.closing.iter_mut().find(|slot| slot.is_none()) {
                    Some(slot) => *slot = Some(parked),
                    None => {
                        // `MAX_VIFS` closing slots for `MAX_VIFS` vifs, so
                        // unreachable; the lease outlives the service rather
                        // than being retired under a live transfer.
                        core::mem::forget(parked);
                        self.orphan_completions = self.orphan_completions.saturating_add(1);
                    }
                }
            }
        }
        self.refresh_packet_filter();
        Ok(id)
    }

    // ── the sweep ──────────────────────────────────────────────────────

    /// **One bounded sweep of the data plane**: reap completions, pump each
    /// vif's transmit ring, classify what arrived, refill, and retire any
    /// drained mapping.
    ///
    /// Driven by the row's run loop after each serve drain, and by host tests
    /// directly — the same code, which is what makes the classifier and the
    /// transmit gates testable with no hardware.
    pub fn drive(&mut self) -> DriveReport {
        self.drive_at(
            kore_rt::clock::uptime()
                .ok()
                .map(kore_abi::clock::Uptime::as_nanos),
        )
    }

    /// The same production sweep with an injected monotonic clock sample.
    /// Absence remains explicit: zero nanoseconds is a valid timestamp.
    pub fn drive_at(&mut self, now_ns: Option<u64>) -> DriveReport {
        let mut report = DriveReport::default();
        let Some(mut backend) = self.host.backend() else {
            return report;
        };
        // `backend` borrows `self.host`; every field touched below is a
        // different one, which is why these are direct field accesses rather
        // than method calls on `self`.
        let vifs = &mut self.vifs;
        let closing = &mut self.closing;

        // 1. Completions first, so the slots they free are available to the
        //    pump in this same sweep.
        let mut done = [TxCompletion {
            cookie: TxCookie(0),
            status: TxStatus::Sent,
        }; TX_REAP];
        loop {
            let n = backend.reap_tx(&mut done);
            for completion in &done[..n] {
                report.tx_completed += 1;
                let (id, _slot) = unpack_cookie(completion.cookie);
                if let Ok(vif) = vifs.get_mut(id) {
                    vif.mapping.completed(1);
                    // Popping is what releases the slot to the client, and it
                    // happens **here** — not at submit — so a client cannot
                    // overwrite a frame the controller is still fetching.
                    if vif.tx.pop().is_some() {
                        vif.tx_popped = vif.tx_popped.wrapping_add(1);
                    }
                    match completion.status {
                        TxStatus::Sent => vif.counters.tx_frames += 1,
                        TxStatus::Failed(_) => vif.counters.tx_failed += 1,
                    }
                } else if let Some(parked) =
                    closing.iter_mut().flatten().find(|parked| parked.id == id)
                {
                    // The vif is gone but its mapping is not: the generation
                    // check that refused the identity is the same one that
                    // finds the right drain.
                    parked.mapping.completed(1);
                } else {
                    self.orphan_completions = self.orphan_completions.saturating_add(1);
                }
            }
            if n < TX_REAP {
                break;
            }
        }

        // Advance the link after its completions landed and before a client
        // can submit under the previous association's state.
        backend.service(now_ns);

        // The link, as a level: an edge is the level having moved since
        //    the last sweep, whatever the device raised or coalesced on the
        //    way. The wake that brought a link-change cause here is the
        //    same route wake every other cause uses.
        let link = backend.link();
        if link != self.link {
            self.link = link;
            self.link_edges = self.link_edges.saturating_add(1);
            report.link_changed = true;
        }

        // 2. Pump transmit, per vif.
        for (id, vif) in vifs.iter_mut() {
            report.tx_submitted += pump_tx(id, vif, &mut backend, &mut report.tx_refused);
        }
        //    …and publish the whole sweep's descriptors with ONE doorbell.
        //    Unconditional: a backend with nothing staged answers with a
        //    no-op, and the engine never has to know which kind it holds.
        backend.flush_tx();

        // 3. Receive: one frame off the wire, fanned out to every vif whose
        //    filter accepts it. The copy is the isolation boundary — USB has
        //    no steering, so posting one client's buffer to the shared bulk
        //    pipe would let it observe every other vif's frames.
        let mut wake = 0u16;
        let mut delivered = 0u32;
        let mut dropped = 0u32;
        while report.rx_polled < RX_BATCH {
            let took = backend.poll_rx(&mut |frame| {
                classify(vifs, frame, &mut wake, &mut delivered, &mut dropped);
            });
            if !took {
                break;
            }
            report.rx_polled += 1;
        }
        report.rx_delivered = delivered;
        report.rx_dropped = dropped;
        report.wake = wake;
        if report.rx_polled > 0 {
            let _ = backend.refill_rx();
        }

        report.service_deadline = backend.service_deadline();

        // 4. Retire whatever finished draining.
        drop(backend);
        self.reap_closing();
        report
    }

    /// Retire the mappings of closed vifs whose transfers have all landed.
    fn reap_closing(&mut self) {
        for slot in self.closing.iter_mut() {
            let Some(parked) = slot.as_ref() else {
                continue;
            };
            if parked.mapping.outstanding() != 0 {
                continue;
            }
            let Some(parked) = slot.take() else { continue };
            let ordinal = parked.ordinal;
            match parked.mapping.retire() {
                Ok(lease) => {
                    self.host.release_window(lease);
                    self.ordinals[ordinal as usize] = false;
                }
                Err((mapping, _)) => {
                    *slot = Some(Closing {
                        id: parked.id,
                        ordinal,
                        mapping,
                    });
                }
            }
        }
    }

    // ── notices ────────────────────────────────────────────────────────

    /// One wake notice per pair named by a sweep's [`DriveReport::wake`].
    ///
    /// The poll is the truth and the notice only wakes: a refused notice is
    /// counted and dropped, and the client's own ring still holds every frame.
    pub async fn pump_notices<T: Transport>(&mut self, transport: &T, wake: u16) -> u32 {
        let mut sent = 0;
        for via in 0..PAIRS {
            if wake & (1 << via) == 0 {
                continue;
            }
            let Some(lane) = self.sessions[via].signal.as_ref() else {
                continue;
            };
            let tx = ChanTx::new(transport, lane.send.slot());
            match tx
                .send(Notice {
                    cookie: COOKIE_NET_RX,
                    payload: via as u64,
                })
                .await
            {
                Ok(()) => sent += 1,
                Err(_) => self.dropped_notices = self.dropped_notices.saturating_add(1),
            }
        }
        sent
    }

    /// One link notice per pair holding a doorbell, after a sweep whose
    /// [`DriveReport::link_changed`] said the level moved.
    ///
    /// The notice says only "the link moved — ask again": the level rides
    /// `Query`, and a client that missed the notice reads the same answer
    /// there. Refused notices are counted with the receive ones.
    pub async fn pump_link_notices<T: Transport>(&mut self, transport: &T) -> u32 {
        let mut sent = 0;
        for (via, session) in self.sessions.iter().enumerate() {
            let Some(lane) = session.signal.as_ref() else {
                continue;
            };
            let tx = ChanTx::new(transport, lane.send.slot());
            match tx
                .send(Notice {
                    cookie: COOKIE_NET_LINK,
                    payload: via as u64,
                })
                .await
            {
                Ok(()) => sent += 1,
                Err(_) => self.dropped_notices = self.dropped_notices.saturating_add(1),
            }
        }
        sent
    }

    // ── the serve body ─────────────────────────────────────────────────

    /// Answer one request. Authority leaves only through `sv`.
    pub fn serve_step(&mut self, req: NetRequest, sv: &mut Serve<'_, NetResponse>) {
        self.served = self.served.saturating_add(1);
        // The table answers; this step only accounts. A verb no row claims
        // cannot arrive — the decoder that produced `req` consulted the same
        // table — but it is refused rather than guessed, which is the same
        // fail-closed posture one layer up.
        let response = NetVerbs::serve(self, sv, &req)
            .unwrap_or(NetResponse::Refused(NetRefusal::Unsupported));
        if matches!(response, NetResponse::Refused(_)) {
            self.refused = self.refused.saturating_add(1);
        }
        sv.reply(response);
    }

    /// The same step, where answering may mean awaiting.
    ///
    /// **This is the one the founded row takes**, and it is not an
    /// optimisation. Three verbs here mint a flow, which installs a filter,
    /// which is `async` at the seam — so they implement `ServeVerb` through
    /// `serve_async` and inherit the synchronous default, which answers
    /// `None`. Dispatched through [`serve_step`](Self::serve_step) they
    /// would every one of them be refused `Unsupported`: present in the
    /// table, decodable on the wire, and dead.
    ///
    /// The sync verbs are unaffected and are not duplicated here, because
    /// `ServeVerb::serve_async` defaults to forwarding to `serve`. One
    /// dispatch reaches all twelve.
    pub async fn serve_step_async(&mut self, req: NetRequest, sv: &mut Serve<'_, NetResponse>) {
        self.served = self.served.saturating_add(1);
        let response = NetVerbs::serve_async(self, sv, &req)
            .await
            .unwrap_or(NetResponse::Refused(NetRefusal::Unsupported));
        if matches!(response, NetResponse::Refused(_)) {
            self.refused = self.refused.saturating_add(1);
        }
        sv.reply(response);
    }

    /// The bound radio's facts, as the backend read them at bring-up.
    pub(crate) fn radio(&self) -> Option<RadioReport> {
        self.host.radio()
    }

    /// The `OpenVif` arm: take the transferred window off the return leg and
    /// run it through [`open_vif`](Self::open_vif).
    ///
    /// The order is deliberate. Policy is checked **before** the delivery is
    /// taken, because leaving a delivery untaken returns it to the serve
    /// host's retryable cleanup — so a client that asks for a power it does
    /// not have neither spends its own window nor makes this service own a
    /// record it has no use for. Taking it means this service now owns the
    /// record's lifetime, which is exactly what a vif that outlives the
    /// request needs.
    pub(crate) fn open_arm(
        &mut self,
        via: u16,
        filter: FrameFilter,
        announced: SlotRef,
        sv: &mut Serve<'_, NetResponse>,
    ) -> NetResponse {
        if let Err(refusal) = self.authorize(via, &filter) {
            return NetResponse::Refused(refusal);
        }
        let Some(returned) = sv.returned_input(announced).and_then(|mut i| i.take()) else {
            return NetResponse::Refused(NetRefusal::BadWindow);
        };
        // Held in an `Option` so the closure can move it out exactly once. If
        // the closure never runs, the capability drops here — retiring the
        // record, which is the right cleanup for a transfer nobody adopted.
        let mut returned = Some(returned);
        let outcome = self.open_vif(via, filter, |host, iova| {
            host.adopt_window(returned.take().ok_or(ShareFault::NoTransfer)?, iova)
        });
        match outcome {
            Ok((vif, mac)) => NetResponse::Opened { vif, mac },
            Err(refusal) => NetResponse::Refused(refusal),
        }
    }

    /// Replace the receive filter on a vif `via` already holds.
    ///
    /// The three checks are the same three `open_vif` makes, in the same
    /// order, and none of them is skipped because the vif already exists:
    ///
    /// 1. **The identity resolves.** A stale generation answers
    ///    [`StaleVif`](NetRefusal::StaleVif), so a holder whose vif was closed
    ///    cannot re-filter the slot's next tenant.
    /// 2. **The pair owns it.** Re-checked here as on every op that names a
    ///    vif, so a second face cannot re-filter by guessing an identity.
    /// 3. **The policy permits the new filter.** In full, by
    ///    [`VifPolicy::authorize_filter`] — holding a vif is not authority to
    ///    widen what it accepts, and the refusal names the dimension that
    ///    exceeded rather than saying "denied".
    ///
    /// Nothing is spent before all three pass, so a refusal leaves the vif
    /// exactly as it was: the old filter still installed, still enforced. The
    /// device packet filter is re-derived afterwards because `promiscuous` is
    /// a device mode rather than a vif one — dropping it from the last vif
    /// that held it must take the device back out of promiscuous mode.
    pub fn set_filter(
        &mut self,
        via: u16,
        id: VifId,
        request: FrameFilter,
    ) -> Result<VifId, NetRefusal> {
        let out = install_filter(&mut self.vifs, &self.policy, via, id, request);
        // Only once, and only after: a promiscuous filter changes what the
        // device itself must accept, and a refused install changed nothing.
        if out.is_ok() {
            self.refresh_packet_filter();
        }
        out
    }

    /// Whether `via` may hold the vif it is asking for — the check
    /// [`open_vif`](Self::open_vif) makes for itself, exposed so the serve arm
    /// can make it before spending anything.
    fn authorize(&self, via: u16, request: &FrameFilter) -> Result<FrameFilter, NetRefusal> {
        let policy = pair_index(via)
            .and_then(|cell| self.policy.get(cell))
            .ok_or(NetRefusal::NotOwner)?;
        // The *count* is keyed on the full `via` and not the cell: a vif
        // belongs to one occupant of a cell, and a previous occupant's vifs
        // must neither be inherited by its successor nor charged against it.
        let held = self.vifs.iter().filter(|(_, vif)| vif.via == via).count();
        Ok(policy.authorize(request, held)?)
    }

    /// The `Stats` arm — ownership re-checked, like every op that names a vif.
    pub(crate) fn stats_arm(&self, via: u16, id: VifId) -> Result<NetStats, NetRefusal> {
        let vif = self.vifs.get(id)?;
        if vif.via != via {
            return Err(NetRefusal::NotOwner);
        }
        Ok(vif.counters.wire())
    }

    /// The `Signal` arm: revoke any standing lane, mint afresh, retain root
    /// and send, and stage the fresh recv derive in this reply's grant
    /// session. Replace semantics, so exactly one live recv per pair exists.
    pub(crate) fn signal_arm(&mut self, via: u16, sv: &mut Serve<'_, NetResponse>) -> NetResponse {
        let Some(session) = pair_index(via).and_then(|cell| self.sessions.get_mut(cell)) else {
            return NetResponse::Refused(NetRefusal::NotOwner);
        };
        let Some(channels) = self.channels.as_ref() else {
            return NetResponse::Refused(NetRefusal::NoChannel);
        };
        if let Some(lane) = session.signal.take() {
            if let Err(lane) = lane.retire() {
                session.signal = Some(lane);
                return NetResponse::Refused(NetRefusal::Busy);
            }
        }
        match verbs::mint_channel(channels) {
            Ok(pair) => {
                if pair.recv.try_retire().is_err() {
                    let _ = SignalLane {
                        root: pair.root,
                        send: pair.send,
                    }
                    .retire();
                    return NetResponse::Refused(NetRefusal::Internal);
                }
                let root_slot = pair.root.slot();
                session.signal = Some(SignalLane {
                    root: pair.root,
                    send: pair.send,
                });
                self.last_signal_via = Some(via);
                sv.grant(root_slot, CHANNEL_ENDPOINT_RIGHTS, GrantArg::ChanRecv);
                NetResponse::Signaled
            }
            Err(_) => NetResponse::Refused(NetRefusal::NoChannel),
        }
    }

    /// The `grants_dropped` repair: the staged recv never reached the client,
    /// so the promise is unwound — the fresh lane is revoked and the reply
    /// rewritten to a refusal.
    pub fn grants_unwound(&mut self, response: &mut NetResponse, emitted: usize) {
        if emitted == 0 {
            return;
        }
        if let Some(via) = self.last_signal_via.take()
            && let Some(session) = pair_index(via).and_then(|cell| self.sessions.get_mut(cell))
            && let Some(lane) = session.signal.take()
            && let Err(lane) = lane.retire()
        {
            session.signal = Some(lane);
        }
        *response = NetResponse::Refused(NetRefusal::NoChannel);
    }
}

// ── the transmit gates ─────────────────────────────────────────────────

/// Pump one vif's transmit ring, returning how many frames were submitted.
///
/// The ring is read **past the head** by the number already in flight, and
/// popped only when a completion lands. A descriptor that fails its gates is
/// not serviced and not clamped: it stalls its own vif until the pipeline
/// drains, at which point it reaches the head and is discarded. A client that
/// sends only bad frames therefore drains only its own ring.
fn pump_tx<L, B: NicBackend>(
    id: VifId,
    vif: &mut Vif<L>,
    backend: &mut B,
    refused: &mut u32,
) -> u32 {
    let mut submitted = 0;
    while vif.mapping.outstanding() < MAX_TX_INFLIGHT {
        let ahead = vif.mapping.outstanding();
        let Some(desc) = vif.tx.peek_at(ahead) else {
            break;
        };
        let slot = position_slot(vif, ahead);
        match validate_tx(vif, slot, &desc) {
            Ok(()) => match vif
                .mapping
                .transmit(backend, slot, desc.len, pack_cookie(id, slot))
            {
                Ok(()) => submitted += 1,
                // The backend would not take it — a full endpoint ring, or a
                // link that is down. Nothing was consumed, so nothing needs
                // unwinding; the next sweep tries the same descriptor again.
                Err(_) => break,
            },
            Err(()) => {
                // A refused descriptor cannot be skipped: releasing a slot is
                // moving the head, and the head cannot pass records the
                // controller is still reading. So it waits until it *is* the
                // head, and is then discarded — a client that sends only bad
                // frames drains only its own ring.
                if ahead == 0 {
                    if vif.tx.pop().is_some() {
                        vif.tx_popped = vif.tx_popped.wrapping_add(1);
                    }
                    vif.counters.tx_refused += 1;
                    *refused += 1;
                    continue;
                }
                break;
            }
        }
    }
    submitted
}

/// The transmit slot a descriptor `ahead` positions past the head occupies.
///
/// Slot ownership is **derived, never stored**: the *n*-th record the client
/// pushes uses slot `n mod TX_DEPTH`, so the service recomputes it from its
/// own cursor rather than reading it out of a field the client controls.
fn position_slot<L>(vif: &Vif<L>, ahead: u32) -> u32 {
    kore_driver::net::tx_slot_of(vif.tx_popped.wrapping_add(ahead as u64))
}

/// The per-frame transmit gates, in order — each one a refusal, never a clamp.
fn validate_tx<L>(vif: &Vif<L>, slot: u32, desc: &FrameDesc) -> Result<(), ()> {
    // 1. A length that could be a frame at all.
    if !desc.len_is_sane() {
        return Err(());
    }
    let Some(at) = tx_frame_at(slot) else {
        return Err(());
    };
    let len = desc.len as usize;
    let Some(bytes) = vif.frames.get(at..at + len) else {
        return Err(());
    };
    // 2. The declared EtherType must be one this vif may speak, and it must
    //    be the one actually on the wire — a descriptor that declares one
    //    protocol and carries another is a refusal, not a relabelling.
    let Some(head) = FrameHead::parse(bytes) else {
        return Err(());
    };
    if head.ethertype.0 != desc.ethertype {
        return Err(());
    }
    if head.ethertype == EtherType::ARP
        && let Some(neighbor) = &vif.neighbor
    {
        return neighbor
            .accepts_request(&head, bytes, vif.mac)
            .then_some(())
            .ok_or(());
    }
    if !vif.filter.ethertype.accepts(head.ethertype) {
        return Err(());
    }
    // 3. The source address must be one this vif may transmit from.
    if head.src != vif.mac {
        return Err(());
    }
    // 4. And the frame must be inside the flow the vif was granted.
    //
    // The gate that was missing. Until this line a flow capability bounded
    // what a holder could *hear* and not what it could *say*: `accepts_flow`
    // had exactly one caller, `accepts_frame`, reached only from `classify`,
    // which is the receive path. A holder of one connection could emit any
    // source address and port it liked -- no race, no window, simply
    // unenforced -- which left the isolation a minted flow promises as half
    // a property, and made the two-task witness it exists for unearnable.
    //
    // Read in the sending direction, against the same filter the classifier
    // reads arriving. It has to be the other reading and not the same one:
    // a grant names one conversation, and our address is its destination on
    // the way in and its source on the way out. Asking the inbound question
    // here would refuse every lawful frame and admit only an impossible one.
    // A vif with no flow bound is unaffected -- the L2-only contract answers
    // vacuously true, exactly as it did.
    let payload = &bytes[kore_driver::net::frame::HEADER_LEN..];
    if !vif.filter.accepts_flow_outbound(head.ethertype, payload) {
        return Err(());
    }
    Ok(())
}

impl<H: NetHost> NetService<H> {
    /// The station's identity, as the row currently holds it.
    pub fn identity(&self) -> Option<&Identity> {
        self.inet.identity()
    }

    /// Give the machine its address.
    pub fn configure(&mut self, identity: Identity) {
        if self.inet.identity() != Some(&identity) {
            for (_, vif) in self.vifs.iter_mut() {
                vif.neighbor = None;
            }
        }
        self.inet.set_identity(identity);
    }

    /// The station's identity as a caller may see it.
    ///
    /// A report rather than the [`Identity`], because that type has no
    /// constructor a holder could use to widen anything and handing it out
    /// would invite the belief that holding it is authority to change it.
    pub fn identity_report(&self) -> Option<IdentityReport> {
        self.inet.identity().map(|id| IdentityReport {
            addr: id.addr(),
            prefix_len: id.prefix_len(),
            gateway: id.gateway(),
            dns: id.dns(),
        })
    }

    /// What this serving pair may ask a mint for.
    ///
    /// Taken from the pair's wiring and never read off the request: a
    /// caller that could state its own grant would be stating its own
    /// authority.
    ///
    /// Every dimension is withheld, which is what the row before this one
    /// did too — it built `OpenGrant::default()` and nothing ever widened
    /// it. Named here rather than left to be discovered: `VifPolicy`
    /// carries no privileged-port dimension, nothing in tree grants
    /// `raw_l2`, and `Powers` has no grant path at all. Each is a policy
    /// slice of its own, and until one lands a pair binds unprivileged
    /// ports and holds no orthogonal power.
    pub fn flow_grant(&self, _via: u16) -> OpenGrant {
        OpenGrant {
            privileged_ports: false,
            raw_l2: false,
            powers: Powers::NONE,
        }
    }

    /// The `Configure` arm: validate the address and take it.
    pub fn configure_arm(
        &mut self,
        addr: Ipv4Addr,
        prefix_len: u8,
        gateway: Option<Ipv4Addr>,
    ) -> Result<(), NetRefusal> {
        let identity = Identity::new(addr, prefix_len, gateway)?;
        self.configure(identity);
        Ok(())
    }

    /// Mint a flow to one peer and install the filter that enforces it.
    ///
    /// The split borrow is the whole trick, and the reason the two rows
    /// could become one: `inet` drives the mint while `vifs` and `policy`
    /// are lent to the installer beside it. Nothing crosses a serving pair,
    /// so there is no face to found and no state where a capability exists
    /// that nothing enforces.
    pub async fn open_flow(
        &mut self,
        via: u16,
        vif: VifId,
        grant: OpenGrant,
        proto: IpProto,
        local_port: Option<u16>,
        peer: (Ipv4Addr, u16),
    ) -> Result<FlowCap, OpenRefusal> {
        let neighbor = self.inet.identity().and_then(|identity| {
            Some(Neighbor {
                local: identity.addr().0,
                next_hop: identity.next_hop(peer.0)?.0,
            })
        });
        let Self {
            inet, vifs, policy, ..
        } = self;
        let mut install = VifInstall {
            vifs,
            policy,
            via,
            touched: false,
            neighbor,
        };
        let out = inet
            .open_flow(vif, grant, proto, local_port, peer, &mut install)
            .await;
        if install.touched {
            self.refresh_packet_filter();
        }
        out
    }

    /// Mint a listening flow: a local endpoint, any peer.
    pub async fn open_listener(
        &mut self,
        via: u16,
        vif: VifId,
        grant: OpenGrant,
        proto: IpProto,
        local_port: u16,
    ) -> Result<FlowCap, OpenRefusal> {
        let Self {
            inet, vifs, policy, ..
        } = self;
        let mut install = VifInstall {
            vifs,
            policy,
            via,
            touched: false,
            neighbor: None,
        };
        let out = inet
            .open_listener(vif, grant, proto, local_port, &mut install)
            .await;
        if install.touched {
            self.refresh_packet_filter();
        }
        out
    }

    /// Release a vif's flows and the ports they hold.
    ///
    /// Reports the refusal rather than swallowing it. Two outcomes a caller
    /// must be able to tell apart hide behind a bare count: a close that
    /// freed nothing because the vif held nothing, and a close that freed
    /// nothing because the vif was never the caller's.
    pub async fn close_flows(
        &mut self,
        via: u16,
        vif: VifId,
    ) -> (usize, Result<(), InstallRefusal>) {
        let Self {
            inet, vifs, policy, ..
        } = self;
        let mut install = VifInstall {
            vifs,
            policy,
            via,
            touched: false,
            neighbor: None,
        };
        let closed = inet.close(vif, &mut install).await;
        if install.touched {
            self.refresh_packet_filter();
        }
        closed
    }
}

/// What the station's identity looks like to a caller.
///
/// A report and not the [`Identity`] itself: that type has no constructor
/// a holder of one could use to widen anything, and handing it out would
/// invite the belief that holding it is authority to change it. Changing
/// the machine's address means asking the row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IdentityReport {
    /// The station address.
    pub addr: Ipv4Addr,
    /// Its prefix length.
    pub prefix_len: u8,
    /// The default gateway, if one was configured.
    pub gateway: Option<Ipv4Addr>,
    /// The resolver, if one was learned.
    pub dns: Option<Ipv4Addr>,
}

/// Replace `id`'s receive filter, re-authorized in full against `via`'s
/// policy.
///
/// A free function over the two fields it needs rather than a method,
/// because the arbitrating row and the vif table are both parts of this
/// service and the row cannot be driven while `&mut self` is held whole.
/// Splitting the borrow here is what lets one row own both halves — which
/// is the whole reason the flow program moved into this one.
///
/// Holding a vif is not authority to widen what it accepts, so this asks
/// the same `authorize_filter` an open asks. It does not touch the device
/// packet filter; the caller does that once, after, and only on success.
/// **The policy and session cell a `via` names.**
///
/// A FIXED pair's `via` *is* its index. A MINTED pair's is an **arrival** —
/// `generation * SLOT_CAPACITY + slot` — so its cell is the slot, and the raw
/// value would be far past either array. Reading the arrival as an index is
/// the bug this exists to prevent: it would miss every fixed-pair collision
/// (every live arrival is numerically past the fixed bank) and, where the
/// array is longer, hand a minted face another pair's authority.
///
/// `None` for a cell outside the bank — refused, never wrapped.
fn pair_index(via: u16) -> Option<usize> {
    let cell = PeerId::from_arrival(via).map_or(usize::from(via), |peer| peer.slot());
    (cell < PAIRS).then_some(cell)
}

fn install_filter<L>(
    vifs: &mut VifTable<Vif<L>>,
    policy: &[VifPolicy; PAIRS],
    via: u16,
    id: VifId,
    request: FrameFilter,
) -> Result<VifId, NetRefusal> {
    let vif = vifs.get(id)?;
    if vif.via != via {
        return Err(NetRefusal::NotOwner);
    }
    let granted = pair_index(via)
        .and_then(|cell| policy.get(cell))
        .ok_or(NetRefusal::NotOwner)?
        .authorize_filter(&request)?;
    // Resolved mutably only once the answer is known, so no borrow is held
    // across the policy consultation and no partial state exists on any
    // refusal path above.
    let vif = vifs.get_mut(id)?;
    vif.filter = granted;
    vif.neighbor = None;
    Ok(id)
}

/// The flow row's install seam, answered in process.
///
/// `res/INET_MIGRATION.md` §5.7b recorded a row that minted capabilities
/// nothing enforced, because installing the filter meant calling another
/// service across a founder face that cannot be built. There is no call and
/// no face: the mint and the enforcement are the same row, so this is a
/// function and `InstallRefusal::NoFrameFace` is unreachable.
struct VifInstall<'a, L> {
    vifs: &'a mut VifTable<Vif<L>>,
    policy: &'a [VifPolicy; PAIRS],
    via: u16,
    /// Whether anything was installed, so the caller refreshes the device
    /// filter once rather than per mint.
    touched: bool,
    neighbor: Option<Neighbor>,
}

impl<L> FlowInstall for VifInstall<'_, L> {
    /// The ownership half of [`install_filter`], asked on its own because a
    /// close needs the answer before it mutates anything. Deliberately the
    /// same two refusals that function gives, so a caller cannot learn more
    /// from a close than from an install.
    fn held(&self, vif: VifId) -> Result<(), InstallRefusal> {
        match self.vifs.get(vif) {
            Ok(held) if held.via == self.via => Ok(()),
            Ok(_) => Err(InstallRefusal::NotOwner),
            Err(_) => Err(InstallRefusal::NoVif),
        }
    }

    async fn install(&mut self, vif: VifId, filter: FrameFilter) -> Result<(), InstallRefusal> {
        match install_filter(self.vifs, self.policy, self.via, vif, filter) {
            Ok(_) => {
                if let Some(cell) = pair_index(self.via)
                    && self.policy[cell].ethertype.accepts(EtherType::ARP)
                {
                    self.vifs
                        .get_mut(vif)
                        .map_err(|_| InstallRefusal::NoVif)?
                        .neighbor = self.neighbor.take();
                }
                self.touched = true;
                Ok(())
            }
            Err(NetRefusal::NoVif) => Err(InstallRefusal::NoVif),
            Err(NetRefusal::StaleVif) => Err(InstallRefusal::NoVif),
            Err(NetRefusal::NotOwner) => Err(InstallRefusal::NotOwner),
            // Everything else `authorize_filter` can answer is a policy
            // dimension. The mapping is written out rather than defaulted,
            // so a new refusal is a non-exhaustive-match error here instead
            // of silently becoming `EtherType`.
            Err(NetRefusal::PolicyEtherType) => {
                Err(InstallRefusal::Policy(PolicyRefusal::EtherType))
            }
            Err(NetRefusal::PolicyMulticast) => {
                Err(InstallRefusal::Policy(PolicyRefusal::Multicast))
            }
            Err(NetRefusal::PolicyBroadcast) => {
                Err(InstallRefusal::Policy(PolicyRefusal::Broadcast))
            }
            Err(NetRefusal::PolicyPromiscuous) => {
                Err(InstallRefusal::Policy(PolicyRefusal::Promiscuous))
            }
            Err(NetRefusal::PolicyVifs) => Err(InstallRefusal::Policy(PolicyRefusal::TooManyVifs)),
            Err(NetRefusal::PolicyFlowRequired) => {
                Err(InstallRefusal::Policy(PolicyRefusal::FlowRequired))
            }
            Err(NetRefusal::PolicyFlowProto) => {
                Err(InstallRefusal::Policy(PolicyRefusal::FlowProto))
            }
            Err(NetRefusal::PolicyFlowLocal) => {
                Err(InstallRefusal::Policy(PolicyRefusal::FlowLocal))
            }
            Err(NetRefusal::PolicyFlowPeer) => Err(InstallRefusal::Policy(PolicyRefusal::FlowPeer)),
            // A vif that vanished between the mint and the install: the one
            // answer that is genuinely unknown rather than refused.
            Err(_) => Err(InstallRefusal::Lost),
        }
    }
}

// ── the classifier ─────────────────────────────────────────────────────

/// Fan one received frame out to every vif whose filter accepts it.
///
/// Bounded twice over: at most [`MAX_RX_FANOUT`] vifs see one frame, and each
/// vif's ring absorbs at most its own depth. A slow vif drops **its own**
/// frames ([`spsc::Producer::push_lossy`]) and never stalls another — the
/// KINP posture, which is what keeps a peer vif from wedging infra.
fn classify<L>(
    vifs: &mut VifTable<Vif<L>>,
    frame: &[u8],
    wake: &mut u16,
    delivered: &mut u32,
    dropped: &mut u32,
) {
    if frame.len() < FRAME_MIN || frame.len() > FRAME_MAX {
        return;
    }
    let Some(head) = FrameHead::parse(frame) else {
        return;
    };
    let mut desc = FrameDesc::new(frame.len() as u16, head.ethertype);
    if head.dst.is_broadcast() {
        desc.flags |= kore_driver::net::frame::flags::BROADCAST;
    } else if head.dst.is_multicast() {
        desc.flags |= kore_driver::net::frame::flags::MULTICAST;
    }

    let mut fanout = 0usize;
    // One frame, one parse. The five-tuple is a property of the frame and
    // not of any vif, but `accepts_frame` parses inside the per-vif call, so
    // a table where several vifs carry flows paid for the same walk once per
    // vif — up to `MAX_VIFS` times for a single arriving frame.
    //
    // Deferred rather than hoisted outright, because the shipping shape
    // binds no flow at all and `accepts_flow` has always returned early in
    // that case. Parsing before the loop would charge every frame on every
    // machine for a question nobody is asking. This way the count is zero
    // when no vif carries a flow, and one when any does.
    let payload = &frame[kore_driver::net::frame::HEADER_LEN..];
    let mut parsed: Option<Option<kore_driver::net::flow::FiveTuple>> = None;
    for (_, vif) in vifs.iter_mut() {
        // The *complete* decision: the L2 half and the flow half. Taking the
        // L2 half alone would deliver another connection's datagrams to a
        // vif that asked for one flow.
        if head.ethertype == EtherType::ARP
            && let Some(neighbor) = &vif.neighbor
        {
            if !neighbor.accepts_reply(&head, frame, vif.mac) {
                continue;
            }
        } else {
            if !vif.filter.accepts(&head, vif.mac) {
                continue;
            }
            if vif.filter.flow.is_some() {
                let tuple = parsed.get_or_insert_with(|| {
                    kore_driver::net::flow::FiveTuple::parse(head.ethertype, payload)
                });
                if !vif.filter.accepts_flow_parsed(tuple.as_ref()) {
                    continue;
                }
            }
        }
        if fanout >= MAX_RX_FANOUT {
            vif.counters.rx_dropped += 1;
            *dropped += 1;
            continue;
        }
        fanout += 1;
        // Room must be established **before** the copy: the slot the frame
        // would go in is the one the client is reading when the ring is full,
        // so a copy-then-push would corrupt a frame in use.
        //
        // `capacity_left` is conservative in exactly the direction that makes
        // this sound — the consumer only ever frees more — so a non-zero
        // reading means the push below cannot fail. The ring's own `dropped`
        // counter therefore stays untouched here (it belongs to a producer
        // that can safely *attempt* the push); this producer's drops are
        // counted below and reported by `Stats`.
        if vif.rx.capacity_left() == 0 {
            vif.counters.rx_dropped += 1;
            *dropped += 1;
            continue;
        }
        let slot = rx_slot_of(vif.rx_count);
        let Some(at) = rx_frame_at(slot) else {
            continue;
        };
        let Some(dst) = vif.frames.get_mut(at..at + frame.len()) else {
            continue;
        };
        dst.copy_from_slice(frame);
        if vif.rx.push(desc).is_ok() {
            vif.rx_count = vif.rx_count.wrapping_add(1);
            vif.counters.rx_frames += 1;
            *delivered += 1;
            // The wake bitmask is indexed by **cell**, the same thing
            // `pump_notices` iterates and `sessions` is indexed by — not by
            // the raw `via`. A minted face's `via` is an arrival (past 255),
            // so indexing it raw set no bit at all and that client was never
            // woken: the ring kept every frame, and a client parked on its
            // doorbell waited forever.
            if let Some(cell) = pair_index(vif.via)
                && cell < 16
            {
                *wake |= 1 << cell;
            }
        } else {
            vif.counters.rx_dropped += 1;
            *dropped += 1;
        }
    }
}

const _: () = assert!(RX_AREA_AT > TX_AREA_AT, "the layout the classifier assumes");
const _: () = assert!(SLOT_BYTES >= FRAME_MAX, "one slot holds one whole frame");

#[cfg(test)]
mod tests {
    use core::slice;
    use std::vec::Vec;

    use kore_abi::task::{Bootstrap, Profile};
    use kore_driver::net::filter::{EtherTypeMatch, MulticastMatch};
    use kore_driver::net::frame::HEADER_LEN;
    use kore_driver::net::window::{RX_RING_AT, tx_slot_at};
    use kore_driver::net::{EtherType, Steering, TxFrame};
    use kore_rt::service::{BindError, Bundle, SystemService, step, step_async};
    use kore_rt::svc::SvcProtocol;

    use super::*;
    use crate::client::format_window;
    use crate::wire::{NET_REPLY_GRANTS, decode_rep, encode_rep};

    /// QEMU's `usb-net` address, so the numbers here match the ones a gate
    /// boot prints.
    const DEVICE_MAC: MacAddr = MacAddr([0x40, 0x01, 0x02, 0x03, 0x04, 0x05]);
    const PEER_MAC: MacAddr = MacAddr([0x52, 0x55, 0x0a, 0x00, 0x02, 0x02]);

    // ── the fake device ────────────────────────────────────────────────

    /// A [`NicBackend`] with no hardware: it records what was posted,
    /// completes on demand, and hands over whatever frames a test injects.
    struct FakeNic {
        mac: MacAddr,
        link: LinkState,
        /// Every packet-filter mode the service has asked for, in order.
        filters: Vec<PacketFilter>,
        /// What the service handed to hardware and has not completed.
        posted: Vec<(IovaAddr, u16, TxCookie)>,
        /// Completions waiting to be reaped.
        done: Vec<TxCompletion>,
        /// Frames waiting to arrive, oldest first.
        inbound: Vec<Vec<u8>>,
        refills: u32,
        /// Transmit doorbells rung — once per drive, whatever was staged.
        flushes: u32,
        service_calls: u32,
        service_time: Option<u64>,
        service_link: Option<(u64, LinkState)>,
        reaps: u32,
        /// Refuse every transmit — a full endpoint ring, or a dead link.
        busy: bool,
        /// What the fake device reports having been through.
        counters: NicCounters,
        /// Where each vif window lives in this process, keyed by the device
        /// address it was mapped at. Standing in for the IOMMU, so a test can
        /// read the bytes the device *would* have fetched and prove the
        /// transmit path is genuinely zero-copy rather than merely plausible.
        windows: Vec<(u64, *mut u8)>,
    }

    impl FakeNic {
        fn new() -> Self {
            Self {
                mac: DEVICE_MAC,
                link: LinkState::Up { mbps: 1000 },
                filters: Vec::new(),
                posted: Vec::new(),
                done: Vec::new(),
                inbound: Vec::new(),
                refills: 0,
                flushes: 0,
                service_calls: 0,
                service_time: None,
                service_link: None,
                reaps: 0,
                busy: false,
                counters: NicCounters::default(),
                windows: Vec::new(),
            }
        }

        /// The bytes the device would fetch for one posted transfer.
        fn fetch(&self, iova: IovaAddr, len: u16) -> Vec<u8> {
            let (base, at) = self
                .windows
                .iter()
                .find_map(|(base, ptr)| {
                    (iova.0 >= *base && iova.0 < *base + WINDOW_BYTES as u64)
                        .then_some((*ptr, (iova.0 - *base) as usize))
                })
                .expect("the transmit address is inside a mapped vif window");
            // SAFETY: `at + len` is inside the window this address maps (the
            // bound is checked above), and the aliasing is the one the window
            // exists to model — see `ClientWindow`'s note.
            unsafe { slice::from_raw_parts(base.add(at), len as usize) }.to_vec()
        }

        /// Complete every outstanding transfer, as the controller's transfer
        /// events would.
        fn complete_all(&mut self) {
            for (_, _, cookie) in self.posted.drain(..) {
                self.done.push(TxCompletion {
                    cookie,
                    status: TxStatus::Sent,
                });
            }
        }
    }

    /// The borrow a [`FakeHost`] lends for one drive cycle.
    struct FakeLease<'a>(&'a mut FakeNic);

    impl NicBackend for FakeLease<'_> {
        type Error = ();

        fn mac(&self) -> MacAddr {
            self.0.mac
        }
        fn link(&self) -> LinkState {
            self.0.link
        }
        fn steering(&self) -> Steering {
            Steering::SOFTWARE
        }
        fn counters(&self) -> NicCounters {
            self.0.counters
        }
        fn set_packet_filter(&mut self, filter: PacketFilter) -> Result<(), ()> {
            self.0.filters.push(filter);
            Ok(())
        }
        fn transmit(&mut self, frame: TxFrame<'_>) -> Result<(), ()> {
            if self.0.busy {
                return Err(());
            }
            self.0
                .posted
                .push((frame.iova(), frame.len(), frame.cookie()));
            Ok(())
        }
        fn service(&mut self, now_ns: Option<u64>) {
            assert!(
                self.0.reaps > self.0.service_calls,
                "completion processing precedes link management"
            );
            self.0.service_calls += 1;
            self.0.service_time = now_ns;
            if let (Some(now), Some((deadline, link))) = (now_ns, self.0.service_link) {
                if now >= deadline {
                    self.0.link = link;
                    self.0.service_link = None;
                }
            }
        }
        fn service_deadline(&self) -> Option<u64> {
            self.0.service_link.map(|(deadline, _)| deadline)
        }
        fn flush_tx(&mut self) {
            self.0.flushes += 1;
        }
        fn reap_tx(&mut self, out: &mut [TxCompletion]) -> usize {
            self.0.reaps += 1;
            let n = out.len().min(self.0.done.len());
            for (slot, completion) in out[..n].iter_mut().zip(self.0.done.drain(..n)) {
                *slot = completion;
            }
            n
        }
        fn poll_rx(&mut self, f: &mut dyn FnMut(&[u8])) -> bool {
            if self.0.inbound.is_empty() {
                return false;
            }
            let frame = self.0.inbound.remove(0);
            f(&frame);
            true
        }
        fn refill_rx(&mut self) -> Result<(), ()> {
            self.0.refills += 1;
            Ok(())
        }
    }

    /// A host that owns a fake device and cannot adopt windows — the test
    /// substitutes [`NetService::open_vif`]'s closure instead, which is the
    /// whole reason the untestable step is a closure.
    struct FakeHost {
        nic: Option<FakeNic>,
        /// A bound radio's facts, on the host that models one.
        radio: Option<RadioReport>,
        /// What this double's device declares as its DMA reach, and how
        /// long its pool window is — the two facts the engine's address
        /// plan is derived from. The default pair is the wide, pool-less
        /// one every other test wants.
        dma_address_bits: u32,
        dma_pool_bytes: u64,
    }

    impl FakeHost {
        fn up() -> Self {
            Self {
                nic: Some(FakeNic::new()),
                radio: None,
                dma_address_bits: 64,
                dma_pool_bytes: 0,
            }
        }
        /// The claim-less machine.
        fn degraded() -> Self {
            Self {
                nic: None,
                radio: None,
                dma_address_bits: 64,
                dma_pool_bytes: 0,
            }
        }
        /// A radio bound and answering with no frame path — the #388 D1
        /// posture: no backend, but a report.
        fn radio_only(report: RadioReport) -> Self {
            Self {
                nic: None,
                radio: Some(report),
                dma_address_bits: 64,
                dma_pool_bytes: 0,
            }
        }
        /// **The MT7925's posture**: a 32-bit DMA master with the standard
        /// 2 MiB driver pool, which `dma_iova_base` therefore places at
        /// `NARROW_DMA_IOVA_BASE`.
        fn narrow() -> Self {
            Self {
                dma_address_bits: kore_mt7925::dma::DMA_ADDR_BITS,
                dma_pool_bytes: 0x20_0000,
                ..Self::up()
            }
        }
        fn nic(&mut self) -> &mut FakeNic {
            self.nic.as_mut().expect("a device")
        }
    }

    impl NicHost for FakeHost {
        type Backend<'a> = FakeLease<'a>;
        fn backend(&mut self) -> Option<FakeLease<'_>> {
            self.nic.as_mut().map(FakeLease)
        }
    }

    impl NetHost for FakeHost {
        type Lease = ();
        fn dma_address_bits(&self) -> u32 {
            self.dma_address_bits
        }
        fn dma_pool_bytes(&self) -> u64 {
            self.dma_pool_bytes
        }
        fn radio(&self) -> Option<RadioReport> {
            self.radio
        }
    }

    // ── the client's half of a vif window ──────────────────────────────

    /// The client side of one vif window.
    ///
    /// The service holds `&'static mut` over these same bytes, and so does
    /// this. That aliasing is precisely what the window models — two address
    /// spaces mapping one lease — and `spsc` sanctions it for host tests in
    /// [`spsc::split`]'s own words: the halves touch the window only through
    /// the ring protocol (disjoint cursors, slot hand-off by Release/Acquire),
    /// and the frame slots are disjoint by direction, this side writing
    /// transmit and reading receive.
    struct ClientWindow {
        base: *mut u8,
        tx: spsc::Producer<'static, FrameDesc>,
        rx: spsc::Consumer<'static, FrameDesc>,
        pushed: u64,
        read: u64,
    }

    impl core::fmt::Debug for ClientWindow {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_str("ClientWindow")
        }
    }

    impl PartialEq for ClientWindow {
        fn eq(&self, other: &Self) -> bool {
            core::ptr::eq(self.base, other.base)
        }
    }

    impl ClientWindow {
        /// A formatted window: the service's `&'static mut`, and this side.
        fn new() -> (&'static mut [u8], Self) {
            let layout = std::alloc::Layout::from_size_align(WINDOW_BYTES, 64).expect("layout");
            // SAFETY: non-zero size, and never freed — the window is
            // deliberately `'static`, as a transferred lease is.
            let base = unsafe { std::alloc::alloc_zeroed(layout) };
            assert!(!base.is_null(), "window allocation");
            // SAFETY: `base` addresses `WINDOW_BYTES` zeroed, 64-aligned bytes.
            let bytes: &'static mut [u8] = unsafe { slice::from_raw_parts_mut(base, WINDOW_BYTES) };
            format_window(bytes).expect("the client formats both rings");
            // SAFETY: the two ring subregions are disjoint and stay within the
            // window; see the type's note on the aliasing this models.
            let tx_ring = unsafe { slice::from_raw_parts_mut(base, RX_RING_AT) };
            let rx_ring =
                unsafe { slice::from_raw_parts_mut(base.add(RX_RING_AT), TX_AREA_AT - RX_RING_AT) };
            let tx = spsc::producer::<FrameDesc>(tx_ring).expect("transmit ring");
            let rx = spsc::consumer::<FrameDesc>(rx_ring).expect("receive ring");
            (
                bytes,
                Self {
                    base,
                    tx,
                    rx,
                    pushed: 0,
                    read: 0,
                },
            )
        }

        /// Stage `frame` in the transmit slot the next push will use, and
        /// publish a descriptor for it — the client's whole transmit path.
        fn send(&mut self, frame: &[u8]) -> bool {
            self.send_declaring(
                frame,
                EtherType::from_wire([frame[12], frame[13]]),
                frame.len(),
            )
        }

        /// [`send`](Self::send) with the descriptor's declared EtherType and
        /// length chosen independently of the bytes — how a hostile client
        /// lies.
        fn send_declaring(&mut self, frame: &[u8], ethertype: EtherType, len: usize) -> bool {
            let slot = kore_driver::net::tx_slot_of(self.pushed) as usize;
            let at = TX_AREA_AT + slot * SLOT_BYTES;
            // SAFETY: `slot < TX_DEPTH`, so `[at, at+SLOT_BYTES)` is inside
            // this window's transmit area, which only this side writes.
            let dst = unsafe { slice::from_raw_parts_mut(self.base.add(at), SLOT_BYTES) };
            dst[..frame.len()].copy_from_slice(frame);
            let pushed = self.tx.push(FrameDesc::new(len as u16, ethertype)).is_ok();
            if pushed {
                self.pushed = self.pushed.wrapping_add(1);
            }
            pushed
        }

        /// Take one received frame, as the client's own drain would.
        fn recv(&mut self) -> Option<Vec<u8>> {
            let desc = self.rx.pop()?;
            let slot = kore_driver::net::rx_slot_of(self.read) as usize;
            self.read = self.read.wrapping_add(1);
            let at = TX_AREA_AT + TX_DEPTH as usize * SLOT_BYTES + slot * SLOT_BYTES;
            // SAFETY: `slot < RX_DEPTH`, so this is inside the receive area,
            // and the descriptor's arrival is the service's Release publish.
            let bytes = unsafe { slice::from_raw_parts(self.base.add(at), desc.len as usize) };
            Some(bytes.to_vec())
        }

        /// Transmit slots the service has released back.
        fn tx_free(&self) -> u32 {
            self.tx.capacity_left()
        }
    }

    // ── fixtures ───────────────────────────────────────────────────────

    /// A pair that may speak three protocols and hold four vifs.
    fn policy(max_vifs: u8) -> VifPolicy {
        VifPolicy {
            flow: None,
            ethertype: EtherTypeMatch::only(&[
                EtherType::ARP,
                EtherType::IPV4,
                EtherType::KORE_PEER,
            ])
            .expect("within the bound"),
            multicast: MulticastMatch::None,
            broadcast: true,
            promiscuous: false,
            source_mac: false,
            max_vifs,
        }
    }

    fn service(host: FakeHost, max_vifs: u8) -> NetService<FakeHost> {
        NetService::host(host, None, [policy(max_vifs); PAIRS])
    }

    /// An arrival is `generation * SLOT_CAPACITY + slot`, and a zero
    /// generation names no live pair.
    const fn arrival_of(slot: u16, generation: u16) -> u16 {
        generation * PeerId::SLOT_CAPACITY as u16 + slot
    }

    /// **The mint's cell rule.** A minted face seated where a FIXED pair
    /// already lives is refused: `via` would otherwise name two different
    /// faces, and the policy table is indexed by exactly that.
    ///
    /// Every fixed index is checked rather than one, because the bound is
    /// what the rule is about — an off-by-one here hands a client the row's
    /// own authority.
    #[test]
    fn a_minted_face_may_not_land_on_a_fixed_pair() {
        let mut svc = service(FakeHost::up(), 2);
        svc.configure_peer_map(0x1000);

        for slot in 0..crate::wire::NET_SERVE_PAIRS as u16 {
            svc.peers.pending = true;
            let peer = PeerId::from_arrival(arrival_of(slot, 1)).expect("a live arrival");
            assert!(
                !svc.peer_admitted(Ok(peer)),
                "slot {slot} is a fixed pair's and must not be minted onto"
            );
        }

        // The first slot past the fixed bank is the first one a face may take.
        svc.peers.pending = true;
        let peer = PeerId::from_arrival(arrival_of(crate::wire::NET_SERVE_PAIRS as u16, 1))
            .expect("a live arrival");
        assert!(svc.peer_admitted(Ok(peer)), "the first free cell admits");
    }

    /// A verdict nobody asked for is not recorded: without a bundle in
    /// flight there is no asker, and inventing one would publish an identity
    /// the next `TakePeerAdmission` would wrongly collect.
    #[test]
    fn a_verdict_without_a_pending_bundle_is_ignored() {
        let mut svc = service(FakeHost::up(), 2);
        svc.configure_peer_map(0x1000);
        let peer = PeerId::from_arrival(arrival_of(crate::wire::NET_SERVE_PAIRS as u16, 1))
            .expect("a live arrival");
        assert!(!svc.peer_admitted(Ok(peer)), "nothing was pending");
    }

    /// Until the row's transient geometry is known the mint has no base to
    /// map into, and says so rather than inventing one.
    #[test]
    fn admission_without_a_map_is_unsupported() {
        let svc = service(FakeHost::up(), 2);
        assert!(svc.peers.map.is_none(), "no geometry configured");
    }

    /// **A minted face's cell is its slot, never its arrival.**
    ///
    /// The arrival is what the wire carries and what `Serve::via` reports, so
    /// the mapping is on every authorization path. Reading it raw would put
    /// every minted face outside both arrays — and the policy lookup would
    /// refuse a face that should work, or, on a longer array, resolve onto
    /// somebody else's.
    #[test]
    fn a_vias_cell_is_its_slot_for_a_minted_face_and_itself_for_a_fixed_one() {
        for fixed in 0..crate::wire::NET_SERVE_PAIRS as u16 {
            assert_eq!(
                pair_index(fixed),
                Some(usize::from(fixed)),
                "a fixed pair indexes itself"
            );
        }
        let slot = crate::wire::NET_SERVE_PAIRS;
        let peer = PeerId::from_arrival(arrival_of(slot as u16, 3)).expect("a live arrival");
        assert_ne!(
            usize::from(peer.arrival()),
            slot,
            "the arrival is not the cell — which is the whole point"
        );
        assert_eq!(pair_index(peer.arrival()), Some(slot));
        // A slot past the bank is refused rather than wrapped onto a live cell.
        assert_eq!(pair_index(arrival_of(PAIRS as u16, 1)), None);
    }

    /// **A minted face is woken.**
    ///
    /// The wake bitmask is indexed by cell, so a face whose `via` is an
    /// arrival still gets its bit. Indexed raw, an arrival is past the
    /// mask's width entirely: no bit was set, the ring still held the frame,
    /// and a client parked on its doorbell waited for a notice that could
    /// never come.
    #[test]
    fn a_minted_faces_vif_sets_its_cells_wake_bit() {
        let mut svc = service(FakeHost::up(), 4);
        let cell = crate::wire::NET_SERVE_PAIRS as u16;
        let minted = arrival_of(cell, 1);
        assert!(
            minted > 15,
            "an arrival is past the wake mask's width — the point of the bug"
        );

        let (_vif, mut client) =
            open(&mut svc, minted, FrameFilter::protocol(EtherType::ARP)).expect("minted vif");
        let arp = frame(MacAddr::BROADCAST, PEER_MAC, EtherType::ARP, 28, 0xa1);
        svc.host_mut().nic().inbound.push(arp.clone());
        let report = svc.drive();

        assert_eq!(report.rx_delivered, 1, "the frame reached the minted vif");
        assert_eq!(client.recv().as_deref(), Some(arp.as_slice()));
        assert_eq!(
            report.wake,
            1 << cell,
            "the bit is the cell's, and it is set"
        );
    }

    /// **A departing face takes its vifs with it.**
    ///
    /// Without this the classifier keeps writing frames into rings nobody
    /// reads and the window leases are never released — a leak that fills
    /// the vif table over a boot's worth of launches.
    #[test]
    fn closing_a_minted_face_retires_the_vifs_it_held() {
        let mut svc = service(FakeHost::up(), 4);
        let minted = arrival_of(crate::wire::NET_SERVE_PAIRS as u16, 1);
        let other = arrival_of(crate::wire::NET_SERVE_PAIRS as u16 + 1, 1);

        open(&mut svc, minted, FrameFilter::protocol(EtherType::ARP)).expect("its vif");
        open(&mut svc, other, FrameFilter::protocol(EtherType::IPV4)).expect("a bystander's");
        assert_eq!(svc.open_vifs(), 2);

        assert!(svc.close_peer_vifs(minted), "nothing of its left");
        assert_eq!(svc.open_vifs(), 1, "only its own went");
    }

    /// A successor in the same cell inherits nothing: ownership is the full
    /// arrival, so the generation separates the two occupants even though
    /// they index the same policy and session.
    #[test]
    fn a_successor_in_a_cell_inherits_nothing_from_its_predecessor() {
        let mut svc = service(FakeHost::up(), 4);
        let cell = crate::wire::NET_SERVE_PAIRS as u16;
        let first = arrival_of(cell, 1);
        let second = arrival_of(cell, 2);
        assert_eq!(
            pair_index(first),
            pair_index(second),
            "the same cell, by construction"
        );

        let (vif, _client) =
            open(&mut svc, first, FrameFilter::protocol(EtherType::ARP)).expect("the first's vif");
        assert_eq!(
            svc.stats_arm(second, vif),
            Err(NetRefusal::NotOwner),
            "the successor may not read its predecessor's vif"
        );
        assert!(
            svc.close_peer_vifs(second),
            "and closing the successor retires none of them"
        );
        assert_eq!(svc.open_vifs(), 1, "the first's vif is untouched");
    }

    /// **Retirement is exact.** A cell records the generation it holds, so a
    /// stale identity names an occupant that has already gone — acting on it
    /// would retire that cell's *successor*.
    #[test]
    fn a_stale_generation_is_not_current_and_its_successor_is() {
        let mut svc = service(FakeHost::up(), 2);
        svc.configure_peer_map(0x1000);
        let cell = crate::wire::NET_SERVE_PAIRS as u16;
        let first = PeerId::from_arrival(arrival_of(cell, 1)).expect("a live arrival");
        let second = PeerId::from_arrival(arrival_of(cell, 2)).expect("a live arrival");

        svc.peers.pending = true;
        assert!(svc.peer_admitted(Ok(first)));
        assert!(svc.peer_is_current(first), "it is the one seated");
        assert!(
            !svc.peer_is_current(second),
            "its successor is not here yet"
        );

        // The cell turns over.
        svc.peers.pending = true;
        assert!(svc.peer_admitted(Ok(second)));
        assert!(
            !svc.peer_is_current(first),
            "the stale generation must not resolve to the new occupant"
        );
        assert!(svc.peer_is_current(second));
    }

    /// A face that held nothing is current until it closes, and not after —
    /// which is exactly what makes `ConfirmRetired` answerable.
    #[test]
    fn closing_a_face_frees_its_cell() {
        let mut svc = service(FakeHost::up(), 2);
        svc.configure_peer_map(0x1000);
        let peer = PeerId::from_arrival(arrival_of(crate::wire::NET_SERVE_PAIRS as u16, 1))
            .expect("a live arrival");
        svc.peers.pending = true;
        assert!(svc.peer_admitted(Ok(peer)));
        assert!(svc.peer_is_current(peer));

        assert!(svc.close_peer_vifs(peer.arrival()), "nothing of it remains");
        assert!(
            !svc.peer_is_current(peer),
            "the cell is free, so the retirement can be confirmed"
        );
    }

    /// A minted face lands on a cell carrying the ordinary client policy —
    /// which is what makes it able to open a vif at all. Only exec's pair is
    /// `DENY`, and a minted face never lands there.
    #[test]
    fn a_minted_face_carries_the_client_policy() {
        let mut svc = service(FakeHost::up(), 2);
        svc.configure_peer_map(0x1000);
        let peer = PeerId::from_arrival(arrival_of(crate::wire::NET_SERVE_PAIRS as u16, 1))
            .expect("a live arrival");
        svc.peers.pending = true;
        assert!(svc.peer_admitted(Ok(peer)));
        let granted = svc
            .policy_of(peer.arrival())
            .expect("a minted face has a policy");
        assert_eq!(granted.max_vifs, 2, "the ordinary client grant, not DENY");
    }

    /// One Ethernet II frame with a `fill`-byte payload.
    fn frame(
        dst: MacAddr,
        src: MacAddr,
        ethertype: EtherType,
        payload: usize,
        fill: u8,
    ) -> Vec<u8> {
        let mut bytes = std::vec![fill; HEADER_LEN + payload];
        FrameHead {
            dst,
            src,
            ethertype,
        }
        .write(&mut bytes)
        .expect("header fits");
        bytes
    }

    /// A UDP datagram inside an Ethernet frame, addressed to the device.
    fn udp_frame(
        src_ip: kore_driver::net::flow::Ipv4Addr,
        dst_ip: kore_driver::net::flow::Ipv4Addr,
        sport: u16,
        dport: u16,
    ) -> Vec<u8> {
        let mut bytes = frame(DEVICE_MAC, PEER_MAC, EtherType::IPV4, 28, 0);
        let ip = &mut bytes[HEADER_LEN..];
        ip[0] = 0x45;
        ip[1] = 0;
        ip[2..4].copy_from_slice(&28u16.to_be_bytes());
        ip[4..9].copy_from_slice(&[0, 0, 0, 0, 64]);
        ip[9] = kore_driver::net::flow::IpProto::UDP.0;
        ip[10..12].copy_from_slice(&[0, 0]);
        ip[12..16].copy_from_slice(&src_ip.0);
        ip[16..20].copy_from_slice(&dst_ip.0);
        ip[20..22].copy_from_slice(&sport.to_be_bytes());
        ip[22..24].copy_from_slice(&dport.to_be_bytes());
        ip[24..28].copy_from_slice(&[0, 8, 0, 0]);
        bytes
    }

    /// Open a vif and register its window with the fake IOMMU, returning the
    /// client's half.
    fn run<T>(f: impl core::future::Future<Output = T>) -> T {
        kore_rt::try_block_on(f, 4).expect("this row's futures settle without an executor")
    }

    fn open(
        svc: &mut NetService<FakeHost>,
        via: u16,
        filter: FrameFilter,
    ) -> Result<(VifId, ClientWindow), NetRefusal> {
        let mut made: Option<(ClientWindow, u64)> = None;
        let opened = svc.open_vif(via, filter, |_, iova| {
            let (bytes, client) = ClientWindow::new();
            made = Some((client, iova.0));
            Ok(AdoptedWindow {
                bytes,
                iova,
                lease: (),
            })
        });
        match opened {
            Ok((id, _mac)) => {
                let (client, iova) = made.expect("the adopt closure ran");
                svc.host_mut().nic().windows.push((iova, client.base));
                Ok((id, client))
            }
            Err(refusal) => {
                assert!(
                    made.is_none(),
                    "a refused open must not have spent a window"
                );
                Err(refusal)
            }
        }
    }

    // ── authority ──────────────────────────────────────────────────────

    /// The forged-`Any` case: a client whose pair may speak three named
    /// protocols asks for every protocol. It is refused **by name**, and — the
    /// half that matters — the refusal costs nothing: no window was adopted,
    /// no device address was reserved.
    #[test]
    fn a_widened_filter_is_refused_before_anything_is_spent() {
        let mut svc = service(FakeHost::up(), 2);
        let mut wide = FrameFilter::protocol(EtherType::ARP);
        wide.ethertype = EtherTypeMatch::Any;
        assert_eq!(open(&mut svc, 0, wide), Err(NetRefusal::PolicyEtherType));
        assert_eq!(svc.open_vifs(), 0);

        let mut promiscuous = FrameFilter::protocol(EtherType::ARP);
        promiscuous.promiscuous = true;
        assert_eq!(
            open(&mut svc, 0, promiscuous),
            Err(NetRefusal::PolicyPromiscuous),
            "promiscuity is separately gated from protocol"
        );

        let mut groups = FrameFilter::protocol(EtherType::ARP);
        groups.multicast = MulticastMatch::All;
        assert_eq!(open(&mut svc, 0, groups), Err(NetRefusal::PolicyMulticast));

        // The narrow request its pair does authorize still works.
        assert!(open(&mut svc, 0, FrameFilter::protocol(EtherType::ARP)).is_ok());
    }

    /// The vif budget is the pair's, not the table's.
    #[test]
    fn the_pairs_vif_budget_is_enforced() {
        let mut svc = service(FakeHost::up(), 1);
        assert!(open(&mut svc, 0, FrameFilter::protocol(EtherType::ARP)).is_ok());
        assert_eq!(
            open(&mut svc, 0, FrameFilter::protocol(EtherType::IPV4)),
            Err(NetRefusal::PolicyVifs)
        );
        assert_eq!(svc.open_vifs(), 1);
    }

    /// Closing a vif gives its ports back.
    ///
    /// `close_vif` retired the vif and its mapping but never touched the port
    /// table, so each bind/close cycle retained an entry. The table has 64
    /// slots and a port held by a vif that no longer exists refuses the next
    /// binder, so the observable is simple: bind a port, close the vif that
    /// holds it, and a *different* vif must be able to bind the same port.
    #[test]
    fn closing_a_vif_frees_the_ports_it_held() {
        let mut svc = service(FakeHost::up(), 1);
        svc.configure_arm(Ipv4Addr([10, 0, 0, 2]), 24, None)
            .expect("the station takes an address");

        let (first, _c1) = open(&mut svc, 0, FrameFilter::protocol(EtherType::IPV4)).expect("open");
        let grant = svc.flow_grant(0);
        run(svc.open_listener(0, first, grant, IpProto::UDP, 8080)).expect("binds its port");
        svc.close_vif(0, first).expect("its owner closes it");

        let (second, _c2) =
            open(&mut svc, 0, FrameFilter::protocol(EtherType::IPV4)).expect("reopen");
        let grant = svc.flow_grant(0);
        run(svc.open_listener(0, second, grant, IpProto::UDP, 8080))
            .expect("the port came back with the closed vif");
    }

    /// A close is two mutations — the port table and the installed filter —
    /// and a caller naming another pair's vif must cause neither.
    ///
    /// The ordering is the whole test: the port release used to run first and
    /// unconditionally, so an intruder's close freed a port that was never
    /// theirs and the count alone could not say so.
    #[test]
    fn another_pairs_close_frees_nothing() {
        let mut svc = service(FakeHost::up(), 1);
        svc.configure_arm(Ipv4Addr([10, 0, 0, 2]), 24, None)
            .expect("the station takes an address");
        let (id, _client) =
            open(&mut svc, 0, FrameFilter::protocol(EtherType::IPV4)).expect("open");
        let grant = svc.flow_grant(0);
        run(svc.open_listener(0, id, grant, IpProto::UDP, 8080)).expect("binds its port");

        assert_eq!(
            run(svc.close_flows(1, id)),
            (0, Err(InstallRefusal::NotOwner)),
            "another pair's close frees nothing, and is told why"
        );
        assert_eq!(
            run(svc.close_flows(0, id)),
            (1, Ok(())),
            "and the port was still there for its owner to free"
        );
    }

    /// Vif ownership is keyed on the arrival pair, so a second face cannot
    /// reach the first's vifs even holding a correct identifier.
    #[test]
    fn another_pairs_vif_is_unreachable_even_by_exact_identity() {
        let mut svc = service(FakeHost::up(), 2);
        let (id, _client) = open(&mut svc, 0, FrameFilter::protocol(EtherType::ARP)).expect("open");
        assert_eq!(svc.stats_arm(1, id), Err(NetRefusal::NotOwner));
        assert_eq!(svc.close_vif(1, id), Err(NetRefusal::NotOwner));
        assert!(
            svc.stats_arm(0, id).is_ok(),
            "its own pair still reaches it"
        );
        assert_eq!(svc.close_vif(0, id), Ok(id));
    }

    // ── identity ───────────────────────────────────────────────────────

    /// The use-after-close case, and the reuse case behind it: closing bumps
    /// the generation, so the old identifier does not become a handle on the
    /// slot's next tenant.
    #[test]
    fn a_closed_identifier_never_reaches_the_slots_next_tenant() {
        let mut svc = service(FakeHost::up(), 4);
        let (first, _a) = open(&mut svc, 0, FrameFilter::protocol(EtherType::ARP)).expect("open");
        assert_eq!(svc.close_vif(0, first), Ok(first));
        assert_eq!(svc.stats_arm(0, first), Err(NetRefusal::StaleVif));
        assert_eq!(svc.close_vif(0, first), Err(NetRefusal::StaleVif));

        let (second, _b) =
            open(&mut svc, 0, FrameFilter::protocol(EtherType::KORE_PEER)).expect("reopen");
        assert_eq!(second.index, first.index, "the slot really was reused");
        assert_ne!(second.generation, first.generation);
        assert_eq!(
            svc.stats_arm(0, first),
            Err(NetRefusal::StaleVif),
            "the old identifier still does not resolve"
        );
    }

    // ── isolation ──────────────────────────────────────────────────────

    /// **The isolation witness.** Two vifs, two separate windows, one wire:
    /// a frame reaches only the vif whose filter accepts it, and the vif that
    /// did not want it records no drop either — it was never offered.
    #[test]
    fn a_frame_reaches_only_the_vifs_whose_filter_accepts_it() {
        let mut svc = service(FakeHost::up(), 4);
        let (infra, mut infra_client) =
            open(&mut svc, 0, FrameFilter::protocol(EtherType::ARP)).expect("infra vif");
        let (peer, mut peer_client) =
            open(&mut svc, 0, FrameFilter::protocol(EtherType::KORE_PEER)).expect("peer vif");

        let arp = frame(MacAddr::BROADCAST, PEER_MAC, EtherType::ARP, 28, 0xa1);
        svc.host_mut().nic().inbound.push(arp.clone());
        let report = svc.drive();

        assert_eq!(report.rx_polled, 1);
        assert_eq!(report.rx_delivered, 1, "one frame, one matching vif");
        assert_eq!(report.rx_dropped, 0);
        assert_eq!(infra_client.recv().as_deref(), Some(arp.as_slice()));
        assert!(peer_client.recv().is_none(), "the peer vif saw nothing");
        assert_eq!(svc.counters(infra).unwrap().rx_frames, 1);
        assert_eq!(svc.counters(peer).unwrap().rx_frames, 0);
        assert_eq!(
            svc.counters(peer).unwrap().rx_dropped,
            0,
            "not offered is not dropped"
        );

        // The mirror image: a KORE_PEER frame reaches the peer vif alone.
        let hello = frame(DEVICE_MAC, PEER_MAC, EtherType::KORE_PEER, 64, 0xb2);
        svc.host_mut().nic().inbound.push(hello.clone());
        svc.drive();
        assert_eq!(peer_client.recv().as_deref(), Some(hello.as_slice()));
        assert!(infra_client.recv().is_none());
    }

    /// **The flow isolation witness.** Two vifs that both bind IPv4 and both
    /// answer to the device's own address — the shape `AddressMode::Shared`
    /// forces, where an L2 filter cannot tell them apart at all. Each binds a
    /// different flow, and a datagram reaches exactly the one it belongs to.
    ///
    /// Without the flow half both vifs match every IPv4 frame and the
    /// classifier copies each datagram into both, which is the confidentiality
    /// hole a second IP consumer would have opened.
    #[test]
    fn two_ip_vifs_on_one_address_receive_only_their_own_flow() {
        use kore_driver::net::flow::{FlowMatch, IpProto, Ipv4Addr};

        const LOCAL: Ipv4Addr = Ipv4Addr([10, 0, 2, 15]);
        const ALICE: Ipv4Addr = Ipv4Addr([10, 0, 2, 2]);
        const BOB: Ipv4Addr = Ipv4Addr([10, 0, 2, 3]);

        let mut svc = service(FakeHost::up(), 4);

        let mut mine = FrameFilter::protocol(EtherType::IPV4);
        mine.flow = Some(FlowMatch::connection(
            IpProto::UDP,
            (LOCAL, 5000),
            (ALICE, 40000),
        ));
        let mut theirs = FrameFilter::protocol(EtherType::IPV4);
        theirs.flow = Some(FlowMatch::connection(
            IpProto::UDP,
            (LOCAL, 5001),
            (BOB, 40001),
        ));

        let (a, mut a_client) = open(&mut svc, 0, mine).expect("vif a");
        let (b, mut b_client) = open(&mut svc, 0, theirs).expect("vif b");

        let to_a = udp_frame(ALICE, LOCAL, 40000, 5000);
        svc.host_mut().nic().inbound.push(to_a.clone());
        let report = svc.drive();

        assert_eq!(report.rx_delivered, 1, "one frame, one matching flow");
        assert_eq!(report.rx_dropped, 0);
        assert_eq!(a_client.recv().as_deref(), Some(to_a.as_slice()));
        assert!(
            b_client.recv().is_none(),
            "the other vif shares the EtherType and the address, and still saw nothing"
        );
        assert_eq!(svc.counters(b).unwrap().rx_frames, 0);
        assert_eq!(
            svc.counters(b).unwrap().rx_dropped,
            0,
            "not offered is not dropped"
        );

        // The mirror image, so neither vif is merely inert.
        let to_b = udp_frame(BOB, LOCAL, 40001, 5001);
        svc.host_mut().nic().inbound.push(to_b.clone());
        svc.drive();
        assert_eq!(b_client.recv().as_deref(), Some(to_b.as_slice()));
        assert!(a_client.recv().is_none());

        // A datagram belonging to neither flow reaches neither vif.
        let stray = udp_frame(ALICE, LOCAL, 40000, 5002);
        svc.host_mut().nic().inbound.push(stray);
        let report = svc.drive();
        assert_eq!(report.rx_delivered, 0, "no vif bound that flow");
        assert!(a_client.recv().is_none());
        assert!(b_client.recv().is_none());
        assert_eq!(svc.counters(a).unwrap().rx_frames, 1);
    }

    // ── re-filtering an open vif ───────────────────────────────────────

    /// **The verb the flow mint needs.** A vif opened for IPv4 at large
    /// receives every IPv4 datagram reaching the station; re-filtered to one
    /// flow, it receives that flow and nothing else.
    ///
    /// This is what makes a minted flow capability *enforced* rather than
    /// described. `OpenVif` fixes a filter at open, and an arbiter one layer
    /// up cannot name a flow before it has arbitrated one — so without a verb
    /// that narrows an open vif, the classifier keeps running whatever the
    /// holder asked for at open, and the capability means nothing.
    #[test]
    fn re_filtering_a_vif_narrows_what_it_receives() {
        use kore_driver::net::flow::{FlowMatch, IpProto, Ipv4Addr};

        const LOCAL: Ipv4Addr = Ipv4Addr([10, 0, 2, 15]);
        const ALICE: Ipv4Addr = Ipv4Addr([10, 0, 2, 2]);

        let mut svc = service(FakeHost::up(), 4);
        let (id, mut client) =
            open(&mut svc, 0, FrameFilter::protocol(EtherType::IPV4)).expect("vif");

        // Opened at L2 only: any IPv4 datagram to the station is delivered.
        let stray = udp_frame(ALICE, LOCAL, 40000, 5002);
        svc.host_mut().nic().inbound.push(stray.clone());
        assert_eq!(svc.drive().rx_delivered, 1);
        assert_eq!(client.recv().as_deref(), Some(stray.as_slice()));

        let mut narrowed = FrameFilter::protocol(EtherType::IPV4);
        narrowed.flow = Some(FlowMatch::connection(
            IpProto::UDP,
            (LOCAL, 5000),
            (ALICE, 40000),
        ));
        assert_eq!(svc.set_filter(0, id, narrowed), Ok(id));

        // The same stray datagram now reaches nobody.
        let stray = udp_frame(ALICE, LOCAL, 40000, 5002);
        svc.host_mut().nic().inbound.push(stray);
        assert_eq!(
            svc.drive().rx_delivered,
            0,
            "the vif no longer binds that flow"
        );
        assert!(client.recv().is_none());

        // And the flow it was narrowed to still arrives, so the narrowing is
        // a narrowing rather than a mute.
        let mine = udp_frame(ALICE, LOCAL, 40000, 5000);
        svc.host_mut().nic().inbound.push(mine.clone());
        assert_eq!(svc.drive().rx_delivered, 1);
        assert_eq!(client.recv().as_deref(), Some(mine.as_slice()));
    }

    /// Holding a vif is not authority to widen what it accepts: the new
    /// filter is re-authorized in full, and the refusal names the dimension
    /// that exceeded.
    ///
    /// The property that keeps this verb from being a policy bypass. If a
    /// client could re-filter freely, every grant checked at `OpenVif` would
    /// be a formality it could step around on its second request.
    #[test]
    fn a_re_filter_is_re_authorized_in_full() {
        let mut svc = service(FakeHost::up(), 4);
        let (id, client) = open(&mut svc, 0, FrameFilter::protocol(EtherType::ARP)).expect("vif");
        core::mem::forget(client);

        let mut capture = FrameFilter::protocol(EtherType::ARP);
        capture.promiscuous = true;
        assert_eq!(
            svc.set_filter(0, id, capture),
            Err(NetRefusal::PolicyPromiscuous),
            "a power the open was refused is refused again here"
        );

        let mut any = FrameFilter::protocol(EtherType::ARP);
        any.ethertype = EtherTypeMatch::Any;
        assert_eq!(
            svc.set_filter(0, id, any),
            Err(NetRefusal::PolicyEtherType),
            "each dimension is named separately, as at open"
        );
    }

    /// A refused re-filter changes nothing: the vif keeps the filter it had,
    /// and keeps receiving what that filter admits.
    ///
    /// Fail-closed in the direction that matters here — a half-applied
    /// re-filter would leave a vif enforcing neither the old authority nor
    /// the new one.
    #[test]
    fn a_refused_re_filter_leaves_the_old_filter_installed() {
        let mut svc = service(FakeHost::up(), 4);
        let (id, mut client) =
            open(&mut svc, 0, FrameFilter::protocol(EtherType::ARP)).expect("vif");

        let mut capture = FrameFilter::protocol(EtherType::ARP);
        capture.promiscuous = true;
        assert!(svc.set_filter(0, id, capture).is_err());

        let arp = frame(DEVICE_MAC, PEER_MAC, EtherType::ARP, 32, 7);
        svc.host_mut().nic().inbound.push(arp.clone());
        assert_eq!(svc.drive().rx_delivered, 1);
        assert_eq!(client.recv().as_deref(), Some(arp.as_slice()));
    }

    /// Ownership and liveness are re-checked exactly as on every other op
    /// that names a vif: another pair cannot re-filter this one's vif, and a
    /// closed identifier cannot re-filter the slot's next tenant.
    ///
    /// Re-filtering is the one verb that can *weaken* a vif's isolation, so
    /// reaching it by guessing an identity would be worse than reaching
    /// `Stats` that way.
    #[test]
    fn a_re_filter_needs_ownership_and_a_live_identity() {
        let mut svc = service(FakeHost::up(), 4);
        let (id, client) = open(&mut svc, 0, FrameFilter::protocol(EtherType::ARP)).expect("vif");
        core::mem::forget(client);

        let plain = FrameFilter::protocol(EtherType::ARP);
        assert_eq!(
            svc.set_filter(1, id, plain),
            Err(NetRefusal::NotOwner),
            "a second face cannot re-filter by guessing the identity"
        );

        svc.close_vif(0, id).expect("closed");
        assert_eq!(
            svc.set_filter(0, id, plain),
            Err(NetRefusal::StaleVif),
            "the identifier died with the vif"
        );
    }

    /// A vif at the holder's granted ceiling may still be re-filtered: no new
    /// vif is created, so the open-vif census must not count the one being
    /// re-filtered.
    ///
    /// The bug this pins would be invisible until a client used its full
    /// grant — exactly the client that most needs to narrow.
    #[test]
    fn a_re_filter_does_not_count_against_the_vif_ceiling() {
        use kore_driver::net::flow::{FlowMatch, IpProto, Ipv4Addr};

        let mut svc = service(FakeHost::up(), 1);
        let (id, client) = open(&mut svc, 0, FrameFilter::protocol(EtherType::IPV4)).expect("vif");
        core::mem::forget(client);
        assert_eq!(
            open(&mut svc, 0, FrameFilter::protocol(EtherType::IPV4)),
            Err(NetRefusal::PolicyVifs),
            "the grant is spent, so a second vif is refused"
        );

        let mut narrowed = FrameFilter::protocol(EtherType::IPV4);
        narrowed.flow = Some(FlowMatch::connection(
            IpProto::UDP,
            (Ipv4Addr([10, 0, 2, 15]), 5000),
            (Ipv4Addr([10, 0, 2, 2]), 40000),
        ));
        assert_eq!(
            svc.set_filter(0, id, narrowed),
            Ok(id),
            "re-filtering creates no vif and must not be refused for the census"
        );
    }

    /// Dropping `promiscuous` from the last vif that held it takes the
    /// *device* back out of promiscuous mode.
    ///
    /// Promiscuity is a device mode, not a vif one (`refresh_packet_filter`),
    /// so a re-filter that only rewrote the vif record would leave the
    /// hardware admitting other stations' traffic after the authority to see
    /// it was gone.
    #[test]
    fn a_re_filter_re_derives_the_device_packet_filter() {
        let mut svc = NetService::host(
            FakeHost::up(),
            None,
            [VifPolicy {
                promiscuous: true,
                ..policy(4)
            }; PAIRS],
        );
        let mut capture = FrameFilter::protocol(EtherType::ARP);
        capture.promiscuous = true;
        let (id, client) = open(&mut svc, 0, capture).expect("vif");
        core::mem::forget(client);
        assert_eq!(
            svc.host_mut().nic().filters.last().copied(),
            Some(PacketFilter::PROMISCUOUS)
        );

        assert_eq!(
            svc.set_filter(0, id, FrameFilter::protocol(EtherType::ARP)),
            Ok(id)
        );
        assert_eq!(
            svc.host_mut().nic().filters.last().copied(),
            Some(PacketFilter::DIRECTED),
            "the device stops admitting other stations' traffic"
        );
    }

    /// A slow vif drops **its own** frames and never stalls another — the
    /// KINP posture, which is what keeps a peer vif from wedging infra.
    #[test]
    fn a_full_vif_ring_drops_only_its_own_frames() {
        let mut svc = service(FakeHost::up(), 4);
        let (slow, _slow_client) =
            open(&mut svc, 0, FrameFilter::protocol(EtherType::ARP)).expect("slow vif");
        let (fast, mut fast_client) =
            open(&mut svc, 0, FrameFilter::protocol(EtherType::ARP)).expect("fast vif");

        for i in 0..RX_DEPTH + 4 {
            let f = frame(MacAddr::BROADCAST, PEER_MAC, EtherType::ARP, 28, i as u8);
            svc.host_mut().nic().inbound.push(f);
            svc.drive();
            // The fast vif drains every round; the slow one never does.
            assert!(fast_client.recv().is_some(), "round {i}");
        }

        let slow = svc.counters(slow).unwrap();
        let fast = svc.counters(fast).unwrap();
        assert_eq!(slow.rx_frames, RX_DEPTH as u64, "it took exactly its depth");
        assert_eq!(slow.rx_dropped, 4, "and dropped the rest");
        assert_eq!(fast.rx_frames, RX_DEPTH as u64 + 4);
        assert_eq!(fast.rx_dropped, 0, "the slow vif cost the fast one nothing");
    }

    /// A received reply does not retire the transmit whose DMA produced it.
    /// The load witness must wait for both facts before judging its counters.
    #[test]
    fn receiving_a_reply_does_not_release_an_unreaped_transmit() {
        let mut svc = service(FakeHost::up(), 2);
        let (id, mut client) =
            open(&mut svc, 0, FrameFilter::protocol(EtherType::ARP)).expect("open");
        let request = frame(PEER_MAC, DEVICE_MAC, EtherType::ARP, 28, 0xc3);
        let reply = frame(DEVICE_MAC, PEER_MAC, EtherType::ARP, 28, 0xd4);
        assert!(client.send(&request));
        assert_eq!(svc.drive().tx_submitted, 1);

        svc.host_mut().nic().inbound.push(reply.clone());
        let received = svc.drive();
        assert_eq!(received.rx_delivered, 1);
        assert_eq!(received.tx_completed, 0);
        assert_eq!(client.recv().as_deref(), Some(reply.as_slice()));
        assert_eq!(client.tx.capacity_left(), client.tx.capacity() - 1);
        assert_eq!(svc.counters(id).unwrap().tx_frames, 0);

        svc.host_mut().nic().complete_all();
        assert_eq!(svc.drive().tx_completed, 1);
        assert_eq!(client.tx.capacity_left(), client.tx.capacity());
        assert_eq!(svc.counters(id).unwrap().tx_frames, 1);
    }

    // ── transmit ───────────────────────────────────────────────────────

    /// Transmit is zero-copy: the address handed to hardware is the client's
    /// own slot, derived from the ring position, and the bytes there are the
    /// bytes the client wrote.
    #[test]
    fn transmit_hands_hardware_the_clients_own_slot() {
        let mut svc = service(FakeHost::up(), 2);
        let (_id, mut client) =
            open(&mut svc, 0, FrameFilter::protocol(EtherType::ARP)).expect("open");

        let first = frame(PEER_MAC, DEVICE_MAC, EtherType::ARP, 28, 0xc3);
        let second = frame(PEER_MAC, DEVICE_MAC, EtherType::ARP, 60, 0xd4);
        assert!(client.send(&first));
        assert!(client.send(&second));

        let report = svc.drive();
        assert_eq!(report.tx_submitted, 2);
        assert_eq!(report.tx_refused, 0);

        let nic = svc.host_mut().nic();
        assert_eq!(nic.posted.len(), 2);
        let base = kore_driver::host::VIF_IOVA_BASE;
        assert_eq!(
            nic.posted[0].0.0,
            base + tx_slot_at(0).unwrap() as u64,
            "the first push takes transmit slot 0"
        );
        assert_eq!(
            nic.posted[1].0.0,
            base + tx_slot_at(1).unwrap() as u64,
            "the second takes slot 1 — position derives the slot"
        );
        assert_eq!(nic.posted[0].1 as usize, first.len());
        let (iova, len, _) = nic.posted[0];
        assert_eq!(
            nic.fetch(iova, len),
            first,
            "the device reads the client's bytes"
        );
        let (iova, len, _) = nic.posted[1];
        assert_eq!(nic.fetch(iova, len), second);
    }

    #[test]
    fn idle_link_deadline_runs_without_vifs_and_reports_its_edge_in_the_same_sweep() {
        let mut svc = service(FakeHost::up(), 2);
        svc.host_mut().nic().service_link = Some((50, LinkState::Down));
        let report = svc.drive_at(Some(0));
        assert!(report.is_idle());
        assert_eq!(report.service_deadline, Some(50));
        let report = svc.drive_at(Some(50));
        assert!(report.link_changed);
        assert_eq!(report.service_deadline, None);
        assert_eq!(svc.host_mut().nic().service_calls, 2);
        assert_eq!(svc.host_mut().nic().service_time, Some(50));
        assert!(!svc.drive_at(Some(51)).link_changed);
    }

    #[test]
    fn clock_absence_reaches_backend_without_becoming_a_timestamp() {
        let mut svc = service(FakeHost::up(), 2);
        assert!(svc.drive_at(None).is_idle());
        assert_eq!(svc.host_mut().nic().service_calls, 1);
        assert_eq!(svc.host_mut().nic().service_time, None);
    }

    /// One doorbell per drive, rung after the pump has staged every vif's
    /// descriptors and even when nothing was staged: the engine does not
    /// know which backend it holds, and a tail-pointer NIC's batching depends
    /// on the flush following the whole sweep rather than any one frame.
    #[test]
    fn drive_flushes_transmit_once_after_the_pump_even_with_nothing_staged() {
        let mut svc = service(FakeHost::up(), 2);
        let (_id, mut client) =
            open(&mut svc, 0, FrameFilter::protocol(EtherType::ARP)).expect("open");

        let idle = svc.drive();
        assert!(idle.is_idle());
        assert_eq!(
            svc.host_mut().nic().flushes,
            1,
            "an idle drive still flushes once"
        );

        assert!(client.send(&frame(PEER_MAC, DEVICE_MAC, EtherType::ARP, 28, 1)));
        assert!(client.send(&frame(PEER_MAC, DEVICE_MAC, EtherType::ARP, 28, 2)));
        let report = svc.drive();
        assert_eq!(report.tx_submitted, 2);
        let nic = svc.host_mut().nic();
        assert_eq!(
            nic.posted.len(),
            2,
            "both descriptors were staged by the pump"
        );
        assert_eq!(
            nic.flushes, 2,
            "…and published by exactly one more doorbell"
        );
    }

    /// A transmit slot is released **only when its completion lands**, because
    /// releasing it is what tells the client the bytes may be overwritten —
    /// and the controller is still reading them.
    #[test]
    fn a_transmit_slot_is_released_only_on_completion() {
        let mut svc = service(FakeHost::up(), 2);
        let (id, mut client) =
            open(&mut svc, 0, FrameFilter::protocol(EtherType::ARP)).expect("open");
        let free = client.tx_free();
        assert!(client.send(&frame(PEER_MAC, DEVICE_MAC, EtherType::ARP, 28, 1)));
        assert_eq!(client.tx_free(), free - 1);

        svc.drive();
        assert_eq!(
            client.tx_free(),
            free - 1,
            "submitted is not completed: the slot stays held"
        );
        assert_eq!(svc.counters(id).unwrap().tx_frames, 0);

        // A wire reply may be observed before the transmit completion is
        // reaped. A load witness must wait for both facts independently.
        svc.host_mut()
            .nic()
            .inbound
            .push(frame(DEVICE_MAC, PEER_MAC, EtherType::ARP, 28, 2));
        assert_eq!(svc.drive().rx_delivered, 1);
        assert_eq!(svc.counters(id).unwrap().rx_frames, 1);
        assert_eq!(svc.counters(id).unwrap().tx_frames, 0);
        assert_eq!(client.tx_free(), free - 1);

        svc.host_mut().nic().complete_all();
        svc.drive();
        assert_eq!(client.tx_free(), free, "the completion is the release");
        assert_eq!(svc.counters(id).unwrap().tx_frames, 1);
    }

    /// Transmit pipelining is bounded, and the bound is what the mapping
    /// counts as outstanding — not a guess.
    #[test]
    fn transmit_pipelines_up_to_the_inflight_bound() {
        let mut svc = service(FakeHost::up(), 2);
        let (_id, mut client) =
            open(&mut svc, 0, FrameFilter::protocol(EtherType::ARP)).expect("open");
        for i in 0..TX_DEPTH {
            assert!(client.send(&frame(PEER_MAC, DEVICE_MAC, EtherType::ARP, 28, i as u8)));
        }
        let report = svc.drive();
        assert_eq!(report.tx_submitted, MAX_TX_INFLIGHT);
        assert_eq!(svc.drive().tx_submitted, 0, "the pipeline is full");
        svc.host_mut().nic().complete_all();
        assert_eq!(svc.drive().tx_submitted, MAX_TX_INFLIGHT, "and refills");
    }

    /// Every per-frame transmit gate refuses rather than clamping or
    /// rewriting, and a refused frame never reaches hardware.
    #[test]
    fn the_transmit_gates_refuse_and_never_rewrite() {
        let elsewhere = MacAddr([0x02, 0xaa, 0xbb, 0xcc, 0xdd, 0xee]);
        let cases: [(&str, Vec<u8>, EtherType, usize); 5] = [
            (
                "a source address this vif may not use",
                frame(PEER_MAC, elsewhere, EtherType::ARP, 28, 1),
                EtherType::ARP,
                HEADER_LEN + 28,
            ),
            (
                "an EtherType outside this vif's own filter",
                frame(PEER_MAC, DEVICE_MAC, EtherType::IPV6, 28, 1),
                EtherType::IPV6,
                HEADER_LEN + 28,
            ),
            (
                "a descriptor that declares one protocol and carries another",
                frame(PEER_MAC, DEVICE_MAC, EtherType::IPV4, 28, 1),
                EtherType::ARP,
                HEADER_LEN + 28,
            ),
            (
                "a length no frame can have",
                frame(PEER_MAC, DEVICE_MAC, EtherType::ARP, 28, 1),
                EtherType::ARP,
                HEADER_LEN - 1,
            ),
            (
                "a length past the slot",
                frame(PEER_MAC, DEVICE_MAC, EtherType::ARP, 28, 1),
                EtherType::ARP,
                FRAME_MAX + 1,
            ),
        ];
        for (why, bytes, declared, len) in cases {
            let mut svc = service(FakeHost::up(), 2);
            let (id, mut client) =
                open(&mut svc, 0, FrameFilter::protocol(EtherType::ARP)).expect("open");
            assert!(client.send_declaring(&bytes, declared, len));
            let report = svc.drive();
            assert_eq!(report.tx_submitted, 0, "{why}");
            assert_eq!(report.tx_refused, 1, "{why}");
            assert!(svc.host_mut().nic().posted.is_empty(), "{why}");
            assert_eq!(svc.counters(id).unwrap().tx_refused, 1, "{why}");
            // And the refusal did not wedge the vif: the next lawful frame goes.
            assert!(client.send(&frame(PEER_MAC, DEVICE_MAC, EtherType::ARP, 28, 9)));
            assert_eq!(svc.drive().tx_submitted, 1, "{why}");
        }
    }

    /// **The transmit gate asks the flow half.** A vif granted one
    /// connection cannot emit a datagram belonging to another.
    ///
    /// For a long time it could: `accepts_flow` had one caller,
    /// `accepts_frame`, reached only from `classify` — the receive path — so
    /// a flow capability bounded what a holder heard and not what it said.
    /// No race was involved; the question was simply never asked. That left
    /// the isolation a minted flow exists to provide as half a property, and
    /// the two-task witness it is for unearnable.
    ///
    /// Both frames below satisfy every dimension the gate checked before:
    /// a sane length, an in-range slot, an EtherType the vif's filter
    /// accepts and that matches the bytes, and this vif's own source MAC.
    /// The flow is the only thing separating them.
    #[test]
    fn the_transmit_gate_refuses_a_datagram_outside_the_vifs_flow() {
        use kore_driver::net::flow::{FlowMatch, IpProto, Ipv4Addr};

        const LOCAL: Ipv4Addr = Ipv4Addr([10, 0, 2, 15]);
        const ALICE: Ipv4Addr = Ipv4Addr([10, 0, 2, 2]);
        const BOB: Ipv4Addr = Ipv4Addr([10, 0, 2, 3]);

        let mut svc = service(FakeHost::up(), 2);
        let mut mine = FrameFilter::protocol(EtherType::IPV4);
        mine.flow = Some(FlowMatch::connection(
            IpProto::UDP,
            (LOCAL, 5000),
            (ALICE, 40000),
        ));
        let (id, mut client) = open(&mut svc, 0, mine).expect("open");

        let outbound = |src_port, dst_ip, dst_port| {
            let mut f = udp_frame(LOCAL, dst_ip, src_port, dst_port);
            FrameHead {
                dst: PEER_MAC,
                src: DEVICE_MAC,
                ethertype: EtherType::IPV4,
            }
            .write(&mut f)
            .expect("header fits");
            f
        };

        // Outside the grant: a peer this authority was never minted for,
        // from a local port it never bound. Sent first, so the refusal is
        // at the head of the ring and is accounted to the vif.
        assert!(client.send(&outbound(31337, BOB, 9)));
        let report = svc.drive();
        assert_eq!(report.tx_submitted, 0, "another conversation does not go");
        assert_eq!(report.tx_refused, 1);
        assert_eq!(svc.counters(id).unwrap().tx_refused, 1);
        assert!(svc.host_mut().nic().posted.is_empty());

        // Inside it: the conversation this vif was minted for, and the
        // refusal did not wedge the ring.
        assert!(client.send(&outbound(5000, ALICE, 40000)));
        assert_eq!(svc.drive().tx_submitted, 1, "its own flow still goes");
    }

    /// A backend that will not take a frame costs nothing: the descriptor is
    /// still there next sweep, and the mapping did not count an outstanding
    /// transfer that never happened.
    #[test]
    fn a_refused_submit_leaves_the_descriptor_where_it_was() {
        let mut svc = service(FakeHost::up(), 2);
        let (_id, mut client) =
            open(&mut svc, 0, FrameFilter::protocol(EtherType::ARP)).expect("open");
        assert!(client.send(&frame(PEER_MAC, DEVICE_MAC, EtherType::ARP, 28, 1)));
        svc.host_mut().nic().busy = true;
        assert_eq!(svc.drive().tx_submitted, 0);
        svc.host_mut().nic().busy = false;
        assert_eq!(svc.drive().tx_submitted, 1, "retried, not lost");
    }

    // ── teardown ───────────────────────────────────────────────────────

    /// Close is two-phase, and the two phases are deliberately unequal: the
    /// identity dies immediately, the mapping only once the controller is
    /// done with it.
    #[test]
    fn close_kills_the_identity_at_once_and_the_mapping_when_it_can() {
        let mut svc = service(FakeHost::up(), 2);
        let (id, mut client) =
            open(&mut svc, 0, FrameFilter::protocol(EtherType::ARP)).expect("open");
        assert!(client.send(&frame(PEER_MAC, DEVICE_MAC, EtherType::ARP, 28, 1)));
        svc.drive();
        assert_eq!(svc.host_mut().nic().posted.len(), 1, "in flight");

        assert_eq!(svc.close_vif(0, id), Ok(id));
        assert_eq!(
            svc.stats_arm(0, id),
            Err(NetRefusal::StaleVif),
            "the security half does not wait"
        );
        assert_eq!(svc.closing_vifs(), 1, "the mapping is parked, not retired");

        svc.host_mut().nic().complete_all();
        svc.drive();
        assert_eq!(svc.closing_vifs(), 0, "drained, then retired");
        assert_eq!(
            svc.orphan_completions(),
            0,
            "the late completion found its parked mapping, not a stranger's ring"
        );
    }

    /// A completion for a vif that was closed **and its slot reused** is
    /// caught by the generation in its own cookie, so it credits neither the
    /// new tenant's ring nor nothing at all.
    #[test]
    fn a_late_completion_never_credits_the_new_tenant() {
        let mut svc = service(FakeHost::up(), 4);
        let (first, mut client) =
            open(&mut svc, 0, FrameFilter::protocol(EtherType::ARP)).expect("open");
        assert!(client.send(&frame(PEER_MAC, DEVICE_MAC, EtherType::ARP, 28, 1)));
        svc.drive();
        assert_eq!(svc.close_vif(0, first), Ok(first));

        let (second, mut next_client) =
            open(&mut svc, 0, FrameFilter::protocol(EtherType::ARP)).expect("reopen");
        assert_eq!(second.index, first.index);
        assert!(next_client.send(&frame(PEER_MAC, DEVICE_MAC, EtherType::ARP, 28, 2)));

        // The first vif's in-flight frame completes *after* its slot was
        // reused. Its cookie carries the dead generation, so it credits the
        // parked mapping and not the new tenant sitting in the same slot.
        svc.host_mut().nic().complete_all();
        svc.drive();
        assert_eq!(
            svc.counters(second).unwrap().tx_frames,
            0,
            "the late completion belonged to the vif that is gone"
        );
        assert_eq!(
            svc.closing_vifs(),
            0,
            "and drained the mapping it did belong to"
        );

        // The new tenant's own frame — submitted by the sweep above —
        // completes normally.
        svc.host_mut().nic().complete_all();
        svc.drive();
        assert_eq!(svc.counters(second).unwrap().tx_frames, 1);
        assert_eq!(
            svc.orphan_completions(),
            0,
            "neither completion went unattributed"
        );
    }

    // ── the window ─────────────────────────────────────────────────────

    /// A window that is not the ring pair the protocol says it is gets no
    /// vif — the attach is the gate, so a malformed or hostile window is
    /// refused rather than trusted.
    #[test]
    fn an_unformatted_window_is_refused_at_attach() {
        let mut svc = service(FakeHost::up(), 2);
        let refusal = svc.open_vif(0, FrameFilter::protocol(EtherType::ARP), |_, _| {
            let layout = std::alloc::Layout::from_size_align(WINDOW_BYTES, 64).unwrap();
            // SAFETY: non-zero size; leaked deliberately, and never formatted
            // — which is the point of this case.
            let base = unsafe { std::alloc::alloc_zeroed(layout) };
            // SAFETY: `base` addresses `WINDOW_BYTES` zeroed bytes.
            Ok(AdoptedWindow {
                bytes: unsafe { slice::from_raw_parts_mut(base, WINDOW_BYTES) },
                iova: IovaAddr(0),
                lease: (),
            })
        });
        assert_eq!(refusal, Err(NetRefusal::BadWindow));
        assert_eq!(svc.open_vifs(), 0);

        // And a window too small to hold the layout.
        let refusal = svc.open_vif(0, FrameFilter::protocol(EtherType::ARP), |_, _| {
            let layout = std::alloc::Layout::from_size_align(64, 64).unwrap();
            // SAFETY: non-zero size, leaked.
            let base = unsafe { std::alloc::alloc_zeroed(layout) };
            // SAFETY: `base` addresses 64 zeroed bytes.
            Ok(AdoptedWindow {
                bytes: unsafe { slice::from_raw_parts_mut(base, 64) },
                iova: IovaAddr(0),
                lease: (),
            })
        });
        assert_eq!(refusal, Err(NetRefusal::BadWindow));
    }

    /// Fill the plan and return every address it proposed, sorted.
    fn exhaust_the_address_plan(svc: &mut NetService<FakeHost>) -> Vec<u64> {
        let mut bases = Vec::new();
        for _ in 0..MAX_VIFS {
            let (_id, client) = open(svc, 0, FrameFilter::protocol(EtherType::ARP)).expect("open");
            let (iova, _) = *svc.host_mut().nic().windows.last().expect("registered");
            bases.push(iova);
            core::mem::forget(client);
        }
        bases.sort_unstable();
        bases
    }

    /// The device address plan: one stride per vif, never overlapping each
    /// other and never reaching the driver's own DMA pool window.
    ///
    /// Asserted against the placement the *host's* device facts produce
    /// rather than against a constant — a constant is what let the plan
    /// drift out of a narrow device's reach in the first place.
    #[test]
    fn vif_windows_never_overlap_each_other_or_the_pool() {
        for host in [FakeHost::up(), FakeHost::narrow()] {
            let bits = host.dma_address_bits();
            let pool_bytes = host.dma_pool_bytes();
            let mut svc = service(host, MAX_VIFS as u8);
            let bases = exhaust_the_address_plan(&mut svc);
            assert_eq!(
                open(&mut svc, 0, FrameFilter::protocol(EtherType::ARP)).unwrap_err(),
                NetRefusal::PolicyVifs
            );
            for pair in bases.windows(2) {
                assert!(
                    pair[1] - pair[0] >= WINDOW_BYTES as u64,
                    "{bits}-bit: adjacent vif windows overlap"
                );
            }
            if pool_bytes > 0 {
                let pool_base =
                    kore_driver::host::dma_iova_base(bits, pool_bytes).expect("a pool placement");
                let last = bases[MAX_VIFS - 1] + WINDOW_BYTES as u64 - 1;
                assert!(
                    bases[0] >= pool_base + pool_bytes || last < pool_base,
                    "{bits}-bit: the vif arena [{:#x}, {last:#x}] runs through the \
                     driver's own pool window [{pool_base:#x}, {:#x})",
                    bases[0],
                    pool_base + pool_bytes
                );
            }
        }
    }

    /// **Every address proposed to a 32-bit master is one it can emit.**
    ///
    /// The MT7925 carries a 32-bit buffer pointer in its WFDMA descriptor
    /// and enables no address extension (`res/MT7925_EVIDENCE.md` §15), so
    /// an unreachable proposal is not refused by the part — it is truncated,
    /// and the controller fetches whatever lives at the low 32 bits.
    ///
    /// The plan used to be the constant 8 GiB for every device, which is
    /// four gibibytes past this ceiling for *every* vif. Checked to the
    /// **last** vif's last byte: an arena whose base is reachable and whose
    /// end is not is the same corruption one window later.
    #[test]
    fn a_narrow_masters_vif_windows_are_all_reachable() {
        let host = FakeHost::narrow();
        let bits = host.dma_address_bits();
        let ceiling = kore_driver::host::dma_iova_ceiling(bits);
        let mut svc = service(host, MAX_VIFS as u8);
        let bases = exhaust_the_address_plan(&mut svc);
        for (ordinal, base) in bases.iter().enumerate() {
            let last = base + WINDOW_BYTES as u64 - 1;
            assert!(
                last <= ceiling,
                "vif {ordinal} ends at {last:#x}, past a {bits}-bit master's \
                 ceiling {ceiling:#x} — the part truncates rather than refusing"
            );
        }
    }

    /// A device whose master can reach no arena this host places is refused
    /// **before** anything is spent — no ordinal, no window, no device leaf.
    #[test]
    fn an_unplaceable_master_is_refused_rather_than_truncated() {
        let mut host = FakeHost::up();
        // 30 bits is 1 GiB, which is exactly where a narrow pool starts:
        // `dma_iova_base` already refuses it, and so must the vif plan.
        host.dma_address_bits = 30;
        host.dma_pool_bytes = 0x20_0000;
        let mut svc = service(host, MAX_VIFS as u8);
        assert_eq!(
            open(&mut svc, 0, FrameFilter::protocol(EtherType::ARP)).unwrap_err(),
            NetRefusal::NoMapping
        );
        assert!(
            svc.ordinals.iter().all(|spent| !spent),
            "a refused placement must not have spent an ordinal"
        );
    }

    // ── device posture ─────────────────────────────────────────────────

    /// The device receive mode follows the vif set, because CDC Ethernet
    /// makes it a device property: per-vif addresses are unreachable unless
    /// the device is admitting frames not addressed to its own station.
    #[test]
    fn the_packet_filter_follows_the_vif_set() {
        let mut svc = service(FakeHost::up(), 4);
        let (id, client) = open(&mut svc, 0, FrameFilter::protocol(EtherType::ARP)).expect("open");
        core::mem::forget(client);
        assert_eq!(
            svc.host_mut().nic().filters.last(),
            Some(&PacketFilter::DIRECTED),
            "one station, its own address plus broadcast"
        );

        svc.set_address_mode(AddressMode::PerVif);
        let (_second, client) =
            open(&mut svc, 0, FrameFilter::protocol(EtherType::IPV4)).expect("open");
        core::mem::forget(client);
        assert_eq!(
            svc.host_mut().nic().filters.last(),
            Some(&PacketFilter::PROMISCUOUS),
            "derived addresses need a device that will receive for them"
        );
        assert_ne!(
            svc.mac_of(_second).unwrap(),
            svc.mac_of(id).unwrap(),
            "and each vif answers to its own address"
        );
        assert!(svc.mac_of(_second).unwrap().is_local());
    }

    /// The default address mode is the one that works everywhere: vifs share
    /// the device's address and separate by protocol alone.
    #[test]
    fn shared_addressing_is_the_default() {
        let mut svc = service(FakeHost::up(), 4);
        let (a, client) = open(&mut svc, 0, FrameFilter::protocol(EtherType::ARP)).expect("open");
        core::mem::forget(client);
        let (b, client) = open(&mut svc, 0, FrameFilter::protocol(EtherType::IPV4)).expect("open");
        core::mem::forget(client);
        assert_eq!(svc.mac_of(a).unwrap(), DEVICE_MAC);
        assert_eq!(svc.mac_of(b).unwrap(), DEVICE_MAC);
    }

    /// **The link edge is derived from the level, once per move.** A link
    /// that was up at founding is not announced as having come up; a drop
    /// and a return are one edge each, however many sweeps sit between and
    /// however many causes the device raised; a sweep on an unchanged link
    /// reports nothing and stays idle.
    #[test]
    fn a_link_edge_is_derived_from_the_level_and_reported_once() {
        let mut svc = service(FakeHost::up(), 2);
        assert!(svc.drive().is_idle(), "up at founding: no edge to report");
        assert!(!svc.drive().link_changed);
        assert_eq!(svc.link_edges(), 0);

        svc.host_mut().nic().link = LinkState::Down;
        let report = svc.drive();
        assert!(report.link_changed);
        assert!(!report.is_idle(), "an edge is not an idle sweep");
        assert!(!svc.drive().link_changed, "the same level again is no edge");
        assert_eq!(svc.link_edges(), 1);

        svc.host_mut().nic().link = LinkState::Up { mbps: 100 };
        assert!(svc.drive().link_changed);
        svc.host_mut().nic().link = LinkState::Up { mbps: 1000 };
        assert!(
            svc.drive().link_changed,
            "a rate change moves the level too"
        );
        assert_eq!(svc.link_edges(), 3);
        assert_eq!(
            svc.info().unwrap().link_mbps,
            1000,
            "and Query reads the level"
        );
    }

    /// `Query` carries the device's counters as the backend reports them,
    /// beside the service's own dropped-notice count — the read-back the
    /// completion path is judged by.
    #[test]
    fn query_carries_the_devices_counters_live() {
        let mut svc = service(FakeHost::up(), 2);
        assert_eq!(svc.info().unwrap().counters, NicCounters::default());
        svc.host_mut().nic().counters = NicCounters {
            irqs: 41,
            link_changes: 2,
            overruns: 1,
            rx_errors: 3,
        };
        let info = svc.info().unwrap();
        assert_eq!(info.counters.irqs, 41);
        assert_eq!(info.counters.link_changes, 2);
        assert_eq!(info.counters.overruns, 1);
        assert_eq!(info.counters.rx_errors, 3);
        assert_eq!(info.dropped_notices, 0);
    }

    /// A machine with no adaptor still serves, and says why rather than
    /// pretending — the degraded-but-serving posture a driver row owes its
    /// boot chain.
    #[test]
    fn a_deviceless_row_serves_and_refuses_by_name() {
        let mut svc = service(FakeHost::degraded(), 2);
        assert!(svc.info().is_none());
        assert!(!svc.ready());
        assert_eq!(
            open(&mut svc, 0, FrameFilter::protocol(EtherType::ARP)),
            Err(NetRefusal::NotReady)
        );
        assert!(svc.drive().is_idle(), "and its sweep does nothing at all");
    }

    /// #388 D1: a bound radio lends no frame path, so `Query` refuses
    /// `NotReady` — and `Radio` still answers, because the facts a
    /// serial-less machine needs are exactly the ones a radio in that
    /// posture has. An Ethernet host has no radio and says so by name, and
    /// the report survives the wire byte-for-byte.
    #[test]
    fn a_radio_reports_its_facts_whether_or_not_it_lends_frames() {
        let report = RadioReport {
            cut: 1,
            country: *b"US",
            secure_boot: true,
            download_packets: 0x1234_5678,
            ..RadioReport::default()
        };
        let mut svc = service(FakeHost::radio_only(report), 2);
        assert!(svc.info().is_none(), "no backend, no link facts");
        let out = step(&mut svc, NetRequest::Query);
        assert_eq!(
            out.reply().copied(),
            Some(NetResponse::Refused(NetRefusal::NotReady))
        );
        let out = step(&mut svc, NetRequest::Radio);
        let reply = out.reply().copied().expect("a reply");
        assert_eq!(reply, NetResponse::Radio(report));
        assert_eq!(decode_rep(&encode_rep(&reply, 0)), Some((reply, 0)));

        let mut svc = service(FakeHost::up(), 2);
        let out = step(&mut svc, NetRequest::Radio);
        assert_eq!(
            out.reply().copied(),
            Some(NetResponse::Refused(NetRefusal::Unsupported)),
            "an Ethernet host has no radio to report"
        );
    }

    // ── the cookie ─────────────────────────────────────────────────────

    /// The transmit cookie carries the vif's whole identity, so completion
    /// routing and use-after-close protection are one mechanism.
    #[test]
    fn the_transmit_cookie_round_trips_identity_and_slot() {
        for index in 0..MAX_VIFS as u16 {
            for generation in [1u16, 2, 0x7fff, u16::MAX] {
                for slot in [0u32, 1, TX_DEPTH - 1] {
                    let id = VifId { index, generation };
                    let cookie = pack_cookie(id, slot);
                    assert_eq!(unpack_cookie(cookie), (id, slot));
                    assert_eq!(
                        cookie.0 & (1 << 63),
                        0,
                        "the top bit stays clear — backends tag their own \
                         receive cookies there"
                    );
                }
            }
        }
    }

    // ── the serve path ─────────────────────────────────────────────────

    struct NoCaps;

    impl Bundle for NoCaps {
        const PROFILE: Profile = Profile::new("net-test", &[]);
        fn resolve(_boot: &Bootstrap<'_>) -> Result<Self, BindError> {
            Ok(Self)
        }
    }

    impl SystemService for NetService<FakeHost> {
        const NAME: &'static str = "kore-svc-net";
        type Endowment = NoCaps;

        fn bind(_endowment: NoCaps) -> Self {
            service(FakeHost::up(), 2)
        }

        fn serve(&mut self, req: NetRequest, sv: &mut Serve<'_, NetResponse>) {
            self.serve_step(req, sv);
        }

        /// Mirrors `NetRow`'s override, so a test drives the path the
        /// founded row takes. Without it the mint verbs answer `None` on
        /// the synchronous default and every one is refused `Unsupported`
        /// — which is exactly how this was missed.
        async fn serve_async(&mut self, req: NetRequest, sv: &mut Serve<'_, NetResponse>) {
            self.serve_step_async(req, sv).await;
        }
    }

    /// **Every verb in the table reaches a row that answers it.**
    ///
    /// The one hole the row list cannot close by itself. A verb registers
    /// its code, its codec and its correlation in one `impl`, and all three
    /// are required; the serve arm is the one part with a default — "answers
    /// nothing" — because `services/inet`'s rows legitimately take it. A
    /// frame-row verb that took it by accident would decode fine and then
    /// fall through to `Unsupported` at run time. Here it falls through in a
    /// test instead.
    #[test]
    fn every_verb_in_the_table_reaches_a_row_that_answers() {
        // A host with both a frame path and a radio, so no verb is refused
        // `Unsupported` for the honest reason and the assertion stays sharp.
        let mut host = FakeHost::up();
        host.radio = Some(RadioReport::default());
        let mut svc = service(host, 2);
        let (vif, client) = open(&mut svc, 0, FrameFilter::protocol(EtherType::ARP)).expect("open");
        core::mem::forget(client);

        let filter = FrameFilter::protocol(EtherType::ARP);
        for req in [
            NetRequest::Query,
            NetRequest::OpenVif {
                filter,
                announced: SlotRef::from_wire(0),
            },
            NetRequest::SetFilter { vif, filter },
            NetRequest::Stats { vif },
            NetRequest::Signal,
            NetRequest::Radio,
            NetRequest::CloseVif { vif },
        ] {
            let out = step(&mut svc, req);
            let reply = out.reply().copied().expect("a reply");
            assert_ne!(
                reply,
                NetResponse::Refused(NetRefusal::Unsupported),
                "{req:?} reached no row in the verb table",
            );
        }
    }

    /// The verbs that need no window go through the real serve step, and each
    /// answer encodes to a reply the wire will decode back to itself.
    #[test]
    fn the_serve_step_answers_and_the_answers_survive_the_wire() {
        let mut svc = service(FakeHost::up(), 2);
        let (id, client) = open(&mut svc, 0, FrameFilter::protocol(EtherType::ARP)).expect("open");
        core::mem::forget(client);

        let out = step(&mut svc, NetRequest::Query);
        let NetResponse::Info(info) = out.reply().copied().expect("a reply") else {
            panic!("Query answers Info");
        };
        assert_eq!(info.mac, DEVICE_MAC);
        assert_eq!(info.window_bytes, WINDOW_BYTES as u32);
        assert_eq!(info.mtu as usize, FRAME_MAX);
        assert!(info.link_up);

        let out = step(&mut svc, NetRequest::Stats { vif: id });
        let reply = out.reply().copied().expect("a reply");
        assert!(matches!(reply, NetResponse::Stats { .. }));
        assert_eq!(decode_rep(&encode_rep(&reply, 0)), Some((reply, 0)));

        let out = step(&mut svc, NetRequest::CloseVif { vif: id });
        assert_eq!(out.reply().copied(), Some(NetResponse::Closed { vif: id }));

        let out = step(&mut svc, NetRequest::CloseVif { vif: id });
        let reply = out.reply().copied().expect("a reply");
        assert_eq!(reply, NetResponse::Refused(NetRefusal::StaleVif));
        assert_eq!(decode_rep(&encode_rep(&reply, 0)), Some((reply, 0)));
    }

    /// **The flow verbs answer on the founded row's own path.**
    ///
    /// Not through `step`. Three of them mint, which installs a filter,
    /// which is `async` at the seam — so they implement `ServeVerb` through
    /// `serve_async` and inherit the synchronous default, which answers
    /// `None`. On the sync step every one is refused `Unsupported` while
    /// decoding perfectly off the wire: present, reachable, dead. This
    /// drives `step_async`, which is what `NetRow` takes.
    ///
    /// And it asserts the install, not just the reply. A mint that answered
    /// `Flow` while leaving the vif's filter untouched would be the exact
    /// thing this whole move was for — a capability nothing enforces.
    #[test]
    fn a_minted_flow_answers_on_the_async_path_and_installs_its_filter() {
        use kore_driver::net::flow::{EndpointMatch, IpProto, Ipv4Addr};

        const ME: Ipv4Addr = Ipv4Addr([10, 0, 2, 15]);
        const GW: Ipv4Addr = Ipv4Addr([10, 0, 2, 2]);
        const PEER: Ipv4Addr = Ipv4Addr([10, 0, 2, 50]);

        let mut svc = service(FakeHost::up(), 2);
        let (vif, client) =
            open(&mut svc, 0, FrameFilter::protocol(EtherType::IPV4)).expect("open");
        core::mem::forget(client);

        // Before: the vif carries the L2-only filter it opened with.
        assert_eq!(svc.vifs.get(vif).expect("live").filter.flow, None);

        let out = run(step_async(
            &mut svc,
            NetRequest::Configure {
                addr: ME,
                prefix_len: 24,
                gateway: Some(GW),
            },
        ));
        let NetResponse::Identity { identity } = out.reply().copied().expect("a reply") else {
            panic!("Configure answers Identity, not {:?}", out.reply());
        };
        assert_eq!(identity.expect("configured").addr, ME);

        let out = run(step_async(
            &mut svc,
            NetRequest::OpenFlow {
                vif,
                proto: IpProto::UDP,
                local_port: Some(5000),
                peer: (PEER, 53),
            },
        ));
        let reply = out.reply().copied().expect("a reply");
        let NetResponse::Flow { proto, local, peer } = reply else {
            panic!("the mint was refused on the path the row actually takes: {reply:?}");
        };
        assert_eq!(proto, IpProto::UDP);
        assert_eq!(local, (ME, 5000));
        assert_eq!(peer, Some((PEER, 53)));
        assert_eq!(decode_rep(&encode_rep(&reply, 0)), Some((reply, 0)));

        // After: the authority is on the vif, and it is the flow that was
        // minted — read off the shipping filter, not off the reply.
        let installed = svc
            .vifs
            .get(vif)
            .expect("live")
            .filter
            .flow
            .expect("the mint installed nothing — the capability would enforce nothing");
        assert_eq!(
            installed.local,
            EndpointMatch::AddrPort {
                addr: ME,
                port: 5000
            }
        );
        assert_eq!(
            installed.peer,
            EndpointMatch::AddrPort {
                addr: PEER,
                port: 53
            }
        );
        let neighbor = svc.vifs.get(vif).unwrap().neighbor.as_ref().unwrap();
        assert_eq!((neighbor.local, neighbor.next_hop), (ME.0, PEER.0));
        svc.configure(Identity::new(GW, 24, None).unwrap());
        assert!(svc.vifs.get(vif).unwrap().neighbor.is_none());
    }

    /// A close gives the ports back and shuts the filter, on the same path.
    #[test]
    fn closing_a_flow_releases_its_port_on_the_async_path() {
        use kore_driver::net::flow::{IpProto, Ipv4Addr};

        const ME: Ipv4Addr = Ipv4Addr([10, 0, 2, 15]);

        let mut svc = service(FakeHost::up(), 2);
        let (vif, client) =
            open(&mut svc, 0, FrameFilter::protocol(EtherType::IPV4)).expect("open");
        core::mem::forget(client);
        run(step_async(
            &mut svc,
            NetRequest::Configure {
                addr: ME,
                prefix_len: 24,
                gateway: None,
            },
        ));
        run(step_async(
            &mut svc,
            NetRequest::OpenListener {
                vif,
                proto: IpProto::TCP,
                local_port: 8080,
            },
        ));

        let out = run(step_async(&mut svc, NetRequest::CloseFlow { vif }));
        assert_eq!(
            out.reply().copied(),
            Some(NetResponse::FlowClosed { released: 1 }),
            "one port held, one port back"
        );
    }

    /// `Signal` is the one verb that moves authority, and a service with no
    /// channel space refuses it by name rather than replying with a grant it
    /// cannot make.
    #[test]
    fn a_channelless_service_refuses_the_doorbell() {
        let mut svc = service(FakeHost::up(), 2);
        let out = step(&mut svc, NetRequest::Signal);
        assert_eq!(
            out.reply().copied(),
            Some(NetResponse::Refused(NetRefusal::NoChannel))
        );
        assert_eq!(out.granted(), 0, "and emits nothing");
    }

    /// Every refusal code round-trips, so a client's decode of a refusal never
    /// silently becomes a different one.
    #[test]
    fn every_refusal_names_itself_on_the_wire() {
        for refusal in [
            NetRefusal::Unsupported,
            NetRefusal::NotReady,
            NetRefusal::PolicyEtherType,
            NetRefusal::PolicyMulticast,
            NetRefusal::PolicyBroadcast,
            NetRefusal::PolicyPromiscuous,
            NetRefusal::PolicyVifs,
            NetRefusal::NoVif,
            NetRefusal::StaleVif,
            NetRefusal::NotOwner,
            NetRefusal::BadWindow,
            NetRefusal::Exhausted,
            NetRefusal::Busy,
            NetRefusal::NoMapping,
            NetRefusal::NoChannel,
            NetRefusal::Internal,
        ] {
            assert_eq!(NetRefusal::from_code(refusal.code()), Some(refusal));
        }
        assert_eq!(NetRefusal::from_code(0), None, "0 is not a refusal");
        assert_eq!(NetRefusal::from_code(200), None);
        assert_eq!(NetService::<FakeHost>::REPLY_GRANTS, NET_REPLY_GRANTS);
    }
}
