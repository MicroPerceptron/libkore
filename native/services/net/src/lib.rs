//! # kore-svc-net — the L2 frame service
//!
//! The design of record is [`res/NETWORK_SERVICE.md`]: the vif model, the
//! frozen window and ring contract, the backend seam, and — §7 — an honest
//! account of what is proven and what is not.
//!
//! **A vif is not a socket.** A client of this service does not open a
//! connection and does not name an address family; it opens a *virtual
//! interface* — a policy-approved frame filter, a queue pair in its own
//! shared window, an assigned station address, and a generation — and then
//! never calls this service again to move a frame. Frames ride the vif's own
//! rings. The control plane grants, revokes, and accounts; it is not in the
//! datapath.
//!
//! That shape is chosen against the two open network issues rather than
//! inherited from Linux. #387 needs a per-client queue-pair capability over
//! raw frames; #388's whole feature is one driver task publishing several
//! class faces such that a peer-side client provably cannot reach the infra
//! interface's ring. Linux is the opposite arrangement — one `net_device`
//! producing one frame stream, demultiplexed *above* the driver by the socket
//! layer, with isolation between mutually-distrusting users bolted on
//! afterwards (macvlan, SR-IOV functions, `AF_PACKET`). Landing the first NIC
//! in that shape would make #388's isolation requirement a rewrite instead of
//! a configuration.
//!
//! ## What holds the isolation up
//!
//! - **Separate windows.** Each vif's rings and frame slots live in a
//!   *different* client-supplied window. A client holds no handle to another
//!   vif's ring, so there is no check to forget.
//! - **A classifier, not a tap.** A received frame is copied into a vif only
//!   if that vif's granted filter accepts it. The copy is the boundary: USB
//!   networking has no receive steering, so handing one client's buffer to
//!   the shared bulk pipe would let it observe everyone's frames.
//! - **Authority from the wiring.** A vif's filter is the intersection of
//!   what the client asked for with the [`VifPolicy`](kore_driver::net::VifPolicy)
//!   its *arrival pair* carries. The pair identity is stamped by the serve
//!   host from the pair's own endpoint, so it is not a field a client could
//!   forge — which is what makes a widened filter unforgeable rather than
//!   merely refused.
//! - **Generations.** Closing a vif bumps its generation, so an identifier
//!   held across a close stops resolving instead of addressing the slot's
//!   next tenant — and a transmit completion that lands after the close is
//!   caught by the same check.
//!
//! ## The device is a vehicle
//!
//! The first backend is CDC-ECM USB Ethernet over the existing xHCI stack
//! ([`kore_usbnet`]), because that is the hardware the metal targets actually
//! have and because CDC-ECM is a standard rather than device-specific glue.
//! Nothing above [`NicBackend`](kore_driver::net::NicBackend) knows that:
//! a PCI NIC is a second implementation of the same seam, not a second
//! service.
//!
//! [`res/NETWORK_SERVICE.md`]: https://github.com/MicroPerceptron/kore/blob/main/res/NETWORK_SERVICE.md

#![no_std]
#![deny(missing_docs)]

#[cfg(test)]
extern crate std;

pub mod arp;
pub mod client;
#[cfg(all(target_arch = "x86_64", target_os = "none"))]
pub mod entry;
pub mod metal;
/// The network this image is built to join — see [`network`].
pub mod network;
pub mod service;
pub mod verbs;
pub mod wire;

pub use client::{NetClient, RingNet};
#[cfg(all(target_arch = "x86_64", target_os = "none"))]
pub use entry::run;
pub use service::{
    AddressMode, AdoptedWindow, DriveReport, NetHost, NetInfo, NetRefusal, NetRequest, NetResponse,
    NetService, NetStats, RadioFirmware, RadioPhy, RadioReport, ShareFault, VifCounters,
};
pub use verbs::NetVerbs;
pub use wire::{
    NET_EXEC_PAIR, NET_REPLY_GRANTS, NET_SERVE_PAIRS, NetWire, decode_radio, encode_radio,
};

/// Architecture-neutral service entry selected at this crate boundary.
///
/// x86 composes the claim-and-serve ladder; other targets keep the same
/// founded control plane and report its honest unavailable posture. Clients
/// and crt0 do not branch on which one they got.
#[cfg(not(all(target_arch = "x86_64", target_os = "none")))]
pub async fn run(rt: kore_rt::entry::Rt) -> ! {
    kore_rt::entry::serve::<metal::NetRow>(rt).await
}

/// The class vocabulary this service is written against, re-exported so a
/// client names one crate rather than three.
pub use kore_driver::net::{
    EtherType, EtherTypeMatch, FrameDesc, FrameFilter, LinkState, MacAddr, MulticastMatch,
    NicCounters, VifId, VifPolicy, window,
};

/// Notice cookie: one or more frames landed in a vif's receive ring.
///
/// The notice carries the serving pair, not a count — the ring is the truth
/// and the doorbell only wakes, so a dropped notice costs a wake and never a
/// frame.
pub const COOKIE_NET_RX: u64 = 0x6B72_5F6E_6574_5258; // "kr_netRX"

/// Notice cookie: the link's state moved (carrier, or rate) since the last
/// sweep.
///
/// The notice carries the serving pair, not the state — the level rides
/// `Query`, so a client that hears this asks once and a client that missed
/// it reads the same answer there. Sent to every pair holding a doorbell,
/// because a link edge is everyone's fact.
pub const COOKIE_NET_LINK: u64 = 0x6B72_5F6E_6574_4C4B; // "kr_netLK"

/// The readiness **note** map — the payload of this row's `COOKIE_SERVING`
/// beacon, printed by the founder as `note: rung N` beside the ready line.
///
/// A note is a *degraded-but-serving* report, never a failure; `0` means the
/// frame service is up: a controller claimed, an adaptor found and brought up,
/// and the founding self-test passed. Rungs 1–5 share the claimant numbering
/// every driver row uses (one decoder ring); 11+ are this row's own; 16+`r`
/// wraps the USB core's own rung and 32+`r` the PCI core's, so a serial-less
/// photo of `note: rung 18` reads directly as "the bring-up ladder refused at
/// the MAC string" and `note: rung 34` as "the 82574L's reset never cleared".
/// Renumber nothing; append only.
pub mod note {
    /// The machine describes no matching controller — or the input router
    /// already took the only one. Ordinary on a single-xHC box.
    pub const NO_MATCH: u64 = 1;
    /// Every matching function is already claimed.
    pub const TAKEN: u64 = 2;
    /// The broker (or the claim transport) refused — infrastructure.
    pub const DENIED: u64 = 3;
    /// The claimed function's registers do not decode.
    pub const NO_CANDIDATE: u64 = 4;
    /// The claim realized no resources, so the row serves without retrying.
    pub const BIND_REFUSED: u64 = 5;
    /// A controller came up but **no Ethernet adaptor is plugged into it**.
    ///
    /// The ordinary state of a machine whose dongle is in the other
    /// controller's port, and the one this row expects an operator to fix by
    /// moving a cable rather than by rebuilding anything.
    ///
    /// Strictly "no port here carries an Ethernet function". A port that
    /// carries one which then refused its bring-up ladder reports
    /// [`BRING_UP_BASE`]`+ r` instead — the two used to be indistinguishable,
    /// and conflating them tells an operator to move a cable that is already
    /// in the right port.
    pub const NO_ADAPTOR: u64 = 11;
    /// An adaptor came up but the founding self-test transmitted and received
    /// nothing within its budget: the row serves, and frames may not move.
    /// The fact a serial-less machine can only learn here.
    pub const NO_FRAME_PATH: u64 = 12;
    /// The self-test could not open its own vif — a window carve or a ring
    /// attach refused. The datapath is unproven, not known broken.
    pub const NO_SELF_VIF: u64 = 13;
    /// The USB bring-up ladder refused: `BRING_UP_BASE + r` for core rung `r`
    /// ([`metal::usbnet_rung`](crate::metal::usbnet_rung)).
    ///
    /// So 17 is the descriptor probe, 18 a refused USB transfer, 19 the MAC
    /// string, 20 a DMA allocation, 21 a bounded resource. The port rides the
    /// detail bits like every other note here, so a photo of `note: rung 19
    /// (detail 0x3)` reads as "the adaptor on root port 3 has a MAC string
    /// that does not decode" with no debugger and no serial cable.
    pub const BRING_UP_BASE: u64 = 16;
    /// The PCI bring-up ladder refused: `PCI_BRING_UP_BASE + r` for
    /// `kore-e1000e` rung `r` (`kore_e1000e::Error::rung`, carried out of the
    /// intake as `E1000eNetBinding::refusal_code`).
    ///
    /// So 33 is a function that does not decode, 34 a reset that never
    /// self-cleared, 35 an NVM that never loaded, 36 no valid station
    /// address, 37 a DMA allocation, 38 a full transmit ring. Above the USB
    /// band with room for that ladder to grow to fifteen rungs; no detail
    /// bits, because a PCI function has no sub-address to report.
    pub const PCI_BRING_UP_BASE: u64 = 32;
    /// The Wi-Fi bring-up ladder refused: `WIFI_BRING_UP_BASE + r` for
    /// `kore-rtw89` rung `r` (`kore_rtw89::Error::rung`, carried out of the
    /// intake as `Rtw89NetBinding::refusal_code`).
    ///
    /// So 49 is a BAR2 that does not decode, 50 no firmware module in the
    /// boot, 51 a module that is not the pinned blob, 52 a power sequence
    /// that did not complete, 53 an efuse that did not read, 54 a DMA
    /// allocation, 55 an engine pre-init that did not complete, 56 a
    /// download that did not complete, 57 a firmware that never answered
    /// its first command. Above the PCI band with room for that ladder to
    /// grow to fifteen rungs, and the whole band below 64 so it never
    /// reaches the note's detail bits.
    pub const WIFI_BRING_UP_BASE: u64 = 48;
    /// The MT7925 bring-up ladder refused: `MT7925_BRING_UP_BASE + r` for
    /// `kore-mt7925` rung `r` (`kore_mt7925::Error::rung`, carried out of
    /// the intake as `Mt7925NetBinding::refusal_code`).
    ///
    /// So 65 is a BAR0 that does not decode, 66 an identity register that
    /// answered with something other than `0x7925`, 67 a part that never
    /// released ownership inside the vendor's own 50 ms.
    ///
    /// **Its own band, not a share of the Wi-Fi one.** The RTL8922AE's
    /// ladder already reaches 60 and the two parts' rungs mean entirely
    /// different things, so a photograph of `note: rung 50` must not read as
    /// one part's efuse on one machine and another's identity on another.
    ///
    /// Above 64, which the bands before it stayed below. That convention was
    /// caution rather than a limit: the rung occupies the note's **low
    /// byte** (`rung & 0xff` in `kore_rt::service`, detail shifted up by 8),
    /// so the field runs to 255 and this band is nowhere near the detail
    /// bits. The testimony carve grows downward from bit 63 and its lowest
    /// member is bit 54, which is further still.
    pub const MT7925_BRING_UP_BASE: u64 = 64;

    /// Where [`mt7925`]'s evidence word sits in the note.
    ///
    /// The rung owns the low byte and the detail the one above it. The
    /// testimony carve grows downward from bit 63 and its lowest member is
    /// bit 48 ([`NET_MT7925_PATCH_RESIDENT`]), so bits 16..=47 are the
    /// whole of the room a note has, and this word takes it.
    ///
    /// [`NET_MT7925_PATCH_RESIDENT`]: kore_abi::founding::NET_MT7925_PATCH_RESIDENT
    pub const MT7925_EVIDENCE_SHIFT: u32 = 16;
    /// How wide that word is — four bytes, each one number.
    ///
    /// Byte-aligned on purpose. On a machine with no serial port the note
    /// is read **off a photograph of a terminal**, and a reader who can
    /// split a hex word into bytes by eye can read the evidence without
    /// the decoder. A packed bitfield would be denser and unreadable.
    pub const MT7925_EVIDENCE_BITS: u32 = 32;

    /// **The note a bound MT7925 publishes**, from what its ladder reached.
    ///
    /// A free function, and tested, because the first version of this
    /// reporting was a `NicBound::founding_note` override — which
    /// `MetalHost::founding_note` never calls for a radio, since it reads
    /// `self.frames` and an MT7925 binds into `self.radio`. It was dead
    /// code with passing tests beside it, and it took a boot to find,
    /// because the composition it belonged to was inline in an `async fn`
    /// that nothing could call.
    ///
    /// Zero whenever there is nothing to say: no part, a part that
    /// answered, or a rung that was never attempted. A row with nothing to
    /// report must read exactly as it did before.
    ///
    /// `evidence` is the rung's own numbers, already packed by
    /// [`crate::metal::Mt7925NetBound::bring_up_evidence`]; zero when the
    /// rung has none.
    pub const fn mt7925(d0: bool, rung: u8, detail: u8, evidence: u32) -> u64 {
        // **`rung == 0` is the whole condition.** There used to be an
        // `mcu_answered` term here as well, meaning "D1c succeeded, so
        // there is nothing to report". That was true while D1c was the
        // only rung that could refuse, and became false the moment a
        // later rung could fail *after* the MCU answered: every C2b
        // refusal was discarded and the row printed `up` with no note.
        //
        // A rung of zero already means nothing refused. Any other value
        // is a refusal that happened, whatever succeeded before it.
        if !d0 || rung == 0 {
            return 0;
        }
        // The rung in the low byte, one byte of evidence above it — the
        // layout `kore_rt::service` fixes and the compositor's rung 4
        // already uses.
        (MT7925_BRING_UP_BASE + rung as u64)
            | ((detail as u64) << 8)
            | ((evidence as u64) << MT7925_EVIDENCE_SHIFT)
    }
}

/// **Pair 0's authority** — what the row's own client face may ask for.
///
/// Wide enough for an infrastructure vif (ARP, IPv4, IPv6) plus the peer link,
/// and no wider: no promiscuous receive, no caller-chosen source address, and
/// a bounded vif count. This is the whole of what a client of this row can
/// ever obtain, and it is a fact about the wiring — a request naming anything
/// outside it is refused by name, not narrowed.
///
/// A second face, when a minted one exists, gets its own entry; #387's peer
/// service would name `KORE_PEER` alone and hold exactly one vif.
pub fn pair_policy() -> VifPolicy {
    VifPolicy {
        // No L3 restriction: this authority predates flows and says
        // nothing about them.
        flow: None,
        ethertype: EtherTypeMatch::only(&[
            EtherType::ARP,
            EtherType::IPV4,
            EtherType::IPV6,
            EtherType::KORE_PEER,
        ])
        .expect("four is the bound"),
        multicast: MulticastMatch::None,
        broadcast: true,
        promiscuous: false,
        source_mac: false,
        max_vifs: 4,
    }
}

/// **Exec's pair's authority** ([`NET_EXEC_PAIR`], the row's final pair):
/// nothing. The resident exec manager issues only the read-only `Query` and
/// `Radio` verbs at a `PROFILE_NETWORK` launch, neither of which consults the
/// policy, so the pair carries [`VifPolicy::DENY`] and an `OpenVif` over it is
/// refused `TooManyVifs` before a filter is read — a launcher that answers
/// every task's spawn must not also be a station on the wire.
pub fn exec_pair_policy() -> VifPolicy {
    VifPolicy::DENY
}
