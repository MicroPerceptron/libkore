//! Service-protocol wire — the frozen request/response forms every
//! `kore-svc-*` speaks across a process boundary (`res/USERLAND.md`
//! §5.3; Phase S, #311).
//!
//! An in-memory enum cannot cross a process boundary; **the wire form is
//! the contract, the enum the convenience**.  Each service keeps its
//! `Request`/`Response` enums in its own crate and carries the
//! enum ⇄ wire codecs in its `wire.rs`; the zerocopy structs themselves
//! live here, beside every other frozen layout, so the two sides of a
//! ring can never drift (the `krlib::image` precedent — kore-abi's
//! charter is "the frozen wire contracts — kernel⇄user *and service
//! protocol*").
//!
//! ## The request path these ride (§5.3)
//!
//! Each client↔service pair shares one mm-leased window holding a
//! request `ring::spsc<*ReqWire>` and a reply `ring::spsc<*RepWire>`;
//! the channel pair beside it carries only notices ("look at your
//! ring") and capability transfers (grant-bearing replies).  **Payloads
//! ride rings; authority rides rendezvous; nothing else exists** — so
//! every field here is data: arena *slot indices* name capabilities the
//! peer already holds or was delivered (#148: no capability ever rides
//! a ring record), and grant-bearing replies say only *how many*
//! transfers to collect (`grants`) — the capabilities themselves land in
//! the client's pre-pinned reply-recv posts (§5.5).
//!
//! ## Layout doctrine
//!
//! The [`crate::task::BootBlock`] discipline throughout:
//! `#[repr(C)]`, little-endian, unaligned-safe, size-pinned, **no
//! parser** — every field admits any bit pattern ([`FromBytes`]), so a
//! record read off a shared ring (whose peer is untrusted) reinterprets
//! in place, and *validation is the codec's* (`wire.rs`): out-of-
//! vocabulary discriminants decode to `None`, fail-closed, never UB.
//! Post-freeze changes are additive only — new `OP_*`/`TAG_*` values,
//! meanings for reserved bytes; never a reorder, resize, or type change.
//!
//! Discriminants are **1-based** so an all-zero (unseeded) record is
//! invalid, not "the first verb" (the [`seg`](crate::image::seg)
//! precedent).

use zerocopy::byteorder::little_endian::{I32, U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use crate::image::{ImageName, ModuleDescriptor};

// ─── mm — the lessor (kore-svc-mm) ──────────────────────────────────

/// The memory-manager request wire form (`MmRequest`'s crossing shape).
/// `op` selects the verb; the other fields are per-verb (unused ones
/// ride zero).  32 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct MmReqWire {
    /// Verb discriminant: [`OP_ALLOC`](Self::OP_ALLOC) /
    /// [`OP_RELEASE`](Self::OP_RELEASE) / [`OP_SUPPLY`](Self::OP_SUPPLY) /
    /// [`OP_STATS`](Self::OP_STATS); zero (unseeded) is invalid.
    pub op: u8,
    /// `Alloc`: require physically-contiguous backing (`0`/`1`).
    pub contiguous: u8,
    /// `Alloc`: NUMA placement hint ([`crate::mem::NumaNode`]).
    pub numa: U16,
    /// `Release`: mm-arena slot holding the delivered extent record
    /// (#148 — the cap moved by rendezvous, the request names the slot).
    pub slot: U32,
    /// `Alloc`: requested size in bytes.
    pub size: U64,
    /// `Alloc`: required alignment in bytes (power of two).
    pub align: U64,
    /// `Alloc`: cache/access attribute bits ([`crate::mem::MemAttrs`]).
    pub attrs: U16,
    /// `Alloc`: DMA addressing class ([`crate::mem::DmaClass`] as `u8`).
    pub dma: u8,
    /// Reserved; must be zero.  Frozen additive headroom.
    pub reserved: [u8; 5],
}

impl MmReqWire {
    /// `MmRequest::Alloc` — grant a fresh extent satisfying the props.
    pub const OP_ALLOC: u8 = 1;
    /// `MmRequest::Release` — take back the extent delivered at `slot`.
    pub const OP_RELEASE: u8 = 2;
    /// `MmRequest::Supply` — mint a fresh **root supply lease** off mm's
    /// region endowment and deliver it whole (the child-memory verb a
    /// launcher conveys onward at spawn).  Reuses `size` and `align`;
    /// every other per-verb field must ride zero — a supply is CPU
    /// write-back R/W by nature, so `Alloc`'s placement vocabulary
    /// (NUMA, DMA class, contiguity, attrs) does not apply and a set bit
    /// refuses at decode.  The delivered lease's backing is zeroed by
    /// the kernel's lease-derive scrub invariant (fresh and reused
    /// frames alike), never by a wire choice.
    pub const OP_SUPPLY: u8 = 3;
    /// Read the lessor's bounded, provider-owned allocator snapshot.
    /// Every argument field must remain zero.
    pub const OP_STATS: u8 = 4;
    /// Exec's administration pair only: one component of a per-launch
    /// client face (a recyclable peer). `contiguous` carries the part code
    /// (`0` ring, `1` kick, `2` return receive, `0x80 | n` grant `n`),
    /// `slot` the announced return landing; every other field is zero.
    pub const OP_ADMIT_PEER: u8 = 5;
    /// Exec's administration pair only: consume the verdict of the
    /// preceding admission. Every argument field is zero.
    pub const OP_TAKE_ADMISSION: u8 = 6;
    /// Exec's administration pair only: retire one exact face generation.
    /// `numa` carries its arrival; every other field is zero.
    pub const OP_RETIRE_PEER: u8 = 7;
    /// Exec's administration pair only: fence retirement. Accepted only
    /// once the face is closed **and** every supply it minted has returned,
    /// so exec reclaims the face's local backing no earlier. `numa` carries
    /// its arrival; every other field is zero.
    pub const OP_CONFIRM_RETIRED: u8 = 8;
    /// A per-launch face retires itself. Every argument field is zero.
    pub const OP_CLOSE_PEER: u8 = 9;
}

/// The memory-manager reply wire form (`MmResponse`'s crossing shape).
/// The append-only snapshot tail is populated only for `TAG_STATS`; every
/// other reply carries [`crate::monitor::MemorySnapshot::EMPTY`]. 72 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct MmRepWire {
    /// Outcome discriminant: [`TAG_GRANTED`](Self::TAG_GRANTED) /
    /// [`TAG_RELEASED`](Self::TAG_RELEASED) /
    /// [`TAG_REFUSED`](Self::TAG_REFUSED) /
    /// [`TAG_STATS`](Self::TAG_STATS).
    pub tag: u8,
    /// `Refused`: the `MmRefusal` discriminant
    /// ([`REFUSE_BAD_PROPS`](Self::REFUSE_BAD_PROPS)…); zero otherwise.
    pub refusal: u8,
    /// Grant-bearing transfers this reply was effected with — how many
    /// capability deliveries the client's pre-pinned reply posts collect
    /// (§5.5; `1` for a grant, `0` on refusal/release or when the
    /// service dropped the grant on a protocol-violating client).
    pub grants: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 5],
    /// `Stats`: the memory manager's allocator and supply books.
    pub memory: crate::monitor::MemorySnapshot,
}

impl MmRepWire {
    /// `MmResponse::Granted` — the extent token arrives by delivery.
    pub const TAG_GRANTED: u8 = 1;
    /// `MmResponse::Released`.
    pub const TAG_RELEASED: u8 = 2;
    /// `MmResponse::Refused` — see `refusal`.
    pub const TAG_REFUSED: u8 = 3;
    /// `MmResponse::Stats` — no capability grant accompanies the sample.
    pub const TAG_STATS: u8 = 4;
    /// `MmResponse::Accepted` — an administration step took effect.
    pub const TAG_ACCEPTED: u8 = 5;
    /// `MmResponse::PeerAdmitted` — `reserved[0..2]` carries the admitted
    /// face's arrival (little-endian); the rest stays zero.
    pub const TAG_PEER_ADMITTED: u8 = 6;

    /// `MmRefusal::BadProps`.
    pub const REFUSE_BAD_PROPS: u8 = 1;
    /// `MmRefusal::Exhausted`.
    pub const REFUSE_EXHAUSTED: u8 = 2;
    /// `MmRefusal::BadReturn`.
    pub const REFUSE_BAD_RETURN: u8 = 3;
    /// The grant session refused after mm reserved the requested window.
    pub const REFUSE_DENIED: u8 = 4;
    /// `MmRefusal::Quota` — the arrival pair's outstanding-supply meter
    /// cannot fit the request (`OP_SUPPLY`'s per-client bound; the
    /// `TableFull` discipline: a loud wire refusal, never a silent
    /// clamp).
    pub const REFUSE_QUOTA: u8 = 5;
    /// `MmRefusal::Pending` — the administration step cannot complete yet
    /// (an admission in flight, or a retired face whose supplies have not
    /// all returned); ask again.
    pub const REFUSE_PENDING: u8 = 6;
}

// ─── monitor — federated telemetry assembly ───────────────────────

/// Monitor request: provider snapshots already owned by the caller. The
/// monitor contributes the privileged system sample and returns one typed
/// aggregate. No capability or arbitrary provider name crosses this wire.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct MonitorReqWire {
    pub op: u8,
    pub reserved: [u8; 7],
    pub memory: crate::monitor::MemorySnapshot,
    pub exec: crate::process::ProcessSnapshot,
}

impl MonitorReqWire {
    pub const OP_SNAPSHOT: u8 = 1;
    pub const OP_PROCESSOR: u8 = 3;
    /// One page of the PCI function census (`crate::device::census`):
    /// `reserved[0]` is the page ordinal, `memory` / `exec` are the zero
    /// records.  The monitor assembles the census from its own
    /// `SystemObserve` capability; the request carries no authority.
    pub const OP_DEVICE_CENSUS: u8 = 2;
}

/// Monitor reply: a complete bounded snapshot or a closed refusal. The
/// service grants no authority, so `grants` is always zero.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct MonitorRepWire {
    pub tag: u8,
    pub refusal: u8,
    pub grants: u8,
    pub reserved: [u8; 5],
    pub snapshot: crate::monitor::TelemetrySnapshot,
    pub processor: crate::processor_status::ProcessorPage,
    /// [`TAG_DEVICE_CENSUS`](Self::TAG_DEVICE_CENSUS) only; the zero page
    /// on every other tag (the codec refuses a stray one).
    pub census: crate::device::DeviceCensusPage,
}

impl MonitorRepWire {
    pub const TAG_PROCESSOR: u8 = 4;
    pub const TAG_SNAPSHOT: u8 = 1;
    pub const TAG_REFUSED: u8 = 2;
    pub const TAG_DEVICE_CENSUS: u8 = 3;
    pub const REFUSE_DENIED: u8 = 1;
}

// ─── swap — the page store (kore-svc-swap) ──────────────────────────

/// The swap request wire form (`SwapRequest`'s crossing shape).  Page
/// bytes never ride this record (#148) — they ride the **staging
/// window**, one shared mapping the client attaches once
/// ([`OP_ATTACH`](Self::OP_ATTACH)) and both sides then address by page
/// index.  16 bytes.
///
/// **Why staging rather than a delivered view per page.**  The original
/// shape delivered a fresh `LeasedRegion` view with every `Out`/`In`,
/// which meant the backend mapped a view per operation — and a receiver's
/// map is one-way: it burns a span of the task's monotone map cursor and
/// the kernel has no receiver-side unmap (`REVOKE_KIND_UNMAP` and
/// `REVOKE_KIND_RETURN` are both lessor verbs).  At 4 KiB a page against
/// a 16 MiB per-thread stride that is ~4096 operations before the backend
/// can never map again — a ceiling both backends carried.  One negotiated
/// window costs one mapping for the service's whole life, removes a
/// session commit from the datapath, and needs no new kernel authority.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct SwapReqWire {
    /// Verb discriminant: [`OP_OUT`](Self::OP_OUT) /
    /// [`OP_IN`](Self::OP_IN) / [`OP_DISCARD`](Self::OP_DISCARD) /
    /// [`OP_ATTACH`](Self::OP_ATTACH).
    pub op: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 3],
    /// `Out`/`In`: the **staging page index** the bytes occupy.
    /// `Attach`: the founder-published return-landing correlation value
    /// naming the transfer this request adopts.  `Discard`: zero.
    pub arg: U32,
    /// `In`/`Discard`: the store token a `Stored` reply returned.
    pub token: U32,
    /// Reserved; must be zero.
    pub reserved2: [u8; 4],
}

impl SwapReqWire {
    /// `SwapRequest::Out` — persist the staging page.
    pub const OP_OUT: u8 = 1;
    /// `SwapRequest::In` — restore into the staging page.
    pub const OP_IN: u8 = 2;
    /// `SwapRequest::Discard` — drop a stored page unrestored.
    pub const OP_DISCARD: u8 = 3;
    /// `SwapRequest::Attach` — adopt the staging window transferred over
    /// the return channel; every later `Out`/`In` addresses it by page.
    pub const OP_ATTACH: u8 = 4;
    /// `SwapRequest::Status` — read the tier's books (no arguments; the
    /// answer is [`SwapRepWire::TAG_STATUS`]'s counter block).
    pub const OP_STATUS: u8 = 5;
}

/// The power request wire form (`PowerRequest`'s crossing shape) — #392.
///
/// One byte of vocabulary and seven of reserve, so the record is the same
/// 8-byte shape as its siblings.  Discriminants are **1-based**: an all-zero
/// (unseeded) record is invalid, not "power the machine off".  That property
/// is worth more here than anywhere else in the tree — a zeroed record must
/// never decode as a machine transition.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct PowerReqWire {
    pub op: u8,
    pub reserved: [u8; 7],
}

impl PowerReqWire {
    pub const OP_POWER_OFF: u8 = 1;
    pub const OP_REBOOT: u8 = 2;
    /// Reversible suspend-to-idle; the machine resumes existing execution on
    /// a wake interrupt rather than running terminal teardown.
    pub const OP_SUSPEND_TO_IDLE: u8 = 3;
    pub const OP_ADMIT_PEER: u8 = 4;
    pub const OP_TAKE_ADMISSION: u8 = 5;
    pub const OP_RETIRE_PEER: u8 = 6;
    pub const OP_CONFIRM_RETIRED: u8 = 7;
    pub const OP_CLOSE_PEER: u8 = 8;
    pub const OP_STATUS: u8 = 9;
    /// Fixed display owner retained-standby acknowledgement and next command.
    pub const OP_DISPLAY_POLL: u8 = 10;
    pub const OP_DISPLAY_STATUS: u8 = 11;
    /// Fixed owner: five-byte startup diagnostic, then two zero bytes.
    pub const OP_DISPLAY_REPORT: u8 = 12;
    /// Read the immutable startup diagnostic; all argument bytes are zero.
    pub const OP_DISPLAY_DIAGNOSTIC: u8 = 13;
    /// Read the cached lid level and runtime close generation; no side effects.
    pub const OP_LID_STATUS: u8 = 14;
    /// Fixed display owner publishes one immutable evidence word: index, u32 LE, zeros.
    pub const OP_DISPLAY_EVIDENCE_REPORT: u8 = 15;
    /// Read one evidence word: index, then zeros. Missing words refuse Unavailable.
    pub const OP_DISPLAY_EVIDENCE: u8 = 16;
}

/// The power reply wire form.  A power service mints no authority, so
/// `grants` is pinned zero and a non-zero value fails the codec closed.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct PowerRepWire {
    pub tag: u8,
    pub refusal: u8,
    /// Always zero — a power service grants nothing, ever.
    pub grants: u8,
    pub reserved: [u8; 5],
}

impl PowerRepWire {
    /// The request was accepted. Administrative/report requests acknowledge
    /// receipt; a transition acknowledgement does not prove completion.
    /// Terminal requests are kernel-latched; coordinated suspend may first
    /// prepare its display owner, and either preparation or entry may abort.
    pub const TAG_ACCEPTED: u8 = 1;
    pub const TAG_REFUSED: u8 = 2;
    pub const TAG_PEER_ADMITTED: u8 = 3;
    pub const TAG_STATUS: u8 = 4;
    pub const TAG_DISPLAY_COMMAND: u8 = 5;
    pub const TAG_DISPLAY_STATUS: u8 = 6;
    pub const TAG_DISPLAY_DIAGNOSTIC: u8 = 7;
    /// Level (0 unknown, 1 open, 2 closed), followed by a little-endian u32 generation.
    pub const TAG_LID_STATUS: u8 = 8;
    /// Echoed evidence index and its u32 LE value.
    pub const TAG_DISPLAY_EVIDENCE: u8 = 9;

    /// The session refused the transition.  Authority (`MODIFY` absent) and
    /// feasibility (no firmware backend realizes it) are **not separable
    /// here**: every `DispatchFault` collapses to `Status::Denied` at the
    /// session boundary, so the service cannot honestly report which it was.
    pub const REFUSE_DENIED: u8 = 1;
    /// The commit itself did not land (no status to read).
    pub const REFUSE_TRANSPORT: u8 = 2;
    pub const REFUSE_PENDING: u8 = 3;
    pub const REFUSE_CONFLICT: u8 = 4;
    pub const REFUSE_UNAVAILABLE: u8 = 5;
}

// ─── block — direct-channel provisioning (kore-svc-block) ───────────

/// Block-provisioner request. The reply's two grants carry the ordinary
/// device and DMA-supply descendants; this record carries policy data only.
/// 16 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct BlockReqWire {
    /// Vocabulary discriminant ([`OP_OPEN`](Self::OP_OPEN)…).
    pub op: u8,
    /// `Open`: 0 = polled completion, 1 = interrupt-capable completion.
    pub mode: u8,
    /// `Open`: 0 = read-only, 1 = writable.
    pub writable: u8,
    /// Reserved; must be zero.
    pub reserved: u8,
    /// `Open`: requested I/O qpair depth.
    pub depth: U16,
    /// Reserved; must be zero.
    pub reserved2: U16,
    /// `Open`: total registered-buffer budget requested by the client.
    pub buffer_bytes: U32,
    /// Reserved; must be zero.
    pub reserved3: U32,
}

impl BlockReqWire {
    /// Provision one direct affine channel authority bundle.
    pub const OP_OPEN: u8 = 1;
    /// Observe whether this machine has an unclaimed controller.
    pub const OP_STATUS: u8 = 2;
}

/// Block-provisioner reply. A successful `Opened` reply carries exactly two
/// rendezvous grants; geometry here is descriptive data, never authority.
/// 32 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct BlockRepWire {
    pub tag: u8,
    pub refusal: u8,
    pub grants: u8,
    /// Geometry flags: bit 0 flush, bit 1 discard, bit 2 read-only.
    pub flags: u8,
    pub block_size: U32,
    pub max_transfer_blocks: U32,
    pub atomic_write_blocks: U32,
    pub block_count: U64,
    /// Live provisioned lineages after this reply.
    pub channels: U16,
    /// Provisioner channel ceiling.
    pub limit: U16,
    pub reserved: U32,
}

impl BlockRepWire {
    pub const TAG_OPENED: u8 = 1;
    pub const TAG_STATUS: u8 = 2;
    pub const TAG_REFUSED: u8 = 3;

    pub const REFUSE_UNAVAILABLE: u8 = 1;
    pub const REFUSE_POLICY: u8 = 2;
    pub const REFUSE_EXHAUSTED: u8 = 3;
    pub const REFUSE_LOST: u8 = 4;
}

// ─── device — the ownership broker (kore-svc-device) ────────────────

// ─── time — alarms over the shared monotonic clock ─────────────────

/// Time-service request wire form. The clock itself is read from the
/// read-only [`crate::clock::ClockPage`]; this control plane exists only for
/// asynchronous alarm lifecycle. 32 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct TimerReqWire {
    pub op: u8,
    pub reserved: [u8; 3],
    pub id: U32,
    pub deadline_ns: U64,
    pub interval_ns: U64,
    pub cookie: U64,
}

impl TimerReqWire {
    pub const OP_ARM_AFTER: u8 = 1;
    pub const OP_ARM_AT: u8 = 2;
    pub const OP_UPDATE: u8 = 3;
    pub const OP_CANCEL: u8 = 4;
    pub const OP_STATUS: u8 = 5;
}

/// Time-service synchronous reply wire form. Alarm expiries use the
/// dedicated [`TimerEventWire`] stream. 48 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct TimerRepWire {
    pub tag: u8,
    pub refusal: u8,
    pub grants: u8,
    pub reserved: u8,
    pub id: U32,
    pub active: U32,
    pub capacity: U32,
    pub armed: U64,
    pub fired: U64,
    pub dropped: U64,
    pub cancelled: U64,
}

impl TimerRepWire {
    pub const TAG_ARMED: u8 = 1;
    pub const TAG_UPDATED: u8 = 2;
    pub const TAG_CANCELLED: u8 = 3;
    pub const TAG_STATUS: u8 = 4;
    pub const TAG_REFUSED: u8 = 5;

    pub const REFUSE_FULL: u8 = 1;
    pub const REFUSE_BAD_ID: u8 = 2;
    pub const REFUSE_BAD_DEADLINE: u8 = 3;
    pub const REFUSE_UNAVAILABLE: u8 = 4;
}

/// One asynchronous alarm expiration. 32 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct TimerEventWire {
    pub id: U32,
    pub expirations: U32,
    pub cookie: U64,
    pub deadline_ns: U64,
    pub fired_ns: U64,
}

/// The device-broker request wire form (`res/DEVICE_SERVICE.md` §10) —
/// **ownership arbitration, never I/O**.
///
/// The vocabulary is ownership arbitration plus read-only inventory
/// observation: a block read, a packet send and a display flip stay per-class,
/// so this protocol never grows a "device request" of its own (§1's
/// god-object ruling).
///
/// A `Claim` carries the claimant's own selector verbatim — the same
/// `SystemService::DEVICES` declaration the manifest cites — so the class
/// a driver may own is a fact about its measured image, never a field it
/// invents at runtime.  16 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct DeviceReqWire {
    /// Vocabulary discriminant ([`OP_CLAIM`](Self::OP_CLAIM)…).  1-based:
    /// an unseeded record is invalid, never a claim.
    pub op: u8,
    /// `Claim`/`Inspect`: which species is wanted
    /// ([`DeviceSpecies`](crate::device::DeviceSpecies)).
    pub species: u8,
    /// `Claim`/`Inspect` (PCI): base class.
    pub class: u8,
    /// `Claim`/`Inspect` (PCI): subclass.
    pub subclass: u8,
    /// `Claim`/`Inspect` (PCI): programming interface.
    pub progif: u8,
    /// `Claim`/`Inspect`: request-vocabulary bits interpreted by the device
    /// codec. These are deliberately not
    /// [`DeviceRecord`](crate::device::DeviceRecord) `HAS_*` bits: the broker
    /// translates claimant requirements into kernel-produced facts, so a peer
    /// cannot assert inventory truth.
    pub flags: u8,
    /// `Claim`/`Inspect` (PCI): required user-mappable BAR bitmap in byte 0;
    /// zero is the frozen legacy spelling for BAR0-only. Byte 1 is reserved
    /// and must be zero. Every other op requires both bytes zero.
    pub reserved: [u8; 2],
    /// `Claim`/`Inspect` (PCI): zero for any identity, otherwise exact
    /// `vendor | device << 16`. `Claim`/`Inspect` (platform): the compressed
    /// ACPI id. `Release`: the inventory index being given back.
    pub word: U32,
    /// `Claim`/`Inspect` (PCI strict MSI-X): little-endian `first_entry: u16`
    /// followed by `count: u16`. Zero for every other request; the
    /// strict-source flag is what makes this payload meaningful.
    pub reserved2: [u8; 4],
}

/// Alternate view of the frozen 16-byte device request for exact-grant
/// admission closure. A CID is correlation data; the serving face supplies
/// ownership and the broker matches its retained kernel claim before acting.
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct DeviceClaimCloseWire {
    pub op: u8,
    pub reserved: [u8; 3],
    pub index: U32,
    pub grant: U64,
}

impl DeviceClaimCloseWire {
    pub fn encode(self) -> DeviceReqWire {
        DeviceReqWire::read_from_bytes(self.as_bytes()).expect("same frozen request size")
    }
    pub fn view(wire: &DeviceReqWire) -> Self {
        Self::read_from_bytes(wire.as_bytes()).expect("same frozen request size")
    }
}
const _: () =
    assert!(core::mem::size_of::<DeviceClaimCloseWire>() == core::mem::size_of::<DeviceReqWire>());

impl DeviceReqWire {
    /// Hand me the first unclaimed entry matching this selector.
    pub const OP_CLAIM: u8 = 1;
    /// I am done with the entry at `word`.
    pub const OP_RELEASE: u8 = 2;
    /// How many entries does the inventory hold?  Lets a claimant tell
    /// "this machine has no such device" from "somebody already has it".
    pub const OP_COUNT: u8 = 3;
    /// Open a claimant-face provisioning window over face slot `word` of
    /// the broker's [`FACE_SUPPLY`](crate::task::role::FACE_SUPPLY) band
    /// — the founding parent's half of the minted-pair handshake.  The
    /// two channel sends the face needs (the broker's grant lane, then
    /// its reply kick) follow as return-plane conveys, each answered by
    /// an implicit [`OP_PROVISION`](Self::OP_PROVISION) step; the second
    /// completes the mint and replies `TAG_MINTED`.
    pub const OP_MINT_FACE: u8 = 4;
    /// **Implicit only** — the serve host synthesizes this step when a
    /// convey lands on the broker's walking return post.  A peer that
    /// writes it onto the request ring is forging the broker's own
    /// internal vocabulary, so the decoder refuses it from the wire.
    pub const OP_PROVISION: u8 = 5;
    /// Observe the first inventory record with the request's species and
    /// identity/class tuple without claiming it. Resource requirements remain
    /// on the wire so the observer can diagnose why the corresponding claim
    /// did not match, but observation grants no authority.
    pub const OP_INSPECT: u8 = 6;
    /// Read one bounded I2C admission page. Only `word` (page ordinal) is
    /// populated; no selector or claim authority is carried.
    pub const OP_INSPECT_I2C: u8 = 7;
    /// Close admission for the exact PCI grant owned by this serving face.
    pub const OP_CLOSE_CLAIM: u8 = 8;
    /// Contain the exact already-closed and drained claim through its issuer.
    pub const OP_QUIESCE_CLAIM: u8 = 9;
    /// Retire registered DMA for the exact closed/drained claim.
    pub const OP_RETIRE_CLAIM_DMA: u8 = 10;
}

/// The device-broker reply wire form — **data only**.  A granted device
/// arrives as a delivered token on the grant channel, never as reply
/// payload; the reply says *which* entry it was, so the claimant can name
/// it in a later `Release`.  20 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct DeviceRepWire {
    /// Outcome discriminant ([`TAG_GRANTED`](Self::TAG_GRANTED)…).
    pub tag: u8,
    /// `Refused`: the refusal discriminant; zero otherwise.
    pub refusal: u8,
    /// Grant-bearing transfers this reply was effected with — exactly one
    /// on `Granted`, zero on everything else.
    pub grants: u8,
    /// `Granted`: which species the delivered token must be possessed as.
    pub species: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 4],
    /// `Granted`: the inventory index handed over.  `Counted`: how many
    /// entries the inventory holds.
    pub word: U32,
    /// `Granted`: the diagnostic word
    /// ([`DeviceRecord::diag`](crate::device::DeviceRecord::diag)) — the
    /// function's geometry facts, for the claimant's readiness note when
    /// realization refuses.  Zero on every other tag.
    pub diag: U32,
    /// `Granted`: PCI vendor id; zero for non-PCI and every other tag.
    pub vendor: U16,
    /// `Granted`: PCI device id; zero for non-PCI and every other tag.
    pub device: U16,
}

impl DeviceRepWire {
    /// The device is yours — one grant rides the reply post.
    pub const TAG_GRANTED: u8 = 1;
    /// The entry is back in the pool.
    pub const TAG_RELEASED: u8 = 2;
    /// `word` is the inventory's size.
    pub const TAG_COUNTED: u8 = 3;
    /// See `refusal`.
    pub const TAG_REFUSED: u8 = 4;
    /// `MintFace` accepted: face slot `word` is reserved and the broker's
    /// walking return post now expects the face's two channel sends.
    pub const TAG_PROVISIONING: u8 = 5;
    /// The first provisioning convey (the grant lane) landed and is
    /// banked; `word` echoes the face.  One more convey completes it.
    pub const TAG_BANKED: u8 = 6;
    /// The face is live: the stride is formatted, the pair is attached,
    /// and requests over it arrive stamped `via = word`.
    pub const TAG_MINTED: u8 = 7;
    /// A read-only inventory observation. `word` is the inventory index,
    /// `diag` is the PCI diagnostic word, `species` names the record kind,
    /// `reserved[0]` is PCI revision, and `reserved[1]` is the settled
    /// [`DeviceRecord`](crate::device::DeviceRecord) flags. No grant rides the
    /// reply.
    pub const TAG_OBSERVED: u8 = 8;
    /// Inert I2C admission page: reserved[0]=row, word/diag=two page words.
    /// All other reserve/species/refusal/grants fields remain zero.
    pub const TAG_I2C_OBSERVED: u8 = 9;
    /// Admission closure only: reserved[0]=state, word=reader count.
    pub const TAG_CLAIM_CLOSED: u8 = 10;
    /// Fabric containment: word=rid, reserved[0]=verdict, reserved[1]=flags.
    pub const TAG_CLAIM_QUIESCED: u8 = 11;
    /// DMA result: reserved[0]=state, word=windows removed this attempt.
    pub const TAG_CLAIM_DMA_RETIRED: u8 = 12;

    /// No entry on this machine matches the selector.
    pub const REFUSE_NO_MATCH: u8 = 1;
    /// Every match is already claimed by somebody.
    pub const REFUSE_TAKEN: u8 = 2;
    /// The hand-out session refused (the kernel's own gate).
    pub const REFUSE_DENIED: u8 = 3;
    /// A `Release` named an index this claimant does not hold.
    pub const REFUSE_NOT_HELD: u8 = 4;
    /// The broker was founded without a [`FACE_SUPPLY`] band, so it can
    /// mint no claimant face at all.
    ///
    /// [`FACE_SUPPLY`]: crate::task::role::FACE_SUPPLY
    pub const REFUSE_NO_SUPPLY: u8 = 5;
    /// `MintFace` named a face slot out of range or not vacant, or a
    /// provisioning window is already open.
    pub const REFUSE_BAD_FACE: u8 = 6;
    /// A return-plane convey arrived that no open provisioning window
    /// expects, or carried the wrong capability kind.  The banked half,
    /// if any, was retired with the refusal (fail-closed: a half-built
    /// face never survives).
    pub const REFUSE_BAD_CONVEY: u8 = 7;
    /// Exact claim is not yet drained or its containment writer is busy.
    pub const REFUSE_BUSY: u8 = 8;
}

/// The swap reply wire form (`SwapResponse`'s crossing shape).  The
/// token is **data** — the server's private map key, never a capability
/// (the SwapBackend contract) — and a swap service grants nothing, ever
/// (`grants` is always zero; pinned so the reply layout matches its
/// siblings).  8 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct SwapRepWire {
    /// Outcome discriminant ([`TAG_STORED`](Self::TAG_STORED)…).
    pub tag: u8,
    /// `Refused`: the `SwapRefusal` discriminant; zero otherwise.
    pub refusal: u8,
    /// Always zero — a swap service can mint no authority.
    pub grants: u8,
    /// [`TAG_STATUS`](Self::TAG_STATUS): the tier's posture bits
    /// ([`STATUS_DEVICE`](Self::STATUS_DEVICE) /
    /// [`STATUS_ROUTE`](Self::STATUS_ROUTE)); zero otherwise.
    pub posture: u8,
    /// `Stored`: the restore token.
    pub token: U32,
    // ── The status books (`TAG_STATUS` only; every other reply zeroes
    //    them rather than leaving stale bytes to misread — the
    //    input-stream reply's rule).  Wide enough for the disk tier's
    //    whole `summarize` row; the zram tier reports no route and no
    //    queue, so its io/msix words stay zero with `posture` saying so.
    /// Store capacity, in pages (zram: its `disk_pages` budget).
    pub capacity_pages: U32,
    /// Pages currently stored.
    pub stored_pages: U32,
    /// Attached staging-window pages (`0` = unattached).
    pub staging_pages: U32,
    /// Reserved; must be zero.
    pub reserved2: [u8; 4],
    /// `Out` ops answered `Stored`.
    pub outs: U64,
    /// `In` ops answered `Restored`.
    pub ins: U64,
    /// `Discard` ops answered `Discarded`.
    pub discards: U64,
    /// Device commands submitted (disk tier; zram: zero).
    pub io_submitted: U64,
    /// Device commands completed.
    pub io_completed: U64,
    /// Device commands that completed in error.
    pub io_errors: U64,
    /// Completion parks resolved as delivered route fires — the MSI-X
    /// delivery witness (`0` beside growing `parks` = delivery-dead).
    pub fires: U64,
    /// Completion parks issued on the held route.
    pub parks: U64,
    /// Route-less operations paced by a clock deadline — no vector, but a
    /// readable clock, so the wait is bounded by elapsed time.
    pub deadline_waits: U64,
    /// Route-less operations paced only by a count of spin hints, because no
    /// clock could supply a deadline at all.
    ///
    /// Carried apart from `deadline_waits` because they are different facts. A
    /// spin count bounds retired instructions, whose wall-clock cost moves
    /// with core frequency, SMT contention, and hypervisor descheduling — so
    /// non-zero here means this row is bounding disk waits by something that
    /// is not a duration. On a serial-less box that appears nowhere else,
    /// which is the whole reason it rides the reply.
    pub spin_waits: U64,
}

impl SwapRepWire {
    /// `SwapResponse::Stored` — restore with `token`.
    pub const TAG_STORED: u8 = 1;
    /// `SwapResponse::Restored`.
    pub const TAG_RESTORED: u8 = 2;
    /// `SwapResponse::Discarded`.
    pub const TAG_DISCARDED: u8 = 3;
    /// `SwapResponse::Refused` — see `refusal`.
    pub const TAG_REFUSED: u8 = 4;
    /// `SwapResponse::Attached` — the staging window is live; `token`
    /// carries how many pages it holds.
    pub const TAG_ATTACHED: u8 = 5;
    /// `SwapResponse::Status` — the books ride the counter block.
    pub const TAG_STATUS: u8 = 6;

    /// [`posture`](Self::posture) bit: a backing device is bound (the
    /// disk tier's controller; always set by the zram tier, whose store
    /// is RAM).
    pub const STATUS_DEVICE: u8 = 1 << 0;
    /// [`posture`](Self::posture) bit: the completion path holds its
    /// delivered interrupt route (`fires`/`parks` are live); clear =
    /// the polled fallback.
    pub const STATUS_ROUTE: u8 = 1 << 1;

    /// `SwapRefusal::BadPage`.
    pub const REFUSE_BAD_PAGE: u8 = 1;
    /// `SwapRefusal::BadDst`.
    pub const REFUSE_BAD_DST: u8 = 2;
    /// `SwapRefusal::BadToken`.
    pub const REFUSE_BAD_TOKEN: u8 = 3;
    /// `SwapRefusal::StoreFull`.
    pub const REFUSE_STORE_FULL: u8 = 4;
    /// `SwapRefusal::Backend`.
    pub const REFUSE_BACKEND: u8 = 5;
    /// `SwapRefusal::NoStaging`.
    pub const REFUSE_NO_STAGING: u8 = 6;
}

// ─── exec — the task manager (kore-svc-exec) ────────────────────────

/// Conveyed-capability table capacity of one exec spawn round — the
/// bound every `convey_idx` field below is validated against (the v2
/// spawn wire, 2026-07-28 ruling: request-side authority transfer rides
/// the return channel, one capability per transfer, each landing in the
/// next exec convey-table slot).
///
/// **v3 held this at 16.**  The v3 shape is *convey roots, carve
/// leaves* — [`SpawnSegment::offset`] lets one conveyed image lease
/// serve every page-disjoint LOAD segment, and [`SpawnWindow`] carves
/// deliverable views out of a conveyed lease instead of demanding one
/// convey per view — so the table got *slacker*, not tighter.
///
/// The bound prices exec's resident loader/shell vocabulary and one maximal
/// request without relying on accidental source reuse.  The client half is
/// derived directly from the spawn ceilings (2 boot entries +
/// `MAX_SPAWN_SEGMENTS` + `MAX_SPAWN_WIRING` + `MAX_SPAWN_WINDOWS` =
/// 2+6+36+9 = 53); the remainder is deliberate bounded headroom for richer
/// child plans.
///
/// 96 (from 62) originally priced the first shell's image client. Linked
/// target resolution and child-loader construction now live behind exec's
/// resident image face, so the pinned shell bundle and maximal child each
/// shrink by ten entries.
///
/// The original 96 rows plus four independently revocable exec-client
/// context banks (seven lanes each). The complete table remains within both
/// the wire's `u8` index and exec's `u128` occupancy vocabulary.
pub const MAX_CONVEY: usize = 124;

/// Image/data segments one `Spawn` may carry (stack, RW data, RO/X
/// text — the client pre-populated the bytes through its own views).
pub const MAX_SPAWN_SEGMENTS: usize = 6;

/// Wiring entries one `Spawn` may carry (channel-end derives
/// delivered into the child's arena — kicks, doorbells, returns).
///
/// Both of the founding's hard budgets are binding and both are
/// asserted in-tree: the request record must keep the whole 8-deep
/// request+reply pair inside the founder's ring window
/// (`kore_svc_exec::wire`'s const-assert), and exec's single-flight
/// result-slot plan must stay under the startup linker's durable band
/// (`kore_svc_exec::service`'s).
///
/// 36 is bounded headroom, not a live plan: the shell's task-local
/// namespace and invocation-context endpoints travel over the post-found
/// `SESSION_CONFIG` channel, so current founding plans wire at most two
/// rows here. Shrinking this ceiling moves the wire (every unused row
/// still costs request bytes), so it is a deliberate #81-style lockstep
/// change, not one to make by drift. The whole request stays inside the
/// four-page exec ring window (`res/boot.toml` `ring_bytes = 0x4000`)
/// at the protocol's eight-deep queue bound.
pub const MAX_SPAWN_WIRING: usize = 36;

/// Deliverable **windows** one `Spawn` may carry — detached
/// `LeasedRegion` tokens carved off a conveyed lease and delivered into
/// the child's arena, which the child installs itself ([`SpawnWindow`]).
///
/// 9 is the maximal plan: the shell's input/output pair rings, four queue
/// views, exec pair, power pair, and mm supply pair.  Linked children add one
/// exec-owned loader-ring view to their caller-authored plan, which remains
/// within the same bound.
pub const MAX_SPAWN_WINDOWS: usize = 9;

/// [`SpawnWindow::flags`]: the child will **forward** this window — move
/// it onward through a rendezvous transfer (the shell hands its queues'
/// sibling views to the input-stream and display services' return posts).
///
/// A transfer is a derive of the moved capability, so `prepare`'s
/// transfer gate demands `DERIVE` on the held record; a window carved
/// without it is mappable but immovable.  Declared per window rather
/// than granted to all, because least authority is the rule: a window
/// the child only maps must not also carry the power to hand it on.
/// The send clamps the delivered rights anyway, so the receiver never
/// inherits the allowance.
pub const WINDOW_FORWARD: u8 = 1 << 0;

/// [`SpawnWindow::flags`]: deliver a **sub-lease**, not a detached view —
/// exec carves `[offset, offset+len)` off the conveyed lease as a durable
/// child `MemLease` (`carve_sublease`, full grant) and the delivery moves
/// the *lessor* position itself into the child's slot.
///
/// This is how a spawned task receives supply-shaped authority the v3
/// window cannot express: a `role::LINK_SUPPLY` the startup linker
/// `claim_supply`s and bump-carves private windows from.  The carve
/// parents off the conveyed lease, so the convey must be a **root**
/// `MemLease` (the kernel bars a sub-of-sub; a view-kind convey refuses
/// at the derive).  [`SpawnWindow::write`] must be `1` (a supply is
/// read-write by nature) and [`WINDOW_FORWARD`] must be clear — the
/// delivered lease already carries `DERIVE` as part of the full grant,
/// and declaring both flavors on one row has no coherent reading.
pub const WINDOW_SUBLEASE: u8 = 1 << 1;

/// **Exec's claimed loader face** — the low 12 convey-table indices the
/// kernel founder pre-lands FIRST, before the first shell's own wiring
/// bundle.  Vocabulary-first is what makes these numbers boot-stable:
/// they never ride behind a founding whose entry count can change, so
/// the constants are invariant across every later bundle edit, and the
/// founder asserts each `Conveyed { index }` reply against them at boot
/// so drift refuses loudly.
///
/// These entries are **exec's own** (slice C): the `claimed` occupancy
/// class — never spawn-referenceable, never swept, never held, never
/// released.  They exist for exactly one consumer: exec's **standing
/// stub** acquire.  At the first
/// [`SPAWN_LINKED`](TaskReqWire::SPAWN_LINKED) spawn, exec derives its
/// own kick/recv ends off these very records, maps its own sibling view
/// of the ring, and `Acquire`s `kore-ld-stub` from the image service —
/// ordinary userland wire traffic.  Exec cannot mint-request a face of
/// its own (the bootstrap circularity: a mint needs a face), so this one
/// stays founder-provided.
///
/// A CHILD's loader face is not here (the slice-C flip): the launcher
/// mints one per spawn from the image service
/// (`ImageReqWire::OP_FACE_MINT` / `OP_FACE_BANK`) and conveys the
/// pieces itself — dynamic `Conveyed { index }` entries, landed per
/// launch, held by the child's row, released at its reap.  Child MEMORY
/// is likewise the launcher's own mint (`MmReqWire::OP_SUPPLY`, slice
/// D).  The convey table's ordinary occupancy recycling IS the re-arm
/// path the old standing vocabulary deferred.
pub const EXEC_CV_LOADER_RING: u8 = 0;
/// Exec's request kick send onto the image row's bell — its stub-acquire
/// kick derives send-from-send off it.
pub const EXEC_CV_LOADER_KICK: u8 = 1;
/// First of the claimed face's `MAX_SEGMENTS` grant recv ends — exec's
/// stub-acquire grant lanes derive off these.
pub const EXEC_CV_LOADER_GRANT_BASE: u8 = 2;

/// The **first shell's bundle indices** — the wiring cross-section the
/// kernel founder lands after the claimed loader face (12..), which the
/// INIT task's spawn record references by number.  Pinned here so the
/// kernel table, init's record, and the boot assertions cite one
/// vocabulary; the founder asserts every `Conveyed { index }` reply
/// against these at boot (the same fail-closed discipline as
/// [`EXEC_CV_LOADER_RING`]).  Unlike the claimed band these are
/// ordinary bundle entries: the shell's spawn references each exactly
/// once, they move landed→held at its founding, and they release at its
/// reap (the convey-table lifecycle IS the T6 custody answer).
///
/// The combined runtime supply (boot page at offset 0, stack, thread
/// block, queues, and the shell's private DATA copy) — a root `MemLease`;
/// the record's offset carves and the `BOOT_FROM_SUPPLY` boot page ride
/// it.  The shell's image bytes are NOT here: its TEXT/RODATA arrive as
/// archive grants init `Acquire`s from the image service and conveys at
/// spawn time (dynamic `Conveyed { index }` replies, past this pinned
/// bundle), and its DATA template is init's pre-copy into this supply.
pub const SHELL_CV_RUNTIME: u8 = 12;
/// The terminal (input) pair ring sub → the shell's `SVC_RING@0`.
pub const SHELL_CV_IN_RING: u8 = 13;
/// The display pair ring sub → `SVC_RING@1`.
pub const SHELL_CV_OUT_RING: u8 = 14;
/// The exec pair ring sub → `SVC_RING@7`.
pub const SHELL_CV_EXEC_RING: u8 = 15;
/// The power pair ring → `SVC_RING@8`.
pub const SHELL_CV_POWER_RING: u8 = 16;
/// The mm pair ring (the shell's supply face — mm's pair 1, the same
/// `shell_client` prefab shape) → `SVC_RING@10`: the
/// generic launcher allocates each child's memory over
/// (`MmReqWire::OP_SUPPLY`), replacing the founder-prefunded spawn
/// supply this index used to carry.
pub const SHELL_CV_MM_RING: u8 = 17;
/// The mm face's one grant-channel recv end (`MemoryManager::
/// REPLY_GRANTS = 1`) → `SVC_GRANT@10`: where an `OP_SUPPLY` reply's
/// lease grant is delivered.
pub const SHELL_CV_MM_GRANT: u8 = 18;
/// First of the eleven fixed wiring leaves (19..=29), in child-slot
/// order: the shell's doorbell recv; terminal/display kicks + returns;
/// `PARENT_SEND` (a leaf off INIT's doorbell root — the witness relay
/// lane); exec kick + grant recv; the bounded `ChannelSpace`
/// grant; and the mm/power request kicks.
pub const SHELL_CV_WIRING_BASE: u8 = 19;
/// One past the last bundle entry — the founding's landed count, and
/// where the table's monotone cursor stands when init spawns the shell.
/// Init's two image-grant conveys (TEXT, RODATA) land here and at `+1`;
/// their indices ride the `Conveyed` replies, never a pin.
pub const SHELL_CV_COUNT: u8 = 30;

/// [`TaskReqWire::boot_write_idx`] flag bit: the boot page is **carved
/// from a landed supply** rather than arriving as two pre-staged
/// conveys.  `boot_write_idx & !BOOT_FROM_SUPPLY` names the supply's
/// convey-table entry (a root `MemLease`; [`MAX_CONVEY`] is below 128, so bit 7
/// is free), and `boot_share_idx` becomes the page offset into it.  Exec
/// carves its own RW stamping view and the child's RO share from the
/// same offset — the only spelling available to a spawner that holds no
/// carveable lease of its own. Current Invoke requires this form; the old
/// detached RW-view/RO-sublease pair remains decodable for a typed refusal.
pub const BOOT_FROM_SUPPLY: u8 = 1 << 7;

/// **Data-only** boot entries one `Spawn` may carry ([`SpawnData`]) —
/// literal numbers stamped into the child's boot block, no capability
/// and no child slot behind them.
pub const MAX_SPAWN_DATA: usize = 4;

/// One spawn **segment**: install a share of the conveyed lease at
/// `convey_idx` — starting [`offset`](Self::offset) bytes into it — into
/// the child's fresh space at `[va, va+len)` with the explicit
/// [`SHARE_ACCESS_R`](crate::session::SHARE_ACCESS_R)`/W/X` bits in
/// `access` (writable shares included — the client pre-populated the
/// bytes; `W && X` is refused, the wire-level W^X backstop).
///
/// **v3 added [`offset`](Self::offset) into four of the six frozen
/// reserved bytes**, so every v2 field keeps its byte offset and the
/// record stays 24 bytes — the sanctioned additive change ("meanings for
/// reserved bytes").  One conveyed image lease now serves all of an
/// image's page-disjoint LOAD segments instead of costing one convey
/// each.  24 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct SpawnSegment {
    /// Convey-table index of the source lease (`< MAX_CONVEY`, and below
    /// the landed count at serve time).
    pub convey_idx: u8,
    /// Explicit access bits ([`SHARE_ACCESS_R`](crate::session::SHARE_ACCESS_R)`/W/X`);
    /// zero and `W|X` refuse.
    pub access: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 2],
    /// Byte offset of this segment's window into the conveyed lease
    /// (page-aligned).  Zero shares the conveyed lease itself — the v2
    /// behaviour exactly, and the only form that works off a lease which
    /// is *already* a sub-lease (the kernel bars a sub-of-sub).  Non-zero
    /// makes the service carve the sub-window first.  `u32` because a
    /// lease-relative offset is bounded by the lease, and the record's
    /// 24-byte budget is the founder ring window's (see
    /// [`MAX_SPAWN_WIRING`]).
    pub offset: U32,
    /// Destination VA in the child (page-aligned, non-zero).
    pub va: U64,
    /// Byte length (page-multiple, non-zero).
    pub len: U64,
}

impl SpawnSegment {
    /// Empty tail entry.
    pub const EMPTY: Self = Self {
        convey_idx: 0,
        access: 0,
        reserved: [0; 2],
        offset: U32::new(0),
        va: U64::new(0),
        len: U64::new(0),
    };
}

/// One spawn **window** — the detached-delivery analogue the deleted
/// init-plan wire spelled `{to_slot != INIT_NO_DELIVERY}`, and the
/// reason an arg-less derive could never express a deliverable view.
///
/// The service carves a **detached** `LeasedRegion` — geometry only, no
/// address space named, no leaf installed — off the conveyed lease at
/// `convey_idx` over `[offset, offset+len)`, and *delivers* it into the
/// child's arena, stamping `role@index` → that child slot in the boot
/// block.  The child performs the sole install through `map_view`, which
/// is exactly why the carve must be the arg-carrying
/// [`WindowCarveReq`](crate::session::WindowCarveReq) form: a detached
/// carve keeps the supply's typed lease state alive on the view, and the
/// kernel's map path refuses a view that carries none.
///
/// Two entries naming the same `convey_idx`/`offset`/`len` are lawful and
/// yield **two independent views** — the sibling-view case (one side maps
/// its half, the other is transferred to the peer).
///
/// `offset`/`len` are `u32` for the same reason [`SpawnSegment::offset`]
/// is: they are lease-relative, and the record must fit the founder's
/// ring window.  16 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct SpawnWindow {
    /// Convey-table index of the supply lease (`< MAX_CONVEY`).
    pub convey_idx: u8,
    /// Receiver write access (`0`/`1`) — an RW ring vs an RO window; the
    /// delivered leaf's write bit comes from this at the child's
    /// `map_view`, never from the founder.
    pub write: u8,
    /// Window flags — [`WINDOW_FORWARD`] and nothing else; unknown bits
    /// refuse at decode.
    pub flags: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 1],
    /// The child's [`role`](crate::task::role) for this delivery.
    pub role: U16,
    /// Instance index within the role.
    pub index: U16,
    /// Byte offset into the supply lease (page-aligned).
    pub offset: U32,
    /// Byte length of the window (page-multiple, non-zero).
    pub len: U32,
}

impl SpawnWindow {
    /// Empty tail entry.
    pub const EMPTY: Self = Self {
        convey_idx: 0,
        write: 0,
        flags: 0,
        reserved: [0; 1],
        role: U16::new(0),
        index: U16::new(0),
        offset: U32::new(0),
        len: U32::new(0),
    };
}

/// One spawn **data** entry: stamp `role@index` → `value` into the child's
/// boot block's dedicated datum table — no capability, no derive, no
/// delivery, and **no child slot consumed**.
///
/// This is what the deleted `InitSpawnBlock` path did with its
/// `(SVC_RETURN, 1, mm_return_land)` row: some boot-block entries are
/// **numbers the child reads**, not handles it holds — a peer service's
/// own landing slot, which the child must cite in an attach request but
/// can never possess (it names a slot in *another* arena).  Duplicate
/// `value`s across entries are legal and expected: two peers may both
/// land at the same slot number in their own arenas.  8 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct SpawnData {
    /// The child's [`role`](crate::task::role) for this datum.
    pub role: U16,
    /// Instance index within the role.
    pub index: U16,
    /// The literal stamped into the boot block's datum table — opaque to the
    /// service and never interpreted as an arena slot by crt0.
    pub value: U32,
}

impl SpawnData {
    /// Empty tail entry.
    pub const EMPTY: Self = Self {
        role: U16::new(0),
        index: U16::new(0),
        value: U32::new(0),
    };
}

/// One v2 spawn **wiring** entry: derive an attenuated child of the
/// conveyed capability at `convey_idx` — `kind` is the requested
/// [`ResourceKind`](crate::cap::ResourceKind) wire value (the client
/// declares it; exec never introspects) and `rights` the requested bits
/// (the kernel's Σ5 attenuation gate is the clamp) — and deliver it into
/// the child's arena.  Child slots are assigned **sequentially from
/// slot 1 in wiring order** (the client orders entries so its intended
/// layout holds); exec stamps the child's
/// [`BootBlock`](crate::task::BootBlock) `(role, index)` → slot map from
/// that same order.  12 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct SpawnWire {
    /// Convey-table index of the derive subject (`< MAX_CONVEY`).
    pub convey_idx: u8,
    /// Wiring flags. [`SPAWN_WIRE_MOVE`] transfers the exact conveyed
    /// capability into the child rather than deriving another attenuation.
    pub reserved: u8,
    /// Requested [`ResourceKind`](crate::cap::ResourceKind) wire value
    /// (`<= 0xFF` — the derive kind byte's envelope).
    pub kind: U16,
    /// The child's [`role`](crate::task::role) for this delivery — what
    /// exec stamps into the child's boot block.
    pub role: U16,
    /// Instance index within the role.
    pub index: U16,
    /// Requested [`Rights`](crate::cap::Rights) bits (`<= 0xFF`; unknown
    /// bits refuse at decode).
    pub rights: U32,
}

/// Move the exact conveyed capability into the founded child.
///
/// This is deliberately distinct from ordinary wiring derivation: a bounded
/// lessor position cannot lawfully grow a sub-of-sub, but a supervisor may
/// hand that same affine position to its worker. The capability's actual
/// kind, rights, lineage, and tag are preserved by the kernel; `kind` remains
/// a fail-closed expected-kind assertion and `rights` must be zero.
pub const SPAWN_WIRE_MOVE: u8 = 1 << 0;

impl SpawnWire {
    /// Empty tail entry.
    pub const EMPTY: Self = Self {
        convey_idx: 0,
        reserved: 0,
        kind: U16::new(0),
        role: U16::new(0),
        index: U16::new(0),
        rights: U32::new(0),
    };
}

/// The exec request wire form (`TaskRequest`'s crossing shape) — **v3**,
/// the request-side authority-transfer protocol (2026-07-28 ruling) in
/// its *convey roots, carve leaves* shape: the legacy block-carrying
/// spawn stays deleted whole (its discriminant `1` is retired, never
/// reused — the [`OpCode`](crate::session::OpCode)-3 precedent).
///
/// `Convey` confirms one capability transferred over the return channel
/// into the current exec convey-table slot; `Spawn` consumes conveyed
/// entries **by table index only** across four tables:
///
/// - [`segments`](Self::segments) — shares into the child's fresh space
///   at a founder-chosen VA, each now at its own
///   [`offset`](SpawnSegment::offset) into the conveyed lease;
/// - [`wiring`](Self::wiring) — attenuation derives delivered to the child;
/// - [`windows`](Self::windows) — detached views carved off a conveyed
///   lease and delivered for the child to install itself;
/// - [`data`](Self::data) — literal numbers stamped into the child's boot
///   block, backed by no capability at all.
///
/// plus the boot page's two conveyed handles (`boot_write_idx`, the RW
/// view exec stamps the child's blocks through; `boot_share_idx`, the
/// sub-lease shared RO into the child at `boot_va`).  Handle
/// presentation opcodes 2/3/4 are frozen retired tombstones: current exec
/// decodes and denies them without touching the bounded convey table. No
/// caller-arena slot number crosses the namespace boundary.
///
/// **v3 grew the record from 336 bytes** and spent two of the
/// three frozen reserved head bytes on the new counts, so every v2 head
/// field keeps its byte offset. The tail name is now privileged installation
/// or retirement metadata only. Ordinary `Invoke` requires it empty; immutable
/// generation selection comes exclusively from the admitting capability.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct TaskReqWire {
    /// Verb discriminant: [`OP_WAIT`](Self::OP_WAIT) /
    /// [`OP_KILL`](Self::OP_KILL) / [`OP_EXIT`](Self::OP_EXIT) /
    /// [`OP_CONVEY`](Self::OP_CONVEY) / [`OP_SPAWN`](Self::OP_SPAWN) /
    /// [`OP_INSTALL`](Self::OP_INSTALL) / [`OP_INVOKE`](Self::OP_INVOKE);
    /// `1` (the legacy spawn) is retired and refuses.
    pub op: u8,
    /// `Spawn`: live prefix of [`segments`](Self::segments)
    /// (`<= MAX_SPAWN_SEGMENTS`).
    pub seg_count: u8,
    /// `Spawn`: live prefix of [`wiring`](Self::wiring)
    /// (`<= MAX_SPAWN_WIRING`).
    pub wiring_count: u8,
    /// Historical `Spawn` spelling: index of the boot page's RW view.
    /// Current Invoke refuses that pair before provisioning; all supported
    /// launchers set `BOOT_FROM_SUPPLY` as described below.
    ///
    /// With [`BOOT_FROM_SUPPLY`](crate::svc::BOOT_FROM_SUPPLY) set, the
    /// low bits name a landed **root supply lease** instead and exec
    /// carves the boot page itself — view and share both — at the page
    /// offset [`boot_share_idx`](Self::boot_share_idx) names.  A spawner
    /// that holds no lease (a sub-of-sub is barred, so a delivered
    /// sub-lease cannot stage boot carves) still founds children this
    /// way; its own sight of the page is a view over the same supply,
    /// delivered separately at its founding.
    pub boot_write_idx: u8,
    /// `Spawn`: convey-table index of the boot page's **sub-lease** (exec
    /// shares it read-only into the child at [`boot_va`](Self::boot_va)).
    ///
    /// With [`BOOT_FROM_SUPPLY`](crate::svc::BOOT_FROM_SUPPLY) set this
    /// is no index at all but the boot page's **page offset** into the
    /// supply (`offset = boot_share_idx * 4096`) — a `u8` of pages spans
    /// exactly the 1 MiB spawn supply.
    pub boot_share_idx: u8,
    /// `Spawn`: live prefix of [`windows`](Self::windows)
    /// (`<= MAX_SPAWN_WINDOWS`).  **v3**, in a frozen reserved byte.
    pub window_count: u8,
    /// `Spawn`: live prefix of [`data`](Self::data)
    /// (`<= MAX_SPAWN_DATA`).  **v3**, in a frozen reserved byte.
    pub data_count: u8,
    /// Op-specific additive feature flags. `Spawn`/`Invoke` use
    /// [`SPAWN_RETURN_HANDLE`](Self::SPAWN_RETURN_HANDLE),
    /// [`SPAWN_LINKED`](Self::SPAWN_LINKED), and the opt-in
    /// [`SPAWN_DELEGABLE_HANDLE`](Self::SPAWN_DELEGABLE_HANDLE), and
    /// [`SPAWN_RETURN_SHARED_WAKE`](Self::SPAWN_RETURN_SHARED_WAKE); privileged
    /// `Install` may use [`INSTALL_CONTEXT`](Self::INSTALL_CONTEXT) exactly
    /// once to request the task-wide generic invocation endpoints. Every other
    /// bit, and every non-zero value on another op, refuses.
    pub reserved: u8,
    /// Frozen opcodes 2/3/4: the historical convey-table index. Retained only
    /// to pin their byte shape; current exec never interprets it.
    pub slot: U32,
    /// Frozen opcode 4: historical exit status; never acted on.
    pub status: I32,
    /// `Spawn`: entry-point VA in the new task.
    pub entry: U64,
    /// `Spawn`: initial stack-pointer VA.
    pub stack: U64,
    /// `Spawn`: cap-arena slots the founded task gets (#207).
    pub arena_slots: U32,
    /// Reserved; must be zero.
    pub reserved2: U32,
    /// `Spawn`: the boot page's VA in the child (page-aligned, non-zero;
    /// deliberately **not** a segment — the boot page is the one window
    /// exec itself stamps, so it rides its own field).
    pub boot_va: U64,
    /// `Spawn`: the segment table; entries past `seg_count` are zero.
    pub segments: [SpawnSegment; MAX_SPAWN_SEGMENTS],
    /// `Spawn`: the wiring table; entries past `wiring_count` are zero.
    pub wiring: [SpawnWire; MAX_SPAWN_WIRING],
    /// `Spawn`: the window table; entries past `window_count` are zero.
    pub windows: [SpawnWindow; MAX_SPAWN_WINDOWS],
    /// `Spawn`: the data table; entries past `data_count` are zero.
    pub data: [SpawnData; MAX_SPAWN_DATA],
    /// Privileged `Install`/`RetireGeneration` metadata. It must be all-zero
    /// for `Invoke`.
    /// This field never authorizes ordinary execution.
    pub target: ImageName,
}

/// Why a library did not seed (`TaskRepWire::LIBRARY_FAILURE_*`): the
/// stage `SeedLibrary` stopped at, so a `libs: L/M` verdict short of M names
/// its cause on the kernel's store line instead of swallowing it.
pub struct LibraryFailure;
impl LibraryFailure {
    pub const NONE: u8 = 0;
    /// The filesystem source window could not be attached or cleaned up.
    pub const SOURCE: u8 = 1;
    /// The `sys-lib` grant or the `lib/` directory could not be opened.
    pub const ROOT: u8 = 2;
    /// Opening the member refused with something other than NotFound.
    pub const OPEN: u8 = 3;
    /// The image service refused to export the archive member.
    pub const EXPORT: u8 = 4;
    /// The member could not be written into the store.
    pub const WRITE: u8 = 5;
    /// The written member did not read back as the exported bytes.
    pub const MATCH: u8 = 6;
    /// A recovered member differed and could not be replaced.
    pub const REPLACE: u8 = 7;
    /// A recovered member's comparison did not complete.
    pub const INDETERMINATE: u8 = 8;
    /// The name is not a catalogue library.
    pub const NAME: u8 = 9;
    /// Write stage: the export lease has no slot.
    pub const WRITE_NO_SLOT: u8 = 10;
    /// Write stage: the export view could not be mapped (the monotone map cursor is the usual wall).
    pub const WRITE_MAP: u8 = 11;
    /// Write stage: the export length does not fit a usize.
    pub const WRITE_LEN: u8 = 12;
    /// Write stage: the mapped export view is shorter than the exported length.
    pub const WRITE_SHORT: u8 = 13;
    /// Write stage: no filesystem client when the write began.
    pub const WRITE_NO_FS: u8 = 14;
    /// Write stage: the store refused to defer commits for the copy.
    pub const WRITE_DEFER: u8 = 15;
    /// Write stage: the typed create of the member refused.
    pub const WRITE_CREATE: u8 = 16;
    /// Write stage: closing the deferral (the one whole-tree commit) refused.
    pub const WRITE_COMMIT: u8 = 17;
    /// Write stage: the member's length does not fit the wire.
    pub const BODY_LEN: u8 = 18;
    /// Write stage: no filesystem client when the body write began.
    pub const BODY_NO_FS: u8 = 19;
    /// Write stage: resizing the fresh member refused.
    pub const BODY_RESIZE: u8 = 20;
    /// Write stage: the source window vanished mid-copy.
    pub const BODY_NO_SOURCE: u8 = 21;
    /// Write stage: the source window is smaller than one chunk.
    pub const BODY_SOURCE_SHORT: u8 = 22;
    /// Write stage: the filesystem client vanished mid-copy.
    pub const BODY_FS_GONE: u8 = 23;
    /// Write stage: a chunk's write refused.
    pub const BODY_WRITE_AT: u8 = 24;
    /// Write stage: closing the written member refused.
    pub const BODY_CLOSE: u8 = 25;
    /// Write stage: exec holds no image-service client.
    pub const NO_IMAGE_CLIENT: u8 = 26;
    /// Write stage: the copy answered false without naming a stage (a new exit lacks its stamp).
    pub const UNCLASSIFIED: u8 = 27;
    /// Source stage: an outstanding filesystem cleanup owner never settled,
    /// so no fresh source request could be issued. Split out of
    /// [`SOURCE`](Self::SOURCE), which used to carry every source-stage
    /// stall as one indistinguishable code.
    pub const SOURCE_CLEANUP: u8 = 28;
    /// Source stage: exec holds no filesystem client at all.
    pub const SOURCE_NO_FS: u8 = 29;
    /// Source stage: the filesystem client was mid-request, so the source
    /// window could not advance on this pass.
    pub const SOURCE_FS_BUSY: u8 = 30;
    /// Source stage: the backing window itself was never produced — the MM
    /// supply refused, or the mapping was absent or short.
    pub const SOURCE_WINDOW: u8 = 31;
    /// Source stage: the window exists but its first `attach_window` did not
    /// complete.
    pub const SOURCE_ATTACH: u8 = 32;
    /// Source stage: an `attach_window` is outstanding and its bounded resume
    /// did not complete.
    pub const SOURCE_ATTACHING: u8 = 33;
    /// Source stage: the window attached but its grant listing or open did not
    /// reach `Ready`.
    pub const SOURCE_STAGE: u8 = 34;
    /// Source stage: the outstanding owner is an open file cleanup.
    pub const SOURCE_CLEANUP_FILE: u8 = 35;
    /// Source stage: the outstanding owner is an immutable revision release.
    pub const SOURCE_CLEANUP_VERSION: u8 = 36;
    /// Source stage: the seed phase still needs maintenance.
    pub const SOURCE_CLEANUP_SEED: u8 = 37;
    /// Source stage: the write phase still needs maintenance.
    pub const SOURCE_CLEANUP_WRITE: u8 = 38;
    /// Source stage: the projection phase still needs maintenance.
    pub const SOURCE_CLEANUP_PROJECTION: u8 = 39;

    // Create stage: `WRITE_CREATE` named only "the typed create refused" and
    // dropped the filesystem's own answer, which is the one fact that says
    // whether the store is full, unreachable, or refusing authority. The band
    // below carries that answer; `WRITE_CREATE` remains for an older image.
    /// Create stage: the open succeeded but the object is not a file.
    pub const CREATE_WRONG_KIND: u8 = 40;
    /// Create stage: the reply was malformed.
    pub const CREATE_WIRE: u8 = 41;
    /// Create stage: the call itself did not complete (transport/ring/timeout).
    pub const CREATE_CALL: u8 = 42;
    /// Create stage: the filesystem refused — `FsError::NotFound`.
    pub const CREATE_NOT_FOUND: u8 = 43;
    /// Create stage: the filesystem refused — `FsError::AlreadyExists`.
    pub const CREATE_ALREADY_EXISTS: u8 = 44;
    /// Create stage: the filesystem refused — `FsError::NotDirectory`.
    pub const CREATE_NOT_DIRECTORY: u8 = 45;
    /// Create stage: the filesystem refused — `FsError::IsDirectory`.
    pub const CREATE_IS_DIRECTORY: u8 = 46;
    /// Create stage: the filesystem refused — `FsError::InvalidName`.
    pub const CREATE_INVALID_NAME: u8 = 47;
    /// Create stage: the filesystem refused — `FsError::InvalidHandle`.
    pub const CREATE_INVALID_HANDLE: u8 = 48;
    /// Create stage: the filesystem refused — `FsError::AccessDenied`.
    pub const CREATE_ACCESS_DENIED: u8 = 49;
    /// Create stage: the filesystem refused — `FsError::AccessLost`.
    pub const CREATE_ACCESS_LOST: u8 = 50;
    /// Create stage: the filesystem refused — `FsError::Conflict`.
    pub const CREATE_CONFLICT: u8 = 51;
    /// Create stage: the filesystem refused — `FsError::OutOfObjects`.
    pub const CREATE_OUT_OF_OBJECTS: u8 = 52;
    /// Create stage: the filesystem refused — `FsError::OutOfEdges`.
    pub const CREATE_OUT_OF_EDGES: u8 = 53;
    /// Create stage: the filesystem refused — `FsError::OutOfHandles`.
    pub const CREATE_OUT_OF_HANDLES: u8 = 54;
    /// Create stage: the filesystem refused — `FsError::OutOfStorage`.
    pub const CREATE_OUT_OF_STORAGE: u8 = 55;
    /// Create stage: the filesystem refused — `FsError::OutOfPlacements`.
    pub const CREATE_OUT_OF_PLACEMENTS: u8 = 56;
    /// Create stage: the filesystem refused — `FsError::OutOfVersionLeases`.
    pub const CREATE_OUT_OF_VERSION_LEASES: u8 = 57;
    /// Create stage: the filesystem refused — `FsError::OutOfIndexSlots`.
    pub const CREATE_OUT_OF_INDEX_SLOTS: u8 = 58;
    /// Create stage: the filesystem refused — `FsError::Range`.
    pub const CREATE_RANGE: u8 = 59;
    /// Create stage: the filesystem refused — `FsError::BudgetExceeded`.
    pub const CREATE_BUDGET: u8 = 60;
    /// Create stage: the filesystem refused — `FsError::Busy`.
    pub const CREATE_BUSY: u8 = 61;
    /// Create stage: the filesystem refused — `FsError::ProviderFailure`.
    pub const CREATE_PROVIDER: u8 = 62;
    /// Create stage: any other filesystem refusal. The `libs:` field is six
    /// bits wide, so 63 is the last code a LIBRARY verdict can carry.
    pub const CREATE_OTHER: u8 = 63;

    // Executable-only stages. `TAG_APP_SEEDED.index` carries seven bits, so
    // these live above the shared band and never appear in a `libs:` verdict.
    /// App stage: the system-region seed posture could not be published.
    pub const APP_POSTURE: u8 = 64;
    /// App stage: the durable revision did not bind to the archive peer.
    pub const APP_REVISION: u8 = 65;
    /// App stage: a catalogued projection edge could not be materialised.
    pub const APP_PROJECTIONS: u8 = 66;
    /// App stage: the closing seed posture could not be published.
    pub const APP_POSTURE_CLOSE: u8 = 67;
    /// App stage: the member was present but its comparison did not complete.
    pub const APP_INDETERMINATE: u8 = 68;
    /// App stage: opening the existing member refused for a reason other than
    /// absence, which must never authorise a replacement create.
    pub const APP_OPEN: u8 = 69;
    /// App stage: the export copy into the store did not complete.
    pub const APP_COPY: u8 = 70;
    /// Projection stage: a preceding source owner did not finish cleanup.
    pub const APP_PROJECTION_CLEANUP: u8 = 71;
    /// Projection stage: opening the seeded source file did not complete.
    pub const APP_PROJECTION_SOURCE: u8 = 72;
    /// Projection stage: opening or creating the directory chain failed.
    pub const APP_PROJECTION_DIRECTORY: u8 = 73;
    /// Projection stage: linking or validating an existing edge failed.
    pub const APP_PROJECTION_LINK: u8 = 74;
    /// Projection stage: retiring an acquired source/directory handle failed.
    pub const APP_PROJECTION_CLOSE: u8 = 75;
    /// App body resize: the exact outstanding request exhausted its observations.
    pub const APP_RESIZE_TIMEOUT: u8 = 76;
    /// App body resize: another retained request already owns the client.
    pub const APP_RESIZE_OUTSTANDING: u8 = 77;
    /// App body resize: another infrastructure failure prevented completion.
    pub const APP_RESIZE_CALL: u8 = 78;
    /// App body resize: the filesystem reply violated the wire protocol.
    pub const APP_RESIZE_WIRE: u8 = 79;
    /// App body resize: the medium has no room for the requested bytes.
    pub const APP_RESIZE_STORAGE: u8 = 80;
    /// App body resize: the bounded placement table is full.
    pub const APP_RESIZE_PLACEMENTS: u8 = 81;
    /// App body resize: the filesystem's work budget was exhausted.
    pub const APP_RESIZE_BUDGET: u8 = 82;
    /// App body resize: another request holds an exact filesystem claim.
    pub const APP_RESIZE_BUSY: u8 = 83;
    /// App body resize: the storage provider reported a failure.
    pub const APP_RESIZE_PROVIDER: u8 = 84;
    /// App body resize: the expected revision was stale.
    pub const APP_RESIZE_CONFLICT: u8 = 85;
    /// App body resize: filesystem authority was denied or lost.
    pub const APP_RESIZE_ACCESS: u8 = 86;
    /// App body resize: another filesystem refusal prevented completion.
    pub const APP_RESIZE_OTHER: u8 = 87;
    /// App body resize: the write owner has no retained file/request.
    pub const APP_RESIZE_NO_OWNER: u8 = 88;
    /// App body resize: the service client is poisoned.
    pub const APP_RESIZE_POISONED: u8 = 89;
    /// App body resize: the request ring is full.
    pub const APP_RESIZE_RING_FULL: u8 = 90;
    /// App body resize: the bounded client has invalid endpoint/timer wiring.
    pub const APP_RESIZE_WIRING: u8 = 91;
    /// App body resize: the transport could not commit.
    pub const APP_RESIZE_TRANSPORT: u8 = 92;
    /// App body resize: runtime-owned landing capacity did not become available.
    pub const APP_RESIZE_BACKPRESSURE: u8 = 93;
    /// App body resize: the request session returned WouldBlock.
    pub const APP_RESIZE_CALL_BUSY: u8 = 94;
    /// App body resize: the request session returned Conflict.
    pub const APP_RESIZE_CALL_CONFLICT: u8 = 95;

    /// Shared member-copy code retains detailed resize causes internally.
    /// Only application replies have room for them: a library/data verdict
    /// preserves the existing six-bit BODY_RESIZE value instead of truncating
    /// a seven-bit application diagnostic into an unrelated library cause.
    pub const fn for_library_reply(failure: u8) -> u8 {
        match failure {
            Self::APP_RESIZE_TIMEOUT..=Self::APP_RESIZE_CALL_CONFLICT => Self::BODY_RESIZE,
            _ => failure,
        }
    }
}

impl TaskReqWire {
    pub const OP_PROCESSOR: u8 = 23;
    // Discriminant 1 is RETIRED: it carried the legacy block-carrying
    // spawn (`Spawn { entry, stack, flags, block }`), deleted whole by
    // the 2026-07-28 v2 ruling.  It stays refused and must not be
    // reused (the `OpCode`-3 precedent).
    /// Retired service `Wait`; decodes only to an effectless denied marker.
    pub const OP_WAIT: u8 = 2;
    /// Retired service `Kill`; decodes only to an effectless denied marker.
    pub const OP_KILL: u8 = 3;
    /// Retired service `Exit`; decodes only to an effectless denied marker.
    pub const OP_EXIT: u8 = 4;
    /// Retired split-invocation `Convey`; ordinary invocation is atomic.
    pub const OP_CONVEY: u8 = 5;
    /// Retired raw `Spawn`; ordinary founding is capability-selected `Invoke`.
    pub const OP_SPAWN: u8 = 6;
    /// Retired name-authorized `Launch`; the opcode remains reserved and
    /// must never be reused (the `OpCode`-3 precedent).
    pub const OP_LAUNCH: u8 = 7;
    /// Retired image-preparation phase of the split launch protocol.
    pub const OP_PREPARE_IMAGES: u8 = 8;
    /// Retired pathname workspace mutation; cwd is a moved structural view.
    pub const OP_SET_WORKSPACE: u8 = 9;
    /// Privileged resolution of a catalog name into executable authority.
    /// `15`: the first value above the retired band — `7`/`8`/`9` carried
    /// `Launch`/`PrepareImages`/`SetWorkspace` and stay frozen refusals.
    pub const OP_INSTALL: u8 = 15;
    /// Retired Invoke whose task-first grant order could not describe a
    /// committed wake-only prefix after terminal Unfounded.
    pub const OP_INVOKE_LEGACY: u8 = 16;
    /// Capability-selected Invoke with wake-first results and explicit
    /// committed-discard refusal. Mixed old/new peers cannot reinterpret it.
    pub const OP_INVOKE: u8 = 25;
    pub const INVOKE_WAKE_GRANT: usize = 0;
    pub const fn invoke_task_grant(return_shared_wake: bool) -> usize {
        return_shared_wake as usize
    }
    /// Irreversibly close the bootstrap-only global-name installation phase
    /// before the ordinary client inherits the serving pair.
    pub const OP_SEAL_INSTALLER: u8 = 17;
    /// Copy one already-registered generation into the durable executable
    /// store. `target` names it; the request carries no plan, window, or
    /// data — seeding conveys bytes the service already validated.
    pub const OP_SEED_APP: u8 = 18;
    /// Load-and-discard probe of the durable executable store: exec resolves
    /// `target` from `app/` via `register_fs_image` and answers whether the
    /// bytes linked, creating no peer, face, or lasting authority. It lets a
    /// machine witness the store read-back path (the seed's mirror) AFTER
    /// installation is sealed — the ordinary `Install { from_tree }` path is
    /// closed by then. Seal-exempt like `OP_SEED_APP`; via==0 only.
    pub const OP_TREE_LOAD_PROBE: u8 = 19;
    /// Acquire one durable-store revision as an execute capability — the
    /// `app/<name>` command spelling.  The FIRST client-callable store verb:
    /// seal-exempt AND via-open, because it mints only what the store
    /// already authenticates.  Digest dedup resolves an unchanged revision
    /// to the resident generation — whose archive-authenticated profile the
    /// reply then carries (authenticated bytes keep their authenticated
    /// metadata); a genuinely fresh tree revision registers at profile 0
    /// (unauthenticated bytes carry none).  Bootstrap-prefixed names and
    /// global-name installation stay out of reach: this verb never binds a
    /// name, shadows an entry, or touches the sealed installer.
    pub const OP_TREE_ACQUIRE: u8 = 20;
    /// One page of the PCI function census (`crate::device::census`),
    /// relayed from the monitor for a launcher stamping an executable's
    /// context prefix; `slot` is the page ordinal.  Bind-inert data, no
    /// grants, answered on any launcher pair.
    pub const OP_DEVICE_CENSUS: u8 = 21;
    /// Copy one archive library into the durable library directory
    /// (`lib/`, projected READ-only as every task's `libs/` view).  The
    /// [`OP_SEED_APP`](Self::OP_SEED_APP) shape — `target` names the archive
    /// member, no plan, no grants — minus the executable half: a library
    /// gains no peer, no projection, and no launch authority; a task links
    /// it into itself at runtime.  Seal-exempt, via==0 only.
    pub const OP_SEED_LIBRARY: u8 = 22;

    /// Every retired `TaskReqWire` discriminant, in one definition the
    /// decoder cites (the [`derive_kind_known`](crate::session::derive_kind_known)
    /// discipline): a frozen value can never be silently re-admitted as a
    /// live operation, and a stale caller receives a typed refusal rather
    /// than a dropped request.
    pub const fn op_retired(op: u8) -> bool {
        (op >= 1 && op <= Self::OP_SET_WORKSPACE) || op == Self::OP_INVOKE_LEGACY
    }
    /// Privileged retirement of the newest immutable generation named by the
    /// installer. Ordinary executable holders cannot issue this operation.
    pub const OP_RETIRE_GENERATION: u8 = 10;
    /// Privileged construction of one post-less, task-owned exec client face.
    pub const OP_MINT_CLIENT: u8 = 11;
    /// Collect the complete reply-grant receive bank for a minted face.
    pub const OP_CLIENT_BANK: u8 = 12;
    /// Collect the task-wide invocation-context endpoints for a minted face.
    pub const OP_CLIENT_CONTEXT: u8 = 13;
    /// Retire one exact minted exec client face.
    pub const OP_RETIRE_CLIENT: u8 = 14;
    /// Confirm physical cleanup of a client retirement. `slot` is the face,
    /// `entry` is the nonzero service-issued retirement token.
    pub const OP_CONFIRM_CLIENT_RETIRED: u8 = 24;
    /// `Spawn` asks exec to return an affine parent handle after any optional
    /// shared-wake grant (see `invoke_task_grant`).
    /// Bootstrap founding clears this bit because the kernel temporarily owns
    /// the pair and consumes only the data reply.
    pub const SPAWN_RETURN_HANDLE: u8 = 1 << 0;
    /// `Spawn` asks exec to add the **standing stub** to the founding: the
    /// linker stub's spans are shared into the child (read-only / exec —
    /// the stub is deliberately RELRO-whole, so one acquired copy serves
    /// every child) and the task enters at the stub's validated entry.
    /// The record's own `entry` must be zero — the child never chooses
    /// where the linker starts.  Exec provisions the stub itself, on
    /// first use, by acquiring `kore-ld-stub` from the image service
    /// over the pre-landed loader face; a spawn whose acquire cannot
    /// complete refuses rather than founding a child with no linker.
    pub const SPAWN_LINKED: u8 = 1 << 1;
    /// `Spawn` asks for the returned lifecycle owner to retain `DERIVE`, so
    /// its caller may attenuate an observer into a sibling worker. This bit
    /// is meaningful only with [`SPAWN_RETURN_HANDLE`](Self::SPAWN_RETURN_HANDLE);
    /// it is available for any otherwise authorized spawn and changes only
    /// the parent's returned handle, not the child's authority. Ordinary exec
    /// clients continue to receive a non-delegable owner.
    pub const SPAWN_DELEGABLE_HANDLE: u8 = 1 << 2;
    /// `Invoke` asks exec to return an optional affine wake at grant lane 0,
    /// with the task handle following at lane 1: an attenuated send descendant of the provisioned service response channel that
    /// also supplies the child's sole executor receive endpoint. This is a
    /// closed application-bundle contract, not caller-selected channel
    /// authority: exec accepts it only for the exactly authenticated
    /// filesystem-window or power-window profile and only while it provisions that
    /// service face in the same founding transaction.
    pub const SPAWN_RETURN_SHARED_WAKE: u8 = 1 << 3;
    /// Privileged `Install` asks exec to include this installer's task-wide
    /// generic invocation context before the generation-bound endpoint. The
    /// service accepts this bit exactly once; subsequent generations return
    /// only their reusable executable endpoint. Reusing bit zero is safe
    /// because `reserved` is an op-discriminated field.
    pub const INSTALL_CONTEXT: u8 = 1 << 0;
    /// Privileged `Install` asks exec to resolve the generation from the
    /// durable executable store (`app/`) rather than the boot archive: exec
    /// copies the immutable filesystem revision into an MM lease and registers
    /// it, shadowing any archive entry of the same name (`res/IMAGE.md` — the
    /// store, not the boot seed, is where an executable lives). A tree-resident
    /// executable carries no archive authentication, so it always registers
    /// with launch profile zero — the wire therefore requires `reserved2 == 0`
    /// when this bit is set. Independent of `INSTALL_CONTEXT` (bit 0).
    pub const INSTALL_FROM_TREE: u8 = 1 << 1;
}

/// The exec reply wire form (`TaskResponse`'s crossing shape). An owned
/// `Spawned` reply carries exactly one task-handle grant. `id` remains
/// correlation data only (hash-derived, unordered). A `Spawned` reply uses
/// [`index`](Self::index) for the selected generation's validated segment
/// count. The append-only snapshot tail is populated only on `Spawned`; all
/// other tags require
/// [`crate::monitor::TelemetrySnapshot::EMPTY`]. The processor tail is populated
/// only by `ProcessorStatus`; other tags require its canonical all-zero value.
/// 616 bytes; both peers must be rebuilt together.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct TaskRepWire {
    /// Outcome discriminant ([`TAG_SPAWNED`](Self::TAG_SPAWNED)…).
    pub tag: u8,
    /// `Refused`: the `TaskRefusal` discriminant; zero otherwise.
    pub refusal: u8,
    /// Grant-bearing transfers this reply was effected with: `1` for an
    /// owned `Spawned` response, `0` for every refusal, lifecycle response,
    /// and the one-shot bootstrap `Spawned` response.
    pub grants: u8,
    /// `Spawned`: the selected generation's validated segment count (`<=
    /// crate::image::MAX_SEGMENTS`; zero for an unlinked spawn).
    /// Client lifecycle replies: face index; other tags use their documented
    /// tag-contextual index or zero.
    pub index: u8,
    /// Frozen tag 3: historical service-wait exit status.
    pub status: I32,
    /// `Spawned`: the founded task's id (diagnostic).
    /// Client retirement replies: nonzero service-issued retirement token.
    pub id: U64,
    /// `Spawned`: exec's bounded process-table snapshot after admitting the
    /// child. Zero/empty on every other tag.
    pub snapshot: crate::monitor::TelemetrySnapshot,
    /// `Spawned`: the launcher-captured filesystem posture, relayed for the
    /// caller to publish into the child's process page. Kept out of
    /// [`TelemetrySnapshot`] so the monitor's own reply record — which
    /// carries that snapshot and nothing else — does not grow an
    /// always-empty filesystem half.
    pub filesystem: crate::monitor::FilesystemSnapshot,
    /// `TAG_SPAWNED`: the launcher's frame-service snapshot, the page's third
    /// producer; empty on every other tag and on launches without the
    /// network profile.
    pub net: crate::monitor::NetSnapshot,
    pub processor: crate::processor_status::ProcessorPage,
    /// [`TAG_DEVICE_CENSUS`](Self::TAG_DEVICE_CENSUS)'s page; the zero
    /// page on every other tag.
    pub census: crate::device::DeviceCensusPage,
}

impl TaskRepWire {
    pub const TAG_PROCESSOR: u8 = 23;
    /// `TaskResponse::Spawned` — paired with one parent-handle grant for an
    /// ordinary owned spawn; the one-shot kernel bootstrap form is data-only.
    pub const TAG_SPAWNED: u8 = 1;
    /// Retired service-wait `Running` tag; current exec never emits it.
    pub const TAG_RUNNING: u8 = 2;
    /// Retired service-wait `Exited` tag; current exec never emits it.
    pub const TAG_EXITED: u8 = 3;
    /// Retired service-kill tag; current exec never emits it.
    pub const TAG_KILLED: u8 = 4;
    /// Retired service-exit acknowledgement; current exec never emits it.
    pub const TAG_DEPARTING: u8 = 5;
    /// `TaskResponse::Refused` — see `refusal`.
    pub const TAG_REFUSED: u8 = 6;
    /// Retired split-convey reply tag; current exec refuses this wire shape.
    pub const TAG_CONVEYED: u8 = 7;
    /// Retired split-launch progress tag; current exec never emits it.
    pub const TAG_IMAGES_PREPARING: u8 = 8;
    /// Retired split-launch readiness tag; current exec never emits it.
    pub const TAG_IMAGES_READY: u8 = 9;
    /// Retired pathname-workspace acknowledgement; current exec never emits it.
    pub const TAG_WORKSPACE_SET: u8 = 10;
    /// A generation-bound executable endpoint package was granted.
    /// `15`: the first value above the retired tag band — `8`/`9`/`10`
    /// carried the split-launch/workspace acknowledgements and stay frozen.
    pub const TAG_INSTALLED: u8 = 15;
    /// The bootstrap-only global-name installer was irreversibly closed.
    pub const TAG_INSTALLER_SEALED: u8 = 16;
    /// The selected immutable image generation and its executable peer root
    /// entered retirement.
    pub const TAG_GENERATION_RETIRED: u8 = 17;
    /// A generation is present in the durable executable store. `index` is
    /// 1 when it is, 0 when no store could be reached — a posture, not a
    /// refusal.
    pub const TAG_APP_SEEDED: u8 = 18;
    /// `TAG_APP_SEEDED.index`: the executable is in the durable store.
    pub const APP_SEEDED: u8 = 1 << 0;
    /// `TAG_APP_SEEDED.index` bits 1..7: why the executable did NOT seed
    /// ([`LibraryFailure`](crate::svc::LibraryFailure), which the executable
    /// path shares and extends). Seven bits, so it reaches the app-only
    /// stages above 63 that a six-bit library verdict cannot carry.
    pub const APP_FAILURE_SHIFT: u8 = 1;
    pub const APP_FAILURE_MASK: u8 = 0xfe;
    pub const APP_SEED_MASK: u8 = Self::APP_SEEDED | Self::APP_FAILURE_MASK;
    /// Reply tag for [`TaskReqWire::OP_TREE_LOAD_PROBE`]: `index` is the
    /// fail-closed [`TREE_PROBE_LINKED`](Self::TREE_PROBE_LINKED) /
    /// [`TREE_PROBE_ALREADY_RESIDENT`](Self::TREE_PROBE_ALREADY_RESIDENT)
    /// bitset. Resident implies linked; every other bit pattern is malformed.
    pub const TAG_TREE_PROBED: u8 = 19;
    /// `TAG_TREE_PROBED.index`: the durable revision linked successfully.
    pub const TREE_PROBE_LINKED: u8 = 1 << 0;
    /// `TAG_TREE_PROBED.index`: digest identity selected an existing peer, so
    /// the probe registered no new image generation.
    pub const TREE_PROBE_ALREADY_RESIDENT: u8 = 1 << 1;
    /// Complete live bit vocabulary for `TAG_TREE_PROBED.index`.
    pub const TREE_PROBE_MASK: u8 = Self::TREE_PROBE_LINKED | Self::TREE_PROBE_ALREADY_RESIDENT;
    /// Reply tag for [`TaskReqWire::OP_TREE_ACQUIRE`]: a store revision's
    /// executable endpoint package was granted.  `index` carries the
    /// generation's segment count (the [`TAG_INSTALLED`](Self::TAG_INSTALLED)
    /// discipline) and `status` carries its launch profile word — the
    /// resident generation's authenticated profile on a digest dedup, `0`
    /// for a fresh tree registration.
    pub const TAG_TREE_ACQUIRED: u8 = 20;
    /// One census page in `census`; every other field zero.
    pub const TAG_DEVICE_CENSUS: u8 = 21;
    /// Reply to [`TaskReqWire::OP_SEED_LIBRARY`]: `index` is the
    /// [`LIBRARY_SEEDED`](Self::LIBRARY_SEEDED) /
    /// [`LIBRARY_MANIFEST`](Self::LIBRARY_MANIFEST) bitset — the library is
    /// in `lib/`, and the system package manifest (`pkgs/kore/manifest`)
    /// has been written, which happens once every executable and library
    /// has seeded.  A zero bit is a posture (no store), never a refusal.
    pub const TAG_LIBRARY_SEEDED: u8 = 22;
    /// `TAG_LIBRARY_SEEDED.index`: the library is in the durable store.
    pub const LIBRARY_SEEDED: u8 = 1 << 0;
    /// `TAG_LIBRARY_SEEDED.index`: the system package manifest stands.
    pub const LIBRARY_MANIFEST: u8 = 1 << 1;
    /// `TAG_LIBRARY_SEEDED.index` bits 2..7: why the library did NOT seed
    /// (0 = no failure), the [`LibraryFailure`](crate::svc::LibraryFailure)
    /// vocabulary — a seed answered false is a posture, and this says which.
    pub const LIBRARY_FAILURE_SHIFT: u8 = 2;
    pub const LIBRARY_FAILURE_MASK: u8 = 0x3f << Self::LIBRARY_FAILURE_SHIFT;
    /// Complete live bit vocabulary for `TAG_LIBRARY_SEEDED.index`.
    pub const LIBRARY_SEED_MASK: u8 =
        Self::LIBRARY_SEEDED | Self::LIBRARY_MANIFEST | Self::LIBRARY_FAILURE_MASK;
    /// A task-owned exec client face was attached; `index` is its bounded id.
    pub const TAG_CLIENT_MINTED: u8 = 11;
    /// A minted face received its complete reply-grant receive bank.
    pub const TAG_CLIENT_BANKED: u8 = 12;
    /// A minted face received its invocation-context endpoint set.
    pub const TAG_CLIENT_CONTEXT: u8 = 13;
    /// One exact minted exec client face entered retirement.
    /// Legacy queued acknowledgement; never proves physical cleanup.
    pub const TAG_CLIENT_RETIRED: u8 = 14;
    /// Retirement accepted or still pending: `index` face, `id` token.
    pub const TAG_CLIENT_RETIRING: u8 = 24;
    /// Physical cleanup confirmed: `index` face, `id` token.
    pub const TAG_CLIENT_RETIREMENT_CONFIRMED: u8 = 25;

    /// `TaskRefusal::BadBlock`.
    pub const REFUSE_BAD_BLOCK: u8 = 1;
    /// `TaskRefusal::BadHandle`.
    pub const REFUSE_BAD_HANDLE: u8 = 2;
    /// `TaskRefusal::LedgerFull`.
    pub const REFUSE_LEDGER_FULL: u8 = 3;
    /// The request needs authority transfer this pair does not carry:
    /// the serving return leg is absent (a return-less boot-chain row —
    /// the constructor posture), so no convey can have landed.  Arena-local slot numbers are
    /// never interpreted across arena namespaces.
    pub const REFUSE_UNWIRED: u8 = 4;
    /// `TaskRefusal::TableFull` — every convey slot is either landed or
    /// retained by a live child.
    pub const REFUSE_TABLE_FULL: u8 = 5;
    /// `TaskRefusal::NothingConveyed` — no capability record has landed
    /// in the current convey-table slot.
    pub const REFUSE_NOTHING_CONVEYED: u8 = 6;
    /// `TaskRefusal::Denied` — an unauthorized data-only spawn attempted to
    /// use exec's closed one-shot bootstrap window.
    pub const REFUSE_DENIED: u8 = 7;
    /// `TaskRefusal::PlanTooLarge` — the composite atomic founding would
    /// exceed the session invocation budget.
    pub const REFUSE_PLAN_TOO_LARGE: u8 = 8;
    /// `TaskRefusal::ImageUnavailable` — the named linked target was absent,
    /// denied, invalid, or not a dynamic executable.
    pub const REFUSE_IMAGE_UNAVAILABLE: u8 = 9;
    /// `TaskRefusal::Retired` — the request named a frozen tombstone
    /// discriminant ([`TaskReqWire::op_retired`]); nothing was attempted.
    pub const REFUSE_RETIRED: u8 = 10;
    /// `TaskRefusal::Contended` — concurrent invocations landed admissions
    /// for more than one generation, so no landing could be attributed to
    /// this request; every in-flight admission was retired and the caller
    /// re-derives and retries.
    pub const REFUSE_CONTENDED: u8 = 11;
    /// `TaskRefusal::PeerAdmissionPending` — the face's service accepted the
    /// child's peer bundle but had not yet published the admission verdict
    /// within the bounded re-ask, so no peer identity could be taken.
    /// Nothing is wrong and nothing is lost; the launch is re-issuable.
    ///
    /// Every launch-time face raises this — filesystem, power, accelerator
    /// and memory — so the code names the stage, never the service.  The
    /// caller learns *which* only from what the launch asked for.
    pub const REFUSE_PEER_ADMISSION_PENDING: u8 = 12;
    /// `TaskRefusal::PeerRefused` — the face's service refused the child's
    /// peer admission for a reason that is neither capacity nor a pending
    /// verdict. The remote side is closed; the refusal is terminal.
    pub const REFUSE_PEER_REFUSED: u8 = 13;
    /// `TaskRefusal::PeerAdmissionLost` — the admission verdict could not be
    /// observed at all (a malformed reply or a transport boundary), so
    /// whether the service admitted the peer is UNKNOWN and its remote
    /// state is reconciled lazily rather than assumed closed.
    pub const REFUSE_PEER_ADMISSION_LOST: u8 = 14;
    /// `TaskRefusal::PeerServiceUnwired` — a launch whose profile asks for a
    /// face reached peer construction without that face's service attached.
    pub const REFUSE_PEER_SERVICE_UNWIRED: u8 = 15;
    /// `TaskRefusal::PeerRingTransfer` — the peer's shared ring could not be
    /// transferred to the face's service.
    pub const REFUSE_PEER_RING_TRANSFER: u8 = 16;
    /// `TaskRefusal::PeerKickTransfer` — the peer's response-send endpoint
    /// could not be transferred to the face's service.
    pub const REFUSE_PEER_KICK_TRANSFER: u8 = 17;
    /// `TaskRefusal::PeerReturnTransfer` — the peer's return-receive endpoint
    /// could not be transferred to the face's service.
    pub const REFUSE_PEER_RETURN_TRANSFER: u8 = 18;
    /// `TaskRefusal::PeerTransientUnwired` — exec had no transient mapping
    /// aperture with which to format the child's private ring.
    pub const REFUSE_PEER_TRANSIENT_UNWIRED: u8 = 19;
    /// `TaskRefusal::PeerMemoryUnwired` — exec had no memory-service face
    /// from which to allocate the child's private ring.
    pub const REFUSE_PEER_MEMORY_UNWIRED: u8 = 20;
    /// `TaskRefusal::PeerGrantTransfer` — the accelerator peer's reply-grant
    /// sender could not be transferred to the service.
    pub const REFUSE_PEER_GRANT_TRANSFER: u8 = 21;
    /// `TaskRefusal::PeerAdmissionWedged` — an EARLIER launch left a face
    /// whose admission verdict was never published, and this launch was
    /// refused before it asked for anything of its own.
    ///
    /// Distinct from [`REFUSE_PEER_ADMISSION_PENDING`](Self::REFUSE_PEER_ADMISSION_PENDING)
    /// because the two say opposite things about a retry: a pending verdict
    /// is an ordinary schedule and the next attempt usually lands, while a
    /// wedged one repeats until the stale face is settled.  Reading one as
    /// the other is what makes a launch look intermittently slow when it is
    /// in fact permanently refused.
    pub const REFUSE_PEER_ADMISSION_WEDGED: u8 = 22;
    /// Terminal Found committed its preceding effects but discarded the
    /// child. An optional wake grant still belongs to this exact response.
    pub const REFUSE_UNFOUNDED: u8 = 23;
}

// Live task discriminants stay disjoint from the frozen retired band on
// both wire directions — the compile-time twin of `op_retired`, so a
// future value choice cannot drift back into a tombstone (#81 discipline).
const _: () = {
    assert!(!TaskReqWire::op_retired(TaskReqWire::OP_RETIRE_GENERATION));
    assert!(!TaskReqWire::op_retired(TaskReqWire::OP_MINT_CLIENT));
    assert!(!TaskReqWire::op_retired(TaskReqWire::OP_CLIENT_BANK));
    assert!(!TaskReqWire::op_retired(TaskReqWire::OP_CLIENT_CONTEXT));
    assert!(!TaskReqWire::op_retired(TaskReqWire::OP_RETIRE_CLIENT));
    assert!(!TaskReqWire::op_retired(
        TaskReqWire::OP_CONFIRM_CLIENT_RETIRED
    ));
    assert!(!TaskReqWire::op_retired(TaskReqWire::OP_INSTALL));
    assert!(TaskReqWire::op_retired(TaskReqWire::OP_INVOKE_LEGACY));
    assert!(!TaskReqWire::op_retired(TaskReqWire::OP_INVOKE));
    assert!(!TaskReqWire::op_retired(TaskReqWire::OP_SEAL_INSTALLER));
    assert!(TaskRepWire::TAG_INSTALLED > TaskRepWire::TAG_WORKSPACE_SET);
    assert!(TaskRepWire::TAG_INSTALLER_SEALED > TaskRepWire::TAG_WORKSPACE_SET);
    assert!(TaskRepWire::TAG_GENERATION_RETIRED > TaskRepWire::TAG_WORKSPACE_SET);
};

// ─── image — the lessor of code (kore-svc-image) ────────────────────

/// The image request wire form (`ImageRequest`'s crossing shape).
/// `Register` names the founder-published return landing that identifies the
/// authority leg used for this call; the image lease itself rides that
/// channel, never this record. 152 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct ImageReqWire {
    /// Verb discriminant: [`OP_REGISTER`](Self::OP_REGISTER) /
    /// [`OP_ACQUIRE`](Self::OP_ACQUIRE) / [`OP_RETIRE`](Self::OP_RETIRE) /
    /// [`OP_DESCRIBE`](Self::OP_DESCRIBE).
    pub op: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 7],
    /// The image name (all verbs).
    pub name: ImageName,
    /// `Register`: the founder-published return landing name. The service
    /// joins it to the capability received in a runtime-owned persistent slot
    /// (#148: no capability ever rides this record).
    pub slot: U32,
    /// Reserved; must be zero.
    pub reserved2: U32,
    /// Reserved; must be zero.
    pub reserved3: U64,
    /// `Register`: exact immutable-source identity and explicit launch
    /// profile, authenticated against the transferred bytes by ImageService.
    /// Every other verb requires the all-zero value.
    pub source: crate::image::ImageSourceMetadata,
}

impl ImageReqWire {
    /// `ImageRequest::Register`.
    pub const OP_REGISTER: u8 = 1;
    /// `ImageRequest::Acquire`.
    pub const OP_ACQUIRE: u8 = 2;
    /// `ImageRequest::Retire`.
    pub const OP_RETIRE: u8 = 3;
    /// `ImageRequest::FaceMint` (slice C) — validate the executable named by
    /// `name` and mint one **per-child loader face**: claim a face slot,
    /// format its ring, mint its grant channels, and attach a fresh serving
    /// pair. The reply ([`TAG_FACE`](ImageRepWire::TAG_FACE)) carries the
    /// validated descriptor, face id, and complete client authority: ring
    /// sub-lease followed by one recv endpoint per image-segment channel.
    pub const OP_FACE_MINT: u8 = 4;
    /// `ImageRequest::FaceBank` — collect a minted face's grant-channel
    /// recv ends: `slot` = the face id; the reply
    /// ([`TAG_BANKED`](ImageRepWire::TAG_BANKED)) rides beside
    /// [`MAX_SEGMENTS`-many](crate::image::MAX_SEGMENTS) grants (one fresh
    /// recv per channel). Retained as an append-only legacy verb; current
    /// `FaceMint` banks atomically, so a re-bank refuses.
    pub const OP_FACE_BANK: u8 = 5;
    /// `ImageRequest::FaceRetire` — retire a minted face: `slot` = the
    /// face id; the service revokes the face's channel roots (T6 closes
    /// any surviving recv leaves anywhere), detaches its serving pair,
    /// and frees the slot for a later mint.  Parent-driven: a leak is
    /// bounded by the parent's own face quota, visible in the image
    /// row's summarize.
    pub const OP_FACE_RETIRE: u8 = 6;
    /// `ImageRequest::Describe` — return the newest grantable image's
    /// validated descriptor without transferring any segment capability.
    /// This is metadata discovery only; consumers must still `Acquire`
    /// before using image bytes.
    pub const OP_DESCRIBE: u8 = 7;
    /// Retired name-selected executable acquisition; frozen refusal.
    pub const OP_EXECUTABLE_AT: u8 = 8;
    /// Retired ordinal-bound executable acquisition; frozen refusal.
    pub const OP_ACQUIRE_BOUND: u8 = 9;
    /// Retired executable-receiver acquisition; frozen refusal.
    pub const OP_EXECUTABLE_RECV_AT: u8 = 10;
    /// Acquire the immutable entry bound to the arrival face. The request
    /// carries no name, slot, or other selector. Keeps its original `11`;
    /// `8`/`9`/`10` carried the retired name/ordinal acquisition family and
    /// stay frozen refusals.
    pub const OP_ACQUIRE_ENTRY: u8 = 11;
    /// Mint a fresh loader face bound to the same immutable generation as a
    /// privileged control-owned template. `slot` is private correlation.
    pub const OP_FACE_FORK: u8 = 12;
    /// Retire the exact immutable generation bound to a privileged template
    /// face. `slot` is service-private correlation and is accepted only on
    /// the owner/control pair; it is never ordinary executable authority.
    pub const OP_RETIRE_BOUND: u8 = 13;
    /// `ImageRequest::Export` — grant one read-only lease over the newest
    /// generation's whole member window, so a holder can copy the exact
    /// registered bytes elsewhere (the filesystem seed). This is the same
    /// span the segment carves come from, at read-only rights and without
    /// the executable blessing: it conveys bytes, never the authority to
    /// run them.
    pub const OP_EXPORT: u8 = 14;

    /// Every retired `ImageReqWire` discriminant, in the one definition the
    /// decoder cites (the `op_retired` discipline shared with `TaskReqWire`).
    pub const fn op_retired(op: u8) -> bool {
        op >= Self::OP_EXECUTABLE_AT && op <= Self::OP_EXECUTABLE_RECV_AT
    }
}

/// The image reply wire form (`ImageResponse`'s crossing shape).  The
/// [`ModuleDescriptor`] is the data half (what the service *says*);
/// authority is the ≤ `MAX_SEGMENTS` lease-cap transfers `grants`
/// announces, landing in the client's pre-pinned reply posts (§5.5) —
/// never descriptor fields.  On `Retired`/`Refused` the descriptor
/// rides [`empty`](ModuleDescriptor::empty) (its `abi` stamp is zero,
/// so it refuses validation — never mistaken for a promise).
/// 496 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct ImageRepWire {
    /// Outcome discriminant ([`TAG_REGISTERED`](Self::TAG_REGISTERED)…).
    pub tag: u8,
    /// `Refused`: the [`ImageRefusal`](crate::image::ImageRefusal)
    /// discriminant; zero otherwise.
    pub refusal: u8,
    /// Grant-bearing transfers this reply was effected with (`Acquired`:
    /// one per segment lease, ≤ [`MAX_SEGMENTS`](crate::image::MAX_SEGMENTS)).
    pub grants: u8,
    /// Reserved; must be zero.
    pub reserved: u8,
    /// `Face` ([`TAG_FACE`](Self::TAG_FACE)): the minted face's id — what
    /// [`OP_FACE_BANK`](ImageReqWire::OP_FACE_BANK) /
    /// [`OP_FACE_RETIRE`](ImageReqWire::OP_FACE_RETIRE) later cite (carved
    /// from the frozen reserved headroom, slice C); zero otherwise.
    pub face: U32,
    /// `Face`: stable service-authored identity of the immutable image
    /// generation selected by the face. Zero for every other response. This
    /// is correlation metadata for privileged service composition; it is not
    /// accepted by any ordinary acquisition or execution request.
    pub generation: U64,
    /// `Registered`/`Acquired`/`Described`: the module's frozen descriptor.
    pub module: ModuleDescriptor,
}

impl ImageRepWire {
    /// `ImageResponse::Registered`.
    pub const TAG_REGISTERED: u8 = 1;
    /// `ImageResponse::Acquired` — grants ride the reply posts.
    pub const TAG_ACQUIRED: u8 = 2;
    /// `ImageResponse::Retired` — and a retired *face*
    /// ([`OP_FACE_RETIRE`](ImageReqWire::OP_FACE_RETIRE)) rides the same
    /// tag: one outcome word for "the named thing is gone".
    pub const TAG_RETIRED: u8 = 3;
    /// `ImageResponse::Refused` — see `refusal`
    /// ([`ImageRefusal`](crate::image::ImageRefusal)'s own 1-based
    /// discriminants ride verbatim).
    pub const TAG_REFUSED: u8 = 4;
    /// `ImageResponse::Face` — a loader face was minted; `face` carries its
    /// id, `module` carries the validated initial executable, and the ring
    /// client sub plus every recv endpoint ride the reply posts.
    pub const TAG_FACE: u8 = 5;
    /// `ImageResponse::Banked` — the face's grant-channel recv ends ride
    /// the reply posts, one per channel
    /// ([`MAX_SEGMENTS`](crate::image::MAX_SEGMENTS)-many).
    pub const TAG_BANKED: u8 = 6;
    /// `ImageResponse::Described` — a validated descriptor with no grants.
    pub const TAG_DESCRIBED: u8 = 7;
    /// One image-bound executable endpoint was provisioned; `face` repeats
    /// the metadata ordinal and exactly one `ChanSend` grant rides the reply.
    pub const TAG_EXECUTABLE: u8 = 8;
    /// Exec's one receive face for an executable binding; `face` repeats the
    /// metadata ordinal and exactly one `ChanRecv` grant rides the reply.
    pub const TAG_EXECUTABLE_RECV: u8 = 9;
    /// `ImageResponse::Exported` — one read-only lease over the member
    /// window rides beside it; `generation` carries the exact registered
    /// byte length (the lease itself spans the page-rounded window).
    ///
    /// Past the executable pair deliberately: 8 and 9 are theirs, and a
    /// second meaning on a live tag is an ambiguous record, not a spare slot.
    pub const TAG_EXPORTED: u8 = 10;
}

const _: () = {
    // Reply tags name one outcome each. A duplicate would decode one
    // response as another and no test could catch it at the far end.
    let tags = [
        ImageRepWire::TAG_REGISTERED,
        ImageRepWire::TAG_ACQUIRED,
        ImageRepWire::TAG_RETIRED,
        ImageRepWire::TAG_REFUSED,
        ImageRepWire::TAG_FACE,
        ImageRepWire::TAG_BANKED,
        ImageRepWire::TAG_DESCRIBED,
        ImageRepWire::TAG_EXECUTABLE,
        ImageRepWire::TAG_EXECUTABLE_RECV,
        ImageRepWire::TAG_EXPORTED,
    ];
    let mut i = 0;
    while i < tags.len() {
        let mut j = i + 1;
        while j < tags.len() {
            assert!(tags[i] != tags[j], "image reply tags must be distinct");
            j += 1;
        }
        i += 1;
    }
};

// ─── input — the multiplexing router (kore-svc-input) ───────────────

/// One decoded HID boot report, homed on a `ring::spsc<InputReport>` in the
/// app-allocated report window (`res/USERLAND.md` §5.8 — the #299 "KINP"
/// queue generalized onto Phase A's spsc, which names this its first
/// production instance).  A **data-plane record**, not a control reply: no
/// capability ever rides it (§3.2), and it reinterprets from any bytes
/// (the window's peer is untrusted) — the reader's `kind` gate is the
/// refusal.  The 16-byte layout is #299's proven one, preserved verbatim:
/// `kind ‖ source ‖ state ‖ data[6] ‖ reserved[7]`, keyboard usage codes
/// or the mouse `[dx, dy, wheel]` packed in `data`.  16 bytes.
///
/// The compositor lane also carries [`crate::pointer::Record`] envelopes in
/// the reserved kind band 0x10..=0x1f. Their versioned codec owns all 14 payload
/// bytes; the legacy keyboard/mouse/contact layouts remain unchanged. Pointer
/// consumers must assemble and validate a whole envelope before applying it.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct InputReport {
    /// Report kind: [`KIND_KEYBOARD`](Self::KIND_KEYBOARD) /
    /// [`KIND_MOUSE`](Self::KIND_MOUSE) /
    /// [`KIND_CONTACT`](Self::KIND_CONTACT); zero is an empty/invalid slot
    /// (the reader drops it — the 1-based-discriminant discipline).
    pub kind: u8,
    /// The stable input-source identity. xHCI reports use the 1-based
    /// root-hub port; the legacy keyboard uses its reserved `0x40` band.
    /// This is useful when an app intentionally distinguishes devices
    /// (#299's `source`) without depending on enumeration-order slot ids.
    pub source: u8,
    /// Keyboard: the HID modifier bitmap.  Mouse: the button bitmap.
    pub state: u8,
    /// Keyboard: up to six pressed HID usage codes (0 = empty).  Mouse:
    /// `[dx, dy, wheel, 0, 0, 0]` (signed deltas as `u8`).
    pub data: [u8; 6],
    /// Zero for legacy records. Contacts may carry version-1 axis metadata:
    /// `[1, class, max_x_le[2], max_y_le[2], 0]`. Coordinates are normalized
    /// to a zero minimum by the descriptor decoder, not scaled to a display.
    pub reserved: [u8; 7],
}

impl InputReport {
    pub const CONTACT_TOUCHPAD: u8 = 1;
    pub const CONTACT_TOUCHSCREEN: u8 = 2;
    pub const CONTACT_PEN: u8 = 3;

    /// Attach descriptor facts without changing the frozen 16-byte record.
    pub const fn with_contact_axes(mut self, class: u8, max_x: u16, max_y: u16) -> Self {
        self.reserved = [
            1,
            class,
            max_x as u8,
            (max_x >> 8) as u8,
            max_y as u8,
            (max_y >> 8) as u8,
            0,
        ];
        self
    }

    /// Known device class and inclusive axis maxima. Unknown versions/ranges
    /// are explicitly absent; a consumer must not guess touchscreen semantics.
    pub fn contact_axes(&self) -> Option<(u8, u16, u16)> {
        let [version, class, x0, x1, y0, y1, flags] = self.reserved;
        let x = u16::from_le_bytes([x0, x1]);
        let y = u16::from_le_bytes([y0, y1]);
        (self.kind == Self::KIND_CONTACT
            && version == 1
            && (1..=3).contains(&class)
            && flags == 0
            && x != 0
            && y != 0)
            .then_some((class, x, y))
    }

    /// A keyboard-state report (`InputReport::Keyboard`, #299).
    pub const KIND_KEYBOARD: u8 = 1;
    /// A mouse-movement report (`InputReport::Mouse`, #299).
    pub const KIND_MOUSE: u8 = 2;
    /// One **absolute contact** from a digitizer — the touchpad and
    /// touchscreen shape (epic #400's pointer seam).
    ///
    /// Distinct from [`KIND_MOUSE`](Self::KIND_MOUSE) because the two
    /// carry genuinely different physics and collapsing them would lose
    /// information the consumer needs: a mouse reports *relative* deltas
    /// already scaled by the device, a digitizer reports an *absolute*
    /// position in its own logical units together with which finger this
    /// is and how many are down. Cursor motion is derived from
    /// successive contacts by the consumer, which is the only layer that
    /// knows the surface geometry and the pointer acceleration it wants —
    /// synthesizing deltas here would bake one policy into the wire.
    ///
    /// A Windows Precision Touchpad usually publishes *both* a mouse
    /// collection and a digitizer collection; a driver that finds the
    /// former may emit [`KIND_MOUSE`](Self::KIND_MOUSE) and never reach
    /// for this kind. This exists so the digitizer-only device is
    /// expressible rather than unsupported.
    pub const KIND_CONTACT: u8 = 3;

    /// [`KIND_CONTACT`](Self::KIND_CONTACT) `state`: the tip switch —
    /// the contact is actually touching the surface, not merely hovering
    /// in range.
    pub const CONTACT_TIP: u8 = 1 << 0;
    /// [`KIND_CONTACT`](Self::KIND_CONTACT) `state`: the digitizer's
    /// confidence bit — it believes this contact is an intentional touch
    /// rather than a palm. A consumer that ignores this reports palm
    /// rests as cursor jumps.
    pub const CONTACT_CONFIDENCE: u8 = 1 << 1;
    /// [`KIND_CONTACT`](Self::KIND_CONTACT) `state`: the contact is in
    /// range of the sensor (hovering) even if [`CONTACT_TIP`] is clear.
    ///
    /// [`CONTACT_TIP`]: Self::CONTACT_TIP
    pub const CONTACT_IN_RANGE: u8 = 1 << 2;
    /// [`KIND_CONTACT`](Self::KIND_CONTACT) `state`: the digitizer's
    /// button (a click on a clickpad's hinge, or a separate button).
    pub const CONTACT_BUTTON: u8 = 1 << 3;

    /// A keyboard-state report: modifier bitmap + six-key rollover.
    pub const fn keyboard(source: u8, modifiers: u8, keys: [u8; 6]) -> Self {
        Self {
            kind: Self::KIND_KEYBOARD,
            source,
            state: modifiers,
            data: keys,
            reserved: [0; 7],
        }
    }

    /// A mouse report: button bitmap + signed relative deltas.
    pub const fn mouse(source: u8, buttons: u8, dx: i8, dy: i8, wheel: i8) -> Self {
        Self {
            kind: Self::KIND_MOUSE,
            source,
            state: buttons,
            data: [dx as u8, dy as u8, wheel as u8, 0, 0, 0],
            reserved: [0; 7],
        }
    }

    /// One absolute contact: `state` flags (`CONTACT_*`), an absolute
    /// position in the digitizer's own logical units, which contact this
    /// is, and how many are down in the same frame.
    ///
    /// `x`/`y` ride `data` as **little-endian `u16`** rather than as
    /// [`U16`] fields, because `data` is the frozen #299 byte array and
    /// widening it would move the 16-byte layout every other kind
    /// already depends on. The pair of accessors below is the only
    /// sanctioned way to read them back, so the packing lives in exactly
    /// one place.
    ///
    /// `count` is the *total* contacts in this frame, not an index: a
    /// consumer needs it to know when it has seen the whole frame, and
    /// a digitizer that reports one contact per report otherwise gives
    /// no way to distinguish "one finger" from "the first of three".
    pub const fn contact(source: u8, state: u8, x: u16, y: u16, contact_id: u8, count: u8) -> Self {
        Self {
            kind: Self::KIND_CONTACT,
            source,
            state,
            data: [
                x as u8,
                (x >> 8) as u8,
                y as u8,
                (y >> 8) as u8,
                contact_id,
                count,
            ],
            reserved: [0; 7],
        }
    }

    /// A [`KIND_CONTACT`](Self::KIND_CONTACT) report's absolute X, in the
    /// digitizer's logical units. Meaningless on any other kind — the
    /// caller's `kind` gate is the guard, as it is for every field here.
    #[inline]
    pub const fn contact_x(&self) -> u16 {
        self.data[0] as u16 | ((self.data[1] as u16) << 8)
    }

    /// A [`KIND_CONTACT`](Self::KIND_CONTACT) report's absolute Y.
    #[inline]
    pub const fn contact_y(&self) -> u16 {
        self.data[2] as u16 | ((self.data[3] as u16) << 8)
    }

    /// Which contact this is, within the frame.
    #[inline]
    pub const fn contact_id(&self) -> u8 {
        self.data[4]
    }

    /// How many contacts are down in this frame.
    #[inline]
    pub const fn contact_count(&self) -> u8 {
        self.data[5]
    }
}

/// The input request wire form (`InputRequest`'s crossing shape).  The
/// **caller identity is not here** (`res/USERLAND.md` §5.8): `SetFocus`
/// privilege rides the *endpoint the request arrives on*, stamped by the
/// router — a subscriber cannot express it on the wire.  `Subscribe`
/// names the delivered report-window view's arena slot (#148); no
/// capability ever rides this record.  12 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct InputReqWire {
    /// Verb discriminant: [`OP_SUBSCRIBE`](Self::OP_SUBSCRIBE) /
    /// [`OP_UNSUBSCRIBE`](Self::OP_UNSUBSCRIBE) /
    /// [`OP_SET_FOCUS`](Self::OP_SET_FOCUS).
    pub op: u8,
    /// `SetFocus`: [`FOCUS_NONE`](Self::FOCUS_NONE) (steer to nobody) or
    /// [`FOCUS_ONE`](Self::FOCUS_ONE) (steer to `subscription`); zero
    /// otherwise.
    pub focus: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 2],
    /// `Subscribe`: the delivered report-window view's service-arena slot.
    pub slot: U32,
    /// `Unsubscribe` / `SetFocus(one)`: the target `SubscriptionId`.
    pub subscription: U32,
}

impl InputReqWire {
    /// Indexed read-only input diagnostics; slot carries the row, other operands zero.
    /// Original I2C rows retain their meanings; later rows cover USB and admission.
    pub const OP_DIAGNOSTIC: u8 = 5;
    pub const OP_I2C_STATUS: u8 = Self::OP_DIAGNOSTIC;
    /// `InputRequest::Subscribe`.
    pub const OP_SUBSCRIBE: u8 = 1;
    /// `InputRequest::Unsubscribe`.
    pub const OP_UNSUBSCRIBE: u8 = 2;
    /// `InputRequest::SetFocus`.
    pub const OP_SET_FOCUS: u8 = 3;
    /// `InputRequest::Status` — read the service's own books. Carries no
    /// operand: it steers nothing and names no subscription.
    pub const OP_STATUS: u8 = 4;

    /// `SetFocus`: steer subsequent reports to nobody.
    pub const FOCUS_NONE: u8 = 0;
    /// `SetFocus`: steer subsequent reports to `subscription`.
    pub const FOCUS_ONE: u8 = 1;
}

/// The input reply wire form (`InputResponse`'s crossing shape).  Data
/// only — a `SubscriptionId` is service-private correlation data, never a
/// capability, and an input service grants nothing, ever (`grants` is
/// always zero). 84 bytes — the legacy counters, xHCI census, and two compact
/// I2C worker snapshots ride after the `Subscribed` correlation word. The
/// size is asserted at the top of this module rather than argued from a
/// sibling's, which is how it came to claim a stale one.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct InputRepWire {
    /// Outcome discriminant ([`TAG_SUBSCRIBED`](Self::TAG_SUBSCRIBED)…).
    pub tag: u8,
    /// `Refused`: the `InputRefusal` discriminant; zero otherwise.
    pub refusal: u8,
    /// Always zero — an input service can mint no authority.
    pub grants: u8,
    /// `Status`: the legacy relay's bring-up rung (`0` = running). Zero on
    /// every other tag, which is exactly what the old `reserved` byte
    /// required, so no existing tag's bytes changed meaning.
    pub rung: u8,
    /// `Subscribed`: the assigned `SubscriptionId` (correlation data).
    pub subscription: U32,
    /// `Status`: line interrupts the relay woke on.
    pub wakes: U32,
    /// `Status`: parks the relay issued on the line — `wakes`'s
    /// denominator, and the witness that separates a relay serviced by
    /// its backstop from one that has stopped parking at all.
    pub parks: U32,
    /// `Status`: reports the relay pushed into the handoff queue.
    pub events: U32,
    /// `Status`: reports the handoff queue had no room for.
    pub dropped: U32,
    /// `Status`: bytes that decoded to a key with no HID meaning — the
    /// wrong-scancode-set witness, since a mismatched set decodes to
    /// nothing else.
    pub unnamed: U32,
    /// `Status`: `1` when the router owns a live xHCI controller.
    pub xhci_bound: U32,
    /// `Status`: packed xHCI intake refusal (`stage << 56 | diagnostic`).
    /// Zero means binding completed.
    pub xhci_bind: U64,
    /// `Status`: HID boot sources armed behind xHCI.
    pub xhci_sources: U32,
    /// `Status`: xHCI discovery/arming failures.
    pub xhci_discovery_errors: U32,
    /// `Status`: xHCI runtime completion/decode/re-arm failures.
    pub xhci_irq_errors: U32,
    /// `Status`: packed state for firmware HID-over-I2C workers 0 and 1.
    pub i2c_state: [U64; 2],
    /// `Status`: completed I2C transfers, saturated to `u32`.
    pub i2c_transfers: [U32; 2],
    /// `Status`: decoded reports, saturated to `u32`.
    pub i2c_reports: [U32; 2],
}

impl InputRepWire {
    /// Indexed diagnostic row: rung=state, subscription=packed identity, remaining
    /// five words=counters. Existing reply tags and record size are unchanged.
    pub const TAG_DIAGNOSTIC: u8 = 6;
    pub const TAG_I2C_STATUS: u8 = Self::TAG_DIAGNOSTIC;
    /// `InputResponse::Subscribed`.
    pub const TAG_SUBSCRIBED: u8 = 1;
    /// `InputResponse::Unsubscribed`.
    pub const TAG_UNSUBSCRIBED: u8 = 2;
    /// `InputResponse::Focused`.
    pub const TAG_FOCUSED: u8 = 3;
    /// `InputResponse::Refused` — see `refusal`.
    pub const TAG_REFUSED: u8 = 4;
    /// `InputResponse::Status` — see `rung` and the four counters.
    pub const TAG_STATUS: u8 = 5;

    /// `InputRefusal::BadQueue` — the delivered slot was absent,
    /// non-writable, or not a well-formed report ring.
    pub const REFUSE_BAD_QUEUE: u8 = 1;
    /// `InputRefusal::TableFull` — the bounded subscriber table is full.
    pub const REFUSE_TABLE_FULL: u8 = 2;
    /// `InputRefusal::Unknown` — the routing key is stale or unknown, or
    /// is not owned by the requesting endpoint.
    pub const REFUSE_UNKNOWN: u8 = 3;
    /// `InputRefusal::Denied` — the endpoint lacks the authority for the
    /// verb (an ordinary subscriber attempting `SetFocus`, §5.8).
    pub const REFUSE_DENIED: u8 = 4;
}

// ─── hypervisor — the userland VMM (kore-svc-hypervisor) ───────────

/// The hypervisor request wire form (`HypervisorRequest`'s crossing
/// shape) — the #385 scaffold (#482), grown the #488 console rail.
///
/// One byte of vocabulary and seven of reserve, so the record is the same
/// 8-byte shape as its siblings.  Discriminants are **1-based**: an
/// all-zero (unseeded) record is invalid, never a verb.
///
/// **The console data plane is not a verb.**  Guest console bytes ride
/// two planes appended to this pair's window (the display commit-plane
/// discipline): an event channel the service produces (guest → terminal —
/// a [`HypervisorConsoleState`] level beside an [`OutChunk`] record ring,
/// [`HYPERVISOR_CONSOLE_OUT_CAP`]) and an [`InChunk`] SPSC plane the client
/// produces (terminal → guest, [`HYPERVISOR_CONSOLE_IN_CAP`]).  The verbs
/// below only found a guest and route its console; the event's wake word is
/// the client's wait and its `halted` level the terminal edge.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct HypervisorReqWire {
    pub op: u8,
    pub reserved: [u8; 7],
}

impl HypervisorReqWire {
    /// `HypervisorRequest::Ping` — the liveness probe.
    pub const OP_PING: u8 = 1;
    /// `HypervisorRequest::SpawnEcho` — found and start the embedded echo
    /// guest (#486's intermediate console witness).  Refused
    /// [`HypervisorRepWire::TAG_VM_BUSY`] while any guest is live or the
    /// startup acceptance lifecycle has not completed.
    pub const OP_SPAWN_ECHO: u8 = 2;
    /// `HypervisorRequest::ConsoleWait` — an immediate terminal-state query,
    /// answered [`HypervisorRepWire::TAG_CONSOLE`]. Console waiters use the
    /// output event plane; data never rides this compatibility reply.
    pub const OP_CONSOLE_WAIT: u8 = 3;
    /// `HypervisorRequest::SpawnHlt` — found and start the acceptance-HLT
    /// guest (one `hlt` at entry) as a *client* guest: it reaches terminal
    /// on its own with no input, so the relay's exit edge
    /// ([`HypervisorRepWire::CONSOLE_HALTED`]) is exercised without the
    /// keystroke round-trip. Same one-guest / post-acceptance gating as
    /// [`OP_SPAWN_ECHO`](Self::OP_SPAWN_ECHO).
    pub const OP_SPAWN_HLT: u8 = 4;
    /// `HypervisorRequest::SpawnCompiled` — found the freestanding-toolchain
    /// witness (`blob::compiled_witness_package`): a `zig cc`-built guest that
    /// prints its marker and halts. Proves the compiled-C image path the real
    /// MicroPython payload rides. Same gating as [`OP_SPAWN_ECHO`].
    pub const OP_SPAWN_COMPILED: u8 = 5;
    /// `HypervisorRequest::SpawnMultiseg` — found the multi-segment witness
    /// (`blob::multiseg_witness_package`): a compiled guest with R|W .data and
    /// NOLOAD .bss, proving the loader carries every segment shape MicroPython
    /// has. Same gating as [`OP_SPAWN_ECHO`].
    pub const OP_SPAWN_MULTISEG: u8 = 6;
    /// `HypervisorRequest::SpawnMpy` — found the MicroPython guest
    /// (`blob::micropython_package`), the #385 payload: a full interpreter
    /// with an interactive REPL over the console rail. Same gating as
    /// [`OP_SPAWN_ECHO`].
    pub const OP_SPAWN_MPY: u8 = 7;
    /// `HypervisorRequest::AttachConsole` — bind the caller's existing
    /// console planes to the live guest or a terminal guest's pending replay.
    /// This never starts, resumes, or otherwise disturbs the vCPU.
    pub const OP_ATTACH_CONSOLE: u8 = 8;
    /// `HypervisorRequest::DetachConsole` — release the live console binding
    /// while leaving the guest, its vCPU, and its memory intact.
    pub const OP_DETACH_CONSOLE: u8 = 9;
    /// `HypervisorRequest::Stop` — request orderly retirement of the one live
    /// guest.  Completion is observable as the console event's `halted`
    /// level ([`HypervisorConsoleState`]); accepting this verb means
    /// retirement was queued.
    pub const OP_STOP: u8 = 10;
    /// `HypervisorRequest::Status` — one coherent, data-only snapshot of the
    /// service's single-guest and text-console books.
    pub const OP_STATUS: u8 = 11;
}

/// Capacity of the hypervisor pair window's **guest-output** console
/// plane (`event<HypervisorConsoleState, OutChunk>`, service-produced).
/// 64 chunks = 1.5 KiB of
/// text in flight — a whole marker burst without a single wake of the
/// consumer, while the plane stays comfortably inside the frozen 3-page
/// pair window beside its sibling.
pub const HYPERVISOR_CONSOLE_OUT_CAP: u32 = 64;

/// Capacity of the hypervisor pair window's **guest-input** console plane
/// (`spsc<InChunk>`, client-produced).  Sized as the out-plane: paste a
/// line, not a file — the UART model's own bounded queue is the real
/// pacing, and a full plane is honest backpressure on the producer.
pub const HYPERVISOR_CONSOLE_IN_CAP: u32 = 64;

/// Capacity of the coherent status-snapshot plane. The client issues one
/// synchronous Status request at a time, so two slots provide one live record
/// plus one generation of headroom without allocating.
pub const HYPERVISOR_STATUS_CAP: u32 = 2;

/// The hypervisor reply wire form (`HypervisorResponse`'s crossing
/// shape). `served` is **data** — a per-tag payload word — and the
/// hypervisor grants nothing, ever (`grants` is pinned zero; a non-zero
/// value fails the codec closed). 8 bytes. Status data uses the new additive
/// [`HypervisorStatusWire`] plane rather than resizing this frozen record.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct HypervisorRepWire {
    /// Outcome discriminant ([`TAG_PONG`](Self::TAG_PONG)…).
    pub tag: u8,
    /// Always zero — the hypervisor can mint no authority.
    pub grants: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 2],
    /// Per-tag payload: `Pong` = requests served so far including this
    /// one; `Console` = the halt flag ([`CONSOLE_HALTED`](Self::CONSOLE_HALTED));
    /// zero for the other tags.
    pub served: U32,
}

impl HypervisorRepWire {
    /// `HypervisorResponse::Pong` — the executor is alive; see `served`.
    pub const TAG_PONG: u8 = 1;
    /// `HypervisorResponse::Spawned` — the echo guest founded and started.
    pub const TAG_SPAWNED: u8 = 2;
    /// `HypervisorResponse::Console` — an immediate snapshot of the terminal
    /// level (`served` = [`CONSOLE_HALTED`](Self::CONSOLE_HALTED) when
    /// terminal), the reply to the reserved `OP_CONSOLE_WAIT`.  No client
    /// waits on it: the console event's wake word is the wait.
    pub const TAG_CONSOLE: u8 = 3;
    /// `HypervisorResponse::VmBusy` — `SpawnEcho` refused: a guest is
    /// live, or the startup acceptance lifecycle has not completed.
    pub const TAG_VM_BUSY: u8 = 4;
    /// `HypervisorResponse::SpawnFailed` — the founding itself refused
    /// (validate/carve/share/found): nothing was left half-assembled,
    /// and the failure is a diagnosis, never a retry hint.
    pub const TAG_SPAWN_FAILED: u8 = 5;
    /// `HypervisorResponse::Attached` — the console is now bound to the live
    /// stream or terminal replay; the guest was not restarted or changed.
    pub const TAG_ATTACHED: u8 = 6;
    /// `HypervisorResponse::Detached` — the console binding is gone while the
    /// guest remains live.
    pub const TAG_DETACHED: u8 = 7;
    /// `HypervisorResponse::StopQueued` — orderly guest retirement was queued.
    pub const TAG_STOP_QUEUED: u8 = 8;
    /// `HypervisorResponse::NoGuest` — no live guest or terminal replay exists.
    pub const TAG_NO_GUEST: u8 = 9;
    /// `HypervisorResponse::StopFailed` — the service could not queue or
    /// complete the requested retirement.
    pub const TAG_STOP_FAILED: u8 = 10;
    /// A coherent snapshot was published to the status plane.
    pub const TAG_STATUS: u8 = 11;
    /// The status plane could not accept a record; the client may retry after
    /// consuming any stale snapshot.
    pub const TAG_STATUS_UNAVAILABLE: u8 = 12;

    /// [`TAG_CONSOLE`](Self::TAG_CONSOLE)'s `served` value once the guest
    /// reached terminal and was reaped — the same level
    /// [`HypervisorConsoleState::halted`] publishes.  `0` = the guest is
    /// still running.
    pub const CONSOLE_HALTED: u32 = 1;
}

/// One coherent, data-only hypervisor snapshot. This is an additive SPSC
/// plane record, not an expansion of the frozen request/reply records. 72
/// bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct HypervisorStatusWire {
    /// Lifecycle state (`STATE_*`).
    pub state: u8,
    /// Embedded payload identity (`PAYLOAD_*`).
    pub payload: u8,
    /// `STATUS_*` posture bits plus the vCPU count.
    pub flags: U16,
    /// Reserved; must be zero.
    pub reserved: U32,
    /// Configured guest RAM bytes.
    pub ram_bytes: U64,
    /// Current or final guest uptime in nanoseconds.
    pub uptime_ns: U64,
    /// Guest exits observed by the service.
    pub exits: U64,
    /// Console bytes accepted from the shell.
    pub console_input_bytes: U64,
    /// Console bytes emitted by the guest.
    pub console_output_bytes: U64,
    /// Bytes in the detached recent-history tail.
    pub console_retained_bytes: U64,
    /// Oldest detached bytes evicted from that fixed tail.
    pub console_dropped_bytes: U64,
    /// Architecture-neutral terminal reason or bounded hardware fault detail.
    pub last_reason: U64,
}

/// Console event level state. Output records share this event's wake word;
/// consumed input bytes also advance it so a full input producer can resume.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct HypervisorConsoleState {
    pub halted: u8,
    pub reserved: [u8; 7],
    pub input_consumed: U64,
}

impl HypervisorStatusWire {
    /// No guest has run yet.
    pub const STATE_IDLE: u8 = 0;
    /// A guest is currently runnable or blocked on its emulated device.
    pub const STATE_RUNNING: u8 = 1;
    /// Orderly retirement has been requested.
    pub const STATE_STOPPING: u8 = 2;
    /// The most recent guest ended normally.
    pub const STATE_HALTED: u8 = 3;
    /// The current or most recent guest encountered a fault.
    pub const STATE_FAULTED: u8 = 4;

    /// No payload has run yet.
    pub const PAYLOAD_NONE: u8 = 0;
    /// The scalar UART echo witness.
    pub const PAYLOAD_ECHO: u8 = 1;
    /// The one-instruction HLT witness.
    pub const PAYLOAD_HLT: u8 = 2;
    /// The compiled C witness.
    pub const PAYLOAD_COMPILED: u8 = 3;
    /// The compiled multi-segment witness.
    pub const PAYLOAD_MULTISEG: u8 = 4;
    /// The MicroPython acceptance payload.
    pub const PAYLOAD_MICROPYTHON: u8 = 5;

    /// A live guest task currently exists.
    pub const STATUS_GUEST_EXISTS: u16 = 1 << 0;
    /// The shell's text-console relay is attached.
    pub const STATUS_CONSOLE_ATTACHED: u16 = 1 << 1;
    /// The startup hardware-virtualization witness completed; guest spawning
    /// is available on this machine.
    pub const STATUS_VIRT_READY: u16 = 1 << 2;
    /// vCPU count occupies bits 8..15 (v1 is exactly one).
    pub const STATUS_VCPU_SHIFT: u32 = 8;
    /// Mask selecting the status record's vCPU count.
    pub const STATUS_VCPU_MASK: u16 = 0xff << Self::STATUS_VCPU_SHIFT;
}

// ─── accel — the accelerator class protocol (kore-svc-accel) ───────────

/// One program binding of an accel `Submit` — which delivered buffer a
/// program slot binds, over which byte range, with which access.  The
/// `buffer` is the service-issued opaque object id (kind-tagged and
/// generational on the serving side); `slot` is the *program's* binding
/// slot number, whose meaning belongs to the loaded artifact.  32 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct AccelBindingWire {
    /// The bound buffer's object id (`AccelRepWire::TAG_BUFFER_MADE`'s
    /// `id`); zero is invalid in a live entry.
    pub buffer: U64,
    /// Byte offset into the buffer.
    pub offset: U64,
    /// Byte length (non-zero in a live entry).
    pub len: U64,
    /// The program's binding slot number.
    pub slot: U32,
    /// Access mode: 1 read, 2 write, 3 read-write (`AccessMode`'s wire
    /// values — protocol 1.0's encoding).
    pub access: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 3],
}

/// The accel request wire form (`AccelRequest`'s crossing shape) — the
/// accelerator-class control plane (`res/DEVICE_SERVICE.md` §7): one
/// vocabulary for every backend (NPU / GPU / CPU-SIMD), mirroring the
/// **virtio-accel protocol 1.0** object model (contexts, buffers,
/// programs, execution queues, submissions, events; the house protocol,
/// `MicroPerceptron/virtio-accel`) over the ordinary kore pair rings.
///
/// `op` selects the verb; `subject` names the verb's primary object id
/// (zero when the verb has none); `a..d` are per-verb words and unused
/// ones ride zero — a set unused field refuses at decode, the same
/// fail-closed posture as every sibling.  Bulk bytes (artifacts, tensor
/// data) never ride this record: they cross through the attached staging
/// window and the record carries only offsets into it.  288 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct AccelReqWire {
    /// Verb discriminant ([`OP_ATTACH`](Self::OP_ATTACH)…); zero
    /// (unseeded) is invalid.
    pub op: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 3],
    /// Protocol flags; reserved-zero in v1 (frozen additive headroom —
    /// a future revision can widen a verb behind a bit here without
    /// moving a byte).
    pub flags: U32,
    /// The verb's primary object id: the context of a create, the target
    /// of a destroy/transfer, the queue of a `Submit`.  Zero when the
    /// verb names no object (`Query`, `Signal`, `Attach`,
    /// `CreateContext`).
    pub subject: U64,
    /// Per-verb word (see the `OP_*` docs).
    pub a: U64,
    /// Per-verb word.
    pub b: U64,
    /// Per-verb word.
    pub c: U64,
    /// Per-verb word.
    pub d: U64,
    /// Op-specific 48-byte region, zero for verbs that name neither use:
    /// - `LoadProgram`: the artifact's opaque target-identity words
    ///   (`TargetIdentity`, 12 LE `u32`s — schema owned by the artifact
    ///   format, not this protocol).
    /// - `Submit`: the deployed legacy geometry in the first 28 bytes — grid
    ///   `[u32; 3]`, group `[u32; 3]`, then SIMD width `u32`, all LE; bytes
    ///   `[28..48]` stay zero. The all-one geometry is the artifact-defined
    ///   sentinel; every other value retains its exact legacy meaning.
    /// - `SubmitLaunch`: explicit workgroup geometry in the first 24 bytes —
    ///   group count `[u32; 3]`, then local size `[u32; 3]`, all LE; bytes
    ///   `[24..48]` stay zero. The uses are op-disjoint, so the record size
    ///   stays unchanged and legacy SIMD is never reinterpreted as portable
    ///   workgroup data.
    pub target: [u8; 48],
    /// `Submit`: bindings `[0..c]` live, the rest all-zero.
    pub bindings: [AccelBindingWire; MAX_ACCEL_BINDINGS],
}

/// Program bindings one `Submit` record can carry — the **wire** bound;
/// a backend may enforce a lower `max_bindings_per_submission`.  Growth
/// past this is additive (a reserved flag bit naming a staging-window
/// binding table), never a record move.
pub const MAX_ACCEL_BINDINGS: usize = 6;

impl AccelReqWire {
    /// `AccelRequest::Attach` — adopt the staging window transferred
    /// over the return channel; `a` = the return-landing correlation
    /// value ([`crate::session::SlotRef`] wire index).  Attach once —
    /// a re-attach replaces the window but spends the service's finite
    /// map cursor, and the service bounds how many it will accept.
    pub const OP_ATTACH: u8 = 1;
    /// `AccelRequest::Query` — the device's identity, capabilities, and
    /// limits ([`AccelRepWire::TAG_INFO`]).
    pub const OP_QUERY: u8 = 2;
    /// `AccelRequest::CreateContext` — `a` = context flags (all
    /// non-empty sets reserved in protocol 1.0).
    pub const OP_CREATE_CONTEXT: u8 = 3;
    /// `AccelRequest::DestroyContext` — `subject` = the context (refuses
    /// while children live).
    pub const OP_DESTROY_CONTEXT: u8 = 4;
    /// `AccelRequest::CreateBuffer` — `subject` = context, `a` = bytes,
    /// `b` = alignment, `c` = domain (1 host / 2 device / 3 shared)
    /// `| usage << 8` (the `BufferUsage` bits).
    pub const OP_CREATE_BUFFER: u8 = 5;
    /// `AccelRequest::FreeBuffer` — `subject` = the buffer.
    pub const OP_FREE_BUFFER: u8 = 6;
    /// `AccelRequest::WriteBuffer` — `subject` = buffer, `a` = buffer
    /// offset, `b` = staging offset, `c` = length.  Bytes move from the
    /// staging window into the buffer.
    pub const OP_WRITE_BUFFER: u8 = 7;
    /// `AccelRequest::ReadBuffer` — `subject` = buffer, `a` = buffer
    /// offset, `b` = staging offset, `c` = length.  Bytes move from the
    /// buffer into the staging window.
    pub const OP_READ_BUFFER: u8 = 8;
    /// `AccelRequest::LoadProgram` — `subject` = context, `a` = staging
    /// offset of the artifact bytes, `b` = artifact length, `c` =
    /// resident-byte budget the program may retain, `d` = artifact
    /// format tag (non-zero; `FORMAT_*` registry in `kore-svc-accel`),
    /// `target` = the format's target-identity words.
    pub const OP_LOAD_PROGRAM: u8 = 9;
    /// `AccelRequest::UnloadProgram` — `subject` = the program.
    pub const OP_UNLOAD_PROGRAM: u8 = 10;
    /// `AccelRequest::CreateQueue` — `subject` = context, `a` = queue
    /// flags (all non-empty sets reserved in protocol 1.0).
    pub const OP_CREATE_QUEUE: u8 = 11;
    /// `AccelRequest::DestroyQueue` — `subject` = the queue.
    pub const OP_DESTROY_QUEUE: u8 = 12;
    /// `AccelRequest::Submit` — `subject` = queue, `a` = program, `b` =
    /// relative timeout in ns (zero = infinite), `c` = binding count
    /// (`1..=`[`MAX_ACCEL_BINDINGS`]), `bindings[0..c]` live, and `target`'s
    /// first 28 bytes carry the deployed grid `[u32; 3]`, group `[u32; 3]`,
    /// and SIMD width `u32`. The scalar all-one geometry is the historical
    /// artifact-defined sentinel; non-scalar geometry reaches a compatibility
    /// backend unchanged and is rejected unless it can be honored exactly.
    /// Replies [`TAG_SUBMITTED`](AccelRepWire::TAG_SUBMITTED) on admission;
    /// completion rides the signal channel and
    /// [`OP_POLL_EVENT`](Self::OP_POLL_EVENT) stays the re-readable
    /// truth.
    pub const OP_SUBMIT: u8 = 13;
    /// `AccelRequest::PollEvent` — `subject` = the event.
    pub const OP_POLL_EVENT: u8 = 14;
    /// `AccelRequest::CancelEvent` — `subject` = the event (supported
    /// only when the device advertises event cancellation).
    pub const OP_CANCEL_EVENT: u8 = 15;
    /// `AccelRequest::DestroyEvent` — `subject` = the (terminal) event.
    pub const OP_DESTROY_EVENT: u8 = 16;
    /// `AccelRequest::Signal` — mint this pair's completion channel and
    /// grant its recv end (the reply's one retained grant).  The client
    /// parks on it; each notice says "poll your events".
    pub const OP_SIGNAL: u8 = 17;
    /// `AccelRequest::SubmitLaunch` — the same object and binding words as
    /// [`OP_SUBMIT`](Self::OP_SUBMIT), with explicit portable workgroup
    /// geometry in `target[0..24]`: group count `[u32; 3]`, then local size
    /// `[u32; 3]`, all LE. Every axis is nonzero and `target[24..48]` is zero.
    /// No SIMD/subgroup width is client-selected. Support is advertised by
    /// `contract::EXPLICIT_WORKGROUPS`; an older decoder rejects
    /// the unknown opcode instead of silently treating it as `Submit`.
    pub const OP_SUBMIT_LAUNCH: u8 = 18;
    /// Fund initial accelerator capacity: `a` = retained supply landing,
    /// `b` = buffer-budget increment, `c` = program-budget increment.
    /// Only before the first context request. All other words are zero.
    pub const OP_PROVISION_MEMORY: u8 = 19;
    /// Retain and exactly map a parent staging lease; `a` is the retained
    /// return landing. Other words are zero. Requires A/M/D/R lease authority.
    pub const OP_ATTACH_OWNED: u8 = 20;
    /// Retire this pair's owned staging mapping and parent; all words zero.
    pub const OP_DETACH_STAGING: u8 = 21;
    /// Founder conveys a peer component: a=landing, b=part (0 ring,1 kick,2 return,128+grant).
    pub const OP_ADMIT_PEER: u8 = 22;
    /// Consume completed admission result.
    pub const OP_TAKE_PEER_ADMISSION: u8 = 23;
    /// Close the authenticated recyclable arrival.
    pub const OP_CLOSE_PEER: u8 = 24;
    /// Owned DRAM import: subject=context, a=bytes, b=alignment, c=usage,
    /// d=authenticated return-landing correlation. Domain is Shared.
    pub const OP_IMPORT_BUFFER: u8 = 25;
    /// Return an imported allocation after quiescent buffer detachment; subject=buffer.
    pub const OP_EXPORT_BUFFER: u8 = 26;
    /// Administration pair: request retirement of the exact recyclable peer
    /// whose generation-stamped arrival is in `subject`.
    pub const OP_RETIRE_PEER: u8 = 27;
    /// Administration pair: fence the same peer's retirement before its
    /// launcher reclaims local backing and channel roots.
    pub const OP_CONFIRM_RETIRED: u8 = 28;
    /// `AccelRequest::ReadLog` — copy the tail of the provider's firmware
    /// log into the staging window: `b` = stage offset, `c` = capacity.
    pub const OP_READ_LOG: u8 = 29;
}

/// The accel reply wire form (`AccelResponse`'s crossing shape).  Data
/// only — the one authority-bearing reply
/// ([`TAG_SIGNALED`](Self::TAG_SIGNALED)) announces its grant count and
/// the recv end rides the grant channel, never these bytes.  96 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct AccelRepWire {
    /// Outcome discriminant ([`TAG_ATTACHED`](Self::TAG_ATTACHED)…);
    /// zero (unseeded) is invalid.
    pub tag: u8,
    /// Grants riding beside this reply — zero except
    /// [`TAG_SIGNALED`](Self::TAG_SIGNALED)'s one.
    pub grants: u8,
    /// The refusal code ([`REFUSE_UNSUPPORTED`](Self::REFUSE_UNSUPPORTED)…)
    /// when `tag` is [`TAG_REFUSED`](Self::TAG_REFUSED) or
    /// [`TAG_SUBMIT_INDETERMINATE`](Self::TAG_SUBMIT_INDETERMINATE), or
    /// the failure code of a [`TAG_EVENT`](Self::TAG_EVENT) whose
    /// `state` is failed; zero otherwise.
    pub refusal: u8,
    /// `TAG_EVENT`: the event's state — 1 pending, 2 complete, 3 failed
    /// (see `refusal`), 4 cancelled.  Zero for every other tag.
    pub state: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 4],
    /// The verb's object word: a create's fresh id, a destroy/transfer's
    /// echoed subject, a submit's event id, an attach's window byte
    /// length.  The echo is load-bearing on this uncorrelated pair: a
    /// reply naming a different subject than the outstanding request is
    /// refused by `reply_matches`, never trusted.
    pub id: U64,
    /// [`TAG_INFO`](Self::TAG_INFO)'s payload: uuid (16) ‖ class (2) ‖
    /// reserved (2) ‖ vendor (4) ‖ device (4) ‖ capabilities (8) ‖ the
    /// eight limit words (6×4 + 2×8). For an external refusal (including
    /// failed events and indeterminate submissions), bytes 0..4 hold the
    /// provider domain (u32 LE), 4..12 the code (i64 LE), and 12..80 are
    /// zero. Zero for every other tag/refusal. Old decoders reject the new
    /// refusal discriminant rather than silently discarding its detail.
    pub info: [u8; 80],
}

impl AccelRepWire {
    /// `AccelResponse::Attached` — the staging window is live; `id` is
    /// its byte length (read off the mapping, never trusted from the
    /// wire).
    pub const TAG_ATTACHED: u8 = 1;
    /// `AccelResponse::Info` — see `info`.
    pub const TAG_INFO: u8 = 2;
    /// `AccelResponse::ContextMade` — `id` is the fresh context.
    pub const TAG_CONTEXT_MADE: u8 = 3;
    /// `AccelResponse::BufferMade` — `id` is the fresh buffer.
    pub const TAG_BUFFER_MADE: u8 = 4;
    /// `AccelResponse::ProgramMade` — `id` is the fresh program.
    pub const TAG_PROGRAM_MADE: u8 = 5;
    /// `AccelResponse::QueueMade` — `id` is the fresh queue.
    pub const TAG_QUEUE_MADE: u8 = 6;
    /// `AccelResponse::Released` — `id` echoes the destroyed subject.
    pub const TAG_RELEASED: u8 = 7;
    /// `AccelResponse::Wrote` — `id` echoes the buffer.
    pub const TAG_WROTE: u8 = 8;
    /// `AccelResponse::Read` — `id` echoes the buffer; the bytes are in
    /// the staging window at the requested offset.
    pub const TAG_READ: u8 = 9;
    /// `AccelResponse::Submitted` — admitted; `id` is the event.
    pub const TAG_SUBMITTED: u8 = 10;
    /// `AccelResponse::SubmitIndeterminate` — acceptance is uncertain;
    /// `id` is the event that **owns the submission's resources** until
    /// it reaches a terminal state (protocol 1.0's acceptance boundary:
    /// an indeterminate failure still returns an event), `refusal` the
    /// error.
    pub const TAG_SUBMIT_INDETERMINATE: u8 = 11;
    /// `AccelResponse::Event` — `id` echoes the event; see `state`.
    pub const TAG_EVENT: u8 = 12;
    /// `AccelResponse::Cancelled` — cancellation won; `id` echoes the
    /// event (now terminal).
    pub const TAG_CANCELLED: u8 = 13;
    /// `AccelResponse::Signaled` — the completion channel's recv rides
    /// this reply's one retained grant.
    pub const TAG_SIGNALED: u8 = 14;
    /// `AccelResponse::Refused` — see `refusal`.
    pub const TAG_REFUSED: u8 = 15;
    /// Additional memory retained and activated; `id` is the supplied byte length.
    pub const TAG_PROVISIONED: u8 = 16;
    /// Owned staging mapping and parent retired; id and other words zero.
    pub const TAG_DETACHED: u8 = 17;
    /// Bundle component retained; no installed peer is promised.
    pub const TAG_PEER_ACCEPTED: u8 = 18;
    /// Installed peer; id is its nonzero-generation arrival stamp.
    pub const TAG_PEER_ADMITTED: u8 = 19;
    /// Orderly peer closure requested; other words zero.
    pub const TAG_PEER_CLOSING: u8 = 20;
    /// Import refused before publication; one retained lease grant accompanies it.
    pub const TAG_IMPORT_RETURNED: u8 = 21;
    /// One exact lease grant; id echoes the detached buffer.
    pub const TAG_BUFFER_EXPORTED: u8 = 22;
    /// The named recyclable peer is absent from the service namespace.
    pub const TAG_PEER_RETIRED: u8 = 23;
    /// `AccelResponse::Log` — `id` is how many log bytes now sit at the
    /// requested stage offset (zero when the provider keeps no log).
    pub const TAG_LOG: u8 = 24;

    /// The backend cannot perform the operation (unsupported verb,
    /// domain, flag set, or cancellation without the capability).
    pub const REFUSE_UNSUPPORTED: u8 = 1;
    /// The pieces are individually valid but cannot compose (objects
    /// from different contexts, an allocation short of its descriptor).
    pub const REFUSE_INCOMPATIBLE: u8 = 2;
    /// A malformed argument (zero length, bad alignment, duplicate
    /// binding slot, unknown domain/access value).
    pub const REFUSE_INVALID_ARGUMENT: u8 = 3;
    /// A range escapes its buffer or the staging window.
    pub const REFUSE_OUT_OF_BOUNDS: u8 = 4;
    /// The object is busy (children live, submissions in flight, a
    /// pending event's destroy, cancellation racing completion).
    pub const REFUSE_BUSY: u8 = 5;
    /// The provider could not allocate.
    pub const REFUSE_OUT_OF_MEMORY: u8 = 6;
    /// A device limit or the service's retained-byte budget is exhausted.
    pub const REFUSE_RESOURCE_LIMIT: u8 = 7;
    /// The submission's relative deadline expired.
    pub const REFUSE_DEADLINE_EXPIRED: u8 = 8;
    /// The device is lost; every later verb refuses the same way until
    /// the row is re-founded (a reset vocabulary is deliberately
    /// deferred).
    pub const REFUSE_DEVICE_LOST: u8 = 9;
    /// A binding's access mode exceeds the buffer's declared usage (the
    /// host-side Wire ABI §4.4 gate).
    pub const REFUSE_PERMISSION_DENIED: u8 = 10;
    /// A provider- or engine-internal fault with no portable meaning.
    pub const REFUSE_INTERNAL: u8 = 11;
    /// The id names no live object: never issued, stale (generational),
    /// the wrong kind, or a different device epoch.
    pub const REFUSE_BAD_OBJECT: u8 = 12;
    /// No staging window is attached, or the addressed range lies
    /// outside it.
    pub const REFUSE_NO_STAGING: u8 = 13;
    /// The object belongs to another client (the `Serve::via` stamp),
    /// or the verb is not this endpoint's to speak.
    pub const REFUSE_DENIED: u8 = 14;
    /// The completion-channel mint refused (no channel space endowed, or
    /// the mint itself failed) — polling remains lawful.
    pub const REFUSE_NO_CHANNEL: u8 = 15;
    /// The row hosts no live engine (the degraded founding posture: the
    /// device absent or its metadata refused validation).
    pub const REFUSE_NOT_READY: u8 = 16;
    /// Opaque provider error; domain/code occupy the first 12 `info` bytes.
    pub const REFUSE_EXTERNAL: u8 = 17;
}

// Frozen wire pins (#81 discipline): a layout drift here is a protocol
// break between two *processes* — both sides read these bytes off a
// shared ring.

// ─── net — the L2 frame service (kore-svc-net) ──────────────────────

/// EtherTypes one vif filter may name on the wire — the wire mirror of
/// `kore_driver::net::filter::MAX_FILTER_ETHERTYPES`.
pub const MAX_NET_ETHERTYPES: usize = 4;

/// Frame-service request. Carries a vif's requested filter as data; the
/// authority to *have* that filter is the arrival pair's, never this record's.
/// 32 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct NetReqWire {
    /// Vocabulary discriminant ([`OP_QUERY`](Self::OP_QUERY)…); 0 invalid.
    pub op: u8,
    /// Filter: which EtherTypes — [`ET_ONLY`](Self::ET_ONLY) /
    /// [`ET_ANY`](Self::ET_ANY).
    pub ethertype_kind: u8,
    /// Filter: how many entries of [`ethertypes`](Self::ethertypes) are live.
    pub ethertype_count: u8,
    /// Filter: which multicast — [`MC_NONE`](Self::MC_NONE) /
    /// [`MC_GROUP`](Self::MC_GROUP) / [`MC_ALL`](Self::MC_ALL).
    pub multicast_kind: u8,
    /// Filter bits: [`FLAG_UNICAST`](Self::FLAG_UNICAST),
    /// [`FLAG_BROADCAST`](Self::FLAG_BROADCAST),
    /// [`FLAG_PROMISCUOUS`](Self::FLAG_PROMISCUOUS).
    pub flags: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 3],
    /// The vif this request names, for the verbs that name one.
    pub vif_index: U16,
    /// That vif's generation — a stale value is refused, never serviced.
    pub vif_generation: U16,
    /// `OpenVif`: the founder-published return-landing correlation slot the
    /// window rides in on.
    pub announced: U32,
    /// Filter: the accepted EtherTypes, host order, zero past the count.
    pub ethertypes: [U16; MAX_NET_ETHERTYPES],
    /// Filter: the joined multicast group, zero unless
    /// [`MC_GROUP`](Self::MC_GROUP).
    pub multicast_group: [u8; 6],
    /// Reserved; must be zero.
    pub reserved2: [u8; 2],
}

impl NetReqWire {
    /// Report link, address, geometry, and this pair's policy.
    pub const OP_QUERY: u8 = 1;
    /// Open a vif over the window riding the return leg.
    pub const OP_OPEN_VIF: u8 = 2;
    /// Close a vif and invalidate every identifier issued for it.
    pub const OP_CLOSE_VIF: u8 = 3;
    /// Mint this pair's receive doorbell.
    pub const OP_SIGNAL: u8 = 4;
    /// Report one vif's counters.
    pub const OP_STATS: u8 = 5;
    /// Report the bound radio's facts ([`NetRadioWire`]) — what its firmware
    /// and efuse said at bring-up. Answered whether or not the radio lends a
    /// frame path; refused `UNSUPPORTED` on a host with no radio.
    pub const OP_RADIO: u8 = 6;

    /// `ethertype_kind`: exactly the listed types.
    pub const ET_ONLY: u8 = 0;
    /// `ethertype_kind`: every EtherType — a capture power, separately granted.
    pub const ET_ANY: u8 = 1;

    /// `multicast_kind`: no multicast.
    pub const MC_NONE: u8 = 0;
    /// `multicast_kind`: one joined group.
    pub const MC_GROUP: u8 = 1;
    /// `multicast_kind`: every group.
    pub const MC_ALL: u8 = 2;

    /// Accept frames addressed to this vif's own address.
    pub const FLAG_UNICAST: u8 = 1 << 0;
    /// Accept broadcast.
    pub const FLAG_BROADCAST: u8 = 1 << 1;
    /// Accept frames addressed to any station — the capture bit.
    pub const FLAG_PROMISCUOUS: u8 = 1 << 2;
}

/// The bound radio's facts, as `TAG_RADIO` carries them: the chip cut, the
/// firmware the row verified and downloaded, the efuse's regulatory root, the
/// secure-boot selection, and the PHY shape the firmware answered at the
/// alive handshake. Every field is the value the bring-up ladder read —
/// nothing here is interpreted by the service, so a reader decodes against
/// the same upstream definitions the driver core cites. 36 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct NetRadioWire {
    /// The chip cut (`R_AX_SYS_CFG1[15:12]`).
    pub cut: u8,
    /// The firmware container's inner header version the download used.
    pub fw_header: u8,
    /// Firmware version, `w1[7:0]`.
    pub fw_major: u8,
    /// Firmware version, `w1[15:8]`.
    pub fw_minor: u8,
    /// Firmware version, `w1[23:16]`.
    pub fw_sub: u8,
    /// Firmware version, `w1[31:24]`.
    pub fw_idx: u8,
    /// The efuse country code: two ASCII letters, or `0xFF 0xFF` when
    /// unprogrammed. The regulatory root.
    pub country: [u8; 2],
    /// The efuse crystal trim.
    pub xtal_cap: u8,
    /// The efuse RF front-end type the firmware's tables are indexed by.
    pub rfe_type: u8,
    /// `RADIO_*` flag bits.
    pub flags: u8,
    /// Secure boot: the key-management device type (index form).
    pub mss_dev_type: u8,
    /// Secure boot: the customer index.
    pub mss_cust_idx: u8,
    /// Secure boot: the key number.
    pub mss_key_num: u8,
    /// PHY capability: receive spatial streams.
    pub rx_nss: u8,
    /// PHY capability: the widest bandwidth code.
    pub bw: u8,
    /// PHY capability: transmit spatial streams.
    pub tx_nss: u8,
    /// PHY capability: protocol capability code.
    pub prot: u8,
    /// PHY capability: NIC type code.
    pub nic: u8,
    /// PHY capability: WLAN function bits.
    pub wl_func: u8,
    /// PHY capability: hardware type code.
    pub hw_type: u8,
    /// PHY capability: transmit antennas.
    pub ant_tx: u8,
    /// PHY capability: receive antennas.
    pub ant_rx: u8,
    /// PHY capability: band selection code.
    pub band_sel: u8,
    /// Firmware version, `w2` (the commit identifier).
    pub fw_commit: U32,
    /// Secure boot: the key-management selector the image section matched.
    pub sb_sel_mgn: U32,
    /// Packets the download posted on the firmware-command channel
    /// (saturating; a download is on the order of a thousand).
    pub download_packets: U32,
}

impl NetRadioWire {
    /// The WCPU verifies the image against a key.
    pub const RADIO_SECURE_BOOT: u8 = 1 << 0;
    /// The MAC's own autoload from the efuse completed.
    pub const RADIO_EFUSE_AUTOLOAD: u8 = 1 << 1;
    /// Normal DMAC/CMAC0 configuration and the subsequent alive reply
    /// completed. Does not advertise PHY or frame-path readiness.
    pub const RADIO_MAC_READY: u8 = 1 << 2;
    /// BB/RF writes, PHY post-init and RF command TX retirement completed.
    /// Does not advertise calibration, channel or running BB-MCU readiness.
    pub const RADIO_PHY_CONFIGURED: u8 = 1 << 3;
    /// BA-CAM user reservations and offload configuration returned successful
    /// firmware done acknowledgements, with all command descriptors retired.
    pub const RADIO_MAC_FEATURES: u8 = 1 << 4;

    /// No radio: every field zero.
    pub const EMPTY: Self = Self {
        cut: 0,
        fw_header: 0,
        fw_major: 0,
        fw_minor: 0,
        fw_sub: 0,
        fw_idx: 0,
        country: [0; 2],
        xtal_cap: 0,
        rfe_type: 0,
        flags: 0,
        mss_dev_type: 0,
        mss_cust_idx: 0,
        mss_key_num: 0,
        rx_nss: 0,
        bw: 0,
        tx_nss: 0,
        prot: 0,
        nic: 0,
        wl_func: 0,
        hw_type: 0,
        ant_tx: 0,
        ant_rx: 0,
        band_sel: 0,
        fw_commit: U32::ZERO,
        sb_sel_mgn: U32::ZERO,
        download_packets: U32::ZERO,
    };
}

/// The frame device's counters, as `TAG_INFO` carries them and as the
/// launcher's [`NetSnapshot`](crate::monitor::NetSnapshot) stores them —
/// one definition, so the snapshot cannot drift from the reply.
///
/// Device-wide and cumulative since the row founded; every one saturates.
/// They name no vif and no client: what they answer is whether the
/// completion path is doing its job — interrupts held against frames, a
/// link that flapped, a receiver that ran dry — on a machine with no serial
/// console.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct NetCountersWire {
    /// Interrupts the driver serviced: drive cycles that found a cause
    /// latched (a message delivered, or several coalesced by moderation).
    pub irqs: U64,
    /// Link-state changes the device reported.
    pub link_changes: U32,
    /// Frames the device dropped for want of a posted receive buffer.
    pub overruns: U32,
    /// Receive descriptors or transfers the device completed in error.
    pub rx_errors: U32,
    /// Wake notices the service could not deliver (the transport refused);
    /// a wake lost, never a frame.
    pub dropped_notices: U32,
}

impl NetCountersWire {
    /// Nothing counted: what every tag but `TAG_INFO` carries.
    pub const EMPTY: Self = Self {
        irqs: U64::ZERO,
        link_changes: U32::ZERO,
        overruns: U32::ZERO,
        rx_errors: U32::ZERO,
        dropped_notices: U32::ZERO,
    };
}

/// Frame-service reply. A `Signaled` reply carries exactly one grant (the
/// receive doorbell's recv end); everything else carries none. 108 bytes:
/// the 48-byte vif/link head every earlier tag used, the 24-byte counters
/// block only `TAG_INFO` fills, then the radio block only `TAG_RADIO`
/// fills — each zero on every other tag, which the fail-closed re-encode
/// comparison enforces.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct NetRepWire {
    /// Vocabulary discriminant ([`TAG_INFO`](Self::TAG_INFO)…); 0 invalid.
    pub tag: u8,
    /// Grants riding beside this reply.
    pub grants: u8,
    /// Refusal code when `tag` is [`TAG_REFUSED`](Self::TAG_REFUSED).
    pub refusal: u8,
    /// Link: 0 down, 1 up.
    pub link: u8,
    /// The vif this reply names.
    pub vif_index: U16,
    /// That vif's generation, so a client records the identity it must use.
    pub vif_generation: U16,
    /// The vif's assigned address, or the device's on `TAG_INFO`.
    pub mac: [u8; 6],
    /// Largest frame the service carries.
    pub mtu: U16,
    /// Negotiated link rate in megabits per second; 0 when unknown.
    pub link_mbps: U32,
    /// Bytes a client must supply for one vif window.
    pub window_bytes: U32,
    /// Frames delivered to this vif.
    pub rx_frames: U64,
    /// Frames transmitted from it.
    pub tx_frames: U64,
    /// Frames dropped for it — a full ring, or fan-out exhaustion.
    pub dropped: U64,
    /// The device's counters on [`TAG_INFO`](Self::TAG_INFO); zero
    /// otherwise.
    pub counters: NetCountersWire,
    /// The bound radio's facts on [`TAG_RADIO`](Self::TAG_RADIO); zero
    /// otherwise.
    pub radio: NetRadioWire,
}

impl NetRepWire {
    /// Link, address, geometry, and the arrival pair's policy.
    pub const TAG_INFO: u8 = 1;
    /// A vif was opened; the reply names its identity and assigned address.
    pub const TAG_OPENED: u8 = 2;
    /// A vif was closed.
    pub const TAG_CLOSED: u8 = 3;
    /// The receive doorbell was minted; one grant rides along.
    pub const TAG_SIGNALED: u8 = 4;
    /// One vif's counters.
    pub const TAG_STATS: u8 = 5;
    /// Refused; see [`refusal`](Self::refusal).
    pub const TAG_REFUSED: u8 = 6;
    /// The bound radio's facts ride [`radio`](Self::radio).
    pub const TAG_RADIO: u8 = 7;

    /// The verb is not available in this posture.
    pub const REFUSE_UNSUPPORTED: u8 = 1;
    /// No device: the row founded degraded.
    pub const REFUSE_NOT_READY: u8 = 2;
    /// The request exceeded this pair's policy — see the specific codes.
    pub const REFUSE_POLICY_ETHERTYPE: u8 = 3;
    /// …a multicast group the pair may not join.
    pub const REFUSE_POLICY_MULTICAST: u8 = 4;
    /// …broadcast without the grant.
    pub const REFUSE_POLICY_BROADCAST: u8 = 5;
    /// …promiscuous receive without the grant.
    pub const REFUSE_POLICY_PROMISCUOUS: u8 = 6;
    /// …more vifs than the pair may hold.
    pub const REFUSE_POLICY_VIFS: u8 = 7;
    /// The vif identifier names no live vif.
    pub const REFUSE_NO_VIF: u8 = 8;
    /// The identifier is from before a close — a use-after-close, named
    /// distinctly so a service can say so rather than reporting a generic miss.
    pub const REFUSE_STALE_VIF: u8 = 9;
    /// The vif belongs to another pair.
    pub const REFUSE_NOT_OWNER: u8 = 10;
    /// The window was absent, malformed, or not a well-formed ring pair.
    pub const REFUSE_BAD_WINDOW: u8 = 11;
    /// The vif table is full.
    pub const REFUSE_EXHAUSTED: u8 = 12;
    /// A transfer is still outstanding, so the vif cannot be torn down yet.
    pub const REFUSE_BUSY: u8 = 13;
    /// The service could not make the window device-visible.
    pub const REFUSE_NO_MAPPING: u8 = 14;
    /// No channel space to mint a doorbell from.
    pub const REFUSE_NO_CHANNEL: u8 = 15;
    /// A refusal the service could not attribute.
    pub const REFUSE_INTERNAL: u8 = 16;
}

const _: () = {
    use core::mem::size_of;
    assert!(size_of::<NetReqWire>() == 32);
    assert!(size_of::<NetRadioWire>() == 36);
    assert!(size_of::<NetCountersWire>() == 24);
    // 48-byte vif/link head ‖ 24-byte counters ‖ 36-byte radio block.
    assert!(size_of::<NetRepWire>() == 48 + 24 + 36);
    // 16-byte head (op ‖ 3 filter kinds ‖ flags ‖ 3 reserved ‖ vif index+generation
    //     ‖ announced) ‖ 4×2 ethertypes ‖ 6-byte group ‖ 2 reserved.
    assert!(size_of::<NetReqWire>() == 16 + MAX_NET_ETHERTYPES * 2 + 6 + 2);
    assert!(size_of::<MmReqWire>() == 32);
    assert!(size_of::<MmRepWire>() == 72);
    assert!(size_of::<MonitorReqWire>() == 104);
    // 8-byte head ‖ 280-byte telemetry (240 + the 16-byte frame-class census
    // + the 24-byte DRAM stream table).
    assert!(
        size_of::<MonitorRepWire>()
            == 8 + 280
                + size_of::<crate::processor_status::ProcessorPage>()
                + size_of::<crate::device::DeviceCensusPage>()
    );
    assert!(size_of::<SwapReqWire>() == 16);
    assert!(size_of::<SwapRepWire>() == 104);
    assert!(size_of::<DeviceReqWire>() == 16);
    assert!(size_of::<DeviceRepWire>() == 20);
    assert!(size_of::<PowerReqWire>() == 8);
    assert!(size_of::<PowerRepWire>() == 8);
    assert!(size_of::<TimerReqWire>() == 32);
    assert!(size_of::<TimerRepWire>() == 48);
    assert!(size_of::<TimerEventWire>() == 32);
    assert!(size_of::<SpawnSegment>() == 24);
    assert!(size_of::<SpawnWire>() == 12);
    assert!(size_of::<SpawnWindow>() == 16);
    assert!(size_of::<SpawnData>() == 8);
    assert!(MAX_SPAWN_DATA <= crate::task::MAX_BOOT_DATA);
    // 48-byte head ‖ 6×24 segments ‖ 36×12 wiring ‖ 9×16 windows
    //     ‖ 4×8 data ‖ 64-byte linked target.  The head is byte-for-byte v2's — the two new
    //     counts came out of its frozen reserved bytes.  Both sides of the
    //     pair read the same `MAX_SPAWN_WIRING`/`MAX_SPAWN_WINDOWS`, so the
    //     record moves in lockstep with those bounded tables.
    assert!(
        size_of::<TaskReqWire>()
            == 48
                + MAX_SPAWN_SEGMENTS * size_of::<SpawnSegment>()
                + MAX_SPAWN_WIRING * size_of::<SpawnWire>()
                + MAX_SPAWN_WINDOWS * size_of::<SpawnWindow>()
                + MAX_SPAWN_DATA * size_of::<SpawnData>()
                + size_of::<ImageName>()
    ); // 864
    assert!(size_of::<TaskReqWire>() == 864);
    // 16-byte head ‖ 256 telemetry ‖ 236 filesystem (with the derived
    // completion-latency moments) ‖ the 76-byte NetSnapshot ‖ the two pages.
    assert!(
        size_of::<TaskRepWire>()
            == 16
                + size_of::<crate::monitor::TelemetrySnapshot>()
                + size_of::<crate::monitor::FilesystemSnapshot>()
                + size_of::<crate::monitor::NetSnapshot>()
                + size_of::<crate::processor_status::ProcessorPage>()
                + size_of::<crate::device::DeviceCensusPage>()
    );
    assert!(size_of::<ImageReqWire>() == 8 + size_of::<ImageName>() + 16 + 64); // 152
    assert!(size_of::<ImageRepWire>() == 16 + size_of::<ModuleDescriptor>()); // 496
    assert!(size_of::<InputReport>() == 16);
    assert!(size_of::<InputReqWire>() == 12);
    assert!(size_of::<InputRepWire>() == 84);
    assert!(size_of::<OutReqWire>() == 32);
    assert!(size_of::<OutRepWire>() == 24);
    assert!(size_of::<OutChunk>() == 32);
    assert!(size_of::<SurReqWire>() == 48);
    assert!(size_of::<SurRepWire>() == 48);
    assert!(size_of::<SurRectWire>() == 8);
    // The commit record: 16-byte head (surface ‖ rect_count ‖ reserve ‖
    // epoch) ‖ 4 rects ‖ 16 reserve = one 64-byte data-plane record
    // (coalescible by construction).
    assert!(size_of::<CommitWire>() == 16 + MAX_COMMIT_RECTS * size_of::<SurRectWire>() + 16);
    assert!(size_of::<CommitWire>() == 64);
    assert!(size_of::<InReqWire>() == 32);
    assert!(size_of::<InRepWire>() == 104);
    assert!(size_of::<InChunk>() == 32);
    assert!(size_of::<HypervisorReqWire>() == 8);
    assert!(size_of::<HypervisorRepWire>() == 8);
    assert!(size_of::<HypervisorStatusWire>() == 72);
    assert!(size_of::<HypervisorConsoleState>() == 16);
    assert!(size_of::<BlockReqWire>() == 16);
    assert!(size_of::<BlockRepWire>() == 32);
    // The filesystem pair: 8-byte head ‖ 4×8 handles/offsets ‖ 4×4 window,
    // length, reserves ‖ 2×32 names.
    assert!(size_of::<FsReqWire>() == 8 + 32 + 16 + 2 * FS_NAME_BYTES); // 120
    assert!(size_of::<FsReqWire>() == 120);
    assert!(size_of::<FsRepWire>() == 48);
    // 16-byte head (len ‖ kind ‖ 2 reserved ‖ 4 reserved ‖ 8 object) ‖ name.
    assert!(size_of::<FsDirEntryWire>() == 16 + FS_NAME_BYTES); // 48
    assert!(size_of::<FsDirEntryWire>() == FsDirEntryWire::BYTES);
    assert!(size_of::<FsSearchHitWire>() == FsSearchHitWire::BYTES);
    assert!(size_of::<FsGrantEntryWire>() == 16 + FS_NAME_BYTES); // 48
    assert!(size_of::<FsGrantEntryWire>() == FsGrantEntryWire::BYTES);
    // 12-byte counters ‖ 2×4 tree counts ‖ 4 reserved ‖ the 80-byte physical
    // storage + health aggregate ‖ 27×8 posture/stats ‖ the 66-byte latency
    // histogram.
    assert!(size_of::<FsSystemStatusWire>() == 24 + 80 + 29 * 8 + 66); // 402
    assert!(size_of::<FsSystemStatusWire>() == FsSystemStatusWire::BYTES);
    // Both 8-deep rings must fit ONE founded pair window, and the filesystem
    // now asks the founding for `SvcProtocol::SERVE_PAIRS` of them — so this
    // is a per-pair budget the founder pays four times over, not once.
    // Spelled here rather than left implicit so a record growth is a build
    // error at the point of growth.
    assert!(align64(192 + 8 * size_of::<FsReqWire>()) == 1152);
    assert!(align64(192 + 8 * size_of::<FsRepWire>()) == 576);
    assert!(1152 + 576 <= 0x2000, "fs pair must fit [layout] ring_bytes");
    // 48-byte head ‖ 48 target words ‖ 6×32 bindings.
    assert!(size_of::<AccelBindingWire>() == 32);
    assert!(size_of::<AccelReqWire>() == 48 + 48 + MAX_ACCEL_BINDINGS * 32); // 288
    assert!(size_of::<AccelReqWire>() == 288);
    assert!(size_of::<AccelRepWire>() == 96);
};

/// `kore-ring`'s ring-region rounding, mirrored for the const-assert above.
///
/// Duplicated rather than imported because `kore-abi` sits *below* `kore-ring`
/// in the crate graph — the ABI crate depends on nothing but zerocopy, and
/// inverting that to share one helper would be a worse trade than eight lines.
/// `crates/ring/src/spsc.rs` is the authority; this must track it.
const fn align64(raw: usize) -> usize {
    (raw + 63) & !63
}

/// The output-service request wire form (`OutputRequest`'s crossing
/// shape; `res/USERLAND.md` §5.9).  `op` selects the verb; the other
/// fields are per-verb (unused ones ride zero).  32 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct OutReqWire {
    /// Verb discriminant: [`OP_ATTACH_TX`](Self::OP_ATTACH_TX) /
    /// [`OP_FLUSH`](Self::OP_FLUSH) / [`OP_DETACH_TX`](Self::OP_DETACH_TX)
    /// / [`OP_STATUS`](Self::OP_STATUS); zero (unseeded) is invalid.
    pub op: u8,
    /// Reserved; must be zero.
    pub reserved0: u8,
    /// Reserved; must be zero.
    pub reserved1: U16,
    /// `AttachTx`: the service-arena slot the client delivered its
    /// queue-window view into (the return channel's landing slot —
    /// #148, no capability ever rides this record).
    pub slot: U32,
    /// `Flush`: the producer's byte watermark — the client asks the
    /// backend to confirm consumption of at least this many bytes
    /// (`res/USERLAND.md` §5.9: a sequence number, never authority).
    pub through: U64,
    /// Reserved; must be zero.
    pub reserved2: U64,
    /// Reserved; must be zero.
    pub reserved3: U64,
}

impl OutReqWire {
    /// `OutputRequest::AttachTx` — attach the delivered queue window.
    pub const OP_ATTACH_TX: u8 = 1;
    /// `OutputRequest::Flush` — confirm backend consumption to `through`.
    pub const OP_FLUSH: u8 = 2;
    /// `OutputRequest::DetachTx` — release the attached queue.
    pub const OP_DETACH_TX: u8 = 3;
    /// `OutputRequest::Status` — snapshot the counters.
    pub const OP_STATUS: u8 = 4;
}

/// The output-service reply wire form (`OutputResponse`'s crossing
/// shape).  24 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct OutRepWire {
    /// Outcome discriminant: [`TAG_ATTACHED`](Self::TAG_ATTACHED) /
    /// [`TAG_FLUSHED`](Self::TAG_FLUSHED) / [`TAG_DETACHED`](Self::TAG_DETACHED)
    /// / [`TAG_STATUS`](Self::TAG_STATUS) / [`TAG_REFUSED`](Self::TAG_REFUSED).
    pub tag: u8,
    /// `Refused`: the `OutputRefusal` discriminant
    /// ([`REFUSE_BUSY`](Self::REFUSE_BUSY)…); zero otherwise.
    pub refusal: u8,
    /// Grant-bearing transfers this reply was effected with (§5.5) —
    /// always `0` for output: authority never rides a reply.
    pub grants: u8,
    /// `Status`: whether a queue was attached when answered (`0`/`1`).
    /// Zero on every other tag (not a status answer).
    pub attached: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 4],
    /// `Flushed`/`Status`: bytes the backend has consumed so far
    /// (monotonic).
    pub through: U64,
    /// `Status`: chunks/bytes dropped on the path so far (monotonic).
    pub dropped: U64,
}

impl OutRepWire {
    /// `OutputResponse::Attached`.
    pub const TAG_ATTACHED: u8 = 1;
    /// `OutputResponse::Flushed` — `through` carries the consumed count.
    pub const TAG_FLUSHED: u8 = 2;
    /// `OutputResponse::Detached`.
    pub const TAG_DETACHED: u8 = 3;
    /// `OutputResponse::Status` — `through` carries the consumed count.
    pub const TAG_STATUS: u8 = 4;
    /// `OutputResponse::Refused` — see `refusal`.
    pub const TAG_REFUSED: u8 = 5;

    /// `OutputRefusal::Busy` — a queue is already attached (the v1
    /// single-peer bound, `res/USERLAND.md` §5.9).
    pub const REFUSE_BUSY: u8 = 1;
    /// `OutputRefusal::NotAttached` — flush/detach with no queue attached.
    pub const REFUSE_NOT_ATTACHED: u8 = 2;
    /// `OutputRefusal::BadQueue` — the delivered slot was absent or not a
    /// well-formed queue window.
    pub const REFUSE_BAD_QUEUE: u8 = 3;
}

/// One data-plane chunk — the `ring::spsc<OutChunk>` record the client
/// produces and the backend consumes (`res/USERLAND.md` §5.9: chunk
/// boundaries are transport mechanics, never application records; the
/// client observes exactly the same byte order however writes are
/// split).  32 bytes — the same record granule as every svc ring.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct OutChunk {
    /// Live bytes in [`bytes`](Self::bytes) (`0..=CHUNK_DATA`).
    pub len: u8,
    /// Typed, producer-originated stream metadata. Unknown bits must be
    /// ignored; they never change the payload byte accounting.
    pub flags: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 6],
    /// The chunk payload.
    pub bytes: [u8; 24],
}

impl OutChunk {
    /// Payload capacity of one chunk.
    pub const CHUNK_DATA: usize = 24;
    /// A genuine input press accompanied this echo chunk. The display accepts
    /// this only from its founder-wired input-stream endpoint; an ordinary
    /// output client cannot use it to dismiss a privileged boot surface.
    pub const FLAG_INPUT_PRESS: u8 = 1 << 0;
}

/// The stream-input-service request wire form (`res/USERLAND.md` §5.9 —
/// the generic byte stream **to** one client; `InReqWire` mirrors
/// [`OutReqWire`]'s layout so the pair reads as one contract).  `op`
/// selects the verb; the other fields are per-verb (unused ones ride
/// zero).  32 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct InReqWire {
    /// Verb discriminant: [`OP_ATTACH_RX`](Self::OP_ATTACH_RX) /
    /// [`OP_DETACH_RX`](Self::OP_DETACH_RX) / [`OP_STATUS`](Self::OP_STATUS);
    /// zero (unseeded) is invalid.
    pub op: u8,
    /// Reserved; must be zero.
    pub reserved0: u8,
    /// Reserved; must be zero.
    pub reserved1: U16,
    /// `AttachRx`: the service-arena slot the client delivered its
    /// queue-window view into (the return channel's landing slot —
    /// #148, no capability ever rides this record).
    pub slot: U32,
    /// Reserved; must be zero.
    pub reserved2: U64,
    /// Reserved; must be zero.
    pub reserved3: U64,
    /// Reserved; must be zero.
    pub reserved4: U64,
}

impl InReqWire {
    pub const OP_DIAGNOSTIC: u8 = 4;
    pub const OP_I2C_STATUS: u8 = Self::OP_DIAGNOSTIC;
    /// `InputStreamRequest::AttachRx` — attach the delivered queue window.
    pub const OP_ATTACH_RX: u8 = 1;
    /// `InputStreamRequest::DetachRx` — release the attached queue.
    pub const OP_DETACH_RX: u8 = 2;
    /// `InputStreamRequest::Status` — snapshot the counters.
    pub const OP_STATUS: u8 = 3;
}

/// The stream-input-service reply wire form. The legacy stream counters are
/// followed by the upstream relay and two compact I2C worker snapshots.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct InRepWire {
    /// Outcome discriminant: [`TAG_ATTACHED`](Self::TAG_ATTACHED) /
    /// [`TAG_DETACHED`](Self::TAG_DETACHED) / [`TAG_STATUS`](Self::TAG_STATUS)
    /// / [`TAG_REFUSED`](Self::TAG_REFUSED).
    pub tag: u8,
    /// `Refused`: the `InputStreamRefusal` discriminant
    /// ([`REFUSE_BUSY`](Self::REFUSE_BUSY)…); zero otherwise.
    pub refusal: u8,
    /// Grant-bearing transfers this reply was effected with (§5.5) —
    /// always `0` for stream input: authority never rides a reply.
    pub grants: u8,
    /// `Status`: whether a queue was attached when answered (`0`/`1`).
    /// Zero on every other tag (not a status answer).
    pub attached: u8,
    /// `Status`: distinct report sources the backend has seen — `2` on a
    /// terminal means two keyboards have genuinely both delivered.  A
    /// count, never a label: the terminal is handed a source byte and
    /// cannot know what hardware is behind it.  Zero on every other tag,
    /// and on a byte-stream backend that has no such notion.
    ///
    /// Carved from the four reserved bytes the legacy head already held.
    pub sources: U32,
    /// `Status`: bytes the backend has produced into the queue so far
    /// (monotonic).
    pub produced: U64,
    /// `Status`: bytes lost to a full queue (the explicit input overrun,
    /// `res/USERLAND.md` §5.9 — counted, never silent; monotonic).
    pub overrun: U64,
    /// `Status`: the upstream input service's relay bring-up rung
    /// (`0` = running).  Carried through, not recomputed: it describes
    /// hardware this service never touches.
    pub relay_rung: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 3],
    /// `Status`: relay line interrupts woken on.
    pub relay_wakes: U32,
    /// `Status`: relay parks issued on the line — `relay_wakes`'s
    /// denominator, and the witness that separates a relay serviced by
    /// its backstop from one that has stopped parking at all.
    pub relay_parks: U32,
    /// `Status`: relay reports pushed into its handoff queue.
    pub relay_events: U32,
    /// `Status`: relay reports its handoff queue had no room for.
    pub relay_dropped: U32,
    /// `Status`: relay bytes that decoded to no HID meaning.
    pub relay_unnamed: U32,
    /// `Status`: `1` when the upstream router owns a live xHCI controller.
    pub xhci_bound: U32,
    /// `Status`: packed upstream xHCI intake refusal
    /// (`stage << 56 | diagnostic`). Zero means binding completed.
    pub xhci_bind: U64,
    /// `Status`: HID boot sources armed behind xHCI.
    pub xhci_sources: U32,
    /// `Status`: xHCI discovery/arming failures.
    pub xhci_discovery_errors: U32,
    /// `Status`: xHCI runtime completion/decode/re-arm failures.
    pub xhci_irq_errors: U32,
    /// `Status`: packed state for the two upstream HID-over-I2C workers.
    pub i2c_state: [U64; 2],
    /// `Status`: completed I2C transfers for each worker.
    pub i2c_transfers: [U32; 2],
    /// `Status`: decoded reports for each worker.
    pub i2c_reports: [U32; 2],
}

impl InRepWire {
    /// Diagnostic row: sources=packed identity, relay_rung=state, relay words=counters.
    pub const TAG_DIAGNOSTIC: u8 = 5;
    pub const TAG_I2C_STATUS: u8 = Self::TAG_DIAGNOSTIC;
    /// `InputStreamResponse::Attached`.
    pub const TAG_ATTACHED: u8 = 1;
    /// `InputStreamResponse::Detached`.
    pub const TAG_DETACHED: u8 = 2;
    /// `InputStreamResponse::Status` — `produced`/`overrun` carry the
    /// counters.
    pub const TAG_STATUS: u8 = 3;
    /// `InputStreamResponse::Refused` — see `refusal`.
    pub const TAG_REFUSED: u8 = 4;

    /// `InputStreamRefusal::Busy` — a queue is already attached (the v1
    /// single-peer bound, `res/USERLAND.md` §5.9).
    pub const REFUSE_BUSY: u8 = 1;
    /// `InputStreamRefusal::NotAttached` — detach with no queue attached.
    pub const REFUSE_NOT_ATTACHED: u8 = 2;
    /// `InputStreamRefusal::BadQueue` — the delivered slot was absent or
    /// not a well-formed queue window.
    pub const REFUSE_BAD_QUEUE: u8 = 3;
}

/// One data-plane chunk — the `ring::spsc<InChunk>` record the backend
/// produces and the client consumes (the exact mirror of [`OutChunk`]:
/// chunk boundaries are transport mechanics, never application records).
/// 32 bytes — the same record granule as every svc ring.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct InChunk {
    /// Live bytes in [`bytes`](Self::bytes) (`0..=CHUNK_DATA`).
    pub len: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 7],
    /// The chunk payload.
    pub bytes: [u8; 24],
}

impl InChunk {
    /// Payload capacity of one chunk.
    pub const CHUNK_DATA: usize = 24;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every wire form reinterprets in place from arbitrary bytes and
    /// round-trips through its own bytes — the shared-ring contract
    /// (any bit pattern is *readable*; the codec's job is refusal).
    #[test]
    fn wire_forms_round_trip_as_plain_bytes() {
        let req = MmReqWire {
            op: MmReqWire::OP_ALLOC,
            contiguous: 1,
            numa: U16::new(2),
            slot: U32::new(0),
            size: U64::new(8192),
            align: U64::new(4096),
            attrs: U16::new(0b11000),
            dma: 1,
            reserved: [0; 5],
        };
        let back = MmReqWire::read_from_bytes(req.as_bytes()).unwrap();
        assert_eq!(back, req);

        let rep = SwapRepWire {
            tag: SwapRepWire::TAG_STATUS,
            refusal: 0,
            grants: 0,
            posture: SwapRepWire::STATUS_DEVICE | SwapRepWire::STATUS_ROUTE,
            token: U32::new(7),
            capacity_pages: U32::new(262144),
            stored_pages: U32::new(3),
            staging_pages: U32::new(8),
            reserved2: [0; 4],
            outs: U64::new(11),
            ins: U64::new(9),
            discards: U64::new(2),
            io_submitted: U64::new(25),
            io_completed: U64::new(25),
            io_errors: U64::new(0),
            fires: U64::new(23),
            parks: U64::new(24),
            deadline_waits: U64::new(26),
            spin_waits: U64::new(27),
        };
        assert_eq!(SwapRepWire::read_from_bytes(rep.as_bytes()).unwrap(), rep);

        // The big nested forms read from raw (attacker-shaped) bytes
        // without refusing at the *layout* level — FromBytes end to end.
        let raw = [0xA5u8; core::mem::size_of::<TaskReqWire>()];
        let t = TaskReqWire::read_from_bytes(&raw).unwrap();
        assert_eq!(t.op, 0xA5, "layout admits any bytes; the codec refuses");
        // The v3 tables reinterpret from the same attacker-shaped bytes.
        assert_eq!(t.window_count, 0xA5);
        assert_eq!(t.data_count, 0xA5);
        assert_eq!(t.windows[7].len.get(), 0xA5A5_A5A5);
        assert_eq!(t.data[3].value.get(), 0xA5A5_A5A5);
        let raw = [0x5Au8; core::mem::size_of::<ImageRepWire>()];
        let i = ImageRepWire::read_from_bytes(&raw).unwrap();
        assert_eq!(i.tag, 0x5A);

        // The input data-plane record reinterprets from any 16 bytes (the
        // shared report window's peer is untrusted) and round-trips.
        let kbd = InputReport::keyboard(3, 0x02, [0x04, 0x05, 0, 0, 0, 0]);
        assert_eq!(InputReport::read_from_bytes(kbd.as_bytes()).unwrap(), kbd);
        let mouse = InputReport::mouse(4, 0x01, 5, -3, 0);
        assert_eq!(mouse.data, [5, (-3i8) as u8, 0, 0, 0, 0]);
        let raw = [0xC3u8; 16];
        assert_eq!(InputReport::read_from_bytes(&raw).unwrap().kind, 0xC3);

        // A digitizer contact packs two little-endian `u16`s into the
        // frozen `data` array; the accessors are the only reader, so a
        // byte-order slip here is caught rather than reaching a cursor.
        let c = InputReport::contact(
            0x41,
            InputReport::CONTACT_TIP | InputReport::CONTACT_CONFIDENCE,
            0x1234,
            0x0ABC,
            1,
            2,
        );
        assert_eq!(c.kind, InputReport::KIND_CONTACT);
        assert_eq!(c.data, [0x34, 0x12, 0xBC, 0x0A, 1, 2]);
        assert_eq!(c.contact_x(), 0x1234);
        assert_eq!(c.contact_y(), 0x0ABC);
        assert_eq!(c.contact_id(), 1);
        assert_eq!(c.contact_count(), 2);
        assert_eq!(InputReport::read_from_bytes(c.as_bytes()).unwrap(), c);
        assert_eq!(c.contact_axes(), None, "legacy contacts are unclassified");
        let axes = c.with_contact_axes(InputReport::CONTACT_TOUCHPAD, 0x1234, 0xabcd);
        assert_eq!(axes.reserved, [1, 1, 0x34, 0x12, 0xcd, 0xab, 0]);
        assert_eq!(
            InputReport::read_from_bytes(axes.as_bytes())
                .unwrap()
                .contact_axes(),
            Some((InputReport::CONTACT_TOUCHPAD, 0x1234, 0xabcd))
        );
        for (index, value) in [(0, 2), (1, 0), (1, 4), (6, 1)] {
            let mut invalid = axes;
            invalid.reserved[index] = value;
            assert_eq!(invalid.contact_axes(), None);
        }
        assert_eq!(c.with_contact_axes(1, 0, 100).contact_axes(), None);
        assert_eq!(c.with_contact_axes(1, 100, 0).contact_axes(), None);
        assert_eq!(mouse.with_contact_axes(1, 100, 100).contact_axes(), None);
        // Extremes: a full-scale coordinate must not wrap or sign-extend.
        let hi = InputReport::contact(0, InputReport::CONTACT_TIP, u16::MAX, u16::MAX, 0, 1);
        assert_eq!((hi.contact_x(), hi.contact_y()), (u16::MAX, u16::MAX));
        // The three kinds stay distinct discriminants (the reader's gate).
        assert_eq!(
            [
                InputReport::KIND_KEYBOARD,
                InputReport::KIND_MOUSE,
                InputReport::KIND_CONTACT
            ],
            [1, 2, 3]
        );

        let req = InputReqWire {
            op: InputReqWire::OP_SET_FOCUS,
            focus: InputReqWire::FOCUS_ONE,
            reserved: [0; 2],
            slot: U32::new(0),
            subscription: U32::new(9),
        };
        assert_eq!(InputReqWire::read_from_bytes(req.as_bytes()).unwrap(), req);
        let rep = InputRepWire {
            tag: InputRepWire::TAG_SUBSCRIBED,
            refusal: 0,
            grants: 0,
            rung: 0,
            subscription: U32::new(9),
            wakes: U32::new(0),
            parks: U32::new(0),
            events: U32::new(0),
            dropped: U32::new(0),
            unnamed: U32::new(0),
            xhci_bound: U32::new(0),
            xhci_bind: U64::new(0),
            xhci_sources: U32::new(0),
            xhci_discovery_errors: U32::new(0),
            xhci_irq_errors: U32::new(0),
            i2c_state: [U64::new(0), U64::new(0)],
            i2c_transfers: [U32::new(0), U32::new(0)],
            i2c_reports: [U32::new(0), U32::new(0)],
        };
        assert_eq!(InputRepWire::read_from_bytes(rep.as_bytes()).unwrap(), rep);

        // The stream pair's forms reinterpreting in place (the client
        // side of each ring is untrusted).
        let req = InReqWire {
            op: InReqWire::OP_ATTACH_RX,
            reserved0: 0,
            reserved1: U16::new(0),
            slot: U32::new(11),
            reserved2: U64::new(0),
            reserved3: U64::new(0),
            reserved4: U64::new(0),
        };
        assert_eq!(InReqWire::read_from_bytes(req.as_bytes()).unwrap(), req);
        let rep = InRepWire {
            tag: InRepWire::TAG_STATUS,
            refusal: 0,
            grants: 0,
            attached: 1,
            sources: U32::new(2),
            produced: U64::new(7),
            overrun: U64::new(0),
            relay_rung: 0,
            reserved: [0; 3],
            relay_wakes: U32::new(9),
            relay_parks: U32::new(11),
            relay_events: U32::new(4),
            relay_dropped: U32::new(0),
            relay_unnamed: U32::new(0),
            xhci_bound: U32::new(0),
            xhci_bind: U64::new(0),
            xhci_sources: U32::new(0),
            xhci_discovery_errors: U32::new(0),
            xhci_irq_errors: U32::new(0),
            i2c_state: [U64::new(0), U64::new(0)],
            i2c_transfers: [U32::new(0), U32::new(0)],
            i2c_reports: [U32::new(0), U32::new(0)],
        };
        assert_eq!(InRepWire::read_from_bytes(rep.as_bytes()).unwrap(), rep);
        let raw = [0x3Cu8; 32];
        assert_eq!(InChunk::read_from_bytes(&raw).unwrap().len, 0x3C);
    }

    /// Discriminants are 1-based: an all-zero (unseeded) record carries
    /// no valid verb/outcome — the `seg` precedent.
    #[test]
    fn zero_is_never_a_discriminant() {
        for d in [
            MmReqWire::OP_ALLOC,
            MmRepWire::TAG_GRANTED,
            SwapReqWire::OP_OUT,
            SwapRepWire::TAG_STORED,
            TaskReqWire::OP_SPAWN,
            TaskReqWire::OP_CONVEY,
            TaskRepWire::TAG_SPAWNED,
            TaskRepWire::TAG_CONVEYED,
            ImageReqWire::OP_REGISTER,
            ImageRepWire::TAG_REGISTERED,
            InputReqWire::OP_SUBSCRIBE,
            InputRepWire::TAG_SUBSCRIBED,
            InputReport::KIND_KEYBOARD,
            InReqWire::OP_ATTACH_RX,
            InRepWire::TAG_ATTACHED,
            OutReqWire::OP_ATTACH_TX,
            OutRepWire::TAG_ATTACHED,
            SurReqWire::OP_QUERY,
            SurReqWire::OP_CREATE_SURFACE,
            SurReqWire::OP_SIGNAL,
            SurRepWire::TAG_INFO,
            SurRepWire::TAG_SURFACE_MADE,
            SurRepWire::TAG_REFUSED,
        ] {
            assert_ne!(d, 0);
        }
    }
}

// ─── surface — the virtual-framebuffer class protocol (kore-svc-compositor) ───
//
// The surface program (#578): a **surface** is display-owned memory a
// client renders into — geometry + backing + epoch/commit state, one
// writer, composed onto the screen by the one authority that owns it.
// A NEW class vocabulary beside the output protocol (never an extension
// of it — `res/USERLAND.md` §5.9 stays frozen), carried over the ordinary
// pair rings with the accel wire disciplines verbatim: byte-exact
// fail-closed decode, per-op reply-tag law with subject echo, grant-count
// law per tag.  The compositor is the LESSOR: `CreateSurface` carves from
// the display row's founded pool and the reply's one grant is the
// detached window view the client maps (client-supplied backing was
// rejected — a client-lessor can cross-space-unmap the compositor
// mid-compose).  The commit data plane never rides these records: it is
// the [`CommitWire`] queue appended inside the standard pair window.

/// The surface request wire form (`SurfaceRequest`'s crossing shape).
/// `op` selects the verb; `subject` names the verb's surface id (zero
/// when the verb has none); `a..d` are per-verb words and unused ones
/// ride zero — a set unused field refuses at decode, the accel posture.
/// 48 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct SurReqWire {
    /// Verb discriminant ([`OP_QUERY`](Self::OP_QUERY)…); zero
    /// (unseeded) is invalid.
    pub op: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 3],
    /// Protocol flags; reserved-zero in v1 (frozen additive headroom).
    pub flags: U32,
    /// The verb's surface id (`DestroySurface` / `Status`); zero when
    /// the verb names none (`Query`, `CreateSurface`, `Signal`,
    /// `RequestClose`).
    pub subject: U64,
    /// Per-verb word (see the `OP_*` docs).
    pub a: U64,
    /// Per-verb word.
    pub b: U64,
    /// Per-verb word.
    pub c: U64,
    /// Per-verb word.
    pub d: U64,
}

impl SurReqWire {
    /// `SurfaceRequest::Query` — the compositor's mode, limits, and pool
    /// headroom ([`SurRepWire::TAG_INFO`]).
    pub const OP_QUERY: u8 = 1;
    /// `SurfaceRequest::CreateSurface` — `a` = `width ‖ height << 16 ‖
    /// format << 32` (pixels, [`crate::session::DISPLAY_FORMAT_XRGB8888`]
    /// is the one v1 format), `b` = reach ([`REACH_CPU`](Self::REACH_CPU)
    /// only in v1), `c` = layout modifier
    /// ([`MODIFIER_LINEAR`](Self::MODIFIER_LINEAR) only in v1).  `reach`
    /// and `modifier` are the #579 horizon reservations: they ride the v1
    /// wire so heterogeneous backings never force a protocol change —
    /// v1 REFUSES every other value rather than ignoring it.
    pub const OP_CREATE_SURFACE: u8 = 2;
    /// `SurfaceRequest::DestroySurface` — `subject` = the surface.  The
    /// lessor-side revoke: the client's mapping dies with the reply.
    pub const OP_DESTROY_SURFACE: u8 = 3;
    /// `SurfaceRequest::Status` — `subject` = the surface; the
    /// re-readable truth behind the done-notice
    /// ([`SurRepWire::TAG_STATUS`]).
    pub const OP_STATUS: u8 = 4;
    /// `SurfaceRequest::Signal` — mint this pair's presented-notice
    /// channel and grant its recv end (the reply's one retained grant);
    /// `COOKIE_DISPLAY_DONE` notices ride it after each present.
    pub const OP_SIGNAL: u8 = 5;
    /// `SurfaceRequest::RequestClose` — this window's own close, asked
    /// through the controller's path instead of taken: the compositor
    /// raises the same `close_requested` the chrome's close dot raises
    /// and echoes `CloseRequested` down this pair's event channel, so the
    /// controller observes the close before the app leaves and the app
    /// leaves only on the echo.  No words; `subject` is zero
    /// ([`SurRepWire::TAG_CLOSE_REQUESTED`]).
    pub const OP_REQUEST_CLOSE: u8 = 6;

    /// [`OP_CREATE_SURFACE`](Self::OP_CREATE_SURFACE) `b`: CPU-reachable
    /// backing — the one v1 reach.
    pub const REACH_CPU: u16 = 0;
    /// [`OP_CREATE_SURFACE`](Self::OP_CREATE_SURFACE) `c`: linear
    /// row-major layout — the one v1 modifier (deliberately the DRM
    /// format-modifier vocabulary: `DRM_FORMAT_MOD_LINEAR = 0`).
    pub const MODIFIER_LINEAR: u64 = 0;
}

/// The surface reply wire form (`SurfaceResponse`'s crossing shape).
/// Data only — the authority-bearing replies announce their grant count
/// and the grant rides the channel, never these bytes.  The `id` echo is
/// load-bearing on this uncorrelated pair: a reply naming a different
/// subject than the outstanding request is refused by the client, never
/// trusted.  48 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct SurRepWire {
    /// Outcome discriminant ([`TAG_INFO`](Self::TAG_INFO)…); zero
    /// (unseeded) is invalid.
    pub tag: u8,
    /// Grants riding beside this reply — zero except
    /// [`TAG_SURFACE_MADE`](Self::TAG_SURFACE_MADE)'s window view and
    /// [`TAG_SIGNALED`](Self::TAG_SIGNALED)'s channel recv (one each).
    pub grants: u8,
    /// The refusal code ([`REFUSE_UNSUPPORTED`](Self::REFUSE_UNSUPPORTED)…)
    /// when `tag` is [`TAG_REFUSED`](Self::TAG_REFUSED); zero otherwise.
    pub refusal: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 5],
    /// The verb's surface word: a create's fresh id, a destroy/status
    /// echo, a refusal's echoed subject; zero for `Query`/`Signal`/
    /// `RequestClose`.
    pub id: U64,
    /// Per-tag word (see the `TAG_*` docs).
    pub a: U64,
    /// Per-tag word.
    pub b: U64,
    /// Per-tag word.
    pub c: U64,
    /// Per-tag word.
    pub d: U64,
}

impl SurRepWire {
    /// `SurfaceResponse::Info` — `a` = `width ‖ height << 16 ‖
    /// format << 32 ‖ max_surfaces(u16) << 40` (the compositor's present
    /// mode + surface-table bound), `b` = pool free bytes, `c` = pool
    /// granule bytes (every backing is carved in whole granules).
    pub const TAG_INFO: u8 = 1;
    /// `SurfaceResponse::SurfaceMade` — `id` = the fresh surface,
    /// `a` = the row stride in bytes (64-aligned), `b` = the backing's
    /// byte length; the reply's one grant is the detached window view of
    /// exactly that backing (the client maps it, the client pays the VA).
    pub const TAG_SURFACE_MADE: u8 = 2;
    /// `SurfaceResponse::Destroyed` — `id` echoes the destroyed surface.
    pub const TAG_DESTROYED: u8 = 3;
    /// `SurfaceResponse::Status` — `id` echoes the surface, `a` = the
    /// committed epoch, `b` = the presented epoch, `c` = commit records
    /// refused so far (the books, never silent), `d` = the timestamp of
    /// the last present (the done-notice payload's re-readable twin).
    pub const TAG_STATUS: u8 = 4;
    /// `SurfaceResponse::Signaled` — the presented-notice channel's recv
    /// rides this reply's one retained grant.
    pub const TAG_SIGNALED: u8 = 5;
    /// `SurfaceResponse::Refused` — see `refusal`; `id` echoes the
    /// subject the refusal names (zero when the verb named none).
    pub const TAG_REFUSED: u8 = 6;
    /// `SurfaceResponse::CloseRequested` — `a` = the configure serial the
    /// close was raised under (never zero: an unconfigured window cannot
    /// be closing).  The echo itself rides the event channel.
    pub const TAG_CLOSE_REQUESTED: u8 = 7;

    /// A v1 vocabulary gate: an unknown op, a non-XRGB8888 format, a
    /// non-CPU reach, a non-LINEAR modifier, a non-empty flag set.
    pub const REFUSE_UNSUPPORTED: u8 = 1;
    /// A malformed argument (zero geometry, a stride/backing overflow).
    pub const REFUSE_INVALID_ARGUMENT: u8 = 2;
    /// The surface table is full (the bounded-slot refusal).
    pub const REFUSE_RESOURCE_LIMIT: u8 = 3;
    /// The pool cannot carve the backing (free granules exhausted or
    /// fragmented).
    pub const REFUSE_OUT_OF_MEMORY: u8 = 4;
    /// `subject` names no live surface.
    pub const REFUSE_UNKNOWN: u8 = 5;
    /// A capability transfer or teardown did not settle. A retiring surface
    /// keeps its backing reserved until revocation succeeds; retry is safe.
    pub const REFUSE_BUSY: u8 = 6;
}

/// One damage rectangle of a [`CommitWire`] record — pixel-addressed,
/// surface-relative.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct SurRectWire {
    /// Left edge (pixels from the surface's left).
    pub x: U16,
    /// Top edge (pixels from the surface's top).
    pub y: U16,
    /// Width in pixels.
    pub w: U16,
    /// Height in pixels.
    pub h: U16,
}

/// One **commit record** — the surface data plane.  These never ride the
/// request rings: the founder appends a commit queue of exactly these
/// records inside the standard pair window (`SvcProtocol::init_pair_window`,
/// the time-service plane precedent), so a client publishes a frame with
/// one ring push + doorbell kick and the compositor coalesces at its own
/// cadence.  Coalescible by construction: two records for one surface
/// union their damage and the higher epoch wins.  64 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct CommitWire {
    /// The surface whose frame this publishes.
    pub surface: U32,
    /// Live rectangles in [`rects`](Self::rects) (`0..=4`); **zero means
    /// full-surface damage** (the whole-frame publish).
    pub rect_count: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 3],
    /// The client's frame counter — strictly increasing per surface; a
    /// record at or below the surface's committed epoch is stale and
    /// refused into the books (last writer already won).
    pub epoch: U64,
    /// Damage rectangles `[0..rect_count]` live, the rest all-zero.
    pub rects: [SurRectWire; MAX_COMMIT_RECTS],
    /// Reserved; must be zero (additive headroom to the 64-byte record).
    pub reserved2: [u8; 16],
}

/// Damage rectangles one commit record can carry.  A frame with more
/// damage than this commits full-surface (`rect_count = 0`) — correct by
/// over-approximation, never a record move.
pub const MAX_COMMIT_RECTS: usize = 4;

// ─── The filesystem protocol ────────────────────────────────────────────────
//
// `res/FILESYSTEM_MVP_IMPLEMENTATION_PLAN.md` §5.  Two properties of this
// protocol are structural rather than conventional, and both are visible in
// the records below:
//
// - **No capability and no address ever rides these bytes.**  A handle is an
//   opaque service-local `(slot, generation)` word; file bytes ride the pair's
//   separately-shared data window, named by an offset and a length *into that
//   window*.  There is no pointer field to forge, which is why the fat verbs
//   (`Read`/`WriteAt`/`Append`) are the same width as the thin ones.
// - **The caller's identity is not here.**  The engine keys every view on the
//   session derived from the pair that delivered the request (plan §3.1), so a
//   `session` field would be a lie a client could tell.  The one session
//   number that *does* appear is `Grant`'s destination, which is a routing
//   argument the service validates against its own attached-session table —
//   naming a session you may send to is not claiming to be one.

/// Bytes of one name component on the wire.
///
/// The engine's own `MAX_NAME_LEN` must equal this; `kore-svc-dagfs` asserts the
/// equality at compile time. Frozen here because a name is inline in the
/// record: changing it is a protocol break, not a tuning knob.
pub const FS_NAME_BYTES: usize = 32;

/// The filesystem request wire form. 120 bytes.
///
/// One fixed record for every verb, with per-verb fields riding zero when
/// unused — the shape every other service protocol here uses. The two name
/// slots are what make `Rename` and `Move` expressible without a second
/// round trip.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct FsReqWire {
    /// Verb discriminant (`OP_*`); zero (unseeded) is invalid.
    pub op: u8,
    /// `Create`: the object kind ([`KIND_FILE`](Self::KIND_FILE) /
    /// [`KIND_DIRECTORY`](Self::KIND_DIRECTORY)); `AdmitPeer`: the bounded
    /// recyclable-bundle component (`KIND_PEER_*`). Zero elsewhere.
    pub kind: u8,
    /// Requested/delegated rights mask (`OpenAt`, `OpenRoot`, `OpenGrant`,
    /// `Grant`). An unknown bit is refused, never masked away.
    pub rights: u8,
    /// Live bytes of [`name`](Self::name) (`0..=FS_NAME_BYTES`).
    pub name_len: u8,
    /// Live bytes of [`name2`](Self::name2) (`0..=FS_NAME_BYTES`).
    pub name2_len: u8,
    /// `Create`: the object subtype to stamp (data, never authority; the
    /// values are `kore-dagfs`'s `ObjectSubtype`). `Search`: the freshness
    /// policy. Zero elsewhere.
    pub reserved0: u8,
    /// `Grant`: the destination session. Zero elsewhere.
    pub session: U16,
    /// Primary handle — the parent directory, or the object itself.
    /// `OpenRoot` uses none and rides zero.
    pub handle: U64,
    /// Secondary handle — `Move`'s destination parent, `Link`'s file,
    /// `Revoke`/`OpenGrant`'s grant id. Zero elsewhere.
    pub handle2: U64,
    /// File byte offset (`Read`, `WriteAt`) or new length (`Resize`).
    pub offset: U64,
    /// Expected revision, or
    /// [`FS_ANY_REVISION`](Self::FS_ANY_REVISION) for an unconditional
    /// mutation.
    pub expected_revision: U64,
    /// Byte offset **into the pair's data window** for a bulk transfer.
    /// Never an address.
    pub window_offset: U32,
    /// Transfer length, or a `List` cursor.
    pub length: U32,
    /// Reserved; must be zero.
    pub reserved1: U32,
    /// Reserved; must be zero.
    pub reserved2: U32,
    /// Primary name component, `name_len` bytes live, remainder zero.
    pub name: [u8; FS_NAME_BYTES],
    /// Secondary name component (`Rename`/`Move` destination).
    pub name2: [u8; FS_NAME_BYTES],
}

impl FsReqWire {
    /// Open the calling session's own private root.
    pub const OP_OPEN_ROOT: u8 = 1;
    /// Resolve one name component beneath an open directory.
    pub const OP_OPEN_AT: u8 = 2;
    /// Open a granted file or subtree root by grant id.
    pub const OP_OPEN_GRANT: u8 = 3;
    pub const OP_CLOSE: u8 = 4;
    pub const OP_STAT: u8 = 5;
    pub const OP_LIST: u8 = 6;
    pub const OP_CREATE: u8 = 7;
    pub const OP_READ: u8 = 8;
    pub const OP_WRITE_AT: u8 = 9;
    pub const OP_APPEND: u8 = 10;
    pub const OP_RESIZE: u8 = 11;
    pub const OP_RENAME: u8 = 12;
    pub const OP_MOVE: u8 = 13;
    pub const OP_LINK: u8 = 14;
    pub const OP_UNLINK: u8 = 15;
    pub const OP_GRANT: u8 = 16;
    pub const OP_REVOKE: u8 = 17;
    pub const OP_SYNC: u8 = 18;
    /// Attach the caller's bulk **data window**.
    ///
    /// Follows [`OutReqWire::OP_ATTACH_TX`]'s precedent exactly: the client
    /// carves a view of its own memory, delivers it into this service's arena
    /// through the return channel, and this record names only the **landing
    /// slot** it went to.  No capability and no address rides the record, and
    /// the service maps what it was given rather than what it was told.
    ///
    /// Until a window is attached, every bulk verb refuses — a `Read` with
    /// nowhere to land is not a read.
    pub const OP_ATTACH_WINDOW: u8 = 19;
    /// List the authority edges this session holds, into the data window.
    ///
    /// Without this a recipient could never learn the [`GrantId`] of a grant
    /// made to it, and so could never open one — the sharing half of the MVP
    /// would be unreachable from a real client. A grant id is not bearer
    /// authority (opening one checks the holder), so listing them to their own
    /// holder gives nothing away.
    ///
    /// [`GrantId`]: https://docs.rs/kore-svc-dagfs
    pub const OP_LIST_GRANTS: u8 = 20;
    /// Freeze one authorized file revision and write its immutable identity
    /// into the caller's data window.
    pub const OP_LEND_VERSION: u8 = 21;
    /// Read bytes from a previously lent immutable version.
    pub const OP_READ_VERSION: u8 = 22;
    /// Validate and commit one normalized result from the data window.
    pub const OP_COMMIT_DERIVED: u8 = 23;
    /// Release a previously lent immutable version.
    pub const OP_RELEASE_VERSION: u8 = 24;
    /// Query the selected derived lineage for an authorized live handle.
    pub const OP_DERIVED_INFO: u8 = 25;
    /// Founder/admin pair: convey one authenticated component of a new
    /// recyclable filesystem client face.  The component capability rides
    /// the retained return leg; `kind` selects its bundle position and
    /// `window_offset` carries only the advertised landing correlation.
    pub const OP_ADMIT_PEER: u8 = 26;
    /// A recyclable client asks the host to retire the exact pair on which
    /// this record arrived.  Pair zero is the permanent administration face
    /// and cannot close itself.
    pub const OP_CLOSE_PEER: u8 = 27;
    /// Founder/admin pair: consume the completion of the most recent peer
    /// admission. A successful reply carries the host-authored generational
    /// arrival ticket in `handle`.
    pub const OP_TAKE_ADMISSION: u8 = 28;
    /// Founder/admin pair: retire the exact recyclable generation named in
    /// `handle`. The ticket routes lifecycle work but conveys no authority;
    /// authority is the fixed administration pair on which the request lands.
    pub const OP_RETIRE_PEER: u8 = 29;
    /// Founder/admin pair: prove that the exact recyclable generation named
    /// in `handle` is no longer resident.  This is deliberately separate
    /// from [`OP_RETIRE_PEER`](Self::OP_RETIRE_PEER): a service reply is
    /// published before its host action runs, so only a later request can
    /// witness that the host has closed the pair and released its mapping.
    pub const OP_CONFIRM_RETIRED: u8 = 30;
    /// Search the live subtree rooted at `handle`. The query and continuation
    /// are bound to the caller's pair-derived filesystem authority; matches
    /// are written into the authenticated data window as
    /// [`FsSearchHitWire`] records.
    pub const OP_SEARCH: u8 = 31;
    /// Admin-only: bind an authenticated returned ChanSend lineage to the
    /// persisted system root. `window_offset` is return-post correlation.
    pub const OP_REGISTER_SYSTEM_VIEW: u8 = 32;
    /// Admin-only: apply an authenticated returned filesystem-view endpoint
    /// to the recyclable peer in `handle`; `rights` is the profile ceiling.
    pub const OP_APPLY_VIEW: u8 = 33;
    /// Bind one caller-minted endpoint lineage to a handle resolved through
    /// this exact filesystem session.
    /// `kind`: 0 retains the issuer's named grants; 1 binds an isolated root.
    /// Other mode bytes are reserved and refused.
    pub const OP_BIND_VIEW: u8 = 34;
    /// Remove one exact native containment edge without requiring the child
    /// subgraph to become unreachable.
    pub const OP_DETACH_EDGE: u8 = 35;
    /// Grant an explicitly attenuated graph/byte/handle quota.
    pub const OP_GRANT_SCOPED: u8 = 36;
    /// Freeze an executable source revision; requires the dedicated
    /// filesystem binding right in addition to read authority.
    pub const OP_LEND_EXECUTABLE: u8 = 37;
    /// Admin-only: bind an authenticated endpoint lineage to a fresh stable
    /// task-private filesystem root. Unlike `OP_REGISTER_SYSTEM_VIEW`, no
    /// persisted/system object is selected.
    pub const OP_REGISTER_PRIVATE_VIEW: u8 = 38;
    /// Grant a route to the stable task view selected by an authenticated
    /// returned target endpoint. No destination session id rides the wire.
    pub const OP_GRANT_VIEW: u8 = 39;
    /// Land one [`FsSystemStatusWire`] diagnostic snapshot in the caller's
    /// data window. Aggregate load and posture only — no name, scope, or
    /// per-session detail crosses this record.
    pub const OP_SYSTEM_STATUS: u8 = 40;
    /// Hold this session's whole-tree commits until the window closes
    /// (`kind` is 1 to open, 0 to close). A bulk write pays one commit
    /// instead of one per chunk; the deferral is per session and cannot
    /// reach another client's durability.
    pub const OP_DEFER_COMMITS: u8 = 41;
    /// Admin-only lifecycle testimony for the measured system catalogue;
    /// `kind` is 1 while verification/copy is active, 0 once complete.
    pub const OP_SET_SYSTEM_SEED: u8 = 42;
    /// The highest defined verb — the codec's range check.
    /// Session-owned atomic file replacement (no wire layout change).
    pub const OP_BEGIN_REPLACE: u8 = 43;
    pub const OP_WRITE_REPLACEMENT: u8 = 44;
    pub const OP_COMMIT_REPLACE: u8 = 45;
    pub const OP_ABORT_REPLACE: u8 = 46;
    pub const OP_MAX: u8 = Self::OP_ABORT_REPLACE;

    /// [`kind`](Self::kind): a file.
    pub const KIND_FILE: u8 = 1;
    /// [`kind`](Self::kind): a directory.
    pub const KIND_DIRECTORY: u8 = 2;
    /// [`OP_ADMIT_PEER`](Self::OP_ADMIT_PEER): the shared ring lease.
    pub const KIND_PEER_RING: u8 = 0x10;
    /// [`OP_ADMIT_PEER`](Self::OP_ADMIT_PEER): the request kick endpoint.
    pub const KIND_PEER_KICK: u8 = 0x11;
    /// [`OP_ADMIT_PEER`](Self::OP_ADMIT_PEER): the retained-return receiver.
    pub const KIND_PEER_RETURN_RECV: u8 = 0x12;
    /// [`OP_ADMIT_PEER`](Self::OP_ADMIT_PEER): base tag for reply-grant
    /// endpoints; the low seven bits carry the bounded grant index.
    pub const KIND_PEER_GRANT_BASE: u8 = 0x80;

    /// [`expected_revision`](Self::expected_revision): do not check.
    ///
    /// Zero, matching the engine's own sentinel — real revisions start at 1,
    /// so the same field carries a conditional and an unconditional request.
    pub const FS_ANY_REVISION: u64 = 0;
}

/// The filesystem reply wire form. 48 bytes.
///
/// [`status`](Self::status) is the whole outcome: `STATUS_OK` or one of the
/// refusal classes, which are the engine's refusal vocabulary one-for-one so
/// no fidelity is lost crossing the boundary.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct FsRepWire {
    /// `STATUS_OK` or a refusal class; zero (unseeded) is invalid.
    pub status: u8,
    /// Object kind on a successful open/create/stat, else zero.
    pub kind: u8,
    /// Reply flags — [`FLAG_MORE`](Self::FLAG_MORE) for a paged listing.
    pub flags: u8,
    /// Grants riding this reply. Pinned zero: the filesystem mints no
    /// kernel authority, so a non-zero value is a protocol break.
    pub grants: u8,
    /// Bytes transferred, or entries in this listing page.
    pub count: U32,
    /// A minted handle, or a minted grant id.
    pub handle: U64,
    /// The object's length.
    pub len: U64,
    /// The object's revision after the operation.
    pub revision: U64,
    /// Cursor to resume a listing at — meaningful only with
    /// [`FLAG_MORE`](Self::FLAG_MORE).
    pub cursor: U32,
    /// Low byte: the object subtype on a successful open-shaped or Stat
    /// reply (data, never authority). Zero elsewhere, and the upper three
    /// bytes are always zero.
    pub reserved0: U32,
    /// Stable object ID for a successful filesystem open or Stat; zero otherwise.
    pub reserved1: U64,
}

impl FsRepWire {
    pub const STATUS_OK: u8 = 1;
    pub const STATUS_NOT_FOUND: u8 = 2;
    pub const STATUS_ALREADY_EXISTS: u8 = 3;
    pub const STATUS_NOT_DIRECTORY: u8 = 4;
    pub const STATUS_IS_DIRECTORY: u8 = 5;
    pub const STATUS_INVALID_NAME: u8 = 6;
    pub const STATUS_INVALID_HANDLE: u8 = 7;
    pub const STATUS_ACCESS_DENIED: u8 = 8;
    /// The route a handle proved through is gone (plan §2.3) — distinct from
    /// `ACCESS_DENIED`, and the distinction is the whole of immediate loss of
    /// access, so it gets its own class rather than a shared "denied".
    pub const STATUS_ACCESS_LOST: u8 = 9;
    pub const STATUS_CONFLICT: u8 = 10;
    pub const STATUS_CYCLE: u8 = 11;
    pub const STATUS_NOT_EMPTY: u8 = 12;
    pub const STATUS_OUT_OF_OBJECTS: u8 = 13;
    pub const STATUS_OUT_OF_EDGES: u8 = 14;
    pub const STATUS_OUT_OF_HANDLES: u8 = 15;
    pub const STATUS_OUT_OF_STORAGE: u8 = 16;
    pub const STATUS_RANGE: u8 = 17;
    pub const STATUS_UNSUPPORTED: u8 = 18;
    pub const STATUS_PROVIDER_FAILURE: u8 = 19;
    /// A bulk verb arrived with no data window attached.
    ///
    /// Its own class rather than folded into `RANGE`: the client's request was
    /// well formed and its authority was sufficient — it simply never gave the
    /// service anywhere to put the bytes, which is a different mistake with a
    /// different fix ([`FsReqWire::OP_ATTACH_WINDOW`]).
    pub const STATUS_NO_WINDOW: u8 = 20;
    /// A concurrent operation owns the exact generational claim required by
    /// this request. No namespace effect was published and retry is lawful.
    pub const STATUS_BUSY: u8 = 21;
    /// A bounded operation exhausted its caller-selected work budget.
    pub const STATUS_BUDGET_EXCEEDED: u8 = 22;
    /// Current-only search encountered a visible, unindexed file revision.
    pub const STATUS_INDEX_INCOMPLETE: u8 = 23;
    /// A bounded placement/extent table is full — the *volume* still has
    /// room.  Deliberately not [`STATUS_OUT_OF_STORAGE`]: a compile-time slot
    /// ceiling and a full medium demand opposite responses, and one code for
    /// both leaves an operator guessing which one they hit.
    pub const STATUS_OUT_OF_PLACEMENTS: u8 = 24;
    /// The bounded pool of immutable version snapshots is full — every lease
    /// is held. Distinct from a full index: the remedy is a lease being
    /// released (or a leak being found), not capacity.
    pub const STATUS_OUT_OF_VERSION_LEASES: u8 = 25;
    /// The bounded derived-result table is full. The volume and the version
    /// pool are both fine; what ran out is room for another batch.
    pub const STATUS_OUT_OF_INDEX_SLOTS: u8 = 26;
    /// The highest defined status — the codec's range check.
    pub const STATUS_MAX: u8 = Self::STATUS_OUT_OF_INDEX_SLOTS;

    /// [`flags`](Self::flags): the listing has more pages; resume at
    /// [`cursor`](Self::cursor).
    pub const FLAG_MORE: u8 = 1 << 0;
}

/// One directory entry as `List` lands it in the data window. 48 bytes.
///
/// Listings ride the window rather than the reply record for the same reason
/// file bytes do: a page of entries would otherwise set the reply's width, and
/// every thin verb would pay for it.  The reply reports how many entries
/// landed; these are what it landed.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct FsDirEntryWire {
    /// Live bytes of [`name`](Self::name) (`1..=FS_NAME_BYTES`).
    pub name_len: u8,
    /// [`FsReqWire::KIND_FILE`] or [`FsReqWire::KIND_DIRECTORY`].
    pub kind: u8,
    /// Low byte: the entry's object subtype (data, never authority); the
    /// high byte is reserved and must be zero.
    pub reserved0: U16,
    /// Reserved; must be zero.
    pub reserved1: U32,
    /// The entry's object id — data, not authority (plan §2.3).
    pub object: U64,
    /// The entry's name, `name_len` bytes live, remainder zero.
    pub name: [u8; FS_NAME_BYTES],
}

impl FsDirEntryWire {
    /// Bytes one entry occupies in the data window.
    pub const BYTES: usize = 48;
}

/// Maximum bytes in one view-relative search path.
///
/// The filesystem admits thirty-two 32-byte components
/// (`kore_dagfs::bounds::MAX_PATH_DEPTH` × `MAX_NAME_LEN`); thirty-one `/`
/// separators make the complete bounded spelling 1,055 bytes. The number is
/// pinned here rather than derived so a service-side bound increase stays an
/// explicit wire decision instead of silently truncating hits — and the
/// client crate asserts at compile time that this spelling covers the
/// deepest legal path, so the two can only move together. Raised from 263
/// (eight components) when the depth bound went 8 → 32: at 263 a hit nine
/// components deep had nowhere to put its ninth separator. Every client and
/// the service are sealed from one tree, so the record's width is not a
/// compatibility surface between builds; it is a page-capacity one, and eight
/// hits (the largest page) take 9,280 of a process's 65,536-byte data window.
pub const FS_SEARCH_PATH_BYTES: usize = 1055;

/// One authority-filtered structural search hit in the pair's data window.
///
/// Every field is inert observation data. In particular, `object` and
/// `stable_object` are not handles and cannot be used to access the object;
/// a caller that wants access must traverse the returned relative path through
/// its already-live authority anchor.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct FsSearchHitWire {
    /// Live bytes of [`path`](Self::path).
    pub path_len: U16,
    /// [`FsReqWire::KIND_FILE`] or [`FsReqWire::KIND_DIRECTORY`].
    pub kind: u8,
    /// Reserved; must be zero.
    pub reserved0: u8,
    /// Search ranking metadata: low 16 bits = score in ten-thousandths, high
    /// 16 bits = live bytes in `snippet`. Zero for structural hits.
    pub ranking: U32,
    /// Boot-local object identity; data, never authority.
    pub object: U64,
    /// Durable object identity; data, never authority.
    pub stable_object: U64,
    /// Exact object revision observed by the query.
    pub revision: U64,
    /// Exact byte length observed by the query.
    pub len: U64,
    /// UTF-8 view-relative path, without a leading slash.
    pub path: [u8; FS_SEARCH_PATH_BYTES],
    /// Authority-filtered text/reference snippet. It is computed only from the
    /// caller's visible candidate set.
    pub snippet: [u8; 64],
    /// Tail padding reserved for an additive record revision.
    pub reserved2: [u8; 1],
}

impl FsSearchHitWire {
    /// Bytes one hit occupies in the data window: the 40-byte fixed prefix,
    /// the bounded path, the 64-byte snippet and one reserved tail byte.
    pub const BYTES: usize = 40 + FS_SEARCH_PATH_BYTES + 64 + 1;
}

/// One authority edge as `ListGrants` lands it in the data window. 48 bytes —
/// the same granule as [`FsDirEntryWire`], so a client's window arithmetic is
/// the same for both listings.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct FsGrantEntryWire {
    /// Live bytes of [`name`](Self::name); zero means the holder chose no
    /// displayed attachment name.
    pub name_len: u8,
    /// The rights this edge conveys.
    pub rights: u8,
    /// [`KIND_PRIVATE_ROOT`](Self::KIND_PRIVATE_ROOT) or
    /// [`KIND_GRANT`](Self::KIND_GRANT).
    pub kind: u8,
    /// Reserved; must be zero.
    pub reserved0: u8,
    /// Reserved; must be zero.
    pub reserved1: U32,
    /// The opaque grant id — what [`FsReqWire::OP_OPEN_GRANT`] takes.
    pub grant: U64,
    /// The displayed attachment name, `name_len` bytes live.
    pub name: [u8; FS_NAME_BYTES],
}

impl FsGrantEntryWire {
    /// Bytes one entry occupies in the data window.
    pub const BYTES: usize = 48;
    /// The session's own starting view.
    pub const KIND_PRIVATE_ROOT: u8 = 1;
    /// A direct grant from another session.
    pub const KIND_GRANT: u8 = 2;
}

/// The filesystem's diagnostic snapshot as `SystemStatus` lands it in the
/// data window: the serving frontend's aggregate load beside the storage
/// backend's posture and counters, one frozen record.
///
/// **Aggregates only.** No name, path, scope, or per-session figure rides
/// here — the record answers "how is the filesystem doing", never "what is
/// anyone doing with it", so any attached session may ask.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct FsSystemStatusWire {
    /// Live engine sessions (fixed and dynamic, session zero included).
    pub sessions: U16,
    /// Live recyclable client pairs.
    pub peers: U16,
    /// Live private invocation views (stable task roots).
    pub views_private: U16,
    /// Live pinned invocation views (cwd/system bindings).
    pub views_pinned: U16,
    /// Attached client data windows.
    pub windows: U16,
    /// `FLAG_*` posture bits.
    pub flags: U16,
    /// Objects reachable from the persisted root.
    pub tree_objects: U32,
    /// Containment edges reachable from the persisted root.
    pub tree_edges: U32,
    /// Conservative user-region lifecycle (`LIFECYCLE_*`).
    pub user_lifecycle: U16,
    /// Rebuildable system-region lifecycle (`LIFECYCLE_*`).
    pub system_lifecycle: U16,
    /// File bytes accounted to the persisted tree.
    pub tree_bytes: U64,
    /// Physical namespace, mounted-region, and allocator-pressure facts.
    /// Empty in the volatile RAM posture.
    pub storage: crate::monitor::FilesystemStorageSnapshot,
    /// The mount's readiness note (`0` is the fully healthy durable
    /// posture; the vocabulary is the fs service's `note` module).
    pub note: U64,
    /// Whole-tree serializations completed since bring-up.
    pub commits: U64,
    /// Provider: block reads issued.
    pub reads: U64,
    /// Provider: block writes issued.
    pub writes: U64,
    /// Provider: flush barriers issued.
    pub flushes: U64,
    /// Provider: discards issued.
    pub discards: U64,
    /// Provider: bytes moved by copy-on-write relocation.
    pub copied_bytes: U64,
    /// Provider: transactions rolled back whole.
    pub rollbacks: U64,
    /// Provider: completions that arrived after their wait expired.
    pub late_completions: U64,
    /// Provider: orphaned extents reclaimed at recovery.
    pub orphan_reclaims: U64,
    /// Provider: reclaims that failed and were left for the next pass.
    pub reclaim_failures: U64,
    /// Channel: commands submitted to the device queue.
    pub submissions: U64,
    /// Channel: completions consumed from the device queue.
    pub completions: U64,
    /// Channel: submissions deferred by a full queue.
    pub backpressure: U64,
    /// Channel: explicit poll passes taken.
    pub poll_calls: U64,
    /// Channel: waits that reached the backend's pacing seam after an empty
    /// fast-path poll.
    pub device_waits: U64,
    /// Backend: waits that genuinely parked on the realized interrupt route.
    pub route_parks: U64,
    /// Backend: parks the interrupt actually woke. A gap against
    /// `route_parks` is delivery the route did not make — the wait ran its
    /// budget out and fell back to polling.
    pub route_fires: U64,
    /// Backend: waits bounded by a clock deadline with no route to park on.
    /// The route-less posture on a machine whose clock IS readable — bounded
    /// by elapsed time, so an expiry here is real evidence about the device.
    pub deadline_waits: U64,
    /// Backend: waits bounded only by a count of spin hints, because the
    /// clock could supply no deadline at all.
    ///
    /// Reported apart from `deadline_waits` because a spin count bounds
    /// retired instructions rather than time, so an expiry here says the
    /// device was slower than N hints on this core under this contention —
    /// which is not a statement about the device. Non-zero means disk waits
    /// are being judged by something that is not a duration.
    pub spin_waits: U64,
    /// Backend: route-less turns that slept this core on the completion
    /// queue's own cache line, woken by the controller's DMA rather than by
    /// any interrupt — the monitored fast tier's numerator.
    ///
    /// **Reported because it is the one thing a green run cannot tell you.**
    /// No emulator implements a user-mode address monitor (QEMU's TCG decodes
    /// neither `UMWAIT` nor `MWAITX`), so under the boot gates this is always
    /// zero while `monitor_turns` climbs — indistinguishable, without this
    /// field, from real silicon on which the tier silently degraded to a spin.
    /// A machine that took the fast path and one that only thinks it did are
    /// otherwise the same observation.
    pub monitor_sleeps: U64,
    /// Backend: route-less turns that reached the monitored tier at all,
    /// counted whether or not this machine could sleep — the denominator.
    ///
    /// `monitor_sleeps / monitor_turns` is the ratio that says whether the
    /// fast tier is live here. `monitor_turns` alone is how many times a
    /// route-less wait went round, which is the completion-latency signal the
    /// spin loop it replaced never exposed.
    pub monitor_turns: U64,
    /// Provider: completion waits that ran their whole budget out without the
    /// awaited completion arriving.
    ///
    /// The numerator of "is this disk dying". Reported because the alternative
    /// was a counter the provider maintained and nothing read: an expiring
    /// wait is the only early evidence a controller is failing, and without it
    /// on the wire the first visible symptom is a retired volume.
    pub expired_waits: U64,
    /// Provider: of those, the ones where the backend reported itself HEALTHY
    /// out of band — so the device was slow, not dead, and the channel was
    /// kept.
    ///
    /// `expired_waits - slow_waits` is the count that actually condemned a
    /// channel. A rising `slow_waits` is a device that needs a longer budget;
    /// a rising difference is a device on its way out.
    pub slow_waits: U64,
    /// Namespace: placements the provider refused to release, after the
    /// namespace effect that disowned them had already committed. Bytes that
    /// are occupied and unreachable.
    pub leaked_placements: U64,
    /// Namespace: reclamation walks that stopped before reaching the whole
    /// unreachable subtree — a structural refusal, or contention that
    /// outlasted `RECLAIM_CONTENTION_TURNS`.
    ///
    /// Non-zero means objects are resident that nothing can reach. It is the
    /// visible half of a trade: the walk abandons rather than retrying
    /// forever, because an unbounded retry inside the founded filesystem is a
    /// wedge no one can observe, and a bounded leak is at least this number.
    pub abandoned_reclaims: U64,
    /// Provider: payload reads answered from the read-ahead window the
    /// provider already held — no device command at all.
    pub readahead_hits: U64,
    /// Provider: device reads issued to fill that window.  Hits over hits
    /// plus reads is the share of payload reads that cost no command.
    pub readahead_reads: U64,
    /// Channel: completion latency and queue depth as the block channel
    /// measured them — the `res/RESIDENCY.md` §8 "NVMe latency at queue
    /// depth" figure, read off the box instead of a datasheet.  Empty when
    /// the backend publishes no clock.
    pub latency: crate::monitor::FilesystemLatencySnapshot,
}

impl FsSystemStatusWire {
    /// Bytes the record occupies in the data window.
    pub const BYTES: usize = 402;
    /// The backend mounted a durable volume (unset: the RAM posture).
    pub const FLAG_DURABLE: u16 = 1 << 0;
    /// The durable mount recovered degraded and is read-only.
    pub const FLAG_DEGRADED_READ_ONLY: u16 = 1 << 1;
    /// The backend holds its realized interrupt route (unset: polled).
    pub const FLAG_ROUTE_HELD: u16 = 1 << 2;
    /// The administration workspace (`work`) anchor stands.
    pub const FLAG_WORKSPACE: u16 = 1 << 3;
    /// The pinned system invocation view stands.
    pub const FLAG_SYSTEM_VIEW: u16 = 1 << 4;
    /// The mounted block channel crossed its one-way lost boundary.
    ///
    /// `FLAG_DURABLE` remains set because the service still owns the mounted
    /// durable namespace; this bit says its backing channel can no longer
    /// perform I/O and must not be mistaken for the healthy mount posture.
    pub const FLAG_CHANNEL_LOST: u16 = 1 << 5;
    /// The controller holds a host memory buffer for its mapping tables
    /// (unset: it was offered none, or refused it — a DRAM-less part then
    /// pages its tables from NAND on every access).
    pub const FLAG_HOST_MEMORY_BUFFER: u16 = 1 << 6;

    /// No lifecycle testimony was supplied (the bare volatile harness).
    pub const LIFECYCLE_UNKNOWN: u16 = 0;
    /// No GPT extent designates this logical region.
    pub const LIFECYCLE_UNDESIGNATED: u16 = 1;
    /// A designation existed but its extent or contents were refused.
    pub const LIFECYCLE_REFUSED: u16 = 2;
    /// The clamped system region is inside its bounded rebuild transaction.
    pub const LIFECYCLE_REFORMATTING: u16 = 3;
    /// Exec is verifying/copying the measured catalogue into the system tree.
    pub const LIFECYCLE_RESEEDING: u16 = 4;
    /// A designated system region could not be made current this boot.
    pub const LIFECYCLE_UNAVAILABLE: u16 = 5;
    /// The system region was rebuilt and its measured catalogue is current.
    pub const LIFECYCLE_REBUILT: u16 = 6;
    /// The mounted region was recovered without a rebuild and is current.
    pub const LIFECYCLE_CURRENT: u16 = 7;
}

/// Startup-only display owner evidence. Data only; reporting never admits sleep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum DisplayProbe {
    NoReport = 0,
    Ready = 1,
    Absent = 2,
    Identity = 3,
    Clock = 4,
    Mmio = 5,
    Psr = 6,
    NoPanel = 7,
    ManyPanels = 8,
    PanelStatus = 9,
    PanelControl = 10,
    PanelTiming = 11,
    AuxBusy = 12,
    WaitTimeout = 13,
    AuxStatus = 14,
    AuxReply = 15,
    SinkCaps = 16,
    ManySinks = 17,
    BacklightOff = 18,
    Unsupported = 19,
    Changed = 20,
    Restore = 21,
    MmioAllOnes = 22,
}
impl DisplayProbe {
    pub fn decode(value: u8) -> Option<Self> {
        Some(match value {
            0 => Self::NoReport,
            1 => Self::Ready,
            2 => Self::Absent,
            3 => Self::Identity,
            4 => Self::Clock,
            5 => Self::Mmio,
            6 => Self::Psr,
            7 => Self::NoPanel,
            8 => Self::ManyPanels,
            9 => Self::PanelStatus,
            10 => Self::PanelControl,
            11 => Self::PanelTiming,
            12 => Self::AuxBusy,
            13 => Self::WaitTimeout,
            14 => Self::AuxStatus,
            15 => Self::AuxReply,
            16 => Self::SinkCaps,
            17 => Self::ManySinks,
            18 => Self::BacklightOff,
            19 => Self::Unsupported,
            20 => Self::Changed,
            21 => Self::Restore,
            22 => Self::MmioAllOnes,
            _ => return None,
        })
    }
}
/// The fixed five-byte startup report fits the existing eight-byte power forms.
/// `detail` is evidence interpreted by the probe, never a pointer or authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DisplayDiagnostic {
    pub probe: DisplayProbe,
    pub detail: u32,
}
impl DisplayDiagnostic {
    pub const NONE: Self = Self::new(DisplayProbe::NoReport, 0);
    pub const fn new(probe: DisplayProbe, detail: u32) -> Self {
        Self { probe, detail }
    }
    pub fn encode(self) -> [u8; 5] {
        let b = self.detail.to_le_bytes();
        [self.probe as u8, b[0], b[1], b[2], b[3]]
    }
    pub fn decode(bytes: [u8; 5]) -> Option<Self> {
        let probe = DisplayProbe::decode(bytes[0])?;
        let detail = u32::from_le_bytes(bytes[1..].try_into().ok()?);
        if probe == DisplayProbe::NoReport && detail != 0 {
            return None;
        }
        Some(Self::new(probe, detail))
    }
}
