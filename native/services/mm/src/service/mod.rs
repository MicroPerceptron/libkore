//! # `service` — the memory manager, the first [`SystemService`]
//!
//! The **lessor of last resort** (the lessor inversion): the service endowed
//! with a memory supply, granting extents from it and taking them back.
//! Its role bundle is one supply lease
//! ([`role::MEM_SUPPLY`](kore_abi::task::role::MEM_SUPPLY)) plus one
//! region grant ([`role::MEM_REGION`](kore_abi::task::role::MEM_REGION),
//! slice D) — each claimed at bind, geometry riding the claimed token,
//! never a config string; its request vocabulary is the allocator
//! lifecycle: [`MmRequest::Alloc`] (the heap-grow verb),
//! [`MmRequest::Release`] (the return leg), and [`MmRequest::Supply`]
//! (the child-memory mint: a fresh **root** lease off the region, quota-
//! metered per serving pair, delivered whole for a launcher to stamp and
//! convey at a spawn — its frames return to the region bitmap when the
//! child's lineage dies. The region bitmap remains the frame ledger; mm keeps
//! a bounded receipt table solely to make its live per-pair quota exact).
//!
//! **DMA is mm's — at the *frames* tier only.**  `AllocProps`' `dma` /
//! `contiguous` are **placement** properties.  This service's current
//! `SupplyToken` endowment carries only `(base, len)`: it does not carry the
//! source NUMA node, cache policy, or a kernel-authenticated DMA class.
//! Consequently the current allocator admits only NUMA-agnostic, CPU-only,
//! write-back R/W requests.  It refuses stronger properties rather than
//! stamping claims the delivered token cannot prove.  DMA *reachability* — an IOVA
//! window in a device domain — is the **device holder's** act with its own
//! bundle authority (`role::DMA_POOL` is an mm grant *delegated into* a
//! driver's bundle at spawn; RUNG-2: one lease → CPU window + IOVA window;
//! a DMA-reachable frame is unswappable — the IOMMU pin rule).  **MMIO is never mm's**: a BAR pre-exists on
//! the device (`dev.bar(n)` — a derive off the DEVICE cap); only the map
//! *verb* is shared.  This mirrors the kernel exactly — one region/extent
//! mechanism, authority partitioned by root (BootWF #4: mm accumulates no
//! device authority).
//!
//! **What mm is *not*: a sharing broker.**  A granted extent is an *owned*
//! token carrying `DERIVE` — sharing is the holder's own act (derive a
//! lease from the unmapped extent, attenuate views, move one over
//! rendezvous; [`kore_rt::cap`]) and never routes back through mm.  mm's
//! lessor position matters for the *swappable* arrangement instead: a
//! pageable heap is one where mm retains the lease and the tenant holds
//! only a view — the pager-verbs tier that extends this vocabulary
//! (`lease`/`share`/`unshare`) later.
//!
//! **Release is wire-faithful** (#148: no capability rides a ring entry):
//! the extent returns by rendezvous **delivery into mm's arena**
//! before the data-plane request names only the **slot**. mm reclaims (bind →
//! retire → books), and returned windows serve later grants first
//! (first-fit reuse; no coalescing yet — the books make every loss
//! visible, never silent).
//!
//! Serving is **allocation-free** in the common case (bump cursor + tables +
//! counters): a client's heap waits inside its grow path while mm serves it,
//! so the receipt table is provisioned at bind for every lane's live supplies
//! ([`PROVISIONED_RESERVATIONS`]). Past that provision it doubles from mm's
//! own heap, which grows through the kernel's carve-and-map and never through
//! an mm client path, so the one allocation a serve step can make cannot
//! re-enter mm, and a growth the heap refuses is a refusal on the wire
//! ([`MmRefusal::Full`]), never a stall.

use alloc::vec::Vec;
use kore_abi::cap::Rights;
use kore_abi::mem::{AllocProps, DmaClass, MemAttrs, NumaNode};

use kore_abi::session::{MemoryClassCensus, SlotRef, SupplyReleaseReport, SystemMetric};
use kore_abi::task::role;

use kore_abi::task::{Bootstrap, Profile, RoleSlot};
use kore_rt::cap::{RegionToken, SupplyToken, verbs};
use kore_rt::service::{
    BindError, Bundle, GrantArg, PeerClose, ReturnMode, Serve, StartFailure, SupplyReleaseGrant,
    SystemService,
};
use kore_rt::svc::{
    PeerAdmissionRefusal, PeerBundleAssembler, PeerBundlePart, PeerId, PeerMapPlan, SvcProtocol,
};

/// Grant granularity — extents are page-multiples, page-aligned.
pub const PAGE_BYTES: u64 = 4096;

/// Exact cache/access contract the current geometry-only supply can prove.
const SUPPLY_ALLOC_ATTRS: MemAttrs = MemAttrs::WRITE_BACK
    .union(MemAttrs::READ)
    .union(MemAttrs::WRITE);

/// Returned windows the lessor can hold for reuse — a fixed, allocation-
/// free ceiling; overflow is *counted* ([`MmStats::leaked_bytes`]), never
/// silent.
pub const FREE_SLOTS: usize = 16;

/// **The part of the 4 KiB class one lane may hold live**: a half.
///
/// The base-class quota ([`MmRequest::Supply`] at 4 KiB alignment) is the
/// most bytes one client face may have in live capability lineages; the
/// kernel reports completion only after the lineage is dead and the region
/// bitmap has accepted the frames. It is accounted per arrival lane
/// ([`Serve::via`] — structural identity, never a wire field), so init's
/// face and the shell's meter independently, and it makes one client's
/// appetite a *policy* refusal ([`MmRefusal::Quota`], loud on the wire)
/// before that client can drain the class.
///
/// **Where a half comes from.** The quota exists so that no one client
/// takes the class alone; a half is the largest share that still leaves
/// everyone else as much as that one client may hold. The whole class
/// would let one face starve every other, and a smaller share would strand
/// memory from the clients that legitimately hold the most: a seat's
/// controller provisions every window it opens (about 20.5 MiB each,
/// `kore_window_session::layout`) and every command it runs from its own
/// lane. Two clients may still fill the class between them, and then the
/// class itself refuses: the kernel's derive fails `Mem(Exhausted)`, and
/// mm, told only that the grant dropped, answers [`MmRefusal::Denied`].
///
/// **What it replaced.** A fixed 128 MiB per lane, whatever the machine:
/// on a 12 GiB guest a seat's sixth window was refused `Quota` with
/// gigabytes of the class free, as it would have been on a 16 GiB laptop.
const BASE_QUOTA_SHARE: u64 = 2;

/// The [`SystemMetric::MemoryClass`] operand of the 4 KiB base class (the
/// census numbers its classes 0 = 4 KiB, 1 = 2 MiB, 2 = 1 GiB).
///
/// [`SystemMetric::MemoryClass`]: kore_abi::session::SystemMetric::MemoryClass
const BASE_CLASS_CENSUS: u16 = 0;

/// The base-class quota of one lane when the 4 KiB class has `carved`
/// bytes: [`BASE_QUOTA_SHARE`]'s part of it, in whole pages. The boot
/// carves about a fifth of usable memory as the 4 KiB class (the 50/30/20
/// ruling, `res/ACCEL_MEMORY_SCALE.md` §1): a 4 GiB QEMU guest's 737 MiB
/// gives each lane 368.5 MiB, a 12 GiB guest's 2,377 MiB gives 1,188.5 MiB,
/// and a 16 GiB laptop's 3.2 GiB about 1.6 GiB.
pub const fn base_quota_for(carved: u64) -> u64 {
    carved / BASE_QUOTA_SHARE / PAGE_BYTES * PAGE_BYTES
}

/// The **huge-class** per-pair quota is not a constant: it is the
/// `role::SUPPLY_QUOTA` boot datum (`services/mm/boot.toml`, two `u32`
/// words), so the deployed working-set ceiling is manifest policy while the
/// base-class quota above stays the launcher's everyday allowance.  A
/// supply asked for at 2 MiB or 1 GiB alignment (`res/ACCEL_MEMORY_SCALE.md`
/// §1.2: the alignment *is* the frame class) is metered against this quota,
/// never against the base class's ([`base_quota_for`]).  This constant is
/// the ceiling a datum is clamped to, and the receipt's unit field holds a
/// supply of it.
pub const HUGE_SUPPLY_QUOTA_MAX: u64 = 64 << 30;

/// Rights a minted supply **delivers**: the full grant — the client is
/// the supply's owner (carves views, shares windows, conveys the root
/// onward at a spawn; `DERIVE` is both the carve edge and the transfer
/// gate's toll, `REVOKE` what lets the holder — or the child row's reap
/// — tear the subtree down).  Σ5 clamps every later child to this set.
pub const SUPPLY_RIGHTS: Rights = Rights::ACCESS
    .union(Rights::MODIFY)
    .union(Rights::DERIVE)
    .union(Rights::REVOKE);

/// The memory manager's role bundle: the supply it grants heap extents
/// from and the region it mints child supplies off — both **possessed**,
/// not described (each token is the claimed record; the supply's
/// [`len`](SupplyToken::len) is its geometry, the region's ledger is the
/// kernel's frame bitmap).
#[derive(Debug)]
pub struct MmEndowment {
    /// The claimed supply-region token.
    pub supply: SupplyToken,
    /// The claimed region grant — the [`MmRequest::Supply`] minting
    /// authority (`role::MEM_REGION`, slice D).
    pub region: RegionToken,
    /// The base-class per-lane quota ([`base_quota_for`]), derived from the
    /// 4 KiB class the boot carved, read once through the row's read-only
    /// observation grant (`role::SYSTEM_OBSERVE`) — the kernel's own census,
    /// so the quota follows the machine instead of a constant.
    pub base_quota: u64,
    /// The huge-class per-pair quota from the `role::SUPPLY_QUOTA` datum
    /// (bytes; low word at index 0, high word at index 1).  Absent datum ⇒
    /// zero ⇒ every huge-class supply refuses `Quota` — fail-closed, and
    /// visible on the wire rather than a silent fall-back to the base class.
    pub huge_quota: u64,
}

// Hand-written rather than `#[derive(Bundle)]` because the bundle carries a
// boot *datum* beside its two capabilities, and the derive speaks handles
// only.  Same profile and claim shape the derive would emit.
impl Bundle for MmEndowment {
    const PROFILE: Profile = Profile::new(
        "mm",
        &[
            RoleSlot::at(role::MEM_SUPPLY),
            RoleSlot::at(role::MEM_REGION),
            RoleSlot::at(role::SYSTEM_OBSERVE),
        ],
    );

    fn resolve(boot: &Bootstrap<'_>) -> Result<Self, BindError> {
        let supply = boot
            .handle(role::MEM_SUPPLY, 0)
            .ok_or(BindError::MissingRole {
                role: role::MEM_SUPPLY,
                index: 0,
            })
            .and_then(|slot| {
                verbs::claim_supply(slot).ok_or(BindError::BadHandle {
                    role: role::MEM_SUPPLY,
                    index: 0,
                })
            })?;
        let region = boot
            .handle(role::MEM_REGION, 0)
            .ok_or(BindError::MissingRole {
                role: role::MEM_REGION,
                index: 0,
            })
            .and_then(|slot| {
                verbs::claim_region(slot).ok_or(BindError::BadHandle {
                    role: role::MEM_REGION,
                    index: 0,
                })
            })?;
        // The census is read once: the classes are carved at boot and never
        // re-carved, so the quota it derives holds for the boot. A grant
        // that will not answer is the row's misconfiguration, refused at
        // bind as a bad handle rather than served under an invented quota.
        let observe_refused = BindError::BadHandle {
            role: role::SYSTEM_OBSERVE,
            index: 0,
        };
        let observe = boot
            .handle(role::SYSTEM_OBSERVE, 0)
            .ok_or(BindError::MissingRole {
                role: role::SYSTEM_OBSERVE,
                index: 0,
            })
            .and_then(|slot| verbs::claim_observe(slot).ok_or(observe_refused))?;
        let census =
            verbs::observe_metric_at(&observe, SystemMetric::MemoryClass, BASE_CLASS_CENSUS)
                .map(|report| MemoryClassCensus::from_observation(report.value, report.aux))
                .ok_or(observe_refused)?;
        let lo = boot.datum(role::SUPPLY_QUOTA, 0).unwrap_or(0);
        let hi = boot.datum(role::SUPPLY_QUOTA, 1).unwrap_or(0);
        let huge_quota = (u64::from(hi) << 32 | u64::from(lo)).min(HUGE_SUPPLY_QUOTA_MAX);
        Ok(Self {
            supply,
            region,
            base_quota: base_quota_for(census.carved),
            huge_quota,
        })
    }
}

/// The lessor's request vocabulary — the allocator lifecycle.  The
/// pager lease tier (`lease`/`share`/`unshare`) and fault
/// notifications extend this enum later — the vocabulary grows, the
/// intake stays one.
#[derive(Clone, Copy, Debug)]
pub enum MmRequest {
    /// Grant a fresh extent satisfying `props` — the [`kore_abi::mem`] wire
    /// artifact the client heap builds
    /// ([`alloc_props`](crate::client::alloc_props)).
    Alloc(AllocProps),
    /// Take back an extent the client **delivered into mm's arena** at
    /// `slot` (#148: the cap moved by rendezvous, the request names the
    /// slot — data only).  The window re-enters the grantable pool.
    Release {
        /// mm-arena slot holding the delivered extent record.
        slot: SlotRef,
    },
    /// Mint a fresh **root supply lease** off the region endowment and
    /// deliver it whole (slice D — the child-memory verb): the launcher
    /// stamps its child's `.kore.ld` through a transient view, conveys
    /// the root at the spawn, and the frames return to the *region
    /// bitmap* when the child's lineage dies (T6) — never to mm's books.
    /// Metered per arrival lane: the 4 KiB class against the derived base
    /// quota ([`base_quota_for`]), the huge classes against the
    /// `role::SUPPLY_QUOTA` datum.
    Supply {
        /// Requested size in bytes (page-rounded by the lessor).
        size: u64,
        /// Requested backing alignment in bytes: `0` for the 4 KiB
        /// default, else exactly 4 KiB / 2 MiB / 1 GiB (the
        /// [`kore_abi::session::LeaseReq`] vocabulary) — anything else
        /// refuses [`MmRefusal::BadProps`].
        align: u64,
    },
    /// Read one bounded snapshot of this provider's allocator and supply
    /// books. The request carries no authority and mints no grant.
    Stats,
    /// Exec's administration pair: one component of a per-launch client
    /// face, delivered on the return leg at `announced`.
    AdmitPeer {
        part: PeerBundlePart,
        announced: SlotRef,
    },
    /// Exec's administration pair: the verdict of the preceding admission.
    TakeAdmission,
    /// Exec's administration pair: retire one exact face generation.
    RetirePeer { peer: PeerId },
    /// Exec's administration pair: fence retirement — accepted once the face
    /// is closed and every supply it minted has returned.
    ConfirmRetired { peer: PeerId },
    /// A per-launch face retires itself.
    ClosePeer,
}

/// The lessor's replies — **data only**.  Granted authority reaches the
/// client as a delivered capability record it possesses through the typed
/// take verbs, never as reply payload; a reply carries the outcome, not a
/// window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmResponse {
    /// Granted — the extent token arrives by delivery.
    Granted,
    /// The returned window is back in the lessor's books.
    Released,
    /// The provider-owned allocator and child-supply snapshot.
    Stats(kore_abi::monitor::MemorySnapshot),
    /// An administration step took effect.
    Accepted,
    /// The preceding admission produced this face.
    PeerAdmitted { peer: PeerId },
    /// Refused — and a refusal emits **no** grant (fail-closed: no
    /// authority leaves on an error path).
    Refused(MmRefusal),
}

/// Why the lessor refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmRefusal {
    /// Props the current supply token cannot honor or prove: malformed
    /// geometry, a named NUMA node, a DMA placement class, or anything other
    /// than write-back read/write CPU access. A contiguous request is admitted
    /// because every window of this contiguous supply is itself contiguous.
    /// A supply too large for a receipt to name (a 4 KiB-class supply past
    /// [`MAX_BASE_SUPPLY_BYTES`]) is malformed geometry too: that much
    /// memory belongs in a huge class.
    BadProps,
    /// The supply cannot fit the request.
    Exhausted,
    /// The returned slot did not bind to an extent of **this** supply —
    /// nothing delivered there, a forged/stale record, or a window that
    /// was never mm's to reclaim.
    BadReturn,
    /// The capability grant session refused. The reserved window was rolled
    /// back into mm's books and no authority left the service.
    Denied,
    /// The arrival lane's outstanding-supply meter cannot fit the request
    /// (its class's quota: [`base_quota_for`], or the huge datum) — a
    /// *policy* refusal, loud on the wire (the `TableFull` discipline),
    /// distinct from [`Exhausted`] (the backing itself cannot fit it).
    ///
    /// [`Exhausted`]: Self::Exhausted
    Quota,
    /// The administration step cannot complete yet (an admission in
    /// flight, or a retired face whose supplies have not all returned).
    Pending,
    /// No per-launch face fits: every serving slot is attached or still
    /// crediting a closed face's returns, or mm's face arena or capability
    /// arena has no room for one more. A capacity refusal — exec reports it
    /// as a full table, and a launch retried after another engine program
    /// exits can succeed — distinct from [`Denied`] (a face that arrived
    /// wrong). [`MmRequest::Supply`] answers it too when mm's own books
    /// cannot hold one more live supply (the reservation table at its
    /// receipt ceiling, or mm's heap unable to grow it): a capacity of mm's,
    /// not the lane's quota, and gone once a supply returns.
    ///
    /// [`Denied`]: Self::Denied
    Full,
}

#[derive(Clone, Copy, Debug)]
enum CarveUndo {
    Fresh {
        cursor: u64,
        len: u64,
    },
    Reused {
        index: usize,
        window: (u64, u64),
        len: u64,
    },
    /// A [`MmRequest::Supply`] reservation: quota + counters to unwind if
    /// the grant session drops (the region claim itself unwound with the
    /// refused session under kernel I3 — mm holds no window to restore).
    Supply {
        pair: usize,
        len: u64,
        receipt: u64,
    },
}

/// **The live supplies mm provisions room for at bind**, every lane
/// together: one reservation per supply minted and not yet completed. Every
/// lane holding as many live supplies as one task can adopt into its heap
/// ([`kore_rt::heap::MAX_ADOPTED`]), the most a program keeps of its own,
/// so the common case never grows the table inside a serve step. A
/// launcher's brokered supplies (a seat's windows, each conveyed to its own
/// child) ride the lanes that hold few. Past the provision the table
/// doubles from mm's heap ([`Reservations`]), up to what a receipt can
/// index ([`RESERVATION_CEILING`]); a table that cannot grow refuses
/// [`MmRefusal::Full`].
///
/// It replaced a table sized for the bootstrap roster filling a fixed
/// 128 MiB quota with single pages: 1.125 MiB allocated at bind, whose
/// size tied the quota to a constant the machine could never move.
const PROVISIONED_RESERVATIONS: usize = LANES * kore_rt::heap::MAX_ADOPTED;

/// What one reservation costs mm's heap: its state word, and its place in
/// the free stack.
const RESERVATION_BYTES: usize = 2 * core::mem::size_of::<u32>();

/// **Every supply lane mm can tell apart**: one per serving slot the host
/// can name — the fixed pairs, then each per-launch face the host grows into
/// [`MemoryManager::peer_arena`]. A [`PeerId`]'s slot is one byte, so this is
/// the host's whole slot space, not the bootstrap table: sizing the lanes by
/// [`MAX_SERVE_PAIRS`](kore_rt::svc::MAX_SERVE_PAIRS) is what once left two
/// faces beside mm's six fixed pairs, and every engine program past the
/// second refused.
const LANES: usize = PeerId::SLOT_CAPACITY;

/// The huge classes' receipt unit: 2 MiB, the finer huge frame class. A
/// 1 GiB-aligned supply is a whole number of these, so one unit names both.
const HUGE_UNIT_BYTES: u64 = 1 << kore_abi::session::LEASE_ALIGN_2M;

/// Completions consumed per pair per serve step. The drain runs inside the
/// client allocator's grow path, so a banked backlog (a launch storm, then
/// idle) must not stall one serve arbitrarily; the tail beyond the cap waits
/// for the next request — at worst one extra [`MmRefusal::Quota`] round.
const SUPPLY_RELEASE_DRAIN_PER_PAIR: usize = 16;

/// The meter row a supply **lane** is booked to. A lane is the arrival a
/// supply was minted on — a fixed pair's index, or a per-launch face's
/// [`PeerId::arrival`] — and it is also the kernel release lane its
/// completion report comes back on (`SupplyReleaseGrant::pair`). A face's
/// row is its slot. The host never seats a face on a fixed pair's slot, and
/// a slot is not reused while its row is non-zero
/// ([`MemoryManager::peer_closed`]), so a report can never credit a later
/// face in the same slot. Total: a fixed pair's index and a face's slot are
/// both below [`LANES`].
fn meter_index(lane: u16) -> usize {
    match PeerId::from_arrival(lane) {
        Some(peer) => peer.slot(),
        None => usize::from(lane),
    }
}

/// Exec's administration pair: `exec_client` takes the final pair of the
/// row's protocol (`services/exec/src/founder/mod.rs`), the last of mm's
/// [`SERVE_PAIRS`](SvcProtocol::SERVE_PAIRS); the pairs of rows declaring mm
/// among their `clients` are numbered after it (`SERVICE_CLIENT_LINKS`). Only
/// it may admit, retire or confirm per-launch faces.
const ADMIN_VIA: u16 = <MemoryManager as SvcProtocol>::SERVE_PAIRS as u16 - 1;

/// Bytes of one per-launch face's ring — the stride of the peer map. mm's
/// wire is two small fixed-size records per slot, so a page holds both rings
/// at the declared capacities (pinned below).
pub const PEER_RING_BYTES: u64 = 0x1000;

const _: () = assert!(
    kore_rt::svc::window_bytes::<MemoryManager>(
        <MemoryManager as SvcProtocol>::REQUEST_CAPACITY,
        <MemoryManager as SvcProtocol>::RESPONSE_CAPACITY,
    ) as u64
        <= PEER_RING_BYTES,
    "a per-launch mm face's rings must fit its page"
);

/// **The face arena** — what the serve host grows mm's per-launch faces
/// into ([`MemoryManager::peer_arena`]). A named heap of its own, over this
/// task's `role::HEAP` supply: [`HeapSupplySource`] carves and maps through
/// the kernel's session engine, the one source that works for mm itself
/// (mm cannot be its own client), and it keeps the face table's growth
/// accounted apart from the global heap's.
///
/// [`HeapSupplySource`]: kore_rt::heap::HeapSupplySource
static FACE_ARENA: kore_rt::Heap<kore_rt::heap::HeapSupplySource> =
    kore_rt::Heap::new_in(kore_rt::heap::HeapSupplySource::new());

/// **What mm's heap must hold** — the floor its manifest row's
/// `heap_supply_bytes` is held to (exec's founder test
/// `mm_heap_supply_holds_every_face`): the reservations provisioned at bind
/// ([`PROVISIONED_RESERVATIONS`]), the service itself, and the face table
/// grown to every lane, twice over — a doubling table maps its next block
/// before it frees the one it outgrew, and every block it ever held sums to
/// under twice its last — plus a pool extent of headroom for each of the two
/// heaps drawing on the one supply (the global allocator's and
/// [`FACE_ARENA`]'s). Derived from the host's own per-slot cost, so a heavier
/// serving pair raises the floor instead of overrunning it. What the row
/// gives past the floor is the reservation table's room to grow.
pub const HEAP_DEMAND_BYTES: u64 = (PROVISIONED_RESERVATIONS * RESERVATION_BYTES
    + core::mem::size_of::<MemoryManager>()
    + 2 * LANES * kore_rt::svc::peer_slot_bytes::<MemoryManager>()
    + 2 * kore_rt::heap::POOL_EXTENT_BYTES) as u64;

/// How a refused admission verdict reads on mm's wire. No room for one more
/// face — no free slot, no arena room to grow one, no capability slot for its
/// scratch — is [`MmRefusal::Full`], a capacity the launcher reports as a
/// full table and a later launch can find freed; anything else is a face
/// that arrived wrong, [`MmRefusal::Denied`]. Exhaustive, so a new refusal
/// cannot fall into either bucket unexamined.
fn admission_refusal(refusal: PeerAdmissionRefusal) -> MmRefusal {
    match refusal {
        PeerAdmissionRefusal::Capacity | PeerAdmissionRefusal::NoScratchSlots => MmRefusal::Full,
        PeerAdmissionRefusal::InvalidMapPlan
        | PeerAdmissionRefusal::WrongRingAuthority
        | PeerAdmissionRefusal::WrongEndpointAuthority
        | PeerAdmissionRefusal::WrongGrantCount
        | PeerAdmissionRefusal::WrongReturnShape
        | PeerAdmissionRefusal::RingGeometry
        | PeerAdmissionRefusal::MapFailed
        | PeerAdmissionRefusal::RingMalformed
        | PeerAdmissionRefusal::PolicyCollision
        | PeerAdmissionRefusal::PolicyUnavailable => MmRefusal::Denied,
    }
}

/// One per-launch face's slot state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PeerLane {
    /// The face's arrival — its quota lane and release lane.
    arrival: u16,
    /// Still attached. A closed face keeps its lane (and its slot, through
    /// [`MemoryManager::peer_closed`]) until its supplies have returned.
    live: bool,
}

// A receipt is mm's own word: the kernel keeps it in the supply's lineage and
// echoes it in the completion report without reading it (`TrackedLeaseReq`),
// so its packing is this file's to choose. Low to high: the reservation's
// index, its lane, its size in its class's unit, the class, and the
// reservation's generation. Each width is chosen, not derived from a quota, so
// the quota can follow the machine while the encoding stays fixed:
//
// - **Index**, 19 bits: half a million live supplies — 4 MiB of table at
//   `RESERVATION_BYTES` a reservation, the scale of mm's whole heap, so the
//   heap refuses a larger table before the field would.
// - **Lane**: every lane ([`LANES`]).
// - **Units**, 20 bits: one supply in its class's unit — pages for the base
//   class, 2 MiB units for the huge ones — so the largest single supply a
//   receipt names is [`MAX_BASE_SUPPLY_BYTES`] (4 GiB less a page) of the
//   4 KiB class and 2 TiB of the huge ones, past the huge quota's ceiling. No
//   page-granular supply a client asks for comes near it: launches, windows
//   and heap leases run to 128 MiB, and anything larger belongs in a huge
//   class.
// - **Class**, one bit: which meter, and which unit, a completion credits.
// - **Generation**, the rest (16 bits): a slot is taken 65,535 times before
//   it is retired for good, and a stale receipt (a duplicate or reordered
//   report) never matches a later reservation in the same slot.
const RECEIPT_INDEX_BITS: u32 = 19;
const RECEIPT_PAIR_BITS: u32 = (LANES - 1).ilog2() + 1;
const RECEIPT_UNITS_BITS: u32 = 20;
const RECEIPT_CLASS_BITS: u32 = 1;
const RECEIPT_PAIR_SHIFT: u32 = RECEIPT_INDEX_BITS;
const RECEIPT_UNITS_SHIFT: u32 = RECEIPT_PAIR_SHIFT + RECEIPT_PAIR_BITS;
const RECEIPT_CLASS_SHIFT: u32 = RECEIPT_UNITS_SHIFT + RECEIPT_UNITS_BITS;
const RECEIPT_GENERATION_SHIFT: u32 = RECEIPT_CLASS_SHIFT + RECEIPT_CLASS_BITS;
const RECEIPT_INDEX_MASK: u64 = (1 << RECEIPT_INDEX_BITS) - 1;
const RECEIPT_PAIR_MASK: u64 = (1 << RECEIPT_PAIR_BITS) - 1;
const RECEIPT_UNITS_MASK: u64 = (1 << RECEIPT_UNITS_BITS) - 1;
const RECEIPT_CLASS_HUGE: u64 = 1 << RECEIPT_CLASS_SHIFT;
const RECEIPT_GENERATION_MAX: u32 = (1 << (64 - RECEIPT_GENERATION_SHIFT)) - 1;
const RESERVATION_LIVE: u32 = 1 << 31;
const RESERVATION_GENERATION_MASK: u32 = RECEIPT_GENERATION_MAX;

/// The most reservations a receipt can index: the table's ceiling.
const RESERVATION_CEILING: usize = 1 << RECEIPT_INDEX_BITS;

/// The largest 4 KiB-class supply a receipt can name: its unit field full of
/// pages. A larger one is refused [`MmRefusal::BadProps`].
pub const MAX_BASE_SUPPLY_BYTES: u64 = RECEIPT_UNITS_MASK * PAGE_BYTES;

/// The unit a receipt counts a supply of its class in.
const fn receipt_unit(huge: bool) -> u64 {
    if huge { HUGE_UNIT_BYTES } else { PAGE_BYTES }
}

/// One receipt's fields ([`pack`](Self::pack) and [`unpack`](Self::unpack)
/// are the one encoding).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Receipt {
    index: usize,
    pair: usize,
    units: u64,
    huge: bool,
    generation: u32,
}

impl Receipt {
    /// The word the kernel carries. Every field is in range by
    /// construction: the index below the table's ceiling, the lane below
    /// [`LANES`], the units refused past the field before a reservation is
    /// made, the generation masked by the slot that minted it.
    const fn pack(self) -> u64 {
        (self.generation as u64) << RECEIPT_GENERATION_SHIFT
            | if self.huge { RECEIPT_CLASS_HUGE } else { 0 }
            | self.units << RECEIPT_UNITS_SHIFT
            | (self.pair as u64) << RECEIPT_PAIR_SHIFT
            | self.index as u64
    }

    const fn unpack(receipt: u64) -> Self {
        Self {
            index: (receipt & RECEIPT_INDEX_MASK) as usize,
            pair: ((receipt >> RECEIPT_PAIR_SHIFT) & RECEIPT_PAIR_MASK) as usize,
            units: (receipt >> RECEIPT_UNITS_SHIFT) & RECEIPT_UNITS_MASK,
            huge: receipt & RECEIPT_CLASS_HUGE != 0,
            generation: (receipt >> RECEIPT_GENERATION_SHIFT) as u32,
        }
    }
}

// The packing only works while the configuration fits the fields. Pin every
// field's capacity where the compiler owns it: a drifted constant would
// otherwise truncate silently in a release image (a lane the field cannot
// name folds into another, so its completions stop matching and quota
// sticks).
const _: () = {
    assert!(
        LANES <= 1 << RECEIPT_PAIR_BITS,
        "every supply lane must be representable in the receipt's lane field"
    );
    assert!(
        HUGE_SUPPLY_QUOTA_MAX / HUGE_UNIT_BYTES <= RECEIPT_UNITS_MASK,
        "a supply of the whole huge ceiling must fit the receipt's unit count"
    );
    assert!(
        HUGE_SUPPLY_QUOTA_MAX % HUGE_UNIT_BYTES == 0,
        "the huge ceiling is a whole number of its receipt unit"
    );
    assert!(
        RECEIPT_GENERATION_SHIFT < 64 && 64 - RECEIPT_GENERATION_SHIFT >= 12,
        "a receipt keeps at least twelve generation bits"
    );
    assert!(
        PROVISIONED_RESERVATIONS <= RESERVATION_CEILING,
        "the provisioned table must be addressable by the receipt's index"
    );
    assert!(
        RESERVATION_CEILING <= u32::MAX as usize,
        "a free-stack entry names every slot"
    );
    assert!(
        RECEIPT_GENERATION_MAX < RESERVATION_LIVE,
        "the live bit must sit above the generation bits it shares a slot with"
    );
};

/// **The live-supply table behind every receipt** — one state word per
/// reservation slot (the live bit and the generation of its current or last
/// reservation) and a stack of the slots free to take, so a reservation and
/// its retirement are O(1) whatever the table's size. It grows with the
/// supplies live at once: provisioned at bind
/// ([`PROVISIONED_RESERVATIONS`]), doubled from mm's heap when a reservation
/// finds no slot free, never past its ceiling ([`RESERVATION_CEILING`]). A
/// slot whose generation is spent is retired for good, never taken again, so
/// no receipt can name two reservations.
///
/// mm's heap grows through the kernel's own carve-and-map, never through an
/// mm client path (`user/services/mm`'s global allocator), so a serve step
/// that grows the table cannot re-enter mm; growth is the only allocation a
/// serve step makes, and it is fallible.
#[derive(Debug)]
struct Reservations {
    /// Per slot: [`RESERVATION_LIVE`] while reserved, and the generation of
    /// its current or last reservation.
    states: Vec<u32>,
    /// The slots free to take, the next one on top. Its capacity is kept at
    /// least the table's, so freeing a slot never allocates.
    free: Vec<u32>,
    /// The most slots the table may grow to.
    ceiling: usize,
}

impl Reservations {
    /// A table of `provisioned` slots that grows to `ceiling`. A heap that
    /// cannot hold the provision starts it empty: the first reservation
    /// grows it, or is refused [`MmRefusal::Full`] if that cannot either.
    fn new(provisioned: usize, ceiling: usize) -> Self {
        let mut table = Self {
            states: Vec::new(),
            free: Vec::new(),
            ceiling,
        };
        let _ = table.grow_to(provisioned.min(ceiling));
        table
    }

    /// Grow the table to `len` slots, the new ones free. `false`, with the
    /// table as it was, when the heap cannot hold them.
    fn grow_to(&mut self, len: usize) -> bool {
        let old = self.states.len();
        if len <= old {
            return true;
        }
        if self.states.try_reserve_exact(len - old).is_err()
            || self.free.try_reserve_exact(len - self.free.len()).is_err()
        {
            return false;
        }
        self.states.resize(len, 0);
        // Highest first, so the lowest free slot is the next taken.
        self.free.extend((old..len).rev().map(|index| index as u32));
        true
    }

    /// Reserve a slot: `(index, generation)` of its new reservation. With no
    /// slot free the table doubles first; `None` at the ceiling, or when the
    /// heap cannot hold the larger table.
    fn take(&mut self) -> Option<(usize, u32)> {
        if self.free.is_empty() {
            let len = self.states.len();
            let next = len.saturating_mul(2).max(1).min(self.ceiling);
            if next == len || !self.grow_to(next) {
                return None;
            }
        }
        let index = self.free.pop()? as usize;
        let generation = (self.states[index] & RESERVATION_GENERATION_MASK) + 1;
        self.states[index] = RESERVATION_LIVE | generation;
        Some((index, generation))
    }

    /// End the reservation `generation` holds at `index`, freeing its slot
    /// unless the slot's generations are spent. `false`, and nothing
    /// changes, unless exactly that reservation is live there.
    fn end(&mut self, index: usize, generation: u32) -> bool {
        let Some(state) = self.states.get_mut(index) else {
            return false;
        };
        if generation == 0 || *state != RESERVATION_LIVE | generation {
            return false;
        }
        *state = generation;
        // Within capacity by construction (`grow_to` keeps the stack's
        // capacity at the table's), so freeing never allocates; a slot that
        // somehow found no room would be retired, never lost to a panic.
        if generation < RECEIPT_GENERATION_MAX && self.free.len() < self.free.capacity() {
            self.free.push(index as u32);
        }
        true
    }
}

/// The per-lane **supply meter** — [`MmRequest::Supply`]'s whole
/// bookkeeping, kept apart from the carve books because it is a *policy*
/// ledger over kernel-owned frames (the region bitmap is the ledger of
/// record), not a window pool.  Pure counters over a table, so the quota
/// arithmetic is host-testable without a bound service.
#[derive(Debug)]
struct SupplyMeter {
    /// Supplies minted.
    supplies: u32,
    /// Bytes those supplies span (the totals row).
    supply_bytes: u64,
    /// Outstanding base-class bytes per lane ([`meter_index`] of the
    /// arrival, [`Serve::via`] — structural identity, never a wire field),
    /// each bounded by [`base_quota`](Self::base_quota).
    pair_supply: [u64; LANES],
    /// Outstanding huge-class (2 MiB / 1 GiB aligned) bytes per lane, each
    /// bounded by [`huge_quota`](Self::huge_quota).
    pair_huge: [u64; LANES],
    /// The base-class per-lane quota ([`MmEndowment::base_quota`]).
    base_quota: u64,
    /// The huge-class per-lane quota ([`MmEndowment::huge_quota`]).
    huge_quota: u64,
    reservations: Reservations,
}

impl SupplyMeter {
    fn new(base_quota: u64, huge_quota: u64) -> Self {
        Self::with_table(
            base_quota,
            huge_quota,
            Reservations::new(PROVISIONED_RESERVATIONS, RESERVATION_CEILING),
        )
    }

    fn with_table(base_quota: u64, huge_quota: u64, reservations: Reservations) -> Self {
        Self {
            supplies: 0,
            supply_bytes: 0,
            pair_supply: [0; LANES],
            pair_huge: [0; LANES],
            base_quota,
            huge_quota: huge_quota.min(HUGE_SUPPLY_QUOTA_MAX),
            reservations,
        }
    }

    /// Validate one supply request and reserve it against `pair`'s meter:
    /// `(rounded len, align_log2, receipt)` on success.  Shape first
    /// ([`MmRefusal::BadProps`] — zero size, an alignment outside the
    /// `LeaseReq` vocabulary, an out-of-range lane, a supply no receipt can
    /// name), then policy ([`MmRefusal::Quota`]), then mm's own room for one
    /// more live supply ([`MmRefusal::Full`]).  The books advance only on
    /// success — a refused request perturbs nothing (a table grown on the
    /// way keeps its free slots).
    fn reserve(&mut self, pair: usize, size: u64, align: u64) -> Result<(u64, u8, u64), MmRefusal> {
        let align_log2 = match align {
            0 => 0,
            0x1000 => kore_abi::session::LEASE_ALIGN_4K,
            0x20_0000 => kore_abi::session::LEASE_ALIGN_2M,
            0x4000_0000 => kore_abi::session::LEASE_ALIGN_1G,
            _ => return Err(MmRefusal::BadProps),
        };
        if size == 0 || pair >= LANES {
            return Err(MmRefusal::BadProps);
        }
        // The alignment names the frame class (the kernel routes the lease
        // to that class's ledger), and a class supply is a whole number of
        // its units — the kernel refuses a 2 MiB-aligned lease that is not
        // 2 MiB-sized, so round here and meter the rounded figure.
        let huge = align_log2 >= kore_abi::session::LEASE_ALIGN_2M;
        let unit = if huge { 1u64 << align_log2 } else { PAGE_BYTES };
        let len = size
            .checked_next_multiple_of(unit)
            .ok_or(MmRefusal::BadProps)?;
        // Exact: `len` is a whole number of the class's rounding unit, and
        // each rounding unit (a page; 2 MiB or 1 GiB) is a whole number of
        // the class's receipt unit. A supply the field cannot name is refused,
        // never encoded as a smaller one — a truncated count would retire the
        // wrong span.
        let units = len / receipt_unit(huge);
        if units > RECEIPT_UNITS_MASK {
            return Err(MmRefusal::BadProps);
        }
        let (outstanding, quota) = if huge {
            (self.pair_huge[pair], self.huge_quota)
        } else {
            (self.pair_supply[pair], self.base_quota)
        };
        if outstanding.checked_add(len).is_none_or(|t| t > quota) {
            return Err(MmRefusal::Quota);
        }
        let (index, generation) = self.reservations.take().ok_or(MmRefusal::Full)?;
        let receipt = Receipt {
            index,
            pair,
            units,
            huge,
            generation,
        }
        .pack();

        self.supplies = self.supplies.saturating_add(1);
        self.supply_bytes = self.supply_bytes.saturating_add(len);
        if huge {
            self.pair_huge[pair] = outstanding + len;
        } else {
            self.pair_supply[pair] = outstanding + len;
        }
        Ok((len, align_log2, receipt))
    }

    /// Unwind one reservation (the grant session dropped).
    fn rollback(&mut self, receipt: u64, pair: usize, len: u64) {
        if self.retire(receipt, pair, len) {
            self.supplies -= 1;
            self.supply_bytes -= len;
        }
    }

    /// Consume one kernel-vouched completed lineage. Exact receipt, pair and
    /// length matching makes duplicate, stale, and cross-pair reports inert.
    /// The report names the supply's *lane* ([`meter_index`]).
    fn release(&mut self, report: SupplyReleaseReport) -> bool {
        self.retire(report.receipt, meter_index(report.pair), report.len)
    }

    /// Read only this arrival lane's policy and live accounting. Transferred
    /// supplies stay charged until the authenticated lineage release arrives.
    fn supply_budget(&self, pair: usize) -> (u64, u64) {
        (self.huge_quota, self.pair_huge[pair])
    }

    /// Whether any supply metered to `pair` is still outstanding.
    fn outstanding(&self, pair: usize) -> bool {
        self.pair_supply[pair] != 0 || self.pair_huge[pair] != 0
    }

    fn retire(&mut self, receipt: u64, pair: usize, len: u64) -> bool {
        let fields = Receipt::unpack(receipt);
        if fields.pair != pair
            || fields.units.checked_mul(receipt_unit(fields.huge)) != Some(len)
            || !self.reservations.end(fields.index, fields.generation)
        {
            return false;
        }
        if fields.huge {
            self.pair_huge[pair] -= len;
        } else {
            self.pair_supply[pair] -= len;
        }
        true
    }
}

/// Occupancy snapshot (the [`HeapStats`](kore_rt::heap::HeapStats) shape).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MmStats {
    /// Extents granted since bind (fresh + reused).
    pub extents: u32,
    /// Bytes granted since bind.
    pub granted_bytes: u64,
    /// Bytes returned via [`MmRequest::Release`].
    pub released_bytes: u64,
    /// Bytes served from returned windows instead of fresh supply.
    pub reused_bytes: u64,
    /// Returned bytes dropped because the free list was full — visible,
    /// never silent.
    pub leaked_bytes: u64,
    /// Fresh supply remaining past the bump cursor.
    pub remaining: u64,
    /// Root supply leases minted via [`MmRequest::Supply`].
    pub supplies: u32,
    /// Bytes those supplies span (page-rounded, cumulative — the region
    /// bitmap is the ledger of record; these are the meter's books).
    pub supply_bytes: u64,
}

/// The memory-manager service: the lessor over one supply region.
#[derive(Debug)]
pub struct MemoryManager {
    /// The possessed supply — every heap grant attenuates it.
    supply: SupplyToken,
    /// The possessed region grant — every [`MmRequest::Supply`] mint
    /// derives off it.
    region: RegionToken,
    /// Fresh-grant cursor (bump).
    cursor: u64,
    /// Returned windows awaiting reuse (supply-relative `(offset, len)`),
    /// first-fit.
    free: [Option<(u64, u64)>; FREE_SLOTS],
    /// Extents granted.
    extents: u32,
    /// Bytes granted.
    granted: u64,
    /// Bytes returned.
    released: u64,
    /// Bytes re-served from returns.
    reused: u64,
    /// Returned bytes dropped on free-list overflow.
    leaked: u64,
    /// The child-supply meter ([`MmRequest::Supply`]'s books).
    meter: SupplyMeter,
    /// Reservation made by the request currently passing through ServeHost.
    /// A successful grant is finalized when the next request begins; a
    /// dropped grant consumes this immediately to restore exact prior books.
    pending_grant: Option<CarveUndo>,
    /// Assembles one per-launch face from exec's returned components.
    peer_assembler: PeerBundleAssembler,
    /// Where admitted faces' rings map (`geom::TRANSIENT`); `None` before
    /// [`start_on_metal`](SystemService::start_on_metal) and off the metal.
    peer_map: Option<PeerMapPlan>,
    /// How many fixed pairs this row serves: the protocol's
    /// [`SERVE_PAIRS`](SvcProtocol::SERVE_PAIRS) plus one per manifest row
    /// declaring mm among its `clients` (the compositor's and image's heap
    /// faces). Their lanes drain like the protocol's, and no face is ever
    /// admitted onto one of their slots.
    fixed_pairs: usize,
    /// Per-launch faces by slot.
    peers: [Option<PeerLane>; LANES],
    /// The verdict of the preceding admission, until exec takes it.
    last_admission: Option<Result<PeerId, PeerAdmissionRefusal>>,
}

impl MemoryManager {
    /// Rights a granted window **delivers**: `ACCESS | MODIFY` — a mapped
    /// R/W *view*, and nothing more.
    ///
    /// A grant is a carve off mm's own supply **lease**, so the only child
    /// the algebra admits is a detached `LeasedRegion` view (an owned
    /// `PhysExtent` sub-carve would be a second release authority over the
    /// same frames).  A view is a **leaf** in the sharing sense: the
    /// sharing pivot stays with the lease-holder, which is mm.  It carries
    /// **no `REVOKE`**: teardown stays with the derivation chain — release
    /// rides [`MmRequest::Release`] (the `Revoke × REVOKE_KIND_RETURN`
    /// verb, whose authority is mm's named lease, not the view's rights).
    /// It carries **`DERIVE`** not as a sharing pivot (nothing derives
    /// from a view but the `map_view` exercise, which is `ACCESS`-gated)
    /// but because the rendezvous **transfer gate** demands it of anything
    /// presented to a send — without it a client could never return the
    /// grant.  The leaf posture is unharmed: revocation is T6-total, so a
    /// transferred-onward view still dies with mm's return or lease
    /// revoke; the worst a client can do with it is leak a window, never
    /// alias one (the return's scoped clear is translate-checked).
    ///
    /// These are the rights *the client ends up holding*.  The scratch
    /// derive the transfer stages carries `DERIVE` on top (the kernel's
    /// transfer gate demands it of anything presented to a send); the send
    /// arm clamps the moved cap back to exactly this set — see
    /// [`stage_grant`](kore_rt::transport::stage_grant).
    pub const VIEW_RIGHTS: Rights = Rights::ACCESS.union(Rights::MODIFY).union(Rights::DERIVE);

    /// Occupancy snapshot.
    pub fn stats(&self) -> MmStats {
        MmStats {
            extents: self.extents,
            granted_bytes: self.granted,
            released_bytes: self.released,
            reused_bytes: self.reused,
            leaked_bytes: self.leaked,
            remaining: self.supply.len() - self.cursor,
            supplies: self.meter.supplies,
            supply_bytes: self.meter.supply_bytes,
        }
    }

    /// Validate + carve one window: `(offset, len)` page-rounded and
    /// aligned to `max(props.align, PAGE_BYTES)` — served from a returned
    /// window when one fits (first-fit, exact-head alignment; the tail
    /// splits back), else from the bump cursor.  State advances only on
    /// success — a refused request perturbs nothing.
    fn carve(&mut self, p: &AllocProps) -> Result<(u64, u64, CarveUndo), MmRefusal> {
        validate_alloc_props(p)?;
        let align = p.align.max(PAGE_BYTES);
        let len = p
            .size
            .checked_next_multiple_of(PAGE_BYTES)
            .ok_or(MmRefusal::Exhausted)?;

        // Reuse first: a returned window whose head satisfies the
        // alignment and whose length covers the ask.
        for (index, slot) in self.free.iter_mut().enumerate() {
            if let Some((off, flen)) = *slot
                && off % align == 0
                && flen >= len
            {
                let tail = flen - len;
                *slot = (tail > 0).then_some((off + len, tail));
                self.extents += 1;
                self.granted += len;
                self.reused += len;
                return Ok((
                    off,
                    len,
                    CarveUndo::Reused {
                        index,
                        window: (off, flen),
                        len,
                    },
                ));
            }
        }

        let start = self
            .cursor
            .checked_next_multiple_of(align)
            .ok_or(MmRefusal::Exhausted)?;
        let end = start.checked_add(len).ok_or(MmRefusal::Exhausted)?;
        if end > self.supply.len() {
            return Err(MmRefusal::Exhausted);
        }
        let cursor = self.cursor;
        self.cursor = end;
        self.extents += 1;
        self.granted += len;
        Ok((start, len, CarveUndo::Fresh { cursor, len }))
    }

    fn rollback_grant(&mut self, undo: CarveUndo) {
        match undo {
            CarveUndo::Fresh { cursor, len } => {
                self.extents -= 1;
                self.cursor = cursor;
                self.granted -= len;
            }
            CarveUndo::Reused { index, window, len } => {
                self.extents -= 1;
                self.free[index] = Some(window);
                self.granted -= len;
                self.reused -= len;
            }
            CarveUndo::Supply { pair, len, receipt } => self.meter.rollback(receipt, pair, len),
        }
    }

    /// Reclaim a delivered extent: bind + retire it out of mm's arena
    /// slot, verify the window is **this** supply's, and pool it.
    fn take_back(
        &mut self,
        returned: Option<kore_rt::cap::ReturnedCapability>,
    ) -> Result<(), MmRefusal> {
        let base = self.supply.base();
        let (rel, len) = returned
            .and_then(|view| verbs::reclaim_returned_extent(view, &self.supply))
            .and_then(|(abs, len)| {
                let rel = abs.checked_sub(base)?;
                (rel.checked_add(len)? <= self.supply.len()).then_some((rel, len))
            })
            .ok_or(MmRefusal::BadReturn)?;
        self.released += len;
        match self.free.iter_mut().find(|s| s.is_none()) {
            Some(s) => *s = Some((rel, len)),
            // Full list: the window is dropped from circulation — counted,
            // never silent (coalescing/a real free structure is the later
            // tier; the books keep the loss honest).
            None => self.leaked += len,
        }
        Ok(())
    }
}

/// Validate only properties the current `SupplyToken` endowment can prove.
///
/// In particular, the ABI carrying `attrs` and `dma` is not evidence that the
/// kernel applied either property. The detached-window derive inherits the
/// source lease's write-back attributes, and its frozen request currently
/// treats DMA class as provenance-only. Rejecting stronger intent here keeps a
/// granted token truthful until a richer, kernel-authenticated supply witness
/// lands.
fn validate_alloc_props(p: &AllocProps) -> Result<(), MmRefusal> {
    if p.size == 0
        || !p.align.is_power_of_two()
        || p.numa != NumaNode::ANY
        || p.dma != DmaClass::None
        || p.attrs != SUPPLY_ALLOC_ATTRS
    {
        Err(MmRefusal::BadProps)
    } else {
        Ok(())
    }
}

impl SystemService for MemoryManager {
    const NAME: &'static str = "mm";
    type Endowment = MmEndowment;

    fn bind(e: MmEndowment) -> Self {
        Self {
            supply: e.supply,
            region: e.region,
            cursor: 0,
            free: [None; FREE_SLOTS],
            extents: 0,
            granted: 0,
            released: 0,
            reused: 0,
            leaked: 0,
            meter: SupplyMeter::new(e.base_quota, e.huge_quota),
            pending_grant: None,
            peer_assembler: PeerBundleAssembler::new(<Self as SvcProtocol>::REPLY_GRANTS, true),
            peer_map: None,
            fixed_pairs: <Self as SvcProtocol>::SERVE_PAIRS,
            peers: [None; LANES],
            last_admission: None,
        }
    }

    /// Per-launch faces arrive with their components on exec's return leg
    /// (rings, kick, grant channel, return receive), which a fixed landing
    /// cannot carry; `Release`'s extent returns ride the same retained leg.
    const RETURN_MODE: ReturnMode = ReturnMode::RetainedAny;

    fn start_on_metal(&mut self, rt: &kore_rt::entry::Rt) -> Result<(), StartFailure> {
        // The peer map places slot `n`'s ring at `base + n * stride`, and the
        // host can grow a face into any slot it can name: the window must
        // hold every lane, not just the bootstrap table.
        let window = rt
            .geom()
            .and_then(|geometry| geometry.window_of(kore_abi::task::geom::TRANSIENT))
            .filter(|window| {
                window.base() != 0
                    && window.base() & (PAGE_BYTES - 1) == 0
                    && window.len() >= PEER_RING_BYTES * LANES as u64
            })
            .ok_or(StartFailure::new(1))?;
        self.fixed_pairs =
            kore_rt::entry::service_pair_count::<Self>(rt).ok_or(StartFailure::new(2))?;
        self.peer_map = Some(PeerMapPlan {
            base: window.base(),
            stride: PEER_RING_BYTES,
        });
        Ok(())
    }

    /// Every engine program holds one face for its life (a JavaScript
    /// window, a Lua or Wasm run, an inference session), so the faces must
    /// grow with the programs, not stop at the bootstrap table's leftovers.
    fn peer_arena(&self) -> Option<&'static (dyn core::alloc::Allocator + Sync)> {
        Some(&FACE_ARENA)
    }

    fn serve(&mut self, req: MmRequest, sv: &mut Serve<'_, MmResponse>) {
        // ServeHost resolves every prior non-empty grant session before it
        // drives the next request, so reaching here finalizes that reservation.
        self.pending_grant = None;
        // The region bitmap has already accepted every reported return. Drain
        // before admission so sequential short-lived launches immediately
        // reuse quota even when their cumulative mints exceed the bound —
        // bounded per pair ([`SUPPLY_RELEASE_DRAIN_PER_PAIR`]), and each poll
        // names its pair so the kernel consults exactly that lifecycle lane.
        // Every fixed pair drains, the declared clients' as much as the
        // protocol's; and only a lane with a supply still outstanding is
        // polled — one whose row is zero has had every mint credited, so it
        // has nothing to report, and the poll would be a wasted kernel call
        // per idle face on every request.
        let fixed = (0..self.fixed_pairs).map(|pair| (pair, pair as u16));
        let faces = self
            .peers
            .iter()
            .enumerate()
            .filter_map(|(slot, lane)| lane.map(|lane| (slot, lane.arrival)));
        for (row, lane) in fixed.chain(faces) {
            if !self.meter.outstanding(row) {
                continue;
            }
            for _ in 0..SUPPLY_RELEASE_DRAIN_PER_PAIR {
                let Some(report) = verbs::take_supply_release(&self.region, lane) else {
                    break;
                };
                let _ = self.meter.release(report);
            }
        }
        let via = sv.via();
        let face = PeerId::from_arrival(via);
        // A face may only mint supplies, read the books and close itself:
        // allocation views and their returns are the fixed pairs' verbs.
        if face.is_some()
            && !matches!(
                req,
                MmRequest::Supply { .. } | MmRequest::Stats | MmRequest::ClosePeer
            )
        {
            sv.reply(MmResponse::Refused(MmRefusal::Denied));
            return;
        }
        match req {
            MmRequest::Alloc(props) => match self.carve(&props) {
                Ok((offset, len, undo)) => {
                    self.pending_grant = Some(undo);
                    // The staged derive: subject = the held supply, the
                    // carve + placement class ride the arg.  The client
                    // gets a token — a `LeasedRegion` view carrying exactly
                    // the delivered [`VIEW_RIGHTS`](Self::VIEW_RIGHTS).
                    sv.grant(
                        self.supply.slot(),
                        Self::VIEW_RIGHTS,
                        GrantArg::Window {
                            offset,
                            len,
                            class: props.dma,
                        },
                    );
                    sv.reply(MmResponse::Granted);
                }
                // Fail-closed: the refusal carries data, never authority.
                Err(r) => sv.reply(MmResponse::Refused(r)),
            },
            MmRequest::Release { slot } => {
                let returned = sv.returned_input(slot).and_then(|mut input| input.take());
                match self.take_back(returned) {
                    Ok(()) => sv.reply(MmResponse::Released),
                    Err(r) => sv.reply(MmResponse::Refused(r)),
                }
            }
            // The child-memory mint (slice D): validate + meter against
            // the ARRIVAL PAIR (structural identity, never a wire
            // field), then stage the `LeaseReq` derive off the region
            // grant — a fresh **root** lease, delivered whole at
            // [`SUPPLY_RIGHTS`].  The backing arrives zeroed by the
            // kernel's lease-derive scrub invariant; the region claim
            // itself happens inside the grant session, so a refused
            // session leaves the bitmap untouched (I3) and
            // `grants_dropped` restores the meter.
            MmRequest::Supply { size, align } => {
                // A face mints only while it is attached: a closing face's
                // lane is kept to credit returns, not to admit new debt.
                let admitted = match face {
                    Some(peer) => {
                        self.peers[peer.slot()].is_some_and(|lane| lane.live && lane.arrival == via)
                    }
                    None => usize::from(via) < self.fixed_pairs,
                };
                if !admitted {
                    sv.reply(MmResponse::Refused(MmRefusal::Denied));
                    return;
                }
                let pair = meter_index(via);
                match self.meter.reserve(pair, size, align) {
                    Ok((len, align_log2, receipt)) => {
                        self.pending_grant = Some(CarveUndo::Supply { pair, len, receipt });
                        sv.grant(
                            self.region.slot(),
                            SUPPLY_RIGHTS,
                            GrantArg::Lease {
                                size: len,
                                align_log2,
                                flags: 0,
                                release: Some(SupplyReleaseGrant { receipt, pair: via }),
                            },
                        );
                        sv.reply(MmResponse::Granted);
                    }
                    Err(r) => sv.reply(MmResponse::Refused(r)),
                }
            }
            MmRequest::Stats => {
                let pair = meter_index(via);
                let stats = self.stats();
                let (limit, used) = self.meter.supply_budget(pair);
                let snapshot = kore_abi::monitor::MemorySnapshot::new(
                    stats.extents,
                    stats.supplies,
                    stats.granted_bytes,
                    stats.released_bytes,
                    stats.reused_bytes,
                    stats.leaked_bytes,
                    stats.remaining,
                    stats.supply_bytes,
                ).with_supply_budget(limit, used);
                sv.reply(match snapshot {
                    Some(snapshot) => MmResponse::Stats(snapshot),
                    None => MmResponse::Refused(MmRefusal::Denied),
                });
            }
            MmRequest::AdmitPeer { part, announced } => {
                // A `Ring` is the boundary between bundles, so one arriving
                // while the assembler still owns components proves the
                // previous assembly was abandoned — its asker spent its
                // bounded re-ask and gave up.  Retire what it left, and
                // refuse the replacement while any of those owners is still
                // closing (`PeerBundleAssembler::retire_incomplete`).
                //
                // Without this the leftovers are permanent: the next launch's
                // matching part answers `Duplicate`, the bundle never
                // completes, no verdict is ever published, and every engine
                // launch for the rest of the boot refuses.
                //
                // Admin only, and evaluated before anything else touches the
                // assembler: a foreign via must never retire the components
                // the administration pair is still assembling.
                let abandoned = via == ADMIN_VIA
                    && matches!(part, PeerBundlePart::Ring)
                    && !self.peer_assembler.is_empty()
                    && !self.peer_assembler.retire_incomplete();
                let reply = if via != ADMIN_VIA || self.last_admission.is_some() || abandoned {
                    MmResponse::Refused(MmRefusal::Pending)
                } else if let (Some(map), Some(mut input)) =
                    (self.peer_map, sv.returned_input(announced))
                {
                    let fits = input
                        .capability()
                        .and_then(|capability| self.peer_assembler.accepts(part, capability).ok())
                        .is_some();
                    match input.take().filter(|_| fits) {
                        Some(capability) => match self.peer_assembler.push(part, capability) {
                            Ok(Some(bundle)) => {
                                sv.admit_peer(bundle, map);
                                MmResponse::Accepted
                            }
                            Ok(None) => MmResponse::Accepted,
                            // `accepts` just validated this part, so a push
                            // cannot refuse it — but answering `Accepted` for
                            // a component that was not taken would promise an
                            // admission that can never be published.
                            Err(_) => MmResponse::Refused(MmRefusal::BadReturn),
                        },
                        None => MmResponse::Refused(MmRefusal::BadReturn),
                    }
                } else {
                    MmResponse::Refused(MmRefusal::Denied)
                };
                sv.reply(reply);
            }
            MmRequest::TakeAdmission => sv.reply(if via != ADMIN_VIA {
                MmResponse::Refused(MmRefusal::Denied)
            } else {
                match self.last_admission.take() {
                    Some(Ok(peer)) => MmResponse::PeerAdmitted { peer },
                    // No room for one more face — the host's slots, its
                    // arena, or mm's capability arena — is capacity, which
                    // the launcher reports as a full table, apart from a face
                    // that arrived wrong.
                    Some(Err(refusal)) => MmResponse::Refused(admission_refusal(refusal)),
                    // No verdict, and the assembler still owns components:
                    // this bundle never completed, so `admit_peer` was never
                    // called and no verdict can EVER be published.  `Pending`
                    // here would be a lie the asker waits on forever — and
                    // `provision_mm_peer` fences new provisioning while an
                    // id-less face exists, so that wait refuses every later
                    // engine launch for the rest of the boot.  Retire the
                    // abandoned assembly and answer terminally, which is
                    // exec's signal to let the peer go.
                    None if !self.peer_assembler.is_empty() => {
                        let _ = self.peer_assembler.retire_incomplete();
                        MmResponse::Refused(MmRefusal::BadReturn)
                    }
                    // An EMPTY assembler means the bundle completed and the
                    // host has not drained its admission action yet: a
                    // verdict IS coming, and settling on it would orphan the
                    // face mm is about to admit.
                    None => MmResponse::Refused(MmRefusal::Pending),
                }
            }),
            MmRequest::RetirePeer { peer } => {
                let live = self.peers[peer.slot()]
                    .is_some_and(|lane| lane.live && lane.arrival == peer.arrival());
                if via == ADMIN_VIA && live {
                    sv.retire_peer(peer, PeerClose::Orderly);
                    sv.reply(MmResponse::Accepted);
                } else {
                    sv.reply(MmResponse::Refused(MmRefusal::Denied));
                }
            }
            MmRequest::ConfirmRetired { peer } => sv.reply(if via != ADMIN_VIA {
                MmResponse::Refused(MmRefusal::Denied)
            } else if self.peers[peer.slot()].is_some_and(|lane| lane.arrival == peer.arrival()) {
                MmResponse::Refused(MmRefusal::Pending)
            } else {
                MmResponse::Accepted
            }),
            MmRequest::ClosePeer => {
                if face.is_some() {
                    sv.close_peer(PeerClose::Orderly);
                    sv.reply(MmResponse::Accepted);
                } else {
                    sv.reply(MmResponse::Refused(MmRefusal::Denied));
                }
            }
        }
    }

    fn peer_admitted(&mut self, result: Result<PeerId, PeerAdmissionRefusal>) -> bool {
        if let Ok(peer) = result {
            // A face never lands on a fixed pair's index — the declared
            // clients' included — nor on a slot whose previous face still
            // owes returns.
            if peer.slot() < self.fixed_pairs || self.peers[peer.slot()].is_some() {
                self.last_admission = Some(Err(PeerAdmissionRefusal::PolicyCollision));
                return false;
            }
            self.peers[peer.slot()] = Some(PeerLane {
                arrival: peer.arrival(),
                live: true,
            });
        }
        self.last_admission = Some(result);
        true
    }

    /// A closed face's slot stays closing — unreusable, its lane still
    /// drained — until every supply it minted has returned: a supply is
    /// conveyed onward at spawns and credits only when its whole lineage
    /// dies, possibly long after the face closed.
    fn peer_closed(&mut self, via: u16, _reason: PeerClose) -> bool {
        let Some(slot) = PeerId::from_arrival(via).map(|peer| peer.slot()) else {
            return true;
        };
        match &mut self.peers[slot] {
            Some(lane) if lane.arrival == via => {
                lane.live = false;
                if self.meter.outstanding(slot) {
                    return false;
                }
                self.peers[slot] = None;
                true
            }
            _ => true,
        }
    }

    fn grants_dropped(&mut self, response: &mut MmResponse, emitted: usize) {
        if emitted == 1 && matches!(response, MmResponse::Granted) {
            if let Some(undo) = self.pending_grant.take() {
                self.rollback_grant(undo);
            }
            *response = MmResponse::Refused(MmRefusal::Denied);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 4 KiB class a 4 GiB QEMU guest carves (`kore: mem classes — …
    /// 4K 737 MiB`), and the base quota every lane derives from it.
    const GUEST_CLASS: u64 = 737 << 20;
    const BASE: u64 = base_quota_for(GUEST_CLASS);

    /// A meter over a table of exactly `slots` reservations that cannot
    /// grow, so a test can fill it.
    fn fixed(slots: usize, huge_quota: u64) -> SupplyMeter {
        SupplyMeter::with_table(BASE, huge_quota, Reservations::new(slots, slots))
    }

    fn cpu_props() -> AllocProps {
        AllocProps {
            size: PAGE_BYTES,
            align: PAGE_BYTES,
            numa: NumaNode::ANY,
            dma: DmaClass::None,
            attrs: SUPPLY_ALLOC_ATTRS,
            contiguous: false,
        }
    }

    /// Allocation admission is a proof boundary: accept only the exact
    /// properties the geometry-only supply endowment can witness.
    #[test]
    fn allocation_props_refuse_unproven_placement_and_cache_claims() {
        let base = cpu_props();
        assert_eq!(validate_alloc_props(&base), Ok(()));

        let contiguous = AllocProps {
            contiguous: true,
            ..base
        };
        assert_eq!(validate_alloc_props(&contiguous), Ok(()));

        for malformed in [
            AllocProps { size: 0, ..base },
            AllocProps { align: 3, ..base },
        ] {
            assert_eq!(validate_alloc_props(&malformed), Err(MmRefusal::BadProps));
        }

        let named_numa = AllocProps {
            numa: NumaNode(0),
            ..base
        };
        assert_eq!(validate_alloc_props(&named_numa), Err(MmRefusal::BadProps));

        for dma in [DmaClass::Dma32, DmaClass::Dma64] {
            let requested = AllocProps { dma, ..base };
            assert_eq!(validate_alloc_props(&requested), Err(MmRefusal::BadProps));
        }

        for attrs in [
            MemAttrs::READ | MemAttrs::WRITE,
            SUPPLY_ALLOC_ATTRS | MemAttrs::UNCACHED,
            MemAttrs::WRITE_COMBINE | MemAttrs::READ | MemAttrs::WRITE,
        ] {
            let requested = AllocProps { attrs, ..base };
            assert_eq!(validate_alloc_props(&requested), Err(MmRefusal::BadProps));
        }
    }

    #[test]
    fn deployed_quota_admits_lfm_model_and_launch_headroom() {
        let quota: u64 = include_str!("../../boot.toml")
            .lines()
            .find_map(|line| line.strip_prefix("supply_quota_bytes = "))
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap();
        let model_bytes = 2_734_526_464;
        let mut old = SupplyMeter::new(BASE, 2 << 30);
        assert_eq!(old.reserve(0, model_bytes, 2 << 20), Err(MmRefusal::Quota));
        let mut deployed = SupplyMeter::new(BASE, quota);
        let (rounded, _, _) = deployed.reserve(0, model_bytes, 2 << 20).unwrap();
        assert!(rounded >= model_bytes);
        assert!(deployed.reserve(0, 512 << 20, 2 << 20).is_ok());
        assert_eq!(deployed.reserve(0, quota, 2 << 20), Err(MmRefusal::Quota));
    }

    #[test]
    fn live_budget_is_per_face_and_only_exact_retirement_reclaims_it() {
        let mut meter = fixed(8, 26 << 30);
        let (bytes, _, receipt) = meter.reserve(0, 24 << 30, 1 << 30).unwrap();
        let view = |meter: &SupplyMeter, pair| {
            let (limit, used) = meter.supply_budget(pair);
            kore_abi::monitor::MemorySnapshot::new(0, 0, 0, 0, 0, 0, 0, 0)
                .with_supply_budget(limit, used).unwrap().huge_supply_remaining().unwrap()
        };
        assert_eq!(view(&meter, 0), 2 << 30);
        assert_eq!(view(&meter, 1), 26 << 30);
        assert_eq!(meter.reserve(0, 3 << 30, 1 << 30), Err(MmRefusal::Quota));
        assert_eq!(view(&meter, 0), 2 << 30);
        assert!(!meter.retire(receipt, 1, bytes));
        assert_eq!(view(&meter, 0), 2 << 30);
        assert!(meter.retire(receipt, 0, bytes));
        assert_eq!(view(&meter, 0), 26 << 30);
        assert!(!meter.retire(receipt, 0, bytes));
    }

    #[test]
    fn deployed_quota_keeps_chunk_headroom_with_eight_large_units() {
        let quota: u64 = include_str!("../../boot.toml")
            .lines()
            .find_map(|line| line.strip_prefix("supply_quota_bytes = "))
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap();
        const GIB: u64 = 1 << 30;
        const MIB: u64 = 1 << 20;
        // Inference's existing 256 MiB heap and 96 MiB GPU scratch are
        // mandatory before its elastic 512/128 MiB cache chunks. The old
        // 11 GiB policy could retain 3712 MiB of chunks beside seven units.
        // NPU programs have artifact-derived charges, tested at admission;
        // this lower-bound case does not invent a fixed program reserve.
        let chunks = [512 * MIB; 7].into_iter().chain([128 * MIB]);
        let mut previous = fixed(16, 11 * GIB);
        for (bytes, class) in [
            (7 * GIB, GIB),
            (256 * MIB, 2 * MIB),
            (96 * MIB, 2 * MIB),
        ] {
            previous.reserve(0, bytes, class).unwrap();
        }
        for bytes in chunks.clone() {
            previous.reserve(0, bytes, 2 * MIB).unwrap();
        }
        // Both launch layouts consume eight large units: model only, or
        // one resident NPU weight supply followed by a seven-unit model.
        for weight_units in [0, 1] {
            let mut deployed = fixed(16, quota);
            let mut old = fixed(16, 11 * GIB);
            for meter in [&mut deployed, &mut old] {
                meter.reserve(0, 256 * MIB, 2 * MIB).unwrap();
                if weight_units != 0 {
                    meter.reserve(0, weight_units * GIB, GIB).unwrap();
                }
                meter.reserve(0, (8 - weight_units) * GIB, GIB).unwrap();
                meter.reserve(0, 96 * MIB, 2 * MIB).unwrap();
            }
            let mut old_refused = false;
            for bytes in chunks.clone() {
                deployed.reserve(0, bytes, 2 * MIB).unwrap();
                match old.reserve(0, bytes, 2 * MIB) {
                    Ok(_) => {}
                    Err(MmRefusal::Quota) => old_refused = true,
                    other => panic!("unexpected old-quota result: {other:?}"),
                }
            }
            assert!(old_refused, "the old policy must displace external chunks");
            assert_eq!(deployed.pair_huge[0], previous.pair_huge[0] + GIB);
            let remaining = quota - deployed.pair_huge[0];
            deployed.reserve(0, remaining, 2 * MIB).unwrap();
            assert_eq!(deployed.reserve(0, 2 * MIB, 2 * MIB), Err(MmRefusal::Quota));
            // Raising huge allowance does not raise the separate base meter.
            deployed.reserve(0, BASE, 0).unwrap();
            assert_eq!(deployed.reserve(0, PAGE_BYTES, 0), Err(MmRefusal::Quota));
        }
    }

    /// The meter's whole vocabulary: rounding, the closed alignment set,
    /// the zero refusal, and the out-of-range pair.
    #[test]
    fn the_supply_meter_speaks_the_lease_req_vocabulary() {
        let mut m = SupplyMeter::new(BASE, HUGE_SUPPLY_QUOTA_MAX);
        // Page rounding + the default alignment.
        let (len, align, _) = m.reserve(0, 1, 0).unwrap();
        assert_eq!((len, align), (PAGE_BYTES, 0));
        // The three explicit alignments map to their log2 constants.
        let (len, align, _) = m.reserve(0, 0x1000, 0x1000).unwrap();
        assert_eq!((len, align), (0x1000, kore_abi::session::LEASE_ALIGN_4K));
        let (len, align, _) = m.reserve(0, 0x20_0000, 0x20_0000).unwrap();
        assert_eq!((len, align), (0x20_0000, kore_abi::session::LEASE_ALIGN_2M));
        // A class supply is a whole number of its units (the kernel refuses
        // a 1 GiB-aligned lease that is not 1 GiB-sized).
        let (len, align, _) = m.reserve(0, 0x1000, 0x4000_0000).unwrap();
        assert_eq!(
            (len, align),
            (0x4000_0000, kore_abi::session::LEASE_ALIGN_1G)
        );
        // Out-of-vocabulary shapes refuse without touching the books.
        let before = (m.supplies, m.supply_bytes);
        assert_eq!(m.reserve(0, 0, 0), Err(MmRefusal::BadProps));
        assert_eq!(m.reserve(0, 0x1000, 0x2000), Err(MmRefusal::BadProps));
        assert_eq!(m.reserve(LANES, 0x1000, 0), Err(MmRefusal::BadProps));
        assert_eq!((m.supplies, m.supply_bytes), before);
    }

    /// Each pair meters independently, the bound is exact, and a
    /// rollback restores the meter to the byte.
    #[test]
    fn the_quota_meters_per_pair_and_rolls_back_exactly() {
        let mut m = SupplyMeter::new(BASE, HUGE_SUPPLY_QUOTA_MAX);
        // Fill pair 1 (the shell's face) to the exact quota.
        let (_, _, pair_one) = m.reserve(1, BASE, 0).unwrap();
        // One more page on the same pair is a policy refusal…
        assert_eq!(m.reserve(1, PAGE_BYTES, 0), Err(MmRefusal::Quota));
        // …while pair 2 (init's face) still has its whole allowance.
        let (_, _, _) = m.reserve(2, PAGE_BYTES, 0).unwrap();
        assert_eq!(m.supplies, 2);
        assert_eq!(m.supply_bytes, BASE + PAGE_BYTES);
        // A dropped grant unwinds pair 1 whole; the allowance returns.
        m.rollback(pair_one, 1, BASE);
        assert_eq!(m.supplies, 1);
        assert_eq!(m.supply_bytes, PAGE_BYTES);
        let (_, _, _) = m.reserve(1, PAGE_BYTES, 0).unwrap();
        assert_eq!(m.pair_supply[1], PAGE_BYTES);
    }

    /// Huge-class supplies meter against the datum quota, independently of
    /// the base class; a zero datum refuses every huge request; completions
    /// credit the class their receipt names.
    #[test]
    fn huge_class_supplies_meter_against_the_datum_not_the_base_quota() {
        let mut m = SupplyMeter::new(BASE, 4 << 30);
        // Base class is untouched by a 3 GiB huge supply...
        let (len, _, huge) = m.reserve(1, 3 << 30, 0x4000_0000).unwrap();
        assert_eq!(len, 3 << 30);
        assert_eq!(m.pair_supply[1], 0);
        assert_eq!(m.pair_huge[1], 3 << 30);
        // ...and the base quota is still whole.
        let (_, _, _) = m.reserve(1, BASE, 0).unwrap();
        // The huge quota is exact: 2 MiB units past 4 GiB refuse.
        let (_, _, _) = m.reserve(1, 1 << 30, 0x20_0000).unwrap();
        assert_eq!(m.reserve(1, 0x20_0000, 0x20_0000), Err(MmRefusal::Quota));
        // Completion credits the huge meter, not the base one.
        assert!(m.release(SupplyReleaseReport {
            receipt: huge,
            pair: 1,
            len: 3 << 30,
        }));
        assert_eq!(m.pair_huge[1], 1 << 30);
        assert_eq!(m.pair_supply[1], BASE);

        // No datum ⇒ no huge quota, loudly.
        let mut none = SupplyMeter::new(BASE, 0);
        assert_eq!(none.reserve(0, 0x20_0000, 0x20_0000), Err(MmRefusal::Quota));
        let (_, _, _) = none.reserve(0, 0x1000, 0).unwrap();
    }

    #[test]
    fn completed_lineages_reuse_quota_exactly_once() {
        let mut m = fixed(2, HUGE_SUPPLY_QUOTA_MAX);
        let (_, _, first) = m.reserve(1, BASE, 0).unwrap();
        assert_eq!(m.reserve(1, PAGE_BYTES, 0), Err(MmRefusal::Quota));

        let report = SupplyReleaseReport {
            receipt: first,
            pair: 1,
            len: BASE,
        };
        assert!(m.release(report));
        assert!(!m.release(report), "duplicate completion must be inert");
        assert_eq!(m.pair_supply[1], 0);

        // Cumulative mints may now exceed the policy bound while live bytes do
        // not; simultaneous live bytes remain bounded.
        let (_, _, _) = m.reserve(1, BASE, 0).unwrap();
        assert_eq!(m.supply_bytes, 2 * BASE);
        assert_eq!(m.reserve(1, PAGE_BYTES, 0), Err(MmRefusal::Quota));
    }

    /// Every field round-trips at its maximum, beside every other field at
    /// its maximum, so no field can bleed into its neighbour.
    #[test]
    fn receipts_round_trip_at_every_fields_maximum() {
        for huge in [false, true] {
            let top = Receipt {
                index: RESERVATION_CEILING - 1,
                pair: LANES - 1,
                units: RECEIPT_UNITS_MASK,
                huge,
                generation: RECEIPT_GENERATION_MAX,
            };
            assert_eq!(Receipt::unpack(top.pack()), top);
            let one = Receipt {
                index: 1,
                pair: 1,
                units: 1,
                huge,
                generation: 1,
            };
            assert_eq!(Receipt::unpack(one.pack()), one);
        }
    }

    /// The last slot of a table on the highest lane round-trips through the
    /// meter, and its receipt credits that lane once and no other.
    #[test]
    fn the_last_slot_on_the_last_lane_releases_exactly_once() {
        let mut meter = fixed(4, HUGE_SUPPLY_QUOTA_MAX);
        let pair = LANES - 1;
        for _ in 0..3 {
            meter.reserve(0, PAGE_BYTES, 0).unwrap();
        }
        let (_, _, receipt) = meter.reserve(pair, BASE, 0).unwrap();
        assert_eq!(Receipt::unpack(receipt).index, 3);
        let report = |pair: usize| SupplyReleaseReport {
            receipt,
            pair: pair as u16,
            len: BASE,
        };
        assert!(!meter.release(report(0)));
        assert!(meter.release(report(pair)));
        assert_eq!(meter.pair_supply[pair], 0);
        assert!(!meter.release(report(pair)));
    }

    #[test]
    fn stale_mismatched_and_reordered_completions_are_inert() {
        let mut m = fixed(2, HUGE_SUPPLY_QUOTA_MAX);
        let (_, _, a) = m.reserve(0, PAGE_BYTES, 0).unwrap();
        let (_, _, b) = m.reserve(0, PAGE_BYTES, 0).unwrap();

        assert!(!m.release(SupplyReleaseReport {
            receipt: a,
            pair: 1,
            len: PAGE_BYTES,
        }));
        assert!(!m.release(SupplyReleaseReport {
            receipt: a,
            pair: 0,
            len: 2 * PAGE_BYTES,
        }));
        assert_eq!(m.pair_supply[0], 2 * PAGE_BYTES);

        assert!(m.release(SupplyReleaseReport {
            receipt: b,
            pair: 0,
            len: PAGE_BYTES,
        }));
        assert!(m.release(SupplyReleaseReport {
            receipt: a,
            pair: 0,
            len: PAGE_BYTES,
        }));
        assert_eq!(m.pair_supply[0], 0);

        // Reuse both slots, then prove the old generation cannot release the
        // new reservation occupying the same index.
        let (_, _, _) = m.reserve(0, PAGE_BYTES, 0).unwrap();
        let (_, _, _) = m.reserve(0, PAGE_BYTES, 0).unwrap();
        assert!(!m.release(SupplyReleaseReport {
            receipt: a,
            pair: 0,
            len: PAGE_BYTES,
        }));
        assert_eq!(m.pair_supply[0], 2 * PAGE_BYTES);
    }

    #[test]
    fn failed_grant_rollback_cannot_later_double_release() {
        let mut m = fixed(1, HUGE_SUPPLY_QUOTA_MAX);
        let (_, _, receipt) = m.reserve(3, PAGE_BYTES, 0).unwrap();
        m.rollback(receipt, 3, PAGE_BYTES);
        assert_eq!(m.pair_supply[3], 0);
        assert!(!m.release(SupplyReleaseReport {
            receipt,
            pair: 3,
            len: PAGE_BYTES,
        }));
        assert_eq!((m.supplies, m.supply_bytes), (0, 0));
    }

    /// A lane is booked to its fixed pair or its face's slot, and every
    /// arrival the host can stamp names a row — the declared clients' fixed
    /// pairs past the protocol's, and faces grown past the bootstrap table.
    #[test]
    fn a_lane_books_to_its_pair_or_its_face_slot() {
        assert_eq!(meter_index(0), 0);
        assert_eq!(meter_index(3), 3);
        assert_eq!(meter_index(5), 5, "a declared client's fixed pair");
        let face = PeerId::from_arrival(0x0105).unwrap();
        assert_eq!(meter_index(face.arrival()), 5);
        let grown = PeerId::from_arrival(0x01ff).unwrap();
        assert_eq!(meter_index(grown.arrival()), LANES - 1);
        for lane in [0, u8::MAX as u16, u8::MAX as u16 + 1, u16::MAX] {
            assert!(meter_index(lane) < LANES, "lane {lane:#x} has a row");
        }
    }

    /// A face grown past the bootstrap table meters, encodes and releases
    /// on its own lane, and its receipt cannot credit the slot's neighbour.
    #[test]
    fn a_face_past_the_bootstrap_table_meters_on_its_own_lane() {
        let face = PeerId::from_arrival(0x0140).unwrap();
        assert!(face.slot() >= kore_rt::svc::MAX_SERVE_PAIRS);
        let pair = meter_index(face.arrival());
        let mut m = fixed(4, HUGE_SUPPLY_QUOTA_MAX);
        let (len, _, receipt) = m.reserve(pair, 5 * PAGE_BYTES, 0).unwrap();
        assert_eq!(
            ((receipt >> RECEIPT_PAIR_SHIFT) & RECEIPT_PAIR_MASK) as usize,
            pair
        );
        let neighbour = PeerId::from_arrival(0x0141).unwrap();
        assert!(!m.release(SupplyReleaseReport {
            receipt,
            pair: neighbour.arrival(),
            len,
        }));
        assert!(m.release(SupplyReleaseReport {
            receipt,
            pair: face.arrival(),
            len,
        }));
        assert!(!m.outstanding(pair));
    }

    /// Each class counts its receipt in its own unit: the largest supply
    /// either quota admits round-trips exactly, and a completion whose
    /// length disagrees with the receipt's count in its class's unit is
    /// inert — a base receipt cannot be read in 2 MiB units, nor a huge one
    /// in pages.
    #[test]
    fn receipts_count_each_class_in_its_own_unit() {
        let mut m = fixed(4, HUGE_SUPPLY_QUOTA_MAX);
        let (base_len, _, base) = m.reserve(7, BASE, 0).unwrap();
        let (huge_len, _, huge) = m.reserve(7, HUGE_SUPPLY_QUOTA_MAX, 0x4000_0000).unwrap();
        assert_eq!((base_len, huge_len), (BASE, HUGE_SUPPLY_QUOTA_MAX));
        assert_eq!(
            (base >> RECEIPT_UNITS_SHIFT) & RECEIPT_UNITS_MASK,
            BASE / PAGE_BYTES
        );
        assert_eq!(
            (huge >> RECEIPT_UNITS_SHIFT) & RECEIPT_UNITS_MASK,
            HUGE_SUPPLY_QUOTA_MAX / HUGE_UNIT_BYTES
        );
        // Read in the other class's unit, each count names a different span.
        let units = (base >> RECEIPT_UNITS_SHIFT) & RECEIPT_UNITS_MASK;
        assert!(!m.release(SupplyReleaseReport {
            receipt: base,
            pair: 7,
            len: units * HUGE_UNIT_BYTES,
        }));
        assert!(m.release(SupplyReleaseReport {
            receipt: base,
            pair: 7,
            len: BASE,
        }));
        assert!(m.release(SupplyReleaseReport {
            receipt: huge,
            pair: 7,
            len: HUGE_SUPPLY_QUOTA_MAX,
        }));
        assert!(!m.outstanding(7));
    }

    /// No room for one more face is capacity; a face that arrived wrong is
    /// a denial — the two refusals the launcher tells apart.
    #[test]
    fn a_full_face_table_is_capacity_and_a_malformed_face_is_denied() {
        assert_eq!(
            admission_refusal(PeerAdmissionRefusal::Capacity),
            MmRefusal::Full
        );
        assert_eq!(
            admission_refusal(PeerAdmissionRefusal::NoScratchSlots),
            MmRefusal::Full
        );
        for wrong in [
            PeerAdmissionRefusal::InvalidMapPlan,
            PeerAdmissionRefusal::WrongRingAuthority,
            PeerAdmissionRefusal::WrongEndpointAuthority,
            PeerAdmissionRefusal::WrongGrantCount,
            PeerAdmissionRefusal::WrongReturnShape,
            PeerAdmissionRefusal::RingGeometry,
            PeerAdmissionRefusal::MapFailed,
            PeerAdmissionRefusal::RingMalformed,
            PeerAdmissionRefusal::PolicyCollision,
            PeerAdmissionRefusal::PolicyUnavailable,
        ] {
            assert_eq!(admission_refusal(wrong), MmRefusal::Denied, "{wrong:?}");
        }
    }

    /// A face's returns credit its slot's row, keyed by its arrival.
    #[test]
    fn a_face_supply_is_metered_and_released_on_its_slot() {
        let face = PeerId::from_arrival(0x0104).unwrap();
        let mut m = fixed(8, 0);
        let pair = meter_index(face.arrival());
        let (len, _, receipt) = m.reserve(pair, 3 * PAGE_BYTES, 0).unwrap();
        assert!(m.outstanding(pair));
        // A report on the wrong lane is inert.
        assert!(!m.release(SupplyReleaseReport {
            receipt,
            pair: 3,
            len,
        }));
        assert!(m.release(SupplyReleaseReport {
            receipt,
            pair: face.arrival(),
            len,
        }));
        assert!(!m.outstanding(pair));
    }

    /// A lane's base quota is half the 4 KiB class, in whole pages: on the
    /// 4 GiB guest every gate boots, more than the 128 MiB constant it
    /// replaced, and on a larger machine proportionally more.
    #[test]
    fn the_base_quota_is_half_the_class_in_whole_pages() {
        assert_eq!(BASE, GUEST_CLASS / 2);
        assert_eq!(BASE % PAGE_BYTES, 0);
        assert!(BASE > 128 << 20);
        assert_eq!(base_quota_for(3 * PAGE_BYTES), PAGE_BYTES);
        assert_eq!(base_quota_for(PAGE_BYTES), 0);
        assert_eq!(base_quota_for(0), 0);
        let laptop = base_quota_for(3_276 << 20);
        assert_eq!(laptop, 1_638 << 20);
    }

    /// Past its provision the table doubles up to its ceiling; there one more
    /// live supply is mm's capacity refused, not the lane's quota, and a
    /// returned supply's slot serves the next under a new generation, which
    /// the old receipt cannot name.
    #[test]
    fn the_table_grows_past_its_provision_up_to_its_ceiling() {
        let mut m = SupplyMeter::with_table(BASE, 0, Reservations::new(2, 8));
        let receipts: Vec<u64> = (0..8)
            .map(|_| m.reserve(3, PAGE_BYTES, 0).unwrap().2)
            .collect();
        assert_eq!(m.reservations.states.len(), 8);
        assert_eq!(m.reserve(3, PAGE_BYTES, 0), Err(MmRefusal::Full));
        assert_eq!(m.pair_supply[3], 8 * PAGE_BYTES, "a refusal books nothing");
        let report = |receipt| SupplyReleaseReport {
            receipt,
            pair: 3,
            len: PAGE_BYTES,
        };
        assert!(m.release(report(receipts[5])));
        let (_, _, again) = m.reserve(3, PAGE_BYTES, 0).unwrap();
        let fields = Receipt::unpack(again);
        assert_eq!((fields.index, fields.generation), (5, 2));
        assert!(
            !m.release(report(receipts[5])),
            "the old generation is inert"
        );
        assert!(m.release(report(again)));
    }

    /// A heap that could not hold the provision leaves the table empty, and
    /// the first supplies grow it.
    #[test]
    fn an_unprovisioned_table_grows_from_nothing() {
        let mut m = SupplyMeter::with_table(BASE, 0, Reservations::new(0, 4));
        assert!(m.reservations.states.is_empty());
        for _ in 0..4 {
            m.reserve(0, PAGE_BYTES, 0).unwrap();
        }
        assert_eq!(m.reserve(0, PAGE_BYTES, 0), Err(MmRefusal::Full));
    }

    /// A slot taken as many times as its generation can count is retired for
    /// good: no receipt ever names two reservations.
    #[test]
    fn a_slot_whose_generations_are_spent_is_retired() {
        let mut m = fixed(1, 0);
        for generation in 1..=RECEIPT_GENERATION_MAX {
            let (_, _, receipt) = m.reserve(0, PAGE_BYTES, 0).unwrap();
            assert_eq!(Receipt::unpack(receipt).generation, generation);
            assert!(m.release(SupplyReleaseReport {
                receipt,
                pair: 0,
                len: PAGE_BYTES,
            }));
        }
        assert_eq!(m.reserve(0, PAGE_BYTES, 0), Err(MmRefusal::Full));
    }

    /// The largest base supply a receipt names is admitted and round-trips;
    /// a page more is malformed whatever the quota, and the huge classes
    /// still take far more in their own unit.
    #[test]
    fn a_base_supply_no_receipt_can_name_is_malformed() {
        let mut m =
            SupplyMeter::with_table(u64::MAX, HUGE_SUPPLY_QUOTA_MAX, Reservations::new(4, 4));
        let (len, _, receipt) = m.reserve(0, MAX_BASE_SUPPLY_BYTES, 0).unwrap();
        assert_eq!(len, MAX_BASE_SUPPLY_BYTES);
        assert_eq!(Receipt::unpack(receipt).units, RECEIPT_UNITS_MASK);
        assert_eq!(
            m.reserve(0, MAX_BASE_SUPPLY_BYTES + 1, 0),
            Err(MmRefusal::BadProps)
        );
        assert!(m.reserve(0, 8 << 30, 0x4000_0000).is_ok());
    }
}
