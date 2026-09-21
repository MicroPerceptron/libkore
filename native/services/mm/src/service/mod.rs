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
//! Serving is **allocation-free** (bump cursor + fixed tables + counters): the
//! receipt table is reserved once at bind, because the serve step runs inside
//! the client allocator's grow path and must never allocate.

use alloc::{boxed::Box, vec};
use kore_abi::cap::Rights;
use kore_abi::mem::{AllocProps, DmaClass, MemAttrs, NumaNode};

use kore_abi::session::{SlotRef, SupplyReleaseReport};
use kore_abi::task::role;

use kore_abi::task::{Bootstrap, Profile, RoleSlot};
use kore_rt::cap::{RegionToken, SupplyToken, verbs};
use kore_rt::service::{
    BindError, Bundle, GrantArg, PeerClose, ReturnMode, Serve, StartFailure, SupplyReleaseGrant,
    SystemService,
};
use kore_rt::svc::{
    MAX_SERVE_PAIRS, PeerAdmissionRefusal, PeerBundleAssembler, PeerBundlePart, PeerId,
    PeerMapPlan, SvcProtocol,
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

/// Per-serving-pair **live outstanding-supply quota** ([`MmRequest::Supply`]):
/// the most bytes one client face may have in live capability lineages.
/// The kernel reports completion only after the lineage is dead and the
/// region bitmap has accepted the frames. Accounted
/// per arrival pair ([`Serve::via`] — structural identity, never a wire
/// field), so init's face and the shell's meter independently.
///
/// The bound makes simultaneous exhaustion a
/// *policy* refusal ([`MmRefusal::Quota`], loud on the wire) instead of allowing
/// one client to drain the region.
pub const SUPPLY_QUOTA_BYTES: u64 = 128 * 1024 * 1024;

/// The **huge-class** per-pair quota is not a constant: it is the
/// `role::SUPPLY_QUOTA` boot datum (`services/mm/boot.toml`, two `u32`
/// words), so the deployed working-set ceiling is manifest policy while the
/// base-class quota above stays the launcher's everyday allowance.  A
/// supply asked for at 2 MiB or 1 GiB alignment (`res/ACCEL_MEMORY_SCALE.md`
/// §1.2: the alignment *is* the frame class) is metered against this quota,
/// never against [`SUPPLY_QUOTA_BYTES`].  This constant is the ceiling the
/// receipt encoding is sized for; a datum above it is clamped to it.
pub const HUGE_SUPPLY_QUOTA_MAX: u64 = 64 << 30;

/// Huge-class reservations one pair may hold live — sizes the reservation
/// table beside the base class's page-granular worst case.  A huge supply
/// is at least 2 MiB, so 4096 of them is 8 GiB of 2 MiB units per pair
/// before any coalescing into 1 GiB units; well past what one client
/// keeps live.
const HUGE_SLOTS_PER_PAIR: usize = 4096;

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
        let lo = boot.datum(role::SUPPLY_QUOTA, 0).unwrap_or(0);
        let hi = boot.datum(role::SUPPLY_QUOTA, 1).unwrap_or(0);
        let huge_quota = (u64::from(hi) << 32 | u64::from(lo)).min(HUGE_SUPPLY_QUOTA_MAX);
        Ok(Self {
            supply,
            region,
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
    /// Metered per arrival pair ([`SUPPLY_QUOTA_BYTES`]).
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
    /// The arrival pair's outstanding-supply meter cannot fit the request
    /// ([`SUPPLY_QUOTA_BYTES`]) — a *policy* refusal, loud on the wire
    /// (the `TableFull` discipline), distinct from [`Exhausted`]
    /// (the backing itself cannot fit it).
    ///
    /// [`Exhausted`]: Self::Exhausted
    Quota,
    /// The administration step cannot complete yet (an admission in
    /// flight, or a retired face whose supplies have not all returned).
    Pending,
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

const MAX_SUPPLY_RESERVATIONS: usize =
    MAX_SERVE_PAIRS * (SUPPLY_QUOTA_BYTES as usize / PAGE_BYTES as usize + HUGE_SLOTS_PER_PAIR);

/// Completions consumed per pair per serve step. The drain runs inside the
/// client allocator's grow path, so a banked backlog (a launch storm, then
/// idle) must not stall one serve arbitrarily; the tail beyond the cap waits
/// for the next request — at worst one extra [`MmRefusal::Quota`] round.
const SUPPLY_RELEASE_DRAIN_PER_PAIR: usize = 16;

/// The meter row a supply **lane** is booked to. A lane is the arrival a
/// supply was minted on — a fixed pair's index, or a per-launch face's
/// [`PeerId::arrival`] — and it is also the kernel release lane its
/// completion report comes back on (`SupplyReleaseGrant::pair`). A face's
/// row is its slot: slots are below [`MAX_SERVE_PAIRS`] (mm never enables
/// peer growth), and a slot is not reused while its row is non-zero
/// ([`MemoryManager::peer_closed`]), so a report can never credit a later
/// face in the same slot.
fn meter_index(lane: u16) -> Option<usize> {
    let pair = match PeerId::from_arrival(lane) {
        Some(peer) => peer.slot(),
        None => usize::from(lane),
    };
    (pair < MAX_SERVE_PAIRS).then_some(pair)
}

/// Exec's administration pair: `exec_client` takes a row's final serving
/// pair (`services/exec/src/founder/mod.rs`), so it is the last of mm's
/// fixed pairs. Only it may admit, retire or confirm per-launch faces.
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

/// One per-launch face's slot state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PeerLane {
    /// The face's arrival — its quota lane and release lane.
    arrival: u16,
    /// Still attached. A closed face keeps its lane (and its slot, through
    /// [`MemoryManager::peer_closed`]) until its supplies have returned.
    live: bool,
}

// Receipt widths derive from admission and quota, so changing the host budget
// cannot alias pair identities or truncate reservation indices. The table holds
// one u32 per reservation (1.125 MiB at eight pairs: the base class's
// page-granular worst case plus `HUGE_SLOTS_PER_PAIR`); it fits the 2 MiB
// founder heap.  The page field is sized by the *huge* ceiling, and one class
// bit names which meter a completion credits.
const RECEIPT_INDEX_BITS: u32 = (MAX_SUPPLY_RESERVATIONS - 1).ilog2() + 1;
const RECEIPT_PAIR_BITS: u32 = (MAX_SERVE_PAIRS - 1).ilog2() + 1;
const RECEIPT_PAGES_BITS: u32 = (HUGE_SUPPLY_QUOTA_MAX / PAGE_BYTES).ilog2() + 1;
const RECEIPT_CLASS_BITS: u32 = 1;
const RECEIPT_PAIR_SHIFT: u32 = RECEIPT_INDEX_BITS;
const RECEIPT_PAGES_SHIFT: u32 = RECEIPT_PAIR_SHIFT + RECEIPT_PAIR_BITS;
const RECEIPT_CLASS_SHIFT: u32 = RECEIPT_PAGES_SHIFT + RECEIPT_PAGES_BITS;
const RECEIPT_GENERATION_SHIFT: u32 = RECEIPT_CLASS_SHIFT + RECEIPT_CLASS_BITS;
const RECEIPT_INDEX_MASK: u64 = (1 << RECEIPT_INDEX_BITS) - 1;
const RECEIPT_PAIR_MASK: u64 = (1 << RECEIPT_PAIR_BITS) - 1;
const RECEIPT_PAGES_MASK: u64 = (1 << RECEIPT_PAGES_BITS) - 1;
const RECEIPT_CLASS_HUGE: u64 = 1 << RECEIPT_CLASS_SHIFT;
const RECEIPT_GENERATION_MAX: u32 = (1 << (64 - RECEIPT_GENERATION_SHIFT)) - 1;
const RESERVATION_LIVE: u32 = 1 << 31;
const RESERVATION_GENERATION_MASK: u32 = RECEIPT_GENERATION_MAX;

// The packing only works while the configuration fits the fields, and the
// width checks inside `reserve` are `debug_assert!`s — compiled out of the
// release image, where a drifted constant would truncate silently (a grown
// `MAX_SERVE_PAIRS` folds pairs together, so their completions stop matching
// and quota sticks). Pin every field's capacity where the compiler owns it.
const _: () = {
    assert!(
        MAX_SERVE_PAIRS <= 1 << RECEIPT_PAIR_BITS,
        "a serving pair must be representable in the receipt's pair field"
    );
    assert!(
        SUPPLY_QUOTA_BYTES / PAGE_BYTES <= RECEIPT_PAGES_MASK
            && HUGE_SUPPLY_QUOTA_MAX / PAGE_BYTES <= RECEIPT_PAGES_MASK,
        "the largest quota-admissible supply of either class must fit the receipt's page count"
    );
    assert!(
        RECEIPT_GENERATION_SHIFT < 64 && 64 - RECEIPT_GENERATION_SHIFT >= 12,
        "a receipt keeps at least twelve generation bits"
    );
    assert!(
        MAX_SUPPLY_RESERVATIONS <= 1 << RECEIPT_INDEX_BITS,
        "every reservation slot must be addressable by the receipt's index"
    );
    assert!(
        RECEIPT_GENERATION_MAX < RESERVATION_LIVE,
        "the live bit must sit above the generation bits it shares a slot with"
    );
};

/// The per-pair **supply meter** — [`MmRequest::Supply`]'s whole
/// bookkeeping, kept apart from the carve books because it is a *policy*
/// ledger over kernel-owned frames (the region bitmap is the ledger of
/// record), not a window pool.  Pure counters, so the quota arithmetic is
/// host-testable without a bound service.
#[derive(Debug)]
struct SupplyMeter {
    /// Supplies minted.
    supplies: u32,
    /// Bytes those supplies span (the totals row).
    supply_bytes: u64,
    /// Outstanding base-class bytes per arrival pair ([`Serve::via`] —
    /// structural identity, never a wire field), each bounded by
    /// [`SUPPLY_QUOTA_BYTES`].
    pair_supply: [u64; MAX_SERVE_PAIRS],
    /// Outstanding huge-class (2 MiB / 1 GiB aligned) bytes per pair, each
    /// bounded by [`huge_quota`](Self::huge_quota).
    pair_huge: [u64; MAX_SERVE_PAIRS],
    /// The huge-class per-pair quota ([`MmEndowment::huge_quota`]).
    huge_quota: u64,
    reservations: Box<[u32]>,
    next: usize,
}

impl SupplyMeter {
    fn new(huge_quota: u64) -> Self {
        Self::with_capacity(MAX_SUPPLY_RESERVATIONS, huge_quota)
    }

    fn with_capacity(capacity: usize, huge_quota: u64) -> Self {
        Self {
            supplies: 0,
            supply_bytes: 0,
            pair_supply: [0; MAX_SERVE_PAIRS],
            pair_huge: [0; MAX_SERVE_PAIRS],
            huge_quota: huge_quota.min(HUGE_SUPPLY_QUOTA_MAX),
            reservations: vec![0; capacity].into_boxed_slice(),
            next: 0,
        }
    }

    /// Validate one supply request and reserve it against `pair`'s meter:
    /// `(rounded len, align_log2)` on success.  Shape first
    /// ([`MmRefusal::BadProps`] — zero size, an alignment outside the
    /// `LeaseReq` vocabulary, an out-of-range pair), then policy
    /// ([`MmRefusal::Quota`]).  State advances only on success — a
    /// refused request perturbs nothing.
    fn reserve(&mut self, pair: usize, size: u64, align: u64) -> Result<(u64, u8, u64), MmRefusal> {
        let align_log2 = match align {
            0 => 0,
            0x1000 => kore_abi::session::LEASE_ALIGN_4K,
            0x20_0000 => kore_abi::session::LEASE_ALIGN_2M,
            0x4000_0000 => kore_abi::session::LEASE_ALIGN_1G,
            _ => return Err(MmRefusal::BadProps),
        };
        if size == 0 || pair >= MAX_SERVE_PAIRS {
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
        let (outstanding, quota) = if huge {
            (self.pair_huge[pair], self.huge_quota)
        } else {
            (self.pair_supply[pair], SUPPLY_QUOTA_BYTES)
        };
        if outstanding.checked_add(len).is_none_or(|t| t > quota) {
            return Err(MmRefusal::Quota);
        }
        let capacity = self.reservations.len();
        let mut found = None;
        for step in 0..capacity {
            let index = (self.next + step) % capacity;
            let state = self.reservations[index];
            if state & RESERVATION_LIVE == 0
                && state & RESERVATION_GENERATION_MASK != RECEIPT_GENERATION_MAX
            {
                found = Some(index);
                break;
            }
        }
        let index = found.ok_or(MmRefusal::Quota)?;
        let generation = (self.reservations[index] & RESERVATION_GENERATION_MASK) + 1;
        self.reservations[index] = RESERVATION_LIVE | generation;
        let pages = len / PAGE_BYTES;
        debug_assert!(index <= RECEIPT_INDEX_MASK as usize);
        debug_assert!(pair as u64 <= RECEIPT_PAIR_MASK);
        // Not a `debug_assert`: both release profiles compile those out, and
        // the page field's width is derived from the compile-time
        // `SUPPLY_QUOTA_BYTES` while the deployed ceiling arrives as the
        // `role::SUPPLY_QUOTA` boot datum (`services/mm/boot.toml` currently
        // carries 2 GiB, which needs 20 bits). A supply the field cannot name
        // must be refused, never encoded as a smaller one — a truncated page
        // count would retire the wrong span.
        if pages > RECEIPT_PAGES_MASK {
            self.reservations[index] = generation;
            return Err(MmRefusal::Quota);
        }
        let receipt = (u64::from(generation) << RECEIPT_GENERATION_SHIFT)
            | if huge { RECEIPT_CLASS_HUGE } else { 0 }
            | (pages << RECEIPT_PAGES_SHIFT)
            | ((pair as u64) << RECEIPT_PAIR_SHIFT)
            | index as u64;
        self.next = (index + 1) % capacity;

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
        meter_index(report.pair).is_some_and(|pair| self.retire(report.receipt, pair, report.len))
    }

    /// Whether any supply metered to `pair` is still outstanding.
    fn outstanding(&self, pair: usize) -> bool {
        self.pair_supply[pair] != 0 || self.pair_huge[pair] != 0
    }

    fn retire(&mut self, receipt: u64, pair: usize, len: u64) -> bool {
        let index = (receipt & RECEIPT_INDEX_MASK) as usize;
        let encoded_pair = ((receipt >> RECEIPT_PAIR_SHIFT) & RECEIPT_PAIR_MASK) as usize;
        let encoded_pages = (receipt >> RECEIPT_PAGES_SHIFT) & RECEIPT_PAGES_MASK;
        let generation = (receipt >> RECEIPT_GENERATION_SHIFT) as u32;
        let Some(state) = self.reservations.get_mut(index) else {
            return false;
        };
        if generation == 0
            || *state != (RESERVATION_LIVE | generation)
            || encoded_pair != pair
            || encoded_pages.checked_mul(PAGE_BYTES) != Some(len)
        {
            return false;
        }
        *state = generation;
        if receipt & RECEIPT_CLASS_HUGE != 0 {
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
    /// Per-launch faces by slot.
    peers: [Option<PeerLane>; MAX_SERVE_PAIRS],
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
            meter: SupplyMeter::new(e.huge_quota),
            pending_grant: None,
            peer_assembler: PeerBundleAssembler::new(<Self as SvcProtocol>::REPLY_GRANTS, true),
            peer_map: None,
            peers: [None; MAX_SERVE_PAIRS],
            last_admission: None,
        }
    }

    /// Per-launch faces arrive with their components on exec's return leg
    /// (rings, kick, grant channel, return receive), which a fixed landing
    /// cannot carry; `Release`'s extent returns ride the same retained leg.
    const RETURN_MODE: ReturnMode = ReturnMode::RetainedAny;

    fn start_on_metal(&mut self, rt: &kore_rt::entry::Rt) -> Result<(), StartFailure> {
        let window = rt
            .geom()
            .and_then(|geometry| geometry.window_of(kore_abi::task::geom::TRANSIENT))
            .filter(|window| {
                window.base() != 0
                    && window.base() & (PAGE_BYTES - 1) == 0
                    && window.len() >= PEER_RING_BYTES * MAX_SERVE_PAIRS as u64
            })
            .ok_or(StartFailure::new(1))?;
        self.peer_map = Some(PeerMapPlan {
            base: window.base(),
            stride: PEER_RING_BYTES,
        });
        Ok(())
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
        let faces = self.peers.map(|lane| lane.map(|lane| lane.arrival));
        let fixed = (0..<Self as SvcProtocol>::SERVE_PAIRS as u16).map(Some);
        for lane in fixed.chain(faces).flatten() {
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
                    None => true,
                };
                let Some(pair) = meter_index(via).filter(|_| admitted) else {
                    sv.reply(MmResponse::Refused(MmRefusal::Denied));
                    return;
                };
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
                let stats = self.stats();
                sv.reply(MmResponse::Stats(kore_abi::monitor::MemorySnapshot::new(
                    stats.extents,
                    stats.supplies,
                    stats.granted_bytes,
                    stats.released_bytes,
                    stats.reused_bytes,
                    stats.leaked_bytes,
                    stats.remaining,
                    stats.supply_bytes,
                )));
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
                    Some(Err(_)) => MmResponse::Refused(MmRefusal::Denied),
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
            // A face never lands on a fixed pair's index, nor on a slot whose
            // previous face still owes returns.
            if peer.slot() < <Self as SvcProtocol>::SERVE_PAIRS || self.peers[peer.slot()].is_some()
            {
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

    /// The meter's whole vocabulary: rounding, the closed alignment set,
    /// the zero refusal, and the out-of-range pair.
    #[test]
    fn the_supply_meter_speaks_the_lease_req_vocabulary() {
        let mut m = SupplyMeter::new(HUGE_SUPPLY_QUOTA_MAX);
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
        assert_eq!(
            m.reserve(MAX_SERVE_PAIRS, 0x1000, 0),
            Err(MmRefusal::BadProps)
        );
        assert_eq!((m.supplies, m.supply_bytes), before);
    }

    /// Each pair meters independently, the bound is exact, and a
    /// rollback restores the meter to the byte.
    #[test]
    fn the_quota_meters_per_pair_and_rolls_back_exactly() {
        let mut m = SupplyMeter::new(HUGE_SUPPLY_QUOTA_MAX);
        // Fill pair 1 (the shell's face) to the exact quota.
        let (_, _, pair_one) = m.reserve(1, SUPPLY_QUOTA_BYTES, 0).unwrap();
        // One more page on the same pair is a policy refusal…
        assert_eq!(m.reserve(1, PAGE_BYTES, 0), Err(MmRefusal::Quota));
        // …while pair 2 (init's face) still has its whole allowance.
        let (_, _, _) = m.reserve(2, PAGE_BYTES, 0).unwrap();
        assert_eq!(m.supplies, 2);
        assert_eq!(m.supply_bytes, SUPPLY_QUOTA_BYTES + PAGE_BYTES);
        // A dropped grant unwinds pair 1 whole; the allowance returns.
        m.rollback(pair_one, 1, SUPPLY_QUOTA_BYTES);
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
        let mut m = SupplyMeter::new(4 << 30);
        // Base class is untouched by a 3 GiB huge supply...
        let (len, _, huge) = m.reserve(1, 3 << 30, 0x4000_0000).unwrap();
        assert_eq!(len, 3 << 30);
        assert_eq!(m.pair_supply[1], 0);
        assert_eq!(m.pair_huge[1], 3 << 30);
        // ...and the base quota is still whole.
        let (_, _, _) = m.reserve(1, SUPPLY_QUOTA_BYTES, 0).unwrap();
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
        assert_eq!(m.pair_supply[1], SUPPLY_QUOTA_BYTES);

        // No datum ⇒ no huge quota, loudly.
        let mut none = SupplyMeter::new(0);
        assert_eq!(none.reserve(0, 0x20_0000, 0x20_0000), Err(MmRefusal::Quota));
        let (_, _, _) = none.reserve(0, 0x1000, 0).unwrap();
    }

    #[test]
    fn completed_lineages_reuse_quota_exactly_once() {
        let mut m = SupplyMeter::with_capacity(2, HUGE_SUPPLY_QUOTA_MAX);
        let (_, _, first) = m.reserve(1, SUPPLY_QUOTA_BYTES, 0).unwrap();
        assert_eq!(m.reserve(1, PAGE_BYTES, 0), Err(MmRefusal::Quota));

        let report = SupplyReleaseReport {
            receipt: first,
            pair: 1,
            len: SUPPLY_QUOTA_BYTES,
        };
        assert!(m.release(report));
        assert!(!m.release(report), "duplicate completion must be inert");
        assert_eq!(m.pair_supply[1], 0);

        // Cumulative mints may now exceed the policy bound while live bytes do
        // not; simultaneous live bytes remain bounded.
        let (_, _, _) = m.reserve(1, SUPPLY_QUOTA_BYTES, 0).unwrap();
        assert_eq!(m.supply_bytes, 2 * SUPPLY_QUOTA_BYTES);
        assert_eq!(m.reserve(1, PAGE_BYTES, 0), Err(MmRefusal::Quota));
    }

    #[test]
    fn highest_receipt_index_and_pair_round_trip_without_aliasing() {
        let mut meter = SupplyMeter::new(HUGE_SUPPLY_QUOTA_MAX);
        meter.next = MAX_SUPPLY_RESERVATIONS - 1;
        let pair = MAX_SERVE_PAIRS - 1;
        let (_, _, receipt) = meter.reserve(pair, SUPPLY_QUOTA_BYTES, 0).unwrap();
        assert_eq!(
            receipt & RECEIPT_INDEX_MASK,
            (MAX_SUPPLY_RESERVATIONS - 1) as u64
        );
        assert!(!meter.release(SupplyReleaseReport {
            receipt,
            pair: 0,
            len: SUPPLY_QUOTA_BYTES,
        }));
        assert!(meter.release(SupplyReleaseReport {
            receipt,
            pair: pair as u16,
            len: SUPPLY_QUOTA_BYTES,
        }));
        assert_eq!(meter.pair_supply[pair], 0);
        assert!(!meter.release(SupplyReleaseReport {
            receipt,
            pair: pair as u16,
            len: SUPPLY_QUOTA_BYTES,
        }));
    }

    #[test]
    fn stale_mismatched_and_reordered_completions_are_inert() {
        let mut m = SupplyMeter::with_capacity(2, HUGE_SUPPLY_QUOTA_MAX);
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
        let mut m = SupplyMeter::with_capacity(1, HUGE_SUPPLY_QUOTA_MAX);
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

    /// A lane is booked to its fixed pair or its face's slot; nothing else.
    #[test]
    fn a_lane_books_to_its_pair_or_its_face_slot() {
        assert_eq!(meter_index(0), Some(0));
        assert_eq!(meter_index(3), Some(3));
        let face = PeerId::from_arrival(0x0105).unwrap();
        assert_eq!(meter_index(face.arrival()), Some(5));
        assert_eq!(
            meter_index(MAX_SERVE_PAIRS as u16),
            None,
            "no such fixed pair"
        );
    }

    /// A face's returns credit its slot's row, keyed by its arrival.
    #[test]
    fn a_face_supply_is_metered_and_released_on_its_slot() {
        let face = PeerId::from_arrival(0x0104).unwrap();
        let mut m = SupplyMeter::with_capacity(8, 0);
        let pair = meter_index(face.arrival()).unwrap();
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
}
