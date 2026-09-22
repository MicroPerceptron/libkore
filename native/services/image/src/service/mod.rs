//! # `service` — the image service, the fourth [`SystemService`] pillar
//!
//! **The lessor of code** (`res/IMAGE.md` §1): the service endowed with
//! the image-byte store and the legacy X-mint, validating ELF at `Register`,
//! carving per-segment extents **in place** (§2's zero-copy carve),
//! finalizing text leases, holding the segment leases, and granting attenuated lease
//! caps at `Acquire`.  Its role bundle is two slots —
//! [`role::IMAGE_STORE`] (the byte source) and [`role::X_MINT`] (the
//! pre-blessed-allocation authority retained by the existing founding
//! profile) — claimed at bind, fail-closed per slot. The consuming
//! writable-to-executable transition is authorized by the full-rights lease
//! itself; it does not consult the X-mint (§6). Privileged control uses names for registration and
//! discovery. Ordinary loader faces are permanently bound to one immutable
//! generation: installation freezes its transitive dependency closure,
//! `AcquireEntry` takes no name, and dependency names are accepted only inside
//! that closure. The service never sees a consumer's address space.
//!
//! **`Acquire` grants lease caps, not views** (§3): each grant carves a fresh
//! whole-segment child from one held persistent segment lease and delivers it
//! with [`LEASE_RIGHTS`](ImageService::LEASE_RIGHTS)
//! (`ACCESS | DERIVE | REVOKE`, `GrantArg::Sub`) — ≤ [`MAX_IMAGE_SEGS`] per
//! image `=` exactly [`MAX_GRANTS`], **by construction rather than
//! coincidence**.  The persistent parent remains the image's revocation pivot,
//! so repeated acquisitions are lawful and one consumer can revoke only its
//! own child subtree.  The consumer derives its own shares at VAs it chooses
//! (per-task ASLR for free); no `MODIFY` (cannot unshare others). W^X is the
//! *record's* property:
//! text segments are finalized through [`verbs::carve_owned_exec_window_outcome`],
//! which consumes the writable carve and leaves one executable successor;
//! a `MODIFY` share off that lease refuses
//! unconditionally — there is no mprotect-to-writable path to close.
//!
//! At startup the service registers every member of its endowed archive. An
//! archive member is either an ELF image or an opaque DATA member
//! ([`ARCHIVE_MEMBER_DATA`](kore_abi::image::ARCHIVE_MEMBER_DATA)): a data
//! member is bounds-checked only and registers a record holding nothing but
//! its member window, so `Export` lends its bytes exactly as it lends an
//! image's while `Acquire`/`Describe` refuse it
//! [`NotImage`](ImageRefusal::NotImage). The request-side `Register { name, slot }` server arm accepts an image lease
//! already delivered into the service arena (#148's no-capability-in-ring
//! rule); both paths share the same checker, carve, and bounded ownership
//! machinery.
//!
//! **Shadowing, not replacement** (§3): a re-`Register` of a live name
//! takes lookup precedence while the old entry — and every consumer's
//! outstanding lease — lives on; `Retire` targets the **newest** version
//! and retires its stashed leases (tag-retire; the T6 cascade over
//! already-executed views is kernel-side). Record storage grows fallibly from
//! the service heap and reuses retired holes; real heap or capability-arena
//! exhaustion refuses ([`TableFull`](ImageRefusal::TableFull)), never silently
//! drops.
//!
//! Per-image bookkeeping stays wire-bounded (fixed segment and dependency
//! arrays), while the total catalogue is allocator-backed and sized by the
//! measured archive plus runtime registrations. A refusal emits **no** grant —
//! no authority leaves on an error path.

#[cfg(feature = "heap")]
use kore_abi::cap::ResourceKind;
use kore_abi::cap::Rights;
use kore_abi::image::{ImageName, ImageRefusal, ImageSourceMetadata, ModuleDescriptor};
#[cfg(feature = "heap")]
use kore_abi::image::{MAX_DEPENDENCIES, kind, seg};
use kore_abi::session::SlotRef;
use kore_abi::task::{FACE_RING_STRIDE, FACE_SUPPLY_BYTES};
#[cfg(feature = "heap")]
use kore_abi::task::{geom, role};

#[cfg(feature = "heap")]
use kore_ld::check::CheckedImage;

#[cfg(feature = "heap")]
use kore_rt::cap::{
    CHANNEL_ENDPOINT_RIGHTS, ChanRecvToken, ChanSendToken, ChannelRoot, ChannelSpaceToken,
    ExecutableCode, ImageStoreToken, MappedLeaseView, ReturnedCapability, StashedLease,
    SupplyToken, UncertainChannel, UncertainExecCarve, UncertainLocalCapability, XMintToken, verbs,
};
use kore_rt::service::MAX_GRANTS;
#[cfg(feature = "heap")]
use kore_rt::service::{GrantArg, MintedPairSpec, ReturnMode, Serve, StartFailure, SystemService};
use kore_rt::svc::MAX_MINTED_PAIRS;

#[cfg(feature = "heap")]
use kore_svc_mm::PAGE_BYTES;

#[cfg(feature = "heap")]
mod validation;
#[cfg(feature = "heap")]
use validation::{ValidationCleanup, ValidationMapping};

/// Segment-lease ceiling per image — the frozen wire's
/// [`MAX_SEGMENTS`](kore_abi::image::MAX_SEGMENTS) (bounded LOAD entries
/// plus TLS init; kinds may repeat). One `Acquire` delivers one image in one
/// bounded step; the generic ceiling has one additional lane so `FaceMint`
/// can return the ring sub-lease and the complete recv bank together.
pub const MAX_IMAGE_SEGS: usize = kore_abi::image::MAX_SEGMENTS;

const _: () = assert!(MAX_IMAGE_SEGS + 1 == MAX_GRANTS);

// The face-supply geometry contract (slice C): the founder mints exactly
// [`FACE_SUPPLY_BYTES`], the face table carves exactly `MAX_MINTED_PAIRS`
// strides of it, and one pair's rings must fit a stride.  Checked here —
// the one place the wire forms, the stride, and the bank size are all
// visible together — so a drift refuses at compile instead of formatting
// a phantom ring.
const _: () = assert!(MAX_MINTED_PAIRS as u64 * FACE_RING_STRIDE == FACE_SUPPLY_BYTES);
const _: () = assert!(
    kore_rt::svc::window_bytes::<ImageService>(kore_rt::svc::REQ_CAP, kore_rt::svc::REP_CAP)
        <= FACE_RING_STRIDE as usize
);

/// One live loader face retains one channel root and one send endpoint per
/// grant lane. The reply's recv descendants transfer directly into the
/// client and therefore do not remain in this arena.
#[cfg(all(feature = "heap", any(target_os = "none", test)))]
const FACE_OWNED_SLOTS: usize = 2 * MAX_IMAGE_SEGS;
/// `mint_channel` publishes root + send + recv together, then retires the
/// fresh recv. Face mints are serialized, so one slot beyond the persistent
/// bank covers that transient peak for every remaining face.
#[cfg(all(feature = "heap", any(target_os = "none", test)))]
const FACE_MINT_SCRATCH_SLOTS: usize = 1;

/// Peak owned-slot demand that must remain satisfiable after admitting an
/// image. Overflow is a refusal, never permission to under-reserve a face.
#[cfg(all(feature = "heap", any(target_os = "none", test)))]
fn owned_slot_requirement(image_demand: usize, vacant_faces: usize) -> Option<usize> {
    vacant_faces
        .checked_mul(FACE_OWNED_SLOTS)
        .and_then(|face_slots| {
            face_slots.checked_add(usize::from(vacant_faces != 0) * FACE_MINT_SCRATCH_SLOTS)
        })
        .and_then(|face_slots| face_slots.checked_add(image_demand))
}

/// The image service's role bundle: the byte store it registers from,
/// the legacy X-mint retained by its founding profile, and the two slice-C face authorities
/// — **possessed**, not described (the store's
/// [`len`](ImageStoreToken::len) is the geometry; the mints are
/// authorities, not quantities).
#[cfg(feature = "heap")]
#[derive(Debug, kore_rt::Bundle)]
#[bundle(name = "image")]
pub struct ImageEndowment {
    /// The claimed image-byte store ([`role::IMAGE_STORE`]).
    #[bundle(role = role::IMAGE_STORE)]
    pub store: ImageStoreToken,
    /// The claimed X-mint ([`role::X_MINT`]) retained for pre-blessed
    /// allocation and the existing profile selector. Lease finalization does
    /// not require it (`res/IMAGE.md` §6).
    #[bundle(role = role::X_MINT)]
    pub xmint: XMintToken,
    /// The claimed face supply ([`role::FACE_SUPPLY`], slice C) — the
    /// lease whose fixed strides back the minted loader-face rings.
    /// Image-owned: pair roots stay with their founding service, so a
    /// child's reap can never unmap the lessor's own serving view.
    #[bundle(role = role::FACE_SUPPLY)]
    pub faces: SupplyToken,
    /// The claimed rendezvous mint grant ([`role::CHANNEL_SPACE`], slice
    /// C) — the authority every minted face's grant channels are rooted
    /// beneath; revoking it cuts the whole face-channel forest (T6).
    #[bundle(role = role::CHANNEL_SPACE)]
    pub channels: ChannelSpaceToken,
}

/// The image vocabulary (`res/IMAGE.md` §3) — registration, acquisition,
/// revocation.  One enum, one intake; the ring muxes.
#[derive(Clone, Copy, Debug)]
pub enum ImageRequest {
    /// Register bytes transferred through the return channel. `slot` names
    /// that pair's founder-published return landing; the runtime joins it to
    /// the persistent arena slot that actually receives the lease (#148: no
    /// capability in the ring).
    /// Validate (kore-ld), carve in place, and finalize text leases.
    Register {
        /// The name later `Acquire`s present.
        name: ImageName,
        /// Founder-published return landing that identifies the transfer leg.
        slot: SlotRef,
        /// Exact immutable-source identity, byte commitment, and explicit
        /// launch profile. It is inert until authenticated against `source`.
        source: ImageSourceMetadata,
    },
    /// name → ≤ [`MAX_IMAGE_SEGS`] grants of
    /// [`LEASE_RIGHTS`](ImageService::LEASE_RIGHTS) lease caps + the
    /// descriptor.
    Acquire {
        /// The name to look up (newest version wins — shadowing).
        name: ImageName,
    },
    /// Acquire the immutable entry generation structurally bound to this
    /// loader face. No caller-supplied name or row selects it.
    AcquireEntry,
    /// Return the newest grantable version's validated descriptor without
    /// transferring its segment leases. This is discovery, not authority:
    /// a later loader still performs its own [`Acquire`](Self::Acquire).
    Describe {
        /// The name to look up (newest version wins — shadowing).
        name: ImageName,
    },
    /// Revoke the newest version's leases; T6 cascades through every
    /// consumer once the service is wired on metal.
    Retire {
        /// The name to retire.
        name: ImageName,
    },
    /// Validate `target` and mint one **per-child loader face** (slice C):
    /// claim a face slot, mint its grant channels, format its ring stride,
    /// attach a fresh serving pair. The reply carries the validated target
    /// descriptor, face id, and complete affine client face.
    /// Exec's verb ([`FACE_OWNER_VIA`](ImageService) gates it): neither the
    /// shell nor a child can mint spawning capacity directly.
    FaceMint {
        /// Dynamic executable the new face will initially load.
        target: ImageName,
    },
    /// Mint a fresh loader face bound to the same immutable generation as a
    /// privileged control-owned template.
    FaceFork { template: u32 },
    /// Collect a minted face's grant-channel recv ends: [`MAX_GRANTS`]
    /// fresh recvs (one per channel), staged in the reply's grant
    /// session so a dropped delivery unwinds them whole (I3).  Once per
    /// mint — a re-bank refuses.
    FaceBank {
        /// The face id a prior [`FaceMint`](Self::FaceMint) replied.
        face: u32,
    },
    /// Retire a minted face: detach its serving pair, revoke its channel
    /// roots (T6 closes every granted recv and delivered send wherever
    /// it traveled), free the slot for a later mint.  Parent-driven; a
    /// leak is bounded by the parent's own face quota, visible in
    /// `summarize`.
    FaceRetire {
        /// The face id to retire.
        face: u32,
    },
    /// Retire the exact immutable generation bound to a privileged template
    /// face. This is cleanup authority on the owner pair, never a selector on
    /// an ordinary loader face.
    RetireBound { template: u32 },
    /// Grant one read-only lease over the newest generation's whole member
    /// window. This conveys the registered *bytes* so a holder can copy them
    /// elsewhere — the filesystem seed — and deliberately not the executable
    /// blessing, so an export can never become a way to run something.
    Export {
        /// The name to export (newest version wins — shadowing).
        name: ImageName,
    },
    /// A frozen tombstone discriminant arrived (the retired name/ordinal
    /// acquisition family). Decoded explicitly so the caller receives a
    /// typed refusal instead of a dropped request; it carries no payload
    /// and can have no effect.
    Retired,
}

/// The image replies — **data only** (`res/IMAGE.md` §3: the descriptor
/// is what the service *says*; authority is the granted lease caps,
/// arriving as delivered tokens, never reply payload).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageResponse {
    /// One read-only member-window lease rides beside this reply; `len` is
    /// the exact registered byte length inside the page-rounded span.
    Exported { len: u64 },
    /// Validated, carved, leased — the image is acquirable.
    Registered {
        /// The image-relative promise the stub links against.
        module: ModuleDescriptor,
    },
    /// The lease-cap grants ride the same outcome; this is the data half.
    Acquired {
        /// The registered descriptor (newest version).
        module: ModuleDescriptor,
    },
    /// Metadata-only discovery result. No segment capability accompanies
    /// this reply; image bytes remain unusable until a later `Acquire`.
    Described {
        /// Stable identity of the newest immutable generation bearing this
        /// display name. Metadata only; no request accepts it as authority.
        generation: u64,
        /// The registered descriptor (newest grantable version).
        module: ModuleDescriptor,
    },
    /// The newest version's leases are retired — and a retired **face**
    /// rides the same variant (one outcome word for "the named thing is
    /// gone").
    Retired,
    /// A loader face was minted (slice C): `face` is its id, `module` is the
    /// target's validated descriptor, and the complete client authority rides
    /// the same outcome (ring sub first, then [`MAX_IMAGE_SEGS`] recvs).
    Face {
        /// The minted face's id — what `FaceRetire` cites.
        face: u32,
        /// Stable identity of the immutable generation selected by this
        /// face. It is never accepted as acquisition authority.
        generation: u64,
        /// Validated metadata for the executable this face will first load.
        module: ModuleDescriptor,
    },
    /// The face's grant-channel recv ends ride the reply posts, one per
    /// channel ([`MAX_GRANTS`] exactly); this is the data half.
    Banked,
    /// Refused — and a refusal emits **no** grant (fail-closed: no
    /// authority leaves on an error path).
    Refused(ImageRefusal),
}

/// One persistent segment authority. Executable text owns only the fresh
/// post-finalization lease; no writable parent or alias survives.
#[cfg(feature = "heap")]
#[derive(Debug)]
enum ImageLease {
    Plain(StashedLease, u64),
    Executable(ExecutableCode, u64),
    UncertainPlain(UncertainLocalCapability),
    UncertainExecutable(UncertainExecCarve),
}

#[cfg(feature = "heap")]
impl ImageLease {
    fn slot(&self) -> Option<SlotRef> {
        match self {
            Self::Plain(lease, _) => lease.slot(),
            Self::Executable(lease, _) => lease.slot(),
            Self::UncertainPlain(_) | Self::UncertainExecutable(_) => None,
        }
    }

    fn len(&self) -> u64 {
        match self {
            Self::Plain(_, len) | Self::Executable(_, len) => *len,
            Self::UncertainPlain(_) | Self::UncertainExecutable(_) => 0,
        }
    }

    fn try_retire(&mut self) -> bool {
        match self {
            Self::Plain(lease, _) => lease.try_retire(),
            Self::Executable(lease, _) => lease.try_retire(),
            Self::UncertainPlain(lease) => lease.try_retire(),
            Self::UncertainExecutable(lease) => lease.try_retire(),
        }
    }

    fn is_live(&self) -> bool {
        match self {
            Self::Plain(lease, _) => lease.is_live(),
            Self::Executable(lease, _) => lease.is_live(),
            Self::UncertainPlain(_) | Self::UncertainExecutable(_) => false,
        }
    }
}

/// One minted loader face's retained authority (slice C): the grant
/// channels' roots (the T6 revocation pivots) and send lanes (the serve
/// host's `grant_tx`), plus the once-per-mint bank latch.  The ring
/// backing needs no per-slot state — slot `k`'s window is the fixed
/// arithmetic stride `[k · FACE_RING_STRIDE, +STRIDE)` of the one
/// whole-supply serving view.
#[cfg(feature = "heap")]
#[derive(Debug)]
struct FaceSlot {
    ring_root: Option<LocalLease>,
    /// Immutable image generation selected by this serving endpoint.
    generation: ImageGeneration,
    /// Exact transitive dependency generations selected at privileged face
    /// installation. A later registration with the same display name cannot
    /// retarget this closure.
    closure: [Option<DependencyBinding>; MAX_DEPENDENCIES],
    /// Per-channel roots — retained for the face's life; `FaceRetire`
    /// revokes them and T6 closes every leaf wherever it traveled.
    roots: [Option<ChannelRoot>; MAX_IMAGE_SEGS],
    /// Per-channel send ends — the attached pair's grant lanes.
    sends: [Option<ChanSendToken>; MAX_IMAGE_SEGS],
    /// A fresh receive endpoint that refused retirement before any grant.
    pending_recv: Option<ChanRecvToken>,
    /// All three original results of a mint with uncertain completion.
    uncertain_channel: Option<UncertainChannel>,
    /// The recv ends were collected (`FaceBank` is once per mint).
    banked: bool,
    /// A retire left refused owners behind — the slot is a tombstone:
    /// never mintable, retryable by a later `FaceRetire`.
    retiring: bool,
    /// True only when no usable client face was published. Explicit retire
    /// retries and attach failures after delivery retain their original id.
    cleanup_pending: bool,
}

#[cfg(feature = "heap")]
#[derive(Debug)]
enum LocalLease {
    Known(StashedLease),
    Uncertain(UncertainLocalCapability),
}

#[cfg(feature = "heap")]
impl LocalLease {
    fn slot(&self) -> Option<SlotRef> {
        match self {
            Self::Known(owner) => owner.slot(),
            Self::Uncertain(_) => None,
        }
    }

    fn try_retire(&mut self) -> bool {
        match self {
            Self::Known(owner) => owner.try_retire(),
            Self::Uncertain(owner) => owner.try_retire(),
        }
    }
}

#[cfg(feature = "heap")]
impl FaceSlot {
    fn preparing(
        generation: ImageGeneration,
        closure: [Option<DependencyBinding>; MAX_DEPENDENCIES],
    ) -> Self {
        Self {
            ring_root: None,
            generation,
            closure,
            roots: [const { None }; MAX_IMAGE_SEGS],
            sends: [const { None }; MAX_IMAGE_SEGS],
            pending_recv: None,
            uncertain_channel: None,
            banked: false,
            retiring: true,
            cleanup_pending: true,
        }
    }

    fn try_cleanup(&mut self) -> bool {
        let mut complete = true;
        if let Some(recv) = self.pending_recv.take() {
            if let Err(recv) = recv.try_retire() {
                self.pending_recv = Some(recv);
                complete = false;
            }
        }
        // The retained recv's root is in this bank. Do not revoke that parent
        // before settling the leaf's own reservation.
        if self.pending_recv.is_none() {
            complete &= ImageService::retire_face_channels(&mut self.roots, &mut self.sends);
        }
        if let Some(owner) = self.uncertain_channel.as_mut() {
            if owner.try_retire() {
                self.uncertain_channel = None;
            } else {
                complete = false;
            }
        }
        if let Some(owner) = self.ring_root.as_mut() {
            if owner.try_retire() {
                self.ring_root = None;
            } else {
                complete = false;
            }
        }
        complete
    }

    fn unwind(&mut self, unpublished: bool) -> bool {
        self.retiring = true;
        self.cleanup_pending = unpublished;
        self.try_cleanup() && unpublished
    }
}

#[cfg(feature = "heap")]
type FaceArena = &'static (dyn core::alloc::Allocator + Sync);

/// Cached heap block: retained across face generations, with a new revocable
/// child root per client. Old clients cannot reach a recycled ring.
#[cfg(feature = "heap")]
struct FaceBacking {
    root: LocalLease,
    bytes: core::ptr::NonNull<[u8]>,
}

/// Stable identity of one occupied image-table generation. The table slot is
/// recyclable; the registration sequence prevents old faces from retargeting.
#[cfg(feature = "heap")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ImageGeneration {
    /// Allocator-backed catalogue index. This is service-private identity,
    /// not a wire field, so it must not inherit an unrelated 8-bit ceiling.
    slot: usize,
    seq: u64,
}

#[cfg(feature = "heap")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DependencyBinding {
    name: ImageName,
    generation: ImageGeneration,
}

#[cfg(feature = "heap")]
const fn generation_matches(generation: ImageGeneration, slot: usize, seq: u64) -> bool {
    generation.slot == slot && generation.seq == seq
}

/// The face-id ⇄ slot correlation, fail-closed: ids are slot indices and
/// nothing else, so an out-of-range id refuses before any table access.
fn face_slot_of(face: u32) -> Option<usize> {
    let k = usize::try_from(face).ok()?;
    (k < u16::MAX as usize - kore_rt::svc::MAX_SERVE_PAIRS).then_some(k)
}

/// A face slot's serving-pair identity — the fixed pairs own
/// `0..MAX_SERVE_PAIRS`, minted faces the next `MAX_MINTED_PAIRS`
/// (collision-free by construction; the host still refuses one).
fn face_via(slot: usize) -> u16 {
    (kore_rt::svc::MAX_SERVE_PAIRS + slot) as u16
}

/// Who owns completion of a non-grantable catalogue entry.
#[cfg(feature = "heap")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ImageLifecycle {
    Live,
    /// No successful registration reply exists. The service owns rollback.
    RegistrationRollback,
    /// An explicit retirement may be retried by name. Keep this generation
    /// named until that request succeeds; background removal could otherwise
    /// make the retry retire an older, still-live version of the same name.
    Retiring,
}

/// One registered image version: its name, frozen descriptor, and owned
/// segment leases whose slots an acquisition's grant subjects name.
#[cfg(feature = "heap")]
#[derive(Debug)]
struct ImageEntry {
    name: ImageName,
    /// Registration stamp — shadowing order (newest = greatest); table
    /// position cannot carry it because retired slots are reused.
    seq: u64,
    descriptor: ModuleDescriptor,
    /// Direct dependency names extracted by the validation gate.
    needed: [Option<ImageName>; MAX_DEPENDENCIES],
    /// Stashed `(lease slot, seg kind)` pairs, in descriptor-segment
    /// order — the `Acquire` grant subjects.
    leases: [Option<(ImageLease, u8)>; MAX_IMAGE_SEGS],
    /// Request-side registrations retain the delivered image lease because it
    /// is the parent of every segment carve. Archive registrations name the
    /// service's endowed store directly and therefore have no per-entry
    /// returned parent.
    source: Option<ReturnedCapability>,
    /// Immutable provenance for request-side sources. Archive entries carry
    /// their equivalent authenticated metadata in the archive index.
    source_metadata: Option<ImageSourceMetadata>,
    /// Changed before the first revoke attempt. A refused teardown remains
    /// named and retryable, but can no longer satisfy `Acquire`.
    lifecycle: ImageLifecycle,
    /// The member window every segment carve came from: the carve subject,
    /// its base, the page-rounded span, and the exact registered length.
    /// Retained so [`ImageRequest::Export`] can carve the same span again at
    /// read-only rights without re-deriving it from the plan.
    window: Option<(SlotRef, u64, u64, u64)>,
    /// An archive DATA member ([`kore_abi::image::ARCHIVE_MEMBER_DATA`]): no
    /// segments, no descriptor (it stays [`ModuleDescriptor::empty`]), only
    /// the `window` above. The one bit that separates "lend these bytes"
    /// from "launch this" — in this table, not beside it.
    data: bool,
}

#[cfg(feature = "heap")]
impl ImageEntry {
    fn try_retire(&mut self) -> bool {
        self.lifecycle = ImageLifecycle::Retiring;
        self.try_cleanup()
    }

    fn try_cleanup(&mut self) -> bool {
        let mut complete = true;
        for (lease, _) in self.leases.iter_mut().flatten() {
            // Do not short-circuit: every independent segment gets one bounded
            // teardown attempt even if an earlier segment refused.
            complete = lease.try_retire() && complete;
        }
        // Parent last: revoking a returned source first would cascade through
        // every child and turn their owned slots into untracked dead records.
        // Keep its retry handle untouched until every child slot has closed.
        if complete && let Some(source) = self.source.as_mut() {
            complete = source.try_retire();
        }
        complete
    }

    fn grantable(&self) -> bool {
        self.lifecycle == ImageLifecycle::Live
            && self
                .leases
                .iter()
                .flatten()
                .all(|(lease, _)| lease.is_live())
            && self.source.as_ref().is_none_or(ReturnedCapability::is_live)
    }

    /// The record an archive DATA member registers: its name, stamp, and
    /// member window over `subject` at `base` — nothing else. `None` when the
    /// window is not page-shaped (unaligned base, empty, or a rounded end
    /// that overflows); the archive walk has already bounded it against the
    /// store.
    fn data(name: ImageName, seq: u64, subject: SlotRef, base: u64, len: u64) -> Option<Self> {
        let rounded = len.checked_next_multiple_of(PAGE_BYTES)?;
        if len == 0 || base % PAGE_BYTES != 0 {
            return None;
        }
        base.checked_add(rounded)?;
        Some(Self {
            name,
            seq,
            descriptor: ModuleDescriptor::empty(),
            needed: [None; MAX_DEPENDENCIES],
            leases: [const { None }; MAX_IMAGE_SEGS],
            source: None,
            source_metadata: None,
            lifecycle: ImageLifecycle::Live,
            window: Some((subject, base, rounded, len)),
            data: true,
        })
    }

    /// The `Acquire` data half: the descriptor and one `(lease slot, len)`
    /// per segment. A data member refuses [`NotImage`](ImageRefusal::NotImage)
    /// before anything else — it has no segments, and answering its empty
    /// descriptor with zero grants would read as an acquirable empty image.
    fn acquisition(
        &self,
    ) -> Result<(ModuleDescriptor, [Option<(SlotRef, u64)>; MAX_IMAGE_SEGS]), ImageRefusal> {
        if self.data {
            return Err(ImageRefusal::NotImage);
        }
        if !self.grantable() {
            return Err(ImageRefusal::Denied);
        }
        let mut grants = [None; MAX_IMAGE_SEGS];
        let mut count = 0usize;
        for (lease, _) in self.leases.iter().flatten() {
            let slot = lease.slot().ok_or(ImageRefusal::Denied)?;
            let dst = grants.get_mut(count).ok_or(ImageRefusal::Denied)?;
            *dst = Some((slot, lease.len()));
            count += 1;
        }
        if count != self.descriptor.seg_count as usize {
            return Err(ImageRefusal::Denied);
        }
        Ok((self.descriptor, grants))
    }

    /// The `Export` data half: `(subject, base, page-rounded span, exact
    /// length)` of the member window — identical for an image and a data
    /// member, which is the point: the seed copies bytes without caring
    /// which it is. A generation whose window was not retained (a shape that
    /// predates the record) exports nothing rather than guessing a span.
    fn export_window(&self) -> Result<(SlotRef, u64, u64, u64), ImageRefusal> {
        if !self.grantable() {
            return Err(ImageRefusal::Denied);
        }
        self.window.ok_or(ImageRefusal::Denied)
    }
}

/// One archive member's admission verdict — the whole-archive preflight's
/// unit, run before any record or lease exists.
#[cfg(feature = "heap")]
enum MemberAdmission {
    /// An ELF image the checker accepted; its plan sizes the slot budget.
    Image(CheckedImage),
    /// A data member: name, profile-zero, and page-shaped bounds checked;
    /// no ELF check, no carve, no owned slot.
    Data,
}

/// Retry at most one failed registration's bounded segment set per pass.
/// Advance past a refusal so it cannot starve independent cleanup records.
/// Live images and client-owned retirement retries are never selected.
#[cfg(feature = "heap")]
fn retry_failed_registration(
    images: &mut [Option<ImageEntry>],
    cursor: &mut usize,
    retire: impl FnMut(&mut ImageEntry) -> bool,
) {
    retry_cleanup_row(
        images,
        cursor,
        |entry| entry.lifecycle == ImageLifecycle::RegistrationRollback,
        retire,
    );
}

#[cfg(feature = "heap")]
fn retry_cleanup_row<T>(
    rows: &mut [Option<T>],
    cursor: &mut usize,
    eligible: impl Fn(&T) -> bool,
    mut retire: impl FnMut(&mut T) -> bool,
) {
    if rows.is_empty() {
        *cursor = 0;
        return;
    }
    let start = *cursor % rows.len();
    let candidate = (start..rows.len())
        .chain(0..start)
        .find(|&idx| rows[idx].as_ref().is_some_and(&eligible));
    if let Some(idx) = candidate {
        *cursor = if idx + 1 == rows.len() { 0 } else { idx + 1 };
        if rows[idx].as_mut().is_some_and(&mut retire) {
            rows[idx] = None;
        }
    }
}

/// Occupancy snapshot (the [`MmStats`](kore_svc_mm::MmStats) shape).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageStats {
    /// Live table entries (all versions of all names).
    pub images: u32,
    /// Successful `Register`s since bind (retires do not subtract).
    pub registered: u32,
    /// Refusals since bind — counted, never silent.
    pub refused: u32,
    /// Occupied face slots (live faces + retire tombstones) — the
    /// parent's spawning capacity, visible so a face leak is loud.
    pub faces: u32,
}

/// Allocation-free startup diagnostics carried above the generic service-start
/// rung. These values are local diagnostics, not part of the image wire ABI.
#[cfg(feature = "heap")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum ImageStartFailure {
    TableAllocation = 1,
    StoreMap = 2,
    ArchiveHeader = 3,
    ArchiveIndex = 4,
    MemberName = 6,
    MemberBounds = 7,
    MemberImage = 8,
    SlotBudget = 9,
    RegisterCarve = 10,
    ValidationGeometry = 11,
    FaceSupplyGeometry = 12,
    FaceWindowMap = 13,
    /// An unknown [`ArchiveEntry::kind`](kore_abi::image::ArchiveEntry), or
    /// a data member carrying a launch profile. Appended, never renumbered.
    MemberKind = 14,
}

#[cfg(feature = "heap")]
impl From<ImageStartFailure> for StartFailure {
    fn from(value: ImageStartFailure) -> Self {
        Self::new(value as u64)
    }
}

/// The acquire-only builds' protocol carrier: the same name, no host
/// state.  A heapless consumer (the startup linker) speaks the frozen
/// wire through `SvcClient<_, ImageService>` — the type parameter carries
/// the protocol constants and the rights vocabulary, never a live table.
#[cfg(not(feature = "heap"))]
#[derive(Debug)]
pub struct ImageService {
    _acquire_only: (),
}

/// The image service: the lessor over one byte store. The legacy X-mint stays
/// in its founding profile for pre-blessed allocations and profile selection;
/// executable finalization itself is lease-authorized.
#[cfg(feature = "heap")]
pub struct ImageService {
    /// The possessed byte store — the archive's own registrations carve it.
    store: ImageStoreToken,
    /// The possessed legacy X-mint. The consuming finalization path never
    /// names or scans it; keeping the role avoids conflating this change with
    /// the founding-profile selector migration.
    #[allow(dead_code)]
    xmint: XMintToken,
    /// Allocator-backed image generations. Retired holes retain stable indices
    /// for every other live generation; growth may reallocate the vector but
    /// faces name `(index, seq)`, never an address into it.
    images: alloc::vec::Vec<Option<ImageEntry>>,
    /// Founder-declared exact VA reused for one request-side validation map.
    validation: Option<kore_abi::task::layout::Window>,
    /// The known or uncertain validation mapping and any source not admitted
    /// into the catalogue. Neither may be replaced before ordered cleanup.
    validation_cleanup: ValidationCleanup,
    /// Shadowing stamp source (monotonic).
    next_seq: u64,
    /// `Register`s honored.
    registered: u32,
    /// Requests refused.
    refused: u32,
    /// The possessed face supply (slice C) — the grant subject every
    /// minted face's ring sub carves off.
    faces: SupplyToken,
    /// The possessed rendezvous mint grant — the face channels' root
    /// authority.
    channels: ChannelSpaceToken,
    /// Named, mm-backed ring arena. Cached allocations stay mapped while
    /// client roots are independently revoked between generations.
    face_heap: &'static kore_rt::Heap<kore_svc_mm::heap::MmSource>,
    memory: Option<kore_svc_mm::RingMm<'static, 'static, kore_rt::transport::TrapTransport>>,
    maintenance_at: Option<u64>,
    registration_cleanup_cursor: usize,
    face_cleanup_cursor: usize,
    heap_seed_cleanup: Option<LocalLease>,
    face_backings: alloc::vec::Vec<Option<FaceBacking>, FaceArena>,
    /// Growable face table; ids are validated indices.
    face_slots: alloc::vec::Vec<Option<FaceSlot>, FaceArena>,
    /// A minted pair awaiting host attach (at most one — the host
    /// settles between serve steps).
    pending_attach: Option<MintedPairSpec>,
    /// A retired face's pair identity awaiting host detach.
    pending_detach: Option<u16>,
    /// The face id of the last `FaceBank` served — `grants_dropped`'s
    /// correlation back-channel (the `Banked` reply carries no id).
    last_bank: Option<u32>,
    /// Faces delivered since bind (a dropped grant session subtracts —
    /// the mint never took effect).
    faces_minted: u32,
    /// Faces fully retired since bind.
    faces_retired: u32,
    /// Member windows exported since bind — the filesystem seed's meter.
    exported: u32,
}

/// Ensure `needed` independently reusable catalogue positions without an
/// infallible allocation. Existing holes are authority-neutral storage and are
/// always consumed before the vector grows; live indices never move.
#[cfg(feature = "heap")]
fn ensure_vacant_slots<T>(
    table: &mut alloc::vec::Vec<Option<T>>,
    needed: usize,
) -> Result<(), ImageRefusal> {
    let vacant = table.iter().filter(|entry| entry.is_none()).count();
    let missing = needed.saturating_sub(vacant);
    if missing == 0 {
        return Ok(());
    }
    let new_len = table
        .len()
        .checked_add(missing)
        .ok_or(ImageRefusal::TableFull)?;
    // Keep Vec's amortized growth policy: runtime registration is a control
    // path, but repeatedly installing generations must not copy the complete
    // catalogue on every successful append.
    table
        .try_reserve(missing)
        .map_err(|_| ImageRefusal::TableFull)?;
    table.resize_with(new_len, || None);
    Ok(())
}

/// The protocol's **rights vocabulary** — present in every build,
/// including the heapless client-only one the startup linker links.  A
/// consumer that only speaks the wire still has to know what a grant
/// carries; that is a fact about the protocol, not about the host.
#[cfg(feature = "heap")]
impl core::fmt::Debug for ImageService {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ImageService")
            .field("faces", &self.face_slots.len())
            .finish_non_exhaustive()
    }
}

impl ImageService {
    /// Extend this task's arena until one admitted image's exact peak demand
    /// and every still-vacant loader face can coexist.
    ///
    /// Image rows grow on demand, so there is no speculative per-row slot
    /// reservation. The face reserve remains structural: image registration
    /// cannot consume the capacity already promised to a later `FaceMint`.
    #[cfg(feature = "heap")]
    fn ensure_owned_slot_budget(&self, image_demand: usize) -> bool {
        #[cfg(target_os = "none")]
        {
            let vacant_faces = self.face_slots.iter().filter(|slot| slot.is_none()).count();
            let Some(required) = owned_slot_requirement(image_demand, vacant_faces) else {
                return false;
            };
            while verbs::available_owned_slots() < required {
                let before = verbs::available_owned_slots();
                if kore_rt::cap::verbs::extend_arena().is_none()
                    || verbs::available_owned_slots() <= before
                {
                    return false;
                }
            }
            true
        }
        #[cfg(not(target_os = "none"))]
        {
            let _ = image_demand;
            true
        }
    }

    /// Exec's permanent image-control pair. This protocol identity is needed
    /// by heapless acquire-only clients as well as by the heap-backed host, so
    /// it must not live behind the host model's `heap` feature.
    const FACE_OWNER_VIA: u16 =
        (<ImageService as kore_rt::svc::SvcProtocol>::SERVE_PAIRS - 1) as u16;

    /// The image-control pair authorized to install a new immutable generation
    /// from a transferred source lease. This is exec's standing loader pair
    /// (`FACE_OWNER_VIA`): registration was folded into it through
    /// `RingImage::with_registration`, and the source lease lands there. The
    /// separate `RingRegistrar` / serve-pair-0 split the type system still
    /// carries was never wired to any client, so the one realized image-control
    /// authority — exec's loader pair, which no other task reaches — holds both
    /// the registrar and the generation-owner roles. They stay distinct request
    /// types on the one pair, so nothing collides. (Was `via == 0`, a stale
    /// gate that refused exec's real registration and left the FS-source
    /// install path dead — register_fs_image never ran on metal to catch it.)
    const fn registrar_via(via: u16) -> bool {
        via == Self::FACE_OWNER_VIA
    }

    const fn generation_owner_via(via: u16) -> bool {
        via == Self::FACE_OWNER_VIA
    }
    /// Rights a request-side registrar must transfer with the image parent.
    /// `ACCESS` maps it for validation, `DERIVE|MODIFY` carves the bounded
    /// segment leases, and `REVOKE` makes malformed registration and final
    /// parent teardown recoverable.
    pub const REGISTER_SOURCE_RIGHTS: Rights = Rights::ACCESS
        .union(Rights::MODIFY)
        .union(Rights::DERIVE)
        .union(Rights::REVOKE);

    /// Rights an `Acquire` lease-cap grant carries: read + **derive** +
    /// **revoke** — deriving views into the *holder's own* space is
    /// exactly how a consumer maps a library (`res/IMAGE.md` §3).  No
    /// `MODIFY`: a holder cannot unshare others, and W on text is
    /// refused at the record layer besides.
    ///
    /// `REVOKE` is what makes the grant *usable as a founding supply*,
    /// and it grants no reach over the service's own lease.  A grant is
    /// an attenuated CHILD; revoking it kills that child's subtree and
    /// nothing above it, so §3's "cannot kill the library" still holds —
    /// the holder can only retire its own view of it.
    ///
    /// Without it an `Acquire` lease cannot be shared into a child at
    /// all: `Mem::share_rights` always mints the view `ACCESS|REVOKE`
    /// (+`MODIFY` on a writable share) and Σ5 is an explicit refusal
    /// rather than a clamp, so a `REVOKE`-less supply refuses every share
    /// off it.  The kernel already states the rule where it founds the
    /// first shell (`sched::found_first_shell`); this is the lessor side
    /// of the same fact.
    pub const LEASE_RIGHTS: Rights = Rights::ACCESS.union(Rights::DERIVE).union(Rights::REVOKE);

    /// Rights a minted loader face's **ring client sub** carries (slice
    /// C): `ACCESS` (the child's mapped window reads replies), `MODIFY`
    /// (its producer half writes requests), `DERIVE` (the spawn window
    /// is a view off the sub), `REVOKE` (the convey/reap custody chain
    /// retires it).
    /// The sub is a sibling of the service's own serving view — both
    /// children of the face supply — so its death never unmaps the
    /// lessor's side.
    pub const FACE_SUB_RIGHTS: Rights = Rights::ACCESS
        .union(Rights::MODIFY)
        .union(Rights::DERIVE)
        .union(Rights::REVOKE);
}

/// The **host model** — the live image table, the registration gate, and
/// the serving surface.  Behind `heap` because its table is fallibly
/// allocated: the acquire-only client seam links into tasks that have no
/// allocator at all (the startup linker's `role::LOADER` face).
#[cfg(feature = "heap")]
impl ImageService {
    /// Reserve enough allocator-backed records for `needed` new generations.
    /// Retired holes are reused before the heap grows.
    fn ensure_image_slots(&mut self, needed: usize) -> Result<(), ImageRefusal> {
        ensure_vacant_slots(&mut self.images, needed)
    }

    /// Retry the validation mapping before its unadmitted source. Unknown map
    /// results require a proved exact unshare before either name can reopen.
    fn retry_validation_cleanup(&mut self) -> bool {
        self.validation_cleanup.try_retire()
    }

    fn needs_maintenance(&self) -> bool {
        self.validation_cleanup.is_pending()
            || self.face_heap.source().needs_maintenance()
            || self.heap_seed_cleanup.is_some()
            || self
                .face_backings
                .iter()
                .flatten()
                .any(|backing| matches!(backing.root, LocalLease::Uncertain(_)))
            || self
                .face_slots
                .iter()
                .flatten()
                .any(|face| face.cleanup_pending)
            || self
                .images
                .iter()
                .flatten()
                .any(|entry| entry.lifecycle == ImageLifecycle::RegistrationRollback)
    }

    fn map_registration_source(
        &mut self,
        subject: SlotRef,
    ) -> Result<MappedLeaseView, ImageRefusal> {
        if !self.retry_validation_cleanup() {
            return Err(ImageRefusal::Denied);
        }
        let window = self.validation.ok_or(ImageRefusal::Denied)?;
        let (_, len) = verbs::lease_geometry(subject).ok_or(ImageRefusal::BadImage)?;
        let fits = len != 0
            && len % PAGE_BYTES == 0
            && window.base() % PAGE_BYTES == 0
            && len <= window.len()
            && window.base().checked_add(len).is_some();
        if !fits {
            return Err(ImageRefusal::BadImage);
        }
        match verbs::map_owned_lease_at_outcome(subject, window.base(), len) {
            Ok(view) => Ok(view),
            Err(verbs::LeaseMapError::Uncertain(owner)) => {
                self.validation_cleanup.mapping = Some(ValidationMapping::Uncertain(owner));
                Err(ImageRefusal::BadImage)
            }
            Err(verbs::LeaseMapError::Refused(_)) => Err(ImageRefusal::BadImage),
        }
    }

    fn finish_validation(&mut self, view: MappedLeaseView, source: Option<ReturnedCapability>) {
        // Registration admission either moved the parent into the catalogue
        // or left it with this validation. No await separates that decision
        // from persistent cleanup custody.
        debug_assert!(!self.validation_cleanup.is_pending());
        self.validation_cleanup.mapping = Some(ValidationMapping::Known(view));
        self.validation_cleanup.source = source;
        let _ = self.retry_validation_cleanup();
    }

    /// Occupancy snapshot.
    pub fn stats(&self) -> ImageStats {
        ImageStats {
            images: u32::try_from(self.images.iter().flatten().count()).unwrap_or(u32::MAX),
            registered: self.registered,
            refused: self.refused,
            faces: u32::try_from(self.face_slots.iter().flatten().count()).unwrap_or(u32::MAX),
        }
    }

    /// Tear down one face's retained channel authority — per channel:
    /// local send leaf first, then its root (the `ImageEntry` order —
    /// revoking a parent first would turn the child's owned slot into an
    /// untracked dead record).  A refused owner stays in place with its
    /// retry handle; `true` means every channel closed.  The root revoke
    /// is the T6 edge: every granted recv and conveyed send dies with
    /// it, wherever it traveled.
    fn retire_face_channels(
        roots: &mut [Option<ChannelRoot>; MAX_IMAGE_SEGS],
        sends: &mut [Option<ChanSendToken>; MAX_IMAGE_SEGS],
    ) -> bool {
        let mut complete = true;
        for (send, root) in sends.iter_mut().zip(roots.iter_mut()) {
            if let Some(tok) = send.take() {
                match tok.try_retire() {
                    Ok(_) => {}
                    Err(tok) => {
                        *send = Some(tok);
                        complete = false;
                        continue;
                    }
                }
            }
            if let Some(tok) = root.take() {
                match tok.try_retire() {
                    Ok(_) => {}
                    Err(tok) => {
                        *root = Some(tok);
                        complete = false;
                    }
                }
            }
        }
        complete
    }

    /// An uncertain allocation carve must close before its pages can return
    /// to Heap. No face ever used this backing, so only its own root is due.
    fn retry_backing_cleanup(&mut self) {
        use core::alloc::{Allocator, Layout};
        let Some(k) = self.face_backings.iter().position(|backing| {
            backing
                .as_ref()
                .is_some_and(|backing| matches!(backing.root, LocalLease::Uncertain(_)))
        }) else {
            return;
        };
        let backing = self.face_backings[k].as_mut().unwrap();
        if !backing.root.try_retire() {
            return;
        }
        let bytes = backing.bytes;
        self.face_backings[k] = None;
        let layout =
            Layout::from_size_align(FACE_RING_STRIDE as usize, PAGE_BYTES as usize).unwrap();
        // SAFETY: no face was exposed from this root and native cleanup has
        // proved that the uncertain carve cannot still reach these pages.
        unsafe {
            self.face_heap.deallocate(bytes.cast(), layout);
        }
    }

    fn prepare_face(&mut self) -> bool {
        use core::alloc::{Allocator, Layout};
        self.retry_backing_cleanup();
        let k = self
            .face_slots
            .iter()
            .position(Option::is_none)
            .unwrap_or(self.face_slots.len());
        if k >= u16::MAX as usize - kore_rt::svc::MAX_SERVE_PAIRS {
            return false;
        }
        if k == self.face_slots.len() {
            if self.face_slots.try_reserve(1).is_err() || self.face_backings.try_reserve(1).is_err()
            {
                return false;
            }
            self.face_slots.push(None);
            self.face_backings.push(None);
        }
        if !self.ensure_owned_slot_budget(4) {
            return false;
        }
        if self.face_backings[k].is_none() {
            let layout =
                Layout::from_size_align(FACE_RING_STRIDE as usize, PAGE_BYTES as usize).unwrap();
            let Ok(bytes) = self.face_heap.allocate(layout) else {
                return false;
            };
            // SAFETY: a live page-exclusive heap allocation; this cached root
            // and its children are never revoked after returning it to Heap.
            let root = unsafe {
                self.face_heap.source().carve_allocation_outcome(
                    bytes.cast::<u8>().as_ptr() as usize,
                    FACE_RING_STRIDE as usize,
                )
            };
            let root = match root {
                Ok(root) => LocalLease::Known(root),
                Err(verbs::LeaseCarveError::Uncertain(root)) => {
                    self.face_backings[k] = Some(FaceBacking {
                        root: LocalLease::Uncertain(root),
                        bytes,
                    });
                    return false;
                }
                Err(verbs::LeaseCarveError::Refused(_)) => {
                    unsafe {
                        self.face_heap.deallocate(bytes.cast(), layout);
                    }
                    return false;
                }
            };
            self.face_backings[k] = Some(FaceBacking { root, bytes });
        }
        self.face_backings[k]
            .as_ref()
            .and_then(|backing| backing.root.slot())
            .is_some()
    }

    /// Unwind a face's local owners. An unpublished face can release its row
    /// after cleanup; a possibly delivered face keeps its id for client retry.
    fn unwind_face(&mut self, k: usize, unpublished: bool) {
        if let Some(entry) = self.face_slots.get_mut(k)
            && let Some(slot) = entry.as_mut()
        {
            if slot.unwind(unpublished) {
                *entry = None;
            }
        }
    }

    /// The `FaceMint` flow (slice C): gate the caller → claim a free
    /// slot → mint the grant channels → format the slot's ring stride →
    /// queue the pair for host attach. Returns the face id; the serve arm
    /// banks the recv endpoints and stages the complete client face in the
    /// same reply. Any failure unwinds whole — a refusal mints nothing.
    fn face_mint(
        &mut self,
        via: u16,
        generation: ImageGeneration,
        closure: [Option<DependencyBinding>; MAX_DEPENDENCIES],
    ) -> Result<u32, ImageRefusal> {
        if via != Self::FACE_OWNER_VIA {
            return Err(ImageRefusal::Denied);
        }
        if self.pending_attach.is_some() {
            // The host settles between serve steps, so a live spec here
            // means the engine invariant broke — refuse rather than
            // alias a pair identity.
            return Err(ImageRefusal::Denied);
        }
        if !self.prepare_face() {
            return Err(ImageRefusal::TableFull);
        }
        let k = self
            .face_slots
            .iter()
            .position(Option::is_none)
            .ok_or(ImageRefusal::TableFull)?;

        let backing = self.face_backings[k].as_ref().ok_or(ImageRefusal::Denied)?;
        let parent = backing.root.slot().ok_or(ImageRefusal::Denied)?;
        let base = backing.bytes.cast::<u8>().as_ptr();

        // Occupy the row before the first native result. Every partial owner
        // remains here through refusal; no operation is adopted by a new face.
        self.face_slots[k] = Some(FaceSlot::preparing(generation, closure));
        let slot = self.face_slots[k].as_mut().unwrap();
        match verbs::carve_owned_lease_window_outcome(parent, 0, FACE_RING_STRIDE) {
            Ok(owner) => slot.ring_root = Some(LocalLease::Known(owner)),
            Err(verbs::LeaseCarveError::Uncertain(owner)) => {
                slot.ring_root = Some(LocalLease::Uncertain(owner));
                self.unwind_face(k, true);
                return Err(ImageRefusal::Denied);
            }
            Err(verbs::LeaseCarveError::Refused(_)) => {
                self.unwind_face(k, true);
                return Err(ImageRefusal::Denied);
            }
        }

        // Mint the grant channels: root + send retained, the mint's own
        // recv retired at once — the banked recvs are FRESH derives in the
        // mint reply's grant session so I3 unwinds them with it. Exactly one
        // live recv per channel ever exists.
        let mut ok = true;
        for i in 0..MAX_IMAGE_SEGS {
            match verbs::mint_channel(&self.channels) {
                Ok(pair) => {
                    slot.roots[i] = Some(pair.root);
                    slot.sends[i] = Some(pair.send);
                    if let Err(recv) = pair.recv.try_retire() {
                        // Retain the actual refused leaf alongside its parent
                        // until cleanup settles; no banked recv is granted.
                        slot.pending_recv = Some(recv);
                        ok = false;
                        break;
                    }
                }
                Err(kore_rt::cap::ChannelMintError::Uncertain(owner)) => {
                    slot.uncertain_channel = Some(owner);
                    ok = false;
                    break;
                }
                Err(_) => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            self.unwind_face(k, true);
            return Err(ImageRefusal::Denied);
        }

        // Format this row's cached, page-exclusive heap block. Its previous
        // client root closed before the row became vacant, and this mint now
        // occupies it, so no old client or other mint can alias the stride.
        // SAFETY: FaceBacking retains the entire FACE_RING_STRIDE allocation
        // in the service's mapped heap; its pages remain cached for every
        // successful face generation. The mutable borrow ends before attach.
        let stride = FACE_RING_STRIDE as usize;
        let pair_bytes =
            kore_rt::svc::window_bytes::<Self>(kore_rt::svc::REQ_CAP, kore_rt::svc::REP_CAP);

        let region = unsafe { core::slice::from_raw_parts_mut(base, stride) };
        region.fill(0);
        if kore_rt::svc::init_window::<Self>(
            &mut region[..pair_bytes],
            kore_rt::svc::REQ_CAP,
            kore_rt::svc::REP_CAP,
        )
        .is_err()
        {
            self.unwind_face(k, true);
            return Err(ImageRefusal::Denied);
        }

        let mut grant_tx: [Option<SlotRef>; MAX_GRANTS] = [None; MAX_GRANTS];
        for (dst, send) in grant_tx.iter_mut().zip(slot.sends.iter()) {
            *dst = send.as_ref().map(ChanSendToken::slot);
        }
        // SAFETY: `region` proved this mapped block in bounds. The occupied
        // face row makes it exclusive until retire detaches the pair, and
        // FaceBacking outlives the attached host. `init_window` formatted it.
        let spec = unsafe {
            MintedPairSpec::new(
                core::ptr::NonNull::new_unchecked(base),
                pair_bytes,
                grant_tx,
                face_via(k),
                // Kickless: the loader-face client spin-polls (the stub
                // shape); a reply kick would be pure spurious wakes.
                None,
            )
        };
        slot.retiring = false;
        slot.cleanup_pending = false;
        self.pending_attach = Some(spec);
        self.faces_minted = self.faces_minted.saturating_add(1);
        Ok(k as u32)
    }

    /// The `FaceBank` gate: correlate the id, latch once-per-mint, and
    /// hand the serve arm the nine root slots its fresh recv derives
    /// name.  The latch is set before the grants stage;
    /// `grants_dropped` reopens it if the session unwound.
    fn face_bank(
        &mut self,
        via: u16,
        face: u32,
    ) -> Result<[Option<SlotRef>; MAX_IMAGE_SEGS], ImageRefusal> {
        if via != Self::FACE_OWNER_VIA {
            return Err(ImageRefusal::Denied);
        }
        let k = face_slot_of(face).ok_or(ImageRefusal::Unknown)?;
        let slot = self
            .face_slots
            .get_mut(k)
            .and_then(Option::as_mut)
            .ok_or(ImageRefusal::Unknown)?;
        if slot.retiring || slot.banked {
            return Err(ImageRefusal::Denied);
        }
        let mut out: [Option<SlotRef>; MAX_IMAGE_SEGS] = [None; MAX_IMAGE_SEGS];
        for (dst, root) in out.iter_mut().zip(slot.roots.iter()) {
            *dst = root.as_ref().map(ChannelRoot::slot);
        }
        if out.iter().any(Option::is_none) {
            return Err(ImageRefusal::Denied);
        }
        slot.banked = true;
        self.last_bank = Some(face);
        Ok(out)
    }

    /// The `FaceRetire` flow: detach the pair (bookkeeping — the face
    /// stops serving even if a token below refuses), then revoke the
    /// channels and free the slot.  A refused owner leaves a retryable
    /// tombstone; the detach re-queues harmlessly on retry (the host's
    /// detach of an unknown identity is a no-op).
    fn face_retire(&mut self, via: u16, face: u32) -> Result<(), ImageRefusal> {
        if via != Self::FACE_OWNER_VIA {
            return Err(ImageRefusal::Denied);
        }
        let k = face_slot_of(face).ok_or(ImageRefusal::Unknown)?;
        if self.face_slots.get(k).and_then(Option::as_ref).is_none() {
            return Err(ImageRefusal::Unknown);
        }
        if self.pending_detach.is_some() {
            // One detach settles per step; a live one here means the
            // engine invariant broke — refuse rather than lose it.
            return Err(ImageRefusal::Denied);
        }
        self.pending_detach = Some(face_via(k));
        let Some(slot) = self.face_slots.get_mut(k).and_then(Option::as_mut) else {
            return Err(ImageRefusal::Unknown);
        };
        slot.retiring = true;
        slot.cleanup_pending = false;
        if !slot.try_cleanup() {
            // Tombstone: the refused owners keep their retry handles and
            // the slot stays occupied (visible in `summarize`) until a
            // later retire closes them.
            return Err(ImageRefusal::Denied);
        }
        self.face_slots[k] = None;
        self.faces_retired = self.faces_retired.saturating_add(1);
        Ok(())
    }

    /// The newest live entry named `name` (shadowing: greatest stamp
    /// wins), or `None`.
    fn lookup(&self, name: &ImageName) -> Option<usize> {
        self.images
            .iter()
            .enumerate()
            .filter_map(|(i, e)| e.as_ref().filter(|e| e.name == *name).map(|e| (i, e.seq)))
            .max_by_key(|&(_, seq)| seq)
            .map(|(i, _)| i)
    }

    /// Resolve an endpoint-stamped serving identity to its immutable image
    /// generation. No public integer is accepted by the request wire.
    fn face_generation(&self, via: u16) -> Option<ImageGeneration> {
        let slot = usize::from(via).checked_sub(kore_rt::svc::MAX_SERVE_PAIRS)?;
        self.face_slots
            .get(slot)
            .and_then(Option::as_ref)
            .filter(|face| !face.retiring)
            .map(|face| face.generation)
    }

    fn template_binding(
        &self,
        face: u32,
    ) -> Option<(
        ImageGeneration,
        [Option<DependencyBinding>; MAX_DEPENDENCIES],
    )> {
        let slot = face_slot_of(face)?;
        self.face_slots
            .get(slot)
            .and_then(Option::as_ref)
            .filter(|face| !face.retiring)
            .map(|face| (face.generation, face.closure))
    }

    fn generation_entry(&self, generation: ImageGeneration) -> Option<&ImageEntry> {
        let slot = generation.slot;
        self.images
            .get(slot)
            .and_then(Option::as_ref)
            .filter(|entry| generation_matches(generation, slot, entry.seq) && entry.grantable())
    }

    /// Resolve and freeze the complete dependency graph at installation.
    /// Missing, non-library, cyclic-overflow, or over-bound graphs refuse
    /// before any executable peer is published.
    fn freeze_closure(
        &self,
        root: ImageGeneration,
    ) -> Result<[Option<DependencyBinding>; MAX_DEPENDENCIES], ImageRefusal> {
        let entry = self.generation_entry(root).ok_or(ImageRefusal::Unknown)?;
        let mut queue = [None; MAX_DEPENDENCIES];
        let mut queued = 0usize;
        for name in entry.needed.into_iter().flatten() {
            if queue[..queued].contains(&Some(name)) {
                continue;
            }
            *queue.get_mut(queued).ok_or(ImageRefusal::Denied)? = Some(name);
            queued += 1;
        }
        let mut closure = [None; MAX_DEPENDENCIES];
        let mut cursor = 0usize;
        while cursor < queued {
            let name = queue[cursor].ok_or(ImageRefusal::Denied)?;
            let slot = self.lookup(&name).ok_or(ImageRefusal::Unknown)?;
            let dependency = self
                .images
                .get(slot)
                .and_then(Option::as_ref)
                .filter(|entry| entry.grantable() && entry.descriptor.kind == kind::LIBRARY)
                .ok_or(ImageRefusal::Denied)?;
            let generation = ImageGeneration {
                slot,
                seq: dependency.seq,
            };
            closure[cursor] = Some(DependencyBinding { name, generation });
            for transitive in dependency.needed.into_iter().flatten() {
                if queue[..queued].contains(&Some(transitive)) {
                    continue;
                }
                *queue.get_mut(queued).ok_or(ImageRefusal::Denied)? = Some(transitive);
                queued += 1;
            }
            cursor += 1;
        }
        Ok(closure)
    }

    fn face_dependency_generation(&self, via: u16, name: &ImageName) -> Option<ImageGeneration> {
        let slot = usize::from(via).checked_sub(kore_rt::svc::MAX_SERVE_PAIRS)?;
        self.face_slots
            .get(slot)
            .and_then(Option::as_ref)
            .filter(|face| !face.retiring)
            .and_then(|face| {
                face.closure
                    .iter()
                    .flatten()
                    .find(|binding| binding.name == *name)
                    .map(|binding| binding.generation)
            })
    }

    fn reply_face(
        &mut self,
        via: u16,
        generation: ImageGeneration,
        closure: [Option<DependencyBinding>; MAX_DEPENDENCIES],
        module: ModuleDescriptor,
        sv: &mut Serve<'_, ImageResponse>,
    ) {
        match self.face_mint(via, generation, closure) {
            Ok(face) => match self.face_bank(via, face) {
                Ok(roots) => {
                    self.last_bank = None;
                    sv.grant(
                        self.face_slots[face as usize]
                            .as_ref()
                            .unwrap()
                            .ring_root
                            .as_ref()
                            .unwrap()
                            .slot()
                            .unwrap(),
                        Self::FACE_SUB_RIGHTS,
                        GrantArg::Sub {
                            offset: 0,
                            len: FACE_RING_STRIDE,
                        },
                    );
                    for root in roots.into_iter().flatten() {
                        sv.grant(root, CHANNEL_ENDPOINT_RIGHTS, GrantArg::ChanRecv);
                    }
                    sv.reply(ImageResponse::Face {
                        face,
                        generation: generation.seq,
                        module,
                    });
                }
                Err(r) => {
                    self.pending_attach = None;
                    if let Some(k) = face_slot_of(face) {
                        self.unwind_face(k, true);
                    }
                    self.faces_minted = self.faces_minted.saturating_sub(1);
                    self.refused = self.refused.saturating_add(1);
                    sv.reply(ImageResponse::Refused(r));
                }
            },
            Err(r) => {
                self.refused = self.refused.saturating_add(1);
                sv.reply(ImageResponse::Refused(r));
            }
        }
    }

    fn acquisition(
        &self,
        generation: ImageGeneration,
    ) -> Result<(ModuleDescriptor, [Option<(SlotRef, u64)>; MAX_IMAGE_SEGS]), ImageRefusal> {
        let slot = generation.slot;
        self.images
            .get(slot)
            .and_then(Option::as_ref)
            .filter(|entry| generation_matches(generation, slot, entry.seq))
            .ok_or(ImageRefusal::Unknown)?
            .acquisition()
    }

    fn reply_acquisition(
        &mut self,
        generation: Result<ImageGeneration, ImageRefusal>,
        sv: &mut Serve<'_, ImageResponse>,
    ) {
        match generation.and_then(|generation| self.acquisition(generation)) {
            Ok((module, grants)) => {
                for (slot, len) in grants.into_iter().flatten() {
                    sv.grant(slot, Self::LEASE_RIGHTS, GrantArg::Sub { offset: 0, len });
                }
                sv.reply(ImageResponse::Acquired { module });
            }
            Err(refusal) => {
                self.refused = self.refused.saturating_add(1);
                sv.reply(ImageResponse::Refused(refusal));
            }
        }
    }

    /// Bounds-check one archive member and return its store-relative base and
    /// bytes.  The wire is untrusted: every narrowing conversion and addition
    /// is checked before slicing.
    fn archive_member<'a>(
        bytes: &'a [u8],
        entry: &kore_abi::image::ArchiveEntry,
    ) -> Option<(u64, &'a [u8])> {
        let (offset, len) = (
            usize::try_from(entry.offset.get()).ok()?,
            usize::try_from(entry.len.get()).ok()?,
        );
        let end = offset.checked_add(len)?;
        Some((entry.offset.get(), bytes.get(offset..end)?))
    }

    /// Admit one archive member without minting anything: bounds for every
    /// kind, the ELF gate for images only. A data member must be
    /// profile-zero and page-shaped inside the store (its `Export` lends the
    /// page-rounded span, so the rounded end — the packer's padding — must
    /// lie inside `bytes` too). Each failure has its own start rung.
    fn admit_member(
        bytes: &[u8],
        entry: &kore_abi::image::ArchiveEntry,
    ) -> Result<MemberAdmission, ImageStartFailure> {
        use kore_abi::image::{ARCHIVE_MEMBER_DATA, ARCHIVE_MEMBER_IMAGE};
        if entry.name.as_str().is_none() || !kore_abi::image::profile_valid(entry.profile.get()) {
            return Err(ImageStartFailure::MemberName);
        }
        let (offset, window) =
            Self::archive_member(bytes, entry).ok_or(ImageStartFailure::MemberBounds)?;
        match entry.kind.get() {
            ARCHIVE_MEMBER_IMAGE => Self::check(window)
                .map(MemberAdmission::Image)
                .map_err(|_| ImageStartFailure::MemberImage),
            ARCHIVE_MEMBER_DATA => {
                if entry.profile.get() != 0 {
                    return Err(ImageStartFailure::MemberKind);
                }
                let rounded_end = (window.len() as u64)
                    .checked_next_multiple_of(PAGE_BYTES)
                    .and_then(|rounded| offset.checked_add(rounded));
                let shaped = !window.is_empty()
                    && offset % PAGE_BYTES == 0
                    && rounded_end.is_some_and(|end| end <= bytes.len() as u64);
                if !shaped {
                    return Err(ImageStartFailure::MemberBounds);
                }
                Ok(MemberAdmission::Data)
            }
            _ => Err(ImageStartFailure::MemberKind),
        }
    }

    /// Persistent arena names one checked image needs: one final lease per
    /// carve. Text's writable scratch is consumed before the session returns.
    fn owned_slots(checked: &CheckedImage) -> Option<usize> {
        checked
            .carve
            .iter()
            .flatten()
            .try_fold(0usize, |n, _| n.checked_add(1))
    }

    /// Text finalization momentarily needs one writable slot beside its final
    /// executable successor. Carves are serialized, so one extra name covers
    /// the complete generation regardless of segment count.
    fn needs_exec_scratch(checked: &CheckedImage) -> bool {
        checked
            .carve
            .iter()
            .flatten()
            .any(|carve| carve.kind == seg::KIND_TEXT)
    }

    /// Self-register every member of the endowed archive (the store's
    /// index walk — one `register` per member over ONE mapped window of
    /// the store).  Fails closed: a store that does not map, a bad
    /// magic, or an index that overruns the store refuses readiness —
    /// the service never beacons over a phantom catalog.
    fn register_archive(&mut self) -> Result<(), ImageStartFailure> {
        use kore_abi::image::{ARCHIVE_MAGIC, ArchiveEntry, ArchiveHeader};
        use zerocopy::TryFromBytes as _;

        let win = verbs::map_held(self.store.slot()).ok_or(ImageStartFailure::StoreMap)?;
        // SAFETY: `map_held` installed the store's span in this task's
        // own space for the task's lifetime.
        let bytes: &[u8] = unsafe { win.as_ref() };
        let (header, _) = ArchiveHeader::try_ref_from_prefix(bytes)
            .map_err(|_| ImageStartFailure::ArchiveHeader)?;
        if header.magic != ARCHIVE_MAGIC || header.reserved.get() != 0 {
            return Err(ImageStartFailure::ArchiveHeader);
        }
        let count = header.count.get() as usize;
        let index_bytes = bytes
            .get(ArchiveHeader::WIRE..)
            .ok_or(ImageStartFailure::ArchiveIndex)?;
        let (entries, _) = <[ArchiveEntry]>::try_ref_from_prefix_with_elems(index_bytes, count)
            .map_err(|_| ImageStartFailure::ArchiveIndex)?;
        // Validate the complete archive and its persistent-slot budget before
        // allocating records or minting the first lease. The authoritative
        // reservation still occurs in each owned carve, but this all-members
        // preflight prevents heap or arena pressure from publishing a prefix.
        let mut required_slots = 0usize;
        let mut needs_exec_scratch = false;
        for entry in entries {
            // A data member costs a record and nothing else: no carve, so
            // no owned slot and no exec scratch.
            let MemberAdmission::Image(checked) = Self::admit_member(bytes, entry)? else {
                continue;
            };
            let member_slots = Self::owned_slots(&checked).ok_or(ImageStartFailure::SlotBudget)?;
            required_slots = required_slots
                .checked_add(member_slots)
                .ok_or(ImageStartFailure::SlotBudget)?;
            needs_exec_scratch |= Self::needs_exec_scratch(&checked);
        }
        self.ensure_image_slots(entries.len())
            .map_err(|_| ImageStartFailure::TableAllocation)?;
        let required_slots = required_slots
            .checked_add(usize::from(needs_exec_scratch))
            .ok_or(ImageStartFailure::SlotBudget)?;
        if !self.ensure_owned_slot_budget(required_slots) {
            return Err(ImageStartFailure::SlotBudget);
        }

        // A zero-member archive is lawful (the store is simply empty) —
        // readiness then means "serving, nothing registered yet".
        for entry in entries {
            let (offset, window) =
                Self::archive_member(bytes, entry).ok_or(ImageStartFailure::MemberBounds)?;
            if entry.kind.get() == kore_abi::image::ARCHIVE_MEMBER_DATA {
                // Admitted above; the record is the window alone.
                self.register_data(entry.name, offset, window.len() as u64)
                    .map_err(|_| ImageStartFailure::RegisterCarve)?;
                continue;
            }
            if self
                .register(
                    entry.name,
                    entry.profile.get(),
                    None,
                    self.store.slot(),
                    window,
                    offset,
                    || None,
                )
                .is_err()
            {
                return Err(ImageStartFailure::RegisterCarve);
            }
        }
        Ok(())
    }

    /// Record one admitted archive DATA member: its window over the store,
    /// under a fresh shadowing stamp. No lease is minted — `Export` carves
    /// its read-only grant straight off the store span, as for an image.
    fn register_data(&mut self, name: ImageName, base: u64, len: u64) -> Result<(), ImageRefusal> {
        if name.as_str().is_none() {
            return Err(ImageRefusal::BadName);
        }
        self.ensure_image_slots(1)?;
        let idx = self
            .images
            .iter()
            .position(Option::is_none)
            .ok_or(ImageRefusal::TableFull)?;
        let next_seq = self
            .next_seq
            .checked_add(1)
            .ok_or(ImageRefusal::TableFull)?;
        let entry = ImageEntry::data(name, self.next_seq, self.store.slot(), base, len)
            .ok_or(ImageRefusal::BadImage)?;
        let dst = self.images.get_mut(idx).ok_or(ImageRefusal::TableFull)?;
        *dst = Some(entry);
        self.next_seq = next_seq;
        self.registered = self.registered.saturating_add(1);
        Ok(())
    }

    /// The `Register` gate — machine-dispatched into the kore-ld checker over
    /// the [`RelocArch`](kore_ld::reloc::RelocArch) impls.  One small seam on
    /// purpose: the check contract ([`CheckedImage`] / `CheckError`) is
    /// frozen and the service never sees ELF internals.
    fn check(bytes: &[u8]) -> Result<CheckedImage, ImageRefusal> {
        use kore_ld::reloc::{AArch64, RelocArch, Riscv64, X86_64};
        // The `e_machine` numbers, read off the RelocArch impls — the
        // seam's one source of truth (a free const, so `match` takes it).
        const EM_X86_64: u16 = X86_64::MACHINE;
        const EM_AARCH64: u16 = AArch64::MACHINE;
        const EM_RISCV64: u16 = Riscv64::MACHINE;
        // `e_machine` rides bytes 18..20 LE in every ELF64 header;
        // anything shorter cannot be an image at all.
        let machine = bytes
            .get(18..20)
            .and_then(|m| <&[u8; 2]>::try_from(m).ok())
            .map(|m| u16::from_le_bytes(*m))
            .ok_or(ImageRefusal::BadImage)?;
        match machine {
            EM_X86_64 => Self::check_via::<X86_64>(bytes),
            EM_AARCH64 => Self::check_via::<AArch64>(bytes),
            EM_RISCV64 => Self::check_via::<Riscv64>(bytes),
            _ => Err(ImageRefusal::BadImage),
        }
    }

    /// The one call onto [`kore_ld::check::check_image`] — the P2 gate this
    /// service registers through (parse → flag gate → reloc gate → carve
    /// plan).  The diagnostic `CheckError` variant collapses to
    /// [`BadImage`](ImageRefusal::BadImage) at the service boundary
    /// (`res/IMAGE.md` §3's closed refusal vocabulary).
    fn check_via<A: kore_ld::reloc::RelocArch>(bytes: &[u8]) -> Result<CheckedImage, ImageRefusal> {
        kore_ld::check::check_image::<A>(bytes).map_err(|_| ImageRefusal::BadImage)
    }

    /// The `Register` flow: validate → carve in place → finalize text
    /// (plain leases for everything else) → stash → record. `bytes` is the
    /// already-mapped image window (the store's span for archive
    /// members, mapped once by [`register_archive`](Self::register_archive);
    /// a delivered registration maps its lease in the serve arm) and
    /// `subject` the held lease the carves name, with `base` the
    /// member's offset into it. A mid-flight failure never publishes an
    /// acquirable entry. Successfully revoked temporary leases release their
    /// reservations; if cleanup itself refuses, a non-grantable tombstone
    /// retains the retry handles rather than leaking authority or reopening a
    /// still-live slot.
    fn register<F>(
        &mut self,
        name: ImageName,
        profile: u32,
        source_metadata: Option<ImageSourceMetadata>,
        subject: SlotRef,
        bytes: &[u8],
        base: u64,
        source: F,
    ) -> Result<ModuleDescriptor, ImageRefusal>
    where
        F: FnOnce() -> Option<ReturnedCapability>,
    {
        if name.as_str().is_none() || !kore_abi::image::profile_valid(profile) {
            return Err(ImageRefusal::BadName);
        }
        if self
            .lookup(&name)
            .and_then(|idx| self.images.get(idx))
            .and_then(Option::as_ref)
            .is_some_and(|entry| entry.lifecycle != ImageLifecycle::Live)
        {
            // A failed teardown keeps the newest version's retry handle. Do
            // not shadow it with a new row and make that handle unreachable.
            return Err(ImageRefusal::Denied);
        }
        // The carve bound is the member window's PAGE-ROUNDED end, not
        // its raw length: the plan rounds segment windows outward to
        // whole pages (the tail page of a member is padding the archive
        // carries, inside the store's span) — bounding at the raw length
        // would refuse every rounded tail carve.
        let rounded_len = (bytes.len() as u64)
            .checked_next_multiple_of(PAGE_BYTES)
            .ok_or(ImageRefusal::BadImage)?;
        let limit = base
            .checked_add(rounded_len)
            .ok_or(ImageRefusal::BadImage)?;
        let checked = Self::check(bytes)?;
        let mut descriptor = checked.descriptor;
        // A delivered (FS-sourced) registration carries no archive
        // authentication for its launch profile: `source_metadata.is_some()`
        // is exactly the runtime-install path, where the profile is a caller
        // assertion.  Force it to zero so a durable, task-reachable executable
        // can never claim a launch capability the boot archive did not grant.
        // The archive path (`source_metadata` is `None`) keeps its trusted,
        // build-time-fixed profile.  (`res/IMAGE.md`: profiles are
        // archive-authenticated metadata.)
        let profile = if source_metadata.is_some() {
            0
        } else {
            profile
        };
        descriptor.profile = profile.into();
        descriptor.reserved = 0.into();
        let required_slots = Self::owned_slots(&checked).ok_or(ImageRefusal::TableFull)?;
        let peak_slots = required_slots
            .checked_add(usize::from(Self::needs_exec_scratch(&checked)))
            .ok_or(ImageRefusal::TableFull)?;
        // Allocate the record and arena capacity before `source()` transfers
        // authority from ServeHost. No admission failure below this line can
        // be mistaken for permission to drop the returned parent.
        self.ensure_image_slots(1)?;
        if !self.ensure_owned_slot_budget(peak_slots) {
            return Err(ImageRefusal::TableFull);
        }
        let idx = self
            .images
            .iter()
            .position(Option::is_none)
            .ok_or(ImageRefusal::TableFull)?;

        // Carve + lease per plan segment, in descriptor order.  Carve
        // offsets are the member window's base + the plan's image-file
        // offsets; the sub-carve's over-grant guard (page-aligned,
        // inside the span) is the kernel's, and any misalignment the
        // plan could carry is refused here first — fail-closed, never a
        // guessed extent.
        let next_seq = self
            .next_seq
            .checked_add(1)
            .ok_or(ImageRefusal::TableFull)?;
        // Everything before this point is pure admission. A request-side
        // input stays owned by `ServeHost` until this exact transition; from
        // here onward either a live entry or a retryable tombstone owns it.
        let source = source();
        let mut leases: [Option<(ImageLease, u8)>; MAX_IMAGE_SEGS] =
            [const { None }; MAX_IMAGE_SEGS];
        let mut ok = true;
        for (i, c) in checked.carve.iter().enumerate() {
            let Some(c) = c else { continue };
            let aligned = c.file_off % PAGE_BYTES == 0
                && c.len % PAGE_BYTES == 0
                && c.len != 0
                && base
                    .checked_add(c.file_off)
                    .and_then(|off| off.checked_add(c.len))
                    .is_some_and(|carve_end| carve_end <= limit);
            if !aligned {
                ok = false;
                break;
            }
            let off = base + c.file_off;
            if c.kind == seg::KIND_TEXT {
                // Text: carve a writable child and consume it into one
                // executable successor in ONE session (`res/IMAGE.md` §6).
                // Only proved success releases the scratch name. An unknown
                // completion retains both names in this entry's rollback.
                match verbs::carve_owned_exec_window_outcome(subject, off, c.len) {
                    Ok(executable) => {
                        leases[i] = Some((ImageLease::Executable(executable, c.len), c.kind));
                    }
                    Err(verbs::ExecCarveError::Uncertain(owner)) => {
                        leases[i] = Some((ImageLease::UncertainExecutable(owner), c.kind));
                        ok = false;
                        break;
                    }
                    Err(verbs::ExecCarveError::Refused(_)) => {
                        ok = false;
                        break;
                    }
                }
            } else {
                match verbs::carve_owned_lease_window_outcome(subject, off, c.len) {
                    Ok(lease) => {
                        leases[i] = Some((ImageLease::Plain(lease, c.len), c.kind));
                    }
                    Err(verbs::LeaseCarveError::Uncertain(owner)) => {
                        leases[i] = Some((ImageLease::UncertainPlain(owner), c.kind));
                        ok = false;
                        break;
                    }
                    Err(verbs::LeaseCarveError::Refused(_)) => {
                        ok = false;
                        break;
                    }
                }
            }
        }
        if !ok {
            // Retain every earlier carve and the returned parent before
            // replying with refusal. Validation may still hold a mapped view
            // of that parent, so cleanup starts only after validation closes.
            // The caller has no successful generation to retire: the service
            // drives this rollback even if no further request arrives.
            let tombstone = ImageEntry {
                window: None,
                name,
                seq: self.next_seq,
                descriptor,
                needed: checked.needed,
                leases,
                source,
                source_metadata,
                lifecycle: ImageLifecycle::RegistrationRollback,
                data: false,
            };
            self.images[idx] = Some(tombstone);
            self.next_seq = next_seq;
            return Err(ImageRefusal::BadImage);
        }

        let seq = self.next_seq;
        self.next_seq = next_seq;
        let entry = ImageEntry {
            name,
            seq,
            descriptor,
            needed: checked.needed,
            leases,
            source,
            source_metadata,
            lifecycle: ImageLifecycle::Live,
            window: Some((subject, base, rounded_len, bytes.len() as u64)),
            data: false,
        };
        let Some(dst) = self.images.get_mut(idx) else {
            return Err(ImageRefusal::TableFull);
        };
        *dst = Some(entry);
        self.registered = self.registered.saturating_add(1);
        Ok(descriptor)
    }

    /// The `Retire` flow: drop the newest version and retire its stashed
    /// leases and their retained parents (tag-retire; module docs for the
    /// T6 posture).
    fn retire(&mut self, name: &ImageName) -> Result<(), ImageRefusal> {
        if name.as_str().is_none() {
            return Err(ImageRefusal::BadName);
        }
        let idx = self.lookup(name).ok_or(ImageRefusal::Unknown)?;
        let entry = self.images[idx].as_mut().ok_or(ImageRefusal::Unknown)?;
        entry.lifecycle = ImageLifecycle::Retiring;
        if self.validation_cleanup.is_pending() {
            return Err(ImageRefusal::Denied);
        }
        let complete = self
            .images
            .get_mut(idx)
            .and_then(Option::as_mut)
            .is_some_and(ImageEntry::try_retire);
        if !complete {
            // Keep the retiring entry and every refused owner in place. A
            // later Retire can retry; Acquire cannot grant a partial image.
            return Err(ImageRefusal::Denied);
        }
        let retired = self
            .images
            .get_mut(idx)
            .and_then(Option::take)
            .ok_or(ImageRefusal::Unknown)?;
        drop(retired);
        Ok(())
    }

    /// Retire one exact generation through a still-live template binding.
    /// Re-registration under the same display name cannot retarget this
    /// operation because no name participates in resolution.
    fn retire_bound(&mut self, via: u16, template: u32) -> Result<(), ImageRefusal> {
        if via != Self::FACE_OWNER_VIA {
            return Err(ImageRefusal::Denied);
        }
        let (generation, _) = self
            .template_binding(template)
            .ok_or(ImageRefusal::Unknown)?;
        let idx = generation.slot;
        let entry = self
            .images
            .get_mut(idx)
            .and_then(Option::as_mut)
            .filter(|entry| generation_matches(generation, idx, entry.seq))
            .ok_or(ImageRefusal::Unknown)?;
        entry.lifecycle = ImageLifecycle::Retiring;
        if self.validation_cleanup.is_pending() {
            return Err(ImageRefusal::Denied);
        }
        let complete = self
            .images
            .get_mut(idx)
            .and_then(Option::as_mut)
            .filter(|entry| generation_matches(generation, idx, entry.seq))
            .is_some_and(ImageEntry::try_retire);
        if !complete {
            return Err(ImageRefusal::Denied);
        }
        let _ = self
            .images
            .get_mut(idx)
            .and_then(Option::take)
            .ok_or(ImageRefusal::Unknown)?;
        Ok(())
    }
}

#[cfg(feature = "heap")]
impl SystemService for ImageService {
    const NAME: &'static str = "image";
    type Endowment = ImageEndowment;

    fn bind(e: ImageEndowment) -> Self {
        let arena: FaceArena = alloc::boxed::Box::leak(
            alloc::boxed::Box::try_new_in(
                kore_rt::Heap::new_in(kore_rt::heap::HeapSupplySource::new()),
                kore_rt::Heap::new_in(kore_rt::heap::HeapSupplySource::new()),
            )
            .expect("image metadata arena"),
        );
        #[cfg(target_os = "none")]
        let mut source = {
            let (base, bytes) = kore_rt::entry::thread_context()
                .and_then(|rt| rt.geom()?.window(geom::SURFACES))
                .expect("image heap address reservation");
            // SAFETY: image alone maps this task-local reservation.
            unsafe { kore_svc_mm::heap::MmSource::in_window(base, bytes) }
        };
        #[cfg(not(target_os = "none"))]
        let mut source = kore_svc_mm::heap::MmSource::new();
        let mut heap_seed_cleanup = None;
        let mut root =
            match verbs::carve_owned_lease_window_outcome(e.faces.slot(), 0, e.faces.len()) {
                Ok(root) => Some(root),
                Err(verbs::LeaseCarveError::Uncertain(owner)) => {
                    heap_seed_cleanup = Some(LocalLease::Uncertain(owner));
                    None
                }
                Err(verbs::LeaseCarveError::Refused(_)) => None,
            };
        let _ = source.seed(&mut root);
        if let Some(root) = root {
            heap_seed_cleanup = Some(LocalLease::Known(root));
        }
        let face_heap = alloc::boxed::Box::leak(
            alloc::boxed::Box::try_new_in(kore_rt::Heap::new_in(source), arena)
                .expect("image face heap"),
        );
        Self {
            store: e.store,
            xmint: e.xmint,
            images: alloc::vec::Vec::new(),
            validation: None,
            validation_cleanup: ValidationCleanup::default(),
            next_seq: 0,
            registered: 0,
            refused: 0,
            faces: e.faces,
            channels: e.channels,
            face_heap,
            memory: None,
            maintenance_at: None,
            registration_cleanup_cursor: 0,
            face_cleanup_cursor: 0,
            heap_seed_cleanup,
            face_backings: alloc::vec::Vec::new_in(arena),
            face_slots: alloc::vec::Vec::new_in(arena),
            pending_attach: None,
            pending_detach: None,
            last_bank: None,
            faces_minted: 0,
            faces_retired: 0,
            exported: 0,
        }
    }

    const RETURN_MODE: ReturnMode = ReturnMode::Retained {
        kind: ResourceKind::MemLease,
        minimum_rights: Self::REGISTER_SOURCE_RIGHTS,
    };

    /// Size the allocator-backed catalogue from the measured archive, then
    /// self-register every member. Segment leases obtain their slots from the
    /// growable runtime allocator; this service never derives a private cursor
    /// from the boot block.
    fn start_on_metal(&mut self, rt: &kore_rt::entry::Rt) -> Result<(), StartFailure> {
        // `window_of` owes the empty/wrapping refusals and `is_aligned_to`
        // the page-granularity at both ends; `base() != 0` refuses the
        // unstamped signature, which alignment alone would admit.  This
        // was one of three hand-rolled copies of that predicate list.
        let window = rt
            .geom()
            .and_then(|g| g.window_of(geom::TRANSIENT))
            .filter(|w| w.base() != 0 && w.is_aligned_to(PAGE_BYTES))
            .ok_or(ImageStartFailure::ValidationGeometry)?;
        self.validation = Some(window);
        self.register_archive()?;
        // The face supply's serving view — mapped R/W once for the
        // task's life (slice C).  Fail-closed at start: a vocabulary
        // the service cannot serve refuses readiness, not first mint.
        // The geometry gate is the abi contract (`FACE_SUPPLY_BYTES`):
        // a founder that minted a different span drifted, and carving
        // strides against it would format a phantom ring.
        if self.faces.len() != FACE_SUPPLY_BYTES {
            return Err(ImageStartFailure::FaceSupplyGeometry.into());
        }
        #[cfg(target_os = "none")]
        {
            self.memory = Some(
                kore_svc_mm::RingMm::attach_declared(rt, 0)
                    .ok_or(ImageStartFailure::FaceWindowMap)?,
            );
        }

        Ok(())
    }

    fn drive_host<T: kore_rt::transport::Transport>(
        host: kore_rt::svc::ServeHost<'_, '_, T, Self>,
    ) -> impl core::future::Future<Output = core::convert::Infallible> {
        async move {
            host.run_deadline_driven().await;
        }
    }

    fn next_deadline(&self) -> Option<u64> {
        self.needs_maintenance()
            .then_some(self.maintenance_at.unwrap_or(0))
    }

    async fn drive_deadlines_async(&mut self, _now_ns: u64) -> u16 {
        // Never revoke a registration's parent while its validation view
        // still owns the mapping. A refused unshare keeps both retry owners.
        if self.retry_validation_cleanup() {
            retry_failed_registration(
                &mut self.images,
                &mut self.registration_cleanup_cursor,
                ImageEntry::try_cleanup,
            );
        }
        retry_cleanup_row(
            &mut self.face_slots,
            &mut self.face_cleanup_cursor,
            |face| face.cleanup_pending,
            FaceSlot::try_cleanup,
        );
        self.retry_backing_cleanup();
        if self
            .heap_seed_cleanup
            .as_mut()
            .is_some_and(LocalLease::try_retire)
        {
            self.heap_seed_cleanup = None;
        }
        let source = self.face_heap.source();
        if source.needs_maintenance() {
            if let Some(memory) = self.memory.as_mut() {
                let _ = source.refill(memory).await;
            }
        }
        // Heap recovery uses the original MM request. Registration rollback
        // and validation unshare also remain driven without a new request;
        // stop the timer only when every background owner has settled.
        self.maintenance_at = if self.needs_maintenance() {
            Some(
                kore_rt::executor::clock_ns()
                    .await
                    .saturating_add(1_000_000_000),
            )
        } else {
            None
        };
        0
    }

    fn minted_pair_arena(&self) -> Option<FaceArena> {
        Some(*self.face_slots.allocator())
    }

    async fn serve_async(&mut self, req: ImageRequest, sv: &mut Serve<'_, ImageResponse>) {
        if sv.via() == Self::FACE_OWNER_VIA
            && matches!(
                req,
                ImageRequest::FaceMint { .. } | ImageRequest::FaceFork { .. }
            )
        {
            // The first face establishes its real mm extension path; the
            // extent then backs ordinary ring/metadata allocations.
            if self.face_backings.is_empty() {
                if let Some(memory) = self.memory.as_mut() {
                    if !self
                        .face_heap
                        .source()
                        .reserve_extent(memory, kore_rt::heap::POOL_EXTENT_BYTES)
                        .await
                    {
                        sv.reply(ImageResponse::Refused(ImageRefusal::TableFull));
                        return;
                    }
                }
            }
            if !self.prepare_face() {
                if let Some(memory) = self.memory.as_mut() {
                    let _ = self.face_heap.source().refill(memory).await;
                }
                if !self.prepare_face() {
                    sv.reply(ImageResponse::Refused(ImageRefusal::TableFull));
                    return;
                }
            }
        }
        self.serve(req, sv);
    }

    fn serve(&mut self, req: ImageRequest, sv: &mut Serve<'_, ImageResponse>) {
        // Retry the one validation mapping and then its unadmitted parent.
        // Each owner gets at most one attempt, even on Acquire/Retire.
        let _ = self.retry_validation_cleanup();
        match req {
            ImageRequest::Register { name, slot, source } => {
                if !Self::registrar_via(sv.via()) {
                    self.refused = self.refused.saturating_add(1);
                    sv.reply(ImageResponse::Refused(ImageRefusal::Denied));
                    return;
                }
                if self.validation_cleanup.is_pending() {
                    // No mapping of this new input was attempted. ServeHost
                    // retains its independent source while the previous
                    // mapping and parent finish their ordered cleanup.
                    self.refused = self.refused.saturating_add(1);
                    sv.reply(ImageResponse::Refused(ImageRefusal::Denied));
                    return;
                }
                // `slot` is the founder-published correlation value for this
                // pair, not the private slot holding authority. Registration
                // takes the parent at the first persistent-carve transition;
                // rejected validation instead takes it into ordered cleanup.
                let rep = match sv.returned_input(slot) {
                    Some(mut input) => {
                        let subject = input
                            .capability()
                            .filter(|source| source.is_lease_with(Self::REGISTER_SOURCE_RIGHTS))
                            .and_then(ReturnedCapability::slot);
                        match subject {
                            Some(subject) => match self.map_registration_source(subject) {
                                Ok(win) => {
                                    // SAFETY: the registrar transferred this
                                    // parent before publishing the request;
                                    // `win` owns the exact read-only mapping
                                    // for this synchronous validation step.
                                    let mapped = unsafe { win.as_ref() };
                                    let exact = usize::try_from(source.len.get())
                                        .ok()
                                        .and_then(|len| mapped.get(..len));
                                    let authenticated = source.validate()
                                        && exact.is_some_and(|bytes| {
                                            blake3::hash(bytes).as_bytes() == &source.digest
                                        });
                                    let result = if authenticated {
                                        self.register(
                                            name,
                                            source.profile.get(),
                                            Some(source),
                                            subject,
                                            exact.unwrap_or_default(),
                                            0,
                                            || input.take(),
                                        )
                                    } else {
                                        Err(ImageRefusal::BadImage)
                                    };
                                    self.finish_validation(win, input.take());
                                    match result {
                                        Ok(module) => ImageResponse::Registered { module },
                                        Err(r) => ImageResponse::Refused(r),
                                    }
                                }
                                Err(r) => {
                                    // A failed map can still own its result
                                    // and exact VA. Keep the delivered source
                                    // until that view has confirmed unshare.
                                    self.validation_cleanup.source = input.take();
                                    let _ = self.retry_validation_cleanup();
                                    ImageResponse::Refused(r)
                                }
                            },
                            None => ImageResponse::Refused(ImageRefusal::Denied),
                        }
                    }
                    None => ImageResponse::Refused(ImageRefusal::Denied),
                };
                if matches!(rep, ImageResponse::Refused(_)) {
                    self.refused = self.refused.saturating_add(1);
                }
                sv.reply(rep);
            }
            ImageRequest::Acquire { name } => {
                if name.as_str().is_none() {
                    self.refused = self.refused.saturating_add(1);
                    sv.reply(ImageResponse::Refused(ImageRefusal::BadName));
                    return;
                }
                let generation = if usize::from(sv.via()) >= kore_rt::svc::MAX_SERVE_PAIRS {
                    self.face_dependency_generation(sv.via(), &name)
                } else {
                    // Fixed privileged/control faces retain global lookup for
                    // registration/bootstrap. Ordinary loader faces enter the
                    // closure-restricted arm above.
                    self.lookup(&name).and_then(|slot| {
                        self.images.get(slot).and_then(Option::as_ref).map(|entry| {
                            ImageGeneration {
                                slot,
                                seq: entry.seq,
                            }
                        })
                    })
                };
                self.reply_acquisition(generation.ok_or(ImageRefusal::Unknown), sv);
            }
            ImageRequest::AcquireEntry => {
                let generation = self.face_generation(sv.via()).ok_or(ImageRefusal::Denied);
                self.reply_acquisition(generation, sv);
            }
            ImageRequest::Describe { name } => {
                if !Self::generation_owner_via(sv.via()) {
                    self.refused = self.refused.saturating_add(1);
                    sv.reply(ImageResponse::Refused(ImageRefusal::Denied));
                    return;
                }
                if name.as_str().is_none() {
                    self.refused = self.refused.saturating_add(1);
                    sv.reply(ImageResponse::Refused(ImageRefusal::BadName));
                    return;
                }
                match self
                    .lookup(&name)
                    .and_then(|i| self.images.get(i))
                    .and_then(Option::as_ref)
                {
                    // A data member has no descriptor to describe.
                    Some(entry) if entry.data => {
                        self.refused = self.refused.saturating_add(1);
                        sv.reply(ImageResponse::Refused(ImageRefusal::NotImage));
                    }
                    Some(entry) if entry.grantable() => {
                        sv.reply(ImageResponse::Described {
                            generation: entry.seq,
                            module: entry.descriptor,
                        });
                    }
                    Some(_) => {
                        self.refused = self.refused.saturating_add(1);
                        sv.reply(ImageResponse::Refused(ImageRefusal::Denied));
                    }
                    None => {
                        self.refused = self.refused.saturating_add(1);
                        sv.reply(ImageResponse::Refused(ImageRefusal::Unknown));
                    }
                }
            }
            ImageRequest::Export { name } => {
                if !Self::generation_owner_via(sv.via()) {
                    self.refused = self.refused.saturating_add(1);
                    sv.reply(ImageResponse::Refused(ImageRefusal::Denied));
                    return;
                }
                if name.as_str().is_none() {
                    self.refused = self.refused.saturating_add(1);
                    sv.reply(ImageResponse::Refused(ImageRefusal::BadName));
                    return;
                }
                // Images and data members export alike: bytes are conveyed
                // only from the exact extent the registration validated.
                let window = self
                    .lookup(&name)
                    .and_then(|i| self.images.get(i))
                    .and_then(Option::as_ref)
                    .ok_or(ImageRefusal::Unknown)
                    .and_then(ImageEntry::export_window);
                let (subject, base, rounded, len) = match window {
                    Ok(window) => window,
                    Err(refusal) => {
                        self.refused = self.refused.saturating_add(1);
                        sv.reply(ImageResponse::Refused(refusal));
                        return;
                    }
                };
                // The same span the segment carves came from, at read-only
                // rights and with no executable blessing: `Export` conveys
                // bytes to copy, never authority to run. `LEASE_RIGHTS` is
                // that shape — no `MODIFY`, no exec seal. `DERIVE` is not a
                // loosening and must stay: mapping a held lease *is* a
                // derive (`map_held` mints the `LeasedRegion` leaf), and the
                // kernel's access-only exception is shape-tight to
                // `map_view`, so a `DERIVE`-less export is a window its
                // holder can never read — every copy refuses `Denied` at the
                // map.
                //
                // Carved by the grant itself, straight off the retained
                // window this entry keeps for exactly this purpose. A
                // handler-local carve cannot be the subject here: the
                // grant session is staged after this step returns, so an
                // affine owner living on the match arm's stack revokes
                // its own capability before the transfer names it, and
                // every export refuses `Revoked` at the derive.
                sv.grant(
                    subject,
                    Self::LEASE_RIGHTS,
                    GrantArg::Sub {
                        offset: base,
                        len: rounded,
                    },
                );
                self.exported = self.exported.saturating_add(1);
                sv.reply(ImageResponse::Exported { len });
            }
            ImageRequest::Retire { name } => {
                if !Self::generation_owner_via(sv.via()) {
                    self.refused = self.refused.saturating_add(1);
                    sv.reply(ImageResponse::Refused(ImageRefusal::Denied));
                    return;
                }
                match self.retire(&name) {
                    Ok(()) => sv.reply(ImageResponse::Retired),
                    Err(r) => {
                        self.refused = self.refused.saturating_add(1);
                        sv.reply(ImageResponse::Refused(r));
                    }
                }
            }
            ImageRequest::FaceMint { target } => {
                // This face is the generation-owner surface held only by
                // exec. Image names are inert installer metadata here: exec's
                // privileged, sealable Install gate decides whether a caller
                // may resolve one. Once resolved, a bootstrap constructor is
                // an ordinary immutable generation and launches through the
                // same capability-selected loader path as every other ET_DYN
                // image. Rejecting its spelling here would reintroduce a
                // name-specific authorization rule below that gate.
                let bound = if target.as_str().is_none() {
                    None
                } else {
                    self.lookup(&target)
                        .and_then(|i| self.images.get(i).map(|entry| (i, entry)))
                        .and_then(|(i, entry)| entry.as_ref().map(|entry| (i, entry)))
                        .map(|(slot, entry)| {
                            (
                                ImageGeneration {
                                    slot,
                                    seq: entry.seq,
                                },
                                entry,
                            )
                        })
                        .filter(|(_, entry)| entry.grantable())
                        .map(|(generation, entry)| (generation, entry.descriptor))
                        .filter(|(_, module)| module.kind == kind::EXECUTABLE && module.validate())
                };
                let Some((generation, module)) = bound else {
                    self.refused = self.refused.saturating_add(1);
                    sv.reply(ImageResponse::Refused(ImageRefusal::Unknown));
                    return;
                };
                let Ok(closure) = self.freeze_closure(generation) else {
                    self.refused = self.refused.saturating_add(1);
                    sv.reply(ImageResponse::Refused(ImageRefusal::Denied));
                    return;
                };
                self.reply_face(sv.via(), generation, closure, module, sv);
            }
            ImageRequest::FaceFork { template } => {
                let bound = self
                    .template_binding(template)
                    .and_then(|(generation, closure)| {
                        self.images
                            .get(generation.slot)
                            .and_then(Option::as_ref)
                            .filter(|entry| {
                                generation_matches(generation, generation.slot, entry.seq)
                            })
                            .filter(|entry| entry.grantable())
                            .map(|entry| (generation, closure, entry.descriptor))
                    });
                let Some((generation, closure, module)) = bound else {
                    self.refused = self.refused.saturating_add(1);
                    sv.reply(ImageResponse::Refused(ImageRefusal::Unknown));
                    return;
                };
                self.reply_face(sv.via(), generation, closure, module, sv);
            }
            ImageRequest::FaceBank { face } => match self.face_bank(sv.via(), face) {
                Ok(roots) => {
                    // One fresh recv per channel, derived off the root in
                    // the grant session — never a held leaf moved (a
                    // dropped session would strand its reservation).
                    for root in roots.into_iter().flatten() {
                        sv.grant(root, CHANNEL_ENDPOINT_RIGHTS, GrantArg::ChanRecv);
                    }
                    sv.reply(ImageResponse::Banked);
                }
                Err(r) => {
                    self.refused = self.refused.saturating_add(1);
                    sv.reply(ImageResponse::Refused(r));
                }
            },
            ImageRequest::FaceRetire { face } => match self.face_retire(sv.via(), face) {
                Ok(()) => sv.reply(ImageResponse::Retired),
                Err(r) => {
                    self.refused = self.refused.saturating_add(1);
                    sv.reply(ImageResponse::Refused(r));
                }
            },
            ImageRequest::RetireBound { template } => match self.retire_bound(sv.via(), template) {
                Ok(()) => sv.reply(ImageResponse::Retired),
                Err(r) => {
                    self.refused = self.refused.saturating_add(1);
                    sv.reply(ImageResponse::Refused(r));
                }
            },
            // A tombstone is answered, never dropped: the retired
            // name/ordinal acquisition family stays a visible refusal.
            ImageRequest::Retired => {
                self.refused = self.refused.saturating_add(1);
                sv.reply(ImageResponse::Refused(ImageRefusal::Retired));
            }
        }
    }

    fn grants_dropped(&mut self, response: &mut ImageResponse, emitted: usize) {
        if emitted == 0 {
            return;
        }
        match *response {
            ImageResponse::Acquired { .. } => {
                // `Acquired` promises that the descriptor and its complete
                // lease batch describe one usable image.  If the atomic grant
                // session refused, leaving that success data paired with
                // `grants = 0` would turn a transport failure into a false
                // catalogue hit.
                *response = ImageResponse::Refused(ImageRefusal::Denied);
                self.refused = self.refused.saturating_add(1);
            }
            ImageResponse::Face { face, .. } => {
                // The mint's promise failed whole (I3 unwound the ring-sub
                // carve and every fresh recv with the session): cancel the
                // queued attach and unwind the mint — the face never existed.
                self.pending_attach = None;
                if let Some(k) = face_slot_of(face) {
                    self.unwind_face(k, true);
                }
                self.faces_minted = self.faces_minted.saturating_sub(1);
                *response = ImageResponse::Refused(ImageRefusal::Denied);
                self.refused = self.refused.saturating_add(1);
            }
            ImageResponse::Exported { .. } => {
                // The member-window lease *is* the export, so the codec
                // refuses a grantless one outright. Emitting the success
                // tag anyway would not read as a false promise — it would
                // fail to decode at all, and a client that cannot decode
                // its own reply poisons the pair for good rather than
                // taking a refusal and moving on.
                *response = ImageResponse::Refused(ImageRefusal::Denied);
                self.exported = self.exported.saturating_sub(1);
                self.refused = self.refused.saturating_add(1);
            }
            ImageResponse::Banked => {
                // The staged fresh recvs unwound with the session; reopen
                // the once-per-mint latch so the parent can retry.
                if let Some(k) = self.last_bank.take().and_then(face_slot_of)
                    && let Some(slot) = self.face_slots.get_mut(k).and_then(Option::as_mut)
                {
                    slot.banked = false;
                }
                *response = ImageResponse::Refused(ImageRefusal::Denied);
                self.refused = self.refused.saturating_add(1);
            }
            _ => {}
        }
    }

    fn pending_pair_attach(&mut self) -> Option<MintedPairSpec> {
        self.pending_attach.take()
    }

    fn pair_attached(&mut self, via: u16, attached: bool) {
        if attached {
            return;
        }
        // Structurally unreachable — slot-keyed identities cannot collide
        // and the host bank is sized to the face table — but fail-closed:
        // an unattached face would never serve, so unwind it whole. The
        // parent may already possess the delivered client face. Preserve this
        // retired id until its explicit retry, even when local cleanup closes,
        // so reuse cannot redirect that retry onto a different client.
        let k = (via as usize).wrapping_sub(kore_rt::svc::MAX_SERVE_PAIRS);
        if self.face_slots.get(k).and_then(Option::as_ref).is_some() {
            self.unwind_face(k, false);
            self.faces_minted = self.faces_minted.saturating_sub(1);
        }
    }

    fn pending_pair_detach(&mut self) -> Option<u16> {
        self.pending_detach.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kore_abi::image::{ArchiveEntry, ArchiveHeader};
    use zerocopy::TryFromBytes as _;

    // Catalogue-only fixtures: no native authority is fabricated. The retry
    // closure models a terminal/nonterminal teardown result; native lease
    // retirement and mapping order require the founded-process gate.
    fn cleanup_record(seq: u64, lifecycle: ImageLifecycle) -> ImageEntry {
        ImageEntry {
            name: ImageName::new("same-name").unwrap(),
            seq,
            descriptor: ModuleDescriptor::empty(),
            needed: [None; MAX_DEPENDENCIES],
            leases: [const { None }; MAX_IMAGE_SEGS],
            source: None,
            source_metadata: None,
            lifecycle,
            window: None,
            data: false,
        }
    }

    #[test]
    fn face_cleanup_preserves_a_delivered_identity_after_attach_failure() {
        // Only metadata and empty owner slots: this tests id reuse policy,
        // not native channel/memory retirement.
        let generation = ImageGeneration { slot: 4, seq: 9 };
        let mut face = FaceSlot::preparing(generation, [None; MAX_DEPENDENCIES]);
        assert!(face.retiring);
        assert!(face.cleanup_pending);
        // Construction failure is reclaimable without a client retry.
        assert!(face.unwind(true));
        // Delivery can precede failed host attachment. Even fully closed
        // local owners must leave an occupied identity for that client's
        // eventual FaceRetire, which carries a slot id rather than generation.
        assert!(!face.unwind(false));
        assert!(face.retiring);
        assert!(!face.cleanup_pending);
        let mut faces = [Some(face)];
        retry_cleanup_row(
            &mut faces,
            &mut 0,
            |face| face.cleanup_pending,
            |_| panic!("delivered face reclaimed without its retire request"),
        );
        assert_eq!(faces[0].as_ref().unwrap().generation, generation);
    }

    #[test]
    fn face_cleanup_advances_past_refusals_without_reclaiming_live_clients() {
        let mut live = FaceSlot::preparing(
            ImageGeneration { slot: 4, seq: 9 },
            [None; MAX_DEPENDENCIES],
        );
        live.retiring = false;
        live.cleanup_pending = false;
        let failed =
            |seq| FaceSlot::preparing(ImageGeneration { slot: 4, seq }, [None; MAX_DEPENDENCIES]);
        let mut faces = [Some(live), Some(failed(10)), Some(failed(11))];
        let mut cursor = 0;
        retry_cleanup_row(
            &mut faces,
            &mut cursor,
            |face| face.cleanup_pending,
            |face| {
                assert_eq!(face.generation.seq, 10);
                false
            },
        );
        assert!(faces[1].is_some());
        retry_cleanup_row(
            &mut faces,
            &mut cursor,
            |face| face.cleanup_pending,
            |face| {
                assert_eq!(face.generation.seq, 11);
                true
            },
        );
        assert!(faces[2].is_none());
        retry_cleanup_row(
            &mut faces,
            &mut cursor,
            |face| face.cleanup_pending,
            |face| {
                assert_eq!(face.generation.seq, 10);
                true
            },
        );
        assert!(faces[1].is_none());
        assert_eq!(faces[0].as_ref().unwrap().generation.seq, 9);
    }

    #[test]
    fn registration_cleanup_advances_past_refusals_and_reuses_closed_records() {
        let mut images = [
            Some(cleanup_record(10, ImageLifecycle::Live)),
            Some(cleanup_record(11, ImageLifecycle::RegistrationRollback)),
            Some(cleanup_record(12, ImageLifecycle::RegistrationRollback)),
        ];
        let mut cursor = 0;
        retry_failed_registration(&mut images, &mut cursor, |entry| {
            assert_eq!(entry.seq, 11);
            assert!(!entry.grantable());
            false
        });
        assert!(images[1].is_some());
        assert_eq!(cursor, 2);
        retry_failed_registration(&mut images, &mut cursor, |entry| {
            assert_eq!(entry.seq, 12);
            true
        });
        assert!(images[2].is_none());
        retry_failed_registration(&mut images, &mut cursor, |entry| {
            assert_eq!(entry.seq, 11);
            true
        });
        assert!(images[1].is_none());
        assert_eq!(images[0].as_ref().unwrap().seq, 10);
        assert!(images[0].as_ref().unwrap().grantable());
        // Reusing a freed row must not let the old cursor retire its new
        // generation, even when it has the same display name.
        images[1] = Some(cleanup_record(13, ImageLifecycle::Live));
        retry_failed_registration(&mut images, &mut cursor, |_| {
            panic!("live generation selected for background retirement")
        });
        assert_eq!(images[1].as_ref().unwrap().seq, 13);
    }

    #[test]
    fn registration_cleanup_preserves_explicit_name_retirement_retries() {
        let mut images = [
            Some(cleanup_record(10, ImageLifecycle::Live)),
            Some(cleanup_record(11, ImageLifecycle::Retiring)),
            Some(cleanup_record(12, ImageLifecycle::RegistrationRollback)),
        ];
        let mut cursor = usize::MAX;
        retry_failed_registration(&mut images, &mut cursor, |entry| {
            assert_eq!(entry.seq, 12);
            true
        });
        assert!(images[2].is_none());
        retry_failed_registration(&mut images, &mut cursor, |_| {
            panic!("explicit retirement must remain correlated until retry")
        });
        // A subsequent Retire{name} still resolves generation 11, never the
        // older generation 10 that becomes visible after successful removal.
        let newest = images
            .iter()
            .flatten()
            .max_by_key(|entry| entry.seq)
            .unwrap();
        assert_eq!(newest.seq, 11);
        assert!(!newest.grantable());
    }

    #[test]
    fn registration_cleanup_handles_an_empty_catalogue() {
        let mut cursor = usize::MAX;
        retry_failed_registration(&mut [], &mut cursor, |_| panic!("empty"));
        assert_eq!(cursor, 0);
    }

    /// Pin the committed boot archive's authority footprint, and — by
    /// running `check` over every member — that each member is registerable.
    /// The ordinary process images keep the existing telemetry profile and
    /// ordinary filesystem commands carry the filesystem profile, `cd`
    /// additionally carries the view-binder bit, and Lua carries the bounded
    /// heap+interactive extension. The editor carries filesystem, heap, and
    /// channel-space authority for its file and window seams; `window` alone
    /// carries the inert controller-broker selector. Neither a spelling
    /// convention nor the caller chooses those grants. The shell's
    /// reserved display name is resolved only by privileged init, after which
    /// its immutable generation is capability-selected like every other
    /// executable. `kore-ld-stub` remains registered so a spawner can acquire
    /// the startup linker's bytes like any other image.
    #[test]
    fn committed_archive_profiles_and_geometry_are_authenticated() {
        let bytes = include_bytes!("../../../../res/x86_64/kore-image-archive.bin");
        let (header, _) = ArchiveHeader::try_ref_from_prefix(bytes).expect("archive header");
        let count = header.count.get() as usize;
        let index = bytes.get(ArchiveHeader::WIRE..).expect("archive index");
        let (entries, _) = <[ArchiveEntry]>::try_ref_from_prefix_with_elems(index, count)
            .expect("archive entries");

        let mut slots = 0usize;
        let mut profiled = 0usize;
        let mut data = alloc::vec::Vec::new();
        for entry in entries {
            let name = entry.name.as_str().expect("archive member name");
            // Every member is admitted exactly as `register_archive` admits
            // it; a data member is never handed to the ELF checker.
            match ImageService::admit_member(bytes, entry).expect("admitted archive member") {
                MemberAdmission::Image(checked) => {
                    assert_eq!(entry.kind.get(), kore_abi::image::ARCHIVE_MEMBER_IMAGE);
                    slots += ImageService::owned_slots(&checked).expect("bounded slot count");
                }
                MemberAdmission::Data => {
                    assert_eq!(entry.kind.get(), kore_abi::image::ARCHIVE_MEMBER_DATA);
                    assert_eq!(entry.profile.get(), 0);
                    data.push(name);
                    continue;
                }
            }
            if name == "status" {
                assert_eq!(
                    entry.profile.get(),
                    kore_abi::image::PROFILE_TELEMETRY | kore_abi::image::PROFILE_NETWORK
                );
                profiled += 1;
            } else if name == "window" {
                assert_eq!(entry.profile.get(), kore_abi::image::PROFILE_WINDOW_CONTROL);
                profiled += 1;
            } else if name == "pwctl" {
                assert_eq!(entry.profile.get(), kore_abi::image::PROFILE_SYSTEM_POWER);
                profiled += 1;
            } else if name == "power-ui" {
                assert_eq!(
                    entry.profile.get(),
                    kore_abi::image::PROFILE_HEAP
                        | kore_abi::image::PROFILE_CHANNELS
                        | kore_abi::image::PROFILE_SYSTEM_POWER
                );
                profiled += 1;
            } else if name == "spmd" || name == "spmd-witness" {
                // The SPMD tool and its validate witness provision their
                // regions' doorbell stars from the launcher
                // `ChannelSpace` grant (§3).
                assert_eq!(entry.profile.get(), kore_abi::image::PROFILE_CHANNELS);
                profiled += 1;
            } else if name == "change-directory" {
                assert_eq!(
                    entry.profile.get(),
                    kore_abi::image::PROFILE_FILESYSTEM | kore_abi::image::PROFILE_FS_VIEW_BINDER
                );
                profiled += 1;
            } else if name == "share" {
                // The share executable is the fs-grant-target: it receives
                // the caller's non-invocable grant selector as an auxiliary.
                assert_eq!(
                    entry.profile.get(),
                    kore_abi::image::PROFILE_FILESYSTEM | kore_abi::image::PROFILE_FS_GRANT_TARGET
                );
                profiled += 1;
            } else if name == "aftershock" {
                assert_eq!(
                    entry.profile.get(),
                    kore_abi::image::PROFILE_HEAP
                        | kore_abi::image::PROFILE_CHANNELS
                        | kore_abi::image::PROFILE_ACCELERATOR
                );
                profiled += 1;
            } else if matches!(name, "boot:kore-terminal" | "skylilt") {
                assert_eq!(
                    entry.profile.get(),
                    kore_abi::image::PROFILE_HEAP | kore_abi::image::PROFILE_CHANNELS
                );
                profiled += 1;
            } else if name == "lua" {
                // `channels`: the VM's worker thread parks on a doorbell
                // minted off CHANNEL_SPACE (`res/DAGFS.md` §4). `gpu`: the
                // REPL's `accel` library drives the GPU row's private face,
                // conveyed best-effort at launch (vacant where the Xe row is
                // parked).
                assert_eq!(
                    entry.profile.get(),
                    kore_abi::image::PROFILE_FILESYSTEM
                        | kore_abi::image::PROFILE_HEAP
                        | kore_abi::image::PROFILE_INTERACTIVE
                        | kore_abi::image::PROFILE_CHANNELS
                        | kore_abi::image::PROFILE_GPU
                );
                profiled += 1;
            } else if name == "editor" {
                assert_eq!(
                    entry.profile.get(),
                    kore_abi::image::PROFILE_FILESYSTEM
                        | kore_abi::image::PROFILE_HEAP
                        | kore_abi::image::PROFILE_CHANNELS
                );
                profiled += 1;
            } else if matches!(
                name,
                "list"
                    | "inspect"
                    | "read"
                    | "write"
                    | "append"
                    | "make-directory"
                    | "create"
                    | "move"
                    | "link"
                    | "remove"
                    | "find"
                    | "shares"
                    | "unshare"
                    | "sync"
            ) {
                assert_eq!(entry.profile.get(), kore_abi::image::PROFILE_FILESYSTEM);
                profiled += 1;
            } else if name == "boot:kore-agent-supervisor" {
                assert_eq!(
                    entry.profile.get(),
                    kore_abi::image::PROFILE_HEAP
                        | kore_abi::image::PROFILE_ENGINE
                        | kore_abi::image::PROFILE_CHANNELS
                );
                profiled += 1;
            } else if name == "wasm-witness" {
                // `engine`: the founder lays the ENGINE plan (a 128 KiB link
                // window, a 16 MiB heap) so the native AOT guest's mappings
                // fit beside the stub (`res/WASM_PLATFORM.md` §4).
                assert_eq!(
                    entry.profile.get(),
                    kore_abi::image::PROFILE_HEAP
                        | kore_abi::image::PROFILE_ENGINE
                        | kore_abi::image::PROFILE_INTERACTIVE
                );
                profiled += 1;
            } else if name == "wasm" {
                // The witness's engine plan, plus what a program the caller
                // names needs: the filesystem it is read from (and the guest
                // sees), and the GPU face `kore:accel` hands the guest.
                assert_eq!(
                    entry.profile.get(),
                    kore_abi::image::PROFILE_FILESYSTEM
                        | kore_abi::image::PROFILE_HEAP
                        | kore_abi::image::PROFILE_ENGINE
                        | kore_abi::image::PROFILE_INTERACTIVE
                        | kore_abi::image::PROFILE_GPU
                );
                profiled += 1;
            } else {
                assert_eq!(entry.profile.get(), 0, "unexpected profile on {name}");
            }
        }

        // 42: the prior 36-member archive plus the standalone window-control
        // executable, `pwctl`, `power-ui`, `editor`, and the two native
        // filesystem verbs no POSIX command carried (`append`, `sync`).
        // This literal is the check `cargo test` alone can make that the
        // COMMITTED blob was repacked after ARCHIVE_MEMBERS grew — the
        // Makefile's `user-archive-verify-names` proves the same thing against
        // the recipe, but only under make.
        // 43: plus `lua_demo.so`, the first native Lua module — a library
        // member seeded into `lib/`, never launched.
        // 44: plus `wasm-witness`, the native AOT WebAssembly witness
        // (`res/WASM_PLATFORM.md`).
        // 45: plus `wasm`, the command that runs a WebAssembly program the
        // caller names (`user/apps/utils/wasm`).
        // 47: plus the first two DATA members, the stock `wasm32-wasip2`
        // programs `user/wasm` builds (seeded to `shared/wasm/`).
        assert_eq!(entries.len(), 49);
        assert_eq!(data, ["hello.wasm", "echo.wasm"]);
        // 178: one persistent final lease per admitted carve (data members
        // carve nothing, so they leave this pin alone). `lua_demo.so`
        // now contributes four PT_LOAD carves: rodata, text, the relocated
        // initializer/GOT RELRO page, and writable constructor state. The
        // state must remain writable after the initializer page is sealed.
        // `wasm-witness` carries five: the usual four plus a `KIND_TLS`
        // carve — the platform leaf keeps its mapping table and wasmtime's
        // TLS slots `#[thread_local]` (`res/WASM_PLATFORM.md` §5), the first
        // archive member to do so.  `wasm` carries the same five, for the
        // same reasons: it links the same engine.
        // The optional supervisor is an available package even on ordinary
        // boot; its four PT_LOAD carves grant no running task by themselves.
        assert_eq!(slots, 191);
        // The resident agent runner adds one entry and four PT_LOAD carves.
        // 27: `window` carries its authenticated controller-broker selector,
        // `pwctl` the power face, `power-ui` a window, `aftershock` a window
        // plus its accelerator face, `editor` its filesystem, heap, and
        // channel-space selector as one authenticated profile, and `append`
        // and `sync` the filesystem profile like every other verb;
        // `wasm-witness` carries heap and the engine plan, and `wasm` those
        // plus the filesystem it reads a program from and the GPU face it
        // hands that program: 29.
        assert_eq!(profiled, 30);
    }

    /// A synthetic one-member archive: `payload` at the first page past the
    /// index, padded to a page, described by one entry of `kind`/`profile`.
    fn one_member_archive(
        name: &str,
        payload: &[u8],
        kind: u32,
        profile: u32,
    ) -> alloc::vec::Vec<u8> {
        use zerocopy::IntoBytes as _;
        let page = PAGE_BYTES as usize;
        let offset = (ArchiveHeader::WIRE + ArchiveEntry::WIRE).next_multiple_of(page);
        let header = ArchiveHeader {
            magic: kore_abi::image::ARCHIVE_MAGIC,
            count: 1.into(),
            reserved: 0.into(),
        };
        let entry = ArchiveEntry {
            name: ImageName::new(name).expect("fixture name"),
            offset: (offset as u64).into(),
            len: (payload.len() as u64).into(),
            profile: profile.into(),
            kind: kind.into(),
        };
        let mut bytes = alloc::vec::Vec::new();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(entry.as_bytes());
        bytes.resize(offset, 0);
        bytes.extend_from_slice(payload);
        bytes.resize((offset + payload.len()).next_multiple_of(page), 0);
        bytes
    }

    fn first_entry(bytes: &[u8]) -> &ArchiveEntry {
        let index = bytes.get(ArchiveHeader::WIRE..).expect("archive index");
        let (entries, _) =
            <[ArchiveEntry]>::try_ref_from_prefix_with_elems(index, 1).expect("one entry");
        &entries[0]
    }

    /// A DATA member registers on name and bounds alone: bytes the ELF
    /// checker refuses outright (a `wasm` magic) are admitted as data, and
    /// the very same bytes marked as an image are refused as one.
    #[test]
    fn data_member_registers_without_the_elf_check() {
        use kore_abi::image::{ARCHIVE_MEMBER_DATA, ARCHIVE_MEMBER_IMAGE};
        let wasm = b"\0asm\x01\0\0\0 a stock wasm32-wasip2 program";
        assert!(ImageService::check(wasm).is_err(), "not an ELF image");

        let archive = one_member_archive("hello.wasm", wasm, ARCHIVE_MEMBER_DATA, 0);
        assert!(matches!(
            ImageService::admit_member(&archive, first_entry(&archive)),
            Ok(MemberAdmission::Data)
        ));

        let as_image = one_member_archive("hello.wasm", wasm, ARCHIVE_MEMBER_IMAGE, 0);
        assert_eq!(
            ImageService::admit_member(&as_image, first_entry(&as_image)).err(),
            Some(ImageStartFailure::MemberImage)
        );

        // A launch profile on something that cannot launch is a packing
        // error; an unknown kind fails closed rather than picking a shape.
        let profiled = one_member_archive(
            "hello.wasm",
            wasm,
            ARCHIVE_MEMBER_DATA,
            kore_abi::image::PROFILE_HEAP,
        );
        assert_eq!(
            ImageService::admit_member(&profiled, first_entry(&profiled)).err(),
            Some(ImageStartFailure::MemberKind)
        );
        let unknown = one_member_archive("hello.wasm", wasm, ARCHIVE_MEMBER_DATA + 1, 0);
        assert_eq!(
            ImageService::admit_member(&unknown, first_entry(&unknown)).err(),
            Some(ImageStartFailure::MemberKind)
        );

        // Bounds: an empty member, or a page-rounded span past the store,
        // refuses.
        let empty = one_member_archive("empty.wasm", b"", ARCHIVE_MEMBER_DATA, 0);
        assert_eq!(
            ImageService::admit_member(&empty, first_entry(&empty)).err(),
            Some(ImageStartFailure::MemberBounds)
        );
        let truncated = &archive[..archive.len() - 1];
        assert_eq!(
            ImageService::admit_member(truncated, first_entry(truncated)).err(),
            Some(ImageStartFailure::MemberBounds)
        );
    }

    /// `Acquire` of a data member refuses `NotImage` — never an empty image
    /// with zero grants — while `Export` serves its exact member window.
    #[test]
    fn data_member_refuses_acquire_and_exports_its_window() {
        let store = SlotRef::from_wire(7);
        let base = 3 * PAGE_BYTES;
        let len = PAGE_BYTES + 5;
        let name = ImageName::new("echo.wasm").unwrap();
        let entry = ImageEntry::data(name, 4, store, base, len).expect("page-shaped window");

        assert!(entry.grantable(), "nothing retiring, no lease to die");
        assert_eq!(entry.acquisition().err(), Some(ImageRefusal::NotImage));
        assert_eq!(
            entry.export_window(),
            Ok((store, base, 2 * PAGE_BYTES, len)),
            "the page-rounded span, and the exact registered length"
        );

        // A retiring data record neither exports nor acquires.
        let mut retiring = ImageEntry::data(name, 5, store, base, len).unwrap();
        assert!(retiring.try_retire(), "no lease to refuse the teardown");
        assert_eq!(retiring.export_window(), Err(ImageRefusal::Denied));
        assert_eq!(retiring.acquisition().err(), Some(ImageRefusal::NotImage));

        // The record admits only page-shaped windows.
        assert!(ImageEntry::data(name, 6, store, base + 1, len).is_none());
        assert!(ImageEntry::data(name, 6, store, base, 0).is_none());
        assert!(ImageEntry::data(name, 6, store, u64::MAX & !(PAGE_BYTES - 1), len).is_none());
    }

    #[test]
    fn catalogue_reuses_holes_and_grows_from_runtime_demand() {
        let mut table = alloc::vec::Vec::new();
        ensure_vacant_slots(&mut table, 3).expect("initial measured archive");
        assert_eq!(table.len(), 3);
        table[0] = Some(10u8);
        table[2] = Some(12u8);

        // The existing hole satisfies the next registration without growth.
        ensure_vacant_slots(&mut table, 1).expect("reuse retired hole");
        assert_eq!(table.len(), 3);
        table[1] = Some(11u8);

        // With no holes, demand grows by exactly the missing record count.
        ensure_vacant_slots(&mut table, 2).expect("runtime growth");
        assert_eq!(table.len(), 5);
        assert_eq!(
            table
                .iter()
                .flatten()
                .copied()
                .collect::<alloc::vec::Vec<_>>(),
            [10, 11, 12]
        );

        // Catalogue identities are not accidentally capped by the old u8 slot.
        let wide = ImageGeneration {
            slot: usize::from(u8::MAX) + 17,
            seq: 9,
        };
        assert!(generation_matches(wide, usize::from(u8::MAX) + 17, 9));

        let mut wide_table: alloc::vec::Vec<Option<()>> = alloc::vec::Vec::new();
        ensure_vacant_slots(&mut wide_table, usize::from(u8::MAX) + 17)
            .expect("catalogue growth beyond the former slot width");
        assert_eq!(wide_table.len(), usize::from(u8::MAX) + 17);
    }

    #[test]
    fn image_admission_preserves_every_vacant_face_slot() {
        assert_eq!(
            owned_slot_requirement(7, 3),
            Some(3 * FACE_OWNED_SLOTS + FACE_MINT_SCRATCH_SLOTS + 7)
        );
        assert_eq!(owned_slot_requirement(7, 0), Some(7));
        assert_eq!(owned_slot_requirement(1, usize::MAX), None);
    }

    /// The face-id ⇄ slot ⇄ pair-identity arithmetic is fail-closed and
    /// collision-free: ids are slot indices, out-of-range ids refuse, and
    /// every face via lands strictly above the fixed pairs.
    #[test]
    fn face_correlation_is_fail_closed() {
        // The final fixed pair is exec's resident control face. Derive this
        // identity from the protocol roster: removing or inserting another
        // fixed client must not leave FaceMint gated by a stale literal.
        assert_eq!(ImageService::FACE_OWNER_VIA, 1);
        assert_eq!(
            usize::from(ImageService::FACE_OWNER_VIA) + 1,
            <ImageService as kore_rt::svc::SvcProtocol>::SERVE_PAIRS,
        );
        for k in 0..u16::MAX as usize - kore_rt::svc::MAX_SERVE_PAIRS {
            assert_eq!(face_slot_of(k as u32), Some(k));
            let via = face_via(k);
            assert!(usize::from(via) >= kore_rt::svc::MAX_SERVE_PAIRS);
            assert_eq!(usize::from(via), kore_rt::svc::MAX_SERVE_PAIRS + k);
        }
        assert_eq!(
            face_slot_of((u16::MAX as usize - kore_rt::svc::MAX_SERVE_PAIRS) as u32),
            None
        );
        assert_eq!(face_slot_of(u32::MAX), None);
        assert!(!ImageService::registrar_via(0));
        assert!(ImageService::registrar_via(ImageService::FACE_OWNER_VIA));
        assert!(ImageService::generation_owner_via(
            ImageService::FACE_OWNER_VIA
        ));
        assert!(!ImageService::generation_owner_via(face_via(0)));
    }

    #[test]
    fn executable_face_and_dependencies_do_not_retarget_on_reregistration() {
        let old_executable = ImageGeneration { slot: 3, seq: 7 };
        let old_library = ImageGeneration { slot: 4, seq: 11 };
        let library = ImageName::new("libdemo.so").unwrap();
        let mut frozen = [None; MAX_DEPENDENCIES];
        frozen[0] = Some(DependencyBinding {
            name: library,
            generation: old_library,
        });

        // Re-registering the two display names publishes different immutable
        // generations. It does not mutate either identity already stored in
        // the face: acquisition still authenticates the old slot+sequence,
        // and dependency lookup returns the frozen old binding.
        let new_executable = ImageGeneration { slot: 5, seq: 12 };
        let new_library = ImageGeneration { slot: 6, seq: 13 };
        assert!(generation_matches(old_executable, 3, 7));
        assert!(!generation_matches(
            old_executable,
            new_executable.slot as usize,
            new_executable.seq
        ));
        assert_eq!(frozen[0].unwrap().generation, old_library);
        assert_ne!(frozen[0].unwrap().generation, new_library);
    }
}
