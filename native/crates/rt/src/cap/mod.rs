//! # `cap` — userland capability witnesses
//!
//! **Possession-shaped**: affine Rust values (no `Copy`, no `Clone`)
//! mirroring the D1 holder-stamped records in this task's arena.  The whole
//! derivation chain guarantees one thing — *physical possession of the
//! capability token plus its access permissions **is** access to the
//! resource* — and these types carry that rule into userland code: where
//! Linux maps `(virt, phys)`, kore maps `(virt, token)` — the opaque token
//! **replaces** the physical address, and an operation that consumes
//! authority (`map`, `lease`) takes the token **by value**, so the affine
//! discipline is compile-checked instead of convention-checked.
//!
//! **No table.**  A capability is a **cryptographically sealed record**
//! (the schema layer's token-central inversion: a CMAC over the fields,
//! kernel-keyed), so the kernel keeps *no* authority table — verification
//! is `bind` (MAC + holder stamp + tag liveness), and the only kernel
//! state is the key and the retired-tag bits.  The consequences userland
//! leans on:
//!
//! - **Metadata rides the token**: `token.len()` reads the record's own
//!   sealed field — readable by the holder, unforgeable by anyone (a
//!   signature, deliberately *not* encryption: the fields describe the
//!   holder's own authority, and opacity would just force query verbs).
//! - **Copying bytes buys nothing**: a record is data and *can* be copied,
//!   but a copy carries the same tag and holder stamp — consuming or
//!   transferring the original retires the tag and every duplicate dies
//!   with it (O(1)), and a leaked record fails `bind` under any other
//!   ambient holder (disclosure ≠ transfer, D1).
//! - **Storage ≠ authority**: the arena is slot-indexed record *storage*
//!   (Move A); the kernel binds what it reads there, never trusts position.
//!
//! ## The sharing flow (the model)
//!
//! ```text
//! A: alloc → lease(RW) → LeaseToken  (A is the lessor — the lease is the
//!                                     sole physical authority over its frames)
//! A: share(&lease, RO) → LeasedToken (an attenuated view; ⊆ the lease's rights)
//! A: ── rendezvous, by move ──▶ B    (A must hold B's channel token; delivery
//!                                     re-stamps the holder + retires A's tag)
//! A/B: map their views               (each consumed into a mapped window)
//! ```
//!
//! No memory-manager mediation: possession of the extent plus `DERIVE` is
//! all the authority sharing needs, and the rendezvous layer exists to move
//! caps between tasks for exactly this reason.
//!
//! **The verb seam** (`res/USERLAND.md` §2): production code reaches
//! the kernel verbs only through [`verbs`] — live possession reads and
//! trap arms on the metal target, and **fail-closed typed refusals** for
//! any verb whose kernel arm has not landed (a refusal constructs no
//! record, so nothing downstream can proceed wrong-but-quiet; the QEMU
//! boot gates are where verb behavior is proven).  Session-shaped
//! effects route through [`crate::transport`] instead.

use kore_abi::cap::Rights;
#[cfg(test)]
use kore_abi::mem::DmaClass;
use kore_abi::session::SlotRef;

use record::SealedRec;

/// **The possession seam** — what a plain authority token is parameterized
/// over: the wire resource kind its delivered record must carry, and the
/// record-kind tag its possessed shadow wears.
///
/// The tree grew nine near-identical `{slot, rec}` tokens whose whole
/// difference was these two constants, each with its own hand-written
/// `possess` / `slot` / `rights` — and whose claim verbs were the same two
/// lines over [`verbs::authority_record`], which *already* took the
/// resource kind as a parameter.  The seam was proven in
/// `endpoint_owner!` (which generates exactly this surface for the two
/// channel-end tokens) and then not applied; this is that macro's design
/// as a type, so a new authority kind is three lines of contract instead
/// of a tenth transcription.
///
/// Deliberately **not** a runtime `kind` field on one token type: the
/// per-kind types are what make a wrong-species handle a *bind failure at
/// the possession verb* rather than a puzzling derive refusal later
/// ([`PlatformToken`]'s doc argues this for the device species, and
/// `RecKind`'s docs for the rest).  Distinct `K` ⇒ distinct
/// `Token<K>` ⇒ that argument holds unchanged, with one body behind it.
///
/// A boot **role deliberately does not live on this trait**. Resource species
/// and deployment role are not a bijection: the same [`SupplyToken`] is a
/// `MEM_SUPPLY`, `DMA_POOL`, or `FACE_SUPPLY` depending on which endowment
/// field owns it. [`derive@crate::Bundle`] records that contextual coordinate
/// on the field while this seam keeps only the record facts that are intrinsic
/// to the token species.
#[cfg(target_os = "none")]
pub(crate) trait CapKind {
    /// The `ResourceKind` the delivered [`kore_abi::cap::CapRecord`] must
    /// carry for a claim to authenticate.
    const RESOURCE: kore_abi::cap::ResourceKind;
    /// The record-kind tag the possessed shadow carries.
    const REC: record::RecKind;
}

/// **A plain authority token** — an arena slot plus the sealed shadow of
/// the rights delivered with it, typed by [`CapKind`].
///
/// The authority itself never lives here: the slot is the wire name every
/// later session re-binds by (Move A), and the kernel re-authenticates
/// stamp, holder, and liveness at that bind.  This is a *name plus a
/// readable shadow*, which is why it is affine but not `Drop` — releasing
/// a fixed protocol name is not a thing that happens (see [`OwnedSlot`]
/// for the names that are recycled).
pub struct Token<K> {
    slot: SlotRef,
    rec: SealedRec,
    /// Zero-sized: `K` is an uninhabited marker, so a token can only ever
    /// be built by a claim verb for exactly one kind.
    _kind: core::marker::PhantomData<K>,
}

impl<K> core::fmt::Debug for Token<K> {
    /// Hand-written rather than derived: a derive would demand `K: Debug`
    /// of every marker for no gain — the marker is uninhabited and has
    /// nothing to print.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Token")
            .field("slot", &self.slot)
            .field("rights", &self.rec.rights)
            .finish_non_exhaustive()
    }
}

impl<K> Token<K> {
    /// The arena slot this authority was claimed from — the wire name a
    /// later session names as its derive subject (Move A).
    pub fn slot(&self) -> SlotRef {
        self.slot
    }

    /// The rights the delivered record carried, read off its own `rights`
    /// field at claim.  A *shadow*: the kernel re-checks the real rights
    /// at every use-trap, so this is for the holder's own decisions, never
    /// a substitute for the gate.
    pub fn rights(&self) -> Rights {
        self.rec.rights
    }
}

// Only the claim tail needs the contract — keeping it off the getters
// means no public method carries a crate-private bound.  The `allow` is
// for the impl block itself: `CapKind` is deliberately crate-private
// (its `REC` names the crate-private `RecKind`) while `Token` is public,
// and every member here is `pub(crate)`, so nothing crate-private is
// actually reachable from outside.  Metal-only, like every possession
// constructor it replaced.
#[cfg(target_os = "none")]
#[allow(private_bounds)]
impl<K: CapKind> Token<K> {
    /// Possess a delivered record at `slot` (the claim verbs' tail).
    pub(crate) fn possess(slot: SlotRef, rights: Rights) -> Self {
        Self {
            slot,
            rec: SealedRec::possessed(K::REC, rights),
            _kind: core::marker::PhantomData,
        }
    }
}

/// Declare one authority kind: its uninhabited marker, its [`CapKind`]
/// contract, and the [`Token`] alias that names it.
///
/// The alias is what keeps every consumer's `RegionToken` / `ExecToken` /
/// … spelling — and their distinctness — while one body serves them all.
macro_rules! cap_kinds {
    ($(
        $(#[$doc:meta])*
        $marker:ident => $token:ident, $resource:ident, $rec:ident;
    )*) => {$(
        $(#[$doc])*
        ///
        /// The uninhabited marker; the token is [`Token`]`<Self>`.
        pub enum $marker {}

        #[cfg(target_os = "none")]
        impl CapKind for $marker {
            const RESOURCE: kore_abi::cap::ResourceKind =
                kore_abi::cap::ResourceKind::$resource;
            const REC: record::RecKind = record::RecKind::$rec;
        }

        $(#[$doc])*
        pub type $token = Token<$marker>;
    )*};
}

cap_kinds! {
    /// The lessor's **minting authority** (`role::MEM_REGION`, slice D) — a
    /// claimed revocable grant of one boot DRAM region, off which mm stages
    /// `LeaseReq` derives ([`GrantArg::Lease`](crate::service::GrantArg)) to
    /// mint fresh root supply leases for spawned children.
    ///
    /// Deliberately *thin* beside [`SupplyToken`]: a region grant carries no
    /// client-side geometry because the region's own frame bitmap is the
    /// ledger of record — every minted lease claims and releases against it
    /// in the kernel, and the lessor's only bookkeeping is its quota meter.
    RegionKind => RegionToken, PhysRegion, MemRegion;

    /// The **task-founding authority** — the exec manager's endowment
    /// ([`role::EXEC_AUTHORITY`](kore_abi::task::role::EXEC_AUTHORITY)), claimed
    /// from its seeded arena slot.  Exec names its slot as the authority subject
    /// in a staged founding session; it can found tasks and nothing else.
    ExecKind => ExecToken, ExecAuthority, Exec;

    /// The **address-space allocator** (`VirtMemory`) — the exec manager's
    /// [`role::VIRT_MEMORY`](kore_abi::task::role::VIRT_MEMORY) endowment
    /// (`res/IMAGE.md` §9e): the authority to mint fresh address spaces, the
    /// spawn session's first leg.  A staged `Derive` produces the held fresh
    /// space **without consuming** the allocator: one allocator, N spaces —
    /// the `VirtMemory → AddrSpace` shape.  An authority, not a quantity;
    /// the record is zero-length.
    VirtMemoryKind => VirtMemoryToken, VirtMemory, VirtMemory;

    /// The **X-mint** — authority for an executable lease at initial
    /// allocation (`res/IMAGE.md` §6, the Fuchsia-VMEX analogue). The
    /// consuming writable-to-executable transition instead requires the
    /// full-lifecycle lease it destroys. An authority, not a quantity — the
    /// record is zero-length.
    XMintKind => XMintToken, ExecMint, XMint;

    /// One **PCIe root complex** (`res/DEVICE_SERVICE.md` §10) — a device
    /// broker row's [`role::ROOT_COMPLEX`](kore_abi::task::role::ROOT_COMPLEX)
    /// endowment, claimed once from its seeded arena slot, and *nobody
    /// else's*.
    ///
    /// The [`ImageStoreToken`] shape for hardware, scoped to one root: a
    /// capability over that root's enumerated functions, off which
    /// [`verbs::device_record`] reads a single entry (root-local index) and
    /// [`verbs::claim_stored_device`] hands one out.  Holding it grants no
    /// register, DMA or interrupt authority of its own — those hang off the
    /// [`DeviceToken`] it produces — so the broker arbitrates devices
    /// without being able to drive one.
    ///
    /// An authority, not a quantity: the record is zero-length, and the
    /// root's inventory size is answered by the fallible, authoritative
    /// [`verbs::inventory_len`] report rather than inferred by probing until a
    /// refused record read. Entries are boot-frozen catalog facts; an observation
    /// failure must never become a shorter inventory.
    RootComplexKind => RootComplexToken, RootComplex, RootComplex;

    /// The **platform-device inventory** — broker row 0's
    /// [`role::PLATFORM_BUS`](kore_abi::task::role::PLATFORM_BUS) endowment:
    /// the [`RootComplexToken`] sibling for firmware-described (non-PCI)
    /// devices, split out of the old flat store (#572).  Same observation and
    /// claim verbs, disjoint species: [`verbs::device_record`] reads a
    /// platform entry, [`verbs::claim_stored_device`] hands a
    /// [`PlatformToken`]-claimable device out.
    PlatformBusKind => PlatformBusToken, PlatformBus, PlatformBus;

    /// The machine virtualization facility endowed to the userland
    /// hypervisor service. It can mint coherent guest address spaces but
    /// carries no CPU-local VMX/SVM controller state.
    HypervisorKind => HypervisorToken, Hypervisor, Hypervisor;
}

/// **Seven authorities, seven different resource kinds** — the property that
/// makes the distinct `Token<K>` types load-bearing rather than cosmetic.
///
/// Two kinds sharing a `RESOURCE` would authenticate the *same* delivered
/// record, so a record of one species would bind through the other's
/// possession verb — which is exactly the confusion the per-kind types
/// exist to refuse ([`PlatformToken`]'s doc makes the same argument for
/// the device species).  Proven here rather than trusted, the way
/// `kore-abi`'s report-kind roster is: a seventh kind that reuses a
/// resource is a build error, not a puzzling bind at bring-up.
#[cfg(target_os = "none")]
const _: () = {
    let claims = [
        RegionKind::RESOURCE as u16,
        ExecKind::RESOURCE as u16,
        VirtMemoryKind::RESOURCE as u16,
        XMintKind::RESOURCE as u16,
        RootComplexKind::RESOURCE as u16,
        PlatformBusKind::RESOURCE as u16,
        HypervisorKind::RESOURCE as u16,
    ];
    let mut i = 0;
    while i < claims.len() {
        let mut j = i + 1;
        while j < claims.len() {
            assert!(
                claims[i] != claims[j],
                "two authority kinds claim one resource kind"
            );
            j += 1;
        }
        i += 1;
    }
};

mod inventory_sealed {
    pub trait Sealed {}
}
/// Marker for the two authority kinds that answer the **inventory
/// vocabulary** ([`verbs::device_record`] / [`verbs::inventory_len`] /
/// [`verbs::claim_stored_device`]): a [`RootComplexToken`]'s root-local
/// PCI entries or the [`PlatformBusToken`]'s platform entries.  One wire
/// shape, two authorities — which namespace an index means is decided by
/// the capability presented, never by a wire field.  Sealed: a third
/// subject is a deliberate model change, not an impl.
pub trait InventorySubject: inventory_sealed::Sealed {}

impl inventory_sealed::Sealed for RootComplexKind {}
impl InventorySubject for RootComplexKind {}
impl inventory_sealed::Sealed for PlatformBusKind {}
impl InventorySubject for PlatformBusKind {}

/// Exact authority carried by the userland hypervisor's facility grant.
///
/// The grant must carry every right an owning `GuestSpace` needs: kore's Σ5
/// attenuation rule never lets a child gain a right absent from its parent.
/// `MODIFY` therefore pays for guest mappings and `REVOKE` lets the service
/// close one guest on shutdown.  The founder still retains its ancestor
/// `REVOKE`; retiring that ancestor kills this grant and every guest beneath
/// it, so giving the service its own subtree-close lever does not weaken the
/// founder's control.
pub const HYPERVISOR_RIGHTS: Rights = Rights::ACCESS
    .union(Rights::MODIFY)
    .union(Rights::DERIVE)
    .union(Rights::REVOKE);

/// Exact authority retained on an owning guest space. `MODIFY` gates future
/// map/unmap operations; `DERIVE` cuts non-owning per-vCPU views.
pub const GUEST_SPACE_RIGHTS: Rights = Rights::ACCESS
    .union(Rights::MODIFY)
    .union(Rights::DERIVE)
    .union(Rights::REVOKE);

const _: () = assert!(HYPERVISOR_RIGHTS.contains(GUEST_SPACE_RIGHTS));

/// Read/execute guest mappings carry an observable, explicitly revocable
/// public name but no write authority.
pub const GUEST_REGION_RO_RIGHTS: Rights = Rights::ACCESS.union(Rights::REVOKE);

/// A guest-writable RAM mapping adds `MODIFY`; physical lifetime still belongs
/// to the RAM lease and guest space, not to this handle's no-op teardown.
pub const GUEST_REGION_RW_RIGHTS: Rights = GUEST_REGION_RO_RIGHTS.union(Rights::MODIFY);

/// Lifecycle authority returned when userland founds a guest vCPU.
///
/// `ACCESS` waits/observes, `MODIFY` starts the dormant vCPU, `REVOKE` tears
/// it down, and `DERIVE` permits an owner to attenuate a supervision handle.
pub const GUEST_VCPU_HANDLE_RIGHTS: Rights = kore_abi::guest::GUEST_VCPU_HANDLE_RIGHTS;

/// Affine ownership of one runtime-minted guest address space and its
/// dynamic arena name. Dropping the token attempts an ordinary capability
/// revoke, whose kernel teardown drains guest windows and closes stage 2; an
/// unresolved close keeps the arena name quarantined.
#[must_use]
#[derive(Debug)]
pub struct GuestSpaceToken {
    owned: OwnedSlot,
    rec: SealedRec,
}

impl GuestSpaceToken {
    #[cfg(any(test, target_os = "none"))]
    fn new(mut owned: OwnedSlot) -> Self {
        owned.activate();
        Self {
            owned,
            rec: SealedRec::possessed(record::RecKind::GuestSpace, GUEST_SPACE_RIGHTS),
        }
    }

    /// The arena name used by later guest-memory and vCPU-view sessions.
    pub fn slot(&self) -> SlotRef {
        self.owned.slot()
    }

    /// Exact authority retained by the owning service.
    pub fn rights(&self) -> Rights {
        self.rec.rights
    }

    /// Explicitly close the guest space. A retryable refusal returns the
    /// same affine owner so the caller cannot lose teardown authority.
    pub fn try_retire(mut self) -> Result<RetireOutcome, Self> {
        let outcome = self.owned.retire_inner();
        if outcome.is_closed() {
            Ok(outcome)
        } else {
            Err(self)
        }
    }
}

/// Affine public name for one lease-backed GPA window. Retiring this token
/// closes the name; the lease/guest-space pair owns the actual mapping and
/// removes it during explicit unshare or either parent's teardown.
#[must_use]
#[derive(Debug)]
pub struct GuestRegionToken {
    owned: OwnedSlot,
    rec: SealedRec,
}

impl GuestRegionToken {
    #[cfg(any(test, target_os = "none"))]
    fn new(mut owned: OwnedSlot, rights: Rights) -> Self {
        owned.activate();
        Self {
            owned,
            rec: SealedRec::possessed(record::RecKind::GuestRegion, rights),
        }
    }

    pub fn slot(&self) -> SlotRef {
        self.owned.slot()
    }

    pub fn rights(&self) -> Rights {
        self.rec.rights
    }

    pub fn try_retire(mut self) -> Result<RetireOutcome, Self> {
        let outcome = self.owned.retire_inner();
        if outcome.is_closed() {
            Ok(outcome)
        } else {
            Err(self)
        }
    }
}

/// A supply-region authority — the memory-manager's endowment
/// ([`role::MEM_SUPPLY`](kore_abi::task::role::MEM_SUPPLY)), claimed from its
/// seeded arena slot.  Grants carve windows out of it; never mapped itself.
#[derive(Debug)]
pub struct SupplyToken {
    slot: SlotRef,
    /// Absolute base of the supply's backing.  Metal (Phase-A checkpoint
    /// E): it caches
    /// the base the kernel **reported** at claim (the delivered
    /// [`CapRecord`](kore_abi::cap::CapRecord) carries no offset/len — geometry
    /// has no metal home — so `claim_supply` recovers it from the founder-seeded
    /// backing and reports it into the R/W scratch).
    base: u64,
    /// Total byte length of the supply — the geometry mm's grant cursor needs.
    /// Cached from the same source as [`base`](Self::base): the kernel's
    /// claim-supply report.
    len: u64,
    rec: SealedRec,
    /// Runtime-owned landing for a supply received through a service grant.
    /// Founder endowments use `None`; a retained reply keeps the arena name
    /// owned for exactly as long as this affine supply token is live.
    _returned: Option<ReturnedCapability>,
}

/// The shared root a lessor holds after `verbs::lease` consumed its
/// extent — the **sole physical authority** over those frames.
/// Views are attenuated off it (`verbs::share`); revoking it kills every
/// view in one cascade.
///
/// Carries its **arena slot**, like [`LeasedToken`] and [`TaskToken`]:
/// a lease is a subject later sessions name (a share into a child's
/// space, a sub-window carve), and a token that cannot say which slot it
/// came from forces the holder to track that out of band — which is the
/// bookkeeping the token type exists to remove.
#[derive(Debug)]
pub struct LeaseToken {
    /// The arena slot this lease was possessed from.
    slot: SlotRef,
    rec: SealedRec,
}

#[cfg(target_os = "none")]
impl LeaseToken {
    /// Possess a delivered `MemLease` at `slot` — [`verbs::take_lease`].
    pub(crate) fn possess(slot: SlotRef, size: u64, rights: Rights) -> Self {
        Self {
            slot,
            rec: SealedRec::possessed(record::RecKind::Extent, rights),
        }
        .with_len(size)
    }

    /// A no-op setter kept private so `possess` reads as one expression;
    /// the sealed record owns the length.
    fn with_len(mut self, size: u64) -> Self {
        self.rec.len = size;
        self
    }
}

impl LeaseToken {
    /// The arena slot this lease is named by — what a later session
    /// cites as the share/carve subject.
    pub fn slot(&self) -> SlotRef {
        self.slot
    }

    /// Replace this writable lease with an executable-only
    /// successor. A pre-commit refusal returns the untouched source token;
    /// [`FinalizeExecError::Consumed`] reports a fail-closed transition whose
    /// source is already gone and produced no executable authority.
    pub fn finalize_exec(self) -> Result<ExecutableCode, FinalizeExecError<Self>> {
        verbs::finalize_exec(self)
    }
}

/// A leased *view* — the sendable artifact of sharing.  Carries
/// `ACCESS (+ MODIFY for RW)` and **never `DERIVE`** (views
/// are access handles; the sharing pivot stays with the lease-holder).
/// Move it to a peer over rendezvous (`verbs::deliver` re-stamps the
/// holder); `verbs::map_view` consumes it into a mapped window.
#[derive(Debug)]
pub struct LeasedToken {
    /// The arena slot this view was possessed from — the wire name
    /// `verbs::map_view` re-binds by (Move A).  The metal
    /// [`take_view`](verbs::take_view) reads it off the R/O `SLOTS_VA`
    /// alias.
    slot: SlotRef,
    /// Client-known geometry — the known founder-layout window size.  The
    /// delivered [`CapRecord`](kore_abi::cap::CapRecord) has **no**
    /// offset/len (Phase-A checkpoint C), so the token caches the window
    /// the client maps; `map_view` re-binds authority by
    /// [`slot`](Self::slot) regardless.
    size: u64,
    rec: SealedRec,
}

/// Exact authority carried by a launcher-owned rendezvous mint grant.
///
/// This is deliberately one contract rather than a minimum: the launcher is
/// given a cuttable child of the machine `ChannelSpace`, and no ambient or
/// accidentally stronger variant is accepted by [`verbs::claim_channel_space`].
pub const CHANNEL_SPACE_RIGHTS: Rights = Rights::ACCESS.union(Rights::DERIVE).union(Rights::REVOKE);

/// Exact authority retained on a freshly minted channel root.
pub const CHANNEL_ROOT_RIGHTS: Rights = Rights::ACCESS.union(Rights::DERIVE).union(Rights::REVOKE);

/// Exact authority carried by either freshly minted endpoint.
///
/// `DERIVE` admits a later capability transfer and `REVOKE` lets an endpoint
/// owner close its own dynamically allocated arena name. `MODIFY` has no
/// rendezvous meaning and is therefore absent.
pub const CHANNEL_ENDPOINT_RIGHTS: Rights =
    Rights::ACCESS.union(Rights::DERIVE).union(Rights::REVOKE);

/// A launcher's bounded, revocable child of the machine rendezvous allocator.
///
/// Claiming this fixed founder endowment does not allocate another arena name.
/// Every [`ChannelPair`] minted through it is rooted below this grant, so
/// revoking the grant cuts the complete launcher-owned channel forest by T6.
#[must_use]
#[derive(Debug)]
pub struct ChannelSpaceToken {
    slot: SlotRef,
    rec: SealedRec,
    /// Runtime-owned landing when this token arrived through an ordinary
    /// parent/child configuration channel. Founder endowments use `None`.
    _returned: Option<ReturnedCapability>,
}

/// Affine ownership of one runtime-minted channel root and its dynamic arena
/// reservation.
///
/// The root is the T6 lifetime pivot for both endpoints. It therefore does
/// **not** revoke implicitly when the Rust value is dropped: a launcher may
/// already have conveyed an endpoint to another task, and an incidental local
/// drop must not kill that remote authority. Call [`try_retire`](Self::try_retire)
/// explicitly after all users are quiescent. Dropping without doing so keeps
/// the arena name quarantined for the task lifetime.
#[must_use = "a channel root must be explicitly retired after every conveyed endpoint is quiescent"]
#[derive(Debug)]
pub struct ChannelRoot {
    owned: core::mem::ManuallyDrop<OwnedSlot>,
    rec: SealedRec,
}

/// Affine ownership of a runtime-minted `ChanSend` endpoint.
#[must_use]
#[derive(Debug)]
pub struct ChanSendToken {
    owned: OwnedSlot,
    rec: SealedRec,
}

/// Affine possession of a delivered `ChanSend` in one of the founder's fixed
/// holding slots.
///
/// Unlike [`ChanSendToken`], this witness owns no dynamic-arena reservation:
/// the founder chose the slot and task teardown owns its final cleanup.  It is
/// nevertheless affine, and the service return protocol consumes it only
/// after the kernel proves that the endpoint moved and scrubbed the source
/// record.  This is the channel counterpart of [`LeaseToken`].
#[must_use]
#[derive(Debug)]
pub struct HeldChanSendToken {
    slot: SlotRef,
    rec: SealedRec,
}

/// Affine possession of a delivered `ChanRecv` in one of the founder's fixed
/// holding slots.
///
/// The runtime-owned landing remains a separate affine value (for example a
/// [`crate::chan::NoticeRx`]'s landing); this token is the held receive
/// authority that may donate it. Like [`HeldChanSendToken`], it owns no
/// dynamic arena reservation and task teardown owns final cleanup.
#[must_use]
#[derive(Debug)]
pub struct HeldChanRecvToken {
    slot: SlotRef,
    rec: SealedRec,
}

/// Affine ownership of a runtime-minted `ChanRecv` endpoint.
#[must_use]
#[derive(Debug)]
pub struct ChanRecvToken {
    owned: OwnedSlot,
    rec: SealedRec,
}

/// The all-or-nothing result of one `ChannelSpace → Channel →
/// (ChanSend, ChanRecv)` session.
///
/// No staged `SlotRef` escapes the mint. Each committed result is returned
/// with the unique [`OwnedSlot`] reservation that governs its arena name.
#[must_use]
#[derive(Debug)]
pub struct ChannelPair {
    /// Root retained by the launcher as the pair's revocation pivot.
    pub root: ChannelRoot,
    /// Send endpoint, independently movable through the typed return protocol.
    pub send: ChanSendToken,
    /// Receive endpoint, independently movable through the typed return protocol.
    pub recv: ChanRecvToken,
}

/// Why an atomic channel-pair mint did not yield three typed owners.
#[must_use]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelMintError {
    /// An endpoint-rights request is empty or exceeds the channel endpoint
    /// vocabulary.
    InvalidRights,
    /// The bounded per-thread dynamic arena has fewer than three free names.
    NoSlots,
    /// The fixed three-invocation session could not be staged.
    Build(crate::session::BuildError),
    /// The trap transport could not issue the session.
    Transport,
    /// The kernel refused a well-formed invocation and rolled the session back.
    Refused(kore_abi::session::Status),
    /// The completion could not prove either commit or rollback. Every
    /// reserved name remains quarantined rather than risking aliasing.
    MalformedCompletion,
    /// This build has no live channel-mint transport.
    Unavailable,
}

/// A **mapped-window accessor** — the safe byte-port over a mapped view,
/// for consumers that must stay `forbid(unsafe)` (a virtio transport's
/// queue payloads).  Every access is a bounds-checked **raw-pointer byte
/// copy**; no `&[u8]`/`&mut [u8]` over the window bytes ever materializes,
/// which is the load-bearing choice: **sibling views of one lease alias
/// the same frames** (that *is* the sharing), and materialized references
/// over aliasing shared memory would be UB where raw copies are merely
/// racy — and races are excluded by the ownership protocol above the
/// window (a virtqueue chain owns its
/// regions).  `Clone` deliberately: authority was spent when the view was
/// consumed into the mapping; cloning the *accessor* is holding the same
/// mapping, exactly like two copies of a mapped pointer.
#[derive(Clone, Debug)]
pub struct Window {
    base: *mut u8,
    len: u64,
}

impl Window {
    /// Byte length of the window.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Whether the window is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copy `target.len()` bytes out at `offset`; `false` when the range
    /// falls outside the window.
    pub fn read_at(&self, offset: u64, target: &mut [u8]) -> bool {
        let Some(end) = offset.checked_add(target.len() as u64) else {
            return false;
        };
        if end > self.len {
            return false;
        }
        // SAFETY: bounds-checked against the mapped length; raw byte copy,
        // no reference over the window is created.  Fast path: memcpy when
        // the ranges are provably disjoint (the overwhelming case — the
        // caller's buffer is a local); memmove only when the caller's
        // buffer is itself an overlapping arena window.
        unsafe {
            let src = self.base.add(offset as usize) as *const u8;
            copy_maybe_overlapping(src, target.as_mut_ptr(), target.len());
        }
        true
    }

    /// Copy `source` into the window at `offset`; `false` when the range
    /// falls outside the window.  Takes `&self` deliberately: the window
    /// is *shared* memory — exclusivity is the owning protocol's property
    /// (queue chain ownership), not a Rust aliasing property.
    pub fn write_at(&self, offset: u64, source: &[u8]) -> bool {
        let Some(end) = offset.checked_add(source.len() as u64) else {
            return false;
        };
        if end > self.len {
            return false;
        }
        // SAFETY: as in `read_at`.
        unsafe {
            copy_maybe_overlapping(
                source.as_ptr(),
                self.base.add(offset as usize),
                source.len(),
            );
        }
        true
    }

    /// Borrow `len` bytes at `offset` as a shared slice — the **zero-copy
    /// fast path** behind
    /// [`ReadableBytes::as_contiguous`](kore-abi-adjacent transport ports):
    /// a decoder walks the frame in place instead of copying it out.
    ///
    /// # Safety
    ///
    /// The caller asserts that **no write reaches `[offset, offset+len)`
    /// for the returned borrow's lifetime** — through this window, any
    /// clone, or any sibling view.  The virtqueue chain-ownership protocol
    /// provides exactly this (a held chain's regions belong to the device
    /// until completion, and publish-time validation refuses intra-chain
    /// overlap), which is why the one caller lives beside that protocol.
    pub unsafe fn bytes(&self, offset: u64, len: u64) -> Option<&[u8]> {
        let end = offset.checked_add(len)?;
        if end > self.len {
            return None;
        }
        // SAFETY: bounds-checked; immutability for the lifetime is the
        // caller's contract above.
        Some(unsafe { core::slice::from_raw_parts(self.base.add(offset as usize), len as usize) })
    }
}

/// memcpy when disjoint, memmove when not — the disjointness of the two
/// ranges is one compare, and `copy_nonoverlapping` unlocks the wide
/// vectorized path.
///
/// # Safety
/// Both pointers valid for `len` bytes (read / write respectively).
unsafe fn copy_maybe_overlapping(src: *const u8, dst: *mut u8, len: usize) {
    let (s, d) = (src as usize, dst as usize);
    // SAFETY: caller contract; the branch picks the overlap-legal intrinsic.
    unsafe {
        if s + len <= d || d + len <= s {
            core::ptr::copy_nonoverlapping(src, dst, len);
        } else {
            core::ptr::copy(src, dst, len);
        }
    }
}

/// An attenuated **task handle** — the parent's `Child` witness (`ACCESS`
/// = wait, `MODIFY` = start, `REVOKE` = kill), or the task's own `SELF`
/// handle (`MODIFY` = exit). Affine like every witness; lifecycle operations
/// must explicitly transfer or stage this authority.
#[derive(Debug)]
pub struct TaskToken {
    /// The arena slot this handle was possessed from. The metal
    /// [`take_handle`](verbs::take_handle) reads it off the R/O `SLOTS_VA`
    /// alias.
    slot: SlotRef,
    /// The founded task's id — the delivered cap's `cid.resource()`
    /// (Phase-A checkpoint C: a
    /// task record's "geometry" is its identity, not an address, and the
    /// resource id *is* carried on the delivered
    /// [`CapRecord`](kore_abi::cap::CapRecord)).
    id: u64,
    rec: SealedRec,
    /// Runtime-owned landing for a handle returned by the exec service.
    /// Founder endowments use `None`; a spawn reply keeps this owner for the
    /// token's whole lifetime so dropping the token revokes the child handle
    /// before the dynamic arena name can be reused.
    _returned: Option<ReturnedCapability>,
}

macro_rules! witness_getters {
    ($t:ty) => {
        impl $t {
            /// Byte length — read from the token's own **sealed** field:
            /// holder-readable metadata, unforgeable (any tamper fails
            /// `bind`).
            pub fn len(&self) -> u64 {
                self.rec.len
            }
            /// Empty is unrepresentable (grants refuse zero size), but the
            /// convention pair keeps clippy honest.
            pub fn is_empty(&self) -> bool {
                self.rec.len == 0
            }
            /// The rights this token carries (sealed, holder-readable).
            pub fn rights(&self) -> Rights {
                self.rec.rights
            }
        }
    };
}
witness_getters!(LeaseToken);

impl LeasedToken {
    /// Byte length — the **client-known geometry** the token caches
    /// ([`size`](Self::size), the known founder-layout window), *not* an
    /// on-record field: the delivered
    /// [`CapRecord`](kore_abi::cap::CapRecord) has no offset/len, so
    /// `map_view` re-binds authority by [`slot`](Self::slot).
    pub fn len(&self) -> u64 {
        self.size
    }
    /// Empty is unrepresentable (a view spans at least one page), but the
    /// convention pair keeps clippy honest.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
    /// The rights this view carries — read off the delivered record's own
    /// `rights` field.
    pub fn rights(&self) -> Rights {
        self.rec.rights
    }
    /// The arena slot this view was possessed from — the wire name
    /// `verbs::map_view` re-binds by (Move A).
    pub fn slot(&self) -> SlotRef {
        self.slot
    }
}

impl SupplyToken {
    /// The arena slot the claimed record sits in — the wire name a staged
    /// grant's `subject` uses (Move A: sessions name caps by slot).
    pub fn slot(&self) -> SlotRef {
        self.slot
    }

    /// Absolute base of the supply's backing — a lessor-side face
    /// (`kore-svc-mm` converts returned absolute windows to supply-relative
    /// books).  Reads the token's **cached geometry** ([`base`](Self::base)),
    /// not an on-record field: the delivered
    /// [`CapRecord`](kore_abi::cap::CapRecord) has no offset (Phase-A checkpoint
    /// E), so [`claim_supply`](verbs::claim_supply) caches the kernel's
    /// report.
    pub fn base(&self) -> u64 {
        self.base
    }

    /// Total byte length of the supply — the **client-known geometry** the token
    /// caches ([`len`](Self::len), the claim-supply report), *not* an
    /// on-record field.  mm's grant cursor bounds
    /// every carve against it.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Empty is unrepresentable (a supply spans at least one page), but the
    /// convention pair keeps clippy honest.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The rights this supply carries — the claim-supply report's possessed
    /// shadow of the record's own `rights` field.
    pub fn rights(&self) -> Rights {
        self.rec.rights
    }

    /// Recover the original retained delivery before publishing any mappings.
    /// A founder-held supply cannot be converted into a returned owner.
    pub fn into_returned_capability(mut self) -> Result<ReturnedCapability, Self> {
        match self._returned.take() {
            Some(owner) => Ok(owner),
            None => Err(self),
        }
    }

    /// Whether this supply arrived through a retained service grant.
    ///
    /// A returned supply already is the caller's bounded sublease, so a
    /// device host may map that root directly instead of attempting the
    /// forbidden sublease-of-sublease shape.
    pub fn is_returned(&self) -> bool {
        self._returned.is_some()
    }

    #[cfg(target_os = "none")]
    pub(crate) fn into_returned_owner(mut self) -> Result<OwnedSlot, Self> {
        let Some(returned) = self._returned.take() else {
            return Err(self);
        };
        let ReturnedCapability { owned } = returned;
        Ok(owned)
    }
}

impl TaskToken {
    /// The task this handle stands over — the **client-known** id the token
    /// caches ([`id`](Self::id)): the delivered cap's
    /// `cid.resource()`.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The rights this handle carries (`ACCESS` wait / `MODIFY` start-or-exit /
    /// `REVOKE` kill) — read off the delivered record's own `rights` field.
    pub fn rights(&self) -> Rights {
        self.rec.rights
    }

    /// The arena slot this handle was possessed from.
    pub fn slot(&self) -> SlotRef {
        self.slot
    }

    /// Explicitly make this completely founded child runnable.
    ///
    /// The token is borrowed, not consumed: a retryable scheduler admission
    /// retains the same authenticated handle, while an idempotent duplicate
    /// over an already active/parked child returns [`TaskStartOutcome::Active`].
    /// The handle must carry [`Rights::MODIFY`]; an observer-only handle fails
    /// closed as [`TaskStartOutcome::Refused`] rather than trusting this
    /// userland rights shadow.
    pub fn start(&self) -> TaskStartOutcome {
        verbs::start_task(self.slot)
    }

    /// **Block until this task exits**, without surrendering the child
    /// handle.
    ///
    /// This is the same kernel join path used by [`crate::thread::join`]: a
    /// report slot is reserved for the duration of the park and the task handle
    /// is named as the join subject. The closed result distinguishes an
    /// observed exit from a reap and from every failure to perform or decode
    /// the join; this affine token remains owned by the caller on every path.
    ///
    /// A kernel [`Status::WouldBlock`](kore_abi::session::Status::WouldBlock)
    /// or [`Status::Conflict`](kore_abi::session::Status::Conflict) is returned
    /// as [`TaskWaitError::Retryable`], not retrapped in a tight loop.
    /// `WouldBlock` can mean the task's bounded joiner set is full or that this
    /// caller could not establish its park; in either case no waiter is
    /// standing, so the caller must yield or back off before trying again.
    ///
    /// The handle must carry [`Rights::ACCESS`]. The kernel re-binds the
    /// sealed record at the session boundary, so a handle without that right
    /// fails closed rather than trusting this userland shadow.
    pub fn wait(&self) -> Result<TaskWaitOutcome, TaskWaitError> {
        crate::thread::wait_task_handle(self.slot)
    }

    /// **Probe this task's lifecycle without blocking.**
    ///
    /// Like [`wait`](Self::wait), this method owns its temporary report slot
    /// from reservation through decoding and release.  Callers therefore see
    /// neither a raw [`SlotRef`] nor an unchecked arena record.  A live task is
    /// [`TaskPollOutcome::Running`]; exit and reap stay distinct terminal
    /// facts.  The token is borrowed, so polling never consumes or duplicates
    /// the parent's affine lifecycle authority.
    ///
    /// The handle must carry [`Rights::ACCESS`].  The kernel re-binds the
    /// sealed record at the session boundary rather than trusting this
    /// userland rights shadow.
    pub fn poll(&self) -> Result<TaskPollOutcome, TaskWaitError> {
        crate::thread::poll_task_handle(self.slot)
    }

    /// **Kill/reap this task** by revoking the held child handle.
    ///
    /// This consumes `self` only after teardown is proved closed. `Ok`
    /// distinguishes a revoke committed by this call from an already-dead
    /// lineage. A retryable refusal returns the *same* affine token in `Err`:
    /// the caller therefore retains both the only authority witness and, for
    /// an exec-returned handle, its quarantined dynamic-slot reservation.
    /// Losing that value would make a later retry impossible while the slot
    /// correctly remained unavailable for reuse.
    ///
    /// Founder-stamped handles have no dynamic reservation and revoke their
    /// fixed arena name directly. Exec-returned handles retire through their
    /// embedded [`ReturnedCapability`] owner so a closed handle and its arena
    /// allocation are released as one lifecycle transition.
    pub fn kill(mut self) -> Result<RetireOutcome, Self> {
        let outcome = match self._returned.as_mut() {
            Some(returned) => returned.try_retire_outcome(),
            None => verbs::retire_slot_outcome(self.slot),
        };
        match outcome {
            RetireOutcome::Retired | RetireOutcome::AlreadyDead => Ok(outcome),
            RetireOutcome::Retryable => Err(self),
        }
    }
}

// ── Metal possession constructors (Phase-A checkpoint C) ─────────────────
//
// The metal read verbs ([`verbs::take_view`]/`take_lease`/`take_handle`)
// read a delivered [`CapRecord`](kore_abi::cap::CapRecord) off the R/O
// cap-slot alias at `SLOTS_VA` and build a
// possession token from it.  The record
// stays in the (read-only) slot; the token caches the reading `slot` (the
// downstream map/lease/deliver trap re-binds by it, re-authenticating CMAC
// + holder + tag liveness), the **client-known geometry**, and a userland
// shadow of the delivered rights ([`SealedRec::possessed`]).  Metal-only —
// the sole callers are the `verbs.rs` metal reads (dead on host builds).
#[cfg(target_os = "none")]
impl LeasedToken {
    /// Possess a delivered `LeasedRegion` view at `slot` — `size` is the
    /// known founder-layout window (the map trap re-binds by slot).
    pub(crate) fn possess(slot: SlotRef, size: u64, rights: Rights) -> Self {
        Self {
            slot,
            size,
            rec: SealedRec::possessed(record::RecKind::Leased, rights),
        }
    }
}

#[cfg(target_os = "none")]
impl TaskToken {
    /// Possess a delivered task handle at `slot` — `id` is the delivered
    /// cap's `cid.resource()`; the tag stays live (the client keeps its own
    /// authority); later sessions re-bind it by slot.
    pub(crate) fn possess(slot: SlotRef, id: u64, rights: Rights) -> Self {
        Self {
            slot,
            id,
            rec: SealedRec::possessed(record::RecKind::TaskHandle, rights),
            _returned: None,
        }
    }

    /// Possess an exec-returned task handle without separating the live arena
    /// name from the runtime reservation that owns that name.
    pub(crate) fn possess_returned(
        slot: SlotRef,
        id: u64,
        rights: Rights,
        returned: ReturnedCapability,
    ) -> Self {
        Self {
            slot,
            id,
            rec: SealedRec::possessed(record::RecKind::TaskHandle, rights),
            _returned: Some(returned),
        }
    }

    /// Possess a task handle published directly into a runtime-reserved
    /// result slot owned by this task.
    pub(crate) fn possess_owned(mut owned: OwnedSlot, id: u64, rights: Rights) -> Self {
        owned.activate();
        let slot = owned.slot();
        Self {
            slot,
            id,
            rec: SealedRec::possessed(record::RecKind::TaskHandle, rights),
            _returned: Some(ReturnedCapability { owned }),
        }
    }
}

#[cfg(target_os = "none")]
impl SupplyToken {
    /// Possess a claimed supply at `slot` from the kernel's claim-supply
    /// **report** (Phase-A checkpoint E) — `base`/`len` are the geometry the
    /// kernel recovered from the founder-seeded backing and wrote into the R/W
    /// scratch (the `CapRecord` carries none), `rights` the reported authority.
    /// The slot stays bindable, so mm grants from [`slot`](Self::slot)
    /// repeatedly; the trap re-authenticates the real record by slot each time.
    pub(crate) fn possess(slot: SlotRef, base: u64, len: u64, rights: Rights) -> Self {
        Self {
            slot,
            base,
            len,
            rec: SealedRec::possessed(record::RecKind::Supply, rights),
            _returned: None,
        }
    }

    /// Possess a service-granted supply while retaining the dynamically
    /// rotated landing that owns its live arena record.
    pub(crate) fn possess_returned(
        slot: SlotRef,
        base: u64,
        len: u64,
        rights: Rights,
        returned: ReturnedCapability,
    ) -> Self {
        Self {
            slot,
            base,
            len,
            rec: SealedRec::possessed(record::RecKind::Supply, rights),
            _returned: Some(returned),
        }
    }
}

/// A **host-test supply** — geometry only, possessing nothing.
///
/// Off the userland target every capability verb is an honest refusal, so
/// a service's host tests could construct no endowment at all and its
/// store logic went untested (the zram backend's whole codec and
/// accounting model, for one).  This closes that: the token carries the
/// `base`/`len` a bound service reads and a record that no verb will
/// accept, so a test can exercise the *logic* while any accidental
/// capability use still refuses.
///
/// Gated on `std`, which only a `[dev-dependencies]` build turns on — it
/// cannot reach a metal image.
#[cfg(all(feature = "std", not(target_os = "none")))]
impl SupplyToken {
    /// A geometry-only supply for host tests.
    pub fn for_test(base: u64, len: u64) -> Self {
        Self {
            slot: SlotRef::from_wire(0),
            base,
            len,
            rec: SealedRec::inert(record::RecKind::Supply, Rights::ACCESS),
            _returned: None,
        }
    }
}

#[cfg(target_os = "none")]
impl ImageStoreToken {
    /// Possess the founder-delivered image-store lease after the kernel has
    /// authenticated the held lease and reported its backing geometry.
    pub(crate) fn possess(slot: SlotRef, base: u64, len: u64, rights: Rights) -> Self {
        Self {
            slot,
            rec: SealedRec {
                kind: record::RecKind::ImageStore,
                rights,
                holder: 0,
                tag: 0,
                offset: base,
                len,
                class: kore_abi::mem::DmaClass::None,
                mac: 0,
            },
        }
    }
}

/// A **device capability** — the driver task's
/// [`role::DEVICE`](kore_abi::task::role::DEVICE) endowment, claimed once from
/// its seeded arena slot (the stand-in for a `PciDevice` cap).  It is
/// the authority to map the device's BARs (`verbs::map_bar`) and derive its
/// interrupt route (`verbs::establish_interrupt`) — the driver-process intake #101
/// names.
// `rec` is the sealed witness payload — the metal verb for this token
// reads it once its kernel arm lands; no reader exists yet.
#[allow(dead_code)]
#[derive(Debug)]
pub struct DeviceToken {
    slot: SlotRef,
    rec: SealedRec,
    /// Runtime-owned landing for a device received from the broker. Founder
    /// endowments use `None`; broker grants keep this owner for the token's
    /// whole lifetime so another reply can never overwrite live authority.
    _returned: Option<ReturnedCapability>,
}

impl DeviceToken {
    /// The arena slot this device capability was claimed from.
    pub fn slot(&self) -> SlotRef {
        self.slot
    }

    /// Rights carried by the founder-delivered device grant.
    pub fn rights(&self) -> Rights {
        self.rec.rights
    }

    #[cfg(target_os = "none")]
    pub(crate) fn possess(slot: SlotRef, rights: Rights) -> Self {
        Self {
            slot,
            rec: SealedRec::possessed(record::RecKind::Device, rights),
            _returned: None,
        }
    }

    #[cfg(target_os = "none")]
    pub(crate) fn possess_returned(
        slot: SlotRef,
        rights: Rights,
        returned: ReturnedCapability,
    ) -> Self {
        Self {
            slot,
            rec: SealedRec::possessed(record::RecKind::Device, rights),
            _returned: Some(returned),
        }
    }
}

/// Affine ownership of one driver's live device-stage-1 MMU generation.
///
/// The token is minted only after the driver has programmed and proved its
/// private translation regime. Retiring it closes the peer-sponsor mint and
/// synchronously sweeps every link this generation admitted before the arena
/// slot can be reused.
#[derive(Debug)]
pub struct DeviceMmuToken {
    owned: OwnedSlot,
}

impl DeviceMmuToken {
    #[cfg(target_os = "none")]
    pub(crate) fn new(mut owned: OwnedSlot) -> Self {
        owned.activate();
        Self { owned }
    }

    /// Arena slot named by a sponsored peer request.
    pub fn slot(&self) -> SlotRef {
        self.owned.slot()
    }

    /// Explicitly close this stage-1 generation and all peer links it
    /// sponsored. A refusal retains the token for task-lifetime containment.
    pub fn try_retire(self) -> Result<RetireOutcome, Self> {
        let mut this = self;
        let outcome = this.owned.retire_inner();
        if outcome.is_closed() {
            Ok(outcome)
        } else {
            Err(this)
        }
    }
}

/// Affine registration of one exact DMA-window lease in the memory fabric.
/// This is lifecycle authority, not a DMA address or a routing permission.
#[derive(Debug)]
pub struct FabricRegistrationToken {
    owned: OwnedSlot,
}

impl FabricRegistrationToken {
    #[cfg(target_os = "none")]
    pub(crate) fn new(mut owned: OwnedSlot) -> Self {
        owned.activate();
        Self { owned }
    }

    /// Read planning facts while retaining this registration's authority.
    pub fn describe(
        &self,
    ) -> Result<kore_abi::session::FabricDomainReport, verbs::FabricRegistrationFault> {
        verbs::describe_fabric(self.owned.slot())
    }

    /// Open a mapping route while retaining the source registration.
    pub fn open_route(
        &self,
        request: &kore_abi::session::FabricRouteReq,
    ) -> Result<FabricRouteToken, verbs::FabricRouteFault> {
        verbs::open_fabric_route(self.owned.slot(), request)
    }

    /// Explicitly retire registration before its owning DMA window.
    /// A refusal returns the token for containment instead of discarding it.
    pub fn try_retire(self) -> Result<RetireOutcome, Self> {
        let mut this = self;
        let outcome = this.owned.retire_inner();
        if outcome.is_closed() {
            Ok(outcome)
        } else {
            Err(this)
        }
    }
}

/// Affine authority for an executable memory-fabric route. Explicit close
/// preserves this owner on busy, quarantined, or uncertain completion.
#[derive(Debug)]
pub struct FabricRouteToken {
    owned: OwnedSlot,
}

impl FabricRouteToken {
    #[cfg(target_os = "none")]
    pub(crate) fn new(mut owned: OwnedSlot) -> Self {
        owned.activate();
        Self { owned }
    }

    /// The possessed route may parent the existing CPU and IOVA mapping verbs.
    pub fn slot(&self) -> SlotRef {
        self.owned.slot()
    }

    /// Lend a bounded range with its own mapping/close lifecycle. The parent
    /// remains busy until the child retires. This does not pin allocator runs;
    /// a buffer provider must retain its allocation for the entire loan.
    pub fn scope(
        &self,
        request: &kore_abi::session::FabricRouteScopeReq,
    ) -> Result<Self, verbs::FabricRouteFault> {
        verbs::scope_fabric_route(self.owned.slot(), request)
    }

    /// Observe immutable mapping geometry before attempting publication.
    pub fn describe(
        &self,
    ) -> Result<kore_abi::session::FabricRouteInfo, verbs::FabricRegistrationFault> {
        verbs::describe_fabric_route(self.slot())
    }

    /// Transfer this exact route authority through a retained service grant.
    pub fn into_returned_capability(self) -> ReturnedCapability {
        ReturnedCapability { owned: self.owned }
    }

    pub(crate) fn release_after_proven_move(mut self) {
        self.owned.release_after_proven_move();
    }

    /// Close admission without consuming authority or touching live children.
    pub fn try_close(&self) -> Result<(), verbs::FabricRouteFault> {
        verbs::close_fabric_route(self.owned.slot())
    }

    /// Retire the route only after every physical child has retired. On any
    /// refusal the complete token is returned so the owner can contain or retry.
    pub fn try_retire(self) -> Result<RetireOutcome, Self> {
        let mut this = self;
        if this.try_close().is_err() {
            return Err(this);
        }
        let outcome = this.owned.retire_inner();
        if outcome.is_closed() {
            Ok(outcome)
        } else {
            Err(this)
        }
    }
}

/// Direction granted to a peer DMA engine over a declared BAR aperture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeerDmaAccess {
    Read,
    ReadWrite,
    PostedWrite,
}

impl PeerDmaAccess {
    #[cfg(target_os = "none")]
    pub(crate) const fn wire(self) -> u8 {
        match self {
            Self::Read => kore_abi::session::PeerDeviceShareReq::ACCESS_READ,
            Self::ReadWrite => kore_abi::session::PeerDeviceShareReq::ACCESS_READ_WRITE,
            Self::PostedWrite => kore_abi::session::PeerDeviceShareReq::ACCESS_WRITE,
        }
    }

    #[cfg(target_os = "none")]
    pub(crate) const fn rights(self) -> Rights {
        let base = Rights::ACCESS.union(Rights::REVOKE);
        match self {
            Self::Read => base,
            Self::ReadWrite | Self::PostedWrite => base.union(Rights::MODIFY),
        }
    }
}

/// A live peer aperture view.
///
/// The owned arena slot is the public `LeasedIova` child beneath the BAR. This
/// value also borrows all three lifecycle owners, so safe code cannot retire
/// the BAR, the sponsor's stage-1 generation, or the requester device while it
/// presents the link as live. Drop performs the exact IOVA-unshare verb; a
/// refusal quarantines the slot and owner lifetimes still provide the
/// task/device-wide containment fallback.
#[derive(Debug)]
pub struct PeerWindow<'a> {
    /// Plain `OwnedSlot` drop is insufficient: `LeasedIova::teardown` is a
    /// no-op, so this owner must exact-unshare through its BAR parent.
    owned: core::mem::ManuallyDrop<OwnedSlot>,
    parent: SlotRef,
    iova: u64,
    len: u64,
    _owners: core::marker::PhantomData<(&'a BarWindow, &'a DeviceMmuToken, &'a DeviceToken)>,
}

impl PeerWindow<'_> {
    #[cfg(target_os = "none")]
    pub(crate) fn new(mut owned: OwnedSlot, parent: SlotRef, iova: u64, len: u64) -> Self {
        owned.activate();
        Self {
            owned: core::mem::ManuallyDrop::new(owned),
            parent,
            iova,
            len,
            _owners: core::marker::PhantomData,
        }
    }

    pub const fn iova(&self) -> u64 {
        self.iova
    }

    pub const fn len(&self) -> u64 {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Retire the exact peer-IOMMU child and its physical window. Retryable
    /// refusals keep the arena name closed rather than silently weakening to a
    /// plain revoke that would leave the IOMMU leaf installed.
    pub fn try_retire_outcome(&mut self) -> RetireOutcome {
        let parent = self.parent;
        self.owned
            .retire_via(|view| crate::cap::verbs::retire_owned_iova(view, parent))
    }

    pub fn is_live(&self) -> bool {
        self.owned.is_live()
    }
}

impl Drop for PeerWindow<'_> {
    fn drop(&mut self) {
        if self.try_retire_outcome().is_closed() {
            // SAFETY: exact IOVA unshare settled this owner to Released (or it
            // was already dead), so the underlying drop is now a no-op.
            unsafe { core::mem::ManuallyDrop::drop(&mut self.owned) };
        }
    }
}

/// A **platform-device capability** (#417) — a driver row's
/// [`role::DEVICE`](kore_abi::task::role::DEVICE) endowment when the row
/// selected a *firmware-described non-PCI* device by its ACPI `_HID`.
///
/// The [`DeviceToken`] twin for the platform world, and deliberately a
/// separate type rather than a flag on that one: the two species support
/// disjoint derives — this one grants declared I/O ports
/// ([`verbs::claim_ports`](crate::cap::verbs::claim_ports)) or its sole
/// firmware MMIO window, plus a firmware line
/// ([`verbs::establish_line`](crate::cap::verbs::establish_line)), where a
/// PCI function grants named BAR windows and MSI(-X) routes. A driver
/// handed the wrong species refuses at the possession verb, which is a
/// bind failure at bring-up rather than a puzzling derive refusal later.
///
/// It accepts no BAR number and binds no IOMMU domain. Any MMIO geometry is
/// fixed in the kernel catalog row and cannot be supplied by the holder.
// `rec` is the sealed witness payload — read once the metal verb for this
// token lands, as with every sibling token here.
#[allow(dead_code)]
#[derive(Debug)]
pub struct PlatformToken {
    slot: SlotRef,
    rec: SealedRec,
    /// Runtime-owned broker-grant landing; see [`DeviceToken::_returned`].
    _returned: Option<ReturnedCapability>,
}

impl PlatformToken {
    /// The arena slot this platform-device capability was claimed from.
    pub fn slot(&self) -> SlotRef {
        self.slot
    }

    /// Rights carried by the founder-delivered grant.
    pub fn rights(&self) -> Rights {
        self.rec.rights
    }

    #[cfg(target_os = "none")]
    pub(crate) fn possess(slot: SlotRef, rights: Rights) -> Self {
        Self {
            slot,
            rec: SealedRec::possessed(record::RecKind::PlatformDevice, rights),
            _returned: None,
        }
    }

    #[cfg(target_os = "none")]
    pub(crate) fn possess_returned(
        slot: SlotRef,
        rights: Rights,
        returned: ReturnedCapability,
    ) -> Self {
        Self {
            slot,
            rec: SealedRec::possessed(record::RecKind::PlatformDevice, rights),
            _returned: Some(returned),
        }
    }
}

mod i2c_subject_sealed {
    pub trait Sealed {}
}

/// The two controller capability species that may observe correlated HID
/// children. Sealed so downstream code cannot substitute a raw slot or a
/// different capability kind for an authenticated controller token.
pub trait I2cHidSubject: i2c_subject_sealed::Sealed {
    fn i2c_subject_slot(&self) -> SlotRef;
}

impl i2c_subject_sealed::Sealed for DeviceToken {}
impl I2cHidSubject for DeviceToken {
    fn i2c_subject_slot(&self) -> SlotRef {
        self.slot()
    }
}

impl i2c_subject_sealed::Sealed for PlatformToken {}
impl I2cHidSubject for PlatformToken {
    fn i2c_subject_slot(&self) -> SlotRef {
        self.slot()
    }
}

/// Result of retiring an affine capability's owned arena name.
///
/// This outcome separates a revoke committed by this call from a bind that
/// proved the named lineage was already dead.  Both close the *local* owned
/// name: a dead-lineage record cannot authorize anything, and a later owned
/// result may overwrite its arena slot.  Every other refusal keeps the
/// allocator reservation and affine retry handle intact.
#[must_use]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetireOutcome {
    /// The capability was retired, or a reservation with no published
    /// capability was released.
    Retired,
    /// The kernel proved that this capability or an ancestor was already
    /// revoked. No revoke committed, but the inert local name was released.
    AlreadyDead,
    /// Teardown did not establish that all owned authority is dead. Every
    /// still-unproven name and allocator reservation remains quarantined for a
    /// later attempt; a compound owner may already have released a member
    /// whose dead lineage was identified unambiguously.
    Retryable,
}

/// **Kernel-minted proof that one device is quiet at the fabric** (#809 §L)
/// — the terminal counterpart of the DMA activation a driver requested at
/// bring-up.  The kernel masked the function's interrupts, closed and read
/// back Bus Master Enable, bound its requester to Abort, and quarantined its
/// DMA epoch; this value is the report of that rendezvous, carried by the
/// class host into its teardown outcome so the shutdown path can say *what
/// the fabric saw* rather than what a driver asserted.
///
/// **Userland cannot mint one.**  The only constructor decodes the
/// `QuiesceReport` the kernel wrote into the caller's result slot after a
/// `MODIFY`-gated `DERIVE_KIND_DEVICE_QUIESCE` committed — the R/O slots
/// alias is kernel-written and the family gate refuses anything that is
/// not a deliberate report — so, like [`RetireOutcome::Retired`], holding
/// this is the same trust as the kernel having said so.  Affine and
/// `#[must_use]`: a witness dropped on the floor is a shutdown that
/// cannot be audited.
#[must_use = "a fabric quiesce witness must be carried to the terminal path"]
#[derive(Debug, PartialEq, Eq)]
pub struct Quiesced {
    rid: u32,
    verdict: FabricVerdict,
    flags: u8,
    _private: (),
}

/// Which mechanism a [`Quiesced`] witness rests on.  Either value means no
/// DMA reaches memory; a shutdown witness line should say which.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FabricVerdict {
    /// Bus Master Enable read back clear, and the requester sits on Abort
    /// wherever it has an IOMMU path at all.  The device itself stopped.
    Proven,
    /// The function refused to drop Bus Master Enable, but its requester is
    /// bound to Abort: the IOMMU is the enforcement point and the DMA epoch
    /// stays quarantined.  Containment, reported truthfully.
    Contained,
}

impl Quiesced {
    /// Decode the kernel's report.  `None` for a report whose verdict byte
    /// this runtime does not know — a newer kernel's vocabulary fails
    /// closed rather than reading as a proof.  Metal-only (plus the host
    /// test of the decode): off the metal no slot is ever written.
    #[cfg(any(test, target_os = "none"))]
    pub(crate) fn minted(report: kore_abi::session::QuiesceReport) -> Option<Self> {
        use kore_abi::session::{QUIESCE_VERDICT_CONTAINED, QUIESCE_VERDICT_PROVEN};
        let verdict = match report.verdict {
            QUIESCE_VERDICT_PROVEN => FabricVerdict::Proven,
            QUIESCE_VERDICT_CONTAINED => FabricVerdict::Contained,
            _ => return None,
        };
        Some(Self {
            rid: report.rid,
            verdict,
            flags: report.flags,
            _private: (),
        })
    }

    /// A witness for host tests of the code that *carries* one.  Nothing
    /// off the metal can prove a device quiet, so this exists only under
    /// `cfg(test)`; production has exactly one constructor, and it reads
    /// the kernel's slot.
    #[cfg(test)]
    pub fn for_test(rid: u32, verdict: FabricVerdict) -> Self {
        Self {
            rid,
            verdict,
            flags: 0,
            _private: (),
        }
    }

    /// The quiesced device's catalog resource id.
    pub fn rid(&self) -> u32 {
        self.rid
    }

    /// What the proof rests on.
    pub fn verdict(&self) -> FabricVerdict {
        self.verdict
    }

    /// The requester's IOMMU context was rebound to Abort (false only for a
    /// function with no IOMMU binding, whose proof rests on BME alone).
    pub fn abort_bound(&self) -> bool {
        self.flags & kore_abi::session::QUIESCE_FLAG_ABORT_BOUND != 0
    }

    /// No peer window this device sponsored stands (#725): its device-MMU
    /// facet was closed and drained before the DMA half, or none was ever
    /// minted.  False when a facet still stood that the rendezvous could
    /// not close.
    pub fn peers_drained(&self) -> bool {
        self.flags & kore_abi::session::QUIESCE_FLAG_PEERS_DRAINED != 0
    }
}

/// Why [`verbs::quiesce_device`](crate::cap::verbs::quiesce_device) could
/// not hand back a [`Quiesced`] witness.  Every arm leaves the device epoch
/// quarantined fail-closed on the kernel side; the class host records the
/// refusal and containment runs regardless.
#[must_use]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuiesceFault {
    /// This task has no report slot or slots alias stamped — a row founded
    /// without the geometry the effect needs.
    Slot,
    /// The session could not be staged (the builder refused).
    Build,
    /// The trap itself failed to commit.
    Transport,
    /// The kernel refused: the rendezvous could not prove the function quiet
    /// (BME refused with no IOMMU behind it, or the Abort rebind failed), or
    /// the subject is not a PCI class device.  Carries the session status.
    Refused(u8),
    /// The kernel answered `Ok` but the result slot held no decodable
    /// report — a raced overwrite or a vocabulary this runtime lacks.
    Report,
    /// Off the metal, or a platform device: no rendezvous exists to run.
    Unenforced,
}

/// Why a writable lease did not become an executable successor.
///
/// Finalization has one abort door: a refusal before commit returns the exact
/// affine source.  Once the kernel has retired that source and begun clearing
/// aliases, rollback is impossible; a later containment or publication
/// failure is therefore reported as [`Consumed`](Self::Consumed), never as a
/// writable lease that no longer exists.
#[must_use]
#[derive(Debug)]
pub enum FinalizeExecError<T> {
    /// The session did not commit; the writable source is unchanged.
    Refused(T),
    /// The source was consumed, but no executable successor was published.
    Consumed,
}

/// A bounded write into a [`WritableCode`] mapping did not fit.
#[must_use]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodeWriteError {
    /// `offset..offset + len` lies outside the code mapping.
    OutOfBounds {
        offset: usize,
        len: usize,
        capacity: usize,
    },
}

/// Why a writable lease could not enter the code-initialization state.
#[must_use]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WritableCodeMapErrorKind {
    /// The affine lease no longer names live authority.
    SourceUnavailable,
    /// The kernel would not report the lease's complete geometry.
    GeometryUnavailable,
    /// The caller's virtual window cannot hold the complete lease mapping.
    WindowTooSmall { needed: u64, available: u64 },
    /// The kernel refused the R/W share or the runtime could not own its view.
    Refused,
}

/// A failed transition into [`WritableCode`] that returns the exact source.
#[must_use]
#[derive(Debug)]
pub struct WritableCodeMapError {
    lease: StashedLease,
    kind: WritableCodeMapErrorKind,
}

impl WritableCodeMapError {
    /// The fail-closed reason; inspecting it does not consume the retry owner.
    pub const fn kind(&self) -> WritableCodeMapErrorKind {
        self.kind
    }

    /// Recover the unchanged writable lease for retry or teardown.
    pub fn into_lease(self) -> StashedLease {
        self.lease
    }
}

/// A code-emission closure failed after possibly changing some bytes.
///
/// Initialization is deliberately not transactional: the exact writable
/// state comes back so the caller may retry, overwrite, or retire it.
#[must_use]
#[derive(Debug)]
pub struct CodeInitializeError<E> {
    code: WritableCode,
    error: E,
}

impl<E> CodeInitializeError<E> {
    /// Recover the still-writable code owner and the closure's own error.
    pub fn into_parts(self) -> (WritableCode, E) {
        (self.code, self.error)
    }
}

/// Why [`WritableCode::finalize`] did not produce executable authority.
#[must_use]
#[derive(Debug)]
pub enum FinalizeCodeError {
    /// Nothing committed; the complete writable state is returned unchanged.
    Refused(WritableCode),
    /// The writable lease was consumed, but no executable successor exists.
    Consumed,
}

/// Failure of [`StashedLease::initialize_executable_at`].
#[must_use]
#[derive(Debug)]
pub enum InitializeExecutableError<E> {
    /// The writable mapping could not be established; the source lease is in
    /// the contained error.
    Map(WritableCodeMapError),
    /// The emitter failed; the mapping and writable lease remain owned.
    Initialize(CodeInitializeError<E>),
    /// Finalization refused before commit; the writable state remains owned.
    FinalizeRefused(WritableCode),
    /// Finalization consumed the writable authority without publishing an
    /// executable successor.
    Consumed,
}

/// Why an [`ExecutableCode`] lease could not be mapped for instruction fetch.
#[must_use]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutableMapError {
    /// The affine executable owner no longer names live authority.
    SourceUnavailable,
    /// The kernel would not report the executable lease's complete geometry.
    GeometryUnavailable,
    /// The caller's virtual window cannot hold the complete lease mapping.
    WindowTooSmall { needed: u64, available: u64 },
    /// The kernel refused the R/X share or the runtime could not own its view.
    Refused,
}

/// Typed completion of [`TaskToken::start`].
#[must_use]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskStartOutcome {
    /// The child is runnable or has already crossed that lifecycle edge.
    /// Duplicate START calls intentionally collapse here.
    Active,
    /// The first scheduler publication did not complete.  The same borrowed
    /// token may be retried; no capability authority was consumed.
    Retryable,
    /// The subject exited, was killed, or its generational run-table identity
    /// is gone.  Retrying START cannot revive it.
    Terminal,
    /// The kernel refused the authenticated request (notably `Denied` for an
    /// ACCESS-only handle), or returned an impossible completion shape.
    Refused(kore_abi::session::Status),
    /// No live trap transport exists, or the fixed one-invocation descriptor
    /// could not be staged.  Host builds report this rather than simulating
    /// capability authority.
    Unavailable,
}

/// A terminal fact returned by the blocking [`TaskToken::wait`].
///
/// Exit and reap are deliberately separate: an exited task retains its
/// recorded status until the parent explicitly revokes its handle, while a
/// reaped task has no exit report left to observe.
#[must_use]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskWaitOutcome {
    /// The task exited normally through its `SELF` handle with this status.
    Exited(i32),
    /// The observed task-handle lineage was revoked before an exit report was
    /// available.
    Reaped,
}

/// A nonblocking lifecycle fact returned by [`TaskToken::poll`].
#[must_use]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskPollOutcome {
    /// The task has not exited and its handle remains live.
    Running,
    /// The task exited normally through its `SELF` handle with this status.
    Exited(i32),
    /// The observed task-handle lineage was revoked before an exit report was
    /// available.
    Reaped,
}

/// Why a task lifecycle observation could not be completed.
#[must_use]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskWaitError {
    /// The caller's bounded capability arena had no vacant report slot.
    NoSlots,
    /// No blocking waiter was installed, or the session lost an OCC race. The
    /// kernel uses this outcome when a task's bounded joiner set is full and
    /// when the caller could not establish its park. The borrowed [`TaskToken`]
    /// remains live; yield or back off before retrying so bounded exhaustion or
    /// contention cannot become a syscall spin.
    Retryable,
    /// The session could not be staged or issued through the live transport.
    Transport,
    /// The kernel refused the authenticated join request.
    Refused(kore_abi::session::Status),
    /// A one-invocation join completion named another failing index or an
    /// unknown status discriminant.
    MalformedCompletion,
    /// The kernel reported `Exited`, but the result slot did not contain the
    /// exact bind-inert [`kore_abi::session::ExitReport`] encoding.
    MalformedReport,
}

impl RetireOutcome {
    /// Whether the affine owner no longer retains a live local name.
    #[inline]
    pub const fn is_closed(self) -> bool {
        matches!(self, Self::Retired | Self::AlreadyDead)
    }
}

impl ChannelSpaceToken {
    /// Arena name of the held mint grant. Sessions re-bind this slot at every
    /// use; the token's metadata is only a typed userland witness.
    pub fn slot(&self) -> SlotRef {
        self.slot
    }

    /// The exact rights contract authenticated when this endowment was
    /// claimed.
    pub fn rights(&self) -> Rights {
        self.rec.rights
    }

    /// Explicitly cut this launcher's complete channel forest.
    ///
    /// Every channel minted through this self-kind grant is a T6 descendant,
    /// so a successful revoke closes roots and local or conveyed endpoints in
    /// one bounded operation. A retryable refusal returns the same affine
    /// token; there is deliberately no hidden revoke in `Drop`.
    pub fn try_retire(self) -> Result<RetireOutcome, Self> {
        let outcome = crate::cap::verbs::retire_slot_outcome(self.slot);
        if outcome.is_closed() {
            Ok(outcome)
        } else {
            Err(self)
        }
    }

    #[cfg(target_os = "none")]
    pub(crate) fn possess(slot: SlotRef, rights: Rights) -> Self {
        debug_assert_eq!(rights, CHANNEL_SPACE_RIGHTS);
        Self {
            slot,
            rec: SealedRec::possessed(record::RecKind::ChannelSpace, rights),
            _returned: None,
        }
    }

    #[cfg(target_os = "none")]
    fn possess_returned(slot: SlotRef, rights: Rights, returned: ReturnedCapability) -> Self {
        debug_assert_eq!(rights, CHANNEL_SPACE_RIGHTS);
        Self {
            slot,
            rec: SealedRec::possessed(record::RecKind::ChannelSpace, rights),
            _returned: Some(returned),
        }
    }
}

/// One dynamically reserved arena slot whose published capability is owned by
/// an affine userland token. The slot is reusable only after a successful
/// kernel revoke or proof that its lineage is already dead; an unresolved
/// teardown deliberately quarantines it.
#[derive(Debug)]
pub(crate) struct OwnedSlot {
    slot: SlotRef,
    state: OwnedSlotState,
    /// Slot indices and allocator bits belong to the current thread's arena
    /// (`%fs`-anchored `ThreadBlock`). This marker keeps every containing
    /// affine token on its originating thread.
    _thread_bound: core::marker::PhantomData<*mut ()>,
}

#[cfg_attr(not(target_os = "none"), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OwnedSlotState {
    /// The allocator bit is held, but no capability was published.
    Reserved,
    /// A staged session may publish this result and then move/scrub it before
    /// returning. Cancellation at this boundary cannot prove either local
    /// occupancy or vacancy, so Drop must retain the allocator reservation.
    InFlight,
    /// The transport returned an impossible/unknown completion after a staged
    /// transfer began. No future operation may reopen or name this slot.
    Indeterminate,
    /// A receive post may still name this vacant slot. It cannot be released
    /// until endpoint closure or post consumption proves no late delivery can
    /// publish authority into it.
    Armed,
    /// A live capability occupies the slot.
    Live,
    /// The record was retired or proved dead and the allocator bit reopened.
    Released,
}

impl OwnedSlot {
    #[cfg(target_os = "none")]
    fn reserve() -> Option<Self> {
        // Growth belongs to the allocator, not to per-service forecasts of
        // every capability a future request might retain. The extension verb
        // uses boot-owned scratch/report slots and cannot re-enter this path.
        let slot = crate::entry::geometry::acquire_cap_slot().or_else(|| {
            crate::cap::verbs::extend_arena()?;
            crate::entry::geometry::acquire_cap_slot()
        })?;
        Some(Self {
            slot,
            state: OwnedSlotState::Reserved,
            _thread_bound: core::marker::PhantomData,
        })
    }

    #[cfg(not(target_os = "none"))]
    fn reserve() -> Option<Self> {
        None
    }

    fn slot(&self) -> SlotRef {
        self.slot
    }

    fn activate(&mut self) {
        debug_assert!(matches!(
            self.state,
            OwnedSlotState::Reserved | OwnedSlotState::Armed
        ));
        self.state = OwnedSlotState::Live;
    }

    fn arm(&mut self) {
        debug_assert!(matches!(
            self.state,
            OwnedSlotState::Reserved | OwnedSlotState::Armed
        ));
        self.state = OwnedSlotState::Armed;
    }

    fn disarm(&mut self) {
        debug_assert_eq!(self.state, OwnedSlotState::Armed);
        self.state = OwnedSlotState::Reserved;
    }

    /// Preserve this allocator name when an impossible completion cannot
    /// prove whether a staged local result was published. Reopening the bit or
    /// guessing that a record exists would each permit a distinct authority
    /// alias, so the only sound outcome is task-lifetime quarantine.
    #[cfg(any(test, target_os = "none"))]
    fn quarantine_unknown_publish(&mut self) {
        debug_assert_eq!(self.state, OwnedSlotState::Reserved);
        self.state = OwnedSlotState::Indeterminate;
    }

    /// Run this owner's state machine with its capability species' retirement
    /// verb. Most owners use plain revoke; mapped views use exact unshare.
    fn retire_via(&mut self, retire: impl FnOnce(SlotRef) -> RetireOutcome) -> RetireOutcome {
        match self.state {
            OwnedSlotState::Released => RetireOutcome::AlreadyDead,
            OwnedSlotState::Reserved => {
                self.release_reservation();
                RetireOutcome::Retired
            }
            // A board post may still publish into this target. Without a recv
            // closure/consumption proof, releasing the allocator bit would let
            // a second owner alias that later delivery.
            OwnedSlotState::Armed => RetireOutcome::Retryable,
            // Neither state proves whether a live record remains locally.
            // Attempting a revoke could itself be a confused-deputy action if
            // the terminal send already scrubbed the result. Keep the bitmap
            // bit closed permanently instead.
            OwnedSlotState::InFlight | OwnedSlotState::Indeterminate => RetireOutcome::Retryable,
            OwnedSlotState::Live => {
                let outcome = retire(self.slot);
                self.settle_retirement(outcome)
            }
        }
    }

    fn retire_inner(&mut self) -> RetireOutcome {
        self.retire_via(crate::cap::verbs::retire_owned_slot)
    }

    fn settle_retirement(&mut self, outcome: RetireOutcome) -> RetireOutcome {
        if outcome.is_closed() {
            self.release_reservation();
        }
        outcome
    }

    fn release_reservation(&mut self) {
        if self.state == OwnedSlotState::Released {
            return;
        }
        self.state = OwnedSlotState::Released;
        #[cfg(target_os = "none")]
        crate::entry::geometry::release_cap_slot(self.slot);
    }

    /// Settle a kernel-proven outward move of this live record.
    ///
    /// A successful rendezvous send clears the sender's arena record, so a
    /// revoke here would target a vacant slot and fail to prove closure. The
    /// capability is already remote; only the local allocator reservation
    /// remains to be released. Ambiguous transfer outcomes deliberately do
    /// not take this path and retain the reservation in quarantine.
    fn release_after_proven_move(&mut self) {
        debug_assert_eq!(self.state, OwnedSlotState::Live);
        self.release_reservation();
    }

    fn is_live(&self) -> bool {
        self.state == OwnedSlotState::Live
    }

    fn retire(mut self) -> bool {
        self.retire_inner().is_closed()
    }
}

impl Drop for OwnedSlot {
    fn drop(&mut self) {
        let _ = self.retire_inner();
    }
}

impl ChannelRoot {
    #[cfg(any(test, target_os = "none"))]
    fn new(mut owned: OwnedSlot) -> Self {
        owned.activate();
        Self {
            owned: core::mem::ManuallyDrop::new(owned),
            rec: SealedRec::possessed(record::RecKind::Channel, CHANNEL_ROOT_RIGHTS),
        }
    }

    /// Borrow the root's arena name for a session builder without separating
    /// it from this affine reservation owner.
    pub fn slot(&self) -> SlotRef {
        self.owned.slot()
    }

    /// Exact rights carried by the mint-created root.
    pub fn rights(&self) -> Rights {
        self.rec.rights
    }

    /// Explicitly revoke the channel lineage and release its dynamic arena
    /// name. Retryable refusal returns this same affine owner intact.
    ///
    /// This is intentionally consuming rather than a hidden `Drop` effect:
    /// the launcher chooses the point after every local and conveyed endpoint
    /// is quiescent, and losing a retry witness cannot silently kill or alias
    /// a still-live peer.
    pub fn try_retire(mut self) -> Result<RetireOutcome, Self> {
        let outcome = self.owned.retire_inner();
        if outcome.is_closed() {
            // SAFETY: `retire_inner` has transitioned the owner to Released;
            // its Drop is now a no-op. This is the one explicit path that
            // consumes the otherwise leak-on-drop root owner.
            unsafe { core::mem::ManuallyDrop::drop(&mut self.owned) };
            Ok(outcome)
        } else {
            Err(self)
        }
    }

    /// Release only this task's dynamic arena reservation after a successful
    /// session has moved the live root into another task's private holdings.
    ///
    /// This is deliberately explicit: callers must invoke it only after a
    /// `Status::Ok` commit that named this exact slot as a held delivery.
    /// Refused or ambiguous commits must retain the owner instead.
    pub fn release_after_proven_move(mut self) {
        self.owned.release_after_proven_move();
        // SAFETY: the embedded owner is Released, so its Drop is a no-op.
        unsafe { core::mem::ManuallyDrop::drop(&mut self.owned) };
    }

    #[cfg(test)]
    fn is_live(&self) -> bool {
        self.owned.is_live()
    }
}

macro_rules! endpoint_owner {
    ($ty:ty, $kind:expr) => {
        impl $ty {
            #[cfg(any(test, target_os = "none"))]
            fn new(mut owned: OwnedSlot) -> Self {
                owned.activate();
                Self {
                    owned,
                    rec: SealedRec::possessed($kind, CHANNEL_ENDPOINT_RIGHTS),
                }
            }

            #[cfg(any(test, target_os = "none"))]
            fn new_with_rights(mut owned: OwnedSlot, rights: Rights) -> Self {
                owned.activate();
                Self {
                    owned,
                    rec: SealedRec::possessed($kind, rights),
                }
            }

            /// Borrow the endpoint's arena name without separating it from
            /// this affine reservation owner.
            pub fn slot(&self) -> SlotRef {
                self.owned.slot()
            }

            /// Exact rights carried by the mint-created endpoint.
            pub fn rights(&self) -> Rights {
                self.rec.rights
            }

            /// Explicitly retire this local endpoint. Retryable refusal
            /// returns the same owner and keeps its arena name quarantined.
            pub fn try_retire(mut self) -> Result<RetireOutcome, Self> {
                let outcome = self.owned.retire_inner();
                if outcome.is_closed() {
                    Ok(outcome)
                } else {
                    Err(self)
                }
            }

            pub(crate) fn live_slot(&self) -> Option<SlotRef> {
                self.owned.is_live().then_some(self.owned.slot())
            }

            pub fn release_after_proven_move(mut self) {
                self.owned.release_after_proven_move();
            }

            #[cfg(test)]
            fn is_live(&self) -> bool {
                self.owned.is_live()
            }
        }
    };
}

endpoint_owner!(ChanSendToken, record::RecKind::ChanSend);
endpoint_owner!(ChanRecvToken, record::RecKind::ChanRecv);

impl ChanSendToken {
    /// Erase the send-end-specific Rust view while preserving the same affine
    /// arena owner. Protocol wrappers use this only to return a partially
    /// converted malformed reply to the generic client's poison cleanup; no
    /// kernel authority moves and the dynamic slot reservation stays closed.
    pub fn into_returned_capability(self) -> ReturnedCapability {
        let Self { owned, .. } = self;
        ReturnedCapability { owned }
    }
}

impl HeldChanSendToken {
    #[cfg(target_os = "none")]
    pub(crate) fn possess(slot: SlotRef, rights: Rights) -> Self {
        Self {
            slot,
            rec: SealedRec::possessed(record::RecKind::ChanSend, rights),
        }
    }

    /// The founder-selected holding containing the delivered endpoint.
    pub fn slot(&self) -> SlotRef {
        self.slot
    }

    /// Exact rights authenticated from the delivered record.
    pub fn rights(&self) -> Rights {
        self.rec.rights
    }
}

impl HeldChanRecvToken {
    #[cfg(target_os = "none")]
    pub(crate) fn possess(slot: SlotRef, rights: Rights) -> Self {
        Self {
            slot,
            rec: SealedRec::possessed(record::RecKind::ChanRecv, rights),
        }
    }

    /// The founder-selected holding containing the delivered endpoint.
    pub fn slot(&self) -> SlotRef {
        self.slot
    }

    /// Exact rights authenticated from the delivered record.
    pub fn rights(&self) -> Rights {
        self.rec.rights
    }
}

impl ChannelPair {
    #[cfg(test)]
    pub(crate) fn from_reserved(root: OwnedSlot, send: OwnedSlot, recv: OwnedSlot) -> Self {
        Self::from_reserved_with_rights(
            root,
            send,
            CHANNEL_ENDPOINT_RIGHTS,
            recv,
            CHANNEL_ENDPOINT_RIGHTS,
        )
    }

    #[cfg(any(test, target_os = "none"))]
    pub(crate) fn from_reserved_with_rights(
        root: OwnedSlot,
        send: OwnedSlot,
        send_rights: Rights,
        recv: OwnedSlot,
        recv_rights: Rights,
    ) -> Self {
        let send = if send_rights == CHANNEL_ENDPOINT_RIGHTS {
            ChanSendToken::new(send)
        } else {
            ChanSendToken::new_with_rights(send, send_rights)
        };
        let recv = if recv_rights == CHANNEL_ENDPOINT_RIGHTS {
            ChanRecvToken::new(recv)
        } else {
            ChanRecvToken::new_with_rights(recv, recv_rights)
        };
        Self {
            root: ChannelRoot::new(root),
            send,
            recv,
        }
    }

    /// Split the pair into independently affine owners. A launcher typically
    /// retains `root` plus one endpoint and consumes the peer endpoint through
    /// [`crate::svc::ReturnCapability`].
    pub fn into_parts(self) -> (ChannelRoot, ChanSendToken, ChanRecvToken) {
        (self.root, self.send, self.recv)
    }
}

/// Runtime ownership of the one result slot used by an atomic
/// derive-then-return transfer.
///
/// This wrapper is crate-private and exposes only the transition vocabulary
/// the generic service client needs. In particular it does not expose the
/// underlying [`OwnedSlot`] or a general-purpose allocator, so service crates
/// cannot create a second naming discipline beside retained grant landings.
#[derive(Debug)]
pub(crate) struct StagedCapabilityReservation {
    owned: OwnedSlot,
}

impl StagedCapabilityReservation {
    pub(crate) fn reserve() -> Option<Self> {
        Some(Self {
            owned: OwnedSlot::reserve()?,
        })
    }

    /// The builder-only result target. The public staged-return value keeps
    /// this name opaque to its caller.
    pub(crate) fn target(&self) -> SlotRef {
        self.owned.slot()
    }

    /// Cross the commit boundary. A cancelled future from here onward leaves
    /// the reservation quarantined by [`OwnedSlot::drop`].
    pub(crate) fn begin_transfer(&mut self) {
        debug_assert_eq!(self.owned.state, OwnedSlotState::Reserved);
        self.owned.state = OwnedSlotState::InFlight;
    }

    /// Settle either a proven move (sender record scrubbed) or an I3-proven
    /// non-commit (result never published). Both prove the local slot vacant.
    pub(crate) fn settle_proven(&mut self) {
        debug_assert_eq!(self.owned.state, OwnedSlotState::InFlight);
        self.owned.release_reservation();
    }

    /// Preserve an allocator tombstone when completion cannot prove whether
    /// the result is live, moved, or scrubbed.
    pub(crate) fn quarantine(&mut self) {
        debug_assert_eq!(self.owned.state, OwnedSlotState::InFlight);
        self.owned.state = OwnedSlotState::Indeterminate;
    }
}

/// One capability received through a runtime-owned rendezvous landing and
/// retained beyond the request that delivered it.
///
/// The receive post donates a runtime-reserved arena slot.  Once a capability
/// lands, this affine value owns both that record and the allocator
/// reservation; dropping it attempts to revoke the record before reuse, and a
/// refused revoke quarantines the slot rather than aliasing live authority.
/// This is the species-neutral sibling of [`StashedLease`]: a typed receive
/// policy may prove that the delivered record is a lease and retype this same
/// owner without surrendering its allocator reservation.
#[derive(Debug)]
pub struct ReturnedCapability {
    owned: OwnedSlot,
}

#[cfg(any(test, target_os = "none"))]
fn record_has_exact_kind_and_rights(
    record: &kore_abi::cap::CapRecord,
    kind: kore_abi::cap::ResourceKind,
    rights: Rights,
) -> bool {
    record.kind == kind as u16 && record.rights == rights
}

impl ReturnedCapability {
    /// Runtime grant pipeline only: the successful exact send consumed this
    /// record, so release its local reservation without revoking remote custody.
    pub(crate) fn release_after_proven_move(mut self) {
        self.owned.release_after_proven_move();
    }

    #[cfg(test)]
    pub(crate) fn test_live(slot: SlotRef) -> Self {
        Self {
            owned: OwnedSlot {
                slot,
                state: OwnedSlotState::Live,
                _thread_bound: core::marker::PhantomData,
            },
        }
    }

    /// Reserve a vacant target for a receive post. Kept inside the runtime:
    /// only the generic service machines may turn a bare reservation into
    /// delivered authority.
    pub(crate) fn reserve() -> Option<Self> {
        Some(Self {
            owned: OwnedSlot::reserve()?,
        })
    }

    /// The vacant target a receive post donates, or the live slot after the
    /// transfer lands.
    pub(crate) fn target(&self) -> SlotRef {
        self.owned.slot()
    }

    /// Mark the reserved target live after the slot-record probe witnesses a
    /// capability delivery.
    pub(crate) fn activate(&mut self) {
        self.owned.activate();
    }

    /// Mark this vacant owner as the target of a post that is known (or may be
    /// known after an ambiguous transport result) to stand on the board.
    pub(crate) fn arm_post(&mut self) {
        self.owned.arm();
    }

    /// Clear the board-target state after endpoint closure or observed post
    /// consumption proves that no delivery can still race slot reuse.
    pub(crate) fn disarm_post(&mut self) {
        self.owned.disarm();
    }

    /// The service-arena name of the live returned capability.  A reserved or
    /// successfully retired value exposes no authority name.
    pub fn slot(&self) -> Option<SlotRef> {
        self.owned.is_live().then_some(self.owned.slot())
    }

    /// Immutable resource identity of this authenticated returned record.
    /// The value is correlation metadata only; the affine owner remains the
    /// authority and every subsequent effect re-binds its slot in the kernel.
    pub fn scope_rid_with(
        &self,
        kind: kore_abi::cap::ResourceKind,
        required: Rights,
    ) -> Option<u64> {
        let slot = self.slot()?;
        verbs::capability_scope_rid(slot, kind, required)
    }

    /// Non-owning ingress check for a returned image parent.  This reads the
    /// authenticated delivery's public slot record as metadata without
    /// constructing a second affine token for the same authority; the later
    /// map/carve/revoke verbs still perform the authoritative bind.
    #[cfg(target_os = "none")]
    pub fn is_lease_with(&self, required: Rights) -> bool {
        use kore_abi::cap::{CapRecord, ResourceKind};

        let Some(slot) = self.slot() else {
            return false;
        };
        let Some(ptr) = crate::entry::geometry::slot_record_va(slot) else {
            return false;
        };
        // SAFETY: `slot_record_va` names this thread's read-only slots alias.
        let record = unsafe { core::ptr::read_volatile(ptr as *const CapRecord) };
        record.kind == ResourceKind::MemLease as u16 && record.rights.contains(required)
    }

    /// Host builds have no live arena record to inspect.
    #[cfg(not(target_os = "none"))]
    pub fn is_lease_with(&self, _required: Rights) -> bool {
        false
    }

    /// Check the kernel-authenticated species and required rights of a
    /// returned capability without constructing a second affine owner.
    ///
    /// Dynamic service-session assembly uses this to distinguish the ring
    /// lease, kick endpoint, and return endpoint delivered through the same
    /// retained return leg. The check reads the delivered record; wire data
    /// never chooses or overrides its kind.
    #[cfg(target_os = "none")]
    pub fn is_kind_with(&self, kind: kore_abi::cap::ResourceKind, required: Rights) -> bool {
        use kore_abi::cap::CapRecord;

        let Some(slot) = self.slot() else {
            return false;
        };
        let Some(ptr) = crate::entry::geometry::slot_record_va(slot) else {
            return false;
        };
        // SAFETY: `slot_record_va` names this thread's read-only slot alias.
        let record = unsafe { core::ptr::read_volatile(ptr as *const CapRecord) };
        record.kind == kind as u16 && record.rights.contains(required)
    }

    /// Check the kernel-authenticated species and exact rights of a returned
    /// capability without changing its affine Rust owner.
    ///
    /// This is the fail-closed ingress check for protocol results whose rights
    /// shape is itself part of the contract: an over-granted endpoint is as
    /// malformed as an under-granted or wrong-species result. The later typed
    /// conversion repeats the species/minimum-rights check before moving the
    /// same [`OwnedSlot`] into its typed witness.
    #[cfg(target_os = "none")]
    pub fn is_kind_with_exact_rights(
        &self,
        kind: kore_abi::cap::ResourceKind,
        rights: Rights,
    ) -> bool {
        use kore_abi::cap::CapRecord;

        let Some(slot) = self.slot() else {
            return false;
        };
        let Some(ptr) = crate::entry::geometry::slot_record_va(slot) else {
            return false;
        };
        // SAFETY: `slot_record_va` names this thread's read-only slot alias.
        let record = unsafe { core::ptr::read_volatile(ptr as *const CapRecord) };
        record_has_exact_kind_and_rights(&record, kind, rights)
    }

    /// Host builds have no authenticated capability-record alias.
    #[cfg(not(target_os = "none"))]
    pub fn is_kind_with(&self, _kind: kore_abi::cap::ResourceKind, _required: Rights) -> bool {
        false
    }

    /// Host builds have no authenticated capability-record alias.
    #[cfg(not(target_os = "none"))]
    pub fn is_kind_with_exact_rights(
        &self,
        _kind: kore_abi::cap::ResourceKind,
        _rights: Rights,
    ) -> bool {
        false
    }

    /// Retype a receiver-owned delivery as a retained lease without changing
    /// either its arena name or its allocator reservation.
    ///
    /// The check is deliberately repeated at this ownership boundary even
    /// when the rendezvous post carried a typed admission policy.  A protocol
    /// wrapper therefore cannot manufacture a lease owner from an arbitrary
    /// returned record merely because both values name arena slots.
    pub fn try_into_stashed_lease(self, required: Rights) -> Result<StashedLease, Self> {
        if !self.is_lease_with(required) {
            return Err(self);
        }
        Ok(self.into_stashed_lease_after_check())
    }

    /// Authenticate a received mapping route without converting it into a
    /// physical allocation lease or changing its ancestry.
    pub fn try_into_fabric_route(self, required: Rights) -> Result<FabricRouteToken, Self> {
        if !self
            .slot()
            .and_then(|slot| {
                verbs::capability_rights(slot, kore_abi::cap::ResourceKind::MemoryFabricRoute)
            })
            .is_some_and(|rights| rights.contains(required))
        {
            return Err(self);
        }
        Ok(FabricRouteToken { owned: self.owned })
    }

    /// Preserve which authority species the exporter actually transferred.
    pub fn try_into_memory(self, required: Rights) -> Result<StashedMemory, Self> {
        match self.try_into_stashed_lease(required) {
            Ok(lease) => Ok(StashedMemory::Lease(lease)),
            Err(owner) => owner
                .try_into_fabric_route(required)
                .map(StashedMemory::Route),
        }
    }

    fn into_stashed_lease_after_check(self) -> StashedLease {
        let Self { owned } = self;
        StashedLease { owned }
    }

    /// Retype a receiver-owned detached memory view without surrendering its
    /// dynamic arena reservation. The returned owner carries no delegation
    /// authority; mapping re-authenticates the live `LeasedRegion` by slot.
    #[cfg(target_os = "none")]
    pub fn try_into_stashed_view(self, required: Rights) -> Result<StashedView, Self> {
        use kore_abi::cap::{CapRecord, ResourceKind};

        let Some(slot) = self.slot() else {
            return Err(self);
        };
        let Some(ptr) = crate::entry::geometry::slot_record_va(slot) else {
            return Err(self);
        };
        // SAFETY: `slot_record_va` names this task's read-only slots alias.
        let record = unsafe { core::ptr::read_volatile(ptr as *const CapRecord) };
        if record.kind != ResourceKind::LeasedRegion as u16 || !record.rights.contains(required) {
            return Err(self);
        }
        let Self { owned } = self;
        Ok(StashedView { owned })
    }

    #[cfg(not(target_os = "none"))]
    pub fn try_into_stashed_view(self, _required: Rights) -> Result<StashedView, Self> {
        Err(self)
    }

    /// Retype a receiver-owned delivery as a channel **recv-end** owner
    /// (the [`try_into_stashed_lease`](Self::try_into_stashed_lease)
    /// discipline, one kind over — slice C: a minted loader face's grant
    /// bank delivers recv ends the parent then conveys at a spawn).  The
    /// kind + rights check is repeated at this ownership boundary; a
    /// mismatched delivery hands the owner back for the caller's poison
    /// or drop path.
    #[cfg(target_os = "none")]
    pub fn try_into_chan_recv(self, required: Rights) -> Result<ChanRecvToken, Self> {
        use kore_abi::cap::{CapRecord, ResourceKind};

        let Some(slot) = self.slot() else {
            return Err(self);
        };
        let Some(ptr) = crate::entry::geometry::slot_record_va(slot) else {
            return Err(self);
        };
        // SAFETY: `slot_record_va` names this thread's read-only slots alias.
        let record = unsafe { core::ptr::read_volatile(ptr as *const CapRecord) };
        if record.kind != ResourceKind::ChanRecv as u16 || !record.rights.contains(required) {
            return Err(self);
        }
        let Self { owned } = self;
        Ok(ChanRecvToken {
            owned,
            rec: SealedRec::possessed(record::RecKind::ChanRecv, record.rights),
        })
    }

    /// Host builds have no live arena record to retype against.
    #[cfg(not(target_os = "none"))]
    pub fn try_into_chan_recv(self, _required: Rights) -> Result<ChanRecvToken, Self> {
        Err(self)
    }

    /// Retype a receiver-owned delivery as a channel **send-end** owner.
    /// This is the capability-lane counterpart of
    /// [`try_into_chan_recv`](Self::try_into_chan_recv): task-local executable
    /// handles are ordinary attenuated send endpoints, so the delivered
    /// record's kind and rights are checked before the affine owner changes
    /// type.
    #[cfg(target_os = "none")]
    pub fn try_into_chan_send(self, required: Rights) -> Result<ChanSendToken, Self> {
        use kore_abi::cap::{CapRecord, ResourceKind};

        let Some(slot) = self.slot() else {
            return Err(self);
        };
        let Some(ptr) = crate::entry::geometry::slot_record_va(slot) else {
            return Err(self);
        };
        // SAFETY: `slot_record_va` names this task's read-only slots alias.
        let record = unsafe { core::ptr::read_volatile(ptr as *const CapRecord) };
        if record.kind != ResourceKind::ChanSend as u16 || !record.rights.contains(required) {
            return Err(self);
        }
        let Self { owned } = self;
        Ok(ChanSendToken {
            owned,
            rec: SealedRec::possessed(record::RecKind::ChanSend, record.rights),
        })
    }

    /// Retype a received self-kind rendezvous-mint grant. The runtime-owned
    /// landing remains embedded in the affine token so it cannot be reused
    /// while this task can still mint descendant channels.
    #[cfg(target_os = "none")]
    pub fn try_into_channel_space(self) -> Result<ChannelSpaceToken, Self> {
        use kore_abi::cap::{CapRecord, ResourceKind};

        let Some(slot) = self.slot() else {
            return Err(self);
        };
        let Some(ptr) = crate::entry::geometry::slot_record_va(slot) else {
            return Err(self);
        };
        // SAFETY: `slot_record_va` names this task's read-only slots alias.
        let record = unsafe { core::ptr::read_volatile(ptr as *const CapRecord) };
        if record.kind != ResourceKind::ChannelSpace as u16 || record.rights != CHANNEL_SPACE_RIGHTS
        {
            return Err(self);
        }
        Ok(ChannelSpaceToken::possess_returned(
            slot,
            record.rights,
            self,
        ))
    }

    #[cfg(not(target_os = "none"))]
    pub fn try_into_channel_space(self) -> Result<ChannelSpaceToken, Self> {
        Err(self)
    }

    #[cfg(not(target_os = "none"))]
    pub fn try_into_chan_send(self, _required: Rights) -> Result<ChanSendToken, Self> {
        Err(self)
    }

    /// Try to retire the returned parent and preserve whether this call
    /// committed the revoke, found dead lineage, or retained a retry handle.
    /// Services with derived children must retire those children first so the
    /// ancestry ordering remains explicit.
    pub fn try_retire_outcome(&mut self) -> RetireOutcome {
        self.owned.retire_inner()
    }

    /// Compatibility predicate for callers that need only local closure.
    /// Prefer [`try_retire_outcome`](Self::try_retire_outcome) when a
    /// dead-lineage observation is operationally relevant.
    pub fn try_retire(&mut self) -> bool {
        self.try_retire_outcome().is_closed()
    }

    /// Whether this value still owns a live delivered record.
    pub fn is_live(&self) -> bool {
        self.owned.is_live()
    }
}

/// A dynamically named lease retained by a service.
///
/// Unlike a bare [`SlotRef`], this value owns both the live capability and the
/// runtime allocator reservation for its arena slot.  Dropping it revokes the
/// lease before reopening the slot; a refused revoke quarantines the slot
/// instead of letting a later capability alias a still-live record.
///
/// The image service uses this for every archive segment it keeps between
/// `Register` and `Acquire`.  Keeping the reservation in the affine value is
/// what prevents those leases from colliding with the runtime's serve scratch,
/// report, heap-carve, or startup-link bands.
#[derive(Debug)]
pub struct StashedLease {
    owned: OwnedSlot,
}

impl StashedLease {
    /// Adopt an already-activated owned slot the kernel just published as a
    /// writable lease — the executable → writable return's landing
    /// (`res/IMAGE.md` §6a). Crate-internal: the caller must have proved
    /// the record's shape first (`writable_result_state`).
    #[cfg(target_os = "none")]
    pub(crate) fn from_owned(owned: OwnedSlot) -> Self {
        Self { owned }
    }
}

/// Transferable memory custody: an allocation lease or a scoped mapping loan.
/// A route never acquires the allocation/release authority of a MemLease.
#[derive(Debug)]
pub enum StashedMemory {
    Lease(StashedLease),
    Route(FabricRouteToken),
}

impl From<StashedLease> for StashedMemory {
    fn from(lease: StashedLease) -> Self {
        Self::Lease(lease)
    }
}

impl StashedMemory {
    pub fn slot(&self) -> Option<SlotRef> {
        match self {
            Self::Lease(lease) => lease.slot(),
            Self::Route(route) => route.owned.is_live().then_some(route.slot()),
        }
    }

    pub fn into_returned_capability(self) -> ReturnedCapability {
        match self {
            Self::Lease(lease) => lease.into_returned_capability(),
            Self::Route(route) => route.into_returned_capability(),
        }
    }

    pub fn try_retire_outcome(&mut self) -> RetireOutcome {
        match self {
            Self::Lease(lease) => lease.try_retire_outcome(),
            Self::Route(route) => {
                if route.try_close().is_err() {
                    return RetireOutcome::Retryable;
                }
                route.owned.retire_inner()
            }
        }
    }
}

/// A dynamically received detached `LeasedRegion` retained with its runtime
/// arena reservation. Unlike [`StashedLease`], this is a viewer, not a lessor:
/// it carries access to one authenticated window and cannot mint descendants.
#[derive(Debug)]
pub struct StashedView {
    owned: OwnedSlot,
}

/// One exact whole-lease mapping, paired with the owned `LeasedRegion` record
/// and the parent lease name that authenticated it. The access is the
/// constructing verb's choice: read-only from
/// [`map_owned_lease_at`](verbs::map_owned_lease_at) (the validation
/// shape), read-write from
/// [`map_owned_lease_at_rw`](verbs::map_owned_lease_at_rw) (the
/// stamp-before-convey shape).
///
/// The image service uses the read-only flavor for its single reusable
/// registration-validation VA; the shell uses the writable one to stamp
/// a child supply's `.kore.ld` page before conveying the root away.
/// Keeping the mapped view affine is what makes VA reuse a lifecycle fact:
/// the next source cannot map there until this owner is retired or proved
/// already dead through its ancestry.
#[derive(Debug)]
pub struct MappedLeaseView {
    /// The default `OwnedSlot` drop performs plain revoke; a mapped view must
    /// instead exact-unshare. Manual drop prevents a refused exact unshare
    /// from falling through to the weaker verb and closing the record while
    /// its PTE remains installed.
    owned: core::mem::ManuallyDrop<OwnedSlot>,
    lease: SlotRef,
    base: core::ptr::NonNull<u8>,
    len: usize,
}

/// One private R/W code mapping paired with the lease that authenticates it.
///
/// This state is affine and thread-bound through both owned capability names.
/// It exposes bounded byte writes rather than a safe `&mut [u8]`: an arbitrary
/// lease may already have mappings in other tasks, so Rust-reference
/// exclusivity cannot be inferred merely from possession of the parent.
#[derive(Debug)]
pub struct WritableCode {
    // Drop the exact view before the parent on ordinary teardown.
    view: MappedLeaseView,
    lease: StashedLease,
}

/// Lexically scoped byte emitter for a [`WritableCode`] mapping.
pub struct CodeWriter<'a> {
    view: &'a mut MappedLeaseView,
}

impl CodeWriter<'_> {
    /// Size of the complete writable mapping.
    pub const fn len(&self) -> usize {
        self.view.len
    }

    /// Whether the mapping contains no bytes. Whole-lease mappings are never
    /// empty, but this completes the conventional `len` surface.
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copy `bytes` into the mapping at `offset` after checking the complete
    /// destination range. Source and destination may overlap.
    pub fn write_at(&mut self, offset: usize, bytes: &[u8]) -> Result<(), CodeWriteError> {
        self.check_range(offset, bytes.len())?;
        // SAFETY: `check_range` proved the destination lies inside the live
        // mapping. Raw copy avoids manufacturing an exclusive Rust slice for
        // backing that may also be mapped elsewhere.
        unsafe {
            core::ptr::copy(
                bytes.as_ptr(),
                self.view.as_mut_ptr().add(offset),
                bytes.len(),
            )
        };
        Ok(())
    }

    /// Fill the complete mapping with one byte value.
    pub fn fill(&mut self, byte: u8) {
        // SAFETY: construction guarantees a live mapping of exactly `len`
        // bytes, and this raw write creates no Rust reference.
        unsafe { core::ptr::write_bytes(self.view.as_mut_ptr(), byte, self.len()) };
    }

    /// Raw base for a compiler backend or C ABI emitter.
    ///
    /// Dereferencing the pointer is unsafe: the caller must remain within
    /// `len()` and uphold any concurrency contract with other mappings.
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.view.as_mut_ptr()
    }

    fn check_range(&self, offset: usize, len: usize) -> Result<(), CodeWriteError> {
        if offset.checked_add(len).is_some_and(|end| end <= self.len()) {
            Ok(())
        } else {
            Err(CodeWriteError::OutOfBounds {
                offset,
                len,
                capacity: self.len(),
            })
        }
    }
}

impl MappedLeaseView {
    #[cfg(target_os = "none")]
    pub(crate) fn new(
        mut owned: OwnedSlot,
        lease: SlotRef,
        base: core::ptr::NonNull<u8>,
        len: usize,
    ) -> Self {
        owned.activate();
        Self {
            owned: core::mem::ManuallyDrop::new(owned),
            lease,
            base,
            len,
        }
    }

    /// Exact mapped extent, without borrowing the underlying memory.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Permanently remove write access from a page-aligned part of this view.
    /// The current thread's held address-space capability authorizes the
    /// existing clear-only protection verb. A refusal leaves the range intact.
    /// Raw pointers previously obtained from this view must honor the new access.
    pub fn seal_read_only(&mut self, offset: usize, len: usize) -> bool {
        if len == 0
            || offset % 4096 != 0
            || len % 4096 != 0
            || offset.checked_add(len).is_none_or(|end| end > self.len)
        {
            return false;
        }
        let Some(va) = (self.base.as_ptr() as usize).checked_add(offset) else {
            return false;
        };
        #[cfg(target_os = "none")]
        {
            let Some(space) = crate::entry::geometry::self_space_slot() else {
                return false;
            };
            let Some(scratch) = OwnedSlot::reserve() else {
                return false;
            };
            // PROTECT mints no record: dropping the reserved name returns it
            // to this thread's allocator after the synchronous session.
            verbs::seal_range(
                space,
                va as u64,
                len as u64,
                kore_abi::session::SHARE_ACCESS_R,
                scratch.slot(),
            )
        }
        #[cfg(not(target_os = "none"))]
        {
            let _ = va;
            false
        }
    }

    /// Borrow the live read-only mapping for validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure the lessor does not mutate or revoke the source
    /// while the returned slice is borrowed. The registration protocol meets
    /// that condition by transferring the parent before publication and by
    /// keeping this affine view local to one synchronous serve step.
    pub unsafe fn as_ref(&self) -> &[u8] {
        // SAFETY: construction follows a successful exact mapping of `len`
        // bytes at `base`; the affine owner keeps that view live for `self`.
        unsafe { core::slice::from_raw_parts(self.base.as_ptr(), self.len) }
    }

    /// Borrow the live mapping mutably — lawful only for a view minted
    /// writable ([`map_owned_lease_at_rw`](verbs::map_owned_lease_at_rw))
    /// which has not subsequently been sealed read-only.
    ///
    /// # Safety
    ///
    /// As [`as_ref`](Self::as_ref), plus: the caller must be the mapping's
    /// sole toucher for the borrow's duration, and the entire mapping must
    /// still be writable — met by the stamp shape,
    /// where the view is private to one control flow and retired before
    /// the lease is conveyed anywhere.
    pub unsafe fn as_mut(&mut self) -> &mut [u8] {
        // SAFETY: construction followed a successful exact `R|W` mapping of
        // `len` bytes at `base`; the affine owner keeps that view live and
        // the caller warrants exclusivity.
        unsafe { core::slice::from_raw_parts_mut(self.base.as_ptr(), self.len) }
    }

    /// Raw base of the live mapping for shared-memory protocols that must not
    /// materialize an exclusive Rust slice while a sibling task maps the same
    /// frames. The affine view still owns the mapping lifetime.
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.base.as_ptr()
    }

    /// Raw base of the live mapping for instruction entry or immutable wire
    /// consumers. The affine view remains the mapping-lifetime owner.
    pub fn as_ptr(&self) -> *const u8 {
        self.base.as_ptr()
    }

    /// Atomically retire this mapped view through exact unshare. The kernel
    /// validates that `lease` is the view's direct parent and removes this
    /// precise share-list edge and PTE before reporting success.
    pub fn try_retire_outcome(&mut self) -> RetireOutcome {
        let lease = self.lease;
        self.owned
            .retire_via(|view| crate::cap::verbs::retire_owned_view(view, lease))
    }

    /// Whether the exact mapping still owns a live local capability name.
    pub fn is_live(&self) -> bool {
        self.owned.is_live()
    }
}

impl Drop for MappedLeaseView {
    fn drop(&mut self) {
        if self.try_retire_outcome().is_closed() {
            // SAFETY: exact unshare transitioned the owner to Released (or it
            // was already Released), so OwnedSlot::drop is now a no-op. A
            // refusal drops the Rust retry owner but deliberately preserves
            // the kernel name and allocator reservation, quarantining this VA
            // for the rest of the task rather than misreporting it reusable.
            unsafe { core::mem::ManuallyDrop::drop(&mut self.owned) };
        }
    }
}

impl StashedLease {
    #[cfg(target_os = "none")]
    pub(crate) fn new(mut owned: OwnedSlot) -> Self {
        owned.activate();
        Self { owned }
    }

    /// Erase the lease-specific Rust view while preserving the same affine
    /// arena owner. This is used only to return a partially converted malformed
    /// reply to the generic client's poison cleanup; no kernel authority moves.
    pub fn into_returned_capability(self) -> ReturnedCapability {
        let Self { owned } = self;
        ReturnedCapability { owned }
    }

    /// The arena slot later grant sessions name as their subject, while the
    /// lease is still live. Successful retirement makes the old name
    /// inaccessible before the allocator may reuse it.
    pub fn slot(&self) -> Option<SlotRef> {
        self.owned.is_live().then_some(self.owned.slot)
    }

    /// Consume this affine owner after the kernel has proved its record moved
    /// to a peer. The record is already absent locally, so this releases the
    /// dynamic slot reservation without attempting a second revoke.
    pub fn release_after_proven_move(mut self) {
        self.owned.release_after_proven_move();
    }

    /// Try to retire the lease and preserve whether this call committed the
    /// revoke, found dead lineage, or retained a retry handle.
    pub fn try_retire_outcome(&mut self) -> RetireOutcome {
        self.owned.retire_inner()
    }

    /// Compatibility predicate for callers that need only local closure.
    /// Prefer [`try_retire_outcome`](Self::try_retire_outcome) when a
    /// dead-lineage observation is operationally relevant.
    pub fn try_retire(&mut self) -> bool {
        self.try_retire_outcome().is_closed()
    }

    /// Whether the lease remains a grantable live capability.
    pub fn is_live(&self) -> bool {
        self.owned.is_live()
    }

    /// Replace this runtime-owned writable lease with an
    /// executable-only successor, transferring its dynamic arena ownership.
    /// A pre-commit refusal returns the exact writable owner for retry or
    /// teardown; a failure after source retirement is reported as consumed.
    pub fn finalize_exec(self) -> Result<ExecutableCode, FinalizeExecError<Self>> {
        verbs::finalize_stashed_exec(self)
    }

    /// Consume this lease into a private whole-lease R/W code mapping.
    ///
    /// `window` is caller-owned VA capacity; only the lease's exact length is
    /// mapped at its base. Every failure returns the original affine lease.
    pub fn into_writable_code_at(
        self,
        window: kore_abi::task::layout::Window,
    ) -> Result<WritableCode, WritableCodeMapError> {
        let source = match self.slot() {
            Some(source) => source,
            None => {
                return Err(WritableCodeMapError {
                    lease: self,
                    kind: WritableCodeMapErrorKind::SourceUnavailable,
                });
            }
        };
        let (_, needed) = match verbs::lease_geometry(source) {
            Some(geometry) => geometry,
            None => {
                return Err(WritableCodeMapError {
                    lease: self,
                    kind: WritableCodeMapErrorKind::GeometryUnavailable,
                });
            }
        };
        if needed > window.len() {
            return Err(WritableCodeMapError {
                lease: self,
                kind: WritableCodeMapErrorKind::WindowTooSmall {
                    needed,
                    available: window.len(),
                },
            });
        }
        let Some(view) = verbs::map_owned_lease_at_rw(source, window.base(), needed) else {
            return Err(WritableCodeMapError {
                lease: self,
                kind: WritableCodeMapErrorKind::Refused,
            });
        };
        Ok(WritableCode { view, lease: self })
    }

    /// Map, initialize, and irreversibly finalize a writable lease.
    ///
    /// The closure is lexical and receives only bounded byte operations plus
    /// an FFI pointer escape hatch. A closure error is not a rollback: its
    /// possibly modified [`WritableCode`] is returned in the error.
    pub fn initialize_executable_at<T, E>(
        self,
        window: kore_abi::task::layout::Window,
        initialize: impl FnOnce(&mut CodeWriter<'_>) -> Result<T, E>,
    ) -> Result<(ExecutableCode, T), InitializeExecutableError<E>> {
        let code = self
            .into_writable_code_at(window)
            .map_err(InitializeExecutableError::Map)?;
        let (code, value) = code
            .initialize(initialize)
            .map_err(InitializeExecutableError::Initialize)?;
        match code.finalize() {
            Ok(executable) => Ok((executable, value)),
            Err(FinalizeCodeError::Refused(code)) => {
                Err(InitializeExecutableError::FinalizeRefused(code))
            }
            Err(FinalizeCodeError::Consumed) => Err(InitializeExecutableError::Consumed),
        }
    }
}

impl WritableCode {
    /// Run one lexical emission step while retaining affine ownership.
    ///
    /// On failure the returned owner contains the same mapping and lease; any
    /// writes completed before the error remain visible.
    pub fn initialize<T, E>(
        mut self,
        initialize: impl FnOnce(&mut CodeWriter<'_>) -> Result<T, E>,
    ) -> Result<(Self, T), CodeInitializeError<E>> {
        let result = initialize(&mut CodeWriter {
            view: &mut self.view,
        });
        match result {
            Ok(value) => Ok((self, value)),
            Err(error) => Err(CodeInitializeError { code: self, error }),
        }
    }

    /// Mapping length in bytes.
    pub const fn len(&self) -> usize {
        self.view.len
    }

    /// Whether the mapping contains no bytes.
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Consume the writable parent while its view is still live, proving that
    /// the kernel's transition reaps descendant aliases before publication.
    pub fn finalize(self) -> Result<ExecutableCode, FinalizeCodeError> {
        let Self { view, lease } = self;
        match lease.finalize_exec() {
            Ok(executable) => {
                drop(view);
                Ok(executable)
            }
            Err(FinalizeExecError::Refused(lease)) => {
                Err(FinalizeCodeError::Refused(Self { view, lease }))
            }
            Err(FinalizeExecError::Consumed) => {
                drop(view);
                Err(FinalizeCodeError::Consumed)
            }
        }
    }
}

/// A **reusable code arena**: one physical window that alternates between
/// writable and executable, and is never both (`res/IMAGE.md` §6a).
///
/// This is the ergonomic form of the W^X cycle. The raw verbs are all
/// affine and correct on their own, but a compiler that emits more than
/// once has to hold the state machine in its head — map, write, finalize,
/// map R/X, run, retract, map again — and each step has a fail-closed arm
/// that returns a different owner. [`JitArena`] holds that machine so a
/// caller writes [`emit`](Self::emit) in a loop.
///
/// The safety argument is structural rather than documentary. The arena
/// owns exactly one of the two states at a time, `emit` takes `&mut self`
/// while [`code`](Self::code) hands out a shared borrow, and the
/// executable authority is consumed by value on the way back — so a live
/// `ExecutableMapping` cannot span a rewrite, and no code path can hold a
/// writable and an executable view of the window at once.
///
/// What it is not: an incremental JIT arena. A generation is published
/// whole. Code that is live and reachable cannot be patched in place,
/// because returning the window closes every executable alias over it —
/// which is exactly the invariant that makes this safe. A tracing JIT that
/// re-patches live traces wants a different (and weaker) memory model; a
/// compile-run-discard JIT, a template JIT, or a
/// method-at-a-time-into-a-fresh-arena JIT fit here.
pub struct JitArena {
    window: kore_abi::task::layout::Window,
    state: Option<ArenaState>,
}

enum ArenaState {
    /// Mapped writable, awaiting bytes.
    Writable(WritableCode),
    /// Published; the current generation is executable.
    Executable(ExecutableCode),
}

/// Why an [`emit`](JitArena::emit) could not publish a generation.
///
/// Every variant except [`Emit`](Self::Emit) leaves the arena **spent**:
/// the window is gone and the arena must be dropped. `Emit` is the caller's
/// own error and leaves the arena usable.
#[derive(Debug)]
pub enum JitError<E> {
    /// The caller's emitter returned an error. The arena keeps its window
    /// and any bytes already written; retry or drop it.
    Emit(E),
    /// The arena was already spent by an earlier failure.
    Spent,
    /// The executable window could not be returned for rewriting.
    Retract,
    /// The window could not be mapped writable.
    Map,
    /// The written bytes could not be published to instruction fetch.
    Publish,
}

impl JitArena {
    /// Open an arena over `window`, backed by `lease`.
    ///
    /// The arena starts writable and unpublished; nothing is executable
    /// until the first [`emit`](Self::emit).
    #[cfg(target_os = "none")]
    pub fn new(
        lease: StashedLease,
        window: kore_abi::task::layout::Window,
    ) -> Result<Self, WritableCodeMapError> {
        let code = lease.into_writable_code_at(window)?;
        Ok(Self {
            window,
            state: Some(ArenaState::Writable(code)),
        })
    }

    /// Emit one generation: rewrite the window and publish it.
    ///
    /// Returns the emitter's own value. Afterwards the arena is executable
    /// and [`code`](Self::code) hands out the authority to map and run it.
    /// Calling `emit` again returns the window to writable first, which
    /// tears down every mapping of the previous generation — so a caller
    /// must not be holding one, and cannot be: `code` borrows the arena.
    #[cfg(target_os = "none")]
    pub fn emit<T, E>(
        &mut self,
        emit: impl FnOnce(&mut CodeWriter<'_>) -> Result<T, E>,
    ) -> Result<T, JitError<E>> {
        // Get to a writable, mapped state — from either state, or fail with
        // the arena already emptied so a spent arena can never be reused.
        let writable = match self.state.take().ok_or(JitError::Spent)? {
            ArenaState::Writable(code) => code,
            ArenaState::Executable(code) => {
                let lease = code.retract().map_err(|_| JitError::Retract)?;
                lease
                    .into_writable_code_at(self.window)
                    .map_err(|_| JitError::Map)?
            }
        };
        let (writable, value) = match writable.initialize(emit) {
            Ok(pair) => pair,
            Err(failed) => {
                // The caller's error is not the arena's: keep the window,
                // mapped and writable, exactly as the emitter left it.
                let CodeInitializeError { code, error } = failed;
                self.state = Some(ArenaState::Writable(code));
                return Err(JitError::Emit(error));
            }
        };
        match writable.finalize() {
            Ok(executable) => {
                self.state = Some(ArenaState::Executable(executable));
                Ok(value)
            }
            Err(FinalizeCodeError::Refused(code)) => {
                self.state = Some(ArenaState::Writable(code));
                Err(JitError::Publish)
            }
            Err(FinalizeCodeError::Consumed) => Err(JitError::Publish),
        }
    }

    /// The published generation, if the last [`emit`](Self::emit)
    /// succeeded. Map it with
    /// [`map_rx_at`](ExecutableCode::map_rx_at) to run it.
    ///
    /// The borrow is what keeps a running generation and a rewrite apart:
    /// `emit` needs `&mut self`, so no mapping taken from here can be live
    /// across one.
    pub fn code(&self) -> Option<&ExecutableCode> {
        match self.state.as_ref() {
            Some(ArenaState::Executable(code)) => Some(code),
            _ => None,
        }
    }

    /// Whether the arena still owns its window. `false` after a failure
    /// that consumed it.
    pub fn is_live(&self) -> bool {
        self.state.is_some()
    }
}

impl StashedView {
    /// The live arena slot re-authenticated by the receiver-map operation.
    pub fn slot(&self) -> Option<SlotRef> {
        self.owned.is_live().then_some(self.owned.slot)
    }

    /// Try to retire the detached view while preserving retryable ownership.
    pub fn try_retire_outcome(&mut self) -> RetireOutcome {
        self.owned.retire_inner()
    }
}

/// Executable-only physical code authority after a writable source lineage
/// was consumed by `REVOKE_KIND_EXEC`.
#[derive(Debug)]
pub struct ExecutableCode {
    owned: OwnedSlot,
}

impl ExecutableCode {
    #[cfg(target_os = "none")]
    pub(crate) fn new(mut owned: OwnedSlot) -> Self {
        owned.activate();
        Self { owned }
    }

    /// Release only this handle's local allocator reservation, after the
    /// committed kernel transition already cleared its arena record.
    #[cfg(target_os = "none")]
    pub(crate) fn release_after_proven_move(&mut self) {
        self.owned.release_after_proven_move();
    }

    /// Give this window back as writable so it can be rewritten
    /// (`res/IMAGE.md` §6a) — the reusable-arena half of the W^X seam.
    ///
    /// Consuming `self` is the local proof that no R/X mapping survives:
    /// [`ExecutableMapping`] borrows this authority, so one cannot outlive
    /// the move. The kernel makes the global argument, closing the whole
    /// executable lineage and every CPU, device and guest alias before the
    /// writable successor exists.
    ///
    /// Refused for a window that never made the forward trade — an X-minted
    /// immutable store is executable forever.
    #[cfg(target_os = "none")]
    pub fn retract(self) -> Result<StashedLease, FinalizeExecError<Self>> {
        verbs::return_writable(self)
    }

    /// The executable lease that an `Acquire` grant or RX share names.
    pub fn slot(&self) -> Option<SlotRef> {
        self.owned.is_live().then_some(self.owned.slot)
    }

    /// Retire the executable lease while preserving a retryable owner on a
    /// transient refusal.
    pub fn try_retire_outcome(&mut self) -> RetireOutcome {
        self.owned.retire_inner()
    }

    /// Compatibility predicate for callers that need only local closure.
    /// Prefer [`try_retire_outcome`](Self::try_retire_outcome) to distinguish
    /// a committed revoke from cleanup after dead lineage.
    pub fn try_retire(&mut self) -> bool {
        self.try_retire_outcome().is_closed()
    }

    /// Whether the executable record remains grantable.
    pub fn is_live(&self) -> bool {
        self.owned.is_live()
    }

    /// Map the complete executable lease R/X at the base of `window`.
    ///
    /// The returned mapping borrows this authority, so executable retirement
    /// cannot race its lexical instruction-fetch lifetime in safe Rust.
    pub fn map_rx_at(
        &self,
        window: kore_abi::task::layout::Window,
    ) -> Result<ExecutableMapping<'_>, ExecutableMapError> {
        let source = self.slot().ok_or(ExecutableMapError::SourceUnavailable)?;
        let (_, needed) =
            verbs::lease_geometry(source).ok_or(ExecutableMapError::GeometryUnavailable)?;
        if needed > window.len() {
            return Err(ExecutableMapError::WindowTooSmall {
                needed,
                available: window.len(),
            });
        }
        let view = verbs::map_owned_lease_at_rx(source, window.base(), needed)
            .ok_or(ExecutableMapError::Refused)?;
        Ok(ExecutableMapping {
            view,
            _code: core::marker::PhantomData,
        })
    }
}

/// One lexical R/X mapping borrowed from [`ExecutableCode`].
#[derive(Debug)]
pub struct ExecutableMapping<'a> {
    view: MappedLeaseView,
    _code: core::marker::PhantomData<&'a ExecutableCode>,
}

impl ExecutableMapping<'_> {
    /// Mapping length in bytes.
    pub const fn len(&self) -> usize {
        self.view.len
    }

    /// Whether the mapping contains no bytes.
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Raw base for instruction entry or immutable metadata inspection.
    pub fn as_ptr(&self) -> *const u8 {
        self.view.as_ptr()
    }

    /// Raw instruction pointer at a checked byte offset.
    pub fn entry_ptr(&self, offset: usize) -> Option<*const u8> {
        (offset < self.len()).then(|| {
            // SAFETY: the checked offset lies within the mapped allocation;
            // pointer creation does not dereference it.
            unsafe { self.as_ptr().add(offset) }
        })
    }

    /// Retire this precise share edge and remove its PTE.
    pub fn try_retire_outcome(&mut self) -> RetireOutcome {
        self.view.try_retire_outcome()
    }

    /// Whether the exact mapping still owns a live local capability name.
    pub fn is_live(&self) -> bool {
        self.view.is_live()
    }
}

/// A **derived interrupt route** — the MSI(-X) completion path a driver
/// establishes off its device cap (the `derive_irq_route` analogue). Holding
/// it is the authority the completion sink is bound to; releasing it
/// (`verbs::release_route`) retires the route — the teardown leg.
// `rec` is the sealed witness payload — the metal verb for this token
// reads it once its kernel arm lands; no reader exists yet.
#[allow(dead_code)]
#[derive(Debug)]
pub struct RouteToken {
    rec: SealedRec,
    owned: OwnedSlot,
    count: u16,
}

impl RouteToken {
    /// The exact held capability name used by route-selecting waits.
    pub(crate) fn slot(&self) -> SlotRef {
        self.owned.slot()
    }

    #[cfg(target_os = "none")]
    pub(crate) fn new(mut owned: OwnedSlot, count: u16) -> Self {
        owned.activate();
        Self {
            rec: SealedRec::possessed(record::RecKind::Route, Rights::ACCESS | Rights::REVOKE),
            owned,
            count,
        }
    }

    /// Number of consecutive device vectors that wake this route.
    #[inline]
    pub const fn count(&self) -> u16 {
        self.count
    }

    pub(crate) fn retire(self) -> bool {
        self.owned.retire()
    }
}

/// A derived **I/O-port grant** (#416) — the authority to execute `in` /
/// `out` against the ports firmware attributed to the granting
/// [`PlatformToken`]'s device.
///
/// **Holding it is the whole mechanism.**  The kernel rebuilds the
/// running CPU's TSS I/O permission bitmap from the *task about to run*
/// on every dispatch, admitting exactly the ranges that task's live
/// grants declare and closing everything else.  Two consequences the
/// caller must design around:
///
/// - **The thread that executes `in`/`out` must be the thread that holds
///   this token.**  Under the unified task model (#407) a thread is a
///   task sharing its parent's address space, and the bitmap is keyed on
///   the *task*, never the space — precisely so a sibling thread that was
///   never granted ports cannot acquire them by sharing an address space
///   with one that was.  A relay worker therefore needs its own grant in
///   its own arena; borrowing the parent's pointer does nothing.
/// - **Revocation is immediate and needs no cooperation.**  The rebuild
///   walks live holdings, so a revoked grant stops contributing at the
///   very next dispatch of the holder; there is no stale-open-port window
///   and nothing to flush.
///
/// The grant's *extent* is firmware's, not the caller's: an arg-free
/// derive takes every range the device's `_CRS` declared and no request
/// shape can reach a port firmware did not attribute to it.  Touching a
/// port outside the grant is a `#GP` on the accessing thread — fail-closed,
/// and loud in the sense that matters: the relay stops and its counters
/// stop advancing, which is visible in the service's status row.
// `rec` is the sealed witness payload — the sibling tokens' posture.
#[allow(dead_code)]
#[derive(Debug)]
pub struct PortGrant {
    rec: SealedRec,
    owned: OwnedSlot,
}

impl PortGrant {
    // Port space is an x86 concept, so the mint stays arch-gated even now
    // that the possession verbs compile on every metal target — see the
    // uninhabitable-off-x86 posture on `read8`/`write8` below.
    #[cfg(all(target_os = "none", target_arch = "x86_64"))]
    pub(crate) fn new(mut owned: OwnedSlot) -> Self {
        owned.activate();
        Self {
            rec: SealedRec::possessed(record::RecKind::Ports, Rights::ACCESS | Rights::REVOKE),
            owned,
        }
    }

    /// Read one byte from `port`.
    ///
    /// `&self` rather than `&mut self` for the same reason
    /// [`Regs`](../../kore_driver/index.html) reads take `&self`: a port
    /// read has device-visible side effects, but the *grant* is not
    /// mutated, and a driver core routinely reads status while holding a
    /// shared borrow.  Serialization is the driver's, as it is for MMIO.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub fn read8(&self, port: u16) -> u8 {
        let value: u8;
        // SAFETY: holding this token means the kernel programmed the
        // running task's I/O bitmap to admit its ranges; a port outside
        // them faults rather than reading someone else's device.
        unsafe {
            core::arch::asm!(
                "in al, dx",
                in("dx") port,
                out("al") value,
                options(nomem, nostack, preserves_flags),
            );
        }
        value
    }

    /// Write one byte to `port`.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub fn write8(&self, port: u16, value: u8) {
        // SAFETY: as `read8` — the bitmap is the bound, and it was
        // programmed from this task's live holdings at dispatch.
        unsafe {
            core::arch::asm!(
                "out dx, al",
                in("dx") port,
                in("al") value,
                options(nomem, nostack, preserves_flags),
            );
        }
    }

    /// I/O port space is an x86 concept: no other architecture kore
    /// targets has `in`/`out` at all.  The type stays present on every
    /// arch — as every sibling token does, so a driver's code is
    /// arch-agnostic — but it is **uninhabitable** off x86: `new` is
    /// gated to the x86 metal target, so no value can reach these.
    #[cfg(not(target_arch = "x86_64"))]
    pub fn read8(&self, _port: u16) -> u8 {
        unreachable!("no PortGrant can be minted on an architecture without port space")
    }

    /// The [`read8`](Self::read8) twin — equally unreachable off x86.
    #[cfg(not(target_arch = "x86_64"))]
    pub fn write8(&self, _port: u16, _value: u8) {
        unreachable!("no PortGrant can be minted on an architecture without port space")
    }

    pub(crate) fn retire(self) -> bool {
        self.owned.retire()
    }
}

/// A mapped **BAR window** — a device register region's base + length, for
/// **volatile** MMIO ([`kore_driver::seam::Regs`](../../kore_driver/index.html)).  Distinct
/// from [`Window`] (safe byte copies over ordinary memory): registers have
/// read/write side effects, so the driver's `Regs` impl accesses `base`
/// with `read_volatile`/`write_volatile` — the mapped MMIO of
/// `claim_bar` + `share_cpu`.
#[derive(Debug)]
pub struct BarWindow {
    base: *mut u8,
    len: u64,
    _owned: OwnedSlot,
}

impl BarWindow {
    #[cfg(target_os = "none")]
    pub(crate) fn new(base: *mut u8, len: u64, mut owned: OwnedSlot) -> Self {
        owned.activate();
        Self {
            base,
            len,
            _owned: owned,
        }
    }
    /// The BAR's CPU-visible base (for volatile MMIO access).
    pub fn base(&self) -> *mut u8 {
        self.base
    }

    /// The BAR window length in bytes.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Whether the window is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Capability parent named by a peer-aperture derive.
    #[cfg(target_os = "none")]
    pub(crate) fn slot(&self) -> SlotRef {
        self._owned.slot()
    }

    /// Explicitly retire this BAR mapping and report whether the kernel
    /// accepted the revoke. `Drop` remains the task-death fallback; device
    /// bindings use this method after their class quiesce hook so teardown
    /// failure is observable instead of silently discarded.
    pub fn retire(self) -> bool {
        self._owned.retire()
    }
}

/// The **image-byte store** — the image service's
/// [`role::IMAGE_STORE`](kore_abi::task::role::IMAGE_STORE) endowment, claimed
/// once from its seeded arena slot.  `Register` carves per-segment extents
/// out of it **in place** (`verbs::carve_image_extent` — `res/IMAGE.md`
/// §2's zero-copy carve); the store itself is never mapped whole and never
/// granted.
#[derive(Debug)]
pub struct ImageStoreToken {
    slot: SlotRef,
    rec: SealedRec,
}

witness_getters!(ImageStoreToken);

impl ImageStoreToken {
    /// The arena slot the claimed record sits in.
    pub fn slot(&self) -> SlotRef {
        self.slot
    }

    /// Absolute base of the store's backing — a service-side face
    /// (`kore-svc-image` converts store-relative carve offsets, never a
    /// client), sealed and holder-readable like every witness field.
    pub fn base(&self) -> u64 {
        self.rec.offset
    }
}

/// A **device-visible DMA sub-window** carved from a driver's DMA pool — a
/// CPU pointer and the [`IovaAddr`](kore_abi::device::IovaAddr) the device sees
/// for the *same* frames (zero-copy, RUNG 2) — the IOVA is the region's
/// device-domain leaf.
#[derive(Debug)]
enum DmaDeviceLeaf {
    Anonymous,
    Named(OwnedSlot),
    Retired,
}

#[derive(Debug)]
pub struct DmaWindow {
    cpu: *mut u8,
    iova: u64,
    len: u64,
    // Drop the child name before the parent containment backstop.
    device_leaf: DmaDeviceLeaf,
    cpu_view: Option<MappedLeaseView>,
    fabric_route: Option<FabricRouteToken>,
    fabric_registration: Option<FabricRegistrationToken>,
    _owned: StashedMemory,
}

impl DmaWindow {
    #[cfg(any(test, target_os = "none"))]
    pub(crate) fn new(cpu: *mut u8, iova: u64, len: u64, owned: OwnedSlot) -> Self {
        // `dma_window` activates as soon as the sub-lease carve commits.  It
        // must do so before the later CPU-map / IOMMU-publish legs because any
        // refusal there must revoke the already-published sub-lease on drop.
        // This constructor receives that live affine owner; activating twice
        // would violate OwnedSlot's Reserved|Armed -> Live transition.
        debug_assert!(owned.is_live());
        Self {
            cpu,
            iova,
            len,
            device_leaf: DmaDeviceLeaf::Anonymous,
            cpu_view: None,
            fabric_route: None,
            fabric_registration: None,
            _owned: StashedMemory::Lease(StashedLease { owned }),
        }
    }

    #[cfg(target_os = "none")]
    pub(crate) fn from_route(cpu: *mut u8, iova: u64, len: u64, route: FabricRouteToken) -> Self {
        Self {
            cpu,
            iova,
            len,
            device_leaf: DmaDeviceLeaf::Anonymous,
            cpu_view: None,
            fabric_route: None,
            fabric_registration: None,
            _owned: StashedMemory::Route(route),
        }
    }

    #[cfg(any(test, target_os = "none"))]
    pub(crate) fn with_device_leaf(mut self, owned: OwnedSlot) -> Self {
        debug_assert!(owned.is_live());
        self.device_leaf = DmaDeviceLeaf::Named(owned);
        self
    }

    #[cfg(any(test, target_os = "none"))]
    pub(crate) fn with_cpu_view(mut self, view: MappedLeaseView) -> Self {
        self.cpu_view = Some(view);
        self
    }

    #[cfg(any(test, target_os = "none"))]
    pub(crate) fn with_fabric_route(
        mut self,
        registration: FabricRegistrationToken,
        route: FabricRouteToken,
    ) -> Self {
        self.fabric_registration = Some(registration);
        self.fabric_route = Some(route);
        self
    }

    /// Borrow the registration retained with this window's executable route.
    /// This is registration authority, not ownership of an allocator subrange.
    pub fn fabric_registration(&self) -> Option<&FabricRegistrationToken> {
        self.fabric_registration.as_ref()
    }

    /// Borrow the route so the window owner can delegate bounded mapping
    /// custody. The provider must separately retain any lent allocator run.
    pub fn fabric_route(&self) -> Option<&FabricRouteToken> {
        self.fabric_route.as_ref().or_else(|| match &self._owned {
            StashedMemory::Route(route) => Some(route),
            _ => None,
        })
    }

    /// Actual parent of the published mappings, distinct from allocation custody.
    fn dma_mapping_slot(&self) -> SlotRef {
        self.fabric_route.as_ref().map_or_else(
            || self._owned.slot().expect("live DMA custody"),
            FabricRouteToken::slot,
        )
    }

    fn try_retire_fabric(&mut self) -> RetireOutcome {
        if let Some(route) = self.fabric_route.take() {
            if let Err(route) = route.try_retire() {
                self.fabric_route = Some(route);
                return RetireOutcome::Retryable;
            }
        }
        if let Some(registration) = self.fabric_registration.take() {
            if let Err(registration) = registration.try_retire() {
                self.fabric_registration = Some(registration);
                return RetireOutcome::Retryable;
            }
        }
        RetireOutcome::Retired
    }

    /// Retire the exact CPU mapping after device access has closed. The
    /// allocation owner remains live for registration cleanup or hand-back.
    /// Anonymous pool mappings cannot prove this separate transition.
    pub fn try_retire_cpu(&mut self) -> RetireOutcome {
        self.retire_cpu_via(crate::cap::verbs::retire_owned_view)
    }

    fn retire_cpu_via(
        &mut self,
        retire: impl FnOnce(SlotRef, SlotRef) -> RetireOutcome,
    ) -> RetireOutcome {
        if !matches!(self.device_leaf, DmaDeviceLeaf::Retired) {
            return RetireOutcome::Retryable;
        }
        let Some(view) = self.cpu_view.as_mut() else {
            return RetireOutcome::Retryable;
        };
        let parent = view.lease;
        let outcome = view.owned.retire_via(|slot| retire(slot, parent));
        if outcome.is_closed() {
            self.cpu = core::ptr::null_mut();
        }
        outcome
    }

    /// Retire only the exact device mapping, retaining CPU access and the
    /// allocation owner for registration cleanup or later hand-back. The
    /// driver must finish dependent device work and stage-1 cleanup first;
    /// refusal is not proof of quiescence. Anonymous pool mappings cannot
    /// prove this separate transition and refuse it.
    pub fn try_retire_device(&mut self) -> RetireOutcome {
        self.retire_device_via(crate::cap::verbs::retire_owned_iova)
    }

    fn retire_device_via(
        &mut self,
        retire: impl FnOnce(SlotRef, SlotRef) -> RetireOutcome,
    ) -> RetireOutcome {
        let parent = self.dma_mapping_slot();
        let outcome = match &mut self.device_leaf {
            DmaDeviceLeaf::Anonymous => return RetireOutcome::Retryable,
            DmaDeviceLeaf::Retired => return RetireOutcome::AlreadyDead,
            DmaDeviceLeaf::Named(owned) => owned.retire_via(|view| retire(view, parent)),
        };
        if outcome.is_closed() {
            self.device_leaf = DmaDeviceLeaf::Retired;
        }
        outcome
    }

    /// The CPU-visible base of the buffer.
    pub fn cpu(&self) -> *mut u8 {
        self.cpu
    }

    /// The device-visible base of the same frames.
    pub fn iova(&self) -> u64 {
        self.iova
    }

    /// The buffer length in bytes.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// The affine sub-lease that owns both CPU and device leaves. Kept
    /// runtime-private: callers activate through the typed verb and never gain
    /// a second raw name for the window authority.
    #[cfg(target_os = "none")]
    fn dma_lease_slot(&self) -> SlotRef {
        self._owned.slot().expect("live DMA custody")
    }

    /// Recover allocation custody only after both named mappings close. The
    /// retained route and registration retire before the lease can transfer.
    /// Anonymous pool windows cannot be detached from their parent this way.
    pub fn try_into_lease(self) -> Result<StashedLease, Self> {
        if !matches!(self._owned, StashedMemory::Lease(_)) {
            return Err(self);
        }
        match self.try_into_memory()? {
            StashedMemory::Lease(lease) => Ok(lease),
            StashedMemory::Route(_) => unreachable!("checked allocation species"),
        }
    }

    /// Return the same allocation or route species after this window's named
    /// mappings retire. A borrowed route remains open for its next receiver.
    pub fn try_into_memory(mut self) -> Result<StashedMemory, Self> {
        if !self.try_retire_device().is_closed() || !self.try_retire_cpu().is_closed() {
            return Err(self);
        }
        if !self.try_retire_fabric().is_closed() {
            return Err(self);
        }
        Ok(self._owned)
    }

    /// Explicitly retire the CPU + IOMMU window and report the revoke result.
    /// The owning driver must quiesce the device before calling this: a failed
    /// quiesce followed by revocation is containment, not graceful shutdown.
    pub fn retire(self) -> bool {
        self.try_retire().is_ok()
    }

    /// Retire the same window while preserving its affine owner on refusal.
    /// Import owners use this to keep the registration/backing cleanup chain
    /// retryable rather than losing the only handle to an outstanding mapping.
    pub fn try_retire(mut self) -> Result<RetireOutcome, Self> {
        if !matches!(self.device_leaf, DmaDeviceLeaf::Anonymous)
            && !self.try_retire_device().is_closed()
        {
            return Err(self);
        }
        if self.cpu_view.is_some() && !self.try_retire_cpu().is_closed() {
            return Err(self);
        }
        if !self.try_retire_fabric().is_closed() {
            return Err(self);
        }
        let outcome = self._owned.try_retire_outcome();
        if outcome.is_closed() {
            Ok(outcome)
        } else {
            Err(self)
        }
    }
}

/// A possessed display-aperture claim ([`role::DISPLAY`](kore_abi::task::role::DISPLAY))
/// — slot-only by design: the map derive re-binds authority by slot, and
/// the geometry the service renders with rides the kernel's
/// `GeomReport`/`DisplayReport` answers, never the record.
#[derive(Debug)]
pub struct DisplayToken {
    slot: SlotRef,
}

impl DisplayToken {
    #[cfg(target_os = "none")]
    pub(crate) fn possess(slot: SlotRef) -> Self {
        Self { slot }
    }

    /// The arena slot the claim sits in — the wire name the map derive
    /// re-binds by.
    pub fn slot(&self) -> SlotRef {
        self.slot
    }
}

/// Possession of the founder-delivered **system power** grant (#392) — the
/// `role::POWER` endowment, claimed out of the boot block.
///
/// Slot-only, like [`DisplayToken`]: what the service needs at runtime is the
/// arena name to derive against. Rights are deliberately **not** mirrored
/// here — the `MODIFY` gate is enforced kernel-side at the derive, so a
/// userland copy would be advisory at best and a second source of truth at
/// worst.
#[derive(Debug)]
pub struct PowerToken {
    slot: SlotRef,
}

/// Possession of the monitor service's read-only machine-observation grant.
#[derive(Debug)]
pub struct ObserveToken {
    slot: SlotRef,
}

impl ObserveToken {
    #[cfg(target_os = "none")]
    pub(crate) fn possess(slot: SlotRef) -> Self {
        Self { slot }
    }

    pub fn slot(&self) -> SlotRef {
        self.slot
    }
}

impl PowerToken {
    #[cfg(target_os = "none")]
    pub(crate) fn possess(slot: SlotRef) -> Self {
        Self { slot }
    }

    /// The arena slot the claim sits in — the wire name the transition derive
    /// re-binds by.
    pub fn slot(&self) -> SlotRef {
        self.slot
    }
}

// One generic derive seam over the existing, independently documented claim
// verbs. The public token type determines the verb; the endowment field still
// determines its contextual boot role. Do not move roles into this table:
// SupplyToken alone lawfully serves MEM_SUPPLY, DMA_POOL, and FACE_SUPPLY.
macro_rules! impl_boot_claim {
    ($($token:ty => $claim:path;)*) => {
        $(
            impl crate::service::boot_claim_sealed::Sealed for $token {}

            impl crate::service::BootClaim for $token {
                #[inline]
                fn claim_boot(slot: SlotRef) -> Option<Self> {
                    $claim(slot)
                }
            }
        )*
    };
}

impl_boot_claim! {
    RegionToken => verbs::claim_region;
    ExecToken => verbs::claim_exec;
    VirtMemoryToken => verbs::claim_virt_memory;
    XMintToken => verbs::claim_xmint;
    RootComplexToken => verbs::claim_root_complex;
    PlatformBusToken => verbs::claim_platform_bus;
    HypervisorToken => verbs::claim_hypervisor;
    SupplyToken => verbs::claim_supply;
    ChannelSpaceToken => verbs::claim_channel_space;
    ImageStoreToken => verbs::claim_image_store;
    DeviceToken => verbs::claim_device;
    PlatformToken => verbs::claim_platform_device;
    DisplayToken => verbs::claim_display;
    PowerToken => verbs::claim_power;
    ObserveToken => verbs::claim_observe;
}

/// What a recv's landing slot yielded to [`verbs::take_delivery`] — the
/// probe-and-consume face `ChanRx::recv` reads (`res/USERLAND.md` §5.2).
/// Ungated — the shape is target-independent; the metal reads construct it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RawDelivery {
    /// A bind-inert doorbell notice `(cookie, payload)` — consumed out
    /// of the slot.
    Notice(u64, u64),
    /// A delivered capability record — left in the slot; the holder
    /// takes it through the possession verbs (`take_view` et al.).
    Cap,
}

mod record;

pub mod verbs;

#[cfg(test)]
mod retirement_tests {
    use super::*;

    fn live_slot(index: u32) -> OwnedSlot {
        OwnedSlot {
            slot: SlotRef::from_wire(index),
            state: OwnedSlotState::Live,
            _thread_bound: core::marker::PhantomData,
        }
    }

    fn reserved_slot(index: u32) -> OwnedSlot {
        OwnedSlot {
            slot: SlotRef::from_wire(index),
            state: OwnedSlotState::Reserved,
            _thread_bound: core::marker::PhantomData,
        }
    }

    fn staged_reservation(index: u32) -> StagedCapabilityReservation {
        StagedCapabilityReservation {
            owned: OwnedSlot {
                slot: SlotRef::from_wire(index),
                state: OwnedSlotState::Reserved,
                _thread_bound: core::marker::PhantomData,
            },
        }
    }

    fn live_executable_code() -> ExecutableCode {
        ExecutableCode {
            owned: live_slot(41),
        }
    }

    fn mapped_test_view(bytes: &mut [u8]) -> MappedLeaseView {
        MappedLeaseView {
            owned: core::mem::ManuallyDrop::new(OwnedSlot {
                slot: SlotRef::from_wire(42),
                state: OwnedSlotState::Released,
                _thread_bound: core::marker::PhantomData,
            }),
            lease: SlotRef::from_wire(43),
            base: core::ptr::NonNull::new(bytes.as_mut_ptr()).expect("test buffer has a base"),
            len: bytes.len(),
        }
    }

    fn writable_test_code(bytes: &mut [u8]) -> WritableCode {
        WritableCode {
            view: mapped_test_view(bytes),
            lease: StashedLease {
                owned: live_slot(44),
            },
        }
    }

    #[test]
    fn guest_space_token_keeps_one_live_dynamic_name() {
        let guest = GuestSpaceToken::new(reserved_slot(49));

        assert_eq!(guest.slot(), SlotRef::from_wire(49));
        assert_eq!(guest.rights(), GUEST_SPACE_RIGHTS);
        assert_eq!(guest.owned.state, OwnedSlotState::Live);

        // Host transport cannot prove a kernel revoke, so explicit retirement
        // must return the same affine owner instead of reopening its name.
        let mut guest = guest
            .try_retire()
            .expect_err("host retirement remains retryable");
        assert_eq!(guest.owned.state, OwnedSlotState::Live);
        guest.owned.release_reservation();
    }

    #[test]
    fn guest_region_token_keeps_one_live_dynamic_name() {
        let rights = GUEST_REGION_RW_RIGHTS;
        let region = GuestRegionToken::new(reserved_slot(50), rights);

        assert_eq!(region.slot(), SlotRef::from_wire(50));
        assert_eq!(region.rights(), rights);
        assert_eq!(region.owned.state, OwnedSlotState::Live);

        // Retiring the public name does not tear down the physical mapping;
        // that lifetime remains with the parent lease and guest space.
        let mut region = region
            .try_retire()
            .expect_err("host retirement remains retryable");
        assert_eq!(region.owned.state, OwnedSlotState::Live);
        region.owned.release_reservation();
    }

    #[test]
    fn dma_window_accepts_the_already_live_carve_owner() {
        let window = DmaWindow::new(core::ptr::null_mut(), 0x4000, 0x1000, live_slot(43));
        assert!(window._owned.slot().is_some());
        core::mem::forget(window);
    }

    #[test]
    fn named_dma_child_retirement_preserves_parent_and_retry_authority() {
        let mut window = DmaWindow::new(core::ptr::null_mut(), 0x4000, 0x1000, live_slot(44))
            .with_device_leaf(live_slot(45));
        assert_eq!(
            window.retire_device_via(|view, parent| {
                assert_eq!(view, SlotRef::from_wire(45));
                assert_eq!(parent, SlotRef::from_wire(44));
                RetireOutcome::Retryable
            }),
            RetireOutcome::Retryable
        );
        assert!(window._owned.slot().is_some());
        assert!(matches!(&window.device_leaf, DmaDeviceLeaf::Named(owner) if owner.is_live()));
        assert_eq!(
            window.retire_device_via(|_, _| RetireOutcome::Retired),
            RetireOutcome::Retired
        );
        assert!(
            window._owned.slot().is_some(),
            "device retirement must retain allocation custody"
        );
        assert_eq!(
            window.retire_device_via(|_, _| panic!("already retired")),
            RetireOutcome::AlreadyDead
        );
        window
            ._owned
            .into_returned_capability()
            .owned
            .release_reservation();
        let mut anonymous = DmaWindow::new(core::ptr::null_mut(), 0x5000, 0x1000, live_slot(46));
        assert_eq!(anonymous.try_retire_device(), RetireOutcome::Retryable);
        assert!(anonymous._owned.slot().is_some());
        anonymous
            ._owned
            .into_returned_capability()
            .owned
            .release_reservation();
    }

    #[test]
    fn routed_dma_retirement_uses_route_parent_and_keeps_allocation_on_refusal() {
        let route = FabricRouteToken {
            owned: live_slot(47),
        };
        let registration = FabricRegistrationToken {
            owned: live_slot(48),
        };
        let mut window = DmaWindow::new(core::ptr::null_mut(), 0x4000, 0x1000, live_slot(44))
            .with_device_leaf(live_slot(45))
            .with_fabric_route(registration, route);
        assert_eq!(
            window.retire_device_via(|view, parent| {
                assert_eq!(view, SlotRef::from_wire(45));
                assert_eq!(
                    parent,
                    SlotRef::from_wire(47),
                    "the allocation is not the IOVA parent"
                );
                RetireOutcome::Retryable
            }),
            RetireOutcome::Retryable
        );
        assert!(window._owned.slot().is_some());
        assert_eq!(
            window.try_retire_fabric(),
            RetireOutcome::Retryable,
            "host transport cannot prove route closure"
        );
        assert!(window.fabric_route.as_ref().unwrap().owned.is_live());
        assert!(window.fabric_registration.as_ref().unwrap().owned.is_live());
        core::mem::forget(window);
    }

    #[test]
    fn named_dma_cpu_retirement_requires_device_closure_and_retains_allocation() {
        let mut bytes = [0u8; 8];
        let mut view = mapped_test_view(&mut bytes);
        view.owned = core::mem::ManuallyDrop::new(live_slot(42));
        let mut window = DmaWindow::new(bytes.as_mut_ptr(), 0x4000, 8, live_slot(43))
            .with_device_leaf(live_slot(45))
            .with_cpu_view(view);
        assert_eq!(
            window.retire_cpu_via(|_, _| panic!("device still live")),
            RetireOutcome::Retryable
        );
        assert_eq!(
            window.retire_device_via(|_, _| RetireOutcome::Retired),
            RetireOutcome::Retired
        );
        assert_eq!(
            window.retire_cpu_via(|cpu, parent| {
                assert_eq!(cpu, SlotRef::from_wire(42));
                assert_eq!(parent, SlotRef::from_wire(43));
                RetireOutcome::Retryable
            }),
            RetireOutcome::Retryable
        );
        assert_eq!(window.cpu(), bytes.as_mut_ptr());
        assert!(window._owned.slot().is_some());
        assert_eq!(
            window.retire_cpu_via(|_, _| RetireOutcome::Retired),
            RetireOutcome::Retired
        );
        assert!(window.cpu().is_null());
        assert!(
            window._owned.slot().is_some(),
            "allocation remains available for hand-back"
        );
        assert_eq!(
            window.retire_cpu_via(|_, _| panic!("already retired")),
            RetireOutcome::AlreadyDead
        );
        let mut lease = window
            .try_into_lease()
            .expect("both named children are closed");
        assert_eq!(lease.slot(), Some(SlotRef::from_wire(43)));
        assert!(lease.is_live());
        lease.owned.release_reservation();
    }

    fn returned_task(returned: ReturnedCapability) -> TaskToken {
        let slot = returned.slot().expect("test owner is live");
        TaskToken {
            slot,
            id: 0xfeed,
            rec: SealedRec {
                kind: record::RecKind::TaskHandle,
                rights: Rights::ACCESS | Rights::REVOKE,
                holder: 0,
                tag: 0,
                offset: 0,
                len: 0,
                class: DmaClass::None,
                mac: 0,
            },
            _returned: Some(returned),
        }
    }

    #[test]
    fn dead_lineage_closes_only_the_owned_local_name() {
        let mut owned = live_slot(40);

        assert_eq!(
            owned.settle_retirement(RetireOutcome::AlreadyDead),
            RetireOutcome::AlreadyDead
        );
        assert_eq!(owned.state, OwnedSlotState::Released);
    }

    #[test]
    fn retryable_retirement_keeps_the_slot_quarantined() {
        let mut owned = live_slot(40);

        assert_eq!(
            owned.settle_retirement(RetireOutcome::Retryable),
            RetireOutcome::Retryable
        );
        assert_eq!(owned.state, OwnedSlotState::Live);
    }

    #[test]
    fn species_specific_retry_keeps_the_owned_slot_quarantined() {
        let mut owned = live_slot(46);
        assert_eq!(
            owned.retire_via(|slot| {
                assert_eq!(slot, SlotRef::from_wire(46));
                RetireOutcome::Retryable
            }),
            RetireOutcome::Retryable,
            "a specialized refusal must retain the same affine owner"
        );
        assert_eq!(owned.state, OwnedSlotState::Live);

        assert_eq!(
            owned.retire_via(|_| RetireOutcome::Retired),
            RetireOutcome::Retired
        );
        assert_eq!(owned.state, OwnedSlotState::Released);
    }

    #[test]
    fn proven_outward_move_releases_only_the_local_reservation() {
        let mut owned = live_slot(40);

        owned.release_after_proven_move();
        assert_eq!(owned.state, OwnedSlotState::Released);
        // Drop is now a no-op: it must not try to revoke the sender record the
        // successful rendezvous already scrubbed.
        drop(owned);
    }

    #[test]
    fn staged_transfer_reopens_only_after_a_proven_completion() {
        let mut reservation = staged_reservation(40);
        reservation.begin_transfer();
        assert_eq!(reservation.owned.state, OwnedSlotState::InFlight);

        reservation.settle_proven();
        assert_eq!(reservation.owned.state, OwnedSlotState::Released);
        drop(reservation);
    }

    #[test]
    fn indeterminate_staged_transfer_keeps_its_allocator_tombstone() {
        let mut reservation = staged_reservation(40);
        reservation.begin_transfer();
        reservation.quarantine();
        assert_eq!(reservation.owned.state, OwnedSlotState::Indeterminate);
        assert_eq!(
            reservation.owned.retire_inner(),
            RetireOutcome::Retryable,
            "unknown occupancy must never reopen or revoke the slot"
        );
        core::mem::forget(reservation);
    }

    #[test]
    fn channel_mint_promotes_all_three_results_only_as_typed_owners() {
        let pair =
            ChannelPair::from_reserved(reserved_slot(46), reserved_slot(47), reserved_slot(48));
        let (root, send, recv) = pair.into_parts();

        assert!(root.is_live());
        assert!(send.is_live());
        assert!(recv.is_live());
        assert_eq!(root.slot(), SlotRef::from_wire(46));
        assert_eq!(send.slot(), SlotRef::from_wire(47));
        assert_eq!(recv.slot(), SlotRef::from_wire(48));
        assert_eq!(root.rights(), CHANNEL_ROOT_RIGHTS);
        assert_eq!(send.rights(), CHANNEL_ENDPOINT_RIGHTS);
        assert_eq!(recv.rights(), CHANNEL_ENDPOINT_RIGHTS);

        // Model successful endpoint conveys: each typed owner releases only
        // its own now-vacant dynamic arena reservation.
        send.release_after_proven_move();
        recv.release_after_proven_move();

        // Host transport refuses the explicit root revoke and must return the
        // same live retry witness. There is no implicit Drop revoke.
        let mut root = root
            .try_retire()
            .expect_err("host root retirement remains retryable");
        assert!(root.is_live());
        // Test-only cleanup of the synthetic owner (no real allocator bit or
        // kernel capability exists on the host).
        root.owned.release_reservation();
        // SAFETY: Released makes the synthetic OwnedSlot drop a no-op.
        unsafe { core::mem::ManuallyDrop::drop(&mut root.owned) };
    }

    #[test]
    fn channel_mint_preserves_asymmetric_endpoint_attenuation() {
        let child_rights = Rights::ACCESS.union(Rights::REVOKE);
        let pair = ChannelPair::from_reserved_with_rights(
            reserved_slot(49),
            reserved_slot(50),
            child_rights,
            reserved_slot(51),
            CHANNEL_ENDPOINT_RIGHTS,
        );
        let (mut root, send, recv) = pair.into_parts();

        assert_eq!(send.rights(), child_rights);
        assert!(!send.rights().contains(Rights::DERIVE));
        assert_eq!(recv.rights(), CHANNEL_ENDPOINT_RIGHTS);

        send.release_after_proven_move();
        recv.release_after_proven_move();
        root.owned.release_reservation();
        // SAFETY: Released makes the synthetic OwnedSlot drop a no-op.
        unsafe { core::mem::ManuallyDrop::drop(&mut root.owned) };
    }

    #[test]
    fn ambiguous_channel_mint_quarantines_every_reserved_name() {
        let mut root = reserved_slot(46);
        let mut send = reserved_slot(47);
        let mut recv = reserved_slot(48);
        root.quarantine_unknown_publish();
        send.quarantine_unknown_publish();
        recv.quarantine_unknown_publish();

        assert_eq!(root.state, OwnedSlotState::Indeterminate);
        assert_eq!(send.state, OwnedSlotState::Indeterminate);
        assert_eq!(recv.state, OwnedSlotState::Indeterminate);
        core::mem::forget((root, send, recv));
    }

    #[test]
    fn retained_lease_retyping_preserves_the_affine_owner() {
        let returned = ReturnedCapability::test_live(SlotRef::from_wire(43));

        let lease = returned.into_stashed_lease_after_check();
        assert_eq!(lease.slot(), Some(SlotRef::from_wire(43)));

        let returned = lease.into_returned_capability();
        assert_eq!(returned.slot(), Some(SlotRef::from_wire(43)));
        core::mem::forget(returned);
    }

    #[test]
    fn retained_send_erasure_preserves_the_affine_owner() {
        let rights = Rights::ACCESS | Rights::REVOKE;
        let send = ChanSendToken::new_with_rights(reserved_slot(52), rights);
        assert_eq!(send.slot(), SlotRef::from_wire(52));
        assert_eq!(send.rights(), rights);

        let returned = send.into_returned_capability();
        assert_eq!(returned.slot(), Some(SlotRef::from_wire(52)));
        assert!(returned.is_live());
        core::mem::forget(returned);
    }

    #[test]
    fn exact_returned_kind_check_rejects_both_under_and_over_grants() {
        use kore_abi::cap::{CapRecord, ResourceKind};

        let expected = Rights::ACCESS | Rights::REVOKE;
        let mut record = CapRecord::vacant();
        record.kind = ResourceKind::ChanSend as u16;
        record.rights = expected;
        assert!(record_has_exact_kind_and_rights(
            &record,
            ResourceKind::ChanSend,
            expected
        ));

        record.rights = Rights::ACCESS;
        assert!(!record_has_exact_kind_and_rights(
            &record,
            ResourceKind::ChanSend,
            expected
        ));
        record.rights = expected | Rights::DERIVE;
        assert!(!record_has_exact_kind_and_rights(
            &record,
            ResourceKind::ChanSend,
            expected
        ));
        record.rights = expected;
        assert!(!record_has_exact_kind_and_rights(
            &record,
            ResourceKind::ChanRecv,
            expected
        ));
    }

    #[test]
    fn retryable_task_kill_returns_the_same_live_owner() {
        let task = returned_task(ReturnedCapability::test_live(SlotRef::from_wire(44)));

        // Host transport cannot commit a revoke. The important result is not
        // merely `Retryable`: ownership must come back to the caller intact.
        let task = match task.kill() {
            Err(task) => task,
            Ok(outcome) => panic!("host kill unexpectedly closed: {outcome:?}"),
        };
        assert_eq!(task.id(), 0xfeed);
        assert_eq!(task.slot(), SlotRef::from_wire(44));
        assert!(
            task._returned
                .as_ref()
                .is_some_and(ReturnedCapability::is_live)
        );
        // Avoid Drop's deliberate second best-effort retirement attempt: this
        // test is about the explicit call preserving its retry witness.
        core::mem::forget(task);
    }

    #[test]
    fn closed_task_kill_consumes_the_affine_owner() {
        let mut returned = ReturnedCapability::test_live(SlotRef::from_wire(45));
        assert_eq!(
            returned.owned.settle_retirement(RetireOutcome::AlreadyDead),
            RetireOutcome::AlreadyDead
        );
        let task = TaskToken {
            slot: SlotRef::from_wire(45),
            id: 0xbeef,
            rec: SealedRec {
                kind: record::RecKind::TaskHandle,
                rights: Rights::ACCESS | Rights::REVOKE,
                holder: 0,
                tag: 0,
                offset: 0,
                len: 0,
                class: DmaClass::None,
                mac: 0,
            },
            _returned: Some(returned),
        };

        assert!(matches!(task.kill(), Ok(RetireOutcome::AlreadyDead)));
    }

    #[test]
    fn executable_code_owns_one_live_name() {
        let code = live_executable_code();
        assert_eq!(code.slot(), Some(SlotRef::from_wire(41)));
        assert!(code.is_live());
        core::mem::forget(code);
    }

    #[test]
    fn code_writer_bounds_every_raw_mapping_write() {
        let mut bytes = [0_u8; 8];
        let mut view = mapped_test_view(&mut bytes);
        let mut writer = CodeWriter { view: &mut view };

        writer.fill(0xcc);
        writer
            .write_at(2, &[1, 2, 3])
            .expect("in-bounds emission succeeds");
        assert_eq!(writer.len(), 8);
        assert!(!writer.is_empty());
        assert_eq!(bytes, [0xcc, 0xcc, 1, 2, 3, 0xcc, 0xcc, 0xcc]);
        assert_eq!(
            writer.write_at(7, &[4, 5]),
            Err(CodeWriteError::OutOfBounds {
                offset: 7,
                len: 2,
                capacity: 8,
            })
        );
        assert_eq!(bytes, [0xcc, 0xcc, 1, 2, 3, 0xcc, 0xcc, 0xcc]);
    }

    #[test]
    fn initializer_error_returns_the_partially_written_affine_state() {
        let mut bytes = [0_u8; 8];
        let code = writable_test_code(&mut bytes);

        let failure = code
            .initialize(|writer| {
                writer.write_at(1, &[7, 8, 9]).expect("write fits");
                Err::<(), _>("emitter stopped")
            })
            .expect_err("closure refusal returns writable ownership");
        let (mut code, error) = failure.into_parts();
        assert_eq!(error, "emitter stopped");
        assert_eq!(code.len(), bytes.len());
        assert_eq!(bytes, [0, 7, 8, 9, 0, 0, 0, 0]);

        code.lease.owned.release_reservation();
    }

    #[test]
    fn precommit_finalize_refusal_returns_mapping_and_lease_together() {
        let mut bytes = [0_u8; 8];
        let code = writable_test_code(&mut bytes);

        let mut code = match code.finalize() {
            Err(FinalizeCodeError::Refused(code)) => code,
            other => panic!("host finalization must refuse, got {other:?}"),
        };
        assert_eq!(code.len(), bytes.len());
        assert!(code.view.is_live() == false);
        assert!(code.lease.is_live());

        code.lease.owned.release_reservation();
    }

    #[test]
    fn writable_map_failure_returns_the_exact_source_lease() {
        let lease = StashedLease {
            owned: live_slot(45),
        };
        let failure = lease
            .into_writable_code_at(kore_abi::task::layout::Window::new(0x1000, 0x1000))
            .expect_err("host geometry query fails closed");
        assert_eq!(
            failure.kind(),
            WritableCodeMapErrorKind::GeometryUnavailable
        );
        let mut lease = failure.into_lease();
        assert_eq!(lease.slot(), Some(SlotRef::from_wire(45)));
        lease.owned.release_reservation();
    }
}
