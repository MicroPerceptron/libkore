//! `record` — the userland-side **sealed capability record**: the plain
//! data the witness tokens of [`cap`](super) stand over.
//!
//! This is a holder-readable shadow of the kernel-owned sealed record.
//! Userland never mints, binds, or retires it: metal possession reads cache
//! only the metadata needed by the typed token, while every effect names the
//! arena slot again and the kernel performs the authoritative bind.

use kore_abi::cap::Rights;
use kore_abi::mem::DmaClass;

/// What a sealed record stands over.  The seeded → claimed transition
/// is **record content** (MAC-bound), not registry state: claiming
/// retires the `Seeded` record's tag and seals a claimed kind in its
/// place, so one-time possession falls out of liveness + kind.
// Constructed only by the metal possession verbs (target-gated), so on
// host builds the variants count as unused.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RecKind {
    /// Spawn-seeded, not yet claimed — binds only to `claim_supply`.
    Seeded,
    /// A claimed supply — the only kind `execute` grants from.
    Supply,
    Extent,
    Lease,
    Leased,
    /// A mapped region (`map` consumed an extent; `unmap` restores it).
    Mapped,
    /// Boot-seeded exec authority, not yet claimed — binds only to
    /// `claim_exec` (disjoint from `Seeded` so a supply handle can
    /// never claim as exec authority, nor the reverse).
    SeededExec,
    /// The claimed task-founding authority.
    Exec,
    /// An attenuated task handle (wait / start / kill / self-exit rights).
    TaskHandle,
    /// Spawn-seeded device capability, not yet claimed — binds only to
    /// `claim_device` (disjoint from `Seeded`/`SeededExec`, so a memory
    /// or exec handle can never claim as a device, nor the reverse).
    SeededDevice,
    /// A claimed device capability — the driver's `role::DEVICE`
    /// endowment; `map_bar` and `establish_interrupt` act on it.
    Device,
    /// A derived interrupt route (the `derive_irq_route` analogue).
    Route,
    /// Spawn-seeded image store, not yet claimed — binds only to
    /// `claim_image_store` (kind-disjoint like every seed: a memory
    /// / exec / device handle can never claim as the store, nor the
    /// reverse).
    SeededImage,
    /// The claimed image-byte store — the image service's
    /// `role::IMAGE_STORE` endowment; `carve_image_extent` carves
    /// per-segment extents out of it in place.
    ImageStore,
    /// Boot-seeded X-mint, not yet claimed — binds only to
    /// `claim_xmint`.
    SeededXMint,
    /// The claimed X-mint (`res/IMAGE.md` §6) — authority for requesting an
    /// executable lease at initial allocation. The later consuming
    /// writable-to-executable transition is authorized by the lease itself.
    XMint,
    /// An executable-only lease (either initially allocated under an X-mint
    /// or produced by consuming a full-rights writable lease) —
    /// shares off it may carry `R|X`, and `MODIFY` is refused
    /// unconditionally (`share` / `execute_share`): immutable
    /// text with no unprotect path, a property of the *record*, not
    /// caller discipline.
    ExecLease,
    /// Boot-seeded `VirtMemory` allocator, not yet claimed — binds only to
    /// `claim_virt_memory` (kind-disjoint like every seed).
    SeededVirtMemory,
    /// The claimed `VirtMemory` (`res/IMAGE.md` §9e) — the exec manager's
    /// space-allocator authority; a staged `Derive` produces fresh
    /// spaces without consuming it.
    VirtMemory,
    /// A **scratch reservation** — a store slot reserved as a staged
    /// session's result destination. It binds to nothing; the kernel may
    /// overwrite it with the committed result.
    Reserved,
    /// A held, revocable grant of the rendezvous allocator facility.
    ChannelSpace,
    /// The root of one rendezvous endpoint pair.
    Channel,
    /// A channel **send end** — the doorbell kick authority.
    ChanSend,
    /// A channel **recv end** — the standing-post authority.
    ChanRecv,
    /// A delivered **doorbell notice** (`res/USERLAND.md` §3.1) — the
    /// bind-inert record a notice-mode send writes into the posted slot:
    /// `tag == 0` (never live, so it can never bind — the userland
    /// shadow of the kernel's `cid == NULL`), `offset` = cookie,
    /// `len` = payload.  Consumed by `verbs::take_notice`.
    Notice,
    /// A claimed **platform-device** capability (#417) — the
    /// [`Device`](Self::Device) twin for a firmware-described non-PCI
    /// device.  Kind-disjoint from `Device` on purpose: the two species
    /// support different derives (ports and a legacy line vs. BARs and
    /// MSI), and a driver handed the wrong one must refuse at the
    /// possession verb rather than at the first derive.
    PlatformDevice,
    /// A derived **I/O-port grant** (#416) — the authority whose
    /// presence in the *running task's* holdings is what opens ports in
    /// that CPU's TSS permission bitmap.
    Ports,
    /// The claimed **root complex** (`res/DEVICE_SERVICE.md` §10) — a
    /// broker row's `role::ROOT_COMPLEX` endowment, off which one of that
    /// root's entries is read (`verbs::device_record`) or handed out
    /// (`verbs::claim_stored_device`).
    ///
    /// Kind-disjoint from [`Device`](Self::Device) and
    /// [`PlatformDevice`](Self::PlatformDevice) for the same reason those
    /// two are disjoint from each other, one level up: a task holding the
    /// *root* can hand out any of its devices, and a task holding a
    /// *device* can drive it — confusing the two would let a driver's own
    /// capability bind as a root's inventory.
    RootComplex,
    /// The claimed **memory-region grant** (`role::MEM_REGION`, slice D)
    /// — the lessor's minting authority, off which `OP_SUPPLY` stages
    /// `LeaseReq` derives for fresh root supply leases.
    ///
    /// Kind-disjoint from [`Supply`](Self::Supply) on purpose: a supply
    /// is one already-minted lease whose extent-*views* mm carves out,
    /// while the region grant mints new sibling *leases* whose frames
    /// live and die against the region bitmap — confusing the two would
    /// let a carve cursor and the region allocator overlap.
    MemRegion,
    /// The claimed **platform-device inventory** — the
    /// [`RootComplex`](Self::RootComplex) sibling for firmware-described
    /// (non-PCI) devices (`role::PLATFORM_BUS`, broker row 0 only).
    /// Kind-disjoint from [`RootComplex`](Self::RootComplex) because the
    /// two authorities vend disjoint species with disjoint claim
    /// consequences (#572).
    PlatformBus,
    /// The claimed machine virtualization facility endowment.
    Hypervisor,
    /// One runtime-minted coherent guest-physical address space.
    GuestSpace,
    /// One lease-backed window installed in a guest-physical address space.
    GuestRegion,
}

/// The sealed capability record — the userland-side **shadow** of a
/// delivered [`CapRecord`](kore_abi::cap::CapRecord) (see the impl
/// below).  **Plain, copyable data**: copying it is possible and
/// useless (same tag, same holder stamp — see the seam docs).  Fields
/// are holder-readable; authority is only ever re-earned at the
/// kernel's `bind`.
// Constructed only by the metal possession verbs (target-gated), so on
// host builds the fields count as unused.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SealedRec {
    pub(crate) kind: RecKind,
    pub(crate) rights: Rights,
    /// D1 holder stamp — who may bind this record.
    pub(crate) holder: u16,
    /// Revocation identity — one liveness bit, never reused.
    pub(crate) tag: u32,
    /// Absolute resource-relative offset cached for holder-readable metadata.
    pub(crate) offset: u64,
    pub(crate) len: u64,
    /// DMA class the backing satisfies — a *placement* property mm
    /// honored; reachability is the device-domain holder's act.
    pub(crate) class: DmaClass,
    pub(crate) mac: u64,
}

impl SealedRec {
    /// The userland-side sealed **shadow** of a *delivered* record read at
    /// the R/O `SLOTS_VA` alias (Phase-A checkpoint C): the metal
    /// possession verbs carry the delivered `rights` through (the one field
    /// the token's `rights()` getter reads off the record), while geometry
    /// rides the token's own cached field — the delivered
    /// [`CapRecord`](kore_abi::cap::CapRecord) has no offset/len, and
    /// `holder`/`tag`/`mac` are **inert** here: the kernel re-authenticates
    /// the real record *by slot* at the use-trap and never binds this
    /// shadow (userland holds no epoch key). This also describes a freshly
    /// committed dynamic result whose exact kind/rights are fixed by the
    /// session verb: authority still resides only in the arena slot, never in
    /// this host-compilable metadata value.
    pub(crate) fn possessed(kind: RecKind, rights: Rights) -> Self {
        Self::inert(kind, rights)
    }

    /// The same inert shadow, reachable off the metal target — the host
    /// half of the possession shape, so a service's host tests can build
    /// a geometry-only endowment (`SupplyToken::for_test`).  Inert either
    /// way: no verb accepts it, because off-target every verb refuses.
    #[allow(dead_code)]
    pub(crate) fn inert(kind: RecKind, rights: Rights) -> Self {
        Self {
            kind,
            rights,
            holder: 0,
            tag: 0,
            offset: 0,
            len: 0,
            class: DmaClass::None,
            mac: 0,
        }
    }
}
