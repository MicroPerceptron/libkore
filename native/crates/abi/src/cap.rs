//! Capability ABI — the wire identity, rights, kinds, and the stamped slot
//! record that lives in a task's shared cap arena.
//!
//! These are *layouts and pure readers*. Minting and verifying a record's
//! stamp needs the kernel epoch key and lives kernel-side; kore-abi can read a
//! record's fields, never forge or check one.

use bitflags::bitflags;

/// Max derivation depth carried in a cap's MAC-bound ancestor chain
/// (SYSCALL.md / KRAPH). A chain at this depth cannot derive further.
/// TODO: Remove all hardcoded caps in `kore-abi` and use `allocator_api` backed DS.
pub const MAX_DEPTH: usize = 16;

/// Monotone revocation tag — globally unique within an epoch. Retiring a
/// tag (kernel-side) revokes the cap and, by closure, its whole subtree.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Tag(pub u32);

/// Capability identity: a resource catalog id (high 32) interleaved with a
/// revocation [`Tag`] (low 32). `Cid(0)` is the reserved null identity —
/// "no parent" for a primordial root, or "no holder" for an unbound cap.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Cid(pub u64);

impl Cid {
    /// The reserved null identity.
    pub const NULL: Self = Self(0);

    #[inline]
    pub const fn new(resource: u32, tag: Tag) -> Self {
        Self((resource as u64) << 32 | tag.0 as u64)
    }

    /// The resource catalog id.
    #[inline]
    pub const fn resource(self) -> u32 {
        (self.0 >> 32) as u32
    }

    /// The revocation tag.
    #[inline]
    pub const fn tag(self) -> Tag {
        Tag(self.0 as u32)
    }

    #[inline]
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }
}

bitflags! {
    /// The authority bits a capability may carry. Four are defined; four are
    /// reserved. A derive may only request a subset of its parent's rights
    /// (attenuation).
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
    pub struct Rights: u8 {
        /// Read projection of the resource.
        const ACCESS = 1 << 0;
        /// Write projection of the resource.
        const MODIFY = 1 << 1;
        /// Create attenuated child capabilities.
        const DERIVE = 1 << 2;
        /// Retire the cap (and, by closure, its subtree).
        const REVOKE = 1 << 3;
    }
}

/// What kind of physical or runtime resource a capability authorizes.
/// `#[repr(u16)]` pins a stable on-wire discriminant; the order mirrors the
/// kernel's `schema::ResourceKind` (the single source of truth, this crate).
#[repr(u16)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ResourceKind {
    // Platform — boot-discovered, immutable after boot.
    LogicalCpu = 0,
    PhysRegion,
    PciDevice,
    IrqController,
    Timer,
    Serial,
    Iommu,
    VirtMemory,
    // Runtime — emergent via derivation.
    AddrSpace,
    PhysExtent,
    Region,
    IovaRegion,
    Subpool,
    /// A mixed-size sub-page heap carved from a `PhysExtent` (#20) —
    /// `Subpool`'s sibling with a TLSF engine inside.
    Heap,
    MemLease,
    LeasedRegion,
    CpuShare,
    /// The channel-space allocator facility — the primordial (`VirtMemory`-shaped)
    /// root a [`Channel`](Self::Channel) derives from.
    ChannelSpace,
    /// A rendezvous channel (KRAPH §12.5) — zero physical backing; the
    /// common parent of a send/recv endpoint pair.
    Channel,
    /// The send endpoint of a channel: presenting it in a `Rendezvous`
    /// invocation transfers a capability to the peer's posted slot.
    ChanSend,
    /// The recv endpoint of a channel: presenting it in a `Rendezvous`
    /// invocation posts a target slot for the next transfer.
    ChanRecv,
    /// The primordial system power-control facility (`VirtMemory`/`ChannelSpace`
    /// shape): the machine-wide reboot / power-off authority, gated `MODIFY`
    /// to effect a transition and `ACCESS` to read supported backends.
    SystemPower,
    /// A claimed device **MMIO region** (#85) — one device BAR window mapped
    /// into one or more address spaces, zero-copy. Derived from the device
    /// (T6 parent); its shared views are ordinary `LeasedRegion`s.
    MmioRegion,
    /// A task's authority to own the **live FPU / SIMD register state** on a
    /// `LogicalCpu` — the `CpuShare` sibling for the vector
    /// register file: the register file is a per-CPU resource, so ownership
    /// of it is CPU-rooted, single-holder, affine, and revocable like a share.
    Fpu,
    /// A task's authority over a device's established interrupt route (#86) —
    /// the `MmioRegion` sibling for interrupts. Device-rooted (tracked):
    /// establishes + owns the device's MSI(-X) route; a task parks on it.
    IrqRoute,
    /// The **X-mint** authority (res/IMAGE.md §6, the VMEX analogue): the
    /// seeded, zero-backing capability whose holder may bless a lease
    /// executable (`LEASE_EXEC`).  Endowed to the image service and nothing
    /// else; EXECUTE is mintable only through it, never ambient.
    ExecMint,
    /// A founder's handle over a **founded task** (res/IMAGE.md §9f): minted
    /// by the spawn session's terminal `Found` into the founder's result slot,
    /// carrying `ACCESS` (wait/observe) + `MODIFY` (explicit START) + `DERIVE`
    /// (attenuate a separately revocable `ACCESS|MODIFY|REVOKE` client handle)
    /// + `REVOKE` (kill — revoking the handle reaps the task).  Rooted under
    /// the founded task's own space root, so the task's death reaps the handle
    /// (T6) while the handle's revoke drives the kill/reap path.  Appended last
    /// (append-only wire).
    Task,
    /// The **exec-authority** singleton (res/IMAGE.md §9f): the seeded,
    /// zero-backing capability whose holder may run the terminal `Found`
    /// verb (founding a dormant task which a task handle may then START).
    /// Endowed to the exec service and nothing else; requiring it at `Found`
    /// makes founding a *held* authority rather than ambient over any `VirtMemory`
    /// holder.  Appended last (append-only wire; kernel
    /// `schema::ResourceKind` twin).
    ExecAuthority,
    /// The **display framebuffer aperture** (the display handoff): a
    /// boot-discovered video-memory window (Limine/UEFI framebuffer or a
    /// GPU aperture later) with its mode geometry — the display service's
    /// endowment, claimed once from the catalog.  Appended last
    /// (append-only wire).
    Display,
    /// A **non-PCI, firmware-described device** (#417): fixed I/O or one
    /// firmware-owned MMIO window, plus an optional interrupt line. Derived
    /// from [`PlatformBus`](Self::PlatformBus); I/O-port, MMIO-region, and
    /// line-route children may derive from it. Appended last
    /// (append-only wire).
    PlatformDevice,
    /// A driver's authority over exactly the I/O ports its
    /// [`PlatformDevice`](Self::PlatformDevice) declares (#416) — the
    /// `IrqRoute` shape for port space.  Device-rooted, claim-once;
    /// holding it means the kernel admits this task's `IN`/`OUT` to
    /// those ports through the per-CPU I/O permission bitmap.
    IoPortGrant,
    /// One **PCIe root complex / host bridge** (KRAPH `$RootComplex$`, an
    /// `R₀` kind) — the per-instance authority a
    /// [`PciDevice`](Self::PciDevice) capability derives from **by
    /// root-local index**, and the endowment of a userland device broker
    /// row (`res/DEVICE_SERVICE.md` §10).
    ///
    /// One capability per firmware-described root: it carries the root's
    /// segment, admitted bus range, NUMA anchor, and config-space backend,
    /// so *which* driver owns *which* function stops being a kernel
    /// decision and cross-root authority is refused by construction.
    /// Enumeration stays the catalog's job; holding this grants no
    /// register, DMA or interrupt authority of its own — only the right to
    /// hand one of *this root's* functions to somebody.
    ///
    /// Wire history: discriminant 31 was `DeviceStore`, the flat
    /// machine-wide inventory singleton this kind replaced (#572); the
    /// value is frozen, the semantics narrowed to one root.  (Kernel
    /// `schema::ResourceKind` twin.)
    RootComplex,
    /// Read-only machine telemetry authority. A holder may sample bounded
    /// scheduler, physical-memory, and kernel-heap counters, but cannot
    /// mutate any of those resources. Appended last (wire ABI).
    SystemObserve,
    /// Inert, one-shot proof that one executable endpoint authorized one
    /// exact invocation supply. It exposes no endpoint or memory authority.
    InvocationWitness,
    /// A task's **SPMD batch-reservation claim** on a CPU (G6c —
    /// `res/SPMD.md` §7): the `CpuShare`/`Fpu` sibling marking a core
    /// spoken for by a gang's all-or-nothing class reservation.
    /// CPU-rooted, tracked, kernel-minted at founding — never a session
    /// derive.  Additive on the frozen vocabulary (#81).
    CpuReserve,
    /// The **platform-device inventory** — the primordial singleton a
    /// [`PlatformDevice`](Self::PlatformDevice) capability derives from
    /// **by index**: the firmware-described (non-PCI) sibling authority of
    /// [`RootComplex`](Self::RootComplex), split out of the old flat
    /// `DeviceStore` index (#572).  Zero physical backing
    /// (`Primordial ⟹ ¬Teardown` trivially); platform devices sit on no
    /// PCIe root, so their delegation root is this machine-wide singleton.
    /// Appended last (append-only wire — behind `CpuReserve`, which
    /// landed on main first; kernel `schema::ResourceKind` twin).
    PlatformBus,
    /// A memory-fabric domain registration (res/MEMORY_FABRIC.md §3, #697) —
    /// the affine authority over one registered allocation domain.  Append-only.
    DomainRegistration,
    /// A memory-fabric route lease (res/MEMORY_FABRIC.md §4, #697) — the affine
    /// operation-specific edge over a domain subrange.  Append-only.
    MemoryFabricRoute,
    /// The primordial memory-fabric registrar (res/MEMORY_FABRIC.md §3, #697) —
    /// the singleton host-DRAM registrations derive from.  Append-only.
    FabricRegistrar,
    /// The primordial machine-virtualization facility.  A holder may mint
    /// coherent guest stage-2 spaces in the hardware format selected at
    /// bring-up.  Appended after main's fabric kinds (wire ABI — positional).
    Hypervisor,
    /// One coherent guest-physical address space (CPU stage-2 + DMA
    /// projection), derived from [`Hypervisor`](Self::Hypervisor).  Appended
    /// last (wire ABI).
    GuestSpace,
    /// One lease-backed window installed in a [`GuestSpace`](Self::GuestSpace).
    /// The lease remains the physical owner; this capability is the typed
    /// lifecycle name for the guest-visible projection. Appended last.
    GuestRegion,
    /// A driver's live **device-stage-1 MMU facet**. Derived from exactly one
    /// `PciDevice` after the driver has programmed its on-device translation
    /// regime (Xe PPGTT/GGTT, iVPU SMMU, or an equivalent contained stage-1).
    /// The facet is claim-once per device epoch; revoking it retires every
    /// peer window it sponsored before the epoch may be reclaimed.
    /// Append-only wire.
    DeviceMmu,
    /// Boot-discovered memory controller; currently kernel-owned.
    MemoryController,
}

/// The stamped capability record as it lives in the shared cap arena — the
/// type-erased wire form of the kernel's typed `Capability`.
///
/// `#[repr(C)]`: kernel and userland link this same definition, so the
/// layout is identical on both sides of the boundary and the kernel reads
/// it zero-copy. The `mac` seals the canonical message (every field above
/// it, serialized in declaration order); **only the kernel holds the epoch
/// key** to compute or check it, so userland can hold these bytes but never
/// fabricate authority. (The canonical-message serializer lands with the
/// `schema` migration, when the exact byte order is pinned against the
/// kernel's stamp input.)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CapRecord {
    /// This capability's identity.
    pub cid: Cid,
    /// Parent identity, or [`Cid::NULL`] for a primordial root.
    pub parent: Cid,
    /// **D1 — the holder**: the address-space root permitted to *use* this
    /// cap. A record presented by any other holder fails the kernel's bind
    /// gate, so a leaked cap is inert. Re-stamped on `rendezvous` transfer.
    pub holder: Cid,
    /// The resource rid this cap authorizes (kind interleaved with id).
    pub scope_rid: u64,
    /// Granted authority bits.
    pub rights: Rights,
    /// Resource-kind discriminant (consistent with `scope_rid`'s kind bits;
    /// carried explicitly for a stable, self-describing wire record).
    pub kind: u16,
    /// Epoch the stamp was sealed under.
    pub epoch: u32,
    /// Ancestor-chain length (<= [`MAX_DEPTH`]).
    pub chain_len: u8,
    /// Ancestor tags, oldest-first — the MAC-bound lineage.
    pub chain: [u32; MAX_DEPTH],
    /// The kernel's 128-bit AES-CMAC stamp over the canonical message
    /// (every field above, serialized in declaration order — the byte
    /// layout `schema::wire` pins).  Only the kernel holds the epoch
    /// key, so userland can carry these bytes but never mint them.
    pub mac: [u8; 16],
}

impl CapRecord {
    /// The all-zero **vacant** record: `cid == `[`Cid::NULL`] is unmintable
    /// (tag 0 is reserved), so a vacant slot can always be told apart from
    /// any sealed record — and any record built over this base without a
    /// real cid stays **bind-inert** (a session naming it reads `Vacant`,
    /// never an authority).  The base the pager's fault notice
    /// ([`Notice`](crate::session::Notice)) encodes over.
    pub const fn vacant() -> Self {
        Self {
            cid: Cid::NULL,
            parent: Cid::NULL,
            holder: Cid::NULL,
            scope_rid: 0,
            rights: Rights::empty(),
            kind: 0,
            epoch: 0,
            chain_len: 0,
            chain: [0; MAX_DEPTH],
            mac: [0; 16],
        }
    }
}

// Frozen wire pins (#81; the crate-header wire-freeze discipline) — the stamped arena-slot record
// the kernel reads zero-copy; a drift here is a security bug (the board's
// per-slot single-writer guard also assumes this 128-byte footprint).
const _: () = {
    use core::mem::size_of;
    assert!(size_of::<CapRecord>() == 128);
    assert!(size_of::<Cid>() == 8);
    assert!(size_of::<Tag>() == 4);
    assert!(size_of::<Rights>() == 1);
};
