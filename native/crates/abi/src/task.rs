//! Task ABI — the **bootstrap context** a task reads at startup, and the
//! **spawn parameters** a parent stages (the frozen #81 wire surface — see the crate header; the unified task
//! model).
//!
//! The kernel knows only *tasks*; **thread vs. process is a userland
//! ergonomic** — a "thread" spawns a task **sharing** the spawner's address
//! space, a "process" spawns one into a **fresh** address space with delegated
//! regions (page-delegation spawn, #31). Both stage a [`SpawnReq`].
//!
//! # Bootstrap: capabilities by role, through a **static**
//!
//! A spawned task (detached VA — a *process*) discovers its world from a
//! [`BootBlock`]: a fixed `#[repr(C)]` **static in its own image** (the
//! `.kore.boot` section), written by the spawner through its delegated
//! mapping of the child page *before* first dispatch, and read through the
//! runtime's safe accessor.  The layout **is** the ABI — there is no
//! parser (v1's magic + var-length records walked parent-controlled bytes
//! in every child), no discovery step (a static resolves at link time —
//! the child knows its own symbol; v1's block needed its own address
//! conveyed), and **no env strings** (a `KEY=VALUE` is copyable — the
//! disclosure ≠ transfer property dies; genuine config is *data* over the
//! parent channel after first rendezvous, never bootstrap).
//!
//! What the block carries is exactly the role-tagged handle table
//! ([`HandleEntry`]) — the object-capability-OS model (Fuchsia
//! `processargs`, seL4 `bootinfo`): a task takes a cap **by role**, not by
//! string, because a capability is a stamped arena slot the kernel seeded,
//! and the entry merely *names* it.  Reading is through [`Bootstrap`] — a
//! **validated view** ([`Bootstrap::of`] checks the ABI stamp + count
//! bound), so an unchecked block cannot reach a resolver.

use bitflags::bitflags;
use zerocopy::byteorder::little_endian::{U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use crate::session::SlotRef;

/// Which core class a task would rather run on — a **preference** the
/// scheduler weighs, never a pin.
///
/// An I/O-bound service whose threads spend their time parked on a
/// completion (a filesystem worker, a block driver, a swap device) does
/// its useful work in a few microseconds between parks and gains nothing
/// from a performance core; leaving those cores to the tasks that saturate
/// them is the whole system's latency.  Declared once by the service
/// ([`SystemService::CORE_PREFERENCE`]), cited by the founding manifest,
/// and applied at every placement as an attenuation on the non-preferred
/// class — so a preferred core that is far busier still loses.
///
/// `Any` is the default and the only value a homogeneous machine ever
/// distinguishes; a preference for a class the machine lacks is inert.
///
/// [`SystemService::CORE_PREFERENCE`]: ../../kore_rt/service/trait.SystemService.html#associatedconstant.CORE_PREFERENCE
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CorePreference {
    /// No preference: every class scores on load and locality alone.
    #[default]
    Any = 0,
    /// Prefer efficiency cores (Intel E-core, ARM LITTLE).
    Efficient = 1,
    /// Prefer performance cores.
    Performance = 2,
}

impl CorePreference {
    /// Decode a stored byte; unknown values read as `Any`, never as a
    /// preference the caller did not declare.
    pub const fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Efficient,
            2 => Self::Performance,
            _ => Self::Any,
        }
    }
}

/// Well-known startup-handle roles — a task discovers its delegated caps by
/// role. Values at or above [`USER_BASE`](role::USER_BASE) are caller-assigned.
pub mod role {
    /// The task's own task capability (self-reference).
    pub const SELF: u16 = 0;
    /// The task's address-space root.
    pub const ADDRESS_SPACE: u16 = 1;
    /// Send endpoint to the parent / driver fabric (the request channel).
    pub const PARENT_SEND: u16 = 2;
    /// Recv endpoint from the parent / driver fabric.
    pub const PARENT_RECV: u16 = 3;
    /// A device capability (a driver's device).
    pub const DEVICE: u16 = 4;
    /// An interrupt route (MSI-X) grant for the device.
    pub const IRQ_ROUTE: u16 = 5;
    /// A DMA-capable memory region / pool.
    pub const DMA_POOL: u16 = 6;
    /// A log sink (console output).
    ///
    /// **Reserved; nothing implements it.** No founder delegates this role
    /// and no code resolves it — console output reaches the display service
    /// over a `ring::spsc<OutChunk>` ring, not through a delegated sink. The
    /// discriminant is frozen (a role number is never reused), so it stays,
    /// but its presence is not evidence of a logging path.
    pub const LOG_SINK: u16 = 7;
    /// A namespace / service-directory capability (peer discovery).
    ///
    /// **Reserved; nothing implements it.** Services are reached only
    /// through the role slots the founder stamps into a task's boot page;
    /// nothing resolves a name to a peer. The one string-keyed lookup in the
    /// system is the image service's archive
    /// ([`ImageName`](crate::image::ImageName)), which is not this.
    pub const NAMESPACE: u16 = 8;
    /// A [`Timer`](crate::cap::ResourceKind::Timer) capability — the authority a
    /// task needs to sleep, arm a timeout/timer, or discipline the wall clock
    /// ([`crate::clock`]). Reading time needs no cap, so a task that only reads
    /// the clock is not delegated one.
    pub const TIMER: u16 = 9;
    /// A memory-supply region — the extent-granting endowment a memory-manager
    /// service (the **lessor** — the kernel holds no swap table) leases task heaps from.
    pub const MEM_SUPPLY: u16 = 10;
    /// The task-founding authority — the **exec manager**'s endowment (the
    /// KRAPH BootWF role beside the memory manager): the task-manager
    /// service founds task roots under it and grants parents attenuated
    /// handles.  Founding-only — it endows nothing (bundles stay
    /// plan-published / parent-delegated), so no super-init arises.
    pub const EXEC_AUTHORITY: u16 = 11;
    /// The image-service channel endpoints — the loader's `Acquire` wire
    /// (`res/IMAGE.md` §4).  A dynamically-linked task's stub resolves
    /// DT_NEEDED libraries over this channel; delivered at spawn
    /// (DEVICE_SERVICE §6: no nameserver).
    pub const LOADER: u16 = 12;
    /// The executable's own segment lease caps (indexed — one entry per
    /// segment), from the parent's `Acquire` of the exe, re-delivered into
    /// the child (`res/IMAGE.md` §4 phase 1).
    pub const IMAGE: u16 = 13;
    /// The image service's byte-source endowment — the boot-module /
    /// embedded-archive supply it registers images from (`res/IMAGE.md`
    /// §3).  Service-side only.
    pub const IMAGE_STORE: u16 = 14;
    /// The **X-mint** — the sole authority that can bless a lease
    /// executable (`LEASE_EXEC`, `res/IMAGE.md` §6; the Fuchsia-VMEX
    /// shape).  Endowed to the image service and nothing else.
    pub const X_MINT: u16 = 15;
    /// The **address-space allocator** (`VirtMemory`) capability — the authority
    /// to mint fresh address spaces (`res/IMAGE.md` §9e, the spawn
    /// session's first leg).  Endowed to the exec manager (the founder);
    /// deriving from it *is* "allocate a new address space".
    pub const VIRT_MEMORY: u16 = 16;
    /// The task's **executor doorbell** (`res/USERLAND.md` §3.1/§5.4):
    /// index 0 = the held `ChanRecv` end's arena slot; index 1 = the
    /// designated **landing slot** (a vacant arena slot every standing
    /// post of the doorbell donates — pinned at founding so the runtime
    /// need not allocate one).
    pub const DOORBELL: u16 = 17;
    /// A doorbell **send** end to a peer (indexed; `res/USERLAND.md`
    /// §5.3's channel pair): on a client, index `i` kicks the `i`-th
    /// consumed service's doorbell; on a service, index 0 kicks its
    /// founding client's.  Notices only — capability transfers ride
    /// [`SVC_GRANT`].
    pub const SVC_SEND: u16 = 18;
    /// A **grant-plane** channel end (indexed; `res/USERLAND.md` §5.5):
    /// one channel per expected grant-carrying reply transfer (the board
    /// law is one standing post per channel, so a bank of channels — not
    /// a bank of posts — carries a multi-grant reply).  Service side
    /// holds the send ends, client side the recv ends.
    pub const SVC_GRANT: u16 = 19;
    /// The client↔service pair's shared **request/reply ring window**
    /// lease (indexed per pair; `res/USERLAND.md` §5.3): the mm lease
    /// whose mapped window holds the request and reply `ring::spsc`s.
    pub const SVC_RING: u16 = 20;
    /// The founder-reserved **growable heap window** lease
    /// (`res/USERLAND.md` §3.3; the sentinel-elimination heap rung): the
    /// contiguous backing a founded service grows its `Heap` into, delivered
    /// as a real cap instead of a fixed `HEAP_VA` convention.
    pub const HEAP: u16 = 21;
    /// A **return-plane** channel end (`res/USERLAND.md` §5.5's reverse
    /// leg): capability returns ride here — a client sends a granted cap
    /// back over it, the service's standing post lands it.  On a service,
    /// index 0 = the held `ChanRecv` end's arena slot, index 1 = the
    /// designated **landing slot** (the [`DOORBELL`] pair's discipline);
    /// the same landing-slot *number* is what the client's data-plane
    /// return request (e.g. mm's `Release { slot }`) names.  On a client,
    /// index 0 = the `ChanSend` end.  Separate from [`SVC_GRANT`] because
    /// the board law is one standing post per channel — a grant post and
    /// a return post can never share a channel.
    pub const SVC_RETURN: u16 = 22;
    /// The **display framebuffer aperture** endowment (the display
    /// handoff): index 0 = the catalog `Display` resource (mode geometry
    /// + write-combining window) the display service claims to own the
    /// present path.  At most one — the machine's boot framebuffer.
    pub const DISPLAY: u16 = 23;
    /// The **system power** endowment (#392): index 0 = the primordial
    /// `SystemPower` facility the power service holds `MODIFY` on, which
    /// is what lets it request a machine transition through the
    /// [`PowerReq`](crate::session::PowerReq) derive flavor.
    ///
    /// At most one, and endowed to exactly one service — power-off is
    /// machine-wide authority, so a second holder would be a second way to
    /// take the machine down.  A task entitled only to *read* the backend
    /// report holds an `ACCESS`-only witness, which the same role carries
    /// at attenuated rights rather than a second role.
    ///
    /// **Rights the endowment must carry: `ACCESS + MODIFY + DERIVE`.**
    /// `MODIFY` is the authority gate the transition is checked against;
    /// `DERIVE` is required because the verb rides the `Derive` op-code,
    /// whose prepare-side `enabled` predicate demands it. That the op mints
    /// nothing does not exempt it — the pager-registration verb
    /// (`Derive × AddrSpace`, `kind = ChanRecv`) is the same mints-nothing
    /// shape and pays the same toll.
    pub const POWER: u16 = 24;
    /// Private backing consumed by the startup-only linker while it builds
    /// this task's writable image, TLS block, and passive module page.
    ///
    /// Distinct from [`MEM_SUPPLY`]: that role is a service's continuing
    /// lessor authority (mm/swap) and remains live after startup. Sharing one
    /// supply between the stub's bump cursor and a resident lessor's own
    /// allocator would admit overlapping sub-leases as soon as both started at
    /// offset zero. `LINK_SUPPLY` is startup-private and is never granted
    /// onward.
    pub const LINK_SUPPLY: u16 = 25;
    /// One **PCIe root complex** — a device-broker row's primary endowment
    /// (`res/DEVICE_SERVICE.md` §10): a revocable self-kind grant of one
    /// firmware-described root, off which a per-device [`DEVICE`] grant is
    /// derived by root-local index.
    ///
    /// Row-relative: every broker row endows `ROOT_COMPLEX@0` = "this
    /// row's root", and the *founder* maps broker ordinal k → the k-th
    /// discovered root — so one endowment type serves every instance.
    /// A row holding this decides *which driver owns which of this root's
    /// functions*; it is deliberately not composable with [`DEVICE`] on
    /// the same row, because a task that both arbitrates devices and
    /// drives one is the god-object `res/DEVICE_SERVICE.md` §1 rules out.
    ///
    /// (Value 26 was `DEVICE_STORE`, the flat machine-wide inventory this
    /// role replaced in #572.)
    pub const ROOT_COMPLEX: u16 = 26;
    /// A bounded, revocable grant of the machine's rendezvous allocator.
    ///
    /// A launcher holding this role may derive fresh `Channel` roots and their
    /// `ChanSend` / `ChanRecv` endpoints.  It receives a self-kind child of the
    /// primordial `ChannelSpace`, never the machine root, so revoking this one
    /// grant cuts every channel the launcher minted by T6.
    ///
    /// **Rights the endowment must carry: `ACCESS + DERIVE + REVOKE`.**
    /// `DERIVE` admits channel minting; `REVOKE` makes the grant and its whole
    /// subtree explicitly reclaimable.
    pub const CHANNEL_SPACE: u16 = 27;
    /// A **memory-region grant** — the lessor-of-last-resort's minting
    /// authority (slice D): a revocable self-kind child of a boot DRAM
    /// region, off which the holder stages `LeaseReq` derives
    /// (`Derive × PhysRegion → MemLease`) to mint fresh **root supply
    /// leases** — the same op the kernel founder allocates task memory
    /// with, reached through an ordinary session.
    ///
    /// Distinct from [`MEM_SUPPLY`]: that role is one *already-minted*
    /// lease mm carves heap-extent views from; this one is the authority
    /// to mint new sibling leases (child supplies), each returning its
    /// frames to the **region bitmap** — never to mm's books — when its
    /// lineage dies (T6).  Endowed to exactly one row (mm); the grant is
    /// a child of the region record, so revoking it cuts every supply mm
    /// ever minted at once.
    ///
    /// **Rights the endowment must carry: `ACCESS + MODIFY + DERIVE +
    /// REVOKE`** — the lease derive's gate demands `DERIVE`, and Σ5
    /// clamps every minted lease (and every view under it) to this set,
    /// so a writable child supply needs `MODIFY` here.
    pub const MEM_REGION: u16 = 28;
    /// A face-minting service's **face supply** (slice C): the backing
    /// whose window strides become that service's runtime-minted serving
    /// pairs.  Two rows carry it today, each in its own row namespace:
    /// the image service's per-child loader-face rings
    /// (`ImageReqWire::OP_FACE_MINT`) and the device broker's claimant
    /// faces (`DeviceReqWire::OP_MINT_FACE` — one pair per driver row,
    /// the founding parent's half of the handshake).  The serving view
    /// stays with the minting service on purpose: pair windows a client
    /// task's reap can never unmap out from under the lessor's own
    /// serving view (the memory-safety half of the invariant, not just
    /// doctrine).
    ///
    /// Distinct from [`MEM_SUPPLY`]/[`LINK_SUPPLY`] for the same
    /// reason those two are distinct: one backing, one carve cursor —
    /// the face table's slot arithmetic must never interleave with a
    /// heap tier's.
    ///
    /// Geometry contract: [`FACE_SUPPLY_BYTES`](super::FACE_SUPPLY_BYTES)
    /// = [`FACE_RING_STRIDE`](super::FACE_RING_STRIDE) rings — the
    /// founder mints exactly that, the face table carves exactly that.
    pub const FACE_SUPPLY: u16 = 29;
    /// Read-only machine telemetry authority, endowed only to the monitor
    /// service. The capability carries `ACCESS | DERIVE`; observation is a
    /// bind-inert report, never ambient access to scheduler internals.
    pub const SYSTEM_OBSERVE: u16 = 30;
    /// A generation-bound ordinary executable send endpoint. The role index
    /// is task-local namespace metadata only; possession of the held
    /// `ChanSend` record is the launch authority.
    pub const EXECUTABLE: u16 = 31;
    /// Generic atomic-invocation context channels: index 0 carries the
    /// executable-possession witness, index 1 carries `ProcessIo`, and index
    /// 2 carries a caller-derived structural filesystem/cwd view.
    pub const EXEC_CONTEXT: u16 = 32;
    /// A caller-minted endpoint whose immutable resource identity FS binds to
    /// a newly resolved cwd.
    pub const FS_VIEW_BIND: u16 = 33;
    /// A parent-owned post-founding configuration channel. The child receives
    /// only this receive endpoint in the atomic `Found` transaction; the
    /// bounded capability package that follows is ordinary rendezvous, not a
    /// second invocation assembler or ambient bootstrap namespace.
    pub const SESSION_CONFIG: u16 = 34;
    /// A non-invocable selector for one stable recipient filesystem view.
    /// It carries REVOKE only: enough to authenticate its channel lineage,
    /// never enough to substitute it as the recipient's cwd authority.
    pub const FS_GRANT_TARGET: u16 = 35;
    /// The compositor's **surface-backing supply** (#578 slice 3): one
    /// founder-staged DRAM arm ([`SURFACE_POOL_BYTES`], 2 MiB-aligned so
    /// every whole-granule carve maps in huge leaves) the display row
    /// carves client surfaces from.  The compositor is the LESSOR —
    /// `CreateSurface` windows are detached views of this arm, and a
    /// client's death or destroy returns its carve whole
    /// (`REVOKE_KIND_RETURN`) without ever unmapping the compositor.
    /// Distinct from [`MEM_SUPPLY`] (mm's continuing lessor supply) and
    /// [`LINK_SUPPLY`] (startup linking): pixels are neither heap nor
    /// text, and sharing a cursor with either would interleave a surface
    /// carve with an allocator grow on one span.
    ///
    /// [`SURFACE_POOL_BYTES`]: crate::task::SURFACE_POOL_BYTES
    pub const SURFACE_POOL: u16 = 36;
    /// The **platform-device inventory** — the [`ROOT_COMPLEX`] sibling
    /// for firmware-described (non-PCI) devices: a revocable self-kind
    /// grant of the `PlatformBus` singleton, off which a per-device
    /// [`DEVICE`] grant is derived by index (#572 split the old flat
    /// `DEVICE_STORE` into the two authorities).
    ///
    /// Endowed to exactly one row — broker ordinal 0 — because platform
    /// devices sit on no PCIe root and need exactly one arbiter.
    pub const PLATFORM_BUS: u16 = 37;
    /// A driver's **firmware byte store** — the sibling of [`IMAGE_STORE`]
    /// for device firmware: the whole-lease boot-module supply a driver row
    /// maps R/O to feed its silicon's bring-up (the Intel NPU's
    /// `vpu_40xx_v1.bin`, `services/accel-ivpu`).  Like the image archive it
    /// is a catalog [`PhysRegion`](crate::cap) over a Limine module's exact
    /// span, leased WHOLE — the endowment's frames ARE the module's bytes,
    /// no copy — and claimed through the same authenticated held-lease
    /// exercise (the kernel object is the common `MemLease`; the role is
    /// what differentiates firmware from an image archive).
    ///
    /// **Optional by declaration** ([`RoleSlot::indexed_optional`]): a build
    /// without the fetched blob ships no module, so the founder stamps the
    /// coordinate vacant and the row founds degraded rather than failing the
    /// boot — device firmware, unlike an image archive, is legitimately absent
    /// when the driver's blob was never fetched for this machine.
    ///
    /// [`RoleSlot::indexed_optional`]: crate::task::RoleSlot::indexed_optional
    pub const FIRMWARE_STORE: u16 = 38;
    /// The measured image-archive era carried as immutable boot data to the
    /// filesystem service: indices 0..=7 are the eight little-endian `u32`
    /// words of the archive's 32-byte BLAKE3 digest.
    ///
    /// This role names **data, not authority**. It therefore appears only in
    /// [`BootBlock::data`](super::BootBlock::data), never in the capability
    /// profile or arena handle table. Keeping the role explicit makes a
    /// missing or partially stamped era fail closed at `FsEndowment::resolve`
    /// instead of silently becoming an all-zero policy input.
    pub const ARCHIVE_ERA: u16 = 39;
    /// Founder-formatted input → compositor report ring. Producer and
    /// consumer are separately endowed; ordinary applications receive neither.
    pub const POINTER_REPORTS: u16 = 40;
    /// Input router's send-only leaf onto the compositor's executor bell.
    pub const POINTER_KICK: u16 = 41;
    /// The machine-virtualization facility grant, endowed only to the
    /// hypervisor service.  Deriving from it creates coherent guest address
    /// spaces; it is unrelated to [`VIRT_MEMORY`], which creates ordinary CPU
    /// process spaces. This appends after the compositor's pointer transport
    /// roles, which now occupy 40–41.
    pub const HYPERVISOR: u16 = 42;
    /// MM's per-client outstanding-supply policy, in bytes: indices 0 and 1
    /// carry the low and high u32 words. Boot data only; grants no authority.
    pub const SUPPLY_QUOTA: u16 = 43;
    /// Read-only model-data lease, delivered by exec for PROFILE_MODEL_DATA.
    /// Index 0 is the lease; data indices 1 and 2 carry the exact byte length
    /// (low/high u32). The receiver maps it; page padding is not model data.
    /// Optional data indices 3..=7 report disk-load phase durations in ms:
    /// setup, source validation, allocation/mapping, read/verify/copy, cleanup.
    /// u32::MAX means unavailable. These values confer no authority.
    pub const MODEL_DATA: u16 = 44;
    /// A filesystem launch's storage-ring window (`res/STORAGE_TRANSPORT.md`
    /// §4): a `MemLease` of exactly `kore_ring::stor::WINDOW_BYTES`, funded by
    /// exec from its own mm face and moved in whole at founding, so the child
    /// attaches a ring without an mm face. Exec-synthesized: a caller row
    /// naming it is refused.
    pub const STOR_RING: u16 = 45;
    /// A filesystem launch's ring read target: a fresh `MemLease` exec funds
    /// beside [`STOR_RING`] and moves in whole. Nothing maps it before the
    /// child registers it on the fabric, which a device route requires — a
    /// range of the launcher's own arena could not be one, since the
    /// launcher keeps that arena mapped. Exec-synthesized, like
    /// [`STOR_RING`].
    pub const STOR_TARGET: u16 = 46;
    /// First caller-assigned role — roles `>=` this are convention-defined.
    pub const USER_BASE: u16 = 0x8000;
}

/// The [`role::SURFACE_POOL`] arm's fixed span — the cross-boundary
/// geometry contract (the `FACE_SUPPLY_BYTES` precedent): the founder
/// stages exactly this many bytes at 2 MiB alignment, and the display row
/// refuses bind on any other span rather than guessing a pool geometry.
/// Initial compositor heap supply. Both screens share it and further extents
/// are supplied through the compositor's declared mm client on demand.
pub const SURFACE_POOL_BYTES: u64 = 128 * 1024 * 1024;

/// Bounded physical-memory budget endowed to the hypervisor service.
/// Eight MiB fits the #485 two-guest witness (two 2-MiB MicroPython RAM
/// extents), both one-page exit/control windows, and small assembly slack.
/// Further guests fail allocation cleanly instead of drawing from ambient
/// kernel memory.
pub const HYPERVISOR_SUPPLY_BYTES: u64 = 8 * 1024 * 1024;

/// Byte stride of one minted face-ring window inside a
/// [`role::FACE_SUPPLY`] backing (slice C) — face slot `k`'s pair rings
/// live at `[k · STRIDE, (k+1) · STRIDE)`.  A page-multiple (the client
/// sub-lease carve demands page alignment) with headroom over each
/// carrying service's actual pair bytes; every face-minting service
/// (image, the device broker) asserts the fit against its own wire
/// forms, where both numbers are visible.
/// One independently recyclable service-face ring window. The common boot
/// layout and every client mapping must use this same full extent. The
/// generated founder plan asserts equality, so growing its wire records
/// cannot leave clients using a stale stride.
pub const FACE_RING_STRIDE: u64 = 0x4000;

/// Role-index namespace for a service's declared provider clients. Provider
/// serving pairs retain low indices; each client uses this base plus its
/// manifest ordinal for SVC_RING, SVC_SEND, and its first SVC_GRANT.
pub const SERVICE_CLIENT_INDEX_BASE: u16 = 0x8000;
/// An inert SVC_RING datum giving the founder's complete serving-pair count.
/// It cannot collide with a capability handle: data and handles are disjoint.
pub const SERVICE_PAIR_COUNT_DATUM: u16 = u16::MAX;

/// Initial inline desktop peer storage, not an admission ceiling. Dynamic
/// owners grow from a named arena; controllers remain boot-profile authority.
pub const DESKTOP_BOOTSTRAP_WINDOW_SLOTS: usize = 6;
/// Independently authenticated desktop controllers in the current boot profile.
pub const DESKTOP_CONTROLLER_CAPACITY: usize = 2;
/// Child loader faces reserved for all windows, their shell controllers, and
/// one foreground command per controller. A full desktop must still be able to
/// run a management command. This transport reservation does not bound exec's
/// allocator-backed cleanup ledger.
pub const INTERACTIVE_CHILD_LOADER_FACES: usize =
    DESKTOP_BOOTSTRAP_WINDOW_SLOTS + 2 * DESKTOP_CONTROLLER_CAPACITY;

/// Live immutable image-generation peers the boot catalogue can hold at once:
/// the 40-row bootstrap catalogue — the desktop's `editor` and `power-ui`
/// beside the native filesystem verbs `append`, `sync` and `pack`, the two
/// WebAssembly rows, `wasm-witness` and `wasm`, `inference` /
/// `model-data-witness`, and `shell`, the guest shell `term` opens — plus the
/// founded shell, the graphical terminal, and three extraction parsers.
/// Exec asserts this covers its catalogue (`kore-svc-exec`), the one place
/// that sees both.  Named
/// separately from the interactive reservation below because the two grow for
/// unrelated reasons: this one when a row joins the catalogue, that one when
/// the desktop admits another window or controller.
pub const BOOTSTRAP_IMAGE_PEERS: usize = 45;

/// Bootstrap face-supply sizing. Image and runtime grow their face tables and
/// rings on demand; this reserves enough initial memory for the boot catalogue
/// and for a full desktop that can still run one command per controller.
pub const MAX_MINTED_FACE_PAIRS: usize = BOOTSTRAP_IMAGE_PEERS + INTERACTIVE_CHILD_LOADER_FACES;

/// Total byte length of a [`role::FACE_SUPPLY`] endowment — exactly
/// [`MAX_MINTED_FACE_PAIRS`] ring strides.
/// The **cross-boundary geometry contract**: the kernel founder mints a
/// lease (or, for the device broker, a full-span sub of the shelf-held
/// claimant band) of exactly this length; the carrying service's face
/// table carves its slots against the same number (its claim checks the
/// span).  One constant, cited by both sides, so a drift refuses at
/// bind instead of carving a phantom ring.
pub const FACE_SUPPLY_BYTES: u64 = MAX_MINTED_FACE_PAIRS as u64 * FACE_RING_STRIDE;

/// **Geometry roles** — the window kinds a task learns its own VAs for from the
/// [`GeomBlock`] (`res/USERLAND.md` §4; the fixed-VA elimination rung).  Unlike
/// [`role`] handle entries (which name an arena *slot*), a geometry entry names
/// a `(va, len)` the founder mapped — the addresses that were hand-copied fixed
/// constants (`M2B_SCRATCH_VA`, `SLOTS_VA`, …) before this block existed.
pub mod geom {
    /// The task's own **descriptor scratch** window — R/W, where the trap
    /// transport stages a session's finished bytes before `commit_current`
    /// (replacing `kore-rt`'s `M2B_SCRATCH_VA`).
    pub const SCRATCH: u16 = 0;
    /// The task's **cap-slot array alias** — R/O, where the possession verbs
    /// read their landed `CapRecord`s (replacing `SLOTS_VA`); `len` is the
    /// ABI 6: records followed by an initialized read-only execution-hint
    /// page; its span is [`crate::execution::alias_bytes`]. The page is not
    /// a cap record and must never contribute to the runtime slot count.
    pub const SLOTS: u16 = 1;
    /// The task's initially-vacant receiver-map allocation window.  Founding
    /// installs no leaves here; it seeds the task's monotone map cursor from
    /// this founder-published range.
    pub const MAP: u16 = 2;
    /// The task's **thread block** — the R/W per-thread runtime block the
    /// entry anchors at `%fs:0` (the unified-task-model thread seam, #407):
    /// its first word is the block's own VA (TLS variant-2 self-pointer),
    /// and the runtime's per-thread state (window geometry, report/heap
    /// slots) lives behind it.  For a spawned **thread** the parent
    /// allocates this window and stamps it here; a **process** entry
    /// anchors its image's own static block instead and carries no entry.
    pub const THREAD_BLOCK: u16 = 3;
    /// The task's **display-aperture** window — where a `Display`-rooted
    /// `MmioRegion` (the framebuffer, the display handoff's present path)
    /// is installed, and the only window kind that is not a view.
    ///
    /// Separate from [`MAP`] because the two are different shapes of
    /// address-space consumption: `MAP` is a cursor over many small
    /// delivered views, strided per thread so no owner can spend a
    /// sibling's stride; an aperture is one fixed mapping whose length the
    /// boot *mode* picks (`pitch × height` — 31.6 MiB at 3840×2160).
    /// Spending it out of the map stride capped every owner's views at
    /// whatever the panel left over, and past 1920×1200 left nothing: the
    /// refusal was typed and clean, and the display service still never
    /// beaconed.
    ///
    /// **Task-level, not per-thread**: unlike the other four this window
    /// has no band — a thread is stamped with none, so its aperture is
    /// fail-closed rather than an alias of its parent's.  Absent from a
    /// geometry table simply means "this task maps no framebuffer", which
    /// is every task but one.
    pub const APERTURE: u16 = 4;
    /// Startup link placement: dynamic modules and linker-owned passive or
    /// transient views. Distinct from [`MAP`], whose kernel-owned monotone
    /// cursor must remain entirely vacant until the linked program starts.
    pub const LINK: u16 = 5;
    /// The task's single reusable transient exact-mapping window. Unlike
    /// [`MAP`], this is not a monotone cursor: a caller installs one whole
    /// lease at this exact VA, uses it synchronously, then exact-unshares the
    /// view before reusing the address. Image registration validation and the
    /// shell's pre-convey loader stamp share this lifecycle in separate tasks.
    pub const TRANSIENT: u16 = 6;
    /// Compatibility spelling for the first consumer of [`TRANSIENT`].
    pub const IMAGE_VALIDATE: u16 = TRANSIENT;
    /// The task's **extension-slot alias** window (slice E): the VA
    /// reservation where `DERIVE_KIND_ARENA_EXTEND` alias-maps each
    /// grown arena chunk's records R/O — chunk `k` at
    /// `base + k · 32 KiB`.  Stamped reservation-only at founding (no
    /// pages mapped until the first extend); its `len` is the growth
    /// ceiling's userland witness.  A second window (rather than a wider
    /// [`SLOTS`](Self::SLOTS)) because the per-thread slots bands sit at
    /// stride immediately above the founding alias.
    pub const SLOTS_EXT: u16 = 7;
    /// The task's **surface** window (#578 slice 3) — where the compositor's
    /// [`role::SURFACE_POOL`] arm is installed whole, or an application's
    /// detached compositor-owned surfaces are receiver-mapped. The
    /// [`APERTURE`](Self::APERTURE) lesson applied verbatim: task-level
    /// (no per-thread band — a thread is stamped with none, fail-closed),
    /// the pool owner's granule arithmetic owns its interior, while a
    /// receiver selects one of two fixed task-level halves. The application
    /// runtime reuses a half only after exact view retirement proves its old
    /// PTE absent; the kernel still refuses an occupied overlap. Address space
    /// is not authority: mapping requires the held pool or detached view, and
    /// neither mapping grants the other capability.
    pub const SURFACES: u16 = 8;
    /// The **device-BAR window** — where `MmioRegion`-backed views from a
    /// claimed PCI function install ([`APERTURE`]'s sibling: the display
    /// BAR got its own window first; the NPU's 16 MiB BAR0 proved every
    /// other big BAR needs one too, because a large BAR on the receiver-map
    /// cursor starves the stride every thread's views share).  Address
    /// space, not authority — stamped for founded tasks; absent for
    /// spawned threads, where a BAR map then refuses fail-closed.  Fills the
    /// entry the v4 [`SURFACES`](Self::SURFACES) widening left spare (GEOM_ABI v5).
    pub const DEVICE: u16 = 9;
    /// The task's **heap window** (GEOM_ABI v7) — where the process heap maps
    /// the granules it carves off its own `role::HEAP` supply, at addresses it
    /// places itself. The [`DEVICE`](Self::DEVICE) lesson for the heap: growth
    /// on the receiver-map cursor spent the one 16 MiB stride every view on
    /// that thread shares, so no heap could map more than 16 MiB however large
    /// its supply (window launches refused larger heaps outright). Address
    /// space, not authority — a reservation with no backing, mapped only by
    /// leases the task already holds; task-level and stamped for founded
    /// tasks, absent for spawned threads, whose growth stays on their stride
    /// because their holdings die with them.
    pub const HEAP: u16 = 10;
}

/// One startup handle: a delegated capability tagged by [`role`] (+ an instance
/// `index`, so a task can hold several of one role — e.g. two devices). 8 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct HandleEntry {
    /// Role discriminant ([`role`]).
    pub role: U16,
    /// Instance index within the role.
    pub index: U16,
    /// Arena slot holding the delegated capability.
    pub slot: U32,
}

/// One immutable startup datum, addressed by the same `(role, index)`
/// vocabulary as a handle but stored in a **separate table**.
///
/// A datum is not an arena coordinate and must never participate in runtime
/// slot layout.  The distinction matters for values such as a retained-return
/// correlation in another task's arena: treating that opaque number as a
/// local capability slot once made crt0 derive its SET_TP result beyond the
/// arena and spin before the executable's first instruction.
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct DatumEntry {
    /// Role discriminant ([`role`]).
    pub role: U16,
    /// Instance index within the role.
    pub index: U16,
    /// Opaque parent-authored value.
    pub value: U32,
}

/// Boot-block ABI version.  v4 separates immutable startup data from the
/// capability handle table.  The symbol is the identity — a zeroed, unseeded
/// block reads `abi == 0` and refuses — and there is still no env.
pub const BOOTSTRAP_ABI: u16 = 4;

/// Handle capacity of a boot block — a frozen ABI constant, bounded like a
/// session (`res/SYSCALL.md` D4: finite by construction, never a stream).
///
/// **24, raised from 16 for the image endowment.** A row's own executable
/// arrives as indexed [`role::IMAGE`] leases — one per descriptor segment,
/// bounded by [`MAX_SEGMENTS`](crate::image::MAX_SEGMENTS) — so the child
/// can link its own image instead of receiving it pre-mapped
/// (`res/IMAGE.md` §4, the degenerate static case). IMAGE_ABI v3 preserves
/// each `PT_LOAD`, so `kore-svc-image` now needs 25 published names: its
/// profile and serving wiring, four image leases, `LINK_SUPPLY`, `HEAP`,
/// `PARENT_SEND`, and the address-space self-view. The former ceiling of 24
/// correctly refused the last name.
///
/// 32 is the next bounded power-of-two envelope and leaves seven entries of
/// real headroom.  Boot publication and founding delivery are independently
/// bounded: boot blocks name vacant landing slots (`DOORBELL@1`,
/// `SVC_RETURN@1`) which are not delivered capabilities, while a contiguous
/// grant bank may be published by one base handle but requires one delivery
/// per channel end.  Confusing these two counts is what made a lawful
/// 23-delivery founding fail while stamping its 25th published name.
///
/// The block leads the boot page's frozen entry blocks. The page is 4096
/// and holds every one of them in well under half of it
/// (`kore_rt::entry`'s layout pin).
/// 48 (from 32): the program-loading ring — the first shell's block
/// gained an image client face (kick + nine grant-channel recv ends +
/// pair ring) and the founder-prefunded spawn supply, and 32 held
/// exactly one spare entry.  The page-budget pin below keeps the grown
/// block honest.
/// The bounded named-capability vocabulary carried in one task's boot page.
///
/// Sixty-four accommodated the production shell's service faces plus its
/// attenuated child-principal sources.  Eighty (from 64): exec's block
/// carries five entries per resident provider face (request kick, pair
/// ring, reply-grant receiver, retained-return sender and landing), and two
/// growths met there at once — the seventh provider, the Xe row (the GPU
/// face a `PROFILE_GPU` launch is admitted on), and the memory manager's
/// return leg, the pair on which exec administers per-launch mm faces (one
/// per engine launch). Each alone took the block past sixty-four at its
/// loader grant bank. The page-budget pins below and in `kore_rt::entry`
/// keep the grown block honest: every later block's offset is derived from
/// this size, and the whole sequence must still fit one 4 KiB boot page.
pub const MAX_BOOT_HANDLES: usize = 80;

/// Literal startup values carried beside, but never confused with, arena
/// handles.  Spawn records currently carry at most four; the eight-entry
/// envelope leaves bounded headroom without widening the hot handle scan.
pub const MAX_BOOT_DATA: usize = 8;

/// The **boot block** — the fixed-layout static a spawned task reads its
/// capability endowment from (`.kore.boot` in the child image; the module
/// docs tell the whole story).  The layout *is* the frozen ABI: 8-byte
/// header + [`MAX_BOOT_HANDLES`] handle entries + [`MAX_BOOT_DATA`] datum
/// entries = 712 bytes, every field
/// little-endian and unaligned-safe, so the spawner writes it through any
/// delegated mapping with plain `IntoBytes` and no serializer exists on
/// either side.
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct BootBlock {
    /// [`BOOTSTRAP_ABI`] — the stamp a validated view checks (stale-image
    /// detection; an unseeded block is all-zero and refuses here).
    pub abi: U16,
    /// Reserved; must be zero.  Frozen additive headroom.
    pub flags: U16,
    /// Live entries in `handles` (`<= MAX_BOOT_HANDLES`).
    pub count: U16,
    /// Live entries in `data` (`<= MAX_BOOT_DATA`).
    pub data_count: U16,
    /// The role-tagged handle table; entries past `count` are zero.
    pub handles: [HandleEntry; MAX_BOOT_HANDLES],
    /// The role-tagged literal table; entries past `data_count` are zero.
    pub data: [DatumEntry; MAX_BOOT_DATA],
}

impl BootBlock {
    /// An empty, correctly-stamped block — the value a parent starts from
    /// (and the initializer the child image's static carries pre-write:
    /// all-zero, hence *invalid* until the spawner stamps it).
    pub const fn new() -> Self {
        Self {
            abi: U16::new(BOOTSTRAP_ABI),
            flags: U16::new(0),
            count: U16::new(0),
            data_count: U16::new(0),
            handles: [HandleEntry {
                role: U16::new(0),
                index: U16::new(0),
                slot: U32::new(0),
            }; MAX_BOOT_HANDLES],
            data: [DatumEntry {
                role: U16::new(0),
                index: U16::new(0),
                value: U32::new(0),
            }; MAX_BOOT_DATA],
        }
    }

    /// Whether `(role, index)` is already stamped.  A refused [`push`](Self::push)
    /// has exactly two causes — a full table or this — and a spawner that
    /// reports the refusal must be able to say which: a founder once spent a
    /// metal boot on "boot block full" that was a duplicate key (#936).
    pub fn contains(&self, role: u16, index: u16) -> bool {
        let n = (self.count.get() as usize).min(MAX_BOOT_HANDLES);
        self.handles[..n]
            .iter()
            .any(|entry| entry.role.get() == role && entry.index.get() == index)
    }

    /// Record a delegated capability at `(role, index)` naming arena
    /// `slot`.  Refuses (`false`) when the table is full or that role key is
    /// already present — bounded and unambiguous like every other emission,
    /// never silently dropped.  A duplicate key would otherwise make the
    /// result depend on table order because [`Bootstrap::handle`] returns the
    /// first match.
    pub fn push(&mut self, role: u16, index: u16, slot: SlotRef) -> bool {
        let n = self.count.get() as usize;
        if n >= MAX_BOOT_HANDLES || self.contains(role, index) {
            return false;
        }
        self.handles[n] = HandleEntry {
            role: U16::new(role),
            index: U16::new(index),
            slot: U32::new(slot.index()),
        };
        self.count.set(n as u16 + 1);
        true
    }

    /// Record one literal startup value at `(role, index)`.  Data has its own
    /// bounded table so no opaque value can raise the runtime's local arena
    /// high-water mark or be mistaken for a capability by [`Bootstrap::handle`].
    pub fn push_data(&mut self, role: u16, index: u16, value: u32) -> bool {
        let n = self.data_count.get() as usize;
        if n >= MAX_BOOT_DATA
            || self.data[..n]
                .iter()
                .any(|entry| entry.role.get() == role && entry.index.get() == index)
        {
            return false;
        }
        self.data[n] = DatumEntry {
            role: U16::new(role),
            index: U16::new(index),
            value: U32::new(value),
        };
        self.data_count.set(n as u16 + 1);
        true
    }

    /// Nothing recorded.
    pub fn is_empty(&self) -> bool {
        self.count.get() == 0 && self.data_count.get() == 0
    }
}

impl Default for BootBlock {
    fn default() -> Self {
        Self::new()
    }
}

/// Geometry-block ABI version.  v1 = the `(role, index, va, len)` window
/// table (`res/USERLAND.md` §4 — the fixed-VA elimination).  A zeroed,
/// unstamped block reads `abi == 0` and every geometry lookup refuses, exactly
/// like [`BootBlock`]/[`LdBlock`](crate::image::LdBlock).
/// v2 adds the startup-only [`geom::LINK`] window; v3 adds the reusable
/// [`geom::TRANSIENT`] exact-mapping window. Both retain the same bounded table
/// layout; [`geom::IMAGE_VALIDATE`] remains its compatibility spelling.
/// v4 widens the table (8 → 10) and adds the task-level [`geom::SURFACES`]
/// pool window (#578 slice 3) — stamped only for a row endowing
/// [`role::SURFACE_POOL`].  v5 names the tenth (previously spare) entry: the
/// task-level [`geom::DEVICE`] BAR window — stamped only for a row that claims
/// a PCI function with a large BAR (the NPU's 16 MiB BAR0).  v7 widens the
/// table (10 → 12) and adds the task-level [`geom::HEAP`] window, leaving one
/// entry spare as v4 did.
pub const GEOM_ABI: u16 = 7;

/// Window capacity of a [`GeomBlock`] — design-bounded like the handle table.
/// A task learns at most a handful of founder-mapped windows (scratch, slots,
/// map cursor, heap, ring); the v4 widening added a tenth entry, now named by
/// [`geom::DEVICE`] (v5), and v7 widened to twelve for [`geom::HEAP`].
pub const MAX_GEOM_WINDOWS: usize = 12;
// Eleven founder entries are now stampable (SCRATCH, SLOTS, MAP, LINK,
// THREAD_BLOCK, APERTURE, TRANSIENT, SLOTS_EXT, SURFACES — display row only —
// DEVICE — device-claiming rows only — and HEAP); one entry is spare.  A
// thirteenth window needs a `MAX_GEOM_WINDOWS` bump (a wire change, the
// GEOM_ABI v4/v7 precedent).
const _: () = assert!(11 <= MAX_GEOM_WINDOWS);

/// One geometry window: a `(va, len)` the founder mapped, tagged by [`geom`]
/// role (+ an `index` so a task could hold several of one kind).  24 bytes,
/// little-endian, unaligned-safe — written through any delegated mapping with
/// plain `IntoBytes`, no serializer, exactly like [`HandleEntry`].
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct GeomEntry {
    /// Geometry role discriminant ([`geom`]).
    pub role: U16,
    /// Instance index within the role.
    pub index: U16,
    /// Reserved; must be zero.  Frozen additive headroom (keeps `va` 8-aligned
    /// within the entry).
    pub reserved: U32,
    /// The window's base virtual address (the founder's mapping).
    pub va: U64,
    /// The window's byte length.
    pub len: U64,
}

/// The **geometry block** — the third stamped block on the boot page (after
/// [`BootBlock`] and [`LdBlock`](crate::image::LdBlock)), carrying the `(va,
/// len)` windows a task used to learn from hand-copied fixed constants
/// (`M2B_SCRATCH_VA`, `SLOTS_VA`, …).  Same additive discipline as its
/// siblings: 8-byte header + [`MAX_GEOM_WINDOWS`] entries = 200 bytes, every
/// field LE and unaligned-safe, an unstamped (all-zero) block invalid until the
/// founder writes it.
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct GeomBlock {
    /// [`GEOM_ABI`] — the stamp a validated lookup checks (an unseeded block
    /// reads zero and every window refuses).
    pub abi: U16,
    /// Reserved; must be zero.  Frozen additive headroom.
    pub flags: U16,
    /// Live entries in `windows` (`<= MAX_GEOM_WINDOWS`).
    pub count: U16,
    /// Reserved; must be zero.
    pub reserved: U16,
    /// The geometry table; entries past `count` are zero.
    pub windows: [GeomEntry; MAX_GEOM_WINDOWS],
}

impl GeomBlock {
    /// An empty, correctly-stamped block — the founder's starting value (and
    /// the child image's pre-write static: all-zero, hence *invalid* until the
    /// founder stamps it).
    pub const fn new() -> Self {
        Self {
            abi: U16::new(GEOM_ABI),
            flags: U16::new(0),
            count: U16::new(0),
            reserved: U16::new(0),
            windows: [GeomEntry {
                role: U16::new(0),
                index: U16::new(0),
                reserved: U32::new(0),
                va: U64::new(0),
                len: U64::new(0),
            }; MAX_GEOM_WINDOWS],
        }
    }

    /// Record a mapped window at `(role, index)` spanning `[va, va+len)`.
    /// Refuses (`false`) when the table is full — bounded, never dropped.
    pub fn push(&mut self, role: u16, index: u16, va: u64, len: u64) -> bool {
        let n = self.count.get() as usize;
        if n >= MAX_GEOM_WINDOWS {
            return false;
        }
        self.windows[n] = GeomEntry {
            role: U16::new(role),
            index: U16::new(index),
            reserved: U32::new(0),
            va: U64::new(va),
            len: U64::new(len),
        };
        self.count = U16::new(n as u16 + 1);
        true
    }

    /// Record a typed [`Window`](layout::Window) at `(role, index)` — the
    /// form a stamper that already holds a checked window should use, since
    /// it cannot pair one window's base with another's length on the way in.
    /// Refuses (`false`) on a full table, exactly like [`push`](Self::push).
    pub fn push_window(&mut self, role: u16, index: u16, window: layout::Window) -> bool {
        self.push(role, index, window.base(), window.len())
    }

    /// The `(role, 0)` window as a typed [`Window`](layout::Window) —
    /// [`window`](Self::window)'s form for a consumer that will carry the
    /// span onward rather than immediately destructure it.  `None` on the
    /// same fail-closed conditions.
    pub fn window_of(&self, role: u16) -> Option<layout::Window> {
        let (va, len) = self.window(role)?;
        layout::Window::try_new(va, len)
    }

    /// Validate the complete geometry table before any window is trusted.
    /// Unknown flags, non-zero reserved fields, over-counts, duplicate role
    /// keys, empty windows, and overflowing ranges all refuse.
    pub fn validate(&self) -> bool {
        let n = self.count.get() as usize;
        if self.abi.get() != GEOM_ABI
            || self.flags.get() != 0
            || self.reserved.get() != 0
            || n > MAX_GEOM_WINDOWS
        {
            return false;
        }
        for (i, window) in self.windows[..n].iter().enumerate() {
            if window.reserved.get() != 0
                || window.len.get() == 0
                || window.va.get().checked_add(window.len.get()).is_none()
                || self.windows[..i].iter().any(|earlier| {
                    earlier.role.get() == window.role.get()
                        && earlier.index.get() == window.index.get()
                })
            {
                return false;
            }
        }
        true
    }

    /// The `(va, len)` of the `(role, 0)` window, or `None` if unstamped
    /// (`abi != GEOM_ABI`), out of range, or absent — fail-closed, so a task
    /// whose founder did not carry a window gets no address to trust.
    pub fn window(&self, role: u16) -> Option<(u64, u64)> {
        self.window_indexed(role, 0)
    }

    /// The `(va, len)` of the `(role, index)` window — [`window`](Self::window)
    /// with an explicit instance index.
    pub fn window_indexed(&self, role: u16, index: u16) -> Option<(u64, u64)> {
        if !self.validate() {
            return None;
        }
        let n = self.count.get() as usize;
        self.windows[..n]
            .iter()
            .find(|w| w.role.get() == role && w.index.get() == index)
            .map(|w| (w.va.get(), w.len.get()))
    }
}

impl Default for GeomBlock {
    fn default() -> Self {
        Self::new()
    }
}

/// # The typed address-layout vocabulary (`res/USERLAND.md` §4)
///
/// Address plans in this tree used to be **bare `u64` constants** with their
/// lengths kept in a sibling constant — or in a comment: a `SCRATCH_VA` here,
/// a `MAP_STRIDE` there, per-thread VAs computed as `BAND + n * STRIDE`, and
/// nothing checking that the windows so named did not overlap, that a thread
/// index stayed inside its band, or that a stride was large enough for the
/// window it strided.  Every consumer re-derived the base/length pairing by
/// hand, and a category error (a boot page's length used as a thread block's
/// window length) read exactly like correct code.
///
/// This module answers that the way the limine request/response API does: the
/// invalid case is not *detected*, it is **unrepresentable**.  A [`Window`]
/// carries its own length, so the two halves can never be mis-paired or
/// mis-ordered; a [`Band`] carries its stride **and** its count, so a
/// per-thread index cannot run off the end and a stride cannot be narrower
/// than the window it strides; and both expose `const` predicates, so a
/// plan's disjointness is a `const _: () = assert!(…)` — a BUILD ERROR rather
/// than a boot-time surprise or a page-table collision.
///
/// It lives in `kore-abi` for two reasons.  A [`GeomEntry`] is literally a
/// `(role, index, va, len)` row, so a window IS the value the geometry table
/// transports — this is that value, named.  And `kore-abi` is the only crate
/// both `kore-rt` (whose `thread` bands carve per-thread windows) and
/// `kore-svc-exec` (whose founder plan names a task's own windows) depend on,
/// which is what lets the two be proven *equal* at compile time instead of by
/// matching comments in two crates.
pub mod layout {
    /// A **half-open virtual-address window** — `[base, base + len)` carried
    /// as one value.
    ///
    /// The fields are private on purpose.  A `Window` can only be built by
    /// [`new`](Window::new) or [`try_new`](Window::try_new), both of which
    /// reject the empty window and the window that wraps the address space,
    /// so every `Window` in the tree names a real span; and because base and
    /// length arrive together and are read back through accessors, no caller
    /// can pair one window's base with another's length.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Window {
        base: u64,
        len: u64,
    }

    impl Window {
        /// The window `[base, base + len)`.
        ///
        /// # Panics
        /// On an empty window (`len == 0`) or one that would wrap past
        /// `u64::MAX`.  In a `const` initializer — which is where every
        /// declared plan is built — that panic is a **compile error**, so an
        /// impossible window never reaches a running kernel.
        pub const fn new(base: u64, len: u64) -> Self {
            assert!(len != 0, "a window with no length names nothing");
            assert!(
                base.checked_add(len).is_some(),
                "a window may not wrap the address space"
            );
            Self { base, len }
        }

        /// [`new`](Self::new)'s fallible twin, for a window computed from
        /// runtime data (a stamped [`GeomBlock`](super::GeomBlock) entry, a
        /// negotiated span) where a refusal is the lawful answer.
        pub const fn try_new(base: u64, len: u64) -> Option<Self> {
            if len == 0 || base.checked_add(len).is_none() {
                return None;
            }
            Some(Self { base, len })
        }

        /// The window's base virtual address.
        pub const fn base(&self) -> u64 {
            self.base
        }

        /// The window's byte length.
        pub const fn len(&self) -> u64 {
            self.len
        }

        /// One past the window's last byte (`base + len`) — the *exclusive*
        /// end, so two windows that abut are disjoint.
        pub const fn end(&self) -> u64 {
            self.base + self.len
        }

        /// Always `false` — a `Window` cannot be empty by construction.
        /// Present so `len()`'s clippy pairing reads honestly rather than
        /// inviting a caller to test for a state that cannot exist.
        pub const fn is_empty(&self) -> bool {
            false
        }

        /// Whether `addr` falls inside this window.
        pub const fn contains_addr(&self, addr: u64) -> bool {
            addr >= self.base && addr < self.end()
        }

        /// Whether `other` lies wholly within this window (the *carve*
        /// relation: a per-thread slot inside the task's band).
        pub const fn contains(&self, other: &Window) -> bool {
            other.base >= self.base && other.end() <= self.end()
        }

        /// Whether the two windows share a byte.  Abutting windows
        /// (`a.end() == b.base()`) do **not** overlap — the map window
        /// abutting the scratch window exactly is a deliberate part of the
        /// plan, not a collision.
        pub const fn overlaps(&self, other: &Window) -> bool {
            self.base < other.end() && other.base < self.end()
        }

        /// Whether this window ends at or before `other` begins.
        pub const fn precedes(&self, other: &Window) -> bool {
            self.end() <= other.base
        }

        /// Structural equality, callable from a `const` assertion (`==` is
        /// not).  This is what proves one crate's window IS another's.
        pub const fn same_as(&self, other: &Window) -> bool {
            self.base == other.base && self.len == other.len
        }

        /// Whether base **and** length are multiples of `align` — the check
        /// a page-granular plan owes, since a window that is page-aligned at
        /// one end only cannot be mapped as stated.
        ///
        /// # Panics
        /// On a zero or non-power-of-two `align` (a compile error at a
        /// `const` call site).
        pub const fn is_aligned_to(&self, align: u64) -> bool {
            assert!(
                align != 0 && align.is_power_of_two(),
                "alignment must be a power of two"
            );
            self.base % align == 0 && self.len % align == 0
        }

        /// **Are these windows pairwise disjoint?** — the whole-plan
        /// predicate, `O(n²)` over a handful of entries and fully
        /// const-evaluable, so a plan proves its own soundness in a
        /// `const _: () = assert!(…)`.
        ///
        /// Order-independent by design: a plan that grows a row does not
        /// have to be re-sorted to keep its proof, and a row inserted in the
        /// wrong place is still caught.
        pub const fn all_disjoint(windows: &[Window]) -> bool {
            let mut i = 0;
            while i < windows.len() {
                let mut j = i + 1;
                while j < windows.len() {
                    if windows[i].overlaps(&windows[j]) {
                        return false;
                    }
                    j += 1;
                }
                i += 1;
            }
            true
        }

        /// Whether every window is [aligned](Self::is_aligned_to) to
        /// `align` — the plan-wide companion to [`all_disjoint`].
        ///
        /// [`all_disjoint`]: Self::all_disjoint
        pub const fn all_aligned(windows: &[Window], align: u64) -> bool {
            let mut i = 0;
            while i < windows.len() {
                if !windows[i].is_aligned_to(align) {
                    return false;
                }
                i += 1;
            }
            true
        }
    }

    /// A **band** — a strided array of equal-length [`Window`]s, one per
    /// index, occupying `[base, base + stride · count)`.
    ///
    /// This is the shape a per-thread address plan actually has: thread `n`'s
    /// window sits at `base + n · stride`, with the gap between `len` and
    /// `stride` left for guard pages.  Spelled as three loose constants
    /// (`BAND`, `STRIDE`, and a length held in a comment) it admitted two
    /// silent failures — a stride narrower than the window it strides, so
    /// consecutive threads' windows overlap; and an index past the band's
    /// room, so a thread's window lands in the next band.  As a `Band` the
    /// first is rejected in the constructor (a compile error for a declared
    /// band) and the second is impossible to express: [`nth`](Band::nth)
    /// returns `Option`, and there is no other way to name a slot.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Band {
        first: Window,
        stride: u64,
        count: u32,
    }

    impl Band {
        /// A band of `count` windows of `len` bytes each, `stride` bytes
        /// apart, starting at `base`.
        ///
        /// # Panics
        /// On `stride < len` (the windows would overlap each other), on
        /// `count == 0` (a band with no slots), and on a span that wraps the
        /// address space — each a compile error for a declared band.
        pub const fn new(base: u64, len: u64, stride: u64, count: u32) -> Self {
            let first = Window::new(base, len);
            assert!(
                stride >= len,
                "a band's stride must cover the window it strides"
            );
            assert!(count != 0, "a band with no slots names nothing");
            // The whole span must exist: `base + stride · count` cannot wrap.
            assert!(
                stride.checked_mul(count as u64).is_some()
                    && base.checked_add(stride * count as u64).is_some(),
                "a band may not wrap the address space"
            );
            Self {
                first,
                stride,
                count,
            }
        }

        /// Slot `n`'s window, or `None` once the band is exhausted — the
        /// bound that used to be nobody's job.  A caller that has a `Window`
        /// therefore *has* a slot that fits.
        pub const fn nth(&self, n: u32) -> Option<Window> {
            if n >= self.count {
                return None;
            }
            Some(Window {
                base: self.first.base() + self.stride * n as u64,
                len: self.first.len(),
            })
        }

        /// The slot whose window starts exactly at `base`, or `None` when
        /// `base` names no slot of this band — the inverse of
        /// [`nth`](Self::nth).
        ///
        /// A band slot is held exclusively by one live thread, so this is
        /// how a thread recovers its own **identity** from a window address
        /// it was handed, without a second source of truth to drift from
        /// (`kore-rt`'s allocator derives its shard this way).  It is exact
        /// rather than nearest: a `base` interior to a slot, misaligned to
        /// the stride, or past the band refuses, because an address that is
        /// not a slot base is not evidence of a slot.
        pub const fn slot_of(&self, base: u64) -> Option<u32> {
            if base < self.first.base() {
                return None;
            }
            let delta = base - self.first.base();
            if delta % self.stride != 0 {
                return None;
            }
            let n = delta / self.stride;
            if n >= self.count as u64 {
                return None;
            }
            Some(n as u32)
        }

        /// Slot 0's window — total, since a band always has at least one
        /// slot.  For an *aliased* band (one whose slot 0 is some other
        /// plan's window) this is what proves the two agree.
        pub const fn first(&self) -> Window {
            self.first
        }

        /// The band's whole extent, `[base, base + stride · count)` — the
        /// value that joins a plan's disjointness proof, since the band's
        /// claim on the address space is its span, not its first slot.
        pub const fn span(&self) -> Window {
            Window {
                base: self.first.base(),
                len: self.stride * self.count as u64,
            }
        }

        /// How many slots the band holds.
        pub const fn count(&self) -> u32 {
            self.count
        }

        /// The distance between consecutive slots.
        pub const fn stride(&self) -> u64 {
            self.stride
        }

        /// Bytes between the end of one slot's window and the start of the
        /// next — the guard gap the stride leaves.  `0` for a band whose
        /// windows tile it exactly.
        pub const fn gap(&self) -> u64 {
            self.stride - self.first.len()
        }

        /// Structural equality, callable from a `const` assertion (`==` is
        /// not) — [`Window::same_as`] over the whole subdivision: base,
        /// slot length, stride, **and** count.
        ///
        /// [`span`](Self::span) equality is deliberately not enough to prove
        /// two crates mean one band.  Two plans can agree on the reservation
        /// to the byte and still disagree about where each slot starts — and
        /// a slot boundary is exactly what separates one thread's window
        /// from its neighbour's, so a span-only proof would let the very
        /// collision the band exists to prevent back in.
        pub const fn same_as(&self, other: &Band) -> bool {
            self.first.same_as(&other.first)
                && self.stride == other.stride
                && self.count == other.count
        }
    }
}

/// Why a boot block failed validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BootstrapError {
    /// The ABI stamp is not [`BOOTSTRAP_ABI`] — a stale image, or an
    /// unseeded (all-zero) block.
    BadAbi,
    /// `count` exceeds [`MAX_BOOT_HANDLES`] — a corrupt write.
    BadCount,
    /// `data_count` exceeds [`MAX_BOOT_DATA`] — a corrupt write.
    BadDataCount,
    /// Two capability entries publish the same `(role, index)` key.
    DuplicateHandle,
    /// Two literal entries publish the same `(role, index)` key.
    DuplicateData,
}

/// The **validated view** over a task's boot block — the only door to the
/// handle table, so an unchecked block cannot reach a resolver.  Zero-copy
/// and allocation-free.
#[derive(Clone, Copy, Debug)]
pub struct Bootstrap<'a> {
    handles: &'a [HandleEntry],
    data: &'a [DatumEntry],
}

impl<'a> Bootstrap<'a> {
    /// Validate `block` (ABI stamp + count bound) and take the view.
    pub fn of(block: &'a BootBlock) -> Result<Self, BootstrapError> {
        if block.abi.get() != BOOTSTRAP_ABI {
            return Err(BootstrapError::BadAbi);
        }
        let n = block.count.get() as usize;
        if n > MAX_BOOT_HANDLES {
            return Err(BootstrapError::BadCount);
        }
        let nd = block.data_count.get() as usize;
        if nd > MAX_BOOT_DATA {
            return Err(BootstrapError::BadDataCount);
        }
        for i in 0..n {
            for j in i + 1..n {
                if block.handles[i].role.get() == block.handles[j].role.get()
                    && block.handles[i].index.get() == block.handles[j].index.get()
                {
                    return Err(BootstrapError::DuplicateHandle);
                }
            }
        }
        for i in 0..nd {
            for j in i + 1..nd {
                if block.data[i].role.get() == block.data[j].role.get()
                    && block.data[i].index.get() == block.data[j].index.get()
                {
                    return Err(BootstrapError::DuplicateData);
                }
            }
        }
        Ok(Self {
            handles: &block.handles[..n],
            data: &block.data[..nd],
        })
    }

    /// The arena slot of the capability at `(role, index)`, if delegated.
    pub fn handle(&self, role: u16, index: u16) -> Option<SlotRef> {
        self.handles
            .iter()
            .find(|h| h.role.get() == role && h.index.get() == index)
            .map(|h| SlotRef::from_wire(h.slot.get()))
    }

    /// All delegated handles as `(role, index, slot)`.
    pub fn handles(&self) -> impl Iterator<Item = (u16, u16, SlotRef)> + '_ {
        self.handles.iter().map(|h| {
            (
                h.role.get(),
                h.index.get(),
                SlotRef::from_wire(h.slot.get()),
            )
        })
    }

    /// The immutable literal at `(role, index)`, if the founder published
    /// one.  Unlike [`handle`](Self::handle), this value names no capability
    /// in the caller's arena.
    pub fn datum(&self, role: u16, index: u16) -> Option<u32> {
        self.data
            .iter()
            .find(|d| d.role.get() == role && d.index.get() == index)
            .map(|d| d.value.get())
    }

    /// All immutable startup data as `(role, index, value)`.
    pub fn data(&self) -> impl Iterator<Item = (u16, u16, u32)> + '_ {
        self.data
            .iter()
            .map(|d| (d.role.get(), d.index.get(), d.value.get()))
    }
}

/// One role a [`Profile`] declares — a `(role, index)` coordinate the founding
/// side binds to an actual arena slot.  The instance `index` distinguishes
/// several caps of one role (two devices, say — matching [`HandleEntry`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoleSlot {
    /// Role discriminant ([`role`]).
    pub role: u16,
    /// Instance index within the role.
    pub index: u16,
    /// Whether this position may stamp **VACANT** — the row declaring
    /// that the coordinate names something the machine might not have,
    /// and that it can serve without it.
    ///
    /// The founding side reads this instead of knowing, itself, which
    /// coordinates are allowed to be missing.  A REQUIRED position that
    /// cannot be seeded still refuses the whole founding; an optional
    /// one stamps vacant, the child's `boot.handle` answers `None`, and
    /// the bundle's `Option<T>` field resolves it.  The founding witness
    /// then proves the vacancy POSITIVELY (the position must be
    /// unstamped), so "optional" never degrades into "unchecked".
    ///
    /// Availability stays the founder's to determine — this declares the
    /// POLICY (may this be absent?), never the fact (is it absent?).
    pub optional: bool,
}

impl RoleSlot {
    /// A role at instance 0 — the common case.
    pub const fn at(role: u16) -> Self {
        Self {
            role,
            index: 0,
            optional: false,
        }
    }

    /// A role at an explicit instance index.
    pub const fn indexed(role: u16, index: u16) -> Self {
        Self {
            role,
            index,
            optional: false,
        }
    }

    /// A role at an explicit instance index that the machine may not
    /// have — see [`optional`](Self::optional).
    pub const fn indexed_optional(role: u16, index: u16) -> Self {
        Self {
            role,
            index,
            optional: true,
        }
    }
}

/// A **capability profile** — the structured, reusable declaration of the role
/// set a task *type* is founded with (a driver: device + DMA pool + IRQ route).
/// It is the founding-side counterpart to a service's
/// `Bundle`: **the same role list the bundle later resolves**, so declaring it
/// once (a service's `PROFILE`) keeps the two sides from drifting.  Pure
/// declaration — `&'static` data + one [`stamp`](Profile::stamp) that is
/// [`BootBlock::push`] in a loop (no unsafe, no policy): the KRAPH no-super-init
/// discipline made ergonomic, replacing ad-hoc per-role stamping.
#[derive(Clone, Copy, Debug)]
pub struct Profile {
    /// Diagnostic name (the task *type* — "driver", "mm", …).
    pub name: &'static str,
    /// The roles a task of this type is founded with, in stamp order.
    pub roles: &'static [RoleSlot],
}

impl Profile {
    /// A profile over a static role list.
    pub const fn new(name: &'static str, roles: &'static [RoleSlot]) -> Self {
        Self { name, roles }
    }

    /// How many roles this profile declares.
    pub const fn len(&self) -> usize {
        self.roles.len()
    }

    /// Whether the profile declares no roles (the endowment-less task).
    pub const fn is_empty(&self) -> bool {
        self.roles.is_empty()
    }

    /// **Stamp** this profile into `block`, binding each declared role
    /// positionally to `slots` (role `i` ← `slots[i]`).  **All-or-nothing**:
    /// an arity mismatch, or a table that cannot hold the whole profile,
    /// refuses *before* writing — so a partially-stamped block never exists
    /// (the bounded-emission discipline, `res/SYSCALL.md` D4).
    ///
    /// A `None` position is a **VACANT** endowment: the founding selected
    /// nothing for it (a non-required `DeviceRequest` that firmware
    /// described no match for), so the role is not stamped at all and the
    /// child's `boot.handle(role, index)` answers `None` — which is exactly
    /// what a fail-closed [`Bundle::resolve`] reads as absent.  Vacancy is
    /// per *position*, so the arity check still holds: the profile and the
    /// founding side agree on the whole list, they only disagree about
    /// which entries the machine could fill.
    ///
    /// [`Bundle::resolve`]: https://docs.rs/kore-rt
    pub fn stamp(
        &self,
        block: &mut BootBlock,
        slots: &[Option<SlotRef>],
    ) -> Result<(), ProfileError> {
        if slots.len() != self.roles.len() {
            return Err(ProfileError::Arity {
                expected: self.roles.len(),
                got: slots.len(),
            });
        }
        // Atomic: refuse before writing if the table cannot hold it all or
        // any filled profile role collides with an existing/publication peer.
        // Only the filled positions are ever pushed, so a row whose vacant
        // device would have overflowed the block still founds.
        let filled = slots.iter().flatten().count();
        if block.count.get() as usize + filled > MAX_BOOT_HANDLES {
            return Err(ProfileError::Overflow);
        }
        for (i, (role_slot, slot)) in self.roles.iter().zip(slots).enumerate() {
            if slot.is_none() {
                continue;
            }
            let n = block.count.get() as usize;
            if block.handles[..n].iter().any(|entry| {
                entry.role.get() == role_slot.role && entry.index.get() == role_slot.index
            }) || self.roles[..i]
                .iter()
                .zip(&slots[..i])
                .any(|(earlier, earlier_slot)| {
                    earlier_slot.is_some()
                        && earlier.role == role_slot.role
                        && earlier.index == role_slot.index
                })
            {
                return Err(ProfileError::Duplicate {
                    role: role_slot.role,
                    index: role_slot.index,
                });
            }
        }
        for (rs, slot) in self.roles.iter().zip(slots) {
            let Some(slot) = slot else {
                continue; // vacant — the role is simply not published
            };
            // Room was pre-checked, so `push` cannot refuse here.
            let ok = block.push(rs.role, rs.index, *slot);
            debug_assert!(ok, "room pre-checked; push cannot fail");
        }
        Ok(())
    }
}

/// The most roles a [`Profile`] can declare and still stamp.  The boot
/// block's own capacity is what bounds it ([`Profile::stamp`] refuses past
/// it), so a founding side sizing a per-role array — the endowment slot
/// list, one entry per declared role including the vacant ones — needs no
/// second number of its own.
pub const MAX_PROFILE_ROLES: usize = MAX_BOOT_HANDLES;

/// Why a [`Profile::stamp`] refused — never a partial write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileError {
    /// `slots.len()` did not match the profile's role count.
    Arity {
        /// Roles the profile declares.
        expected: usize,
        /// Slots supplied.
        got: usize,
    },
    /// The block cannot hold the whole profile ([`MAX_BOOT_HANDLES`]).
    Overflow,
    /// A filled profile role duplicates an existing or earlier role key.
    Duplicate {
        /// Duplicated role discriminant.
        role: u16,
        /// Duplicated role instance.
        index: u16,
    },
}

bitflags! {
    /// Spawn parameters — the [`SpawnReq::flags`] word.
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
    pub struct SpawnFlags: u32 {
        /// The new task **shares** the spawner's address space (a *thread*);
        /// unset spawns into a **fresh** address space (a *process*). This bit
        /// is the entire kernel-visible thread-vs-process distinction.
        const SHARE_SPACE = 1 << 0;
        /// Start detached — the spawner keeps no join handle.
        const DETACHED = 1 << 1;
        /// An SPMD worker founding (`res/SPMD.md` §7, G6): the kernel
        /// pins this task to ONE class-chosen core (a single `CpuShare`,
        /// class-homogeneous across a region's workers) instead of the
        /// whole-node affinity an ordinary founding receives — a
        /// barrier-synced region runs every phase at its slowest worker,
        /// so scattering across hybrid core classes (or migrating
        /// mid-region) is the measured pathology this bit exists to
        /// stop.  Lawful only on a `SHARE_SPACE` thread founding; any
        /// other shape refuses at dispatch.
        const SPMD_WORKER = 1 << 2;
    }
}

/// The parameters a parent stages to spawn a task (thread or process). The
/// address-space view / fresh root, stack region, and CPU share are staged as
/// `derive` invocations in the same session (Flow 1); this carries the rest.
/// 24 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, KnownLayout, Immutable, Unaligned)]
pub struct SpawnReq {
    /// Entry-point virtual address in the new task.
    pub entry: U64,
    /// Initial stack-pointer virtual address.
    pub stack: U64,
    /// Arena slot of the delegated region containing the child's
    /// [`BootBlock`] static (its `.kore.boot` page — what the spawner
    /// writes through before first dispatch), or `u32::MAX` for none.
    pub bootstrap: U32,
    /// [`SpawnFlags`].
    pub flags: U32,
}

impl SpawnReq {
    /// `bootstrap` sentinel meaning "no bootstrap region delegated".
    pub const NO_BOOTSTRAP: u32 = u32::MAX;
}

// Frozen wire pins (#81 — the crate header's freeze discipline).
const _: () = {
    use core::mem::size_of;
    assert!(size_of::<HandleEntry>() == 8);
    assert!(size_of::<DatumEntry>() == 8);
    assert!(size_of::<BootBlock>() == 8 + 8 * (MAX_BOOT_HANDLES + MAX_BOOT_DATA)); // 712
    // Publication and movement are separate bounded vocabularies.  Neither
    // count dominates the other: vacant landing names cost only a boot entry;
    // arithmetic grant-bank members cost only delivery rows.  Each wire uses
    // U16 counts/indices and Found rejects its own over-capacity vector.
    assert!(MAX_BOOT_HANDLES <= u16::MAX as usize);
    assert!(MAX_BOOT_DATA <= u16::MAX as usize);
    assert!(crate::session::MAX_FOUND_DELIVERIES <= u16::MAX as usize);
    assert!(size_of::<SpawnReq>() == 24);
    assert!(size_of::<GeomEntry>() == 24);
    assert!(size_of::<GeomBlock>() == 8 + 24 * MAX_GEOM_WINDOWS); // 296
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boot_block_finds_caps_by_role() {
        let mut b = BootBlock::new();
        assert!(b.push(role::DEVICE, 0, SlotRef::from_wire(7)));
        assert!(b.push(role::PARENT_SEND, 0, SlotRef::from_wire(3)));
        assert!(b.push(role::DEVICE, 1, SlotRef::from_wire(9)));
        let bs = Bootstrap::of(&b).expect("a stamped block validates");

        // Caps by role (a driver takes its device + its parent channel).
        assert_eq!(bs.handle(role::DEVICE, 0), Some(SlotRef::from_wire(7)));
        assert_eq!(
            bs.handle(role::DEVICE, 1),
            Some(SlotRef::from_wire(9)),
            "a second device"
        );
        assert_eq!(bs.handle(role::PARENT_SEND, 0), Some(SlotRef::from_wire(3)));
        // A role the block never carried reads as absent rather than as some
        // neighbouring slot.  `LOG_SINK` is the case worth pinning: it is a
        // reserved discriminant nothing implements, so this is also the only
        // place in the tree that names it — deliberately, and as a negative.
        assert_eq!(bs.handle(role::LOG_SINK, 0), None, "not delegated");
        assert_eq!(bs.handles().count(), 3);
    }

    #[test]
    fn boot_block_refuses_ambiguous_role_keys() {
        let mut block = BootBlock::new();
        assert!(block.push(role::SVC_GRANT, 1, SlotRef::from_wire(7)));
        assert!(
            !block.push(role::SVC_GRANT, 1, SlotRef::from_wire(9)),
            "a later slot must not shadow the first role publication"
        );
        assert!(block.push_data(role::SVC_RETURN, 0, 41));
        assert!(!block.push_data(role::SVC_RETURN, 0, 42));

        // Validation is a second boundary: a corrupt or independently
        // produced wire block cannot bypass the write-side guard.
        block.handles[1] = HandleEntry {
            role: U16::new(role::SVC_GRANT),
            index: U16::new(1),
            slot: U32::new(9),
        };
        block.count = U16::new(2);
        assert_eq!(
            Bootstrap::of(&block).unwrap_err(),
            BootstrapError::DuplicateHandle
        );

        let mut data = BootBlock::new();
        data.data[0] = DatumEntry {
            role: U16::new(role::SVC_RETURN),
            index: U16::new(0),
            value: U32::new(41),
        };
        data.data[1] = DatumEntry {
            role: U16::new(role::SVC_RETURN),
            index: U16::new(0),
            value: U32::new(42),
        };
        data.data_count = U16::new(2);
        assert_eq!(
            Bootstrap::of(&data).unwrap_err(),
            BootstrapError::DuplicateData
        );
    }

    #[test]
    fn boot_data_never_aliases_a_handle_or_raises_its_slot_run() {
        let mut b = BootBlock::new();
        assert!(b.push(role::ADDRESS_SPACE, 0, SlotRef::from_wire(7)));
        assert!(b.push_data(role::SVC_RETURN, 1, 0xffff_0003));
        let bs = Bootstrap::of(&b).expect("a stamped block validates");

        assert_eq!(bs.handle(role::SVC_RETURN, 1), None);
        assert_eq!(bs.datum(role::SVC_RETURN, 1), Some(0xffff_0003));
        assert_eq!(
            bs.handles().map(|(_, _, slot)| slot.index()).max(),
            Some(7),
            "opaque data cannot become crt0's arena high-water mark"
        );
    }

    #[test]
    fn validation_refuses_unseeded_stale_and_corrupt_blocks() {
        // The child image's pre-write initializer is all-zero — reading
        // before the spawner stamps it must refuse (abi = 0).
        let zeroed = BootBlock::read_from_bytes(&[0u8; size_of::<BootBlock>()]).unwrap();
        assert_eq!(Bootstrap::of(&zeroed).unwrap_err(), BootstrapError::BadAbi);

        // A stale image (v1 stamp).
        let mut stale = BootBlock::new();
        stale.abi = U16::new(1);
        assert_eq!(Bootstrap::of(&stale).unwrap_err(), BootstrapError::BadAbi);

        // A corrupt count past the table.
        let mut corrupt = BootBlock::new();
        corrupt.count = U16::new(MAX_BOOT_HANDLES as u16 + 1);
        assert_eq!(
            Bootstrap::of(&corrupt).unwrap_err(),
            BootstrapError::BadCount
        );

        let mut corrupt_data = BootBlock::new();
        corrupt_data.data_count = U16::new(MAX_BOOT_DATA as u16 + 1);
        assert_eq!(
            Bootstrap::of(&corrupt_data).unwrap_err(),
            BootstrapError::BadDataCount
        );
    }

    #[test]
    fn the_table_is_bounded_and_refuses_overflow() {
        let mut b = BootBlock::new();
        for i in 0..MAX_BOOT_HANDLES {
            assert!(b.push(role::USER_BASE, i as u16, SlotRef::from_wire(i as u32)));
        }
        assert!(
            !b.push(role::USER_BASE, 99, SlotRef::from_wire(99)),
            "refused, not dropped"
        );
        let bs = Bootstrap::of(&b).unwrap();
        assert_eq!(bs.handles().count(), MAX_BOOT_HANDLES);
    }

    /// The write side is `IntoBytes` — the whole spawner-side "serializer":
    /// the block's bytes land in the child page verbatim and read back
    /// identical.  Layout is the ABI; nothing else exists to test.
    #[test]
    fn the_block_round_trips_as_plain_bytes() {
        let mut b = BootBlock::new();
        b.push(role::MEM_SUPPLY, 0, SlotRef::from_wire(42));
        b.push_data(role::SVC_RETURN, 1, 0xffff_0001);
        let bytes = b.as_bytes();
        assert_eq!(bytes.len(), size_of::<BootBlock>());
        let back = BootBlock::read_from_bytes(bytes).unwrap();
        let bs = Bootstrap::of(&back).unwrap();
        assert_eq!(bs.handle(role::MEM_SUPPLY, 0), Some(SlotRef::from_wire(42)));
        assert_eq!(bs.datum(role::SVC_RETURN, 1), Some(0xffff_0001));
    }

    #[test]
    fn spawn_flags_distinguish_thread_from_process() {
        assert!(SpawnFlags::SHARE_SPACE.contains(SpawnFlags::SHARE_SPACE));
        // A "thread" shares the space; a "process" does not.
        let thread = SpawnFlags::SHARE_SPACE;
        let process = SpawnFlags::empty();
        assert!(thread.contains(SpawnFlags::SHARE_SPACE));
        assert!(!process.contains(SpawnFlags::SHARE_SPACE));
    }

    #[test]
    fn an_empty_block_is_valid() {
        let b = BootBlock::new();
        let bs = Bootstrap::of(&b).expect("an empty context is valid");
        assert_eq!(bs.handles().count(), 0);
        assert_eq!(bs.handle(role::DEVICE, 0), None);
    }

    /// The driver profile stamps its three roles positionally, and the
    /// resulting block resolves each by role — the founding side (stamp) and
    /// the resolving side (`handle`) agree on one declared list.
    #[test]
    fn a_profile_stamps_its_roles_positionally() {
        const DRIVER: Profile = Profile::new(
            "driver",
            &[
                RoleSlot::at(role::DEVICE),
                RoleSlot::at(role::DMA_POOL),
                RoleSlot::at(role::IRQ_ROUTE),
            ],
        );
        assert_eq!(DRIVER.len(), 3);
        let mut b = BootBlock::new();
        DRIVER
            .stamp(
                &mut b,
                &[
                    Some(SlotRef::from_wire(4)),
                    Some(SlotRef::from_wire(6)),
                    Some(SlotRef::from_wire(5)),
                ],
            )
            .expect("arity matches");
        let bs = Bootstrap::of(&b).unwrap();
        assert_eq!(bs.handle(role::DEVICE, 0), Some(SlotRef::from_wire(4)));
        assert_eq!(bs.handle(role::DMA_POOL, 0), Some(SlotRef::from_wire(6)));
        assert_eq!(bs.handle(role::IRQ_ROUTE, 0), Some(SlotRef::from_wire(5)));
        assert_eq!(bs.handles().count(), 3);
    }

    /// An instance index distinguishes several caps of one role.
    #[test]
    fn a_profile_honors_instance_indices() {
        const TWO_DEV: Profile = Profile::new(
            "dual",
            &[
                RoleSlot::at(role::DEVICE),
                RoleSlot::indexed(role::DEVICE, 1),
            ],
        );
        let mut b = BootBlock::new();
        TWO_DEV
            .stamp(
                &mut b,
                &[Some(SlotRef::from_wire(7)), Some(SlotRef::from_wire(9))],
            )
            .unwrap();
        let bs = Bootstrap::of(&b).unwrap();
        assert_eq!(bs.handle(role::DEVICE, 0), Some(SlotRef::from_wire(7)));
        assert_eq!(bs.handle(role::DEVICE, 1), Some(SlotRef::from_wire(9)));
    }

    /// **A vacant position founds the row without the role.**  The second
    /// device of a two-device row is the live case: a machine whose
    /// firmware describes no i8042 still founds the input service, and the
    /// child reads `DEVICE@1` as absent rather than as some other slot's
    /// capability — the positions do not slide up to close the gap.
    #[test]
    fn a_vacant_position_is_skipped_without_shifting_the_rest() {
        const TWO_DEV: Profile = Profile::new(
            "dual",
            &[
                RoleSlot::at(role::DEVICE),
                RoleSlot::at(role::DMA_POOL),
                RoleSlot::indexed(role::DEVICE, 1),
            ],
        );
        let mut b = BootBlock::new();
        TWO_DEV
            .stamp(
                &mut b,
                &[
                    Some(SlotRef::from_wire(7)),
                    Some(SlotRef::from_wire(8)),
                    None,
                ],
            )
            .expect("a vacant position still stamps the rest");
        let bs = Bootstrap::of(&b).unwrap();
        assert_eq!(bs.handle(role::DEVICE, 0), Some(SlotRef::from_wire(7)));
        assert_eq!(bs.handle(role::DMA_POOL, 0), Some(SlotRef::from_wire(8)));
        assert_eq!(bs.handle(role::DEVICE, 1), None, "vacant reads as absent");
        assert_eq!(bs.handles().count(), 2, "only the filled positions publish");
    }

    /// A wholly vacant profile is still an arity match — and writes nothing.
    #[test]
    fn an_all_vacant_profile_stamps_nothing() {
        const P: Profile = Profile::new("p", &[RoleSlot::at(role::DEVICE)]);
        let mut b = BootBlock::new();
        P.stamp(&mut b, &[None]).expect("arity matches");
        assert!(b.is_empty(), "nothing was selected, nothing is published");
    }

    /// The overflow check counts only what will actually be pushed: a
    /// profile whose FILLED positions fit stamps even when its declared
    /// arity would not.  (Fail-closed the other way would refuse a row for
    /// authority it was never going to receive.)
    #[test]
    fn overflow_counts_the_filled_positions_only() {
        let mut b = BootBlock::new();
        for i in 0..(MAX_BOOT_HANDLES - 1) {
            assert!(b.push(role::USER_BASE, i as u16, SlotRef::from_wire(i as u32)));
        }
        const TWO: Profile = Profile::new(
            "two",
            &[
                RoleSlot::at(role::DEVICE),
                RoleSlot::indexed(role::DEVICE, 1),
            ],
        );
        TWO.stamp(&mut b, &[Some(SlotRef::from_wire(1)), None])
            .expect("one filled position fits the last entry");
        assert_eq!(b.count.get() as usize, MAX_BOOT_HANDLES);
    }

    /// Arity mismatch refuses without writing a single entry.
    #[test]
    fn a_profile_refuses_arity_mismatch_atomically() {
        const P: Profile = Profile::new(
            "p",
            &[RoleSlot::at(role::DEVICE), RoleSlot::at(role::DMA_POOL)],
        );
        let mut b = BootBlock::new();
        assert_eq!(
            P.stamp(&mut b, &[Some(SlotRef::from_wire(4))]),
            Err(ProfileError::Arity {
                expected: 2,
                got: 1
            })
        );
        assert!(b.is_empty(), "a refused stamp writes nothing");
    }

    /// A profile too large for the block's remaining room refuses before
    /// writing — no partial stamp.
    #[test]
    fn a_profile_refuses_overflow_atomically() {
        let mut b = BootBlock::new();
        // Fill all but two slots.
        for i in 0..(MAX_BOOT_HANDLES - 2) {
            assert!(b.push(role::USER_BASE, i as u16, SlotRef::from_wire(i as u32)));
        }
        const THREE: Profile = Profile::new(
            "three",
            &[
                RoleSlot::at(role::DEVICE),
                RoleSlot::at(role::DMA_POOL),
                RoleSlot::at(role::IRQ_ROUTE),
            ],
        );
        let before = b.count.get();
        assert_eq!(
            THREE.stamp(
                &mut b,
                &[
                    Some(SlotRef::from_wire(1)),
                    Some(SlotRef::from_wire(2)),
                    Some(SlotRef::from_wire(3)),
                ],
            ),
            Err(ProfileError::Overflow)
        );
        assert_eq!(b.count.get(), before, "overflow left the block untouched");
    }

    #[test]
    fn a_profile_refuses_duplicate_roles_atomically() {
        const DUPLICATE: Profile = Profile::new(
            "duplicate",
            &[RoleSlot::at(role::DEVICE), RoleSlot::at(role::DEVICE)],
        );
        let mut block = BootBlock::new();
        assert_eq!(
            DUPLICATE.stamp(
                &mut block,
                &[Some(SlotRef::from_wire(1)), Some(SlotRef::from_wire(2))]
            ),
            Err(ProfileError::Duplicate {
                role: role::DEVICE,
                index: 0,
            })
        );
        assert!(block.is_empty(), "a duplicate profile writes nothing");

        assert!(block.push(role::DMA_POOL, 0, SlotRef::from_wire(3)));
        const COLLIDES: Profile = Profile::new("collides", &[RoleSlot::at(role::DMA_POOL)]);
        let before = block.count.get();
        assert_eq!(
            COLLIDES.stamp(&mut block, &[Some(SlotRef::from_wire(4))]),
            Err(ProfileError::Duplicate {
                role: role::DMA_POOL,
                index: 0,
            })
        );
        assert_eq!(block.count.get(), before, "existing block stayed intact");
    }

    #[test]
    fn geom_block_round_trips_and_fails_closed() {
        let mut g = GeomBlock::new();
        assert!(g.push(geom::SCRATCH, 0, 0x6000_0000, 4096));
        assert!(g.push(geom::SLOTS, 0, 0x7000_0000, 16384));
        assert_eq!(g.window(geom::SCRATCH), Some((0x6000_0000, 4096)));
        assert_eq!(g.window(geom::SLOTS), Some((0x7000_0000, 16384)));
        // Absent role → None (fail-closed, not a bogus address).
        assert_eq!(g.window(role::HEAP), None);
    }

    #[test]
    fn an_unstamped_geom_block_trusts_nothing() {
        // All-zero (abi == 0) — a child image's pre-write static.
        let g = GeomBlock::read_from_bytes(&[0u8; core::mem::size_of::<GeomBlock>()]).unwrap();
        assert_eq!(g.abi.get(), 0);
        assert_eq!(
            g.window(geom::SCRATCH),
            None,
            "no stamp ⇒ no window to trust"
        );
    }

    #[test]
    fn geom_block_refuses_overflow_without_corruption() {
        let mut g = GeomBlock::new();
        for i in 0..MAX_GEOM_WINDOWS {
            assert!(g.push(geom::SLOTS, i as u16, 0x1000 * i as u64, 0x1000));
        }
        assert!(
            !g.push(geom::SLOTS, 99, 0xdead, 0x1000),
            "full table refuses"
        );
        assert_eq!(g.count.get() as usize, MAX_GEOM_WINDOWS);
    }

    // ─── The typed layout vocabulary ──────────────────────────────────────

    use layout::{Band, Window};

    #[test]
    fn a_window_carries_its_own_span() {
        let w = Window::new(0x6000_0000, 0x1000);
        assert_eq!(w.base(), 0x6000_0000);
        assert_eq!(w.len(), 0x1000);
        assert_eq!(w.end(), 0x6000_1000);
        assert!(w.contains_addr(0x6000_0fff));
        // The end is EXCLUSIVE: the first byte past the window is outside.
        assert!(!w.contains_addr(0x6000_1000));
    }

    #[test]
    fn abutting_windows_are_disjoint_but_straddling_ones_are_not() {
        // The map window abuts the scratch window exactly — deliberate, and
        // it must not read as a collision.
        let map = Window::new(0x5800_0000, 0x0800_0000);
        let scratch = Window::new(0x6000_0000, 0x1000);
        assert!(!map.overlaps(&scratch));
        assert!(map.precedes(&scratch));
        assert!(Window::all_disjoint(&[map, scratch]));

        // One byte of straddle is caught.
        let straddle = Window::new(0x5fff_f000, 0x2000);
        assert!(straddle.overlaps(&scratch));
        assert!(!Window::all_disjoint(&[map, scratch, straddle]));
    }

    #[test]
    fn all_disjoint_is_order_independent() {
        let a = Window::new(0x1000, 0x1000);
        let b = Window::new(0x3000, 0x1000);
        let c = Window::new(0x2000, 0x1000);
        // Listed out of address order and still proven sound …
        assert!(Window::all_disjoint(&[b, a, c]));
        // … while a genuine overlap is caught wherever it sits in the list.
        let bad = Window::new(0x2800, 0x1000);
        assert!(!Window::all_disjoint(&[b, a, bad]));
    }

    #[test]
    fn window_containment_is_the_carve_relation() {
        let band = Window::new(0x5800_0000, 0x0800_0000);
        let slot = Window::new(0x5900_0000, 0x0100_0000);
        assert!(band.contains(&slot));
        assert!(!slot.contains(&band));
        // A slot that runs one byte past the parent is not contained.
        let over = Window::new(0x5f00_0000, 0x0100_0001);
        assert!(!band.contains(&over));
    }

    #[test]
    fn alignment_is_checked_at_both_ends() {
        assert!(Window::new(0x6000_0000, 0x1000).is_aligned_to(0x1000));
        // A page-aligned base with a sub-page length is NOT a page window.
        assert!(!Window::new(0x6000_0000, 0x800).is_aligned_to(0x1000));
        assert!(!Window::new(0x6000_0800, 0x1000).is_aligned_to(0x1000));
    }

    #[test]
    fn an_empty_or_wrapping_window_cannot_be_built() {
        assert_eq!(Window::try_new(0x1000, 0), None);
        assert_eq!(Window::try_new(u64::MAX, 2), None);
        assert!(Window::try_new(u64::MAX - 1, 1).is_some());
    }

    #[test]
    #[should_panic(expected = "no length")]
    fn a_zero_length_window_panics_in_new() {
        let _ = Window::new(0x1000, 0);
    }

    #[test]
    fn a_band_bounds_its_own_index() {
        // Four 4 KiB windows, 8 KiB apart (a 4 KiB guard gap each).
        let b = Band::new(0x7fff_fd00_0000, 0x1000, 0x2000, 4);
        assert_eq!(b.count(), 4);
        assert_eq!(b.stride(), 0x2000);
        assert_eq!(b.gap(), 0x1000);
        assert_eq!(b.nth(0), Some(Window::new(0x7fff_fd00_0000, 0x1000)));
        assert_eq!(b.nth(3), Some(Window::new(0x7fff_fd00_6000, 0x1000)));
        // Past the band there is no window to hand out — the bound that
        // used to be nobody's job.
        assert_eq!(b.nth(4), None);
        assert_eq!(b.nth(u32::MAX), None);
    }

    #[test]
    fn a_bands_slots_never_overlap_each_other() {
        let b = Band::new(0x6000_0000, 0x1000, 0x1000, 8);
        let mut slots = [b.first(); 8];
        for (n, slot) in slots.iter_mut().enumerate() {
            *slot = b.nth(n as u32).expect("every slot below count exists");
        }
        assert!(Window::all_disjoint(&slots));
        // Tiled exactly: the span is the union of the slots.
        assert_eq!(b.gap(), 0);
        assert_eq!(b.span(), Window::new(0x6000_0000, 0x8000));
        assert_eq!(b.first(), b.nth(0).unwrap());
    }

    #[test]
    fn slot_of_inverts_nth_for_every_slot() {
        // The round trip is the whole claim: a thread handed slot `n`'s
        // window recovers exactly `n`, so two live threads holding distinct
        // slots can never report the same identity.
        let b = Band::new(0x6800_0000, 0x1000, 0x1000, 8);
        for n in 0..b.count() {
            let base = b.nth(n).expect("every slot below count exists").base();
            assert_eq!(b.slot_of(base), Some(n));
        }
    }

    #[test]
    fn slot_of_refuses_an_address_that_is_not_a_slot_base() {
        // A strided band with a gap: only the slot bases are slots. An
        // address interior to a window, one in the guard gap, one below the
        // band, and one past its count are all "no slot" — never a nearest
        // match, which would hand two threads one identity.
        let b = Band::new(0x7fff_fd00_0000, 0x1_0000, 0x2_0000, 4);
        assert_eq!(b.slot_of(0x7fff_fd00_0000), Some(0));
        assert_eq!(b.slot_of(0x7fff_fd00_0008), None, "interior to slot 0");
        assert_eq!(b.slot_of(0x7fff_fd01_0000), None, "inside the guard gap");
        assert_eq!(b.slot_of(0x7fff_fcff_0000), None, "below the band");
        assert_eq!(b.slot_of(0x7fff_fd08_0000), None, "past the last slot");
    }

    #[test]
    #[should_panic(expected = "stride must cover")]
    fn a_stride_narrower_than_its_window_is_rejected() {
        // 8 KiB windows 4 KiB apart would overlap each other — the failure a
        // loose `BAND` / `STRIDE` pair could express silently.
        let _ = Band::new(0x6000_0000, 0x2000, 0x1000, 4);
    }

    #[test]
    fn a_band_span_joins_a_plan_proof() {
        // A band's claim on the address space is its SPAN, not its first
        // slot: a plan that checked only slot 0 would miss this collision.
        let band = Band::new(0x6000_0000, 0x1000, 0x1000, 8);
        let neighbour = Window::new(0x6000_4000, 0x1000);
        assert!(!band.first().overlaps(&neighbour));
        assert!(band.span().overlaps(&neighbour));
    }

    #[test]
    fn band_equality_pins_the_subdivision_not_just_the_span() {
        // Same reservation, different slot boundaries — the shape a
        // span-only proof calls equal and a thread pays for.
        let eight = Band::new(0x5800_0000, 0x0100_0000, 0x0100_0000, 8);
        let four = Band::new(0x5800_0000, 0x0200_0000, 0x0200_0000, 4);
        assert!(
            eight.span().same_as(&four.span()),
            "the spans agree exactly"
        );
        assert!(
            !eight.same_as(&four),
            "but slot 1 sits in a different place"
        );
        assert!(eight.same_as(&Band::new(0x5800_0000, 0x0100_0000, 0x0100_0000, 8)));
        // A guard gap is part of the subdivision too.
        let gapped = Band::new(0x5800_0000, 0x0080_0000, 0x0100_0000, 8);
        assert!(eight.span().same_as(&gapped.span()));
        assert!(!eight.same_as(&gapped));
    }

    #[test]
    fn a_typed_window_round_trips_through_the_geom_block() {
        let mut g = GeomBlock::new();
        let scratch = Window::new(0x6000_0000, 0x1000);
        assert!(g.push_window(geom::SCRATCH, 0, scratch));
        assert_eq!(g.window_of(geom::SCRATCH), Some(scratch));
        assert_eq!(g.window_of(geom::MAP), None);
    }
}

/// Opaque, per-boot task identity. An identifier is never authority.
#[repr(transparent)]
#[derive(
    Clone, Copy, PartialEq, Eq, Hash, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct TaskId([u8; 16]);
impl TaskId {
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }
    pub const fn bytes(self) -> [u8; 16] {
        self.0
    }
}
impl core::fmt::Display for TaskId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}
impl core::fmt::Debug for TaskId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(self, f)
    }
}
/// Provenance, not a promise that the platform's seed was trustworthy.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskIdQuality {
    Unverified = 0,
    HardwareOrFirmware = 1,
    TimingJitter = 2,
    Degenerate = 3,
}
impl TaskIdQuality {
    pub const fn decode(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Unverified),
            1 => Some(Self::HardwareOrFirmware),
            2 => Some(Self::TimingJitter),
            3 => Some(Self::Degenerate),
            _ => None,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaskIdentity {
    pub id: TaskId,
    pub quality: TaskIdQuality,
}

/// Owned diagnostic label. Empty means unspecified; labels convey no authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaskName {
    bytes: [u8; 63],
    len: u8,
}
impl TaskName {
    pub const EMPTY: Self = Self {
        bytes: [0; 63],
        len: 0,
    };
    pub const CAPACITY: usize = 63;
    /// Reject oversized names instead of silently making distinct labels equal.
    pub fn new(name: &str) -> Option<Self> {
        if name.len() > Self::CAPACITY {
            return None;
        }
        let mut value = Self::EMPTY;
        value.bytes[..name.len()].copy_from_slice(name.as_bytes());
        value.len = name.len() as u8;
        Some(value)
    }
    pub fn as_str(&self) -> &str {
        core::str::from_utf8(&self.bytes[..self.len as usize]).expect("validated task name")
    }
}

#[cfg(test)]
mod diagnostic_name_tests {
    use super::TaskName;
    #[test]
    fn names_are_owned_utf8_and_bounded_in_bytes() {
        let mut source = *b"hello/worker-2";
        let name = TaskName::new(core::str::from_utf8(&source).unwrap()).unwrap();
        source.fill(0);
        assert_eq!(name.as_str(), "hello/worker-2");
        let ascii = [b'x'; 64];
        let text = core::str::from_utf8(&ascii).unwrap();
        assert!(TaskName::new(&text[..63]).is_some());
        assert!(TaskName::new(text).is_none());
        let utf8: [u8; 64] = core::array::from_fn(|i| if i % 2 == 0 { 0xc3 } else { 0xa9 });
        let text = core::str::from_utf8(&utf8).unwrap();
        assert!(TaskName::new(&text[..62]).is_some());
        assert!(TaskName::new(text).is_none());
    }
}
