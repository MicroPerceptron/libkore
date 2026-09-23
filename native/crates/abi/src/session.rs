//! Session ABI — a batch of capability invocations the kernel commits as one
//! KRAPH session (SYSCALL.md): one trap, `prepare` -> `commit`, closure
//! applied once. The descriptor is staged in the caller's arena; the kernel
//! reads it zero-copy.

use bitflags::bitflags;
use zerocopy::byteorder::little_endian::{I32, U16, U32, U64};
use zerocopy::{FromBytes, Immutable, KnownLayout, TryFromBytes, Unaligned};

use crate::cap::Rights;

/// Marks a session descriptor carried directly in the syscall's caller-saved
/// GPR bank rather than staged in the task's descriptor scratch.
///
/// The low byte carries the exact descriptor length.  No supported 64-bit ISA
/// has more than 31 general-purpose registers, so even an eventual all-GPR
/// carrier remains below 256 bytes.  Keeping every other control bit reserved
/// makes a corrupted or future control word refuse instead of being mistaken
/// for a scratch offset.
pub const INLINE_DESCRIPTOR_TAG: u64 = 1 << 63;

/// Reserved control sentinel for the kernel debug console (`kdbg`), the one
/// userland→console channel that survives the shell READY edge: init's parent
/// beacons are drained only by the boot driver, so after READY a task has no
/// way to say anything a human can read, and a steady-state defect can only be
/// guessed at.
///
/// This is deliberately NOT a descriptor and consumes no vocabulary: it names
/// an inline carrier whose length can never fit the register bank, so every
/// build that does not compile the handler still refuses it exactly as before.
/// The handler itself is behind the kernel's `kdbg` feature and is absent from
/// a production image, so this can never become an ambient print authority.
pub const DEBUG_EMIT_CONTROL: u64 = u64::MAX;

/// Length bits admitted beside [`INLINE_DESCRIPTOR_TAG`].
pub const INLINE_DESCRIPTOR_LENGTH_MASK: u64 = u8::MAX as u64;

/// Encode the control word for an inline descriptor.
///
/// Empty descriptors are not sessions (the header alone is eight bytes), and
/// a length wider than the frozen one-byte carrier cannot be represented.
#[inline]
pub const fn inline_descriptor_control(len: usize) -> Option<u64> {
    if len == 0 || len > INLINE_DESCRIPTOR_LENGTH_MASK as usize {
        None
    } else {
        Some(INLINE_DESCRIPTOR_TAG | len as u64)
    }
}

/// Decode a well-formed inline control word.  `None` covers both a scratch
/// control word (tag clear) and a tagged word with reserved bits or a zero
/// length; callers that need to distinguish those cases first test the tag.
#[inline]
pub const fn inline_descriptor_len(control: u64) -> Option<usize> {
    if control & INLINE_DESCRIPTOR_TAG == 0
        || control & !(INLINE_DESCRIPTOR_TAG | INLINE_DESCRIPTOR_LENGTH_MASK) != 0
    {
        return None;
    }
    let len = (control & INLINE_DESCRIPTOR_LENGTH_MASK) as usize;
    if len == 0 { None } else { Some(len) }
}

/// An index into the caller's cap arena — how a descriptor names a
/// capability (or a result destination). **Not** a `Cid`: a slot is
/// meaningful only within the owning task's arena, which is what keeps the
/// name unforgeable across tasks.
///
/// **Opaque** (the slot-token refactor): the index is private, and there
/// are exactly two lawful mints —
///
/// - [`from_wire`](Self::from_wire), the **audit point** for every place
///   a raw index enters from bytes a peer wrote (a service decoding its
///   request ring, the kernel parsing a descriptor) *or* is authored
///   into a descriptor the kernel founder builds about its own arena.
///   Greppable by design: every `from_wire` call is a place a reviewer
///   must ask "whose number is this?".
/// - `SlotAlloc` (`kore-rt::session::slots`), the allocator a task's own
///   runtime claims fresh slots from — the only way to *pick* a slot.
///
/// What this prevents is not cross-task forgery (arenas are per-task and
/// records holder-stamped — a conjured index never names another task's
/// caps) but the **in-task confused deputy**: code paths handling
/// peer-influenced data could previously conjure `SlotRef(n)` and steer
/// which of the task's OWN capabilities a session exercises, and
/// hand-picked literals collided silently (the founder's slot-35/40/45
/// class).  Egress (encoding a slot into a frozen wire form) reads
/// [`index`](Self::index).
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SlotRef(u32);

impl SlotRef {
    /// Mint a slot name from a **wire-carried index** — the ingress /
    /// descriptor-authoring audit point (see the type docs).  The value
    /// is untrusted here by construction: the kernel validates every
    /// slot against the caller's arena (bounds, vacancy, kind, rights,
    /// holder stamp) at use, and a service consuming a peer-named slot
    /// must validate the record before acting on it (the
    /// `claim_*` verbs' fail-closed discipline).
    #[inline]
    pub const fn from_wire(raw: u32) -> Self {
        Self(raw)
    }

    /// The raw index, for **egress**: encoding into a frozen wire form,
    /// arena indexing kernel-side, and diagnostics.
    #[inline]
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// The capability operation an invocation performs — the minimal schema set
/// (SYSCALL.md: derive / revoke / rendezvous is sufficient). Device, MMIO,
/// and IOMMU operations are [`Derive`](OpCode::Derive) flavors (a mapping is
/// a derived cap): [`BarReq`], [`DeviceShareReq`], [`IrqReq`].
///
/// Discriminant **3 is retired**: it carried a `Device` op that was
/// declared here but never wired kernel-side — every one of its verbs
/// landed as a `Derive` flavor instead, and the op was pruned.  It stays
/// refused as an out-of-range discriminant and must not be reused.
///
/// Derives [`TryFromBytes`], so discriminant validation is generated from
/// the variant list itself: the kernel converts the wire word with
/// [`zerocopy::try_transmute!`], and a new variant can never desync from a
/// hand-maintained decoder `match`.  An out-of-range discriminant is a
/// checked refusal, never UB.
#[repr(u16)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, TryFromBytes, KnownLayout, Immutable)]
pub enum OpCode {
    /// Carve an attenuated child cap into the result slot.
    Derive = 0,
    /// Retire a cap; its subtree dies by closure at commit.
    Revoke = 1,
    /// Re-derive a cap under a peer's subtree (capability transfer).
    ///
    /// `subject` binds a channel *endpoint*, whose kind selects the half:
    /// a `ChanRecv` subject **posts** — `result` names the slot in the
    /// caller's own arena the next transfer lands in — and a `ChanSend`
    /// subject **transfers** — `result` names the caller's slot holding
    /// the capability to re-derive into the peer's posted slot, with
    /// `requested` the (attenuated) rights the fresh cap carries.
    /// Non-blocking (SYSCALL.md D2): a send with no posted peer recv
    /// refuses the whole session with [`Status::WouldBlock`].
    Rendezvous = 2,
}

/// One invocation in a session: an op, the capability it binds, where a
/// produced cap is written, and an offset to any op-specific argument.
///
/// This is the **validated, in-memory** form — the kernel builds it from
/// [`InvocationWire`] after checking the op discriminant and rights bits.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Invocation {
    /// The operation to perform.
    pub op: OpCode,
    /// The `(op × kind)` disambiguator: the target `ResourceKind` a derive
    /// produces when the subject kind alone is ambiguous — a `MemLease`
    /// subject yields a `LeasedRegion` (CPU share) or a `LeasedIova` (device
    /// share, `kind = IovaRegion`).  `0` = unspecified (the dispatch infers
    /// from subject + arg); consulted only where a subject produces >1 kind.
    pub kind: u8,
    /// Rights requested for a derive (must attenuate the subject's).
    pub requested: Rights,
    /// The capability this invocation operates on / derives from.
    pub subject: SlotRef,
    /// Where a produced capability is written (derive / map).  For a
    /// [`Rendezvous`](OpCode::Rendezvous) it is the op-specific slot: the
    /// posted delivery target (recv half) or the transfer subject (send
    /// half) — see the op-code's doc.
    pub result: SlotRef,
    /// Byte offset into the descriptor's argument region for the op-specific
    /// payload (e.g. a [`BarReq`]); `u32::MAX` = none.
    pub arg: u32,
}

impl Invocation {
    /// Sentinel `arg` meaning "no argument payload".
    pub const NO_ARG: u32 = u32::MAX;
}

/// `Derive × MemLease → LeasedIova` whose kernel twin is retained in the
/// caller's holdings but whose arena result is deliberately bind-inert.
///
/// A userland DMA allocation owns the mapping through its affine sub-lease:
/// revoking that one public name tears down both the CPU and IOMMU leaves.
/// Publishing a second `LeasedIova` name would add no authority and would make
/// a shared scratch result slot part of the mapping lifecycle.  This
/// discriminator preserves the ordinary typed derive and device footprint,
/// while returning [`crate::cap::CapRecord::vacant`] instead of an alias.
///
/// Kept outside the [`crate::cap::ResourceKind`] range so it cannot be confused
/// with a concrete resource kind.  Like the revoke-kind constants below, this
/// is an `(op × kind)` flavor, not a new object type.
pub const DERIVE_KIND_IOVA_ANON: u8 = 0xfe;

/// `Derive × SystemObserve` observation flavor. The operation writes one
/// bind-inert [`SystemObserveReport`] and mints no child.
pub const DERIVE_KIND_SYSTEM_OBSERVE: u8 = 0xf5;

/// `Derive × Task → Task`: attenuate a held lifecycle handle into an
/// `ACCESS`-only observer.  Ordinary `kind = ResourceKind::Task` is the
/// terminal `Found` verb, so an existing task handle cannot use that byte
/// without being misdecoded as a second founding.  This shape is deliberately
/// narrow: held Task subject, no argument, exactly `ACCESS` requested.  It
/// mints a fresh descendant tag over the same generational task key; the
/// parent retains its owning handle while the child can only wait.
///
/// The byte is shared lawfully with [`DERIVE_KIND_SYSTEM_OBSERVE`]: their
/// authenticated subject kinds are disjoint, following the existing
/// `SET_TP` / retired `EXEC_SEAL` vocabulary rule.
pub const DERIVE_KIND_TASK_OBSERVER: u8 = DERIVE_KIND_SYSTEM_OBSERVE;

/// `Derive × PciDevice → IrqRoute` with a strict **MSI-X source**
/// requirement.
///
/// The ordinary [`ResourceKind::IrqRoute`](crate::cap::ResourceKind::IrqRoute)
/// flavor deliberately means "prefer MSI-X, otherwise isolated MSI". Some
/// device protocols assign distinct meanings to MSI-X table entries and cannot
/// safely consume the aggregate MSI fallback (Xe's shared root/GT/GuC/display
/// entry zero plus distinct HWE entry one are one example). This additive high
/// flavor carries the same frozen
/// [`IrqReq`] payload and produces the same affine `IrqRoute`, but the kernel
/// refuses an MSI-only function instead of silently changing source semantics.
///
/// `0xf4` is the next unused selector below [`DERIVE_KIND_SYSTEM_OBSERVE`].
pub const DERIVE_KIND_IRQ_ROUTE_MSIX: u8 = 0xf4;

/// `Derive × PciDevice/PlatformDevice` observation of the HID-over-I2C children firmware
/// placed behind that exact controller.
///
/// The subject is the already-claimed controller capability with DERIVE.
/// Requested rights are empty: the result is
/// bind-inert [`I2cHidReport`] / [`I2cHidCountReport`] data, never another
/// authority.  Requiring the controller subject keeps the parameters attached
/// to the device they describe: a row cannot ask for an arbitrary AML path or
/// accidentally pair one controller's slave address with another controller's
/// BAR.
///
/// `0xf3` is the next unused high flavor below
/// [`DERIVE_KIND_IRQ_ROUTE_MSIX`].
pub const DERIVE_KIND_I2C_HID_RECORD: u8 = 0xf3;

/// `Derive × PhysRegion` lifecycle observation for MM supply accounting.
/// The operation writes one bind-inert [`SupplyReleaseReport`] for a returned
/// tracked supply, or refuses when no completed return is pending.
///
/// `0xf2` is the next unused high flavor below
/// [`DERIVE_KIND_I2C_HID_RECORD`].
pub const DERIVE_KIND_SUPPLY_RELEASE: u8 = 0xf2;

/// Map a detached `LeasedRegion` into one of the receiver's declared windows.
/// This is an ACCESS-only exercise, not a new grant: rights, length and cache
/// policy come from the view, and the result is a bind-inert [`GeomReport`].
/// [`ViewMapReq`] selects address-space placement, never backing authority.
/// The new closed flavor makes older kernels refuse rather than silently
/// placing a large surface on the ordinary per-thread map cursor.
pub const DERIVE_KIND_VIEW_MAP: u8 = 0xf1;

/// `Derive × PciDevice → IrqRoute` for the exact controller INTx line
/// resolved from initialized firmware. No argument is accepted: the caller
/// cannot choose a GSI, polarity, trigger, or substitute a HID child's GPIO.
/// The ordinary route and strict MSI-X flavors retain their source semantics.
pub const DERIVE_KIND_IRQ_ROUTE_LINE: u8 = 0xf0;

/// `Derive × PciDevice/PlatformDevice → IrqRoute` for one initialized
/// HID-over-I2C child's firmware-correlated readiness source. The argument is an
/// [`I2cHidReq`] record query whose index is interpreted beneath the already
/// authenticated controller capability; no GPIO community, pin, GSI or
/// polarity is caller-selectable.
pub const DERIVE_KIND_I2C_HID_READY_ROUTE: u8 = 0xef;

/// `Derive × MmioRegion → LeasedIova` for an exact, sponsored peer aperture.
///
/// Unlike the ordinary [`DeviceShareReq`] (host memory into one device), this
/// flavor consumes authority from three independently authenticated objects:
/// the BAR subject remains the sole capability parent, `device` is the
/// requester whose outer IOMMU gains a mapping, and `sponsor` names a live
/// [`ResourceKind::DeviceMmu`](crate::cap::ResourceKind::DeviceMmu) facet that
/// vouches for one endpoint's private stage-1 containment. `offset` and
/// `bytes` select the exact BAR subrange; neither a whole-BAR inference nor an
/// ambient PCI-class whitelist exists.
///
/// `0xec` is the next unused selector below [`DERIVE_KIND_I2C_HID_READY_ROUTE`]
/// once `0xee` (the terminal device quiesce, #847) and `0xed` (its
/// next-flavor probe) are set aside — the two arcs must compose in either
/// landing order.
pub const DERIVE_KIND_PEER_IOVA: u8 = 0xec;

/// ACCESS-only observation of a held or staged DomainRegistration; no argument
/// or requested rights. Returns a bind-inert FabricDomainReport.
pub const DERIVE_KIND_FABRIC_OBSERVE: u8 = 0xeb;

/// Map a whole held or staged MemLease at a kernel-selected VA and retain the
/// installed LeasedRegion capability in the result. No argument; ACCESS and
/// REVOKE are required, MODIFY selects write access. Placement matches map_owned.
pub const DERIVE_KIND_OWNED_MAP: u8 = 0xea;

/// Observe a held or staged installed CPU view's virtual geometry. ACCESS-only,
/// no argument or requested rights. Detached grants are refused; no map occurs.
pub const DERIVE_KIND_CPU_VIEW_OBSERVE: u8 = 0xe9;

/// Post-commit, argument-free close of a held route. One unflagged invocation;
/// returns FabricRouteCloseReport and retains the route on Busy/Quarantined.
pub const DERIVE_KIND_FABRIC_CLOSE: u8 = 0xe8;

/// Derive a bounded child from a held or staged mapping route. Endpoints and
/// operation are inherited; FabricRouteScopeReq only attenuates range/access.
pub const DERIVE_KIND_FABRIC_SCOPE: u8 = 0xe7;

/// ACCESS-only observation of a held or staged mapping route. No arguments or
/// requested rights; returns inert FabricRouteInfo before any mapping occurs.
pub const DERIVE_KIND_FABRIC_ROUTE_OBSERVE: u8 = 0xe6;

/// Fixed lifecycle rights of a driver-held `DeviceMmu` attestation. `ACCESS`
/// makes the task-kill pre-quiesce borrowable, `DERIVE` mints one transient
/// sponsor witness per peer install, and `REVOKE` owns closure of the stage-1
/// generation. No raw mutation projection is exposed.
pub const DEVICE_MMU_RIGHTS: crate::cap::Rights = crate::cap::Rights::ACCESS
    .union(crate::cap::Rights::DERIVE)
    .union(crate::cap::Rights::REVOKE);

/// `Derive × PciDevice` observation flavor. The operation returns either the
/// number of validated modern VirtIO PCI vendor-capability rows or one
/// bind-inert [`VirtioPciCapReport`] and mints no child.
///
/// `ACCESS` on the exact held device authorizes the observation. Ring 3 sees
/// only a BAR number and BAR-relative `(offset, length)`; PCI config-space
/// offsets and physical BAR addresses never cross this boundary. The kernel
/// also proves every reported interval lies wholly inside that BAR's
/// user-mappable prefix before encoding it.
pub const DERIVE_KIND_VIRTIO_PCI_LAYOUT: u8 = 0xdc;

/// `Derive × AddrSpace` (the caller's **own** space — any thread view of
/// it; the kernel verifies by page-table root, the same discriminator
/// `Found`'s `SHARE_SPACE` check uses) that **sets the calling task's
/// thread pointer** (`IA32_FS_BASE` / `TPIDR_EL0` / `tp`) — the unified
/// task model's self-verb (#407), this ABI's `arch_prctl(ARCH_SET_FS)`.
/// An *effect*, not a derivation (the [`PagerReq`]/[`PowerReq`] idiom):
/// it mints no child, the result slot stays vacant, and the new base
/// takes effect on the very return from this session (the kernel writes
/// both the saved context and the live register).  `MODIFY` on the space
/// authorizes it; the VA must be canonical user-half.  A spawned thread's
/// entry issues this as its first session — before any transport
/// machinery that reads thread-anchored state — from a stack-built
/// descriptor (`commit_current` takes the descriptor VA per call, so no
/// scratch window is required).
pub const DERIVE_KIND_SET_TP: u8 = 0xfd;

/// `Derive × AddrSpace` (the caller's **own** space, verified by
/// page-table root exactly as [`DERIVE_KIND_SET_TP`] is) that **narrows
/// the protection** of an already-mapped VA range to
/// [`ProtectReq::access`] — the RELRO seal and the passive-page seal
/// (`res/IMAGE.md` §4 phase 2, `kore_ld`'s `LinkWorld::seal_ro`).  An
/// *effect*, not a derivation: it mints no child, the result slot stays
/// vacant, and the leaves' physical address and granule are untouched —
/// only their flags change.  `MODIFY` on the space authorizes it, the
/// same write projection `SET_TP` takes.
///
/// **Monotone by construction: a request may only CLEAR access bits,
/// never set one.**  Naming a bit the range does not already carry is
/// [`Status::Denied`], never a silent clamp (Σ5's posture).  That is the
/// whole security argument for the verb: every reachable outcome is a
/// subset of what the range already permitted, so `MODIFY` on one's own
/// space buys no escalation.  Without it a linker could re-open its own
/// text `W` and defeat both the per-lease W^X gate and [`LEASE_EXEC`]'s
/// "immutable text with no unprotect path".
///
/// Why a verb, rather than unmap-then-share-read-only over the existing
/// ABI: the two-step leaves the range **unmapped** in between, so it is
/// not a downgrade but a teardown and a rebuild.  It can fail halfway —
/// leaving a hole where a sealed page belongs — it discards the leaves'
/// identity, and it re-enters the share path, whose W^X gate is written
/// in terms of a *new* mapping rather than a narrowing of an old one.  A
/// seal is one transition of leaves that already exist, which is what
/// the paging walker's `protect` already expresses one level down; this
/// constant is the authority gate over it.
///
/// `va`/`len` are page-granular and canonical user-half.  A range not
/// exactly tiled by installed leaves refuses rather than sealing a
/// prefix.
pub const DERIVE_KIND_PROTECT: u8 = 0xfa;

/// `Derive × Task` (the caller's **own** task handle) that **exits the
/// calling task** with the [`ExitReq`] status — the unified task model's
/// other self-verb, beside [`DERIVE_KIND_SET_TP`].  An *effect*, not a
/// derivation (the [`PagerReq`] idiom): it mints no child, the result
/// slot stays vacant, and the session never returns — the caller's very
/// next instruction does not run.
///
/// `MODIFY` on the handle authorizes it (the write projection over the
/// task, as `SET_TP` takes `MODIFY` over the space); `ACCESS` alone is
/// the *observer's* grant and cannot kill its subject.  The kernel then
/// checks the bound handle names the **ambient caller** — a task exits
/// itself and nothing else, so presenting a peer's handle is `Denied`
/// rather than a remote kill (killing a peer is the handle's `REVOKE`,
/// which already exists and reaps rather than exits).
///
/// An exited task stops being scheduled but is **not** reaped: its exit
/// status has to outlive it for a joiner ([`DERIVE_KIND_TASK_JOIN`]) to
/// read.  The reap is still the handle's `Revoke`.
pub const DERIVE_KIND_TASK_EXIT: u8 = 0xfc;

/// Close the calling process owner's worker admission and physically drain
/// every peer in that exact address space. SELF + MODIFY, sole invocation,
/// no arguments. Workers and peer handles cannot request this operation.
/// The result is a bind-inert `ProcessDrainReport`; WouldBlock reports no proof
/// and may be retried after a concurrent physical reaper completes.
pub const DERIVE_KIND_PROCESS_DRAIN: u8 = 0xe2;

/// Read one immutable crash-evidence chunk through an owning task handle.
/// ACCESS + REVOKE, sole invocation. The argument is a little-endian u32
/// chunk index followed by four zero bytes. No record returns Exited;
/// live subjects return WouldBlock. Lifecycle observers cannot read stacks.
pub const DERIVE_KIND_CRASH_READ: u8 = 0xe1;
/// Report a fatal software panic on SELF. The payload is UTF-8 with a u32
/// length, four reserved zero bytes, and exactly that many message bytes.
/// MODIFY, standalone. Never grants the caller authority over another process.
/// Panic arguments: UTF-8 length u32, truncated u8 (0/1), zero[3], message bytes.
pub const DERIVE_KIND_PANIC_REPORT: u8 = 0xe0;
/// Bind a parent-owned, unmapped RAM lease before TASK_START. Arguments are
/// lease slot u32, reserved-zero u32, page-aligned offset u64 and length u64.
/// Task MODIFY|REVOKE and lease MODIFY are required. Standalone, no result.
pub const DERIVE_KIND_CRASH_BIND: u8 = 0xdf;
/// Assign a bounded UTF-8 task label once before START (Task MODIFY).
/// Arguments: byte length u8 followed by that many UTF-8 bytes, at most 63.
pub const DERIVE_KIND_TASK_NAME: u8 = 0xde;
/// Task ACCESS; standalone, no arguments. Returns an inert TaskIdentity report.
pub const DERIVE_KIND_TASK_IDENTITY: u8 = 0xdd;
pub const TASK_IDENTITY_KIND: u16 = 0xFF16;
pub const TASK_IDENTITY_MAGIC: u64 = 0x6b6f_7265_5449_4431;
impl crate::task::TaskIdentity {
    pub fn encode(self) -> crate::cap::CapRecord {
        let mut rec = inert_frame(TASK_IDENTITY_KIND, TASK_IDENTITY_MAGIC);
        rec.mac = self.id.bytes();
        rec.scope_rid = self.quality as u64;
        rec
    }
    pub fn decode(rec: &crate::cap::CapRecord) -> Option<Self> {
        if !is_inert_report(rec, TASK_IDENTITY_KIND, TASK_IDENTITY_MAGIC)
            || rec.holder != crate::cap::Cid::NULL
            || !rec.rights.is_empty()
            || rec.epoch != 0
            || rec.chain_len != 0
            || rec.chain != [0; crate::cap::MAX_DEPTH]
            || rec.scope_rid > u8::MAX as u64
        {
            return None;
        }
        Some(Self {
            id: crate::task::TaskId::from_bytes(rec.mac),
            quality: crate::task::TaskIdQuality::decode(rec.scope_rid as u8)?,
        })
    }
}
pub const CRASH_CHUNK_BYTES: usize = 16;
pub const CRASH_CHUNK_KIND: u16 = 0xFF15;
pub const CRASH_CHUNK_MAGIC: u64 = 0x6b6f_7265_4352_5348;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CrashChunk {
    pub index: u32,
    pub bytes: [u8; CRASH_CHUNK_BYTES],
}
impl CrashChunk {
    pub fn encode(self) -> crate::cap::CapRecord {
        let mut rec = inert_frame(CRASH_CHUNK_KIND, CRASH_CHUNK_MAGIC);
        rec.scope_rid = u64::from(self.index);
        rec.mac = self.bytes;
        rec
    }
    pub fn decode(rec: &crate::cap::CapRecord) -> Option<Self> {
        if !is_inert_report(rec, CRASH_CHUNK_KIND, CRASH_CHUNK_MAGIC)
            || rec.scope_rid >= (crate::crash::MIN_BUFFER_BYTES / CRASH_CHUNK_BYTES) as u64
            || rec.holder != crate::cap::Cid::NULL
            || !rec.rights.is_empty()
            || rec.epoch != 0
            || rec.chain_len != 0
            || rec.chain != [0; crate::cap::MAX_DEPTH]
        {
            return None;
        }
        Some(Self {
            index: rec.scope_rid as u32,
            bytes: rec.mac,
        })
    }
}

/// Successful terminal worker barrier. Forced shutdown skips user finalizers:
/// arbitrary interrupted user code may have left shared state inconsistent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProcessDrainReport {
    pub forced: bool,
}
pub const PROCESS_DRAIN_REPORT_KIND: u16 = 0xFF14;
pub const PROCESS_DRAIN_REPORT_MAGIC: u64 = 0x6b6f_7265_4452_414e;
impl ProcessDrainReport {
    pub fn encode(self) -> crate::cap::CapRecord {
        let mut rec = inert_frame(PROCESS_DRAIN_REPORT_KIND, PROCESS_DRAIN_REPORT_MAGIC);
        rec.scope_rid = u64::from(self.forced);
        rec
    }
    pub fn decode(rec: &crate::cap::CapRecord) -> Option<Self> {
        if !is_inert_report(rec, PROCESS_DRAIN_REPORT_KIND, PROCESS_DRAIN_REPORT_MAGIC)
            || rec.scope_rid > 1
        {
            return None;
        }
        let result = Self {
            forced: rec.scope_rid != 0,
        };
        let canonical = result.encode();
        (rec.holder == canonical.holder
            && rec.rights == canonical.rights
            && rec.epoch == canonical.epoch
            && rec.chain_len == canonical.chain_len
            && rec.chain == canonical.chain
            && rec.mac == canonical.mac)
            .then_some(result)
    }
}

/// `Derive × Task` that **joins** the subject task: the result slot
/// receives a bind-inert [`ExitReport`] once it has exited.  `ACCESS` on
/// the handle authorizes it — observing a task's end is the observer's
/// grant, and holding the handle *is* the authority to wait on it.
///
/// Two shapes, chosen by the session's [`SessionFlags::WAIT_TASK`] bit:
/// flagged, the commit **parks** until the subject exits or is reaped
/// (waking [`Status::Exited`] / [`Status::Revoked`]); unflagged, it is a
/// non-blocking probe that answers `Exited` for an already-exited subject
/// and refuses the whole session [`Status::WouldBlock`] for a live one —
/// the [`OpCode::Rendezvous`] send's posture, so "did not block" never
/// reads as "did not exit".
///
/// Like every derive here the subject is resolved through the ordinary
/// bind (stamp ‖ D1 holder ‖ liveness) and the joined task's identity is
/// taken from the **authenticated** capability the caller holds — never
/// from a wire task id, which no verb in this ABI accepts.
pub const DERIVE_KIND_TASK_JOIN: u8 = 0xfb;

/// `Derive × RootComplex` / `Derive × PlatformBus` that **observes the
/// subject's inventory**: record mode writes a bind-inert [`DeviceReport`]
/// describing [`InventoryReq::index`], count mode writes a
/// [`DeviceCountReport`], and neither mode mints a capability.
///
/// The [`DERIVE_KIND_TASK_JOIN`] shape — an *observation*, not a
/// derivation.  It exists so a device broker can match a selector against
/// its root's inventory without minting and revoking a capability per
/// candidate: reading what firmware described is not an authority
/// transfer, and making it one would put device-shaped cap churn on the
/// path of every claim.
///
/// `ACCESS` on the subject authorizes it — reading the inventory is the
/// observer's grant, exactly as `ACCESS` on a task handle authorizes a
/// join.  Handing a device over needs `DERIVE`, and that is the separate
/// flavor below.
pub const DERIVE_KIND_DEVICE_RECORD: u8 = 0xf9;

/// `Derive × Task` that makes a completely founded, still-dormant task
/// **runnable**.  This is the explicit close of `Found != runnable`: exec may
/// assemble and return a child handle without racing the child's first
/// instruction, and only the holder of that handle with `MODIFY` may publish
/// the finished child to the scheduler.
///
/// An *effect*, not a derivation: `requested` is zero, there is no argument,
/// and the result slot is bind-inert.  The authenticated held task handle is
/// the whole request; no wire task id exists.  START is idempotent once the
/// child is queued, running, interrupted, or parked.  A first admission which
/// cannot yet be routed answers [`Status::WouldBlock`] and may be retried with
/// the same affine handle; a dead/exited subject answers [`Status::Revoked`].
///
/// Kept in the global derive-flavor namespace outside
/// [`crate::cap::ResourceKind`].  `0xf8` is the next unused selector below
/// [`DERIVE_KIND_DEVICE_RECORD`]; unlike `SET_TP`/`EXEC_SEAL`, every flavor
/// sharing a `Task` subject must have a distinct value.
pub const DERIVE_KIND_TASK_START: u8 = 0xf8;

/// `Derive × AddrSpace` (the task's **own** space view, the
/// `SET_TP`/`PROTECT` acting-on-self family) that **grows the caller's
/// capability arena by one chunk** (slice E): the kernel allocates a
/// 256-record extension chunk, publishes the raised slot count, and
/// alias-maps the chunk's records R/O into the caller's
/// `geom::SLOTS_EXT` window.  One invocation = one chunk; repeat to grow
/// further, up to the window reservation (the structural ceiling's
/// userland witness).
///
/// An *effect*, not a derivation: `requested` is zero, the argument is
/// an 8-byte reserved-zero block (the `EXEC_SEAL` disambiguation
/// discipline), it mints **no capability** — arena capacity is kernel
/// storage, invisible to the KRAPH capability set — and it runs
/// post-commit (I3: an aborted session grows nothing).  The result slot
/// receives a bind-inert [`GeomReport`] naming the new chunk's user
/// alias window `(base, len)`, from which the runtime derives its new
/// live slot bound.  The subject must bind as an `AddrSpace` carrying
/// `MODIFY` whose lineage roots in the **caller's own** space — a task
/// grows its own namespace, never a peer's.
///
/// `0xf7` is the next unused selector below [`DERIVE_KIND_TASK_START`];
/// distinct from every other `AddrSpace`-subject flavor
/// (`SET_TP` = `0xfd`, `PROTECT` = `0xfa`).
pub const DERIVE_KIND_ARENA_EXTEND: u8 = 0xf7;

/// `Derive × PciDevice` that opens PCI Command.BusMasterEnable only after the
/// request-named DMA sub-lease has installed a live IOMMU window.
///
/// This is a post-commit **effect**, not a resource derive: it mints no child,
/// requests no rights, and must stand alone because config-space readback can
/// refuse after the empty capability fold committed. `MODIFY` on the device
/// authorizes the mutation; `MODIFY` on [`DeviceActivateReq::dma_lease`]
/// proves the caller still owns the exact affine window lifetime it prepared.
/// Window retirement closes BME before unmapping, so no separate activation
/// capability or second KRAPH parent is introduced.
///
/// `0xf6` is the next unused selector below [`DERIVE_KIND_ARENA_EXTEND`].
pub const DERIVE_KIND_DEVICE_ACTIVATE: u8 = 0xf6;

/// `Derive × PciDevice` / `Derive × PlatformDevice` **terminal quiesce**
/// (#809 §L): the kernel proves the function quiet at the fabric — masks
/// MSI/MSI-X, clears Bus Master Enable and reads it back, rebinds the
/// requester's IOMMU context to Abort — and answers with a
/// [`QuiesceReport`] in the caller's result slot.  No D-state changes
/// hands (firmware reprograms the function on the next boot anyway), so
/// this is the DMA half of a suspend and nothing else.
///
/// Like [`DERIVE_KIND_DEVICE_ACTIVATE`] this is a post-commit **effect**:
/// it mints no child, requests no rights, and must stand alone because the
/// config-space readback can refuse after the empty capability fold
/// committed.  `MODIFY` on the device authorizes it — the same right that
/// gates `DeviceControl`, so a holder attenuated to `ACCESS` provably
/// cannot drive the terminal transition.  The report is the *only* way a
/// userland `Quiesced` witness comes to exist: a class driver asserts what
/// it stopped, the kernel proves what the fabric saw.
///
/// `0xee` is the next unused selector below
/// [`DERIVE_KIND_I2C_HID_READY_ROUTE`].
pub const DERIVE_KIND_DEVICE_QUIESCE: u8 = 0xee;

/// `Derive × GuestSpace` terminal founding flavor: create one schedulable
/// vCPU task from a service-owned guest address space.
///
/// Unlike ordinary [`FoundReq`], this does not consume the guest-space owner:
/// the kernel derives one non-owning view for the vCPU, while the service
/// retains the owner that governs guest-wide teardown.  The request names
/// three **held** affine facets which move into the new vCPU's private
/// holdings: the exit/control-window lease, its standing resume receive end,
/// and the service-doorbell send end.  Moving the window owner (rather than
/// retaining a raw borrowed pointer) makes its backing lifetime exactly the
/// vCPU task's lifetime.
///
/// This is terminal like `Found`: it must be the session's last effectual
/// invocation and its result receives the ordinary `Task` lifecycle handle.
/// `0xf4` is the next unused global flavor below
/// [`DERIVE_KIND_SYSTEM_OBSERVE`].
pub const DERIVE_KIND_GUEST_FOUND: u8 = 0xf4;

/// Argument payload for [`DERIVE_KIND_GUEST_FOUND`].
///
/// The first three words are the architecture-neutral initial vCPU image.
/// The three slot names are authenticated second capabilities in the caller's
/// arena and must already be held: the service formats/maps the window and
/// establishes its doorbell topology before founding.  LE byteorder
/// (`align(1)`).
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct GuestFoundReq {
    /// Guest instruction pointer at first entry.
    pub entry: U64,
    /// Guest stack pointer at first entry.
    pub stack: U64,
    /// Guest-physical root of the guest's own stage-1 page tables.
    pub page_table_root: U64,
    /// Held `MemLease` owning the formatted exit/control window page.
    pub window: U32,
    /// Held `ChanRecv` used as the vCPU's standing resume post.
    pub resume_recv: U32,
    /// Held `ChanSend` used to kick the hypervisor service's doorbell.
    pub exit_send: U32,
    /// Reserved; must be zero.
    pub reserved: U32,
}

impl GuestFoundReq {
    /// Wire size (`3·8 + 4·4`).
    pub const WIRE: usize = 40;
}

/// Fixed rights of an inert executable-possession proof. It can be inspected
/// and retired but grants no authority over its source endpoint or supply.
pub const INVOCATION_WITNESS_RIGHTS: crate::cap::Rights =
    crate::cap::Rights::ACCESS.union(crate::cap::Rights::REVOKE);

/// `Derive × ChanSend -> InvocationWitness` argument. `supply` names the
/// exact held lease moved through that endpoint in the same descriptor.
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct InvocationWitnessReq {
    pub supply: U32,
    pub reserved: [u8; 4],
}

impl InvocationWitnessReq {
    pub const WIRE: usize = 8;
}

/// Argument payload for [`DERIVE_KIND_DEVICE_ACTIVATE`]. The device is the
/// invocation subject; this second, footprinted capability names the mapped
/// DMA sub-lease whose internal owner stamp identifies its live IOVA window.
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct DeviceActivateReq {
    /// Arena slot of the `MemLease` sub-lease owning the DMA window.
    pub dma_lease: U32,
    /// Reserved; must be zero.
    pub reserved: [u8; 4],
}

/// Select one record or the count from [`DERIVE_KIND_VIRTIO_PCI_LAYOUT`].
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct VirtioPciLayoutReq {
    /// Row in capability-chain order. Must be zero for a count query.
    pub index: u8,
    /// [`VIRTIO_PCI_LAYOUT_QUERY_RECORD`] or
    /// [`VIRTIO_PCI_LAYOUT_QUERY_COUNT`].
    pub query: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 6],
}

impl VirtioPciLayoutReq {
    pub const WIRE: usize = 8;
}

pub const VIRTIO_PCI_LAYOUT_QUERY_RECORD: u8 = 0;
pub const VIRTIO_PCI_LAYOUT_QUERY_COUNT: u8 = 1;

impl DeviceActivateReq {
    /// Wire size (repr(C), 4+4).
    pub const WIRE: usize = 8;
}

/// Argument payload for the [`DERIVE_KIND_TASK_EXIT`] flavor: the exit
/// status the caller ends with, which its joiners read back out of an
/// [`ExitReport`].  `i32` (LE, unaligned-safe) — signed because a status
/// is a *value* a program chose, not a resource id, and the negative half
/// is the conventional room for "ended abnormally".
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct ExitReq {
    /// The exiting task's status.
    pub status: I32,
    /// Reserved; must be zero.
    pub reserved: [u8; 4],
}

impl ExitReq {
    /// Wire size (repr(C), 4+4).
    pub const WIRE: usize = 8;
}

/// The **exit report** a join answers with — the notice family's fifth
/// member, [`GeomReport`]'s shape carrying a task fact instead of an
/// address-space one: the kernel vouches that the joined task ended, and
/// with what status.
///
/// A report rather than a status-word field on purpose.  The status
/// word's high half is already the clock stamp on every path that crosses
/// to ring 3, so an `i32` cannot ride there without contending for it;
/// and a task's end is exactly the kind of fact that grows fields (a
/// signal, a fault address, a CPU-time total) — a record leaves room, a
/// packed word does not.
///
/// The encoding rides the frozen 128-byte record bind-inert (the family
/// contract): `cid = NULL`, `kind = `[`EXIT_REPORT_KIND`], the magic in
/// `parent`, and the sign-extended status in `scope_rid`.  Kernel encodes
/// (the join's result-slot write, at commit or at the park's resume),
/// userland decodes — one impl, no drift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExitReport {
    /// The status the joined task exited with.
    pub status: i32,
}

/// The `kind` discriminant of an encoded [`ExitReport`] — outside the
/// [`ResourceKind`](crate::cap::ResourceKind) space, beside
/// [`GEOM_REPORT_KIND`] (a report is data, not a resource).
pub const EXIT_REPORT_KIND: u16 = 0xFF05;

/// The `parent` magic of an encoded [`ExitReport`].
pub const EXIT_REPORT_MAGIC: u64 = 0x6B6F_7265_4558_4954; // "koreEXIT"

impl ExitReport {
    /// Encode as a bind-inert [`CapRecord`](crate::cap::CapRecord)
    /// (kernel side — the join arm's result-slot write).
    pub fn encode(self) -> crate::cap::CapRecord {
        let mut rec = inert_frame(EXIT_REPORT_KIND, EXIT_REPORT_MAGIC);
        rec.scope_rid = self.status as i64 as u64;
        rec
    }

    /// Decode only the exact bind-inert exit encoding, including canonical
    /// sign extension of its i32 payload. Malformed high bits must not become
    /// another status (in particular success) through a narrowing cast.
    pub fn decode(rec: &crate::cap::CapRecord) -> Option<Self> {
        if !is_inert_report(rec, EXIT_REPORT_KIND, EXIT_REPORT_MAGIC) {
            return None;
        }
        let status = i32::try_from(rec.scope_rid as i64).ok()?;
        let report = Self { status };
        let canonical = report.encode();
        (rec.holder == canonical.holder
            && rec.rights == canonical.rights
            && rec.epoch == canonical.epoch
            && rec.chain_len == canonical.chain_len
            && rec.chain == canonical.chain
            && rec.mac == canonical.mac)
            .then_some(report)
    }
}

/// The [`FoundReq::authority`] sentinel for a **thread founding**
/// (#407, the unified task model): lawful ONLY with
/// [`SpawnFlags::SHARE_SPACE`](crate::task::SpawnFlags) over the
/// caller's OWN page-table root (the kernel's `is_thread_view` proof
/// refuses every other shape).  A same-space thread is the task's own
/// concurrency, not a new protection domain — it can reach no authority
/// its parent does not already hold, so it needs no `ExecAuthority`.
/// Any other founding with this sentinel refuses at prepare.
pub const NO_AUTHORITY: u16 = u16::MAX;

/// Argument payload for a **port-grant** derive on a `PlatformDevice`
/// (#416): one sub-range of what the device declares.  An arg-free
/// derive asks for every declared range instead.
///
/// The kernel bounds this against the FIRMWARE description — a range
/// that is not inside a single declared range refuses — so this
/// structure can narrow a grant but never widen one.
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct PortReq {
    /// First port of the requested range.
    pub base: U16,
    /// Number of contiguous ports.
    pub len: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 5],
}

impl PortReq {
    /// Wire size.
    pub const WIRE: usize = 8;
}

/// Argument payload for the [`DERIVE_KIND_SET_TP`] flavor: the thread
/// pointer to install.  LE, unaligned-safe, 8 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct TpReq {
    /// The new thread-pointer VA (`%fs` base) — canonical user space.
    pub tp: U64,
}

impl TpReq {
    /// Wire size.
    pub const WIRE: usize = 8;
}

/// Argument payload for the [`DERIVE_KIND_PROTECT`] flavor: the range to
/// narrow and the access it is narrowed **to**.  LE, unaligned-safe,
/// 24 bytes.
///
/// `access` is an explicit [`SHARE_ACCESS_R`]`/W/X` mask, the same
/// vocabulary a share states its mapping in — so a seal and the share it
/// narrows speak one language, and `0` (no access at all) is expressible
/// rather than being read as "legacy/unspecified" the way
/// [`ShareToReq::access`] treats it.  The kernel refuses any bit not
/// already present on every leaf in the range: see the constant's docs
/// for why that monotonicity is the verb's security argument.
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct ProtectReq {
    /// First byte of the range — page-aligned, canonical user-half.
    pub va: U64,
    /// Range length in bytes — a non-zero multiple of the page size.
    pub len: U64,
    /// The access the range carries **after** the seal
    /// ([`SHARE_ACCESS_R`]`/W/X`).  Must be a subset of what it carries
    /// now.
    pub access: u8,
    /// Must be zero — the growth room every frozen block in this ABI
    /// keeps, so a later field cannot be read out of a stale sender's
    /// padding.
    pub reserved: [u8; 7],
}

impl ProtectReq {
    /// Wire size.
    pub const WIRE: usize = 24;
}

/// Retired selector for the former derive-an-executable-alias operation.
///
/// Frozen ABI values are never reused.  New code uses
/// [`REVOKE_KIND_EXEC`], whose consuming lease transition closes the writable
/// lineage instead of leaving a parent or pre-existing W alias alive.
///
/// Kept outside the [`crate::cap::ResourceKind`] range, like
/// [`DERIVE_KIND_IOVA_ANON`]: an `(op × kind)` flavor, not a new object
/// type.
pub const DERIVE_KIND_EXEC_SEAL: u8 = 0xfd;

/// The on-wire layout of one invocation — 16 bytes, LE, `align(1)`:
/// `op u16 ‖ rights u8 ‖ kind u8 ‖ subject u32 ‖ result u32 ‖ arg u32`.
///
/// Every field admits any bit pattern ([`FromBytes`]), so the kernel
/// reinterprets the whole descriptor run in place
/// (`<[InvocationWire]>::ref_from_prefix_with_elems`) and then validates
/// each element into an [`Invocation`]: `op` through [`OpCode`]'s
/// generated discriminant check, `rights` through the reserved-bit-closed
/// [`Rights::from_bits`].  `kind` admits any byte — it is a `(op × kind)`
/// selector the kernel compares against known discriminants, never
/// transmuted.  Semantic validation stays a kernel refusal with the failing
/// index; *layout* validation lives here in the types.
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, KnownLayout, Immutable, Unaligned)]
pub struct InvocationWire {
    /// [`OpCode`] discriminant, unvalidated.
    pub op: U16,
    /// [`Rights`] bits, unvalidated.
    pub requested: u8,
    /// [`Invocation::kind`] — the `(op × kind)` disambiguator (`0` = none).
    pub kind: u8,
    /// [`Invocation::subject`] slot index.
    pub subject: U32,
    /// [`Invocation::result`] slot index.
    pub result: U32,
    /// [`Invocation::arg`] byte offset.
    pub arg: U32,
}

/// Per-invocation outcome the kernel writes back into the completion region.
#[repr(u16)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    /// The invocation committed.
    Ok = 0,
    /// Rights insufficient for the requested op.
    Denied = 1,
    /// A bound cap was forged (bad stamp) or presented by the wrong holder
    /// (D1).
    Forged = 2,
    /// A bound cap, or one of its ancestors, was revoked.
    Revoked = 3,
    /// The session lost an optimistic-commit race; retry it (SYSCALL.md
    /// OCC).
    Conflict = 4,
    /// Malformed descriptor: out-of-range slot, forward reference, or
    /// over-length session.
    Malformed = 5,
    /// A non-blocking rendezvous found no posted peer recv (or lost the
    /// post to a rival send) — the session did not commit; re-stage and
    /// retry once the peer has posted (SYSCALL.md D2: no kernel waiter
    /// state, blocking arrives with the completion ring).
    WouldBlock = 6,
    /// A combined `WAIT_DELIVERY | WAIT_SLEEP` park resolved by its
    /// **deadline** (USERLAND.md §3.4.4): the session committed, the
    /// caller parked, and the timer wheel — not a delivery — woke it.
    /// The standing post is left standing (the doorbell is not lost);
    /// this outcome word is what distinguishes the deadline win from a
    /// delivery ([`Ok`](Self::Ok)) or a reap ([`Revoked`](Self::Revoked)),
    /// which recv-tag liveness alone cannot (a standing post's tag is
    /// live at a deadline wake).  Additive on the frozen vocabulary.
    Deadline = 7,
    /// A park carrying `WAIT_IRQ` resolved by its **interrupt arm**: the
    /// session committed, the caller parked, and the held route fired —
    /// not a delivery, reap, or deadline.  The standing post is left
    /// standing (exactly [`Deadline`](Self::Deadline)'s rule), and the
    /// route stays held — a fired route is re-armed by the caller's next
    /// park.  Additive on the frozen vocabulary.
    Irq = 8,
    /// A park carrying `WAIT_TASK` resolved by its **join arm**: the
    /// joined task **exited**, and the join's result slot holds its
    /// [`ExitReport`].  Its own outcome word for the same reason
    /// [`Deadline`](Self::Deadline) and [`Irq`](Self::Irq) have theirs —
    /// a composite park's winner cannot be read off any liveness the
    /// other arms share.  `Ok` on such a park still means *delivery*, so
    /// a joiner that kept its doorbell live never has to guess which
    /// edge it woke on, and a reaped joinee stays
    /// [`Revoked`](Self::Revoked) (the wait died, exactly as for a reaped
    /// post or a revoked route).  Additive on the frozen vocabulary.
    Exited = 9,
    /// A bounded two-post `WAIT_DELIVERY` park resolved through its
    /// **alternate** standing post.  The ordinary [`Ok`](Self::Ok) outcome
    /// names the primary post; this distinct committed outcome prevents a
    /// receiver from guessing the winner by reading landing-slot bytes before
    /// the kernel has published capability custody.  If both posts complete
    /// before resolution, the alternate wins this classification: primary
    /// doorbell work is level-triggered, while an affine capability delivery
    /// must not lose its completion edge.
    AltDelivery = 10,
    /// The session **committed**, but its terminal founding could not be
    /// assembled afterward: the staged child was discarded and the
    /// founder's handle/result slots stay vacant, while every other
    /// committed effect (derives, moves, revocations) stands.  A distinct
    /// outcome because the [`WouldBlock`](Self::WouldBlock) contract —
    /// "did not commit; re-stage and retry" — must never be attached to a
    /// committed descriptor: replaying one re-runs its effects.  Additive
    /// on the frozen vocabulary; older decoders read the raw value as
    /// [`Malformed`](Self::Malformed), which is equally terminal.
    Unfounded = 11,
}

/// Hard cap on a session's invocation count — heapless, fixed-capacity
/// (SYSCALL.md I9). A spawn/teardown session fits well within this.
///
/// The image-bound executable migration adds an independent image face: one
/// window operation and ten endpoint derivations.  107 is the former bounded
/// ceiling (96) plus that exact face cost.  Together with the terminal
/// founding run's [`MAX_SESSION_FOUNDS`] `FoundReq` allowances and the park
/// allowance, the full descriptor is 8160 bytes (both sides'
/// `DESCRIPTOR_MAX`) and therefore fits the explicitly provisioned
/// two-page descriptor window.
pub const MAX_INVOCATIONS: usize = 107;

/// `Found`s one session may carry — the G5 founding run's width, priced
/// as the two-page descriptor window's exact capacity: each `Found` past
/// the 24-byte repeatable payload costs a `FoundReq::WIRE - 24` argument
/// allowance, and a thirteenth would burst the window (both sides pin
/// `DESCRIPTOR_MAX <= 0x2000` at build time).  A gang wider than this
/// chunks into `ceil(w / 12)` sessions; the kernel needs no runtime bound
/// of its own — the argument region is the enforcement.
pub const MAX_SESSION_FOUNDS: usize = 12;

/// The frozen descriptor-ABI version (#81; the crate-header wire-freeze
/// discipline). The kernel
/// refuses a descriptor whose [`SessionHeader::abi`] is not this. **v2**
/// reserved the flags word so the #71 blocking tier landed additively; **v3**
/// grows [`FoundReq`] with the founding **delivery map** (U1 — cross-space
/// cap delivery at founding: session children placed into the founded
/// child's arena at named slots). **v4** makes the
/// [`DeviceShareReq::access`] selector explicit: v3 decoded *every* nonzero
/// byte as read-write, so accepting a v4 posted-write-only request under v3
/// would silently add read authority and bypass the completion-route proof.
/// The exact header mismatch makes mixed v3/v4 binaries fail closed.
pub const DESCRIPTOR_ABI: u16 = 4;

bitflags! {
    /// Session-level flags — the [`SessionHeader::flags`] word (descriptor ABI
    /// v2). Reserved in the frozen ABI now so the blocking tier lands additively.
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
    pub struct SessionFlags: u16 {
        /// Park the caller after commit until a posted recv delivers (#71 D2
        /// tier 1, honored by the engine).  Two forms: the session posts
        /// **exactly one** recv and parks on it, or it posts **zero** and an
        /// argument-region [`ParkReq`] names a **standing** post from a prior
        /// commit (USERLAND.md §3.4.3 — the doorbell mux's idle park).
        /// Multiple posts refuse [`Status::Malformed`].
        const WAIT_DELIVERY = 1 << 0;
        /// Park the caller after commit on its held [`IrqRoute`] until the
        /// interrupt fires — the repeatable driver loop (#86b: derive the route
        /// once, then park→handle→park). Woken by a `deliver` (`Ok`, or
        /// [`Status::Irq`] in combination) or by the route's revoke
        /// (`Revoked`, the wake-on-revoke integration).  Combined with
        /// [`WAIT_DELIVERY`](Self::WAIT_DELIVERY) (± [`WAIT_SLEEP`](Self::WAIT_SLEEP))
        /// it is the **IRQ-or-doorbell park** (the input fan-in's steady
        /// state): first of {delivery → [`Status::Ok`], reap →
        /// [`Status::Revoked`], deadline → [`Status::Deadline`], IRQ →
        /// [`Status::Irq`]} wins, and an IRQ or deadline win leaves the
        /// standing post standing. Without `WAIT_IRQ_SELECTED`, carries no
        /// payload and uses the legacy first live route in task holdings.
        /// Multi-route callers select their held route with `IrqParkReq`.
        const WAIT_IRQ = 1 << 1;
        /// Park the caller after commit on the per-CPU timer wheel until the
        /// [`SleepReq`] deadline (USERLAND.md §3.4.2 — v1 sleep-only, the
        /// `WAIT_IRQ` shape).  Combined with [`WAIT_DELIVERY`](Self::WAIT_DELIVERY)
        /// it is the deadline-or-doorbell park (§3.4.4): first of
        /// {delivery → [`Status::Ok`], reap → [`Status::Revoked`],
        /// deadline → [`Status::Deadline`]} wins, and a deadline win leaves
        /// the standing post standing.
        const WAIT_SLEEP = 1 << 2;
        /// Park the caller after commit on the **join** its session staged
        /// ([`DERIVE_KIND_TASK_JOIN`]) until the joined task exits or is
        /// reaped — a proper return from a thread, rather than the poll a
        /// userland exit ledger is reduced to.
        ///
        /// The wait authority is the *held* `Task` handle the join derive
        /// bound, so the authority question ("who may join?") is answered
        /// by possession, like every other verb here — never by naming a
        /// task id.  Exactly one join may ride a flagged session (the
        /// `WAIT_DELIVERY` one-post rule), and a flagged session with no
        /// staged join refuses [`Status::Malformed`] before anything
        /// commits.
        ///
        /// Combines with the other three under the same first-wins rule —
        /// a joiner keeps its doorbell live while blocked: delivery →
        /// [`Status::Ok`], exit → [`Status::Exited`] (the join's result
        /// slot holds the [`ExitReport`]), reap of either the post or the
        /// joinee → [`Status::Revoked`], deadline → [`Status::Deadline`],
        /// IRQ → [`Status::Irq`].  Delivery is checked first because it is
        /// the one **destructive** edge (a post is consumed and a
        /// capability has landed); an exit is not consumed by observing
        /// it, so a join that loses that race wins the caller's very next
        /// park.
        const WAIT_TASK = 1 << 3;
        /// Add a second **standing** delivery post to an otherwise empty
        /// [`WAIT_DELIVERY`](Self::WAIT_DELIVERY) park. The park block carries
        /// two adjacent [`ParkReq`]s; either delivery/reap wakes the caller,
        /// and an optional [`WAIT_SLEEP`](Self::WAIT_SLEEP) deadline races both.
        ///
        /// This is the bounded L4-style receive-any shape used when a service
        /// must sleep on its ordinary doorbell and one affine reply/return
        /// endpoint without a second kick. It is deliberately exactly two
        /// posts: the 8-byte header plus 32-byte post block (plus the optional
        /// 8-byte deadline) remains wholly in x86-64's 48-byte GPR carrier.
        /// A primary delivery resolves [`Status::Ok`]; an alternate delivery
        /// resolves [`Status::AltDelivery`]. If both complete before the task
        /// resumes, the alternate outcome wins so affine custody cannot be
        /// hidden behind a coalescible doorbell wake.
        /// It is malformed without `WAIT_DELIVERY`, with an in-session post,
        /// or combined with `WAIT_IRQ`/`WAIT_TASK`.
        const WAIT_DELIVERY_ALT = 1 << 4;
        /// Select the held route named by [`IrqParkReq`] instead of the task's
        /// legacy default route. Requires `WAIT_IRQ`; the request follows the
        /// delivery and sleep payloads in the park block. The kernel binds
        /// the arena record and resolves the endpoint from the held capability.
        const WAIT_IRQ_SELECTED = 1 << 5;
    }
}

/// # The park block (the flag-gated argument-region convention)
///
/// A parking session's wire payloads ride the **front of the argument
/// region** — never trailing descriptor bytes (the descriptor has no end
/// marker, so "trailing" bytes would just be argument-region offsets that
/// alias any invocation's `arg == 0`), and never a dedicated carrier
/// invocation (the canonical parking commit is an *empty* session, which
/// has no invocation to carry an `arg`).  The flags say exactly what the
/// block holds, in pinned order:
///
/// | flags                                        | args\[0..\]              |
/// |----------------------------------------------|--------------------------|
/// | `WAIT_SLEEP`                                 | [`SleepReq`]             |
/// | `WAIT_DELIVERY`, zero posts                  | [`ParkReq`]              |
/// | `WAIT_DELIVERY` (zero posts) + `WAIT_SLEEP`  | [`ParkReq`] ‖ [`SleepReq`] |
/// | `WAIT_DELIVERY | WAIT_DELIVERY_ALT`         | [`ParkReq`] ‖ [`ParkReq`] |
/// | above + `WAIT_SLEEP`                         | [`ParkReq`] ‖ [`ParkReq`] ‖ [`SleepReq`] |
/// | `WAIT_DELIVERY`, one post (+ `WAIT_SLEEP`)   | — ([`SleepReq`] if flagged) |
/// | any of the above + `WAIT_IRQ`                | unchanged — legacy default route |
/// | above + `WAIT_IRQ_SELECTED`                  | append [`IrqParkReq`] |
/// | any of the above + `WAIT_TASK`               | unchanged — the staged join derive carries its own subject |
///
/// A flagged session whose argument region cannot supply the block
/// refuses [`Status::Malformed`] before anything commits.  Ordinary
/// invocation payloads follow the block at higher offsets (an
/// invocation's `arg` may legally alias the block — every reader
/// bounds-checks and validates its own bytes).
///
/// Argument payload for a **`WAIT_SLEEP`** park (USERLAND.md §3.4.2): the
/// absolute monotonic-nanosecond deadline the committed session parks
/// until on the per-CPU timer wheel.  LE byteorder (`align(1)`), read via
/// [`try_ref_from_prefix`](zerocopy::TryFromBytes::try_ref_from_prefix)
/// — the `*Req` argument-region discipline.
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, KnownLayout, Immutable, Unaligned)]
pub struct SleepReq {
    /// Absolute deadline, monotonic nanoseconds since boot.
    pub deadline_ns: U64,
}

impl SleepReq {
    /// Wire size (repr(C), 8).
    pub const WIRE: usize = 8;
}

/// An explicitly selected held IRQ route. The slot names an authenticated
/// capability record, never a raw interrupt number or endpoint generation.
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, KnownLayout, Immutable, Unaligned)]
pub struct IrqParkReq {
    pub slot: U32,
    /// Reserved; must be zero.
    pub reserved: U32,
}

impl IrqParkReq {
    pub const WIRE: usize = 8;
}

/// Argument payload for a **park-on-standing-post** `WAIT_DELIVERY`
/// (USERLAND.md §3.4.3; SYSCALL.md D2 tier 1): names the standing recv
/// post — committed by a *previous* session — the caller parks on.  The
/// kernel bind-checks the named post against the ambient caller (the
/// board entry's **holder** must be the caller and `recv_tag` must be the
/// post's live recv tag) — without that check any task could park on
/// another's post and displace its waiter.  Rides the park block (see
/// [`SleepReq`]'s convention table).  LE byteorder (`align(1)`).
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, KnownLayout, Immutable, Unaligned)]
pub struct ParkReq {
    /// The channel whose board entry holds the standing post.
    pub chan: U32,
    /// The posted arena slot (the board packs slots as `u16`).
    pub slot: U16,
    /// The posting recv end's tag — the post's liveness pin.
    pub recv_tag: U32,
    /// Reserved; must be zero.
    pub reserved: [u8; 6],
}

impl ParkReq {
    /// Wire size (repr(C), 4+2+4+6).
    pub const WIRE: usize = 16;
}

/// Header of a session descriptor staged in the arena: a bounded run of
/// [`Invocation`]s the kernel folds into one perturbation and commits with
/// a single closure pass.
///
/// LE byteorder fields, any bit pattern valid — the kernel reads it off
/// the front of the untrusted descriptor with
/// [`read_from_prefix`](FromBytes::read_from_prefix) and range-checks
/// `count`/`abi` itself. **8 bytes** (`count ‖ abi ‖ flags ‖ reserved` — laid
/// down in ABI v2, unchanged by v3, which grew only [`FoundReq`]):
/// the `flags`/`reserved` words are frozen headroom (the #71 blocking tier and
/// future session-level bits) so growth stays additive, never a layout break.
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, KnownLayout, Immutable, Unaligned)]
pub struct SessionHeader {
    /// Number of [`Invocation`]s that follow (<= [`MAX_INVOCATIONS`]).
    pub count: U16,
    /// ABI version of this descriptor layout ([`DESCRIPTOR_ABI`]).
    pub abi: U16,
    /// Session flags ([`SessionFlags`]): `WAIT_DELIVERY` / `WAIT_IRQ` /
    /// `WAIT_SLEEP` / `WAIT_TASK` / `WAIT_DELIVERY_ALT`, all honored by the engine; unknown bits refuse
    /// [`Status::Malformed`].
    pub flags: U16,
    /// Reserved; must be zero. Frozen additive headroom.
    pub reserved: U16,
}

// Frozen wire pins (#81): a layout drift here is a *security* bug, not a style
// one — the kernel reads these bytes off untrusted memory.
const _: () = {
    use core::mem::size_of;
    assert!(size_of::<SessionHeader>() == 8);
    assert!(size_of::<InvocationWire>() == 16);
    assert!(size_of::<RegionMapReq>() == RegionMapReq::WIRE);
    assert!(size_of::<LeaseReq>() == LeaseReq::WIRE);
    assert!(size_of::<TrackedLeaseReq>() == TrackedLeaseReq::WIRE);
    assert!(size_of::<SupplyReleaseReq>() == SupplyReleaseReq::WIRE);
    assert!(size_of::<BarReq>() == BarReq::WIRE);
    assert!(size_of::<InventoryReq>() == InventoryReq::WIRE);
    assert!(size_of::<I2cHidReq>() == I2cHidReq::WIRE);
    assert!(size_of::<IrqReq>() == IrqReq::WIRE);
    assert!(size_of::<ShareReq>() == ShareReq::WIRE);
    assert!(size_of::<ViewMapReq>() == ViewMapReq::WIRE);
    assert!(size_of::<DeviceShareReq>() == DeviceShareReq::WIRE);
    assert!(size_of::<GuestShareReq>() == GuestShareReq::WIRE);
    assert!(size_of::<PeerDeviceShareReq>() == PeerDeviceShareReq::WIRE);
    assert!(size_of::<UnshareReq>() == UnshareReq::WIRE);
    assert!(size_of::<PagerReq>() == PagerReq::WIRE);
    assert!(size_of::<DelegateReq>() == DelegateReq::WIRE);
    assert!(size_of::<ShareToReq>() == ShareToReq::WIRE);
    assert!(size_of::<FoundReq>() == FoundReq::WIRE);
    assert!(size_of::<SleepReq>() == SleepReq::WIRE);
    assert!(size_of::<IrqParkReq>() == IrqParkReq::WIRE);
    assert!(size_of::<ExitReq>() == ExitReq::WIRE);
    assert!(size_of::<DeviceActivateReq>() == DeviceActivateReq::WIRE);
    assert!(size_of::<GuestFoundReq>() == GuestFoundReq::WIRE);
    assert!(size_of::<ParkReq>() == ParkReq::WIRE);
    assert!(size_of::<RecvPolicyReq>() == RecvPolicyReq::WIRE);
    assert!(size_of::<NoticeReq>() == NoticeReq::WIRE);
    assert!(size_of::<TpReq>() == TpReq::WIRE);
    assert!(size_of::<ProtectReq>() == ProtectReq::WIRE);
    assert!(size_of::<PortReq>() == PortReq::WIRE);
    // The self-verb discriminants share one space above `ResourceKind`
    // and are told apart by SUBJECT kind, not by value — `SET_TP` and
    // `EXEC_SEAL` are deliberately both `0xfd` (AddrSpace vs MemLease).
    // `PROTECT` takes an **AddrSpace** subject, so it is `SET_TP` it must
    // differ from; a collision there would make one verb silently select
    // the other on the caller's own space.
    assert!(DERIVE_KIND_PROTECT != DERIVE_KIND_SET_TP);
    // Resource-kind derives occupy the low, append-only band.  START owns one
    // value in the global high-flavor band, and must remain distinct from
    // every defined flavor even where subject kinds currently disambiguate.
    assert!((crate::cap::ResourceKind::SharedRamRegion as u16) < DERIVE_KIND_TASK_START as u16);
    assert!(DERIVE_KIND_TASK_START != DERIVE_KIND_IOVA_ANON);
    assert!(DERIVE_KIND_TASK_START != DERIVE_KIND_SET_TP);
    assert!(DERIVE_KIND_TASK_START != DERIVE_KIND_PROTECT);
    assert!(DERIVE_KIND_TASK_START != DERIVE_KIND_TASK_EXIT);
    assert!(DERIVE_KIND_TASK_START != DERIVE_KIND_TASK_JOIN);
    assert!(DERIVE_KIND_TASK_START != DERIVE_KIND_DEVICE_RECORD);
    assert!(DERIVE_KIND_TASK_START != DERIVE_KIND_EXEC_SEAL);
    // ARENA_EXTEND shares the AddrSpace subject with SET_TP and PROTECT,
    // so all three must be pairwise distinct — a collision would make one
    // self-verb silently select another on the caller's own space.  It
    // also owns one value in the global high-flavor band.
    assert!(DERIVE_KIND_ARENA_EXTEND != DERIVE_KIND_SET_TP);
    assert!(DERIVE_KIND_ARENA_EXTEND != DERIVE_KIND_PROTECT);
    assert!(DERIVE_KIND_ARENA_EXTEND != DERIVE_KIND_TASK_START);
    assert!(DERIVE_KIND_ARENA_EXTEND != DERIVE_KIND_TASK_EXIT);
    assert!(DERIVE_KIND_ARENA_EXTEND != DERIVE_KIND_TASK_JOIN);
    assert!(DERIVE_KIND_ARENA_EXTEND != DERIVE_KIND_DEVICE_RECORD);
    assert!(DERIVE_KIND_ARENA_EXTEND != DERIVE_KIND_IOVA_ANON);
    assert!(DERIVE_KIND_ARENA_EXTEND != DERIVE_KIND_EXEC_SEAL);
    assert!((crate::cap::ResourceKind::SharedRamRegion as u16) < DERIVE_KIND_ARENA_EXTEND as u16);
    // DEVICE_ACTIVATE shares the PciDevice subject with resource derives but
    // owns the next high-band byte; keep it distinct from every effect flavor.
    assert!(DERIVE_KIND_DEVICE_ACTIVATE != DERIVE_KIND_ARENA_EXTEND);
    assert!(DERIVE_KIND_DEVICE_ACTIVATE != DERIVE_KIND_TASK_START);
    assert!(DERIVE_KIND_DEVICE_ACTIVATE != DERIVE_KIND_TASK_EXIT);
    assert!(DERIVE_KIND_DEVICE_ACTIVATE != DERIVE_KIND_TASK_JOIN);
    assert!(DERIVE_KIND_DEVICE_ACTIVATE != DERIVE_KIND_DEVICE_RECORD);
    assert!(DERIVE_KIND_DEVICE_ACTIVATE != DERIVE_KIND_IOVA_ANON);
    // DeviceMmu is the current highest resource kind; the whole band must
    // stay below the derive-kind flavors.
    assert!(
        (crate::cap::ResourceKind::SharedRamRegion as u16) < DERIVE_KIND_DEVICE_ACTIVATE as u16
    );
    // The strict MSI-X route shares the PciDevice subject with ordinary route,
    // BAR, self-grant, and DEVICE_ACTIVATE derives. Its high flavor must remain
    // outside the resource band and distinct from the other PciDevice effect.
    assert!((crate::cap::ResourceKind::SharedRamRegion as u16) < DERIVE_KIND_IRQ_ROUTE_MSIX as u16);
    assert!(DERIVE_KIND_IRQ_ROUTE_MSIX != DERIVE_KIND_DEVICE_ACTIVATE);
    assert!(DERIVE_KIND_IRQ_ROUTE_MSIX != DERIVE_KIND_SYSTEM_OBSERVE);
    // The I2C-HID observation shares the PciDevice subject with BAR, route,
    // activation, and strict-MSI-X derives. Its selector must therefore be
    // distinct from both high flavors on that same subject.
    assert!((crate::cap::ResourceKind::PlatformBus as u16) < DERIVE_KIND_I2C_HID_RECORD as u16);
    assert!(DERIVE_KIND_I2C_HID_RECORD != DERIVE_KIND_DEVICE_ACTIVATE);
    assert!(DERIVE_KIND_I2C_HID_RECORD != DERIVE_KIND_IRQ_ROUTE_MSIX);
    assert!(
        (crate::cap::ResourceKind::PlatformBus as u16) < DERIVE_KIND_I2C_HID_READY_ROUTE as u16
    );
    assert!(DERIVE_KIND_I2C_HID_READY_ROUTE != DERIVE_KIND_IRQ_ROUTE_LINE);
    assert!(DERIVE_KIND_I2C_HID_READY_ROUTE != DERIVE_KIND_I2C_HID_RECORD);
    assert!((crate::cap::ResourceKind::SharedRamRegion as u16) < DERIVE_KIND_PEER_IOVA as u16);
    assert!(DERIVE_KIND_PEER_IOVA != DERIVE_KIND_IRQ_ROUTE_MSIX);
    assert!(DERIVE_KIND_PEER_IOVA != DERIVE_KIND_DEVICE_ACTIVATE);
    // Task-subject flavors must be pairwise distinct for the same reason
    // the AddrSpace family is — one subject kind, so only the value tells
    // the verbs apart (the gap the 0xFF05 report collision proved real).
    assert!(DERIVE_KIND_TASK_EXIT != DERIVE_KIND_TASK_JOIN);
    // GUEST_FOUND shares its GuestSpace subject with the ordinary low-band
    // self-derive.  Keeping it outside that band and distinct from every
    // registered high flavor makes an old/new ABI mismatch refuse loudly.
    assert!((crate::cap::ResourceKind::GuestRegion as u16) < DERIVE_KIND_GUEST_FOUND as u16);
    assert!(DERIVE_KIND_GUEST_FOUND != DERIVE_KIND_SYSTEM_OBSERVE);
    assert!(DERIVE_KIND_GUEST_FOUND != DERIVE_KIND_DEVICE_ACTIVATE);
    assert!(DERIVE_KIND_GUEST_FOUND != DERIVE_KIND_ARENA_EXTEND);
    assert!(DERIVE_KIND_GUEST_FOUND != DERIVE_KIND_TASK_START);
};

/// **The `Derive` high-flavor roster** — every `DERIVE_KIND_*` value, in
/// one table.  Two jobs: the prepare engine's fail-closed vocabulary gate
/// ([`derive_kind_known`] — the `Revoke` arm has refused unknown kind
/// bytes since it existed, while an unknown high byte on `Derive` used to
/// fall through and run as a *plain* derive, which is exactly how a
/// future flavor from a newer userland would silently misfire on an
/// older kernel), and the anchor a new flavor registers itself on (a
/// missing row here means the new verb refuses everywhere, which is the
/// loud failure). Values repeat lawfully: `SET_TP` and retired `EXEC_SEAL`
/// share `0xfd`, while `TASK_OBSERVER` and `SYSTEM_OBSERVE` share `0xf5`.
/// Their authenticated subject kinds disambiguate both pairs. This table is a
/// vocabulary, not a uniqueness proof; the pairwise same-subject assertions
/// above carry the distinctness that matters.
const DERIVE_FLAVORS: [u8; 36] = [
    DERIVE_KIND_TASK_IDENTITY,
    DERIVE_KIND_TASK_NAME,
    DERIVE_KIND_PROCESS_DRAIN,
    DERIVE_KIND_CRASH_READ,
    DERIVE_KIND_PANIC_REPORT,
    DERIVE_KIND_CRASH_BIND,
    DERIVE_KIND_VIRTIO_PCI_LAYOUT,
    DERIVE_KIND_IOVA_ANON,
    DERIVE_KIND_SET_TP,
    DERIVE_KIND_PROTECT,
    DERIVE_KIND_TASK_EXIT,
    DERIVE_KIND_TASK_JOIN,
    DERIVE_KIND_DEVICE_RECORD,
    DERIVE_KIND_TASK_START,
    DERIVE_KIND_ARENA_EXTEND,
    DERIVE_KIND_DEVICE_ACTIVATE,
    DERIVE_KIND_SYSTEM_OBSERVE,
    DERIVE_KIND_TASK_OBSERVER,
    DERIVE_KIND_IRQ_ROUTE_MSIX,
    DERIVE_KIND_I2C_HID_RECORD,
    DERIVE_KIND_SUPPLY_RELEASE,
    DERIVE_KIND_VIEW_MAP,
    DERIVE_KIND_IRQ_ROUTE_LINE,
    DERIVE_KIND_I2C_HID_READY_ROUTE,
    DERIVE_KIND_GUEST_FOUND,
    DERIVE_KIND_DEVICE_QUIESCE,
    DERIVE_KIND_PEER_IOVA,
    DERIVE_KIND_FABRIC_OBSERVE,
    DERIVE_KIND_OWNED_MAP,
    DERIVE_KIND_CPU_VIEW_OBSERVE,
    DERIVE_KIND_FABRIC_CLOSE,
    DERIVE_KIND_FABRIC_SCOPE,
    DERIVE_KIND_FABRIC_ROUTE_OBSERVE,
    DERIVE_KIND_DEVICE_CLAIM_CLOSE,
    DERIVE_KIND_DEVICE_CLAIM_QUIESCE,
    DERIVE_KIND_DEVICE_CLAIM_DMA_RETIRE,
];

/// Is `kind` a byte this ABI's `Derive` speaks? — the low, append-only
/// [`ResourceKind`](crate::cap::ResourceKind) band (`0` = unspecified,
/// then the resource vocabulary), or a registered high flavor.  Anything
/// between is a future ABI this kernel does not speak, and the engine
/// refuses it rather than running it as an unflavored derive (the
/// `Revoke` arm's rule, finally applied to `Derive`).
pub const fn derive_kind_known(kind: u8) -> bool {
    if kind <= crate::cap::ResourceKind::SharedRamRegion as u8 {
        return true;
    }
    let mut i = 0;
    while i < DERIVE_FLAVORS.len() {
        if DERIVE_FLAVORS[i] == kind {
            return true;
        }
        i += 1;
    }
    false
}

const _: () = {
    // Every registered flavor sits above the resource band — the two
    // vocabularies must never interleave, or the predicate's low-band
    // admit would swallow a flavor.
    let mut i = 0;
    while i < DERIVE_FLAVORS.len() {
        assert!(DERIVE_FLAVORS[i] > crate::cap::ResourceKind::SharedRamRegion as u8);
        i += 1;
    }
};

/// **The bind-inert report family's kind roster** — every `*_KIND`
/// discriminant, in one table, so distinctness is a loop instead of a
/// hand-picked (and, history shows, incomplete) list of inequalities:
/// the family shipped with [`EXIT_REPORT_KIND`] and [`DEVICE_REPORT_KIND`]
/// both `0xFF05` for ~80 commits because nothing owned the proof.  A new
/// member joins by adding its row; forgetting to is caught by the arity
/// assert any time the member count moves.
const REPORT_KINDS: [u16; 22] = [
    TASK_IDENTITY_KIND,
    PROCESS_DRAIN_REPORT_KIND,
    CRASH_CHUNK_KIND,
    VIRTIO_PCI_CAP_REPORT_KIND,
    VIRTIO_PCI_CAP_COUNT_REPORT_KIND,
    NOTICE_KIND,
    GEOM_REPORT_KIND,
    DISPLAY_REPORT_KIND,
    EXIT_REPORT_KIND,
    DEVICE_REPORT_KIND,
    DEVICE_COUNT_REPORT_KIND,
    SYSTEM_OBSERVE_REPORT_KIND,
    I2C_HID_REPORT_KIND,
    I2C_HID_COUNT_REPORT_KIND,
    SUPPLY_RELEASE_REPORT_KIND,
    I2C_ADMISSION_REPORT_KIND,
    QUIESCE_REPORT_KIND,
    FABRIC_DOMAIN_REPORT_KIND,
    FABRIC_ROUTE_CLOSE_REPORT_KIND,
    FABRIC_ROUTE_INFO_KIND,
    DEVICE_CLAIM_REPORT_KIND,
    DEVICE_CLAIM_CLOSE_REPORT_KIND,
];
const _: () = {
    // Pairwise distinct — the kind is the member's identity.
    let mut i = 0;
    while i < REPORT_KINDS.len() {
        let mut j = i + 1;
        while j < REPORT_KINDS.len() {
            assert!(
                REPORT_KINDS[i] != REPORT_KINDS[j],
                "two report members share a kind discriminant"
            );
            j += 1;
        }
        i += 1;
    }
    // And every member sits outside the ResourceKind space — a report is
    // data, not a resource, and must never bind as one.
    let mut i = 0;
    while i < REPORT_KINDS.len() {
        assert!(REPORT_KINDS[i] > crate::cap::ResourceKind::InvocationWitness as u16);
        i += 1;
    }
};

impl Status {
    /// Pack a per-session status word for the gate's return register:
    /// `[31:16] failing invocation index | [15:0] status`.  Pure layout —
    /// the kernel writes it, userland unpacks it ([`Committed`] — the
    /// pack/unpack pair lives together here so the layout cannot drift).
    ///
    /// The high half (`[63:32]`) is the **clock stamp** ([`stamp_clock`]),
    /// applied by the kernel at the ring-3 boundary; `encode` itself leaves
    /// it zero (kernel-internal status words are unstamped).
    #[inline]
    pub const fn encode(self, index: u16) -> u64 {
        ((index as u64) << 16) | self as u64
    }
}

/// Stamp the kernel's monotonic clock into a status word's free high half:
/// `[63:32] monotonic milliseconds (mod 2^32) | [31:0] the word unchanged`.
///
/// Applied by the kernel to **every** real status word crossing to ring 3
/// (the syscall return and the parked-resume return register), so a task
/// gets a fresh clock anchor from each trap without a dedicated time verb —
/// the userland executor's event-driven clock
/// (`kore-rt`'s `sleep_until`/`sleep_for`) advances on it instead of
/// lagging boot-age behind the wheel it arms deadlines on.
///
/// **Wrap**: 2^32 ms ≈ 49.7 days. Consumers must extend the low word against
/// their full monotonic anchor, rather than compare it as an absolute time.
/// Without a full clock source, ordering requires samples less than half a
/// period apart; the low word alone cannot recover an unknown boot epoch.
/// A full calibrated clock can supersede this stamp without unpacking it:
/// readers use [`Committed::clock_ms`], writers use this, and both live
/// here so the layout cannot drift.
///
/// **Sentinel disjointness**: the kernel's non-status sentinels (its park
/// unwind and no-caller words) carry `0xFFFF` in bits `[15:0]` — not in
/// the [`Status`] vocabulary and never produced by [`Status::encode`] — so
/// no stamped real status can collide with them at any clock value.
#[inline]
pub const fn stamp_clock(word: u64, now_ms: u32) -> u64 {
    (word & 0xFFFF_FFFF) | ((now_ms as u64) << 32)
}

/// The gate's packed per-session **status word**, typed — the unpack half
/// of [`Status::encode`] (the pair lives together here, `res/USERLAND.md`
/// §4): `[31:16] failing invocation index | [15:0] status`.  The kernel
/// writes the raw word into the commit's return register; the userland
/// transport seam (`kore-rt`'s `Transport`) wraps it in this newtype and
/// callers read it through the typed accessors, never by shifting bits.
///
/// Produced capabilities do **not** ride this word — results land in the
/// caller's own arena result slots named at staging; `Committed` carries
/// only the outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Committed(u64);

impl Committed {
    /// Wrap a raw status word (the gate's return register / a mock
    /// engine's [`Status::encode`] product).
    #[inline]
    pub const fn from_raw(word: u64) -> Self {
        Self(word)
    }

    /// The raw packed word.
    #[inline]
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// The per-session [`Status`] (bits `[15:0]`).  **Fail-closed
    /// decode**: a word whose status bits fall outside the [`Status`]
    /// vocabulary reads as [`Status::Malformed`] — an unrecognized
    /// outcome is never mistaken for success.
    #[inline]
    pub const fn status(self) -> Status {
        match self.0 as u16 {
            0 => Status::Ok,
            1 => Status::Denied,
            2 => Status::Forged,
            3 => Status::Revoked,
            4 => Status::Conflict,
            5 => Status::Malformed,
            6 => Status::WouldBlock,
            7 => Status::Deadline,
            8 => Status::Irq,
            9 => Status::Exited,
            10 => Status::AltDelivery,
            11 => Status::Unfounded,
            _ => Status::Malformed,
        }
    }

    /// The failing invocation's index (bits `[31:16]`) — meaningful when
    /// [`status`](Self::status) is not [`Status::Ok`]; the kernel encodes
    /// `0` on success.
    #[inline]
    pub const fn failing_index(self) -> u16 {
        (self.0 >> 16) as u16
    }

    /// The kernel's clock stamp (bits `[63:32]`, [`stamp_clock`]): monotonic
    /// milliseconds mod 2^32, stamped at the ring-3 boundary.  Reads `0` on
    /// an unstamped word (a kernel-internal status, a mock transport). Zero
    /// is also a valid wrapped native stamp; consumers extend native samples
    /// against their full clock anchor rather than treating zero as absent.
    #[inline]
    pub const fn clock_ms(self) -> u32 {
        (self.0 >> 32) as u32
    }
}

/// Argument payload for a **guarded region map** derive (Flow 1's stack
/// map): the caller owns its address-space layout, so it names the
/// virtual window; the kernel allocates the backing (manifold-placed)
/// and installs the map with the requested guard gap (#27).  Referenced
/// from [`Invocation::arg`] as a byte offset into the descriptor's
/// argument region.
///
/// The fields are little-endian byteorder types ([`U64`]/[`U32`]), so the
/// struct is `align(1)` and the kernel reads it straight out of the
/// untrusted argument bytes with [`try_ref_from_prefix`] — no
/// hand-rolled `from_le_bytes`, and the LE wire contract is in the types
/// rather than a comment.  Endianness-portable by construction; every
/// [`get`](U64::get) yields the host-order value.
///
/// [`try_ref_from_prefix`]: zerocopy::TryFromBytes::try_ref_from_prefix
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct RegionMapReq {
    /// Virtual start of the window (page-aligned).
    pub va: U64,
    /// Window length in bytes (page-granular).
    pub bytes: U64,
    /// Guard pages below the window.
    pub guard_before: U32,
    /// Guard pages above the window.
    pub guard_after: U32,
}

impl RegionMapReq {
    /// Wire size (repr(C), 8+8+4+4).
    pub const WIRE: usize = 24;
}

/// Argument payload for a **lease** derive (`Derive × PhysRegion →
/// MemLease`): allocate a contiguous buffer of `size` bytes from the region
/// and lease it — the shared root a driver then shares into its own space (a
/// CPU window) *and* its device's IOMMU domain (an IOVA window), zero-copy.
/// Referenced from [`Invocation::arg`].  LE byteorder (`align(1)`), read via
/// [`try_ref_from_prefix`](zerocopy::TryFromBytes::try_ref_from_prefix).
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct LeaseReq {
    /// Size in bytes of the buffer to allocate and lease (must be non-zero).
    pub size: U64,
    /// Lease attribute flags — [`LEASE_SEALED`] in bit 0, [`LEASE_EXEC`]
    /// in bit 1.  The remaining bits (and `reserved`) are frozen zero
    /// headroom (additive).
    pub flags: u8,
    /// Requested physical alignment of the backing, as a log2 byte count:
    /// `0` = unspecified (the 4 KiB default), else exactly one of
    /// [`LEASE_ALIGN_4K`] / [`LEASE_ALIGN_2M`] / [`LEASE_ALIGN_1G`] — any
    /// other value refuses (`BadArgs`).  A huge-aligned backing is what
    /// lets every later map of the window take the matching large leaf
    /// (the walker picks the best leaf from virt/phys/len alignment), so
    /// a driver's DMA pool can ride one 2 MiB TLB entry instead of 512
    /// small ones.  Alignment is a placement request only — it neither
    /// rounds `size` nor promises large leaves by itself.
    pub align_log2: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 6],
}

/// [`LeaseReq::flags`] bit 0: **swappable-sealed** (the swap seal) — the
/// kernel encrypts + authenticates each page under its own key on eviction,
/// so neither the lessor nor the swap backend ever sees plaintext (nor can
/// tamper undetected).  The tenant's choice: sealing defeats the backend's
/// compression, so it is opt-in per lease.
pub const LEASE_SEALED: u8 = 1 << 0;

/// [`LeaseReq::flags`] bit 1: request an **initially executable** lease
/// (res/IMAGE.md §6, the VMEX analogue) — shares off this lease may request
/// [`SHARE_ACCESS_X`]; `WRITE` shares are refused unconditionally,
/// forever (immutable text with no unprotect path).  Honored by dispatch
/// only when the invoking task holds a live
/// [`ExecMint`](crate::cap::ResourceKind::ExecMint) capability — the
/// mint is checked once at lease-derive, never per-share. Ordinary writable
/// leases can instead reach the same terminal property through consuming
/// [`REVOKE_KIND_EXEC`] finalization.
pub const LEASE_EXEC: u8 = 1 << 1;

/// [`LeaseReq::flags`] bit 2: the request carries the additive
/// [`TrackedLeaseReq`] tail. This is kernel/service lifecycle vocabulary;
/// ordinary client leases remain the frozen 16-byte [`LeaseReq`] shape.
pub const LEASE_TRACK_RELEASE: u8 = 1 << 2;

/// [`LeaseReq::align_log2`]: the 4 KiB default, spelled explicitly.
pub const LEASE_ALIGN_4K: u8 = 12;
/// [`LeaseReq::align_log2`]: a 2 MiB-aligned backing (one x86 large leaf).
pub const LEASE_ALIGN_2M: u8 = 21;
/// [`LeaseReq::align_log2`]: a 1 GiB-aligned backing (one x86 giant leaf).
pub const LEASE_ALIGN_1G: u8 = 30;

impl LeaseReq {
    /// Wire size (repr(C), 8+1+1+6).
    pub const WIRE: usize = 16;
}

/// Additive tracked-supply lease request. The [`LeaseReq`] prefix preserves
/// the v1 wire layout; the tail is accepted only with [`LEASE_TRACK_RELEASE`].
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct TrackedLeaseReq {
    pub lease: LeaseReq,
    /// MM-owned generational reservation identity.
    pub receipt: U64,
    /// Structural arrival-pair identity supplied by MM, never by its client.
    pub pair: U16,
    /// Reserved; must be zero.
    pub reserved: [u8; 6],
}

impl TrackedLeaseReq {
    pub const WIRE: usize = 32;
}

/// Argument payload for a supply-release take (`Derive × PhysRegion`,
/// [`DERIVE_KIND_SUPPLY_RELEASE`]): the arrival pair whose completions the
/// caller polls. Pair-keyed so the kernel consults exactly that pair's
/// lifecycle lane instead of sweeping every lane per poll.
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct SupplyReleaseReq {
    /// Structural arrival-pair identity, as [`TrackedLeaseReq::pair`].
    pub pair: U16,
    /// Reserved; must be zero.
    pub reserved: [u8; 6],
}

impl SupplyReleaseReq {
    pub const WIRE: usize = 8;
}

/// Argument payload for a **BAR claim** derive (`Derive × PciDevice →
/// MmioRegion`, #85): claim BAR `bar` of the *subject* device as a device-
/// rooted MMIO region, ready to map into an address space (the second hop, a
/// [`ShareReq`] `Derive × MmioRegion → LeasedRegion`).  The device is the
/// invocation subject (not an arg slot — a BAR belongs to its device), so this
/// carries only the BAR index; the window's uncached attrs come from the
/// hardware claim, not the caller. Canonical producers set
/// [`Invocation::kind`] to
/// [`ResourceKind::MmioRegion`](crate::cap::ResourceKind::MmioRegion). The
/// original frozen descriptor used `kind = 0`; kernels accept that exact legacy
/// spelling, but no other selector may fall through to the BAR decoder. The
/// MMIO sibling of [`LeaseReq`]. LE byteorder (`align(1)`).
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct BarReq {
    /// Which BAR of the subject device to claim (0–5).
    pub bar: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 7],
}

impl BarReq {
    /// Wire size (repr(C), 1+7).
    pub const WIRE: usize = 8;
}

/// Argument payload for **device-inventory** derives off an inventory
/// subject — a `RootComplex` (PCI functions, root-local index) or the
/// `PlatformBus` (firmware-described devices): which query to run and, for
/// an entry query, which subject-local index to act on.
///
/// One arg struct for both subjects and all flavors, because the selector
/// shape is the same — [`DERIVE_KIND_DEVICE_RECORD`] reads an entry or the
/// authoritative per-subject count, and a `Derive × RootComplex →
/// PciDevice` / `Derive × PlatformBus → PlatformDevice` hands one entry
/// over. The index is the *only* thing a claim names, and it is local to
/// the subject: which root's namespace it indexes is decided by the
/// capability presented, never by a wire field.
///
/// LE byteorder (`align(1)`), the [`BarReq`] discipline. (Wire history:
/// named `DeviceStoreReq` until #572; the 8-byte layout is frozen.)
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct InventoryReq {
    /// The subject-local inventory index to read or claim. Must be zero
    /// for [`INVENTORY_QUERY_COUNT`].
    pub index: U32,
    /// Query selector. Claims accept only [`INVENTORY_QUERY_RECORD`].
    pub query: u8,
    /// Reserved; must be zero in every query.
    pub reserved: [u8; 3],
}

impl InventoryReq {
    /// Wire size (repr(C), 4+1+3).
    pub const WIRE: usize = 8;
}

/// Close an exact PCI grant through its inventory root's REVOKE authority.
/// This is a singular, unflagged post-commit effect with a bind-inert report.
/// It closes admission only: never releases a device or proves DMA cleanup.
pub const DERIVE_KIND_DEVICE_CLAIM_CLOSE: u8 = 0xe5;

/// Issuer-owned fabric containment after the exact grant is closed and drained.
/// Same canonical DeviceClaimCloseReq, singular unflagged crossing, held root
/// with REVOKE. Returns QuiesceReport; never releases the claim or backing.
pub const DERIVE_KIND_DEVICE_CLAIM_QUIESCE: u8 = 0xe4;

/// Retire registered DMA windows under the exact closed/drained claim issuer.
/// Uses DeviceClaimCloseReq and returns DeviceClaimDmaReport, never a regrant.
pub const DERIVE_KIND_DEVICE_CLAIM_DMA_RETIRE: u8 = 0xe3;

#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct DeviceClaimCloseReq {
    pub index: U32,
    pub reserved: U32,
    pub generation: U64,
    pub grant: U64,
}

impl DeviceClaimCloseReq {
    pub const WIRE: usize = 24;
    pub fn valid(&self) -> bool {
        self.reserved.get() == 0
            && self.generation.get() != 0
            && DeviceClaimReport::decode(
                &DeviceClaimReport {
                    generation: self.generation.get(),
                    grant: crate::cap::Cid(self.grant.get()),
                }
                .encode(),
            )
            .is_some()
    }
}

/// Read or claim one inventory entry named by [`InventoryReq::index`].
pub const INVENTORY_QUERY_RECORD: u8 = 0;

/// Read the authoritative, boot-frozen count of the subject's inventory.
///
/// The result is a bind-inert [`DeviceCountReport`]. The index must be zero,
/// which keeps the count shape canonical and rejects a selector smuggled
/// beside an entry index.
pub const INVENTORY_QUERY_COUNT: u8 = 1;

/// Read a bounded I2C admission page from one RootComplex. The index is a
/// page ordinal, not a caller-supplied BDF or device authority. PlatformBus
/// does not implement this PCI-specific diagnostic query.
pub const INVENTORY_QUERY_I2C_ADMISSION: u8 = 2;

/// Observe the exact committed PCI claim at this root-local index. Only a
/// RootComplex accepts this selector; it neither claims nor releases a device.
pub const INVENTORY_QUERY_DEVICE_CLAIM: u8 = 3;

/// Query one claimed platform controller's firmware-resolved HID-over-I2C
/// children. The 8-byte shape deliberately mirrors [`InventoryReq`], but is
/// a distinct type because the index names a child *behind one device*, not an
/// entry in the platform-bus inventory.
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct I2cHidReq {
    /// Controller-local HID child ordinal. Must be zero for
    /// [`I2C_HID_QUERY_COUNT`].
    pub index: U32,
    /// [`I2C_HID_QUERY_RECORD`] or [`I2C_HID_QUERY_COUNT`].
    pub query: u8,
    /// Reserved; every byte must be zero.
    pub reserved: [u8; 3],
}

impl I2cHidReq {
    /// Wire size (`u32 ‖ u8 ‖ [u8; 3]`).
    pub const WIRE: usize = 8;
}

/// Read one controller-local HID child.
pub const I2C_HID_QUERY_RECORD: u8 = 0;

/// Read the authoritative number of HID children behind the controller.
pub const I2C_HID_QUERY_COUNT: u8 = 1;

/// Argument payload for an interrupt-route derive (`Derive × PciDevice →
/// IrqRoute`). The route owns one consecutive MSI/MSI-X block.
///
/// `count` must be a non-zero power of two. `first_entry` selects the first
/// MSI-X table entry and must be zero for ordinary MSI. `affinity` is a
/// preferred logical CPU; [`IRQ_AFFINITY_ANY`] delegates placement entirely
/// to the kernel.
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct IrqReq {
    pub first_entry: U16,
    pub count: U16,
    pub affinity: U32,
}

impl IrqReq {
    pub const WIRE: usize = 8;
}

pub const IRQ_AFFINITY_ANY: u32 = u32::MAX;

/// Argument payload for a **CPU share** derive (`Derive × MemLease →
/// LeasedRegion`): install the lease's buffer into the caller's own address
/// space at the virtual window `[va, va+bytes)` with the requested access,
/// zero-copy (the same frames the device also maps).  Referenced from
/// [`Invocation::arg`].  LE byteorder (`align(1)`).
///
/// **Access selection** (res/IMAGE.md §9c, additive): `access == 0` keeps
/// the legacy `write` byte (`0` = RO, non-zero = RW); `access != 0` selects
/// explicit [`SHARE_ACCESS_R`]`/`[`SHARE_ACCESS_W`]`/`[`SHARE_ACCESS_X`]
/// bits and `write` is ignored.  `W+X` in one request is refused
/// unconditionally (the wire-level W^X backstop); `X` legality is the
/// subject lease's executable property, checked kernel-side.  The USER
/// leaf bit is **never** wire-selectable — the kernel derives it from the
/// target address space (bit 3 stays reserved-zero).
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct ShareReq {
    /// Virtual start of the window (page-aligned).
    pub va: U64,
    /// Window length in bytes (should equal the lease's size).
    pub bytes: U64,
    /// Legacy access byte: `0` = read-only, non-zero = read-write.
    /// Consulted only when `access == 0`.
    pub write: u8,
    /// Explicit access bits ([`SHARE_ACCESS_R`]`/W/X`); `0` = legacy
    /// (`write` selects).  Bits 3+ are frozen reserved-zero.
    pub access: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 6],
}

impl ShareReq {
    /// Wire size (repr(C), 8+8+1+1+6) — identical 24-byte layout to the
    /// pre-`access` revision (the byte carved from `reserved[0]`).
    pub const WIRE: usize = 24;
}

/// Receiver-owned placement for a detached view. Neither choice grants a
/// mapping: the task must hold the view and the founder must stamp the window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewMapWindow {
    /// Ordinary small views, bounded to this thread's stride — and heap
    /// growth on a thread without a [`Heap`](Self::Heap) window.
    Map,
    /// Large compositor-owned surfaces, separate from that heap stride.
    Surfaces,
    /// The task's heap window (`geom::HEAP`, GEOM_ABI v7): where the process
    /// heap maps the carves it takes off its own supply, a monotone cursor
    /// like [`Map`](Self::Map)'s but task-level and off the per-thread stride
    /// every view shares. Stamped for founded tasks only.
    Heap,
}

/// One of the two fixed, reusable halves of a task's `SURFACES` window.
/// A lane selects placement only. It does not grant mapping or unmap authority.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceMapLane {
    First = 0,
    Second = 1,
}

impl SurfaceMapLane {
    pub const COUNT: usize = 2;

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn from_wire(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::First),
            1 => Some(Self::Second),
            _ => None,
        }
    }
}

impl ViewMapWindow {
    pub const fn role(self) -> u16 {
        match self {
            Self::Map => crate::task::geom::MAP,
            Self::Surfaces => crate::task::geom::SURFACES,
            Self::Heap => crate::task::geom::HEAP,
        }
    }
}

/// The whole-view map request for [`DERIVE_KIND_VIEW_MAP`]. No caller-supplied
/// VA, length, destination space or access mask: each is derived from the
/// receiver's stamped geometry and authenticated view. Reserved bytes are zero.
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct ViewMapReq {
    pub window: U16,
    /// `0` for ordinary cursor placement. For `SURFACES`, selects one of
    /// [`SurfaceMapLane`]'s two fixed halves.
    pub lane: u8,
    pub reserved: [u8; 5],
}

impl ViewMapReq {
    pub const WIRE: usize = 8;

    pub fn placement(&self) -> Option<ViewMapWindow> {
        if self.reserved != [0; 5] {
            return None;
        }
        match self.window.get() {
            crate::task::geom::MAP if self.lane == 0 => Some(ViewMapWindow::Map),
            crate::task::geom::HEAP if self.lane == 0 => Some(ViewMapWindow::Heap),
            crate::task::geom::SURFACES if SurfaceMapLane::from_wire(self.lane).is_some() => {
                Some(ViewMapWindow::Surfaces)
            }
            _ => None,
        }
    }

    pub fn surface_lane(&self) -> Option<SurfaceMapLane> {
        (self.placement() == Some(ViewMapWindow::Surfaces))
            .then(|| SurfaceMapLane::from_wire(self.lane))
            .flatten()
    }
}

/// Argument payload for a **detached window carve** (`Derive × MemLease →
/// LeasedRegion`, [`kind`](Invocation::kind)` = LeasedRegion`, ckG2a-1 / F2):
/// carve a *transferable, geometry-only* window `[offset, offset+len)` off a
/// held supply lease — **no** install (the lease-holder, e.g. mm, has no
/// handle to the client's space), so the receiver installs its own leaf
/// later (`map_view`).  The kernel guards `offset`/`len` page-aligned and
/// `offset+len ≤ span` before minting (the sole over-grant defense).  Same
/// 24-byte layout `kore_rt`'s `GrantArg::Window` stages (`offset ‖ len ‖
/// class ‖ zeros`).  Referenced from [`Invocation::arg`]; LE byteorder
/// (`align(1)`).
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct WindowCarveReq {
    /// Byte offset of the window into the supply lease's contiguous backing
    /// (page-aligned).
    pub offset: U64,
    /// Window length in bytes (page-granular, non-zero, within the span).
    pub len: U64,
    /// DMA placement class byte (the tenant's [`GrantArg::Window`] `class`) —
    /// carried for the delivered token's provenance; the CPU-view carve does
    /// not consult it.
    pub class: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 7],
}

impl WindowCarveReq {
    /// Wire size (repr(C), 8+8+1+7) — identical to `GrantArg::Window`'s
    /// 24-byte staged payload.
    pub const WIRE: usize = 24;
}

/// [`ShareReq::access`] bit 0: **read**.
pub const SHARE_ACCESS_R: u8 = 1 << 0;

/// [`ShareReq::access`] bit 1: **write**.  Never combinable with
/// [`SHARE_ACCESS_X`] in one request (W^X, res/IMAGE.md §6).
pub const SHARE_ACCESS_W: u8 = 1 << 1;

/// [`ShareReq::access`] bit 2: **execute**.  Honored only for shares off a
/// executable lease (res/IMAGE.md §6); refused everywhere else.
pub const SHARE_ACCESS_X: u8 = 1 << 2;

/// Argument payload for a **device share** derive (`Derive × (MemLease |
/// MmioRegion) → LeasedIova`, [`kind`](Invocation::kind)` = IovaRegion`):
/// install a lease's buffer or claimed BAR into a device's IOMMU domain at the
/// IOVA window `[iova, iova+bytes)`, zero-copy over the same backing.
/// The device is named by an arena slot (`device`) — a *second* capability
/// the op reads, footprinted at prepare (a read edge, **never a parent**:
/// the leaf is single-parent under the lease).  Referenced from
/// [`Invocation::arg`].  LE byteorder (`align(1)`).
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct DeviceShareReq {
    /// Arena slot of the `PciDevice` capability (its IOMMU domain is the map
    /// target) — footprinted at prepare, not a derive parent.
    pub device: U32,
    /// Target device address (IOVA) to map the window at.
    pub iova: U64,
    /// Window length in bytes (should equal the lease's size).
    pub bytes: U64,
    /// Access selector: [`Self::ACCESS_READ`], [`Self::ACCESS_READ_WRITE`], or
    /// [`Self::ACCESS_WRITE`]. Values outside that frozen vocabulary are
    /// refused.
    pub access: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 3],
}

impl DeviceShareReq {
    /// The device may read the mapped window.
    pub const ACCESS_READ: u8 = 0;
    /// The device may read and write the mapped window.
    pub const ACCESS_READ_WRITE: u8 = 1;
    /// The device may issue posted writes, but may not read the mapped window.
    pub const ACCESS_WRITE: u8 = 2;

    /// Wire size (repr(C), 4+8+8+1+3).
    pub const WIRE: usize = 24;
}

/// Argument payload for a **guest-memory share** derive
/// (`Derive × MemLease → GuestRegion`, `kind = GuestRegion`): install the
/// lease's contiguous backing at `[gpa, gpa+bytes)` in the guest space named
/// by `space`. The guest space is a second authenticated capability and a
/// prepare-time read footprint, never a second derive parent.
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct GuestShareReq {
    /// Arena slot of the target `GuestSpace`, required to carry `MODIFY`.
    pub space: U32,
    /// Guest-physical base address.
    pub gpa: U64,
    /// Window length in bytes; must equal the lease's contiguous span.
    pub bytes: U64,
    /// Explicit access bits ([`SHARE_ACCESS_R`] / W / X).
    pub access: u8,
    /// Frozen reserved-zero tail.
    pub reserved: [u8; 3],
}

impl GuestShareReq {
    /// Wire size (`space@0 ‖ gpa@4 ‖ bytes@12 ‖ access@20 ‖ zero@21`).
    pub const WIRE: usize = 24;
}

/// Argument payload for [`DERIVE_KIND_PEER_IOVA`].
///
/// The invocation subject is the claimed `MmioRegion` that remains the
/// child's sole lineage parent. `device` names the requester whose IOMMU
/// domain receives the leaf; `sponsor` names a live `DeviceMmu` facet whose
/// transient witness is consumed by this install. Both secondary slots are
/// footprinted at prepare. The physical backing is exactly
/// `[BAR.start + offset, BAR.start + offset + bytes)`, never the implicit
/// whole BAR. LE byte order (`align(1)`).
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct PeerDeviceShareReq {
    /// Arena slot of the requester `PciDevice` capability (`MODIFY`).
    pub device: U32,
    /// Arena slot of the sponsoring `DeviceMmu` capability (`DERIVE`).
    pub sponsor: U32,
    /// Target device address in the requester's IOMMU domain.
    pub iova: U64,
    /// Byte offset into the subject BAR.
    pub offset: U64,
    /// Exact aperture length.
    pub bytes: U64,
    /// Frozen access vocabulary from [`DeviceShareReq`].
    pub access: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 7],
}

impl PeerDeviceShareReq {
    pub const ACCESS_READ: u8 = DeviceShareReq::ACCESS_READ;
    pub const ACCESS_READ_WRITE: u8 = DeviceShareReq::ACCESS_READ_WRITE;
    pub const ACCESS_WRITE: u8 = DeviceShareReq::ACCESS_WRITE;

    /// Wire size (repr(C), 4+4+8+8+8+1+7).
    pub const WIRE: usize = 40;
}

/// The [`Invocation::kind`] selector for the **paged unshare** flavor of
/// [`OpCode::Revoke`] (the lessor's eviction verb — installs the paged-out marker):
/// return the leased view like a plain revoke, but leave **paged-out
/// markers** where its leaves were, so the tenant's next touch routes to
/// the space's registered pager instead of faulting hard.  `kind = 0`
/// stays the plain authority revoke.
pub const REVOKE_KIND_PAGED: u8 = 1;

/// Argument payload for an **owned-view unshare** ([`OpCode::Revoke`]): the
/// subject is the view being returned and `lease` names the arena slot of its
/// immediate owning parent — a *second* capability the op reads (the
/// [`DeviceShareReq::device`] idiom), footprinted at prepare, whose share
/// bookkeeping the committed teardown updates. CPU exact/paged unshare names
/// a `MemLease`; [`REVOKE_KIND_UNSHARE_IOVA`] names an `MmioRegion`.
/// Referenced from [`Invocation::arg`]. LE byteorder (`align(1)`).
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct UnshareReq {
    /// Arena slot of the immediate parent. CPU views name `MemLease`; peer
    /// IOVA views name `MmioRegion`. Paged/return and peer-IOMMU unshare gate
    /// `MODIFY`; CPU exact hard unshare uses it only as identity/routing proof.
    pub lease: U32,
    /// Reserved; must be zero.
    pub reserved: [u8; 4],
}

impl UnshareReq {
    /// Wire size (repr(C), 4+4).
    pub const WIRE: usize = 8;
}

/// The [`Invocation::kind`] selector for the **unmap** flavor of
/// [`OpCode::Revoke`] (res/IMAGE.md §9a, the #83 return-leg): revoke the
/// subject region like a plain revoke, but clear its leaves and yield the
/// **bare backing extent** ([`PhysExtent`](crate::cap::ResourceKind::PhysExtent))
/// back into the invocation's `result` slot — the consume-op's return leg
/// as a session verb (`Mem::unmap`).  No argument payload: the subject
/// names the region, `result` the landing slot.
pub const REVOKE_KIND_UNMAP: u8 = 2;

/// The [`Invocation::kind`] selector for the **grant return** flavor of
/// [`OpCode::Revoke`] (the lessor's take-back — mm's `Release`): the
/// subject is a held `LeasedRegion` view delivered back into the lessor's
/// arena; the argument payload is an [`UnshareReq`] naming the owning
/// lease — the same second-cap, `MODIFY`-gated idiom as
/// [`REVOKE_KIND_PAGED`].  The view's tag retires like a plain revoke,
/// and every CPU leaf installed over the view's physical window — by any
/// receiver of the grant, in any space — is unmapped and forgotten (the
/// window-scoped sibling of the lease's full clear), so the lessor
/// re-pools a provably unaliased range.  The invocation's `result` slot
/// receives a bind-inert [`GeomReport`] of the window's
/// `(physical base, len)` — kernel-vouched geometry, because a delivered
/// record carries no offset/len and a client-echoed range would let a
/// tenant poison the lessor's free list.
///
/// The subject is deliberately **not** `REVOKE`-gated: a grant's rights
/// are `ACCESS|MODIFY` (teardown stays with the derivation chain), so the
/// authority is the named lease's `MODIFY` — the paged flavor's basis.
/// This is the hard sibling of [`REVOKE_KIND_PAGED`]: hard unmap, no
/// markers, plus the report.
pub const REVOKE_KIND_RETURN: u8 = 3;

/// The [`Invocation::kind`] selector for **exact hard unshare** through
/// [`OpCode::Revoke`]: retire one held
/// [`LeasedRegion`](crate::cap::ResourceKind::LeasedRegion), remove its exact
/// `(space, virtual range)` share record from the immediate parent lease, and
/// clear that mapping only when it still names the view's original physical
/// window.  Unlike [`REVOKE_KIND_RETURN`], this never clears sibling mappings
/// and yields no geometry report.  The argument is an [`UnshareReq`] naming
/// the view's immediate-parent `MemLease`; that second capability is an
/// identity/routing witness and does **not** require `MODIFY`.
///
/// This append-only selector is the ordinary early-close verb for a mapped
/// lease view.  A stale view whose record is already gone (or whose VA has
/// since been reused) retires without touching the replacement mapping.
pub const REVOKE_KIND_UNSHARE: u8 = 4;

/// The [`Invocation::kind`] selector for the consuming **writable lease →
/// executable lease** transition through [`OpCode::Revoke`].
///
/// The subject is a held or same-session `MemLease` carrying the complete
/// lifecycle grant `ACCESS|MODIFY|DERIVE|REVOKE`. Commit consumes that
/// capability and its complete
/// descendant lineage, removes every CPU/DMA/guest alias, performs the
/// target's instruction-cache publication sequence, and writes a fresh
/// executable-only `MemLease` into `result`.  The successor carries
/// `ACCESS|DERIVE|REVOKE`; it has no `MODIFY`, every WRITE share is refused,
/// and the source name is cleared after its tag retires.
///
/// This is intentionally lease-gated rather than VA-gated: the lease is the
/// capability-backed physical authority that can prove all aliases closed.
pub const REVOKE_KIND_EXEC: u8 = 5;

/// The [`Invocation::kind`] selector for exact **peer IOVA unshare** through
/// [`OpCode::Revoke`]. The subject is a held
/// [`IovaRegion`](crate::cap::ResourceKind::IovaRegion) derived directly from
/// the `MmioRegion` named by [`UnshareReq::lease`]. Commit retires the view and
/// removes only its generational live-window record; sibling peer mappings and
/// every CPU mapping of the BAR remain intact.
///
/// The parent is `MODIFY`-gated because its device-share list is mutated. The
/// subject still requires `REVOKE`, and the direct-parent check prevents a
/// caller from using some other writable BAR as a teardown oracle.
pub const REVOKE_KIND_UNSHARE_IOVA: u8 = 6;

/// The [`Invocation::kind`] selector for the **executable → writable
/// return** through [`OpCode::Revoke`] — the reverse of
/// [`REVOKE_KIND_EXEC`], and the other half of a reusable JIT arena
/// (`res/IMAGE.md` §6a).
///
/// The subject is a held executable `MemLease` that *this* discipline
/// produced: only a window that reached `X` through consuming finalization
/// may return. An X-minted [`LEASE_EXEC`] lease is executable forever, so
/// immutable text keeps the guarantee its own documentation gives it, and
/// the refusal is a lease property rather than a rights check a caller
/// could shop around for.
///
/// Commit retires the source tag, closes the executable lease-owner
/// lineage, reaps every CPU/DMA/guest alias, and issues one fresh sibling
/// lease carrying `ACCESS|MODIFY|DERIVE|REVOKE` — `MODIFY` returns,
/// because the window is writable again.
///
/// This is **not** an `mprotect` analogue and does not weaken W^X: it is
/// the same replacement discipline read backwards. No mapping is ever
/// changed in place, and no single physical window is writable and
/// executable at the same instant, through any alias. What it removes is
/// only the need to *abandon* an arena to rewrite it.
pub const REVOKE_KIND_EXEC_RETURN: u8 = 7;

/// Argument payload for a **page delegation** derive (`Derive`,
/// [`kind`](Invocation::kind)` = Region` on a held/staged `Region`
/// subject; res/IMAGE.md §9a): re-home the subject region
/// into the address space named by arena slot `dest_space` at
/// `[va, va+bytes)` — *the pages stay put, only page-table ownership
/// moves*.  The source region revokes (its tag retires with the commit)
/// and a fresh `Region` derived under the destination space lands in the
/// invocation's `result` slot, stamped to the **invoking** task (the
/// spawner holds the delegated region until it hands the spawnee its boot
/// block).  `dest_space` is a *second* capability the op reads (the
/// [`DeviceShareReq::device`] idiom), footprinted at prepare and
/// pre-flighted with the full destination grant
/// (`DERIVE+ACCESS+MODIFY+REVOKE` — `Mem::delegate`'s gating).
/// `write`/`access` select the destination leaf flags exactly as
/// [`ShareReq`]'s do.  LE byteorder (`align(1)`).
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct DelegateReq {
    /// Arena slot of the destination `AddrSpace` capability.
    pub dest_space: U32,
    /// Virtual start of the destination window (page-aligned).
    pub va: U64,
    /// Window length in bytes (the region's mapped size).
    pub bytes: U64,
    /// Legacy access byte (`0` = RO, non-zero = RW); consulted only when
    /// `access == 0` — the [`ShareReq::write`] contract.
    pub write: u8,
    /// Explicit access bits ([`SHARE_ACCESS_R`]`/W/X`); `0` = legacy.
    pub access: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 2],
}

impl DelegateReq {
    /// Wire size (repr(C), 4+8+8+1+1+2).
    pub const WIRE: usize = 24;
}

/// Argument payload for an **explicit-destination share** derive
/// (`Derive × MemLease → LeasedRegion`,
/// [`kind`](Invocation::kind)` = AddrSpace`; res/IMAGE.md §9b): install
/// the subject lease's buffer into the address space named by arena slot
/// `dest_space` — the [`ShareReq`] sibling whose target is *not* the
/// caller's own space.  `dest_space` is a *second* capability the op
/// reads (the [`DeviceShareReq::device`] idiom, its kind naming the
/// second cap as [`PagerReq`]'s `ChanRecv` does), footprinted at prepare
/// and bound with `MODIFY` held by the invoker (the install mutates its
/// page tables — `Mem::share_into`'s gate).
///
/// This is the **spawn-session** verb: the parent stages the child's
/// windows while it still holds the child space capability; once spawn
/// consumes that capability into the founded task, the parent can no
/// longer name the space and the authority window closes structurally.
/// LE byteorder (`align(1)`).
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct ShareToReq {
    /// Arena slot of the destination `AddrSpace` capability (MODIFY-gated).
    pub dest_space: U32,
    /// Virtual start of the window in the destination space (page-aligned).
    pub va: U64,
    /// Window length in bytes (should equal the lease's size).
    pub bytes: U64,
    /// Legacy access byte (`0` = RO, non-zero = RW); consulted only when
    /// `access == 0` — the [`ShareReq::write`] contract.
    pub write: u8,
    /// Explicit access bits ([`SHARE_ACCESS_R`]`/W/X`); `0` = legacy.
    pub access: u8,
    /// **Consume the result name into a same-session `Found`** (non-zero):
    /// when this exact destination space is successfully founded later in
    /// the descriptor, the install's page-table leaf and typed lease-list
    /// edge move with the retained `SessionChild::Leased` twin into the
    /// destination task and the caller's temporary arena name is cleared.
    ///
    /// This is the same-session carve+map `consume_slot` mechanism (#336
    /// rung 2) offered to the explicit-destination share, and it is what
    /// makes the doc comment above literally true: the parent's install
    /// authority "closes structurally" at the commit that founds the
    /// child, rather than lingering as an arena name the parent must hold
    /// for the child's whole life.  A spawner that shares N segments no
    /// longer pays N retained slots per live child. If no successful `Found`
    /// consumes that destination, the result name remains live with the
    /// caller; the kernel never creates an unnamed mapping owner.
    ///
    /// Zero keeps the v1 posture (the name is retained), so every
    /// existing caller is unchanged.  Note this consumes the *name*, not
    /// the capability: revoking any founder-side ancestor still kills the
    /// mapping (T6), exactly as before.
    pub consume: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 1],
}

impl ShareToReq {
    /// Wire size (repr(C), 4+8+8+1+1+1+1).
    pub const WIRE: usize = 24;
}

/// One founding **delivery** ([`FoundReq::deliver`], descriptor ABI v3 —
/// U1): place the session child staged into founder result slot `from`
/// into the founded child's arena at slot `to`.  Delivery is *placement*
/// of the already-derived cap (KRAPH §12.1 spawn — the parent defines the
/// child's initial cross-section as part of the founding cluster): no
/// re-derivation, no fresh tag — the record is re-sealed to the child's
/// holder (D1) and the kernel twin moves into the child's holdings, so
/// revoking any founder-side ancestor still kills the delivered cap (T6).
/// LE byteorder (`align(1)`).
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, KnownLayout, Immutable, Unaligned)]
pub struct DeliverEntry {
    /// Founder result slot of the staged child to deliver (an SSA name —
    /// the slot a same-session derive assigned), or
    /// [`DELIVER_FROM_SELF_TASK`].
    pub from: U16,
    /// Destination slot in the **child's** arena (`< FoundReq::arena_slots`).
    pub to: U16,
}

/// [`DeliverEntry::from`] sentinel: deliver the founded task's **own**
/// [`Task`](crate::cap::ResourceKind::Task) handle, at `ACCESS|MODIFY`,
/// into the child's arena slot `to` — the SELF handle a task presents to
/// [`DERIVE_KIND_TASK_EXIT`].
///
/// A sentinel rather than an ordinary map entry because the handle does
/// not exist while the session is staged: it is minted from the child's
/// own root *after* `admit`, at the same post-commit assembly step that
/// mints the founder's, so no SSA name can refer to it.  The child's copy
/// deliberately withholds the founder's `REVOKE`: a task may end itself
/// but never reap itself out from under a joiner, so the status it leaves
/// is always still there to be read.  (Waiting on its own end is refused
/// at the park — a self-join is a shape error, not a right.)
///
/// At most one entry per founding may carry it, and it is the one `from`
/// exempt from the "names a slot this session assigned" rule; `to` obeys
/// every other delivery-map rule unchanged.
pub const DELIVER_FROM_SELF_TASK: u16 = u16::MAX;

/// [`DeliverEntry::to`] sentinel: move the delivered capability into the
/// founded task's private holdings without publishing an arena record.
pub const DELIVER_TO_HOLDINGS: u16 = u16::MAX;

/// Marker for a founding delivery whose source is a capability already held
/// by the founder rather than a child staged earlier in the same session.
///
/// The low 15 bits name the founder's arena slot.  Founding moves the exact
/// affine capability (and its existing rights/tag) into the child; it does
/// not derive a second lineage.  This is the ownership-preserving handoff for
/// delegatable roots such as a worker supervisor's memory supply.  Arena slot
/// counts are bounded far below this marker by the kernel's arena growth
/// ceiling (8192 slots), which the kernel compile-pins against both this
/// marker and the sentinel alias below.
pub const DELIVER_FROM_HELD: u16 = 1 << 15;

/// Encode a founder-held arena slot for [`DeliverEntry::from`].
///
/// Slot `0x7FFF` is refused even though it fits the 15-bit field: its
/// encoding is bit-identical to [`DELIVER_FROM_SELF_TASK`], and a sentinel
/// must never be mintable as a real name.
#[inline]
pub const fn deliver_from_held(slot: u16) -> Option<u16> {
    if slot < DELIVER_FROM_HELD && (DELIVER_FROM_HELD | slot) != DELIVER_FROM_SELF_TASK {
        Some(DELIVER_FROM_HELD | slot)
    } else {
        None
    }
}

// The sentinel alias stays unreachable from the encoder, and every encodable
// slot survives the decoder round-trip — pinned here so a widened arena or a
// re-valued sentinel fails the build instead of minting a self-task handle.
const _: () = {
    assert!(deliver_from_held(0x7FFF).is_none());
    assert!(
        matches!(deliver_from_held(0x7FFE), Some(from) if matches!(held_delivery_slot(from), Some(0x7FFE)))
    );
    assert!(held_delivery_slot(DELIVER_FROM_SELF_TASK).is_none());
};

/// Decode a founder-held delivery source, excluding the self-task sentinel.
#[inline]
pub const fn held_delivery_slot(from: u16) -> Option<u16> {
    if from != DELIVER_FROM_SELF_TASK && from & DELIVER_FROM_HELD != 0 {
        Some(from & !DELIVER_FROM_HELD)
    } else {
        None
    }
}

/// Delivery-map capacity of a [`FoundReq`] — a pinned ABI constant,
/// bounded like a session (`res/SYSCALL.md` D4 — never a stream).
///
/// Raising it **moves the wire** — the map is the tail of a `FoundReq`,
/// so both sides of the pair must cite this one constant and the size
/// pin below must move with it (the #81 discipline: deliberate, in
/// lockstep, never by drift). The current maximal capability-selected process
/// consumes 78 deliveries with its explicit task-local namespace, four
/// generic invocation auxiliaries, and cleanup custody. The 78-row wire is
/// the exact bounded ceiling and keeps the request well inside `DESCRIPTOR_MAX`,
/// the shape paired with `DESCRIPTOR_MAX = 4096`.
///
/// This count and [`MAX_BOOT_HANDLES`](crate::task::MAX_BOOT_HANDLES)
/// deliberately do not dominate one another.  A boot entry may merely
/// publish a vacant landing name (`DOORBELL@1`, `SVC_RETURN@1`) and cost
/// no delivery, while a contiguous grant bank is published by its base
/// handle but still costs one delivery per channel end.
pub const MAX_FOUND_DELIVERIES: usize = 78;

/// [`InvocationWire::result`] flag selecting atomic publication of a founded
/// task handle through a held `ChanSend` instead of into a local arena slot.
/// Remaining bits name that held slot; the frozen Found payload is unchanged.
pub const FOUND_RETURN_DIRECT: u32 = 1 << 31;

/// Decode a direct-return channel from a Found invocation result word.
pub const fn found_return_slot(result: u32) -> Option<u32> {
    if result & FOUND_RETURN_DIRECT != 0 {
        Some(result & !FOUND_RETURN_DIRECT)
    } else {
        None
    }
}

/// Argument payload for the **`Found`** terminal verb (`Derive`,
/// [`kind`](Invocation::kind)` = Task` on a **staged** `Space` child;
/// res/IMAGE.md §9f): terminate the spawn session by founding the subject
/// space as a complete but dormant process (a `SHARE_SPACE` thread is the
/// single-session exception and is dispatched immediately).  The subject names the session's staged child
/// space (the fresh-`AddrSpace` mint, §9e — or a thread view for
/// [`SHARE_SPACE`](crate::task::SpawnFlags::SHARE_SPACE)); the frozen
/// [`SpawnReq`](crate::task::SpawnReq) rides whole, plus the **per-task
/// arena slot count** (#207: the wire is what carries the sizing — the
/// founded task's cap arena is built `with_slots(arena_slots)`,
/// `1..=ARENA_SLOTS`), plus the **delivery map** (descriptor ABI v3 — U1):
/// `deliver_count` live [`DeliverEntry`]s naming which staged children the
/// founding places into the child's arena, and where.  `Found`s form the
/// session's terminal **founding run** (the G5 Found-tail rule): once one
/// is staged only further `Found`s may follow — a gang founds N workers as
/// N tail entries in one trap, and a spawn session still *ends* in
/// founding (D4, relaxed from the original at-most-one).  Each Found
/// names its own staged view (one view founds exactly once); the founder's task **handle**
/// ([`Task`](crate::cap::ResourceKind::Task), `ACCESS|MODIFY|DERIVE|REVOKE`) lands
/// in the invocation's `result` slot after the commit's writeback — or stays
/// vacant for a [`DETACHED`](crate::task::SpawnFlags::DETACHED) spawn.  Its
/// `DERIVE` edge lets the founder attenuate an independently revocable
/// `ACCESS|MODIFY|REVOKE` client handle without surrendering the retained
/// parent; `MODIFY` is the explicit START authority.
/// LE byteorder (`align(1)`).
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, KnownLayout, Immutable, Unaligned)]
pub struct FoundReq {
    /// The frozen spawn parameters (`entry`, `stack`, `bootstrap`, `flags`).
    pub spawn: crate::task::SpawnReq,
    /// Cap-arena slots the founded task gets (`1..=ARENA_SLOTS` — #207).
    pub arena_slots: U32,
    /// Live entries in [`deliver`](Self::deliver)
    /// (`<= MAX_FOUND_DELIVERIES`); entries past the count must be zero.
    pub deliver_count: U16,
    /// **The founder's exec-authority slot** (U3b CP-4, res/IMAGE.md §9f):
    /// the caller-arena slot ([`SlotRef`], U16-encoded — a slot is
    /// `< ARENA_SLOTS`, the `DeliverEntry` encoding) holding the
    /// [`ExecAuthority`](crate::cap::ResourceKind::ExecAuthority) capability
    /// that authorizes this `Found`.  The kernel's prepare fold binds it as
    /// `kind == ExecAuthority` for the caller **before Δ** (I3): possession
    /// of the founding authority is what admits a `Found`, closing the
    /// latent "any `VirtMemory` holder can found a task" over-authority.  (Took
    /// the former reserved-zero headroom — no layout shift of its own.)
    pub authority: U16,
    /// The delivery map (U1): staged-child placements into the child's
    /// arena, executed by the spawn assembly after the arena is built.
    pub deliver: [DeliverEntry; MAX_FOUND_DELIVERIES],
}

impl FoundReq {
    /// Wire size (repr(C), 24+4+2+2+78·4).
    pub const WIRE: usize = 344;
}

/// Argument payload for a **pager registration** derive (`Derive ×
/// AddrSpace`, [`kind`](Invocation::kind)` = ChanRecv`): bind the channel
/// whose recv end lives in arena slot `chan_recv` as the **managing
/// pager** of the caller's address space (one pager
/// per space; `MODIFY` on the space authorizes the binding, and the
/// kernel rings this channel's standing recv post on every fault against
/// a paged-out page).  The recv slot is a *second* capability the op
/// reads (the [`DeviceShareReq::device`] idiom), footprinted at prepare.
/// Mints no child — the result slot is left vacant.  Referenced from
/// [`Invocation::arg`].  LE byteorder (`align(1)`).
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct PagerReq {
    /// Arena slot of the `ChanRecv` capability naming the doorbell channel.
    pub chan_recv: U32,
    /// Reserved; must be zero.
    pub reserved: [u8; 4],
}

impl PagerReq {
    /// Wire size (repr(C), 4+4).
    pub const WIRE: usize = 8;
}

/// Which machine-wide transition a [`PowerReq`] asks for.
///
/// Derives [`TryFromBytes`] so the discriminant check is generated from the
/// variant list — the [`OpCode`] discipline: an out-of-range byte is a checked
/// refusal, never UB, and a new variant cannot desync from a hand-maintained
/// decoder `match`.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub enum PowerTransition {
    /// Power the machine off (ACPI S5 on x86).
    PowerOff = 0,
    /// Reset the machine.
    Reboot = 1,
    /// Reversibly freeze scheduler activity and enter suspend-to-idle.
    ///
    /// Unlike [`PowerOff`](Self::PowerOff) and [`Reboot`](Self::Reboot), this
    /// transition is non-terminal: a wake interrupt resumes the existing
    /// machine and its userland tasks.
    SuspendToIdle = 2,
    /// Service-selected lid policy: hold until the observed lid opens.
    SuspendWhileLidClosed = 3,
}

/// Argument payload for a **system power** derive (`Derive × SystemPower`,
/// #392): request the machine transition named by `transition`.
///
/// `MODIFY` on the `SystemPower` subject authorizes it — the same gate the
/// kernel-side `SystemPowerControl` projection enforces, so power-off is a
/// facet of a held capability rather than ambient authority. An `ACCESS`-only
/// holder can read the backend report and is refused here.
///
/// **Mints no child** — the result slot is left vacant (the `PagerReq`
/// idiom), because a transition is an *effect*, not a derivation.
///
/// **The commit does not perform the transition.** Dispatch checks authority
/// and admission synchronously, so an unauthorized, unsupported, or
/// conflicting request refuses with the runtime fully active; it then
/// *latches* an accepted transition for the run-loop bottom half. That context
/// owns either graceful teardown plus the terminal firmware trigger
/// (`PowerOff` / `Reboot`), or the reversible freeze → idle → resume
/// attempt (`SuspendToIdle`). A session commit is the wrong context for
/// either operation: the requester is current inside the session path. So a
/// successful commit means only "accepted". For suspend-to-idle, the request
/// may reply before preparation begins; preparation can still abort. If idle
/// is entered, wake is observed out of band when the existing machine resumes.
///
/// Referenced from [`Invocation::arg`]. LE byteorder (`align(1)`).
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct PowerReq {
    /// The requested transition; validated by [`PowerTransition`]'s generated
    /// discriminant check when the descriptor is read.
    pub transition: PowerTransition,
    /// Reserved; must be zero.
    pub reserved: [u8; 7],
}

impl PowerReq {
    /// Wire size (repr(C), 1+7).
    pub const WIRE: usize = 8;
}

/// One read-only machine metric exposed through `SystemObserve`.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub enum SystemMetric {
    /// Monotonic timestamp in nanoseconds.
    Clock = 1,
    /// Aggregate cumulative CPU busy nanoseconds; auxiliary = CPU count.
    Cpu = 2,
    /// Usable physical bytes; auxiliary = currently free bytes.
    PhysicalMemory = 3,
    /// Kernel-heap capacity; auxiliary = live payload bytes.
    KernelHeap = 4,
    /// Conservative largest single kernel-heap free run.
    KernelHeapLargest = 5,
    /// Cumulative IOMMU translation-fault records drained since boot;
    /// auxiliary = the LAST record, packed (`[15:0]` source-id,
    /// `[23:16]` raw reason, `[24]` write, `[63:25]` faulting-IOVA
    /// page).  Sampling DRAINS any records still latched in hardware —
    /// the IOMMU's own testimony that a device DMA'd and translation
    /// refused it, and a serial-less machine's only view of it.
    IommuFaults = 6,
    /// Live memory-fabric domain registrations (res/MEMORY_FABRIC.md, #697);
    /// value packs domain count in the low word and registry capacity in the
    /// high word. Auxiliary packs registered extent KiB in `[31:0]`, domains
    /// with at least one lendable window in `[47:32]`, and domains requiring a
    /// producer/consumer visibility fence in `[63:48]`. This remains a
    /// data-only census: it does not attest capability liveness or an
    /// executable device route.
    Fabric = 7,
    /// Manifold barrier exponents (MANIFOLD §6.3, §14.4): value packs
    /// `α_cpu × 10⁶` in the low word and `α_mem × 10⁶` in the high word;
    /// auxiliary carries the detail flags the snapshot renders — which α were
    /// *measured* rather than taken from theory, and whether node 0's joint
    /// pressure is finite.  That last bit is the §6.1 guard's witness on a
    /// machine with no serial: the clamp exists so the barrier stays finite.
    ManifoldBarrier = 8,
    /// Manifold node field (§5): value packs the busiest domain slot's
    /// `Re(Z_cpu) × 10⁶` in the low word and `Re(Z_mem) × 10⁶` in the high
    /// word; auxiliary is the worst slot's joint pressure `𝒫 × 1000`,
    /// **saturating** — pressure spans tens
    /// of orders of magnitude near the wall, so the flag above, not this
    /// number, is what says the barrier held.
    ManifoldField = 9,
    /// Manifold volatility census (§3.3, §4): value packs the number of tasks
    /// the last sweep saw carrying `Im ≠ 0` in the low word and the number it
    /// sampled in the high word; auxiliary is the highest per-task Markov
    /// transition rate `× 10⁶`.  Zero volatile tasks on a busy machine means
    /// the behavioural chain is not feeding the estimator.
    ManifoldVolatility = 10,
    /// PCI functions the boot-frozen catalog enumerated (every segment and
    /// root, bridges included); auxiliary = root complexes described.  The
    /// bound on [`DeviceRow`](Self::DeviceRow)'s operand.
    DeviceCount = 11,
    /// One catalog function as a bind-inert census row: the operand
    /// ([`SystemObserveReq::operand`]) is the catalog index, and `value` /
    /// `aux` are [`DeviceCensusRow::to_observation`](crate::device::DeviceCensusRow::to_observation)
    /// with that index echoed.  An operand at or past the count refuses.
    /// Descriptions only — no configuration-space access, no authority.
    DeviceRow = 12,
    /// Per-CPU identity, selected by catalog ordinal.
    ProcessorIdentity = 13,
    /// Per-CPU topology and capacity/feedback metadata.
    ProcessorTopology = 14,
    /// Per-CPU cache geometry.
    ProcessorCache = 15,
    /// Per-CPU cumulative busy time and static frequency envelope.
    ProcessorRuntime = 16,
    /// Portable suspend status encoded as SleepStatus::to_observation().
    /// Snapshot only: this does not authorize a transition.
    Sleep = 17,
    /// Last lid level and close-edge generation; see LidSnapshot.
    Lid = 18,
    /// Scheduler-timer hardware arms since boot, summed over every CPU;
    /// auxiliary packs `[63:32]` arms that needed a second try (a
    /// TSC-deadline write that read back as zero before its instant and was
    /// written again, or a one-shot count that rounded to zero and was
    /// clamped to one tick), `[31:0]` arms refused after the second deadline
    /// write also read zero (the CPU then wakes only on other interrupts
    /// until an arm lands).  Zero auxiliary on a machine that shows
    /// deadline-shaped stalls acquits the timer; a growing one convicts it.
    /// Architectures without a counted arm report zeros.
    TimerArms = 19,
    /// One physical frame-size class of the boot partition
    /// (`res/ACCEL_MEMORY_SCALE.md` §1): the operand is the class
    /// (0 = 4 KiB base, 1 = 2 MiB, 2 = 1 GiB), and the pair is
    /// [`MemoryClassCensus::to_observation`] — bytes carved into the class,
    /// bytes still free in it, and requests the class could not serve from
    /// its own ledger.  An operand past 2 refuses.
    MemoryClass = 20,
    /// The boot-time DRAM stream measurement (`res/RESIDENCY.md` §8 row
    /// 1), three views by operand range: `0..MEMORY_STREAM_CLASS_BASE` one
    /// node per index in measurement order (0 is the first node measured,
    /// which on a UMA machine is the only one; `slot` = node slot, HMAT
    /// beside it); `MEMORY_STREAM_CLASS_BASE + class` one CPU of that core
    /// class alone (`slot` = the CPU that ran it); `MEMORY_STREAM_AGGREGATE`
    /// every CPU at once (`slot` = how many).
    /// [`MemoryStreamSample::to_observation`].  An operand with no sample
    /// answers all-zero rather than refusing, so a reader can walk a range.
    MemoryStream = 21,
    /// One CPU's boot DRAM stream, alone — operand = the CPU's ordinal in
    /// the catalog, like the other `Processor*` metrics.  `value` = builtin
    /// fill rate, bytes per second; `aux` = copy rate in `[39:0]` and the
    /// scrub fill in 100 MB/s units in `[63:40]`; all zero when that CPU
    /// was not measured (no run to lend, or it never answered).
    ProcessorStream = 22,
}

/// The three physical frame-size classes, as the census operand names them.
pub const MEMORY_CLASS_COUNT: u16 = 3;
/// [`SystemMetric::MemoryStream`] operands at and above this name a core
/// class (`CoreClass` discriminant 0..=3) measured alone.
pub const MEMORY_STREAM_CLASS_BASE: u16 = 64;
/// Core classes the stream metric distinguishes.
pub const MEMORY_STREAM_CLASSES: u16 = 4;
/// The [`SystemMetric::MemoryStream`] operand of the all-CPUs measurement.
pub const MEMORY_STREAM_AGGREGATE: u16 = MEMORY_STREAM_CLASS_BASE + MEMORY_STREAM_CLASSES;

/// Authority-free record of one boot DRAM stream measurement
/// ([`SystemMetric::MemoryStream`]).  A measured rate, never a link ceiling.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryStreamSample {
    /// The Manifold node slot the run was lent from (node view), the CPU
    /// that ran it (class view), or the CPU count (aggregate view).
    pub slot: u8,
    /// Fill rate through the builtin fill, bytes per second.
    pub write_bps: u64,
    /// Copy rate, bytes copied per second.
    pub copy_bps: u64,
    /// HMAT's self-access bandwidth for the node, bytes per second, or `0`.
    pub hmat_bps: u64,
    /// Fill rate through the kernel's scrub fill (streaming stores where
    /// the ISA has them), bytes per second.
    pub streaming_bps: u64,
}

impl MemoryStreamSample {
    /// `value` = builtin fill rate in `[39:0]` (1.1 TB/s), the scrub fill in
    /// 100 MB/s units in `[55:40]` (6.5 TB/s), the slot in `[63:56]`;
    /// `aux` = copy rate in `[39:0]` and HMAT in MB/s in `[63:40]`.
    pub const fn to_observation(self) -> (u64, u64) {
        const RATE: u64 = (1 << 40) - 1;
        let write = if self.write_bps > RATE {
            RATE
        } else {
            self.write_bps
        };
        let copy = if self.copy_bps > RATE {
            RATE
        } else {
            self.copy_bps
        };
        let streaming = self.streaming_bps / 100_000_000;
        let streaming = if streaming > u16::MAX as u64 {
            u16::MAX as u64
        } else {
            streaming
        };
        let hmat = self.hmat_bps / 1_000_000;
        let hmat = if hmat > (1 << 24) - 1 {
            (1 << 24) - 1
        } else {
            hmat
        };
        (
            write | streaming << 40 | (self.slot as u64) << 56,
            copy | hmat << 40,
        )
    }

    pub const fn from_observation(value: u64, aux: u64) -> Self {
        Self {
            slot: (value >> 56) as u8,
            write_bps: value & ((1 << 40) - 1),
            copy_bps: aux & ((1 << 40) - 1),
            hmat_bps: (aux >> 40) * 1_000_000,
            streaming_bps: ((value >> 40) & 0xFFFF) * 100_000_000,
        }
    }

    /// Whether a measurement is present at all.
    pub const fn measured(self) -> bool {
        self.write_bps != 0
    }
}

/// Authority-free census of one frame-size class
/// ([`SystemMetric::MemoryClass`]).  Capacity facts only; it grants nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryClassCensus {
    /// Bytes carved into the class at boot (free or not).
    pub carved: u64,
    /// Bytes free in the class now.
    pub free: u64,
    /// Requests of this class served from a finer ledger because the class
    /// was exhausted — the reservation-miss count the 40/30/30 ruling is
    /// judged by.
    pub misses: u64,
}

impl MemoryClassCensus {
    /// `value` = carved bytes; `aux` = free bytes in `[47:0]` (256 TiB) and
    /// misses saturated to 16 bits in `[63:48]`.
    pub const fn to_observation(self) -> (u64, u64) {
        let free = if self.free > (1 << 48) - 1 {
            (1 << 48) - 1
        } else {
            self.free
        };
        let misses = if self.misses > u16::MAX as u64 {
            u16::MAX as u64
        } else {
            self.misses
        };
        (self.carved, free | misses << 48)
    }

    pub const fn from_observation(value: u64, aux: u64) -> Self {
        Self {
            carved: value,
            free: aux & ((1 << 48) - 1),
            misses: aux >> 48,
        }
    }
}

/// Authority-free aggregate returned by [`SystemMetric::Fabric`].
///
/// This type is the single wire contract between the kernel registry census
/// and the monitor service.  It describes registered planning facts only; it
/// does not prove that a capability remains live or that a device route can
/// execute.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FabricCensus {
    pub domains: u32,
    pub capacity: u32,
    pub extent_bytes: u64,
    pub lendable_domains: u32,
    pub fence_required_domains: u32,
}

/// Number of bytes represented by one extent unit in `SystemObserve`.
const FABRIC_EXTENT_UNIT_BYTES: u64 = 1 << 10;

impl FabricCensus {
    /// Encode into the generic observation pair used by `SystemObserve`.
    /// Extent is represented in KiB and auxiliary domain counts saturate at
    /// their fixed-width wire fields.
    pub const fn to_observation(self) -> (u64, u64) {
        let extent_kib = self.extent_bytes / FABRIC_EXTENT_UNIT_BYTES;
        let extent_kib = if extent_kib > u32::MAX as u64 {
            u32::MAX
        } else {
            extent_kib as u32
        };
        let lendable = if self.lendable_domains > u16::MAX as u32 {
            u16::MAX
        } else {
            self.lendable_domains as u16
        };
        let fences = if self.fence_required_domains > u16::MAX as u32 {
            u16::MAX
        } else {
            self.fence_required_domains as u16
        };
        (
            self.domains as u64 | ((self.capacity as u64) << 32),
            extent_kib as u64 | ((lendable as u64) << 32) | ((fences as u64) << 48),
        )
    }

    /// Decode and validate one fabric observation.
    pub const fn from_observation(value: u64, aux: u64) -> Option<Self> {
        let out = Self {
            domains: value as u32,
            capacity: (value >> 32) as u32,
            extent_bytes: (aux as u32 as u64) * FABRIC_EXTENT_UNIT_BYTES,
            lendable_domains: ((aux >> 32) as u16) as u32,
            fence_required_domains: ((aux >> 48) as u16) as u32,
        };
        if (out.capacity != 0 && out.domains > out.capacity)
            || out.lendable_domains > out.domains
            || out.fence_required_domains > out.domains
        {
            None
        } else {
            Some(out)
        }
    }
}

/// Read-only lid mechanism data. Generation advances only on a runtime close
/// edge; a machine discovered with its lid closed starts at generation zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LidSnapshot {
    pub closed: Option<bool>,
    pub generation: u32,
}

impl LidSnapshot {
    pub const fn to_observation(self) -> (u64, u64) {
        (
            match self.closed {
                None => 0,
                Some(false) => 1,
                Some(true) => 2,
            },
            self.generation as u64,
        )
    }

    pub fn from_observation(value: u64, aux: u64) -> Option<Self> {
        let closed = match value {
            0 => None,
            1 => Some(false),
            2 => Some(true),
            _ => return None,
        };
        let generation = u32::try_from(aux).ok()?;
        if closed.is_none() && generation != 0 {
            return None;
        }
        Some(Self { closed, generation })
    }
}

/// End-to-end suspend availability, including scheduler, clock and wake route.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub enum SleepSupport {
    Unsupported = 0,
    Unavailable = 1,
    Ready = 2,
}

/// Latest machine-wide attempt. Accepted is admission, not completed sleep.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub enum SleepPhase {
    Idle = 0,
    Accepted = 1,
    Preparing = 2,
    Quiesced = 3,
    Resumed = 4,
    Aborted = 5,
    Failed = 6,
    /// Scheduler is live; the display owner must still restore panel control.
    Restoring = 7,
}

/// Latest attempt's result; no claim of device retention or deep residency.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub enum SleepFailure {
    None = 0,
    Busy = 1,
    Unsupported = 2,
    Park = 3,
    Resume = 4,
    Mechanism = 5,
    /// Display preparation or exact restoration did not complete.
    Display = 6,
}

/// Portable status data. Snapshot of the latest attempt, not a completion
/// ticket for a particular caller. Hardware residency is deliberately absent.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct SleepStatus {
    pub support: SleepSupport,
    pub phase: SleepPhase,
    pub failure: SleepFailure,
}

impl SleepStatus {
    pub const fn encode(self) -> [u8; 3] {
        [self.support as u8, self.phase as u8, self.failure as u8]
    }

    pub fn decode(bytes: [u8; 3]) -> Option<Self> {
        Self::try_read_from_bytes(&bytes).ok()
    }

    pub const fn to_observation(self) -> (u64, u64) {
        (
            self.support as u64 | (self.phase as u64) << 8 | (self.failure as u64) << 16,
            0,
        )
    }

    pub fn from_observation(value: u64, aux: u64) -> Option<Self> {
        if value >> 24 != 0 || aux != 0 {
            return None;
        }
        Self::decode([value as u8, (value >> 8) as u8, (value >> 16) as u8])
    }
}

/// Argument for [`DERIVE_KIND_SYSTEM_OBSERVE`].
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct SystemObserveReq {
    /// Metric to sample.
    pub metric: SystemMetric,
    /// `[0..2]`: the little-endian operand of an indexed metric
    /// (device census or processor status); zero for aggregate metrics.
    /// `[2..]`: reserved, must be zero.
    pub reserved: [u8; 7],
}

impl SystemObserveReq {
    /// Wire size (`metric: u8 + reserved: [u8; 7]`).
    pub const WIRE: usize = 8;

    /// The indexed metrics' operand (`reserved[0..2]`, little-endian).
    pub const fn operand(&self) -> u16 {
        u16::from_le_bytes([self.reserved[0], self.reserved[1]])
    }

    /// Whether `metric` takes an operand at all.
    pub const fn takes_operand(metric: SystemMetric) -> bool {
        matches!(
            metric,
            SystemMetric::DeviceRow
                | SystemMetric::ProcessorIdentity
                | SystemMetric::ProcessorTopology
                | SystemMetric::ProcessorCache
                | SystemMetric::ProcessorRuntime
                | SystemMetric::MemoryClass
                | SystemMetric::MemoryStream
                | SystemMetric::ProcessorStream
        )
    }

    /// The argument bytes for `metric` with `operand` — the one encoder,
    /// so the kernel's shape check and the verb never disagree.
    pub const fn encode(metric: SystemMetric, operand: u16) -> [u8; Self::WIRE] {
        let [lo, hi] = operand.to_le_bytes();
        [metric as u8, lo, hi, 0, 0, 0, 0, 0]
    }
}

/// Receiver-authored policy for a rendezvous **capability post**
/// (`Rendezvous × ChanRecv`).  When present, the board accepts only a
/// capability whose kind is exactly `kind` and whose delivered rights contain
/// `minimum_rights`; notices and other kinds leave the post standing and the
/// sender is refused.  An absent argument preserves the original untyped-post
/// behavior.
///
/// `kind` is the append-only [`ResourceKind`](crate::cap::ResourceKind)
/// discriminant.  The kernel validates both it and the rights bits before the
/// post can be staged.  LE byteorder (`align(1)`), the `*Req`
/// argument-region discipline.
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, KnownLayout, Immutable, Unaligned)]
pub struct RecvPolicyReq {
    /// Exact capability kind accepted by this post.
    pub kind: U16,
    /// Minimum rights the delivered capability must carry.
    pub minimum_rights: u8,
    /// Reserved; must be zero.
    pub reserved: [u8; 5],
}

impl RecvPolicyReq {
    /// Wire size (repr(C), 2+1+5).
    pub const WIRE: usize = 8;
}

/// Argument payload for the **notice-mode send** (USERLAND.md §3.4.1;
/// SYSCALL.md Move C's arg-gated `Rendezvous` flavor): a `ChanSend`
/// invocation whose `arg` names one of these delivers a **bind-inert**
/// [`Notice`] — `cookie` + `payload` — into the peer's posted slot.  No
/// `resolve_slot`, no `DERIVE`, no tag mint; enablement is `ACCESS` on
/// the send end plus a live posted recv (`WouldBlock`/`PeerRevoked`
/// refusals unchanged, whole-session).  The doorbell mux's kick verb
/// (§3.1).  LE byteorder (`align(1)`), the `*Req` argument-region
/// discipline.
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, KnownLayout, Immutable, Unaligned)]
pub struct NoticeReq {
    /// The wake-source cookie (doorbell notice cookie / ring id /
    /// timer id — USERLAND.md §3.3): a *hint* naming which source kicked;
    /// the receiver's drain is total, so a coalesced-away cookie loses
    /// nothing.
    pub cookie: U64,
    /// One opaque payload word riding beside the cookie (source-defined).
    pub payload: U64,
}

impl NoticeReq {
    /// Wire size (repr(C), 8+8).
    pub const WIRE: usize = 16;
}

/// Compact [`OpCode::Rendezvous`] flavor for the service request-ring kick.
///
/// The ordinary notice form carries an arbitrary `(cookie, payload)` in a
/// 16-byte [`NoticeReq`]. A service request kick has one frozen meaning and
/// no payload, so repeating those bytes would make `kick + ParkReq` 56 bytes
/// and spill past x86-64's six-word inline carrier. This append-only flavor
/// encodes the same bind-inert notice with `arg = NO_ARG`, empty requested
/// rights, and a dead zero result. Header + invocation + `ParkReq` is then
/// 40 bytes on every architecture.
pub const RENDEZVOUS_KIND_REQUEST_KICK: u8 = 0xff;

/// Exact affine capability move through a rendezvous send.
///
/// Unlike the ordinary derive-shaped send, this form preserves the held
/// capability's kind, rights, lineage, and tag while changing its holder. The
/// wire must request zero rights; the kernel authenticates the source record
/// and supports only held affine leaf species. It exists for handoff of a
/// deliberately non-delegable capability, such as a structural cwd binding.
pub const RENDEZVOUS_KIND_EXACT_MOVE: u8 = 0xfe;

/// Cookie delivered by [`RENDEZVOUS_KIND_REQUEST_KICK`]. The receiver's
/// service drain is total; this remains a hint, not request identity.
pub const NOTICE_COOKIE_SERVICE_REQUEST: u64 = 0x6B72_5F73_7663_5251; // "kr_svcRQ"

/// # The bind-inert report family
///
/// (One family, one name — "notice family" and "report family" were used
/// interchangeably before; the ordinal-numbered member docs are gone with
/// them, since two members had already claimed "fifth".)
///
/// A **report** is data the kernel answers or delivers without minting a
/// capability: it rides the frozen 128-byte
/// [`CapRecord`](crate::cap::CapRecord) with
/// `cid = `[`Cid::NULL`](crate::cap::Cid::NULL) — architecturally
/// **vacant**, so a session naming the slot as a subject reads `Vacant`,
/// never a forged authority (I4/I6 intact) — a `kind` discriminant
/// deliberately **outside** the [`ResourceKind`](crate::cap::ResourceKind)
/// space ([`REPORT_KINDS`] proves the roster pairwise distinct), a magic
/// in `parent` distinguishing a deliberate report from a scrubbed slot,
/// and its payload in the record's inert columns.  Kernel encodes,
/// userland decodes — each member's one impl holds both halves, so a
/// layout cannot drift, and the per-member decoders are load-bearing:
/// several members share one result slot across consecutive sessions,
/// and cross-type decode *rejection* is what keeps a stale sibling from
/// reading as fresh.
///
/// Members: [`Notice`] (the doorbell — delivery-slot classifier),
/// [`GeomReport`], [`DisplayReport`], [`ExitReport`], [`DeviceReport`],
/// [`DeviceCountReport`].
///
/// The retired first member, `PagerNotice`, folded into [`Notice`] under
/// [`NOTICE_COOKIE_PAGER_FAULT`]: it was byte-for-byte `Notice`'s shape
/// under a private kind, and that private kind is exactly why the
/// userland delivery classifier (`take_delivery`) dropped every pager
/// fault notice as "vacant/scrubbed" — the fold is what made the fault
/// wake *consumable*.
///
/// The **doorbell notice** a notice-mode send delivers (USERLAND.md
/// §3.1/§3.4.1) — a caller-supplied cookie + payload: `cookie` names the
/// wake source ("look here"), `payload` rides one opaque word beside it.
/// Level-triggered and coalescing by design — the state that matters is
/// in the rings, the notice only says "look".
///
/// The kernel itself speaks it too: the pager's **fault notice** is a
/// `Notice` with [`NOTICE_COOKIE_PAGER_FAULT`] and the faulting VA as
/// its payload, written into the lessor's posted slot when a tenant
/// faults on a paged-out page.
///
/// The encoding rides the frozen 128-byte record bind-inert (the family
/// contract above): `cid = NULL`, `kind = `[`NOTICE_KIND`], the magic in
/// `parent`, `cookie` in `scope_rid`, and `payload` in the (equally
/// inert) `holder` word.  Kernel encodes (the notice-mode dispatch's
/// slot write), userland decodes — one impl, no drift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Notice {
    /// The wake-source cookie ([`NoticeReq::cookie`]).
    pub cookie: u64,
    /// The opaque payload word ([`NoticeReq::payload`]).
    pub payload: u64,
}

/// The `kind` discriminant of an encoded [`Notice`] — outside the
/// [`ResourceKind`](crate::cap::ResourceKind) space (a notice is data,
/// not a resource).  `0xFF01` stays retired with the folded
/// `PagerNotice` so a stale encoder can never read as a doorbell.
pub const NOTICE_KIND: u16 = 0xFF02;

/// The `parent` magic of an encoded [`Notice`].
pub const NOTICE_MAGIC: u64 = 0x6B6F_7265_4E54_4345; // "koreNTCE"

/// The reserved [`Notice::cookie`] of the kernel pager's **fault
/// notice**: `payload` carries the faulting VA (any address within the
/// page).  Reserved here, beside the type, because the kernel is the
/// producer — the `kr_svc*` cookie run in `kore-rt` is service-protocol
/// vocabulary and this is not.
///
/// **Kernel-only, and enforced.**  Since the fault notice is an ordinary
/// [`Notice`] (the fold that made it visible to a lessor's normal
/// delivery drain), this cookie is the *only* thing separating "the
/// kernel observed a fault at this VA" from "a peer holding the pager
/// channel's send end said so" — a distinction a private `kind` used to
/// carry for free.  So the session engine **refuses** a userland
/// notice-mode send that names it (`PrepareFault::Denied`, before
/// anything commits), which is what keeps a decoded fault notice
/// trustworthy.
pub const NOTICE_COOKIE_PAGER_FAULT: u64 = 0x6B6F_7265_4641_4C54; // "koreFALT"

/// The reserved [`Notice::cookie`] of the kernel's **terminal quiesce
/// ask** (#809 §L): on `shutdown` / `reboot`, before the scheduler drain
/// stops user dispatch, the kernel posts this notice into each
/// device-owning service row's registered quiesce channel; `payload`
/// carries the row's remaining budget in nanoseconds.  The row answers by
/// stopping its class engine and driving [`DERIVE_KIND_DEVICE_QUIESCE`] on
/// each device it holds — completion is observed by the kernel at the
/// device's DMA epoch, never by a reply.
///
/// **Kernel-only, and enforced** exactly as [`NOTICE_COOKIE_PAGER_FAULT`]:
/// the session engine refuses a userland notice-mode send that names it,
/// so a decoded quiesce ask is trustworthy — a peer holding the channel's
/// send end cannot make a driver believe the machine is going down.
pub const NOTICE_COOKIE_POWER_QUIESCE: u64 = 0x6B6F_7265_5153_4345; // "koreQSCE"

// Two kernel-reserved cookies, two distinct words — the refusal gate keys
// on equality, so a collision would silently merge two protocols.
const _: () = assert!(NOTICE_COOKIE_PAGER_FAULT != NOTICE_COOKIE_POWER_QUIESCE);

/// **The family frame** — a vacant record stamped with one member's kind
/// and magic, and nothing else.
///
/// Every member's `encode` starts here and then writes only its own
/// payload columns.  Factored because this half is the *contract* (what
/// makes a report bind-inert and identifiable) while the payload half is
/// genuinely per-member — several members pack bit layouts that must stay
/// readable at their own site, which is why this is two small functions
/// rather than a macro that would generate the marshalling too.
const fn inert_frame(kind: u16, magic: u64) -> crate::cap::CapRecord {
    let mut rec = crate::cap::CapRecord::vacant();
    rec.parent = crate::cap::Cid(magic);
    rec.kind = kind;
    rec
}

/// **The family gate** — is `rec` a deliberate report of this member?
///
/// All three clauses are load-bearing, which is exactly why they are
/// written once: `cid == NULL` is what keeps a real capability from ever
/// decoding as data (I4/I6), the `kind` is the member's identity, and the
/// `parent` magic separates a deliberate report from a scrubbed slot.  A
/// member that forgot one of these would be a silent authority bug, and
/// the family already shipped one kind collision that nothing guarded.
const fn is_inert_report(rec: &crate::cap::CapRecord, kind: u16, magic: u64) -> bool {
    rec.cid.0 == crate::cap::Cid::NULL.0 && rec.kind == kind && rec.parent.0 == magic
}

impl Notice {
    /// Encode as a bind-inert [`CapRecord`](crate::cap::CapRecord)
    /// (kernel side — the notice-mode dispatch's slot write).
    pub fn encode(self) -> crate::cap::CapRecord {
        let mut rec = inert_frame(NOTICE_KIND, NOTICE_MAGIC);
        rec.scope_rid = self.cookie;
        rec.holder = crate::cap::Cid(self.payload);
        rec
    }

    /// Decode from a delivered arena record (receiver side).  `None` for
    /// anything that is not a deliberate doorbell notice.
    pub fn decode(rec: &crate::cap::CapRecord) -> Option<Self> {
        is_inert_report(rec, NOTICE_KIND, NOTICE_MAGIC).then_some(Self {
            cookie: rec.scope_rid,
            payload: rec.holder.0,
        })
    }
}

/// The **geometry report** a map-flavor derive answers with (USERLAND.md
/// §5.6; #336 rung 2) — the notice family's third member, [`Notice`]'s
/// shape carrying an address-space fact instead of a doorbell: the kernel
/// installed a window into the **caller's own** space and reports *where*.
/// The caller's result slot receives this instead of a minted capability —
/// the map's authority is the *subject* (the bound `LeasedRegion` view);
/// the report is data about the caller's own address space, not a grant.
///
/// The encoding rides the frozen 128-byte record bind-inert (the family
/// contract above): `cid = NULL`, `kind = `[`GEOM_REPORT_KIND`], the magic
/// in `parent`, `base` in `scope_rid`, `len` in the (equally inert)
/// `holder` word.  Kernel encodes (the map arm's result record), userland
/// decodes ([`map_view`]) — one impl, no drift.
///
/// [`map_view`]: https://docs.rs/kore-rt — `kore_rt::cap::verbs::map_view`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeomReport {
    /// Base VA of the reported window, in the caller's own space.
    pub base: u64,
    /// Byte length of the reported window.
    pub len: u64,
}

/// The `kind` discriminant of an encoded [`GeomReport`] — outside the
/// [`ResourceKind`](crate::cap::ResourceKind) space, beside
/// [`NOTICE_KIND`] (a report is data, not a resource).
pub const GEOM_REPORT_KIND: u16 = 0xFF03;

/// The `parent` magic of an encoded [`GeomReport`].
pub const GEOM_REPORT_MAGIC: u64 = 0x6B6F_7265_4745_4F4D; // "koreGEOM"

impl GeomReport {
    /// Encode as a bind-inert [`CapRecord`](crate::cap::CapRecord)
    /// (kernel side — the map arm's result-slot write).
    pub fn encode(self) -> crate::cap::CapRecord {
        let mut rec = inert_frame(GEOM_REPORT_KIND, GEOM_REPORT_MAGIC);
        rec.scope_rid = self.base;
        rec.holder = crate::cap::Cid(self.len);
        rec
    }

    /// Decode from the caller's result slot (client side).  `None` for
    /// anything that is not a deliberate geometry report.
    pub fn decode(rec: &crate::cap::CapRecord) -> Option<Self> {
        is_inert_report(rec, GEOM_REPORT_KIND, GEOM_REPORT_MAGIC).then_some(Self {
            base: rec.scope_rid,
            len: rec.holder.0,
        })
    }
}

/// The **fabric quiesce report** a [`DERIVE_KIND_DEVICE_QUIESCE`] effect
/// answers with (#809 §L) — the notice family's member carrying the kernel's
/// verdict on one device's DMA path: after the rendezvous, can the function
/// still reach memory through the fabric?  The caller's result slot receives
/// this instead of a minted capability; the authority is the *subject*
/// (`MODIFY` on the device), the report is data about hardware state.
///
/// Bind-inert per the family contract: `cid = NULL`, `kind` =
/// [`QUIESCE_REPORT_KIND`], the magic in `parent`.  `scope_rid` packs
/// `rid | verdict << 32 | flags << 40`; the `holder` word is reserved zero
/// for the device-MMU generation #725 introduces (a `Quiesced` bound to a
/// sponsor epoch changes the payload, never the shape).  Kernel encodes,
/// userland decodes — one impl, no drift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuiesceReport {
    /// The device's catalog resource id (what the requester registry and
    /// the power witness lines name the function by).
    pub rid: u32,
    /// [`QUIESCE_VERDICT_PROVEN`] or [`QUIESCE_VERDICT_CONTAINED`].
    pub verdict: u8,
    /// `QUIESCE_FLAG_*` bits — which halves of the rendezvous closed.
    pub flags: u8,
}

/// Mapping route request. Identities and synchronization are planning inputs,
/// not requester authority or proof of a hardware completion fence. Actual
/// CPU/IOVA children authenticate their destination through the normal mapper.
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct FabricRouteReq {
    pub destination: U64,
    pub offset: U64,
    pub bytes: U64,
    pub source_reset: U32,
    pub source_power: U32,
    pub source_allocation: U32,
    pub destination_reset: U32,
    pub destination_power: U32,
    pub destination_allocation: U32,
    /// READ=1, WRITE=2, EXCLUSIVE=4. At least one data direction is required.
    pub access: u8,
    /// CPU map=0, IOMMU map=1.
    pub operation: u8,
    /// Unfenced=0, caller-declared fenced=1; never hardware attestation.
    pub synchronization: u8,
    pub reserved: [u8; 5],
}

bitflags! {
    /// Route access on the wire (`FabricRouteReq::access`,
    /// `FabricRouteScopeReq::access`): the three bits the kernel's
    /// `runtime::fabric::FabricAccess` decodes.  At least one data direction
    /// is required; `EXCLUSIVE` bars every independently admitted alias.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct FabricAccess: u8 {
        const READ      = 1 << 0;
        const WRITE     = 1 << 1;
        const EXCLUSIVE = 1 << 2;
    }
}

/// The mapping operation a route admits (`FabricRouteReq::operation`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum FabricOp {
    /// A CPU mapping child (`LeasedRegion`).
    CpuMap = 0,
    /// A device mapping child (`LeasedIova`); also admits CPU children.
    IommuMap = 1,
}

/// The caller's declared visibility state (`FabricRouteReq::synchronization`).
/// A declaration, never hardware attestation of a producer fence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum FabricSync {
    Unfenced = 0,
    Fenced = 1,
}

impl FabricRouteReq {
    /// A route between two described domains over `offset..offset+bytes`
    /// of the source, carrying both endpoints' captured generations.  The
    /// twelve-field literal every call site used to spell by hand.
    pub fn between(
        src: &FabricDomainReport,
        dst: &FabricDomainReport,
        offset: u64,
        bytes: u64,
        op: FabricOp,
        access: FabricAccess,
        sync: FabricSync,
    ) -> Self {
        Self {
            destination: dst.domain.into(),
            offset: offset.into(),
            bytes: bytes.into(),
            source_reset: src.reset.into(),
            source_power: src.power.into(),
            source_allocation: src.allocation.into(),
            destination_reset: dst.reset.into(),
            destination_power: dst.power.into(),
            destination_allocation: dst.allocation.into(),
            access: access.bits(),
            operation: op as u8,
            synchronization: sync as u8,
            reserved: [0; 5],
        }
    }

    /// The common case: a route whose source and destination are the same
    /// registered domain (host DRAM mapped into a CPU space or a device
    /// domain), over the first `bytes` of the registration.
    pub fn self_route(
        facts: &FabricDomainReport,
        bytes: u64,
        op: FabricOp,
        access: FabricAccess,
        sync: FabricSync,
    ) -> Self {
        Self::between(facts, facts, 0, bytes, op, access, sync)
    }

    pub fn encode(&self) -> [u8; Self::WIRE] {
        let mut bytes = [0u8; Self::WIRE];
        for (i, value) in [self.destination, self.offset, self.bytes]
            .iter()
            .enumerate()
        {
            bytes[i * 8..i * 8 + 8].copy_from_slice(&value.get().to_le_bytes());
        }
        for (i, value) in [
            self.source_reset,
            self.source_power,
            self.source_allocation,
            self.destination_reset,
            self.destination_power,
            self.destination_allocation,
        ]
        .iter()
        .enumerate()
        {
            bytes[24 + i * 4..28 + i * 4].copy_from_slice(&value.get().to_le_bytes());
        }
        bytes[48] = self.access;
        bytes[49] = self.operation;
        bytes[50] = self.synchronization;
        bytes[51..56].copy_from_slice(&self.reserved);
        bytes
    }

    pub const WIRE: usize = 56;
    pub fn valid(&self) -> bool {
        self.destination.get() != 0
            && self.bytes.get() != 0
            && self.offset.get().checked_add(self.bytes.get()).is_some()
            && self.access & !7 == 0
            && self.access & 3 != 0
            && self.operation <= 1
            && self.synchronization <= 1
            && self.reserved == [0; 5]
    }
}

/// Relative, page-granular delegation beneath a mapping route.
#[repr(C)]
#[derive(Clone, Copy, Debug, TryFromBytes, KnownLayout, Immutable, Unaligned)]
pub struct FabricRouteScopeReq {
    pub offset: U64,
    pub bytes: U64,
    pub access: u8,
    pub reserved: [u8; 7],
}

impl FabricRouteScopeReq {
    /// A page-granular relative range with attenuated access.
    pub const fn new(offset: u64, bytes: u64, access: FabricAccess) -> Self {
        Self {
            offset: U64::new(offset),
            bytes: U64::new(bytes),
            access: access.bits(),
            reserved: [0; 7],
        }
    }

    pub const WIRE: usize = 24;
    pub fn valid(&self) -> bool {
        self.bytes.get() != 0
            && self.offset.get().checked_add(self.bytes.get()).is_some()
            && self.offset.get() % 4096 == 0
            && self.bytes.get() % 4096 == 0
            && self.access & !7 == 0
            && self.access & 3 != 0
            && self.reserved == [0; 7]
    }
    pub fn encode(&self) -> [u8; Self::WIRE] {
        let mut bytes = [0; Self::WIRE];
        bytes[..8].copy_from_slice(&self.offset.get().to_le_bytes());
        bytes[8..16].copy_from_slice(&self.bytes.get().to_le_bytes());
        bytes[16] = self.access;
        bytes[17..].copy_from_slice(&self.reserved);
        bytes
    }
}

/// A route-close result is data, never new authority. Closed means admission
/// is barred and every mapping child has retired; the route still needs revoke.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FabricRouteCloseReport {
    pub route: u64,
    /// 0=closed, 1=busy, 2=quarantined.
    pub state: u8,
}
pub const FABRIC_ROUTE_CLOSE_REPORT_KIND: u16 = 0xFF0F;
pub const FABRIC_ROUTE_CLOSE_REPORT_MAGIC: u64 = 0x6B6F_7265_4652_434C;
impl FabricRouteCloseReport {
    pub fn encode(self) -> crate::cap::CapRecord {
        let mut rec = inert_frame(
            FABRIC_ROUTE_CLOSE_REPORT_KIND,
            FABRIC_ROUTE_CLOSE_REPORT_MAGIC,
        );
        rec.scope_rid = self.route;
        rec.holder = crate::cap::Cid(u64::from(self.state));
        rec
    }
    pub fn decode(rec: &crate::cap::CapRecord) -> Option<Self> {
        if !is_inert_report(
            rec,
            FABRIC_ROUTE_CLOSE_REPORT_KIND,
            FABRIC_ROUTE_CLOSE_REPORT_MAGIC,
        ) || rec.scope_rid == 0
            || rec.holder.0 > 2
            || !rec.rights.is_empty()
            || rec.epoch != 0
            || rec.chain_len != 0
            || rec.chain.iter().any(|word| *word != 0)
            || rec.mac != [0; 16]
        {
            return None;
        }
        Some(Self {
            route: rec.scope_rid,
            state: rec.holder.0 as u8,
        })
    }
}

/// Mapping authority geometry only: no physical address or ownership transfer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FabricRouteInfo {
    pub bytes: u64,
    pub access: u8,
    /// CPU map=0, IOMMU map=1. Other operations have no mapping admission.
    pub operation: u8,
}
pub const FABRIC_ROUTE_INFO_KIND: u16 = 0xFF10;
pub const FABRIC_ROUTE_INFO_MAGIC: u64 = 0x6B6F_7265_4652_494E;
impl FabricRouteInfo {
    pub fn encode(self) -> crate::cap::CapRecord {
        let mut rec = inert_frame(FABRIC_ROUTE_INFO_KIND, FABRIC_ROUTE_INFO_MAGIC);
        rec.scope_rid = self.bytes;
        rec.holder = crate::cap::Cid(u64::from(self.access) | (u64::from(self.operation) << 8));
        rec
    }
    pub fn decode(rec: &crate::cap::CapRecord) -> Option<Self> {
        if !is_inert_report(rec, FABRIC_ROUTE_INFO_KIND, FABRIC_ROUTE_INFO_MAGIC)
            || rec.scope_rid == 0
            || rec.scope_rid % 4096 != 0
            || rec.holder.0 & !0x107 != 0
            || rec.holder.0 & 3 == 0
            || !rec.rights.is_empty()
            || rec.epoch != 0
            || rec.chain_len != 0
            || rec.chain.iter().any(|word| *word != 0)
            || rec.mac != [0; 16]
        {
            return None;
        }
        Some(Self {
            bytes: rec.scope_rid,
            access: rec.holder.0 as u8,
            operation: (rec.holder.0 >> 8) as u8,
        })
    }
}

/// Planning facts from a live host-DRAM registration. Neither the identity nor
/// this report authorizes mapping, routing or transfer. No address is encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FabricDomainReport {
    pub domain: u64,
    pub bytes: u64,
    pub owner: u64,
    pub reset: u32,
    pub power: u32,
    pub allocation: u32,
    /// 0 = write-back, 1 = write-combine, 2 = uncached.
    pub cache: u8,
    pub coherent: bool,
    pub needs_fence: bool,
}

pub const FABRIC_DOMAIN_REPORT_KIND: u16 = 0xFF0E;
pub const FABRIC_DOMAIN_REPORT_MAGIC: u64 = 0x6B6F_7265_4641_4252; // koreFABR

impl FabricDomainReport {
    pub fn encode(self) -> crate::cap::CapRecord {
        let mut rec = inert_frame(FABRIC_DOMAIN_REPORT_KIND, FABRIC_DOMAIN_REPORT_MAGIC);
        rec.scope_rid = self.domain;
        rec.holder = crate::cap::Cid(self.bytes);
        rec.chain[0] = self.owner as u32;
        rec.chain[1] = (self.owner >> 32) as u32;
        rec.chain[2] = self.reset;
        rec.chain[3] = self.power;
        rec.chain[4] = self.allocation;
        rec.chain[5] = u32::from(self.cache)
            | (u32::from(self.coherent) << 8)
            | (u32::from(self.needs_fence) << 9);
        rec
    }

    pub fn decode(rec: &crate::cap::CapRecord) -> Option<Self> {
        if !is_inert_report(rec, FABRIC_DOMAIN_REPORT_KIND, FABRIC_DOMAIN_REPORT_MAGIC)
            || rec.scope_rid == 0
            || rec.holder.0 == 0
            || !rec.rights.is_empty()
            || rec.epoch != 0
            || rec.chain_len != 0
            || rec.chain[6..].iter().any(|word| *word != 0)
            || rec.mac != [0; 16]
            || rec.chain[5] & !0x303 != 0
            || rec.chain[5] & 3 > 2
        {
            return None;
        }
        Some(Self {
            domain: rec.scope_rid,
            bytes: rec.holder.0,
            owner: u64::from(rec.chain[0]) | (u64::from(rec.chain[1]) << 32),
            reset: rec.chain[2],
            power: rec.chain[3],
            allocation: rec.chain[4],
            cache: (rec.chain[5] & 3) as u8,
            coherent: rec.chain[5] & 0x100 != 0,
            needs_fence: rec.chain[5] & 0x200 != 0,
        })
    }
}

/// The `kind` discriminant of an encoded [`QuiesceReport`].
pub const QUIESCE_REPORT_KIND: u16 = 0xFF0D;
/// The `parent` magic of an encoded [`QuiesceReport`].
pub const QUIESCE_REPORT_MAGIC: u64 = 0x6B6F_7265_5155_4945; // "koreQUIE"

/// Bus Master Enable read back clear; the requester is on Abort wherever it
/// has an IOMMU path at all.  The device provably issues no DMA.
pub const QUIESCE_VERDICT_PROVEN: u8 = 0;
/// The function refused to drop Bus Master Enable, but its requester is
/// bound to Abort: the IOMMU is the enforcement point and no DMA reaches
/// memory.  The DMA epoch stays quarantined.
pub const QUIESCE_VERDICT_CONTAINED: u8 = 1;

/// MSI / MSI-X Enable were cleared at the source.
pub const QUIESCE_FLAG_IRQ_MASKED: u8 = 1 << 0;
/// The requester's IOMMU context was rebound to Abort.
pub const QUIESCE_FLAG_ABORT_BOUND: u8 = 1 << 1;
/// No peer window this device sponsored stands (#725): its device-MMU
/// facet was closed and drained before the DMA half — by the rendezvous or
/// already by its holder — or none was ever minted.  Clear only when a
/// facet still stood that the path could not close (a rival closer past its
/// bounded wait, or the post-drain containment, which observes only).
pub const QUIESCE_FLAG_PEERS_DRAINED: u8 = 1 << 2;

impl QuiesceReport {
    /// Encode as a bind-inert [`CapRecord`](crate::cap::CapRecord)
    /// (kernel side — the quiesce effect's result-slot write).
    pub fn encode(self) -> crate::cap::CapRecord {
        let mut rec = inert_frame(QUIESCE_REPORT_KIND, QUIESCE_REPORT_MAGIC);
        rec.scope_rid =
            u64::from(self.rid) | (u64::from(self.verdict) << 32) | (u64::from(self.flags) << 40);
        rec
    }

    /// Decode from the caller's result slot (client side).  `None` for
    /// anything that is not a deliberate quiesce report.
    pub fn decode(rec: &crate::cap::CapRecord) -> Option<Self> {
        let report = Self {
            rid: rec.scope_rid as u32,
            verdict: (rec.scope_rid >> 32) as u8,
            flags: (rec.scope_rid >> 40) as u8,
        };
        (is_inert_report(rec, QUIESCE_REPORT_KIND, QUIESCE_REPORT_MAGIC)
            && matches!(
                report.verdict,
                QUIESCE_VERDICT_PROVEN | QUIESCE_VERDICT_CONTAINED
            )
            && report.flags
                & !(QUIESCE_FLAG_IRQ_MASKED
                    | QUIESCE_FLAG_ABORT_BOUND
                    | QUIESCE_FLAG_PEERS_DRAINED)
                == 0
            && rec.scope_rid >> 48 == 0
            && rec.holder == crate::cap::Cid::NULL
            && rec.rights.is_empty()
            && rec.epoch == 0
            && rec.chain_len == 0
            && rec.chain.iter().all(|&word| word == 0)
            && rec.mac == [0; 16])
            .then_some(report)
    }
}

/// The **display mode report** a display-aperture claim answers with (the
/// display handoff, A.2) — the notice family's fourth member,
/// [`GeomReport`]'s shape carrying the framebuffer's full mode instead of
/// a bare `(base, len)`: the kernel vouches for the aperture's physical
/// base **and** its programmed geometry, because the frozen record
/// deliberately carries neither and a service rendering into the window
/// cannot guess a pitch or a pixel format.
///
/// The encoding rides the frozen 128-byte record bind-inert (the family
/// contract): `cid = `[`Cid::NULL`](crate::cap::Cid::NULL), `kind =
/// `[`DISPLAY_REPORT_KIND`], the magic in `parent`, `base` in `scope_rid`,
/// `pitch | width << 32 | height << 48` packed in the (equally inert)
/// `holder` word, and the format byte in the inert `rights` field.
/// Kernel encodes (the claim flavor's result record), userland decodes —
/// one impl, no drift.  Width and height ride `u16` each (65 535 px is
/// the frozen envelope; a later 65k+ mode needs a new kind, never a
/// reinterpretation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DisplayReport {
    /// Physical base of the aperture.
    pub base: u64,
    /// Bytes per scanline.
    pub pitch: u32,
    /// Width in pixels.
    pub width: u16,
    /// Height in pixels.
    pub height: u16,
    /// Pixel format (`1` = XRGB8888, the v1 layout).
    pub format: u8,
}

/// The `kind` discriminant of an encoded [`DisplayReport`] — outside the
/// [`ResourceKind`](crate::cap::ResourceKind) space, beside
/// [`GEOM_REPORT_KIND`] (a report is data, not a resource).
pub const DISPLAY_REPORT_KIND: u16 = 0xFF04;

/// The `parent` magic of an encoded [`DisplayReport`].
pub const DISPLAY_REPORT_MAGIC: u64 = 0x6B6F_7265_4453_5059; // "koreDSPY"

/// [`DisplayReport::format`] value for 32-bit little-endian RGB.
pub const DISPLAY_FORMAT_XRGB8888: u8 = 1;

impl DisplayReport {
    /// Encode as a bind-inert [`CapRecord`](crate::cap::CapRecord)
    /// (kernel side — the claim flavor's result-slot write).
    pub fn encode(self) -> crate::cap::CapRecord {
        let mut rec = inert_frame(DISPLAY_REPORT_KIND, DISPLAY_REPORT_MAGIC);
        rec.scope_rid = self.base;
        rec.holder = crate::cap::Cid(
            u64::from(self.pitch) | (u64::from(self.width) << 32) | (u64::from(self.height) << 48),
        );
        rec.rights = crate::cap::Rights::from_bits_retain(self.format);
        rec
    }

    /// Decode from the caller's result slot (client side).  `None` for
    /// anything that is not a deliberate display report.
    pub fn decode(rec: &crate::cap::CapRecord) -> Option<Self> {
        let holder = rec.holder.0;
        is_inert_report(rec, DISPLAY_REPORT_KIND, DISPLAY_REPORT_MAGIC).then_some(Self {
            base: rec.scope_rid,
            pitch: holder as u32,
            width: (holder >> 32) as u16,
            height: (holder >> 48) as u16,
            format: rec.rights.bits(),
        })
    }
}

/// The `kind` discriminant of an encoded device-inventory entry, beside
/// [`GEOM_REPORT_KIND`] and [`DISPLAY_REPORT_KIND`].
///
/// `0xFF07`, **not** the `0xFF05` it first shipped as: that value was
/// already [`EXIT_REPORT_KIND`]'s, a collision the decoders' magic gate
/// and the two reports' provably-disjoint landing slots kept latent —
/// and nothing guarded.  The kind IS the member's identity
/// (res/SYSCALL.md pins members by it), so the double-use is broken and
/// the whole family's distinctness is now a const proof below.
pub const DEVICE_REPORT_KIND: u16 = 0xFF07;

/// The `parent` magic of an encoded device-inventory entry.
pub const DEVICE_REPORT_MAGIC: u64 = 0x6B6F_7265_4445_5643; // "koreDEVC"

/// One **device-inventory entry** as the [`DERIVE_KIND_DEVICE_RECORD`]
/// flavor answers it — [`DisplayReport`]'s sibling for hardware the kernel
/// enumerated but has not handed to anybody.
///
/// The encoding rides the frozen 128-byte record bind-inert (the family
/// contract): `cid = `[`Cid::NULL`](crate::cap::Cid::NULL), `kind =
/// `[`DEVICE_REPORT_KIND`], the magic in `parent`, the PCI identity packed
/// into the inert `scope_rid` word, the locator and segment into the
/// equally inert `holder` word, and the species byte in `rights`.  Kernel
/// encodes, userland decodes — one impl, no drift.
///
/// [`DeviceRecord::index`](crate::device::DeviceRecord::index) is **not**
/// on the wire: the caller named the index in its
/// [`InventoryReq`] and fills it back in, so the one field the client
/// already knows does not cost frozen-record space that a future flag byte
/// will want.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceReport {
    /// Species byte — [`DeviceSpecies`](crate::device::DeviceSpecies).
    pub species: u8,
    /// PCI vendor id, or the low half of a platform device's compressed
    /// ACPI id.
    pub vendor: u16,
    /// PCI device id, or the high half of that compressed id.
    pub device: u16,
    /// PCI base class (`0` for a platform device).
    pub class: u8,
    /// PCI subclass (`0` for a platform device).
    pub subclass: u8,
    /// PCI programming interface (`0` for a platform device).
    pub progif: u8,
    /// The `HAS_*` facts of [`DeviceRecord`](crate::device::DeviceRecord).
    pub flags: u8,
    /// Resolved MSI-X table size. Rides the low 16 bits of the bind-inert
    /// report frame's first lineage word; zero means no valid MSI-X table.
    pub msix_table_size: u16,
    /// `bus << 8 | device << 3 | function`, or a platform device's GSI.
    pub locator: u32,
    /// PCI segment group (`0` for a platform device).
    pub segment: u16,
    /// The diagnostic word ([`DeviceRecord::diag`](crate::device::DeviceRecord::diag)),
    /// riding the frame's inert `epoch` field (the family gate checks
    /// only `cid`/`kind`/`parent`, and reports carry no epoch).
    pub diag: u32,
    /// PCI revision id, riding the `holder` word's spare bits [55:48]
    /// (locator + segment use [47:0]).
    pub revision: u8,
}

impl DeviceReport {
    /// Encode as a bind-inert [`CapRecord`](crate::cap::CapRecord)
    /// (kernel side — the record flavor's result-slot write).
    pub fn encode(self) -> crate::cap::CapRecord {
        let mut rec = inert_frame(DEVICE_REPORT_KIND, DEVICE_REPORT_MAGIC);
        rec.scope_rid = u64::from(self.vendor)
            | (u64::from(self.device) << 16)
            | (u64::from(self.class) << 32)
            | (u64::from(self.subclass) << 40)
            | (u64::from(self.progif) << 48)
            | (u64::from(self.flags) << 56);
        rec.holder = crate::cap::Cid(
            u64::from(self.locator)
                | (u64::from(self.segment) << 32)
                | (u64::from(self.revision) << 48),
        );
        rec.rights = crate::cap::Rights::from_bits_retain(self.species);
        rec.epoch = self.diag;
        rec.chain[0] = u32::from(self.msix_table_size);
        rec
    }

    /// Decode from the caller's result slot (client side).  `None` for
    /// anything that is not a deliberate device report.
    pub fn decode(rec: &crate::cap::CapRecord) -> Option<Self> {
        let ident = rec.scope_rid;
        let holder = rec.holder.0;
        (is_inert_report(rec, DEVICE_REPORT_KIND, DEVICE_REPORT_MAGIC)
            && rec.chain[0] <= u32::from(u16::MAX))
        .then_some(Self {
            species: rec.rights.bits(),
            vendor: ident as u16,
            device: (ident >> 16) as u16,
            class: (ident >> 32) as u8,
            subclass: (ident >> 40) as u8,
            progif: (ident >> 48) as u8,
            diag: rec.epoch,
            revision: (holder >> 48) as u8,
            flags: (ident >> 56) as u8,
            msix_table_size: rec.chain[0] as u16,
            locator: holder as u32,
            segment: (holder >> 32) as u16,
        })
    }
}

/// The authoritative number of entries in one inventory subject — a
/// `RootComplex`'s PCI functions or the `PlatformBus`'s platform devices.
///
/// This is the device-report family's bounded terminator: clients fetch it
/// once and then treat every failed record read inside `0..len` as an error.
/// No refused session is therefore overloaded to mean end-of-inventory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceCountReport {
    /// Number of valid inventory indices.
    pub len: u32,
}

/// The `kind` discriminant of an encoded [`DeviceCountReport`].
pub const DEVICE_COUNT_REPORT_KIND: u16 = 0xFF06;

/// The `parent` magic of an encoded [`DeviceCountReport`].
pub const DEVICE_COUNT_REPORT_MAGIC: u64 = 0x6B6F_7265_4443_4E54; // "koreDCNT"

impl DeviceCountReport {
    /// Encode as a bind-inert [`CapRecord`](crate::cap::CapRecord).
    pub fn encode(self) -> crate::cap::CapRecord {
        let mut rec = inert_frame(DEVICE_COUNT_REPORT_KIND, DEVICE_COUNT_REPORT_MAGIC);
        rec.scope_rid = u64::from(self.len);
        rec
    }

    /// Decode an authoritative count report. Non-zero high bits are rejected
    /// rather than truncated into a smaller inventory.
    pub fn decode(rec: &crate::cap::CapRecord) -> Option<Self> {
        (is_inert_report(rec, DEVICE_COUNT_REPORT_KIND, DEVICE_COUNT_REPORT_MAGIC)
            && rec.scope_rid <= u64::from(u32::MAX))
        .then_some(Self {
            len: rec.scope_rid as u32,
        })
    }
}

/// Kernel-published correlation for one root-local PCI claim. The all-zero
/// pair means no committed identity was observed (vacant, pending or changing).
/// Nonzero data is historical observation, never authority or proof of cleanup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceClaimReport {
    pub generation: u64,
    pub grant: crate::cap::Cid,
}

pub const DEVICE_CLAIM_REPORT_KIND: u16 = 0xFF11;
pub const DEVICE_CLAIM_REPORT_MAGIC: u64 = 0x6B6F_7265_4443_4C4D; // koreDCLM

impl DeviceClaimReport {
    pub fn encode(self) -> crate::cap::CapRecord {
        let mut record = inert_frame(DEVICE_CLAIM_REPORT_KIND, DEVICE_CLAIM_REPORT_MAGIC);
        record.scope_rid = self.generation;
        record.holder = self.grant;
        record
    }

    pub fn decode(record: &crate::cap::CapRecord) -> Option<Self> {
        let generation = record.scope_rid;
        let grant = record.holder;
        let valid_pair = (generation == 0 && grant == crate::cap::Cid::NULL)
            || (generation >> 63 == 0
                && generation & 1 != 0
                && grant.0 as u32 != 0
                && grant.0 as u32 != u32::MAX);
        (is_inert_report(record, DEVICE_CLAIM_REPORT_KIND, DEVICE_CLAIM_REPORT_MAGIC)
            && valid_pair
            && record.rights.is_empty()
            && record.epoch == 0
            && record.chain_len == 0
            && record.chain.iter().all(|&word| word == 0)
            && record.mac == [0; 16])
            .then_some(Self { generation, grant })
    }
}

/// Result of exact-grant admission closure. Drained counts only tracked
/// operations and never authorizes physical cleanup or device reuse by itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum DeviceClaimCloseState {
    Mismatch = 0,
    Busy = 1,
    Untracked = 2,
    Pending = 3,
    AdmissionDrained = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceClaimCloseReport {
    pub state: DeviceClaimCloseState,
    pub readers: u32,
}

pub const DEVICE_CLAIM_CLOSE_REPORT_KIND: u16 = 0xFF12;
pub const DEVICE_CLAIM_CLOSE_REPORT_MAGIC: u64 = 0x6B6F_7265_4443_4C53; // koreDCLS

impl DeviceClaimCloseReport {
    pub fn encode(self) -> crate::cap::CapRecord {
        let mut record = inert_frame(
            DEVICE_CLAIM_CLOSE_REPORT_KIND,
            DEVICE_CLAIM_CLOSE_REPORT_MAGIC,
        );
        record.scope_rid = u64::from(self.state as u8) | u64::from(self.readers) << 8;
        record
    }
    pub fn decode(record: &crate::cap::CapRecord) -> Option<Self> {
        if !is_inert_report(
            record,
            DEVICE_CLAIM_CLOSE_REPORT_KIND,
            DEVICE_CLAIM_CLOSE_REPORT_MAGIC,
        ) || record.scope_rid >> 40 != 0
            || record.holder != crate::cap::Cid::NULL
            || !record.rights.is_empty()
            || record.epoch != 0
            || record.chain_len != 0
            || record.chain.iter().any(|&word| word != 0)
            || record.mac != [0; 16]
        {
            return None;
        }
        let state = match record.scope_rid as u8 {
            0 => DeviceClaimCloseState::Mismatch,
            1 => DeviceClaimCloseState::Busy,
            2 => DeviceClaimCloseState::Untracked,
            3 => DeviceClaimCloseState::Pending,
            4 => DeviceClaimCloseState::AdmissionDrained,
            _ => return None,
        };
        let readers = (record.scope_rid >> 8) as u32;
        ((state == DeviceClaimCloseState::Pending) == (readers != 0))
            .then_some(Self { state, readers })
    }
}

/// Bind-inert evidence for one registered-DMA drain attempt. The count is per
/// attempt; even Retired does not release CPU mappings, IRQs or the claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum DeviceClaimDmaState {
    Retired = 0,
    EntryRetained = 1,
    AbortRetained = 2,
    EntryRetirementFailed = 3,
    ContainmentFailed = 4,
    WindowRetirementFailed = 5,
    Unbound = 6,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceClaimDmaReport {
    pub state: DeviceClaimDmaState,
    pub removed: u32,
}

pub const DEVICE_CLAIM_DMA_REPORT_KIND: u16 = 0xFF13;
pub const DEVICE_CLAIM_DMA_REPORT_MAGIC: u64 = 0x6B6F_7265_4444_4D41; // koreDDMA

impl DeviceClaimDmaReport {
    pub fn encode(self) -> crate::cap::CapRecord {
        let mut record = inert_frame(DEVICE_CLAIM_DMA_REPORT_KIND, DEVICE_CLAIM_DMA_REPORT_MAGIC);
        record.scope_rid = u64::from(self.state as u8) | u64::from(self.removed) << 8;
        record
    }
    pub fn decode(record: &crate::cap::CapRecord) -> Option<Self> {
        if !is_inert_report(
            record,
            DEVICE_CLAIM_DMA_REPORT_KIND,
            DEVICE_CLAIM_DMA_REPORT_MAGIC,
        ) || record.scope_rid >> 40 != 0
            || record.holder != crate::cap::Cid::NULL
            || !record.rights.is_empty()
            || record.epoch != 0
            || record.chain_len != 0
            || record.chain.iter().any(|&word| word != 0)
            || record.mac != [0; 16]
        {
            return None;
        }
        let state = match record.scope_rid as u8 {
            0 => DeviceClaimDmaState::Retired,
            1 => DeviceClaimDmaState::EntryRetained,
            2 => DeviceClaimDmaState::AbortRetained,
            3 => DeviceClaimDmaState::EntryRetirementFailed,
            4 => DeviceClaimDmaState::ContainmentFailed,
            5 => DeviceClaimDmaState::WindowRetirementFailed,
            6 => DeviceClaimDmaState::Unbound,
            _ => return None,
        };
        let removed = (record.scope_rid >> 8) as u32;
        if matches!(
            state,
            DeviceClaimDmaState::ContainmentFailed | DeviceClaimDmaState::Unbound
        ) && removed != 0
        {
            return None;
        }
        Some(Self { state, removed })
    }
}

/// An I2C admission page from a held inventory root. Explicitly bind-inert;
/// it cannot be possessed as a controller or used to program the reported GSI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct I2cAdmissionReport {
    pub page: crate::device::I2cAdmissionPage,
}

pub const I2C_ADMISSION_REPORT_KIND: u16 = 0xFF0C;
pub const I2C_ADMISSION_REPORT_MAGIC: u64 = 0x6B6F_7265_4932_4341; // koreI2CA

impl I2cAdmissionReport {
    pub fn encode(self) -> crate::cap::CapRecord {
        let mut record = inert_frame(I2C_ADMISSION_REPORT_KIND, I2C_ADMISSION_REPORT_MAGIC);
        record.scope_rid = u64::from(self.page.row);
        record.holder =
            crate::cap::Cid(u64::from(self.page.words[0]) | u64::from(self.page.words[1]) << 32);
        record
    }

    pub fn decode(record: &crate::cap::CapRecord) -> Option<Self> {
        if !is_inert_report(
            record,
            I2C_ADMISSION_REPORT_KIND,
            I2C_ADMISSION_REPORT_MAGIC,
        ) || record.scope_rid > u64::from(u8::MAX)
            || !record.rights.is_empty()
            || record.epoch != 0
            || record.chain_len != 0
            || record.chain.iter().any(|&tag| tag != 0)
            || record.mac != [0; 16]
        {
            return None;
        }
        let page = crate::device::I2cAdmissionPage {
            row: record.scope_rid as u8,
            words: [record.holder.0 as u32, (record.holder.0 >> 32) as u32],
        };
        page.valid().then_some(Self { page })
    }
}

/// Firmware-provided SCL high/low counts for one I2C speed mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct I2cTimingReport {
    pub hcnt: u16,
    pub lcnt: u16,
}

/// ACPI's three-valued GPIO interrupt polarity.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum I2cGpioPolarity {
    High = 0,
    Low = 1,
    Both = 2,
}

impl I2cGpioPolarity {
    const fn from_bits(bits: u32) -> Option<Self> {
        match bits {
            0 => Some(Self::High),
            1 => Some(Self::Low),
            2 => Some(Self::Both),
            _ => None,
        }
    }
}

/// Kernel-side readiness of one firmware-described I2C-HID child route.
///
/// This is configuration evidence, not authority: it explains whether the
/// immutable boot-time GPIO inventory admitted the pad. The caller must still
/// derive the actual route capability, and that later bind may independently
/// refuse.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum I2cReadyRouteState {
    /// Firmware did not describe a readiness interrupt for this child.
    #[default]
    NotDescribed = 0,
    /// The `GpioInt` source path did not resolve to a discovered community.
    NoCommunity = 1,
    /// The community did not provide the complete dynamic register layout.
    NoLayout = 2,
    /// The community did not provide a usable MMIO aperture.
    NoAperture = 3,
    /// The community did not provide a parent interrupt line.
    NoParentLine = 4,
    /// The firmware pad ordinal falls outside the described group geometry.
    PadOutOfRange = 5,
    /// No catalog GPIO device matched the firmware community resource.
    NoCatalogDevice = 6,
    /// The catalog resource could not provide its owned MMIO view.
    NoRegisterView = 7,
    /// The community interrupt banks could not be safely quiesced.
    QuiesceRefused = 8,
    /// The parent GSI could not be installed in the interrupt fabric.
    ParentRouteRefused = 9,
    /// The source is described and ready for capability derivation. Line
    /// admission still occurs in the pin fabric when deriving the capability.
    Ready = 10,
    /// GPIO setup did not publish a diagnostic for this described child.
    SetupUnavailable = 11,
    /// The firmware-described pad could not be placed in the requested GPIO
    /// input/event mode with a confirmed register readback.
    PadConfigurationRefused = 12,
    /// More than one child named the same physical community pad. Independent
    /// route lifecycles cannot safely share its one enable/status bit.
    PadAliasRefused = 13,
}

impl I2cReadyRouteState {
    pub const fn from_bits(bits: u32) -> Option<Self> {
        match bits {
            0 => Some(Self::NotDescribed),
            1 => Some(Self::NoCommunity),
            2 => Some(Self::NoLayout),
            3 => Some(Self::NoAperture),
            4 => Some(Self::NoParentLine),
            5 => Some(Self::PadOutOfRange),
            6 => Some(Self::NoCatalogDevice),
            7 => Some(Self::NoRegisterView),
            8 => Some(Self::QuiesceRefused),
            9 => Some(Self::ParentRouteRefused),
            10 => Some(Self::Ready),
            11 => Some(Self::SetupUnavailable),
            12 => Some(Self::PadConfigurationRefused),
            13 => Some(Self::PadAliasRefused),
            _ => None,
        }
    }
}

/// One HID-over-I2C child as settled by the kernel's firmware walk.
///
/// This is configuration data only. The controller's claimed `PciDevice`
/// remains the authority for BAR and interrupt derives; this record only says
/// which slave and protocol registers exist behind it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct I2cHidReport {
    pub slave_addr: u8,
    /// Firmware's explicit 16-bit register offset; zero is valid, not absence.
    pub descriptor_register: u16,
    pub speed_hz: u32,
    pub standard_timing: Option<I2cTimingReport>,
    pub fast_timing: Option<I2cTimingReport>,
    /// Controller input clock established by firmware integration data.
    pub input_clock_hz: Option<u32>,
    pub ready_interrupt: Option<I2cReadyInterruptReport>,
    pub ready_route_state: I2cReadyRouteState,
    pub no_irq: bool,
    pub no_power_after_reset: bool,
}

/// Data only: deriving this source still requires the exact controller cap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum I2cReadyInterruptReport {
    Gpio(I2cGpioInterruptReport),
    Line {
        gsi: u32,
        edge: bool,
        active_high: bool,
    },
}

/// One HID child's data-ready GPIO description.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct I2cGpioInterruptReport {
    pub pin: u16,
    pub edge: bool,
    pub polarity: I2cGpioPolarity,
}

/// Bind-inert kind for [`I2cHidReport`].
pub const I2C_HID_REPORT_KIND: u16 = 0xFF09;
/// Report-family magic for [`I2cHidReport`].
pub const I2C_HID_REPORT_MAGIC: u64 = 0x6B6F_7265_4932_4348; // "koreI2CH"

const I2C_STD_TIMING: u8 = 1 << 0;
const I2C_FAST_TIMING: u8 = 1 << 1;
const I2C_GPIO_INTERRUPT: u8 = 1 << 2;
const I2C_GPIO_EDGE: u8 = 1 << 3;
const I2C_QUIRK_NO_IRQ: u8 = 1 << 4;
const I2C_QUIRK_NO_POWER_AFTER_RESET: u8 = 1 << 5;
const I2C_INPUT_CLOCK: u8 = 1 << 6;
const I2C_LINE_INTERRUPT: u8 = 1 << 7;
const I2C_REPORT_FLAGS: u8 = I2C_STD_TIMING
    | I2C_FAST_TIMING
    | I2C_GPIO_INTERRUPT
    | I2C_GPIO_EDGE
    | I2C_QUIRK_NO_IRQ
    | I2C_QUIRK_NO_POWER_AFTER_RESET
    | I2C_INPUT_CLOCK
    | I2C_LINE_INTERRUPT;

impl I2cHidReport {
    /// Encode as bind-inert data. Zero is the absence sentinel for timing
    /// pairs; firmware timing validation already refuses a live zero count.
    pub fn encode(self) -> crate::cap::CapRecord {
        let mut rec = inert_frame(I2C_HID_REPORT_KIND, I2C_HID_REPORT_MAGIC);
        rec.scope_rid = u64::from(self.slave_addr)
            | (u64::from(self.descriptor_register) << 8)
            | (u64::from(self.speed_hz) << 24);

        let standard = self
            .standard_timing
            .unwrap_or(I2cTimingReport { hcnt: 0, lcnt: 0 });
        let fast = self
            .fast_timing
            .unwrap_or(I2cTimingReport { hcnt: 0, lcnt: 0 });
        rec.holder = crate::cap::Cid(
            u64::from(standard.hcnt)
                | (u64::from(standard.lcnt) << 16)
                | (u64::from(fast.hcnt) << 32)
                | (u64::from(fast.lcnt) << 48),
        );

        let mut flags = 0;
        if self.standard_timing.is_some() {
            flags |= I2C_STD_TIMING;
        }
        if self.fast_timing.is_some() {
            flags |= I2C_FAST_TIMING;
        }
        if let Some(interrupt) = self.ready_interrupt {
            let edge = match interrupt {
                I2cReadyInterruptReport::Gpio(gpio) => {
                    flags |= I2C_GPIO_INTERRUPT;
                    rec.epoch = u32::from(gpio.pin);
                    rec.chain[0] = gpio.polarity as u32;
                    gpio.edge
                }
                I2cReadyInterruptReport::Line {
                    gsi,
                    edge,
                    active_high,
                } => {
                    flags |= I2C_LINE_INTERRUPT;
                    rec.epoch = gsi;
                    rec.chain[0] = u32::from(active_high);
                    edge
                }
            };
            if edge {
                flags |= I2C_GPIO_EDGE;
            }
        }
        if self.no_irq {
            flags |= I2C_QUIRK_NO_IRQ;
        }
        if self.no_power_after_reset {
            flags |= I2C_QUIRK_NO_POWER_AFTER_RESET;
        }
        if let Some(hz) = self.input_clock_hz {
            flags |= I2C_INPUT_CLOCK;
            rec.chain[1] = hz;
        }
        rec.chain[2] = self.ready_route_state as u32;
        rec.rights = crate::cap::Rights::from_bits_retain(flags);
        rec
    }

    /// Decode only the canonical record shape. Every absence sentinel is
    /// checked, so stale payload in a supposedly absent field refuses rather
    /// than being silently ignored.
    pub fn decode(rec: &crate::cap::CapRecord) -> Option<Self> {
        if !is_inert_report(rec, I2C_HID_REPORT_KIND, I2C_HID_REPORT_MAGIC) {
            return None;
        }
        let flags = rec.rights.bits();
        if flags & !I2C_REPORT_FLAGS != 0 || rec.scope_rid >> 56 != 0 {
            return None;
        }
        let slave_addr = rec.scope_rid as u8;
        let descriptor_register = (rec.scope_rid >> 8) as u16;
        let speed_hz = (rec.scope_rid >> 24) as u32;
        if slave_addr > 0x7f || !matches!(speed_hz, 100_000 | 400_000) {
            return None;
        }

        let timing = rec.holder.0;
        let standard = I2cTimingReport {
            hcnt: timing as u16,
            lcnt: (timing >> 16) as u16,
        };
        let fast = I2cTimingReport {
            hcnt: (timing >> 32) as u16,
            lcnt: (timing >> 48) as u16,
        };
        let standard_timing = if flags & I2C_STD_TIMING != 0 {
            Some((standard.hcnt != 0 && standard.lcnt != 0).then_some(standard)?)
        } else {
            if standard != (I2cTimingReport { hcnt: 0, lcnt: 0 }) {
                return None;
            }
            None
        };
        let fast_timing = if flags & I2C_FAST_TIMING != 0 {
            Some((fast.hcnt != 0 && fast.lcnt != 0).then_some(fast)?)
        } else {
            if fast != (I2cTimingReport { hcnt: 0, lcnt: 0 }) {
                return None;
            }
            None
        };

        let ready_interrupt = match flags & (I2C_GPIO_INTERRUPT | I2C_LINE_INTERRUPT) {
            I2C_GPIO_INTERRUPT => {
                if rec.epoch > u32::from(u16::MAX) {
                    return None;
                }
                Some(I2cReadyInterruptReport::Gpio(I2cGpioInterruptReport {
                    pin: rec.epoch as u16,
                    edge: flags & I2C_GPIO_EDGE != 0,
                    polarity: I2cGpioPolarity::from_bits(rec.chain[0])?,
                }))
            }
            I2C_LINE_INTERRUPT => {
                if rec.chain[0] > 1 {
                    return None;
                }
                Some(I2cReadyInterruptReport::Line {
                    gsi: rec.epoch,
                    edge: flags & I2C_GPIO_EDGE != 0,
                    active_high: rec.chain[0] != 0,
                })
            }
            0 => {
                if flags & I2C_GPIO_EDGE != 0 || rec.epoch != 0 || rec.chain[0] != 0 {
                    return None;
                }
                None
            }
            _ => return None, // A child cannot name two different source kinds.
        };
        let input_clock_hz = if flags & I2C_INPUT_CLOCK != 0 {
            Some((rec.chain[1] != 0).then_some(rec.chain[1])?)
        } else {
            if rec.chain[1] != 0 {
                return None;
            }
            None
        };
        let ready_route_state = I2cReadyRouteState::from_bits(rec.chain[2])?;
        if ready_interrupt.is_some() == (ready_route_state == I2cReadyRouteState::NotDescribed) {
            return None;
        }
        if rec.chain[3..].iter().any(|&word| word != 0) {
            return None;
        }

        Some(Self {
            slave_addr,
            descriptor_register,
            speed_hz,
            standard_timing,
            fast_timing,
            input_clock_hz,
            ready_interrupt,
            ready_route_state,
            no_irq: flags & I2C_QUIRK_NO_IRQ != 0,
            no_power_after_reset: flags & I2C_QUIRK_NO_POWER_AFTER_RESET != 0,
        })
    }
}

/// Authoritative number of HID-over-I2C children behind one claimed
/// controller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct I2cHidCountReport {
    pub len: u32,
}

pub const I2C_HID_COUNT_REPORT_KIND: u16 = 0xFF0A;
pub const I2C_HID_COUNT_REPORT_MAGIC: u64 = 0x6B6F_7265_4932_434E; // "koreI2CN"

impl I2cHidCountReport {
    pub fn encode(self) -> crate::cap::CapRecord {
        let mut rec = inert_frame(I2C_HID_COUNT_REPORT_KIND, I2C_HID_COUNT_REPORT_MAGIC);
        rec.scope_rid = u64::from(self.len);
        rec
    }

    pub fn decode(rec: &crate::cap::CapRecord) -> Option<Self> {
        (is_inert_report(rec, I2C_HID_COUNT_REPORT_KIND, I2C_HID_COUNT_REPORT_MAGIC)
            && rec.scope_rid <= u64::from(u32::MAX))
        .then_some(Self {
            len: rec.scope_rid as u32,
        })
    }
}

/// One redacted, kernel-validated modern VirtIO PCI vendor-capability row.
///
/// Offsets are relative to the named BAR's user-visible base. They carry no
/// physical address and confer no mapping authority. `cfg_type` is the VirtIO
/// specification value (common=1, notify=2, ISR=3, device=4, PCI=5,
/// shared-memory=8).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VirtioPciCapReport {
    pub cfg_type: u8,
    pub bar: u8,
    pub id: u8,
    pub offset: u32,
    pub length: u32,
    /// Non-zero only for `cfg_type = 2` (notification capability).
    pub notify_multiplier: u32,
}

pub const VIRTIO_PCI_CAP_REPORT_KIND: u16 = 0xFF17;
pub const VIRTIO_PCI_CAP_REPORT_MAGIC: u64 = 0x6B6F_7265_5650_4349; // "koreVPCI"

impl VirtioPciCapReport {
    pub fn encode(self) -> crate::cap::CapRecord {
        let mut rec = inert_frame(VIRTIO_PCI_CAP_REPORT_KIND, VIRTIO_PCI_CAP_REPORT_MAGIC);
        rec.scope_rid = u64::from(self.offset) | (u64::from(self.length) << 32);
        rec.holder = crate::cap::Cid(
            u64::from(self.notify_multiplier)
                | (u64::from(self.cfg_type) << 32)
                | (u64::from(self.bar) << 40)
                | (u64::from(self.id) << 48),
        );
        rec
    }

    pub fn decode(rec: &crate::cap::CapRecord) -> Option<Self> {
        let holder = rec.holder.0;
        if holder >> 56 != 0 {
            return None;
        }
        is_inert_report(rec, VIRTIO_PCI_CAP_REPORT_KIND, VIRTIO_PCI_CAP_REPORT_MAGIC).then_some(
            Self {
                cfg_type: (holder >> 32) as u8,
                bar: (holder >> 40) as u8,
                id: (holder >> 48) as u8,
                offset: rec.scope_rid as u32,
                length: (rec.scope_rid >> 32) as u32,
                notify_multiplier: holder as u32,
            },
        )
    }
}

/// Authoritative number of validated VirtIO PCI rows on the exact held
/// device. A refused in-range record query is always an error, never EOF.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VirtioPciCapCountReport {
    pub len: u8,
}

pub const VIRTIO_PCI_CAP_COUNT_REPORT_KIND: u16 = 0xFF18;
pub const VIRTIO_PCI_CAP_COUNT_REPORT_MAGIC: u64 = 0x6B6F_7265_5650_434E; // "koreVPCN"

impl VirtioPciCapCountReport {
    pub fn encode(self) -> crate::cap::CapRecord {
        let mut rec = inert_frame(
            VIRTIO_PCI_CAP_COUNT_REPORT_KIND,
            VIRTIO_PCI_CAP_COUNT_REPORT_MAGIC,
        );
        rec.scope_rid = u64::from(self.len);
        rec
    }

    pub fn decode(rec: &crate::cap::CapRecord) -> Option<Self> {
        (is_inert_report(
            rec,
            VIRTIO_PCI_CAP_COUNT_REPORT_KIND,
            VIRTIO_PCI_CAP_COUNT_REPORT_MAGIC,
        ) && rec.scope_rid <= u64::from(u8::MAX))
        .then_some(Self {
            len: rec.scope_rid as u8,
        })
    }
}

/// One read-only system observation. The metric owns the interpretation of
/// `value` and `aux`; combining unlike samples belongs to userland's monitor
/// service, not this capability operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SystemObserveReport {
    pub metric: SystemMetric,
    pub value: u64,
    pub aux: u64,
}

/// Bind-inert report kind for [`SystemObserveReport`].
pub const SYSTEM_OBSERVE_REPORT_KIND: u16 = 0xFF08;
/// Report-family magic for [`SystemObserveReport`].
pub const SYSTEM_OBSERVE_REPORT_MAGIC: u64 = 0x6B6F_7265_5359_534F; // "koreSYSO"

impl SystemObserveReport {
    /// Encode as inert data rather than authority.
    pub fn encode(self) -> crate::cap::CapRecord {
        let mut rec = inert_frame(SYSTEM_OBSERVE_REPORT_KIND, SYSTEM_OBSERVE_REPORT_MAGIC);
        rec.scope_rid = self.value;
        rec.holder = crate::cap::Cid(self.aux);
        rec.chain_len = self.metric as u8;
        rec
    }

    /// Decode only the exact report family member and metric vocabulary.
    pub fn decode(rec: &crate::cap::CapRecord) -> Option<Self> {
        if !is_inert_report(rec, SYSTEM_OBSERVE_REPORT_KIND, SYSTEM_OBSERVE_REPORT_MAGIC) {
            return None;
        }
        // Decode through the enum's own validity predicate — a
        // hand-maintained match here silently orphaned the first metric
        // added after it was written (the monitor's whole system sample
        // failed on the new metric's reply, `status` went "provider
        // unavailable", and the exit-code-only harness never noticed).
        let metric = <SystemMetric as zerocopy::TryFromBytes>::try_read_from_bytes(
            core::slice::from_ref(&rec.chain_len),
        )
        .ok()?;
        Some(Self {
            metric,
            value: rec.scope_rid,
            aux: rec.holder.0,
        })
    }
}

/// One authenticated completion of an MM-issued supply lifetime. The kernel
/// emits this only after the complete lease lineage is dead and the physical
/// region has accepted the returned frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SupplyReleaseReport {
    pub receipt: u64,
    pub pair: u16,
    pub len: u64,
}

pub const SUPPLY_RELEASE_REPORT_KIND: u16 = 0xFF0B;
pub const SUPPLY_RELEASE_REPORT_MAGIC: u64 = 0x6B6F_7265_5352_454C; // "koreSREL"

impl SupplyReleaseReport {
    pub fn encode(self) -> crate::cap::CapRecord {
        let mut rec = inert_frame(SUPPLY_RELEASE_REPORT_KIND, SUPPLY_RELEASE_REPORT_MAGIC);
        rec.scope_rid = self.receipt;
        rec.holder = crate::cap::Cid(self.len);
        rec.epoch = u32::from(self.pair);
        rec
    }

    pub fn decode(rec: &crate::cap::CapRecord) -> Option<Self> {
        (is_inert_report(rec, SUPPLY_RELEASE_REPORT_KIND, SUPPLY_RELEASE_REPORT_MAGIC)
            && rec.epoch <= u32::from(u16::MAX))
        .then_some(Self {
            receipt: rec.scope_rid,
            pair: rec.epoch as u16,
            len: rec.holder.0,
        })
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn task_identity_reports_validate_quality_and_never_carry_authority() {
        use crate::task::{TaskId, TaskIdQuality, TaskIdentity};
        for quality in [
            TaskIdQuality::Unverified,
            TaskIdQuality::HardwareOrFirmware,
            TaskIdQuality::TimingJitter,
            TaskIdQuality::Degenerate,
        ] {
            let identity = TaskIdentity {
                id: TaskId::from_bytes([0xa5; 16]),
                quality,
            };
            let rec = identity.encode();
            assert_eq!(TaskIdentity::decode(&rec), Some(identity));
            assert_eq!(rec.cid, crate::cap::Cid::NULL);
            for field in 0..9 {
                let mut bad = rec;
                match field {
                    0 => bad.cid = crate::cap::Cid(1),
                    1 => bad.holder = crate::cap::Cid(1),
                    2 => bad.parent = crate::cap::Cid(0),
                    3 => bad.rights = Rights::ACCESS,
                    4 => bad.epoch = 1,
                    5 => bad.chain_len = 1,
                    6 => bad.chain[0] = 1,
                    7 => bad.scope_rid = 4,
                    _ => bad.scope_rid = 256,
                }
                assert_eq!(TaskIdentity::decode(&bad), None, "field {field}");
            }
        }
    }

    #[test]
    fn claim_dma_report_is_canonical_and_bind_inert() {
        use super::{DeviceClaimDmaReport as Report, DeviceClaimDmaState as State};
        for state in [
            State::Retired,
            State::EntryRetained,
            State::AbortRetained,
            State::EntryRetirementFailed,
            State::ContainmentFailed,
            State::WindowRetirementFailed,
            State::Unbound,
        ] {
            let removed = if matches!(state, State::ContainmentFailed | State::Unbound) {
                0
            } else {
                7
            };
            let report = Report { state, removed };
            assert_eq!(Report::decode(&report.encode()), Some(report));
            for mutate in [
                |r: &mut crate::cap::CapRecord| r.scope_rid |= 1 << 40,
                |r: &mut crate::cap::CapRecord| r.scope_rid = 0xff,
                |r: &mut crate::cap::CapRecord| r.holder = crate::cap::Cid(1),
                |r: &mut crate::cap::CapRecord| r.rights = crate::cap::Rights::ACCESS,
                |r: &mut crate::cap::CapRecord| r.epoch = 1,
                |r: &mut crate::cap::CapRecord| r.chain_len = 1,
                |r: &mut crate::cap::CapRecord| r.chain[0] = 1,
                |r: &mut crate::cap::CapRecord| r.mac[0] = 1,
            ] {
                let mut record = report.encode();
                mutate(&mut record);
                assert_eq!(Report::decode(&record), None);
            }
        }
        for state in [State::ContainmentFailed, State::Unbound] {
            assert_eq!(Report::decode(&Report { state, removed: 1 }.encode()), None);
        }
        assert_eq!(
            Report::decode(
                &super::DeviceClaimCloseReport {
                    state: super::DeviceClaimCloseState::AdmissionDrained,
                    readers: 0
                }
                .encode()
            ),
            None
        );
    }
    use super::*;

    #[test]
    fn claim_close_reports_never_alias_pending_and_drained() {
        use DeviceClaimCloseState::*;
        for state in [Mismatch, Busy, Untracked, Pending, AdmissionDrained] {
            let report = DeviceClaimCloseReport {
                state,
                readers: if state == Pending { 2 } else { 0 },
            };
            assert_eq!(
                DeviceClaimCloseReport::decode(&report.encode()),
                Some(report)
            );
            let mut invalid = report;
            invalid.readers = if state == Pending { 0 } else { 2 };
            assert_eq!(DeviceClaimCloseReport::decode(&invalid.encode()), None);
            let mut frame = report.encode();
            frame.holder = crate::cap::Cid(7);
            assert_eq!(DeviceClaimCloseReport::decode(&frame), None);
        }
        let request = DeviceClaimCloseReq {
            index: 0.into(),
            reserved: 0.into(),
            generation: 1.into(),
            grant: 7.into(),
        };
        assert!(request.valid());
        for (generation, grant, reserved) in [
            (0, 7, 0),
            (2, 7, 0),
            (1, 0, 0),
            (1, u32::MAX as u64, 0),
            (1, 7, 1),
        ] {
            assert!(
                !DeviceClaimCloseReq {
                    generation: generation.into(),
                    grant: grant.into(),
                    reserved: reserved.into(),
                    ..request
                }
                .valid()
            );
        }
    }

    #[test]
    fn device_claim_report_is_canonical_and_bind_inert() {
        use crate::cap::Cid;
        for report in [
            DeviceClaimReport {
                generation: 0,
                grant: Cid::NULL,
            },
            DeviceClaimReport {
                generation: 3,
                grant: Cid(0x1234_0000_0007),
            },
        ] {
            let record = report.encode();
            assert_eq!(record.cid, Cid::NULL);
            assert_eq!(DeviceClaimReport::decode(&record), Some(report));
            for corrupt in 0..8 {
                let mut bad = record;
                match corrupt {
                    0 => bad.cid = Cid(1),
                    1 => bad.kind = DEVICE_REPORT_KIND,
                    2 => bad.parent = Cid(0),
                    3 => bad.rights = crate::cap::Rights::ACCESS,
                    4 => bad.epoch = 1,
                    5 => bad.chain_len = 1,
                    6 => bad.chain[7] = 1,
                    _ => bad.mac[0] = 1,
                }
                assert_eq!(DeviceClaimReport::decode(&bad), None);
            }
        }
        for (generation, grant) in [
            (0, 7),
            (2, 7),
            (1 << 63 | 1, 7),
            (1, 0),
            (1, u32::MAX as u64),
        ] {
            assert_eq!(
                DeviceClaimReport::decode(
                    &DeviceClaimReport {
                        generation,
                        grant: Cid(grant)
                    }
                    .encode()
                ),
                None
            );
        }
    }

    #[test]
    fn line_and_i2c_gpio_flavors_are_known_without_aliasing() {
        assert!(derive_kind_known(DERIVE_KIND_IRQ_ROUTE_LINE));
        assert!(derive_kind_known(DERIVE_KIND_I2C_HID_READY_ROUTE));
        assert_ne!(DERIVE_KIND_IRQ_ROUTE_LINE, DERIVE_KIND_IRQ_ROUTE_MSIX);
        assert_ne!(DERIVE_KIND_IRQ_ROUTE_LINE, DERIVE_KIND_I2C_HID_RECORD);
        assert_ne!(DERIVE_KIND_IRQ_ROUTE_LINE, DERIVE_KIND_I2C_HID_READY_ROUTE);
        // The byte below the GPIO route is the terminal device-quiesce flavor
        // (#809 §L); the one below *that* is the next free selector.
        assert_eq!(
            DERIVE_KIND_I2C_HID_READY_ROUTE - 1,
            DERIVE_KIND_DEVICE_QUIESCE
        );
        assert!(derive_kind_known(DERIVE_KIND_DEVICE_QUIESCE));
        assert!(!derive_kind_known(DERIVE_KIND_DEVICE_QUIESCE - 1));
    }

    /// The tagged register carrier is exact and fail-closed: every admitted
    /// length round-trips, while scratch controls, zero, overflow, and any
    /// reserved control bit cannot be reinterpreted as an inline descriptor.
    #[test]
    fn inline_descriptor_control_round_trips_only_its_frozen_shape() {
        for len in 1..=INLINE_DESCRIPTOR_LENGTH_MASK as usize {
            let control = inline_descriptor_control(len).expect("representable inline length");
            assert_eq!(inline_descriptor_len(control), Some(len));
        }
        assert_eq!(inline_descriptor_control(0), None);
        assert_eq!(inline_descriptor_control(256), None);
        assert_eq!(inline_descriptor_len(0), None, "tag-clear is scratch");
        assert_eq!(inline_descriptor_len(INLINE_DESCRIPTOR_TAG), None);
        assert_eq!(
            inline_descriptor_len(INLINE_DESCRIPTOR_TAG | (1 << 16) | 8),
            None,
            "reserved control bits refuse"
        );
    }

    /// [`Status::encode`] ⇄ [`Committed`] round-trip — the pack/unpack
    /// pair agree for every status and a representative index spread.
    #[test]
    fn committed_round_trips_status_encode() {
        let all = [
            Status::Ok,
            Status::Denied,
            Status::Forged,
            Status::Revoked,
            Status::Conflict,
            Status::Malformed,
            Status::WouldBlock,
            Status::Deadline,
            Status::Irq,
            Status::Exited,
            Status::AltDelivery,
        ];
        for s in all {
            for index in [0u16, 1, 7, 63, u16::MAX] {
                let c = Committed::from_raw(s.encode(index));
                assert_eq!(c.status(), s);
                assert_eq!(c.failing_index(), index);
                assert_eq!(c.raw(), s.encode(index));
            }
        }
    }

    /// An out-of-vocabulary status word decodes fail-closed as
    /// [`Status::Malformed`] — never mistaken for success.
    #[test]
    fn committed_decodes_unknown_status_fail_closed() {
        let c = Committed::from_raw((3u64 << 16) | 0x7777);
        assert_eq!(c.status(), Status::Malformed);
        assert_eq!(c.failing_index(), 3);
    }

    /// The power-transition ABI is append-only and exact: terminal values keep
    /// their frozen encodings, the two reversible modes take 2/3, and the next byte
    /// remains invalid rather than being guessed as a transition.
    #[test]
    fn power_transition_vocabulary_is_exact_and_fail_closed() {
        use zerocopy::TryFromBytes;

        for (raw, expected) in [
            (0, PowerTransition::PowerOff),
            (1, PowerTransition::Reboot),
            (2, PowerTransition::SuspendToIdle),
            (3, PowerTransition::SuspendWhileLidClosed),
        ] {
            let bytes = [raw, 0, 0, 0, 0, 0, 0, 0];
            let (req, rest) = PowerReq::try_ref_from_prefix(&bytes[..])
                .expect("registered power transition decodes");
            assert!(rest.is_empty());
            assert_eq!(req.transition, expected);
        }

        assert!(
            PowerReq::try_ref_from_prefix(&[4, 0, 0, 0, 0, 0, 0, 0][..]).is_err(),
            "the next unregistered transition must fail closed"
        );
        assert_eq!(core::mem::size_of::<PowerReq>(), PowerReq::WIRE);
    }

    /// The clock stamp rides the free high half without perturbing the low
    /// half's status/index decode, round-trips through [`clock_ms`], and an
    /// unstamped word reads stamp `0` (the "no anchor" value).
    ///
    /// [`clock_ms`]: Committed::clock_ms
    #[test]
    fn clock_stamp_rides_the_high_half_untouched_by_decode() {
        for now_ms in [0u32, 1, 1_000, u32::MAX] {
            let word = stamp_clock(Status::WouldBlock.encode(7), now_ms);
            let c = Committed::from_raw(word);
            assert_eq!(c.status(), Status::WouldBlock);
            assert_eq!(c.failing_index(), 7);
            assert_eq!(c.clock_ms(), now_ms);
        }
        // Unstamped ⇒ stamp 0; re-stamping REPLACES (never accumulates).
        assert_eq!(Committed::from_raw(Status::Ok.encode(0)).clock_ms(), 0);
        let twice = stamp_clock(stamp_clock(Status::Ok.encode(0), 5), 9);
        assert_eq!(Committed::from_raw(twice).clock_ms(), 9);
    }

    /// The v3 [`FoundReq`] layout, pinned: the spawn block and
    /// `arena_slots` keep their v2 offsets (additive growth), the
    /// delivery map rides at offset 32 (78 bounded entries, tail-additive
    /// like v3 itself, offsets 0..32 frozen), and
    /// a hand-laid byte image reads back field-for-field through the
    /// same zerocopy face the kernel decode uses.
    ///
    /// What "frozen" pins is the HEAD, not the length: everything through
    /// offset 32 keeps its place and the map grows only at the tail, which is
    /// why the byte image below still writes the same offsets and reads back
    /// unchanged. A widening that disturbed a head offset would fail here —
    /// that is the property worth guarding, and the size line is the review
    /// gate that makes each widening deliberate.
    #[test]
    fn found_req_v3_delivery_map_layout_is_frozen() {
        use zerocopy::TryFromBytes;
        assert_eq!(core::mem::size_of::<FoundReq>(), FoundReq::WIRE);
        assert_eq!(FoundReq::WIRE, 344);
        assert_eq!(core::mem::size_of::<DeliverEntry>(), 4);

        let mut bytes = [0u8; FoundReq::WIRE];
        bytes[0..8].copy_from_slice(&0x4000u64.to_le_bytes()); // entry
        bytes[8..16].copy_from_slice(&0x7FFF_0000u64.to_le_bytes()); // stack
        bytes[16..20].copy_from_slice(&u32::MAX.to_le_bytes()); // bootstrap
        bytes[20..24].copy_from_slice(&0u32.to_le_bytes()); // flags
        bytes[24..28].copy_from_slice(&33u32.to_le_bytes()); // arena_slots
        bytes[28..30].copy_from_slice(&2u16.to_le_bytes()); // deliver_count
        bytes[30..32].copy_from_slice(&9u16.to_le_bytes()); // authority slot
        // deliver[0] = (from 5 → to 7); deliver[1] = (from 6 → to 8).
        bytes[32..34].copy_from_slice(&5u16.to_le_bytes());
        bytes[34..36].copy_from_slice(&7u16.to_le_bytes());
        bytes[36..38].copy_from_slice(&6u16.to_le_bytes());
        bytes[38..40].copy_from_slice(&8u16.to_le_bytes());

        let (req, rest) = FoundReq::try_ref_from_prefix(&bytes[..]).expect("decodes");
        assert!(rest.is_empty());
        assert_eq!(req.spawn.entry.get(), 0x4000);
        assert_eq!(req.spawn.stack.get(), 0x7FFF_0000);
        assert_eq!(req.spawn.bootstrap.get(), u32::MAX);
        assert_eq!(req.arena_slots.get(), 33);
        assert_eq!(req.deliver_count.get(), 2);
        assert_eq!(
            req.authority.get(),
            9,
            "the exec-authority slot rides at offset 30"
        );
        assert_eq!(req.deliver[0].from.get(), 5);
        assert_eq!(req.deliver[0].to.get(), 7);
        assert_eq!(req.deliver[1].from.get(), 6);
        assert_eq!(req.deliver[1].to.get(), 8);
        // Entries past the count are zero (the reserved-zero discipline the
        // kernel prepare enforces).
        assert_eq!(req.deliver[2].from.get(), 0);
        assert_eq!(req.deliver[2].to.get(), 0);
    }

    /// The count selector consumes one byte of the existing request reserve;
    /// the frozen 8-byte layout and the three remaining zero bytes do not move.
    /// (Named `DeviceStoreReq` until #572 — the bytes are the contract.)
    #[test]
    fn inventory_query_layout_is_frozen() {
        use zerocopy::TryFromBytes;

        let bytes = [0x78, 0x56, 0x34, 0x12, INVENTORY_QUERY_COUNT, 0, 0, 0];
        let (req, rest) = InventoryReq::try_ref_from_prefix(&bytes[..]).expect("decodes");
        assert!(rest.is_empty());
        assert_eq!(req.index.get(), 0x1234_5678);
        assert_eq!(req.query, INVENTORY_QUERY_COUNT);
        assert_eq!(req.reserved, [0; 3]);
    }

    #[test]
    fn fabric_census_owns_and_validates_its_observation_layout() {
        const TEST_EXTENT_BYTES: u64 = 96 << 20;
        let census = FabricCensus {
            domains: 3,
            capacity: 128,
            extent_bytes: TEST_EXTENT_BYTES,
            lendable_domains: 2,
            fence_required_domains: 1,
        };
        let (value, aux) = census.to_observation();
        assert_eq!(FabricCensus::from_observation(value, aux), Some(census));

        let invalid_lendable = u64::from(4_u16) << 32;
        assert_eq!(
            FabricCensus::from_observation(
                value,
                (aux & !(u64::from(u16::MAX) << 32)) | invalid_lendable,
            ),
            None
        );
        let invalid_domains = census.capacity + 1;
        assert_eq!(
            FabricCensus::from_observation(
                u64::from(invalid_domains) | (u64::from(census.capacity) << 32),
                0,
            ),
            None
        );

        let saturated = FabricCensus {
            domains: 1,
            capacity: 1,
            extent_bytes: u64::MAX,
            lendable_domains: u32::MAX,
            fence_required_domains: u32::MAX,
        }
        .to_observation();
        assert_eq!(saturated.1 as u32, u32::MAX);
        assert_eq!((saturated.1 >> 32) as u16, u16::MAX);
        assert_eq!((saturated.1 >> 48) as u16, u16::MAX);
    }

    /// DEVICE_ACTIVATE consumes the next registered high flavor and its
    /// second authenticated subject stays an additive, reserved-zero request.
    #[test]
    fn device_activate_request_and_flavor_are_frozen() {
        use zerocopy::TryFromBytes;

        let bytes = [0x78, 0x56, 0x34, 0x12, 0, 0, 0, 0];
        let (req, rest) = DeviceActivateReq::try_ref_from_prefix(&bytes[..]).expect("decodes");
        assert!(rest.is_empty());
        assert_eq!(DeviceActivateReq::WIRE, 8);
        assert_eq!(req.dma_lease.get(), 0x1234_5678);
        assert_eq!(req.reserved, [0; 4]);

        assert!(derive_kind_known(DERIVE_KIND_DEVICE_ACTIVATE));
        assert!(
            derive_kind_known(crate::cap::ResourceKind::Hypervisor as u8)
                && derive_kind_known(crate::cap::ResourceKind::GuestSpace as u8)
                && derive_kind_known(crate::cap::ResourceKind::GuestRegion as u8),
            "the append-only resource band includes the guest authority kinds"
        );
        assert!(
            derive_kind_known(DERIVE_KIND_SYSTEM_OBSERVE),
            "the read-only system observer is registered"
        );
        assert!(
            derive_kind_known(DERIVE_KIND_IRQ_ROUTE_MSIX),
            "the strict MSI-X route flavor is registered"
        );
        assert!(derive_kind_known(DERIVE_KIND_I2C_HID_RECORD));
        assert!(
            derive_kind_known(DERIVE_KIND_PEER_IOVA),
            "the sponsored peer-aperture flavor is registered"
        );
        assert!(
            derive_kind_known(DERIVE_KIND_SUPPLY_RELEASE),
            "the supply lifecycle flavor is registered"
        );
        assert!(
            derive_kind_known(DERIVE_KIND_VIEW_MAP),
            "the receiver-view placement flavor is registered"
        );
        assert!(
            derive_kind_known(DERIVE_KIND_GUEST_FOUND),
            "the guest-vCPU founding flavor is registered"
        );
        assert!(
            derive_kind_known(DERIVE_KIND_DEVICE_QUIESCE),
            "the terminal device-quiesce flavor is registered"
        );
        assert!(
            !derive_kind_known(0xed),
            "the next future flavor fails closed"
        );
        assert!(
            !derive_kind_known(0xff),
            "unregistered high bytes fail closed"
        );
    }

    /// Guest founding is a fixed, architecture-neutral 40-byte request.  The
    /// slot words are deliberately separate from the three guest-address
    /// words so every affine input is authenticated through the caller arena.
    #[test]
    fn guest_found_request_layout_is_frozen() {
        use zerocopy::TryFromBytes;

        let mut bytes = [0u8; GuestFoundReq::WIRE];
        bytes[0..8].copy_from_slice(&0x1000u64.to_le_bytes());
        bytes[8..16].copy_from_slice(&0x8000u64.to_le_bytes());
        bytes[16..24].copy_from_slice(&0x3000u64.to_le_bytes());
        bytes[24..28].copy_from_slice(&7u32.to_le_bytes());
        bytes[28..32].copy_from_slice(&8u32.to_le_bytes());
        bytes[32..36].copy_from_slice(&9u32.to_le_bytes());

        let (req, rest) = GuestFoundReq::try_ref_from_prefix(&bytes[..]).expect("decodes");
        assert!(rest.is_empty());
        assert_eq!(core::mem::size_of::<GuestFoundReq>(), 40);
        assert_eq!(req.entry.get(), 0x1000);
        assert_eq!(req.stack.get(), 0x8000);
        assert_eq!(req.page_table_root.get(), 0x3000);
        assert_eq!(req.window.get(), 7);
        assert_eq!(req.resume_recv.get(), 8);
        assert_eq!(req.exit_send.get(), 9);
        assert_eq!(req.reserved.get(), 0);
    }

    #[test]
    fn view_map_request_is_a_closed_window_selector() {
        for window in [
            ViewMapWindow::Map,
            ViewMapWindow::Surfaces,
            ViewMapWindow::Heap,
        ] {
            let mut bytes = [0u8; ViewMapReq::WIRE];
            bytes[..2].copy_from_slice(&window.role().to_le_bytes());
            let (req, rest) = ViewMapReq::try_ref_from_prefix(&bytes).unwrap();
            assert!(rest.is_empty());
            assert_eq!(req.placement(), Some(window));
            assert_eq!(
                req.surface_lane(),
                (window == ViewMapWindow::Surfaces).then_some(SurfaceMapLane::First)
            );
            for at in 3..bytes.len() {
                let mut dirty = bytes;
                dirty[at] = 1;
                let (req, _) = ViewMapReq::try_ref_from_prefix(&dirty).unwrap();
                assert_eq!(req.placement(), None);
            }
        }
        let mut second = [0u8; ViewMapReq::WIRE];
        second[..2].copy_from_slice(&ViewMapWindow::Surfaces.role().to_le_bytes());
        second[2] = SurfaceMapLane::Second as u8;
        let (second, _) = ViewMapReq::try_ref_from_prefix(&second).unwrap();
        assert_eq!(second.placement(), Some(ViewMapWindow::Surfaces));
        assert_eq!(second.surface_lane(), Some(SurfaceMapLane::Second));
        let mut invalid = *second;
        invalid.lane = 2;
        assert_eq!(invalid.placement(), None);
        let invalid_map_lane = ViewMapReq {
            window: ViewMapWindow::Map.role().into(),
            lane: SurfaceMapLane::Second as u8,
            reserved: [0; 5],
        };
        assert_eq!(invalid_map_lane.placement(), None);
        for role in [
            crate::task::geom::SCRATCH,
            crate::task::geom::APERTURE,
            crate::task::geom::DEVICE,
            u16::MAX,
        ] {
            let req = ViewMapReq {
                window: role.into(),
                lane: 0,
                reserved: [0; 5],
            };
            assert_eq!(req.placement(), None);
        }
    }

    #[test]
    fn guest_share_request_layout_is_frozen() {
        use zerocopy::TryFromBytes;

        let mut bytes = [0u8; GuestShareReq::WIRE];
        bytes[0..4].copy_from_slice(&7u32.to_le_bytes());
        bytes[4..12].copy_from_slice(&0x20_0000u64.to_le_bytes());
        bytes[12..20].copy_from_slice(&0x20_0000u64.to_le_bytes());
        bytes[20] = SHARE_ACCESS_R | SHARE_ACCESS_W;
        let (req, rest) = GuestShareReq::try_ref_from_prefix(&bytes[..]).expect("decodes");
        assert!(rest.is_empty());
        assert_eq!(req.space.get(), 7);
        assert_eq!(req.gpa.get(), 0x20_0000);
        assert_eq!(req.bytes.get(), 0x20_0000);
        assert_eq!(req.access, SHARE_ACCESS_R | SHARE_ACCESS_W);
        assert_eq!(req.reserved, [0; 3]);
    }

    #[test]
    fn memory_class_census_round_trips_and_saturates() {
        let census = MemoryClassCensus {
            carved: 6 << 30,
            free: 5 << 30,
            misses: 3,
        };
        let (value, aux) = census.to_observation();
        assert_eq!(MemoryClassCensus::from_observation(value, aux), census);
        let big = MemoryClassCensus {
            carved: u64::MAX,
            free: u64::MAX,
            misses: u64::MAX,
        };
        let (value, aux) = big.to_observation();
        let back = MemoryClassCensus::from_observation(value, aux);
        assert_eq!(back.carved, u64::MAX);
        assert_eq!(back.free, (1 << 48) - 1);
        assert_eq!(back.misses, u64::from(u16::MAX));
        assert!(SystemObserveReq::takes_operand(SystemMetric::MemoryClass));
    }

    #[test]
    fn memory_stream_sample_round_trips_and_saturates() {
        let sample = MemoryStreamSample {
            slot: 64,
            write_bps: 31_400_000_000,
            copy_bps: 12_900_000_000,
            hmat_bps: 136_500_000_000,
            streaming_bps: 80_100_000_000,
        };
        let (value, aux) = sample.to_observation();
        assert_eq!(MemoryStreamSample::from_observation(value, aux), sample);
        assert!(sample.measured());
        assert!(!MemoryStreamSample::default().measured());
        let big = MemoryStreamSample {
            slot: 255,
            write_bps: u64::MAX,
            copy_bps: u64::MAX,
            hmat_bps: u64::MAX,
            streaming_bps: u64::MAX,
        };
        let (value, aux) = big.to_observation();
        let back = MemoryStreamSample::from_observation(value, aux);
        assert_eq!(back.slot, 255);
        assert_eq!(back.write_bps, (1 << 40) - 1);
        assert_eq!(back.streaming_bps, u64::from(u16::MAX) * 100_000_000);
        assert_eq!(back.copy_bps, (1 << 40) - 1);
        assert_eq!(back.hmat_bps, ((1u64 << 24) - 1) * 1_000_000);
        assert!(SystemObserveReq::takes_operand(SystemMetric::MemoryStream));
    }

    #[test]
    fn fabric_route_request_layout_and_refusals() {
        assert_eq!(core::mem::size_of::<FabricRouteReq>(), FabricRouteReq::WIRE);
        let mut bytes = [0u8; FabricRouteReq::WIRE];
        bytes[0..8].copy_from_slice(&7u64.to_le_bytes());
        bytes[8..16].copy_from_slice(&4096u64.to_le_bytes());
        bytes[16..24].copy_from_slice(&8192u64.to_le_bytes());
        for i in 0..6 {
            bytes[24 + i * 4] = (i + 1) as u8;
        }
        bytes[48] = 7;
        bytes[49] = 1;
        bytes[50] = 1;
        let req = FabricRouteReq::try_read_from_bytes(&bytes).unwrap();
        assert!(req.valid());
        assert_eq!(req.encode(), bytes);
        assert_eq!(req.destination_allocation.get(), 6);
        for (offset, value) in [(48, 0), (48, 4), (48, 8), (49, 2), (50, 2), (51, 1)] {
            let mut malformed = bytes;
            malformed[offset] = value;
            assert!(
                !FabricRouteReq::try_read_from_bytes(&malformed)
                    .unwrap()
                    .valid()
            );
        }
        let mut overflow = req;
        overflow.offset = u64::MAX.into();
        assert!(!overflow.valid());
        let mut empty = req;
        empty.bytes = 0.into();
        assert!(!empty.valid());
        empty = req;
        empty.destination = 0.into();
        assert!(!empty.valid());
    }

    /// The typed constructors spell exactly the literal the wire expects,
    /// and every constructor product is valid.
    #[test]
    fn fabric_request_constructors_match_the_wire_literal() {
        let facts = FabricDomainReport {
            domain: 7,
            bytes: 1 << 20,
            owner: 9,
            reset: 1,
            power: 2,
            allocation: 3,
            cache: 0,
            coherent: true,
            needs_fence: false,
        };
        let req = FabricRouteReq::self_route(
            &facts,
            8192,
            FabricOp::IommuMap,
            FabricAccess::all(),
            FabricSync::Fenced,
        );
        assert!(req.valid());
        let literal = FabricRouteReq {
            destination: 7.into(),
            offset: 0.into(),
            bytes: 8192.into(),
            source_reset: 1.into(),
            source_power: 2.into(),
            source_allocation: 3.into(),
            destination_reset: 1.into(),
            destination_power: 2.into(),
            destination_allocation: 3.into(),
            access: 7,
            operation: 1,
            synchronization: 1,
            reserved: [0; 5],
        };
        assert_eq!(req.encode(), literal.encode());
        let other = FabricDomainReport {
            domain: 11,
            reset: 4,
            power: 5,
            allocation: 6,
            ..facts
        };
        let between = FabricRouteReq::between(
            &facts,
            &other,
            4096,
            4096,
            FabricOp::CpuMap,
            FabricAccess::READ,
            FabricSync::Unfenced,
        );
        assert!(between.valid());
        assert_eq!(between.destination.get(), 11);
        assert_eq!(between.destination_allocation.get(), 6);
        assert_eq!(between.source_allocation.get(), 3);
        assert_eq!(
            (between.access, between.operation, between.synchronization),
            (1, 0, 0)
        );
        // An exclusive-only access is not a route: the kernel refuses it and
        // so does `valid`, so a constructor cannot smuggle it in silently.
        let bad = FabricRouteReq::self_route(
            &facts,
            4096,
            FabricOp::CpuMap,
            FabricAccess::EXCLUSIVE,
            FabricSync::Fenced,
        );
        assert!(!bad.valid());
        let scope = FabricRouteScopeReq::new(4096, 8192, FabricAccess::all());
        assert!(scope.valid());
        assert_eq!(
            scope.encode(),
            FabricRouteScopeReq {
                offset: 4096.into(),
                bytes: 8192.into(),
                access: 7,
                reserved: [0; 7],
            }
            .encode()
        );
    }

    #[test]
    fn fabric_scope_wire_refuses_noncanonical_or_unmappable_ranges() {
        assert_eq!(
            core::mem::size_of::<FabricRouteScopeReq>(),
            FabricRouteScopeReq::WIRE
        );
        let req = FabricRouteScopeReq {
            offset: 4096.into(),
            bytes: 8192.into(),
            access: 7,
            reserved: [0; 7],
        };
        let bytes = req.encode();
        let decoded = FabricRouteScopeReq::try_read_from_bytes(&bytes).unwrap();
        assert!(decoded.valid());
        assert_eq!(decoded.offset.get(), 4096);
        assert_eq!(decoded.bytes.get(), 8192);
        for offset in 17..24 {
            let mut bad = bytes;
            bad[offset] = 1;
            assert!(
                !FabricRouteScopeReq::try_read_from_bytes(&bad)
                    .unwrap()
                    .valid()
            );
        }
        for access in [0, 4, 8, 255] {
            assert!(!FabricRouteScopeReq { access, ..req }.valid());
        }
        for (offset, len) in [(0, 0), (1, 4096), (0, 4095), (u64::MAX - 4095, 4096)] {
            assert!(
                !FabricRouteScopeReq {
                    offset: offset.into(),
                    bytes: len.into(),
                    ..req
                }
                .valid()
            );
        }
    }

    #[test]
    fn route_info_refuses_noncanonical_or_nonmapping_authority() {
        let info = FabricRouteInfo {
            bytes: 8192,
            access: 7,
            operation: 1,
        };
        let record = info.encode();
        assert_eq!(FabricRouteInfo::decode(&record), Some(info));
        for flags in [0, 4, 8, 0x203, 0x108] {
            let mut bad = record;
            bad.holder = crate::cap::Cid(flags);
            assert!(FabricRouteInfo::decode(&bad).is_none());
        }
        for bytes in [0, 1, 4095, u64::MAX] {
            let mut bad = record;
            bad.scope_rid = bytes;
            assert!(FabricRouteInfo::decode(&bad).is_none());
        }
        let mut bad = record;
        bad.mac[0] = 1;
        assert!(FabricRouteInfo::decode(&bad).is_none());
        let mut bad = record;
        bad.chain[0] = 1;
        assert!(FabricRouteInfo::decode(&bad).is_none());
    }

    #[test]
    fn route_close_report_is_inert_and_canonical() {
        for state in 0..=2 {
            let report = FabricRouteCloseReport {
                route: 0x100000007,
                state,
            };
            let encoded = report.encode();
            assert_eq!(FabricRouteCloseReport::decode(&encoded), Some(report));
            assert_eq!(FabricDomainReport::decode(&encoded), None);
            for field in 0..8 {
                let mut corrupt = encoded;
                match field {
                    0 => corrupt.cid = crate::cap::Cid(1),
                    1 => corrupt.holder = crate::cap::Cid(3),
                    2 => corrupt.scope_rid = 0,
                    3 => corrupt.rights = crate::cap::Rights::ACCESS,
                    4 => corrupt.chain_len = 1,
                    5 => corrupt.chain[0] = 1,
                    6 => corrupt.epoch = 1,
                    _ => corrupt.mac[0] = 1,
                }
                assert_eq!(FabricRouteCloseReport::decode(&corrupt), None);
            }
        }
        assert_eq!(
            FabricRouteCloseReport::decode(&GeomReport { base: 1, len: 2 }.encode()),
            None
        );
    }

    #[test]
    fn fabric_report_rejects_noncanonical_and_authority_records() {
        let report = FabricDomainReport {
            domain: 1 << 63,
            bytes: u64::MAX,
            owner: u64::MAX,
            reset: u32::MAX,
            power: 19,
            allocation: 23,
            cache: 2,
            coherent: true,
            needs_fence: true,
        };
        let encoded = report.encode();
        assert_eq!(FabricDomainReport::decode(&encoded), Some(report));
        for field in 0..9 {
            let mut corrupt = encoded;
            match field {
                0 => corrupt.cid = crate::cap::Cid(1),
                1 => corrupt.chain[5] |= 4,
                2 => corrupt.chain[5] |= 3,
                3 => corrupt.chain[6] = 1,
                4 => corrupt.chain_len = 1,
                5 => corrupt.epoch = 1,
                6 => corrupt.mac[0] = 1,
                7 => corrupt.rights = crate::cap::Rights::ACCESS,
                _ => corrupt.scope_rid = 0,
            }
            assert_eq!(FabricDomainReport::decode(&corrupt), None);
        }
        assert_eq!(
            FabricDomainReport::decode(&GeomReport { base: 1, len: 2 }.encode()),
            None
        );
        assert_eq!(GeomReport::decode(&encoded), None);
    }

    /// **The family gate's `cid == NULL` clause is load-bearing, for every
    /// member** — the I4/I6 property, pinned once now that one
    /// [`is_inert_report`] serves them all.
    ///
    /// A record carrying a member's exact kind AND magic but a **live
    /// cid** must decode as nothing: it is a capability (or something
    /// pretending to be), and letting it read as data would hand a
    /// consumer a payload marshalled out of a sealed record's columns.
    /// This is the clause a seventh member would have been most likely to
    /// forget while transcribing the gate by hand — which is the whole
    /// reason the gate now has exactly one implementation.
    #[test]
    fn a_live_cid_never_decodes_as_any_report() {
        // One (kind, magic) pair per member, and a decoder that must
        // refuse it once the record stops being architecturally vacant.
        let decoders: [(u16, u64, fn(&crate::cap::CapRecord) -> bool); 13] = [
            (FABRIC_ROUTE_INFO_KIND, FABRIC_ROUTE_INFO_MAGIC, |r| {
                FabricRouteInfo::decode(r).is_some()
            }),
            (
                FABRIC_ROUTE_CLOSE_REPORT_KIND,
                FABRIC_ROUTE_CLOSE_REPORT_MAGIC,
                |r| FabricRouteCloseReport::decode(r).is_some(),
            ),
            (FABRIC_DOMAIN_REPORT_KIND, FABRIC_DOMAIN_REPORT_MAGIC, |r| {
                FabricDomainReport::decode(r).is_some()
            }),
            (NOTICE_KIND, NOTICE_MAGIC, |r| Notice::decode(r).is_some()),
            (GEOM_REPORT_KIND, GEOM_REPORT_MAGIC, |r| {
                GeomReport::decode(r).is_some()
            }),
            (DISPLAY_REPORT_KIND, DISPLAY_REPORT_MAGIC, |r| {
                DisplayReport::decode(r).is_some()
            }),
            (EXIT_REPORT_KIND, EXIT_REPORT_MAGIC, |r| {
                ExitReport::decode(r).is_some()
            }),
            (DEVICE_REPORT_KIND, DEVICE_REPORT_MAGIC, |r| {
                DeviceReport::decode(r).is_some()
            }),
            (DEVICE_COUNT_REPORT_KIND, DEVICE_COUNT_REPORT_MAGIC, |r| {
                DeviceCountReport::decode(r).is_some()
            }),
            (
                SYSTEM_OBSERVE_REPORT_KIND,
                SYSTEM_OBSERVE_REPORT_MAGIC,
                |r| SystemObserveReport::decode(r).is_some(),
            ),
            (I2C_HID_REPORT_KIND, I2C_HID_REPORT_MAGIC, |r| {
                I2cHidReport::decode(r).is_some()
            }),
            (I2C_HID_COUNT_REPORT_KIND, I2C_HID_COUNT_REPORT_MAGIC, |r| {
                I2cHidCountReport::decode(r).is_some()
            }),
            (
                SUPPLY_RELEASE_REPORT_KIND,
                SUPPLY_RELEASE_REPORT_MAGIC,
                |r| SupplyReleaseReport::decode(r).is_some(),
            ),
        ];
        for (kind, magic, decodes) in decoders {
            // Vacant + this member's stamps ⇒ it IS this member's report.
            let mut rec = if kind == I2C_HID_REPORT_KIND {
                I2cHidReport {
                    slave_addr: 0x2c,
                    descriptor_register: 1,
                    speed_hz: 400_000,
                    standard_timing: None,
                    fast_timing: None,
                    input_clock_hz: None,
                    ready_interrupt: None,
                    ready_route_state: I2cReadyRouteState::NotDescribed,
                    no_irq: false,
                    no_power_after_reset: false,
                }
                .encode()
            } else if kind == I2C_HID_COUNT_REPORT_KIND {
                I2cHidCountReport { len: 0 }.encode()
            } else {
                let mut rec = crate::cap::CapRecord::vacant();
                rec.parent = crate::cap::Cid(magic);
                rec.kind = kind;
                rec
            };
            if kind == FABRIC_DOMAIN_REPORT_KIND {
                rec.scope_rid = 1 << 63;
                rec.holder = crate::cap::Cid(4096);
            }
            if kind == FABRIC_ROUTE_CLOSE_REPORT_KIND {
                rec.scope_rid = 1;
            }
            if kind == FABRIC_ROUTE_INFO_KIND {
                rec.scope_rid = 4096;
                rec.holder = crate::cap::Cid(0x103);
            }
            if kind == SYSTEM_OBSERVE_REPORT_KIND {
                rec.chain_len = SystemMetric::Clock as u8;
            }
            assert!(
                decodes(&rec),
                "kind {kind:#x}: the fixture must decode before the cid is dirtied"
            );
            // The same record with a live cid ⇒ nothing, for every member.
            rec.cid = crate::cap::Cid(0x1234_5678_9abc_def0);
            assert!(
                !decodes(&rec),
                "kind {kind:#x}: a live cid must never decode as a report"
            );
        }
    }

    #[test]
    fn system_observation_round_trips_as_bind_inert_data() {
        let report = SystemObserveReport {
            metric: SystemMetric::PhysicalMemory,
            value: 0x8000_0000,
            aux: 0x2000_0000,
        };
        let rec = report.encode();
        assert_eq!(rec.cid, crate::cap::Cid::NULL);
        assert_eq!(SystemObserveReport::decode(&rec), Some(report));

        let mut unknown = rec;
        unknown.chain_len = 0xff;
        assert_eq!(SystemObserveReport::decode(&unknown), None);
    }

    #[test]
    fn i2c_admission_pages_are_strictly_inert_and_bounded() {
        use crate::device::{I2cAdmissionPage, I2cAdmissionSnapshot, I2cInventoryState};
        let snapshot = I2cAdmissionSnapshot {
            inventory_state: I2cInventoryState::Initialized as u8,
            ..I2cAdmissionSnapshot::default()
        };
        let report = I2cAdmissionReport {
            page: snapshot.page(0).unwrap(),
        };
        let encoded = report.encode();
        assert_eq!(encoded.cid, crate::cap::Cid::NULL);
        assert_eq!(I2cAdmissionReport::decode(&encoded), Some(report));
        for change in [
            |r: &mut crate::cap::CapRecord| r.cid = crate::cap::Cid(1),
            |r: &mut crate::cap::CapRecord| r.rights = crate::cap::Rights::ACCESS,
            |r: &mut crate::cap::CapRecord| {
                r.scope_rid = u64::from(crate::device::I2C_ADMISSION_ROWS)
            },
            |r: &mut crate::cap::CapRecord| r.epoch = 1,
            |r: &mut crate::cap::CapRecord| r.chain_len = 1,
            |r: &mut crate::cap::CapRecord| r.chain[0] = 1,
            |r: &mut crate::cap::CapRecord| r.mac[0] = 1,
        ] {
            let mut dirty = encoded;
            change(&mut dirty);
            assert_eq!(I2cAdmissionReport::decode(&dirty), None);
        }
        let last = I2cAdmissionReport {
            page: I2cAdmissionPage {
                row: 24,
                words: [u32::MAX, u32::MAX],
            },
        };
        assert_eq!(I2cAdmissionReport::decode(&last.encode()), Some(last));
        let build = crate::device::AmlBuildDiagnostic::failure(
            crate::device::AmlBuildStage::SsdtLoad,
            crate::device::AmlBuildErrorClass::Reference,
            u32::MAX,
            u16::MAX,
        )
        .unwrap();
        let report = I2cAdmissionReport {
            page: I2cAdmissionPage {
                row: crate::device::I2C_ADMISSION_BUILD_ROW,
                words: build.words(),
            },
        };
        let mut encoded = report.encode();
        assert_eq!(I2cAdmissionReport::decode(&encoded), Some(report));
        encoded.rights = crate::cap::Rights::ACCESS;
        assert!(I2cAdmissionReport::decode(&encoded).is_none());
    }

    #[test]
    fn i2c_hid_reports_round_trip_every_optional_field() {
        let report = I2cHidReport {
            slave_addr: 0x2c,
            descriptor_register: 1,
            speed_hz: 400_000,
            standard_timing: Some(I2cTimingReport {
                hcnt: 0x214,
                lcnt: 0x272,
            }),
            fast_timing: Some(I2cTimingReport {
                hcnt: 0x50,
                lcnt: 0xa0,
            }),
            input_clock_hz: Some(19_200_000),
            ready_interrupt: Some(I2cReadyInterruptReport::Gpio(I2cGpioInterruptReport {
                pin: 37,
                edge: false,
                polarity: I2cGpioPolarity::Low,
            })),
            ready_route_state: I2cReadyRouteState::Ready,
            no_irq: true,
            no_power_after_reset: true,
        };
        let encoded = report.encode();
        assert_eq!(encoded.cid, crate::cap::Cid::NULL);
        assert_eq!(I2cHidReport::decode(&encoded), Some(report));
        assert_eq!(I2cHidCountReport::decode(&encoded), None);

        let minimal = I2cHidReport {
            slave_addr: 0x15,
            descriptor_register: 0xabcd,
            speed_hz: 100_000,
            standard_timing: None,
            fast_timing: None,
            input_clock_hz: None,
            ready_interrupt: None,
            ready_route_state: I2cReadyRouteState::NotDescribed,
            no_irq: false,
            no_power_after_reset: false,
        };
        assert_eq!(I2cHidReport::decode(&minimal.encode()), Some(minimal));
    }

    #[test]
    fn i2c_child_line_retains_full_gsi_and_electrical_mode_and_refuses_ambiguous_sources() {
        for gsi in [0, 84, 108, u32::MAX] {
            for (edge, active_high) in [(false, false), (false, true), (true, false), (true, true)]
            {
                let report = I2cHidReport {
                    slave_addr: 0x16,
                    descriptor_register: 1,
                    speed_hz: 400_000,
                    standard_timing: None,
                    fast_timing: None,
                    input_clock_hz: None,
                    ready_interrupt: Some(I2cReadyInterruptReport::Line {
                        gsi,
                        edge,
                        active_high,
                    }),
                    ready_route_state: I2cReadyRouteState::Ready,
                    no_irq: false,
                    no_power_after_reset: false,
                };
                let encoded = report.encode();
                assert_eq!(I2cHidReport::decode(&encoded), Some(report));
                let mut ambiguous = encoded;
                ambiguous.rights = crate::cap::Rights::from_bits_retain(
                    ambiguous.rights.bits() | I2C_GPIO_INTERRUPT,
                );
                assert!(I2cHidReport::decode(&ambiguous).is_none());
                let mut invalid_polarity = encoded;
                invalid_polarity.chain[0] = 2;
                assert!(I2cHidReport::decode(&invalid_polarity).is_none());
                let mut absent = encoded;
                absent.chain[2] = I2cReadyRouteState::NotDescribed as u32;
                assert!(I2cHidReport::decode(&absent).is_none());
            }
        }
    }

    #[test]
    fn i2c_hid_descriptor_offsets_preserve_both_u16_boundaries() {
        for descriptor_register in [0, u16::MAX] {
            let report = I2cHidReport {
                slave_addr: 0x2c,
                descriptor_register,
                speed_hz: 400_000,
                standard_timing: None,
                fast_timing: None,
                input_clock_hz: None,
                ready_interrupt: None,
                ready_route_state: I2cReadyRouteState::NotDescribed,
                no_irq: false,
                no_power_after_reset: false,
            };
            assert_eq!(I2cHidReport::decode(&report.encode()), Some(report));
        }
    }

    #[test]
    fn i2c_hid_reports_refuse_noncanonical_absence_and_ranges() {
        let base = I2cHidReport {
            slave_addr: 0x2c,
            descriptor_register: 1,
            speed_hz: 400_000,
            standard_timing: None,
            fast_timing: None,
            input_clock_hz: None,
            ready_interrupt: None,
            ready_route_state: I2cReadyRouteState::NotDescribed,
            no_irq: false,
            no_power_after_reset: false,
        };

        let mut stale_timing = base.encode();
        stale_timing.holder = crate::cap::Cid(1);
        assert_eq!(I2cHidReport::decode(&stale_timing), None);

        let mut stale_gpio = base.encode();
        stale_gpio.epoch = 1;
        assert_eq!(I2cHidReport::decode(&stale_gpio), None);

        let mut route_without_gpio = base.encode();
        route_without_gpio.chain[2] = I2cReadyRouteState::Ready as u32;
        assert_eq!(I2cHidReport::decode(&route_without_gpio), None);

        let mut unknown_route_state = base.encode();
        unknown_route_state.chain[2] = u32::MAX;
        assert_eq!(I2cHidReport::decode(&unknown_route_state), None);

        let mut missing_route_state = I2cHidReport {
            ready_interrupt: Some(I2cReadyInterruptReport::Gpio(I2cGpioInterruptReport {
                pin: 18,
                edge: false,
                polarity: I2cGpioPolarity::Low,
            })),
            ready_route_state: I2cReadyRouteState::Ready,
            ..base
        }
        .encode();
        missing_route_state.chain[2] = I2cReadyRouteState::NotDescribed as u32;
        assert_eq!(I2cHidReport::decode(&missing_route_state), None);

        let mut stale_clock = base.encode();
        stale_clock.chain[1] = 19_200_000;
        assert_eq!(I2cHidReport::decode(&stale_clock), None);

        let mut zero_clock = base.encode();
        zero_clock.rights = crate::cap::Rights::from_bits_retain(I2C_INPUT_CLOCK);
        assert_eq!(I2cHidReport::decode(&zero_clock), None);

        let mut wide_slave = base.encode();
        wide_slave.scope_rid = (wide_slave.scope_rid & !0xff) | 0x80;
        assert_eq!(I2cHidReport::decode(&wide_slave), None);

        for len in [0, 1, u32::MAX] {
            let count = I2cHidCountReport { len };
            assert_eq!(I2cHidCountReport::decode(&count.encode()), Some(count));
        }
    }

    #[test]
    fn tracked_supply_request_and_completion_are_frozen() {
        use zerocopy::TryFromBytes;

        let mut bytes = [0u8; TrackedLeaseReq::WIRE];
        bytes[0..8].copy_from_slice(&0x20_0000u64.to_le_bytes());
        bytes[8] = LEASE_TRACK_RELEASE;
        bytes[9] = LEASE_ALIGN_2M;
        bytes[16..24].copy_from_slice(&0x1234_5678_9abc_def0u64.to_le_bytes());
        bytes[24..26].copy_from_slice(&3u16.to_le_bytes());
        let (req, rest) = TrackedLeaseReq::try_ref_from_prefix(&bytes[..]).expect("decodes");
        assert!(rest.is_empty());
        assert_eq!(req.lease.size.get(), 0x20_0000);
        assert_eq!(req.lease.flags, LEASE_TRACK_RELEASE);
        assert_eq!(req.lease.align_log2, LEASE_ALIGN_2M);
        assert_eq!(req.receipt.get(), 0x1234_5678_9abc_def0);
        assert_eq!(req.pair.get(), 3);
        assert_eq!(req.reserved, [0; 6]);

        let report = SupplyReleaseReport {
            receipt: u64::MAX,
            pair: u16::MAX,
            len: u64::MAX,
        };
        let rec = report.encode();
        assert_eq!(rec.cid, crate::cap::Cid::NULL);
        assert_eq!(SupplyReleaseReport::decode(&rec), Some(report));

        let mut out_of_range = rec;
        out_of_range.epoch = u32::from(u16::MAX) + 1;
        assert_eq!(SupplyReleaseReport::decode(&out_of_range), None);
    }

    /// The doorbell [`Notice`] round-trips through its bind-inert record
    /// encoding and stays architecturally vacant (`cid == NULL`) — and
    /// the pager's fault notice is one of them (the folded `PagerNotice`:
    /// the reserved cookie, the VA in the payload), so the delivery
    /// classifier that recognizes `Notice` recognizes the fault wake.
    #[test]
    fn notice_round_trips_bind_inert_including_the_pager_fault() {
        let n = Notice {
            cookie: 0xDEAD_BEEF_u64,
            payload: 42,
        };
        let rec = n.encode();
        assert_eq!(rec.cid, crate::cap::Cid::NULL, "bind-inert: reads Vacant");
        assert_eq!(Notice::decode(&rec), Some(n));

        let fault = Notice {
            cookie: NOTICE_COOKIE_PAGER_FAULT,
            payload: 0x4000_2000,
        };
        let frec = fault.encode();
        assert_eq!(
            Notice::decode(&frec),
            Some(fault),
            "the fault notice IS a doorbell notice — that is the fold"
        );
        // A stale pre-fold encoder (kind 0xFF01, magic "korePWNT") must
        // read as nothing, not as a doorbell.
        let mut stale = crate::cap::CapRecord::vacant();
        stale.parent = crate::cap::Cid(0x6B6F_7265_5057_4E54);
        stale.scope_rid = 0x4000_2000;
        stale.kind = 0xFF01;
        assert_eq!(Notice::decode(&stale), None, "retired kind stays dead");

        // A scrubbed / vacant record decodes as nothing.
        let vacant = crate::cap::CapRecord::vacant();
        assert_eq!(Notice::decode(&vacant), None);
    }

    /// The [`GeomReport`] round-trips through its bind-inert record
    /// encoding, stays architecturally vacant, and is kind-disjoint from
    /// the doorbell notice.
    #[test]
    fn geom_report_round_trips_bind_inert_and_disjoint() {
        let g = GeomReport {
            base: 0x5800_0000,
            len: 0x1000,
        };
        let rec = g.encode();
        assert_eq!(rec.cid, crate::cap::Cid::NULL, "bind-inert: reads Vacant");
        assert_eq!(GeomReport::decode(&rec), Some(g));
        assert_eq!(Notice::decode(&rec), None, "kind-disjoint families");

        let n = Notice {
            cookie: 1,
            payload: 2,
        }
        .encode();
        assert_eq!(
            GeomReport::decode(&n),
            None,
            "a doorbell is not a geometry report"
        );
        assert_eq!(GeomReport::decode(&crate::cap::CapRecord::vacant()), None);
    }

    /// The [`QuiesceReport`] round-trips every field through its packed
    /// `scope_rid` word, stays bind-inert, and is kind-disjoint from the
    /// doorbell and geometry members.
    #[test]
    fn quiesce_report_round_trips_bind_inert_and_disjoint() {
        let q = QuiesceReport {
            rid: 0xDEAD_0042,
            verdict: QUIESCE_VERDICT_CONTAINED,
            flags: QUIESCE_FLAG_IRQ_MASKED | QUIESCE_FLAG_ABORT_BOUND | QUIESCE_FLAG_PEERS_DRAINED,
        };
        let rec = q.encode();
        assert_eq!(rec.cid, crate::cap::Cid::NULL, "bind-inert: reads Vacant");
        assert_eq!(
            rec.holder,
            crate::cap::Cid(0),
            "holder word reserved for #725"
        );
        assert_eq!(QuiesceReport::decode(&rec), Some(q));
        assert_eq!(Notice::decode(&rec), None, "kind-disjoint families");
        assert_eq!(GeomReport::decode(&rec), None, "kind-disjoint families");

        let proven = QuiesceReport {
            rid: 7,
            verdict: QUIESCE_VERDICT_PROVEN,
            flags: 0,
        };
        assert_eq!(QuiesceReport::decode(&proven.encode()), Some(proven));
        assert_eq!(
            QuiesceReport::decode(&GeomReport { base: 1, len: 2 }.encode()),
            None,
            "a geometry report is not a quiesce verdict"
        );
        assert_eq!(
            QuiesceReport::decode(&crate::cap::CapRecord::vacant()),
            None
        );
    }

    /// The [`ExitReport`] round-trips through its bind-inert record, and a
    /// negative status survives it — the status is a value a program chose,
    /// so the sign is part of the payload, not a bit pattern to lose.
    #[test]
    fn exit_report_round_trips_bind_inert() {
        for status in [
            0i32,
            1,
            255,
            256,
            257,
            65536,
            i32::MAX,
            -1,
            -256,
            -42,
            i32::MIN,
        ] {
            let rec = ExitReport { status }.encode();
            assert_eq!(rec.cid, crate::cap::Cid::NULL, "a report binds to nothing");
            assert_eq!(ExitReport::decode(&rec), Some(ExitReport { status }));
        }
        // Not confusable with the family's other members, nor with a
        // vacant slot — a caller that never got a report must not read one.
        let geom = GeomReport { base: 8, len: 16 }.encode();
        assert_eq!(ExitReport::decode(&geom), None);
        assert_eq!(GeomReport::decode(&ExitReport { status: 0 }.encode()), None);
        assert_eq!(ExitReport::decode(&crate::cap::CapRecord::vacant()), None);
    }

    #[test]
    fn exit_report_rejects_high_bit_aliases_and_capability_residue() {
        for bits in [
            1u64 << 32,
            0xffff_ffff,
            0x8000_0000,
            0x1_0000_0100,
            0xffff_fffe_0000_0000,
        ] {
            let mut rec = ExitReport { status: 0 }.encode();
            rec.scope_rid = bits;
            assert_eq!(ExitReport::decode(&rec), None, "{bits:#x}");
        }
        let pristine = ExitReport { status: 256 }.encode();
        let mut variants = [pristine; 6];
        variants[0].holder = crate::cap::Cid(1);
        variants[1].rights = crate::cap::Rights::ACCESS;
        variants[2].epoch = 1;
        variants[3].chain_len = 1;
        variants[4].chain[0] = 1;
        variants[5].mac[0] = 1;
        for rec in variants {
            assert_eq!(ExitReport::decode(&rec), None);
        }
    }

    /// The [`DeviceReport`] round-trips every field at its widest value —
    /// the packing crosses six field boundaries inside one inert word, so
    /// an all-ones probe is what catches a shift that silently eats a
    /// neighbour (a class byte bleeding into `progif` would hand a broker
    /// a device of the wrong class and read as a selector bug).
    #[test]
    fn device_report_round_trips_bind_inert_and_disjoint() {
        for r in [
            DeviceReport {
                species: 0,
                vendor: 0x8086,
                device: 0x9a13,
                class: 0x0c,
                subclass: 0x03,
                progif: 0x30,
                flags: DEVICE_FLAG_PROBE,
                msix_table_size: 2,
                locator: 0x00a0,
                segment: 0,
                diag: 0,
                revision: 0,
            },
            DeviceReport {
                species: u8::MAX,
                vendor: u16::MAX,
                device: u16::MAX,
                class: u8::MAX,
                subclass: u8::MAX,
                progif: u8::MAX,
                flags: u8::MAX,
                msix_table_size: u16::MAX,
                locator: u32::MAX,
                segment: u16::MAX,
                diag: 0,
                revision: 0,
            },
            DeviceReport {
                species: 1,
                vendor: 0,
                device: 0,
                class: 0,
                subclass: 0,
                progif: 0,
                flags: 0,
                msix_table_size: 0,
                locator: 0,
                segment: 0,
                diag: 0,
                revision: 0,
            },
        ] {
            let rec = r.encode();
            assert_eq!(rec.cid, crate::cap::Cid::NULL, "a report binds to nothing");
            assert_eq!(DeviceReport::decode(&rec), Some(r));
        }
        // Kind-disjoint from every other member of the family, in both
        // directions — a broker that never got an entry must not read one.
        // (This comment used to be FALSE: EXIT_REPORT_KIND and
        // DEVICE_REPORT_KIND shared 0xFF05 and only the parent magic told
        // them apart.  The kinds are disjoint now and REPORT_KINDS proves
        // it at compile time; the runtime asserts below cover the decode
        // half in both directions.)
        let d = DeviceReport {
            species: 0,
            vendor: 1,
            device: 2,
            class: 3,
            subclass: 4,
            progif: 5,
            flags: 6,
            msix_table_size: 7,
            locator: 7,
            segment: 8,
            diag: 0,
            revision: 0,
        }
        .encode();
        assert_eq!(GeomReport::decode(&d), None);
        assert_eq!(Notice::decode(&d), None);
        assert_eq!(ExitReport::decode(&d), None);
        assert_eq!(
            DeviceReport::decode(&GeomReport { base: 8, len: 16 }.encode()),
            None
        );
        // The reverse of the 0xFF05 collision — the direction the old
        // suite never asserted.
        assert_eq!(
            DeviceReport::decode(&ExitReport { status: -7 }.encode()),
            None
        );
        assert_eq!(
            ExitReport::decode(
                &DeviceReport {
                    species: 0,
                    vendor: 1,
                    device: 2,
                    class: 3,
                    subclass: 4,
                    progif: 5,
                    flags: 6,
                    msix_table_size: 7,
                    locator: 7,
                    segment: 8,
                    diag: 0,
                    revision: 0,
                }
                .encode()
            ),
            None
        );
        assert_eq!(DeviceReport::decode(&crate::cap::CapRecord::vacant()), None);

        let mut wide_msix = DeviceReport {
            species: 0,
            vendor: 0x8086,
            device: 0x56a0,
            class: 3,
            subclass: 0,
            progif: 0,
            flags: DEVICE_FLAG_PROBE,
            msix_table_size: 2,
            locator: 0,
            segment: 0,
            diag: 0,
            revision: 0,
        }
        .encode();
        wide_msix.chain[0] = u32::from(u16::MAX) + 1;
        assert_eq!(DeviceReport::decode(&wide_msix), None);
    }

    /// A count is an explicit successful report, not a refused record read.
    /// Its encoding is disjoint from an inventory entry and refuses a value
    /// wider than the public `u32` count vocabulary.
    #[test]
    fn device_count_report_round_trips_bind_inert_and_fail_closed() {
        for len in [0, 1, 31, u32::MAX] {
            let report = DeviceCountReport { len };
            let rec = report.encode();
            assert_eq!(rec.cid, crate::cap::Cid::NULL, "a report binds to nothing");
            assert_eq!(DeviceCountReport::decode(&rec), Some(report));
            assert_eq!(DeviceReport::decode(&rec), None);
        }

        let mut wide = DeviceCountReport { len: 1 }.encode();
        wide.scope_rid |= 1u64 << 32;
        assert_eq!(DeviceCountReport::decode(&wide), None);
        assert_eq!(
            DeviceCountReport::decode(&crate::cap::CapRecord::vacant()),
            None
        );
    }
    #[test]
    fn virtio_pci_reports_round_trip_and_remain_bind_inert() {
        let row = VirtioPciCapReport {
            cfg_type: 2,
            bar: 4,
            id: 7,
            offset: 0x1234_5678,
            length: 0x8765_4321,
            notify_multiplier: 4,
        };
        let encoded = row.encode();
        assert_eq!(encoded.cid, crate::cap::Cid::NULL);
        assert_eq!(VirtioPciCapReport::decode(&encoded), Some(row));
        assert_eq!(VirtioPciCapCountReport::decode(&encoded), None);
        assert_eq!(DeviceReport::decode(&encoded), None);

        for len in [0, 1, 8, u8::MAX] {
            let count = VirtioPciCapCountReport { len };
            let encoded = count.encode();
            assert_eq!(encoded.cid, crate::cap::Cid::NULL);
            assert_eq!(VirtioPciCapCountReport::decode(&encoded), Some(count));
            assert_eq!(VirtioPciCapReport::decode(&encoded), None);
        }

        let mut malformed = row.encode();
        malformed.holder.0 |= 1u64 << 63;
        assert_eq!(VirtioPciCapReport::decode(&malformed), None);

        let mut wide = VirtioPciCapCountReport { len: 1 }.encode();
        wide.scope_rid = 256;
        assert_eq!(VirtioPciCapCountReport::decode(&wide), None);
    }
}

/// A representative flag mix for the round-trip probe above — every
/// settled fact a PCI driver row's selector asks for.
#[cfg(test)]
const DEVICE_FLAG_PROBE: u8 = crate::device::DeviceRecord::HAS_BINDING
    | crate::device::DeviceRecord::HAS_REMAPPED_MSI
    | crate::device::DeviceRecord::HAS_BAR0
    | crate::device::DeviceRecord::DRIVER_ENDPOINT
    | crate::device::DeviceRecord::HAS_REMAPPED_MSIX;

#[cfg(test)]
mod process_drain_tests {
    use super::*;
    #[test]
    fn drain_report_is_canonical_data_not_authority() {
        for forced in [false, true] {
            let report = ProcessDrainReport { forced };
            let record = report.encode();
            assert_eq!(ProcessDrainReport::decode(&record), Some(report));
            let mut bad = record;
            bad.scope_rid = 2;
            assert_eq!(ProcessDrainReport::decode(&bad), None);
            let mut bad = record;
            bad.cid = crate::cap::Cid(1);
            assert_eq!(ProcessDrainReport::decode(&bad), None);
            let mut bad = record;
            bad.chain_len = 1;
            assert_eq!(ProcessDrainReport::decode(&bad), None);
            let mut bad = record;
            bad.mac[0] = 1;
            assert_eq!(ProcessDrainReport::decode(&bad), None);
            assert!(derive_kind_known(DERIVE_KIND_PROCESS_DRAIN));
        }
    }
}
