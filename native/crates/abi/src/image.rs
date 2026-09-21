//! Image ABI — the loader wire types (`res/IMAGE.md`; epic #291, P1 #292).
//!
//! What crosses a boundary in the image subsystem, and nothing else:
//!
//! - [`ImageName`] — the `Acquire`/`Register` key (name → capability is the
//!   whole service protocol, the `fuchsia.ldsvc` lesson).
//! - [`ModuleDescriptor`] — what the image service *says* about a
//!   registered image: segment geometry, dynamic-section offsets, TLS
//!   shape.  **Data only** — authority is the granted lease caps, never
//!   descriptor fields.  Rides service replies *and* the parent-written
//!   [`LdBlock`] (`.kore.ld`).
//! - [`ModuleRecord`]/[`ModulePage`] — the **passive ABI** (`res/IMAGE.md`
//!   §8): the read-only, layout-frozen record of what the startup linker
//!   did, left behind for debuggers / `dlsym` / out-of-process linking.
//!   The linker's live state dies with the stub; this page is the seam.
//! - [`ImageRefusal`] — the service's closed refusal vocabulary.
//!
//! Everything here follows the [`BootBlock`](crate::task::BootBlock)
//! discipline: `#[repr(C)]`, little-endian, unaligned-safe, size-pinned,
//! **no parser** (the layout is the ABI; validated views refuse rather
//! than interpret), and post-freeze changes are additive only.
//!
//! Offsets in a descriptor are **image-relative** (`p_vaddr` space, base
//! subtracted); a consumer rebases against the load base *it* chose —
//! per-task ASLR means no absolute address is ever meaningful on this
//! wire.  The [`ModuleRecord`], by contrast, records the *chosen*
//! absolute placement: the descriptor is the promise, the record is the
//! outcome.

use zerocopy::byteorder::little_endian::{U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

/// Image-ABI version stamp ([`ModuleDescriptor::abi`], [`LdBlock::abi`],
/// [`ModulePage::abi`]).  A zeroed block reads `0` and refuses.
pub const IMAGE_ABI: u16 = 3;

/// Bytes the startup linker reserves **at the thread pointer** for the
/// runtime's per-thread block — the loader ⇄ runtime half of the TLS
/// contract.
///
/// A thread pointer has two claimants: the ELF TLS ABI, which owns the
/// static block, and `kore-rt`, whose per-thread runtime block is what the
/// trap transport reads its scratch VA out of.  A linked program used to be
/// able to reach one or the other and not both, because each claimed the
/// register for a region of its own.  They coexist instead by the linker
/// reserving this much room at `tp` and the runtime living there:
///
/// - **Variant 2** (x86_64): the static blocks sit *below* `tp` at negative
///   offsets, so the reserve extends the TCB upward.  Its first eight bytes
///   remain the psABI self-pointer, which is also the runtime block's first
///   field.
/// - **Variant 1** (aarch64, riscv64): the static blocks sit *above* `tp`
///   after a reserved header, and the header's size is the linker's to
///   choose because it is the linker that resolves every TPREL offset.  The
///   reserve simply grows that header.
///
/// Either way the runtime block is exactly `[tp, tp + RUNTIME_TCB_BYTES)`
/// on every architecture, so nothing per-arch reads the thread pointer
/// differently.  One page is what the founder maps for the same block on a
/// task the linker never touched; `kore-rt` asserts its block fits.
pub const RUNTIME_TCB_BYTES: u64 = 4096;

/// Launch profile: the task requests one federated telemetry snapshot.
/// Profiles are archive-authenticated metadata, not executable-name policy.
pub const PROFILE_TELEMETRY: u32 = 1 << 0;
/// Launch profile: exec provisions one private recyclable filesystem face.
///
/// The bit authorizes *which service face* accompanies the image. It does not
/// identify an executable by name and conveys no filesystem authority by
/// itself: exec still constructs a fresh pair and fs authenticates the
/// resulting peer generation before the task is founded.
pub const PROFILE_FILESYSTEM: u32 = 1 << 1;
/// Launch profile: the caller may delegate one affine endpoint that a
/// filesystem executable can bind to a newly resolved structural cwd view.
/// The bit grants no filesystem authority and does not select a path; it only
/// permits that caller-owned context member to occupy the generic auxiliary
/// lane.
pub const PROFILE_FS_VIEW_BINDER: u32 = 1 << 2;
/// Launch profile: the caller may delegate one non-invocable task-view target
/// endpoint to a filesystem executable. The endpoint selects only where an
/// attenuated FS grant is installed; it cannot act as that task's cwd.
pub const PROFILE_FS_GRANT_TARGET: u32 = 1 << 3;
/// Every profile bit this ABI version understands.
/// The launcher `ChannelSpace` grant (`res/SPMD.md` §3): the child
/// receives a cuttable, revocable sub-space of exec's own rendezvous
/// allocator at `role::CHANNEL_SPACE` — what lets it mint doorbell
/// channels for a region's star.  Standalone: no co-requirement.
pub const PROFILE_CHANNELS: u32 = 1 << 4;
/// Launch profile: the founder carves one bounded heap-supply sub-lease into
/// the child at `role::HEAP@0`.
///
/// This archive-authenticated bit is a request for that launch shape, not heap
/// authority by itself. Authority exists only in the affine lease that the
/// founder actually derives and delivers; an image name or descriptor cannot
/// manufacture it.
pub const PROFILE_HEAP: u32 = 1 << 5;
/// Launch profile: the founder and child receive the two ends of one bounded
/// interactive byte-stream window.
///
/// The bit is authenticated metadata, not an ambient terminal capability: the
/// parent retains its end and decides whether and where to relay it. Heap is an
/// independent request: a bounded, allocation-free interactive program is a
/// valid process shape even though Lua itself requests both resources.
pub const PROFILE_INTERACTIVE: u32 = 1 << 6;
/// Launch profile: the launcher may interpret a successful child result as
/// one bounded request against its own retained window-controller state.
///
/// The child receives no compositor pair, session owner, or task token. This
/// archive-authenticated bit selects only the launcher's broker transition;
/// every returned `(slot, generation, peer)` is inert and must be revalidated
/// against authority the launcher still holds.
pub const PROFILE_WINDOW_CONTROL: u32 = 1 << 7;
/// Launch profile: the child receives a **power face** — a private view of
/// the power service's request ring and a send onto its doorbell — and may
/// therefore ask for a machine transition.
///
/// This is the only way to reach `SystemPower` from a command, and it is
/// archive-authenticated: an image not packed with this bit cannot acquire
/// the face by asking, and the launcher does not hold one to lend. The
/// face is derived per launch and dies with the child, so authority to
/// power the machine off is never resident in a long-lived task.
pub const PROFILE_SYSTEM_POWER: u32 = 1 << 8;
/// Launch profile: the launcher asks the frame service for its facts — the
/// link's (`Query`) and the bound radio's (`Radio`) — and publishes them as
/// the process page's [`NetSnapshot`](crate::monitor::NetSnapshot).
///
/// A diagnostic request, not network authority: the child receives no face
/// and cannot open a vif, transmit, or receive. Exec alone holds the frame
/// service's pair, issues the two read-only verbs, and stamps the answers
/// before the child starts — the filesystem snapshot's shape exactly.
pub const PROFILE_NETWORK: u32 = 1 << 9;
/// The program hosts an **execution engine** — an interpreter or a loaded
/// native image — whose working set exceeds the ordinary launch arena.
///
/// This names a *class*, not a size: the launcher owns the numbers
/// (`kore_svc_exec::founder::LaunchPlan`), exactly as it already decides
/// whether to charge a launch for the trailing heap interval at all from
/// [`PROFILE_HEAP`]. A bit rather than a per-row byte count because the
/// profile word already crosses exec's wire, authenticated with the image,
/// while a new size field would widen the reply record. Meaningful only
/// beside [`PROFILE_HEAP`]: an engine with no heap has nowhere to run.
pub const PROFILE_ENGINE: u32 = 1 << 11;
/// Launch profile: exec provisions one private recyclable accelerator face.
///
/// The bit grants no device authority by itself. Exec admits a fresh service
/// peer for this launch, and the accelerator host binds every object created
/// through that peer to its exact recyclable generation. The face disappears
/// with the task and cannot address another application's contexts, buffers,
/// programs, queues, or events.
pub const PROFILE_ACCELERATOR: u32 = 1 << 10;
/// Launch profile: exec provisions one private recyclable face on the **GPU**
/// accelerator row (the Intel Xe row), at its own per-process slots beside
/// [`PROFILE_ACCELERATOR`]'s NPU face.  Same authority shape as that bit: a
/// fresh service peer per launch, bound to its recyclable generation, gone
/// with the task.  Best-effort where the NPU face is not: a machine whose
/// GPU row is parked (no GuC, no Intel GPU) still launches the process, with
/// the face's slots vacant, so an interpreter carrying this bit runs
/// everywhere and answers "unavailable" where it must.
///
/// Bit 11 is taken by the engine launch plan on a sibling branch; this is
/// bit 12 so the two merge without renumbering.
pub const PROFILE_GPU: u32 = 1 << 12;

pub const PROFILE_KNOWN: u32 = PROFILE_TELEMETRY
    | PROFILE_FILESYSTEM
    | PROFILE_FS_VIEW_BINDER
    | PROFILE_FS_GRANT_TARGET
    | PROFILE_CHANNELS
    | PROFILE_HEAP
    | PROFILE_INTERACTIVE
    | PROFILE_WINDOW_CONTROL
    | PROFILE_SYSTEM_POWER
    | PROFILE_NETWORK
    | PROFILE_ACCELERATOR
    | PROFILE_ENGINE
    | PROFILE_GPU;

/// Validate the profile vocabulary and its dependencies: both filesystem
/// auxiliary endpoint kinds are meaningful only beside an actual FS face.
pub const fn profile_valid(profile: u32) -> bool {
    profile & !PROFILE_KNOWN == 0
        && (profile & PROFILE_FS_VIEW_BINDER == 0 || profile & PROFILE_FILESYSTEM != 0)
        && (profile & PROFILE_FS_GRANT_TARGET == 0 || profile & PROFILE_FILESYSTEM != 0)
        && (profile & PROFILE_ENGINE == 0 || profile & PROFILE_HEAP != 0)
}

/// Segment ceiling per image — up to eight `PT_LOAD`s plus one `PT_TLS`
/// initialization view (`res/IMAGE.md` §2).  Kinds may repeat: in
/// particular, lld emits distinct writable LOADs for RELRO and ordinary
/// data.  `Acquire` delivers one lease per entry, by construction.
pub const MAX_SEGMENTS: usize = 9;

/// Maximum distinct libraries in one executable generation's validated,
/// transitively frozen `DT_NEEDED` closure. The startup linker is bounded by
/// the same no-allocation image vocabulary; exceeding this is a registration
/// or installation refusal, never a partial closure.
pub const MAX_DEPENDENCIES: usize = 8;

/// Name capacity — a fixed inline key, never a heap string.
pub const NAME_BYTES: usize = 64;

/// Authenticated metadata for one request-side immutable image source.
///
/// The source/version pair is inert correlation data from the filesystem's
/// immutable-version authority.  `len` and `digest` bind this record to the
/// exact bytes transferred beside the registration request; `profile` is the
/// installer's explicit service-authority policy.  None of these fields select
/// an image or confer launch authority.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct ImageSourceMetadata {
    pub source: U64,
    pub version: U64,
    pub len: U64,
    pub digest: [u8; 32],
    pub profile: U32,
    /// Must remain zero.
    pub reserved: U32,
}

impl ImageSourceMetadata {
    pub const fn empty() -> Self {
        Self {
            source: U64::new(0),
            version: U64::new(0),
            len: U64::new(0),
            digest: [0; 32],
            profile: U32::new(0),
            reserved: U32::new(0),
        }
    }

    pub fn validate(&self) -> bool {
        self.source.get() != 0
            && self.version.get() != 0
            && self.len.get() != 0
            && self.digest != [0; 32]
            && profile_valid(self.profile.get())
            && self.reserved.get() == 0
    }
}

/// Reserved image-name namespace for boot constructors. Members in this
/// namespace may be acquired by the boot chain, but the image service will
/// not mint a child loader face for them. This keeps constructor artifacts
/// out of the ordinary executable namespace without teaching a shell or the
/// scheduler individual program names.
pub const BOOTSTRAP_IMAGE_PREFIX: &str = "boot:";

/// Segment kind + access vocabulary ([`SegmentDesc::kind`],
/// [`SegmentDesc::access`]).  Kinds are 1-based so an all-zero
/// (unseeded) descriptor entry is invalid, not "text".  Access is the
/// R/W/X bit encoding the session share verb adopts wire-wide (#297
/// item c) — defined once, here.
pub mod seg {
    /// Executable text — an executable lease, shared `R|X`, never `W`.
    pub const KIND_TEXT: u8 = 1;
    /// Read-only data — shared `R` in place.
    pub const KIND_RODATA: u8 = 2;
    /// Data template (`.data` + RELRO image) — shared `R`, **copied** by
    /// the consumer into its private RW region (`filesz` bytes; `memsz`
    /// adds `.bss`).
    pub const KIND_DATA: u8 = 3;
    /// TLS initialization image (`PT_TLS`) — shared `R`, copied
    /// per-thread into static TLS.
    pub const KIND_TLS: u8 = 4;

    /// Readable.
    pub const ACCESS_R: u8 = 1 << 0;
    /// Writable — never set on any image segment (`res/IMAGE.md` §2:
    /// image leases install no `W` leaves; the consumer's *copy* is
    /// writable, the lease is not).
    pub const ACCESS_W: u8 = 1 << 1;
    /// Executable — legal only on [`KIND_TEXT`], only off an executable lease
    /// (`res/IMAGE.md` §6).
    pub const ACCESS_X: u8 = 1 << 2;
}

/// The fixed inline image name: UTF-8 bytes, zero-padded.  The service's
/// lookup key and nothing more — no paths, no versions-in-the-name (a
/// re-`Register` shadows; `res/IMAGE.md` §3).  8-bit-clean on the wire;
/// [`as_str`](Self::as_str) is the validated read.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct ImageName(pub [u8; NAME_BYTES]);

impl ImageName {
    /// The empty (invalid) name — all zero.
    pub const fn empty() -> Self {
        Self([0; NAME_BYTES])
    }

    /// Build from a `&str`; refuses (`None`) when it does not fit or is
    /// empty, or embeds NUL (NUL is the padding, not data).
    pub fn new(name: &str) -> Option<Self> {
        let bytes = name.as_bytes();
        if bytes.is_empty() || bytes.len() > NAME_BYTES || bytes.contains(&0) {
            return None;
        }
        let mut buf = [0u8; NAME_BYTES];
        buf[..bytes.len()].copy_from_slice(bytes);
        Some(Self(buf))
    }

    /// Live length (bytes before the zero padding).
    pub fn len(&self) -> usize {
        self.0.iter().position(|&b| b == 0).unwrap_or(NAME_BYTES)
    }

    /// Whether the name is unseeded/empty.
    pub fn is_empty(&self) -> bool {
        self.0[0] == 0
    }

    /// The validated string view; `None` for empty or non-UTF-8 bytes
    /// (a corrupt or hostile wire value refuses, it never panics).
    pub fn as_str(&self) -> Option<&str> {
        if self.is_empty() {
            return None;
        }
        core::str::from_utf8(&self.0[..self.len()]).ok()
    }

    /// Whether this name belongs to the reserved boot-constructor namespace.
    pub fn is_bootstrap_only(&self) -> bool {
        self.as_str()
            .is_some_and(|name| name.starts_with(BOOTSTRAP_IMAGE_PREFIX))
    }

    /// FNV-1a of the live bytes — the [`ModuleRecord::name_hash`]
    /// identity.  A *correlation* value (like a task id), never an
    /// authority: acting on a module always takes a capability.
    pub fn fnv1a(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in &self.0[..self.len()] {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }
}

/// One segment's geometry, image-relative (`res/IMAGE.md` §2).  The
/// consumer maps the granted lease whole — there are no file offsets on
/// this wire, because the lease *is* the segment's bytes (the in-place
/// carve).  32 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct SegmentDesc {
    /// Image-relative virtual address (`p_vaddr` − image base), page-
    /// aligned (`-z separate-code`; validated at `Register`).
    pub vaddr: U64,
    /// In-memory size.  For [`seg::KIND_DATA`], `memsz − filesz` is the
    /// `.bss` tail the consumer zero-allocates.
    pub memsz: U64,
    /// Bytes present in the lease (`p_filesz`); `<= memsz`.
    pub filesz: U64,
    /// [`seg`] kind discriminant; zero = vacant entry.
    pub kind: u8,
    /// [`seg`] access bits the consumer's share may request.
    pub access: u8,
    /// Reserved; zero.
    pub reserved: [u8; 6],
}

impl SegmentDesc {
    /// A vacant (all-zero) entry.
    pub const fn vacant() -> Self {
        Self {
            vaddr: U64::new(0),
            memsz: U64::new(0),
            filesz: U64::new(0),
            kind: 0,
            access: 0,
            reserved: [0; 6],
        }
    }

    /// The exact page-rounded extent backing this segment's granted lease.
    ///
    /// `Register` carves immutable and data LOADs at their normalized mapping
    /// span, which is already `memsz`. TLS is copy-only: its small logical
    /// template begins at the original ELF page offset, recoverable from
    /// image-relative `vaddr` because the image base is page-aligned. Thus a
    /// consumer can derive the mapping shape from the frozen descriptor
    /// instead of issuing a geometry syscall for every segment. The eventual
    /// share remains authoritative: a descriptor/lease mismatch is refused by
    /// the pager before any leaf is installed.
    pub fn lease_len(&self) -> Option<u64> {
        const PAGE: u64 = 4096;
        match self.kind {
            seg::KIND_TLS => (self.vaddr.get() & (PAGE - 1))
                .checked_add(self.filesz.get())?
                .checked_next_multiple_of(PAGE),
            seg::KIND_TEXT | seg::KIND_RODATA | seg::KIND_DATA
                if self.memsz.get() != 0 && self.memsz.get() % PAGE == 0 =>
            {
                Some(self.memsz.get())
            }
            _ => None,
        }
    }
}

/// TLS geometry (`PT_TLS`; `res/IMAGE.md` §7).  Zero `memsz` = no TLS.
/// The stub owns the layout (static block, IE/LE only); the kernel never
/// reads this.  24 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct TlsDesc {
    /// Total per-thread block size (`p_memsz`).
    pub memsz: U64,
    /// Required alignment (`p_align`).
    pub align: U64,
    /// Initialization-image bytes (`p_filesz`; the [`seg::KIND_TLS`]
    /// lease's live prefix).  The `memsz − init_sz` tail is `tbss`.
    pub init_sz: U64,
}

/// Dynamic-section geometry — every image-relative range the eager
/// relocation pass reads (`res/IMAGE.md` §7's tag list).  A range the
/// image lacks is `(0, 0)`.  `dynamic_off/sz` is `PT_DYNAMIC` itself:
/// DT_NEEDED name walking is the *stub's* job over its own mapping (the
/// service never walks graphs), so the raw section must be reachable.
/// 152 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct DynDesc {
    /// `PT_DYNAMIC` range.
    pub dynamic_off: U64,
    pub dynamic_sz: U64,
    /// `DT_RELA`/`DT_RELASZ`.
    pub rela_off: U64,
    pub rela_sz: U64,
    /// `DT_RELR`/`DT_RELRSZ` (packed relative; optional).
    pub relr_off: U64,
    pub relr_sz: U64,
    /// `DT_JMPREL`/`DT_PLTRELSZ` (eager under `-z now`).
    pub jmprel_off: U64,
    pub jmprel_sz: U64,
    /// `DT_SYMTAB` (entry size is fixed 24 for ELF64 — `DT_SYMENT` is
    /// validated at `Register`, not carried).
    pub symtab_off: U64,
    /// `DT_STRTAB`/`DT_STRSZ`.
    pub strtab_off: U64,
    pub strtab_sz: U64,
    /// `DT_GNU_HASH` (the one lookup structure; SysV hash is ignored).
    pub gnu_hash_off: U64,
    /// `DT_INIT_ARRAY`/`SZ`, `DT_FINI_ARRAY`/`SZ`.
    pub init_array_off: U64,
    pub init_array_sz: U64,
    pub fini_array_off: U64,
    pub fini_array_sz: U64,
    /// `PT_GNU_RELRO` (image-relative) — the window the stub re-protects
    /// read-only after relocation (`res/IMAGE.md` §7).  `(0, 0)` when the
    /// image carries none.  The seal itself is the stub's verb; this is
    /// how the window crosses the `Acquire` wire.
    pub relro_off: U64,
    pub relro_sz: U64,
    /// Mirrored `DT_FLAGS` (validated `BIND_NOW`; `STATIC_TLS` honored;
    /// `TEXTREL` refused at `Register` — this copy is for the stub's
    /// defense-in-depth re-check).
    pub flags: U64,
}

/// Image kind ([`ModuleDescriptor::kind`]).  1-based; zero refuses.
pub mod kind {
    /// A dynamically-linked PIE executable (`ET_DYN` with an entry).
    /// Addresses in its descriptor are **image-relative**: the consumer
    /// chooses a load base and adds it.
    pub const EXECUTABLE: u8 = 1;
    /// A shared library (`ET_DYN`, entry meaningless).  Image-relative,
    /// as [`EXECUTABLE`].
    pub const LIBRARY: u8 = 2;
    /// A fixed-address executable (`ET_EXEC`, zero relocations) — every
    /// founded row and the linker stub itself.
    ///
    /// **Addresses in its descriptor are ABSOLUTE**: its link address is
    /// its runtime address, so there is no base to add and no relocation
    /// to apply.
    ///
    /// This kind exists so the descriptor says which convention it uses
    /// rather than leaving the consumer to infer it from the image type
    /// it cannot see.  The alternative — writing absolute addresses into
    /// a struct documented "image-relative throughout" and relying on
    /// out-of-band knowledge — type-checks perfectly and is wrong exactly
    /// once, in a consumer that jumps to what it read.
    pub const STATIC_EXECUTABLE: u8 = 3;

    /// Whether a descriptor of this kind carries **absolute** addresses.
    /// Unknown kinds answer `false`: a consumer that cannot name the
    /// convention must not assume the one that needs no base.
    pub const fn is_absolute(kind: u8) -> bool {
        matches!(kind, STATIC_EXECUTABLE)
    }

    /// Whether this is a kind at all — the vocabulary, in one place.
    ///
    /// [`ModuleDescriptor::validate`](super::ModuleDescriptor::validate)
    /// spelled the list out instead, and adding [`STATIC_EXECUTABLE`]
    /// did not update it: the founder wrote a descriptor its own crate
    /// then called invalid.  It went unnoticed because the one consumer
    /// on that path reads the stamp and [`is_absolute`] directly rather
    /// than validating first.  A predicate here makes the next kind one
    /// edit, beside the variant it adds.
    pub const fn is_known(kind: u8) -> bool {
        matches!(kind, EXECUTABLE | LIBRARY | STATIC_EXECUTABLE)
    }
}

/// What the image service says about one registered image — the
/// `Registered`/`Acquired` reply payload and the [`LdBlock`] body.
/// **Data only** (authority = the lease-cap grants).  Addresses are
/// image-relative or absolute according to [`kind`](Self::kind) — see
/// [`kind::is_absolute`].  488 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct ModuleDescriptor {
    /// [`IMAGE_ABI`] stamp; zero refuses.
    pub abi: U16,
    /// [`kind`] discriminant.
    pub kind: u8,
    /// Live entries in `segs`.
    pub seg_count: u8,
    /// `e_machine` — the consumer's sanity cross-check (the service
    /// validated it; a mismatch here is corruption, not negotiation).
    pub machine: U16,
    /// `DT_NEEDED` count — lets the stub size its walk before touching
    /// `PT_DYNAMIC`, and bounds a hostile dynamic section.
    pub needed_count: U16,
    /// Archive-authenticated launch profile bits ([`PROFILE_KNOWN`]).
    pub profile: U32,
    /// Reserved; must be zero.
    pub reserved: U32,
    /// Entry point; zero for a library.
    ///
    /// **Whose meaning [`kind`](Self::kind) decides** — image-relative
    /// for [`kind::EXECUTABLE`], absolute for
    /// [`kind::STATIC_EXECUTABLE`] ([`kind::is_absolute`]).  Read the
    /// kind before the address.
    pub entry: U64,
    /// Segment table; entries past `seg_count` are vacant.
    pub segs: [SegmentDesc; MAX_SEGMENTS],
    /// TLS geometry (zero `memsz` = none).
    pub tls: TlsDesc,
    /// Dynamic-section geometry.
    pub dynz: DynDesc,
}

impl ModuleDescriptor {
    /// An unseeded descriptor (refuses validation until stamped).
    pub const fn empty() -> Self {
        Self {
            abi: U16::new(0),
            kind: 0,
            seg_count: 0,
            machine: U16::new(0),
            needed_count: U16::new(0),
            profile: U32::new(0),
            reserved: U32::new(0),
            entry: U64::new(0),
            segs: [SegmentDesc::vacant(); MAX_SEGMENTS],
            tls: TlsDesc {
                memsz: U64::new(0),
                align: U64::new(0),
                init_sz: U64::new(0),
            },
            dynz: DynDesc {
                dynamic_off: U64::new(0),
                dynamic_sz: U64::new(0),
                rela_off: U64::new(0),
                rela_sz: U64::new(0),
                relr_off: U64::new(0),
                relr_sz: U64::new(0),
                jmprel_off: U64::new(0),
                jmprel_sz: U64::new(0),
                symtab_off: U64::new(0),
                strtab_off: U64::new(0),
                strtab_sz: U64::new(0),
                gnu_hash_off: U64::new(0),
                init_array_off: U64::new(0),
                init_array_sz: U64::new(0),
                fini_array_off: U64::new(0),
                fini_array_sz: U64::new(0),
                relro_off: U64::new(0),
                relro_sz: U64::new(0),
                flags: U64::new(0),
            },
        }
    }

    /// Structural sanity — the *consumer-side* validated view (the
    /// [`Bootstrap`](crate::task::Bootstrap) pattern): stamp, kind,
    /// bounded segment count, and a consumer-safe shape on every live
    /// entry. Content truth (ranges inside the image) was the service's
    /// `Register` gate; this cheap second gate refuses a stale, zeroed, or
    /// corrupt block before a world can issue an unaligned mapping verb.
    pub fn validate(&self) -> bool {
        const PAGE: u64 = 4096;
        let count = self.seg_count as usize;
        if self.abi.get() != IMAGE_ABI
            || !kind::is_known(self.kind)
            || !profile_valid(self.profile.get())
            || self.reserved.get() != 0
            || count == 0
            || count > MAX_SEGMENTS
        {
            return false;
        }
        let live = &self.segs[..count];
        if !live.iter().all(|s| {
            let geometry = s.memsz.get() != 0
                && s.filesz.get() <= s.memsz.get()
                && s.vaddr.get().checked_add(s.memsz.get()).is_some();
            let access = match s.kind {
                seg::KIND_TEXT => s.access == seg::ACCESS_R | seg::ACCESS_X,
                seg::KIND_RODATA | seg::KIND_DATA | seg::KIND_TLS => s.access == seg::ACCESS_R,
                _ => false,
            };
            let mapping =
                s.kind == seg::KIND_TLS || (s.vaddr.get() % PAGE == 0 && s.memsz.get() % PAGE == 0);
            geometry && access && mapping && s.lease_len().is_some()
        }) {
            return false;
        }
        if live.iter().filter(|s| s.kind == seg::KIND_TLS).count() > 1 {
            return false;
        }
        let mut data_start: Option<u64> = None;
        let mut data_end: u64 = 0;
        let mut data_bytes: u64 = 0;
        for s in live.iter().filter(|s| s.kind == seg::KIND_DATA) {
            data_start = Some(data_start.map_or(s.vaddr.get(), |at| at.min(s.vaddr.get())));
            data_end = data_end.max(s.vaddr.get() + s.memsz.get());
            let Some(total) = data_bytes.checked_add(s.memsz.get()) else {
                return false;
            };
            data_bytes = total;
        }
        if let Some(start) = data_start
            && data_end.checked_sub(start) != Some(data_bytes)
        {
            // The linker realizes all data LOADs as one private mapping.
            // Requiring page-contiguity prevents that union from making an
            // ELF hole accessible as anonymous writable memory.
            return false;
        }

        // Mapping entries must be disjoint. TLS is a copy-only view into a
        // data template and is therefore intentionally excluded.
        for (i, a) in live.iter().enumerate() {
            if a.kind == seg::KIND_TLS {
                continue;
            }
            let a_end = a.vaddr.get() + a.memsz.get();
            for b in &live[i + 1..] {
                if b.kind == seg::KIND_TLS {
                    continue;
                }
                let b_end = b.vaddr.get() + b.memsz.get();
                if a.vaddr.get() < b_end && b.vaddr.get() < a_end {
                    return false;
                }
            }
        }
        true
    }

    /// The live prefix of `segs` (bounded by `seg_count`, clamped to the
    /// table) — the [`Bootstrap`](crate::task::Bootstrap) bounded-read
    /// discipline: consumers walk this, never index past the live count.
    pub fn live_segs(&self) -> &[SegmentDesc] {
        let n = (self.seg_count as usize).min(MAX_SEGMENTS);
        self.segs.get(..n).unwrap_or(&[])
    }
}

/// The `.kore.ld` block (`res/IMAGE.md` §4) — the loader metadata the
/// **parent** writes into the child's private data page beside
/// `.kore.boot`, and `found_user`'s `arg` points at: the executable's
/// descriptor and the name under which the stub acquires it. Caps ride the
/// boot block (`role::LOADER`, `role::IMAGE`); this is the data half. 552
/// bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct LdBlock {
    /// [`IMAGE_ABI`] stamp; an unseeded static reads zero and refuses.
    pub abi: U16,
    /// Reserved; zero.
    pub flags: U16,
    /// Reserved; zero.
    pub reserved: U32,
    /// The executable name the parent acquired. [`ModuleDescriptor`] carries
    /// no name by design; this keeps the real world's name-match gate honest.
    pub exe_name: ImageName,
    /// The executable's descriptor (from the parent's `Acquire`).
    pub exe: ModuleDescriptor,
}

impl LdBlock {
    /// A stamped block around an exe descriptor — the parent's write.
    pub const fn new(exe_name: ImageName, exe: ModuleDescriptor) -> Self {
        Self {
            abi: U16::new(IMAGE_ABI),
            flags: U16::new(0),
            reserved: U32::new(0),
            exe_name,
            exe,
        }
    }
}

/// One placed segment in the passive ABI — **absolute** addresses (the
/// chosen placement) + the backing lease's resource id for provenance.
/// 24 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct ModuleSeg {
    /// Absolute VA the segment landed at.
    pub vaddr: U64,
    /// Mapped size.
    pub size: U64,
    /// The backing lease's resource id (correlation/provenance — not
    /// authority; the cap chain is the authority record).
    pub lease_id: U32,
    /// [`seg`] kind.
    pub kind: u8,
    /// [`seg`] access actually installed.
    pub access: u8,
    /// Reserved; zero.
    pub reserved: [u8; 2],
}

impl ModuleSeg {
    /// A vacant (all-zero) entry — [`SegmentDesc::vacant`]'s passive-ABI
    /// sibling.
    pub const fn vacant() -> Self {
        Self {
            vaddr: U64::new(0),
            size: U64::new(0),
            lease_id: U32::new(0),
            kind: 0,
            access: 0,
            reserved: [0; 2],
        }
    }
}

/// One loaded module in the passive ABI (`res/IMAGE.md` §8).  248 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct ModuleRecord {
    /// [`ImageName::fnv1a`] of the module's name.
    pub name_hash: U64,
    /// The chosen load base.
    pub base: U64,
    /// Absolute entry (executable) or zero.
    pub entry: U64,
    /// Live entries in `segs`.
    pub seg_count: u8,
    /// [`kind`] discriminant.
    pub kind: u8,
    /// Reserved; zero.
    pub reserved: [u8; 6],
    /// Placed segments.
    pub segs: [ModuleSeg; MAX_SEGMENTS],
}

impl ModuleRecord {
    /// A vacant (all-zero) record — the value entries past
    /// [`ModulePage::count`] hold ([`CapRecord::vacant`]'s shape:
    /// unstamped, distinguishable from any live record by its zero
    /// `kind`).
    ///
    /// [`CapRecord::vacant`]: crate::cap::CapRecord::vacant
    pub const fn vacant() -> Self {
        Self {
            name_hash: U64::new(0),
            base: U64::new(0),
            entry: U64::new(0),
            seg_count: 0,
            kind: 0,
            reserved: [0; 6],
            segs: [ModuleSeg::vacant(); MAX_SEGMENTS],
        }
    }

    /// The live prefix of `segs` (bounded by `seg_count`, clamped to the
    /// table) — the [`Bootstrap`](crate::task::Bootstrap) bounded-read
    /// discipline: consumers never index past the live count.
    pub fn live_segs(&self) -> &[ModuleSeg] {
        let n = (self.seg_count as usize).min(MAX_SEGMENTS);
        self.segs.get(..n).unwrap_or(&[])
    }
}

/// Record capacity of one passive-ABI page: header + 16 × 248 B fits a
/// 4 KiB page and exactly covers the startup linker's bounded module set.
pub const MAX_MODULE_RECORDS: usize = 16;

/// Where the passive-ABI page ([`ModulePage`]) lives in a task's address
/// space: the **last page of the `geom::LINK` window**, given that
/// window's `(base, len)`.  The startup stub places every module below it
/// and writes the page here; a late loader (`res/IMAGE.md` §8 — "a
/// separate library consuming the record") reads it back from the same
/// address, because nothing else records where the page went.  One
/// function cited by both sides, so the two cannot drift; a window
/// shorter than a page yields `None`.
pub const fn module_page_va(link_base: u64, link_len: u64) -> Option<u64> {
    if link_len < MODULE_PAGE_BYTES {
        return None;
    }
    link_base.checked_add(link_len - MODULE_PAGE_BYTES)
}

/// The passive page's span: one page, the unit the stub allocates it in.
pub const MODULE_PAGE_BYTES: u64 = 4096;

/// The passive-ABI page — written once by the startup stub, then sealed
/// read-only.  The future seam for debuggers, `dlsym`, and
/// out-of-process linking; the stub's live state is *not* ABI, this is.
/// 3976 bytes (fits one page).
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned)]
pub struct ModulePage {
    /// [`IMAGE_ABI`] stamp.
    pub abi: U16,
    /// Reserved; zero.
    pub flags: U16,
    /// Live records.
    pub count: U16,
    /// How much of the link window the startup placement consumed, in
    /// pages counted **down from this page's own address**; `0` means
    /// *unrecorded* (a page stamped before this field existed, or a world
    /// that does not report placement).
    ///
    /// Exists because [`high_water`](Self::high_water) covers only the
    /// segments the records name, and the startup stub places more than
    /// that: the copied data/TLS template views and the static TLS block
    /// are placed into the link window right after the image but are not
    /// module segments. Once an executable's image grew past a 64 KiB
    /// boundary, the late loader's first window (the next 64 KiB line
    /// above the records) landed inside that placed-but-unnamed range and
    /// the kernel refused it `AlreadyMapped`. The unit is pages below the
    /// page rather than an absolute address because that fits the sixteen
    /// bits this word already had — the same layout, the same
    /// [`IMAGE_ABI`], and every existing page reads as unrecorded. A link
    /// window wider than `u16::MAX` pages refuses at link time
    /// (`kore_ld::link`), never truncates here.
    pub placed: U16,
    /// The module table; entries past `count` are zero.
    pub records: [ModuleRecord; MAX_MODULE_RECORDS],
}

impl ModulePage {
    /// An empty, correctly-stamped page (the stub's starting value).
    pub const fn new() -> Self {
        Self {
            abi: U16::new(IMAGE_ABI),
            flags: U16::new(0),
            count: U16::new(0),
            placed: U16::new(0),
            records: [ModuleRecord::vacant(); MAX_MODULE_RECORDS],
        }
    }

    /// Append a record; refuses (`false`) when full — bounded like every
    /// other emission, never silently dropped.
    pub fn push(&mut self, record: ModuleRecord) -> bool {
        let n = self.count.get() as usize;
        if n >= MAX_MODULE_RECORDS {
            return false;
        }
        self.records[n] = record;
        self.count.set(n as u16 + 1);
        true
    }

    /// Nothing recorded.
    pub fn is_empty(&self) -> bool {
        self.count.get() == 0
    }

    /// Record the startup placement high water: `pages` pages below this
    /// page's own address. See [`placed`](Self::placed).
    pub fn record_placed(&mut self, pages: u16) {
        self.placed = U16::new(pages);
    }

    /// One past the highest address the startup **placement** reached —
    /// template views and the TLS block included — given where this page
    /// lives (`module_page_va`). `None` when unrecorded. A late placer
    /// takes the larger of this and [`high_water`](Self::high_water);
    /// neither alone is the floor.
    pub fn placed_high_water(&self, module_page_va: u64) -> Option<u64> {
        let pages = u64::from(self.placed.get());
        if pages == 0 {
            return None;
        }
        module_page_va.checked_sub(pages.checked_mul(MODULE_PAGE_BYTES)?)
    }
    /// One past the highest address any recorded segment occupies,
    /// page-rounded — where a **late** loader (`res/IMAGE.md` §8) may start
    /// placing, *for the segments the records name*. The startup stub also
    /// places what no record names (the static TLS block); take the larger
    /// of this and [`placed_high_water`](Self::placed_high_water).  `None`
    /// for an empty page or on overflow (a lying record is refused, not
    /// clamped).
    pub fn high_water(&self) -> Option<u64> {
        let mut top: u64 = 0;
        let count = usize::from(self.count.get()).min(MAX_MODULE_RECORDS);
        for record in &self.records[..count] {
            for s in record.live_segs() {
                let end = s.vaddr.get().checked_add(s.size.get())?;
                top = top.max(end);
            }
        }
        if top == 0 {
            return None;
        }
        top.checked_add(MODULE_PAGE_BYTES - 1)
            .map(|v| v & !(MODULE_PAGE_BYTES - 1))
    }
}

impl Default for ModulePage {
    fn default() -> Self {
        Self::new()
    }
}

/// Why a passive-ABI page failed validation —
/// [`BootstrapError`](crate::task::BootstrapError)'s vocabulary, applied
/// to the [`ModulePage`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModulePageError {
    /// The ABI stamp is not [`IMAGE_ABI`] — a stale stub, or an unsealed
    /// (all-zero) page.
    BadAbi,
    /// `count` exceeds [`MAX_MODULE_RECORDS`] — a corrupt write.
    BadCount,
}

/// The **validated view** over a sealed [`ModulePage`] — the read door
/// of the passive ABI ([`Bootstrap`](crate::task::Bootstrap)'s shape:
/// reading is through the view, so an unchecked page cannot reach a
/// consumer).  Zero-copy and allocation-free; what a debugger, `dlsym`,
/// or an out-of-process linker walks.
#[derive(Clone, Copy, Debug)]
pub struct ModuleTable<'a> {
    records: &'a [ModuleRecord],
}

impl<'a> ModuleTable<'a> {
    /// Validate `page` (ABI stamp + count bound) and take the view.
    pub fn of(page: &'a ModulePage) -> Result<Self, ModulePageError> {
        if page.abi.get() != IMAGE_ABI {
            return Err(ModulePageError::BadAbi);
        }
        let n = page.count.get() as usize;
        if n > MAX_MODULE_RECORDS {
            return Err(ModulePageError::BadCount);
        }
        Ok(Self {
            records: &page.records[..n],
        })
    }

    /// The live records, in link order (the executable is record 0).
    pub fn records(&self) -> impl Iterator<Item = &'a ModuleRecord> {
        self.records.iter()
    }

    /// The record whose [`name_hash`](ModuleRecord::name_hash) matches
    /// `hash` ([`ImageName::fnv1a`]), if the module was linked.
    pub fn record(&self, hash: u64) -> Option<&'a ModuleRecord> {
        self.records.iter().find(|r| r.name_hash.get() == hash)
    }
}

/// The image service's closed refusal vocabulary (`res/IMAGE.md` §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ImageRefusal {
    /// `Register` validation failed — malformed/hostile ELF, reject-list
    /// reloc present, unaligned segments, `TEXTREL`, missing `BIND_NOW`.
    /// A bad image never gets a lease.
    BadImage = 1,
    /// `Acquire`/`Retire` of a name never registered (or fully retired).
    Unknown = 2,
    /// The fixed image table is full — refused, counted, never dropped.
    TableFull = 3,
    /// Empty / oversized / non-UTF-8 name.
    BadName = 4,
    /// Caller lacks the gate the verb requires.
    Denied = 5,
    /// The request named a frozen tombstone discriminant
    /// ([`ImageReqWire::op_retired`](crate::svc::ImageReqWire::op_retired));
    /// answered, never silently dropped, and never revived.
    Retired = 6,
    /// The name is an archive DATA member ([`ARCHIVE_MEMBER_DATA`]): its
    /// bytes are exportable, but it has no segments, no descriptor, and no
    /// launch shape, so `Acquire`/`Describe` refuse it rather than answer
    /// an empty image.
    NotImage = 7,
}

// Frozen wire pins (#81; the crate-header wire-freeze discipline).
const _: () = {
    assert!(size_of::<ImageSourceMetadata>() == 64);
    use core::mem::size_of;
    assert!(size_of::<ImageName>() == NAME_BYTES);
    assert!(size_of::<SegmentDesc>() == 32);
    assert!(size_of::<TlsDesc>() == 24);
    assert!(size_of::<DynDesc>() == 152);
    assert!(size_of::<ModuleDescriptor>() == 8 + 16 + MAX_SEGMENTS * 32 + 24 + 152);
    assert!(size_of::<LdBlock>() == 8 + size_of::<ImageName>() + size_of::<ModuleDescriptor>());
    assert!(size_of::<ModuleSeg>() == 24);
    assert!(size_of::<ModuleRecord>() == 32 + MAX_SEGMENTS * 24);
    assert!(size_of::<ModulePage>() == 8 + MAX_MODULE_RECORDS * (32 + MAX_SEGMENTS * 24));
    // The passive-ABI page must fit one 4 KiB page.
    assert!(size_of::<ModulePage>() <= 4096);
    assert!(size_of::<ArchiveHeader>() == 16);
    assert!(size_of::<ArchiveEntry>() == 88);
    // `kind` took the former reserve's exact place: the last word.
    assert!(core::mem::offset_of!(ArchiveEntry, kind) == 84);
    // Zero must stay "image", or every committed v2 archive changes meaning.
    assert!(ARCHIVE_MEMBER_IMAGE == 0);
};

/// The embedded image archive's profile-bearing v2 magic (`"KIMGAR2\0"`) — the Limine
/// module the image service's store lease covers (`res/IMAGE.md` §3's
/// "boot-module / embedded-archive supply").
pub const ARCHIVE_MAGIC: [u8; 8] = *b"KIMGAR2\0";

/// The embedded archive's header — the member count, nothing more.  The
/// index follows contiguously; member blobs are each page-aligned after
/// the index.  The layout is the ABI (the BootBlock discipline); the
/// service's walk refuses rather than interprets.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct ArchiveHeader {
    /// [`ARCHIVE_MAGIC`], verbatim.
    pub magic: [u8; 8],
    /// Live [`ArchiveEntry`] records following the header.
    pub count: U32,
    /// Reserved; must be zero.
    pub reserved: U32,
}

impl ArchiveHeader {
    /// Wire size (repr(C), 16).
    pub const WIRE: usize = 16;
}

/// [`ArchiveEntry::kind`]: an ELF image — checked, carved, and acquirable
/// (`res/IMAGE.md` §3). Zero, so every archive packed before the field had a
/// name (it was a must-be-zero reserve) reads back as all images.
pub const ARCHIVE_MEMBER_IMAGE: u32 = 0;
/// [`ArchiveEntry::kind`]: an opaque DATA member (a `.wasm` module, a font, a
/// table) — bounds-checked only, never ELF-checked, never carved, never
/// launchable. The image service registers its window so `Export` can lend
/// the bytes read-only (the filesystem seed copies them into `share/`);
/// `Acquire` refuses it typed ([`ImageRefusal::NotImage`]). A data member
/// carries `profile == 0`: a launch profile on something that cannot launch
/// is a packing error, refused at registration.
pub const ARCHIVE_MEMBER_DATA: u32 = 1;

/// Whether `kind` is in the archive member vocabulary. An unknown kind fails
/// closed at registration rather than being treated as either shape.
pub const fn archive_member_kind_known(kind: u32) -> bool {
    matches!(kind, ARCHIVE_MEMBER_IMAGE | ARCHIVE_MEMBER_DATA)
}

/// One archive member's index record: where its bytes live in the blob.
///
/// The magic stays [`ARCHIVE_MAGIC`] (`KIMGAR2`) across the introduction of
/// [`kind`](Self::kind): the field occupies the former must-be-zero reserve,
/// and zero is [`ARCHIVE_MEMBER_IMAGE`], so every committed v2 archive is
/// already a valid archive of the extended format and the layout (88 bytes)
/// is unchanged. A reader that predates the field refuses a data member
/// (non-zero reserve) rather than mis-reading it, which is the fail-closed
/// direction; a new magic would have bought nothing but a flag day.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct ArchiveEntry {
    /// The name `Acquire` (images) or `Export` (both kinds) presents.
    pub name: ImageName,
    /// Byte offset into the archive blob (page-aligned).
    pub offset: U64,
    /// Byte length (the blob is page-rounded past this; an image's own
    /// ELF parse bounds what's validated, a data member's length is its
    /// exact exported length).
    pub len: U64,
    /// Launch profile bits copied into the checked module descriptor. Zero
    /// on a [`ARCHIVE_MEMBER_DATA`] member.
    pub profile: U32,
    /// [`ARCHIVE_MEMBER_IMAGE`] or [`ARCHIVE_MEMBER_DATA`] (formerly a
    /// must-be-zero reserve; see the type docs for why the magic held).
    pub kind: U32,
}

impl ArchiveEntry {
    /// Wire size (repr(C), 88).
    pub const WIRE: usize = 88;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_roundtrip_and_bounds() {
        let n = ImageName::new("kore-rt.runtime").unwrap();
        assert_eq!(n.as_str(), Some("kore-rt.runtime"));
        assert_eq!(n.len(), "kore-rt.runtime".len());
        assert!(!n.is_empty());
        // Distinct names hash apart (a smoke check, not a proof).
        let m = ImageName::new("kore-rt.runtimf").unwrap();
        assert_ne!(n.fnv1a(), m.fnv1a());

        assert!(ImageName::new("").is_none());
        assert!(ImageName::new(core::str::from_utf8(&[b'a'; 65]).unwrap()).is_none());
        assert!(ImageName::new("has\0nul").is_none());
        // Exactly at capacity fits.
        let max = core::str::from_utf8(&[b'a'; NAME_BYTES]).unwrap();
        let full = ImageName::new(max).unwrap();
        assert_eq!(full.len(), NAME_BYTES);
        assert_eq!(full.as_str(), Some(max));
    }

    /// The member-kind vocabulary is closed: image (the old zero reserve)
    /// and data are known, the next value is not.
    #[test]
    fn archive_member_kind_vocabulary_is_closed() {
        assert!(archive_member_kind_known(ARCHIVE_MEMBER_IMAGE));
        assert!(archive_member_kind_known(ARCHIVE_MEMBER_DATA));
        assert!(!archive_member_kind_known(ARCHIVE_MEMBER_DATA + 1));
        assert!(!archive_member_kind_known(u32::MAX));
    }

    #[test]
    fn empty_name_refuses_reads() {
        let e = ImageName::empty();
        assert!(e.is_empty());
        assert_eq!(e.as_str(), None);
    }

    #[test]
    fn bootstrap_namespace_is_explicit_and_prefix_bounded() {
        assert!(
            ImageName::new("boot:kore-shell")
                .unwrap()
                .is_bootstrap_only()
        );
        assert!(!ImageName::new("kore-shell").unwrap().is_bootstrap_only());
        assert!(!ImageName::new("bootstrap").unwrap().is_bootstrap_only());
    }

    #[test]
    fn descriptor_validation_refuses_zeroed_and_corrupt() {
        // Unseeded refuses.
        assert!(!ModuleDescriptor::empty().validate());

        let mut d = ModuleDescriptor::empty();
        d.abi = U16::new(IMAGE_ABI);
        d.kind = kind::LIBRARY;
        d.seg_count = 1;
        d.segs[0].memsz = U64::new(4096);
        d.segs[0].kind = seg::KIND_TEXT;
        d.segs[0].access = seg::ACCESS_R | seg::ACCESS_X;
        assert!(d.validate());

        // A live entry with a vacant (zero) kind refuses.
        d.segs[0].kind = 0;
        assert!(!d.validate());
        d.segs[0].kind = seg::KIND_TEXT;

        // Over-count refuses.
        d.seg_count = MAX_SEGMENTS as u8 + 1;
        assert!(!d.validate());
        d.seg_count = 1;

        // Bad kind discriminant refuses.
        d.kind = 0;
        assert!(!d.validate());
        d.kind = 4;
        assert!(!d.validate(), "one past the vocabulary is not a kind");
        d.kind = kind::LIBRARY;

        d.profile = U32::new(PROFILE_KNOWN);
        assert!(d.validate(), "every registered launch profile validates");
        d.profile = U32::new(PROFILE_KNOWN << 1);
        assert!(!d.validate(), "unknown profile bits fail closed");
        d.profile = U32::new(PROFILE_FS_VIEW_BINDER);
        assert!(
            !d.validate(),
            "a cwd-binding context is invalid without a filesystem face"
        );
        d.profile = U32::new(PROFILE_FS_GRANT_TARGET);
        assert!(
            !d.validate(),
            "a grant target is invalid without a filesystem face"
        );
        d.profile = U32::new(PROFILE_INTERACTIVE);
        assert!(
            d.validate(),
            "an allocation-free interactive image is valid"
        );
        d.profile = U32::new(0);
        d.reserved = U32::new(1);
        assert!(!d.validate(), "descriptor reserve stays zero");
    }

    #[test]
    fn launch_profile_dependencies_fail_closed() {
        assert!(profile_valid(0));
        assert!(profile_valid(PROFILE_FILESYSTEM));
        assert!(profile_valid(PROFILE_HEAP));
        assert!(profile_valid(PROFILE_INTERACTIVE));
        assert!(profile_valid(PROFILE_WINDOW_CONTROL));
        // A power face stands alone: it names no dependency the way the
        // filesystem auxiliaries do.
        assert!(profile_valid(PROFILE_SYSTEM_POWER));
        assert!(profile_valid(PROFILE_SYSTEM_POWER | PROFILE_TELEMETRY));
        assert!(profile_valid(PROFILE_ACCELERATOR));
        assert!(profile_valid(PROFILE_GPU));
        assert!(profile_valid(PROFILE_ACCELERATOR | PROFILE_GPU));
        assert!(profile_valid(PROFILE_HEAP | PROFILE_INTERACTIVE));
        assert!(profile_valid(
            PROFILE_FILESYSTEM | PROFILE_HEAP | PROFILE_INTERACTIVE
        ));

        assert!(!profile_valid(PROFILE_FS_VIEW_BINDER));
        assert!(!profile_valid(PROFILE_FS_GRANT_TARGET));
        assert!(!profile_valid(1 << 31));
    }

    #[test]
    fn segment_lease_geometry_is_derived_without_a_kernel_query() {
        let mut text = SegmentDesc::vacant();
        text.kind = seg::KIND_TEXT;
        text.vaddr = U64::new(0x2000);
        text.memsz = U64::new(0x3000);
        text.filesz = U64::new(0x2a17);
        assert_eq!(text.lease_len(), Some(0x3000));

        // A TLS template is logical rather than mappable geometry. Its
        // granted source includes the in-page prefix and rounded tail.
        let mut tls = SegmentDesc::vacant();
        tls.kind = seg::KIND_TLS;
        tls.vaddr = U64::new(0x1234);
        tls.memsz = U64::new(0x80);
        tls.filesz = U64::new(0x30);
        assert_eq!(tls.lease_len(), Some(0x1000));

        text.memsz = U64::new(0x3001);
        assert_eq!(text.lease_len(), None, "mapping spans stay page-shaped");
    }

    /// **Every kind in the vocabulary validates.**  `validate` used to
    /// spell out `EXECUTABLE || LIBRARY`, so adding
    /// [`kind::STATIC_EXECUTABLE`] left the founder writing a `.kore.ld`
    /// descriptor this crate called invalid — a refusal waiting for the
    /// first consumer that validated before reading, which is exactly
    /// what the linker engine does.  Driven off
    /// [`kind::is_known`] so a fourth kind fails here rather than in a
    /// consumer.
    #[test]
    fn validation_accepts_the_whole_kind_vocabulary() {
        for k in [kind::EXECUTABLE, kind::LIBRARY, kind::STATIC_EXECUTABLE] {
            assert!(kind::is_known(k), "kind {k} is in the vocabulary");
            let mut d = ModuleDescriptor::empty();
            d.abi = U16::new(IMAGE_ABI);
            d.kind = k;
            d.seg_count = 1;
            d.segs[0].memsz = U64::new(4096);
            d.segs[0].kind = seg::KIND_TEXT;
            d.segs[0].access = seg::ACCESS_R | seg::ACCESS_X;
            assert!(d.validate(), "kind {k} must validate");
        }
    }

    /// The placement high water reads back as pages below the page, and a
    /// page that never recorded one says so rather than answering an
    /// address a placer would then land on.
    #[test]
    fn placed_high_water_reads_back_and_zero_means_unrecorded() {
        let page_va = 0x5000_0000 + 0x0800_0000 - MODULE_PAGE_BYTES;
        let mut page = ModulePage::new();
        assert_eq!(page.placed_high_water(page_va), None);
        page.record_placed(3);
        assert_eq!(
            page.placed_high_water(page_va),
            Some(page_va - 3 * MODULE_PAGE_BYTES)
        );
        // A count that would reach below address zero refuses, not wraps.
        page.record_placed(u16::MAX);
        assert_eq!(page.placed_high_water(2 * MODULE_PAGE_BYTES), None);
    }

    #[test]
    fn module_page_push_bounds() {
        let mut page = ModulePage::new();
        assert!(page.is_empty());
        let rec = ModuleRecord {
            name_hash: U64::new(1),
            base: U64::new(0x1000),
            kind: kind::LIBRARY,
            ..ModuleRecord::vacant()
        };
        for _ in 0..MAX_MODULE_RECORDS {
            assert!(page.push(rec));
        }
        // Full page refuses, count unchanged.
        assert!(!page.push(rec));
        assert_eq!(page.count.get() as usize, MAX_MODULE_RECORDS);
        assert!(!page.is_empty());
    }

    #[test]
    fn module_table_walks_the_live_records() {
        let mut page = ModulePage::new();
        let exe = ImageName::new("app").unwrap();
        let lib = ImageName::new("lib.rt").unwrap();
        for (nm, base) in [(&exe, 0x1000u64), (&lib, 0x8000)] {
            let mut rec = ModuleRecord::vacant();
            rec.name_hash = U64::new(nm.fnv1a());
            rec.base = U64::new(base);
            rec.kind = kind::LIBRARY;
            assert!(page.push(rec));
        }
        let table = ModuleTable::of(&page).expect("a stamped page validates");
        assert_eq!(table.records().count(), 2);
        assert_eq!(
            table.record(lib.fnv1a()).map(|r| r.base.get()),
            Some(0x8000),
            "lookup by name hash"
        );
        assert!(
            table.record(0xDEAD).is_none(),
            "an unlinked hash finds nothing"
        );
    }

    #[test]
    fn module_table_refuses_unsealed_and_corrupt_pages() {
        // A zeroed page (never written by a stub) refuses.
        let zeroed = ModulePage::read_from_bytes(&[0u8; size_of::<ModulePage>()]).unwrap();
        assert_eq!(
            ModuleTable::of(&zeroed).unwrap_err(),
            ModulePageError::BadAbi
        );

        // A corrupt count past the table refuses.
        let mut corrupt = ModulePage::new();
        corrupt.count = U16::new(MAX_MODULE_RECORDS as u16 + 1);
        assert_eq!(
            ModuleTable::of(&corrupt).unwrap_err(),
            ModulePageError::BadCount
        );
    }

    #[test]
    fn live_prefixes_stay_bounded() {
        // The descriptor's live prefix clamps at both the count and the
        // table (a corrupt count cannot walk past MAX_SEGMENTS).
        let mut d = ModuleDescriptor::empty();
        assert!(d.live_segs().is_empty());
        d.seg_count = 2;
        d.segs[0].kind = seg::KIND_TEXT;
        d.segs[1].kind = seg::KIND_DATA;
        assert_eq!(d.live_segs().len(), 2);
        d.seg_count = MAX_SEGMENTS as u8 + 7;
        assert_eq!(d.live_segs().len(), MAX_SEGMENTS, "clamped, not panicked");

        // The record's live prefix follows the same discipline.
        let mut r = ModuleRecord::vacant();
        assert!(r.live_segs().is_empty());
        r.seg_count = u8::MAX;
        assert_eq!(r.live_segs().len(), MAX_SEGMENTS, "clamped, not panicked");
    }

    #[test]
    fn ld_block_stamps() {
        let b = LdBlock::new(ImageName::new("test").unwrap(), ModuleDescriptor::empty());
        assert_eq!(b.abi.get(), IMAGE_ABI);
    }
}
