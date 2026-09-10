//! # `entry` — the §5.4 entry contract + crt0 (`res/USERLAND.md`;
//! `res/IMAGE.md` §4).
//!
//! `fn main()` cannot work yet — the `start` lang item lives in `std`,
//! and kore user binaries are `no_std`/`no_main`.  The contract
//! (cortex-m-rt / embassy prior art):
//!
//! 1. **The stub jumps to the ELF entry** with the **boot-page pointer
//!    in the arch argument register** (x86_64 `rdi` / aarch64 `x0` /
//!    riscv64 `a0` — the register the stub itself received `found_user`'s
//!    `arg` in; IMAGE.md §4).  The boot page is a **dedicated
//!    parent-delegated page** — never part of any module's data
//!    template — holding `.kore.boot` ([`BootBlock`]) at offset 0 and
//!    `.kore.ld` ([`LdBlock`]) adjacent ([`LD_BLOCK_AT`]); crt0
//!    *receives* it rather than locating a static.
//! 2. **crt0** (`_start`, target-gated per arch): minimal — the stack
//!    is already live (delegated at spawn; `found_user`'s `stack`) and
//!    TP is already set (the stub's §7 TLS work) — so it only
//!    establishes the call frame and enters the Rust half.
//! 3. **rt init**: nothing imperative — the heap needs no bring-up call
//!    (the global-allocator static is `const`-constructed and grows
//!    lazily through §2.2's statically-bound sync spine on first miss).
//! 4. **[`boot_context`]** validates the pointed-at page fail-closed
//!    (the [`Bootstrap`] stamp; the optional [`LdBlock`]) into the entry
//!    context [`Rt`].
//! 5. **`__kore_rt_main`** — the symbol `#[kore_rt::main]` emitted —
//!    is called with it; [`run_main`] pins the root future in that
//!    frame and runs the executor forever (§3.1/§3.3). A declaration
//!    with startup tasks reaches [`run_main_with_tasks`], which admits
//!    the complete bounded set before the root's first poll.
//!
//! The **host-testable seam** is steps 3–5: plain functions over a
//! synthetic boot page (`tests/entry.rs` fabricates one).  The per-arch
//! naked entries compile only for the kore-user targets
//! (`target_os = "none"`, per the `targets/*-unknown-kore-user.json`
//! regime — the host never sees them); the on-metal proof of the whole
//! chain is #96/#298's (§8: nothing here pretends otherwise).
//!
//! **The std trajectory (§6)**: this module is precisely what
//! `lang_start` replaces — `Rt` is the substrate argument a future
//! `std::sys::kore` consumes, and `#[kore_rt::main]` binaries migrate
//! to plain `fn main()` with no redesign.

use kore_abi::clock::ClockPage;
use kore_abi::founding::FoundingBlock;
use kore_abi::image::{IMAGE_ABI, LdBlock};
use kore_abi::machine::MachineBlock;
use kore_abi::session::SlotRef;
use kore_abi::task::{BootBlock, Bootstrap, BootstrapError, GeomBlock, role};

use zerocopy::FromBytes;

use crate::cap::verbs;
use crate::executor::{Doorbell, Executor, MAX_TASKS, SpawnRefused, SpawnToken};

/// The boot page's size — one page, parent-delegated (`res/IMAGE.md`
/// §4).
pub const BOOT_PAGE_BYTES: usize = 4096;

/// Byte offset of the `.kore.boot` [`BootBlock`] in the boot page.
pub const BOOT_BLOCK_AT: usize = 0;

/// Byte offset of the `.kore.ld` [`LdBlock`] — adjacent to the boot
/// block (both are unaligned-safe frozen layouts, so adjacency is the
/// whole convention).
pub const LD_BLOCK_AT: usize = core::mem::size_of::<BootBlock>();

/// Byte offset of the `.kore.geom` [`GeomBlock`](kore_abi::task::GeomBlock) —
/// the third block, adjacent to the ld block (`res/USERLAND.md` §4).  Carries
/// the founder-mapped `(va, len)` windows a task used to learn from fixed
/// constants (`M2B_SCRATCH_VA`, `SLOTS_VA`); an unstamped block reads `abi == 0`
/// and every window lookup refuses.
pub const GEOM_BLOCK_AT: usize = LD_BLOCK_AT + core::mem::size_of::<LdBlock>();

/// Byte offset of the founder-published immutable clock calibration.  It is
/// adjacent to the geometry block so a boot page remains one typed sequence;
/// a zeroed legacy page has `abi == 0` and therefore simply exposes no fast
/// clock.
pub const CLOCK_BLOCK_AT: usize = GEOM_BLOCK_AT + core::mem::size_of::<GeomBlock>();

const _: () = assert!(
    CLOCK_BLOCK_AT + core::mem::size_of::<ClockPage>() <= BOOT_PAGE_BYTES,
    "the frozen boot-page blocks must fit one page"
);
const _: () = assert!(
    CLOCK_BLOCK_AT.is_multiple_of(core::mem::align_of::<ClockPage>()),
    "the clock block must preserve ClockPage's natural alignment"
);

/// Byte offset of the **founding record**
/// ([`FoundingBlock`](kore_abi::founding::FoundingBlock)) — the fifth
/// block: the boot chain's per-row testimony (ready / NACK rung / beacon
/// note), stamped as a consistent snapshot at spawn so a shell program
/// can answer "which rows came up, and why not" on a machine whose
/// console output was never visible.  A zeroed region reads `abi == 0`
/// and exposes no record.
pub const FOUNDING_BLOCK_AT: usize = CLOCK_BLOCK_AT + core::mem::size_of::<ClockPage>();

const _: () = assert!(
    FOUNDING_BLOCK_AT + core::mem::size_of::<FoundingBlock>() <= BOOT_PAGE_BYTES,
    "the frozen boot-page blocks must fit one page"
);
const _: () = assert!(
    FOUNDING_BLOCK_AT.is_multiple_of(core::mem::align_of::<FoundingBlock>()),
    "the founding block must preserve its natural alignment"
);

/// Byte offset of the **machine-facts block**
/// ([`MachineBlock`](kore_abi::machine::MachineBlock)) — the sixth
/// block: the logical CPU count and the founding node's best-class
/// reservation population, kernel-stamped at spawn so the runtime can
/// size a launch to the machine ([`crate::spmd::dispatch`]) without an
/// IPC round trip.  A zeroed region reads `abi == 0` and exposes
/// nothing.
pub const MACHINE_BLOCK_AT: usize = FOUNDING_BLOCK_AT + core::mem::size_of::<FoundingBlock>();

const _: () = assert!(
    MACHINE_BLOCK_AT + core::mem::size_of::<MachineBlock>() <= BOOT_PAGE_BYTES,
    "the frozen boot-page blocks must fit one page"
);
const _: () = assert!(
    MACHINE_BLOCK_AT.is_multiple_of(core::mem::align_of::<MachineBlock>()),
    "the machine block must preserve its natural alignment"
);

/// Why the entry refused to come up (fail-closed: a task with an
/// invalid boot page never reaches `main`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryError {
    /// The page is too small to hold the boot block.
    ShortPage,
    /// The `.kore.boot` block failed validation ([`Bootstrap::of`]).
    BadBoot(BootstrapError),
    /// A second crt0 pass described a different runtime arena layout.
    RuntimeLayout,
}

/// The **entry context** — what `#[kore_rt::main]`'s
/// `async fn main(ctx: Rt)` receives: the validated bootstrap view over
/// the parent-written boot page, plus the loader metadata when the
/// parent staged it.  The one argument `std`'s runtime will one day
/// synthesize (§6).
#[derive(Clone, Copy, Debug)]
pub struct Rt {
    bootstrap: Bootstrap<'static>,
    ld: Option<&'static LdBlock>,
    geom: Option<&'static GeomBlock>,
    clock: Option<&'static ClockPage>,
    founding: Option<&'static FoundingBlock>,
    machine: Option<&'static MachineBlock>,
    page: &'static [u8],
}

#[cfg(any(test, target_os = "none"))]
const PROCESS_IO_ATTACHMENT: usize = 1 << 0;
#[cfg(any(test, target_os = "none"))]
const PROCESS_STDIO_ATTACHMENT: usize = 1 << 1;

/// Atomically spend one runtime attachment name.
///
/// This is deliberately a bit in the process-root
/// [`geometry::ThreadBlock`], not an `Rt` field: `Rt` is a cheap parsed view,
/// can lawfully be reconstructed, and can be shared with another worker in
/// the same address space. Every worker's stamped geometry carries the
/// process-root block, which remains mapped for the process lifetime even
/// after TLS migrates the initial thread's current block. Every worker
/// therefore spends the same VAS-local atomic word, so only the first caller
/// can construct a mutable attachment.
#[cfg(any(test, target_os = "none"))]
fn claim_attachment_word(claims: &core::sync::atomic::AtomicUsize, mask: usize) -> bool {
    debug_assert!(mask.is_power_of_two());
    claims.fetch_or(mask, core::sync::atomic::Ordering::AcqRel) & mask == 0
}

/// Proof that a protocol-fixed arena band was claimed out of the runtime's
/// persistent-slot allocator.
///
/// The claim is deliberately permanent for this thread: fixed protocol names
/// may hold live authority until process teardown, so dropping this witness
/// never reopens their bitmap bits. Services keep the value to make ownership
/// of a hand-planned band explicit alongside their other affine endowment.
#[derive(Debug)]
pub struct FixedSlotBand {
    /// Arena ownership is per thread; never let this witness cross to another
    /// arena merely because it contains no pointer-sized payload.
    _thread_bound: core::marker::PhantomData<*mut ()>,
}

impl Rt {
    /// The validated role-tagged handle table (the task's endowment).
    pub fn bootstrap(&self) -> &Bootstrap<'static> {
        &self.bootstrap
    }

    /// The `.kore.ld` loader metadata, if the parent staged it (a
    /// statically-staged service may have none).
    pub fn ld(&self) -> Option<&'static LdBlock> {
        self.ld
    }

    /// The passive module page the startup stub wrote (`res/IMAGE.md`
    /// §8): one record per module it mapped, at the fixed address
    /// [`kore_abi::image::module_page_va`] names inside `geom::LINK`.
    /// Ambient data like the founding block — reading it grants nothing.
    /// `None` without a LINK window (host builds, a task founded without
    /// the stub) or when the bytes there do not validate as a page.
    #[cfg(target_os = "none")]
    pub fn modules(&self) -> Option<&'static kore_abi::image::ModulePage> {
        use kore_abi::image::{IMAGE_ABI, MAX_MODULE_RECORDS, MODULE_PAGE_BYTES, ModulePage};
        use zerocopy::FromBytes;

        let (base, len) = self.geom()?.window(kore_abi::task::geom::LINK)?;
        let va = kore_abi::image::module_page_va(base, len)?;
        let va = usize::try_from(va).ok()?;
        // SAFETY: the stub wrote and sealed the page read-only at exactly
        // this address for the task's whole life (`LinkWorld::
        // module_page_base`), and every task with a LINK window entered
        // through the stub; the slice is one page of that mapping.
        // The record is a PREFIX of the page (3976 of 4096 bytes: 8-byte
        // header + 16 records), so parse it as one — `ref_from_bytes`
        // demands an exact-size slice and refused the whole page, which
        // read as "no passive module page" on every `require`.
        let bytes =
            unsafe { core::slice::from_raw_parts(va as *const u8, MODULE_PAGE_BYTES as usize) };
        let (page, _tail) = ModulePage::ref_from_prefix(bytes).ok()?;
        (page.abi.get() == IMAGE_ABI
            && usize::from(page.count.get()) <= MAX_MODULE_RECORDS
            && !page.is_empty())
        .then_some(page)
    }

    /// Off metal there is no stub and no page.
    #[cfg(not(target_os = "none"))]
    pub fn modules(&self) -> Option<&'static kore_abi::image::ModulePage> {
        None
    }

    /// The `.kore.geom` window table, if the founder stamped it — where the
    /// task learns its founder-mapped `(va, len)` windows (scratch, slots, …)
    /// instead of hardcoded constants (`res/USERLAND.md` §4).  `None` for a
    /// task whose founder carried no geometry (host builds, a statically-staged
    /// blob before the founder stamps one) — every geometry-dependent verb then
    /// fail-closes, exactly as it did when its VA const was absent.
    pub fn geom(&self) -> Option<&'static GeomBlock> {
        self.geom
    }

    /// The founder-published monotonic-counter calibration.  Reading it is
    /// ambient data, not timer authority; arming still rides the executor's
    /// bounded deadline table / `WAIT_SLEEP` path.
    pub fn clock(&self) -> Option<&'static ClockPage> {
        self.clock
    }

    /// The founding record — the boot chain's per-row testimony (ready /
    /// NACK rung / beacon note), snapshotted into this task's boot page
    /// at spawn.  Ambient data like the clock block: reading it grants
    /// nothing.  `None` on a pre-record founder or a zeroed region.
    pub fn founding(&self) -> Option<&'static FoundingBlock> {
        self.founding
    }

    /// The machine-facts block — the logical CPU count and the founding
    /// node's best-class reservation population, kernel-stamped into
    /// this task's boot page at spawn.  Ambient data like the clock and
    /// founding blocks: reading it grants nothing.  `None` on a
    /// pre-record founder or a zeroed region.
    pub fn machine(&self) -> Option<&'static MachineBlock> {
        self.machine
    }

    /// The immutable parent-written boot page.
    ///
    /// Generic runtime code consumes the standard blocks through the typed
    /// accessors above.  A service with an additional frozen startup contract
    /// parses it at its declared offset from this same page; it never
    /// discovers authority by scanning slots.
    pub fn boot_page(&self) -> &'static [u8] {
        self.page
    }

    /// Spend this task's sole safe `ProcessIo` attachment.
    ///
    /// The claim word lives in the geometry-stamped process-root block shared
    /// by every worker in this address space. Even a worker handed a copy of
    /// this original `Rt` therefore collides with the first attachment.
    /// Worker boot blocks also omit inherited holdings and the delivered view
    /// is non-derivable, but those are defense in depth rather than the
    /// one-shot premise.
    #[cfg(target_os = "none")]
    pub(crate) fn claim_process_io_attachment(&self) -> bool {
        geometry::claim_runtime_attachment(PROCESS_IO_ATTACHMENT)
    }

    /// Spend this task's sole safe foreground-stream attachment.
    ///
    /// This has the same process-address-space scope as
    /// [`claim_process_io_attachment`](Self::claim_process_io_attachment).
    /// Keeping the claim outside `Rt` closes both repeated `boot_context` calls
    /// and repeated [`thread_context`] reconstruction on any worker.
    #[cfg(target_os = "none")]
    pub(crate) fn claim_process_stdio_attachment(&self) -> bool {
        geometry::claim_runtime_attachment(PROCESS_STDIO_ATTACHMENT)
    }

    /// Permanently claim the half-open arena interval `[first, end)` for a
    /// service's compile-time protocol layout.
    ///
    /// The interval must lie wholly inside the runtime-opened persistent-slot
    /// window (after report + heap-carve scratch and before the startup-linker
    /// band), and every member must still be vacant. A collision refuses as a
    /// unit; no prefix remains claimed. Peer-controlled values must never be
    /// routed here—dynamic capability ownership uses the typed allocators in
    /// [`crate::cap`] instead.
    pub fn claim_fixed_slot_band(&self, first: u32, end: u32) -> Option<FixedSlotBand> {
        #[cfg(target_os = "none")]
        {
            return geometry::claim_fixed_slot_band(first as usize, end as usize).then_some(
                FixedSlotBand {
                    _thread_bound: core::marker::PhantomData,
                },
            );
        }
        #[cfg(not(target_os = "none"))]
        {
            let _ = (first, end);
            None
        }
    }

    /// [`claim_fixed_slot_band`](Self::claim_fixed_slot_band)'s sibling
    /// over the **extension** slot space (slice E): atomically claim a
    /// service-owned fixed band of slots at or above
    /// [`ARENA_SLOT_CEILING`], bounded by what this task's extends have
    /// actually aliased — so the claim can only follow a successful
    /// [`extend_arena`](crate::cap::verbs::extend_arena).  Same
    /// all-or-nothing rollback contract; same thread-bound witness.
    pub fn claim_ext_slot_band(&self, first: u32, end: u32) -> Option<FixedSlotBand> {
        #[cfg(target_os = "none")]
        {
            return geometry::claim_ext_slot_band(first as usize, end as usize).then_some(
                FixedSlotBand {
                    _thread_bound: core::marker::PhantomData,
                },
            );
        }
        #[cfg(not(target_os = "none"))]
        {
            let _ = (first, end);
            None
        }
    }

    /// Resolve the task's **executor doorbell** from its
    /// [`role::DOORBELL`] entries: index 0 = the held `ChanRecv` slot,
    /// index 1 = the designated landing slot (the founding pins both).
    /// `None` when the roles are absent or the recv end does not bind —
    /// a doorbell-less task cannot idle (§3.1), so [`run_main`] refuses.
    pub fn doorbell(&self) -> Option<Doorbell> {
        let recv = self.bootstrap.handle(role::DOORBELL, 0)?;
        let land = self.bootstrap.handle(role::DOORBELL, 1)?;
        let (chan, recv_tag) = verbs::recv_pin(recv)?;
        Some(Doorbell {
            chan,
            recv,
            land,
            recv_tag,
        })
    }
}

/// Validate the boot page into the entry context — **fail-closed**: an
/// unseeded/stale/corrupt `.kore.boot` refuses (its zero ABI stamp is
/// the unseeded signature), and a `.kore.ld` whose stamp is not
/// [`IMAGE_ABI`] simply reads as absent (the statically-staged case
/// leaves the region zeroed).
pub fn boot_context(page: &'static [u8]) -> Result<Rt, EntryError> {
    let (bootstrap, geom, clock, founding, machine) = boot_parse(page)?;
    let ld = page
        .get(LD_BLOCK_AT..)
        .and_then(|b| LdBlock::ref_from_prefix(b).ok())
        .map(|(ld, _)| ld)
        .filter(|ld| ld.abi.get() == IMAGE_ABI);
    // **Install the geometry statics** (metal only) before any verb or the trap
    // transport runs — they replace the hand-copied `M2B_SCRATCH_VA`/`SLOTS_VA`
    // constants with the founder's actual mapping, learned from the boot page.
    #[cfg(target_os = "none")]
    if let Some(g) = geom {
        geometry::install(g);
    }
    // **Derive + install the session report slot** (#336) — the arena slot
    // the report-answered verbs (map, claim, heap-grow) name as their
    // session's result, one past the serve-scratch run (the same
    // derived-not-declared convention as [`SERVE_SCRATCH_SLOTS`]: everything
    // past `max(named)` is untouched arena).  The slot after it is the grow
    // session's carve scratch, so the bound covers both.  A task whose boot
    // block names no handles gets none — every report-answered verb then
    // refuses, fail-closed like every geometry lookup.
    #[cfg(target_os = "none")]
    if let Some(top) = bootstrap.handles().map(|(_, _, s)| s.index()).max()
        && let Some(slots) = runtime_slots_after(top)
    {
        if !geometry::install_report_slot(slots.report as usize) {
            return Err(EntryError::RuntimeLayout);
        }
    }
    // The `role::HEAP` supply slot (#336 rung 6) — the founder-seeded lease
    // the allocator's grow session carves from.
    #[cfg(target_os = "none")]
    if let Some(heap) = bootstrap.handle(role::HEAP, 0) {
        geometry::install_heap_slot(heap);
    }
    Ok(Rt {
        bootstrap,
        ld,
        geom,
        clock,
        founding,
        machine,
        page,
    })
}

/// The task's founder-mapped window geometry (`res/USERLAND.md` §4), learned
/// from the `.kore.geom` boot block at crt0 and read by the trap transport and
/// the possession verbs — the runtime-internal successor to the fixed
/// `M2B_SCRATCH_VA` / `SLOTS_VA` constants.  Metal-only (the trap and the verbs
/// that read these are `target_os = "none"`); a value of `0` means *unstamped*,
/// so every reader fail-closes exactly as it did when its VA const was absent.
/// A zero-trap snapshot of this task's last dispatch advice. Migration may
/// immediately stale it; never use it to authorize an instruction set.
#[inline]
pub fn execution_hint() -> Option<kore_abi::execution::ExecutionHint> {
    #[cfg(target_os = "none")]
    {
        geometry::execution_hint()
    }
    #[cfg(not(target_os = "none"))]
    {
        None
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReportSlotInstall {
    Initialize,
    Preserve,
    Conflict,
}

const fn report_slot_install(current_p1: usize, requested: usize) -> ReportSlotInstall {
    if current_p1 == 0 {
        ReportSlotInstall::Initialize
    } else if current_p1 == requested + 1 {
        ReportSlotInstall::Preserve
    } else {
        ReportSlotInstall::Conflict
    }
}

#[cfg(target_os = "none")]
pub(crate) mod geometry {
    use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    use kore_abi::task::{GeomBlock, geom};

    use super::{
        ARENA_SLOT_CEILING, OWNED_SLOT_WORDS, ReportSlotInstall, STARTUP_LINK_SLOT_BASE,
        report_slot_install, slot_interval_word,
    };

    /// The **per-thread runtime block** (#407) — the `%fs:0`-anchored home
    /// of everything this module used to keep in process-global statics.
    /// One per task: the founder maps and names a zeroed R/W page
    /// (`geom::THREAD_BLOCK`); crt0 stamps the self-pointer and anchors
    /// `%fs` at it through the SET_TP verb ([`super::anchor_boot_thread`]),
    /// and a spawned thread's parent provisions its own page the same
    /// way.  The layout is **rt-owned** — nothing kernel-side reads it —
    /// and almost every field is per-ARENA state, which under the unified task
    /// model means per-THREAD state: two threads sharing one address space
    /// each stage into their own scratch, read their own slots alias, and
    /// allocate from their own bitmap. The explicitly documented
    /// `runtime_attachments` exception is read only through the stamped
    /// process-root pointer so one process-wide claim survives `Rt` transfer
    /// across workers.
    #[repr(C)]
    pub(crate) struct ThreadBlock {
        /// `%fs:0` — the block's own VA (TLS variant 2's self-pointer);
        /// what [`block`] dereferences.
        self_ptr: AtomicUsize,
        /// Stable process-root block VA. The initial thread names its own
        /// founder window; every worker receives that same window as
        /// `geom::THREAD_BLOCK@1`. A TLS migration copies this value intact.
        process_root_va: AtomicUsize,
        /// Initial-thread anchor state: `0` = untouched, `1` = prepared,
        /// `2` = the kernel accepted SET_TP.  This makes crt0's anchor
        /// idempotent across the startup-stub -> program-entry handoff
        /// without treating a merely stamped page as proof that `%fs` is
        /// live.
        anchor_state: AtomicUsize,
        /// The R/W descriptor-scratch window base (`geom::SCRATCH`);
        /// `0` = unstamped.
        scratch_va: AtomicUsize,
        /// The R/O cap-slot-alias base (`geom::SLOTS`); `0` = unstamped.
        slots_va: AtomicUsize,
        /// The aliased cap-slot span in bytes (`arena_slots · 128`).
        slots_len: AtomicUsize,
        /// The arena-**extension** alias base (`geom::SLOTS_EXT`, slice E);
        /// `0` = unstamped — a spawned thread, or a founding that predates
        /// the growable arena; every extension read and every
        /// [`extend_arena`](crate::cap::verbs::extend_arena) then refuses,
        /// fail-closed.
        slots_ext_va: AtomicUsize,
        /// The **live** aliased extension span in bytes — how far past the
        /// founding array this task's records actually reach, bumped
        /// monotonically from each extend's kernel report.  NOT the
        /// stamped reservation length: the reservation is address space,
        /// this is proof-of-alias, and an un-bumped length simply hides
        /// slots the kernel has already mapped (fail-closed, never a wild
        /// read).
        slots_ext_len: AtomicUsize,
        /// The session verbs' derived **report slot** plus one (#336 —
        /// slot 0 is real, so `0` marks *underived*).
        report_slot_p1: AtomicUsize,
        /// One-shot ownership bits for safe runtime mappings which yield
        /// mutable references or SPSC endpoints. Only the geometry-stamped
        /// process-root block is the authority for these bits; worker blocks
        /// retain this field solely because every ThreadBlock has one frozen
        /// layout. Unlike an `Rt` field, the root word survives parsed context
        /// reconstruction and is shared by every worker in the VAS.
        runtime_attachments: AtomicUsize,
        /// Allocation bitmap for persistent userland-owned capability
        /// names (`1` = reserved/in use); opened by
        /// [`install_report_slot`].
        owned_slots: [AtomicU64; OWNED_SLOT_WORDS],
        /// The `role::HEAP` supply's arena slot plus one (`0` =
        /// unendowed — the allocator's grow refuses, fail-closed).
        heap_slot_p1: AtomicUsize,
        /// This thread's own `role::ADDRESS_SPACE` arena slot plus one
        /// (#407 — cached at anchor so `thread::spawn` can derive views
        /// without re-parsing the boot page; `0` = unanchored).
        self_space_slot_p1: AtomicUsize,
        /// This thread's own boot-page VA (`0` = unanchored).
        ///
        /// Thread 0 receives its page as crt0's argument and carries it
        /// in [`Rt`](super::Rt) for the task's whole life; a worker
        /// thread's entry takes no arguments, so its page is remembered
        /// here instead.  Same page, same parse — a worker resolves its
        /// endowment through the ordinary [`Bootstrap`](kore_abi::task::Bootstrap)
        /// lookup rather than through some thread-only side channel.
        boot_page_va: AtomicUsize,
        /// Panic-witness reentrancy guard for this arena/thread. A worker
        /// failure must not poison a later reportable panic on another thread
        /// in the same process.
        panicking: AtomicUsize,
        /// This thread's [`BANDS`](crate::thread::BANDS)`.thread_block` slot
        /// plus one (`0` = unstamped).
        ///
        /// The thread's identity as a *number*, recovered at anchor from the
        /// one address that already names it uniquely — its own block window
        /// — so there is no second source of truth to drift from.  The
        /// allocator reads it as its shard index
        /// ([`current_shard`](crate::heap::shard::current_shard)), which is
        /// sound precisely because a band slot is held exclusively by one
        /// live thread: distinct live threads therefore have distinct
        /// indices, which is the injectivity the shards' owner-only
        /// free-lists rest on.
        shard_index_p1: AtomicUsize,
        /// `1` once this thread's block moved into a linker-built TCB —
        /// i.e. this image has a **static TLS block** around its thread
        /// pointer ([`migrate_thread_block`](super::migrate_thread_block)
        /// runs only when `kore_ld` built one).
        ///
        /// Child spawning requires a retained template when this is set.
        /// Deliberately *not* inherited: it describes the address the
        /// block sits at, and a child's block is its own.
        static_tls: AtomicUsize,
        // Immutable process-root TLS template, published by the startup linker.
        tls_template_base: AtomicUsize,
        tls_template_bytes: AtomicUsize,
        tls_template_tp: AtomicUsize,
        tls_template_align: AtomicUsize,
        thread_owner_next: AtomicUsize,
        thread_owner_id: AtomicUsize,
    }

    // The block must fit the founder's one-page window.
    const _: () = assert!(core::mem::size_of::<ThreadBlock>() <= 4096);

    impl ThreadBlock {
        /// Stamp a zeroed, founder-mapped page into a live block: the
        /// self-pointer `%fs:0` will dereference and the all-reserved
        /// slot bitmap ([`install_report_slot`] opens the lawful window).
        /// Everything else keeps the page's zero = "unstamped".
        ///
        /// # Safety
        /// `va` is this thread's live R/W `geom::THREAD_BLOCK` window —
        /// founder-mapped, page-sized, exclusively this thread's.
        /// `process_root_va` is the live complete root-block window inherited
        /// from immutable geometry and shared for the process lifetime.
        /// `boot_page` is this thread's live boot-page VA.
        pub(crate) unsafe fn init(
            va: usize,
            process_root_va: usize,
            self_space: kore_abi::session::SlotRef,
            boot_page: usize,
        ) {
            // SAFETY: caller contract — a live, exclusive, zeroed page.
            let b = unsafe { &*(va as *const ThreadBlock) };
            b.self_ptr.store(va, Ordering::Relaxed);
            b.process_root_va.store(process_root_va, Ordering::Relaxed);
            for word in &b.owned_slots {
                word.store(u64::MAX, Ordering::Relaxed);
            }
            b.self_space_slot_p1
                .store(self_space.index() as usize + 1, Ordering::Relaxed);
            b.boot_page_va.store(boot_page, Ordering::Relaxed);
            // The thread's own identity, read off the window it was handed.
            // A block outside the declared band leaves this `0`: the
            // allocator then refuses to claim a private shard rather than
            // guess one it cannot prove is unshared.
            if let Some(slot) = crate::thread::BANDS.thread_block.slot_of(va as u64) {
                b.shard_index_p1.store(slot as usize + 1, Ordering::Relaxed);
            }
            b.anchor_state.store(1, Ordering::Release);
        }

        /// Has SET_TP actually committed for the block at `va`?
        ///
        /// Read through the founder-mapped VA, not `%fs`: this predicate is
        /// specifically for the interval before the caller knows whether its
        /// thread pointer is live.
        ///
        /// # Safety
        /// `va` names this thread's live R/W `geom::THREAD_BLOCK` window.
        pub(crate) unsafe fn is_anchored(va: usize) -> bool {
            // SAFETY: caller contract.
            let b = unsafe { &*(va as *const ThreadBlock) };
            b.self_ptr.load(Ordering::Acquire) == va && b.anchor_state.load(Ordering::Acquire) == 2
        }

        /// Re-point a **migrated** block's self-pointer at its new home.
        ///
        /// A byte copy carries the old VA in `self_ptr`, and the block's
        /// identity is its own address (`%fs:0` *is* this field on x86_64),
        /// so the copy is not a block until this runs.  Everything else in
        /// the block is position-independent and moves verbatim — which is
        /// the point of migrating rather than re-initializing: the report
        /// slot, the owned-slot bitmap, and the heap supply the linker
        /// already spent are this thread's, not this address's.
        ///
        /// # Safety
        /// `va` names a live R/W region of at least `size_of::<ThreadBlock>()`
        /// bytes, exclusively this thread's, already holding the copy.
        pub(crate) unsafe fn restamp_self(va: usize) {
            // SAFETY: caller contract.
            let b = unsafe { &*(va as *const ThreadBlock) };
            b.self_ptr.store(va, Ordering::Release);
        }

        /// Publish successful SET_TP completion after the kernel accepts it.
        ///
        /// # Safety
        /// Same live exclusive block contract as [`Self::init`].
        pub(crate) unsafe fn mark_anchored(va: usize) {
            // SAFETY: caller contract.
            let b = unsafe { &*(va as *const ThreadBlock) };
            b.anchor_state.store(2, Ordering::Release);
        }
    }

    /// This thread's boot-page VA, or `None` before it is anchored.
    pub(crate) fn boot_page_va() -> Option<usize> {
        match block().boot_page_va.load(Ordering::Relaxed) {
            0 => None,
            va => Some(va),
        }
    }

    /// Claim this thread's one panic-witness attempt. Returns `true` only to
    /// the first caller on this anchored thread.
    pub(crate) fn begin_panic() -> bool {
        block().panicking.swap(1, Ordering::AcqRel) == 0
    }

    /// Where this thread's block lives — the value the thread pointer holds,
    /// recovered the same way every other accessor recovers it.
    pub(crate) fn block_va() -> usize {
        block() as *const ThreadBlock as usize
    }

    /// Does this image carry a static TLS block around the thread pointer?
    ///
    /// True exactly when the startup linker built one and moved this block
    /// into its TCB.  A static-only image answers `false`: its thread
    /// pointer names nothing but the block.
    pub(crate) fn has_static_tls() -> bool {
        block().static_tls.load(Ordering::Relaxed) != 0
    }

    /// Record that this block now sits inside a linker-built TCB.
    ///
    /// # Safety
    /// `va` names this thread's live block, already migrated there.
    pub(crate) unsafe fn mark_static_tls(va: usize) {
        // SAFETY: caller contract.
        let b = unsafe { &*(va as *const ThreadBlock) };
        b.static_tls.store(1, Ordering::Release);
    }

    /// The calling thread's block, dereferenced off the arch thread-pointer
    /// register the SET_TP verb anchored: `%fs` on x86_64 (read as the
    /// `fs:[0]` self-pointer — without FSGSBASE, ring 3 cannot read its own
    /// FS base, which is what the TLS-variant-2 self-pointer is for), and
    /// `TPIDR_EL0` / `tp` on aarch64 / riscv64, which hold the block VA
    /// directly and read as one register move.
    ///
    /// Lawful only after [`super::anchor_boot_thread`] ran on this thread
    /// (crt0's first act; a spawned thread's trampoline's first act) —
    /// which is strictly before any verb, transport, or executor code
    /// that reaches here.
    #[inline]
    fn block() -> &'static ThreadBlock {
        let ptr: *const ThreadBlock;
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `%fs` was anchored at this thread's block before any
        // accessor runs; `fs:[0]` is the block's self-pointer, constant
        // for the thread's whole life.
        unsafe {
            core::arch::asm!(
                "mov {}, fs:[0]",
                out(reg) ptr,
                options(nostack, pure, readonly, preserves_flags)
            );
        }
        #[cfg(target_arch = "aarch64")]
        // SAFETY: SET_TP anchored `TPIDR_EL0` at this thread's block before
        // any accessor runs.
        //
        // NOT `pure`: with no inputs and `nomem`, `pure` would tell LLVM this
        // read is a loop-invariant constant — free to compute once and reuse
        // everywhere — when the value lives in a CPU register Rust does not
        // model.  `SET_TP` rewrites it mid-thread (`crt0_start`'s anchor, and
        // the verb thereafter), so a hoisted or CSE'd read could hand back
        // the *pre-anchor* thread pointer.  The x86 sibling above gets away
        // with `pure` only because it says `readonly` rather than `nomem`:
        // its result is memory-dependent, and the memory-clobbering syscall
        // that performs a SET_TP invalidates it.  A register read has nothing
        // to be ordered against, so it cannot be pure at all.
        unsafe {
            core::arch::asm!(
                "mrs {}, tpidr_el0",
                out(reg) ptr,
                options(nostack, nomem, preserves_flags)
            );
        }
        #[cfg(target_arch = "riscv64")]
        // SAFETY: SET_TP anchored `tp` at this thread's block before any
        // accessor runs; the kernel trampoline restores it on every trap
        // return.  Not `pure` — see the aarch64 sibling: same register-read
        // reasoning, same SET_TP hazard.
        unsafe {
            core::arch::asm!(
                "mv {}, tp",
                out(reg) ptr,
                options(nostack, nomem, preserves_flags)
            );
        }
        // SAFETY: the anchored thread pointer names the live, founder-mapped
        // block.
        unsafe { &*ptr }
    }

    /// A process-unique capability-arena identity, preserved across TCB moves
    /// and never reused when a thread band is reclaimed.
    #[cfg(target_os = "none")]
    pub(crate) fn thread_owner_identity() -> Option<usize> {
        let local = block();
        let id = local.thread_owner_id.load(Ordering::Acquire);
        if id != 0 {
            return Some(id);
        }
        let root = process_root_block()?;
        let id = root
            .thread_owner_next
            .fetch_update(Ordering::AcqRel, Ordering::Relaxed, |n| n.checked_add(1))
            .ok()?
            .checked_add(1)?;
        match local
            .thread_owner_id
            .compare_exchange(0, id, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => Some(id),
            Err(existing) => Some(existing),
        }
    }

    #[cfg(target_os = "none")]
    pub(crate) fn tls_template() -> Option<crate::tls::Template> {
        let root = process_root_block()?;
        let base = root.tls_template_base.load(Ordering::Acquire);
        crate::tls::Template::new(
            base as u64,
            root.tls_template_bytes.load(Ordering::Relaxed) as u64,
            root.tls_template_tp.load(Ordering::Relaxed) as u64,
            root.tls_template_align.load(Ordering::Relaxed) as u64,
        )
    }

    #[cfg(target_os = "none")]
    pub(crate) unsafe fn publish_tls_template(template: crate::tls::Template) -> bool {
        let Some(root) = process_root_block() else {
            return false;
        };
        if root.tls_template_base.load(Ordering::Acquire) != 0 {
            return false;
        }
        root.tls_template_bytes
            .store(template.bytes as usize, Ordering::Relaxed);
        root.tls_template_tp
            .store(template.tp_offset as usize, Ordering::Relaxed);
        root.tls_template_align
            .store(template.align as usize, Ordering::Relaxed);
        root.tls_template_base
            .store(template.base as usize, Ordering::Release);
        true
    }

    /// The geometry-stamped process-lifetime runtime block.
    ///
    /// Every spawned worker shares this address space, while each separate
    /// process carries its own founder-mapped root page. The initial thread
    /// may later move its *current* block into a TLS TCB, but the root page
    /// remains installed and is the stable ownership home for process-wide
    /// attachment claims.
    #[cfg(target_os = "none")]
    fn process_root_block() -> Option<&'static ThreadBlock> {
        let va = block().process_root_va.load(Ordering::Acquire);
        if va == 0 || !va.is_multiple_of(core::mem::align_of::<ThreadBlock>()) {
            return None;
        }
        // SAFETY: anchor validation accepted this complete ThreadBlock window
        // from immutable founder/parent-stamped geometry. Spawned workers
        // share that mapping, and process teardown bounds its life.
        let root = unsafe { &*(va as *const ThreadBlock) };
        debug_assert_eq!(root.self_ptr.load(Ordering::Acquire), va);
        Some(root)
    }

    /// Populate the window fields from the founder-stamped block — called
    /// once per thread, before the executor or any verb runs.  `Relaxed`
    /// is sufficient: the write happens on this thread strictly before
    /// any of its readers, and the block is per-thread by construction.
    pub(crate) fn install(g: &GeomBlock) {
        let b = block();
        if let Some(w) = g.window_of(geom::SCRATCH) {
            b.scratch_va.store(w.base() as usize, Ordering::Relaxed);
        }
        if let Some(w) = g.window_of(geom::SLOTS) {
            b.slots_va.store(w.base() as usize, Ordering::Relaxed);
            b.slots_len.store(
                w.len().saturating_sub(kore_abi::execution::HINT_PAGE_BYTES) as usize,
                Ordering::Relaxed,
            );
        }
        // The extension alias window (slice E): the BASE alone — the
        // stamped length is a reservation, and the live aliased length
        // starts at zero and grows only with each extend's kernel report
        // (`grow_slots_ext`).  Threads are stamped no such window and
        // stay fail-closed.
        if let Some(w) = g.window_of(geom::SLOTS_EXT) {
            b.slots_ext_va.store(w.base() as usize, Ordering::Relaxed);
        }
    }

    /// The founder-stamped `geom::SCRATCH` window base, or `None` when the
    /// founder carried no geometry — the trap then has nowhere to stage and
    /// refuses, rather than writing at a guessed address.
    pub(crate) fn scratch_va() -> Option<usize> {
        match block().scratch_va.load(Ordering::Relaxed) {
            0 => None, // unstamped ⇒ no window to trust (fail-closed)
            va => Some(va),
        }
    }

    /// Record the derived report slot — `boot_context`'s one-shot install.
    ///
    /// A linked process enters crt0 twice: first in the startup stub, then in
    /// the final executable. The stub may already own dynamic grant landings
    /// when the second pass arrives, so a matching re-install preserves the
    /// bitmap exactly. A different slot would reinterpret live arena names and
    /// therefore refuses rather than resetting their ownership ledger.
    pub(crate) fn install_report_slot(slot: usize) -> bool {
        let b = block();
        match report_slot_install(b.report_slot_p1.load(Ordering::Acquire), slot) {
            ReportSlotInstall::Preserve => return true,
            ReportSlotInstall::Conflict => return false,
            ReportSlotInstall::Initialize => {}
        }
        let first = slot.saturating_add(2);
        let slots = (b.slots_len.load(Ordering::Relaxed)
            / core::mem::size_of::<kore_abi::cap::CapRecord>())
        .min(STARTUP_LINK_SLOT_BASE as usize)
        .min(ARENA_SLOT_CEILING as usize);
        for (index, word) in b.owned_slots.iter().enumerate() {
            // `1` means unavailable.  Only the runtime-owned interval below
            // the durable linker band is opened; low named/scratch slots,
            // the link band, and final-word padding remain reserved.
            word.store(!slot_interval_word(index, first, slots), Ordering::Relaxed);
        }
        // Publish the initialized ledger last. This block is per-thread and
        // crt0 is its sole installer; release/acquire documents that later
        // same-thread entry passes observe the complete bitmap.
        b.report_slot_p1.store(slot + 1, Ordering::Release);
        true
    }

    /// The arena slot the session verbs name as their result — `None` when
    /// underived (a handle-less task): every report-answered verb then
    /// refuses, fail-closed.
    pub(crate) fn report_slot() -> Option<kore_abi::session::SlotRef> {
        match block().report_slot_p1.load(Ordering::Relaxed) {
            0 => None,
            // Derived from the founder-stamped block (parent-written wire)
            // at install — the module is the thread's slot authority.
            p1 => Some(kore_abi::session::SlotRef::from_wire((p1 - 1) as u32)),
        }
    }

    /// The grow session's **carve scratch** — the slot after the report slot
    /// (the staged detached window lands here before the same session maps
    /// it); bounds were checked together at install.
    pub(crate) fn carve_slot() -> Option<kore_abi::session::SlotRef> {
        report_slot().map(|s| kore_abi::session::SlotRef::from_wire(s.index() + 1))
    }

    /// **How many bitmap words this task's slots can actually reach** —
    /// the founding array always, plus one word per 64 extension slots it
    /// has aliased.
    ///
    /// The bitmap is sized for the structural ceiling (128 words), but a
    /// task that never extends — every spawned thread, whose 64-slot arena
    /// refuses extension structurally — can only ever own names in the
    /// first four.  Scanning all 128 regardless would make slice E's
    /// widening cost every allocation on every task 124 extra Acquire
    /// loads for words that are permanently `u64::MAX`.  Bounding here
    /// restores the pre-growth cost for the non-extending case and keeps
    /// the extending case proportional to what it grew.
    fn live_slot_words() -> usize {
        const PER_WORD: usize = u64::BITS as usize;
        let rec_bytes = core::mem::size_of::<kore_abi::cap::CapRecord>();
        let ext_slots = block().slots_ext_len.load(Ordering::Acquire) / rec_bytes;
        let live = ARENA_SLOT_CEILING as usize + ext_slots;
        // Round up, and never past the array.
        live.div_ceil(PER_WORD).min(OWNED_SLOT_WORDS)
    }

    /// Reserve one unique persistent result slot for an affine userland
    /// capability. `None` means the bounded arena is exhausted.
    pub(crate) fn acquire_cap_slot() -> Option<kore_abi::session::SlotRef> {
        for (word_index, owned) in block().owned_slots[..live_slot_words()].iter().enumerate() {
            let mut used = owned.load(Ordering::Acquire);
            loop {
                let free = !used;
                if free == 0 {
                    break;
                }
                let bit = free.trailing_zeros() as usize;
                let next = used | (1u64 << bit);
                match owned.compare_exchange_weak(used, next, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => {
                        let slot = word_index * u64::BITS as usize + bit;
                        return Some(kore_abi::session::SlotRef::from_wire(slot as u32));
                    }
                    Err(observed) => used = observed,
                }
            }
        }
        None
    }

    /// Spend one one-shot mapping attachment across this complete process.
    pub(crate) fn claim_runtime_attachment(mask: usize) -> bool {
        process_root_block()
            .is_some_and(|root| super::claim_attachment_word(&root.runtime_attachments, mask))
    }

    /// Snapshot the number of persistent capability slots still available to
    /// this thread.  Admission remains authoritative at
    /// [`acquire_cap_slot`]; this count is a bounded startup preflight so a
    /// service can refuse before publishing a partial catalog.
    pub(crate) fn available_cap_slots() -> usize {
        block().owned_slots[..live_slot_words()]
            .iter()
            .map(|word| (!word.load(Ordering::Acquire)).count_ones() as usize)
            .sum()
    }

    /// Atomically claim one service-owned fixed band out of the persistent
    /// allocator. If any word collides, roll back every word this attempt set;
    /// the caller observes all-or-nothing just like a session commit.
    pub(crate) fn claim_fixed_slot_band(first: usize, end: usize) -> bool {
        let b = block();
        let Some(report) = report_slot().map(|slot| slot.index() as usize) else {
            return false;
        };
        let dynamic_first = report.saturating_add(2);
        let slots = (b.slots_len.load(Ordering::Relaxed)
            / core::mem::size_of::<kore_abi::cap::CapRecord>())
        .min(STARTUP_LINK_SLOT_BASE as usize)
        .min(ARENA_SLOT_CEILING as usize);
        if first < dynamic_first || first >= end || end > slots {
            return false;
        }

        let mut claimed = [0u64; OWNED_SLOT_WORDS];
        for (word_index, owned) in b.owned_slots.iter().enumerate() {
            let mask = slot_interval_word(word_index, first, end);
            if mask == 0 {
                continue;
            }
            let mut observed = owned.load(Ordering::Acquire);
            loop {
                if observed & mask != 0 {
                    for (rollback_index, rollback) in claimed.iter().copied().enumerate() {
                        if rollback != 0 {
                            b.owned_slots[rollback_index].fetch_and(!rollback, Ordering::Release);
                        }
                    }
                    return false;
                }
                match owned.compare_exchange_weak(
                    observed,
                    observed | mask,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        claimed[word_index] = mask;
                        break;
                    }
                    Err(current) => observed = current,
                }
            }
        }
        true
    }

    /// [`claim_fixed_slot_band`]'s extension-space sibling (slice E):
    /// claim `[first, end)` out of the slots this task's extends have
    /// actually aliased.  The lower bound is the founding ceiling (the
    /// startup-link band and every founding slot are the other
    /// claimer's); the upper bound is the LIVE aliased total — a band no
    /// extend has backed refuses, fail-closed.  Same all-or-nothing
    /// word-CAS rollback as the founding claimer.
    pub(crate) fn claim_ext_slot_band(first: usize, end: usize) -> bool {
        let b = block();
        let rec_bytes = core::mem::size_of::<kore_abi::cap::CapRecord>();
        let live_end =
            ARENA_SLOT_CEILING as usize + b.slots_ext_len.load(Ordering::Acquire) / rec_bytes;
        if first < ARENA_SLOT_CEILING as usize
            || first >= end
            || end > live_end
            || end > super::ARENA_EXT_SLOT_CEILING as usize
        {
            return false;
        }

        let mut claimed = [0u64; OWNED_SLOT_WORDS];
        for (word_index, owned) in b.owned_slots.iter().enumerate() {
            let mask = slot_interval_word(word_index, first, end);
            if mask == 0 {
                continue;
            }
            let mut observed = owned.load(Ordering::Acquire);
            loop {
                if observed & mask != 0 {
                    for (rollback_index, rollback) in claimed.iter().copied().enumerate() {
                        if rollback != 0 {
                            b.owned_slots[rollback_index].fetch_and(!rollback, Ordering::Release);
                        }
                    }
                    return false;
                }
                match owned.compare_exchange_weak(
                    observed,
                    observed | mask,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        claimed[word_index] = mask;
                        break;
                    }
                    Err(current) => observed = current,
                }
            }
        }
        true
    }

    /// Return a persistent result slot only after its live capability was
    /// successfully revoked (or when no capability was ever published
    /// there).  Runtime-owned founding slots and extension slots (slice E)
    /// both release; the startup-link band
    /// (`STARTUP_LINK_SLOT_BASE..ARENA_SLOT_CEILING`) stays permanently
    /// reserved, and anything past the structural ceiling is not this
    /// bitmap's to touch.
    pub(crate) fn release_cap_slot(slot: kore_abi::session::SlotRef) {
        let slot = slot.index() as usize;
        let linker_band =
            (STARTUP_LINK_SLOT_BASE as usize..ARENA_SLOT_CEILING as usize).contains(&slot);
        if !linker_band && slot < super::ARENA_EXT_SLOT_CEILING as usize {
            let word = slot / u64::BITS as usize;
            let bit = slot % u64::BITS as usize;
            block().owned_slots[word].fetch_and(!(1u64 << bit), Ordering::Release);
        }
    }

    /// Record the `role::HEAP` supply slot — `boot_context`'s one-shot
    /// install from the stamped boot block.
    pub(crate) fn install_heap_slot(slot: kore_abi::session::SlotRef) {
        block()
            .heap_slot_p1
            .store(slot.index() as usize + 1, Ordering::Relaxed);
    }

    /// The `role::HEAP` supply's arena slot — `None` for a task founded
    /// without one: its allocator cannot grow (fail-closed `AllocError`).
    pub(crate) fn heap_supply_slot() -> Option<kore_abi::session::SlotRef> {
        match block().heap_slot_p1.load(Ordering::Relaxed) {
            0 => None,
            p1 => Some(kore_abi::session::SlotRef::from_wire((p1 - 1) as u32)),
        }
    }

    /// This thread's allocator shard index — its `thread_block` band slot,
    /// stamped at anchor.
    ///
    /// `None` before the anchor, and for a thread whose block lies outside
    /// the declared band.  Both are "this thread cannot prove which shard is
    /// privately its own"; the allocator's fallback for that answer is the
    /// only place the distinction is interpreted.
    pub(crate) fn shard_index() -> Option<usize> {
        match block().shard_index_p1.load(Ordering::Relaxed) {
            0 => None,
            p1 => Some(p1 - 1),
        }
    }

    /// This thread's own `role::ADDRESS_SPACE` arena slot (#407) —
    /// cached at anchor; `None` before it (structurally unreachable:
    /// every accessor runs post-anchor).
    pub(crate) fn self_space_slot() -> Option<kore_abi::session::SlotRef> {
        match block().self_space_slot_p1.load(Ordering::Relaxed) {
            0 => None,
            p1 => Some(kore_abi::session::SlotRef::from_wire((p1 - 1) as u32)),
        }
    }

    /// The base VA of a cap-slot's record in the R/O alias — the
    /// founder-stamped `geom::SLOTS` window for founding slots, the
    /// extend-time `geom::SLOTS_EXT` aliases for extension slots (slice
    /// E).  `None` when the governing window is unstamped or the record
    /// would fall outside its **live** aliased span.  The arithmetic
    /// itself is [`super::slot_alias_va`], pure and host-proven.
    pub(crate) fn slot_record_va(slot: kore_abi::session::SlotRef) -> Option<usize> {
        let b = block();
        super::slot_alias_va(
            slot.index() as usize,
            b.slots_va.load(Ordering::Relaxed),
            b.slots_len.load(Ordering::Relaxed),
            b.slots_ext_va.load(Ordering::Relaxed),
            b.slots_ext_len.load(Ordering::Acquire),
        )
    }

    /// Read this task's advisory snapshot without entering the kernel.
    pub(crate) fn execution_hint() -> Option<kore_abi::execution::ExecutionHint> {
        let b = block();
        let base = b.slots_va.load(Ordering::Relaxed);
        let len = b.slots_len.load(Ordering::Relaxed);
        if base == 0 || len == 0 {
            return None;
        }
        let ptr = base.checked_add(len)?;
        if ptr % kore_abi::execution::HINT_PAGE_BYTES as usize != 0 {
            return None;
        }
        // SAFETY: ABI-6 founding installs this task's initialized read-only
        // arena hint page immediately after the records. The thread block
        // and arena alias share the task lifetime, including worker teardown.
        unsafe { (&*(ptr as *const kore_abi::execution::ExecutionHintCell)).read() }
    }

    /// The stamped arena-extension alias base (`geom::SLOTS_EXT`), or
    /// `None` for a task the founder never stamped one — the extend
    /// verb's cheap local gate before it ever traps.
    pub(crate) fn slots_ext_va() -> Option<usize> {
        match block().slots_ext_va.load(Ordering::Relaxed) {
            0 => None,
            va => Some(va),
        }
    }

    /// Current live aliased extension span.  A report is validated and its
    /// bitmap interval opened before this value is raised, so an admitted
    /// allocator scan can never name a record whose bit is still reserved.
    pub(crate) fn slots_ext_len() -> usize {
        block().slots_ext_len.load(Ordering::Acquire)
    }

    /// Publish a validated, bitmap-opened extension span.  Release pairs
    /// with [`slot_record_va`]'s and [`live_slot_words`]'s Acquire reads; a
    /// raced or replayed report cannot rewind visibility.
    pub(crate) fn publish_slots_ext(new_len: usize) {
        block().slots_ext_len.fetch_max(new_len, Ordering::Release);
    }

    /// Open the persistent-name bitmap over the extension slot interval
    /// `[first, end)` (slice E) — [`install_report_slot`]'s opener, one
    /// band up: bits arrive reserved and only ever open here, so the
    /// startup-link band (below [`ARENA_SLOT_CEILING`]) and every
    /// not-yet-extended slot stay closed.  Refuses (opening nothing) any
    /// interval that reaches below the founding ceiling or past the
    /// structural one — the caller derived it from a kernel report, but
    /// the bitmap is this module's to guard.
    pub(crate) fn extend_cap_slots(first: usize, end: usize) -> bool {
        if first < ARENA_SLOT_CEILING as usize
            || end > super::ARENA_EXT_SLOT_CEILING as usize
            || first >= end
        {
            return false;
        }
        for (word_index, owned) in block().owned_slots.iter().enumerate() {
            let mask = slot_interval_word(word_index, first, end);
            if mask != 0 {
                // `0` = available; these bits were reserved-since-init and
                // no acquire can have claimed them, so the clear is the
                // first and only transition.
                owned.fetch_and(!mask, Ordering::Release);
            }
        }
        true
    }
}

/// **Anchor this thread** (#407, crt0's first act): stamp the
/// founder-mapped `geom::THREAD_BLOCK` page into a live
/// [`geometry::ThreadBlock`] and point the arch thread-pointer register
/// (`%fs` base / `TPIDR_EL0` / `tp`) at it through the SET_TP verb —
/// one session, staged into the descriptor scratch whose VA this
/// function parses from the boot page ITSELF (nothing here reads the
/// thread pointer; that is the whole point).  After this returns `true`,
/// every `geometry` accessor — and therefore every verb, the trap
/// transport, and the executor — is lawful on this thread.
///
/// Fail-closed `false` on any missing piece: no `.kore.geom`, no
/// `THREAD_BLOCK` window, no `role::ADDRESS_SPACE` self-view (a founding
/// that predates the thread substrate), no room for the SSA result, or a
/// refused session.  crt0 then [`refuse`]s — a task that cannot anchor
/// has no lawful session path at all.
#[cfg(target_os = "none")]
pub fn anchor_boot_thread(page: &'static [u8]) -> bool {
    anchor_boot_thread_inner(page)
}

/// **Carry this thread's runtime block into the TCB at `new_va`** — the
/// startup linker's TLS handoff, performed *before* the thread pointer
/// moves there.
///
/// A dynamically linked program's static TLS block claims the thread
/// pointer: `kore_ld`'s phase 4 points `%fs` (`TPIDR_EL0`, `tp`) at a TCB,
/// because that is what the TLS ABI requires of it.  The runtime block —
/// which the trap transport reads its scratch VA out of — used to be a
/// *different* page with nothing carrying it across, so a linked program
/// could reach its thread-locals **or** issue sessions, not both.
///
/// It carries across now.  `kore_ld` reserves
/// [`RUNTIME_TCB_BYTES`](kore_abi::image::RUNTIME_TCB_BYTES) at `tp` on
/// every architecture — above the blocks on variant 2, inside the header
/// it already reserves on variant 1 — and this moves the live block into
/// that reservation, the way glibc's TCB carries `struct pthread`.  One
/// pointer then serves both, and nothing is unreachable afterwards.
///
/// The block is **moved, not re-initialized**: the report slot, owned-slot
/// bitmap and heap supply the stub spent while linking belong to this
/// thread and its arena, not to the address the block happened to sit at.
/// Everything in it is position-independent except the self-pointer, which
/// is restamped.
///
/// Ordering is the same discipline `anchor_boot_thread` follows — the block
/// at the destination must already dereference the instant the register
/// moves — so the caller issues its SET_TP only after this returns `true`.
/// Fail-closed `false` on a null, misaligned, or unanchored source.
#[cfg(target_os = "none")]
pub fn migrate_thread_block(new_va: u64) -> bool {
    const BLOCK: usize = core::mem::size_of::<geometry::ThreadBlock>();
    const ALIGN: usize = core::mem::align_of::<geometry::ThreadBlock>();
    // The reservation the linker makes is the contract this moves into; a
    // block that outgrew it would be copied past the destination's end.
    const _: () = assert!(
        BLOCK as u64 <= kore_abi::image::RUNTIME_TCB_BYTES,
        "the runtime block must fit the TCB reservation kore_ld leaves at tp"
    );

    let new = new_va as usize;
    if new == 0 || new % ALIGN != 0 {
        return false;
    }
    let old = geometry::block_va();
    if old == 0 {
        return false;
    }
    if old == new {
        return true;
    }
    // SAFETY: `old` is this thread's live anchored block (its own thread
    // pointer names it); `new` is the linker's freshly mapped, zeroed,
    // exclusively-this-thread TCB reservation, at least `BLOCK` bytes and
    // disjoint from `old` (a private allocation, not the founder window).
    // Startup is single-threaded, so no peer observes the half-copied block.
    unsafe {
        core::ptr::copy_nonoverlapping(old as *const u8, new as *mut u8, BLOCK);
        geometry::ThreadBlock::restamp_self(new);
        // The startup linker and child startup call this only for a TCB
        // built from the static TLS layout, so this image has static TLS. Child spawning requires the process-root template before
        // allowing another thread to execute TLS-bearing code.
        geometry::mark_static_tls(new);
    }
    true
}

/// Off the metal targets there is no thread pointer, and no block to move.
#[cfg(not(target_os = "none"))]
pub fn migrate_thread_block(_new_va: u64) -> bool {
    false
}

/// [`anchor_boot_thread`]'s body.
#[cfg(target_os = "none")]
fn anchor_boot_thread_inner(page: &'static [u8]) -> bool {
    use kore_abi::session::{DERIVE_KIND_SET_TP, SlotRef, Status};

    let Ok(rt_probe) = boot_parse(page) else {
        return false;
    };
    let (bootstrap, Some(geom), _, _, _) = rt_probe else {
        return false;
    };
    let Some((tblock_va, tblock_len)) = geom.window(kore_abi::task::geom::THREAD_BLOCK) else {
        return false;
    };
    if (tblock_len as usize) < core::mem::size_of::<geometry::ThreadBlock>() {
        return false;
    }
    let process_root_va = match geom.window_indexed(kore_abi::task::geom::THREAD_BLOCK, 1) {
        Some((va, len))
            if (len as usize) >= core::mem::size_of::<geometry::ThreadBlock>()
                && (va as usize).is_multiple_of(core::mem::align_of::<geometry::ThreadBlock>()) =>
        {
            va as usize
        }
        Some(_) => return false,
        // Thread zero is the root and predates the indexed spelling; every
        // runtime-spawned worker is stamped with @1 below.
        None => tblock_va as usize,
    };
    // The startup stub and the final image deliberately share this crt0.
    // Stub-side linking already anchored the initial thread because it issued
    // sessions; direct-static entry (currently exec's first shell) did not.
    // Observe only the post-commit state, never the preparatory self-pointer:
    // a refused SET_TP leaves state 1 and must be retried, not mistaken for a
    // live `%fs` base.
    // SAFETY: the geometry block names this thread's live founder-mapped
    // THREAD_BLOCK window, and its size was checked above.
    if unsafe { geometry::ThreadBlock::is_anchored(tblock_va as usize) } {
        return true;
    }
    let Some((scratch_va, _)) = geom.window(kore_abi::task::geom::SCRATCH) else {
        return false;
    };
    let Some(addr_space) = bootstrap.handle(role::ADDRESS_SPACE, 0) else {
        return false;
    };
    // The SSA result: one past the highest block-named slot (the
    // exhaustive-list convention) — SET_TP writes only the vacant record
    // there, so the serve-scratch run that later derives the same base
    // still sees vacancy.
    let Some(top) = bootstrap.handles().map(|(_, _, s)| s.index()).max() else {
        return false;
    };
    let Some(result) = top.checked_add(1).filter(|&slot| slot < ARENA_SLOT_CEILING) else {
        return false;
    };
    // Stamp the block FIRST (the value SET_TP installs must already
    // dereference to a valid self-pointer the instant `%fs` goes live).
    // SAFETY: the founder mapped `[tblock_va, +len)` R/W, zeroed,
    // exclusively this thread's.
    unsafe {
        geometry::ThreadBlock::init(
            tblock_va as usize,
            process_root_va,
            addr_space,
            page.as_ptr() as usize,
        )
    };
    let mut b = crate::session::SessionBuilder::new();
    let tp_arg = tblock_va.to_le_bytes();
    if b.derive_raw(
        addr_space,
        kore_abi::cap::Rights::empty(),
        DERIVE_KIND_SET_TP,
        SlotRef::from_wire(result),
        Some(&tp_arg),
    )
    .is_err()
    {
        return false;
    }
    // The common transport first tries the register carrier, so this compact
    // SET_TP normally needs no scratch access at all.  `scratch_va` remains the
    // pre-thread-pointer fallback for an unexpectedly enlarged descriptor —
    // ordinary transport geometry is not readable until this call succeeds.
    let Ok(committed) = crate::transport::trap_session(&b, Some(scratch_va as usize)) else {
        return false;
    };
    if committed.status() != Status::Ok {
        return false;
    }
    // SAFETY: the same checked exclusive THREAD_BLOCK window; publication is
    // after the successful SET_TP commit, which is the fact state 2 denotes.
    unsafe { geometry::ThreadBlock::mark_anchored(tblock_va as usize) };
    true
}

/// **The calling worker thread's entry context** — what a
/// [`Worker`](crate::service::Worker) entry uses to reach the
/// capabilities its service declared for it.
///
/// Thread 0 receives its boot page as crt0's argument and carries it in
/// an [`Rt`] for the task's life; a worker's entry takes no argument, so
/// it asks here instead. The answer is the same
/// validated page and the same [`Bootstrap`] lookup — a worker resolves
/// `role::DEVICE` exactly as a founded service does, and nothing about
/// being a thread gives it a shortcut past the possession verbs.
///
/// `None` before the thread is anchored, or if its page fails
/// validation — fail-closed, like every other geometry reader.
#[cfg(target_os = "none")]
pub fn thread_context() -> Option<Rt> {
    let va = geometry::boot_page_va()?;
    // SAFETY: the founding delegated this page into the thread's space
    // for its whole life, and `anchor_boot_thread` already parsed it once
    // from the same address.
    let page: &'static [u8] =
        unsafe { core::slice::from_raw_parts(va as *const u8, BOOT_PAGE_BYTES) };
    boot_context(page).ok()
}

/// This thread's validated fast-clock block without rebuilding an [`Rt`].
/// The boot-page VA is per-thread state; the block itself is immutable for
/// the lifetime of that page.
#[cfg(target_os = "none")]
pub(crate) fn current_clock_page() -> Option<&'static ClockPage> {
    let va = geometry::boot_page_va()?.checked_add(CLOCK_BLOCK_AT)?;
    // SAFETY: the anchored thread block names its live one-page bootstrap
    // mapping; the compile-time fit assertion keeps ClockPage inside it and
    // CLOCK_BLOCK_AT preserves its natural alignment.
    let page = unsafe { &*(va as *const ClockPage) };
    page.validate().then_some(page)
}

#[cfg(not(target_os = "none"))]
pub(crate) fn current_clock_page() -> Option<&'static ClockPage> {
    None
}

/// Off the metal userland targets — no thread, so no thread context.
#[cfg(not(target_os = "none"))]
pub fn thread_context() -> Option<Rt> {
    None
}

/// The shared fail-closed boot-page parse — [`boot_context`]'s reading
/// half, also used by [`anchor_boot_thread`] before `%fs` exists. Returns the
/// validated bootstrap plus the geometry block when stamped.
fn boot_parse(
    page: &'static [u8],
) -> Result<
    (
        Bootstrap<'static>,
        Option<&'static GeomBlock>,
        Option<&'static ClockPage>,
        Option<&'static FoundingBlock>,
        Option<&'static MachineBlock>,
    ),
    EntryError,
> {
    let boot_bytes = page.get(BOOT_BLOCK_AT..).ok_or(EntryError::ShortPage)?;
    let (block, _) = BootBlock::ref_from_prefix(boot_bytes).map_err(|_| EntryError::ShortPage)?;
    let bootstrap = Bootstrap::of(block).map_err(EntryError::BadBoot)?;
    let geom = page
        .get(GEOM_BLOCK_AT..)
        .and_then(|b| GeomBlock::ref_from_prefix(b).ok())
        .map(|(g, _)| g)
        .filter(|g| g.validate());
    let clock = page
        .get(CLOCK_BLOCK_AT..)
        .and_then(|b| ClockPage::ref_from_prefix(b).ok())
        .map(|(clock, _)| clock)
        .filter(|clock| clock.validate());
    let founding = page
        .get(FOUNDING_BLOCK_AT..)
        .and_then(|b| FoundingBlock::ref_from_prefix(b).ok())
        .map(|(f, _)| f)
        .filter(|f| f.validate());
    let machine = page
        .get(MACHINE_BLOCK_AT..)
        .and_then(|b| MachineBlock::ref_from_prefix(b).ok())
        .map(|(m, _)| m)
        .filter(|m| m.validate());
    Ok((bootstrap, geom, clock, founding, machine))
}

/// The **entry transport** — statically bound like the allocator's grow
/// transport (§2.2): the trap transport, unconditionally.  Off the
/// userland target every commit refuses [`TrapUnavailable`]
/// (`crate::transport::TrapUnavailable`) — fail-closed, never simulated.
type EntryTransport = crate::transport::TrapTransport;

/// **The task's one trap transport**, bound `'static`.
///
/// Zero-sized and `const`-constructible, so a static costs nothing — and
/// it is what makes the §5.4 entry contract expressible as a single
/// `async fn main(rt: Rt) -> !`.  A root future that borrows a *stack
/// local* transport cannot also BE the root the executor drives: the
/// borrow outlives the frame that owns it.  Every entry used to work
/// around that by owning its executor privately, which is why
/// `#[kore_rt::main]` had no consumers and the idiom it documents was
/// never the idiom anyone followed.
static TRANSPORT: EntryTransport = EntryTransport::new();

/// The task's trap transport — see [`TRANSPORT`].  Handed out by
/// reference so a bespoke entry composes with the same one the executor
/// and every client/host in the task already use.
pub fn transport() -> &'static EntryTransport {
    &TRANSPORT
}

/// Run the service's root task forever — what the `#[kore_rt::main]`
/// expansion calls: resolve the doorbell, bind the executor over the
/// statically-selected transport, **pin the root future in this frame**
/// (§3.3 — this frame *is* the crt0 call chain), and run the §3.1
/// cycle. Never returns: a completed root (impossible for a conforming
/// `-> !` main) [`refuse`]s; a missing doorbell is still late enough to
/// [`refuse_with`] the shared attestation rung because `Rt` is already live.
pub fn run_main<F, Fut>(rt: Rt, root: F) -> !
where
    F: FnOnce(Rt) -> Fut,
    Fut: core::future::Future,
{
    run_main_at(rt, RUNG_SERVICE_ATTEST, root)
}

/// Run an ordinary root future while preserving an established service's
/// attestation rung vocabulary.
///
/// New services use [`run_main`] and the shared [`RUNG_SERVICE_ATTEST`]. This
/// seam exists for older boot rows whose on-metal NACK decoder already assigns
/// another append-only rung to a missing executor doorbell.
#[doc(hidden)]
pub fn run_main_at<F, Fut>(rt: Rt, attest_rung: u64, root: F) -> !
where
    F: FnOnce(Rt) -> Fut,
    Fut: core::future::Future,
{
    run_main_with_tasks_at(rt, attest_rung, root, || {
        Ok::<[SpawnToken; 0], SpawnRefused>([])
    })
}

/// Run a root future with one **declared startup-task set**.
///
/// This is the behavioral spine behind
/// `#[kore_rt::main(tasks(worker(), ...))]`. `startup` constructs the
/// complete fixed array before any token is admitted; Rust drops every
/// already-constructed affine token if a later constructor refuses. The
/// fresh executor then admits the whole array before `root(rt)` is even
/// created, so the root's first poll observes the declared task topology
/// rather than racing its construction. `N` is bounded by [`MAX_TASKS`]
/// and no heap or dynamic task list exists.
///
/// A constructor or admission refusal is fatal at the shared service-entry
/// attestation rung: serving with fewer tasks than the declaration would
/// advertise a topology that does not exist.
#[doc(hidden)]
pub fn run_main_with_tasks<F, Fut, S, const N: usize>(rt: Rt, root: F, startup: S) -> !
where
    F: FnOnce(Rt) -> Fut,
    Fut: core::future::Future,
    S: FnOnce() -> Result<[SpawnToken; N], SpawnRefused>,
{
    run_main_with_tasks_at(rt, RUNG_SERVICE_ATTEST, root, startup)
}

/// [`run_main_with_tasks`] with an established service-specific attestation
/// rung. See [`run_main_at`].
#[doc(hidden)]
pub fn run_main_with_tasks_at<F, Fut, S, const N: usize>(
    rt: Rt,
    attest_rung: u64,
    root: F,
    startup: S,
) -> !
where
    F: FnOnce(Rt) -> Fut,
    Fut: core::future::Future,
    S: FnOnce() -> Result<[SpawnToken; N], SpawnRefused>,
{
    let Some(doorbell) = rt.doorbell() else {
        // No wake edge means the task cannot idle lawfully (§3.1), but this
        // failure occurs after crt0 built a valid runtime and therefore must
        // use the founder's diagnostic channel rather than disappear into a
        // bare spin.
        refuse_with(&rt, attest_rung)
    };
    if N > MAX_TASKS {
        refuse_with(&rt, attest_rung)
    }
    let mut ex = Executor::new(&TRANSPORT, doorbell);
    let Ok(tasks) = startup() else {
        refuse_with(&rt, attest_rung)
    };
    for task in tasks {
        if ex.spawn(task).is_err() {
            // `N <= MAX_TASKS` and this executor is fresh, so this is an
            // invariant witness rather than ordinary backpressure.
            refuse_with(&rt, attest_rung)
        }
    }
    // Keep the root future in this crt0-owned frame.  Passing it by value to
    // `Executor::run` would materialize the large service future in both this
    // frame and the executor's nested frame; the widest service then exhausts
    // the fixed founding stack before its first poll.
    let mut root = core::pin::pin!(root(rt));
    let _ = ex.run_pinned(root.as_mut());
    refuse()
}

/// **Spawn a service's declared worker threads**
/// ([`SystemService::WORKERS`](crate::service::SystemService::WORKERS)) —
/// the one place the runtime turns a declaration into threads, so no
/// service body carries spawn code.
///
/// Runs after `start_on_metal` (the service's own state is built, and a
/// worker may read it) and before readiness is advertised: a client that
/// observes `COOKIE_SERVING` is talking to a service whose workers
/// already exist, never to a half-built one.
///
/// A refused spawn is **fatal to the service**: it declared a wait it
/// cannot perform, and serving anyway would advertise a topology that
/// does not exist (a driver whose relay never spawned answers requests
/// while its device goes unread).  The refusal names the worker on the
/// beacon rung, then the caller refuses.
///
/// **Public** because two services drive their own entry loop rather
/// than [`run_service`] — a driver row that must arm an IRQ park or
/// prove its controller before serving cannot use the generic ladder.
/// Those loops still call this, so the *mechanism* has exactly one
/// implementation no matter which loop runs it; what a bespoke loop
/// customizes is its own readiness proof, never how its threads come to
/// exist.
fn worker_caps_with_heap(
    heap: kore_abi::session::SlotRef,
    declared: &crate::service::WorkerEndowment,
    out: &mut [crate::thread::WorkerCap; crate::thread::MAX_WORKER_CAPS],
) -> Option<usize> {
    use kore_abi::cap::{ResourceKind, Rights};
    use kore_abi::task::role;

    // The global allocator is process-wide memory but its grow operation is
    // capability-gated per task. A worker therefore needs its own explicit
    // lessor position over the founder-seeded supply; sharing the parent's VAS
    // does not make the parent's holding reachable. The no-arg self-kind
    // MemLease derive is the session engine's founding/endowment attenuation:
    // the parent keeps the supply and the child receives a whole-span sublease.
    if declared.len() >= out.len() {
        return None;
    }
    out[0] = crate::thread::WorkerCap {
        role: role::HEAP,
        index: 0,
        subject: heap,
        kind: ResourceKind::MemLease as u16 as u8,
        // Heap growth needs only carve + map authority. The parent retains
        // REVOKE over this sub-lessor; the worker does not receive teardown
        // authority it never exercises.
        rights: Rights::ACCESS | Rights::MODIFY | Rights::DERIVE,
    };
    for (i, cap) in declared.caps().enumerate() {
        // HEAP@0 is runtime infrastructure, not a service-selected device
        // authority. Reject a duplicate instead of letting boot stamping fail
        // later with a less precise symptom.
        if cap.role == role::HEAP && cap.index == 0 {
            return None;
        }
        out[i + 1] = cap;
    }
    Some(declared.len() + 1)
}

pub fn spawn_workers<S: crate::service::SystemService>(rt: &Rt, service: &S) -> bool {
    for (n, w) in S::WORKERS.iter().enumerate() {
        if !service.worker_enabled(n) {
            continue;
        }
        // The service names what this worker must hold; the runtime owns
        // the mechanism.  Asked per worker rather than once, so two
        // workers of one service can be endowed differently — a relay
        // holding its device, a witness holding nothing.
        let mut endow = crate::service::WorkerEndowment::new();
        service.worker_endowment(n, &mut endow);
        let mut caps = [crate::thread::WorkerCap {
            role: 0,
            index: 0,
            subject: kore_abi::session::SlotRef::from_wire(0),
            kind: 0,
            rights: kore_abi::cap::Rights::empty(),
        }; crate::thread::MAX_WORKER_CAPS];
        let Some(heap) = rt.bootstrap().handle(kore_abi::task::role::HEAP, 0) else {
            refuse_with(rt, RUNG_WORKER_SPAWN)
        };
        let Some(cap_count) = worker_caps_with_heap(heap, &endow, &mut caps) else {
            refuse_with(rt, RUNG_WORKER_SPAWN)
        };
        // **Detached on purpose**: a declared worker owns its thread for
        // the service's whole life, so there is no join to come back to
        // and nothing to give its band back to.  Named rather than
        // dropped, so the decision reads as one.
        match crate::thread::spawn_with(w.entry, &caps[..cap_count]) {
            Some(thread) => thread.detach(),
            None => refuse_with(rt, RUNG_WORKER_SPAWN),
        }
    }
    true
}

/// The refusal rung a service reports when a declared worker could not
/// be spawned (see [`refuse_with`] — the founder prints the number).
pub const RUNG_WORKER_SPAWN: u64 = 3;

/// Generic service startup rungs.  They are shared because the ladder itself
/// is shared; service-specific bespoke entries keep their own maps.
pub const RUNG_SERVICE_ATTEST: u64 = 1;
pub const RUNG_SERVICE_ENDOWMENT: u64 = 2;
pub const RUNG_SERVICE_START: u64 = 4;
pub const RUNG_SERVICE_SERVE_SETUP: u64 = 5;

/// The honest terminal refusal: no console, and — deliberately — no exit.
/// Spin, so the failure is *visible* (a wedged task) rather than
/// undefined.
///
/// **Not** `thread::exit`, even though that now exists: every caller here
/// is a task that could not build the very machinery an exit needs (no
/// geometry, no anchored `%fs`, no boot page, so no SELF handle to
/// present). Exiting would also erase the wedge, and a wedged task is the only
/// diagnostic a serial-less machine leaves behind. A task that *can* end
/// lawfully uses [`crate::thread::exit`], which presents its SELF capability
/// directly to the kernel; the retired exec-service exit opcode is never a
/// lifecycle path.
///
/// Prefer [`refuse_with`] wherever an `Rt` is in hand: a mute refusal tells
/// the founder *nothing*, and it will burn its whole readiness budget
/// before declaring "no signal".
///
/// A refused task is terminal either way; what differs is the cost of the
/// corpse.  This used to be a bare `spin_loop`, which burns the task's whole
/// core forever — on a laptop, a hot zombie found by fan noise; on a busy
/// machine, a core stolen from live work; and in every case a wedge
/// indistinguishable from a hang.  It now **pauses on a hardware deadline
/// with nothing watched**: the refusal semantics are identical (never
/// returns, never progresses), but the monitor gives the core back.  Where
/// the machine or this early context has no monitor/clock,
/// [`pause_until`](crate::hint::pause_until) degrades by contract to exactly
/// the spin this always was.
pub fn refuse() -> ! {
    loop {
        // `pause_until`, not `wait_on` on a never-stored word: there is no
        // store coming, so watching for one arms this core's address monitor
        // once a second, forever, to observe an event that cannot happen —
        // and there is exactly ONE monitor per logical processor, so a wedged
        // task was quietly retiring whatever else had armed it. A timed pause
        // watches nothing (Intel's `TPAUSE` arms nothing at all), which is the
        // honest shape for "this task is over, give the core back".
        let _ = crate::hint::pause_until(core::time::Duration::from_secs(1));
    }
}

/// Refuse **with a NACK**: best-effort send
/// `Notice { cookie: COOKIE_REFUSED, payload: rung }` synchronously on the
/// founder-delivered `role::PARENT_SEND` endpoint, then exit through the
/// task's founder-delivered `role::SELF` handle. An early context without
/// SELF retains the fail-closed terminal park ([`refuse`]).
///
/// The founder observing the beacon channel fails the row *immediately*
/// with the rung code on its console — instead of waiting out the full
/// readiness budget and reporting "no signal", which on a serial-less
/// machine was the difference between a named rung and two mute minutes.
/// `rung` values are the service's own ladder positions, documented at its
/// entry (each service's map is its own; the founder prints the number).
///
/// The synchronous notice path is allocation-free and needs no executor or
/// doorbell.  That is load-bearing for startup and panic failures: nesting a
/// second executor on an already-deep 64 KiB stack can reproduce the failure
/// it is trying to report. Exiting is the persistent post-readiness witness:
/// the one-shot parent channel may no longer be drained, but the founder
/// still owns the task handle and can observe the typed exit rung before
/// targeted teardown. A NACK sender acquires no new delivery/IRQ/join wait.
/// Missing beacon authority still exits when SELF exists; missing SELF
/// degrades to [`refuse`].
pub fn refuse_with(rt: &Rt, rung: u64) -> ! {
    use crate::svc::COOKIE_REFUSED;

    if let Some(beacon) = rt.bootstrap().handle(role::PARENT_SEND, 0) {
        let _ = verbs::post_notice(beacon, COOKIE_REFUSED, rung);
    }
    if rt.bootstrap().handle(role::SELF, 0).is_some() {
        crate::thread::exit(crate::exit::ExitCode::service_fault(rung));
    }
    refuse()
}

/// The bring-up rung a panic reports. A panic is not a deliberate service
/// ladder step, so it occupies a distinct high value.
pub const RUNG_PANIC: u64 = 0x7f;

#[cfg(not(target_os = "none"))]
fn begin_panic() -> bool {
    use core::sync::atomic::{AtomicBool, Ordering};

    // Host builds have no TP-anchored ThreadBlock. `panic_refuse` is a
    // metal panic-handler helper, but retain a recursion-safe fallback so the
    // library remains independently buildable.
    static PANICKING: AtomicBool = AtomicBool::new(false);
    !PANICKING.swap(true, Ordering::AcqRel)
}

#[cfg(target_os = "none")]
fn begin_panic() -> bool {
    geometry::begin_panic()
}

/// Pack a panic site into one refusal payload.
///
/// `[7:0]` is [`RUNG_PANIC`], `[39:8]` the source line, and `[63:40]` a
/// 24-bit FNV-1a digest of the file name.  This remains one allocation-free
/// notice while distinguishing equal line numbers across modules.
fn panic_payload(file: &str, line: u32) -> u64 {
    let mut hash: u32 = 0x811c_9dc5;
    let mut i = 0;
    while i < file.len() {
        hash ^= file.as_bytes()[i] as u32;
        hash = hash.wrapping_mul(0x0100_0193);
        i += 1;
    }
    RUNG_PANIC | ((line as u64) << 8) | ((u64::from(hash) & 0xff_ffff) << 40)
}

/// Shared `#[panic_handler]` body for founded tasks.
///
/// The current thread's already-anchored runtime block names its own boot page;
/// rebuilding [`Rt`] from that page avoids a writable global context, avoids
/// cross-thread arena confusion, and deliberately excludes the startup linker
/// stub (which has no anchored final-image thread context).  Any failure in
/// this emergency path degrades to the terminal park ([`refuse`]).
pub fn panic_refuse(info: &core::panic::PanicInfo<'_>) -> ! {
    let Some(rt) = thread_context() else {
        refuse();
    };
    if !begin_panic() {
        refuse();
    }
    let payload = info
        .location()
        .map_or(RUNG_PANIC, |at| panic_payload(at.file(), at.line()));
    panic_terminate(&rt, payload)
}

/// Exit status used by an exec-founded task whose panic handler can present
/// its own `role::SELF` authority. Worker supervisors translate this stable
/// value into a typed panic outcome rather than an application exit code.
///
/// The raw `i32` spelling of [`ExitCode::PANIC`](crate::exit::ExitCode::PANIC),
/// kept because a supervisor matches it against the ABI's signed status word
/// as it comes back off a join.
pub const PANIC_EXIT_STATUS: i32 = crate::exit::ExitCode::PANIC.get();

#[cfg(all(target_arch = "x86_64", target_os = "none"))]
fn panic_terminate(rt: &Rt, payload: u64) -> ! {
    use crate::svc::COOKIE_REFUSED;

    // Best-effort attribution first. Every fully founded task receives SELF;
    // contexts that fail before that delivery retain the terminal park.
    if let Some(beacon) = rt.bootstrap().handle(role::PARENT_SEND, 0) {
        let _ = verbs::post_notice(beacon, COOKIE_REFUSED, payload);
    }
    if rt.bootstrap().handle(role::SELF, 0).is_some() {
        crate::thread::exit(crate::exit::ExitCode::PANIC);
    }
    refuse()
}

#[cfg(not(all(target_arch = "x86_64", target_os = "none")))]
fn panic_terminate(rt: &Rt, payload: u64) -> ! {
    use crate::svc::COOKIE_REFUSED;

    if let Some(beacon) = rt.bootstrap().handle(role::PARENT_SEND, 0) {
        let _ = verbs::post_notice(beacon, COOKIE_REFUSED, payload);
    }
    if rt.bootstrap().handle(role::SELF, 0).is_some() {
        crate::thread::exit(crate::exit::ExitCode::PANIC);
    }
    refuse()
}

// ─── run_service — the bootstrapped-service entry (the reader half) ───

// The `CLAIM_SUPPLY` sentinel family (`CLAIM_SUPPLY` / `CLAIM_SUPPLY_OK` /
// the 17-byte `CLAIM_REPORT_OFF` scratch report) is DELETED (#336 rung 4):
// [`verbs::claim_supply`](crate::cap::verbs::claim_supply) now stages a
// no-arg `Derive` on the supply slot through the trap transport; the
// geometry answers as a bind-inert [`GeomReport`](kore_abi::session::GeomReport)
// in the derived report slot, and the rights ride the supply record itself.

// The `MAP_VIEW` sentinel family (`MAP_VIEW` / `MAP_VIEW_OK` / `MAP_VIEW_VA`)
// is DELETED (#336 rung 2): [`verbs::map_view`](crate::cap::verbs::map_view)
// now stages a real session (`Derive` on the delivered view's slot) through
// the trap transport; the kernel's map flavor installs at the task's
// founder-installed map-window cursor and answers with a bind-inert
// [`GeomReport`](kore_abi::session::GeomReport) in the derived
// [`map_report_slot`](geometry) — no fixed VA, no in-band status word.

// The `HEAP_GROW` sentinel family (`HEAP_GROW` / `HEAP_VA` /
// `HEAP_CAPACITY`) is DELETED (#336 rung 6): a founded service's allocator
// grows through ONE ordinary session per miss — a detached window carve off
// its founder-seeded `role::HEAP` supply plus the staged map flavor — with
// each extent landing at the task's map-window cursor.  The supply's size is
// the boot manifest's `[layout] heap_bytes` row; no fixed VA, no page-count
// trap, no capability-free kernel arm.

/// **The cap-arena slot ceiling** — the kernel's `ARENA_SLOTS`, mirrored here so
/// the userland side of the founder layout can bound a slot convention without
/// guessing.  **Cross-boundary contract**: this MUST equal
/// `crate::runtime::session::ARENA_SLOTS` (the kernel asserts the equality at
/// compile time); a founded service is founded with exactly this many slots
/// (`found_service`'s deliberate full-arena override, so its scratch/slot
/// aliases start page-aligned).
///
/// 256 (from 192): the program-loading ring widened
/// exec's convey table to `MAX_CONVEY = 96` and both spawn ceilings with
/// it, pushing the engine's single-flight bands past the old linker
/// base; two more pages give the widened engine its room and move the
/// durable band up intact.  `256 % 32 == 0`: an eight-page arena.
///
/// 256, final (slice E): the ladder of raises ends here — this is now the
/// **founding** ceiling and the extension **granule**, not a wall.  A
/// task that needs more slots grows in 256-slot chunks at runtime
/// ([`ARENA_EXT_SLOT_CEILING`] bounds the growth), so the next pressure
/// on this number is answered by an extend, not a fifth move.
pub const ARENA_SLOT_CEILING: u32 = 256;

/// **The growable arena's structural ceiling** (slice E): the founding
/// array plus every extension chunk the kernel's 31-entry directory can
/// hold — `256 + 31·256 = 8192` slots.
///
/// [`ARENA_SLOT_CEILING`] above stops being a wall with this: it is the
/// *founding* ceiling and the extension **granule** — a task founded at
/// the full 256 grows in 256-slot chunks through
/// [`verbs::extend_arena`](crate::cap::verbs::extend_arena), each chunk
/// aliased read-only in the `geom::SLOTS_EXT` window so extension slots
/// read as one dense VA run above slot 255.  Slots `224..256` (the
/// startup-link band) stay permanently reserved; extension slots live
/// entirely above them.  Asserted equal to the kernel's
/// `ARENA_EXT_SLOT_CEILING` by the kernel's own host tests, exactly as
/// the founding ceiling is.
pub const ARENA_EXT_SLOT_CEILING: u32 = 8192;

const _: () = assert!(
    ARENA_EXT_SLOT_CEILING % ARENA_SLOT_CEILING == 0 && ARENA_EXT_SLOT_CEILING > ARENA_SLOT_CEILING,
    "extension chunks are whole founding granules"
);

/// First arena slot reserved for durable startup-linker names.
///
/// The linker stub and a resident service share one arena across the jump.
/// Everything below this boundary belongs to the service/runtime protocol;
/// everything at or above it is the stub's monotone result band. Keeping the
/// boundary here makes both users prove against the same value.
///
/// The band keeps its 32-slot width; it moved up with the ceiling (128 →
/// 160 → 224) so the widened spawn engine below it gained the whole of
/// each new page.  It did NOT move when the arena learned to grow (slice
/// E): extension slots begin at [`ARENA_SLOT_CEILING`], so the band is a
/// permanent island between the runtime-owned founding slots and the
/// extension space.
pub const STARTUP_LINK_SLOT_BASE: u32 = 224;

/// Storage words needed to represent every arena slot this task could
/// ever own — derived from [`ARENA_EXT_SLOT_CEILING`] (slice E widened it
/// from the founding ceiling: a founding-sized bitmap would have silently
/// capped persistent ownership at 256 names after the arena grew, the
/// exact bug the original single-word bitmap had at 64).  128 words —
/// 1 KiB of the thread block's one page, bounds re-proven below.
const OWNED_SLOT_WORDS: usize =
    (ARENA_EXT_SLOT_CEILING as usize + u64::BITS as usize - 1) / u64::BITS as usize;

const _: () = assert!(
    OWNED_SLOT_WORDS * u64::BITS as usize >= ARENA_EXT_SLOT_CEILING as usize,
    "the owned-slot bitmap must cover the whole extensible arena"
);

/// Record VA of `slot` under the **two-window slot-alias plan** (slice E):
/// a founding slot reads at `slots_va + slot·128` inside the stamped
/// founding span; an extension slot (`≥ ARENA_SLOT_CEILING`) reads at
/// `slots_ext_va + (slot − 256)·128` inside the **live** aliased span —
/// extension chunks alias back-to-back at chunk stride, so the extension
/// space is one dense run and the arithmetic needs no chunk index.
///
/// Fail-closed on every edge: an unstamped window (`va == 0`), a founding
/// slot past the founding span (a 64-slot thread arena's slot 65 is
/// *nothing*, never an extension read), an extension slot past what the
/// kernel has actually reported aliased, and any overflow.  Pure — the
/// host suite proves the arithmetic without an anchored thread block.
#[cfg(any(target_os = "none", test))]
fn slot_alias_va(
    slot: usize,
    slots_va: usize,
    slots_len: usize,
    slots_ext_va: usize,
    slots_ext_len: usize,
) -> Option<usize> {
    let rec_bytes = core::mem::size_of::<kore_abi::cap::CapRecord>(); // 128, pinned
    if slot < ARENA_SLOT_CEILING as usize {
        if slots_va == 0 {
            return None; // unstamped ⇒ no alias to read (fail-closed)
        }
        let off = slot.checked_mul(rec_bytes)?;
        // The record must lie wholly within the aliased founding span.
        if off.checked_add(rec_bytes)? > slots_len {
            return None;
        }
        return slots_va.checked_add(off);
    }
    if slots_ext_va == 0 || slot >= ARENA_EXT_SLOT_CEILING as usize {
        return None;
    }
    let off = (slot - ARENA_SLOT_CEILING as usize).checked_mul(rec_bytes)?;
    // ...and an extension record wholly within the LIVE aliased span.
    if off.checked_add(rec_bytes)? > slots_ext_len {
        return None;
    }
    slots_ext_va.checked_add(off)
}

/// Bits in `word` belonging to the half-open slot interval `[first, limit)`.
/// Padding beyond the arena and the startup-link result band are absent from
/// the mask and therefore remain permanently reserved.
#[cfg(any(target_os = "none", test))]
const fn slot_interval_word(word: usize, first: usize, limit: usize) -> u64 {
    const WORD_BITS: usize = u64::BITS as usize;

    let base = word * WORD_BITS;
    let lo_raw = first.saturating_sub(base);
    let hi_raw = limit.saturating_sub(base);
    let lo = if lo_raw < WORD_BITS {
        lo_raw
    } else {
        WORD_BITS
    };
    let hi = if hi_raw < WORD_BITS {
        hi_raw
    } else {
        WORD_BITS
    };
    let below_hi = if hi == WORD_BITS {
        u64::MAX
    } else if hi == 0 {
        0
    } else {
        (1u64 << hi) - 1
    };
    let below_lo = if lo == WORD_BITS {
        u64::MAX
    } else if lo == 0 {
        0
    } else {
        (1u64 << lo) - 1
    };
    below_hi & !below_lo
}

/// **The serve-scratch slot convention** (Phase-A checkpoint G2a-3a) — where a
/// serving service's
/// [`ServeEndpoints::scratch`](crate::svc::ServeEndpoints::scratch) slots live:
/// the [`MAX_GRANTS`](crate::service::MAX_GRANTS) arena slots **immediately past
/// the highest slot the founding's stamped [`BootBlock`] names**.
///
/// A window-carving grant stages its derive into one of these (SSA, one per
/// possible grant) and the following send moves the staged child; they hold no
/// authority between requests, so they need no role, no stamp, and no ABI
/// change — only *vacancy*.  The convention is derived, not declared: the boot
/// block is the exhaustive list of slots the founding wrote (its endowment,
/// wiring, and beacon roles plus the reserved doorbell landing slot), so
/// everything above `max(named) + 1` is untouched arena, and the founder's
/// full-[`ARENA_SLOT_CEILING`] arena leaves ample room (mm's highest named slot
/// is the beacon at 7, so its scratch begins at 8 with ample room below 128).
///
/// Chosen over a stamped `SVC_SCRATCH` role because a role costs four of the
/// sixteen frozen boot-handle entries per service *and* a founder change, to
/// express a fact the block already determines.  [`serve_scratch_base`] computes
/// it; a base that would run past the ceiling refuses (fail-closed).
pub const SERVE_SCRATCH_SLOTS: usize = crate::service::MAX_GRANTS;

/// The derived runtime-only arena layout after the boot block's highest named
/// slot.  All four consumers use this one calculation so a budget change
/// cannot make serve scratch, the report answer, heap carve, and startup-link
/// result band silently overlap again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RuntimeSlots {
    scratch_base: u32,
    report: u32,
}

fn runtime_slots_after(top: u32) -> Option<RuntimeSlots> {
    let scratch_base = top.checked_add(1)?;
    let report = scratch_base.checked_add(SERVE_SCRATCH_SLOTS as u32)?;
    let carve = report.checked_add(1)?;
    // `STARTUP_LINK_SLOT_BASE..ARENA_SLOT_CEILING` is the durable linker
    // result band. The heap carve is the last runtime-owned slot and therefore
    // must remain strictly below it.
    (carve < STARTUP_LINK_SLOT_BASE).then_some(RuntimeSlots {
        scratch_base,
        report,
    })
}

#[cfg(test)]
mod runtime_slot_tests {
    use super::*;

    fn worker_cap(role: u16, index: u16) -> crate::thread::WorkerCap {
        crate::thread::WorkerCap {
            role,
            index,
            subject: kore_abi::session::SlotRef::from_wire(90 + u32::from(index)),
            kind: kore_abi::cap::ResourceKind::PlatformDevice as u16 as u8,
            rights: kore_abi::cap::Rights::ACCESS,
        }
    }

    fn vacant_worker_caps() -> [crate::thread::WorkerCap; crate::thread::MAX_WORKER_CAPS] {
        [worker_cap(0, 0); crate::thread::MAX_WORKER_CAPS]
    }

    #[test]
    fn service_workers_receive_heap_authority_before_declared_devices() {
        use kore_abi::cap::{ResourceKind, Rights};
        use kore_abi::task::role;

        let heap = kore_abi::session::SlotRef::from_wire(41);
        let mut declared = crate::service::WorkerEndowment::new();
        assert!(declared.push(worker_cap(role::DEVICE, 3)));
        let mut caps = vacant_worker_caps();

        let len =
            worker_caps_with_heap(heap, &declared, &mut caps).expect("one heap and one device fit");
        assert_eq!(len, 2);
        assert_eq!(caps[0].role, role::HEAP);
        assert_eq!(caps[0].index, 0);
        assert_eq!(caps[0].subject, heap);
        assert_eq!(caps[0].kind, ResourceKind::MemLease as u16 as u8);
        assert_eq!(
            caps[0].rights,
            Rights::ACCESS | Rights::MODIFY | Rights::DERIVE
        );
        assert_eq!(caps[1].role, role::DEVICE);
        assert_eq!(caps[1].index, 3);
    }

    #[test]
    fn runtime_heap_endowment_is_bounded_and_cannot_be_shadowed() {
        use kore_abi::task::role;

        let heap = kore_abi::session::SlotRef::from_wire(41);
        let mut caps = vacant_worker_caps();
        let mut duplicate = crate::service::WorkerEndowment::new();
        assert!(duplicate.push(worker_cap(role::HEAP, 0)));
        assert_eq!(worker_caps_with_heap(heap, &duplicate, &mut caps), None);

        let mut full = crate::service::WorkerEndowment::new();
        for index in 0..crate::thread::MAX_WORKER_CAPS {
            assert!(full.push(worker_cap(role::DEVICE, index as u16)));
        }
        assert_eq!(worker_caps_with_heap(heap, &full, &mut caps), None);
    }

    #[test]
    fn process_runtime_attachments_are_atomic_one_shots() {
        let claims = core::sync::atomic::AtomicUsize::new(0);
        assert!(claim_attachment_word(&claims, PROCESS_IO_ATTACHMENT));
        assert!(!claim_attachment_word(&claims, PROCESS_IO_ATTACHMENT));
        assert!(claim_attachment_word(&claims, PROCESS_STDIO_ATTACHMENT));
        assert!(!claim_attachment_word(&claims, PROCESS_STDIO_ATTACHMENT));
    }

    #[test]
    fn second_crt0_pass_preserves_the_live_slot_ledger() {
        assert_eq!(report_slot_install(0, 33), ReportSlotInstall::Initialize);
        assert_eq!(
            report_slot_install(34, 33),
            ReportSlotInstall::Preserve,
            "the final executable must not reopen linker-owned arena names"
        );
        assert_eq!(
            report_slot_install(34, 34),
            ReportSlotInstall::Conflict,
            "a different crt0 layout must fail closed"
        );
    }

    #[test]
    fn image_row_layout_keeps_runtime_slots_out_of_the_link_band() {
        let slots = runtime_slots_after(25).expect("image row runtime slots");
        assert_eq!(slots.scratch_base, 26);
        assert_eq!(
            slots.report,
            slots.scratch_base + SERVE_SCRATCH_SLOTS as u32
        );
        assert_eq!(
            slots.report + 1,
            slots.scratch_base + SERVE_SCRATCH_SLOTS as u32 + 1,
            "heap carve follows the report"
        );
        assert!(slots.report + 1 < STARTUP_LINK_SLOT_BASE);
    }

    #[test]
    fn runtime_layout_refuses_before_the_startup_link_band() {
        let top = STARTUP_LINK_SLOT_BASE - SERVE_SCRATCH_SLOTS as u32 - 3;
        let last = runtime_slots_after(top).expect("last complete runtime layout");
        assert_eq!(last.report + 1, STARTUP_LINK_SLOT_BASE - 1);
        assert_eq!(runtime_slots_after(top + 1), None);
    }

    #[test]
    fn owned_slot_words_cross_u64_without_opening_the_link_band() {
        assert!(STARTUP_LINK_SLOT_BASE > u64::BITS);
        let first = 61;
        let limit = STARTUP_LINK_SLOT_BASE as usize;

        assert_eq!(slot_interval_word(0, first, limit), 0xe000_0000_0000_0000);
        assert_eq!(slot_interval_word(1, first, limit), u64::MAX);
        assert_eq!(slot_interval_word(2, first, limit), u64::MAX);
        assert_eq!(slot_interval_word(3, first, limit), 0x0000_0000_ffff_ffff);
    }

    /// The extension-open interval never reaches the founding words: the
    /// startup-link band (`224..256`) and every runtime-owned founding
    /// slot sit strictly below the first extension bit, so an opener over
    /// `[256, …)` leaves words 0..3 untouched — the bitmap-side half of
    /// the "the linker band is a permanent island" claim (slice E).
    #[test]
    fn extension_intervals_never_touch_the_founding_words() {
        let first = ARENA_SLOT_CEILING as usize;
        let limit = first + 512; // two chunks
        for word in 0..(ARENA_SLOT_CEILING as usize / u64::BITS as usize) {
            assert_eq!(slot_interval_word(word, first, limit), 0);
        }
        assert_eq!(slot_interval_word(4, first, limit), u64::MAX);
        assert_eq!(slot_interval_word(12, first, limit), 0);
    }

    /// [`slot_alias_va`]'s two-window arithmetic (slice E), every edge:
    /// founding slots ride the founding window and stop at its span; a
    /// sub-founding arena's out-of-span slot is *nothing* (never an
    /// extension read); extension slots ride the extension base at dense
    /// `(slot − 256)·128` offsets, bounded by the LIVE aliased length;
    /// unstamped and un-bumped states refuse everything they govern.
    #[test]
    fn slot_alias_va_is_two_windows_and_fail_closed() {
        const REC: usize = 128;
        let founding = 0x7000_0000usize;
        let ext = 0x7600_0000usize;
        let full = ARENA_SLOT_CEILING as usize * REC;

        // Founding slots, full arena: dense from the founding base.
        assert_eq!(slot_alias_va(0, founding, full, ext, 0), Some(founding));
        assert_eq!(
            slot_alias_va(255, founding, full, ext, 0),
            Some(founding + 255 * REC)
        );
        // Slot 256 with nothing aliased yet: fail-closed.
        assert_eq!(slot_alias_va(256, founding, full, ext, 0), None);
        // One chunk aliased: 256..512 read densely at the ext base...
        let one_chunk = ARENA_SLOT_CEILING as usize * REC;
        assert_eq!(
            slot_alias_va(256, founding, full, ext, one_chunk),
            Some(ext)
        );
        assert_eq!(
            slot_alias_va(511, founding, full, ext, one_chunk),
            Some(ext + 255 * REC)
        );
        // ...and slot 512 waits for the next report (un-bumped hides it).
        assert_eq!(slot_alias_va(512, founding, full, ext, one_chunk), None);
        assert_eq!(
            slot_alias_va(512, founding, full, ext, 2 * one_chunk),
            Some(ext + one_chunk)
        );
        // A thread's 64-slot arena: slot 65 is past its founding span and
        // is NOT an extension slot — nothing, even with a (foreign) ext
        // state supplied.
        assert_eq!(slot_alias_va(65, founding, 64 * REC, ext, one_chunk), None);
        // Unstamped windows refuse what they govern.
        assert_eq!(slot_alias_va(0, 0, full, ext, one_chunk), None);
        assert_eq!(slot_alias_va(300, founding, full, 0, one_chunk), None);
        // The structural ceiling is the end of the address plan.
        assert_eq!(
            slot_alias_va(
                ARENA_EXT_SLOT_CEILING as usize,
                founding,
                full,
                ext,
                usize::MAX
            ),
            None
        );
        assert_eq!(
            slot_alias_va(
                ARENA_EXT_SLOT_CEILING as usize - 1,
                founding,
                full,
                ext,
                (ARENA_EXT_SLOT_CEILING - ARENA_SLOT_CEILING) as usize * REC,
            ),
            Some(ext + (ARENA_EXT_SLOT_CEILING - ARENA_SLOT_CEILING) as usize * REC - REC)
        );
    }
}

/// Run a bootstrapped system service — the **reader counterpart** to the
/// exec founder's stamper, and the service sibling of [`run_main`].  A
/// founded service's crt0 calls this (through the thin `__kore_rt_main`
/// its binary carries) with the validated [`Rt`]:
///
/// 1. **Attest and bind.** Confirm the founding stamped the
///    executor wake edge ([`role::DOORBELL`]) and every role the service's
///    endowment [`Profile`](kore_abi::task::Profile) declares, then resolve the
///    whole typed endowment.  A missing or unimplemented possession verb is a
///    refusal: the service does not advertise readiness for authority it never
///    bound.
/// 2. **Serve.** Straight to [`serve_forever`] — doorbell, ring window,
///    endpoints, beacon, [`ServeHost`](crate::svc::ServeHost) on the
///    executor — and **never comes back**.  Every manifest row serves
///    (the `_ON_METAL` opt-ins are deleted: a service that cannot serve
///    lawfully is completed or removed from the manifest, never gated).
///
/// Never returns: a founded service serves forever or parks ([`refuse`]). A
/// finite process uses [`crate::thread::exit`] with its SELF capability; the
/// service shape deliberately has no ordinary completion path.
pub fn run_service<S: crate::service::SystemService>(rt: Rt) -> ! {
    run_main(rt, serve::<S>)
}

/// The service ladder **as a root future** — what `#[kore_rt::main]`
/// drives for the canned service shape:
///
/// ```ignore
/// #[kore_rt::main]
/// async fn main(rt: kore_rt::entry::Rt) -> ! {
///     kore_rt::entry::serve::<MemoryManager>(rt).await
/// }
/// ```
///
/// [`run_service`] is the same thing with the attribute spelled out, kept
/// so a binary that wants no macro still has one line to write.  Both
/// reach the executor through [`run_main`], so the entry contract has one
/// implementation rather than two that drift — which is what happened
/// while services bypassed it: this ladder grew endowment resolution and
/// the readiness beacon, and `run_main` did not.
fn service_pair_count<S: crate::service::SystemService>(rt: &Rt) -> Option<usize> {
    let count = rt
        .bootstrap()
        .datum(role::SVC_RING, kore_abi::task::SERVICE_PAIR_COUNT_DATUM)
        .map_or(S::SERVE_PAIRS, |value| value as usize);
    (count >= S::SERVE_PAIRS && count != 0 && count <= crate::svc::MAX_SERVE_PAIRS).then_some(count)
}

pub async fn serve<S: crate::service::SystemService>(rt: Rt) -> ! {
    use crate::service::Bundle;
    if !service_ready::<S>(&rt) {
        refuse_with(&rt, RUNG_SERVICE_ATTEST);
    }
    let Ok(endowment) = <S::Endowment as Bundle>::resolve(rt.bootstrap()) else {
        refuse_with(&rt, RUNG_SERVICE_ENDOWMENT);
    };
    // A retained-return service must own its first donated target before its
    // startup admission runs. Image startup budgets every remaining dynamic
    // slot for archive leases; reserving this successor afterwards would let
    // an exactly-full archive report success and then fail serve setup.
    let mut return_targets = [const { None }; crate::svc::MAX_SERVE_PAIRS];
    if S::RETURN_MODE.is_retained() {
        for (pair, target) in return_targets
            .iter_mut()
            .enumerate()
            .take(service_pair_count::<S>(&rt).unwrap_or(0))
        {
            let recv = rt.bootstrap().handle(role::SVC_RETURN, (2 * pair) as u16);
            let land = rt
                .bootstrap()
                .handle(role::SVC_RETURN, (2 * pair + 1) as u16);
            match (recv, land) {
                (Some(_), Some(_)) => {
                    *target = match crate::cap::ReturnedCapability::reserve() {
                        Some(target) => Some(target),
                        None => refuse_with(&rt, RUNG_SERVICE_SERVE_SETUP),
                    };
                }
                // A static face may deliberately be post-less. Requests that
                // require returned authority then reach the service with none
                // and refuse through its ordinary typed vocabulary.
                (None, None) => {}
                // Half a return leg can neither stand nor be safely ignored.
                _ => refuse_with(&rt, RUNG_SERVICE_SERVE_SETUP),
            }
        }
    }
    // A service is one affine owner for its whole lifetime. Put it in the
    // founded heap at construction: large bounded service arenas must not be
    // copied through each async host frame on the fixed user stack.
    let mut service = crate::svc::own_service(S::bind(endowment));
    if let Err(failure) = service.start_on_metal(&rt) {
        refuse_with(&rt, failure.payload(RUNG_SERVICE_START));
    }
    // The declared workers, before readiness (see `spawn_workers`).
    spawn_workers::<S>(&rt, &service);
    serve_forever_with_return_targets::<S>(&rt, service, return_targets, 0).await;
    // A serving row whose ring/endpoints refuse must not downgrade its
    // promise to mere readiness.
    refuse_with(&rt, RUNG_SERVICE_SERVE_SETUP)
}

/// **The serve loop, on metal** — [`run_service`]'s step 2.  Assembles the §5.5 serve machine out of the
/// founding's own wiring and hands it to the executor, which never gives control
/// back: between requests the task is parked on its doorbell (`commit_and_park`
/// → `WAIT_DELIVERY`), which is exactly what "the service is up" means.
///
/// The order is load-bearing:
///
/// 1. **Doorbell** ([`Rt::doorbell`] → `recv_pin` off the R/O `SLOTS_VA` alias)
///    — without a wake edge a task cannot idle lawfully (§3.1), so this is the
///    first refusal.
/// 2. **The ring window** ([`serve_window`]: `role::SVC_RING` → `take_view` →
///    `map_view`) — the service earns its *own* mapping of the pair window from
///    the delivered view.  The delivered view is the single owner of the
///    service's ring mapping; nothing here consults a founder-installed VA.
/// 3. **The ring halves** ([`serve_rings`](crate::svc::serve_rings) at the
///    [`REQ_CAP`](crate::svc::REQ_CAP) contract) — fail-closed: a window the
///    founder did not format refuses here rather than serving garbage.
/// 4. **The endpoints** — the client kick (`role::SVC_SEND`), the grant sends
///    (`role::SVC_GRANT`, one per [`SvcProtocol::REPLY_GRANTS`]), and the
///    [`SERVE_SCRATCH_SLOTS`] derive scratch ([`serve_scratch_base`]).
///    A multi-client row ([`SvcProtocol::SERVE_PAIRS`] > 1) repeats 2–4 per
///    pair at that pair's role indices and attaches each to the same host.
/// 5. **The ready beacon** — one [`COOKIE_SERVING`](crate::svc::COOKIE_SERVING)
///    notice on `role::PARENT_SEND`, sent from *inside* the executor as the
///    root future's first act, so it stands after every piece above resolved:
///    the beacon means "serving", never merely "attested".  It rides the
///    executor (not a bare pre-`run` commit) because the executor establishes
///    the standing doorbell post first — so the instant the founder observes
///    readiness, a peer's kick is already droppable-safe (§3.1's premise).
/// 6. **[`ServeHost::run`](crate::svc::ServeHost::run)** — forever.
///
/// Returns only on a refusal in 1–4 (the caller falls back to readiness);
/// `Executor::run` over a `-> !` root does not return on metal.
///
/// Public because a **driver row's bespoke entry** ends here too: a row
/// that claims its controller from the device broker before serving
/// (`res/DEVICE_SERVICE.md` §10) runs its claim on the executor and then
/// becomes exactly this ladder — one implementation of the serve machine,
/// however a row's endowment arrived.
///
/// `note` rides the `COOKIE_SERVING` beacon's payload — the row's
/// **non-fatal note**, printed by the founder as `note: rung N` beside the
/// ready line (the input relay's precedent): a service that came up
/// degraded (a driver row whose machine has no matching controller) says
/// so on the one line a serial-less boot leaves behind, without failing a
/// row that can lawfully serve.  `0` = nothing to report.
pub async fn serve_forever<S: crate::service::SystemService>(
    rt: &Rt,
    svc: S,
    return_target: Option<crate::cap::ReturnedCapability>,
    note: u64,
) {
    let mut return_targets = [const { None }; crate::svc::MAX_SERVE_PAIRS];
    return_targets[0] = return_target;
    serve_forever_with_return_targets(rt, crate::svc::own_service(svc), return_targets, note).await
}

/// Reserve pair `pair`'s affine landing owner for a bespoke service entry.
///
/// Driver rows must claim hardware before entering [`serve_forever`], so they
/// cannot use the canned [`serve`] ladder that ordinarily performs this
/// reservation. This helper preserves the same guardrails: only a retained
/// protocol, a declared pair, and a completely stamped return leg can consume
/// a runtime-owned landing.
pub fn reserve_retained_return_target<S: crate::service::SystemService>(
    rt: &Rt,
    pair: usize,
) -> Option<crate::cap::ReturnedCapability> {
    if !S::RETURN_MODE.is_retained() || pair >= service_pair_count::<S>(rt)? {
        return None;
    }
    rt.bootstrap().handle(role::SVC_RETURN, (2 * pair) as u16)?;
    rt.bootstrap()
        .handle(role::SVC_RETURN, (2 * pair + 1) as u16)?;
    crate::cap::ReturnedCapability::reserve()
}

/// Internal form of [`serve_forever`] carrying the complete bank
/// of retained-return targets reserved before service startup. A multi-pair
/// retained service must reserve every target before its startup allocator can
/// consume the remaining arena slots; delaying pairs 1..N made their ability
/// to serve depend on archive size.
async fn serve_forever_with_return_targets<S: crate::service::SystemService>(
    rt: &Rt,
    mut svc: crate::svc::ServiceOwner<S>,
    mut return_targets: [Option<crate::cap::ReturnedCapability>; crate::svc::MAX_SERVE_PAIRS],
    note: u64,
) {
    use kore_abi::session::{Notice, SlotRef};

    use crate::chan::ChanTx;
    use crate::service::MAX_GRANTS;
    use crate::svc::{
        COOKIE_SERVING, MAX_SERVE_PAIRS, ServeEndpoints, ServeHost, serve_rings, window_bytes,
    };

    let boot = rt.bootstrap();
    let Some(pair_count) = service_pair_count::<S>(rt) else {
        return;
    };
    // 1 · the wake edge.
    let Some(doorbell) = rt.doorbell() else {
        return;
    };
    // 2 · this side's own mapping of the pair window.
    let Some(window) = serve_window(rt) else {
        return;
    };
    // SAFETY: `map_view` installed its reported `[base, +len)` R/W-USER in **this
    // task's own** space over the pair's leased backing, page-aligned (hence
    // 64-byte aligned, the spsc layout unit) and live for the task's whole life
    // (nothing unmaps it; the service never returns).  This task is the sole
    // service-side attacher of that window — the client half lives in another
    // space and touches it only through the SPSC protocol — so the `&mut` is
    // unaliased in the sense the ring halves require.
    let window: &'static mut [u8] =
        unsafe { core::slice::from_raw_parts_mut(window.cast::<u8>().as_ptr(), window.len()) };
    let control_bytes = window_bytes::<S>(S::REQUEST_CAPACITY, S::RESPONSE_CAPACITY);
    if window.len() < control_bytes {
        return;
    }
    let (control, aux) = window.split_at_mut(control_bytes);
    // 3 · the halves, at the founder-shared geometry.
    let Ok(rings) = serve_rings::<S>(control, S::REQUEST_CAPACITY, S::RESPONSE_CAPACITY) else {
        return;
    };
    if !svc.attach_pair_aux(0, aux) {
        return;
    };
    // 4 · the pair's endpoints.
    let Some(kick) = boot.handle(role::SVC_SEND, 0) else {
        return;
    };
    let grants = S::REPLY_GRANTS.min(MAX_GRANTS);
    let mut grant_tx = [None; MAX_GRANTS];
    for (k, tx) in grant_tx.iter_mut().enumerate().take(grants) {
        let Some(slot) = boot.handle(role::SVC_GRANT, k as u16) else {
            return;
        };
        *tx = Some(slot);
    }
    let Some(scratch_base) = serve_scratch_base(rt) else {
        return;
    };
    let mut scratch = [None; MAX_GRANTS];
    for (k, sc) in scratch.iter_mut().enumerate().take(SERVE_SCRATCH_SLOTS) {
        *sc = Some(SlotRef::from_wire(scratch_base + k as u32));
    }
    let ep = ServeEndpoints {
        kick,
        grant_tx,
        scratch,
        // The return channel (`role::SVC_RETURN`): recv end + landing
        // slot — the §5.5 reverse leg's standing post.  Optional: a
        // protocol with no return leg leaves the roles unstamped.
        return_post: boot
            .handle(role::SVC_RETURN, 0)
            .zip(boot.handle(role::SVC_RETURN, 1)),
    };
    // 5 · the ready beacon's send end (absent ⇒ serve anyway; the founder's
    //     own drain is what reports the missing edge, never a wedged service).
    let beacon = boot.handle(role::PARENT_SEND, 0);

    // The doorbell is resolved above only to prove the wake edge was
    // stamped; the executor that consumes it is the caller's now.
    let _ = doorbell;
    let mut host = ServeHost::<_, S>::new_owned_with_return_target(
        svc,
        &TRANSPORT,
        rings,
        ep,
        return_targets[0].take(),
    );
    #[cfg(feature = "heap")]
    if let Some(arena) = host.service().minted_pair_arena() {
        if !host.enable_minted_growth(arena) {
            return;
        }
    }
    // The extra pairs of a multi-client row — same ladder, that pair's
    // role indices: ring `SVC_RING@k`, kick `SVC_SEND@k`, grant channels
    // `SVC_GRANT@(k·grants ..)`, return legs `SVC_RETURN@2k/2k+1`.  The
    // derive scratch is shared: the serve loop is sequential, and a
    // scratch slot lives only inside one request's grant session.
    for k in 1..pair_count {
        let Some(window) = serve_window_at(rt, k as u16) else {
            return;
        };
        // SAFETY: the same contract as pair 0's window above — this task's
        // own `map_view` of pair `k`'s leased backing, whose sole other
        // toucher is that pair's client, through the SPSC protocol.
        let window: &'static mut [u8] =
            unsafe { core::slice::from_raw_parts_mut(window.cast::<u8>().as_ptr(), window.len()) };
        if window.len() < control_bytes {
            return;
        }
        let (control, aux) = window.split_at_mut(control_bytes);
        let Ok(rings) = serve_rings::<S>(control, S::REQUEST_CAPACITY, S::RESPONSE_CAPACITY) else {
            return;
        };
        if !host.service_mut().attach_pair_aux(k, aux) {
            return;
        };
        let Some(kick) = boot.handle(role::SVC_SEND, k as u16) else {
            return;
        };
        // An extra pair's grant bank is named by its **base handle** and
        // derived from there: the founder stamps that run with one
        // arithmetic expression (`base + g`), so stamping every member
        // spent `REPLY_GRANTS - 1` boot handles restating it — which is
        // what overflowed a multi-pair row's `MAX_BOOT_HANDLES` budget
        // (image: three banks of nine).
        //
        // Pair 0 is deliberately NOT derived this way (see its own loop
        // above): its slots come from a hand-laid founder map whose grant
        // tail lives in a separate band, so its run is discontinuous and
        // only the explicit handles can name it.
        let mut grant_tx = [None; MAX_GRANTS];
        if grants > 0 {
            let Some(base) = boot.handle(role::SVC_GRANT, (k * grants) as u16) else {
                return;
            };
            for (g, tx) in grant_tx.iter_mut().enumerate().take(grants) {
                *tx = Some(kore_abi::session::SlotRef::from_wire(
                    base.index() + g as u32,
                ));
            }
        }
        let ep = ServeEndpoints {
            kick,
            grant_tx,
            scratch,
            return_post: boot
                .handle(role::SVC_RETURN, (2 * k) as u16)
                .zip(boot.handle(role::SVC_RETURN, (2 * k + 1) as u16)),
        };
        // Service-selected landing rotations live on pair 0 alone: an extra
        // pair of a Walking protocol is lawful only POST-LESS (a control-plane
        // pair that can never receive transfers). Retained protocols instead
        // receive one independently reserved target per founded pair.
        if ep.return_post.is_some() && matches!(S::RETURN_MODE, crate::service::ReturnMode::Walking)
        {
            return;
        }
        if !host.attach_pair_with_return_target(k, rings, ep, return_targets[k].take()) {
            return;
        }
    }
    // 6 · readiness, then forever. Stand the return posts first, advance any
    //     service-owned asynchronous proof on this same executor/doorbell,
    //     beacon once, then become the §5.5 serve loop. The posts BEFORE both
    //     the proof and beacon preserve the board law: a client that sees
    //     `COOKIE_SERVING` can never win a return send against an un-stood
    //     post (no visible post ⇒ `WouldBlock`).
    if !host.stand_return_post().await {
        return;
    }
    loop {
        match host.service_mut().advance_readiness() {
            Ok(true) => break,
            Ok(false) => {
                let _ = crate::executor::next_wake().await;
            }
            Err(failure) => refuse_with(rt, failure.payload(RUNG_SERVICE_START)),
        }
    }
    if let Some(send) = beacon {
        let _ = ChanTx::new(&TRANSPORT, send)
            .send(Notice {
                cookie: COOKIE_SERVING,
                // The static entry note joined with whatever the service
                // learned during its own readiness proof (`ready_note`);
                // both use the rung/detail encoding, and 0 stays silent.
                payload: note | host.service().ready_note(),
            })
            .await;
    }
    let never = S::drive_host(host).await;
    match never {}
}

/// The [`SERVE_SCRATCH_SLOTS`] base for this task — one past the highest arena
/// slot the founding's stamped boot block names (see [`SERVE_SCRATCH_SLOTS`] for
/// why the block is the exhaustive list).  `None` when the whole scratch run
/// would not fit below [`STARTUP_LINK_SLOT_BASE`] together with the report and
/// heap-carve slots — fail-closed: a service with no room to stage a grant
/// refuses rather than deriving into durable linker authority.
/// The serve host's grant-derive scratch base — the first of
/// [`SERVE_SCRATCH_SLOTS`] arena slots past the founding's stamped
/// handles.  Public for the bespoke driver-row loops (swap-nvme's
/// shape) whose protocols carry reply grants: they open-code the serve
/// ladder, and the scratch plan must be the one the generic entry
/// would have used.
pub fn serve_scratch_base(rt: &Rt) -> Option<u32> {
    let top = rt.bootstrap().handles().map(|(_, _, s)| s.index()).max()?;
    runtime_slots_after(top).map(|slots| slots.scratch_base)
}

/// Resolve the bounded derive-scratch bank for a bespoke service entry that
/// assembles [`ServeHost`](crate::svc::ServeHost) directly.
///
/// Most services use [`serve_forever`], which performs this step internally.
/// Authority-bearing services with a custom walking return post use this
/// helper so they share the same stamped-geometry bounds rather than
/// duplicating startup-slot arithmetic.
pub fn serve_scratch(rt: &Rt) -> Option<[Option<SlotRef>; crate::service::MAX_GRANTS]> {
    let base = serve_scratch_base(rt)?;
    let mut scratch = [None; crate::service::MAX_GRANTS];
    for (index, slot) in scratch.iter_mut().enumerate().take(SERVE_SCRATCH_SLOTS) {
        *slot = Some(SlotRef::from_wire(base + index as u32));
    }
    Some(scratch)
}

/// The pure readiness attestation — [`run_service`] step 1.  All slot
/// reads, no verbs: this is exactly the founding's stamp side read back.
fn service_ready<S: crate::service::SystemService>(rt: &Rt) -> bool {
    use crate::service::Bundle;
    let boot = rt.bootstrap();
    // The executor's wake edge — stamped for every service (the founding's
    // `stage_pair` doorbell recv/landing pair).
    if boot.handle(role::DOORBELL, 0).is_none() {
        return false;
    }
    // Every role the endowment profile declares (mm: `[MEM_SUPPLY]`) —
    // except the optional ones, whose declared-possible absence is the
    // vacancy the founder stamps no handle for (`ROOT_COMPLEX@2` on a
    // two-root machine, `DISPLAY@1` on a one-panel one): the bundle's
    // `Option<T>` field resolves it, and turning it into a generic
    // readiness NACK here would make "optional" unfoundable — the exact
    // posture the WIRING loop below already documents.
    for r in <S::Endowment as Bundle>::PROFILE.roles {
        if !r.optional && boot.handle(r.role, r.index).is_none() {
            return false;
        }
    }
    // Required topology captures declared beside the bundle. Optional captures
    // remain observable to `resolve` without turning absence into a generic
    // readiness failure.
    for r in <S::Endowment as Bundle>::WIRING {
        if r.required && boot.handle(r.role, r.index).is_none() {
            return false;
        }
    }
    true
}

/// Map this service's request/reply ring window — [`serve_forever`]'s step 2.
/// The founding **delivers** the pair window as a `LeasedRegion` view into the
/// service's arena at [`role::SVC_RING`]; this earns the mapping from it:
/// `take_view` reads the delivered record off the R/O `SLOTS_VA` alias and
/// `map_view` installs its backing into **this task's own** space at
/// the VA its `GeomReport` names (#336 rung 2).  That view is the single owner of the
/// service's ring mapping — the service never names a founder-chosen child VA.
/// `None` off the metal target (the verbs refuse honestly) or on any refusal.
fn serve_window(rt: &Rt) -> Option<core::ptr::NonNull<[u8]>> {
    serve_window_at(rt, 0)
}

/// Pair `index`'s ring window — the multi-pair generalization
/// ([`SvcProtocol::SERVE_PAIRS`](crate::svc::SvcProtocol::SERVE_PAIRS)):
/// each serving pair's window arrives as its own `SVC_RING@index` view.
fn serve_window_at(rt: &Rt, index: u16) -> Option<core::ptr::NonNull<[u8]>> {
    let view = rt.bootstrap().handle(role::SVC_RING, index)?;
    let leased = verbs::take_view(view)?;
    verbs::map_view(leased)
}

// ─── crt0 — the target half (`target_os = "none"`: the kore-user
//     targets; the host never compiles these) ───────────────────────

/// The Rust half of crt0: adopt the boot page the stub forwarded and
/// enter the service's declared main.
///
/// # Safety
/// `ptr` must be the child VA of this task's live, parent-written boot
/// page (`found_user`'s `arg`, forwarded by the stub — the spawn
/// contract); called exactly once, on the founding dispatch.
#[cfg(all(target_os = "none", feature = "crt0"))]
unsafe extern "C" fn crt0_start(ptr: *const u8) -> ! {
    // SAFETY: the spawn contract above — one page, live for the task's
    // whole life (parent-delegated), never remapped.
    let page: &'static [u8] = unsafe { core::slice::from_raw_parts(ptr, BOOT_PAGE_BYTES) };
    // Idempotent across both lawful entry paths: the startup linker already
    // anchored before issuing sessions, while a direct-static image reaches
    // crt0 as the owner of its first anchor.  `anchor_boot_thread` observes
    // only a post-SET_TP commit marker, so it preserves the linker's durable
    // slot state without trusting a merely initialized block.
    if !anchor_boot_thread(page) {
        refuse();
    }
    match boot_context(page) {
        // `__kore_rt_main` is declared `safe` below: a conforming binary
        // carries exactly one `#[kore_rt::main]` expansion, and linking
        // is what enforces it (a missing/duplicate symbol fails the
        // link, not the call site).
        Ok(rt) => __kore_rt_main(rt),
        Err(_) => refuse(),
    }
}

#[cfg(all(target_os = "none", feature = "crt0"))]
unsafe extern "Rust" {
    /// The `#[kore_rt::main]`-emitted entry (`kore-rt-macros`), or a service
    /// image's thin hand-written `__kore_rt_main` → [`run_service`].
    safe fn __kore_rt_main(rt: Rt) -> !;
}

/// The x86_64 ELF entry: the stub forwarded the boot-page pointer in
/// `rdi` — already the SysV first-argument register — so crt0 only
/// normalizes the frame: 16-align the stack and `call` (the push makes
/// the callee see the ABI's `rsp % 16 == 8` entry state).
#[cfg(all(target_os = "none", target_arch = "x86_64", feature = "crt0"))]
#[unsafe(naked)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _start() -> ! {
    core::arch::naked_asm!(
        "xor rbp, rbp",   // terminate the frame chain
        "and rsp, -16",   // spawn hands a 16-aligned stack; enforce it
        "call {start}",   // rdi rides through untouched
        "ud2",            // start diverges; trap if it ever returns
        start = sym crt0_start,
    )
}

/// The aarch64 ELF entry: pointer in `x0` (the AAPCS64 first argument);
/// terminate the frame-record chain and branch.
#[cfg(all(target_os = "none", target_arch = "aarch64", feature = "crt0"))]
#[unsafe(naked)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _start() -> ! {
    core::arch::naked_asm!(
        "mov x29, xzr",   // no caller frame
        "mov x30, xzr",   // no return address
        "b {start}",      // x0 rides through untouched
        start = sym crt0_start,
    )
}

/// The riscv64 ELF entry: pointer in `a0` (the first argument register);
/// terminate the frame chain and jump.
#[cfg(all(target_os = "none", target_arch = "riscv64", feature = "crt0"))]
#[unsafe(naked)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _start() -> ! {
    core::arch::naked_asm!(
        "mv fp, zero",    // no caller frame
        "mv ra, zero",    // no return address
        "tail {start}",   // a0 rides through untouched
        start = sym crt0_start,
    )
}

/// Publish an immutable process-lifetime TLS template during single-threaded startup.
/// # Safety
/// Bytes are mapped read-only, initialized, and remain live for the process.
/// Call exactly once before exposing any worker; no concurrent publisher exists.
#[cfg(target_os = "none")]
pub unsafe fn publish_tls_template(template: crate::tls::Template) -> bool {
    unsafe { geometry::publish_tls_template(template) }
}
#[cfg(not(target_os = "none"))]
pub unsafe fn publish_tls_template(_: crate::tls::Template) -> bool {
    false
}

#[cfg(target_os = "none")]
pub(crate) fn install_child_tls(page: &'static [u8]) -> bool {
    let Ok((_, Some(geom), _, _, _)) = boot_parse(page) else {
        return false;
    };
    let Some((base, len)) = geom.window_indexed(kore_abi::task::geom::THREAD_BLOCK, 2) else {
        return geometry::tls_template().is_none();
    };
    let Some(template) = geometry::tls_template() else {
        return false;
    };
    if base % template.align != 0 || len < template.bytes {
        return false;
    }
    let Some(tp) = base.checked_add(template.tp_offset) else {
        return false;
    };
    let Some(space) = geometry::self_space_slot() else {
        return false;
    };
    let Some(slot) = geometry::acquire_cap_slot() else {
        return false;
    };
    if !migrate_thread_block(tp) {
        geometry::release_cap_slot(slot);
        return false;
    }
    let installed = crate::cap::verbs::set_thread_pointer(space, tp, slot);
    // SET_TP is an effect: its result remains vacant on success or refusal.
    // On success the copied block is now active and owns this reservation.
    geometry::release_cap_slot(slot);
    installed
}
