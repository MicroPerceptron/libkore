//! The startup link engine — the in-process linker stub's brain
//! (`res/IMAGE.md` §4 phase 2, §8; epic #291, P5 #296), generic over a
//! **world seam** ([`LinkWorld`]).
//!
//! ## Shape
//!
//! One task, once, at startup: acquire the `DT_NEEDED` closure breadth-
//! first from the executable, place every module at world-chosen bases,
//! map shared segments (text R|X, rodata R), copy data templates into
//! private RW memory, eagerly relocate through [`crate::apply`], seal
//! each RELRO window, build the static TLS block and set the thread
//! pointer, surface the `DT_INIT_ARRAY`/`DT_FINI_ARRAY` pointers in
//! dependency order, and leave the read-only passive-ABI
//! [`ModulePage`] behind.  The yield is a
//! [`LinkedProgram`]; the *caller* (P7's crt0) runs the init pointers
//! and jumps to `entry` — the engine never executes child code.
//!
//! ## The world seam
//!
//! Every side effect goes through [`LinkWorld`] — the capability-shaped
//! replacement for a classical linker's ambient-authority mapper.  Ops
//! are **capability-moving**: a per-segment lease handle is affine
//! (mapping consumes it), placement is the world's act ([`choose_base`]
//! — per-task ASLR lives there, the engine only states size/alignment),
//! and the engine's write authority is exactly the RW windows the world
//! handed it.  On real kore the world is the child-side session-verb
//! adapter (`Acquire` over `role::LOADER`, share/allocate/unmap/seal
//! session verbs); on the host it is a recording mock — the engine
//! cannot tell, which is the point.
//!
//! Where the executable comes from is also the world's business: the
//! engine calls [`acquire`] for it like any module, and the real world
//! answers *that one* name from the spawn endowment (`.kore.ld`'s
//! descriptor + the `role::IMAGE` lease caps) instead of the loader
//! channel — the degenerate static case (`res/IMAGE.md` §4) falls out
//! of the same seam.
//!
//! ## Regime (enforced, not extended)
//!
//! - Bounded everything: ≤ [`MAX_LINK_MODULES`] modules (refused, never
//!   truncated), ≤ [`MAX_INIT_FNS`] init/fini pointers, fixed tables,
//!   no allocation.
//! - Dedup by [`ImageName`]: a diamond's shared base loads **once**.
//! - A `DT_NEEDED` **cycle refuses** ([`LinkError::DependencyCycle`]):
//!   under eager binding with init ordering there is no lawful order
//!   for a cycle to run in, so it is a foreign layout, not a case.
//! - Resolution scope: executable first, then libraries in BFS load
//!   order (module 0 first is **load-bearing**: the exe's local-exec
//!   TLS offsets assume its block is the variant's first).
//! - Init order: dependency-before-dependent (libC before libA/libB
//!   before the exe), ties broken by load order; `fini` is the exact
//!   reverse.
//! - Map-RW → relocate → seal, never protect-before-relocate; the seal
//!   range is the wire descriptor's `relro_off/relro_sz` rebased by
//!   [`crate::apply`].
//!
//! [`acquire`]: LinkWorld::acquire
//! [`choose_base`]: LinkWorld::choose_base

use core::alloc::Allocator;

use elf::abi::{DT_NEEDED, DT_NULL};
use heapless::Vec as HVec;
use kore_abi::image::{
    ImageName, MAX_MODULE_RECORDS, MAX_SEGMENTS, ModuleDescriptor, ModulePage, ModuleRecord,
    ModuleSeg, SegmentDesc, kind, seg,
};
use zerocopy::IntoBytes;

use crate::apply::{
    ApplyError, MAX_MODULES, ModuleImage, ModuleView, SymbolScope, TlsLayout, apply_module,
};
use crate::parse::PAGE_SIZE;
use crate::reloc::{RelocArch, RelocKind, TlsRules};
use crate::store::Slots;

/// Load-set **structural ceiling** — the BootBlock-budget scale (16
/// bootstrap entries, `res/IMAGE.md` §4), the width of the dependency
/// bitmasks, and the passive page's record capacity.  The *working*
/// capacity is the caller's, per link ([`LinkCaps::max_modules`], `1..=`
/// this) — the tables are arena-carved to the caps, never to the
/// ceiling (Phase A, `res/USERLAND.md` §7).
pub const MAX_LINK_MODULES: usize = 16;

const _: () = {
    assert!(MAX_LINK_MODULES <= MAX_MODULE_RECORDS);
    assert!(MAX_LINK_MODULES <= MAX_MODULES);
    // The dependency masks are u16 — the bound may not outgrow them.
    assert!(MAX_LINK_MODULES <= 16);
};

/// Init/fini pointer **structural ceiling** across the whole load set
/// (the [`LinkedProgram`] return arrays' inline capacity); the working
/// bound is [`LinkCaps::max_inits`] (`1..=` this) — overflow refuses
/// ([`LinkError::TooManyInits`]).
pub const MAX_INIT_FNS: usize = 32;

/// The caller's **capacity decision** for one link (Phase A,
/// `res/USERLAND.md` §7): how many modules and init/fini pointers this
/// program may carry.  The engine's tables are allocated from the
/// caller's arena at exactly these sizes — a two-module static program
/// pays for two modules' tables, not the ceiling's sixteen.  Both dials
/// are validated against the structural ceilings
/// ([`MAX_LINK_MODULES`]/[`MAX_INIT_FNS`]); out-of-range caps refuse
/// [`LinkError::Scratch`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinkCaps {
    /// Modules the load set may reach (`1..=MAX_LINK_MODULES`).
    pub max_modules: usize,
    /// Init/fini pointers across the set (`1..=MAX_INIT_FNS`).
    pub max_inits: usize,
}

impl LinkCaps {
    /// The ceilings themselves — the pre-parameterization behavior.
    pub const DEFAULT: Self = Self {
        max_modules: MAX_LINK_MODULES,
        max_inits: MAX_INIT_FNS,
    };

    fn valid(&self) -> bool {
        (1..=MAX_LINK_MODULES).contains(&self.max_modules)
            && (1..=MAX_INIT_FNS).contains(&self.max_inits)
    }
}

/// One `.dynamic` entry (ELF64) — the `DT_NEEDED` walk's stride.
const DYN_ENTRY: u64 = 16;

/// What one [`LinkWorld::acquire`] yields: the frozen descriptor plus at
/// most one lease per descriptor segment, parallel to `descriptor.segs`.
pub type Acquired<L> = (ModuleDescriptor, [Option<L>; MAX_SEGMENTS]);

/// Everything the engine may do to the world — the whole side-effect
/// surface of startup linking, capability-shaped (module docs).  `Lease`
/// handles are **affine**: [`map_image`](Self::map_image) and
/// [`read_segment`](Self::read_segment) consume them, exactly as the
/// real share-derive consumes its staged capability.
pub trait LinkWorld {
    /// World-side failure detail (an acquire refusal, a map failure…).
    /// The engine wraps it in [`LinkError::World`] — typed fail-closed
    /// end to end.
    type Error;
    /// One granted segment-lease capability (kore-rt: an executed
    /// `LeasedToken`).
    type Lease;
    /// A mapped read view — durable for [`map_image`](Self::map_image)
    /// (it *is* the segment's presence in the address space), transient
    /// for [`read_segment`](Self::read_segment) (dropping it is the
    /// unmap).
    type Ro: AsRef<[u8]>;
    /// A private zeroed RW window of **exactly** the requested length —
    /// the engine's only write authority.  Dropping the handle releases
    /// nothing: the memory stays the task's (data regions, TLS, the
    /// passive page all outlive the stub).
    type Rw: AsRef<[u8]> + AsMut<[u8]>;

    /// One `Acquire` round-trip: name → the frozen descriptor + at most
    /// one lease per descriptor segment, **parallel to
    /// `descriptor.segs`** (the service's stash order).  A refusal
    /// grants nothing.
    fn acquire(&mut self, name: &ImageName) -> Result<Acquired<Self::Lease>, Self::Error>;

    /// Choose a fresh span: `len` bytes at an address satisfying `align`
    /// (a power of two).  Placement policy — ASLR, guard gaps, layout —
    /// is entirely the world's; the engine re-checks the alignment and
    /// otherwise trusts nothing else about the value.
    fn choose_base(&mut self, len: u64, align: u64) -> Result<u64, Self::Error>;

    /// Durably share a segment lease at exactly `va` with `access`
    /// ([`seg`] bits).  The engine only ever asks `R|X` (text) or `R`
    /// (rodata) — never `W`; a world may (and the mock does) refuse a
    /// writable ask outright, the wire's W^X backstop.
    fn map_image(
        &mut self,
        lease: Self::Lease,
        va: u64,
        access: u8,
    ) -> Result<Self::Ro, Self::Error>;

    /// Transiently map a lease read-only wherever the world likes — the
    /// copy source for data templates and TLS init images.
    fn read_segment(&mut self, lease: Self::Lease) -> Result<Self::Ro, Self::Error>;

    /// Retire a transient view after its bytes have been copied.
    ///
    /// The default preserves the simple host-world contract: dropping the
    /// handle performs (or models) the unmap. A session-backed world may queue
    /// that affine retirement until [`finalize_link`](Self::finalize_link),
    /// amortizing several independent revokes into one crossing. No placement
    /// may reuse the VA before finalization.
    fn retire_read(&mut self, view: Self::Ro) -> Result<(), Self::Error> {
        drop(view);
        Ok(())
    }

    /// Fresh private **zeroed** RW memory at exactly `va`, `len` bytes
    /// (the kernel's zeroed-delegation discipline: no stale bytes reach
    /// a task).  `.bss` and `tbss` are the zeroing, not a memset here.
    fn alloc_private(&mut self, va: u64, len: u64) -> Result<Self::Rw, Self::Error>;

    /// Retain immutable initialization bytes for future native threads before
    /// any initial-thread TCB state is written. Worlds without thread creation
    /// may discard this; their runtime must continue refusing TLS child spawn.
    fn retain_tls_template(
        &mut self,
        _bytes: &[u8],
        _tp_offset: u64,
        _align: u64,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Downgrade `[va, va + len)` to read-only — the RELRO seal and the
    /// passive-page seal.  Called only after every write into the range
    /// has happened (the engine's ordering obligation).
    fn seal_ro(&mut self, va: u64, len: u64) -> Result<(), Self::Error>;

    /// Install the initial thread pointer (the arch's TP register, via
    /// whatever gate the world has — `res/IMAGE.md` §9g).
    fn set_tp(&mut self, tp: u64) -> Result<(), Self::Error>;

    /// Commit every final protection change and, when present, install the
    /// initial thread pointer as the last effect.
    ///
    /// The conservative default retains the original one-operation calls.
    /// Metal worlds can issue the entire bounded list as one session after all
    /// relocation/copy writes are complete. `set_tp` is deliberately last:
    /// subsequent runtime verbs may no longer be able to address their old
    /// per-thread transport block.
    fn finalize_link(&mut self, seals: &[(u64, u64)], tp: Option<u64>) -> Result<(), Self::Error> {
        for &(va, len) in seals {
            self.seal_ro(va, len)?;
        }
        if let Some(tp) = tp {
            self.set_tp(tp)?;
        }
        Ok(())
    }

    /// Where the passive-ABI page (`res/IMAGE.md` §8) goes.  The default
    /// is one more placement; a world whose task will read the page back
    /// later — the startup stub, for the late loader — returns the one
    /// **fixed** address the ABI names for it (`geom::LINK`'s last page),
    /// because nothing else records where the page went.
    fn module_page_base(&mut self, len: u64) -> Result<u64, Self::Error> {
        self.choose_base(len, len)
    }

    /// Provenance id of a lease for the passive ABI
    /// ([`ModuleSeg::lease_id`]) — correlation, never authority; worlds
    /// without one report 0.
    fn lease_id(&self, lease: &Self::Lease) -> u32 {
        let _ = lease;
        0
    }
}

/// Why the link refused.  Total failure: any error leaves no
/// half-linked program (mappings already made die with the task that
/// cannot start — the spawn session's failure path).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkError<E> {
    /// A world op failed (acquire refusal, map/alloc/seal failure) —
    /// the world's own detail rides along.
    World(E),
    /// The `DT_NEEDED` closure exceeds [`MAX_LINK_MODULES`].
    TooManyModules,
    /// The dependency graph has a cycle — no lawful init order exists.
    DependencyCycle,
    /// A module's descriptor lies: fails `validate()`, wrong machine,
    /// wrong kind (library where an executable belongs or vice versa),
    /// segment access outside the §2 table, or geometry that cannot be
    /// placed.  Carries the module's load index.
    BadModule(usize),
    /// A live descriptor segment arrived without its lease (the grant
    /// budget promises one per segment).
    MissingLease(usize),
    /// The mapped `.dynamic`/strtab contents contradict the descriptor
    /// (ranges escape the views, `DT_NEEDED` count mismatch).
    BadDynamic(usize),
    /// A `DT_NEEDED` name does not fit [`ImageName`] (empty, > 64
    /// bytes, or embedded NUL).
    NeededName(usize),
    /// The relocation engine refused this module.
    Apply(ApplyError, usize),
    /// The static TLS layout or the block copy refused (lying geometry).
    Tls(ApplyError),
    /// More init/fini pointers than [`MAX_INIT_FNS`].
    TooManyInits,
    /// Address arithmetic on the chosen base overflowed, or the world
    /// returned a base violating the requested alignment.
    Layout(usize),
    /// The caller's scratch refused: the [`LinkCaps`] were out of range,
    /// or the arena could not back the engine's tables — size the buffer
    /// (or the caps) to the load set (Phase A, `res/USERLAND.md` §7).
    Scratch,
}

/// What the stub hands the real crt0: everything left to *do* after the
/// engine went dormant.  Pointers are **absolute** in the linked task's
/// address space; the engine has executed none of them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinkedProgram {
    /// The executable's absolute entry point (the jump target).
    pub entry: u64,
    /// The initial thread-pointer value for [`LinkWorld::set_tp`]'s
    /// register (also recorded here for the caller); 0 when the load
    /// set carries no TLS at all (no block was built, `set_tp` never
    /// ran).
    pub tp: u64,
    /// Static TLS block size per thread (variant-shaped:
    /// [`TlsLayout::total`]) — the static layout used by thread spawning.  0 = no TLS.
    pub tls_total: u64,
    /// Strictest TLS block alignment (≥ 1).
    pub tls_align: u64,
    /// Modules linked (the passive page's `count`).
    pub module_count: usize,
    /// Absolute VA of the sealed passive-ABI [`ModulePage`].
    pub module_page: u64,
    /// `DT_INIT_ARRAY` pointers across the load set, **dependency-
    /// before-dependent** (crt0 runs them in this order, then jumps).
    pub init: HVec<u64, MAX_INIT_FNS>,
    /// `DT_FINI_ARRAY` pointers, dependent-before-dependency (the exact
    /// reverse walk) — collected now because the descriptors die with
    /// the stub and nothing else records them.
    pub fini: HVec<u64, MAX_INIT_FNS>,
}

/// The loaded-module tables — the engine's read half during apply
/// (borrowed immutably while one module's private data is written).
/// Capacity is [`LinkCaps::max_modules`], arena-carved (Phase A).
struct Loaded<W: LinkWorld, A: Allocator + Copy> {
    names: Slots<ImageName, A>,
    descs: Slots<ModuleDescriptor, A>,
    bases: Slots<u64, A>,
    /// Durable read windows, parallel to each descriptor's `segs`
    /// (text/rodata only — data is private, TLS is transient).
    ro: Slots<[Option<W::Ro>; MAX_SEGMENTS], A>,
    /// Lease provenance ids, parallel to `segs` ([`ModuleSeg::lease_id`]).
    ids: Slots<[u32; MAX_SEGMENTS], A>,
    /// Dependency bitmask: bit `j` set in `needs[i]` = module `i` names
    /// module `j` in `DT_NEEDED` (the init-order/cycle input).  Inline —
    /// the mask *width* is the [`MAX_LINK_MODULES`] pin itself.
    needs: [u16; MAX_LINK_MODULES],
    /// The caps this table set was carved for (`names.len()` etc.).
    cap: usize,
    count: usize,
}

impl<W: LinkWorld, A: Allocator + Copy> Loaded<W, A> {
    fn new(cap: usize, alloc: A) -> Result<Self, LinkError<W::Error>> {
        Ok(Self {
            names: Slots::new_from(cap, alloc, |_| ImageName::empty())
                .map_err(|_| LinkError::Scratch)?,
            descs: Slots::new_from(cap, alloc, |_| ModuleDescriptor::empty())
                .map_err(|_| LinkError::Scratch)?,
            bases: Slots::new_from(cap, alloc, |_| 0).map_err(|_| LinkError::Scratch)?,
            ro: Slots::new_from(cap, alloc, |_| core::array::from_fn(|_| None))
                .map_err(|_| LinkError::Scratch)?,
            ids: Slots::new_from(cap, alloc, |_| [0; MAX_SEGMENTS])
                .map_err(|_| LinkError::Scratch)?,
            needs: [0; MAX_LINK_MODULES],
            cap,
            count: 0,
        })
    }

    /// The already-loaded module named `name`, if any (the dedup gate —
    /// a diamond's shared base resolves here the second time).
    fn find(&self, name: &ImageName) -> Option<usize> {
        self.names.as_slice()[..self.count]
            .iter()
            .position(|n| n == name)
    }
}

/// Link one program: BFS the `DT_NEEDED` closure from `exe`, place, map,
/// copy, relocate, seal, lay out TLS, record.  The module contract's
/// entry point; the whole sequence is the module docs' regime.
///
/// Capacity is the **caller's** (Phase A, `res/USERLAND.md` §7): the
/// engine's tables are carved from `alloc` — arena-backed, typically a
/// [`LinkArena`](crate::store::LinkArena) over a buffer the stub owns —
/// at exactly `caps`' sizes.  [`LinkCaps::DEFAULT`] reproduces the
/// pre-parameterization ceilings.
pub fn link_program<A: RelocArch, W: LinkWorld, S: Allocator + Copy>(
    world: &mut W,
    exe: &ImageName,
    alloc: S,
    caps: LinkCaps,
) -> Result<LinkedProgram, LinkError<W::Error>> {
    if !caps.valid() {
        return Err(LinkError::Scratch);
    }
    let mut lt: Loaded<W, S> = Loaded::new(caps.max_modules, alloc)?;
    let mut data: Slots<Option<W::Rw>, S> =
        Slots::new_from(caps.max_modules, alloc, |_| None).map_err(|_| LinkError::Scratch)?;
    let mut tls_leases: Slots<Option<W::Lease>, S> =
        Slots::new_from(caps.max_modules, alloc, |_| None).map_err(|_| LinkError::Scratch)?;

    // ── Phase 1: the BFS closure (dedup by name, bounded) ───────────────
    add_module::<A, W, S>(
        world,
        &mut lt,
        &mut data,
        &mut tls_leases,
        exe,
        Role::Executable,
    )?;
    let mut cursor = 0;
    while cursor < lt.count {
        walk_needed::<A, W, S>(world, &mut lt, &mut data, &mut tls_leases, cursor)?;
        cursor += 1;
    }
    let count = lt.count;

    // ── Phase 2: init order + the cycle gate ────────────────────────────
    let order = init_order(&lt.needs, count).ok_or(LinkError::DependencyCycle)?;

    // ── Phase 3: eager relocation + deferred RELRO seal ─────────────────
    // Scope views borrow the read half; each module's private data is
    // the one write target — map-RW → relocate → seal, never the
    // reverse.  A module's GOT is complete after its own apply (cross-
    // module relocations write only into the applying module). Seal ranges
    // are retained in a bounded list and committed together at the final
    // world crossing; no user code runs in between.
    let views: Slots<ModuleView<'_>, S> = Slots::new_from(count, alloc, |i| {
        let mut ro: [Option<&[u8]>; MAX_SEGMENTS] = [None; MAX_SEGMENTS];
        for (k, w) in lt.ro[i].iter().enumerate() {
            ro[k] = w.as_ref().map(|w| w.as_ref());
        }
        ModuleView {
            base: lt.bases[i],
            descriptor: &lt.descs[i],
            ro,
        }
    })
    .map_err(|_| LinkError::Scratch)?;
    let scope = SymbolScope {
        modules: views.as_slice(),
    };
    let tls_layout = TlsLayout::build::<A>(views.as_slice(), alloc).map_err(LinkError::Tls)?;

    let mut final_seals: HVec<(u64, u64), { MAX_LINK_MODULES + 1 }> = HVec::new();
    for i in 0..count {
        let d = &lt.descs[i];
        let relro = match d.dynz.relro_sz.get() {
            0 => None,
            sz => Some((d.dynz.relro_off.get(), sz)),
        };
        let mut module = ModuleImage {
            scope_index: i,
            view: views[i],
            data: data[i].as_mut().map(|w| w.as_mut()).unwrap_or(&mut []),
            relro,
        };
        let applied = apply_module::<A, _>(&mut module, &scope, &tls_layout)
            .map_err(|e| LinkError::Apply(e, i))?;
        if let Some((va, len)) = applied.relro {
            final_seals
                .push((va, len))
                .map_err(|_| LinkError::TooManyModules)?;
        }
    }

    // ── Phase 4: the static TLS block + TP ──────────────────────────────
    // The layout (tp-relative offsets) fed the relocations above; the
    // block itself is per-thread memory built once here for the initial
    // thread.  V1 quirk guarded: an empty layout still reports the TCB
    // reserve as `total`, so presence is "any module has TLS".
    let have_tls = (0..count).any(|i| lt.descs[i].tls.memsz.get() > 0);
    let mut tp = 0u64;
    if have_tls {
        tp = build_tls_block::<A, W, S>(world, &lt, &mut tls_leases, &tls_layout, count)?;
        // The INSTALL is deferred to the very end of linking (after
        // phase 6): on a session-backed world the thread-pointer move
        // takes effect immediately, and every later world verb would
        // dereference the NEW `%fs` — which no longer names the thread
        // block the trap transport reads its scratch through.  The
        // layout, not the live TP, feeds the relocations above, so
        // nothing before the tail observes the difference.
    }

    // ── Phase 5: init/fini pointers, dependency-ordered ─────────────────
    // Read from the relocated private copies: the array words are
    // absolute function addresses by now.  Sentinel entries (0 and
    // ~0 — legacy crt glue) are skipped, as every runtime linker does.
    let mut init: HVec<u64, MAX_INIT_FNS> = HVec::new();
    let mut fini: HVec<u64, MAX_INIT_FNS> = HVec::new();
    for &i in order[..count].iter() {
        let d = &lt.descs[i as usize];
        collect_fns(
            &lt,
            &data,
            i as usize,
            d.dynz.init_array_off.get(),
            d.dynz.init_array_sz.get(),
            &mut init,
            caps.max_inits,
        )?;
    }
    for &i in order[..count].iter().rev() {
        let d = &lt.descs[i as usize];
        collect_fns(
            &lt,
            &data,
            i as usize,
            d.dynz.fini_array_off.get(),
            d.dynz.fini_array_sz.get(),
            &mut fini,
            caps.max_inits,
        )?;
    }

    // ── Phase 6: the passive ABI (`res/IMAGE.md` §8) ────────────────────
    let page_va = write_module_page::<W, S>(world, &lt, count)?;
    final_seals
        .push((page_va, PAGE_SIZE))
        .map_err(|_| LinkError::TooManyModules)?;

    // The deferred TP install (phase 4's tail): the LAST world action, so
    // no verb ever runs under the moved thread pointer.  The program
    // entered next reads its TLS through exactly this value.
    world
        .finalize_link(&final_seals, have_tls.then_some(tp))
        .map_err(LinkError::World)?;

    let entry = lt.bases[0]
        .checked_add(lt.descs[0].entry.get())
        .ok_or(LinkError::Layout(0))?;

    Ok(LinkedProgram {
        entry,
        tp,
        tls_total: if have_tls { tls_layout.total } else { 0 },
        tls_align: if have_tls { tls_layout.align } else { 1 },
        module_count: count,
        module_page: page_va,
        init,
        fini,
    })
}

/// What part a name plays in the load graph — the **caller's** knowledge,
/// which is the graph edge it walked to reach that name.
///
/// Which descriptor *kinds* satisfy a role is the engine's rule, not the
/// caller's, and the two are not one-to-one: an executable slot admits
/// both addressing conventions — [`kind::EXECUTABLE`], the PIE whose
/// addresses are image-relative, and [`kind::STATIC_EXECUTABLE`], the
/// fixed-address image every founded row and the linker stub itself are —
/// while a `DT_NEEDED` name admits only a library.
///
/// Naming the kind directly (as this once did) collapsed the two and made
/// the static case unrepresentable: the exe slot could name exactly one
/// convention, so a descriptor declaring the other was refused as a
/// foreign graph. Fail-closed, and wrong — `res/IMAGE.md` §4's degenerate
/// static case is precisely a `STATIC_EXECUTABLE` in the exe slot.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Role {
    /// The program being linked.
    Executable,
    /// A name reached through `DT_NEEDED`.
    Library,
}

impl Role {
    /// Whether a descriptor of `kind` may fill this role.  An unknown
    /// kind fills neither: the engine acts on the convention a descriptor
    /// declares, and one it cannot name is not one it may guess.
    const fn admits(self, kind: u8) -> bool {
        match self {
            Role::Executable => matches!(kind, kind::EXECUTABLE | kind::STATIC_EXECUTABLE),
            Role::Library => kind == kind::LIBRARY,
        }
    }
}

/// Acquire, validate, place, and map one module (BFS's per-node work).
/// Returns the module's load index.
fn add_module<A: RelocArch, W: LinkWorld, S: Allocator + Copy>(
    world: &mut W,
    lt: &mut Loaded<W, S>,
    data: &mut Slots<Option<W::Rw>, S>,
    tls_leases: &mut Slots<Option<W::Lease>, S>,
    name: &ImageName,
    role: Role,
) -> Result<usize, LinkError<W::Error>> {
    if lt.count >= lt.cap {
        return Err(LinkError::TooManyModules);
    }
    let idx = lt.count;
    let (desc, mut leases) = world.acquire(name).map_err(LinkError::World)?;

    // The cheap consumer re-check (`apply` re-checks again — defense in
    // depth): stamp, machine, and the *role* this name plays in the
    // graph.  A library with an entry point or an exe named in
    // DT_NEEDED is a foreign graph, refused.
    if !desc.validate() || desc.machine.get() != A::MACHINE || !role.admits(desc.kind) {
        return Err(LinkError::BadModule(idx));
    }
    let live = desc.seg_count as usize;
    // The grant budget promises exactly one lease per live segment; a
    // stray extra grant marks a lying world.
    if leases[live..].iter().any(Option::is_some) {
        return Err(LinkError::BadModule(idx));
    }

    // Address-space span: mappable segments only (KIND_TLS is a copy
    // template, never a mapping — its geometry is per-thread, not
    // address-space).
    let mut span = 0u64;
    let mut has_tls_seg = false;
    for s in desc.live_segs() {
        if s.kind == seg::KIND_TLS {
            has_tls_seg = true;
            continue;
        }
        let end = s
            .vaddr
            .get()
            .checked_add(s.memsz.get())
            .and_then(|e| align_up(e, PAGE_SIZE))
            .ok_or(LinkError::BadModule(idx))?;
        span = span.max(end);
    }
    if span == 0 {
        return Err(LinkError::BadModule(idx));
    }
    // Only nonempty initialization bytes need a source lease. Pure .tbss
    // still allocates a private zeroed block through the TLS descriptor.
    if desc.tls.init_sz.get() > desc.tls.memsz.get()
        || (desc.tls.init_sz.get() > 0) != has_tls_seg
        || desc.live_segs().iter().any(|s| {
            s.kind == seg::KIND_TLS
                && (s.filesz.get() != desc.tls.init_sz.get()
                    || s.memsz.get() != desc.tls.memsz.get())
        })
    {
        return Err(LinkError::BadModule(idx));
    }

    // Placement is the world's act — for an image-relative module.  An
    // **absolute** descriptor's link address IS its runtime address
    // ([`kind::is_absolute`]), so there is no base to choose and the
    // world is not asked: its answer would be a perfectly legitimate
    // address that maps the image somewhere its own relocationless code
    // does not believe it is.  Nothing later distinguishes that from a
    // correct placement — the segments land, the seal succeeds, and the
    // first absolute reference reads whatever else lives there.
    //
    // Zero rather than "the descriptor's own low address" because the
    // engine adds `base + s.vaddr` uniformly below; the convention is
    // carried by the base, not by a second code path.
    let base = if kind::is_absolute(desc.kind) {
        0
    } else {
        world
            .choose_base(span, PAGE_SIZE)
            .map_err(LinkError::World)?
    };
    if base % PAGE_SIZE != 0 || base.checked_add(span).is_none() {
        return Err(LinkError::Layout(idx));
    }

    // Every writable LOAD is realized inside one private, zeroed region.
    // The union preserves the engine's single write authority; individual
    // templates retain their own descriptor bounds and source leases.
    let mut data_start: Option<u64> = None;
    let mut data_end: u64 = 0;
    for s in desc.live_segs().iter().filter(|s| s.kind == seg::KIND_DATA) {
        data_start = Some(data_start.map_or(s.vaddr.get(), |at| at.min(s.vaddr.get())));
        data_end = data_end.max(
            s.vaddr
                .get()
                .checked_add(s.memsz.get())
                .ok_or(LinkError::BadModule(idx))?,
        );
    }
    if let Some(start) = data_start {
        let va = base.checked_add(start).ok_or(LinkError::Layout(idx))?;
        let len = data_end
            .checked_sub(start)
            .ok_or(LinkError::BadModule(idx))?;
        data[idx] = Some(world.alloc_private(va, len).map_err(LinkError::World)?);
    }

    for k in 0..live {
        let s: SegmentDesc = desc.segs[k];
        let lease = leases[k].take().ok_or(LinkError::MissingLease(idx))?;
        lt.ids[idx][k] = world.lease_id(&lease);
        let va = base
            .checked_add(s.vaddr.get())
            .ok_or(LinkError::Layout(idx))?;
        match s.kind {
            // Text shares R|X off the X-blessed lease; rodata plain R.
            // Access is pinned by kind, not echoed from the descriptor —
            // a descriptor asking anything else (any W above all) lies.
            seg::KIND_TEXT => {
                if s.access != seg::ACCESS_R | seg::ACCESS_X {
                    return Err(LinkError::BadModule(idx));
                }
                let w = world
                    .map_image(lease, va, seg::ACCESS_R | seg::ACCESS_X)
                    .map_err(LinkError::World)?;
                lt.ro[idx][k] = Some(w);
            }
            seg::KIND_RODATA => {
                if s.access != seg::ACCESS_R {
                    return Err(LinkError::BadModule(idx));
                }
                let w = world
                    .map_image(lease, va, seg::ACCESS_R)
                    .map_err(LinkError::World)?;
                lt.ro[idx][k] = Some(w);
            }
            // Each data template: transient read view → its declared window
            // inside the module's single private data union. Bss remains
            // zero from the allocation contract; validation permits no gaps.
            seg::KIND_DATA => {
                if s.access != seg::ACCESS_R {
                    return Err(LinkError::BadModule(idx));
                }
                let template = world.read_segment(lease).map_err(LinkError::World)?;
                let filesz =
                    usize::try_from(s.filesz.get()).map_err(|_| LinkError::BadModule(idx))?;
                let start = data_start.ok_or(LinkError::BadModule(idx))?;
                let dst_start = s
                    .vaddr
                    .get()
                    .checked_sub(start)
                    .and_then(|off| usize::try_from(off).ok())
                    .ok_or(LinkError::BadModule(idx))?;
                let dst_end = dst_start
                    .checked_add(filesz)
                    .ok_or(LinkError::BadModule(idx))?;
                let src = template
                    .as_ref()
                    .get(..filesz)
                    .ok_or(LinkError::BadModule(idx))?;
                data[idx]
                    .as_mut()
                    .ok_or(LinkError::BadModule(idx))?
                    .as_mut()
                    .get_mut(dst_start..dst_end)
                    .ok_or(LinkError::BadModule(idx))?
                    .copy_from_slice(src);
                world.retire_read(template).map_err(LinkError::World)?;
            }
            // TLS init: held until the whole load set's layout exists
            // (IE/LE legality is "the set is closed first").
            seg::KIND_TLS => {
                if s.access != seg::ACCESS_R {
                    return Err(LinkError::BadModule(idx));
                }
                tls_leases[idx] = Some(lease);
            }
            _ => return Err(LinkError::BadModule(idx)),
        }
    }

    lt.names[idx] = *name;
    lt.descs[idx] = desc;
    lt.bases[idx] = base;
    lt.count += 1;
    Ok(idx)
}

/// Walk module `i`'s `PT_DYNAMIC` for `DT_NEEDED`, resolving each name
/// against the table (dedup) or loading it (BFS append).  Names are
/// copied out before any mutation — the table's slots never move.
fn walk_needed<A: RelocArch, W: LinkWorld, S: Allocator + Copy>(
    world: &mut W,
    lt: &mut Loaded<W, S>,
    data: &mut Slots<Option<W::Rw>, S>,
    tls_leases: &mut Slots<Option<W::Lease>, S>,
    i: usize,
) -> Result<(), LinkError<W::Error>> {
    let dyn_off = lt.descs[i].dynz.dynamic_off.get();
    let dyn_sz = lt.descs[i].dynz.dynamic_sz.get();
    if dyn_sz % DYN_ENTRY != 0 {
        return Err(LinkError::BadDynamic(i));
    }
    let entries = dyn_sz / DYN_ENTRY;
    let mut found: u16 = 0;
    for e in 0..entries {
        // Re-resolve per entry: the borrow must not outlive the
        // `add_module` call a NEEDED entry may trigger.  Checked walk —
        // a lying `dynamic_off` must refuse, never wrap.
        let at = e
            .checked_mul(DYN_ENTRY)
            .and_then(|o| dyn_off.checked_add(o))
            .ok_or(LinkError::BadDynamic(i))?;
        let bytes =
            module_bytes::<W, S>(lt, data, i, at, DYN_ENTRY).ok_or(LinkError::BadDynamic(i))?;
        let tag = i64::from_le_bytes(
            bytes[..8]
                .try_into()
                .map_err(|_| LinkError::BadDynamic(i))?,
        );
        let val = u64::from_le_bytes(
            bytes[8..]
                .try_into()
                .map_err(|_| LinkError::BadDynamic(i))?,
        );
        if tag == DT_NULL {
            break;
        }
        if tag != DT_NEEDED {
            continue;
        }
        let nm = needed_name::<W, S>(lt, data, i, val).ok_or(LinkError::NeededName(i))?;
        let j = match lt.find(&nm) {
            Some(j) => j,
            None => add_module::<A, W, S>(world, lt, data, tls_leases, &nm, Role::Library)?,
        };
        lt.needs[i] |= 1u16 << j;
        found = found.checked_add(1).ok_or(LinkError::BadDynamic(i))?;
    }
    // The descriptor promised this count at Register; a mapped section
    // saying otherwise marks a swapped or corrupt mapping.
    if found != lt.descs[i].needed_count.get() {
        return Err(LinkError::BadDynamic(i));
    }
    Ok(())
}

/// The NUL-terminated strtab name at `off`, as an [`ImageName`].
fn needed_name<W: LinkWorld, S: Allocator + Copy>(
    lt: &Loaded<W, S>,
    data: &Slots<Option<W::Rw>, S>,
    i: usize,
    off: u64,
) -> Option<ImageName> {
    let strtab_off = lt.descs[i].dynz.strtab_off.get();
    let strtab_sz = lt.descs[i].dynz.strtab_sz.get();
    let avail = strtab_sz.checked_sub(off)?;
    let tail = module_bytes::<W, S>(lt, data, i, strtab_off.checked_add(off)?, avail)?;
    let len = tail.iter().position(|&b| b == 0)?;
    ImageName::new(core::str::from_utf8(&tail[..len]).ok()?)
}

/// Resolve an image-relative range to mapped bytes: the module's
/// text/rodata read windows first (capped at `filesz` and the window's
/// real length), then its private data copy.  A range nothing contains
/// refuses — a lying offset cannot escape the module's own memory.
fn module_bytes<'a, W: LinkWorld, S: Allocator + Copy>(
    lt: &'a Loaded<W, S>,
    data: &'a Slots<Option<W::Rw>, S>,
    i: usize,
    off: u64,
    len: u64,
) -> Option<&'a [u8]> {
    let d = &lt.descs[i];
    for (k, s) in d.live_segs().iter().enumerate() {
        let rel = match off.checked_sub(s.vaddr.get()) {
            Some(r) => r,
            None => continue,
        };
        let end = rel.checked_add(len)?;
        match s.kind {
            seg::KIND_TEXT | seg::KIND_RODATA => {
                let w = lt.ro[i].get(k)?.as_ref()?;
                let bytes = w.as_ref();
                let avail = s.filesz.get().min(u64::try_from(bytes.len()).ok()?);
                if end > avail {
                    continue;
                }
                return bytes.get(usize::try_from(rel).ok()?..usize::try_from(end).ok()?);
            }
            seg::KIND_DATA => {
                if end > s.memsz.get() {
                    continue;
                }
                let bytes = data.as_slice().get(i)?.as_ref()?.as_ref();
                let data_start = d
                    .live_segs()
                    .iter()
                    .filter(|s| s.kind == seg::KIND_DATA)
                    .map(|s| s.vaddr.get())
                    .min()?;
                let union_rel = off.checked_sub(data_start)?;
                let union_end = union_rel.checked_add(len)?;
                return bytes
                    .get(usize::try_from(union_rel).ok()?..usize::try_from(union_end).ok()?);
            }
            _ => continue,
        }
    }
    None
}

/// Dependency-before-dependent order over the `needs` masks — Kahn's
/// walk, deterministic (lowest load index first among the ready), and
/// the cycle gate in one: `None` = no lawful order exists.
fn init_order(needs: &[u16; MAX_LINK_MODULES], count: usize) -> Option<[u8; MAX_LINK_MODULES]> {
    let mut order = [0u8; MAX_LINK_MODULES];
    let mut emitted: u16 = 0;
    let mut n = 0;
    while n < count {
        let ready = (0..count).find(|&i| emitted & (1 << i) == 0 && needs[i] & !emitted == 0)?;
        order[n] = ready as u8;
        emitted |= 1 << ready;
        n += 1;
    }
    Some(order)
}

/// Allocate the initial thread's static TLS block, copy every module's
/// init image into its layout slot, and yield the TP value.  Variant
/// math per [`TlsRules`] (the psABI conventions [`TlsLayout`] pins):
///
/// - **V2** (x86_64): blocks below `tp`; the window is
///   `align_up(total, align)` bytes of block + 8 bytes at `tp` holding
///   the TCB self-pointer (`%fs:0`); `tp = base + align_up(total,
///   align)`.
/// - **V1** (aarch64 tcb 16 / riscv64 tcb 0): `tp = base`; blocks above
///   at their positive offsets; the TCB reserve stays zeroed (no DTV
///   exists in this regime, ever).
///
/// The allocation is over-aligned to `max(align, 16)` so `tp` satisfies
/// both the strictest block alignment and the ABI's own TP-alignment
/// floor.
fn build_tls_block<A: RelocArch, W: LinkWorld, S: Allocator + Copy>(
    world: &mut W,
    lt: &Loaded<W, S>,
    tls_leases: &mut Slots<Option<W::Lease>, S>,
    layout: &TlsLayout<S>,
    count: usize,
) -> Result<u64, LinkError<W::Error>> {
    let align = layout.align.max(16);
    let body = align_up(layout.total, align).ok_or(LinkError::Tls(ApplyError::Input))?;
    // The block also carries the runtime's per-thread block at `tp`
    // (`RUNTIME_TCB_BYTES`, res/IMAGE.md §7's TLS contract), so a linked
    // program can reach its thread-locals *and* issue sessions.  V2 reserves
    // it above `tp`, past the static blocks that sit below; V1's reserve is
    // already inside `body`, because `TlsLayout::build` started its cursor
    // past it.
    let (len, tp_off_in_block) = match A::TLS {
        TlsRules::V2 => (
            body.checked_add(kore_abi::image::RUNTIME_TCB_BYTES)
                .ok_or(LinkError::Tls(ApplyError::Input))?,
            body,
        ),
        TlsRules::V1 { .. } => (body, 0),
    };
    let base = world.choose_base(len, align).map_err(LinkError::World)?;
    if base % align != 0 || base.checked_add(len).is_none() {
        return Err(LinkError::Layout(0));
    }
    let mut block = world.alloc_private(base, len).map_err(LinkError::World)?;
    let tp = base + tp_off_in_block;

    for i in 0..count {
        let d = &lt.descs[i];
        if d.tls.memsz.get() == 0 {
            continue;
        }
        let init_sz =
            usize::try_from(d.tls.init_sz.get()).map_err(|_| LinkError::Tls(ApplyError::Input))?;
        if init_sz == 0 {
            // Pure .tbss carries geometry but needs no initialization lease.
            continue;
        }
        let lease = tls_leases[i].take().ok_or(LinkError::MissingLease(i))?;
        let win = world.read_segment(lease).map_err(LinkError::World)?;
        // The init image's offset inside the page-rounded lease window:
        // `vaddr % page` — sound by the Register gate's page-congruence
        // check (`crate::check`).
        let tls_seg = d
            .live_segs()
            .iter()
            .find(|s| s.kind == seg::KIND_TLS)
            .ok_or(LinkError::Tls(ApplyError::Input))?;
        let in_off = usize::try_from(tls_seg.vaddr.get() % PAGE_SIZE)
            .map_err(|_| LinkError::Tls(ApplyError::Input))?;
        let src = win
            .as_ref()
            .get(
                in_off
                    ..in_off
                        .checked_add(init_sz)
                        .ok_or(LinkError::Tls(ApplyError::Input))?,
            )
            .ok_or(LinkError::Tls(ApplyError::Input))?;
        // The block slot: tp-relative offset folded into the window.
        let off = layout
            .tp_off(i, 0)
            .ok_or(LinkError::Tls(ApplyError::Tls(i)))?;
        let dst_off = i64::try_from(tp_off_in_block)
            .ok()
            .and_then(|tpo| tpo.checked_add(off))
            .and_then(|o| usize::try_from(o).ok())
            .ok_or(LinkError::Tls(ApplyError::Input))?;
        block
            .as_mut()
            .get_mut(
                dst_off
                    ..dst_off
                        .checked_add(init_sz)
                        .ok_or(LinkError::Tls(ApplyError::Input))?,
            )
            .ok_or(LinkError::Tls(ApplyError::Input))?
            .copy_from_slice(src);
        world.retire_read(win).map_err(LinkError::World)?;
    }

    world
        .retain_tls_template(block.as_ref(), tp_off_in_block, align)
        .map_err(LinkError::World)?;

    // V2's TCB self-pointer: `%fs:0` holds `tp` itself (psABI).
    if matches!(A::TLS, TlsRules::V2) {
        let at = usize::try_from(tp_off_in_block).map_err(|_| LinkError::Tls(ApplyError::Input))?;
        block
            .as_mut()
            .get_mut(at..at + 8)
            .ok_or(LinkError::Tls(ApplyError::Input))?
            .copy_from_slice(&tp.to_le_bytes());
    }
    Ok(tp)
}

/// Append one module's init/fini array words (absolute by now — the
/// relocations ran first), skipping the 0 / ~0 sentinels.  Bounded by
/// the caller's [`LinkCaps::max_inits`] dial (≤ the inline
/// [`MAX_INIT_FNS`] ceiling).
fn collect_fns<W: LinkWorld, S: Allocator + Copy>(
    lt: &Loaded<W, S>,
    data: &Slots<Option<W::Rw>, S>,
    i: usize,
    off: u64,
    sz: u64,
    out: &mut HVec<u64, MAX_INIT_FNS>,
    max_inits: usize,
) -> Result<(), LinkError<W::Error>> {
    if sz == 0 {
        return Ok(());
    }
    if sz % 8 != 0 {
        return Err(LinkError::BadDynamic(i));
    }
    let bytes = module_bytes::<W, S>(lt, data, i, off, sz).ok_or(LinkError::BadDynamic(i))?;
    for w in bytes.chunks_exact(8) {
        let fn_addr = u64::from_le_bytes(w.try_into().map_err(|_| LinkError::BadDynamic(i))?);
        if fn_addr == 0 || fn_addr == u64::MAX {
            continue;
        }
        if out.len() >= max_inits {
            return Err(LinkError::TooManyInits);
        }
        out.push(fn_addr).map_err(|_| LinkError::TooManyInits)?;
    }
    Ok(())
}

/// Build, write, and seal the passive-ABI page (`res/IMAGE.md` §8):
/// one [`ModuleRecord`] per module, **placed** (absolute) geometry, the
/// private data copy recorded R|W (its RELRO subrange is already
/// sealed), KIND_TLS omitted — the record lists address-space presence,
/// and a TLS template's product is the per-thread block, not a mapping.
fn write_module_page<W: LinkWorld, S: Allocator + Copy>(
    world: &mut W,
    lt: &Loaded<W, S>,
    count: usize,
) -> Result<u64, LinkError<W::Error>> {
    use zerocopy::byteorder::little_endian::{U32, U64};
    let mut page = ModulePage::new();
    for i in 0..count {
        let d = &lt.descs[i];
        let mut segs = [ModuleSeg::vacant(); MAX_SEGMENTS];
        let mut sc = 0u8;
        for (k, s) in d.live_segs().iter().enumerate() {
            if s.kind == seg::KIND_TLS {
                continue;
            }
            let access = match s.kind {
                seg::KIND_TEXT => seg::ACCESS_R | seg::ACCESS_X,
                seg::KIND_RODATA => seg::ACCESS_R,
                // The private copy actually installed is writable — the
                // record reports the outcome, not the template's share.
                _ => seg::ACCESS_R | seg::ACCESS_W,
            };
            let entry = &mut segs[sc as usize];
            entry.vaddr = U64::new(
                lt.bases[i]
                    .checked_add(s.vaddr.get())
                    .ok_or(LinkError::Layout(i))?,
            );
            entry.size = U64::new(align_up(s.memsz.get(), PAGE_SIZE).ok_or(LinkError::Layout(i))?);
            entry.lease_id = U32::new(lt.ids[i][k]);
            entry.kind = s.kind;
            entry.access = access;
            sc += 1;
        }
        let record = ModuleRecord {
            name_hash: U64::new(lt.names[i].fnv1a()),
            base: U64::new(lt.bases[i]),
            // The entry means what the module's own convention says it
            // means (`ModuleDescriptor::entry` — "read the kind before
            // the address").  The absolute arm is not merely the
            // relative one with a zero base: it states that the address
            // rides through untouched, so a later change to how an
            // absolute module is placed cannot silently start biasing an
            // entry that was never an offset.
            entry: U64::new(match d.kind {
                kind::EXECUTABLE => lt.bases[i]
                    .checked_add(d.entry.get())
                    .ok_or(LinkError::Layout(i))?,
                kind::STATIC_EXECUTABLE => d.entry.get(),
                // A library has nothing to transfer to.
                _ => 0,
            }),
            seg_count: sc,
            kind: d.kind,
            reserved: [0; 6],
            segs,
        };
        // Unreachable by the MAX_LINK_MODULES ≤ MAX_MODULE_RECORDS pin;
        // refused anyway — bounded emission, never silent.
        if !page.push(record) {
            return Err(LinkError::TooManyModules);
        }
    }

    let va = world
        .module_page_base(PAGE_SIZE)
        .map_err(LinkError::World)?;
    if va % PAGE_SIZE != 0 {
        return Err(LinkError::Layout(0));
    }
    let mut window = world
        .alloc_private(va, PAGE_SIZE)
        .map_err(LinkError::World)?;
    let bytes = page.as_bytes();
    window
        .as_mut()
        .get_mut(..bytes.len())
        .ok_or(LinkError::Layout(0))?
        .copy_from_slice(bytes);
    drop(window);
    Ok(va)
}

// ============================================================================
// Self-relocation (the P7 crt0 seam)
// ============================================================================

/// Why [`relocate_self`] refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelfRelocError {
    /// The rela table is not a whole number of entries.
    Shape,
    /// A non-`R_*_RELATIVE` entry (or one naming a symbol) — the
    /// static-PIE stub may carry nothing else; anything else marks a
    /// mislinked stub, refused before a single write.
    Reloc(u32, usize),
    /// `r_offset` escapes the image span.
    Bounds(usize),
    /// `base + addend` overflowed.
    Overflow(usize),
}

/// Apply a static-PIE image's own `R_*_RELATIVE` set — the one pass the
/// real stub must run over *itself* before calling anything relocatable
/// (`res/IMAGE.md` §4 phase 2's first step).  P7's crt0 entry calls
/// this with its load base, its own `.rela.dyn` bytes, and a writable
/// view of its own image (`image[0]` at `load_base`; link base 0, so
/// `r_offset` indexes it directly); constructing those without a heap
/// or relocations is the ~20-line per-arch asm's job, **not** built
/// here.  Host-tested over synthetic tables; returns the write count.
pub fn relocate_self<A: RelocArch>(
    load_base: u64,
    rela: &[u8],
    image: &mut [u8],
) -> Result<usize, SelfRelocError> {
    if rela.len() % 24 != 0 {
        return Err(SelfRelocError::Shape);
    }
    let mut applied = 0usize;
    for (idx, e) in rela.chunks_exact(24).enumerate() {
        let r_offset = u64::from_le_bytes(e[0..8].try_into().map_err(|_| SelfRelocError::Shape)?);
        let r_info = u64::from_le_bytes(e[8..16].try_into().map_err(|_| SelfRelocError::Shape)?);
        let addend = i64::from_le_bytes(e[16..24].try_into().map_err(|_| SelfRelocError::Shape)?);
        let r_type = (r_info & 0xffff_ffff) as u32;
        // RELATIVE only, symbol-free — everything else refuses.
        if (r_info >> 32) != 0 || !matches!(A::classify(r_type), Ok(RelocKind::Relative)) {
            return Err(SelfRelocError::Reloc(r_type, idx));
        }
        let value = load_base
            .checked_add_signed(addend)
            .ok_or(SelfRelocError::Overflow(idx))?;
        let at = usize::try_from(r_offset).map_err(|_| SelfRelocError::Bounds(idx))?;
        image
            .get_mut(at..at.checked_add(8).ok_or(SelfRelocError::Bounds(idx))?)
            .ok_or(SelfRelocError::Bounds(idx))?
            .copy_from_slice(&value.to_le_bytes());
        applied += 1;
    }
    Ok(applied)
}

/// Round up to a power-of-two alignment; `None` on wrap.
fn align_up(v: u64, align: u64) -> Option<u64> {
    let mask = align.checked_sub(1)?;
    v.checked_add(mask).map(|x| x & !mask)
}
