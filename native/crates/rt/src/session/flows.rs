//! The specialized batched APIs — each stages a whole arrangement the
//! docs name as one session, returning typed [`Staged`] refs so later
//! legs chain on results that exist only after commit (SSA).

use kore_abi::cap::{ResourceKind, Rights};
use kore_abi::session::SlotRef;
use kore_abi::task::SpawnFlags;

use super::{BuildError, Deliver, SessionBuilder, Staged};

/// The **process spawn** session's yield (res/IMAGE.md §4/§9f).
#[derive(Clone, Copy, Debug)]
pub struct ProcessParts {
    /// The child's fresh address space (held by the spawner until
    /// [`found`](SessionBuilder::found) consumes it).
    pub space: Staged,
    /// The bootstrap region delegated into the child (the `.kore.boot`
    /// page), or `None` for a bootstrap-less spawn.
    pub bootstrap: Option<Staged>,
    /// The founder's task handle (`ACCESS|MODIFY|DERIVE|REVOKE`), or `None`
    /// for a `DETACHED` spawn — drop = detach.  A delivered client handle is
    /// an attenuated `ACCESS|MODIFY|REVOKE` child, not this retained parent.
    pub handle: Option<Staged>,
}

/// The private regions a process spawn delegates into the child — one
/// `(region, dest_va)`: a source region the spawner holds (stack, data,
/// or the boot page) re-homed into the fresh space at `dest_va`.  The
/// *page-delegation* substrate (#31): the pages stay put, only
/// page-table ownership moves.
#[derive(Clone, Copy, Debug)]
pub struct PrivateRegion {
    /// Arena slot of the source region the spawner holds.
    pub region: SlotRef,
    /// Destination VA in the child's space.
    pub dest_va: u64,
    /// Byte length of the region.
    pub bytes: u64,
    /// Whether the child gets write access (RW vs RO).
    pub write: bool,
}

/// A read-only **image segment** a process spawn installs into the child.
/// Executable segments are R|X shares of an X-blessed lease; data-only
/// segments are R-only. Both use §9b's explicit-destination `ShareToReq`,
/// gated on holding the child space with `MODIFY` — authority consumed by
/// `Found`, so post-spawn injection is structurally impossible.
#[derive(Clone, Copy, Debug)]
pub struct SharedSegment {
    /// Arena slot of the held lease.
    pub lease: SlotRef,
    /// Destination VA in the child's space (page-aligned).
    pub dest_va: u64,
    /// Byte length of the share (the lease's size).
    pub bytes: u64,
    /// Whether the child mapping carries execute permission.
    pub execute: bool,
}

/// A detached data-window token a process spawn delivers to the child.
///
/// Unlike an image [`SharedSegment`], this does not let the founder choose or
/// install a child VA. The founder carves a geometry-only `LeasedRegion` from
/// its held lease; the child receives that affine token and performs the sole
/// install into its own founder-bounded map window. The client↔service ring is
/// the canonical instance (`res/USERLAND.md` §5.3).
#[derive(Clone, Copy, Debug)]
pub struct DetachedWindow {
    /// Arena slot of the lease (held, or a staged result of this session).
    pub lease: SlotRef,
    /// Byte offset into the lease (page-aligned).
    pub offset: u64,
    /// Byte length of the detached window.
    pub bytes: u64,
    /// Receiver write access (RW ring vs RO window).
    pub write: bool,
}

/// The founding **wiring** a process spawn threads through its one bounded
/// session (U1 — cross-space cap delivery at founding): extra
/// explicit-destination `windows` staged between the fresh space and the
/// terminal `Found` (each result landing in the paired `window_results`
/// slot), plus the `deliver` map placing staged children — channel ends,
/// leases, the window views just staged — into the child's arena at the
/// slots its stamped `BootBlock` names.  [`SpawnWiring::NONE`] for a plain
/// spawn.
#[derive(Clone, Copy, Debug)]
pub struct SpawnWiring<'a> {
    /// Detached RW/RO tokens delivered to the child (the §5.3 ring window).
    pub windows: &'a [DetachedWindow],
    /// One result slot per window, in order (SSA).
    pub window_results: &'a [SlotRef],
    /// The delivery map: staged children → child arena slots.
    pub deliver: &'a [Deliver],
    /// Stage a **self-view** of the founded space (#407): a thread view
    /// (`ACCESS | MODIFY | DERIVE`) derived off the staged space *before*
    /// the terminal `Found` consumes it, landing in this result slot — the
    /// caller's `deliver` map then places it into the child's arena at the
    /// slot its boot block names as [`role::ADDRESS_SPACE`]
    /// (`kore_abi::task::role::ADDRESS_SPACE`).  This is the capability
    /// the child's SET_TP verb subjects (crt0's `%fs` anchor) and its own
    /// thread spawns derive from.  `None` = no self-view (a child that
    /// will never set a TP or spawn threads).
    pub self_view: Option<SlotRef>,
}

impl SpawnWiring<'_> {
    /// No wiring — the plain spawn (windows and deliveries both empty).
    pub const NONE: SpawnWiring<'static> = SpawnWiring {
        windows: &[],
        window_results: &[],
        deliver: &[],
        self_view: None,
    };
}

/// **The process spawn in one bounded session** (res/IMAGE.md §4 phase
/// 1, §9f): mint the child's fresh address space off the held `VirtMemory`,
/// delegate each private region into it (the last one may be the
/// bootstrap `.kore.boot` page — pass its index in `bootstrap_region`),
/// share each read-only [`SharedSegment`] into it (R-only or R|X, legal
/// only while the spawner still holds the space), stage the
/// [`wiring`](SpawnWiring) — its shared windows (the
/// §5.3 ring) and its **delivery map** (U1: staged children placed into
/// the child's arena at named slots) — then `Found`, consuming the space
/// into the founded task.  *One crossing, one closure* (SYSCALL.md D4: a
/// spawn fits in one trap).
///
/// `regions` are delegated in order (`results` supplies one slot per
/// delegation), then `segments` are shared (`segment_results` one slot each),
/// then `wiring.windows` (`wiring.window_results` one slot each).
/// `bootstrap_region` names which delegation (by index into `regions`)
/// is the boot page — its child-VA base rides the founded task's `arg`;
/// `None` = no bootstrap.  The founder's handle (or nothing, for
/// `DETACHED`) lands in `handle_slot`.
#[allow(clippy::too_many_arguments)]
pub fn spawn_process(
    b: &mut SessionBuilder,
    virt_memory: SlotRef,
    entry: u64,
    stack: u64,
    flags: SpawnFlags,
    arena_slots: u32,
    authority: SlotRef,
    regions: &[PrivateRegion],
    bootstrap_region: Option<usize>,
    segments: &[SharedSegment],
    space_slot: SlotRef,
    results: &[SlotRef],
    segment_results: &[SlotRef],
    handle_slot: SlotRef,
    wiring: &SpawnWiring<'_>,
) -> Result<ProcessParts, BuildError> {
    if results.len() < regions.len()
        || segment_results.len() < segments.len()
        || wiring.window_results.len() < wiring.windows.len()
    {
        return Err(BuildError::TooManyBuffers);
    }
    // A "process" must NOT set SHARE_SPACE (that is the thread case,
    // paired with a thread-view space — a different flow).
    if flags.contains(SpawnFlags::SHARE_SPACE) {
        return Err(BuildError::BadSpawnFlags);
    }
    // 1 · The fresh space — full grant (the founded root maps, derives,
    //     and must be killable).
    let space = b.derive_fresh_space(
        virt_memory,
        Rights::ACCESS | Rights::MODIFY | Rights::DERIVE | Rights::REVOKE,
        space_slot,
    )?;
    // 1b · The child's self-view (#407): a thread view of the space it is
    //      about to be founded into, staged while the space capability
    //      still exists (`Found` consumes it — same injection window as
    //      the segment shares below).  MODIFY authorizes the child's
    //      SET_TP; DERIVE lets it stage thread views for its own spawns;
    //      deliberately no REVOKE (the space's life belongs to the
    //      founder's kept root, never to a view a thread could retire).
    if let Some(slot) = wiring.self_view {
        b.derive_view(
            space.into(),
            // REVOKE included: on a VIEW it retires only the view's own
            // tag (`AddrSpace` has no `Teardown` — the space and its ASID
            // are untouched), and the thread-spawn delegate's destination
            // gate demands the full grant on the staged view.
            Rights::ACCESS | Rights::MODIFY | Rights::DERIVE | Rights::REVOKE,
            slot,
        )?;
    }
    // 2 · Delegate each private region into the child.
    let mut bootstrap = None;
    for (i, r) in regions.iter().enumerate() {
        let staged = b.delegate(
            r.region,
            space.into(),
            r.dest_va,
            r.bytes,
            r.write,
            0,
            results[i],
        )?;
        if bootstrap_region == Some(i) {
            bootstrap = Some(staged);
        }
    }
    // 3 · Share each read-only image segment while the space capability is
    //     still ours (Found consumes it — the injection window closes).
    for (i, segment) in segments.iter().enumerate() {
        let access = if segment.execute {
            kore_abi::session::SHARE_ACCESS_R | kore_abi::session::SHARE_ACCESS_X
        } else {
            kore_abi::session::SHARE_ACCESS_R
        };
        b.share_to(
            segment.lease,
            Rights::ACCESS,
            space.into(),
            segment.dest_va,
            segment.bytes,
            false,
            access,
            segment_results[i],
        )?;
    }
    // 3b · Carve each wiring window as a detached token. No child PTE exists
    //      yet: `Found` delivers the token and the child performs the sole
    //      install through `map_view`. REVOKE lets task teardown retire the
    //      affine view name; MODIFY controls the eventual leaf's write bit.
    for (i, w) in wiring.windows.iter().enumerate() {
        let requested = if w.write {
            Rights::ACCESS | Rights::MODIFY | Rights::REVOKE
        } else {
            Rights::ACCESS | Rights::REVOKE
        };
        b.carve_window(
            w.lease,
            requested,
            w.offset,
            w.bytes,
            wiring.window_results[i],
        )?;
    }
    // 4 · Found — the terminal, consuming the space; the delivery map
    //     rides the FoundReq (descriptor ABI v3).
    let handle = b.found_deliver(
        space,
        entry,
        stack,
        bootstrap,
        flags,
        arena_slots,
        authority,
        handle_slot,
        wiring.deliver,
    )?;
    Ok(ProcessParts {
        space,
        bootstrap,
        handle: (!flags.contains(SpawnFlags::DETACHED)).then_some(handle),
    })
}

/// How many capabilities one founding may endow its child with.
///
/// Re-derived for the doorbell star (`res/SPMD.md` §3): the widest
/// lawful endowment list is the star's HUB — one send into every
/// possible sibling's doorbell (`MAX_THREADS − 1`) plus its own recv —
/// so the bound is that plan, not a taste cap.  (The old four-cap
/// reading — "a worker needing more is a service" — undercounted
/// exactly this relay, whose whole job is holding its siblings'
/// doorbells.)  Still bounded for the ordinary reason every session
/// quantity here is (`res/SYSCALL.md` D4): each endowment costs an
/// invocation and a delivery-map entry inside limits the kernel
/// enforces anyway — 32 + 2 fixed deliveries fit the 78-entry map, and
/// 32 derives fit the 107-invocation session.
pub const MAX_THREAD_ENDOW: usize = 1 + (crate::thread::MAX_THREADS as usize - 1);

/// One capability a [`spawn_thread`] founding derives into the **child's**
/// arena — the thread-scale analogue of a founded service's endowment
/// recipe.
///
/// The parent names a subject it holds and the kind to derive; the child
/// receives an attenuation of it, sealed to the child's own holder stamp,
/// at a slot its boot block names under a role. The parent keeps the
/// subject, so revoking the endowment stays the parent's lever — the same
/// posture the founder takes toward a service it founds.
#[derive(Clone, Copy, Debug)]
pub struct ThreadEndow {
    /// The **parent-held** subject to derive from.
    pub subject: SlotRef,
    /// Rights the child receives — an attenuation of the subject's,
    /// re-checked against the twin at the kernel (Σ5).
    pub rights: Rights,
    /// The kind the derive produces (`ResourceKind as u8`), naming which
    /// of a subject's several derives is meant — a device's self-kind
    /// grant, its interrupt route, its port grant.
    pub kind: u8,
    /// The **parent-arena** SSA result slot the derive lands in.
    pub result: SlotRef,
    /// Where the delivered capability lands in the **child's** arena.
    pub child_slot: u16,
}

/// **The thread spawn in one bounded session** (#407 — the unified task
/// model): a task founds a THREAD of itself.  To the kernel the thread
/// is an ordinary task — its own arena, doorbell-less parks, scheduler
/// identity — differing only in the address space, which is the
/// caller's own:
///
/// 1. a **thread view** of the caller's space, derived off its delivered
///    `role::ADDRESS_SPACE` self-view — the root `Found` consumes;
/// 2. a **second view**, delivered into the child's arena at the slot its
///    boot block names as `role::ADDRESS_SPACE` (the child's own SET_TP
///    subject and its own future spawns' parent);
/// 3. the **boot page**: a parent-held, PRE-WRITTEN `Region` (the
///    provisioning session mapped it; the parent stamped `.kore.boot` +
///    `.kore.geom` naming the child's scratch/slots/map bands and its
///    runtime block) — delegated here so `Found` reads the geometry and
///    installs the child's own aliases, and its dest VA rides to the
///    child as `arg`;
/// 4. `Found` with `SHARE_SPACE` + [`NO_AUTHORITY`] — the root-proven
///    thread founding needs no `ExecAuthority` (a same-space thread adds
///    no reachable authority);
/// 5. the thread's **own task handle** at `role::SELF`
///    ([`Deliver::self_task`]) — the subject `kore_rt::thread::exit`
///    presents.  It carries no `REVOKE`, so a thread can end itself but
///    cannot reap itself out from under whoever is joining it.
///
/// The stack and runtime-block pages were mapped by the provisioning
/// session and need no staging here — they are already visible in the
/// shared space.
///
/// ## Endowing the thread
///
/// A thread's arena starts effectively empty, and that is not an
/// oversight to route around: capability holdings are **per task**, and a
/// thread is a task.  Sharing an address space shares memory, never
/// authority — which is exactly what makes the I/O-port bitmap safe to
/// key on the running task (a sibling that was never granted ports
/// cannot acquire them by sharing a space with one that was).
///
/// So a thread that must *hold* something — a relay parking on its own
/// interrupt route, a worker driving its own device — is endowed the same
/// way a founded service is: the parent stages one derive per
/// [`ThreadEndow`] off a subject it holds and delivers each result into
/// the child's arena, where the child's boot block names it under a role.
/// Nothing is moved out of the parent: each is an attenuation of a
/// subject the parent keeps.
///
/// **Whether the parent can then *revoke* it depends on the subject.**
/// `REVOKE` is subject to Σ5 like any other right, so an endowment
/// carries it only if the parent's own capability does — which is true
/// of a founder-seeded device grant and **false of every founder-carved
/// `ChanSend`** (`stage_pair` grants sends `ACCESS | DERIVE`). So a
/// service can revoke a worker's device authority but not the doorbell
/// send it derived into it; that end dies only when the founder's
/// channel root does, and T6 takes every end with it at once.
///
/// Nothing depends on the difference today — two threads of one task
/// share a lifetime — but it is an asymmetry rather than a
/// simplification, and this doc previously claimed the stronger
/// property for all of them.
///
/// [`NO_AUTHORITY`]: kore_abi::session::NO_AUTHORITY
#[allow(clippy::too_many_arguments)]
pub fn spawn_thread(
    b: &mut SessionBuilder,
    self_space: SlotRef,
    entry: u64,
    stack: u64,
    spmd: bool,
    boot_region: SlotRef,
    boot_va: u64,
    boot_bytes: u64,
    arena_slots: u32,
    space_slot: SlotRef,
    child_view_slot: SlotRef,
    boot_result: SlotRef,
    handle_slot: SlotRef,
    child_addr_space_slot: u16,
    child_self_slot: u16,
    endow: &[ThreadEndow],
) -> Result<Staged, BuildError> {
    if endow.len() > MAX_THREAD_ENDOW {
        return Err(BuildError::TooManyDeliveries);
    }
    let space = b.derive_view(
        self_space,
        Rights::ACCESS | Rights::MODIFY | Rights::DERIVE | Rights::REVOKE,
        space_slot,
    )?;
    // Derived off the STAGED ROOT VIEW (not the caller's own view): the
    // delivered self-view must be a DESCENDANT of the child's root —
    // `Task::adopt`'s T6 lineage gate — so killing the thread reaps it,
    // and a sibling view would be silently dropped from the holdings
    // (record without possession: SET_TP would find no member).
    let child_view = b.derive_view(
        space.into(),
        Rights::ACCESS | Rights::MODIFY | Rights::DERIVE | Rights::REVOKE,
        child_view_slot,
    )?;
    let bootstrap = b.delegate(
        boot_region,
        space.into(),
        boot_va,
        boot_bytes,
        false,
        0,
        boot_result,
    )?;
    // The self-view first, then the thread's own task handle, then one
    // staged derive per endowment.  The array is pre-filled with the
    // self-view delivery rather than a sentinel so there is no "empty"
    // `Deliver` to construct: only the first `n` entries are ever passed
    // on.
    let mut deliver = [Deliver::new(child_view, child_addr_space_slot); 2 + MAX_THREAD_ENDOW];
    // The SELF handle (`role::SELF`) — what `thread::exit` presents.  A
    // thread had none, which is exactly why it could only park forever:
    // the exit verb needs a subject, and a task's authority over *itself*
    // is a capability like any other, not an ambient privilege.
    deliver[1] = Deliver::self_task(child_self_slot);
    let mut n = 2;
    for e in endow {
        let staged = if e.kind == ResourceKind::Task as u8 && e.rights == Rights::ACCESS {
            b.derive_task_observer(e.subject, e.result)?
        } else {
            b.derive_raw(e.subject, e.rights, e.kind, e.result, None)?
        };
        deliver[n] = Deliver::new(staged, e.child_slot);
        n += 1;
    }
    // An SPMD worker rides the G6 pin flag (`res/SPMD.md` §7): the
    // kernel pins it to one class-chosen core instead of the whole node.
    let flags = if spmd {
        SpawnFlags::SHARE_SPACE | SpawnFlags::SPMD_WORKER
    } else {
        SpawnFlags::SHARE_SPACE
    };
    b.found_deliver(
        space,
        entry,
        stack,
        Some(bootstrap),
        flags,
        arena_slots,
        SlotRef::from_wire(u32::from(kore_abi::session::NO_AUTHORITY)),
        handle_slot,
        &deliver[..n],
    )
}

/// One worker's staging inputs for a [`spawn_gang`] founding — the
/// per-worker half of [`spawn_thread`]'s parameter list; the shared half
/// (the space, the flags, the child boot-block slots) rides the call.
#[derive(Clone, Copy, Debug)]
pub struct GangWorker<'a> {
    /// The worker's entry PC.
    pub entry: u64,
    /// The worker's initial stack pointer.
    pub stack: u64,
    /// SSA result slot of the staged root view this worker's `Found`
    /// consumes.
    pub space_slot: SlotRef,
    /// SSA result slot of the child's delivered self-view.
    pub child_view_slot: SlotRef,
    /// The pre-written boot page: the parent-held `Region`…
    pub boot_region: SlotRef,
    /// …its destination VA and byte length in the shared space…
    pub boot_va: u64,
    /// (see `boot_va`)
    pub boot_bytes: u64,
    /// …and the delegate's SSA result slot.
    pub boot_result: SlotRef,
    /// Where the founder's task handle for this worker lands.
    pub handle_slot: SlotRef,
    /// Per-worker endowments (each list ≤ [`MAX_THREAD_ENDOW`]).
    pub endow: &'a [ThreadEndow],
}

/// **The gang founding in one bounded session** (G5 — `res/SPMD.md` §7):
/// N thread foundings of the caller's own space staged as N leg runs
/// followed by the **Found tail**, so the whole gang pays the microkernel
/// tax once.  Per worker, the legs and the `Found` are exactly
/// [`spawn_thread`]'s — the same two views (the delivered self-view
/// derived off the STAGED root, the T6 lineage rule), the same delegated
/// boot page, the same `SHARE_SPACE`(`|SPMD_WORKER`) flags, the same
/// delivery map shape — and every non-Found invocation (the legs AND the
/// endowment derives) is staged in phase 1, because the tail admits only
/// `Found`s.  Refuses more than
/// [`MAX_SESSION_FOUNDS`](super::builder::MAX_SESSION_FOUNDS) workers
/// (the descriptor's exact founding capacity — chunk the launch, the
/// `spawn_spmd` remedy) and any endowment list past [`MAX_THREAD_ENDOW`].
pub fn spawn_gang(
    b: &mut SessionBuilder,
    self_space: SlotRef,
    spmd: bool,
    arena_slots: u32,
    child_addr_space_slot: u16,
    child_self_slot: u16,
    workers: &[GangWorker<'_>],
) -> Result<(), BuildError> {
    use super::builder::MAX_SESSION_FOUNDS;
    if workers.len() > MAX_SESSION_FOUNDS {
        return Err(BuildError::TooManyFounds);
    }
    // Phase 1 · every worker's non-Found invocations, in worker order:
    // the two views, the boot delegate, and the endowment derives — each
    // worker's finished delivery map is recorded for the tail.
    type Legs = (Staged, Staged, [Deliver; 2 + MAX_THREAD_ENDOW], usize);
    let mut legs: [Option<Legs>; MAX_SESSION_FOUNDS] = [None; MAX_SESSION_FOUNDS];
    for (i, w) in workers.iter().enumerate() {
        if w.endow.len() > MAX_THREAD_ENDOW {
            return Err(BuildError::TooManyDeliveries);
        }
        let space = b.derive_view(
            self_space,
            Rights::ACCESS | Rights::MODIFY | Rights::DERIVE | Rights::REVOKE,
            w.space_slot,
        )?;
        let child_view = b.derive_view(
            space.into(),
            Rights::ACCESS | Rights::MODIFY | Rights::DERIVE | Rights::REVOKE,
            w.child_view_slot,
        )?;
        let bootstrap = b.delegate(
            w.boot_region,
            space.into(),
            w.boot_va,
            w.boot_bytes,
            false,
            0,
            w.boot_result,
        )?;
        let mut deliver = [Deliver::new(child_view, child_addr_space_slot); 2 + MAX_THREAD_ENDOW];
        deliver[1] = Deliver::self_task(child_self_slot);
        let mut n = 2;
        for e in w.endow {
            let staged = b.derive_raw(e.subject, e.rights, e.kind, e.result, None)?;
            deliver[n] = Deliver::new(staged, e.child_slot);
            n += 1;
        }
        legs[i] = Some((space, bootstrap, deliver, n));
    }
    // Phase 2 · the Found tail — one per worker, each with its own map.
    let flags = if spmd {
        SpawnFlags::SHARE_SPACE | SpawnFlags::SPMD_WORKER
    } else {
        SpawnFlags::SHARE_SPACE
    };
    for (i, w) in workers.iter().enumerate() {
        let Some((space, bootstrap, deliver, n)) = legs[i] else {
            // Unreachable: phase 1 filled `0..workers.len()`.
            continue;
        };
        b.found_deliver(
            space,
            w.entry,
            w.stack,
            Some(bootstrap),
            flags,
            arena_slots,
            SlotRef::from_wire(u32::from(kore_abi::session::NO_AUTHORITY)),
            w.handle_slot,
            &deliver[..n],
        )?;
    }
    Ok(())
}

/// The peer-share arrangement's yield.
#[derive(Clone, Copy, Debug)]
pub struct SharedParts {
    /// The lease — the caller keeps it: **sole physical authority**,
    /// and the lessor position (only your lessor can page you).
    pub lease: Staged,
    /// The caller's own mapped window.
    pub mine: Staged,
}

/// **Share with a peer, one crossing**: lease a fresh buffer, map my
/// window, and transfer an attenuated child to the peer over its
/// channel — [lease, share_cpu, rendezvous-send].  The peer stages its
/// own `share_cpu` on the delivered cap (its crossing, its VA).  On a
/// conventional kernel this is five syscalls and two processes'
/// round-trips; here it is one commit (all-or-nothing: no posted peer
/// recv ⇒ `WouldBlock`, nothing half-shared).
pub fn share_with(
    b: &mut SessionBuilder,
    region: SlotRef,
    chan_send: SlotRef,
    size: u64,
    my_va: u64,
    peer_rights: Rights,
    results: [SlotRef; 2],
) -> Result<SharedParts, BuildError> {
    let lease = b.lease(
        region,
        Rights::ACCESS | Rights::MODIFY | Rights::DERIVE,
        size,
        results[0],
    )?;
    let mine = b.share_cpu(
        lease,
        Rights::ACCESS | Rights::MODIFY,
        my_va,
        size,
        true,
        results[1],
    )?;
    b.send(chan_send, lease, peer_rights)?;
    Ok(SharedParts { lease, mine })
}

/// The driver bring-up's yield.
#[derive(Clone, Copy, Debug)]
pub struct DeviceParts {
    /// The claimed BAR (an `MmioRegion`).
    pub bar: Staged,
    /// The BAR mapped into the driver's space.
    pub regs: Staged,
    /// The DMA buffer's lease (the shared root).
    pub lease: Staged,
    /// The buffer's CPU window.
    pub buf: Staged,
    /// The buffer's IOVA window in the device's domain.
    pub iova: Staged,
    /// The interrupt route (`WAIT_IRQ` parks on it).
    pub irq: Staged,
}

/// Parameters for [`device_setup`] — the whole bring-up, named once.
#[derive(Clone, Copy, Debug)]
pub struct DeviceSetup {
    /// Arena slot of the `PciDevice` capability (`role::DEVICE`).
    pub device: SlotRef,
    /// Arena slot of the DMA-capable region (`role::DMA_POOL`).
    pub region: SlotRef,
    /// Which BAR to claim.
    pub bar: u8,
    /// VA to map the BAR registers at.
    pub regs_va: u64,
    /// BAR window length.
    pub regs_bytes: u64,
    /// DMA buffer size to lease.
    pub buf_size: u64,
    /// VA for the buffer's CPU window.
    pub buf_va: u64,
    /// IOVA for the buffer in the device's domain.
    pub iova: u64,
    /// Device write access to the buffer (RW DMA vs device-read-only).
    pub dma_write: bool,
}

/// **The whole driver bring-up in one trap** (the canonical device-setup
/// fold): claim the BAR → map the registers → lease the DMA
/// buffer → CPU window + IOVA window over the *same frames* (RUNG 2,
/// zero-copy) → establish the interrupt route.  *6 invocations, 1
/// crossing* — and atomic: a mid-sequence refusal commits nothing, so
/// no half-configured device exists, ever.
pub fn device_setup(
    b: &mut SessionBuilder,
    p: DeviceSetup,
    results: [SlotRef; 6],
) -> Result<DeviceParts, BuildError> {
    let bar = b.claim_bar(p.device, Rights::ACCESS | Rights::MODIFY, p.bar, results[0])?;
    let regs = b.share_cpu(
        bar,
        Rights::ACCESS | Rights::MODIFY,
        p.regs_va,
        p.regs_bytes,
        true,
        results[1],
    )?;
    let lease = b.lease(
        p.region,
        Rights::ACCESS | Rights::MODIFY | Rights::DERIVE,
        p.buf_size,
        results[2],
    )?;
    let buf = b.share_cpu(
        lease,
        Rights::ACCESS | Rights::MODIFY,
        p.buf_va,
        p.buf_size,
        true,
        results[3],
    )?;
    let iova = b.share_device(
        lease,
        Rights::ACCESS | Rights::MODIFY,
        p.device,
        p.iova,
        p.buf_size,
        p.dma_write,
        results[4],
    )?;
    let irq = b.derive_irq_route(p.device, Rights::ACCESS, results[5])?;
    Ok(DeviceParts {
        bar,
        regs,
        lease,
        buf,
        iova,
        irq,
    })
}

/// **Batch teardown**: revoke `subjects` in one crossing — a cluster
/// reap (T6 takes each subtree at commit).  This is also the swap
/// page-out's precursor: eviction is `unshare×N` in one session once
/// the paged-unshare verb (leaf-clear + managed-absent marker)
/// lands — same shape, sharper effect.
pub fn revoke_batch(b: &mut SessionBuilder, subjects: &[SlotRef]) -> Result<(), BuildError> {
    for &s in subjects {
        b.revoke(s)?;
    }
    Ok(())
}

// ── The generalized multi-buffer DMA primitive (the driver keystone) ─

/// One DMA buffer's placement: a `size`-byte buffer mapped into the
/// caller's space at `cpu_va` and into the device's IOMMU domain at
/// `iova`, `write`-able by the device (RW) or read-only.
#[derive(Clone, Copy, Debug)]
pub struct BufSpec {
    /// Buffer size in bytes.
    pub size: u64,
    /// VA for the caller's CPU window.
    pub cpu_va: u64,
    /// IOVA for the device's window over the *same* frames.
    pub iova: u64,
    /// Device write access (RW DMA) vs. device-read-only.
    pub write: bool,
}

/// One buffer's three caps: the shared root lease and its two windows
/// (RUNG 2 — one lease, a CPU window and an IOVA window over the same
/// frames).
#[derive(Clone, Copy, Debug)]
pub struct DmaWindows {
    /// The lease — the caller holds it (sole physical authority, the
    /// lessor position).
    pub lease: Staged,
    /// The caller's CPU window.
    pub cpu: Staged,
    /// The device's IOVA window.
    pub iova: Staged,
}

/// Invocations one [`BufSpec`] stages (lease + CPU share + device share).
pub const INVS_PER_BUFFER: usize = 3;

/// **Stage N device-visible DMA buffers in one crossing** — the
/// generalized multi-buffer DMA primitive [`device_setup`] leases *one*
/// of.  Each [`BufSpec`] becomes a (lease → `share_cpu` → `share_device`)
/// triple over the same frames (RUNG 2, zero-copy); all N land in one
/// session commit, atomic (a mid-batch refusal commits nothing).
///
/// Slots are caller-supplied (SSA): `results` holds
/// [`INVS_PER_BUFFER`]` × specs.len()` slots (lease, cpu, iova per
/// buffer, in order); each buffer's [`DmaWindows`] is written into
/// `out`.  This is every driver's ring/buffer allocation — NVMe stages
/// its admin SQ+CQ and each qpair's SQ+CQ through it.
pub fn dma_buffers(
    b: &mut SessionBuilder,
    region: SlotRef,
    device: SlotRef,
    specs: &[BufSpec],
    results: &[SlotRef],
    out: &mut [DmaWindows],
) -> Result<usize, BuildError> {
    if results.len() < specs.len() * INVS_PER_BUFFER || out.len() < specs.len() {
        return Err(BuildError::TooManyBuffers);
    }
    for (i, spec) in specs.iter().enumerate() {
        let r = &results[i * INVS_PER_BUFFER..];
        let lease = b.lease(
            region,
            Rights::ACCESS | Rights::MODIFY | Rights::DERIVE,
            spec.size,
            r[0],
        )?;
        let cpu = b.share_cpu(
            lease,
            Rights::ACCESS | Rights::MODIFY,
            spec.cpu_va,
            spec.size,
            true,
            r[1],
        )?;
        let iova = b.share_device(
            lease,
            Rights::ACCESS | Rights::MODIFY,
            device,
            spec.iova,
            spec.size,
            spec.write,
            r[2],
        )?;
        out[i] = DmaWindows { lease, cpu, iova };
    }
    Ok(specs.len())
}

// ── NVMe compositions (the first customer of the primitive) ──────────

/// The controller bring-up's yield.
#[derive(Clone, Copy, Debug)]
pub struct NvmeBringup {
    /// The claimed BAR0 (`MmioRegion`).
    pub bar: Staged,
    /// BAR0 mapped into the driver's space (the register window).
    pub regs: Staged,
    /// The admin submission queue's windows.
    pub admin_sq: DmaWindows,
    /// The admin completion queue's windows.
    pub admin_cq: DmaWindows,
    /// The controller's interrupt route (`WAIT_IRQ` parks on it).
    pub irq: Staged,
}

/// Parameters for [`nvme_bringup`].
#[derive(Clone, Copy, Debug)]
pub struct NvmeBringupParams {
    /// `role::DEVICE` slot (the NVMe `PciDevice`).
    pub device: SlotRef,
    /// `role::DMA_POOL` slot (the DMA-capable region).
    pub region: SlotRef,
    /// BAR to claim (0 for NVMe).
    pub bar: u8,
    /// VA + length for the BAR0 register window.
    pub regs_va: u64,
    /// BAR0 window length.
    pub regs_bytes: u64,
    /// Admin SQ buffer placement.
    pub admin_sq: BufSpec,
    /// Admin CQ buffer placement.
    pub admin_cq: BufSpec,
}

/// **The whole NVMe controller bring-up in one crossing**: claim BAR0
/// → map the register window → lease the admin SQ + CQ with CPU + IOVA
/// windows ([`dma_buffers`]) → establish the interrupt route.  *9
/// invocations, 1 trap* (vs. [`device_setup`]'s 6 — this leases two
/// buffers, not one), and atomic.  The controller-config MMIO writes
/// (`CC`/`AQA`/`ASQ`/`ACQ`) ride the mapped register window afterward,
/// not the session.
pub fn nvme_bringup(
    b: &mut SessionBuilder,
    p: NvmeBringupParams,
    bar: SlotRef,
    regs: SlotRef,
    buf_slots: &[SlotRef; 2 * INVS_PER_BUFFER],
    irq: SlotRef,
) -> Result<NvmeBringup, BuildError> {
    let bar = b.claim_bar(p.device, Rights::ACCESS | Rights::MODIFY, p.bar, bar)?;
    let regs = b.share_cpu(
        bar,
        Rights::ACCESS | Rights::MODIFY,
        p.regs_va,
        p.regs_bytes,
        true,
        regs,
    )?;
    let mut win = [DmaWindows {
        lease: bar,
        cpu: bar,
        iova: bar,
    }; 2];
    dma_buffers(
        b,
        p.region,
        p.device,
        &[p.admin_sq, p.admin_cq],
        buf_slots,
        &mut win,
    )?;
    let irq = b.derive_irq_route(p.device, Rights::ACCESS, irq)?;
    Ok(NvmeBringup {
        bar,
        regs,
        admin_sq: win[0],
        admin_cq: win[1],
        irq,
    })
}

/// An I/O queue pair's yield (its SQ + CQ windows).
#[derive(Clone, Copy, Debug)]
pub struct NvmeQpair {
    /// The submission queue's windows.
    pub sq: DmaWindows,
    /// The completion queue's windows.
    pub cq: DmaWindows,
}

/// **An NVMe I/O qpair's memory in one crossing**: lease the SQ + CQ
/// with CPU + IOVA windows ([`dma_buffers`]).  *6 invocations, 1 trap.*
/// The two NVMe-level admin commands (Create-CQ / Create-SQ) ride the
/// admin qpair afterward, not the session.  At SPDK's one-qpair-per-core
/// scale this crossing runs once per core — the batching amortizes hard.
pub fn nvme_qpair(
    b: &mut SessionBuilder,
    region: SlotRef,
    device: SlotRef,
    sq: BufSpec,
    cq: BufSpec,
    buf_slots: &[SlotRef; 2 * INVS_PER_BUFFER],
) -> Result<NvmeQpair, BuildError> {
    // Placeholder windows, overwritten by `dma_buffers` (its `out`);
    // `Staged` is the parent module's, constructible from this child.
    let ph = Staged(SlotRef::from_wire(0));
    let mut win = [DmaWindows {
        lease: ph,
        cpu: ph,
        iova: ph,
    }; 2];
    dma_buffers(b, region, device, &[sq, cq], buf_slots, &mut win)?;
    Ok(NvmeQpair {
        sq: win[0],
        cq: win[1],
    })
}

// ── xHCI compositions (the second customer of the primitive) ─────────

/// The xHCI controller bring-up's yield.
#[derive(Clone, Copy, Debug)]
pub struct XhciBringup {
    /// The claimed BAR0 (`MmioRegion`).
    pub bar: Staged,
    /// BAR0 mapped into the driver's space (the register window).
    pub regs: Staged,
    /// The Device Context Base Address Array windows.
    pub dcbaa: DmaWindows,
    /// The command ring windows.
    pub cmd_ring: DmaWindows,
    /// Interrupter 0's event ring windows.
    pub event_ring: DmaWindows,
    /// The Event Ring Segment Table windows.
    pub erst: DmaWindows,
    /// The controller's interrupt route (`WAIT_IRQ` parks on it).
    pub irq: Staged,
}

/// Parameters for [`xhci_bringup`].
#[derive(Clone, Copy, Debug)]
pub struct XhciBringupParams {
    /// `role::DEVICE` slot (the xHCI `PciDevice`).
    pub device: SlotRef,
    /// `role::DMA_POOL` slot (the DMA-capable region).
    pub region: SlotRef,
    /// BAR to claim (0 for xHCI).
    pub bar: u8,
    /// VA + length for the BAR0 register window.
    pub regs_va: u64,
    /// BAR0 window length.
    pub regs_bytes: u64,
    /// DCBAA buffer placement.
    pub dcbaa: BufSpec,
    /// Command ring buffer placement.
    pub cmd_ring: BufSpec,
    /// Event ring buffer placement.
    pub event_ring: BufSpec,
    /// ERST buffer placement.
    pub erst: BufSpec,
}

/// **The whole xHCI controller bring-up in one crossing**: claim BAR0 →
/// map the register window → lease the DCBAA + command ring + event ring
/// + ERST with CPU + IOVA windows ([`dma_buffers`]) → establish the
/// interrupt route.  *15 invocations, 1 trap* — the register-level config
/// (CONFIG / CRCR / ERSTBA / doorbells) rides the mapped window afterward,
/// not the session.  (Scratchpad pages, when the controller wants them,
/// are a follow-on `dma_buffers` batch.)
pub fn xhci_bringup(
    b: &mut SessionBuilder,
    p: XhciBringupParams,
    bar: SlotRef,
    regs: SlotRef,
    buf_slots: &[SlotRef; 4 * INVS_PER_BUFFER],
    irq: SlotRef,
) -> Result<XhciBringup, BuildError> {
    let bar = b.claim_bar(p.device, Rights::ACCESS | Rights::MODIFY, p.bar, bar)?;
    let regs = b.share_cpu(
        bar,
        Rights::ACCESS | Rights::MODIFY,
        p.regs_va,
        p.regs_bytes,
        true,
        regs,
    )?;
    let ph = Staged(SlotRef::from_wire(0));
    let mut win = [DmaWindows {
        lease: ph,
        cpu: ph,
        iova: ph,
    }; 4];
    dma_buffers(
        b,
        p.region,
        p.device,
        &[p.dcbaa, p.cmd_ring, p.event_ring, p.erst],
        buf_slots,
        &mut win,
    )?;
    let irq = b.derive_irq_route(p.device, Rights::ACCESS, irq)?;
    Ok(XhciBringup {
        bar,
        regs,
        dcbaa: win[0],
        cmd_ring: win[1],
        event_ring: win[2],
        erst: win[3],
        irq,
    })
}

/// A per-device enumeration cluster's memory.
#[derive(Clone, Copy, Debug)]
pub struct XhciDevice {
    /// The context page (input + output device contexts packed).
    pub ctx: DmaWindows,
    /// The EP0 control transfer ring.
    pub ep0_ring: DmaWindows,
    /// The control-transfer data buffer (descriptors land here).
    pub data: DmaWindows,
}

/// **A device's enumeration memory in one crossing**: the packed context
/// page + the EP0 ring + the control data buffer ([`dma_buffers`]).  *9
/// invocations, 1 trap*, run once per device the driver addresses — the
/// batching that keeps enumeration cheap (and the reason the two contexts
/// share one page: `mm` rounds a lease to a page, so packing is one fewer
/// buffer *and* one fewer crossing).  The interrupt endpoint's ring +
/// report buffer, added at Configure Endpoint, are a follow-on batch.
pub fn xhci_device(
    b: &mut SessionBuilder,
    region: SlotRef,
    device: SlotRef,
    ctx: BufSpec,
    ep0_ring: BufSpec,
    data: BufSpec,
    buf_slots: &[SlotRef; 3 * INVS_PER_BUFFER],
) -> Result<XhciDevice, BuildError> {
    let ph = Staged(SlotRef::from_wire(0));
    let mut win = [DmaWindows {
        lease: ph,
        cpu: ph,
        iova: ph,
    }; 3];
    dma_buffers(
        b,
        region,
        device,
        &[ctx, ep0_ring, data],
        buf_slots,
        &mut win,
    )?;
    Ok(XhciDevice {
        ctx: win[0],
        ep0_ring: win[1],
        data: win[2],
    })
}
