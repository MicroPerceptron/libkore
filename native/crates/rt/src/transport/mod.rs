//! # `transport` — the transport seam (`res/USERLAND.md` §2/§4)
//!
//! Every effect a userland component asks of the kernel flows through
//! one boundary: **stage a session, commit it, read results back from
//! the caller's own arena result slots**.  [`Transport`] is that
//! boundary as a trait — the §4 drift-stopper pin — and
//! [`TrapTransport`] is its one implementation: `finish` into the
//! architecture's caller-saved GPR carrier when it fits, otherwise into the
//! task's arena descriptor scratch, then `commit_current(control, payload)`
//! through the arch trap gate — x86_64 `SYSCALL`, aarch64 `SVC`,
//! riscv64 `ECALL`, each a per-arch [`raw_trap`] shim over the same
//! contract (carrier control + the architecture's maximal caller-saved payload
//! bank in, packed status out).
//! **Live on the x86 userland target end-to-end** (first-boot M2b): a
//! real kore-rt blob in ring 3 stages a session and traps, the kernel's
//! live `commit_current` runs it.  The aarch64/riscv shims target their
//! kernels' V1 session landings (mechanism-complete, awaiting their
//! boot chains — #96/#298's aarch64/riscv siblings).  Off the metal
//! targets — host builds — every commit refuses with the typed
//! [`TrapUnavailable`], an honest runtime refusal, never a panic and
//! never a simulation.  (The mock transport that used to model commits
//! host-side was deleted by the 2026-07-24 ruling: transport behavior
//! is proven by the QEMU boot gates.)
//!
//! [`Committed`] — the typed status word — is re-exported from
//! [`kore_abi::session`], where its pack/unpack pair
//! ([`Status::encode`](kore_abi::session::Status::encode) /
//! [`Committed::status`]) live together beside the frozen layout.

use kore_abi::cap::{ResourceKind, Rights};
use kore_abi::device::DeviceSpecies;
use kore_abi::session::{InventoryReq, LeaseReq, SessionFlags, SlotRef, TrackedLeaseReq};

/// One standing receive post that may be folded into a parking trap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParkPost {
    /// The board channel id.
    pub chan: u32,
    /// The caller-owned landing slot.
    pub land: SlotRef,
    /// The live recv-end tag authenticating this post.
    pub recv_tag: u32,
}

use crate::service::{Grant, GrantArg};
use crate::session::{BuildError, SessionBuilder, Staged};

pub use kore_abi::session::{Committed, Status};

/// The transport seam (§2).  Stage against the builder; results
/// land in the caller's arena result slots, never in the return value.
///
/// Clients are generic over this seam (dyn-compatibility is not required), so
/// `async fn` in the trait is the pinned shape. Executors are single-threaded
/// per task (§3.3), so no auto-trait bound is owed.
#[allow(async_fn_in_trait)]
pub trait Transport {
    /// The transport-level failure — *could not commit* (e.g. the trap
    /// gate is not yet issued).  Per-invocation outcomes ride the
    /// [`Committed`] status word instead, exactly as the gate returns
    /// them. `Debug` so that every caller can report the failure it got
    /// rather than substitute a guess for it.
    type Error: core::fmt::Debug;

    /// Commit one bounded staged session; resolves when results land.
    /// `TrapTransport` finishes into the architecture's caller-saved GPR
    /// carrier when it fits, otherwise into arena descriptor scratch, then
    /// `commit_current(control, payload)` — WAIT bits clear (debug-asserted): a
    /// WAIT_* commit parks the whole task in-kernel, stalling every
    /// sibling on the executor.  The flagged form is the runtime-internal
    /// `commit_and_park`, reserved to callers that OWN their park: the
    /// executor's §3.1 idle / §3.4 deadline, §2.2's sync paths, and a
    /// relay task's WAIT_IRQ loop (§3.1a).
    async fn commit(&self, session: &SessionBuilder) -> Result<Committed, Self::Error>;

    /// Commit one bounded non-parking affine transition without placing its
    /// sole ownership witness on a cancellable future's stack.
    ///
    /// This is narrower than the ordinary async path: the descriptor must not
    /// carry WAIT flags, and callers use it only after every potentially
    /// blocking preflight has completed. The operation is then one complete
    /// transport boundary, allowing a state machine to change from retained
    /// local custody to correlated remote custody atomically with respect to
    /// Rust future cancellation. The default shares the synchronous trap
    /// implementation; transports that distinguish these paths may override
    /// it but must preserve the non-parking contract.
    fn commit_affine(&self, session: &SessionBuilder) -> Result<Committed, Self::Error> {
        debug_assert!(
            !session
                .flags()
                .intersects(SessionFlags::WAIT_DELIVERY | SessionFlags::WAIT_IRQ),
            "affine commit cannot carry WAIT bits"
        );
        self.commit_now(session)
    }

    /// §2.2's sync spine: commit with no executor and no WAIT flags —
    /// the global allocator's grow path (statically-bound transport) and
    /// pre-executor init only. `TrapTransport` issues exactly one trap.
    fn commit_now(&self, session: &SessionBuilder) -> Result<Committed, Self::Error>;
}

/// The trap transport's typed refusal: the syscall gate is not yet
/// issued from userland (the runtime link is Phase A's; first boot is
/// the #96/#298 boundary).  Every commit returns this until the trap
/// lands — fail-closed and visible, never `todo!`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrapUnavailable;

mod sealed {
    /// The [`ParkTransport`](super::ParkTransport) seal: parking commits
    /// are **runtime-internal** (`res/USERLAND.md` §4 — reserved to
    /// callers that OWN their park), so only the in-crate transport
    /// implements the flagged form.
    pub trait Sealed {}
    impl Sealed for super::TrapTransport {}
}

/// What a parking commit parks **on** — the executor's standing doorbell
/// post (`res/USERLAND.md` §3.1), plus the folded deadline when one is
/// pending (§3.4.4), plus the held IRQ route when the executor parks an
/// IRQ-owning service (the IRQ-or-doorbell park).  The staged `session`
/// already carries the same facts on the wire (the builder's park block);
/// the spec is the executor's transport-level description of that same
/// wait.
#[derive(Clone, Copy, Debug)]
pub struct ParkSpec {
    /// The doorbell channel (the standing post's board entry).
    pub chan: u32,
    /// The standing post's landing slot — where a delivered notice sits.
    pub land: SlotRef,
    /// Optional second standing post in the bounded receive-any form.
    pub alternate: Option<ParkPost>,
    /// The folded deadline (absolute monotonic ns), when a sleep is
    /// pending — the combined `WAIT_DELIVERY | WAIT_SLEEP` form.
    pub deadline_ns: Option<u64>,
    /// The held route rides the park (`WAIT_IRQ` in combination): the
    /// first of {delivery, reap, deadline, IRQ} wakes it, and an IRQ win
    /// answers [`Status::Irq`](kore_abi::session::Status::Irq) with the
    /// post left standing.
    pub irq: bool,
}

/// **`commit_and_park`** — the runtime-internal *flagged* commit
/// (`res/USERLAND.md` §4: the form [`Transport::commit`] debug-asserts
/// away): commit the staged session, then park until the standing post
/// delivers or the folded deadline expires.  The outcome rides the
/// [`Committed`] status word exactly as the gate returns it —
/// `Ok` = delivered, `Revoked` = the post was reaped,
/// `Deadline` = the deadline won (the post stands), `WouldBlock` = could
/// not park.
///
/// **Sealed** — reserved to callers that OWN their park: the executor's
/// §3.1 idle / §3.4 deadline, §2.2's sync paths, and a relay task's
/// WAIT_IRQ loop (§3.1a).
pub trait ParkTransport: Transport + sealed::Sealed {
    /// Commit `session`, then park per `spec`.  See the trait docs.
    fn commit_and_park(
        &self,
        session: &SessionBuilder,
        spec: &ParkSpec,
    ) -> Result<Committed, Self::Error>;

    /// Commit a [`SessionFlags::WAIT_IRQ`] session and park the dedicated relay
    /// task on its held interrupt route. No doorbell [`ParkSpec`] is accepted:
    /// the affine route in the task's holdings is the complete wait authority.
    ///
    /// A folded [`SessionFlags::WAIT_SLEEP`] deadline may ride (the
    /// **sleep-or-IRQ park**, `verbs::wait_interrupt_until`): the route
    /// stays the wait authority and the deadline only bounds the park,
    /// so the same "no doorbell" rule holds — a `WAIT_DELIVERY` arm is
    /// the executor's composite and belongs to
    /// [`commit_and_park`](Self::commit_and_park).
    fn commit_irq_park(&self, session: &SessionBuilder) -> Result<Committed, Self::Error>;

    /// Commit a [`SessionFlags::WAIT_TASK`] session and park on the join
    /// it staged — the **blocking join** (`kore_rt::thread::join`).
    ///
    /// Its own entry rather than a [`ParkSpec`] field because the wait
    /// authority is a *held capability* (the joined task's handle, named
    /// by the staged join derive), exactly as the route is for
    /// [`commit_irq_park`](Self::commit_irq_park) — there is nothing for
    /// a spec to describe that the wire does not already carry.  A
    /// doorbell, deadline, or route may ride alongside; the composite is
    /// resolved by the first-wins rule the flags document.
    fn commit_join_park(&self, session: &SessionBuilder) -> Result<Committed, Self::Error>;
}

impl ParkTransport for TrapTransport {
    /// **The live parking trap** (metal userland targets, bringup U2): `finish`
    /// the staged session — its `WAIT_*` flags ride the header `SessionFlags`
    /// and the `ParkReq`/`SleepReq` ride the args region, both **existing wire
    /// forms** the kernel `execute_for` decodes, so **no `DESCRIPTOR_ABI` bump**
    /// — into the GPR carrier when compact, otherwise arena scratch, and trap.
    /// It calls the shared trap primitive
    /// directly because [`issue`](TrapTransport::issue) is deliberately the
    /// non-parking contract and asserts that WAIT bits are clear. The kernel
    /// parks the ring-3 caller off its syscall stack (Model Y) and resumes it
    /// here with the WAIT status in the return register — decoded as
    /// [`Committed`]. `spec` mirrors facts already carried on the wire.
    /// (The persistent syscall-frame capture exists on x86-64 and aarch64;
    /// riscv64 remains at its non-parking V1 landing.)
    #[cfg(target_os = "none")]
    fn commit_and_park(
        &self,
        session: &SessionBuilder,
        _spec: &ParkSpec,
    ) -> Result<Committed, TrapUnavailable> {
        let _parked = crate::wake::Parked::enter();
        self.trap(session)
    }

    #[cfg(target_os = "none")]
    fn commit_irq_park(&self, session: &SessionBuilder) -> Result<Committed, TrapUnavailable> {
        debug_assert!(
            session.flags().contains(SessionFlags::WAIT_IRQ)
                && !session.flags().contains(SessionFlags::WAIT_DELIVERY),
            "IRQ park must carry WAIT_IRQ without a delivery arm"
        );
        // **No bell guard here — nothing is owed, and the old landmine is
        // now closed at its source.**
        //
        // This park belongs to dedicated route-owning workers (a relay thread,
        // a driver's pacing loop) whose bells no executor ever sets, so
        // `wake::Parked`'s `was_awake` would be false and the guard inert by
        // its own rule (see `wake::Parked`): there is no notice to protect.
        //
        // Historically the guard was also kept off here because adding it — and
        // to no other method — deterministically degraded the x86 boot gate to
        // 14/17 with `first-shell: FAIL (init rung 573)`, a `BadHandle` refusal
        // in exec's fs-peer assembly, for reasons then unexplained (bisected:
        // guard on `commit_and_park` alone PASS, + `commit_join_park` PASS, all
        // three FAIL). It is explained now: the guard's two atomic RMWs are a
        // pure *timing* shift, and exec's `provision_fs_peer` minted
        // `channel_space` channels without first reclaiming a just-finished
        // command's slots — so any scheduling perturbation could race it into a
        // spurious `TableFull`/`BadHandle` (`init rung 566`/`573`). QEMU's
        // deterministic timing hid it; real Lunar Lake metal, with the shipped
        // guard placement, did not (a 13/16 degraded boot at rung 566). That
        // race is now closed at its source — exec's `mint_fs_channel`
        // reclaims-and-retries before refusing (`services/exec/src/service/mod.rs`)
        // — so guard placement no longer bears on it. The guard stays off here
        // purely because nothing is owed.
        self.trap(session)
    }

    #[cfg(target_os = "none")]
    fn commit_join_park(&self, session: &SessionBuilder) -> Result<Committed, TrapUnavailable> {
        debug_assert!(
            session.flags().contains(SessionFlags::WAIT_TASK),
            "a join park must carry WAIT_TASK"
        );
        let _parked = crate::wake::Parked::enter();
        self.trap(session)
    }

    /// Off the metal userland targets (host tests): the honest typed refusal —
    /// same posture as the plain verbs, minus their WAIT-bits-clear assertion
    /// (this IS the flagged form).
    #[cfg(not(target_os = "none"))]
    fn commit_and_park(
        &self,
        _session: &SessionBuilder,
        _spec: &ParkSpec,
    ) -> Result<Committed, TrapUnavailable> {
        Err(TrapUnavailable)
    }

    #[cfg(not(target_os = "none"))]
    fn commit_irq_park(&self, _session: &SessionBuilder) -> Result<Committed, TrapUnavailable> {
        Err(TrapUnavailable)
    }

    #[cfg(not(target_os = "none"))]
    fn commit_join_park(&self, _session: &SessionBuilder) -> Result<Committed, TrapUnavailable> {
        Err(TrapUnavailable)
    }
}

// The task's descriptor-scratch and cap-slot-alias VAs were fixed constants
// here (`M2B_SCRATCH_VA` = 0x6000_0000, `SLOTS_VA` = 0x7000_0000), pinned to the
// kernel founders' mappings only by comment.  They are now founder-carried
// geometry: the founder names each window in the task's `.kore.geom` boot block,
// crt0 installs it (`crate::entry::geometry`), and the trap transport + the
// possession verbs read it there — no cross-boundary constant to drift
// (`res/USERLAND.md` §4).

/// The **trap transport** — the real path's transport (§2.1): one
/// commit = `finish` the staged descriptor into caller-saved registers when it
/// fits, otherwise into arena scratch, plus `commit_current(control, payload)`
/// through the gate. It compiles with the runtime unconditionally. On the
/// **metal userland targets** the trap is live: trap through
/// the per-arch [`raw_trap`] shim (`syscall` / `svc` / `ecall`), decode
/// the returned register as [`Committed`].  Off the metal targets —
/// host tests — both verbs refuse [`TrapUnavailable`], documented
/// honestly rather than stubbed with a panic.
pub struct TrapTransport;

impl TrapTransport {
    /// A trap-bound transport handle (zero-sized; `const` so a
    /// `#[global_allocator]`-adjacent static can bind it).
    pub const fn new() -> Self {
        Self
    }

    /// Guard the plain-commit contract, then issue.  A `WAIT_*` commit parks
    /// the whole task in-kernel (stalling every sibling on the executor), so
    /// the parking form is reserved to [`ParkTransport::commit_and_park`]; a
    /// plain verb debug-asserts the bits clear (`res/USERLAND.md` §4).
    fn issue(&self, session: &SessionBuilder) -> Result<Committed, TrapUnavailable> {
        debug_assert!(
            !session
                .flags()
                .intersects(SessionFlags::WAIT_DELIVERY | SessionFlags::WAIT_IRQ),
            "plain commit with WAIT bits set — parking commits are commit_and_park's (Phase A)"
        );
        self.trap(session)
    }

    /// **The live trap** (metal userland targets): `finish` the staged
    /// descriptor into the architecture's caller-saved GPR carrier when it
    /// fits, otherwise into this task's founder-mapped descriptor scratch,
    /// then trap. The kernel's live `commit_current` runs the session against
    /// the caller's own arena/space (I7) and returns the packed status word in
    /// the architecture's return register, decoded here as [`Committed`].
    #[cfg(target_os = "none")]
    fn trap(&self, session: &SessionBuilder) -> Result<Committed, TrapUnavailable> {
        trap_session(session, None)
    }

    /// Off the metal userland targets — host builds — the trap is not issued:
    /// the honest typed refusal (`res/USERLAND.md` §2.1), never a panic.
    #[cfg(not(target_os = "none"))]
    fn trap(&self, _session: &SessionBuilder) -> Result<Committed, TrapUnavailable> {
        Err(TrapUnavailable)
    }
}

/// Number of descriptor payload words carried in caller-saved GPRs by this
/// architecture's syscall ABI.  These are the maximal useful banks that do
/// not borrow callee-saved state (and therefore do not add spill/restore
/// traffic merely to avoid the descriptor copy):
///
/// - x86-64: `rsi, rdx, r8, r9, r10, rax` (48 bytes; `rcx/r11` are consumed by
///   `SYSCALL`),
/// - aarch64: `x1..=x17` (136 bytes; `x18` remains the platform register),
/// - riscv64: `a1..=a7, t0..=t6` (112 bytes; `ra` remains the return link).
#[cfg(all(target_arch = "x86_64", target_os = "none"))]
const INLINE_WORDS: usize = 6;
#[cfg(all(target_arch = "aarch64", target_os = "none"))]
const INLINE_WORDS: usize = 17;
#[cfg(all(target_arch = "riscv64", target_os = "none"))]
const INLINE_WORDS: usize = 14;

/// Finish and issue one session, preferring the zero-shared-memory GPR
/// carrier. `scratch_hint` exists for crt0's pre-thread-pointer SET_TP call;
/// ordinary anchored callers resolve the founder-stamped scratch lazily only
/// when the descriptor is too large for registers.
#[cfg(target_os = "none")]
pub(crate) fn trap_session(
    session: &SessionBuilder,
    scratch_hint: Option<usize>,
) -> Result<Committed, TrapUnavailable> {
    let len = session.encoded_len().map_err(|_| TrapUnavailable)?;
    let mut payload = [0u64; INLINE_WORDS];
    if len <= core::mem::size_of_val(&payload) {
        // SAFETY: `u64` has no invalid bit patterns; this is the exact mutable
        // byte view of the local register image, whose alignment is stronger
        // than `u8` requires and whose lifetime ends after the trap.
        let out = unsafe {
            core::slice::from_raw_parts_mut(
                payload.as_mut_ptr().cast::<u8>(),
                core::mem::size_of_val(&payload),
            )
        };
        let written = session.finish(out).map_err(|_| TrapUnavailable)?;
        debug_assert_eq!(written, len);
        let control = kore_abi::session::inline_descriptor_control(len)
            .expect("architecture GPR capacity exceeded inline length carrier");
        // SAFETY: `payload[..len]` contains the just-finished descriptor and
        // `control` encodes that exact initialized prefix.
        return Ok(Committed::from_raw(unsafe { raw_trap(control, &payload) }));
    }

    // The founder mapped the descriptor scratch user-writable and named its
    // VA in the task's `.kore.geom` block.  Resolve it only on this fallback:
    // most steady-state sessions never touch the shared page at all.
    let scratch = scratch_hint
        .or_else(crate::entry::geometry::scratch_va)
        .ok_or(TrapUnavailable)?;
    // SAFETY: the founder mapped `[scratch, +DESCRIPTOR_MAX)` user-writable,
    // aliasing this task's arena descriptor scratch; staging owns this byte
    // run until the synchronous trap snapshots it.
    let out = unsafe {
        core::slice::from_raw_parts_mut(scratch as *mut u8, crate::session::DESCRIPTOR_MAX)
    };
    let written = session.finish(out).map_err(|_| TrapUnavailable)?;
    payload[0] = written as u64;
    // SAFETY: control zero names scratch offset zero and payload word zero
    // carries the exact initialized descriptor length.
    Ok(Committed::from_raw(unsafe { raw_trap(0, &payload) }))
}

/// **The raw `SYSCALL`** (x86 userland target): issue the trap with a
/// descriptor carrier control plus the maximal caller-saved GPR payload bank,
/// returning the packed status word.  The kernel trampoline
/// (`src/platform/arch/amd64/processor/syscall.rs`) switches to its own
/// stack (`rsp` untouched), preserves the callee-saved set, and restores
/// `rcx`/`r11` for `sysretq` — so the call clobbers the C caller-saved
/// set, with `rcx`/`r11` named explicitly.  Any prior store of the
/// descriptor bytes is ordered before the trap (the asm has no `nomem`,
/// so it is a memory barrier).
///
/// Shared by [`TrapTransport::trap`] (which resolves the scratch through
/// the thread-pointer-anchored geometry) and crt0's thread anchor
/// (`entry::anchor_boot_thread` — which stages *before* the TP is live,
/// against a scratch VA it parsed from the boot page itself).
///
/// # Safety
/// `control` must describe the initialized prefix in `payload`, or name a
/// finished descriptor in this task's live scratch. The kernel validates every
/// byte, so garbage refuses rather than faults; the unsafe here is only the
/// inline asm itself.
#[cfg(all(target_arch = "x86_64", target_os = "none"))]
pub(crate) unsafe fn raw_trap(control: u64, payload: &[u64; INLINE_WORDS]) -> u64 {
    let mut status = payload[5];
    // SAFETY: caller contract above.
    unsafe {
        core::arch::asm!(
            "syscall",
            in("rdi") control,
            in("rsi") payload[0],
            in("rdx") payload[1],
            in("r8") payload[2],
            in("r9") payload[3],
            in("r10") payload[4],
            inlateout("rax") status,
            lateout("rcx") _,
            lateout("r11") _,
            clobber_abi("C"),
            options(nostack),
        );
    }
    status
}

/// **The raw `SVC`** (aarch64 userland target) — the x86 shim's sibling,
/// same contract: carrier control in `x0`, payload in `x1..=x17`, and the
/// packed status word back in `x0`. The kernel's shared
/// `VBAR_EL1` table routes the Lower-EL synchronous class to the session
/// gate (`src/platform/arch/arm64/processor/trap.rs`), which decodes the
/// `SVC` class and funnels to `commit_current`.  The kernel's full-frame
/// trap path happens to restore every GPR, but the *contract* stays the
/// pinned gate one — caller-saved registers are dead across the gate —
/// so `clobber_abi("C")` here keeps callers off the stronger incidental
/// behavior.  Any prior store of the descriptor bytes is ordered before
/// the trap (no `nomem`, so the asm is a memory barrier).
///
/// # Safety
/// As the x86 shim: a well-formed register carrier or live scratch descriptor;
/// the kernel validates every byte.
#[cfg(all(target_arch = "aarch64", target_os = "none"))]
pub(crate) unsafe fn raw_trap(control: u64, payload: &[u64; INLINE_WORDS]) -> u64 {
    let status: u64;
    // SAFETY: caller contract above.
    unsafe {
        core::arch::asm!(
            "svc #0",
            inlateout("x0") control => status,
            in("x1") payload[0],
            in("x2") payload[1],
            in("x3") payload[2],
            in("x4") payload[3],
            in("x5") payload[4],
            in("x6") payload[5],
            in("x7") payload[6],
            in("x8") payload[7],
            in("x9") payload[8],
            in("x10") payload[9],
            in("x11") payload[10],
            in("x12") payload[11],
            in("x13") payload[12],
            in("x14") payload[13],
            in("x15") payload[14],
            in("x16") payload[15],
            in("x17") payload[16],
            clobber_abi("C"),
            options(nostack),
        );
    }
    status
}

/// **The raw `ECALL`** (riscv64 userland target) — the x86 shim's
/// sibling, same contract: carrier control in `a0`, payload in
/// `a1..=a7,t0..=t6`, and the packed status word back in `a0`. The
/// kernel's `stvec` trampoline
/// (`src/platform/arch/riscv64/processor/syscall.rs`) decodes the
/// U-mode environment call, funnels to `commit_current`, and preserves
/// only `sp`/`tp` plus the ABI callee-saved set — exactly the pinned
/// gate contract `clobber_abi("C")` declares.  Any prior store of the
/// descriptor bytes is ordered before the trap (no `nomem`, so the asm
/// is a memory barrier).
///
/// # Safety
/// As the x86 shim: a well-formed register carrier or live scratch descriptor;
/// the kernel validates every byte.
#[cfg(all(target_arch = "riscv64", target_os = "none"))]
pub(crate) unsafe fn raw_trap(control: u64, payload: &[u64; INLINE_WORDS]) -> u64 {
    let status: u64;
    // SAFETY: caller contract above.
    unsafe {
        core::arch::asm!(
            "ecall",
            inlateout("a0") control => status,
            in("a1") payload[0],
            in("a2") payload[1],
            in("a3") payload[2],
            in("a4") payload[3],
            in("a5") payload[4],
            in("a6") payload[5],
            in("a7") payload[6],
            in("t0") payload[7],
            in("t1") payload[8],
            in("t2") payload[9],
            in("t3") payload[10],
            in("t4") payload[11],
            in("t5") payload[12],
            in("t6") payload[13],
            clobber_abi("C"),
            options(nostack),
        );
    }
    status
}

impl Default for TrapTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl Transport for TrapTransport {
    type Error = TrapUnavailable;

    async fn commit(&self, session: &SessionBuilder) -> Result<Committed, TrapUnavailable> {
        // No executor/park in M2b — the async spine delegates to the sync trap
        // (Ready on first poll, the §3.3 shape).
        self.issue(session)
    }

    fn commit_now(&self, session: &SessionBuilder) -> Result<Committed, TrapUnavailable> {
        self.issue(session)
    }
}

/// Byte length of a staged [`GrantArg::Window`] payload:
/// `offset u64 ‖ len u64 ‖ class u8 ‖ zeros ×7`, LE — 24 bytes.
const WINDOW_ARG: usize = 24;

/// Lower one serve-step [`Grant`] to its staged derive — the wire form
/// of [`Serve::grant`](crate::service::Serve::grant)'s artifact (the
/// framework's contract: "the real dispatch stages it as a `Derive`").
/// Window and sublease grants ride the argument region as the 24-byte carve
/// payload above; [`GrantArg::None`] stages a bare attenuation.
/// `result` names the caller's own arena scratch slot the committed child
/// lands in.
///
/// **The window carve's kind + rights** (ckG2a-2, the F2 alignment).  A
/// `GrantArg::Window` stages `kind = LeasedRegion`: the kernel's `(op ×
/// kind)` disambiguator selects the **detached window carve** off the
/// held supply lease (`derive_from_held_lease`) — a geometry-only
/// [`ResourceKind::LeasedRegion`] view, the only kind a `MemLease` admits
/// (a `PhysExtent` sub-carve would be a second release authority over the
/// same frames).  The staged child is therefore a `SessionChild::Leased`,
/// which the following `Send` **moves**.
///
/// [`Grant::rights`] are the **delivered** rights the client receives; the
/// scratch derive additionally carries [`Rights::DERIVE`], which the
/// kernel's transfer gate demands of any capability presented to a
/// rendezvous send (`prepare.rs`: "re-derivation is a derive of the
/// transferred cap"). That extra right never reaches the peer — the send
/// arm clamps the moved cap to the send's `requested` rights, stripping
/// the inert `DERIVE` (a `LeasedRegion` has no outbound attenuate edge).
pub fn stage_grant(
    b: &mut SessionBuilder,
    grant: &Grant,
    result: SlotRef,
) -> Result<Staged, BuildError> {
    match grant.arg {
        GrantArg::Direct | GrantArg::Exact => Err(BuildError::BadDeliverySource),
        GrantArg::None => b.derive_raw(grant.subject, grant.rights, 0, result, None),
        GrantArg::Attenuate { kind } => b.derive_raw(
            grant.subject,
            grant.rights | Rights::DERIVE,
            kind as u8,
            result,
            None,
        ),
        GrantArg::Window { offset, len, class } => {
            let mut req = [0u8; WINDOW_ARG];
            req[0..8].copy_from_slice(&offset.to_le_bytes());
            req[8..16].copy_from_slice(&len.to_le_bytes());
            req[16] = class as u8;
            b.derive_raw(
                grant.subject,
                grant.rights | Rights::DERIVE,
                ResourceKind::LeasedRegion as u8,
                result,
                Some(&req),
            )
        }
        GrantArg::Sublease { offset, len } => {
            let mut req = [0u8; WINDOW_ARG];
            req[0..8].copy_from_slice(&offset.to_le_bytes());
            req[8..16].copy_from_slice(&len.to_le_bytes());
            b.derive_raw(
                grant.subject,
                grant.rights | Rights::DERIVE,
                ResourceKind::MemLease as u8,
                result,
                Some(&req),
            )
        }
        // **A device hand-out** (`res/DEVICE_SERVICE.md` §10): stage the
        // per-device derive off the broker's inventory into scratch, for
        // the following `Send` to move into the claimant's posted slot.
        //
        // Same shape as the window carve, including the extra `DERIVE` the
        // kernel's transfer gate demands of any capability presented to a
        // rendezvous send.  Unlike a `LeasedRegion`, a device *does* have
        // outbound attenuate edges — BARs, routes, IOVA windows — so the
        // send's `requested` rights are what decide whether the driver can
        // use them; the clamp is the grant's declared rights, exactly as
        // for a window.
        GrantArg::Device { index, species } => {
            let mut req = [0u8; InventoryReq::WIRE];
            req[0..4].copy_from_slice(&index.to_le_bytes());
            let kind = match species {
                DeviceSpecies::Pci => ResourceKind::PciDevice,
                DeviceSpecies::Platform => ResourceKind::PlatformDevice,
            };
            b.derive_raw(
                grant.subject,
                grant.rights | Rights::DERIVE,
                kind as u8,
                result,
                Some(&req),
            )
        }
        // **A fresh root supply lease** off the granting service's region
        // endowment: the [`SessionBuilder::lease_aligned`] wire form
        // (`Derive × PhysRegion → MemLease`, kind 0 + a `LeaseReq` arg),
        // staged into scratch for the following `Send` to move.  The
        // extra `DERIVE` is the transfer gate's toll, exactly as above;
        // the send clamps the moved root to the grant's declared rights.
        GrantArg::Lease {
            size,
            align_log2,
            flags,
            release,
        } => {
            if let Some(release) = release {
                let mut req = [0u8; TrackedLeaseReq::WIRE];
                req[0..8].copy_from_slice(&size.to_le_bytes());
                req[8] = flags | kore_abi::session::LEASE_TRACK_RELEASE;
                req[9] = align_log2;
                req[16..24].copy_from_slice(&release.receipt.to_le_bytes());
                req[24..26].copy_from_slice(&release.pair.to_le_bytes());
                b.derive_raw(
                    grant.subject,
                    grant.rights | Rights::DERIVE,
                    0,
                    result,
                    Some(&req),
                )
            } else {
                let mut req = [0u8; LeaseReq::WIRE];
                req[0..8].copy_from_slice(&size.to_le_bytes());
                req[8] = flags;
                req[9] = align_log2;
                b.derive_raw(
                    grant.subject,
                    grant.rights | Rights::DERIVE,
                    0,
                    result,
                    Some(&req),
                )
            }
        }
        // **A sub-lease carve** off the held subject (slice C: a minted
        // face's ring client sub) — `carve_sublease`'s wire form, staged
        // into scratch for the send to move whole; the same transfer-gate
        // `DERIVE` toll and delivered-rights clamp as every arm above.
        GrantArg::Sub { offset, len } => b.carve_sublease(
            grant.subject,
            grant.rights | Rights::DERIVE,
            offset,
            len,
            result,
        ),
        // **A fresh channel recv end** off the held subject root (slice
        // C: a minted face's grant-bank lane) — `derive_chan_recv`'s
        // form; the root stays the service's revocation pivot.
        GrantArg::ChanRecv => {
            b.derive_chan_recv(grant.subject, grant.rights | Rights::DERIVE, result)
        }
        GrantArg::ChanSend => {
            b.derive_chan_send(grant.subject, grant.rights | Rights::DERIVE, result)
        }
    }
}
