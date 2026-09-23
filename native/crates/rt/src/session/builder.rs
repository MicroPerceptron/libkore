//! `builder` — the zero-alloc staged-session writer: [`SessionBuilder`],
//! its SSA [`Staged`] refs, and the frozen-descriptor `finish`.

use kore_abi::cap::{ResourceKind, Rights};
use kore_abi::device::DeviceSpecies;
use kore_abi::session::{
    DERIVE_KIND_DEVICE_RECORD, DERIVE_KIND_GUEST_FOUND, DERIVE_KIND_I2C_HID_RECORD,
    DERIVE_KIND_IOVA_ANON, DERIVE_KIND_IRQ_ROUTE_MSIX, DERIVE_KIND_PEER_IOVA, DESCRIPTOR_ABI,
    FoundReq, GuestFoundReq, I2C_HID_QUERY_COUNT, I2cHidReq, INVENTORY_QUERY_COUNT,
    IRQ_AFFINITY_ANY, InventoryReq, Invocation, IrqReq, MAX_FOUND_DELIVERIES, MAX_INVOCATIONS,
    OpCode, SessionFlags, SlotRef,
};
use kore_abi::task::{SpawnFlags, SpawnReq};

/// `Found`s one descriptor can carry — the ABI's [`MAX_SESSION_FOUNDS`]
/// (the shared wire-limit vocabulary, beside `MAX_INVOCATIONS`),
/// re-exported where the builder's other ceilings live.  A gang wider
/// than this chunks into `ceil(w / 12)` sessions (`spawn_spmd`), never
/// refusing a launch.
pub use kore_abi::session::MAX_SESSION_FOUNDS;

/// Argument-region capacity inside the frozen two-page descriptor window.
/// This legacy price admits both complete useful extremes: all
/// `MAX_INVOCATIONS` 40-byte peer-aperture requests (4,280 B), or the complete
/// 12-`FoundReq` founding batch plus the former 24-byte repeatable tail.
/// A larger mixed Found+peer batch stops at `ArgsOverflow` and must be split;
/// it never grows the task's bounded scratch or truncates a request.
const ARGS_MAX: usize = MAX_INVOCATIONS * 24 + MAX_SESSION_FOUNDS * (FoundReq::WIRE - 24);

/// Descriptor bytes a full session needs — the buffer floor for
/// [`SessionBuilder::finish`].  The `+ 32` is the park block ceiling that
/// can accompany invocations (`ParkReq` 16 + `SleepReq` 8 + `IrqParkReq` 8). The larger
/// two-post block is restricted to an empty session and is only 48 bytes
/// including the header, so it does not raise this full-descriptor bound.
pub const DESCRIPTOR_MAX: usize = 8 + MAX_INVOCATIONS * 16 + 32 + ARGS_MAX;

// A full descriptor must fit the two-page `[layout]` scratch window
// (`scratch = { bytes = 0x2000 }` in res/boot.toml), priced with
// `MAX_INVOCATIONS = 107`.  A ceiling bump that bursts the window is
// a build error here, never a truncated descriptor at stage time.
const _: () = assert!(DESCRIPTOR_MAX <= 0x2000);

/// Why staging refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuildError {
    /// The session is at [`MAX_INVOCATIONS`] (I9: sessions are bounded —
    /// split the batch, don't grow it).
    TooManyInvocations,
    /// The argument region is full.
    ArgsOverflow,
    /// A post-commit effect whose retry outcome must describe the whole
    /// session was staged after another invocation or with park flags.
    /// Rebuild it as a one-invocation session.
    EffectMustStandAlone,
    /// The output buffer cannot hold the finished descriptor.
    OutTooSmall,
    /// A [`flows::dma_buffers`](super::flows::dma_buffers) batch (with its fixed legs) would exceed
    /// [`MAX_INVOCATIONS`], or the caller's result / output slices are too
    /// short for the requested buffer / delegation count — split the batch.
    TooManyBuffers,
    /// The spawn flow's flag contract is violated:
    /// [`flows::spawn_process`](super::flows::spawn_process) refuses [`SpawnFlags::SHARE_SPACE`] — a
    /// shared-space spawn is the thread case, paired with a thread-view
    /// space (`flows::spawn_thread`), not a fresh mint.
    BadSpawnFlags,
    /// A founding delivery map exceeds
    /// [`MAX_FOUND_DELIVERIES`](kore_abi::session::MAX_FOUND_DELIVERIES)
    /// (the bounded U1 map — shrink the wiring, never a stream).
    TooManyDeliveries,
    /// The session already carries [`MAX_SESSION_FOUNDS`] `Found`s — the
    /// descriptor's exact founding capacity.  The remedy is the gang
    /// chunking `spawn_spmd` performs: split the launch across sessions,
    /// never grow the descriptor.
    TooManyFounds,
    /// A held founding-delivery source does not fit the reserved 15-bit
    /// arena-slot encoding.
    BadDeliverySource,
    /// The bounded standing-post wait is malformed: an alternate post lacks
    /// its primary, aliases the same channel, or accompanies invocations.
    BadParkShape,
    /// TASK_EXIT requires the typed [`SessionBuilder::exit`] entry point.
    UntypedExit,
}

/// A staged invocation's **result** — the typed SSA reference (`Move D`:
/// a session is a small closed program over arena slots).  Use it as a
/// later invocation's subject to chain on a capability that will only
/// exist once the session commits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Staged(pub(super) SlotRef);

impl Staged {
    /// The arena slot the committed result lands in.
    pub fn slot(self) -> SlotRef {
        self.0
    }
}

impl From<Staged> for SlotRef {
    fn from(s: Staged) -> SlotRef {
        s.0
    }
}

/// One founding **delivery** (descriptor ABI v3 — U1): place the session
/// child staged into `from` (its SSA result) into the founded child's
/// arena at `to_slot`.  Consumed by
/// [`found_deliver`](SessionBuilder::found_deliver) /
/// [`flows::spawn_process`](super::flows::spawn_process); the kernel's
/// spawn assembly seals the cap to the child's holder (D1) and adopts its
/// twin into the child's holdings — a *move*, the founder's slot scrubbed.
#[derive(Clone, Copy, Debug)]
pub struct Deliver {
    /// The staged child to deliver — its SSA result slot (a [`Staged`]
    /// converts via `Into<SlotRef>`; a raw `SlotRef` names a result whose
    /// `Staged` the caller does not hold, e.g. a flow-internal share).
    pub from: SlotRef,
    /// Destination slot in the **child's** arena (`< arena_slots`).
    pub to_slot: u16,
}

impl Deliver {
    /// A delivery of the child staged at `from` into child slot `to_slot`.
    pub fn new(from: impl Into<SlotRef>, to_slot: u16) -> Self {
        Self {
            from: from.into(),
            to_slot,
        }
    }

    /// Move one capability already held by the founder into the child.
    ///
    /// Unlike [`new`](Self::new), this preserves the exact affine lineage and
    /// rights instead of naming a same-session derive.  It is useful when a
    /// child must become the next lessor/owner and another attenuation would
    /// violate a resource's bounded delegation depth.
    pub fn held(from: SlotRef, to_slot: u16) -> Result<Self, BuildError> {
        let slot = u16::try_from(from.index()).map_err(|_| BuildError::BadDeliverySource)?;
        let encoded =
            kore_abi::session::deliver_from_held(slot).ok_or(BuildError::BadDeliverySource)?;
        Ok(Self {
            from: SlotRef::from_wire(encoded as u32),
            to_slot,
        })
    }

    /// Move a staged child into the founded task's private holdings without
    /// publishing a child-arena record for it.
    pub fn private(from: impl Into<SlotRef>) -> Self {
        Self::new(from, kore_abi::session::DELIVER_TO_HOLDINGS)
    }

    /// Move one founder-held capability into the founded task's private
    /// holdings without publishing a child-arena record for it.
    pub fn held_private(from: SlotRef) -> Result<Self, BuildError> {
        Self::held(from, kore_abi::session::DELIVER_TO_HOLDINGS)
    }

    /// Deliver the founded task its **own** handle
    /// ([`DELIVER_FROM_SELF_TASK`]) — the SELF capability it presents to
    /// exit itself.
    ///
    /// The one delivery whose source is not a staged child: the handle is
    /// minted from the child's own root after the run-table intake, which
    /// is past everything this session can name.  It rides the delivery
    /// map anyway, so a founding still describes its child's whole
    /// starting cross-section in one place.
    ///
    /// [`DELIVER_FROM_SELF_TASK`]: kore_abi::session::DELIVER_FROM_SELF_TASK
    pub fn self_task(to_slot: u16) -> Self {
        Self {
            from: SlotRef::from_wire(u32::from(kore_abi::session::DELIVER_FROM_SELF_TASK)),
            to_slot,
        }
    }
}

/// One staged invocation (the builder's in-memory form; the wire form is
/// [`kore_abi::session::InvocationWire`], written LE at
/// [`finish`](SessionBuilder::finish)).  Builder-internal storage — the
/// kernel reads only the finished wire bytes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Inv {
    pub(crate) op: u16,
    pub(crate) requested: u8,
    pub(crate) kind: u8,
    pub(crate) subject: u32,
    pub(crate) result: u32,
    pub(crate) arg: u32,
}

const NO_INV: Inv = Inv {
    op: 0,
    requested: 0,
    kind: 0,
    subject: 0,
    result: 0,
    arg: Invocation::NO_ARG,
};

/// The staged-session builder: accumulate invocations + argument payloads,
/// then [`finish`](Self::finish) the descriptor bytes the kernel decode
/// reads.  Inline fixed-capacity — no allocation anywhere.
pub struct SessionBuilder {
    invs: [Inv; MAX_INVOCATIONS],
    count: usize,
    args: [u8; ARGS_MAX],
    args_len: usize,
    flags: SessionFlags,
    /// Whether a **standalone post-commit effect** (START / ARENA_EXTEND —
    /// the kernel's sole-invocation, unflagged rule) has been staged.  Kept
    /// in the builder's structural state so every later push rejects in
    /// O(1); the kernel still rechecks the finished descriptor at the trust
    /// boundary.
    has_effect: bool,
    /// Park-on-standing-post target ([`park_standing`](Self::park_standing))
    /// — emitted as the park block's `ParkReq` by [`finish`](Self::finish).
    park: Option<(u32, u16, u32)>,
    /// Optional second standing-post arm. It is valid only beside `park` on
    /// an empty session and emits the second adjacent `ParkReq`.
    park_alt: Option<(u32, u16, u32)>,
    /// Deadline park ([`wait_sleep`](Self::wait_sleep)) — emitted as the
    /// park block's `SleepReq` by [`finish`](Self::finish).
    sleep: Option<u64>,
    /// Exact held route, emitted after the other park payloads.
    irq: Option<SlotRef>,
    /// `Found`s staged so far — refused past [`MAX_SESSION_FOUNDS`], the
    /// argument region's exact founding capacity.
    founds: usize,
}

impl SessionBuilder {
    /// An empty session.
    pub const fn new() -> Self {
        Self {
            invs: [NO_INV; MAX_INVOCATIONS],
            count: 0,
            args: [0u8; ARGS_MAX],
            args_len: 0,
            flags: SessionFlags::empty(),
            has_effect: false,
            park: None,
            park_alt: None,
            sleep: None,
            irq: None,
            founds: 0,
        }
    }

    /// Invocations staged so far.
    pub fn staged(&self) -> usize {
        self.count
    }

    /// Whether the descriptor tail beginning at `checkpoint` is a non-empty,
    /// unflagged derivation chain whose final result is `staged`.
    ///
    /// Return-channel clients first confine caller code behind their
    /// `ReturnDeriver` facade, which exposes only approved capability-producing
    /// verbs. This is the second, structural check: the resulting tail is
    /// non-empty, unflagged, contains no non-`Derive` opcode, and returns its
    /// last product. The facade is what excludes effectful flavors that share
    /// the multiplexed `Derive` opcode; opcode inspection alone is not that
    /// semantic proof.
    pub(crate) fn is_derivation_tail_since(&self, checkpoint: usize, staged: Staged) -> bool {
        if checkpoint >= self.count || !self.flags.is_empty() {
            return false;
        }
        let tail = &self.invs[checkpoint..self.count];
        tail.iter().all(|inv| inv.op == OpCode::Derive as u16)
            && tail
                .last()
                .is_some_and(|inv| inv.result == staged.slot().index())
    }

    /// The session-level flags staged so far ([`SessionFlags`]) — what a
    /// transport inspects to enforce the WAIT-bits-clear contract
    /// (`res/USERLAND.md` §4: a plain commit must not park).
    pub fn flags(&self) -> SessionFlags {
        self.flags
    }

    /// Park after commit until the session's single posted recv delivers
    /// ([`SessionFlags::WAIT_DELIVERY`], #71 tier 1).
    pub fn wait_delivery(&mut self) -> &mut Self {
        self.flags |= SessionFlags::WAIT_DELIVERY;
        self
    }

    /// Park after commit on the held `IrqRoute` until the interrupt fires
    /// ([`SessionFlags::WAIT_IRQ`], #86b — the driver loop's park).
    pub fn wait_irq(&mut self) -> &mut Self {
        self.flags |= SessionFlags::WAIT_IRQ;
        self
    }

    /// Park on this exact held route, including when the task owns several.
    pub fn wait_irq_on(&mut self, route: SlotRef) -> &mut Self {
        self.flags |= SessionFlags::WAIT_IRQ | SessionFlags::WAIT_IRQ_SELECTED;
        self.irq = Some(route);
        self
    }

    /// Park after commit on the session's staged **join** until the joined
    /// task exits or is reaped ([`SessionFlags::WAIT_TASK`]) — the
    /// blocking join.  Without it a staged [`join`](Self::join) is a
    /// non-blocking probe.
    pub fn wait_task(&mut self) -> &mut Self {
        self.flags |= SessionFlags::WAIT_TASK;
        self
    }

    /// `Derive × Task × TASK_EXIT` — **end the calling task** with
    /// `status`, presenting its own `role::SELF` handle (`MODIFY`).  The
    /// commit does not return.
    pub fn exit(
        &mut self,
        self_handle: SlotRef,
        status: crate::exit::ExitCode,
    ) -> Result<(), BuildError> {
        let mut req = [0u8; 8];
        req[0..4].copy_from_slice(&status.get().to_le_bytes());
        self.derive_inner(
            self_handle,
            Rights::empty(),
            kore_abi::session::DERIVE_KIND_TASK_EXIT,
            // An effect mints nothing, so the result slot is inert; name
            // the subject's own slot, which is never written.
            self_handle,
            Some(&req),
        )
        .map(|_| ())
    }

    /// Report the caller's fatal panic with a bounded UTF-8 message.
    pub fn report_panic(&mut self, task: SlotRef, message: &str) -> Result<(), BuildError> {
        self.report_panic_with_truncation(task, message, false)
    }

    pub(crate) fn report_panic_with_truncation(
        &mut self,
        task: SlotRef,
        message: &str,
        truncated: bool,
    ) -> Result<(), BuildError> {
        if self.count != 0 || !self.flags.is_empty() {
            return Err(BuildError::EffectMustStandAlone);
        }
        let mut bytes = [0u8; 8 + kore_abi::crash::MESSAGE_BYTES];
        let mut len = message.len().min(kore_abi::crash::MESSAGE_BYTES);
        while !message.is_char_boundary(len) {
            len -= 1;
        }
        bytes[..4].copy_from_slice(&(len as u32).to_le_bytes());
        bytes[4] = u8::from(truncated || len != message.len());
        bytes[8..8 + len].copy_from_slice(&message.as_bytes()[..len]);
        self.derive_raw(
            task,
            Rights::empty(),
            kore_abi::session::DERIVE_KIND_PANIC_REPORT,
            task,
            Some(&bytes[..8 + len]),
        )
        .map(|_| ())
    }

    /// Read a task's opaque public identity through an ACCESS handle.
    pub fn task_identity(&mut self, task: SlotRef, result: SlotRef) -> Result<(), BuildError> {
        if self.count != 0 || !self.flags.is_empty() {
            return Err(BuildError::EffectMustStandAlone);
        }
        self.derive_raw(
            task,
            Rights::empty(),
            kore_abi::session::DERIVE_KIND_TASK_IDENTITY,
            result,
            None,
        )
        .map(|_| ())
    }

    /// Name a dormant task once. Names are copied into kernel-owned storage.
    pub fn name_task(
        &mut self,
        task: SlotRef,
        name: kore_abi::task::TaskName,
    ) -> Result<(), BuildError> {
        if self.count != 0 || !self.flags.is_empty() {
            return Err(BuildError::EffectMustStandAlone);
        }
        let mut bytes = [0u8; 64];
        let text = name.as_str().as_bytes();
        bytes[0] = text.len() as u8;
        bytes[1..1 + text.len()].copy_from_slice(text);
        self.derive_raw(
            task,
            Rights::empty(),
            kore_abi::session::DERIVE_KIND_TASK_NAME,
            task,
            Some(&bytes[..1 + text.len()]),
        )
        .map(|_| ())
    }

    /// Bind an unmapped parent lease before START. Custody remains with parent;
    /// the kernel reserves the range exclusively until process reaping.
    pub fn bind_crash(
        &mut self,
        task: SlotRef,
        lease: SlotRef,
        offset: u64,
        length: u64,
    ) -> Result<(), BuildError> {
        if self.count != 0 || !self.flags.is_empty() {
            return Err(BuildError::EffectMustStandAlone);
        }
        let mut arg = [0u8; 24];
        arg[..4].copy_from_slice(&lease.index().to_le_bytes());
        arg[8..16].copy_from_slice(&offset.to_le_bytes());
        arg[16..24].copy_from_slice(&length.to_le_bytes());
        self.derive_raw(
            task,
            Rights::empty(),
            kore_abi::session::DERIVE_KIND_CRASH_BIND,
            task,
            Some(&arg),
        )
        .map(|_| ())
    }

    /// Read one evidence chunk before revoking the owning process handle.
    pub fn read_crash(
        &mut self,
        task: SlotRef,
        report: SlotRef,
        index: u32,
    ) -> Result<(), BuildError> {
        if self.count != 0 || !self.flags.is_empty() {
            return Err(BuildError::EffectMustStandAlone);
        }
        let mut arg = [0u8; 8];
        arg[..4].copy_from_slice(&index.to_le_bytes());
        self.derive_raw(
            task,
            Rights::empty(),
            kore_abi::session::DERIVE_KIND_CRASH_READ,
            report,
            Some(&arg),
        )
        .map(|_| ())
    }

    /// End worker admission and drain the calling process owner's peers.
    pub fn drain_process(
        &mut self,
        self_handle: SlotRef,
        report: SlotRef,
    ) -> Result<(), BuildError> {
        if self.count != 0 || !self.flags.is_empty() {
            return Err(BuildError::EffectMustStandAlone);
        }
        self.derive_raw(
            self_handle,
            Rights::empty(),
            kore_abi::session::DERIVE_KIND_PROCESS_DRAIN,
            report,
            None,
        )
        .map(|_| ())
    }

    /// `Derive × Task × TASK_JOIN` — **wait for `handle`'s task to end**,
    /// its [`ExitReport`](kore_abi::session::ExitReport) landing in
    /// `report`.  Pair with [`wait_task`](Self::wait_task) to block;
    /// alone it is the non-blocking probe.
    pub fn join(&mut self, handle: SlotRef, report: SlotRef) -> Result<Staged, BuildError> {
        self.derive_raw(
            handle,
            Rights::empty(),
            kore_abi::session::DERIVE_KIND_TASK_JOIN,
            report,
            None,
        )
    }

    /// `Derive × Task × TASK_START` — make a completely founded task
    /// runnable, presenting its parent handle (`MODIFY`).  START is a
    /// bind-inert, argument-free effect: it assigns no result and is staged as
    /// the session's sole invocation so a retry cannot hide other committed
    /// capability effects.
    pub fn start(&mut self, handle: SlotRef) -> Result<(), BuildError> {
        if self.count != 0 || !self.flags.is_empty() {
            return Err(BuildError::EffectMustStandAlone);
        }
        self.derive_raw(
            handle,
            Rights::empty(),
            kore_abi::session::DERIVE_KIND_TASK_START,
            // An effect mints nothing.  Reusing the subject index is inert:
            // the kernel's START arm never treats this as an assignment.
            handle,
            None,
        )
        .map(|_| ())
    }

    /// `Derive × PciDevice × DEVICE_ACTIVATE` — open the prepared device's
    /// bus-master latch against the exact DMA sub-lease that owns its live
    /// IOMMU window. The effect mints nothing and must stand alone because its
    /// post-commit config-space readback may refuse.
    pub fn activate_device(
        &mut self,
        device: SlotRef,
        dma_lease: SlotRef,
    ) -> Result<(), BuildError> {
        if self.count != 0 || !self.flags.is_empty() {
            return Err(BuildError::EffectMustStandAlone);
        }
        let mut req = [0u8; kore_abi::session::DeviceActivateReq::WIRE];
        req[0..4].copy_from_slice(&dma_lease.index().to_le_bytes());
        self.derive_raw(
            device,
            Rights::empty(),
            kore_abi::session::DERIVE_KIND_DEVICE_ACTIVATE,
            // No result is assigned; the authenticated device slot is inert.
            device,
            Some(&req),
        )
        .map(|_| ())
    }

    /// `Derive × PciDevice × DEVICE_QUIESCE` (#809 §L) — the terminal
    /// rendezvous: mask the function's interrupts, close and read back Bus
    /// Master Enable, bind its requester to Abort, quarantine its DMA epoch.
    /// The kernel answers with a bind-inert `QuiesceReport` in
    /// `report_slot`, which is where a `Quiesced` witness comes from.  Mints
    /// nothing and must stand alone: its config-space readback may refuse
    /// after the empty fold committed.
    pub fn quiesce_device(
        &mut self,
        device: SlotRef,
        report_slot: SlotRef,
    ) -> Result<(), BuildError> {
        if self.count != 0 || !self.flags.is_empty() {
            return Err(BuildError::EffectMustStandAlone);
        }
        self.derive_raw(
            device,
            Rights::empty(),
            kore_abi::session::DERIVE_KIND_DEVICE_QUIESCE,
            report_slot,
            None,
        )
        .map(|_| ())
    }

    /// Park after commit on the timer wheel until `deadline_ns` (absolute
    /// monotonic nanoseconds) — [`SessionFlags::WAIT_SLEEP`]
    /// (`res/USERLAND.md` §3.4.2).  [`finish`](Self::finish) emits the
    /// `SleepReq` into the park block at the argument region's front (the
    /// kore-abi flag-gated convention).  Combined with
    /// [`park_standing`](Self::park_standing) (or one staged recv post +
    /// [`wait_delivery`](Self::wait_delivery)) it is the
    /// deadline-or-doorbell park (§3.4.4).  **Runtime-internal in
    /// spirit**: a parking commit stalls the whole task — the executor's
    /// idle path is the intended caller (`commit_and_park`).
    pub fn wait_sleep(&mut self, deadline_ns: u64) -> &mut Self {
        self.flags |= SessionFlags::WAIT_SLEEP;
        self.sleep = Some(deadline_ns);
        self
    }

    /// Park after commit on a **standing** recv post a *previous* commit
    /// published — the park-on-standing-post `WAIT_DELIVERY` form
    /// (`res/USERLAND.md` §3.4.3): the session stages **zero** recv posts
    /// and [`finish`](Self::finish) emits `ParkReq { chan, slot,
    /// recv_tag }` at the park block's front.  The kernel bind-checks the
    /// named post against the ambient caller (holder + tag), so only the
    /// post's owner can park on it.
    pub fn park_standing(&mut self, chan: u32, slot: u16, recv_tag: u32) -> &mut Self {
        self.flags |= SessionFlags::WAIT_DELIVERY;
        self.park = Some((chan, slot, recv_tag));
        self
    }

    /// Add a second standing recv post to the empty delivery park.
    ///
    /// [`finish`](Self::finish) validates that a primary post exists, the two
    /// channels differ, and no invocation rides. With an optional deadline the
    /// finished descriptor is exactly 48 bytes, keeping x86-64 on its inline
    /// caller-saved-register transport rather than the arena scratch path.
    pub fn park_standing_alt(&mut self, chan: u32, slot: u16, recv_tag: u32) -> &mut Self {
        self.flags |= SessionFlags::WAIT_DELIVERY | SessionFlags::WAIT_DELIVERY_ALT;
        self.park_alt = Some((chan, slot, recv_tag));
        self
    }

    fn push(&mut self, inv: Inv) -> Result<(), BuildError> {
        if self.has_effect {
            return Err(BuildError::EffectMustStandAlone);
        }
        if self.count == MAX_INVOCATIONS {
            return Err(BuildError::TooManyInvocations);
        }
        let is_effect = inv.op == OpCode::Derive as u16
            && (inv.kind == kore_abi::session::DERIVE_KIND_PROCESS_DRAIN
                || inv.kind == kore_abi::session::DERIVE_KIND_TASK_IDENTITY
                || inv.kind == kore_abi::session::DERIVE_KIND_TASK_NAME
                || inv.kind == kore_abi::session::DERIVE_KIND_CRASH_BIND
                || inv.kind == kore_abi::session::DERIVE_KIND_PANIC_REPORT
                || inv.kind == kore_abi::session::DERIVE_KIND_CRASH_READ
                || inv.kind == kore_abi::session::DERIVE_KIND_TASK_START
                || inv.kind == kore_abi::session::DERIVE_KIND_ARENA_EXTEND
                || inv.kind == kore_abi::session::DERIVE_KIND_DEVICE_ACTIVATE
                || inv.kind == kore_abi::session::DERIVE_KIND_DEVICE_QUIESCE);
        self.invs[self.count] = inv;
        self.count += 1;
        self.has_effect = is_effect;
        Ok(())
    }

    fn stage_arg(&mut self, bytes: &[u8]) -> Result<u32, BuildError> {
        let off = self.args_len;
        let end = off + bytes.len();
        if end > ARGS_MAX {
            return Err(BuildError::ArgsOverflow);
        }
        self.args[off..end].copy_from_slice(bytes);
        self.args_len = end;
        Ok(off as u32)
    }

    /// Stage a raw derive:
    /// `Derive(subject, requested, kind) → result`, with an optional
    /// op-specific payload already serialized. TASK_EXIT is refused: use
    /// [`exit`](Self::exit) with a typed code instead.
    pub fn derive_raw(
        &mut self,
        subject: SlotRef,
        requested: Rights,
        kind: u8,
        result: SlotRef,
        arg: Option<&[u8]>,
    ) -> Result<Staged, BuildError> {
        if kind == kore_abi::session::DERIVE_KIND_TASK_EXIT {
            return Err(BuildError::UntypedExit);
        }
        self.derive_inner(subject, requested, kind, result, arg)
    }

    fn derive_inner(
        &mut self,
        subject: SlotRef,
        requested: Rights,
        kind: u8,
        result: SlotRef,
        arg: Option<&[u8]>,
    ) -> Result<Staged, BuildError> {
        let arg = match arg {
            Some(bytes) => self.stage_arg(bytes)?,
            None => Invocation::NO_ARG,
        };
        self.push(Inv {
            op: OpCode::Derive as u16,
            requested: requested.bits(),
            kind,
            subject: subject.index(),
            result: result.index(),
            arg,
        })?;
        Ok(Staged(result))
    }

    /// Attenuate a held task handle into an `ACCESS`-only lifecycle observer.
    ///
    /// This cannot use the ordinary `ResourceKind::Task` derive byte: that is
    /// the terminal `Found` operation. The dedicated flavor keeps the held
    /// observer edge distinct from founding while producing the same Task
    /// resource kind in the result slot.
    pub fn derive_task_observer(
        &mut self,
        subject: SlotRef,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.derive_raw(
            subject,
            Rights::ACCESS,
            kore_abi::session::DERIVE_KIND_TASK_OBSERVER,
            result,
            None,
        )
    }

    /// Stage a revoke: retire `subject`; its subtree dies by closure at
    /// commit (T6).
    pub fn revoke(&mut self, subject: SlotRef) -> Result<(), BuildError> {
        self.push(Inv {
            op: OpCode::Revoke as u16,
            requested: 0,
            kind: 0,
            subject: subject.index(),
            result: 0,
            arg: Invocation::NO_ARG,
        })
    }

    /// Stage a rendezvous **post** (recv half): the next transfer on the
    /// channel lands in `land_in` (the caller's own arena).
    pub fn post_recv(
        &mut self,
        chan_recv: SlotRef,
        land_in: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.push(Inv {
            op: OpCode::Rendezvous as u16,
            requested: 0,
            kind: 0,
            subject: chan_recv.index(),
            result: land_in.index(),
            arg: Invocation::NO_ARG,
        })?;
        Ok(Staged(land_in))
    }

    /// Stage a rendezvous **typed post** (recv half): accept exactly
    /// `kind`, with at least `minimum_rights`, into `land_in`.  A mismatched
    /// capability or a notice is refused at the board before the sender moves
    /// any authority, and this post remains standing for a matching sender.
    pub fn post_recv_typed(
        &mut self,
        chan_recv: SlotRef,
        land_in: SlotRef,
        kind: ResourceKind,
        minimum_rights: Rights,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; kore_abi::session::RecvPolicyReq::WIRE];
        req[0..2].copy_from_slice(&(kind as u16).to_le_bytes());
        req[2] = minimum_rights.bits();
        let arg = self.stage_arg(&req)?;
        self.push(Inv {
            op: OpCode::Rendezvous as u16,
            requested: 0,
            kind: 0,
            subject: chan_recv.index(),
            result: land_in.index(),
            arg,
        })?;
        Ok(Staged(land_in))
    }

    /// Stage a rendezvous **transfer** (send half): re-derive the caller's
    /// `cap` under the peer's posted slot, attenuated to `rights`
    /// (disclosure ≠ transfer: the peer gets a fresh stamped child, never
    /// these bytes).  Non-blocking — no posted peer recv refuses the whole
    /// session with `WouldBlock`.
    pub fn send(
        &mut self,
        chan_send: SlotRef,
        cap: impl Into<SlotRef>,
        rights: Rights,
    ) -> Result<(), BuildError> {
        self.push(Inv {
            op: OpCode::Rendezvous as u16,
            requested: rights.bits(),
            kind: 0,
            subject: chan_send.index(),
            result: cap.into().index(),
            arg: Invocation::NO_ARG,
        })
    }

    /// Move one exact held affine capability through a posted rendezvous.
    ///
    /// No attenuation is expressible: the source's authenticated rights,
    /// lineage, and tag are preserved, and successful commit scrubs the
    /// sender's name. This is distinct from [`send`](Self::send), whose
    /// derive-shaped transfer requires `DERIVE` on the source.
    pub fn send_exact(
        &mut self,
        chan_send: SlotRef,
        cap: impl Into<SlotRef>,
    ) -> Result<(), BuildError> {
        self.push(Inv {
            op: OpCode::Rendezvous as u16,
            requested: 0,
            kind: kore_abi::session::RENDEZVOUS_KIND_EXACT_MOVE,
            subject: chan_send.index(),
            result: cap.into().index(),
            arg: Invocation::NO_ARG,
        })
    }

    /// Stage a **notice-mode send** (`res/USERLAND.md` §3.4.1; SYSCALL.md
    /// Move C's arg-gated `Rendezvous` flavor): deliver the bind-inert
    /// `(cookie, payload)` notice into the peer's posted slot — no
    /// re-derivation, no rights move, no tag mint.  The doorbell kick:
    /// **rides its own session** in production (`WouldBlock` is a
    /// whole-session refusal, and under the §3.1 order a would-blocked
    /// kick is droppable), which `ChanTx::send` enforces.
    pub fn notice(
        &mut self,
        chan_send: SlotRef,
        cookie: u64,
        payload: u64,
    ) -> Result<(), BuildError> {
        let mut req = [0u8; 16];
        req[0..8].copy_from_slice(&cookie.to_le_bytes());
        req[8..16].copy_from_slice(&payload.to_le_bytes());
        let arg = self.stage_arg(&req)?;
        self.push(Inv {
            op: OpCode::Rendezvous as u16,
            requested: 0,
            kind: 0,
            subject: chan_send.index(),
            result: 0, // dead field — a notice resolves no slot
            arg,
        })
    }

    /// Stage the compact service request-ring kick.
    ///
    /// This is semantically the fixed `Notice { cookie: COOKIE_REQUEST,
    /// payload: 0 }` total-drain hint, but its append-only rendezvous flavor
    /// needs no [`NoticeReq`](kore_abi::session::NoticeReq) argument. The
    /// compact shape is what lets a request kick and a standing reply park
    /// share one 40-byte, all-GPR session on x86-64.
    pub fn request_kick(&mut self, chan_send: SlotRef) -> Result<(), BuildError> {
        self.push(Inv {
            op: OpCode::Rendezvous as u16,
            requested: 0,
            kind: kore_abi::session::RENDEZVOUS_KIND_REQUEST_KICK,
            subject: chan_send.index(),
            result: 0,
            arg: Invocation::NO_ARG,
        })
    }

    // ── Typed memory verbs (the pager + driver session vocabulary) ─────

    /// `Derive × AddrSpace → AsView` — a thread view of the subject space
    /// (Flow 1's first leg).  No payload.
    pub fn derive_view(
        &mut self,
        space: SlotRef,
        rights: Rights,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.derive_raw(space, rights, 0, result, None)
    }

    /// `Derive × space → Region` — a guarded region map
    /// ([`kore_abi::session::RegionMapReq`]): the caller names its virtual
    /// window, the kernel places the backing and installs guards (#27).
    pub fn map_region(
        &mut self,
        space: impl Into<SlotRef>,
        rights: Rights,
        va: u64,
        bytes: u64,
        guard_before: u32,
        guard_after: u32,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; 24];
        req[0..8].copy_from_slice(&va.to_le_bytes());
        req[8..16].copy_from_slice(&bytes.to_le_bytes());
        req[16..20].copy_from_slice(&guard_before.to_le_bytes());
        req[20..24].copy_from_slice(&guard_after.to_le_bytes());
        self.derive_raw(space.into(), rights, 0, result, Some(&req))
    }

    /// `Derive × PhysRegion → MemLease` ([`kore_abi::session::LeaseReq`]):
    /// allocate + lease a contiguous buffer — the shared root CPU and
    /// device windows both hang off (RUNG 2).
    pub fn lease(
        &mut self,
        region: SlotRef,
        rights: Rights,
        size: u64,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.lease_flags(region, rights, size, 0, result)
    }

    /// [`lease`](Self::lease) with explicit [`LeaseReq::flags`] — pass
    /// [`kore_abi::session::LEASE_SEALED`] for a **swap-sealed** lease
    /// (the swap seal: the kernel encrypts + authenticates each page on
    /// eviction, so an untrusted swap backend never sees plaintext).
    pub fn lease_flags(
        &mut self,
        region: SlotRef,
        rights: Rights,
        size: u64,
        flags: u8,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.lease_aligned(region, rights, size, flags, 0, result)
    }

    /// [`lease_flags`](Self::lease_flags) with an explicit backing
    /// alignment ([`LeaseReq::align_log2`]): `0` for the 4 KiB default,
    /// else one of [`kore_abi::session::LEASE_ALIGN_4K`] /
    /// [`LEASE_ALIGN_2M`](kore_abi::session::LEASE_ALIGN_2M) /
    /// [`LEASE_ALIGN_1G`](kore_abi::session::LEASE_ALIGN_1G).  A
    /// huge-aligned (and huge-rounded, the caller's job) window is what
    /// lets its maps ride large leaves — the driver DMA-pool shape.
    pub fn lease_aligned(
        &mut self,
        region: SlotRef,
        rights: Rights,
        size: u64,
        flags: u8,
        align_log2: u8,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; 16];
        req[0..8].copy_from_slice(&size.to_le_bytes());
        req[8] = flags;
        req[9] = align_log2;
        self.derive_raw(region, rights, 0, result, Some(&req))
    }

    /// Carve an affine sub-lease window from a held supply lease.  The sub is
    /// the single public owner of every CPU and device leaf subsequently
    /// derived over this window.
    pub fn carve_sublease(
        &mut self,
        supply: SlotRef,
        rights: Rights,
        offset: u64,
        len: u64,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; 24];
        req[0..8].copy_from_slice(&offset.to_le_bytes());
        req[8..16].copy_from_slice(&len.to_le_bytes());
        self.derive_raw(
            supply,
            rights,
            (ResourceKind::MemLease as u16) as u8,
            result,
            Some(&req),
        )
    }

    /// Carve a detached, geometry-only window from a held supply lease.
    ///
    /// No address space is named and no leaf is installed by this operation:
    /// the returned `LeasedRegion` is an affine transfer token carrying the
    /// supply's typed lifecycle state. Its receiver performs the sole install
    /// with [`crate::cap::verbs::map_view`].
    pub fn carve_window(
        &mut self,
        supply: SlotRef,
        rights: Rights,
        offset: u64,
        len: u64,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; 24];
        req[0..8].copy_from_slice(&offset.to_le_bytes());
        req[8..16].copy_from_slice(&len.to_le_bytes());
        self.derive_raw(
            supply,
            rights,
            (ResourceKind::LeasedRegion as u16) as u8,
            result,
            Some(&req),
        )
    }

    /// Map the complete held lease/sub-lease (or held BAR) into the task's
    /// own space.  The result slot receives a bind-inert geometry report;
    /// the backing root supplies the exact length.
    ///
    /// **Which window is the kernel's choice, not the caller's**: an
    /// ordinary backing lands at the task's map cursor (`geom::MAP`), while
    /// a `Display`-rooted BAR — the framebuffer aperture, one huge fixed
    /// mapping rather than one of many small views — lands in the task's
    /// `geom::APERTURE` window.  Nothing in this request names either, so a
    /// caller cannot put a framebuffer back on the cursor by asking.
    pub fn map_owned(
        &mut self,
        lease: SlotRef,
        rights: Rights,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.derive_raw(
            lease,
            rights,
            (ResourceKind::LeasedRegion as u16) as u8,
            result,
            None,
        )
    }

    /// Map a whole lease at the kernel-selected VA, retaining a named CPU
    /// view for exact unshare. Observe its geometry with `cpu_view_geometry`.
    pub fn map_owned_named(
        &mut self,
        lease: impl Into<SlotRef>,
        rights: Rights,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.derive_raw(
            lease.into(),
            rights,
            kore_abi::session::DERIVE_KIND_OWNED_MAP,
            result,
            None,
        )
    }

    /// Read an installed CPU view's geometry without installing another map.
    pub fn cpu_view_geometry(
        &mut self,
        view: impl Into<SlotRef>,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.derive_raw(
            view.into(),
            Rights::empty(),
            kore_abi::session::DERIVE_KIND_CPU_VIEW_OBSERVE,
            result,
            None,
        )
    }

    /// Exercise a detached view in a receiver-owned, founder-stamped window.
    /// The held or same-session view needs ACCESS, not delegation authority.
    /// No rights, length or destination address space are caller-selectable.
    pub fn map_view_in(
        &mut self,
        view: impl Into<SlotRef>,
        window: kore_abi::session::ViewMapWindow,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.map_view_request(view, window, 0, result)
    }

    /// Map one surface into a fixed reusable lane. The lane controls VA
    /// placement only; authority and geometry still come from the held view.
    pub fn map_view_surface_lane(
        &mut self,
        view: impl Into<SlotRef>,
        lane: kore_abi::session::SurfaceMapLane,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.map_view_request(
            view,
            kore_abi::session::ViewMapWindow::Surfaces,
            lane as u8,
            result,
        )
    }

    fn map_view_request(
        &mut self,
        view: impl Into<SlotRef>,
        window: kore_abi::session::ViewMapWindow,
        lane: u8,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; kore_abi::session::ViewMapReq::WIRE];
        req[..2].copy_from_slice(&window.role().to_le_bytes());
        req[2] = lane;
        self.derive_raw(
            view.into(),
            Rights::empty(),
            kore_abi::session::DERIVE_KIND_VIEW_MAP,
            result,
            Some(&req),
        )
    }

    /// Consume a writable lease and return its executable-only
    /// successor (`Revoke × REVOKE_KIND_EXEC`).
    ///
    /// The kernel requires `ACCESS|MODIFY|DERIVE|REVOKE` on `subject`, kills
    /// its complete descendant lineage, removes every CPU/DMA/guest alias,
    /// publishes the instruction stream, and writes an
    /// `ACCESS|DERIVE|REVOKE` lease into `result`.
    pub fn finalize_exec(
        &mut self,
        subject: SlotRef,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.push(Inv {
            op: OpCode::Revoke as u16,
            requested: 0,
            kind: kore_abi::session::REVOKE_KIND_EXEC,
            subject: subject.index(),
            result: result.index(),
            arg: Invocation::NO_ARG,
        })?;
        Ok(Staged(result))
    }

    /// Consume an executable lease and return its **writable** successor
    /// (`Revoke × REVOKE_KIND_EXEC_RETURN`) — the other half of a reusable
    /// JIT arena (`res/IMAGE.md` §6a).
    ///
    /// The kernel requires `ACCESS|DERIVE|REVOKE` on `subject` and refuses
    /// unless this exact window reached `X` through
    /// [`finalize_exec`](Self::finalize_exec): an X-minted immutable store
    /// is executable forever. It kills the executable descendant lineage,
    /// removes every CPU/DMA/guest alias, and writes an
    /// `ACCESS|MODIFY|DERIVE|REVOKE` lease into `result`.
    ///
    /// This does not weaken W^X. No mapping changes in place, and the
    /// window is never writable and executable at the same instant through
    /// any alias; what it removes is the need to *abandon* an arena in
    /// order to rewrite it.
    pub fn return_writable(
        &mut self,
        subject: SlotRef,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.push(Inv {
            op: OpCode::Revoke as u16,
            requested: 0,
            kind: kore_abi::session::REVOKE_KIND_EXEC_RETURN,
            subject: subject.index(),
            result: result.index(),
            arg: Invocation::NO_ARG,
        })?;
        Ok(Staged(result))
    }

    /// `Derive × DERIVE_KIND_ARENA_EXTEND` on the task's own space
    /// self-view (slice E): grow the caller's cap arena by one chunk.
    /// An **effect**, not a mint — the request is empty, the arg is the
    /// flavor's reserved-zero block, and the answer is a bind-inert
    /// [`GeomReport`](kore_abi::session::GeomReport) of the new chunk's
    /// user alias window in `result`.  **Sole invocation, unflagged**
    /// (the kernel takes START's rule): stage nothing else in this
    /// session.
    pub fn extend_arena(
        &mut self,
        self_view: SlotRef,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        // The START precondition, structurally: a standalone effect may
        // not follow other invocations (push() bars the reverse order and
        // finish() bars park flags — the kernel re-refuses all three).
        if self.count != 0 || !self.flags.is_empty() {
            return Err(BuildError::EffectMustStandAlone);
        }
        self.derive_raw(
            self_view,
            Rights::empty(),
            kore_abi::session::DERIVE_KIND_ARENA_EXTEND,
            result,
            Some(&[0u8; 8]),
        )
    }

    /// `Derive × {MemLease, MmioRegion} → LeasedRegion`
    /// ([`kore_abi::session::ShareReq`]): install the subject's window into
    /// the **caller's own** space at `va`.
    pub fn share_cpu(
        &mut self,
        subject: impl Into<SlotRef>,
        rights: Rights,
        va: u64,
        bytes: u64,
        write: bool,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; 24];
        req[0..8].copy_from_slice(&va.to_le_bytes());
        req[8..16].copy_from_slice(&bytes.to_le_bytes());
        req[16] = write as u8;
        self.derive_raw(subject.into(), rights, 0, result, Some(&req))
    }

    /// [`share_cpu`](Self::share_cpu) with an **explicit access byte**
    /// ([`kore_abi::session::SHARE_ACCESS_R`]`/W/X`, res/IMAGE.md §9c) —
    /// the R|X share of an exec-blessed text lease is the canonical use.
    /// `access == 0` would fall back to the legacy write byte (left 0
    /// here), so pass a non-zero bitfield.
    pub fn share_cpu_access(
        &mut self,
        subject: impl Into<SlotRef>,
        rights: Rights,
        va: u64,
        bytes: u64,
        access: u8,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; 24];
        req[0..8].copy_from_slice(&va.to_le_bytes());
        req[8..16].copy_from_slice(&bytes.to_le_bytes());
        req[17] = access;
        self.derive_raw(subject.into(), rights, 0, result, Some(&req))
    }

    /// `Derive × MemLease → LeasedRegion` (`kind = AddrSpace`,
    /// [`kore_abi::session::ShareToReq`], res/IMAGE.md §9b): install the
    /// subject lease's window into the **`dest_space`-named** address
    /// space — the spawn-session share (stage the child's windows while
    /// the child space capability is still yours; founding consumes it
    /// and closes the window structurally).  `access` follows the
    /// [`share_cpu_access`](Self::share_cpu_access) bitfield; `0` selects
    /// the legacy `write` byte semantics via `write`.
    #[allow(clippy::too_many_arguments)]
    pub fn share_to(
        &mut self,
        lease: impl Into<SlotRef>,
        rights: Rights,
        dest_space: SlotRef,
        va: u64,
        bytes: u64,
        write: bool,
        access: u8,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.share_to_inner(
            lease, rights, dest_space, va, bytes, write, access, result, false,
        )
    }

    /// [`share_to`](Self::share_to) that **consumes its result name into a
    /// same-session `Found`**
    /// ([`ShareToReq::consume`](kore_abi::session::ShareToReq::consume)):
    /// the install lands and its `SessionChild::Leased` twin — owned and
    /// reaped by the *destination* task's teardown — keeps the mapping
    /// authority, while `result` is cleared as that `Found` commits.
    ///
    /// This is the form a **spawner** wants for a child's segments: the
    /// plain `share_to` obliges the founder to hold one arena name per
    /// shared window for the child's entire life, which is what bounded
    /// exec to a single founding round.  `result` is still required (the
    /// staged child needs its SSA name inside the session) — it is simply
    /// not live after the commit, so the caller must not read it back.
    ///
    /// Consumes the **name**, never the capability: revoking any
    /// founder-side ancestor still tears the mapping down (T6). If no
    /// successful `Found` consumes the named destination, the result remains
    /// with the caller so the live twin never becomes ownerless.
    #[allow(clippy::too_many_arguments)]
    pub fn share_to_consuming(
        &mut self,
        lease: impl Into<SlotRef>,
        rights: Rights,
        dest_space: SlotRef,
        va: u64,
        bytes: u64,
        write: bool,
        access: u8,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.share_to_inner(
            lease, rights, dest_space, va, bytes, write, access, result, true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn share_to_inner(
        &mut self,
        lease: impl Into<SlotRef>,
        rights: Rights,
        dest_space: SlotRef,
        va: u64,
        bytes: u64,
        write: bool,
        access: u8,
        result: SlotRef,
        consume: bool,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; 24];
        req[0..4].copy_from_slice(&dest_space.index().to_le_bytes());
        req[4..12].copy_from_slice(&va.to_le_bytes());
        req[12..20].copy_from_slice(&bytes.to_le_bytes());
        req[20] = write as u8;
        req[21] = access;
        req[22] = consume as u8;
        self.derive_raw(
            lease.into(),
            rights,
            (ResourceKind::AddrSpace as u16) as u8,
            result,
            Some(&req),
        )
    }

    /// `Derive × Region` (`kind = Region`,
    /// [`kore_abi::session::DelegateReq`], res/IMAGE.md §9a):
    /// **delegate** the subject region into the `dest_space`-named
    /// address space at `va` — the pages stay put, only page-table
    /// ownership moves.  The source region retires with the commit; the
    /// fresh region (stamped to the caller — the spawner holds it until
    /// it hands the child its boot block) lands in `result`.  Carries no
    /// rights request (the new region's grant is the map's fixed one).
    #[allow(clippy::too_many_arguments)]
    pub fn delegate(
        &mut self,
        region: impl Into<SlotRef>,
        dest_space: SlotRef,
        va: u64,
        bytes: u64,
        write: bool,
        access: u8,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; 24];
        req[0..4].copy_from_slice(&dest_space.index().to_le_bytes());
        req[4..12].copy_from_slice(&va.to_le_bytes());
        req[12..20].copy_from_slice(&bytes.to_le_bytes());
        req[20] = write as u8;
        req[21] = access;
        self.derive_raw(
            region.into(),
            Rights::empty(),
            (ResourceKind::Region as u16) as u8,
            result,
            Some(&req),
        )
    }

    /// `Derive × VirtMemory → AddrSpace` (res/IMAGE.md §9e — the spawn session's
    /// first leg): mint a **fresh** address space off the held `VirtMemory`
    /// allocator capability (the process case; a thread view is the no-arg
    /// `derive_view` on the space root).  No payload — a `VirtMemory` subject is
    /// unambiguous, so the kernel routes on the subject kind alone.  The
    /// fresh space lands in `result`, still held by the caller (the
    /// spawner stages the child's windows into it, then consumes it with
    /// [`found`](Self::found)).
    pub fn derive_fresh_space(
        &mut self,
        virt_memory: SlotRef,
        rights: Rights,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.derive_raw(virt_memory, rights, 0, result, None)
    }

    /// The spawn session's terminal **`Found`** (`Derive × kind = Task`,
    /// res/IMAGE.md §9f): found the subject `space` (a
    /// [`derive_fresh_space`](Self::derive_fresh_space) result, or a
    /// thread view for `SHARE_SPACE`) as a complete task, consuming it —
    /// the child space is handed to the kernel's spawn assembly, so the
    /// spawner can no longer name it (the authority window closes
    /// structurally).  A process remains dormant until an authenticated
    /// [`start`](Self::start); a single-session thread is dispatched by
    /// assembly because no post-founding initialization aperture remains.
    /// Carries the frozen [`SpawnReq`] whole plus the
    /// per-task arena slot count (#207).  A `Found` must be the session's
    /// **last** effectual invocation; the founder's task handle
    /// (`ACCESS|MODIFY|DERIVE|REVOKE`) lands in `result` (vacant for a
    /// `DETACHED` spawn).  The founder may attenuate it to an
    /// `ACCESS|MODIFY|REVOKE` client handle while retaining its parent
    /// authority; `MODIFY` is the child's explicit START authority.
    #[allow(clippy::too_many_arguments)]
    pub fn found(
        &mut self,
        space: impl Into<SlotRef>,
        entry: u64,
        stack: u64,
        bootstrap: Option<Staged>,
        flags: SpawnFlags,
        arena_slots: u32,
        authority: SlotRef,
        handle: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.found_deliver(
            space,
            entry,
            stack,
            bootstrap,
            flags,
            arena_slots,
            authority,
            handle,
            &[],
        )
    }

    /// [`found`](Self::found) with a **delivery map** (descriptor ABI v3 —
    /// U1, cross-space cap delivery at founding): each [`Deliver`] places a
    /// session child this builder staged — named by its SSA result
    /// ([`Staged`]) — into the founded child's arena at `to_slot`, sealed
    /// to the child's holder (D1) by the kernel's spawn assembly.
    /// Delivery *moves* the cap: the founder's own result slot is scrubbed
    /// at commit and the kernel twin joins the child's holdings.  Only the
    /// tracked leaf class delivers (channel ends, leases, leased views,
    /// shares…) — a `Space` is spawn material and a `Region` moves by
    /// [`delegate`](Self::delegate), both refused kernel-side.
    #[allow(clippy::too_many_arguments)]
    pub fn found_deliver(
        &mut self,
        space: impl Into<SlotRef>,
        entry: u64,
        stack: u64,
        bootstrap: Option<Staged>,
        flags: SpawnFlags,
        arena_slots: u32,
        authority: SlotRef,
        handle: SlotRef,
        deliver: &[Deliver],
    ) -> Result<Staged, BuildError> {
        self.found_deliver_inner(
            space,
            entry,
            stack,
            bootstrap,
            flags,
            arena_slots,
            authority,
            handle,
            deliver,
            Rights::empty(),
        )
    }

    /// Found while publishing the sole lifecycle handle directly through a
    /// held send endpoint whose peer has a typed task post.
    #[allow(clippy::too_many_arguments)]
    pub fn found_deliver_returning(
        &mut self,
        space: impl Into<SlotRef>,
        entry: u64,
        stack: u64,
        bootstrap: Option<Staged>,
        flags: SpawnFlags,
        arena_slots: u32,
        authority: SlotRef,
        return_chan: SlotRef,
        deliver: &[Deliver],
    ) -> Result<Staged, BuildError> {
        assert!(return_chan.index() < kore_abi::session::FOUND_RETURN_DIRECT);
        let result =
            SlotRef::from_wire(kore_abi::session::FOUND_RETURN_DIRECT | return_chan.index());
        self.found_deliver_inner(
            space,
            entry,
            stack,
            bootstrap,
            flags,
            arena_slots,
            authority,
            result,
            deliver,
            Rights::empty(),
        )
    }

    /// [`found_deliver_returning`](Self::found_deliver_returning) with an
    /// explicit direct-return handle grant. The kernel accepts only its two
    /// fixed lifecycle-owner contracts; this method exists so an opt-in
    /// controller may retain `DERIVE` and attenuate an observer into a worker.
    #[allow(clippy::too_many_arguments)]
    pub fn found_deliver_returning_with_rights(
        &mut self,
        space: impl Into<SlotRef>,
        entry: u64,
        stack: u64,
        bootstrap: Option<Staged>,
        flags: SpawnFlags,
        arena_slots: u32,
        authority: SlotRef,
        return_chan: SlotRef,
        deliver: &[Deliver],
        handle_rights: Rights,
    ) -> Result<Staged, BuildError> {
        assert!(return_chan.index() < kore_abi::session::FOUND_RETURN_DIRECT);
        let result =
            SlotRef::from_wire(kore_abi::session::FOUND_RETURN_DIRECT | return_chan.index());
        self.found_deliver_inner(
            space,
            entry,
            stack,
            bootstrap,
            flags,
            arena_slots,
            authority,
            result,
            deliver,
            handle_rights,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn found_deliver_inner(
        &mut self,
        space: impl Into<SlotRef>,
        entry: u64,
        stack: u64,
        bootstrap: Option<Staged>,
        flags: SpawnFlags,
        arena_slots: u32,
        authority: SlotRef,
        result: SlotRef,
        deliver: &[Deliver],
        handle_rights: Rights,
    ) -> Result<Staged, BuildError> {
        if deliver.len() > MAX_FOUND_DELIVERIES {
            return Err(BuildError::TooManyDeliveries);
        }
        if self.founds >= MAX_SESSION_FOUNDS {
            // The typed refusal names the remedy (split the launch); the
            // args math below would refuse anyway, as `ArgsOverflow`.
            return Err(BuildError::TooManyFounds);
        }
        // FoundReq v3: spawn(entry@0, stack@8, bootstrap@16, flags@20) ‖
        // arena_slots@24 ‖ deliver_count@28 ‖ authority@30 ‖
        // [DeliverEntry; MAX_FOUND_DELIVERIES = 78]@32 — the frozen
        // 344-byte (`FoundReq::WIRE`) layout.  The
        // `authority` slot names the founder's `ExecAuthority` cap the
        // kernel's prepare fold binds before founding (U3b CP-4, §9f).
        let mut req = [0u8; FoundReq::WIRE];
        req[0..8].copy_from_slice(&entry.to_le_bytes());
        req[8..16].copy_from_slice(&stack.to_le_bytes());
        let boot = bootstrap.map_or(SpawnReq::NO_BOOTSTRAP, |s| s.slot().index());
        req[16..20].copy_from_slice(&boot.to_le_bytes());
        req[20..24].copy_from_slice(&flags.bits().to_le_bytes());
        req[24..28].copy_from_slice(&arena_slots.to_le_bytes());
        req[28..30].copy_from_slice(&(deliver.len() as u16).to_le_bytes());
        req[30..32].copy_from_slice(&(authority.index() as u16).to_le_bytes());
        for (k, d) in deliver.iter().enumerate() {
            let at = 32 + k * 4;
            req[at..at + 2].copy_from_slice(&(d.from.index() as u16).to_le_bytes());
            req[at + 2..at + 4].copy_from_slice(&d.to_slot.to_le_bytes());
        }
        let staged = self.derive_raw(
            space.into(),
            handle_rights,
            (ResourceKind::Task as u16) as u8,
            result,
            Some(&req),
        )?;
        self.founds += 1;
        Ok(staged)
    }

    /// Found one schedulable guest vCPU from a held `GuestSpace` owner.
    ///
    /// `window`, `resume_recv`, and `exit_send` name held affine facets which
    /// the kernel moves into the new vCPU task only after the complete session
    /// commits.  The caller retains the guest-space owner and receives the
    /// ordinary lifecycle `Task` handle in `handle`.  Like process `Found`,
    /// this begins a terminal Found tail: only guest Founds for this same
    /// owner may follow, up to `MAX_SESSION_FOUNDS` members.
    #[allow(clippy::too_many_arguments)]
    pub fn found_guest(
        &mut self,
        guest_space: impl Into<SlotRef>,
        entry: u64,
        stack: u64,
        page_table_root: u64,
        window: SlotRef,
        resume_recv: SlotRef,
        exit_send: SlotRef,
        handle: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; GuestFoundReq::WIRE];
        req[0..8].copy_from_slice(&entry.to_le_bytes());
        req[8..16].copy_from_slice(&stack.to_le_bytes());
        req[16..24].copy_from_slice(&page_table_root.to_le_bytes());
        req[24..28].copy_from_slice(&window.index().to_le_bytes());
        req[28..32].copy_from_slice(&resume_recv.index().to_le_bytes());
        req[32..36].copy_from_slice(&exit_send.index().to_le_bytes());
        // bytes 36..40 are the reserved-zero word.
        self.derive_raw(
            guest_space.into(),
            Rights::empty(),
            DERIVE_KIND_GUEST_FOUND,
            handle,
            Some(&req),
        )
    }

    /// `Revoke × REVOKE_KIND_UNMAP` (res/IMAGE.md §9a — the #83
    /// return-leg): revoke the subject `region`, clear its leaves, and
    /// receive the **bare backing extent** in `result` — the consume-op's
    /// return leg as a session verb.  No argument payload.
    pub fn unmap(
        &mut self,
        region: impl Into<SlotRef>,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.push(Inv {
            op: OpCode::Revoke as u16,
            requested: 0,
            kind: kore_abi::session::REVOKE_KIND_UNMAP,
            subject: region.into().index(),
            result: result.index(),
            arg: Invocation::NO_ARG,
        })?;
        Ok(Staged(result))
    }

    /// `Derive × (MemLease | MmioRegion) → LeasedIova` (`kind = IovaRegion`,
    /// [`kore_abi::session::DeviceShareReq`]): install the backing window
    /// into `device`'s IOMMU domain at `iova` — the zero-copy DMA leaf
    /// over the same frames a CPU share maps.  The device slot is a read
    /// edge (footprinted, never a parent).
    pub fn share_device(
        &mut self,
        lease: impl Into<SlotRef>,
        rights: Rights,
        device: SlotRef,
        iova: u64,
        bytes: u64,
        write: bool,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let access = if write {
            kore_abi::session::DeviceShareReq::ACCESS_READ_WRITE
        } else {
            kore_abi::session::DeviceShareReq::ACCESS_READ
        };
        self.share_device_access(
            lease.into(),
            rights,
            device,
            iova,
            bytes,
            access,
            (ResourceKind::IovaRegion as u16) as u8,
            result,
        )
    }

    /// `Derive × (MemLease | MmioRegion) → LeasedIova`, granting only
    /// **posted writes** into the target window. This is the peer-BAR form:
    /// it deliberately carries no read authority, so admission needs request
    /// routing but not an unproven completion route.
    pub fn share_device_write_only(
        &mut self,
        backing: impl Into<SlotRef>,
        rights: Rights,
        device: SlotRef,
        iova: u64,
        bytes: u64,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.share_device_access(
            backing.into(),
            rights,
            device,
            iova,
            bytes,
            kore_abi::session::DeviceShareReq::ACCESS_WRITE,
            (ResourceKind::IovaRegion as u16) as u8,
            result,
        )
    }

    fn share_device_access(
        &mut self,
        backing: SlotRef,
        rights: Rights,
        device: SlotRef,
        iova: u64,
        bytes: u64,
        access: u8,
        kind: u8,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; 24];
        req[0..4].copy_from_slice(&device.index().to_le_bytes());
        req[4..12].copy_from_slice(&iova.to_le_bytes());
        req[12..20].copy_from_slice(&bytes.to_le_bytes());
        req[20] = access;
        self.derive_raw(backing, rights, kind, result, Some(&req))
    }

    /// `Derive × MemLease → GuestRegion` (`kind = GuestRegion`): install
    /// the lease at a caller-chosen GPA in the `space` capability. The space
    /// is a prepare-footprinted second subject, never the child's parent.
    #[allow(clippy::too_many_arguments)]
    pub fn share_guest(
        &mut self,
        lease: impl Into<SlotRef>,
        rights: Rights,
        space: SlotRef,
        gpa: u64,
        bytes: u64,
        access: u8,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; kore_abi::session::GuestShareReq::WIRE];
        req[0..4].copy_from_slice(&space.index().to_le_bytes());
        req[4..12].copy_from_slice(&gpa.to_le_bytes());
        req[12..20].copy_from_slice(&bytes.to_le_bytes());
        req[20] = access;
        self.derive_raw(
            lease.into(),
            rights,
            (ResourceKind::GuestRegion as u16) as u8,
            result,
            Some(&req),
        )
    }

    /// `Derive × MmioRegion → LeasedIova` through an exact, sponsored peer
    /// aperture ([`kore_abi::session::PeerDeviceShareReq`]). The BAR remains
    /// the sole child parent; `requester` and `sponsor` are independently
    /// footprinted authority inputs.
    #[allow(clippy::too_many_arguments)]
    pub fn share_peer_device(
        &mut self,
        bar: impl Into<SlotRef>,
        rights: Rights,
        requester: SlotRef,
        sponsor: SlotRef,
        iova: u64,
        offset: u64,
        bytes: u64,
        access: u8,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; kore_abi::session::PeerDeviceShareReq::WIRE];
        req[0..4].copy_from_slice(&requester.index().to_le_bytes());
        req[4..8].copy_from_slice(&sponsor.index().to_le_bytes());
        req[8..16].copy_from_slice(&iova.to_le_bytes());
        req[16..24].copy_from_slice(&offset.to_le_bytes());
        req[24..32].copy_from_slice(&bytes.to_le_bytes());
        req[32] = access;
        self.derive_raw(
            bar.into(),
            rights,
            DERIVE_KIND_PEER_IOVA,
            result,
            Some(&req),
        )
    }

    /// The affine-lifecycle form of [`share_device`](Self::share_device):
    /// install the same typed `LeasedIova` child, but leave `result`
    /// bind-inert.  The child remains tracked in the task holdings and is
    /// reaped through the public lease/sub-lease parent.
    pub fn share_device_anonymous(
        &mut self,
        lease: impl Into<SlotRef>,
        rights: Rights,
        device: SlotRef,
        iova: u64,
        bytes: u64,
        write: bool,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let access = if write {
            kore_abi::session::DeviceShareReq::ACCESS_READ_WRITE
        } else {
            kore_abi::session::DeviceShareReq::ACCESS_READ
        };
        self.share_device_access(
            lease.into(),
            rights,
            device,
            iova,
            bytes,
            access,
            DERIVE_KIND_IOVA_ANON,
            result,
        )
    }

    /// Anonymous-lifecycle form of
    /// [`share_device_write_only`](Self::share_device_write_only).
    pub fn share_device_write_only_anonymous(
        &mut self,
        backing: impl Into<SlotRef>,
        rights: Rights,
        device: SlotRef,
        iova: u64,
        bytes: u64,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.share_device_access(
            backing.into(),
            rights,
            device,
            iova,
            bytes,
            kore_abi::session::DeviceShareReq::ACCESS_WRITE,
            DERIVE_KIND_IOVA_ANON,
            result,
        )
    }

    /// `Derive × PciDevice → MmioRegion` ([`kore_abi::session::BarReq`],
    /// #85): claim BAR `bar` of the subject device — ready for the
    /// [`share_cpu`](Self::share_cpu) second hop.
    pub fn claim_bar(
        &mut self,
        device: SlotRef,
        rights: Rights,
        bar: u8,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; 8];
        req[0] = bar;
        self.derive_raw(
            device,
            rights,
            ResourceKind::MmioRegion as u8,
            result,
            Some(&req),
        )
    }

    /// `Derive × PciDevice → SharedRamRegion`: claim exact ivshmem-plain
    /// (`1af4:1110`) BAR2.  Argument-free by design: neither the BAR number
    /// nor its fixed writeback/coherent cache policy is caller-selected.
    pub fn claim_shared_ram(
        &mut self,
        device: SlotRef,
        rights: Rights,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.derive_raw(
            device,
            rights,
            kore_abi::cap::ResourceKind::SharedRamRegion as u8,
            result,
            None,
        )
    }

    /// Read the number of validated modern VirtIO PCI capability rows from
    /// the exact held device. The result is a bind-inert count report.
    pub fn read_virtio_pci_cap_count(
        &mut self,
        device: SlotRef,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; kore_abi::session::VirtioPciLayoutReq::WIRE];
        req[1] = kore_abi::session::VIRTIO_PCI_LAYOUT_QUERY_COUNT;
        self.derive_raw(
            device,
            Rights::empty(),
            kore_abi::session::DERIVE_KIND_VIRTIO_PCI_LAYOUT,
            result,
            Some(&req),
        )
    }

    /// Read one validated modern VirtIO PCI capability row in chain order.
    /// The result is redacted bind-inert data and mints no mapping authority.
    pub fn read_virtio_pci_cap(
        &mut self,
        device: SlotRef,
        index: u8,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; kore_abi::session::VirtioPciLayoutReq::WIRE];
        req[0] = index;
        req[1] = kore_abi::session::VIRTIO_PCI_LAYOUT_QUERY_RECORD;
        self.derive_raw(
            device,
            Rights::empty(),
            kore_abi::session::DERIVE_KIND_VIRTIO_PCI_LAYOUT,
            result,
            Some(&req),
        )
    }

    /// `Derive × PciDevice → DeviceMmu`: attest that the caller has completed
    /// class-specific stage-1 programming and now owns its lifecycle.
    pub fn attest_device_mmu(
        &mut self,
        device: impl Into<SlotRef>,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.derive_raw(
            device.into(),
            kore_abi::session::DEVICE_MMU_RIGHTS,
            ResourceKind::DeviceMmu as u8,
            result,
            None,
        )
    }

    /// Register the exact lease backing as a memory-fabric domain. No caller
    /// supplies extent, physical address, cache policy or device identity.
    pub fn register_fabric_lease(
        &mut self,
        lease: SlotRef,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.derive_raw(
            lease,
            Rights::ACCESS
                .union(Rights::MODIFY)
                .union(Rights::DERIVE)
                .union(Rights::REVOKE),
            ResourceKind::DomainRegistration as u8,
            result,
            None,
        )
    }

    /// Derive an executable mapping route from a held or staged registration.
    pub fn open_fabric_route(
        &mut self,
        registration: impl Into<SlotRef>,
        request: &kore_abi::session::FabricRouteReq,
        rights: Rights,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.derive_raw(
            registration.into(),
            rights,
            ResourceKind::MemoryFabricRoute as u8,
            result,
            Some(&request.encode()),
        )
    }

    /// Delegate a relative range from a held or staged route.
    pub fn scope_fabric_route(
        &mut self,
        route: impl Into<SlotRef>,
        request: &kore_abi::session::FabricRouteScopeReq,
        rights: Rights,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.derive_raw(
            route.into(),
            rights,
            kore_abi::session::DERIVE_KIND_FABRIC_SCOPE,
            result,
            Some(&request.encode()),
        )
    }

    /// Attempt to close a held route's admission. Run alone and unflagged;
    /// the result is a FabricRouteCloseReport, and the route remains owned.
    pub fn close_fabric_route(
        &mut self,
        route: SlotRef,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.derive_raw(
            route,
            Rights::empty(),
            kore_abi::session::DERIVE_KIND_FABRIC_CLOSE,
            result,
            None,
        )
    }

    /// Observe a held or staged registration without deriving authority.
    pub fn describe_fabric(
        &mut self,
        registration: impl Into<SlotRef>,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.derive_raw(
            registration.into(),
            Rights::empty(),
            kore_abi::session::DERIVE_KIND_FABRIC_OBSERVE,
            result,
            None,
        )
    }

    /// `Derive × PlatformDevice → MmioRegion`: claim the subject's sole
    /// firmware-declared register window.  Arg-free by construction; a caller
    /// cannot substitute a BAR number or physical address.
    pub fn claim_platform_mmio(
        &mut self,
        device: SlotRef,
        rights: Rights,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.derive_raw(
            device,
            rights,
            kore_abi::cap::ResourceKind::MmioRegion as u16 as u8,
            result,
            None,
        )
    }

    /// `Derive × RootComplex` / `Derive × PlatformBus`
    /// (`kind = DEVICE_RECORD`, `res/DEVICE_SERVICE.md` §10): **read** the
    /// subject's inventory entry `index` — the result slot receives a
    /// bind-inert [`DeviceReport`](kore_abi::session::DeviceReport) and no
    /// capability is minted.  Which namespace the index means is the
    /// subject capability's, never a wire field's.
    ///
    /// `ACCESS` on the subject is enough, and the request is empty: a
    /// broker walking its inventory to match a selector is observing, not
    /// acquiring.
    pub fn read_device_record(
        &mut self,
        store: SlotRef,
        index: u32,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; InventoryReq::WIRE];
        req[0..4].copy_from_slice(&index.to_le_bytes());
        self.derive_raw(
            store,
            Rights::empty(),
            DERIVE_KIND_DEVICE_RECORD,
            result,
            Some(&req),
        )
    }

    /// `Derive × RootComplex` / `Derive × PlatformBus`
    /// (`kind = DEVICE_RECORD`, count query): read the subject's
    /// authoritative boot-frozen inventory length into a bind-inert
    /// [`DeviceCountReport`](kore_abi::session::DeviceCountReport).
    ///
    /// The query rides the selector byte reserved in [`InventoryReq`]; the
    /// wire layout does not grow, and the zero index keeps the count shape
    /// disjoint from entry reads and claims.
    pub fn read_inventory_len(
        &mut self,
        store: SlotRef,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; InventoryReq::WIRE];
        req[4] = INVENTORY_QUERY_COUNT;
        self.derive_raw(
            store,
            Rights::empty(),
            DERIVE_KIND_DEVICE_RECORD,
            result,
            Some(&req),
        )
    }

    /// Observe the committed PCI claim beneath this inventory root. The
    /// result is bind-inert correlation; it conveys no retirement authority.
    pub fn read_device_claim(
        &mut self,
        root: SlotRef,
        index: u32,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; InventoryReq::WIRE];
        req[0..4].copy_from_slice(&index.to_le_bytes());
        req[4] = kore_abi::session::INVENTORY_QUERY_DEVICE_CLAIM;
        self.derive_raw(
            root,
            Rights::empty(),
            DERIVE_KIND_DEVICE_RECORD,
            result,
            Some(&req),
        )
    }

    /// Close only this exact committed grant's admission. Use a singular,
    /// unflagged session; the report never authorizes physical device reuse.
    pub fn close_device_claim(
        &mut self,
        root: SlotRef,
        index: u32,
        identity: kore_abi::session::DeviceClaimReport,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; kore_abi::session::DeviceClaimCloseReq::WIRE];
        req[0..4].copy_from_slice(&index.to_le_bytes());
        req[8..16].copy_from_slice(&identity.generation.to_le_bytes());
        req[16..24].copy_from_slice(&identity.grant.0.to_le_bytes());
        self.derive_raw(
            root,
            Rights::empty(),
            kore_abi::session::DERIVE_KIND_DEVICE_CLAIM_CLOSE,
            result,
            Some(&req),
        )
    }

    /// Contain this exact already-closed and drained grant. Use a singular,
    /// unflagged session; the report never authorizes physical device reuse.
    pub fn quiesce_device_claim(
        &mut self,
        root: SlotRef,
        index: u32,
        identity: kore_abi::session::DeviceClaimReport,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; kore_abi::session::DeviceClaimCloseReq::WIRE];
        req[0..4].copy_from_slice(&index.to_le_bytes());
        req[8..16].copy_from_slice(&identity.generation.to_le_bytes());
        req[16..24].copy_from_slice(&identity.grant.0.to_le_bytes());
        self.derive_raw(
            root,
            Rights::empty(),
            kore_abi::session::DERIVE_KIND_DEVICE_CLAIM_QUIESCE,
            result,
            Some(&req),
        )
    }

    /// Retire registered DMA under the original closed/drained claim issuer.
    pub fn retire_device_claim_dma(
        &mut self,
        root: SlotRef,
        index: u32,
        identity: kore_abi::session::DeviceClaimReport,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; kore_abi::session::DeviceClaimCloseReq::WIRE];
        req[0..4].copy_from_slice(&index.to_le_bytes());
        req[8..16].copy_from_slice(&identity.generation.to_le_bytes());
        req[16..24].copy_from_slice(&identity.grant.0.to_le_bytes());
        self.derive_raw(
            root,
            Rights::empty(),
            kore_abi::session::DERIVE_KIND_DEVICE_CLAIM_DMA_RETIRE,
            result,
            Some(&req),
        )
    }

    /// Read one inert admission page under an already-held inventory root.
    /// The result is diagnostic data; this never claims a PCI device.
    pub fn read_i2c_admission_page(
        &mut self,
        root: SlotRef,
        row: u8,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; InventoryReq::WIRE];
        req[0..4].copy_from_slice(&u32::from(row).to_le_bytes());
        req[4] = kore_abi::session::INVENTORY_QUERY_I2C_ADMISSION;
        self.derive_raw(
            root,
            Rights::empty(),
            DERIVE_KIND_DEVICE_RECORD,
            result,
            Some(&req),
        )
    }

    /// Observe one firmware-resolved HID-over-I2C child behind an already
    /// claimed PCI or platform controller. The result is a bind-inert
    /// [`I2cHidReport`](kore_abi::session::I2cHidReport).
    pub fn read_i2c_hid_record(
        &mut self,
        device: SlotRef,
        index: u32,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; I2cHidReq::WIRE];
        req[0..4].copy_from_slice(&index.to_le_bytes());
        self.derive_raw(
            device,
            Rights::empty(),
            DERIVE_KIND_I2C_HID_RECORD,
            result,
            Some(&req),
        )
    }

    /// Observe the authoritative number of HID-over-I2C children behind one
    /// claimed PCI or platform controller.
    pub fn read_i2c_hid_len(
        &mut self,
        device: SlotRef,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; I2cHidReq::WIRE];
        req[4] = I2C_HID_QUERY_COUNT;
        self.derive_raw(
            device,
            Rights::empty(),
            DERIVE_KIND_I2C_HID_RECORD,
            result,
            Some(&req),
        )
    }

    /// `Derive × RootComplex → PciDevice` / `Derive × PlatformBus →
    /// PlatformDevice` (`res/DEVICE_SERVICE.md` §10): hand out the
    /// subject's inventory entry `index` as a device capability — the
    /// derive a userland broker runs to give a driver its device.
    ///
    /// `species` names which child kind is wanted, and a species the
    /// subject does not vend **refuses** rather than coercing: the two
    /// grant disjoint authorities, and a driver handed the wrong one fails
    /// deep inside bring-up instead of here.
    pub fn claim_stored_device(
        &mut self,
        store: SlotRef,
        rights: Rights,
        index: u32,
        species: DeviceSpecies,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; InventoryReq::WIRE];
        req[0..4].copy_from_slice(&index.to_le_bytes());
        let kind = match species {
            DeviceSpecies::Pci => ResourceKind::PciDevice,
            DeviceSpecies::Platform => ResourceKind::PlatformDevice,
        };
        self.derive_raw(store, rights, (kind as u16) as u8, result, Some(&req))
    }

    /// `Derive × PciDevice → IrqRoute` (`kind = IrqRoute`, #86): establish
    /// the device's interrupt route — the cap `WAIT_IRQ` parks on.
    pub fn derive_irq_route(
        &mut self,
        device: SlotRef,
        rights: Rights,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.derive_irq_route_with(device, rights, result, 0, 1, None)
    }

    /// Establish a consecutive MSI/MSI-X block with an optional CPU-affinity
    /// hint. The kernel validates source bounds/capacity and may choose another
    /// eligible CPU when the hint cannot host the full block.
    pub fn derive_irq_route_with(
        &mut self,
        device: SlotRef,
        rights: Rights,
        result: SlotRef,
        first_entry: u16,
        count: u16,
        affinity: Option<u32>,
    ) -> Result<Staged, BuildError> {
        self.derive_irq_route_with_kind(
            device,
            rights,
            result,
            first_entry,
            count,
            affinity,
            (ResourceKind::IrqRoute as u16) as u8,
        )
    }

    /// Establish a consecutive block whose PCI source must be MSI-X.
    ///
    /// This is not an interrupt-remapping preference: every userland route
    /// still requires the remapper. It is a device-source contract for
    /// protocols whose table-entry identity cannot be represented by the
    /// aggregate MSI fallback.
    pub fn derive_msix_route_with(
        &mut self,
        device: SlotRef,
        rights: Rights,
        result: SlotRef,
        first_entry: u16,
        count: u16,
        affinity: Option<u32>,
    ) -> Result<Staged, BuildError> {
        self.derive_irq_route_with_kind(
            device,
            rights,
            result,
            first_entry,
            count,
            affinity,
            DERIVE_KIND_IRQ_ROUTE_MSIX,
        )
    }

    /// Establish only this PCI controller's firmware-resolved INTx line.
    /// No IRQ geometry is accepted from the caller.
    pub fn derive_pci_line_route(
        &mut self,
        device: SlotRef,
        rights: Rights,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.derive_raw(
            device,
            rights,
            kore_abi::session::DERIVE_KIND_IRQ_ROUTE_LINE,
            result,
            None,
        )
    }

    /// Establish one initialized HID-over-I2C child's firmware-correlated
    /// Child-ready route beneath an authenticated PCI or platform controller.
    pub fn derive_i2c_hid_ready_route(
        &mut self,
        device: SlotRef,
        rights: Rights,
        child: u32,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; I2cHidReq::WIRE];
        req[0..4].copy_from_slice(&child.to_le_bytes());
        self.derive_raw(
            device,
            rights,
            kore_abi::session::DERIVE_KIND_I2C_HID_READY_ROUTE,
            result,
            Some(&req),
        )
    }

    fn derive_irq_route_with_kind(
        &mut self,
        device: SlotRef,
        rights: Rights,
        result: SlotRef,
        first_entry: u16,
        count: u16,
        affinity: Option<u32>,
        kind: u8,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; IrqReq::WIRE];
        req[0..2].copy_from_slice(&first_entry.to_le_bytes());
        req[2..4].copy_from_slice(&count.to_le_bytes());
        req[4..8].copy_from_slice(&affinity.unwrap_or(IRQ_AFFINITY_ANY).to_le_bytes());
        self.derive_raw(device, rights, kind, result, Some(&req))
    }

    /// `Derive × ChannelSpace → Channel` (`kind = Channel`, §12.5/#204 —
    /// the channel-mint's first leg): mint a fresh rendezvous channel off
    /// the held `ChannelSpace` record (the primordial facility, endowed
    /// like the `VirtMemory`).  No payload — the record kind routes it; the
    /// endpoints then carve off the staged channel
    /// ([`derive_chan_send`](Self::derive_chan_send) /
    /// [`derive_chan_recv`](Self::derive_chan_recv)), so a full pair is
    /// three derives in one session (the kernel boottest's
    /// `channel_mint_descriptor` shape, as typed verbs — `res/USERLAND.md`
    /// §5.2).  The mint conventionally carries `ACCESS|DERIVE` (the
    /// carves) `+ REVOKE` (the T6 reap of both ends).
    pub fn derive_channel(
        &mut self,
        channel_space: SlotRef,
        rights: Rights,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.derive_raw(
            channel_space,
            rights,
            (ResourceKind::Channel as u16) as u8,
            result,
            None,
        )
    }

    /// `Derive × ChannelSpace → ChannelSpace`: cut a revocable mint grant
    /// for a founded launcher without handing it the primordial machine root.
    ///
    /// This is the self-kind sibling of [`derive_channel`](Self::derive_channel)
    /// and mirrors the `VirtMemory → VirtMemory` founding grant. It is arg-free: the
    /// singleton scope is fully named by the held subject, while rights and
    /// the fresh lineage tag are supplied by the ordinary derive contract.
    pub fn derive_channel_space_grant(
        &mut self,
        channel_space: SlotRef,
        rights: Rights,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.derive_raw(
            channel_space,
            rights,
            (ResourceKind::ChannelSpace as u16) as u8,
            result,
            None,
        )
    }

    /// `Derive × Channel → ChanSend` (`kind = ChanSend`): carve the
    /// **send** endpoint off a staged [`derive_channel`](Self::derive_channel)
    /// result — the end a rendezvous transfer / doorbell kick binds.
    /// Arg-free: the endpoint is fully named by its parent + kind.
    pub fn derive_chan_send(
        &mut self,
        channel: impl Into<SlotRef>,
        rights: Rights,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.derive_raw(
            channel.into(),
            rights,
            (ResourceKind::ChanSend as u16) as u8,
            result,
            None,
        )
    }

    /// Mint an inert, one-shot proof that `executable` authorized moving the
    /// exact held `supply` in this same descriptor. The kernel fixes the
    /// witness rights and requires its transfer to precede the supply move.
    pub fn invocation_witness(
        &mut self,
        executable: SlotRef,
        supply: SlotRef,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; kore_abi::session::InvocationWitnessReq::WIRE];
        req[0..4].copy_from_slice(&supply.index().to_le_bytes());
        self.derive_raw(
            executable,
            Rights::from_bits_truncate(kore_abi::session::INVOCATION_WITNESS_RIGHTS.bits()),
            (ResourceKind::InvocationWitness as u16) as u8,
            result,
            Some(&req),
        )
    }

    /// `Derive × Channel → ChanRecv` (`kind = ChanRecv`): carve the
    /// **recv** endpoint off a staged [`derive_channel`](Self::derive_channel)
    /// result — the end a rendezvous post (and a pager registration)
    /// binds.  Arg-free, the [`derive_chan_send`](Self::derive_chan_send)
    /// sibling.
    pub fn derive_chan_recv(
        &mut self,
        channel: impl Into<SlotRef>,
        rights: Rights,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        self.derive_raw(
            channel.into(),
            rights,
            (ResourceKind::ChanRecv as u16) as u8,
            result,
            None,
        )
    }

    /// `Derive × AddrSpace` (`kind = ChanRecv`,
    /// [`kore_abi::session::PagerReq`], the pager-registration verb): register the
    /// channel whose recv end lives in `chan_recv` as the caller space's
    /// **managing pager** — the kernel rings its standing recv post (with
    /// a fault [`Notice`](kore_abi::session::Notice)) on every fault
    /// against a paged-out page.  Mints no child: `result` must name a
    /// scratch slot (it is left vacant).
    pub fn register_pager(
        &mut self,
        space: SlotRef,
        chan_recv: SlotRef,
        result: SlotRef,
    ) -> Result<(), BuildError> {
        let mut req = [0u8; 8];
        req[0..4].copy_from_slice(&chan_recv.index().to_le_bytes());
        self.derive_raw(
            space,
            Rights::empty(),
            (ResourceKind::ChanRecv as u16) as u8,
            result,
            Some(&req),
        )
        .map(|_| ())
    }

    /// `Revoke × REVOKE_KIND_PAGED` ([`kore_abi::session::UnshareReq`],
    /// the paged-out-marker install — the lessor's **eviction verb**): return the
    /// leased `view` like a plain [`revoke`](Self::revoke), but leave
    /// **paged-out markers** where its leaves were, so the tenant's next
    /// touch routes to the registered pager.  `lease` names the owning
    /// lease's slot (a read edge, MODIFY-gated) whose share bookkeeping
    /// the eviction updates; the restore is an ordinary
    /// [`share_cpu`](Self::share_cpu) on that same lease.
    pub fn unshare_paged(
        &mut self,
        view: impl Into<SlotRef>,
        lease: impl Into<SlotRef>,
    ) -> Result<(), BuildError> {
        let mut req = [0u8; 8];
        req[0..4].copy_from_slice(&lease.into().index().to_le_bytes());
        let arg = self.stage_arg(&req)?;
        self.push(Inv {
            op: OpCode::Revoke as u16,
            requested: 0,
            kind: kore_abi::session::REVOKE_KIND_PAGED,
            subject: view.into().index(),
            result: 0,
            arg,
        })
    }

    /// `Revoke × REVOKE_KIND_UNSHARE` — retire one held leased `view` and
    /// hard-unmap exactly its own mapping, leaving every sibling share
    /// untouched. `lease` names the held immediate parent. It is a routing
    /// and lineage witness rather than mutation authority, so this exact
    /// flavor requires no `MODIFY` right on the lease (the view itself must
    /// carry `REVOKE`). A stale view cannot tear down a replacement mapping
    /// at the same VA.
    pub fn unshare(
        &mut self,
        view: impl Into<SlotRef>,
        lease: impl Into<SlotRef>,
    ) -> Result<(), BuildError> {
        let mut req = [0u8; 8];
        req[0..4].copy_from_slice(&lease.into().index().to_le_bytes());
        let arg = self.stage_arg(&req)?;
        self.push(Inv {
            op: OpCode::Revoke as u16,
            requested: 0,
            kind: kore_abi::session::REVOKE_KIND_UNSHARE,
            subject: view.into().index(),
            result: 0,
            arg,
        })
    }

    /// `Revoke × REVOKE_KIND_UNSHARE_IOVA` — retire one held IOMMU
    /// `view` and remove exactly its own generational window from the held
    /// `MemLease` or `MmioRegion` parent. The parent must carry `MODIFY`; direct
    /// lineage is re-authenticated, so another writable backing cannot be used as
    /// a teardown oracle.
    pub fn unshare_iova(
        &mut self,
        view: impl Into<SlotRef>,
        parent: impl Into<SlotRef>,
    ) -> Result<(), BuildError> {
        let mut req = [0u8; 8];
        req[0..4].copy_from_slice(&parent.into().index().to_le_bytes());
        let arg = self.stage_arg(&req)?;
        self.push(Inv {
            op: OpCode::Revoke as u16,
            requested: 0,
            kind: kore_abi::session::REVOKE_KIND_UNSHARE_IOVA,
            subject: view.into().index(),
            result: 0,
            arg,
        })
    }

    /// `Revoke × REVOKE_KIND_RETURN` (the lessor's take-back — mm's
    /// `Release`): return the leased `view` like a plain
    /// [`revoke`](Self::revoke), unmap **every leaf its receivers
    /// installed over the window** (any task, any space — the
    /// window-scoped clear), and receive the window's kernel-vouched
    /// `(physical base, len)` as a bind-inert `GeomReport` in `result`.
    /// `lease` names the owning lease's slot (a read edge, MODIFY-gated —
    /// the return's authority basis, since a grant's rights carry no
    /// `REVOKE`).  The hard sibling of [`unshare_paged`](Self::unshare_paged).
    pub fn return_grant(
        &mut self,
        view: impl Into<SlotRef>,
        lease: impl Into<SlotRef>,
        result: SlotRef,
    ) -> Result<Staged, BuildError> {
        let mut req = [0u8; 8];
        req[0..4].copy_from_slice(&lease.into().index().to_le_bytes());
        let arg = self.stage_arg(&req)?;
        self.push(Inv {
            op: OpCode::Revoke as u16,
            requested: 0,
            kind: kore_abi::session::REVOKE_KIND_RETURN,
            subject: view.into().index(),
            result: result.index(),
            arg,
        })?;
        Ok(Staged(result))
    }

    /// Exact byte length of the frozen descriptor this builder will emit.
    ///
    /// **The park block** (the kore-abi flag-gated convention): a
    /// [`park_standing`](Self::park_standing) target emits a `ParkReq`
    /// at the argument region's front, a [`wait_sleep`](Self::wait_sleep)
    /// deadline a `SleepReq` right after it — and every staged
    /// invocation's `arg` offset is relocated past the block, so staging
    /// order never constrains parking.
    pub fn encoded_len(&self) -> Result<usize, BuildError> {
        if self.has_effect && (self.count != 1 || !self.flags.is_empty()) {
            return Err(BuildError::EffectMustStandAlone);
        }
        if let Some((alt_chan, _, _)) = self.park_alt {
            let Some((chan, _, _)) = self.park else {
                return Err(BuildError::BadParkShape);
            };
            if self.count != 0 || chan == alt_chan {
                return Err(BuildError::BadParkShape);
            }
        }
        let block = self.park.map_or(0, |_| 16)
            + self.park_alt.map_or(0, |_| 16)
            + self.sleep.map_or(0, |_| 8)
            + self.irq.map_or(0, |_| kore_abi::session::IrqParkReq::WIRE);
        Ok(8 + self.count * 16 + block + self.args_len)
    }

    /// Write the finished descriptor (`header ‖ invocations ‖ [park
    /// block ‖] args`) into `out` — ultimately either the architecture's
    /// caller-saved GPR carrier or the task's arena descriptor scratch — and
    /// return its exact length.
    ///
    /// [`encoded_len`](Self::encoded_len) lets the transport choose that
    /// carrier before writing a byte; both paths then use this one encoder.
    pub fn finish(&self, out: &mut [u8]) -> Result<usize, BuildError> {
        let need = self.encoded_len()?;
        let block = self.park.map_or(0, |_| 16)
            + self.park_alt.map_or(0, |_| 16)
            + self.sleep.map_or(0, |_| 8)
            + self.irq.map_or(0, |_| kore_abi::session::IrqParkReq::WIRE);
        if out.len() < need {
            return Err(BuildError::OutTooSmall);
        }
        out[0..2].copy_from_slice(&(self.count as u16).to_le_bytes());
        out[2..4].copy_from_slice(&DESCRIPTOR_ABI.to_le_bytes());
        out[4..6].copy_from_slice(&self.flags.bits().to_le_bytes());
        out[6..8].copy_from_slice(&0u16.to_le_bytes());
        let mut at = 8;
        for inv in &self.invs[..self.count] {
            // Relocate arg offsets past the park block (`NO_ARG` rides).
            let arg = if inv.arg == Invocation::NO_ARG {
                Invocation::NO_ARG
            } else {
                inv.arg + block as u32
            };
            out[at..at + 2].copy_from_slice(&inv.op.to_le_bytes());
            out[at + 2] = inv.requested;
            out[at + 3] = inv.kind;
            out[at + 4..at + 8].copy_from_slice(&inv.subject.to_le_bytes());
            out[at + 8..at + 12].copy_from_slice(&inv.result.to_le_bytes());
            out[at + 12..at + 16].copy_from_slice(&arg.to_le_bytes());
            at += 16;
        }
        if let Some((chan, slot, recv_tag)) = self.park {
            // ParkReq: chan@0 ‖ slot@4 ‖ recv_tag@6 ‖ reserved@10..16.
            out[at..at + 4].copy_from_slice(&chan.to_le_bytes());
            out[at + 4..at + 6].copy_from_slice(&slot.to_le_bytes());
            out[at + 6..at + 10].copy_from_slice(&recv_tag.to_le_bytes());
            out[at + 10..at + 16].fill(0);
            at += 16;
        }
        if let Some((chan, slot, recv_tag)) = self.park_alt {
            out[at..at + 4].copy_from_slice(&chan.to_le_bytes());
            out[at + 4..at + 6].copy_from_slice(&slot.to_le_bytes());
            out[at + 6..at + 10].copy_from_slice(&recv_tag.to_le_bytes());
            out[at + 10..at + 16].fill(0);
            at += 16;
        }
        if let Some(deadline_ns) = self.sleep {
            out[at..at + 8].copy_from_slice(&deadline_ns.to_le_bytes());
            at += 8;
        }
        if let Some(route) = self.irq {
            out[at..at + 4].copy_from_slice(&route.index().to_le_bytes());
            out[at + 4..at + 8].fill(0);
            at += kore_abi::session::IrqParkReq::WIRE;
        }
        out[at..at + self.args_len].copy_from_slice(&self.args[..self.args_len]);
        Ok(need)
    }
}

impl Default for SessionBuilder {
    fn default() -> Self {
        Self::new()
    }
}
