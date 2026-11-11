//! Process-lifetime owner for native join capabilities.
//!
//! Clients exchange bounded, generation-checked identifiers through shared
//! records. Only the controller touches native tickets. A detached record is
//! reusable only after native exit and successful reclamation. Storage is
//! supplied by the process and must outlive every client and child.
//!
//! # Every wait is a futex park
//!
//! Nothing here polls. The controller thread waits on its hub word
//! ([`futex::Hub`]), and everything that needs it — a create request, a
//! child's end, a wake relayed through it — stores, then wakes that word, the
//! order [`futex`] requires. A client waits on its record's event word, which
//! the controller bumps and wakes after every transition a client can wait
//! for. A child announces its end through the hook [`thread::exit`] runs, on
//! every way a thread ends that runs code; the kernel's exit commits a moment
//! later, an edge no store can announce, so the controller reaps through a
//! park whose join arm watches that child beside its word
//! ([`futex::wait_or_end`]).
//!
//! # The controller is its threads' hub
//!
//! The star's rule (`res/SPMD.md` §3) for threads that come and go. The
//! starter mints the controller's doorbell, keeps a send into it (an adopted
//! peer), and founds it holding that doorbell, its own kick, the starter's
//! kick, the host's wake, the heap and a channel-space grant. For each child
//! the controller mints a doorbell, keeps the send (an adopted peer, withdrawn
//! before the child's band can be reused), and founds the child holding it, a
//! send into the controller, the starter's kick, the host's wake and the heap.
//! Children and the starter relay through the controller's hub, so a wake
//! between two threads neither can kick directly still arrives: the controller
//! sweeps ([`futex::sweep`]).
//!
//! Both the controller and every child take a founder's arena (the
//! founding granule of capability slots): the controller holds each live child's handle,
//! its stack and runtime-block regions, its staged space and the root and send
//! of its doorbell until the reap, a whole process's worth of them, and a
//! child may found an SPMD gang of its own.
//!
//! # Degrade, never wedge
//!
//! A starter with no channel space founds a controller without a doorbell:
//! neither it nor its children can park, and each of their waits takes the
//! deadline park every thread can ([`pace::cede`]). A client the controller
//! cannot kick waits in bounded slices ([`UNREACHED_SLICE`]). A thread that
//! cannot wake a parking controller is refused a create, which would
//! otherwise wait on a request nothing announced.
//!
//! [`thread::exit`]: crate::thread::exit
//! [`pace::cede`]: crate::pace::cede

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use core::time::Duration;

use super::futex::{self, Hub, WaitOutcome};

const FREE: u64 = 0;
const BUILD: u64 = 1;
const REQUEST: u64 = 2;
const RUNNING: u64 = 3;
const JOIN: u64 = 4;
const DETACHED: u64 = 5;
const COMPLETE: u64 = 6;
const JOIN_READY: u64 = 7;
const CONSUME: u64 = 8;
const FAILED: u64 = 9;
const RETIRED: u64 = 15;
const MASK: u64 = 15;
const COUNT: usize = super::thread::MAX_THREADS as usize - 2;

/// Lifecycle of the controller thread itself, in [`Controller::started`]:
/// not started, being founded, or started — parking on its doorbell, or
/// without one, taking the deadline park.
const IDLE: u32 = 0;
const STARTING: u32 = 1;
const PARKING: u32 = 2;
#[cfg_attr(
    not(target_os = "none"),
    expect(dead_code, reason = "only a metal founding starts")
)]
const POLLING: u32 = 3;

/// How long one park lasts for a client the controller cannot kick — a
/// thread it holds no send into, such as an initial thread launched without
/// a self-kick. Nothing can end such a park early, so its slice is the
/// latency it pays, and this is the cadence those waits paced themselves
/// at before the controller parked (`pace::CEDE_NS`): a client the wake
/// fabric cannot reach is no worse off than it was. Shorter would wake an
/// unreachable joiner more often for nothing; longer would delay it.
const UNREACHED_SLICE: Duration = Duration::from_nanos(crate::pace::CEDE_NS);

/// How long the controller rides its doorbell after a reap the kernel did
/// not complete — a refused lifecycle observation or reclamation, which no
/// later event is bound to retry. The joiner's own pacing for a join that
/// cannot establish its park (`thread::join_park`, two milliseconds).
#[cfg_attr(
    not(target_os = "none"),
    expect(dead_code, reason = "reaping is metal-only")
)]
const REAP_RETRY: Duration = Duration::from_millis(2);

/// An identifier within one controller, never a capability-slot number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Id {
    controller: usize,
    slot: usize,
    generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Unavailable,
    Full,
    Invalid,
    Busy,
    Deadlock,
    /// Native termination occurred without the callback returning a value.
    Terminated,
}

#[derive(Clone, Copy)]
struct Start {
    call: fn(usize) -> usize,
    argument: usize,
    /// Read by the controller thread, after the reap.
    #[cfg_attr(not(target_os = "none"), allow(dead_code))]
    retire: Option<fn(usize)>,
    /// The controller this record belongs to, for the child's hooks.
    home: usize,
}
fn empty(_: usize) -> usize {
    0
}

/// One managed thread's shared record, on its own interference span: its
/// creator, its joiner, the child and the controller each write it, and a
/// waiting client's monitored spin sleeps on its event word's line, which a
/// neighbour's traffic would end for nothing. The span is the one the
/// barrier isolates against (`barrier::Line`).
#[repr(align(128))]
struct Record {
    state: AtomicU64,
    /// Bumped by the controller after each transition a client can wait
    /// for; clients park on it. An event count: the state is the truth.
    event: AtomicU32,
    /// Set by the child's exit hook: it has stopped running its own code
    /// and is committing its end.
    ending: AtomicU32,
    start: UnsafeCell<Start>,
    value: AtomicUsize,
    identity: AtomicUsize,
    returned: AtomicUsize,
}
// The BUILD claimant alone writes start. REQUEST publishes it; it stays
// immutable until the controller has joined the child and released the record.
unsafe impl Sync for Record {}
impl Record {
    const fn new() -> Self {
        Self {
            state: AtomicU64::new(FREE),
            event: AtomicU32::new(0),
            ending: AtomicU32::new(0),
            start: UnsafeCell::new(Start {
                call: empty,
                argument: 0,
                retire: None,
                home: 0,
            }),
            value: AtomicUsize::new(0),
            identity: AtomicUsize::new(0),
            returned: AtomicUsize::new(0),
        }
    }
    fn release(&self, generation: u64) {
        self.state.store(
            generation.checked_add(16).unwrap_or(generation | RETIRED),
            Ordering::Release,
        );
    }
    /// Publish a transition a client may be parked on: the state was
    /// stored; now the event moves, then its waiters are woken.
    #[cfg_attr(not(target_os = "none"), allow(dead_code))]
    fn announce(&self) {
        self.event.fetch_add(1, Ordering::SeqCst);
        futex::wake_all(&self.event);
    }
}

/// Named process-owned storage. Start once during application startup and
/// share its static reference with workers. This consumes one native thread
/// for the process lifetime. No native ticket escapes its owner.
pub struct Controller {
    /// [`IDLE`], [`STARTING`], [`PARKING`] or [`POLLING`]; clients park on
    /// it while the controller is being founded.
    started: AtomicU32,
    /// What the controller thread waits on, and what its threads relay
    /// through.
    hub: Hub,
    /// The controller thread's band, plus one (`0` until it is founded).
    band: AtomicU32,
    /// The band of the thread that started the controller, plus one.
    #[cfg_attr(not(target_os = "none"), allow(dead_code))]
    starter: AtomicU32,
    /// The bands whose doorbells the controller can kick, a bit each: the
    /// clients its wakes reach. A child's bit is set once it is founded and
    /// cleared before its band can be reused.
    reach: AtomicU32,
    records: [Record; COUNT],
}
impl Default for Controller {
    fn default() -> Self {
        Self::new()
    }
}
// The reach bitmap has one bit per band.
const _: () = assert!(super::thread::MAX_THREADS <= u32::BITS);

impl Controller {
    pub const fn new() -> Self {
        Self {
            started: AtomicU32::new(IDLE),
            hub: Hub::new(),
            band: AtomicU32::new(0),
            starter: AtomicU32::new(0),
            reach: AtomicU32::new(0),
            records: [const { Record::new() }; COUNT],
        }
    }

    /// Start the owner. Concurrent/repeated calls do not create another owner.
    ///
    /// The calling thread becomes the controller's starter: it mints the
    /// controller's wake channel from its channel-space grant and holds that
    /// channel's root, as a `DoorbellWorker`'s parent holds its worker's, so
    /// start from a thread that outlives the controller's use — the initial
    /// thread, or a program's main.
    pub fn start(&'static self) -> Result<(), Error> {
        self.started
            .compare_exchange(IDLE, STARTING, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::Busy)?;
        let mode = self.found();
        self.started.store(mode.unwrap_or(IDLE), Ordering::Release);
        futex::wake_all(&self.started);
        mode.map(|_| ()).ok_or(Error::Unavailable)
    }

    /// How the started controller waits, waiting out its founding; `None`
    /// when it is not started.
    fn mode(&self) -> Option<u32> {
        loop {
            match self.started.load(Ordering::Acquire) {
                IDLE => return None,
                STARTING => {
                    // The starter wakes every thread it can when it is done;
                    // a thread it cannot reach rides its slice.
                    park(&self.started, STARTING, Some(UNREACHED_SLICE));
                }
                mode => return Some(mode),
            }
        }
    }

    /// Start a callback. Any pointed-to argument storage must remain valid
    /// through callback return (or join); the integer itself grants no authority.
    /// A safe Rust callback must validate any pointer it chooses to dereference.
    pub fn create(&'static self, call: fn(usize) -> usize, argument: usize) -> Result<Id, Error> {
        self.create_with_cleanup(call, argument, None)
    }

    /// Transfer callback context custody after a successful create. `retire`
    /// runs once on the controller after native reclamation, including when the
    /// callback exits without returning. On error custody remains with caller.
    /// The cleanup must not make blocking calls back into this controller.
    ///
    /// Call it from a thread that can wake the controller — its starter, or a
    /// thread it created. A parking controller sleeps until something wakes
    /// it, so any other thread is refused [`Error::Unavailable`] rather than
    /// left waiting on a request nothing announced.
    pub fn create_with_cleanup(
        &'static self,
        call: fn(usize) -> usize,
        argument: usize,
        retire: Option<fn(usize)>,
    ) -> Result<Id, Error> {
        let mode = self.mode().ok_or(Error::Unavailable)?;
        if mode == PARKING && !self.wakes_controller() {
            return Err(Error::Unavailable);
        }
        for (slot, record) in self.records.iter().enumerate() {
            let old = record.state.load(Ordering::Acquire);
            if old & MASK != FREE {
                continue;
            }
            if record
                .state
                .compare_exchange(old, old | BUILD, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                continue;
            }
            // SAFETY: exclusive BUILD claim; prior child was joined before FREE.
            unsafe {
                *record.start.get() = Start {
                    call,
                    argument,
                    retire,
                    home: self as *const Self as usize,
                };
            }
            record.identity.store(0, Ordering::Relaxed);
            record.value.store(0, Ordering::Relaxed);
            record.returned.store(0, Ordering::Relaxed);
            record.ending.store(0, Ordering::Relaxed);
            record.state.store(old | REQUEST, Ordering::Release);
            self.hub.signal();
            let state = self.settle(record, |state| state != old | REQUEST);
            if state == old | FAILED {
                record.release(old);
                return Err(Error::Unavailable);
            }
            return Ok(Id {
                controller: self as *const Self as usize,
                slot,
                generation: old,
            });
        }
        Err(Error::Full)
    }

    /// Whether this thread holds a send into the controller's doorbell.
    fn wakes_controller(&self) -> bool {
        self.band
            .load(Ordering::Acquire)
            .checked_sub(1)
            .is_some_and(futex::reaches)
    }

    /// Park until `record`'s state satisfies `done`, and return that state.
    fn settle(&self, record: &Record, done: impl Fn(u64) -> bool) -> u64 {
        loop {
            // The event count before the state: a transition published after
            // this load moves the count, which the park re-reads once it is
            // registered (`futex`'s order), so none is missed.
            let seen = record.event.load(Ordering::SeqCst);
            let state = record.state.load(Ordering::Acquire);
            if done(state) {
                return state;
            }
            let reached = futex::current_thread()
                .is_some_and(|band| self.reach.load(Ordering::SeqCst) & (1 << band) != 0);
            park(&record.event, seen, (!reached).then_some(UNREACHED_SLICE));
        }
    }

    fn record(&self, id: Id) -> Result<&Record, Error> {
        if id.controller != self as *const Self as usize {
            return Err(Error::Invalid);
        }
        let record = self.records.get(id.slot).ok_or(Error::Invalid)?;
        if record.state.load(Ordering::Acquire) & !MASK != id.generation {
            return Err(Error::Invalid);
        }
        Ok(record)
    }

    /// Identify the current managed native thread. Initial/unmanaged threads
    /// are not represented by a managed record.
    pub fn current(&self) -> Option<Id> {
        #[cfg(target_os = "none")]
        {
            let identity = crate::entry::geometry::thread_owner_identity()?;
            for (slot, record) in self.records.iter().enumerate() {
                let state = record.state.load(Ordering::Acquire);
                if matches!(state & MASK, REQUEST | RUNNING | JOIN | DETACHED)
                    && record.identity.load(Ordering::Acquire) == identity
                {
                    return Some(Id {
                        controller: self as *const Self as usize,
                        slot,
                        generation: state & !MASK,
                    });
                }
            }
        }
        None
    }

    /// Join from any native worker. Exactly one joiner or detacher wins.
    /// Returns the callback value only after native resources are reclaimed.
    pub fn join(&self, id: Id) -> Result<usize, Error> {
        let record = self.record(id)?;
        #[cfg(target_os = "none")]
        if record.identity.load(Ordering::Acquire) != 0
            && Some(record.identity.load(Ordering::Relaxed))
                == crate::entry::geometry::thread_owner_identity()
        {
            return Err(Error::Deadlock);
        }
        loop {
            let old = record.state.load(Ordering::Acquire);
            if old & !MASK != id.generation {
                return Err(Error::Invalid);
            }
            let next = match old & MASK {
                RUNNING => JOIN,
                COMPLETE => CONSUME,
                FREE | RETIRED => return Err(Error::Invalid),
                _ => return Err(Error::Busy),
            };
            if record
                .state
                .compare_exchange(
                    old,
                    id.generation | next,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_err()
            {
                continue;
            }
            if next == JOIN {
                self.settle(record, |state| state == id.generation | JOIN_READY);
            }
            // Acquire completion before reading the value it publishes. The owner
            // is a different thread from the callback on weakly ordered ISAs.
            let returned = record.returned.load(Ordering::Acquire) != 0;
            let result = record.value.load(Ordering::Relaxed);
            record.release(id.generation);
            return if returned {
                Ok(result)
            } else {
                Err(Error::Terminated)
            };
        }
    }

    /// Give up joining; the owner still reaps the native thread after exit.
    pub fn detach(&self, id: Id) -> Result<(), Error> {
        let record = self.record(id)?;
        loop {
            let old = record.state.load(Ordering::Acquire);
            if old & !MASK != id.generation {
                return Err(Error::Invalid);
            }
            let next = match old & MASK {
                RUNNING => DETACHED,
                COMPLETE => CONSUME,
                FREE | RETIRED => return Err(Error::Invalid),
                _ => return Err(Error::Busy),
            };
            if record
                .state
                .compare_exchange(
                    old,
                    id.generation | next,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_err()
            {
                continue;
            }
            if next == CONSUME {
                record.release(id.generation);
            }
            return Ok(());
        }
    }
}

/// One wait on `word`, still holding `expected`: the futex park, or — on a
/// thread that cannot park at all — the deadline park every thread can take,
/// which a spin hint stands in for only off metal.
fn park(word: &AtomicU32, expected: u32, slice: Option<Duration>) {
    if futex::wait(word, expected, slice) == WaitOutcome::Unsupported {
        pause();
    }
}

fn pause() {
    if !crate::pace::cede() {
        core::hint::spin_loop();
    }
}

/// The managed thread's entry: announce its end however it comes, relay
/// through the controller, run the callback.
fn child_entry(argument: usize) {
    // SAFETY: controller storage is static; record cannot be reused until join.
    let record = unsafe { &*(argument as *const Record) };
    #[cfg(target_os = "none")]
    {
        crate::thread::on_exit(child_ending, argument);
        record.identity.store(
            crate::entry::geometry::thread_owner_identity().unwrap_or(0),
            Ordering::Release,
        );
    }
    // Acquire the request publication (or the owner's subsequent state)
    // before reading callback storage. It remains immutable through native join.
    let _ = record.state.load(Ordering::Acquire);
    let start = unsafe { *record.start.get() };
    // SAFETY: `home` names the static controller whose record this is.
    let _ = futex::adopt_hub(&unsafe { &*(start.home as *const Controller) }.hub);
    record
        .value
        .store((start.call)(start.argument), Ordering::Release);
    record.returned.store(1, Ordering::Release);
}

/// The child's exit hook: say the thread is ending, then wake the
/// controller to reap it.
#[cfg(target_os = "none")]
fn child_ending(argument: usize) {
    // SAFETY: as in `child_entry`; the record outlives its child.
    let record = unsafe { &*(argument as *const Record) };
    record.ending.store(1, Ordering::SeqCst);
    // SAFETY: `start` is immutable until the controller has reaped this
    // thread, which it cannot do before this hook returns.
    let start = unsafe { *record.start.get() };
    let controller = unsafe { &*(start.home as *const Controller) };
    controller.hub.signal();
}

#[cfg(target_os = "none")]
mod owner {
    //! The controller thread and its founding.

    use core::sync::atomic::Ordering;

    use kore_abi::cap::{ResourceKind, Rights};
    use kore_abi::session::SlotRef;
    use kore_abi::task::role;

    use super::{
        COMPLETE, COUNT, Controller, DETACHED, JOIN, JOIN_READY, MASK, PARKING, POLLING,
        REAP_RETRY, RUNNING, Record, child_entry,
    };
    use crate::cap::{
        CHANNEL_ENDPOINT_RIGHTS, ChanSendToken, ChannelRoot, ChannelSpaceToken, verbs,
    };
    use crate::futex::{self, PEER_BASE, WaitOutcome};
    use crate::thread::{self, JoinScratch, Thread, WAKE_SEND_RIGHTS, WorkerCap};

    /// The most capabilities either founding here endows: the controller's
    /// doorbell, its own kick, the heap, the host's wake, a channel-space
    /// grant and the starter's kick.
    const MAX_ENDOW: usize = 6;

    /// A founding's endowment, filled in order.
    struct Endowment {
        caps: [WorkerCap; MAX_ENDOW],
        len: usize,
    }

    impl Endowment {
        fn new() -> Self {
            let vacant = WorkerCap {
                role: 0,
                index: 0,
                subject: SlotRef::from_wire(0),
                kind: 0,
                rights: Rights::empty(),
            };
            Self {
                caps: [vacant; MAX_ENDOW],
                len: 0,
            }
        }

        /// Append `cap`, if any; `true` when it is now in the endowment.
        fn push(&mut self, cap: Option<WorkerCap>) -> bool {
            let (Some(cap), Some(at)) = (cap, self.caps.get_mut(self.len)) else {
                return false;
            };
            *at = cap;
            self.len += 1;
            true
        }

        fn as_slice(&self) -> &[WorkerCap] {
            &self.caps[..self.len]
        }
    }

    /// A send at `DOORBELL@index`, derived from `subject` with `rights`.
    fn send(index: u16, subject: SlotRef, rights: Rights) -> WorkerCap {
        WorkerCap {
            role: role::DOORBELL,
            index,
            subject,
            kind: ResourceKind::ChanSend as u16 as u8,
            rights,
        }
    }

    /// A thread's own doorbell (`DOORBELL@0`), derived from its channel's root.
    fn doorbell(root: &ChannelRoot) -> WorkerCap {
        WorkerCap {
            role: role::DOORBELL,
            index: 0,
            subject: root.slot(),
            kind: ResourceKind::ChanRecv as u16 as u8,
            rights: CHANNEL_ENDPOINT_RIGHTS,
        }
    }

    /// One thread's wake channel: its root and the send its founder keeps.
    /// The founder's own recv is surplus — the thread's recv is a fresh
    /// derive of the root at its founding (the star's rule).
    struct Wires {
        root: ChannelRoot,
        kick: ChanSendToken,
    }

    impl Wires {
        fn mint(space: &ChannelSpaceToken) -> Option<Self> {
            let pair = verbs::mint_channel(space).ok()?;
            let _ = pair.recv.try_retire();
            Some(Self {
                root: pair.root,
                kick: pair.send,
            })
        }

        fn retire(self) {
            let _ = self.kick.try_retire();
            let _ = self.root.try_retire();
        }
    }

    impl Controller {
        /// Found the controller thread from the calling thread: the mode it
        /// will wait in, or `None` when it could not be founded.
        pub(super) fn found(&'static self) -> Option<u32> {
            let starter = futex::current_thread()? as u32;
            self.starter.store(starter + 1, Ordering::Release);
            let rt = crate::entry::thread_context()?;
            let space = rt
                .bootstrap()
                .handle(role::CHANNEL_SPACE, 0)
                .and_then(verbs::claim_channel_space);
            let wires = space.as_ref().and_then(Wires::mint);
            let own_kick = thread::own_kick();

            let mut endowment = Endowment::new();
            if let Some(wires) = &wires {
                endowment.push(Some(doorbell(&wires.root)));
                // The controller derives every child's send into itself from
                // this one, so it keeps DERIVE.
                endowment.push(Some(send(
                    futex::OWN_KICK,
                    wires.root.slot(),
                    WAKE_SEND_RIGHTS,
                )));
            }
            endowment.push(thread::heap_endowment());
            // Peer 0. For a starter that is thread 0 it is that thread's own
            // kick, and the starter is reached through it.
            let peer_zero = endowment.push(thread::wake_endowment());
            endowment.push(thread::channel_space_endowment());
            // Any other starter is reached at its own band's peer index.
            let reached = if starter == 0 {
                own_kick.is_some() && peer_zero
            } else {
                endowment.push(
                    own_kick.map(|kick| send(PEER_BASE + starter as u16, kick, WAKE_SEND_RIGHTS)),
                )
            };
            let Some(ticket) = thread::spawn_founder_with_arg(
                super::controller_entry,
                self as *const Self as usize,
                endowment.as_slice(),
            ) else {
                if let Some(wires) = wires {
                    wires.retire();
                }
                return None;
            };
            let band = ticket.band_index();
            self.band.store(band + 1, Ordering::Release);
            if reached {
                self.reach.fetch_or(1 << starter, Ordering::SeqCst);
            }
            // This owner deliberately lives until process teardown. Managed
            // children never use native detach: their tickets are retained.
            ticket.detach();
            // The claimed grant is a witness, not a reservation: dropping it
            // retires nothing (the star's rule, `star::Wires`).
            drop(space);
            let Some(wires) = wires else {
                return Some(POLLING);
            };
            // Wake the controller directly, and relay through it.
            if !futex::adopt_peer(band, wires.kick.slot()) {
                wires.retire();
                return Some(POLLING);
            }
            let _ = futex::adopt_hub(&self.hub);
            // The channel lives as long as the controller does: its root and
            // this thread's send stay in this thread's arena for good.
            core::mem::forget(wires);
            Some(PARKING)
        }
    }

    /// A managed child the controller founded and will reap.
    struct Child {
        thread: Thread,
        band: u32,
        /// Its doorbell's root and the controller's send into it; `None`
        /// when the controller has no channel space to mint them.
        wires: Option<Wires>,
    }

    /// The controller thread's own state, on its stack for the process life.
    pub(super) struct Owner {
        controller: &'static Controller,
        band: u32,
        /// The grant children's doorbells are minted from.
        space: Option<ChannelSpaceToken>,
        /// This thread's own kick: every child's send into it derives from it.
        kick: Option<SlotRef>,
        /// The starter's band and the send into it, when the starter is not
        /// thread 0 (which children reach as peer 0).
        starter: Option<(u32, SlotRef)>,
        /// One report landing, reused serially by every probe, join arm and
        /// reap: the controller never spends an arena slot to observe.
        scratch: Option<JoinScratch>,
        children: [Option<Child>; COUNT],
        /// The hub's relay count at the last sweep.
        swept: u32,
    }

    /// What one reap attempt came to.
    enum Reap {
        /// Nothing left to reap.
        Done,
        /// The child announced its end and has not committed it.
        Pending,
        /// The kernel refused an observation or the reclamation.
        Failed,
    }

    impl Owner {
        pub(super) fn new(controller: &'static Controller) -> Self {
            let band = futex::current_thread().map_or(u32::MAX, |band| band as u32);
            let rt = crate::entry::thread_context();
            let boot = rt.as_ref().map(|rt| rt.bootstrap());
            let space = boot
                .as_ref()
                .and_then(|boot| boot.handle(role::CHANNEL_SPACE, 0))
                .and_then(verbs::claim_channel_space);
            // Derivable, or no child is handed a send into this thread.
            let kick = thread::own_kick();
            let starter = controller
                .starter
                .load(Ordering::Acquire)
                .checked_sub(1)
                .filter(|&starter| starter != 0)
                .and_then(|starter| {
                    let send = boot
                        .as_ref()?
                        .handle(role::DOORBELL, PEER_BASE + starter as u16)?;
                    Some((starter, send))
                });
            Self {
                controller,
                band,
                space,
                kick,
                starter,
                scratch: JoinScratch::reserve(),
                children: [const { None }; COUNT],
                swept: controller.hub.relays(),
            }
        }

        /// Serve the records for the process's life.
        pub(super) fn serve(mut self) -> ! {
            let controller = self.controller;
            loop {
                let seen = controller.hub.word().load(Ordering::SeqCst);
                let owed = controller.hub.relays();
                if owed != self.swept {
                    self.swept = owed;
                    futex::sweep();
                }
                let mut watch = None;
                let mut failed = false;
                for (slot, record) in controller.records.iter().enumerate() {
                    let state = record.state.load(Ordering::Acquire);
                    if state & MASK == super::REQUEST {
                        self.found_child(slot, record, state & !MASK);
                    }
                    if record.ending.load(Ordering::Acquire) != 0 {
                        match self.reap(slot, record) {
                            Reap::Done => {}
                            Reap::Pending => {
                                watch.get_or_insert(slot);
                            }
                            Reap::Failed => failed = true,
                        }
                    }
                }
                let word = controller.hub.word();
                let outcome = match (watch, &self.scratch) {
                    (Some(slot), Some(scratch)) if !failed => match &self.children[slot] {
                        Some(child) => futex::wait_or_end(
                            word,
                            seen,
                            None,
                            child.thread.handle(),
                            scratch.report_slot(),
                        ),
                        None => futex::wait_quiet(word, seen, None),
                    },
                    _ => futex::wait_quiet(word, seen, failed.then_some(REAP_RETRY)),
                };
                if outcome == WaitOutcome::Unsupported {
                    super::pause();
                }
            }
        }

        /// Found the child `record` asks for, then publish the outcome.
        fn found_child(&mut self, slot: usize, record: &'static Record, generation: u64) {
            let child = self.spawn(record);
            let next = if child.is_some() {
                RUNNING
            } else {
                super::FAILED
            };
            self.children[slot] = child;
            record.state.store(generation | next, Ordering::Release);
            record.announce();
        }

        fn spawn(&mut self, record: &'static Record) -> Option<Child> {
            let wires = self.space.as_ref().and_then(Wires::mint);
            let mut endowment = Endowment::new();
            if let Some(wires) = &wires {
                endowment.push(Some(doorbell(&wires.root)));
            }
            // A child hands these on to no one (the threads it asks for are
            // founded here, not by it): notice rights only.
            endowment.push(
                self.kick
                    .map(|kick| send(PEER_BASE + self.band as u16, kick, verbs::NOTICE_RIGHTS)),
            );
            endowment.push(self.starter.map(|(starter, kick)| {
                send(PEER_BASE + starter as u16, kick, verbs::NOTICE_RIGHTS)
            }));
            endowment.push(thread::heap_endowment());
            endowment.push(thread::wake_endowment());
            let Some(thread) = thread::spawn_founder_with_arg(
                child_entry,
                record as *const Record as usize,
                endowment.as_slice(),
            ) else {
                if let Some(wires) = wires {
                    wires.retire();
                }
                return None;
            };
            let band = thread.band_index();
            if let Some(wires) = &wires
                && futex::adopt_peer(band, wires.kick.slot())
            {
                self.controller.reach.fetch_or(1 << band, Ordering::SeqCst);
            }
            Some(Child {
                thread,
                band,
                wires,
            })
        }

        /// Reap `slot`'s child if it has ended, then publish its end.
        fn reap(&mut self, slot: usize, record: &Record) -> Reap {
            if self.scratch.is_none() {
                // The landing was not available when this thread started.
                self.scratch = JoinScratch::reserve();
            }
            let (Some(child), Some(scratch)) =
                (self.children[slot].as_ref(), self.scratch.as_mut())
            else {
                return if self.children[slot].is_none() {
                    Reap::Done
                } else {
                    Reap::Failed
                };
            };
            match thread::poll_reserved(&child.thread, scratch) {
                Ok(crate::cap::TaskPollOutcome::Running) => return Reap::Pending,
                Ok(_) => {}
                Err(_) => return Reap::Failed,
            }
            // Withdraw the child's kick before its band can be handed to
            // another thread: no wake may present this send, and no client
            // at that band may count on this controller reaching it.
            futex::forget_peer(child.band);
            self.controller
                .reach
                .fetch_and(!(1 << child.band), Ordering::SeqCst);
            let Some(Child {
                thread,
                band,
                wires,
            }) = self.children[slot].take()
            else {
                return Reap::Done;
            };
            if let Err(error) = thread::join_reserved(thread, scratch) {
                self.children[slot] = Some(Child {
                    thread: error.into_thread(),
                    band,
                    wires,
                });
                return Reap::Failed;
            }
            if let Some(wires) = wires {
                wires.retire();
            }
            // Child can no longer access its context; cleanup also runs
            // for abrupt native exit and detached completion.
            let start = unsafe { *record.start.get() };
            if let Some(retire) = start.retire {
                retire(start.argument);
            }
            record.ending.store(0, Ordering::Relaxed);
            loop {
                let old = record.state.load(Ordering::Acquire);
                let generation = old & !MASK;
                let next = match old & MASK {
                    RUNNING => COMPLETE,
                    JOIN => JOIN_READY,
                    DETACHED => {
                        record.release(generation);
                        break;
                    }
                    // Only this thread moves a founded child's record out of
                    // these three.
                    _ => break,
                };
                if record
                    .state
                    .compare_exchange(old, generation | next, Ordering::Release, Ordering::Acquire)
                    .is_ok()
                {
                    break;
                }
            }
            record.announce();
            Reap::Done
        }
    }
}

fn controller_entry(argument: usize) {
    // SAFETY: start requires static storage.
    let controller = unsafe { &*(argument as *const Controller) };
    #[cfg(target_os = "none")]
    owner::Owner::new(controller).serve();
    #[cfg(not(target_os = "none"))]
    let _ = controller;
}

#[cfg(not(target_os = "none"))]
impl Controller {
    /// Off metal no thread founds another: the controller never starts.
    fn found(&'static self) -> Option<u32> {
        let _ = child_entry;
        let _ = controller_entry;
        None
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    fn completed(controller: &Controller, generation: u64) -> Id {
        let record = &controller.records[0];
        record.value.store(73, Ordering::Relaxed);
        record.returned.store(1, Ordering::Relaxed);
        record.state.store(generation | COMPLETE, Ordering::Release);
        Id {
            controller: controller as *const Controller as usize,
            slot: 0,
            generation,
        }
    }
    #[test]
    fn concurrent_joiners_cannot_consume_or_release_twice() {
        let controller = Controller::new();
        let id = completed(&controller, 0);
        std::thread::scope(|scope| {
            let a = scope.spawn(|| controller.join(id));
            let b = scope.spawn(|| controller.join(id));
            let outcomes = [a.join().unwrap(), b.join().unwrap()];
            assert_eq!(
                outcomes.iter().filter(|result| **result == Ok(73)).count(),
                1
            );
        });
        assert_eq!(controller.records[0].state.load(Ordering::Acquire), 16);
        let next = completed(&controller, 16);
        assert_eq!(controller.join(id), Err(Error::Invalid));
        assert_eq!(controller.join(next), Ok(73));
    }
    #[test]
    fn controller_identity_exit_without_return_and_generation_exhaustion() {
        let controller = Controller::new();
        let other = Controller::new();
        let id = completed(&controller, 0);
        assert_eq!(other.detach(id), Err(Error::Invalid));
        controller.records[0].returned.store(0, Ordering::Relaxed);
        assert_eq!(controller.join(id), Err(Error::Terminated));
        let final_id = completed(&controller, u64::MAX & !MASK);
        assert_eq!(controller.detach(final_id), Ok(()));
        assert_eq!(
            controller.records[0].state.load(Ordering::Acquire),
            u64::MAX
        );
        assert_eq!(controller.join(final_id), Err(Error::Invalid));
    }
    #[test]
    fn a_parked_joiner_is_released_by_the_announced_transition() {
        // The joiner's half of the protocol, with a host thread standing in
        // for the controller: it moves the record JOIN → JOIN_READY and
        // announces it; the joiner returns the value only after that.
        let controller = Controller::new();
        let record = &controller.records[0];
        record.state.store(RUNNING, Ordering::Release);
        let id = Id {
            controller: &controller as *const Controller as usize,
            slot: 0,
            generation: 0,
        };
        std::thread::scope(|scope| {
            let joiner = scope.spawn(|| controller.join(id));
            while record.state.load(Ordering::Acquire) != JOIN {
                std::thread::yield_now();
            }
            record.value.store(5, Ordering::Relaxed);
            record.returned.store(1, Ordering::Release);
            let before = record.event.load(Ordering::SeqCst);
            record.state.store(JOIN_READY, Ordering::Release);
            record.announce();
            assert_eq!(record.event.load(Ordering::SeqCst), before.wrapping_add(1));
            assert_eq!(joiner.join().unwrap(), Ok(5));
        });
        assert_eq!(record.state.load(Ordering::Acquire), 16);
    }
    #[test]
    fn an_unstarted_controller_refuses_and_never_waits() {
        let controller: &'static Controller =
            std::boxed::Box::leak(std::boxed::Box::new(Controller::new()));
        assert_eq!(controller.create(|_| 0, 0), Err(Error::Unavailable));
        // Off metal the founding cannot happen; the refusal leaves it idle
        // and restartable rather than stuck in its founding state.
        assert_eq!(controller.start(), Err(Error::Unavailable));
        assert_eq!(controller.started.load(Ordering::Acquire), IDLE);
        assert_eq!(controller.create(|_| 0, 0), Err(Error::Unavailable));
    }
    #[test]
    fn records_own_their_interference_span() {
        assert_eq!(core::mem::align_of::<Record>(), 128);
        assert_eq!(core::mem::size_of::<Record>(), 128);
    }
}
