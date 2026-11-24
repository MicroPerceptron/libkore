//! The internal keyboard's **relay thread** — the input service's second
//! source (#420, PS/2 epic #414 step P6).
//!
//! ## Why a thread rather than another service row
//!
//! One input service, two devices (the epic's binding ruling). The
//! terminal must receive both keyboards through one router, one focus
//! table, and one line discipline, or the two would compete for the
//! terminal instead of feeding it. A second manifest row would have
//! given the internal keyboard its own router and its own focus, and
//! then required a third thing to merge them.
//!
//! But one **task** parks on one interrupt endpoint, and the two devices
//! interrupt independently: the xHCI controller through MSI, the i8042
//! through an IOAPIC pin. So the service is one task with two threads —
//! the unified task model's (#407) whole point — with thread 0 serving
//! doorbell-parked beside its xHCI drain, and this relay parked on the
//! legacy line.
//!
//! ## What this thread holds
//!
//! Its own capabilities, not thread 0's. Holdings are per task and a
//! thread is a task: sharing an address space shares memory, never
//! authority. That is exactly what makes the I/O-port bitmap safe to key
//! on the running task — a sibling thread that was never granted ports
//! cannot acquire them by sharing a space with one that was — and it is
//! why this thread is founded holding a derived `PlatformDevice` and a
//! derived doorbell send
//! ([`SystemService::worker_endowment`](kore_rt::service::SystemService::worker_endowment)),
//! off the first of which it claims its own ports and establishes its
//! own line.
//!
//! ## The handoff
//!
//! Decoded reports cross to thread 0 through an in-space SPSC queue and
//! a doorbell notice. Thread 0 routes them, because routing state — the
//! subscriber table and the focus — is its own; this thread's job ends
//! at "a key was pressed, in HID's vocabulary". The queue is
//! single-producer/single-consumer because that is literally what it is:
//! this thread pushes, thread 0 pops.
//!
//! The notice is not an optimization. Without it a keystroke on the
//! internal keyboard would sit in the queue until some *unrelated* event
//! woke thread 0 — a USB keystroke, a control request — which on a
//! machine whose only keyboard is this one means forever. Waking on a
//! cadence instead was ruled out at the top of this epic: cadence
//! polling ships nowhere.
//!
//! The queue is memory, so it is shared for free; the notice is
//! *authority*, so it is not. The founder gives the service a send end
//! into its own doorbell (`role::DOORBELL@2`) — but into **thread 0's**
//! arena, which is the only task it knows about at founding. This thread
//! therefore needs its own, derived off that one the same way its device
//! is, and for the same reason: holdings are per task. The first version
//! read `role::DOORBELL@2` straight out of this thread's boot page,
//! where nobody had put anything, and refused at [`rung::NO_KICK`] on a
//! QEMU boot with a perfectly good i8042 already brought up.

use core::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};

use kore_abi::cap::{ResourceKind, Rights};
use kore_abi::keyboard::Locks;
use kore_abi::svc::InputReport;
use kore_abi::task::role;

use kore_driver::Regs;
use kore_i8042::hid::Contribution;
use kore_i8042::{
    Decoded, Decoder, Indicators, Key, KeyEvent, PortAccess, PortRegs, Reply, bring_up,
};
use kore_rt::cap::verbs;
use kore_rt::service::Worker;
use kore_rt::thread::WorkerCap;

use kore_ring::spsc;

/// The **source byte** the internal keyboard's reports carry.
///
/// A distinct band from the xHCI source (which uses root-hub port
/// numbers, small integers from 1) so the two never collide: the terminal
/// keeps a press-synthesis snapshot *per source*, and two devices
/// sharing a byte would diff against each other's held keys.
pub const SOURCE_LEGACY: u8 = 0x40;

/// The architectural i8042 data port.
const PORT_DATA: u16 = 0x60;
/// The architectural i8042 status/command port.
const PORT_STATUS: u16 = 0x64;

/// Reports the handoff queue holds. Small on purpose: it buffers one
/// wake's worth of keystrokes, not a session's — a human cannot outrun
/// it, and a stuck consumer should be visible as drops rather than
/// hidden by depth.
const RELAY_CAPACITY: u32 = 64;

/// The `role::DOORBELL` index of the send end into the **service executor's**
/// doorbell — the self-kick the founder delivers to thread 0.
///
/// Named once because two things read it: the handle lookup that acquires the
/// send end, and the [`wake`](kore_rt::wake) bell rung beside it. The doorbell
/// mesh numbers peers `PEER_BASE + thread`, so this index *is* the target's
/// identity — and if the two came from different constants they could come to
/// mean different threads without anything failing to compile.
const SERVICE_KICK_PEER: u16 = 2;

/// The backstop period on a line that is doing its job — a pure
/// **liveness** check, and the default.
///
/// Nothing waits on this. A keystroke is announced by the interrupt, and
/// the only thing the deadline has left to do is notice if the line ever
/// stops announcing them. One wake a second is what that costs, and on
/// an untouched machine it is the entire steady-state expense of having
/// a keyboard.
const BACKSTOP_LIVENESS_MS: u64 = 1_000;

/// The backstop period once the line has been **caught not delivering**
/// — the fallback, and only ever that.
///
/// Here the deadline *is* the keyboard, so it has to be a typing latency
/// rather than a liveness one. 16 ms is under the threshold where typing
/// stops feeling like typing. #452 is why this mode exists at all: for a
/// week the Yoga's pin was programmed, unmasked, and blocked by the
/// IOMMU, and this period was the whole input path.
const BACKSTOP_DEGRADED_MS: u64 = 16;

/// **The backstop policy**: how long the next park's deadline should be,
/// given what the line has actually been observed to do.
///
/// The rule is that polling is *strictly* a fallback. The deadline
/// starts slow and stays slow; it tightens only on evidence that the
/// line is not delivering, and loosens again the moment it does.
///
/// The evidence is one specific observation, and it is the only one that
/// distinguishes a dead line from a quiet one: **a drain that produced
/// reports after a park the deadline won.** Bytes were sitting in the
/// controller's output buffer and no interrupt announced them — which is
/// exactly what a blocked or misprogrammed line looks like from inside
/// the relay, and exactly what nothing else looks like.
///
/// The earlier policy keyed on *keystrokes* instead, tightening whenever
/// reports flowed. That was right only under the assumption the line was
/// dead — which it was, for the week #452 was open — and it is wrong now
/// that it fires: a healthy machine would poll hardest precisely while
/// the interrupt was already doing the work.
///
/// A fired park recovers immediately, so a false degrade (a byte landing
/// in the window between a deadline resolving and the drain reading it)
/// costs one fast period and heals itself on the next real interrupt.
/// That is why no debounce is needed here, and why one would only slow
/// down the recovery it is meant to protect.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Backstop {
    /// The line has been caught not delivering; poll as the fallback.
    degraded: bool,
    /// The park just resolved on its deadline rather than the line —
    /// so the drain that follows is the one that can convict it.
    deadline_won: bool,
}

impl Backstop {
    const fn new() -> Self {
        Self {
            degraded: false,
            deadline_won: false,
        }
    }

    /// The deadline for the next park.
    const fn period_ms(&self) -> u64 {
        if self.degraded {
            BACKSTOP_DEGRADED_MS
        } else {
            BACKSTOP_LIVENESS_MS
        }
    }

    /// Record how a park resolved. A real fire is proof of a working
    /// line and returns the backstop to liveness duty at once.
    fn parked(&mut self, fired: bool) {
        self.deadline_won = !fired;
        if fired {
            self.degraded = false;
        }
    }

    /// Record what the following drain produced. Reports found after a
    /// deadline-won park are bytes the line never announced.
    fn drained(&mut self, delivered: bool) {
        if delivered && self.deadline_won {
            self.degraded = true;
        }
    }
}

/// Bytes the handoff queue's region needs.
const RELAY_BYTES: usize = spsc::region_bytes::<InputReport>(RELAY_CAPACITY);

/// The in-space handoff region.
///
/// A static rather than a leased window because both ends are threads of
/// **one task** in one address space: there is no boundary to cross, so
/// there is no capability to carry and nothing to revoke. (A queue to
/// another *task* is a leased window — that is what every subscriber's
/// report ring is.)
static RELAY_REGION: SharedRegion = SharedRegion(core::cell::UnsafeCell::new([0; RELAY_BYTES]));

/// The region wrapper that makes the static shareable.
///
/// The `Sync` claim is the SPSC contract itself, which the queue's own
/// atomics enforce: exactly one producer (this relay thread) and one
/// consumer (thread 0) touch disjoint indices. Nothing else in this
/// crate may take a handle — both constructors below are private and
/// each is called once.
///
/// `align(64)` is not decoration: `spsc::init` refuses a window that is
/// not 64-byte aligned (`RingError::Misaligned`), and a bare byte array
/// is aligned to 1. Every other ring in the system gets this for free
/// by living in a page-granular mapped window; this one is a static, so
/// it must say so.
#[repr(align(64))]
struct SharedRegion(core::cell::UnsafeCell<[u8; RELAY_BYTES]>);

// SAFETY: the only two handles ever taken are the single producer (this
// thread, in `run`) and the single consumer (thread 0, in `open`), which
// is precisely the discipline `spsc` is built on.
unsafe impl Sync for SharedRegion {}

impl SharedRegion {
    /// A `&'static mut` view of the region.
    ///
    /// # Safety
    /// At most one producer and one consumer handle may be alive over
    /// this region at a time, and each may be taken only once.
    unsafe fn view(&self) -> &'static mut [u8] {
        // SAFETY: caller contract; the static lives for the program.
        unsafe { &mut *self.0.get() }
    }
}

/// Live relay counters, published for thread 0's status row.
///
/// Atomics rather than a lock: this is written by one thread and read by
/// another on a path that must not block either. `Relaxed` for the counters —
/// they are a display, and no other state's visibility depends on them.
///
/// [`rung`](Self::rung) is the exception, and the one field that is not
/// merely a display: it is also the futex word thread 0 waits on for this
/// relay's verdict, so the publishing store and that wait's load are `SeqCst`
/// (see [`publish_rung`]). Reads that only *show* it — the status row, the
/// test — stay `Relaxed`, because reading a value is not joining a handshake.
pub struct RelayStats {
    /// Line wakes handled.
    pub wakes: AtomicU64,
    /// Reports handed to thread 0.
    pub events: AtomicU64,
    /// Reports dropped against a full handoff queue.
    pub dropped: AtomicU64,
    /// Scancodes the decoder could not name — carried as
    /// `Key::Unknown`, so they are counted rather than lost.
    pub unnamed: AtomicU64,
    /// Parks issued on the line.
    ///
    /// The denominator [`wakes`](Self::wakes) has always needed: on its
    /// own a wake count says the line fires, never how much of the work
    /// it carries. `wakes / parks` is the honest interrupt-versus-backstop
    /// ratio, and the epic's "zero empty polls" bar is a claim about
    /// exactly that number.
    ///
    /// It is also the liveness witness a park has no other way to
    /// publish: a relay whose park never returns freezes this counter
    /// while every other one stays plausible.
    pub parks: AtomicU64,
    /// Which rung refused, plus one (`0` = the relay is running).
    pub rung: AtomicU32,
    /// The lock state the keyboard's indicators should show ([`Locks`]
    /// bits), written by thread 0 when the keymap owner names this source
    /// and read by the relay — the only thread holding the ports — on its
    /// next wake. A word, not a queue: only the latest state matters.
    pub indicators: AtomicU8,
    /// Keyboard self-test replies seen after bring-up — the keyboard reset
    /// or was replugged, so its held keys were released and its indicators
    /// rewritten.
    pub resets: AtomicU64,
    /// Key detection errors / internal overruns the keyboard reported:
    /// keystrokes lost inside the keyboard itself.
    pub overruns: AtomicU64,
}

/// The relay's published counters.
pub static RELAY: RelayStats = RelayStats {
    wakes: AtomicU64::new(0),
    events: AtomicU64::new(0),
    dropped: AtomicU64::new(0),
    unnamed: AtomicU64::new(0),
    parks: AtomicU64::new(0),
    rung: AtomicU32::new(RUNG_UNSTARTED),
    indicators: AtomicU8::new(0),
    resets: AtomicU64::new(0),
    overruns: AtomicU64::new(0),
};

/// Ask the internal keyboard to show `locks` on its indicators. Latest
/// wins; the relay converges on its next wake (an interrupt, or at most
/// the liveness backstop).
pub fn set_indicators(locks: Locks) {
    RELAY.indicators.store(locks.bits(), Ordering::Relaxed);
}

/// Rung codes, published in [`RelayStats::rung`]. They are the whole
/// diagnosis on a serial-less machine, so each names one refusal rather
/// than collapsing into "relay failed".
pub mod rung {
    /// Spawned but not yet past bring-up.
    pub const UNSTARTED: u32 = 1;
    /// The thread could not read its own boot page.
    pub const NO_CONTEXT: u32 = 2;
    /// This thread's `role::DEVICE` index 0 is vacant — the service's
    /// legacy claim delivered no i8042. **Not a fault**: a machine with
    /// no legacy controller is a machine this relay has nothing to do
    /// on.  The note's detail half carries WHICH leg of the claim came
    /// back empty ([`no_device_detail`](super::no_device_detail)) — on a
    /// machine whose firmware DID describe an i8042, that hex is the
    /// difference between a diagnosis and a flash cycle.
    pub const NO_DEVICE: u32 = 3;
    /// The delivered capability was not a platform device.
    pub const BAD_DEVICE: u32 = 4;
    /// The port grant was refused.
    pub const NO_PORTS: u32 = 5;
    /// The interrupt line could not be established — a relay that
    /// cannot park is a refusal, per the epic's no-polling bar.
    pub const NO_LINE: u32 = 6;
    /// No self-kick send end: the relay could decode keystrokes but
    /// never wake thread 0 to route them.
    ///
    /// Reports what the **service declared**, which is a real state: a
    /// row whose founding carried no self-kick (`role::DOORBELL@2`)
    /// endows none, so this thread is founded without one. It is not a
    /// state the input row can reach — it declares a worker, so its
    /// founding carves the kick — and that is the point of keeping the
    /// rung: if it is ever published again, the endowment is missing,
    /// not the boot page.
    pub const NO_KICK: u32 = 7;
    /// The handoff queue could not be formed.
    pub const NO_QUEUE: u32 = 8;
    /// Controller bring-up refused; the low byte carries the ladder's
    /// own [`Rung`](kore_i8042::Rung) discriminant.
    pub const BRINGUP: u32 = 9;
}

use rung::UNSTARTED as RUNG_UNSTARTED;

/// The [`rung::NO_DEVICE`] note's detail vocabulary — WHICH leg of the
/// service's legacy claim (`claim_devices`' i8042 arm) came back empty,
/// packed into the note's `detail << 8` half by thread 0.  The codes
/// name the legs in claim order, so the printed hex reads as how far
/// the claim got.
pub mod no_device_detail {
    /// The service's `DEVICES` table declares no second selector.
    pub const NO_SELECTOR: u64 = 1;
    /// `MultiBroker` opened no face 0 (ordinal-0 broker face vacant).
    pub const NO_FACE: u64 = 2;
    /// The broker answered `NoMatch` — no unclaimed platform entry
    /// matched the selector.
    pub const REFUSED_NO_MATCH: u64 = 3;
    /// The broker answered `Taken`.
    pub const REFUSED_TAKEN: u64 = 4;
    /// The broker answered `Denied` — a kernel gate or the report
    /// transport refused inside the broker.
    pub const REFUSED_DENIED: u64 = 5;
    /// Any other broker refusal (`NotHeld`/`NoSupply`/`BadFace`/…).
    pub const REFUSED_OTHER: u64 = 6;
    /// The grant arrived but as the wrong species (a PCI device answered
    /// a platform ask) or its possession verb refused the returned
    /// capability.
    pub const WRONG_GRANT: u64 = 7;
    /// The claim SUCCEEDED — the service holds the platform token.  Seen
    /// beside `NO_DEVICE` this is a contradiction worth its own code:
    /// the token existed and the relay's arena slot is still vacant, so
    /// the broken leg is the worker-endowment carve, not the claim.
    pub const CLAIMED: u64 = 0x7F;
}

/// How long thread 0 waits for this relay's verdict before beaconing
/// without it.
///
/// **Derived from the ladder's own budget**, not chosen: `bring_up` runs
/// at most two dozen bounded handshakes (`write_cmd` / `read_data` /
/// `device_cmd`), and each is capped at `ATTEMPTS` — 10_000 — status
/// reads of port 0x64 (`drivers/i8042/src/bringup.rs`). A legacy 8-bit
/// port cycle costs about a microsecond; that is the architectural figure
/// the old `outb 0x80` delay loops were built on, and a KVM PIO exit is
/// the same order. So the ladder's absolute worst case is
/// ~24 × 10_000 × 1 µs ≈ 0.25 s, and this is four times that: long enough
/// that a relay which is going to finish has finished, short enough to be
/// noise against the founder's 120 s readiness budget.
const SETTLE_BUDGET_NS: u64 = 1_000_000_000;

/// Wait — bounded, and *parked* — for the relay to publish a verdict, and
/// answer the note thread 0's readiness beacon should carry.
///
/// `0` means the relay is running normally, and is exactly the payload the
/// founder prints nothing for. Anything else is a [`rung`] code, already
/// packed in the founder's `rung | detail << 8` shape, which it renders
/// beside the row's ready line.
///
/// **A timeout is not a failure and never becomes one.** A controller
/// still climbing the ladder is a slow embedded part, not a fault, and the
/// input service must serve either way: the USB keyboard is unaffected by
/// this thread, and a machine with no i8042 at all is the ordinary case.
/// The note reported then is [`rung::UNSTARTED`], which reads on screen as
/// precisely what it is.
///
/// [`RELAY`] is a plain static in the *same address space* — thread 0 and
/// this relay are threads of one task — so **reading** the verdict is a load
/// and not a channel. Sharing an address space shares memory; that is the one
/// thing the thread model does give both halves for free.
///
/// **Waiting** for it is the half memory cannot give. Holdings are per task,
/// so the relay announcing "I have answered" is capability traffic however
/// cheap the read is — a notice on the doorbell send it was endowed with. The
/// futex is exactly that pairing: the word is shared memory, the wake is
/// authority, and neither one alone is a condition variable.
pub async fn settle_note() -> u64 {
    // **One wait, not five hundred.** This used to re-read the rung every
    // 2 ms until the budget ran out — a condition variable hand-rolled as
    // cadence polling, because until the futex landed there was no way for
    // the relay to *say* it had answered. Now there is: `publish_rung` kicks
    // this thread's doorbell, so the common case is one park that ends the
    // instant the verdict exists, and the budget is a backstop rather than
    // the mechanism.
    //
    // The async form deliberately, not the blocking one: this runs on the
    // executor, and a kernel park here would stall every sibling future and
    // swallow a doorbell notice one of them was owed.
    //
    // `SeqCst` pairs with `publish_rung`'s store — the futex registers this
    // thread before the load below, and the relay stores before it scans, so
    // the two cannot miss each other.
    //
    // The budget is **absolute**, computed once. Passing the whole budget to
    // each wait would restart it on every spurious wake, and the executor's
    // doorbell is shared with everything else that kicks this thread — so the
    // total wait could run arbitrarily past a bound that reads like a
    // guarantee.
    let deadline = kore_rt::executor::clock_ns()
        .await
        .saturating_add(SETTLE_BUDGET_NS);
    loop {
        let code = RELAY.rung.load(Ordering::SeqCst);
        if code != rung::UNSTARTED {
            return u64::from(code);
        }
        // A timeout is the honest end of the wait, not a failure: a machine
        // with no i8042 never publishes anything, and `UNSTARTED` is exactly
        // what its row should read.
        let now = kore_rt::executor::clock_ns().await;
        if now >= deadline {
            return u64::from(rung::UNSTARTED);
        }
        match kore_rt::futex::wait_async(
            &RELAY.rung,
            rung::UNSTARTED,
            Some(core::time::Duration::from_nanos(deadline - now)),
        )
        .await
        {
            // Woken, or the word already moved — re-read it at the top.
            // `Retry` is not a retry of the *wait*, it is the fast path
            // saying the answer arrived; the loop's own load is what reads
            // it, so there is exactly one place the value is interpreted.
            kore_rt::futex::WaitOutcome::Retry | kore_rt::futex::WaitOutcome::Woken => continue,
            kore_rt::futex::WaitOutcome::TimedOut => return u64::from(rung::UNSTARTED),
            // No doorbell or no thread identity — this thread cannot park at
            // all. Falling back to the old cadence would be a spin in
            // disguise; the budget has to be abandoned rather than burned, so
            // report what is actually known right now.
            kore_rt::futex::WaitOutcome::Unsupported => {
                return u64::from(RELAY.rung.load(Ordering::SeqCst));
            }
        }
    }
}

/// The declared worker — what a service puts in
/// [`SystemService::WORKERS`](kore_rt::service::SystemService::WORKERS).
pub const WORKER: Worker = Worker {
    name: "i8042-relay",
    entry: run,
};

/// What this worker must be founded holding: a derived grant over the
/// service's platform device.
///
/// A derive, not a move — the service keeps its own device capability,
/// so revoking the relay's authority stays its lever.
///
/// `DERIVE` rides because the relay's whole job starts with two derives
/// of its own (its ports and its line). **`REVOKE` rides too**, and the
/// first version's omission of it is why the relay refused on the first
/// KVM boot: Σ5 clamps a derived child's rights to a subset of its
/// parent's, and both `claim_ports` and `establish_line` ask for
/// `ACCESS | REVOKE` so their tokens can retire themselves. Withholding
/// `REVOKE` here made every one of those derives `Denied`, and the relay
/// parked at its no-ports rung with the keyboard dead.
///
/// The omission came from reading `REVOKE` as "may retire the device".
/// It does not mean that here: T6 cascades only *downward* from what a
/// task holds, so this right lets the relay retire the port grant and
/// the route it derived — its own children — and reaches neither the
/// service's device capability nor the catalog's.
///
/// `MODIFY` is dropped for the same reason it was never needed: it gates
/// the device's control facet, which a relay does not touch.
pub fn endowment(device: kore_abi::session::SlotRef) -> WorkerCap {
    WorkerCap {
        role: role::DEVICE,
        index: 0,
        subject: device,
        kind: ResourceKind::PlatformDevice as u16 as u8,
        rights: Rights::ACCESS | Rights::DERIVE | Rights::REVOKE,
    }
}

/// What this worker must be founded holding to *reach* thread 0: its own
/// `ChanSend` into the service's doorbell, derived off the self-kick the
/// founder delivered to the service.
///
/// A `ChanSend` derived from a `ChanSend` — the ordinary attenuation, not
/// a special case: the doorbell channel is the subject either way, and
/// posting a notice on a grandchild end is indistinguishable from posting
/// on the parent. Which is exactly the property the self-kick was carved
/// for: thread 0 drains its ring on any wake, so it never learns — or
/// needs to learn — which of its sources rang.
///
/// **Derived rather than shared** because there is nothing to share. The
/// handoff queue is memory and both threads see it for free; a capability
/// is not memory. The service's `role::DOORBELL@2` lives in thread 0's
/// holdings, and holdings are per task — the same law that makes the I/O
/// bitmap safe to key on the running task makes thread 0's send end
/// unreachable from here, no matter that the two threads share every
/// page.
///
/// Delivered under the **same** `(role, index)` the service knows it by,
/// so "`DOORBELL@2` is a send end into my own doorbell" reads as one
/// sentence whether the reader is thread 0 or this thread.
///
/// [`NOTICE_RIGHTS`](verbs::NOTICE_RIGHTS) and no more, and both halves
/// of that are load-bearing. It must cover what
/// [`post_notice`](verbs::post_notice) asks of a send end, or the kick is
/// `Denied` and keystrokes sit in the queue unannounced — the same shape
/// as the port grant's failure one rung up. It must also stay *within*
/// [`SEND_RIGHTS`](kore_rt::svc::SEND_RIGHTS), because Σ5 clamps a
/// derived child to a subset of its parent's, and that is the trap this
/// file already fell into once: the instinct after the port bug is to add
/// `REVOKE`, which here is outside the founder's carve and would refuse
/// the **spawn** — losing the whole input row, not just the keyboard.
///
/// `REVOKE` would buy nothing anyway — the relay never retires its kick —
/// but note what its absence costs, because it is the one way this
/// endowment differs from the device above. The service can revoke the
/// relay's *device* authority and cannot revoke its *kick*: its own
/// self-kick carries no `REVOKE` to pass down, so the only reap is from
/// above, where the founder holds the channel mint and T6 cascades
/// through every end at once. Two threads of one task share a lifetime,
/// so nothing depends on that here — but it is a real asymmetry rather
/// than a simplification.
pub fn kick_endowment(kick: kore_abi::session::SlotRef) -> WorkerCap {
    WorkerCap {
        role: role::DOORBELL,
        index: 2,
        subject: kick,
        kind: ResourceKind::ChanSend as u16 as u8,
        rights: verbs::NOTICE_RIGHTS,
    }
}

/// The relay's books, sampled for the control plane.
///
/// `Relaxed` throughout, like every other read of these counters: they
/// are a display, and no other state's visibility hangs off them. Thread
/// 0 reads a static this thread writes — the two are threads of one
/// task, so this is a load and not a channel, the same property
/// [`settle_note`] relies on.
///
/// The counters are `u64` and the wire carries `u32`: saturating rather
/// than truncating, so a wrapped display can never read as a small
/// number and be mistaken for a quiet relay.
pub fn books() -> crate::service::RelayBooks {
    fn clamp(v: u64) -> u32 {
        u32::try_from(v).unwrap_or(u32::MAX)
    }
    crate::service::RelayBooks {
        rung: RELAY.rung.load(Ordering::Relaxed) as u8,
        wakes: clamp(RELAY.wakes.load(Ordering::Relaxed)),
        parks: clamp(RELAY.parks.load(Ordering::Relaxed)),
        events: clamp(RELAY.events.load(Ordering::Relaxed)),
        dropped: clamp(RELAY.dropped.load(Ordering::Relaxed)),
        unnamed: clamp(RELAY.unnamed.load(Ordering::Relaxed)),
        xhci_bound: false,
        xhci_bind: 0,
        xhci_sources: 0,
        xhci_discovery_errors: 0,
        xhci_irq_errors: 0,
        i2c: [crate::service::I2cRelayBooks::default(); 2],
    }
}

/// Take the consumer half of the handoff queue — thread 0's end.
///
/// Called exactly once, from the service's `start_on_metal`, strictly
/// before the relay is spawned (so the region is formed before anything
/// pushes into it).
pub fn open() -> Option<spsc::Consumer<'static, InputReport>> {
    // SAFETY: the single consumer handle, taken once. `start_on_metal`
    // runs before `spawn_workers`, so no producer exists yet.
    let window = unsafe { RELAY_REGION.view() };
    spsc::init::<InputReport>(window, RELAY_CAPACITY).ok()?;
    // SAFETY: as above — one consumer, and the producer takes its own
    // handle over disjoint state.
    let window = unsafe { RELAY_REGION.view() };
    spsc::consumer::<InputReport>(window).ok()
}

/// The i8042's two ports, reached through the thread's own grant.
struct GrantPorts(kore_rt::cap::PortGrant);

impl PortAccess for GrantPorts {
    #[inline]
    fn inb(&self, off: usize) -> u8 {
        self.0.read8(port_of(off))
    }

    #[inline]
    fn outb(&self, off: usize, val: u8) {
        self.0.write8(port_of(off), val);
    }
}

/// The driver seam addresses its ports by offset; these are the two
/// architectural addresses a `PNP0303` decodes.
#[inline]
const fn port_of(off: usize) -> u16 {
    if off == kore_i8042::port::DATA {
        PORT_DATA
    } else {
        PORT_STATUS
    }
}

/// The **held-key state** this relay maintains, because a HID boot
/// report is a *state* snapshot (modifiers plus up to six held usages)
/// while a PS/2 controller emits *events* (this key went down, that one
/// came up).
///
/// Somebody has to hold the difference, and it has to be here: the
/// terminal downstream diffs snapshots to synthesize presses, so handing
/// it events dressed as snapshots would make every keystroke look like a
/// fresh press of everything held.
#[derive(Clone, Copy, Default)]
struct Held {
    modifiers: u8,
    keys: [u8; 6],
}

impl Held {
    /// Record a press; `false` if it changed nothing (a key repeating
    /// under the controller's own auto-repeat, which the terminal
    /// already handles by diffing).
    fn press(&mut self, what: Contribution) -> bool {
        match what {
            Contribution::Modifier(bit) => {
                let before = self.modifiers;
                self.modifiers |= bit;
                before != self.modifiers
            }
            Contribution::Usage(u) => {
                if self.keys.contains(&u) {
                    return false;
                }
                // Six-key rollover, as the boot report defines it: a
                // seventh simultaneous key is not reportable, and
                // dropping it is what the format says to do.
                match self.keys.iter_mut().find(|k| **k == 0) {
                    Some(slot) => {
                        *slot = u;
                        true
                    }
                    None => false,
                }
            }
            Contribution::None => false,
        }
    }

    /// Record a release; `false` if the key was not held.
    fn release(&mut self, what: Contribution) -> bool {
        match what {
            Contribution::Modifier(bit) => {
                let before = self.modifiers;
                self.modifiers &= !bit;
                before != self.modifiers
            }
            Contribution::Usage(u) => match self.keys.iter_mut().find(|k| **k == u) {
                Some(slot) => {
                    *slot = 0;
                    true
                }
                None => false,
            },
            Contribution::None => false,
        }
    }

    /// Let go of everything — the keyboard reset itself, so whatever it
    /// held is up. `false` if nothing was held.
    fn release_all(&mut self) -> bool {
        let held = self.modifiers != 0 || self.keys != [0; 6];
        *self = Self::default();
        held
    }

    /// This state as the report the router forwards.
    fn report(&self) -> InputReport {
        InputReport::keyboard(SOURCE_LEGACY, self.modifiers, self.keys)
    }
}

/// **Publish this relay's verdict**, and wake thread 0 if it is waiting on it.
///
/// The one place [`RelayStats::rung`] is written, so the store and the wake
/// cannot drift apart: a verdict published without the kick would leave
/// [`settle_note`] asleep until its budget expired, reporting
/// [`rung::UNSTARTED`] for a relay that had already answered.
///
/// **Order is the contract, not a style choice.** The store happens first and
/// the wake second, because a waiter that misses the kick must be able to
/// re-read the word and find the answer already there — that is the half of
/// the futex's lost-wakeup argument this side owns. `SeqCst` for the same
/// reason: the waiter registers then loads, this stores then scans, and only
/// sequential consistency forbids both from missing each other.
///
/// Every other read of `rung` stays `Relaxed`, and correctly — the status row
/// and the test are displays of a value, not participants in a handshake.
fn publish_rung(code: u32) {
    RELAY.rung.store(code, Ordering::SeqCst);
    // Best effort by construction: `false` means thread 0 was not waiting
    // (the common case — it usually samples the answer before ever parking),
    // or this thread holds no doorbell send end because the founding carved
    // none. Neither is a failure worth reporting, and neither loses the
    // verdict: it is already published above, and `settle_note`'s budget
    // bounds the wait either way.
    let _ = kore_rt::futex::wake(&RELAY.rung);
}

/// Write one byte of an indicator exchange, once the controller has taken
/// the last byte written. A controller that never drains its input buffer
/// within the bound abandons the exchange (as an unanswered byte would)
/// instead of stalling the key path.
fn send_indicator_byte<P: PortAccess>(regs: &PortRegs<P>, indicators: &mut Indicators, byte: u8) {
    let ready = kore_driver::handshake(
        regs,
        kore_driver::Budget::new(INDICATOR_WRITE_BUDGET_NS, 64, kore_driver::host::clock_now_ns),
        || core::hint::spin_loop(),
        PortRegs::input_clear,
        PortRegs::vanished,
    );
    if ready == kore_driver::Ack::Acked {
        regs.write32(kore_i8042::port::DATA, u32::from(byte));
    } else {
        // Abandoned now, never left to time out; the abandonment owes the
        // keyboard an `Enable` in case it was waiting for this very byte.
        indicators.write_failed();
    }
}

/// How long the relay waits for the controller to take an indicator byte.
/// The input buffer is empty at rest; this only bounds a wedged controller.
const INDICATOR_WRITE_BUDGET_NS: u64 = 2_000_000;

/// Park the relay with a named rung.
///
/// **Deliberately does not exit**, now that it could.  A refused relay is
/// a *diagnostic state*, not a finished job: the rung stays published, and
/// thread 0's status row keeps naming the refusal for as long as the
/// machine runs — which on a serial-less box is the entire post-mortem.
/// Exiting would end the thread and take the row's subject with it, so the
/// one machine that most needs the answer would be the one that loses it.
fn refuse(code: u32) -> ! {
    publish_rung(code);
    // **Sleep, do not spin.** A refusing relay is the *expected* state on
    // every machine with no legacy keyboard controller, so this path runs
    // for the whole uptime of an ordinary boot — and a `spin_loop` here
    // would burn a core for that entire time to accomplish nothing.
    // (Preemption means it would not wedge the machine; it would just
    // quietly cost a CPU forever, which is worse for being invisible.)
    //
    // Day-long strides are the honest "forever": the thread has to keep
    // existing to keep reporting, and this is what existing costs when
    // there is nothing to wait for. A refused clock read still yields a
    // park attempt rather than falling through to a spin.
    let mut now_ms = verbs::sleep_until(1).map_or(0, |word| (word >> 32) as u64);
    loop {
        let _ = verbs::sleep_until((now_ms + 86_400_000) * 1_000_000);
        now_ms += 86_400_000;
    }
}

/// The relay thread.
///
/// Declared `-> ()` like every worker entry, though it never returns: its
/// steady state is the park loop below and its failure state is
/// [`refuse`], which parks too (see there for why a refused relay stays
/// resident instead of exiting).
fn run() {
    let Some(rt) = kore_rt::entry::thread_context() else {
        refuse(rung::NO_CONTEXT);
    };
    let boot = rt.bootstrap();

    // The endowment the parent derived into this thread's own arena.
    let Some(slot) = boot.handle(role::DEVICE, 0) else {
        refuse(rung::NO_DEVICE);
    };
    let Some(device) = verbs::claim_platform_device(slot) else {
        refuse(rung::BAD_DEVICE);
    };

    // This thread's ports — held HERE, because the kernel programs the
    // I/O bitmap from the task about to run.
    let Some(ports) = verbs::claim_ports(&device) else {
        refuse(rung::NO_PORTS);
    };
    // ...and this thread's line. A relay that cannot park is a refusal:
    // the alternative is a cadence poll, which this epic ruled out.
    let Some(route) = verbs::establish_line(&device) else {
        refuse(rung::NO_LINE);
    };
    // The wake edge to thread 0 — this thread's OWN send end, derived
    // into this arena beside the device above (`kick_endowment`), not
    // the service's. Reading the service's `role::DOORBELL@2` from here
    // is what the first version did, and it found an empty boot page:
    // the founder delivers the self-kick to thread 0, because at
    // founding thread 0 is the only task that exists.
    //
    // Without a kick a keystroke would sit in the queue until something
    // unrelated woke the router — on a machine whose only keyboard is
    // this one, forever. So it is a refusal, not a degraded mode.
    let Some(kick) = boot.handle(role::DOORBELL, SERVICE_KICK_PEER) else {
        refuse(rung::NO_KICK);
    };

    let regs = PortRegs(GrantPorts(ports));
    // Paced, not spun ([`kore_rt::pace`]), and **clocked**. This thread is
    // spawned inside the founding window, where the founder's drive loop is
    // the boot splash's only pump, so the ladder's two dozen handshakes are
    // time the founder does not get.
    //
    // The pacing was only half of that. Each handshake was also bounded by a
    // probe *count* (`ATTEMPTS`, 10 000), and a count is not a duration: these
    // probes are port I/O to 0x60/0x64, so on a machine with no controller
    // behind them each one is an unclaimed legacy cycle costing tens of
    // microseconds with the core stalled inside the `in`. Ten thousand of
    // those is minutes, per rung, and no amount of ceding reaches a core that
    // is not executing. `clock_now_ns` turns every rung into a real 50 ms
    // bound; it degrades to the old count only where no counter is calibrated.
    let brought = match bring_up(
        &regs,
        kore_driver::host::clock_now_ns,
        kore_rt::pace::ceding(),
    ) {
        Ok(b) => b,
        Err(r) => refuse(rung::BRINGUP | ((rung_code(r) as u32) << 8)),
    };
    let mut decoder = Decoder::new(brought.set);

    // SAFETY: the single producer handle, taken once, on the only
    // thread that pushes. `open()` formed the region before this thread
    // was spawned.
    let window = unsafe { RELAY_REGION.view() };
    let Ok(mut queue) = spsc::producer::<InputReport>(window) else {
        refuse(rung::NO_QUEUE);
    };

    publish_rung(0);
    let mut held = Held::default();
    let mut indicators = Indicators::new();
    let mut backstop = Backstop::new();
    loop {
        // **Drain first, park second.** The obvious order is the other
        // way round, and it drops keystrokes: bring-up cleared the
        // controller's output buffer, so a key struck between then and
        // the first park raises its edge while nothing is waiting for
        // it. On an edge-triggered line no further edge arrives until
        // the buffer is read, so that keystroke — and every one after
        // it — would be lost forever. Draining at the top of each
        // iteration makes the first pass flush whatever is already
        // there, and costs one status read per wake.
        //
        // Bounded by the queue depth: a controller stuck asserting OBF
        // must not hold this thread inside one wake forever.
        let mut delivered = false;
        let mut publish = |held: &Held, delivered: &mut bool| {
            if queue.push_lossy(held.report()) {
                RELAY.events.fetch_add(1, Ordering::Relaxed);
                *delivered = true;
            } else {
                RELAY.dropped.fetch_add(1, Ordering::Relaxed);
            }
        };
        for _ in 0..RELAY_CAPACITY {
            let status = regs.status();
            if status & kore_i8042::port::status::OBF == 0 {
                break;
            }
            let byte = regs.read32(kore_i8042::port::DATA) as u8;
            // A byte from the AUX channel is the mouse's, not the
            // keyboard's. Bring-up disabled that port, so this should
            // not occur — but feeding pointer bytes to a scancode
            // decoder produces plausible-looking garbage keys rather
            // than an error, and a shared controller is exactly where
            // that would happen. Read and discarded: the byte must
            // still leave the buffer or the line stays asserted.
            if status & kore_i8042::port::status::AUX != 0 {
                continue;
            }
            let Some(decoded) = decoder.feed(byte) else {
                continue; // a prefix, or a sequence still in flight
            };
            let event = match decoded {
                Decoded::Key(event) => event,
                Decoded::Reply(reply) => {
                    match reply {
                        Reply::SelfTestPassed => {
                            // The keyboard reset (or was replugged): every
                            // key it held is up, and its indicators went
                            // dark — `Indicators` re-lights them below.
                            RELAY.resets.fetch_add(1, Ordering::Relaxed);
                            if held.release_all() {
                                publish(&held, &mut delivered);
                            }
                        }
                        Reply::Error => {
                            RELAY.overruns.fetch_add(1, Ordering::Relaxed);
                        }
                        Reply::Ack | Reply::Resend => {}
                    }
                    let now = kore_driver::host::clock_now_ns();
                    if let Some(next) = indicators.reply(reply, now) {
                        send_indicator_byte(&regs, &mut indicators, next);
                    }
                    continue;
                }
            };
            let key = match event {
                KeyEvent::Press(key) | KeyEvent::Release(key) | KeyEvent::Tap(key) => key,
            };
            // A key proves the keyboard is scanning, whatever an abandoned
            // indicator exchange left uncertain.
            indicators.scanned();
            if matches!(key, Key::Unknown { .. }) {
                RELAY.unnamed.fetch_add(1, Ordering::Relaxed);
            }
            match event {
                KeyEvent::Press(key) => {
                    if held.press(key.hid()) {
                        publish(&held, &mut delivered);
                    }
                }
                KeyEvent::Release(key) => {
                    if held.release(key.hid()) {
                        publish(&held, &mut delivered);
                    }
                }
                // A key with no break code: one snapshot holding it, one
                // without, so a downstream diff sees exactly one press.
                KeyEvent::Tap(key) => {
                    if held.press(key.hid()) {
                        publish(&held, &mut delivered);
                        held.release(key.hid());
                        publish(&held, &mut delivered);
                    }
                }
            }
        }

        // **Converge the indicators** — after the drain, so a reply that
        // arrived with this batch has already advanced the exchange. An
        // overdue acknowledgement abandons the exchange (bounded tries per
        // state); otherwise a desired state the keyboard may not show
        // starts one. Never blocks the key path: the exchange advances on
        // the replies the drain above decodes.
        let now = kore_driver::host::clock_now_ns();
        indicators.expire(now);
        let desired =
            Locks::from_bits(RELAY.indicators.load(Ordering::Relaxed)).unwrap_or_default();
        if let Some(byte) = indicators.start(desired, now) {
            send_indicator_byte(&regs, &mut indicators, byte);
        }

        // One kick per drained batch, not per report: the queue is the
        // truth and the notice is advisory, so a consumer that is
        // already awake needs no second nudge (the router's own
        // `kick_focused` takes exactly this posture).
        //
        // Since the wake bell landed, "already awake" is something this
        // thread can *ask* rather than assume. Ringing thread 0's bell is
        // itself the wake edge for a monitoring executor — the store ends
        // its `UMWAIT` — and the notice is posted only when the bell answers
        // that the service is parked in the kernel, where a store cannot
        // reach it.
        //
        // Safe in every degraded direction by construction: an unrung bell,
        // an out-of-band peer, or a service that has not started its
        // executor all answer "post the notice", which is exactly what this
        // line did before.
        if delivered && kore_rt::wake::ring_doorbell_peer(SERVICE_KICK_PEER) {
            let _ = verbs::post_notice(kick, kore_rt::svc::COOKIE_REPORT, 0);
        }
        // Reports found after a park the deadline won are bytes the line
        // never announced — the one observation that convicts it (see
        // [`Backstop`]).
        backstop.drained(delivered);

        // Park on the line. A revoked route ends the relay: its device
        // was taken away, and there is nothing further it may lawfully
        // do.
        //
        // **The deadline is a liveness backstop, not a cadence poll**,
        // and since #452 it is strictly that: the line is the wake edge,
        // a keystroke is served at interrupt latency, and the deadline
        // only tightens once the line has been caught not delivering.
        // The epic's "zero empty polls" bar is a claim about exactly
        // this, and `wakes` against `parks` is the number that measures
        // it.
        //
        // What the deadline buys is that a dead line degrades to a slow
        // keyboard instead of a silent one — and says so, because
        // `events` climbing while `wakes` stays flat is precisely the
        // shape of "the bytes were always there and only the interrupt
        // is missing". Not hypothetical: it is what the Yoga did for a
        // week with its pin programmed, unmasked, and blocked by the
        // IOMMU, and without this the relay would have parked forever
        // beside a buffer full of keystrokes nobody was reading.
        let period_ms = backstop.period_ms();
        RELAY.parks.fetch_add(1, Ordering::Relaxed);
        let parked = match verbs::sleep_until(1).map(|word| (word >> 32) as u64) {
            Some(now_ms) => verbs::wait_interrupt_until(&route, (now_ms + period_ms) * 1_000_000),
            // No clock. A deadline computed from a bad `now` could name
            // an instant already past and turn this park into a spin, so
            // wait on the line alone rather than risk burning a core.
            None => verbs::wait_interrupt(&route),
        };
        match parked {
            Ok(verbs::InterruptWait::Fired) => {
                RELAY.wakes.fetch_add(1, Ordering::Relaxed);
                backstop.parked(true);
            }
            // The backstop expired: fall through to the drain at the top
            // of the loop, which is the whole point of arming it — and
            // which is also where this park finds out whether the line
            // should have beaten it there.
            Ok(verbs::InterruptWait::Deadline) => backstop.parked(false),
            Ok(verbs::InterruptWait::Revoked) => refuse(rung::NO_LINE),
            // A refused wait is a dead route — fail closed rather than
            // spinning on a wait that will never work, which would burn
            // a core reporting nothing.
            _ => refuse(rung::NO_LINE),
        }
    }
}

/// The bring-up ladder's rung as a byte, so it can ride the published
/// code's low half and name *which* rung refused rather than only that
/// bring-up did.
const fn rung_code(r: kore_i8042::Rung) -> u8 {
    use kore_i8042::Rung as R;
    match r {
        R::Absent => 1,
        R::InputStuck => 2,
        R::ReadConfig => 3,
        R::SelfTest(_) => 4,
        R::PortTest(_) => 5,
        R::NoDeviceAck => 6,
        R::Reset(_) => 7,
        R::WriteConfig => 8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kore_abi::keyboard::{modifier, usage};
    use kore_i8042::Key;

    /// The key for an ASCII letter, by its HID usage.
    fn letter(c: u8) -> Key {
        Key::Usage(usage::A + (c - b'a'))
    }

    #[test]
    fn a_modifier_and_a_key_occupy_different_halves_of_the_report() {
        let mut h = Held::default();
        assert!(h.press(Key::Usage(usage::LEFT_SHIFT).hid()));
        assert!(h.press(letter(b'a').hid()));
        let r = h.report();
        assert_eq!(r.source, SOURCE_LEGACY);
        assert_eq!(r.state, modifier::LEFT_SHIFT);
        assert_eq!(r.data[0], 0x04);
    }

    /// Auto-repeat: the controller re-sends a make code while a key is
    /// held, and the state it describes has not changed. Forwarding
    /// those would be pure noise — the terminal synthesizes repeats
    /// from held state, it does not want them re-announced.
    #[test]
    fn a_repeated_press_of_a_held_key_changes_nothing() {
        let mut h = Held::default();
        assert!(h.press(letter(b'a').hid()));
        assert!(!h.press(letter(b'a').hid()));
    }

    #[test]
    fn releasing_clears_the_slot_it_held() {
        let mut h = Held::default();
        h.press(letter(b'a').hid());
        h.press(letter(b'b').hid());
        assert!(h.release(letter(b'a').hid()));
        // 'b' keeps its slot; 'a' vacates its own, not the array.
        assert_eq!(h.report().data, [0, 0x05, 0, 0, 0, 0]);
        // A release of something never held is not a state change.
        assert!(!h.release(letter(b'z').hid()));
    }

    /// Six-key rollover is the boot report's own limit, and dropping
    /// the seventh is what the format prescribes — but it must not
    /// disturb the six already held.
    #[test]
    fn a_seventh_simultaneous_key_is_dropped_not_substituted() {
        let mut h = Held::default();
        for c in b"abcdef" {
            assert!(h.press(letter(*c).hid()));
        }
        let full = h.report().data;
        assert!(!h.press(letter(b'g').hid()));
        assert_eq!(h.report().data, full, "the held six are untouched");
    }

    /// The keyboard's self-test reply means it reset: whatever it held is
    /// up, GUI included — the modifier the PS/2 path now reports.
    #[test]
    fn a_keyboard_reset_releases_everything_it_held() {
        let mut h = Held::default();
        assert!(!h.release_all(), "nothing held, nothing to publish");
        assert!(h.press(Key::Usage(usage::LEFT_GUI).hid()));
        assert!(h.press(letter(b'a').hid()));
        assert_eq!(h.report().state, modifier::LEFT_GUI);
        assert!(h.release_all());
        assert_eq!(h.report(), InputReport::keyboard(SOURCE_LEGACY, 0, [0; 6]));
    }

    /// A key the decoder cannot name contributes nothing to the report
    /// — but it is still not silently lost: the relay counts it.
    #[test]
    fn an_unnamed_key_does_not_change_the_report() {
        let mut h = Held::default();
        assert!(!h.press(
            Key::Unknown {
                extended: false,
                code: 0x7f
            }
            .hid()
        ));
        assert_eq!(h.report().data, [0; 6]);
    }

    /// **The regression pin for the first KVM boot's failure.** The
    /// relay's endowment must grant at least what its own derives ask
    /// for, because Σ5 clamps a derived child to a subset of its
    /// parent's rights. The first version withheld `REVOKE`, so
    /// `claim_ports` was refused `Denied` and the relay parked at its
    /// no-ports rung — the keyboard simply did not work, and the only
    /// evidence was one wire-level refusal line in the boot log.
    ///
    /// Asserted against the verbs' own constant rather than a copied
    /// literal: a literal here would agree with the old bug just as
    /// happily as with the fix.
    #[test]
    fn the_endowment_covers_what_the_relays_own_derives_request() {
        let granted = endowment(kore_abi::session::SlotRef::from_wire(9)).rights;
        assert!(
            granted.contains(verbs::DEVICE_CHILD_RIGHTS),
            "the relay cannot derive a token carrying rights its own \
             endowment lacks: granted {granted:?}, needs {:?}",
            verbs::DEVICE_CHILD_RIGHTS,
        );
        // DERIVE is the toll for riding the Derive op-code at all — the
        // prepare-side gate demands it of the subject, separately from
        // the Σ5 subset check on the child.
        assert!(
            granted.contains(kore_abi::cap::Rights::DERIVE),
            "the endowment must carry DERIVE or the op-code itself refuses"
        );
    }

    /// **The regression pin for the first QEMU boot's failure.** The
    /// relay reached `[pin] gsi=1` — device claimed, ports claimed, line
    /// established — and then refused at [`rung::NO_KICK`], because it
    /// looked for its wake edge in its own boot page while the founder
    /// had delivered the self-kick to thread 0. The fix is a second
    /// declared endowment, so this asserts the two things a rights
    /// number here can get wrong, from both directions.
    ///
    /// Against the verbs' own constants rather than copied literals, for
    /// the reason the sibling test above records: a literal agrees with
    /// the bug as readily as with the fix.
    #[test]
    fn the_kick_endowment_is_exactly_what_a_notice_costs() {
        let granted = kick_endowment(kore_abi::session::SlotRef::from_wire(9)).rights;
        // The floor: a send end short of this cannot post the notice,
        // and the keystroke is decoded, queued, and never announced.
        assert!(
            granted.contains(verbs::NOTICE_RIGHTS),
            "the relay cannot kick thread 0 with a send end that lacks \
             what a notice costs: granted {granted:?}, needs {:?}",
            verbs::NOTICE_RIGHTS,
        );
        // The ceiling: Σ5 clamps a derived child to a subset of its
        // parent's, and the parent is the founder's self-kick carve. Ask
        // past it — `REVOKE` is the tempting one, after the port bug —
        // and the DERIVE is refused, which fails the spawn and takes the
        // whole input row down rather than just this thread.
        assert!(
            kore_rt::svc::SEND_RIGHTS.contains(granted),
            "the self-kick's carve is the ceiling: granted {granted:?}, \
             parent holds {:?}",
            kore_rt::svc::SEND_RIGHTS,
        );
    }

    /// The two endowments must land at **distinct** `(role, index)`
    /// pairs. `Bootstrap::handle` answers with the *first* entry that
    /// matches, so a collision does not refuse — it silently shadows,
    /// and the relay claims a channel as a platform device. Cheap to
    /// state, and the failure it prevents would surface on metal as a
    /// `BAD_DEVICE` refusal with nothing to suggest the kick was to
    /// blame.
    #[test]
    fn the_relays_two_endowments_do_not_share_a_boot_block_name() {
        let dev = endowment(kore_abi::session::SlotRef::from_wire(9));
        let kick = kick_endowment(kore_abi::session::SlotRef::from_wire(10));
        assert_ne!((dev.role, dev.index), (kick.role, kick.index));
    }

    /// **The note the readiness beacon carries decodes the way the founder
    /// prints it.**  The founder splits any beacon payload as
    /// `[7:0]` rung, `[63:8]` detail — one convention for a NACK and for
    /// a note — so this relay's `BRINGUP | ladder << 8` packing must
    /// already be in that shape.  A self-test failure has to read as rung
    /// 9, detail 4 ("bring-up refused, at the controller self-test"); the
    /// halves the other way round would name a rung that does not exist
    /// and send a reader to the wrong ladder.
    ///
    /// The whole beacon *send* is not host-constructible (it needs a
    /// founder, a rendezvous board and a running executor), so what is
    /// pinned here is the packing that send carries.
    #[test]
    fn a_bringup_refusal_decodes_as_rung_then_ladder_detail() {
        let published = rung::BRINGUP | ((rung_code(kore_i8042::Rung::SelfTest(0x55)) as u32) << 8);
        let payload = u64::from(published);
        assert_eq!(payload & 0xFF, u64::from(rung::BRINGUP));
        assert_eq!(payload >> 8, 4, "the i8042 ladder's self-test rung");
    }

    /// `0` is the founder's "nothing to report" — it renders to no text at
    /// all — so no code this relay can publish may be zero.  `UNSTARTED`
    /// included: it is what thread 0 reports when its bounded wait expires
    /// on a relay still climbing the ladder, and reporting that as "running
    /// normally" would erase exactly the case worth seeing.  Each must also
    /// fit the payload's low byte, or its detail half would collide with a
    /// neighbouring rung's.
    #[test]
    fn no_publishable_rung_collides_with_the_healthy_zero() {
        for code in [
            rung::UNSTARTED,
            rung::NO_CONTEXT,
            rung::NO_DEVICE,
            rung::BAD_DEVICE,
            rung::NO_PORTS,
            rung::NO_LINE,
            rung::NO_KICK,
            rung::NO_QUEUE,
            rung::BRINGUP,
        ] {
            assert_ne!(code, 0, "0 is reserved for 'the relay is running'");
            assert!(code < 256, "a rung must fit the payload's low byte");
        }
    }

    /// The relay's steady state publishes the healthy zero, and its
    /// refusals do not — the two ends of what thread 0 samples.  Pinned
    /// against the static's own initial value so a renumbering of
    /// `UNSTARTED` cannot quietly make a freshly spawned relay look ready.
    #[test]
    fn a_freshly_spawned_relay_reports_itself_unstarted() {
        assert_eq!(RELAY.rung.load(Ordering::Relaxed), rung::UNSTARTED);
    }

    /// **Polling is strictly a fallback.** The backstop starts at its
    /// liveness period and stays there for as long as the line does its
    /// job — including while somebody is typing, which is the case the
    /// previous policy got backwards. It tightened whenever reports
    /// flowed, so a healthy machine polled hardest exactly when the
    /// interrupt was already doing the work.
    #[test]
    fn a_working_line_never_makes_the_relay_poll_faster() {
        let mut b = Backstop::new();
        assert_eq!(b.period_ms(), BACKSTOP_LIVENESS_MS, "slow by default");
        // A burst of real interrupts, each delivering keystrokes.
        for _ in 0..64 {
            b.parked(true);
            b.drained(true);
            assert_eq!(
                b.period_ms(),
                BACKSTOP_LIVENESS_MS,
                "the line announced these bytes — nothing to fall back to"
            );
        }
        // An idle machine wakes on the deadline and finds nothing. That
        // is a quiet keyboard, not a broken line.
        for _ in 0..64 {
            b.parked(false);
            b.drained(false);
            assert_eq!(b.period_ms(), BACKSTOP_LIVENESS_MS, "quiet is not dead");
        }
    }

    /// The one observation that convicts the line: bytes were sitting in
    /// the output buffer and the deadline, not the interrupt, is what
    /// found them. This is the #452 shape exactly.
    #[test]
    fn bytes_the_line_never_announced_are_what_degrade_the_backstop() {
        let mut b = Backstop::new();
        b.parked(false); // the deadline won…
        b.drained(true); // …and there were keystrokes waiting
        assert_eq!(
            b.period_ms(),
            BACKSTOP_DEGRADED_MS,
            "the line missed those, so the deadline is the keyboard now"
        );
        // It stays degraded for as long as the line stays silent.
        b.parked(false);
        b.drained(true);
        assert_eq!(b.period_ms(), BACKSTOP_DEGRADED_MS);
    }

    /// Recovery is immediate and needs no debounce, which is what makes
    /// a false degrade cheap: a byte landing between a deadline
    /// resolving and the drain reading it costs one fast period, and the
    /// next real interrupt undoes it.
    #[test]
    fn one_real_interrupt_restores_the_liveness_period() {
        let mut b = Backstop::new();
        b.parked(false);
        b.drained(true);
        assert_eq!(b.period_ms(), BACKSTOP_DEGRADED_MS);

        b.parked(true);
        assert_eq!(
            b.period_ms(),
            BACKSTOP_LIVENESS_MS,
            "the line delivered — it is not the deadline's job again"
        );
        // And the drain that follows a *fired* park never convicts it,
        // however many reports it produces.
        b.drained(true);
        assert_eq!(b.period_ms(), BACKSTOP_LIVENESS_MS);
    }

    /// The liveness period is the idle cost of owning a keyboard, so it
    /// is pinned: a regression that dropped it back to tens of
    /// milliseconds would reintroduce the cadence poll this epic ruled
    /// out, and would do it silently.
    #[test]
    fn the_liveness_period_is_an_order_of_magnitude_slower_than_the_fallback() {
        assert!(
            BACKSTOP_LIVENESS_MS >= BACKSTOP_DEGRADED_MS * 10,
            "a backstop that polls at typing speed while healthy is a cadence poll"
        );
        assert!(
            BACKSTOP_DEGRADED_MS <= 20,
            "the fallback IS the keyboard when it runs, so it must feel like typing"
        );
    }

    #[test]
    fn the_two_ports_are_the_architectural_ones() {
        assert_eq!(port_of(kore_i8042::port::DATA), 0x60);
        assert_eq!(port_of(kore_i8042::port::STATUS), 0x64);
    }

    /// The handoff region must actually form a ring. A static byte
    /// array is aligned to 1 and `spsc::init` refuses anything under 64
    /// — a failure that would have shown up only on metal, as a relay
    /// that brought its controller up perfectly and then dropped every
    /// keystroke.
    #[test]
    fn the_handoff_region_forms_a_ring() {
        // SAFETY: the test is single-threaded and takes one handle.
        let window = unsafe { RELAY_REGION.view() };
        assert_eq!(
            window.as_ptr().align_offset(64),
            0,
            "the region must be 64-byte aligned or spsc::init refuses it"
        );
        spsc::init::<InputReport>(window, RELAY_CAPACITY).expect("forms a ring");
    }
}
