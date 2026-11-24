//! # `service` — the input service, the multiplexing router
//!
//! **One keyboard, many apps, one focus** (`res/DEVICE_SERVICE.md` §4):
//! [`InputService`] holds a fixed subscriber table (each entry the
//! service's **producer** half of one app's report queue) and a single
//! focus pointer; its serve step arbitrates the control plane
//! ([`Subscribe`](InputVerb::Subscribe) / [`Unsubscribe`](InputVerb::Unsubscribe)
//! / [`SetFocus`](InputVerb::SetFocus) / [`SetIndicators`](InputVerb::SetIndicators))
//! and the [`route`](InputService::route)
//! step steers a decoded [`InputReport`] to the focused subscriber's queue.
//! No per-report work crosses the session (§3): reports ride the shared
//! `ring::spsc`, the service only sets up, tears down, and steers.
//!
//! **Caller identity is structural** (`res/USERLAND.md` §5.8 — the #299
//! fix): every request carries the [`Endpoint`] it *arrived on*, stamped by
//! the router ([`InputHost`]) from the endpoint's own class — never a wire
//! field a subscriber could forge.  `SetFocus` and `SetIndicators` refuse on
//! any endpoint but the distinguished [`Focus`](Endpoint::Focus) one; the
//! wire-decode default
//! ([`Anonymous`](Endpoint::Anonymous)) is least privilege.  The router IS
//! the multiplexing generalization of the single-pair
//! [`ServeHost`](kore_rt::svc::ServeHost): a fan-in over per-endpoint pairs,
//! stamping identity by arrival.
//!
//! **The endowment is the claiming-driver shape** (`res/DEVICE_SERVICE.md`
//! §5/§10): a founded [`role::DMA_POOL`] plus the manifest-stamped device
//! broker face. Startup claims and class-probes the xHCI controller before
//! consuming that pool into `BoundDevice`; the bind then establishes a
//! remapped MSI-family route and brings the real
//! [`kore_xhci::Controller`] to
//! `Running`, enumerates every HID boot keyboard/mouse, and arms their
//! interrupt endpoints over the capability-backed register/DMA seams.
//! Readiness is advertised only after the service correlates its own
//! submitted No-op command off the event ring
//! ([`settle_interrupt_probe`](InputService::settle_interrupt_probe)),
//! waiting for it on the held route under a deadline; thereafter it
//! repeats `park → bounded drain → park` on the executor's
//! IRQ-or-doorbell combination.
//!
//! Bookkeeping is **allocation-free** (fixed tables + counters); a refusal
//! is counted, never silent, and delivers nothing.

pub mod admission;
mod diagnostics;
pub mod i2c_relay;
pub mod relay;
pub mod serve;
pub mod xhci_sources;

use kore_abi::keyboard::Locks;
use kore_abi::session::SlotRef;
use kore_abi::svc::InputReport;
use kore_abi::task::role;

use kore_i8042::HID_KEYBOARD as HID_PS2_KEYBOARD;

const HID_XHCI: u32 = match kore_i8042::eisa_id("PNP0D15") {
    Some(id) => id,
    None => panic!("PNP0D15 is a well-formed EISA id"),
};

/// Indices shared by the measured device declaration and its broker caller.
#[cfg(any(test, target_os = "none"))]
pub(crate) mod device_index {
    pub const PCI_XHCI: usize = 0;
    pub const I8042: usize = 1;
    pub const I2C_HID: usize = 2;
    pub const PLATFORM_XHCI: usize = 3;
    pub const PCI_I2C_HID: usize = 4;
}

use kore_rt::cap::{RouteToken, verbs};
use kore_rt::service::{Serve, SystemService};

use kore_driver::host::{
    CapPool, CapRegs, ClassQuiesce, DeviceBinding, DeviceResources, DmaActive, DmaPrepared,
    InterruptBlock, InterruptPolicy, PciBindingContract, PciClaimPosture, pci_selector_for_binding,
};
use kore_ring::event::EventSource;
use kore_ring::spsc::{self, Producer};
use kore_xhci::{
    Controller, Error as XhciError, InputDriver, InterruptProbe, PointerRegisterAccess,
    PreparedController,
};

pub use relay::{SOURCE_LEGACY, rung};
pub use serve::{InputHost, InputHostStats};

/// Subscriber-table ceiling — a fixed, allocation-free table; overflow
/// refuses ([`TableFull`](InputRefusal::TableFull)), never silently drops.
pub const MAX_INPUT_SUBSCRIBERS: usize = 16;

/// Client-pair ceiling the [`InputHost`] router fans in over — the
/// subscribing apps plus the distinguished focus endpoint.
pub const MAX_INPUT_ENDPOINTS: usize = 16;

/// Stable xHCI intake diagnostic carried through the input status protocol.
/// The high byte names the stage and the complete intake diagnostic occupies
/// the lower 56 bits. Zero is reserved for a completed bind.
pub mod xhci_bind {
    pub const PCI_CLAIM: u8 = 1;
    pub const PCI_PROBE: u8 = 2;
    pub const PLATFORM_CLAIM: u8 = 3;
    pub const PLATFORM_PROBE: u8 = 4;
    pub const BIND: u8 = 5;

    pub const REFUSED_NO_MATCH: u64 = 0x80;
    pub const REFUSED_TAKEN: u64 = 0x81;
    pub const REFUSED_DENIED: u64 = 0x82;
    pub const WRONG_SPECIES: u64 = 0x83;
    pub const REFUSED_OTHER: u64 = 0x85;
    /// A lawful root remains bound for hotplug, but launch discovery found no
    /// HID source. Keep this distinct from zero, which means HID-backed bind.
    pub const BOUND_NO_HID: u64 = kore_abi::input_status::USB_BOUND_NO_HID_DETAIL;

    /// Stage-0 (running) posture bits — what the firmware → OS handoff
    /// found in the controller, a fact about the machine, not a refusal.
    pub const RUNNING_DBC_DISABLED: u64 = kore_abi::input_status::USB_RUNNING_DBC_DISABLED_DETAIL;
    pub const RUNNING_DBC_STUCK: u64 = kore_abi::input_status::USB_RUNNING_DBC_STUCK_DETAIL;

    pub const fn pack(stage: u8, diagnostic: u64) -> u64 {
        ((stage as u64) << 56) | (diagnostic & 0x00ff_ffff_ffff_ffff)
    }

    /// The running state of a HID-backed bind: stage 0 carrying the
    /// handoff's posture bits (zero for a controller firmware left clean).
    pub const fn running(handoff: kore_xhci::Handoff) -> u64 {
        match handoff.dbc {
            kore_xhci::DbcHandoff::Absent | kore_xhci::DbcHandoff::Idle => 0,
            kore_xhci::DbcHandoff::Disabled => RUNNING_DBC_DISABLED,
            kore_xhci::DbcHandoff::Stuck => RUNNING_DBC_STUCK,
        }
    }
}

/// The default report-queue depth an app allocates (#299's proven
/// capacity) — power-of-two so the spsc index is one mask.
pub const INPUT_QUEUE_CAPACITY: u32 = 128;

/// Bytes an app-allocated report window must hold for one
/// [`INPUT_QUEUE_CAPACITY`]-deep [`InputReport`] ring (header + slots,
/// 64-byte-rounded) — what the app sizes its lease to.
pub const fn input_queue_bytes() -> usize {
    spsc::region_bytes::<InputReport>(INPUT_QUEUE_CAPACITY)
}

/// A founding-time endpoint index — the structural key of one client pair
/// (which subscriber it is), stamped by the [`InputHost`] on arrival.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EndpointId(pub u16);

/// **The structural caller identity** (`res/USERLAND.md` §5.8): which
/// endpoint a request arrived on.  It is the *router's* stamp — the wire
/// carries no identity, so a subscriber cannot express
/// [`Focus`](Self::Focus).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Endpoint {
    /// The distinguished focus-authority endpoint (the compositor / input
    /// manager) — the only endpoint `SetFocus` and `SetIndicators` are
    /// honored on.
    Focus,
    /// An ordinary subscriber endpoint, keyed by its founding index — may
    /// `Subscribe`/`Unsubscribe` its own subscriptions, never `SetFocus`.
    Subscriber(EndpointId),
    /// No endpoint context — the fail-closed wire-decode default (least
    /// privilege).  The router always restamps a real endpoint; this
    /// value only reaches `serve` through a mis-wired or direct caller,
    /// and refuses everything that needs identity.
    Anonymous,
}

impl Endpoint {
    /// May this endpoint steer focus?  Only the distinguished one.
    pub fn can_set_focus(self) -> bool {
        matches!(self, Endpoint::Focus)
    }

    /// Does this endpoint carry an ownable identity (may it subscribe)?
    fn is_endpoint(self) -> bool {
        !matches!(self, Endpoint::Anonymous)
    }
}

/// A service-private routing key handed back at `Subscribe` — **data, not
/// authority** (`res/USERLAND.md` §5.8): the authority to change focus is
/// the focus endpoint capability, never this key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubscriptionId(pub u32);

/// The input control-plane vocabulary (`res/USERLAND.md` §5.8).  The `via`
/// endpoint is the **structural identity** the router stamps; the client
/// constructors default it to [`Anonymous`](Endpoint::Anonymous) (the wire
/// carries no identity — the client cannot set its own privilege).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputRequest {
    /// The endpoint this request arrived on — the router's stamp.
    pub via: Endpoint,
    /// The requested verb.
    pub verb: InputVerb,
}

impl InputRequest {
    /// A `Subscribe` from the wire-decode default endpoint (the router
    /// restamps `via`).
    pub fn subscribe(queue: SlotRef) -> Self {
        Self {
            via: Endpoint::Anonymous,
            verb: InputVerb::Subscribe { queue },
        }
    }

    /// An `Unsubscribe` from the wire-decode default endpoint.
    pub fn unsubscribe(subscription: SubscriptionId) -> Self {
        Self {
            via: Endpoint::Anonymous,
            verb: InputVerb::Unsubscribe { subscription },
        }
    }

    /// A `SetFocus` from the wire-decode default endpoint.
    pub fn set_focus(target: Option<SubscriptionId>) -> Self {
        Self {
            via: Endpoint::Anonymous,
            verb: InputVerb::SetFocus { target },
        }
    }

    /// A `Status` from the wire-decode default endpoint. Needs no
    /// identity: it reads counters and steers nothing.
    pub fn status() -> Self {
        Self {
            via: Endpoint::Anonymous,
            verb: InputVerb::Status,
        }
    }

    /// A `SetIndicators` from the wire-decode default endpoint (the router
    /// restamps `via`; only the focus endpoint is honoured).
    pub fn set_indicators(source: u8, locks: Locks) -> Self {
        Self {
            via: Endpoint::Anonymous,
            verb: InputVerb::SetIndicators { source, locks },
        }
    }

    /// The same request as arriving on `via` — the router's authenticated
    /// endpoint identity stamp.
    pub fn via(self, via: Endpoint) -> Self {
        Self { via, ..self }
    }
}

/// The control verbs (`res/USERLAND.md` §5.8).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputVerb {
    /// Read one bounded input diagnostic row; no routing authority.
    Diagnostic { row: u8 },
    /// Attach the report-queue view rendezvous delivered into the service's
    /// arena at `queue`, and route the focused stream to it.
    Subscribe {
        /// The delivered writable report-window view's arena slot (#148).
        queue: SlotRef,
    },
    /// Stop routing to — and forget — a subscription (the owning endpoint
    /// or the focus authority only).
    Unsubscribe {
        /// The routing key `Subscribe` returned.
        subscription: SubscriptionId,
    },
    /// Steer subsequent reports to one subscription, or to nobody
    /// (`None`) — the **focus authority only** (§5.8).
    SetFocus {
        /// The subscription to focus, or `None` for "nobody".
        target: Option<SubscriptionId>,
    },
    /// Read the service's books — see [`RelayBooks`]. Steers nothing and
    /// names no subscription, so it needs no authority beyond being a
    /// client at all: the counters are diagnostics, not routing state.
    Status,
    /// Show `locks` on keyboard `source`'s indicators — the **focus
    /// authority only**, because lock state is the keymap owner's: the
    /// endpoint that steers focus is the one consuming the keys whose
    /// meaning the locks change.
    ///
    /// The answer is that the state is recorded, not that it is lit: the
    /// keyboard converges on it asynchronously (a PS/2 exchange on the
    /// relay's next wake, a USB output report completed off the event
    /// ring), and a replugged or reset keyboard is re-lit from the record
    /// without being asked again.
    SetIndicators {
        /// The wire source identity the keyboard reports under.
        source: u8,
        /// The state to show.
        locks: Locks,
    },
    /// Bytes for the device to **show** — a serial line's transmit side
    /// (`res/IMAGE_PROFILES.md` §4.3), the way
    /// [`SetIndicators`](Self::SetIndicators) is output a keyboard shows.
    /// Carried inline (`InputReqWire::OP_TRANSMIT`, at most
    /// `TRANSMIT_BYTES`); the keyboard router refuses it
    /// [`Unsupported`](InputRefusal::Unsupported).
    Transmit {
        /// The bytes; only the first `len` are live.
        bytes: [u8; kore_abi::svc::InputReqWire::TRANSMIT_BYTES],
        /// How many are live (`1..=TRANSMIT_BYTES`).
        len: u8,
    },
}

/// The input control-plane replies — **data only** (no capability rides an
/// input reply, §3.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputResponse {
    Diagnostic(kore_abi::input_status::InputDiagnostic),
    /// The queue is attached; route the focused stream with this key.
    Subscribed(SubscriptionId),
    /// The subscription was removed.
    Unsubscribed,
    /// Focus was changed.
    Focused,
    /// The indicator state was recorded for the keyboard to converge on.
    Indicated,
    /// How many of a `Transmit`'s bytes the device accepted, in order; a
    /// stalled line accepts fewer and the client sends the rest again.
    Transmitted { accepted: u8 },
    /// The service's books — see [`RelayBooks`].
    Status(RelayBooks),
    /// Refused without changing routing — a refusal is counted, never
    /// silent, and steers nothing.
    Refused(InputRefusal),
}

/// The legacy relay's books, as the control plane reports them.
///
/// These exist because the relay is otherwise **unobservable**: it is a
/// worker thread whose only output is reports, so "no keystroke arrived"
/// cannot be told apart from "the line never fired", "the bytes decoded
/// to nothing" or "the handoff queue was full" — three different bugs
/// with one symptom, on a machine whose only keyboard is the one that is
/// not working.
///
/// Read in order they separate cleanly: `parks` frozen is a relay that
/// stopped; `wakes == 0` against a climbing `parks` is a dead line
/// serviced entirely by the backstop; `wakes > 0 && events == 0` is a
/// decode yielding nothing, and `unnamed` then says whether the scancode
/// set is wrong; `dropped > 0` is a consumer falling behind. `rung != 0`
/// short-circuits all of it — bring-up never finished, and the rung
/// names where.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RelayBooks {
    /// The bring-up rung the relay stopped at; `0` is running.
    pub rung: u8,
    /// Line interrupts woken on.
    pub wakes: u32,
    /// Parks issued on the line — `wakes`'s denominator, and the relay's
    /// liveness witness.
    pub parks: u32,
    /// Reports pushed into the handoff queue.
    pub events: u32,
    /// Reports the handoff queue had no room for.
    pub dropped: u32,
    /// Bytes that decoded to a key carrying no HID meaning.
    pub unnamed: u32,
    /// Whether the router received a live xHCI controller capability.
    pub xhci_bound: bool,
    /// Packed xHCI intake state: `[63:56] stage`, `[55:0] exact diagnostic`.
    /// Zero means HID-backed bind; `BOUND_NO_HID` names a live controller
    /// retained for hotplug, and other nonzero values are intake refusals.
    pub xhci_bind: u64,
    /// HID boot sources enumerated and armed behind that controller.
    pub xhci_sources: u32,
    /// Ports or interfaces that could not be discovered or armed.
    pub xhci_discovery_errors: u32,
    /// Failed xHCI completions, decodes, source matches, or re-arms.
    pub xhci_irq_errors: u32,
    /// The two firmware-described HID-over-I2C workers. A zeroed slot means
    /// no such device was endowed; otherwise `state` packs the worker's
    /// current rung, protocol phase, failure class, and controller detail.
    pub i2c: [I2cRelayBooks; 2],
}

/// One HID-over-I2C worker's bounded diagnostic snapshot.
///
/// `state` is deliberately a single atomic publication on metal:
/// `[7:0] rung`, `[15:8] phase`, `[23:16] failure`, `[55:24] detail`,
/// and `[63:56]` recovery outcome. `rung == 0` means the worker reached
/// steady-state polling; a non-zero failure there records the most recent
/// runtime transfer failure without falsely claiming bring-up failed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct I2cRelayBooks {
    pub state: u64,
    pub transfers: u32,
    pub reports: u32,
}

/// One broker-claimed I2C controller and the immutable child topology read
/// while the input service still owns the controller capability.
///
/// `regs` precedes `device` so the mapped child is retired before its affine
/// parent. Workers receive only borrowed process-local geometry; the owning
/// service retains both authorities for its lifetime. In particular, a
/// spawned DesignWare worker has no task-level `geom::DEVICE` window and must
/// never try to install BAR0 for itself.
pub struct I2cClaim {
    pub regs: Option<kore_driver::host::CapRegs>,
    pub device: I2cDevice,
    pub kind: i2c_relay::ControllerKind,
    pub bindings: [Option<kore_abi::session::I2cHidReport>; i2c_relay::MAX_CHILDREN],
    pub count: usize,
}

/// The real broker-granted controller parent, without fabricating a platform
/// row for a PCI device or discarding the grant's lifetime.
pub enum I2cDevice {
    Pci(kore_rt::cap::DeviceToken),
    Platform(kore_rt::cap::PlatformToken),
}

impl I2cDevice {
    fn slot(&self) -> SlotRef {
        match self {
            Self::Pci(device) => device.slot(),
            Self::Platform(device) => device.slot(),
        }
    }
}

/// Why an input control request was refused (`res/USERLAND.md` §5.8) —
/// a closed vocabulary, 1-based discriminants (the `seg` precedent).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputRefusal {
    /// The delivered slot was absent, non-writable, or not a well-formed
    /// [`InputReport`] ring.
    BadQueue = 1,
    /// The bounded subscriber table is full.
    TableFull = 2,
    /// The routing key is stale or unknown (or not owned by the requesting
    /// endpoint).
    Unknown = 3,
    /// The endpoint lacks the authority for the verb — an ordinary
    /// subscriber attempting `SetFocus`, or a non-owner `Unsubscribe`
    /// (§5.8's structural gate).
    Denied = 4,
    /// The source has no such side: a keyboard router asked for a
    /// transmit stream.
    Unsupported = 5,
}

/// xHCI's class-specific bound owner. The generic host realizes only the
/// declared capability policy; this class owns the controller/input-core
/// composition and therefore controls its child-first teardown order.
pub struct XhciBound {
    input: InputDriver<CapPool>,
    controller: Controller<CapRegs, CapPool>,
    // Kept intact for xHCI's borrowed seams. An owning controller class may
    // instead take BAR/pool values from this set and restore them in unbind.
    resources: DeviceResources<DmaActive>,
}

impl XhciBound {
    /// What the firmware → OS handoff found and did on this controller.
    pub fn handoff(&self) -> kore_xhci::Handoff {
        self.controller.handoff()
    }
}

impl XhciBound {
    /// Launch-time evidence used to choose among several xHCs. The binding
    /// remains affine; selection reads only the driver's discovery census.
    #[cfg(target_os = "none")]
    pub(crate) fn selection_evidence(&self) -> (usize, kore_xhci::DiscoveryCensus) {
        (self.input.source_count(), self.input.discovery_census())
    }
}

/// xHCI after its complete DMA graph is programmed while Run/Stop and
/// interrupter enable remain clear. The generic host owns the only transition
/// from this state to [`XhciBound`].
pub struct XhciPrepared {
    controller: PreparedController<CapRegs, CapPool>,
}

pub struct XhciBinding;

impl PciBindingContract for XhciBinding {
    const BARS: kore_rt::service::PciBarMask = kore_rt::service::PciBarMask::BAR0;
    const INTERRUPTS: InterruptPolicy = InterruptPolicy::Required(InterruptBlock::SINGLE);
}

impl DeviceBinding for XhciBinding {
    type Pci = Self;
    type Config = PointerRegisterAccess;
    type Prepared = XhciPrepared;
    type Bound = XhciBound;
    type Error = XhciError;

    /// Every handshake below is paced by [`kore_rt::pace::ceding`] rather
    /// than a bare `core::hint::spin_loop`, and that is a boot-liveness
    /// property, not a nicety.
    ///
    /// This runs on thread 0 inside the founding claim, between two parks —
    /// which is to say inside the founder's own drive loop, whose yields
    /// into ring 3 are when the wake drain, the manifold service, and the
    /// **boot splash's only pump** get to run. xHCI bring-up is a ladder of
    /// handshakes each bounded at `MAX_POLLS` (500 000 probes): a BIOS
    /// handoff firmware never completes, a controller-not-ready bit that
    /// never clears, a port reset that hangs. Spun, each of those is a
    /// half-second during which the founder does not run and the machine's
    /// only console stops moving; several of them in a row is the
    /// seconds-long frozen splash a serial-less machine cannot explain.
    /// Paced, the same budget and the same refusal survive — the wait is
    /// merely interruptible.
    fn prepare(
        resources: &mut DeviceResources<DmaPrepared>,
        pointer_access: Self::Config,
    ) -> Result<Self::Prepared, Self::Error> {
        let regs = resources.bar(0).ok_or(XhciError::Unsupported)?;
        let pool = resources.pool().ok_or(XhciError::Unsupported)?;
        let controller = Controller::prepare_with_pointer_access(
            regs,
            pool,
            pointer_access,
            kore_driver::host::clock_now_ns,
            kore_rt::pace::ceding(),
        )?;
        Ok(XhciPrepared { controller })
    }

    fn start(
        prepared: Self::Prepared,
        resources: DeviceResources<DmaActive>,
    ) -> Result<Self::Bound, Self::Error> {
        let regs = resources.bar(0).ok_or(XhciError::Unsupported)?;
        let pool = resources.pool().ok_or(XhciError::Unsupported)?;
        let mut controller = prepared.controller.start(
            regs,
            kore_driver::host::clock_now_ns,
            kore_rt::pace::ceding(),
        )?;
        let input = InputDriver::discover(
            &mut controller,
            regs,
            pool,
            kore_driver::host::clock_now_ns,
            kore_rt::pace::ceding(),
        );
        Ok(XhciBound {
            input,
            controller,
            resources,
        })
    }

    fn quiesce(bound: &mut Self::Bound) -> Result<ClassQuiesce, Self::Error> {
        let regs = bound.resources.bar(0).ok_or(XhciError::Unsupported)?;
        bound.controller.quiesce(
            regs,
            kore_driver::host::clock_now_ns,
            kore_rt::pace::ceding(),
        )?;
        Ok(ClassQuiesce::Stopped)
    }

    fn unbind(bound: Self::Bound) -> DeviceResources<DmaActive> {
        let XhciBound {
            input,
            controller,
            resources,
        } = bound;
        drop(input);
        drop(controller);
        resources
    }
}

/// The input service's role bundle — **the DMA supply and nothing else**
/// (`res/DEVICE_SERVICE.md` §10).
///
/// The controllers are not here any more.  This row used to be founded
/// holding its xHCI function and its i8042 outright, which meant the
/// kernel had to *choose* them: match a class triple, read `CAPLENGTH` and
/// `PORTSC` through the candidate's BAR to reject a D3-parked or portless
/// controller, and size this pool from `HCSPARAMS2`.  All of that is xHCI
/// knowledge, and none of it belongs in ring 0.
///
/// So the devices are **claimed** instead, from the device broker, at
/// startup — and because the claimant is the driver, the reject tier that
/// used to force the kernel to understand xHCI is now just the driver
/// looking at its own controller and asking for the next one
/// ([`claim_devices`](crate::claim_devices)).
///
/// The pool stays founded because it is class-agnostic: a bounded,
/// 2 MiB-aligned supply the driver sub-allocates. It is sized generously
/// rather than exactly, which costs frames and not correctness (the pool
/// bump-allocates), and buys the kernel's independence from every driver
/// crate.
#[derive(kore_rt::Bundle)]
#[bundle(name = "driver")]
pub struct InputEndowment {
    /// The founded DMA supply the claimed device's window is opened over.
    #[bundle(role = role::DMA_POOL)]
    pub pool: kore_rt::cap::SupplyToken,
    /// This task's **self-kick** (`role::DOORBELL@2`) — a `ChanSend`
    /// into its own doorbell, carved by the founder for exactly one
    /// purpose: to be re-derived into a worker thread's arena, so the
    /// worker can wake the serve loop it feeds.
    ///
    /// A slot rather than a claimed token, because nothing here ever
    /// posts on it. Thread 0 is the doorbell's *receiver*; kicking
    /// itself would be a wake it is already awake for. Its whole use is
    /// as the subject of [`relay::kick_endowment`]'s derive.
    ///
    /// `Option` because the founding only carves it for a row that
    /// declares workers. This row does — so on a live machine it is
    /// always present — but a bind is the wrong place to enforce a
    /// manifest fact: a row that dropped its workers would refuse to
    /// bind for a reason that no longer named the decision. Absent
    /// endows nothing, and the relay says [`rung::NO_KICK`].
    ///
    /// Declared last: it holds no authority to retire, so it takes no
    /// position in the child-first drop order above.
    #[bundle(role = role::DOORBELL, index = 2, capture, optional)]
    pub kick: Option<SlotRef>,
}

/// One live subscription: its routing key, the endpoint that owns it, and
/// the service's producer half of the app's report ring.
struct Subscriber {
    id: SubscriptionId,
    owner: Endpoint,
    queue: Producer<'static, InputReport>,
}

/// The compositor's separately bounded data lane, independent of
/// keyboard focus. Counts attempted publications so overflow also
/// wakes the consumer to observe the ring's loss counter.
pub struct PointerLane<'a> {
    queue: Option<EventSource<'a, kore_abi::pointer::PointerState, InputReport>>,
    pub publications: u64,
    observed: u64,
    rejected: u64,
    resets: u64,
    sequences: [u32; 256],
    generations: [u32; 256],
    /// Each source's pointer class, as its `Record::Begin` carries it:
    /// [`xhci_sources::MOUSE_CLASS`] unless its device was admitted as
    /// something else ([`Self::set_class`]).
    classes: [u8; 256],
    pending: [InputReport; kore_abi::pointer::MAX_FRAME_RECORDS],
    pending_len: usize,
}

impl Default for PointerLane<'_> {
    fn default() -> Self {
        Self {
            queue: None,
            publications: 0,
            observed: 0,
            rejected: 0,
            resets: 0,
            sequences: [0; 256],
            generations: [1; 256],
            classes: [xhci_sources::MOUSE_CLASS; 256],
            pending: [InputReport::mouse(0, 0, 0, 0, 0); kore_abi::pointer::MAX_FRAME_RECORDS],
            pending_len: 0,
        }
    }
}

impl<'a> PointerLane<'a> {
    pub fn attach(&mut self, queue: EventSource<'a, kore_abi::pointer::PointerState, InputReport>) {
        self.queue = Some(queue);
    }

    /// The class a legacy mouse source's frames carry from its next report:
    /// set once its device is admitted, before it is armed to report.
    pub fn set_class(&mut self, source: u8, class: u8) {
        self.classes[usize::from(source)] = class;
    }

    /// Republish the lane's books into the channel state, then hand the
    /// channel back so the caller can end the batch.
    ///
    /// The kick policy is not this type's to make: `commit_and_kick` sends a
    /// notice only when the consumer actually parked, which is what keeps a
    /// busy compositor free of syscalls it does not need.  Publishing the
    /// books here rather than there keeps the counters and their meaning in
    /// one place.
    pub fn publish_books(
        &mut self,
    ) -> Option<&mut EventSource<'a, kore_abi::pointer::PointerState, InputReport>> {
        let (publications, rejected, resets) = (self.publications, self.rejected, self.resets);
        let channel = self.queue.as_mut()?;
        channel.state(|state| {
            state.publications = publications;
            state.rejected = rejected;
            state.resets = resets;
        });
        Some(channel)
    }

    fn route(&mut self, report: InputReport) -> Option<bool> {
        use kore_abi::pointer::{Record, is_pointer_kind};
        if !is_pointer_kind(report.kind) {
            return None;
        }
        self.observed = self.observed.saturating_add(1);
        // Pointer data must never fall through to a keyboard subscriber.
        let Some(queue) = self.queue.as_mut() else {
            self.rejected = self.rejected.saturating_add(1);
            return Some(false);
        };
        if report.kind == InputReport::KIND_MOUSE {
            if report.reserved != [0; 7] || report.data[3..] != [0; 3] {
                self.rejected = self.rejected.saturating_add(1);
                return Some(false);
            }
            let source = usize::from(report.source);
            if self.generations[source] == 0 {
                self.rejected = self.rejected.saturating_add(1);
                return Some(false);
            }
            self.sequences[source] = self.sequences[source].wrapping_add(1);
            let sequence = self.sequences[source];
            let packet = [
                Record::Begin {
                    sequence,
                    received_ns: kore_rt::clock::uptime().ok().map(|t| t.as_nanos()),
                    class: self.classes[source],
                },
                Record::Motion {
                    dx: i32::from(report.data[0] as i8),
                    dy: i32::from(report.data[1] as i8),
                },
                Record::Buttons(u64::from(report.state)),
                Record::Scroll {
                    horizontal: 0,
                    vertical: i32::from(report.data[2] as i8),
                    denominator: 1,
                    unit: 0,
                },
                Record::End {
                    sequence,
                    generation: self.generations[source],
                    records: 3,
                    reported_contacts: None,
                    complete: true,
                },
            ]
            .map(|r| r.encode(report.source));
            self.publications = self.publications.wrapping_add(packet.len() as u64);
            return Some(queue.push_slice_lossy(&packet));
        }
        let decoded = Record::decode(report);
        if let Ok(Record::Reset { all }) = decoded {
            self.resets = self.resets.saturating_add(1);
            if all || (self.pending_len != 0 && self.pending[0].source == report.source) {
                self.pending_len = 0;
            }
            self.publications = self.publications.wrapping_add(1);
            return Some(queue.push_lossy(report));
        }
        if matches!(decoded, Ok(Record::Begin { .. })) {
            if self.pending_len != 0 {
                self.rejected = self.rejected.saturating_add(1);
                self.resets = self.resets.saturating_add(1);
                self.publications = self.publications.wrapping_add(1);
                queue.push_lossy(Record::Reset { all: false }.encode(self.pending[0].source));
            }
            self.pending[0] = report;
            self.pending_len = 1;
            return Some(true);
        }
        if decoded.is_err()
            || self.pending_len == 0
            || self.pending_len >= self.pending.len()
            || self.pending[0].source != report.source
        {
            self.pending_len = 0;
            self.rejected = self.rejected.saturating_add(1);
            self.resets = self.resets.saturating_add(1);
            self.publications = self.publications.wrapping_add(1);
            queue.push_lossy(Record::Reset { all: true }.encode(report.source));
            return Some(false);
        }
        self.pending[self.pending_len] = report;
        self.pending_len += 1;
        if matches!(decoded, Ok(Record::End { .. })) {
            let count = self.pending_len;
            self.pending_len = 0;
            self.publications = self.publications.wrapping_add(count as u64);
            return Some(queue.push_slice_lossy(&self.pending[..count]));
        }
        Some(true)
    }

    fn reset_source(&mut self, source: u8) -> bool {
        let index = usize::from(source);
        // Never reuse a wrapped generation. Exhaustion requires rediscovery
        // under a fresh identity rather than accepting stale input.
        if self.generations[index] != 0 {
            self.generations[index] = self.generations[index].checked_add(1).unwrap_or(0);
        }
        self.route(kore_abi::pointer::Record::Reset { all: false }.encode(source))
            .unwrap_or(false)
    }

    #[cfg(test)]
    fn lost_upstream(&mut self) {
        let _ = self.route(kore_abi::pointer::Record::Reset { all: true }.encode(0));
    }
}

/// Occupancy snapshot (the sibling services' `*Stats` shape).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputStats {
    /// Enumerated and armed HID boot sources.
    pub sources: u32,
    /// Ports/interfaces that could not be discovered or armed.
    pub discovery_errors: u32,
    /// [`discovery_errors`](Self::discovery_errors) split by what each
    /// examination actually concluded — the difference between "this
    /// controller hosts no keyboard" and "this controller would not
    /// answer", which the total cannot express.
    pub discovery: kore_xhci::DiscoveryCensus,
    /// Interrupt-route wakeups handled by the resident relay.
    pub irq_wakes: u64,
    /// Wakes the executor resolved as REAL route deliveries
    /// (`Status::Irq`) — the MSI-landed witness.  `0` with a growing
    /// [`irq_wakes`](Self::irq_wakes) = a delivery-dead machine running
    /// kick-paced.
    pub irq_delivered: u64,
    /// Transfer events consumed from the xHCI event ring.
    pub irq_events: u64,
    /// Failed completions, decodes, source matches, or re-arms.
    pub irq_errors: u64,
    /// Halted endpoints cleared and resumed.
    ///
    /// A transfer error halts the endpoint, and a halted endpoint ignores
    /// its doorbell — so this counts the recoveries that keep a device
    /// answering. Steady non-zero says the link is marginal (cable, hub,
    /// port); non-zero followed by no further reports says the recovery
    /// itself is not taking, which is a different bug entirely.
    pub irq_halts_recovered: u64,
    /// Live subscriptions.
    pub subscribers: u32,
    /// Whether a subscription currently holds focus.
    pub focused: bool,
    /// Reports steered into the focused queue since bind.
    pub delivered: u64,
    /// Reports dropped (no focus, or the focused queue was full).
    pub dropped: u64,
    /// Control requests refused since bind — counted, never silent.
    pub refused: u32,
    /// LED output reports USB keyboards accepted.
    pub indicator_reports: u64,
    /// LED output reports USB keyboards refused — a keyboard without
    /// indicators STALLs its output report, so a non-zero count with keys
    /// still arriving is a fact about the keyboard, not a fault.
    pub indicator_failures: u64,
}

/// The input service: one router over every owned input edge, arbitrating a
/// bounded set of subscribers and steering reports to the focused one.
pub struct InputService {
    // The class binding owns the controller, endpoint DMA children, and the
    // capability resources as one affine teardown unit.
    //
    // `Option` carries two lawful states: `None` between `bind` and the
    // broker result, and `None` afterwards when the inventory has no PCI
    // xHCI satisfying this class's BAR/DMA/MSI contract. In the latter state
    // the control-plane router and optional legacy relay still serve.
    device: Option<kore_driver::host::BoundDevice<XhciBinding>>,
    /// Exact controller claim/probe/bind outcome retained even when xHCI is
    /// optional and HID-over-I2C keeps the service alive.
    xhci_bind: u64,
    xhci_identity: u32,
    xhci_sources: xhci_sources::SourceMap,
    /// The founded DMA supply, held until the claimed device's window is
    /// opened over it — `None` afterwards, because the realized
    /// [`CapPool`](kore_driver::host) owns it from then on.
    pool: Option<kore_rt::cap::SupplyToken>,
    /// The internal keyboard controller, when firmware described one.
    /// Held, never driven from this thread: the relay derives its own
    /// authority off it, because the I/O-port bitmap is keyed on the
    /// running *task* and thread 0 is not the relay.
    legacy: Option<kore_rt::cap::PlatformToken>,
    /// Platform I2C controllers with firmware-resolved HID children behind
    /// their exact MMIO windows. The service owns every root and any GENI
    /// mapping; workers receive only the authority their backend requires.
    i2c: [Option<I2cClaim>; i2c_relay::MAX_CONTROLLERS],
    /// This task's self-kick (`role::DOORBELL@2`), held solely as the
    /// subject the relay's wake-edge endowment derives from — see
    /// [`InputEndowment::kick`].
    kick: Option<SlotRef>,
    /// Thread 0's end of the relay handoff queue. `None` until
    /// `start_on_metal` forms it.
    relay_rx: Option<kore_ring::spsc::Consumer<'static, InputReport>>,
    /// Thread 0's consumer half of the HID-over-I2C worker handoff.
    i2c_rx: [Option<kore_ring::spsc::Consumer<'static, InputReport>>; i2c_relay::MAX_CONTROLLERS],
    i2c_dropped: [u64; i2c_relay::MAX_CONTROLLERS],
    subscribers: [Option<Subscriber>; MAX_INPUT_SUBSCRIBERS],
    pub pointer: PointerLane<'static>,
    focus: Option<SubscriptionId>,
    next_id: u32,
    irq_wakes: u64,
    irq_delivered: u64,
    irq_events: u64,
    irq_errors: u64,
    irq_halts_recovered: u64,
    delivered: u64,
    dropped: u64,
    refused: u32,
    indicator_reports: u64,
    indicator_failures: u64,
    /// Remaining discovery attempts before discovery goes purely
    /// event-driven (see [`drain_interrupt`](Self::drain_interrupt)).
    /// Refunded to [`DISCOVER_BUDGET`] by a genuine **connect** change —
    /// never by a Port Status Change edge on its own.
    discover_budget: u8,
}

/// Discovery attempts allowed without a connect-status change — the
/// **backstop**, not the mechanism.
///
/// What bounds discovery is the driver's per-port memo: a port whose device
/// has not changed is skipped whole, so a connected non-HID device (the
/// stick the machine booted from) is examined once and never again while it
/// stays plugged in.  This budget covers only what the memo cannot answer:
/// the initial attach whose PSC events bring-up may already have consumed,
/// and a port whose enumeration failed *inconclusively* (no memo is recorded
/// for those, so a few retries are still owed).
///
/// The history is worth keeping, because the first fix was this budget alone
/// and it did not hold.  A connected non-HID device whose discovery kept
/// failing re-ran port resets + full descriptor fetches on EVERY wake,
/// forever — a hot ring-3 loop that on a real Lenovo boot (USB stick, no
/// keyboard) showed up as an endless stream of `session refused … status=Irq`.
/// The budget was refunded by any Port Status Change edge, and *a port reset
/// raises one*: discovery's own side effect kept refunding the limiter meant
/// to bound it.  The refund is now gated on a real connect edge
/// ([`InputDriver::note_port_changes`](kore_xhci::InputDriver::note_port_changes)),
/// so our own resets no longer pay for themselves.
const DISCOVER_BUDGET: u8 = 8;

impl InputService {
    /// The claimed controller.
    ///
    /// Panics unless [`has_xhci`](Self::has_xhci) is true. Hardwareless
    /// routers never enter these methods; the production entry gates the
    /// readiness probe and IRQ park on that fact. Same posture as
    /// `BoundDevice::parts`'s own `expect` one level down.
    fn bound(&self) -> &XhciBound {
        self.device
            .as_ref()
            .expect("the input row serves only after its controller is claimed")
            .parts()
    }

    /// The claimed controller, mutably.
    fn bound_mut(&mut self) -> &mut XhciBound {
        self.device
            .as_mut()
            .expect("the input row serves only after its controller is claimed")
            .parts_mut()
    }

    /// Take the founded DMA supply — consumed by the entry loop's claim,
    /// which opens the claimed device's window over it.
    ///
    /// Affine: a second call yields `None`, so two claims can never open
    /// two windows over one supply.
    pub fn take_pool(&mut self) -> Option<kore_rt::cap::SupplyToken> {
        self.pool.take()
    }

    /// Adopt the controller the entry loop claimed from the broker
    /// (`res/DEVICE_SERVICE.md` §10), and the firmware-described keyboard
    /// beside it when the machine has one.
    ///
    /// The one door out of the unclaimed window. `None` is the explicit
    /// hardwareless-router posture: the machine exposed no supported xHCI,
    /// but the control plane and any legacy relay can still serve.
    pub fn adopt_device(
        &mut self,
        device: Option<kore_driver::host::BoundDevice<XhciBinding>>,
        legacy: Option<kore_rt::cap::PlatformToken>,
        i2c: [Option<I2cClaim>; i2c_relay::MAX_CONTROLLERS],
        xhci_bind: u64,
        xhci_identity: u32,
    ) {
        self.device = device;
        self.xhci_bind = xhci_bind;
        self.xhci_identity = xhci_identity;
        if let Some(device) = self.device.as_ref() {
            let input = &device.parts().input;
            for source in input.sources() {
                // Not announced: at bind nothing has subscribed yet, and
                // every keyboard's indicators are dark — what any seat
                // shows before its first key.
                let keyboard = input.kind(source) == Some(kore_xhci::ReportKind::Keyboard);
                if let Some((id, _)) = self.xhci_sources.admit(source, keyboard)
                    && !keyboard
                {
                    self.pointer
                        .set_class(id, xhci_sources::mouse_class(input.identity(source)));
                }
            }
        }
        self.legacy = legacy;
        self.i2c = i2c;
    }

    /// Whether this router owns a live xHCI binding.
    pub fn has_xhci(&self) -> bool {
        self.device.is_some()
    }

    /// Independently labeled controller coverage. Slots beyond the primary
    /// are not managed by this service; they are not evidence of absent USB
    /// hardware. Claiming them would currently steal controller ownership
    /// from other USB consumers such as the network service.
    pub fn xhci_books(&self, controller: usize) -> Option<xhci_sources::ControllerBooks> {
        use xhci_sources::{ControllerBooks, MAX_CONTROLLERS, STATE_NOT_MANAGED, UNKNOWN_IDENTITY};
        if controller >= MAX_CONTROLLERS {
            return None;
        }
        if controller >= xhci_sources::MANAGED_CONTROLLER_CAPACITY {
            return Some(ControllerBooks {
                state: STATE_NOT_MANAGED,
                identity: UNKNOWN_IDENTITY,
                ..ControllerBooks::default()
            });
        }
        let (sources, discovery_errors) = self.device.as_ref().map_or((0, 0), |device| {
            let input = &device.parts().input;
            (input.source_count() as u32, input.discovery_errors())
        });
        Some(ControllerBooks {
            state: self.xhci_bind,
            identity: self.xhci_identity,
            sources,
            discovery_errors,
            irq_wakes: self.irq_wakes,
            irq_delivered: self.irq_delivered,
            irq_events: self.irq_events,
            irq_errors: self.irq_errors,
            reports: self.xhci_sources.reports(),
            dropped: self.xhci_sources.dropped(),
            source_exhausted: self.xhci_sources.exhausted(),
        })
    }

    /// Occupancy snapshot.
    pub fn stats(&self) -> InputStats {
        let (sources, discovery_errors, discovery) = self.device.as_ref().map_or_else(
            || (0, 0, kore_xhci::DiscoveryCensus::default()),
            |device| {
                let bound = device.parts();
                (
                    bound.input.source_count() as u32,
                    bound.input.discovery_errors(),
                    bound.input.discovery_census(),
                )
            },
        );
        InputStats {
            sources,
            discovery_errors,
            discovery,
            irq_wakes: self.irq_wakes,
            irq_delivered: self.irq_delivered,
            irq_events: self.irq_events,
            irq_errors: self.irq_errors,
            irq_halts_recovered: self.irq_halts_recovered,
            subscribers: self.subscribers.iter().flatten().count() as u32,
            focused: self.focus.is_some(),
            delivered: self.delivered,
            dropped: self.dropped,
            refused: self.refused,
            indicator_reports: self.indicator_reports,
            indicator_failures: self.indicator_failures,
        }
    }

    /// The current focus (diagnostic / tests).
    pub fn focus(&self) -> Option<SubscriptionId> {
        self.focus
    }

    /// The endpoint that owns the focused subscription — the data-plane
    /// kick's addressee (`serve::InputHost::kick_focused`).  `None` when
    /// nothing holds focus.
    pub fn focused_owner(&self) -> Option<Endpoint> {
        let id = self.focus?;
        self.subscribers
            .iter()
            .flatten()
            .find(|s| s.id == id)
            .map(|s| s.owner)
    }

    /// The affine route the runtime's resident relay parks on. This borrow
    /// keeps route authority live across `WAIT_IRQ`; there is no numeric IRQ
    /// or kernel callback shortcut.
    pub fn interrupt_route(&self) -> &RouteToken {
        self.bound()
            .resources
            .route()
            .expect("xHCI binding requires an interrupt route")
    }

    /// Submit the deterministic xHCI No-op whose command-completion event is
    /// the service's startup MSI-family delivery proof. The affine witness is
    /// held across the route wait and consumed by
    /// [`complete_interrupt_probe`](Self::complete_interrupt_probe).
    pub fn arm_interrupt_probe(&mut self) -> InterruptProbe {
        let XhciBound {
            controller,
            resources,
            ..
        } = self.bound_mut();
        controller.arm_interrupt_probe(resources.bar(0).expect("xHCI binding requires BAR0"))
    }

    /// Settle the armed probe by **parking**, not spinning: wait on the
    /// held route until it fires or `deadline_ns` (absolute monotonic
    /// nanoseconds) passes, and poll the event ring once per wake.
    /// Reports whether the exact No-op completion arrived successfully.
    ///
    /// This is the honest shape of the readiness proof.  Its two
    /// predecessors each failed on real hardware in a different
    /// direction: parking on a bare `wait_interrupt` blocked readiness on
    /// the completion's *remapped MSI delivery*, so a machine whose
    /// delivery path does not fire parked forever inside the verb — no
    /// NACK, no beacon, and every terminal-chain row behind it starved;
    /// replacing that with a bounded spin burned a core through the
    /// founding chain and called the result a timeout without ever having
    /// waited.  The combined park is both: the CPU is yielded while the
    /// completion is outstanding, and the deadline is a real edge the
    /// service can refuse on.
    ///
    /// A wake polls once and then re-parks on the **same absolute**
    /// deadline — so the total wait is bounded no matter how many wakes
    /// arrive, and a spurious wake cannot extend it.
    ///
    /// The controller proof and the delivery proof stay separate: this
    /// claims the first (the event-ring completion), and the route —
    /// established at `start_on_metal`, held by the executor's
    /// IRQ-or-doorbell park — earns the second in steady state, where a
    /// delivery-dead machine degrades to kick-paced drains instead of
    /// never booting.
    pub fn settle_interrupt_probe(&mut self, probe: InterruptProbe, deadline_ns: u64) -> bool {
        use kore_rt::cap::verbs::InterruptWait;
        use kore_xhci::ProbePoll;

        let XhciBound {
            controller,
            resources,
            ..
        } = self.bound_mut();
        let route = resources
            .route()
            .expect("xHCI binding requires an interrupt route");
        let regs = resources.bar(0).expect("xHCI binding requires BAR0");
        let mut held = probe;
        loop {
            // The route is the preferred edge; the deadline only bounds
            // it.  A refusal (no transport, no run queue) is not a
            // completion — poll once more, then give up rather than
            // spin on a verb that cannot park.
            let waited = verbs::wait_interrupt_until(route, deadline_ns);
            // Poll on EVERY wake, whichever arm won — including the
            // deadline's.  A completion that landed without a delivered
            // interrupt is still a completion, and it is exactly what a
            // delivery-dead machine has to show for itself.
            held = match controller.poll_interrupt_probe(regs, held) {
                ProbePoll::Complete => return true,
                ProbePoll::Failed(_) => return false,
                ProbePoll::Pending(p) => p,
            };
            if !matches!(waited, Ok(InterruptWait::Fired)) {
                // Deadline, revoke, or a refused park: the command is
                // still in flight, so abort the ring before reporting —
                // otherwise the next command staged behind it inherits
                // the stall.
                controller.abandon_interrupt_probe(
                    regs,
                    held,
                    kore_driver::host::clock_now_ns,
                    kore_rt::pace::ceding(),
                );
                return false;
            }
        }
    }

    /// The failed probe's register snapshot for the NACK payload's detail
    /// bits — see
    /// [`Controller::interrupt_probe_detail`](kore_xhci::Controller::interrupt_probe_detail)
    /// for the bit map (USBSTS + CRR + IMAN.IP).
    pub fn interrupt_probe_detail(&self) -> u64 {
        let bound = self.bound();
        bound
            .controller
            .interrupt_probe_detail(bound.resources.bar(0).expect("xHCI binding requires BAR0"))
    }

    /// Drain one bounded xHCI event batch after a route wake, route valid HID
    /// reports into the focused subscriber queue, and re-arm each completed
    /// endpoint. The hot path is fixed-stack and allocation-free.
    ///
    /// Discovery is **event-driven, not per-wake**: the ring drains first
    /// (routing transfer events and latching any consumed Port Status
    /// Change), then the latched edge is resolved against PORTSC — a real
    /// connect change refunds the attach budget, a fresh plug deserving
    /// fresh attempts, while a PSC our own port reset raised buys nothing.
    /// Enumeration itself runs while the bounded [`DISCOVER_BUDGET`] lasts,
    /// and is a per-port no-op for every port whose device the driver has
    /// already examined.  Budget exhausted with no connect edge ⇒ quiescent
    /// until the hardware reports one.
    pub fn drain_interrupt(&mut self) -> bool {
        let Self {
            device,
            xhci_sources,
            pointer,
            subscribers,
            focus,
            irq_wakes,
            irq_events,
            irq_errors,
            irq_halts_recovered,
            delivered,
            dropped,
            discover_budget,
            indicator_reports,
            indicator_failures,
            ..
        } = self;
        let Some(device) = device.as_mut() else {
            return false;
        };
        let XhciBound {
            input,
            controller,
            resources,
        } = device.parts_mut();
        let regs = resources.bar(0).expect("xHCI binding requires BAR0");

        *irq_wakes = irq_wakes.saturating_add(1);
        // Drain FIRST: with no armed source this consumes (and acks) the
        // pending events — including the Port Status Change whose latch the
        // discovery gate below reads.  The old order discovered first, so
        // the plug event that justified the discovery was still unread when
        // the decision was made, forcing the every-wake fallback.
        let mut sink =
            |report| route_xhci_report(subscribers, *focus, delivered, dropped, pointer, report);
        // Detached interfaces retain their IDs until cancellation reaches
        // every affected lane. Retry once on this real wake, never by adding
        // a full consumer to the executor's pending-work loop.
        xhci_sources.retry_retired(&mut sink);
        let drained = input.drain_with_sources(
            controller,
            regs,
            kore_driver::host::clock_now_ns,
            kore_driver::host::relax,
            &mut |source, report| xhci_sources.route(source, report, &mut sink),
        );
        *irq_events = irq_events.saturating_add(drained.events as u64);
        *irq_errors = irq_errors.saturating_add(drained.errors as u64);
        *irq_halts_recovered = irq_halts_recovered.saturating_add(drained.halts_recovered as u64);
        *indicator_reports = indicator_reports.saturating_add(u64::from(drained.outputs));
        *indicator_failures = indicator_failures.saturating_add(u64::from(drained.output_failures));
        // A completed output report frees its device's control pipe: send
        // whatever state moved on while it was in flight.
        sync_indicators(input, controller, regs, xhci_sources);
        // A Port Status Change event is not by itself news — **our own port
        // reset raises one** — so the latch is only the cue to go look at
        // PORTSC.  The refund is owed to a real plug or unplug, which is
        // what `note_port_changes` reads out (and acknowledges) there.
        //
        // Read BEFORE any early return, and unconditionally. A change bit
        // that is never acknowledged stays latched forever, and the edge it
        // records is the only evidence a device arrived or left — so a path
        // that can skip this read can never learn either. That is precisely
        // what happened: the source-count check below used to sit above this
        // line, so the first keyboard to attach made the whole port-change
        // path unreachable for the rest of the boot. Unplug went unnoticed,
        // the dead source kept the count non-zero, and the re-plug that
        // followed was never enumerated.
        let edge = controller.take_port_change()
            && input.note_source_changes_with(controller, regs, &mut |source| {
                xhci_sources.retire(source, &mut sink);
            });
        if edge {
            *discover_budget = DISCOVER_BUDGET;
        }

        // With no real edge, a controller that already has its sources has
        // nothing to discover — the cheap path, and the whole reason the
        // check exists. It is a statement about *this wake*, not a permanent
        // one: a real edge falls through to re-enumerate, whether that adds a
        // source or retires one.
        if !edge && input.source_count() > 0 {
            return true;
        }
        let Some(remaining) = discover_budget.checked_sub(1) else {
            return false; // quiescent: only a real port edge re-opens discovery
        };
        *discover_budget = remaining;
        let discovered = input.discover_after_interrupt(
            controller,
            regs,
            resources
                .pool()
                .expect("xHCI binding requires its DMA pool"),
            // Paced, like the bring-up path: enumeration here runs on a
            // real port edge, and a port reset that hangs would otherwise
            // hold this serve loop — and every subscriber's keystrokes —
            // for the whole `MAX_POLLS` budget.
            kore_driver::host::clock_now_ns,
            kore_rt::pace::ceding(),
        );
        for source in input.sources() {
            let keyboard = input.kind(source) == Some(kore_xhci::ReportKind::Keyboard);
            match xhci_sources.admit(source, keyboard) {
                // Announced through the ordinary router, so a refusal is
                // counted like any undelivered report.
                Some((id, true)) if keyboard => {
                    route_xhci_report(
                        subscribers,
                        *focus,
                        delivered,
                        dropped,
                        pointer,
                        InputReport::keyboard(id, 0, [0; 6]),
                    );
                }
                Some((id, _)) if !keyboard => {
                    pointer.set_class(id, xhci_sources::mouse_class(input.identity(source)));
                }
                _ => {}
            }
        }
        // A keyboard replugged under an identity the keymap owner already
        // told is owed that state now; it will not be told again.
        sync_indicators(input, controller, regs, xhci_sources);
        if !discovered {
            *irq_errors = irq_errors.saturating_add(1);
            return false;
        }
        true
    }

    /// **Drain the relay thread's handoff queue** — the internal
    /// keyboard's half of the data plane, routed through exactly the
    /// same `route_report` the xHCI drain uses.
    ///
    /// That sameness is the design: the router is source-agnostic, so
    /// the second keyboard is a second *producer*, not a second routing
    /// path. Focus, subscription ownership, and the lossy-drop posture
    /// all apply to it without a line of new policy.
    ///
    /// Returns how many reports were routed, so the caller knows
    /// whether to kick the focused subscriber.
    pub fn drain_relay(&mut self) -> u32 {
        // Destructured once, as `drain_interrupt` does: the consumer and
        // the routing state are disjoint fields, and saying so to the
        // borrow checker is cheaper than working around it.
        let Self {
            relay_rx,
            pointer,
            subscribers,
            focus,
            delivered,
            dropped,
            ..
        } = self;
        let Some(rx) = relay_rx.as_mut() else {
            return 0;
        };
        let mut n = 0;
        while let Some(report) = rx.pop() {
            route_report(subscribers, *focus, delivered, dropped, pointer, report);
            n += 1;
        }
        n
    }

    /// Drain the HID-over-I2C worker through the same router policy as USB
    /// and i8042 reports.
    pub fn drain_i2c_relay(&mut self) -> u32 {
        let Self {
            i2c_rx,
            i2c_dropped,
            pointer,
            subscribers,
            focus,
            delivered,
            dropped,
            ..
        } = self;
        let mut total = 0;
        for (controller, (rx, lost)) in i2c_rx.iter_mut().zip(i2c_dropped.iter_mut()).enumerate() {
            let Some(rx) = rx.as_mut() else { continue };
            let before = rx.dropped();
            let mut batch = [InputReport::mouse(0, 0, 0, 0, 0); i2c_relay::RELAY_CAPACITY as usize];
            let mut n = 0;
            for report in i2c_relay::published_batch(rx) {
                batch[n] = report;
                n += 1;
            }
            let after = rx.dropped();
            total += n as u32;
            if before != *lost || after != before {
                *lost = after;
                // Loss in this ring invalidates this controller's sources only.
                // USB, PS/2 and other I2C controllers retain their contact history.
                for offset in 0..i2c_relay::SOURCES_PER_CONTROLLER {
                    pointer.reset_source(i2c_relay::source_base(controller, 0) + offset as u8);
                }
                continue;
            }
            for report in batch[..n].iter().copied() {
                route_report(subscribers, *focus, delivered, dropped, pointer, report);
            }
        }
        total
    }

    pub fn has_pending_relay_input(&self) -> bool {
        self.i2c_rx.iter().flatten().any(|q| q.outstanding() != 0)
            || self.relay_rx.as_ref().is_some_and(|q| q.outstanding() != 0)
    }

    /// Count one wake the executor resolved as a REAL route delivery
    /// (`Status::Irq`) — the relay's per-wake call when
    /// [`next_wake`](kore_rt::executor::next_wake) reports it.
    pub fn note_irq_delivery(&mut self) {
        self.irq_delivered = self.irq_delivered.saturating_add(1);
    }

    /// **Steer one decoded report to the focused subscriber's queue** — the
    /// data-plane push the device-drain half calls per report (`res/USERLAND.md`
    /// §5.8; sync, lock-free, allocation-free).  `false` when there is no
    /// focus or the focused queue was full (dropped-and-counted — the KINP
    /// lossy posture: drop the oldest *news*, never block the producer).
    pub fn route(&mut self, report: InputReport) -> bool {
        route_report(
            &mut self.subscribers,
            self.focus,
            &mut self.delivered,
            &mut self.dropped,
            &mut self.pointer,
            report,
        )
    }

    /// `Subscribe`: take possession of the delivered writable view and
    /// attach the producer half over the app's report ring — fail-closed
    /// (a window that is not a well-formed [`InputReport`] ring refuses
    /// [`BadQueue`](InputRefusal::BadQueue)).
    fn subscribe(&mut self, via: Endpoint, queue: SlotRef) -> InputResponse {
        // A real endpoint owns the subscription; the wire-decode default
        // (Anonymous) has no owner and cannot subscribe.
        if !via.is_endpoint() {
            return InputResponse::Refused(InputRefusal::Denied);
        }
        let Some(idx) = self.subscribers.iter().position(Option::is_none) else {
            return InputResponse::Refused(InputRefusal::TableFull);
        };
        let Some(view) = verbs::take_view(queue) else {
            return InputResponse::Refused(InputRefusal::BadQueue);
        };
        let Some(win) = verbs::map_view(view) else {
            return InputResponse::Refused(InputRefusal::BadQueue);
        };
        // SAFETY: the delivered writable view is now exclusively this
        // service's producer half of the app's report ring; the mapping is
        // stable for the subscription's life (the lease outlives it), and
        // the app touches the window only as the ring's consumer (the SPSC
        // discipline — one producer here, one consumer there).
        let bytes: &'static mut [u8] = unsafe { &mut *win.as_ptr() };
        let Ok(producer) = spsc::producer::<InputReport>(bytes) else {
            return InputResponse::Refused(InputRefusal::BadQueue);
        };
        let id = SubscriptionId(self.next_id);
        self.next_id += 1;
        self.subscribers[idx] = Some(Subscriber {
            id,
            owner: via,
            queue: producer,
        });
        InputResponse::Subscribed(id)
    }

    /// `Unsubscribe`: the **owning** endpoint (or the focus authority) drops
    /// a subscription; a non-owner refuses [`Denied`](InputRefusal::Denied).
    /// Losing the focused subscription steers focus to nobody.
    fn unsubscribe(&mut self, via: Endpoint, subscription: SubscriptionId) -> InputResponse {
        let Some(idx) = self
            .subscribers
            .iter()
            .position(|s| s.as_ref().is_some_and(|s| s.id == subscription))
        else {
            return InputResponse::Refused(InputRefusal::Unknown);
        };
        let owner = self.subscribers[idx].as_ref().map(|s| s.owner);
        // Structural authorization: the requesting endpoint must own it, or
        // be the focus authority.
        if owner != Some(via) && !via.can_set_focus() {
            return InputResponse::Refused(InputRefusal::Denied);
        }
        self.subscribers[idx] = None;
        if self.focus == Some(subscription) {
            self.focus = None;
        }
        InputResponse::Unsubscribed
    }

    /// `SetIndicators`: record the state `source`'s keyboard should show
    /// and start it converging — the focus authority only.
    ///
    /// A USB identity's state is recorded whether or not a keyboard holds it
    /// now, so one replugged under it is lit with the seat's current state.
    /// [`Unknown`](InputRefusal::Unknown) names a source with no indicator
    /// path here at all — the internal keyboard on a machine without one,
    /// or HID-over-I2C (whose output reports are not driven yet) — and so
    /// one the keymap owner need not ask again.
    fn set_indicators(&mut self, via: Endpoint, source: u8, locks: Locks) -> InputResponse {
        match indicator_path(via, source, self.legacy.is_some()) {
            Err(refusal) => return InputResponse::Refused(refusal),
            Ok(IndicatorPath::Legacy) => {
                relay::set_indicators(locks);
                return InputResponse::Indicated;
            }
            Ok(IndicatorPath::Usb) => {}
        }
        if !self.xhci_sources.set_indicators(source, locks) {
            return InputResponse::Refused(InputRefusal::Unknown);
        }
        let Self {
            device,
            xhci_sources,
            ..
        } = self;
        if let Some(device) = device.as_mut() {
            let XhciBound {
                input,
                controller,
                resources,
            } = device.parts_mut();
            let regs = resources.bar(0).expect("xHCI binding requires BAR0");
            sync_indicators(input, controller, regs, xhci_sources);
        }
        InputResponse::Indicated
    }

    /// `SetFocus`: **the structural privilege** — only the distinguished
    /// focus endpoint may steer (§5.8).  Any other endpoint refuses
    /// [`Denied`](InputRefusal::Denied), and it cannot forge the endpoint
    /// (identity is the router's stamp, off-wire).  A non-`None` target
    /// must name a live subscription (fail-closed).
    fn set_focus(&mut self, via: Endpoint, target: Option<SubscriptionId>) -> InputResponse {
        if !via.can_set_focus() {
            return InputResponse::Refused(InputRefusal::Denied);
        }
        if let Some(id) = target
            && !self.subscribers.iter().flatten().any(|s| s.id == id)
        {
            return InputResponse::Refused(InputRefusal::Unknown);
        }
        self.focus = target;
        InputResponse::Focused
    }
}

/// A claimed controller is not yet a runnable relay. Only successful queue
/// preparation admits its worker; the legacy relay retains its own absent-
/// device diagnostic. Taking the real consumers keeps this predicate tied to
/// the handoff the service can actually drain.
fn relay_worker_enabled(
    worker: usize,
    i2c_rx: &[Option<spsc::Consumer<'_, InputReport>>; i2c_relay::MAX_CONTROLLERS],
) -> bool {
    worker == 0
        || worker
            .checked_sub(1)
            .and_then(|i| i2c_rx.get(i))
            .is_some_and(Option::is_some)
}

impl SystemService for InputService {
    /// The terminal path (#809 §L): halt and tear down the claimed xHC.  The
    /// i8042 and I2C-HID relays hold no binding and are knowingly outside
    /// the sweep (§L.6).
    async fn quiesce_devices(&mut self, _deadline_ns: u64) -> kore_rt::service::QuiesceSweep {
        let mut sweep = kore_rt::service::QuiesceSweep::NONE;
        if let Some(device) = self.device.take() {
            device.teardown().record(&mut sweep);
        }
        sweep
    }
    const NAME: &'static str = "input";
    type Endowment = InputEndowment;
    /// **The four input-controller selectors.**
    ///
    /// Index 0 is an optional PCI xHCI controller — registers over BAR0,
    /// DMA through a live IOMMU binding, completions over MSI(-X). A machine
    /// whose integrated xHCI is an ACPI MMIO device (the Snapdragon X case)
    /// still founds the router without fabricating a PCI identity. Index 1 is
    /// the machine's internal keyboard controller, an i8042 on the legacy
    /// bus, matched by its ACPI `_HID` and demanding an interrupt line
    /// (a relay that cannot park has no lawful way to work). Index 2 is an
    /// optional firmware-owned I2C controller: one catalog-fixed MMIO window,
    /// explicitly without DMA, and accepted only when firmware correlates at
    /// least one HID child with that exact window. Its completion line is
    /// optional at admission: GENI supports bounded FIFO polling, while the
    /// DesignWare worker checks its own routed-completion requirements.
    /// Index 3 is the ACPI `PNP0D15` xHCI peer, requiring both its real line
    /// and its platform requester/IOMMU binding before claiming DMA authority.
    ///
    /// One service, several device edges, rather than competing manifest
    /// rows: the terminal must receive all keyboards and pointers through one
    /// router, one
    /// focus table, and one line discipline, or the two would compete
    /// for the terminal instead of feeding it.
    ///
    /// The i8042 is **not required**. A machine may route its built-in
    /// keyboard over USB or I2C and declare no 8042 at all; such a
    /// machine founds this row with index 1 vacant and works, with one
    /// keyboard, exactly as it did before.
    const DEVICES: &'static [kore_rt::service::DeviceRequest] = &[
        kore_rt::service::DeviceRequest {
            selector: kore_rt::service::DeviceSelector::Pci(
                pci_selector_for_binding::<XhciBinding>(
                    0x0c,
                    0x03,
                    0x30,
                    kore_rt::service::PciIdentitySelector::ANY,
                    PciClaimPosture::RouteCapable,
                ),
            ),
            required: false,
        },
        kore_rt::service::DeviceRequest {
            selector: kore_rt::service::DeviceSelector::Platform {
                hid: HID_PS2_KEYBOARD,
                irq: true,
                dma: false,
            },
            required: false,
        },
        kore_rt::service::DeviceRequest {
            selector: kore_rt::service::DeviceSelector::PlatformI2cHid { irq: false },
            required: false,
        },
        kore_rt::service::DeviceRequest {
            selector: kore_rt::service::DeviceSelector::Platform {
                hid: HID_XHCI,
                irq: true,
                dma: true,
            },
            required: false,
        },
        kore_rt::service::DeviceRequest {
            selector: kore_rt::service::DeviceSelector::PciI2cHid { irq: true },
            required: false,
        },
    ];

    /// Who stops each of the five (#809 §L.5): the two xHCI roots are class
    /// bindings; the i8042 and both I2C-HID controllers are held raw by relay
    /// workers, issue no DMA, and are knowingly outside the sweep (§L.6).
    const DEVICE_CLASSES: &'static [kore_rt::service::DeviceClass] = &[
        kore_driver::host::device_class::<XhciBinding>(),
        kore_rt::service::DeviceClass::RelayHeld,
        kore_rt::service::DeviceClass::RelayHeld,
        kore_driver::host::device_class::<XhciBinding>(),
        kore_rt::service::DeviceClass::RelayHeld,
    ];

    /// One PS/2 worker and one independently endowed worker per I2C controller.
    /// Vacant or unsuccessfully prepared I2C slots consume neither a thread
    /// nor a capability budget.
    const WORKERS: &'static [kore_rt::service::Worker] = &[
        relay::WORKER,
        i2c_relay::WORKERS[0],
        i2c_relay::WORKERS[1],
        i2c_relay::WORKERS[2],
        i2c_relay::WORKERS[3],
    ];

    fn worker_enabled(&self, worker: usize) -> bool {
        relay_worker_enabled(worker, &self.i2c_rx)
    }

    /// Each relay is founded holding **two derives**: its device and the
    /// service self-kick. For i8042, the first is this service's
    /// platform device, off which it claims its own ports and
    /// establishes its own line, and this task's self-kick, off which it
    /// wakes the serve loop. Two, because a thread that can read the
    /// keyboard but not announce a keystroke is as dead as one that
    /// cannot read it — and the first version declared only the first,
    /// which is precisely how a QEMU boot got as far as `[pin] gsi=1`
    /// and then refused at [`rung::NO_KICK`].
    ///
    /// Each is conditional on its own subject, and independently: a
    /// machine whose firmware described no i8042 declares neither (the
    /// thread still spawns, finds no device, and parks with that named
    /// in the status row — the honest report for a machine with no
    /// legacy keyboard), and a founding that carved no self-kick
    /// declares only the device. Neither absence is a bind failure;
    /// both are states the relay's rungs already name.
    fn worker_endowment(&self, worker: usize, out: &mut kore_rt::service::WorkerEndowment) {
        match worker {
            0 => {
                if let Some(dev) = self.legacy.as_ref() {
                    out.push(relay::endowment(dev.slot()));
                }
                if let Some(kick) = self.kick {
                    out.push(relay::kick_endowment(kick));
                }
            }
            1..=i2c_relay::MAX_CONTROLLERS => {
                if let Some(claim) = self.i2c[worker - 1].as_ref()
                    && i2c_relay::worker_needs_device(claim.kind)
                {
                    out.push(i2c_relay::endowment(claim.device.slot(), claim.kind));
                }
                if let Some(kick) = self.kick {
                    out.push(i2c_relay::kick_endowment(kick));
                }
            }
            _ => {}
        }
    }

    fn bind(e: InputEndowment) -> Self {
        Self {
            // Unclaimed until the entry loop asks the broker
            // (`res/DEVICE_SERVICE.md` §10).
            device: None,
            xhci_bind: 0,
            xhci_identity: xhci_sources::UNKNOWN_IDENTITY,
            xhci_sources: xhci_sources::SourceMap::new(0),
            legacy: None,
            i2c: core::array::from_fn(|_| None),
            pool: Some(e.pool),
            kick: e.kick,
            relay_rx: None,
            i2c_rx: [const { None }; i2c_relay::MAX_CONTROLLERS],
            i2c_dropped: [0; i2c_relay::MAX_CONTROLLERS],
            subscribers: [const { None }; MAX_INPUT_SUBSCRIBERS],
            pointer: PointerLane::default(),
            focus: None,
            next_id: 0,
            irq_wakes: 0,
            irq_delivered: 0,
            irq_events: 0,
            irq_errors: 0,
            irq_halts_recovered: 0,
            delivered: 0,
            dropped: 0,
            refused: 0,
            indicator_reports: 0,
            indicator_failures: 0,
            discover_budget: DISCOVER_BUDGET,
        }
    }

    /// Form the relay handoff queue before any worker exists to push
    /// into it — `spawn_workers` runs strictly after this.
    fn start_on_metal(
        &mut self,
        _rt: &kore_rt::entry::Rt,
    ) -> Result<(), kore_rt::service::StartFailure> {
        self.relay_rx = relay::open();
        for (index, claim) in self.i2c.iter_mut().enumerate() {
            let Some(claim) = claim else {
                continue;
            };
            // Install controller registers in the owning task before worker
            // creation. Spawned workers share this address space but are
            // intentionally stamped no task-level DEVICE window; asking one
            // to map a PCI BAR therefore commits as `Denied` before probe.
            // The worker receives the already-mapped geometry and derives
            // only its own scoped completion route from the device endowment.
            let window = match (&claim.device, claim.kind) {
                (I2cDevice::Pci(device), i2c_relay::ControllerKind::PciDesignWare) => {
                    kore_rt::cap::verbs::map_bar(device, 0)
                }
                (
                    I2cDevice::Platform(device),
                    i2c_relay::ControllerKind::PlatformDesignWare
                    | i2c_relay::ControllerKind::PlatformGeni,
                ) => kore_rt::cap::verbs::map_platform_mmio(device),
                _ => {
                    i2c_relay::note_claim(index, i2c_relay::rung::BAD_DEVICE);
                    continue;
                }
            };
            let mapping = match window {
                Ok(window) => {
                    let regs = kore_driver::host::CapRegs::from_window(window);
                    let mapping = Some((regs.base(), regs.len() as u64));
                    claim.regs = Some(regs);
                    mapping
                }
                Err(fault) => {
                    i2c_relay::map_failed(index, fault.code());
                    continue;
                }
            };
            // SAFETY: called before spawn_workers; each slot is prepared once.
            // The owning claim retains its mapping until the service exits.
            self.i2c_rx[index] = unsafe {
                i2c_relay::prepare(index, claim.kind, claim.bindings, claim.count, mapping)
            };
        }
        // A queue that could not be formed is fatal to that source but not
        // to the router: other input edges still work, and each relay reports
        // its own refusal. Not a reason to refuse the whole row.
        Ok(())
    }

    fn serve(&mut self, req: InputRequest, sv: &mut Serve<'_, InputResponse>) {
        let rep = match req.verb {
            InputVerb::Subscribe { queue } => self.subscribe(req.via, queue),
            InputVerb::Unsubscribe { subscription } => self.unsubscribe(req.via, subscription),
            InputVerb::SetFocus { target } => self.set_focus(req.via, target),
            InputVerb::Status => {
                let mut books = relay::books();
                let stats = self.stats();
                books.xhci_bound = self.has_xhci();
                books.xhci_bind = self.xhci_bind;
                books.xhci_sources = stats.sources;
                books.xhci_discovery_errors = stats.discovery_errors;
                books.xhci_irq_errors = u32::try_from(stats.irq_errors).unwrap_or(u32::MAX);
                books.i2c = i2c_relay::books();
                InputResponse::Status(books)
            }
            InputVerb::Diagnostic { row } => self
                .diagnostic(row)
                .map(InputResponse::Diagnostic)
                .unwrap_or(InputResponse::Refused(InputRefusal::Unknown)),
            InputVerb::SetIndicators { source, locks } => {
                self.set_indicators(req.via, source, locks)
            }
            // The router routes keys; it shows no text.
            InputVerb::Transmit { .. } => InputResponse::Refused(InputRefusal::Unsupported),
        };
        if matches!(rep, InputResponse::Refused(_)) {
            self.refused += 1;
        }
        sv.reply(rep);
    }
}

/// Which keyboard edge an indicator request for a source reaches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IndicatorPath {
    /// The internal keyboard, through its relay.
    Legacy,
    /// A USB keyboard, through the xHCI identity table.
    Usb,
}

/// Authorize an indicator request and name the edge it reaches. Only the
/// focus endpoint may ask; the PS/2 identity exists only when the machine
/// has the controller; HID-over-I2C identities have no output path yet.
fn indicator_path(
    via: Endpoint,
    source: u8,
    has_legacy: bool,
) -> Result<IndicatorPath, InputRefusal> {
    if !via.can_set_focus() {
        return Err(InputRefusal::Denied);
    }
    match source {
        SOURCE_LEGACY if has_legacy => Ok(IndicatorPath::Legacy),
        SOURCE_LEGACY => Err(InputRefusal::Unknown),
        source if source >= i2c_relay::SOURCE_I2C_BASE => Err(InputRefusal::Unknown),
        _ => Ok(IndicatorPath::Usb),
    }
}

/// Converge every live USB keyboard on the lock state its identity was
/// asked to show. Each call is a comparison per keyboard and at most one
/// submitted output report per idle control pipe — never a wait.
fn sync_indicators(
    input: &mut InputDriver<CapPool>,
    controller: &mut Controller<CapRegs, CapPool>,
    regs: &CapRegs,
    sources: &xhci_sources::SourceMap,
) {
    for (source, locks) in sources.indicators() {
        let _ = input.sync_leds(controller, regs, source, locks.bits());
    }
}

/// USB resets arrive tagged with the same mapped identity as reports. Advance
/// that identity's legacy-mouse generation before publishing its reset, so a
/// later interface that reuses the map slot cannot inherit contact history.
fn route_xhci_report(
    subscribers: &mut [Option<Subscriber>; MAX_INPUT_SUBSCRIBERS],
    focus: Option<SubscriptionId>,
    delivered: &mut u64,
    dropped: &mut u64,
    pointer: &mut PointerLane<'_>,
    report: InputReport,
) -> bool {
    if matches!(
        kore_abi::pointer::Record::decode(report),
        Ok(kore_abi::pointer::Record::Reset { all: false })
    ) {
        return pointer.reset_source(report.source);
    }
    route_report(subscribers, focus, delivered, dropped, pointer, report)
}

fn route_report(
    subscribers: &mut [Option<Subscriber>; MAX_INPUT_SUBSCRIBERS],
    focus: Option<SubscriptionId>,
    delivered: &mut u64,
    dropped: &mut u64,
    pointer: &mut PointerLane<'_>,
    report: InputReport,
) -> bool {
    if let Some(landed) = pointer.route(report) {
        return landed;
    }
    let Some(focus) = focus else {
        *dropped = dropped.saturating_add(1);
        return false;
    };
    let landed = subscribers
        .iter_mut()
        .flatten()
        .find(|s| s.id == focus)
        .is_some_and(|sub| sub.queue.push_lossy(report));
    if landed {
        *delivered = delivered.saturating_add(1);
    } else {
        *dropped = dropped.saturating_add(1);
    }
    landed
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    use kore_rt::service::{DeviceSelector, PciIdentitySelector};

    #[test]
    fn broker_indices_resolve_the_complete_optional_device_table() {
        let requests = <InputService as SystemService>::DEVICES;
        assert_eq!(requests.len(), 5);
        assert!(requests.iter().all(|request| !request.required));
        assert_eq!(
            requests[device_index::PCI_XHCI].selector,
            DeviceSelector::Pci(pci_selector_for_binding::<XhciBinding>(
                0x0c,
                0x03,
                0x30,
                PciIdentitySelector::ANY,
                PciClaimPosture::RouteCapable,
            )),
        );
        assert_eq!(
            requests[device_index::I8042].selector,
            DeviceSelector::Platform {
                hid: HID_PS2_KEYBOARD,
                irq: true,
                dma: false,
            },
        );
        assert_eq!(
            requests[device_index::I2C_HID].selector,
            DeviceSelector::PlatformI2cHid { irq: false },
            "polling-capable controllers must reach their backend's admission",
        );
        assert_eq!(
            requests[device_index::PLATFORM_XHCI].selector,
            DeviceSelector::Platform {
                hid: kore_i8042::eisa_id("PNP0D15").unwrap(),
                irq: true,
                dma: true,
            },
            "the platform probe must have a declared, fully endowed selector",
        );
        assert_eq!(
            requests[device_index::PCI_I2C_HID].selector,
            DeviceSelector::PciI2cHid { irq: true },
            "PCI SerialIO needs a native controller capability and INTx, not DMA/MSI",
        );
    }

    #[test]
    fn relay_workers_require_their_own_prepared_consumer() {
        #[repr(align(64))]
        struct Ring([u8; 4096]);

        assert_eq!(
            <InputService as SystemService>::WORKERS.len(),
            i2c_relay::MAX_CONTROLLERS + 1,
        );
        for prepared in 0..i2c_relay::MAX_CONTROLLERS {
            let mut ring = Ring([0; 4096]);
            let (_tx, rx) = spsc::split::<InputReport>(&mut ring.0, 16).unwrap();
            let mut queues = core::array::from_fn(|_| None);
            assert!(relay_worker_enabled(0, &queues));
            for worker in 1..=i2c_relay::MAX_CONTROLLERS {
                assert!(!relay_worker_enabled(worker, &queues));
            }
            queues[prepared] = Some(rx);
            for worker in 1..=i2c_relay::MAX_CONTROLLERS {
                assert_eq!(
                    relay_worker_enabled(worker, &queues),
                    worker == prepared + 1,
                    "one prepared source must not admit its unprepared siblings",
                );
            }
            assert!(!relay_worker_enabled(
                i2c_relay::MAX_CONTROLLERS + 1,
                &queues
            ));
            assert!(!relay_worker_enabled(usize::MAX, &queues));
            drop(queues[prepared].take());
            assert!(!relay_worker_enabled(prepared + 1, &queues));
        }
    }
}

#[cfg(test)]
mod pointer_tests {
    use super::*;

    // Already 64-byte aligned, which the pointer channel needs too: it places
    // its seqlock'd state and wake words on their own cache lines.
    #[repr(align(64))]
    struct Ring([u8; 4096]);

    #[test]
    fn an_emulated_mouse_frames_as_virtual_and_a_physical_one_as_a_mouse() {
        use kore_abi::pointer::Record;
        use kore_xhci::UsbIdentity;
        let qemu = UsbIdentity {
            vendor: 0x0627,
            product: 0x0001,
        };
        let logitech = UsbIdentity {
            vendor: 0x046d,
            product: 0xc077,
        };
        assert_eq!(xhci_sources::mouse_class(Some(qemu)), xhci_sources::VIRTUAL_CLASS);
        assert_eq!(xhci_sources::mouse_class(Some(logitech)), xhci_sources::MOUSE_CLASS);
        // Another of QEMU's vendor's products is not its HID emulation.
        let other = UsbIdentity {
            vendor: 0x0627,
            product: 0x0002,
        };
        assert_eq!(xhci_sources::mouse_class(Some(other)), xhci_sources::MOUSE_CLASS);
        assert_eq!(xhci_sources::mouse_class(None), xhci_sources::MOUSE_CLASS);

        let mut ring = Ring([0; 4096]);
        let (tx, mut rx) = kore_ring::event::split::<kore_abi::pointer::PointerState, InputReport>(
            &mut ring.0,
            64,
        )
        .unwrap();
        let mut pointer = PointerLane::default();
        pointer.attach(tx);
        pointer.set_class(7, xhci_sources::mouse_class(Some(qemu)));
        for (source, class) in [(7, xhci_sources::VIRTUAL_CLASS), (8, xhci_sources::MOUSE_CLASS)] {
            assert_eq!(pointer.route(InputReport::mouse(source, 0, 5, -5, 0)), Some(true));
            match Record::decode(rx.pop().unwrap()) {
                Ok(Record::Begin { class: framed, .. }) => assert_eq!(framed, class),
                other => panic!("expected a frame's begin, got {other:?}"),
            }
            while let Some(record) = rx.pop() {
                if matches!(Record::decode(record), Ok(Record::End { .. })) {
                    break;
                }
            }
        }
    }

    #[test]
    fn mapped_usb_detach_reuses_identity_with_new_generation_not_its_siblings() {
        use kore_abi::pointer::Record;
        use kore_xhci::HidSource;
        let mut ring = Ring([0; 4096]);
        let (tx, mut rx) = kore_ring::event::split::<kore_abi::pointer::PointerState, InputReport>(
            &mut ring.0,
            64,
        )
        .unwrap();
        let mut pointer = PointerLane::default();
        pointer.attach(tx);
        let mut subscribers = core::array::from_fn(|_| None);
        let (mut delivered, mut dropped) = (0, 0);
        let mut sources = xhci_sources::SourceMap::new(0);
        let first = HidSource {
            port: 64,
            interface: 3,
        };
        let sibling = HidSource {
            port: 64,
            interface: 4,
        };
        let first_id = sources.register(first).unwrap();
        let sibling_id = sources.register(sibling).unwrap();
        let mut sink = |report| {
            route_xhci_report(
                &mut subscribers,
                None,
                &mut delivered,
                &mut dropped,
                &mut pointer,
                report,
            )
        };
        for source in [first, sibling] {
            assert!(sources.route(source, InputReport::mouse(0, 1, 2, -3, 0), &mut sink));
        }
        for id in [first_id, sibling_id] {
            for _ in 0..4 {
                assert_eq!(rx.pop().unwrap().source, id);
            }
            let end = rx.pop().unwrap();
            assert_eq!(end.source, id);
            assert!(matches!(
                Record::decode(end),
                Ok(Record::End { generation: 1, .. })
            ));
        }
        sources.retire(first, &mut sink);
        assert_eq!(
            rx.pop(),
            Some(Record::Reset { all: false }.encode(first_id))
        );
        assert_eq!(rx.pop(), None);
        let replacement = HidSource {
            port: 255,
            interface: 255,
        };
        assert_eq!(sources.register(replacement), Some(first_id));
        assert_eq!(sources.register(sibling), Some(sibling_id));
        for source in [replacement, sibling] {
            assert!(sources.route(source, InputReport::mouse(128, 0, 1, 0, 0), &mut sink));
        }
        for (id, generation) in [(first_id, 2), (sibling_id, 1)] {
            for _ in 0..4 {
                assert_eq!(rx.pop().unwrap().source, id);
            }
            let end = rx.pop().unwrap();
            assert_eq!(end.source, id);
            assert!(
                matches!(Record::decode(end), Ok(Record::End { generation: actual, .. }) if actual == generation)
            );
        }
        assert_eq!(rx.pop(), None);
    }

    #[test]
    fn only_the_focus_authority_reaches_a_keyboards_indicators() {
        for via in [
            Endpoint::Anonymous,
            Endpoint::Subscriber(EndpointId(0)),
            Endpoint::Subscriber(EndpointId(7)),
        ] {
            for source in [1, SOURCE_LEGACY, 200] {
                assert_eq!(
                    indicator_path(via, source, true),
                    Err(InputRefusal::Denied),
                    "{via:?} naming {source}"
                );
            }
        }
        let via = Endpoint::Focus;
        assert_eq!(
            indicator_path(via, SOURCE_LEGACY, true),
            Ok(IndicatorPath::Legacy)
        );
        assert_eq!(
            indicator_path(via, SOURCE_LEGACY, false),
            Err(InputRefusal::Unknown),
            "no controller, no internal keyboard"
        );
        assert_eq!(indicator_path(via, 1, false), Ok(IndicatorPath::Usb));
        assert_eq!(
            indicator_path(via, xhci_sources::source_id(3, 30).unwrap(), true),
            Ok(IndicatorPath::Usb)
        );
        assert_eq!(
            indicator_path(via, i2c_relay::SOURCE_I2C_BASE, true),
            Err(InputRefusal::Unknown),
            "no HID-over-I2C output path"
        );
    }

    #[test]
    fn wrapped_legacy_generation_cannot_be_revived_by_another_usb_reset() {
        let mut ring = Ring([0; 4096]);
        let (tx, rx) = kore_ring::event::split::<kore_abi::pointer::PointerState, InputReport>(
            &mut ring.0,
            16,
        )
        .unwrap();
        let mut pointer = PointerLane::default();
        pointer.attach(tx);
        pointer.generations[1] = u32::MAX;
        assert!(pointer.reset_source(1));
        assert_eq!(pointer.generations[1], 0);
        assert!(pointer.reset_source(1));
        assert_eq!(
            pointer.generations[1], 0,
            "generation exhaustion is permanent for this identity"
        );
        assert_eq!(
            pointer.route(InputReport::mouse(1, 0, 1, 0, 0)),
            Some(false)
        );
        assert_eq!(
            rx.outstanding(),
            2,
            "only the resets, never a reused-generation frame"
        );
    }

    #[test]
    fn pointer_routes_without_keyboard_focus_and_overflow_still_requests_a_wake() {
        let mut ring = Ring([0; 4096]);
        let (tx, mut rx) = kore_ring::event::split::<kore_abi::pointer::PointerState, InputReport>(
            &mut ring.0,
            16,
        )
        .unwrap();
        let mut pointer = PointerLane::default();
        pointer.attach(tx);
        let mut subscribers = core::array::from_fn(|_| None);
        let (mut delivered, mut dropped) = (0, 0);
        let mouse = InputReport::mouse(1, 1, 3, -2, 1);
        for _ in 0..3 {
            assert!(route_report(
                &mut subscribers,
                None,
                &mut delivered,
                &mut dropped,
                &mut pointer,
                mouse
            ));
        }
        assert!(!route_report(
            &mut subscribers,
            None,
            &mut delivered,
            &mut dropped,
            &mut pointer,
            mouse
        ));
        assert_eq!(pointer.publications, 20);
        assert_eq!(rx.dropped(), 5, "a whole frame was dropped, not a prefix");
        for sequence in 1..=3 {
            use kore_abi::pointer::Record;
            assert_eq!(
                Record::decode(rx.pop().unwrap()).unwrap(),
                Record::Begin {
                    sequence,
                    received_ns: None,
                    class: 0
                }
            );
            for _ in 0..3 {
                rx.pop().unwrap();
            }
            assert_eq!(
                Record::decode(rx.pop().unwrap()).unwrap(),
                Record::End {
                    sequence,
                    generation: 1,
                    records: 3,
                    reported_contacts: None,
                    complete: true
                }
            );
        }
        assert_eq!(rx.pop(), None);
        assert!(!route_report(
            &mut subscribers,
            None,
            &mut delivered,
            &mut dropped,
            &mut pointer,
            InputReport::keyboard(1, 0, [0; 6])
        ));
        assert_eq!(
            pointer.publications, 20,
            "keyboard did not enter compositor lane"
        );
        assert_eq!((delivered, dropped), (0, 1));
    }

    #[test]
    fn pointer_frames_are_atomic_and_upstream_loss_is_visible() {
        use kore_abi::pointer::Record;
        let mut ring = Ring([0; 4096]);
        let (tx, mut rx) = kore_ring::event::split::<kore_abi::pointer::PointerState, InputReport>(
            &mut ring.0,
            16,
        )
        .unwrap();
        let mut pointer = PointerLane::default();
        pointer.attach(tx);
        let start = Record::Begin {
            sequence: 1,
            received_ns: Some(123),
            class: 0,
        }
        .encode(128);
        assert_eq!(pointer.route(start), Some(true));
        assert_eq!(pointer.route(Record::Buttons(0).encode(128)), Some(true));
        assert_eq!(rx.pop(), None, "no frame prefix is published");
        pointer.lost_upstream();
        assert_eq!(
            Record::decode(rx.pop().unwrap()),
            Ok(Record::Reset { all: true })
        );
        // A stale END cannot revive the discarded prefix.
        assert_eq!(
            pointer.route(
                Record::End {
                    sequence: 1,
                    generation: 1,
                    records: 1,
                    reported_contacts: None,
                    complete: true
                }
                .encode(128)
            ),
            Some(false)
        );
        assert_eq!(
            Record::decode(rx.pop().unwrap()),
            Ok(Record::Reset { all: true })
        );
        pointer.reset_source(1);
        rx.pop().unwrap();
        assert_eq!(pointer.route(InputReport::mouse(1, 0, 0, 0, 0)), Some(true));
        for _ in 0..4 {
            rx.pop().unwrap();
        }
        assert!(matches!(
            Record::decode(rx.pop().unwrap()),
            Ok(Record::End { generation: 2, .. })
        ));
    }

    #[test]
    fn descriptor_to_relay_to_compositor_preserves_samples_and_unrelated_reset() {
        use kore_abi::pointer::{MAX_FRAME_RECORDS, Record};
        use kore_axiom::pointer::{Dispatch, PointerSeat, Viewport};
        // One touchpad contact: tip, 16-bit ID, unsigned 16-bit X/Y.
        const DESCRIPTOR: &[u8] = &[
            0x05, 0x0d, 0x09, 0x05, 0xa1, 0x01, 0x09, 0x22, 0xa1, 0x02, 0x09, 0x42, 0x15, 0, 0x25,
            1, 0x75, 1, 0x95, 1, 0x81, 2, 0x75, 7, 0x81, 3, 0x09, 0x51, 0x26, 0xff, 0x7f, 0x75, 16,
            0x81, 2, 0x05, 1, 0x09, 0x30, 0x09, 0x31, 0x26, 0x10, 0x27, 0x95, 2, 0x81, 2, 0xc0,
            0xc0,
        ];
        let table = kore_i2c_hid::parse::build_table(DESCRIPTOR).unwrap();
        let mut ring = Ring([0; 4096]);
        let (tx, mut rx) = kore_ring::event::split::<kore_abi::pointer::PointerState, InputReport>(
            &mut ring.0,
            128,
        )
        .unwrap();
        let mut pointer = PointerLane::default();
        pointer.attach(tx);
        let mut seat = PointerSeat::new(Viewport::new(1000, 800).unwrap());
        let mut records = [InputReport::mouse(0, 0, 0, 0, 0); MAX_FRAME_RECORDS];
        let mut frames = 0;
        for (sequence, x) in [(1, 100u16), (2, 200u16)] {
            let [lo, hi] = x.to_le_bytes();
            let data = [1, 0xe8, 3, lo, hi, 0xf4, 1];
            let len = table
                .decode_framed_into(
                    &data,
                    128,
                    1,
                    sequence,
                    Some(sequence as u64 * 1_000_000),
                    &mut records,
                )
                .unwrap();
            assert_eq!(pointer.route(records[0]), Some(true));
            assert_eq!(rx.pop(), None);
            // A USB port reset must not tear an unrelated in-progress I2C envelope.
            pointer.reset_source(1);
            assert_eq!(
                Record::decode(rx.pop().unwrap()),
                Ok(Record::Reset { all: false })
            );
            for &record in &records[1..len] {
                assert_eq!(pointer.route(record), Some(true));
            }
            while let Some(record) = rx.pop() {
                seat.feed_with(record, &mut |event| {
                    if let Dispatch::Frame(frame) = event {
                        frames += 1;
                        assert_eq!(frame.contacts[0].id, 1000);
                        assert_eq!(frame.contacts[0].x, i32::from(x));
                        assert_eq!(frame.contacts[0].confidence, None);
                        assert_eq!(frame.timestamp.unwrap().0, sequence as u64 * 1_000_000);
                    }
                });
            }
        }
        assert_eq!(frames, 2);
        assert_eq!(seat.rejected_frames, 0);
        assert_eq!(seat.position().x, 510);
    }

    /// Two fingers on a precision touchpad, through the production decoder,
    /// the relay lane and the compositor's seat with its touchpad policy:
    /// they scroll, in pixels, toward the top as content follows them down,
    /// and never move the pointer.
    #[test]
    fn two_fingers_through_the_real_decoder_scroll_the_compositor_seat() {
        use kore_abi::pointer::MAX_FRAME_RECORDS;
        use kore_axiom::pointer::{
            Dispatch, PointerSeat, ScrollUnit, SeatEventKind, Viewport, touchpad::TouchpadPolicy,
        };
        // Two Finger collections (tip, 16-bit ID, X and Y up to 10000), then
        // a contact count.
        const FINGER: &[u8] = &[
            0x05, 0x0d, 0x09, 0x22, 0xa1, 0x02, 0x09, 0x42, 0x15, 0, 0x25, 1, 0x75, 1, 0x95, 1,
            0x81, 2, 0x75, 7, 0x81, 3, 0x09, 0x51, 0x26, 0xff, 0x7f, 0x75, 16, 0x81, 2, 0x05, 1,
            0x09, 0x30, 0x09, 0x31, 0x26, 0x10, 0x27, 0x95, 2, 0x81, 2, 0xc0,
        ];
        let mut descriptor = std::vec![0x05, 0x0d, 0x09, 0x05, 0xa1, 0x01];
        descriptor.extend_from_slice(FINGER);
        descriptor.extend_from_slice(FINGER);
        descriptor.extend_from_slice(&[0x05, 0x0d, 0x09, 0x54, 0x25, 0x7f, 0x75, 8, 0x95, 1, 0x81, 2, 0xc0]);
        let table = kore_i2c_hid::parse::build_table(&descriptor).unwrap();
        let mut ring = Ring([0; 4096]);
        let (tx, mut rx) = kore_ring::event::split::<kore_abi::pointer::PointerState, InputReport>(
            &mut ring.0,
            128,
        )
        .unwrap();
        let mut pointer = PointerLane::default();
        pointer.attach(tx);
        let mut seat =
            PointerSeat::with_policy(Viewport::new(1000, 800).unwrap(), TouchpadPolicy::default());
        let start = seat.position();
        let mut records = [InputReport::mouse(0, 0, 0, 0, 0); MAX_FRAME_RECORDS];
        let (mut vertical, mut horizontal, mut moved) = (0, 0, false);
        for step in 0..=24u32 {
            let y = (2000 + 100 * step as u16).to_le_bytes();
            let mut data = std::vec::Vec::new();
            for (id, x) in [(1u16, 4000u16), (2, 6000)] {
                data.push(1);
                data.extend_from_slice(&id.to_le_bytes());
                data.extend_from_slice(&x.to_le_bytes());
                data.extend_from_slice(&y);
            }
            data.push(2);
            let len = table
                .decode_framed_into(
                    &data,
                    128,
                    1,
                    step + 1,
                    Some(1_000_000_000 + u64::from(step) * 8_333_333),
                    &mut records,
                )
                .unwrap();
            for &record in &records[..len] {
                assert_eq!(pointer.route(record), Some(true));
            }
            while let Some(record) = rx.pop() {
                seat.feed_with(record, &mut |event| {
                    let Dispatch::Pointer(event) = event else { return };
                    match event.kind {
                        SeatEventKind::Scroll {
                            horizontal: h,
                            vertical: v,
                            unit,
                        } => {
                            assert_eq!(unit, ScrollUnit::Pixel);
                            horizontal += h;
                            vertical += v;
                        }
                        SeatEventKind::Motion => moved = true,
                        _ => {}
                    }
                });
            }
        }
        assert_eq!(seat.rejected_frames, 0);
        assert!(!moved && seat.position() == start, "two fingers never point");
        assert_eq!(horizontal, 0, "a straight stroke stays on its axis");
        // 2400 units of a 10000-unit axis: 240 reference pixels, less the
        // 2 mm (24 pixels) that decided the gesture.
        assert!((vertical - 216).abs() <= 12, "scrolled {vertical}");
    }
}
