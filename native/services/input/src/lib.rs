//! # kore-svc-input — the input service
//!
//! **The multiplexing keyboard/mouse router** (`res/USERLAND.md` §5.8;
//! `res/DEVICE_SERVICE.md` §4/§7 — the "one device, many clients" shape):
//! the #299 successor, rebuilt on the Phase R/M/A/S foundations.  One
//! keyboard, many subscribing apps, one focused app: the service arbitrates
//! (`Subscribe` / `Unsubscribe` / `SetFocus`) and steers decoded HID reports
//! to the focused subscriber's queue — it does no per-report work.
//!
//! **The two planes stay split** (`res/DEVICE_SERVICE.md` §3):
//!
//! - the **control plane** crosses the service session ([`service`]): a
//!   fixed subscriber table, a focus pointer, the closed
//!   [`InputRequest`]/[`InputResponse`] vocabulary — data-only replies, no
//!   capability ever rides one (§3.2);
//! - the **data plane** never crosses it: reports ride a
//!   [`ring::spsc<InputReport>`](kore_ring::spsc) in an **app-allocated**
//!   window (the surviving #299 "KINP" queue, homed on Phase A's spsc,
//!   which names it its first production instance).  The app keeps the
//!   consumer; the service holds the producer sibling (`Subscribe` names
//!   the arena slot rendezvous delivered the app's writable view into).
//!
//! **The #299 caller-identity gap closes structurally** (§5.8): `SetFocus`
//! privilege is carried by *which endpoint a request arrives on*, not a
//! trusted parameter — the compositor holds a distinguished focus endpoint,
//! an ordinary subscriber's endpoint cannot `SetFocus`.  Identity is the
//! router's stamp ([`Endpoint`], threaded through the [`InputService`]
//! serve step), never a wire field a subscriber could forge — the thing
//! #299 could not test, now testable (the service's unit suite).
//!
//! One concern per module (`res/USERLAND.md` §1.2 rule 4; `main.rs` — the
//! process entry — arrives with the founder's spawn, #96/#298, exactly as
//! its swap/image/exec/mm siblings):
//!
//! - [`service`] — the serve step: [`InputService`] over its subscriber
//!   table + focus, the endpoint-stamped [`InputRequest`] vocabulary, and
//!   the [`InputHost`] fan-in router (the multiplexing generalization of
//!   the single-pair [`ServeHost`](kore_rt::svc::ServeHost)).
//! - [`client`] — the client face: the async [`InputClient`] (§4 pinned),
//!   and the [`RingInput`] ring + channel proxy (whose endpoint
//!   carries the caller's identity — the router's structural stamp).
//! - [`wire`] — the fail-closed enum ⇄ wire codecs over
//!   [`kore_abi::svc`]'s frozen forms (§5.3).
//!
//! The **xHCI report producer** (the device edge, §3.1a/§5.8) lives in
//! `kore_xhci::input`, but this service owns and drives it: startup enumerates
//! and arms HID boot endpoints when a supported xHCI exists, then the
//! resident task repeats `WAIT_IRQ → bounded drain/ack/decode/route/re-arm →
//! WAIT_IRQ`. PCI controllers carry an identity-selected MMIO policy; an
//! ACPI/DT-described controller uses the generic split-dword policy.
//! Discovery may allocate its source table from the founded heap; the
//! interrupt path is fixed-stack, mock-free, and has no per-report kernel
//! crossing.

#![no_std]

#[cfg(test)]
extern crate std;

#[cfg(any(test, target_os = "none"))]
use kore_abi::session::I2cHidReport;
#[cfg(any(test, target_os = "none"))]
use kore_rt::cap::verbs::InventoryReadError;

pub mod client;
pub mod service;
pub mod wire;

/// This row's append-only NACK rung for a missing executor doorbell.
///
/// The generic entry macro accepts it through `attest(...)` so migrating the
/// service off its private executor does not renumber metal diagnostics.
#[cfg(target_os = "none")]
pub const ENTRY_ATTEST_RUNG: u64 = 3;

/// Run the **input service** as the fan-in [`InputHost`] router it was built
/// to be (the `#299` gap's entry): the multiplexing control plane
/// (Subscribe / SetFocus, each request stamped with the arrival endpoint's
/// structural identity) as the executor root, and the xHCI IRQ drain
/// folded into the same loop — replacing the relay-only
/// `run_interrupt_service` entry once the control plane exists.
///
/// Wake model — the **IRQ-or-doorbell park** (`WAIT_DELIVERY | WAIT_IRQ`,
/// the ABI's combined form): the idle park arms the standing doorbell
/// post AND the task's held remapped MSI-family route under one park token, so
/// the first of {client kick, route fire, route revoke} wakes the loop —
/// no deadline, no grace bound, no poll.  A bare `WAIT_IRQ` park cannot
/// be woken by a client's doorbell kick (a startup Subscribe would stall
/// until the user types), and a bare doorbell park cannot be woken by the
/// device; the composite is the only honest steady state for a service
/// that is both planes at once.  Every wake drains the event ring and
/// serves every pending request; the route's pending-fire coalescing
/// (the route table's `pending` word) means a fire between parks is
/// never lost.
///
/// Readiness is a deterministic controller proof: the service submits an
/// xHCI No-op command, parks on its held route **with a deadline**
/// ([`READINESS_BUDGET_NS`]) and polls the event ring once per wake,
/// correlating and acknowledging that command's completion *before* the
/// beacon.  It never depends on a user moving a mouse or on a connected
/// HID endpoint producing unsolicited traffic — and never on the route
/// actually delivering: a delivery-dead machine spends its budget
/// parked, not spinning, then NACKs rung 12 with the controller's
/// register snapshot instead of wedging the founding chain.  The steady
/// state after it is the combined loop above.
///
/// The endpoint bank is v1-small by construction: endpoint 0 is the
/// founding client (the shell) and carries the **focus authority**
/// ([`Endpoint::Focus`]) — the terminal-manager tier multiplexes
/// subscribers behind it later; the router's identity machinery is
/// unchanged by that.
/// The bring-up ladder's rung codes — the payload of a `COOKIE_REFUSED`
/// NACK, printed by the founder as `service REFUSED (NACK) at rung N`.
/// One code per refusal site, in ladder order, so a failing metal boot
/// names its rung on screen instead of spending the readiness budget in a
/// silent spin.  Renumber nothing; append only.
///
/// **Payload convention**: bits `[7:0]` are the rung; bits `[63:8]` may
/// carry rung-specific DETAIL (the founder prints it as hex beside the
/// rung).  Rung 12 packs the controller's register snapshot there
/// (`Controller::interrupt_probe_detail`).  All rungs stay `< 256`.
///
/// **This map decodes NACKs only.**  The readiness beacon carries a
/// payload too — the row's non-fatal *note*, printed as
/// `note: rung N` beside the ready line — and that one decodes against
/// [`service::rung`], the i8042 relay's map.  The two are read apart by
/// the line they appear on: `REFUSED (NACK)` is this map, `ready … note:`
/// is the relay's.  Same packing convention, deliberately, so neither
/// needs a second decoder ring.
#[cfg(target_os = "none")]
mod rung {
    /// 1 — endowment resolve: device claim, DMA pool, aperture maps.
    pub const RUNG_ENDOWMENT: u64 = 1;
    /// 2 — `start_on_metal`: xHCI bring-up + interrupt-route establish.
    pub const RUNG_START_ON_METAL: u64 = 2;
    /// 3 — doorbell recv-pin missing (NACK degrades to the spin).
    #[allow(dead_code)]
    pub const RUNG_DOORBELL: u64 = super::ENTRY_ATTEST_RUNG;
    /// 4 — the pair ring window (`SVC_RING` take/map).
    pub const RUNG_RING_WINDOW: u64 = 4;
    /// 5 — ring formatting (`serve_rings` magic check).
    pub const RUNG_RING_FORMAT: u64 = 5;
    /// 6 — the client kick endpoint (`SVC_SEND`).
    pub const RUNG_KICK_ENDPOINT: u64 = 6;
    /// 7 — the return-pair recv half (`SVC_RETURN@0`).
    pub const RUNG_RETURN_RECV: u64 = 7;
    /// 8 — the return-pair landing half (`SVC_RETURN@1`).
    pub const RUNG_RETURN_LAND: u64 = 8;
    /// 10 — host/endpoint attach.
    pub const RUNG_HOST_ATTACH: u64 = 10;
    /// 11 — RETIRED (2026-07-29): readiness no longer parks on the
    /// remapped delivery of the probe's completion — that blocking wait is
    /// exactly what wedged the first full metal boot (delivery-dead
    /// hardware ⇒ eternal park inside the verb, no NACK possible).  The
    /// number stays reserved so old photos still decode.
    #[allow(dead_code)]
    pub const RUNG_IRQ_PROBE_WAIT: u64 = 11;
    /// 12 — the probe fired but its completion event did not match.
    pub const RUNG_IRQ_PROBE_COMPLETE: u64 = 12;
    /// 13 — RETIRED: the private executor returned (never happened on metal).
    #[allow(dead_code)]
    pub const RUNG_EXECUTOR_EXIT: u64 = 13;
    /// 14 — the return post would not stand (stage or commit refused).
    /// Beaconing past this silently is how the first metal boots stranded
    /// every subscriber: the terminal row's Subscribe transfer has nowhere
    /// to land, retries until its convergence bound, and the whole
    /// terminal chain fails while the router reports healthy.
    pub const RUNG_RETURN_POST_STAND: u64 = 14;
    /// 15 — a declared worker thread could not be spawned (#420).
    /// Fatal: the row declared a topology it does not have, and serving
    /// anyway would advertise a second keyboard that nothing is reading.
    /// A relay that spawns and then *refuses* is a different thing
    /// entirely — it reports its own rung in the status row and the
    /// service serves on with one keyboard.
    pub const RUNG_WORKER_SPAWN: u64 = 15;
    /// The device broker path failed, or a granted PCI xHCI could not be brought
    /// up (`res/DEVICE_SERVICE.md` §10). A genuine `NoMatch` is instead the
    /// non-fatal hardwareless-router posture (readiness note 17).
    ///
    /// Distinct from [`RUNG_ENDOWMENT`] on purpose: the endowment rung
    /// means the founder did not hand this row what its profile declared,
    /// and this one means the row asked another *service* for hardware
    /// and did not get it.  On a serial-less machine that is the whole
    /// difference between "the manifest is wrong" and "this machine has
    /// no controller I can drive".
    pub const RUNG_DEVICE_CLAIM: u64 = 16;
    /// 18 — the compositor's pointer report view is absent or malformed.
    pub const RUNG_POINTER_REPORTS: u64 = 18;
    /// 19 — the compositor's pointer wake endpoint is absent.
    pub const RUNG_POINTER_KICK: u64 = 19;
}
#[cfg(target_os = "none")]
use rung::*;

/// How long the readiness probe waits for its No-op completion before
/// NACKing rung 12 — the whole proof's bound, not one park's.
///
/// An xHCI command completes in microseconds on any real part, so two
/// seconds is pure generosity toward slow firmware while still failing
/// *fast* on a machine whose delivery path is dead: the founder's own
/// readiness budget is 120 s, and spending it here would cost the boot
/// two silent minutes to learn what the register snapshot in the rung-12
/// NACK says immediately.
#[cfg(target_os = "none")]
const READINESS_BUDGET_NS: u64 = 2_000_000_000;

/// Non-fatal readiness note: this machine exposed no PCI or ACPI xHCI satisfying
/// the driver's BAR/DMA/interrupt contract. The high bits carry the optional
/// i8042 relay note, preserving both diagnoses in one beacon payload.
#[cfg(target_os = "none")]
const NOTE_NO_XHCI: u64 = 17;

/// The class-owned candidate tiers formerly implemented in the kernel.
#[cfg(any(test, target_os = "none"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum XhciCandidateRank {
    /// Registers do not decode or reported port geometry leaves BAR0.
    Reject,
    /// Lawful controller with no currently connected root-hub port.
    Fallback,
    /// Lawful controller with at least one connected root-hub port.
    Connected,
}

#[cfg(any(test, target_os = "none"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct XhciCandidateEvidence {
    rank: XhciCandidateRank,
    max_ports: usize,
}

#[cfg(any(test, target_os = "none"))]
fn xhci_candidate_evidence(ports: Option<(usize, usize)>) -> XhciCandidateEvidence {
    match ports {
        Some((connected, max_ports)) if connected != 0 => XhciCandidateEvidence {
            rank: XhciCandidateRank::Connected,
            max_ports,
        },
        Some((_, max_ports)) => XhciCandidateEvidence {
            rank: XhciCandidateRank::Fallback,
            max_ports,
        },
        None => XhciCandidateEvidence {
            rank: XhciCandidateRank::Reject,
            max_ports: 0,
        },
    }
}

/// Prefer the root hub with the larger declared reach when no controller has
/// produced a HID source. `PORTSC.CCS` before reset is only a hint: firmware
/// may leave every port looking idle until the class driver starts the xHC.
/// Equal-width roots preserve inventory order; connectivity is not a useful
/// tie-break after class discovery has already proved that it is not HID.
#[cfg(any(test, target_os = "none"))]
fn prefer_xhci_fallback(held: XhciCandidateEvidence, candidate: XhciCandidateEvidence) -> bool {
    candidate.max_ports > held.max_ports
}

#[cfg(any(test, target_os = "none"))]
struct XhciCandidate<D> {
    index: u32,
    identity: u32,
    device: D,
    evidence: XhciCandidateEvidence,
    vendor: u16,
    device_id: u16,
}

#[cfg(any(test, target_os = "none"))]
fn retain_xhci_candidate<D>(held: &mut Option<XhciCandidate<D>>, candidate: XhciCandidate<D>) {
    let replace = match held.as_ref() {
        None => true,
        Some(current) => prefer_xhci_fallback(current.evidence, candidate.evidence),
    };
    if replace {
        drop(held.replace(candidate));
    } else {
        drop(candidate);
    }
}

#[cfg(any(test, target_os = "none"))]
fn take_best_xhci_candidate<D, const N: usize>(
    candidates: &mut [Option<XhciCandidate<D>>; N],
) -> Option<XhciCandidate<D>> {
    let mut best: Option<usize> = None;
    for (index, candidate) in candidates.iter().enumerate() {
        let Some(candidate) = candidate.as_ref() else {
            continue;
        };
        if best.is_none_or(|held| {
            prefer_xhci_fallback(
                candidates[held].as_ref().unwrap().evidence,
                candidate.evidence,
            )
        }) {
            best = Some(index);
        }
    }
    candidates[best?].take()
}

/// Copy a claimed controller's complete bounded topology before admitting a
/// worker. A refused child read is not a shorter inventory; no partial table
/// escapes, and the caller can record the failure without losing other input.
#[cfg(any(test, target_os = "none"))]
fn read_i2c_bindings(
    count: Result<u32, InventoryReadError>,
    mut read: impl FnMut(u32) -> Result<I2cHidReport, InventoryReadError>,
) -> Result<
    (
        [Option<I2cHidReport>; service::i2c_relay::MAX_CHILDREN],
        usize,
    ),
    u32,
> {
    use service::i2c_relay::{MAX_CHILDREN, rung};

    let count = count.map_err(|_| rung::NO_BINDINGS)?;
    let count = usize::try_from(count).map_err(|_| rung::TOO_MANY_BINDINGS)?;
    if count == 0 {
        return Err(rung::NO_BINDINGS);
    }
    if count > MAX_CHILDREN {
        return Err(rung::TOO_MANY_BINDINGS);
    }
    let mut bindings = [None; MAX_CHILDREN];
    for (child, binding) in bindings[..count].iter_mut().enumerate() {
        *binding = Some(read(child as u32).map_err(|_| rung::NO_BINDINGS)?);
    }
    Ok((bindings, count))
}

/// What launch-time enumeration proved about one live controller.
#[cfg(any(test, target_os = "none"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum XhciSelection {
    Hid,
    NonHid,
    Inconclusive,
}

#[cfg(any(test, target_os = "none"))]
fn xhci_selection(sources: usize, census: kore_xhci::DiscoveryCensus) -> XhciSelection {
    if sources != 0 {
        XhciSelection::Hid
    } else if census.examined != 0 && census.no_hid == census.examined {
        XhciSelection::NonHid
    } else {
        XhciSelection::Inconclusive
    }
}

/// Result of the optional USB class bind. A contained bind refusal removes
/// USB from this founding but must not suppress independently claimed PS/2 or
/// I2C input. The consumed device and DMA roots remain quarantined; this route
/// carries only the diagnostic, never reusable authority.
#[cfg(any(test, target_os = "none"))]
#[derive(Debug, PartialEq, Eq)]
enum XhciBindRoute<T> {
    Bound(T),
    IndependentOnly { diagnostic: u64 },
}

#[cfg(any(test, target_os = "none"))]
fn route_xhci_bind<T>(result: Result<T, kore_driver::host::IntakeError>) -> XhciBindRoute<T> {
    match result {
        Ok(bound) => XhciBindRoute::Bound(bound),
        Err(error) => XhciBindRoute::IndependentOnly {
            diagnostic: service::xhci_bind::pack(service::xhci_bind::BIND, error.diagnostic_code()),
        },
    }
}

/// **DIAGNOSTIC KNOBS — the gigatron ~10 s founding-window freeze bisection
/// (PR #793). Not a feature; delete this module when the verdict is in.**
///
/// The freeze stops the boot splash on EVERY CPU at once, and the splash
/// pump is per-CPU redundant (#518) — so the cause must stop the whole
/// machine, which no ring-3 spin and no single stalled MMIO read can do.
/// The remaining suspects are firmware-level (SMM residency provoked by
/// something this row touches) or a machine-wide sleep on swallowed timer
/// arms — and this row is the only row that touches the two classic SMM
/// trap surfaces: ports 0x60/0x64 (firmware legacy-input emulation) and
/// the xHCI BIOS-handoff/HCRST pair (USB legacy SMI).
///
/// Each knob makes one device edge inert, exactly along postures that
/// already exist (`NOTE_NO_XHCI`, the relay's `NO_DEVICE` rung), so a
/// diagnostic boot degrades honestly instead of wedging. Bisect by splash
/// motion alone — no serial needed:
///
/// - all three `true` (flash A): the row founds inert. Splash keeps
///   turning ⇒ the freeze lives in one of the skipped edges; go to
///   flash B. Splash still freezes ⇒ the freeze is OUTSIDE this row's
///   device bring-up entirely, and the hunt moves to the founding
///   machinery around this row's slot.
/// - flash B: set `SKIP_I8042 = false` alone. Freeze returns ⇒ the i8042
///   ladder's port I/O is the trigger (SMM-trapped 0x60/0x64 — firmware
///   waiting on its EC inside SMM stops every core per command).
/// - flash C: set `SKIP_USB = false` alone. Freeze returns ⇒ the xHCI
///   claim path is the trigger (candidate BAR probes, the BIOS-handoff
///   SMI, or HCRST on a firmware-owned controller).
/// **All `false` on this branch** — behavior is byte-identical to the
/// pre-diagnostic head, because the QEMU boot gate hard-requires the
/// input row's remapped-MSI route witness
/// (`kore-svc-input-remapped-msi-irq: PASS`), which only a really-claimed
/// xHC can print: an inert row can never pass that gate, by design (CI
/// proved the inert boot itself healthy — 19/19, first shell ready — and
/// failed exactly that assertion). The flash-ready knobs-on state lives
/// on the `diag/793-flash-a` branch; flip these to `true` (flash A) or
/// selectively (flashes B/C) when building a metal diagnostic image.
#[cfg(target_os = "none")]
mod diag {
    /// Skip the whole xHCI claim: no candidate BAR probe, no BIOS
    /// handoff, no HCRST, no enumeration. The row takes the existing
    /// hardwareless-router posture (readiness note 17).
    pub const SKIP_USB: bool = false;
    /// Skip the i8042 claim: the relay finds no device and parks at its
    /// `NO_DEVICE` rung without one port access.
    pub const SKIP_I8042: bool = false;
    /// Skip the HID-over-I2C claim: the i2c relay likewise stays inert.
    pub const SKIP_I2C: bool = false;
}

#[cfg(target_os = "none")]
pub async fn run(rt: kore_rt::entry::Rt) -> ! {
    use kore_abi::session::Notice;
    use kore_abi::task::role;
    use kore_rt::entry::refuse_with;
    use kore_rt::service::{Bundle, SystemService};
    use kore_rt::svc::{COOKIE_REFUSED, COOKIE_SERVING, REQ_CAP, serve_rings};
    use kore_rt::transport::Transport;
    use kore_rt::{ChanTx, SessionBuilder};
    use service::{Endpoint, InputHost};

    let boot = rt.bootstrap();
    let Ok(endowment) = InputEndowment::resolve(boot) else {
        refuse_with(&rt, RUNG_ENDOWMENT);
    };
    let mut service = InputService::bind(endowment);
    let transport = kore_rt::entry::transport();
    // ── Claim the hardware (`res/DEVICE_SERVICE.md` §10) ────────────────
    //
    // The controllers are asked for, not founded. This runs before
    // `start_on_metal` because the PCI path that follows touches hardware,
    // and it runs on the executor because a claim is a *client call* to
    // another service — the broker holds the machine's inventory and
    // decides, in ring 3, which function this row gets. A true `NoMatch`
    // instead selects the router-only posture without touching an xHC.
    //
    // Which is exactly what the kernel used to do: match a class triple,
    // read `CAPLENGTH` and `PORTSC` through the candidate's BAR to skip a
    // D3-parked or portless controller, size a DMA pool from
    // `HCSPARAMS2`.  The reject tier survives — it is the one thing that
    // makes this row find the *right* xHC on a machine with two — but it
    // now runs where the xHCI knowledge already lives.
    let Some(pool) = service.take_pool() else {
        refuse_with(&rt, RUNG_ENDOWMENT);
    };
    let claimed = claim_devices(transport, boot, pool).await;
    let Some((device, legacy, i2c, legacy_detail, xhci_bind, xhci_identity)) = claimed else {
        refuse_with(&rt, RUNG_DEVICE_CLAIM);
    };
    service.adopt_device(device, legacy, i2c, xhci_bind, xhci_identity);
    let has_xhci = service.has_xhci();
    if service.start_on_metal(&rt).is_err() {
        refuse_with(&rt, RUNG_START_ON_METAL);
    }
    // The declared worker threads (#420: the i8042 relay), spawned by
    // the runtime — this loop is bespoke because it arms an IRQ park and
    // proves its controller before serving, not because it wants its own
    // way of making threads.  Ordering matters and is the runtime's:
    // after `start_on_metal` (the relay's handoff queue exists) and
    // before readiness (a client that sees COOKIE_SERVING is talking to
    // a service whose threads already exist).
    if !kore_rt::entry::spawn_workers::<InputService>(&rt, &service) {
        refuse_with(&rt, RUNG_WORKER_SPAWN);
    }
    let Some(window) = serve_window(&rt, 0) else {
        refuse_with(&rt, RUNG_RING_WINDOW);
    };
    // SAFETY: `map_view` installed its reported `[base, +len)` R/W-USER in
    // **this task's own** space over the pair's leased backing, page-aligned
    // (hence 64-byte aligned, the spsc layout unit) and live for the task's
    // whole life (nothing unmaps it; the service never returns).  This task
    // is the sole service-side attacher of that window — the client half
    // lives in another space and touches it only through the SPSC protocol —
    // so the `&mut` is unaliased in the sense the ring halves require.
    let window: &'static mut [u8] =
        unsafe { core::slice::from_raw_parts_mut(window.cast::<u8>().as_ptr(), window.len()) };
    let Ok(rings) = serve_rings::<InputService>(window, REQ_CAP, kore_rt::svc::REP_CAP) else {
        refuse_with(&rt, RUNG_RING_FORMAT);
    };
    let Some(kick) = boot.handle(role::SVC_SEND, 0) else {
        refuse_with(&rt, RUNG_KICK_ENDPOINT);
    };
    let Some(return_recv) = boot.handle(role::SVC_RETURN, 0) else {
        refuse_with(&rt, RUNG_RETURN_RECV);
    };
    let Some(return_land) = boot.handle(role::SVC_RETURN, 1) else {
        refuse_with(&rt, RUNG_RETURN_LAND);
    };
    let Some(beacon) = boot.handle(role::PARENT_SEND, 0) else {
        // No beacon endpoint ⇒ no NACK channel either; the spin is honest.
        refuse();
    };

    // The same transport and executor the claim above ran on — one of
    // each per task, so the claim's client call and the serve loop share
    // them rather than standing up a second pair.
    let mut host = InputHost::new(service, transport);
    let pointer_view = boot
        .handle(role::POINTER_REPORTS, 0)
        .and_then(kore_rt::cap::verbs::take_view)
        .and_then(kore_rt::cap::verbs::map_view);
    let Some(pointer_view) = pointer_view else {
        refuse_with(&rt, RUNG_POINTER_REPORTS)
    };
    // SAFETY: dedicated founder-owned ring, input's sole producer mapping.
    // The only sibling accessor is display's SPSC consumer, not a byte stream.
    let pointer_bytes: &'static mut [u8] = unsafe { &mut *pointer_view.as_ptr() };
    // The founder formatted this window as an event channel, so attach that
    // half rather than the bare ring: the books ride the seqlock'd state and
    // the wake becomes conditional (`res/EVENT.md`).
    let Ok(pointer) =
        kore_ring::event::source::<kore_abi::pointer::PointerState, InputReport>(pointer_bytes)
    else {
        refuse_with(&rt, RUNG_POINTER_REPORTS)
    };
    host.svc_mut().pointer.attach(pointer);
    let Some(pointer_kick) = boot.handle(role::POINTER_KICK, 0) else {
        refuse_with(&rt, RUNG_POINTER_KICK)
    };
    if !host.attach(0, rings, kick, Endpoint::Focus) {
        refuse_with(&rt, RUNG_HOST_ATTACH);
    }

    if has_xhci {
        // The steady park arms the held route beside the doorbell — the
        // route was established by the binding, so the kernel's combination
        // is lawful here (and is deliberately not requested without xHCI).
        kore_rt::executor::enable_irq_parks().await;
        // Deterministic readiness proof: stage one xHCI No-op and demand
        // its command completion off the event ring — by **parking on the
        // held route with a deadline**, polling once per wake.
        let probe = host.svc_mut().arm_interrupt_probe();
        let deadline = kore_rt::executor::clock_ns()
            .await
            .saturating_add(READINESS_BUDGET_NS);
        if !host.svc_mut().settle_interrupt_probe(probe, deadline) {
            // The NACK payload's high bits carry the controller's register
            // snapshot (USBSTS/CRR/IP — the founder prints them as `detail`).
            let detail = host.svc().interrupt_probe_detail();
            let _ = ChanTx::new(transport, beacon)
                .send(Notice {
                    cookie: COOKIE_REFUSED,
                    payload: RUNG_IRQ_PROBE_COMPLETE | (detail << 8),
                })
                .await;
            kore_rt::entry::refuse();
        }
    }
    // The return post stands BEFORE the serving beacon (the svc serve
    // discipline): a client that sees COOKIE_SERVING can deliver its
    // report queue immediately (the board law: no visible post ⇒
    // WouldBlock).  v1: one post, one founding subscriber.
    //
    // The outcome is REQUIRED, not best-effort: a router that beacons
    // without its return post reports healthy while every subscriber's
    // report-queue transfer refuses forever — the terminal chain dies
    // one row later with no line naming this cause.  `WouldBlock`
    // alone passes (a post already stands — same as the doorbell
    // re-post's rule); anything else NACKs the rung.
    let mut post = SessionBuilder::new();
    let stood = post.post_recv(return_recv, return_land).is_ok()
        && matches!(
            transport.commit(&post).await.map(|c| c.status()),
            Ok(kore_abi::session::Status::Ok | kore_abi::session::Status::WouldBlock)
        );
    if !stood {
        let _ = ChanTx::new(transport, beacon)
            .send(Notice {
                cookie: COOKIE_REFUSED,
                payload: RUNG_RETURN_POST_STAND,
            })
            .await;
        kore_rt::entry::refuse();
    }
    // **The relay's outcome, reported rather than refused.** A relay
    // that cannot bring its controller up is otherwise completely
    // silent: it publishes a rung, parks to keep publishing it, and
    // nothing anywhere reads that word — so on the serial-less metal
    // target "is the internal keyboard working?" costs a flash cycle.
    // It rides the READINESS beacon and not `COOKIE_REFUSED` because a
    // NACK *fails the row*, and a failed relay must not fail this
    // service: the USB keyboard is untouched by it, and a machine with
    // no i8042 at all is the ordinary case, not an error.  The wait is
    // bounded and its expiry is likewise not a failure — see
    // `relay::settle_note`.
    let relay_note = service::relay::settle_note().await;
    // A NO_DEVICE verdict alone says "no i8042" — true on most machines,
    // maddening on one whose scout just printed the keyboard's GSI.  The
    // claim's own outcome rides the note's detail half (the rung map's
    // `rung | detail << 8` shape, rendered by the founder as hex), naming
    // the leg that answered empty — or `CLAIMED`, the contradiction that
    // convicts the worker-endowment carve instead.
    let relay_note = if relay_note == u64::from(service::relay::rung::NO_DEVICE) {
        relay_note | (legacy_detail << 8)
    } else {
        relay_note
    };
    // With no usable PCI or ACPI-MMIO xHCI, retain the relay's complete
    // diagnostic word behind the distinct router-only readiness rung.
    let readiness_note = if has_xhci {
        relay_note
    } else {
        NOTE_NO_XHCI | (relay_note << 8)
    };
    let tx = ChanTx::new(transport, beacon);
    let _ = tx
        .send(Notice {
            cookie: COOKIE_SERVING,
            payload: readiness_note,
        })
        .await;
    loop {
        // Fold the device drain into every wake — kick or IRQ-late —
        // then serve everything pending, then go idle: the executor's
        // IRQ-or-doorbell park is the only sleep this loop needs.
        // A batch that DELIVERED gets one advisory data-plane kick to
        // the focused owner (`COOKIE_REPORT`) — the wake a
        // doorbell-parked consumer otherwise never gets; the routing
        // itself stays kick-free and lossless-drain.
        let before = host.svc().stats().delivered;
        let pointer_before = host.svc().pointer.publications;
        let _ = host.svc_mut().drain_interrupt();
        // The internal keyboard's half — routed through the same
        // `route_report`, so focus and the lossy-drop posture apply
        // to it without a line of new policy.  Drained on EVERY
        // wake, not only the relay's own notice: a wake for any
        // reason should leave no keystroke sitting in the queue.
        let _ = host.svc_mut().drain_relay();
        let _ = host.svc_mut().drain_i2c_relay();
        if host.svc().stats().delivered != before {
            let _ = host.kick_focused().await;
        }
        if host.svc().pointer.publications != pointer_before {
            // End the batch through the channel: `commit_and_kick` advances the
            // wake word and sends a notice **only if the compositor had parked**.
            // The previous unconditional send cost a syscall on every batch even
            // while the consumer was awake and about to drain anyway — precisely
            // the per-frame notice traffic the event model exists to remove, and
            // worst exactly at load, when input is busiest.
            let kick = ChanTx::new(transport, pointer_kick);
            if let Some(channel) = host.svc_mut().pointer.publish_books() {
                let _ = kore_rt::event::commit_and_kick(channel, &kick).await;
            }
        }
        while host.serve_next().await {}
        // Park until the next wake and RE-RUN the body — a
        // `pending().await` here is a one-way door (the root re-polls
        // straight into it forever, and no later subscribe, focus
        // steer, or report drain ever runs again).  The wake reason is
        // the MSI-landed witness: a route-resolved park counts, so the
        // htop row (and a metal photo) can tell real delivery from
        // kick-paced degradation.
        if host.svc().has_pending_relay_input() {
            kore_rt::executor::yield_now().await;
            continue;
        }
        if kore_rt::executor::next_wake().await == kore_rt::executor::WakeReason::Irq {
            host.svc_mut().note_irq_delivery();
        }
    }
}

/// **Claim this row's hardware from the device broker**
/// (`res/DEVICE_SERVICE.md` §10).
///
/// The xHCI controller first. BAR-only probes reject dead geometry; every
/// lawful root is brought up one at a time because a pre-reset `PORTSC.CCS`
/// snapshot cannot prove that an idle-looking controller has no HID. Actual
/// class enumeration chooses the winner. If none wins, the widest root hub is
/// retained for hotplug. Stop claiming immediately once one wins: later xHCs
/// may belong to another class service (the QEMU production profile's
/// USB-network controller is the concrete witness).
/// Each rejected class probe is quiesced and its complete DMA/BAR graph retired
/// before the one parent pool is reused. Rejected roots are dropped but their
/// broker rows stay quarantined: no cross-task protocol can prove their
/// descendants were reaped, so reopening them would be an unsafe promise.
/// A contained xHCI bind failure likewise stops USB selection without refusing
/// the row: the consumed pool is never reused, while independent PS/2 and I2C
/// claims still run and the exact bind diagnostic remains visible in status.
///
/// That is the whole point.  The old ring-0 tier read `CAPLENGTH` to skip
/// a D3-parked function and swept `PORTSC.CCS` to skip a live-but-portless
/// one — Lunar Lake ships both, and its Type-C xHC enumerates *before* the
/// PCH controller that actually hosts the keyboard.  Here the same two
/// questions are asked by the driver that already has the registers open.
///
/// Then the i8042, which is **optional**: a laptop may route its built-in
/// keyboard over USB or I2C and describe no 8042 at all, and that machine
/// must still bring up its input service with one keyboard rather than
/// refusing to found.
#[cfg(target_os = "none")]
async fn claim_devices(
    transport: &kore_rt::transport::TrapTransport,
    boot: &kore_abi::task::Bootstrap<'_>,
    pool: kore_rt::cap::SupplyToken,
) -> Option<(
    Option<kore_driver::host::BoundDevice<service::XhciBinding>>,
    Option<kore_rt::cap::PlatformToken>,
    [Option<service::I2cClaim>; service::i2c_relay::MAX_CONTROLLERS],
    u64,
    u64,
    u32,
)> {
    use kore_svc_device::client::{Claimed, ClaimedDevice, DeviceClient, FaceLayout, MultiBroker};
    use kore_svc_device::service::DeviceRequest;

    // The broker faces the founder stamped — one per live root complex
    // (#572).  `None` means NO face opened: this row was founded off the
    // chain or ahead of every broker — it can claim nothing, and says so
    // rather than sending into a dead channel.
    let mut broker = MultiBroker::open(
        transport,
        boot,
        FaceLayout::for_bootstrap(
            boot,
            <InputService as kore_rt::svc::SvcProtocol>::SERVE_PAIRS,
            <InputService as kore_rt::svc::SvcProtocol>::BOOT_REPLY_GRANTS,
        ),
    )?;
    service::admission::collect(&mut broker).await;

    // The xHCI selector this row declares — cited from its own
    // `SystemService::DEVICES`, so what it may own stays a property of its
    // measured image rather than a number written here.
    let selector = <InputService as kore_rt::service::SystemService>::DEVICES
        .get(service::device_index::PCI_XHCI)?
        .selector;

    // Claim and test lazily. The broker cannot lawfully reopen a row from an
    // integer `Release`, so eagerly claiming the whole class would quarantine
    // controllers that input never needed and starve later class services.
    // Every idle-looking PCI candidate fits because each of the bounded root
    // faces can grant at most MAX_CLAIMS. They are tried widest-first only
    // after the scan; this avoids consuming the sole DMA pool on the first
    // Type-C root merely because its inventory row happened to come first.
    const PCI_XHCI_CANDIDATES: usize =
        kore_abi::device::MAX_ROOT_COMPLEXES * kore_svc_device::MAX_CLAIMS;
    let mut hard_fail = false;
    let mut xhci_bind = service::xhci_bind::pack(
        service::xhci_bind::PCI_CLAIM,
        service::xhci_bind::REFUSED_NO_MATCH,
    );
    let mut pool = Some(pool);
    let mut fallbacks: [Option<XhciCandidate<kore_rt::cap::DeviceToken>>; PCI_XHCI_CANDIDATES] =
        core::array::from_fn(|_| None);
    let mut fallback_count = 0usize;
    let mut retained: Option<XhciCandidate<kore_rt::cap::DeviceToken>> = None;
    let mut xhci_device = None;
    let mut selected_identity = service::xhci_sources::UNKNOWN_IDENTITY;
    let mut usb_bind_failed = false;
    'faces: for k in 0..kore_abi::device::MAX_ROOT_COMPLEXES {
        if diag::SKIP_USB {
            // DIAGNOSTIC (#793): claim no xHC at all — no candidate wins and
            // the row takes the hardwareless-router posture.
            break 'faces;
        }
        let Some(face) = broker.face(k) else {
            continue;
        };
        let mut claims = 0usize;
        loop {
            match face.claim(DeviceRequest::Claim(selector)).await {
                Ok(Claimed {
                    index,
                    device,
                    diag: _,
                    revision: _,
                    vendor,
                    device_id,
                }) => {
                    claims += 1;
                    // Correlation already requires PCI for this selector;
                    // retain the sealed-kind check as the last fail-closed
                    // boundary.
                    let token = match device {
                        ClaimedDevice::Pci(token) => token,
                        ClaimedDevice::Platform(_) => {
                            xhci_bind = service::xhci_bind::pack(
                                service::xhci_bind::PCI_CLAIM,
                                service::xhci_bind::WRONG_SPECIES,
                            );
                            hard_fail = true;
                            break 'faces;
                        }
                    };
                    let probe = match kore_driver::host::DeviceProbe::open(token, 0) {
                        Ok(probe) => probe,
                        Err(error) => {
                            xhci_bind = service::xhci_bind::pack(
                                service::xhci_bind::PCI_PROBE,
                                error.diagnostic_code(),
                            );
                            // The root was consumed: map failure can occur
                            // after a BAR derive whose cleanup was retryable,
                            // so it is not lawful reusable authority.
                            if claims == kore_svc_device::MAX_CLAIMS {
                                continue 'faces;
                            }
                            continue;
                        }
                    };
                    let ports = kore_xhci::regs::connected_ports(probe.regs(), probe.regs().len());
                    let evidence = xhci_candidate_evidence(ports);
                    let token = match probe.finish() {
                        Ok(token) => token,
                        Err(error) => {
                            xhci_bind = service::xhci_bind::pack(
                                service::xhci_bind::PCI_PROBE,
                                error.diagnostic_code(),
                            );
                            hard_fail = true;
                            break 'faces;
                        }
                    };
                    let candidate = XhciCandidate {
                        index,
                        identity: service::xhci_sources::controller_identity(k, index, false),
                        device: token,
                        evidence,
                        vendor,
                        device_id,
                    };
                    match evidence.rank {
                        XhciCandidateRank::Reject => drop(candidate),
                        XhciCandidateRank::Fallback => {
                            let Some(slot) = fallbacks.get_mut(fallback_count) else {
                                // The array is the product of both producer
                                // bounds, so reaching this arm means those
                                // protocol constants no longer agree.
                                hard_fail = true;
                                drop(candidate);
                                break 'faces;
                            };
                            *slot = Some(candidate);
                            fallback_count += 1;
                        }
                        XhciCandidateRank::Connected => {
                            let XhciCandidate {
                                index,
                                identity,
                                device: token,
                                evidence,
                                vendor,
                                device_id,
                            } = candidate;
                            let dma_pool = pool.take()?;
                            let access = kore_xhci::pointer_access_for_pci(vendor, device_id);
                            let bound = match route_xhci_bind(kore_driver::host::BoundDevice::<
                                service::XhciBinding,
                            >::claim_with_config(
                                token, dma_pool, access
                            )) {
                                XhciBindRoute::Bound(bound) => bound,
                                XhciBindRoute::IndependentOnly { diagnostic } => {
                                    xhci_bind = diagnostic;
                                    usb_bind_failed = true;
                                    drop(retained.take());
                                    break 'faces;
                                }
                            };
                            let (sources, census) = bound.parts().selection_evidence();
                            if xhci_selection(sources, census) == XhciSelection::Hid {
                                xhci_bind = service::xhci_bind::running(bound.parts().handoff());
                                selected_identity = identity;
                                xhci_device = Some(bound);
                                drop(retained.take());
                                break 'faces;
                            }

                            let kore_driver::host::BindingRecovery {
                                quiesce,
                                resources,
                                device: recovered,
                                pool: recovered_pool,
                            } = bound.teardown_recover();
                            if quiesce.is_err() || !resources.complete() {
                                return None;
                            }
                            pool = Some(recovered_pool?);
                            let recovered = XhciCandidate {
                                index,
                                identity,
                                device: recovered?,
                                evidence,
                                vendor,
                                device_id,
                            };
                            retain_xhci_candidate(&mut retained, recovered);
                        }
                    }
                    if claims == kore_svc_device::MAX_CLAIMS {
                        continue 'faces;
                    }
                }
                Err(
                    refusal @ (kore_svc_device::DeviceRefusal::Taken
                    | kore_svc_device::DeviceRefusal::NoMatch),
                ) => {
                    let detail = match refusal {
                        kore_svc_device::DeviceRefusal::Taken => service::xhci_bind::REFUSED_TAKEN,
                        _ => service::xhci_bind::REFUSED_NO_MATCH,
                    };
                    xhci_bind = service::xhci_bind::pack(service::xhci_bind::PCI_CLAIM, detail);
                    // This root is exhausted; the next may still carry a
                    // connected controller.
                    continue 'faces;
                }
                Err(refusal) => {
                    let detail = match refusal {
                        kore_svc_device::DeviceRefusal::Denied => {
                            service::xhci_bind::REFUSED_DENIED
                        }
                        _ => service::xhci_bind::REFUSED_OTHER,
                    };
                    xhci_bind = service::xhci_bind::pack(service::xhci_bind::PCI_CLAIM, detail);
                    hard_fail = true;
                    break 'faces;
                }
            }
        }
    }
    if hard_fail {
        return None;
    }

    // No connected candidate produced a HID source. Try every idle-looking
    // root now, widest first. This is the path the Lunar Lake PCH takes when
    // firmware has not exposed CCS before the class driver resets the xHC.
    while xhci_device.is_none() && !usb_bind_failed {
        let Some(candidate) = take_best_xhci_candidate(&mut fallbacks) else {
            break;
        };
        let XhciCandidate {
            index,
            identity,
            device,
            evidence,
            vendor,
            device_id,
        } = candidate;
        let dma_pool = pool.take()?;
        let access = kore_xhci::pointer_access_for_pci(vendor, device_id);
        let bound = match route_xhci_bind(
            kore_driver::host::BoundDevice::<service::XhciBinding>::claim_with_config(
                device, dma_pool, access,
            ),
        ) {
            XhciBindRoute::Bound(bound) => bound,
            XhciBindRoute::IndependentOnly { diagnostic } => {
                // The intake contract consumes both roots on a failed bind.
                // Widest-first makes the best candidate the first attempt;
                // continuing would fabricate a DMA parent we no longer own.
                xhci_bind = diagnostic;
                usb_bind_failed = true;
                drop(retained.take());
                break;
            }
        };
        let (sources, census) = bound.parts().selection_evidence();
        if xhci_selection(sources, census) == XhciSelection::Hid {
            xhci_bind = service::xhci_bind::running(bound.parts().handoff());
            selected_identity = identity;
            xhci_device = Some(bound);
            drop(retained.take());
            break;
        }

        let kore_driver::host::BindingRecovery {
            quiesce,
            resources,
            device: recovered,
            pool: recovered_pool,
        } = bound.teardown_recover();
        if quiesce.is_err() || !resources.complete() {
            return None;
        }
        pool = Some(recovered_pool?);
        retain_xhci_candidate(
            &mut retained,
            XhciCandidate {
                index,
                identity,
                device: recovered?,
                evidence,
                vendor,
                device_id,
            },
        );
    }

    // ACPI/DT platform xHCI stays a peer candidate class. It deliberately
    // keeps the generic split-dword pointer policy: the AMD native-qword
    // exception is selected only from an exact PCI identity. Platform roots
    // are not part of the recoverable PCI probe protocol, so choose one by
    // lawful register geometry exactly as main did, and bind it only after no
    // PCI controller proved a HID source.
    if !diag::SKIP_USB && xhci_device.is_none() && !usb_bind_failed {
        let mut platform_selected = None;
        let mut platform_fallback = None;
        if let Some(selector) = <InputService as kore_rt::service::SystemService>::DEVICES
            .get(service::device_index::PLATFORM_XHCI)
            .map(|d| d.selector)
            && let Some(face) = broker.face(0)
        {
            let mut claims = 0usize;
            loop {
                match face.claim(DeviceRequest::Claim(selector)).await {
                    Ok(Claimed {
                        index,
                        device: claimed,
                        vendor: _,
                        device_id: _,
                        ..
                    }) => {
                        claims += 1;
                        let token = match claimed {
                            ClaimedDevice::Platform(token) => token,
                            ClaimedDevice::Pci(_) => {
                                xhci_bind = service::xhci_bind::pack(
                                    service::xhci_bind::PLATFORM_CLAIM,
                                    service::xhci_bind::WRONG_SPECIES,
                                );
                                hard_fail = true;
                                break;
                            }
                        };
                        let probe = match kore_driver::host::PlatformDeviceProbe::open(token) {
                            Ok(probe) => probe,
                            Err(error) => {
                                xhci_bind = service::xhci_bind::pack(
                                    service::xhci_bind::PLATFORM_PROBE,
                                    error.diagnostic_code(),
                                );
                                if claims == kore_svc_device::MAX_CLAIMS {
                                    break;
                                }
                                continue;
                            }
                        };
                        let evidence = xhci_candidate_evidence(kore_xhci::regs::connected_ports(
                            probe.regs(),
                            probe.regs().len(),
                        ));
                        let token = match probe.finish() {
                            Ok(token) => token,
                            Err(error) => {
                                xhci_bind = service::xhci_bind::pack(
                                    service::xhci_bind::PLATFORM_PROBE,
                                    error.diagnostic_code(),
                                );
                                hard_fail = true;
                                break;
                            }
                        };
                        match evidence.rank {
                            XhciCandidateRank::Connected => {
                                platform_selected = Some((
                                    token,
                                    service::xhci_sources::controller_identity(0, index, true),
                                ));
                                break;
                            }
                            XhciCandidateRank::Fallback if platform_fallback.is_none() => {
                                platform_fallback = Some((
                                    token,
                                    service::xhci_sources::controller_identity(0, index, true),
                                ));
                            }
                            XhciCandidateRank::Fallback | XhciCandidateRank::Reject => drop(token),
                        }
                        if claims == kore_svc_device::MAX_CLAIMS {
                            break;
                        }
                    }
                    Err(kore_svc_device::DeviceRefusal::Taken) => {
                        xhci_bind = service::xhci_bind::pack(
                            service::xhci_bind::PLATFORM_CLAIM,
                            service::xhci_bind::REFUSED_TAKEN,
                        );
                        continue;
                    }
                    Err(kore_svc_device::DeviceRefusal::NoMatch) => {
                        xhci_bind = service::xhci_bind::pack(
                            service::xhci_bind::PLATFORM_CLAIM,
                            service::xhci_bind::REFUSED_NO_MATCH,
                        );
                        break;
                    }
                    Err(refusal) => {
                        let detail = match refusal {
                            kore_svc_device::DeviceRefusal::Denied => {
                                service::xhci_bind::REFUSED_DENIED
                            }
                            _ => service::xhci_bind::REFUSED_OTHER,
                        };
                        xhci_bind =
                            service::xhci_bind::pack(service::xhci_bind::PLATFORM_CLAIM, detail);
                        hard_fail = true;
                        break;
                    }
                }
            }
        }
        if hard_fail {
            return None;
        }
        if let Some((token, identity)) = platform_selected.or(platform_fallback) {
            let dma_pool = pool.take()?;
            xhci_device = match route_xhci_bind(kore_driver::host::BoundDevice::<
                service::XhciBinding,
            >::claim_platform(token, dma_pool))
            {
                XhciBindRoute::Bound(bound) => {
                    selected_identity = identity;
                    let (sources, census) = bound.parts().selection_evidence();
                    xhci_bind = if xhci_selection(sources, census) == XhciSelection::Hid {
                        service::xhci_bind::running(bound.parts().handoff())
                    } else {
                        service::xhci_bind::pack(
                            service::xhci_bind::BIND,
                            service::xhci_bind::BOUND_NO_HID,
                        )
                    };
                    Some(bound)
                }
                XhciBindRoute::IndependentOnly { diagnostic } => {
                    xhci_bind = diagnostic;
                    usb_bind_failed = true;
                    None
                }
            };
            drop(retained.take());
        }
    }

    if xhci_device.is_none()
        && !usb_bind_failed
        && let Some(candidate) = retained.take()
    {
        let dma_pool = pool.take()?;
        let access = kore_xhci::pointer_access_for_pci(candidate.vendor, candidate.device_id);
        xhci_device = match route_xhci_bind(
            kore_driver::host::BoundDevice::<service::XhciBinding>::claim_with_config(
                candidate.device,
                dma_pool,
                access,
            ),
        ) {
            XhciBindRoute::Bound(bound) => {
                selected_identity = candidate.identity;
                let (sources, census) = bound.parts().selection_evidence();
                xhci_bind = if xhci_selection(sources, census) == XhciSelection::Hid {
                    service::xhci_bind::running(bound.parts().handoff())
                } else {
                    service::xhci_bind::pack(
                        service::xhci_bind::BIND,
                        service::xhci_bind::BOUND_NO_HID,
                    )
                };
                Some(bound)
            }
            XhciBindRoute::IndependentOnly { diagnostic } => {
                xhci_bind = diagnostic;
                usb_bind_failed = true;
                None
            }
        };
    }
    debug_assert!(!usb_bind_failed || pool.is_none());
    // No later device class can use an xHCI candidate. Release untried roots
    // before i8042/I2C admission so their dynamic landing slots and broker
    // claims do not remain live for the rest of this async startup future.
    drop(fallbacks);
    drop(retained);
    drop(pool);

    // The i8042, if firmware described one — through face 0 alone: the
    // platform bus rides broker ordinal 0 (#572).  A refusal here is not
    // a failure of this row: `None` means the machine has no legacy
    // keyboard controller, and the relay reports that in its status row.
    use service::relay::no_device_detail as detail;
    let (legacy, legacy_detail) = if diag::SKIP_I8042 {
        // DIAGNOSTIC (#793): no legacy claim ⇒ the relay's endowment
        // stays vacant and it refuses `NO_DEVICE` before one port access.
        (None, detail::NO_SELECTOR)
    } else {
        match <InputService as kore_rt::service::SystemService>::DEVICES
            .get(service::device_index::I8042)
            .map(|d| d.selector)
        {
            None => (None, detail::NO_SELECTOR),
            Some(legacy_selector) => match broker.face(0) {
                None => (None, detail::NO_FACE),
                Some(face) => match face.claim(DeviceRequest::Claim(legacy_selector)).await {
                    Ok(claimed) => match claimed.device {
                        ClaimedDevice::Platform(dev) => (Some(dev), detail::CLAIMED),
                        // Correlation rejects this on the wire; retain the
                        // sealed-kind check as the final fail-closed boundary.
                        ClaimedDevice::Pci(_) => (None, detail::WRONG_GRANT),
                    },
                    Err(refusal) => {
                        use kore_svc_device::service::DeviceRefusal;
                        let code = match refusal {
                            DeviceRefusal::NoMatch => detail::REFUSED_NO_MATCH,
                            DeviceRefusal::Taken => detail::REFUSED_TAKEN,
                            DeviceRefusal::Denied => detail::REFUSED_DENIED,
                            _ => detail::REFUSED_OTHER,
                        };
                        (None, code)
                    }
                },
            },
        }
    };
    // Platform I2C-HID: the broker grants the exact MMIO
    // controller row promoted from firmware topology, and the child query is
    // valid only under that capability.  No BDF, BAR number, or MSI fallback
    // participates in this species. PCI controllers are claimed separately.
    let i2c_selector = if diag::SKIP_I2C {
        // DIAGNOSTIC (#793): no claims ⇒ every I2C worker slot stays vacant.
        None
    } else {
        <InputService as kore_rt::service::SystemService>::DEVICES
            .get(service::device_index::I2C_HID)
            .map(|d| d.selector)
    };
    let mut i2c = core::array::from_fn(|_| None);
    // The platform catalog belongs to face 0, regardless of the number of
    // PCI roots. Its claim ledger advances past previously granted rows.
    if let (Some(selector), Some(face)) = (i2c_selector, broker.face(0)) {
        use service::i2c_relay::{note_claim, rung};
        for (index, target) in i2c.iter_mut().enumerate() {
            match face.claim(DeviceRequest::Claim(selector)).await {
                Ok(claimed) => match accept_i2c_claim(claimed, false) {
                    Ok(claim) => {
                        note_claim(index, rung::UNSTARTED);
                        *target = Some(claim);
                    }
                    Err(code) => note_claim(index, code),
                },
                Err(kore_svc_device::DeviceRefusal::NoMatch) => break,
                Err(kore_svc_device::DeviceRefusal::Taken) => {
                    note_claim(index, rung::CLAIM_TAKEN);
                    break;
                }
                Err(_) => {
                    note_claim(index, rung::CLAIM_FAILED);
                    break;
                }
            }
        }
    }
    if !diag::SKIP_I2C {
        use service::i2c_relay::{MAX_CONTROLLERS, note_claim, rung};
        let selector = <InputService as kore_rt::service::SystemService>::DEVICES
            .get(service::device_index::PCI_I2C_HID)
            .map(|request| request.selector);
        if let Some(selector) = selector {
            // A PCI controller belongs to its native root, not necessarily
            // broker face zero. Each ledger advances past every actual claim.
            for root in 0..kore_abi::device::MAX_ROOT_COMPLEXES {
                let Some(face) = broker.face(root) else {
                    continue;
                };
                for _ in 0..MAX_CONTROLLERS {
                    let Some(index) = i2c.iter().position(Option::is_none) else {
                        break;
                    };
                    match face.claim(DeviceRequest::Claim(selector)).await {
                        Ok(claimed) => match accept_i2c_claim(claimed, true) {
                            Ok(claim) => {
                                note_claim(index, rung::UNSTARTED);
                                i2c[index] = Some(claim);
                            }
                            Err(code) => note_claim(index, code),
                        },
                        Err(kore_svc_device::DeviceRefusal::NoMatch) => break,
                        Err(kore_svc_device::DeviceRefusal::Taken) => {
                            note_claim(index, rung::CLAIM_TAKEN);
                            break;
                        }
                        Err(_) => {
                            note_claim(index, rung::CLAIM_FAILED);
                            break;
                        }
                    }
                }
            }
        }
    }
    Some((
        xhci_device,
        legacy,
        i2c,
        legacy_detail,
        xhci_bind,
        selected_identity,
    ))
}

/// The grant is the only controller identity accepted by the child query.
/// PCI diagnostic geometry never doubles as a platform controller model.
#[cfg(target_os = "none")]
fn accept_i2c_claim(
    claimed: kore_svc_device::client::Claimed,
    pci: bool,
) -> Result<service::I2cClaim, u32> {
    use kore_svc_device::client::ClaimedDevice;
    use service::{
        I2cDevice,
        i2c_relay::{ControllerKind, rung},
    };

    let (device, kind) = match claimed.device {
        ClaimedDevice::Pci(device) if pci => {
            (I2cDevice::Pci(device), ControllerKind::PciDesignWare)
        }
        ClaimedDevice::Platform(device) if !pci => {
            let kind = ControllerKind::from_platform_model(claimed.diag).ok_or(rung::BAD_DEVICE)?;
            (I2cDevice::Platform(device), kind)
        }
        _ => return Err(rung::BAD_DEVICE),
    };
    fn children<C: kore_rt::cap::I2cHidSubject>(
        device: &C,
    ) -> Result<
        (
            [Option<kore_abi::session::I2cHidReport>; service::i2c_relay::MAX_CHILDREN],
            usize,
        ),
        u32,
    > {
        use kore_rt::cap::verbs;
        read_i2c_bindings(verbs::i2c_hid_len(device), |child| {
            verbs::i2c_hid_record(device, child)
        })
    }
    let (bindings, count) = match &device {
        I2cDevice::Pci(device) => children(device)?,
        I2cDevice::Platform(device) => children(device)?,
    };
    Ok(service::I2cClaim {
        regs: None,
        device,
        kind,
        bindings,
        count,
    })
}

/// The ring-window half of the relay entry's wake edge (a deliverable
/// `LeasedRegion` view earned into this task's own space).
#[cfg(target_os = "none")]
fn serve_window(rt: &kore_rt::entry::Rt, index: u16) -> Option<core::ptr::NonNull<[u8]>> {
    use kore_abi::task::role;
    use kore_rt::cap::verbs;

    let view = rt.bootstrap().handle(role::SVC_RING, index)?;
    let leased = verbs::take_view(view)?;
    verbs::map_view(leased)
}

#[cfg(target_os = "none")]
fn refuse() -> ! {
    // The shared parked refusal (kore-rt): terminal like the bare spin this
    // once was, but the monitor gives the core back instead of burning it —
    // a wedged row must not also be a hot one.
    kore_rt::entry::refuse()
}

pub use client::{InputClient, RingInput};
pub use service::{
    Endpoint, EndpointId, INPUT_QUEUE_CAPACITY, InputEndowment, InputHost, InputHostStats,
    InputRefusal, InputRequest, InputResponse, InputService, InputStats, InputVerb,
    MAX_INPUT_ENDPOINTS, MAX_INPUT_SUBSCRIBERS, RelayBooks, SubscriptionId, input_queue_bytes,
};

// The report record + queue live in kore-abi / kore-ring (the frozen wire
// contract crate + the spsc family) — re-exported so a consumer names one
// crate.
pub use kore_abi::svc::InputReport;

#[cfg(test)]
mod candidate_tests {
    use super::{
        XhciBindRoute, XhciCandidate, XhciCandidateEvidence, XhciCandidateRank, XhciSelection,
        prefer_xhci_fallback, retain_xhci_candidate, route_xhci_bind, take_best_xhci_candidate,
        xhci_candidate_evidence, xhci_selection,
    };

    fn candidate(identity: u32, connected: usize, max_ports: usize) -> XhciCandidate<u8> {
        XhciCandidate {
            index: identity,
            identity,
            device: identity as u8,
            evidence: xhci_candidate_evidence(Some((connected, max_ports))),
            vendor: 0,
            device_id: 0,
        }
    }

    fn assert_bind_failure_routes_to_independent_input(error: kore_driver::host::IntakeError) {
        let expected = crate::service::xhci_bind::pack(
            crate::service::xhci_bind::BIND,
            error.diagnostic_code(),
        );
        assert_eq!(
            route_xhci_bind::<()>(Err(error)),
            XhciBindRoute::IndependentOnly {
                diagnostic: expected,
            },
        );
    }

    #[test]
    fn candidate_tiers_reject_bad_geometry_and_prefer_connections() {
        assert_eq!(
            xhci_candidate_evidence(None),
            XhciCandidateEvidence {
                rank: XhciCandidateRank::Reject,
                max_ports: 0,
            },
            "an out-of-BAR port walk is not a live fallback",
        );
        assert_eq!(
            xhci_candidate_evidence(Some((0, 8))),
            XhciCandidateEvidence {
                rank: XhciCandidateRank::Fallback,
                max_ports: 8,
            },
        );
        assert_eq!(
            xhci_candidate_evidence(Some((1, 8))),
            XhciCandidateEvidence {
                rank: XhciCandidateRank::Connected,
                max_ports: 8,
            },
        );
    }

    #[test]
    fn source_free_fallback_keeps_the_root_with_the_largest_hotplug_reach() {
        let small = xhci_candidate_evidence(Some((0, 4)));
        let large = xhci_candidate_evidence(Some((0, 16)));
        assert!(prefer_xhci_fallback(small, large));
        assert!(!prefer_xhci_fallback(large, small));
    }

    #[test]
    fn equal_width_fallback_is_stable_after_class_discovery() {
        let idle = xhci_candidate_evidence(Some((0, 16)));
        let connected = xhci_candidate_evidence(Some((1, 16)));
        assert!(!prefer_xhci_fallback(idle, idle));
        assert!(!prefer_xhci_fallback(idle, connected));
        assert!(!prefer_xhci_fallback(connected, idle));
    }

    #[test]
    fn idle_index_twelve_is_probed_before_the_earlier_type_c_root() {
        let mut candidates = [Some(candidate(7, 0, 4)), Some(candidate(12, 0, 16))];
        let pch = take_best_xhci_candidate(&mut candidates).expect("widest candidate");
        assert_eq!(pch.identity, 12);
        assert_eq!(pch.device, 12);
        let type_c = take_best_xhci_candidate(&mut candidates).expect("remaining candidate");
        assert_eq!(type_c.identity, 7);
        assert!(take_best_xhci_candidate(&mut candidates).is_none());
    }

    #[test]
    fn all_non_hid_probes_retain_the_widest_recovered_root_for_hotplug() {
        let mut pending = [Some(candidate(7, 0, 4)), Some(candidate(12, 0, 16))];
        let mut retained = None;
        while let Some(probed) = take_best_xhci_candidate(&mut pending) {
            // The integer is the host-test stand-in for the affine token that
            // teardown_recover returns only after complete closure.
            retain_xhci_candidate(&mut retained, probed);
        }
        let retained = retained.expect("one hotplug root");
        assert_eq!(retained.identity, 12);
        assert_eq!(retained.device, 12);
    }

    #[test]
    fn launch_census_distinguishes_hid_non_hid_and_uncertain() {
        use kore_xhci::DiscoveryCensus;

        assert_eq!(
            xhci_selection(1, DiscoveryCensus::default()),
            XhciSelection::Hid
        );
        assert_eq!(
            xhci_selection(
                0,
                DiscoveryCensus {
                    examined: 2,
                    no_hid: 2,
                    ..DiscoveryCensus::default()
                }
            ),
            XhciSelection::NonHid
        );
        assert_eq!(
            xhci_selection(
                0,
                DiscoveryCensus {
                    examined: 2,
                    no_hid: 1,
                    err_timeout: 1,
                    ..DiscoveryCensus::default()
                }
            ),
            XhciSelection::Inconclusive
        );
        assert_eq!(
            xhci_selection(0, DiscoveryCensus::default()),
            XhciSelection::Inconclusive
        );
    }

    #[test]
    fn connected_pci_bind_failure_preserves_diagnostic_and_independent_input() {
        assert_bind_failure_routes_to_independent_input(
            kore_driver::host::IntakeError::ClassBindRefused {
                at: kore_driver::host::ClassStage::Start,
                code: 0x31,
            },
        );
    }

    #[test]
    fn platform_bind_failure_preserves_diagnostic_and_independent_input() {
        assert_bind_failure_routes_to_independent_input(
            kore_driver::host::IntakeError::InterruptUnavailable,
        );
    }

    #[test]
    fn fallback_bind_failure_preserves_diagnostic_and_independent_input() {
        assert_bind_failure_routes_to_independent_input(kore_driver::host::IntakeError::DmaMap {
            code: 0x2d,
        });
    }

    #[test]
    fn successful_bind_keeps_the_bound_controller() {
        assert_eq!(route_xhci_bind(Ok(7u8)), XhciBindRoute::Bound(7));
    }
}

#[cfg(test)]
mod i2c_admission_tests {
    use super::read_i2c_bindings;
    use crate::service::i2c_relay::{MAX_CHILDREN, rung};
    use kore_abi::session::I2cHidReport;
    use kore_rt::cap::verbs::InventoryReadError;

    fn binding(child: u32) -> I2cHidReport {
        I2cHidReport {
            slave_addr: 0x20 + child as u8,
            descriptor_register: 1,
            speed_hz: 400_000,
            standard_timing: None,
            fast_timing: None,
            input_clock_hz: Some(19_200_000),
            ready_interrupt: None,
            ready_route_state: kore_abi::session::I2cReadyRouteState::NotDescribed,
            no_irq: false,
            no_power_after_reset: false,
        }
    }

    #[test]
    fn invalid_child_counts_keep_precise_diagnostics_without_reading_records() {
        for (count, expected) in [
            (Ok(0), rung::NO_BINDINGS),
            (Ok(MAX_CHILDREN as u32 + 1), rung::TOO_MANY_BINDINGS),
            (Ok(u32::MAX), rung::TOO_MANY_BINDINGS),
            (Err(InventoryReadError::Unavailable), rung::NO_BINDINGS),
            (Err(InventoryReadError::MalformedReport), rung::NO_BINDINGS),
        ] {
            let mut reads = 0;
            let result = read_i2c_bindings(count, |child| {
                reads += 1;
                Ok(binding(child))
            });
            assert_eq!(result, Err(expected));
            assert_eq!(reads, 0, "invalid admission cannot begin a child walk");
        }
    }

    #[test]
    fn failed_child_read_never_admits_a_partial_controller() {
        let mut reads = 0;
        let result = read_i2c_bindings(Ok(MAX_CHILDREN as u32), |child| {
            reads += 1;
            if child == 1 {
                Err(InventoryReadError::MalformedReport)
            } else {
                Ok(binding(child))
            }
        });
        assert_eq!(result, Err(rung::NO_BINDINGS));
        assert_eq!(
            reads, 2,
            "stop on the failed read, not on a truncated table"
        );

        // A failed controller consumes no shared admission state. A later,
        // independent controller still exposes every authoritative child.
        for count in [1, MAX_CHILDREN] {
            let mut reads = 0;
            let (bindings, admitted) = read_i2c_bindings(Ok(count as u32), |child| {
                assert_eq!(child, reads);
                reads += 1;
                Ok(binding(child))
            })
            .unwrap();
            assert_eq!(admitted, count);
            assert_eq!(reads, count as u32);
            for (child, record) in bindings.iter().enumerate() {
                assert_eq!(*record, (child < count).then(|| binding(child as u32)));
            }
        }
    }
}
