//! **The xHCI input report producer** — the device edge of the input
//! service (`res/USERLAND.md` §5.8 / §3.1a; `res/DEVICE_SERVICE.md` §4).
//!
//! The `kore-svc-input` service task links this core and runs its drain as
//! the driver's **IRQ relay** (§3.1a: `WAIT_IRQ` on the held route): on
//! each wake the task [`InputDriver::drain`]s one bounded batch of completed
//! interrupt reports from the primary interrupter, decodes the boot report
//! ([`KeyboardReport`]/[`MouseReport`]), delivers the
//! [`InputReport`] into the focused subscriber's queue through a `sink`, and
//! re-arm for the next.  **No per-report kernel crossing, allocation, lock,
//! or copy beyond the fixed report buffer** — the §5.8 contract.
//!
//! The `sink` is the seam that keeps focus/routing where it belongs (in the
//! service, `kore_svc_input::InputService::route`) and decode here: this core
//! never sees a subscriber, a capability, or the mock — it takes a
//! `FnMut(InputReport) -> bool` and fills it.  Structurally mock-free (§2.1):
//! it names only the `kore_driver` seam and `kore_abi`.
//!
//! [`InputReport`]: kore_abi::svc::InputReport

use kore_abi::svc::InputReport;

use kore_driver::seam::{DmaPool, Regs64};

use crate::Controller;
use crate::device::Device;
#[cfg(any(test, feature = "alloc"))]
use crate::device::{ControlProgress, ControlSubmission};
#[cfg(any(test, feature = "alloc"))]
use crate::device::{MAX_ENDPOINTS, Speed};
use crate::hid::{KeyboardReport, MouseReport};
#[cfg(any(test, feature = "alloc"))]
use crate::{Error, InterruptEvent, Result};

#[cfg(any(test, feature = "alloc"))]
use crate::descriptor::{
    ConfigDescriptor, ConfigItem, ConfigWalk, EndpointDescriptor, HID_PROTOCOL_KEYBOARD,
    HID_PROTOCOL_MOUSE, InterfaceDescriptor, desc_type,
};

/// Which HID boot device a configured interface is — the boot report format
/// is fixed per class (USB HID 1.11 Appendix B), so decode branches on this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReportKind {
    /// A boot keyboard (interface protocol 1): 8-byte modifier + 6-key report.
    Keyboard,
    /// A boot mouse (interface protocol 2): buttons + signed X/Y (+ wheel).
    Mouse,
}

/// The `idVendor` and `idProduct` a USB device's descriptor declares (USB 2.0
/// §9.6.1). Self-reported, so it names a device's kind for policy, never its
/// authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UsbIdentity {
    pub vendor: u16,
    pub product: u16,
}

/// Controller-local HID identity, not a wire source ID or device authority.
/// The service assigns a disjoint wire ID across controllers and retires it
/// on the tagged lifecycle edge before this tuple can be reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HidSource {
    /// One-based root-hub port.
    pub port: u8,
    /// Actual `bInterfaceNumber`, not its ordinal in the descriptor stream.
    pub interface: u8,
}

#[cfg(any(test, feature = "alloc"))]
#[derive(Clone, Copy)]
struct HidInterface {
    number: u8,
    kind: ReportKind,
    dci: u8,
    armed: Option<u64>,
    active: bool,
    reset_pending: bool,
    /// A keyboard's indicators; inert on any other kind.
    leds: Leds,
}

/// Failed LED reports tolerated for one state before an interface stops
/// being asked to show it — until the state changes. The PS/2 indicator
/// path's bound, for the same reason: a keyboard that refuses its output
/// report must cost a few requests, not one per wake forever.
#[cfg(any(test, feature = "alloc"))]
const MAX_LED_FAILURES: u8 = 3;

/// One keyboard interface's indicator state.
#[cfg(any(test, feature = "alloc"))]
#[derive(Clone, Copy, Debug, Default)]
struct Leds {
    /// What the keyboard is known to show — `None` until a report lands,
    /// and again after one fails (the keyboard's state is then unknown).
    shown: Option<u8>,
    /// Consecutive failed reports for [`failed`](Self::failed).
    failures: u8,
    failed: u8,
}

/// The LED report in flight on a device's control pipe.
#[cfg(any(test, feature = "alloc"))]
#[derive(Clone, Copy, Debug)]
struct LedOutput {
    interface: u8,
    leds: u8,
    submission: ControlSubmission,
}

/// What [`InputDriver::sync_leds`] found or did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LedSync {
    /// The keyboard is known to show the requested state.
    Shown,
    /// A report carrying it was submitted; its completion arrives through
    /// the drain.
    Submitted,
    /// The device's control pipe carries an earlier report still. Ask
    /// again after a drain — the one that completes it.
    Busy,
    /// The keyboard refused this state too often (or its control pipe
    /// could not be recovered); it is not asked again until the state
    /// changes or the device is replugged.
    Refused,
    /// No live keyboard interface answers to this identity.
    Unknown,
}

/// One owned USB device and its bounded set of HID interfaces. Device/DMA
/// authority is held exactly once even when a receiver exposes both classes.
#[cfg(any(test, feature = "alloc"))]
pub struct InputSource<P: DmaPool> {
    /// xHCI slot, transfer rings, and report buffer.
    pub device: Device<P>,
    /// The device descriptor's `idVendor` and `idProduct`, once read.
    identity: Option<UsbIdentity>,
    interfaces: [Option<HidInterface>; MAX_ENDPOINTS],
    /// The one LED report this device's control pipe carries, if any —
    /// EP0 and its data buffer are per device, so a composite's keyboard
    /// interfaces take turns.
    output: Option<LedOutput>,
    /// EP0 failed a report and could not be recovered: nothing more is
    /// submitted on it. Input is unaffected — reports ride the interrupt
    /// endpoints.
    output_wedged: bool,
}

#[cfg(any(test, feature = "alloc"))]
impl<P: DmaPool> InputSource<P> {
    /// The live interface numbered `number`.
    fn interface(&self, number: u8) -> Option<&HidInterface> {
        self.interfaces
            .iter()
            .flatten()
            .find(|interface| interface.number == number && interface.active)
    }

    /// The interface numbered `number`, live or not — a completion may
    /// land after its interface stopped reporting.
    fn interface_mut(&mut self, number: u8) -> Option<&mut HidInterface> {
        self.interfaces
            .iter_mut()
            .flatten()
            .find(|interface| interface.number == number)
    }
}

/// One IRQ batch's visible outcome.
#[cfg(any(test, feature = "alloc"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InputDrain {
    /// Transfer events consumed from the xHCI event ring.
    pub events: u32,
    /// Well-formed reports accepted by the service sink.
    pub delivered: u32,
    /// Unknown endpoints, failed completions/decodes, sink refusals, or
    /// re-arm failures.
    pub errors: u32,
    /// Endpoints found halted and recovered this batch (see
    /// [`Controller::reset_endpoint`]).
    ///
    /// Distinct from `errors` because it is the *cure*, not the disease: a
    /// non-zero count means a transfer failed AND the endpoint was brought
    /// back. A machine reporting steady halts is telling you about its cable
    /// or its hub, and one reporting halts that stop being followed by
    /// reports is telling you the recovery itself is not taking.
    pub halts_recovered: u32,
    /// The most recent non-success transfer completion code (xHCI 6.4.5:
    /// 3 = Babble, 4 = USB Transaction Error, 6 = Stall, 13 = Short Packet).
    ///
    /// Kept because the drain used to discard it, which left a frozen
    /// keyboard indistinguishable from an idle one in the only diagnostic a
    /// serial-less machine has.
    pub last_completion_code: u8,
    /// LED reports a keyboard accepted this batch — the cue to
    /// [`sync_leds`](InputDriver::sync_leds) again, since its control pipe
    /// is free.
    pub outputs: u32,
    /// LED reports a keyboard refused this batch. Not counted in
    /// [`errors`](Self::errors): a keyboard without indicators that STALLs
    /// its output report is a fact about the keyboard, not a fault in the
    /// input path.
    pub output_failures: u32,
}

/// **What each root-hub port's last examination concluded** — the record
/// that makes enumeration idempotent.
///
/// Enumerating a port is not a read: it drives a port reset and two full
/// descriptor fetches, and the reset raises a Port Status Change event of
/// its own.  Without a memory of what the last examination found, a port
/// whose device is not a HID boot keyboard/mouse is re-examined on every
/// wake, and its own resets supply the wakes — the unbounded loop a real
/// Lenovo boot hit with a USB stick attached and no keyboard (QEMU never
/// reproduced it, because every gate topology had a keyboard and discovery
/// succeeded on the first pass).
///
/// The memo is keyed on CCS (Current Connect Status), because CCS *is* the
/// question a re-examination would answer: a port whose connect state
/// matches what we already examined has the same device on it, and a reset
/// plus two descriptor fetches cannot say anything the first pass did not.
///
/// Two 256-bit words, indexed by the 1-based port number — a `u8` port count
/// tops out at 255, so the whole table is 64 bytes of `InputDriver`, fixed
/// and allocation-free on a path that must not allocate.  Hand-rolled rather
/// than reusing `src/collections`: that lives in the kernel and is
/// unreachable from `drivers/`.
#[cfg(any(test, feature = "alloc"))]
#[derive(Clone, Copy)]
struct PortMemo {
    /// Ports whose last examination reached a conclusion.
    examined: [u32; 8],
    /// What CCS read at that examination (meaningful only where
    /// `examined`).
    connected: [u32; 8],
}

#[cfg(any(test, feature = "alloc"))]
impl PortMemo {
    const fn new() -> Self {
        Self {
            examined: [0; 8],
            connected: [0; 8],
        }
    }

    /// Word index + bit mask for port `n` (1-based; `n = 0` is not a port
    /// but lands harmlessly in bit 0, which no port number claims).
    const fn bit(n: u8) -> (usize, u32) {
        ((n >> 5) as usize, 1u32 << (n & 31))
    }

    fn is_examined(&self, n: u8) -> bool {
        let (w, m) = Self::bit(n);
        self.examined[w] & m != 0
    }

    fn was_connected(&self, n: u8) -> bool {
        let (w, m) = Self::bit(n);
        self.connected[w] & m != 0
    }

    /// Record what an examination of port `n` concluded.
    fn settle(&mut self, n: u8, connected: bool) {
        let (w, m) = Self::bit(n);
        self.examined[w] |= m;
        if connected {
            self.connected[w] |= m;
        } else {
            self.connected[w] &= !m;
        }
    }

    /// Discard port `n`'s record — its next examination starts fresh.
    fn forget(&mut self, n: u8) {
        let (w, m) = Self::bit(n);
        self.examined[w] &= !m;
        self.connected[w] &= !m;
    }

    /// Would examining port `n` tell us anything new, given its current
    /// CCS?  True for a port never examined (first attach) and for one whose
    /// connect state has moved since.
    fn stale(&self, n: u8, connected: bool) -> bool {
        !self.is_examined(n) || self.was_connected(n) != connected
    }

    /// **Is this port's state news?** — and forget its record if so.
    ///
    /// `connect_changed` is the port's latched CSC.  The second clause
    /// catches a connect edge whose CSC was consumed by something else (our
    /// own [`reset_port`](Controller::reset_port) clears CSC, so a plug that
    /// lands inside that window leaves no latch behind); comparing CCS
    /// against the record recovers it at no extra cost, since the PORTSC
    /// read already happened.
    fn note_connect_change(&mut self, n: u8, connected: bool, connect_changed: bool) -> bool {
        let changed =
            connect_changed || (self.is_examined(n) && self.was_connected(n) != connected);
        if changed {
            self.forget(n);
        }
        changed
    }
}

/// The configured HID edge owned by a device task. Discovery is startup-only;
/// [`drain`](Self::drain) is fixed-stack, allocation-free, and runs only after
/// the task wakes from its held interrupt route.
#[cfg(any(test, feature = "alloc"))]
pub struct InputDriver<P: DmaPool> {
    sources: alloc::vec::Vec<InputSource<P>>,
    discovery_errors: u32,
    /// Connected ports examined that **answered**: the descriptors were
    /// read and the device exposes no HID boot interface. A conclusion
    /// about the hardware, not a fault — a disk, a hub, a webcam each
    /// land here.
    discovery_no_hid: u32,
    /// Connected ports examined that **did not answer**: the port never
    /// got far enough to say what is attached, so nothing was learned.
    ///
    /// Split from [`discovery_no_hid`](Self::discovery_no_hid) because
    /// the two are opposite diagnoses that one "errors" total cannot tell
    /// apart, and on a machine with no serial port that total is all
    /// anyone can see. A controller hosting four non-keyboards and a
    /// controller whose every descriptor fetch is refused both read as
    /// "4 errors" — and they need entirely different fixes.
    discovery_inconclusive: u32,
    /// Sources found and then lost at `arm_interrupt`.
    discovery_arm_failed: u32,
    /// `inconclusive` split by the error that ended the examination.
    discovery_err_dma_alloc: u32,
    discovery_err_not_ready: u32,
    discovery_err_timeout: u32,
    discovery_err_completion: u32,
    discovery_last_completion_code: u8,
    discovery_err_unsupported: u32,
    /// Connected ports examined at all — the denominator for the three
    /// counters above, and by itself the answer to "is there anything on
    /// the controller we picked?".
    discovery_examined: u32,
    /// What each port's last examination concluded — see [`PortMemo`].
    ports: PortMemo,
}

/// The split behind `InputDriver`'s discovery total — see
/// [`InputDriver::discovery_census`].
#[cfg(any(test, feature = "alloc"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DiscoveryCensus {
    /// Connected ports examined.
    pub examined: u32,
    /// …of which answered, and are not HID boot devices.
    pub no_hid: u32,
    /// …of which never answered.
    pub inconclusive: u32,
    /// …of which were HID but could not be armed.
    pub arm_failed: u32,
    /// Of [`inconclusive`](Self::inconclusive): the pool had no DMA
    /// buffer left. Distinctive because it is *cumulative* — a run of
    /// successes followed by nothing but this is an exhaustion or a leak,
    /// not a property of the devices.
    pub err_dma_alloc: u32,
    /// Of `inconclusive`: the port or controller never became ready
    /// (a reset that did not take, CNR stuck, or controller-fatal
    /// `USBSTS.HCE`; QEMU may also raise HCE for a failed DMA access).
    pub err_not_ready: u32,
    /// Of `inconclusive`: a command or transfer produced no completion
    /// event within the poll budget. The signature of a transfer whose
    /// data never moved; only a matching IOMMU fault record proves that
    /// translation refusal was the cause.
    pub err_timeout: u32,
    /// Of `inconclusive`: the device answered with a non-success
    /// completion code — the controller and the DMA path both worked.
    pub err_completion: u32,
    /// The most recent non-success completion code, when
    /// [`err_completion`](Self::err_completion) is non-zero. xHCI 6.4.5:
    /// 4 = USB Transaction Error, 6 = Stall, 13 = Short Packet.
    pub last_completion_code: u8,
    /// Of `inconclusive`: geometry or a descriptor this core cannot serve.
    pub err_unsupported: u32,
}

#[cfg(any(test, feature = "alloc"))]
impl<P: DmaPool> InputDriver<P> {
    /// An unenumerated input owner. This is the startup state when the
    /// controller is running before firmware/emulation has exposed a connected
    /// port; the service may leave it only after a real port-change interrupt.
    pub const fn empty() -> Self {
        Self {
            sources: alloc::vec::Vec::new(),
            discovery_errors: 0,
            discovery_no_hid: 0,
            discovery_inconclusive: 0,
            discovery_arm_failed: 0,
            discovery_err_dma_alloc: 0,
            discovery_err_not_ready: 0,
            discovery_err_timeout: 0,
            discovery_err_completion: 0,
            discovery_last_completion_code: 0,
            discovery_err_unsupported: 0,
            discovery_examined: 0,
            ports: PortMemo::new(),
        }
    }

    /// Enumerate every currently connected HID boot keyboard/mouse, then arm
    /// them only after all command/EP0 traffic on interrupter 0 is quiescent.
    pub fn discover<R: Regs64>(
        controller: &mut Controller<R, P>,
        regs: &R,
        pool: &P,
        clock: fn() -> Option<u64>,
        pause: impl FnMut(),
    ) -> Self {
        let mut this = Self::empty();
        this.enumerate(controller, regs, pool, clock, pause);
        this
    }

    /// **Examine every port the memo cannot already answer for**, arming
    /// what turns out to be a HID boot device.
    ///
    /// A port whose CCS matches its record is skipped whole — no reset, no
    /// Address Device, no descriptor fetch.  That is what keeps a connected
    /// non-HID device (the stick the machine booted from) from being
    /// re-enumerated on every wake, and it also means the boot medium is
    /// reset exactly once per connect edge instead of once per attempt.
    fn enumerate<R: Regs64>(
        &mut self,
        controller: &mut Controller<R, P>,
        regs: &R,
        pool: &P,
        clock: fn() -> Option<u64>,
        mut pause: impl FnMut(),
    ) {
        for port in 1..=controller.port_count() {
            let connected = controller.port(regs, port).connected();
            if !self.ports.stale(port, connected) {
                continue;
            }
            // A quick unplug/replug leaves CCS true and CSC set. Retire the
            // previous affine owner before re-addressing even in that case.
            // A failed Disable Slot retains its DMA and blocks reuse.
            if let Err(error) = self.retire_port(controller, regs, port, clock, &mut pause) {
                self.record_discovery_error(error);
                continue;
            }
            if !connected {
                self.ports.settle(port, false);
                continue;
            }
            match discover_source(controller, regs, pool, port, clock, &mut pause) {
                Ok((source, outcome)) => {
                    self.discovery_examined = self.discovery_examined.saturating_add(1);
                    let live = source.interfaces.iter().any(Option::is_some);
                    self.sources.push(source);
                    if !live {
                        // Descriptors/no-HID and failed class negotiation also
                        // own a slot. Release it before freeing the buffers.
                        let _ = self.retire_port(controller, regs, port, clock, &mut pause);
                    }
                    match outcome {
                        Ok(()) => {
                            self.ports.settle(port, true);
                            if !live {
                                self.discovery_errors = self.discovery_errors.saturating_add(1);
                                self.discovery_no_hid = self.discovery_no_hid.saturating_add(1);
                            }
                        }
                        Err(error) => self.record_discovery_error(error),
                    }
                }
                Err(err) => {
                    // Inconclusive — the port never got far enough to say what
                    // is attached, so a retry could still learn something.
                    // Deliberately NOT recorded; the caller's discovery budget
                    // is what bounds these, and it is a backstop precisely
                    // because this arm exists.
                    self.discovery_examined = self.discovery_examined.saturating_add(1);
                    self.record_discovery_error(err);
                }
            }
        }

        // Do not arm an endpoint while later ports still issue commands or
        // control transfers on the same event ring. A HID completion in that
        // interval could otherwise be consumed as an unrelated event.
        for source in &mut self.sources {
            for entry in &mut source.interfaces {
                let Some(interface) = entry.as_mut() else {
                    continue;
                };
                if !interface.active || interface.armed.is_some() {
                    continue;
                }
                if let Ok(submitted) =
                    controller.arm_interrupt_at(regs, &mut source.device, interface.dci, 0)
                {
                    interface.armed = Some(submitted);
                } else {
                    interface.active = false;
                    self.discovery_errors = self.discovery_errors.saturating_add(1);
                    self.discovery_arm_failed = self.discovery_arm_failed.saturating_add(1);
                }
            }
        }
    }

    fn record_discovery_error(&mut self, error: Error) {
        self.discovery_errors = self.discovery_errors.saturating_add(1);
        self.discovery_inconclusive = self.discovery_inconclusive.saturating_add(1);
        let counter = match error {
            Error::DmaAlloc => &mut self.discovery_err_dma_alloc,
            Error::NotReady => &mut self.discovery_err_not_ready,
            Error::Timeout => &mut self.discovery_err_timeout,
            Error::Unsupported => &mut self.discovery_err_unsupported,
            Error::Completion(code) => {
                self.discovery_last_completion_code = code;
                &mut self.discovery_err_completion
            }
        };
        *counter = counter.saturating_add(1);
    }

    /// **Consume the root hub's connect edges**, reporting whether any port
    /// actually saw a device arrive or leave.
    ///
    /// A Port Status Change event says only "some port's status moved", and
    /// *our own enumeration raises one* — a port reset sets PRC.  So a PSC
    /// edge is not evidence of a new device, and treating it as such is what
    /// let discovery's side effect keep refunding the budget meant to bound
    /// it.  The distinction lives in PORTSC: `reset_port` acknowledges CSC,
    /// so a latched CSC is a real plug or unplug and nothing else.
    ///
    /// Each edge is acknowledged as it is taken (an unacked change bit stays
    /// latched forever and would report the same plug on every wake), and the
    /// port's memo is dropped so the next
    /// [`discover_after_interrupt`](Self::discover_after_interrupt)
    /// re-enumerates it — which is exactly what makes hot-plug work.
    pub fn note_port_changes<R: Regs64>(
        &mut self,
        controller: &Controller<R, P>,
        regs: &R,
    ) -> bool {
        self.note_port_changes_with(controller, regs, &mut |_| {})
    }

    /// Publish lifecycle edges at the same point that retires discovery memo.
    /// The router can cancel held input before a port identity is reused.
    pub fn note_port_changes_with<R: Regs64>(
        &mut self,
        controller: &Controller<R, P>,
        regs: &R,
        changed_port: &mut impl FnMut(u8),
    ) -> bool {
        self.note_changes(controller, regs, changed_port, &mut |_| {})
    }

    /// Retire each interface identity before a port can be re-enumerated.
    /// Production routers must use this callback, not the lossy port-only
    /// compatibility helper, so a composite's sources are reset separately.
    pub fn note_source_changes_with<R: Regs64>(
        &mut self,
        controller: &Controller<R, P>,
        regs: &R,
        changed_source: &mut impl FnMut(HidSource),
    ) -> bool {
        self.note_changes(controller, regs, &mut |_| {}, changed_source)
    }

    fn note_changes<R: Regs64>(
        &mut self,
        controller: &Controller<R, P>,
        regs: &R,
        changed_port: &mut impl FnMut(u8),
        changed_source: &mut impl FnMut(HidSource),
    ) -> bool {
        let mut changed = false;
        for n in 1..=controller.port_count() {
            let port = controller.port(regs, n);
            if self
                .ports
                .note_connect_change(n, port.connected(), port.connect_changed())
            {
                controller.ack_connect_change(regs, n);
                for source in self
                    .sources
                    .iter_mut()
                    .filter(|source| source.device.port == n)
                {
                    for entry in &mut source.interfaces {
                        if let Some(interface) = entry.take() {
                            changed_source(HidSource {
                                port: n,
                                interface: interface.number,
                            });
                        }
                    }
                }
                changed_port(n);
                changed = true;
            }
        }
        changed
    }

    #[inline]
    pub fn source_count(&self) -> usize {
        self.sources().count()
    }

    /// Every live controller-local interface identity, bounded by four per
    /// owned device. Enumeration/refusal never aliases excess interfaces.
    pub fn sources(&self) -> impl Iterator<Item = HidSource> + '_ {
        self.sources.iter().flat_map(|source| {
            source
                .interfaces
                .iter()
                .flatten()
                .filter(|interface| interface.active)
                .map(move |interface| HidSource {
                    port: source.device.port,
                    interface: interface.number,
                })
        })
    }

    /// The boot class a live interface reports as.
    pub fn kind(&self, source: HidSource) -> Option<ReportKind> {
        self.sources
            .iter()
            .filter(|owner| owner.device.port == source.port)
            .find_map(|owner| owner.interface(source.interface))
            .map(|interface| interface.kind)
    }

    /// The USB identity of the device a live interface belongs to: what its
    /// device descriptor declared, which the device itself chose to say.
    pub fn identity(&self, source: HidSource) -> Option<UsbIdentity> {
        self.sources
            .iter()
            .filter(|owner| owner.device.port == source.port)
            .find(|owner| owner.interface(source.interface).is_some())
            .and_then(|owner| owner.identity)
    }

    #[inline]
    pub const fn discovery_errors(&self) -> u32 {
        self.discovery_errors
    }

    /// The same total, split by what each examination actually concluded.
    pub const fn discovery_census(&self) -> DiscoveryCensus {
        DiscoveryCensus {
            examined: self.discovery_examined,
            no_hid: self.discovery_no_hid,
            inconclusive: self.discovery_inconclusive,
            arm_failed: self.discovery_arm_failed,
            err_dma_alloc: self.discovery_err_dma_alloc,
            err_not_ready: self.discovery_err_not_ready,
            err_timeout: self.discovery_err_timeout,
            err_completion: self.discovery_err_completion,
            last_completion_code: self.discovery_last_completion_code,
            err_unsupported: self.discovery_err_unsupported,
        }
    }

    /// Retire a port after its lifecycle callback. DMA remains owned until a
    /// successful Disable Slot proves that no endpoint can read it again.
    /// A failed disable quarantines this one slot, not its live siblings.
    fn retire_port<R: Regs64>(
        &mut self,
        controller: &mut Controller<R, P>,
        regs: &R,
        port: u8,
        clock: fn() -> Option<u64>,
        pause: &mut impl FnMut(),
    ) -> Result<()> {
        let mut i = 0;
        while i < self.sources.len() {
            if self.sources[i].device.port != port {
                i += 1;
                continue;
            }
            let slot = self.sources[i].device.slot_id;
            self.sources[i].interfaces.fill(None);
            controller.disable_slot(regs, slot, clock, &mut *pause)?;
            self.sources.swap_remove(i);
        }
        Ok(())
    }

    /// Retry startup discovery after an observed xHCI route edge. This is not
    /// a polling fallback: callers invoke it only after `WAIT_IRQ` reports a
    /// real delivery. Existing live sources are never replaced.
    ///
    /// Cheap to call on a wake that has nothing to offer: the memo answers
    /// for every port whose device has not changed, so a retry over a
    /// settled root hub touches PORTSC once per port and stops there.
    pub fn discover_after_interrupt<R: Regs64>(
        &mut self,
        controller: &mut Controller<R, P>,
        regs: &R,
        pool: &P,
        clock: fn() -> Option<u64>,
        pause: impl FnMut(),
    ) -> bool {
        // No "we already have one" short-circuit: `sources` being non-empty
        // says nothing about the port that just changed. Guarding on it is
        // what made a re-plug unreachable, because the guard could only be
        // lifted by a list that nothing ever shortened.
        //
        // Skipping the guard costs nothing: [`PortMemo`] already answers for
        // every port whose device has not changed, so a wake with no real
        // edge reads PORTSC once per port and stops there.
        self.enumerate(controller, regs, pool, clock, pause);
        self.source_count() != 0
    }

    /// Drain one bounded event batch, route each completion to its source,
    /// decode valid bytes, and re-arm. Failed completion codes are counted and
    /// never exposed as report data.
    pub fn drain<R: Regs64>(
        &mut self,
        controller: &mut Controller<R, P>,
        regs: &R,
        clock: fn() -> Option<u64>,
        mut pause: impl FnMut(),
        sink: &mut impl FnMut(InputReport) -> bool,
    ) -> InputDrain {
        // A port-only wire identity cannot represent a composite device.
        // Keep single-interface callers compatible, but never silently merge
        // two interfaces under one source. Production uses the tagged API.
        self.drain_inner(
            controller,
            regs,
            clock,
            &mut pause,
            false,
            &mut |source, mut report| {
                report.source = source.port;
                sink(report)
            },
        )
    }

    /// Drain with the interface's controller-local identity. `report.source`
    /// is deliberately zero: the router must assign its own wire namespace.
    /// Failed endpoint transfers deliver a tagged Reset before recovery.
    pub fn drain_with_sources<R: Regs64>(
        &mut self,
        controller: &mut Controller<R, P>,
        regs: &R,
        clock: fn() -> Option<u64>,
        mut pause: impl FnMut(),
        sink: &mut impl FnMut(HidSource, InputReport) -> bool,
    ) -> InputDrain {
        self.drain_inner(controller, regs, clock, &mut pause, true, sink)
    }

    /// **Converge `source`'s keyboard LEDs on `leds`** (the boot output
    /// report's bits — HID 1.11 Appendix B.1: Num Lock bit 0, Caps Lock
    /// bit 1, Scroll Lock bit 2).
    ///
    /// Never waits. A keyboard that may not show `leds` and whose device
    /// has no report in flight is sent one now; its completion is claimed by
    /// a later [`drain_with_sources`](Self::drain_with_sources), after which
    /// a further call answers [`LedSync::Shown`] — so a caller holding a
    /// desired state per source converges by calling this after every
    /// drain, the shape Linux 7.2.11 `usbkbd.c` takes (one request in
    /// flight; on completion, send again if the wanted state moved).
    pub fn sync_leds<R: Regs64>(
        &mut self,
        controller: &mut Controller<R, P>,
        regs: &R,
        source: HidSource,
        leds: u8,
    ) -> LedSync {
        let Some(owner) = self.sources.iter_mut().find(|owner| {
            owner.device.port == source.port && owner.interface(source.interface).is_some()
        }) else {
            return LedSync::Unknown;
        };
        let InputSource {
            device,
            interfaces,
            output,
            output_wedged,
            ..
        } = owner;
        let Some(interface) = interfaces
            .iter_mut()
            .flatten()
            .find(|interface| interface.number == source.interface && interface.active)
        else {
            return LedSync::Unknown;
        };
        if interface.kind != ReportKind::Keyboard {
            return LedSync::Unknown;
        }
        let state = &mut interface.leds;
        if state.shown == Some(leds) {
            return LedSync::Shown;
        }
        if *output_wedged || (state.failures >= MAX_LED_FAILURES && state.failed == leds) {
            return LedSync::Refused;
        }
        if output.is_some() {
            return LedSync::Busy;
        }
        match controller.submit_set_leds(regs, device, u16::from(source.interface), leds) {
            Ok(submission) => {
                *output = Some(LedOutput {
                    interface: source.interface,
                    leds,
                    submission,
                });
                LedSync::Submitted
            }
            // Only a request this core cannot express refuses here — a
            // one-byte OUT report always fits — so it is the keyboard's
            // `Refused`, not a retry.
            Err(_) => LedSync::Refused,
        }
    }

    /// Claim a default-control-pipe event for the LED report it belongs
    /// to; `None` when no device has a report in flight that it names.
    ///
    /// A failed report leaves EP0 halted on every code that halts an
    /// endpoint, a STALL among them — the answer of a keyboard without
    /// indicators — so it is recovered here, before anything else is sent
    /// on it. A pipe that cannot be recovered is retired from output for
    /// the device's life; its interrupt endpoints, and so its keys, are
    /// untouched.
    fn complete_output<R: Regs64>(
        sources: &mut [InputSource<P>],
        controller: &mut Controller<R, P>,
        regs: &R,
        clock: fn() -> Option<u64>,
        pause: &mut impl FnMut(),
        event: &InterruptEvent,
    ) -> Option<ControlProgress> {
        let (owner, sent, progress) = sources.iter_mut().find_map(|owner| {
            let sent = owner.output?;
            let progress = sent.submission.progress(event)?;
            Some((owner, sent, progress))
        })?;
        match progress {
            ControlProgress::Pending => {}
            ControlProgress::Complete => {
                owner.output = None;
                if let Some(interface) = owner.interface_mut(sent.interface) {
                    interface.leds = Leds {
                        shown: Some(sent.leds),
                        ..Leds::default()
                    };
                }
            }
            ControlProgress::Failed(_) => {
                owner.output = None;
                if let Some(interface) = owner.interface_mut(sent.interface) {
                    let leds = &mut interface.leds;
                    leds.shown = None;
                    if leds.failed == sent.leds {
                        leds.failures = leds.failures.saturating_add(1);
                    } else {
                        leds.failed = sent.leds;
                        leds.failures = 1;
                    }
                }
                if controller
                    .reset_endpoint_at(regs, &mut owner.device, 1, clock, &mut *pause)
                    .is_err()
                {
                    owner.output_wedged = true;
                }
            }
        }
        Some(progress)
    }

    fn drain_inner<R: Regs64>(
        &mut self,
        controller: &mut Controller<R, P>,
        regs: &R,
        clock: fn() -> Option<u64>,
        pause: &mut impl FnMut(),
        tagged: bool,
        sink: &mut impl FnMut(HidSource, InputReport) -> bool,
    ) -> InputDrain {
        const EVENT_BATCH: usize = 32;

        let mut events = [InterruptEvent::default(); EVENT_BATCH];
        let n = controller.drain_interrupts(regs, &mut events);
        let mut out = InputDrain {
            events: n as u32,
            ..InputDrain::default()
        };

        // A full sink cannot turn a reset into permission to publish a later
        // generation of state. Retain it, retry before fresh data, and keep
        // failed interface metadata until the physical detach callback.
        for source in &mut self.sources {
            for interface in source.interfaces.iter_mut().flatten() {
                if interface.reset_pending && (tagged || source.device.int_ep().is_some()) {
                    let identity = HidSource {
                        port: source.device.port,
                        interface: interface.number,
                    };
                    interface.reset_pending = !sink(
                        identity,
                        kore_abi::pointer::Record::Reset { all: false }.encode(0),
                    );
                    if interface.reset_pending {
                        out.errors = out.errors.saturating_add(1);
                    }
                }
            }
        }

        for event in &events[..n] {
            // The default control pipe carries only LED reports at runtime,
            // and only the one each device has in flight can claim it.
            if event.endpoint_id == 1 {
                match Self::complete_output(
                    &mut self.sources,
                    controller,
                    regs,
                    clock,
                    pause,
                    event,
                ) {
                    Some(ControlProgress::Pending) => {}
                    Some(ControlProgress::Complete) => out.outputs = out.outputs.saturating_add(1),
                    Some(ControlProgress::Failed(_)) => {
                        out.output_failures = out.output_failures.saturating_add(1);
                    }
                    None => out.errors = out.errors.saturating_add(1),
                }
                continue;
            }
            let Some((source, interface_index)) = self.sources.iter_mut().find_map(|source| {
                if source.device.slot_id != event.slot_id {
                    return None;
                }
                let index = source.interfaces.iter().position(|entry| {
                    entry.is_some_and(|interface| {
                        interface.dci == event.endpoint_id && interface.armed == Some(event.trb)
                    })
                })?;
                Some((source, index))
            }) else {
                out.errors = out.errors.saturating_add(1);
                continue;
            };

            let interface = source.interfaces[interface_index].expect("matched interface");
            let identity = HidSource {
                port: source.device.port,
                interface: interface.number,
            };
            let unambiguous = tagged || source.device.int_ep().is_some();
            source.interfaces[interface_index]
                .as_mut()
                .expect("matched interface")
                .armed = None;

            if event.data_completed() {
                let endpoint = source
                    .device
                    .endpoint(interface.dci)
                    .expect("matched endpoint remains configured");
                let len = (endpoint.mps as usize).saturating_sub(event.residual as usize);
                if unambiguous
                    && !interface.reset_pending
                    && feed_report(
                        interface.kind,
                        0,
                        source.device.read_report_at(interface.dci, len),
                        &mut |report| sink(identity, report),
                    )
                {
                    out.delivered = out.delivered.saturating_add(1);
                } else {
                    out.errors = out.errors.saturating_add(1);
                    if unambiguous {
                        source.interfaces[interface_index]
                            .as_mut()
                            .expect("matched interface")
                            .reset_pending = true;
                    }
                }
            } else {
                out.errors = out.errors.saturating_add(1);
                out.last_completion_code = event.completion_code;
                if unambiguous {
                    let landed = sink(
                        identity,
                        kore_abi::pointer::Record::Reset { all: false }.encode(0),
                    );
                    source.interfaces[interface_index]
                        .as_mut()
                        .expect("matched interface")
                        .reset_pending = !landed;
                    if !landed {
                        out.errors = out.errors.saturating_add(1);
                    }
                }
                // **A failed transfer may have halted the endpoint**, and a
                // halted endpoint ignores the doorbell the re-arm below is
                // about to ring. Clear it first or the re-arm is a no-op and
                // this source never speaks again — the freeze-under-load this
                // driver shipped with. Recovery is attempted for any failure
                // rather than a curated code list: the codes that halt vary
                // by part, and `reset_endpoint` already tolerates being told
                // to reset an endpoint that was not halted.
                if controller
                    .reset_endpoint_at(regs, &mut source.device, interface.dci, clock, &mut *pause)
                    .is_ok()
                {
                    out.halts_recovered = out.halts_recovered.saturating_add(1);
                } else {
                    out.errors = out.errors.saturating_add(1);
                    source.interfaces[interface_index]
                        .as_mut()
                        .expect("matched interface")
                        .active = false;
                    continue;
                }
            }

            if let Ok(submitted) =
                controller.arm_interrupt_at(regs, &mut source.device, interface.dci, 0)
            {
                source.interfaces[interface_index]
                    .as_mut()
                    .expect("matched interface")
                    .armed = Some(submitted);
            } else {
                out.errors = out.errors.saturating_add(1);
                let interface = source.interfaces[interface_index]
                    .as_mut()
                    .expect("matched interface");
                interface.active = false;
                interface.reset_pending = true;
            }
        }
        out
    }
}

#[cfg(any(test, feature = "alloc"))]
fn discover_source<R: Regs64, P: DmaPool>(
    controller: &mut Controller<R, P>,
    regs: &R,
    pool: &P,
    port_num: u8,
    clock: fn() -> Option<u64>,
    pause: &mut impl FnMut(),
) -> Result<(InputSource<P>, Result<()>)> {
    controller.reset_port(regs, port_num, clock, &mut *pause)?;
    let port = controller.port(regs, port_num);
    let device = controller.address_device(regs, pool, &port, clock, &mut *pause)?;
    let mut source = InputSource {
        device,
        identity: None,
        interfaces: [None; MAX_ENDPOINTS],
        output: None,
        output_wedged: false,
    };
    // Return the affine owner even on a later error: its caller must disable
    // the slot before dropping descriptor/endpoint DMA.
    let outcome = configure_hid(controller, regs, pool, &mut source, clock, pause);
    Ok((source, outcome))
}

#[cfg(any(test, feature = "alloc"))]
#[derive(Clone, Copy)]
struct HidCandidate {
    kind: ReportKind,
    interface: InterfaceDescriptor,
    endpoint: EndpointDescriptor,
}

/// Parse the complete configuration before issuing any class request. Only
/// alternate zero is active after SET_CONFIGURATION; selecting another
/// alternate without SET_INTERFACE would arm a pipe that does not exist.
#[cfg(any(test, feature = "alloc"))]
fn hid_candidates(bytes: &[u8], speed: Speed) -> Result<[Option<HidCandidate>; MAX_ENDPOINTS]> {
    // ConfigWalk is intentionally permissive for general inspection. Input
    // admission must distinguish a complete descriptor stream from a walk
    // that simply stopped at a malformed/truncated tail.
    let mut position = 0;
    while position < bytes.len() {
        let tail = &bytes[position..];
        let len = usize::from(*tail.first().ok_or(Error::Unsupported)?);
        if len < 2 || len > tail.len() {
            return Err(Error::Unsupported);
        }
        let required = match tail[1] {
            desc_type::CONFIGURATION | desc_type::INTERFACE => 9,
            desc_type::ENDPOINT => 7,
            _ => 2,
        };
        if len < required {
            return Err(Error::Unsupported);
        }
        position += len;
    }
    let mut candidates: [Option<HidCandidate>; MAX_ENDPOINTS] = [None; MAX_ENDPOINTS];
    let mut count = 0;
    let mut active: Option<(ReportKind, InterfaceDescriptor)> = None;
    let mut interfaces = [0u32; 8];
    let mut endpoints = 0u32;
    for item in ConfigWalk::new(bytes) {
        match item {
            ConfigItem::Interface(interface) => {
                active = None;
                if interface.alternate != 0 {
                    continue;
                }
                let word = usize::from(interface.number / 32);
                let bit = 1u32 << (interface.number % 32);
                if interfaces[word] & bit != 0 {
                    return Err(Error::Unsupported);
                }
                interfaces[word] |= bit;
                let kind = match interface.protocol {
                    HID_PROTOCOL_KEYBOARD if interface.is_hid_boot(HID_PROTOCOL_KEYBOARD) => {
                        ReportKind::Keyboard
                    }
                    HID_PROTOCOL_MOUSE if interface.is_hid_boot(HID_PROTOCOL_MOUSE) => {
                        ReportKind::Mouse
                    }
                    _ => continue,
                };
                active = Some((kind, interface));
            }
            ConfigItem::Endpoint(endpoint) if endpoint.is_in() && endpoint.is_interrupt() => {
                let Some((kind, interface)) = active else {
                    continue;
                };
                let minimum = match kind {
                    ReportKind::Keyboard => 8,
                    ReportKind::Mouse => 3,
                };
                let maximum = match speed {
                    Speed::Low => 8,
                    Speed::Full => 64,
                    _ => 1024,
                };
                if endpoint.number() == 0
                    || endpoint.address & 0x70 != 0
                    || endpoint.max_packet_size < minimum
                    || endpoint.max_packet_size > maximum
                    || endpoint.interval == 0
                    || (matches!(speed, Speed::High | Speed::Super) && endpoint.interval > 16)
                    || interface.num_endpoints == 0
                {
                    return Err(Error::Unsupported);
                }
                let bit = 1u32 << endpoint.dci();
                if endpoints & bit != 0 || count == MAX_ENDPOINTS {
                    return Err(Error::Unsupported);
                }
                endpoints |= bit;
                candidates[count] = Some(HidCandidate {
                    kind,
                    interface,
                    endpoint,
                });
                count += 1;
                // A boot interface has one report pipe. Do not invent another
                // source for a second endpoint under the same interface ID.
                active = None;
            }
            _ => {}
        }
    }
    Ok(candidates)
}

#[cfg(any(test, feature = "alloc"))]
fn configure_hid<R: Regs64, P: DmaPool>(
    controller: &mut Controller<R, P>,
    regs: &R,
    pool: &P,
    source: &mut InputSource<P>,
    clock: fn() -> Option<u64>,
    pause: &mut impl FnMut(),
) -> Result<()> {
    let device = &mut source.device;
    let descriptor = controller.device_descriptor(regs, device, clock, &mut *pause)?;
    source.identity = Some(UsbIdentity {
        vendor: descriptor.vendor,
        product: descriptor.product,
    });
    let device = &mut source.device;

    let n = controller.get_descriptor(
        regs,
        device,
        desc_type::CONFIGURATION,
        0,
        ConfigDescriptor::LEN as u16,
        clock,
        &mut *pause,
    )?;
    let header = ConfigDescriptor::parse(device.read_data(n)).ok_or(Error::Unsupported)?;
    if n != ConfigDescriptor::LEN
        || device.read_data(n)[0] != ConfigDescriptor::LEN as u8
        || header.total_length as usize > 4096
        || usize::from(header.total_length) < ConfigDescriptor::LEN
        || header.value == 0
    {
        return Err(Error::Unsupported);
    }
    let n = controller.get_descriptor(
        regs,
        device,
        desc_type::CONFIGURATION,
        0,
        header.total_length,
        clock,
        &mut *pause,
    )?;

    if n != usize::from(header.total_length)
        || ConfigDescriptor::parse(device.read_data(n)) != Some(header)
    {
        return Err(Error::Unsupported);
    }
    let candidates = hid_candidates(device.read_data(n), device.speed)?;
    if candidates.iter().all(Option::is_none) {
        return Ok(());
    }
    controller.set_configuration(regs, device, header.value, clock, &mut *pause)?;

    let mut accepted = [None; MAX_ENDPOINTS];
    let mut endpoints = [EndpointDescriptor::default(); MAX_ENDPOINTS];
    let mut count = 0;
    for candidate in candidates.iter().flatten() {
        let interface = u16::from(candidate.interface.number);
        match controller.set_protocol(regs, device, true, interface, clock, &mut *pause) {
            Ok(()) => {}
            Err(Error::Completion(6)) => {
                // Required boot mode is unsupported on this sibling only.
                // Recover EP0 before negotiating the next interface.
                controller.reset_endpoint_at(regs, device, 1, clock, &mut *pause)?;
                continue;
            }
            Err(error) => return Err(error),
        }
        match controller.set_idle(regs, device, 0, interface, clock, &mut *pause) {
            Ok(()) => {}
            Err(Error::Completion(6)) => {
                // HID1.11 Appendix G makes SET_IDLE optional for mice, but
                // mandatory for keyboards. Only protocol STALL is optional;
                // a timeout or a transport fault remains an actual failure.
                controller.reset_endpoint_at(regs, device, 1, clock, &mut *pause)?;
                if candidate.kind == ReportKind::Keyboard {
                    continue;
                }
            }
            Err(error) => return Err(error),
        }
        accepted[count] = Some(HidInterface {
            number: candidate.interface.number,
            kind: candidate.kind,
            dci: candidate.endpoint.dci(),
            armed: None,
            active: true,
            reset_pending: false,
            leds: Leds::default(),
        });
        endpoints[count] = candidate.endpoint;
        count += 1;
    }
    controller.configure_endpoints(regs, pool, device, &endpoints[..count], clock, &mut *pause)?;
    source.interfaces = accepted;
    Ok(())
}

/// Decode one raw HID boot report of `kind` from root-hub port `source` into an
/// [`InputReport`] and deliver it through `sink`.  Returns whether a
/// well-formed report was delivered — a too-short report decodes to `None`
/// and is dropped (not delivered), and the sink's own `false` (a full queue)
/// is propagated.  Allocation-free.
pub fn feed_report(
    kind: ReportKind,
    source: u8,
    raw: &[u8],
    sink: &mut impl FnMut(InputReport) -> bool,
) -> bool {
    let report = match kind {
        ReportKind::Keyboard => {
            KeyboardReport::parse(raw).map(|k| InputReport::keyboard(source, k.modifiers, k.keys))
        }
        ReportKind::Mouse => MouseReport::parse(raw)
            .map(|m| InputReport::mouse(source, m.buttons, m.dx, m.dy, m.wheel)),
    };
    match report {
        Some(r) => sink(r),
        None => false,
    }
}

/// **The §3.1a device drain**: pull up to `budget` completed interrupt
/// reports off `dev`'s IN endpoint, decode + deliver each through `sink`, and
/// **re-arm** the endpoint after each so the next report is caught.  Returns
/// how many well-formed reports were delivered.  The whole loop the input
/// service's driver task runs per IRQ-relay wake — level-triggered (the
/// device-written event ring is the state; the notice only says "look"), and
/// **allocation-free** (the fixed 8-byte boot-report buffer is the only
/// per-report state).  Stops early on a re-arm failure (surfaced by the
/// shrunk count, never a panic).
pub fn drain_reports<R: Regs64, P: DmaPool>(
    ctl: &mut Controller<R, P>,
    regs: &R,
    dev: &mut Device<P>,
    kind: ReportKind,
    source: u8,
    budget: usize,
    sink: &mut impl FnMut(InputReport) -> bool,
) -> usize {
    let mut delivered = 0;
    for _ in 0..budget {
        let Some(len) = ctl.poll_interrupt(regs, dev) else {
            break; // the event ring is drained — nothing more pending
        };
        // Copy the boot report out of the endpoint buffer before re-arming
        // (arming reuses it).  Boot reports are <= 8 bytes.
        let mut raw = [0u8; 8];
        let n = {
            let report = dev.read_report(len);
            let n = report.len().min(raw.len());
            raw[..n].copy_from_slice(&report[..n]);
            n
        };
        if feed_report(kind, source, &raw[..n], sink) {
            delivered += 1;
        }
        // Re-arm for the next report; a failed re-arm ends the drain.
        if ctl.arm_interrupt(regs, dev, 0).is_err() {
            break;
        }
    }
    delivered
}

#[cfg(test)]
#[path = "input/usb_admission_tests.rs"]
mod usb_admission_tests;

#[cfg(test)]
#[path = "input/composite_tests.rs"]
mod composite_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::{ConfigDescriptor, ConfigItem, ConfigWalk};
    use crate::mock::{MockPool, MockXhciDevice};

    fn spin() {}

    /// Collect delivered reports into a growable buffer (the test sink).
    fn collector(out: &mut alloc::vec::Vec<InputReport>) -> impl FnMut(InputReport) -> bool + '_ {
        move |r| {
            out.push(r);
            true
        }
    }

    /// Enumerate + configure the HID device on `port_num` and select boot
    /// protocol — the same setup the mock's own HID tests do, over the
    /// public driver API (so this test needs no mock-private helper).
    fn hid_ready(
        port_num: u8,
    ) -> (
        MockXhciDevice,
        Controller<MockXhciDevice, MockPool>,
        Device<MockPool>,
    ) {
        let dev = MockXhciDevice::new();
        let pool = MockPool;
        let mut ctrl = Controller::bring_up(&dev, &pool, || None, spin).expect("bring-up");
        ctrl.reset_port(&dev, port_num, || None, spin)
            .expect("reset");
        let port = ctrl.port(&dev, port_num);
        let mut usb = ctrl
            .address_device(&dev, &pool, &port, || None, spin)
            .expect("address");

        ctrl.get_descriptor(&dev, &mut usb, 2, 0, 9, || None, spin)
            .unwrap();
        let hdr = ConfigDescriptor::parse(usb.read_data(9)).unwrap();
        ctrl.get_descriptor(&dev, &mut usb, 2, 0, hdr.total_length, || None, spin)
            .unwrap();
        let mut ep = None;
        for item in ConfigWalk::new(usb.read_data(hdr.total_length as usize)) {
            if let ConfigItem::Endpoint(e) = item {
                if e.is_in() && e.is_interrupt() {
                    ep = Some(e);
                }
            }
        }
        let ep = ep.expect("interrupt IN endpoint");
        ctrl.set_configuration(&dev, &mut usb, hdr.value, || None, spin)
            .unwrap();
        ctrl.configure_endpoint(&dev, &pool, &mut usb, &ep, || None, spin)
            .unwrap();
        ctrl.set_protocol(&dev, &mut usb, true, 0, || None, spin)
            .unwrap();
        (dev, ctrl, usb)
    }

    /// `feed_report` decodes each boot class into the frozen [`InputReport`]
    /// layout, drops a too-short report, and honors sink backpressure.
    #[test]
    fn feed_report_decodes_and_respects_the_sink() {
        let mut out = alloc::vec::Vec::new();
        // Left-Shift held, 'a' (0x04) + 'b' (0x05).
        let raw = [0x02, 0x00, 0x04, 0x05, 0, 0, 0, 0];
        assert!(feed_report(
            ReportKind::Keyboard,
            3,
            &raw,
            &mut collector(&mut out)
        ));
        assert_eq!(
            out,
            alloc::vec![InputReport::keyboard(3, 0x02, [0x04, 0x05, 0, 0, 0, 0])]
        );

        // Left button, +5 X, -3 Y (3-byte boot mouse — no wheel).
        let raw = [0x01, 5, (-3i8) as u8];
        assert!(feed_report(
            ReportKind::Mouse,
            4,
            &raw,
            &mut collector(&mut out)
        ));
        assert_eq!(out[1], InputReport::mouse(4, 0x01, 5, -3, 0));

        // A too-short keyboard report decodes to nothing — dropped.
        assert!(!feed_report(
            ReportKind::Keyboard,
            3,
            &[0x02, 0x00],
            &mut collector(&mut out)
        ));
        assert_eq!(out.len(), 2, "the short report was dropped");

        // A refusing sink (a full queue) is reported as not-delivered.
        assert!(!feed_report(ReportKind::Keyboard, 3, &[0; 8], &mut |_| {
            false
        }));
    }

    /// `drain_reports` polls, decodes, delivers, and **re-arms** — so a
    /// stream of injected reports drains one per cycle into the sink, each
    /// decoded, no allocation on the path.
    #[test]
    fn drain_reports_pumps_the_stream_and_rearms() {
        let (dev, mut ctrl, mut usb) = hid_ready(1); // port 1 is the keyboard
        let slot = usb.slot_id;
        ctrl.arm_interrupt(&dev, &mut usb, 0).expect("initial arm");

        let mut out = alloc::vec::Vec::new();
        // Three keypresses in sequence; drain re-arms so each is caught.
        let injects: [[u8; 8]; 3] = [
            [0x02, 0x00, 0x04, 0, 0, 0, 0, 0], // Shift + 'a'
            [0x00, 0x00, 0x05, 0, 0, 0, 0, 0], // 'b'
            [0x00, 0x00, 0x00, 0, 0, 0, 0, 0], // all released
        ];
        for raw in injects {
            dev.inject_report(slot, &raw);
            let n = drain_reports(
                &mut ctrl,
                &dev,
                &mut usb,
                ReportKind::Keyboard,
                slot,
                4,
                &mut collector(&mut out),
            );
            assert_eq!(n, 1, "one report drained and delivered per cycle");
        }

        assert_eq!(
            out,
            alloc::vec![
                InputReport::keyboard(slot, 0x02, [0x04, 0, 0, 0, 0, 0]),
                InputReport::keyboard(slot, 0x00, [0x05, 0, 0, 0, 0, 0]),
                InputReport::keyboard(slot, 0x00, [0x00, 0, 0, 0, 0, 0]),
            ],
        );
        // An empty ring drains nothing.
        assert_eq!(
            drain_reports(
                &mut ctrl,
                &dev,
                &mut usb,
                ReportKind::Keyboard,
                slot,
                4,
                &mut collector(&mut out)
            ),
            0
        );
    }

    #[test]
    fn input_driver_discovers_then_drains_from_the_irq_edge() {
        let regs = MockXhciDevice::new();
        let pool = MockPool;
        let mut controller = Controller::bring_up(&regs, &pool, || None, spin).expect("bring-up");
        let mut input = InputDriver::discover(&mut controller, &regs, &pool, || None, spin);
        assert_eq!(input.source_count(), 2, "keyboard and mouse armed");
        assert_eq!(input.discovery_errors(), 0);

        // Enumeration assigns slot 1 to the keyboard in the mock.
        regs.inject_report(1, &[0x00, 0x00, 0x04, 0, 0, 0, 0, 0]);
        let mut reports = alloc::vec::Vec::new();
        let drained = input.drain(
            &mut controller,
            &regs,
            || None,
            spin,
            &mut collector(&mut reports),
        );
        assert_eq!(drained.events, 1);
        assert_eq!(drained.delivered, 1);
        assert_eq!(drained.errors, 0);
        assert_eq!(
            reports,
            alloc::vec![InputReport::keyboard(1, 0, [0x04, 0, 0, 0, 0, 0])]
        );
        assert!(
            regs.interrupt_acknowledged(0),
            "driver drain uses the full interrupt acknowledgement"
        );
    }

    // ── The enumeration memo ────────────────────────────────────────
    //
    // The metal failure these pin is a *state machine* over the port table,
    // not a controller behaviour, so they are driven as one: a port's CCS +
    // CSC in, a skip-or-examine decision out.  The mock is a simulated
    // controller and the 2026-07-24 ruling says not to grow that world; it
    // also carries no non-HID device, which is precisely the shape that
    // broke.  Only the last test needs a controller, and it uses the mock
    // exactly as it already is.

    /// A connected device that turned out not to be a HID boot keyboard —
    /// the boot stick — is examined once and then skipped whole.  This is
    /// the loop: without it, every wake re-ran a port reset and two
    /// descriptor fetches, and the reset supplied the next wake.
    #[test]
    fn an_unchanged_connected_port_is_not_re_examined() {
        let mut memo = PortMemo::new();

        // Never examined: the first attach must be looked at.
        assert!(memo.stale(3, true));

        // Examined, no HID boot interface found — still connected next wake.
        memo.settle(3, true);
        assert!(
            !memo.stale(3, true),
            "the same device on the same port has nothing new to say"
        );

        // And an empty port, once examined, likewise stays quiet.
        memo.settle(4, false);
        assert!(!memo.stale(4, false));
    }

    /// A latched CSC is a real plug or unplug: it drops the port's record,
    /// so the very next pass enumerates it again.  Hot-plug lives here.
    #[test]
    fn a_connect_change_re_opens_the_port() {
        let mut memo = PortMemo::new();
        memo.settle(2, true);
        assert!(!memo.stale(2, true));

        // Unplug: CSC latched, CCS now clear.
        assert!(memo.note_connect_change(2, false, true));
        assert!(memo.stale(2, false), "the port must be examined again");
        memo.settle(2, false);

        // Replug: CSC latched again, CCS back.  A keyboard plugged into a
        // port that previously held one is still a fresh enumeration.
        assert!(memo.note_connect_change(2, true, true));
        assert!(memo.stale(2, true));

        // Unplug-then-replug between two wakes leaves CCS unchanged and only
        // CSC to show for it — which is exactly why the latch, not the level,
        // is the signal.
        memo.settle(2, true);
        assert!(memo.note_connect_change(2, true, true));
        assert!(memo.stale(2, true));
    }

    /// **The bug, stated as a rule**: a Port Status Change with no connect
    /// change is not news, and must not refund the discovery budget.  Our own
    /// port reset raises exactly that edge.
    #[test]
    fn a_psc_without_a_connect_change_is_not_news() {
        let mut memo = PortMemo::new();
        memo.settle(1, true);

        // PRC-only wake: CCS unchanged, CSC clear (`reset_port` acked it).
        assert!(
            !memo.note_connect_change(1, true, false),
            "a reset-raised PSC must buy no refund — it is our own side effect"
        );
        assert!(!memo.stale(1, true), "and it must not re-open the port");

        // A CCS that disagrees with the record is still news even with no
        // latch, for the plug that lands inside `reset_port`'s CSC ack.
        assert!(memo.note_connect_change(1, false, false));
        assert!(memo.stale(1, false));

        // A never-examined port is not a *change* — nothing is known to have
        // moved.  The initial budget, not a refund, covers first attach.
        assert!(!memo.note_connect_change(7, true, false));
    }

    /// The table addresses every port a `u8` count can name, without two
    /// ports sharing a bit.
    #[test]
    fn the_memo_spans_the_whole_port_range() {
        let mut memo = PortMemo::new();
        for n in 1..=255u8 {
            memo.settle(n, n % 3 == 0);
        }
        for n in 1..=255u8 {
            assert!(memo.is_examined(n), "port {n} recorded");
            assert_eq!(memo.was_connected(n), n % 3 == 0, "port {n} connect state");
        }
        memo.forget(200);
        assert!(!memo.is_examined(200));
        assert!(
            memo.is_examined(199) && memo.is_examined(201),
            "no neighbour lost"
        );
    }

    /// **The unplug-replug cycle, end to end** — the bug this file shipped.
    ///
    /// `PortMemo` had this right all along (see
    /// `a_connect_change_re_opens_the_port`), and it still did not save us:
    /// the memo was simply never consulted once a source existed, because
    /// both `discover_after_interrupt` and its caller returned early on a
    /// non-empty source list, and nothing ever shortened that list. So the
    /// component was correct and the assembly was not — which is exactly the
    /// seam a unit test cannot see, and why this one drives the whole driver.
    ///
    /// The old code fails this at the second assertion: the source count
    /// stays at 2 across the unplug, because the departed keyboard's source
    /// is never retired.
    #[test]
    fn a_keyboard_can_be_unplugged_and_plugged_back_in() {
        let regs = MockXhciDevice::new();
        let pool = MockPool;
        let mut controller = Controller::bring_up(&regs, &pool, || None, spin).expect("bring-up");

        let mut input = InputDriver::discover(&mut controller, &regs, &pool, || None, spin);
        assert_eq!(
            input.source_count(),
            2,
            "keyboard on port 1, mouse on port 2"
        );

        // Unplug the keyboard. The edge must be news, and the source it
        // invalidated must go with it.
        regs.set_port_connected(1, false);
        let mut lifecycle_edges = 0;
        assert!(
            input.note_port_changes_with(&controller, &regs, &mut |port| {
                assert_eq!(port, 1);
                lifecycle_edges += 1;
            }),
            "a disconnect is a real connect edge"
        );
        assert_eq!(lifecycle_edges, 1);
        assert!(!input.note_port_changes_with(&controller, &regs, &mut |_| {
            panic!("acknowledged edge cannot reset the same generation again")
        }));
        input.discover_after_interrupt(&mut controller, &regs, &pool, || None, spin);
        assert_eq!(
            input.source_count(),
            1,
            "the departed keyboard's source must be retired, not merely \
             forgotten by the memo — a list that only grows is what latched \
             discovery shut"
        );

        // Plug it back in. This is the step that was unreachable.
        regs.set_port_connected(1, true);
        assert!(
            input.note_port_changes(&controller, &regs),
            "a re-plug is a real connect edge"
        );
        assert!(
            input.discover_after_interrupt(&mut controller, &regs, &pool, || None, spin),
            "re-enumeration must succeed"
        );
        assert_eq!(
            input.source_count(),
            2,
            "the re-plugged keyboard is enumerated again"
        );
    }

    /// **A report that lands during halt recovery is still delivered.**
    ///
    /// Recovery runs two commands, and waiting for a command completion
    /// drains the event ring — so every Transfer Event it has to consume on
    /// the way is off the ring for good unless something keeps it. Dropping
    /// them would trade a frozen keyboard for one that silently eats
    /// keystrokes under exactly the load that provoked the halt, which is
    /// not a fix.
    #[test]
    fn a_report_arriving_during_a_command_wait_is_not_lost() {
        let regs = MockXhciDevice::new();
        let pool = MockPool;
        let mut controller = Controller::bring_up(&regs, &pool, || None, spin).expect("bring-up");
        let mut input = InputDriver::discover(&mut controller, &regs, &pool, || None, spin);
        assert_eq!(input.source_count(), 2);

        // Park a report on the ring, then run a command that must scan past
        // it to reach its own completion.
        regs.inject_report(1, &[0, 0, 0x04, 0, 0, 0, 0, 0]);
        let _ = controller
            .enable_slot(&regs, || None, spin)
            .expect("a command runs");

        // The report survived the scan and reaches the decoder.
        let mut seen = 0u32;
        let drained = input.drain(&mut controller, &regs, || None, spin, &mut |_report| {
            seen += 1;
            true
        });
        assert_eq!(
            controller.stashed_dropped(),
            0,
            "the stash must be deep enough for a command wait"
        );
        assert!(
            drained.events >= 1 && seen >= 1,
            "the report consumed by the command wait must still be delivered \
             (events={}, delivered={seen})",
            drained.events
        );
    }

    /// A wake with no connect edge must still cost nothing once sources
    /// exist — the cheap path the source-count check is actually for.
    /// Removing that check outright would re-enumerate the world on every
    /// interrupt; the memo is what makes keeping it unnecessary safe.
    #[test]
    fn a_settled_root_hub_re_enumerates_nothing() {
        let regs = MockXhciDevice::new();
        let pool = MockPool;
        let mut controller = Controller::bring_up(&regs, &pool, || None, spin).expect("bring-up");

        let mut input = InputDriver::discover(&mut controller, &regs, &pool, || None, spin);
        let examined = input.discovery_census().examined;
        assert_eq!(input.source_count(), 2);

        // No port moved, so nothing is news and nothing is re-examined.
        assert!(!input.note_port_changes(&controller, &regs));
        input.discover_after_interrupt(&mut controller, &regs, &pool, || None, spin);
        assert_eq!(input.source_count(), 2, "no source added or lost");
        assert_eq!(
            input.discovery_census().examined,
            examined,
            "a settled hub costs one PORTSC read per port and no examination"
        );
    }

    /// Over the real controller: power-on connect edges are news exactly
    /// once, and **enumeration's own port resets are not news at all** —
    /// `reset_port` acknowledges CSC, so what it leaves behind is PRC.
    #[test]
    fn enumeration_does_not_look_like_a_plug() {
        let regs = MockXhciDevice::new();
        let pool = MockPool;
        let mut controller = Controller::bring_up(&regs, &pool, || None, spin).expect("bring-up");

        let mut watcher = InputDriver::<MockPool>::empty();
        assert!(
            watcher.note_port_changes(&controller, &regs),
            "the populated ports latch CSC at power-on: a real connect edge"
        );
        assert!(
            !watcher.note_port_changes(&controller, &regs),
            "the edge is acknowledged when taken, so it cannot pay twice"
        );

        // Enumerate for real — two port resets, two Address Devices, four
        // descriptor fetches.
        let discovered = InputDriver::discover(&mut controller, &regs, &pool, || None, spin);
        assert_eq!(discovered.source_count(), 2);
        assert!(
            !watcher.note_port_changes(&controller, &regs),
            "a reset raises PSC and sets PRC, never CSC — discovery must not \
             refund the budget that bounds it"
        );
    }

    // ── Keyboard indicators ─────────────────────────────────────────

    /// Discovery over the stock mock, and its keyboard's identity (port 1).
    fn keyboard_ready() -> (
        MockXhciDevice,
        Controller<MockXhciDevice, MockPool>,
        InputDriver<MockPool>,
        HidSource,
    ) {
        let regs = MockXhciDevice::new();
        let pool = MockPool;
        let mut controller = Controller::bring_up(&regs, &pool, || None, spin).expect("bring-up");
        let input = InputDriver::discover(&mut controller, &regs, &pool, || None, spin);
        let keyboard = input
            .sources()
            .find(|source| source.port == 1)
            .expect("the mock's keyboard");
        (regs, controller, input, keyboard)
    }

    /// The boot output report reaches the keyboard as the class request
    /// Linux sends, without the caller waiting for it; the drain that sees
    /// its completion is what makes the state "shown".
    #[test]
    fn a_led_report_is_submitted_without_waiting_and_settles_on_the_drain() {
        let (regs, mut controller, mut input, keyboard) = keyboard_ready();
        let caps = 0b010;
        assert_eq!(
            input.sync_leds(&mut controller, &regs, keyboard, caps),
            LedSync::Submitted
        );
        assert_eq!(
            input.sync_leds(&mut controller, &regs, keyboard, caps),
            LedSync::Busy,
            "one report per control pipe until its completion is drained"
        );
        assert_eq!(
            regs.class_requests().last(),
            Some(&(0x09, 0x0200, u16::from(keyboard.interface)))
        );
        assert_eq!(
            regs.class_payloads().last(),
            Some(&(u16::from(keyboard.interface), alloc::vec![caps]))
        );

        let drained = input.drain_with_sources(&mut controller, &regs, || None, spin, &mut |_, _| {
            panic!("an output completion is not a report")
        });
        assert_eq!((drained.outputs, drained.output_failures, drained.errors), (1, 0, 0));
        assert_eq!(
            input.sync_leds(&mut controller, &regs, keyboard, caps),
            LedSync::Shown
        );
        assert_eq!(
            input.sync_leds(&mut controller, &regs, keyboard, caps | 0b001),
            LedSync::Submitted,
            "a new state is a new report"
        );

        // A mouse has no indicators, and an absent interface is no keyboard.
        let mouse = input.sources().find(|source| source.port == 2).unwrap();
        assert_eq!(
            input.sync_leds(&mut controller, &regs, mouse, caps),
            LedSync::Unknown
        );
        let absent = HidSource {
            port: 4,
            interface: 0,
        };
        assert_eq!(
            input.sync_leds(&mut controller, &regs, absent, caps),
            LedSync::Unknown
        );
    }

    /// A keyboard without indicators STALLs its output report. EP0 is
    /// recovered every time — the halt would otherwise swallow the next
    /// request — and the keyboard is asked a bounded number of times per
    /// state, while its keys keep flowing.
    #[test]
    fn a_refused_led_report_recovers_ep0_and_is_retried_a_bounded_number_of_times() {
        let (regs, mut controller, mut input, keyboard) = keyboard_ready();
        regs.fail_hid_request(1, 0x09, u16::from(keyboard.interface), 6);
        let recoveries = regs.ep0_recoveries();
        for attempt in 1..=MAX_LED_FAILURES {
            assert_eq!(
                input.sync_leds(&mut controller, &regs, keyboard, 0b100),
                LedSync::Submitted,
                "attempt {attempt}"
            );
            let drained =
                input.drain_with_sources(&mut controller, &regs, || None, spin, &mut |_, _| true);
            assert_eq!((drained.outputs, drained.output_failures, drained.errors), (0, 1, 0));
            assert_eq!(regs.ep0_recoveries(), recoveries + u32::from(attempt));
        }
        assert_eq!(
            input.sync_leds(&mut controller, &regs, keyboard, 0b100),
            LedSync::Refused
        );
        assert_eq!(
            input.sync_leds(&mut controller, &regs, keyboard, 0b001),
            LedSync::Submitted,
            "a new state is a new chance"
        );
        let _ = input.drain_with_sources(&mut controller, &regs, || None, spin, &mut |_, _| true);

        regs.inject_report(1, &[0, 0, 0x04, 0, 0, 0, 0, 0]);
        let mut keys = 0;
        input.drain_with_sources(&mut controller, &regs, || None, spin, &mut |_, report| {
            keys += usize::from(report.data[0] == 0x04);
            true
        });
        assert_eq!(keys, 1, "the keys never depended on the indicators");
    }

    /// A keyboard that never answers costs its own report and nothing
    /// else: the caller is told the pipe is busy, every report still flows,
    /// and a late answer settles the state like a prompt one.
    #[test]
    fn an_unanswered_led_report_stalls_nothing_but_itself() {
        let (regs, mut controller, mut input, keyboard) = keyboard_ready();
        regs.hold_ep0();
        assert_eq!(
            input.sync_leds(&mut controller, &regs, keyboard, 0b010),
            LedSync::Submitted
        );
        regs.inject_report(1, &[0, 0, 0x04, 0, 0, 0, 0, 0]);
        let mut keys = 0;
        let drained =
            input.drain_with_sources(&mut controller, &regs, || None, spin, &mut |_, _| {
                keys += 1;
                true
            });
        assert_eq!((keys, drained.outputs, drained.errors), (1, 0, 0));
        assert_eq!(
            input.sync_leds(&mut controller, &regs, keyboard, 0b010),
            LedSync::Busy
        );

        regs.release_ep0(1);
        let drained =
            input.drain_with_sources(&mut controller, &regs, || None, spin, &mut |_, _| true);
        assert_eq!((drained.outputs, drained.errors), (1, 0));
        assert_eq!(
            input.sync_leds(&mut controller, &regs, keyboard, 0b010),
            LedSync::Shown
        );
    }

    /// An EP0 completion no report claims is somebody's bug, and is
    /// counted as one rather than taken for an answer.
    #[test]
    fn a_control_event_nothing_submitted_is_an_error_not_an_answer() {
        let (regs, mut controller, mut input, keyboard) = keyboard_ready();
        assert_eq!(
            input.sync_leds(&mut controller, &regs, keyboard, 0b010),
            LedSync::Submitted
        );
        // Settle the real report first: afterwards nothing is in flight,
        // so an EP0 event — even one naming the right slot — is a stray.
        let drained =
            input.drain_with_sources(&mut controller, &regs, || None, spin, &mut |_, _| true);
        assert_eq!((drained.outputs, drained.errors), (1, 0));
        let stray = InterruptEvent {
            slot_id: 1,
            endpoint_id: 1,
            residual: 0,
            completion_code: 1,
            trb: 0xdead_0000,
        };
        let mut sources = core::mem::take(&mut input.sources);
        assert_eq!(
            InputDriver::complete_output(
                &mut sources,
                &mut controller,
                &regs,
                || None,
                &mut spin,
                &stray
            ),
            None
        );
        input.sources = sources;
    }
}
