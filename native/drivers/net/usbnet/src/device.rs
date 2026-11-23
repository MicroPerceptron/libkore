//! **Bring-up and the data plane** — the ladder from an addressed USB device
//! to a live [`NicBackend`].

use core::marker::PhantomData;

use kore_abi::device::IovaAddr;
use kore_driver::net::backend::{
    LinkState, NicBackend, NicCounters, PacketFilter, Steering, TxCompletion, TxFrame, TxStatus,
};
use kore_driver::net::{FRAME_MAX, MacAddr};
use kore_driver::seam::{Dma, DmaPool, Regs64};
use kore_xhci::descriptor::{ConfigDescriptor, DeviceDescriptor, string_ascii};
use kore_xhci::device::{EndpointKind, Setup};
use kore_xhci::{Controller, InterruptEvent, Port};

use crate::cdc;
use crate::probe::{EcmPlan, plan_from_config};
use crate::{Error, Result};

/// Receive buffers posted to the bulk IN pipe.
///
/// A design bound. Each is a whole page from the pool, drawn once at bring-up
/// and reused forever — the userland pool is a bump allocator that never
/// frees, so a driver that allocated per frame would exhaust it in seconds.
pub const RX_DEPTH: usize = 16;

/// Transfer events drained in one pass.
const EVENT_BATCH: usize = 16;

/// Bytes requested per bulk IN transfer — a whole frame, since the class
/// carries one Ethernet frame per transfer and terminates it with a short
/// packet.
const RX_TRANSFER: u16 = FRAME_MAX as u16;

/// Cookie band for receive submissions, so a completion says which pipe it
/// came from before anything else is decided.
const RX_COOKIE_TAG: u64 = 1 << 63;

/// Bytes of one class notification this driver reads: the 8-byte header
/// plus the 8-byte speed-change payload, the longest notification it
/// models (`cdc::Notification::parse`). A longer one is truncated to this
/// and parses as whatever its head says; the decoder refuses a short
/// payload rather than inventing a rate.
const NOTIFY_BYTES: usize = cdc::NOTIFY_HEADER_LEN + 8;

/// A brought-up CDC Ethernet device: the addressed USB device, the plan its
/// descriptors yielded, and the receive ring.
///
/// Holds no register handle and no controller — those are borrowed for the
/// duration of a drive cycle by [`EcmNic`], because the service owns them and
/// the same controller also carries other devices.
pub struct EcmDevice<P: DmaPool> {
    usb: kore_xhci::Device<P>,
    plan: EcmPlan,
    mac: MacAddr,
    link: LinkState,
    /// Receive buffers, indexed by cursor modulo [`RX_DEPTH`].
    rx: [Option<P::Buf>; RX_DEPTH],
    /// Frame length landed in each buffer.
    rx_len: [u16; RX_DEPTH],
    /// Transfers posted to the bulk IN pipe.
    posted: u64,
    /// Completions reaped from it.  Bulk transfers on one endpoint complete
    /// in order, so a completion always refers to `reaped % RX_DEPTH` — the
    /// cursor *is* the correlation, and no per-buffer table is needed.
    reaped: u64,
    /// Frames handed to the classifier.
    read: u64,
    /// What the device has been through: drains that found transfer events
    /// waiting, link notifications, and transfers that completed short or
    /// in error. `overruns` stays zero here by the class's nature — a bulk
    /// IN pipe with no buffer posted holds the frame (NAK) rather than
    /// dropping it — and is reported as such rather than invented.
    counters: NicCounters,
}

impl<P: DmaPool> EcmDevice<P> {
    /// The device's own hardware address, decoded from its MAC string.
    pub fn mac(&self) -> MacAddr {
        self.mac
    }

    /// The plan the descriptor walk produced.
    pub fn plan(&self) -> &EcmPlan {
        &self.plan
    }

    /// Current link state, as last reported by a class notification.
    pub fn link(&self) -> LinkState {
        self.link
    }

    /// The device-wide counters, as [`NicBackend::counters`] reports them.
    pub fn counters(&self) -> NicCounters {
        self.counters
    }

    /// The addressed USB device, for a caller that needs its slot.
    pub fn usb(&self) -> &kore_xhci::Device<P> {
        &self.usb
    }

    /// **Bring up** a CDC Ethernet device on `port`.
    ///
    /// Each step is a distinct refusal so a failure on a machine with no
    /// serial console still says which rung it fell off.
    pub fn bring_up<R: Regs64>(
        ctrl: &mut Controller<R, P>,
        regs: &R,
        pool: &P,
        port: &Port,
        clock: fn() -> Option<u64>,
        mut pause: impl FnMut(),
    ) -> Result<Self> {
        let mut usb = ctrl.address_device(regs, pool, port, clock, &mut pause)?;

        // 1. How many configurations are there to walk?
        let n = ctrl.get_descriptor(
            regs,
            &mut usb,
            1,
            0,
            DeviceDescriptor::LEN as u16,
            clock,
            &mut pause,
        )?;
        let device = DeviceDescriptor::parse(usb.read_data(n))
            .ok_or(Error::Probe(crate::probe::ProbeError::Malformed))?;

        // 2. Find the CDC-ECM configuration by DESCRIPTOR, walking the array
        //    by index.  `bConfigurationValue` is a field inside the
        //    descriptor, not the index, and the two are routinely different:
        //    QEMU's `usb-net` puts RNDIS at index 0 with value 2 and CDC-ECM
        //    at index 1 with value 1.  Taking "configuration 1" or "the first
        //    one" selects the wrong function.
        let mut plan = None;
        for index in 0..device.num_configurations {
            let n = ctrl.get_descriptor(
                regs,
                &mut usb,
                2,
                index,
                ConfigDescriptor::LEN as u16,
                clock,
                &mut pause,
            )?;
            let Some(header) = ConfigDescriptor::parse(usb.read_data(n)) else {
                continue;
            };
            let n = ctrl.get_descriptor(
                regs,
                &mut usb,
                2,
                index,
                header.total_length,
                clock,
                &mut pause,
            )?;
            if let Ok(found) = plan_from_config(usb.read_data(n), index) {
                plan = Some(found);
                break;
            }
        }
        let plan = plan.ok_or(Error::Probe(crate::probe::ProbeError::NotEcm))?;

        // 3. Select it, then read the MAC out of its string descriptor.
        ctrl.set_configuration(regs, &mut usb, plan.config_value, clock, &mut pause)?;
        let n = ctrl.get_string_descriptor(
            regs,
            &mut usb,
            plan.mac_string,
            Controller::<R, P>::LANGID_EN_US,
            64,
            clock,
            &mut pause,
        )?;
        let mut ascii = [0u8; 32];
        let len = string_ascii(usb.read_data(n), &mut ascii).ok_or(Error::BadMacString)?;
        let mac = cdc::mac_from_hex(&ascii[..len]).ok_or(Error::BadMacString)?;

        // 4. The data interface's endpoints only exist on its bulk alternate
        //    setting; a device left on altsetting 0 enumerates cleanly and
        //    then cannot move a byte.
        ctrl.set_interface(
            regs,
            &mut usb,
            u16::from(plan.data_interface),
            u16::from(plan.data_altsetting),
            clock,
            &mut pause,
        )?;

        // 5. Notification, bulk IN and bulk OUT in ONE Configure Endpoint —
        //    the slot context's Context Entries must name the highest DCI
        //    across the whole set at once.
        ctrl.configure_endpoints(
            regs,
            pool,
            &mut usb,
            &[plan.notify, plan.bulk_in, plan.bulk_out],
            clock,
            &mut pause,
        )?;

        let mut this = Self {
            usb,
            plan,
            mac,
            // Assumed connected until a class notification says otherwise:
            // the NetworkConnection notification is how an ECM device
            // reports a *change*, and a device that never sends one (QEMU's
            // `usb-net`, and many dongles at plug-in) has a live link, not a
            // dead one. Linux's `cdc_ether` takes the same posture (no
            // `FLAG_LINK_INTR`, so `usbnet_probe` leaves the carrier on).
            // The rate is unknown until a speed-change notification names it.
            link: LinkState::Up { mbps: 0 },
            rx: [const { None }; RX_DEPTH],
            rx_len: [0; RX_DEPTH],
            posted: 0,
            reaped: 0,
            read: 0,
            counters: NicCounters::default(),
        };

        // 6. Draw the receive buffers once.
        for slot in this.rx.iter_mut() {
            *slot = Some(pool.alloc(0x1000).ok_or(Error::DmaAlloc)?);
        }

        // 7. Arm the notification pipe — one interrupt IN transfer standing
        //    on the device's report buffer, routed to the primary
        //    interrupter so the same drain that reaps bulk transfers sees
        //    it. Without this the device has nowhere to put a link change
        //    and the state above would be the last word for the row's life.
        ctrl.arm_interrupt(regs, &mut this.usb, 0)?;

        // 7. The opening receive posture: our own address plus broadcast.
        //    The class publishes no descriptor saying which bits a device
        //    honours, so this is a request, not a guarantee — the frame
        //    service's classifier is what actually decides delivery.
        this.set_filter(ctrl, regs, PacketFilter::DIRECTED, clock, &mut pause)?;

        // 8. Post the receive ring so the first frame is not dropped.
        this.post_rx(ctrl, regs)?;
        Ok(this)
    }

    /// Issue `SetEthernetPacketFilter`.
    fn set_filter<R: Regs64>(
        &mut self,
        ctrl: &mut Controller<R, P>,
        regs: &R,
        filter: PacketFilter,
        clock: fn() -> Option<u64>,
        pause: impl FnMut(),
    ) -> Result<()> {
        let req = Setup {
            request_type: cdc::REQ_CLASS_INTERFACE_OUT,
            request: cdc::SET_ETHERNET_PACKET_FILTER,
            value: cdc::encode_packet_filter(filter),
            index: u16::from(self.plan.control_interface),
        };
        ctrl.control(regs, &mut self.usb, req, 0, clock, pause)?;
        Ok(())
    }

    /// Post receive buffers until the ring is full.
    fn post_rx<R: Regs64>(&mut self, ctrl: &mut Controller<R, P>, regs: &R) -> Result<()> {
        let dci = self
            .usb
            .dci_of(EndpointKind::BulkIn)
            .ok_or(Error::Usb(kore_xhci::Error::Unsupported))?;
        while self.posted.wrapping_sub(self.read) < RX_DEPTH as u64 {
            let index = (self.posted % RX_DEPTH as u64) as usize;
            let Some(buf) = self.rx[index].as_ref() else {
                return Err(Error::Exhausted);
            };
            let iova = buf.iova();
            ctrl.bulk_submit(
                regs,
                &mut self.usb,
                dci,
                iova,
                RX_TRANSFER,
                RX_COOKIE_TAG | self.posted,
                0,
            )?;
            self.posted = self.posted.wrapping_add(1);
        }
        Ok(())
    }

    /// Drain the event ring, routing each transfer event to the pipe it
    /// belongs to.
    ///
    /// The event ring is shared across every endpoint on the controller, so
    /// exactly one place must consume it, and that place is `reap_tx` — the
    /// head of every drive cycle by the class contract. `poll_rx` consumes
    /// what the drain landed and never touches the ring: a sweep that polls
    /// sixty-four frames costs one pass over the event ring, not sixty-five,
    /// and a receive that lands mid-sweep is latched by the held route and
    /// ends the next park rather than being chased per frame.
    ///
    /// One drain that finds anything waiting is one serviced interrupt in
    /// the counters, whatever mix of transfers it carried.
    fn drain<R: Regs64>(
        &mut self,
        ctrl: &mut Controller<R, P>,
        regs: &R,
        out: &mut [TxCompletion],
    ) -> usize {
        let mut events = [InterruptEvent::default(); EVENT_BATCH];
        let mut written = 0;
        let mut serviced = false;
        loop {
            let n = ctrl.drain_interrupts(regs, &mut events);
            if n == 0 {
                if serviced {
                    self.counters.irqs = self.counters.irqs.saturating_add(1);
                }
                return written;
            }
            serviced = true;
            for event in &events[..n] {
                let Some(done) = self.usb.complete_bulk(event) else {
                    // Not a bulk completion for this device — the interrupt
                    // endpoint's notifications land here too.
                    self.absorb_notification(ctrl, regs, event);
                    continue;
                };
                if done.cookie & RX_COOKIE_TAG != 0 {
                    // A receive landed.  Bulk completes in order on one
                    // endpoint, so this is buffer `reaped % RX_DEPTH`.
                    let index = (self.reaped % RX_DEPTH as u64) as usize;
                    self.rx_len[index] = if done.ok() {
                        (done.transferred as u16).min(RX_TRANSFER)
                    } else {
                        self.counters.rx_errors = self.counters.rx_errors.saturating_add(1);
                        0
                    };
                    self.reaped = self.reaped.wrapping_add(1);
                } else if written < out.len() {
                    out[written] = TxCompletion {
                        cookie: kore_driver::net::TxCookie(done.cookie),
                        status: if done.ok() {
                            TxStatus::Sent
                        } else {
                            TxStatus::Failed(done.completion_code)
                        },
                    };
                    written += 1;
                }
            }
            if written == out.len() {
                if serviced {
                    self.counters.irqs = self.counters.irqs.saturating_add(1);
                }
                return written;
            }
        }
    }

    /// Fold a non-bulk transfer event into link state, and re-arm the
    /// notification pipe it completed on.
    ///
    /// Every completion on the pipe consumes the standing transfer, so the
    /// re-arm is unconditional — a notification this driver does not model
    /// must not be the last one the device can deliver.
    fn absorb_notification<R: Regs64>(
        &mut self,
        ctrl: &mut Controller<R, P>,
        regs: &R,
        event: &InterruptEvent,
    ) {
        let Some(notify) = self.usb.endpoint(self.plan.notify.dci()) else {
            return;
        };
        if event.endpoint_id != notify.dci || event.slot_id != self.usb.slot_id {
            return;
        }
        let Some(buf) = notify.data.as_ref() else {
            return;
        };
        let len = (notify.mps as usize)
            .saturating_sub(event.residual as usize)
            .min(NOTIFY_BYTES);
        // Copied out before the re-arm below hands the buffer back to the
        // device: a notification is at most a header and one small payload.
        let mut bytes = [0u8; NOTIFY_BYTES];
        // SAFETY: the endpoint's driver-owned buffer is one page and the
        // device wrote `len ≤ mps` bytes of notification into it.
        bytes[..len]
            .copy_from_slice(unsafe { core::slice::from_raw_parts(buf.cpu() as *const u8, len) });
        // A refused re-arm is structural (`Unsupported`: no such endpoint,
        // no buffer) and bring-up already armed this pipe once, so it
        // cannot happen on a device that got here; if it did, the link
        // would simply stop updating while frames keep moving.
        let _ = ctrl.arm_interrupt(regs, &mut self.usb, 0);
        match cdc::Notification::parse(&bytes[..len]) {
            Some(cdc::Notification::NetworkConnection { connected }) => {
                self.counters.link_changes = self.counters.link_changes.saturating_add(1);
                self.link = if connected {
                    match self.link {
                        LinkState::Up { mbps } => LinkState::Up { mbps },
                        LinkState::Down => LinkState::Up { mbps: 0 },
                    }
                } else {
                    LinkState::Down
                };
            }
            Some(cdc::Notification::SpeedChange { downlink, .. }) => {
                if self.link.is_up() {
                    self.link = LinkState::Up {
                        mbps: downlink / 1_000_000,
                    };
                }
            }
            None => {}
        }
    }
}

/// What a root-port walk concluded about this controller.
///
/// Three outcomes, not two, because "there is no Ethernet adaptor here" and
/// "an Ethernet adaptor is here and would not come up" are different facts
/// that call for different actions — and the walk is the only place that can
/// still tell them apart.
pub enum Discovery<P: DmaPool> {
    /// An adaptor came up, on this root port.  The port number rides along so
    /// a diagnostic can say *where*: on a machine with several controllers it
    /// is the only way to tell which one answered.
    Found(u8, EcmDevice<P>),
    /// A connected port carried a CDC Ethernet function that refused its
    /// bring-up ladder, and where.  The row serves degraded either way, but
    /// the note can now name the rung instead of saying "nothing plugged in"
    /// to someone whose cable is already in the right port.
    Refused(u8, Error),
    /// No connected port carries an Ethernet function at all — the ordinary
    /// state of a machine with no dongle, or with the dongle on the other
    /// controller.  A fact for the readiness note, not a refusal.
    Absent,
}

/// **Find and bring up the first CDC Ethernet function on this controller.**
///
/// Walks the root ports in order and runs the bring-up ladder on each
/// connected one. A port that turns out to carry something else — a keyboard,
/// the stick the machine booted from — is a *conclusion*, not a failure: the
/// walk moves on, so a controller shared with other devices is the ordinary
/// case rather than an error. That is exactly and only
/// [`ProbeError::NotEcm`](crate::ProbeError::NotEcm): the descriptors said
/// this is some other class of device.
///
/// Every other refusal came from a port that *did* present an Ethernet
/// function — a MAC string that will not decode, no bulk pair, a control
/// transfer the controller refused — and is reported as
/// [`Discovery::Refused`]. The **first** such port wins, so the answer is
/// deterministic in port order rather than depending on how many ports follow
/// it.
///
/// `budget` bounds how many ports are *examined* (not how many are walked):
/// bring-up talks to the device, so a hub full of unresponsive ports must not
/// be able to stall founding indefinitely.
pub fn discover<R: Regs64, P: DmaPool>(
    ctrl: &mut Controller<R, P>,
    regs: &R,
    pool: &P,
    budget: u8,
    clock: fn() -> Option<u64>,
    mut pause: impl FnMut(),
) -> Discovery<P> {
    let mut examined = 0u8;
    let mut refusal: Option<(u8, Error)> = None;
    for port in 1..=ctrl.port_count() {
        if examined >= budget {
            break;
        }
        if !ctrl.port(regs, port).connected() {
            continue;
        }
        examined += 1;
        let handle = ctrl.port(regs, port);
        match EcmDevice::bring_up(ctrl, regs, pool, &handle, clock, &mut pause) {
            Ok(dev) => return Discovery::Found(port, dev),
            // Some other class of device on this port.  Keep walking; if
            // nothing else answers, that is `Absent`, not a refusal.
            Err(Error::Probe(crate::ProbeError::NotEcm)) => {}
            Err(e) => {
                if refusal.is_none() {
                    refusal = Some((port, e));
                }
            }
        }
    }
    match refusal {
        Some((port, e)) => Discovery::Refused(port, e),
        None => Discovery::Absent,
    }
}

/// The [`NicBackend`] view — a borrow of the controller, its registers, and
/// one device's ECM state, valid for a single drive cycle.
///
/// The backend is a *view* rather than an owner because the controller is
/// shared: the service owns it and the same one may carry other devices, so
/// a backend that took it would make a second device unreachable.
pub struct EcmNic<'a, R: Regs64, P: DmaPool> {
    ctrl: &'a mut Controller<R, P>,
    regs: &'a R,
    dev: &'a mut EcmDevice<P>,
    /// This runtime's monotonic reading, for the control transfers the
    /// `NicBackend` methods issue. Carried on the borrow because the trait
    /// gives them nowhere else to get one, and a control transfer with no
    /// clock is bounded only in probes.
    clock: fn() -> Option<u64>,
    /// Runtime-selected relaxation for the control transfers issued through
    /// the trait's serving-time methods.
    pause: fn(),
    _r: PhantomData<R>,
}

impl<'a, R: Regs64, P: DmaPool> EcmNic<'a, R, P> {
    /// Borrow a device as a backend for one cycle.
    pub fn new(
        ctrl: &'a mut Controller<R, P>,
        regs: &'a R,
        dev: &'a mut EcmDevice<P>,
        clock: fn() -> Option<u64>,
        pause: fn(),
    ) -> Self {
        Self {
            ctrl,
            regs,
            dev,
            clock,
            pause,
            _r: PhantomData,
        }
    }
}

// `flush_tx` keeps the trait's no-op default: a bulk OUT submit rings the
// endpoint's doorbell per transfer inside `transmit`, so there is nothing
// staged for a later publish. The once-per-drive flush is the tail-pointer
// NIC's affair.
impl<R: Regs64, P: DmaPool> NicBackend for EcmNic<'_, R, P> {
    type Error = Error;

    fn mac(&self) -> MacAddr {
        self.dev.mac
    }

    fn link(&self) -> LinkState {
        self.dev.link
    }

    fn steering(&self) -> Steering {
        // USB networking presents one bulk pipe carrying everything: there is
        // no queue to steer into and no address filter to program per vif.
        // Declaring that honestly is what tells the frame service its own
        // classifier is the isolation boundary.
        Steering::SOFTWARE
    }

    fn counters(&self) -> NicCounters {
        self.dev.counters
    }

    fn set_packet_filter(&mut self, filter: PacketFilter) -> Result<()> {
        self.dev
            .set_filter(self.ctrl, self.regs, filter, self.clock, self.pause)
    }

    fn transmit(&mut self, frame: TxFrame<'_>) -> Result<()> {
        let dci = self
            .dev
            .usb
            .dci_of(EndpointKind::BulkOut)
            .ok_or(Error::Usb(kore_xhci::Error::Unsupported))?;
        // The cookie is the caller's and comes back verbatim; the receive tag
        // is the one bit this driver reserves, and a transmit must not carry
        // it or its completion would be mistaken for a frame arriving.
        let cookie = frame.cookie().0 & !RX_COOKIE_TAG;
        self.ctrl.bulk_submit(
            self.regs,
            &mut self.dev.usb,
            dci,
            IovaAddr(frame.iova().0),
            frame.len(),
            cookie,
            0,
        )?;
        Ok(())
    }

    fn reap_tx(&mut self, out: &mut [TxCompletion]) -> usize {
        self.dev.drain(self.ctrl, self.regs, out)
    }

    fn poll_rx(&mut self, f: &mut dyn FnMut(&[u8])) -> bool {
        // The event ring was drained at the head of this sweep (`reap_tx`,
        // the class contract); this consumes what that drain landed and
        // touches the controller not at all.
        if self.dev.read >= self.dev.reaped {
            return false;
        }
        let index = (self.dev.read % RX_DEPTH as u64) as usize;
        let len = self.dev.rx_len[index] as usize;
        self.dev.read = self.dev.read.wrapping_add(1);
        if len == 0 {
            // A failed or empty transfer occupies a slot but carries no
            // frame; consuming it frees the buffer without inventing bytes.
            return true;
        }
        if let Some(buf) = self.dev.rx[index].as_ref() {
            // SAFETY: the buffer is a whole page the controller filled with
            // `len ≤ RX_TRANSFER` bytes; the borrow ends with the callback.
            let bytes = unsafe { core::slice::from_raw_parts(buf.cpu() as *const u8, len) };
            f(bytes);
        }
        true
    }

    fn refill_rx(&mut self) -> Result<()> {
        self.dev.post_rx(self.ctrl, self.regs)
    }
}

#[cfg(test)]
mod tests {
    use kore_driver::net::backend::TxCookie;
    use kore_driver::net::window::tx_slot_at;
    use kore_driver::net::{EtherType, FrameHead};
    use kore_xhci::mock::{ECM_MAC, MockPool, MockXhciDevice};

    use super::*;

    fn spin() {}

    /// Bring the mock's CDC-ECM device all the way up through the real
    /// ladder — configuration selection, MAC read, altsetting, endpoints,
    /// packet filter, receive ring.
    fn brought_up() -> (
        MockXhciDevice,
        MockPool,
        Controller<MockXhciDevice, MockPool>,
        EcmDevice<MockPool>,
    ) {
        let dev = MockXhciDevice::new_with_ecm();
        let pool = MockPool;
        let mut ctrl = Controller::bring_up(&dev, &pool, || None, spin).expect("controller");
        let port = ctrl.port(&dev, 3);
        let ecm =
            EcmDevice::bring_up(&mut ctrl, &dev, &pool, &port, || None, spin).expect("bring-up");
        (dev, pool, ctrl, ecm)
    }

    /// The port walk finds the adaptor wherever it is plugged in, and says
    /// where — the row's readiness note prints that number, and on a machine
    /// with several controllers it is the only way to tell which one answered.
    #[test]
    fn the_walk_finds_the_adaptor_and_names_its_port() {
        let dev = MockXhciDevice::new_with_ecm();
        let pool = MockPool;
        let mut ctrl = Controller::bring_up(&dev, &pool, || None, spin).expect("controller");
        let super::Discovery::Found(port, ecm) =
            super::discover(&mut ctrl, &dev, &pool, 8, || None, spin)
        else {
            panic!("the adaptor")
        };
        assert_eq!(port, 3, "the mock hangs its ECM device off root port 3");
        assert_eq!(ecm.mac(), kore_driver::net::MacAddr(ECM_MAC));
    }

    /// A controller with no adaptor is the ordinary state of a machine that
    /// has none plugged in: the walk concludes, and concluding is not an
    /// error — the row founds degraded on it rather than taking the boot down.
    #[test]
    fn a_controller_without_an_adaptor_concludes_rather_than_failing() {
        let dev = MockXhciDevice::new();
        let pool = MockPool;
        let mut ctrl = Controller::bring_up(&dev, &pool, || None, spin).expect("controller");
        assert!(matches!(
            super::discover(&mut ctrl, &dev, &pool, 8, || None, spin),
            super::Discovery::Absent,
        ));
    }

    /// **An adaptor that will not come up is not an absent adaptor.**
    ///
    /// A port carrying a real CDC Ethernet function that refuses partway up
    /// the ladder used to be indistinguishable from an empty port: the walk
    /// dropped the error and returned `None`, the row noted `NO_ADAPTOR`, and
    /// the only advice a serial-less machine could give was "move the cable"
    /// — to someone whose cable was already in the right port.  The walk now
    /// says which port and which rung, and the row's note carries it.
    #[test]
    fn an_adaptor_that_refuses_bring_up_is_reported_as_a_refusal() {
        let dev = MockXhciDevice::new_with_ecm_bad_mac();
        let pool = MockPool;
        let mut ctrl = Controller::bring_up(&dev, &pool, || None, spin).expect("controller");
        let super::Discovery::Refused(port, error) =
            super::discover(&mut ctrl, &dev, &pool, 8, || None, spin)
        else {
            panic!("a connected ECM function that refuses must not read as Absent")
        };
        assert_eq!(port, 3, "the refusing device is on the mock's ECM port");
        assert!(
            matches!(error, crate::Error::BadMacString),
            "the ladder refused at the MAC string, and the note must say so",
        );
    }

    /// The examination budget bounds founding: a hub of unresponsive ports
    /// cannot stall the boot chain behind this row.
    #[test]
    fn the_walk_stops_at_its_examination_budget() {
        let dev = MockXhciDevice::new_with_ecm();
        let pool = MockPool;
        let mut ctrl = Controller::bring_up(&dev, &pool, || None, spin).expect("controller");
        assert!(
            matches!(
                super::discover(&mut ctrl, &dev, &pool, 0, || None, spin),
                super::Discovery::Absent,
            ),
            "a zero budget examines nothing"
        );
    }

    /// The whole ladder, and specifically the configuration trap: the device
    /// offers RNDIS at index 0 (value 2) and CDC-ECM at index 1 (value 1), so
    /// a driver that takes the first configuration, or assumes value 1 lives
    /// at index 0, drives the wrong function.
    #[test]
    fn bring_up_selects_the_ecm_configuration_and_reads_its_mac() {
        let (dev, _pool, _ctrl, ecm) = brought_up();

        assert_eq!(ecm.plan().config_index, 1, "ECM is the second descriptor");
        assert_eq!(ecm.plan().config_value, 1, "but its value is 1");
        assert_eq!(ecm.mac(), kore_driver::net::MacAddr(ECM_MAC));
        assert_eq!(ecm.plan().max_segment, 1514);
        assert_eq!(ecm.plan().data_altsetting, 1);

        // The device was really told to select that configuration and that
        // alternate setting — not merely asked in a call that returned Ok.
        let seen = dev.class_requests();
        assert!(
            seen.iter()
                .any(|(req, value, _)| *req == cdc::SET_ETHERNET_PACKET_FILTER
                    && *value == cdc::encode_packet_filter(PacketFilter::DIRECTED)),
            "the opening packet filter reached the wire: {seen:?}"
        );

        // All three endpoints came up in one Configure Endpoint.
        assert!(ecm.usb().dci_of(EndpointKind::BulkIn).is_some());
        assert!(ecm.usb().dci_of(EndpointKind::BulkOut).is_some());
        assert!(ecm.usb().int_ep().is_some(), "the notification endpoint");
    }

    /// A frame written through `NicBackend::transmit` reaches the wire, and
    /// the completion comes back carrying the caller's own cookie.
    #[test]
    fn transmit_reaches_the_wire_and_returns_the_cookie() {
        let (dev, pool, mut ctrl, mut ecm) = brought_up();

        let buf = pool.alloc(0x1000).expect("frame buffer");
        let frame = b"\xff\xff\xff\xff\xff\xff\x40\x01\x02\x03\x04\x05\x88\xb5hello";
        // The frame goes in transmit slot 0, which sits at its offset inside
        // the vif window — not at the window base. Writing at the base is
        // exactly the mistake the mapping's offset math exists to prevent.
        let at = tx_slot_at(0).expect("slot 0");
        // SAFETY: a whole leaked page backs `buf`; slot 0 and the frame both
        // fit inside it.
        unsafe { core::ptr::copy_nonoverlapping(frame.as_ptr(), buf.cpu().add(at), frame.len()) };

        let mut mapping = kore_driver::net::VifMapping::new((), buf.iova());
        {
            let mut nic = EcmNic::new(&mut ctrl, &dev, &mut ecm, || None, || {});
            // Slot 0 of the mapping is the buffer's own base.
            mapping
                .transmit(&mut nic, 0, frame.len() as u16, TxCookie(0xBEEF))
                .expect("transmit");
        }
        assert_eq!(mapping.outstanding(), 1);

        let mut done = [TxCompletion {
            cookie: TxCookie(0),
            status: TxStatus::Sent,
        }; 4];
        let n = {
            let mut nic = EcmNic::new(&mut ctrl, &dev, &mut ecm, || None, || {});
            nic.reap_tx(&mut done)
        };
        assert_eq!(n, 1);
        assert_eq!(done[0].cookie, TxCookie(0xBEEF));
        assert_eq!(done[0].status, TxStatus::Sent);

        mapping.completed(n as u32);
        assert!(
            mapping.retire().is_ok(),
            "quiesced, so the lease comes back"
        );

        let sent = dev.transmitted();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].as_slice(), frame);
    }

    /// The full round trip over a looped-back wire: a transmitted frame
    /// comes back through the receive ring, byte for byte, with its header
    /// intact. This is the shape the boot self-test asserts.
    #[test]
    fn a_frame_round_trips_through_the_backend() {
        let (dev, pool, mut ctrl, mut ecm) = brought_up();
        dev.set_loopback(true);

        let buf = pool.alloc(0x1000).expect("frame buffer");
        let frame = b"\xff\xff\xff\xff\xff\xff\x40\x01\x02\x03\x04\x05\x88\xb5payload here";
        let at = tx_slot_at(0).expect("slot 0");
        // SAFETY: a whole leaked page backs `buf`; slot 0 and the frame both
        // fit inside it.
        unsafe { core::ptr::copy_nonoverlapping(frame.as_ptr(), buf.cpu().add(at), frame.len()) };

        let mut mapping = kore_driver::net::VifMapping::new((), buf.iova());
        let mut nic = EcmNic::new(&mut ctrl, &dev, &mut ecm, || None, || {});
        mapping
            .transmit(&mut nic, 0, frame.len() as u16, TxCookie(1))
            .expect("transmit");

        let mut received = alloc::vec::Vec::new();
        let mut delivered = 0;
        let mut done = [TxCompletion {
            cookie: TxCookie(0),
            status: TxStatus::Sent,
        }; 4];
        // The engine's sweep: reap first — the one drain of the shared event
        // ring — then poll what it landed. The transmit completion and the
        // receive completion arrive in the same batch.
        for _ in 0..4 {
            let _ = nic.reap_tx(&mut done);
            while nic.poll_rx(&mut |bytes| {
                received.extend_from_slice(bytes);
                delivered += 1;
            }) {}
            if delivered > 0 {
                break;
            }
        }

        assert_eq!(delivered, 1, "exactly one frame came back");
        assert_eq!(received.as_slice(), frame, "byte for byte");

        let head = FrameHead::parse(&received).expect("a parseable header");
        assert_eq!(head.ethertype, EtherType::KORE_PEER);
        assert_eq!(head.src, kore_driver::net::MacAddr(ECM_MAC));
        assert!(head.dst.is_broadcast());
    }

    /// **One drain per sweep.** The shared event ring is consumed at the
    /// head of the sweep and nowhere else: a frame that lands after the
    /// reap is not chased by `poll_rx` — it waits for the next sweep, which
    /// the held route wakes — and a sweep of many polls costs one pass over
    /// the ring. The counter says so: one serviced interrupt per drain that
    /// found something, however many frames it carried.
    #[test]
    fn poll_rx_consumes_only_what_the_sweeps_one_drain_landed() {
        let (dev, _pool, mut ctrl, mut ecm) = brought_up();
        let slot = dev.last_slot();
        let frame = b"\xff\xff\xff\xff\xff\xff\x40\x01\x02\x03\x04\x05\x88\xb5one";
        let mut nic = EcmNic::new(&mut ctrl, &dev, &mut ecm, || None, || {});
        let mut done = [TxCompletion {
            cookie: TxCookie(0),
            status: TxStatus::Sent,
        }; 4];
        assert_eq!(nic.reap_tx(&mut done), 0);
        assert_eq!(nic.counters().irqs, 0, "an idle drain is not an interrupt");

        // Landed after the sweep's drain: invisible to this sweep's polls.
        dev.inject_frame(slot, frame);
        dev.inject_frame(slot, frame);
        assert!(!nic.poll_rx(&mut |_| panic!("poll_rx drained the ring")));
        assert!(!nic.poll_rx(&mut |_| panic!("still not")));

        // The next sweep's reap lands both, in one serviced interrupt.
        assert_eq!(
            nic.reap_tx(&mut done),
            0,
            "receives are not transmit completions"
        );
        let mut polled = 0;
        while nic.poll_rx(&mut |bytes| {
            assert_eq!(bytes, frame);
            polled += 1;
        }) {}
        assert_eq!(polled, 2);
        assert_eq!(
            nic.counters().irqs,
            1,
            "two frames, one drain, one interrupt"
        );
        assert_eq!(nic.counters().rx_errors, 0);
        assert_eq!(
            nic.counters().overruns,
            0,
            "a bulk pipe backpressures, never drops"
        );
    }

    /// A class link notification is an edge the counters keep and the state
    /// the backend reports from then on, both from the same one drain.
    #[test]
    fn link_notifications_move_the_state_and_count_the_change() {
        let (dev, _pool, mut ctrl, mut ecm) = brought_up();
        let slot = dev.last_slot();
        let mut nic = EcmNic::new(&mut ctrl, &dev, &mut ecm, || None, || {});
        let mut done = [TxCompletion {
            cookie: TxCookie(0),
            status: TxStatus::Sent,
        }; 1];
        assert_eq!(
            nic.link(),
            LinkState::Up { mbps: 0 },
            "assumed connected, rate unknown, until the device says otherwise"
        );
        let before = nic.counters().link_changes;
        let down = [
            cdc::NOTIFY_REQUEST_TYPE,
            cdc::notification::NETWORK_CONNECTION,
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        dev.inject_report(slot, &down);
        assert_eq!(
            nic.reap_tx(&mut done),
            0,
            "a notification is not a completion"
        );
        assert_eq!(nic.link(), LinkState::Down);
        assert_eq!(nic.counters().link_changes, before + 1);
        let up = [
            cdc::NOTIFY_REQUEST_TYPE,
            cdc::notification::NETWORK_CONNECTION,
            1,
            0,
            0,
            0,
            0,
            0,
        ];
        dev.inject_report(slot, &up);
        let _ = nic.reap_tx(&mut done);
        assert!(nic.link().is_up());
        assert_eq!(nic.counters().link_changes, before + 2);
        // The pipe was re-armed after each: a third notification still
        // lands, so a flapping cable is never silenced after its first edge.
        dev.inject_report(slot, &down);
        let _ = nic.reap_tx(&mut done);
        assert_eq!(nic.link(), LinkState::Down);
        assert_eq!(nic.counters().link_changes, before + 3);
    }

    /// A backend that cannot steer must say so, because that is what tells
    /// the frame service its own classifier is the isolation boundary.
    #[test]
    fn the_backend_declares_no_hardware_steering() {
        let (dev, _pool, mut ctrl, mut ecm) = brought_up();
        let nic = EcmNic::new(&mut ctrl, &dev, &mut ecm, || None, || {});
        assert!(!nic.steering().separates_vifs());
        assert_eq!(nic.steering().rx_queues, 1);
        assert!(!nic.steering().by_address);
    }

    /// A packet-filter change really reaches the device.
    #[test]
    fn setting_the_packet_filter_reaches_the_device() {
        let (dev, _pool, mut ctrl, mut ecm) = brought_up();
        {
            let mut nic = EcmNic::new(&mut ctrl, &dev, &mut ecm, || None, || {});
            nic.set_packet_filter(PacketFilter::PROMISCUOUS)
                .expect("promiscuous");
        }
        let want = cdc::encode_packet_filter(PacketFilter::PROMISCUOUS);
        assert!(
            dev.class_requests()
                .iter()
                .any(|(req, value, _)| *req == cdc::SET_ETHERNET_PACKET_FILTER && *value == want),
            "the device was told to go promiscuous"
        );
    }
}
