//! `serial` — the headless image's serial line, as an input device that
//! also shows text (`res/IMAGE_PROFILES.md` §4.3).
//!
//! The founder gives every row one protocol for all the pairs it serves, and
//! the terminal (input-stream) speaks the input-device protocol to its
//! provider.  So the serial row speaks that protocol to both of its clients:
//!
//! - **pair 0** — the founding pair, whose client half the founder hands the
//!   terminal (the headless image's `client_of` override).  The terminal
//!   subscribes a report queue; received bytes arrive there as
//!   [`KIND_TEXT`](InputReport::KIND_TEXT) reports, at most
//!   [`TEXT_BYTES`](InputReport::TEXT_BYTES) each, and the host wakes the
//!   terminal after every batch.
//! - **pair 1** — the console's (`clients`), the line's transmit side.  The
//!   console sends what the shell writes and the terminal echoes as
//!   [`Transmit`](InputVerb::Transmit) requests; each reply says how many
//!   bytes the UART took, so a stalled line is backpressure the console
//!   retries, never a silent drop.  (Inline, because the founder stages a
//!   declared extra pair without a return post, so no queue can be
//!   delivered; ruled 2026-11-19.)
//!
//! The line is a firmware-described COM port ([`COM_PORT`], ruled
//! 2026-11-19): the ACPI `PNP0501` the kernel files as a platform row,
//! never the one it logs to.  Its registers are the eight I/O ports from the
//! base its `_CRS` declares, reached through this task's own port grant.
//! The selector asks for no line to park on, so the receive side is polled on a bounded cadence through the host's
//! deadline drive: [`RX_ACTIVE_NS`] while bytes are arriving,
//! [`RX_IDLE_NS`] once the line goes quiet.  Each pass reads at most
//! [`RX_BURST`] bytes and stops early when the terminal's queue is full,
//! leaving the rest in the UART for the next pass.

use kore_abi::svc::{InputReport, InputReqWire, InputRepWire};
use kore_rt::service::{Serve, SystemService};
use kore_svc_input::{
    InputRefusal, InputRequest, InputResponse, InputVerb, RelayBooks, SubscriptionId,
};

use crate::service::{Backend, UartBackend};

/// The serial row's endowment: nothing.  A driver row claims its hardware
/// from the device broker at runtime (`res/DEVICE_SERVICE.md` §10), and a
/// polled 16550 needs no DMA pool.
///
/// The selector it claims with: a firmware-described 16550 COM port
/// (`PNP0500`/`PNP0501`, filed under
/// [`HID_SERIAL_16550`](kore_abi::device::HID_SERIAL_16550)), with no line
/// demanded — polled, so a port whose `_CRS` names no interrupt still
/// serves.  Required: a row with no line carries no byte.
pub const COM_PORT: &[kore_rt::service::DeviceRequest] = &[kore_rt::service::DeviceRequest {
    selector: kore_rt::service::DeviceSelector::Platform {
        hid: kore_abi::device::HID_SERIAL_16550,
        irq: false,
        dma: false,
    },
    required: true,
}];

#[derive(Debug, kore_rt::Bundle)]
#[bundle(name = "serial")]
pub struct SerialEndowment;

/// Startup rungs, each a distinct NACK the founder prints.
pub mod rung {
    /// The device-broker faces did not resolve from the boot block.
    pub const DEVICE_FACE: u64 = 1;
    /// No COM port matched (none described, or only the kernel's own), or
    /// the broker refused the claim.
    pub const NO_LINE: u64 = 2;
    /// A claimed device was not a platform device, or its record named no
    /// port base.
    pub const NOT_PORTS: u64 = 3;
    /// The port grant was refused.
    pub const PORTS: u64 = 4;
    /// The serve loop ended (it never should).
    pub const SERVE_ENDED: u64 = 5;
}

/// **The serial row's entry** (`kore-svc-output`'s image): claim the COM
/// port through the device broker, then serve the input-device protocol on
/// both founded pairs (`kore_rt::entry::serve_forever` stands the return
/// posts, beacons, and enters the deadline drive).
#[cfg(target_os = "none")]
pub async fn run(rt: kore_rt::entry::Rt) -> ! {
    use kore_rt::entry::refuse_with;
    use kore_rt::svc::SvcProtocol;
    use kore_svc_device::client::{Claimed, ClaimedDevice, FaceLayout, MultiBroker};

    let boot = rt.bootstrap();
    let transport = kore_rt::entry::transport();
    let Some(mut broker) = MultiBroker::open(
        transport,
        boot,
        FaceLayout::for_bootstrap(
            boot,
            <SerialService as SvcProtocol>::SERVE_PAIRS,
            <SerialService as SvcProtocol>::BOOT_REPLY_GRANTS,
        ),
    ) else {
        refuse_with(&rt, rung::DEVICE_FACE);
    };
    let (token, base) = match broker.claim_first(COM_PORT[0].selector).await {
        Ok(Claimed {
            device: ClaimedDevice::Platform(token),
            diag,
            ..
        }) => (token, (diag & kore_abi::device::DeviceRecord::PLATFORM_PORT_BASE_MASK) as u16),
        Ok(_) => refuse_with(&rt, rung::NOT_PORTS),
        Err(_) => refuse_with(&rt, rung::NO_LINE),
    };
    if base == 0 {
        refuse_with(&rt, rung::NOT_PORTS);
    }
    // The grant is taken HERE, by the task that serves the line: the
    // kernel admits port I/O from the running task's own holdings.
    let line = UartBackend::adopt_ports(token, base);
    if !line.online() {
        refuse_with(&rt, rung::PORTS);
    }
    kore_rt::entry::serve_forever(&rt, SerialDevice::with_line(line), None, 0).await;
    refuse_with(&rt, rung::SERVE_ENDED)
}

/// The serial row as the manifest cites it.
pub type SerialService = SerialDevice<UartBackend>;

/// The founding pair: the terminal's, where reports go and where focus and
/// indicators are asked for.
pub const REPORT_PAIR: u16 = 0;
/// The console's pair: the transmit side — the first declared link after
/// the row's own [`SERVE_PAIRS`](kore_rt::svc::SvcProtocol::SERVE_PAIRS)
/// (`founder::SERVICE_CLIENT_LINKS`), so it is pair 1 while the row serves
/// one pair of its own.
pub const TRANSMIT_PAIR: u16 = 1;
const _: () = assert!(
    TRANSMIT_PAIR as usize
        == <SerialDevice<UartBackend> as kore_rt::svc::SvcProtocol>::SERVE_PAIRS,
    "the console's declared link lands right after the row's own pairs"
);
/// The source identity this line's reports carry: its own band, apart from
/// the router's xHCI ports (1-based) and legacy keyboard (`0x40`).
pub const SERIAL_SOURCE: u8 = 0x60;
/// Receive poll interval while bytes are arriving: a typed line echoes
/// within a millisecond of its last byte.
pub const RX_ACTIVE_NS: u64 = 1_000_000;
/// Receive poll interval once the line is quiet: an idle console costs one
/// register read every 10 ms.
pub const RX_IDLE_NS: u64 = 10_000_000;
/// The most bytes one pass drains — the deadline drive's bounded unit of
/// work.  A 16550's FIFO is 16 bytes; 256 covers a pasted command at an
/// active cadence without letting a flooding host hold the task.
pub const RX_BURST: usize = 256;

/// A byte line both ways: the transmit seam of [`Backend`] plus a receive
/// side.
pub trait SerialLine: Backend {
    /// The next received byte, if the line holds one.
    fn get(&mut self) -> Option<u8>;
}

impl SerialLine for UartBackend {
    fn get(&mut self) -> Option<u8> {
        UartBackend::get(self)
    }
}

/// The line's counters (the `Status` reply's books).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SerialBooks {
    /// Bytes received and reported.
    pub received: u64,
    /// Bytes the UART accepted for transmission.
    pub transmitted: u64,
    /// Passes that stopped early because the terminal's queue was full.
    pub report_backpressure: u64,
}

/// The serial row: one line, one subscriber, one transmit client.
pub struct SerialDevice<L> {
    line: L,
    reports: Option<kore_ring::spsc::Producer<'static, InputReport>>,
    subscription: Option<SubscriptionId>,
    /// The pair the request being served arrived on (stamped by the host in
    /// [`request_ready`](SystemService::request_ready)).
    via: u16,
    /// When the next receive pass is due (absolute monotonic ns).
    next_poll_ns: u64,
    books: SerialBooks,
}

impl<L: SerialLine> SerialDevice<L> {
    /// A device over `line`, its first receive pass due at once.
    pub fn with_line(line: L) -> Self {
        Self {
            line,
            reports: None,
            subscription: None,
            via: REPORT_PAIR,
            next_poll_ns: 0,
            books: SerialBooks::default(),
        }
    }

    /// When the next receive pass is due (absolute monotonic ns).
    pub fn next_poll_ns(&self) -> u64 {
        self.next_poll_ns
    }

    /// The counters.
    pub fn books(&self) -> SerialBooks {
        self.books
    }

    /// The line (tests read its recorded transmit side).
    pub fn line(&self) -> &L {
        &self.line
    }

    /// Attach the terminal's report queue (the metal `Subscribe` tail, and
    /// the host-test entry).
    pub fn attach_reports(
        &mut self,
        producer: kore_ring::spsc::Producer<'static, InputReport>,
    ) -> InputResponse {
        if self.reports.is_some() {
            return InputResponse::Refused(InputRefusal::TableFull);
        }
        self.reports = Some(producer);
        let id = SubscriptionId(1);
        self.subscription = Some(id);
        InputResponse::Subscribed(id)
    }

    /// One receive pass: read up to [`RX_BURST`] bytes into text reports.
    /// Answers whether any report was published (the host then wakes the
    /// terminal) and schedules the next pass from `now_ns`.
    pub fn receive(&mut self, now_ns: u64) -> bool {
        let mut published = false;
        let mut taken = 0;
        if let Some(reports) = self.reports.as_mut() {
            let mut run = [0u8; InputReport::TEXT_BYTES];
            let mut len = 0;
            while taken < RX_BURST {
                // Leave bytes in the UART rather than read what cannot be
                // reported: a full queue is backpressure, not loss.
                if len == 0 && reports.capacity_left() == 0 {
                    self.books.report_backpressure += 1;
                    break;
                }
                let Some(byte) = self.line.get() else { break };
                taken += 1;
                run[len] = byte;
                len += 1;
                if len == run.len() {
                    published |= Self::publish(reports, &run[..len]);
                    len = 0;
                }
            }
            if len != 0 {
                published |= Self::publish(reports, &run[..len]);
            }
        }
        self.books.received += taken as u64;
        let interval = if taken != 0 { RX_ACTIVE_NS } else { RX_IDLE_NS };
        self.next_poll_ns = now_ns.saturating_add(interval);
        published
    }

    fn publish(
        reports: &mut kore_ring::spsc::Producer<'static, InputReport>,
        run: &[u8],
    ) -> bool {
        // `run` is 1..=TEXT_BYTES by construction, and the free-slot check
        // above held before the first byte of it was read.
        InputReport::text(SERIAL_SOURCE, run).is_some_and(|report| reports.push(report).is_ok())
    }

    /// Transmit the live bytes of a `Transmit`, in order, up to the first
    /// the UART refuses.
    fn transmit(&mut self, bytes: &[u8]) -> u8 {
        let mut accepted = 0u8;
        for &byte in bytes {
            if !self.line.put(byte) {
                break;
            }
            accepted += 1;
        }
        self.line.flush();
        self.books.transmitted += u64::from(accepted);
        accepted
    }

    /// The step body for one request on the stamped pair.
    pub fn dispatch(&mut self, req: InputRequest) -> InputResponse {
        let terminal = self.via == REPORT_PAIR;
        match req.verb {
            InputVerb::Subscribe { queue } if terminal => self.subscribe(queue),
            InputVerb::Unsubscribe { subscription } if terminal => {
                if self.subscription == Some(subscription) {
                    self.reports = None;
                    self.subscription = None;
                    InputResponse::Unsubscribed
                } else {
                    InputResponse::Refused(InputRefusal::Unknown)
                }
            }
            // The terminal owns focus and lock state; a line has one
            // subscriber and no indicators, so both are recorded and inert.
            InputVerb::SetFocus { .. } if terminal => InputResponse::Focused,
            InputVerb::SetIndicators { .. } if terminal => InputResponse::Indicated,
            InputVerb::Transmit { bytes, len } if self.via == TRANSMIT_PAIR => {
                let len = usize::from(len).min(InputReqWire::TRANSMIT_BYTES);
                InputResponse::Transmitted {
                    accepted: self.transmit(&bytes[..len]),
                }
            }
            InputVerb::Status => InputResponse::Status(RelayBooks {
                events: u32::try_from(self.books.received).unwrap_or(u32::MAX),
                dropped: u32::try_from(self.books.report_backpressure).unwrap_or(u32::MAX),
                ..RelayBooks::default()
            }),
            InputVerb::Diagnostic { .. } => InputResponse::Refused(InputRefusal::Unknown),
            // Every remaining case is a verb on the wrong pair.
            _ => InputResponse::Refused(InputRefusal::Denied),
        }
    }

    #[cfg(target_os = "none")]
    fn subscribe(&mut self, queue: kore_abi::session::SlotRef) -> InputResponse {
        use kore_rt::cap::verbs;
        let Some(view) = verbs::take_view(queue) else {
            return InputResponse::Refused(InputRefusal::BadQueue);
        };
        let Some(window) = verbs::map_view(view) else {
            return InputResponse::Refused(InputRefusal::BadQueue);
        };
        // The delivered name stays in the fixed landing.  The terminal sends
        // its queue ACCESS|MODIFY (its handshake with the HID router, whose
        // bespoke host never closes the name), so this row cannot close it
        // the way the output row's attach does; the generic host's retire of
        // the landing is refused, it defers that cleanup and still replies.
        // One subscription per boot, as with the router, so the landing is
        // never needed again.
        // SAFETY: the delivered writable view is now this service's producer
        // half of the terminal's report ring, mapped for the task's life; the
        // terminal touches it only as the ring's consumer.
        let bytes: &'static mut [u8] = unsafe { &mut *window.as_ptr() };
        match kore_ring::spsc::producer::<InputReport>(bytes) {
            Ok(producer) => self.attach_reports(producer),
            Err(_) => InputResponse::Refused(InputRefusal::BadQueue),
        }
    }

    /// Off the userland target there is no view to map (tests attach a
    /// producer directly through [`attach_reports`](Self::attach_reports)).
    #[cfg(not(target_os = "none"))]
    fn subscribe(&mut self, _queue: kore_abi::session::SlotRef) -> InputResponse {
        InputResponse::Refused(InputRefusal::BadQueue)
    }
}

/// The input-device protocol, served on both pairs — the terminal's client
/// half is founded against exactly this wire, so it is the router's,
/// delegated whole.
impl<L> kore_rt::svc::SvcProtocol for SerialDevice<L> {
    type Request = InputRequest;
    type Response = InputResponse;
    type ReqWire = InputReqWire;
    type RepWire = InputRepWire;
    /// Data only, as the router: and the terminal derives its return landing
    /// from the router's zero, so it must stay zero.
    const REPLY_GRANTS: usize = 0;
    /// The row's own pair: the terminal's.  The console's is not counted
    /// here: it is a declared link (`clients` in the console's fragment),
    /// which the founder adds after the row's own pairs — so it lands at
    /// pair 1, [`TRANSMIT_PAIR`].  Counting it here too stamped a pair 1
    /// with no client and pushed the console's to pair 2.
    const SERVE_PAIRS: usize = 1;

    fn encode_req(req: &Self::Request) -> Self::ReqWire {
        kore_svc_input::wire::encode_req(req)
    }
    fn decode_req(w: &Self::ReqWire) -> Option<Self::Request> {
        kore_svc_input::wire::decode_req(w)
    }
    fn encode_rep(rep: &Self::Response, grants: u8) -> Self::RepWire {
        kore_svc_input::wire::encode_rep(rep, grants)
    }
    fn decode_rep(w: &Self::RepWire) -> Option<(Self::Response, u8)> {
        kore_svc_input::wire::decode_rep(w)
    }
}

const _: () = assert!(
    <SerialDevice<UartBackend> as kore_rt::svc::SvcProtocol>::REPLY_GRANTS
        == <kore_svc_input::InputService as kore_rt::svc::SvcProtocol>::REPLY_GRANTS,
    "the terminal derives its return landing from the router's grants"
);

impl SystemService for SerialDevice<UartBackend> {
    const NAME: &'static str = "serial";
    type Endowment = SerialEndowment;

    /// The line: a firmware-described 16550 COM port, polled.
    const DEVICES: &'static [kore_rt::service::DeviceRequest] = COM_PORT;
    /// A polled 16550: no DMA, no route.  Nothing to stop on the way down.
    const DEVICE_CLASSES: &'static [kore_rt::service::DeviceClass] =
        &[kore_rt::service::DeviceClass::NoDma];

    /// The line comes from the broker claim in [`run`], not the endowment;
    /// a row bound without one stays offline and fails its start.
    fn bind(_endowment: Self::Endowment) -> Self {
        Self::with_line(UartBackend::offline())
    }

    /// A line whose registers never materialized must fail closed, never
    /// beacon.
    fn start_on_metal(
        &mut self,
        _rt: &kore_rt::entry::Rt,
    ) -> Result<(), kore_rt::service::StartFailure> {
        if !self.line.online() {
            return Err(kore_rt::service::StartFailure::UNSPECIFIED);
        }
        Ok(())
    }

    fn request_ready(&mut self, _request: &Self::Request, via: u16) -> bool {
        self.via = via;
        true
    }

    fn next_deadline(&self) -> Option<u64> {
        Some(self.next_poll_ns())
    }

    fn drive_deadlines(&mut self, now_ns: u64) -> u16 {
        if self.receive(now_ns) { 1 << REPORT_PAIR } else { 0 }
    }

    fn drive_host<T: kore_rt::transport::Transport>(
        host: kore_rt::svc::ServeHost<'_, '_, T, Self>,
    ) -> impl core::future::Future<Output = core::convert::Infallible> {
        async move { host.run_deadline_driven().await }
    }

    fn serve(&mut self, req: Self::Request, sv: &mut Serve<'_, Self::Response>) {
        let rep = self.dispatch(req);
        sv.reply(rep);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kore_svc_input::Endpoint;

    /// A scripted line: `rx` is replayed, `tx` records what was sent, and
    /// `tx_cap` bytes are accepted before the line stalls.
    struct ScriptLine {
        rx: std::vec::Vec<u8>,
        at: usize,
        tx: std::vec::Vec<u8>,
        tx_cap: usize,
    }

    impl Backend for ScriptLine {
        fn put(&mut self, byte: u8) -> bool {
            if self.tx.len() < self.tx_cap {
                self.tx.push(byte);
                true
            } else {
                false
            }
        }
    }

    impl SerialLine for ScriptLine {
        fn get(&mut self) -> Option<u8> {
            let byte = self.rx.get(self.at).copied();
            self.at += usize::from(byte.is_some());
            byte
        }
    }

    fn line(rx: &[u8]) -> ScriptLine {
        ScriptLine {
            rx: rx.to_vec(),
            at: 0,
            tx: std::vec::Vec::new(),
            tx_cap: usize::MAX,
        }
    }

    fn queue(
        capacity: u32,
    ) -> (
        kore_ring::spsc::Producer<'static, InputReport>,
        kore_ring::spsc::Consumer<'static, InputReport>,
    ) {
        let layout = std::alloc::Layout::from_size_align(8192, 64).unwrap();
        // SAFETY: a fresh, leaked, ring-aligned window for the test's life.
        let window: &'static mut [u8] = unsafe {
            let p = std::alloc::alloc_zeroed(layout);
            assert!(!p.is_null());
            core::slice::from_raw_parts_mut(p, 8192)
        };
        kore_ring::spsc::split::<InputReport>(window, capacity).expect("split")
    }

    fn texts(consumer: &mut kore_ring::spsc::Consumer<'_, InputReport>) -> std::vec::Vec<u8> {
        let mut out = std::vec::Vec::new();
        while let Some(report) = consumer.pop() {
            assert_eq!(report.source, SERIAL_SOURCE);
            let (bytes, len) = report.text_bytes().expect("a text report");
            out.extend_from_slice(&bytes[..len]);
        }
        out
    }

    fn request(verb: InputVerb) -> InputRequest {
        InputRequest {
            via: Endpoint::Anonymous,
            verb,
        }
    }

    #[test]
    fn received_bytes_become_text_reports_and_the_cadence_follows_traffic() {
        let mut device = SerialDevice::with_line(line(b"ls -l /home\rpwd\r"));
        let (producer, mut consumer) = queue(32);
        assert_eq!(device.attach_reports(producer), InputResponse::Subscribed(SubscriptionId(1)));
        assert!(device.receive(1_000));
        assert_eq!(texts(&mut consumer), b"ls -l /home\rpwd\r");
        assert_eq!(device.next_poll_ns(), 1_000 + RX_ACTIVE_NS);
        // A quiet line backs off to the idle cadence and wakes nobody.
        assert!(!device.receive(2_000));
        assert_eq!(device.next_poll_ns(), 2_000 + RX_IDLE_NS);
        assert_eq!(device.books().received, 16);
    }

    #[test]
    fn a_full_report_queue_leaves_bytes_in_the_line() {
        let script: std::vec::Vec<u8> = (0..60).map(|i| b'a' + (i % 26) as u8).collect();
        let mut device = SerialDevice::with_line(line(&script));
        let (producer, mut consumer) = queue(2);
        device.attach_reports(producer);
        assert!(device.receive(0));
        // Two reports' worth was read; the rest waits in the line.
        assert_eq!(texts(&mut consumer).len(), 2 * InputReport::TEXT_BYTES);
        assert_eq!(device.books().report_backpressure, 1);
        while device.receive(0) {
            let _ = texts(&mut consumer);
        }
        assert_eq!(device.books().received, 60);
    }

    #[test]
    fn transmit_answers_what_the_line_took_and_only_on_the_console_pair() {
        let mut device = SerialDevice::with_line(ScriptLine {
            tx_cap: 4,
            ..line(b"")
        });
        let mut bytes = [0u8; InputReqWire::TRANSMIT_BYTES];
        bytes[..6].copy_from_slice(b"hello!");
        device.via = TRANSMIT_PAIR;
        assert_eq!(
            device.dispatch(request(InputVerb::Transmit { bytes, len: 6 })),
            InputResponse::Transmitted { accepted: 4 }
        );
        assert_eq!(device.line().tx, b"hell");
        // The terminal's pair may not transmit, and the console's may not
        // subscribe or steer focus.
        device.via = REPORT_PAIR;
        assert_eq!(
            device.dispatch(request(InputVerb::Transmit { bytes, len: 6 })),
            InputResponse::Refused(InputRefusal::Denied)
        );
        device.via = TRANSMIT_PAIR;
        assert_eq!(
            device.dispatch(request(InputVerb::SetFocus { target: None })),
            InputResponse::Refused(InputRefusal::Denied)
        );
    }

    #[test]
    fn one_subscription_and_unsubscribe_needs_its_id() {
        let mut device = SerialDevice::with_line(line(b""));
        let (p1, _c1) = queue(8);
        let (p2, _c2) = queue(8);
        assert_eq!(device.attach_reports(p1), InputResponse::Subscribed(SubscriptionId(1)));
        assert_eq!(device.attach_reports(p2), InputResponse::Refused(InputRefusal::TableFull));
        assert_eq!(
            device.dispatch(request(InputVerb::Unsubscribe {
                subscription: SubscriptionId(9)
            })),
            InputResponse::Refused(InputRefusal::Unknown)
        );
        assert_eq!(
            device.dispatch(request(InputVerb::Unsubscribe {
                subscription: SubscriptionId(1)
            })),
            InputResponse::Unsubscribed
        );
    }
}
