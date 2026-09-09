//! `service` — the output serve step: one attached queue, a draining
//! backend, and counters that never lie.

use kore_abi::session::SlotRef;
use kore_abi::svc::OutChunk;

use kore_rt::service::{Serve, SystemService};

/// A byte sink behind the service — the backend seam (`res/USERLAND.md`
/// §5.9: UART today, a terminal emulator or remote console tomorrow, the
/// shell's source unchanged).
pub trait Backend {
    /// Observe one complete record before its payload is emitted. `endpoint`
    /// is structural serving identity, not data supplied by the producer.
    /// Byte-only sinks ignore this hook; composing sinks may consume typed
    /// metadata from a founder-wired endpoint.
    fn begin_chunk(&mut self, _endpoint: usize, _chunk: &OutChunk) {}

    /// Emit one byte. `false` means this byte was not accepted. The service
    /// retains the chunk suffix and retries it before any later chunk.
    fn put(&mut self, byte: u8) -> bool;

    /// **End of a drained batch** — the backend may now make its accepted
    /// bytes visible.  Default: nothing (a byte-at-a-time sink like the
    /// 16550 is already visible when `put` returns).
    ///
    /// The seam exists for a **deferred** backend: the display's console
    /// keeps the authoritative text in a RAM grid and repaints dirty rows
    /// here, so a whole batch lands as a few full-width write-combining
    /// sweeps instead of per-glyph strided writes (the kernel `FbConsole`
    /// discipline).  Infallible by design: a batch's visibility is not a
    /// per-byte accounting event — bytes were already counted at `put`.
    fn flush(&mut self) {}
}

/// The client→service request vocabulary (`res/USERLAND.md` §5.9).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputRequest {
    /// Attach the queue window the client delivered into this service's
    /// arena at `slot` (the return channel's landing slot).
    AttachTx { slot: SlotRef },
    /// Confirm the backend consumed at least `through` bytes.
    Flush { through: u64 },
    /// Release the attached queue.
    DetachTx,
    /// Snapshot the counters.
    Status,
}

/// The reply vocabulary — data only; authority never rides a reply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputResponse {
    /// The queue attached.
    Attached,
    /// The backend's consumed-byte count after the drain.
    Flushed { through: u64 },
    /// The queue detached.
    Detached,
    /// The counters snapshot.
    Status(OutputStatus),
    /// The request refused (fail-closed).
    Refused(OutputRefusal),
}

/// Why a request was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputRefusal {
    /// A queue is already attached (the v1 single-peer bound).
    Busy,
    /// Flush/detach with no queue attached.
    NotAttached,
    /// The delivered slot was absent, unmappable, or not a well-formed
    /// queue window.
    BadQueue,
}

/// The service's counters — a display snapshot, never a data channel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OutputStatus {
    /// A queue is attached.
    pub attached: bool,
    /// Bytes the backend has consumed (monotonic).
    pub consumed: u64,
    /// Producer-reported ring drops. Backend backpressure loses no bytes;
    /// it leaves `consumed` behind the producer's flush watermark.
    pub dropped: u64,
}

/// Peers the service can fan in over.  Endpoint 0 is the founding
/// client (the v1 single-peer bound lives on *per endpoint* as `Busy`);
/// endpoint 1 exists for the display row's second pair — the terminal
/// discipline's echo queue (the split-terminal ruling, 2026-07-28).
/// Which pair a request arrived on is **structural** identity, stamped
/// by the serving host — never a wire field (the #299 lesson).
pub const MAX_OUTPUT_PEERS: usize = 2;

/// One attached peer: its queue and its own honest books.  `Flush` and
/// `Status` answer **per peer** — a global count would let one client's
/// flush watermark pass on another client's bytes.
struct Peer {
    consumer: kore_ring::spsc::Consumer<'static, OutChunk>,
    consumed: u64,
    pending: Option<(OutChunk, usize)>,
}

/// The output service: one backend, up to [`MAX_OUTPUT_PEERS`] attached
/// queues (one per serving endpoint).
pub struct OutputService<B> {
    backend: B,
    peers: [Option<Peer>; MAX_OUTPUT_PEERS],
    /// A stalled chunk must finish before streams can interleave again.
    active: Option<usize>,
}

impl<B: Backend> OutputService<B> {
    /// Wrap a constructed backend — the test/loopback construction; the
    /// metal path is [`bind`](OutputService::bind) via the endowment.
    pub fn with_backend(backend: B) -> Self {
        Self {
            backend,
            peers: [const { None }; MAX_OUTPUT_PEERS],
            active: None,
        }
    }

    /// The wrapped backend (mut) — a composing service's `start_on_metal`
    /// drives its own witness through it (the display service's render).
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Drain every attached queue to the backend in **descending endpoint
    /// order** (highest first), counting emitted bytes per peer.
    /// Interleave is at **chunk** boundaries only (a chunk is one client's
    /// contiguous bytes; the accepted v1 echo/output semantics).
    ///
    /// The order is causal, not cosmetic: the fan-in convention puts the
    /// **upstream** stream at the higher endpoint (the terminal's echo at
    /// endpoint 1, the program's output at endpoint 0), and the terminal
    /// echoes a line **before** delivering it — so everything the program
    /// writes in response is younger than the echo it answers.  A wake
    /// that finds both queues pending (the completed-line case: the
    /// echoed `'\n'` and the response arrive between two pumps) must
    /// render the echo first or the response concatenates onto the typed
    /// text and the stale newline strands the next prompt — the merge
    /// only a causally-ordered drain gets right.
    ///
    /// A backend stall halts the whole drain without discarding the suffix.
    /// Each pass consumes only a snapshot of published records, so a live
    /// producer cannot keep this call running indefinitely. Snapshot the
    /// downstream output before upstream echo: any response in that snapshot
    /// already has its causally older echo in the later upstream snapshot.
    fn drain(&mut self) {
        let Self {
            backend,
            peers,
            active,
        } = self;
        let published: [u32; MAX_OUTPUT_PEERS] = core::array::from_fn(|endpoint| {
            peers[endpoint].as_ref().map_or(0, |peer| {
                peer.consumer.outstanding().min(peer.consumer.capacity())
            })
        });
        if let Some(endpoint) = *active {
            if let Some(peer) = peers[endpoint].as_mut() {
                if !Self::finish_chunk(backend, peer) {
                    backend.flush();
                    return;
                }
            }
            *active = None;
        }
        for (endpoint, peer) in peers.iter_mut().enumerate().rev() {
            let Some(peer) = peer else { continue };
            for _ in 0..published[endpoint] {
                let Some(chunk) = peer.consumer.pop() else {
                    break;
                };
                backend.begin_chunk(endpoint, &chunk);
                peer.pending = Some((chunk, 0));
                if !Self::finish_chunk(backend, peer) {
                    *active = Some(endpoint);
                    backend.flush();
                    return;
                }
            }
        }
        // One visibility point per drained batch (the deferred backend's
        // whole reason to exist); a no-op for an immediate sink.
        backend.flush();
    }

    fn finish_chunk(backend: &mut B, peer: &mut Peer) -> bool {
        if let Some((chunk, at)) = &mut peer.pending {
            let n = usize::from(chunk.len).min(OutChunk::CHUNK_DATA);
            while *at < n {
                if !backend.put(chunk.bytes[*at]) {
                    return false;
                }
                *at += 1;
                peer.consumed += 1;
            }
        }
        peer.pending = None;
        true
    }

    /// The request-free drain — the serving loop's per-wake pump, so a
    /// producer's doorbell kick renders without waiting for a `Flush`
    /// round trip (the echo path's latency bound).
    pub fn pump(&mut self) {
        self.drain();
    }

    /// Work deferred by the bounded pump or a stalled backend. A resident
    /// caller must arrange another pass rather than relying on a fresh edge
    /// from a producer whose earlier notification may already be consumed.
    pub fn has_pending_output(&self) -> bool {
        self.peers
            .iter()
            .flatten()
            .any(|peer| peer.pending.is_some() || peer.consumer.outstanding() != 0)
    }

    /// The counters snapshot for one endpoint's peer.
    pub fn stats_for(&mut self, ep: usize) -> OutputStatus {
        let peer = self.peers.get_mut(ep).and_then(Option::as_mut);
        OutputStatus {
            attached: peer.is_some(),
            consumed: peer.as_ref().map_or(0, |p| p.consumed),
            dropped: peer.map_or(0, |p| p.consumer.dropped()),
        }
    }

    /// The founding endpoint's snapshot (the single-pair face).
    pub fn stats(&mut self) -> OutputStatus {
        self.stats_for(0)
    }

    /// Attach the delivered queue window (`slot` in this task's arena) as
    /// endpoint `ep`'s peer.  Every step is fail-closed: an absent record,
    /// an unmappable view, or a window that is not an initialized
    /// `OutChunk` ring is a [`OutputRefusal::BadQueue`], never a guess.
    #[cfg(target_os = "none")]
    fn attach(&mut self, ep: usize, slot: SlotRef) -> OutputResponse {
        use kore_rt::cap::verbs;

        let Some(view) = verbs::take_view(slot) else {
            return OutputResponse::Refused(OutputRefusal::BadQueue);
        };
        let Some(window) = verbs::map_view(view) else {
            return OutputResponse::Refused(OutputRefusal::BadQueue);
        };
        // The queue mapping is supply-tracked independently of this arena
        // record. Close the delivered name so the founder-published fixed
        // landing can stand again for another authenticated terminal.
        if !verbs::retire_slot_outcome(slot).is_closed() {
            return OutputResponse::Refused(OutputRefusal::BadQueue);
        }
        // SAFETY: map_view installed this task's own R/W alias of the
        // delivered window for the task's lifetime (nothing unmaps it in
        // v1); the ring crate validates the producer's magic before any
        // cursor is trusted.
        let window: &'static mut [u8] =
            unsafe { core::slice::from_raw_parts_mut(window.cast::<u8>().as_ptr(), window.len()) };
        self.attach_window_at(ep, window)
    }

    /// Off the x86 userland target — the honest typed refusal
    /// (`res/USERLAND.md` §2.1): there is no slot alias to read and no
    /// trap to map through.  Tests drive
    /// [`attach_window_at`](Self::attach_window_at) /
    /// [`attach_consumer_at`](Self::attach_consumer_at) directly.
    #[cfg(not(target_os = "none"))]
    fn attach(&mut self, _ep: usize, _slot: SlotRef) -> OutputResponse {
        OutputResponse::Refused(OutputRefusal::BadQueue)
    }

    /// Attach an already-mapped queue window at endpoint `ep` — the tail
    /// of the metal [`attach`](Self::attach), and the host-test entry.  A
    /// second attach on the same endpoint is [`OutputRefusal::Busy`] (the
    /// single-peer-per-endpoint bound); an out-of-range endpoint is a
    /// serving-host bug answered as `Busy` (fail-closed, no panic path).
    pub fn attach_window_at(&mut self, ep: usize, window: &'static mut [u8]) -> OutputResponse {
        match kore_ring::spsc::consumer::<OutChunk>(window) {
            Ok(consumer) => self.attach_consumer_at(ep, consumer),
            Err(_) => OutputResponse::Refused(OutputRefusal::BadQueue),
        }
    }

    /// [`attach_window_at`](Self::attach_window_at) for the founding
    /// endpoint (the single-pair face).
    pub fn attach_window(&mut self, window: &'static mut [u8]) -> OutputResponse {
        self.attach_window_at(0, window)
    }

    /// Attach a ready consumer half at endpoint `ep` (the `spsc::split`
    /// host-test entry).
    pub fn attach_consumer_at(
        &mut self,
        ep: usize,
        consumer: kore_ring::spsc::Consumer<'static, OutChunk>,
    ) -> OutputResponse {
        match self.peers.get_mut(ep) {
            Some(slot @ None) => {
                *slot = Some(Peer {
                    consumer,
                    consumed: 0,
                    pending: None,
                });
                OutputResponse::Attached
            }
            _ => OutputResponse::Refused(OutputRefusal::Busy),
        }
    }

    /// [`attach_consumer_at`](Self::attach_consumer_at) for the founding
    /// endpoint.
    pub fn attach_consumer(
        &mut self,
        consumer: kore_ring::spsc::Consumer<'static, OutChunk>,
    ) -> OutputResponse {
        self.attach_consumer_at(0, consumer)
    }
}

impl<B: Backend> OutputService<B> {
    /// The step body for a request that arrived on endpoint `ep` — `pub`
    /// so a fan-in host (the display row's two pairs) dispatches with the
    /// structural identity it alone knows.  `Flush` drains **every**
    /// attached queue (the backend is one sink; a per-peer drain would
    /// starve the other queue behind a stalled watermark) but answers the
    /// asking peer's own `consumed` — the watermark a client compares
    /// against its own `pushed` is never inflated by another client's
    /// bytes.
    pub fn dispatch_from(&mut self, ep: usize, req: OutputRequest) -> OutputResponse {
        match req {
            OutputRequest::AttachTx { slot } => self.attach(ep, slot),
            OutputRequest::Flush { .. } => {
                if self.peers.get(ep).is_none_or(Option::is_none) {
                    return OutputResponse::Refused(OutputRefusal::NotAttached);
                }
                self.drain();
                OutputResponse::Flushed {
                    through: self.stats_for(ep).consumed,
                }
            }
            OutputRequest::DetachTx => match self.peers.get_mut(ep) {
                Some(slot @ Some(_)) => {
                    *slot = None;
                    OutputResponse::Detached
                }
                _ => OutputResponse::Refused(OutputRefusal::NotAttached),
            },
            OutputRequest::Status => {
                let stats = self.stats_for(ep);
                OutputResponse::Status(stats)
            }
        }
    }

    /// Reply adapter for services speaking the output-only protocol. A
    /// multiplexing service can wrap `dispatch_from` in its own response.
    pub fn step_from(&mut self, ep: usize, req: OutputRequest, sv: &mut Serve<'_, OutputResponse>) {
        sv.reply(self.dispatch_from(ep, req));
    }

    /// The founding endpoint's step (the single-pair face — the UART row
    /// and the generic `run_service` path).
    pub fn step(&mut self, req: OutputRequest, sv: &mut Serve<'_, OutputResponse>) {
        self.step_from(0, req, sv);
    }
}

/// The output service's endowment: the PCI-serial device whose BAR is
/// the backend's register window (`role::DEVICE` — the input driver's
/// claim shape).
#[derive(Debug, kore_rt::Bundle)]
#[bundle(name = "output")]
pub struct OutputEndowment {
    /// The claimed `PciDevice` token.
    #[bundle(role = kore_abi::task::role::DEVICE)]
    pub device: kore_rt::cap::DeviceToken,
}

impl SystemService for OutputService<UartBackend> {
    const NAME: &'static str = "output";
    type Endowment = OutputEndowment;

    /// The founding selector: a 16550-family PCI serial card (QEMU
    /// `-device pci-serial`) — registers over BAR0, polled (no IRQ, no DMA).
    /// Required: an output service with no port is a service that cannot
    /// output.
    const DEVICES: &'static [kore_rt::service::DeviceRequest] =
        &[kore_rt::service::DeviceRequest {
            selector: kore_rt::service::DeviceSelector::Pci(kore_rt::service::PciSelector {
                class: 0x07,
                subclass: 0x00,
                progif: 0x02,
                identity: kore_rt::service::PciIdentitySelector::ANY,
                bars: kore_rt::service::PciBarMask::BAR0,
                dma: false,
                interrupts: kore_rt::service::PciInterruptRequirement::None,
            }),
            required: true,
        }];

    /// A polled 16550: no DMA, no route.  Nothing to stop on the way down.
    const DEVICE_CLASSES: &'static [kore_rt::service::DeviceClass] =
        &[kore_rt::service::DeviceClass::NoDma];

    fn bind(endowment: Self::Endowment) -> Self {
        Self::with_backend(UartBackend::from_device(&endowment.device))
    }

    /// The backend brings itself online and writes its witness before
    /// readiness is advertised (`res/USERLAND.md` §5.9: a real backend
    /// owns the hardware capability — a service whose device or BAR never
    /// materialized must fail closed, never beacon).
    fn start_on_metal(
        &mut self,
        _rt: &kore_rt::entry::Rt,
    ) -> Result<(), kore_rt::service::StartFailure> {
        if !self.backend.online() {
            return Err(kore_rt::service::StartFailure::UNSPECIFIED);
        }
        for &byte in b"kore-svc-output: backend online\n" {
            self.backend.put(byte);
        }
        Ok(())
    }

    fn serve(&mut self, req: Self::Request, sv: &mut Serve<'_, Self::Response>) {
        self.step(req, sv);
    }
}

/// The test-only no-endowment bundle (kore-rt's `NoCaps` is private to
/// its own suite).
#[cfg(test)]
#[derive(Debug, kore_rt::Bundle)]
#[bundle(name = "output-test")]
pub struct TestCaps;

/// The same serve contract over any backend — the host-test face of the
/// one step (`OutputService<LoopBackend>` in this crate's tests).
#[cfg(test)]
impl SystemService for OutputService<LoopBackend> {
    const NAME: &'static str = "output";
    type Endowment = TestCaps;

    fn bind(_: Self::Endowment) -> Self {
        Self::with_backend(LoopBackend::default())
    }

    fn serve(&mut self, req: Self::Request, sv: &mut Serve<'_, Self::Response>) {
        self.step(req, sv);
    }
}

/// The clockless fallback for [`kore_uart::DEFAULT_TX_TIMEOUT_NS`] — what
/// this backend shipped with, still correct where `kore_rt::clock` cannot answer.
const TX_SPIN_LIMIT: u32 = 100_000;

#[cfg(target_os = "none")]
#[inline]
fn tx_pause() {
    let _ = kore_rt::hint::pause_until(kore_rt::pace::PROBE_PAUSE);
}

/// The v1 metal backend: a 16550-family UART over a mapped PCI-serial
/// BAR (QEMU `-device pci-serial`: BAR0, 32 bytes, stride 1), built on
/// `kore-uart`'s register logic.  Bytes are written verbatim — no CRLF
/// expansion, no line discipline (`res/USERLAND.md` §5.9: opaque bytes).
/// A failed bounded wait leaves the byte unconsumed, so a stalled UART
/// cannot silently advance the queue cursor.
pub struct UartBackend {
    window: Option<kore_rt::cap::BarWindow>,
    uart: kore_uart::Uart16550,
}

impl UartBackend {
    /// Claim + map BAR0 of the endowed device and construct the backend.
    /// A device with no usable BAR yields a backend with **no window**:
    /// every `put` is a counted stall (observable via `consumed`), never
    /// a panic at bind.
    pub fn from_device(device: &kore_rt::cap::DeviceToken) -> Self {
        let window = kore_rt::cap::verbs::map_bar(device, 0).ok();
        Self {
            window,
            uart: kore_uart::Uart16550::new(1),
        }
    }

    /// A backend over an already-mapped register window (host tests).
    #[cfg(test)]
    pub fn over_window(window: kore_rt::cap::BarWindow) -> Self {
        Self {
            window: Some(window),
            uart: kore_uart::Uart16550::new(1),
        }
    }

    /// Whether the register window materialized (the device was found and
    /// its BAR mapped) — [`start_on_metal`](SystemService::start_on_metal)'s
    /// fail-closed gate.
    pub fn online(&self) -> bool {
        self.window.is_some()
    }
}

impl Backend for UartBackend {
    #[cfg(target_os = "none")]
    fn put(&mut self, byte: u8) -> bool {
        use kore_uart::RegisterAccess;

        struct Mmio {
            base: *mut u8,
        }
        impl RegisterAccess for Mmio {
            fn read8(&self, offset: usize) -> u8 {
                // SAFETY: the BAR window is mapped for the task's lifetime
                // and offset comes from the 16550's 8-register layout.
                unsafe { self.base.add(offset).read_volatile() }
            }
            fn write8(&self, offset: usize, value: u8) {
                // SAFETY: as above.
                unsafe { self.base.add(offset).write_volatile(value) }
            }
            fn read32(&self, offset: usize) -> u32 {
                // SAFETY: as above; the access is a genuine 32-bit load at
                // the byte offset (the RegisterAccess contract — a byte
                // widen would break any 32-bit-register backend, e.g.
                // PL011/SBSA).
                unsafe { (self.base.add(offset) as *const u32).read_volatile() }
            }
            fn write32(&self, offset: usize, value: u32) {
                // SAFETY: as above (a genuine 32-bit store).
                unsafe { (self.base.add(offset) as *mut u32).write_volatile(value) }
            }
        }

        let Some(window) = &self.window else {
            return false; // counted stall: consumed does not advance
        };
        let regs = Mmio {
            base: window.base(),
        };
        // Share the bounded poll with both kernel console paths. A lost
        // clock refuses this byte too; the queue keeps it unconsumed.
        kore_uart::transmit_byte(
            &self.uart,
            &regs,
            byte,
            kore_uart::DEFAULT_TX_TIMEOUT_NS,
            TX_SPIN_LIMIT,
            || kore_rt::clock::uptime().ok().map(|now| now.as_nanos()),
            tx_pause,
        )
    }

    /// Off the x86 userland target the UART backend is inert
    /// (`res/USERLAND.md` §2.1) — host tests use the loop sink.
    #[cfg(not(target_os = "none"))]
    fn put(&mut self, _byte: u8) -> bool {
        false
    }
}

/// A bounded recording sink for tests: bytes are collected up to the
/// capacity, then the sink **stalls** (puts become no-ops) — exactly the
/// observable-stall contract, so a test can drive both directions.
#[cfg(test)]
#[derive(Debug)]
pub struct LoopBackend {
    /// The recorded byte stream (test reads it back).
    pub bytes: [u8; 4096],
    /// Live prefix of `bytes`.
    pub len: usize,
}

#[cfg(test)]
impl LoopBackend {
    /// The recorded prefix.
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}

#[cfg(test)]
impl Default for LoopBackend {
    fn default() -> Self {
        Self {
            bytes: [0; 4096],
            len: 0,
        }
    }
}

#[cfg(test)]
impl Backend for LoopBackend {
    fn put(&mut self, byte: u8) -> bool {
        if self.len < self.bytes.len() {
            self.bytes[self.len] = byte;
            self.len += 1;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One split queue in a leaked, ring-aligned window (the metal map's
    /// 'static shape).
    fn queue() -> (
        kore_ring::spsc::Producer<'static, OutChunk>,
        kore_ring::spsc::Consumer<'static, OutChunk>,
    ) {
        let layout = std::alloc::Layout::from_size_align(4096, 64).unwrap();
        let window: &'static mut [u8] = unsafe {
            let p = std::alloc::alloc(layout);
            assert!(!p.is_null());
            core::slice::from_raw_parts_mut(p, 4096)
        };
        kore_ring::spsc::split::<OutChunk>(window, 32).expect("split")
    }

    fn push_chunk(producer: &mut kore_ring::spsc::Producer<'_, OutChunk>, bytes: &[u8]) {
        let mut chunk = OutChunk {
            len: bytes.len() as u8,
            flags: 0,
            reserved: [0; 6],
            bytes: [0; 24],
        };
        chunk.bytes[..bytes.len()].copy_from_slice(bytes);
        producer.push(chunk).expect("queue has room");
    }

    #[test]
    fn attach_write_flush_drains_in_order() {
        let (mut producer, consumer) = queue();
        let mut svc = OutputService::with_backend(LoopBackend::default());
        assert_eq!(svc.attach_consumer(consumer), OutputResponse::Attached);

        push_chunk(&mut producer, b"hello, ");
        push_chunk(&mut producer, b"uart");

        let out = kore_rt::service::step(&mut svc, OutputRequest::Flush { through: 11 });
        assert_eq!(
            out.into_reply(),
            Some(OutputResponse::Flushed { through: 11 })
        );
        assert_eq!(svc.backend.as_slice(), b"hello, uart");
        assert_eq!(svc.stats().consumed, 11);
    }

    #[test]
    fn busy_refusal_then_detach_then_not_attached() {
        let (_p1, c1) = queue();
        let (_p2, c2) = queue();
        let mut svc = OutputService::with_backend(LoopBackend::default());

        assert_eq!(svc.attach_consumer(c1), OutputResponse::Attached);
        assert_eq!(
            svc.attach_consumer(c2),
            OutputResponse::Refused(OutputRefusal::Busy)
        );
        let out = kore_rt::service::step(&mut svc, OutputRequest::DetachTx);
        assert_eq!(out.into_reply(), Some(OutputResponse::Detached));
        let out = kore_rt::service::step(&mut svc, OutputRequest::DetachTx);
        assert_eq!(
            out.into_reply(),
            Some(OutputResponse::Refused(OutputRefusal::NotAttached))
        );
    }

    #[test]
    fn flushed_reports_the_current_consumed_count() {
        let (mut producer, consumer) = queue();
        let mut svc = OutputService::with_backend(LoopBackend::default());
        assert_eq!(svc.attach_consumer(consumer), OutputResponse::Attached);
        push_chunk(&mut producer, b"abc");

        let out = kore_rt::service::step(&mut svc, OutputRequest::Flush { through: 3 });
        assert_eq!(
            out.into_reply(),
            Some(OutputResponse::Flushed { through: 3 })
        );
        // A second flush answers the CURRENT total, not the request's.
        let out = kore_rt::service::step(&mut svc, OutputRequest::Flush { through: 9 });
        assert_eq!(
            out.into_reply(),
            Some(OutputResponse::Flushed { through: 3 })
        );
        assert_eq!(svc.backend.as_slice(), b"abc");
    }

    #[test]
    fn status_snapshots_the_counters() {
        let (mut producer, consumer) = queue();
        let mut svc = OutputService::with_backend(LoopBackend::default());
        assert_eq!(svc.attach_consumer(consumer), OutputResponse::Attached);
        push_chunk(&mut producer, b"xy");
        let out = kore_rt::service::step(&mut svc, OutputRequest::Flush { through: 2 });
        assert!(out.into_reply().is_some());
        let out = kore_rt::service::step(&mut svc, OutputRequest::Status);
        assert_eq!(
            out.into_reply(),
            Some(OutputResponse::Status(OutputStatus {
                attached: true,
                consumed: 2,
                dropped: 0,
            }))
        );
    }

    /// A test fan-in wrapper: the `DisplayService` stamp idiom in
    /// miniature, so `step_from` is drivable through the framework step.
    struct FanIn {
        svc: OutputService<LoopBackend>,
        via: usize,
    }

    /// The wrapper fans in over the one output vocabulary — it wraps a
    /// host, it does not invent a protocol, so its wire is the crate's
    /// wire, delegated whole.
    impl kore_rt::svc::SvcProtocol for FanIn {
        type Request = OutputRequest;
        type Response = OutputResponse;
        type ReqWire = kore_abi::svc::OutReqWire;
        type RepWire = kore_abi::svc::OutRepWire;
        const REPLY_GRANTS: usize = 0;

        fn encode_req(req: &Self::Request) -> Self::ReqWire {
            crate::wire::encode_req(req)
        }
        fn decode_req(w: &Self::ReqWire) -> Option<Self::Request> {
            crate::wire::decode_req(w)
        }
        fn encode_rep(rep: &Self::Response, grants: u8) -> Self::RepWire {
            crate::wire::encode_rep(rep, grants)
        }
        fn decode_rep(w: &Self::RepWire) -> Option<(Self::Response, u8)> {
            crate::wire::decode_rep(w).map(|r| (r, w.grants))
        }
    }

    impl SystemService for FanIn {
        const NAME: &'static str = "output-fanin-test";
        type Endowment = TestCaps;

        fn bind(_: Self::Endowment) -> Self {
            Self {
                svc: OutputService::with_backend(LoopBackend::default()),
                via: 0,
            }
        }

        fn serve(&mut self, req: Self::Request, sv: &mut Serve<'_, Self::Response>) {
            self.svc.step_from(self.via, req, sv);
        }
    }

    fn fan_in() -> FanIn {
        FanIn {
            svc: OutputService::with_backend(LoopBackend::default()),
            via: 0,
        }
    }

    #[test]
    fn two_peers_attach_and_account_independently() {
        let (mut p0, c0) = queue();
        let (mut p1, c1) = queue();
        let mut svc = OutputService::with_backend(LoopBackend::default());
        assert_eq!(svc.attach_consumer_at(0, c0), OutputResponse::Attached);
        assert_eq!(svc.attach_consumer_at(1, c1), OutputResponse::Attached);

        push_chunk(&mut p0, b"shell:");
        push_chunk(&mut p1, b"echo");
        svc.pump();

        // Descending endpoint order (the causal drain: the upstream echo
        // stream at the higher endpoint renders first), chunk-boundary
        // interleave; per-peer books.
        assert_eq!(svc.backend.as_slice(), b"echoshell:");
        assert_eq!(svc.stats_for(0).consumed, 6);
        assert_eq!(svc.stats_for(1).consumed, 4);
    }

    /// The completed-line merge — the race the causal drain order exists
    /// for.  The terminal echoes `"hello\n"` (endpoint 1) and *then*
    /// delivers the line; the shell's response + next prompt (endpoint 0)
    /// land before the display's next pump.  One pump must render the
    /// echo first: `prompt · typed line · newline · response · prompt` —
    /// the endpoint-0-first order instead concatenates the response onto
    /// the typed text and strands the fresh prompt on a bare line (the
    /// first interactive boot's observed garbling).
    #[test]
    fn completed_line_renders_echo_before_response() {
        let (mut shell, c0) = queue();
        let (mut echo, c1) = queue();
        let mut svc = OutputService::with_backend(LoopBackend::default());
        assert_eq!(svc.attach_consumer_at(0, c0), OutputResponse::Attached);
        assert_eq!(svc.attach_consumer_at(1, c1), OutputResponse::Attached);

        // The prompt renders on an earlier pump; typing echoes live.
        push_chunk(&mut shell, b"kore> ");
        svc.pump();
        push_chunk(&mut echo, b"hello");
        svc.pump();

        // Enter: the echo's newline is enqueued BEFORE the line is
        // delivered, so the response is causally younger — but both are
        // pending by the next pump (the coalesced wake).  Chunks stay
        // within `OutChunk::CHUNK_DATA`, as the shell's writer splits.
        push_chunk(&mut echo, b"\n");
        push_chunk(&mut shell, b"kore: unknown command: ");
        push_chunk(&mut shell, b"hello\n");
        push_chunk(&mut shell, b"kore> ");
        svc.pump();

        assert_eq!(
            svc.backend.as_slice(),
            b"kore> hello\nkore: unknown command: hello\nkore> ",
        );
    }

    #[test]
    fn flush_answers_the_asking_peers_watermark() {
        let (mut p0, c0) = queue();
        let (mut p1, c1) = queue();
        let mut host = fan_in();
        assert_eq!(host.svc.attach_consumer_at(0, c0), OutputResponse::Attached);
        assert_eq!(host.svc.attach_consumer_at(1, c1), OutputResponse::Attached);
        push_chunk(&mut p0, b"program output");
        push_chunk(&mut p1, b"echo");

        // The echo peer's flush drains everything (one sink) but answers
        // ITS OWN consumed — never inflated by the shell's bytes.
        host.via = 1;
        let out = kore_rt::service::step(&mut host, OutputRequest::Flush { through: 4 });
        assert_eq!(
            out.into_reply(),
            Some(OutputResponse::Flushed { through: 4 })
        );
        assert_eq!(host.svc.backend.len, 18);
    }

    #[test]
    fn attach_detach_and_status_are_per_endpoint() {
        let (_p0, c0) = queue();
        let (_p0b, c0b) = queue();
        let (_p1, c1) = queue();
        let mut host = fan_in();
        assert_eq!(host.svc.attach_consumer_at(0, c0), OutputResponse::Attached);
        assert_eq!(
            host.svc.attach_consumer_at(0, c0b),
            OutputResponse::Refused(OutputRefusal::Busy)
        );
        assert_eq!(host.svc.attach_consumer_at(1, c1), OutputResponse::Attached);

        host.via = 0;
        let out = kore_rt::service::step(&mut host, OutputRequest::DetachTx);
        assert_eq!(out.into_reply(), Some(OutputResponse::Detached));
        let out = kore_rt::service::step(&mut host, OutputRequest::Status);
        assert_eq!(
            out.into_reply(),
            Some(OutputResponse::Status(OutputStatus {
                attached: false,
                consumed: 0,
                dropped: 0,
            }))
        );
        host.via = 1;
        let out = kore_rt::service::step(&mut host, OutputRequest::Status);
        assert_eq!(
            out.into_reply(),
            Some(OutputResponse::Status(OutputStatus {
                attached: true,
                consumed: 0,
                dropped: 0,
            }))
        );
    }

    #[test]
    fn out_of_range_endpoint_fails_closed() {
        let (_p, c) = queue();
        let mut host = fan_in();
        assert_eq!(
            host.svc.attach_consumer_at(MAX_OUTPUT_PEERS, c),
            OutputResponse::Refused(OutputRefusal::Busy)
        );
        host.via = MAX_OUTPUT_PEERS;
        let out = kore_rt::service::step(&mut host, OutputRequest::Flush { through: 0 });
        assert_eq!(
            out.into_reply(),
            Some(OutputResponse::Refused(OutputRefusal::NotAttached))
        );
    }

    /// A sink that accepts exactly `cap` bytes, then wedges.
    struct TinySink {
        cap: usize,
        len: usize,
    }

    impl Backend for TinySink {
        fn put(&mut self, _byte: u8) -> bool {
            if self.len < self.cap {
                self.len += 1;
                true
            } else {
                false
            }
        }
    }

    #[test]
    fn a_backend_stall_preserves_the_suffix_and_resumes_before_the_other_peer() {
        let (mut p0, c0) = queue();
        let (mut p1, c1) = queue();
        let mut svc = OutputService::with_backend(TinySink { cap: 10, len: 0 });
        assert_eq!(svc.attach_consumer_at(0, c0), OutputResponse::Attached);
        assert_eq!(svc.attach_consumer_at(1, c1), OutputResponse::Attached);
        push_chunk(&mut p1, b"twenty-four bytes here!!");
        push_chunk(&mut p0, b"x");
        svc.pump();
        assert!(svc.has_pending_output());

        // Peer 1 lands ten bytes. Its suffix is retained, not counted lost;
        // peer 0 remains queued behind that exact chunk.
        assert_eq!(svc.stats_for(1).consumed, 10);
        assert_eq!(svc.stats_for(1).dropped, 0);
        assert_eq!(svc.stats_for(0).consumed, 0);
        assert_eq!(svc.stats_for(0).dropped, 0);
        svc.backend_mut().cap = 23;
        svc.pump();
        assert_eq!(svc.stats_for(1).consumed, 23);
        assert_eq!(svc.stats_for(0).consumed, 0);
        svc.backend_mut().cap = 64;
        svc.pump();
        assert_eq!(svc.stats_for(1).consumed, 24);
        assert_eq!(svc.stats_for(0).consumed, 1);
        assert_eq!(svc.stats_for(1).dropped, 0);
        assert!(!svc.has_pending_output());
    }

    #[test]
    fn a_refilling_producer_cannot_extend_a_pump_past_its_snapshot() {
        struct RefillingSink {
            producer: kore_ring::spsc::Producer<'static, OutChunk>,
            accepted: usize,
        }
        impl Backend for RefillingSink {
            fn put(&mut self, _byte: u8) -> bool {
                self.accepted += 1;
                // A finite upper bound also makes a regressed drain-to-empty
                // test fail promptly instead of hanging the test process.
                if self.accepted < 128 {
                    push_chunk(&mut self.producer, b"x");
                }
                true
            }
        }
        let (mut producer, consumer) = queue();
        push_chunk(&mut producer, b"x");
        let mut svc = OutputService::with_backend(RefillingSink {
            producer,
            accepted: 0,
        });
        assert_eq!(svc.attach_consumer(consumer), OutputResponse::Attached);
        svc.pump();
        assert_eq!(svc.backend.accepted, 1);
        assert!(svc.has_pending_output());
        svc.pump();
        assert_eq!(svc.backend.accepted, 2);
        assert_eq!(svc.stats().dropped, 0);
    }

    #[test]
    fn echo_and_response_arriving_during_a_pump_stay_in_the_next_batch() {
        struct ArrivingSink {
            echo: kore_ring::spsc::Producer<'static, OutChunk>,
            output: kore_ring::spsc::Producer<'static, OutChunk>,
            bytes: LoopBackend,
            injected: bool,
        }
        impl Backend for ArrivingSink {
            fn put(&mut self, byte: u8) -> bool {
                if !core::mem::replace(&mut self.injected, true) {
                    push_chunk(&mut self.echo, b"\n");
                    push_chunk(&mut self.output, b"response");
                }
                self.bytes.put(byte)
            }
        }
        let (output, output_rx) = queue();
        let (mut echo, echo_rx) = queue();
        push_chunk(&mut echo, b"command");
        let mut svc = OutputService::with_backend(ArrivingSink {
            echo,
            output,
            bytes: LoopBackend::default(),
            injected: false,
        });
        assert_eq!(
            svc.attach_consumer_at(0, output_rx),
            OutputResponse::Attached
        );
        assert_eq!(svc.attach_consumer_at(1, echo_rx), OutputResponse::Attached);
        svc.pump();
        assert_eq!(svc.backend.bytes.as_slice(), b"command");
        assert!(svc.has_pending_output());
        svc.pump();
        assert_eq!(svc.backend.bytes.as_slice(), b"command\nresponse");
        assert!(!svc.has_pending_output());
    }
}
