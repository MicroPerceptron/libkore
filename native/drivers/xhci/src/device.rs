//! **Slot lifecycle + EP0 control transfers + Configure Endpoint** — the
//! device half of enumeration, over the [`kore_driver`] transfer ring.
//!
//! The batching the architect called for lands here two ways:
//! - **One doorbell per control transfer.** A control transfer is Setup +
//!   (Data) + Status — up to three TRBs — staged onto the EP0 ring and then
//!   *one* doorbell write ([`SubRing::ring`]); the controller walks the TD.
//! - **One page per device's contexts.** A device needs an *input* context
//!   (the driver writes, the controller reads for a command) and an *output*
//!   device context (the controller owns).  For 32-byte contexts both fit
//!   one 4 KiB page — input at offset 0, output at 2048 — so a device is
//!   three buffers (context page + EP0 ring + data), not four, and the
//!   userland `xhci_device` flow leases them in **one crossing** (X1c).
//!   Since `mm` rounds a lease up to a page anyway, packing sub-page
//!   structures into one page is free space *and* one fewer crossing.
//!
//! **Both context sizes** are handled: the layout is built through
//! `&dyn InputHandler` and serialized at a CSZ-scaled stride
//! ([`ctx_dwords`](Controller::ctx_dwords) — 8 dwords for 32-byte contexts,
//! QEMU + the mock; 16 for the 64-byte contexts many real Intel parts use),
//! so the only per-size branch is the container (`Input<8>` vs `Input<16>`).

use kore_driver::ring::{SubRing, Wrap};
use kore_driver::seam::{Dma, DmaPool, Regs64};

// Only `InputHandler` is imported for method resolution — `Input`'s own
// `control_mut`/`device_mut`.  The sub-handler methods (`slot_mut`,
// `endpoint_mut`, the field setters, `as_ref`) are called on `&dyn …` trait
// objects, which carry their vtable without the trait in scope.
use xhci::context::{EndpointType, Input, InputHandler};
use xhci::ring::trb::event;
use xhci::ring::trb::transfer::{Direction, TransferType};
use xhci::ring::trb::{Link, command, transfer};

use kore_abi::device::IovaAddr;

use crate::descriptor::{DeviceDescriptor, EndpointDescriptor, desc_type};
use crate::port::Port;
use crate::trb::Trb;
use crate::{Controller, InterruptEvent};
use crate::{Error, Result};

/// Ring depth (one page = 256 TRBs), matching the controller's rings.
const RING_TRBS: u16 = 256;
/// The output device context's offset within the shared context page (the
/// packing) — 64-byte aligned, clear of the ≤5-context input area.
const OUTPUT_CTX_OFFSET: usize = 2048;

/// A USB device's negotiated bus speed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Speed {
    /// Low speed (1.5 Mb/s).
    Low,
    /// Full speed (12 Mb/s).
    Full,
    /// High speed (480 Mb/s).
    High,
    /// SuperSpeed (5 Gb/s+).
    Super,
    /// An unrecognized protocol-speed-ID.
    Unknown,
}

impl Speed {
    /// Decode the default xHCI Protocol Speed ID (PORTSC speed field).
    pub fn from_psi(psi: u8) -> Self {
        match psi {
            1 => Speed::Full,
            2 => Speed::Low,
            3 => Speed::High,
            4 => Speed::Super,
            _ => Speed::Unknown,
        }
    }

    /// The value to program into the slot context's Speed field.
    pub fn psi(self) -> u8 {
        match self {
            Speed::Full => 1,
            Speed::Low => 2,
            Speed::High => 3,
            Speed::Super => 4,
            Speed::Unknown => 0,
        }
    }

    /// The initial EP0 max packet size to guess before the device descriptor
    /// confirms it (xHCI 4.3): LS/FS = 8, HS = 64, SS = 512.
    pub fn control_mps(self) -> u16 {
        match self {
            Speed::Low | Speed::Full | Speed::Unknown => 8,
            Speed::High => 64,
            Speed::Super => 512,
        }
    }

    /// Decode `bMaxPacketSize0` without treating a reserved value as a packet
    /// size. USB2 uses bytes; SuperSpeed uses the exponent (USB3 §9.6.1).
    fn descriptor_control_mps(self, encoded: u8) -> Result<u16> {
        match (self, encoded) {
            (Self::Low, 8) => Ok(8),
            (Self::Full, 8 | 16 | 32 | 64) => Ok(u16::from(encoded)),
            (Self::High, 64) => Ok(64),
            (Self::Super, 9) => Ok(512),
            _ => Err(Error::Unsupported),
        }
    }
}

/// Endpoints one device may hold beyond EP0.
///
/// A **design bound, not a tuning knob**: the device classes kore drives need
/// a HID interrupt IN (1), or a CDC networking notification IN plus a bulk
/// IN/OUT data pair (3).  Four leaves headroom for one more without making
/// [`Device`] a large value to move.
pub const MAX_ENDPOINTS: usize = 4;

/// Transfers one endpoint may have submitted but uncompleted.
///
/// Bounds the correlation table, and therefore how deep a bulk queue can run
/// before [`Controller::bulk_submit`] refuses.  A NIC's TX depth rides this.
pub const MAX_INFLIGHT: usize = 32;

/// What an endpoint is for — the xHCI endpoint type restricted to the shapes
/// this driver configures.  Control is not here: EP0 is the only control
/// endpoint and [`Device`] holds it separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndpointKind {
    /// Periodic IN — HID boot reports, a CDC network's connection
    /// notifications.
    InterruptIn,
    /// Bulk IN — received frames.
    BulkIn,
    /// Bulk OUT — transmitted frames.
    BulkOut,
}

impl EndpointKind {
    /// Classify a descriptor, or `None` for a transfer type this driver does
    /// not configure (control duplicates, isochronous).
    pub fn of(ep: &EndpointDescriptor) -> Option<Self> {
        match (ep.is_bulk(), ep.is_interrupt(), ep.is_in()) {
            (true, _, true) => Some(Self::BulkIn),
            (true, _, false) => Some(Self::BulkOut),
            (_, true, true) => Some(Self::InterruptIn),
            _ => None,
        }
    }

    /// The xHCI endpoint-context type this kind programs.
    fn context(self) -> EndpointType {
        match self {
            Self::InterruptIn => EndpointType::InterruptIn,
            Self::BulkIn => EndpointType::BulkIn,
            Self::BulkOut => EndpointType::BulkOut,
        }
    }

    /// Whether this kind needs a driver-owned data buffer.  Interrupt IN does
    /// (the report lands somewhere the driver reads); bulk transfers out of
    /// caller-supplied device memory, so it does not.
    fn needs_buffer(self) -> bool {
        matches!(self, Self::InterruptIn)
    }
}

/// One submitted-but-uncompleted transfer, awaiting its Transfer Event.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct InFlight {
    /// Device address of the Normal TRB carrying this transfer.  The Transfer
    /// Event names the TRB that generated it, so this is the correlation key
    /// — and checking it is what turns a mis-ordered or spurious event into a
    /// refusal instead of a cookie attributed to the wrong caller.
    trb: u64,
    /// The submitter's opaque cookie, returned verbatim on completion.
    cookie: u64,
    /// Bytes requested, so a short packet yields a transferred length rather
    /// than a residual the caller has to reconstruct.
    len: u32,
}

/// A completed bulk transfer, correlated back to its submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BulkCompletion {
    /// The cookie [`Controller::bulk_submit`] was given.
    pub cookie: u64,
    /// Bytes actually moved (`requested` − residual).
    pub transferred: u32,
    /// Raw xHCI completion code (xHCI 1.2 Table 6-90).
    pub completion_code: u8,
}

impl BulkCompletion {
    /// Whether the buffer holds valid bytes.  Bulk completes Success on a
    /// full transfer and ShortPacket on a partial one; both moved data.
    pub fn ok(self) -> bool {
        self.completion_code == event::CompletionCode::Success as u8
            || self.completion_code == event::CompletionCode::ShortPacket as u8
    }
}

/// A configured non-control endpoint — its transfer ring, DCI, packet size,
/// an optional driver-owned data buffer, and the in-flight table correlating
/// Transfer Events back to the cookies their submissions carried.
///
/// Populated by [`Controller::configure_endpoints`].  The HID boot driver
/// (X1c) arms and polls the interrupt kind; a networking class submits and
/// reaps the bulk kinds.
pub struct Endpoint<P: DmaPool> {
    /// The endpoint's transfer ring.
    pub ring: SubRing<Trb, P::Buf>,
    /// Device Context Index.
    pub dci: u8,
    /// Max packet size.
    pub mps: u16,
    /// What this endpoint is for.
    pub kind: EndpointKind,
    /// The driver-owned DMA buffer, for kinds that need one
    /// ([`EndpointKind::needs_buffer`]).  Bulk endpoints hold `None` and
    /// transfer out of memory the caller names.
    pub data: Option<P::Buf>,
    /// FIFO of submitted transfers.  An endpoint's transfer ring is processed
    /// in order and its TDs complete in order, so a queue — not a map — is
    /// the exact structure; the stored TRB address is checked on completion
    /// so an out-of-order or spurious event refuses rather than mis-attributes.
    inflight: [InFlight; MAX_INFLIGHT],
    /// Index of the oldest outstanding transfer (free-running, masked).
    head: u32,
    /// One past the newest (free-running, masked).
    tail: u32,
}

impl<P: DmaPool> Endpoint<P> {
    /// Transfers submitted and not yet completed.
    pub fn outstanding(&self) -> u32 {
        self.tail.wrapping_sub(self.head)
    }

    /// Whether another submission would exceed [`MAX_INFLIGHT`].
    pub fn full(&self) -> bool {
        self.outstanding() as usize >= MAX_INFLIGHT
    }

    /// Record a submission.  `Err` when the correlation table is full —
    /// refused, never overwritten, because dropping the oldest entry would
    /// silently orphan a caller's cookie.
    fn track(&mut self, trb: u64, cookie: u64, len: u32) -> Result<()> {
        if self.full() {
            return Err(Error::Unsupported);
        }
        let slot = (self.tail as usize) % MAX_INFLIGHT;
        self.inflight[slot] = InFlight { trb, cookie, len };
        self.tail = self.tail.wrapping_add(1);
        Ok(())
    }

    /// Retire the oldest submission against a Transfer Event naming `trb`.
    ///
    /// `None` when nothing is outstanding or the event names a TRB that is
    /// not the oldest — the caller must treat that as a refusal, not retry
    /// against the next entry, because a cookie handed to the wrong caller
    /// is worse than a dropped completion.
    fn retire(&mut self, trb: u64, residual: u32, code: u8) -> Option<BulkCompletion> {
        if self.outstanding() == 0 {
            return None;
        }
        let slot = (self.head as usize) % MAX_INFLIGHT;
        let entry = self.inflight[slot];
        if entry.trb != trb {
            return None;
        }
        self.head = self.head.wrapping_add(1);
        Some(BulkCompletion {
            cookie: entry.cookie,
            transferred: entry.len.saturating_sub(residual),
            completion_code: code,
        })
    }

    /// Drop every outstanding correlation — used after an endpoint halt or
    /// reset, where the controller will not deliver the pending events.
    pub fn flush(&mut self) {
        self.head = self.tail;
    }
}

/// An addressed USB device on a slot — its EP0 ring, contexts, a control
/// data buffer, and (once configured) an interrupt IN endpoint.
pub struct Device<P: DmaPool> {
    /// The slot id the controller assigned.
    pub slot_id: u8,
    /// The root-hub port number (1-based).
    pub port: u8,
    /// The negotiated speed.
    pub speed: Speed,
    /// EP0 (default control) transfer ring.
    ep0: SubRing<Trb, P::Buf>,
    /// EP0 max packet size (post-descriptor).
    ep0_mps: u16,
    /// The shared context page (input @0, output device ctx @2048).
    ctx: P::Buf,
    /// The control-transfer data buffer (descriptors land here, and OUT
    /// payloads are staged here before a control write).
    data: P::Buf,
    /// Configured non-control endpoints, in configuration order.  Bounded by
    /// [`MAX_ENDPOINTS`]; looked up by DCI over a scan whose length is that
    /// architectural constant, so there is no per-device index to keep
    /// coherent.
    endpoints: [Option<Endpoint<P>>; MAX_ENDPOINTS],
}

impl<P: DmaPool> Device<P> {
    /// The control-data buffer contents after a transfer (e.g. a descriptor).
    pub fn read_data(&self, len: usize) -> &[u8] {
        // SAFETY: `data` is at least one page; `len` is bounded by a control
        // transfer (≤ a page); the bytes were written by the device (DMA) or
        // the mock and are valid for read.
        unsafe { core::slice::from_raw_parts(self.data.cpu() as *const u8, len) }
    }

    /// Stage `src` into the control-data buffer for a host-to-device control
    /// transfer.  Refused rather than truncated when it would leave the
    /// buffer — a short class request is a wrong class request.
    pub fn write_data(&mut self, src: &[u8]) -> Result<()> {
        if src.len() > self.data.len() {
            return Err(Error::Unsupported);
        }
        // SAFETY: `data` is a device-visible buffer of `data.len()` bytes and
        // the length was just checked; `src` is a distinct borrow, so the
        // regions cannot overlap.
        unsafe { core::ptr::copy_nonoverlapping(src.as_ptr(), self.data.cpu(), src.len()) };
        Ok(())
    }

    /// The interrupt IN endpoint's report buffer after a completed transfer
    /// (empty if no such endpoint is configured).
    pub fn read_report(&self, len: usize) -> &[u8] {
        self.int_ep()
            .map_or(&[], |ep| self.read_report_at(ep.dci, len))
    }

    /// Report bytes from a specific interrupt-IN endpoint. A composite HID
    /// device owns one `Device`, but each interface has its own buffer/DCI.
    /// Invalid endpoints or lengths are refused rather than creating an
    /// out-of-bounds DMA slice.
    pub fn read_report_at(&self, dci: u8, len: usize) -> &[u8] {
        let Some(ep) = self.endpoint(dci) else {
            return &[];
        };
        let Some(buf) = ep.data.as_ref() else {
            return &[];
        };
        if ep.kind != EndpointKind::InterruptIn || len > usize::from(ep.mps) || len > buf.len() {
            return &[];
        }
        // SAFETY: the configured DMA buffer owns these bounded bytes. The
        // caller reads it only after the corresponding completion.
        unsafe { core::slice::from_raw_parts(buf.cpu() as *const u8, len) }
    }

    /// The packet size actually committed to the default control endpoint.
    pub fn control_max_packet_size(&self) -> u16 {
        self.ep0_mps
    }

    /// The configured endpoint at `dci`, if any.
    pub fn endpoint(&self, dci: u8) -> Option<&Endpoint<P>> {
        self.endpoints.iter().flatten().find(|ep| ep.dci == dci)
    }

    /// The configured endpoint at `dci`, mutably.
    pub fn endpoint_mut(&mut self, dci: u8) -> Option<&mut Endpoint<P>> {
        self.endpoints.iter_mut().flatten().find(|ep| ep.dci == dci)
    }

    /// The first configured endpoint of `kind`, if any. Composite callers
    /// must retain their interface's DCI and use [`Self::endpoint`] instead.
    pub fn endpoint_of(&self, kind: EndpointKind) -> Option<&Endpoint<P>> {
        self.endpoints.iter().flatten().find(|ep| ep.kind == kind)
    }

    /// The first configured endpoint of `kind`, mutably.
    pub fn endpoint_of_mut(&mut self, kind: EndpointKind) -> Option<&mut Endpoint<P>> {
        self.endpoints
            .iter_mut()
            .flatten()
            .find(|ep| ep.kind == kind)
    }

    /// The unique interrupt IN endpoint. Ambiguous composite devices return
    /// `None`; a legacy single-interface caller must not silently pick one.
    pub fn int_ep(&self) -> Option<&Endpoint<P>> {
        let mut matches = self
            .endpoints
            .iter()
            .flatten()
            .filter(|ep| ep.kind == EndpointKind::InterruptIn);
        let endpoint = matches.next()?;
        matches.next().is_none().then_some(endpoint)
    }

    /// The interrupt IN endpoint, mutably.
    pub fn int_ep_mut(&mut self) -> Option<&mut Endpoint<P>> {
        let dci = self.int_ep()?.dci;
        self.endpoint_mut(dci)
    }

    /// Install a freshly configured endpoint.  `Err` when [`MAX_ENDPOINTS`]
    /// is already taken — refused, never silently replacing a live endpoint
    /// whose ring the controller may still be walking.
    fn install(&mut self, ep: Endpoint<P>) -> Result<()> {
        let slot = self
            .endpoints
            .iter_mut()
            .find(|s| s.is_none())
            .ok_or(Error::Unsupported)?;
        *slot = Some(ep);
        Ok(())
    }

    /// **Correlate** a drained Transfer Event back to the bulk submission
    /// that caused it.
    ///
    /// `None` when the event is not for a configured bulk endpoint on this
    /// device, or when it names a TRB that is not that endpoint's oldest
    /// outstanding transfer — a refusal, deliberately, because handing a
    /// cookie to the wrong caller is worse than dropping a completion.
    pub fn complete_bulk(&mut self, ev: &InterruptEvent) -> Option<BulkCompletion> {
        if ev.slot_id != self.slot_id {
            return None;
        }
        let ep = self.endpoint_mut(ev.endpoint_id)?;
        if !matches!(ep.kind, EndpointKind::BulkIn | EndpointKind::BulkOut) {
            return None;
        }
        ep.retire(ev.trb, ev.residual, ev.completion_code)
    }

    /// The DCI of this device's endpoint of `kind`, if configured — the
    /// handle [`Controller::bulk_submit`] takes.
    pub fn dci_of(&self, kind: EndpointKind) -> Option<u8> {
        self.endpoint_of(kind).map(|ep| ep.dci)
    }

    /// The highest DCI in use, which the slot context's Context Entries field
    /// must name (xHCI 6.2.2): EP0 is DCI 1, so a device with no configured
    /// endpoint still reports 1.
    fn highest_dci(&self) -> u8 {
        self.endpoints
            .iter()
            .flatten()
            .map(|ep| ep.dci)
            .max()
            .unwrap_or(1)
    }
}

/// A USB SETUP packet's fields (the 8-byte immediate a Setup Stage TRB
/// carries).
#[derive(Clone, Copy, Debug)]
pub struct Setup {
    /// `bmRequestType`.
    pub request_type: u8,
    /// `bRequest`.
    pub request: u8,
    /// `wValue`.
    pub value: u16,
    /// `wIndex`.
    pub index: u16,
}

/// A control transfer in flight on a device's EP0 — what
/// [`Controller::control_submit`] hands back so the drain that later sees
/// its Transfer Events can claim them.
///
/// The TRB addresses are the correlation key, as they are for the
/// synchronous form and for a bulk endpoint's queue: an event naming any
/// other TRB is somebody else's, however plausible its slot and endpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControlSubmission {
    slot_id: u8,
    setup: u64,
    data: Option<u64>,
    status: u64,
}

/// What one Transfer Event says about a [`ControlSubmission`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlProgress {
    /// An intermediate stage reported success; the Status stage is still
    /// owed.
    Pending,
    /// The Status stage completed: the device accepted the request.
    Complete,
    /// A stage failed with this completion code (xHCI 1.2 Table 6-90). The
    /// TD is over, and EP0 is halted on every code that halts an endpoint
    /// (a protocol STALL among them), so the caller recovers it with
    /// [`Controller::reset_endpoint_at`] before the next request.
    Failed(u8),
}

impl ControlSubmission {
    /// Classify `event` against this submission; `None` when the event is
    /// not one of its stages.
    pub fn progress(&self, event: &InterruptEvent) -> Option<ControlProgress> {
        let ours = event.slot_id == self.slot_id
            && event.endpoint_id == 1
            && (event.trb == self.setup || Some(event.trb) == self.data || event.trb == self.status);
        if !ours {
            return None;
        }
        Some(match event.completion_code {
            // Status is its own TD: success there with residue is a
            // controller saying something this request cannot mean.
            code if code == event::CompletionCode::Success as u8 && event.trb == self.status => {
                if event.residual == 0 {
                    ControlProgress::Complete
                } else {
                    ControlProgress::Failed(code)
                }
            }
            code if code == event::CompletionCode::Success as u8 => ControlProgress::Pending,
            code => ControlProgress::Failed(code),
        })
    }
}

/// Completion Code 19, Context State Error (xHCI 6.4.5) — the controller
/// refusing a command because the endpoint or slot is not in the state the
/// command requires. Named because [`Controller::reset_endpoint`] tolerates
/// exactly this one and no other.
const CC_CONTEXT_STATE_ERROR: u8 = 19;

/// Standard requests + descriptor read helpers.
impl<R: Regs64, P: DmaPool> Controller<R, P> {
    /// Dwords per context entry — 16 for 64-byte contexts (HCCPARAMS1.CSZ),
    /// else 8.  The one number that scales the input-context layout.
    fn ctx_dwords(&self) -> usize {
        self.caps().ctx_bytes() / 4
    }

    /// Enable a device slot, returning its id.
    pub fn enable_slot(
        &mut self,
        regs: &R,
        clock: fn() -> Option<u64>,
        pause: impl FnMut(),
    ) -> Result<u8> {
        let cc = self.command(regs, command::EnableSlot::new().into(), clock, pause)?;
        Ok(cc.slot_id())
    }

    /// **Clear a halted endpoint and resume it** — Reset Endpoint, then Set
    /// TR Dequeue Pointer, so the next doorbell is heard again.
    ///
    /// A transfer error (Babble, USB Transaction Error, Stall) transitions
    /// the endpoint to **Halted** (xHCI 4.10.2.1), and a halted endpoint
    /// **ignores its doorbell**. So a driver that merely counts the error and
    /// re-arms writes a TRB nobody will fetch and rings a bell nobody hears:
    /// the endpoint is silently dead for the rest of the boot, with the
    /// transfer ring quietly filling behind it. That is not a rare path —
    /// transaction errors rise with traffic, so it presents as a device that
    /// works and then freezes under exactly the load that matters.
    ///
    /// The dequeue pointer is republished at the ring's **current enqueue
    /// slot** with the ring's producer cycle state, rather than at whatever
    /// TRB failed. The halted TD is abandoned deliberately: its report is
    /// already lost, the controller's dequeue pointer is left pointing into
    /// the middle of a TD it will not resume, and pointing it where the
    /// driver will stage next is the one position both sides agree on. Set TR
    /// Dequeue Pointer is what makes the controller's cycle state match the
    /// ring's again — omit it and the endpoint resumes on a stale cycle bit
    /// and fetches nothing.
    pub fn reset_endpoint(
        &mut self,
        regs: &R,
        dev: &mut Device<P>,
        clock: fn() -> Option<u64>,
        pause: impl FnMut(),
    ) -> Result<()> {
        let dci = dev.int_ep().ok_or(Error::Unsupported)?.dci;
        self.reset_endpoint_at(regs, dev, dci, clock, pause)
    }

    /// Recover the named endpoint, including EP0 after an optional HID
    /// request STALL. A control protocol STALL halts EP0 in xHCI too; merely
    /// ignoring its completion code leaves every later interface request
    /// stuck (xHCI 1.2 §4.8.3, default-control-endpoint note).
    pub fn reset_endpoint_at(
        &mut self,
        regs: &R,
        dev: &mut Device<P>,
        dci: u8,
        clock: fn() -> Option<u64>,
        mut pause: impl FnMut(),
    ) -> Result<()> {
        let slot_id = dev.slot_id;
        let ring = if dci == 1 {
            &dev.ep0
        } else {
            &dev.endpoint(dci).ok_or(Error::Unsupported)?.ring
        };
        // Where the driver will stage next, and the cycle bit it will stage
        // with — read before the commands, since neither moves the ring.
        let resume = ring.slot_iova(ring.tail()).0;
        let cycle = ring.cycle();

        let mut reset = command::ResetEndpoint::new();
        reset.set_slot_id(slot_id).set_endpoint_id(dci);
        // A Context State Error here means the endpoint was not halted after
        // all — the recovery raced someone else's, or the code that brought us
        // here does not halt on this part. Nothing to clear, so carry on to
        // republish the dequeue pointer rather than refusing: the endpoint is
        // running and the caller still needs it armed.
        match self.command(regs, reset.into(), clock, &mut pause) {
            Ok(_) => {}
            Err(Error::Completion(code)) if code == CC_CONTEXT_STATE_ERROR => {}
            Err(e) => return Err(e),
        }

        let mut deq = command::SetTrDequeuePointer::new();
        deq.set_slot_id(slot_id)
            .set_endpoint_id(dci)
            .set_new_tr_dequeue_pointer(resume);
        // DCS must match the ring's producer cycle, or the endpoint resumes
        // looking for the opposite bit and fetches nothing.
        if cycle {
            deq.set_dequeue_cycle_state();
        } else {
            deq.clear_dequeue_cycle_state();
        }
        self.command(regs, deq.into(), clock, &mut pause)?;
        Ok(())
    }

    /// **Release a device slot** — Disable Slot, then retire its DCBAA entry.
    ///
    /// The counterpart to [`enable_slot`](Self::enable_slot), and the step a
    /// driver that only ever attaches does not need — which is exactly why it
    /// was missing. A slot is a bounded controller resource (`MaxSlots`), and
    /// its DCBAA entry points at a device context in a pool that never frees,
    /// so a detach that skips this leaks both: the controller keeps a slot it
    /// will not hand out again, and the pages stay charged to the row's DMA
    /// supply for the machine's uptime.
    ///
    /// The DCBAA entry is cleared **after** the command completes, never
    /// before: the controller may read the output context while retiring the
    /// slot, and zeroing a pointer it is still following is how a detach turns
    /// into a fault. Zero is the spec's "no device context" value, so the
    /// entry is left definite rather than dangling at a page the pool may
    /// later hand to someone else.
    pub fn disable_slot(
        &mut self,
        regs: &R,
        slot_id: u8,
        clock: fn() -> Option<u64>,
        pause: impl FnMut(),
    ) -> Result<()> {
        let mut cmd = command::DisableSlot::new();
        cmd.set_slot_id(slot_id);
        self.command(regs, cmd.into(), clock, pause)?;
        self.set_dcbaa_entry(slot_id, 0);
        Ok(())
    }

    /// **Address a device** on `port`: Enable Slot → build the input context
    /// (slot + EP0) → Address Device (BSR = 0, one step — the compatibility
    /// two-step dance is unnecessary for these targets).  Returns the
    /// addressed [`Device`] with a live EP0 control ring.
    pub fn address_device(
        &mut self,
        regs: &R,
        pool: &P,
        port: &Port,
        clock: fn() -> Option<u64>,
        mut pause: impl FnMut(),
    ) -> Result<Device<P>> {
        let speed = port.speed();
        let slot_id = self.enable_slot(regs, clock, &mut pause)?;

        // The device's three buffers (userland: one `xhci_device` crossing).
        let ctx = pool.alloc(0x1000).ok_or(Error::DmaAlloc)?;
        let ep0_buf = pool.alloc(0x1000).ok_or(Error::DmaAlloc)?;
        let data = pool.alloc(0x1000).ok_or(Error::DmaAlloc)?;

        // EP0 transfer ring (Link-terminated) + its dequeue for the context.
        install_link(&ep0_buf);
        let ep0_iova = ep0_buf.iova().0;
        let ep0 = SubRing::new(ep0_buf, RING_TRBS, Wrap::LinkTrb);
        let ep0_mps = speed.control_mps();

        // Output device context (zeroed by the pool); DCBAA[slot] → it.
        self.set_dcbaa_entry(slot_id, ctx.iova().0 + OUTPUT_CTX_OFFSET as u64);

        // Input context: add slot (0) + EP0 (DCI 1); the slot's speed / root
        // port / one context entry; EP0 as a Control endpoint over the fresh
        // ring.  Built through the handlers and serialized to the spec offsets
        // — CSZ-generic: the only per-size difference is the container
        // (`Input<8>` vs `Input<16>`) and the context stride.
        let page = ctx.cpu();
        let cw = self.ctx_dwords();
        let fill = |input: &mut dyn InputHandler| {
            input.control_mut().set_add_context_flag(0);
            input.control_mut().set_add_context_flag(1);
            {
                let slot = input.device_mut().slot_mut();
                slot.set_speed(speed.psi());
                slot.set_context_entries(1);
                slot.set_root_hub_port_number(port.number);
            }
            {
                let ep0c = input.device_mut().endpoint_mut(1);
                ep0c.set_endpoint_type(EndpointType::Control);
                ep0c.set_max_packet_size(ep0_mps);
                ep0c.set_tr_dequeue_pointer(ep0_iova);
                ep0c.set_dequeue_cycle_state();
                ep0c.set_error_count(3);
            }
        };
        if self.caps().csz64 {
            let mut input = Input::new_64byte();
            fill(&mut input);
            serialize_input(&input, page, cw, &[1]);
        } else {
            let mut input = Input::new_32byte();
            fill(&mut input);
            serialize_input(&input, page, cw, &[1]);
        }

        // Address Device (BSR = 0: the controller drives SET_ADDRESS).
        let mut cmd = command::AddressDevice::new();
        cmd.set_input_context_pointer(ctx.iova().0);
        cmd.set_slot_id(slot_id);
        self.command(regs, cmd.into(), clock, &mut pause)?;

        Ok(Device {
            slot_id,
            port: port.number,
            speed,
            ep0,
            ep0_mps,
            ctx,
            data,
            endpoints: [const { None }; MAX_ENDPOINTS],
        })
    }

    /// A **control transfer** on EP0: stage Setup + (Data if `len` > 0) +
    /// Status (opposite direction, IOC), then **one doorbell**.  Returns the
    /// number of bytes transferred (`len` − residual).
    ///
    /// The data direction is read out of `req.request_type` bit 7, which
    /// *is* the USB direction bit — never a second parameter that could
    /// disagree with the SETUP packet actually on the wire.  An IN transfer
    /// lands in the device's data buffer ([`Device::read_data`]); an OUT
    /// transfer sends whatever the caller staged there with
    /// [`Device::write_data`].
    pub fn control(
        &mut self,
        regs: &R,
        dev: &mut Device<P>,
        req: Setup,
        len: u16,
        clock: fn() -> Option<u64>,
        mut pause: impl FnMut(),
    ) -> Result<usize> {
        let to_host = req.request_type & 0x80 != 0;
        let ControlSubmission {
            slot_id: slot,
            setup: setup_trb,
            data: data_trb,
            status: status_trb,
        } = self.stage_control(regs, dev, req, len)?;

        let mut transferred = usize::from(len);
        let mut saw_short = false;
        let got = self.wait_event_consuming(regs, clock, &mut pause, |raw| {
            use core::ops::ControlFlow::{Break, Continue};
            let Ok(event::Allowed::TransferEvent(te)) = event::Allowed::try_from(raw) else {
                return None;
            };
            let pointer = te.trb_pointer();
            if te.slot_id() != slot
                || te.endpoint_id() != 1
                || (pointer != setup_trb && Some(pointer) != data_trb && pointer != status_trb)
            {
                return None;
            }
            Some(match te.completion_code() {
                Ok(event::CompletionCode::Success) if pointer == status_trb => {
                    // Status is its own TD: its residual is NOT the Data
                    // length (xHCI1.2b §4.11.2.2). A successful status
                    // completes the request using the earlier Data-short.
                    if te.trb_transfer_length() != 0 {
                        Break(Err(Error::Unsupported))
                    } else {
                        Break(Ok(transferred))
                    }
                }
                Ok(event::CompletionCode::ShortPacket)
                    if Some(pointer) == data_trb && to_host && !saw_short =>
                {
                    let residue = te.trb_transfer_length() as usize;
                    if residue > usize::from(len) {
                        Break(Err(Error::Unsupported))
                    } else {
                        transferred = usize::from(len) - residue;
                        saw_short = true;
                        // Consume this owned event, but await Status with
                        // the same overall deadline. Returning here would
                        // let the previous Status finish the next request.
                        Continue(())
                    }
                }
                Ok(event::CompletionCode::Success) => Continue(()),
                Ok(other) => Break(Err(Error::Completion(other as u8))),
                Err(code) => Break(Err(Error::Completion(code))),
            })
        });
        got.unwrap_or(Err(Error::Timeout))
    }

    /// A host-to-device control transfer **submitted without waiting** —
    /// the split-phase face of [`control`](Self::control), for a request
    /// whose answer nothing should stall on.
    ///
    /// A keyboard's LED report is the case that needs it. The synchronous
    /// form spends up to the bring-up budget (a second) on a device that
    /// NAKs, and on a runtime path that budget is time every *other*
    /// device's input waits. Submitted, the request costs a doorbell; its
    /// completion arrives on the event ring with every other Transfer Event
    /// and [`ControlSubmission::progress`] claims it there, while a device
    /// that never answers costs nothing but its own request.
    ///
    /// Only requests without an IN data stage: a read's bytes land in the
    /// device's single data buffer, which a later submission would
    /// overwrite before anyone read them. The data of an OUT stage must be
    /// staged first ([`Device::write_data`]). One request per device at a
    /// time is the caller's discipline — EP0 has one data buffer, and its
    /// TDs complete in order.
    pub fn control_submit(
        &mut self,
        regs: &R,
        dev: &mut Device<P>,
        req: Setup,
        len: u16,
    ) -> Result<ControlSubmission> {
        if req.request_type & 0x80 != 0 {
            return Err(Error::Unsupported);
        }
        self.stage_control(regs, dev, req, len)
    }

    /// Stage one control TD (Setup + Data when `len` > 0 + Status) on EP0
    /// and ring its doorbell once; the TRB addresses its events will name.
    fn stage_control(
        &mut self,
        regs: &R,
        dev: &mut Device<P>,
        req: Setup,
        len: u16,
    ) -> Result<ControlSubmission> {
        if usize::from(len) > dev.data.len() {
            return Err(Error::Unsupported);
        }
        // bmRequestType bit 7: 1 = device-to-host (IN).
        let to_host = req.request_type & 0x80 != 0;
        let mut setup = transfer::SetupStage::new();
        setup
            .set_request_type(req.request_type)
            .set_request(req.request)
            .set_value(req.value)
            .set_index(req.index)
            .set_length(len)
            .set_transfer_type(match (len > 0, to_host) {
                (false, _) => TransferType::No,
                (true, true) => TransferType::In,
                (true, false) => TransferType::Out,
            });
        let setup_index = dev.ep0.stage(setup.into());
        let setup_trb = dev.ep0.slot_iova(setup_index).0;

        let data_trb = if len > 0 {
            let mut data = transfer::DataStage::new();
            data.set_data_buffer_pointer(dev.data.iova().0)
                .set_trb_transfer_length(len as u32)
                .set_direction(if to_host {
                    Direction::In
                } else {
                    Direction::Out
                });
            if to_host {
                data.set_interrupt_on_short_packet();
            }
            let index = dev.ep0.stage(data.into());
            Some(dev.ep0.slot_iova(index).0)
        } else {
            None
        };

        // Status stage: opposite the data stage — OUT (bit 0) after an IN
        // data stage, IN (bit 1) after an OUT one and for a no-data
        // transfer; interrupt on completion.  (StatusStage's direction is a
        // bare bit setter, unlike DataStage's enum.)
        let mut status = transfer::StatusStage::new();
        if len > 0 && to_host {
            status.clear_direction(); // OUT
        } else {
            status.set_direction(); // IN
        }
        status.set_interrupt_on_completion();
        let status_index = dev.ep0.stage(status.into());
        let status_trb = dev.ep0.slot_iova(status_index).0;

        // One doorbell for the whole TD: the slot's doorbell, DCI 1 (EP0).
        let db = self.caps().doorbell(dev.slot_id);
        dev.ep0.ring(|_tail| regs.write32(db, 1));

        Ok(ControlSubmission {
            slot_id: dev.slot_id,
            setup: setup_trb,
            data: data_trb,
            status: status_trb,
        })
    }

    /// `GET_DESCRIPTOR` into the device's data buffer; returns the byte count
    /// transferred (read the bytes with [`Device::read_data`]).
    pub fn get_descriptor(
        &mut self,
        regs: &R,
        dev: &mut Device<P>,
        desc_type: u8,
        index: u8,
        len: u16,
        clock: fn() -> Option<u64>,
        pause: impl FnMut(),
    ) -> Result<usize> {
        let req = Setup {
            request_type: 0x80, // device-to-host, standard, device
            request: 6,         // GET_DESCRIPTOR
            value: ((desc_type as u16) << 8) | index as u16,
            index: 0,
        };
        self.control(regs, dev, req, len, clock, pause)
    }

    /// Read the device descriptor, first negotiating EP0's real packet size.
    /// xHCI 1.2b §3.3.6 / §6.2.3.3 require an eight-byte prefix and an
    /// Evaluate Context (A1 only) before longer transfers when MPS0 differs.
    /// Full-speed devices may use 8, 16, 32, or 64 bytes; the initial guess is
    /// not a substitute for this handshake.
    pub fn device_descriptor(
        &mut self,
        regs: &R,
        dev: &mut Device<P>,
        clock: fn() -> Option<u64>,
        mut pause: impl FnMut(),
    ) -> Result<DeviceDescriptor> {
        let n = self.get_descriptor(regs, dev, desc_type::DEVICE, 0, 8, clock, &mut pause)?;
        if n != 8 {
            return Err(Error::Unsupported);
        }
        let prefix = dev.read_data(n);
        if prefix[0] != DeviceDescriptor::LEN as u8 || prefix[1] != desc_type::DEVICE {
            return Err(Error::Unsupported);
        }
        let mps = dev.speed.descriptor_control_mps(prefix[7])?;
        if mps != dev.ep0_mps {
            let cw = self.ctx_dwords();
            let page = dev.ctx.cpu();
            zero_input_ctx(page);
            let fill = |input: &mut dyn InputHandler| {
                input.control_mut().set_add_context_flag(1);
                input.device_mut().endpoint_mut(1).set_max_packet_size(mps);
            };
            if self.caps().csz64 {
                let mut input = Input::new_64byte();
                fill(&mut input);
                serialize_input(&input, page, cw, &[1]);
            } else {
                let mut input = Input::new_32byte();
                fill(&mut input);
                serialize_input(&input, page, cw, &[1]);
            }
            let mut evaluate = command::EvaluateContext::new();
            evaluate
                .set_input_context_pointer(dev.ctx.iova().0)
                .set_slot_id(dev.slot_id);
            self.command(regs, evaluate.into(), clock, &mut pause)?;
            dev.ep0_mps = mps;
        }
        let n = self.get_descriptor(
            regs,
            dev,
            desc_type::DEVICE,
            0,
            DeviceDescriptor::LEN as u16,
            clock,
            &mut pause,
        )?;
        let descriptor = DeviceDescriptor::parse(dev.read_data(n)).ok_or(Error::Unsupported)?;
        if dev
            .speed
            .descriptor_control_mps(descriptor.max_packet_size0)?
            != mps
            || descriptor.num_configurations == 0
        {
            return Err(Error::Unsupported);
        }
        Ok(descriptor)
    }

    /// `SET_CONFIGURATION` (no data) — select `config` (the config
    /// descriptor's `bConfigurationValue`).
    pub fn set_configuration(
        &mut self,
        regs: &R,
        dev: &mut Device<P>,
        config: u8,
        clock: fn() -> Option<u64>,
        pause: impl FnMut(),
    ) -> Result<()> {
        let req = Setup {
            request_type: 0x00, // host-to-device, standard, device
            request: 9,         // SET_CONFIGURATION
            value: config as u16,
            index: 0,
        };
        self.control(regs, dev, req, 0, clock, pause).map(|_| ())
    }

    /// `SET_INTERFACE` (no data) — select `alternate` on `interface`.
    ///
    /// Required for any class whose data endpoints live on a non-default
    /// alternate setting: a CDC Ethernet data interface's altsetting 0 has
    /// **no endpoints at all**, so a device left on it enumerates cleanly and
    /// then cannot move a byte.
    pub fn set_interface(
        &mut self,
        regs: &R,
        dev: &mut Device<P>,
        interface: u16,
        alternate: u16,
        clock: fn() -> Option<u64>,
        pause: impl FnMut(),
    ) -> Result<()> {
        let req = Setup {
            request_type: 0x01, // host-to-device, standard, interface
            request: 11,        // SET_INTERFACE
            value: alternate,
            index: interface,
        };
        self.control(regs, dev, req, 0, clock, pause).map(|_| ())
    }

    /// `GET_DESCRIPTOR(STRING)` into the device's data buffer; returns the
    /// byte count transferred (read it with [`Device::read_data`] and decode
    /// with [`string_ascii`](crate::descriptor::string_ascii)).
    ///
    /// `langid` is the language from the index-0 LANGID table; pass
    /// [`LANGID_EN_US`](Self::LANGID_EN_US) when reading a
    /// machine-readable string whose content is language-independent, such
    /// as a CDC MAC address.
    pub fn get_string_descriptor(
        &mut self,
        regs: &R,
        dev: &mut Device<P>,
        index: u8,
        langid: u16,
        len: u16,
        clock: fn() -> Option<u64>,
        pause: impl FnMut(),
    ) -> Result<usize> {
        let req = Setup {
            request_type: 0x80, // device-to-host, standard, device
            request: 6,         // GET_DESCRIPTOR
            value: ((crate::descriptor::desc_type::STRING as u16) << 8) | index as u16,
            index: langid,
        };
        self.control(regs, dev, req, len, clock, pause)
    }

    /// The LANGID every device that ships English strings carries.
    pub const LANGID_EN_US: u16 = 0x0409;

    /// **Configure** one endpoint — the single-endpoint convenience over
    /// [`configure_endpoints`](Self::configure_endpoints), for callers whose
    /// device really has a single pipe.
    pub fn configure_endpoint(
        &mut self,
        regs: &R,
        pool: &P,
        dev: &mut Device<P>,
        ep: &EndpointDescriptor,
        clock: fn() -> Option<u64>,
        pause: impl FnMut(),
    ) -> Result<()> {
        self.configure_endpoints(regs, pool, dev, core::slice::from_ref(ep), clock, pause)
    }

    /// **Configure a set of endpoints in one command.**
    ///
    /// Allocates each endpoint's transfer ring (and, for the kinds that need
    /// one, its data buffer), builds an input context adding the slot plus
    /// **every** endpoint, and issues a single Configure Endpoint.
    ///
    /// One command, not one per endpoint, because that is what the hardware
    /// contract asks for: a CDC networking data interface's bulk IN and bulk
    /// OUT come up together, and the slot context's Context Entries field
    /// must name the highest DCI across all of them at once (xHCI 6.2.2).
    /// Adding them one at a time would publish an intermediate slot context
    /// that understates the endpoint set.
    ///
    /// Refuses — allocating nothing and issuing no command — when a
    /// descriptor names a transfer type this driver does not configure, when
    /// two descriptors collide on a DCI, or when the set would exceed
    /// [`MAX_ENDPOINTS`].  A partially configured device is not a state this
    /// returns.
    pub fn configure_endpoints(
        &mut self,
        regs: &R,
        pool: &P,
        dev: &mut Device<P>,
        eps: &[EndpointDescriptor],
        clock: fn() -> Option<u64>,
        mut pause: impl FnMut(),
    ) -> Result<()> {
        if eps.is_empty() {
            return Ok(());
        }
        // Validate the whole set before allocating anything: classify each
        // descriptor, refuse a DCI that collides with a live endpoint or with
        // another descriptor in this batch, and refuse a set that would not
        // fit.  Doing this first is what makes the refusal path allocation-free.
        let live = dev.endpoints.iter().flatten().count();
        if live + eps.len() > MAX_ENDPOINTS {
            return Err(Error::Unsupported);
        }
        let mut kinds = [EndpointKind::InterruptIn; MAX_ENDPOINTS];
        for (i, ep) in eps.iter().enumerate() {
            let dci = ep.dci();
            // The packed input area must stay below the live output context.
            // With 64-byte contexts, DCI 31 would overlap it; refuse instead
            // of corrupting the slot merely because a descriptor chose EP15.
            if dci < 2
                || (usize::from(dci) + 2) * self.ctx_dwords() * 4 > OUTPUT_CTX_OFFSET
                || dev.endpoint(dci).is_some()
                || ep.address & 0x70 != 0
                || ep.max_packet_size == 0
                || ep.max_packet_size > 1024
            {
                return Err(Error::Unsupported);
            }
            if eps[..i].iter().any(|prior| prior.dci() == dci) {
                return Err(Error::Unsupported);
            }
            kinds[i] = EndpointKind::of(ep).ok_or(Error::Unsupported)?;
        }

        // Allocate every ring and buffer up front.  A pool exhaustion here
        // leaves the device untouched — no command has been issued.
        let mut staged: [Option<Endpoint<P>>; MAX_ENDPOINTS] = [const { None }; MAX_ENDPOINTS];
        for (i, ep) in eps.iter().enumerate() {
            let ring_buf = pool.alloc(0x1000).ok_or(Error::DmaAlloc)?;
            install_link(&ring_buf);
            let data = if kinds[i].needs_buffer() {
                Some(pool.alloc(0x1000).ok_or(Error::DmaAlloc)?)
            } else {
                None
            };
            staged[i] = Some(Endpoint {
                ring: SubRing::new(ring_buf, RING_TRBS, Wrap::LinkTrb),
                dci: ep.dci(),
                mps: ep.max_packet_size,
                kind: kinds[i],
                data,
                inflight: [InFlight::default(); MAX_INFLIGHT],
                head: 0,
                tail: 0,
            });
        }

        // Rebuild the input context fresh (the Address Device flags are
        // stale): add slot (0) + every endpoint in the batch.  Context
        // Entries must cover the highest DCI the device will hold, counting
        // endpoints already configured.
        let cw = self.ctx_dwords();
        let page = dev.ctx.cpu();
        zero_input_ctx(page);

        let speed = dev.speed;
        let port = dev.port;
        let entries = eps
            .iter()
            .map(|e| e.dci())
            .chain(core::iter::once(dev.highest_dci()))
            .max()
            .unwrap_or(1);
        let fill = |input: &mut dyn InputHandler| {
            input.control_mut().set_add_context_flag(0);
            {
                let slot = input.device_mut().slot_mut();
                slot.set_speed(speed.psi());
                slot.set_context_entries(entries);
                slot.set_root_hub_port_number(port);
            }
            for (i, ep) in eps.iter().enumerate() {
                let dci = ep.dci() as usize;
                let staged = staged[i].as_ref().expect("staged above");
                input.control_mut().set_add_context_flag(dci);
                let epc = input.device_mut().endpoint_mut(dci);
                epc.set_endpoint_type(kinds[i].context());
                epc.set_max_packet_size(ep.max_packet_size);
                epc.set_tr_dequeue_pointer(staged.ring.slot_iova(0).0);
                epc.set_dequeue_cycle_state();
                epc.set_error_count(3);
                // Interval is a periodic-endpoint field; a bulk endpoint
                // leaves it zero (xHCI 6.2.3.6).
                if kinds[i] == EndpointKind::InterruptIn {
                    epc.set_interval(xhci_interval(speed, ep.interval));
                }
            }
        };
        let dcis: [u8; MAX_ENDPOINTS] =
            core::array::from_fn(|i| eps.get(i).map(|e| e.dci()).unwrap_or(0));
        if self.caps().csz64 {
            let mut input = Input::new_64byte();
            fill(&mut input);
            serialize_input(&input, page, cw, &dcis[..eps.len()]);
        } else {
            let mut input = Input::new_32byte();
            fill(&mut input);
            serialize_input(&input, page, cw, &dcis[..eps.len()]);
        }

        let mut cmd = command::ConfigureEndpoint::new();
        cmd.set_input_context_pointer(dev.ctx.iova().0);
        cmd.set_slot_id(dev.slot_id);
        self.command(regs, cmd.into(), clock, &mut pause)?;

        // The command succeeded: the endpoints are live, so publish them.
        for slot in staged.iter_mut().take(eps.len()) {
            let ep = slot.take().expect("staged above");
            dev.install(ep)?;
        }
        Ok(())
    }

    /// **Submit** one bulk transfer: stage a Normal TRB naming `iova`/`len`
    /// (interrupt-on-completion, routed to interrupter `target`), record the
    /// correlation, and ring the slot doorbell for the endpoint's DCI.
    ///
    /// `cookie` is opaque and comes back verbatim on the matching
    /// [`BulkCompletion`], so the caller — not this driver — decides what a
    /// transfer means.
    ///
    /// The buffer at `iova` must stay mapped and untouched until that
    /// completion is reaped: the controller reads or writes it
    /// asynchronously, and submit returning is **not** the transfer
    /// finishing.
    ///
    /// Refuses when the endpoint is not configured, is not a bulk endpoint,
    /// or already has [`MAX_INFLIGHT`] transfers outstanding.
    pub fn bulk_submit(
        &mut self,
        regs: &R,
        dev: &mut Device<P>,
        dci: u8,
        iova: IovaAddr,
        len: u16,
        cookie: u64,
        target: u16,
    ) -> Result<()> {
        let slot_id = dev.slot_id;
        let db = self.caps().doorbell(slot_id);
        let ep = dev.endpoint_mut(dci).ok_or(Error::Unsupported)?;
        if !matches!(ep.kind, EndpointKind::BulkIn | EndpointKind::BulkOut) {
            return Err(Error::Unsupported);
        }
        if ep.full() {
            return Err(Error::Unsupported);
        }
        let mut normal = transfer::Normal::new();
        normal
            .set_data_buffer_pointer(iova.0)
            .set_trb_transfer_length(len as u32)
            .set_interrupt_on_completion()
            .set_interrupter_target(target);
        // Stage first so the TRB's own address is known, then record the
        // correlation, and only then ring: the doorbell is what lets the
        // controller run, so an event can never arrive before its entry.
        let index = ep.ring.stage(normal.into());
        let trb = ep.ring.slot_iova(index).0;
        ep.track(trb, cookie, len as u32)?;
        ep.ring.ring(|_tail| regs.write32(db, dci as u32));
        Ok(())
    }
}

/// Serialize an input context's sub-contexts — InputControl, Slot, and each
/// endpoint in `dcis` — to `page` at CSZ-scaled offsets (`ctx_dwords` = 8 for
/// 32-byte contexts, 16 for 64-byte).  Over `&dyn InputHandler`, so one path
/// serves both `Input<8>` and `Input<16>`; the `.as_ref()` slices are 8 or
/// 16 dwords accordingly.
fn serialize_input(input: &dyn InputHandler, page: *mut u8, ctx_dwords: usize, dcis: &[u8]) {
    write_ctx(page, 0, input.control().as_ref());
    write_ctx(page, ctx_dwords, input.device().slot().as_ref());
    for &dci in dcis {
        write_ctx(
            page,
            (1 + dci as usize) * ctx_dwords,
            input.device().endpoint(dci as usize).as_ref(),
        );
    }
}

/// Write `src` dwords into the context page at `dword_off`.
///
/// The context page is **DMA-coherent host memory** (`Dma::cpu`), not MMIO,
/// and the Input Context is *quiescent* until the ConfigureEndpoint /
/// AddressDevice command executes — the controller reads it only *after* that
/// command's doorbell.  The command ring rings that doorbell behind a
/// `fence(Ordering::Release)` (`Controller::command` → `SubRing::ring`), which
/// orders every prior write here before the doorbell.  A bulk `copy_from_slice`
/// (one memcpy) is therefore correct — per-dword volatile is unnecessary for a
/// buffer the device never polls.
fn write_ctx(page: *mut u8, dword_off: usize, src: &[u32]) {
    // SAFETY: `page` is a 4 KiB context page; the input area stays well inside
    // it (even 64-byte contexts up to a high DCI are far under the output
    // context at 2048 B).  `page + dword_off·4` is u32-aligned (page is
    // 4 KiB-aligned), and `dst.len() == src.len()`.
    unsafe {
        let dst = core::slice::from_raw_parts_mut((page as *mut u32).add(dword_off), src.len());
        dst.copy_from_slice(src);
    }
}

/// Zero the input-context region before a fresh Configure Endpoint reuses the
/// page.  128 dwords = 512 B covers InputControl + slot + endpoints for a
/// boot device under either context size (64-byte: 8 contexts), and stays
/// clear of the output device context at 2048 B.
fn zero_input_ctx(page: *mut u8) {
    // SAFETY: 128 dwords = 512 B, clear of the output device context at 2048.
    // Same ordering as `write_ctx`: this quiescent DMA-coherent region is read
    // only after the ConfigureEndpoint doorbell, which the command ring's
    // release fence orders after this write — so a single `write_bytes`
    // (memset) replaces the 128-iteration dword loop.
    unsafe {
        core::ptr::write_bytes(page, 0, 512);
    }
}

/// Install a Link TRB at the ring's last slot (pointing back to slot 0,
/// toggle-cycle set, cycle = 1) — the transfer-ring equivalent of the
/// controller's command-ring link.
fn install_link<B: Dma>(buf: &B) {
    let mut link = Link::new();
    link.set_ring_segment_pointer(buf.iova().0)
        .set_toggle_cycle()
        .set_cycle_bit();
    let raw: Trb = link.into();
    // SAFETY: the last slot (RING_TRBS-1) is inside the 4 KiB ring page.
    unsafe {
        (buf.cpu() as *mut Trb)
            .add(RING_TRBS as usize - 1)
            .write_volatile(raw);
    }
}

/// Encode a USB `bInterval` into the xHCI endpoint-context Interval field
/// (125 µs × 2^Interval).  HS/SS carry a µframe exponent already
/// (`bInterval − 1`); LS/FS carry a frame count, so Interval ≈ 3 +
/// ⌊log2(bInterval)⌋ (1 ms = 8 × 125 µs = 2^3).  Approximate — the exact
/// value affects only polling cadence, not enumeration correctness; X1d
/// tunes it on hardware.
fn xhci_interval(speed: Speed, b_interval: u8) -> u8 {
    match speed {
        Speed::High | Speed::Super => b_interval.saturating_sub(1).min(15),
        _ => {
            let b = b_interval.max(1);
            let log2 = 7u32.saturating_sub(b.leading_zeros()) as u8; // ⌊log2(b)⌋, b:u8
            (3 + log2).clamp(3, 10)
        }
    }
}
