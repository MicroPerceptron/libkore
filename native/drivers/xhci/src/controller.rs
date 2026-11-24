//! The **controller** (≈ `xhci_hcd`) — bring-up + the command ring.
//!
//! Bring-up is the xHCI §4.2 sequence, mapped onto the seam and the [`kore_driver`]
//! rings: wait-CNR → halt → reset → `CONFIG` slots → DCBAA (+ scratchpad) →
//! command ring ([`SubRing`], `LinkTrb` wrap) → interrupter 0 (single-segment
//! event ring [`CompRing`]) → run.
//!
//! The command ring runs **one command in flight**: ordinary bring-up /
//! enumeration commands synchronously stage the TRB, ring doorbell 0, and
//! poll the event ring for the Command Completion whose *command TRB pointer*
//! matches the slot we staged. The readiness probe splits that same operation
//! around a real IRQ wait with an affine [`InterruptProbe`]. In both cases the
//! device echoes the command TRB's address (exactly the role NVMe's CID plays);
//! on budget exhaustion the ring is aborted via `CRCR.CA`.

use core::marker::PhantomData;

use kore_driver::Budget;
use kore_driver::ring::{CompRing, SubRing, Wrap};
use kore_driver::seam::{Dma, DmaPool, Regs64};

use xhci::ring::trb::Link;
use xhci::ring::trb::command;
use xhci::ring::trb::event::{self, CompletionCode};

use crate::regs::{
    CONFIG, CRCR, CRCR_CA, CRCR_CRR, CRCR_RCS, Caps, DCBAAP, ERDP, ERDP_EHB, ERSTBA, ERSTSZ, IMAN,
    IMAN_IE, IMAN_IP, USBCMD, USBCMD_HCRST, USBCMD_INTE, USBCMD_RS, USBSTS, USBSTS_CNR,
    USBSTS_EINT, USBSTS_HCE, USBSTS_HCH,
};
use crate::trb::Trb;
use crate::{Error, Result};

/// Ring depth — one 4 KiB page = 256 TRBs (the page never crosses the xHCI
/// 64 KiB boundary; a single segment keeps the event ring a plain modulo
/// consumer and the command ring one Link-terminated segment).
const RING_TRBS: u16 = 256;

/// How long a bring-up handshake gets: reset acknowledgement, controller-not-
/// ready, firmware handoff, an expected event on interrupter 0.
///
/// One second, which is what the xHCI specification actually asks for — §4.22.1
/// gives firmware up to a second to release ownership, and §5.4.1 bounds
/// `HCRST` and `CNR` well inside that.
///
/// This was 500 000 probes, and a probe count could not express it. Against a
/// live register the budget was over in well under a millisecond on a modern
/// core — far short of the second the spec wants, so a slow controller was
/// declared `NotReady` while still legitimately resetting. Against a
/// controller that is power-gated or not there, each probe is a stalled bus
/// cycle of tens of microseconds and the same number is minutes. The count
/// was simultaneously too small to be correct and too large to be safe.
pub(crate) const BRINGUP_BUDGET_NS: u64 = 1_000_000_000;

/// The clockless fallback for [`BRINGUP_BUDGET_NS`] — the probe count this
/// driver shipped with, still the only bound expressible before a counter is
/// calibrated.
pub(crate) const MAX_POLLS: u32 = 500_000;

/// Bus transaction shape for xHCI's architecturally 64-bit pointer registers.
///
/// The xHCI default is the specification's low-Dword-then-high-Dword sequence.
/// Native Qword MMIO is an explicit, identity-selected hardware quirk; it is
/// deliberately unrelated to `HCCPARAMS1.AC64`, which describes the addresses
/// the controller can generate for DMA, not the width of software MMIO cycles.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PointerRegisterAccess {
    /// Publish/read the low Dword first, then the high Dword.
    #[default]
    SplitDwords,
    /// Perform one naturally aligned 64-bit MMIO transaction.
    NativeQword,
}

/// Select the proven pointer-register quirk for one PCI xHCI function.
///
/// These are the four Krackan Point xHCs measured on the ASUS Vivobook from
/// #516. Keep the table exact: an AMD vendor id, or AC64 alone, is not evidence
/// that an unrelated controller requires the same non-default transaction.
pub const fn pointer_access_for_pci(vendor: u16, device: u16) -> PointerRegisterAccess {
    match (vendor, device) {
        (0x1022, 0x1118 | 0x111c | 0x111e | 0x1128) => PointerRegisterAccess::NativeQword,
        _ => PointerRegisterAccess::SplitDwords,
    }
}

/// The **command TRB device address** correlates a completion (xHCI echoes
/// it in the Command Completion event) — the driver holds the ring so it
/// knows the address it staged at.
pub struct Controller<R: Regs64, P: DmaPool> {
    caps: Caps,
    pointer_access: PointerRegisterAccess,
    /// The command ring (Link-terminated, single segment).
    cmd: SubRing<Trb, P::Buf>,
    /// Interrupter 0's event ring (single segment, modulo).
    event: CompRing<Trb, P::Buf>,
    /// The Device Context Base Address Array (written per slot at Address
    /// Device).  Retained: the device reads it for the controller's life.
    dcbaa: P::Buf,
    /// Retained bring-up DMA (the frames outlive these handles per the seam
    /// contract — the kernel seam leases+leaks, the mock leaks).
    _erst: P::Buf,
    _scratchpad_array: Option<P::Buf>,
    /// Scratchpad pages are device-owned for the controller lifetime.  The
    /// userland seam revokes a DMA sublease when its affine handle drops, so
    /// retaining these is load-bearing rather than a mock-only nicety.
    #[cfg(any(test, feature = "alloc"))]
    _scratchpads: alloc::vec::Vec<P::Buf>,
    /// A Port Status Change Event was consumed since the last
    /// [`take_port_change`](Self::take_port_change) — the topology edge
    /// (plug, unplug, reset) that gates re-discovery.  Latched here because
    /// the event itself is drained (and must be acked) in the same batch as
    /// transfer events; discarding it silently forced callers to re-discover
    /// on every wake instead.
    port_change: bool,
    /// Transfer Events consumed while waiting for a **command** completion.
    ///
    /// [`drain_pass`](Self::drain_pass) pulls every event off the ring to
    /// find the one its `match_fn` wants, and a rejected event is already
    /// gone — the ring advanced past it. Dropping those was harmless while
    /// the only commands ran during enumeration, where no endpoint is armed
    /// and no Transfer Event can exist; it becomes silent input loss the
    /// moment a command runs with live endpoints, which endpoint-halt
    /// recovery is the first path to do.
    ///
    /// Fixed-size and allocation-free: this sits on the interrupt drain.
    stashed: [InterruptEvent; STASHED_EVENTS],
    stashed_len: usize,
    /// Transfer Events dropped because [`stashed`](Self::stashed) was full.
    ///
    /// Counted rather than tolerated quietly: a non-zero value means reports
    /// were lost, and a bound that silently discards is indistinguishable
    /// from a device that went quiet.
    stashed_dropped: u32,
    handoff: Handoff,
    _r: PhantomData<R>,
}

/// Transfer Events retained across one command wait.
///
/// A command completes in microseconds, so the ring cannot accumulate many
/// reports behind it; this is sized for the burst a fast typist plus a mouse
/// can land in that window, with headroom, not for an unbounded backlog.
const STASHED_EVENTS: usize = 16;

/// A halted, reset controller whose complete DMA graph has been allocated and
/// initialized, but whose device pointers have not been published to hardware.
/// This is the type-state boundary a capability-backed host uses to publish the
/// pool's IOMMU leaf before any register write can make the controller fetch a
/// DMA-owned structure.
pub struct PreparedController<R: Regs64, P: DmaPool> {
    caps: Caps,
    pointer_access: PointerRegisterAccess,
    cmd: SubRing<Trb, P::Buf>,
    event: CompRing<Trb, P::Buf>,
    dcbaa: P::Buf,
    erst: P::Buf,
    _scratchpad_array: Option<P::Buf>,
    #[cfg(any(test, feature = "alloc"))]
    _scratchpads: alloc::vec::Vec<P::Buf>,
    handoff: Handoff,
    _r: PhantomData<R>,
}

/// What the firmware → OS handoff found and did before the first operational
/// write — facts about the controller kore inherited, carried to the owning
/// row so its status can say them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Handoff {
    /// A USB Legacy Support capability was present and firmware released
    /// ownership (xHCI §7.1.1).
    pub legacy_released: bool,
    /// What became of the Debug Capability (xHCI §7.6).
    pub dbc: DbcHandoff,
}

/// The Debug Capability at handoff.  A DbC is a DMA engine of its own, under
/// `DCCTRL.DCE`, that **HCRST does not reset**: firmware that leaves it
/// enabled hands kore a controller that keeps fetching its debug context from
/// firmware's buffers the moment Bus Master reopens — under kore's IOMMU
/// context, so the fetch is refused, and the census records one reason-6
/// read fault against the xHC on every boot.  Disabling it before the first
/// operational write is what makes the fault census kore's own.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DbcHandoff {
    /// No Debug Capability in the extended-capability chain.
    #[default]
    Absent,
    /// Present with `DCE` clear: nothing to stop.
    Idle,
    /// Present and running; `DCE` cleared and the engine observed stopped.
    Disabled,
    /// Present and running; `DCE` cleared but `DCE`/`DCR` did not read clear
    /// within the bring-up budget.  Bring-up proceeds: the IOMMU refuses its
    /// fetches, and a keyboard is worth more than a clean census.
    Stuck,
}

impl<R: Regs64, P: DmaPool> PreparedController<R, P> {
    /// Geometry decoded during preparation, available before publication.
    pub fn caps(&self) -> &Caps {
        &self.caps
    }

    /// What the firmware → OS handoff found and did.
    pub fn handoff(&self) -> Handoff {
        self.handoff
    }

    /// Publish the complete DMA graph, then enable interrupter 0 and Run/Stop,
    /// after the host has made the prepared pool device-visible. Pointer
    /// publication belongs to this post-activation transition: controllers may
    /// fetch the ERST as soon as ERSTBA is written, even while Run/Stop is
    /// clear. On a refused start, immediately close Run/Stop and mask the
    /// interrupter again before the prepared buffers can be dropped or their
    /// capability window retired.
    pub fn start(
        self,
        regs: &R,
        clock: fn() -> Option<u64>,
        mut pause: impl FnMut(),
    ) -> Result<Controller<R, P>> {
        self.publish_dma_graph(regs);
        if regs.read32(self.caps.op(USBSTS)) & USBSTS_HCE != 0 {
            return Err(Error::NotReady);
        }
        regs.write32(self.caps.intr(0, IMAN), IMAN_IE | IMAN_IP);
        regs.write32(self.caps.op(USBCMD), USBCMD_RS | USBCMD_INTE);
        if let Err(e) = Controller::<R, P>::wait(regs, clock, &mut pause, || {
            regs.read32(self.caps.op(USBSTS)) & USBSTS_HCH == 0
        }) {
            regs.write32(self.caps.intr(0, IMAN), IMAN_IP);
            let cmd = regs.read32(self.caps.op(USBCMD));
            regs.write32(self.caps.op(USBCMD), cmd & !(USBCMD_RS | USBCMD_INTE));
            let _ = Controller::<R, P>::wait(regs, clock, &mut pause, || {
                regs.read32(self.caps.op(USBSTS)) & USBSTS_HCH != 0
            });
            return Err(e);
        }

        Ok(Controller {
            caps: self.caps,
            pointer_access: self.pointer_access,
            cmd: self.cmd,
            event: self.event,
            dcbaa: self.dcbaa,
            _erst: self.erst,
            _scratchpad_array: self._scratchpad_array,
            #[cfg(any(test, feature = "alloc"))]
            _scratchpads: self._scratchpads,
            handoff: self.handoff,
            port_change: false,
            stashed: [InterruptEvent::EMPTY; STASHED_EVENTS],
            stashed_len: 0,
            stashed_dropped: 0,
            _r: PhantomData,
        })
    }

    /// The single hardware-publication point for every controller-owned DMA
    /// address prepared above. The release fence orders the initialized ring,
    /// ERST, DCBAA, and scratchpad contents before a controller can fetch any
    /// of them. ERSTBA stays last among the event-ring registers because its
    /// write may immediately trigger the first ERST fetch.
    fn publish_dma_graph(&self, regs: &R) {
        core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
        write_xhci64(
            regs,
            self.pointer_access,
            self.caps.op(DCBAAP),
            self.dcbaa.iova().0,
        );
        write_xhci64(
            regs,
            self.pointer_access,
            self.caps.op(CRCR),
            self.cmd.iova().0 | CRCR_RCS,
        );
        regs.write32(self.caps.intr(0, ERSTSZ), 1);
        write_xhci64(
            regs,
            self.pointer_access,
            self.caps.intr(0, ERDP),
            self.event.iova().0,
        );
        write_xhci64(
            regs,
            self.pointer_access,
            self.caps.intr(0, ERSTBA),
            self.erst.iova().0,
        );
    }
}

#[inline]
fn read_xhci64(regs: &impl Regs64, access: PointerRegisterAccess, off: usize) -> u64 {
    match access {
        PointerRegisterAccess::SplitDwords => regs.read64(off),
        PointerRegisterAccess::NativeQword => regs.read64_once(off),
    }
}

#[inline]
fn write_xhci64(regs: &impl Regs64, access: PointerRegisterAccess, off: usize, value: u64) {
    match access {
        PointerRegisterAccess::SplitDwords => regs.write64(off, value),
        PointerRegisterAccess::NativeQword => regs.write64_once(off, value),
    }
}

/// One affine command-completion witness held across an interrupt wait.
///
/// The command TRB pointer is xHCI's completion correlation key. Keeping it in
/// a move-only value prevents a caller from completing the same in-flight
/// command twice or accidentally accepting an unrelated event.
pub struct InterruptProbe {
    command_trb_pointer: u64,
}

/// What one non-blocking pass of [`Controller::poll_interrupt_probe`]
/// found.  The in-flight command is still the controller's until it
/// completes, so the affine witness rides the [`Pending`](Self::Pending)
/// arm back to the caller: the type makes "park again" the only way to
/// continue, and losing the witness impossible without deciding the
/// probe's fate ([`Controller::abandon_interrupt_probe`]).
pub enum ProbePoll {
    /// The exact No-op completion arrived, succeeded, and its batch was
    /// acknowledged.
    Complete,
    /// The completion is not on the ring yet — wait again with this
    /// witness.
    Pending(InterruptProbe),
    /// The correlated completion arrived carrying a non-Success
    /// completion code (xHCI 1.2 Table 6-90).  The command is finished,
    /// so the witness is consumed: nothing remains to abandon.
    Failed(u8),
}

/// One completed transfer decoded from a Transfer Event — interrupt or bulk.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InterruptEvent {
    /// Device slot that completed the transfer.
    pub slot_id: u8,
    /// Device Context Index of the endpoint.
    pub endpoint_id: u8,
    /// Bytes the controller did not transfer.
    pub residual: u32,
    /// Raw xHCI completion code (xHCI 1.2 Table 6-90).
    pub completion_code: u8,
    /// Device address of the TRB that generated this event (xHCI 6.4.2.1).
    ///
    /// The correlation key for a queued endpoint: an interrupt endpoint runs
    /// one transfer at a time and never needs it, but a bulk endpoint with
    /// several submissions outstanding has no other way to say *which* one
    /// finished.  [`Device::complete_bulk`](crate::device::Device::complete_bulk)
    /// checks it rather than trusting arrival order alone.
    pub trb: u64,
}

impl InterruptEvent {
    /// A zeroed event, for const-initialising fixed-size stashes.
    const EMPTY: Self = Self {
        slot_id: 0,
        endpoint_id: 0,
        residual: 0,
        completion_code: 0,
        trb: 0,
    };

    /// Whether the endpoint buffer contains valid report bytes. Interrupt IN
    /// commonly completes with either Success or Short Packet.
    pub fn data_completed(self) -> bool {
        self.completion_code == CompletionCode::Success as u8
            || self.completion_code == CompletionCode::ShortPacket as u8
    }
}

/// The one-page scratchpad-array ceiling: 512 × 8-byte entries fill the
/// single page [`Controller::bring_up`] allocates for the array.  Every
/// real controller sits far below it (tens on current Intel/AMD parts;
/// the architectural field maxes at 1023) — a part beyond it refuses
/// `Unsupported` rather than corrupting past the array page.
pub const MAX_SCRATCHPADS: u32 = 512;

/// DMA-pool pages [`Controller::bring_up`] draws for the **controller
/// itself**, given its `HCSPARAMS2` scratchpad demand: DCBAA + command
/// ring + event ring + ERST (one page each), plus — when the controller
/// wants scratchpads — the scratchpad array page and one page per buffer.
/// Mirrors `bring_up`'s draws exactly (the mock test pins the two
/// together); the kernel founder sizes the `role::DMA_POOL` endowment
/// from this, so a real part's scratchpad appetite can never again
/// out-eat a fixed lease mid-`bring_up`.  Per-device demand is separate:
/// enumeration draws [`DEVICE_DMA_PAGES`] for a single-interface HID device,
/// plus two pages for each additional configured HID interface.
pub fn bring_up_dma_pages(max_scratchpads: u32) -> usize {
    let n = max_scratchpads.min(MAX_SCRATCHPADS) as usize;
    4 + if n > 0 { 1 + n } else { 0 }
}

/// DMA-pool pages one single-interface HID device costs: slot/input context page,
/// EP0 transfer ring, control-data page, then for a configured HID boot
/// interface its interrupt ring and report buffer (`device.rs`'s five
/// `pool.alloc(0x1000)` draws). Composite HID costs `3 + 2 * interfaces`;
/// this compatibility constant is not the maximum per-device budget.
pub const DEVICE_DMA_PAGES: usize = 5;

impl<R: Regs64, P: DmaPool> Controller<R, P> {
    /// What the firmware → OS handoff found and did on this controller
    /// (carried from [`PreparedController::handoff`]).
    pub fn handoff(&self) -> Handoff {
        self.handoff
    }

    /// The decoded controller geometry.
    pub fn caps(&self) -> &Caps {
        &self.caps
    }

    /// Bring the controller from reset to **running**. Convenience composition
    /// of [`prepare`](Self::prepare) and [`PreparedController::start`] for
    /// hosts whose DMA pool is already device-visible (the mock and legacy
    /// first-light paths).
    pub fn bring_up(
        regs: &R,
        pool: &P,
        clock: fn() -> Option<u64>,
        mut pause: impl FnMut(),
    ) -> Result<Self> {
        Self::prepare(regs, pool, clock, &mut pause)?.start(regs, clock, pause)
    }

    /// Halt/reset the controller and allocate + initialize its retained DMA
    /// objects without publishing any DMA pointer. A capability host can
    /// allocate from a CPU-only prepared pool here, publish that pool into the
    /// device domain, and only then consume this value with
    /// [`PreparedController::start`], which performs the pointer writes.
    pub fn prepare(
        regs: &R,
        pool: &P,
        clock: fn() -> Option<u64>,
        mut pause: impl FnMut(),
    ) -> Result<PreparedController<R, P>> {
        Self::prepare_with_pointer_access(
            regs,
            pool,
            PointerRegisterAccess::SplitDwords,
            clock,
            &mut pause,
        )
    }

    /// Prepare with an explicitly selected pointer-register transaction.
    ///
    /// Callers should normally use [`prepare`](Self::prepare). A platform
    /// inventory may select a measured PCI quirk through
    /// [`pointer_access_for_pci`]; controller capability bits do not select it.
    pub fn prepare_with_pointer_access(
        regs: &R,
        pool: &P,
        pointer_access: PointerRegisterAccess,
        clock: fn() -> Option<u64>,
        mut pause: impl FnMut(),
    ) -> Result<PreparedController<R, P>> {
        let caps = Caps::read(regs);
        if caps.max_slots == 0 || caps.max_ports == 0 {
            return Err(Error::Unsupported);
        }

        // Take ownership from the firmware before touching operational
        // registers (real Intel metal owns the xHC for its USB boot keyboard;
        // QEMU / the mock advertise no legacy capability, so this is a no-op).
        let mut handoff = Self::bios_handoff(regs, &caps, clock, &mut pause)?;
        // Then stop the one DMA engine the reset below does not: a Debug
        // Capability firmware left enabled would otherwise fetch firmware's
        // debug context through kore's domain once Bus Master reopens.
        handoff.dbc = Self::quiesce_debug_capability(regs, &caps, clock, &mut pause)?;

        // Controller Not Ready must clear before any operational write.
        Self::wait(regs, clock, &mut pause, || {
            regs.read32(caps.op(USBSTS)) & USBSTS_CNR == 0
        })?;

        // Halt: clear Run/Stop, wait Halted.
        let cmd = regs.read32(caps.op(USBCMD));
        regs.write32(caps.op(USBCMD), cmd & !USBCMD_RS);
        Self::wait(regs, clock, &mut pause, || {
            regs.read32(caps.op(USBSTS)) & USBSTS_HCH != 0
        })?;

        // Reset: set HCRST, wait it self-clears and CNR clears.
        regs.write32(caps.op(USBCMD), USBCMD_HCRST);
        Self::wait(regs, clock, &mut pause, || {
            let sts = regs.read32(caps.op(USBSTS));
            regs.read32(caps.op(USBCMD)) & USBCMD_HCRST == 0 && sts & USBSTS_CNR == 0
        })?;
        if regs.read32(caps.op(USBSTS)) & USBSTS_HCE != 0 {
            return Err(Error::NotReady);
        }

        // Enable all device slots.
        regs.write32(caps.op(CONFIG), caps.max_slots as u32);

        // ── DCBAA (+ scratchpad) ────────────────────────────────────────
        let dcbaa = pool.alloc(0x1000).ok_or(Error::DmaAlloc)?;
        let (scratchpad_array, scratchpads) = Self::setup_scratchpad(&caps, pool, &dcbaa)?;
        #[cfg(not(any(test, feature = "alloc")))]
        let _ = scratchpads;
        // ── Command ring: install the Link; start publishes CRCR ───────
        let cmd_buf = pool.alloc(0x1000).ok_or(Error::DmaAlloc)?;
        Self::install_command_link(&cmd_buf);
        let cmd = SubRing::new(cmd_buf, RING_TRBS, Wrap::LinkTrb);

        // ── Interrupter 0: single-segment event ring + ERST ─────────────
        let event_buf = pool.alloc(0x1000).ok_or(Error::DmaAlloc)?;
        let event_iova = event_buf.iova().0;
        let event = CompRing::new(event_buf, RING_TRBS);

        let erst = pool.alloc(0x1000).ok_or(Error::DmaAlloc)?;
        // ERST entry 0: { ring_base (u64), size (u16), rsvd }.
        // SAFETY: `erst` is a fresh zeroed page; entry 0 fits.
        unsafe {
            let e = erst.cpu() as *mut u32;
            e.write_volatile(event_iova as u32);
            e.add(1).write_volatile((event_iova >> 32) as u32);
            e.add(2).write_volatile(RING_TRBS as u32);
            e.add(3).write_volatile(0);
        }
        // No event-ring register is touched here. In particular, writing
        // ERSTBA makes QEMU immediately read ERST[0] even while halted. Start
        // publishes ERSTSZ/ERDP/ERSTBA only after the IOMMU leaf is live.

        Ok(PreparedController {
            caps,
            pointer_access,
            cmd,
            event,
            dcbaa,
            erst,
            _scratchpad_array: scratchpad_array,
            #[cfg(any(test, feature = "alloc"))]
            _scratchpads: scratchpads,
            handoff,
            _r: PhantomData,
        })
    }

    /// Quiesce and reset the controller before its BAR, route, and DMA
    /// authority are retired. This is the reverse of the load-bearing end of
    /// [`bring_up`](Self::bring_up): mask interrupter 0, stop Run/Stop, wait
    /// for HCHalted, then reset and wait for both HCRST and CNR to clear.
    ///
    /// A failure is returned to the owning class service. The capability host
    /// may still revoke DMA afterward for containment, but must report that as
    /// an unclean teardown rather than calling it graceful shutdown.
    pub fn quiesce(
        &mut self,
        regs: &R,
        clock: fn() -> Option<u64>,
        mut pause: impl FnMut(),
    ) -> Result<()> {
        // Disable new interrupt generation and acknowledge a pending IP.
        regs.write32(self.caps.intr(0, IMAN), IMAN_IP);

        let cmd = regs.read32(self.caps.op(USBCMD));
        regs.write32(self.caps.op(USBCMD), cmd & !(USBCMD_RS | USBCMD_INTE));
        Self::wait(regs, clock, &mut pause, || {
            regs.read32(self.caps.op(USBSTS)) & USBSTS_HCH != 0
        })?;

        regs.write32(self.caps.op(USBCMD), USBCMD_HCRST);
        Self::wait(regs, clock, &mut pause, || {
            let sts = regs.read32(self.caps.op(USBSTS));
            regs.read32(self.caps.op(USBCMD)) & USBCMD_HCRST == 0 && sts & USBSTS_CNR == 0
        })?;
        if regs.read32(self.caps.op(USBSTS)) & USBSTS_HCE != 0 {
            return Err(Error::NotReady);
        }
        Ok(())
    }

    /// Allocate scratchpad buffers if the controller wants any, wiring the
    /// array into `DCBAA[0]`.  Userland retains every affine page handle; the
    /// heapless kernel seam has process-lifetime DMA and deliberately forgets
    /// the handles after publishing their IOVAs.
    #[cfg(any(test, feature = "alloc"))]
    fn setup_scratchpad(
        caps: &Caps,
        pool: &P,
        dcbaa: &P::Buf,
    ) -> Result<(Option<P::Buf>, alloc::vec::Vec<P::Buf>)> {
        let n = caps.max_scratchpads;
        if n == 0 {
            // DCBAA[0] stays 0 (no scratchpad array).
            return Ok((None, alloc::vec::Vec::new()));
        }
        if n > MAX_SCRATCHPADS {
            // Past 512 the entries no longer fit the one array page below
            // (the architectural field reaches 1023); refuse rather than
            // write past it.
            return Err(Error::Unsupported);
        }
        let array = pool.alloc(0x1000).ok_or(Error::DmaAlloc)?;
        let mut pages = alloc::vec::Vec::new();
        pages
            .try_reserve_exact(n as usize)
            .map_err(|_| Error::DmaAlloc)?;
        for i in 0..n {
            let page = pool.alloc(0x1000).ok_or(Error::DmaAlloc)?;
            // SAFETY: `array` holds `n` u64 entries — `i < n ≤`
            // [`MAX_SCRATCHPADS`]` = 512`, and `512 · 8 = 4096` fills the
            // array page exactly (the guard above refuses beyond it).
            unsafe {
                (array.cpu() as *mut u64)
                    .add(i as usize)
                    .write_volatile(page.iova().0);
            }
            pages.push(page);
        }
        // DCBAA[0] = scratchpad array base.
        // SAFETY: `dcbaa` entry 0 is the scratchpad array pointer slot.
        unsafe {
            (dcbaa.cpu() as *mut u64).write_volatile(array.iova().0);
        }
        Ok((Some(array), pages))
    }

    #[cfg(not(any(test, feature = "alloc")))]
    fn setup_scratchpad(caps: &Caps, pool: &P, dcbaa: &P::Buf) -> Result<(Option<P::Buf>, ())> {
        let n = caps.max_scratchpads;
        if n == 0 {
            return Ok((None, ()));
        }
        if n > MAX_SCRATCHPADS {
            // One array page holds 512 entries — refuse past it (see the
            // affine arm above).
            return Err(Error::Unsupported);
        }
        let array = pool.alloc(0x1000).ok_or(Error::DmaAlloc)?;
        for i in 0..n {
            let page = pool.alloc(0x1000).ok_or(Error::DmaAlloc)?;
            unsafe {
                (array.cpu() as *mut u64)
                    .add(i as usize)
                    .write_volatile(page.iova().0);
            }
            // The heapless kernel first-light seam has no capability teardown
            // and its DMA allocation is boot-lifetime. Forgetting is explicit
            // here; the affine userland build takes the retaining arm above.
            core::mem::forget(page);
        }
        unsafe {
            (dcbaa.cpu() as *mut u64).write_volatile(array.iova().0);
        }
        Ok((Some(array), ()))
    }

    /// Write the Link TRB into the command ring's last slot: segment pointer
    /// back to slot 0, toggle-cycle set, cycle = 1 (the initial PCS, so the
    /// controller follows it on the first wrap).
    fn install_command_link(buf: &P::Buf) {
        let mut link = Link::new();
        link.set_ring_segment_pointer(buf.iova().0)
            .set_toggle_cycle()
            .set_cycle_bit();
        let raw: Trb = link.into();
        // SAFETY: the last slot (RING_TRBS-1) is inside the 4 KiB page.
        unsafe {
            (buf.cpu() as *mut Trb)
                .add(RING_TRBS as usize - 1)
                .write_volatile(raw);
        }
    }

    /// Issue one command TRB and wait for its completion (one in flight).
    /// The staged TRB's device address correlates the Command Completion
    /// event; other events drained along the way are ignored (X1a has none
    /// but ports arrive here in X1b).
    pub fn command(
        &mut self,
        regs: &R,
        trb: Trb,
        clock: fn() -> Option<u64>,
        mut pause: impl FnMut(),
    ) -> Result<event::CommandCompletion> {
        let want = self.submit_command(regs, trb);
        self.complete_command(regs, want, clock, &mut pause)
    }

    /// Submit a deterministic No-op command whose completion must arrive
    /// through the controller's enabled interrupter. The returned affine
    /// witness is consumed by [`complete_interrupt_probe`](Self::complete_interrupt_probe)
    /// after the owning service wakes on its remapped MSI-family route.
    pub fn arm_interrupt_probe(&mut self, regs: &R) -> InterruptProbe {
        let trb: Trb = command::Noop::new().into();
        InterruptProbe {
            command_trb_pointer: self.submit_command(regs, trb),
        }
    }

    /// **One** non-blocking drain pass for the exact No-op completion
    /// submitted by [`arm_interrupt_probe`](Self::arm_interrupt_probe)
    /// — the poll half of park-with-deadline-then-poll-once.
    ///
    /// It burns no budget and never spins: the *caller* owns the
    /// waiting, by parking on its interrupt route
    /// (`kore-rt`'s `verbs::wait_interrupt_until`), and calls this once
    /// per wake.  That is what makes readiness both honest and bounded —
    /// the CPU is yielded while the completion is outstanding, and a
    /// machine whose delivery path never fires still reaches its
    /// deadline with the controller's registers intact to report
    /// ([`interrupt_probe_detail`](Self::interrupt_probe_detail)).
    ///
    /// Readiness therefore proves the xHCI event-ring completion; a
    /// wake that arrived by route delivery *also* proves the remapped
    /// MSI path, which the caller distinguishes by its park's outcome.
    ///
    /// A pass that finds nothing acknowledges an outstanding
    /// interrupter-0 `IMAN.IP` for the same reason
    /// [`drain_interrupts`](Self::drain_interrupts) does: an unacked IP
    /// suppresses **every future interrupt** on real silicon (xHCI
    /// 4.17.5), which would turn one raced wake into a permanently
    /// delivery-dead probe.
    pub fn poll_interrupt_probe(&mut self, regs: &R, probe: InterruptProbe) -> ProbePoll {
        let want = probe.command_trb_pointer;
        let got = self.drain_pass(regs, |raw| match event::Allowed::try_from(raw) {
            Ok(event::Allowed::CommandCompletion(cc)) if cc.command_trb_pointer() == want => {
                Some(cc)
            }
            _ => None,
        });
        match got {
            Some(cc) => match cc.completion_code() {
                Ok(CompletionCode::Success) => ProbePoll::Complete,
                Ok(other) => ProbePoll::Failed(other as u8),
                Err(code) => ProbePoll::Failed(code),
            },
            None => {
                if regs.read32(self.caps.intr(0, IMAN)) & IMAN_IP != 0 {
                    self.ack_events(regs);
                }
                ProbePoll::Pending(probe)
            }
        }
    }

    /// Give up on an outstanding probe: consume the affine witness and
    /// **abort the command ring** (`CRCR.CA`), the same hygiene the
    /// synchronous [`command`](Self::command) path applies when its
    /// budget runs out.  Without it a controller that never completed
    /// the No-op is left with the command still in flight, and the next
    /// command staged behind it inherits the stall.
    pub fn abandon_interrupt_probe(
        &mut self,
        regs: &R,
        probe: InterruptProbe,
        clock: fn() -> Option<u64>,
        mut pause: impl FnMut(),
    ) {
        let _ = probe;
        self.abort(regs, clock, &mut pause);
    }

    /// Post-mortem register snapshot for a FAILED readiness probe — the
    /// diagnostic a serial-less service packs into its NACK payload's
    /// high bits (one `u64` is the whole channel it has):
    ///
    /// - `[15:0]` — `USBSTS` low half.  The tells: HSE (bit 2) = Host
    ///   System Error; HCH (bit 0) = halted; EINT (bit 3) = it DID post an
    ///   event at some point; CNR (bit 11) = never became ready; HCE (bit
    ///   12) = controller-fatal. HCE can also report a DMA-access failure
    ///   (QEMU uses it when a controller DMA read/write fails), so neither
    ///   status bit alone proves an IOMMU translation refusal; correlate it
    ///   with the IOMMU fault log.
    /// - `[16]` — `CRCR.CRR`: the command ring engine reports running.
    /// - `[17]` — interrupter-0 `IMAN.IP`: an interrupt is pending.
    ///
    /// All zero across `[17:0]` except HCH reads as "the controller is
    /// alive, idle, and simply never executed the command" — which
    /// points at the command-ring DMA fetch rather than the event path.
    pub fn interrupt_probe_detail(&self, regs: &R) -> u64 {
        let sts = (regs.read32(self.caps.op(USBSTS)) & 0xFFFF) as u64;
        // CRR is bit 3 of CRCR's LOW dword — reachable through the
        // 32-bit register face; the pointer half stays unread.
        let crr = ((regs.read32(self.caps.op(CRCR)) as u64) & CRCR_CRR) << 13;
        let ip = ((regs.read32(self.caps.intr(0, IMAN)) & IMAN_IP) as u64) << 17;
        sts | crr | ip
    }

    /// Stage one command and ring doorbell zero, returning xHCI's correlation
    /// key. Exactly one command may be outstanding; the public asynchronous
    /// arm/complete pair preserves that discipline with an affine witness.
    fn submit_command(&mut self, regs: &R, trb: Trb) -> u64 {
        let slot = self.cmd.stage(trb);
        let want = self.cmd.slot_iova(slot).0;
        // Ring doorbell 0 (host / command ring), value 0.
        self.cmd
            .ring(|_tail| regs.write32(self.caps.doorbell(0), 0));
        want
    }

    /// Poll and consume the completion correlated with `want`.
    fn complete_command(
        &mut self,
        regs: &R,
        want: u64,
        clock: fn() -> Option<u64>,
        pause: &mut impl FnMut(),
    ) -> Result<event::CommandCompletion> {
        let got = self.wait_event(regs, clock, pause, |raw| {
            match event::Allowed::try_from(raw) {
                Ok(event::Allowed::CommandCompletion(cc)) if cc.command_trb_pointer() == want => {
                    Some(cc)
                }
                _ => None,
            }
        });
        match got {
            Some(cc) => match cc.completion_code() {
                Ok(CompletionCode::Success) => Ok(cc),
                Ok(other) => Err(Error::Completion(other as u8)),
                Err(code) => Err(Error::Completion(code)),
            },
            // No completion in budget: abort the command ring.
            None => {
                self.abort(regs, clock, pause);
                Err(Error::Timeout)
            }
        }
    }

    /// Poll the interrupter-0 event ring until `match_fn` claims an event
    /// (returns `Some`), acknowledging each consumed batch **once**, or the
    /// budget runs out (`None`). Rejected events still advance the ring, so
    /// they must be published through ERDP before the next pause.
    pub(crate) fn wait_event<T>(
        &mut self,
        regs: &R,
        clock: fn() -> Option<u64>,
        pause: &mut impl FnMut(),
        mut match_fn: impl FnMut([u32; 4]) -> Option<T>,
    ) -> Option<T> {
        self.wait_event_consuming(regs, clock, pause, |raw| {
            match_fn(raw).map(core::ops::ControlFlow::Break)
        })
    }

    /// An owned intermediate event may be consumed without finishing the
    /// wait. `None` still means foreign and is retained for its owner; a
    /// `Continue` is owned progress, and `Break` is final. One Budget covers
    /// all stages, including a control Data-short followed by Status.
    pub(crate) fn wait_event_consuming<T>(
        &mut self,
        regs: &R,
        clock: fn() -> Option<u64>,
        pause: &mut impl FnMut(),
        mut match_fn: impl FnMut([u32; 4]) -> Option<core::ops::ControlFlow<T>>,
    ) -> Option<T> {
        let mut budget = Budget::new(BRINGUP_BUDGET_NS, MAX_POLLS, clock);
        while budget.may_wait() {
            if let Some(t) = self.drain_pass_consuming(regs, &mut match_fn) {
                return Some(t);
            }
            pause();
        }
        self.drain_pass_consuming(regs, &mut match_fn)
    }

    /// **One** pass of the interrupter-0 event ring: drain until
    /// `match_fn` claims an event or the ring runs dry, acknowledging
    /// the consumed batch exactly **once**.  Rejected events still
    /// advance the ring, so they must be published through ERDP before
    /// the caller pauses — which is why the ack is here and not at the
    /// call sites.
    ///
    /// The shared body of the two waiting disciplines: [`wait_event`]'s
    /// bounded spin repeats it, and
    /// [`poll_interrupt_probe`](Self::poll_interrupt_probe) runs it once
    /// per route wake.  A dry pass acknowledges nothing — the quiet-path
    /// `IMAN.IP` rule belongs to the callers that park (a spin re-reads
    /// the ring in microseconds; a park may not run again for a whole
    /// deadline), and one MMIO read per spin iteration would cost the
    /// bring-up path dearly.
    ///
    /// [`wait_event`]: Self::wait_event
    fn drain_pass<T>(
        &mut self,
        regs: &R,
        mut match_fn: impl FnMut([u32; 4]) -> Option<T>,
    ) -> Option<T> {
        self.drain_pass_consuming(regs, |raw| match_fn(raw).map(core::ops::ControlFlow::Break))
    }

    fn drain_pass_consuming<T>(
        &mut self,
        regs: &R,
        mut match_fn: impl FnMut([u32; 4]) -> Option<core::ops::ControlFlow<T>>,
    ) -> Option<T> {
        let mut consumed = false;
        while let Some(raw) = self.event.poll() {
            consumed = true;
            if let Some(flow) = match_fn(raw.raw()) {
                match flow {
                    core::ops::ControlFlow::Break(t) => {
                        self.ack_events(regs);
                        return Some(t);
                    }
                    core::ops::ControlFlow::Continue(()) => continue,
                }
            }
            // Not what this wait is for — but it is off the ring now, so
            // dropping it loses it for good. A Transfer Event is somebody's
            // report and a Port Status Change is somebody's hot-plug; both
            // are retained for the drain that *is* looking for them.
            self.retain_foreign(raw.raw());
        }
        if consumed {
            self.ack_events(regs);
        }
        None
    }

    /// Keep an event this wait did not want, so the drain that does want it
    /// still sees it.
    ///
    /// Decoded here rather than stored raw: the decode is the same one
    /// [`drain_interrupts`](Self::drain_interrupts) performs, and doing it at
    /// retention keeps the stash a fixed-size `InterruptEvent` array instead
    /// of a second copy of the raw-TRB vocabulary.
    fn retain_foreign(&mut self, raw: [u32; 4]) {
        match event::Allowed::try_from(raw) {
            Ok(event::Allowed::TransferEvent(te)) => {
                if self.stashed_len < STASHED_EVENTS {
                    self.stashed[self.stashed_len] = InterruptEvent {
                        slot_id: te.slot_id(),
                        endpoint_id: te.endpoint_id(),
                        residual: te.trb_transfer_length(),
                        completion_code: match te.completion_code() {
                            Ok(code) => code as u8,
                            Err(code) => code,
                        },
                        // Carried like every other field: a stashed
                        // completion must still name the TRB that produced
                        // it, or the bulk path cannot correlate it.
                        trb: te.trb_pointer(),
                    };
                    self.stashed_len += 1;
                } else {
                    self.stashed_dropped = self.stashed_dropped.saturating_add(1);
                }
            }
            Ok(event::Allowed::PortStatusChange(_)) => self.port_change = true,
            _ => {}
        }
    }

    /// Transfer Events lost to a full stash since bring-up — see
    /// [`stashed_dropped`](Self::stashed_dropped).
    pub fn stashed_dropped(&self) -> u32 {
        self.stashed_dropped
    }

    /// Drain ready Transfer Events into `out`, acknowledging controller
    /// status, interrupter pending, and ERDP once for the consumed batch.
    /// Other event kinds are consumed and acknowledged but omitted. Completion
    /// status is retained; callers must not turn a failed transfer into bytes.
    pub fn drain_interrupts(&mut self, regs: &R, out: &mut [InterruptEvent]) -> usize {
        if out.is_empty() {
            return 0;
        }
        let mut consumed = false;
        let mut written = 0;
        // Anything a command wait had to consume comes first: it is strictly
        // older than whatever is still on the ring, and reports must reach
        // the decoder in the order the device produced them.
        while written < out.len() && written < self.stashed_len {
            out[written] = self.stashed[written];
            written += 1;
        }
        if written > 0 {
            self.stashed.copy_within(written..self.stashed_len, 0);
            self.stashed_len -= written;
        }
        while written < out.len() {
            let Some(raw) = self.event.poll() else {
                break;
            };
            consumed = true;
            match event::Allowed::try_from(raw.raw()) {
                Ok(event::Allowed::TransferEvent(te)) => {
                    out[written] = InterruptEvent {
                        slot_id: te.slot_id(),
                        endpoint_id: te.endpoint_id(),
                        residual: te.trb_transfer_length(),
                        completion_code: match te.completion_code() {
                            Ok(code) => code as u8,
                            Err(code) => code,
                        },
                        trb: te.trb_pointer(),
                    };
                    written += 1;
                }
                // The topology edge: latch it for `take_port_change` so the
                // caller can gate re-discovery on a REAL port event instead
                // of re-enumerating on every wake.
                Ok(event::Allowed::PortStatusChange(_)) => self.port_change = true,
                _ => {}
            }
        }
        if consumed {
            self.ack_events(regs);
        } else if regs.read32(self.caps.intr(0, IMAN)) & IMAN_IP != 0 {
            // An IRQ wake can race a poll path that already consumed the
            // events, leaving `IMAN.IP` (+ `ERDP.EHB`) set with nothing left
            // to read — and an unacked IP suppresses **every future
            // interrupt** on real silicon (xHCI 4.17.5: generation is edge on
            // the IP transition, and events posted under EHB do not
            // re-raise).  Ack exactly when the hardware reports an interrupt
            // outstanding, preserving the one-ack-sequence-per-batch
            // discipline on the quiet path.
            self.ack_events(regs);
        }
        written
    }

    /// Take (clear + report) the latched Port Status Change edge — whether
    /// any PSC event was consumed since the last take.  The re-discovery
    /// gate's read ([`drain_interrupts`](Self::drain_interrupts) latches it).
    pub fn take_port_change(&mut self) -> bool {
        core::mem::take(&mut self.port_change)
    }

    /// Compatibility one-shot over [`drain_interrupts`](Self::drain_interrupts).
    pub(crate) fn poll_transfer(&mut self, regs: &R) -> Option<InterruptEvent> {
        let mut event = [InterruptEvent::default(); 1];
        (self.drain_interrupts(regs, &mut event) == 1).then_some(event[0])
    }

    /// **Create a secondary interrupter** (`index` in `1..MaxIntrs`): allocate
    /// its own event ring + ERST and program its register set.  xHCI routes a
    /// transfer's completion to the interrupter its TRB targets, and MSI-X
    /// vector *n* maps to interrupter *n* — so per-core interrupters give each
    /// core a private, contention-free completion path (the SPDK-shaped
    /// scaling xHCI supports natively).  Interrupter 0 stays the controller's
    /// own (commands + enumeration); the caller owns and polls each secondary.
    pub fn create_interrupter(
        &self,
        regs: &R,
        pool: &P,
        index: u16,
    ) -> Result<Interrupter<P::Buf>> {
        if index == 0 || index >= self.caps.max_intrs {
            return Err(Error::Unsupported);
        }
        let event_buf = pool.alloc(0x1000).ok_or(Error::DmaAlloc)?;
        let event_iova = event_buf.iova().0;
        let event = CompRing::new(event_buf, RING_TRBS);

        let erst = pool.alloc(0x1000).ok_or(Error::DmaAlloc)?;
        // SAFETY: fresh zeroed page; ERST entry 0 fits.
        unsafe {
            let e = erst.cpu() as *mut u32;
            e.write_volatile(event_iova as u32);
            e.add(1).write_volatile((event_iova >> 32) as u32);
            e.add(2).write_volatile(RING_TRBS as u32);
            e.add(3).write_volatile(0);
        }
        regs.write32(self.caps.intr(index, ERSTSZ), 1);
        write_xhci64(
            regs,
            self.pointer_access,
            self.caps.intr(index, ERDP),
            event_iova,
        );
        write_xhci64(
            regs,
            self.pointer_access,
            self.caps.intr(index, ERSTBA),
            erst.iova().0,
        );
        regs.write32(self.caps.intr(index, IMAN), IMAN_IE | IMAN_IP);

        Ok(Interrupter {
            index,
            event,
            status_off: self.caps.op(USBSTS),
            iman_off: self.caps.intr(index, IMAN),
            erdp_off: self.caps.intr(index, ERDP),
            pointer_access: self.pointer_access,
            _erst: erst,
        })
    }

    /// Write `ctx_iova` into `DCBAA[slot]` — the slot's output device context
    /// pointer, installed at Address Device.
    pub(crate) fn set_dcbaa_entry(&self, slot: u8, ctx_iova: u64) {
        // SAFETY: `dcbaa` holds `max_slots + 1` u64 entries; `slot ≤
        // max_slots`, and one page covers the array for the spec max.
        unsafe {
            (self.dcbaa.cpu() as *mut u64)
                .add(slot as usize)
                .write_volatile(ctx_iova);
        }
    }

    /// Acknowledge the drained event batch in the xHCI-defined order:
    /// controller `USBSTS.EINT` (RW1C), interrupter `IMAN.IP` (RW1C while
    /// preserving IE), then ERDP with EHB. One sequence per batch.
    fn ack_events(&self, regs: &R) {
        acknowledge_interrupt(
            regs,
            self.pointer_access,
            self.caps.op(USBSTS),
            self.caps.intr(0, IMAN),
            self.caps.intr(0, ERDP),
            self.event.head_iova().0,
        );
    }

    /// Abort the command ring (`CRCR.CA`) and wait for it to stop.
    fn abort(&self, regs: &R, clock: fn() -> Option<u64>, pause: &mut impl FnMut()) {
        let crcr = read_xhci64(regs, self.pointer_access, self.caps.op(CRCR));
        write_xhci64(
            regs,
            self.pointer_access,
            self.caps.op(CRCR),
            crcr | CRCR_CA,
        );
        let _ = Self::wait(regs, clock, pause, || {
            read_xhci64(regs, self.pointer_access, self.caps.op(CRCR)) & CRCR_CRR == 0
        });
    }

    /// Walk the extended-capability chain (xHCI §7, Table 7-1), handing
    /// `visit` each capability's id and register offset until it breaks.
    /// `Ok(true)` when it broke, `Ok(false)` when the chain ended first — or
    /// there is no xECP at all (QEMU, the mock).  A cyclic or overlong chain
    /// is refused: it is not evidence of absence, and operational registers
    /// must not be touched under an ownership state that could not be
    /// established.
    fn walk_xecp(
        regs: &R,
        caps: &Caps,
        mut visit: impl FnMut(u8, usize) -> Result<core::ops::ControlFlow<()>>,
    ) -> Result<bool> {
        if caps.xecp == 0 {
            return Ok(false);
        }
        // xECP is a *dword* offset from the register base.
        let mut off = caps.xecp as usize * 4;
        for _ in 0..256 {
            let cap = regs.read32(off);
            let id = (cap & 0xff) as u8;
            let next = ((cap >> 8) & 0xff) as usize;
            if visit(id, off)?.is_break() {
                return Ok(true);
            }
            if next == 0 {
                return Ok(false);
            }
            off += next * 4;
        }
        Err(Error::Unsupported)
    }

    /// **BIOS → OS handoff** (xHCI §7.1.1): find the USB Legacy Support cap;
    /// if present, set HC OS Owned and wait for HC BIOS Owned to clear, then
    /// quiesce SMIs in USBLEGCTLSTS.  A no-op when there is no xECP or no
    /// legacy cap (QEMU, the mock).
    fn bios_handoff(
        regs: &R,
        caps: &Caps,
        clock: fn() -> Option<u64>,
        pause: &mut impl FnMut(),
    ) -> Result<Handoff> {
        use core::ops::ControlFlow;
        // Legacy Support capability id (xHCI Table 7-1).
        const CAP_LEGACY: u8 = 1;
        const HC_BIOS_OWNED: u32 = 1 << 16;
        const HC_OS_OWNED: u32 = 1 << 24;

        let mut handoff = Handoff::default();
        Self::walk_xecp(regs, caps, |id, off| {
            if id != CAP_LEGACY {
                return Ok(ControlFlow::Continue(()));
            }
            regs.write32(off, regs.read32(off) | HC_OS_OWNED);
            let mut released = false;
            let mut budget = Budget::new(BRINGUP_BUDGET_NS, MAX_POLLS, clock);
            while budget.may_wait() {
                if regs.read32(off) & HC_BIOS_OWNED == 0 {
                    released = true;
                    break;
                }
                pause();
            }
            // The final paced interval may be the one in which firmware
            // releases ownership; sample once after it before declaring
            // the bounded handshake exhausted.
            released |= regs.read32(off) & HC_BIOS_OWNED == 0;
            if !released {
                return Err(Error::NotReady);
            }
            // USBLEGCTLSTS (next dword): zero the SMI enables, clear the
            // RW1C SMI status bits (top three) so no SMI is left pending.
            regs.write32(off + 4, 0xe000_0000);
            handoff.legacy_released = true;
            Ok(ControlFlow::Break(()))
        })?;
        Ok(handoff)
    }

    /// **Debug Capability quiesce** (xHCI §7.6): find the DbC; if firmware
    /// left it enabled, clear `DCCTRL.DCE` and wait for the engine to report
    /// stopped (`DCE` and `DCR` clear) before any operational write.  The
    /// wait is the bring-up budget; an engine that will not stop is reported
    /// as [`DbcHandoff::Stuck`] and bring-up proceeds (the IOMMU refuses its
    /// fetches, so this costs a fault-census line, not memory).
    fn quiesce_debug_capability(
        regs: &R,
        caps: &Caps,
        clock: fn() -> Option<u64>,
        pause: &mut impl FnMut(),
    ) -> Result<DbcHandoff> {
        use core::ops::ControlFlow;
        // Debug Capability id (xHCI Table 7-1) and its register block
        // (§7.6.8, dword offsets from the capability header).
        const CAP_DEBUG: u8 = 10;
        const DCCTRL: usize = 0x20;
        const DCCTRL_DCR: u32 = 1 << 0;
        const DCCTRL_HOT: u32 = 1 << 2;
        const DCCTRL_HIT: u32 = 1 << 3;
        const DCCTRL_DRC: u32 = 1 << 4;
        const DCCTRL_DCE: u32 = 1 << 31;

        let mut outcome = DbcHandoff::Absent;
        Self::walk_xecp(regs, caps, |id, off| {
            if id != CAP_DEBUG {
                return Ok(ControlFlow::Continue(()));
            }
            let ctrl = regs.read32(off + DCCTRL);
            if ctrl & DCCTRL_DCE == 0 {
                outcome = DbcHandoff::Idle;
                return Ok(ControlFlow::Break(()));
            }
            // Clear DCE alone.  HOT/HIT are write-1-to-set and DRC is
            // write-1-to-clear: writing them back as read would re-issue
            // them, so they go down as zero; LSE and the read-only fields
            // keep their values.
            regs.write32(
                off + DCCTRL,
                ctrl & !(DCCTRL_DCE | DCCTRL_HOT | DCCTRL_HIT | DCCTRL_DRC),
            );
            outcome = match Self::wait(regs, clock, &mut *pause, || {
                regs.read32(off + DCCTRL) & (DCCTRL_DCE | DCCTRL_DCR) == 0
            }) {
                Ok(()) => DbcHandoff::Disabled,
                Err(_) => DbcHandoff::Stuck,
            };
            Ok(ControlFlow::Break(()))
        })?;
        Ok(outcome)
    }

    /// Spin `cond` up to [`MAX_POLLS`] times, pacing with `pause`.  `Ok` when
    /// it holds, [`Error::NotReady`] on budget exhaustion.
    /// Poll `cond` until it holds, up to [`MAX_POLLS`] with `pause` between.
    ///
    /// The loop is [`kore_driver::wait::handshake_until`], shared with every
    /// other driver's bring-up waits.
    ///
    /// Deliberately *without* a fault predicate, even though `USBSTS.HCE`
    /// looks like one: `HCRST` resets the controller and clears `USBSTS`, so a
    /// stale HCE observed mid-reset would abort a reset that was about to
    /// succeed. [`Controller::bring_up`] therefore keeps its HCE check where
    /// it is — *after* the reset acknowledges — which is why it was written
    /// that way to begin with.
    fn wait(
        regs: &R,
        clock: fn() -> Option<u64>,
        pause: &mut impl FnMut(),
        mut cond: impl FnMut() -> bool,
    ) -> Result<()> {
        kore_driver::wait::handshake_until(
            regs,
            Budget::new(BRINGUP_BUDGET_NS, MAX_POLLS, clock),
            pause,
            |_: &R| cond(),
        )
        .is_acked()
        .then_some(())
        .ok_or(Error::NotReady)
    }
}

/// Clear the controller summary before the interrupter pending flag, then
/// publish the consumed event-ring head and clear EHB. `IMAN.IE` is preserved;
/// every other software-writable bit is reserved in the current register.
fn acknowledge_interrupt(
    regs: &impl Regs64,
    pointer_access: PointerRegisterAccess,
    status_off: usize,
    iman_off: usize,
    erdp_off: usize,
    head: u64,
) {
    regs.write32(status_off, USBSTS_EINT);
    let ie = regs.read32(iman_off) & IMAN_IE;
    regs.write32(iman_off, ie | IMAN_IP);
    write_xhci64(regs, pointer_access, erdp_off, head | ERDP_EHB);
}

/// A **secondary interrupter** (≈ `xhci_interrupter`) — its own event ring +
/// ERST, created by [`Controller::create_interrupter`].  A transfer whose TRB
/// targets this interrupter's index completes here, so the owning core polls
/// its own ring without touching interrupter 0 (or any other core's).
pub struct Interrupter<B: Dma> {
    index: u16,
    event: CompRing<Trb, B>,
    status_off: usize,
    iman_off: usize,
    /// Cached `caps.intr(index, ERDP)` — the ack register.
    erdp_off: usize,
    pointer_access: PointerRegisterAccess,
    /// Retained (the device reads it for the interrupter's life).
    _erst: B,
}

impl<B: Dma> Interrupter<B> {
    /// This interrupter's index (= its MSI-X vector).
    pub fn index(&self) -> u16 {
        self.index
    }

    /// **Poll this interrupter's event ring once** (non-blocking).  Consumes
    /// one event if present, acks ERDP (with EHB), and returns
    /// `(slot_id, endpoint_id, residual)` for a Transfer Event — the private
    /// completion path the owning core drains.
    pub fn poll<R: Regs64>(&mut self, regs: &R) -> Option<(u8, u8, u32)> {
        let raw = self.event.poll()?;
        acknowledge_interrupt(
            regs,
            self.pointer_access,
            self.status_off,
            self.iman_off,
            self.erdp_off,
            self.event.head_iova().0,
        );
        match event::Allowed::try_from(raw.raw()) {
            Ok(event::Allowed::TransferEvent(te)) => {
                Some((te.slot_id(), te.endpoint_id(), te.trb_transfer_length()))
            }
            _ => None,
        }
    }
}
