//! **The NIC backend contract** — what a hardware network driver must
//! provide, and the mapping discipline that keeps a transmit's memory alive
//! until the hardware is done with it.
//!
//! # Why a transmit does not take an address
//!
//! The obvious signature is `transmit(iova, len)`. It is wrong, and the way
//! it is wrong does not show up until hardware is involved.
//!
//! A transmit completes **asynchronously**: `transmit` returning means the
//! descriptor was posted, not that the bytes left. The controller reads the
//! buffer some time later. A bare address carries no evidence that the
//! mapping behind it still exists at that later moment — so nothing stops a
//! service from retiring a client's window between the post and the read, and
//! the failure surfaces as a device DMAing into memory that now belongs to
//! someone else.
//!
//! So the authority to transmit *from* a window is a value, not an address:
//!
//! - [`VifMapping`] owns the device mapping and counts outstanding transfers.
//! - [`TxFrame`] borrows the mapping, so it cannot outlive it — the
//!   *submit-time* half of the guarantee, checked by the compiler.
//! - [`VifMapping::retire`] refuses while transfers are outstanding and hands
//!   the lease back only once quiesced — the *completion-time* half. There is
//!   no way to get the lease out of a busy mapping, so "retire it later" is
//!   not a discipline anyone has to remember.

use core::marker::PhantomData;

use kore_abi::device::IovaAddr;

use super::addr::MacAddr;
use super::frame::{FRAME_MAX, FRAME_MIN};
use super::window::{SLOT_BYTES, TX_AREA_AT, TX_DEPTH, tx_slot_at};

/// A transmit's caller-chosen correlation token, returned verbatim on
/// completion.  Opaque here on purpose: what a transfer *means* belongs to
/// the layer that submitted it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TxCookie(pub u64);

/// How a transmit ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TxStatus {
    /// The frame reached the wire.
    Sent,
    /// The device reported an error; the backend's own code is carried so a
    /// diagnostic can name it without this enum growing a variant per part.
    Failed(u8),
}

/// A completed transmit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TxCompletion {
    /// The cookie the submission carried.
    pub cookie: TxCookie,
    /// How it ended.
    pub status: TxStatus,
}

/// A frame authorized for transmit out of a live [`VifMapping`].
///
/// The lifetime is the point: this cannot outlive the mapping it came from,
/// so a backend can never be handed an address whose mapping has already
/// gone. It is inert data otherwise — the backend reads the three fields and
/// posts them.
#[derive(Clone, Copy, Debug)]
pub struct TxFrame<'m> {
    iova: IovaAddr,
    len: u16,
    cookie: TxCookie,
    _mapping: PhantomData<&'m ()>,
}

impl TxFrame<'_> {
    /// Device-visible address of the frame's first byte.
    pub fn iova(&self) -> IovaAddr {
        self.iova
    }
    /// Frame length in bytes, header included.
    pub fn len(&self) -> u16 {
        self.len
    }
    /// Whether the frame is empty — never true for a frame this type will
    /// construct, present only for the clippy convention pair.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// The correlation token to return on completion.
    pub fn cookie(&self) -> TxCookie {
        self.cookie
    }
}

/// Why a frame could not be authorized out of a mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapFault {
    /// The slot index is outside the transmit subregion.
    ///
    /// Reachable only from a bug, not from a hostile peer: the slot is
    /// *derived* from the service's own push count, never received. It is
    /// still checked, because a derivation that drifts is exactly the kind of
    /// off-by-one that reaches hardware.
    Slot,
    /// The declared length cannot be a frame, or would leave the slot.
    /// Refused, never clamped — a clamped length transmits a truncated frame
    /// that looks valid on the wire.
    Length,
}

/// Why a mapping could not be retired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StillOutstanding {
    /// Transfers the hardware has not yet completed.
    pub outstanding: u32,
}

/// A vif's transmit subregion, mapped into the device's address space.
///
/// Generic over the lease `L` it owns so this stays hardware- and
/// runtime-neutral: the frame service parameterizes it with whatever handle
/// keeps the device mapping alive, and this module never needs to know what
/// that is.
#[derive(Debug)]
pub struct VifMapping<L> {
    lease: L,
    /// Device-visible base of the vif window's transmit slot subregion.
    tx_base: IovaAddr,
    outstanding: u32,
}

impl<L> VifMapping<L> {
    /// Adopt `lease`, whose vif window is device-visible at `window_base`.
    ///
    /// The transmit subregion's device address is derived from the window
    /// base by the same layout both sides compile in, rather than passed
    /// separately — one number to get wrong instead of two.
    pub fn new(lease: L, window_base: IovaAddr) -> Self {
        Self {
            lease,
            tx_base: IovaAddr(window_base.0 + TX_AREA_AT as u64),
            outstanding: 0,
        }
    }

    /// Transfers submitted and not yet completed.
    pub fn outstanding(&self) -> u32 {
        self.outstanding
    }

    /// The lease, borrowed — for a service that needs to consult it without
    /// giving up the mapping.
    pub fn lease(&self) -> &L {
        &self.lease
    }

    /// Authorize slot `slot`'s first `len` bytes for transmit.
    ///
    /// Private, and the only constructor of a [`TxFrame`]: bounds are checked
    /// here or nowhere.
    fn frame(&self, slot: u32, len: u16, cookie: TxCookie) -> Result<TxFrame<'_>, MapFault> {
        // The slot must be a transmit slot. `tx_slot_at` refuses out of range;
        // the offset it returns is relative to the window, and `tx_base`
        // already points at the subregion, so the address is derived from the
        // slot's index within it.
        if tx_slot_at(slot).is_none() {
            return Err(MapFault::Slot);
        }
        let len_usize = len as usize;
        if !(FRAME_MIN..=FRAME_MAX).contains(&len_usize) || len_usize > SLOT_BYTES {
            return Err(MapFault::Length);
        }
        Ok(TxFrame {
            iova: IovaAddr(self.tx_base.0 + slot as u64 * SLOT_BYTES as u64),
            len,
            cookie,
            _mapping: PhantomData,
        })
    }

    /// Submit slot `slot` through `backend`.
    ///
    /// The single place the outstanding count moves up, and it moves **only
    /// if the backend accepted the transfer** — so a refused submit cannot
    /// leave the mapping believing hardware holds a reference it does not.
    /// Bounds are checked before the backend is told anything.
    pub fn transmit<B: NicBackend>(
        &mut self,
        backend: &mut B,
        slot: u32,
        len: u16,
        cookie: TxCookie,
    ) -> Result<(), TxError<B::Error>> {
        let frame = self.frame(slot, len, cookie).map_err(TxError::Map)?;
        backend.transmit(frame).map_err(TxError::Backend)?;
        self.outstanding = self.outstanding.saturating_add(1);
        Ok(())
    }

    /// Account `n` completions reaped from the backend.
    pub fn completed(&mut self, n: u32) {
        self.outstanding = self.outstanding.saturating_sub(n);
    }

    /// Give the lease back, so the caller may unmap the window.
    ///
    /// Refuses while any transfer is outstanding, returning the mapping so
    /// the caller can drain and try again. This is the whole safety story:
    /// the lease is unreachable through any other path, so the window
    /// **cannot** be unmapped while the device might still read it.
    pub fn retire(self) -> Result<L, (Self, StillOutstanding)> {
        if self.outstanding != 0 {
            let outstanding = self.outstanding;
            return Err((self, StillOutstanding { outstanding }));
        }
        Ok(self.lease)
    }
}

/// A transmit that did not happen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TxError<E> {
    /// The frame failed its bounds check against the mapping.
    Map(MapFault),
    /// The backend refused it.
    Backend(E),
}

/// Whether the link is up, and at what rate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LinkState {
    /// No carrier.
    #[default]
    Down,
    /// Carrier present.
    Up {
        /// Negotiated rate in megabits per second, 0 when unknown.
        mbps: u32,
    },
}

impl LinkState {
    /// Whether frames can move.
    pub fn is_up(self) -> bool {
        matches!(self, Self::Up { .. })
    }
}

/// What receive steering the hardware can do for itself.
///
/// Declared by every backend from the start, even the ones that can do
/// nothing, because it is what decides whether the service must copy a
/// received frame per vif or may hand a vif's own buffer to the hardware. A
/// backend that gains steering later changes this value; nothing above it
/// changes shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Steering {
    /// Independent hardware receive queues. One means no steering.
    pub rx_queues: u8,
    /// Whether hardware can filter by destination address, so each vif's
    /// queue sees only its own frames.
    pub by_address: bool,
}

impl Steering {
    /// No hardware steering: one queue, no address filtering.
    ///
    /// The honest declaration for USB networking, where the host sees one
    /// bulk pipe carrying everything. It is what forces the service to copy
    /// per matching vif — and that copy, not a hardware filter, is what keeps
    /// one vif's frames out of another's window.
    pub const SOFTWARE: Self = Self {
        rx_queues: 1,
        by_address: false,
    };

    /// Whether the hardware can separate vifs by itself.
    pub fn separates_vifs(&self) -> bool {
        self.rx_queues > 1 && self.by_address
    }
}

/// What the hardware side has been through, device-wide.
///
/// The read-back for the completion path: without these a driver that
/// takes one interrupt per frame and one that takes one per fifty look the
/// same from above, and a link that flapped a hundred times is
/// indistinguishable from one that came up once. Every backend declares
/// them from the start, like [`Steering`], because a zero it cannot count
/// is an honest answer and a missing field is not.
///
/// Saturating, never wrapping: a counter that wrapped would read as a
/// healthier device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NicCounters {
    /// Drive cycles that found a cause latched — an interrupt the device
    /// raised and the driver serviced. An upper bound on messages
    /// delivered, exact when every message ends one park; the number to
    /// hold against frames moved to see whether interrupt moderation is
    /// doing its job.
    pub irqs: u64,
    /// Link-state changes the device reported (a link-status-change cause,
    /// a class notification). Counted where reported, so a flapping cable
    /// is legible even when the service saw the link up on every sweep.
    pub link_changes: u32,
    /// Frames the device dropped for want of a posted receive buffer.
    pub overruns: u32,
    /// Receive descriptors or transfers the device completed with an
    /// error: consumed without a frame being delivered.
    pub rx_errors: u32,
}

/// The device-wide receive mode.
///
/// A **device** property, not a vif one, and explicit for that reason. CDC
/// Ethernet models exactly this as a packet-filter mode, and Linux's CDC
/// Ethernet driver drives promiscuity through it — so per-vif addresses are
/// only *receivable* if the device has been put in a mode that admits them.
/// A service that assumed per-vif hardware address filtering would silently
/// receive nothing for its second vif.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PacketFilter {
    /// Accept every frame regardless of destination.
    pub promiscuous: bool,
    /// Accept every multicast group.
    pub all_multicast: bool,
    /// Accept frames addressed to the device's own unicast address.
    pub directed: bool,
    /// Accept broadcast.
    pub broadcast: bool,
    /// Accept the specifically-joined multicast groups.
    pub multicast: bool,
}

impl PacketFilter {
    /// One station: its own address plus broadcast.
    pub const DIRECTED: Self = Self {
        promiscuous: false,
        all_multicast: false,
        directed: true,
        broadcast: true,
        multicast: false,
    };

    /// Everything on the wire — what several vifs with distinct addresses
    /// require of a device that cannot filter per vif.
    pub const PROMISCUOUS: Self = Self {
        promiscuous: true,
        all_multicast: true,
        directed: true,
        broadcast: true,
        multicast: true,
    };
}

/// One hardware network device, reduced to the mechanism a frame service
/// needs.
///
/// Deliberately not a "network interface": there is no address assignment, no
/// filtering policy, no queueing discipline and no notion of a vif here.
/// Those are the service's, above this line. A backend moves frames, reports
/// what the link and the hardware can do, and nothing else.
pub trait NicBackend {
    /// Why an operation failed, in the backend's own vocabulary.
    type Error;

    /// The device's own hardware address.
    fn mac(&self) -> MacAddr;

    /// Current link state.
    fn link(&self) -> LinkState;

    /// What receive steering this hardware can do.
    fn steering(&self) -> Steering;

    /// Device-wide counters: interrupts serviced, link changes, receive
    /// overruns and errors. Read on every `Query`, so a client can hold
    /// them against the frames it moved.
    fn counters(&self) -> NicCounters;

    /// Put the device in a receive mode.
    fn set_packet_filter(&mut self, filter: PacketFilter) -> Result<(), Self::Error>;

    /// Post one frame for transmit.
    ///
    /// Returning `Ok` means the transfer was handed to hardware, **not** that
    /// it completed: the buffer must stay mapped and unmodified until the
    /// matching [`TxCompletion`] is reaped. That is what [`VifMapping`]
    /// enforces, and why this takes a [`TxFrame`] rather than an address.
    fn transmit(&mut self, frame: TxFrame<'_>) -> Result<(), Self::Error>;

    /// Advance link management once per drive, after completion processing
    /// and before the transmit pump, even with no frames or vifs.
    ///
    /// `None` explicitly means no calibrated monotonic clock is available;
    /// a timed link must refuse or use its declared unclocked budget, never
    /// invent elapsed time. This method must not block or drain the shared
    /// completion source again. Wired backends retain the no-op default.
    fn service(&mut self, _now_ns: Option<u64>) {}

    /// Earliest absolute monotonic deadline requiring another service call.
    /// Read after the whole sweep, since RX can change the link's deadline.
    /// The owner parks on this deadline together with its existing IRQ/request
    /// wake, so an idle link can time out without periodic polling.
    fn service_deadline(&self) -> Option<u64> {
        None
    }

    /// Publish everything [`transmit`](Self::transmit) staged since the last
    /// flush — the once-per-drive doorbell.
    ///
    /// A tail-pointer NIC batches every descriptor a drive cycle posts behind
    /// **one** register write, and this is where that write happens: the
    /// engine calls it exactly once per drive, after the transmit pump and
    /// before the receive poll, never per frame and never from
    /// [`VifMapping::transmit`]. A backend whose `transmit` already reached
    /// the hardware (a USB bulk submit rings its doorbell per transfer) keeps
    /// this default and loses nothing.
    ///
    /// Staging without flushing is therefore never a completed transmit: the
    /// `Ok` from `transmit` promises the frame is *queued*, and the mapping
    /// discipline (`VifMapping` holds the slot until the completion is
    /// reaped) covers the interval either way.
    fn flush_tx(&mut self) {}

    /// Reap completed transmits into `out`, returning how many were written.
    ///
    /// **The head of every drive cycle.** The engine calls this first, and
    /// repeats it while it fills `out`, so a backend services its
    /// completion source here — acknowledging causes, draining a shared
    /// event ring, re-arming a mask — exactly once per sweep, and
    /// [`poll_rx`](Self::poll_rx) may rely on that having happened rather
    /// than touching the source again per frame.
    fn reap_tx(&mut self, out: &mut [TxCompletion]) -> usize;

    /// Hand the next received frame to `f`, if one is ready.
    ///
    /// Called after [`reap_tx`](Self::reap_tx) in the same sweep, up to the
    /// engine's batch; it consumes what that reap (or the device's own
    /// write-back) already landed and services no completion source of its
    /// own.
    ///
    /// The frame is **borrowed** for the call rather than returned, so the
    /// classifier can fan it out to every matching vif from the backend's own
    /// buffer — one receive, N copies — without the backend having to know
    /// how many copies there will be or the service having to take ownership
    /// of a buffer it must then give back.
    ///
    /// Returns whether a frame was delivered.
    fn poll_rx(&mut self, f: &mut dyn FnMut(&[u8])) -> bool;

    /// Return consumed receive buffers to the hardware.
    ///
    /// Separate from [`poll_rx`](Self::poll_rx) so a service drains a batch
    /// and re-posts once, rather than re-posting inside the drain loop.
    fn refill_rx(&mut self) -> Result<(), Self::Error>;
}

/// Slots on the transmit ring — re-exported here because a backend sizing its
/// own queue against the service's depth should read one number.
pub const TX_SLOTS: u32 = TX_DEPTH;

/// **Lends a backend for one drive cycle.**
///
/// A frame engine is generic over *this*, not over [`NicBackend`] directly, and
/// the reason is structural rather than stylistic: a backend is a short-lived
/// borrow of hardware someone else owns. A USB Ethernet backend, for instance,
/// borrows the host controller, its registers, and the device's state together —
/// it cannot be stored, so an engine holding a `B: NicBackend` would not
/// typecheck, and an engine that *owned* the controller would make every other
/// device on it unreachable.
///
/// So the split is: the host owns the hardware and knows how to assemble a
/// backend from it; the engine borrows one per cycle and knows nothing about
/// where it came from.
pub trait NicHost {
    /// The backend this host lends, borrowing from `self`.
    type Backend<'a>: NicBackend
    where
        Self: 'a;

    /// Borrow a backend for one drive cycle.
    ///
    /// `None` when the hardware is not in a state that can serve one — a row
    /// that founded degraded, or a resource that went away. The engine treats
    /// that as a refusal, never as an empty success.
    fn backend(&mut self) -> Option<Self::Backend<'_>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::window::{RX_AREA_AT, TX_AREA_BYTES};

    /// A backend that records what it was asked to transmit and can be made
    /// to refuse, so the mapping's accounting can be observed from outside.
    struct FakeNic {
        posted: alloc::vec::Vec<(u64, u16, TxCookie)>,
        refuse: bool,
        pending: alloc::vec::Vec<TxCookie>,
        /// Doorbells rung — what a tail-pointer NIC would count.
        flushes: u32,
    }

    impl FakeNic {
        fn new() -> Self {
            Self {
                posted: alloc::vec::Vec::new(),
                refuse: false,
                pending: alloc::vec::Vec::new(),
                flushes: 0,
            }
        }
    }

    impl NicBackend for FakeNic {
        type Error = ();

        fn mac(&self) -> MacAddr {
            MacAddr([0x02, 0, 0, 0, 0, 1])
        }
        fn link(&self) -> LinkState {
            LinkState::Up { mbps: 480 }
        }
        fn steering(&self) -> Steering {
            Steering::SOFTWARE
        }
        fn counters(&self) -> NicCounters {
            NicCounters::default()
        }
        fn set_packet_filter(&mut self, _f: PacketFilter) -> Result<(), ()> {
            Ok(())
        }
        fn transmit(&mut self, frame: TxFrame<'_>) -> Result<(), ()> {
            if self.refuse {
                return Err(());
            }
            self.posted
                .push((frame.iova().0, frame.len(), frame.cookie()));
            self.pending.push(frame.cookie());
            Ok(())
        }
        fn flush_tx(&mut self) {
            self.flushes += 1;
        }
        fn reap_tx(&mut self, out: &mut [TxCompletion]) -> usize {
            let n = self.pending.len().min(out.len());
            for (slot, cookie) in out.iter_mut().zip(self.pending.drain(..n)) {
                *slot = TxCompletion {
                    cookie,
                    status: TxStatus::Sent,
                };
            }
            n
        }
        fn poll_rx(&mut self, _f: &mut dyn FnMut(&[u8])) -> bool {
            false
        }
        fn refill_rx(&mut self) -> Result<(), ()> {
            Ok(())
        }
    }

    const BASE: IovaAddr = IovaAddr(0x1_0000_0000);

    /// The doorbell is the engine's to ring, once per drive: a mapping's
    /// `transmit` stages and never flushes, however many frames it posts, so
    /// a tail-pointer NIC sees one register write per cycle rather than one
    /// per frame.
    #[test]
    fn flush_tx_is_never_implied_by_transmit() {
        let mut nic = FakeNic::new();
        let mut mapping = VifMapping::new((), BASE);
        for slot in 0..4 {
            mapping
                .transmit(&mut nic, slot, 64, TxCookie(u64::from(slot)))
                .expect("accepted");
        }
        assert_eq!(nic.posted.len(), 4);
        assert_eq!(nic.flushes, 0, "staging is not publishing");
        nic.flush_tx();
        assert_eq!(nic.flushes, 1);
    }

    /// **The correction this type exists for.** A window must not be unmapped
    /// while the device may still read it, and `retire` is the only way to
    /// get the lease back — so "unmap it later" is not a discipline anyone
    /// has to remember, it is the only reachable path.
    #[test]
    fn a_mapping_cannot_be_retired_while_a_transfer_is_outstanding() {
        let mut nic = FakeNic::new();
        let mut mapping = VifMapping::new("lease", BASE);

        mapping
            .transmit(&mut nic, 0, 64, TxCookie(1))
            .expect("submit");
        assert_eq!(mapping.outstanding(), 1);

        // The lease is unreachable while the device holds a reference.
        let mapping = match mapping.retire() {
            Err((back, still)) => {
                assert_eq!(still, StillOutstanding { outstanding: 1 });
                back
            }
            Ok(_) => panic!("retired a busy mapping"),
        };

        // Drain, account, and only then does the lease come back.
        let mut done = [TxCompletion {
            cookie: TxCookie(0),
            status: TxStatus::Sent,
        }; 4];
        let n = nic.reap_tx(&mut done);
        assert_eq!(n, 1);
        assert_eq!(done[0].cookie, TxCookie(1));

        let mut mapping = mapping;
        mapping.completed(n as u32);
        assert_eq!(mapping.outstanding(), 0);
        assert_eq!(mapping.retire().ok(), Some("lease"));
    }

    /// A refused submit must not leave the mapping believing hardware holds a
    /// reference — otherwise the window becomes permanently unretirable.
    #[test]
    fn a_refused_submit_does_not_count_as_outstanding() {
        let mut nic = FakeNic::new();
        nic.refuse = true;
        let mut mapping = VifMapping::new((), BASE);

        assert_eq!(
            mapping.transmit(&mut nic, 0, 64, TxCookie(1)),
            Err(TxError::Backend(()))
        );
        assert_eq!(mapping.outstanding(), 0, "the backend never took it");
        assert!(mapping.retire().is_ok(), "and the mapping is retirable");
    }

    /// The bounds check happens before the backend is told anything, so a bad
    /// frame never reaches hardware and never moves the count.
    #[test]
    fn bounds_are_checked_before_the_backend_hears_about_it() {
        let mut nic = FakeNic::new();
        let mut mapping = VifMapping::new((), BASE);

        // A slot past the transmit depth.
        assert_eq!(
            mapping.transmit(&mut nic, TX_DEPTH, 64, TxCookie(1)),
            Err(TxError::Map(MapFault::Slot))
        );
        // Too short to be a frame, and longer than a frame may be.
        assert_eq!(
            mapping.transmit(&mut nic, 0, (FRAME_MIN - 1) as u16, TxCookie(2)),
            Err(TxError::Map(MapFault::Length))
        );
        assert_eq!(
            mapping.transmit(&mut nic, 0, (FRAME_MAX + 1) as u16, TxCookie(3)),
            Err(TxError::Map(MapFault::Length))
        );

        assert!(nic.posted.is_empty(), "nothing reached the backend");
        assert_eq!(mapping.outstanding(), 0);
    }

    /// Every authorized frame addresses memory inside the **transmit**
    /// subregion.  A frame that reached into the receive slots would let a
    /// client transmit whatever the service had just received into it.
    #[test]
    fn every_authorized_frame_stays_inside_the_transmit_subregion() {
        let mut nic = FakeNic::new();
        let mut mapping = VifMapping::new((), BASE);

        for slot in 0..TX_DEPTH {
            mapping
                .transmit(&mut nic, slot, FRAME_MAX as u16, TxCookie(slot as u64))
                .expect("in range");
        }

        let tx_lo = BASE.0 + TX_AREA_AT as u64;
        let tx_hi = tx_lo + TX_AREA_BYTES as u64;
        let rx_lo = BASE.0 + RX_AREA_AT as u64;
        for (iova, len, _) in &nic.posted {
            assert!(*iova >= tx_lo, "before the transmit area");
            assert!(
                iova + *len as u64 <= tx_hi,
                "a frame ran past the transmit area"
            );
            assert!(iova + *len as u64 <= rx_lo, "reached into receive slots");
        }
        // Slots are distinct and ordered — the derivation, observed.
        for w in nic.posted.windows(2) {
            assert_eq!(w[1].0 - w[0].0, SLOT_BYTES as u64);
        }
    }

    /// Completion accounting saturates rather than wrapping: a duplicated or
    /// spurious completion must not underflow the count into "billions
    /// outstanding", which would wedge the mapping forever.
    #[test]
    fn completion_accounting_saturates() {
        let mut mapping = VifMapping::new((), BASE);
        mapping.completed(5);
        assert_eq!(mapping.outstanding(), 0);
        assert!(mapping.retire().is_ok());
    }

    /// A backend with no steering forces the service to separate vifs itself.
    #[test]
    fn software_steering_declares_that_it_separates_nothing() {
        assert!(!Steering::SOFTWARE.separates_vifs());
        assert!(
            Steering {
                rx_queues: 4,
                by_address: true
            }
            .separates_vifs()
        );
        assert!(
            !Steering {
                rx_queues: 4,
                by_address: false
            }
            .separates_vifs(),
            "queues without address filtering do not separate vifs"
        );
    }
}
