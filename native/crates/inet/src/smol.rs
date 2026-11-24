//! **The smoltcp adapter** — [`Ifnet`] presented as a [`smoltcp::phy::Device`].
//!
//! The one place smoltcp's architecture is allowed to appear
//! (`res/INET_STACK.md` §6.1). Everything above it speaks kore's seam; this
//! module is the join, and keeping it to one file is what makes the TCP row
//! replaceable later.
//!
//! # What the join costs, measured rather than assumed
//!
//! `res/INET_MIGRATION.md` §5.3 flagged this as the thing to prototype before
//! any protocol depended on it, because smoltcp's token model and a fixed vif
//! ring might not fit. The answer, now that it is written:
//!
//! - **Transmit is zero-copy.** [`TxToken::consume`] asks for a buffer of a
//!   stated length and writes the whole frame into it, starting at the
//!   Ethernet header. That is exactly `reserve_headers(0)` followed by
//!   `commit` — smoltcp writes directly into the vif slot the controller will
//!   read, with no staging buffer in between. The seam needed no change to
//!   express it.
//!
//! - **Receive costs one copy.** [`Device::receive`] must hand back an
//!   *`RxToken` and a `TxToken` at the same time*, both borrowing the device.
//!   [`Ifnet::poll`] delivers its bytes inside a closure borrowing the whole
//!   interface, so the received frame cannot still be borrowed from the
//!   interface while a transmit slot is also held. The frame is therefore
//!   copied into caller-supplied scratch — one `memcpy` of at most an MTU per
//!   received frame.
//!
//! # The fix, when the copy is worth removing
//!
//! Split the seam into halves that borrow independently — `fn split(&mut self)
//! -> (RxHalf<'_>, TxHalf<'_>)` — and let the device hold both. [`VifIfnet`]
//! already stores its receive ring, receive slots, transmit ring and transmit
//! slots as four disjoint fields, so the split is mechanical and the borrow
//! checker will accept it; what it is not is free, because `Ifnet` is a
//! committed contract with two implementations and a test suite written
//! against it.
//!
//! That trade is deliberately **not** taken here. The copy is bounded, it is
//! on the receive path only, and no measurement yet says it is the bottleneck.
//! Recording the cost and the remedy is what this prototype was for; spending
//! the refactor before a profile justifies it would be the other mistake.
//!
//! # The other half of the join: the socket budget
//!
//! smoltcp's `SocketSet` is the second piece of its architecture that reaches
//! into this tree, and it arrives with a panic
//! (`vendor/smoltcp/src/iface/socket_set.rs:83`). [`SocketBudget`] is the
//! answer, and it lives here for the same reason the device adapter does: this
//! file is where smoltcp's shape is allowed to be, so the rest of the crate
//! never holds a `SocketSet` and therefore never holds the panic.
//!
//! [`VifIfnet`]: crate::VifIfnet

use kore_driver::net::EtherType;
use kore_driver::net::frame::HEADER_LEN;
use smoltcp::iface::{Interface, PollResult, SocketHandle, SocketSet, SocketStorage};
use smoltcp::phy::{Device, DeviceCapabilities, Medium, RxToken, TxToken};
use smoltcp::socket::{AnySocket, Socket};
use smoltcp::time::Instant;

use crate::ifnet::{Ifnet, TxSlot};

/// Scratch one adapter needs: one frame.
pub const SMOL_SCRATCH_BYTES: usize = kore_driver::net::FRAME_MAX;

/// An [`Ifnet`] presented to smoltcp.
///
/// Heapless like the rest of this crate: the caller supplies the one frame of
/// receive scratch, so the adapter allocates nothing and its footprint is
/// visible at the call site.
pub struct SmolDevice<'s, I: Ifnet> {
    net: I,
    scratch: &'s mut [u8],
    /// Bytes of `scratch` holding a frame not yet delivered. A frame polled
    /// off the interface but not yet paired with a transmit slot stays here
    /// rather than being dropped, so a full transmit ring costs latency and
    /// never a received frame.
    held: usize,
}

impl<'s, I: Ifnet> SmolDevice<'s, I> {
    /// Wrap `net`, using `scratch` for the receive staging copy.
    ///
    /// `scratch` must be at least [`SMOL_SCRATCH_BYTES`].
    pub fn new(net: I, scratch: &'s mut [u8]) -> Option<Self> {
        (scratch.len() >= SMOL_SCRATCH_BYTES).then_some(Self {
            net,
            scratch,
            held: 0,
        })
    }

    /// The wrapped interface.
    pub fn ifnet(&self) -> &I {
        &self.net
    }

    /// The wrapped interface, mutably.
    pub fn ifnet_mut(&mut self) -> &mut I {
        &mut self.net
    }

    /// Give the interface back.
    pub fn into_ifnet(self) -> I {
        self.net
    }

    /// Stage one frame off the interface if none is already held.
    fn fill(&mut self) {
        if self.held != 0 {
            return;
        }
        let Self { net, scratch, held } = self;
        net.poll(|bytes, _| {
            let n = bytes.len().min(scratch.len());
            scratch[..n].copy_from_slice(&bytes[..n]);
            *held = n;
        });
    }
}

impl<I: Ifnet> Device for SmolDevice<'_, I> {
    type RxToken<'a>
        = SmolRx<'a>
    where
        Self: 'a;
    type TxToken<'a>
        = SmolTx<'a, I>
    where
        Self: 'a;

    fn receive(&mut self, _now: Instant) -> Option<(Self::RxToken<'_>, Self::TxToken<'_>)> {
        self.fill();
        if self.held == 0 {
            return None;
        }
        let Self { net, scratch, held } = self;
        // The pair is all-or-nothing: without a transmit slot smoltcp cannot
        // answer the frame, so the frame stays staged and the next sweep
        // retries. Dropping it here would lose a received packet to a full
        // transmit ring, which is a far worse trade than a retry.
        let slot = net.reserve()?;
        let n = *held;
        *held = 0;
        Some((SmolRx { bytes: &scratch[..n] }, SmolTx { slot }))
    }

    fn transmit(&mut self, _now: Instant) -> Option<Self::TxToken<'_>> {
        self.net.reserve().map(|slot| SmolTx { slot })
    }

    fn capabilities(&self) -> DeviceCapabilities {
        let mut caps = DeviceCapabilities::default();
        caps.medium = Medium::Ethernet;
        caps.max_transmission_unit = self.net.mtu() + HEADER_LEN;
        caps
    }
}

/// One staged received frame.
pub struct SmolRx<'a> {
    bytes: &'a [u8],
}

impl RxToken for SmolRx<'_> {
    fn consume<R, F>(self, f: F) -> R
    where
        F: FnOnce(&[u8]) -> R,
    {
        f(self.bytes)
    }
}

/// One reserved transmit slot, held for smoltcp to fill.
///
/// The slot is taken when the token is *created* rather than when it is
/// consumed, because [`TxToken::consume`] has no way to report that no slot
/// was available — it must return the closure's value. Reserving up front
/// turns "the ring is full" into `transmit` answering `None`, which smoltcp
/// does handle.
pub struct SmolTx<'a, I: Ifnet + 'a> {
    slot: I::TxSlot<'a>,
}

impl<I: Ifnet> TxToken for SmolTx<'_, I> {
    fn consume<R, F>(mut self, len: usize, f: F) -> R
    where
        F: FnOnce(&mut [u8]) -> R,
    {
        // smoltcp writes the entire frame from the Ethernet header onward, so
        // the header reservation is zero and `commit`'s payload *is* the
        // frame. No staging buffer: the closure writes into the vif slot the
        // controller will read.
        let Some(buf) = self.slot.reserve_headers(0) else {
            // The slot cannot hold a frame this long. The closure still has
            // to run — its value is the return type — so it is given a
            // truncated view and the frame is not committed.
            let mut nowhere = [0u8; HEADER_LEN];
            return f(&mut nowhere);
        };
        let n = len.min(buf.len());
        let r = f(&mut buf[..n]);
        // The EtherType the descriptor declares is read back from the bytes
        // smoltcp just wrote, so the two cannot disagree.
        let ethertype = if n >= HEADER_LEN {
            EtherType::from_wire([buf[12], buf[13]])
        } else {
            EtherType(0)
        };
        let _ = self.slot.commit(n, ethertype);
        r
    }
}

// ── the socket budget ──────────────────────────────────────────────────

/// A vif's socket budget is spent.
///
/// Refused rather than panicked. `SocketSet::add` panics when a *borrowed*
/// storage slice is full (`vendor/smoltcp/src/iface/socket_set.rs:83`), and
/// borrowed is the only kind this tree has: `alloc` is off
/// (`vendor/smoltcp/PATCHES.md`), so the `ManagedSlice::Owned` arm that would
/// have grown the slice is compiled out and the panic is the *only* remaining
/// branch. A kernel service that aborts when a client reaches a resource bound
/// is a denial of service wearing a diagnostic's clothes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SocketsExhausted;

/// A [`SocketSet`] that cannot be overfilled, and whose handles cannot be
/// misread.
///
/// # Why this type exists rather than a patch
///
/// `vendor/smoltcp/PATCHES.md` ranks every modification we might want to
/// make to the vendored tree and takes none; `vendor/smoltcp/PATCHES.md`
/// records the vendoring at zero deviations and says why keeping it there is
/// the point. So the four reachable panic sites that §5.3 names are closed on
/// *our* side of the seam. This closes the three in `socket_set.rs`:
///
/// | upstream | how it is closed |
/// | --- | --- |
/// | `add` on a full borrowed slice (`:83`) | [`try_add`](Self::try_add) refuses before `add` is called |
/// | `get`/`get_mut` on a handle that is out of range, retired, or names another socket type (`:102`, `:116`) | [`try_get_mut`](Self::try_get_mut) answers `None`; it never indexes and never downcasts with `expect` |
/// | `remove` on a handle that is out of range or retired (`:127`) | [`try_remove`](Self::try_remove) answers `None` |
///
/// # Why the panic is unreachable, not merely unlikely
///
/// The set is a **private field and no `&mut SocketSet` ever leaves this
/// type** — [`poll`](Self::poll) drives the interface over it rather than
/// lending it out. `SocketSet::add` therefore has exactly one caller in the
/// tree, inside [`try_add`](Self::try_add), behind a guard that returns first.
/// That is a reachability argument about the call graph, not a claim that the
/// guard is written carefully.
///
/// # The budget is per-vif, and that is what preserves isolation
///
/// A budget is the storage slice a row borrows, declared at construction and
/// unable to grow afterwards. One [`SocketBudget`] belongs to one row, which
/// drives one [`Ifnet`] — one vif. There is no shared pool, so a holder that
/// spends its entire budget has spent only its own: the exhaustion refuses
/// *it* and reaches no other vif. A global free list would be the opposite —
/// a resource two mutually-distrusting tasks could deny each other, which is
/// precisely the isolation `crates/inet/tests/flow_isolation.rs` exists to
/// prove this stack does not have.
///
/// Per-vif rather than per-[`FlowCap`](crate::FlowCap) because a vif carries
/// exactly one [`FrameFilter`] and therefore at most one minted flow
/// ([`OpenRefusal::VifAlreadyBound`](crate::OpenRefusal::VifAlreadyBound)), so
/// the two scopes coincide today — and per-vif is the one that stays correct
/// when they stop coinciding, because the vif is what the storage is attached
/// to.
///
/// [`FrameFilter`]: kore_driver::net::FrameFilter
pub struct SocketBudget<'a> {
    set: SocketSet<'a>,
    /// Slots the borrowed storage has. Recorded at construction because
    /// `SocketSet` does not report its storage's length, and because a budget
    /// that could be read back off the thing it bounds would not be a budget.
    budget: usize,
}

impl<'a> SocketBudget<'a> {
    /// Take `storage` as one vif's entire socket budget.
    pub fn new(storage: &'a mut [SocketStorage<'a>]) -> Self {
        let budget = storage.len();
        Self {
            set: SocketSet::new(storage),
            budget,
        }
    }

    /// Slots this vif was given.
    pub const fn budget(&self) -> usize {
        self.budget
    }

    /// Slots in use.
    ///
    /// Counted rather than cached: upstream's handles carry no accessor for
    /// the index they wrap, so a side table could not be kept in step with
    /// certainty, and a count that can drift from the set is worse than a
    /// walk over at most `budget` entries.
    pub fn live(&self) -> usize {
        self.set.iter().count()
    }

    /// Slots still free.
    pub fn spare(&self) -> usize {
        self.budget - self.live()
    }

    /// Whether the next [`try_add`](Self::try_add) would be refused.
    pub fn is_full(&self) -> bool {
        self.live() >= self.budget
    }

    /// Add a socket, or refuse because this vif's budget is spent.
    pub fn try_add<T: AnySocket<'a>>(
        &mut self,
        socket: T,
    ) -> Result<SocketHandle, SocketsExhausted> {
        if self.is_full() {
            return Err(SocketsExhausted);
        }
        // Past the guard the panic branch cannot be entered: `budget` is the
        // slice's own length, so a set holding fewer than `budget` sockets has
        // a free slot for `add` to find, and `add` panics only after finding
        // none.
        Ok(self.set.add(socket))
    }

    /// This set's socket at `handle`, if it has one of that type.
    pub fn try_get<T: AnySocket<'a>>(&self, handle: SocketHandle) -> Option<&T> {
        self.set
            .iter()
            .find(|(h, _)| *h == handle)
            .and_then(|(_, s)| T::downcast(s))
    }

    /// This set's socket at `handle`, mutably.
    ///
    /// Found by walking rather than indexed, because indexing is what panics:
    /// a handle out of range indexes past the slice and a retired one reads an
    /// empty slot, and upstream answers both with a `panic!`. The walk sees
    /// only live slots of this set, so a stale handle, a handle minted by
    /// another set and a handle naming another socket type all come back
    /// `None`.
    ///
    /// What this does **not** fix: a handle is an index, so one minted by
    /// another set that happens to be in range names *this* set's socket at
    /// that index. That aliasing is upstream's model and closing it needs a
    /// lifetime brand; removing the panic does not depend on it, and no caller
    /// in this crate holds two sets.
    pub fn try_get_mut<T: AnySocket<'a>>(&mut self, handle: SocketHandle) -> Option<&mut T> {
        self.set
            .iter_mut()
            .find(|(h, _)| *h == handle)
            .and_then(|(_, s)| T::downcast_mut(s))
    }

    /// Retire the socket at `handle`, returning its slot to the budget.
    pub fn try_remove(&mut self, handle: SocketHandle) -> Option<Socket<'a>> {
        // Established before the call, so `remove`'s own two panics — an index
        // past the slice and an already-empty slot — are both ruled out.
        let present = self.set.iter().any(|(h, _)| h == handle);
        present.then(|| self.set.remove(handle))
    }

    /// Drive `iface` over `device` against these sockets.
    ///
    /// The set is driven rather than lent out: handing back a `&mut SocketSet`
    /// would hand back `SocketSet::add`, and that this type is the only route
    /// to it is the whole unreachability argument.
    pub fn poll<D: Device + ?Sized>(
        &mut self,
        iface: &mut Interface,
        device: &mut D,
        now: Instant,
    ) -> PollResult {
        iface.poll(now, device, &mut self.set)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ifnet::{MOCK_SCRATCH_BYTES, MockIfnet};
    use kore_driver::net::{FrameHead, MacAddr};

    const MINE: MacAddr = MacAddr([0x02, 0, 0, 0, 0, 0x01]);
    const PEER: MacAddr = MacAddr([0x02, 0, 0, 0, 0, 0x02]);

    #[repr(align(64))]
    struct Aligned<const N: usize>([u8; N]);

    fn frame(fill: u8, payload: usize) -> [u8; 128] {
        let mut b = [fill; 128];
        FrameHead {
            dst: PEER,
            src: MINE,
            ethertype: EtherType::IPV4,
        }
        .write(&mut b)
        .expect("header fits");
        for byte in b[HEADER_LEN..HEADER_LEN + payload].iter_mut() {
            *byte = fill;
        }
        b
    }

    /// **The A1b witness.** A frame goes out through smoltcp's `TxToken` and
    /// lands in the interface, and a frame injected into the interface comes
    /// back through smoltcp's `RxToken` — the join works in both directions.
    #[test]
    fn the_adapter_moves_a_frame_both_ways() {
        let mut ifscratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut rxscratch = Aligned([0u8; SMOL_SCRATCH_BYTES]);
        let net = MockIfnet::new(&mut ifscratch.0, MINE).expect("scratch");
        let mut dev = SmolDevice::new(net, &mut rxscratch.0).expect("scratch");

        // Out: smoltcp writes the whole frame, header included.
        let out = frame(0xa5, 46);
        let len = HEADER_LEN + 46;
        let token = dev.transmit(Instant::from_millis(0)).expect("a slot");
        token.consume(len, |buf| buf.copy_from_slice(&out[..len]));

        let (sent, ethertype) = dev.ifnet().last_sent().expect("committed");
        assert_eq!(sent, &out[..len], "the bytes reached the interface intact");
        assert_eq!(
            ethertype,
            EtherType::IPV4,
            "the descriptor's EtherType was read back from what smoltcp wrote"
        );

        // In: a frame the interface receives reaches smoltcp.
        let inbound = frame(0x5a, 46);
        dev.ifnet_mut()
            .inject(&inbound[..len], EtherType::IPV4)
            .expect("injects");

        let (rx, _tx) = dev.receive(Instant::from_millis(1)).expect("a pair");
        let seen = rx.consume(|bytes| {
            let mut copy = [0u8; 128];
            copy[..bytes.len()].copy_from_slice(bytes);
            (bytes.len(), copy)
        });
        assert_eq!(seen.0, len);
        assert_eq!(&seen.1[..len], &inbound[..len]);
    }

    /// Transmit is zero-copy: the buffer smoltcp fills *is* the interface's
    /// slot, so the bytes it writes are the bytes committed, with no staging
    /// step that could diverge.
    #[test]
    fn transmit_writes_straight_into_the_slot() {
        let mut ifscratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut rxscratch = Aligned([0u8; SMOL_SCRATCH_BYTES]);
        let net = MockIfnet::new(&mut ifscratch.0, MINE).expect("scratch");
        let mut dev = SmolDevice::new(net, &mut rxscratch.0).expect("scratch");

        let out = frame(0x3c, 60);
        let len = HEADER_LEN + 60;
        let token = dev.transmit(Instant::from_millis(0)).expect("a slot");
        let written = token.consume(len, |buf| {
            buf.copy_from_slice(&out[..len]);
            buf.len()
        });

        assert_eq!(written, len, "the closure saw exactly the length it asked for");
        assert_eq!(dev.ifnet().sent_count(), 1);
        assert_eq!(dev.ifnet().last_sent().expect("committed").0, &out[..len]);
    }

    /// A full transmit ring costs latency, never a received frame: the staged
    /// frame is held and delivered on the next sweep.  smoltcp's `receive`
    /// must hand back both tokens at once, so without this the pairing would
    /// silently drop inbound traffic whenever transmit was busy.
    #[test]
    fn a_full_transmit_ring_does_not_lose_a_received_frame() {
        let mut ifscratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut rxscratch = Aligned([0u8; SMOL_SCRATCH_BYTES]);
        let mut net = MockIfnet::new(&mut ifscratch.0, MINE).expect("scratch");
        net.set_tx_full(true);
        let mut dev = SmolDevice::new(net, &mut rxscratch.0).expect("scratch");

        let inbound = frame(0x77, 46);
        let len = HEADER_LEN + 46;
        dev.ifnet_mut()
            .inject(&inbound[..len], EtherType::IPV4)
            .expect("injects");

        assert!(
            dev.receive(Instant::from_millis(0)).is_none(),
            "no transmit slot, so no pair"
        );

        // The ring drains; the frame staged during the refusal is still there.
        dev.ifnet_mut().set_tx_full(false);
        let (rx, _tx) = dev.receive(Instant::from_millis(1)).expect("a pair now");
        let n = rx.consume(|bytes| bytes.len());
        assert_eq!(n, len, "the frame survived the full ring");
    }

    /// An idle interface yields no pair, so "nothing arrived" is
    /// distinguishable from a delivered frame.
    #[test]
    fn an_idle_interface_yields_no_pair() {
        let mut ifscratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut rxscratch = Aligned([0u8; SMOL_SCRATCH_BYTES]);
        let net = MockIfnet::new(&mut ifscratch.0, MINE).expect("scratch");
        let mut dev = SmolDevice::new(net, &mut rxscratch.0).expect("scratch");
        assert!(dev.receive(Instant::from_millis(0)).is_none());
    }

    /// The capabilities smoltcp plans against are the interface's own, so a
    /// link that carries less cannot be overrun by a stack that assumed more.
    #[test]
    fn capabilities_follow_the_interface() {
        let mut ifscratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut rxscratch = Aligned([0u8; SMOL_SCRATCH_BYTES]);
        let mut net = MockIfnet::new(&mut ifscratch.0, MINE).expect("scratch");
        net.set_mtu(900);
        let dev = SmolDevice::new(net, &mut rxscratch.0).expect("scratch");

        let caps = dev.capabilities();
        assert_eq!(caps.medium, Medium::Ethernet);
        assert_eq!(
            caps.max_transmission_unit,
            900 + HEADER_LEN,
            "smoltcp's MTU includes the Ethernet header; ours does not"
        );
    }

    // ── the socket budget ──────────────────────────────────────────────

    use smoltcp::socket::tcp;

    /// A pile of socket buffers a test draws pairs from.
    ///
    /// Flat, and split by hand, because every socket borrows its two buffers
    /// for as long as the set holds it: the pile has to hand out *disjoint*
    /// slices rather than lend itself twice.
    struct Buffers<'b> {
        rest: &'b mut [u8],
    }

    impl<'b> Buffers<'b> {
        /// Receive and transmit buffer for one socket.
        const EACH: usize = 128;

        fn new(arena: &'b mut [u8]) -> Self {
            Self { rest: arena }
        }

        /// One TCP socket over the next pair.
        fn socket(&mut self) -> tcp::Socket<'b> {
            let rest = core::mem::take(&mut self.rest);
            let (pair, tail) = rest.split_at_mut(Self::EACH);
            self.rest = tail;
            let (rx, tx) = pair.split_at_mut(Self::EACH / 2);
            tcp::Socket::new(tcp::SocketBuffer::new(rx), tcp::SocketBuffer::new(tx))
        }
    }

    /// **The witness this type exists for.** A set driven to capacity answers
    /// a refusal; upstream's `add` would have panicked, and a panic in a
    /// kernel service is a failure, not a diagnostic.
    #[test]
    fn a_full_budget_refuses_rather_than_panicking() {
        let mut arena = [0u8; 3 * Buffers::EACH];
        let mut bufs = Buffers::new(&mut arena);
        let mut storage = [SocketStorage::EMPTY; 2];
        let mut budget = SocketBudget::new(&mut storage[..]);
        assert_eq!(budget.budget(), 2);
        assert_eq!(budget.live(), 0);

        for n in 1..=2 {
            assert!(budget.try_add(bufs.socket()).is_ok(), "slot {n}");
            assert_eq!(budget.live(), n);
        }

        assert!(budget.is_full());
        assert_eq!(budget.spare(), 0);
        assert_eq!(
            budget.try_add(bufs.socket()).err(),
            Some(SocketsExhausted),
            "the slot past the budget is refused, not taken by force"
        );
        assert_eq!(budget.live(), 2, "and the refusal spent nothing");
    }

    /// A budget of nothing refuses its first socket. The degenerate case is
    /// the one a caller reaches by mistake — an empty slice is easy to pass —
    /// and it is exactly the case upstream panics on.
    #[test]
    fn an_empty_budget_refuses_the_first_socket() {
        let mut arena = [0u8; Buffers::EACH];
        let mut bufs = Buffers::new(&mut arena);
        let mut storage: [SocketStorage; 0] = [];
        let mut budget = SocketBudget::new(&mut storage[..]);
        assert!(budget.is_full());
        assert_eq!(budget.try_add(bufs.socket()).err(), Some(SocketsExhausted));
    }

    /// Retiring a socket returns its slot, so a long-lived row is bounded by
    /// what it holds at once rather than by what it has ever held.
    #[test]
    fn a_retired_socket_returns_its_slot() {
        let mut arena = [0u8; 2 * Buffers::EACH];
        let mut bufs = Buffers::new(&mut arena);
        let mut storage = [SocketStorage::EMPTY; 1];
        let mut budget = SocketBudget::new(&mut storage[..]);
        let handle = budget.try_add(bufs.socket()).expect("the first slot");
        assert!(budget.is_full());

        assert!(budget.try_remove(handle).is_some());
        assert_eq!(budget.live(), 0);
        assert!(budget.try_add(bufs.socket()).is_ok(), "the slot came back");
    }

    /// A handle that no longer names a socket reads as absent.
    ///
    /// Upstream answers this with `panic!("handle does not refer to a valid
    /// socket")` on both the read and the second remove. A use-after-retire is
    /// an ordinary bug, and it must not be able to take a service down.
    #[test]
    fn a_retired_handle_reads_as_absent_not_as_a_panic() {
        let mut arena = [0u8; Buffers::EACH];
        let mut bufs = Buffers::new(&mut arena);
        let mut storage = [SocketStorage::EMPTY; 1];
        let mut budget = SocketBudget::new(&mut storage[..]);
        let handle = budget.try_add(bufs.socket()).expect("the first slot");
        assert!(budget.try_remove(handle).is_some());

        assert!(budget.try_get::<tcp::Socket>(handle).is_none());
        assert!(budget.try_get_mut::<tcp::Socket>(handle).is_none());
        assert!(budget.try_remove(handle).is_none(), "and again");
    }

    /// A handle from a set that never issued it reads as absent rather than
    /// indexing past the storage — upstream's other panic on this path.
    #[test]
    fn a_foreign_handle_does_not_index_past_the_storage() {
        let mut arena = [0u8; 5 * Buffers::EACH];
        let mut bufs = Buffers::new(&mut arena);
        let mut big = [SocketStorage::EMPTY; 4];
        let mut small = [SocketStorage::EMPTY; 1];
        let mut wide = SocketBudget::new(&mut big[..]);
        let mut narrow = SocketBudget::new(&mut small[..]);

        let mut far = wide.try_add(bufs.socket()).expect("a slot");
        for _ in 1..4 {
            far = wide.try_add(bufs.socket()).expect("a slot");
        }
        narrow.try_add(bufs.socket()).expect("its one slot");

        assert!(
            narrow.try_get_mut::<tcp::Socket>(far).is_none(),
            "a handle naming slot 3 of another set is out of range here"
        );
        assert!(narrow.try_remove(far).is_none());
    }

    /// **Two vifs, two budgets.** A holder that spends its whole budget
    /// refuses only itself: a budget is the storage its own row borrows, so
    /// there is no shared pool for one mutually-distrusting task to exhaust
    /// against another's. This is `flow_isolation`'s property, at the socket
    /// layer.
    #[test]
    fn one_exhausted_budget_does_not_deny_another() {
        let mut arena = [0u8; 3 * Buffers::EACH];
        let mut bufs = Buffers::new(&mut arena);
        let mut storage_a = [SocketStorage::EMPTY; 1];
        let mut storage_b = [SocketStorage::EMPTY; 1];
        let mut a = SocketBudget::new(&mut storage_a[..]);
        let mut b = SocketBudget::new(&mut storage_b[..]);

        a.try_add(bufs.socket()).expect("a's one slot");
        assert_eq!(
            a.try_add(bufs.socket()).err(),
            Some(SocketsExhausted),
            "a is spent"
        );

        assert!(
            b.try_add(bufs.socket()).is_ok(),
            "and b, which spent nothing, still has everything it was given"
        );
        assert_eq!(b.spare(), 0);
        assert_eq!(a.live(), 1, "neither took from the other");
    }
}
