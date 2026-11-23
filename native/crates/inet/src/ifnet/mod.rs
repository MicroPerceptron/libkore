//! **The ifnet seam** — the one contract every packet source implements, and
//! the only variation point the protocol engine is generic over.
//!
//! # Where a frame actually starts
//!
//! A transmit slot is [`SLOT_BYTES`] of a vif window, and the frame it carries
//! begins at **offset zero**: [`FrameDesc`] states a length and an EtherType
//! and nothing else, so there is no field in which a frame could declare that
//! it starts part-way into its slot.
//!
//! That rules out the usual trick of reserving headroom at the front and
//! prepending headers backwards into it. The discipline here is the other one,
//! and it costs exactly as little: **reserve the header length you are about
//! to write, put the payload after it, then fill the headers in place.** A
//! sender knows its own encapsulation — Ethernet + IPv4 + UDP is 42 bytes — so
//! the reservation is a constant, not a guess, and no byte is copied twice
//! either way. [`TxSlot::reserve_headers`] is that operation, and it is the
//! only way to obtain the payload region, so the reservation cannot be
//! forgotten.
//!
//! # What is deliberately absent
//!
//! There is no scatter/gather here, and therefore no service-owned header
//! segment chained to a client-owned payload. That shape — the fix
//! `res/NETWORK_SERVICE.md` §7 names for the validate-then-DMA window — needs a
//! descriptor that can *describe* two segments, and [`FrameDesc`] cannot. It is
//! a contract change of its own, not something this seam can smuggle in, and
//! `res/INET_STACK.md` §3 records that it is owed.

mod mock;
mod vif;

pub use mock::{MOCK_SCRATCH_BYTES, MockIfnet};
pub use vif::{VifIfnet, VifTxSlot};

use kore_driver::net::{EtherType, LinkState, MacAddr};

/// Why an interface could not be attached, or a frame not placed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IfnetError {
    /// The window is smaller than one vif window.
    WindowTooSmall,
    /// The window's rings are absent, malformed, or a different ABI — refused
    /// at attach rather than trusted, because the peer across the window
    /// writes them too.
    MalformedWindow,
    /// More bytes were committed than the slot or the link can carry.
    TooLong,
    /// Fewer bytes than an Ethernet header.
    TooShort,
}

/// A source and sink of link-layer frames.
///
/// The one variation point the protocol engine is generic over. Per-interface
/// encoding lives in [`TxSlot`], so the engine never names an interface kind
/// and selection is type instantiation (`res/INET_STACK.md` §2.3).
pub trait Ifnet {
    /// This interface's reserved transmit slot.
    ///
    /// Borrowed from the interface, so a slot that is dropped without
    /// [`TxSlot::commit`] releases rather than leaks — a lost descriptor
    /// becomes a borrow-check outcome instead of a runtime one.
    type TxSlot<'a>: TxSlot
    where
        Self: 'a;

    /// The station address this interface answers to.
    fn hwaddr(&self) -> MacAddr;

    /// The largest payload this interface carries, excluding the Ethernet
    /// header.
    fn mtu(&self) -> usize;

    /// Whether the link has carrier.
    fn link(&self) -> LinkState;

    /// Reserve one transmit slot, or `None` when the ring has no room.
    ///
    /// Never blocks and never allocates. A caller that cannot get a slot has
    /// learned the ring is full, which is a scheduling fact rather than an
    /// error.
    fn reserve(&mut self) -> Option<Self::TxSlot<'_>>;

    /// Present each received frame to `f` as `(bytes, ethertype)`, where
    /// `bytes` borrows the receive slot and begins at the Ethernet header.
    ///
    /// Returns `true` when more may be waiting, so a caller can bound one
    /// sweep and come back — the drive-cycle posture the frame row already
    /// uses rather than draining to exhaustion under an unknown arrival rate.
    fn poll<F>(&mut self, f: F) -> bool
    where
        F: FnMut(&[u8], EtherType);
}

/// One reserved transmit slot.
///
/// The slot is consumed by [`commit`](Self::commit), so a frame is published
/// exactly once and a builder cannot be reused after it has been sent.
pub trait TxSlot {
    /// Bytes this slot can carry, headers included.
    fn capacity(&self) -> usize;

    /// Reserve `len` bytes of header at the front and borrow the payload
    /// region that follows.
    ///
    /// The only way to reach the payload, so the reservation is structural:
    /// a caller cannot write a payload at offset zero and then discover it has
    /// nowhere to put its headers. `None` when the slot cannot hold `len`.
    fn reserve_headers(&mut self, len: usize) -> Option<&mut [u8]>;

    /// Borrow the header region reserved by
    /// [`reserve_headers`](Self::reserve_headers), to fill in place.
    ///
    /// Empty before any reservation.
    fn headers_mut(&mut self) -> &mut [u8];

    /// Publish a frame of `payload` bytes after the reserved headers.
    ///
    /// The committed length is the reservation plus `payload`, which is the
    /// whole frame — the caller never computes an offset, so a header length
    /// and a frame length cannot disagree.
    fn commit(self, payload: usize, ethertype: EtherType) -> Result<(), IfnetError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use kore_driver::net::frame::HEADER_LEN;
    use kore_driver::net::window::{
        RX_DEPTH, RX_RING_AT, TX_DEPTH, TX_RING_AT, WINDOW_BYTES,
    };
    use kore_driver::net::{FRAME_MAX, FrameDesc, FrameHead};
    use kore_ring::spsc;

    const MINE: MacAddr = MacAddr([0x02, 0, 0, 0, 0, 0x01]);
    const PEER: MacAddr = MacAddr([0x02, 0, 0, 0, 0, 0x02]);

    /// A window must be 64-byte aligned, which a bare stack array is not.
    #[repr(align(64))]
    struct Aligned<const N: usize>([u8; N]);

    /// Write an Ethernet header into a slot's reserved region and a pattern
    /// into its payload, the way a protocol above this seam would.
    fn put_frame<S: TxSlot>(slot: &mut S, payload_len: usize, fill: u8) {
        let payload = slot
            .reserve_headers(HEADER_LEN)
            .expect("a slot holds an Ethernet header");
        for b in payload[..payload_len].iter_mut() {
            *b = fill;
        }
        FrameHead {
            dst: PEER,
            src: MINE,
            ethertype: EtherType::IPV4,
        }
        .write(slot.headers_mut())
        .expect("the reservation is exactly a header");
    }

    // ── the seam, over the double ──────────────────────────────────────

    /// The end-to-end seam witness: a frame built through the contract and
    /// read back through the contract, byte for byte, with its EtherType.
    #[test]
    fn a_frame_goes_out_and_comes_back() {
        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut net = MockIfnet::new(&mut scratch.0, MINE)
            .expect("scratch is big enough")
            .with_loopback();

        let mut slot = net.reserve().expect("the double always has a slot");
        put_frame(&mut slot, 46, 0xa5);
        slot.commit(46, EtherType::IPV4).expect("commits");

        assert_eq!(net.sent_count(), 1);

        let mut seen = 0usize;
        let mut got = [0u8; 64];
        let mut got_type = EtherType(0);
        net.poll(|bytes, ethertype| {
            seen = bytes.len();
            got[..bytes.len()].copy_from_slice(bytes);
            got_type = ethertype;
        });

        assert_eq!(seen, HEADER_LEN + 46, "header and payload, nothing else");
        assert_eq!(got_type, EtherType::IPV4);

        let head = FrameHead::parse(&got[..seen]).expect("a well-formed header");
        assert_eq!(head.dst, PEER);
        assert_eq!(head.src, MINE);
        assert_eq!(head.ethertype, EtherType::IPV4);
        assert!(
            got[HEADER_LEN..seen].iter().all(|&b| b == 0xa5),
            "the payload survived the round trip"
        );
    }

    /// The reservation is the *only* way to the payload, and the two regions
    /// do not overlap — the invariant that replaces the headroom a
    /// `FrameDesc` cannot express.
    #[test]
    fn the_payload_begins_after_the_reserved_headers() {
        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut net = MockIfnet::new(&mut scratch.0, MINE).expect("scratch");
        let mut slot = net.reserve().expect("slot");

        assert_eq!(slot.headers_mut().len(), 0, "nothing reserved yet");

        let payload = slot.reserve_headers(HEADER_LEN).expect("reserves");
        assert_eq!(
            payload.len(),
            slot_capacity() - HEADER_LEN,
            "the payload is the rest of the usable slot"
        );
        for b in payload.iter_mut() {
            *b = 0x11;
        }

        let headers = slot.headers_mut();
        assert_eq!(headers.len(), HEADER_LEN);
        assert!(
            headers.iter().all(|&b| b == 0),
            "writing the payload did not reach into the header region"
        );
    }

    fn slot_capacity() -> usize {
        FRAME_MAX
    }

    /// A reservation larger than the slot is refused rather than truncated.
    #[test]
    fn an_impossible_reservation_is_refused() {
        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut net = MockIfnet::new(&mut scratch.0, MINE).expect("scratch");
        let mut slot = net.reserve().expect("slot");
        assert!(slot.reserve_headers(FRAME_MAX + 1).is_none());
    }

    /// Both frame bounds are enforced at commit, where the total is known —
    /// a caller cannot publish a runt or an oversize frame.
    #[test]
    fn commit_enforces_both_frame_bounds() {
        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut net = MockIfnet::new(&mut scratch.0, MINE).expect("scratch");

        let mut slot = net.reserve().expect("slot");
        slot.reserve_headers(HEADER_LEN).expect("reserves");
        assert_eq!(
            slot.commit(FRAME_MAX, EtherType::IPV4),
            Err(IfnetError::TooLong),
            "header + payload is the frame, and it must fit"
        );

        let mut slot = net.reserve().expect("slot");
        slot.reserve_headers(4).expect("reserves");
        assert_eq!(
            slot.commit(0, EtherType::IPV4),
            Err(IfnetError::TooShort),
            "shorter than an Ethernet header"
        );
    }

    /// A double with no loopback and nothing injected yields nothing, so an
    /// empty poll is distinguishable from a delivered frame.
    #[test]
    fn an_idle_interface_delivers_nothing() {
        let mut scratch = Aligned([0u8; MOCK_SCRATCH_BYTES]);
        let mut net = MockIfnet::new(&mut scratch.0, MINE).expect("scratch");
        let mut called = false;
        assert!(!net.poll(|_, _| called = true));
        assert!(!called);
    }

    // ── the seam, over a real vif window ───────────────────────────────

    fn format(window: &mut [u8]) {
        let (rings, _) = window.split_at_mut(RX_RING_AT + spsc::region_bytes::<FrameDesc>(RX_DEPTH));
        let (tx, rx) = rings.split_at_mut(RX_RING_AT);
        spsc::init::<FrameDesc>(tx, TX_DEPTH).expect("tx ring");
        spsc::init::<FrameDesc>(rx, RX_DEPTH).expect("rx ring");
        let _ = TX_RING_AT;
    }

    /// A window the client never formatted is refused at attach, not trusted
    /// — the peer across it writes these bytes too.
    #[test]
    fn an_unformatted_window_is_refused() {
        let mut w = Aligned([0u8; WINDOW_BYTES]);
        assert_eq!(
            VifIfnet::attach(&mut w.0, MINE).err(),
            Some(IfnetError::MalformedWindow)
        );
    }

    /// A window too small to hold the layout both sides compile in is
    /// refused before anything reads it.
    #[test]
    fn a_short_window_is_refused() {
        let mut w = Aligned([0u8; 64]);
        assert_eq!(
            VifIfnet::attach(&mut w.0, MINE).err(),
            Some(IfnetError::WindowTooSmall)
        );
    }

    /// Committing publishes a descriptor into the transmit ring: the slot the
    /// service will read is spoken for, which is what `tx_free` reports.
    #[test]
    fn committing_publishes_into_the_transmit_ring() {
        let mut w = Aligned([0u8; WINDOW_BYTES]);
        format(&mut w.0);
        let mut net = VifIfnet::attach(&mut w.0, MINE).expect("attaches");

        assert_eq!(net.tx_free(), TX_DEPTH);
        assert_eq!(net.mtu(), FRAME_MAX - HEADER_LEN);
        assert_eq!(net.hwaddr(), MINE);
        assert!(!net.link().is_up(), "no carrier until the row says so");

        let mut slot = net.reserve().expect("an empty ring has a slot");
        put_frame(&mut slot, 46, 0x5a);
        slot.commit(46, EtherType::IPV4).expect("commits");

        assert_eq!(
            net.tx_free(),
            TX_DEPTH - 1,
            "one descriptor is now the service's to read"
        );
    }

    /// The ring is the backpressure: a full transmit ring hands out no slot
    /// rather than overwriting one the service has not read.
    #[test]
    fn a_full_transmit_ring_hands_out_no_slot() {
        let mut w = Aligned([0u8; WINDOW_BYTES]);
        format(&mut w.0);
        let mut net = VifIfnet::attach(&mut w.0, MINE).expect("attaches");

        for i in 0..TX_DEPTH {
            let mut slot = net.reserve().unwrap_or_else(|| panic!("slot {i}"));
            put_frame(&mut slot, 46, i as u8);
            slot.commit(46, EtherType::IPV4).expect("commits");
        }
        assert_eq!(net.tx_free(), 0);
        assert!(net.reserve().is_none(), "no slot while the ring is full");
    }

    /// The wake a publish owes the row, and the ordering that makes it a
    /// wake rather than a tick: the debt exists the moment a descriptor is
    /// published, and is only discharged by the caller claiming it.
    ///
    /// A window that has published nothing owes nothing — a kick per sweep
    /// on an idle vif would be the polling this exists to avoid.
    #[test]
    fn a_published_descriptor_owes_the_row_one_wake() {
        let mut w = Aligned([0u8; WINDOW_BYTES]);
        format(&mut w.0);
        let mut net = VifIfnet::attach(&mut w.0, MINE).expect("attaches");

        assert!(!net.tx_kick_pending(), "an idle window owes nothing");
        assert!(!net.take_tx_kick(), "and claiming finds nothing to claim");

        let mut slot = net.reserve().expect("a slot");
        put_frame(&mut slot, 46, 0x11);
        slot.commit(46, EtherType::IPV4).expect("commits");

        assert!(
            net.tx_kick_pending(),
            "the descriptor is in the ring and the row has not been told"
        );
        assert!(net.take_tx_kick(), "the caller claims the wake it owes");
        assert!(
            !net.take_tx_kick(),
            "and does not owe a second one for the same publish"
        );

        // Batching: several publishes between two claims are one wake, not
        // one each. This is what keeps a burst from kicking per frame.
        for i in 0..3u8 {
            let mut slot = net.reserve().expect("a slot");
            put_frame(&mut slot, 46, i);
            slot.commit(46, EtherType::IPV4).expect("commits");
        }
        assert!(net.take_tx_kick(), "three publishes, one wake");
        assert!(!net.take_tx_kick(), "and nothing left owed");
    }
}
