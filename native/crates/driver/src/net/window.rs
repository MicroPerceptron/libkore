//! **The per-vif window layout** — the shared-memory contract between a
//! client and the frame service, fixed here so both sides compute it from one
//! definition rather than agreeing by convention.
//!
//! ```text
//! 0                    spsc<FrameDesc> TX ring   client produces, service consumes
//! RX_RING_AT           spsc<FrameDesc> RX ring   service produces, client consumes
//! TX_AREA_AT           TX_DEPTH slots × SLOT_BYTES   client writes, service reads
//! RX_AREA_AT           RX_DEPTH slots × SLOT_BYTES   service writes, client reads
//! ```
//!
//! # Two properties the layout exists to guarantee
//!
//! **The transmit and receive slot subregions are disjoint.** Nothing the
//! client writes can land in a slot the service is filling, so a client
//! cannot forge a receive by writing into its own window.
//!
//! **A slot's index is the ring position that named it.** The producer's
//! *n*-th push occupies slot `n mod depth`; each side tracks only its own
//! monotonic count. So no slot index travels on the wire for a hostile peer
//! to disagree with, and there is no bounds check that can be skipped — the
//! index is computed, never received.
//!
//! Every constant here is **architectural**. They fix the window size a
//! client must supply and the depth the service arms, so both sides compile
//! them in rather than negotiating.

use kore_ring::spsc;

use super::frame::{FRAME_MAX, FrameDesc};

/// Bytes per frame slot — [`FRAME_MAX`] rounded up to a power of two, so slot
/// addressing is a shift.
pub const SLOT_BYTES: usize = 2048;

const _: () = assert!(SLOT_BYTES >= FRAME_MAX, "a slot must hold a whole frame");

/// Transmit descriptors, and therefore transmit slots.
pub const TX_DEPTH: u32 = 32;
/// Receive descriptors, and therefore receive slots.
pub const RX_DEPTH: u32 = 32;

/// Vifs one received frame may be copied into.
///
/// Broadcast reaches every vif that accepts broadcast, so the classifier's
/// per-frame work is bounded by this rather than by how many vifs happen to
/// be open. A frame matching more vifs than this is delivered to the first
/// [`MAX_RX_FANOUT`] and **counted as dropped** for the rest — bounded, and
/// visibly so.
pub const MAX_RX_FANOUT: usize = 8;

/// Byte offset of the transmit ring.
pub const TX_RING_AT: usize = 0;
/// Byte offset of the receive ring.
pub const RX_RING_AT: usize = spsc::region_bytes::<FrameDesc>(TX_DEPTH);
/// Byte offset of the first transmit slot.
pub const TX_AREA_AT: usize = RX_RING_AT + spsc::region_bytes::<FrameDesc>(RX_DEPTH);
/// Bytes of transmit slots.
pub const TX_AREA_BYTES: usize = TX_DEPTH as usize * SLOT_BYTES;
/// Byte offset of the first receive slot.
pub const RX_AREA_AT: usize = TX_AREA_AT + TX_AREA_BYTES;
/// Bytes of receive slots.
pub const RX_AREA_BYTES: usize = RX_DEPTH as usize * SLOT_BYTES;
/// Total bytes one vif's window occupies.
pub const WINDOW_BYTES: usize = RX_AREA_AT + RX_AREA_BYTES;

/// Byte offset of transmit slot `slot`, or `None` when out of range.
pub const fn tx_slot_at(slot: u32) -> Option<usize> {
    if slot >= TX_DEPTH {
        return None;
    }
    Some(TX_AREA_AT + slot as usize * SLOT_BYTES)
}

/// Byte offset of receive slot `slot`, or `None` when out of range.
pub const fn rx_slot_at(slot: u32) -> Option<usize> {
    if slot >= RX_DEPTH {
        return None;
    }
    Some(RX_AREA_AT + slot as usize * SLOT_BYTES)
}

/// The transmit slot a producer's `count`-th push occupies.
/// Bytes of the frame area — the transmit and receive slots together,
/// the part of the window past both rings.
///
/// A service maps the rings and the slots as *separate* borrows of one
/// window (the ring halves hold theirs for the vif's life), so slot
/// addressing inside the service is relative to this area rather than to
/// the window. [`tx_frame_at`] and [`rx_frame_at`] are that view of the same
/// layout; the tests below pin the two views together so they cannot drift.
pub const FRAME_AREA_BYTES: usize = TX_AREA_BYTES + RX_AREA_BYTES;

/// Byte offset of transmit slot `slot` **within the frame area**, or `None`
/// past [`TX_DEPTH`].
pub const fn tx_frame_at(slot: u32) -> Option<usize> {
    if slot >= TX_DEPTH {
        return None;
    }
    Some(slot as usize * SLOT_BYTES)
}

/// Byte offset of receive slot `slot` **within the frame area**, or `None`
/// past [`RX_DEPTH`].
pub const fn rx_frame_at(slot: u32) -> Option<usize> {
    if slot >= RX_DEPTH {
        return None;
    }
    Some(TX_AREA_BYTES + slot as usize * SLOT_BYTES)
}

pub const fn tx_slot_of(count: u64) -> u32 {
    (count % TX_DEPTH as u64) as u32
}

/// The receive slot a producer's `count`-th push occupies.
pub const fn rx_slot_of(count: u64) -> u32 {
    (count % RX_DEPTH as u64) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two slot subregions must not overlap, and neither may reach into
    /// the rings.  A client that could write into a receive slot could forge
    /// a frame it never received.
    #[test]
    fn transmit_and_receive_subregions_are_disjoint() {
        assert!(TX_RING_AT < RX_RING_AT);
        assert!(RX_RING_AT < TX_AREA_AT, "rings end before the slot area");
        assert_eq!(
            TX_AREA_AT + TX_AREA_BYTES,
            RX_AREA_AT,
            "adjacent, not overlapping"
        );
        assert_eq!(RX_AREA_AT + RX_AREA_BYTES, WINDOW_BYTES);

        let last_tx = tx_slot_at(TX_DEPTH - 1).unwrap() + SLOT_BYTES;
        assert!(
            last_tx <= RX_AREA_AT,
            "the last transmit slot stops short of receive"
        );
        let last_rx = rx_slot_at(RX_DEPTH - 1).unwrap() + SLOT_BYTES;
        assert_eq!(last_rx, WINDOW_BYTES);
    }

    /// A slot index past the depth resolves to nothing rather than to an
    /// offset inside the neighbouring subregion.
    #[test]
    fn the_frame_area_view_is_the_window_view_shifted() {
        assert_eq!(TX_AREA_AT + FRAME_AREA_BYTES, WINDOW_BYTES);
        for slot in 0..TX_DEPTH {
            assert_eq!(
                tx_frame_at(slot).unwrap() + TX_AREA_AT,
                tx_slot_at(slot).unwrap(),
                "the two views name the same byte"
            );
        }
        for slot in 0..RX_DEPTH {
            assert_eq!(
                rx_frame_at(slot).unwrap() + TX_AREA_AT,
                rx_slot_at(slot).unwrap()
            );
        }
        assert!(tx_frame_at(TX_DEPTH).is_none());
        assert!(rx_frame_at(RX_DEPTH).is_none());
        let last = rx_frame_at(RX_DEPTH - 1).unwrap() + SLOT_BYTES;
        assert_eq!(
            last, FRAME_AREA_BYTES,
            "the last receive slot ends the area"
        );
    }

    #[test]
    fn out_of_range_slots_refuse() {
        assert!(tx_slot_at(TX_DEPTH).is_none());
        assert!(rx_slot_at(RX_DEPTH).is_none());
        assert!(tx_slot_at(u32::MAX).is_none());
        assert!(rx_slot_at(u32::MAX).is_none());
    }

    /// The derivation both sides rely on: the *n*-th push lands in slot
    /// `n mod depth`, and wraps without ever leaving the subregion.
    #[test]
    fn slot_index_derives_from_the_push_count() {
        assert_eq!(tx_slot_of(0), 0);
        assert_eq!(tx_slot_of(TX_DEPTH as u64 - 1), TX_DEPTH - 1);
        assert_eq!(tx_slot_of(TX_DEPTH as u64), 0, "wraps to the first slot");
        assert_eq!(rx_slot_of(RX_DEPTH as u64 + 3), 3);
        // Every derived index is in range, including at a u64 boundary.
        for n in [0u64, 1, 31, 32, 1_000_000, u64::MAX] {
            assert!(tx_slot_at(tx_slot_of(n)).is_some());
            assert!(rx_slot_at(rx_slot_of(n)).is_some());
        }
    }
}
