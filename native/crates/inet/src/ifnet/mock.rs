//! **[`MockIfnet`]** — the host double, and the reason every protocol above
//! this seam is testable without QEMU or hardware.
//!
//! The pattern `drivers/net/dot11` already uses for its in-memory AP: a double
//! that speaks the same contract production does, so a test exercises the real
//! engine rather than a parallel one. Heapless like everything else here — the
//! caller supplies the scratch bytes, exactly as a vif window is supplied to
//! [`VifIfnet`](super::VifIfnet).

use kore_driver::net::window::SLOT_BYTES;
use kore_driver::net::frame::HEADER_LEN;
use kore_driver::net::{EtherType, FRAME_MAX, FRAME_MIN, LinkState, MacAddr};

use super::{Ifnet, IfnetError, TxSlot};

/// Scratch an interface double needs: one slot each way.
pub const MOCK_SCRATCH_BYTES: usize = SLOT_BYTES * 2;

/// An in-memory interface.
///
/// One frame in flight each way, which is all a seam test needs; depth belongs
/// to whichever slice first has a protocol that can produce a burst.
pub struct MockIfnet<'b> {
    tx_slot: &'b mut [u8],
    rx_slot: &'b mut [u8],
    sent: Option<(usize, EtherType)>,
    sent_count: usize,
    pending: Option<(usize, EtherType)>,
    loopback: bool,
    /// Refuse every transmit reservation, so a caller's full-ring path is
    /// reachable without filling a real ring.
    tx_full: bool,
    hwaddr: MacAddr,
    link: LinkState,
    mtu: usize,
}

impl<'b> MockIfnet<'b> {
    /// A double over `scratch`, which must be at least
    /// [`MOCK_SCRATCH_BYTES`].
    pub fn new(scratch: &'b mut [u8], hwaddr: MacAddr) -> Result<Self, IfnetError> {
        if scratch.len() < MOCK_SCRATCH_BYTES {
            return Err(IfnetError::WindowTooSmall);
        }
        let (tx_slot, rest) = scratch.split_at_mut(SLOT_BYTES);
        let (rx_slot, _) = rest.split_at_mut(SLOT_BYTES);
        Ok(Self {
            tx_slot,
            rx_slot,
            sent: None,
            sent_count: 0,
            pending: None,
            loopback: false,
            tx_full: false,
            hwaddr,
            link: LinkState::Up { mbps: 0 },
            mtu: FRAME_MAX - HEADER_LEN,
        })
    }

    /// Return every committed frame as a received one.
    ///
    /// What makes a seam test end-to-end: a protocol transmits through the
    /// contract and reads its own bytes back through the contract, so both
    /// halves are exercised rather than asserted about.
    pub fn with_loopback(mut self) -> Self {
        self.loopback = true;
        self
    }

    /// Refuse transmit reservations, as a full ring does.
    pub fn set_tx_full(&mut self, full: bool) {
        self.tx_full = full;
    }

    /// Set the reported link state.
    pub fn set_link(&mut self, link: LinkState) {
        self.link = link;
    }

    /// Set the reported MTU, to drive a protocol's fragmentation decisions.
    pub fn set_mtu(&mut self, mtu: usize) {
        self.mtu = mtu;
    }

    /// Queue `frame` as the next received frame.
    pub fn inject(&mut self, frame: &[u8], ethertype: EtherType) -> Result<(), IfnetError> {
        if frame.len() > self.rx_slot.len() || frame.len() > FRAME_MAX {
            return Err(IfnetError::TooLong);
        }
        if frame.len() < FRAME_MIN {
            return Err(IfnetError::TooShort);
        }
        self.rx_slot[..frame.len()].copy_from_slice(frame);
        self.pending = Some((frame.len(), ethertype));
        Ok(())
    }

    /// The most recently committed frame.
    pub fn last_sent(&self) -> Option<(&[u8], EtherType)> {
        self.sent.map(|(len, et)| (&self.tx_slot[..len], et))
    }

    /// Frames committed, ever.
    pub fn sent_count(&self) -> usize {
        self.sent_count
    }
}

impl<'b> Ifnet for MockIfnet<'b> {
    type TxSlot<'a>
        = MockTxSlot<'a, 'b>
    where
        Self: 'a;

    fn hwaddr(&self) -> MacAddr {
        self.hwaddr
    }

    fn mtu(&self) -> usize {
        self.mtu
    }

    fn link(&self) -> LinkState {
        self.link
    }

    fn reserve(&mut self) -> Option<Self::TxSlot<'_>> {
        (!self.tx_full).then(|| MockTxSlot {
            net: self,
            headers: 0,
        })
    }

    fn poll<F>(&mut self, mut f: F) -> bool
    where
        F: FnMut(&[u8], EtherType),
    {
        let Some((len, ethertype)) = self.pending.take() else {
            return false;
        };
        f(&self.rx_slot[..len], ethertype);
        self.pending.is_some()
    }
}

/// One reserved transmit slot of a [`MockIfnet`].
pub struct MockTxSlot<'a, 'b> {
    net: &'a mut MockIfnet<'b>,
    headers: usize,
}

impl TxSlot for MockTxSlot<'_, '_> {
    fn capacity(&self) -> usize {
        FRAME_MAX.min(self.net.tx_slot.len())
    }

    fn reserve_headers(&mut self, len: usize) -> Option<&mut [u8]> {
        if len > self.capacity() {
            return None;
        }
        self.headers = len;
        let cap = self.capacity();
        Some(&mut self.net.tx_slot[len..cap])
    }

    fn headers_mut(&mut self) -> &mut [u8] {
        &mut self.net.tx_slot[..self.headers]
    }

    fn commit(self, payload: usize, ethertype: EtherType) -> Result<(), IfnetError> {
        let len = self.headers + payload;
        if len > FRAME_MAX || len > self.net.tx_slot.len() {
            return Err(IfnetError::TooLong);
        }
        if len < FRAME_MIN {
            return Err(IfnetError::TooShort);
        }
        self.net.sent = Some((len, ethertype));
        self.net.sent_count += 1;
        if self.net.loopback {
            // Two disjoint fields of one borrow: no aliasing, no unsafe.
            let MockIfnet {
                tx_slot, rx_slot, ..
            } = &mut *self.net;
            rx_slot[..len].copy_from_slice(&tx_slot[..len]);
            self.net.pending = Some((len, ethertype));
        }
        Ok(())
    }
}
