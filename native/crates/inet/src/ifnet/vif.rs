//! **[`VifIfnet`]** — the one production [`Ifnet`], over a vif window.
//!
//! The window is the frozen layout both sides compile in
//! (`kore_driver::net::window`): two SPSC rings and two slot areas, formatted
//! by the client before transfer and validated by the service on attach. This
//! is the *client* half of it — the transmit ring's producer and the receive
//! ring's consumer — which is the half a protocol stack drives.
//!
//! Nothing here knows which NIC is on the other end, and that is the whole
//! compatibility argument: a descriptor ring, a USB bulk pipe and a Wi-Fi
//! part all reach this type as the same two rings.

use kore_driver::net::window::{
    RX_AREA_BYTES, RX_RING_AT, SLOT_BYTES, TX_AREA_AT, TX_AREA_BYTES, WINDOW_BYTES, rx_slot_of,
    tx_slot_of,
};
use kore_driver::net::frame::HEADER_LEN;
use kore_driver::net::{EtherType, FRAME_MAX, FRAME_MIN, FrameDesc, LinkState, MacAddr};
use kore_ring::spsc;

use super::{Ifnet, IfnetError, TxSlot};

/// A vif window, driven from the client side.
pub struct VifIfnet<'w> {
    tx: spsc::Producer<'w, FrameDesc>,
    rx: spsc::Consumer<'w, FrameDesc>,
    tx_frames: &'w mut [u8],
    rx_frames: &'w mut [u8],
    /// Transmit descriptors published, ever. The slot a push occupies is
    /// derived from this rather than stored, so the two sides agree on slot
    /// ownership by arithmetic instead of by a shared cursor.
    pushed: u64,
    /// Transmit descriptors the row has been told about. The difference from
    /// `pushed` is the wake this window owes it.
    kicked: u64,
    /// Receive descriptors taken, ever. Same derivation, other direction.
    read: u64,
    hwaddr: MacAddr,
    link: LinkState,
}

impl<'w> VifIfnet<'w> {
    /// Attach to a window the client has already formatted.
    ///
    /// Formatting is the client's job (`spsc::init` needs exclusive access),
    /// so this attaches with `producer`/`consumer`, which validate the magic,
    /// the ABI revision and the record size. A window this process did not
    /// format is refused here rather than trusted.
    pub fn attach(window: &'w mut [u8], hwaddr: MacAddr) -> Result<Self, IfnetError> {
        if window.len() < WINDOW_BYTES {
            return Err(IfnetError::WindowTooSmall);
        }
        let (rings, frames) = window[..WINDOW_BYTES].split_at_mut(TX_AREA_AT);
        let (tx_ring, rx_ring) = rings.split_at_mut(RX_RING_AT);
        let (tx_frames, rx_frames) = frames.split_at_mut(TX_AREA_BYTES);
        debug_assert_eq!(rx_frames.len(), RX_AREA_BYTES);

        let tx = spsc::producer::<FrameDesc>(tx_ring).map_err(|_| IfnetError::MalformedWindow)?;
        let rx = spsc::consumer::<FrameDesc>(rx_ring).map_err(|_| IfnetError::MalformedWindow)?;

        Ok(Self {
            tx,
            rx,
            tx_frames,
            rx_frames,
            pushed: 0,
            kicked: 0,
            read: 0,
            hwaddr,
            link: LinkState::Down,
        })
    }

    /// Record the link state the row reported.
    ///
    /// Derived from the service's reply rather than observed here: this half
    /// of the window has no way to see carrier, and inventing one would be a
    /// second source of truth for a fact the row already owns.
    pub fn set_link(&mut self, link: LinkState) {
        self.link = link;
    }

    /// Transmit slots the service has released back.
    pub fn tx_free(&self) -> u32 {
        self.tx.capacity_left()
    }

    /// Whether this window owes the row a wake, without claiming it.
    pub fn tx_kick_pending(&self) -> bool {
        self.kicked != self.pushed
    }

    /// Claim the wake this window owes the frame row.
    ///
    /// `true` once after each run of publishes, and ringing the doorbell is
    /// the caller's — the half that holds it, which this crate deliberately
    /// does not. [`TxSlot::commit`] can only push into the ring: the row may
    /// have swept this window moments before and parked, and a wired backend
    /// gives it no periodic service deadline, so without this the frame waits
    /// on traffic that has no reason to arrive.
    ///
    /// Batched rather than per frame — one wake answers for every descriptor
    /// published since the last, which is what makes this a kick per sweep
    /// instead of a kick per packet.
    pub fn take_tx_kick(&mut self) -> bool {
        if self.kicked == self.pushed {
            return false;
        }
        self.kicked = self.pushed;
        true
    }
}

impl<'w> Ifnet for VifIfnet<'w> {
    type TxSlot<'a>
        = VifTxSlot<'a, 'w>
    where
        Self: 'a;

    fn hwaddr(&self) -> MacAddr {
        self.hwaddr
    }

    fn mtu(&self) -> usize {
        FRAME_MAX - HEADER_LEN
    }

    fn link(&self) -> LinkState {
        self.link
    }

    fn reserve(&mut self) -> Option<Self::TxSlot<'_>> {
        if self.tx.capacity_left() == 0 {
            return None;
        }
        let slot = tx_slot_of(self.pushed) as usize;
        let at = slot * SLOT_BYTES;
        Some(VifTxSlot {
            bytes: &mut self.tx_frames[at..at + SLOT_BYTES],
            tx: &mut self.tx,
            pushed: &mut self.pushed,
            headers: 0,
        })
    }

    fn poll<F>(&mut self, mut f: F) -> bool
    where
        F: FnMut(&[u8], EtherType),
    {
        let Some(desc) = self.rx.pop() else {
            return false;
        };
        let slot = rx_slot_of(self.read) as usize;
        self.read = self.read.wrapping_add(1);

        // The descriptor comes from a window the peer also writes, so its
        // length is validated against the slot rather than believed. A frame
        // that does not fit is dropped: the slot is already consumed, and
        // handing a protocol a truncated frame is worse than losing it.
        let len = desc.len as usize;
        if (FRAME_MIN..=FRAME_MAX).contains(&len) && len <= SLOT_BYTES {
            let at = slot * SLOT_BYTES;
            f(&self.rx_frames[at..at + len], desc.ethertype());
        }
        self.rx.outstanding() > 0
    }
}

/// One reserved transmit slot of a vif window.
pub struct VifTxSlot<'a, 'w> {
    bytes: &'a mut [u8],
    tx: &'a mut spsc::Producer<'w, FrameDesc>,
    pushed: &'a mut u64,
    headers: usize,
}

impl TxSlot for VifTxSlot<'_, '_> {
    fn capacity(&self) -> usize {
        // The slot is larger than a frame may be; the frame ceiling is what a
        // caller can actually use.
        FRAME_MAX.min(self.bytes.len())
    }

    fn reserve_headers(&mut self, len: usize) -> Option<&mut [u8]> {
        if len > self.capacity() {
            return None;
        }
        self.headers = len;
        let cap = self.capacity();
        Some(&mut self.bytes[len..cap])
    }

    fn headers_mut(&mut self) -> &mut [u8] {
        &mut self.bytes[..self.headers]
    }

    fn commit(self, payload: usize, ethertype: EtherType) -> Result<(), IfnetError> {
        let len = self.headers + payload;
        if len > FRAME_MAX || len > self.bytes.len() {
            return Err(IfnetError::TooLong);
        }
        if len < FRAME_MIN {
            return Err(IfnetError::TooShort);
        }
        // Capacity was established at `reserve`, and this half of the window
        // is the ring's only producer, so the push cannot fail between there
        // and here.
        if self.tx.push(FrameDesc::new(len as u16, ethertype)).is_ok() {
            *self.pushed = self.pushed.wrapping_add(1);
        }
        Ok(())
    }
}
