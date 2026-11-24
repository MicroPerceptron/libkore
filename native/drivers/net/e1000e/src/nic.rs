//! **The ring engine and the bring-up ladder.**
//!
//! Two affine states, the NVMe template (`drivers/nvme/src/controller.rs`):
//! [`Prepared`] is the controller reset, identified, its rings allocated and
//! pointed at, with **both engines still disabled** and nothing posted — the
//! state a capability host reaches before the device may master memory
//! (`DmaPrepared`); [`E1000e`] is the running device. [`E1000eNic`] is the
//! per-drive-cycle view that implements the class contract.
//!
//! Register access follows xHCI rather than NVMe: the core stores no
//! register window; every method takes `regs`, because in the row the BAR
//! lives in the device's resource set and the backend is a short borrow.
//!
//! # Rings, reused
//!
//! The transmit ring is a [`SubRing`] (`stage`, then one `ring` per drive —
//! the doorbell is `TDT`); the receive ring is a [`CompRing`] whose validity
//! is the descriptor's own `DD` bit ([`desc`](crate::desc)); transmit
//! cookies ride a [`CookieTable`] keyed by descriptor index, because the
//! device completes descriptors in ring order and the index is the exact
//! correlation; receive needs no table for the same reason — the cursor is
//! the correlation, as in `kore-usbnet`.
//!
//! # Interrupts
//!
//! One MSI-family message. `IVAR` routes every cause to entry 0; `EIAC`
//! lets the controller auto-clear the queue causes on delivery, which also
//! auto-masks them from `IMS`. Every drive therefore begins by reading and
//! acknowledging `ICR` and re-arming `IMS` — the first thing `reap_tx` does
//! — so a delivered message can never leave the device silent. A cause
//! raised while the row is inside a drive is latched by the held route and
//! ends the next park.
//!
//! The message rate is **moderated** at bring-up: `ITR` and `EITR[0]` hold
//! [`ITR_INTERVAL`](crate::regs::ITR_INTERVAL), so a burst raises one
//! message per interval and the sweep behind it reaps the whole batch.
//! What the moderation does is counted: [`E1000e::counters`] reports the
//! drives that found a cause latched, the link-status changes among them,
//! and the receive overruns and errored descriptors, so the number of
//! interrupts can be held against the number of frames.

use core::ptr;
use core::sync::atomic::{Ordering, fence};

use kore_driver::net::{
    LinkState, MacAddr, NicBackend, NicCounters, PacketFilter, Steering, TxCompletion, TxCookie,
    TxFrame, TxStatus,
};
use kore_driver::seam::{Dma, DmaPool, DmaShared, Regs};
use kore_driver::{Budget, CompRing, CookieTable, SubRing, Wrap, handshake_field};

use crate::desc::{DESC_BYTES, RxDesc, TXD_STATUS_AT, TxDesc};
use crate::product::Product;
use crate::regs::i82574::*;
use crate::regs::{
    EIAC_ARMED, IMS_ARMED, ITR_INTERVAL, IVAR_ALL_TO_ENTRY0, QUEUE, RCTL_RUN, RXDCTL_RUN, TCTL_RUN,
    TXDCTL_RUN, controller_present, reg,
};
use crate::{Error, Result};

/// Transmit descriptors — one page of 16-byte entries. `TDLEN` must be a
/// multiple of 128 bytes (it is); ≥ `MAX_VIFS × MAX_TX_INFLIGHT` (64) so a
/// full engine sweep never finds the ring short; and exactly the heapless
/// [`CookieTable`] ceiling, so the alloc-free posture holds the same depth.
pub const TX_RING: u16 = 256;
/// Receive descriptors — one page, and ≥ 2 × the engine's `RX_BATCH` (64)
/// so one drive's batch never drains the ring dry.
pub const RX_RING: u16 = 256;
/// Bytes per receive buffer: `RCTL.BSIZE` = 0 with `BSEX` clear selects
/// 2048, which is ≥ `FRAME_MAX` (1514) with the FCS stripped and equals the
/// vif window's `SLOT_BYTES` — one frame, one buffer, no chaining.
pub const RX_BUF_BYTES: usize = 2048;
/// Bytes one ring occupies.
const RING_BYTES: usize = TX_RING as usize * DESC_BYTES;
const _: () = assert!(RING_BYTES.is_multiple_of(128), "TDLEN/RDLEN granularity");
const _: () = assert!(RX_RING as usize * DESC_BYTES == RING_BYTES);
/// Bytes the receive pool occupies.
const RX_POOL_BYTES: usize = RX_RING as usize * RX_BUF_BYTES;
/// Descriptors software may hand the device at once: the ring's capacity
/// less one, because `head == tail` means *empty* to the controller and a
/// fully posted ring would read as one.
const MAX_POSTED: u64 = TX_RING as u64 - 1;
const _: () = assert!(RX_RING == TX_RING, "one MAX_POSTED serves both rings");
/// Ring and pool alignment. A page is what the descriptor base registers
/// require; the IOTLB economy `res/PEER_LINK.md` §6.3 asks for is already
/// the endowment's — the whole DMA supply maps as one 2 MiB leaf, so every
/// carve inside it shares that entry and no larger alignment is needed.
const DMA_ALIGN: usize = 4096;

/// Nanoseconds for `CTRL.RST` to self-clear and the NVM auto-read to
/// complete: the datasheet's 10 ms post-reset settle plus its 10 ms
/// auto-read, doubled for margin. Expiry is evidence about the device.
const RESET_BUDGET_NS: u64 = 40_000_000;
/// The unclocked form: probes, on a machine with no calibrated counter.
const RESET_SPINS: u32 = 40_000;
/// Nanoseconds `start` waits for the link before serving: QEMU's emulated
/// autonegotiation completes ~500 ms after reset, metal copper takes 1–3 s.
/// The wait is bounded and its expiry is **not a rung** — a machine with no
/// cable founds and serves with the link down.
const LINK_BUDGET_NS: u64 = 2_000_000_000;
const LINK_SPINS: u32 = 2_000;
/// Words in the multicast table array.
const MTA_WORDS: u32 = 128;

/// The controller reset, identified, its rings allocated and pointed at,
/// both engines disabled.
pub struct Prepared<P: DmaPool> {
    product: Product,
    mac: MacAddr,
    tx: SubRing<TxDesc, P::Buf>,
    rx: CompRing<RxDesc, P::Buf>,
    rx_bufs: P::Buf,
    cookies: CookieTable,
}

impl<P: DmaPool> Prepared<P> {
    /// **Bring the controller to the prepared state.** Each step is a
    /// distinct refusal.
    ///
    /// `bar_len` is the mapped BAR0 window's length; `clock` and `pause`
    /// are the wait tier's, as in every core.
    pub fn prepare<R: Regs>(
        regs: &R,
        pool: &P,
        product: Product,
        bar_len: usize,
        clock: fn() -> Option<u64>,
        mut pause: impl FnMut(),
    ) -> Result<Self> {
        // 1. Does the window decode a controller at all?
        if !controller_present(regs, bar_len) {
            return Err(Error::NotPresent);
        }

        // 2. Quiet the device before touching anything it might be doing:
        //    every cause masked, both engines off (TCTL keeps PSP so the
        //    register is never a bare zero the datasheet reserves).
        regs.write32(reg(E1000_IMC), u32::MAX);
        regs.write32(reg(E1000_RCTL), 0);
        regs.write32(reg(E1000_TCTL), E1000_TCTL_PSP);

        // 3. Reset, and wait for the bit to self-clear.
        let ctrl = regs.read32(reg(E1000_CTRL));
        regs.write32(reg(E1000_CTRL), ctrl | E1000_CTRL_RST);
        let budget = Budget::new(RESET_BUDGET_NS, RESET_SPINS, clock);
        if !handshake_field(regs, reg(E1000_CTRL), E1000_CTRL_RST, 0, budget, &mut pause).is_acked()
        {
            return Err(Error::ResetTimeout);
        }

        // 4. The NVM auto-read loads the station address; until it reports
        //    done, RAL/RAH are whatever the reset left.
        let budget = Budget::new(RESET_BUDGET_NS, RESET_SPINS, clock);
        if !handshake_field(
            regs,
            reg(E1000_EECD),
            E1000_EECD_AUTO_RD,
            1,
            budget,
            &mut pause,
        )
        .is_acked()
        {
            return Err(Error::NvmNotLoaded);
        }

        // 5. The reset raised causes of its own; mask again and acknowledge
        //    whatever is pending so the first armed interrupt is a real one.
        regs.write32(reg(E1000_IMC), u32::MAX);
        let stale = regs.read32(reg(E1000_ICR));
        if stale != 0 {
            regs.write32(reg(E1000_ICR), stale);
        }

        // 6. The station address, from the NVM-loaded receive-address entry.
        let ral = regs.read32(reg(e1000_ral(0)));
        let rah = regs.read32(reg(e1000_rah(0)));
        if rah & E1000_RAH_AV == 0 {
            return Err(Error::NoStationAddress);
        }
        let mac = MacAddr([
            ral as u8,
            (ral >> 8) as u8,
            (ral >> 16) as u8,
            (ral >> 24) as u8,
            rah as u8,
            (rah >> 8) as u8,
        ]);
        if !mac.is_valid_source() {
            return Err(Error::NoStationAddress);
        }

        // 7. Draw the rings and the receive pool once. The pool never frees,
        //    so this is the row's allocation for its whole life.
        let rx_bufs = pool
            .alloc_aligned(RX_POOL_BYTES, DMA_ALIGN)
            .ok_or(Error::DmaAlloc)?;
        let tx_buf = pool
            .alloc_aligned(RING_BYTES, DMA_ALIGN)
            .ok_or(Error::DmaAlloc)?;
        let rx_buf = pool
            .alloc_aligned(RING_BYTES, DMA_ALIGN)
            .ok_or(Error::DmaAlloc)?;
        let tx = SubRing::new(tx_buf, TX_RING, Wrap::Modulo);
        let rx = CompRing::new(rx_buf, RX_RING);
        // Every receive descriptor points at its buffer with `DD` clear, so
        // `start` has only the tail to move.
        for slot in 0..RX_RING {
            write_rx_desc(&rx, slot, RxDesc::empty(rx_buf_iova(&rx_bufs, slot)));
        }

        // 8. Point the controller at them — engines still disabled, tails at
        //    zero, so nothing is fetched yet — and program the receive
        //    filters, the descriptor mode, and the interrupt plumbing.
        regs.write32(reg(e1000_tdbal(QUEUE)), tx.iova().0 as u32);
        regs.write32(reg(e1000_tdbah(QUEUE)), (tx.iova().0 >> 32) as u32);
        regs.write32(reg(e1000_tdlen(QUEUE)), RING_BYTES as u32);
        regs.write32(reg(e1000_tdh(QUEUE)), 0);
        regs.write32(reg(e1000_tdt(QUEUE)), 0);
        regs.write32(reg(e1000_rdbal(QUEUE)), rx.iova().0 as u32);
        regs.write32(reg(e1000_rdbah(QUEUE)), (rx.iova().0 >> 32) as u32);
        regs.write32(reg(e1000_rdlen(QUEUE)), RING_BYTES as u32);
        regs.write32(reg(e1000_rdh(QUEUE)), 0);
        regs.write32(reg(e1000_rdt(QUEUE)), 0);
        for word in 0..MTA_WORDS {
            regs.write32(reg(E1000_MTA + 4 * word), 0);
        }
        regs.write32(reg(e1000_ral(0)), ral);
        regs.write32(reg(e1000_rah(0)), rah | E1000_RAH_AV);
        // Legacy receive descriptors (`EXTEN` clear), and the 82574's MSI-X
        // ACK-accelerate workaround the reference drivers apply.
        let rfctl = regs.read32(reg(E1000_RFCTL));
        regs.write32(
            reg(E1000_RFCTL),
            (rfctl & !E1000_RFCTL_EXTEN) | E1000_RFCTL_ACK_DIS,
        );
        regs.write32(reg(e1000_txdctl(QUEUE)), TXDCTL_RUN);
        regs.write32(reg(e1000_rxdctl(QUEUE)), RXDCTL_RUN);
        regs.write32(reg(E1000_IVAR), IVAR_ALL_TO_ENTRY0);
        let ctrl_ext = regs.read32(reg(E1000_CTRL_EXT));
        regs.write32(
            reg(E1000_CTRL_EXT),
            (ctrl_ext | E1000_CTRL_EXT_EIAME | E1000_CTRL_EXT_PBA_CLR) & !E1000_CTRL_EXT_IAME,
        );
        regs.write32(reg(E1000_EIAC), EIAC_ARMED);

        Ok(Self {
            product,
            mac,
            tx,
            rx,
            rx_bufs,
            cookies: CookieTable::new(TX_RING),
        })
    }

    /// The station address the NVM loaded.
    pub fn mac(&self) -> MacAddr {
        self.mac
    }

    /// **Start**: enable both engines, post the receive ring, arm the
    /// interrupt mask, and wait — bounded, never refusing — for the link.
    pub fn start<R: Regs>(
        self,
        regs: &R,
        clock: fn() -> Option<u64>,
        pause: impl FnMut(),
    ) -> Result<E1000e<P>> {
        let filter = PacketFilter::DIRECTED;

        // 9. Engines on. `SLU` with forced speed/duplex clear lets the MAC
        //    follow the PHY's negotiation.
        regs.write32(reg(E1000_RCTL), RCTL_RUN | rctl_filter_bits(filter));
        regs.write32(reg(E1000_TCTL), TCTL_RUN);
        let ctrl = regs.read32(reg(E1000_CTRL));
        regs.write32(
            reg(E1000_CTRL),
            (ctrl | E1000_CTRL_SLU) & !(E1000_CTRL_FRCSPD | E1000_CTRL_FRCDPX),
        );

        // 10. Post every receive slot but one: the descriptors were written
        //     in `prepare`, so this is the release fence and the tail.
        fence(Ordering::Release);
        regs.write32(reg(e1000_rdt(QUEUE)), (RX_RING - 1) as u32);

        // 11. Moderate, acknowledge anything pending, then arm. Both
        //     throttle registers take the one interval, because the claim
        //     may have realized either delivery mode (`IMS_ARMED`'s
        //     argument) and an unmoderated fallback would be the silent one.
        regs.write32(reg(E1000_ITR), ITR_INTERVAL);
        regs.write32(reg(e1000_eitr_82574(0)), ITR_INTERVAL);
        let stale = regs.read32(reg(E1000_ICR));
        if stale != 0 {
            regs.write32(reg(E1000_ICR), stale);
        }
        regs.write32(reg(E1000_IMS), IMS_ARMED);

        // The link is a physical fact, not a rung: wait for it within a
        // budget so the row's founding self-test sends into a link that is
        // up, and serve either way.
        let budget = Budget::new(LINK_BUDGET_NS, LINK_SPINS, clock);
        let _ = handshake_field(regs, reg(E1000_STATUS), E1000_STATUS_LU, 1, budget, pause);

        Ok(E1000e {
            product: self.product,
            mac: self.mac,
            tx: self.tx,
            rx: self.rx,
            rx_bufs: self.rx_bufs,
            cookies: self.cookies,
            tx_posted: 0,
            tx_reaped: 0,
            staged: false,
            rx_posted: u64::from(RX_RING - 1),
            rx_read: 0,
            filter,
            counters: NicCounters::default(),
        })
    }
}

/// The running controller.
pub struct E1000e<P: DmaPool> {
    product: Product,
    mac: MacAddr,
    tx: SubRing<TxDesc, P::Buf>,
    rx: CompRing<RxDesc, P::Buf>,
    rx_bufs: P::Buf,
    cookies: CookieTable,
    /// Transmit descriptors staged, ever. `tx_posted % TX_RING` is the next
    /// slot, and equals the ring's tail by construction.
    tx_posted: u64,
    /// Transmit descriptors reaped, ever.
    tx_reaped: u64,
    /// Whether descriptors were staged since the last flush.
    staged: bool,
    /// Receive descriptors handed to the device, ever.
    rx_posted: u64,
    /// Receive descriptors consumed, ever.
    rx_read: u64,
    /// The receive posture last programmed.
    filter: PacketFilter,
    /// What the device has been through: drives that found a cause latched,
    /// `LSC` causes among them, `RXO` overruns, and descriptors written
    /// back with an error or without `EOP`.
    counters: NicCounters,
}

impl<P: DmaPool> E1000e<P> {
    /// The station address.
    pub fn mac(&self) -> MacAddr {
        self.mac
    }

    /// The generation this controller resolved to.
    pub fn product(&self) -> Product {
        self.product
    }

    /// Transmit descriptors posted and not yet reaped.
    pub fn tx_in_flight(&self) -> u16 {
        (self.tx_posted - self.tx_reaped) as u16
    }

    /// Receive descriptors the device currently owns.
    pub fn rx_outstanding(&self) -> u16 {
        (self.rx_posted - self.rx_read) as u16
    }

    /// The device-wide counters, as [`NicBackend::counters`] reports them.
    pub fn counters(&self) -> NicCounters {
        self.counters
    }

    /// **Quiesce**: mask every cause, stop both engines, reset. After `Ok`
    /// the device masters nothing, so the DMA and BAR authority above it can
    /// be retired without racing a transfer.
    pub fn quiesce<R: Regs>(
        &mut self,
        regs: &R,
        clock: fn() -> Option<u64>,
        pause: impl FnMut(),
    ) -> Result<()> {
        regs.write32(reg(E1000_IMC), u32::MAX);
        let rctl = regs.read32(reg(E1000_RCTL));
        regs.write32(reg(E1000_RCTL), rctl & !E1000_RCTL_EN);
        let tctl = regs.read32(reg(E1000_TCTL));
        regs.write32(reg(E1000_TCTL), tctl & !E1000_TCTL_EN);
        let ctrl = regs.read32(reg(E1000_CTRL));
        regs.write32(reg(E1000_CTRL), ctrl | E1000_CTRL_RST);
        let budget = Budget::new(RESET_BUDGET_NS, RESET_SPINS, clock);
        if handshake_field(regs, reg(E1000_CTRL), E1000_CTRL_RST, 0, budget, pause).is_acked() {
            Ok(())
        } else {
            Err(Error::ResetTimeout)
        }
    }

    /// Acknowledge every pending cause and re-arm the mask — the head of
    /// every drive. The controller auto-masks the queue causes when it
    /// delivers their message (`EIAC`), so without this a device would
    /// interrupt exactly once.
    ///
    /// A non-zero `ICR` is one serviced interrupt, whatever it carries: the
    /// causes were latched together and the moderated message reported
    /// them together. `LSC` is counted on its own so a flapping link is
    /// legible; the link *state* is never taken from here — it is read live
    /// from `STATUS` by [`NicBackend::link`], and the service derives the
    /// edge from that.
    fn service_interrupt<R: Regs>(&mut self, regs: &R) {
        let icr = regs.read32(reg(E1000_ICR));
        if icr != 0 {
            regs.write32(reg(E1000_ICR), icr);
            let c = &mut self.counters;
            c.irqs = c.irqs.saturating_add(1);
            if icr & E1000_ICR_LSC != 0 {
                c.link_changes = c.link_changes.saturating_add(1);
            }
            if icr & E1000_ICR_RXO != 0 {
                c.overruns = c.overruns.saturating_add(1);
            }
        }
        regs.write32(reg(E1000_IMS), IMS_ARMED);
    }
}

/// Device address of receive buffer `slot` inside the pool.
fn rx_buf_iova(pool: &impl Dma, slot: u16) -> u64 {
    pool.iova().0 + slot as u64 * RX_BUF_BYTES as u64
}

/// Write receive descriptor `slot` whole, volatile — the device may be
/// reading a neighbour, never this one (the caller owns it).
fn write_rx_desc<B: Dma>(rx: &CompRing<RxDesc, B>, slot: u16, desc: RxDesc) {
    debug_assert!(slot < RX_RING);
    // SAFETY: the ring buffer is `RX_RING * DESC_BYTES` device-visible bytes
    // (allocated so in `prepare`), `slot < RX_RING`, and the descriptor is
    // naturally aligned at a 16-byte multiple of a page-aligned base.
    unsafe {
        ptr::write_volatile((rx.cpu() as *mut RxDesc).add(slot as usize), desc);
    }
}

/// The `RCTL` bits a packet filter asks for. Broadcast is a bit, unicast
/// promiscuity is a bit, and both multicast modes map to multicast
/// promiscuity: the controller's hash table would be a *narrower* filter
/// than the group list a vif names, and the frame engine's classifier is
/// the filter of record either way — over-accepting here costs a copy the
/// classifier drops, never a frame it should have seen.
fn rctl_filter_bits(filter: PacketFilter) -> u32 {
    let mut bits = 0;
    if filter.broadcast {
        bits |= E1000_RCTL_BAM;
    }
    if filter.promiscuous {
        bits |= E1000_RCTL_UPE | E1000_RCTL_MPE;
    }
    if filter.all_multicast || filter.multicast {
        bits |= E1000_RCTL_MPE;
    }
    bits
}

/// The link, read live from `STATUS`.
fn link_from_status(status: u32) -> LinkState {
    if status & E1000_STATUS_LU == 0 {
        return LinkState::Down;
    }
    let mbps = match status & E1000_STATUS_SPEED_MASK {
        E1000_STATUS_SPEED_1000 => 1000,
        E1000_STATUS_SPEED_100 => 100,
        _ => 10,
    };
    LinkState::Up { mbps }
}

/// The class-contract view of a running controller, for one drive cycle.
///
/// A view rather than an owner, like `EcmNic`: the registers belong to the
/// device's resource set and the controller state to the bound device, and
/// the engine borrows both for exactly one sweep.
pub struct E1000eNic<'a, R: Regs, P: DmaPool> {
    regs: &'a R,
    dev: &'a mut E1000e<P>,
}

impl<'a, R: Regs, P: DmaPool> E1000eNic<'a, R, P> {
    /// Borrow `dev` over `regs` for one drive cycle.
    pub fn new(regs: &'a R, dev: &'a mut E1000e<P>) -> Self {
        Self { regs, dev }
    }
}

impl<R: Regs, P: DmaPool> NicBackend for E1000eNic<'_, R, P> {
    type Error = Error;

    fn mac(&self) -> MacAddr {
        self.dev.mac
    }

    fn link(&self) -> LinkState {
        link_from_status(self.regs.read32(reg(E1000_STATUS)))
    }

    fn steering(&self) -> Steering {
        // One queue, no per-vif address filter programmed: the frame
        // service's classifier is the isolation boundary. The 82574L *can*
        // steer (two queues, sixteen exact-match addresses); advertising it
        // waits for the engine that consumes it (`res/PEER_LINK.md` P6).
        Steering::SOFTWARE
    }

    fn counters(&self) -> NicCounters {
        self.dev.counters
    }

    fn set_packet_filter(&mut self, filter: PacketFilter) -> Result<()> {
        let rctl =
            self.regs.read32(reg(E1000_RCTL)) & !(E1000_RCTL_UPE | E1000_RCTL_MPE | E1000_RCTL_BAM);
        self.regs
            .write32(reg(E1000_RCTL), rctl | rctl_filter_bits(filter));
        self.dev.filter = filter;
        Ok(())
    }

    fn transmit(&mut self, frame: TxFrame<'_>) -> Result<()> {
        let dev = &mut *self.dev;
        if dev.tx_posted - dev.tx_reaped >= MAX_POSTED {
            return Err(Error::TxFull);
        }
        let slot = dev.tx.stage(TxDesc::frame(frame.iova().0, frame.len()));
        debug_assert_eq!(u64::from(slot), dev.tx_posted % u64::from(TX_RING));
        dev.cookies.put(slot, frame.cookie().0);
        dev.tx_posted += 1;
        dev.staged = true;
        Ok(())
    }

    fn flush_tx(&mut self) {
        if self.dev.staged {
            let regs = self.regs;
            self.dev
                .tx
                .ring(|tail| regs.write32(reg(e1000_tdt(QUEUE)), u32::from(tail)));
            self.dev.staged = false;
        }
    }

    fn reap_tx(&mut self, out: &mut [TxCompletion]) -> usize {
        let dev = &mut *self.dev;
        dev.service_interrupt(self.regs);
        let status_words = DmaShared::new(dev.tx.buf());
        let mut written = 0;
        while written < out.len() && dev.tx_reaped < dev.tx_posted {
            let slot = (dev.tx_reaped % u64::from(TX_RING)) as u16;
            let at = slot as usize * DESC_BYTES + TXD_STATUS_AT;
            let status = match status_words.read_u8_at(at) {
                Ok(status) => status,
                Err(_) => break,
            };
            let probe = TxDesc {
                status,
                ..TxDesc::default()
            };
            if !probe.done() {
                break;
            }
            // `DD` is set: the device published this write-back — observe the
            // rest coherently, and only then release the slot to its owner.
            fence(Ordering::Acquire);
            out[written] = TxCompletion {
                cookie: TxCookie(dev.cookies.get(slot)),
                status: if probe.failed() {
                    TxStatus::Failed(status)
                } else {
                    TxStatus::Sent
                },
            };
            dev.tx_reaped += 1;
            written += 1;
        }
        written
    }

    fn poll_rx(&mut self, f: &mut dyn FnMut(&[u8])) -> bool {
        let dev = &mut *self.dev;
        let slot = dev.rx.head();
        let Some(desc) = dev.rx.poll() else {
            return false;
        };
        debug_assert_eq!(u64::from(slot), dev.rx_read % u64::from(RX_RING));
        dev.rx_read += 1;
        let len = usize::from(desc.length);
        if !desc.delivers() || len > RX_BUF_BYTES {
            // A descriptor the device wrote back but did not fill with a
            // whole, clean frame occupies its slot and carries no bytes;
            // consuming it frees the slot without inventing a frame.
            dev.counters.rx_errors = dev.counters.rx_errors.saturating_add(1);
            return true;
        }
        // SAFETY: the pool is `RX_RING * RX_BUF_BYTES` bytes; `slot <
        // RX_RING` and `len <= RX_BUF_BYTES`, so the range is inside buffer
        // `slot`; the device wrote it before setting `DD`, which the ring's
        // acquire fence ordered before this read; and the device will not
        // write it again until `refill_rx` re-posts the slot, which happens
        // after this borrow ends.
        let bytes = unsafe {
            core::slice::from_raw_parts(dev.rx_bufs.cpu().add(slot as usize * RX_BUF_BYTES), len)
        };
        f(bytes);
        true
    }

    fn refill_rx(&mut self) -> Result<()> {
        let dev = &mut *self.dev;
        let mut posted = false;
        while dev.rx_posted - dev.rx_read < MAX_POSTED {
            let slot = (dev.rx_posted % u64::from(RX_RING)) as u16;
            // `DD` cleared *before* the tail moves past the slot: that order
            // is what lets validity be the done bit alone.
            write_rx_desc(
                &dev.rx,
                slot,
                RxDesc::empty(rx_buf_iova(&dev.rx_bufs, slot)),
            );
            dev.rx_posted += 1;
            posted = true;
        }
        if posted {
            fence(Ordering::Release);
            self.regs.write32(
                reg(e1000_rdt(QUEUE)),
                (dev.rx_posted % u64::from(RX_RING)) as u32,
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use kore_abi::device::IovaAddr;
    use kore_driver::net::EtherType;
    use kore_driver::net::window::{SLOT_BYTES, TX_AREA_AT, WINDOW_BYTES, tx_slot_at};
    use kore_driver::net::{FrameHead, MacAddr, VifMapping};

    use super::*;
    use crate::mock::{MOCK_MAC, MockE1000e, MockPool};

    fn no_clock() -> Option<u64> {
        None
    }

    fn prepared(mock: &MockE1000e) -> Prepared<MockPool> {
        Prepared::prepare(mock, &MockPool, Product::I82574L, 0x20000, no_clock, || {})
            .expect("prepare")
    }

    fn brought_up(mock: &MockE1000e) -> E1000e<MockPool> {
        prepared(mock).start(mock, no_clock, || {}).expect("start")
    }

    /// A client vif window in process memory: the mock's device domain is
    /// host memory, so the window's address is its IOVA.
    struct Window {
        base: *mut u8,
    }

    impl Window {
        fn new() -> Self {
            let layout = core::alloc::Layout::from_size_align(WINDOW_BYTES, 4096).unwrap();
            // SAFETY: non-zero layout; leaked for the test process.
            let base = unsafe { alloc::alloc::alloc_zeroed(layout) };
            assert!(!base.is_null());
            Self { base }
        }
        fn iova(&self) -> IovaAddr {
            IovaAddr(self.base as u64)
        }
        fn write_tx_slot(&self, slot: u32, bytes: &[u8]) {
            let at = tx_slot_at(slot).unwrap();
            assert!(bytes.len() <= SLOT_BYTES);
            // SAFETY: inside the window, disjoint per slot.
            unsafe {
                core::ptr::copy_nonoverlapping(bytes.as_ptr(), self.base.add(at), bytes.len());
            }
        }
    }

    fn frame(payload: usize, fill: u8) -> Vec<u8> {
        let mut bytes = std::vec![fill; 14 + payload];
        FrameHead {
            dst: MacAddr::BROADCAST,
            src: MacAddr(MOCK_MAC),
            ethertype: EtherType::ARP,
        }
        .write(&mut bytes)
        .unwrap();
        bytes
    }

    #[test]
    fn bring_up_resets_waits_for_the_nvm_and_reads_the_station_address_from_ral0() {
        let mock = MockE1000e::new();
        let dev = brought_up(&mock);
        assert_eq!(dev.mac(), MacAddr(MOCK_MAC));
        assert_eq!(dev.product(), Product::I82574L);
        assert_eq!(mock.resets(), 1);
        assert_eq!(mock.reg(reg(e1000_rah(0))) & E1000_RAH_AV, E1000_RAH_AV);
    }

    #[test]
    fn an_address_valid_bit_clear_refuses_rather_than_inventing_a_mac() {
        let mock = MockE1000e::new();
        mock.no_station_address();
        let err = Prepared::prepare(&mock, &MockPool, Product::I82574L, 0x20000, no_clock, || {})
            .err()
            .expect("refused");
        assert_eq!(err, Error::NoStationAddress);
        assert_eq!(err.rung(), 4);
    }

    #[test]
    fn a_reset_that_never_self_clears_times_out_on_its_budget() {
        let mock = MockE1000e::new();
        mock.hold_reset();
        let err = Prepared::prepare(&mock, &MockPool, Product::I82574L, 0x20000, no_clock, || {})
            .err()
            .expect("refused");
        assert_eq!(err, Error::ResetTimeout);
    }

    #[test]
    fn an_nvm_that_never_loads_is_its_own_rung() {
        let mock = MockE1000e::new();
        mock.nvm_never_loads();
        let err = Prepared::prepare(&mock, &MockPool, Product::I82574L, 0x20000, no_clock, || {})
            .err()
            .expect("refused");
        assert_eq!(err, Error::NvmNotLoaded);
        assert_eq!(err.rung(), 3);
    }

    #[test]
    fn a_short_bar_is_not_present() {
        let mock = MockE1000e::new();
        let err = Prepared::prepare(&mock, &MockPool, Product::I82574L, 0x1000, no_clock, || {})
            .err()
            .expect("refused");
        assert_eq!(err, Error::NotPresent);
        assert_eq!(mock.resets(), 0, "refused before touching the device");
    }

    #[test]
    fn prepare_programs_both_rings_but_enables_neither() {
        let mock = MockE1000e::new();
        let p = prepared(&mock);
        assert_eq!(mock.reg(reg(E1000_RCTL)) & E1000_RCTL_EN, 0);
        assert_eq!(mock.reg(reg(E1000_TCTL)) & E1000_TCTL_EN, 0);
        assert_eq!(mock.reg(reg(e1000_rdt(0))), 0, "nothing posted yet");
        assert_eq!(mock.reg(reg(e1000_tdlen(0))), 4096);
        assert_eq!(mock.reg(reg(e1000_rdlen(0))), 4096);
        let tx_base = u64::from(mock.reg(reg(e1000_tdbal(0))))
            | (u64::from(mock.reg(reg(e1000_tdbah(0)))) << 32);
        assert_eq!(tx_base, p.tx.iova().0);
        assert_ne!(
            mock.reg(reg(e1000_tdbah(0))),
            0,
            "a 64-bit host heap puts the high half in play, as the 4 GiB+ IOVA base will"
        );
        assert_eq!(mock.reg(reg(E1000_IVAR)), IVAR_ALL_TO_ENTRY0);
        assert_eq!(mock.reg(reg(E1000_EIAC)), EIAC_ARMED);
        assert_eq!(mock.reg(reg(E1000_RFCTL)) & E1000_RFCTL_EXTEN, 0, "legacy");
        assert_ne!(mock.reg(reg(E1000_RFCTL)) & E1000_RFCTL_ACK_DIS, 0);
        assert_eq!(mock.reg(reg(E1000_IMS)), 0, "not armed until start");
        for word in 0..MTA_WORDS {
            assert_eq!(mock.reg(reg(E1000_MTA + 4 * word)), 0);
        }
    }

    #[test]
    fn start_enables_both_engines_and_posts_all_but_one_receive_slot() {
        let mock = MockE1000e::new();
        let dev = brought_up(&mock);
        assert_ne!(mock.reg(reg(E1000_RCTL)) & E1000_RCTL_EN, 0);
        assert_ne!(mock.reg(reg(E1000_RCTL)) & E1000_RCTL_SECRC, 0);
        assert_ne!(
            mock.reg(reg(E1000_RCTL)) & E1000_RCTL_BAM,
            0,
            "DIRECTED accepts broadcast"
        );
        assert_eq!(mock.reg(reg(E1000_RCTL)) & E1000_RCTL_UPE, 0);
        assert_eq!(mock.reg(reg(E1000_TCTL)), TCTL_RUN);
        assert_ne!(mock.reg(reg(E1000_CTRL)) & E1000_CTRL_SLU, 0);
        assert_eq!(mock.reg(reg(e1000_rdt(0))), 255);
        assert_eq!(dev.rx_outstanding(), 255);
        assert_eq!(mock.reg(reg(E1000_IMS)), IMS_ARMED);
    }

    /// Interrupt moderation is programmed before the mask is armed, into
    /// both throttle registers — the claim may realize either delivery
    /// mode, and the one left unmoderated would be the one that fires per
    /// frame. The value is the derived interval, not a tuned number.
    #[test]
    fn start_moderates_both_delivery_modes_with_the_derived_interval() {
        let mock = MockE1000e::new();
        let _dev = brought_up(&mock);
        assert_eq!(
            mock.reg(reg(E1000_ITR)),
            ITR_INTERVAL,
            "legacy/MSI throttle"
        );
        assert_eq!(
            mock.reg(reg(e1000_eitr_82574(0))),
            ITR_INTERVAL,
            "the MSI-X vector every cause routes to"
        );
        assert_eq!(ITR_INTERVAL & !E1000_ITR_INTERVAL_MASK, 0);
        assert_eq!(
            u64::from(ITR_INTERVAL) * 256,
            u64::from(crate::regs::ITR_INTERVAL_NS) - u64::from(crate::regs::ITR_INTERVAL_NS) % 256,
            "the register holds the interval in 256 ns units"
        );
    }

    /// A drive that finds a cause latched is one serviced interrupt whatever
    /// the cause carries; a link-status change is counted on its own; a
    /// drive that finds nothing counts nothing. The link *state* is still
    /// read live, so the counter and the state cannot disagree.
    #[test]
    fn counters_report_serviced_interrupts_and_link_changes() {
        let mock = MockE1000e::new();
        let mut dev = brought_up(&mock);
        let mut nic = E1000eNic::new(&mock, &mut dev);
        let mut out = [TxCompletion {
            cookie: TxCookie(0),
            status: TxStatus::Sent,
        }; 1];
        assert_eq!(nic.counters(), NicCounters::default(), "nothing yet");
        nic.reap_tx(&mut out);
        assert_eq!(nic.counters().irqs, 0, "an idle drive is not an interrupt");

        mock.inject_rx(&frame(30, 1));
        mock.inject_rx(&frame(30, 2));
        nic.reap_tx(&mut out);
        let c = nic.counters();
        assert_eq!(
            c.irqs, 1,
            "two frames latched together are one serviced cause"
        );
        assert_eq!(c.link_changes, 0);

        mock.set_link(false);
        assert_eq!(nic.link(), LinkState::Down, "read live, before any drive");
        nic.reap_tx(&mut out);
        mock.set_link(true);
        nic.reap_tx(&mut out);
        let c = nic.counters();
        assert_eq!(c.link_changes, 2, "down and up, one LSC each");
        assert_eq!(c.irqs, 3);
        assert_eq!(nic.link(), LinkState::Up { mbps: 1000 });
        assert_eq!(c.overruns, 0);
        assert_eq!(c.rx_errors, 0);
    }

    /// `RXO` — the controller dropped a frame for want of a posted
    /// descriptor — is counted where it is acknowledged.
    #[test]
    fn a_receiver_overrun_is_counted_when_acknowledged() {
        let mock = MockE1000e::new();
        let mut dev = brought_up(&mock);
        let mut nic = E1000eNic::new(&mock, &mut dev);
        mock.raise_cause(E1000_ICR_RXO);
        let mut out = [TxCompletion {
            cookie: TxCookie(0),
            status: TxStatus::Sent,
        }; 1];
        nic.reap_tx(&mut out);
        assert_eq!(nic.counters().overruns, 1);
        assert_eq!(nic.counters().irqs, 1);
        assert_eq!(mock.reg(reg(E1000_ICR)) & E1000_ICR_RXO, 0, "acknowledged");
    }

    #[test]
    fn transmit_stages_a_descriptor_and_only_flush_moves_the_tail() {
        let mock = MockE1000e::new();
        let mut dev = brought_up(&mock);
        let window = Window::new();
        let bytes = frame(46, 0xa5);
        window.write_tx_slot(0, &bytes);
        let mut mapping = VifMapping::new((), window.iova());
        // Bring-up zeroes the tail once; every write after that is a publish.
        let zeroing = mock.tdt_writes();
        assert_eq!(zeroing, 1);
        {
            let mut nic = E1000eNic::new(&mock, &mut dev);
            mapping
                .transmit(&mut nic, 0, bytes.len() as u16, TxCookie(7))
                .expect("accepted");
            assert_eq!(mock.reg(reg(e1000_tdt(0))), 0, "staged, not published");
            assert_eq!(mock.tdt_writes(), zeroing, "transmit touched no register");
            assert!(mock.transmitted().is_empty());
            nic.flush_tx();
        }
        assert_eq!(mock.reg(reg(e1000_tdt(0))), 1);
        assert_eq!(mock.transmitted(), std::vec![bytes.clone()]);
        assert_eq!(mock.tdt_writes(), zeroing + 1);
        {
            // A second flush with nothing staged rings nothing.
            let mut nic = E1000eNic::new(&mock, &mut dev);
            nic.flush_tx();
        }
        assert_eq!(mock.tdt_writes(), zeroing + 1);
        let desc = mock.tx_desc(0);
        assert_eq!(desc.cmd & E1000_TXD_CMD_DEXT as u8, 0, "legacy");
        assert_eq!(desc.cmd, TXD_CMD_FRAME_FOR_TEST);
        assert_eq!(desc.addr, window.iova().0 + TX_AREA_AT as u64);
    }
    const TXD_CMD_FRAME_FOR_TEST: u8 = 0x0b;

    #[test]
    fn reap_returns_each_written_back_cookie_in_ring_order() {
        let mock = MockE1000e::new();
        let mut dev = brought_up(&mock);
        let window = Window::new();
        let mut mapping = VifMapping::new((), window.iova());
        let mut nic = E1000eNic::new(&mock, &mut dev);
        for slot in 0..3u32 {
            window.write_tx_slot(slot, &frame(30 + slot as usize, slot as u8));
            mapping
                .transmit(
                    &mut nic,
                    slot,
                    (44 + slot) as u16,
                    TxCookie(100 + u64::from(slot)),
                )
                .expect("accepted");
        }
        let mut out = [TxCompletion {
            cookie: TxCookie(0),
            status: TxStatus::Sent,
        }; 8];
        assert_eq!(
            nic.reap_tx(&mut out),
            0,
            "nothing written back before the flush"
        );
        nic.flush_tx();
        let n = nic.reap_tx(&mut out);
        assert_eq!(n, 3);
        assert_eq!(out[0].cookie, TxCookie(100));
        assert_eq!(out[1].cookie, TxCookie(101));
        assert_eq!(out[2].cookie, TxCookie(102));
        assert!(out[..3].iter().all(|c| c.status == TxStatus::Sent));
        assert_eq!(nic.dev.tx_in_flight(), 0);
        mapping.completed(3);
        assert!(mapping.retire().is_ok());
    }

    #[test]
    fn the_transmit_ring_refuses_when_one_slot_short_of_full_and_frees_after_reap() {
        let mock = MockE1000e::new();
        mock.hold_tx_writeback();
        let mut dev = brought_up(&mock);
        let window = Window::new();
        let mut mapping = VifMapping::new((), window.iova());
        let mut nic = E1000eNic::new(&mock, &mut dev);
        window.write_tx_slot(0, &frame(46, 1));
        for _ in 0..255 {
            mapping
                .transmit(&mut nic, 0, 60, TxCookie(1))
                .expect("255 fit");
        }
        assert_eq!(
            mapping.transmit(&mut nic, 0, 60, TxCookie(1)).err(),
            Some(kore_driver::net::TxError::Backend(Error::TxFull)),
            "the 256th would make head == tail"
        );
        nic.flush_tx();
        mock.release_tx_writeback();
        let mut out = [TxCompletion {
            cookie: TxCookie(0),
            status: TxStatus::Sent,
        }; 16];
        let mut reaped = 0;
        loop {
            let n = nic.reap_tx(&mut out);
            if n == 0 {
                break;
            }
            reaped += n;
        }
        assert_eq!(reaped, 255);
        assert!(mapping.transmit(&mut nic, 0, 60, TxCookie(2)).is_ok());
    }

    #[test]
    fn a_frame_on_the_wire_lands_by_ring_order_and_refill_reposts_the_slot() {
        let mock = MockE1000e::new();
        let mut dev = brought_up(&mock);
        let mut nic = E1000eNic::new(&mock, &mut dev);
        let a = frame(50, 0x11);
        let b = frame(70, 0x22);
        mock.inject_rx(&a);
        mock.inject_rx(&b);
        let mut got: Vec<Vec<u8>> = Vec::new();
        assert!(nic.poll_rx(&mut |bytes| got.push(bytes.to_vec())));
        assert!(nic.poll_rx(&mut |bytes| got.push(bytes.to_vec())));
        assert!(!nic.poll_rx(&mut |_| panic!("nothing more")));
        assert_eq!(got, std::vec![a, b]);
        assert_eq!(nic.dev.rx_outstanding(), 253);
        assert_eq!(mock.reg(reg(e1000_rdt(0))), 255, "the tail has not moved");
        nic.refill_rx().unwrap();
        assert_eq!(nic.dev.rx_outstanding(), 255);
        assert_eq!(
            mock.reg(reg(e1000_rdt(0))),
            1,
            "255 → 0 → 1: two slots re-posted"
        );
        assert_eq!(mock.rx_desc(0).status, 0, "DD cleared on re-post");
        assert_eq!(
            mock.rx_desc(0).addr,
            mock.rx_desc_addr_expected(&nic.dev.rx_bufs, 0)
        );
    }

    #[test]
    fn receive_wraps_the_ring_without_consulting_the_phase() {
        let mock = MockE1000e::new();
        let mut dev = brought_up(&mock);
        let mut nic = E1000eNic::new(&mock, &mut dev);
        let mut seen = 0u32;
        for round in 0..3u8 {
            for i in 0..100u8 {
                mock.inject_rx(&frame(20, i.wrapping_add(round)));
            }
            while nic.poll_rx(&mut |bytes| {
                assert_eq!(bytes.len(), 34);
                seen += 1;
            }) {}
            nic.refill_rx().unwrap();
            assert_eq!(nic.dev.rx_outstanding(), 255);
        }
        assert_eq!(
            seen, 300,
            "every frame delivered exactly once across the wrap"
        );
        assert!(!nic.poll_rx(&mut |_| panic!("a stale DD would re-deliver")));
    }

    #[test]
    fn a_frame_round_trips_through_loopback() {
        let mock = MockE1000e::new();
        mock.set_loopback(true);
        let mut dev = brought_up(&mock);
        let window = Window::new();
        let bytes = frame(46, 0x5a);
        window.write_tx_slot(3, &bytes);
        let mut mapping = VifMapping::new((), window.iova());
        let mut nic = E1000eNic::new(&mock, &mut dev);
        mapping
            .transmit(&mut nic, 3, bytes.len() as u16, TxCookie(9))
            .unwrap();
        nic.flush_tx();
        let mut got = None;
        assert!(nic.poll_rx(&mut |b| got = Some(b.to_vec())));
        assert_eq!(got.unwrap(), bytes);
    }

    #[test]
    fn an_error_descriptor_is_consumed_without_inventing_bytes() {
        let mock = MockE1000e::new();
        let mut dev = brought_up(&mock);
        let mut nic = E1000eNic::new(&mock, &mut dev);
        mock.inject_rx_with_errors(&frame(40, 1), E1000_RXD_ERR_CE as u8);
        mock.inject_rx(&frame(40, 2));
        assert!(nic.poll_rx(&mut |_| panic!("an errored frame is not delivered")));
        assert_eq!(nic.counters().rx_errors, 1);
        let mut n = 0;
        assert!(nic.poll_rx(&mut |_| n += 1));
        assert_eq!(n, 1, "the clean frame behind it still arrives");
    }

    #[test]
    fn the_packet_filter_programs_upe_and_mpe_and_keeps_bam() {
        let mock = MockE1000e::new();
        let mut dev = brought_up(&mock);
        let mut nic = E1000eNic::new(&mock, &mut dev);
        nic.set_packet_filter(PacketFilter::PROMISCUOUS).unwrap();
        let rctl = mock.reg(reg(E1000_RCTL));
        assert_ne!(rctl & E1000_RCTL_UPE, 0);
        assert_ne!(rctl & E1000_RCTL_MPE, 0);
        assert_ne!(rctl & E1000_RCTL_BAM, 0);
        assert_ne!(rctl & E1000_RCTL_EN, 0, "the engine stays on");
        nic.set_packet_filter(PacketFilter::DIRECTED).unwrap();
        let rctl = mock.reg(reg(E1000_RCTL));
        assert_eq!(rctl & (E1000_RCTL_UPE | E1000_RCTL_MPE), 0);
        assert_ne!(rctl & E1000_RCTL_BAM, 0);
    }

    #[test]
    fn link_state_follows_status_lu_and_speed() {
        let mock = MockE1000e::new();
        let mut dev = brought_up(&mock);
        let nic = E1000eNic::new(&mock, &mut dev);
        assert_eq!(nic.link(), LinkState::Up { mbps: 1000 });
        mock.set_link(false);
        assert_eq!(nic.link(), LinkState::Down);
    }

    #[test]
    fn a_link_that_never_comes_up_is_not_a_rung() {
        let mock = MockE1000e::new();
        mock.set_link(false);
        let dev = brought_up(&mock);
        assert_eq!(dev.mac(), MacAddr(MOCK_MAC));
        assert_eq!(mock.reg(reg(E1000_STATUS)) & E1000_STATUS_LU, 0);
    }

    #[test]
    fn every_drive_acknowledges_icr_and_rearms_what_the_notify_auto_masked() {
        let mock = MockE1000e::new();
        let mut dev = brought_up(&mock);
        let mut nic = E1000eNic::new(&mock, &mut dev);
        mock.inject_rx(&frame(30, 1));
        assert_eq!(mock.notifies(), std::vec![0], "one message, entry 0");
        assert_eq!(
            mock.reg(reg(E1000_IMS)) & E1000_ICR_RXQ0,
            0,
            "the controller auto-masked the queue cause on delivery"
        );
        mock.inject_rx(&frame(30, 2));
        assert_eq!(mock.notifies().len(), 1, "masked: no second message");
        let mut out = [TxCompletion {
            cookie: TxCookie(0),
            status: TxStatus::Sent,
        }; 1];
        nic.reap_tx(&mut out);
        assert_eq!(mock.reg(reg(E1000_IMS)), IMS_ARMED, "the drive re-armed");
        assert_eq!(mock.reg(reg(E1000_ICR)), 0, "…and acknowledged everything");
        mock.inject_rx(&frame(30, 3));
        assert_eq!(mock.notifies().len(), 2);
    }

    #[test]
    fn the_ivar_routes_every_cause_to_entry_zero() {
        let mock = MockE1000e::new();
        let mut dev = brought_up(&mock);
        let window = Window::new();
        window.write_tx_slot(0, &frame(46, 1));
        let mut mapping = VifMapping::new((), window.iova());
        let mut nic = E1000eNic::new(&mock, &mut dev);
        mapping.transmit(&mut nic, 0, 60, TxCookie(1)).unwrap();
        nic.flush_tx();
        mock.set_link(false);
        mock.inject_rx(&frame(30, 1));
        assert!(!mock.notifies().is_empty());
        assert!(mock.notifies().iter().all(|&v| v == 0));
    }

    #[test]
    fn the_backend_declares_no_hardware_steering() {
        let mock = MockE1000e::new();
        let mut dev = brought_up(&mock);
        let nic = E1000eNic::new(&mock, &mut dev);
        assert_eq!(nic.steering(), Steering::SOFTWARE);
    }

    #[test]
    fn quiesce_masks_interrupts_and_stops_both_engines() {
        let mock = MockE1000e::new();
        let mut dev = brought_up(&mock);
        dev.quiesce(&mock, no_clock, || {}).unwrap();
        assert_eq!(mock.reg(reg(E1000_IMS)), 0);
        assert_eq!(mock.reg(reg(E1000_RCTL)) & E1000_RCTL_EN, 0);
        assert_eq!(mock.reg(reg(E1000_TCTL)) & E1000_TCTL_EN, 0);
        assert_eq!(mock.resets(), 2);
    }

    #[test]
    fn the_receive_length_excludes_the_fcs() {
        let mock = MockE1000e::new();
        let mut dev = brought_up(&mock);
        let mut nic = E1000eNic::new(&mock, &mut dev);
        let full = frame(1500, 0xee);
        assert_eq!(full.len(), 1514);
        mock.inject_rx(&full);
        let mut len = 0;
        assert!(nic.poll_rx(&mut |b| len = b.len()));
        assert_eq!(len, 1514, "SECRC: the FCS never reaches the classifier");
    }
}
