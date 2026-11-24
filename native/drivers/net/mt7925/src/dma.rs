//! **WFDMA — the engine the firmware download rides.**
//!
//! Three things live here and nothing else: how far this part can *reach*,
//! what one descriptor *is*, and where a queue's four control words *are*.
//!
//! > **This module is a seam, not a live path.** Nothing under the `boot`
//! > feature calls [`Descriptor`], [`QueueRegs`] or [`Queue`] yet — the D1
//! > ladder that programs them has not landed. The one piece that *is* live
//! > is [`DMA_ADDR_BITS`], which the net row's binding declares as its
//! > `PciBindingContract::DMA_ADDRESS_BITS` and which therefore decides
//! > where the host places this part's pool on every boot.
//!
//! # The reach is 32 bits, and that is the first thing to read
//!
//! `struct mt76_desc`'s buffer pointer is a **`__le32`** (`dma.h:120-125`),
//! the vendor driver asks the kernel for `DMA_BIT_MASK(32)` on everything
//! that is not an MT7928 (`mt7925/pci.c:651`), and
//! `MT_WFDMA0_GLO_CFG_ADDR_EXT_EN` is set only by `mt7927_wfdma_setup`
//! (`mt792x_dma.c:199`) and the MT7928 branch of `mt792x_dma_enable`
//! (`:305`) — never for this part. Three readings, one answer: **every
//! address this engine can emit is below 4 GiB.**
//!
//! `kore_driver::host::DMA_IOVA_BASE` is `0x1_0000_0000` — 4 GiB exactly —
//! so on the tree as it stands the *first* buffer a driver draws already
//! sits at the ceiling. [`addressable`] is therefore not a formality: it is
//! the check that turns a silent address truncation into a named rung.
//! `res/MT7925_EVIDENCE.md` §15 is the whole finding.
//!
//! # Why the substrate's rings fit here and the staging does not
//!
//! A WFDMA queue is a fixed array of identical 16-byte descriptors indexed
//! by a pair of head/tail registers, which is exactly
//! [`SubRing`](kore_driver::SubRing) / [`CompRing`](kore_driver::CompRing)'s
//! shape — and [`Descriptor`]'s `valid` is e1000e's reading of the same
//! problem: `DMA_DONE` is a **done bit, not a phase bit**, so it is tested
//! against itself and the ring's phase is ignored, with the re-zero on
//! re-post doing the work the phase would have done.

use core::mem::size_of;

use kore_driver::{CqEntry, SqEntry};

use crate::regs::mt7925::*;

/// Every target this crate builds for is little-endian, and [`Descriptor`]
/// is read by the part's own engine as little-endian words. A big-endian
/// host would need byte encoders instead of `#[repr(C)]`, so the assumption
/// is stated where it would break rather than left to a reader.
const _: () = assert!(cfg!(target_endian = "little"));

/// How many address bits this part's DMA master can emit.
///
/// **32**, with three independent citations in the module docs. The MT7927
/// and MT7928 siblings are wider; they take a different fixed map and are
/// not admitted by [`Product::from_pci`](crate::Product::from_pci), so this
/// constant does not need to be a per-product field until one of them is.
pub const DMA_ADDR_BITS: u32 = 32;

/// Whether a buffer is wholly inside this part's reach.
///
/// The arithmetic is `kore_driver::within_dma_reach`'s, shared with
/// `kore_rtw89::desc::addressable` — the same question of a 36-bit part.
/// Only the width is this crate's, because only the width is a fact about
/// this silicon.
#[inline]
pub const fn addressable(iova: u64, len: u64) -> bool {
    kore_driver::within_dma_reach(iova, len, DMA_ADDR_BITS)
}

/// One WFDMA descriptor — `struct mt76_desc`, `dma.h:120-125`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Descriptor {
    /// The buffer's device address. **32 bits**; see [`DMA_ADDR_BITS`].
    pub buf0: u32,
    /// `MT_DMA_CTL_*` — the segment length, the last-segment flag, and the
    /// done bit the engine sets.
    pub ctrl: u32,
    /// The second segment's address. Always zero here: every buffer this
    /// driver posts is one segment.
    pub buf1: u32,
    /// Per-descriptor information. Zero for both of the download's queues.
    pub info: u32,
}

/// Bytes of one [`Descriptor`] on the wire.
pub const DESC_BYTES: usize = 16;
const _: () = assert!(size_of::<Descriptor>() == DESC_BYTES);

/// The largest segment one descriptor can describe.
///
/// `MT_DMA_CTL_SD_LEN0` is `GENMASK(29, 16)` — fourteen bits, so 16,383
/// bytes. The download's own chunk is 4,096 ([`crate::mcu::FW_SCATTER_MAX`])
/// and sits well inside it; the bound exists so that a future caller cannot
/// silently truncate a length into the field.
pub const MAX_SEGMENT: u32 = MT_DMA_CTL_SD_LEN0 >> MT_DMA_CTL_SD_LEN0.trailing_zeros();

impl Descriptor {
    /// A transmit descriptor for one whole buffer.
    ///
    /// `mt76_dma_add_buf` with `nbufs == 1` (`dma.c:321-388`): the length
    /// goes in `SD_LEN0`, `LAST_SEC0` marks the end of the packet, and
    /// `buf1`/`info` stay zero because there is no second segment.
    ///
    /// Returns `None` rather than truncating when the buffer is out of the
    /// part's reach or longer than one segment — the two ways this encoding
    /// can be a lie.
    pub const fn transmit(iova: u64, len: u32) -> Option<Self> {
        if len == 0 || len > MAX_SEGMENT || !addressable(iova, len as u64) {
            return None;
        }
        Some(Self {
            buf0: iova as u32,
            ctrl: (len << MT_DMA_CTL_SD_LEN0.trailing_zeros()) | MT_DMA_CTL_LAST_SEC0,
            buf1: 0,
            info: 0,
        })
    }

    /// A receive descriptor handing the engine an empty buffer.
    ///
    /// `mt76_dma_add_rx_buf` (`dma.c:247-320`): the same length field, and
    /// **no** done bit — the engine sets `MT_DMA_CTL_DMA_DONE` when it has
    /// written the buffer, which is what [`CqEntry::valid`] reads.
    pub const fn receive(iova: u64, len: u32) -> Option<Self> {
        if len == 0 || len > MAX_SEGMENT || !addressable(iova, len as u64) {
            return None;
        }
        Some(Self {
            buf0: iova as u32,
            ctrl: len << MT_DMA_CTL_SD_LEN0.trailing_zeros(),
            buf1: 0,
            info: 0,
        })
    }

    /// Bytes the engine wrote into this descriptor's buffer.
    pub const fn written(self) -> u32 {
        (self.ctrl & MT_DMA_CTL_SD_LEN0) >> MT_DMA_CTL_SD_LEN0.trailing_zeros()
    }

    /// Whether the engine has finished with this descriptor.
    pub const fn done(self) -> bool {
        self.ctrl & MT_DMA_CTL_DMA_DONE != 0
    }

    /// This descriptor's control word with the done bit taken back out.
    ///
    /// The engine sets that bit and never clears it, so a slot whose
    /// answer has already been read is indistinguishable from one the
    /// engine has just filled. Writing this back makes an answer
    /// *consumed* — a second read of the same slot waits, as it should,
    /// instead of returning a reply that was already delivered.
    pub const fn consumed(self) -> u32 {
        self.ctrl & !MT_DMA_CTL_DMA_DONE
    }

    /// The four words, in the order the engine reads them.
    ///
    /// A descriptor table is memory the engine reads and writes while the
    /// driver runs, so it is published a word at a time through
    /// [`kore_driver::seam::DmaShared`] rather than stored as a struct.
    /// This is the only place that says what that word order is.
    pub const fn words(self) -> [u32; 4] {
        [self.buf0, self.ctrl, self.buf1, self.info]
    }

    /// A descriptor known only by its control word.
    ///
    /// What a completion poll actually reads: one volatile `u32`, not a
    /// sixteen-byte struct. Reconstituting it here keeps [`Self::done`] and
    /// [`Self::written`] the only readers of those bit fields.
    pub const fn from_ctrl(ctrl: u32) -> Self {
        Self {
            buf0: 0,
            ctrl,
            buf1: 0,
            info: 0,
        }
    }

    /// Byte offset of slot `n`'s first word within a descriptor table.
    pub const fn slot_at(n: usize) -> usize {
        n * DESC_BYTES
    }

    /// Byte offset of slot `n`'s control word — the one the engine sets
    /// [`MT_DMA_CTL_DMA_DONE`] in, and the only word a completion poll
    /// needs to read.
    pub const fn ctrl_at(n: usize) -> usize {
        n * DESC_BYTES + 4
    }
}

impl SqEntry for Descriptor {
    // No in-entry cycle bit: the engine is told about new work by the
    // queue's `cpu_idx` register, exactly as e1000e's `TDT` does it.
}

impl CqEntry for Descriptor {
    /// **The done bit is not a phase bit.**
    ///
    /// `MT_DMA_CTL_DMA_DONE` is set by the engine in the slot the driver
    /// posted and does not alternate on a wrap, so the ring's `phase` is
    /// ignored and the driver's re-post — which writes the control word
    /// without the bit — is what keeps a stale completion from being read
    /// twice. This is `kore_e1000e::desc::RxDesc`'s reading of the same
    /// problem, and deliberately not a second one.
    fn valid(&self, _phase: bool) -> bool {
        self.done()
    }
}

/// The four control words of one WFDMA queue.
///
/// `struct mt76_queue_regs` (`mt76.h:236-241`), banked at
/// `ring_base + qid * MT_RING_SIZE` (`mt76_dma_alloc_queue`, `dma.c:838`).
/// A type rather than four additions at each call site: the stride and the
/// base are the two things a transcription gets wrong, and a wrong bank
/// programs a queue the part is not using.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueueRegs {
    base: u32,
}

impl QueueRegs {
    /// The bank of a transmit queue.
    pub const fn tx(qid: u32) -> Self {
        Self {
            base: MT_TX_RING_BASE + qid * MT_RING_SIZE,
        }
    }

    /// The bank of a receive (event) queue.
    pub const fn rx(qid: u32) -> Self {
        Self {
            base: MT_RX_EVENT_RING_BASE + qid * MT_RING_SIZE,
        }
    }

    /// The bank of a receive **data** queue.
    ///
    /// Upstream names a second base for these (`MT_RX_DATA_RING_BASE`,
    /// `mt7925/regs.h:78`) and it is the same `MT_WFDMA0(0x500)` as the
    /// event base on this part. Spelled separately anyway, so the bank is
    /// the one upstream's layout names and a part where the two differ
    /// changes one constant rather than a silent sum.
    pub const fn rx_data(qid: u32) -> Self {
        Self {
            base: MT_RX_DATA_RING_BASE + qid * MT_RING_SIZE,
        }
    }

    /// Where the ring's base device address is written.
    pub const fn desc_base(self) -> u32 {
        self.base
    }

    /// Where the ring's slot count is written.
    pub const fn ring_size(self) -> u32 {
        self.base + 4
    }

    /// The doorbell — the driver's producer index.
    pub const fn cpu_idx(self) -> u32 {
        self.base + 8
    }

    /// The engine's own index.
    pub const fn dma_idx(self) -> u32 {
        self.base + 0xc
    }
}

/// The queues this driver programs.
///
/// The transmit data queue (`MT7925_TXQ_BAND0`) is M-ladder work and is
/// deliberately absent. The receive data queue was too, on the reasoning
/// that a queue programmed and never serviced is a ring the part can fill
/// while nobody reads it — and that reasoning held while it was never
/// programmed *and* never fetched. It was both less and more than that: the
/// prefetch window for it was always written (`PREFETCH_SETUP`), over a
/// base and count left at their reset values. See [`Queue::EventData`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Queue {
    /// Commands to the WM core — everything but the firmware payload.
    McuWm,
    /// The firmware-download queue. `mt7925_mcu_send_message` routes
    /// `FW_SCATTER` here and every other command to [`Queue::McuWm`]
    /// (`mt7925/pci_mcu.c:21-24`).
    Fwdl,
    /// Events from the WM core — where every command's response arrives.
    EventWm,
    /// **Receive ring 2** (`MT7925_RXQ_BAND0`), programmed from M1 as a
    /// second place an event may land, and serviced by the exchange.
    ///
    /// Upstream programs it before the firmware download
    /// (`mt7925_dma_layout`, `mt7925/pci.c:392`), and its receive path
    /// dispatches `PKT_TYPE_RX_EVENT` from **any** ring
    /// (`mt7925_queue_rx_skb`, `mt7925/mac.c:1451`). Three metal photos
    /// (2026-11-18/19) show the RAM firmware alive, unalarmed, fetching the
    /// first unified command — and nothing ever arriving on ring 0. The ROM
    /// answered there; the RAM code may not.
    EventData,
}

/// The producer index that hands a freshly armed receive ring to the engine.
///
/// **One short of the depth**, and never the depth itself. Upstream fills a
/// receive ring while `q->queued < q->ndesc - 1` (`mt76_dma_rx_fill`,
/// `dma.c`) and kicks `cpu_idx` to that head, so a full ring and an empty one
/// never share an index — and the index is always a slot that exists.
///
/// This driver used to post `depth` on ring 0: `cpu_idx = 512` on a
/// 512-slot ring, an index one past its last slot. The ROM answered over it
/// (C2's twelve replies), which is how it survived to M1; the RAM firmware,
/// which re-reads the ring's state when it takes over, is the first reader
/// that had to make sense of it.
pub const fn rx_fill_index(depth: u16) -> u16 {
    depth - 1
}

/// How many slots receive ring 2 is given — **not** upstream's 1,536.
///
/// `MT7925_RX_RING_SIZE` sizes the ring that carries every received frame
/// on a working link: 3 MiB of buffers the part may write, against a
/// 2 MiB pool this row is endowed with. At M1 the ring carries no frames —
/// no interface exists to receive them — so its only possible traffic is
/// events, and its depth is a question this rung can answer by measuring
/// rather than by transcription. The count register is the contract: the
/// part walks the depth it is told.
///
/// Thirty-two, which is eight of the four-slot prefetch windows
/// `PREFETCH_SETUP` gives the ring, at 64 KiB of buffers. If the exchange
/// ever reports this ring full, that is the measurement that sizes the
/// data path's ring, and the reason to grow it.
pub const EVENT_DATA_DEPTH: u16 = 32;

impl Queue {
    /// This queue's register bank.
    pub const fn regs(self) -> QueueRegs {
        match self {
            Self::McuWm => QueueRegs::tx(MT7925_TXQ_MCU_WM),
            Self::Fwdl => QueueRegs::tx(MT7925_TXQ_FWDL),
            Self::EventWm => QueueRegs::rx(MT7925_RXQ_MCU_WM),
            Self::EventData => QueueRegs::rx_data(MT7925_RXQ_BAND0),
        }
    }

    /// How many descriptors upstream gives this queue.
    ///
    /// Taken verbatim (`mt7925/mt7925.h:18,19,22`) rather than re-derived:
    /// the depths are what the part's own scheduler was tuned against, and
    /// a smaller ring is a throughput decision this rung has no measurement
    /// to make. **Except ring 2**, which carries no throughput at M1 — see
    /// [`EVENT_DATA_DEPTH`].
    pub const fn depth(self) -> u16 {
        match self {
            Self::McuWm => MT7925_TX_MCU_RING_SIZE as u16,
            Self::Fwdl => MT7925_TX_FWDL_RING_SIZE as u16,
            Self::EventWm => MT7925_RX_MCU_RING_SIZE as u16,
            Self::EventData => EVENT_DATA_DEPTH,
        }
    }

    /// Bytes of descriptor table this queue needs.
    pub const fn table_bytes(self) -> usize {
        self.depth() as usize * DESC_BYTES
    }
}

/// `GENMASK(18, 4)`, the last term of `MT_INT_TX_DONE_ALL`
/// (`mt7925/regs.h:74-76`) — written out because upstream spells it inline
/// rather than naming it.
const TX_DONE_GENMASK_18_4: u32 = 0x0007_fff0;

/// **The interrupt mask the vendor arms on this part, verbatim.**
///
/// `mt792x_dma_enable` ends with `mt76_connac_irq_enable(tx.all_complete_mask
/// | rx.all_complete_mask | MT_INT_MCU_CMD)` (`mt792x_dma.c:352-355`); for
/// the MT7925 those are `MT_INT_TX_DONE_ALL` (`mt7925/regs.h:72-76`),
/// `MT7925_INT_RX_DONE_ALL` (`:50`) and `MCU2HOST_SW_INT_ENA`. The 2026-11-19
/// capture of `mt7925e` on this machine writes exactly this word,
/// `0x2e07fff5`, and holds it for the whole firmware load.
///
/// This replaced a narrowed `DOWNLOAD_INT_MASK` that was never written at
/// all — the engine ran with `HOST_INT_ENA` at zero through D1, C2 and M1,
/// on the reasoning that a polled ladder needs no interrupt. M1b's RAM
/// firmware fetched its first unified command and answered nothing on six
/// flashes, with this the last visible difference from the capture. The
/// bits for rings this driver does not program cannot fire.
pub const HOST_INT_MASK: u32 = HOST_TX_DONE_INT_ENA15
    | HOST_TX_DONE_INT_ENA17
    | HOST_TX_DONE_INT_ENA16
    | HOST_TX_DONE_INT_ENA0
    | TX_DONE_GENMASK_18_4
    | HOST_RX_DONE_INT_ENA2
    | HOST_RX_DONE_INT_ENA0
    | MCU2HOST_SW_INT_ENA;

/// `MT7925_PCIE_MAC_INT_ENABLE`'s armed value: every source.
/// `mt7925_pci_probe` writes `0xff` (`mt7925/pci.c:747`), and so does the
/// capture, once, before the engine is touched.
pub const PCIE_MAC_INT_ALL: u32 = 0xff;

/// The `MT_WFDMA0_GLO_CFG` word `mt792x_dma_enable` **sets** before the two
/// enable bits, for a part that is not an MT7928 (`mt792x_dma.c:330-340`).
///
/// `DMA_SIZE` is 3 there against the MT7928's 1; both are transcribed
/// values, not choices made here.
pub const GLO_CFG_SETUP: u32 = MT_WFDMA0_GLO_CFG_TX_WB_DDONE
    | MT_WFDMA0_GLO_CFG_FIFO_LITTLE_ENDIAN
    | MT_WFDMA0_GLO_CFG_CLK_GAT_DIS
    | MT_WFDMA0_GLO_CFG_OMIT_TX_INFO
    | (3 << MT_WFDMA0_GLO_CFG_DMA_SIZE.trailing_zeros())
    | MT_WFDMA0_GLO_CFG_FIFO_DIS_CHECK
    | MT_WFDMA0_GLO_CFG_RX_WB_DDONE
    | MT_WFDMA0_GLO_CFG_CSR_DISP_BASE_PTR_CHAIN_EN
    | MT_WFDMA0_GLO_CFG_OMIT_RX_INFO_PFET2;

/// `PREFETCH(base, depth)` — `mt792x_dma.c:155`.
///
/// One 32-bit word: the window's byte offset into the engine's shared
/// descriptor cache in the high half, the number of descriptors it holds
/// in the low half.
pub const fn prefetch(base: u32, depth: u32) -> u32 {
    (base << 16) | depth
}

/// Every write `mt7925_dma_prefetch_setup` makes, in its own order
/// (`mt792x_dma.c:157-172`).
///
/// **`GLO_CFG_SETUP` sets `CSR_DISP_BASE_PTR_CHAIN_EN`, which is the bit
/// that makes the engine use these windows at all.** Enable the chain and
/// leave the windows at reset and every ring's slice is a zero-length
/// window at offset zero — ten rings aliased onto one another. An engine
/// with only receive work never notices, because a receive descriptor is
/// fetched on the engine's own schedule from a slice it has to itself.
/// The first *transmit* fetch is what walks the collision.
///
/// All ten are written, not just the three this ladder posts to: the
/// windows are carved out of one shared pool, so leaving the seven unused
/// rings pointing at offset zero would overlap the three that are used.
pub const PREFETCH_SETUP: [(u32, u32); 10] = [
    // The receive rings.
    (MT_WFDMA0_RX_RING0_EXT_CTRL, prefetch(0x0000, 0x4)),
    (MT_WFDMA0_RX_RING1_EXT_CTRL, prefetch(0x0040, 0x4)),
    (MT_WFDMA0_RX_RING2_EXT_CTRL, prefetch(0x0080, 0x4)),
    (MT_WFDMA0_RX_RING3_EXT_CTRL, prefetch(0x00c0, 0x4)),
    // The transmit rings. 15 is `McuWm` and 16 is `Fwdl`.
    (MT_WFDMA0_TX_RING0_EXT_CTRL, prefetch(0x0100, 0x10)),
    (MT_WFDMA0_TX_RING1_EXT_CTRL, prefetch(0x0200, 0x10)),
    (MT_WFDMA0_TX_RING2_EXT_CTRL, prefetch(0x0300, 0x10)),
    (MT_WFDMA0_TX_RING3_EXT_CTRL, prefetch(0x0400, 0x10)),
    (MT_WFDMA0_TX_RING15_EXT_CTRL, prefetch(0x0500, 0x4)),
    (MT_WFDMA0_TX_RING16_EXT_CTRL, prefetch(0x0540, 0x4)),
];

/// The two arbitration-priority words `mt7925_wfdma_setup` **sets**
/// (`mt792x_dma.c:176-177`). Transcribed literals; upstream gives them no
/// symbolic form.
pub const INT_RX_PRI_SETUP: u32 = 0x0F00;
/// See [`INT_RX_PRI_SETUP`].
pub const INT_TX_PRI_SETUP: u32 = 0x7F00;

/// The bits `mt792x_dma_disable` clears (`mt792x_dma.c:449-455`).
pub const GLO_CFG_TEARDOWN: u32 = MT_WFDMA0_GLO_CFG_TX_DMA_EN
    | MT_WFDMA0_GLO_CFG_RX_DMA_EN
    | MT_WFDMA0_GLO_CFG_CSR_DISP_BASE_PTR_CHAIN_EN
    | MT_WFDMA0_GLO_CFG_OMIT_TX_INFO
    | MT_WFDMA0_GLO_CFG_OMIT_RX_INFO
    | MT_WFDMA0_GLO_CFG_OMIT_RX_INFO_PFET2;

/// Both busy bits, which must read clear before the engine is quiet.
pub const GLO_CFG_BUSY: u32 =
    MT_WFDMA0_GLO_CFG_TX_DMA_BUSY | MT_WFDMA0_GLO_CFG_RX_DMA_BUSY;

#[cfg(test)]
mod tests {
    use super::*;

    /// The part's reach is 32 bits, and the pool's floor is exactly there.
    ///
    /// **This test documents a blocker, not a guarantee.**
    /// `kore_driver::host::DMA_IOVA_BASE` is `0x1_0000_0000`, so the first
    /// buffer a driver draws from the production pool is already past this
    /// ceiling — `res/MT7925_EVIDENCE.md` §15. The check below is what makes
    /// that a named refusal instead of a truncated descriptor.
    #[test]
    fn the_parts_reach_is_four_gibibytes() {
        assert_eq!(DMA_ADDR_BITS, 32);
        assert!(addressable(0, 4096));
        assert!(addressable(0xffff_f000, 0x1000), "the last page is reachable");
        assert!(!addressable(0xffff_f000, 0x1001), "one byte past is not");
        assert!(!addressable(0x1_0000_0000, 1), "the pool's own base is not");
        assert!(!addressable(u64::MAX, 1), "and the sum does not wrap");
    }

    /// A descriptor refuses to encode what it cannot describe, rather than
    /// truncating into the field.
    #[test]
    fn a_descriptor_refuses_what_it_cannot_describe() {
        assert_eq!(Descriptor::transmit(0x1000, 0), None, "an empty segment");
        assert_eq!(
            Descriptor::transmit(0x1000, MAX_SEGMENT + 1),
            None,
            "longer than SD_LEN0"
        );
        assert_eq!(
            Descriptor::transmit(0x1_0000_0000, 64),
            None,
            "out of the part's reach"
        );
        assert_eq!(MAX_SEGMENT, 0x3fff);
    }

    /// The transmit encoding is the one upstream writes, bit for bit.
    #[test]
    fn a_transmit_descriptor_matches_the_upstream_word() {
        let d = Descriptor::transmit(0x8000_1000, 4096).expect("in reach");
        assert_eq!(d.buf0, 0x8000_1000);
        assert_eq!(d.buf1, 0, "one segment, so no second address");
        assert_eq!(d.info, 0);
        // SD_LEN0 is GENMASK(29, 16); LAST_SEC0 is BIT(30).
        assert_eq!(d.ctrl, (4096 << 16) | (1 << 30));
        assert_eq!(d.written(), 4096);
        assert!(!d.done(), "the engine has not seen it yet");
    }

    /// A receive descriptor hands over a buffer **without** the done bit,
    /// which is the whole of the completion protocol.
    #[test]
    fn a_receive_descriptor_leaves_the_done_bit_to_the_engine() {
        let mut d = Descriptor::receive(0x9000_0000, 2048).expect("in reach");
        assert_eq!(d.ctrl & MT_DMA_CTL_DMA_DONE, 0);
        assert!(!d.valid(true) && !d.valid(false), "nothing to consume yet");
        // The engine writes a length and raises the bit.
        d.ctrl = (1500 << 16) | MT_DMA_CTL_DMA_DONE;
        assert!(d.valid(true) && d.valid(false), "phase is not consulted");
        assert_eq!(d.written(), 1500);
    }

    /// **The three banks, resolved.** A wrong stride or a wrong base
    /// programs a queue the part is not using, and the part says nothing.
    /// **The engine configuration, against what this part is really
    /// driven with.**
    ///
    /// Every other test in this file checks our transcription against the
    /// vendor's *source*. This one checks it against the vendor's
    /// *behaviour*: the writes below were recorded from `mt76`'s own
    /// `reg_wr` tracepoint on the lab machine (14c3:0717) during a
    /// `modprobe mt7925e`, on 2026-11-10, with `tools/mt7925_trace.py`.
    /// They are BAR0 offsets and values produced by code we did not write,
    /// driving the silicon we are targeting.
    ///
    /// That matters more than it sounds. The standing hazard in this crate
    /// is a test that agrees with the driver because both were written from
    /// the same belief — it has already cost one retired witness bit. A
    /// capture cannot share a belief with us.
    #[test]
    fn the_engine_sequence_matches_what_this_part_is_really_driven_with() {
        // The ten prefetch windows, offset and value, in the order the
        // vendor writes them.
        const OBSERVED_PREFETCH: [(u32, u32); 10] = [
            (0x000d_4680, 0x0000_0004),
            (0x000d_4684, 0x0040_0004),
            (0x000d_4688, 0x0080_0004),
            (0x000d_468c, 0x00c0_0004),
            (0x000d_4600, 0x0100_0010),
            (0x000d_4604, 0x0200_0010),
            (0x000d_4608, 0x0300_0010),
            (0x000d_460c, 0x0400_0010),
            (0x000d_463c, 0x0500_0004),
            (0x000d_4640, 0x0540_0004),
        ];
        assert_eq!(PREFETCH_SETUP, OBSERVED_PREFETCH);

        // `GLO_CFG` across the two writes that bring the engine up. The
        // first is the state the vendor's own disable step left behind.
        const OBSERVED_BEFORE: u32 = 0x0010_3860;
        const OBSERVED_AFTER_SETUP: u32 = 0x5030_b870;
        const OBSERVED_AFTER_ENABLES: u32 = 0x5030_b875;

        // Every bit we set, the part was really set with.
        assert_eq!(OBSERVED_AFTER_SETUP & GLO_CFG_SETUP, GLO_CFG_SETUP);

        // **And every bit we do not set survived from the previous value.**
        // This is the assertion that says `mt76_set` is a read-OR-write and
        // not a store: `GLO_CFG_SETUP` is 0x5020b870, the part was left at
        // 0x5030b870, and the difference — bit 20 — is carried, not
        // written. A flat `write(GLO_CFG, GLO_CFG_SETUP)` would have
        // cleared it, which is exactly what this driver used to do.
        assert_eq!(
            OBSERVED_AFTER_SETUP & !GLO_CFG_SETUP,
            OBSERVED_BEFORE & !GLO_CFG_SETUP,
            "the configuration word is OR'd onto what is there, not stored"
        );
        assert_ne!(
            GLO_CFG_SETUP, OBSERVED_AFTER_SETUP,
            "if these were ever equal this test would prove nothing"
        );

        // Then the two direction bits, and nothing else.
        let enables = MT_WFDMA0_GLO_CFG_TX_DMA_EN | MT_WFDMA0_GLO_CFG_RX_DMA_EN;
        assert_eq!(OBSERVED_AFTER_SETUP | enables, OBSERVED_AFTER_ENABLES);

        // The part-specific tail.
        assert_eq!(INT_RX_PRI_SETUP, 0x0000_0f00);
        assert_eq!(INT_TX_PRI_SETUP, 0x0000_7f00);
        // Both index resets are written across every ring.
        assert_eq!(MT_WFDMA0_RST_DTX_PTR, 0x000d_420c);
        assert_eq!(MT_WFDMA0_RST_DRX_PTR, 0x000d_4280);
        // And the register whose two lawful names cost two debugging
        // sessions: the vendor writes BAR0 offset 0xd42b4, which is what
        // `MT_UWFDMA0_GLO_CFG_EXT1` must route to.
        assert_eq!(
            crate::regs::route(MT_UWFDMA0_GLO_CFG_EXT1),
            crate::regs::Route::Fixed(0x000d_42b4)
        );
    }

    #[test]
    fn the_queue_banks_land_where_upstream_computes_them() {
        // MT_TX_RING_BASE 0xd4300 + 15 * 0x10.
        assert_eq!(Queue::McuWm.regs().desc_base(), 0x000d_43f0);
        // …+ 16 * 0x10.
        assert_eq!(Queue::Fwdl.regs().desc_base(), 0x000d_4400);
        // MT_RX_EVENT_RING_BASE 0xd4500 + 0.
        assert_eq!(Queue::EventWm.regs().desc_base(), 0x000d_4500);
        // The four words, in `struct mt76_queue_regs` order.
        let q = Queue::Fwdl.regs();
        assert_eq!(q.ring_size(), q.desc_base() + 4);
        assert_eq!(q.cpu_idx(), q.desc_base() + 8);
        assert_eq!(q.dma_idx(), q.desc_base() + 0xc);
        // Two queues never share a bank.
        assert_ne!(Queue::McuWm.regs(), Queue::Fwdl.regs());
        // MT_RX_DATA_RING_BASE 0xd4500 + 2 * 0x10 — ring 2, the bank whose
        // prefetch window `PREFETCH_SETUP` has always written.
        assert_eq!(Queue::EventData.regs().desc_base(), 0x000d_4520);
        assert_ne!(Queue::EventData.regs(), Queue::EventWm.regs());
    }

    /// **A freshly filled receive ring is posted one short, on both rings.**
    ///
    /// An index equal to the depth names a slot that does not exist, which
    /// is what ring 0 carried from D1b to M1b.
    #[test]
    fn a_receive_ring_is_posted_one_short_of_its_depth() {
        for queue in [Queue::EventWm, Queue::EventData] {
            let depth = queue.depth();
            assert_eq!(rx_fill_index(depth), depth - 1, "{queue:?}");
            assert!(rx_fill_index(depth) < depth, "{queue:?} posted past its ring");
        }
        assert_eq!(rx_fill_index(Queue::EventWm.depth()), 511);
    }

    /// **Ring 2 is at least one prefetch window deep.**
    ///
    /// The engine fetches four of its descriptors at a time
    /// (`PREFETCH(0x0080, 0x4)`). A ring shallower than its window would
    /// have the engine prefetch slots the count register says do not
    /// exist, and the depth is a choice made here rather than one copied —
    /// so it is the one that needs the guard.
    #[test]
    fn ring_two_is_deeper_than_its_prefetch_window() {
        let window = PREFETCH_SETUP
            .iter()
            .find(|(reg, _)| *reg == MT_WFDMA0_RX_RING2_EXT_CTRL)
            .map(|(_, value)| value & 0xFFFF)
            .expect("ring 2 has a prefetch window");
        assert_eq!(window, 4);
        assert!(u32::from(Queue::EventData.depth()) >= window);
        assert_eq!(u32::from(Queue::EventData.depth()) % window, 0);
    }

    /// Every WFDMA register the ladder touches is a **direct** BAR0 offset,
    /// so a ring doorbell is one transaction and not four.
    #[test]
    fn the_engines_registers_need_no_window() {
        use crate::regs::{Route, route};
        for addr in [
            MT_WFDMA0_GLO_CFG,
            MT_WFDMA0_RST_DTX_PTR,
            MT_WFDMA0_RST_DRX_PTR,
            MT_WFDMA0_PRI_DLY_INT_CFG0,
            MT_WFDMA0_HOST_INT_ENA,
            MT_MCU2HOST_SW_INT_ENA,
            Queue::McuWm.regs().cpu_idx(),
            Queue::Fwdl.regs().cpu_idx(),
            Queue::EventWm.regs().cpu_idx(),
        ] {
            assert_eq!(route(addr), Route::Direct(addr), "{addr:#x}");
        }
        // Every prefetch window is direct too.
        for (reg, _) in PREFETCH_SETUP {
            assert_eq!(route(reg), Route::Direct(reg), "{reg:#x}");
        }
        // **`MT_UWFDMA0_GLO_CFG_EXT1` is `MT_WFDMA0(0x2b4)` under its other
        // lawful name**, and this is the assertion that says so rather than
        // leaving it to be rediscovered. `MT_UWFDMA0(ofs)` is
        // `0x7c024000 + ofs` (`mt792x_regs.h:491`) and the fixed map sends
        // `0x7c02_xxxx` to `0x000d_xxxx`, so the two spellings are one
        // register. The define keeps upstream's spelling so it diffs 1:1;
        // the route is what makes that safe.
        assert_eq!(MT_UWFDMA0_GLO_CFG_EXT1, 0x7c02_42b4);
        assert_eq!(route(MT_UWFDMA0_GLO_CFG_EXT1), Route::Fixed(0x000d_42b4));
        assert_eq!(route(MT_WFDMA0_INT_RX_PRI), Route::Direct(MT_WFDMA0_INT_RX_PRI));
        assert_eq!(route(MT_WFDMA0_INT_TX_PRI), Route::Direct(MT_WFDMA0_INT_TX_PRI));
        // The two that are not: both are in the fixed map, so still one
        // transaction each.
        assert_eq!(route(MT_WFDMA_DUMMY_CR), Route::Fixed(0x2120));
        assert_eq!(route(MT_DMASHDL_SW_CONTROL), Route::Fixed(0xd6004));
        assert_eq!(route(MT_WFSYS_SW_RST), Route::Fixed(0xf0140));
    }

    /// The ring depths are upstream's, and the tables they imply fit the
    /// driver's share of the 2 MiB per-device pool.
    #[test]
    fn the_rings_fit_the_pools_quarter() {
        assert_eq!(Queue::McuWm.depth(), 256);
        assert_eq!(Queue::Fwdl.depth(), 128);
        assert_eq!(Queue::EventWm.depth(), 512);
        let tables = Queue::McuWm.table_bytes()
            + Queue::Fwdl.table_bytes()
            + Queue::EventWm.table_bytes();
        assert_eq!(tables, (256 + 128 + 512) * 16);
        // `kore_rtw89::nic` asserts its whole carve against a quarter of
        // `DRIVER_DMA_POOL_BYTES` (0x20_0000); the descriptor tables alone
        // are a small part of that.
        assert!(tables <= 0x20_0000 / 4);
    }

    /// The global-configuration words are the vendor's, composed once.
    #[test]
    fn the_global_configuration_words_are_transcribed_not_chosen() {
        // DMA_SIZE is GENMASK(5, 4) and upstream fills it with 3 for this
        // part — the MT7928 branch uses 1, which is why the field is
        // composed rather than written as a literal.
        assert_eq!(
            GLO_CFG_SETUP & MT_WFDMA0_GLO_CFG_DMA_SIZE,
            MT_WFDMA0_GLO_CFG_DMA_SIZE
        );
        // Setup never contains the enable bits: `mt792x_dma_enable` sets
        // them in a second write, after the prefetch and the index resets.
        assert_eq!(
            GLO_CFG_SETUP
                & (MT_WFDMA0_GLO_CFG_TX_DMA_EN | MT_WFDMA0_GLO_CFG_RX_DMA_EN),
            0
        );
        // Teardown does, and also the two chain/omit bits the engine must
        // not be left holding.
        assert_ne!(GLO_CFG_TEARDOWN & MT_WFDMA0_GLO_CFG_TX_DMA_EN, 0);
        assert_ne!(GLO_CFG_TEARDOWN & MT_WFDMA0_GLO_CFG_RX_DMA_EN, 0);
        assert_eq!(GLO_CFG_BUSY, (1 << 1) | (1 << 3));
    }

    /// **The mask is the word the vendor writes on this part.**
    ///
    /// Composed from upstream's names, checked against the capture: a
    /// transcription that agreed only with itself is how this crate has
    /// already lost a witness.
    #[test]
    fn the_interrupt_mask_is_the_captured_word() {
        assert_eq!(HOST_INT_MASK, 0x2e07_fff5, "MT_WFDMA0_HOST_INT_ENA in the 2026-11-19 capture");
        // Every ring this driver services is armed.
        for bit in [
            HOST_RX_DONE_INT_ENA0,
            HOST_RX_DONE_INT_ENA2,
            HOST_TX_DONE_INT_ENA15,
            HOST_TX_DONE_INT_ENA16,
            MCU2HOST_SW_INT_ENA,
        ] {
            assert_ne!(HOST_INT_MASK & bit, 0, "{bit:#x} not armed");
        }
        assert_eq!(PCIE_MAC_INT_ALL, 0xff);
    }
}
