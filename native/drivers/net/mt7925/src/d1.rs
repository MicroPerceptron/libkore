//! **D1a — the part accepts a ring.**
//!
//! The rung between [`d0`](crate::d0) (the part answers its own identity)
//! and the firmware download (bytes cross into the chip). Nothing is
//! downloaded here and no MCU command is sent: this programs one queue's
//! four control words and reads two of them back.
//!
//! # Why this is its own rung
//!
//! The download needs three things to be true at once — the ring memory is
//! where the part can reach it, the WFDMA register bank routes correctly,
//! and the engine latches a ring it is given. When a download fails, those
//! three are indistinguishable from each other and from a dozen protocol
//! mistakes above them. One flash per experiment is the budget on a machine
//! with no serial port (`res/NETWORK_RESTART.md`), so each of the three is
//! worth separating from the others before any of them is assumed.
//!
//! **The reach check is the one that pays for itself.** This part is a
//! 32-bit DMA master ([`DMA_ADDR_BITS`](crate::dma::DMA_ADDR_BITS)), and the
//! host's default pool placement is 4 GiB *exactly* — the first address it
//! cannot express. A ring handed to the engine above its ceiling is not
//! refused by the engine: the descriptor truncates and the part fetches from
//! somebody else's page. This rung refuses that here, where the address is
//! still a number, rather than discovering it as corruption later.
//!
//! # What a pass proves, and what it does not
//!
//! A pass proves the engine latched a base and a depth this driver chose,
//! read back through the same window that wrote them. It does **not** prove
//! the engine will fetch a descriptor — nothing has rung the doorbell, and
//! `cpu_idx` is left at zero on purpose, so the ring is programmed and
//! empty. The part is given nothing to do.

use core::sync::atomic::{Ordering, fence};

use kore_driver::seam::{Dma, DmaRead, DmaShared, DmaWrite, Regs};
use kore_driver::wait::{Budget, handshake_until};

use crate::d0::Core;
use crate::dma::{self, Descriptor, GLO_CFG_BUSY, GLO_CFG_SETUP, GLO_CFG_TEARDOWN, Queue};
use crate::mcu::{self, Command, McuError, Response};
use crate::regs::mt7925::{
    MT_DMASHDL_DMASHDL_BYPASS, MT_DMASHDL_SW_CONTROL, MT_UWFDMA0_GLO_CFG_EXT1,
    MT_UWFDMA0_GLO_CFG_EXT1_BIT28, MT_WFDMA0_GLO_CFG, MT_WFDMA0_GLO_CFG_EXT0,
    MT_WFDMA0_GLO_CFG_EXT0_CSR_TX_DMASHDL_EN, MT_WFDMA0_GLO_CFG_RX_DMA_EN,
    MT_WFDMA0_GLO_CFG_TX_DMA_EN, MT_WFDMA0_INT_RX_PRI, MT_WFDMA0_INT_TX_PRI,
    MT_WFDMA0_PRI_DLY_INT_CFG0, MT_WFDMA0_RST, MT_WFDMA0_RST_DMASHDL_ALL_RST,
    MT_WFDMA0_RST_DRX_PTR, MT_WFDMA0_RST_DTX_PTR, MT_WFDMA0_RST_LOGIC_RST,
    MT_HW_EMI_CTL, MT_HW_EMI_CTL_SLPPROT_EN, MT_MCU2HOST_SW_INT_ENA,
    MT_MCU_CMD_WAKE_RX_PCIE, MT_WFDMA_DUMMY_CR, MT_WFDMA_NEED_REINIT,
    MT_WFSYS_SW_RST, WFSYS_SW_INIT_DONE,
    WFSYS_SW_RST_B, MCU2HOST_SW_INT_ENA, MT7925_PCIE_MAC_INT_ENABLE, MT_MCU_CMD,
    MT_WFDMA0_HOST_INT_ENA, MT_WFDMA0_HOST_INT_STA,
};
use crate::regs::{Route, route};

/// What D1a could not establish.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum D1Refusal {
    /// The ring does not lie inside this part's DMA reach.
    ///
    /// Carries the span rather than a bit, because the number is the
    /// diagnosis: an address at or above 4 GiB names the host's default
    /// placement and says the binding's
    /// `PciBindingContract::DMA_ADDRESS_BITS` did not take effect.
    Unreachable {
        /// Where the ring was placed.
        iova: u64,
        /// How long it is.
        len: u64,
    },
    /// The descriptor base did not read back as written.
    ///
    /// Both values are carried: all-ones is an unmapped window, all-zeroes a
    /// bank that never decoded, and anything else a routing error. Those are
    /// three different repairs.
    BaseNotLatched {
        /// What was written.
        wrote: u32,
        /// What read back.
        read: u32,
    },
    /// The ring depth did not read back as written.
    SizeNotLatched {
        /// What was written.
        wrote: u32,
        /// What read back.
        read: u32,
    },
    /// The engine never went quiet inside its budget.
    ///
    /// Carries the `MT_WFDMA0_GLO_CFG` word, because the two busy bits are
    /// separate questions: transmit busy with receive idle is a different
    /// machine state from both busy, and from a window reading all-ones.
    EngineBusy {
        /// The last `GLO_CFG` read.
        glo_cfg: u32,
    },
    /// The command does not fit the staging buffer.
    CommandTooLong {
        /// Bytes the command needs.
        bytes: usize,
        /// Bytes the staging buffer has.
        room: usize,
    },
    /// No answer arrived inside the budget.
    ///
    /// Carries the engine's receive index, because this is the first rung
    /// where that number is evidence: zero says nothing was written at all,
    /// and non-zero says something arrived that the descriptor's done bit
    /// did not agree about.
    NoAnswer {
        /// The transmit queue's own index when the budget ran out.
        ///
        /// **The question this rung could not answer before.** Zero means
        /// the engine never consumed the command descriptor, so the
        /// command never left the host; non-zero means it did, and the
        /// silence is the MCU's. Two different repairs, and nothing else
        /// in the refusal told them apart.
        ///
        /// Carried, not interpreted: that `dma_idx` advances when a
        /// transmit descriptor is consumed is a belief about this part,
        /// and a belief asserted in a test is how the D1b witness was
        /// earned against a fiction.
        tx_idx: u32,
        /// The sequence that went out.
        seq: u8,
        /// The event ring's engine index.
        dma_idx: u32,
        /// The host's own command cursor — what the doorbell advertised.
        ///
        /// `tx_idx` alone cannot say whether the engine *caught up*: it
        /// is only evidence beside the index it was asked to reach. Equal
        /// (modulo the ring) says the command left the host; behind says
        /// the engine stopped fetching.
        tx_next: u32,
        /// The host's own event cursor — the slot the poll was watching.
        ///
        /// Beside `dma_idx` it says whether the engine wrote past the slot
        /// the poll read: an engine ahead of this cursor with nothing
        /// seen done is a descriptor the host cannot observe, not silence.
        rx_next: u32,
        /// Replies this exchange consumed as somebody else's.
        ///
        /// **Silence and a triaged-away answer read identically without
        /// it.** M1 is the first rung whose replies pass through the
        /// unsolicited/sequence triage with traffic that could fail it;
        /// a non-zero count says the part *did* speak. Saturating.
        ///
        /// Counts both receive rings.
        skipped: u8,
        /// Slots receive ring 2 holds past the host's cursor, or `None`
        /// when the exchange was not given that ring.
        ///
        /// `dma_idx - data_next` on ring 2's own depth. Non-zero with
        /// nothing seen done is the same "written but not observed" fact
        /// `rx_next` gives for ring 0; zero, beside a zero on ring 0, is
        /// a part that wrote to neither.
        data_pending: Option<u32>,
    },
    /// An answer arrived and was not this request's.
    BadAnswer(McuError),

    /// The part did not hand ownership back to the driver.
    ///
    /// Carries `MT_CONN_ON_LPCTL`, because "still firmware-owned",
    /// "never answered the strobe" and an all-ones read are three
    /// different repairs.
    NotOwned { lpctl: u32 },
    /// The Wi-Fi subsystem did not come back from its reset.
    ///
    /// Carries the register, because the three states it can be in are
    /// three different repairs: still asserted, released but never
    /// signalling init-done, or an all-ones read meaning the window died
    /// under us.
    WfsysNotReady { sw_rst: u32 },
    /// A ring or staging buffer the caller passed is smaller than the
    /// access it was asked for.
    ///
    /// Unreachable from the binding, which sizes every buffer from
    /// [`Queue::table_bytes`] — but the seam bounds-checks each access and
    /// its refusal has to land somewhere honest rather than be unwrapped.
    RingTooSmall { need: usize, have: usize },
    /// The engine did not take the receive buffers it was given.
    ///
    /// `dma_idx` is the **engine's** index. It is the first value in this
    /// ladder that hardware produces rather than echoes, so a stuck one is
    /// the first evidence that the part is not merely accepting writes.
    RxNotTaken {
        /// How many buffers were posted.
        posted: u32,
        /// Where the engine's index actually stood.
        dma_idx: u32,
    },
}

impl D1Refusal {
    /// The founding-ladder rung this refusal reports as.
    ///
    /// Numbered from 1 within D1a; the caller offsets them into its own
    /// ladder, the way [`d0`](crate::d0)'s are.
    pub const fn rung(self) -> u8 {
        match self {
            Self::Unreachable { .. } => 1,
            Self::BaseNotLatched { .. } => 2,
            Self::SizeNotLatched { .. } => 3,
            Self::EngineBusy { .. } => 4,
            Self::RxNotTaken { .. } => 5,
            Self::CommandTooLong { .. } => 6,
            Self::NoAnswer { .. } => 7,
            Self::BadAnswer(_) => 8,
            Self::RingTooSmall { .. } => 9,
            Self::WfsysNotReady { .. } => 10,
            Self::NotOwned { .. } => 11,
        }
    }
}

impl<R: Regs> Core<'_, R> {
    /// Program one queue's descriptor ring and prove the engine took it.
    ///
    /// `iova`/`len` describe the descriptor table — device-visible base and
    /// length, from the pool allocation that backs it. The caller owns that
    /// memory; this writes only registers.
    ///
    /// The order is `mt76_dma_queue_reset`'s (`dma.c:216-230`): base, then
    /// depth, then the producer index cleared. `dma_idx` is the engine's own
    /// and is never written.
    pub fn program_queue(&self, queue: Queue, iova: u64, len: u64) -> Result<(), D1Refusal> {
        // Before anything is written: a ring the part cannot address is a
        // silent truncation, not a refusal, so it is refused here.
        if !dma::addressable(iova, len) {
            return Err(D1Refusal::Unreachable { iova, len });
        }
        let bank = queue.regs();
        // `addressable` just proved the span fits in 32 bits, so the
        // narrowing is the check's own conclusion rather than a cast that
        // hopes.
        let base = iova as u32;
        let depth = u32::from(queue.depth());

        // **`ring_size` before `desc_base`**, which is `mt76_dma_sync_idx`'s
        // order (`dma.c:206-214`) and not the obvious one. Transcribed
        // rather than reasoned about: a part that latches a base against a
        // depth it has not been told yet is a part walking a ring whose
        // length it is guessing.
        self.write(bank.ring_size(), depth);
        self.write(bank.desc_base(), base);
        // The producer index is cleared, not advanced: the ring is handed
        // over programmed and empty. The part is given nothing to fetch.
        self.write(bank.cpu_idx(), 0);

        // Read back through the same window that wrote, so a pass is a
        // round trip rather than a hope about a posted write.
        let read_size = self.read(bank.ring_size());
        if read_size != depth {
            return Err(D1Refusal::SizeNotLatched {
                wrote: depth,
                read: read_size,
            });
        }
        let read_base = self.read(bank.desc_base());
        if read_base != base {
            return Err(D1Refusal::BaseNotLatched {
                wrote: base,
                read: read_base,
            });
        }
        Ok(())
    }
}

/// How long the MCU gets to answer one command.
///
/// **Twenty seconds, which is deliberately more patient than upstream.**
///
/// An earlier reading of this had it as "this part's own number", on the
/// strength of `mt7925_mcu_fill_message` setting
/// `mdev->mcu.timeout = 20 * HZ` on every message it builds
/// (`mt7925/mcu.c:3777`). That reading was wrong, and the correction is
/// worth keeping because the code it describes is four lines long:
/// `mt7925_mcu_send_message` is the **only** `mcu_skb_send_msg` this
/// part's PCIe path installs (`mt7925/pci_mcu.c:29-36`), and it calls
/// `fill_message` and then immediately overwrites the field —
///
/// ```text
/// ret = mt7925_mcu_fill_message(mdev, skb, cmd, seq);  /* 20 * HZ */
/// if (ret) return ret;
/// mdev->mcu.timeout = 3 * HZ;                          /* pci_mcu.c:19 */
/// ```
///
/// — unconditionally. **Upstream's effective budget for this part is
/// three seconds**, and the twenty is dead on arrival.
///
/// The number here stays at twenty anyway, and that is a choice rather
/// than an oversight. A budget is the longest this driver will wait
/// before calling a silent part a refusal; being more patient than
/// upstream costs a slower refusal on a boot that was going to fail, and
/// buys nothing being wrong on a boot that was going to succeed. Every
/// MCU rung on metal so far was measured with this value, so cutting it
/// to three would change timing on a proven path to match a number we do
/// not need. If a rung ever refuses here with the part plainly alive,
/// three seconds is the figure to test against — not a guess.
pub const MCU_BUDGET_NS: u64 = 20_000_000_000;
/// Probes when the clock cannot answer. One per microsecond of the budget
/// would be three million; this is the unclocked fallback the other rungs
/// use, scaled to the same shape.
pub const MCU_SPINS_UNCLOCKED: u32 = 300_000;
/// [`MCU_BUDGET_NS`] and [`MCU_SPINS_UNCLOCKED`] as the one pair every
/// answered command waits under.
const MCU_WAIT: (u64, u32) = (MCU_BUDGET_NS, MCU_SPINS_UNCLOCKED);

/// How long [`Core::restart`] listens for a reply it does not need.
///
/// Upstream does not wait for one at all (`mt76_connac_mcu_restart` sends
/// with `wait_resp = false`), and then polls `MT_CONN_ON_MISC` for up to
/// one second (`mt792x_load_firmware`, `mt792x_core.c:1103`). One second
/// here as well: long enough to consume a reply the ROM does send, so it
/// never lands in the next exchange's triage, and short enough that a
/// ROM which stays quiet costs what upstream's own poll costs.
pub const RESTART_BUDGET_NS: u64 = 1_000_000_000;
/// The unclocked fallback for [`RESTART_BUDGET_NS`], on
/// [`MCU_SPINS_UNCLOCKED`]'s scale.
pub const RESTART_SPINS_UNCLOCKED: u32 = 15_000;
/// How long [`Core::restart`] polls for the power-on state after it.
pub const POWER_ON_BUDGET_NS: u64 = 1_000_000_000;
/// The unclocked fallback for [`POWER_ON_BUDGET_NS`].
pub const POWER_ON_SPINS_UNCLOCKED: u32 = 15_000;

/// How long the engine gets to go quiet.
///
/// `mt792x_dma_disable` polls `GLO_CFG` for both busy bits to clear on a
/// 1,000-iteration, 1 us budget (`mt792x_dma.c:459`), so **1 ms** is the
/// vendor's own number rather than one chosen here. The unclocked fallback
/// is the same 1,000 probes.
/// How long `mt792x_wfsys_reset` holds the subsystem in reset: 50 ms flat
/// (`msleep(50)`), which the capture from this part confirms is really
/// needed — init-done had not dropped before ~52 ms.
pub const WFSYS_HOLD_NS: u64 = 50_000_000;
/// The hold, for a machine with no clock. One `relax` is 4 us, so this is
/// the same 50 ms expressed in pauses.
pub const WFSYS_HOLD_SPINS: u32 = 12_500;
/// The vendor's own budget for `WFSYS_SW_INIT_DONE` after the reset is
/// released: `__mt76_poll_msec(..., 500)`.
pub const WFSYS_INIT_BUDGET_NS: u64 = 500_000_000;
/// The same wait, unclocked.
pub const WFSYS_INIT_SPINS: u32 = 125_000;

pub const QUIET_BUDGET_NS: u64 = 1_000_000;
/// Probes to take when the machine's clock cannot answer. See
/// [`QUIET_BUDGET_NS`].
pub const QUIET_SPINS_UNCLOCKED: u32 = 1_000;

impl<'r, R: Regs + ?Sized> Core<'r, R> {
    /// Wait for both `GLO_CFG` busy bits to read clear.
    ///
    /// Budgeted, never spun: `kore_driver::wait` is the in-tree mechanism
    /// and an unbounded device poll is the anti-pattern it exists to
    /// remove. `GLO_CFG` is a `Direct` route, so each probe is one MMIO
    /// read and the budget is not quietly multiplied by a window write.
    pub fn await_quiet(
        &self,
        delay: impl FnMut(),
        now_ns: fn() -> Option<u64>,
    ) -> Result<(), D1Refusal> {
        let (Route::Direct(off) | Route::Fixed(off)) = route(MT_WFDMA0_GLO_CFG) else {
            return Err(D1Refusal::EngineBusy { glo_cfg: 0 });
        };
        let budget = Budget::new(QUIET_BUDGET_NS, QUIET_SPINS_UNCLOCKED, now_ns);
        let settled = handshake_until(self.regs(), budget, delay, |r| {
            r.read32(off as usize) & GLO_CFG_BUSY == 0
        });
        if settled.is_acked() {
            return Ok(());
        }
        // The register, not the `Ack` variant, is what is carried: a timeout
        // and a fault are the same repair from here, while `glo_cfg` tells
        // transmit-busy from receive-busy from an all-ones window, which are
        // three different ones.
        Err(D1Refusal::EngineBusy {
            glo_cfg: self.read(MT_WFDMA0_GLO_CFG),
        })
    }

    /// Reset the Wi-Fi subsystem and wait for it to say it is ready —
    /// `mt792x_wfsys_reset` (`mt792x_core.c`).
    ///
    /// **This runs before anything else, because upstream runs it before
    /// anything else**: `mt7925_pci_probe` calls it at `pci.c:742`, ahead
    /// of `mt7925_dma_init` at `:757` and of `mt7925_register_device` —
    /// and therefore ahead of the first MCU command, which is the one this
    /// ladder could not get an answer to.
    ///
    /// It puts the MCU into a known state. Without it the part keeps
    /// whatever state it was left in — by this machine's own Linux, by a
    /// warm reboot that did not clear the function, or by firmware already
    /// running — and a ROM-level question asked of a part that is past the
    /// ROM has no reason to be answered.
    ///
    /// The 50 ms is a **flat hold, not a poll**: there is nothing to watch
    /// while the reset is asserted. Upstream sleeps it, and the capture
    /// from this silicon agrees — init-done had not dropped until roughly
    /// 52 ms in. Then the bit goes back and `WFSYS_SW_INIT_DONE` is
    /// polled, on the vendor's own 500 ms budget.
    pub fn reset_wfsys(
        &self,
        mut delay: impl FnMut(),
        now_ns: fn() -> Option<u64>,
    ) -> Result<(), D1Refusal> {
        // **Sleep protection first.** `mt7925_pci_probe` sets this on the
        // line before it calls the reset (`mt7925/pci.c:740-742`), so it is
        // not visible inside `mt792x_wfsys_reset` itself — reading that
        // function alone, as this driver did, does not reveal it.
        let emi = self.read(MT_HW_EMI_CTL);
        self.write(MT_HW_EMI_CTL, emi | MT_HW_EMI_CTL_SLPPROT_EN);
        // Drop the reset.
        let rst = self.read(MT_WFSYS_SW_RST);
        self.write(MT_WFSYS_SW_RST, rst & !WFSYS_SW_RST_B);
        // Hold it down. Deliberately running a budget to expiry, which is
        // the only wait in this driver with nothing to observe.
        let mut hold = Budget::new(WFSYS_HOLD_NS, WFSYS_HOLD_SPINS, now_ns);
        while hold.may_wait() {
            delay();
        }
        // And raise it.
        let rst = self.read(MT_WFSYS_SW_RST);
        self.write(MT_WFSYS_SW_RST, rst | WFSYS_SW_RST_B);

        let budget = Budget::new(WFSYS_INIT_BUDGET_NS, WFSYS_INIT_SPINS, now_ns);
        let (Route::Direct(off) | Route::Fixed(off)) = route(MT_WFSYS_SW_RST) else {
            return Err(D1Refusal::WfsysNotReady { sw_rst: 0 });
        };
        let settled = handshake_until(self.regs(), budget, delay, |r| {
            let v = r.read32(off as usize);
            v != u32::MAX && v & WFSYS_SW_INIT_DONE != 0
        });
        if settled.is_acked() {
            return Ok(());
        }
        Err(D1Refusal::WfsysNotReady {
            sw_rst: self.read(MT_WFSYS_SW_RST),
        })
    }

    /// Take the engine down and reset its logic — `mt792x_dma_disable` with
    /// `force` (`mt792x_dma.c:447-485`).
    ///
    /// **This runs before any ring is programmed**, which is where
    /// `mt792x_dma_init` runs it (`mt792x_dma.c:95`): the logic reset at the
    /// end throws away the engine's ring state, so a descriptor base written
    /// before it would not survive.
    ///
    /// The DMA scheduler is the part of this that transmit cannot do
    /// without. `dmashdl` arbitrates the *transmit* path only; left enabled
    /// and unconfigured it gates command descriptors while receive traffic
    /// flows past it untouched. Upstream turns it off and then bypasses it
    /// outright, and so does this.
    pub fn reset_engine(
        &self,
        mut delay: impl FnMut(),
        now_ns: fn() -> Option<u64>,
    ) -> Result<(), D1Refusal> {
        // Clear the enables and the format bits, then see it go quiet.
        self.disable_engine(&mut delay, now_ns)?;
        // Disable the DMA scheduler, then bypass it.
        let ext0 = self.read(MT_WFDMA0_GLO_CFG_EXT0);
        self.write(
            MT_WFDMA0_GLO_CFG_EXT0,
            ext0 & !MT_WFDMA0_GLO_CFG_EXT0_CSR_TX_DMASHDL_EN,
        );
        let shdl = self.read(MT_DMASHDL_SW_CONTROL);
        self.write(MT_DMASHDL_SW_CONTROL, shdl | MT_DMASHDL_DMASHDL_BYPASS);
        // The logic reset: both bits low, then both high. Two separate
        // read-modify-writes, as `mt76_clear` then `mt76_set`.
        let bits = MT_WFDMA0_RST_DMASHDL_ALL_RST | MT_WFDMA0_RST_LOGIC_RST;
        let rst = self.read(MT_WFDMA0_RST);
        self.write(MT_WFDMA0_RST, rst & !bits);
        let rst = self.read(MT_WFDMA0_RST);
        self.write(MT_WFDMA0_RST, rst | bits);
        Ok(())
    }

    /// Bring the engine up — `mt792x_dma_enable`'s non-MT7928 branch
    /// (`mt792x_dma.c:280-359`) followed by `mt7925_wfdma_setup`
    /// (`:174-179`).
    ///
    /// **Every ring must already be programmed and the receive ring filled**
    /// before this is called. An engine enabled over an empty receive ring
    /// consumes descriptors whose buffers are not yet mapped.
    ///
    /// **The interrupt enable at the tail is here now** (`mt76_connac_irq_enable`,
    /// `:352-357`, [`dma::HOST_INT_MASK`]). It was deliberately left out
    /// from D1 to M1b on the reasoning that a ladder which reads the
    /// descriptor's own done bit needs no interrupt — true of the engine,
    /// and untested of the firmware. Every waiter now parks on the route and
    /// services the status the way `mt792x_irq_tasklet` does
    /// ([`Self::service_interrupt`]).
    pub fn enable_engine(&self) {
        // **The prefetch windows first**, before anything can fetch
        // through them. See [`dma::PREFETCH_SETUP`] for why an engine
        // that receives correctly can still fail its first transmit.
        for (reg, value) in dma::PREFETCH_SETUP {
            self.write(reg, value);
        }
        // Reset the engine's own ring indices. Without this it keeps
        // whatever `dma_idx` the ROM left in each bank, and the first
        // doorbell is read against a stale index — a producer index of one
        // against a consumer index that is already past it.
        self.write(MT_WFDMA0_RST_DTX_PTR, u32::MAX);
        // Receive too, which is a MT7925/MT7928-only line upstream
        // (`mt792x_dma.c:289-290`).
        self.write(MT_WFDMA0_RST_DRX_PTR, u32::MAX);
        // No interrupt coalescing delay.
        self.write(MT_WFDMA0_PRI_DLY_INT_CFG0, 0);
        // The configuration word. `mt76_set` is a read-OR-write and not a
        // plain store (`:328`), so whatever the ROM configured survives;
        // writing `GLO_CFG_SETUP` flat would clear it.
        let cfg = self.read(MT_WFDMA0_GLO_CFG);
        self.write(MT_WFDMA0_GLO_CFG, cfg | GLO_CFG_SETUP);
        // The two direction bits, a second write, so the engine never runs
        // for an instant under a half-written configuration (`:350-351`).
        let cfg = self.read(MT_WFDMA0_GLO_CFG);
        self.write(
            MT_WFDMA0_GLO_CFG,
            cfg | MT_WFDMA0_GLO_CFG_TX_DMA_EN | MT_WFDMA0_GLO_CFG_RX_DMA_EN,
        );
        // `mt7925_wfdma_setup`. `MT_UWFDMA0_GLO_CFG_EXT1` is the same
        // register as `MT_WFDMA0(0x2b4)` under its other lawful name; the
        // route test pins that identity rather than leaving it to be
        // rediscovered.
        let ext1 = self.read(MT_UWFDMA0_GLO_CFG_EXT1);
        self.write(
            MT_UWFDMA0_GLO_CFG_EXT1,
            ext1 | MT_UWFDMA0_GLO_CFG_EXT1_BIT28,
        );
        let rx_pri = self.read(MT_WFDMA0_INT_RX_PRI);
        self.write(MT_WFDMA0_INT_RX_PRI, rx_pri | dma::INT_RX_PRI_SETUP);
        let tx_pri = self.read(MT_WFDMA0_INT_TX_PRI);
        self.write(MT_WFDMA0_INT_TX_PRI, tx_pri | dma::INT_TX_PRI_SETUP);
        // Tell the MCU the host re-initialised the engine under it.
        let dummy = self.read(MT_WFDMA_DUMMY_CR);
        self.write(MT_WFDMA_DUMMY_CR, dummy | MT_WFDMA_NEED_REINIT);
        // Arm the host interrupt sources — `mt76_connac_irq_enable`, the
        // vendor's word for this part, in the vendor's position: after the
        // reinit flag, before the MCU's own route (`:352-358`).
        self.write(MT_WFDMA0_HOST_INT_ENA, dma::HOST_INT_MASK);
        // **And let the MCU wake the host's receive path over PCIe.**
        // `mt792x_dma_enable`'s last line (`mt792x_dma.c:358`), and the
        // one step of it this driver skipped on the reasoning that a
        // polled ladder needs no interrupts. That reasoning does not apply
        // to this register: `MT_WFDMA0_HOST_INT_ENA` is what gates an MSI
        // and is left at zero, so nothing is armed here. This enables the
        // MCU's own route for telling the host it has written something —
        // the bit is called `WAKE_RX_PCIE` — and the capture shows the
        // vendor setting it on every engine enable.
        let sw_int = self.read(MT_MCU2HOST_SW_INT_ENA);
        self.write(MT_MCU2HOST_SW_INT_ENA, sw_int | MT_MCU_CMD_WAKE_RX_PCIE);
    }

    /// Stop the engine and wait for it to go quiet.
    ///
    /// `mt792x_dma_disable` (`mt792x_dma.c:449-461`): clear the bits, then
    /// poll both busy flags. **This must complete before the DMA pool is
    /// released** — the engine holds the ring's address, and freeing pages
    /// it is still walking is a write into whatever is mapped there next.
    pub fn disable_engine(
        &self,
        delay: impl FnMut(),
        now_ns: fn() -> Option<u64>,
    ) -> Result<(), D1Refusal> {
        let cfg = self.read(MT_WFDMA0_GLO_CFG);
        self.write(MT_WFDMA0_GLO_CFG, cfg & !GLO_CFG_TEARDOWN);
        self.await_quiet(delay, now_ns)
    }

    /// Hand the engine `posted` receive buffers by advancing `cpu_idx`.
    ///
    /// The descriptors must already be written into the table; this is the
    /// doorbell, and it is the only write that tells the engine there is
    /// work. `mt76_dma_kick_queue` (`dma.c:234`).
    ///
    /// `posted` is a **ring index**, so it is always below the depth — see
    /// [`dma::rx_fill_index`] for what an index equal to it did.
    pub fn post_receive(&self, queue: Queue, posted: u16) {
        debug_assert!(
            posted < queue.depth(),
            "{queue:?}: producer index {posted} names no slot of a {}-deep ring",
            queue.depth()
        );
        self.write(queue.regs().cpu_idx(), u32::from(posted));
    }

    /// Whether the answer in `slot` belongs to the exchange waiting on
    /// `seq`.
    ///
    /// Three outcomes collapse to two here, deliberately:
    ///
    /// - **Unsolicited** — the part spoke unprompted. Not ours, and never
    ///   will be. Discard.
    /// - **A different sequence** — somebody's reply, not this one's.
    ///   Discard, exactly as upstream's `-EAGAIN` loop does.
    /// - **Too short to tell** — kept, so that
    ///   [`mcu::Response::parse`] refuses it with
    ///   [`mcu::McuError::TooShort`]. A header this driver cannot read is
    ///   not something it may throw away silently; a malformed answer must
    ///   name itself rather than time out as `NoAnswer` and send a reader
    ///   looking at the transport.
    ///
    /// Reads only the header, and only through the checked accessor: a
    /// descriptor whose length the ring cannot honour is also "cannot
    /// tell", and takes the same conservative arm.
    fn answer_is_ours<D: Dma + ?Sized>(
        &self,
        buffers: &D,
        buf_bytes: usize,
        slot: usize,
        desc: &Descriptor,
        seq: u8,
    ) -> bool {
        let written = (desc.written() as usize).min(buf_bytes);
        if written < mcu::MCU_RXD_BYTES {
            return true;
        }
        let at = slot * buf_bytes;
        let Ok(ptr) = buffers.checked_read_ptr(at, mcu::MCU_RXD_BYTES) else {
            return true;
        };
        // SAFETY: `checked_read_ptr` proved `MCU_RXD_BYTES` readable bytes
        // at `at`, and the caller's acquire fence orders them after the
        // done bit that published them. The slice does not outlive this
        // call.
        let header = unsafe { core::slice::from_raw_parts(ptr, mcu::MCU_RXD_BYTES) };
        if mcu::is_unsolicited(header) {
            return false;
        }
        match mcu::peek_sequence(header) {
            Some(got) => got == seq,
            None => true,
        }
    }

    /// Read the engine's own index for a queue.
    ///
    /// Never written by this driver — see [`Self::program_queue`]. This is
    /// what makes [`D1Refusal::RxNotTaken`] evidence rather than an echo.
    pub fn engine_index(&self, queue: Queue) -> u32 {
        self.read(queue.regs().dma_idx())
    }

    /// Look at the slot `cursor` names on one receive ring, and say what
    /// is there: nothing yet, this exchange's answer, or somebody else's.
    ///
    /// One reading of one slot, shared by both rings so the triage cannot
    /// come to mean two things. Consuming is the caller's, because what
    /// "consumed" costs differs per ring — see [`DataRing`].
    fn poll_ring<D: Dma + ?Sized>(
        &self,
        table: &D,
        buffers: &D,
        buf_bytes: usize,
        cursor: u32,
        queue: Queue,
        seq: u8,
    ) -> Result<Polled, D1Refusal> {
        let slot = (cursor % u32::from(queue.depth())) as usize;
        // **Volatile, so the compiler cannot hoist it out of the caller's
        // loop** — which it otherwise would, this being memory nothing
        // inside the loop writes. A plain read here does not hang: the
        // budget still runs down and the exchange still returns. It
        // returns `NoAnswer` to an answer that did arrive, which is worse,
        // because the refusal names the part.
        let ctrl = DmaShared::new(table)
            .read_u32_at(Descriptor::ctrl_at(slot))
            .map_err(|_| D1Refusal::RingTooSmall {
                need: dma::DESC_BYTES,
                have: table.len(),
            })?;
        let desc = Descriptor::from_ctrl(ctrl);
        if !desc.done() {
            return Ok(Polled::Empty);
        }
        // Ordered before the header is read: the done bit is what
        // published these bytes.
        fence(Ordering::Acquire);
        Ok(if self.answer_is_ours(buffers, buf_bytes, slot, &desc, seq) {
            Polled::Ours(slot, desc)
        } else {
            Polled::Theirs(slot, desc)
        })
    }

    /// Give ring 2's slot back to the engine once the host has read it.
    ///
    /// **Ring 0 does not do this and does not need to**: it is 512 deep
    /// and the whole ladder has used thirteen of it. Ring 2 is 32 deep
    /// ([`dma::EVENT_DATA_DEPTH`]) and exists *because* the part may talk
    /// on it unprompted, so a ring that is only ever filled is a ring that
    /// stalls after thirty-one events — which could itself silence the
    /// engine's receive side.
    ///
    /// The slot is re-armed whole (address, full length, done clear),
    /// published, and then the producer index moves to the slot *before*
    /// the host's cursor: upstream keeps one slot unposted so a full ring
    /// and an empty one never share an index (`mt76_dma_rx_fill` fills to
    /// `ndesc - 1`).
    fn recycle_data_slot<D: Dma + ?Sized>(
        &self,
        data: &DataRing<'_, D>,
        buf_bytes: usize,
        slot: usize,
        next: u32,
    ) {
        let iova = data.buffers.iova().0 + (slot * buf_bytes) as u64;
        if let Some(desc) = Descriptor::receive(iova, buf_bytes as u32) {
            let table = DmaShared::new(data.table);
            for (word, value) in desc.words().into_iter().enumerate() {
                let _ = table.write_u32_at(Descriptor::slot_at(slot) + word * 4, value);
            }
        }
        fence(Ordering::Release);
        let depth = u32::from(Queue::EventData.depth());
        self.post_receive(Queue::EventData, ((next + depth - 1) % depth) as u16);
    }
}

/// What [`Core::restart`] saw. Neither half is a refusal; both are
/// evidence about a step upstream does not check either.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Restarted {
    /// The ROM replied to `NIC_POWER_CTRL` inside [`RESTART_BUDGET_NS`].
    pub answered: bool,
    /// `MT_CONN_ON_MISC`'s firmware state read power-on inside
    /// [`POWER_ON_BUDGET_NS`] — the condition upstream polls for.
    pub powered: bool,
}

/// What one pass of [`Core::service_interrupt`] found and acknowledged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IrqSeen {
    /// `MT_WFDMA0_HOST_INT_STA`, masked to the armed sources — exactly the
    /// bits acknowledged.
    pub sta: u32,
    /// `MT_MCU_CMD` as read and acknowledged, or zero when the status did
    /// not carry the MCU's software interrupt and it was not touched.
    pub mcu_cmd: u32,
}

impl<'r, R: Regs + ?Sized> Core<'r, R> {
    /// Let the PCIe MAC raise interrupts at all — `mt7925_pci_probe`'s
    /// `mt76_wr(dev, pcie_reg->imask, 0xff)` (`mt7925/pci.c:747`).
    ///
    /// Upstream writes it once, after the Wi-Fi subsystem reset and before
    /// the engine is touched, and so does the capture of this part.
    pub fn arm_pcie_interrupts(&self) {
        self.write(MT7925_PCIE_MAC_INT_ENABLE, dma::PCIE_MAC_INT_ALL);
    }

    /// Take every interrupt source back — the mask, then the PCIe MAC.
    ///
    /// `quiesce`'s first step: an engine about to be stopped must not raise
    /// a delivery into a route its owner is releasing.
    pub fn disarm_interrupts(&self) {
        self.write(MT_WFDMA0_HOST_INT_ENA, 0);
        self.write(MT7925_PCIE_MAC_INT_ENABLE, 0);
    }

    /// **`mt792x_irq_tasklet`'s register half** (`mt792x_dma.c:28-58`):
    /// mask, read the status, acknowledge exactly what was read, acknowledge
    /// the MCU's software interrupt if it is among it, unmask.
    ///
    /// Called on every wake of a waiter, fired or not. The status is
    /// write-one-to-clear and an MSI is an edge: a status bit left set is a
    /// source that cannot raise the next one, so a waiter that parked
    /// without acknowledging would hear the first interrupt and none after.
    ///
    /// `MT_MCU_CMD` is the MCU telling the host something — upstream reads
    /// it and writes the same value back to acknowledge it whenever
    /// `MCU2HOST_SW_INT` is pending. This driver never did, from D1 to M1b.
    pub fn service_interrupt(&self) -> IrqSeen {
        self.write(MT_WFDMA0_HOST_INT_ENA, 0);
        let sta = self.read(MT_WFDMA0_HOST_INT_STA) & dma::HOST_INT_MASK;
        // Written even when zero, as upstream does — the capture shows
        // `HOST_INT_STA = 0` written back on every pass.
        self.write(MT_WFDMA0_HOST_INT_STA, sta);
        let mcu_cmd = if sta & MCU2HOST_SW_INT_ENA != 0 {
            let cmd = self.read(MT_MCU_CMD);
            self.write(MT_MCU_CMD, cmd);
            cmd
        } else {
            0
        };
        self.write(MT_WFDMA0_HOST_INT_ENA, dma::HOST_INT_MASK);
        IrqSeen { sta, mcu_cmd }
    }
}

/// What one receive slot held when it was looked at.
enum Polled {
    /// Not done yet.
    Empty,
    /// This exchange's answer.
    Ours(usize, Descriptor),
    /// Done, and somebody else's — unsolicited, or another sequence.
    Theirs(usize, Descriptor),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dma::DMA_ADDR_BITS;
    use crate::mock::{MockMt7925, ReachablePool};

    /// A ring inside the part's reach is programmed and reads back.
    #[test]
    fn a_reachable_ring_is_latched() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        let queue = Queue::Fwdl;
        let iova = 0x4000_0000;
        assert_eq!(
            core.program_queue(queue, iova, queue.table_bytes() as u64),
            Ok(())
        );
        let bank = queue.regs();
        // Read back the way the driver does, through `route`. A BAR offset
        // below `0x200000` is reachable under two chip names — directly, and
        // through the fixed block whose window covers it — so `peek` on the
        // other name reads a different cell and proves nothing. The round
        // trip is the assertion worth making anyway.
        assert_eq!(core.read(bank.desc_base()), iova as u32);
        assert_eq!(core.read(bank.ring_size()), u32::from(queue.depth()));
        // Handed over empty: the part is given nothing to fetch.
        assert_eq!(core.read(bank.cpu_idx()), 0);
        // The engine's own index is never written by the driver.
        let dma_idx_off = match crate::regs::route(bank.dma_idx()) {
            crate::regs::Route::Direct(off) | crate::regs::Route::Fixed(off) => off,
            _ => unreachable!("a ring bank is directly addressable"),
        };
        assert!(
            !dev.writes
                .borrow()
                .iter()
                .any(|(a, _)| *a == dma_idx_off || *a == bank.dma_idx())
        );
    }

    /// **The regression that matters.** `DMA_IOVA_BASE` is 4 GiB exactly,
    /// which is the first address a 32-bit master cannot express. A ring
    /// there must be refused before a register is written — a truncated
    /// descriptor base is a fetch from somebody else's page, and the part
    /// reports nothing.
    #[test]
    fn the_hosts_default_placement_is_refused_not_truncated() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        let queue = Queue::Fwdl;
        let len = queue.table_bytes() as u64;
        let iova = 1u64 << DMA_ADDR_BITS;
        assert_eq!(
            core.program_queue(queue, iova, len),
            Err(D1Refusal::Unreachable { iova, len })
        );
        // Nothing was written. A refusal that had already programmed the
        // bank would leave the engine holding a truncated base.
        assert!(dev.writes.borrow().is_empty());
    }

    /// A span that starts inside the reach and ends past it is refused too:
    /// the ceiling applies to the last byte, not the first.
    #[test]
    fn a_ring_that_straddles_the_ceiling_is_refused() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        let queue = Queue::Fwdl;
        let len = queue.table_bytes() as u64;
        let iova = (1u64 << DMA_ADDR_BITS) - 16;
        assert!(matches!(
            core.program_queue(queue, iova, len),
            Err(D1Refusal::Unreachable { .. })
        ));
    }

    /// Each queue programs its own bank. A wrong bank is a queue the part
    /// is not using, which looks exactly like a working one from here.
    #[test]
    fn each_queue_programs_its_own_bank() {
        let banks = [Queue::McuWm, Queue::Fwdl, Queue::EventWm].map(|q| q.regs().desc_base());
        assert_ne!(banks[0], banks[1]);
        assert_ne!(banks[1], banks[2]);
        assert_ne!(banks[0], banks[2]);
    }


    /// **The index ring 0 carried from D1b to M1b is refused.** `512` on a
    /// 512-slot ring names no slot; see [`dma::rx_fill_index`].
    #[test]
    #[should_panic(expected = "names no slot")]
    fn posting_the_depth_itself_is_refused() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        let queue = Queue::EventWm;
        core.post_receive(queue, queue.depth());
    }

    /// **`dma_idx` does not move because buffers were posted.**
    ///
    /// The regression this pins is a wrong *witness*, which is the kind a
    /// green test suite hides best. D1b originally claimed the engine had
    /// taken its receive buffers, evidenced by `dma_idx` advancing to the
    /// doorbell's count — and the mock had been written to do exactly that,
    /// so five tests agreed. On metal the index read zero, correctly: for a
    /// receive queue `dma_idx` is how far the engine has **written received
    /// data** (`mt76_dma_rx_process`, `dma.c:1014-1025`), and a part with no
    /// firmware receives nothing.
    ///
    /// So the model must now refuse to move it, or the next witness built
    /// on this one inherits the same fiction.
    #[test]
    fn posting_buffers_does_not_move_the_engines_index() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        let queue = Queue::EventWm;
        core.program_queue(queue, 0x4000_0000, queue.table_bytes() as u64)
            .expect("a reachable ring");
        core.post_receive(queue, dma::rx_fill_index(queue.depth()));
        assert_eq!(
            core.engine_index(queue),
            0,
            "an empty buffer is not a received frame"
        );
    }

    /// Enabling sets the configuration word first and the two direction
    /// bits after, never both in one write: an engine must not run for an
    /// instant under a half-written configuration.
    #[test]
    fn the_configuration_lands_before_the_enables() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        core.enable_engine();
        let writes: alloc::vec::Vec<u32> = dev
            .writes
            .borrow()
            .iter()
            .filter(|(a, _)| *a == dev.cell_of(MT_WFDMA0_GLO_CFG))
            .map(|(_, v)| *v)
            .collect();
        assert_eq!(writes.len(), 2, "configuration, then enables");
        let enables = MT_WFDMA0_GLO_CFG_TX_DMA_EN | MT_WFDMA0_GLO_CFG_RX_DMA_EN;
        assert_eq!(writes[0] & enables, 0, "the first write enables nothing");
        assert_eq!(writes[1] & enables, enables, "the second enables both");
        assert_eq!(
            writes[1] & GLO_CFG_SETUP,
            GLO_CFG_SETUP,
            "and keeps the configuration it just wrote"
        );
    }

    /// **The engine arms the vendor's interrupt mask, in the vendor's
    /// place**: after the reinit flag and before the MCU's own route
    /// (`mt792x_dma.c:349-358`). From D1 to M1b this write did not exist.
    #[test]
    fn enabling_arms_the_vendors_interrupt_mask_in_its_place() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        core.enable_engine();
        let writes = dev.writes.borrow();
        let at = |reg: u32| {
            writes
                .iter()
                .position(|(a, _)| *a == dev.cell_of(reg))
                .unwrap_or_else(|| panic!("{reg:#x} never written"))
        };
        let ena = at(MT_WFDMA0_HOST_INT_ENA);
        assert_eq!(writes[ena].1, dma::HOST_INT_MASK);
        assert!(at(MT_WFDMA_DUMMY_CR) < ena, "armed before the reinit flag");
        assert!(ena < at(MT_MCU2HOST_SW_INT_ENA), "armed after the MCU's route");
    }

    /// **One service pass is `mt792x_irq_tasklet`'s register half, in its
    /// order**: mask, read and acknowledge exactly the armed status bits,
    /// acknowledge the MCU's software interrupt when it is among them,
    /// unmask.
    ///
    /// The mock's status is a plain cell, so this asserts the writes the
    /// driver makes — not a belief about write-one-to-clear.
    #[test]
    fn servicing_acknowledges_what_it_read_and_the_mcus_word() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        // An RX0 completion, the MCU's software interrupt, and a bit the
        // mask does not arm (31), which must not be acknowledged.
        let sta = crate::regs::mt7925::HOST_RX_DONE_INT_ENA0 | MCU2HOST_SW_INT_ENA | (1 << 31);
        dev.poke(dev.cell_of(MT_WFDMA0_HOST_INT_STA), sta);
        dev.poke(dev.cell_of(MT_MCU_CMD), MT_MCU_CMD_WAKE_RX_PCIE);
        let seen = core.service_interrupt();
        assert_eq!(seen.sta, crate::regs::mt7925::HOST_RX_DONE_INT_ENA0 | MCU2HOST_SW_INT_ENA);
        assert_eq!(seen.mcu_cmd, MT_MCU_CMD_WAKE_RX_PCIE);
        let writes: alloc::vec::Vec<(u32, u32)> = dev.writes.borrow().clone();
        let want = [
            (dev.cell_of(MT_WFDMA0_HOST_INT_ENA), 0),
            (dev.cell_of(MT_WFDMA0_HOST_INT_STA), seen.sta),
            (dev.cell_of(MT_MCU_CMD), MT_MCU_CMD_WAKE_RX_PCIE),
            (dev.cell_of(MT_WFDMA0_HOST_INT_ENA), dma::HOST_INT_MASK),
        ];
        assert_eq!(writes, want);

        // Without the software interrupt, `MT_MCU_CMD` is not touched.
        dev.writes.borrow_mut().clear();
        dev.poke(dev.cell_of(MT_WFDMA0_HOST_INT_STA), crate::regs::mt7925::HOST_RX_DONE_INT_ENA0);
        let seen = core.service_interrupt();
        assert_eq!(seen.mcu_cmd, 0);
        assert!(
            dev.writes.borrow().iter().all(|(a, _)| *a != dev.cell_of(MT_MCU_CMD)),
            "the MCU's word was acknowledged without being raised"
        );
    }

    /// **Disarming takes both sources back**, the mask and the PCIe MAC.
    #[test]
    fn disarming_clears_the_mask_and_the_pcie_mac() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        core.arm_pcie_interrupts();
        core.enable_engine();
        core.disarm_interrupts();
        assert_eq!(dev.peek(dev.cell_of(MT_WFDMA0_HOST_INT_ENA)), 0);
        assert_eq!(dev.peek(dev.cell_of(MT7925_PCIE_MAC_INT_ENABLE)), 0);
        core.arm_pcie_interrupts();
        assert_eq!(dev.peek(dev.cell_of(MT7925_PCIE_MAC_INT_ENABLE)), 0xff);
    }

    /// **The prefetch windows are configured before the engine can fetch
    /// through them, and they carry upstream's own values.**
    ///
    /// This is the defect that cost a boot. `GLO_CFG_SETUP` sets
    /// `CSR_DISP_BASE_PTR_CHAIN_EN`, which tells the engine its descriptor
    /// cache is carved into per-ring windows — and the windows were left at
    /// reset, so all ten named a zero-length slice at offset zero. An
    /// engine with only receive work ran correctly for a whole boot; the
    /// first transmit did not.
    ///
    /// The mock models nothing here: these are plain cells, and what is
    /// asserted is only that the driver wrote the transcription. That is
    /// the right shape of test for a transcription, and it is deliberately
    /// not a behavioural claim — a mock that modelled a prefetch window
    /// would be modelling my belief about it.
    #[test]
    fn the_prefetch_windows_are_written_before_the_engine_is_configured() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        core.enable_engine();
        let writes = dev.writes.borrow().clone();
        let glo = dev.cell_of(MT_WFDMA0_GLO_CFG);
        let first_glo = writes
            .iter()
            .position(|(a, _)| *a == glo)
            .expect("the engine is configured");
        for (reg, value) in dma::PREFETCH_SETUP {
            let cell = dev.cell_of(reg);
            let at = writes
                .iter()
                .position(|(a, v)| *a == cell && *v == value)
                .unwrap_or_else(|| panic!("{reg:#x} is given its window"));
            assert!(
                at < first_glo,
                "{reg:#x} is configured before the engine is told to chain"
            );
        }
        // The ten windows do not overlap: each one's base must clear the
        // previous window's extent. This is what a zero-filled map fails.
        let mut spans: alloc::vec::Vec<(u32, u32)> = dma::PREFETCH_SETUP
            .iter()
            .map(|(_, v)| (v >> 16, v & 0xffff))
            .collect();
        spans.sort_unstable();
        for pair in spans.windows(2) {
            let (base, depth) = pair[0];
            let (next, _) = pair[1];
            assert!(depth > 0, "a window with no descriptors in it");
            assert!(
                next > base,
                "two rings share the prefetch window at {base:#x}"
            );
        }
    }

    /// The engine's own ring indices are reset, so the first doorbell is
    /// not read against whatever the ROM left behind.
    #[test]
    fn the_engine_indices_are_reset_before_the_enables() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        core.enable_engine();
        let writes = dev.writes.borrow().clone();
        let at = |chip: u32| {
            let cell = dev.cell_of(chip);
            writes.iter().position(|(a, _)| *a == cell)
        };
        for ptr in [MT_WFDMA0_RST_DTX_PTR, MT_WFDMA0_RST_DRX_PTR] {
            let cell = dev.cell_of(ptr);
            assert!(
                writes.iter().any(|(a, v)| *a == cell && *v == u32::MAX),
                "{ptr:#x} is reset across every ring"
            );
        }
        let enables = at(MT_WFDMA0_GLO_CFG).expect("the engine is configured");
        assert!(
            at(MT_WFDMA0_RST_DTX_PTR).is_some_and(|i| i < enables),
            "the indices are reset before the engine runs"
        );
        // The coalescing delay is cleared too.
        let dly = dev.cell_of(MT_WFDMA0_PRI_DLY_INT_CFG0);
        assert!(writes.iter().any(|(a, v)| *a == dly && *v == 0));
    }

    /// **The DMA scheduler is bypassed.** `dmashdl` arbitrates the transmit
    /// path only, so an engine left with it enabled and unconfigured
    /// receives perfectly and never fetches a command descriptor — the
    /// second of the two transmit-only mechanisms that were missing.
    #[test]
    fn resetting_bypasses_the_dma_scheduler_and_toggles_the_logic_reset() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        assert_eq!(core.reset_engine(|| {}, || None), Ok(()));

        assert_ne!(
            core.read(MT_DMASHDL_SW_CONTROL) & MT_DMASHDL_DMASHDL_BYPASS,
            0,
            "the scheduler is bypassed"
        );
        assert_eq!(
            core.read(MT_WFDMA0_GLO_CFG_EXT0) & MT_WFDMA0_GLO_CFG_EXT0_CSR_TX_DMASHDL_EN,
            0,
            "and disabled"
        );

        // The reset is a toggle: both bits low, then both high. A single
        // write of the set value would leave a part that was already held
        // in reset untouched.
        let bits = MT_WFDMA0_RST_DMASHDL_ALL_RST | MT_WFDMA0_RST_LOGIC_RST;
        let cell = dev.cell_of(MT_WFDMA0_RST);
        let rst: alloc::vec::Vec<u32> = dev
            .writes
            .borrow()
            .iter()
            .filter(|(a, _)| *a == cell)
            .map(|(_, v)| *v)
            .collect();
        assert_eq!(rst.len(), 2, "low, then high");
        assert_eq!(rst[0] & bits, 0);
        assert_eq!(rst[1] & bits, bits);
    }

    /// **The Wi-Fi subsystem reset, against what this part is really
    /// driven with.**
    ///
    /// The capture (`tools/mt7925_trace.py`, lab machine, 2026-11-10) shows
    /// BAR0 offset `0xf0140` going `0x11` -> write `0x10` -> (52 ms) ->
    /// `0x00` -> write `0x01` -> (20 ms) -> `0x11`. So: clear bit 0, hold,
    /// set bit 0, wait for bit 4. That is `mt792x_wfsys_reset_default`
    /// exactly, and this asserts we do the same two writes in the same
    /// order.
    #[test]
    fn the_wifi_subsystem_is_reset_the_way_the_vendor_resets_it() {
        let dev = MockMt7925::new();
        // The part answers as it did on metal: reset released, init done.
        dev.poke(MT_WFSYS_SW_RST, WFSYS_SW_RST_B | WFSYS_SW_INIT_DONE);
        let core = Core::new(&dev);
        assert_eq!(core.reset_wfsys(|| {}, || None), Ok(()));

        let cell = dev.cell_of(MT_WFSYS_SW_RST);
        let writes: alloc::vec::Vec<u32> = dev
            .writes
            .borrow()
            .iter()
            .filter(|(a, _)| *a == cell)
            .map(|(_, v)| *v)
            .collect();
        assert_eq!(writes.len(), 2, "down, then up");
        assert_eq!(writes[0] & WFSYS_SW_RST_B, 0, "the reset is asserted");
        assert_ne!(writes[1] & WFSYS_SW_RST_B, 0, "and then released");
        // Both are read-modify-writes, so a bit the part owns survives.
        assert_ne!(
            writes[0] & WFSYS_SW_INIT_DONE,
            0,
            "the other bits are carried, not cleared"
        );
    }

    /// A subsystem that never signals init-done refuses inside its budget,
    /// and carries the register rather than a bare rung: still-asserted,
    /// released-but-silent and an all-ones window are three repairs.
    #[test]
    fn a_subsystem_that_never_initialises_refuses() {
        let dev = MockMt7925::new();
        // Reset bit tracks writes; init-done never appears.
        dev.poke(MT_WFSYS_SW_RST, 0);
        let core = Core::new(&dev);
        let mut probes = 0u32;
        let out = core.reset_wfsys(|| probes += 1, || None);
        assert!(matches!(out, Err(D1Refusal::WfsysNotReady { .. })));
        assert!(
            probes <= WFSYS_HOLD_SPINS + WFSYS_INIT_SPINS + 1,
            "budgeted, not spun: {probes}"
        );
    }

    /// An engine that will not quiet refuses the reset, and the bind then
    /// leaves it unstarted rather than programming a part in an unknown
    /// state.
    #[test]
    fn a_reset_over_a_busy_engine_refuses() {
        let dev = MockMt7925::new();
        dev.busy_stuck.set(true);
        let core = Core::new(&dev);
        assert!(matches!(
            core.reset_engine(|| {}, || None),
            Err(D1Refusal::EngineBusy { .. })
        ));
        assert_eq!(
            core.read(MT_DMASHDL_SW_CONTROL) & MT_DMASHDL_DMASHDL_BYPASS,
            0,
            "nothing past the quiet wait ran"
        );
    }

    /// The engine is stopped and seen to be quiet.
    #[test]
    fn disabling_waits_for_the_engine_to_go_quiet() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        core.enable_engine();
        assert_eq!(core.disable_engine(|| {}, || None), Ok(()));
        let cfg = core.read(MT_WFDMA0_GLO_CFG);
        assert_eq!(cfg & GLO_CFG_TEARDOWN, 0, "the teardown bits are clear");
        assert_eq!(cfg & GLO_CFG_BUSY, 0, "and the engine is idle");
    }

    /// **An engine that never goes quiet must not let the pool be freed.**
    /// This is the refusal that stands between a teardown and a write into
    /// pages that now belong to somebody else.
    #[test]
    fn an_engine_that_never_quiets_refuses_within_its_budget() {
        let dev = MockMt7925::new();
        dev.busy_stuck.set(true);
        let core = Core::new(&dev);
        core.enable_engine();
        let mut probes = 0u32;
        let refusal = core.disable_engine(|| probes += 1, || None);
        assert!(matches!(refusal, Err(D1Refusal::EngineBusy { .. })));
        assert!(
            probes <= QUIET_SPINS_UNCLOCKED,
            "the wait is budgeted, not spun: {probes} probes"
        );
    }

    /// **A second caller continues the conversation; it does not restart
    /// it.**
    ///
    /// This is the flash of 2026-11-11, as a test. `ask_the_part` ran two
    /// exchanges and left the part at `cpu_idx = 2` with two answers done
    /// in the event ring. `download_patch` then built its own `Rings` with
    /// the counters and the sequence at zero, and the part's own replies
    /// to D1c came back as replies to C2b's first two commands — matching,
    /// because the sequence had restarted with them. Two commands the part
    /// never saw were reported answered, and the first rung that could not
    /// be faked refused.
    ///
    /// So this builds **two** `Rings` over one set of buffers, the way two
    /// callers do, and asserts the second one cannot be lied to: the
    /// doorbell goes forward, the cursor goes forward, and a slot whose
    /// answer was already taken does **not** answer again.
    /// Stage one 44-byte answer header into event slot `slot`, and mark
    /// that slot done.
    ///
    /// `option` is the byte at 38 the triage reads; `seq` the byte at 37.
    fn stage_answer(
        rx_table: &crate::mock::ReachableBuf,
        rx_buffers: &crate::mock::ReachableBuf,
        slot: usize,
        seq: u8,
        option: u8,
    ) {
        const SLOT: usize = 2048;
        let at = slot * SLOT;
        let answers = DmaShared::new(rx_buffers);
        answers.write_u8_at(at + 36, Command::PatchSem.id()).unwrap();
        answers.write_u8_at(at + 37, seq).unwrap();
        answers
            .write_u8_at(at + crate::mcu::MCU_RXD_OPTION_AT, option)
            .unwrap();
        answers.write_u8_at(at + 40, 2).unwrap();
        let posted =
            Descriptor::receive(rx_buffers.iova().0 + at as u64, SLOT as u32).expect("reachable");
        DmaShared::new(rx_table)
            .write_u32_at(
                Descriptor::ctrl_at(slot),
                posted.ctrl | crate::regs::mt7925::MT_DMA_CTL_DMA_DONE,
            )
            .unwrap();
    }

    /// Build the four buffers an exchange needs, with `slots` event slots.
    fn exchange_rings(
        pool: &ReachablePool,
        slots: usize,
    ) -> (
        crate::mock::ReachableBuf,
        crate::mock::ReachableBuf,
        crate::mock::ReachableBuf,
        crate::mock::ReachableBuf,
    ) {
        (
            pool.alloc(dma::DESC_BYTES * 4),
            pool.alloc(128),
            pool.alloc(dma::DESC_BYTES * (slots + 1)),
            pool.alloc(2048 * slots),
        )
    }

    /// **An unsolicited event is not an answer.**
    ///
    /// The event ring is shared with the part's own traffic — firmware
    /// logs, coredumps, HIF-control notices — and
    /// `mt7925_mcu_rx_event` routes on `option & UNSOLICITED_EVENT`
    /// *before* it looks at a sequence. A driver that took the next done
    /// descriptor would read one of those as its reply and then fail the
    /// sequence check, naming the transport for somebody else's message.
    ///
    /// Nothing in C2 could reach this: no unsolicited source is enabled
    /// and this driver deliberately does not turn the firmware log on.
    /// The rung exists so that stops being true by luck.
    #[test]
    fn an_unsolicited_event_is_stepped_over_rather_than_answered() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let (tx_table, tx_payload, rx_table, rx_buffers) = exchange_rings(&pool, 2);

        // Slot 0: the part talking to nobody. Slot 1: this exchange's
        // actual answer, for the first sequence the driver stamps.
        stage_answer(&rx_table, &rx_buffers, 0, 0x7f, crate::mcu::MCU_UNI_CMD_UNSOLICITED_EVENT);
        stage_answer(&rx_table, &rx_buffers, 1, 1, 0);

        let mut conv = Conversation::new();
        let mut rings = Rings {
            tx_table: &tx_table,
            fwdl_table: &tx_table,
            fwdl_payload: &tx_payload,
            fwdl_chunk_bytes: tx_payload.len(),
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: 2048,
            data: None,
            conv: &mut conv,
        };
        let answer = core
            .exchange(
                &mut rings,
                Command::PatchSem,
                &crate::mcu::patch_sem_body(true),
                || {},
                || None,
            )
            .expect("the answer in slot one");
        assert_eq!(answer.seq, 1, "the exchange took the event as its reply");

        // Both slots consumed: the cursor stepped over the event rather
        // than leaving it to be read again forever.
        assert_eq!(rings.conv.rx_next, 2, "the event was not consumed");
        let table = DmaShared::new(&rx_table);
        for slot in 0..2 {
            let ctrl = table.read_u32_at(Descriptor::ctrl_at(slot)).unwrap();
            assert!(
                !Descriptor::from_ctrl(ctrl).done(),
                "slot {slot} still reads done",
            );
        }
    }

    /// **A reply bearing somebody else's sequence is discarded, not
    /// refused.**
    ///
    /// Upstream's `do { … } while (ret == -EAGAIN)` loop is load-bearing:
    /// `-EAGAIN` is exactly what a sequence mismatch returns, and it keeps
    /// reading until its deadline. This driver refused on the first
    /// descriptor instead, which turns a stale reply into a fault report
    /// against the current command.
    #[test]
    fn a_reply_for_another_sequence_is_discarded_and_the_poll_continues() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let (tx_table, tx_payload, rx_table, rx_buffers) = exchange_rings(&pool, 2);

        // Slot 0 is a solicited reply — option clear — but for sequence 9,
        // which no outstanding exchange is waiting on.
        stage_answer(&rx_table, &rx_buffers, 0, 9, 0);
        stage_answer(&rx_table, &rx_buffers, 1, 1, 0);

        let mut conv = Conversation::new();
        let mut rings = Rings {
            tx_table: &tx_table,
            fwdl_table: &tx_table,
            fwdl_payload: &tx_payload,
            fwdl_chunk_bytes: tx_payload.len(),
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: 2048,
            data: None,
            conv: &mut conv,
        };
        let answer = core
            .exchange(
                &mut rings,
                Command::PatchSem,
                &crate::mcu::patch_sem_body(true),
                || {},
                || None,
            )
            .expect("the answer in slot one");
        assert_eq!(answer.seq, 1);
        assert_eq!(rings.conv.rx_next, 2, "the stale reply was not stepped over");
    }

    /// The rings [`Core::restart`]'s tests share: one reply slot, nothing
    /// on ring 2.
    fn restart_rings<'b>(
        bufs: &'b (
            crate::mock::ReachableBuf,
            crate::mock::ReachableBuf,
            crate::mock::ReachableBuf,
            crate::mock::ReachableBuf,
        ),
        conv: &'b mut Conversation,
    ) -> Rings<'b, crate::mock::ReachableBuf> {
        Rings {
            tx_table: &bufs.0,
            fwdl_table: &bufs.0,
            fwdl_payload: &bufs.1,
            fwdl_chunk_bytes: bufs.1.len(),
            tx_payload: &bufs.1,
            rx_table: &bufs.2,
            rx_buffers: &bufs.3,
            rx_buf_bytes: 2048,
            data: None,
            conv,
        }
    }

    /// **The restart is the vendor's bytes, exactly.** The first command
    /// `mt7925e` sent on this laptop (2026-11-20 capture, `TX hw_idx=15
    /// head=0 len=68`), byte for byte: the legacy descriptor with
    /// `NIC_POWER_CTRL` (`0x04`), sequence 1, and a power-mode byte of 1.
    #[test]
    fn the_restart_is_the_captured_command() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let bufs = exchange_rings(&pool, 1);
        let mut conv = Conversation::new();
        let mut rings = restart_rings(&bufs, &mut conv);
        let _ = core.restart(&mut rings, || {}, || None);

        let mut want = [0u8; 68];
        want[..8].copy_from_slice(&[0x44, 0x00, 0x00, 0x41, 0x00, 0x40, 0x00, 0x00]);
        want[32..40].copy_from_slice(&[0x24, 0x00, 0x00, 0x80, 0x04, 0xa0, 0x03, 0x01]);
        want[64] = 0x01;
        let mut sent = [0u8; 68];
        bufs.1.read_bytes_at(0, &mut sent).unwrap();
        assert_eq!(sent, want);
        // On the command queue, behind one descriptor.
        assert_eq!(dev.peek(dev.cell_of(Queue::McuWm.regs().cpu_idx())), 1);
    }

    /// **A ROM that fetches the restart and stays quiet has not refused**
    /// — upstream never waits for a reply. Power-on is reported, not
    /// required.
    #[test]
    fn a_fetched_unanswered_restart_is_not_a_refusal() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let bufs = exchange_rings(&pool, 1);
        let mut conv = Conversation::new();
        let mut rings = restart_rings(&bufs, &mut conv);
        // The engine took the descriptor; the firmware state is power-on.
        dev.poke(dev.cell_of(Queue::McuWm.regs().dma_idx()), 1);
        dev.poke(dev.cell_of(crate::regs::mt7925::MT_CONN_ON_MISC), 0x1);
        assert_eq!(
            core.restart(&mut rings, || {}, || None),
            Ok(Restarted { answered: false, powered: true })
        );
    }

    /// **Power-on is the field, not the bit.** A running firmware reads
    /// `0x3`, which has `FW_PWR_ON` set and is not the state upstream
    /// polls for.
    #[test]
    fn a_running_core_is_not_power_on() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let bufs = exchange_rings(&pool, 1);
        let mut conv = Conversation::new();
        let mut rings = restart_rings(&bufs, &mut conv);
        dev.poke(dev.cell_of(Queue::McuWm.regs().dma_idx()), 1);
        dev.poke(dev.cell_of(crate::regs::mt7925::MT_CONN_ON_MISC), 0x3);
        assert_eq!(
            core.restart(&mut rings, || {}, || None),
            Ok(Restarted { answered: false, powered: false })
        );
    }

    /// **A restart the engine never fetched refuses**: its bytes are still
    /// in the staging buffer the next command would overwrite.
    #[test]
    fn an_unfetched_restart_refuses() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let bufs = exchange_rings(&pool, 1);
        let mut conv = Conversation::new();
        let mut rings = restart_rings(&bufs, &mut conv);
        assert!(matches!(
            core.restart(&mut rings, || {}, || None),
            Err(D1Refusal::NoAnswer { tx_idx: 0, tx_next: 1, .. })
        ));
    }

    /// **A reply is consumed here**, so it cannot land in the semaphore
    /// probe's triage as somebody else's.
    #[test]
    fn an_answered_restart_consumes_its_reply() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let bufs = exchange_rings(&pool, 1);
        stage_answer(&bufs.2, &bufs.3, 0, 1, 0);
        let mut conv = Conversation::new();
        let mut rings = restart_rings(&bufs, &mut conv);
        let out = core.restart(&mut rings, || {}, || None);
        assert!(matches!(out, Ok(Restarted { answered: true, .. })), "{out:?}");
        assert_eq!(conv.rx_next, 1, "the reply was left for the next exchange");
    }

    /// **The restart waits on its own budget**, not the twenty seconds an
    /// answered command gets: a quiet ROM costs what upstream's own
    /// power-on poll costs. Counted in probes, on the unclocked fallback.
    #[test]
    fn the_restart_listens_for_a_second() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let bufs = exchange_rings(&pool, 1);
        let mut conv = Conversation::new();
        let mut rings = restart_rings(&bufs, &mut conv);
        dev.poke(dev.cell_of(Queue::McuWm.regs().dma_idx()), 1);
        let mut probes = 0u32;
        let _ = core.restart(&mut rings, || probes += 1, || None);
        assert!(
            probes <= RESTART_SPINS_UNCLOCKED + POWER_ON_SPINS_UNCLOCKED + 2,
            "{probes} probes: the restart waited like an answered command"
        );
        assert!(RESTART_SPINS_UNCLOCKED < MCU_SPINS_UNCLOCKED);
    }

    /// **A refusal after the triage gave a reply away says so.**
    ///
    /// M1b's first metal photograph read `NoAnswer` and nothing else, and
    /// that word could not tell silence from an answer this exchange
    /// discarded as somebody else's. The only staged reply here is for
    /// another sequence, so the exchange runs out of budget — and must
    /// report that one reply came and went, beside its own cursors.
    #[test]
    fn a_refusal_after_a_discarded_reply_counts_it() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let (tx_table, tx_payload, rx_table, rx_buffers) = exchange_rings(&pool, 2);

        stage_answer(&rx_table, &rx_buffers, 0, 9, 0);

        let mut conv = Conversation::new();
        let mut rings = Rings {
            tx_table: &tx_table,
            fwdl_table: &tx_table,
            fwdl_payload: &tx_payload,
            fwdl_chunk_bytes: tx_payload.len(),
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: 2048,
            data: None,
            conv: &mut conv,
        };
        let out = core.exchange(
            &mut rings,
            Command::PatchSem,
            &crate::mcu::patch_sem_body(true),
            || {},
            || None,
        );
        match out {
            Err(D1Refusal::NoAnswer {
                seq,
                tx_next,
                rx_next,
                skipped,
                ..
            }) => {
                assert_eq!(seq, 1);
                assert_eq!(skipped, 1, "the discarded reply was not counted");
                assert_eq!(tx_next, 1, "the doorbell's own cursor");
                assert_eq!(rx_next, 1, "the poll's cursor, past the discarded slot");
            }
            other => panic!("expected NoAnswer, got {other:?}"),
        }
    }

    /// **An answer on receive ring 2 is read, counted, and its slot
    /// handed back.**
    ///
    /// M1b's third photograph: the RAM firmware alive, unalarmed, and
    /// silent on ring 0. Upstream takes an event from whichever ring it
    /// lands on. Here ring 0 stays empty and the answer is on ring 2, so
    /// an exchange that polled ring 0 alone runs its budget out — and one
    /// that read the answer through ring 0's buffers parses the wrong
    /// bytes.
    #[test]
    fn an_answer_on_ring_two_is_read_counted_and_recycled() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let (tx_table, tx_payload, rx_table, rx_buffers) = exchange_rings(&pool, 2);
        let depth = Queue::EventData.depth() as usize;
        let data_table = pool.alloc(dma::DESC_BYTES * depth);
        let data_buffers = pool.alloc(2048 * depth);

        stage_answer(&data_table, &data_buffers, 0, 1, 0);

        let mut conv = Conversation::new();
        let mut rings = Rings {
            tx_table: &tx_table,
            fwdl_table: &tx_table,
            fwdl_payload: &tx_payload,
            fwdl_chunk_bytes: tx_payload.len(),
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: 2048,
            data: Some(DataRing {
                table: &data_table,
                buffers: &data_buffers,
            }),
            conv: &mut conv,
        };
        let answer = core
            .exchange(
                &mut rings,
                Command::PatchSem,
                &crate::mcu::patch_sem_body(true),
                || {},
                || None,
            )
            .expect("the answer on ring 2");
        assert_eq!(answer.seq, 1);
        assert_eq!(rings.conv.data_answers, 1, "the ring that answered was not counted");
        assert_eq!(rings.conv.data_next, 1);
        assert_eq!(rings.conv.rx_next, 0, "ring 0's cursor moved for ring 2's answer");

        // The slot is re-armed whole: its own buffer, the full length,
        // done clear.
        let armed = Descriptor::receive(data_buffers.iova().0, 2048).expect("reachable");
        let table = DmaShared::new(&data_table);
        for (word, want) in armed.words().into_iter().enumerate() {
            assert_eq!(
                table.read_u32_at(Descriptor::slot_at(0) + word * 4).unwrap(),
                want,
                "slot 0 word {word} was not re-armed",
            );
        }
        // And given back: one slot short of the cursor, the way upstream
        // keeps a full ring and an empty one apart.
        let cpu = dev.peek(dev.cell_of(Queue::EventData.regs().cpu_idx()));
        assert_eq!(cpu, 0, "ring 2's slot was not handed back to the engine");
    }

    /// **The part's own traffic on ring 2 is stepped over and recycled.**
    ///
    /// The ring exists because the RAM firmware may talk on it
    /// unprompted, so the triage must apply there exactly as on ring 0 —
    /// and a slot it discards must go back to the engine, or thirty-one
    /// log lines stall the ring.
    #[test]
    fn an_unsolicited_event_on_ring_two_is_recycled_not_answered() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let (tx_table, tx_payload, rx_table, rx_buffers) = exchange_rings(&pool, 2);
        let depth = Queue::EventData.depth() as usize;
        let data_table = pool.alloc(dma::DESC_BYTES * depth);
        let data_buffers = pool.alloc(2048 * depth);

        stage_answer(&data_table, &data_buffers, 0, 0x7f, crate::mcu::MCU_UNI_CMD_UNSOLICITED_EVENT);
        stage_answer(&data_table, &data_buffers, 1, 1, 0);

        let mut conv = Conversation::new();
        let mut rings = Rings {
            tx_table: &tx_table,
            fwdl_table: &tx_table,
            fwdl_payload: &tx_payload,
            fwdl_chunk_bytes: tx_payload.len(),
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: 2048,
            data: Some(DataRing {
                table: &data_table,
                buffers: &data_buffers,
            }),
            conv: &mut conv,
        };
        let answer = core
            .exchange(
                &mut rings,
                Command::PatchSem,
                &crate::mcu::patch_sem_body(true),
                || {},
                || None,
            )
            .expect("the answer in ring 2's slot one");
        assert_eq!(answer.seq, 1, "the event was taken as the reply");
        assert_eq!(rings.conv.data_next, 2);
        assert_eq!(rings.conv.data_answers, 1, "a discarded event counted as an answer");
        let table = DmaShared::new(&data_table);
        for slot in 0..2 {
            let ctrl = table.read_u32_at(Descriptor::ctrl_at(slot)).unwrap();
            assert!(!Descriptor::from_ctrl(ctrl).done(), "slot {slot} still reads done");
        }
        let cpu = dev.peek(dev.cell_of(Queue::EventData.regs().cpu_idx()));
        assert_eq!(cpu, 1, "both slots should be back with the engine");
    }

    /// **A silent part says what ring 2 held.**
    ///
    /// The next photograph's question: if the answer still does not come,
    /// did the part write into ring 2 where the host could not see it?
    /// The engine's index is three slots past the host's cursor here, and
    /// nothing reads done — which must come back as three, not as
    /// silence.
    #[test]
    fn a_silent_part_reports_what_ring_two_holds() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let (tx_table, tx_payload, rx_table, rx_buffers) = exchange_rings(&pool, 1);
        let depth = Queue::EventData.depth() as usize;
        let data_table = pool.alloc(dma::DESC_BYTES * depth);
        let data_buffers = pool.alloc(2048 * depth);
        dev.poke(dev.cell_of(Queue::EventData.regs().dma_idx()), 4);

        let mut conv = Conversation::new();
        conv.data_next = 65;
        let mut rings = Rings {
            tx_table: &tx_table,
            fwdl_table: &tx_table,
            fwdl_payload: &tx_payload,
            fwdl_chunk_bytes: tx_payload.len(),
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: 2048,
            data: Some(DataRing {
                table: &data_table,
                buffers: &data_buffers,
            }),
            conv: &mut conv,
        };
        let out = core.exchange(
            &mut rings,
            Command::PatchSem,
            &crate::mcu::patch_sem_body(true),
            || {},
            || None,
        );
        match out {
            Err(D1Refusal::NoAnswer { data_pending, .. }) => {
                // The cursor is absolute (65, two laps and one); the index
                // is a ring index (4). Three past it — not four, which is
                // the index read as if the host had consumed nothing.
                assert_eq!(data_pending, Some(3));
            }
            other => panic!("expected NoAnswer, got {other:?}"),
        }
    }

    /// **A header too short to read is refused, never discarded.**
    ///
    /// The triage discards what it can prove is not ours. A slot it cannot
    /// read is not proof of anything, and throwing it away would turn a
    /// malformed answer into `NoAnswer` — a refusal that names the
    /// transport for a fault in the message. It is kept, so
    /// `Response::parse` refuses it as `TooShort` and the note says so.
    #[test]
    fn a_header_too_short_to_triage_is_refused_rather_than_skipped() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let (tx_table, tx_payload, rx_table, rx_buffers) = exchange_rings(&pool, 2);

        // A done descriptor naming fewer bytes than a header needs.
        let short = Descriptor::receive(rx_buffers.iova().0, 8).expect("reachable");
        DmaShared::new(&rx_table)
            .write_u32_at(
                Descriptor::ctrl_at(0),
                short.ctrl | crate::regs::mt7925::MT_DMA_CTL_DMA_DONE,
            )
            .unwrap();
        // A perfectly good answer behind it, which must NOT be reached.
        stage_answer(&rx_table, &rx_buffers, 1, 1, 0);

        let mut conv = Conversation::new();
        let mut rings = Rings {
            tx_table: &tx_table,
            fwdl_table: &tx_table,
            fwdl_payload: &tx_payload,
            fwdl_chunk_bytes: tx_payload.len(),
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: 2048,
            data: None,
            conv: &mut conv,
        };
        let refusal = core
            .exchange(
                &mut rings,
                Command::PatchSem,
                &crate::mcu::patch_sem_body(true),
                || {},
                || None,
            )
            .expect_err("a short header must refuse");
        assert_eq!(
            refusal,
            D1Refusal::BadAnswer(crate::mcu::McuError::TooShort),
            "a short header was skipped instead of refused",
        );
    }

    #[test]
    fn a_second_rings_over_the_same_part_continues_the_conversation() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let tx_table = pool.alloc(dma::DESC_BYTES * 4);
        let tx_payload = pool.alloc(128);
        let rx_table = pool.alloc(dma::DESC_BYTES * 4);
        let rx_buffers = pool.alloc(2048 * 2);

        // One answer, in slot zero, for the first sequence the driver will
        // stamp. The part writes nothing else for the rest of the test.
        let answers = DmaShared::new(&rx_buffers);
        answers.write_u8_at(36, Command::PatchSem.id()).unwrap();
        answers.write_u8_at(37, 1).unwrap();
        answers.write_u8_at(40, 2).unwrap();
        let posted = Descriptor::receive(rx_buffers.iova().0, 2048).expect("reachable");
        DmaShared::new(&rx_table)
            .write_u32_at(
                Descriptor::ctrl_at(0),
                posted.ctrl | crate::regs::mt7925::MT_DMA_CTL_DMA_DONE,
            )
            .unwrap();

        // **The conversation outlives both `Rings`.** That is the whole
        // fix: it is borrowed, so neither caller can start it over.
        let mut conv = Conversation::new();
        let body = crate::mcu::patch_sem_body(true);

        {
            let mut first = Rings {
                tx_table: &tx_table,
                fwdl_table: &tx_table,
                fwdl_payload: &tx_payload,
                fwdl_chunk_bytes: tx_payload.len(),
                tx_payload: &tx_payload,
                rx_table: &rx_table,
                rx_buffers: &rx_buffers,
                rx_buf_bytes: 2048,
                data: None,
                conv: &mut conv,
            };
            let answer = core
                .exchange(&mut first, Command::PatchSem, &body, || {}, || None)
                .expect("the one staged answer");
            assert_eq!(answer.seq, 1);
        }
        assert_eq!(conv.tx_next, 1);
        assert_eq!(conv.rx_next, 1);

        // **Slot zero's done bit was taken with the answer.** Before this,
        // the engine set it and nobody cleared it, so the slot answered
        // every poll that ever reached it.
        let taken = DmaShared::new(&rx_table)
            .read_u32_at(Descriptor::ctrl_at(0))
            .unwrap();
        assert!(
            !Descriptor::from_ctrl(taken).done(),
            "a consumed answer still reads as done, and will be read again",
        );

        // The second caller. Same buffers, same part, same conversation —
        // and no new answer staged, because the part was never asked
        // anything it could reply to.
        let mut second = Rings {
            tx_table: &tx_table,
            fwdl_table: &tx_table,
            fwdl_payload: &tx_payload,
            fwdl_chunk_bytes: tx_payload.len(),
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: 2048,
            data: None,
            conv: &mut conv,
        };
        let out = core.exchange(&mut second, Command::PatchSem, &body, || {}, || None);
        assert!(
            matches!(out, Err(D1Refusal::NoAnswer { seq: 2, .. })),
            "the second caller was answered by the first caller's reply: {out:?}",
        );

        // And the doorbell went *forward*. Restarting at zero wrote an
        // index below the engine's own, which is the stale-pointer state
        // `enable_engine` resets `RST_DTX_PTR` to escape.
        let cell = dev.cell_of(Queue::McuWm.regs().cpu_idx());
        let rung: alloc::vec::Vec<u32> = dev
            .writes
            .borrow()
            .iter()
            .filter(|(a, _)| *a == cell)
            .map(|(_, v)| *v)
            .collect();
        assert_eq!(
            rung,
            alloc::vec![1, 2],
            "the second caller rang the first caller's doorbell again",
        );
    }

    /// A command is staged, published, and its answer read back.
    ///
    /// **The answer here is written by the test, not by the mock.** The
    /// mock models registers; it does not model a running MCU, and a model
    /// that invented a reply would be the D1b mistake again — agreeing with
    /// the belief instead of checking it. What this pins is the framing and
    /// the plumbing on both sides of a reply, which is all a host test can
    /// honestly claim.
    #[test]
    fn a_command_is_staged_and_its_answer_is_read() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        // **Real DMA buffers, not stack arrays.** A stack array is not
        // memory a device writes behind your back, which is how a
        // non-volatile completion poll passed here for as long as it did.
        let pool = ReachablePool::new();
        let tx_table = pool.alloc(dma::DESC_BYTES);
        let tx_payload = pool.alloc(128);
        let rx_table = pool.alloc(dma::DESC_BYTES);
        let rx_buffers = pool.alloc(2048);

        // The part's side: an answer for sequence 7, with the done bit the
        // engine would have set. Written through the same volatile seam the
        // engine's own write is modelled by.
        let answer_bytes = DmaShared::new(&rx_buffers);
        answer_bytes.write_u8_at(36, Command::PatchSem.id()).unwrap();
        answer_bytes.write_u8_at(37, 1).unwrap();
        answer_bytes.write_u8_at(40, 2).unwrap(); // PATCH_NOT_DL_SEM_SUCCESS
        let posted = Descriptor::receive(rx_buffers.iova().0, 2048).expect("reachable");
        let completed = Descriptor::from_ctrl(
            posted.ctrl | crate::regs::mt7925::MT_DMA_CTL_DMA_DONE,
        );
        DmaShared::new(&rx_table)
            .write_u32_at(Descriptor::ctrl_at(0), completed.ctrl)
            .unwrap();

        let mut conv = Conversation::new();
        let mut rings = Rings {
            tx_table: &tx_table,
            fwdl_table: &tx_table,
            fwdl_payload: &tx_payload,
            fwdl_chunk_bytes: tx_payload.len(),
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: 2048,
            data: None,
            conv: &mut conv,
        };
        let answer = core
            .exchange(
                &mut rings,
                Command::PatchSem,
                &crate::mcu::patch_sem_body(true),
                || {},
                || None,
            )
            .expect("an answer");
        assert_eq!(answer.seq, 1);
        assert_eq!(
            crate::mcu::Semaphore::from_status(answer.status),
            Ok(crate::mcu::Semaphore::Granted)
        );
        // The command really went into the staging buffer and the
        // descriptor really carries that buffer's own address and length —
        // an address the caller can no longer get wrong, because it is read
        // off the buffer rather than passed beside it.
        let published = DmaShared::new(&tx_table);
        assert_eq!(
            published.read_u32_at(Descriptor::slot_at(0)).unwrap(),
            tx_payload.iova().0 as u32
        );
        let ctrl = published.read_u32_at(Descriptor::ctrl_at(0)).unwrap();
        assert_eq!(
            Descriptor::from_ctrl(ctrl).written(),
            (crate::mcu::MCU_TXD_BYTES + 4) as u32
        );
    }

    /// **Two commands in a row, which is what C2b actually does.**
    ///
    /// Every earlier test sent exactly one. D1c sent one whose answer it
    /// read and one whose answer it discarded, so `exchange` carried two
    /// single-use assumptions for a long time and neither could show:
    /// the doorbell was written as a constant `1`, which is a no-op the
    /// second time and leaves the command unfetched; and the answer was
    /// always polled in event slot zero, whose done bit the engine never
    /// clears, so the second call read the first call's completion
    /// immediately and parsed a stale response.
    ///
    /// It cost a boot and a note 79. This asserts the ring positions
    /// move.
    #[test]
    fn a_second_command_rings_a_new_doorbell_and_reads_a_new_slot() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let tx_table = pool.alloc(Queue::McuWm.table_bytes());
        let tx_payload = pool.alloc(128);
        let rx_table = pool.alloc(Queue::EventWm.table_bytes());
        let rx_buffers = pool.alloc(4096);
        let mut conv = Conversation::new();
        let mut rings = Rings {
            tx_table: &tx_table,
            fwdl_table: &tx_table,
            fwdl_payload: &tx_payload,
            fwdl_chunk_bytes: tx_payload.len(),
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: 2048,
            data: None,
            conv: &mut conv,
        };

        // Stage an answer in each of the first two event slots, each in
        // its own buffer, exactly as the engine fills them.
        let answers = DmaShared::new(&rx_buffers);
        let slots = DmaShared::new(&rx_table);
        let mut posted = [Descriptor::default(); 2];
        for (n, seq) in [(0usize, 1u8), (1, 2)] {
            let at = n * 2048;
            answers.write_u8_at(at + 36, Command::PatchSem.id()).unwrap();
            answers.write_u8_at(at + 37, seq).unwrap();
            answers.write_u8_at(at + 40, 2).unwrap();
            posted[n] = Descriptor::receive(rx_buffers.iova().0 + at as u64, 2048)
                .expect("reachable");
        }
        // **Only the first slot is done.** The second answer arrives while
        // the driver waits, which is the whole discriminator: a call that
        // polls slot zero again finds a done bit already set and returns
        // without waiting at all. Staging both up front hides that — the
        // payload offset compensates and the stale read passes.
        slots
            .write_u32_at(
                Descriptor::ctrl_at(0),
                posted[0].ctrl | crate::regs::mt7925::MT_DMA_CTL_DMA_DONE,
            )
            .unwrap();

        let body = crate::mcu::patch_sem_body(true);
        let first = core
            .exchange(&mut rings, Command::PatchSem, &body, || {}, || None)
            .expect("the first answer");
        assert_eq!(first.seq, 1);
        let mut probes = 0u32;
        let second = core
            .exchange(
                &mut rings,
                Command::PatchSem,
                &body,
                || {
                    probes += 1;
                    if probes == 4 {
                        slots
                            .write_u32_at(
                                Descriptor::ctrl_at(1),
                                posted[1].ctrl | crate::regs::mt7925::MT_DMA_CTL_DMA_DONE,
                            )
                            .unwrap();
                    }
                },
                || None,
            )
            .expect("the second answer, from its own slot");
        assert_eq!(second.seq, 2, "the second call read slot zero again");
        assert!(
            probes >= 4,
            "it did not wait: slot zero's stale done bit was read ({probes} probes)"
        );

        assert_eq!(rings.conv.tx_next, 2);
        assert_eq!(rings.conv.rx_next, 2);

        // The doorbell advertised 1 then 2. Writing 1 twice would leave
        // the second command sitting unfetched in the ring.
        let cell = dev.cell_of(Queue::McuWm.regs().cpu_idx());
        let rung: alloc::vec::Vec<u32> = dev
            .writes
            .borrow()
            .iter()
            .filter(|(a, _)| *a == cell)
            .map(|(_, v)| *v)
            .collect();
        assert_eq!(rung, alloc::vec![1, 2], "the producer index must move");
    }

    /// A part that never answers refuses inside its budget, and says where
    /// the engine's index stood — zero meaning nothing was written at all.
    #[test]
    fn a_silent_part_refuses_within_its_budget() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let tx_table = pool.alloc(dma::DESC_BYTES);
        let tx_payload = pool.alloc(128);
        let rx_table = pool.alloc(dma::DESC_BYTES);
        let rx_buffers = pool.alloc(2048);
        let mut conv = Conversation::new();
        let mut rings = Rings {
            tx_table: &tx_table,
            fwdl_table: &tx_table,
            fwdl_payload: &tx_payload,
            fwdl_chunk_bytes: tx_payload.len(),
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: 2048,
            data: None,
            conv: &mut conv,
        };
        let mut probes = 0u32;
        let out = core.exchange(
            &mut rings,
            Command::PatchSem,
            &crate::mcu::patch_sem_body(true),
            || probes += 1,
            || None,
        );
        assert!(matches!(
            out,
            Err(D1Refusal::NoAnswer { seq: 1, dma_idx: 0, tx_idx: 0, tx_next: 1, rx_next: 0, skipped: 0, data_pending: None })
        ));
        assert!(probes <= MCU_SPINS_UNCLOCKED, "budgeted, not spun");
    }

    /// **The answer arrives while the driver is waiting**, which is the
    /// only way it ever arrives on hardware.
    ///
    /// Every other exchange test pre-sets the done bit, so they prove the
    /// parse and not the poll: a driver that read the control word exactly
    /// once would pass all of them. Here the part answers on the eighth
    /// delay, so the exchange has to keep looking — and keep looking at
    /// memory, not at a value it loaded before the loop.
    ///
    /// This is also the shape the compiler is entitled to break. The
    /// control word is memory nothing inside the loop writes, so a
    /// non-volatile read of it may be hoisted out; the exchange would then
    /// run its budget down and refuse `NoAnswer` to an answer that did
    /// arrive. A debug test cannot observe that choice, which is exactly
    /// why the access is volatile by construction rather than by hope.
    #[test]
    fn an_answer_that_arrives_mid_wait_is_seen() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let tx_table = pool.alloc(dma::DESC_BYTES);
        let tx_payload = pool.alloc(128);
        let rx_table = pool.alloc(dma::DESC_BYTES);
        let rx_buffers = pool.alloc(2048);

        let answer = DmaShared::new(&rx_buffers);
        answer.write_u8_at(36, Command::PatchSem.id()).unwrap();
        answer.write_u8_at(37, 1).unwrap();
        answer.write_u8_at(40, 2).unwrap(); // PATCH_NOT_DL_SEM_SUCCESS
        let posted = Descriptor::receive(rx_buffers.iova().0, 2048).expect("reachable");
        let slot = DmaShared::new(&rx_table);
        // Posted, but **not** done: the engine has not written yet.
        slot.write_u32_at(Descriptor::ctrl_at(0), posted.ctrl).unwrap();

        let mut conv = Conversation::new();
        let mut rings = Rings {
            tx_table: &tx_table,
            fwdl_table: &tx_table,
            fwdl_payload: &tx_payload,
            fwdl_chunk_bytes: tx_payload.len(),
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: 2048,
            data: None,
            conv: &mut conv,
        };
        let mut probes = 0u32;
        let out = core.exchange(
            &mut rings,
            Command::PatchSem,
            &crate::mcu::patch_sem_body(true),
            || {
                probes += 1;
                if probes == 8 {
                    // The part, answering: the done bit appears in memory
                    // the driver is already waiting on.
                    slot.write_u32_at(
                        Descriptor::ctrl_at(0),
                        posted.ctrl | crate::regs::mt7925::MT_DMA_CTL_DMA_DONE,
                    )
                    .unwrap();
                }
            },
            || None,
        );
        let answer = out.expect("the answer that arrived");
        assert_eq!(answer.seq, 1);
        assert_eq!(
            crate::mcu::Semaphore::from_status(answer.status),
            Ok(crate::mcu::Semaphore::Granted)
        );
        assert!(probes >= 8, "it really waited: {probes} probes");
        assert!(probes < MCU_SPINS_UNCLOCKED, "and did not run its budget down");
    }

    /// A command larger than the staging buffer is refused, not truncated.
    #[test]
    fn an_oversized_command_is_refused() {
        let dev = MockMt7925::new();
        let core = Core::new(&dev);
        let pool = ReachablePool::new();
        let tx_table = pool.alloc(dma::DESC_BYTES);
        // Sixteen bytes is smaller than the header alone, and the buffer
        // reports sixteen whatever the host double allocated beneath it.
        let tx_payload = pool.alloc(16);
        let rx_table = pool.alloc(dma::DESC_BYTES);
        let rx_buffers = pool.alloc(64);
        let mut conv = Conversation::new();
        let mut rings = Rings {
            tx_table: &tx_table,
            fwdl_table: &tx_table,
            fwdl_payload: &tx_payload,
            fwdl_chunk_bytes: tx_payload.len(),
            tx_payload: &tx_payload,
            rx_table: &rx_table,
            rx_buffers: &rx_buffers,
            rx_buf_bytes: 64,
            data: None,
            conv: &mut conv,
        };
        assert!(matches!(
            core.exchange(&mut rings, Command::PatchSem, &[0u8; 4], || {}, || None),
            Err(D1Refusal::CommandTooLong { .. })
        ));
        let published = DmaShared::new(&tx_table);
        for word in 0..4 {
            assert_eq!(
                published
                    .read_u32_at(Descriptor::slot_at(0) + word * 4)
                    .unwrap(),
                0,
                "nothing was published"
            );
        }
    }

    /// Every rung is distinct, so a photographed number names one repair.
    #[test]
    fn the_rungs_are_distinct() {
        let rungs = [
            D1Refusal::Unreachable { iova: 0, len: 0 }.rung(),
            D1Refusal::BaseNotLatched { wrote: 0, read: 0 }.rung(),
            D1Refusal::SizeNotLatched { wrote: 0, read: 0 }.rung(),
            D1Refusal::EngineBusy { glo_cfg: 0 }.rung(),
            D1Refusal::RxNotTaken { posted: 0, dma_idx: 0 }.rung(),
            D1Refusal::CommandTooLong { bytes: 0, room: 0 }.rung(),
            D1Refusal::NoAnswer { seq: 0, dma_idx: 0, tx_idx: 0, tx_next: 0, rx_next: 0, skipped: 0, data_pending: None }.rung(),
            D1Refusal::BadAnswer(McuError::TooShort).rung(),
        ];
        let mut seen = rungs;
        seen.sort_unstable();
        assert!(seen.windows(2).all(|w| w[0] != w[1]));
        assert!(rungs.iter().all(|r| *r != 0));
    }
}

/// The two rings one MCU exchange rides, as plain memory.
///
/// The driver crate does no pointer arithmetic over DMA backings — the
/// binding owns those and hands them here as slices, so everything below is
/// ordinary indexing a test can build. `*_iova` are the device addresses of
/// the matching slices; they are what the descriptors carry.
pub struct Rings<'b, D: Dma + ?Sized> {
    /// The command ring's descriptor table.
    pub tx_table: &'b D,
    /// Where a command's bytes are staged. Its own `iova()` is the address
    /// the descriptor carries, so there is no way to name a buffer and then
    /// hand the part a different address for it.
    pub tx_payload: &'b D,
    /// The **firmware download** ring's descriptor table — `Fwdl`'s own.
    /// See [`Rings::bank`] for why it cannot share `tx_table`.
    pub fwdl_table: &'b D,
    /// The firmware download's staging slab — **one buffer per ring slot**.
    ///
    /// Separate from [`Self::tx_payload`], and sliced per slot, because a
    /// chunk is published and *not* waited for. A command is staged, sent,
    /// and answered before the next one is staged, so one buffer serves it;
    /// forty-six chunks are published back to back, and a single buffer
    /// would have every descriptor naming the same address while the bytes
    /// at it were overwritten forty-five times. The part would then read
    /// the last chunk forty-six times over.
    ///
    /// Sized by the caller: `Fwdl.depth()` buffers of
    /// [`Self::fwdl_chunk_bytes`] each, so the slab and the ring wrap
    /// together and slot *n* always owns the same bytes.
    pub fwdl_payload: &'b D,
    /// Bytes per download staging buffer — the slab's slot stride.
    pub fwdl_chunk_bytes: usize,
    /// The event ring's descriptor table.
    pub rx_table: &'b D,
    /// The event ring's receive buffers, one `rx_buf_bytes` slab per slot.
    pub rx_buffers: &'b D,
    /// Bytes per receive slot. Both receive rings use the same stride.
    pub rx_buf_bytes: usize,
    /// **Receive ring 2**, when the caller programmed it.
    ///
    /// `None` for an exchange over ring 0 alone — every test written
    /// before M1's third photograph, and any caller that never gave the
    /// engine this ring. See [`Queue::EventData`] for why it exists.
    pub data: Option<DataRing<'b, D>>,
    /// **The conversation with the MCU so far**, borrowed rather than
    /// owned.
    ///
    /// These counters used to be two `u32` fields of this struct, and
    /// every one of their hazards was written down here — and then a
    /// second caller built a second `Rings` over the same hardware with
    /// both set to zero, which is how C2b spent a flash replaying D1c's
    /// answers to itself. A `&mut` borrow gives the state exactly one
    /// owner and makes that construction unwritable.
    pub conv: &'b mut Conversation,
}

/// Receive ring 2's two backings, as plain memory.
///
/// Its stride is [`Rings::rx_buf_bytes`] and its depth
/// [`Queue::EventData`]'s; its cursor lives in [`Conversation::data_next`]
/// for the reason every cursor does.
pub struct DataRing<'b, D: Dma + ?Sized> {
    /// The ring's descriptor table.
    pub table: &'b D,
    /// Its receive buffers, one `rx_buf_bytes` slab per slot.
    pub buffers: &'b D,
}

// By hand: a derive would demand `D: Copy`, and `D` is unsized.
impl<D: Dma + ?Sized> Clone for DataRing<'_, D> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<D: Dma + ?Sized> Copy for DataRing<'_, D> {}

/// **What the host and the MCU have said to each other so far.**
///
/// Three numbers that belong to *the part*, not to any one caller: the
/// command ring's producer index, the event ring's consumer cursor, and
/// the sequence stamped into each command. Each outlives the `Rings` that
/// carries it, and all three must advance together or an exchange reads
/// someone else's answer.
///
/// **This exists because all three were reset at once.** `ask_the_part`
/// left the hardware at `cpu_idx = 2` with two answers sitting done in the
/// event ring; `download_patch` then started a fresh conversation at zero,
/// drove the producer index *backwards*, and read those two answers back
/// as replies to its own first two commands — which their sequence
/// numbers matched, because the sequence had restarted too. Two commands
/// the part never saw were reported as answered.
pub struct Conversation {
    /// How many commands have been published on the command ring.
    ///
    /// **The doorbell is an absolute producer index.** `exchange` used to
    /// write `cpu_idx = 1` every time, which is correct for the first
    /// command and a no-op for every one after it. Worse than a no-op
    /// across callers: a second caller starting at zero writes an index
    /// *below* the engine's own, which is the stale-pointer hazard
    /// `enable_engine` resets `RST_DTX_PTR` to avoid.
    pub tx_next: u32,
    /// Which event-ring slot the next answer will land in.
    ///
    /// The engine fills slots in order and sets the done bit in each. It
    /// does not clear the previous one, so an exchange that polled slot
    /// zero again would read the *first* answer's completion immediately
    /// and parse a stale response. [`Core::exchange`] now also clears the
    /// bit it consumed, so a cursor that does desync refuses rather than
    /// lies.
    pub rx_next: u32,
    /// The sequence stamped into the next command.
    ///
    /// Carried here, and stamped by [`Core::exchange`] itself, because a
    /// sequence that restarts is what lets a stale answer match: the
    /// check that is supposed to catch a desync agrees with it instead.
    pub seq: mcu::Sequence,
    /// How many firmware chunks have been published on the `Fwdl` ring.
    ///
    /// **The same hazard as [`Self::tx_next`], one queue over, and it was
    /// still a local when C2b went to metal.** `send_patch_sections` kept
    /// `let mut producer = 0` — correct for one download beside an engine
    /// `enable_engine` had just zeroed, and wrong the instant a second
    /// download exists. C2c is that second download: it would have written
    /// `cpu_idx = 1` while the engine's `dma_idx` sat at forty-six, driving
    /// the doorbell *backwards*.
    ///
    /// Worse, it would not have refused. [`Core::send_payload`]'s lapping
    /// guard asks whether `engine_index + depth >= producer`, which a
    /// producer that has just restarted satisfies trivially — so the
    /// download would have overwritten slots the engine still held and
    /// reported success. Carried here for the same reason as the sequence:
    /// a counter the hardware owns cannot be a local of whoever happens to
    /// be driving it.
    pub fwdl_next: u32,
    /// How many firmware chunks the **engine** has consumed, absolute.
    ///
    /// **`dma_idx` is a ring index, not a count.** `program_queue` tells
    /// the engine `ring_size = depth` and upstream reads the register as
    /// `q->tail`, an index into that ring. So it wraps at `depth` and
    /// cannot be compared with [`Self::fwdl_next`], which never wraps.
    ///
    /// That comparison is what C2c's first flash found on 2026-11-12:
    /// `send_payload` asked whether `dma_idx + depth >= producer`, which
    /// at `dma_idx = 125` and `producer = 159` is `253 >= 159` — true for
    /// any producer the ring could reach. The one guard protecting a
    /// lapped ring could not fire once the ring had lapped.
    ///
    /// Recovered here instead: the engine cannot advance by a whole lap
    /// between two publishes, so the absolute count is the unique value
    /// congruent to the register modulo `depth` that lies at or after the
    /// last one read. Monotone by construction, and the subtraction
    /// against `fwdl_next` means what it says again.
    pub fwdl_taken: u32,
    /// Which slot of receive ring 2 the next event will land in.
    ///
    /// [`Self::rx_next`]'s twin, one ring over. Absolute, like it.
    pub data_next: u32,
    /// **How many exchanges were answered on ring 2.**
    ///
    /// The question the ring was programmed to answer: does the RAM
    /// firmware reply somewhere other than where the ROM did? Counted,
    /// not inferred, so a boot that succeeds can still say which ring it
    /// succeeded on.
    pub data_answers: u32,
}

impl Conversation {
    /// A part that has been asked nothing yet.
    ///
    /// **Correct exactly once per engine start.** The hardware's own
    /// indices are zeroed by `enable_engine`, and nothing after that may
    /// begin a second conversation at zero.
    pub const fn new() -> Self {
        Self {
            tx_next: 0,
            rx_next: 0,
            seq: mcu::Sequence::new(),
            fwdl_next: 0,
            fwdl_taken: 0,
            data_next: 0,
            data_answers: 0,
        }
    }
}

impl Default for Conversation {
    fn default() -> Self {
        Self::new()
    }
}

impl<'b, D: Dma + ?Sized> Rings<'b, D> {
    /// The descriptor table `queue` was programmed with.
    ///
    /// **The pairing lives here and nowhere else.** The download drives
    /// two queues — commands on `McuWm`, firmware chunks on `Fwdl` — and
    /// each has its own descriptor base programmed into the engine.
    /// Writing a descriptor into one queue's table while ringing the
    /// other's doorbell is invisible at the call site and silent on the
    /// wire: the engine reads the base it was actually given, finds
    /// zeros, and stops. That is a stall at chunk zero, and it cost a
    /// boot.
    pub fn bank(&self, queue: Queue) -> &'b D {
        match queue {
            Queue::Fwdl => self.fwdl_table,
            _ => self.tx_table,
        }
    }

    /// Where `queue`'s slot `slot` stages its bytes.
    ///
    /// The companion of [`Self::bank`], and here for the same reason: a
    /// descriptor's address and the bytes it names must be decided in one
    /// place, or they drift apart silently. `Fwdl` gets its own slot of the
    /// download slab; every other queue stages at the front of the command
    /// buffer, which is correct for it because a command is answered before
    /// the next is written.
    pub fn staging(&self, queue: Queue, slot: usize) -> Staging<'b, D> {
        match queue {
            Queue::Fwdl => Staging {
                bank: self.fwdl_payload,
                at: slot * self.fwdl_chunk_bytes,
                room: self.fwdl_chunk_bytes,
            },
            _ => Staging {
                bank: self.tx_payload,
                at: 0,
                room: self.tx_payload.len(),
            },
        }
    }
}

/// One slot's worth of staging: the buffer, the offset into it, and how
/// many bytes that slot owns.
pub struct Staging<'b, D: Dma + ?Sized> {
    /// The buffer the bytes go in.
    pub bank: &'b D,
    /// The byte offset of this slot inside it.
    pub at: usize,
    /// How many bytes this slot may hold.
    pub room: usize,
}


impl<'r, R: Regs + ?Sized> Core<'r, R> {
    /// Send one command and read its answer — **D1c**.
    ///
    /// The round trip the whole download rides on, and the first exchange
    /// in which the part does something only firmware can do. The ROM
    /// answers MCU commands before any blob is loaded, which is why
    /// upstream's first download step is a semaphore request
    /// (`mt76_connac2_load_patch`, `mt76_connac_mcu.c:3330`).
    ///
    /// `slot` is which transmit descriptor to use; the caller owns the
    /// producer index and passes the slot it is about to publish.
    pub fn exchange<D: Dma + ?Sized>(
        &self,
        rings: &mut Rings<'_, D>,
        cmd: Command,
        body: &[u8],
        mut delay: impl FnMut(),
        now_ns: fn() -> Option<u64>,
    ) -> Result<Response, D1Refusal> {
        // **The sequence is stamped here, not by the caller.** A caller
        // that owns its own counter can restart it, and a restarted
        // sequence makes a stale answer validate.
        let seq = rings.conv.seq.next();
        let txd = mcu::command(cmd, seq, body.len() as u16);
        // `FW_SCATTER` rides `Fwdl`; every other command rides `McuWm`
        // (`mt7925/pci_mcu.c:21-24`).
        let queue = if matches!(cmd, Command::FwScatter) {
            Queue::Fwdl
        } else {
            Queue::McuWm
        };
        self.exchange_framed(rings, &txd, body, seq, queue, None, MCU_WAIT, delay, now_ns)
            .map(|(response, _)| response)
    }

    /// **Tell the ROM a download is about to begin** — upstream's first
    /// MCU command, which this driver never sent.
    ///
    /// `mt792x_load_firmware` opens with `mt76_connac_mcu_restart`
    /// (`mt792x_core.c:1101`): `NIC_POWER_CTRL` with a power-mode byte of
    /// 1, sent without waiting, then a poll of `MT_CONN_ON_MISC` for the
    /// power-on state that only *warns* when it does not come
    /// (`:1103-1106`). The 2026-11-20 capture of `mt7925e` on this laptop
    /// has it as the first command after the engine starts, ahead of the
    /// patch semaphore — and it is the **only** command in that capture,
    /// up to and including the capability query, that this driver did
    /// not send. Every other one matches in identity, order and body.
    ///
    /// The ROM accepted a whole download without it, so whatever it sets
    /// up is not something the download needs. M1b is the first rung
    /// that asks the RAM firmware a question, and the first that went
    /// unanswered.
    ///
    /// **Not answered is not refused**, as upstream treats it. What is
    /// refused is a command the engine never fetched: the staging buffer
    /// is shared, so the next command would overwrite this one while it
    /// is still pending. Every other transport refusal passes through.
    pub fn restart<D: Dma + ?Sized>(
        &self,
        rings: &mut Rings<'_, D>,
        mut delay: impl FnMut(),
        now_ns: fn() -> Option<u64>,
    ) -> Result<Restarted, D1Refusal> {
        let seq = rings.conv.seq.next();
        let body = mcu::restart_body();
        let txd = mcu::command(Command::NicPowerCtrl, seq, body.len() as u16);
        let answered = match self.exchange_framed(
            rings,
            &txd,
            &body,
            seq,
            Queue::McuWm,
            None,
            (RESTART_BUDGET_NS, RESTART_SPINS_UNCLOCKED),
            &mut delay,
            now_ns,
        ) {
            // An answer this driver cannot parse is still the ROM
            // speaking, and upstream never reads it either.
            Ok(_) | Err(D1Refusal::BadAnswer(_)) => true,
            Err(D1Refusal::NoAnswer { tx_idx, tx_next, .. })
                if tx_idx == tx_next % u32::from(Queue::McuWm.depth()) =>
            {
                false
            }
            Err(other) => return Err(other),
        };
        let mut budget = Budget::new(POWER_ON_BUDGET_NS, POWER_ON_SPINS_UNCLOCKED, now_ns);
        let powered = loop {
            if self.powered_on() {
                break true;
            }
            if !budget.may_wait() {
                break false;
            }
            delay();
        };
        Ok(Restarted { answered, powered })
    }

    /// Send one **unified** command and wait for its answer.
    ///
    /// The other descriptor class — see [`mcu::MCU_UNI_TXD_BYTES`] for why
    /// it is a separate encoder rather than a flag. Everything after the
    /// descriptor is shared with [`Self::exchange`]: the staging, the
    /// doorbell, the triage and the parse all live in
    /// [`Self::exchange_framed`], because a second copy of the triage is
    /// how two paths come to disagree about which reply was theirs.
    ///
    /// Always `McuWm`: `FW_SCATTER` is the only command that rides
    /// `Fwdl`, and it is not a unified command.
    pub fn exchange_uni<D: Dma + ?Sized>(
        &self,
        rings: &mut Rings<'_, D>,
        cmd: mcu::UniCommand,
        body: &[u8],
        query: bool,
        delay: impl FnMut(),
        now_ns: fn() -> Option<u64>,
    ) -> Result<Response, D1Refusal> {
        let seq = rings.conv.seq.next();
        let txd = mcu::uni_command(cmd, seq, body.len() as u16, query);
        self.exchange_framed(rings, &txd, body, seq, Queue::McuWm, None, MCU_WAIT, delay, now_ns)
            .map(|(response, _)| response)
    }

    /// Send one unified command and keep its reply's body.
    ///
    /// [`Self::exchange_uni`] with the bytes behind the header, for the
    /// commands whose answer *is* that body — `CHIP_CONFIG` and
    /// `EFUSE_CTRL` both take `mt7925_mcu_parse_response`'s plain
    /// `skb_pull(skb, sizeof(*rxd))` arm and carry no status byte, so
    /// there is nothing to read in the header but the sequence that
    /// proved the answer was ours.
    ///
    /// Returns how many bytes landed in `out`.
    pub fn exchange_uni_into<D: Dma + ?Sized>(
        &self,
        rings: &mut Rings<'_, D>,
        cmd: mcu::UniCommand,
        body: &[u8],
        query: bool,
        out: &mut [u8],
        delay: impl FnMut(),
        now_ns: fn() -> Option<u64>,
    ) -> Result<(Response, usize), D1Refusal> {
        let seq = rings.conv.seq.next();
        let txd = mcu::uni_command(cmd, seq, body.len() as u16, query);
        self.exchange_framed(rings, &txd, body, seq, Queue::McuWm, Some(out), MCU_WAIT, delay, now_ns)
    }

    /// Stage a descriptor and its body, ring the doorbell, and read back
    /// the answer that belongs to `seq`.
    ///
    /// **Everything both command classes share.** The two differ only in
    /// the bytes of `txd` and in which queue carries them; the sequence is
    /// stamped by the caller above, because only the caller knows which
    /// encoder framed it.
    fn exchange_framed<D: Dma + ?Sized>(
        &self,
        rings: &mut Rings<'_, D>,
        txd: &[u8],
        body: &[u8],
        seq: u8,
        queue: Queue,
        out: Option<&mut [u8]>,
        (budget_ns, spins): (u64, u32),
        mut delay: impl FnMut(),
        now_ns: fn() -> Option<u64>,
    ) -> Result<(Response, usize), D1Refusal> {
        let total = txd.len() + body.len();
        if total > rings.tx_payload.len() {
            return Err(D1Refusal::CommandTooLong {
                bytes: total,
                room: rings.tx_payload.len(),
            });
        }
        // **Bulk, and deliberately not volatile.** `DmaShared`'s own
        // contract draws this line: control words are volatile, payloads
        // are plain copies published by one of those control words plus the
        // caller's fence. The fence is below, before the doorbell.
        let geometry = |_| D1Refusal::RingTooSmall {
            need: total,
            have: rings.tx_payload.len(),
        };
        rings.tx_payload.write_bytes_at(0, txd).map_err(geometry)?;
        rings
            .tx_payload
            .write_bytes_at(txd.len(), body)
            .map_err(geometry)?;

        let iova = rings.tx_payload.iova().0;
        let Some(desc) = Descriptor::transmit(iova, total as u32) else {
            return Err(D1Refusal::Unreachable {
                iova,
                len: total as u64,
            });
        };

        // **Volatile, word by word.** The table is memory the engine reads
        // on its own schedule; a plain store into it may be sunk, merged,
        // or reordered against the doorbell below, none of which the engine
        // would forgive.
        // The slot this command occupies, and the index the doorbell
        // will advertise. Both move; see `Rings::tx_next`.
        let depth = queue.depth() as u32;
        let slot = (rings.conv.tx_next % depth) as usize;
        // The table this queue's descriptor base actually points at.
        let tx = DmaShared::new(rings.bank(queue));
        for (word, value) in desc.words().into_iter().enumerate() {
            tx.write_u32_at(Descriptor::slot_at(slot) + word * 4, value)
                .map_err(|_| D1Refusal::RingTooSmall {
                    need: dma::DESC_BYTES,
                    have: rings.tx_table.len(),
                })?;
        }
        // **The descriptor is published before the doorbell rings.**
        // `mt76_dma_kick_queue` has the same barrier, and for the same
        // reason (`dma.c:232-235`): the engine may fetch the instant
        // `cpu_idx` moves, and must not fetch a half-written descriptor.
        fence(Ordering::Release);
        rings.conv.tx_next += 1;
        // **The doorbell names a slot, not a count.** No command ring has
        // reached its 256 descriptors, so this is a no-op today — and it
        // is the same expression that broke the 128-deep download ring at
        // its 128th chunk, so it is written the one correct way in both
        // places rather than left to be found twice.
        self.post_receive(queue, (rings.conv.tx_next % u32::from(queue.depth())) as u16);

        // The answer lands in an event ring slot, and the engine marks it
        // done. Budgeted, never spun.
        let mut budget = Budget::new(budget_ns, spins, now_ns);
        // How many done slots the triage below handed to somebody else.
        let mut skipped: u8 = 0;
        // **Not every done descriptor is this exchange's answer**, and the
        // two ways it can belong to somebody else are different facts.
        //
        // The event ring is shared with the part's unsolicited traffic —
        // firmware logs, coredumps, HIF-control notices — which
        // `mt7925_mcu_rx_event` routes on `option & UNSOLICITED_EVENT`
        // *before* it looks at a sequence. And a solicited reply may still
        // not be ours: upstream discards a sequence that does not match
        // and keeps reading until its deadline, which is what its
        // `while (ret == -EAGAIN)` loop is for.
        //
        // So a slot that is not ours is **consumed** — done bit cleared,
        // cursor advanced — and the poll moves to the next one inside the
        // same budget. Leaving it would read it again forever; refusing on
        // it would name this exchange for another's message.
        //
        // **Both receive rings, when the caller gave two.** Upstream's
        // receive path takes an event from whichever ring it arrives on
        // (`mt7925_queue_rx_skb`, `mt7925/mac.c:1451`), and the RAM
        // firmware was silent on ring 0 for its first unified command.
        let data = rings.data;
        let (answer_slot, answer, on_data) = loop {
            match self.poll_ring(
                rings.rx_table,
                rings.rx_buffers,
                rings.rx_buf_bytes,
                rings.conv.rx_next,
                Queue::EventWm,
                seq,
            )? {
                Polled::Ours(slot, desc) => break (slot, desc, false),
                Polled::Theirs(slot, desc) => {
                    // Consume it exactly as the accepted path does, so the
                    // next poll reads the next slot rather than this one.
                    rings.conv.rx_next += 1;
                    skipped = skipped.saturating_add(1);
                    let _ = DmaShared::new(rings.rx_table)
                        .write_u32_at(Descriptor::ctrl_at(slot), desc.consumed());
                    continue;
                }
                Polled::Empty => {}
            }
            if let Some(data) = data.as_ref() {
                match self.poll_ring(
                    data.table,
                    data.buffers,
                    rings.rx_buf_bytes,
                    rings.conv.data_next,
                    Queue::EventData,
                    seq,
                )? {
                    Polled::Ours(slot, desc) => break (slot, desc, true),
                    Polled::Theirs(slot, _) => {
                        rings.conv.data_next += 1;
                        skipped = skipped.saturating_add(1);
                        self.recycle_data_slot(data, rings.rx_buf_bytes, slot, rings.conv.data_next);
                        continue;
                    }
                    Polled::Empty => {}
                }
            }
            if !budget.may_wait() {
                let data_pending = data.map(|_| {
                    let depth = u32::from(Queue::EventData.depth());
                    self.engine_index(Queue::EventData)
                        .wrapping_sub(rings.conv.data_next % depth)
                        % depth
                });
                return Err(D1Refusal::NoAnswer {
                    seq,
                    dma_idx: self.engine_index(Queue::EventWm),
                    tx_idx: self.engine_index(queue),
                    tx_next: rings.conv.tx_next,
                    rx_next: rings.conv.rx_next,
                    skipped,
                    data_pending,
                });
            }
            delay();
        };
        // The done bit published the buffer; this orders the read of it
        // after the read of the bit.
        fence(Ordering::Acquire);
        // Which ring's buffers hold the answer.
        let (answer_buffers, answer_table) = match (on_data, data.as_ref()) {
            (true, Some(data)) => (data.buffers, data.table),
            _ => (rings.rx_buffers, rings.rx_table),
        };
        if on_data {
            rings.conv.data_next += 1;
            rings.conv.data_answers += 1;
        } else {
            rings.conv.rx_next += 1;
        }
        // **The answer is consumed, so its done bit goes with it.**
        // The engine sets this bit and never clears it, which made a
        // stale slot indistinguishable from a fresh answer: whoever
        // polled it next read a completion that had already been read.
        // Clearing it costs one write and turns that silent replay into
        // a `NoAnswer` — a rung that refuses, rather than a rung that
        // reports someone else's reply as its own.
        //
        // Written before the length is read out of `answer`, which is a
        // copy of the word taken above, so nothing below depends on the
        // descriptor still holding it. Ring 2's slot is re-armed and
        // returned to the engine *after* the bytes are copied out below.
        let _ = DmaShared::new(answer_table)
            .write_u32_at(Descriptor::ctrl_at(answer_slot), answer.consumed());

        let written = answer.written() as usize;
        let end = written.min(rings.rx_buf_bytes);
        // **The answer is in that slot's own buffer**, not in the first.
        // One contiguous slab holds `depth` buffers of `rx_buf_bytes`.
        let at = answer_slot * rings.rx_buf_bytes;
        let ptr = answer_buffers
            .checked_read_ptr(at, end)
            .map_err(|_| D1Refusal::RingTooSmall {
                need: end,
                have: answer_buffers.len(),
            })?;
        // SAFETY: `checked_read_ptr` proved `end` readable bytes at offset
        // zero of this buffer, and the acquire fence above orders them
        // after the done bit that published them. The slice does not
        // outlive this call.
        let bytes = unsafe { core::slice::from_raw_parts(ptr, end) };
        // **Read out, then released.** Everything below copies out of the
        // slot, and ring 2's slot goes back to the engine only once the
        // copy is done — on every arm, including the refusals.
        let read_out = (move || {
            let response = Response::parse(bytes, seq).map_err(D1Refusal::BadAnswer)?;

            // **The body, for the callers that have one.**
            //
            // Every C2 command's answer was its header: a semaphore verdict
            // and a finish status both live at offset 40, inside the
            // forty-four bytes `Response` reads. From M1 the interesting part
            // is behind it — `mt7925_mcu_parse_response` pulls exactly
            // `sizeof(*rxd)` for `CHIP_CONFIG` and `EFUSE_CTRL` and hands the
            // rest to the caller (`mt7925/mcu.c:69-71`).
            //
            // Copied rather than borrowed: these bytes live in a ring slot
            // this exchange is about to release, and a borrow would outlive
            // the only thing keeping them still. The caller's buffer bounds
            // the copy, and a reply too large for it refuses with both
            // numbers — truncating would hand a parser a short body and make
            // it report the wrong fault.
            // **`None` is "I do not want it", not "I have no room".** Every
            // C2 caller's answer is its header, and reading the body for them
            // would make a 2 KiB slot's worth of slack look like a reply too
            // large to hold — which is what refusing on an empty buffer did.
            // The two are different facts and only one of them is a fault.
            let Some(out) = out else {
                return Ok((response, 0));
            };
            let payload = bytes.get(mcu::MCU_RXD_BYTES..).unwrap_or(&[]);
            if payload.len() > out.len() {
                return Err(D1Refusal::RingTooSmall {
                    need: payload.len(),
                    have: out.len(),
                });
            }
            out[..payload.len()].copy_from_slice(payload);
            Ok((response, payload.len()))
        })();
        if on_data {
            if let Some(data) = data.as_ref() {
                self.recycle_data_slot(data, rings.rx_buf_bytes, answer_slot, rings.conv.data_next);
            }
        }
        read_out
    }
}
