//! **The MAC** — power on and off, and the DMA engine's pre-init: the
//! sequences `rtw89_mac_partial_init` runs before firmware
//! (`rtw89_mac_dmac_pre_init` and `rtw89_pci_ops_mac_pre_init_be`), split
//! at the one step that lets the device fetch from host memory, so the
//! generic host's `DmaPrepared` → `DmaActive` transition is honoured.
//!
//! The power sequences are **data**: [`POWER_ON`] and [`POWER_OFF`] are
//! the 8922A's `pwr_on_func` / `pwr_off_func` transcribed one operation
//! per line, so a reviewer diffs them against rtw8922a.c:267-411 and
//! :413-500 rather than against control flow, and the host mock records
//! exactly the operations the device saw.

use kore_driver::seam::RegsNarrow;

use crate::Waits;
pub mod normal;
use crate::regs::be::*;
use crate::regs::{
    CHIP_CAV, CMAC_FUNC_EN_ALL, CMAC_SHARE_FUNC_EN_ALL, DLE_BOUND_UNIT, DLE_CLK_EN, DLE_FUNC_EN,
    DMA_BUSY1_CHECK_BE, DMAC_FUNC_EN_ALL, PLE_MGN_INI_RDY, PWR_ACT, RPQ0, RX_CHANNELS, RX_DMA_BUSY,
    RXBD_CLR_ALL, RXQ0, TX_CHANNELS, TX_STOP_ALL, TX_STOP_DATA, TX_STOP_UNUSED, TXBD_CLR_ALL,
    TxChannelRegs, WDE_MGN_INI_RDY, encode, get, reg, replace, xtal_si_write,
};

/// One operation of a MAC register sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// `rtw89_write32_set`.
    Set32(u32, u32),
    /// `rtw89_write32_clr`.
    Clr32(u32, u32),
    /// `rtw89_write8_set`.
    Set8(u32, u8),
    /// `rtw89_write8_clr`.
    Clr8(u32, u8),
    /// `rtw89_write32`.
    Write32(u32, u32),
    /// Replace a packed field, preserving every bit outside its mask.
    Field32(u32, u32, u32),
    /// Halfword operations must not touch adjacent device registers.
    Write16(u32, u16),
    Set16(u32, u16),
    Field16(u32, u16, u16),
    PollClear16(u32, u16, u64),
    /// `read_poll_timeout` until `reg & mask != 0`, within the budget.
    PollSet(u32, u32, u64),
    /// `read_poll_timeout` until `reg & mask == 0`, within the budget.
    PollClear(u32, u32, u64),
    /// `rtw89_mac_write_xtal_si(offset, val, mask)`.
    XtalSi(u8, u8, u8),
    /// `mdelay`, in nanoseconds.
    Delay(u64),
    /// The A-die patch: run only when the cut is not the A-cut
    /// (`hal->cv != CHIP_CAV`).
    CutPatch(&'static [Op]),
}

/// Why a sequence or the DMA pre-init refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MacError {
    /// A MAC-sequence poll expired: the operation index in the
    /// sequence, so the refusal names the step.
    PowerPoll(u8),
    /// A crystal-interface write never completed.
    XtalSi,
    /// A DLE engine never reported ready.
    DleNotReady,
    /// A DMA channel stayed busy.
    DmaBusy,
    /// The BD-RAM bound reset never completed.
    BdramBound,
}

/// The polls' budget in `rtw8922a_pwr_on_func` / `_off_func`:
/// `read_poll_timeout(…, 1000, 3000000, …)` — 3 s.
const PWR_POLL_NS: u64 = 3_000_000_000;
/// The crystal interface's completion budget (`mac_be.c:390-391`): 50 ms.
const XTAL_SI_NS: u64 = 50_000_000;
/// `mdelay(1)` in the A-die patch.
const MS: u64 = 1_000_000;
/// `chk_dle_rdy_be` (mac_be.c:281): 2 ms.
const DLE_READY_NS: u64 = 2_000_000;
/// `rtw89_pci_poll_txdma_ch_idle_be` / `_rxdma_` (pci_be.c:189, :198):
/// 1 ms.
const DMA_IDLE_NS: u64 = 1_000_000;
/// `rtw89_pci_rst_bdram_be` (pci_be.c:264): 0.5 s.
const BDRAM_NS: u64 = 500_000_000;

/// `rtw8922a_pwr_on_func` (rtw8922a.c:267-411), the dossier's §B.5 table.
pub const POWER_ON: &[Op] = &[
    Op::Clr32(R_BE_SYS_PW_CTRL, B_BE_AFSM_WLSUS_EN | B_BE_AFSM_PCIE_SUS_EN),
    Op::Set32(R_BE_SYS_PW_CTRL, B_BE_DIS_WLBT_PDNSUSEN_SOPC),
    Op::Set32(R_BE_WLLPS_CTRL, B_BE_DIS_WLBT_LPSEN_LOPC),
    Op::Clr32(R_BE_SYS_PW_CTRL, B_BE_APDM_HPDN),
    Op::Clr32(R_BE_SYS_PW_CTRL, B_BE_APFM_SWLPS),
    Op::PollSet(R_BE_SYS_PW_CTRL, B_BE_RDY_SYSPWR, PWR_POLL_NS),
    Op::Set32(R_BE_SYS_PW_CTRL, B_BE_EN_WLON),
    Op::Set32(R_BE_WLRESUME_CTRL, B_BE_LPSROP_CMAC0 | B_BE_LPSROP_CMAC1),
    Op::Set32(R_BE_SYS_PW_CTRL, B_BE_APFN_ONMAC),
    Op::PollClear(R_BE_SYS_PW_CTRL, B_BE_APFN_ONMAC, PWR_POLL_NS),
    Op::Clr32(R_BE_AFE_ON_CTRL1, B_BE_REG_CK_MON_CK960M_EN),
    Op::Set8(
        R_BE_ANAPAR_POW_MAC,
        (B_BE_POW_PC_LDO_PORT0 | B_BE_POW_PC_LDO_PORT1) as u8,
    ),
    Op::Clr32(
        R_BE_FEN_RST_ENABLE,
        B_BE_R_SYM_ISO_ADDA_P02PP | B_BE_R_SYM_ISO_ADDA_P12PP,
    ),
    Op::Set8(R_BE_PLATFORM_ENABLE, B_BE_PLATFORM_EN as u8),
    Op::Set32(R_BE_HCI_OPT_CTRL, B_BE_HAXIDMA_IO_EN),
    Op::PollSet(R_BE_HCI_OPT_CTRL, B_BE_HAXIDMA_IO_ST, PWR_POLL_NS),
    Op::PollClear(
        R_BE_HCI_OPT_CTRL,
        B_BE_HAXIDMA_BACKUP_RESTORE_ST,
        PWR_POLL_NS,
    ),
    Op::Set32(R_BE_HCI_OPT_CTRL, B_BE_HCI_WLAN_IO_EN),
    Op::PollSet(R_BE_HCI_OPT_CTRL, B_BE_HCI_WLAN_IO_ST, PWR_POLL_NS),
    Op::Clr32(R_BE_SYS_SDIO_CTRL, B_BE_PCIE_FORCE_IBX_EN),
    Op::XtalSi(XTAL_SI_PLL as u8, 0x02, 0x02),
    Op::XtalSi(XTAL_SI_PLL as u8, 0x01, 0x01),
    Op::Set32(R_BE_SYS_ADIE_PAD_PWR_CTRL, B_BE_SYM_PADPDN_WL_RFC1_1P3),
    Op::XtalSi(XTAL_SI_ANAPAR_WL as u8, 0x40, 0x40),
    Op::Set32(R_BE_SYS_ADIE_PAD_PWR_CTRL, B_BE_SYM_PADPDN_WL_RFC0_1P3),
    Op::XtalSi(XTAL_SI_ANAPAR_WL as u8, 0x20, 0x20),
    Op::XtalSi(XTAL_SI_ANAPAR_WL as u8, 0x04, 0x04),
    Op::XtalSi(XTAL_SI_ANAPAR_WL as u8, 0x08, 0x08),
    Op::XtalSi(XTAL_SI_ANAPAR_WL as u8, 0x00, 0x10),
    Op::XtalSi(XTAL_SI_WL_RFC_S0 as u8, 0xEB, 0xFF),
    Op::XtalSi(XTAL_SI_WL_RFC_S1 as u8, 0xEB, 0xFF),
    Op::XtalSi(XTAL_SI_ANAPAR_WL as u8, 0x01, 0x01),
    Op::XtalSi(XTAL_SI_ANAPAR_WL as u8, 0x02, 0x02),
    Op::XtalSi(XTAL_SI_ANAPAR_WL as u8, 0x00, 0x80),
    Op::XtalSi(XTAL_SI_XREF_RF1 as u8, 0x00, 0x40),
    Op::XtalSi(XTAL_SI_XREF_RF2 as u8, 0x00, 0x40),
    Op::XtalSi(XTAL_SI_PLL_1 as u8, 0x40, 0x60),
    Op::CutPatch(&[
        Op::Set32(R_BE_PMC_DBG_CTRL2, B_BE_SYSON_DIS_PMCR_BE_WRMSK),
        Op::Set32(R_BE_SYS_ISO_CTRL, B_BE_ISO_EB2CORE),
        Op::Clr32(R_BE_SYS_ISO_CTRL, B_BE_PWC_EV2EF_B),
        Op::Delay(MS),
        Op::Clr32(R_BE_SYS_ISO_CTRL, B_BE_PWC_EV2EF_S),
        Op::Clr32(R_BE_PMC_DBG_CTRL2, B_BE_SYSON_DIS_PMCR_BE_WRMSK),
    ]),
    Op::Set32(R_BE_DMAC_FUNC_EN, DMAC_FUNC_EN_ALL),
    Op::Set32(R_BE_CMAC_SHARE_FUNC_EN, CMAC_SHARE_FUNC_EN_ALL),
    Op::Set32(R_BE_CMAC_FUNC_EN, CMAC_FUNC_EN_ALL),
    Op::Set32(
        R_BE_FEN_RST_ENABLE,
        B_BE_FEN_BB_IP_RSTN | B_BE_FEN_BBPLAT_RSTB,
    ),
];

/// `rtw8922a_pwr_off_func` (rtw8922a.c:413-500).
pub const POWER_OFF: &[Op] = &[
    Op::XtalSi(XTAL_SI_ANAPAR_WL as u8, 0x10, 0x10),
    Op::XtalSi(XTAL_SI_ANAPAR_WL as u8, 0x00, 0x08),
    Op::XtalSi(XTAL_SI_ANAPAR_WL as u8, 0x00, 0x04),
    Op::XtalSi(XTAL_SI_WL_RFC_S0 as u8, 0xC6, 0xFF),
    Op::XtalSi(XTAL_SI_WL_RFC_S1 as u8, 0xC6, 0xFF),
    Op::XtalSi(XTAL_SI_ANAPAR_WL as u8, 0x80, 0x80),
    Op::XtalSi(XTAL_SI_ANAPAR_WL as u8, 0x00, 0x02),
    Op::XtalSi(XTAL_SI_ANAPAR_WL as u8, 0x00, 0x01),
    Op::XtalSi(XTAL_SI_PLL as u8, 0x02, 0xFF),
    Op::XtalSi(XTAL_SI_PLL as u8, 0x00, 0xFF),
    Op::Set32(
        R_BE_FEN_RST_ENABLE,
        B_BE_R_SYM_ISO_ADDA_P02PP | B_BE_R_SYM_ISO_ADDA_P12PP,
    ),
    Op::Clr8(
        R_BE_ANAPAR_POW_MAC,
        (B_BE_POW_PC_LDO_PORT0 | B_BE_POW_PC_LDO_PORT1) as u8,
    ),
    Op::Set32(R_BE_SYS_PW_CTRL, B_BE_EN_WLON),
    Op::Clr8(
        R_BE_FEN_RST_ENABLE,
        (B_BE_FEN_BB_IP_RSTN | B_BE_FEN_BBPLAT_RSTB) as u8,
    ),
    Op::Clr32(R_BE_SYS_ADIE_PAD_PWR_CTRL, B_BE_SYM_PADPDN_WL_RFC0_1P3),
    Op::XtalSi(XTAL_SI_ANAPAR_WL as u8, 0x00, 0x20),
    Op::Clr32(R_BE_SYS_ADIE_PAD_PWR_CTRL, B_BE_SYM_PADPDN_WL_RFC1_1P3),
    Op::XtalSi(XTAL_SI_ANAPAR_WL as u8, 0x00, 0x40),
    Op::Clr32(R_BE_HCI_OPT_CTRL, B_BE_HAXIDMA_IO_EN),
    Op::PollClear(R_BE_HCI_OPT_CTRL, B_BE_HAXIDMA_IO_ST, PWR_POLL_NS),
    Op::PollClear(
        R_BE_HCI_OPT_CTRL,
        B_BE_HAXIDMA_BACKUP_RESTORE_ST,
        PWR_POLL_NS,
    ),
    Op::Clr32(R_BE_HCI_OPT_CTRL, B_BE_HCI_WLAN_IO_EN),
    Op::PollClear(R_BE_HCI_OPT_CTRL, B_BE_HCI_WLAN_IO_ST, PWR_POLL_NS),
    Op::Set32(R_BE_SYS_PW_CTRL, B_BE_APFM_OFFMAC),
    Op::PollClear(R_BE_SYS_PW_CTRL, B_BE_APFM_OFFMAC, PWR_POLL_NS),
    Op::Write32(R_BE_WLLPS_CTRL, 0x0000_A1B2),
    Op::Set32(R_BE_SYS_PW_CTRL, B_BE_XTAL_OFF_A_DIE),
    Op::Set32(R_BE_SYS_PW_CTRL, B_BE_APFM_SWLPS),
    Op::Write32(R_BE_UDM1, 0),
];

/// Run one sequence.
pub fn run<R: RegsNarrow>(regs: &R, waits: &Waits, cv: u8, ops: &[Op]) -> Result<(), MacError> {
    for (i, op) in ops.iter().enumerate() {
        match *op {
            Op::Set32(r, bits) => {
                let v = regs.read32(reg(r));
                regs.write32(reg(r), v | bits);
            }
            Op::Clr32(r, bits) => {
                let v = regs.read32(reg(r));
                regs.write32(reg(r), v & !bits);
            }
            Op::Set8(r, bits) => {
                let v = regs.read8(reg(r));
                regs.write8(reg(r), v | bits);
            }
            Op::Clr8(r, bits) => {
                let v = regs.read8(reg(r));
                regs.write8(reg(r), v & !bits);
            }
            Op::Write32(r, val) => regs.write32(reg(r), val),
            Op::Field32(r, mask, val) => {
                regs.write32(reg(r), replace(regs.read32(reg(r)), mask, val));
            }
            Op::Write16(r, val) => regs.write16(reg(r), val),
            Op::Set16(r, bits) => regs.write16(reg(r), regs.read16(reg(r)) | bits),
            Op::Field16(r, mask, val) => {
                let old = regs.read16(reg(r));
                regs.write16(reg(r), replace(old.into(), mask.into(), val.into()) as u16);
            }
            Op::PollClear16(r, mask, ns) => {
                if !kore_driver::handshake_until(regs, waits.budget(ns), waits.pause, |x: &R| {
                    x.read16(reg(r)) & mask == 0
                })
                .is_acked()
                {
                    return Err(MacError::PowerPoll(i as u8));
                }
            }
            Op::PollSet(r, mask, ns) => {
                if !waits.poll_set(regs, r, mask, ns) {
                    return Err(MacError::PowerPoll(i as u8));
                }
            }
            Op::PollClear(r, mask, ns) => {
                if !waits.poll_clear(regs, r, mask, ns) {
                    return Err(MacError::PowerPoll(i as u8));
                }
            }
            Op::XtalSi(offset, val, mask) => write_xtal_si(regs, waits, offset, val, mask)?,
            Op::Delay(ns) => waits.delay(ns),
            Op::CutPatch(patch) => {
                if cv != CHIP_CAV {
                    run(regs, waits, cv, patch)?;
                }
            }
        }
    }
    Ok(())
}

/// `rtw89_mac_write_xtal_si_be` (mac_be.c:376-399): one command word,
/// then the poll bit clearing.
pub fn write_xtal_si<R: RegsNarrow>(
    regs: &R,
    waits: &Waits,
    offset: u8,
    val: u8,
    mask: u8,
) -> Result<(), MacError> {
    regs.write32(
        reg(R_BE_WLAN_XTAL_SI_CTRL),
        xtal_si_write(offset, val, mask),
    );
    if waits.poll_clear(
        regs,
        R_BE_WLAN_XTAL_SI_CTRL,
        B_BE_WL_XTAL_SI_CMD_POLL,
        XTAL_SI_NS,
    ) {
        Ok(())
    } else {
        Err(MacError::XtalSi)
    }
}

/// Whether the WLAN MAC reads powered (`R_BE_IC_PWR_STATE[9:8] ==
/// PWR_ACT`, the guard at mac.c:1483-1487).
pub fn is_powered<R: RegsNarrow>(regs: &R) -> bool {
    get(regs.read32(reg(R_BE_IC_PWR_STATE)), B_BE_WLMAC_PWR_STE_MASK) == PWR_ACT
}

/// `rtw89_mac_power_switch(true)` tail (mac.c:1497-1500): the scoreboard
/// byte that tells the firmware which host driver generation it faces.
pub fn notify_tp_major<R: RegsNarrow>(regs: &R) {
    regs.write8(reg(R_AX_SCOREBOARD) + 3, MAC_AX_NOTIFY_TP_MAJOR as u8);
}

/// `rtw89_mac_pwr_on` (mac.c:1516-1529) with the already-on guard folded
/// in: a MAC that reads powered is switched off first (upstream's
/// `-EBUSY` → off → on), and a sequence that refuses is retried once
/// after an off.
pub fn power_on<R: RegsNarrow>(regs: &R, waits: &Waits, cv: u8) -> Result<(), MacError> {
    if is_powered(regs) {
        run(regs, waits, cv, POWER_OFF)?;
    }
    if let Err(first) = run(regs, waits, cv, POWER_ON) {
        // The off sequence's own refusal is secondary: report the on
        // sequence's, which is the one the retry could not clear.
        let _ = run(regs, waits, cv, POWER_OFF);
        run(regs, waits, cv, POWER_ON).map_err(|_| first)?;
    }
    notify_tp_major(regs);
    Ok(())
}

/// `rtw89_mac_power_switch(false)`.
pub fn power_off<R: RegsNarrow>(regs: &R, waits: &Waits, cv: u8) -> Result<(), MacError> {
    run(regs, waits, cv, POWER_OFF)
}

// ── The DLE (data-link engine) quotas for firmware download ──────────────
//
// `rtw8922a_dle_mem_pcie[RTW89_QTA_DLFW]` (rtw8922a.c:71-75) resolved
// through `rtw89_mac_size` (mac.c:1646-1745): the memory carve the WCPU
// boots against. The dossier's §C.2 tables, and the size identity
// `rtw89_mac_dle_init` asserts (mac.c:2145) as a compile-time assertion.

/// `wde_size4_v1` (mac.c:1655): 64-byte pages, 0 linked, 3328 unlinked,
/// start offset 0.
const WDE_PAGE_SIZE: u32 = RTW89_WDE_PG_64;
const WDE_LNK_PAGES: u32 = 0;
const WDE_UNLNK_PAGES: u32 = 3328;
const WDE_START: u32 = 0;
/// `ple_size3_v1` (mac.c:1672): 128-byte pages, 2928 linked, 0 unlinked,
/// start offset 212992.
const PLE_PAGE_SIZE: u32 = RTW89_PLE_PG_128;
const PLE_LNK_PAGES: u32 = 2928;
const PLE_UNLNK_PAGES: u32 = 0;
const PLE_START: u32 = 212_992;
/// `rsvd1_size0` (mac.c:1745): 2048 bytes; `rsvd0_size0` is 0.
const RSVD1_SIZE: u32 = 2048;
/// `chip->fifo_size` (rtw8922a.c:2857): what the carve must sum to.
const FIFO_SIZE: u32 = 589_824;
const _: () = assert!(
    WDE_PAGE_SIZE * (WDE_LNK_PAGES + WDE_UNLNK_PAGES)
        + PLE_PAGE_SIZE * (PLE_LNK_PAGES + PLE_UNLNK_PAGES)
        + RSVD1_SIZE
        == FIFO_SIZE,
    "[ERR]wd/dle mem cfg"
);
const _: () = assert!(PLE_START == WDE_PAGE_SIZE * (WDE_LNK_PAGES + WDE_UNLNK_PAGES));

/// `wde_qt4` (mac.c:1692) is all zero; the WCPU quota is overridden by
/// the SCC table's `wde_qt0_v1.wcpu` = 6 (mac.c:1690, mac.c:2142) —
/// `ext_wde_min_qt_wcpu`, applied to both min and max of `R_BE_WDE_QTA1`.
const WDE_QT_WCPU: u32 = 6;
/// `ple_qt9` (mac.c:1710), min == max: c2h 32, h2c 256, cpu_io 1, the
/// other ten zero, in the register order of `ple_quota_cfg_be`
/// (mac_be.c:320-332): cma0_tx, cma1_tx, c2h, h2c, wcpu, mpdu_proc,
/// cma0_dma, cma1_dma, bb_rpt, wd_rel, cpu_io, tx_rpt, h2d.
const PLE_QT_DLFW: [u32; 13] = [0, 0, 32, 256, 0, 0, 0, 0, 0, 0, 1, 0, 0];

/// Only audited profiles can reach the DLE programming engine. Both use
/// the same physical partition; linked pages and quota ownership change.
struct DleProfile {
    wde_linked: u32,
    ple_linked: u32,
    wde: [u32; 5],
    ple_min: [u32; 13],
    ple_max: [u32; 13],
}
const DOWNLOAD_DLE: DleProfile = DleProfile {
    wde_linked: WDE_LNK_PAGES,
    ple_linked: PLE_LNK_PAGES,
    wde: [0, WDE_QT_WCPU, 0, 0, 0],
    ple_min: PLE_QT_DLFW,
    ple_max: PLE_QT_DLFW,
};
// Linux v6.17 mac.c:1652,1671,1690,1704-1705,1742; rtw8922a.c:61-65.
const NORMAL_DLE: DleProfile = DleProfile {
    wde_linked: 3328,
    ple_linked: 2688,
    wde: [3302, 6, 0, 0, 20],
    ple_min: [320, 320, 32, 16, 13, 13, 292, 292, 64, 18, 1, 4, 0],
    ple_max: [
        320, 320, 32, 16, 1316, 1316, 1595, 1595, 1367, 1321, 1, 1307, 0,
    ],
};
const _: () = assert!(3328 * WDE_PAGE_SIZE == PLE_START);
const _: () = assert!(PLE_START + (2688 + 240) * PLE_PAGE_SIZE + RSVD1_SIZE == FIFO_SIZE);
const _: () = assert!(2 + 107 + 107 + 6 + 6 + 6 + 6 == 240);
/// `R_BE_PLE_QTA0_CFG..QTA12_CFG` step by 4 (reg.h:5395-5443).
const PLE_QTA_STRIDE: u32 = 4;
const _: () = assert!(R_BE_PLE_QTA0_CFG + 12 * PLE_QTA_STRIDE == R_BE_PLE_QTA12_CFG);
/// `R_BE_WDE_QTA0_CFG..QTA4_CFG` likewise (reg.h:5274-5290).
const _: () = assert!(R_BE_WDE_QTA0_CFG + 4 * PLE_QTA_STRIDE == R_BE_WDE_QTA4_CFG);

/// One quota register: `min` in `[11:0]`, `max` in `[27:16]`
/// (`SET_QUOTA_VAL`, mac_be.c:286-291; the WDE and PLE masks are the same
/// positions, reg.h:5275-5276 / 5396-5397).
const fn quota(min: u32, max: u32) -> u32 {
    encode(min, B_BE_WDE_Q0_MIN_SIZE_MASK) | encode(max, B_BE_WDE_Q0_MAX_SIZE_MASK)
}
const _: () = assert!(B_BE_WDE_Q0_MIN_SIZE_MASK == B_BE_PLE_Q0_MIN_SIZE_MASK);
const _: () = assert!(B_BE_WDE_Q0_MAX_SIZE_MASK == B_BE_PLE_Q0_MAX_SIZE_MASK);

/// `rtw89_mac_dle_init` (mac.c:2115-2175) with the BE
/// hooks: engines off, clocks on, the packet-buffer carve, the quotas,
/// engines on, both ready polls.
fn dle_init<R: RegsNarrow>(regs: &R, waits: &Waits, profile: &DleProfile) -> Result<(), MacError> {
    let v = regs.read32(reg(R_BE_DMAC_FUNC_EN));
    regs.write32(reg(R_BE_DMAC_FUNC_EN), v & !DLE_FUNC_EN);
    let v = regs.read32(reg(R_BE_DMAC_CLK_EN));
    regs.write32(reg(R_BE_DMAC_CLK_EN), v | DLE_CLK_EN);

    // `dle_mix_cfg_be` (mac_be.c:210-266).
    let mut v = regs.read32(reg(R_BE_WDE_PKTBUF_CFG));
    v = replace(v, B_BE_WDE_PAGE_SEL_MASK, S_AX_WDE_PAGE_SEL_64);
    v = replace(v, B_BE_WDE_START_BOUND_MASK, WDE_START / DLE_BOUND_UNIT);
    v = replace(v, B_BE_WDE_FREE_PAGE_NUM_MASK, profile.wde_linked);
    regs.write32(reg(R_BE_WDE_PKTBUF_CFG), v);
    let mut v = regs.read32(reg(R_BE_PLE_PKTBUF_CFG));
    v = replace(v, B_BE_PLE_PAGE_SEL_MASK, S_AX_PLE_PAGE_SEL_128);
    v = replace(v, B_BE_PLE_START_BOUND_MASK, PLE_START / DLE_BOUND_UNIT);
    v = replace(v, B_BE_PLE_FREE_PAGE_NUM_MASK, profile.ple_linked);
    regs.write32(reg(R_BE_PLE_PKTBUF_CFG), v);

    // `wde_quota_cfg_be` (mac_be.c:297-312): hif, wcpu, (0, 0), pkt_in,
    // cpu_io — all zero in DLFW but the overridden wcpu.
    for (i, q) in profile.wde.into_iter().enumerate() {
        regs.write32(
            reg(R_BE_WDE_QTA0_CFG + i as u32 * PLE_QTA_STRIDE),
            quota(q, q),
        );
    }
    for (i, (min, max)) in profile.ple_min.into_iter().zip(profile.ple_max).enumerate() {
        regs.write32(
            reg(R_BE_PLE_QTA0_CFG + i as u32 * PLE_QTA_STRIDE),
            quota(min, max),
        );
    }

    let v = regs.read32(reg(R_BE_DMAC_FUNC_EN));
    regs.write32(reg(R_BE_DMAC_FUNC_EN), v | DLE_FUNC_EN);
    for (r, mask) in [
        (R_AX_WDE_INI_STATUS, WDE_MGN_INI_RDY),
        (R_AX_PLE_INI_STATUS, PLE_MGN_INI_RDY),
    ] {
        if !waits.poll_field(regs, r, mask, mask >> mask.trailing_zeros(), DLE_READY_NS) {
            return Err(MacError::DleNotReady);
        }
    }
    Ok(())
}

/// `hfc_prec_cfg_c2.h2c_prec` (mac.c:1649): the H2C precharge page count
/// the pre-init programs. It does not fit `B_BE_PREC_PAGE_CH12_V1_MASK`
/// and upstream writes the truncated zero; so does this (dossier open
/// question 1).
const HFC_H2C_PREC: u32 = 256;

/// `rtw89_mac_hfc_init(reset = true, en = false, h2c_en = true)`
/// (mac.c:1174-1197) with the BE hooks: flow control off, the H2C
/// precharge written, the channel-12 enable set.
fn hfc_pre_init<R: RegsNarrow>(regs: &R) {
    let v = regs.read32(reg(R_BE_HCI_FC_CTRL));
    regs.write32(
        reg(R_BE_HCI_FC_CTRL),
        v & !(B_BE_HCI_FC_EN | B_BE_HCI_FC_CH12_EN),
    );
    regs.write32(
        reg(R_BE_CH_PAGE_CTRL),
        encode(HFC_H2C_PREC, B_BE_PREC_PAGE_CH12_V1_MASK),
    );
    let v = regs.read32(reg(R_BE_HCI_FC_CTRL));
    regs.write32(
        reg(R_BE_HCI_FC_CTRL),
        (v & !B_BE_HCI_FC_EN) | B_BE_HCI_FC_CH12_EN,
    );
}

/// `rtw89_mac_ctrl_hci_dma_trx(true)` + `rtw89_mac_dmac_pre_init`
/// (mac.c:4048, mac.c:3995-4013): the HCI DMA function bits, the DMA mode
/// and address mode, the DLE carve, the flow-control pre-init.
pub fn dmac_pre_init<R: RegsNarrow>(regs: &R, waits: &Waits) -> Result<(), MacError> {
    // `rtw89_mac_hci_func_en_be` twice over (mac.c:4048 and mac.c:4000):
    // idempotent.
    let v = regs.read32(reg(R_BE_HCI_FUNC_EN));
    regs.write32(
        reg(R_BE_HCI_FUNC_EN),
        v | B_BE_HCI_TXDMA_EN | B_BE_HCI_RXDMA_EN,
    );
    // `rtw89_mac_dmac_func_pre_en_be` (mac_be.c:340-374), PCIe branch.
    let v = regs.read32(reg(R_BE_HAXI_INIT_CFG1));
    regs.write32(
        reg(R_BE_HAXI_INIT_CFG1),
        replace(v, B_BE_DMA_MODE_MASK, S_BE_DMA_MOD_PCIE_NO_DATA_CPU),
    );
    let v = regs.read32(reg(R_BE_HAXI_DMA_STOP1));
    regs.write32(reg(R_BE_HAXI_DMA_STOP1), v & !TX_STOP_ALL);
    let v = regs.read32(reg(R_BE_DMAC_TABLE_CTRL));
    regs.write32(reg(R_BE_DMAC_TABLE_CTRL), v | B_BE_DMAC_ADDR_MODE);
    dle_init(regs, waits, &DOWNLOAD_DLE)?;
    hfc_pre_init(regs);
    Ok(())
}

/// The rings the pre-init points the engine at: one device address per
/// transmit channel and per receive queue, and each kind's entry count.
#[derive(Clone, Copy, Debug)]
pub struct RingSetup {
    /// Transmit BD ring bases, CH0..CH12.
    pub tx: [u64; TX_CHANNELS],
    /// Entries per transmit ring (the 16-bit `num` register).
    pub tx_len: u16,
    /// Receive BD ring bases, RXQ0 then RPQ0.
    pub rx: [u64; RX_CHANNELS],
    /// Entries per receive ring.
    pub rx_len: u16,
}

/// `rtw89_pci_set_io_rcy_be` with `io_rcy_en` (pci_be.c:78-107): the
/// PCIe recovery watchdogs, all at their default constants.
fn set_io_rcy<R: RegsNarrow>(regs: &R) {
    let ana = PL0_TMR_ANA_172US / PL0_TMR_SCALE_ASIC;
    for r in [
        R_BE_AON_WDT_TMR,
        R_BE_MDIO_WDT_TMR,
        R_BE_LA_MODE_WDT_TMR,
        R_BE_WDT_AR_TMR,
        R_BE_WDT_AW_TMR,
        R_BE_WDT_W_TMR,
        R_BE_WDT_B_TMR,
        R_BE_WDT_R_TMR,
    ] {
        regs.write32(reg(r), ana);
    }
    let mac = PL0_TMR_MAC_1MS / PL0_TMR_SCALE_ASIC;
    regs.write32(reg(R_BE_WLAN_WDT_TMR), mac);
    regs.write32(reg(R_BE_AXIDMA_WDT_TMR), mac);
    regs.write32(
        reg(R_BE_LOCAL_WDT_TMR),
        PL0_TMR_AUX_1MS / PL0_TMR_SCALE_ASIC,
    );
}

/// `rtw89_pci_ctrl_trxdma_pcie_be` (pci_be.c:131-160) with every argument
/// decided: both engines and the AXI master together.
fn ctrl_trxdma<R: RegsNarrow>(regs: &R, enable: bool) {
    let mut v = regs.read32(reg(R_BE_HAXI_INIT_CFG1));
    if enable {
        v = (v | B_BE_TXDMA_EN | B_BE_RXDMA_EN) & !B_BE_STOP_AXI_MST;
    } else {
        v = (v & !(B_BE_TXDMA_EN | B_BE_RXDMA_EN)) | B_BE_STOP_AXI_MST;
    }
    regs.write32(reg(R_BE_HAXI_INIT_CFG1), v);
    if enable {
        let v = regs.read32(reg(R_BE_HAXI_MST_WDT_TIMEOUT_SEL_V1));
        regs.write32(
            reg(R_BE_HAXI_MST_WDT_TIMEOUT_SEL_V1),
            replace(v, B_BE_HAXI_MST_WDT_TIMEOUT_SEL_MASK, 4),
        );
    }
}

/// `rtw89_pci_clr_idx_all_be` (pci_be.c:162-183): every ring's indices to
/// zero, then the receive host indices to `len - 1` (`rx_ring_eq_is_full`).
fn clr_idx_all<R: RegsNarrow>(regs: &R, rx_len: u16) {
    regs.write32(reg(R_BE_TXBD_RWPTR_CLR1), TXBD_CLR_ALL);
    regs.write32(reg(R_BE_RXBD_RWPTR_CLR1_V1), RXBD_CLR_ALL);
    regs.write16(reg(RXQ0.idx), rx_len - 1);
    regs.write16(reg(RPQ0.idx), rx_len - 1);
}

/// `rtw89_pci_reset_trx_rings` (pci.c:1610-1675) for a part without BD
/// RAM: each ring's length and base, the receive host indices at
/// `len - 1`, then the 16 write-pointer lanes (`rtw89_pci_init_wp_16sel`,
/// pci.c:1591-1608: the identity map, four bytes per dword).
fn reset_trx_rings<R: RegsNarrow>(regs: &R, rings: &RingSetup) {
    for (ch, &base) in rings.tx.iter().enumerate() {
        let r = TxChannelRegs::of(ch);
        regs.write16(reg(r.num), rings.tx_len);
        regs.write32(reg(r.desa_l), base as u32);
        regs.write32(reg(r.desa_l) + 4, (base >> 32) as u32);
    }
    for (r, &base) in [RXQ0, RPQ0].iter().zip(rings.rx.iter()) {
        regs.write16(reg(r.num), rings.rx_len);
        regs.write32(reg(r.desa_l), base as u32);
        regs.write32(reg(r.desa_l) + 4, (base >> 32) as u32);
        regs.write16(reg(r.idx), rings.rx_len - 1);
    }
    for (i, r) in [
        R_BE_WP_ADDR_H_SEL0_3_V1,
        R_BE_WP_ADDR_H_SEL4_7_V1,
        R_BE_WP_ADDR_H_SEL8_11_V1,
        R_BE_WP_ADDR_H_SEL12_15_V1,
    ]
    .into_iter()
    .enumerate()
    {
        let lane = 4 * i as u32;
        regs.write32(
            reg(r),
            lane | (lane + 1) << 8 | (lane + 2) << 16 | (lane + 3) << 24,
        );
    }
}

/// `rtw89_pci_mode_op_be` (pci_be.c:223-256) with `rtw8922a_pci_info`'s
/// values (dossier §A.3): packet-mode receive BDs, 256-byte transmit and
/// 128-byte receive bursts, 8 tags, 256 ns write-pointer DMA intervals.
/// The append-mode register is read and written back unchanged, as
/// upstream does in packet mode.
fn mode_op<R: RegsNarrow>(regs: &R) {
    /// `PCIE_RXBD_NORM` (pci_be.c:12).
    const RXBD_NORM: u32 = 0;
    /// `MAC_AX_TX_BURST_V1_256B` (pci.h:1162).
    const TX_BURST: u32 = 2;
    /// `MAC_AX_RX_BURST_V1_128B` (pci.h:1175).
    const RX_BURST: u32 = 1;
    /// `MAC_AX_WD_DMA_INTVL_256NS` (pci.h:1182).
    const WD_INTVL: u32 = 1;
    /// `MAC_AX_TAG_NUM_8` (pci.h:1202).
    const TAG_NUM: u32 = 7;
    let mut init1 = regs.read32(reg(R_BE_HAXI_INIT_CFG1));
    let rxapp = regs.read32(reg(R_BE_RX_APPEND_MODE));
    let mut exp = regs.read32(reg(R_BE_HAXI_EXP_CTRL_V1));
    init1 = replace(init1, B_BE_RXQ_RXBD_MODE_MASK, RXBD_NORM);
    init1 = replace(init1, B_BE_MAX_TXDMA_MASK, TX_BURST);
    init1 = replace(init1, B_BE_MAX_RXDMA_MASK, RX_BURST);
    exp = replace(exp, B_BE_MAX_TAG_NUM_MASK, TAG_NUM);
    init1 = replace(init1, B_BE_CFG_WD_PERIOD_IDLE_MASK, WD_INTVL);
    init1 = replace(init1, B_BE_CFG_WD_PERIOD_ACTIVE_MASK, WD_INTVL);
    regs.write32(reg(R_BE_HAXI_INIT_CFG1), init1);
    regs.write32(reg(R_BE_RX_APPEND_MODE), rxapp);
    regs.write32(reg(R_BE_HAXI_EXP_CTRL_V1), exp);
}

/// `rtw89_pci_ops_mac_pre_init_be` (pci_be.c:353-390), steps 1-15 of the
/// dossier's §C.4: everything up to, but not including, the transmit and
/// receive engines' enable — the one write after which the device fetches
/// from the rings, which [`trxdma_enable`] does under `DmaActive`.
pub fn pci_pre_init<R: RegsNarrow>(
    regs: &R,
    waits: &Waits,
    cv: u8,
    rings: &RingSetup,
) -> Result<(), MacError> {
    set_io_rcy(regs);
    // `_patch_pcie_power_wake_be(true)`.
    let v = regs.read32(reg(R_BE_HCI_OPT_CTRL));
    regs.write32(reg(R_BE_HCI_OPT_CTRL), v | BIT_WAKE_CTRL_V1);
    // `rtw89_pci_ctrl_wpdma_pcie_be(false)`.
    let v = regs.read32(reg(R_BE_HAXI_DMA_STOP1));
    regs.write32(reg(R_BE_HAXI_DMA_STOP1), v | B_BE_STOP_WPDMA);
    ctrl_trxdma(regs, false);
    clr_idx_all(regs, rings.rx_len);
    // `rtw89_pci_poll_dma_all_idle_be`.
    if !waits.poll_clear(regs, R_BE_HAXI_DMA_BUSY1, DMA_BUSY1_CHECK_BE, DMA_IDLE_NS)
        || !waits.poll_clear(regs, R_BE_HAXI_DMA_BUSY1, RX_DMA_BUSY, DMA_IDLE_NS)
    {
        return Err(MacError::DmaBusy);
    }
    mode_op(regs);
    reset_trx_rings(regs, rings);
    // `rtw89_pci_rst_bdram_be`.
    let v = regs.read32(reg(R_BE_HAXI_INIT_CFG1));
    regs.write32(reg(R_BE_HAXI_INIT_CFG1), v | B_BE_SET_BDRAM_BOUND);
    if !waits.poll_clear(regs, R_BE_HAXI_INIT_CFG1, B_BE_SET_BDRAM_BOUND, BDRAM_NS) {
        return Err(MacError::BdramBound);
    }
    // `rtw89_pci_debounce_be`.
    let v = regs.read32(reg(R_BE_SYS_PAGE_CLK_GATED));
    regs.write32(
        reg(R_BE_SYS_PAGE_CLK_GATED),
        replace(v, B_BE_PCIE_PRST_DEBUNC_PERIOD_MASK, 0) | B_BE_SYM_PRST_DEBUNC_SEL,
    );
    // `rtw89_pci_ldo_low_pwr_be`.
    let v = regs.read32(reg(R_BE_SYS_PW_CTRL));
    regs.write32(reg(R_BE_SYS_PW_CTRL), v | B_BE_PSUS_OFF_CAPC_EN);
    let v = regs.read32(reg(R_BE_SYS_PAGE_CLK_GATED));
    regs.write32(
        reg(R_BE_SYS_PAGE_CLK_GATED),
        v | B_BE_SOP_OFFPOOBS_PC | B_BE_CPHY_AUXCLK_OP | B_BE_CPHY_POWER_READY_CHK,
    );
    let v = regs.read32(reg(R_BE_SYS_SDIO_CTRL));
    regs.write32(
        reg(R_BE_SYS_SDIO_CTRL),
        v & !(B_BE_PCIE_FORCE_IBX_EN | B_BE_PCIE_DIS_L2_RTK_PERST | B_BE_PCIE_DIS_L2__CTRL_LDO_HCI),
    );
    let v = regs.read32(reg(R_BE_L1_2_CTRL_HCILDO));
    regs.write32(
        reg(R_BE_L1_2_CTRL_HCILDO),
        v & !B_BE_PCIE_DIS_L1_2_CTRL_HCILDO,
    );
    // `rtw89_pci_pcie_setting_be`.
    let v = regs.read32(reg(R_BE_PCIE_FRZ_CLK));
    regs.write32(reg(R_BE_PCIE_FRZ_CLK), v | B_BE_PCIE_EN_AUX_CLK);
    let v = regs.read32(reg(R_BE_PCIE_PS_CTRL));
    regs.write32(reg(R_BE_PCIE_PS_CTRL), v & !B_BE_CMAC_EXIT_L1_EN);
    if cv != CHIP_CAV {
        let v = regs.read32(reg(R_BE_EFUSE_CTRL_2_V1));
        regs.write32(
            reg(R_BE_EFUSE_CTRL_2_V1),
            v | B_BE_R_SYM_AUTOLOAD_WITH_PMC_SEL,
        );
        let v = regs.read32(reg(R_BE_PCIE_LAT_CTRL));
        regs.write32(reg(R_BE_PCIE_LAT_CTRL), v | B_BE_SYM_AUX_CLK_SEL);
    }
    // `rtw89_pci_ser_setting_be`.
    regs.write32(reg(R_BE_PL1_DBG_INFO), 0);
    let v = regs.read32(reg(R_BE_FWS1IMR));
    regs.write32(reg(R_BE_FWS1IMR), v | B_BE_PCIE_SER_TIMEOUT_INDIC_EN);
    let v = regs.read32(reg(R_BE_SER_PL1_CTRL));
    regs.write32(reg(R_BE_SER_PL1_CTRL), v | B_BE_PL1_SER_PL1_EN);
    let v = regs.read32(reg(R_BE_SER_PL1_CTRL));
    regs.write32(
        reg(R_BE_SER_PL1_CTRL),
        replace(v, B_BE_PL1_TIMER_UNIT_MASK, 1),
    );
    let v = regs.read32(reg(R_BE_REG_PL1_MASK));
    regs.write32(
        reg(R_BE_REG_PL1_MASK),
        v | B_BE_SER_PMU_IMR
            | B_BE_SER_L1SUB_IMR
            | B_BE_SER_PM_MASTER_IMR
            | B_BE_SER_LTSSM_IMR
            | B_BE_SER_PM_CLK_MASK
            | B_BE_SER_PCLKREQ_ACK_MASK,
    );
    // `rtw89_pci_ctrl_txdma_ch_be(false)`: the data channels stopped.
    let v = regs.read32(reg(R_BE_HAXI_DMA_STOP1));
    regs.write32(reg(R_BE_HAXI_DMA_STOP1), v | TX_STOP_UNUSED | TX_STOP_DATA);
    // `rtw89_pci_ctrl_txdma_fw_ch_be(true)`: the firmware channel open.
    let v = regs.read32(reg(R_BE_HAXI_DMA_STOP1));
    regs.write32(reg(R_BE_HAXI_DMA_STOP1), v & !B_BE_STOP_CH12);
    Ok(())
}

/// Step 16 of `rtw89_pci_ops_mac_pre_init_be`: `rtw89_pci_ctrl_trxdma_pcie_be(ENABLE, ENABLE, ENABLE)`.
/// After this the device fetches channel 12's descriptors and fills the
/// receive rings.
pub fn trxdma_enable<R: RegsNarrow>(regs: &R) {
    ctrl_trxdma(regs, true);
}

/// PHY0 BB-MCU memory/reset prerequisite for firmware loading. Register
/// facts: Linux v6.17 rtw8922a.c `rtw8922a_bb_preinit`/`bb_mcu0_init_reg`,
/// phy.h `rtw89_bbmcu_write32`. PHY1/DBCC is not selected on this path.
/// This runs while the host still holds DmaPrepared: no host DMA is enabled.
pub fn baseband_pre_init<R: RegsNarrow>(regs: &R, waits: &Waits) {
    for (addr, mask, value) in [
        (R_BE_DMAC_SYS_CR32B, B_BE_DMAC_BB_PHY0_MASK, 0x7ff9),
        (R_BE_FEN_RST_ENABLE, B_BE_FEN_BB_IP_RSTN, 0),
        (R_BE_FEN_RST_ENABLE, B_BE_FEN_BBPLAT_RSTB, 0),
        (R_BE_FEN_RST_ENABLE, B_BE_FEN_BB_IP_RSTN, 1),
        (R_BE_FEN_RST_ENABLE, B_BE_BOOT_RDY0, 0),
        (R_BE_MEM_PWR_CTRL, B_BE_MEM_BBMCU0_DS_V1, 0),
    ] {
        let old = regs.read32(reg(addr));
        regs.write32(reg(addr), replace(old, mask, value));
    }
    waits.delay(1_000);
    // BB-MCU registers use their own 0x30000 aperture, not the PHY aperture.
    for (offset, value) in [
        (0x6990, 0),
        (0x6994, 0),
        (0x6998, 0),
        (0x6820, 0xffff_fffe),
        (0x6800, 0xc000_0ffe),
        (0x6808, 0x7654_3210),
        (0x6814, 0xbfbf_b000),
        (0x6818, 0x0478_c009),
        (0x6800, 0xc000_0fff),
        (0x6820, 0xffff_ffff),
    ] {
        regs.write32(reg(RTW89_BBMCU_ADDR_OFFSET + offset), value);
    }
}

/// The highest BB-MCU control register above is 0x6998 (a 32-bit access).
/// The D1 MAC-only 128 KiB mapping is insufficient for this aperture.
pub const BASEBAND_REGS_SPAN: usize = RTW89_BBMCU_ADDR_OFFSET as usize + 0x6998 + 4;

/// `rtw89_pci_ops_mac_pre_deinit_be` (pci_be.c:392-407): drop the power
/// wake, and — if the MAC is still on — stop both engines and clear every
/// index.
pub fn pci_pre_deinit<R: RegsNarrow>(regs: &R, rx_len: u16) {
    let v = regs.read32(reg(R_BE_HCI_OPT_CTRL));
    regs.write32(reg(R_BE_HCI_OPT_CTRL), v & !BIT_WAKE_CTRL_V1);
    if get(regs.read32(reg(R_BE_IC_PWR_STATE)), B_BE_WLMAC_PWR_STE_MASK) == 0 {
        return;
    }
    ctrl_trxdma(regs, false);
    clr_idx_all(regs, rx_len);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sequences_match_the_dossier_step_counts_and_key_values() {
        // §B.5: 42 numbered rows, the A-die patch (row 38 and its six
        // sub-steps) one entry here.
        assert_eq!(POWER_ON.len(), 42);
        assert_eq!(POWER_ON[38], Op::Set32(R_BE_DMAC_FUNC_EN, 0x7FFF_4F80));
        assert_eq!(
            POWER_ON[39],
            Op::Set32(R_BE_CMAC_SHARE_FUNC_EN, 0x4000_0007)
        );
        assert_eq!(POWER_ON[40], Op::Set32(R_BE_CMAC_FUNC_EN, 0x7000_01FF));
        assert!(matches!(
            POWER_ON[5],
            Op::PollSet(R_BE_SYS_PW_CTRL, B_BE_RDY_SYSPWR, _)
        ));
        assert_eq!(POWER_OFF.len(), 29);
        assert_eq!(quota(6, 6), 6 | 6 << 16);
        assert_eq!(quota(256, 256), 256 | 256 << 16);
        assert_eq!(PLE_START / DLE_BOUND_UNIT, 26);
    }
}
