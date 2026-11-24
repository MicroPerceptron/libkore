//! The RTL8922AE register face.
//!
//! [`be`] is **generated** from `tools/rtw89_be.h`, the cited transcription
//! of the Linux `rtw89` headers — do not hand-edit it; change the header and
//! regenerate (the command is in the header's comment). This file is the
//! hand-written half: the compositions the ladder programs, each with the
//! derivation the number came from, the field helpers, and the pool-free
//! decode probe.

pub mod be;

use kore_driver::seam::Regs;

use be::*;

/// Bytes of BAR2 the D1 MAC-only ladder reads or writes: the highest register it
/// touches is `R_BE_CMAC_FUNC_EN` at `0x10000` (power-on step 41), the
/// first register of the CMAC block, so the test is a window that decodes
/// that whole 64 KiB block above it. The part's BAR2 is 1 MiB; a shorter
/// window is a function that is not this part.
pub const REGS_SPAN: usize = 0x2_0000;
const _: () = assert!(R_BE_CMAC_FUNC_EN as usize + 4 <= REGS_SPAN);
const _: () = assert!(R_BE_CMAC_FUNC_EN as usize == 0x1_0000);

/// `GENMASK(hi, lo)`, for the compositions below that name a bit range
/// upstream spells out bit by bit (the generated file keeps its own copy
/// private).
#[inline]
pub const fn genmask(hi: u32, lo: u32) -> u32 {
    (u32::MAX >> (31 - (hi - lo))) << lo
}

/// A generated `u32` offset as the `usize` the [`Regs`] seam takes.
#[inline]
pub const fn reg(offset: u32) -> usize {
    offset as usize
}

/// Extract `mask`'s field from `word`, shifted down to its own zero
/// (Linux `u32_get_bits`).
#[inline]
pub const fn get(word: u32, mask: u32) -> u32 {
    (word & mask) >> mask.trailing_zeros()
}

/// `val` placed in `mask`'s field (Linux `u32_encode_bits`). A value that
/// does not fit is truncated exactly as upstream's macro truncates it —
/// deliberately, because one upstream write relies on it (`hfc_h2c_cfg_be`
/// encodes 256 into a six-bit field and writes the resulting zero; open
/// question 1 of the dossier).
#[inline]
pub const fn encode(val: u32, mask: u32) -> u32 {
    (val << mask.trailing_zeros()) & mask
}

/// `word` with `mask`'s field replaced by `val` (Linux `u32_replace_bits`).
#[inline]
pub const fn replace(word: u32, mask: u32, val: u32) -> u32 {
    (word & !mask) | encode(val, mask)
}

/// The cut (chip revision) the MFW table and two silicon workarounds key
/// on: `R_AX_SYS_CFG1[15:12]` (`rtw89_read_chip_ver`, core.c:5352). No
/// BE-specific spelling exists; the register is readable before power-on
/// and is upstream's first MMIO access.
#[inline]
pub fn cut_version(regs: &impl Regs) -> u8 {
    get(regs.read32(reg(R_AX_SYS_CFG1)), B_AX_CHIP_VER_MASK) as u8
}

/// The A-cut. `enum rtw89_core_chip_cv { CHIP_CAV, CHIP_CBV, … }` —
/// core.h; the two `cv == CHIP_CAV` guards skip the A-die patch on the
/// first silicon (`rtw8922a_pwr_on_func` step 38, `pcie_setting_be`,
/// `enable_efuse_pwr_cut_ddv_be`).
pub const CHIP_CAV: u8 = 0;

/// `R_BE_IC_PWR_STATE[9:8]` when the WLAN MAC is powered (`PWR_ACT`,
/// mac.c:1462) — the already-on guard `rtw89_mac_power_switch` refuses on.
pub const PWR_ACT: u32 = 1;

/// `R_BE_DMAC_FUNC_EN`'s power-on value: every function the 8922A enables
/// at once (`rtw8922a_pwr_on_func`, rtw8922a.c:385-392). Bits 30..16, 14
/// and 11..7: `0x7FFF0000 | 0x4000 | 0x0F80` = `0x7FFF4F80`.
pub const DMAC_FUNC_EN_ALL: u32 = B_BE_MAC_FUNC_EN
    | B_BE_DMAC_FUNC_EN
    | B_BE_MPDU_PROC_EN
    | B_BE_WD_RLS_EN
    | B_BE_DLE_WDE_EN
    | B_BE_TXPKT_CTRL_EN
    | B_BE_STA_SCH_EN
    | B_BE_DLE_PLE_EN
    | B_BE_PKT_BUF_EN
    | B_BE_DMAC_TBL_EN
    | B_BE_PKT_IN_EN
    | B_BE_DLE_CPUIO_EN
    | B_BE_DISPATCHER_EN
    | B_BE_BBRPT_EN
    | B_BE_MAC_SEC_EN
    | B_BE_H_AXIDMA_EN
    | B_BE_DMAC_MLO_EN
    | B_BE_PLRLS_EN
    | B_BE_P_AXIDMA_EN
    | B_BE_DLE_DATACPUIO_EN
    | B_BE_LTR_CTL_EN;
const _: () = assert!(DMAC_FUNC_EN_ALL == 0x7FFF_4F80);

/// `R_BE_CMAC_SHARE_FUNC_EN`'s power-on value (rtw8922a.c:396-398):
/// bit 30 and bits 2..0 = `0x40000007`.
pub const CMAC_SHARE_FUNC_EN_ALL: u32 =
    B_BE_CMAC_SHARE_EN | B_BE_RESPBA_EN | B_BE_ADDRSRCH_EN | B_BE_BTCOEX_EN;
const _: () = assert!(CMAC_SHARE_FUNC_EN_ALL == 0x4000_0007);

/// `R_BE_CMAC_FUNC_EN`'s power-on value (rtw8922a.c:399-403): bits 30..28
/// and 8..0 = `0x700001FF`.
pub const CMAC_FUNC_EN_ALL: u32 = B_BE_CMAC_EN
    | B_BE_CMAC_TXEN
    | B_BE_CMAC_RXEN
    | B_BE_SIGB_EN
    | B_BE_PHYINTF_EN
    | B_BE_CMAC_DMA_EN
    | B_BE_PTCLTOP_EN
    | B_BE_SCHEDULER_EN
    | B_BE_TMAC_EN
    | B_BE_RMAC_EN
    | B_BE_TXTIME_EN
    | B_BE_RESP_PKTCTL_EN;
const _: () = assert!(CMAC_FUNC_EN_ALL == 0x7000_01FF);

/// Both DLE engines' function bits (`dle_func_en_be`, mac_be.c:190).
pub const DLE_FUNC_EN: u32 = B_BE_DLE_WDE_EN | B_BE_DLE_PLE_EN;
/// …and their clock bits (`dle_clk_en_be`, mac_be.c:200) — the same two
/// positions in the clock register.
pub const DLE_CLK_EN: u32 = B_BE_DLE_WDE_CLK_EN | B_BE_DLE_PLE_CLK_EN;
const _: () = assert!(DLE_FUNC_EN == DLE_CLK_EN);

/// `WDE_MGN_INI_RDY` / `PLE_MGN_INI_RDY` (mac.h): both the queue-manager
/// and buffer-manager ready bits of the init-status registers, bits 1 and
/// 0. `chk_dle_rdy_be` polls for the whole mask (mac_be.c:281).
pub const WDE_MGN_INI_RDY: u32 = B_AX_WDE_Q_MGN_INI_RDY | B_AX_WDE_BUF_MGN_INI_RDY;
/// The PLE twin.
pub const PLE_MGN_INI_RDY: u32 = B_AX_PLE_Q_MGN_INI_RDY | B_AX_PLE_BUF_MGN_INI_RDY;
const _: () = assert!(WDE_MGN_INI_RDY == 0x3 && PLE_MGN_INI_RDY == 0x3);

/// `DLE_BOUND_UNIT` (mac.h:589): the start-bound fields of the packet
/// buffer configuration count 8 KiB units.
pub const DLE_BOUND_UNIT: u32 = 8 * 1024;

/// Every transmit channel's STOP bit, CH0..CH14 = bits 0..14
/// (`B_BE_STOP_CH0..CH14`, reg.h:5978-5992). `dmac_func_pre_en_be` clears
/// them all (mac_be.c:366-371); the pre-init sets them all back but CH12.
pub const TX_STOP_ALL: u32 = genmask(14, 0);
const _: () = assert!(TX_STOP_ALL & B_BE_STOP_CH0 != 0 && TX_STOP_ALL & B_BE_STOP_CH14 != 0);
/// The data channels' STOP bits, CH0..CH11 (`rtw89_pci_ctrl_txdma_ch_be`'s
/// `mask_all`, pci_be.c:325-328): stopped for the whole of D1, because no
/// frame is ever posted on them.
pub const TX_STOP_DATA: u32 = genmask(11, 0);
const _: () = assert!(TX_STOP_DATA & B_BE_STOP_CH11 != 0 && TX_STOP_DATA & B_BE_STOP_CH12 == 0);
/// The two channels the driver never allocates rings for, held stopped
/// by every `ctrl_txdma_ch_be` call (pci_be.c:331).
pub const TX_STOP_UNUSED: u32 = B_BE_STOP_CH13 | B_BE_STOP_CH14;

/// `DMA_BUSY1_CHECK_BE` (pci.h:1039-1043): every transmit channel's BUSY
/// bit, `B_BE_CH0_BUSY..CH14_BUSY` = bits 0..14 (pci.h:1024-1038).
pub const DMA_BUSY1_CHECK_BE: u32 = genmask(14, 0);
/// Both receive queues' BUSY bits (`rtw89_pci_poll_rxdma_ch_idle_be`,
/// pci_be.c:196).
pub const RX_DMA_BUSY: u32 = B_BE_RXQ0_BUSY_V1 | B_BE_RPQ0_BUSY_V1;

/// Every transmit ring's index-clear bit, `B_BE_CLR_CH0_IDX..CH14_IDX` =
/// bits 0..14 (pci.h:989-1003), as `rtw89_pci_clr_idx_all_be` writes them.
pub const TXBD_CLR_ALL: u32 = genmask(14, 0);
const _: () =
    assert!(TXBD_CLR_ALL & B_BE_CLR_CH0_IDX != 0 && TXBD_CLR_ALL & B_BE_CLR_CH14_IDX != 0);
/// Both receive rings' index-clear bits.
pub const RXBD_CLR_ALL: u32 = B_BE_CLR_RXQ0_IDX | B_BE_CLR_RPQ0_IDX;

/// The transmit channels, `RTW89_TXCH_NUM` (txrx.h:647): ACH0..7, CH8..11
/// (management/high), CH12 (firmware command).
pub const TX_CHANNELS: usize = 13;
/// The firmware-command channel, `RTW89_TXCH_CH12` (txrx.h:644): the only
/// transmit channel D1 posts on.
pub const TXCH_FWCMD: usize = 12;
/// The receive channels, `RTW89_RXCH_NUM` (txrx.h:656): RXQ0 (packets and
/// C2H) and RPQ0 (release reports).
pub const RX_CHANNELS: usize = 2;

/// The per-channel transmit ring registers (`rtw89_pci_ch_dma_addr_set_be`
/// resolved in the dossier §G.5): `num` is 16-bit, `idx` 32-bit (host
/// index `[11:0]`, hardware index `[27:16]`), `desa_l`/`desa_h` the ring
/// base. CH0..CH11 step by 2 in `num` and by 4/8 in the others; CH12 is the
/// same arithmetic continued, and its resolved values are asserted below
/// against the generated `_V1` constants so the stride is a fact, not a
/// pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TxChannelRegs {
    /// `R_BE_CHn_TXBD_NUM_V1` — 16-bit.
    pub num: u32,
    /// `R_BE_CHn_TXBD_IDX_V1` — 32-bit read, 16-bit host-index write.
    pub idx: u32,
    /// `R_BE_CHn_TXBD_DESA_L_V1`; `_H` is the next dword.
    pub desa_l: u32,
}

impl TxChannelRegs {
    /// Channel `ch`'s registers.
    pub const fn of(ch: usize) -> Self {
        assert!(ch < TX_CHANNELS);
        let ch = ch as u32;
        Self {
            num: R_BE_CH0_TXBD_NUM_V1 + 2 * ch,
            idx: R_BE_CH0_TXBD_IDX_V1 + 4 * ch,
            desa_l: R_BE_CH0_TXBD_DESA_L_V1 + 8 * ch,
        }
    }
}
const _: () = {
    let ch12 = TxChannelRegs::of(TXCH_FWCMD);
    assert!(ch12.num == R_BE_CH12_TXBD_NUM_V1);
    assert!(ch12.idx == R_BE_CH12_TXBD_IDX_V1);
    assert!(ch12.desa_l == R_BE_CH12_TXBD_DESA_L_V1);
    assert!(ch12.desa_l + 4 == R_BE_CH12_TXBD_DESA_H_V1);
};

/// The two receive rings' registers (dossier §G.5, pci.h:771-843).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RxChannelRegs {
    /// `R_BE_RXQ0/RPQ0_RXBD_NUM_V1` — 16-bit.
    pub num: u32,
    /// `R_BE_RXQ0/RPQ0_RXBD_IDX_V1` — 16-bit host-index write.
    pub idx: u32,
    /// `R_BE_RXQ0/RPQ0_RXBD_DESA_L_V1`; `_H` is the next dword.
    pub desa_l: u32,
}

/// RXQ0: packets and C2H messages.
pub const RXQ0: RxChannelRegs = RxChannelRegs {
    num: R_BE_RXQ0_RXBD_NUM_V1,
    idx: R_BE_RXQ0_RXBD_IDX_V1,
    desa_l: R_BE_RXQ0_RXBD_DESA_L_V1,
};
/// RPQ0: transmit release reports.
pub const RPQ0: RxChannelRegs = RxChannelRegs {
    num: R_BE_RPQ0_RXBD_NUM_V1,
    idx: R_BE_RPQ0_RXBD_IDX_V1,
    desa_l: R_BE_RPQ0_RXBD_DESA_L_V1,
};
const _: () = assert!(RXQ0.desa_l + 4 == R_BE_RXQ0_RXBD_DESA_H_V1);
const _: () = assert!(RPQ0.desa_l + 4 == R_BE_RPQ0_RXBD_DESA_H_V1);

/// The register mailbox's four H2C data words (`chip->h2c_regs`,
/// rtw8922a.c:80-83), in write order.
pub const H2CREG_DATA: [u32; RTW89_H2CREG_MAX as usize] = [
    R_BE_H2CREG_DATA0,
    R_BE_H2CREG_DATA1,
    R_BE_H2CREG_DATA2,
    R_BE_H2CREG_DATA3,
];
/// …and the four C2H data words (rtw8922a.c:85-88).
pub const C2HREG_DATA: [u32; RTW89_C2HREG_MAX as usize] = [
    R_BE_C2HREG_DATA0,
    R_BE_C2HREG_DATA1,
    R_BE_C2HREG_DATA2,
    R_BE_C2HREG_DATA3,
];
/// The mailbox counters live in one byte: `R_BE_UDM1 + 1`, the H2C
/// dequeue count in its low nibble and the C2H enqueue count in its high
/// nibble (`chip->h2c_counter_reg` / `c2h_counter_reg`, rtw8922a.c:2950-
/// 2953: `B_BE_UDM1_HALMAC_H2C_DEQ_CNT_MASK >> 8` = `0x0F`,
/// `..._C2H_ENQ_CNT_MASK >> 8` = `0xF0`).
pub const MAILBOX_COUNTER_BYTE: u32 = R_BE_UDM1 + 1;
/// The H2C nibble of [`MAILBOX_COUNTER_BYTE`].
pub const H2C_COUNTER_MASK: u8 = (B_BE_UDM1_HALMAC_H2C_DEQ_CNT_MASK >> 8) as u8;
/// The C2H nibble of [`MAILBOX_COUNTER_BYTE`].
pub const C2H_COUNTER_MASK: u8 = (B_BE_UDM1_HALMAC_C2H_ENQ_CNT_MASK >> 8) as u8;
const _: () = assert!(H2C_COUNTER_MASK == 0x0F && C2H_COUNTER_MASK == 0xF0);

/// One xtal_si (crystal serial interface) write command
/// (`rtw89_mac_write_xtal_si_be`, mac_be.c:382-387): offset in `[7:0]`,
/// data `[15:8]`, bit-mask `[23:16]`, `XTAL_SI_NORMAL_WRITE` in `[25:24]`,
/// chip id 0 in `[30:28]`, and the poll bit the hardware clears on
/// completion.
#[inline]
pub const fn xtal_si_write(offset: u8, val: u8, mask: u8) -> u32 {
    encode(offset as u32, B_BE_WL_XTAL_SI_ADDR_MASK)
        | encode(val as u32, B_BE_WL_XTAL_SI_DATA_MASK)
        | encode(mask as u32, B_BE_WL_XTAL_SI_BITMASK_MASK)
        | encode(XTAL_SI_NORMAL_WRITE, B_BE_WL_XTAL_SI_MODE_MASK)
        | encode(0, B_BE_WL_XTAL_SI_CHIPID_MASK)
        | B_BE_WL_XTAL_SI_CMD_POLL
}

/// The `R_BE_SECURE_BOOT_MALLOC_INFO` value the 8922A's NORMAL image needs
/// before its header is pushed (`rtw89_fw_download_suit`, fw.c:1774-1776).
/// The register has no bit definitions anywhere upstream; the value is
/// opaque and is written verbatim (dossier open question 3).
pub const SECURE_BOOT_MALLOC_INFO_8922A: u32 = 0x2024_8000;

/// Whether the window decodes a live controller: long enough for every
/// MAC-only register (BB preparation checks its additional aperture), and the two registers upstream reads first
/// not all-ones (a function that is unrouted or power-gated reads
/// `0xFFFFFFFF`; a live one cannot, because `R_BE_SYS_PW_CTRL` has
/// reserved-zero bits). Read-only — a probe the row may run before it
/// spends its pool on this function.
pub fn controller_present(regs: &impl Regs, bar_len: usize) -> bool {
    if bar_len < REGS_SPAN {
        return false;
    }
    let cfg1 = regs.read32(reg(R_AX_SYS_CFG1));
    let pw = regs.read32(reg(R_BE_SYS_PW_CTRL));
    cfg1 != u32::MAX && pw != u32::MAX
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixed {
        cfg1: u32,
        pw: u32,
    }

    impl Regs for Fixed {
        fn read32(&self, off: usize) -> u32 {
            match off as u32 {
                R_AX_SYS_CFG1 => self.cfg1,
                R_BE_SYS_PW_CTRL => self.pw,
                _ => 0,
            }
        }
        fn write32(&self, _off: usize, _val: u32) {}
    }

    #[test]
    fn the_probe_refuses_a_short_window_and_a_dead_function() {
        let live = Fixed {
            cfg1: 0x0000_1000,
            pw: 0x0002_0000,
        };
        assert!(controller_present(&live, REGS_SPAN));
        assert!(!controller_present(&live, REGS_SPAN - 4));
        assert!(!controller_present(
            &Fixed {
                cfg1: u32::MAX,
                pw: u32::MAX
            },
            1 << 20
        ));
        assert_eq!(cut_version(&live), 1);
    }

    #[test]
    fn the_field_helpers_match_the_linux_macros() {
        assert_eq!(get(0x0000_1000, B_AX_CHIP_VER_MASK), 1);
        assert_eq!(encode(4, B_BE_HAXI_MST_WDT_TIMEOUT_SEL_MASK), 4);
        assert_eq!(replace(0xFFFF_FFFF, B_BE_DMA_MODE_MASK, 0), 0xFFFF_F8FF);
        // The upstream truncation the HFC pre-init relies on.
        assert_eq!(encode(256, B_BE_PREC_PAGE_CH12_V1_MASK), 0);
        assert_eq!(xtal_si_write(0xE0, 0x02, 0x02), 0x8002_02E0);
    }

    #[test]
    fn the_channel_tables_resolve_to_the_dossier() {
        assert_eq!(TxChannelRegs::of(0).num, 0xB030);
        assert_eq!(TxChannelRegs::of(11).idx, 0xB12C);
        assert_eq!(TxChannelRegs::of(11).desa_l, 0xB258);
        assert_eq!(RXQ0.num, 0xB050);
        assert_eq!(RPQ0.idx, 0xB164);
    }
}
