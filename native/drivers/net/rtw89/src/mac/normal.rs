//! Single-PHY PCIe MAC configuration after both firmware transfers retire.
//!
//! Register facts: Linux v6.17 mac_be.c dmac_init_be/cmac_init_be and
//! rtw8922a.c's SCC tables. The existing operation engine preserves access
//! widths and bounded polls. Data TX channels and all ports remain stopped;
//! interrupt admission stays with the held route's owner, not these tables.
use super::{MacError, NORMAL_DLE, Op, dle_init, run};
use crate::{
    Waits,
    regs::{be::*, encode, reg},
};
use kore_driver::seam::RegsNarrow;

const US: u64 = 1_000;
const INIT_NS: u64 = 2_000 * US;

/// Reconfigure the device's packet memory, scheduler, security and CMAC0.
/// No channel, vif, key, RX filter admission or data TX is granted here.
pub fn init<R: RegsNarrow>(regs: &R, waits: &Waits, cut: u8) -> Result<(), MacError> {
    // rtw8922a_mac_enable_bb_rf. BOOT_RDY0 stays clear until PHY post-init.
    run(
        regs,
        waits,
        cut,
        &[
            Op::Set8(
                R_BE_FEN_RST_ENABLE,
                (B_BE_FEN_BBPLAT_RSTB | B_BE_FEN_BB_IP_RSTN) as u8,
            ),
            Op::Write32(R_BE_DMAC_SYS_CR32B, 0x7ff9_7ff9),
        ],
    )?;
    dle_init(regs, waits, &NORMAL_DLE)?;
    run(regs, waits, cut, PRELOAD)?;
    flow_control(regs, waits, cut)?;
    run(regs, waits, cut, DMAC)?;
    run(regs, waits, cut, CMAC)?;
    if cut == 0 {
        run(
            regs,
            waits,
            cut,
            &[Op::Field32(R_BE_PTCL_RRSR1, B_BE_RSC_MASK, 1)],
        )?;
    }
    run(regs, waits, cut, REPORTS)
}

// The SCC host-flow budget from rtw8922a.c:25-58. Group assignment is by
// hardware channel, not an index inferred from the active data queue count.
fn flow_control<R: RegsNarrow>(regs: &R, waits: &Waits, cut: u8) -> Result<(), MacError> {
    run(
        regs,
        waits,
        cut,
        &[Op::Clr32(
            R_BE_HCI_FC_CTRL,
            B_BE_HCI_FC_EN | B_BE_HCI_FC_CH12_EN,
        )],
    )?;
    for (channel, group) in [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1].into_iter().enumerate() {
        let value = encode(2, B_AX_MIN_PG_MASK)
            | encode(1641, B_AX_MAX_PG_MASK)
            | if group == 1 { B_AX_GRP } else { 0 };
        regs.write32(reg(R_BE_CH0_PAGE_CTRL + channel as u32 * 4), value);
    }
    run(
        regs,
        waits,
        cut,
        &[
            Op::Write32(
                R_BE_PUB_PAGE_CTRL1,
                encode(1651, B_BE_PUBPG_G0_MASK) | encode(1651, B_BE_PUBPG_G1_MASK),
            ),
            Op::Write32(R_BE_WP_PAGE_CTRL2, 0),
            Op::Write32(
                R_BE_CH_PAGE_CTRL,
                encode(2, B_BE_PREC_PAGE_CH011_V1_MASK) | encode(32, B_BE_PREC_PAGE_CH12_V1_MASK),
            ),
            Op::Write32(R_BE_PUB_PAGE_CTRL2, encode(3302, B_BE_PUBPG_ALL_MASK)),
            Op::Write32(R_BE_WP_PAGE_CTRL1, 0),
            Op::Field32(R_BE_HCI_FC_CTRL, B_BE_HCI_FC_MODE_MASK, 0), // core.h:4515 RTW89_HCIFC_POH
            Op::Clr32(
                R_BE_HCI_FC_CTRL,
                B_BE_HCI_FC_WD_FULL_COND_MASK
                    | B_BE_HCI_FC_CH12_FULL_COND_MASK
                    | B_BE_HCI_FC_WP_CH07_FULL_COND_MASK
                    | B_BE_HCI_FC_WP_CH811_FULL_COND_MASK,
            ),
            Op::Set32(R_BE_HCI_FC_CTRL, B_BE_HCI_FC_EN | B_BE_HCI_FC_CH12_EN),
            Op::Delay(10 * US),
        ],
    )
}

const PRELOAD: &[Op] = &[
    Op::Field32(
        R_AX_TXPKTCTL_B0_PRELD_CFG0,
        B_AX_B0_PRELD_USEMAXSZ_MASK,
        PRELD_B0_ENT_NUM * PRELD_AMSDU_SIZE,
    ),
    Op::Set32(R_AX_TXPKTCTL_B0_PRELD_CFG0, B_AX_B0_PRELD_FEN),
    Op::Field32(
        R_AX_TXPKTCTL_B0_PRELD_CFG1,
        B_AX_B0_PRELD_NXT_TXENDWIN_MASK,
        PRELD_NEXT_WND,
    ),
    Op::Field32(
        R_AX_TXPKTCTL_B0_PRELD_CFG1,
        B_AX_B0_PRELD_NXT_RSVMINSZ_MASK,
        PRELD_AMSDU_SIZE,
    ),
];

const DMAC: &[Op] = &[
    Op::Set8(R_BE_SS_CTRL, B_BE_SS_EN as u8),
    Op::PollSet(R_BE_SS_CTRL, B_BE_SS_INIT_DONE, INIT_NS),
    Op::Set32(R_BE_SS_CTRL, B_BE_WARM_INIT),
    Op::Clr32(R_BE_SS_CTRL, B_BE_BAND_TRIG_EN | B_BE_BAND1_TRIG_EN),
    Op::Set32(R_BE_MPDU_PROC, B_BE_APPEND_FCS),
    Op::Write32(R_BE_CUT_AMSDU_CTRL, TRXCFG_MPDU_PROC_CUT_CTRL),
    Op::Clr32(R_BE_HDR_SHCUT_SETTING, B_BE_TX_ADDR_MLD_TO_LIK),
    Op::Set32(
        R_BE_HDR_SHCUT_SETTING,
        B_BE_TX_HW_SEQ_EN | B_BE_TX_HW_ACK_POLICY_EN | B_BE_TX_MAC_MPDU_PROC_EN,
    ),
    Op::Write32(R_BE_RX_HDRTRNS, TRXCFG_MPDU_PROC_RX_HDR_CONV),
    Op::Field32(R_BE_DISP_FWD_WLAN_0, B_BE_FWD_WLAN_CPU_TYPE_0_DATA_MASK, 1),
    Op::Field32(R_BE_DISP_FWD_WLAN_0, B_BE_FWD_WLAN_CPU_TYPE_0_MNG_MASK, 1),
    Op::Field32(R_BE_DISP_FWD_WLAN_0, B_BE_FWD_WLAN_CPU_TYPE_0_CTL_MASK, 1),
    Op::Field32(R_BE_DISP_FWD_WLAN_0, B_BE_FWD_WLAN_CPU_TYPE_1_MASK, 1),
    Op::Set32(
        R_BE_SEC_ENG_CTRL,
        B_BE_CLK_EN_CGCMP
            | B_BE_CLK_EN_WAPI
            | B_BE_CLK_EN_WEP_TKIP
            | B_BE_SEC_TX_ENC
            | B_BE_SEC_RX_DEC
            | B_BE_MC_DEC
            | B_BE_BC_DEC
            | B_BE_BMC_MGNT_DEC
            | B_BE_UC_MGNT_DEC
            | B_BE_SEC_PRE_ENQUE_TX,
    ),
    Op::Set32(R_BE_SEC_MPDU_PROC, B_BE_APPEND_ICV | B_BE_APPEND_MIC),
    // Reserved MPDU metadata begins at the end of the 2688 linked PLE pages.
    Op::Field32(
        R_BE_TXPKTCTL_MPDUINFO_CFG,
        B_BE_MPDUINFO_PKTID_MASK,
        NORMAL_DLE.ple_linked,
    ),
    Op::Field32(
        R_BE_TXPKTCTL_MPDUINFO_CFG,
        B_BE_MPDUINFO_B1_BADDR_MASK,
        MPDU_INFO_B1_OFST,
    ),
    Op::Set32(R_BE_TXPKTCTL_MPDUINFO_CFG, B_BE_MPDUINFO_FEN),
    Op::Set32(R_BE_MLO_INIT_CTL, B_BE_MLO_TABLE_REINIT),
    Op::Clr32(R_BE_MLO_INIT_CTL, B_BE_MLO_TABLE_REINIT),
    Op::PollSet(R_BE_MLO_INIT_CTL, B_BE_MLO_TABLE_INIT_DONE, 1_000 * US),
    Op::Set32(R_BE_SS_CTRL, B_BE_MLO_HW_CHGLINK_EN),
    Op::Set32(R_BE_CMAC_SHARE_ACQCHK_CFG_0, B_BE_R_MACID_ACQ_CHK_EN),
];

const CMAC: &[Op] = &[
    Op::Write32(
        R_BE_HE_CTN_CHK_CCA_NAV,
        B_BE_HE_CTN_CHK_CCA_P20
            | B_BE_HE_CTN_CHK_EDCCA_P20
            | B_BE_HE_CTN_CHK_CCA_BITMAP
            | B_BE_HE_CTN_CHK_EDCCA_BITMAP
            | B_BE_HE_CTN_CHK_NO_GNT_WL
            | B_BE_HE_CTN_CHK_BASIC_NAV
            | B_BE_HE_CTN_CHK_INTRA_NAV
            | B_BE_HE_CTN_CHK_TX_NAV,
    ),
    Op::Write32(
        R_BE_HE_SIFS_CHK_CCA_NAV,
        B_BE_HE_SIFS_CHK_EDCCA_P20 | B_BE_HE_SIFS_CHK_EDCCA_BITMAP | B_BE_HE_SIFS_CHK_NO_GNT_WL,
    ),
    Op::Write32(
        R_BE_TB_CHK_CCA_NAV,
        B_BE_TB_CHK_EDCCA_BITMAP | B_BE_TB_CHK_NO_GNT_WL | B_BE_TB_CHK_BASIC_NAV,
    ),
    Op::Clr32(R_BE_CCA_CFG_0, B_BE_NO_GNT_WL_EN),
    Op::Field32(R_BE_PREBKF_CFG_0, B_BE_PREBKF_TIME_MASK, SCH_PREBKF_24US),
    Op::Field32(R_BE_CTN_CFG_0, B_BE_PREBKF_TIME_NONAC_MASK, SCH_PREBKF_24US),
    Op::Field32(R_BE_EDCA_BCNQ_PARAM, B_BE_BCNQ_CW_MASK, 0x32),
    Op::Field32(R_BE_EDCA_BCNQ_PARAM, B_BE_BCNQ_AIFS_MASK, BCN_IFS_25US),
    Op::Field32(
        R_BE_ADDR_CAM_CTRL,
        B_BE_ADDR_CAM_RANGE_MASK,
        ADDR_CAM_SERCH_RANGE,
    ),
    Op::Set32(R_BE_ADDR_CAM_CTRL, B_BE_ADDR_CAM_EN | B_BE_ADDR_CAM_CLR),
    Op::PollClear16(R_BE_ADDR_CAM_CTRL, B_BE_ADDR_CAM_CLR as u16, INIT_NS),
    // No unowned RX stream: firmware C2H remains available, on-air traffic
    // is dropped until the data-ring/scan owner admits it in W3/W4.
    Op::Write32(R_BE_MGNT_FLTR, RX_FLTR_FRAME_DROP_BE),
    Op::Write32(R_BE_CTRL_FLTR, RX_FLTR_FRAME_DROP_BE),
    Op::Write32(R_BE_DATA_FLTR, RX_FLTR_FRAME_DROP_BE),
    Op::Write32(R_BE_RX_FLTR_OPT, encode(15, B_BE_UID_FILTER_MASK)),
    Op::Write16(
        R_BE_PLCP_HDR_FLTR,
        (B_BE_HE_SIGB_CRC_CHK
            | B_BE_VHT_MU_SIGB_CRC_CHK
            | B_BE_VHT_SU_SIGB_CRC_CHK
            | B_BE_SIGA_CRC_CHK
            | B_BE_LSIG_PARITY_CHK_EN
            | B_BE_CCK_SIG_CHK
            | B_BE_CCK_CRC_CHK) as u16,
    ),
    Op::Clr32(R_BE_WMAC_NAV_CTL, B_BE_WMAC_PLCP_UP_NAV_EN),
    Op::Set32(
        R_BE_WMAC_NAV_CTL,
        B_BE_WMAC_TF_UP_NAV_EN | B_BE_WMAC_NAV_UPPER_EN,
    ),
    Op::Field32(R_BE_WMAC_NAV_CTL, B_BE_WMAC_NAV_UPPER_MASK, NAV_25MS),
    Op::Clr8(R_BE_RX_SR_CTRL, (B_BE_SR_EN | B_BE_SR_CTRL_PLCP_EN) as u8),
    Op::Set8(R_BE_BSSID_SRC_CTRL, B_BE_PLCP_SRC_EN as u8),
    Op::Clr32(R_BE_TB_PPDU_CTRL, B_BE_QOSNULL_UPD_MUEDCA_EN),
    Op::Field32(
        R_BE_WMTX_TCR_BE_4,
        B_BE_EHT_HE_PPDU_4XLTF_ZLD_USTIMER_MASK,
        0x12,
    ),
    Op::Field32(
        R_BE_WMTX_TCR_BE_4,
        B_BE_EHT_HE_PPDU_2XLTF_ZLD_USTIMER_MASK,
        0xe,
    ),
    Op::Field32(
        R_BE_MAC_LOOPBACK,
        B_BE_MACLBK_PLCP_DLY_MASK,
        S_BE_MACLBK_PLCP_DLY_DEF,
    ),
    Op::Clr32(R_BE_MAC_LOOPBACK, B_BE_MACLBK_EN),
    Op::Field32(
        R_BE_TRXPTCL_RESP_0,
        B_BE_WMAC_SPEC_SIFS_CCK_MASK,
        WMAC_SPEC_SIFS_CCK,
    ),
    Op::Field32(
        R_BE_TRXPTCL_RESP_0,
        B_BE_WMAC_SPEC_SIFS_OFDM_MASK,
        WMAC_SPEC_SIFS_OFDM_1115E,
    ),
    Op::Clr32(
        R_BE_WMAC_ACK_BA_RESP_LEGACY,
        B_BE_ACK_BA_RESP_LEGACY_CHK_EDCCA,
    ),
    Op::Clr32(R_BE_WMAC_ACK_BA_RESP_HE, B_BE_ACK_BA_RESP_HE_CHK_EDCCA),
    Op::Clr32(
        R_BE_WMAC_ACK_BA_RESP_EHT_LEG_PUNC,
        B_BE_ACK_BA_EHT_LEG_PUNC_CHK_EDCCA,
    ),
    Op::Set32(R_BE_RXTRIG_TEST_USER_2, B_BE_RXTRIG_FCSCHK_EN),
    Op::Clr32(
        R_BE_TRXPTCL_RESP_1,
        !(B_BE_FTM_RRSR_RATE_EN_MASK
            | B_BE_WMAC_RESP_DOPPLEB_BE_EN
            | B_BE_WMAC_RESP_DCM_EN
            | B_BE_WMAC_RESP_REF_RATE_MASK),
    ),
    Op::Field32(R_BE_TRXPTCL_RESP_1, B_BE_WMAC_RESP_REF_RATE_SEL, 0),
    Op::Clr32(
        R_BE_PTCL_RRSR1,
        !(B_BE_RRSR_RATE_EN_MASK | B_BE_RRSR_CCK_MASK | B_BE_RSC_MASK),
    ),
    Op::Clr32(
        R_BE_PTCL_RRSR0,
        !(B_BE_RRSR_OFDM_MASK | B_BE_RRSR_HT_MASK | B_BE_RRSR_VHT_MASK | B_BE_RRSR_HE_MASK),
    ),
    Op::Field32(
        R_BE_RESPBA_CAM_CTRL,
        B_BE_BACAM_RST_MASK,
        S_BE_BACAM_RST_ALL,
    ),
    Op::PollClear(R_BE_RESPBA_CAM_CTRL, B_BE_BACAM_RST_MASK, 1_000 * US),
    Op::Field16(
        R_BE_DLK_PROTECT_CTL,
        B_BE_RX_DLK_DATA_TIME_MASK as u16,
        TRXCFG_RMAC_DATA_TO as u16,
    ),
    Op::Field16(
        R_BE_DLK_PROTECT_CTL,
        B_BE_RX_DLK_CCA_TIME_MASK as u16,
        TRXCFG_RMAC_CCA_TO as u16,
    ),
    Op::Set16(R_BE_DLK_PROTECT_CTL, B_BE_RX_DLK_RST_EN as u16),
    // min(292 PLE pages, 127) * 128, capped to 11454+512 bytes, /512.
    Op::Field32(R_BE_RX_FLTR_OPT, B_BE_RX_MPDU_MAX_LEN_MASK, 23),
    Op::Clr8(R_BE_PLCP_HDR_FLTR, B_BE_VHT_SU_SIGB_CRC_CHK as u8),
    Op::Set16(R_BE_RCR, B_BE_BUSY_CHKSN as u16),
    Op::Set16(R_BE_RX_PLCP_EXT_OPTION_1, B_BE_PLCP_SU_PSDU_LEN_SRC as u16),
    Op::Field32(
        R_BE_RESP_CSI_RESERVED_PAGE,
        B_BE_CSI_RESERVED_START_PAGE_MASK,
        NORMAL_DLE.ple_linked + 2,
    ),
    Op::Field32(
        R_BE_RESP_CSI_RESERVED_PAGE,
        B_BE_CSI_RESERVED_PAGE_NUM_MASK,
        107,
    ),
    Op::Field32(R_BE_TX_SUB_BAND_VALUE, B_BE_TXSB_20M_MASK, S_BE_TXSB_20M_8),
    Op::Field32(R_BE_TX_SUB_BAND_VALUE, B_BE_TXSB_40M_MASK, S_BE_TXSB_40M_4),
    Op::Field32(R_BE_TX_SUB_BAND_VALUE, B_BE_TXSB_80M_MASK, S_BE_TXSB_80M_2),
    Op::Field32(
        R_BE_TX_SUB_BAND_VALUE,
        B_BE_TXSB_160M_MASK,
        S_BE_TXSB_160M_1,
    ),
    Op::Field32(
        R_BE_SIFS_SETTING,
        B_BE_HW_CTS2SELF_PKT_LEN_TH_MASK,
        S_AX_CTS2S_TH_1K,
    ),
    Op::Field32(
        R_BE_SIFS_SETTING,
        B_BE_HW_CTS2SELF_PKT_LEN_TH_TWW_MASK,
        S_AX_CTS2S_TH_SEC_256B,
    ),
    Op::Set32(R_BE_SIFS_SETTING, B_BE_HW_CTS2SELF_EN),
    Op::Field32(
        R_BE_PTCL_FSM_MON,
        B_BE_PTCL_TX_ARB_TO_THR_MASK,
        S_AX_PTCL_TO_2MS,
    ),
    Op::Clr32(R_BE_PTCL_FSM_MON, B_BE_PTCL_TX_ARB_TO_MODE),
    Op::Set8(
        R_BE_PTCL_COMMON_SETTING_0,
        (B_BE_CMAC_TX_MODE_0 | B_BE_CMAC_TX_MODE_1) as u8,
    ),
    Op::Clr8(
        R_BE_PTCL_COMMON_SETTING_0,
        (B_BE_PTCL_TRIGGER_SS_EN_0 | B_BE_PTCL_TRIGGER_SS_EN_1 | B_BE_PTCL_TRIGGER_SS_EN_UL) as u8,
    ),
    Op::Field32(
        R_BE_AMPDU_AGG_LIMIT,
        B_BE_AMPDU_MAX_TIME_MASK,
        AMPDU_MAX_TIME,
    ),
    Op::Field32(
        R_BE_RX_CTRL_1,
        B_BE_RXDMA_TXRPT_QUEUE_ID_SW_MASK,
        WLCPU_RXCH2_QID,
    ),
    Op::Field32(
        R_BE_RX_CTRL_1,
        B_BE_RXDMA_F2PCMDRPT_QUEUE_ID_SW_MASK,
        WLCPU_RXCH2_QID,
    ),
];

const REPORTS: &[Op] = &[
    Op::Field32(R_BE_WDRLS_CFG, B_BE_WDRLS_MODE_MASK, 0), // RTW89_RPR_MODE_POH
    Op::Field32(
        R_BE_RLSRPT0_CFG1,
        B_BE_RLSRPT0_FLTR_MAP_MASK,
        S_BE_WDRLS_FLTR_TXOK
            | S_BE_WDRLS_FLTR_RTYLMT
            | S_BE_WDRLS_FLTR_LIFTIM
            | S_BE_WDRLS_FLTR_MACID,
    ),
    Op::Field32(R_BE_RLSRPT0_CFG1, B_BE_RLSRPT0_AGGNUM_MASK, 30),
    Op::Field32(R_BE_RLSRPT0_CFG1, B_BE_RLSRPT0_TO_MASK, 255),
    Op::Clr32(R_BE_RSP_CHK_SIG, B_BE_RSP_STATIC_RTS_CHK_SERV_BW_EN),
];
