/*
 * Realtek RTL8922AE (rtw89, BE generation) — the register subset this
 * driver core programs, copied line by line from the Linux v6.17 rtw89
 * headers (drivers/net/wireless/realtek/rtw89/{reg.h,pci.h,mac.h,fw.h,
 * txrx.h,efuse_be.c}); the upstream file and line each define came from is
 * on the line. There is no public datasheet for this part, so the Linux
 * header IS the register reference — cited, never copied wholesale: only
 * what the bring-up ladder touches is here. Composite masks upstream builds
 * from other defines (the DLE ready pairs, the DMA stop/busy sets, the
 * needed-element bitmap) are hand-written with their derivation in
 * src/regs/mod.rs, not transcribed.
 *
 * Regenerate src/regs/be.rs from this file — never edit that file:
 *   python3 drivers/ivpu/tools/gen_ivpu_regs.py --strict \
 *     --source 'Realtek RTL8922AE (BE generation) register subset, copied from Linux v6.17 drivers/net/wireless/realtek/rtw89 with the upstream line cited per define ({header}).' \
 *     --about 'RTL8922AE BE-generation register offsets and field masks over the `Regs` seam (BAR2 read32/write32); names are verbatim from the Linux rtw89 headers so a reviewer can diff them 1:1 against the cited upstream lines.' \
 *     drivers/net/rtw89/tools/rtw89_be.h > drivers/net/rtw89/src/regs/be.rs
 */
/* SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause */
/*
 * rtw89_be_used.h -- register / constant subset used by DOSSIER.md
 *
 * Target: RTL8922AE, PCI 10ec:8922 (and 10ec:892B "-VS"), chip gen
 * RTW89_CHIP_BE, driver rtw8922a.  Every line below is copied verbatim from
 * the pinned Linux v6.17 rtw89 sources; the trailing comment gives file:line.
 *
 * Scope is only what the dossier uses: claim BAR2, power the MAC on, read the
 * efuse RF map, download and start firmware, and exchange one H2C/C2H pair
 * over the PCI DMA rings.
 */

#ifndef __RTW89_BE_USED_H__
#define __RTW89_BE_USED_H__


/* ====================================================================== */
/* 1. SYS / POWER-ON  (rtw8922a_pwr_on_func, rtw8922a.c:267-411) */
/* ====================================================================== */

#define R_BE_SYS_ISO_CTRL 0x0000 /* reg.h:3776 */
#define B_BE_PWC_EV2EF_B BIT(15) /* reg.h:3777 */
#define B_BE_PWC_EV2EF_S BIT(14) /* reg.h:3778 */
#define B_BE_ISO_EB2CORE BIT(8) /* reg.h:3784 */
#define R_BE_SYS_PW_CTRL 0x0004 /* reg.h:3793 */
#define B_BE_DIS_WLBT_PDNSUSEN_SOPC BIT(18) /* reg.h:3807 */
#define B_BE_RDY_SYSPWR BIT(17) /* reg.h:3808 */
#define B_BE_EN_WLON BIT(16) /* reg.h:3809 */
#define B_BE_APDM_HPDN BIT(15) /* reg.h:3810 */
#define B_BE_PSUS_OFF_CAPC_EN BIT(14) /* reg.h:3811 */
#define B_BE_AFSM_PCIE_SUS_EN BIT(12) /* reg.h:3812 */
#define B_BE_AFSM_WLSUS_EN BIT(11) /* reg.h:3813 */
#define B_BE_APFM_SWLPS BIT(10) /* reg.h:3814 */
#define B_BE_APFN_ONMAC BIT(8) /* reg.h:3816 */
#define R_BE_SYS_CLK_CTRL 0x0008 /* reg.h:3820 */
#define B_BE_CPU_CLK_EN BIT(14) /* reg.h:3821 */
#define R_BE_SYS_WL_EFUSE_CTRL 0x000A /* reg.h:3830 */
#define B_BE_AUTOLOAD_SUS BIT(5) /* reg.h:3837 */
#define R_BE_SYS_PAGE_CLK_GATED 0x000C /* reg.h:3840 */
#define B_BE_CPHY_POWER_READY_CHK BIT(24) /* reg.h:3844 */
#define B_BE_PCIE_PRST_DEBUNC_PERIOD_MASK GENMASK(23, 22) /* reg.h:3845 */
#define B_BE_SYM_PRST_DEBUNC_SEL BIT(21) /* reg.h:3846 */
#define B_BE_CPHY_AUXCLK_OP BIT(20) /* reg.h:3847 */
#define B_BE_SOP_OFFPOOBS_PC BIT(18) /* reg.h:3849 */
#define R_BE_ANAPAR_POW_MAC 0x0016 /* reg.h:3869 */
#define B_BE_POW_PC_LDO_PORT1 BIT(3) /* reg.h:3870 */
#define B_BE_POW_PC_LDO_PORT0 BIT(2) /* reg.h:3871 */
#define R_BE_SYS_ADIE_PAD_PWR_CTRL 0x0018 /* reg.h:3875 */
#define B_BE_SYM_PADPDN_WL_RFC1_1P3 BIT(6) /* reg.h:3876 */
#define B_BE_SYM_PADPDN_WL_RFC0_1P3 BIT(5) /* reg.h:3877 */
#define R_BE_RSV_CTRL 0x001C /* reg.h:3879 */
#define B_BE_R_DIS_PRST BIT(6) /* reg.h:3884 */
#define B_BE_WLOCK_1C_BIT6 BIT(5) /* reg.h:3885 */
#define R_BE_SYS_SDIO_CTRL 0x0070 /* reg.h:3994 */
#define B_BE_PCIE_DIS_L2__CTRL_LDO_HCI BIT(15) /* reg.h:4003 */
#define B_BE_PCIE_DIS_L2_RTK_PERST BIT(14) /* reg.h:4004 */
#define B_BE_PCIE_FORCE_IBX_EN BIT(12) /* reg.h:4006 */
#define R_BE_HCI_OPT_CTRL 0x0074 /* reg.h:4020 */
#define B_BE_HCI_WLAN_IO_ST BIT(31) /* reg.h:4021 */
#define B_BE_HAXIDMA_IO_ST BIT(27) /* reg.h:4023 */
#define B_BE_HAXIDMA_BACKUP_RESTORE_ST BIT(26) /* reg.h:4024 */
#define B_BE_HAXIDMA_IO_EN BIT(24) /* reg.h:4025 */
#define B_BE_HCI_WLAN_IO_EN BIT(28) /* reg.h:4022 */
#define R_BE_FEN_RST_ENABLE 0x0084 /* reg.h:4073 */
#define B_BE_R_SYM_ISO_ADDA_P12PP BIT(21) /* reg.h:4080 */
#define B_BE_R_SYM_ISO_ADDA_P02PP BIT(20) /* reg.h:4081 */
#define B_BE_BOOT_RDY1 BIT(10) /* reg.h:4086 */
#define B_BE_BOOT_RDY0 BIT(2) /* reg.h:4091 */
#define B_BE_FEN_BB1_IP_RSTN BIT(9) /* reg.h:4087 */
#define B_BE_FEN_BB1PLAT_RSTB BIT(8) /* reg.h:4088 */
#define B_BE_FEN_BB_IP_RSTN BIT(1) /* reg.h:4092 */
#define B_BE_FEN_BBPLAT_RSTB BIT(0) /* reg.h:4093 */
#define B_BE_CMAC1_FEN BIT(17) /* reg.h:4082 */
#define R_BE_PLATFORM_ENABLE 0x0088 /* reg.h:4095 */
#define B_BE_HOLD_AFTER_RESET BIT(11) /* reg.h:4096 */
#define B_BE_WCPU_EN BIT(1) /* reg.h:4105 */
#define B_BE_PLATFORM_EN BIT(0) /* reg.h:4106 */
#define R_BE_WLLPS_CTRL 0x0090 /* reg.h:4108 */
#define B_BE_DIS_WLBT_LPSEN_LOPC BIT(1) /* reg.h:4127 */
#define R_BE_WLRESUME_CTRL 0x0094 /* reg.h:4130 */
#define B_BE_LPSROP_CMAC1 BIT(13) /* reg.h:4144 */
#define B_BE_LPSROP_CMAC0 BIT(12) /* reg.h:4145 */
#define R_BE_PMC_DBG_CTRL2 0x00CC /* reg.h:4169 */
#define B_BE_SYSON_DIS_PMCR_BE_WRMSK BIT(2) /* reg.h:4179 */
#define R_BE_MEM_PWR_CTRL 0x00D0 /* reg.h:4182 */
#define B_BE_MEM_BBMCU0_DS_V1 BIT(17) /* reg.h:4196 */
#define R_BE_AFE_ON_CTRL1 0x0244 /* reg.h:4377 */
#define B_BE_REG_CK_MON_CK960M_EN BIT(28) /* reg.h:4379 */
#define R_BE_IC_PWR_STATE 0x03F0 /* reg.h:4427 */
#define B_BE_WHOLE_SYS_PWR_STE_MASK GENMASK(25, 16) /* reg.h:4428 */
#define B_BE_WLMAC_PWR_STE_MASK GENMASK(9, 8) /* reg.h:4430 */
#define MAC_AX_SYS_ACT 0x220 /* reg.h:4429 */
#define R_AX_IC_PWR_STATE 0x03F0 /* reg.h:313 */
#define B_AX_WLMAC_PWR_STE_MASK GENMASK(9, 8) /* reg.h:315 */
#define R_AX_SYS_CFG1 0x00F0 /* reg.h:181 */
#define B_AX_CHIP_VER_MASK GENMASK(15, 12) /* reg.h:182 */
#define R_AX_SCOREBOARD  0x00AC /* reg.h:155 */
#define MAC_AX_NOTIFY_TP_MAJOR 0x81 /* reg.h:160 */

/* ====================================================================== */
/* 2. EFUSE (efuse_be.c) */
/* ====================================================================== */

#define R_BE_EFUSE_CTRL 0x0030 /* reg.h:3953 */
#define B_BE_EF_RDY BIT(29) /* reg.h:3955 */
#define B_BE_EF_ADDR_MASK GENMASK(15, 0) /* reg.h:3957 */
#define R_BE_EFUSE_CTRL_1_V1 0x0034 /* reg.h:3959 */
#define R_BE_EFUSE_CTRL_2_V1 0x00A4 /* reg.h:4153 */
#define B_BE_R_SYM_AUTOLOAD_WITH_PMC_SEL BIT(24) /* reg.h:4158 */
#define B_BE_EF_BURST BIT(19) /* reg.h:4160 */
#define R_BE_WL_BT_PWR_CTRL 0x0068 /* reg.h:3971 */
#define B_BE_BT_DISN_EN BIT(16) /* reg.h:3982 */
#define RTW89_EFUSE_BLOCK_ID_MASK GENMASK(31, 16) /* efuse.h:10 */
#define RTW89_EFUSE_BLOCK_SIZE_MASK GENMASK(15, 0) /* efuse.h:11 */
#define RTW89_EFUSE_MAX_BLOCK_SIZE 0x10000 /* efuse.h:12 */
#define TSSI_CCK_CH_GROUP_NUM 6 /* core.h:5340 */
#define TSSI_MCS_2G_CH_GROUP_NUM 5 /* core.h:5341 */
#define TSSI_MCS_5G_CH_GROUP_NUM 14 /* core.h:5342 */
#define TSSI_MCS_6G_CH_GROUP_NUM 32 /* core.h:5343 */

/* ====================================================================== */
/* 3. CPU / FIRMWARE DOWNLOAD CONTROL (mac_be.c, fw.c) */
/* ====================================================================== */

#define R_BE_WCPU_FW_CTRL 0x01E0 /* reg.h:4318 */
#define B_BE_RUN_ENV_MASK GENMASK(31, 30) /* reg.h:4319 */
#define B_BE_WCPU_FWDL_STATUS_MASK GENMASK(29, 26) /* reg.h:4320 */
#define B_BE_WDT_PLT_RST_EN BIT(17) /* reg.h:4321 */
#define B_BE_BBMCU1_FWDL_EN BIT(12) /* reg.h:4324 */
#define B_BE_BBMCU0_FWDL_EN BIT(11) /* reg.h:4325 */
#define B_BE_DATACPU_FWDL_EN BIT(10) /* reg.h:4326 */
#define B_BE_WLANCPU_FWDL_EN BIT(9) /* reg.h:4327 */
#define B_BE_WCPU_ROM_CUT_GET BIT(8) /* reg.h:4328 */
#define B_BE_H2C_PATH_RDY BIT(1) /* reg.h:4331 */
#define B_BE_DLFW_PATH_RDY BIT(0) /* reg.h:4332 */
#define R_BE_BOOT_REASON 0x01E6 /* reg.h:4334 */
#define B_BE_BOOT_REASON_MASK GENMASK(2, 0) /* reg.h:4335 */
#define R_BE_HALT_H2C_CTRL 0x0160 /* reg.h:4218 */
#define R_BE_HALT_C2H_CTRL 0x0164 /* reg.h:4221 */
#define R_BE_HALT_H2C 0x0168 /* reg.h:4224 */
#define R_BE_HALT_C2H 0x016C /* reg.h:4227 */
#define R_AX_HALT_H2C_CTRL 0x0160 /* reg.h:194 */
#define R_AX_HALT_C2H_CTRL 0x0164 /* reg.h:197 */
#define R_BE_SYS_CFG5 0x0170 /* reg.h:4231 */
#define B_BE_WDT_WAKE_PCIE_EN BIT(10) /* reg.h:4234 */
#define B_BE_WDT_WAKE_USB_EN BIT(9) /* reg.h:4235 */
#define R_BE_SECURE_BOOT_MALLOC_INFO 0x0184 /* reg.h:4240 */
#define R_BE_UDM0 0x01F0 /* reg.h:4341 */
#define R_BE_UDM1 0x01F4 /* reg.h:4353 */
#define B_BE_UDM1_HALMAC_C2H_ENQ_CNT_MASK GENMASK(15, 12) /* reg.h:4355 */
#define B_BE_UDM1_HALMAC_H2C_DEQ_CNT_MASK GENMASK(11, 8) /* reg.h:4356 */
#define R_BE_UDM2 0x01F8 /* reg.h:4360 */
#define R_BE_DCPU_PLATFORM_ENABLE 0x0888 /* reg.h:4440 */
#define B_BE_DCPU_PLATFORM_EN BIT(0) /* reg.h:4447 */
#define R_BE_WLCPU_PORT_PC 0x03FC /* reg.h:4436 */
#define R_AX_WCPU_FW_CTRL 0x01E0 /* reg.h:200 */
#define R_AX_BOOT_DBG 0x83F0 /* reg.h:530 */
#define R_BE_HIMR0 0x01A0 /* reg.h:4264 */
#define B_BE_WDT_TIMEOUT_INT_EN BIT(22) /* reg.h:4267 */
#define B_BE_HALT_C2H_INT_EN BIT(21) /* reg.h:4268 */
#define R_BE_HISR0 0x01A4 /* reg.h:4291 */
#define B_BE_WDT_TIMEOUT_INT BIT(22) /* reg.h:4294 */
#define B_BE_HALT_C2H_INT BIT(21) /* reg.h:4295 */
#define FWDL_WAIT_CNT 400000 /* fw.h:4701 */

/* ====================================================================== */
/* 4. H2C/C2H REGISTER MAILBOX (rtw89_fw_msg_reg; used by phycap read) */
/* ====================================================================== */

#define R_BE_H2CREG_DATA0 0x7140 /* reg.h:4579 */
#define R_BE_H2CREG_DATA1 0x7144 /* reg.h:4580 */
#define R_BE_H2CREG_DATA2 0x7148 /* reg.h:4581 */
#define R_BE_H2CREG_DATA3 0x714C /* reg.h:4582 */
#define R_BE_C2HREG_DATA0 0x7150 /* reg.h:4583 */
#define R_BE_C2HREG_DATA1 0x7154 /* reg.h:4584 */
#define R_BE_C2HREG_DATA2 0x7158 /* reg.h:4585 */
#define R_BE_C2HREG_DATA3 0x715C /* reg.h:4586 */
#define R_BE_H2CREG_CTRL 0x7160 /* reg.h:4587 */
#define R_BE_C2HREG_CTRL 0x7164 /* reg.h:4589 */
#define B_AX_H2CREG_TRIGGER BIT(0) /* reg.h:521 */
#define RTW89_H2CREG_MAX 4 /* fw.h:113 */
#define RTW89_C2HREG_MAX 4 /* fw.h:114 */
#define RTW89_H2CREG_HDR_LEN 2 /* fw.h:116 */
#define RTW89_C2HREG_HDR_LEN 2 /* fw.h:115 */
#define RTW89_H2CREG_HDR_FUNC_MASK GENMASK(6, 0) /* fw.h:97 */
#define RTW89_H2CREG_HDR_LEN_MASK GENMASK(11, 8) /* fw.h:98 */
#define RTW89_C2HREG_HDR_FUNC_MASK GENMASK(6, 0) /* fw.h:25 */
#define RTW89_C2HREG_HDR_LEN_MASK GENMASK(11, 8) /* fw.h:27 */
#define RTW89_C2H_TIMEOUT 1000000 /* fw.h:117 */

/* ====================================================================== */
/* 5. DMAC / CMAC FUNCTION ENABLE, DLE, HFC (mac_be.c) */
/* ====================================================================== */

#define R_BE_HCI_FUNC_EN 0x7880 /* reg.h:4592 */
#define B_BE_HCI_RXDMA_EN BIT(1) /* reg.h:4595 */
#define B_BE_HCI_TXDMA_EN BIT(0) /* reg.h:4596 */
#define R_AX_HCI_FUNC_EN 0x8380 /* reg.h:526 */
#define B_AX_HCI_RXDMA_EN BIT(1) /* reg.h:527 */
#define B_AX_HCI_TXDMA_EN BIT(0) /* reg.h:528 */
#define R_BE_DMAC_FUNC_EN 0x8400 /* reg.h:4601 */
#define B_BE_MAC_FUNC_EN BIT(30) /* reg.h:4603 */
#define B_BE_DMAC_FUNC_EN BIT(29) /* reg.h:4604 */
#define B_BE_MPDU_PROC_EN BIT(28) /* reg.h:4605 */
#define B_BE_WD_RLS_EN BIT(27) /* reg.h:4606 */
#define B_BE_DLE_WDE_EN BIT(26) /* reg.h:4607 */
#define B_BE_TXPKT_CTRL_EN BIT(25) /* reg.h:4608 */
#define B_BE_STA_SCH_EN BIT(24) /* reg.h:4609 */
#define B_BE_DLE_PLE_EN BIT(23) /* reg.h:4610 */
#define B_BE_PKT_BUF_EN BIT(22) /* reg.h:4611 */
#define B_BE_DMAC_TBL_EN BIT(21) /* reg.h:4612 */
#define B_BE_PKT_IN_EN BIT(20) /* reg.h:4613 */
#define B_BE_DLE_CPUIO_EN BIT(19) /* reg.h:4614 */
#define B_BE_DISPATCHER_EN BIT(18) /* reg.h:4615 */
#define B_BE_BBRPT_EN BIT(17) /* reg.h:4616 */
#define B_BE_MAC_SEC_EN BIT(16) /* reg.h:4617 */
#define B_BE_H_AXIDMA_EN BIT(14) /* reg.h:4619 */
#define B_BE_DMAC_MLO_EN BIT(11) /* reg.h:4620 */
#define B_BE_PLRLS_EN BIT(10) /* reg.h:4621 */
#define B_BE_P_AXIDMA_EN BIT(9) /* reg.h:4622 */
#define B_BE_DLE_DATACPUIO_EN BIT(8) /* reg.h:4623 */
#define B_BE_LTR_CTL_EN BIT(7) /* reg.h:4624 */
#define R_BE_DMAC_CLK_EN 0x8404 /* reg.h:4626 */
#define B_BE_DLE_WDE_CLK_EN BIT(26) /* reg.h:4631 */
#define B_BE_DLE_PLE_CLK_EN BIT(23) /* reg.h:4634 */
#define R_BE_DMAC_TABLE_CTRL 0x8420 /* reg.h:4669 */
#define B_BE_DMAC_ADDR_MODE BIT(12) /* reg.h:4672 */
#define R_BE_DMAC_SYS_CR32B 0x842C /* reg.h:4682 */
#define B_BE_DMAC_BB_PHY1_MASK GENMASK(31, 16) /* reg.h:4683 */
#define B_BE_DMAC_BB_PHY0_MASK GENMASK(15, 0) /* reg.h:4684 */
#define R_BE_WDE_PKTBUF_CFG 0x8C08 /* reg.h:5178 */
#define B_BE_WDE_FREE_PAGE_NUM_MASK GENMASK(28, 16) /* reg.h:5179 */
#define B_BE_WDE_START_BOUND_MASK GENMASK(14, 8) /* reg.h:5180 */
#define B_BE_WDE_PAGE_SEL_MASK GENMASK(1, 0) /* reg.h:5181 */
#define R_BE_WDE_QTA0_CFG 0x8C40 /* reg.h:5274 */
#define R_BE_WDE_QTA1_CFG 0x8C44 /* reg.h:5278 */
#define R_BE_WDE_QTA2_CFG 0x8C48 /* reg.h:5282 */
#define R_BE_WDE_QTA3_CFG 0x8C4C /* reg.h:5286 */
#define R_BE_WDE_QTA4_CFG 0x8C50 /* reg.h:5290 */
#define B_BE_WDE_Q0_MAX_SIZE_MASK GENMASK(27, 16) /* reg.h:5275 */
#define B_BE_WDE_Q0_MIN_SIZE_MASK GENMASK(11, 0) /* reg.h:5276 */
#define R_BE_PLE_PKTBUF_CFG 0x9008 /* reg.h:5299 */
#define B_BE_PLE_FREE_PAGE_NUM_MASK GENMASK(28, 16) /* reg.h:5300 */
#define B_BE_PLE_START_BOUND_MASK GENMASK(14, 8) /* reg.h:5301 */
#define B_BE_PLE_PAGE_SEL_MASK GENMASK(1, 0) /* reg.h:5302 */
#define R_BE_PLE_QTA0_CFG 0x9040 /* reg.h:5395 */
#define R_BE_PLE_QTA12_CFG 0x9070 /* reg.h:5443 */
#define B_BE_PLE_Q0_MAX_SIZE_MASK GENMASK(27, 16) /* reg.h:5396 */
#define B_BE_PLE_Q0_MIN_SIZE_MASK GENMASK(11, 0) /* reg.h:5397 */
#define R_AX_WDE_INI_STATUS 0x8D00 /* reg.h:1364 */
#define B_AX_WDE_Q_MGN_INI_RDY BIT(1) /* reg.h:1365 */
#define B_AX_WDE_BUF_MGN_INI_RDY BIT(0) /* reg.h:1366 */
#define R_AX_PLE_INI_STATUS 0x9100 /* reg.h:1572 */
#define B_AX_PLE_Q_MGN_INI_RDY BIT(1) /* reg.h:1573 */
#define B_AX_PLE_BUF_MGN_INI_RDY BIT(0) /* reg.h:1574 */
#define RTW89_WDE_PG_64		64 /* mac.h:593 */
#define S_AX_WDE_PAGE_SEL_64	0 /* mac.h:597 */
#define RTW89_PLE_PG_128	128 /* mac.h:602 */
#define S_AX_PLE_PAGE_SEL_128	1 /* mac.h:606 */
#define R_BE_HCI_FC_CTRL 0xB700 /* reg.h:6031 */
#define B_BE_HCI_FC_CH12_FULL_COND_MASK GENMASK(11, 10) /* reg.h:6035 */
#define B_BE_HCI_FC_WP_CH811_FULL_COND_MASK GENMASK(9, 8) /* reg.h:6036 */
#define B_BE_HCI_FC_WP_CH07_FULL_COND_MASK GENMASK(7, 6) /* reg.h:6037 */
#define B_BE_HCI_FC_WD_FULL_COND_MASK GENMASK(5, 4) /* reg.h:6038 */
#define B_BE_HCI_FC_CH12_EN BIT(3) /* reg.h:6039 */
#define B_BE_HCI_FC_MODE_MASK GENMASK(2, 1) /* reg.h:6040 */
#define B_BE_HCI_FC_EN BIT(0) /* reg.h:6041 */
#define R_BE_CH_PAGE_CTRL 0xB704 /* reg.h:6043 */
#define B_BE_PREC_PAGE_CH12_V1_MASK GENMASK(21, 16) /* reg.h:6044 */
#define B_BE_PREC_PAGE_CH011_V1_MASK GENMASK(5, 0) /* reg.h:6045 */
#define R_BE_PUB_PAGE_CTRL1 0xB790 /* reg.h:6060 */
#define B_BE_PUBPG_G1_MASK GENMASK(28, 16) /* reg.h:6061 */
#define B_BE_PUBPG_G0_MASK GENMASK(12, 0) /* reg.h:6062 */
#define R_BE_PUB_PAGE_CTRL2 0xB794 /* reg.h:6064 */
#define B_BE_PUBPG_ALL_MASK GENMASK(12, 0) /* reg.h:6065 */
#define R_BE_WP_PAGE_CTRL1 0xB7A4 /* reg.h:6074 */
#define B_BE_PREC_PAGE_WP_CH811_MASK GENMASK(24, 16) /* reg.h:6075 */
#define B_BE_PREC_PAGE_WP_CH07_MASK GENMASK(8, 0) /* reg.h:6076 */
#define R_BE_WP_PAGE_CTRL2 0xB7A8 /* reg.h:6078 */
#define B_BE_WP_THRD_MASK GENMASK(12, 0) /* reg.h:6079 */
#define R_BE_CMAC_SHARE_FUNC_EN 0x0E000 /* reg.h:6087 */
#define B_BE_CMAC_SHARE_EN BIT(30) /* reg.h:6089 */
#define B_BE_RESPBA_EN BIT(2) /* reg.h:6093 */
#define B_BE_ADDRSRCH_EN BIT(1) /* reg.h:6094 */
#define B_BE_BTCOEX_EN BIT(0) /* reg.h:6095 */
#define R_BE_CMAC_FUNC_EN 0x10000 /* reg.h:6138 */
#define B_BE_CMAC_EN BIT(30) /* reg.h:6141 */
#define B_BE_CMAC_TXEN BIT(29) /* reg.h:6142 */
#define B_BE_CMAC_RXEN BIT(28) /* reg.h:6143 */
#define B_BE_TXTIME_EN BIT(8) /* reg.h:6155 */
#define B_BE_RESP_PKTCTL_EN BIT(7) /* reg.h:6156 */
#define B_BE_SIGB_EN BIT(6) /* reg.h:6157 */
#define B_BE_PHYINTF_EN BIT(5) /* reg.h:6158 */
#define B_BE_CMAC_DMA_EN BIT(4) /* reg.h:6159 */
#define B_BE_PTCLTOP_EN BIT(3) /* reg.h:6160 */
#define B_BE_SCHEDULER_EN BIT(2) /* reg.h:6161 */
#define B_BE_TMAC_EN BIT(1) /* reg.h:6162 */
#define B_BE_RMAC_EN BIT(0) /* reg.h:6163 */
#define R_BE_WDRLS_CFG 0x9408 /* reg.h:5466 */
#define B_BE_WDRLS_MODE_MASK GENMASK(1, 0) /* reg.h:5470 */
#define R_BE_RLSRPT0_CFG1 0x9444 /* reg.h:5504 */
#define B_BE_RLSRPT0_FLTR_MAP_MASK GENMASK(27, 24) /* reg.h:5505 */
#define B_BE_RLSRPT0_TO_MASK GENMASK(23, 16) /* reg.h:5510 */
#define B_BE_RLSRPT0_AGGNUM_MASK GENMASK(7, 0) /* reg.h:5511 */
#define S_BE_WDRLS_FLTR_TXOK 1 /* reg.h:5506 */
#define S_BE_WDRLS_FLTR_RTYLMT 2 /* reg.h:5507 */
#define S_BE_WDRLS_FLTR_LIFTIM 4 /* reg.h:5508 */
#define S_BE_WDRLS_FLTR_MACID 8 /* reg.h:5509 */
#define R_BE_RSP_CHK_SIG 0x11000 /* reg.h:7084 */
#define B_BE_RSP_STATIC_RTS_CHK_SERV_BW_EN BIT(30) /* reg.h:7086 */
#define RTW89_MAC_BE_BAND_REG_OFFSET 0x4000 /* mac.h:583 */
#define RTW89_BBMCU_ADDR_OFFSET	0x30000 /* phy.h:10 */

/* ====================================================================== */
/* 6. HAXI / PCIe DMA ENGINE (pci_be.c, pci.h) */
/* ====================================================================== */

#define R_BE_HAXI_INIT_CFG1 0xB000 /* reg.h:5954 */
#define B_BE_CFG_WD_PERIOD_IDLE_MASK GENMASK(31, 28) /* reg.h:5955 */
#define B_BE_CFG_WD_PERIOD_ACTIVE_MASK GENMASK(27, 24) /* reg.h:5956 */
#define B_BE_SET_BDRAM_BOUND BIT(16) /* reg.h:5960 */
#define B_BE_RXQ_RXBD_MODE_MASK GENMASK(12, 11) /* reg.h:5963 */
#define B_BE_DMA_MODE_MASK GENMASK(10, 8) /* reg.h:5964 */
#define S_BE_DMA_MOD_PCIE_NO_DATA_CPU 0x0 /* reg.h:5965 */
#define B_BE_MAX_RXDMA_MASK GENMASK(3, 2) /* reg.h:5973 */
#define B_BE_MAX_TXDMA_MASK GENMASK(1, 0) /* reg.h:5974 */
#define B_BE_STOP_AXI_MST BIT(7) /* reg.h:5969 */
#define B_BE_RXDMA_EN BIT(5) /* reg.h:5971 */
#define B_BE_TXDMA_EN BIT(4) /* reg.h:5972 */
#define R_BE_HAXI_MST_WDT_TIMEOUT_SEL_V1 0xB02C /* reg.h:5994 */
#define B_BE_HAXI_MST_WDT_TIMEOUT_SEL_MASK GENMASK(4, 0) /* reg.h:5995 */
#define R_BE_HAXI_DMA_STOP1 0xB010 /* reg.h:5976 */
#define B_BE_STOP_WPDMA BIT(31) /* reg.h:5977 */
#define B_BE_STOP_CH14 BIT(14) /* reg.h:5978 */
#define B_BE_STOP_CH13 BIT(13) /* reg.h:5979 */
#define B_BE_STOP_CH12 BIT(12) /* reg.h:5980 */
#define B_BE_STOP_CH11 BIT(11) /* reg.h:5981 */
#define B_BE_STOP_CH0 BIT(0) /* reg.h:5992 */
/* pci.h:763 */
#define R_BE_TXBD_RWPTR_CLR1 0xB014 /* pci.h:988 */
#define B_BE_CLR_CH14_IDX BIT(14) /* pci.h:989 */
#define B_BE_CLR_CH0_IDX BIT(0) /* pci.h:1003 */
#define R_BE_RXBD_RWPTR_CLR1_V1 0xB018 /* pci.h:1005 */
#define B_BE_CLR_RPQ0_IDX BIT(1) /* pci.h:1010 */
#define B_BE_CLR_RXQ0_IDX BIT(0) /* pci.h:1011 */
#define R_BE_HAXI_DMA_BUSY1 0xB01C /* pci.h:1013 */
#define B_BE_HAXI_MST_BUSY BIT(31) /* pci.h:1014 */
#define B_BE_RPQ0_BUSY_V1 BIT(17) /* pci.h:1021 */
#define B_BE_RXQ0_BUSY_V1 BIT(16) /* pci.h:1022 */
/* pci.h:1039 */
#define R_BE_HAXI_EXP_CTRL_V1 0xB020 /* pci.h:1045 */
#define B_BE_MAX_TAG_NUM_MASK GENMASK(3, 0) /* pci.h:1049 */
#define R_BE_RX_APPEND_MODE 0x8920 /* pci.h:984 */
#define B_BE_APPEND_LEN_MASK GENMASK(15, 0) /* pci.h:986 */
#define R_BE_WP_ADDR_H_SEL0_3_V1 0xB420 /* pci.h:845 */

/* ====================================================================== */
/* 7. BE TX/RX RING REGISTERS (rtw89_pci_ch_dma_addr_set_be, pci.c:1034-1055) */
/* ====================================================================== */

#define R_BE_CH0_TXBD_NUM_V1 0xB030 /* pci.h:771 */
#define R_BE_CH12_TXBD_NUM_V1 0xB048 /* pci.h:783 */
#define R_BE_RXQ0_RXBD_NUM_V1 0xB050 /* pci.h:787 */
#define R_BE_RPQ0_RXBD_NUM_V1 0xB052 /* pci.h:788 */
#define R_BE_CH0_TXBD_IDX_V1 0xB100 /* pci.h:790 */
#define R_BE_CH12_TXBD_IDX_V1 0xB130 /* pci.h:802 */
#define R_BE_RXQ0_RXBD_IDX_V1 0xB160 /* pci.h:806 */
#define R_BE_RPQ0_RXBD_IDX_V1 0xB164 /* pci.h:807 */
#define R_BE_CH0_TXBD_DESA_L_V1 0xB200 /* pci.h:809 */
#define R_BE_CH0_TXBD_DESA_H_V1 0xB204 /* pci.h:810 */
#define R_BE_CH12_TXBD_DESA_L_V1 0xB260 /* pci.h:833 */
#define R_BE_CH12_TXBD_DESA_H_V1 0xB264 /* pci.h:834 */
#define R_BE_RXQ0_RXBD_DESA_L_V1 0xB300 /* pci.h:840 */
#define R_BE_RXQ0_RXBD_DESA_H_V1 0xB304 /* pci.h:841 */
#define R_BE_RPQ0_RXBD_DESA_L_V1 0xB308 /* pci.h:842 */
#define R_BE_RPQ0_RXBD_DESA_H_V1 0xB30C /* pci.h:843 */
#define TXBD_HW_IDX_MASK	GENMASK(27, 16) /* pci.h:519 */
#define TXBD_HOST_IDX_MASK	GENMASK(11, 0) /* pci.h:520 */

/* ====================================================================== */
/* 8. INTERRUPTS ON BE (pci.c recognize/enable/disable _v2) */
/* ====================================================================== */

#define R_BE_PCIE_HIMR0 0x30B0 /* pci.h:336 */
#define B_BE_HCI_AXIDMA_INT_EN0 BIT(5) /* pci.h:352 */
#define B_BE_HS1_IND_INT_EN0 BIT(1) /* pci.h:356 */
#define B_BE_HS0_IND_INT_EN0 BIT(0) /* pci.h:357 */
#define R_BE_PCIE_HISR 0x30B4 /* pci.h:359 */
#define B_BE_HCI_AXIDMA_INT BIT(5) /* pci.h:367 */
#define B_BE_HS1ISR_IND_INT BIT(1) /* pci.h:371 */
#define B_BE_HS0ISR_IND_INT BIT(0) /* pci.h:372 */
#define R_BE_PCIE_DMA_IMR_0_V1 0x30B8 /* pci.h:374 */
#define B_BE_PCIE_RX_RPQ0_IMR0_V1 BIT(17) /* pci.h:381 */
#define B_BE_PCIE_RX_RX0P2_IMR0_V1 BIT(16) /* pci.h:382 */
#define R_BE_PCIE_DMA_ISR 0x30BC /* pci.h:399 */
#define B_BE_PCIE_RX_RPQ0_ISR_V1 BIT(17) /* pci.h:406 */
#define B_BE_PCIE_RX_RX0P2_ISR_V1 BIT(16) /* pci.h:407 */
#define R_BE_HAXI_HIMR00 0xB0B0 /* pci.h:424 */
#define B_BE_RDU_CH1_INT_IMR_V1 BIT(26) /* pci.h:429 */
#define B_BE_RDU_CH0_INT_IMR_V1 BIT(25) /* pci.h:430 */
#define R_BE_HAXI_HISR00 0xB0B4 /* pci.h:457 */
#define B_BE_RDU_CH1_INT_V1 BIT(26) /* pci.h:462 */
#define B_BE_RDU_CH0_INT_V1 BIT(25) /* pci.h:463 */

/* ====================================================================== */
/* 9. PCIe LINK / LTR / MIT / SER (pci_be.c) */
/* ====================================================================== */

/* enum members, not #define:                                              */
/*   PCIE_L1DLY_16US      = 4          pci.h:1112                           */
/*   PCIE_CLKDLY_HW_V1_0  = 0          pci.h:1128                           */
/*   MAC_AX_PCIE_DISABLE  = 0          pci.h:1222                           */
/*   MAC_AX_PCIE_ENABLE   = 1          pci.h:1223                           */
/*   MAC_AX_PCIE_IGNORE   = 0xFF       pci.h:1225                           */
/*   MAC_AX_BD_TRUNC      = 1          pci.h:1138                           */
/*   MAC_AX_RXBD_PKT      = 0          pci.h:1143                           */
/*   MAC_AX_TAG_MULTI     = 1          pci.h:1150                           */
/*   MAC_AX_TX_BURST_V1_256B = 2       pci.h:1162                           */
/*   MAC_AX_RX_BURST_V1_128B = 1       pci.h:1174                           */
/*   MAC_AX_WD_DMA_INTVL_256NS = 1     pci.h:1182                           */
/*   MAC_AX_TAG_NUM_8     = 7          pci.h:1202                           */
/*   MAC_AX_LBC_TMR_2MS   = 8          pci.h:1215                           */
/*   MAC_AX_IO_RCY_ANA_TMR_DEF = 0xFE  pci.h:1232                           */
/*   PCIE_RXBD_NORM       = 0          pci_be.c:12                          */
/* pci_be.c:17-20 timer scale constants:                                    */
/*   PL0_TMR_SCALE_ASIC 1 ; PL0_TMR_ANA_172US 0x800 ;                       */
/*   PL0_TMR_MAC_1MS 0x27100 ; PL0_TMR_AUX_1MS 0x1E848                      */
#define R_BE_PCIE_FRZ_CLK 0x3004 /* pci.h:273 */
#define B_BE_PCIE_FRZ_REG_RST BIT(26) /* pci.h:279 */
#define B_BE_PCIE_EN_AUX_CLK BIT(0) /* pci.h:296 */
#define R_BE_PCIE_PS_CTRL 0x3008 /* pci.h:298 */
#define B_BE_CMAC_EXIT_L1_EN BIT(7) /* pci.h:300 */
#define R_BE_PCIE_MIX_CFG 0x300C /* pci.h:309 */
#define B_BE_ASPM_CTRL_L1 BIT(17) /* pci.h:311 */
#define B_BE_RTK_ASPM_CTRL_MASK GENMASK(17, 16) /* pci.h:313 */
#define B_BE_L1SUB_ENABLE BIT(0) /* pci.h:318 */
#define R_BE_L1_CLK_CTRL 0x3010 /* pci.h:320 */
#define B_BE_CLK_PM_EN BIT(0) /* pci.h:323 */
#define R_BE_PCIE_LAT_CTRL 0x3044 /* pci.h:325 */
#define B_BE_SYM_AUX_CLK_SEL BIT(15) /* pci.h:329 */
#define B_BE_CLK_REQ_LAT_MASK GENMASK(7, 4) /* pci.h:332 */
#define R_BE_L1_2_CTRL_HCILDO 0x3110 /* pci.h:922 */
#define B_BE_PCIE_DIS_L1_2_CTRL_HCILDO BIT(0) /* pci.h:923 */
#define R_BE_PCIE_HRPWM 0x30C0 /* pci.h:919 */
#define R_BE_PCIE_CRPWM 0x30C4 /* pci.h:920 */
#define R_BE_PL1_DBG_INFO 0x3120 /* pci.h:925 */
#define R_BE_FWS1IMR 0x0198 /* reg.h:4242 */
#define B_BE_PCIE_SER_TIMEOUT_INDIC_EN BIT(21) /* reg.h:4245 */
#define R_BE_SER_PL1_CTRL 0x34A8 /* pci.h:968 */
#define B_BE_PL1_SER_PL1_EN BIT(31) /* pci.h:969 */
#define B_BE_PL1_TIMER_UNIT_MASK GENMASK(19, 17) /* pci.h:971 */
#define R_BE_REG_PL1_MASK 0x34B0 /* pci.h:974 */
#define B_BE_SER_PCLKREQ_ACK_MASK BIT(5) /* pci.h:975 */
#define B_BE_SER_PM_CLK_MASK BIT(4) /* pci.h:976 */
#define B_BE_SER_LTSSM_IMR BIT(3) /* pci.h:977 */
#define B_BE_SER_PM_MASTER_IMR BIT(2) /* pci.h:978 */
#define B_BE_SER_L1SUB_IMR BIT(1) /* pci.h:979 */
#define B_BE_SER_PMU_IMR BIT(0) /* pci.h:980 */
#define R_BE_REG_PL1_ISR 0x34B4 /* pci.h:982 */
#define R_BE_PCIE_MIT0_TMR 0x3330 /* pci.h:929 */
#define B_BE_PCIE_MIT0_RX_TMR_MASK GENMASK(5, 4) /* pci.h:930 */
#define BE_MIT0_TMR_UNIT_1MS 0 /* pci.h:931 */
#define R_BE_PCIE_MIT0_CNT 0x3334 /* pci.h:937 */
#define B_BE_PCIE_RX_MIT0_CNT_MASK GENMASK(31, 24) /* pci.h:938 */
#define B_BE_PCIE_RX_MIT0_TMR_CNT_MASK GENMASK(15, 8) /* pci.h:940 */
#define R_BE_PCIE_MIT_CH_EN 0x3338 /* pci.h:943 */
#define R_BE_LTR_CTRL_0 0x8410 /* reg.h:4648 */
#define B_BE_LTR_HW_EN BIT(0) /* reg.h:4653 */
#define R_BE_LTR_CFG_0 0x8414 /* reg.h:4655 */
#define B_BE_LTR_IDX_IDLE_MASK GENMASK(15, 14) /* reg.h:4657 */
#define B_BE_LTR_IDX_ACTIVE_MASK GENMASK(13, 12) /* reg.h:4658 */
#define B_BE_LTR_IDLE_TIMER_IDX_MASK GENMASK(10, 8) /* reg.h:4651 */
#define R_BE_LTR_CFG_1 0x8418 /* reg.h:4665 */
#define B_BE_LTR_CMAC1_RX_USE_PG_TH_MASK GENMASK(27, 16) /* reg.h:4666 */
#define B_BE_LTR_CMAC0_RX_USE_PG_TH_MASK GENMASK(11, 0) /* reg.h:4667 */
#define R_BE_LTR_DECISION_CTRL_V1 0x3610 /* reg.h:4557 */
#define B_BE_ENABLE_LTR_CTL_DECISION BIT(31) /* reg.h:4558 */
#define B_BE_LTR_EN_PORT_V1_MASK GENMASK(6, 4) /* reg.h:4568 */
#define B_BE_LTR_IDX_DISABLE_V1_MASK GENMASK(9, 8) /* reg.h:4567 */
#define B_BE_LTR_HW_DEC_EN_V1 BIT(4) /* reg.h:4571 */
#define B_BE_LTR_SPACE_IDX_MASK GENMASK(1, 0) /* reg.h:4572 */
#define R_BE_LTR_LATENCY_IDX0_V1 0x3614 /* reg.h:4574 */
#define R_BE_LTR_LATENCY_IDX1_V1 0x3618 /* reg.h:4575 */
#define R_BE_LTR_LATENCY_IDX3_V1 0x3620 /* reg.h:4577 */
#define PCI_LTR_IDLE_TIMER_3_2MS 7 /* reg.h:573 */
#define PCI_LTR_SPC_500US 2 /* reg.h:580 */
#define R_BE_AON_WDT_TMR 0x306C /* reg.h:4497 */
#define R_BE_MDIO_WDT_TMR 0x3090 /* reg.h:4498 */
#define R_BE_LA_MODE_WDT_TMR 0x309C /* reg.h:4499 */
#define R_BE_WDT_AR_TMR 0x3144 /* reg.h:4500 */
#define R_BE_WDT_AW_TMR 0x3150 /* reg.h:4501 */
#define R_BE_WLAN_WDT_TMR 0x3054 /* reg.h:4502 */
#define R_BE_WDT_W_TMR 0x315C /* reg.h:4503 */
#define R_BE_AXIDMA_WDT_TMR 0x3060 /* reg.h:4504 */
#define R_BE_WDT_B_TMR 0x3164 /* reg.h:4505 */
#define R_BE_WDT_R_TMR 0x316C /* reg.h:4506 */
#define R_BE_LOCAL_WDT_TMR 0x3084 /* reg.h:4507 */
#define RTW89_PCIE_ASPM_CTRL		0x070F /* pci.h:1077 */
#define RTW89_L1DLY_MASK		GENMASK(5, 3) /* pci.h:1078 */
#define BIT_WAKE_CTRL_V1 BIT(23) /* reg.h:129 */
#define R_RAC_DIRECT_OFFSET_BE_LANE0_G1 0x3800 /* pci.h:143 */
#define R_RAC_DIRECT_OFFSET_BE_LANE0_G2 0x3900 /* pci.h:145 */
#define RAC_MULT 2 /* pci.h:141 */

/* ====================================================================== */
/* 10. PCI DESCRIPTOR / RING CONSTANTS (pci.h) */
/* ====================================================================== */

/* RTW89_PCI_ADDR_NUM(x) ((x) & GENMASK(5, 0))                pci.h:1399   */
#define RTW89_PCI_TXBD_NUM_MAX		256 /* pci.h:1051 */
#define RTW89_PCI_RXBD_NUM_MAX		256 /* pci.h:1052 */
#define RTW89_PCI_TXWD_NUM_MAX		512 /* pci.h:1053 */
#define RTW89_PCI_TXWD_PAGE_SIZE	128 /* pci.h:1054 */
#define RTW89_PCI_ADDRINFO_MAX		4 /* pci.h:1055 */
#define RTW89_PCI_MULTITAG		8 /* pci.h:1060 */
#define RTW89_PCI_TXBD_OPT_LS		BIT(14) /* pci.h:1382 */
#define RTW89_PCI_TXBD_OPT_DMA_HI	GENMASK(13, 6) /* pci.h:1383 */
#define RTW89_PCI_TXWP_VALID		BIT(15) /* pci.h:1387 */
#define RTW89_PCI_ADDR_MSDU_LS		BIT(15) /* pci.h:1396 */
#define RTW89_PCI_ADDR_LS		BIT(14) /* pci.h:1397 */
#define RTW89_PCI_ADDR_HIGH_MASK	GENMASK(13, 6) /* pci.h:1398 */
#define RTW89_TXADDR_INFO_NR_V1		10 /* pci.h:1407 */
#define B_PCIADDR_LEN_V1_MASK		GENMASK(10, 0) /* pci.h:1411 */
#define B_PCIADDR_HIGH_SEL_V1_MASK	GENMASK(14, 11) /* pci.h:1412 */
#define B_PCIADDR_LS_V1_MASK		BIT(15) /* pci.h:1413 */
#define RTW89_PCI_RPP_POLLUTED		BIT(31) /* pci.h:1419 */
#define RTW89_PCI_RPP_SEQ		GENMASK(30, 16) /* pci.h:1420 */
#define RTW89_PCI_RPP_TX_STATUS		GENMASK(15, 13) /* pci.h:1421 */
#define RTW89_PCI_RPP_QSEL		GENMASK(12, 8) /* pci.h:1426 */
#define RTW89_PCI_RPP_MACID		GENMASK(7, 0) /* pci.h:1427 */
#define RTW89_PCI_RXBD_OPT_DMA_HI	GENMASK(13, 6) /* pci.h:1436 */
#define RTW89_PCI_RXBD_FS		BIT(15) /* pci.h:1440 */
#define RTW89_PCI_RXBD_LS		BIT(14) /* pci.h:1441 */
#define RTW89_PCI_RXBD_WRITE_SIZE	GENMASK(13, 0) /* pci.h:1442 */
#define RTW89_PCI_RXBD_TAG		GENMASK(28, 16) /* pci.h:1443 */
#define RTW89_RX_TAG_MAX		0x1fff /* pci.h:1483 */
#define RTW89_R32_EA		0xEAEAEAEA /* mac.h:576 */
#define RTW89_R32_DEAD		0xDEADBEEF /* mac.h:577 */
#define MAC_REG_POOL_COUNT	10 /* mac.h:578 */

/* ====================================================================== */
/* 11. H2C / C2H PACKET HEADERS AND FWDL CONSTANTS (fw.h) */
/* ====================================================================== */

/* H2C_FUNC_OFLD_CFG = 0x14   (enum rtw89_fw_ofld_h2c_func)   fw.h:4245    */
/* H2C_OFLD_CFG_LEN  = 8                                      fw.c:4363    */
/* H2C_LOG_CFG_LEN   = 12                                     fw.c:2514    */
/* RTW89_C2H_CAT_TEST=0, RTW89_C2H_CAT_MAC=1,                              */
/*   RTW89_C2H_CAT_OUTSRC=2                                   fw.h:167-169 */
/* RTW89_MAC_C2H_CLASS_INFO      = 0x0                        mac.h:466    */
/* RTW89_MAC_C2H_FUNC_REC_ACK    = 0                          mac.h:418    */
/* RTW89_MAC_C2H_FUNC_DONE_ACK   = 1                          mac.h:419    */
/* RTW89_MAC_C2H_FUNC_C2H_LOG    = 2                          mac.h:420    */
/* RTW89_CORE_RX_TYPE_C2H  = 10                               core.h:198   */
/* RTW89_CORE_RX_TYPE_H2C  = 13                               core.h:201   */
/* RTW89_CORE_RX_TYPE_FWDL = 14                               core.h:202   */
/* RTW89_TXCH_CH12 = 12 (FW CMD)                              txrx.h:644   */
/* RTW89_RXCH_RXQ  = 0 ; RTW89_RXCH_RPQ = 1                   txrx.h:652-3 */
/* RTW89_DMA_H2C   = 12                                       core.h:3819  */
#define H2C_HEADER_LEN			8 /* fw.h:4151 */
#define H2C_HDR_CAT			GENMASK(1, 0) /* fw.h:4152 */
#define H2C_HDR_CLASS			GENMASK(7, 2) /* fw.h:4153 */
#define H2C_HDR_FUNC			GENMASK(15, 8) /* fw.h:4154 */
#define H2C_HDR_DEL_TYPE		GENMASK(19, 16) /* fw.h:4155 */
#define H2C_HDR_H2C_SEQ			GENMASK(31, 24) /* fw.h:4156 */
#define H2C_HDR_TOTAL_LEN		GENMASK(13, 0) /* fw.h:4157 */
#define H2C_HDR_REC_ACK			BIT(14) /* fw.h:4158 */
#define H2C_HDR_DONE_ACK		BIT(15) /* fw.h:4159 */
#define FWCMD_TYPE_H2C			0 /* fw.h:4161 */
#define H2C_CAT_MAC		0x1 /* fw.h:4169 */
#define H2C_CL_FW_INFO			0x0 /* fw.h:4172 */
#define H2C_FUNC_LOG_CFG		0x0 /* fw.h:4173 */
#define H2C_CL_MAC_FWDL		0x3 /* fw.h:4215 */
#define H2C_FUNC_MAC_FWHDR_DL		0x0 /* fw.h:4216 */
#define H2C_CL_MAC_FW_OFLD		0x9 /* fw.h:4239 */
#define RTW89_C2H_HDR_W0_CATEGORY GENMASK(1, 0) /* fw.h:3565 */
#define RTW89_C2H_HDR_W0_CLASS GENMASK(7, 2) /* fw.h:3566 */
#define RTW89_C2H_HDR_W0_FUNC GENMASK(15, 8) /* fw.h:3567 */
#define RTW89_C2H_HDR_W1_LEN GENMASK(13, 0) /* fw.h:3568 */
#define RTW89_C2H_DONE_ACK_W2_CAT GENMASK(1, 0) /* fw.h:3592 */
#define RTW89_C2H_DONE_ACK_W2_CLASS GENMASK(7, 2) /* fw.h:3593 */
#define RTW89_C2H_DONE_ACK_W2_FUNC GENMASK(15, 8) /* fw.h:3594 */
#define RTW89_C2H_DONE_ACK_W2_H2C_RETURN GENMASK(23, 16) /* fw.h:3595 */
#define RTW89_C2H_DONE_ACK_W2_H2C_SEQ GENMASK(31, 24) /* fw.h:3597 */
#define FWDL_SECTION_MAX_NUM 10 /* fw.h:280 */
#define FWDL_SECTION_CHKSUM_LEN	8 /* fw.h:281 */
#define FWDL_SECTION_PER_PKT_LEN 2020 /* fw.h:282 */
#define FWDL_SECURITY_SECTION_TYPE 9 /* fw.h:540 */
#define FWDL_SECURITY_SIGLEN 512 /* fw.h:541 */
#define FWDL_SECURITY_CHKSUM_LEN 8 /* fw.h:542 */
#define FWSECTION_HDR_W0_DL_ADDR GENMASK(31, 0) /* fw.h:562 */
#define FWSECTION_HDR_W1_SECTIONTYPE GENMASK(27, 24) /* fw.h:564 */
#define FWSECTION_HDR_W1_SEC_SIZE GENMASK(23, 0) /* fw.h:565 */
#define FWSECTION_HDR_W1_CHECKSUM BIT(28) /* fw.h:566 */
#define FWSECTION_HDR_W1_REDL BIT(29) /* fw.h:567 */
#define FW_HDR_W1_MAJOR_VERSION GENMASK(7, 0) /* fw.h:583 */
#define FW_HDR_W1_MINOR_VERSION GENMASK(15, 8) /* fw.h:584 */
#define FW_HDR_W1_SUBVERSION GENMASK(23, 16) /* fw.h:585 */
#define FW_HDR_W1_SUBINDEX GENMASK(31, 24) /* fw.h:586 */
#define FW_HDR_W3_LEN GENMASK(23, 16) /* fw.h:588 */
#define FW_HDR_W3_HDR_VER GENMASK(31, 24) /* fw.h:589 */
#define FW_HDR_W6_SEC_NUM GENMASK(15, 8) /* fw.h:595 */
#define FW_HDR_W7_PART_SIZE GENMASK(15, 0) /* fw.h:596 */
#define FW_HDR_W7_DYN_HDR BIT(16) /* fw.h:597 */
#define FW_HDR_W7_IDMEM_SHARE_MODE GENMASK(21, 18) /* fw.h:598 */
#define FW_HDR_W7_CMD_VERSERION GENMASK(31, 24) /* fw.h:599 */
#define FWSECTION_HDR_V1_W0_DL_ADDR GENMASK(31, 0) /* fw.h:608 */
#define FWSECTION_HDR_V1_W1_SECTIONTYPE GENMASK(27, 24) /* fw.h:610 */
#define FWSECTION_HDR_V1_W1_SEC_SIZE GENMASK(23, 0) /* fw.h:611 */
#define FWSECTION_HDR_V1_W1_CHECKSUM BIT(28) /* fw.h:612 */
#define FWSECTION_HDR_V1_W1_REDL BIT(29) /* fw.h:613 */
#define FWSECTION_HDR_V1_W2_MSSC GENMASK(7, 0) /* fw.h:614 */
#define FORMATTED_MSSC 0xFF /* fw.h:615 */
#define FORMATTED_MSSC_MASK GENMASK(7, 0) /* fw.h:616 */
#define FWSECTION_HDR_V1_W2_BBMCU_IDX GENMASK(27, 24) /* fw.h:617 */
#define FW_HDR_V1_W1_MAJOR_VERSION GENMASK(7, 0) /* fw.h:635 */
#define FW_HDR_V1_W3_CMD_VERSERION GENMASK(23, 16) /* fw.h:640 */
#define FW_HDR_V1_W3_HDR_VER GENMASK(31, 24) /* fw.h:641 */
#define FW_HDR_V1_W5_YEAR GENMASK(15, 0) /* fw.h:646 */
#define FW_HDR_V1_W5_HDR_SIZE GENMASK(31, 16) /* fw.h:647 */
#define FW_HDR_V1_W6_SEC_NUM GENMASK(15, 8) /* fw.h:648 */
#define FW_HDR_V1_W6_DSP_CHKSUM BIT(24) /* fw.h:649 */
#define FW_HDR_V1_W7_PART_SIZE GENMASK(15, 0) /* fw.h:650 */
#define FW_HDR_V1_W7_DYN_HDR BIT(16) /* fw.h:651 */
#define FW_HDR_V1_W7_IDMEM_SHARE_MODE GENMASK(21, 18) /* fw.h:652 */
#define RTW89_MFW_SIG	0xFF /* fw.h:3903 */
#define RTW89_FW_ELEMENT_ALIGN 16 /* fw.h:3935 */
/* fw.h:3991 */

/* ====================================================================== */
/* 12. XTAL_SI SERIAL INTERFACE (mac_be.c:376-425, used by pwr_on_func)   */
/* ====================================================================== */

#define R_BE_WLAN_XTAL_SI_CTRL 0x0270 /* reg.h:4419 */
#define B_BE_WL_XTAL_SI_CMD_POLL BIT(31) /* reg.h:4420 */
#define B_BE_WL_XTAL_SI_CHIPID_MASK GENMASK(30, 28) /* reg.h:4421 */
#define B_BE_WL_XTAL_SI_MODE_MASK GENMASK(25, 24) /* reg.h:4422 */
#define B_BE_WL_XTAL_SI_BITMASK_MASK GENMASK(23, 16) /* reg.h:4423 */
#define B_BE_WL_XTAL_SI_DATA_MASK GENMASK(15, 8) /* reg.h:4424 */
#define B_BE_WL_XTAL_SI_ADDR_MASK GENMASK(7, 0) /* reg.h:4425 */
#define XTAL_SI_NORMAL_WRITE 0x00 /* reg.h:260 */
#define XTAL_SI_NORMAL_READ 0x01 /* reg.h:261 */
/* xtal_si offsets used by rtw8922a_pwr_on_func (enum rtw89_mac_xtal_si_offset): */
/*   XTAL_SI_XREF_RF1   = 0x2D      mac.h:1515                             */
/*   XTAL_SI_XREF_RF2   = 0x2E      mac.h:1516                             */
/*   XTAL_SI_WL_RFC_S0  = 0x80      mac.h:1526                             */
/*   XTAL_SI_WL_RFC_S1  = 0x81      mac.h:1529                             */
/*   XTAL_SI_ANAPAR_WL  = 0x90      mac.h:1532                             */
/*   XTAL_SI_PLL        = 0xE0      mac.h:1545                             */
/*   XTAL_SI_PLL_1      = 0xE1      mac.h:1546                             */

/* ---- D1 ladder additions: power-off, the write-pointer lanes, the xtal_si
 * offsets (enum rtw89_mac_xtal_si_offset members transcribed as defines),
 * the register mailbox, the BE receive descriptor, the efuse record header,
 * secure-boot recognition, the MSS key pool, the PCIe watchdog timers. ---- */
#define B_BE_XTAL_OFF_A_DIE BIT(22) /* reg.h:3803 */
#define B_BE_APFM_OFFMAC BIT(9) /* reg.h:3815 */
#define FWSECTION_HDR_W2_MSSC GENMASK(31, 0) /* fw.h:567 */
#define R_BE_WP_ADDR_H_SEL4_7_V1 0xB424 /* pci.h:846 */
#define R_BE_WP_ADDR_H_SEL8_11_V1 0xB428 /* pci.h:847 */
#define R_BE_WP_ADDR_H_SEL12_15_V1 0xB42C /* pci.h:848 */
#define XTAL_SI_XREF_RF1 0x2D /* mac.h:1515 */
#define XTAL_SI_XREF_RF2 0x2E /* mac.h:1516 */
#define XTAL_SI_WL_RFC_S0 0x80 /* mac.h:1526 */
#define XTAL_SI_WL_RFC_S1 0x81 /* mac.h:1529 */
#define XTAL_SI_ANAPAR_WL 0x90 /* mac.h:1532 */
#define XTAL_SI_PLL 0xE0 /* mac.h:1545 */
#define XTAL_SI_PLL_1 0xE1 /* mac.h:1546 */
#define RTW89_H2CREG_GET_FEATURE_PART_NUM GENMASK(23, 16) /* fw.h:111 */
#define RTW89_FWCMD_H2CREG_FUNC_GET_FEATURE 3 /* fw.h:144 */
#define RTW89_FWCMD_C2HREG_FUNC_PHY_CAP 3 /* fw.h:158 */
#define RTW89_C2HREG_HDR_ACK BIT(7) /* fw.h:27 */
#define RTW89_C2HREG_HDR_SEQ_MASK GENMASK(15, 12) /* fw.h:29 */
#define RTW89_C2HREG_PHYCAP_W0_RX_NSS GENMASK(23, 16) /* fw.h:41 */
#define RTW89_C2HREG_PHYCAP_W0_BW GENMASK(31, 24) /* fw.h:42 */
#define RTW89_C2HREG_PHYCAP_W1_TX_NSS GENMASK(7, 0) /* fw.h:43 */
#define RTW89_C2HREG_PHYCAP_W1_PROT GENMASK(15, 8) /* fw.h:44 */
#define RTW89_C2HREG_PHYCAP_W1_NIC GENMASK(23, 16) /* fw.h:45 */
#define RTW89_C2HREG_PHYCAP_W1_WL_FUNC GENMASK(31, 24) /* fw.h:46 */
#define RTW89_C2HREG_PHYCAP_W2_HW_TYPE GENMASK(7, 0) /* fw.h:47 */
#define RTW89_C2HREG_PHYCAP_W3_ANT_TX_NUM GENMASK(15, 8) /* fw.h:48 */
#define RTW89_C2HREG_PHYCAP_W3_ANT_RX_NUM GENMASK(23, 16) /* fw.h:49 */
#define RTW89_C2HREG_PHYCAP_W3_BAND_SEL GENMASK(31, 24) /* fw.h:50 */
#define RTW89_CORE_RX_TYPE_C2H 10 /* core.h:198 */
#define RTW89_CORE_RX_TYPE_H2C 13 /* core.h:201 */
#define RTW89_CORE_RX_TYPE_FWDL 14 /* core.h:202 */
#define BE_RXD_RPKT_LEN_MASK GENMASK(13, 0) /* txrx.h:469 */
#define BE_RXD_SHIFT_MASK GENMASK(15, 14) /* txrx.h:470 */
#define BE_RXD_DRV_INFO_SZ_MASK GENMASK(19, 18) /* txrx.h:471 */
#define BE_RXD_HDR_CNV_SZ_MASK GENMASK(21, 20) /* txrx.h:472 */
#define BE_RXD_PHY_RPT_SZ_MASK GENMASK(23, 22) /* txrx.h:473 */
#define BE_RXD_RPKT_TYPE_MASK GENMASK(29, 24) /* txrx.h:474 */
#define BE_RXD_BB_SEL BIT(30) /* txrx.h:475 */
#define BE_RXD_LONG_RXD BIT(31) /* txrx.h:476 */
#define EFUSE_EXTERNALPN_ADDR_BE 0x1580 /* efuse_be.c:10 */
#define EFUSE_SERIALNUM_ADDR_BE 0x1581 /* efuse_be.c:11 */
#define EFUSE_SB_CRYP_SEL_ADDR 0x1582 /* efuse_be.c:12 */
#define EFUSE_SB_CRYP_SEL_DEFAULT 0xFFFF /* efuse_be.c:14 */
#define EFUSE_SEC_BE_START 0x1580 /* efuse_be.c:16 */
#define EFUSE_SEC_BE_SIZE 4 /* efuse_be.c:17 */
#define EFUSE_HDR_PAGE_MASK GENMASK(19, 17) /* efuse_be.c:196 */
#define EFUSE_HDR_OFFSET_MASK GENMASK(16, 4) /* efuse_be.c:197 */
#define EFUSE_HDR_WORD_EN_MASK GENMASK(3, 0) /* efuse_be.c:199 */
#define EFUSE_B1_MSSDEVTYPE_MASK GENMASK(3, 0) /* efuse.c:14 */
#define EFUSE_B1_MSSCUSTIDX0_MASK GENMASK(7, 4) /* efuse.c:15 */
#define EFUSE_B2_MSSKEYNUM_MASK GENMASK(3, 0) /* efuse.c:16 */
#define EFUSE_B2_MSSCUSTIDX1_MASK BIT(6) /* efuse.c:17 */
#define MSS_DEV_TYPE_FWSEC_DEF 0xF /* efuse.c:39 */
#define MSS_DEV_TYPE_FWSEC_WINLIN_INBOX 0xC /* efuse.c:40 */
#define MSS_DEV_TYPE_FWSEC_NONLIN_INBOX_NON_COB 0xA /* efuse.c:41 */
#define MSS_DEV_TYPE_FWSEC_NONLIN_INBOX_COB 0x9 /* efuse.c:42 */
#define MSS_DEV_TYPE_FWSEC_NONWIN_INBOX 0x6 /* efuse.c:43 */
#define RTW89_FW_MSS_DEV_TYPE_FWSEC_DEF 0xF /* core.h:4687 */
#define RTW89_FW_MSS_DEV_TYPE_FWSEC_INV 0xFF /* core.h:4688 */
#define MSS_POOL_RMP_TBL_BITMASK 0x0 /* fw.h:655 */
#define FWDL_MSS_POOL_DEFKEYSETS_SIZE 8 /* fw.h:659 */
#define PL0_TMR_SCALE_ASIC 1 /* pci_be.c:17 */
#define PL0_TMR_ANA_172US 0x800 /* pci_be.c:18 */
#define PL0_TMR_MAC_1MS 0x27100 /* pci_be.c:19 */
#define PL0_TMR_AUX_1MS 0x1E848 /* pci_be.c:20 */


/* W1 normal MAC register facts. */
#define ADDR_CAM_SERCH_RANGE  0x7f /* reg.h:7434 */
#define AMPDU_MAX_TIME 0x9E /* reg.h:6623 */
#define BCN_IFS_25US 0x19 /* reg.h:6324 */
#define B_AX_B0_PRELD_FEN BIT(31) /* reg.h:1958 */
#define B_AX_B0_PRELD_NXT_RSVMINSZ_MASK GENMASK(7, 0) /* reg.h:1968 */
#define B_AX_B0_PRELD_NXT_TXENDWIN_MASK GENMASK(11, 8) /* reg.h:1966 */
#define B_AX_B0_PRELD_USEMAXSZ_MASK GENMASK(25, 16) /* reg.h:1959 */
#define B_AX_GRP BIT(31) /* reg.h:1077 */
#define B_AX_MAX_PG_MASK GENMASK(28, 16) /* reg.h:1075 */
#define B_AX_MIN_PG_MASK GENMASK(12, 0) /* reg.h:1076 */
#define B_BE_ACK_BA_EHT_LEG_PUNC_CHK_EDCCA BIT(1) /* reg.h:7358 */
#define B_BE_ACK_BA_RESP_HE_CHK_EDCCA BIT(1) /* reg.h:7338 */
#define B_BE_ACK_BA_RESP_LEGACY_CHK_EDCCA BIT(1) /* reg.h:7318 */
#define B_BE_ADDR_CAM_CLR BIT(8) /* reg.h:7438 */
#define B_BE_ADDR_CAM_EN BIT(0) /* reg.h:7441 */
#define B_BE_ADDR_CAM_RANGE_MASK GENMASK(23, 16) /* reg.h:7433 */
#define B_BE_AMPDU_MAX_TIME_MASK GENMASK(31, 24) /* reg.h:6622 */
#define B_BE_APPEND_FCS BIT(0) /* reg.h:5673 */
#define B_BE_APPEND_ICV BIT(1) /* reg.h:5752 */
#define B_BE_APPEND_MIC BIT(0) /* reg.h:5753 */
#define B_BE_BACAM_RST_MASK GENMASK(1, 0) /* reg.h:7454 */
#define B_BE_BAND1_TRIG_EN BIT(9) /* reg.h:5928 */
#define B_BE_BAND_TRIG_EN BIT(28) /* reg.h:5918 */
#define B_BE_BCNQ_AIFS_MASK GENMASK(23, 16) /* reg.h:6323 */
#define B_BE_BCNQ_CW_MASK GENMASK(31, 24) /* reg.h:6322 */
#define B_BE_BC_DEC BIT(2) /* reg.h:5740 */
#define B_BE_BMC_MGNT_DEC BIT(5) /* reg.h:5737 */
#define B_BE_BUSY_CHKSN BIT(15) /* reg.h:7363 */
#define B_BE_CCK_CRC_CHK BIT(0) /* reg.h:7396 */
#define B_BE_CCK_SIG_CHK BIT(1) /* reg.h:7395 */
#define B_BE_CLK_EN_CGCMP BIT(10) /* reg.h:5734 */
#define B_BE_CLK_EN_WAPI BIT(9) /* reg.h:5735 */
#define B_BE_CLK_EN_WEP_TKIP BIT(8) /* reg.h:5736 */
#define B_BE_CMAC_TX_MODE_0 BIT(0) /* reg.h:6608 */
#define B_BE_CMAC_TX_MODE_1 BIT(1) /* reg.h:6607 */
#define B_BE_CSI_RESERVED_PAGE_NUM_MASK GENMASK(27, 16) /* reg.h:7556 */
#define B_BE_CSI_RESERVED_START_PAGE_MASK GENMASK(11, 0) /* reg.h:7557 */
#define B_BE_EHT_HE_PPDU_2XLTF_ZLD_USTIMER_MASK GENMASK(20, 16) /* reg.h:7080 */
#define B_BE_EHT_HE_PPDU_4XLTF_ZLD_USTIMER_MASK GENMASK(28, 24) /* reg.h:7079 */
#define B_BE_FTM_RRSR_RATE_EN_MASK GENMASK(28, 24) /* reg.h:7119 */
#define B_BE_FWD_WLAN_CPU_TYPE_0_CTL_MASK GENMASK(5, 4) /* reg.h:5174 */
#define B_BE_FWD_WLAN_CPU_TYPE_0_DATA_MASK GENMASK(1, 0) /* reg.h:5176 */
#define B_BE_FWD_WLAN_CPU_TYPE_0_MNG_MASK GENMASK(3, 2) /* reg.h:5175 */
#define B_BE_FWD_WLAN_CPU_TYPE_1_MASK GENMASK(7, 6) /* reg.h:5173 */
#define B_BE_HE_CTN_CHK_BASIC_NAV BIT(13) /* reg.h:6455 */
#define B_BE_HE_CTN_CHK_CCA_BITMAP BIT(2) /* reg.h:6466 */
#define B_BE_HE_CTN_CHK_CCA_P20 BIT(0) /* reg.h:6468 */
#define B_BE_HE_CTN_CHK_EDCCA_BITMAP BIT(3) /* reg.h:6465 */
#define B_BE_HE_CTN_CHK_EDCCA_P20 BIT(1) /* reg.h:6467 */
#define B_BE_HE_CTN_CHK_INTRA_NAV BIT(14) /* reg.h:6454 */
#define B_BE_HE_CTN_CHK_NO_GNT_WL BIT(12) /* reg.h:6456 */
#define B_BE_HE_CTN_CHK_TX_NAV BIT(15) /* reg.h:6453 */
#define B_BE_HE_SIFS_CHK_EDCCA_BITMAP BIT(3) /* reg.h:6446 */
#define B_BE_HE_SIFS_CHK_EDCCA_P20 BIT(1) /* reg.h:6448 */
#define B_BE_HE_SIFS_CHK_NO_GNT_WL BIT(12) /* reg.h:6437 */
#define B_BE_HE_SIGB_CRC_CHK BIT(6) /* reg.h:7390 */
#define B_BE_HW_CTS2SELF_EN BIT(16) /* reg.h:6638 */
#define B_BE_HW_CTS2SELF_PKT_LEN_TH_MASK GENMASK(31, 24) /* reg.h:6636 */
#define B_BE_HW_CTS2SELF_PKT_LEN_TH_TWW_MASK GENMASK(23, 18) /* reg.h:6637 */
#define B_BE_LSIG_PARITY_CHK_EN BIT(2) /* reg.h:7394 */
#define B_BE_MACLBK_EN BIT(0) /* reg.h:7135 */
#define B_BE_MACLBK_PLCP_DLY_MASK GENMASK(16, 8) /* reg.h:7132 */
#define B_BE_MC_DEC BIT(3) /* reg.h:5739 */
#define B_BE_MLO_HW_CHGLINK_EN BIT(10) /* reg.h:5927 */
#define B_BE_MLO_TABLE_INIT_DONE BIT(31) /* reg.h:5880 */
#define B_BE_MLO_TABLE_REINIT BIT(23) /* reg.h:5882 */
#define B_BE_MPDUINFO_B1_BADDR_MASK GENMASK(5, 0) /* reg.h:5796 */
#define B_BE_MPDUINFO_FEN BIT(31) /* reg.h:5794 */
#define B_BE_MPDUINFO_PKTID_MASK GENMASK(27, 16) /* reg.h:5795 */
#define B_BE_NO_GNT_WL_EN BIT(5) /* reg.h:6359 */
#define B_BE_PLCP_SRC_EN BIT(0) /* reg.h:7485 */
#define B_BE_PLCP_SU_PSDU_LEN_SRC BIT(8) /* reg.h:7545 */
#define B_BE_PREBKF_TIME_MASK GENMASK(4, 0) /* reg.h:6333 */
#define B_BE_PREBKF_TIME_NONAC_MASK GENMASK(12, 8) /* reg.h:6371 */
#define B_BE_PTCL_TRIGGER_SS_EN_0 BIT(2) /* reg.h:6606 */
#define B_BE_PTCL_TRIGGER_SS_EN_1 BIT(3) /* reg.h:6605 */
#define B_BE_PTCL_TRIGGER_SS_EN_UL BIT(4) /* reg.h:6604 */
#define B_BE_PTCL_TX_ARB_TO_MODE BIT(6) /* reg.h:6780 */
#define B_BE_PTCL_TX_ARB_TO_THR_MASK GENMASK(5, 0) /* reg.h:6781 */
#define B_BE_QOSNULL_UPD_MUEDCA_EN BIT(3) /* reg.h:6616 */
#define B_BE_RRSR_CCK_MASK GENMASK(3, 0) /* reg.h:6231 */
#define B_BE_RRSR_HE_MASK GENMASK(31, 24) /* reg.h:6221 */
#define B_BE_RRSR_HT_MASK GENMASK(15, 8) /* reg.h:6223 */
#define B_BE_RRSR_OFDM_MASK GENMASK(7, 0) /* reg.h:6224 */
#define B_BE_RRSR_RATE_EN_MASK GENMASK(12, 8) /* reg.h:6229 */
#define B_BE_RRSR_VHT_MASK GENMASK(23, 16) /* reg.h:6222 */
#define B_BE_RSC_MASK GENMASK(7, 6) /* reg.h:6230 */
#define B_BE_RXDMA_F2PCMDRPT_QUEUE_ID_SW_MASK GENMASK(23, 18) /* reg.h:6911 */
#define B_BE_RXDMA_TXRPT_QUEUE_ID_SW_MASK GENMASK(30, 25) /* reg.h:6910 */
#define B_BE_RXTRIG_FCSCHK_EN BIT(20) /* reg.h:7155 */
#define B_BE_RX_DLK_CCA_TIME_MASK GENMASK(15, 8) /* reg.h:7376 */
#define B_BE_RX_DLK_DATA_TIME_MASK GENMASK(7, 4) /* reg.h:7378 */
#define B_BE_RX_DLK_RST_EN BIT(1) /* reg.h:7382 */
#define B_BE_RX_MPDU_MAX_LEN_MASK GENMASK(21, 16) /* reg.h:7402 */
#define B_BE_R_MACID_ACQ_CHK_EN BIT(0) /* reg.h:6102 */
#define B_BE_SEC_PRE_ENQUE_TX BIT(11) /* reg.h:5733 */
#define B_BE_SEC_RX_DEC BIT(1) /* reg.h:5741 */
#define B_BE_SEC_TX_ENC BIT(0) /* reg.h:5742 */
#define B_BE_SIGA_CRC_CHK BIT(3) /* reg.h:7393 */
#define B_BE_SR_CTRL_PLCP_EN BIT(1) /* reg.h:7477 */
#define B_BE_SR_EN BIT(0) /* reg.h:7478 */
#define B_BE_SS_EN BIT(0) /* reg.h:5936 */
#define B_BE_SS_INIT_DONE BIT(31) /* reg.h:5915 */
#define B_BE_TB_CHK_BASIC_NAV BIT(13) /* reg.h:6417 */
#define B_BE_TB_CHK_EDCCA_BITMAP BIT(3) /* reg.h:6427 */
#define B_BE_TB_CHK_NO_GNT_WL BIT(12) /* reg.h:6418 */
#define B_BE_TXSB_160M_MASK GENMASK(15, 12) /* reg.h:6203 */
#define B_BE_TXSB_20M_MASK GENMASK(3, 0) /* reg.h:6214 */
#define B_BE_TXSB_40M_MASK GENMASK(7, 4) /* reg.h:6210 */
#define B_BE_TXSB_80M_MASK GENMASK(11, 8) /* reg.h:6206 */
#define B_BE_TX_ADDR_MLD_TO_LIK BIT(4) /* reg.h:5652 */
#define B_BE_TX_HW_ACK_POLICY_EN BIT(1) /* reg.h:5655 */
#define B_BE_TX_HW_SEQ_EN BIT(0) /* reg.h:5656 */
#define B_BE_TX_MAC_MPDU_PROC_EN BIT(2) /* reg.h:5654 */
#define B_BE_UC_MGNT_DEC BIT(4) /* reg.h:5738 */
#define B_BE_UID_FILTER_MASK GENMASK(31, 24) /* reg.h:7400 */
#define B_BE_VHT_MU_SIGB_CRC_CHK BIT(5) /* reg.h:7391 */
#define B_BE_VHT_SU_SIGB_CRC_CHK BIT(4) /* reg.h:7392 */
#define B_BE_WARM_INIT BIT(29) /* reg.h:5917 */
#define B_BE_WMAC_NAV_UPPER_EN BIT(26) /* reg.h:7143 */
#define B_BE_WMAC_NAV_UPPER_MASK GENMASK(15, 8) /* reg.h:7147 */
#define B_BE_WMAC_PLCP_UP_NAV_EN BIT(17) /* reg.h:7145 */
#define B_BE_WMAC_RESP_DCM_EN BIT(20) /* reg.h:7122 */
#define B_BE_WMAC_RESP_DOPPLEB_BE_EN BIT(21) /* reg.h:7121 */
#define B_BE_WMAC_RESP_REF_RATE_MASK GENMASK(11, 0) /* reg.h:7125 */
#define B_BE_WMAC_RESP_REF_RATE_SEL BIT(12) /* reg.h:7124 */
#define B_BE_WMAC_SPEC_SIFS_CCK_MASK GENMASK(7, 0) /* reg.h:7114 */
#define B_BE_WMAC_SPEC_SIFS_OFDM_MASK GENMASK(15, 8) /* reg.h:7112 */
#define B_BE_WMAC_TF_UP_NAV_EN BIT(16) /* reg.h:7146 */
#define MPDU_INFO_B1_OFST 18 /* reg.h:5797 */
#define NAV_25MS 0xC4 /* reg.h:7148 */
#define PRELD_AMSDU_SIZE 52 /* reg.h:1961 */
#define PRELD_B0_ENT_NUM 10 /* reg.h:1960 */
#define PRELD_NEXT_WND 1 /* reg.h:1967 */
#define RX_FLTR_FRAME_DROP_BE 0x0000 /* reg.h:7420 */
#define R_AX_TXPKTCTL_B0_PRELD_CFG0 0x9F48 /* reg.h:1957 */
#define R_AX_TXPKTCTL_B0_PRELD_CFG1 0x9F4C /* reg.h:1965 */
#define R_BE_ADDR_CAM_CTRL 0x11434 /* reg.h:7431 */
#define R_BE_AMPDU_AGG_LIMIT 0x10810 /* reg.h:6620 */
#define R_BE_BSSID_SRC_CTRL 0x1144B /* reg.h:7480 */
#define R_BE_CCA_CFG_0 0x10340 /* reg.h:6342 */
#define R_BE_CH0_PAGE_CTRL 0xB718 /* reg.h:6047 */
#define R_BE_CMAC_SHARE_ACQCHK_CFG_0 0x0E010 /* reg.h:6097 */
#define R_BE_CTN_CFG_0 0x1034C /* reg.h:6366 */
#define R_BE_CTRL_FLTR 0x11424 /* reg.h:7417 */
#define R_BE_CUT_AMSDU_CTRL 0x9C94 /* reg.h:5690 */
#define R_BE_DATA_FLTR 0x1142C /* reg.h:7427 */
#define R_BE_DISP_FWD_WLAN_0 0x8938 /* reg.h:5160 */
#define R_BE_DLK_PROTECT_CTL 0x11402 /* reg.h:7374 */
#define R_BE_EDCA_BCNQ_PARAM 0x10324 /* reg.h:6320 */
#define R_BE_HDR_SHCUT_SETTING 0x9B00 /* reg.h:5651 */
#define R_BE_HE_CTN_CHK_CCA_NAV 0x103C4 /* reg.h:6451 */
#define R_BE_HE_SIFS_CHK_CCA_NAV 0x103B4 /* reg.h:6432 */
#define R_BE_MAC_LOOPBACK 0x11020 /* reg.h:7127 */
#define R_BE_MGNT_FLTR 0x11428 /* reg.h:7423 */
#define R_BE_MLO_INIT_CTL 0xA114 /* reg.h:5879 */
#define R_BE_MPDU_PROC 0x9C00 /* reg.h:5663 */
#define R_BE_PLCP_HDR_FLTR 0x11404 /* reg.h:7385 */
#define R_BE_PREBKF_CFG_0 0x10338 /* reg.h:6328 */
#define R_BE_PTCL_COMMON_SETTING_0 0x10800 /* reg.h:6597 */
#define R_BE_PTCL_FSM_MON 0x108E8 /* reg.h:6772 */
#define R_BE_PTCL_RRSR0 0x1008C /* reg.h:6219 */
#define R_BE_PTCL_RRSR1 0x10090 /* reg.h:6226 */
#define R_BE_RCR 0x11400 /* reg.h:7361 */
#define R_BE_RESPBA_CAM_CTRL 0x1143C /* reg.h:7443 */
#define R_BE_RESP_CSI_RESERVED_PAGE 0x11810 /* reg.h:7554 */
#define R_BE_RXTRIG_TEST_USER_2 0x110B0 /* reg.h:7151 */
#define R_BE_RX_CTRL_1 0x10C0C /* reg.h:6908 */
#define R_BE_RX_FLTR_OPT 0x11420 /* reg.h:7398 */
#define R_BE_RX_HDRTRNS 0x9CC0 /* reg.h:5704 */
#define R_BE_RX_PLCP_EXT_OPTION_1 0x11514 /* reg.h:7536 */
#define R_BE_RX_SR_CTRL 0x1144A /* reg.h:7473 */
#define R_BE_SEC_ENG_CTRL 0x9D00 /* reg.h:5719 */
#define R_BE_SEC_MPDU_PROC 0x9D04 /* reg.h:5744 */
#define R_BE_SIFS_SETTING 0x10824 /* reg.h:6634 */
#define R_BE_SS_CTRL 0xA310 /* reg.h:5914 */
#define R_BE_TB_CHK_CCA_NAV 0x103AC /* reg.h:6413 */
#define R_BE_TB_PPDU_CTRL 0x1080C /* reg.h:6610 */
#define R_BE_TRXPTCL_RESP_0 0x11004 /* reg.h:7098 */
#define R_BE_TRXPTCL_RESP_1 0x11008 /* reg.h:7116 */
#define R_BE_TXPKTCTL_MPDUINFO_CFG 0x9F10 /* reg.h:5793 */
#define R_BE_TX_SUB_BAND_VALUE 0x10088 /* reg.h:6199 */
#define R_BE_WMAC_ACK_BA_RESP_EHT_LEG_PUNC 0x11208 /* reg.h:7341 */
#define R_BE_WMAC_ACK_BA_RESP_HE 0x11204 /* reg.h:7321 */
#define R_BE_WMAC_ACK_BA_RESP_LEGACY 0x11200 /* reg.h:7301 */
#define R_BE_WMAC_NAV_CTL 0x11080 /* reg.h:7141 */
#define R_BE_WMTX_TCR_BE_4 0x10E2C /* reg.h:7075 */
#define SCH_PREBKF_24US 0x18 /* reg.h:2232 */
#define S_AX_CTS2S_TH_1K 4 /* reg.h:2555 */
#define S_AX_CTS2S_TH_SEC_256B 1 /* reg.h:2546 */
#define S_AX_PTCL_TO_2MS 0x3F /* reg.h:2664 */
#define S_BE_BACAM_RST_ALL 2 /* reg.h:7457 */
#define S_BE_MACLBK_PLCP_DLY_DEF 0x28 /* reg.h:7133 */
#define S_BE_TXSB_160M_1 1 /* reg.h:6205 */
#define S_BE_TXSB_20M_8 8 /* reg.h:6215 */
#define S_BE_TXSB_40M_4 4 /* reg.h:6213 */
#define S_BE_TXSB_80M_2 2 /* reg.h:6208 */
#define TRXCFG_MPDU_PROC_CUT_CTRL	0x010E05F0 /* reg.h:1814 */
#define TRXCFG_MPDU_PROC_RX_HDR_CONV	0x00000000 /* reg.h:5711 */
#define TRXCFG_RMAC_CCA_TO 32 /* reg.h:7377 */
#define TRXCFG_RMAC_DATA_TO 15 /* reg.h:7379 */
#define WLCPU_RXCH2_QID 0xA /* reg.h:6915 */
#define WMAC_SPEC_SIFS_CCK	 0xA /* reg.h:3038 */
#define WMAC_SPEC_SIFS_OFDM_1115E 0x11 /* reg.h:7113 */

#endif /* __RTW89_BE_USED_H__ */

/* PHY0 BB post-init and RF-v2 fields. */
#define B_BY_SLOPE GENMASK(31, 24) /* reg.h:8948 */
#define B_CLK_640M BIT(2) /* reg.h:7958 */
#define B_EHTTB_EN BIT(15) /* reg.h:8960 */
#define B_EHT_MCS14 BIT(31) /* reg.h:8951 */
#define B_EHT_MCS15 BIT(31) /* reg.h:8954 */
#define B_EHT_RATE_TH GENMASK(31, 28) /* reg.h:8940 */
#define B_HEERSU_EN BIT(19) /* reg.h:8959 */
#define B_HEMU_EN BIT(21) /* reg.h:8958 */
#define B_HE_RATE_TH GENMASK(30, 27) /* reg.h:8952 */
#define B_HT_VHT_TH GENMASK(11, 0) /* reg.h:8955 */
#define B_HWGEN_EN BIT(25) /* reg.h:8965 */
#define B_HWSI_DATA_ADDR GENMASK(7, 0) /* reg.h:8340 */
#define B_HWSI_DATA_VAL GENMASK(27, 8) /* reg.h:8339 */
#define B_MAG_AB GENMASK(23, 0) /* reg.h:8949 */
#define B_MGA_AEND GENMASK(31, 24) /* reg.h:8946 */
#define B_PWROFST_COMP BIT(20) /* reg.h:8966 */
#define B_SC_CORNER GENMASK(10, 0) /* reg.h:8944 */
#define B_SLOPE_A GENMASK(13, 0) /* reg.h:8942 */
#define B_SLOPE_B GENMASK(27, 14) /* reg.h:8941 */
#define B_SU_PUNC_EN BIT(1) /* reg.h:8963 */
#define B_TB_EN BIT(23) /* reg.h:8957 */
#define B_TXFCTR_EN BIT(19) /* reg.h:8894 */
#define B_TXFCTR_THD GENMASK(19, 10) /* reg.h:8892 */
#define B_UDP_COEEF BIT(19) /* reg.h:8143 */
#define R_BBCLK 0x0000 /* reg.h:7957 */
#define R_BEDGE 0x6BFC /* reg.h:8950 */
#define R_BEDGE2 0x6C00 /* reg.h:8953 */
#define R_BEDGE3 0x6C04 /* reg.h:8956 */
#define R_BEDGE5 0x6C10 /* reg.h:8964 */
#define R_MAG_A 0x6BF4 /* reg.h:8945 */
#define R_MAG_AB 0x6BF8 /* reg.h:8947 */
#define R_SC_CORNER 0x6B70 /* reg.h:8943 */
#define R_SLOPE 0x6B6C /* reg.h:8939 */
#define R_SU_PUNC 0x6C08 /* reg.h:8962 */
#define R_TXFCTR 0x627C /* reg.h:8891 */
#define R_TXSCALE 0x6284 /* reg.h:8893 */
#define R_UDP_COEEF 0x0CBC /* reg.h:8142 */
