/*
 * MediaTek MT7925 (RZ717, PCI 14c3:0717) — the register subset the D0/D1
 * bring-up ladder programs, copied line by line from the upstream mt76
 * driver (github.com/openwrt/mt76 @ 0936617dc529ff9da0385221d3fa6de732231177,
 * files mt7925/regs.h, mt7925/pci.c and mt792x_regs.h); the upstream file
 * and line each define came from is on the line.
 *
 * There is no public datasheet for this part, so the vendor driver's header
 * IS the register reference — cited, never copied wholesale: only what the
 * ladder touches is here. The fixed-map resolutions and the L1-remap
 * composition are hand-written with their derivation in src/regs/mod.rs,
 * not transcribed, exactly as rtw89_be.h does it.
 *
 * LICENCE, and why this part is easier than the Realtek one: every file
 * cited below is SPDX-License-Identifier: BSD-3-Clause-Clear (© MediaTek),
 * NOT GPL-2.0. drivers/net/rtw89/tools/rtw89_be.h cites GPL-2.0 headers;
 * this one does not, so the provenance question that hangs over the rtw89
 * transcription does not arise here.
 *
 * SCOPE — D0 and the D1 download path. D0 claims BAR0, drives the L1 remap
 * window, reads the chip identity back, takes driver ownership and reads the
 * firmware-ready state. D1 adds the WFDMA engine, the two MCU queues the
 * download rides, and the command framing — and NOTHING ELSE. No data path,
 * no MAC, no PHY, no efuse: res/NETWORK_RESTART.md S3 rule 2 finishes the
 * narrow rung before the next is started, and the M-ladder's registers are
 * not transcribed until an M rung needs them.
 *
 * SIZES, corrected by res/MT7925_EVIDENCE.md S14: the two blobs are
 * 1,580,280 bytes of FILE and 1,275,712 bytes of TRANSFER. The difference is
 * the patch's own header, the RAM code's table and trailer, a 172-byte
 * build-info block, and the 303,936-byte CLC region the download skips.
 *
 * Regenerate src/regs/mt7925.rs from this file — never edit that file:
 *   python3 drivers/ivpu/tools/gen_ivpu_regs.py --strict \
 *     --source 'MediaTek MT7925 D0 register subset, copied from mt76 @ 0936617dc529 (BSD-3-Clause-Clear) with the upstream line cited per define ({header}).' \
 *     --about 'MT7925 D0 register offsets and field masks over the `Regs` seam (BAR0 read32/write32); names are verbatim from the upstream mt76 headers so a reviewer can diff them 1:1 against the cited upstream lines.' \
 *     drivers/net/mt7925/tools/mt7925.h > drivers/net/mt7925/src/regs/mt7925.rs
 */
/* SPDX-License-Identifier: BSD-3-Clause-Clear */
/*
 * mt7925.h -- register / constant subset used by the D0 rung.
 *
 * Target: MT7925, PCI 14c3:0717, driver mt7925e.  Every line below is copied
 * verbatim from the pinned mt76 sources; the trailing comment gives
 * file:line.
 */

/* ---- The host-interface remap windows -------------------------------
 *
 * BAR0 is 2 MiB and `__mt7925_reg_addr` returns `addr` unchanged for
 * `addr < 0x200000` (mt7925/pci.c:317) — so every offset in this block is a
 * DIRECT BAR0 offset and needs no translation. Everything above 2 MiB is
 * reached either through the fixed map or through these windows.
 */
#define MT_HIF_REMAP_L1			0x155024	/* mt7925/regs.h:83 */
#define MT_HIF_REMAP_L1_MASK		GENMASK(31, 16)	/* mt7925/regs.h:84 */
#define MT_HIF_REMAP_L1_OFFSET		GENMASK(15, 0)	/* mt7925/regs.h:85 */
#define MT_HIF_REMAP_L1_BASE		GENMASK(31, 16)	/* mt7925/regs.h:86 */
#define MT_HIF_REMAP_BASE_L1		0x130000	/* mt7925/regs.h:87 */
#define MT_HIF_REMAP_L2			0x0120		/* mt7925/regs.h:89 */
#define MT_HIF_REMAP_BASE_L2		0x18500000	/* mt7925/regs.h:90 */

/* ---- Chip identity --------------------------------------------------
 *
 * Both are logical addresses in 0x70000000..0x78000000, which
 * `__mt7925_reg_addr` sends down the L1-remap path (mt7925/pci.c:348-351):
 * they are NOT direct BAR0 offsets. mt7925/pci.c:723 composes the revision
 * as `(rr(MT_HW_CHIPID) << 16) | (rr(MT_HW_REV) & 0xff)`.
 *
 * On the lab's Framework 13 that composition printed 0x79250000
 * (res/MT7925_EVIDENCE.md S11), so MT_HW_CHIPID reads 0x7925 and MT_HW_REV
 * reads 0x00 on this stepping. That is the D0 witness's expected value.
 */
#define MT_HW_CHIPID			0x70010200	/* mt792x_regs.h:429 */
#define MT_HW_REV			0x70010204	/* mt792x_regs.h:430 */

/* ---- Driver/firmware ownership --------------------------------------
 *
 * Logical addresses under 0x7c060000, which the default fixed map carries at
 * BAR0 0x0e0000 with a 0x10000 span (mt7925/pci.c:277, "CONN_INFRA,
 * conn_host_csr_top") — so these resolve WITHOUT touching the L1 window.
 * The resolution itself is hand-written in src/regs/mod.rs.
 *
 * mt792x_core.c:1045 writes CLR_OWN and then polls OWN_SYNC to zero; :1092
 * writes SET_OWN and polls OWN_SYNC to 4 when handing the part back.
 */
#define MT_CONN_ON_LPCTL		0x7c060010	/* mt792x_regs.h:500 */
#define PCIE_LPCR_HOST_SET_OWN		BIT(0)		/* mt792x_regs.h:501 */
#define PCIE_LPCR_HOST_CLR_OWN		BIT(1)		/* mt792x_regs.h:502 */
#define PCIE_LPCR_HOST_OWN_SYNC		BIT(2)		/* mt792x_regs.h:503 */

/* ---- Firmware-ready state -------------------------------------------
 *
 * Same 0x7c060000 fixed-map block as the ownership register above.
 * MT_TOP_MISC2_FW_N9_RDY is the pair FW_PWR_ON|FW_N9_ON, which is what
 * upstream polls to decide the N9 core is up.
 */
#define MT_CONN_ON_MISC			0x7c0600f0	/* mt792x_regs.h:505 */
#define MT_TOP_MISC2_FW_PWR_ON		BIT(0)		/* mt792x_regs.h:506 */
#define MT_TOP_MISC2_FW_N9_ON		BIT(1)		/* mt792x_regs.h:507 */
#define MT_TOP_MISC2_FW_N9_RDY		GENMASK(1, 0)	/* mt792x_regs.h:508 */

/* ---- Firmware state word --------------------------------------------
 *
 * MT_TOP_MISC is MT_TOP(0xf0) = 0x18060000 + 0xf0 (mt792x_regs.h:404,411),
 * resolved here rather than left as the composition, because the generator
 * transcribes values and this file's rule is one value per line.
 * 0x180600f0 is in 0x18000000..0x18c00000, so it takes the L1-remap path.
 */
#define MT_TOP_MISC			0x180600f0	/* mt792x_regs.h:411 (MT_TOP(0xf0), base :404) */
#define MT_TOP_MISC_FW_STATE		GENMASK(2, 0)	/* mt792x_regs.h:412 */

/* ---- WFDMA0 — the engine the download rides ------------------------
 *
 * MT_WFDMA0_BASE is 0xd4000, inside BAR0's 2 MiB, so every WFDMA register
 * below is a DIRECT offset and costs one transaction. That is why the
 * download's inner loop is affordable at all: a ring doorbell on the L1
 * window would be four transactions per chunk across ~312 chunks.
 *
 * Upstream composes these as MT_WFDMA0(ofs) (mt792x_regs.h:263); this file's
 * rule is one value per line, so each is resolved.
 */
#define MT_WFDMA0_BASE			0xd4000		/* mt792x_regs.h:262 */
#define MT_WFDMA0_RST			0xd4100		/* mt792x_regs.h:265 (MT_WFDMA0(0x100)) */
#define MT_WFDMA0_RST_LOGIC_RST		BIT(4)		/* mt792x_regs.h:266 */
#define MT_WFDMA0_RST_DMASHDL_ALL_RST	BIT(5)		/* mt792x_regs.h:267 */
#define MT_MCU2HOST_SW_INT_ENA		0xd41f4		/* mt792x_regs.h:283 (MT_WFDMA0(0x1f4)) */
#define MT_MCU_CMD			0xd41f0		/* mt792x_regs.h:274 (MT_WFDMA0(0x1f0)) */
#define MT_MCU_CMD_WAKE_RX_PCIE		BIT(0)		/* mt792x_regs.h:275 */
#define MT_MCU_CMD_ERROR_MASK		GENMASK(5, 1)	/* mt792x_regs.h:281 */
#define MT_WFDMA0_HOST_INT_ENA		0xd4204		/* mt7925/regs.h:31 (MT_WFDMA0(0x204)) */
#define MT_WFDMA0_HOST_INT_DIS		0xd422c		/* mt7925/regs.h:32 (MT_WFDMA0(0x22c)) */
#define MT_WFDMA0_HOST_INT_STA		0xd4200		/* mt792x_regs.h:285 (MT_WFDMA0(0x200)) */
#define MT7925_PCIE_MAC_INT_ENABLE	0x10188		/* mt7925/regs.h:109 (MT7925_PCIE_MAC(0x188)) */

/* ---- WFDMA0 global configuration ------------------------------------
 *
 * mt792x_dma_enable (mt792x_dma.c:280-360) SETS the first block and then
 * TX_DMA_EN|RX_DMA_EN separately; mt792x_dma_disable (:447-485) clears them
 * and polls the two BUSY bits to zero on a 100 ms budget.
 */
#define MT_WFDMA0_GLO_CFG		0xd4208		/* mt792x_regs.h:292 (MT_WFDMA0(0x208)) */
#define MT_WFDMA0_GLO_CFG_TX_DMA_EN	BIT(0)		/* mt792x_regs.h:293 */
#define MT_WFDMA0_GLO_CFG_TX_DMA_BUSY	BIT(1)		/* mt792x_regs.h:294 */
#define MT_WFDMA0_GLO_CFG_RX_DMA_EN	BIT(2)		/* mt792x_regs.h:295 */
#define MT_WFDMA0_GLO_CFG_RX_DMA_BUSY	BIT(3)		/* mt792x_regs.h:296 */
#define MT_WFDMA0_GLO_CFG_DMA_SIZE	GENMASK(5, 4)	/* mt792x_regs.h:297 */
#define MT_WFDMA0_GLO_CFG_TX_WB_DDONE	BIT(6)		/* mt792x_regs.h:298 */
#define MT_WFDMA0_GLO_CFG_FIFO_DIS_CHECK	BIT(11)	/* mt792x_regs.h:300 */
#define MT_WFDMA0_GLO_CFG_FIFO_LITTLE_ENDIAN	BIT(12)	/* mt792x_regs.h:301 */
#define MT_WFDMA0_GLO_CFG_RX_WB_DDONE	BIT(13)		/* mt792x_regs.h:302 */
#define MT_WFDMA0_GLO_CFG_CSR_DISP_BASE_PTR_CHAIN_EN	BIT(15)	/* mt792x_regs.h:303 */
#define MT_WFDMA0_GLO_CFG_OMIT_RX_INFO_PFET2	BIT(21)	/* mt792x_regs.h:305 */
#define MT_WFDMA0_GLO_CFG_OMIT_RX_INFO	BIT(27)		/* mt792x_regs.h:307 */
#define MT_WFDMA0_GLO_CFG_OMIT_TX_INFO	BIT(28)		/* mt792x_regs.h:308 */
#define MT_WFDMA0_GLO_CFG_CLK_GAT_DIS	BIT(30)		/* mt792x_regs.h:309 */
#define MT_WFDMA0_GLO_CFG_EXT0		0xd42b0		/* mt792x_regs.h:344 (MT_WFDMA0(0x2b0)) */
#define MT_WFDMA0_GLO_CFG_EXT0_CSR_TX_DMASHDL_EN	BIT(6)	/* mt792x_regs.h:346 */
#define MT_WFDMA0_RST_DTX_PTR		0xd420c		/* mt792x_regs.h:333 (MT_WFDMA0(0x20c)) */
#define MT_WFDMA0_RST_DRX_PTR		0xd4280		/* mt792x_regs.h:341 (MT_WFDMA0(0x280)) */
#define MT_WFDMA0_PRI_DLY_INT_CFG0	0xd42f0		/* mt792x_regs.h:357 (MT_WFDMA0(0x2f0)) */

/* ---- The per-ring prefetch windows ----------------------------------
 *
 * MT_WFDMA0_GLO_CFG_CSR_DISP_BASE_PTR_CHAIN_EN turns on the engine's
 * internal descriptor cache, and these registers are where each ring's
 * slice of it lives: PREFETCH(base, depth) = base << 16 | depth
 * (mt792x_dma.c:155). Leave them at reset and every ring's window is a
 * zero-length slice at offset zero -- all of them aliased onto each
 * other. A receive-only engine never notices; the first transmit walks
 * the wrong cache line.
 *
 * mt7925_dma_prefetch_setup (mt792x_dma.c:157-172) writes ten. The three
 * this ladder needs are TX_RING15 (MCU_WM), TX_RING16 (FWDL) and
 * RX_RING0 (the event queue); the other seven are written too, because a
 * half-configured prefetch map is its own hazard -- the windows are
 * carved out of one shared pool and the unused rings' slices must not be
 * left overlapping the used ones'.
 */
#define MT_WFDMA0_TX_RING0_EXT_CTRL	0xd4600		/* mt792x_regs.h:359 (MT_WFDMA0(0x600)) */
#define MT_WFDMA0_TX_RING1_EXT_CTRL	0xd4604		/* mt792x_regs.h:360 (MT_WFDMA0(0x604)) */
#define MT_WFDMA0_TX_RING2_EXT_CTRL	0xd4608		/* mt792x_regs.h:361 (MT_WFDMA0(0x608)) */
#define MT_WFDMA0_TX_RING3_EXT_CTRL	0xd460c		/* mt792x_regs.h:362 (MT_WFDMA0(0x60c)) */
#define MT_WFDMA0_TX_RING15_EXT_CTRL	0xd463c		/* mt792x_regs.h:366 (MT_WFDMA0(0x63c)) */
#define MT_WFDMA0_TX_RING16_EXT_CTRL	0xd4640		/* mt792x_regs.h:367 (MT_WFDMA0(0x640)) */
#define MT_WFDMA0_RX_RING0_EXT_CTRL	0xd4680		/* mt792x_regs.h:373 (MT_WFDMA0(0x680)) */
#define MT_WFDMA0_RX_RING1_EXT_CTRL	0xd4684		/* mt792x_regs.h:374 (MT_WFDMA0(0x684)) */
#define MT_WFDMA0_RX_RING2_EXT_CTRL	0xd4688		/* mt792x_regs.h:375 (MT_WFDMA0(0x688)) */
#define MT_WFDMA0_RX_RING3_EXT_CTRL	0xd468c		/* mt792x_regs.h:376 (MT_WFDMA0(0x68c)) */

/* ---- mt7925_wfdma_setup (mt792x_dma.c:174-179) ----------------------
 *
 * Three writes the part-specific tail of mt792x_dma_enable makes. The
 * two priority words are the arbitration weights between rings; BIT(28)
 * of GLO_CFG_EXT1 upstream names nothing at all, so neither does this.
 *
 * MT_UWFDMA0_GLO_CFG_EXT1 is THE SAME REGISTER as MT_WFDMA0(0x2b4):
 * MT_UWFDMA0(ofs) is 0x7c024000 + ofs (mt792x_regs.h:491) and the fixed
 * map sends 0x7c02_xxxx to 0x000d_xxxx, so 0x7c0242b4 and 0xd42b4 are two
 * names for one address. It is kept in its upstream spelling so the
 * define diffs 1:1, and regs/mod.rs's route() resolves it -- see the
 * route test, which pins the identity rather than trusting it.
 */
#define MT_WFDMA0_INT_RX_PRI		0xd4298		/* mt792x_regs.h:342 (MT_WFDMA0(0x298)) */
#define MT_WFDMA0_INT_TX_PRI		0xd429c		/* mt792x_regs.h:343 (MT_WFDMA0(0x29c)) */
#define MT_UWFDMA0_GLO_CFG_EXT1		0x7c0242b4	/* mt792x_regs.h:494 (MT_UWFDMA0(0x2b4), base :491) */
#define MT_UWFDMA0_GLO_CFG_EXT1_BIT28	BIT(28)		/* mt792x_dma.c:176 (upstream writes the bare BIT(28)) */

/* ---- The queue register banks ---------------------------------------
 *
 * Each queue has four words at `ring_base + qid * MT_RING_SIZE`:
 * desc_base, ring_size, cpu_idx, dma_idx (struct mt76_queue_regs,
 * mt76.h:236-241), and mt76_dma_alloc_queue computes the bank as
 * `regs + ring_base + idx * MT_RING_SIZE` (dma.c:838).
 */
#define MT_RING_SIZE			0x10		/* dma.h:12 */

/*
 * One receive buffer. Every mt76 part posts the event queue with buffers of
 * this size; the MCU response that D1c reads is an `MCU_RXD_BYTES` header
 * plus a body that fits well inside it.
 */
#define MT_RX_BUF_SIZE			2048		/* mt76.h:24 */
#define MT_TX_RING_BASE			0xd4300		/* mt792x_regs.h:382 (MT_WFDMA0(0x300)) */
#define MT_RX_EVENT_RING_BASE		0xd4500		/* mt792x_regs.h:383 (MT_WFDMA0(0x500)) */
#define MT_RX_DATA_RING_BASE		0xd4500		/* mt7925/regs.h:78 (MT_WFDMA0(0x500)) */

/* The queue indices this driver uses. The transmit data queue is M-ladder
 * work and is deliberately absent; the receive data queue is programmed from
 * M1 as an event sink (see `Queue::EventData`). */
#define MT7925_TXQ_MCU_WM		15		/* mt7925/mt7925.h:124 */
#define MT7925_TXQ_FWDL			16		/* mt7925/mt7925.h:125 (MT7925_TXQ_MCU_WM + 1) */
#define MT7925_RXQ_MCU_WM		0		/* mt7925/mt7925.h:131 */
#define MT7925_RXQ_BAND0		2		/* mt7925/mt7925.h:129 */
#define MT7925_TX_MCU_RING_SIZE		256		/* mt7925/mt7925.h:18 */
#define MT7925_TX_FWDL_RING_SIZE	128		/* mt7925/mt7925.h:19 */
#define MT7925_RX_MCU_RING_SIZE		512		/* mt7925/mt7925.h:22 */

/* ---- Interrupt enable bits ------------------------------------------ */
#define HOST_RX_DONE_INT_ENA0		BIT(0)		/* mt792x_regs.h:311 */
#define HOST_RX_DONE_INT_ENA2		BIT(2)		/* mt792x_regs.h:313 */
#define HOST_TX_DONE_INT_ENA0		BIT(4)		/* mt792x_regs.h:318 */
#define HOST_TX_DONE_INT_ENA15		BIT(25)		/* mt7925/regs.h:41 */
#define HOST_TX_DONE_INT_ENA16		BIT(26)		/* mt7925/regs.h:42 */
#define HOST_TX_DONE_INT_ENA17		BIT(27)		/* mt7925/regs.h:43 */
#define MCU2HOST_SW_INT_ENA		BIT(29)		/* mt792x_regs.h:329 */

/* ---- The reinit flag and the DMA scheduler bypass -------------------- */
#define MT_WFDMA_DUMMY_CR		0x54000120	/* mt792x_regs.h:417 (MT_MCU_WPDMA0(0x120), base :413) */
#define MT_WFDMA_NEED_REINIT		BIT(1)		/* mt792x_regs.h:418 */
#define MT_DMASHDL_SW_CONTROL		0x7c026004	/* mt792x_regs.h:438 (MT_DMA_SHDL(0x004), base :437) */
#define MT_DMASHDL_DMASHDL_BYPASS	BIT(28)		/* mt792x_regs.h:439 */

/* ---- WFSYS reset ----------------------------------------------------
 *
 * mt792x_wfsys_reset_default (mt792x_dma.c:603-616) clears SW_RST_B, waits
 * 50 ms, sets it, and polls SW_INIT_DONE for 500 ms. The address is
 * 0x7c000140 for a connac3 part (0x18000140 is the connac2 one, and is NOT
 * this part's).
 */
#define MT_WFSYS_SW_RST			0x7c000140	/* mt792x_dma.c:605 */

/* ---- Sleep protection, set immediately before the subsystem reset ----
 *
 * `mt7925_pci_probe` does `mt76_rmw_field(MT_HW_EMI_CTL, SLPPROT_EN, 1)`
 * on the line before `mt792x_wfsys_reset` (`mt7925/pci.c:740-742`). It is
 * not part of the reset function, which is why reading that function
 * alone does not reveal it.
 */
#define MT_HW_EMI_CTL			0x18011100	/* mt792x_regs.h:432 */
#define MT_HW_EMI_CTL_SLPPROT_EN	BIT(1)		/* mt792x_regs.h:433 */
#define WFSYS_SW_RST_B			BIT(0)		/* mt792x_regs.h:515 */
#define WFSYS_SW_INIT_DONE		BIT(4)		/* mt792x_regs.h:516 */

/* ---- The WFDMA descriptor's control word -----------------------------
 *
 * struct mt76_desc is { __le32 buf0; __le32 ctrl; __le32 buf1; __le32 info; }
 * (dma.h:120-125). buf0 is THIRTY-TWO BITS and this part enables no address
 * extension — see res/MT7925_EVIDENCE.md S15.
 */
#define MT_DMA_CTL_SD_LEN1		GENMASK(13, 0)	/* dma.h:15 */
#define MT_DMA_CTL_LAST_SEC1		BIT(14)		/* dma.h:16 */
#define MT_DMA_CTL_SD_LEN0		GENMASK(29, 16)	/* dma.h:18 */
#define MT_DMA_CTL_LAST_SEC0		BIT(30)		/* dma.h:19 */
#define MT_DMA_CTL_DMA_DONE		BIT(31)		/* dma.h:20 */

/* ---- The MCU command descriptor's first two words --------------------
 *
 * mt7925_mcu_fill_message (mt7925/mcu.c:3849-3856) writes exactly these two
 * words ahead of the connac2 MCU txd; the rest of the txd is the command
 * header, which src/mcu.rs encodes.
 */
#define MT_TXD0_TX_BYTES		GENMASK(15, 0)	/* mt76_connac2_mac.h:53 */
#define MT_TXD0_PKT_FMT			GENMASK(24, 23)	/* mt76_connac2_mac.h:51 */
#define MT_TXD0_Q_IDX			GENMASK(31, 25)	/* mt76_connac2_mac.h:50 */
#define MT_TXD1_HDR_FORMAT		GENMASK(15, 14)	/* mt76_connac3_mac.h:232 -- mt7925/mac.h includes connac3, NOT connac2 (17:16) */
#define MT_TX_TYPE_CMD			2		/* mt76_connac2_mac.h:14-19 (enum tx_pkt_type) */
#define MT_HDR_FORMAT_CMD		1		/* mt76_connac2_mac.h:7-12 (enum tx_header_format) */
#define MT_TX_MCU_PORT_RX_Q0		0x20		/* mt76_connac2_mac.h:346 */
#define MT_TX_PORT_IDX_MCU		1		/* mt76_connac2_mac.h:353-356 (enum tx_port_idx) */
#define MCU_PKT_ID			0xa0		/* mt76_connac_mcu.h:53 */
#define MCU_S2D_H2N			0		/* mt76_connac_mcu.h:1125-1130 */
#define MCU_Q_QUERY			0		/* mt76_connac_mcu.h:1118-1123 */
#define MCU_Q_SET			1		/* mt76_connac_mcu.h:1118-1123 */
#define MCU_Q_NA			3		/* mt76_connac_mcu.h:1118-1123 */

/* ---- The MCU commands the download issues ---------------------------- */
#define MCU_CMD_TARGET_ADDRESS_LEN_REQ	0x01		/* mt76_connac_mcu.h:1375 */
#define MCU_CMD_FW_START_REQ		0x02		/* mt76_connac_mcu.h:1376 */
#define MCU_CMD_NIC_POWER_CTRL		0x04		/* mt76_connac_mcu.h:1378 */
#define MCU_CMD_PATCH_START_REQ		0x05		/* mt76_connac_mcu.h:1379 */
#define MCU_CMD_PATCH_FINISH_REQ	0x07		/* mt76_connac_mcu.h:1380 */
#define MCU_CMD_PATCH_SEM_CONTROL	0x10		/* mt76_connac_mcu.h:1381 */
#define MCU_CMD_FW_SCATTER		0xee		/* mt76_connac_mcu.h:1387 */
#define MCU_CMD_RESTART_DL_REQ		0xef		/* mt76_connac_mcu.h:1388 */
#define FW_START_OVERRIDE		BIT(0)		/* mt76_connac_mcu.h:25 */
#define PATCH_SEM_RELEASE		0		/* mt76_connac_mcu.h:1420-1423 */
#define PATCH_SEM_GET			1		/* mt76_connac_mcu.h:1420-1423 */
#define PATCH_NOT_DL_SEM_FAIL		0		/* mt76_connac_mcu.h:1132-1137 */
#define PATCH_IS_DL			1		/* mt76_connac_mcu.h:1132-1137 */
#define PATCH_NOT_DL_SEM_SUCCESS	2		/* mt76_connac_mcu.h:1132-1137 */
#define PATCH_REL_SEM_SUCCESS		3		/* mt76_connac_mcu.h:1132-1137 */
