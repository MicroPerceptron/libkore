/*
 * Intel 82574 GbE Controller Family — the register map this driver core
 * programs, transcribed by hand from the 82574 datasheet (register map
 * chapter; the section is named on each line) in the spelling Linux's
 * e1000e headers use, so a reviewer can diff either way.
 *
 * Only what the core touches is here. Queue-indexed registers keep the
 * 82574's 0x100 stride for its two queues; the 0x0C000/0x0E000 alias
 * blocks later parts add are deliberately absent.
 *
 * Regenerate src/regs/i82574.rs from this file — never edit that file:
 *   python3 drivers/ivpu/tools/gen_ivpu_regs.py --strict \
 *     --source 'Intel 82574 GbE Controller Family Datasheet, transcribed by hand ({header}).' \
 *     --about '82574L register offsets and field masks over the `Regs` seam (BAR0 read32/write32); names follow the Linux e1000e headers so a reviewer can diff them 1:1 against the datasheet transcription.' \
 *     drivers/net/e1000e/tools/i82574.h > drivers/net/e1000e/src/regs/i82574.rs
 */

/* ── General ─────────────────────────────────────────────── §10.2.2 */
#define E1000_CTRL      0x00000 /* Device Control - RW */
#define E1000_STATUS    0x00008 /* Device Status - RO */
#define E1000_EECD      0x00010 /* EEPROM/Flash Control - RW */
#define E1000_EERD      0x00014 /* EEPROM Read - RW */
#define E1000_CTRL_EXT  0x00018 /* Extended Device Control - RW */
#define E1000_MDIC      0x00020 /* MDI Control - RW */

/* ── Interrupts ──────────────────────────────────────────── §10.2.4 */
#define E1000_ICR       0x000C0 /* Interrupt Cause Read - R/W1C */
#define E1000_ITR       0x000C4 /* Interrupt Throttling Rate - RW */
#define E1000_ICS       0x000C8 /* Interrupt Cause Set - WO */
#define E1000_IMS       0x000D0 /* Interrupt Mask Set/Read - RW */
#define E1000_IMC       0x000D8 /* Interrupt Mask Clear - WO */
#define E1000_EIAC      0x000DC /* Extended Interrupt Auto Clear (82574) - RW */
#define E1000_IAM       0x000E0 /* Interrupt Acknowledge Auto Mask - RW */
#define E1000_IVAR      0x000E4 /* Interrupt Vector Allocation (82574 MSI-X) - RW */
#define E1000_EITR_82574(_n) (0x000E8 + (0x4 * (_n))) /* Extended Interrupt Throttle, MSI-X vector n (0..3) - RW */

/* ── Receive ─────────────────────────────────────────────── §10.2.5 */
#define E1000_RCTL      0x00100 /* Receive Control - RW */
#define E1000_RDBAL(_n) (0x02800 + ((_n) * 0x100)) /* RX Descriptor Base Low, queue n - RW */
#define E1000_RDBAH(_n) (0x02804 + ((_n) * 0x100)) /* RX Descriptor Base High - RW */
#define E1000_RDLEN(_n) (0x02808 + ((_n) * 0x100)) /* RX Descriptor Length - RW */
#define E1000_RDH(_n)   (0x02810 + ((_n) * 0x100)) /* RX Descriptor Head - RW */
#define E1000_RDT(_n)   (0x02818 + ((_n) * 0x100)) /* RX Descriptor Tail - RW */
#define E1000_RDTR      0x02820 /* RX Delay Timer - RW */
#define E1000_RXDCTL(_n) (0x02828 + ((_n) * 0x100)) /* RX Descriptor Control - RW */
#define E1000_RADV      0x0282C /* RX Interrupt Absolute Delay Timer - RW */
#define E1000_RFCTL     0x05008 /* Receive Filter Control - RW */
#define E1000_MTA       0x05200 /* Multicast Table Array, 128 x 32 bit - RW */
#define E1000_RAL(_i)   (0x05400 + ((_i) * 8)) /* Receive Address Low, entry i - RW */
#define E1000_RAH(_i)   (0x05404 + ((_i) * 8)) /* Receive Address High, entry i - RW */

/* ── Transmit ────────────────────────────────────────────── §10.2.6 */
#define E1000_TCTL      0x00400 /* Transmit Control - RW */
#define E1000_TIPG      0x00410 /* Transmit Inter Packet Gap - RW */
#define E1000_TDBAL(_n) (0x03800 + ((_n) * 0x100)) /* TX Descriptor Base Low, queue n - RW */
#define E1000_TDBAH(_n) (0x03804 + ((_n) * 0x100)) /* TX Descriptor Base High - RW */
#define E1000_TDLEN(_n) (0x03808 + ((_n) * 0x100)) /* TX Descriptor Length - RW */
#define E1000_TDH(_n)   (0x03810 + ((_n) * 0x100)) /* TX Descriptor Head - RW */
#define E1000_TDT(_n)   (0x03818 + ((_n) * 0x100)) /* TX Descriptor Tail - RW */
#define E1000_TIDV      0x03820 /* TX Interrupt Delay Value - RW */
#define E1000_TXDCTL(_n) (0x03828 + ((_n) * 0x100)) /* TX Descriptor Control - RW */
#define E1000_TADV      0x0382C /* TX Interrupt Absolute Delay Timer - RW */

/* ── CTRL bits ───────────────────────────────────────────── §10.2.2.1 */
#define E1000_CTRL_FD       0x00000001 /* Full duplex: 0 = half, 1 = full */
#define E1000_CTRL_ASDE     0x00000020 /* Auto-speed detection enable */
#define E1000_CTRL_SLU      0x00000040 /* Set link up */
#define E1000_CTRL_FRCSPD   0x00000800 /* Force speed */
#define E1000_CTRL_FRCDPX   0x00001000 /* Force duplex */
#define E1000_CTRL_RST      0x04000000 /* Device reset; self-clearing */
#define E1000_CTRL_PHY_RST  0x80000000 /* PHY reset */

/* ── STATUS bits ─────────────────────────────────────────── §10.2.2.2 */
#define E1000_STATUS_FD         0x00000001 /* Full duplex */
#define E1000_STATUS_LU         0x00000002 /* Link up */
#define E1000_STATUS_SPEED_MASK 0x000000C0 /* Link speed */
#define E1000_STATUS_SPEED_10   0x00000000
#define E1000_STATUS_SPEED_100  0x00000040
#define E1000_STATUS_SPEED_1000 0x00000080

/* ── EECD bits ───────────────────────────────────────────── §10.2.2.3 */
#define E1000_EECD_PRES     0x00000100 /* NVM present */
#define E1000_EECD_AUTO_RD  0x00000200 /* NVM auto-read done */

/* ── CTRL_EXT bits ───────────────────────────────────────── §10.2.2.5 */
#define E1000_CTRL_EXT_EIAME   0x01000000 /* Extended interrupt acknowledge auto-mask enable */
#define E1000_CTRL_EXT_IAME    0x08000000 /* Interrupt acknowledge auto-mask enable */
#define E1000_CTRL_EXT_PBA_CLR 0x80000000 /* MSI-X PBA clear on read */

/* ── ICR / ICS / IMS / IMC bits ──────────────────────────── §10.2.4.1 */
#define E1000_ICR_TXDW      0x00000001 /* Transmit descriptor written back */
#define E1000_ICR_LSC       0x00000004 /* Link status change */
#define E1000_ICR_RXSEQ     0x00000008 /* Receive sequence error */
#define E1000_ICR_RXDMT0    0x00000010 /* Receive descriptor minimum threshold, queue 0 */
#define E1000_ICR_RXO       0x00000040 /* Receiver overrun */
#define E1000_ICR_RXT0      0x00000080 /* Receiver timer, queue 0 */
#define E1000_ICR_RXQ0      0x00100000 /* Receive queue 0 (MSI-X) */
#define E1000_ICR_RXQ1      0x00200000 /* Receive queue 1 (MSI-X) */
#define E1000_ICR_TXQ0      0x00400000 /* Transmit queue 0 (MSI-X) */
#define E1000_ICR_TXQ1      0x00800000 /* Transmit queue 1 (MSI-X) */
#define E1000_ICR_OTHER     0x01000000 /* Other causes (MSI-X) */
#define E1000_ICR_INT_ASSERTED 0x80000000 /* An interrupt is asserted */

/* ── IVAR (82574) ────────────────────────────────────────── §10.2.4.11 */
#define E1000_IVAR_INT_ALLOC_VALID 0x00000008 /* Vector valid, per 4-bit field */
#define E1000_IVAR_RXQ0_SHIFT      0
#define E1000_IVAR_RXQ1_SHIFT      4
#define E1000_IVAR_TXQ0_SHIFT      8
#define E1000_IVAR_TXQ1_SHIFT      12
#define E1000_IVAR_OTHER_SHIFT     16
#define E1000_IVAR_TX_INT_EVERY_WB 0x80000000 /* TX interrupt on every write-back */

/* ── ITR / EITR (82574) ──────────────────────────────────── §10.2.4.2, §10.2.4.9 */
/* Both hold one field: the minimum inter-interrupt interval in 256 ns
 * units, bits 15:0; zero disables throttling.  ITR governs legacy/MSI
 * delivery, EITR[n] the MSI-X vector n — the same encoding in both, which
 * is why Linux's e1000e writes the one raw value to whichever mode is live. */
#define E1000_ITR_INTERVAL_MASK 0x0000FFFF /* Minimum inter-interrupt interval, 256 ns units */

/* ── EIAC (82574) ────────────────────────────────────────── §10.2.4.10 */
#define E1000_EIAC_MASK_82574 0x01F00000 /* RXQ0|RXQ1|TXQ0|TXQ1|OTHER may auto-clear */

/* ── RCTL bits ───────────────────────────────────────────── §10.2.5.1 */
#define E1000_RCTL_EN         0x00000002 /* Receiver enable */
#define E1000_RCTL_SBP        0x00000004 /* Store bad packets */
#define E1000_RCTL_UPE        0x00000008 /* Unicast promiscuous enable */
#define E1000_RCTL_MPE        0x00000010 /* Multicast promiscuous enable */
#define E1000_RCTL_LPE        0x00000020 /* Long packet enable */
#define E1000_RCTL_LBM_MASK   0x000000C0 /* Loopback mode */
#define E1000_RCTL_RDMTS_MASK 0x00000300 /* Receive descriptor minimum threshold size */
#define E1000_RCTL_DTYP_MASK  0x00000C00 /* Descriptor type: 00 = legacy */
#define E1000_RCTL_MO_MASK    0x00003000 /* Multicast offset */
#define E1000_RCTL_BAM        0x00008000 /* Broadcast accept mode */
#define E1000_RCTL_BSIZE_MASK 0x00030000 /* Receive buffer size: 00 = 2048 with BSEX clear */
#define E1000_RCTL_VFE        0x00040000 /* VLAN filter enable */
#define E1000_RCTL_BSEX       0x02000000 /* Buffer size extension */
#define E1000_RCTL_SECRC      0x04000000 /* Strip Ethernet CRC */

/* ── RFCTL bits ──────────────────────────────────────────── §10.2.5.21 */
#define E1000_RFCTL_ACK_DIS   0x00001000 /* ACK accelerate disable */
#define E1000_RFCTL_EXTEN     0x00008000 /* Extended status enable: 1 = extended RX descriptors */

/* ── RAH bits ────────────────────────────────────────────── §10.2.5.24 */
#define E1000_RAH_AV          0x80000000 /* Address valid */

/* ── RXDCTL / TXDCTL bits ────────────────────────────────── §10.2.5.12, §10.2.6.9 */
#define E1000_RXDCTL_WTHRESH_1 0x00010000 /* Write-back threshold: one descriptor */
#define E1000_RXDCTL_GRAN      0x01000000 /* Thresholds in descriptors, not cache lines */
#define E1000_TXDCTL_WTHRESH_1 0x00010000
#define E1000_TXDCTL_GRAN      0x01000000

/* ── TCTL bits ───────────────────────────────────────────── §10.2.6.1 */
#define E1000_TCTL_EN         0x00000002 /* Transmit enable */
#define E1000_TCTL_PSP        0x00000008 /* Pad short packets */
#define E1000_TCTL_CT_MASK    0x00000FF0 /* Collision threshold */
#define E1000_TCTL_COLD_MASK  0x003FF000 /* Collision distance */
#define E1000_TCTL_RTLC       0x01000000 /* Re-transmit on late collision */
#define E1000_CT_SHIFT        4
#define E1000_COLD_SHIFT      12
#define E1000_COLLISION_THRESHOLD 15 /* Datasheet default */
#define E1000_COLLISION_DISTANCE  63 /* Full-duplex default */

/* ── Legacy receive descriptor, 16 bytes ─────────────────── §7.1.4 */
#define E1000_RXD_STAT_DD     0x01 /* Descriptor done */
#define E1000_RXD_STAT_EOP    0x02 /* End of packet */
#define E1000_RXD_ERR_CE      0x01 /* CRC error */
#define E1000_RXD_ERR_SE      0x02 /* Symbol error */
#define E1000_RXD_ERR_SEQ     0x04 /* Sequence error */
#define E1000_RXD_ERR_CXE     0x10 /* Carrier extension error */
#define E1000_RXD_ERR_RXE     0x80 /* RX data error */

/* ── Legacy transmit descriptor, 16 bytes ────────────────── §7.2.3 */
#define E1000_TXD_CMD_EOP     0x01 /* End of packet */
#define E1000_TXD_CMD_IFCS    0x02 /* Insert FCS */
#define E1000_TXD_CMD_IC      0x04 /* Insert checksum */
#define E1000_TXD_CMD_RS      0x08 /* Report status */
#define E1000_TXD_CMD_DEXT    0x20 /* Descriptor extension: must stay clear for legacy */
#define E1000_TXD_CMD_VLE     0x40 /* VLAN enable */
#define E1000_TXD_CMD_IDE     0x80 /* Interrupt delay enable */
#define E1000_TXD_STAT_DD     0x01 /* Descriptor done */
#define E1000_TXD_STAT_EC     0x02 /* Excess collisions */
#define E1000_TXD_STAT_LC     0x04 /* Late collision */
