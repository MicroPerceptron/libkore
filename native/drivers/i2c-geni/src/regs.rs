//! Qualcomm GENI serial-engine registers used by the FIFO I2C personality.
//! Offsets and bits are transcribed from Linux `geni-se.h`,
//! `qcom-geni-se.c`, and `i2c-qcom-geni.c`.

pub const GENI_FORCE_DEFAULT_REG: usize = 0x20;
pub const GENI_OUTPUT_CTRL: usize = 0x24;
pub const SE_GENI_CGC_CTRL: usize = 0x28;
pub const SE_GENI_STATUS: usize = 0x40;
pub const GENI_SER_M_CLK_CFG: usize = 0x48;
pub const GENI_FW_REVISION_RO: usize = 0x68;
pub const SE_GENI_CLK_SEL: usize = 0x7c;
pub const SE_GENI_BYTE_GRAN: usize = 0x254;
pub const SE_GENI_DMA_MODE_EN: usize = 0x258;
pub const SE_GENI_TX_PACKING_CFG0: usize = 0x260;
pub const SE_GENI_TX_PACKING_CFG1: usize = 0x264;
pub const SE_I2C_TX_TRANS_LEN: usize = 0x26c;
pub const SE_I2C_RX_TRANS_LEN: usize = 0x270;
pub const SE_I2C_SCL_COUNTERS: usize = 0x278;
pub const SE_GENI_RX_PACKING_CFG0: usize = 0x284;
pub const SE_GENI_RX_PACKING_CFG1: usize = 0x288;
pub const SE_GENI_M_CMD0: usize = 0x600;
pub const SE_GENI_M_CMD_CTRL_REG: usize = 0x604;
pub const SE_GENI_M_IRQ_STATUS: usize = 0x610;
pub const SE_GENI_M_IRQ_EN: usize = 0x614;
pub const SE_GENI_M_IRQ_CLEAR: usize = 0x618;
pub const SE_GENI_S_IRQ_CLEAR: usize = 0x648;
pub const SE_GENI_TX_FIFO: usize = 0x700;
pub const SE_GENI_RX_FIFO: usize = 0x780;
pub const SE_GENI_TX_FIFO_STATUS: usize = 0x800;
pub const SE_GENI_RX_FIFO_STATUS: usize = 0x804;
pub const SE_GENI_TX_WATERMARK: usize = 0x80c;
pub const SE_GENI_RX_WATERMARK: usize = 0x810;
pub const SE_GENI_RX_RFR_WATERMARK: usize = 0x814;
pub const SE_GENI_IOS: usize = 0x908;
pub const SE_GENI_M_GP_LENGTH: usize = 0x910;
pub const SE_HW_PARAM_0: usize = 0xe24;
pub const SE_HW_PARAM_1: usize = 0xe28;
pub const SE_GSI_EVENT_EN: usize = 0xe18;
pub const SE_IRQ_EN: usize = 0xe1c;
pub const SE_DMA_GENERAL_CFG: usize = 0xe30;

pub const PROTOCOL_I2C: u32 = 3;
pub const FW_PROTOCOL_MASK: u32 = 0xff << 8;
pub const FW_PROTOCOL_SHIFT: u32 = 8;
pub const M_CMD_ACTIVE: u32 = 1 << 0;
pub const M_OPCODE_SHIFT: u32 = 27;
pub const SLAVE_ADDR_SHIFT: u32 = 9;
pub const STOP_STRETCH: u32 = 1 << 2;

pub const I2C_WRITE: u32 = 1;
pub const I2C_READ: u32 = 2;
pub const I2C_WRITE_READ: u32 = 3;

pub const IRQ_CMD_DONE: u32 = 1 << 0;
pub const IRQ_CMD_OVERRUN: u32 = 1 << 1;
pub const IRQ_ILLEGAL_CMD: u32 = 1 << 2;
pub const IRQ_CMD_FAILURE: u32 = 1 << 3;
pub const IRQ_CMD_CANCEL: u32 = 1 << 4;
pub const IRQ_CMD_ABORT: u32 = 1 << 5;
pub const IRQ_TIMESTAMP: u32 = 1 << 6;
pub const IRQ_GP_1: u32 = 1 << 10;
pub const IRQ_GP_3: u32 = 1 << 12;
pub const IRQ_GP_4: u32 = 1 << 13;
pub const IRQ_IO_DATA_DEASSERT: u32 = 1 << 22;
pub const IRQ_IO_DATA_ASSERT: u32 = 1 << 23;
pub const IRQ_RX_FIFO_RD_ERR: u32 = 1 << 24;
pub const IRQ_RX_FIFO_WR_ERR: u32 = 1 << 25;
pub const IRQ_RX_WATERMARK: u32 = 1 << 26;
pub const IRQ_RX_LAST: u32 = 1 << 27;
pub const IRQ_TX_FIFO_RD_ERR: u32 = 1 << 28;
pub const IRQ_TX_FIFO_WR_ERR: u32 = 1 << 29;
pub const IRQ_TX_WATERMARK: u32 = 1 << 30;
/// `M_COMMON_GENI_M_IRQ_EN` from the shared GENI implementation. These are
/// status-generation enables; timestamp and line-observation sources are not
/// themselves I2C transfer failures.
pub const IRQ_COMMON: u32 = IRQ_CMD_OVERRUN
    | IRQ_ILLEGAL_CMD
    | IRQ_CMD_FAILURE
    | IRQ_CMD_CANCEL
    | IRQ_CMD_ABORT
    | IRQ_TIMESTAMP
    | IRQ_IO_DATA_DEASSERT
    | IRQ_IO_DATA_ASSERT
    | IRQ_RX_FIFO_RD_ERR
    | IRQ_RX_FIFO_WR_ERR
    | IRQ_TX_FIFO_RD_ERR
    | IRQ_TX_FIFO_WR_ERR;
pub const IRQ_ERRORS: u32 = IRQ_CMD_OVERRUN
    | IRQ_ILLEGAL_CMD
    | IRQ_CMD_FAILURE
    | IRQ_CMD_CANCEL
    | IRQ_CMD_ABORT
    | IRQ_GP_1
    | IRQ_GP_3
    | IRQ_GP_4
    | IRQ_RX_FIFO_RD_ERR
    | IRQ_RX_FIFO_WR_ERR
    | IRQ_TX_FIFO_RD_ERR
    | IRQ_TX_FIFO_WR_ERR;
pub const IRQ_ENABLE: u32 = IRQ_COMMON | IRQ_GP_1 | IRQ_GP_3 | IRQ_GP_4;
pub const IRQ_TRANSFER: u32 = IRQ_CMD_DONE | IRQ_RX_WATERMARK | IRQ_RX_LAST | IRQ_TX_WATERMARK;

pub const RX_LAST: u32 = 1 << 31;
pub const RX_LAST_VALID_MASK: u32 = 0x7 << 28;
pub const RX_LAST_VALID_SHIFT: u32 = 28;
pub const RX_FIFO_WORDS_MASK: u32 = 0x01ff_ffff;
pub const TX_FIFO_WORDS_MASK: u32 = 0x0fff_ffff;
/// QUP versions before 3.10 encode FIFO depth in bits 21:16. 3.10 and
/// later extend the field through bit 23. The wrapper version register which
/// selects between them is outside a serial element's bounded MMIO aperture.
pub const FIFO_DEPTH_MASK: u32 = 0x3f << 16;
pub const FIFO_DEPTH_EXTENDED_MASK: u32 = 0xc0 << 16;
pub const FIFO_DEPTH_SHIFT: u32 = 16;
pub const FIFO_WIDTH_MASK: u32 = 0x3f << 24;
pub const FIFO_WIDTH_SHIFT: u32 = 24;

pub const SER_CLK_ENABLE: u32 = 1;
pub const CLK_DIV_SHIFT: u32 = 4;
pub const SCL_HIGH_SHIFT: u32 = 20;
pub const SCL_LOW_SHIFT: u32 = 10;
pub const DEFAULT_CGC_ENABLE: u32 = 0x7f;
pub const DEFAULT_IO_OUTPUT_ENABLE: u32 = 0x7f;
pub const FORCE_DEFAULT: u32 = 1;
pub const DMA_INTERFACE_CLOCKS: u32 = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3);
pub const PRIMARY_IRQ_ROUTE: u32 = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3);
pub const M_CMD_CANCEL: u32 = 1 << 0;
pub const M_CMD_ABORT: u32 = 1 << 1;
