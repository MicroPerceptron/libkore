//! The DesignWare I2C register map.
//!
//! **Every constant here is transcribed from the Synopsys DW_apb_i2c
//! databook and cross-checked against Linux's
//! `drivers/i2c/busses/i2c-designware-core.h`.** None is derived, guessed,
//! or inferred from a neighbouring value — a plausible-looking register
//! encoding that is wrong is the single most expensive class of bug on a
//! machine whose only diagnostic is a photograph of the screen.
//!
//! Offsets are byte offsets from the controller's register window base.

/// Control. Master enable, speed, restart enable, slave disable.
pub const IC_CON: usize = 0x00;
/// Target address for the next transfer.
pub const IC_TAR: usize = 0x04;
/// The command/data FIFO port: writes queue a command, reads pop RX.
pub const IC_DATA_CMD: usize = 0x10;
/// Standard-mode SCL high count.
pub const IC_SS_SCL_HCNT: usize = 0x14;
/// Standard-mode SCL low count.
pub const IC_SS_SCL_LCNT: usize = 0x18;
/// Fast-mode SCL high count.
pub const IC_FS_SCL_HCNT: usize = 0x1c;
/// Fast-mode SCL low count.
pub const IC_FS_SCL_LCNT: usize = 0x20;
/// Masked interrupt status.
pub const IC_INTR_STAT: usize = 0x2c;
/// Interrupt mask — which conditions assert the controller's interrupt.
pub const IC_INTR_MASK: usize = 0x30;
/// Raw interrupt status, regardless of the mask. The polled path reads
/// this, so masking never hides progress from the state machine.
pub const IC_RAW_INTR_STAT: usize = 0x34;
/// RX FIFO threshold.
pub const IC_RX_TL: usize = 0x38;
/// TX FIFO threshold.
pub const IC_TX_TL: usize = 0x3c;
/// Read-to-clear: clears all clearable interrupts.
pub const IC_CLR_INTR: usize = 0x40;
/// Read-to-clear: TX abort.
pub const IC_CLR_TX_ABRT: usize = 0x54;
/// Read-to-clear: STOP detected.
pub const IC_CLR_STOP_DET: usize = 0x60;
/// Enable/disable the controller, and the transfer-abort request.
pub const IC_ENABLE: usize = 0x6c;
/// FIFO and activity status.
pub const IC_STATUS: usize = 0x70;
/// TX FIFO level.
pub const IC_TXFLR: usize = 0x74;
/// RX FIFO level.
pub const IC_RXFLR: usize = 0x78;
/// SDA hold time, in input-clock ticks.
pub const IC_SDA_HOLD: usize = 0x7c;
/// Why the last transfer aborted — one bit per cause.
pub const IC_TX_ABRT_SOURCE: usize = 0x80;
/// Whether the controller is actually enabled (the handshake register).
pub const IC_ENABLE_STATUS: usize = 0x9c;
/// FIFO depths and feature encoding.
pub const IC_COMP_PARAM_1: usize = 0xf4;
/// Core version, as four packed ASCII bytes.
pub const IC_COMP_VERSION: usize = 0xf8;
/// Component type — the identity register.
pub const IC_COMP_TYPE: usize = 0xfc;

/// What [`IC_COMP_TYPE`] reads on a DesignWare I2C core: ASCII `"DW"`
/// followed by `0x0140`.
pub const COMP_TYPE_VALUE: u32 = 0x4457_0140;

/// The lowest [`IC_COMP_VERSION`] that implements [`IC_SDA_HOLD`] —
/// ASCII `"111*"`. Writing the register on an older core writes a
/// reserved address.
pub const SDA_HOLD_MIN_VERSION: u32 = 0x3131_312A;

/// [`IC_CON`] bits.
pub mod con {
    /// Master mode enable.
    pub const MASTER: u32 = 1 << 0;
    /// Speed field, bits 2:1.
    pub const SPEED_MASK: u32 = 0b11 << 1;
    /// Standard mode (100 kHz).
    pub const SPEED_STD: u32 = 0b01 << 1;
    /// Fast mode (400 kHz).
    pub const SPEED_FAST: u32 = 0b10 << 1;
    /// 10-bit addressing in master mode. Never set by this driver — the
    /// seam refuses addresses above 0x7F rather than programming it.
    pub const ADDR_10BIT_MASTER: u32 = 1 << 4;
    /// Allow repeated START. Required: a write-then-read without it
    /// becomes two transactions with a STOP between them.
    pub const RESTART_EN: u32 = 1 << 5;
    /// Disable the slave block. Required in master mode, or the core
    /// answers its own transfers.
    pub const SLAVE_DISABLE: u32 = 1 << 6;
}

/// [`IC_DATA_CMD`] bits.
pub mod cmd {
    /// The data byte, bits 7:0.
    pub const DATA_MASK: u32 = 0xff;
    /// This command is a **read** rather than a write.
    pub const READ: u32 = 1 << 8;
    /// Issue a STOP after this byte.
    pub const STOP: u32 = 1 << 9;
    /// Issue a repeated START before this byte.
    pub const RESTART: u32 = 1 << 10;
}

/// [`IC_ENABLE`] bits.
pub mod enable {
    /// Controller enable.
    pub const ENABLE: u32 = 1 << 0;
    /// Request an abort of the in-flight transfer.
    pub const ABORT: u32 = 1 << 1;
}

/// [`IC_STATUS`] bits.
pub mod status {
    /// Any activity on the bus.
    pub const ACTIVITY: u32 = 1 << 0;
    /// TX FIFO not full — another command may be queued.
    pub const TFNF: u32 = 1 << 1;
    /// TX FIFO empty.
    pub const TFE: u32 = 1 << 2;
    /// RX FIFO not empty — a byte may be popped.
    pub const RFNE: u32 = 1 << 3;
    /// RX FIFO full.
    pub const RFF: u32 = 1 << 4;
    /// The master state machine is mid-transfer.
    pub const MST_ACTIVITY: u32 = 1 << 5;
}

/// [`IC_RAW_INTR_STAT`] / [`IC_INTR_MASK`] bits.
pub mod intr {
    /// RX FIFO underflow.
    pub const RX_UNDER: u32 = 1 << 0;
    /// RX FIFO overflow — bytes were lost.
    pub const RX_OVER: u32 = 1 << 1;
    /// RX FIFO reached its threshold.
    pub const RX_FULL: u32 = 1 << 2;
    /// TX FIFO overflow.
    pub const TX_OVER: u32 = 1 << 3;
    /// TX FIFO at or below its threshold (level; TX_TL=0 means empty).
    pub const TX_EMPTY: u32 = 1 << 4;
    /// The transfer was aborted; [`super::IC_TX_ABRT_SOURCE`] says why.
    pub const TX_ABRT: u32 = 1 << 6;
    /// A STOP was detected — the transfer is over.
    pub const STOP_DET: u32 = 1 << 9;

    /// What a master must be woken for: completion, an abort, and the
    /// two FIFO thresholds that mean "there is work to do now".
    ///
    /// This is the upper bound of progress sources, not a mask to leave
    /// armed unconditionally: TX_EMPTY is useful only while a command
    /// can still be queued, and every source is masked after completion
    /// or failure. RX/STOP/abort wakeups remain armed during a transfer
    /// even when TX issuance is finished or waiting for RX credits.
    pub const MASTER_MASK: u32 = RX_FULL | TX_EMPTY | TX_ABRT | STOP_DET | RX_OVER | RX_UNDER;
}
