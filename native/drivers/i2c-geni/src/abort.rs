use kore_i2c::AbortDetail;

/// A decoded GENI I2C master failure. Multiple raw IRQ bits can stand; the
/// named cause follows the upstream driver's priority and the raw word is
/// retained for metal diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeniAbortCause {
    AddressNack,
    DataNack,
    BusProtocol,
    ArbitrationLost,
    CommandOverrun,
    IllegalCommand,
    CommandFailure,
    CommandCancelled,
    CommandAborted,
    Unexpected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeniAbort {
    pub cause: GeniAbortCause,
    pub raw: u32,
}

impl GeniAbort {
    pub const fn decode(raw: u32, gp_length: u32, writing: bool) -> Self {
        use crate::regs;
        let cause = if raw & regs::IRQ_GP_1 != 0 {
            if gp_length == 0 || !writing {
                GeniAbortCause::AddressNack
            } else {
                GeniAbortCause::DataNack
            }
        } else if raw & regs::IRQ_GP_3 != 0 {
            GeniAbortCause::BusProtocol
        } else if raw & regs::IRQ_GP_4 != 0 {
            GeniAbortCause::ArbitrationLost
        } else if raw & regs::IRQ_CMD_OVERRUN != 0 {
            GeniAbortCause::CommandOverrun
        } else if raw & regs::IRQ_ILLEGAL_CMD != 0 {
            GeniAbortCause::IllegalCommand
        } else if raw & regs::IRQ_CMD_FAILURE != 0 {
            GeniAbortCause::CommandFailure
        } else if raw & regs::IRQ_CMD_CANCEL != 0 {
            GeniAbortCause::CommandCancelled
        } else if raw & regs::IRQ_CMD_ABORT != 0 {
            GeniAbortCause::CommandAborted
        } else {
            GeniAbortCause::Unexpected
        };
        Self { cause, raw }
    }
}

impl AbortDetail for GeniAbort {
    fn is_address_nack(&self) -> bool {
        self.cause == GeniAbortCause::AddressNack
    }

    fn raw(&self) -> u32 {
        self.raw
    }
}
