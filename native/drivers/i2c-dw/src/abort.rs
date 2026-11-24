//! `IC_TX_ABRT_SOURCE`, decoded.
//!
//! The register latches one bit per cause and keeps them until read-cleared
//! via `IC_CLR_TX_ABRT`. More than one may stand at once, so the decode
//! names the **first** cause in databook bit order while carrying the raw
//! word for the witness line — reporting only "aborted" would throw away
//! the one distinction that changes what the caller should do.

use kore_i2c::AbortDetail;

/// A named `IC_TX_ABRT_SOURCE` cause.
///
/// Only the causes a **master** can provoke are named. The slave-mode bits
/// exist in the register and are deliberately not enumerated: this driver
/// programs `SLAVE_DISABLE`, so a slave abort means the core is not in the
/// state the driver believes, which [`DwAbortCause::Unexpected`] reports
/// honestly rather than dressing up as a known cause.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DwAbortCause {
    /// No device acknowledged the 7-bit address.
    AddrNack,
    /// The 10-bit address's first byte was not acknowledged.
    Addr10First,
    /// The 10-bit address's second byte was not acknowledged.
    Addr10Second,
    /// The target acknowledged its address, then NACKed a data byte.
    DataNack,
    /// A general call went unacknowledged.
    GeneralCallNack,
    /// A read was issued after a general call.
    GeneralCallRead,
    /// High-speed master code was acknowledged when it should not be.
    HighSpeedAckDet,
    /// The START byte was acknowledged.
    StartByteAckDet,
    /// High-speed mode needs a restart that is disabled.
    HighSpeedNoRestart,
    /// A START byte needs a restart that is disabled.
    StartByteNoRestart,
    /// A 10-bit read needs a restart that is disabled.
    Read10NoRestart,
    /// A master operation was attempted with master mode disabled.
    MasterDisabled,
    /// Arbitration was lost to another master.
    ArbitrationLost,
    /// The core reported a cause outside the master set.
    Unexpected,
}

impl DwAbortCause {
    /// Decode the lowest set bit, in databook order.
    const fn from_word(w: u32) -> Self {
        // Ordered by bit index so a multi-cause word always decodes the
        // same way. `match` on the trailing-zeros index rather than a
        // chain of `if`s so a missed bit is a compile-time gap, not a
        // silently-taken fallthrough.
        match w.trailing_zeros() {
            0 => Self::AddrNack,
            1 => Self::Addr10First,
            2 => Self::Addr10Second,
            3 => Self::DataNack,
            4 => Self::GeneralCallNack,
            5 => Self::GeneralCallRead,
            6 => Self::HighSpeedAckDet,
            7 => Self::StartByteAckDet,
            8 => Self::HighSpeedNoRestart,
            9 => Self::StartByteNoRestart,
            10 => Self::Read10NoRestart,
            11 => Self::MasterDisabled,
            12 => Self::ArbitrationLost,
            _ => Self::Unexpected,
        }
    }
}

/// One decoded abort: the named first cause, and the raw register.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DwAbort {
    /// The first cause in databook bit order.
    pub cause: DwAbortCause,
    /// The whole `IC_TX_ABRT_SOURCE` word.
    pub raw: u32,
}

impl DwAbort {
    /// Decode a register read.
    pub const fn decode(raw: u32) -> Self {
        Self {
            cause: DwAbortCause::from_word(raw),
            raw,
        }
    }
}

impl AbortDetail for DwAbort {
    fn is_address_nack(&self) -> bool {
        matches!(
            self.cause,
            DwAbortCause::AddrNack | DwAbortCause::Addr10First | DwAbortCause::Addr10Second
        )
    }

    fn raw(&self) -> u32 {
        self.raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_named_bit_decodes_to_its_own_cause() {
        let expect = [
            (0, DwAbortCause::AddrNack),
            (1, DwAbortCause::Addr10First),
            (2, DwAbortCause::Addr10Second),
            (3, DwAbortCause::DataNack),
            (4, DwAbortCause::GeneralCallNack),
            (5, DwAbortCause::GeneralCallRead),
            (6, DwAbortCause::HighSpeedAckDet),
            (7, DwAbortCause::StartByteAckDet),
            (8, DwAbortCause::HighSpeedNoRestart),
            (9, DwAbortCause::StartByteNoRestart),
            (10, DwAbortCause::Read10NoRestart),
            (11, DwAbortCause::MasterDisabled),
            (12, DwAbortCause::ArbitrationLost),
        ];
        for (bit, cause) in expect {
            let a = DwAbort::decode(1 << bit);
            assert_eq!(a.cause, cause, "bit {bit}");
            assert_eq!(a.raw, 1 << bit);
        }
    }

    #[test]
    fn a_slave_side_bit_is_reported_as_unexpected_not_mislabelled() {
        // Bits 13..=17 are the slave-mode causes this driver's
        // configuration should make unreachable.
        for bit in 13..=17 {
            assert_eq!(DwAbort::decode(1 << bit).cause, DwAbortCause::Unexpected);
        }
        assert_eq!(DwAbort::decode(0).cause, DwAbortCause::Unexpected);
    }

    #[test]
    fn a_multi_cause_word_decodes_deterministically_and_keeps_the_raw() {
        // Address NACK plus arbitration lost: the lower bit wins, and the
        // full word survives for the witness line.
        let a = DwAbort::decode((1 << 0) | (1 << 12));
        assert_eq!(a.cause, DwAbortCause::AddrNack);
        assert_eq!(a.raw, 0x1001);
        assert!(a.is_address_nack());
    }

    #[test]
    fn only_the_address_causes_read_as_an_address_nack() {
        assert!(DwAbort::decode(1 << 0).is_address_nack());
        assert!(DwAbort::decode(1 << 1).is_address_nack());
        assert!(DwAbort::decode(1 << 2).is_address_nack());
        // A data NACK is a device rejecting a command, not an absent
        // device — conflating them would make a probe report success.
        assert!(!DwAbort::decode(1 << 3).is_address_nack());
        assert!(!DwAbort::decode(1 << 12).is_address_nack());
    }
}
