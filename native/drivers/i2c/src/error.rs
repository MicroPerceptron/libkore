//! The **one error vocabulary** every I2C controller answers in.
//!
//! Deliberately IP-neutral: nothing here names a DesignWare register or a
//! GENI command word. What a failing controller knows about *itself* rides
//! in two places that are opaque to this crate — [`Snapshot`], two
//! IP-defined register words, and the [`AbortDetail`] associated type.
//!
//! # Why the snapshot exists
//!
//! The metal target has no serial port: a failed bring-up gets one
//! photograph of the framebuffer and that is the entire diagnostic
//! surface. A bare `Err(Timeout)` costs a whole flash cycle to learn
//! nothing. So every failure that a register could explain carries the
//! registers, and the service's NACK-rung detail packers name them per
//! controller (#409's rung convention). The seam never interprets them —
//! it only guarantees they survive the trip.

use core::fmt;

use crate::master::RecoveryOutcome;

/// Two **IP-defined** register words captured at the instant of failure.
///
/// Neutral by construction: this crate never reads the fields, and no
/// consumer may assume a meaning for them without knowing which
/// [`I2cMaster`](crate::master::I2cMaster) produced them. The DesignWare
/// impl documents its own pairing (enable-status ‖ raw-interrupt for a
/// stuck enable, status ‖ RX-FIFO-level for a timeout); another IP will
/// pair different registers under the same two slots.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Snapshot(pub u32, pub u32);

impl Snapshot {
    /// An empty snapshot — for the failures no register explains (a
    /// caller-shaped request error, say).
    pub const NONE: Self = Self(0, 0);
}

/// What a controller's own abort word means, in the two terms the seam
/// needs and no more.
///
/// The IP decodes its abort register into a typed value; this trait is the
/// only thing the generic engine may ask of that value. Everything else —
/// which of eighteen DesignWare abort bits fired, in what order — stays in
/// the impl, where it can be printed but never branched on generically.
pub trait AbortDetail: Copy + fmt::Debug + PartialEq + Eq {
    /// Whether this abort is an **address NACK**: nobody answered at the
    /// target address.
    ///
    /// The one abort a *prober* must distinguish, because it means "no
    /// device here", which is a lawful answer, while every other abort
    /// means "the bus or the controller misbehaved", which is a fault.
    /// Enumeration would be impossible without this distinction and
    /// nothing else about the abort word is generic, which is exactly why
    /// this is the trait's only predicate.
    fn is_address_nack(&self) -> bool;

    /// The raw IP abort word, for the witness line.
    fn raw(&self) -> u32;
}

/// A request the seam refused before touching hardware — always a caller
/// bug, never a device condition.
///
/// Separated from the device failures because the two call for opposite
/// responses: a device failure may be worth a retry or a recovery, and a
/// malformed request never is. Folding them together is how a driver ends
/// up retrying its own bug against innocent silicon.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestError {
    /// The 7-bit target address was above `0x7F`.
    ///
    /// 10-bit addressing is not modelled: no HID-over-I2C device uses it,
    /// and admitting the address here while no impl programs the 10-bit
    /// mode bit would be a silent wrong-target write.
    AddressTooWide(u16),
    /// A transfer named no messages at all.
    Empty,
    /// The caller's read buffer is smaller than the bytes the transfer
    /// asked to read — refused up front, because a short buffer
    /// discovered mid-FIFO leaves the bus mid-sequence.
    ReadBufferTooSmall { need: usize, have: usize },
    /// The transfer's total byte count exceeds what one queued sequence
    /// may carry (see [`MAX_XFER_BYTES`](crate::bus::MAX_XFER_BYTES)).
    TooLong { bytes: usize },
}

/// Every way an I2C transfer can fail, in IP-neutral terms.
///
/// Generic over the controller's own abort detail so the *shape* is shared
/// while the *evidence* stays exact — the alternative, flattening every
/// IP's abort word into a common enum, would either lose detail or grow a
/// variant per IP, which is the silo this seam exists to prevent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum I2cError<A> {
    /// The controller did not identify as the IP this impl drives.
    ///
    /// Always fatal and always first: every other register read is
    /// meaningless if the window does not decode to the expected core,
    /// and a mis-mapped window otherwise presents as a stream of
    /// plausible-looking timeouts.
    Identity {
        /// What the identity register actually read.
        found: u32,
        /// What this impl requires.
        expected: u32,
    },
    /// The controller never reached the requested enable/disable state.
    EnableHandshake(Snapshot),
    /// The controller integration supplied a source clock for which this
    /// implementation has no authoritative timing table.
    UnsupportedInputClock(u32),
    /// The controller aborted the transfer and said why.
    Abort {
        detail: A,
        snapshot: Snapshot,
        recovery: RecoveryOutcome,
    },
    /// The transfer did not complete within its pacing budget.
    Timeout {
        snapshot: Snapshot,
        recovery: RecoveryOutcome,
    },
    /// A transfer was started while a previous abort stood uncleared.
    ///
    /// A programming error rather than a bus condition: the abort source
    /// register latches until it is cleared, so starting over it would
    /// read the *old* failure as the new one's cause. The engine refuses
    /// instead of recovering silently, because a driver that never
    /// noticed its own abort will otherwise mis-attribute every
    /// subsequent failure.
    AbortPending,
    /// The request was malformed.
    Request(RequestError),
}

impl<A: AbortDetail> I2cError<A> {
    /// Whether this failure is "no device answered at that address" —
    /// the one outcome a probe treats as information rather than fault.
    pub fn is_address_nack(&self) -> bool {
        matches!(self, Self::Abort { detail, .. } if detail.is_address_nack())
    }

    /// The registers captured at the failure, for the witness line.
    pub fn snapshot(&self) -> Snapshot {
        match self {
            Self::EnableHandshake(s)
            | Self::Abort { snapshot: s, .. }
            | Self::Timeout { snapshot: s, .. } => *s,
            Self::Identity { found, expected } => Snapshot(*found, *expected),
            Self::UnsupportedInputClock(_) | Self::AbortPending | Self::Request(_) => {
                Snapshot::NONE
            }
        }
    }

    /// Recovery result retained beside the original transfer evidence.
    pub const fn recovery(&self) -> Option<RecoveryOutcome> {
        match self {
            Self::Abort { recovery, .. } | Self::Timeout { recovery, .. } => Some(*recovery),
            _ => None,
        }
    }
}
