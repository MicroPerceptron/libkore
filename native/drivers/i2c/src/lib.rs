//! **The kore I2C master seam.**
//!
//! One generic engine over one contract trait, with a concrete controller
//! chosen by type instantiation — the tree's standard seam construction
//! (`kore-trait-architecture` §1), applied to I2C because the fleet has
//! more than one controller IP and will not stop at two:
//!
//! | Machine | Controller | Where it lives |
//! |---|---|---|
//! | Intel (LPSS, PCI-mode) | Synopsys DesignWare | `kore-i2c-dw` |
//! | AMD (`AMDI0010`, ACPI-mode) | Synopsys DesignWare | `kore-i2c-dw` |
//! | Qualcomm (`QCOM0C10`) | GENI serial engine | `kore-i2c-geni` |
//!
//! ```text
//!   kore-i2c-hid ──holds──> I2cBus            (protocol, IP-blind)
//!                              ▲
//!                          Bus<M, P>          (engine bound to regs + pacing)
//!                              │
//!                       I2cEngine<M>          (the pump, written once)
//!                              │
//!                          I2cMaster          (the contract)
//!                          ╱        ╲
//!                    DwMaster       GeniMaster
//! ```
//!
//! The seam's whole value is that the protocol driver above it and the
//! service above that are written once. A `#[cfg]` or a `match ip` in this
//! crate would give that away; there are none, and there must be none.

#![no_std]
#![forbid(unsafe_code)]

#[cfg(any(test, feature = "mock"))]
extern crate alloc;

pub mod bus;
pub mod engine;
pub mod error;
pub mod master;
pub mod pacing;

#[cfg(any(test, feature = "mock"))]
pub mod mock;

pub use bus::{I2cBus, MAX_ADDR, MAX_XFER_BYTES, Xfer};
pub use engine::{Bus, I2cEngine, I2cStats};
pub use error::{AbortDetail, I2cError, RequestError, Snapshot};
pub use master::{
    Budget, BusConfig, Counts, I2cMaster, Pacing, RateBudget, RecoveryOutcome, RecoveryStart,
    Speed, XferState,
};
pub use pacing::DeadlinePacing;
