//! **HID-over-I2C** — the Microsoft protocol, transport-generic.
//!
//! This crate holds everything about *the protocol* and nothing about any
//! controller. Its only view of the bus is [`kore_i2c::I2cBus`]'s two
//! methods, which is what lets one implementation serve every machine in
//! the fleet:
//!
//! | Machine | Controller | This crate |
//! |---|---|---|
//! | Intel Lunar Lake | DesignWare (LPSS, PCI-mode) | unchanged |
//! | AMD | DesignWare (`AMDI0010`, ACPI-mode) | unchanged |
//! | Qualcomm Snapdragon X | GENI serial engine | unchanged |
//!
//! That is not a hope. HID-over-I2C is a published standard and every one
//! of those machines reaches its input devices through a `PNP0C50`
//! compatible id — including the Qualcomm box's internal *keyboard*
//! (`ECKB`), which is an I2C-HID device where the other two machines have
//! a touchpad.
//!
//! # Shape
//!
//! - [`descriptor`] — the 30-byte block, validated.
//! - [`command`] — command encoding, including the report-id sentinel.
//! - [`framing`] — the length prefix, and the three things that are not
//!   a report.
//! - [`bringup`] — the ladder, as a state machine that **demands its
//!   delays** rather than sleeping.
//! - [`features`] — descriptor-declared digitizer configuration, via bounded
//!   GET / read-modify-write SET feature reports before input goes online.
//! - [`table`] — the parse-once extraction table and the hot path,
//!   including **multi-contact** decode: a digitizer frame carries one
//!   record per finger, so a five-finger touchpad and a ten-finger
//!   touchscreen are the same code path as a one-finger one.
//! - `parse` — report descriptor → table, behind the `alloc` feature.
//!
//! # Why bring-up demands its delays
//!
//! Bring-up waits 60 ms for a power settle and up to a second for a reset
//! acknowledgement. Performing those waits inside the driver would make it
//! a driver that spins, and nothing above could undo that. So they are
//! [`bringup::Step::Delay`] demands the caller satisfies from whatever
//! deadline discipline it already owns.

#![no_std]
#![forbid(unsafe_code)]

pub mod bringup;
pub mod command;
pub mod descriptor;
pub mod features;
pub mod framed;
pub mod framing;
pub mod table;

#[cfg(feature = "alloc")]
pub mod parse;

pub use bringup::{BringUp, BringUpError, Quirks, Step};
pub use command::{Command, Opcode, PowerState, ReportType};
pub use descriptor::{DescriptorError, HidDescriptor};
pub use features::{FeatureError, FeaturePhase, FeaturePlan, FeatureSetup};
pub use framing::{Frame, FramingError};
pub use table::{ExtractionTable, Field, ReportShape, TableError};

#[cfg(test)]
mod tests;
