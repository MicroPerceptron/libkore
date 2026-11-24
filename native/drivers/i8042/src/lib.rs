//! # `kore-i8042` — the PS/2 controller driver (#419, epic #414)
//!
//! The internal keyboard of the metal target is PS/2 behind the embedded
//! controller. This crate is the userland driver for it: the controller
//! bring-up ladder and the scancode decoder, over I/O ports the task
//! reaches through a capability ([`IoPortGrant`], #416) and an interrupt
//! it parks on ([`IrqRoute`], #418).
//!
//! # No hardware double, deliberately
//!
//! There is no mock here and none is wanted (the 2026-07-24 mock-free
//! ruling). The I2C arc needed doubles because QEMU cannot model LPSS
//! I2C at all; **QEMU q35 has a real i8042**, so the bring-up ladder is
//! exercised against an actual controller in the boot gate, and
//! [`decode`] is a pure function tested with real byte sequences. A fake
//! would only prove the fake matches the driver.
//!
//! # Not the boottest path
//!
//! `src/boottest/shell.rs`'s `I8042` polls `0x64`/`0x60` with no
//! initialization, no claim, no interrupt, discards AUX bytes and assumes
//! translated set 1. It was right for an early-debug shell and is the
//! **anti-model** here: this driver initializes the controller, reads the
//! scancode set rather than assuming it, and is interrupt-driven.

#![no_std]

#[cfg(test)]
extern crate alloc;

pub mod bringup;
pub mod decode;
pub mod hid;
pub mod ident;
pub mod leds;
pub mod port;
#[rustfmt::skip]
mod scancode;

pub use bringup::{Brought, Rung, bring_up};
pub use decode::{Decoded, Decoder, Key, KeyEvent, Reply, ScancodeSet};
pub use hid::Contribution;
pub use leds::Indicators;
pub use ident::{HID_KEYBOARD, HID_MOUSE, eisa_id};
pub use port::{PortAccess, PortRegs};
