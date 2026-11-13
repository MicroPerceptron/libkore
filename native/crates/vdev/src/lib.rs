//! # kore-vdev: virtual devices
//!
//! A virtual device is a device with no hardware behind it, hosted by a
//! userland service. A driver reaches it the way it reaches hardware, and
//! each of the three things a driver holds is ordinary shared memory or an
//! ordinary notice:
//!
//! - **registers** are a lease both sides map: the driver reads and writes
//!   them as it would a BAR;
//! - **DMA** is a lease the driver owns and lends to the device. A device
//!   address is an offset into it, and the device can touch nothing outside
//!   it ([`mem`]);
//! - **an interrupt** is a notice the driver parks on.
//!
//! No kernel mechanism is added for any of it, and nothing is emulated: the
//! mappings are real, so the MMU is what a driver's access goes through.
//!
//! What this crate holds is the part every virtual device shares:
//!
//! - [`mem`], the device's view of memory it shares with a driver;
//! - [`regs`], the register block a driver maps and steps through;
//! - [`ring`], the split virtqueue laid out in the driver's memory;
//! - [`transport`], which runs the three for one driver on each doorbell;
//! - [`device`], the contract a device model implements. A model is a
//!   separate crate (`kore-vdev-accel` is the first) and sees none of the
//!   above: it is handed a request to read and a response to write;
//! - [`driver`], the driver's half of the same registers and queue, for a
//!   driver written in Rust. A driver in C writes those steps by hand
//!   against the layout in [`regs`].

#![no_std]

extern crate alloc;

pub mod device;
pub mod driver;
pub mod mem;
pub mod regs;
pub mod ring;
pub mod transport;

#[cfg(test)]
extern crate std;
