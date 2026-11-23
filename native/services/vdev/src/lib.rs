//! # kore-svc-vdev: the virtual-device service
//!
//! A virtual device is a device with no hardware behind it (`kore-vdev`).
//! This service is where one lives: it hosts a device model for each driver
//! that asks for one, and what passes between a driver and its device is
//! memory the two share and two wakes.
//!
//! - **Registers** are a page of the driver's own face with the service, so
//!   the driver has them mapped from its first instruction and the device
//!   writes them when the driver opens it ([`VdevRequest::Open`]).
//! - **DMA** is memory the driver owns and lends ([`VdevRequest::Lend`]).
//!   A device address is an offset into it; the service maps exactly what
//!   was lent and can reach nothing else of the driver's.
//! - **The doorbell** is a notice on the face's kick. It carries nothing:
//!   the service looks at every device it hosts when one arrives.
//! - **The interrupt** is the register block's interrupt count
//!   (`kore_vdev::regs::INTERRUPTS`), which the service moves and then
//!   wakes by the memory. A driver parks on the word.
//!
//! No kernel mechanism is specific to any of it. The face is one exec
//! mints for a program at launch and retires when the program ends, as it
//! does the power face ([`service`] documents the administration half).
//!
//! The device table ([`host`]) is plain data over `kore-vdev`'s transport
//! and is tested off metal; [`service`] is the shell that feeds it from the
//! runtime.

#![no_std]
// The peer table grows with the programs that hold a face, from the
// service's own heap through the `Allocator` seam.
#![cfg_attr(feature = "host", feature(allocator_api))]

extern crate alloc;

#[cfg(test)]
extern crate std;

pub mod client;
pub mod face;
#[cfg(feature = "host")]
pub mod host;
#[cfg(feature = "host")]
pub mod service;
pub mod wire;

#[cfg(target_os = "none")]
pub use client::{ProcessVdev, ThreadVdev, attach_process_face, attach_thread_face};
pub use client::{RingVdev, RingVdevAdmin, THREAD_SHARES, Waited, process_face};
#[cfg(not(feature = "host"))]
pub use face::VdevService;
#[cfg(feature = "host")]
pub use service::{VdevEndowment, VdevService};
pub use wire::{VdevRefusal, VdevRequest, VdevResponse};
