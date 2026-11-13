//! **The register block**: what a driver maps where hardware would have a
//! BAR.
//!
//! One page both sides map. Each register belongs to one side: the driver
//! never writes a device register and the device never writes a driver
//! register, so no word has two writers.
//!
//! The layout follows Virtio's own transports in what it carries (identity,
//! feature words, a status the driver steps through, the queue's size and
//! three addresses, a device configuration area) and departs from
//! `virtio-mmio` in one way that is forced: that layout selects what a
//! register shows by a write to another register, which only works when a
//! write traps. A write to shared memory does not, so nothing here is behind
//! a selector, and the status is two words: the one the driver writes and
//! the one the device answers with.
//!
//! # The doorbell
//!
//! The device does not see a write happen. It looks at the driver's
//! registers and at the queue when the driver rings its **doorbell**, the
//! transport's one notification, and it raises the driver's **interrupt**
//! when it has changed [`DEVICE_STATUS`] or used a chain. Virtio leaves
//! notification to each transport (a register on `virtio-mmio`, a hypercall
//! on channel I/O); here it is a notice on a channel, in both directions.
//!
//! The interrupt is a count the driver can sleep on: [`INTERRUPTS`] moves
//! each time the device raises one, so a driver reads it, does its work,
//! and parks until it differs from what it read. An interrupt raised
//! between the read and the park is not lost, because the park compares.
//!
//! A doorbell that announces a write to [`DRIVER_STATUS`] is always answered
//! with an interrupt, whether or not the device took the step: a driver
//! writes its status, rings, waits for the interrupt and reads
//! [`DEVICE_STATUS`]. A step the device refused is one its status does not
//! carry.
//!
//! # Bringing a device up
//!
//! 1. Check [`MAGIC`] and [`VERSION`], read [`DEVICE_ID`].
//! 2. Write [`STATUS_ACKNOWLEDGE`]` | `[`STATUS_DRIVER`] to
//!    [`DRIVER_STATUS`].
//! 3. Read [`DEVICE_FEATURES`], write the accepted subset to
//!    [`DRIVER_FEATURES`], add [`STATUS_FEATURES_OK`], ring the doorbell,
//!    and wait for the interrupt. [`DEVICE_STATUS`] carries
//!    `FEATURES_OK` only if the device accepted the subset.
//! 4. Lay the queue out in DMA memory; write [`QUEUE_NUM`], the three
//!    addresses and [`QUEUE_READY`].
//! 5. Add [`STATUS_DRIVER_OK`], ring, wait. The device is live when
//!    [`DEVICE_STATUS`] carries it.
//!
//! Writing zero to [`DRIVER_STATUS`] and ringing resets the device; the
//! reset is done when [`DEVICE_STATUS`] reads zero. A device that can no
//! longer serve sets [`STATUS_DEVICE_NEEDS_RESET`] there.

/// Bytes in the register block: one page.
pub const BYTES: u64 = 0x1000;

/// `u32`, device. [`MAGIC_VALUE`]: this is a kore virtual device.
pub const MAGIC: u64 = 0x00;
/// `u32`, device. [`VERSION_VALUE`]: the layout in this module.
pub const VERSION: u64 = 0x04;
/// `u32`, device. Which device model this is; see the model's own crate.
pub const DEVICE_ID: u64 = 0x08;
/// `u32`, device. Who made it: [`VENDOR_KORE`] for a device kore hosts.
pub const VENDOR_ID: u64 = 0x0c;
/// `u64`, device. The features the device offers.
pub const DEVICE_FEATURES: u64 = 0x10;
/// `u64`, driver. The features the driver accepts, a subset of the offer.
pub const DRIVER_FEATURES: u64 = 0x18;
/// `u32`, driver. The status the driver has reached (`STATUS_*`).
pub const DRIVER_STATUS: u64 = 0x20;
/// `u32`, device. The status the device has accepted, and its own
/// [`STATUS_DEVICE_NEEDS_RESET`].
pub const DEVICE_STATUS: u64 = 0x24;
/// `u32`, device. The largest queue the device takes.
pub const QUEUE_NUM_MAX: u64 = 0x28;
/// `u32`, driver. The queue's size: a power of two, at most the maximum.
pub const QUEUE_NUM: u64 = 0x2c;
/// `u32`, driver. Nonzero once the queue's size and addresses are written.
pub const QUEUE_READY: u64 = 0x30;
/// `u32`, device. Changes whenever the configuration area does.
pub const CONFIG_GENERATION: u64 = 0x34;
/// `u64`, driver. The descriptor table's device address.
pub const QUEUE_DESC: u64 = 0x38;
/// `u64`, driver. The available ring's device address.
pub const QUEUE_DRIVER: u64 = 0x40;
/// `u64`, driver. The used ring's device address.
pub const QUEUE_DEVICE: u64 = 0x48;
/// `u32`, device. Bytes of device configuration at [`CONFIG`].
pub const CONFIG_BYTES: u64 = 0x50;
/// `u32`, device. Interrupts the device has raised, wrapping: the word a
/// driver parks on.
pub const INTERRUPTS: u64 = 0x54;
/// The device configuration area, device-written, to the end of the page.
pub const CONFIG: u64 = 0x100;

/// `"kvdv"`, little-endian: a kore virtual device.
pub const MAGIC_VALUE: u32 = u32::from_le_bytes(*b"kvdv");
/// The layout's version.
pub const VERSION_VALUE: u32 = 1;
/// `"kore"`, little-endian.
pub const VENDOR_KORE: u32 = u32::from_le_bytes(*b"kore");

/// The driver has noticed the device (Virtio 1.3 §2.1).
pub const STATUS_ACKNOWLEDGE: u32 = 1;
/// The driver knows how to drive it.
pub const STATUS_DRIVER: u32 = 2;
/// The driver is set up and the device may serve.
pub const STATUS_DRIVER_OK: u32 = 4;
/// Feature negotiation is complete.
pub const STATUS_FEATURES_OK: u32 = 8;
/// The device cannot go on and must be reset. Device-written.
pub const STATUS_DEVICE_NEEDS_RESET: u32 = 64;
/// The driver has given up on the device.
pub const STATUS_FAILED: u32 = 128;

/// The device and driver follow Virtio 1.0 or later (feature bit 32). A
/// device offers it and a driver must accept it.
pub const F_VERSION_1: u64 = 1 << 32;
