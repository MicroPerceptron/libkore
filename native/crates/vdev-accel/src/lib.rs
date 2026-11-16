//! # kore-vdev-accel: the virtual accelerator
//!
//! A virtio-accel 1.0 device with no hardware behind it. It exists so a
//! driver can be written, run and debugged on a machine that has no
//! accelerator, against a device whose every answer is known in advance.
//!
//! What it computes is deliberately small and exact: six functions of one
//! FP8 (E4M3) value, elementwise over a buffer. See [`fp8`]. An E4M3 value
//! is a byte, so each function is a 256-entry table, and the table *is* the
//! device's definition. A driver is right when it reads back those bytes.
//!
//! The device is an instance of the CPU backend, `kore-svc-accel-spmd`: its
//! programs are that backend's registered kernels, and this crate supplies
//! the registry ([`fp8::KERNELS`]). What a driver exchanges with it is the
//! protocol's own frames, decoded by the protocol's own command engine
//! ([`device::AccelDevice`]). Nothing here knows how a frame reaches the
//! device; that is the transport's business.

#![no_std]

pub mod device;
pub mod fp8;

#[cfg(test)]
extern crate std;
