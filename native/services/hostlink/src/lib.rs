//! Transport-independent founded HostLink service core.
//!
//! Device discovery, Virtio composition, and entry/founding wrappers live
//! outside this crate. The core owns bounded protocol state and binds the
//! structural broker/workspace routes plus generation-installed agent
//! mailbox endpoints supplied locally.

#![no_std]

#[cfg(test)]
extern crate std;

mod client;
#[cfg(target_os = "none")]
mod entry;
mod front;
mod frontend;
mod mailbox;
mod retirement;
mod service;
pub mod shmem;
mod supervisor;
pub mod virtio;
mod wire;

pub use client::*;
#[cfg(target_os = "none")]
pub use entry::*;
pub use front::*;
pub use frontend::*;
pub use mailbox::*;
pub use retirement::*;
pub use service::*;
pub use supervisor::*;
pub use wire::*;
