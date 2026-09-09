//! Window protocol and client-side transport, independent of compositor policy.
//! All IDs are data; authority is the authenticated endpoint and its leases.
#![no_std]

pub mod appearance;
#[cfg(feature = "runtime")]
pub mod channel;
#[cfg(feature = "runtime")]
pub mod client;
pub mod command;
#[cfg(feature = "runtime")]
pub mod configure;
#[cfg(feature = "runtime")]
pub mod control;
#[cfg(feature = "runtime")]
pub mod event;
mod face;
#[cfg(feature = "runtime")]
pub mod relay;
#[cfg(feature = "runtime")]
pub mod wire;

/// Desktop window admission budget, including the terminal renderer.
/// Six windows fit the retained chrome scene and two buffers each fit the
/// sixteen-slot surface engine. Controllers reserve separate serving peers.
pub const BOOTSTRAP_WINDOW_SLOTS: usize = kore_abi::task::DESKTOP_BOOTSTRAP_WINDOW_SLOTS;

/// Bounded number of in-flight surface commits.
#[cfg(feature = "runtime")]
pub const COMMIT_CAP: u32 = 32;

#[cfg(feature = "runtime")]
pub use control::{DisplayRequest, DisplayResponse};
#[cfg(feature = "runtime")]
pub use control::{DisplayRequest as Request, DisplayResponse as Response, WindowWire};
pub use face::WindowFace;
