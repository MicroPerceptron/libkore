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

/// How long an application window may take to leave after its controller's
/// close request was accepted (`Event::CloseRequested` enqueued): its current
/// frame, its own orderly shutdown, and resident-worker and surface
/// retirement. Past it the controller revokes the window task. The
/// controller's clock (`kore-window-session`) and any app that budgets its own
/// close against it (a `term` window asking its program to stop) read this
/// one value, so the app's budget cannot outlast the controller's.
pub const APPLICATION_CLOSE_GRACE_NS: u64 = 15_000_000_000;

/// The prefix a launcher puts before a hosted program's command line in its
/// terminal window's arguments to ask for the conversation presentation
/// (`term --chat <command>`, `res/SHELL_TERMINAL.md` §6): the same stream
/// host, drawn as a chat with a Stop control instead of a grid.  The window
/// strips it from its title.
pub const HOSTED_CHAT_PRESENTATION: &str = "--chat ";

#[cfg(feature = "runtime")]
pub use control::{DisplayRequest, DisplayResponse};
#[cfg(feature = "runtime")]
pub use control::{DisplayRequest as Request, DisplayResponse as Response, WindowWire};
pub use face::WindowFace;
