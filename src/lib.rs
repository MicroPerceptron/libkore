//! Kore's application-facing standard library.
//!
//! Import the package as `kore` to write `use kore::prelude::*` and
//! `#[kore::main]`. This is a userland facade, not the kernel package.
//! Filesystem support is enabled by default; attachment still requires the
//! process's actual founded face. No feature grants authority, installs an
//! allocator, or starts a second executor.
#![no_std]
#![deny(unsafe_code)]
#![doc = include_str!("../README.md")]

// These stable module paths also serve the runtime macros' facade expansion.
pub use kore_abi as abi;
pub use kore_app as app;
pub use kore_rt as runtime;
pub use kore_rt::{Bundle, ExitCode, main, spmd, task};
pub use kore_rt::{process, random};

/// Native driver substrate: capability-backed registers, bounded DMA and rings.
/// Hardware effects still require a device-specific owner and lifecycle policy.
#[cfg(feature = "driver")]
pub use kore_driver as driver;

/// Native clocks and deadlines; wall time never coerces into monotonic time.
pub mod time {
    pub use core::time::Duration;
    pub use kore_abi::clock::{TimerError, TimerResult, UnixTime, Uptime};
    pub use kore_rt::clock::{calibration, uptime, uptime_from, wall, wall_from};
    pub use kore_rt::{sleep_for, sleep_until};
}

#[cfg(feature = "fs")]
pub mod fs;

/// Common application types. Raw slots and kernel verbs stay explicitly named.
pub mod prelude {
    #[cfg(feature = "fs")]
    pub use crate::fs::{Fs, View};
    pub use core::time::Duration;
    pub use kore_abi::clock::{UnixTime, Uptime};
    pub use kore_app::prelude::*;
    pub use kore_rt::{ExitCode, entry::Rt};
}
