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
/// Process-local atomic-word waits, including native async waits.
pub use kore_rt::futex;
pub use kore_rt::process;
pub use kore_rt::{Bundle, ExitCode, Termination, exit_codes, main, spmd, task};
pub mod collections;
pub mod random;
pub use kore_rt::executor as task;
pub use kore_rt::{thread, thread_controller};

/// Native driver substrate: capability-backed registers, bounded DMA and rings.
/// Hardware effects still require a device-specific owner and lifecycle policy.
#[cfg(feature = "driver")]
pub use kore_driver as driver;

/// Native clocks and deadlines; wall time never coerces into monotonic time.
pub mod time {
    pub use core::time::Duration;
    pub use kore_abi::clock::{TimerError, TimerResult, UnixTime, Uptime};
    pub use kore_rt::clock::{calibration, uptime, uptime_from, wall, wall_from};
    /// Raw relative nanoseconds, retained for compatibility.
    pub use kore_rt::sleep_for;
    pub use kore_rt::sleep_until as sleep_until_ns;

    /// Suspend on the native executor for a checked duration.
    pub async fn sleep(duration: Duration) -> TimerResult<()> {
        let ns = u64::try_from(duration.as_nanos()).map_err(|_| TimerError::OutOfRange)?;
        kore_rt::sleep_for(ns).await;
        Ok(())
    }

    /// Suspend until an absolute monotonic instant; wall time is not accepted.
    pub async fn sleep_until(deadline: Uptime) {
        kore_rt::sleep_until(deadline.as_nanos()).await;
    }
}

#[cfg(feature = "fs")]
pub mod fs;

/// Common application types. Raw slots and kernel verbs stay explicitly named.
pub mod prelude {
    pub use crate::collections::{HashMap, HashSet, HashTable};
    #[cfg(feature = "fs")]
    pub use crate::fs::{Fs, View};
    pub use core::time::Duration;
    pub use kore_abi::clock::{UnixTime, Uptime};
    pub use kore_app::prelude::*;
    pub use kore_rt::{ExitCode, entry::Rt};
}

#[cfg(test)]
mod time_tests {
    use core::{
        future::Future,
        pin::pin,
        task::{Context, Poll, Waker},
        time::Duration,
    };
    #[test]
    fn oversized_sleep_refuses_before_requiring_an_executor() {
        let mut future = pin!(crate::time::sleep(Duration::MAX));
        assert_eq!(
            future
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop())),
            Poll::Ready(Err(crate::time::TimerError::OutOfRange))
        );
    }
}
