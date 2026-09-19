//! `wasi:clocks` — kore's monotonic and wall clocks, and pollable deadlines.

use alloc::boxed::Box;

use wasmtime::component::Resource;
use wasmtime_wasi_io::poll::{DynPollable, Pollable, subscribe};

use crate::WasiCtxView;
use crate::bindings::wasi::clocks::{monotonic_clock, wall_clock};

/// Both clocks count nanoseconds (`kore::time`), which is the finest
/// resolution WASI can express; the calibrated tick is finer still.
const RESOLUTION_NS: u64 = 1;

fn uptime_ns() -> wasmtime::Result<u64> {
    kore::time::uptime()
        .map(|t| t.as_nanos())
        .map_err(|e| wasmtime::format_err!("monotonic clock unavailable: {e:?}"))
}

impl monotonic_clock::Host for WasiCtxView<'_> {
    fn now(&mut self) -> wasmtime::Result<monotonic_clock::Instant> {
        uptime_ns()
    }

    fn resolution(&mut self) -> wasmtime::Result<monotonic_clock::Duration> {
        Ok(RESOLUTION_NS)
    }

    fn subscribe_instant(
        &mut self,
        when: monotonic_clock::Instant,
    ) -> wasmtime::Result<Resource<DynPollable>> {
        let deadline = if when > uptime_ns()? {
            Deadline::At(when)
        } else {
            Deadline::Past { yielded: false }
        };
        let deadline = self.table.push(deadline)?;
        subscribe(self.table, deadline)
    }

    fn subscribe_duration(
        &mut self,
        duration: monotonic_clock::Duration,
    ) -> wasmtime::Result<Resource<DynPollable>> {
        let deadline = if duration == 0 {
            Deadline::Past { yielded: false }
        } else {
            match uptime_ns()?.checked_add(duration) {
                Some(at) => Deadline::At(at),
                // Past the clock's range: wait forever rather than trap, as
                // upstream does.
                None => Deadline::Never,
            }
        };
        let deadline = self.table.push(deadline)?;
        subscribe(self.table, deadline)
    }
}

impl wall_clock::Host for WasiCtxView<'_> {
    fn now(&mut self) -> wasmtime::Result<wall_clock::Datetime> {
        let ns = kore::time::wall()
            .map(|t| t.as_nanos())
            .map_err(|e| wasmtime::format_err!("wall clock unavailable: {e:?}"))?;
        Ok(wall_clock::Datetime {
            seconds: ns / 1_000_000_000,
            nanoseconds: (ns % 1_000_000_000) as u32,
        })
    }

    fn resolution(&mut self) -> wasmtime::Result<wall_clock::Datetime> {
        Ok(wall_clock::Datetime {
            seconds: 0,
            nanoseconds: RESOLUTION_NS as u32,
        })
    }
}

/// A monotonic deadline as a pollable, sleeping on the kore executor.
enum Deadline {
    /// Already due. Ready after one yield, so a guest polling a zero timeout
    /// in a loop still lets the executor run its other futures — the
    /// starvation upstream's `Deadline::Past` guards against.
    Past {
        yielded: bool,
    },
    At(u64),
    Never,
}

#[wasmtime_wasi_io::async_trait]
impl Pollable for Deadline {
    async fn ready(&mut self) {
        match self {
            Deadline::Past { yielded: true } => {}
            Deadline::Past { yielded } => {
                *yielded = true;
                let mut once = false;
                core::future::poll_fn(|cx| {
                    if once {
                        core::task::Poll::Ready(())
                    } else {
                        once = true;
                        cx.waker().wake_by_ref();
                        core::task::Poll::Pending
                    }
                })
                .await;
            }
            // The Kore sleep names this thread's executor; the pollable must
            // be `Send`, so the sleep is pinned to the thread (a store never
            // leaves it).
            Deadline::At(at) => {
                let sleep = match kore::runtime::Pinned::new(kore::time::sleep_until_ns(*at)) {
                    Ok(sleep) => sleep,
                    Err(_) => return,
                };
                sleep.await
            }
            Deadline::Never => core::future::pending::<()>().await,
        }
    }
}
