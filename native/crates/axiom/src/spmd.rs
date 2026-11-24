//! Borrowed Kore execution for independent Axiom work items.
//!
//! The final runtime founds and retires the region; this adapter neither
//! creates a pool nor chooses graph placement. Bodies run pure work or write
//! disjoint CPU-mapped ranges. They must not share an `Accelerator` queue.
//! The same entry serves many batches, with no allocation per submission.
//! `res/AXIOM_INTEGRATION.md` distinguishes this direct, client-owned tenancy
//! from `accel-spmd`: both must consume the same `ax-exec` partition rather
//! than compile separate graph semantics.

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use kore_rt::spmd::{ResidentRegion, ResidentTicket, ResidentTicketError, SpmdCtx};

/// One bounded, one-use set of independent work items. Dynamic claiming evens
/// out variable-cost tiles without changing their geometry or kernel policy.
pub struct Batch<'a> {
    items: u32,
    body: &'a (dyn Fn(u32, u32) -> bool + Sync),
    next: AtomicU32,
    finished: AtomicU32,
    failed: AtomicU32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Report {
    pub items: u32,
    pub finished: u32,
    pub failed: u32,
}

impl Report {
    pub fn succeeded(self) -> bool {
        self.finished == self.items && self.failed == 0
    }
}

impl<'a> Batch<'a> {
    /// `body(worker, item)` receives a dense worker index and an item claimed
    /// exactly once. `false` records a failure; the remaining items still drain.
    pub fn new(items: u32, body: &'a (dyn Fn(u32, u32) -> bool + Sync)) -> Self {
        Self {
            items,
            body,
            next: AtomicU32::new(0),
            finished: AtomicU32::new(0),
            failed: AtomicU32::new(0),
        }
    }

    /// Target-neutral work loop, also used by scoped host-thread tests. It is
    /// not a pretend Kore launch: no tasks, tickets or completion are minted.
    pub fn run_worker(&self, worker: u32) {
        while let Ok(item) = self
            .next
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                (n < self.items).then(|| n + 1)
            })
        {
            if !(self.body)(worker, item) {
                self.failed.fetch_add(1, Ordering::Relaxed);
            }
            self.finished.fetch_add(1, Ordering::Release);
        }
    }

    /// Telemetry, not a replacement for the region's completion fence or a
    /// display presentation acknowledgment.
    pub fn report(&self) -> Report {
        Report {
            items: self.items,
            finished: self.finished.load(Ordering::Acquire),
            failed: self.failed.load(Ordering::Relaxed),
        }
    }
}

/// Entry supplied to the final runtime's `found_resident` call. Submit only
/// through this module's typed descriptor contract, not an unrelated `env`.
pub fn entry(ctx: &'static SpmdCtx, worker: u32) {
    kore_rt::spmd::resident_run(ctx, worker, |ctx, worker| {
        // SAFETY: submit's contract binds this entry to a stable live Batch;
        // the resident loop acquires job publication before reading env.
        let batch = unsafe { &*ctx.env().cast::<Batch<'_>>() };
        batch.run_worker(worker);
    });
}

/// One borrowed job in flight. Explicit async completion is the normal path;
/// dropping it drains synchronously before borrowed memory can be reclaimed.
#[must_use = "observe completion before reusing work or presenting its output"]
pub struct Pending<'r, 'b, const N: usize> {
    region: &'r mut ResidentRegion<N>,
    batch: &'b Batch<'b>,
    ticket: ResidentTicket,
}

/// Publish without blocking or allocating. The caller owns region admission,
/// scratch storage, the descriptor, and its transitive borrows.
///
/// # Safety
/// `region` must have been founded with `entry`. `batch` must be new and must
/// remain at its address, with its body and all borrowed memory live, until
/// the returned job is completed or dropped. Do not forget the returned guard
/// while borrowing non-static storage. No body may panic/exit its worker or
/// cross a worker barrier. CPU writes from different items must be disjoint.
pub unsafe fn submit<'r, 'b, const N: usize>(
    region: &'r mut ResidentRegion<N>,
    batch: &'b Batch<'b>,
) -> Option<Pending<'r, 'b, N>> {
    if batch.items == 0 || batch.next.load(Ordering::Relaxed) != 0 {
        return None;
    }
    let active = (region.width() as u32).min(batch.items);
    // SAFETY: the descriptor and its entry are paired by the caller above.
    let ticket = unsafe {
        region.submit(
            active,
            batch.items,
            (batch as *const Batch<'_>).cast_mut().cast(),
        )
    }?;
    Some(Pending {
        region,
        batch,
        ticket,
    })
}

impl<const N: usize> Pending<'_, '_, N> {
    pub fn poll(&self) -> Result<Option<Report>, ResidentTicketError> {
        self.region
            .poll(self.ticket)
            .map(|done| done.then(|| self.batch.report()))
    }

    /// Cooperatively await this ticket; no provider queue or host yield.
    pub async fn wait_async(&self) -> Result<Report, ResidentTicketError> {
        self.region.wait_async(self.ticket).await?;
        Ok(self.batch.report())
    }
}

impl<const N: usize> Drop for Pending<'_, '_, N> {
    fn drop(&mut self) {
        // The exclusive region borrow prevents a different epoch or submit.
        // This is a drain-only cancellation backstop, not a hidden submit wait.
        self.region
            .wait(self.ticket)
            .expect("pending owns this region's issued ticket");
    }
}

/// A worker's exclusive prepared state — the arena, program fork or stage
/// a [`Batch`] body reaches through its dense worker index.  The checked
/// claim refuses a duplicate index instead of creating two mutable aliases,
/// even in release builds; nothing spins, waits or allocates on the claim.
pub struct WorkerSlot<T> {
    claimed: AtomicBool,
    value: UnsafeCell<T>,
}

// SAFETY: `with` exclusively claims before lending a mutable reference and
// releases after the callback. `T` may cross tasks only when it is `Send`,
// and no reference can escape the callback's borrow.
unsafe impl<T: Send> Sync for WorkerSlot<T> {}

impl<T> WorkerSlot<T> {
    pub const fn new(value: T) -> Self {
        Self {
            claimed: AtomicBool::new(false),
            value: UnsafeCell::new(value),
        }
    }

    /// Lend the value to `work`, or `None` when another claim holds it.
    pub fn with<R>(&self, work: impl FnOnce(&mut T) -> R) -> Option<R> {
        self.claimed
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .ok()?;
        struct Release<'a>(&'a AtomicBool);
        impl Drop for Release<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Release);
            }
        }
        let _release = Release(&self.claimed);
        // SAFETY: this claim is the only path to the owned value.
        Some(work(unsafe { &mut *self.value.get() }))
    }

    pub fn into_inner(self) -> T {
        self.value.into_inner()
    }
}

/// The compute lane a Kore renderer runs its batches on: a resident region
/// founded once over `ctx` with this module's [`entry`], or — when the
/// machine fits one worker, founding is refused, or the gang cannot park —
/// the owner lane alone, which then drains every batch itself (`res/SPMD.md`
/// R9/R10).  One job in flight; every wait is cooperative.
pub struct Lane<const N: usize> {
    region: Option<ResidentRegion<N>>,
}

impl<const N: usize> Lane<N> {
    /// Found `width` workers, or settle for the owner lane.  A gang that
    /// founded but cannot park (starless) is joined here rather than kept
    /// hot behind an idle renderer.
    pub async fn found(ctx: &'static SpmdCtx, width: usize) -> Self {
        let region = (width > 1)
            .then(|| kore_rt::spmd::found_resident::<N>(ctx, entry, width))
            .flatten();
        let region = match region {
            Some(region) if region.star_active() => Some(region),
            Some(region) => {
                retire_region(region).await;
                None
            }
            None => None,
        };
        Self { region }
    }

    /// Workers a batch body may be claimed on: the region's width, or one.
    pub fn width(&self) -> usize {
        self.region.as_ref().map_or(1, |region| region.width())
    }

    /// Run `batch` to completion — every item claimed exactly once by the
    /// region's workers, or by the owner lane alone — and report.
    ///
    /// # Safety
    /// As [`submit`]: no body may panic, exit its worker or cross a worker
    /// barrier, and CPU writes from different items must be disjoint.  The
    /// batch and everything it borrows outlive this call by construction.
    pub async unsafe fn run(&mut self, batch: &Batch<'_>) -> Result<Report, crate::Error> {
        match self.region.as_mut() {
            Some(region) => {
                // SAFETY: the region was founded with `entry`; `batch` is
                // new and pinned on this frame, which waits for completion.
                let pending = unsafe { submit(region, batch) }.ok_or(crate::Error::Schedule)?;
                pending
                    .wait_async()
                    .await
                    .map_err(|_| crate::Error::Schedule)
            }
            None => {
                batch.run_worker(0);
                Ok(batch.report())
            }
        }
    }

    /// Join the gang; the owner lane has nothing to retire.
    pub async fn retire(self) {
        if let Some(region) = self.region {
            retire_region(region).await;
        }
    }
}

async fn retire_region<const N: usize>(region: ResidentRegion<N>) {
    let mut result = region.retire();
    while let Err(error) = result {
        kore_rt::executor::next_wake_for(250_000).await;
        result = error.spmd.join_all().map(|_| ());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uneven_work_is_claimed_once_and_failures_still_drain() {
        for width in [1, 2, 3, 8] {
            let seen = [const { AtomicU32::new(0) }; 257];
            let body = |_: u32, item: u32| {
                seen[item as usize].fetch_add(1, Ordering::Relaxed);
                item != 17
            };
            let batch = Batch::new(seen.len() as u32, &body);
            std::thread::scope(|scope| {
                for worker in 0..width {
                    let batch = &batch;
                    scope.spawn(move || batch.run_worker(worker));
                }
            });
            assert!(seen.iter().all(|n| n.load(Ordering::Relaxed) == 1));
            assert_eq!(
                batch.report(),
                Report {
                    items: 257,
                    finished: 257,
                    failed: 1
                }
            );
            batch.run_worker(0);
            assert_eq!(batch.report().finished, 257);
        }
    }

    #[test]
    fn empty_work_never_calls_the_body() {
        let batch = Batch::new(0, &|_, _| panic!("empty batch"));
        batch.run_worker(0);
        assert!(batch.report().succeeded());
    }

    #[test]
    fn worker_slot_refuses_overlapping_access_and_releases_afterwards() {
        let slot = WorkerSlot::new(0_usize);
        assert_eq!(
            slot.with(|value| {
                *value = 1;
                assert!(slot.with(|_| ()).is_none());
                *value = 2;
                *value
            }),
            Some(2)
        );
        assert_eq!(slot.with(|value| *value), Some(2));
        assert_eq!(slot.into_inner(), 2);
    }

    #[test]
    fn the_owner_lane_drains_a_batch_itself() {
        static CTX: SpmdCtx = SpmdCtx::new(4);
        let seen = [const { AtomicU32::new(0) }; 9];
        let body = |worker: u32, item: u32| {
            assert_eq!(worker, 0, "the owner lane is worker 0");
            seen[item as usize].fetch_add(1, Ordering::Relaxed);
            true
        };
        let batch = Batch::new(seen.len() as u32, &body);
        let report = kore_rt::try_block_on(
            async {
                let mut lane = Lane::<4>::found(&CTX, 4).await;
                assert_eq!(lane.width(), 1, "no workers off the metal");
                let report = unsafe { lane.run(&batch) }.await.unwrap();
                lane.retire().await;
                report
            },
            1,
        )
        .unwrap();
        assert!(report.succeeded());
        assert!(seen.iter().all(|n| n.load(Ordering::Relaxed) == 1));
    }
}
