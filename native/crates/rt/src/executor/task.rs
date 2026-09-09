//! `task` — type-erased task records and the cookie waker
//! (`res/USERLAND.md` §3.3).
//!
//! The executor's fixed slab holds **task records** — a state word, a
//! ready bit, and erased poll/drop fn pointers — never futures by value:
//! an `async fn`'s future is unnameable, so each spawned task's future
//! lives in a per-task static [`TaskStorage<F>`] (the embassy pattern),
//! and the record ([`TaskHeader`], the storage's first field) is what the
//! slab points at.  `#[kore_rt::task]` (Phase S) will emit the statics;
//! this is the non-macro spine it wraps: `TaskStorage::new().spawn(fut)`
//! yields a [`SpawnToken`] the executor admits.
//!
//! The waker is a **source-cookie over a raw vtable, no alloc, no `dyn`**:
//! its data pointer *is* the task record (`clone` copies the pointer,
//! `wake` sets the ready bit), and [`header_of`] recovers the record from
//! any waker built here — how [`sleep_until`](super::sleep_until) reaches
//! its executor's timer table without a handle argument.

use core::cell::UnsafeCell;
use core::future::Future;
use core::mem::MaybeUninit;
use core::pin::Pin;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU8, Ordering};
use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

use super::ExecCtx;

/// Storage slot states ([`TaskHeader::state`]).
const VACANT: u8 = 0;
/// A future is written and a [`SpawnToken`] is outstanding / admitted.
const ARMED: u8 = 1;

/// The type-erased **task record** (§3.3): state word, ready bit, erased
/// poll/drop, and the polling executor's context pointer (set before
/// every poll — how context-free futures like `sleep_until` find their
/// timer table).  Always the **first field** of a [`TaskStorage`]
/// (`repr(C)`), so an erased header pointer recovers the typed storage.
pub(crate) struct TaskHeader {
    state: AtomicU8,
    ready: AtomicBool,
    /// `unsafe fn(NonNull<TaskHeader>) -> bool` (`true` = completed).
    poll_fn: AtomicPtr<()>,
    /// `unsafe fn(NonNull<TaskHeader>)` — drop the stored future.
    drop_fn: AtomicPtr<()>,
    pub(crate) exec: AtomicPtr<ExecCtx>,
}

impl TaskHeader {
    pub(crate) const fn new() -> Self {
        Self {
            state: AtomicU8::new(VACANT),
            ready: AtomicBool::new(false),
            poll_fn: AtomicPtr::new(core::ptr::null_mut()),
            drop_fn: AtomicPtr::new(core::ptr::null_mut()),
            exec: AtomicPtr::new(core::ptr::null_mut()),
        }
    }

    /// Mark runnable (the waker's whole job — §3.3: wake = set the bit;
    /// the drain makes duplicates harmless).
    #[inline]
    pub(crate) fn mark_ready(&self) {
        self.ready.store(true, Ordering::Relaxed);
    }

    /// Consume the ready bit — the executor's per-pass claim.
    #[inline]
    pub(crate) fn take_ready(&self) -> bool {
        self.ready.swap(false, Ordering::Relaxed)
    }
}

/// Per-task future storage — the embassy pattern (`res/USERLAND.md`
/// §3.3): a `'static` cell the task's unnameable future lives in, so the
/// executor's slab holds only erased records.  Spawn capacity is exactly
/// the statically declared task set — design-bounded, refusal on
/// overflow ([`SpawnRefused`]).
///
/// ```ignore
/// let storage: &'static TaskStorage<_> = /* a static (Phase S's macro) */;
/// let token = storage.spawn(async move { /* … */ })?;
/// executor.spawn(token)?;
/// ```
#[repr(C)]
pub struct TaskStorage<F: Future> {
    header: TaskHeader,
    fut: UnsafeCell<MaybeUninit<F>>,
}

// SAFETY: a `TaskStorage` static is shared only between the spawner and
// its one executor (single-threaded per task, §3.3); the state word
// serializes the future cell's writer, and the future itself is only
// ever touched by the executor's thread.
unsafe impl<F: Future> Sync for TaskStorage<F> {}

/// The storage already holds a live task — the §3.3/§8 capacity refusal
/// (spawn capacity is the statically declared task set).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpawnRefused;

/// A spawned-but-unadmitted task — hand it to
/// [`Executor::spawn`](super::Executor::spawn).  Dropping it un-spawns:
/// the stored future is dropped and the storage reopens.
#[must_use = "pass the token to Executor::spawn — dropping it un-spawns the task"]
pub struct SpawnToken {
    hdr: NonNull<TaskHeader>,
}

impl SpawnToken {
    pub(crate) fn header(&self) -> NonNull<TaskHeader> {
        self.hdr
    }
}

impl Drop for SpawnToken {
    fn drop(&mut self) {
        // SAFETY: the header came from a live `TaskStorage` whose future
        // was written by `spawn` and not yet admitted (the token is the
        // admission); dropping it un-spawns exactly once.
        unsafe {
            let h = self.hdr.as_ref();
            let drop_fn: unsafe fn(NonNull<TaskHeader>) =
                core::mem::transmute(h.drop_fn.load(Ordering::Acquire));
            drop_fn(self.hdr);
            h.state.store(VACANT, Ordering::Release);
        }
    }
}

impl<F: Future<Output = ()>> TaskStorage<F> {
    /// An empty storage cell (a `static`'s initializer).
    pub const fn new() -> Self {
        Self {
            header: TaskHeader::new(),
            fut: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    /// Move `fut` into the storage and hand back the admission token.
    /// Refuses when a live task already occupies the cell (§8: overflow
    /// refuses — capacity is the declared task set).
    pub fn spawn(&'static self, fut: F) -> Result<SpawnToken, SpawnRefused> {
        self.header
            .state
            .compare_exchange(VACANT, ARMED, Ordering::AcqRel, Ordering::Relaxed)
            .map_err(|_| SpawnRefused)?;
        // SAFETY: the ARMED claim above makes this thread the cell's
        // sole writer until completion/un-spawn reopens it.
        unsafe { (*self.fut.get()).write(fut) };
        // Preserve function-pointer provenance through publication and load.
        // Integer bits transmuted back into a callable pointer do not carry it.
        // The ARMED owner and release/acquire handoff are unchanged.
        self.header
            .poll_fn
            .store(poll_erased::<F> as *mut (), Ordering::Release);
        self.header
            .drop_fn
            .store(drop_erased::<F> as *mut (), Ordering::Release);
        Ok(SpawnToken {
            // Erasure must retain provenance for the entire storage, including
            // the future's UnsafeCell. A pointer borrowed from just `header`
            // cannot later recover access to its sibling field. repr(C) puts
            // the header at offset zero without narrowing the original borrow.
            hdr: NonNull::from(self).cast(),
        })
    }
}

impl<F: Future<Output = ()>> Default for TaskStorage<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Poll the future stored behind `hdr`; `true` when it completed (the
/// future is dropped in place and the storage reopens).
///
/// # Safety
/// `hdr` must head a live `TaskStorage<F>` in the ARMED state, on the
/// executor's own thread.
unsafe fn poll_erased<F: Future<Output = ()>>(hdr: NonNull<TaskHeader>) -> bool {
    // SAFETY: `TaskHeader` is the first field of the `repr(C)` storage.
    let storage = unsafe { &*(hdr.as_ptr() as *const TaskStorage<F>) };
    // SAFETY: ARMED ⇒ the future is initialized; the storage never moves
    // (it is a static), so pinning the interior reference is sound.
    let fut = unsafe { Pin::new_unchecked((*storage.fut.get()).assume_init_mut()) };
    let waker = waker_for(hdr);
    let mut cx = Context::from_waker(&waker);
    match fut.poll(&mut cx) {
        Poll::Ready(()) => {
            // SAFETY: completed — drop in place, reopen the cell.
            unsafe { (*storage.fut.get()).assume_init_drop() };
            storage.header.state.store(VACANT, Ordering::Release);
            true
        }
        Poll::Pending => false,
    }
}

/// Drop the future stored behind `hdr` (the un-spawn leg).
///
/// # Safety
/// As [`poll_erased`], and the future must not be mid-poll.
unsafe fn drop_erased<F: Future<Output = ()>>(hdr: NonNull<TaskHeader>) {
    let storage = unsafe { &*(hdr.as_ptr() as *const TaskStorage<F>) };
    // SAFETY: ARMED ⇒ initialized; sole access (see caller contracts).
    unsafe { (*storage.fut.get()).assume_init_drop() };
}

/// Poll the task behind an erased header — the executor's drain step.
///
/// # Safety
/// `hdr` must be an admitted, ARMED task record on this thread.
pub(crate) unsafe fn poll_task(hdr: NonNull<TaskHeader>) -> bool {
    // SAFETY: caller contract; the fn pointer was stored by `spawn`.
    unsafe {
        let poll_fn: unsafe fn(NonNull<TaskHeader>) -> bool =
            core::mem::transmute(hdr.as_ref().poll_fn.load(Ordering::Acquire));
        poll_fn(hdr)
    }
}

/// The one waker vtable — a `static` so vtable identity ([`header_of`]'s
/// check) has exactly one address.
static VTABLE: RawWakerVTable = RawWakerVTable::new(
    |p| RawWaker::new(p, &VTABLE),
    |p| unsafe { wake_raw(p) },
    |p| unsafe { wake_raw(p) },
    |_| {},
);

/// # Safety
/// `p` came from [`waker_for`] — a live `TaskHeader`.
unsafe fn wake_raw(p: *const ()) {
    // SAFETY: caller contract.
    unsafe { (*(p as *const TaskHeader)).mark_ready() };
}

/// Build the cookie waker for a task record.  No alloc, no `dyn`: the
/// waker *is* the record pointer (§3.3).
///
/// The returned waker (and every clone) must not outlive the header —
/// upheld structurally: spawned headers live in `'static` storage, and
/// the root task's stack header confines its waker to `run`'s frame
/// (nothing hands the root waker to another task, and the executor
/// clears its tables on exit).
pub(crate) fn waker_for(hdr: NonNull<TaskHeader>) -> Waker {
    // SAFETY: the vtable's ops are exactly the contract above.
    unsafe { Waker::from_raw(RawWaker::new(hdr.as_ptr() as *const (), &VTABLE)) }
}

/// Recover the task record behind a waker **built by this executor** —
/// `None` for any foreign waker (vtable identity is the gate).  The
/// introspection seam context-free futures ([`sleep_until`](super::sleep_until))
/// use to find their executor.
pub(crate) fn header_of(waker: &Waker) -> Option<NonNull<TaskHeader>> {
    if !core::ptr::eq(waker.vtable(), &VTABLE) {
        return None;
    }
    NonNull::new(waker.data() as *mut TaskHeader)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::AtomicUsize;

    #[test]
    fn erased_poll_and_drop_preserve_provenance_and_reopen_storage() {
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        struct Probe {
            polled: bool,
        }
        impl Future for Probe {
            type Output = ();
            fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
                if self.polled {
                    Poll::Ready(())
                } else {
                    self.polled = true;
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
            }
        }
        impl Drop for Probe {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::Relaxed);
            }
        }
        static STORAGE: TaskStorage<Probe> = TaskStorage::new();

        drop(STORAGE.spawn(Probe { polled: false }).unwrap());
        assert_eq!(DROPS.load(Ordering::Relaxed), 1);
        let token = STORAGE.spawn(Probe { polled: false }).unwrap();
        let hdr = token.header();
        // Adopt the token as Executor::spawn does; this test is the sole poller.
        core::mem::forget(token);
        // SAFETY: the token transferred its initialized ARMED storage to us;
        // Pending keeps that ownership, and Ready retires it exactly once.
        unsafe {
            assert!(!poll_task(hdr));
            assert!(hdr.as_ref().take_ready());
            assert_eq!(DROPS.load(Ordering::Relaxed), 1);
            assert!(poll_task(hdr));
        }
        assert_eq!(DROPS.load(Ordering::Relaxed), 2);
        drop(STORAGE.spawn(Probe { polled: false }).unwrap());
        assert_eq!(DROPS.load(Ordering::Relaxed), 3);
    }
}
