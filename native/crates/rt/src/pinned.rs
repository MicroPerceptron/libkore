//! **`Pinned<T>`** — a thread-bound value carried through an API that
//! demands `Send + Sync`.
//!
//! Much of kore's userland authority is thread-bound by construction: a
//! capability slot index names a cell in the *current thread's* arena, so
//! [`OwnedSlot`](crate::cap::OwnedSlot) and everything holding one is
//! `!Send`. Foreign runtimes ask for the opposite — Wasmtime's stack
//! creator and its WASI stream traits are `Send + Sync + 'static`, because a
//! host `Store` *may* move between threads. On kore a store never does, and
//! claiming `Send` for the slot-holding value would be a lie the type system
//! could no longer catch.
//!
//! `Pinned` states what is actually true: the value is owned by the thread
//! that pinned it, and every access checks that the caller *is* that
//! thread. From anywhere else [`get`](Pinned::get) and
//! [`get_mut`](Pinned::get_mut) answer `None` — a typed refusal, never a
//! reference — and dropping it off its owner **leaks** the value instead of
//! running its destructor on a thread whose arena its slots do not name.
//! Leaking is the fail-closed direction: a leaked slot stays claimed; a
//! slot released from the wrong arena would free someone else's.
//!
//! A pinned **future** is a `Send` future: polled on its owner it polls the
//! inner future; polled anywhere else it panics, because a kore future's
//! wakes and parks name its own thread's posts, and there is no honest
//! `Pending` to give a foreign executor. On kore that poll cannot happen — a
//! store and its executor never migrate — so the panic marks a bug, not a
//! state.
//!
//! The owner is the thread's capability-arena identity — process-unique and
//! never reused ([`thread_owner_identity`](crate::entry::geometry)), the
//! same identity [`thread::join`](crate::thread::join) checks.

use core::fmt;
use core::mem::ManuallyDrop;

/// A value usable only on the thread that pinned it. See the
/// [module docs](self).
pub struct Pinned<T> {
    owner: usize,
    value: ManuallyDrop<T>,
}

// SAFETY: the value is reachable only through `get`/`get_mut`/`into_inner`,
// each of which returns it only on the owning thread, and `Drop` runs `T`'s
// destructor only there. Moving or sharing the wrapper therefore never lets
// another thread touch `T`: a foreign holder can drop the wrapper (leaking
// `T`) and do nothing else.
unsafe impl<T> Send for Pinned<T> {}
// SAFETY: as above — `&Pinned<T>` on a foreign thread yields no `&T`.
unsafe impl<T> Sync for Pinned<T> {}

impl<T> Pinned<T> {
    /// Pin `value` to the calling thread; the value back where the thread
    /// has no identity to pin to (no runtime block — a pre-runtime context).
    pub fn new(value: T) -> Result<Self, T> {
        match current_owner() {
            Some(owner) => Ok(Self {
                owner,
                value: ManuallyDrop::new(value),
            }),
            None => Err(value),
        }
    }

    /// Whether the calling thread owns the value.
    pub fn is_owner(&self) -> bool {
        current_owner() == Some(self.owner)
    }

    /// The value, on its owning thread; `None` elsewhere.
    pub fn get(&self) -> Option<&T> {
        self.is_owner().then(|| &*self.value)
    }

    /// The value, mutably, on its owning thread; `None` elsewhere.
    pub fn get_mut(&mut self) -> Option<&mut T> {
        if self.is_owner() {
            Some(&mut *self.value)
        } else {
            None
        }
    }

    /// Unpin: the value back, on its owning thread; the wrapper, unchanged,
    /// anywhere else.
    pub fn into_inner(self) -> Result<T, Self> {
        if !self.is_owner() {
            return Err(self);
        }
        let mut this = ManuallyDrop::new(self);
        // SAFETY: `this` is never dropped, so the value is taken exactly
        // once, and on its owning thread.
        Ok(unsafe { ManuallyDrop::take(&mut this.value) })
    }
}

impl<F: Future> Future for Pinned<F> {
    type Output = F::Output;

    fn poll(
        self: core::pin::Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<F::Output> {
        assert!(
            self.is_owner(),
            "a pinned kore future was polled off its owning thread"
        );
        // SAFETY: structural pinning of `value`: `Pinned` never moves it out
        // while pinned (`into_inner` takes `self` by value, which a pinned
        // wrapper cannot give up), and `Drop` drops it in place.
        let inner = unsafe { self.map_unchecked_mut(|this| &mut *this.value) };
        inner.poll(cx)
    }
}

impl<T> Drop for Pinned<T> {
    fn drop(&mut self) {
        if self.is_owner() {
            // SAFETY: dropped once, here, on the owning thread.
            unsafe { ManuallyDrop::drop(&mut self.value) }
        }
        // Off the owner the value is leaked (see the module docs).
    }
}

impl<T> fmt::Debug for Pinned<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pinned")
            .field("owner", &self.owner)
            .field("here", &self.is_owner())
            .finish_non_exhaustive()
    }
}

/// The calling thread's owner identity.
#[cfg(target_os = "none")]
fn current_owner() -> Option<usize> {
    crate::entry::geometry::thread_owner_identity()
}

/// The host stand-in: the address of a per-thread cell — unique among live
/// threads, which is what the host suite exercises. (A metal identity is
/// never reused; a host address may be once its thread has exited.)
#[cfg(all(not(target_os = "none"), feature = "std"))]
fn current_owner() -> Option<usize> {
    std::thread_local! {
        static CELL: u8 = const { 0 };
    }
    Some(CELL.with(|cell| cell as *const u8 as usize))
}

/// No threads to tell apart without `std` off-target.
#[cfg(all(not(target_os = "none"), not(feature = "std")))]
fn current_owner() -> Option<usize> {
    None
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::Pinned;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct Counted(Arc<AtomicUsize>);
    impl Drop for Counted {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn owner_sees_the_value_and_drops_it() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut pinned = Pinned::new(Counted(drops.clone())).ok().unwrap();
        assert!(pinned.get().is_some());
        assert!(pinned.get_mut().is_some());
        drop(pinned);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_foreign_thread_gets_nothing_and_leaks_on_drop() {
        let drops = Arc::new(AtomicUsize::new(0));
        let pinned = Pinned::new(Counted(drops.clone())).ok().unwrap();
        let back = std::thread::spawn(move || {
            assert!(!pinned.is_owner());
            assert!(pinned.get().is_none());
            let pinned = pinned.into_inner().err().expect("refused off the owner");
            drop(pinned);
        });
        back.join().unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 0, "leaked, never dropped off-owner");
    }

    #[test]
    fn a_pinned_future_polls_on_its_owner() {
        let future = Pinned::new(async { 5u8 }).ok().unwrap();
        assert_eq!(crate::block_on(future), 5);
    }

    #[test]
    fn into_inner_on_the_owner_returns_the_value() {
        let pinned = Pinned::new(7u32).ok().unwrap();
        assert_eq!(pinned.into_inner().ok(), Some(7));
    }
}
