//! The executor-free future driver — one `block_on`, five former copies.
//!
//! This is **not** the [`Executor`](crate::executor::Executor).  It holds no
//! task slab, stands no doorbell post, and keeps no wake registry: it pins one
//! future and polls it on a noop waker until it is `Ready`.  That makes it the
//! blocking face of an async client, which is the only reason it exists —
//! `res/USERLAND.md` §6's std trajectory needs `std::sys`'s synchronous
//! surface to be a *consumer* of the async protocol code rather than a second
//! implementation of it, and the same need already produced four hand-rolled
//! copies of this loop (the startup linker's, exec's "imported wholesale" twin,
//! and two host-test drivers) plus a fifth in `kore-svc-dagfs`.
//!
//! # What may be driven here
//!
//! A future polled on a noop waker makes progress only if nothing has to
//! *wake* it.  Three shapes qualify, and they are the shapes the runtime
//! actually produces:
//!
//! 1. **A direct-reply-wait client.**  A [`SvcClient`](crate::svc::SvcClient)
//!    built with
//!    [`with_direct_reply_wait`](crate::svc::SvcClient::with_direct_reply_wait)
//!    exclusively owns its doorbell and *fuses the parking commit into the
//!    poll*: the kernel parks the caller inside `poll`, and the poll that
//!    returns is the one that already has the reply.  Such a future is `Ready`
//!    on its first poll and never spins.  This is the shape `std::sys::kore`
//!    will use.
//! 2. **Level-triggered awaits over a concurrently-progressing service.**  The
//!    startup linker and exec's founder both poll in a spin while the service
//!    on the other side of the pair makes progress; every await re-reads state
//!    rather than waiting for an edge, so a re-poll observes the progress the
//!    moment it happens.
//! 3. **Futures that never yield.**  The filesystem engine's RAM provider is
//!    async only because the *seam* must admit a parking provider; over RAM
//!    every future is `Ready` on first poll.  Host tests drive them here.
//!
//! A future that yields expecting the executor's wake registry — anything
//! awaiting [`sleep_until`](crate::executor::sleep_until), a channel recv, or
//! an executor-owned reply wait — will **never** progress under this driver.
//! That is a caller error, not a condition to recover from: use
//! [`Executor::run`](crate::executor::Executor::run) for those.
//!
//! # Why the unbounded form spins rather than parks
//!
//! Parking requires a doorbell to park *on*, and a doorbell is the executor's
//! standing post or a direct client's own — this driver owns neither.  Shape 1
//! never reaches the spin (its park is inside the poll) and shape 3 cannot; the
//! spin exists for shape 2, where it is deliberate: a service that never
//! replies spins forever, which is the same disposition as a parked trap, and
//! the boot gate's timeout is the watchdog.  Callers that must not hang — host
//! tests, where nothing else can make progress — use [`try_block_on`], whose
//! exhaustion names the real bug rather than wedging the run.

use core::{
    future::Future,
    hint,
    pin::{Pin, pin},
    task::{Context, Poll, Waker},
};

/// Drive `future` to completion, polling on a noop waker.
///
/// Pins `future` in this frame and delegates to [`block_on_pinned`]; callers
/// with a large future that is already pinned in their own frame use that
/// directly so it is not materialized in two nested frames (the
/// [`Executor::run`](crate::executor::Executor::run) /
/// [`run_pinned`](crate::executor::Executor::run_pinned) split, mirrored).
///
/// See the module docs for which futures may be driven here.  A future that
/// awaits an executor wake never completes under this driver; that is a
/// caller error, and [`try_block_on`] is the form that reports it.
pub fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    block_on_pinned(future.as_mut())
}

/// [`block_on`] over an already-pinned future.
pub fn block_on_pinned<F: Future + ?Sized>(mut future: Pin<&mut F>) -> F::Output {
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(output) = future.as_mut().poll(&mut cx) {
            return output;
        }
        hint::spin_loop();
    }
}

/// [`block_on`], bounded: give up after `max_polls` and return `None`.
///
/// For callers that must fail rather than wedge — host tests, where no other
/// party can make the future progress, so exhaustion is proof that the future
/// needs a real executor.  `max_polls` is a poll count, not a duration: a
/// future that is `Ready` on first poll never consumes more than one.
pub fn try_block_on<F: Future>(future: F, max_polls: usize) -> Option<F::Output> {
    let mut future = pin!(future);
    let mut cx = Context::from_waker(Waker::noop());
    for _ in 0..max_polls {
        if let Poll::Ready(output) = future.as_mut().poll(&mut cx) {
            return Some(output);
        }
        hint::spin_loop();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::Cell;

    #[test]
    fn returns_a_ready_value() {
        assert_eq!(block_on(async { 7u32 }), 7);
    }

    #[test]
    fn drives_a_chain_of_awaits() {
        async fn inner(x: u32) -> u32 {
            x + 1
        }
        async fn outer() -> u32 {
            inner(inner(1).await).await
        }
        assert_eq!(block_on(outer()), 3);
    }

    /// Shape 2: a future that yields a bounded number of times — the
    /// level-triggered client re-polled until the peer has made progress —
    /// completes without any wake.
    #[test]
    fn drives_a_future_that_yields_before_completing() {
        let polls = Cell::new(0usize);
        let fut = core::future::poll_fn(|cx| {
            let seen = polls.get() + 1;
            polls.set(seen);
            if seen < 4 {
                // A level-triggered source re-polls on its own; a real wake is
                // what this driver deliberately does not require.
                cx.waker().wake_by_ref();
                Poll::Pending
            } else {
                Poll::Ready(seen)
            }
        });
        assert_eq!(block_on(fut), 4);
        assert_eq!(polls.get(), 4);
    }

    #[test]
    fn try_block_on_returns_the_value_within_budget() {
        assert_eq!(try_block_on(async { 9u8 }, 4), Some(9));
    }

    /// The caller error the bounded form exists to name: a future that only an
    /// executor could wake exhausts the budget instead of wedging the run.
    #[test]
    fn try_block_on_gives_up_on_a_future_that_needs_a_wake() {
        let fut = core::future::pending::<()>();
        assert_eq!(try_block_on(fut, 8), None);
    }

    #[test]
    fn try_block_on_polls_a_ready_future_exactly_once() {
        let polls = Cell::new(0usize);
        let fut = core::future::poll_fn(|_| {
            polls.set(polls.get() + 1);
            Poll::Ready(())
        });
        assert_eq!(try_block_on(fut, 64), Some(()));
        assert_eq!(polls.get(), 1);
    }
}
