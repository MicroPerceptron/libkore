//! Foreign tasks use the native statically declared task set, with no second
//! scheduler. Rust founding admits prepared SpawnTokens to its real executor.
use crate::KoreStatus;
use core::{
    ffi::c_void,
    sync::atomic::{AtomicU64, Ordering},
};
use kore::runtime::executor::{Executor, SpawnToken};

pub const KORE_TASK_CAPACITY: usize = 4;
pub const KORE_TASK_DONE: u32 = 0;
pub const KORE_TASK_YIELD: u32 = 1;
pub const KORE_TASK_SLEEP_FOR: u32 = 2;
pub const KORE_TASK_SLEEP_UNTIL: u32 = 3;
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct KoreTaskAction {
    pub operation: u32,
    pub reserved: u32,
    pub nanoseconds: u64,
}
/// Return an action, then relinquish control to the native executor. The callback
/// receives no waker pointer or borrowed Rust future. It must return promptly.
pub type KoreTaskStep = Option<unsafe extern "C" fn(*mut c_void) -> KoreTaskAction>;
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct KoreTaskHandle {
    pub generation: u64,
    pub slot: u32,
    pub reserved: u32,
}
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KoreTaskState {
    Pending = 0,
    Complete = 1,
    Cancelled = 2,
    InvalidAction = 3,
    Stale = 4,
}
const FREE: u64 = 0;
const PENDING: u64 = 1;
const COMPLETE: u64 = 2;
const CANCELLED: u64 = 3;
const INVALID: u64 = 4;
const STATE_BITS: u32 = 3;
const STATE_MASK: u64 = 7;
const MAX_GENERATION: u64 = u64::MAX >> STATE_BITS;
// Generation and lifecycle publish as one atomic value. A stale release can
// never free a newly prepared task, even when racing reuse on another thread.
struct Control {
    word: AtomicU64,
}
impl Control {
    const fn new() -> Self {
        Self {
            word: AtomicU64::new(FREE),
        }
    }
    fn reserve(&self) -> Option<u64> {
        let old = self.word.load(Ordering::Acquire);
        let generation = old >> STATE_BITS;
        if old & STATE_MASK != FREE || generation == MAX_GENERATION {
            return None;
        }
        let next = generation + 1;
        self.word
            .compare_exchange(
                old,
                (next << STATE_BITS) | PENDING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .ok()?;
        Some(next)
    }
}
static CONTROLS: [Control; KORE_TASK_CAPACITY] = [const { Control::new() }; KORE_TASK_CAPACITY];
const _: () = assert!(KORE_TASK_CAPACITY <= kore::runtime::executor::MAX_TASKS);

/// Bootstrap-only batch of native task tokens. Prepare from C/Zig, then use
/// `take` with the existing entry API or `admit` on the founded executor.
/// Dropping an unadmitted batch drops its native tokens and cancels its jobs.
/// The foreign handles remain queryable until explicitly released.
pub struct KoreTaskBatch {
    tokens: [Option<SpawnToken>; KORE_TASK_CAPACITY],
}
impl Default for KoreTaskBatch {
    fn default() -> Self {
        Self::new()
    }
}
impl KoreTaskBatch {
    pub const fn new() -> Self {
        Self {
            tokens: [const { None }; KORE_TASK_CAPACITY],
        }
    }
    pub fn take(&mut self, slot: usize) -> Option<SpawnToken> {
        self.tokens.get_mut(slot)?.take()
    }
    /// Native admission is per token, not transactional. Returns admitted count;
    /// refused tokens are dropped (Cancelled), while admitted tasks stay owned
    /// by the executor. Every prepared handle can be queried for its own result.
    pub fn admit<T: kore::runtime::transport::ParkTransport<Error: core::fmt::Debug>>(
        &mut self,
        executor: &Executor<'_, T>,
    ) -> usize {
        self.tokens
            .iter_mut()
            .filter_map(Option::take)
            .fold(0, |count, token| {
                count + usize::from(executor.spawn(token).is_ok())
            })
    }
}
struct Job {
    step: unsafe extern "C" fn(*mut c_void) -> KoreTaskAction,
    user: *mut c_void,
    control: &'static Control,
    generation: u64,
    terminal: u64,
}
impl Drop for Job {
    fn drop(&mut self) {
        // Release publishes all callback writes. A terminal query is the
        // foreign caller's acquire witness that its user storage is no longer
        // accessed, including cancellation before the future's first poll.
        self.control.word.store(
            (self.generation << STATE_BITS) | self.terminal,
            Ordering::Release,
        );
    }
}
async fn run(mut job: Job) {
    let mut turns = 0;
    loop {
        // SAFETY: prepare requires user to outlive the terminal handle state;
        // the native executor serializes this callback and owns the future.
        let action = unsafe { (job.step)(job.user) };
        if action.reserved != 0 {
            job.terminal = INVALID;
            return;
        }
        match action.operation {
            KORE_TASK_DONE => {
                job.terminal = COMPLETE;
                return;
            }
            KORE_TASK_YIELD => kore::runtime::executor::yield_now().await,
            KORE_TASK_SLEEP_FOR => kore::time::sleep_for(action.nanoseconds).await,
            KORE_TASK_SLEEP_UNTIL => kore::time::sleep_until_ns(action.nanoseconds).await,
            _ => {
                job.terminal = INVALID;
                return;
            }
        }
        // Zero-duration sleeps may already be ready; retain bounded poll work.
        turns += 1;
        if turns == 8 {
            turns = 0;
            kore::runtime::executor::yield_now().await;
        }
    }
}
#[kore::task]
async fn task0(job: Job) {
    run(job).await;
}
#[kore::task]
async fn task1(job: Job) {
    run(job).await;
}
#[kore::task]
async fn task2(job: Job) {
    run(job).await;
}
#[kore::task]
async fn task3(job: Job) {
    run(job).await;
}

/// Prepare a callback as one of the native statically declared tasks. Queued
/// means prepared, not executor admission. Rust founding must take/admit tokens.
/// # Safety
/// batch is live and exclusively accessed. out is aligned writable storage
/// disjoint from batch and user. step/user remain valid until a terminal state
/// is observed, even if the batch is moved or forgotten; forgetting a token is
/// not cancellation. No concurrent accesses to unsynchronized user storage,
/// retained callback-local pointers, unwind or longjmp. Handles name completion,
/// not kernel authority, and never grant a task new capabilities.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_task_prepare(
    batch: *mut KoreTaskBatch,
    step: KoreTaskStep,
    user: *mut c_void,
    out: *mut KoreTaskHandle,
) -> KoreStatus {
    let Some(step) = step else {
        return KoreStatus::InvalidArgument;
    };
    if out.is_null() {
        return KoreStatus::InvalidArgument;
    }
    // SAFETY: preparation requires a live, exclusively borrowed batch.
    let Some(batch) = (unsafe { batch.as_mut() }) else {
        return KoreStatus::InvalidArgument;
    };
    for (slot, control) in CONTROLS.iter().enumerate() {
        if batch.tokens[slot].is_some() {
            continue;
        }
        let Some(generation) = control.reserve() else {
            continue;
        };
        let job = Job {
            step,
            user,
            control,
            generation,
            terminal: CANCELLED,
        };
        let token = match slot {
            0 => task0(job),
            1 => task1(job),
            2 => task2(job),
            _ => task3(job),
        };
        match token {
            Ok(token) => {
                batch.tokens[slot] = Some(token);
                // SAFETY: caller supplies aligned disjoint output storage.
                unsafe {
                    out.write(KoreTaskHandle {
                        generation,
                        slot: slot as u32,
                        reserved: 0,
                    });
                }
                return KoreStatus::Queued;
            }
            Err(_) => {
                control
                    .word
                    .store(generation << STATE_BITS, Ordering::Release);
            }
        }
    }
    KoreStatus::Capacity
}
fn control(handle: KoreTaskHandle) -> Option<(&'static Control, u64)> {
    let control = CONTROLS.get(handle.slot as usize)?;
    let word = control.word.load(Ordering::Acquire);
    if handle.reserved != 0 || handle.generation == 0 || word >> STATE_BITS != handle.generation {
        return None;
    }
    Some((control, word))
}
/// Acquire a completion witness. Pending never permits reclaiming user storage.
/// Stale means the record was already released/reused, not that work was cancelled.
#[unsafe(no_mangle)]
pub extern "C" fn kore_task_poll(handle: KoreTaskHandle) -> KoreTaskState {
    let Some((_, word)) = control(handle) else {
        return KoreTaskState::Stale;
    };
    match word & STATE_MASK {
        PENDING => KoreTaskState::Pending,
        COMPLETE => KoreTaskState::Complete,
        CANCELLED => KoreTaskState::Cancelled,
        INVALID => KoreTaskState::InvalidAction,
        _ => KoreTaskState::Stale,
    }
}
/// Release a terminal record. Pending refuses; stale copies cannot release a
/// newer generation, including concurrent release/reuse. No native task is killed.
#[unsafe(no_mangle)]
pub extern "C" fn kore_task_release(handle: KoreTaskHandle) -> KoreStatus {
    let Some((control, word)) = control(handle) else {
        return KoreStatus::StaleHandle;
    };
    match word & STATE_MASK {
        PENDING => return KoreStatus::Busy,
        COMPLETE | CANCELLED | INVALID => {}
        _ => return KoreStatus::StaleHandle,
    }
    match control.word.compare_exchange(
        word,
        word & !STATE_MASK,
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        Ok(_) => KoreStatus::Ok,
        Err(_) => KoreStatus::StaleHandle,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use core::{
        future::Future,
        pin::pin,
        task::{Context, Poll, Waker},
    };
    #[test]
    fn unpolled_native_token_drop_cancels_without_invoking_callback() {
        unsafe extern "C" fn never(_: *mut c_void) -> KoreTaskAction {
            panic!("unadmitted callback executed")
        }
        let mut batch = KoreTaskBatch::new();
        let mut handle = KoreTaskHandle::default();
        unsafe {
            assert_eq!(
                kore_task_prepare(&mut batch, Some(never), core::ptr::null_mut(), &mut handle),
                KoreStatus::Queued
            );
        }
        assert_eq!(kore_task_poll(handle), KoreTaskState::Pending);
        assert_eq!(kore_task_release(handle), KoreStatus::Busy);
        drop(batch);
        assert_eq!(kore_task_poll(handle), KoreTaskState::Cancelled);
        assert_eq!(kore_task_release(handle), KoreStatus::Ok);
        assert_eq!(kore_task_poll(handle), KoreTaskState::Stale);
    }
    #[test]
    fn standalone_callback_program_yields_then_completes() {
        unsafe extern "C" fn step(user: *mut c_void) -> KoreTaskAction {
            let count = unsafe { &mut *user.cast::<u32>() };
            *count += 1;
            KoreTaskAction {
                operation: if *count == 1 {
                    KORE_TASK_YIELD
                } else {
                    KORE_TASK_DONE
                },
                ..Default::default()
            }
        }
        static CONTROL: Control = Control::new();
        let generation = CONTROL.reserve().unwrap();
        let mut count = 0u32;
        {
            let mut future = pin!(run(Job {
                step,
                user: (&mut count as *mut u32).cast(),
                control: &CONTROL,
                generation,
                terminal: CANCELLED
            }));
            let mut context = Context::from_waker(Waker::noop());
            assert_eq!(future.as_mut().poll(&mut context), Poll::Pending);
            assert_eq!(future.as_mut().poll(&mut context), Poll::Ready(()));
        }
        assert_eq!(count, 2);
        assert_eq!(CONTROL.word.load(Ordering::Acquire) & STATE_MASK, COMPLETE);
    }
    #[test]
    fn generations_retire_and_stale_release_cannot_free_a_new_reservation() {
        let control = Control::new();
        let first = control.reserve().unwrap();
        let terminal = (first << STATE_BITS) | COMPLETE;
        control.word.store(terminal, Ordering::Release);
        control
            .word
            .compare_exchange(
                terminal,
                first << STATE_BITS,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .unwrap();
        let second = control.reserve().unwrap();
        assert_ne!(first, second);
        assert!(
            control
                .word
                .compare_exchange(
                    terminal,
                    first << STATE_BITS,
                    Ordering::AcqRel,
                    Ordering::Acquire
                )
                .is_err()
        );
        control
            .word
            .store(MAX_GENERATION << STATE_BITS, Ordering::Release);
        assert!(control.reserve().is_none());
    }
    #[cfg(all(feature = "test-foreign", not(miri), not(target_os = "none")))]
    #[test]
    fn c_and_zig_prepare_actual_native_tokens_and_observe_cancellation() {
        #[allow(improper_ctypes)] // opaque pointees; only thin pointers cross
        unsafe extern "C" {
            fn c_prepare_task(
                batch: *mut KoreTaskBatch,
                turns: *mut u32,
                handle: *mut KoreTaskHandle,
            ) -> u32;
            fn c_check_cancelled(handle: KoreTaskHandle) -> u32;
            fn c_task_step(user: *mut c_void) -> KoreTaskAction;
            fn zig_task_step(user: *mut c_void) -> KoreTaskAction;
            fn zig_prepare_task(batch: *mut KoreTaskBatch, handle: *mut KoreTaskHandle) -> u32;
        }
        let mut batch = KoreTaskBatch::new();
        let mut turns = 0;
        let mut c = KoreTaskHandle::default();
        let mut zig = KoreTaskHandle::default();
        unsafe {
            assert_eq!(
                c_prepare_task(&mut batch, &mut turns, &mut c),
                KoreStatus::Queued as u32
            );
            assert_eq!(
                zig_prepare_task(&mut batch, &mut zig),
                KoreStatus::Queued as u32
            );
        }
        drop(batch);
        assert_eq!(turns, 0);
        unsafe {
            assert_eq!(c_check_cancelled(c), 0);
        }
        assert_eq!(kore_task_poll(zig), KoreTaskState::Cancelled);
        assert_eq!(kore_task_release(zig), KoreStatus::Ok);
        static C_CONTROL: Control = Control::new();
        static ZIG_CONTROL: Control = Control::new();
        for (step, control, yields) in [
            (
                c_task_step as unsafe extern "C" fn(*mut c_void) -> KoreTaskAction,
                &C_CONTROL,
                true,
            ),
            (zig_task_step, &ZIG_CONTROL, false),
        ] {
            let generation = control.reserve().unwrap();
            let mut future = pin!(run(Job {
                step,
                user: (&mut turns as *mut u32).cast(),
                control,
                generation,
                terminal: CANCELLED
            }));
            let mut context = Context::from_waker(Waker::noop());
            if yields {
                assert_eq!(future.as_mut().poll(&mut context), Poll::Pending);
            }
            assert_eq!(future.as_mut().poll(&mut context), Poll::Ready(()));
            assert_eq!(control.word.load(Ordering::Acquire) & STATE_MASK, COMPLETE);
        }
        assert_eq!(turns, 2);
    }
}
