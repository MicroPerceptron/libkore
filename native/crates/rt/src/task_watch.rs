//! Event-driven task observation shared by shell jobs and window sessions.
//! One bounded sibling waits on the existing kernel join and wakes the owner.
//! The caller retains the lifecycle capability and must join the observer
//! before reclaiming its cell. Dropping a live observer retains storage until
//! process teardown, matching native thread custody.
use crate::{
    cap::{TaskPollOutcome, TaskToken, TaskWaitError, TaskWaitOutcome, verbs},
    thread::{self, Thread, WorkerCap},
};
use alloc::boxed::Box;
use core::alloc::Allocator;
use core::sync::atomic::{AtomicI32, AtomicU32, Ordering};
use kore_abi::{
    cap::{ResourceKind, Rights},
    session::SlotRef,
    task::role,
};

// Local publication states; never wire values. Only the owner reuses a cell,
// after joining the observer that last published into it.
const WAITER_VACANT: u32 = 0;
const WAITER_WAITING: u32 = 1;
const WAITER_EXITED: u32 = 2;
const WAITER_REAPED: u32 = 3;
const WAITER_FAULT: u32 = 4;
const WAITER_TASK_ROLE: u16 = role::USER_BASE;
// The existing controller thread-0 doorbell endowment index.
const WAITER_KICK_PEER: u16 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WatchError {
    Allocation,
    Busy,
    NoDoorbell,
    Spawn,
}

#[must_use = "retain and settle the observer before releasing its backing"]
pub struct TaskWatch<A: Allocator + 'static> {
    cell: Option<Box<TaskWaitCell, A>>,
    worker: Option<Thread>,
}
impl<A: Allocator + 'static> TaskWatch<A> {
    pub fn new_in(arena: A) -> Result<Self, WatchError> {
        Ok(Self {
            cell: Some(
                Box::try_new_in(TaskWaitCell::new(), arena).map_err(|_| WatchError::Allocation)?,
            ),
            worker: None,
        })
    }
    pub fn is_idle(&self) -> bool {
        self.worker.is_none()
    }
    pub fn poll(&self) -> Result<TaskPollOutcome, TaskWaitError> {
        self.cell.as_ref().expect("owned observer cell").poll()
    }
    /// Delegate ACCESS only; the owner keeps its revoke authority. Requires
    /// a delegable task handle and the existing controller doorbell.
    pub fn start(&mut self, task: &TaskToken) -> Result<(), WatchError> {
        if self.worker.is_some() {
            return Err(WatchError::Busy);
        }
        let kick = crate::entry::thread_context()
            .and_then(|rt| rt.bootstrap().handle(role::DOORBELL, WAITER_KICK_PEER))
            .ok_or(WatchError::NoDoorbell)?;
        let cell = self.cell.as_ref().expect("owned observer cell");
        if !cell.claim() {
            return Err(WatchError::Busy);
        }
        let endow = [
            WorkerCap {
                role: WAITER_TASK_ROLE,
                index: 0,
                subject: task.slot(),
                kind: ResourceKind::Task as u16 as u8,
                rights: Rights::ACCESS,
            },
            WorkerCap {
                role: role::DOORBELL,
                index: WAITER_KICK_PEER,
                subject: kick,
                kind: ResourceKind::ChanSend as u16 as u8,
                rights: verbs::NOTICE_RIGHTS,
            },
        ];
        let Some(worker) =
            thread::spawn_with_arg(task_waiter, &**cell as *const TaskWaitCell as usize, &endow)
        else {
            cell.release();
            return Err(WatchError::Spawn);
        };
        self.worker = Some(worker);
        Ok(())
    }
    /// One nonblocking retirement pass. None means custody is retained;
    /// terminal observation is returned only after the worker has joined.
    pub fn settle(&mut self) -> Option<Result<TaskPollOutcome, TaskWaitError>> {
        let worker = self.worker.take()?;
        if matches!(worker.poll(), Ok(TaskPollOutcome::Running) | Err(_)) {
            self.worker = Some(worker);
            return None;
        }
        match thread::join(worker) {
            Ok(_) => {
                let cell = self.cell.as_ref().expect("owned observer cell");
                let result = cell.poll();
                cell.release();
                Some(result)
            }
            Err(error) => {
                self.worker = Some(error.into_thread());
                None
            }
        }
    }
}
impl<A: Allocator + 'static> Drop for TaskWatch<A> {
    fn drop(&mut self) {
        if self.worker.is_some() {
            if let Some(cell) = self.cell.take() {
                let _ = Box::leak(cell);
            }
        }
    }
}

struct TaskWaitCell {
    state: AtomicU32,
    status: AtomicI32,
}

impl TaskWaitCell {
    const fn new() -> Self {
        Self {
            state: AtomicU32::new(WAITER_VACANT),
            status: AtomicI32::new(0),
        }
    }

    fn claim(&self) -> bool {
        let claimed = self
            .state
            .compare_exchange(
                WAITER_VACANT,
                WAITER_WAITING,
                Ordering::SeqCst,
                Ordering::SeqCst,
            )
            .is_ok();
        if claimed {
            self.status.store(0, Ordering::Relaxed);
        }
        claimed
    }

    fn publish(&self, state: u32, status: i32) {
        self.status.store(status, Ordering::Relaxed);
        self.state.store(state, Ordering::Release);
    }

    fn poll(&self) -> Result<TaskPollOutcome, TaskWaitError> {
        match self.state.load(Ordering::Acquire) {
            WAITER_WAITING => Ok(TaskPollOutcome::Running),
            WAITER_EXITED => Ok(TaskPollOutcome::Exited(self.status.load(Ordering::Relaxed))),
            WAITER_REAPED => Ok(TaskPollOutcome::Reaped),
            WAITER_FAULT => Err(TaskWaitError::Transport),
            _ => Err(TaskWaitError::MalformedCompletion),
        }
    }

    fn release(&self) {
        self.status.store(0, Ordering::Relaxed);
        self.state.store(WAITER_VACANT, Ordering::SeqCst);
    }
}

fn publish_task_wait(
    cell: &TaskWaitCell,
    outcome: Result<TaskWaitOutcome, ()>,
    kick: Option<SlotRef>,
) {
    match outcome {
        Ok(TaskWaitOutcome::Exited(status)) => cell.publish(WAITER_EXITED, status),
        Ok(TaskWaitOutcome::Reaped) => cell.publish(WAITER_REAPED, 0),
        Err(()) => cell.publish(WAITER_FAULT, 0),
    }
    // The cell is durable truth; this notice is only the condition-variable
    // edge that brings thread 0 back to inspect it. The wake bell closes the
    // publish-vs-park race and suppresses a notice while the executor is awake.
    if crate::wake::ring(0)
        && let Some(kick) = kick
    {
        let _ = verbs::post_notice(kick, crate::svc::COOKIE_REPORT, 0);
    }
}

fn task_waiter(address: usize) {
    let Some(rt) = crate::entry::thread_context() else {
        // No bootstrap or kick authority; returning still completes the thread
        // join ticket, which the controller observes during waiter teardown.
        return;
    };
    let boot = rt.bootstrap();
    let kick = boot.handle(role::DOORBELL, WAITER_KICK_PEER);
    // The parent stamps the waiter index on its one task capability. This
    // removes the per-index entry-point table when the window budget changes.
    let Some((_, _, slot)) = boot
        .handles()
        .find(|(role, _, _)| *role == WAITER_TASK_ROLE)
    else {
        return;
    };
    let outcome = verbs::take_handle(slot)
        .ok_or(())
        .and_then(|task| task.wait().map_err(|_| ()));
    // SAFETY: the parent owns the stable heap cell until this exact thread
    // has joined. Abandoning a running Session leaks the cell fail-closed.
    let cell = unsafe { &*(address as *const TaskWaitCell) };
    publish_task_wait(cell, outcome, kick);
}

#[cfg(test)]
mod task_wait_cell_preserves_signed_exit_and_bars_reuse_until_release_tests {
    use super::*;
    #[test]
    fn task_wait_cell_preserves_signed_exit_and_bars_reuse_until_release() {
        let cell = TaskWaitCell::new();
        assert!(cell.claim());
        assert!(!cell.claim(), "one slot cannot name two waiter generations");
        assert_eq!(cell.poll(), Ok(TaskPollOutcome::Running));

        cell.publish(WAITER_EXITED, -17);
        assert_eq!(cell.poll(), Ok(TaskPollOutcome::Exited(-17)));
        assert!(
            !cell.claim(),
            "terminal truth remains owned until worker join"
        );

        cell.release();
        assert!(
            cell.claim(),
            "joined worker makes the bounded slot reusable"
        );
    }
}

#[cfg(test)]
mod task_wait_cell_keeps_reap_and_wait_failure_distinct_tests {
    use super::*;
    #[test]
    fn task_wait_cell_keeps_reap_and_wait_failure_distinct() {
        let cell = TaskWaitCell::new();
        assert!(cell.claim());
        cell.publish(WAITER_REAPED, 0);
        assert_eq!(cell.poll(), Ok(TaskPollOutcome::Reaped));

        cell.release();
        assert!(cell.claim());
        cell.publish(WAITER_FAULT, 0);
        assert_eq!(cell.poll(), Err(TaskWaitError::Transport));
    }
}
