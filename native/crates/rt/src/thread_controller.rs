//! Process-lifetime owner for native join capabilities.
//!
//! Clients exchange bounded, generation-checked identifiers through shared
//! records. Only the controller touches native tickets. A detached record is
//! reusable only after native exit and successful reclamation. Storage is
//! supplied by the process and must outlive every client and child.

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

const FREE: u64 = 0;
const BUILD: u64 = 1;
const REQUEST: u64 = 2;
const RUNNING: u64 = 3;
const JOIN: u64 = 4;
const DETACHED: u64 = 5;
const COMPLETE: u64 = 6;
const JOIN_READY: u64 = 7;
const CONSUME: u64 = 8;
const FAILED: u64 = 9;
const RETIRED: u64 = 15;
const MASK: u64 = 15;
const COUNT: usize = super::thread::MAX_THREADS as usize - 2;

/// An identifier within one controller, never a capability-slot number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Id {
    controller: usize,
    slot: usize,
    generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Unavailable,
    Full,
    Invalid,
    Busy,
    Deadlock,
    /// Native termination occurred without the callback returning a value.
    Terminated,
}

#[derive(Clone, Copy)]
struct Start {
    call: fn(usize) -> usize,
    argument: usize,
    retire: Option<fn(usize)>,
}
fn empty(_: usize) -> usize {
    0
}

struct Record {
    state: AtomicU64,
    start: UnsafeCell<Start>,
    value: AtomicUsize,
    identity: AtomicUsize,
    returned: AtomicUsize,
}
// The BUILD claimant alone writes start. REQUEST publishes it; it stays
// immutable until the controller has joined the child and released the record.
unsafe impl Sync for Record {}
impl Record {
    const fn new() -> Self {
        Self {
            state: AtomicU64::new(FREE),
            start: UnsafeCell::new(Start {
                call: empty,
                argument: 0,
                retire: None,
            }),
            value: AtomicUsize::new(0),
            identity: AtomicUsize::new(0),
            returned: AtomicUsize::new(0),
        }
    }
    fn release(&self, generation: u64) {
        self.state.store(
            generation.checked_add(16).unwrap_or(generation | RETIRED),
            Ordering::Release,
        );
    }
}

/// Named process-owned storage. Start once during application startup and
/// share its static reference with workers. This consumes one native thread
/// for the process lifetime. No native ticket escapes its owner.
pub struct Controller {
    started: AtomicUsize,
    records: [Record; COUNT],
}
impl Default for Controller {
    fn default() -> Self {
        Self::new()
    }
}
impl Controller {
    pub const fn new() -> Self {
        Self {
            started: AtomicUsize::new(0),
            records: [const { Record::new() }; COUNT],
        }
    }

    /// Start the owner. Concurrent/repeated calls do not create another owner.
    pub fn start(&'static self) -> Result<(), Error> {
        self.started
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::Busy)?;
        let heap = heap_endowment();
        match super::thread::spawn_with_arg(
            controller_entry,
            self as *const Self as usize,
            heap.as_slice(),
        ) {
            Some(ticket) => {
                // This owner deliberately lives until process teardown. Managed
                // children never use native detach: their tickets are retained.
                ticket.detach();
                self.started.store(2, Ordering::Release);
                Ok(())
            }
            None => {
                self.started.store(0, Ordering::Release);
                Err(Error::Unavailable)
            }
        }
    }

    /// Start a callback. Any pointed-to argument storage must remain valid
    /// through callback return (or join); the integer itself grants no authority.
    /// A safe Rust callback must validate any pointer it chooses to dereference.
    pub fn create(&'static self, call: fn(usize) -> usize, argument: usize) -> Result<Id, Error> {
        self.create_with_cleanup(call, argument, None)
    }

    /// Transfer callback context custody after a successful create. `retire`
    /// runs once on the controller after native reclamation, including when the
    /// callback exits without returning. On error custody remains with caller.
    /// The cleanup must not make blocking calls back into this controller.
    pub fn create_with_cleanup(
        &'static self,
        call: fn(usize) -> usize,
        argument: usize,
        retire: Option<fn(usize)>,
    ) -> Result<Id, Error> {
        if self.started.load(Ordering::Acquire) != 2 {
            return Err(Error::Unavailable);
        }
        for (slot, record) in self.records.iter().enumerate() {
            let old = record.state.load(Ordering::Acquire);
            if old & MASK != FREE {
                continue;
            }
            if record
                .state
                .compare_exchange(old, old | BUILD, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                continue;
            }
            // SAFETY: exclusive BUILD claim; prior child was joined before FREE.
            unsafe {
                *record.start.get() = Start {
                    call,
                    argument,
                    retire,
                };
            }
            record.identity.store(0, Ordering::Relaxed);
            record.value.store(0, Ordering::Relaxed);
            record.returned.store(0, Ordering::Relaxed);
            record.state.store(old | REQUEST, Ordering::Release);
            loop {
                let state = record.state.load(Ordering::Acquire);
                if state == old | FAILED {
                    record.release(old);
                    return Err(Error::Unavailable);
                }
                if state != old | REQUEST {
                    return Ok(Id {
                        controller: self as *const Self as usize,
                        slot,
                        generation: old,
                    });
                }
                pause();
            }
        }
        Err(Error::Full)
    }

    fn record(&self, id: Id) -> Result<&Record, Error> {
        if id.controller != self as *const Self as usize {
            return Err(Error::Invalid);
        }
        let record = self.records.get(id.slot).ok_or(Error::Invalid)?;
        if record.state.load(Ordering::Acquire) & !MASK != id.generation {
            return Err(Error::Invalid);
        }
        Ok(record)
    }

    /// Identify the current managed native thread. Initial/unmanaged threads
    /// are not represented by a managed record.
    pub fn current(&self) -> Option<Id> {
        #[cfg(target_os = "none")]
        {
            let identity = crate::entry::geometry::thread_owner_identity()?;
            for (slot, record) in self.records.iter().enumerate() {
                let state = record.state.load(Ordering::Acquire);
                if matches!(state & MASK, REQUEST | RUNNING | JOIN | DETACHED)
                    && record.identity.load(Ordering::Acquire) == identity
                {
                    return Some(Id {
                        controller: self as *const Self as usize,
                        slot,
                        generation: state & !MASK,
                    });
                }
            }
        }
        None
    }

    /// Join from any native worker. Exactly one joiner or detacher wins.
    /// Returns the callback value only after native resources are reclaimed.
    pub fn join(&self, id: Id) -> Result<usize, Error> {
        let record = self.record(id)?;
        #[cfg(target_os = "none")]
        if record.identity.load(Ordering::Acquire) != 0
            && Some(record.identity.load(Ordering::Relaxed))
                == crate::entry::geometry::thread_owner_identity()
        {
            return Err(Error::Deadlock);
        }
        loop {
            let old = record.state.load(Ordering::Acquire);
            if old & !MASK != id.generation {
                return Err(Error::Invalid);
            }
            let next = match old & MASK {
                RUNNING => JOIN,
                COMPLETE => CONSUME,
                FREE | RETIRED => return Err(Error::Invalid),
                _ => return Err(Error::Busy),
            };
            if record
                .state
                .compare_exchange(
                    old,
                    id.generation | next,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_err()
            {
                continue;
            }
            if next == JOIN {
                while record.state.load(Ordering::Acquire) != id.generation | JOIN_READY {
                    pause();
                }
            }
            // Acquire completion before reading the value it publishes. The owner
            // is a different thread from the callback on weakly ordered ISAs.
            let returned = record.returned.load(Ordering::Acquire) != 0;
            let result = record.value.load(Ordering::Relaxed);
            record.release(id.generation);
            return if returned {
                Ok(result)
            } else {
                Err(Error::Terminated)
            };
        }
    }

    /// Give up joining; the owner still reaps the native thread after exit.
    pub fn detach(&self, id: Id) -> Result<(), Error> {
        let record = self.record(id)?;
        loop {
            let old = record.state.load(Ordering::Acquire);
            if old & !MASK != id.generation {
                return Err(Error::Invalid);
            }
            let next = match old & MASK {
                RUNNING => DETACHED,
                COMPLETE => CONSUME,
                FREE | RETIRED => return Err(Error::Invalid),
                _ => return Err(Error::Busy),
            };
            if record
                .state
                .compare_exchange(
                    old,
                    id.generation | next,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_err()
            {
                continue;
            }
            if next == CONSUME {
                record.release(id.generation);
            }
            return Ok(());
        }
    }
}

fn pause() {
    if !crate::pace::cede() {
        core::hint::spin_loop();
    }
}

fn child_entry(argument: usize) {
    // SAFETY: controller storage is static; record cannot be reused until join.
    let record = unsafe { &*(argument as *const Record) };
    #[cfg(target_os = "none")]
    record.identity.store(
        crate::entry::geometry::thread_owner_identity().unwrap_or(0),
        Ordering::Release,
    );
    // Acquire the request publication (or the owner's subsequent state)
    // before reading callback storage. It remains immutable through native join.
    let _ = record.state.load(Ordering::Acquire);
    let start = unsafe { *record.start.get() };
    record
        .value
        .store((start.call)(start.argument), Ordering::Release);
    record.returned.store(1, Ordering::Release);
}

fn controller_entry(argument: usize) {
    // SAFETY: start requires static storage.
    let controller = unsafe { &*(argument as *const Controller) };
    let heap = heap_endowment();
    let mut tickets: [Option<super::thread::Thread>; COUNT] = [const { None }; COUNT];
    loop {
        for (record, ticket) in controller.records.iter().zip(tickets.iter_mut()) {
            let state = record.state.load(Ordering::Acquire);
            let generation = state & !MASK;
            if state & MASK == REQUEST {
                *ticket = super::thread::spawn_with_arg(
                    child_entry,
                    record as *const Record as usize,
                    heap.as_slice(),
                );
                record.state.store(
                    generation | if ticket.is_some() { RUNNING } else { FAILED },
                    Ordering::Release,
                );
            }
            let Some(child) = ticket.as_ref() else {
                continue;
            };
            if !matches!(
                child.poll(),
                Ok(crate::cap::TaskPollOutcome::Exited(_) | crate::cap::TaskPollOutcome::Reaped)
            ) {
                continue;
            }
            match super::thread::join(ticket.take().unwrap()) {
                Err(error) => *ticket = Some(error.into_thread()),
                Ok(_) => {
                    // Child can no longer access its context; cleanup also runs
                    // for abrupt native exit and detached completion.
                    let start = unsafe { *record.start.get() };
                    if let Some(retire) = start.retire {
                        retire(start.argument);
                    }
                    loop {
                        let old = record.state.load(Ordering::Acquire);
                        let next = match old & MASK {
                            RUNNING => COMPLETE,
                            JOIN => JOIN_READY,
                            DETACHED => {
                                record.release(generation);
                                break;
                            }
                            _ => unreachable!(),
                        };
                        if record
                            .state
                            .compare_exchange(
                                old,
                                generation | next,
                                Ordering::Release,
                                Ordering::Acquire,
                            )
                            .is_ok()
                        {
                            break;
                        }
                    }
                }
            }
        }
        pause();
    }
}

// Each owner derives the process heap into its children using its own local
// slot. No creator's slot number is ever forwarded across the controller seam.
fn heap_endowment() -> Option<super::thread::WorkerCap> {
    #[cfg(target_os = "none")]
    {
        use kore_abi::{
            cap::{ResourceKind, Rights},
            task::role,
        };
        let rt = crate::entry::thread_context()?;
        Some(super::thread::WorkerCap {
            role: role::HEAP,
            index: 0,
            subject: rt.bootstrap().handle(role::HEAP, 0)?,
            kind: ResourceKind::MemLease as u16 as u8,
            rights: Rights::ACCESS | Rights::MODIFY | Rights::DERIVE,
        })
    }
    #[cfg(not(target_os = "none"))]
    None
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    fn completed(controller: &Controller, generation: u64) -> Id {
        let record = &controller.records[0];
        record.value.store(73, Ordering::Relaxed);
        record.returned.store(1, Ordering::Relaxed);
        record.state.store(generation | COMPLETE, Ordering::Release);
        Id {
            controller: controller as *const Controller as usize,
            slot: 0,
            generation,
        }
    }
    #[test]
    fn concurrent_joiners_cannot_consume_or_release_twice() {
        let controller = Controller::new();
        let id = completed(&controller, 0);
        std::thread::scope(|scope| {
            let a = scope.spawn(|| controller.join(id));
            let b = scope.spawn(|| controller.join(id));
            let outcomes = [a.join().unwrap(), b.join().unwrap()];
            assert_eq!(
                outcomes.iter().filter(|result| **result == Ok(73)).count(),
                1
            );
        });
        assert_eq!(controller.records[0].state.load(Ordering::Acquire), 16);
        let next = completed(&controller, 16);
        assert_eq!(controller.join(id), Err(Error::Invalid));
        assert_eq!(controller.join(next), Ok(73));
    }
    #[test]
    fn controller_identity_exit_without_return_and_generation_exhaustion() {
        let controller = Controller::new();
        let other = Controller::new();
        let id = completed(&controller, 0);
        assert_eq!(other.detach(id), Err(Error::Invalid));
        controller.records[0].returned.store(0, Ordering::Relaxed);
        assert_eq!(controller.join(id), Err(Error::Terminated));
        let final_id = completed(&controller, u64::MAX & !MASK);
        assert_eq!(controller.detach(final_id), Ok(()));
        assert_eq!(
            controller.records[0].state.load(Ordering::Acquire),
            u64::MAX
        );
        assert_eq!(controller.join(final_id), Err(Error::Invalid));
    }
}
