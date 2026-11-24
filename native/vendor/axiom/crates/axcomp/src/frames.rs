//! Frame ownership above the target ring. The ring remains the single source
//! of producer/visibility/presentation/release truth; this layer retains input
//! owners and prevents old-configuration work from being newly committed.
use crate::{TargetId, TargetRing, TargetRingError};
use alloc::vec::Vec;
use axgfx::{RetainUntil, SubmissionState, SurfaceEpoch, TransitionError};
use core::alloc::Allocator;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameId {
    serial: u64,
    target: TargetId,
    configuration: u64,
    epoch: SurfaceEpoch,
}
impl FrameId {
    pub const fn target(self) -> TargetId {
        self.target
    }
    pub const fn configuration(self) -> u64 {
        self.configuration
    }
    pub const fn epoch(self) -> SurfaceEpoch {
        self.epoch
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameError {
    Target(TargetRingError),
    Transition(TransitionError),
    ResourceLimit,
    InvalidFrame,
    StaleConfiguration,
    EpochExhausted,
}
impl From<TargetRingError> for FrameError {
    fn from(error: TargetRingError) -> Self {
        Self::Target(error)
    }
}
impl From<TransitionError> for FrameError {
    fn from(error: TransitionError) -> Self {
        Self::Transition(error)
    }
}

struct Flight<I> {
    id: FrameId,
    inputs: Option<I>,
    accepted: bool,
}

/// Bounded, caller-driven frame ownership; no rendering or WM policy.
///
/// `I` owns the input leases/registrations needed by composition, and `T` owns
/// the output target. Neither is a borrowed pixel slice or a wire capability.
/// The adapter supplies truthful observations and releases its actual provider
/// event before calling `submission_released`. Input owners then return to the
/// caller; output ownership stays until presentation/abandonment permits reuse.
///
/// Reconfiguration does not cancel work or fabricate completion. Old work is
/// drained and abandoned, or its already-committed presentation is acknowledged.
/// Dropping with accepted frames still outstanding quarantines owned inputs and
/// targets instead of destroying backing a device/consumer might still use.
/// Successful callers explicitly drain/recycle and take retired targets.
pub struct FrameQueue<T, I, A: Allocator> {
    targets: Option<TargetRing<T, A>>,
    flights: Vec<Option<Flight<I>>, A>,
    configuration: u64,
    installed: usize,
    next_frame: u64,
}

impl<T, I, A: Allocator + Clone> FrameQueue<T, I, A> {
    pub fn with_capacity_in(capacity: usize, allocator: A) -> Result<Self, FrameError> {
        if capacity == 0 {
            return Err(FrameError::ResourceLimit);
        }
        let targets = TargetRing::with_capacity_in(capacity, allocator.clone())?;
        let mut flights = Vec::new_in(allocator);
        flights
            .try_reserve_exact(capacity)
            .map_err(|_| TargetRingError::Allocation)?;
        flights.resize_with(capacity, || None);
        Ok(Self {
            targets: Some(targets),
            flights,
            configuration: 1,
            installed: 0,
            next_frame: 1,
        })
    }
}

impl<T, I, A: Allocator> FrameQueue<T, I, A> {
    pub const fn configuration(&self) -> u64 {
        self.configuration
    }

    pub fn install(&mut self, target: T) -> Result<TargetId, FrameError> {
        if self.installed == self.flights.len() {
            return Err(FrameError::ResourceLimit);
        }
        let id = self.targets.as_mut().unwrap().install(target)?;
        self.installed += 1;
        Ok(id)
    }

    /// No provider work has been accepted yet. On backpressure/refusal the
    /// complete input owner is returned, with no graph execution or retry.
    pub fn begin(&mut self, epoch: SurfaceEpoch, inputs: I) -> Result<FrameId, (FrameError, I)> {
        let Some(next_frame) = self.next_frame.checked_add(1) else {
            return Err((FrameError::EpochExhausted, inputs));
        };
        let Some(slot) = self.flights.iter().position(Option::is_none) else {
            return Err((FrameError::ResourceLimit, inputs));
        };
        let target = match self.targets.as_mut().unwrap().acquire(epoch) {
            Ok(Some(id)) => id,
            Ok(None) => return Err((FrameError::ResourceLimit, inputs)),
            Err(error) => return Err((error.into(), inputs)),
        };
        let id = FrameId {
            serial: self.next_frame,
            target,
            configuration: self.configuration,
            epoch,
        };
        self.next_frame = next_frame;
        self.flights[slot] = Some(Flight {
            id,
            inputs: Some(inputs),
            accepted: false,
        });
        Ok(id)
    }

    pub fn target(&self, id: FrameId) -> Result<&T, FrameError> {
        self.slot(id)?;
        Ok(self.targets.as_ref().unwrap().target(id.target)?)
    }
    pub fn target_mut(&mut self, id: FrameId) -> Result<&mut T, FrameError> {
        self.slot(id)?;
        Ok(self.targets.as_mut().unwrap().target_mut(id.target)?)
    }
    pub fn inputs(&self, id: FrameId) -> Result<&I, FrameError> {
        self.flights[self.slot(id)?]
            .as_ref()
            .unwrap()
            .inputs
            .as_ref()
            .ok_or(FrameError::InvalidFrame)
    }

    /// Also used for indeterminate acceptance: retain the actual event and
    /// report its later state, never reinterpret uncertainty as rejection.
    pub fn submitted(&mut self, id: FrameId, retain: RetainUntil) -> Result<(), FrameError> {
        let slot = self.slot(id)?;
        self.targets
            .as_mut()
            .unwrap()
            .mark_submitted(id.target, retain)?;
        self.flights[slot].as_mut().unwrap().accepted = true;
        Ok(())
    }

    pub fn reject(&mut self, id: FrameId) -> Result<I, FrameError> {
        let slot = self.slot(id)?;
        self.targets.as_mut().unwrap().reject(id.target)?;
        Ok(self.flights[slot].take().unwrap().inputs.unwrap())
    }

    pub fn observe(&mut self, id: FrameId, state: SubmissionState) -> Result<(), FrameError> {
        self.slot(id)?;
        self.targets
            .as_mut()
            .unwrap()
            .retirement_mut(id.target)?
            .observe(state)?;
        Ok(())
    }

    pub fn can_release_submission(&mut self, id: FrameId) -> Result<bool, FrameError> {
        self.slot(id)?;
        Ok(self
            .targets
            .as_mut()
            .unwrap()
            .retirement_mut(id.target)?
            .can_release_submission())
    }

    /// Call only after actual backend submission release succeeds. This is
    /// distinct from completion, and is the earliest return of input owners.
    pub fn submission_released(&mut self, id: FrameId) -> Result<I, FrameError> {
        let slot = self.slot(id)?;
        self.targets
            .as_mut()
            .unwrap()
            .retirement_mut(id.target)?
            .mark_submission_released()?;
        self.flights[slot]
            .as_mut()
            .unwrap()
            .inputs
            .take()
            .ok_or(FrameError::InvalidFrame)
    }

    pub fn commit(&mut self, id: FrameId) -> Result<(), FrameError> {
        self.slot(id)?;
        if id.configuration != self.configuration {
            return Err(FrameError::StaleConfiguration);
        }
        self.targets
            .as_mut()
            .unwrap()
            .retirement_mut(id.target)?
            .mark_committed()?;
        Ok(())
    }
    pub fn presented(&mut self, id: FrameId, epoch: SurfaceEpoch) -> Result<(), FrameError> {
        self.slot(id)?;
        self.targets
            .as_mut()
            .unwrap()
            .retirement_mut(id.target)?
            .mark_presented(epoch)?;
        Ok(())
    }
    pub fn abandon(&mut self, id: FrameId) -> Result<(), FrameError> {
        self.slot(id)?;
        self.targets
            .as_mut()
            .unwrap()
            .retirement_mut(id.target)?
            .abandon()?;
        Ok(())
    }

    pub fn recycle(&mut self, id: FrameId) -> Result<(), FrameError> {
        let slot = self.slot(id)?;
        self.targets.as_mut().unwrap().recycle(id.target)?;
        // is_reusable requires submission release; that operation returned I.
        debug_assert!(self.flights[slot].as_ref().unwrap().inputs.is_none());
        self.flights[slot] = None;
        Ok(())
    }

    pub fn begin_reconfigure(&mut self) -> Result<u64, FrameError> {
        self.configuration = self
            .configuration
            .checked_add(1)
            .ok_or(FrameError::EpochExhausted)?;
        self.targets.as_mut().unwrap().begin_reconfigure();
        Ok(self.configuration)
    }

    pub fn take_retired(&mut self) -> Option<(TargetId, T)> {
        let retired = self.targets.as_mut().unwrap().take_retired()?;
        self.installed -= 1;
        Some(retired)
    }

    pub fn in_flight(&self) -> usize {
        self.flights.iter().filter(|v| v.is_some()).count()
    }

    fn slot(&self, id: FrameId) -> Result<usize, FrameError> {
        self.flights
            .iter()
            .position(|v| v.as_ref().is_some_and(|v| v.id == id))
            .ok_or(FrameError::InvalidFrame)
    }
}

impl<T, I, A: Allocator> Drop for FrameQueue<T, I, A> {
    fn drop(&mut self) {
        if self.flights.iter().flatten().any(|flight| flight.accepted) {
            // A portable layer cannot cancel a device or wait for a consumer.
            // Leak/quarantine on this exceptional path, never release early.
            core::mem::forget(self.targets.take());
            for flight in self.flights.iter_mut().flatten() {
                core::mem::forget(flight.inputs.take());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axgfx::{DeviceEpoch, TargetReuse, VisibilityEpoch};
    use std::{alloc::Global, cell::Cell, rc::Rc};

    fn epoch(n: u64) -> SurfaceEpoch {
        SurfaceEpoch::new(n).unwrap()
    }
    fn visible(n: u64) -> SubmissionState {
        SubmissionState::Visible {
            device: DeviceEpoch::new(n).unwrap(),
            visibility: VisibilityEpoch::new(n).unwrap(),
        }
    }

    #[test]
    fn resize_drains_pending_and_presenting_frames_without_reusing_their_targets() {
        let mut q = FrameQueue::with_capacity_in(3, Global).unwrap();
        q.install(10).unwrap();
        q.install(20).unwrap();
        let first = q.begin(epoch(1), "window one").unwrap();
        q.submitted(first, RetainUntil::Visible).unwrap();
        q.observe(first, visible(1)).unwrap();
        assert_eq!(q.submission_released(first).unwrap(), "window one");
        q.commit(first).unwrap();
        assert!(q.recycle(first).is_err());
        let second = q.begin(epoch(2), "window two").unwrap();
        q.submitted(second, RetainUntil::Presented).unwrap();
        assert!(q.commit(second).is_err());
        assert!(!q.can_release_submission(second).unwrap());
        q.begin_reconfigure().unwrap();
        assert!(q.take_retired().is_none());
        q.install(30).unwrap();
        let third = q.begin(epoch(3), "new configuration").unwrap();
        assert_eq!(*q.target(third).unwrap(), 30);
        q.submitted(third, RetainUntil::Visible).unwrap();
        q.observe(second, visible(2)).unwrap();
        assert_eq!(q.commit(second), Err(FrameError::StaleConfiguration));
        assert!(q.submission_released(second).is_err());
        q.abandon(second).unwrap();
        assert_eq!(q.submission_released(second).unwrap(), "window two");
        q.recycle(second).unwrap();
        assert_eq!(q.take_retired().unwrap().1, 20);
        q.presented(first, epoch(1)).unwrap();
        q.recycle(first).unwrap();
        assert_eq!(q.take_retired().unwrap().1, 10);
        assert_eq!(q.target(first), Err(FrameError::InvalidFrame));
        q.observe(
            third,
            SubmissionState::Cancelled {
                target: TargetReuse::Safe,
            },
        )
        .unwrap();
        q.submission_released(third).unwrap();
        q.recycle(third).unwrap();
        assert_eq!(q.in_flight(), 0);
    }

    #[test]
    fn rejected_attempts_never_alias_a_later_frame_with_the_same_epoch() {
        let mut q = FrameQueue::with_capacity_in(1, Global).unwrap();
        q.install(()).unwrap();
        let old = q.begin(epoch(7), 42).unwrap();
        assert_eq!(q.begin(epoch(8), 99), Err((FrameError::ResourceLimit, 99)));
        assert_eq!(q.reject(old).unwrap(), 42);
        let new = q.begin(epoch(7), 43).unwrap();
        assert_ne!(old, new);
        assert_eq!(old.target(), new.target());
        assert_eq!(
            q.submitted(old, RetainUntil::Visible),
            Err(FrameError::InvalidFrame)
        );
        q.reject(new).unwrap();
    }

    #[derive(Debug)]
    struct Owner(Rc<Cell<usize>>);
    impl Drop for Owner {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    #[test]
    fn unknown_cancellation_cannot_release_and_drop_quarantines_owned_backing() {
        let dropped = Rc::new(Cell::new(0));
        let mut q = FrameQueue::with_capacity_in(1, Global).unwrap();
        q.install(Owner(dropped.clone())).unwrap();
        let id = q.begin(epoch(1), Owner(dropped.clone())).unwrap();
        q.submitted(id, RetainUntil::Visible).unwrap();
        q.observe(
            id,
            SubmissionState::Cancelled {
                target: TargetReuse::Quarantine,
            },
        )
        .unwrap();
        assert!(!q.can_release_submission(id).unwrap());
        assert!(q.submission_released(id).is_err());
        assert!(q.recycle(id).is_err());
        drop(q);
        // Two deliberately quarantined tiny test owners; no device work.
        assert_eq!(dropped.get(), 0);
    }
}
