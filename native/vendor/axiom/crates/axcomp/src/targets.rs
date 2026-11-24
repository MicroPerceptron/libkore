use alloc::{collections::TryReserveError, vec::Vec};
use core::alloc::Allocator;

use axgfx::{RetainUntil, SurfaceEpoch, SurfaceRetirement};

/// Generation-checked index of a compositor-owned target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TargetId {
    index: usize,
    generation: u64,
}

/// Invalid target-ring operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetRingError {
    /// The supplied allocator could not grow target metadata.
    Allocation,
    /// The target ID is stale or no longer installed.
    InvalidTarget,
    /// The target is in the wrong lifecycle state.
    InvalidState,
    /// A submitted target has not completed all reuse requirements.
    Busy,
    /// The producer epoch does not advance the last accepted epoch.
    StaleEpoch,
}

impl From<TryReserveError> for TargetRingError {
    fn from(_: TryReserveError) -> Self {
        Self::Allocation
    }
}

enum SlotState {
    Available,
    Acquired(SurfaceEpoch),
    Submitted(SurfaceRetirement),
}

struct TargetSlot<T> {
    target: Option<T>,
    generation: u64,
    retire_on_reuse: bool,
    state: SlotState,
}

/// Caller-driven target ring with presentation-aware reuse and reconfigure
/// retirement.
///
/// Provider events remain owned by the runtime adapter. The ring owns only the
/// platform target and the portable [`SurfaceRetirement`] books, keeping Kore
/// handles and backend submission types out of the compositor core.
pub struct TargetRing<T, A: Allocator> {
    slots: Vec<TargetSlot<T>, A>,
    cursor: usize,
    next_generation: u64,
    last_submitted: Option<SurfaceEpoch>,
}

impl<T, A: Allocator> TargetRing<T, A> {
    /// Creates a ring and reserves target metadata up front.
    pub fn with_capacity_in(capacity: usize, allocator: A) -> Result<Self, TargetRingError> {
        let mut slots = Vec::new_in(allocator);
        slots.try_reserve_exact(capacity)?;
        Ok(Self {
            slots,
            cursor: 0,
            next_generation: 1,
            last_submitted: None,
        })
    }

    /// Installs a platform target without assuming how it was allocated.
    pub fn install(&mut self, target: T) -> Result<TargetId, TargetRingError> {
        let generation = self.next_generation;
        self.next_generation = self.next_generation.wrapping_add(1).max(1);
        if let Some((index, slot)) = self
            .slots
            .iter_mut()
            .enumerate()
            .find(|(_, slot)| slot.target.is_none())
        {
            *slot = TargetSlot {
                target: Some(target),
                generation,
                retire_on_reuse: false,
                state: SlotState::Available,
            };
            return Ok(TargetId { index, generation });
        }
        self.slots.try_reserve(1)?;
        let index = self.slots.len();
        self.slots.push(TargetSlot {
            target: Some(target),
            generation,
            retire_on_reuse: false,
            state: SlotState::Available,
        });
        Ok(TargetId { index, generation })
    }

    /// Acquires the next reusable target for an exact producer epoch.
    pub fn acquire(&mut self, epoch: SurfaceEpoch) -> Result<Option<TargetId>, TargetRingError> {
        if self
            .last_submitted
            .is_some_and(|accepted| epoch <= accepted)
        {
            return Err(TargetRingError::StaleEpoch);
        }
        if self
            .slots
            .iter()
            .any(|slot| matches!(slot.state, SlotState::Acquired(_)))
        {
            // Keep the provider acceptance edge serial: a caller must either
            // mark or reject the acquired target before issuing another
            // producer epoch. Submitted/presenting targets remain concurrent.
            return Err(TargetRingError::Busy);
        }
        if self.slots.is_empty() {
            return Ok(None);
        }
        let slot_count = self.slots.len();
        for offset in 0..slot_count {
            let index = (self.cursor + offset) % slot_count;
            let slot = &mut self.slots[index];
            if slot.target.is_some()
                && !slot.retire_on_reuse
                && matches!(slot.state, SlotState::Available)
            {
                slot.state = SlotState::Acquired(epoch);
                self.cursor = (index + 1) % slot_count;
                return Ok(Some(TargetId {
                    index,
                    generation: slot.generation,
                }));
            }
        }
        Ok(None)
    }

    /// Borrows an installed target for rendering or commit.
    pub fn target(&self, id: TargetId) -> Result<&T, TargetRingError> {
        self.slot(id)?
            .target
            .as_ref()
            .ok_or(TargetRingError::InvalidTarget)
    }

    /// Mutably borrows an acquired target for rendering.
    pub fn target_mut(&mut self, id: TargetId) -> Result<&mut T, TargetRingError> {
        let slot = self.slot_mut(id)?;
        if !matches!(slot.state, SlotState::Acquired(_)) {
            return Err(TargetRingError::InvalidState);
        }
        slot.target.as_mut().ok_or(TargetRingError::InvalidTarget)
    }

    /// Records backend acceptance and starts portable retirement books.
    pub fn mark_submitted(
        &mut self,
        id: TargetId,
        retain_until: RetainUntil,
    ) -> Result<(), TargetRingError> {
        let epoch = match self.slot(id)?.state {
            SlotState::Acquired(epoch) => epoch,
            _ => return Err(TargetRingError::InvalidState),
        };
        if self
            .last_submitted
            .is_some_and(|accepted| epoch <= accepted)
        {
            return Err(TargetRingError::StaleEpoch);
        }
        let slot = self.slot_mut(id)?;
        let SlotState::Acquired(epoch) = slot.state else {
            return Err(TargetRingError::InvalidState);
        };
        slot.state = SlotState::Submitted(SurfaceRetirement::new(epoch, retain_until));
        self.last_submitted = Some(epoch);
        Ok(())
    }

    /// Borrows the lifecycle books so a caller can feed bounded backend and
    /// presenter progress into them.
    pub fn retirement_mut(
        &mut self,
        id: TargetId,
    ) -> Result<&mut SurfaceRetirement, TargetRingError> {
        let slot = self.slot_mut(id)?;
        match &mut slot.state {
            SlotState::Submitted(retirement) => Ok(retirement),
            _ => Err(TargetRingError::InvalidState),
        }
    }

    /// Returns an acquired target that never crossed provider acceptance to
    /// the available or explicitly retired state.
    pub fn reject(&mut self, id: TargetId) -> Result<(), TargetRingError> {
        let slot = self.slot_mut(id)?;
        if !matches!(slot.state, SlotState::Acquired(_)) {
            return Err(TargetRingError::InvalidState);
        }
        slot.state = SlotState::Available;
        Ok(())
    }

    /// Recycles a fully retired target. An old-mode target remains explicitly
    /// retired until [`Self::take_retired`] returns its authority to the
    /// runtime.
    pub fn recycle(&mut self, id: TargetId) -> Result<(), TargetRingError> {
        let slot = self.slot_mut(id)?;
        match slot.state {
            SlotState::Submitted(retirement) if retirement.is_reusable() => {}
            SlotState::Submitted(_) => return Err(TargetRingError::Busy),
            _ => return Err(TargetRingError::InvalidState),
        }
        slot.state = SlotState::Available;
        Ok(())
    }

    /// Retires the current mode's targets. They remain installed but cannot be
    /// acquired again. The runtime retrieves each one with
    /// [`Self::take_retired`] only after it is available.
    pub fn begin_reconfigure(&mut self) {
        for slot in &mut self.slots {
            if slot.target.is_none() {
                continue;
            }
            slot.retire_on_reuse = true;
        }
    }

    /// Returns one old-mode target whose producer and presentation lifetimes
    /// are finished. This is the explicit platform destroy/release boundary.
    pub fn take_retired(&mut self) -> Option<(TargetId, T)> {
        let (index, slot) = self.slots.iter_mut().enumerate().find(|(_, slot)| {
            slot.target.is_some()
                && slot.retire_on_reuse
                && matches!(slot.state, SlotState::Available)
        })?;
        let id = TargetId {
            index,
            generation: slot.generation,
        };
        slot.retire_on_reuse = false;
        Some((id, slot.target.take().expect("matched installed target")))
    }

    /// Returns the number of currently installed platform targets.
    pub fn installed_len(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| slot.target.is_some())
            .count()
    }

    fn slot(&self, id: TargetId) -> Result<&TargetSlot<T>, TargetRingError> {
        let slot = self
            .slots
            .get(id.index)
            .ok_or(TargetRingError::InvalidTarget)?;
        (slot.generation == id.generation && slot.target.is_some())
            .then_some(slot)
            .ok_or(TargetRingError::InvalidTarget)
    }

    fn slot_mut(&mut self, id: TargetId) -> Result<&mut TargetSlot<T>, TargetRingError> {
        let slot = self
            .slots
            .get_mut(id.index)
            .ok_or(TargetRingError::InvalidTarget)?;
        (slot.generation == id.generation && slot.target.is_some())
            .then_some(slot)
            .ok_or(TargetRingError::InvalidTarget)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axgfx::{DeviceEpoch, SubmissionState, TargetReuse, VisibilityEpoch};
    use std::alloc::System;

    fn complete(retirement: &mut SurfaceRetirement, epoch: SurfaceEpoch) {
        let device = DeviceEpoch::new(epoch.get()).unwrap();
        let visibility = VisibilityEpoch::new(epoch.get()).unwrap();
        retirement
            .observe(SubmissionState::DeviceComplete(device))
            .unwrap();
        retirement
            .observe(SubmissionState::Visible { device, visibility })
            .unwrap();
        retirement.mark_committed().unwrap();
        retirement.mark_presented(epoch).unwrap();
        retirement.mark_submission_released().unwrap();
    }

    #[test]
    fn two_targets_are_not_reacquired_before_presentation() {
        let mut ring = TargetRing::with_capacity_in(2, System).unwrap();
        ring.install("first").unwrap();
        ring.install("second").unwrap();
        let epoch1 = SurfaceEpoch::new(1).unwrap();
        let epoch2 = SurfaceEpoch::new(2).unwrap();
        let first = ring.acquire(epoch1).unwrap().unwrap();
        ring.mark_submitted(first, RetainUntil::Visible).unwrap();
        let second = ring.acquire(epoch2).unwrap().unwrap();
        ring.mark_submitted(second, RetainUntil::Visible).unwrap();
        assert!(
            ring.acquire(SurfaceEpoch::new(3).unwrap())
                .unwrap()
                .is_none()
        );
        assert_eq!(*ring.target(first).unwrap(), "first");
        assert_eq!(*ring.target(second).unwrap(), "second");

        complete(ring.retirement_mut(first).unwrap(), epoch1);
        ring.recycle(first).unwrap();

        let reacquired = ring
            .acquire(SurfaceEpoch::new(3).unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(reacquired, first);
        assert_eq!(*ring.target(reacquired).unwrap(), "first");
    }

    #[test]
    fn reconfigure_keeps_old_in_flight_target_until_it_is_reusable() {
        let mut ring = TargetRing::with_capacity_in(2, System).unwrap();
        ring.install("idle old mode").unwrap();
        ring.install("in flight old mode").unwrap();
        let epoch = SurfaceEpoch::new(7).unwrap();
        let idle = ring
            .acquire(SurfaceEpoch::new(6).unwrap())
            .unwrap()
            .unwrap();
        ring.reject(idle).unwrap();
        let in_flight = ring.acquire(epoch).unwrap().unwrap();
        ring.mark_submitted(in_flight, RetainUntil::Presented)
            .unwrap();

        ring.begin_reconfigure();
        assert_eq!(ring.installed_len(), 2);
        assert_eq!(ring.take_retired().unwrap().1, "idle old mode");
        assert_eq!(ring.installed_len(), 1);
        let current = ring.install("new mode").unwrap();
        assert_eq!(ring.installed_len(), 2);
        let acquired = ring
            .acquire(SurfaceEpoch::new(8).unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(acquired, current);
        assert_eq!(*ring.target(acquired).unwrap(), "new mode");

        complete(ring.retirement_mut(in_flight).unwrap(), epoch);
        ring.recycle(in_flight).unwrap();
        assert_eq!(ring.installed_len(), 2);
        assert_eq!(ring.take_retired().unwrap().1, "in flight old mode");
        assert_eq!(ring.installed_len(), 1);
        assert_eq!(ring.target(in_flight), Err(TargetRingError::InvalidTarget));
    }

    #[test]
    fn unsafe_failed_target_never_becomes_reusable() {
        let mut ring = TargetRing::with_capacity_in(1, System).unwrap();
        ring.install(1_u8).unwrap();
        let target = ring
            .acquire(SurfaceEpoch::new(1).unwrap())
            .unwrap()
            .unwrap();
        ring.mark_submitted(target, RetainUntil::Visible).unwrap();
        let retirement = ring.retirement_mut(target).unwrap();
        retirement
            .observe(SubmissionState::Failed {
                error: axgfx::GraphicsError::DeviceLost,
                target: TargetReuse::Quarantine,
            })
            .unwrap();
        assert_eq!(ring.recycle(target), Err(TargetRingError::Busy));
    }

    #[test]
    fn accepted_epochs_are_strictly_monotonic_but_rejected_epochs_can_retry() {
        let mut ring = TargetRing::with_capacity_in(2, System).unwrap();
        ring.install(1_u8).unwrap();
        ring.install(2_u8).unwrap();
        let epoch1 = SurfaceEpoch::new(1).unwrap();
        let rejected = ring.acquire(epoch1).unwrap().unwrap();
        assert_eq!(
            ring.acquire(SurfaceEpoch::new(2).unwrap()),
            Err(TargetRingError::Busy)
        );
        ring.reject(rejected).unwrap();
        let retried = ring.acquire(epoch1).unwrap().unwrap();
        ring.mark_submitted(retried, RetainUntil::Visible).unwrap();

        assert_eq!(ring.acquire(epoch1), Err(TargetRingError::StaleEpoch));
    }
}
