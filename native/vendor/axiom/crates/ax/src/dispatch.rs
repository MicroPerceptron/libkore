//! Cross-graph admission over backend queue-capacity snapshots.
//!
//! Graph placement answers where one graph should execute in isolation.
//! Interactive systems need one more decision: which already-compiled graph
//! variant should claim which available backend lane now. This module keeps
//! that policy in `ax` without importing a host runtime, clock, thread, or
//! provider event type.
//!
//! Times are caller-defined absolute ticks. A host adapter can use
//! nanoseconds; a kore service can use scheduler quanta. Durations should come
//! from measured warm executions or a calibrated graph cost model. Multiple
//! lanes with the same backend ID express provider or SPMD capacity without
//! pretending that a backend is either wholly idle or wholly occupied.

/// One independently reservable execution lane in a backend domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DispatchLane {
    pub backend: u16,
    /// Earliest absolute tick at which this lane can accept more work.
    pub ready_at: u64,
}

impl DispatchLane {
    pub const fn new(backend: u16, ready_at: u64) -> Self {
        Self { backend, ready_at }
    }
}

/// One compiled implementation or quality tier eligible for a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DispatchCandidate {
    /// Opaque caller-owned identifier for the executable/quality variant.
    pub variant: u16,
    pub backend: u16,
    /// Estimated warm execution time in the request clock's ticks.
    pub duration: u64,
    /// Higher values retain more fidelity. Equal values mean that only
    /// latency and backend capacity distinguish the candidates.
    pub fidelity: u16,
}

impl DispatchCandidate {
    pub const fn new(variant: u16, backend: u16, duration: u64, fidelity: u16) -> Self {
        Self {
            variant,
            backend,
            duration,
            fidelity,
        }
    }
}

/// Timing and minimum-quality contract for one graph invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DispatchRequest {
    /// The invocation and its inputs cannot be consumed before this tick.
    pub ready_at: u64,
    /// Desired completion tick. Use `u64::MAX` for throughput-only work.
    pub deadline: u64,
    /// Candidates below this fidelity are ineligible even if faster.
    pub minimum_fidelity: u16,
}

impl DispatchRequest {
    pub const fn new(ready_at: u64, deadline: u64, minimum_fidelity: u16) -> Self {
        Self {
            ready_at,
            deadline,
            minimum_fidelity,
        }
    }
}

/// Selected executable and the capacity reservation it requires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DispatchDecision {
    pub candidate_index: usize,
    pub lane_index: usize,
    pub variant: u16,
    pub backend: u16,
    pub fidelity: u16,
    pub start_at: u64,
    pub finish_at: u64,
    pub deadline_met: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionError {
    NoLanes,
    NoCandidates,
    NoEligibleCandidate,
    InvalidLane,
    ReservationNotTail,
}

/// Allocation-free admission planner over caller-owned lane state.
pub struct AdmissionPlanner<'a> {
    lanes: &'a mut [DispatchLane],
}

impl<'a> AdmissionPlanner<'a> {
    pub fn new(lanes: &'a mut [DispatchLane]) -> Result<Self, AdmissionError> {
        if lanes.is_empty() {
            return Err(AdmissionError::NoLanes);
        }
        Ok(Self { lanes })
    }

    pub fn lanes(&self) -> &[DispatchLane] {
        self.lanes
    }

    /// Select without reserving capacity.
    ///
    /// An on-time candidate always beats a late one. Among on-time choices,
    /// fidelity wins first and finish time second. If every choice is late,
    /// earliest completion wins before fidelity so overload degrades
    /// gracefully instead of preserving quality while latency runs away.
    pub fn select(
        &self,
        request: DispatchRequest,
        candidates: &[DispatchCandidate],
    ) -> Result<DispatchDecision, AdmissionError> {
        if candidates.is_empty() {
            return Err(AdmissionError::NoCandidates);
        }
        let mut best = None;
        for (candidate_index, candidate) in candidates.iter().enumerate() {
            if candidate.fidelity < request.minimum_fidelity {
                continue;
            }
            for (lane_index, lane) in self.lanes.iter().enumerate() {
                if lane.backend != candidate.backend {
                    continue;
                }
                let start_at = request.ready_at.max(lane.ready_at);
                let finish_at = start_at.saturating_add(candidate.duration);
                let decision = DispatchDecision {
                    candidate_index,
                    lane_index,
                    variant: candidate.variant,
                    backend: candidate.backend,
                    fidelity: candidate.fidelity,
                    start_at,
                    finish_at,
                    deadline_met: finish_at <= request.deadline,
                };
                if best.is_none_or(|current| better(decision, current)) {
                    best = Some(decision);
                }
            }
        }
        best.ok_or(AdmissionError::NoEligibleCandidate)
    }

    /// Select and reserve the chosen lane through its estimated finish.
    pub fn admit(
        &mut self,
        request: DispatchRequest,
        candidates: &[DispatchCandidate],
    ) -> Result<DispatchDecision, AdmissionError> {
        let decision = self.select(request, candidates)?;
        self.lanes[decision.lane_index].ready_at = decision.finish_at;
        Ok(decision)
    }

    /// Reconcile an estimated reservation with a provider event or scheduler
    /// completion observation. This is the feedback seam for host adapters
    /// and kore; `ax` does not own the clock or poll the event.
    pub fn reconcile(
        &mut self,
        reservation: DispatchDecision,
        ready_at: u64,
    ) -> Result<(), AdmissionError> {
        let lane = self
            .lanes
            .get_mut(reservation.lane_index)
            .ok_or(AdmissionError::InvalidLane)?;
        if lane.backend != reservation.backend {
            return Err(AdmissionError::InvalidLane);
        }
        // An older completion cannot overwrite a lane that has since
        // accepted more work. A deeper adapter queue must reconcile its own
        // downstream reservations before presenting a new tail to `ax`.
        if lane.ready_at != reservation.finish_at {
            return Err(AdmissionError::ReservationNotTail);
        }
        lane.ready_at = ready_at;
        Ok(())
    }
}

fn better(candidate: DispatchDecision, current: DispatchDecision) -> bool {
    match (candidate.deadline_met, current.deadline_met) {
        (true, false) => true,
        (false, true) => false,
        (true, true) => {
            (candidate.fidelity, core::cmp::Reverse(candidate.finish_at))
                > (current.fidelity, core::cmp::Reverse(current.finish_at))
        }
        (false, false) => {
            (core::cmp::Reverse(candidate.finish_at), candidate.fidelity)
                > (core::cmp::Reverse(current.finish_at), current.fidelity)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CPU: u16 = 0;
    const ACCEL: u16 = 1;
    const NATIVE: u16 = 0;
    const INTERACTIVE: u16 = 1;

    fn frame_candidates() -> [DispatchCandidate; 3] {
        [
            DispatchCandidate::new(NATIVE, CPU, 47, 100),
            DispatchCandidate::new(INTERACTIVE, CPU, 20, 80),
            DispatchCandidate::new(NATIVE, ACCEL, 9, 100),
        ]
    }

    #[test]
    fn free_accelerator_keeps_native_fidelity() {
        let mut lanes = [DispatchLane::new(CPU, 0), DispatchLane::new(ACCEL, 0)];
        let mut planner = AdmissionPlanner::new(&mut lanes).unwrap();
        let decision = planner
            .admit(DispatchRequest::new(10, 43, 0), &frame_candidates())
            .unwrap();
        assert_eq!(decision.variant, NATIVE);
        assert_eq!(decision.backend, ACCEL);
        assert_eq!(decision.finish_at, 19);
        assert!(decision.deadline_met);
        assert_eq!(planner.lanes()[1].ready_at, 19);
    }

    #[test]
    fn busy_inference_accelerator_selects_cpu_interaction_tier() {
        let mut lanes = [
            DispatchLane::new(CPU, 0),
            // A resident inference submission already owns the accelerator.
            DispatchLane::new(ACCEL, 100),
        ];
        let mut planner = AdmissionPlanner::new(&mut lanes).unwrap();
        let decision = planner
            .admit(DispatchRequest::new(10, 43, 0), &frame_candidates())
            .unwrap();
        assert_eq!(decision.variant, INTERACTIVE);
        assert_eq!(decision.backend, CPU);
        assert_eq!(decision.finish_at, 30);
        assert!(decision.deadline_met);
        assert_eq!(planner.lanes()[0].ready_at, 30);
        assert_eq!(planner.lanes()[1].ready_at, 100);
    }

    #[test]
    fn calibrated_native_cpu_frame_meets_deadline_beside_resident_inference() {
        let mut lanes = [
            DispatchLane::new(CPU, 0),
            // Local inference retains accelerator residency beyond the UI
            // frame's deadline.
            DispatchLane::new(ACCEL, 100),
        ];
        let mut planner = AdmissionPlanner::new(&mut lanes).unwrap();
        let native_candidates = [
            // Conservative whole-millisecond calibration of the portable
            // CPU provider's native 1265x795 retained graph.
            DispatchCandidate::new(NATIVE, CPU, 16, 100),
            DispatchCandidate::new(NATIVE, ACCEL, 9, 100),
        ];
        let decision = planner
            .admit(DispatchRequest::new(10, 43, 100), &native_candidates)
            .unwrap();
        assert_eq!(decision.variant, NATIVE);
        assert_eq!(decision.backend, CPU);
        assert_eq!(decision.finish_at, 26);
        assert!(decision.deadline_met);
        assert_eq!(planner.lanes()[0].ready_at, 26);
        assert_eq!(planner.lanes()[1].ready_at, 100);
    }

    #[test]
    fn overload_chooses_earliest_completion_when_all_tiers_miss() {
        let mut lanes = [DispatchLane::new(CPU, 0), DispatchLane::new(ACCEL, 100)];
        let planner = AdmissionPlanner::new(&mut lanes).unwrap();
        let decision = planner
            .select(DispatchRequest::new(10, 25, 0), &frame_candidates())
            .unwrap();
        assert_eq!(decision.variant, INTERACTIVE);
        assert_eq!(decision.finish_at, 30);
        assert!(!decision.deadline_met);
    }

    #[test]
    fn fidelity_floor_can_forbid_the_interaction_tier() {
        let mut lanes = [DispatchLane::new(CPU, 0), DispatchLane::new(ACCEL, 100)];
        let planner = AdmissionPlanner::new(&mut lanes).unwrap();
        let decision = planner
            .select(DispatchRequest::new(10, 43, 100), &frame_candidates())
            .unwrap();
        assert_eq!(decision.variant, NATIVE);
        assert_eq!(decision.backend, CPU);
        assert_eq!(decision.finish_at, 57);
        assert!(!decision.deadline_met);
    }

    #[test]
    fn duplicate_backend_lanes_express_real_capacity() {
        let mut lanes = [DispatchLane::new(CPU, 80), DispatchLane::new(CPU, 4)];
        let mut planner = AdmissionPlanner::new(&mut lanes).unwrap();
        let candidate = [DispatchCandidate::new(NATIVE, CPU, 12, 100)];
        let decision = planner
            .admit(DispatchRequest::new(10, 40, 0), &candidate)
            .unwrap();
        assert_eq!(decision.lane_index, 1);
        assert_eq!(decision.start_at, 10);
        assert_eq!(decision.finish_at, 22);
    }

    #[test]
    fn provider_feedback_reconciles_estimates_without_a_host_clock_dependency() {
        let mut lanes = [DispatchLane::new(CPU, 0)];
        let mut planner = AdmissionPlanner::new(&mut lanes).unwrap();
        let candidate = [DispatchCandidate::new(NATIVE, CPU, 20, 100)];
        let reservation = planner
            .admit(DispatchRequest::new(0, 30, 0), &candidate)
            .unwrap();
        assert_eq!(planner.lanes()[0].ready_at, 20);
        planner.reconcile(reservation, 17).unwrap();
        assert_eq!(planner.lanes()[0].ready_at, 17);
        let mut invalid = reservation;
        invalid.lane_index = 1;
        assert_eq!(
            planner.reconcile(invalid, 0),
            Err(AdmissionError::InvalidLane)
        );
    }

    #[test]
    fn an_older_completion_cannot_erase_a_newer_lane_reservation() {
        let mut lanes = [DispatchLane::new(CPU, 0)];
        let mut planner = AdmissionPlanner::new(&mut lanes).unwrap();
        let candidate = [DispatchCandidate::new(NATIVE, CPU, 20, 100)];
        let first = planner
            .admit(DispatchRequest::new(0, 30, 0), &candidate)
            .unwrap();
        planner
            .admit(DispatchRequest::new(20, 60, 0), &candidate)
            .unwrap();
        assert_eq!(planner.lanes()[0].ready_at, 40);
        assert_eq!(
            planner.reconcile(first, 17),
            Err(AdmissionError::ReservationNotTail)
        );
        assert_eq!(planner.lanes()[0].ready_at, 40);
    }
}
