//! Surviving local owner's transport retirement transaction. Native task reap
//! precedes Close; client reply drains precede Open. This state never claims
//! that an application is running merely because its routes are available.
use crate::{
    Endpoint, HOSTLINK_SUPERVISOR_WORKERS, HostLinkService, MailboxDelivery, ServiceError,
    SupervisorOperation, SupervisorPhase, SupervisorState,
};
use kore_hostlink::{HostLinkTransport, TaskGeneration};

pub struct SupervisorOwner {
    state: SupervisorState,
    routes_closed: bool,
}

impl SupervisorOwner {
    pub fn new(generation: TaskGeneration) -> Self {
        Self {
            state: SupervisorState {
                phase: SupervisorPhase::Available,
                incarnation: generation,
                workers: [generation; HOSTLINK_SUPERVISOR_WORKERS],
            },
            routes_closed: false,
        }
    }

    pub fn state(&self) -> SupervisorState {
        self.state
    }

    pub fn client_active(&self) -> bool {
        self.state.phase == SupervisorPhase::Available
    }

    pub fn needs_request_drain(&self) -> bool {
        self.routes_closed && self.state.phase == SupervisorPhase::Closing
    }

    /// Called by the native serving owner only after all old request rings
    /// drained without resetting their live cursors.
    pub fn confirm_request_drain(&mut self) -> bool {
        if !self.needs_request_drain() {
            return false;
        }
        self.state.phase = SupervisorPhase::Closed;
        true
    }

    pub fn operate<T: HostLinkTransport, M: MailboxDelivery>(
        &mut self,
        core: &mut HostLinkService<T, M>,
        operation: SupervisorOperation,
        generation: u64,
    ) -> Result<SupervisorState, ServiceError> {
        if operation == SupervisorOperation::Observe {
            if generation != 0 {
                return Err(ServiceError::Malformed);
            }
            if self.client_active() {
                for (slot, value) in self.state.workers.iter_mut().enumerate() {
                    *value = core
                        .generation(Endpoint::Agent(slot as u8))
                        .ok_or(ServiceError::Route)?;
                }
                if core.supervisor_restart_requested(self.state.incarnation) {
                    self.state.phase = SupervisorPhase::RestartRequested;
                }
            }
            return Ok(self.state);
        }
        match operation {
            SupervisorOperation::Close => {
                if generation != self.state.incarnation.get()
                    || self.state.phase == SupervisorPhase::Opening
                {
                    return Err(ServiceError::Route);
                }
                if self.state.phase == SupervisorPhase::Closed {
                    return Ok(self.state);
                }
                if matches!(
                    self.state.phase,
                    SupervisorPhase::Available | SupervisorPhase::RestartRequested
                ) {
                    for (slot, value) in self.state.workers.iter_mut().enumerate() {
                        *value = core
                            .generation(Endpoint::Agent(slot as u8))
                            .ok_or(ServiceError::Route)?;
                    }
                    self.state.phase = SupervisorPhase::Closing;
                }
                for (slot, generation) in self.state.workers.iter().enumerate() {
                    core.close_agent_route(slot as u8, *generation)?;
                }
                core.close_broker_route(self.state.incarnation)?;
                self.routes_closed = true;
                Ok(self.state)
            }
            SupervisorOperation::Open => {
                if self.client_active() && generation == self.state.incarnation.get() {
                    return Ok(self.state);
                }
                if self.state.phase == SupervisorPhase::Closed {
                    if self.state.incarnation.get().checked_add(1) != Some(generation)
                        || self.state.workers.iter().any(|g| g.get() == u64::MAX)
                    {
                        return Err(ServiceError::Route);
                    }
                    self.state.phase = SupervisorPhase::Opening;
                } else if self.state.phase != SupervisorPhase::Opening {
                    return Err(ServiceError::Route);
                }
                if self.state.incarnation.get().checked_add(1) != Some(generation) {
                    return Err(ServiceError::Route);
                }
                let next = TaskGeneration::new(generation).ok_or(ServiceError::Route)?;
                if core.generation(Endpoint::Broker) != Some(next) {
                    core.rotate_broker_route(next)?;
                }
                for (slot, old) in self.state.workers.iter().enumerate() {
                    let next =
                        TaskGeneration::new(old.get().checked_add(1).ok_or(ServiceError::Route)?)
                            .ok_or(ServiceError::Route)?;
                    if core.generation(Endpoint::Agent(slot as u8)) != Some(next) {
                        core.rotate_agent_route(slot as u8, next)?;
                    }
                }
                self.state.incarnation = next;
                for generation in &mut self.state.workers {
                    *generation = TaskGeneration::new(generation.get() + 1).unwrap();
                }
                self.state.phase = SupervisorPhase::Available;
                self.routes_closed = false;
                Ok(self.state)
            }
            SupervisorOperation::Observe => unreachable!(),
        }
    }
}
