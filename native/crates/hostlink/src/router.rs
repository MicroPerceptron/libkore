//! Capability-established route binding.

use heapless::Vec;

use crate::{AdmitError, EnvelopeV1, RouteError, RouteId, RouteState};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BindError {
    DuplicateRoute,
    CapacityExhausted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessError<E> {
    Admit(AdmitError),
    Rejected(E),
}

/// One route and the endpoint authority established for it by the guest.
///
/// `M` is deliberately unconstrained.  When it is an affine mailbox
/// capability, this type remains affine: a wire route id can select this row,
/// but it can never manufacture or replace its mailbox.
pub struct BoundRoute<M> {
    state: RouteState,
    mailbox: M,
}

impl<M> BoundRoute<M> {
    #[must_use]
    pub const fn state(&self) -> &RouteState {
        &self.state
    }

    #[must_use]
    pub fn state_mut(&mut self) -> &mut RouteState {
        &mut self.state
    }

    #[must_use]
    pub const fn mailbox(&self) -> &M {
        &self.mailbox
    }

    #[must_use]
    pub fn mailbox_mut(&mut self) -> &mut M {
        &mut self.mailbox
    }

    #[must_use]
    pub fn into_parts(self) -> (RouteState, M) {
        (self.state, self.mailbox)
    }
}

/// A caller-owned, bounded table of capability-established routes.
pub struct RouteTable<M, const N: usize> {
    routes: Vec<BoundRoute<M>, N>,
}

impl<M, const N: usize> Default for RouteTable<M, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M, const N: usize> RouteTable<M, N> {
    #[must_use]
    pub const fn new() -> Self {
        Self { routes: Vec::new() }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.routes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.routes.is_empty()
    }

    /// Resolve a trusted local route identity to its structurally bound row.
    /// Wire callers still use `process_received`, which validates epoch,
    /// generation, and sequence before exposing the mailbox.
    pub fn bound_mut(&mut self, route: RouteId) -> Option<&mut BoundRoute<M>> {
        self.routes
            .iter_mut()
            .find(|bound| bound.state.route_id() == route)
    }

    /// Bind an already-created route state to an already-held mailbox.
    pub fn bind(&mut self, state: RouteState, mailbox: M) -> Result<(), (BindError, M)> {
        if self
            .routes
            .iter()
            .any(|bound| bound.state.route_id() == state.route_id())
        {
            return Err((BindError::DuplicateRoute, mailbox));
        }
        self.routes
            .push(BoundRoute { state, mailbox })
            .map_err(|bound| (BindError::CapacityExhausted, bound.mailbox))
    }

    /// Resolve, validate, and process a frame against its mailbox-bound row.
    /// The receive sequence advances only when `process` succeeds.
    pub fn process_received<R, E>(
        &mut self,
        envelope: &EnvelopeV1,
        process: impl FnOnce(&mut BoundRoute<M>) -> Result<R, E>,
    ) -> Result<R, ProcessError<E>> {
        let bound = self
            .routes
            .iter_mut()
            .find(|bound| bound.state.route_id() == envelope.route_id)
            .ok_or(ProcessError::Admit(AdmitError::Route(
                RouteError::RouteUnknown,
            )))?;
        bound
            .state
            .check_received(envelope)
            .map_err(ProcessError::Admit)?;
        let result = process(bound).map_err(ProcessError::Rejected)?;
        bound
            .state
            .commit_received(envelope)
            .map_err(ProcessError::Admit)?;
        Ok(result)
    }

    /// Remove a route and return its mailbox to the service for deterministic
    /// terminal notification/revocation.
    pub fn unbind(&mut self, route: RouteId) -> Option<BoundRoute<M>> {
        let index = self
            .routes
            .iter()
            .position(|bound| bound.state.route_id() == route)?;
        Some(self.routes.swap_remove(index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EnvelopeFields, FAMILY_LINK, LinkEpoch, RequestId, RouteRole, TaskGeneration};

    fn id(byte: u8) -> [u8; 16] {
        [byte; 16]
    }

    fn state(route: u8, generation: u64) -> RouteState {
        RouteState::new(
            LinkEpoch(id(1)),
            RouteId(id(route)),
            TaskGeneration::new(generation).unwrap(),
            RouteRole::Agent,
            2,
            8,
            2,
        )
        .unwrap()
    }

    fn envelope(route: u8, generation: u64) -> EnvelopeV1 {
        EnvelopeV1::new(EnvelopeFields {
            family: FAMILY_LINK,
            schema_minor: 0,
            kind: 1,
            epoch: LinkEpoch(id(1)),
            route: RouteId(id(route)),
            generation,
            sequence: 0,
            request: RequestId(id(9)),
            payload_len: 0,
        })
    }

    #[test]
    fn wire_route_can_only_reach_its_prebound_mailbox() {
        let mut routes = RouteTable::<u8, 2>::new();
        routes.bind(state(2, 10), 42).unwrap();
        routes.bind(state(3, 11), 77).unwrap();

        assert_eq!(
            routes
                .process_received(&envelope(2, 10), |bound| Ok::<_, ()>(*bound.mailbox()))
                .unwrap(),
            42
        );
        assert_eq!(
            routes
                .process_received(&envelope(3, 11), |bound| Ok::<_, ()>(*bound.mailbox()))
                .unwrap(),
            77
        );

        assert!(matches!(
            routes.process_received(&envelope(2, 11), |_| Ok::<_, ()>(())),
            Err(ProcessError::Admit(AdmitError::Route(
                RouteError::GenerationStale
            )))
        ));
        assert!(matches!(
            routes.process_received(&envelope(4, 10), |_| Ok::<_, ()>(())),
            Err(ProcessError::Admit(AdmitError::Route(
                RouteError::RouteUnknown
            )))
        ));
    }

    #[test]
    fn payload_rejection_does_not_advance_route_sequence() {
        let mut routes = RouteTable::<u8, 1>::new();
        routes.bind(state(2, 10), 42).unwrap();
        assert_eq!(
            routes.process_received(&envelope(2, 10), |_| Err::<(), _>("malformed")),
            Err(ProcessError::Rejected("malformed"))
        );
        assert_eq!(
            routes.process_received(&envelope(2, 10), |_| Ok::<_, ()>(())),
            Ok(())
        );
    }
}
