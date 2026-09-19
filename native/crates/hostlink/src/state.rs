//! Per-route bounded sequencing, request admission, and credits.

use heapless::Vec;

use crate::{EnvelopeV1, LinkEpoch, RequestId, RouteId, TaskGeneration};

pub const REQUESTS_PER_ROUTE_MAX: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouteRole {
    Broker,
    Agent,
    Importer,
    Exporter,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouteLifecycle {
    Open,
    Closed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Direction {
    Receive,
    Send,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SequenceDisposition {
    Next,
    Duplicate,
    OutOfOrder,
    Exhausted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouteError {
    EpochStale,
    RouteUnknown,
    GenerationStale,
    LinkControlOnRoute,
    RouteClosed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestError {
    Zero,
    Duplicate,
    CapacityExhausted,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CreditError {
    ArithmeticOverflow,
    AboveNegotiatedMaximum,
    Backpressure,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmitError {
    Route(RouteError),
    Sequence(SequenceDisposition),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RequestEntry {
    id: RequestId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Credits {
    available: u64,
    maximum: u64,
}

impl Credits {
    const fn new(maximum: u64) -> Self {
        Self {
            available: maximum,
            maximum,
        }
    }

    fn consume(&mut self, amount: u64) -> Result<(), CreditError> {
        if amount > self.available {
            return Err(CreditError::Backpressure);
        }
        self.available -= amount;
        Ok(())
    }

    fn replenish(&mut self, amount: u64) -> Result<(), CreditError> {
        let next = self
            .available
            .checked_add(amount)
            .ok_or(CreditError::ArithmeticOverflow)?;
        if next > self.maximum {
            return Err(CreditError::AboveNegotiatedMaximum);
        }
        self.available = next;
        Ok(())
    }
}

/// State owned by exactly one capability-established route.
pub struct RouteState {
    epoch: LinkEpoch,
    route: RouteId,
    generation: TaskGeneration,
    role: RouteRole,
    lifecycle: RouteLifecycle,
    recv_next: u64,
    send_next: u64,
    requests: Vec<RequestEntry, REQUESTS_PER_ROUTE_MAX>,
    control: Credits,
    bulk_bytes: Credits,
    bulk_chunks: Credits,
}

impl RouteState {
    #[must_use]
    pub fn new(
        epoch: LinkEpoch,
        route: RouteId,
        generation: TaskGeneration,
        role: RouteRole,
        control_credits: u64,
        bulk_byte_credits: u64,
        bulk_chunk_credits: u64,
    ) -> Option<Self> {
        if epoch.is_zero() || route.is_zero() {
            return None;
        }
        Some(Self {
            epoch,
            route,
            generation,
            role,
            lifecycle: RouteLifecycle::Open,
            recv_next: 0,
            send_next: 0,
            requests: Vec::new(),
            control: Credits::new(control_credits),
            bulk_bytes: Credits::new(bulk_byte_credits),
            bulk_chunks: Credits::new(bulk_chunk_credits),
        })
    }

    #[must_use]
    pub const fn role(&self) -> RouteRole {
        self.role
    }

    #[must_use]
    pub const fn route_id(&self) -> RouteId {
        self.route
    }

    #[must_use]
    pub const fn generation(&self) -> TaskGeneration {
        self.generation
    }

    #[must_use]
    pub const fn lifecycle(&self) -> RouteLifecycle {
        self.lifecycle
    }

    /// Retire the route and return every request that must be completed with
    /// `HostDisconnected` (or the caller's more specific terminal reason).
    pub fn disconnect(&mut self) -> Vec<RequestId, REQUESTS_PER_ROUTE_MAX> {
        self.lifecycle = RouteLifecycle::Closed;
        core::mem::take(&mut self.requests)
            .into_iter()
            .map(|entry| entry.id)
            .collect()
    }

    /// Validate trusted routing fields and sequence before parsing payload.
    /// This does not advance the sequence window.
    pub fn check_received(&self, envelope: &EnvelopeV1) -> Result<(), AdmitError> {
        self.validate_route(envelope).map_err(AdmitError::Route)?;
        match classify_sequence(self.recv_next, envelope.sequence.get()) {
            SequenceDisposition::Next => Ok(()),
            other => Err(AdmitError::Sequence(other)),
        }
    }

    /// Advance after the caller has validated and durably admitted the
    /// payload to bounded route-owned state.
    pub fn commit_received(&mut self, envelope: &EnvelopeV1) -> Result<(), AdmitError> {
        self.check_received(envelope)?;
        self.recv_next = self
            .recv_next
            .checked_add(1)
            .ok_or(AdmitError::Sequence(SequenceDisposition::Exhausted))?;
        Ok(())
    }

    pub fn next_send_sequence(&mut self) -> Result<u64, SequenceDisposition> {
        let current = self.send_sequence()?;
        self.commit_sent(current)?;
        Ok(current)
    }

    /// Inspect the next sequence before attempting a fallible transport send.
    pub fn send_sequence(&self) -> Result<u64, SequenceDisposition> {
        if self.send_next == u64::MAX {
            Err(SequenceDisposition::Exhausted)
        } else {
            Ok(self.send_next)
        }
    }

    /// Advance only after the transport has atomically accepted the frame.
    pub fn commit_sent(&mut self, sequence: u64) -> Result<(), SequenceDisposition> {
        if sequence != self.send_next {
            return Err(if sequence < self.send_next {
                SequenceDisposition::Duplicate
            } else {
                SequenceDisposition::OutOfOrder
            });
        }
        self.send_next = self
            .send_next
            .checked_add(1)
            .ok_or(SequenceDisposition::Exhausted)?;
        Ok(())
    }

    pub fn open_request(&mut self, id: RequestId) -> Result<(), RequestError> {
        if id.is_zero() {
            return Err(RequestError::Zero);
        }
        if self.requests.iter().any(|entry| entry.id == id) {
            return Err(RequestError::Duplicate);
        }
        self.requests
            .push(RequestEntry { id })
            .map_err(|_| RequestError::CapacityExhausted)
    }

    pub fn close_request(&mut self, id: RequestId) -> Result<(), RequestError> {
        let index = self
            .requests
            .iter()
            .position(|entry| entry.id == id)
            .ok_or(RequestError::Unknown)?;
        self.requests.swap_remove(index);
        Ok(())
    }

    /// Test whether an exact request is still live on this route. Host-side
    /// backend events use this before forwarding a streamed part so a late
    /// worker response cannot attach to a terminal or replacement request.
    #[must_use]
    pub fn has_request(&self, id: RequestId) -> bool {
        self.requests.iter().any(|entry| entry.id == id)
    }

    pub fn consume_control(&mut self, frames: u64) -> Result<(), CreditError> {
        self.control.consume(frames)
    }

    pub fn replenish_control(&mut self, frames: u64) -> Result<(), CreditError> {
        self.control.replenish(frames)
    }

    pub fn consume_bulk(&mut self, bytes: u64, chunks: u64) -> Result<(), CreditError> {
        self.bulk_bytes.consume(bytes)?;
        if let Err(error) = self.bulk_chunks.consume(chunks) {
            // Roll back the first reservation; this cannot exceed the maximum
            // because it restores credits consumed in this operation.
            self.bulk_bytes.available += bytes;
            return Err(error);
        }
        Ok(())
    }

    pub fn replenish_bulk(&mut self, bytes: u64, chunks: u64) -> Result<(), CreditError> {
        let next_bytes = self
            .bulk_bytes
            .available
            .checked_add(bytes)
            .ok_or(CreditError::ArithmeticOverflow)?;
        let next_chunks = self
            .bulk_chunks
            .available
            .checked_add(chunks)
            .ok_or(CreditError::ArithmeticOverflow)?;
        if next_bytes > self.bulk_bytes.maximum || next_chunks > self.bulk_chunks.maximum {
            return Err(CreditError::AboveNegotiatedMaximum);
        }
        self.bulk_bytes.available = next_bytes;
        self.bulk_chunks.available = next_chunks;
        Ok(())
    }

    fn validate_route(&self, envelope: &EnvelopeV1) -> Result<(), RouteError> {
        if self.lifecycle != RouteLifecycle::Open {
            return Err(RouteError::RouteClosed);
        }
        if envelope.link_epoch != self.epoch {
            return Err(RouteError::EpochStale);
        }
        if envelope.route_id != self.route {
            return Err(RouteError::RouteUnknown);
        }
        if envelope.task_generation.get() != self.generation.get() {
            return Err(RouteError::GenerationStale);
        }
        Ok(())
    }
}

#[must_use]
fn classify_sequence(next: u64, received: u64) -> SequenceDisposition {
    if next == u64::MAX {
        SequenceDisposition::Exhausted
    } else if received == next {
        SequenceDisposition::Next
    } else if received < next {
        SequenceDisposition::Duplicate
    } else {
        SequenceDisposition::OutOfOrder
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EnvelopeFields, FAMILY_LINK, RequestId};

    fn id(byte: u8) -> [u8; 16] {
        [byte; 16]
    }

    fn route() -> RouteState {
        RouteState::new(
            LinkEpoch(id(1)),
            RouteId(id(2)),
            TaskGeneration::new(7).unwrap(),
            RouteRole::Agent,
            2,
            8,
            2,
        )
        .unwrap()
    }

    fn envelope(sequence: u64) -> EnvelopeV1 {
        EnvelopeV1::new(EnvelopeFields {
            family: FAMILY_LINK,
            schema_minor: 0,
            kind: 1,
            epoch: LinkEpoch(id(1)),
            route: RouteId(id(2)),
            generation: 7,
            sequence,
            request: RequestId(id(3)),
            payload_len: 0,
        })
    }

    #[test]
    fn exact_sequence_only_and_replay_has_no_effect() {
        let mut state = route();
        assert_eq!(state.commit_received(&envelope(0)), Ok(()));
        assert_eq!(
            state.commit_received(&envelope(0)),
            Err(AdmitError::Sequence(SequenceDisposition::Duplicate))
        );
        assert_eq!(
            state.commit_received(&envelope(2)),
            Err(AdmitError::Sequence(SequenceDisposition::OutOfOrder))
        );
        assert_eq!(state.commit_received(&envelope(1)), Ok(()));
    }

    #[test]
    fn routing_identity_is_checked_before_sequence() {
        let mut state = route();
        let mut forged = envelope(9);
        forged.task_generation.set(8);
        assert_eq!(
            state.commit_received(&forged),
            Err(AdmitError::Route(RouteError::GenerationStale))
        );
        assert_eq!(state.commit_received(&envelope(0)), Ok(()));
    }

    #[test]
    fn rejected_payload_does_not_consume_its_sequence() {
        let mut state = route();
        assert_eq!(state.check_received(&envelope(0)), Ok(()));
        // Payload validation rejects here, so there is no commit.
        assert_eq!(state.check_received(&envelope(0)), Ok(()));
        assert_eq!(state.commit_received(&envelope(0)), Ok(()));
    }

    #[test]
    fn request_table_is_bounded_and_deduplicated() {
        let mut state = route();
        for byte in 1..=REQUESTS_PER_ROUTE_MAX as u8 {
            assert_eq!(state.open_request(RequestId(id(byte))), Ok(()));
        }
        assert_eq!(
            state.open_request(RequestId(id(1))),
            Err(RequestError::Duplicate)
        );
        assert_eq!(
            state.open_request(RequestId(id(42))),
            Err(RequestError::CapacityExhausted)
        );
        assert_eq!(state.close_request(RequestId(id(4))), Ok(()));
        assert_eq!(state.open_request(RequestId(id(42))), Ok(()));
    }

    #[test]
    fn bulk_credit_reservation_is_atomic() {
        let mut state = route();
        assert_eq!(state.consume_bulk(4, 3), Err(CreditError::Backpressure));
        assert_eq!(state.consume_bulk(8, 2), Ok(()));
        assert_eq!(state.consume_bulk(1, 1), Err(CreditError::Backpressure));
        assert_eq!(state.replenish_bulk(8, 2), Ok(()));
    }

    #[test]
    fn disconnect_retires_requests_and_rejects_late_replies() {
        let mut state = route();
        state.open_request(RequestId(id(4))).unwrap();
        state.open_request(RequestId(id(5))).unwrap();
        let retired = state.disconnect();
        assert_eq!(retired.as_slice(), &[RequestId(id(4)), RequestId(id(5))]);
        assert_eq!(state.lifecycle(), RouteLifecycle::Closed);
        assert_eq!(
            state.check_received(&envelope(0)),
            Err(AdmitError::Route(RouteError::RouteClosed))
        );
    }
}
