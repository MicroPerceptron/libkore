//! One retained import, with immutable request data and explicit custody recovery.
use super::{ImportReply, ImportTarget, RingAccel};
use crate::service::AccelRefusal;
use kore_rt::{
    cap::StashedMemory,
    svc::{RetainedReturnCallError, ReturnCapability},
    transport::Transport,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportOutcome {
    Imported {
        buffer: u64,
    },
    Returned(AccelRefusal),
    /// The destination retained custody for cleanup or containment.
    Consumed(AccelRefusal),
}

struct Custody<C, R> {
    source: Option<C>,
    returned: Option<R>,
    outcome: Option<ImportOutcome>,
}
impl<C, R> Custody<C, R> {
    fn recover(&mut self, idle: bool) -> Option<C> {
        if idle && self.outcome.is_none() {
            self.source.take()
        } else {
            None
        }
    }
    fn returned(&mut self, refusal: AccelRefusal, owner: R) -> ImportOutcome {
        self.returned = Some(owner);
        let outcome = ImportOutcome::Returned(refusal);
        self.outcome = Some(outcome);
        outcome
    }
    fn take_returned(&mut self) -> Option<R> {
        if matches!(self.outcome, Some(ImportOutcome::Returned(_))) {
            self.returned.take()
        } else {
            None
        }
    }
}

/// Own the source outside cancellable futures and borrow its destination face.
/// Retry advance after uncertainty: it resumes the same immutable request.
/// A known disposition is cached, and a returned owner can be taken once.
///
/// Stop using local aliases before publication. Dropping this continuation
/// retires only custody still held locally; it does not cancel an outstanding
/// request or reclaim destination-owned buffers. Retain the client for teardown
/// when abandoning a published import.
#[must_use = "retain the import until its custody disposition is known"]
pub struct BufferImport<'a, 'w, 't, T: Transport, C: ReturnCapability> {
    accel: &'a mut RingAccel<'w, 't, T>,
    target: ImportTarget,
    custody: Custody<C, StashedMemory>,
}
impl<'w, 't, T: Transport> RingAccel<'w, 't, T> {
    /// Construction publishes nothing. A non-idle or poisoned pair returns the
    /// original source unchanged. Geometry and rights remain provider checks.
    pub fn start_buffer_import<C: ReturnCapability>(
        &mut self,
        target: ImportTarget,
        source: C,
    ) -> Result<BufferImport<'_, 'w, 't, T, C>, C> {
        if !self.client.is_idle() {
            return Err(source);
        }
        Ok(BufferImport {
            accel: self,
            target,
            custody: Custody {
                source: Some(source),
                returned: None,
                outcome: None,
            },
        })
    }
}
impl<T: Transport, C: ReturnCapability> BufferImport<'_, '_, '_, T, C> {
    /// Recover the original source only before transfer, with no pending call.
    /// Failure returns the whole continuation so exact resumption remains possible.
    pub fn recover_untransferred(mut self) -> Result<C, Self> {
        match self.custody.recover(self.accel.client.is_idle()) {
            Some(source) => Ok(source),
            None => Err(self),
        }
    }
    pub fn take_returned(&mut self) -> Option<StashedMemory> {
        self.custody.take_returned()
    }
    pub async fn advance_bounded(
        &mut self,
        budget_ns: u64,
    ) -> Result<ImportOutcome, RetainedReturnCallError> {
        if let Some(outcome) = self.custody.outcome {
            return Ok(outcome);
        }
        let reply = self
            .accel
            .advance_retained_import(self.target, &mut self.custody.source, budget_ns)
            .await?;
        let outcome = match reply {
            ImportReply::Imported { buffer } => ImportOutcome::Imported { buffer },
            ImportReply::Returned { refusal, lease } => {
                return Ok(self.custody.returned(refusal, lease));
            }
            ImportReply::Refused(refusal) => ImportOutcome::Consumed(refusal),
        };
        self.custody.outcome = Some(outcome);
        Ok(outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_recovery_requires_both_custody_and_an_idle_unfinished_pair() {
        let mut custody = Custody::<_, u32> {
            source: Some(17),
            returned: None,
            outcome: None,
        };
        assert_eq!(custody.recover(false), None);
        assert_eq!(custody.recover(true), Some(17));
        assert_eq!(custody.recover(true), None);
        custody.source = Some(19);
        custody.outcome = Some(ImportOutcome::Consumed(AccelRefusal::DeviceLost));
        assert_eq!(custody.recover(true), None);
    }
    #[test]
    fn returned_owner_survives_observer_cancellation_and_is_taken_once() {
        use core::cell::Cell;
        struct Owner<'a>(&'a Cell<u32>);
        impl Drop for Owner<'_> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        let drops = Cell::new(0);
        let mut custody = Custody::<u32, _> {
            source: None,
            returned: None,
            outcome: None,
        };
        assert_eq!(
            custody.returned(AccelRefusal::NotReady, Owner(&drops)),
            ImportOutcome::Returned(AccelRefusal::NotReady)
        );
        assert_eq!(drops.get(), 0);
        assert_eq!(custody.recover(true), None);
        let returned = custody.take_returned().unwrap();
        assert!(custody.take_returned().is_none());
        assert_eq!(
            custody.outcome,
            Some(ImportOutcome::Returned(AccelRefusal::NotReady))
        );
        drop(returned);
        assert_eq!(drops.get(), 1);
    }
    #[test]
    fn lost_correlation_after_transfer_never_republishes_or_recovers_a_source() {
        use crate::wire::AccelWire;
        use core::{
            future::Future,
            task::{Context, Poll, Waker},
        };
        use kore_abi::session::SlotRef;
        use kore_rt::{
            svc::{CallError, ClientEndpoints, REP_CAP, REQ_CAP, SvcClient},
            transport::TrapTransport,
        };
        const BYTES: usize = kore_rt::svc::window_bytes::<AccelWire>(REQ_CAP, REP_CAP);
        #[repr(align(64))]
        struct Window([u8; BYTES]);
        let mut window = Window([0; BYTES]);
        let (rings, mut server) =
            kore_rt::svc::split_window::<AccelWire>(&mut window.0, REQ_CAP, REP_CAP).unwrap();
        let transport = TrapTransport::new();
        let mut accel = RingAccel {
            client: SvcClient::new(
                &transport,
                rings,
                ClientEndpoints {
                    kick: SlotRef::from_wire(1),
                    grants: [None; kore_rt::service::MAX_GRANTS],
                },
            ),
            return_tx: SlotRef::from_wire(2),
            return_land: SlotRef::from_wire(3),
        };
        // Model missing local custody and unavailable correlation. No capability
        // or successful kernel transfer is fabricated by this host test.
        let import = BufferImport {
            accel: &mut accel,
            target: ImportTarget {
                context: 1,
                bytes: 4096,
                alignment: 4096,
                usage: 1,
            },
            custody: Custody::<StashedMemory, StashedMemory> {
                source: None,
                returned: None,
                outcome: None,
            },
        };
        let mut import = match import.recover_untransferred() {
            Err(import) => import,
            Ok(_) => panic!("missing custody cannot be recovered"),
        };
        for _ in 0..2 {
            let mut future = core::pin::pin!(import.advance_bounded(1_000_000));
            assert_eq!(
                future
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop())),
                Poll::Ready(Err(RetainedReturnCallError::AfterTransfer(
                    CallError::NoOutstanding
                )))
            );
            assert!(server.req.pop().is_none());
        }
        assert!(import.take_returned().is_none());
    }
}
