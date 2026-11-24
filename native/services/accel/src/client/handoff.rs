//! Allocation-custody handoff between two possessed accelerator faces.

use super::{ExportReply, ImportReply, RingAccel};
use crate::service::AccelRefusal;
use kore_rt::{
    cap::StashedMemory,
    svc::{CallError, RetainedReturnCallError},
    transport::Transport,
};

/// Destination binding data. The destination still validates context ownership,
/// size, alignment and usage against the transferred allocation and its limits.
#[derive(Clone, Copy, Debug)]
pub struct ImportTarget {
    pub context: u64,
    pub bytes: u64,
    pub alignment: u64,
    pub usage: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandoffOutcome {
    Imported {
        buffer: u64,
    },
    /// No allocation was returned. Busy preserves the source buffer;
    /// DeviceLost or other containment refusals do not prove it remains usable.
    SourceRefused(AccelRefusal),
    /// Destination refused and returned authenticated allocation custody.
    Returned(AccelRefusal),
    /// Destination refused without returning an allocation. No local reuse is
    /// authorized; the destination owns cleanup or containment.
    Consumed(AccelRefusal),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandoffProgress {
    Pending,
    Finished(HandoffOutcome),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandoffError {
    Export(CallError),
    Import(RetainedReturnCallError),
}

#[derive(Clone, Copy)]
enum Phase {
    Export,
    Import,
    Finished(HandoffOutcome),
}

// The affine owner lives outside every cancellable future. Parameterization
// keeps the custody transitions independent of the capability representation.
struct Custody<L> {
    phase: Phase,
    lease: Option<L>,
}
impl<L> Custody<L> {
    fn exported(&mut self, lease: L) {
        self.lease = Some(lease);
        self.phase = Phase::Import;
    }
    fn finish(&mut self, outcome: HandoffOutcome) -> HandoffProgress {
        self.phase = Phase::Finished(outcome);
        HandoffProgress::Finished(outcome)
    }
    fn returned(&mut self, refusal: AccelRefusal, lease: L) -> HandoffProgress {
        self.lease = Some(lease);
        self.finish(HandoffOutcome::Returned(refusal))
    }
    fn take_untransferred(&mut self, destination_idle: bool) -> Option<L> {
        if destination_idle && matches!(self.phase, Phase::Import) {
            self.lease.take()
        } else {
            None
        }
    }
    fn take_returned(&mut self) -> Option<L> {
        if matches!(self.phase, Phase::Finished(HandoffOutcome::Returned(_))) {
            self.lease.take()
        } else {
            None
        }
    }
}

/// Export retirement, exact lease transfer, then destination admission.
///
/// Retain this value when cancelling an `advance_bounded` future. Both face
/// borrows prevent unrelated calls from consuming a pending reply, and the
/// lease stays here until the synchronous capability-transfer boundary. Every
/// retry resumes the exact request before considering fresh publication.
///
/// This moves host-DRAM custody without copying bytes. It depends on the
/// source's completed export retirement/cache acquisition; it does not prove a
/// foreign device fence or implement peer-aperture P2PDMA. Pool buffers whose
/// provider cannot export individual allocations still refuse at the source.
/// Dropping this value does not undo a published request. Pending endpoint
/// state remains with its client; an untransferred local lease retires on drop.
#[must_use = "retain the handoff until its ownership disposition is known"]
pub struct BufferHandoff<'a, 'sw, 'st, 'dw, 'dt, S: Transport, D: Transport> {
    source: &'a mut RingAccel<'sw, 'st, S>,
    destination: &'a mut RingAccel<'dw, 'dt, D>,
    buffer: u64,
    target: ImportTarget,
    custody: Custody<StashedMemory>,
}

impl<'sw, 'st, S: Transport> RingAccel<'sw, 'st, S> {
    /// Start only on two idle, unpoisoned pairs. Construction publishes nothing.
    pub fn handoff<'a, 'dw, 'dt, D: Transport>(
        &'a mut self,
        buffer: u64,
        destination: &'a mut RingAccel<'dw, 'dt, D>,
        target: ImportTarget,
    ) -> Option<BufferHandoff<'a, 'sw, 'st, 'dw, 'dt, S, D>> {
        if !self.client.is_idle() || !destination.client.is_idle() {
            return None;
        }
        Some(BufferHandoff {
            source: self,
            destination,
            buffer,
            target,
            custody: Custody {
                phase: Phase::Export,
                lease: None,
            },
        })
    }
}

impl<S: Transport, D: Transport> BufferHandoff<'_, '_, '_, '_, '_, S, D> {
    /// Recover a destination-returned allocation exactly once. It is never
    /// accessible while export/import outcome is uncertain or admission is live.
    pub fn take_returned(&mut self) -> Option<StashedMemory> {
        self.custody.take_returned()
    }

    /// Abandon destination admission only while the exported lease is still
    /// local and that pair has no outstanding request. Refusal returns this
    /// entire continuation, preserving exact resumption after publication.
    pub fn recover_untransferred(mut self) -> Result<StashedMemory, Self> {
        match self
            .custody
            .take_untransferred(self.destination.client.is_idle())
        {
            Some(lease) => Ok(lease),
            None => Err(self),
        }
    }

    /// Advance one protocol phase. Every export refusal (including Busy)
    /// finishes without destination admission. Busy preserves the source
    /// buffer so callers can release both borrows, retire events and retry.
    /// Import refusal has returned custody
    /// distinguished from destination-owned cleanup. The budget bounds reply
    /// observation, not device completion or capability-post preparation.
    pub async fn advance_bounded(
        &mut self,
        budget_ns: u64,
    ) -> Result<HandoffProgress, HandoffError> {
        match self.custody.phase {
            Phase::Finished(outcome) => Ok(HandoffProgress::Finished(outcome)),
            Phase::Export => {
                match self
                    .source
                    .advance_export_buffer(self.buffer, budget_ns)
                    .await
                    .map_err(HandoffError::Export)?
                {
                    ExportReply::Returned(lease) => {
                        self.custody.exported(lease);
                        Ok(HandoffProgress::Pending)
                    }
                    ExportReply::Refused(refusal) => {
                        Ok(self.custody.finish(HandoffOutcome::SourceRefused(refusal)))
                    }
                }
            }
            Phase::Import => {
                match self
                    .destination
                    .advance_retained_import(self.target, &mut self.custody.lease, budget_ns)
                    .await
                    .map_err(HandoffError::Import)?
                {
                    ImportReply::Imported { buffer } => {
                        Ok(self.custody.finish(HandoffOutcome::Imported { buffer }))
                    }
                    ImportReply::Returned { refusal, lease } => {
                        Ok(self.custody.returned(refusal, lease))
                    }
                    ImportReply::Refused(refusal) => {
                        Ok(self.custody.finish(HandoffOutcome::Consumed(refusal)))
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn custody_is_unavailable_during_transfer_and_returned_exactly_once() {
        use core::cell::Cell;
        struct Owner<'a>(&'a Cell<u32>);
        impl Drop for Owner<'_> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        let drops = Cell::new(0);
        let mut custody = Custody {
            phase: Phase::Export,
            lease: None,
        };
        custody.exported(Owner(&drops));
        assert!(custody.take_returned().is_none());
        let transferred = custody.lease.take().unwrap(); // synchronous transfer boundary
        assert_eq!(drops.get(), 0);
        assert!(custody.take_returned().is_none());
        assert_eq!(
            custody.returned(AccelRefusal::NotReady, transferred),
            HandoffProgress::Finished(HandoffOutcome::Returned(AccelRefusal::NotReady))
        );
        let returned = custody.take_returned().unwrap();
        assert_eq!(drops.get(), 0);
        drop(returned);
        assert_eq!(drops.get(), 1);
        assert!(custody.take_returned().is_none());
        assert!(matches!(
            custody.phase,
            Phase::Finished(HandoffOutcome::Returned(_))
        ));
    }

    #[test]
    fn recovery_requires_local_custody_and_no_destination_request() {
        let mut custody = Custody {
            phase: Phase::Export,
            lease: None,
        };
        assert_eq!(custody.take_untransferred(true), None);
        custody.exported(17);
        assert_eq!(custody.take_untransferred(false), None);
        assert_eq!(custody.take_untransferred(true), Some(17));
        assert_eq!(custody.take_untransferred(true), None);
        // Publication consumed the local owner: an idle observation alone
        // cannot manufacture it, even if the client has lost correlation.
        custody.phase = Phase::Import;
        assert_eq!(custody.take_untransferred(true), None);
    }

    #[test]
    fn missing_export_grant_lane_never_publishes_either_request() {
        use crate::wire::AccelWire;
        use core::{
            future::Future,
            task::{Context, Poll, Waker},
        };
        use kore_abi::session::SlotRef;
        use kore_rt::{
            svc::{ClientEndpoints, REP_CAP, REQ_CAP, SvcClient},
            transport::TrapTransport,
        };
        const BYTES: usize = kore_rt::svc::window_bytes::<AccelWire>(REQ_CAP, REP_CAP);
        #[repr(align(64))]
        struct Window([u8; BYTES]);
        let mut sw = Window([0; BYTES]);
        let mut dw = Window([0; BYTES]);
        let (sr, mut ss) =
            kore_rt::svc::split_window::<AccelWire>(&mut sw.0, REQ_CAP, REP_CAP).unwrap();
        let (dr, mut ds) =
            kore_rt::svc::split_window::<AccelWire>(&mut dw.0, REQ_CAP, REP_CAP).unwrap();
        let transport = TrapTransport::new();
        let mut source = RingAccel {
            client: SvcClient::new(
                &transport,
                sr,
                ClientEndpoints {
                    kick: SlotRef::from_wire(1),
                    grants: [None; kore_rt::service::MAX_GRANTS],
                },
            ),
            return_tx: SlotRef::from_wire(2),
            return_land: SlotRef::from_wire(3),
        };
        let mut destination = RingAccel {
            client: SvcClient::new(
                &transport,
                dr,
                ClientEndpoints {
                    kick: SlotRef::from_wire(4),
                    grants: [None; kore_rt::service::MAX_GRANTS],
                },
            ),
            return_tx: SlotRef::from_wire(5),
            return_land: SlotRef::from_wire(6),
        };
        let mut handoff = source
            .handoff(
                42,
                &mut destination,
                ImportTarget {
                    context: 1,
                    bytes: 4096,
                    alignment: 4096,
                    usage: 1,
                },
            )
            .unwrap();
        for _ in 0..2 {
            let mut future = core::pin::pin!(handoff.advance_bounded(1_000_000));
            assert_eq!(
                future
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop())),
                Poll::Ready(Err(HandoffError::Export(CallError::Wiring)))
            );
            assert!(ss.req.pop().is_none());
            assert!(ds.req.pop().is_none());
        }
        assert!(handoff.take_returned().is_none());
    }
}
