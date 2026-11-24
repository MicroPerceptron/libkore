//! Bounded founder-owned peer assembly; installed identity comes from the host.
use super::{AccelRefusal as E, AccelRequest as Q, AccelResponse as R};
use kore_rt::{
    service::{PeerClose, Serve},
    svc::{PeerAdmissionRefusal, PeerBundleAssembler, PeerBundlePart, PeerId, PeerMapPlan},
};

pub(super) struct Peers {
    assembler: PeerBundleAssembler,
    pub map: Option<PeerMapPlan>,
    pending: bool,
    result: Option<Result<PeerId, E>>,
}
impl Peers {
    pub const fn new() -> Self {
        Self {
            assembler: PeerBundleAssembler::new(crate::wire::ACCEL_REPLY_GRANTS, true),
            map: None,
            pending: false,
            result: None,
        }
    }
    pub fn pending(&self) -> bool {
        self.pending
    }
    pub fn admitted(
        &mut self,
        result: Result<PeerId, PeerAdmissionRefusal>,
        accepted: bool,
    ) -> bool {
        if !self.pending {
            return false;
        }
        self.pending = false;
        self.result = Some(match result {
            Ok(peer) if accepted => Ok(peer),
            Ok(_) => Err(E::Busy),
            Err(PeerAdmissionRefusal::Capacity) => Err(E::ResourceLimit),
            Err(PeerAdmissionRefusal::NoScratchSlots) => Err(E::OutOfMemory),
            Err(PeerAdmissionRefusal::MapFailed) => Err(E::Internal),
            Err(_) => Err(E::InvalidArgument),
        });
        accepted
    }
    pub fn handle(&mut self, req: Q, sv: &mut Serve<'_, R>) -> Option<R> {
        Some(match req {
            Q::AdmitPeer { part, announced } => {
                if !crate::wire::is_admin_pair(sv.via()) {
                    return Some(R::Refused(E::Denied));
                }
                // A `Ring` is the boundary between bundles, so one arriving
                // while the assembler still owns components proves the
                // previous assembly was abandoned — its asker spent its
                // bounded re-ask and gave up.  Retire what it left, and refuse
                // the replacement while any of those owners is still closing
                // (`PeerBundleAssembler::retire_incomplete`).
                //
                // Without this the leftovers are permanent: the next launch's
                // matching part answers `Duplicate`, the bundle never
                // completes, no verdict is ever published, and every later
                // launch that needs this face refuses for the rest of the boot.
                //
                // Behind the administration gate, and checked before anything
                // else touches the assembler: a via this service does not
                // trust to assemble must never retire what is being assembled.
                if matches!(part, PeerBundlePart::Ring)
                    && !self.assembler.is_empty()
                    && !self.assembler.retire_incomplete()
                {
                    return Some(R::Refused(E::Busy));
                }
                if self.pending || self.result.is_some() {
                    return Some(R::Refused(E::Busy));
                }
                let Some(map) = self.map else {
                    return Some(R::Refused(E::Unsupported));
                };
                let Some(mut input) = sv.returned_input(announced) else {
                    return Some(R::Refused(E::InvalidArgument));
                };
                let Some(cap) = input.capability() else {
                    return Some(R::Refused(E::InvalidArgument));
                };
                if self.assembler.accepts(part, cap).is_err() {
                    return Some(R::Refused(E::InvalidArgument));
                }
                let owner = input.take().expect("validated owner remains present");
                if let Some(bundle) = self
                    .assembler
                    .push(part, owner)
                    .expect("same synchronous validation")
                {
                    self.pending = true;
                    sv.admit_peer(bundle, map);
                }
                R::PeerAccepted
            }
            Q::TakePeerAdmission => {
                if !crate::wire::is_admin_pair(sv.via()) {
                    return Some(R::Refused(E::Denied));
                }
                match self.result.take() {
                    Some(Ok(peer)) => R::PeerAdmitted { peer },
                    Some(Err(error)) => R::Refused(error),
                    // No verdict, and the assembler still owns components:
                    // this bundle never completed, so `admit_peer` was never
                    // called and no verdict can EVER be published.  `Busy`
                    // here would be a lie the asker waits on forever — and
                    // exec fences new provisioning while an id-less face
                    // exists, so that wait refuses every later launch that
                    // needs this row.
                    None if !self.assembler.is_empty() => {
                        let _ = self.assembler.retire_incomplete();
                        R::Refused(E::BadObject)
                    }
                    // An EMPTY assembler means the bundle completed and the
                    // host has not drained its admission action yet: a
                    // verdict IS coming (`pending`), and settling here would
                    // orphan the face it is about to admit.
                    None => R::Refused(E::Busy),
                }
            }
            Q::ClosePeer => {
                if PeerId::from_arrival(sv.via()).is_none() {
                    return Some(R::Refused(E::Denied));
                }
                sv.close_peer(PeerClose::Orderly);
                R::PeerClosing
            }
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service::{AccelService, arrival_of};
    type MockAccelerator = kore_driver::accel::ArtifactBackend<virtio_accel_mock::MockAccelerator>;
    #[test]
    fn admission_result_requires_pending_work_and_binds_the_exact_arrival() {
        let mut host = AccelService::host(MockAccelerator::default(), None, 7, 1 << 24, 1 << 24);
        let peer = PeerId::from_arrival(arrival_of(2, 1)).unwrap();
        assert!(!host.peer_admitted(Ok(peer)));
        assert!(host.sessions[2].arrival.is_none());
        host.peers.pending = true; // simulate only the outer runtime callback boundary
        assert!(host.peer_admitted(Ok(peer)));
        assert_eq!(host.sessions[2].arrival, Some(arrival_of(2, 1)));
        assert_eq!(host.peers.result, Some(Ok(peer)));
        assert!(!host.peers.pending);
        assert!(!host.peer_admitted(Ok(PeerId::from_arrival(arrival_of(2, 2)).unwrap())));
        assert_eq!(host.sessions[2].arrival, Some(arrival_of(2, 1)));
    }
    #[test]
    fn failed_or_colliding_admission_does_not_replace_session_custody() {
        let mut host = AccelService::host(MockAccelerator::default(), None, 7, 1 << 24, 1 << 24);
        assert!(host.claim_arrival(1));
        host.peers.pending = true;
        assert!(!host.peer_admitted(Ok(PeerId::from_arrival(arrival_of(1, 1)).unwrap())));
        assert_eq!(host.sessions[1].arrival, Some(1));
        assert_eq!(host.peers.result, Some(Err(E::Busy)));
        host.peers.pending = true;
        assert!(!host.peer_admitted(Err(PeerAdmissionRefusal::Capacity)));
        assert_eq!(host.peers.result, Some(Err(E::ResourceLimit)));
    }

    /// A recyclable peer seated in an administration pair's cell is the
    /// prior-session collision: `pair_index` would fold it onto the same
    /// `PairSession` as the exec face. Refused at admission, fail-closed,
    /// with the cell's custody untouched.
    #[test]
    fn a_recyclable_peer_in_an_admin_cell_is_refused() {
        let mut host = AccelService::host(MockAccelerator::default(), None, 7, 1 << 24, 1 << 24);
        for slot in 0..crate::wire::ACCEL_SERVE_PAIRS as u16 {
            host.peers.pending = true;
            let peer = PeerId::from_arrival(arrival_of(slot, 1)).unwrap();
            assert!(!host.peer_admitted(Ok(peer)));
            assert!(host.sessions[usize::from(slot)].arrival.is_none());
            assert_eq!(host.peers.result, Some(Err(E::Busy)));
        }
        host.peers.pending = true;
        let peer =
            PeerId::from_arrival(arrival_of(crate::wire::ACCEL_SERVE_PAIRS as u16, 1)).unwrap();
        assert!(host.peer_admitted(Ok(peer)));
    }

    #[test]
    fn terminal_provider_still_admits_an_identity_face() {
        let mut host = AccelService::host(MockAccelerator::default(), None, 7, 1 << 24, 1 << 24);
        assert!(host.retire_live_epoch());
        assert!(host.restart_required());
        host.peers.pending = true; // the runtime has accepted the complete bundle
        let peer = PeerId::from_arrival(arrival_of(2, 1)).unwrap();
        assert!(host.peer_admitted(Ok(peer)));
        assert_eq!(host.sessions[2].arrival, Some(peer.arrival()));
        assert_eq!(host.peers.result, Some(Ok(peer)));
    }
}
