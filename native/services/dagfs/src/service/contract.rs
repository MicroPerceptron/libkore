//! The tree's service contract: [`kore_rt::service::Bundle`], [`SystemService`],
//! [`SvcProtocol`].
//!
//! Implementing these is what makes `kore-svc-dagfs` a *service* rather than a
//! library with a codec — the founder stamps [`FsEndowment::PROFILE`] into the
//! child's boot block, `kore_rt` drives [`SystemService::serve`], and the pair's
//! ring helpers become usable at this protocol's exact record widths.
//!
//! # The capability endowment is empty, and that is the point
//!
//! The filesystem owns no ambient hardware authority. Its namespace and grant
//! authority are service-local, and it never holds `PciDevice`, BAR, IRQ, or
//! IOMMU capabilities — those belong to the block owner below it. On x86 the
//! durable entry reaches storage through
//! an ordinary typed block-service client face; other architectures currently
//! select the bounded RAM provider at the entry seam. Neither provider changes
//! the filesystem's [`kore_rt::service::Bundle`] or wire contract. An empty
//! [`kore_abi::task::Profile`] states
//! that boundary in the one place the founder reads.
//!
//! # The session, and why `serve` is thin
//!
//! [`SystemService::serve`] receives a request and a [`Serve`] — and no
//! identity for the pair that sent it.
//! [`Serve::via`] is what supplies it: `ServeHost` fans in over up to
//! `SERVE_PAIRS` client pairs and stamps each request with the pair it arrived
//! on, so the pair *is* the session and nothing is inferred from wire data.
//! `via` is 0 for the ordinary single-pair row, which is why one client and
//! several are the same line.
//!
//! What several pairs add is the piece one pair gets for free: a pair's session
//! must be *created*, not assumed. `ServeHost::attach_pair` wires the ring pair
//! but knows nothing of sessions, so `serve` below mints one with
//! `ensure_session` on first contact rather than at bind time.
//!
//! # Bulk-window custody
//!
//! Bulk verbs operate only after the client has delivered a writable view on
//! the pair's retained return leg and committed [`FsRequest::AttachWindow`].
//! [`Serve::returned_input`] exposes that exact returned capability to the
//! service; [`FsService::serve_request`] adopts it into the pair-bound session.
//! A pair with no successfully adopted window refuses [`FsError::NoWindow`]
//! rather than reading an address supplied in untrusted wire data. Exec uses
//! this production path for immutable filesystem-version identities and reads;
//! recyclable child faces use the same mechanism with distinct retained
//! landing slots.

use kore_abi::{
    svc::{FsRepWire, FsReqWire},
    task::{Profile, geom, role},
};
use kore_rt::{
    service::{BindError, Bundle, PeerClose, ReturnMode, Serve, StartFailure, SystemService},
    svc::{MAX_SERVE_PAIRS, PeerAdmissionRefusal, PeerId, PeerMapPlan, SvcProtocol},
};

use kore_dagfs::DagfsProtocol;

use crate::{
    authority::SessionId,
    error::FsError,
    service::{FsRequest, FsResponse, FsService, RamService},
};

/// Exec's fixed administration pair. Pair zero remains the boot/legacy
/// client; recyclable process faces begin after both static pairs.
pub(crate) const ADMIN_VIA: u16 = 1;

/// The filesystem's authority-free endowment plus its immutable archive era.
///
/// `archive_era` is data, not a capability: eight `role::ARCHIVE_ERA` boot
/// literals carry the measured archive's 32-byte BLAKE3 digest. The profile
/// remains empty, preserving the stronger claim that FS owns no ambient
/// hardware authority. A handwritten [`Bundle`] is warranted here because the
/// ordinary derive is deliberately capability-shaped and must not pretend a
/// copyable digest is authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FsEndowment {
    archive_era: [u8; crate::persistence::ERA_BYTES],
}

impl FsEndowment {
    /// The provider-neutral host/RAM posture. Production durable binding
    /// resolves a founder-stamped era instead; zero never authorizes a system
    /// reformat.
    pub const UNSTAMPED: Self = Self {
        archive_era: [0; crate::persistence::ERA_BYTES],
    };

    pub const fn archive_era(self) -> [u8; crate::persistence::ERA_BYTES] {
        self.archive_era
    }
}

impl Bundle for FsEndowment {
    const PROFILE: Profile = Profile::new("fs", &[]);

    fn resolve(boot: &kore_abi::task::Bootstrap<'_>) -> Result<Self, BindError> {
        let mut archive_era = [0u8; crate::persistence::ERA_BYTES];
        let mut index = 0usize;
        while index < archive_era.len() / 4 {
            let word =
                boot.datum(role::ARCHIVE_ERA, index as u16)
                    .ok_or(BindError::MissingDatum {
                        role: role::ARCHIVE_ERA,
                        index: index as u16,
                    })?;
            archive_era[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
            index += 1;
        }
        Ok(Self { archive_era })
    }
}

impl SystemService for RamService {
    /// The filesystem is an I/O worker whichever backend founds it: parked
    /// on completions for most of its life.  Prefer the efficiency cores.
    const CORE_PREFERENCE: kore_abi::task::CorePreference =
        kore_abi::task::CorePreference::Efficient;
    const DEVICE_CLASSES: &'static [kore_rt::service::DeviceClass] = crate::FS_DEVICE_CLASSES;
    const NAME: &'static str = "fs";
    type Endowment = FsEndowment;

    /// Each client donates one writable `LeasedRegion` data window. The
    /// generic host rotates a distinct retained landing for every serving
    /// pair, so session N can never satisfy session M's attachment.
    const RETURN_MODE: ReturnMode = ReturnMode::RetainedAny;

    fn bind(_endowment: FsEndowment) -> Self {
        // Pair 0's root, minted early. Every other pair's arrives on first
        // contact in `serve`, and `ensure_session` is idempotent, so the
        // single-pair and many-pair shapes agree rather than fight.
        let mut service = Self::default();
        let session = SessionId::new(0);
        // A service that cannot give its one client a root is a service that
        // can do nothing; the refusal surfaces on the client's first call
        // rather than as a panic here.
        let _ = service.ensure_session(session);
        service.bind_session(session);
        service
    }

    fn start_on_metal(&mut self, rt: &kore_rt::entry::Rt) -> Result<(), StartFailure> {
        self.configure_peer_map(peer_map(rt)?);
        if !self.establish_admin_source() {
            return Err(StartFailure::new(2));
        }
        // The RAM composition is not durable, but exec discovers its workspace
        // by the same grant name on every composition. Publishing it here keeps
        // the volatile and durable shapes interchangeable from exec's side —
        // the data does not survive, the wiring does.
        if !self.establish_admin_workspace() {
            return Err(StartFailure::new(3));
        }
        Ok(())
    }

    /// **The session is the pair the record arrived on**, never anything the
    /// request says: caller identity is derived from the pair/session context,
    /// never trusted from a caller-supplied task ID.
    ///
    /// [`Serve::via`] is the founded wiring speaking — a client cannot name a
    /// session it does not own, because nothing a client can write selects
    /// one. The mapping is the identity, pair *n* is session *n*: deliberately
    /// trivial and deliberately total, so there is no lookup that could be made
    /// to miss and no table a client could influence. `via` is 0 for the
    /// ordinary single-pair shape, so one client and several are the same line.
    fn serve(&mut self, req: FsRequest, sv: &mut Serve<'_, FsResponse>) {
        let response = self.serve_request(req, sv);
        sv.reply(response);
    }

    fn peer_admitted(&mut self, result: Result<PeerId, PeerAdmissionRefusal>) -> bool {
        FsService::peer_admitted(self, result)
    }

    fn peer_closed(&mut self, via: u16, reason: PeerClose) -> bool {
        FsService::peer_closed(self, via, reason)
    }
}

impl<P: crate::provider::StorageProvider> FsService<P> {
    /// Shared request dispatch for RAM and durable service compositions.
    pub(crate) fn serve_request(
        &mut self,
        req: FsRequest,
        sv: &mut Serve<'_, FsResponse>,
    ) -> FsResponse {
        crate::block_on(self.serve_request_async(req, sv))
    }

    /// Async request dispatch used by compositions whose storage provider can
    /// park. Keeping the `Serve` borrow inside this future preserves the real
    /// executor waker across dependent service calls; a nested noop-waker
    /// driver would strand the request after publishing it.
    pub(crate) async fn serve_request_async(
        &mut self,
        req: FsRequest,
        sv: &mut Serve<'_, FsResponse>,
    ) -> FsResponse {
        match req {
            FsRequest::RegisterPrivateView {
                rights,
                name,
                announced,
            } => {
                if sv.via() != ADMIN_VIA {
                    return FsResponse::Refused(FsError::AccessDenied);
                }
                let Some(mut input) = sv.returned_input(announced) else {
                    return FsResponse::Refused(FsError::AccessDenied);
                };
                let Some(returned) = input.take() else {
                    return FsResponse::Refused(FsError::AccessDenied);
                };
                let canary =
                    match returned.try_into_chan_send(crate::FsViewCapability::INVOCATION_RIGHTS) {
                        Ok(canary) => canary,
                        Err(_) => return FsResponse::Refused(FsError::AccessDenied),
                    };
                let Some(scope) = kore_rt::cap::verbs::capability_scope_rid(
                    canary.slot(),
                    kore_abi::cap::ResourceKind::ChanSend,
                    kore_abi::cap::Rights::ACCESS,
                ) else {
                    let _ = canary.try_retire();
                    return FsResponse::Refused(FsError::AccessDenied);
                };
                return match self.register_private_invocation_view(
                    scope,
                    &name,
                    rights,
                    Some(canary),
                ) {
                    Ok(()) => FsResponse::Done,
                    Err(error) => FsResponse::Refused(error),
                };
            }
            FsRequest::RegisterSystemView { rights, announced } => {
                if sv.via() != ADMIN_VIA {
                    return FsResponse::Refused(FsError::AccessDenied);
                }
                let scope = sv.returned_input(announced).and_then(|input| {
                    input.capability().and_then(|capability| {
                        capability.scope_rid_with(
                            kore_abi::cap::ResourceKind::ChanSend,
                            kore_abi::cap::Rights::ACCESS,
                        )
                    })
                });
                return match scope.and_then(|scope| {
                    self.register_system_invocation_view(scope, rights)
                        .ok()
                        .map(|_| ())
                }) {
                    Some(()) => FsResponse::Done,
                    None => FsResponse::Refused(FsError::AccessDenied),
                };
            }
            FsRequest::ApplyView {
                peer,
                rights,
                announced,
            } => {
                if sv.via() != ADMIN_VIA {
                    return FsResponse::Refused(FsError::AccessDenied);
                }
                let scope = sv.returned_input(announced).and_then(|input| {
                    input.capability().and_then(|capability| {
                        capability.scope_rid_with(
                            kore_abi::cap::ResourceKind::ChanSend,
                            kore_abi::cap::Rights::ACCESS,
                        )
                    })
                });
                return match scope.and_then(|scope| {
                    self.apply_invocation_view(scope, peer, rights)
                        .ok()
                        .map(|_| ())
                }) {
                    Some(()) => FsResponse::Done,
                    None => FsResponse::Refused(FsError::AccessDenied),
                };
            }
            FsRequest::AdmitPeer { part, announced } => {
                return self.admit_peer_component(part, announced, sv);
            }
            FsRequest::TakeAdmission => {
                if sv.via() != ADMIN_VIA {
                    return FsResponse::Refused(FsError::AccessDenied);
                }
                return self.take_peer_admission();
            }
            FsRequest::RetirePeer { peer } => {
                if sv.via() != ADMIN_VIA
                    || self.peer_via.get(peer.slot()).copied().flatten() != Some(peer.arrival())
                {
                    return FsResponse::Refused(FsError::AccessDenied);
                }
                sv.retire_peer(peer, PeerClose::Orderly);
                return FsResponse::Done;
            }
            FsRequest::ConfirmRetired { peer } => {
                if sv.via() != ADMIN_VIA {
                    return FsResponse::Refused(FsError::AccessDenied);
                }
                // A different generation in the same cell is already proof
                // that this exact peer was retired.  Never let a stale ticket
                // name or disturb its replacement.
                if !self.peer_is_retired(peer) {
                    return FsResponse::Refused(FsError::Conflict);
                }
                return FsResponse::Done;
            }
            FsRequest::ClosePeer => {
                if sv.via() == 0 || self.session_for_via(sv.via()).is_none() {
                    return FsResponse::Refused(FsError::AccessDenied);
                }
                sv.close_peer(PeerClose::Orderly);
                return FsResponse::Done;
            }
            _ => {}
        }
        let Some(session) = self.session_for_via(sv.via()) else {
            return FsResponse::Refused(FsError::AccessDenied);
        };
        // A pair's per-client state is its root: receiving a pair creates
        // one. Minting it on first contact rather than at bind time keeps
        // the two facts inseparable: a pair that can be polled is a pair
        // whose session exists. Attaching a fixed set up front is what
        // previously let a pair be wired, stamped and polled while its
        // session had never been attached, so its first `OpenRoot` refused
        // with nothing to blame. Dynamic sessions root through the pinned
        // system view (the invocation posture) so their facts persist;
        // fixed sessions keep private roots.
        // REVISIT: `OutOfObjects` because a private root *is* an object and a
        // full object arena is the realistic failure — but `attach_session`
        // can also refuse for session-table reasons, which this then
        // mislabels. The const-assert below makes the session-table case
        // unreachable by construction, so the label is honest today and wrong
        // only if that assert is ever weakened.
        if !self.ensure_dynamic_session(session) {
            return FsResponse::Refused(FsError::OutOfObjects);
        }
        match req {
            FsRequest::AttachWindow { slot, .. } => {
                let announced = kore_abi::session::SlotRef::from_wire(slot);
                let adopted = sv
                    .returned_input(announced)
                    .and_then(|mut input| input.take())
                    .ok_or(FsError::NoWindow)
                    .and_then(|owner| self.adopt_window(session, slot, owner));
                match adopted {
                    Ok(()) => FsResponse::WindowAttached {
                        max_transfer: self.transfer_limit(),
                    },
                    Err(error) => FsResponse::Refused(error),
                }
            }
            FsRequest::BindView {
                handle,
                rights,
                mode,
                announced,
            } => {
                let Some(mut input) = sv.returned_input(announced) else {
                    return FsResponse::Refused(FsError::AccessDenied);
                };
                let Some(endpoint) = input.take() else {
                    return FsResponse::Refused(FsError::AccessDenied);
                };
                match self.register_invocation_endpoint(session, handle, rights, mode, endpoint)
                {
                    Ok(()) => FsResponse::Done,
                    Err(error) => FsResponse::Refused(error),
                }
            }
            FsRequest::GrantView {
                handle,
                rights,
                display_name,
                announced,
            } => {
                let Some(source_view) = self.binding_owner_for_session(session) else {
                    return FsResponse::Refused(FsError::AccessDenied);
                };
                let Some(mut input) = sv.returned_input(announced) else {
                    return FsResponse::Refused(FsError::AccessDenied);
                };
                let Some(returned) = input.take() else {
                    return FsResponse::Refused(FsError::AccessDenied);
                };
                let target = match returned.try_into_chan_send(crate::FS_GRANT_TARGET_RIGHTS) {
                    Ok(target) => target,
                    Err(_) => return FsResponse::Refused(FsError::AccessDenied),
                };
                if target.rights() != crate::FS_GRANT_TARGET_RIGHTS {
                    let _ = target.try_retire();
                    return FsResponse::Refused(FsError::AccessDenied);
                }
                let scope = kore_rt::cap::verbs::capability_scope_rid(
                    target.slot(),
                    kore_abi::cap::ResourceKind::ChanSend,
                    crate::FS_GRANT_TARGET_RIGHTS,
                );
                let retired = target.try_retire().is_ok();
                let Some(destination_view) =
                    scope.and_then(|scope| self.task_view_for_scope(scope))
                else {
                    return FsResponse::Refused(FsError::AccessDenied);
                };
                if !retired || destination_view == source_view {
                    return FsResponse::Refused(FsError::AccessDenied);
                }
                return match self.engine.grant_between_task_views(
                    session,
                    handle,
                    source_view,
                    destination_view,
                    rights,
                    Some(display_name),
                ) {
                    Ok(grant) => FsResponse::Granted { grant },
                    Err(error) => FsResponse::Refused(error),
                };
            }
            _ => self.dispatch_mapped_async(session, &req).await,
        }
    }
}

/// Each dynamic peer occupies one page-aligned ring cell in the service's
/// founder-reserved transient band. The generic host validates the actual
/// protocol byte requirement against this stride before mapping.
pub(crate) const PEER_MAP_STRIDE: u64 = 0x2000;

/// Derive the only lawful recyclable-ring placement from founder geometry.
/// Wire data never carries a VA and a service without the exact reservation
/// refuses readiness rather than accepting a peer it cannot map.
pub(crate) fn peer_map(rt: &kore_rt::entry::Rt) -> Result<PeerMapPlan, StartFailure> {
    let window = rt
        .geom()
        .and_then(|geometry| geometry.window_of(geom::TRANSIENT))
        .filter(|window| {
            window.base() != 0
                && window.base() & 0xfff == 0
                && window.len() >= PEER_MAP_STRIDE * MAX_SERVE_PAIRS as u64
        })
        .ok_or(StartFailure::new(1))?;
    Ok(PeerMapPlan {
        base: window.base(),
        stride: PEER_MAP_STRIDE,
    })
}

impl SvcProtocol for RamService {
    type Request = <DagfsProtocol as SvcProtocol>::Request;
    type Response = <DagfsProtocol as SvcProtocol>::Response;
    type ReqWire = <DagfsProtocol as SvcProtocol>::ReqWire;
    type RepWire = <DagfsProtocol as SvcProtocol>::RepWire;

    /// The protocol is the leaf's ([`DagfsProtocol`]): zero grants ever, two
    /// founder-stamped pairs, and the fail-closed codec. Delegating every item
    /// is what keeps the serving side from drifting from the client it answers.
    const REPLY_GRANTS: usize = <DagfsProtocol as SvcProtocol>::REPLY_GRANTS;
    const SERVE_PAIRS: usize = <DagfsProtocol as SvcProtocol>::SERVE_PAIRS;

    fn reply_matches(request: &FsReqWire, response: &FsResponse, grants: u8) -> bool {
        DagfsProtocol::reply_matches(request, response, grants)
    }

    fn encode_req(req: &FsRequest) -> FsReqWire {
        DagfsProtocol::encode_req(req)
    }

    fn decode_req(w: &FsReqWire) -> Option<FsRequest> {
        DagfsProtocol::decode_req(w)
    }

    fn encode_rep(rep: &FsResponse, grants: u8) -> FsRepWire {
        DagfsProtocol::encode_rep(rep, grants)
    }

    fn decode_rep(w: &FsRepWire) -> Option<(FsResponse, u8)> {
        DagfsProtocol::decode_rep(w)
    }

    fn decode_rep_for(request: &FsReqWire, reply: &FsRepWire) -> Option<(FsResponse, u8)> {
        DagfsProtocol::decode_rep_for(request, reply)
    }
}

impl<P: crate::provider::StorageProvider> FsService<P> {
    /// Bind the fixed administration pair's session.
    pub fn bind_session(&mut self, session: SessionId) {
        self.session = Some(session);
    }

    /// The bound session, defaulting to session 0 for a service constructed
    /// without one.
    pub fn bound_session(&self) -> SessionId {
        self.session.unwrap_or(SessionId::new(0))
    }
}

/// The pair window must hold both 8-deep rings at this protocol's widths,
/// inside the founder's `[layout] ring_bytes`.
///
/// Asserted here through the tree's own `window_bytes`, so this cannot drift
/// from the arithmetic the founder and the serving host actually use.
const _: () = {
    use kore_rt::svc::{REP_CAP, REQ_CAP, window_bytes};
    assert!(window_bytes::<RamService>(REQ_CAP, REP_CAP) == 1728);
    assert!(window_bytes::<RamService>(REQ_CAP, REP_CAP) <= 0x2000);
};

/// Every pair the founding may stamp must be nameable as a session, or
/// [`SystemService::serve`] would be handed a client the engine can never root.
///
/// `SERVE_PAIRS` is what the founder stamps; `MAX_SESSIONS` is what the engine
/// can root. This is the one place those two bounds meet, so a change to either
/// breaks here rather than at a client's first `OpenRoot`.
const _: () = assert!(
    crate::bounds::MAX_SESSIONS == kore_rt::svc::MAX_SERVE_PAIRS + crate::bounds::MAX_TASK_VIEWS,
    "the engine's session arena must cover every host pair plus every stable task view"
);

/// `ServeHost` requires the request/reply-plane scope; the filesystem meets it.
///
/// Stated here rather than left to the first instantiation so that a change to
/// `REPLY_GRANTS` or `RETURN_MODE` breaks at this file — where the reason lives
/// — instead of at some distant construction site.
const _: () = {
    assert!(<RamService as SvcProtocol>::REPLY_GRANTS == 0);
    assert!(matches!(
        <RamService as SystemService>::RETURN_MODE,
        kore_rt::service::ReturnMode::RetainedAny
    ));
};

#[cfg(test)]
mod endowment_tests {
    use super::*;
    use kore_abi::task::{BootBlock, Bootstrap};

    #[test]
    fn archive_era_round_trips_without_entering_the_capability_profile() {
        let expected: [u8; crate::persistence::ERA_BYTES] =
            core::array::from_fn(|index| index as u8);
        let mut block = BootBlock::new();
        for (index, bytes) in expected.chunks_exact(4).enumerate() {
            assert!(block.push_data(
                role::ARCHIVE_ERA,
                index as u16,
                u32::from_le_bytes(bytes.try_into().unwrap()),
            ));
        }
        let boot = Bootstrap::of(&block).unwrap();
        assert_eq!(FsEndowment::resolve(&boot).unwrap().archive_era(), expected);
        assert!(FsEndowment::PROFILE.roles.is_empty());
        assert_eq!(boot.handles().count(), 0);
    }

    #[test]
    fn a_partial_archive_era_refuses_binding() {
        let mut block = BootBlock::new();
        for index in 0..7u16 {
            assert!(block.push_data(role::ARCHIVE_ERA, index, u32::from(index)));
        }
        let boot = Bootstrap::of(&block).unwrap();
        assert_eq!(
            FsEndowment::resolve(&boot),
            Err(BindError::MissingDatum {
                role: role::ARCHIVE_ERA,
                index: 7,
            })
        );
    }
}
