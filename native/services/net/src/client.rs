//! `client` — the frame-service client face for peers (`res/USERLAND.md` §4).
//!
//! [`RingNet`] is the born-async ring proxy over the pair's shared window,
//! fixed to the backend-free [`NetWire`] marker: a client never imports a
//! hardware backend type to place a call, and byte identity between what a
//! client sends and what a host decodes holds by construction rather than by
//! test (`wire.rs`'s delegation).
//!
//! Two flows matter beyond the plain typed calls:
//!
//! - **`open_vif` rides the return leg.** The caller formats its own window —
//!   both rings, with [`format_window`] — and transfers a lease of it with the
//!   request. `spsc::init` needs exclusive access, so the *client* formats and
//!   the service validates on attach; a malformed window is refused there
//!   rather than trusted.
//! - **`signal` collects the reply's one retained grant**, possesses it as a
//!   [`ChanRecvToken`], and hands it back. The caller parks a
//!   [`ChanRx`](kore_rt::chan::ChanRx) on it and treats every
//!   [`COOKIE_NET_RX`](crate::COOKIE_NET_RX) notice as "drain your receive
//!   ring" — the ring stays the truth, so a missed notice loses no frame —
//!   and every [`COOKIE_NET_LINK`](crate::COOKIE_NET_LINK) notice as "the
//!   link moved, `query` again" — the level rides the reply, so a missed
//!   notice loses no state either.
//!
//! An infrastructure failure of the request path **panics** (the executor's
//! transport-failure posture): the closed [`NetRefusal`](crate::NetRefusal)
//! vocabulary has no honest infra arm, and inventing one would let a torn pair
//! masquerade as a policy refusal.

use kore_abi::cap::Rights;
use kore_abi::session::SlotRef;
use kore_driver::net::window::{RX_DEPTH, RX_RING_AT, TX_DEPTH, WINDOW_BYTES};
use kore_driver::net::{FrameDesc, FrameFilter, VifId};
use kore_ring::spsc::{self, RingError};
use kore_rt::cap::{ChanRecvToken, ReturnedCapability};
use kore_rt::svc::{ReturnCallError, ReturnCapability, SvcClient};
use kore_rt::transport::Transport;

use crate::service::{NetRequest, NetResponse};
use crate::wire::NetWire;

/// The rights a vif-window transfer must carry.
///
/// `ACCESS | MODIFY` because both directions write their own subregions of one
/// window: the client writes transmit slots and reads receive ones, and the
/// service does the mirror image. One window, two producers, disjoint
/// subregions.
///
/// `DERIVE` because the service makes *two* children of the delivery -- the
/// CPU leaf it validates frames through and the device leaf the controller
/// fetches them through. Both name the same frames; that identity is what
/// "zero-copy" means here, and only a lease that can be derived parents both.
///
/// **This is the single definition, and it is the contract both halves obey.**
/// It was previously spelled twice -- here without `DERIVE`, and in
/// [`crate::metal`] with it -- while the service's `RETURN_MODE` names its
/// `minimum_rights` as the `metal` spelling. The kernel refuses a retained
/// delivery below that minimum, so every real client's `OpenVif` was refused
/// at the return post, and only the row's own self-test -- whose windows come
/// from the row's own pool rather than a transfer -- ever worked. The two
/// spellings are now one so they cannot drift again (#1194).
///
/// A client's supply lease can meet this: `MmClient::supply` hands back
/// `SUPPLY_RIGHTS` (`ACCESS | MODIFY | DERIVE | REVOKE`,
/// `services/mm/src/service/mod.rs:119`), so the sublease carves legitimately
/// rather than being refused one layer further down.
pub const WINDOW_RIGHTS: Rights = Rights::ACCESS
    .union(Rights::MODIFY)
    .union(Rights::DERIVE);

/// **Format a vif window before transferring it.**
///
/// [`spsc::init`] needs exclusive access to the bytes it formats, so this is
/// the client's job and cannot be the service's. The service attaches with
/// [`spsc::consumer`]/[`spsc::producer`], which validate the magic, the ABI
/// revision and the record size — so a window this function did not produce is
/// refused at attach rather than trusted.
///
/// `window` must be at least [`WINDOW_BYTES`] and 64-byte aligned.
pub fn format_window(window: &mut [u8]) -> Result<(), RingError> {
    if window.len() < WINDOW_BYTES {
        return Err(RingError::TooSmall);
    }
    let (rings, _) = window.split_at_mut(RX_RING_AT + spsc::region_bytes::<FrameDesc>(RX_DEPTH));
    let (tx, rx) = rings.split_at_mut(RX_RING_AT);
    spsc::init::<FrameDesc>(tx, TX_DEPTH)?;
    spsc::init::<FrameDesc>(rx, RX_DEPTH)
}

/// The **client** side of the frame protocol. Replies are data; the two
/// capability movements — the window in, the doorbell recv out — are the
/// explicit methods.
#[allow(async_fn_in_trait)] // clients are generic over the impl (§4)
pub trait NetClient {
    /// Issue one plain request (anything but `OpenVif`/`Signal`).
    async fn request(&mut self, req: NetRequest) -> NetResponse;

    /// The device's link, address, and the window geometry to supply.
    async fn query(&mut self) -> NetResponse {
        self.request(NetRequest::Query).await
    }

    /// Replace the receive filter on a vif this pair already holds.
    ///
    /// `filter` is a *request*, exactly as at open: what comes back is a
    /// [`NetResponse::Filtered`] naming the vif it took effect on, or a
    /// refusal naming the power that was missing. A refusal changes nothing —
    /// the vif keeps the filter it had.
    async fn set_filter(&mut self, vif: VifId, filter: FrameFilter) -> NetResponse {
        self.request(NetRequest::SetFilter { vif, filter }).await
    }

    /// Close a vif. Its identifier stops resolving immediately, whether or not
    /// its mapping has finished retiring.
    async fn close_vif(&mut self, vif: VifId) -> NetResponse {
        self.request(NetRequest::CloseVif { vif }).await
    }

    /// One vif's counters.
    async fn stats(&mut self, vif: VifId) -> NetResponse {
        self.request(NetRequest::Stats { vif }).await
    }

    /// Transfer a formatted window and open a vif over it.
    ///
    /// `filter` is a *request*: what comes back is the intersection with the
    /// arrival pair's policy, or a refusal naming the power that was missing.
    /// On a refusal taken before the transfer moved, the capability comes back
    /// to the caller.
    async fn open_vif<C: ReturnCapability>(
        &mut self,
        filter: FrameFilter,
        window: C,
    ) -> Result<NetResponse, ReturnCallError<C>>;

    /// Mint this pair's receive doorbell ([`NetRequest::Signal`]): the
    /// possessed recv end arrives beside the reply, or `None` when the service
    /// refused (polling the ring remains lawful either way).
    async fn signal(&mut self) -> (NetResponse, Option<ChanRecvToken>);
}

/// The ring proxy (`res/USERLAND.md` §5.6): request pushed, doorbell kicked,
/// reply awaited; `Signal`'s retained grant lands in a runtime-rotated owned
/// slot and is possessed here.
pub struct RingNet<'w, 't, T: Transport> {
    /// The pair's call machine.
    pub client: SvcClient<'w, 't, T, NetWire>,
    /// The return leg's send end (`role::SVC_RETURN`, the odd index).
    pub return_tx: SlotRef,
    /// The founder-published return-landing correlation value the `OpenVif`
    /// request announces (`role::SVC_RETURN`, the even index).
    pub return_land: SlotRef,
}

impl<T: Transport> RingNet<'_, '_, T> {
    /// Ring the row's request doorbell without publishing a request.
    ///
    /// The wake a vif's producer owes the row. A transmit descriptor reaches
    /// the row's window by a shared ring and nothing else, so a client that
    /// publishes one after the row has swept and parked must say so — see
    /// [`VifIfnet::take_tx_kick`](kore_inet::VifIfnet::take_tx_kick), which
    /// says when one is owed.
    ///
    /// An ordinary droppable notice, deliberately: the descriptor is already
    /// in the ring, so a kick that does not land costs a sweep's latency and
    /// never a frame.
    pub async fn kick(&self) {
        let _ = self.client.kick().await;
    }

    async fn call(&mut self, req: &NetRequest) -> (NetResponse, Option<ReturnedCapability>) {
        self.try_call(req)
            .await
            .expect("net request path failed (torn rings/transport — a broken pair)")
    }

    async fn try_call(
        &mut self,
        req: &NetRequest,
    ) -> Option<(NetResponse, Option<ReturnedCapability>)> {
        let mut reply = self.client.call(req).await.ok()?;
        let delivered = (reply.grants > 0)
            .then(|| reply.retained.get_mut(0).and_then(Option::take))
            .flatten();
        Some((reply.response, delivered))
    }

    // ── the peer program (`res/PEER_LINK.md` P1) ──────────────────────
    //
    // The administration half of the row's mint, for the launcher that
    // builds a client face. A frame client cannot be founder-wired — both
    // static routes are refused at infrastructure build — so this is how one
    // comes to exist at all. The accel row's shape verbatim: the assembly is
    // `kore_rt`'s, and a second spelling of it here would be a second thing
    // to get wrong.

    /// Derive and convey one bundle component atomically, retaining the
    /// source lineage.
    ///
    /// [`NetResponse::PeerAccepted`](crate::NetResponse::PeerAccepted)
    /// acknowledges custody only — the mint completes after the bundle's
    /// last component, so the identity is
    /// [`take_peer_admission`](Self::take_peer_admission)'s to collect.
    ///
    /// [`StagedReturnError::AfterTransfer`] means custody **moved**: resume
    /// this exact component rather than deriving and publishing a second
    /// one, which would be refused `Duplicate` and strand the bundle.
    pub async fn admit_component_staged<F>(
        &mut self,
        part: kore_rt::svc::PeerBundlePart,
        rights: Rights,
        stage: F,
    ) -> Result<NetResponse, kore_rt::svc::StagedReturnError>
    where
        F: FnOnce(
            kore_rt::svc::ReturnDeriver<'_>,
        ) -> Result<kore_rt::svc::StagedReturn, kore_rt::svc::ReturnDeriveError>,
    {
        let request = NetRequest::AdmitPeer {
            part,
            announced: self.return_land,
        };
        let reply = self
            .client
            .call_with_staged_return(&request, self.return_tx, rights, stage)
            .await?;
        Ok(reply.response)
    }

    /// **Open a vif over a sublease of a window this client keeps.**
    ///
    /// [`NetClient::open_vif`] moves the capability it is given, which costs
    /// the caller the window it must go on writing transmit slots into. A
    /// client therefore derives a *sublease* and conveys that, retaining the
    /// source — the discipline the bundle components already use, and the
    /// reason `format_window` is the client's job: the bytes must be
    /// formatted before the service attaches to them, and the client must
    /// still reach them afterwards.
    pub async fn open_vif_staged<F>(
        &mut self,
        filter: FrameFilter,
        stage: F,
    ) -> Result<NetResponse, kore_rt::svc::StagedReturnError>
    where
        F: FnOnce(
            kore_rt::svc::ReturnDeriver<'_>,
        ) -> Result<kore_rt::svc::StagedReturn, kore_rt::svc::ReturnDeriveError>,
    {
        let request = NetRequest::OpenVif {
            filter,
            announced: self.return_land,
        };
        let reply = self
            .client
            .call_with_staged_return(&request, self.return_tx, WINDOW_RIGHTS, stage)
            .await?;
        Ok(reply.response)
    }

    /// Resume **only this component** after an uncertain transport result.
    ///
    /// A different pending request is left untouched for its original owner:
    /// re-asking the wrong one would publish a component whose capability
    /// already moved.
    pub async fn resume_component(
        &mut self,
        part: kore_rt::svc::PeerBundlePart,
    ) -> Result<NetResponse, kore_rt::svc::CallError> {
        let request = NetRequest::AdmitPeer {
            part,
            announced: self.return_land,
        };
        self.client
            .resume_exact(&request)
            .await
            .map(|reply| reply.response)
    }

    /// Collect the host-authored identity of the face most recently
    /// assembled.
    pub async fn take_peer_admission(&mut self) -> Result<NetResponse, kore_rt::svc::CallError> {
        self.client
            .call(&NetRequest::TakePeerAdmission)
            .await
            .map(|reply| reply.response)
    }

    /// One peer-control call, observed as the caller asks.
    ///
    /// A control reply carries no grant; one that does is a protocol break,
    /// and the retained owner is poisoned rather than optimistically bound.
    async fn peer_control(
        &mut self,
        request: NetRequest,
        observation: kore_rt::svc::Observation,
    ) -> Result<NetResponse, kore_rt::svc::CallError> {
        let reply = self.client.call_exact(&request, observation).await?;
        if reply.grants != 0 {
            self.client.poison_retained(reply.retained);
            return Err(kore_rt::svc::CallError::Wire);
        }
        Ok(reply.response)
    }

    /// Collect an admission verdict under `observation`.
    pub async fn take_peer_admission_observed(
        &mut self,
        observation: kore_rt::svc::Observation,
    ) -> Result<NetResponse, kore_rt::svc::CallError> {
        self.peer_control(NetRequest::TakePeerAdmission, observation)
            .await
    }

    /// Ask the row to retire one exact minted generation.
    pub async fn retire_peer_observed(
        &mut self,
        peer: kore_rt::svc::PeerId,
        observation: kore_rt::svc::Observation,
    ) -> Result<NetResponse, kore_rt::svc::CallError> {
        self.peer_control(NetRequest::RetirePeer { peer }, observation)
            .await
    }

    /// Ask whether that generation's cell is free again.
    pub async fn confirm_retired_observed(
        &mut self,
        peer: kore_rt::svc::PeerId,
        observation: kore_rt::svc::Observation,
    ) -> Result<NetResponse, kore_rt::svc::CallError> {
        self.peer_control(NetRequest::ConfirmRetired { peer }, observation)
            .await
    }

    /// Retire the minted pair this client holds.
    ///
    /// Only lawful from a minted face: a fixed pair is the founder's.
    pub async fn close_peer(&mut self) -> Result<NetResponse, kore_rt::svc::CallError> {
        self.client
            .call(&NetRequest::ClosePeer)
            .await
            .map(|reply| reply.response)
    }

    /// [`NetClient::request`] without the panic: `None` when the pair's
    /// transport failed, for a caller — the launcher federating a snapshot —
    /// that must outlive a torn row rather than die with it. A grant that
    /// arrives on a plain verb is dropped, which revokes it, exactly as in
    /// `request`.
    pub async fn try_request(&mut self, req: NetRequest) -> Option<NetResponse> {
        let (response, delivered) = self.try_call(&req).await?;
        drop(delivered);
        Some(response)
    }
}

impl<T: Transport> NetClient for RingNet<'_, '_, T> {
    async fn request(&mut self, req: NetRequest) -> NetResponse {
        let (response, delivered) = self.call(&req).await;
        // No plain verb may carry a grant; a delivery here is a protocol break
        // and dropping the owner revokes it — fail-closed, so no authority
        // leaks in on an unexpected grant.
        drop(delivered);
        response
    }

    async fn open_vif<C: ReturnCapability>(
        &mut self,
        filter: FrameFilter,
        window: C,
    ) -> Result<NetResponse, ReturnCallError<C>> {
        let request = NetRequest::OpenVif {
            filter,
            announced: self.return_land,
        };
        let reply = self
            .client
            .call_with_return(&request, self.return_tx, window, WINDOW_RIGHTS)
            .await?;
        Ok(reply.response)
    }

    async fn signal(&mut self) -> (NetResponse, Option<ChanRecvToken>) {
        let (response, delivered) = self.call(&NetRequest::Signal).await;
        let recv = delivered.and_then(|owner| {
            // Kind and rights are checked at possession; a mismatched delivery
            // drops through the owner (revoked, slot reopened) rather than
            // being optimistically bound.
            owner.try_into_chan_recv(Rights::ACCESS).ok()
        });
        (response, recv)
    }
}

/// **Where a `PROFILE_NETWORK` process finds its private frame face.**
///
/// The cross-boundary layout shared by the launcher that stamps the child's
/// block and the child that binds it, kept beside [`RingNet`] for the reason
/// the other rows keep theirs beside their clients: a topology change should
/// be one compile-time edit in one contract, not two matching integer lists
/// in exec and the program.
///
/// The `SVC_*` namespaces are crowded and a single launch may hold several
/// faces at once, so every index here is the next free one across every face
/// a launch can compose — the filesystem's (`SVC_SEND@0`, `SVC_RING` 0/1,
/// `SVC_RETURN` 0/1), the NPU's, the GPU's and an engine's mm face. Exec
/// pins the disjointness where it sees all of them.
pub mod process_face {
    /// Frame-service request doorbell (`role::SVC_SEND`).
    pub const REQUEST_KICK: u16 = 4;
    /// Request/reply ring backing (`role::SVC_RING`).
    pub const CONTROL_RING: u16 = 5;
    /// Receiver for the row's reply grants (`role::SVC_GRANT`) — the receive
    /// doorbell `Signal` mints rides this lane.
    pub const GRANT_RECV: u16 = 4;
    /// Client-to-service retained return sender (`role::SVC_RETURN`).
    ///
    /// A frame face needs a return leg where power's does not: `open_vif`
    /// transfers the client's own formatted window on it.
    pub const RETURN_SEND: u16 = 6;
    /// Correlation datum for the row's retained return landing.
    pub const RETURN_LAND: u16 = 7;
}

/// A process's own frame face, plus the affine ring lease that keeps its
/// mapped control window alive.
#[cfg(target_os = "none")]
pub struct ProcessNet<'t> {
    /// The face itself.
    pub client: RingNet<'static, 't, kore_rt::transport::TrapTransport>,
    /// Dropping this retires the control window, so it is held for the
    /// process's life rather than returned to the caller.
    _ring: kore_rt::cap::LeaseToken,
}

/// Attach the minted frame face from a profiled process boot block.
///
/// `None` on a launch that carries no face — which is every launch today
/// that exec did not mint one for, and is the posture a program must handle
/// rather than assume: a machine with no adaptor founds the row degraded and
/// exec mints nothing.
#[cfg(target_os = "none")]
pub fn attach_process_face<'t>(
    rt: &kore_rt::entry::Rt,
    transport: &'t kore_rt::transport::TrapTransport,
) -> Option<ProcessNet<'t>> {
    use kore_abi::task::role;
    use kore_rt::cap::verbs;
    use kore_rt::service::MAX_GRANTS;
    use kore_rt::svc::{ClientEndpoints, GrantPin, SvcClient, client_rings};

    let boot = rt.bootstrap();
    let ring = verbs::take_lease(boot.handle(role::SVC_RING, process_face::CONTROL_RING)?)?;
    let window = verbs::map_held_rw(ring.slot())?;
    // SAFETY: this process is the sole client-side attachment over the
    // task-lifetime ring lease exec delivered, and it lives as long as the
    // lease held beside it.
    let window: &'static mut [u8] =
        unsafe { core::slice::from_raw_parts_mut(window.cast::<u8>().as_ptr(), window.len()) };
    let rings = client_rings::<NetWire>(
        window,
        <NetWire as kore_rt::svc::SvcProtocol>::REQUEST_CAPACITY,
        <NetWire as kore_rt::svc::SvcProtocol>::RESPONSE_CAPACITY,
    )
    .ok()?;
    let kick = boot.handle(role::SVC_SEND, process_face::REQUEST_KICK)?;
    let grant = boot.handle(role::SVC_GRANT, process_face::GRANT_RECV)?;
    let return_tx = boot.handle(role::SVC_RETURN, process_face::RETURN_SEND)?;
    let return_land = SlotRef::from_wire(boot.datum(role::SVC_RETURN, process_face::RETURN_LAND)?);
    let mut grants = [const { None }; MAX_GRANTS];
    grants[0] = Some(GrantPin::retained(grant));
    Some(ProcessNet {
        client: RingNet {
            client: SvcClient::new(transport, rings, ClientEndpoints { kick, grants })
                .with_reply_wait(rt.doorbell()?),
            return_tx,
            return_land,
        },
        _ring: ring,
    })
}
