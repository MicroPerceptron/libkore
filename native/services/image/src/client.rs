//! `client` — the image client seam: the loader channel
//! (`res/USERLAND.md` §4/§5.6).
//!
//! What the linker stub's world issues [`ImageRequest`]s through and
//! awaits an [`ImageResponse`] (plus its delivered lease caps) back —
//! the `role::LOADER` channel to the image service.  [`ImageClient`]
//! is the §4 pinned async trait and [`RingImage`] is the born-async ring +
//! channel proxy (§5.6). The early linker (`kore-ld`'s `LinkWorld`) is
//! synchronous by nature and uses its own explicit test worlds; the real
//! userland service path here always speaks through the possessed pair.

use kore_abi::image::{ImageName, ImageSourceMetadata};
use kore_abi::session::{SlotRef, Status};

use kore_rt::cap::{CHANNEL_ENDPOINT_RIGHTS, ChanRecvToken, StashedLease};
use kore_rt::svc::{
    CallError, RetainedReturnCallError, ReturnCallError, ReturnCapability, SvcClient, SvcProtocol,
};
use kore_rt::transport::Transport;

use crate::service::{ImageRequest, ImageResponse, ImageService, MAX_IMAGE_SEGS};

/// An `Acquire`/`AcquireEntry`'s delivered authority: at most [`MAX_IMAGE_SEGS`]
/// runtime-owned lease tokens, in emission order (`None` past the granted
/// count, and all `None` on any refusal — a refusal delivers no authority).
/// Each token keeps its dynamically rotated arena name reserved until its
/// lease is retired, so a later `Acquire` cannot overwrite live authority.
pub type ImageGrants = [Option<StashedLease>; MAX_IMAGE_SEGS];

/// An archive member's bytes and its one actual read-lease owner.
#[derive(Debug)]
pub struct ExportedImage {
    pub len: u64,
    pub lease: StashedLease,
}

/// A `FaceBank`'s delivered authority (slice C): exactly
/// [`MAX_IMAGE_SEGS`] grant-channel recv ends on success, all `None` on
/// a refusal.  The parent conveys each whole at the spawn
/// ([`ChanRecvToken`] is a [`ReturnCapability`](kore_rt::svc::ReturnCapability))
/// — no derive, so the child's landings never depend on a leaf the
/// parent keeps.
pub type FaceRecvs = [Option<ChanRecvToken>; MAX_IMAGE_SEGS];

/// Convert one complete grant bank or restore every owner at its original
/// index. Retyping is synchronous and reversible: an error must leave native
/// retirement to the pair's retained cleanup, including earlier successes.
fn convert_grant_bank<C, R, const N: usize>(
    retained: &mut [Option<C>],
    mut convert: impl FnMut(C) -> Result<R, C>,
    mut erase: impl FnMut(R) -> C,
) -> Result<[Option<R>; N], ()> {
    let retained = retained.get_mut(..N).ok_or(())?;
    let mut converted = [const { None }; N];
    for i in 0..N {
        if let Some(owner) = retained[i].take() {
            match convert(owner) {
                Ok(owner) => {
                    converted[i] = Some(owner);
                    continue;
                }
                Err(owner) => retained[i] = Some(owner),
            }
        }
        for (dst, owner) in retained.iter_mut().zip(&mut converted) {
            if let Some(owner) = owner.take() {
                *dst = Some(erase(owner));
            }
        }
        return Err(());
    }
    Ok(converted)
}

/// The complete client side of one freshly minted loader face. The ring
/// sub-lease and every grant-channel recv arrive in one atomic service
/// outcome; keeping them in one affine value makes partial possession
/// unrepresentable on the success path.
#[derive(Debug)]
pub struct FaceAuthority {
    ring: StashedLease,
    recvs: FaceRecvs,
}

impl FaceAuthority {
    /// Split the face into the owners exec retains until child reap.
    pub fn into_parts(self) -> (StashedLease, FaceRecvs) {
        (self.ring, self.recvs)
    }
}

/// The **client** side of the image protocol (`res/USERLAND.md` §4,
/// pinned): issue one request, await the data reply beside its
/// delivered lease grants.
#[allow(async_fn_in_trait)] // clients are generic over the impl (§4)
pub trait ImageClient {
    /// Issue one request; the grants arrive **possessed** — view-ready
    /// tokens beside the data reply.
    ///
    /// Infrastructure failures remain distinct from the image service's
    /// closed refusal vocabulary. A torn pair must neither panic the client
    /// nor masquerade as a lessor refusal.
    async fn request(
        &mut self,
        req: ImageRequest,
    ) -> Result<(ImageResponse, ImageGrants), CallError>;

    /// Continue the sole request already published by this pair after a
    /// transport/backpressure error or cancellation boundary.
    async fn resume_request(&mut self) -> Result<(ImageResponse, ImageGrants), CallError>;
}

/// The authority-bearing registration face. Kept separate from
/// [`ImageClient`] because linker clients only acquire existing images, while
/// registrars additionally hold the pair's return send and an image lease.
#[allow(async_fn_in_trait)]
pub trait ImageRegistrar {
    /// Transfer `source` into the service's standing retained-return post,
    /// then publish the `Register` ring record and await its data reply.
    async fn register<C: ReturnCapability>(
        &mut self,
        name: ImageName,
        metadata: ImageSourceMetadata,
        source: C,
    ) -> Result<ImageResponse, ReturnCallError<C>>;

    /// Continue a registration whose authority transfer and request publish
    /// completed but whose kick/reply leg was interrupted. A new registration
    /// must not be published until this uncorrelated SPSC reply is drained.
    async fn resume_registration(&mut self) -> Result<ImageResponse, CallError>;
}

/// The **ring + channel proxy** (`res/USERLAND.md` §5.6): the image
/// client over the pair's shared window.  An `Acquired` reply announces
/// ≤ [`MAX_IMAGE_SEGS`] transfers — one per grant channel — which this
/// client takes out of its pre-pinned reply posts
/// as runtime-owned [`StashedLease`] values. An `Acquire` grant is an
/// attenuated `MemLease`, not a `LeasedRegion` view; retaining the dynamic
/// landing owner prevents later replies from overwriting it.
///
/// Infrastructure failures remain the outer [`CallError`]; the closed
/// [`ImageRefusal`] vocabulary describes only an answer received from the
/// lessor. This keeps a torn pair from either panicking the client or
/// masquerading as a lessor refusal and corrupting authority accounting.
///
/// [`ImageRefusal`]: kore_abi::image::ImageRefusal
pub struct RingImage<'w, 't, T: Transport> {
    pair: SvcClient<'w, 't, T, ImageService>,
    registration: Option<RegistrationLeg>,
}

#[derive(Clone, Copy)]
struct RegistrationLeg {
    return_tx: SlotRef,
    return_land: SlotRef,
}

impl<'w, 't, T: Transport> RingImage<'w, 't, T> {
    /// Assemble an acquire-only image client. No return-channel authority is
    /// required to load already-registered code.
    pub fn new(pair: SvcClient<'w, 't, T, ImageService>) -> Self {
        Self {
            pair,
            registration: None,
        }
    }

    /// Assemble an image client that can also register new immutable sources.
    ///
    /// The return leg is founder-published routing authority for this exact
    /// pair. Keeping it inside the client prevents acquire and registration
    /// from constructing competing SPSC state machines over one ring window.
    pub fn with_registration(
        pair: SvcClient<'w, 't, T, ImageService>,
        return_tx: SlotRef,
        return_land: SlotRef,
    ) -> Self {
        Self {
            pair,
            registration: Some(RegistrationLeg {
                return_tx,
                return_land,
            }),
        }
    }

    /// The request-doorbell authority of this exact image-service face.
    ///
    /// A founder that delegates a bound loader face to a child derives the
    /// child's request endpoint from the same live endpoint used by this
    /// client. Keeping this coordinate behind the typed client prevents a
    /// second, hand-maintained bootstrap/convey coordinate from becoming an
    /// alternate owner of the loader topology.
    pub fn request_kick(&self) -> SlotRef {
        self.pair.request_kick()
    }

    pub fn is_idle(&self) -> bool {
        self.pair.is_idle()
    }

    /// Keep the source in its caller's persistent record until the native
    /// transfer commits. Exact transfer preserves its rights; ImageService
    /// validates the registration lease contract at possession.
    pub async fn register_retained(
        &mut self,
        name: ImageName,
        metadata: ImageSourceMetadata,
        source: &mut Option<StashedLease>,
    ) -> Result<ImageResponse, RetainedReturnCallError> {
        let Some(leg) = self.registration else {
            return Err(RetainedReturnCallError::NotMoved(CallError::Refused(
                Status::Denied,
            )));
        };
        let request = ImageRequest::Register {
            name,
            slot: leg.return_land,
            source: metadata,
        };
        let reply = self
            .pair
            .call_with_exact_retained_return(&request, leg.return_tx, source)
            .await?;
        finish_registration(&mut self.pair, reply).map_err(RetainedReturnCallError::AfterTransfer)
    }

    pub async fn resume_registration_bounded(
        &mut self,
        name: ImageName,
        metadata: ImageSourceMetadata,
        budget_ns: u64,
    ) -> Result<ImageResponse, CallError> {
        let leg = self
            .registration
            .ok_or(CallError::Refused(Status::Denied))?;
        let request = ImageRequest::Register {
            name,
            slot: leg.return_land,
            source: metadata,
        };
        let reply = self.pair.resume_exact_bounded(&request, budget_ns).await?;
        finish_registration(&mut self.pair, reply)
    }

    /// Cleanup for the sole registrar while it still owns the latest name.
    /// The caller excludes new registrations and acquisitions until this
    /// exact request has settled; use bound-generation retirement otherwise.
    pub async fn retire_registration_bounded(
        &mut self,
        name: ImageName,
        resume: bool,
        budget_ns: u64,
    ) -> Result<ImageResponse, CallError> {
        if self.registration.is_none() {
            return Err(CallError::Refused(Status::Denied));
        }
        let request = ImageRequest::Retire { name };
        let reply = if resume {
            self.pair.resume_exact_bounded(&request, budget_ns).await?
        } else {
            self.pair.call_bounded(&request, budget_ns).await?
        };
        if reply.grants == 0
            && reply.delivered.iter().all(Option::is_none)
            && reply.retained.iter().all(Option::is_none)
            && matches!(
                reply.response,
                ImageResponse::Retired | ImageResponse::Refused(_)
            )
        {
            Ok(reply.response)
        } else {
            self.pair.poison_retained(reply.retained);
            Err(CallError::Wire)
        }
    }

    /// Start an export on this pair. Cancellation leaves its original request
    /// in the pair; a caller must retain `name` before awaiting this method.
    pub async fn export(&mut self, name: ImageName) -> Result<Option<ExportedImage>, CallError> {
        let reply = self.pair.call(&ImageRequest::Export { name }).await?;
        self.finish_export(reply)
    }

    /// Collect only the original export, without consuming another operation
    /// or an export of another member. The slice bounds observation, not the
    /// lifetime of the original operation or its delivered authority.
    pub async fn resume_export_bounded(
        &mut self,
        name: ImageName,
        budget_ns: u64,
    ) -> Result<Option<ExportedImage>, CallError> {
        let reply = self
            .pair
            .resume_exact_bounded(&ImageRequest::Export { name }, budget_ns)
            .await?;
        self.finish_export(reply)
    }

    fn finish_export(
        &mut self,
        mut reply: kore_rt::svc::CallReply<ImageResponse>,
    ) -> Result<Option<ExportedImage>, CallError> {
        if reply.delivered.iter().any(Option::is_some)
            || reply.retained.iter().skip(1).any(Option::is_some)
        {
            self.pair.poison_retained(reply.retained);
            return Err(CallError::Wire);
        }
        match reply.response {
            ImageResponse::Exported { len } if len != 0 && reply.grants == 1 => {
                let Some(owner) = reply.retained[0].take() else {
                    self.pair.poison_retained(reply.retained);
                    return Err(CallError::Wire);
                };
                match owner.try_into_stashed_lease(ImageService::LEASE_RIGHTS) {
                    Ok(lease) => Ok(Some(ExportedImage { len, lease })),
                    Err(owner) => {
                        reply.retained[0] = Some(owner);
                        self.pair.poison_retained(reply.retained);
                        Err(CallError::Wire)
                    }
                }
            }
            ImageResponse::Refused(_) if reply.grants == 0 && reply.retained[0].is_none() => {
                Ok(None)
            }
            _ => {
                self.pair.poison_retained(reply.retained);
                Err(CallError::Wire)
            }
        }
    }
}

/// Capability-bearing image registrar. Possession of this type proves the
/// caller has both an image-service pair and the return send needed to convey
/// a new image parent; acquire-only loaders use [`RingImage`] instead.
pub struct RingRegistrar<'w, 't, T: Transport> {
    pair: SvcClient<'w, 't, T, ImageService>,
    return_tx: SlotRef,
    return_land: SlotRef,
}

impl<'w, 't, T: Transport> RingRegistrar<'w, 't, T> {
    /// Assemble a registrar from its pair and founder-published return leg.
    pub fn new(
        pair: SvcClient<'w, 't, T, ImageService>,
        return_tx: SlotRef,
        return_land: SlotRef,
    ) -> Self {
        Self {
            pair,
            return_tx,
            return_land,
        }
    }
}

impl<T: Transport> ImageClient for RingImage<'_, '_, T> {
    async fn request(
        &mut self,
        req: ImageRequest,
    ) -> Result<(ImageResponse, ImageGrants), CallError> {
        // `Register` needs the return leg this client does not hold; the face
        // verbs deliver heterogeneous authority, so their typed methods own
        // conversion. Letting either ride the lease path would poison the pair
        // on a lawful reply.
        if matches!(
            req,
            ImageRequest::Register { .. }
                | ImageRequest::FaceMint { .. }
                | ImageRequest::FaceFork { .. }
                | ImageRequest::FaceBank { .. }
        ) {
            return Err(CallError::Refused(Status::Denied));
        }
        let reply = self.pair.call(&req).await?;
        self.finish_reply(reply)
    }

    async fn resume_request(&mut self) -> Result<(ImageResponse, ImageGrants), CallError> {
        let reply = self.pair.resume().await?;
        self.finish_reply(reply)
    }
}

impl<T: Transport> RingImage<'_, '_, T> {
    /// Acquire one named image with bounded observation of this request.
    pub async fn acquire_bounded(
        &mut self,
        name: ImageName,
        budget_ns: u64,
    ) -> Result<(ImageResponse, ImageGrants), CallError> {
        let reply = self
            .pair
            .call_bounded(&ImageRequest::Acquire { name }, budget_ns)
            .await?;
        self.finish_reply(reply)
    }

    /// Resume only the original named acquisition without publishing another.
    pub async fn resume_acquire_bounded(
        &mut self,
        name: ImageName,
        budget_ns: u64,
    ) -> Result<(ImageResponse, ImageGrants), CallError> {
        let reply = self
            .pair
            .resume_exact_bounded(&ImageRequest::Acquire { name }, budget_ns)
            .await?;
        self.finish_reply(reply)
    }

    /// Retire exactly one held face, or resume its original retirement.
    pub async fn retire_face_bounded(
        &mut self,
        face: u32,
        budget_ns: u64,
    ) -> Result<ImageResponse, CallError> {
        self.retire_identity_bounded(ImageRequest::FaceRetire { face }, budget_ns)
            .await
    }

    /// Retire the immutable generation bound by this retained template.
    pub async fn retire_bound_bounded(
        &mut self,
        template: u32,
        budget_ns: u64,
    ) -> Result<ImageResponse, CallError> {
        self.retire_identity_bounded(ImageRequest::RetireBound { template }, budget_ns)
            .await
    }

    async fn retire_identity_bounded(
        &mut self,
        request: ImageRequest,
        budget_ns: u64,
    ) -> Result<ImageResponse, CallError> {
        let reply = if self.pair.is_idle() {
            self.pair.call_bounded(&request, budget_ns).await?
        } else {
            self.pair.resume_exact_bounded(&request, budget_ns).await?
        };
        if reply.grants == 0
            && reply.delivered.iter().all(Option::is_none)
            && reply.retained.iter().all(Option::is_none)
            && matches!(
                reply.response,
                ImageResponse::Retired | ImageResponse::Refused(_)
            )
        {
            Ok(reply.response)
        } else {
            self.pair.poison_retained(reply.retained);
            Err(CallError::Wire)
        }
    }

    /// Validate `target`, mint one loader face, and possess its complete
    /// client authority in one reply. A refusal returns `None`; a successful
    /// `Face` response carries exactly one ring sub-lease followed by
    /// [`MAX_IMAGE_SEGS`] recv endpoints.
    pub async fn face_mint(
        &mut self,
        target: ImageName,
    ) -> Result<(ImageResponse, Option<FaceAuthority>), CallError> {
        let reply = self.pair.call(&ImageRequest::FaceMint { target }).await?;
        self.finish_face(reply)
    }

    /// Mint a fresh loader face bound to a retained generation template.
    pub async fn face_fork(
        &mut self,
        template: u32,
    ) -> Result<(ImageResponse, Option<FaceAuthority>), CallError> {
        let reply = self.pair.call(&ImageRequest::FaceFork { template }).await?;
        self.finish_face(reply)
    }

    /// Collect only the original template fork; never publish a replacement.
    /// The caller retains its pending template and any late face for cleanup.
    pub async fn resume_face_fork_bounded(
        &mut self,
        template: u32,
        budget_ns: u64,
    ) -> Result<(ImageResponse, Option<FaceAuthority>), CallError> {
        let reply = self
            .pair
            .resume_exact_bounded(&ImageRequest::FaceFork { template }, budget_ns)
            .await?;
        self.finish_face(reply)
    }

    fn finish_face(
        &mut self,
        mut reply: kore_rt::svc::CallReply<ImageResponse>,
    ) -> Result<(ImageResponse, Option<FaceAuthority>), CallError> {
        let is_face = matches!(reply.response, ImageResponse::Face { .. });
        if (!is_face && !matches!(reply.response, ImageResponse::Refused(_)))
            || (is_face && usize::from(reply.grants) != MAX_IMAGE_SEGS + 1)
            || (!is_face && reply.grants != 0)
            || (reply.grants == 0
                && (reply.delivered.iter().any(Option::is_some)
                    || reply.retained.iter().any(Option::is_some)))
        {
            self.pair.poison_retained(reply.retained);
            return Err(CallError::Wire);
        }
        if reply.grants == 0 {
            return Ok((reply.response, None));
        }
        let response = reply.response;
        let Some(owner) = reply.retained[0].take() else {
            self.pair.poison_retained(reply.retained);
            return Ok((response, None));
        };
        let ring = match owner.try_into_stashed_lease(ImageService::LEASE_RIGHTS) {
            Ok(ring) => ring,
            Err(owner) => {
                reply.retained[0] = Some(owner);
                self.pair.poison_retained(reply.retained);
                return Ok((response, None));
            }
        };
        let recvs = match convert_grant_bank(
            &mut reply.retained[1..],
            |owner| owner.try_into_chan_recv(CHANNEL_ENDPOINT_RIGHTS),
            ChanRecvToken::into_returned_capability,
        ) {
            Ok(recvs) => recvs,
            Err(()) => {
                reply.retained[0] = Some(ring.into_returned_capability());
                self.pair.poison_retained(reply.retained);
                // Exec still needs the authenticated face identity to retire
                // the service allocation even though possession was partial.
                return Ok((response, None));
            }
        };
        Ok((response, Some(FaceAuthority { ring, recvs })))
    }

    /// Collect a minted face's grant-channel recv ends (slice C): issue
    /// `FaceBank` and take the [`MAX_IMAGE_SEGS`] delivered recvs as
    /// typed owners.  A refusal delivers none; a malformed reply (wrong
    /// kind or count in the posts) poisons the pair and retains every owner
    /// until local retirement is proven. The parent retires the remote face.
    pub async fn face_bank(&mut self, face: u32) -> Result<(ImageResponse, FaceRecvs), CallError> {
        let reply = self.pair.call(&ImageRequest::FaceBank { face }).await?;
        self.finish_bank(reply)
    }

    fn finish_bank(
        &mut self,
        mut reply: kore_rt::svc::CallReply<ImageResponse>,
    ) -> Result<(ImageResponse, FaceRecvs), CallError> {
        let count = match reply.response {
            ImageResponse::Banked => MAX_IMAGE_SEGS,
            ImageResponse::Refused(_) => 0,
            _ => {
                self.pair.poison_retained(reply.retained);
                return Err(CallError::Wire);
            }
        };
        if usize::from(reply.grants) != count
            || reply.retained[count..].iter().any(Option::is_some)
            || reply.delivered[count..].iter().any(Option::is_some)
        {
            self.pair.poison_retained(reply.retained);
            return Err(CallError::Wire);
        }
        if count == 0 {
            return Ok((reply.response, [const { None }; MAX_IMAGE_SEGS]));
        }
        let recvs = match convert_grant_bank(
            &mut reply.retained,
            |owner| owner.try_into_chan_recv(CHANNEL_ENDPOINT_RIGHTS),
            ChanRecvToken::into_returned_capability,
        ) {
            Ok(recvs) => recvs,
            Err(()) => {
                self.pair.poison_retained(reply.retained);
                return Err(CallError::Wire);
            }
        };
        Ok((reply.response, recvs))
    }

    fn finish_reply(
        &mut self,
        mut reply: kore_rt::svc::CallReply<ImageResponse>,
    ) -> Result<(ImageResponse, ImageGrants), CallError> {
        let mut leases: ImageGrants = [const { None }; MAX_IMAGE_SEGS];
        for k in 0..usize::from(reply.grants) {
            let Some(owner) = reply.retained.get_mut(k).and_then(Option::take) else {
                return self.malformed_reply(reply, leases);
            };
            let lease = match owner.try_into_stashed_lease(ImageService::LEASE_RIGHTS) {
                Ok(lease) => lease,
                Err(owner) => {
                    reply.retained[k] = Some(owner);
                    return self.malformed_reply(reply, leases);
                }
            };
            let Some(dst) = leases.get_mut(k) else {
                reply.retained[k] = Some(lease.into_returned_capability());
                return self.malformed_reply(reply, leases);
            };
            *dst = Some(lease);
        }
        Ok((reply.response, leases))
    }

    fn malformed_reply(
        &mut self,
        mut reply: kore_rt::svc::CallReply<ImageResponse>,
        mut leases: ImageGrants,
    ) -> Result<(ImageResponse, ImageGrants), CallError> {
        for lease in leases.iter_mut().filter_map(Option::take) {
            if let Some(dst) = reply.retained.iter_mut().find(|entry| entry.is_none()) {
                *dst = Some(lease.into_returned_capability());
            } else {
                drop(lease);
            }
        }
        self.pair.poison_retained(reply.retained);
        Err(CallError::Wire)
    }
}

impl<T: Transport> ImageRegistrar for RingRegistrar<'_, '_, T> {
    async fn register<C: ReturnCapability>(
        &mut self,
        name: ImageName,
        metadata: ImageSourceMetadata,
        source: C,
    ) -> Result<ImageResponse, ReturnCallError<C>> {
        let request = ImageRequest::Register {
            name,
            slot: self.return_land,
            source: metadata,
        };
        let reply = self
            .pair
            .call_with_return(
                &request,
                self.return_tx,
                source,
                ImageService::REGISTER_SOURCE_RIGHTS,
            )
            .await?;
        finish_registration(&mut self.pair, reply).map_err(ReturnCallError::AfterTransfer)
    }

    async fn resume_registration(&mut self) -> Result<ImageResponse, CallError> {
        resume_pending_registration(&mut self.pair, self.return_land).await
    }
}

impl<T: Transport> ImageRegistrar for RingImage<'_, '_, T> {
    async fn register<C: ReturnCapability>(
        &mut self,
        name: ImageName,
        metadata: ImageSourceMetadata,
        source: C,
    ) -> Result<ImageResponse, ReturnCallError<C>> {
        let Some(leg) = self.registration else {
            return Err(ReturnCallError::NotMoved {
                error: CallError::Refused(Status::Denied),
                capability: source,
            });
        };
        let request = ImageRequest::Register {
            name,
            slot: leg.return_land,
            source: metadata,
        };
        let reply = self
            .pair
            .call_with_return(
                &request,
                leg.return_tx,
                source,
                ImageService::REGISTER_SOURCE_RIGHTS,
            )
            .await?;
        finish_registration(&mut self.pair, reply).map_err(ReturnCallError::AfterTransfer)
    }

    async fn resume_registration(&mut self) -> Result<ImageResponse, CallError> {
        let leg = self
            .registration
            .ok_or(CallError::Refused(Status::Denied))?;
        resume_pending_registration(&mut self.pair, leg.return_land).await
    }
}

async fn resume_pending_registration<T: Transport>(
    pair: &mut SvcClient<'_, '_, T, ImageService>,
    return_land: SlotRef,
) -> Result<ImageResponse, CallError> {
    let reply = if pair.outstanding_request().is_none() {
        // Idle/poisoned only: preserve SvcClient's quarantine-cleanup retry.
        // There is no foreign request here to consume as a registration.
        pair.resume().await?
    } else {
        let request = pending_registration(pair, return_land)?;
        pair.resume_exact(&request).await?
    };
    finish_registration(pair, reply)
}

fn pending_registration<T: Transport>(
    pair: &SvcClient<'_, '_, T, ImageService>,
    return_land: SlotRef,
) -> Result<ImageRequest, CallError> {
    let wire = pair.outstanding_request().ok_or(if pair.is_idle() {
        CallError::NoOutstanding
    } else {
        CallError::Poisoned
    })?;
    match ImageService::decode_req(wire) {
        Some(request @ ImageRequest::Register { slot, .. }) if slot == return_land => Ok(request),
        _ => Err(CallError::Outstanding),
    }
}

fn finish_registration<T: Transport>(
    pair: &mut SvcClient<'_, '_, T, ImageService>,
    reply: kore_rt::svc::CallReply<ImageResponse>,
) -> Result<ImageResponse, CallError> {
    let accepted = if reply.grants != 0
        || reply.delivered.iter().any(Option::is_some)
        || reply.retained.iter().any(Option::is_some)
    {
        None
    } else {
        match reply.response {
            response @ ImageResponse::Registered { module } if module.validate() => Some(response),
            response @ ImageResponse::Refused(_) => Some(response),
            _ => None,
        }
    };
    if let Some(response) = accepted {
        return Ok(response);
    }
    pair.poison_retained(reply.retained);
    Err(CallError::Wire)
}

#[cfg(test)]
mod export_tests {
    use super::*;
    use core::cell::Cell;
    use kore_abi::{image::ImageRefusal, session::Committed};
    use kore_rt::svc::{CallReply, ClientEndpoints, REP_CAP, REQ_CAP, ServeRings, SvcProtocol};

    struct ReplyTransport(Cell<bool>, Cell<usize>);
    impl Transport for ReplyTransport {
        type Error = ();
        async fn commit(&self, s: &kore_rt::session::SessionBuilder) -> Result<Committed, ()> {
            self.commit_now(s)
        }
        fn commit_now(&self, _: &kore_rt::session::SessionBuilder) -> Result<Committed, ()> {
            self.1.set(self.1.get() + 1);
            if self.0.get() {
                Ok(Committed::from_raw(Status::Ok.encode(0)))
            } else {
                Err(())
            }
        }
    }
    const WINDOW_BYTES: usize = kore_rt::svc::window_bytes::<ImageService>(REQ_CAP, REP_CAP);
    #[repr(align(64))]
    struct Window([u8; WINDOW_BYTES]);
    fn with_pair(
        test: impl FnOnce(
            &mut RingImage<'_, '_, ReplyTransport>,
            &mut ServeRings<'_, ImageService>,
            &ReplyTransport,
        ),
    ) {
        let transport = ReplyTransport(Cell::new(true), Cell::new(0));
        let mut window = Window([0; WINDOW_BYTES]);
        let (client, mut serve) =
            kore_rt::svc::split_window::<ImageService>(&mut window.0, REQ_CAP, REP_CAP).unwrap();
        let mut image = RingImage::new(SvcClient::new(
            &transport,
            client,
            ClientEndpoints {
                kick: SlotRef::from_wire(100),
                grants: [None; kore_rt::service::MAX_GRANTS],
            },
        ));
        test(&mut image, &mut serve, &transport);
    }
    fn run<F: core::future::Future>(future: F) -> F::Output {
        let park = kore_rt::transport::TrapTransport::new();
        kore_rt::Executor::from_standing(
            &park,
            kore_rt::executor::Doorbell {
                chan: 0,
                recv: SlotRef::from_wire(101),
                land: SlotRef::from_wire(102),
                recv_tag: 7,
            },
        )
        .run(future)
    }
    fn reply(response: ImageResponse, grants: u8) -> CallReply<ImageResponse> {
        CallReply {
            response,
            grants,
            delivered: [None; kore_rt::service::MAX_GRANTS],
            retained: [const { None }; kore_rt::service::MAX_GRANTS],
        }
    }

    fn metadata(version: u64) -> ImageSourceMetadata {
        ImageSourceMetadata {
            source: 7.into(),
            version: version.into(),
            len: 4096.into(),
            digest: [version as u8; 32],
            profile: 0.into(),
            reserved: 0.into(),
        }
    }

    #[test]
    fn fork_resume_preserves_foreign_retirement_and_never_publishes_when_idle() {
        with_pair(|image, serve, transport| {
            transport.0.set(false);
            assert!(matches!(
                run(image.retire_face_bounded(17, 5_000_000)),
                Err(CallError::Transport)
            ));
            assert!(serve.req.pop().is_some());
            serve
                .rep
                .push(ImageService::encode_rep(&ImageResponse::Retired, 0))
                .unwrap();
            let commits = transport.1.get();
            assert!(matches!(
                run(image.resume_face_fork_bounded(17, 5_000_000)),
                Err(CallError::Outstanding)
            ));
            assert_eq!(commits, transport.1.get());
            transport.0.set(true);
            assert!(matches!(
                run(image.retire_face_bounded(17, 5_000_000)),
                Ok(ImageResponse::Retired)
            ));
            assert!(matches!(
                run(image.resume_face_fork_bounded(17, 5_000_000)),
                Err(CallError::NoOutstanding)
            ));
            assert!(serve.req.pop().is_none());
        });
    }

    #[test]
    fn acquire_resume_cannot_consume_a_foreign_retirement_or_publish_when_idle() {
        with_pair(|image, serve, transport| {
            let name = ImageName::new("kore-ld-stub").unwrap();
            assert!(matches!(
                run(image.resume_acquire_bounded(name, 5_000_000)),
                Err(CallError::NoOutstanding)
            ));
            assert!(serve.req.pop().is_none());
            transport.0.set(false);
            assert!(matches!(
                run(image.retire_face_bounded(17, 5_000_000)),
                Err(CallError::Transport)
            ));
            assert!(serve.req.pop().is_some());
            serve
                .rep
                .push(ImageService::encode_rep(&ImageResponse::Retired, 0))
                .unwrap();
            let commits = transport.1.get();
            assert!(matches!(
                run(image.resume_acquire_bounded(name, 5_000_000)),
                Err(CallError::Outstanding)
            ));
            assert_eq!(transport.1.get(), commits);
            transport.0.set(true);
            assert!(matches!(
                run(image.retire_face_bounded(17, 5_000_000)),
                Ok(ImageResponse::Retired)
            ));
            assert!(serve.req.pop().is_none());
        });
    }

    #[test]
    fn face_decoder_rejects_foreign_shape_and_grants_on_refusal() {
        for (response, grants) in [
            (ImageResponse::Retired, 0),
            (ImageResponse::Refused(ImageRefusal::Unknown), 1),
        ] {
            with_pair(|image, _, _| {
                assert!(matches!(
                    image.finish_face(reply(response, grants)),
                    Err(CallError::Wire)
                ));
                assert!(!image.is_idle());
            });
        }
        with_pair(|image, _, _| {
            let mut malformed = reply(ImageResponse::Refused(ImageRefusal::Unknown), 0);
            malformed.delivered[0] = Some(SlotRef::from_wire(33));
            assert!(matches!(image.finish_face(malformed), Err(CallError::Wire)));
            assert!(!image.is_idle());
        });
    }

    #[test]
    fn face_possession_failure_keeps_the_remote_retirement_identity() {
        // The wire decoder owns descriptor validation. This fixture reaches
        // only the typed reply boundary, with no fabricated native grants.
        let response = ImageResponse::Face {
            face: 17,
            generation: 23,
            module: kore_abi::image::ModuleDescriptor::empty(),
        };
        for grants in [0, MAX_IMAGE_SEGS as u8, u8::MAX] {
            with_pair(|image, _, _| {
                assert!(matches!(
                    image.finish_face(reply(response, grants)),
                    Err(CallError::Wire)
                ));
            });
        }
        with_pair(|image, _, _| {
            let (response, authority) = image
                .finish_face(reply(response, (MAX_IMAGE_SEGS + 1) as u8))
                .unwrap();
            assert!(matches!(
                response,
                ImageResponse::Face {
                    face: 17,
                    generation: 23,
                    ..
                }
            ));
            assert!(authority.is_none());
            assert!(!image.is_idle());
        });
    }

    #[test]
    fn bank_decoder_requires_a_complete_bank_or_an_empty_refusal() {
        for (response, grants) in [
            (ImageResponse::Retired, 0),
            (ImageResponse::Banked, 0),
            (ImageResponse::Banked, MAX_IMAGE_SEGS as u8 - 1),
            (ImageResponse::Banked, MAX_IMAGE_SEGS as u8 + 1),
            (ImageResponse::Refused(ImageRefusal::Unknown), 1),
            // A complete count with no retained owners is still malformed.
            (ImageResponse::Banked, MAX_IMAGE_SEGS as u8),
        ] {
            with_pair(|image, _, _| {
                assert!(matches!(
                    image.finish_bank(reply(response, grants)),
                    Err(CallError::Wire)
                ));
                assert!(!image.is_idle());
            });
        }
        with_pair(|image, _, _| {
            let (response, recvs) = image
                .finish_bank(reply(ImageResponse::Refused(ImageRefusal::Unknown), 0))
                .unwrap();
            assert!(matches!(
                response,
                ImageResponse::Refused(ImageRefusal::Unknown)
            ));
            assert!(recvs.iter().all(Option::is_none));
            assert!(image.is_idle());
            let mut malformed = reply(ImageResponse::Refused(ImageRefusal::Unknown), 0);
            malformed.delivered[MAX_IMAGE_SEGS] = Some(SlotRef::from_wire(33));
            assert!(matches!(image.finish_bank(malformed), Err(CallError::Wire)));
        });
    }

    // Pure affine custody witnesses, not synthetic native capabilities. The
    // production conversion loop is exercised with distinct non-Copy owners
    // whose destruction is observable, including failure after each prefix.
    struct BankOwner<'a> {
        id: usize,
        drops: &'a Cell<usize>,
    }
    impl Drop for BankOwner<'_> {
        fn drop(&mut self) {
            self.drops.set(self.drops.get() + 1);
        }
    }
    struct TypedBankOwner<'a>(BankOwner<'a>);

    #[test]
    fn partial_bank_conversion_restores_exact_owners_without_dropping_them() {
        const N: usize = MAX_IMAGE_SEGS;
        for missing in [false, true] {
            for failed in 0..N {
                let drops = Cell::new(0);
                let mut retained: [Option<BankOwner<'_>>; N] = core::array::from_fn(|id| {
                    if missing && id == failed {
                        None
                    } else {
                        Some(BankOwner { id, drops: &drops })
                    }
                });
                let result: Result<[Option<TypedBankOwner<'_>>; N], ()> = convert_grant_bank(
                    &mut retained,
                    |owner| {
                        if owner.id == failed {
                            Err(owner)
                        } else {
                            Ok(TypedBankOwner(owner))
                        }
                    },
                    |owner| owner.0,
                );
                assert!(result.is_err());
                assert_eq!(drops.get(), 0);
                for (id, owner) in retained.iter().enumerate() {
                    assert_eq!(
                        owner.as_ref().map(|owner| owner.id),
                        (!(missing && id == failed)).then_some(id)
                    );
                }
                drop(retained);
                assert_eq!(drops.get(), N - usize::from(missing));
            }
        }
    }

    #[test]
    fn bank_conversion_preserves_outside_owners_and_checks_bounds_before_taking() {
        const N: usize = MAX_IMAGE_SEGS;
        let drops = Cell::new(0);
        let mut retained: [Option<BankOwner<'_>>; N + 1] =
            core::array::from_fn(|id| Some(BankOwner { id, drops: &drops }));
        let short: Result<[Option<TypedBankOwner<'_>>; N], ()> = convert_grant_bank(
            &mut retained[1..N],
            |owner| Ok(TypedBankOwner(owner)),
            |owner| owner.0,
        );
        assert!(short.is_err());
        assert!(retained.iter().all(Option::is_some));
        assert_eq!(drops.get(), 0);
        let complete: [Option<TypedBankOwner<'_>>; N] = convert_grant_bank(
            &mut retained[1..],
            |owner| Ok(TypedBankOwner(owner)),
            |owner| owner.0,
        )
        .ok()
        .unwrap();
        assert_eq!(retained[0].as_ref().unwrap().id, 0);
        assert!(retained[1..].iter().all(Option::is_none));
        for (i, owner) in complete.iter().enumerate() {
            assert_eq!(owner.as_ref().unwrap().0.id, i + 1);
        }
        assert_eq!(drops.get(), 0);
        drop(complete);
        assert_eq!(drops.get(), N);
        drop(retained);
        assert_eq!(drops.get(), N + 1);
    }

    #[test]
    fn registration_resume_matches_the_original_name_metadata_and_return_route() {
        with_pair(|image, serve, transport| {
            let name = ImageName::new("tool").unwrap();
            let leg = RegistrationLeg {
                return_tx: SlotRef::from_wire(110),
                return_land: SlotRef::from_wire(111),
            };
            image.registration = Some(leg);
            let original = ImageRequest::Register {
                name,
                slot: leg.return_land,
                source: metadata(4),
            };
            // Exercise SvcClient's correlation without fabricating a native source.
            // There is no ImageService running or authority transfer in this fixture.
            transport.0.set(false);
            assert!(matches!(
                run(image.pair.call(&original)),
                Err(CallError::Transport)
            ));
            let wire = serve.req.pop().unwrap();
            serve
                .rep
                .push(ImageService::encode_rep(
                    &ImageResponse::Refused(ImageRefusal::BadImage),
                    0,
                ))
                .unwrap();
            let calls = transport.1.get();
            assert!(matches!(
                run(image.resume_registration_bounded(name, metadata(5), 1)),
                Err(CallError::Outstanding)
            ));
            assert!(matches!(
                run(image.resume_registration_bounded(
                    ImageName::new("other").unwrap(),
                    metadata(4),
                    1
                )),
                Err(CallError::Outstanding)
            ));
            image.registration.as_mut().unwrap().return_land = SlotRef::from_wire(112);
            assert!(matches!(
                run(image.resume_registration()),
                Err(CallError::Outstanding)
            ));
            image.registration = Some(leg);
            assert_eq!(image.pair.outstanding_request(), Some(&wire));
            assert_eq!(transport.1.get(), calls);
            transport.0.set(true);
            assert!(matches!(
                run(image.resume_registration_bounded(name, metadata(4), 1)),
                Ok(ImageResponse::Refused(ImageRefusal::BadImage))
            ));
            assert!(image.is_idle());
            assert!(serve.req.pop().is_none());
        });
    }

    #[test]
    fn registration_resume_preserves_a_foreign_reply_and_unstarted_source() {
        with_pair(|image, serve, transport| {
            let name = ImageName::new("tool").unwrap();
            image.registration = Some(RegistrationLeg {
                return_tx: SlotRef::from_wire(110),
                return_land: SlotRef::from_wire(111),
            });
            transport.0.set(false);
            assert!(matches!(
                run(image.request(ImageRequest::Describe { name })),
                Err(CallError::Transport)
            ));
            let wire = serve.req.pop().unwrap();
            serve
                .rep
                .push(ImageService::encode_rep(
                    &ImageResponse::Refused(ImageRefusal::Unknown),
                    0,
                ))
                .unwrap();
            let calls = transport.1.get();
            assert!(matches!(
                run(image.resume_registration()),
                Err(CallError::Outstanding)
            ));
            assert!(matches!(
                run(image.resume_registration_bounded(name, metadata(4), 1)),
                Err(CallError::Outstanding)
            ));
            let mut source = None;
            assert!(matches!(
                run(image.register_retained(name, metadata(4), &mut source)),
                Err(RetainedReturnCallError::NotMoved(CallError::Outstanding))
            ));
            assert_eq!(image.pair.outstanding_request(), Some(&wire));
            assert_eq!(transport.1.get(), calls);
            transport.0.set(true);
            assert!(run(image.resume_request()).is_ok());
            assert!(image.is_idle());
            assert!(serve.req.pop().is_none());
        });
    }

    #[test]
    fn registration_retirement_resumes_only_the_original_name() {
        with_pair(|image, serve, transport| {
            let name = ImageName::new("tool").unwrap();
            image.registration = Some(RegistrationLeg {
                return_tx: SlotRef::from_wire(110),
                return_land: SlotRef::from_wire(111),
            });
            transport.0.set(false);
            assert!(matches!(
                run(image.retire_registration_bounded(name, false, 5_000_000)),
                Err(CallError::Transport)
            ));
            let wire = serve.req.pop().unwrap();
            serve
                .rep
                .push(ImageService::encode_rep(&ImageResponse::Retired, 0))
                .unwrap();
            let calls = transport.1.get();
            assert!(matches!(
                run(image.retire_registration_bounded(
                    ImageName::new("other").unwrap(),
                    true,
                    5_000_000
                )),
                Err(CallError::Outstanding)
            ));
            assert!(matches!(
                run(image.resume_registration()),
                Err(CallError::Outstanding)
            ));
            assert_eq!(image.pair.outstanding_request(), Some(&wire));
            assert_eq!(transport.1.get(), calls);
            transport.0.set(true);
            assert!(matches!(
                run(image.retire_registration_bounded(name, true, 5_000_000)),
                Ok(ImageResponse::Retired)
            ));
            assert!(image.is_idle());
            assert!(serve.req.pop().is_none());
        });
    }

    #[test]
    fn registration_refusal_requires_a_grant_free_reply() {
        with_pair(|image, _, _| {
            assert!(matches!(
                finish_registration(
                    &mut image.pair,
                    reply(ImageResponse::Refused(ImageRefusal::BadImage), 0)
                ),
                Ok(ImageResponse::Refused(_))
            ));
            let mut bad = reply(ImageResponse::Refused(ImageRefusal::BadImage), 0);
            bad.delivered[0] = Some(SlotRef::from_wire(22));
            assert!(matches!(
                finish_registration(&mut image.pair, bad),
                Err(CallError::Wire)
            ));
            assert!(!image.is_idle());
        });
    }

    #[test]
    fn export_resume_preserves_a_foreign_operation_and_queued_reply() {
        with_pair(|image, serve, transport| {
            let name = ImageName::new("ls").unwrap();
            let original = ImageRequest::Describe { name };
            transport.0.set(false);
            assert!(matches!(
                run(image.request(original)),
                Err(CallError::Transport)
            ));
            let wire = serve.req.pop().unwrap();
            serve
                .rep
                .push(ImageService::encode_rep(
                    &ImageResponse::Refused(ImageRefusal::Unknown),
                    0,
                ))
                .unwrap();
            let calls = transport.1.get();
            assert!(matches!(
                run(image.export(name)),
                Err(CallError::Outstanding)
            ));
            assert!(matches!(
                run(image.resume_export_bounded(name, 1)),
                Err(CallError::Outstanding)
            ));
            assert_eq!(image.pair.outstanding_request(), Some(&wire));
            assert_eq!(transport.1.get(), calls);
            assert!(serve.req.pop().is_none());
            transport.0.set(true);
            assert!(matches!(
                run(image.resume_request()),
                Ok((ImageResponse::Refused(ImageRefusal::Unknown), _))
            ));
            assert!(image.is_idle());
        });
    }

    #[test]
    fn export_preflight_without_retained_landing_never_publishes() {
        with_pair(|image, serve, _| {
            let name = ImageName::new("ls").unwrap();
            assert!(run(image.export(name)).is_err());
            assert!(image.is_idle());
            assert!(serve.req.pop().is_none());
            assert!(matches!(
                run(image.resume_export_bounded(name, 1)),
                Err(CallError::NoOutstanding)
            ));
        });
    }

    #[test]
    fn export_rejects_missing_authority_empty_bytes_and_foreign_reply_shapes() {
        for response in [
            ImageResponse::Exported { len: 4096 },
            ImageResponse::Exported { len: 0 },
            ImageResponse::Retired,
        ] {
            with_pair(|image, _, _| {
                assert!(matches!(
                    image.finish_export(reply(response, 1)),
                    Err(CallError::Wire)
                ));
                assert!(!image.is_idle());
                assert!(matches!(
                    run(image.resume_export_bounded(ImageName::new("ls").unwrap(), 1)),
                    Err(CallError::Poisoned)
                ));
            });
        }
        with_pair(|image, _, _| {
            assert!(
                image
                    .finish_export(reply(ImageResponse::Refused(ImageRefusal::Unknown), 0))
                    .unwrap()
                    .is_none()
            );
            assert!(image.is_idle());
            let mut malformed = reply(ImageResponse::Refused(ImageRefusal::Unknown), 0);
            malformed.delivered[0] = Some(SlotRef::from_wire(33));
            assert!(matches!(
                image.finish_export(malformed),
                Err(CallError::Wire)
            ));
            assert!(!image.is_idle());
        });
    }
}
