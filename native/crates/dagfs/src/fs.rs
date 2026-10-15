//! The process's own filesystem face — the one client every command,
//! interpreter, and application attaches.
//!
//! The founder stamps a complete filesystem face: one control pair, one
//! local data-window view, one sibling view moved into the service's retained
//! return post, and the request/return endpoints. The two views are separate
//! capabilities over the same bounded backing; neither process borrows an
//! address from the other. Image profile metadata, authenticated by the
//! image service, is what causes exec to install these roles: an ordinary
//! process cannot obtain them by spelling a command name or inventing a slot
//! number.
//!
//! # No hand-rolled protocol
//!
//! Every request is an [`FsRequest`], encoded and decoded by the leaf's own
//! [`DagfsProtocol`](crate::DagfsProtocol). Nothing here touches a wire
//! field, so a wire change is a compile error in one place rather than a
//! silent misread here. The synchronous [`FsClient`](crate::FsClient) over
//! its one-method transport cannot be used on metal — a pair round trip
//! parks, and `exchange` returns a value rather than a future — which is why
//! this face is the async [`RingFs`].
//!
//! # Every intermediate handle is closed
//!
//! A session holds at most [`MAX_HANDLES_PER_SESSION`] open handles, so a
//! client that leaked one per path component would answer `OutOfHandles`
//! to every later command and the user would have no way to clear it. Each
//! verb closes what it opened on the refusal path too, which is why a result
//! is computed, then the handles closed, and only then `?`-propagated.

use kore_abi::{session::SlotRef, task::role};
use kore_rt::{
    cap::{HeldChanSendToken, LeaseToken, verbs},
    entry::Rt,
    service::MAX_GRANTS,
    svc::{ClientEndpoints, REQ_CAP, SvcClient, client_rings},
    transport::TrapTransport,
};

use crate::{
    DagfsProtocol, FS_GRANT_TARGET_RIGHTS, FsRequest, FsResponse, FsViewCapability, Handle,
    Refusal, RingFs, bounds::MAX_HANDLES_PER_SESSION, process_face,
};

/// A real-machine scheduling turn between recyclable-peer admission and the
/// first command request.  The bound is deliberately on paced attempts: a
/// stopped filesystem still fails closed, while a healthy service gets time
/// to repair its rotating retained post without a hot retry storm.
const FS_ATTACH_TRIES: u32 = 3_000;
const FS_ATTACH_RETRY_NS: u64 = 1_000_000;

/// The deepest walk holds two handles, and the widest verb (`link`, `move`
/// across directories) holds three at once. A session's ceiling is
/// [`MAX_HANDLES_PER_SESSION`]; this asserts a command's peak is inside it
/// with room, so a future verb that holds one more cannot silently start
/// answering `OutOfHandles` on a deep path.
pub const PEAK_HANDLES: usize = 4;
const _: () = assert!(PEAK_HANDLES <= MAX_HANDLES_PER_SESSION);

/// This process's private filesystem pair.
impl Fs<'_, '_> {
    /// Drive replies by re-polling the ring rather than the executor's clock
    /// ([`RingFs::set_direct_polling`]): for a caller that completes its
    /// filesystem calls under `block_on` from inside a task's poll, and
    /// restores the executor mode before the executor drives them again.
    pub fn set_direct_polling(&mut self, direct: bool) {
        self.client.set_direct_polling(direct);
    }
}

pub struct Fs<'w, 't> {
    pub(crate) client: RingFs<'w, 't, TrapTransport>,
    pub(crate) window: &'static mut [u8],
    /// Retained lessor positions. They keep the mapped client alive and are
    /// the only sources from which this process may derive service views.
    _ring_lease: LeaseToken,
    _data_lease: LeaseToken,
    /// Optional caller-minted view endpoint delivered only to executables
    /// whose authenticated profile permits a cwd transition. It is moved
    /// whole to FS after the target has been resolved through this private
    /// pair, so child teardown cannot cut the installed view below the
    /// shell-owned root.
    pub(crate) view_binding: Option<HeldChanSendToken>,
    /// Optional one-shot selector for the stable recipient task view. This
    /// endpoint carries REVOKE only, so it cannot be used as a cwd or further
    /// delegated by the command.
    pub(crate) grant_target: Option<HeldChanSendToken>,
    /// This launch's storage ring, when its founding delivered one
    /// ([`role::STOR_RING`] and [`role::STOR_TARGET`]): set up on the first
    /// read that streams through it ([`crate::stor::FacadeRing`]).
    pub(crate) ring: crate::stor::FacadeRing,
    /// Files read whole through that ring.
    pub(crate) ring_files: u64,
    /// Why the last whole-file read did or did not take that ring.
    pub(crate) ring_note: &'static str,
}

impl<'w, 't> Fs<'w, 't> {
    /// Attach the exec-synthesized filesystem face.
    ///
    /// `None` when the founding delivered no face (the image's profile does
    /// not carry the filesystem) or the service could not adopt the data
    /// window within the paced attach budget.
    pub async fn attach(rt: &Rt, transport: &'t TrapTransport) -> Option<Self> {
        let ring_lease = verbs::take_lease(
            rt.bootstrap()
                .handle(role::SVC_RING, process_face::CONTROL_RING)?,
        )?;
        let window = verbs::map_held_rw(ring_lease.slot())?;
        // SAFETY: `map_view` installed its reported `[base, +len)` R/W-USER
        // in this task's own space over the pair's leased backing,
        // page-aligned (hence 64-byte aligned, the spsc layout unit) and
        // live for the task's whole life. This task is the sole client-side
        // attacher; the service half lives in another space and touches the
        // window only through the SPSC protocol.
        let window: &'static mut [u8] =
            unsafe { core::slice::from_raw_parts_mut(window.cast::<u8>().as_ptr(), window.len()) };
        // An *attach*, not an init: the filesystem row's own `init_window`
        // formatted this ring at founding, and `client_rings` fail-closes
        // on an unformatted window rather than reading a zeroed page as an
        // empty ring.
        let rings = client_rings::<DagfsProtocol>(window, REQ_CAP, kore_rt::svc::REP_CAP).ok()?;
        let kick = rt
            .bootstrap()
            .handle(role::SVC_SEND, process_face::REQUEST_KICK)?;
        let data_lease = verbs::take_lease(
            rt.bootstrap()
                .handle(role::SVC_RING, process_face::DATA_WINDOW)?,
        )?;
        let data = verbs::map_held_rw(data_lease.slot())?;
        // SAFETY: as above — the founder's own mapping of this task's data
        // window, page-aligned and live for the task's whole life.
        let data: &'static mut [u8] =
            unsafe { core::slice::from_raw_parts_mut(data.cast::<u8>().as_ptr(), data.len()) };
        let return_tx = rt
            .bootstrap()
            .handle(role::SVC_RETURN, process_face::RETURN_SEND)?;
        let announced = SlotRef::from_wire(
            rt.bootstrap()
                .datum(role::SVC_RETURN, process_face::RETURN_LAND)?,
        );
        let bytes = u32::try_from(data.len()).ok()?;
        let view_binding = rt
            .bootstrap()
            .handle(role::FS_VIEW_BIND, 0)
            .and_then(|slot| verbs::take_chan_send(slot, FsViewCapability::INVOCATION_RIGHTS));
        let grant_target = rt
            .bootstrap()
            .handle(role::FS_GRANT_TARGET, 0)
            .and_then(|slot| verbs::take_chan_send(slot, FS_GRANT_TARGET_RIGHTS));
        let pair = SvcClient::new(
            transport,
            rings,
            ClientEndpoints {
                // The filesystem mints service-local authority only, so no
                // kernel capability ever rides a reply and there is no grant
                // channel to pre-pin (`REPLY_GRANTS == 0`, asserted
                // service-side).
                kick,
                grants: [None; MAX_GRANTS],
            },
        );
        let mut client = RingFs::new(pair, return_tx, announced);
        let mut attached = false;
        for remaining in (0..FS_ATTACH_TRIES).rev() {
            // `attach_window` may have published the request before its
            // transport boundary failed.  The SPSC reply stream carries no
            // request ids, so resume that exact request whenever the pair is
            // non-idle; publishing a replacement could let a late reply
            // satisfy the wrong operation.
            let result = if client.is_idle() {
                client.attach_window(&data_lease, bytes).await
            } else {
                client.resume_attach_window(bytes).await
            };
            match result {
                Ok(()) => {
                    attached = true;
                    break;
                }
                Err(error) if remaining > 0 && error.retryable_transport() => {
                    kore_rt::sleep_for(FS_ATTACH_RETRY_NS).await;
                }
                Err(_) => return None,
            }
        }
        if !attached {
            return None;
        }
        let ring = crate::stor::FacadeRing::delivered(
            rt.bootstrap()
                .handle(role::STOR_RING, 0)
                .and_then(verbs::take_lease),
            rt.bootstrap()
                .handle(role::STOR_TARGET, 0)
                .and_then(verbs::take_lease),
        );
        Some(Self {
            client,
            window: data,
            _ring_lease: ring_lease,
            _data_lease: data_lease,
            view_binding,
            grant_target,
            ring,
            ring_files: 0,
            ring_note: "unused",
        })
    }

    /// The bulk data window, read side. Bytes a verb landed here are valid
    /// until the next request that uses the window.
    pub fn window(&self) -> &[u8] {
        self.window
    }

    /// The bulk data window, write side — where a verb stages bytes it is
    /// about to hand the service.
    pub fn window_mut(&mut self) -> &mut [u8] {
        self.window
    }

    /// A bounded, validated read of the service's aggregate posture. A
    /// timeout leaves the exact status request pending on this pair.
    pub async fn system_status_bounded(
        &mut self,
        budget_ns: u64,
    ) -> Result<kore_abi::svc::FsSystemStatusWire, crate::client::RingFsError> {
        use kore_abi::svc::FsSystemStatusWire;
        let len = self.client.system_status_bounded(0, budget_ns).await?;
        if len as usize != FsSystemStatusWire::BYTES {
            return Err(crate::client::RingFsError::Wire);
        }
        crate::client::read_system_status(self.window, 0)
            .map_err(crate::client::RingFsError::Refused)
    }

    /// Whether a new filesystem request can safely be published.
    pub fn status_pair_idle(&self) -> bool {
        self.client.is_idle()
    }

    /// One round trip. A transport failure is [`Refusal::Pair`] — a caller
    /// never turns "unreachable" into a filesystem answer.
    pub async fn call(&mut self, request: &FsRequest) -> Result<FsResponse, Refusal> {
        match self.client.call(request).await {
            Ok(reply) => Ok(reply),
            Err(_) => Err(Refusal::Pair),
        }
    }

    /// [`Self::call`], with a service refusal folded into the error channel.
    pub async fn checked(&mut self, request: &FsRequest) -> Result<FsResponse, Refusal> {
        match self.call(request).await? {
            FsResponse::Refused(err) => Err(Refusal::Fs(err)),
            other => Ok(other),
        }
    }

    /// Close `handle`, ignoring the answer.
    ///
    /// The one place a refusal is not surfaced, and the reason is that there
    /// is nothing behind it to act on: the handle is gone from this client's
    /// reckoning either way, and a stale or already-closed handle is
    /// precisely what a `Close` refusal reports. Nothing is truncated and no
    /// result is discarded — a close has no result.
    pub async fn release(&mut self, handle: Handle) {
        let _ = self.call(&FsRequest::Close { handle }).await;
    }
}
