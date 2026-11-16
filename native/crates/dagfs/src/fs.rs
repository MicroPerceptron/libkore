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
    DagfsProtocol, FS_GRANT_TARGET_RIGHTS, FS_VIEW_ANCHOR_RIGHTS, FsRequest, FsResponse, FsViewCapability, Handle,
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

    /// Opt into the existing exact-pair deadline policy for data-only
    /// control calls. On expiry this face must not be reused for unrelated
    /// work; `status_pair_idle` remains false. Authority-return operations
    /// keep their own custody policy. Change only while the pair is idle.
    pub fn set_observation(&mut self, observation: kore_rt::svc::Observation) -> bool {
        self.client.set_observation(observation)
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
    /// Optional one-shot selector for the launcher's own cwd view, delivered
    /// with a line whose paths the launcher spelled from that cwd's route
    /// ([`Fs::confirm_position`]). REVOKE only, like the grant target.
    pub(crate) view_anchor: Option<HeldChanSendToken>,
    /// This launch's storage ring, when its founding delivered one
    /// ([`role::STOR_RING`] and [`role::STOR_TARGET`]): set up on the first
    /// read that streams through it ([`crate::stor::FacadeRing`]).
    pub(crate) ring: crate::stor::FacadeRing,
    /// Files read whole through that ring.
    pub(crate) ring_files: u64,
    /// Why the last whole-file read did or did not take that ring.
    pub(crate) ring_note: &'static str,
    /// This pair's watch plane — the event channel the service formats
    /// after the rings in this pair's own window (`crate::watch`) — or
    /// `None` on a pair formatted without one.
    pub(crate) watch: Option<crate::watch::WatchSink<'static>>,
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
        // An *attach*, not an init: the filesystem row's own formatter
        // laid out this window at founding — the rings, then the watch
        // plane — and both attaches fail closed on an unformatted window
        // rather than reading a zeroed page as an empty ring.
        let (pair, tail) = window.split_at_mut(crate::watch::PLANE_AT.min(window.len()));
        let rings = client_rings::<DagfsProtocol>(pair, REQ_CAP, kore_rt::svc::REP_CAP).ok()?;
        let watch = crate::watch::sink(tail).ok();
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
        let view_anchor = rt
            .bootstrap()
            .handle(role::FS_VIEW_ANCHOR, 0)
            .and_then(|slot| verbs::take_chan_send(slot, FS_VIEW_ANCHOR_RIGHTS));
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
            view_anchor,
            ring,
            ring_files: 0,
            ring_note: "unused",
            watch,
        })
    }

    /// This pair's watch plane, if the pair carries one.
    pub fn watch_plane(&self) -> Option<&crate::watch::WatchSink<'static>> {
        self.watch.as_ref()
    }

    /// Arm watch `slot` on the directory `dir`, tagged `tag` (a non-zero
    /// name of the client's own for this arming): the plane counts every
    /// change to what listing `dir` shows from now on. The view stays the
    /// caller's — closing it ends the watch. `Unsupported` on a pair without
    /// a plane.
    pub async fn watch(&mut self, dir: &crate::View, slot: u8, tag: u32) -> Result<(), Refusal> {
        if self.watch.is_none() {
            return Err(Refusal::Fs(crate::FsError::Unsupported));
        }
        self.checked(&FsRequest::Watch {
            handle: dir.handle(),
            slot,
            tag,
        })
        .await
        .map(|_| ())
    }

    /// Disarm watch `slot`.
    pub async fn unwatch(&mut self, slot: u8) -> Result<(), Refusal> {
        self.checked(&FsRequest::Unwatch { slot }).await.map(|_| ())
    }

    /// The plane's wake word, read before a caller drains its sources: a
    /// park published against it ([`Self::arm_watch_park`]) is lossless.
    /// Zero, and never moving, on a pair without a plane.
    pub fn watch_seq(&self) -> u32 {
        self.watch.as_ref().map_or(0, crate::watch::WatchSink::seq)
    }

    /// The plane's state, `None` on a pair without one (or mid-publish,
    /// which the caller treats as nothing new and reads again next pass —
    /// the publish ends by moving the wake word).
    pub fn watch_state(&self) -> Option<kore_abi::svc::FsWatchState> {
        self.watch.as_ref().and_then(crate::watch::WatchSink::state)
    }

    /// Publish this task's intent to wait on the plane, for a caller about to
    /// enter one executor wait that every source shares (the multiplexed
    /// park, `res/FILES_DESKTOP.md`): the service's next batch then owes
    /// this task a notice on its doorbell. `false` when the plane already
    /// moved past `seen` — drain again instead of waiting. Always `true` on
    /// a pair without a plane, which has nothing to wait for.
    pub fn arm_watch_park(&self, seen: u32) -> bool {
        self.watch.as_ref().is_none_or(|sink| sink.park(seen))
    }

    /// Withdraw the intent [`Self::arm_watch_park`] published, however the
    /// wait ended, so a departed waiter is never charged a notice.
    pub fn withdraw_watch_park(&self) {
        if let Some(sink) = self.watch.as_ref() {
            sink.unpark();
        }
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
