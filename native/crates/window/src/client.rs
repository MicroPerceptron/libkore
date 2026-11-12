//! Receiver-mapped window transport and bounded backing replacement.
//!
//! The application borrows its own pair mapping, not a fabricated static slice.
//! Pixels stay in compositor-owned leases; this client retains at most the new
//! backing and the last confirmed displayed backing. A resize never retires the
//! latter until the new surface has actually been presented.

use core::{marker::PhantomData, ptr::NonNull};
use kore_abi::{
    cap::Rights,
    session::{DISPLAY_FORMAT_XRGB8888, Notice, SlotRef, SurfaceMapLane},
    svc::{CommitWire, MAX_COMMIT_RECTS, OutChunk, SurReqWire},
    task::role,
};
use kore_ring::spsc::{Consumer, Producer};
use kore_rt::{
    cap::{ReturnedCapability, StashedView, verbs},
    chan::SendError,
    entry::Rt,
    executor::Doorbell,
    service::MAX_GRANTS,
    svc::{ClientEndpoints, GrantPin, REP_CAP, REQ_CAP, SvcClient},
    transport::{ParkTransport, Transport},
};

use crate::{
    Request, Response, WindowFace, WindowWire,
    configure::Configure,
    control::{ControlRefusal, TabModel},
    event::Event,
    wire::{self, Commit, Damage, SurRect, SurfaceRefusal, SurfaceRequest, SurfaceResponse},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClientError {
    Bootstrap,
    AlreadyAttached,
    Ring,
    Protocol,
    Transport,
    Refused(ControlRefusal),
    SurfaceRefused(SurfaceRefusal),
    Geometry,
    Mapping,
    NoSurface,
    /// Re-read configure; the previous frame no longer describes this window.
    Reconfigure,
    /// Cleanup retains the exact owner and can be retried.
    Busy,
    CommitQueue,
    CommitRejected,
    EpochExhausted,
    EventMalformed,
    /// A menu the encoder refused before anything reached the compositor.
    Menu(crate::menu::MenuError),
    /// A clipboard text past [`crate::clipboard::CLIPBOARD_BYTES`], or one
    /// longer than its reader can hold: refused whole, never cut short.
    TooLong,
}

impl ClientError {
    pub const fn is_retryable(self) -> bool {
        matches!(
            self,
            Self::Reconfigure
                | Self::Busy
                | Self::Refused(
                    ControlRefusal::Busy | ControlRefusal::Pending | ControlRefusal::Stale
                )
                | Self::SurfaceRefused(SurfaceRefusal::Busy)
        )
    }

    /// Configuration replacement may temporarily exceed the compositor's
    /// bounded surface budget while the last presented backing is retained.
    /// Only the configure loop may retry these capacity refusals; treating
    /// them as generally retryable could hide a permanently invalid request.
    pub const fn is_configuration_retryable(self) -> bool {
        self.is_retryable()
            || matches!(
                self,
                Self::SurfaceRefused(SurfaceRefusal::ResourceLimit | SurfaceRefusal::OutOfMemory)
            )
    }
}

/// Local attachment lifetime for a founder-delivered pair. The parent keeps its
/// ancestor until this process exits; only the process owns this client half.
pub struct PairMapping {
    memory: NonNull<[u8]>,
    kick: SlotRef,
    grant: SlotRef,
    bell: Doorbell,
    attached: bool,
}

impl PairMapping {
    /// Map this application's endowed pair, without an aperture or pool root.
    ///
    /// # Safety
    /// Call exactly once in the process, on the initial runtime task. No other
    /// caller may attach the client halves of this same pair. The parent must
    /// stop the process before revoking the pair's ancestor (the Session law).
    pub unsafe fn map_bootstrap(rt: &Rt) -> Result<Self, ClientError> {
        // SAFETY: delegated unchanged from this method's sole-attachment
        // contract; only the inert boot-block coordinates differ.
        unsafe { Self::map_bootstrap_at(rt, WindowFace::STANDALONE) }
    }

    /// Map a founder-staged pair at one validated application-bundle face.
    ///
    /// # Safety
    /// The same sole-process attachment law as [`Self::map_bootstrap`] applies.
    pub unsafe fn map_bootstrap_at(rt: &Rt, face: WindowFace) -> Result<Self, ClientError> {
        let boot = rt.bootstrap();
        let view = boot
            .handle(role::SVC_RING, face.ring())
            .and_then(verbs::take_view)
            .ok_or(ClientError::Bootstrap)?;
        if !view.rights().contains(Rights::ACCESS | Rights::MODIFY) {
            return Err(ClientError::Bootstrap);
        }
        let memory = verbs::map_view(view).ok_or(ClientError::Bootstrap)?;
        if memory.len() < wire::SURFACE_PAIR_BYTES {
            return Err(ClientError::Bootstrap);
        }
        Ok(Self {
            memory,
            kick: boot
                .handle(role::SVC_SEND, face.request_kick())
                .ok_or(ClientError::Bootstrap)?,
            grant: boot
                .handle(role::SVC_GRANT, face.grant_recv())
                .ok_or(ClientError::Bootstrap)?,
            bell: rt.doorbell().ok_or(ClientError::Bootstrap)?,
            attached: false,
        })
    }

    pub fn attach<'m, 't, T: ParkTransport>(
        &'m mut self,
        transport: &'t T,
    ) -> Result<WindowClient<'m, 't, T>, ClientError> {
        if self.attached {
            return Err(ClientError::AlreadyAttached);
        }
        self.attached = true;
        let mut grants = [const { None }; MAX_GRANTS];
        grants[0] = Some(GrantPin::retained(self.grant));
        // SAFETY: map_bootstrap's sole-attachment contract, permanently spent
        // above. The returned rings borrow this local mapping owner, not 'static.
        let memory = unsafe {
            core::slice::from_raw_parts_mut(self.memory.cast::<u8>().as_ptr(), self.memory.len())
        };
        Ok(WindowClient::from_memory(
            transport,
            memory,
            ClientEndpoints {
                kick: self.kick,
                grants,
            },
        )?
        .with_reply_wait(self.bell))
    }
}

/// A frozen frame, mapped read-only for its capture writer. The view's
/// owner is held for as long as the pixels are borrowed.
pub struct CaptureView {
    frame: crate::capture::CaptureFrame,
    pixels: NonNull<u32>,
    _view: StashedView,
}

impl CaptureView {
    pub const fn frame(&self) -> crate::capture::CaptureFrame {
        self.frame
    }

    /// Row `y`: `width` XRGB pixels, packed.
    pub fn row(&self, y: u32) -> &[u32] {
        let (width, height) = (usize::from(self.frame.width), u32::from(self.frame.height));
        assert!(y < height, "row {y} of a {height}-row frame");
        // SAFETY: `take_capture` checked the mapping holds `width * height`
        // aligned pixels; the lend outlives `self`, and nothing writes the
        // frozen frame once it is lent.
        unsafe { core::slice::from_raw_parts(self.pixels.as_ptr().add(y as usize * width), width) }
    }
}

/// Checked linear XRGB geometry; raw pointer access borrows the backing owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceInfo {
    pub surface: u64,
    pub width: u16,
    pub height: u16,
    pub stride: usize,
    pub bytes: usize,
}

/// One consumer-side event-ring poll.
///
/// `advanced` is queue progress, not a delivered-event count: it remains true
/// when every record consumed by this poll was stale for the current geometry
/// and [`event`](Self::event) is therefore `None`. A caller draining a batch
/// must aggregate this bit and issue one [`WindowClient::notify_consumed`]
/// after the drain so a backpressured compositor can retry pending state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EventPoll {
    pub event: Option<Event>,
    pub advanced: bool,
}

impl SurfaceInfo {
    fn checked(
        surface: u64,
        width: u16,
        height: u16,
        stride: u64,
        bytes: u64,
        mapped: usize,
    ) -> Result<Self, ClientError> {
        let stride = usize::try_from(stride).map_err(|_| ClientError::Geometry)?;
        let bytes = usize::try_from(bytes).map_err(|_| ClientError::Geometry)?;
        if surface == 0
            || width == 0
            || height == 0
            || stride % 4 != 0
            || usize::from(width)
                .checked_mul(4)
                .is_none_or(|row| row > stride)
            || stride
                .checked_mul(usize::from(height))
                .is_none_or(|span| span > bytes)
            || bytes != mapped
            || bytes > isize::MAX as usize
        {
            return Err(ClientError::Geometry);
        }
        Ok(Self {
            surface,
            width,
            height,
            stride,
            bytes,
        })
    }
}

/// No exclusive Rust pixel slice is exposed over compositor-shared memory.
pub struct Surface<'a> {
    info: SurfaceInfo,
    pixels: NonNull<u8>,
    _owner: PhantomData<&'a StashedView>,
}

impl Surface<'_> {
    pub const fn info(&self) -> SurfaceInfo {
        self.info
    }

    /// The caller's unsafe writes must stay within info().geometry, be mutually
    /// disjoint across workers, and finish before commit, resize or destruction.
    pub fn as_mut_ptr(&self) -> *mut u32 {
        self.pixels.cast::<u32>().as_ptr()
    }
}

struct Backing {
    info: SurfaceInfo,
    view: StashedView,
    pixels: NonNull<u8>,
    serial: u64,
    epoch: u64,
    presented: u64,
    refused: Option<u64>,
    destroyed: bool,
    lane: SurfaceMapLane,
}

enum CleanupOwner {
    Returned(ReturnedCapability),
    View(StashedView),
}

/// A server surface exists as soon as `SurfaceMade` succeeds, before its grant
/// has been retyped, mapped or geometry-checked. Retain that partial state so a
/// failed setup can still destroy the lessor root and retire the exact returned
/// owner instead of relying on eventual peer teardown.
struct BackingTombstone {
    surface: u64,
    owner: Option<CleanupOwner>,
    destroyed: bool,
    lane: Option<SurfaceMapLane>,
}

#[derive(Default)]
struct SurfaceLanes(u8);

impl SurfaceLanes {
    fn claim(&mut self) -> Option<SurfaceMapLane> {
        for lane in [SurfaceMapLane::First, SurfaceMapLane::Second] {
            let bit = 1_u8 << lane.index();
            if self.0 & bit == 0 {
                self.0 |= bit;
                return Some(lane);
            }
        }
        None
    }

    /// Release only after the server acknowledged lessor teardown and this
    /// exact view name reported closed. The first proves the PTE absent; the
    /// second prevents the affine owner or its arena slot from aliasing reuse.
    fn release_after_closed(&mut self, lane: SurfaceMapLane) -> bool {
        let bit = 1_u8 << lane.index();
        if self.0 & bit == 0 {
            return false;
        }
        self.0 &= !bit;
        true
    }

    #[cfg(test)]
    fn live(&self) -> u32 {
        self.0.count_ones()
    }
}

#[derive(Clone, Copy)]
enum BackingSlot {
    Current,
    Previous,
}

/// One application endpoint, one commit producer, and two incoming queues.
/// This object holds no lessor, controller, filesystem, or exec capability.
pub struct WindowClient<'w, 't, T: Transport> {
    control: SvcClient<'w, 't, T, WindowWire>,
    commits: Producer<'w, CommitWire>,
    output: Consumer<'w, OutChunk>,
    channel: crate::channel::WindowChannelSink<'w>,
    /// The menu plane ([`crate::menu`]): written here, snapshotted by the
    /// compositor on `ShowMenu` before it answers.
    menu: &'w mut [u8],
    current: Option<Backing>,
    previous: Option<Backing>,
    tombstone: Option<BackingTombstone>,
    surface_lanes: SurfaceLanes,
    configuration: Option<Configure>,
    needs_configuration: bool,
    focused: bool,
    visible: bool,
    /// The compositor's capture ask ([`WindowState::CAPTURE`]) as last
    /// synced, and whether a frame was presented since it was raised.
    ///
    /// [`WindowState::CAPTURE`]: crate::channel::WindowState::CAPTURE
    capture_asked: bool,
    capture_answered: bool,
    observed_scheme: Option<crate::appearance::ColorScheme>,
    observed_density: Option<crate::appearance::Density>,
    queried: bool,
}

/// Upper bound on one park while waiting for the compositor during
/// `present`.  Progress arrives on the doorbell (commit consumed, frame
/// presented); this only recovers a lost kick.
const PRESENT_WAIT_BOUND_NS: u64 = 50_000_000;

impl<'w, 't, T: Transport> WindowClient<'w, 't, T> {
    /// Borrow an exclusively owned client attachment, including its auxiliary
    /// queues. The caller keeps the actual mapping live for this entire borrow.
    pub fn from_memory(
        transport: &'t T,
        memory: &'w mut [u8],
        endpoints: ClientEndpoints,
    ) -> Result<Self, ClientError> {
        let pair_bytes = kore_rt::svc::window_bytes::<WindowWire>(REQ_CAP, REP_CAP);
        if memory.len() < wire::SURFACE_PAIR_BYTES || pair_bytes > memory.len() {
            return Err(ClientError::Ring);
        }
        let (pair, aux) = memory.split_at_mut(pair_bytes);
        let rings = kore_rt::svc::client_rings::<WindowWire>(pair, REQ_CAP, REP_CAP)
            .map_err(|_| ClientError::Ring)?;
        let aux = wire::split_window_aux(aux).ok_or(ClientError::Ring)?;
        Ok(Self {
            control: SvcClient::new(transport, rings, endpoints),
            commits: kore_ring::spsc::producer(aux.commits).map_err(|_| ClientError::Ring)?,
            output: kore_ring::spsc::consumer(aux.output).map_err(|_| ClientError::Ring)?,
            channel: crate::channel::sink(aux.events).map_err(|_| ClientError::Ring)?,
            menu: aux.menu,
            current: None,
            previous: None,
            tombstone: None,
            surface_lanes: SurfaceLanes::default(),
            configuration: None,
            needs_configuration: true,
            focused: false,
            visible: true,
            capture_asked: false,
            capture_answered: false,
            observed_scheme: None,
            observed_density: None,
            queried: false,
        })
    }

    pub fn with_reply_wait(mut self, bell: Doorbell) -> Self
    where
        T: ParkTransport,
    {
        self.control = self.control.with_reply_wait(bell);
        self
    }

    /// Whether any pixel of this window can reach the screen, from the
    /// channel's last synced state (`res/EVENT.md`): clear while the
    /// window is minimized, closing, or fully occluded under a higher
    /// window's opaque core.  While `false`, a client may stop producing
    /// frames and park on [`wait_change`](Self::wait_change).
    ///
    /// A screenshot of this window (`res/SCREENSHOTS.md`, window capture)
    /// reads as visible until a frame is presented for it, so an app that
    /// pauses while hidden draws its current state for the shot.
    pub const fn visible(&self) -> bool {
        self.visible
    }

    /// Park until the compositor publishes anything new on this window's
    /// channel (an event record, a configure, a presented frame, a
    /// visibility flip), or `bound` elapses.  Level-triggered: drain
    /// [`pop_event`](Self::pop_event) and re-check state after every
    /// return.
    /// Work a wait must not sleep past: a failed status sync, unconsumed
    /// records, or an appearance the caller has not yet observed.
    fn change_pending(&mut self) -> bool {
        self.sync_status().is_err()
            || self.channel.outstanding() != 0
            || self.channel.state().is_some_and(|state| {
                self.observed_scheme
                    .is_some_and(|seen| state.color_scheme() != seen)
                    || self
                        .observed_density
                        .is_some_and(|seen| state.density() != seen)
            })
    }

    pub async fn wait_change(&mut self, bound: core::time::Duration) {
        let seen = self.channel.seq();
        if self.change_pending() || self.answer_capture().await {
            return;
        }
        kore_rt::event::wait_past(&self.channel, seen, Some(bound)).await;
        let _ = self.sync_status();
    }

    /// A screenshot waits on this window and no frame has been presented
    /// since it asked: present the current backing, unchanged, so the shot
    /// is a whole frame of this window's own rather than the compositor's
    /// late copy of a backing that may be mid-paint. Called only where the
    /// app is about to park, so every writer has finished. Once per ask:
    /// a refused or overtaken frame leaves the ask to the compositor's
    /// bound. Whether a frame was offered.
    async fn answer_capture(&mut self) -> bool {
        // A capture must not publish an unpainted replacement backing.
        if !self.capture_asked || self.capture_answered || self.needs_configuration || self.needs_initial_frame() {
            return false;
        }
        self.capture_answered = true;
        let mut rects = [SurRect::default(); MAX_COMMIT_RECTS];
        rects[0] = SurRect { x: 0, y: 0, w: 1, h: 1 };
        let _ = self.present(Damage::Rects { rects, count: 1 }).await;
        true
    }

    /// [`wait_change`](Self::wait_change) for a window whose task also drains
    /// other planes woken through the same executor doorbell — a stream host
    /// whose child rings it (`res/SHELL_TERMINAL.md` §6). It publishes this
    /// channel's park intent, so a compositor batch still wakes it, and then
    /// returns on *any* executor wake: a single-plane wait would re-park after
    /// the other plane's notice and hide it. The caller must have armed every
    /// other plane before calling, drains all of them on return, and keeps
    /// its own deadlines.
    pub async fn wait_any(&mut self, bound: core::time::Duration) {
        let seen = self.channel.seq();
        if self.change_pending() || self.answer_capture().await {
            return;
        }
        kore_rt::event::wait_once(&self.channel, seen, Some(bound)).await;
        let _ = self.sync_status();
    }

    pub const fn configuration(&self) -> Option<Configure> {
        self.configuration
    }
    pub const fn configuration_needed(&self) -> bool {
        self.needs_configuration
    }
    pub const fn focused(&self) -> bool {
        self.focused
    }

    /// A new backing has no retained pixels, even if an overtaken resize
    /// returned to the application's previous dimensions. Retained renderers
    /// must paint it completely before publishing its first frame.
    pub fn needs_initial_frame(&self) -> bool {
        self.current.as_ref().is_some_and(|current| !current.destroyed && current.presented == 0)
    }

    pub fn surface(&self) -> Result<Surface<'_>, ClientError> {
        if self.needs_configuration {
            return Err(ClientError::Reconfigure);
        }
        let current = self.current.as_ref().ok_or(ClientError::NoSurface)?;
        if current.destroyed
            || current.serial == 0
            || self
                .configuration
                .is_none_or(|config| config.serial != current.serial)
        {
            return Err(ClientError::NoSurface);
        }
        Ok(Surface {
            info: current.info,
            pixels: current.pixels,
            _owner: PhantomData,
        })
    }

    pub fn pop_output(&mut self) -> Option<OutChunk> {
        self.output.pop()
    }

    /// Observe native rendering density; this also closes the state-before-park race.
    pub fn density(&mut self) -> crate::appearance::Density {
        if let Some(state) = self.channel.state() {
            self.observed_density = Some(state.density());
        }
        self.observed_density.unwrap_or_default()
    }

    /// Read and acknowledge the resolved appearance. Opting in makes
    /// `wait_change` return if another scheme arrives before the next park.
    /// A torn snapshot retains the last observed value; clients that never
    /// query appearance keep their ordinary event/visibility wait behavior.
    pub fn color_scheme(&mut self) -> crate::appearance::ColorScheme {
        if let Some(state) = self.channel.state() {
            self.observed_scheme = Some(state.color_scheme());
        }
        self.observed_scheme.unwrap_or_default()
    }

    /// Consume all event types. Old-coordinate pointer input is discarded;
    /// newer pointer input first requests a configure and resets held
    /// buttons. Keyboard snapshots, focus, menu, close and reset carry no geometry
    /// and remain actionable regardless of a superseded configure serial; a
    /// newer serial on any of them still requests the configure.
    pub fn pop_event(&mut self) -> Result<EventPoll, ClientError> {
        let mut advanced = false;
        for _ in 0..crate::event::EVENT_CAP {
            let Some(wire) = self.channel.pop() else {
                return Ok(EventPoll {
                    event: None,
                    advanced,
                });
            };
            advanced = true;
            let (serial, event) = wire.decode().ok_or(ClientError::EventMalformed)?;
            let current = self.configuration.map_or(0, |config| config.serial);
            let Some(event) = scoped_event(current, serial, event) else {
                continue;
            };
            if serial > current || event == Event::Configure {
                self.needs_configuration = true;
            }
            if let Event::Focus(focused) = event {
                self.focused = focused;
                if let Some(config) = self.configuration.as_mut() {
                    config.focused = focused;
                }
            }
            return Ok(EventPoll {
                event: Some(event),
                advanced,
            });
        }
        Ok(EventPoll {
            event: None,
            advanced,
        })
    }

    /// Output/event queues are level-triggered; a droppable kick only tells the
    /// compositor to resume a backpressured producer.
    pub async fn notify_consumed(&self) -> Result<(), ClientError> {
        self.kick().await
    }

    /// Ring the compositor's request doorbell: one droppable, level-triggered
    /// notice on this peer's kick endpoint.  The commit plane is data only —
    /// a record on it wakes nobody — so every producer-side edge the
    /// compositor must act on (a commit to present, a consumed event queue)
    /// rings this bell; the compositor pumps every plane at each wake, so a
    /// dropped notice (`WouldBlock`) is subsumed by the one already standing.
    async fn kick(&self) -> Result<(), ClientError> {
        match kore_rt::ChanTx::new(self.control.transport(), self.control.request_kick())
            .send(Notice {
                cookie: kore_rt::svc::COOKIE_REPORT,
                payload: 0,
            })
            .await
        {
            Ok(()) | Err(SendError::WouldBlock) => Ok(()),
            Err(_) => Err(ClientError::Transport),
        }
    }

    /// Re-read the authoritative placement, create exact-size backing if needed,
    /// then ACK that serial and surface. A later configure can safely overtake
    /// any await; Stale becomes Reconfigure with both backing owners retained.
    pub async fn refresh_configuration(&mut self) -> Result<bool, ClientError> {
        if !self.queried {
            match self.surface_request(SurfaceRequest::Query).await? {
                SurfaceResponse::Info {
                    format: DISPLAY_FORMAT_XRGB8888,
                    ..
                } => self.queried = true,
                _ => return Err(ClientError::Protocol),
            }
        }
        let config = self.read_configuration().await?;
        self.focused = config.focused;
        if self
            .configuration
            .is_some_and(|previous| previous.serial == config.serial)
            && self
                .current
                .as_ref()
                .is_some_and(|current| current.serial == config.serial && !current.destroyed)
        {
            self.configuration = Some(config);
            self.needs_configuration = false;
            return Ok(false);
        }
        self.needs_configuration = true;
        let matching = self.current.as_ref().is_some_and(|current| {
            !current.destroyed
                && current.info.width == config.bounds.width
                && current.info.height == config.bounds.height
        });
        if !matching {
            self.prepare_replacement().await?;
            if let Err(error) = self.create_backing(config).await {
                if error == ClientError::SurfaceRefused(SurfaceRefusal::InvalidArgument)
                    && self.read_configuration().await?.serial != config.serial
                {
                    return Err(ClientError::Reconfigure);
                }
                return Err(error);
            }
        }
        let surface = self
            .current
            .as_ref()
            .ok_or(ClientError::NoSurface)?
            .info
            .surface;
        let reply = self
            .control
            .call(&Request::Acknowledge {
                serial: config.serial,
                surface,
            })
            .await
            .map_err(|_| ClientError::Transport)?;
        match reply.response {
            Response::Accepted => {}
            Response::ControlRefused(ControlRefusal::Stale) => {
                return Err(ClientError::Reconfigure);
            }
            Response::ControlRefused(reason) => return Err(ClientError::Refused(reason)),
            _ => return Err(ClientError::Protocol),
        }
        self.current.as_mut().unwrap().serial = config.serial;
        self.configuration = Some(config);
        self.needs_configuration = false;
        Ok(true)
    }

    /// Publish and wait for actual presentation. The caller must finish every
    /// writer before entering this method. A configure that overtakes the frame
    /// returns Reconfigure; it never acknowledges stale application damage.
    pub async fn present(&mut self, damage: Damage) -> Result<u64, ClientError> {
        let _ = self.surface()?;
        let serial = self.current.as_ref().unwrap().serial;
        // A stale commit from the configuration this surface replaced can be
        // refused after that frame already returned Reconfigure.  First wait
        // until the SPSC plane proves every older record consumed — the
        // compositor pumps the plane before republishing its books, so the
        // baseline below is then past them.
        loop {
            let seen = self.channel.seq();
            self.sync_status()?;
            if self.commits.capacity_left() == self.commits.capacity() {
                break;
            }
            let published = self.published_serial();
            if published != 0 && published != serial {
                self.needs_configuration = true;
                return Err(ClientError::Reconfigure);
            }
            kore_rt::event::wait_past(
                &self.channel,
                seen,
                Some(core::time::Duration::from_nanos(PRESENT_WAIT_BOUND_NS)),
            )
            .await;
        }
        let current = self.current.as_ref().unwrap();
        let baseline = current.refused.unwrap_or(0);
        let epoch = current
            .epoch
            .checked_add(1)
            .ok_or(ClientError::EpochExhausted)?;
        let wire = wire::encode_window_commit(
            &Commit {
                surface: current.info.surface,
                epoch,
                damage,
            },
            serial,
        )
        .ok_or(ClientError::Geometry)?;
        self.commits
            .push(wire)
            .map_err(|_| ClientError::CommitQueue)?;
        self.current.as_mut().unwrap().epoch = epoch;
        // The first frame consumed after a capture ask is the one the
        // compositor copies.
        self.capture_answered |= self.capture_asked;
        // The record alone wakes nobody: without this kick the compositor
        // learns of the frame only when something else wakes it, and this
        // peer's status re-read below (bounded at `PRESENT_WAIT_BOUND_NS`)
        // becomes the cadence — the polling the bound exists to avoid.
        self.kick().await?;
        loop {
            let seen = self.channel.seq();
            self.sync_status()?;
            let current = self.current.as_ref().unwrap();
            if current.presented >= epoch {
                if let Err(error) = self.retire_backing(BackingSlot::Previous).await {
                    if !error.is_retryable() {
                        return Err(error);
                    }
                }
                return Ok(epoch);
            }
            let refused = current.refused.unwrap_or(0) > baseline;
            let published = self.published_serial();
            if published != 0 && published != serial {
                self.needs_configuration = true;
                return Err(ClientError::Reconfigure);
            }
            if refused {
                return Err(ClientError::CommitRejected);
            }
            // The kick above wakes the compositor, which pumps the commit,
            // presents, republishes the channel and — only if this side
            // parked — rings the doorbell; the bound is a safety net
            // against a lost wake, not the cadence.
            kore_rt::event::wait_past(
                &self.channel,
                seen,
                Some(core::time::Duration::from_nanos(PRESENT_WAIT_BOUND_NS)),
            )
            .await;
        }
    }

    /// Publish this window's tab strip for compositor-drawn chrome. The
    /// model is metadata only; a version older than the one already
    /// published is refused as `Stale` and the client keeps its own model.
    pub async fn set_tabs(&mut self, tabs: TabModel) -> Result<(), ClientError> {
        let reply = self
            .control
            .call(&Request::SetTabs(tabs))
            .await
            .map_err(|_| ClientError::Transport)?;
        match reply.response {
            Response::Accepted => Ok(()),
            Response::ControlRefused(reason) => Err(ClientError::Refused(reason)),
            _ => Err(ClientError::Protocol),
        }
    }

    /// Publish this window's layout hints (`res/WM_LAYOUT.md` §2): the
    /// hard limits and preferences the WM applies to every placement.
    /// Inert metadata like [`set_tabs`](Self::set_tabs); a version older
    /// than the published one is refused as `Stale`.
    pub async fn set_layout_hints(
        &mut self,
        hints: crate::control::LayoutHints,
    ) -> Result<(), ClientError> {
        let reply = self
            .control
            .call(&Request::SetLayoutHints(hints))
            .await
            .map_err(|_| ClientError::Transport)?;
        match reply.response {
            Response::Accepted => Ok(()),
            Response::ControlRefused(reason) => Err(ClientError::Refused(reason)),
            _ => Err(ClientError::Protocol),
        }
    }

    /// Graceful leaf cleanup. The controller still kills/reaps this process and
    /// closes its exact peer before releasing ancestors. Retryable failures keep
    /// the affected owner here; no successful ring enqueue counts as retirement.
    pub async fn close(&mut self) -> Result<(), ClientError> {
        self.needs_configuration = true;
        self.retire_backing(BackingSlot::Current).await?;
        self.retire_backing(BackingSlot::Previous).await?;
        self.retire_tombstone().await?;
        self.configuration = None;
        Ok(())
    }

    /// Ask the controller to close this window — the keyboard's ESC/Q and
    /// an app's own close action take the path the chrome's close dot
    /// takes. The compositor raises the same `close_requested` latch and
    /// echoes `Event::CloseRequested` down this window's channel; the app
    /// then leaves through its ordinary close path once it sees the echo,
    /// after the controller has begun its own teardown. `Ok(serial)` is
    /// the configure serial the request is held under; an already-asked
    /// window answers again rather than refusing. A `Busy` refusal is
    /// retryable: the window is already closing, or the event ring could
    /// not take the echo this instant.
    pub async fn request_close(&mut self) -> Result<u64, ClientError> {
        match self.surface_request(SurfaceRequest::RequestClose).await? {
            SurfaceResponse::CloseRequested { serial } => Ok(serial),
            _ => Err(ClientError::Protocol),
        }
    }

    /// Show `cursor` while the pointer is over this window's content
    /// ([`crate::cursor`]): the shape that says what a press there would
    /// do. It holds until the window asks again, so ask on a change, not
    /// per frame. A `Busy` refusal is retryable.
    pub async fn set_cursor(&mut self, cursor: crate::cursor::Cursor) -> Result<(), ClientError> {
        match self
            .surface_request(SurfaceRequest::SetCursor { cursor })
            .await?
        {
            SurfaceResponse::CursorSet { cursor: held } if held == cursor => Ok(()),
            _ => Err(ClientError::Protocol),
        }
    }

    /// Ask this window's controller to open the filesystem object with
    /// stable identity `object` in the default handler for its `kind`.
    /// Inert: the controller decides. A `Busy` refusal (its queue is full)
    /// is retryable.
    pub async fn open_object(
        &mut self,
        object: u64,
        kind: crate::document::DocumentKind,
    ) -> Result<(), ClientError> {
        match self
            .surface_request(SurfaceRequest::OpenObject { object, kind })
            .await?
        {
            SurfaceResponse::OpenQueued { object: queued } if queued == object => Ok(()),
            _ => Err(ClientError::Protocol),
        }
    }

    /// Show a context menu at the content-relative anchor `(x, y)`: `build`
    /// encodes it into the menu plane, and the compositor draws it once it
    /// answers. The menu ends with an [`Event::Menu`] naming `serial`
    /// (nonzero): the chosen entry's id, or zero if it was dismissed. A
    /// refusal leaves no menu open; `Busy` is retryable.
    pub async fn show_menu(
        &mut self,
        x: i32,
        y: i32,
        serial: u32,
        build: impl FnOnce(&mut crate::menu::MenuBuilder<'_>),
    ) -> Result<(), ClientError> {
        if serial == 0 {
            return Err(ClientError::Protocol);
        }
        let mut menu = crate::menu::MenuBuilder::new(self.menu);
        build(&mut menu);
        let len = menu.finish().map_err(ClientError::Menu)?;
        let len = u16::try_from(len).map_err(|_| ClientError::Protocol)?;
        match self
            .surface_request(SurfaceRequest::ShowMenu { x, y, len, serial })
            .await?
        {
            SurfaceResponse::MenuShown { serial: shown } if shown == serial => Ok(()),
            _ => Err(ClientError::Protocol),
        }
    }

    /// Ask the user, through this window's controller, to choose a file of
    /// the types `types` accepts (a nonzero mask over `kore_kit::kinds`
    /// rows). The answer arrives as [`Event::Picked`] naming `serial`
    /// (nonzero), a chosen file's name just before it in `PickedName`
    /// records ([`crate::pick::NameParts`]). `Busy` (the controller's queue
    /// is full) is retryable.
    pub async fn pick_object(&mut self, types: u64, serial: u32) -> Result<(), ClientError> {
        if types == 0 || serial == 0 {
            return Err(ClientError::Protocol);
        }
        match self
            .surface_request(SurfaceRequest::PickObject { types, serial })
            .await?
        {
            SurfaceResponse::PickQueued { serial: queued } if queued == serial => Ok(()),
            _ => Err(ClientError::Protocol),
        }
    }

    /// Bind this window to the pick it was launched for, by its `token`.
    pub async fn claim_pick(&mut self, token: u64) -> Result<(), ClientError> {
        match self.surface_request(SurfaceRequest::ClaimPick { token }).await? {
            SurfaceResponse::PickClaimed { token: claimed } if claimed == token => Ok(()),
            _ => Err(ClientError::Protocol),
        }
    }

    /// Answer the pick this window claimed: `object` and its `name` chosen,
    /// or `object` zero and no name when the user cancelled. The name is
    /// written at the head of the menu plane, which the compositor
    /// snapshots before it answers.
    pub async fn picked(&mut self, token: u64, object: u64, name: &[u8]) -> Result<(), ClientError> {
        let chosen = object != 0;
        if chosen != crate::pick::valid_name(name) || (!chosen && !name.is_empty()) {
            return Err(ClientError::Protocol);
        }
        self.menu[..name.len()].copy_from_slice(name);
        let len = name.len() as u8;
        match self
            .surface_request(SurfaceRequest::Picked { token, object, len })
            .await?
        {
            SurfaceResponse::PickAnswered { token: answered } if answered == token => Ok(()),
            _ => Err(ClientError::Protocol),
        }
    }

    /// Put `text` on this window's seat's clipboard (`res/CLIPBOARD.md`), a
    /// request plane at a time. Only the seat's focused window may, on a key
    /// press, button press or menu choice it was just delivered and has not
    /// spent ([`SurfaceRefusal::Denied`] otherwise). An empty text is no
    /// copy; one past the clipboard's bound is refused before anything is
    /// sent, and one that is not UTF-8 the compositor refuses whole.
    /// `OutOfMemory` leaves the clipboard and the gesture as they were, so a
    /// retry may succeed.
    pub async fn copy(&mut self, text: &[u8]) -> Result<(), ClientError> {
        let total = u32::try_from(text.len())
            .ok()
            .filter(|&total| total as usize <= crate::clipboard::CLIPBOARD_BYTES)
            .ok_or(ClientError::TooLong)?;
        if total == 0 {
            return Err(ClientError::Protocol);
        }
        let mut offset = 0_u32;
        for piece in text.chunks(crate::clipboard::CHUNK_BYTES.min(self.menu.len())) {
            self.menu[..piece.len()].copy_from_slice(piece);
            let len = piece.len() as u16;
            match self
                .surface_request(SurfaceRequest::CopyChunk { offset, len, total })
                .await?
            {
                SurfaceResponse::Copied { through } if through == offset + u32::from(len) => {
                    offset = through;
                }
                _ => return Err(ClientError::Protocol),
            }
        }
        Ok(())
    }

    /// Read this window's seat's clipboard (`res/CLIPBOARD.md`): `begin`
    /// learns the text's length before any of it, and may refuse it
    /// (`false`: longer than the caller can hold, [`ClientError::TooLong`]);
    /// `piece` then takes the text in order, a request plane at a time, and
    /// answers whether it wants more (`false` ends the read there: a reader
    /// that needs only the text's start reads no more of it). Answers the
    /// length, zero when the clipboard holds no text. Gated as [`Self::copy`]
    /// is; a copy landing while the read is under way refuses it
    /// ([`SurfaceRefusal::Unknown`]) rather than mixing two texts.
    pub async fn paste(
        &mut self,
        begin: impl FnOnce(u32) -> bool,
        mut piece: impl FnMut(&[u8]) -> bool,
    ) -> Result<u32, ClientError> {
        let mut begin = Some(begin);
        let mut offset = 0_u32;
        let mut buffer = [0_u8; PASTE_STAGE_BYTES];
        loop {
            let (at, len, total) = match self.surface_request(SurfaceRequest::Paste { offset }).await? {
                SurfaceResponse::Clipboard { offset: at, len, total } => (at, len, total),
                SurfaceResponse::ClipboardEmpty if offset == 0 => return Ok(0),
                _ => return Err(ClientError::Protocol),
            };
            if at != offset || usize::from(len) > self.menu.len() {
                return Err(ClientError::Protocol);
            }
            if let Some(begin) = begin.take()
                && !begin(total)
            {
                return Err(ClientError::TooLong);
            }
            // The compositor wrote the plane before it answered; read it as
            // the shared memory it is, after the reply's acquire.
            let plane = shared_plane(self.menu);
            for chunk in plane[..usize::from(len)].chunks(buffer.len()) {
                for (byte, shared) in buffer.iter_mut().zip(chunk) {
                    *byte = shared.load(core::sync::atomic::Ordering::Relaxed);
                }
                if !piece(&buffer[..chunk.len()]) {
                    return Ok(total);
                }
            }
            offset += u32::from(len);
            if offset == total {
                return Ok(total);
            }
        }
    }

    /// Ask for the clipboard to be typed into this terminal window's shell
    /// line (`res/CLIPBOARD.md` §Shell): queued for its controller, which
    /// takes what the line can hold. Gated as [`Self::copy`] is, and only the
    /// seat's terminal window may ask. Answers the text's length, zero when
    /// the clipboard holds no text.
    pub async fn paste_to_shell(&mut self) -> Result<u32, ClientError> {
        match self.surface_request(SurfaceRequest::PasteToShell).await? {
            SurfaceResponse::PasteQueued { total } => Ok(total),
            SurfaceResponse::ClipboardEmpty => Ok(0),
            _ => Err(ClientError::Protocol),
        }
    }

    async fn read_configuration(&mut self) -> Result<Configure, ClientError> {
        let reply = self
            .control
            .call(&Request::Configure)
            .await
            .map_err(|_| ClientError::Transport)?;
        let config = match reply.response {
            Response::Configured(config) => config,
            Response::ControlRefused(reason) => return Err(ClientError::Refused(reason)),
            _ => return Err(ClientError::Protocol),
        };
        if !valid_configuration(self.configuration, config) {
            return Err(ClientError::Protocol);
        }
        Ok(config)
    }

    async fn surface_request(
        &mut self,
        request: SurfaceRequest,
    ) -> Result<SurfaceResponse, ClientError> {
        let reply = self
            .control
            .call(&Request::Surface(request))
            .await
            .map_err(|_| ClientError::Transport)?;
        match reply.response {
            Response::Surface(SurfaceResponse::Refused { refusal, .. }) => {
                Err(ClientError::SurfaceRefused(refusal))
            }
            Response::Surface(response) => Ok(response),
            _ => Err(ClientError::Protocol),
        }
    }

    /// Fold the channel's published books into the current backing — the
    /// memory-read replacement for the per-frame `Status` round trip
    /// (`res/EVENT.md`).  The state names the peer's **active** surface;
    /// while the current backing is not it (fresh, or superseded), its
    /// books read as a fresh surface's (`presented = 0`, `refused = 0`).
    /// A torn snapshot (the bounded seqlock cap) changes nothing; the
    /// caller's loop re-reads after the next wake.
    fn sync_status(&mut self) -> Result<(), ClientError> {
        let Some(state) = self.channel.state() else {
            return Ok(());
        };
        use crate::channel::WindowState;
        self.capture_asked = state.flags & WindowState::CAPTURE != 0;
        // A new ask re-arms only after one was seen settled.
        self.capture_answered &= self.capture_asked;
        self.visible = state.flags & (WindowState::VISIBLE | WindowState::CAPTURE) != 0;
        let Some(current) = self.current.as_mut() else {
            return Ok(());
        };
        if state.surface == current.info.surface {
            if state.presented > current.epoch
                || state.presented < current.presented
                || current
                    .refused
                    .is_some_and(|old| state.refused_commits < old)
            {
                return Err(ClientError::Protocol);
            }
            current.presented = state.presented;
            current.refused = Some(state.refused_commits);
        } else if current.refused.is_none() {
            current.refused = Some(0);
        }
        Ok(())
    }

    /// The compositor-published configure serial, `0` before the first
    /// publish — `present`'s drift check, from memory instead of a
    /// `Configure` round trip.
    fn published_serial(&self) -> u64 {
        self.channel
            .state()
            .map_or(0, |state| state.configure_serial)
    }

    async fn prepare_replacement(&mut self) -> Result<(), ClientError> {
        self.retire_tombstone().await?;
        // Wait out the earlier commit plane before deciding whether this
        // candidate ever became the displayed backing (the compositor pumps
        // the plane before republishing its books).  No writer is live.
        if self.current.as_ref().is_some_and(|current| {
            !current.destroyed && current.presented == 0 && current.epoch != 0
        }) {
            while self.commits.capacity_left() != self.commits.capacity() {
                let seen = self.channel.seq();
                if self.commits.capacity_left() == self.commits.capacity() {
                    break;
                }
                kore_rt::event::wait_past(
                    &self.channel,
                    seen,
                    Some(core::time::Duration::from_nanos(PRESENT_WAIT_BOUND_NS)),
                )
                .await;
            }
            self.sync_status()?;
        }
        if self.previous.is_some() {
            if self
                .current
                .as_ref()
                .is_some_and(|current| current.presented != 0)
            {
                self.retire_backing(BackingSlot::Previous).await?;
            } else {
                // A second configure superseded a never-presented candidate.
                // Keep the last displayed previous backing, not that candidate.
                self.retire_backing(BackingSlot::Current).await?;
            }
        }
        if self.current.is_some() {
            if self
                .current
                .as_ref()
                .is_some_and(|current| current.presented != 0 && !current.destroyed)
            {
                debug_assert!(self.previous.is_none());
                self.previous = self.current.take();
            } else {
                self.retire_backing(BackingSlot::Current).await?;
            }
        }
        Ok(())
    }

    /// The capture writer's one verb (`res/SCREENSHOTS.md`): borrow the
    /// frozen frame this pair was admitted for. The lend carries ACCESS
    /// without MODIFY, so the kernel maps it with no write permission; the
    /// frame is checked against the mapping before any pixel is read.
    pub async fn take_capture(&mut self) -> Result<CaptureView, ClientError> {
        let mut reply = self
            .control
            .call(&Request::Surface(SurfaceRequest::TakeCapture))
            .await
            .map_err(|_| ClientError::Transport)?;
        let frame = match reply.response {
            Response::Surface(SurfaceResponse::Capture(frame)) => frame,
            Response::Surface(SurfaceResponse::Refused { refusal, .. }) => {
                return Err(ClientError::SurfaceRefused(refusal));
            }
            _ => return Err(ClientError::Protocol),
        };
        let returned = reply.retained[0].take().ok_or(ClientError::Protocol)?;
        let view = returned
            .try_into_stashed_view(Rights::ACCESS)
            .map_err(|_| ClientError::Protocol)?;
        let lane = self.surface_lanes.claim().ok_or(ClientError::Busy)?;
        let mapped =
            verbs::map_stashed_view_surface_lane(&view, lane).ok_or(ClientError::Mapping)?;
        let pixels = usize::try_from(frame.pixel_bytes()).map_err(|_| ClientError::Geometry)?;
        if !frame.is_valid()
            || mapped.len() < pixels
            || mapped.cast::<u8>().as_ptr().addr() % 4 != 0
        {
            return Err(ClientError::Geometry);
        }
        Ok(CaptureView {
            frame,
            pixels: mapped.cast::<u32>(),
            _view: view,
        })
    }

    /// Read or change preferences on a Settings-authorized pair.
    pub async fn desktop(
        &mut self,
        request: crate::desktop::DesktopRequest,
    ) -> Result<Response, ClientError> {
        // Shared uploads must retain their returned owner in `wallpaper`.
        if matches!(request, crate::desktop::DesktopRequest::CreateWallpaper { .. }) {
            return Err(ClientError::Protocol);
        }
        let reply = self
            .control
            .call(&Request::Desktop(request))
            .await
            .map_err(|_| ClientError::Transport)?;
        match reply.response {
            Response::ControlRefused(reason) => Err(ClientError::Refused(reason)),
            Response::Surface(SurfaceResponse::Refused { refusal, .. }) => {
                Err(ClientError::SurfaceRefused(refusal))
            }
            response => Ok(response),
        }
    }

    /// Upload an output-sized wallpaper through one temporary shared surface.
    /// Cleanup keeps its exact view and lane in `tombstone` until revocation
    /// settles; no mapping is reused after an ambiguous response.
    pub async fn wallpaper(
        &mut self,
        width: u16,
        height: u16,
        mut pixel: impl FnMut(u32, u32) -> u32,
    ) -> Result<(), ClientError> {
        self.retire_tombstone().await?;
        let mut reply = self
            .control
            .call(&Request::Desktop(
                crate::desktop::DesktopRequest::CreateWallpaper { width, height },
            ))
            .await
            .map_err(|_| ClientError::Transport)?;
        let (surface, stride, bytes) = match reply.response {
            Response::Surface(SurfaceResponse::SurfaceMade {
                surface,
                stride,
                bytes,
            }) => (surface, stride, bytes),
            Response::ControlRefused(reason) => return Err(ClientError::Refused(reason)),
            Response::Surface(SurfaceResponse::Refused { refusal, .. }) => {
                return Err(ClientError::SurfaceRefused(refusal));
            }
            _ => return Err(ClientError::Protocol),
        };
        self.tombstone = Some(BackingTombstone {
            surface,
            owner: reply.retained[0].take().map(CleanupOwner::Returned),
            destroyed: false,
            lane: None,
        });
        let Some(CleanupOwner::Returned(returned)) =
            self.tombstone.as_mut().and_then(|t| t.owner.take())
        else {
            return Err(ClientError::Protocol);
        };
        let view = match returned.try_into_stashed_view(Rights::ACCESS | Rights::MODIFY) {
            Ok(view) => view,
            Err(returned) => {
                self.tombstone.as_mut().unwrap().owner = Some(CleanupOwner::Returned(returned));
                return Err(ClientError::Protocol);
            }
        };
        self.tombstone.as_mut().unwrap().owner = Some(CleanupOwner::View(view));
        let lane = self.surface_lanes.claim().ok_or(ClientError::Busy)?;
        self.tombstone.as_mut().unwrap().lane = Some(lane);
        let Some(CleanupOwner::View(view)) = self.tombstone.as_ref().and_then(|t| t.owner.as_ref())
        else {
            return Err(ClientError::Protocol);
        };
        let mapped =
            verbs::map_stashed_view_surface_lane(view, lane).ok_or(ClientError::Mapping)?;
        let info = SurfaceInfo::checked(surface, width, height, stride, bytes, mapped.len())?;
        if mapped.cast::<u8>().as_ptr().addr() % 4 != 0 {
            return Err(ClientError::Geometry);
        }
        for y in 0..u32::from(height) {
            for x in 0..u32::from(width) {
                // SAFETY: checked stride and span; this process is the upload's sole
                // writer, and the compositor reads only after the following call.
                unsafe {
                    mapped
                        .cast::<u32>()
                        .as_ptr()
                        .add(y as usize * info.stride / 4 + x as usize)
                        .write(pixel(x, y));
                }
            }
        }
        let applied = self
            .desktop(crate::desktop::DesktopRequest::ApplyWallpaper { surface })
            .await;
        let cleanup = self.retire_tombstone().await;
        match applied? {
            Response::Accepted => cleanup,
            _ => Err(ClientError::Protocol),
        }
    }

    async fn create_backing(&mut self, config: Configure) -> Result<(), ClientError> {
        debug_assert!(self.current.is_none());
        debug_assert!(self.tombstone.is_none());
        let mut reply = self
            .control
            .call(&Request::Surface(SurfaceRequest::CreateSurface {
                width: config.bounds.width,
                height: config.bounds.height,
                format: DISPLAY_FORMAT_XRGB8888,
                reach: SurReqWire::REACH_CPU,
                modifier: SurReqWire::MODIFIER_LINEAR,
            }))
            .await
            .map_err(|_| ClientError::Transport)?;
        let (surface, stride, bytes) = match reply.response {
            Response::Surface(SurfaceResponse::SurfaceMade {
                surface,
                stride,
                bytes,
            }) => (surface, stride, bytes),
            Response::Surface(SurfaceResponse::Refused { refusal, .. }) => {
                return Err(ClientError::SurfaceRefused(refusal));
            }
            _ => return Err(ClientError::Protocol),
        };
        if surface == 0 {
            return Err(ClientError::Protocol);
        }
        self.tombstone = Some(BackingTombstone {
            surface,
            owner: reply.retained[0].take().map(CleanupOwner::Returned),
            destroyed: false,
            lane: None,
        });
        let owner = self
            .tombstone
            .as_mut()
            .unwrap()
            .owner
            .take()
            .ok_or(ClientError::Protocol)?;
        let CleanupOwner::Returned(returned) = owner else {
            unreachable!()
        };
        let view = match returned.try_into_stashed_view(Rights::ACCESS | Rights::MODIFY) {
            Ok(view) => view,
            Err(returned) => {
                self.tombstone.as_mut().unwrap().owner = Some(CleanupOwner::Returned(returned));
                return Err(ClientError::Protocol);
            }
        };
        self.tombstone.as_mut().unwrap().owner = Some(CleanupOwner::View(view));
        let lane = self.surface_lanes.claim().ok_or(ClientError::Busy)?;
        self.tombstone.as_mut().unwrap().lane = Some(lane);
        let view = match self.tombstone.as_ref().unwrap().owner.as_ref() {
            Some(CleanupOwner::View(view)) => view,
            _ => unreachable!(),
        };
        let pixels =
            verbs::map_stashed_view_surface_lane(view, lane).ok_or(ClientError::Mapping)?;
        let info = SurfaceInfo::checked(
            surface,
            config.bounds.width,
            config.bounds.height,
            stride,
            bytes,
            pixels.len(),
        )?;
        if pixels.cast::<u8>().as_ptr().addr() % 4 != 0 {
            return Err(ClientError::Geometry);
        }
        let owner = self.tombstone.as_mut().unwrap().owner.take().unwrap();
        let CleanupOwner::View(view) = owner else {
            unreachable!()
        };
        self.tombstone = None;
        self.current = Some(Backing {
            info,
            view,
            pixels: pixels.cast(),
            serial: 0,
            epoch: 0,
            presented: 0,
            refused: None,
            destroyed: false,
            lane,
        });
        Ok(())
    }

    async fn retire_tombstone(&mut self) -> Result<(), ClientError> {
        let Some(tombstone) = self.tombstone.as_ref() else {
            return Ok(());
        };
        let surface = tombstone.surface;
        if !tombstone.destroyed {
            match self
                .surface_request(SurfaceRequest::DestroySurface { surface })
                .await
            {
                Ok(SurfaceResponse::Destroyed { surface: actual }) if actual == surface => {}
                Err(ClientError::SurfaceRefused(SurfaceRefusal::Unknown)) => {}
                Err(error) => return Err(error),
                _ => return Err(ClientError::Protocol),
            }
            self.tombstone.as_mut().unwrap().destroyed = true;
        }
        if let Some(owner) = self.tombstone.as_mut().unwrap().owner.as_mut() {
            let closed = match owner {
                CleanupOwner::Returned(owner) => owner.try_retire(),
                CleanupOwner::View(owner) => owner.try_retire_outcome().is_closed(),
            };
            if !closed {
                return Err(ClientError::Busy);
            }
        }
        if let Some(lane) = self.tombstone.as_ref().unwrap().lane {
            if !self.surface_lanes.release_after_closed(lane) {
                return Err(ClientError::Protocol);
            }
        }
        self.tombstone = None;
        Ok(())
    }

    async fn retire_backing(&mut self, which: BackingSlot) -> Result<(), ClientError> {
        let backing = match which {
            BackingSlot::Current => &self.current,
            BackingSlot::Previous => &self.previous,
        };
        let Some(backing) = backing.as_ref() else {
            return Ok(());
        };
        let surface = backing.info.surface;
        if !backing.destroyed {
            match self
                .surface_request(SurfaceRequest::DestroySurface { surface })
                .await
            {
                Ok(SurfaceResponse::Destroyed { surface: actual }) if actual == surface => {}
                // The lessor already proved it dead during a previous cleanup.
                Err(ClientError::SurfaceRefused(SurfaceRefusal::Unknown)) => {}
                Err(error) => return Err(error),
                _ => return Err(ClientError::Protocol),
            }
            match which {
                BackingSlot::Current => &mut self.current,
                BackingSlot::Previous => &mut self.previous,
            }
            .as_mut()
            .unwrap()
            .destroyed = true;
        }
        let backing = match which {
            BackingSlot::Current => &mut self.current,
            BackingSlot::Previous => &mut self.previous,
        };
        let lane = {
            let backing = backing.as_mut().unwrap();
            if !backing.view.try_retire_outcome().is_closed() {
                return Err(ClientError::Busy);
            }
            backing.lane
        };
        if !self.surface_lanes.release_after_closed(lane) {
            return Err(ClientError::Protocol);
        }
        *backing = None;
        Ok(())
    }
}

/// A paste's piece reaches its reader through this much stack at a time:
/// the plane is read atomically, a byte at a time, and handed on as a slice.
/// Small enough for any task's stack, large enough that a 4 KiB piece is
/// sixteen calls.
const PASTE_STAGE_BYTES: usize = 256;

/// The request plane as the memory it is: shared with the compositor, which
/// writes it while it answers a paste.
fn shared_plane(plane: &[u8]) -> &[core::sync::atomic::AtomicU8] {
    // SAFETY: `AtomicU8` has `u8`'s size and alignment, and every access the
    // returned view makes is atomic, so the compositor's writes race with none
    // of this process's plain accesses: the plane's plain writers run only
    // between requests, never while this view is live.
    unsafe { &*(core::ptr::from_ref(plane) as *const [core::sync::atomic::AtomicU8]) }
}

fn valid_configuration(previous: Option<Configure>, next: Configure) -> bool {
    next.window != 0
        && next.serial != 0
        && next.bounds.is_valid()
        && u32::from(next.bounds.x) + u32::from(next.bounds.width) <= u32::from(u16::MAX)
        && u32::from(next.bounds.y) + u32::from(next.bounds.height) <= u32::from(u16::MAX)
        && previous.is_none_or(|previous| {
            next.window == previous.window
                && next.serial >= previous.serial
                && (next.serial != previous.serial
                    || (next.bounds.width == previous.bounds.width
                        && next.bounds.height == previous.bounds.height))
        })
}

fn scoped_event(current: u64, serial: u64, event: Event) -> Option<Event> {
    // A configure serial scopes geometry. A pointer report is geometry — its
    // coordinates mean nothing under a superseded aperture — but a keyboard
    // snapshot carries none, and the app derives presses by diffing it
    // against its own previous snapshot, so a held key across a re-layout
    // cannot become a phantom press. Dropping or resetting it instead loses
    // the character outright: the compositor stamps keys with the layout
    // still on screen while this client may already hold a newer serial it
    // has not presented, and two quick re-layouts leave that window open
    // exactly when a user types into the window that just grew (#1054).
    //
    // Focus carries no geometry either, and the compositor stamps it with
    // the live configure serial, so during a re-layout it is always ahead
    // of this client. Rewriting it to a Reset left the app unaware it was
    // focused until the app layer's corrective Focus after the drain — too
    // late for a key drained in the same pass, which the app's own focus
    // guard then discarded. A menu choice names its own menu serial, not
    // geometry, so a reconfigure while it was open cannot invalidate it.
    if matches!(
        event,
        Event::Reset
            | Event::CloseRequested
            | Event::Keyboard { .. }
            | Event::Focus(_)
            | Event::Menu { .. }
    ) {
        return Some(event);
    }
    if serial < current {
        return None;
    }
    if serial > current && !matches!(event, Event::Configure) {
        return Some(Event::Reset);
    }
    Some(event)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::EventWire;

    use core::{cell::Cell, convert::Infallible};
    use kore_abi::session::{Committed, Status};
    use kore_rt::session::SessionBuilder;

    struct CountingTransport {
        commits: Cell<u32>,
    }

    impl CountingTransport {
        const fn new() -> Self {
            Self {
                commits: Cell::new(0),
            }
        }
    }

    impl Transport for CountingTransport {
        type Error = Infallible;

        async fn commit(&self, session: &SessionBuilder) -> Result<Committed, Self::Error> {
            assert_ne!(session.staged(), 0);
            self.commits.set(self.commits.get() + 1);
            Ok(Committed::from_raw(Status::Ok.encode(0)))
        }

        fn commit_now(&self, session: &SessionBuilder) -> Result<Committed, Self::Error> {
            assert_ne!(session.staged(), 0);
            self.commits.set(self.commits.get() + 1);
            Ok(Committed::from_raw(Status::Ok.encode(0)))
        }
    }

    fn drive<F: core::future::Future>(future: F) -> F::Output {
        kore_rt::try_block_on(future, 8).expect("future did not resolve")
    }

    #[test]
    fn density_published_before_wait_is_observed_without_parking() {
        use crate::channel::WindowState;
        use kore_rt::svc::SvcProtocol;
        #[repr(align(64))]
        struct Buffer<const N: usize>([u8; N]);
        let mut pair = Buffer([0; wire::SURFACE_PAIR_BYTES]);
        let mut event_memory = Buffer([0; 4096]);
        WindowWire::init_pair_window(&mut pair.0, REQ_CAP, REP_CAP).unwrap();
        let transport = CountingTransport::new();
        let mut client = WindowClient::from_memory(
            &transport,
            &mut pair.0,
            ClientEndpoints {
                kick: SlotRef::from_wire(1),
                grants: [const { None }; MAX_GRANTS],
            },
        )
        .unwrap();
        let (mut producer, channel) = kore_ring::event::split::<WindowState, EventWire>(
            &mut event_memory.0,
            crate::event::EVENT_CAP,
        )
        .unwrap();
        client.channel = channel;
        let original_flags = WindowState::VISIBLE | WindowState::FOCUSED | WindowState::LIGHT;
        let mut state = WindowState {
            flags: original_flags,
            ..WindowState::default()
        };
        producer.replace_state(state);
        assert_eq!(client.density(), crate::appearance::Density::Native);
        state.set_density(crate::appearance::Density::Double);
        producer.replace_state(state);
        // No event record: the state change and its wake can precede wait_change.
        // Poll once; an actual event wait would be Pending on the unchanged seq.
        let mut future = core::pin::pin!(client.wait_change(core::time::Duration::from_secs(1)));
        let mut cx = core::task::Context::from_waker(core::task::Waker::noop());
        assert!(core::future::Future::poll(future.as_mut(), &mut cx).is_ready());
        drop(future);
        assert_eq!(
            state.flags & (WindowState::FOCUSED | WindowState::VISIBLE | WindowState::LIGHT),
            original_flags
        );
        state.set_density(crate::appearance::Density::Native);
        assert_eq!(state.density(), crate::appearance::Density::Native);
        assert_eq!(
            state.flags & (WindowState::FOCUSED | WindowState::VISIBLE | WindowState::LIGHT),
            original_flags
        );
    }

    /// A capture ask reads as visibility and is answered once before a park:
    /// the same ask never draws a second frame, however the first fared,
    /// and a later ask re-arms only after the first was seen settled.
    #[test]
    fn a_capture_ask_is_visible_and_answered_once_per_ask() {
        use crate::channel::WindowState;
        use kore_rt::svc::SvcProtocol;
        #[repr(align(64))]
        struct Buffer<const N: usize>([u8; N]);
        let mut pair = Buffer([0; wire::SURFACE_PAIR_BYTES]);
        let mut event_memory = Buffer([0; 4096]);
        WindowWire::init_pair_window(&mut pair.0, REQ_CAP, REP_CAP).unwrap();
        let transport = CountingTransport::new();
        let mut client = WindowClient::from_memory(
            &transport,
            &mut pair.0,
            ClientEndpoints {
                kick: SlotRef::from_wire(1),
                grants: [const { None }; MAX_GRANTS],
            },
        )
        .unwrap();
        let (mut producer, channel) = kore_ring::event::split::<WindowState, EventWire>(
            &mut event_memory.0,
            crate::event::EVENT_CAP,
        )
        .unwrap();
        client.channel = channel;

        // Minimized, and the app has not configured yet: the ask reads as
        // visibility, but there is no frame to offer.
        let mut state = WindowState::default();
        producer.replace_state(state);
        client.sync_status().unwrap();
        assert!(!client.visible());
        state.flags = WindowState::CAPTURE;
        producer.replace_state(state);
        client.sync_status().unwrap();
        assert!(client.visible(), "an ask reads as visible");
        assert!(!drive(client.answer_capture()), "no configuration: nothing to present");
        assert!(!client.capture_answered);

        // Configured, the backing not yet live: the one offer is spent even
        // though the frame failed, so a wait returns once and then parks.
        client.needs_configuration = false;
        assert!(drive(client.answer_capture()), "the ask is answered before parking");
        assert!(client.capture_answered);
        assert!(!drive(client.answer_capture()), "the same ask never draws twice");
        client.sync_status().unwrap();
        assert!(client.capture_answered, "a resync of the same ask keeps it spent");
        assert_eq!(transport.commits.get(), 0, "a frame that failed kicks nobody");

        // The compositor settles the ask: the window is hidden again, and a
        // fresh ask is answered anew.
        state.flags = 0;
        producer.replace_state(state);
        client.sync_status().unwrap();
        assert!(!client.visible() && !client.capture_answered);
        assert!(!drive(client.answer_capture()));
        state.flags = WindowState::CAPTURE | WindowState::VISIBLE;
        producer.replace_state(state);
        client.sync_status().unwrap();
        assert!(drive(client.answer_capture()), "a new ask is answered");
        assert!(client.visible());
    }

    #[test]
    fn actual_event_queue_filters_old_geometry_and_updates_focus() {
        use kore_rt::svc::SvcProtocol;
        #[repr(align(64))]
        struct Buffer<const N: usize>([u8; N]);
        let mut pair = Buffer([0; wire::SURFACE_PAIR_BYTES]);
        let mut event_memory = Buffer([0; 4096]);
        WindowWire::init_pair_window(&mut pair.0, REQ_CAP, REP_CAP).unwrap();
        let transport = kore_rt::transport::TrapTransport::new();
        let mut client = WindowClient::from_memory(
            &transport,
            &mut pair.0,
            ClientEndpoints {
                kick: SlotRef::from_wire(1),
                grants: [const { None }; MAX_GRANTS],
            },
        )
        .unwrap();
        // Only the typed SPSC halves share this host buffer, just as the
        // service/client halves do. No capability or successful trap is faked.
        let (mut producer, channel) = kore_ring::event::split::<
            crate::channel::WindowState,
            EventWire,
        >(&mut event_memory.0, crate::event::EVENT_CAP)
        .unwrap();
        client.channel = channel;
        client.configuration = Some(Configure {
            window: 1,
            serial: 2,
            bounds: crate::configure::Bounds {
                x: 0,
                y: 0,
                width: 1280,
                height: 800,
            },
            focused: false,
        });
        client.needs_configuration = false;
        // Focus is not geometry and is never filtered by serial (see
        // `a_focus_change_reaches_the_app_before_the_key_typed_under_it`);
        // the old-geometry filtering under test here is the pointer's.
        producer
            .push(EventWire::encode(2, Event::Focus(true)))
            .unwrap();
        assert_eq!(
            client.pop_event(),
            Ok(EventPoll {
                event: Some(Event::Focus(true)),
                advanced: true,
            })
        );
        assert!(client.focused());
        assert!(client.configuration().unwrap().focused);
        assert!(!client.configuration_needed());
        producer
            .push(EventWire::encode(
                3,
                Event::Pointer {
                    x: 10,
                    y: 20,
                    buttons: 1,
                    scroll: crate::event::Scroll::NONE,
                },
            ))
            .unwrap();
        assert_eq!(
            client.pop_event(),
            Ok(EventPoll {
                event: Some(Event::Reset),
                advanced: true,
            })
        );
        assert!(client.configuration_needed());
        producer
            .push(EventWire::encode(3, Event::Configure))
            .unwrap();
        assert_eq!(
            client.pop_event(),
            Ok(EventPoll {
                event: Some(Event::Configure),
                advanced: true,
            })
        );
        producer
            .push(EventWire::encode(1, Event::CloseRequested))
            .unwrap();
        assert_eq!(
            client.pop_event(),
            Ok(EventPoll {
                event: Some(Event::CloseRequested),
                advanced: true,
            })
        );
        assert_eq!(
            client.pop_event(),
            Ok(EventPoll {
                event: None,
                advanced: false,
            })
        );
        producer.push(EventWire::encode(0, Event::Reset)).unwrap();
        assert_eq!(client.pop_event(), Err(ClientError::EventMalformed));
    }

    #[test]
    fn stale_only_drain_still_kicks_the_presenter_retry() {
        use kore_rt::svc::SvcProtocol;

        #[repr(align(64))]
        struct Buffer<const N: usize>([u8; N]);

        let mut pair = Buffer([0; wire::SURFACE_PAIR_BYTES]);
        let mut event_memory = Buffer([0; 4096]);
        WindowWire::init_pair_window(&mut pair.0, REQ_CAP, REP_CAP).unwrap();
        let transport = CountingTransport::new();
        let mut client = WindowClient::from_memory(
            &transport,
            &mut pair.0,
            ClientEndpoints {
                kick: SlotRef::from_wire(1),
                grants: [const { None }; MAX_GRANTS],
            },
        )
        .unwrap();
        let (mut producer, channel) = kore_ring::event::split::<
            crate::channel::WindowState,
            EventWire,
        >(&mut event_memory.0, crate::event::EVENT_CAP)
        .unwrap();
        client.channel = channel;
        client.configuration = Some(Configure {
            window: 1,
            serial: 2,
            bounds: crate::configure::Bounds {
                x: 0,
                y: 0,
                width: 1280,
                height: 800,
            },
            focused: true,
        });
        client.needs_configuration = false;

        // A resize can overtake coordinate records already in the bounded
        // ring. They are intentionally hidden from the application, but
        // consuming them still returns producer capacity and must wake a
        // pending Reset retry. (Keyboard snapshots are not hidden this way;
        // see `a_keystroke_crosses_a_configure_boundary_in_either_direction`.)
        producer
            .push(EventWire::encode(
                1,
                Event::Pointer {
                    x: 23,
                    y: 12,
                    buttons: 1,
                    scroll: crate::event::Scroll::NONE,
                },
            ))
            .unwrap();
        let poll = client.pop_event().unwrap();
        assert_eq!(
            poll,
            EventPoll {
                event: None,
                advanced: true,
            }
        );
        if poll.advanced {
            drive(client.notify_consumed()).unwrap();
        }
        assert_eq!(transport.commits.get(), 1, "one kick for the drain");

        let empty = client.pop_event().unwrap();
        assert_eq!(
            empty,
            EventPoll {
                event: None,
                advanced: false,
            }
        );
        if empty.advanced {
            drive(client.notify_consumed()).unwrap();
        }
        assert_eq!(
            transport.commits.get(),
            1,
            "an empty poll adds no redundant kick"
        );
    }

    #[test]
    fn configuration_serial_fixes_shape_but_allows_focus_and_origin_changes() {
        let previous = Configure {
            window: 1,
            serial: 7,
            bounds: crate::configure::Bounds {
                x: 10,
                y: 20,
                width: 640,
                height: 480,
            },
            focused: false,
        };
        assert!(valid_configuration(None, previous));
        assert!(valid_configuration(Some(previous), previous));
        assert!(valid_configuration(
            Some(previous),
            Configure {
                focused: true,
                ..previous
            }
        ));
        assert!(valid_configuration(
            Some(previous),
            Configure {
                bounds: crate::configure::Bounds {
                    x: 120,
                    y: 75,
                    ..previous.bounds
                },
                ..previous
            }
        ));
        let resized = Configure {
            bounds: crate::configure::Bounds {
                width: 800,
                ..previous.bounds
            },
            ..previous
        };
        assert!(!valid_configuration(Some(previous), resized));
        assert!(valid_configuration(
            Some(previous),
            Configure {
                serial: 8,
                ..resized
            }
        ));
        for next in [
            Configure {
                window: 0,
                ..previous
            },
            Configure {
                serial: 0,
                ..previous
            },
            Configure {
                serial: 6,
                ..previous
            },
            Configure {
                window: 2,
                ..previous
            },
            Configure {
                bounds: crate::configure::Bounds {
                    width: 0,
                    ..previous.bounds
                },
                ..previous
            },
            Configure {
                bounds: crate::configure::Bounds {
                    x: u16::MAX,
                    ..previous.bounds
                },
                ..previous
            },
            Configure {
                bounds: crate::configure::Bounds {
                    y: u16::MAX,
                    ..previous.bounds
                },
                ..previous
            },
        ] {
            assert!(!valid_configuration(Some(previous), next));
        }
    }

    #[test]
    fn checked_backings_refuse_overflow_short_stride_and_wrong_mapping() {
        let good =
            SurfaceInfo::checked(1, 1920, 1200, 7680, 10 * 1024 * 1024, 10 * 1024 * 1024).unwrap();
        assert_eq!(good.width, 1920);
        for (surface, width, height, stride, bytes, mapped) in [
            (0, 1920, 1200, 7680, 10_000_000, 10_000_000),
            (1, 0, 1200, 7680, 10_000_000, 10_000_000),
            (1, 1920, 1200, 7676, 10_000_000, 10_000_000),
            (1, 1920, 1200, 7681, 10_000_000, 10_000_000),
            (1, 1920, 1200, 7680, 1000, 1000),
            (1, 1920, 1200, 7680, 10_000_000, 9_000_000),
            (1, 1920, 1200, u64::MAX - 3, u64::MAX, usize::MAX),
        ] {
            assert_eq!(
                SurfaceInfo::checked(surface, width, height, stride, bytes, mapped),
                Err(ClientError::Geometry)
            );
        }
    }

    #[test]
    fn surface_lanes_fail_closed_while_live_and_reuse_after_closed_churn() {
        const FRAME_BYTES: u64 = 10 * 1024 * 1024;
        const OLD_WINDOW_BYTES: u64 = 64 * 1024 * 1024;

        let mut lanes = SurfaceLanes::default();
        let first = lanes.claim().expect("first live surface");
        let second = lanes.claim().expect("previous live surface");
        assert_eq!(first, SurfaceMapLane::First);
        assert_eq!(second, SurfaceMapLane::Second);
        assert_eq!(lanes.live(), 2);
        assert_eq!(lanes.claim(), None, "a third live mapping must stay busy");

        // A retryable destroy/view-close leaves the bit claimed. Only a later
        // acknowledged lessor teardown followed by proven view closure invokes
        // release_after_closed.
        assert_eq!(lanes.claim(), None, "a retry must not imply vacancy");

        let mut cumulative = 2 * FRAME_BYTES;
        for generation in 0..128 {
            let retired = if generation & 1 == 0 { first } else { second };
            assert!(lanes.release_after_closed(retired));
            assert_eq!(lanes.live(), 1);
            assert_eq!(lanes.claim(), Some(retired), "reuse the exact vacant lane");
            assert_eq!(lanes.live(), 2);
            cumulative += FRAME_BYTES;
        }
        assert!(
            cumulative > OLD_WINDOW_BYTES,
            "fixture must exceed the former cumulative VA budget"
        );

        assert!(lanes.release_after_closed(first));
        assert!(
            !lanes.release_after_closed(first),
            "double release fails closed"
        );
        assert!(lanes.release_after_closed(second));
        assert_eq!(lanes.live(), 0);
    }

    /// #1054: a keyboard snapshot has no geometry, so a configure serial must
    /// never decide whether it reaches the app. Across two quick re-layouts
    /// the compositor keeps stamping keys with the layout still on screen
    /// while this client has already acknowledged a newer configure it has
    /// not presented yet; a key typed then was silently dropped as stale, and
    /// one typed under an even newer stamp was replaced by a Reset. Either
    /// way the character was gone and the window looked focused and alive.
    #[test]
    fn a_keystroke_crosses_a_configure_boundary_in_either_direction() {
        let keyboard = Event::Keyboard {
            source: 1,
            modifiers: 0x02,
            keys: [0x1e, 0, 0, 0, 0, 0],
            locks: kore_abi::keyboard::Locks::NONE,
        };
        assert_eq!(
            scoped_event(2, 1, keyboard),
            Some(keyboard),
            "a key stamped with the still-presented layout is not stale input"
        );
        assert_eq!(
            scoped_event(2, 3, keyboard),
            Some(keyboard),
            "a key stamped ahead of this client's configure is still the key"
        );
    }

    /// #1054, the half that decides whether the delivered key is *used*: the
    /// compositor stamps `Focus` with the live configure serial, so during a
    /// re-layout it is always ahead of this client and was rewritten to a
    /// Reset. The app never learned it was focused, and a key drained in the
    /// same pass — before the app layer's corrective Focus — was discarded by
    /// the app's own `if focused` guard. Focus carries no geometry either.
    #[test]
    fn a_focus_change_reaches_the_app_before_the_key_typed_under_it() {
        let keyboard = Event::Keyboard {
            source: 1,
            modifiers: 0x02,
            keys: [0x1e, 0, 0, 0, 0, 0],
            locks: kore_abi::keyboard::Locks::NONE,
        };
        assert_eq!(
            scoped_event(2, 3, Event::Focus(true)),
            Some(Event::Focus(true))
        );
        assert_eq!(
            scoped_event(2, 1, Event::Focus(false)),
            Some(Event::Focus(false))
        );
        assert_eq!(scoped_event(2, 3, keyboard), Some(keyboard));
    }

    #[test]
    fn input_never_uses_an_old_coordinate_configuration_and_reset_survives() {
        let pointer = Event::Pointer {
            x: 23,
            y: 12,
            buttons: 1,
            scroll: crate::event::Scroll::NONE,
        };
        let keyboard = Event::Keyboard {
            source: 2,
            modifiers: 0,
            keys: [4, 0, 0, 0, 0, 0],
            locks: kore_abi::keyboard::Locks::CAPS,
        };
        assert_eq!(scoped_event(2, 1, pointer), None);
        assert_eq!(scoped_event(2, 3, pointer), Some(Event::Reset));
        assert_eq!(scoped_event(2, 2, pointer), Some(pointer));
        // Keyboard snapshots are not coordinate input: every serial delivers
        // the key itself (see `a_keystroke_crosses_a_configure_boundary_in_either_direction`).
        assert_eq!(scoped_event(2, 2, keyboard), Some(keyboard));
        assert_eq!(scoped_event(2, 1, Event::Reset), Some(Event::Reset));
        assert_eq!(
            scoped_event(2, 1, Event::CloseRequested),
            Some(Event::CloseRequested)
        );
        assert_eq!(scoped_event(2, 3, Event::Configure), Some(Event::Configure));
    }

    #[test]
    fn only_bounded_transient_refusals_are_retryable() {
        assert!(ClientError::Reconfigure.is_retryable());
        assert!(ClientError::Busy.is_retryable());
        assert!(ClientError::Refused(ControlRefusal::Stale).is_retryable());
        assert!(ClientError::SurfaceRefused(SurfaceRefusal::Busy).is_retryable());
        assert!(
            ClientError::SurfaceRefused(SurfaceRefusal::ResourceLimit).is_configuration_retryable()
        );
        assert!(
            ClientError::SurfaceRefused(SurfaceRefusal::OutOfMemory).is_configuration_retryable()
        );
        assert!(!ClientError::SurfaceRefused(SurfaceRefusal::ResourceLimit).is_retryable());
        assert!(!ClientError::SurfaceRefused(SurfaceRefusal::OutOfMemory).is_retryable());
        assert!(!ClientError::Transport.is_retryable());
        assert!(!ClientError::Geometry.is_retryable());
        assert!(!ClientError::CommitRejected.is_retryable());
        assert!(!ClientError::Refused(ControlRefusal::AccessDenied).is_retryable());
    }
}
