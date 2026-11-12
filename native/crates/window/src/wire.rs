//! Fail-closed codecs over [`kore_abi::svc`]'s frozen surface forms.
//!
//! Every decode is **byte-exact fail-closed** (the accel `wire.rs`
//! discipline verbatim): after the structural lift, the decoded value is
//! re-encoded and compared to the incoming record — any stray bit (a set
//! reserve, a non-zero unused per-verb word, a live byte in a dead
//! damage-rect slot) refuses.  The peer wrote garbage or a future
//! revision, and a wrong guess would be a protocol break, not a
//! convenience.
//!
//! The commit data plane ([`CommitWire`]) decodes here too: the records
//! ride the pair window's appended queue, not the request rings, but
//! their fail-closed law is the same.

use kore_abi::session::DISPLAY_FORMAT_XRGB8888;
use kore_abi::svc::{CommitWire, MAX_COMMIT_RECTS, SurRectWire, SurRepWire, SurReqWire};

use crate::capture::{CaptureFrame, CaptureMode};
use crate::clipboard::{CHUNK_BYTES, CLIPBOARD_BYTES};
use crate::cursor::Cursor;
use crate::document::DocumentKind;
use crate::menu::MENU_PLANE_BYTES;
use crate::pick::PICK_NAME_BYTES;
use zerocopy::little_endian::{U16, U32, U64};

/// Grant channels a surface client pre-pins — exactly one: it carries
/// [`SurfaceResponse::SurfaceMade`]'s detached window view.
pub const SURFACE_REPLY_GRANTS: usize = 1;

/// One page-rounded pair lease, including commit and terminal byte planes.
/// The ordinary surface vocabulary does not require consuming the byte plane.
pub const SURFACE_PAIR_BYTES: usize = 0x4000;
pub const TERMINAL_OUTPUT_CAP: u32 = 128;

/// Split only the auxiliary tail; callers split the canonical pair first.
pub fn split_surface_aux(aux: &mut [u8]) -> Option<(&mut [u8], &mut [u8])> {
    let commits = kore_ring::spsc::region_bytes::<CommitWire>(crate::COMMIT_CAP);
    let output = kore_ring::spsc::region_bytes::<kore_abi::svc::OutChunk>(TERMINAL_OUTPUT_CAP);
    if aux.len() < commits.checked_add(output)? {
        return None;
    }
    let (commits, tail) = aux.split_at_mut(commits);
    Some((commits, &mut tail[..output]))
}

/// The planes of a window pair's auxiliary tail, in lease order.
pub struct WindowAux<'a> {
    pub commits: &'a mut [u8],
    pub output: &'a mut [u8],
    /// The event channel (`crate::channel`, `res/EVENT.md`): the
    /// compositor's level state and event records, with one conditional
    /// wake.
    pub events: &'a mut [u8],
    /// The window's context menu as it asks for it ([`crate::menu`]): the
    /// app writes, the compositor snapshots on `ShowMenu`.
    pub menu: &'a mut [u8],
}

/// Window clients share the same four capability auxiliaries as surface
/// clients. The event channel and the menu plane occupy the pair's
/// remaining bytes, not a fifth endpoint.
pub fn split_window_aux(aux: &mut [u8]) -> Option<WindowAux<'_>> {
    let commits = kore_ring::spsc::region_bytes::<CommitWire>(crate::COMMIT_CAP);
    let output = kore_ring::spsc::region_bytes::<kore_abi::svc::OutChunk>(TERMINAL_OUTPUT_CAP);
    let channel = crate::channel::region_bytes();
    let total = commits.checked_add(output)?.checked_add(channel)?.checked_add(MENU_PLANE_BYTES)?;
    if aux.len() < total {
        return None;
    }
    let (commits, tail) = aux.split_at_mut(commits);
    let (output, tail) = tail.split_at_mut(output);
    let (events, tail) = tail.split_at_mut(channel);
    Some(WindowAux {
        commits,
        output,
        events,
        menu: &mut tail[..MENU_PLANE_BYTES],
    })
}

/// The backend-free wire marker — what a *client* names to speak the
/// surface class (`SvcClient<…, SurfaceWire>`, the `AccelWire`
/// discipline: a consumer of the wire must not import a compositor
/// backend to place a call).  It can never be a host — no
/// `SystemService` impl — and it declares **zero founder-staged pairs**:
/// surface pairs are the MINTED road (slice 4's `ensure_display_surface`
/// face), each minted window formatted through this impl's
/// [`init_pair_window`] (the standard pair regions + the commit plane,
/// exactly the display row's own founding shape).
///
/// [`init_pair_window`]: kore_rt::svc::SvcProtocol::init_pair_window
pub struct SurfaceWire;

impl kore_rt::svc::SvcProtocol for SurfaceWire {
    type Request = SurfaceRequest;
    type Response = SurfaceResponse;
    type ReqWire = SurReqWire;
    type RepWire = SurRepWire;
    const REPLY_GRANTS: usize = SURFACE_REPLY_GRANTS;
    const SERVE_PAIRS: usize = 0;

    fn retain_grant(index: usize) -> bool {
        // Both the surface view and the presented-channel receiver outlive
        // this call. Reusing one fixed landing would overwrite live custody.
        index == 0
    }

    fn request_grants(request: &Self::ReqWire) -> usize {
        usize::from(decode_req(request).is_some_and(|request| request_carries_grant(&request)))
    }

    fn init_pair_window(
        window: &mut [u8],
        req_cap: u32,
        rep_cap: u32,
    ) -> Result<(), kore_ring::spsc::RingError> {
        kore_rt::svc::init_window::<Self>(window, req_cap, rep_cap)?;
        let at = kore_rt::svc::window_bytes::<Self>(req_cap, rep_cap);
        let Some(plane) = window.get_mut(at..) else {
            return Err(kore_ring::spsc::RingError::TooSmall);
        };
        let aux = split_window_aux(plane).ok_or(kore_ring::spsc::RingError::TooSmall)?;
        kore_ring::spsc::init::<CommitWire>(aux.commits, crate::COMMIT_CAP)?;
        kore_ring::spsc::init::<kore_abi::svc::OutChunk>(aux.output, TERMINAL_OUTPUT_CAP)?;
        aux.menu.fill(0);
        crate::channel::init(aux.events)
    }

    fn encode_req(req: &Self::Request) -> Self::ReqWire {
        encode_req(req)
    }
    fn decode_req(w: &Self::ReqWire) -> Option<Self::Request> {
        decode_req(w)
    }
    fn encode_rep(rep: &Self::Response, grants: u8) -> Self::RepWire {
        encode_rep(rep, grants)
    }
    fn decode_rep(w: &Self::RepWire) -> Option<(Self::Response, u8)> {
        decode_rep(w)
    }

    fn reply_matches(request: &Self::ReqWire, response: &Self::Response, grants: u8) -> bool {
        reply_matches(request, response, grants)
    }
}

/// The surface-class request vocabulary (`res/DEVICE_SERVICE.md` §4/§7's
/// "multiplexing (compositor)" class, made real).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceRequest {
    /// The compositor's mode, limits, and pool headroom.
    Query,
    /// Carve a surface from the pool; the reply's one grant is the
    /// backing's detached window view.  `reach`/`modifier` are the #579
    /// horizon reservations — v1 accepts only
    /// [`SurReqWire::REACH_CPU`] + [`SurReqWire::MODIFIER_LINEAR`] and
    /// REFUSES the rest (never ignores).
    CreateSurface {
        width: u16,
        height: u16,
        format: u8,
        reach: u16,
        modifier: u64,
    },
    /// The lessor-side revoke: the backing returns to the pool and the
    /// client's mapping dies with it.
    DestroySurface { surface: u64 },
    /// The re-readable presentation truth behind the done-notice —
    /// diagnostics and tests; the per-frame truth is the event channel's
    /// state (`crate::channel`, `res/EVENT.md`).
    Status { surface: u64 },
    /// This window's own close — ESC, Q, an app's close action — asked
    /// through the controller's path rather than taken.  The compositor
    /// raises the same `close_requested` the chrome's close dot raises and
    /// echoes [`Event::CloseRequested`](crate::event::Event::CloseRequested)
    /// down this window's own channel; the controller's next lifecycle
    /// observation then drives the one teardown, and the app exits only on
    /// the echo.  An app that exits on its own instead leaves the
    /// controller, the compositor and the app tearing one window down in
    /// three orders at once.
    RequestClose,
    /// The cursor this window shows while the pointer is over its content
    /// ([`crate::cursor`]). It holds until the window asks again; the
    /// compositor's own pixels choose for themselves.
    SetCursor { cursor: Cursor },
    /// Ask this window's controller to open the filesystem object with
    /// stable identity `object` (never zero) in the default handler for
    /// its `kind` ([`DocumentKind::Unknown`]: the document app). Inert: the
    /// controller decides, and the identity is data the opened window
    /// locates from its own root.
    OpenObject { object: u64, kind: DocumentKind },
    /// Draw the context menu this window encoded in its menu plane
    /// (`len` bytes, [`crate::menu`]) at the content-relative anchor
    /// `(x, y)`. `serial` (nonzero) names it in the
    /// [`Event::Menu`](crate::event::Event::Menu) that ends it.
    ShowMenu { x: i32, y: i32, len: u16, serial: u32 },
    /// A capture writer's one verb (`res/SCREENSHOTS.md`): lend the frozen
    /// frame this pair was admitted for. The reply's one grant reads it
    /// and cannot write it. Any other peer is refused.
    TakeCapture,
    /// Ask the user, through this window's controller, to choose a file of
    /// the types `types` accepts: a nonzero mask over `kore_kit::kinds`
    /// rows. `serial` (nonzero) names the pick in the
    /// [`Event::Picked`](crate::event::Event::Picked) that ends it
    /// ([`crate::pick`]). Inert: the controller launches the picker.
    PickObject { types: u64, serial: u32 },
    /// Bind this window, launched for a pick, to that pick by the token it
    /// was launched with.
    ClaimPick { token: u64 },
    /// Answer the pick this window claimed: `object` is the chosen file's
    /// stable identity, or zero if the user cancelled; its name is the
    /// first `len` bytes of this window's menu plane (none when cancelled).
    Picked { token: u64, object: u64, len: u8 },
    /// Put text on this window's seat's clipboard ([`crate::clipboard`]), a
    /// piece at a time: bytes `offset..offset + len` of a `total`-byte text
    /// are at the head of this window's request plane. Offset zero begins the
    /// copy (focused window, unspent gesture); the piece that reaches `total`
    /// ends it, and the text then replaces the clipboard's.
    CopyChunk { offset: u32, len: u16, total: u32 },
    /// Read the seat's clipboard from `offset` into this window's request
    /// plane. Offset zero begins the read (focused window, unspent gesture);
    /// later offsets continue it until it reaches the end.
    Paste { offset: u32 },
    /// The seat's terminal window asks for the clipboard to be typed into its
    /// shell's line: queued as an inert
    /// [`ShellAction::Paste`](crate::control::ShellAction::Paste) for its
    /// controller (focused window, unspent gesture).
    PasteToShell,
}

/// The surface-class reply vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceResponse {
    /// The present mode + limits: geometry/format of the target, the
    /// surface-table bound, pool free bytes, and the carve granule.
    Info {
        width: u16,
        height: u16,
        format: u8,
        max_surfaces: u16,
        free_bytes: u64,
        granule: u64,
    },
    /// The fresh surface: its id, its 64-aligned row stride, and the
    /// backing's byte length (the granted window's exact span).
    SurfaceMade {
        surface: u64,
        stride: u64,
        bytes: u64,
    },
    /// The echoed subject is gone; its granules are free.
    Destroyed { surface: u64 },
    /// The books: committed/presented epochs, refused commit records,
    /// and the last present's timestamp (the done-notice payload's
    /// re-readable twin).
    Status {
        surface: u64,
        committed: u64,
        presented: u64,
        refused_commits: u64,
        presented_at: u64,
    },
    /// The close request is held: `serial` is the configure serial it was
    /// raised under (never zero).  Idempotent — a window already asked to
    /// close answers this again rather than refusing.
    CloseRequested { serial: u64 },
    /// The window's cursor is now `cursor`, echoed.
    CursorSet { cursor: Cursor },
    /// The open is queued for this window's controller, echoed.
    OpenQueued { object: u64 },
    /// The menu is open; its plane may be reused. The serial, echoed.
    MenuShown { serial: u32 },
    /// The frozen frame; its read-only view rides the reply's one grant.
    Capture(CaptureFrame),
    /// The pick is queued for this window's controller. The serial, echoed.
    PickQueued { serial: u32 },
    /// This window is the pick's picker now. The token, echoed.
    PickClaimed { token: u64 },
    /// The answer is delivered and the pick has ended; the plane may be
    /// reused. The token, echoed.
    PickAnswered { token: u64 },
    /// The copy has taken the text through `through`; when that is the whole
    /// text, the text is the seat's clipboard.
    Copied { through: u32 },
    /// Bytes `offset..offset + len` of the clipboard's `total`-byte text are
    /// at the head of the request plane.
    Clipboard { offset: u32, len: u16, total: u32 },
    /// The seat's clipboard holds no text.
    ClipboardEmpty,
    /// The shell paste is queued for this window's controller; the text is
    /// `total` bytes.
    PasteQueued { total: u32 },
    /// The refusal, echoing the subject the verb named (zero if none).
    Refused {
        refusal: SurfaceRefusal,
        subject: u64,
    },
}

/// The refusal vocabulary — one variant per [`SurRepWire`] `REFUSE_*`
/// code (an unknown wire code refuses at decode; a reachable state has a
/// name or the decode is lying).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceRefusal {
    /// A v1 vocabulary gate: unknown op, non-XRGB8888 format, non-CPU
    /// reach, non-LINEAR modifier.
    Unsupported,
    /// A malformed argument (zero geometry, a stride/backing overflow).
    InvalidArgument,
    /// The surface table is full (the bounded-slot refusal).
    ResourceLimit,
    /// The pool cannot carve the backing.
    OutOfMemory,
    /// The subject names no live surface.
    Unknown,
    /// Authority delivery or revocation has not settled; retry is safe.
    Busy,
    /// The user did not ask: a clipboard verb from a window that is not its
    /// seat's focused window, or with no unspent gesture behind it.
    Denied,
}

impl SurfaceRefusal {
    const fn code(self) -> u8 {
        match self {
            Self::Unsupported => SurRepWire::REFUSE_UNSUPPORTED,
            Self::InvalidArgument => SurRepWire::REFUSE_INVALID_ARGUMENT,
            Self::ResourceLimit => SurRepWire::REFUSE_RESOURCE_LIMIT,
            Self::OutOfMemory => SurRepWire::REFUSE_OUT_OF_MEMORY,
            Self::Unknown => SurRepWire::REFUSE_UNKNOWN,
            Self::Busy => SurRepWire::REFUSE_BUSY,
            Self::Denied => SurRepWire::REFUSE_DENIED,
        }
    }

    const fn from_code(code: u8) -> Option<Self> {
        Some(match code {
            SurRepWire::REFUSE_UNSUPPORTED => Self::Unsupported,
            SurRepWire::REFUSE_INVALID_ARGUMENT => Self::InvalidArgument,
            SurRepWire::REFUSE_RESOURCE_LIMIT => Self::ResourceLimit,
            SurRepWire::REFUSE_OUT_OF_MEMORY => Self::OutOfMemory,
            SurRepWire::REFUSE_UNKNOWN => Self::Unknown,
            SurRepWire::REFUSE_BUSY => Self::Busy,
            SurRepWire::REFUSE_DENIED => Self::Denied,
            _ => return None,
        })
    }
}

/// One decoded commit record — the data-plane crossing shape the engine
/// consumes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Commit {
    /// The surface whose frame this publishes.
    pub surface: u64,
    /// The client's strictly-increasing frame counter.
    pub epoch: u64,
    /// What changed.
    pub damage: Damage,
}

/// Window-layer extension of the frozen 64-byte surface commit. The first
/// reserved word carries a nonzero configure serial; the remaining headroom
/// remains zero. Legacy surface decoding still rejects this extension.
pub fn encode_window_commit(commit: &Commit, serial: u64) -> Option<CommitWire> {
    if serial == 0 {
        return None;
    }
    if let Damage::Rects { count, .. } = commit.damage {
        if count == 0 || usize::from(count) > MAX_COMMIT_RECTS {
            return None;
        }
    }
    let mut wire = encode_commit(commit);
    if decode_commit(&wire)? != *commit {
        return None;
    }
    wire.reserved2[..8].copy_from_slice(&serial.to_le_bytes());
    Some(wire)
}

pub fn decode_window_commit(wire: &CommitWire) -> Option<(Commit, u64)> {
    let serial = u64::from_le_bytes(wire.reserved2[..8].try_into().ok()?);
    if serial == 0 {
        return None;
    }
    let mut surface = *wire;
    surface.reserved2[..8].fill(0);
    Some((decode_commit(&surface)?, serial))
}

/// A commit's damage claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Damage {
    /// The whole surface (a `rect_count = 0` record — also the correct
    /// over-approximation for a frame with more than
    /// [`MAX_COMMIT_RECTS`] dirty regions).
    Full,
    /// `1..=MAX_COMMIT_RECTS` live rectangles, surface-relative pixels.
    Rects {
        rects: [SurRect; MAX_COMMIT_RECTS],
        count: u8,
    },
}

/// One surface-relative damage rectangle (host form of [`SurRectWire`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SurRect {
    pub x: u16,
    pub y: u16,
    pub w: u16,
    pub h: u16,
}

/// The geometry word both directions share: `width ‖ height << 16 ‖
/// format << 32` (`CreateSurface.a`; `Info.a` adds `max << 40`).
const fn geometry_word(width: u16, height: u16, format: u8) -> u64 {
    width as u64 | (height as u64) << 16 | (format as u64) << 32
}

/// Lower a request to its wire form (total).
pub fn encode_req(req: &SurfaceRequest) -> SurReqWire {
    let mut w = SurReqWire {
        op: 0,
        reserved: [0; 3],
        flags: U32::ZERO,
        subject: U64::ZERO,
        a: U64::ZERO,
        b: U64::ZERO,
        c: U64::ZERO,
        d: U64::ZERO,
    };
    match *req {
        SurfaceRequest::Query => {
            w.op = SurReqWire::OP_QUERY;
        }
        SurfaceRequest::CreateSurface {
            width,
            height,
            format,
            reach,
            modifier,
        } => {
            w.op = SurReqWire::OP_CREATE_SURFACE;
            w.a = U64::new(geometry_word(width, height, format));
            w.b = U64::new(u64::from(reach));
            w.c = U64::new(modifier);
        }
        SurfaceRequest::DestroySurface { surface } => {
            w.op = SurReqWire::OP_DESTROY_SURFACE;
            w.subject = U64::new(surface);
        }
        SurfaceRequest::Status { surface } => {
            w.op = SurReqWire::OP_STATUS;
            w.subject = U64::new(surface);
        }
        SurfaceRequest::RequestClose => {
            w.op = SurReqWire::OP_REQUEST_CLOSE;
        }
        SurfaceRequest::SetCursor { cursor } => {
            w.op = SurReqWire::OP_SET_CURSOR;
            w.a = U64::new(cursor.code());
        }
        SurfaceRequest::OpenObject { object, kind } => {
            w.op = SurReqWire::OP_OPEN_OBJECT;
            w.a = U64::new(object);
            w.b = U64::new(u64::from(kind.code()));
        }
        SurfaceRequest::ShowMenu { x, y, len, serial } => {
            w.op = SurReqWire::OP_SHOW_MENU;
            w.a = U64::new(u64::from(x as u32) | u64::from(y as u32) << 32);
            w.b = U64::new(u64::from(len) | u64::from(serial) << 32);
        }
        SurfaceRequest::TakeCapture => {
            w.op = SurReqWire::OP_TAKE_CAPTURE;
        }
        SurfaceRequest::PickObject { types, serial } => {
            w.op = SurReqWire::OP_PICK_OBJECT;
            w.a = U64::new(types);
            w.b = U64::new(u64::from(serial));
        }
        SurfaceRequest::ClaimPick { token } => {
            w.op = SurReqWire::OP_CLAIM_PICK;
            w.a = U64::new(token);
        }
        SurfaceRequest::Picked { token, object, len } => {
            w.op = SurReqWire::OP_PICKED;
            w.a = U64::new(token);
            w.b = U64::new(object);
            w.c = U64::new(u64::from(len));
        }
        SurfaceRequest::CopyChunk { offset, len, total } => {
            w.op = SurReqWire::OP_COPY_CHUNK;
            w.a = U64::new(piece_word(offset, len));
            w.b = U64::new(u64::from(total));
        }
        SurfaceRequest::Paste { offset } => {
            w.op = SurReqWire::OP_PASTE;
            w.a = U64::new(u64::from(offset));
        }
        SurfaceRequest::PasteToShell => {
            w.op = SurReqWire::OP_PASTE_TO_SHELL;
        }
    }
    w
}

/// The piece word a copy and a paste share: `offset | len << 32`.
const fn piece_word(offset: u32, len: u16) -> u64 {
    offset as u64 | (len as u64) << 32
}

/// A clipboard text's length: 1 to [`CLIPBOARD_BYTES`].
fn text_length(word: u64) -> Option<u32> {
    u32::try_from(word)
        .ok()
        .filter(|&len| len != 0 && len as usize <= CLIPBOARD_BYTES)
}

/// A piece of a clipboard text, lifted: `offset..offset + len` of `total`
/// bytes, `len` 1 to [`CHUNK_BYTES`] and `total` 1 to [`CLIPBOARD_BYTES`].
fn piece(word: u64, total: u64) -> Option<(u32, u16, u32)> {
    let (offset, len) = (word as u32, (word >> 32) as u16);
    let total = u32::try_from(total).ok()?;
    let fits = u64::from(offset) + u64::from(len) <= u64::from(total);
    (word >> 48 == 0
        && len != 0
        && usize::from(len) <= CHUNK_BYTES
        && total != 0
        && total as usize <= CLIPBOARD_BYTES
        && fits)
        .then_some((offset, len, total))
}

/// Structural lift; stray bits are the re-encode comparison's to catch.
fn raw_decode_req(w: &SurReqWire) -> Option<SurfaceRequest> {
    Some(match w.op {
        SurReqWire::OP_QUERY => SurfaceRequest::Query,
        SurReqWire::OP_CREATE_SURFACE => {
            let geom = w.a.get();
            SurfaceRequest::CreateSurface {
                width: (geom & 0xFFFF) as u16,
                height: (geom >> 16 & 0xFFFF) as u16,
                format: (geom >> 32 & 0xFF) as u8,
                reach: u16::try_from(w.b.get()).ok()?,
                modifier: w.c.get(),
            }
        }
        SurReqWire::OP_DESTROY_SURFACE => SurfaceRequest::DestroySurface {
            surface: w.subject.get(),
        },
        SurReqWire::OP_STATUS => SurfaceRequest::Status {
            surface: w.subject.get(),
        },
        SurReqWire::OP_REQUEST_CLOSE => SurfaceRequest::RequestClose,
        SurReqWire::OP_SET_CURSOR => SurfaceRequest::SetCursor {
            cursor: Cursor::from_code(w.a.get())?,
        },
        SurReqWire::OP_OPEN_OBJECT => {
            let object = w.a.get();
            if object == 0 {
                return None;
            }
            SurfaceRequest::OpenObject {
                object,
                kind: DocumentKind::from_code(w.b.get())?,
            }
        }
        SurReqWire::OP_SHOW_MENU => {
            let (a, b) = (w.a.get(), w.b.get());
            let (len, serial) = (b as u16, (b >> 32) as u32);
            if len == 0 || usize::from(len) > MENU_PLANE_BYTES || serial == 0 {
                return None;
            }
            SurfaceRequest::ShowMenu {
                x: a as u32 as i32,
                y: (a >> 32) as u32 as i32,
                len,
                serial,
            }
        }
        SurReqWire::OP_TAKE_CAPTURE => SurfaceRequest::TakeCapture,
        SurReqWire::OP_PICK_OBJECT => {
            let (types, serial) = (w.a.get(), u32::try_from(w.b.get()).ok()?);
            if types == 0 || serial == 0 {
                return None;
            }
            SurfaceRequest::PickObject { types, serial }
        }
        SurReqWire::OP_CLAIM_PICK => {
            let token = w.a.get();
            if token == 0 {
                return None;
            }
            SurfaceRequest::ClaimPick { token }
        }
        SurReqWire::OP_PICKED => {
            let (token, object) = (w.a.get(), w.b.get());
            let len = u8::try_from(w.c.get()).ok()?;
            // A choice carries its name and a cancel none.
            if token == 0 || usize::from(len) > PICK_NAME_BYTES || (object == 0) != (len == 0) {
                return None;
            }
            SurfaceRequest::Picked { token, object, len }
        }
        SurReqWire::OP_COPY_CHUNK => {
            let (offset, len, total) = piece(w.a.get(), w.b.get())?;
            SurfaceRequest::CopyChunk { offset, len, total }
        }
        SurReqWire::OP_PASTE => {
            let offset = u32::try_from(w.a.get()).ok()?;
            if offset as usize >= CLIPBOARD_BYTES {
                return None;
            }
            SurfaceRequest::Paste { offset }
        }
        SurReqWire::OP_PASTE_TO_SHELL => SurfaceRequest::PasteToShell,
        _ => return None,
    })
}

/// Lift a request off the wire — byte-exact fail-closed.
pub fn decode_req(w: &SurReqWire) -> Option<SurfaceRequest> {
    let req = raw_decode_req(w)?;
    (encode_req(&req) == *w).then_some(req)
}

/// The grant-count law: which tags carry authority beside their bytes.
pub const fn reply_grants(rep: &SurfaceResponse) -> u8 {
    match rep {
        // The backing's detached window view; the frozen frame's
        // read-only view.
        SurfaceResponse::SurfaceMade { .. } | SurfaceResponse::Capture(_) => 1,
        _ => 0,
    }
}

/// The verbs whose answer carries a grant: what a client reserves a landing
/// for before it asks.
pub const fn request_carries_grant(request: &SurfaceRequest) -> bool {
    matches!(
        request,
        SurfaceRequest::CreateSurface { .. } | SurfaceRequest::TakeCapture
    )
}

/// The capture word `Capture.a` carries: `width ‖ height << 16 ‖ mode << 32
/// ‖ screen << 40 ‖ density << 48`.
const fn capture_word(frame: &CaptureFrame) -> u64 {
    frame.width as u64
        | (frame.height as u64) << 16
        | (frame.mode.code() as u64) << 32
        | (frame.screen as u64) << 40
        | (frame.density.quarters() as u64) << 48
}

/// `Capture.d`'s bit 32: the window gave no frame in time, and its backing
/// was copied as it stood ([`CaptureFrame::synced`] is `false`).
const CAPTURE_UNSYNCED: u64 = 1 << 32;

/// The placement word `Capture.d` carries: `x ‖ y << 16`, and
/// [`CAPTURE_UNSYNCED`].
const fn placement_word(frame: &CaptureFrame) -> u64 {
    frame.x as u64 | (frame.y as u64) << 16 | if frame.synced { 0 } else { CAPTURE_UNSYNCED }
}

/// Lower a response to its wire form (total).  `grants` is the count
/// actually riding the channel — the law says what it must be
/// ([`reply_grants`]), the serving layer says what it was.
pub fn encode_rep(rep: &SurfaceResponse, grants: u8) -> SurRepWire {
    let mut w = SurRepWire {
        tag: 0,
        grants,
        refusal: 0,
        reserved: [0; 5],
        id: U64::ZERO,
        a: U64::ZERO,
        b: U64::ZERO,
        c: U64::ZERO,
        d: U64::ZERO,
    };
    match *rep {
        SurfaceResponse::Info {
            width,
            height,
            format,
            max_surfaces,
            free_bytes,
            granule,
        } => {
            w.tag = SurRepWire::TAG_INFO;
            w.a = U64::new(geometry_word(width, height, format) | (max_surfaces as u64) << 40);
            w.b = U64::new(free_bytes);
            w.c = U64::new(granule);
        }
        SurfaceResponse::SurfaceMade {
            surface,
            stride,
            bytes,
        } => {
            w.tag = SurRepWire::TAG_SURFACE_MADE;
            w.id = U64::new(surface);
            w.a = U64::new(stride);
            w.b = U64::new(bytes);
        }
        SurfaceResponse::Destroyed { surface } => {
            w.tag = SurRepWire::TAG_DESTROYED;
            w.id = U64::new(surface);
        }
        SurfaceResponse::Status {
            surface,
            committed,
            presented,
            refused_commits,
            presented_at,
        } => {
            w.tag = SurRepWire::TAG_STATUS;
            w.id = U64::new(surface);
            w.a = U64::new(committed);
            w.b = U64::new(presented);
            w.c = U64::new(refused_commits);
            w.d = U64::new(presented_at);
        }
        SurfaceResponse::CloseRequested { serial } => {
            w.tag = SurRepWire::TAG_CLOSE_REQUESTED;
            w.a = U64::new(serial);
        }
        SurfaceResponse::CursorSet { cursor } => {
            w.tag = SurRepWire::TAG_CURSOR_SET;
            w.a = U64::new(cursor.code());
        }
        SurfaceResponse::OpenQueued { object } => {
            w.tag = SurRepWire::TAG_OPEN_QUEUED;
            w.a = U64::new(object);
        }
        SurfaceResponse::MenuShown { serial } => {
            w.tag = SurRepWire::TAG_MENU_SHOWN;
            w.a = U64::new(u64::from(serial));
        }
        SurfaceResponse::Capture(frame) => {
            w.tag = SurRepWire::TAG_CAPTURE;
            w.id = U64::new(frame.freeze_ns);
            w.a = U64::new(capture_word(&frame));
            w.b = U64::new(frame.wall_ns.unwrap_or(0));
            w.c = U64::new(frame.uptime_ns);
            w.d = U64::new(placement_word(&frame));
        }
        SurfaceResponse::PickQueued { serial } => {
            w.tag = SurRepWire::TAG_PICK_QUEUED;
            w.a = U64::new(u64::from(serial));
        }
        SurfaceResponse::PickClaimed { token } => {
            w.tag = SurRepWire::TAG_PICK_CLAIMED;
            w.a = U64::new(token);
        }
        SurfaceResponse::PickAnswered { token } => {
            w.tag = SurRepWire::TAG_PICK_ANSWERED;
            w.a = U64::new(token);
        }
        SurfaceResponse::Copied { through } => {
            w.tag = SurRepWire::TAG_COPIED;
            w.a = U64::new(u64::from(through));
        }
        SurfaceResponse::Clipboard { offset, len, total } => {
            w.tag = SurRepWire::TAG_CLIPBOARD;
            w.a = U64::new(piece_word(offset, len));
            w.b = U64::new(u64::from(total));
        }
        SurfaceResponse::ClipboardEmpty => {
            w.tag = SurRepWire::TAG_CLIPBOARD_EMPTY;
        }
        SurfaceResponse::PasteQueued { total } => {
            w.tag = SurRepWire::TAG_PASTE_QUEUED;
            w.a = U64::new(u64::from(total));
        }
        SurfaceResponse::Refused { refusal, subject } => {
            w.tag = SurRepWire::TAG_REFUSED;
            w.refusal = refusal.code();
            w.id = U64::new(subject);
        }
    }
    w
}

/// Structural lift; stray bits are the re-encode comparison's to catch.
fn raw_decode_rep(w: &SurRepWire) -> Option<SurfaceResponse> {
    Some(match w.tag {
        SurRepWire::TAG_INFO => {
            let a = w.a.get();
            SurfaceResponse::Info {
                width: (a & 0xFFFF) as u16,
                height: (a >> 16 & 0xFFFF) as u16,
                format: (a >> 32 & 0xFF) as u8,
                max_surfaces: (a >> 40 & 0xFFFF) as u16,
                free_bytes: w.b.get(),
                granule: w.c.get(),
            }
        }
        SurRepWire::TAG_SURFACE_MADE => SurfaceResponse::SurfaceMade {
            surface: w.id.get(),
            stride: w.a.get(),
            bytes: w.b.get(),
        },
        SurRepWire::TAG_DESTROYED => SurfaceResponse::Destroyed {
            surface: w.id.get(),
        },
        SurRepWire::TAG_STATUS => SurfaceResponse::Status {
            surface: w.id.get(),
            committed: w.a.get(),
            presented: w.b.get(),
            refused_commits: w.c.get(),
            presented_at: w.d.get(),
        },
        SurRepWire::TAG_CLOSE_REQUESTED => SurfaceResponse::CloseRequested { serial: w.a.get() },
        SurRepWire::TAG_CURSOR_SET => SurfaceResponse::CursorSet {
            cursor: Cursor::from_code(w.a.get())?,
        },
        SurRepWire::TAG_OPEN_QUEUED => SurfaceResponse::OpenQueued { object: w.a.get() },
        SurRepWire::TAG_MENU_SHOWN => SurfaceResponse::MenuShown {
            serial: u32::try_from(w.a.get()).ok()?,
        },
        SurRepWire::TAG_CAPTURE => {
            let (a, d) = (w.a.get(), w.d.get());
            if a >> 56 != 0 || d >> 33 != 0 {
                return None;
            }
            let mode = CaptureMode::from_code((a >> 32) as u8)?;
            let synced = d & CAPTURE_UNSYNCED == 0;
            // Only a window can be copied unsynchronized: a screen or
            // region is the compositor's own complete frame.
            if !synced && mode != CaptureMode::Window {
                return None;
            }
            SurfaceResponse::Capture(CaptureFrame {
                width: a as u16,
                height: (a >> 16) as u16,
                mode,
                screen: (a >> 40) as u8,
                density: crate::appearance::Density::from_quarters((a >> 48) as u8)?,
                wall_ns: (w.b.get() != 0).then_some(w.b.get()),
                uptime_ns: w.c.get(),
                freeze_ns: w.id.get(),
                x: d as u16,
                y: (d >> 16) as u16,
                synced,
            })
        }
        SurRepWire::TAG_PICK_QUEUED => SurfaceResponse::PickQueued {
            serial: u32::try_from(w.a.get()).ok()?,
        },
        SurRepWire::TAG_PICK_CLAIMED => SurfaceResponse::PickClaimed { token: w.a.get() },
        SurRepWire::TAG_PICK_ANSWERED => SurfaceResponse::PickAnswered { token: w.a.get() },
        SurRepWire::TAG_COPIED => SurfaceResponse::Copied {
            through: text_length(w.a.get())?,
        },
        SurRepWire::TAG_CLIPBOARD => {
            let (offset, len, total) = piece(w.a.get(), w.b.get())?;
            SurfaceResponse::Clipboard { offset, len, total }
        }
        SurRepWire::TAG_CLIPBOARD_EMPTY => SurfaceResponse::ClipboardEmpty,
        SurRepWire::TAG_PASTE_QUEUED => SurfaceResponse::PasteQueued {
            total: text_length(w.a.get())?,
        },
        SurRepWire::TAG_REFUSED => SurfaceResponse::Refused {
            refusal: SurfaceRefusal::from_code(w.refusal)?,
            subject: w.id.get(),
        },
        _ => return None,
    })
}

/// Lift a response off the wire — byte-exact fail-closed; returns the
/// response and the announced grant count.
pub fn decode_rep(w: &SurRepWire) -> Option<(SurfaceResponse, u8)> {
    let rep = raw_decode_rep(w)?;
    (w.grants == reply_grants(&rep) && encode_rep(&rep, w.grants) == *w).then_some((rep, w.grants))
}

/// Correlate an untagged pair's reply before the client accepts its grant.
/// Wire shape alone cannot distinguish a stale status for another surface,
/// a response to another verb, or a successful create without its backing.
pub fn reply_matches(request: &SurReqWire, response: &SurfaceResponse, grants: u8) -> bool {
    let Some(request) = decode_req(request) else {
        return false;
    };
    if grants != reply_grants(response) {
        return false;
    }
    if let SurfaceResponse::Refused { subject, .. } = *response {
        let expected = match request {
            SurfaceRequest::DestroySurface { surface } | SurfaceRequest::Status { surface } => {
                surface
            }
            _ => 0,
        };
        return subject == expected;
    }
    match (request, *response) {
        (
            SurfaceRequest::Query,
            SurfaceResponse::Info {
                width,
                height,
                format,
                max_surfaces,
                granule,
                free_bytes,
            },
        ) => {
            width != 0
                && height != 0
                && format == SURFACE_FORMAT_XRGB8888
                && max_surfaces != 0
                && granule >= 4096
                && granule.is_power_of_two()
                && free_bytes % granule == 0
        }
        (
            SurfaceRequest::CreateSurface {
                width,
                height,
                format,
                reach,
                modifier,
            },
            SurfaceResponse::SurfaceMade {
                surface,
                stride,
                bytes,
            },
        ) => {
            width != 0
                && height != 0
                && format == SURFACE_FORMAT_XRGB8888
                && reach == SurReqWire::REACH_CPU
                && modifier == SurReqWire::MODIFIER_LINEAR
                && surface != 0
                && u32::try_from(surface).is_ok()
                && stride >= u64::from(width) * 4
                && stride % 64 == 0
                && bytes % 4096 == 0
                && stride
                    .checked_mul(u64::from(height))
                    .is_some_and(|needed| bytes >= needed)
        }
        (
            SurfaceRequest::DestroySurface { surface },
            SurfaceResponse::Destroyed { surface: got },
        ) => surface == got,
        (
            SurfaceRequest::Status { surface },
            SurfaceResponse::Status {
                surface: got,
                committed,
                presented,
                ..
            },
        ) => surface == got && presented <= committed,
        // A close raised under no configure serial names a window that was
        // never presented; the compositor refuses that instead.
        (SurfaceRequest::RequestClose, SurfaceResponse::CloseRequested { serial }) => serial != 0,
        // The echo names the cursor asked for, never another.
        (SurfaceRequest::SetCursor { cursor }, SurfaceResponse::CursorSet { cursor: got }) => {
            cursor == got
        }
        (SurfaceRequest::OpenObject { object, .. }, SurfaceResponse::OpenQueued { object: got }) => {
            object == got
        }
        (SurfaceRequest::ShowMenu { serial, .. }, SurfaceResponse::MenuShown { serial: got }) => {
            serial == got
        }
        (SurfaceRequest::TakeCapture, SurfaceResponse::Capture(frame)) => frame.is_valid(),
        (SurfaceRequest::PickObject { serial, .. }, SurfaceResponse::PickQueued { serial: got }) => {
            serial == got
        }
        (SurfaceRequest::ClaimPick { token }, SurfaceResponse::PickClaimed { token: got })
        | (SurfaceRequest::Picked { token, .. }, SurfaceResponse::PickAnswered { token: got }) => {
            token == got
        }
        // The copy echoes how far it has taken the text: exactly to the end
        // of the piece just sent.
        (SurfaceRequest::CopyChunk { offset, len, .. }, SurfaceResponse::Copied { through }) => {
            u64::from(through) == u64::from(offset) + u64::from(len)
        }
        // A read answers from the offset asked, and an empty clipboard
        // answers only the read's first piece: a read under way that finds
        // the clipboard changed is refused instead.
        (SurfaceRequest::Paste { offset }, SurfaceResponse::Clipboard { offset: got, .. }) => {
            offset == got
        }
        (SurfaceRequest::Paste { offset }, SurfaceResponse::ClipboardEmpty) => offset == 0,
        (SurfaceRequest::PasteToShell, SurfaceResponse::PasteQueued { .. }) => true,
        (SurfaceRequest::PasteToShell, SurfaceResponse::ClipboardEmpty) => true,
        _ => false,
    }
}

/// Lower a commit to its data-plane record (total).
pub fn encode_commit(c: &Commit) -> CommitWire {
    let mut w = CommitWire {
        surface: U32::new(u32::try_from(c.surface).unwrap_or(u32::MAX)),
        rect_count: 0,
        reserved: [0; 3],
        epoch: U64::new(c.epoch),
        rects: [SurRectWire {
            x: U16::ZERO,
            y: U16::ZERO,
            w: U16::ZERO,
            h: U16::ZERO,
        }; MAX_COMMIT_RECTS],
        reserved2: [0; 16],
    };
    if let Damage::Rects { rects, count } = c.damage {
        w.rect_count = count;
        for (dst, src) in w.rects.iter_mut().zip(rects.iter()).take(count as usize) {
            dst.x = U16::new(src.x);
            dst.y = U16::new(src.y);
            dst.w = U16::new(src.w);
            dst.h = U16::new(src.h);
        }
    }
    w
}

/// Structural lift; stray bits are the re-encode comparison's to catch.
fn raw_decode_commit(w: &CommitWire) -> Option<Commit> {
    let damage = match w.rect_count {
        0 => Damage::Full,
        count @ 1..=4 => {
            let mut rects = [SurRect::default(); MAX_COMMIT_RECTS];
            for (dst, src) in rects.iter_mut().zip(w.rects.iter()).take(count as usize) {
                *dst = SurRect {
                    x: src.x.get(),
                    y: src.y.get(),
                    w: src.w.get(),
                    h: src.h.get(),
                };
            }
            Damage::Rects { rects, count }
        }
        _ => return None,
    };
    Some(Commit {
        surface: u64::from(w.surface.get()),
        epoch: w.epoch.get(),
        damage,
    })
}

/// Lift a commit record off the queue — byte-exact fail-closed (a live
/// byte in a dead rect slot, a set reserve, an over-limit count all
/// refuse; the compositor books the drop, never guesses).
pub fn decode_commit(w: &CommitWire) -> Option<Commit> {
    let c = raw_decode_commit(w)?;
    (encode_commit(&c) == *w).then_some(c)
}

/// The one v1 surface pixel format — re-exported so engine and codec
/// name the same registry byte (`kore_abi::session`'s, never a twin).
pub const SURFACE_FORMAT_XRGB8888: u8 = DISPLAY_FORMAT_XRGB8888;

#[cfg(test)]
mod tests {
    use crate::capture::{CaptureFrame, CaptureMode};

    fn frame() -> CaptureFrame {
        CaptureFrame {
            width: 1280,
            height: 800,
            mode: CaptureMode::Screen,
            screen: 0,
            density: crate::appearance::Density::OneHalf,
            wall_ns: Some(1_800_000_000_000_000_000),
            uptime_ns: 42_000_000_000,
            freeze_ns: 1_234_567,
            x: 0,
            y: 0,
            synced: true,
        }
    }

    /// A window's placement and its sync travel with the lend; only a window
    /// may arrive unsynchronized, and stray placement bits refuse it.
    #[test]
    fn a_window_capture_carries_its_placement_and_sync() {
        let window = CaptureFrame {
            mode: CaptureMode::Window,
            width: 640,
            height: 400,
            x: 312,
            y: 0xfffe,
            synced: false,
            ..frame()
        };
        let region = CaptureFrame {
            mode: CaptureMode::Region,
            x: 7,
            y: 9,
            ..frame()
        };
        for frame in [window, CaptureFrame { synced: true, ..window }, region] {
            let response = SurfaceResponse::Capture(frame);
            assert_eq!(decode_rep(&encode_rep(&response, 1)), Some((response, 1)));
        }
        let unsynced_screen = SurfaceResponse::Capture(CaptureFrame {
            synced: false,
            ..frame()
        });
        assert_eq!(
            decode_rep(&encode_rep(&unsynced_screen, 1)),
            None,
            "a screen is always the compositor's own complete frame"
        );
        let mut stray = encode_rep(&SurfaceResponse::Capture(window), 1);
        stray.d = (stray.d.get() | 1 << 40).into();
        assert_eq!(decode_rep(&stray), None);
    }

    /// A capture lend is one grant, byte-exact, and pairs only with the
    /// verb that asked for it.
    #[test]
    fn a_capture_lend_round_trips_with_its_one_grant() {
        let request = encode_req(&SurfaceRequest::TakeCapture);
        assert_eq!(decode_req(&request), Some(SurfaceRequest::TakeCapture));
        assert!(request_carries_grant(&SurfaceRequest::TakeCapture));
        assert_eq!(
            <SurfaceWire as kore_rt::svc::SvcProtocol>::request_grants(&request),
            1
        );
        for frame in [frame(), CaptureFrame { wall_ns: None, ..frame() }] {
            let response = SurfaceResponse::Capture(frame);
            assert_eq!(reply_grants(&response), 1);
            let wire = encode_rep(&response, 1);
            assert_eq!(decode_rep(&wire), Some((response, 1)));
            assert!(reply_matches(&request, &response, 1));
            assert!(!reply_matches(&request, &response, 0), "the lend must ride it");
            assert_eq!(decode_rep(&encode_rep(&response, 0)), None);
            let status = encode_req(&SurfaceRequest::Status { surface: 1 });
            assert!(!reply_matches(&status, &response, 1));
            // The picker and screenshot protocols share this wire, but a
            // pick must never consume the capture writer's read-only grant.
            let pick = encode_req(&SurfaceRequest::PickObject { types: 1, serial: 1 });
            assert_ne!(pick.op, request.op);
            assert!(!reply_matches(&pick, &response, 1));
            let queued = SurfaceResponse::PickQueued { serial: 1 };
            assert_ne!(encode_rep(&queued, 0).tag, wire.tag);
            assert!(!reply_matches(&request, &queued, 0));
        }
        let empty = SurfaceResponse::Capture(CaptureFrame {
            height: 0,
            ..frame()
        });
        assert!(!reply_matches(&request, &empty, 1), "a lend with no pixels");
        let mut stray = encode_rep(&SurfaceResponse::Capture(frame()), 1);
        stray.a = (stray.a.get() | 1 << 56).into();
        assert_eq!(decode_rep(&stray), None);
        let mut density = encode_rep(&SurfaceResponse::Capture(frame()), 1);
        density.a = (density.a.get() & !(0xff << 48) | 1 << 48).into();
        assert_eq!(decode_rep(&density), None, "a density no screen draws at");
    }

    use super::*;
    use kore_rt::svc::SvcProtocol;
    use zerocopy::{FromBytes, IntoBytes};

    #[test]
    fn replies_match_the_outstanding_verb_subject_and_grant() {
        assert!(SurfaceWire::retain_grant(0));
        assert!(!SurfaceWire::retain_grant(1));
        let requests = [
            SurfaceRequest::Query,
            SurfaceRequest::CreateSurface {
                width: 320,
                height: 200,
                format: SURFACE_FORMAT_XRGB8888,
                reach: SurReqWire::REACH_CPU,
                modifier: SurReqWire::MODIFIER_LINEAR,
            },
            SurfaceRequest::DestroySurface { surface: 17 },
            SurfaceRequest::Status { surface: 17 },
            SurfaceRequest::RequestClose,
        ];
        let replies = [
            SurfaceResponse::Info {
                width: 640,
                height: 480,
                format: SURFACE_FORMAT_XRGB8888,
                max_surfaces: 8,
                free_bytes: 2 << 20,
                granule: 2 << 20,
            },
            SurfaceResponse::SurfaceMade {
                surface: 17,
                stride: 1280,
                bytes: 2 << 20,
            },
            SurfaceResponse::Destroyed { surface: 17 },
            SurfaceResponse::Status {
                surface: 17,
                committed: 3,
                presented: 2,
                refused_commits: 0,
                presented_at: 123,
            },
            SurfaceResponse::CloseRequested { serial: 7 },
        ];
        for (i, request) in requests.iter().enumerate() {
            let wire = encode_req(request);
            assert_eq!(SurfaceWire::request_grants(&wire), usize::from(i == 1));
            for (j, response) in replies.iter().enumerate() {
                assert_eq!(
                    SurfaceWire::reply_matches(&wire, response, reply_grants(response)),
                    i == j
                );
                assert!(!SurfaceWire::reply_matches(
                    &wire,
                    response,
                    reply_grants(response) + 1
                ));
                assert!(decode_rep(&encode_rep(response, reply_grants(response) + 1)).is_none());
            }
            let subject = if i == 2 || i == 3 { 17 } else { 0 };
            assert!(reply_matches(
                &wire,
                &SurfaceResponse::Refused {
                    refusal: SurfaceRefusal::Unknown,
                    subject,
                },
                0
            ));
            assert!(!reply_matches(
                &wire,
                &SurfaceResponse::Refused {
                    refusal: SurfaceRefusal::Unknown,
                    subject: subject + 1,
                },
                0
            ));
        }
        assert!(!reply_matches(
            &encode_req(&requests[3]),
            &SurfaceResponse::Status {
                surface: 18,
                committed: 3,
                presented: 2,
                refused_commits: 0,
                presented_at: 123,
            },
            0
        ));
        assert!(!reply_matches(
            &encode_req(&requests[3]),
            &SurfaceResponse::Status {
                surface: 17,
                committed: 3,
                presented: 4,
                refused_commits: 0,
                presented_at: 123,
            },
            0
        ));
    }

    #[test]
    fn creates_cannot_claim_short_overflowed_or_unrepresentable_backings() {
        let request = encode_req(&SurfaceRequest::CreateSurface {
            width: 320,
            height: 200,
            format: SURFACE_FORMAT_XRGB8888,
            reach: SurReqWire::REACH_CPU,
            modifier: SurReqWire::MODIFIER_LINEAR,
        });
        for (surface, stride, bytes) in [
            (0, 1280, 2 << 20),
            (u64::from(u32::MAX) + 1, 1280, 2 << 20),
            (17, 64, 2 << 20),
            (17, 1281, 2 << 20),
            (17, 1280, 4096),
            (17, 1280, (2 << 20) + 1),
            (17, u64::MAX - 63, u64::MAX - 4095),
        ] {
            assert!(!reply_matches(
                &request,
                &SurfaceResponse::SurfaceMade {
                    surface,
                    stride,
                    bytes,
                },
                1
            ));
        }
    }

    fn every_request() -> [SurfaceRequest; 19] {
        [
            SurfaceRequest::Query,
            SurfaceRequest::CreateSurface {
                width: 800,
                height: 600,
                format: SURFACE_FORMAT_XRGB8888,
                reach: SurReqWire::REACH_CPU,
                modifier: SurReqWire::MODIFIER_LINEAR,
            },
            SurfaceRequest::CreateSurface {
                width: 1,
                height: 1,
                format: 7,
                reach: 3,
                modifier: 0xFFEE_DDCC_BBAA_9988,
            },
            SurfaceRequest::DestroySurface { surface: 0x11 },
            SurfaceRequest::Status { surface: 0x2002 },
            SurfaceRequest::RequestClose,
            SurfaceRequest::SetCursor {
                cursor: Cursor::Default,
            },
            SurfaceRequest::SetCursor {
                cursor: Cursor::Progress,
            },
            SurfaceRequest::OpenObject {
                object: 0x5eed,
                kind: DocumentKind::Unknown,
            },
            SurfaceRequest::OpenObject {
                object: u64::MAX,
                kind: DocumentKind::Image,
            },
            SurfaceRequest::ShowMenu {
                x: -12,
                y: 4000,
                len: MENU_PLANE_BYTES as u16,
                serial: u32::MAX,
            },
            SurfaceRequest::PickObject {
                types: u64::MAX,
                serial: u32::MAX,
            },
            SurfaceRequest::ClaimPick { token: u64::MAX },
            SurfaceRequest::Picked {
                token: 0x7071_ce4e,
                object: u64::MAX,
                len: PICK_NAME_BYTES as u8,
            },
            SurfaceRequest::CopyChunk {
                offset: 0,
                len: 1,
                total: 1,
            },
            SurfaceRequest::CopyChunk {
                offset: (CLIPBOARD_BYTES - CHUNK_BYTES) as u32,
                len: CHUNK_BYTES as u16,
                total: CLIPBOARD_BYTES as u32,
            },
            SurfaceRequest::Paste { offset: 0 },
            SurfaceRequest::Paste {
                offset: CLIPBOARD_BYTES as u32 - 1,
            },
            SurfaceRequest::PasteToShell,
        ]
    }

    fn every_response() -> [SurfaceResponse; 19] {
        [
            SurfaceResponse::Info {
                width: 1920,
                height: 1200,
                format: SURFACE_FORMAT_XRGB8888,
                max_surfaces: 8,
                free_bytes: 64 << 20,
                granule: 2 << 20,
            },
            SurfaceResponse::SurfaceMade {
                surface: 0x11,
                stride: 3200,
                bytes: 2 << 21,
            },
            SurfaceResponse::Destroyed { surface: 0x11 },
            SurfaceResponse::Status {
                surface: 0x11,
                committed: 9,
                presented: 8,
                refused_commits: 2,
                presented_at: 0xABCD_EF01,
            },
            SurfaceResponse::Refused {
                refusal: SurfaceRefusal::OutOfMemory,
                subject: 0x11,
            },
            SurfaceResponse::Refused {
                refusal: SurfaceRefusal::Busy,
                subject: 0,
            },
            SurfaceResponse::CloseRequested { serial: 0x2A },
            SurfaceResponse::CursorSet {
                cursor: Cursor::Text,
            },
            SurfaceResponse::OpenQueued { object: 0x5eed },
            SurfaceResponse::MenuShown { serial: 7 },
            SurfaceResponse::PickQueued { serial: 3 },
            SurfaceResponse::PickClaimed { token: 0x7071_ce4e },
            SurfaceResponse::PickAnswered { token: 0x7071_ce4e },
            SurfaceResponse::Copied { through: 1 },
            SurfaceResponse::Copied {
                through: CLIPBOARD_BYTES as u32,
            },
            SurfaceResponse::Clipboard {
                offset: 4096,
                len: 17,
                total: 4113,
            },
            SurfaceResponse::ClipboardEmpty,
            SurfaceResponse::PasteQueued { total: 9 },
            SurfaceResponse::Refused {
                refusal: SurfaceRefusal::Denied,
                subject: 0,
            },
        ]
    }

    /// A clipboard piece names a span inside its text, the text inside the
    /// clipboard's bound and the piece inside the plane; nothing crosses
    /// with a grant; and each reply answers only the verb and offset asked.
    #[test]
    fn clipboard_pieces_stay_inside_their_text_and_answer_their_own_offset() {
        let copy = SurfaceRequest::CopyChunk {
            offset: 4096,
            len: 100,
            total: 4196,
        };
        let wire = encode_req(&copy);
        assert_eq!(SurfaceWire::request_grants(&wire), 0);
        assert_eq!(decode_req(&wire), Some(copy));
        assert!(reply_matches(&wire, &SurfaceResponse::Copied { through: 4196 }, 0));
        assert!(!reply_matches(&wire, &SurfaceResponse::Copied { through: 4096 }, 0));
        assert!(!reply_matches(&wire, &SurfaceResponse::Copied { through: 4196 }, 1));
        for (offset, len, total) in [
            (0, 0, 1),
            (0, 1, 0),
            (1, 1, 1),
            (0, CHUNK_BYTES as u16 + 1, CLIPBOARD_BYTES as u32),
            (0, 1, CLIPBOARD_BYTES as u32 + 1),
            (u32::MAX, 1, CLIPBOARD_BYTES as u32),
        ] {
            let wire = encode_req(&SurfaceRequest::CopyChunk { offset, len, total });
            assert_eq!(decode_req(&wire), None, "{offset} {len} {total}");
        }
        let mut stray = encode_req(&copy);
        stray.a = (stray.a.get() | 1 << 50).into();
        assert_eq!(decode_req(&stray), None);

        let read = encode_req(&SurfaceRequest::Paste { offset: 0 });
        let piece = SurfaceResponse::Clipboard {
            offset: 0,
            len: 12,
            total: 12,
        };
        assert!(reply_matches(&read, &piece, 0));
        assert!(reply_matches(&read, &SurfaceResponse::ClipboardEmpty, 0));
        let later = encode_req(&SurfaceRequest::Paste { offset: 12 });
        assert!(!reply_matches(&later, &piece, 0));
        assert!(
            !reply_matches(&later, &SurfaceResponse::ClipboardEmpty, 0),
            "a read under way never learns the clipboard emptied: it is refused"
        );
        let past = encode_req(&SurfaceRequest::Paste {
            offset: CLIPBOARD_BYTES as u32,
        });
        assert_eq!(decode_req(&past), None);
        for piece in [
            (0, 13, 12),
            (0, 0, 12),
            (0, CHUNK_BYTES as u16 + 1, CLIPBOARD_BYTES as u32),
        ] {
            let mut w = encode_rep(&SurfaceResponse::ClipboardEmpty, 0);
            w.tag = SurRepWire::TAG_CLIPBOARD;
            w.a = piece_word(piece.0, piece.1).into();
            w.b = u64::from(piece.2).into();
            assert_eq!(decode_rep(&w), None, "{piece:?}");
        }

        let shell = encode_req(&SurfaceRequest::PasteToShell);
        assert!(reply_matches(&shell, &SurfaceResponse::PasteQueued { total: 5 }, 0));
        assert!(reply_matches(&shell, &SurfaceResponse::ClipboardEmpty, 0));
        assert!(!reply_matches(&shell, &SurfaceResponse::Copied { through: 5 }, 0));
        let denied = SurfaceResponse::Refused {
            refusal: SurfaceRefusal::Denied,
            subject: 0,
        };
        for request in [copy, SurfaceRequest::Paste { offset: 0 }, SurfaceRequest::PasteToShell] {
            assert!(reply_matches(&encode_req(&request), &denied, 0));
        }
        for zero in [
            SurfaceResponse::Copied { through: 0 },
            SurfaceResponse::PasteQueued { total: 0 },
        ] {
            assert_eq!(decode_rep(&encode_rep(&zero, 0)), None);
        }
    }

    /// A pick is answered only by its own serial's echo, a claim and an
    /// answer only by their token's, none carries a grant, and a pick of no
    /// types, a zero serial or token, or an answer whose name contradicts
    /// its object refuses.
    #[test]
    fn picks_are_answered_only_by_their_own_serial_or_token() {
        let pick = SurfaceRequest::PickObject { types: 0b1100, serial: 4 };
        let wire = encode_req(&pick);
        assert_eq!(SurfaceWire::request_grants(&wire), 0);
        assert_eq!(decode_req(&wire), Some(pick));
        assert!(reply_matches(&wire, &SurfaceResponse::PickQueued { serial: 4 }, 0));
        assert!(!reply_matches(&wire, &SurfaceResponse::PickQueued { serial: 5 }, 0));
        assert!(!reply_matches(&wire, &SurfaceResponse::MenuShown { serial: 4 }, 0));
        let claim = encode_req(&SurfaceRequest::ClaimPick { token: 9 });
        assert!(reply_matches(&claim, &SurfaceResponse::PickClaimed { token: 9 }, 0));
        assert!(!reply_matches(&claim, &SurfaceResponse::PickAnswered { token: 9 }, 0));
        let answer = encode_req(&SurfaceRequest::Picked { token: 9, object: 0x77, len: 9 });
        assert!(reply_matches(&answer, &SurfaceResponse::PickAnswered { token: 9 }, 0));
        assert!(!reply_matches(&answer, &SurfaceResponse::PickAnswered { token: 8 }, 0));
        let cancel = SurfaceRequest::Picked { token: 9, object: 0, len: 0 };
        assert_eq!(decode_req(&encode_req(&cancel)), Some(cancel));
        for bad in [
            SurfaceRequest::PickObject { types: 0, serial: 4 },
            SurfaceRequest::PickObject { types: 1, serial: 0 },
            SurfaceRequest::ClaimPick { token: 0 },
            SurfaceRequest::Picked { token: 0, object: 1, len: 1 },
            SurfaceRequest::Picked { token: 9, object: 1, len: 0 },
            SurfaceRequest::Picked { token: 9, object: 0, len: 1 },
            SurfaceRequest::Picked { token: 9, object: 1, len: PICK_NAME_BYTES as u8 + 1 },
        ] {
            assert_eq!(decode_req(&encode_req(&bad)), None, "{bad:?}");
        }
        // The serial's word holds a `u32`; its high bits are not spare.
        let mut stray = wire;
        stray.b = U64::new(stray.b.get() | 1 << 40);
        assert_eq!(decode_req(&stray), None);
    }

    /// A menu request is answered only by its own serial's echo, carries
    /// no grant, and an empty or oversized plane or a zero serial refuses.
    #[test]
    fn a_menu_request_is_answered_only_by_its_own_serial() {
        let show = SurfaceRequest::ShowMenu {
            x: 40,
            y: -3,
            len: 96,
            serial: 9,
        };
        let wire = encode_req(&show);
        assert_eq!(SurfaceWire::request_grants(&wire), 0);
        assert_eq!(decode_req(&wire), Some(show));
        assert!(reply_matches(&wire, &SurfaceResponse::MenuShown { serial: 9 }, 0));
        assert!(!reply_matches(&wire, &SurfaceResponse::MenuShown { serial: 8 }, 0));
        assert!(!reply_matches(&wire, &SurfaceResponse::OpenQueued { object: 9 }, 0));
        for (len, serial) in [(0, 9), (MENU_PLANE_BYTES as u16 + 1, 9), (96, 0)] {
            let bad = encode_req(&SurfaceRequest::ShowMenu {
                x: 0,
                y: 0,
                len,
                serial,
            });
            assert_eq!(decode_req(&bad), None, "len {len} serial {serial}");
        }
        // The length's spare high bits are not the serial's to borrow.
        let mut stray = wire;
        stray.b = U64::new(stray.b.get() | 1 << 20);
        assert_eq!(decode_req(&stray), None);
        // An open names a kind a class has, or refuses.
        let mut open = encode_req(&SurfaceRequest::OpenObject {
            object: 1,
            kind: DocumentKind::Text,
        });
        open.b = U64::new(DocumentKind::ALL.len() as u64);
        assert_eq!(decode_req(&open), None);
    }

    /// A cursor request is answered only by the echo of the cursor it
    /// named, carries no grant, and a code no cursor has refuses at decode.
    #[test]
    fn a_cursor_request_is_answered_only_by_its_own_echo() {
        for cursor in Cursor::ALL {
            let wire = encode_req(&SurfaceRequest::SetCursor { cursor });
            assert_eq!(SurfaceWire::request_grants(&wire), 0);
            assert_eq!(decode_req(&wire), Some(SurfaceRequest::SetCursor { cursor }));
            for echoed in Cursor::ALL {
                let held = SurfaceResponse::CursorSet { cursor: echoed };
                assert_eq!(reply_matches(&wire, &held, 0), echoed == cursor);
            }
        }
        let mut stray = encode_req(&SurfaceRequest::SetCursor {
            cursor: Cursor::Text,
        });
        stray.a = U64::new(Cursor::ALL.len() as u64);
        assert_eq!(decode_req(&stray), None);
        let mut reply = encode_rep(
            &SurfaceResponse::CursorSet {
                cursor: Cursor::Text,
            },
            0,
        );
        reply.a = U64::new(u64::MAX);
        assert_eq!(decode_rep(&reply), None);
    }

    /// A close request answers only under a live configure serial, and a
    /// refusal of it names no subject.
    #[test]
    fn a_close_request_is_held_only_under_a_live_serial() {
        let wire = encode_req(&SurfaceRequest::RequestClose);
        assert_eq!(SurfaceWire::request_grants(&wire), 0);
        assert!(reply_matches(
            &wire,
            &SurfaceResponse::CloseRequested { serial: 1 },
            0
        ));
        assert!(!reply_matches(
            &wire,
            &SurfaceResponse::CloseRequested { serial: 0 },
            0
        ));
        assert!(!reply_matches(
            &wire,
            &SurfaceResponse::CloseRequested { serial: 1 },
            1
        ));
        assert!(reply_matches(
            &wire,
            &SurfaceResponse::Refused {
                refusal: SurfaceRefusal::Busy,
                subject: 0,
            },
            0
        ));
        assert!(!reply_matches(
            &wire,
            &SurfaceResponse::Refused {
                refusal: SurfaceRefusal::Busy,
                subject: 1,
            },
            0
        ));
        assert!(!reply_matches(
            &wire,
            &SurfaceResponse::Destroyed { surface: 0 },
            0
        ));
    }

    /// The codec obligation: every request round-trips byte-exactly.
    #[test]
    fn every_request_round_trips() {
        for req in every_request() {
            let w = encode_req(&req);
            assert_eq!(decode_req(&w), Some(req), "{req:?}");
        }
    }

    /// The codec obligation: every response round-trips with its grant
    /// count, and the grant law holds per tag.
    #[test]
    fn every_response_round_trips_with_the_grant_law() {
        for rep in every_response() {
            let grants = reply_grants(&rep);
            let w = encode_rep(&rep, grants);
            assert_eq!(decode_rep(&w), Some((rep, grants)), "{rep:?}");
            // The law: authority rides exactly the made tag.
            let expect = matches!(rep, SurfaceResponse::SurfaceMade { .. });
            assert_eq!(grants == 1, expect, "{rep:?}");
        }
    }

    /// Fail-closed: a stray bit anywhere in a request refuses — swept
    /// exhaustively by flipping every byte of every encoded form.
    #[test]
    fn any_stray_request_byte_refuses_or_reencodes_exactly() {
        for req in every_request() {
            let w = encode_req(&req);
            for i in 0..core::mem::size_of::<SurReqWire>() {
                let mut bytes = [0u8; core::mem::size_of::<SurReqWire>()];
                bytes.copy_from_slice(w.as_bytes());
                bytes[i] ^= 0x80;
                let mutated = SurReqWire::read_from_bytes(&bytes[..]).unwrap();
                if let Some(lifted) = decode_req(&mutated) {
                    // A byte that survives decode must be a *meaningful*
                    // field — the lift must re-encode to the mutated
                    // record, never to something third.
                    assert_eq!(encode_req(&lifted), mutated, "{req:?} byte {i}");
                }
            }
        }
    }

    /// The zero record (unseeded ring slot) is invalid in both
    /// directions and for commits.
    #[test]
    fn the_zero_record_refuses_everywhere() {
        let req = SurReqWire::read_from_bytes(&[0u8; 48][..]).unwrap();
        assert_eq!(decode_req(&req), None);
        let rep = SurRepWire::read_from_bytes(&[0u8; 48][..]).unwrap();
        assert_eq!(decode_rep(&rep), None);
        // A zero commit is surface 0 / epoch 0 / full damage — struct-
        // urally decodable; the ENGINE refuses it (no surface 0 exists).
        let c = CommitWire::read_from_bytes(&[0u8; 64][..]).unwrap();
        assert!(
            decode_commit(&c).is_some(),
            "commit zero decodes; the engine books it"
        );
    }

    /// An unknown refusal code refuses at decode (a reachable state has
    /// a name or the decode is lying).
    #[test]
    fn unknown_refusal_codes_refuse() {
        let mut w = encode_rep(
            &SurfaceResponse::Refused {
                refusal: SurfaceRefusal::Unknown,
                subject: 3,
            },
            0,
        );
        w.refusal = 99;
        assert_eq!(decode_rep(&w), None);
    }

    /// Commit records round-trip: full damage, each live rect count, and
    /// the fail-closed edges (over-limit count, live bytes in dead
    /// slots, set reserves).
    #[test]
    fn commit_records_round_trip_and_fail_closed() {
        let rect = |x, y, w, h| SurRect { x, y, w, h };
        // Dead slots stay zero on the host side too: the wire law
        // (dead slot bytes must be zero) makes the host struct with
        // `count` live entries its exact mirror.
        let mut one = [SurRect::default(); MAX_COMMIT_RECTS];
        one[0] = rect(1, 2, 30, 40);
        let mut two = one;
        two[1] = rect(0, 0, 1920, 1);
        let rects = two;
        for damage in [
            Damage::Full,
            Damage::Rects {
                rects: one,
                count: 1,
            },
            Damage::Rects {
                rects: two,
                count: 2,
            },
        ] {
            let c = Commit {
                surface: 0x21,
                epoch: 7,
                damage,
            };
            let w = encode_commit(&c);
            assert_eq!(decode_commit(&w), Some(c), "{damage:?}");
        }

        // Over-limit count refuses.
        let mut w = encode_commit(&Commit {
            surface: 0x21,
            epoch: 7,
            damage: Damage::Full,
        });
        w.rect_count = 5;
        assert_eq!(decode_commit(&w), None);

        // A live byte in a dead rect slot refuses.
        let mut w = encode_commit(&Commit {
            surface: 0x21,
            epoch: 7,
            damage: Damage::Rects { rects, count: 1 },
        });
        w.rects[3].w = U16::new(9);
        assert_eq!(decode_commit(&w), None);

        // A set reserve refuses.
        let mut w = encode_commit(&Commit {
            surface: 0x21,
            epoch: 7,
            damage: Damage::Full,
        });
        w.reserved2[15] = 1;
        assert_eq!(decode_commit(&w), None);
    }
}
