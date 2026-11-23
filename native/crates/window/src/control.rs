//! One display face for founder-owned output/control and admitted surfaces.
//!
//! Surface records keep their canonical ABI, so a `SurfaceWire` client does
//! not depend on the terminal facade. The extra namespaces are disjoint;
//! arrival identity, not an opcode or a terminal number supplied by a client,
//! decides which vocabulary the service permits.

use crate::capture::{CaptureMode, CaptureRefusal};
use crate::configure::{Configure, Content, Preference};
use crate::document::{DocumentKind, KindMask};
use crate::wire::{self as surface, SurfaceRequest, SurfaceResponse};
use kore_abi::{
    session::SlotRef,
    svc::{OutRepWire, OutReqWire, SurRepWire, SurReqWire},
};
use kore_rt::svc::{PeerBundlePart, PeerId};
use kore_svc_output::service::{OutputRequest, OutputResponse};

use crate::appearance::Density;
use crate::desktop::{DesktopRequest, DesktopState, Mode};
const PREPARE_SETTINGS: u8 = 0x52;
const DESKTOP: u8 = 0x53;

const OUTPUT: u8 = 0x80;
const ADMIT: u8 = 0x40;
const TAKE: u8 = 0x41;
const RETIRE: u8 = 0x42;
const PRESENTATION: u8 = 0x43;
const PREPARE: u8 = 0x44;
const CONFIGURE: u8 = 0x45;
const ACK_CONFIGURE: u8 = 0x46;
const FOCUS: u8 = 0x47;
const CANCEL: u8 = 0x48;
const REQUEST_CLOSE: u8 = 0x49;
const SET_MODE: u8 = 0x4a;
const SET_TABS: u8 = 0x4b;
const PUBLISH_DOCK_ITEM: u8 = 0x4c;
const TAKE_SHELL_ACTION: u8 = 0x4d;
const SET_LAYOUT_HINTS: u8 = 0x4e;
const PREPARE_CAPTURE: u8 = 0x4f;
const REQUEST_CAPTURE: u8 = 0x50;
const RELEASE_CAPTURE: u8 = 0x51;
const READ_CLIPBOARD: u8 = 0x54;
const ACCEPTED: u8 = 0x40;
const ADMITTED: u8 = 0x41;
const REFUSED: u8 = 0x42;
const SHELL_ACTION: u8 = 0x4d;
const CLIPBOARD_TEXT: u8 = 0x54;

/// Bytes of clipboard text one [`DisplayRequest::ReadClipboard`] answer
/// carries inline: the reply record's four data words. A shell paste takes at
/// most one line of the shell's line editor, a few of these.
pub const CLIPBOARD_TEXT_BYTES: usize = 32;

/// Bound on one window's published tab strip.
pub const MAX_TABS: usize = 8;
/// Bound on the compositor-drawn dock catalog.
pub const MAX_DOCK_ITEMS: usize = crate::MAX_DOCK_ITEMS;
/// NUL-padded printable-ASCII bytes carried by one dock label.
pub const DOCK_LABEL_BYTES: usize = 24;

/// The first `count` bytes printable ASCII, every later byte zero.
const fn printable_prefix(bytes: &[u8], count: usize) -> bool {
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        let printable = byte >= 0x20 && byte <= 0x7e;
        if (index < count && !printable) || (index >= count && byte != 0) {
            return false;
        }
        index += 1;
    }
    true
}

/// One window's tab strip as published by its client: the count, the
/// active index, and one printable-ASCII initial per tab. The compositor
/// draws it; it is never authority, and a stale version is refused.
///
/// `version` 0 is the unpublished state; a client's first publication
/// carries 1 and every replacement carries an equal or larger version.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TabModel {
    version: u16,
    count: u8,
    active: u8,
    initials: [u8; MAX_TABS],
}

impl TabModel {
    pub const UNPUBLISHED: Self = Self {
        version: 0,
        count: 0,
        active: 0,
        initials: [0; MAX_TABS],
    };

    /// Structural validation: `1 ≤ count ≤ MAX_TABS`, `active < count`,
    /// the first `count` initials printable ASCII and the rest zero.
    pub const fn new(
        version: u16,
        count: u8,
        active: u8,
        initials: [u8; MAX_TABS],
    ) -> Option<Self> {
        if version == 0
            || count == 0
            || count as usize > MAX_TABS
            || active >= count
            || !printable_prefix(&initials, count as usize)
        {
            return None;
        }
        Some(Self {
            version,
            count,
            active,
            initials,
        })
    }

    /// Build from tab labels; each label's first byte is its initial.
    pub fn from_labels(version: u16, active: u8, labels: &[&str]) -> Option<Self> {
        if labels.is_empty() || labels.len() > MAX_TABS {
            return None;
        }
        let mut initials = [0; MAX_TABS];
        for (slot, label) in initials.iter_mut().zip(labels) {
            *slot = *label.as_bytes().first()?;
        }
        Self::new(version, labels.len() as u8, active, initials)
    }

    pub const fn version(&self) -> u16 {
        self.version
    }

    pub const fn count(&self) -> u8 {
        self.count
    }

    pub const fn active(&self) -> u8 {
        self.active
    }

    pub const fn initials(&self) -> &[u8; MAX_TABS] {
        &self.initials
    }

    pub const fn is_published(&self) -> bool {
        self.version != 0
    }
}

/// Dock row kind: a launchable application window.
pub const DOCK_KIND_APPLICATION: u8 = 0;
/// Dock row kind: the controller's terminal presentation.
pub const DOCK_KIND_TERMINAL: u8 = 1;

/// One dock row as published by the controller: a kind discriminant the
/// shell and compositor agree on out of band, inert state flags, plus a
/// NUL-padded label. The flags are testimony (the controller's view of its
/// own retained owners), never authority: the compositor draws them and
/// routes a click on a running row to WM policy instead of a launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DockItem {
    icon: u64,
    kind: u8,
    flags: u8,
    /// The document classes this row's app opens — its handler contract,
    /// so "Open With" lists it for those kinds. Data, never authority: the
    /// controller still decides what an activation launches.
    opens: KindMask,
    label: [u8; DOCK_LABEL_BYTES],
}

impl DockItem {
    /// The controller currently retains a live owner for this row.
    pub const RUNNING: u8 = 1 << 0;
    /// Every flag bit this wire revision defines; any other bit fails closed.
    pub const FLAGS: u8 = Self::RUNNING;

    /// Structural validation: a non-empty printable-ASCII label, NUL-padded.
    /// The row starts with no flags set.
    pub const fn new(kind: u8, label: [u8; DOCK_LABEL_BYTES]) -> Option<Self> {
        let mut count = 0;
        while count < DOCK_LABEL_BYTES && label[count] != 0 {
            count += 1;
        }
        if count == 0 || !printable_prefix(&label, count) {
            return None;
        }
        Some(Self {
            kind,
            flags: 0,
            icon: 0,
            opens: KindMask::NONE,
            label,
        })
    }

    /// Structural validation of every wire word: `new` plus a flag set that
    /// carries only defined bits.
    pub const fn with_flags(kind: u8, flags: u8, label: [u8; DOCK_LABEL_BYTES]) -> Option<Self> {
        if flags & !Self::FLAGS != 0 {
            return None;
        }
        match Self::new(kind, label) {
            Some(item) => Some(Self { flags, ..item }),
            None => None,
        }
    }

    pub fn from_label(kind: u8, label: &str) -> Option<Self> {
        let bytes = label.as_bytes();
        if bytes.len() > DOCK_LABEL_BYTES {
            return None;
        }
        let mut padded = [0; DOCK_LABEL_BYTES];
        padded[..bytes.len()].copy_from_slice(bytes);
        Self::new(kind, padded)
    }

    /// Build-prepared image identity, never a surface or executable capability.
    /// Zero and unknown identities select the compositor's letter fallback.
    pub const fn with_icon(self, icon: u64) -> Self {
        Self { icon, ..self }
    }

    pub const fn icon(&self) -> u64 {
        self.icon
    }

    /// The same row declaring the document classes its app opens.
    pub const fn with_opens(self, opens: KindMask) -> Self {
        Self { opens, ..self }
    }

    pub const fn opens(&self) -> KindMask {
        self.opens
    }

    pub const fn kind(&self) -> u8 {
        self.kind
    }

    pub const fn flags(&self) -> u8 {
        self.flags
    }

    pub const fn running(&self) -> bool {
        self.flags & Self::RUNNING != 0
    }

    /// The same row with its running bit set or cleared.
    pub const fn with_running(self, running: bool) -> Self {
        let flags = if running {
            self.flags | Self::RUNNING
        } else {
            self.flags & !Self::RUNNING
        };
        Self { flags, ..self }
    }

    pub const fn label(&self) -> &[u8; DOCK_LABEL_BYTES] {
        &self.label
    }

    /// The label without its NUL padding.
    pub fn label_bytes(&self) -> &[u8] {
        let end = self
            .label
            .iter()
            .position(|b| *b == 0)
            .unwrap_or(DOCK_LABEL_BYTES);
        &self.label[..end]
    }
}

/// An inert request the compositor's dock queues for the controller. The
/// controller decides what, if anything, to launch; the compositor never
/// spawns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShellAction {
    /// Dock row `index` of catalog `version` was activated.
    Launch { version: u16, index: u8 },
    /// A window this controller launched asked to open the filesystem
    /// object with stable identity `object` (never zero) in the default
    /// handler for its `kind`. Data only: the opened window locates it from
    /// its own root.
    Open { object: u64, kind: DocumentKind },
    /// A window's "Open With" menu chose dock row `index` of catalog
    /// `version` to open the object with stable identity `object`.
    OpenWith { version: u16, index: u8, object: u64 },
    /// The seat's screen is frozen (`res/SCREENSHOTS.md`): launch the
    /// capture writer on a pair prepared with [`DisplayRequest::
    /// PrepareCapture`] for `ticket`. The geometry is data for the
    /// controller's own books; the writer reads the frame's own.
    /// `window` is the window a [`CaptureMode::Window`] frame was copied
    /// from, so the controller can name its app; `None` for every other
    /// mode. Inert identity: the controller matches it against the peers it
    /// launched.
    Captured {
        ticket: u32,
        mode: CaptureMode,
        width: u16,
        height: u16,
        window: Option<PeerId>,
    },
    /// A capture of `mode` did not happen, and why. The controller says so.
    CaptureRefused {
        mode: CaptureMode,
        refusal: CaptureRefusal,
    },
    /// A window this controller launched asked the user to choose a file of
    /// the types `types` accepts (`res/FILE_PICKER.md`): launch the picker
    /// with `token` (never zero), by which it claims the pick. Data only;
    /// the compositor brokers the answer.
    Pick { token: u64, types: u64 },
    /// The user pasted into this controller's terminal window
    /// (`res/CLIPBOARD.md`): read the clipboard with
    /// [`DisplayRequest::ReadClipboard`] under `ticket` (never zero) and type
    /// what the line can hold. The ticket reads the text the user pasted and
    /// no other: a copy made since refuses it.
    Paste { ticket: u64 },
}

impl ShellAction {
    /// The dock row word `Launch` and `OpenWith` share: `version | index << 16`.
    fn row(word: u64) -> Option<(u16, u8)> {
        let (version, index) = (word as u16, (word >> 16) as u8);
        (word >> 24 == 0 && version != 0 && (index as usize) < MAX_DOCK_ITEMS).then_some((version, index))
    }

    fn from_wire(id: u64, a: u64, b: u64) -> Option<Option<Self>> {
        match id {
            0 if a == 0 && b == 0 => Some(None),
            1 if b == 0 => {
                let (version, index) = Self::row(a)?;
                Some(Some(Self::Launch { version, index }))
            }
            2 if a != 0 => Some(Some(Self::Open {
                object: a,
                kind: DocumentKind::from_code(b)?,
            })),
            3 if a != 0 => {
                let (version, index) = Self::row(b)?;
                Some(Some(Self::OpenWith {
                    version,
                    index,
                    object: a,
                }))
            }
            4 if a as u32 != 0 && a >> 40 == 0 && b >> 48 == 0 => {
                let (width, height) = (b as u16, (b >> 16) as u16);
                (width != 0 && height != 0).then_some(())?;
                let mode = CaptureMode::from_code((a >> 32) as u8)?;
                let window = match (b >> 32) as u16 {
                    0 => None,
                    arrival => Some(PeerId::from_arrival(arrival)?),
                };
                // Only a window capture names a window, and it always does.
                (window.is_some() == (mode == CaptureMode::Window)).then_some(())?;
                Some(Some(Self::Captured {
                    ticket: a as u32,
                    mode,
                    width,
                    height,
                    window,
                }))
            }
            5 if a >> 8 == 0 && b >> 8 == 0 => Some(Some(Self::CaptureRefused {
                mode: CaptureMode::from_code(a as u8)?,
                refusal: CaptureRefusal::from_code(b as u8)?,
            })),
            6 if a != 0 && b != 0 => Some(Some(Self::Pick { token: a, types: b })),
            7 if a != 0 && b == 0 => Some(Some(Self::Paste { ticket: a })),
            _ => None,
        }
    }

    const fn words(action: Option<Self>) -> (u64, u64, u64) {
        match action {
            None => (0, 0, 0),
            Some(Self::Launch { version, index }) => {
                (1, version as u64 | ((index as u64) << 16), 0)
            }
            Some(Self::Open { object, kind }) => (2, object, kind.code() as u64),
            Some(Self::OpenWith {
                version,
                index,
                object,
            }) => (3, object, version as u64 | ((index as u64) << 16)),
            Some(Self::Captured {
                ticket,
                mode,
                width,
                height,
                window,
            }) => (
                4,
                ticket as u64 | (mode.code() as u64) << 32,
                width as u64
                    | (height as u64) << 16
                    | match window {
                        Some(peer) => (peer.arrival() as u64) << 32,
                        None => 0,
                    },
            ),
            Some(Self::CaptureRefused { mode, refusal }) => {
                (5, mode.code() as u64, refusal.code() as u64)
            }
            Some(Self::Pick { token, types }) => (6, token, types),
            Some(Self::Paste { ticket }) => (7, ticket, 0),
        }
    }
}

fn label_words(label: &[u8; DOCK_LABEL_BYTES]) -> [u64; 3] {
    core::array::from_fn(|word| {
        u64::from_le_bytes(label[word * 8..word * 8 + 8].try_into().unwrap())
    })
}

fn label_bytes(words: [u64; 3]) -> [u8; DOCK_LABEL_BYTES] {
    let mut label = [0; DOCK_LABEL_BYTES];
    for (word, chunk) in words.into_iter().zip(label.chunks_exact_mut(8)) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    label
}

/// Client-only protocol marker. Importing it does not pull in the compositor
/// implementation, its backing pool, a shader compiler, or a display driver.
pub struct WindowWire;

/// Compositor-owned visibility/layout state for one admitted application.
///
/// The controller pair remains the authority boundary. Copying this value or
/// sending it beside a stale/wrong-controller peer grants nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum WindowMode {
    #[default]
    Normal = 0,
    Minimized = 1,
    Maximized = 2,
}

impl WindowMode {
    const fn from_wire(raw: u64) -> Option<Self> {
        match raw {
            0 => Some(Self::Normal),
            1 => Some(Self::Minimized),
            2 => Some(Self::Maximized),
            _ => None,
        }
    }
}

impl kore_rt::svc::SvcProtocol for WindowWire {
    type Request = DisplayRequest;
    type Response = DisplayResponse;
    type ReqWire = SurReqWire;
    type RepWire = SurRepWire;
    const REPLY_GRANTS: usize = 1;
    const SERVE_PAIRS: usize = 0;

    fn retain_grant(index: usize) -> bool {
        index == 0
    }
    fn request_grants(request: &SurReqWire) -> usize {
        usize::from(
            matches!(
                decode_req(request),
                Some(DisplayRequest::Desktop(
                    DesktopRequest::CreateWallpaper { .. }
                ))
            ) || matches!(
                decode_req(request),
                Some(DisplayRequest::Surface(request)) if surface::request_carries_grant(&request)
            ),
        )
    }
    fn init_pair_window(
        window: &mut [u8],
        req_cap: u32,
        rep_cap: u32,
    ) -> Result<(), kore_ring::spsc::RingError> {
        <surface::SurfaceWire as kore_rt::svc::SvcProtocol>::init_pair_window(
            window, req_cap, rep_cap,
        )
    }
    fn encode_req(request: &Self::Request) -> SurReqWire {
        encode_req(request)
    }
    fn decode_req(wire: &SurReqWire) -> Option<Self::Request> {
        decode_req(wire)
    }
    fn encode_rep(response: &Self::Response, grants: u8) -> SurRepWire {
        encode_rep(response, grants)
    }
    fn decode_rep(wire: &SurRepWire) -> Option<(Self::Response, u8)> {
        decode_rep(wire)
    }
    fn reply_matches(wire: &SurReqWire, response: &Self::Response, grants: u8) -> bool {
        reply_matches(wire, response, grants)
    }
}

/// The client's published **layout hints** (`res/WM_LAYOUT.md` §2) —
/// what this window may usefully be sized to.  Inert metadata like
/// [`TabModel`]: versioned, validated, applied by the WM's clamp, never
/// authority.  Limits are hard (`max`, the aspect lock,
/// [`NEVER_MAXIMIZE`](Self::NEVER_MAXIMIZE)); `min` is advisory (a
/// policy input, not a clamp — enforcing it would let one window starve
/// its siblings).  `version` 0 is the unpublished state; a replacement
/// carries an equal or larger version, and a stale one is refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayoutHints {
    version: u16,
    flags: u16,
    min: (u16, u16),
    max: (u16, u16),
    aspect: (u16, u16),
}

impl LayoutHints {
    /// `SetMode(Maximized)` is refused and a maximized placement clamps
    /// back through `max`/aspect — the window never covers the domain.
    pub const NEVER_MAXIMIZE: u16 = 1 << 0;
    /// Placements preserve `aspect` exactly: the largest aspect-true
    /// rect inside the cell, centred.
    pub const FIXED_ASPECT: u16 = 1 << 1;
    /// The admission default this client would choose; the user's
    /// explicit tile/float override always wins.
    pub const PREFER_FLOAT: u16 = 1 << 2;

    const FLAG_MASK: u16 = Self::NEVER_MAXIMIZE | Self::FIXED_ASPECT | Self::PREFER_FLOAT;

    pub const UNPUBLISHED: Self = Self {
        version: 0,
        flags: 0,
        min: (0, 0),
        max: (0, 0),
        aspect: (0, 0),
    };

    /// Structural validation: a nonzero version; known flags only; each
    /// `max` axis, where bounded (nonzero), at least its `min`; the
    /// aspect either absent `(0, 0)` or fully present, and present when
    /// [`FIXED_ASPECT`](Self::FIXED_ASPECT) claims it.
    pub const fn new(
        version: u16,
        flags: u16,
        min: (u16, u16),
        max: (u16, u16),
        aspect: (u16, u16),
    ) -> Option<Self> {
        if version == 0
            || flags & !Self::FLAG_MASK != 0
            || (max.0 != 0 && max.0 < min.0)
            || (max.1 != 0 && max.1 < min.1)
            || ((aspect.0 == 0) != (aspect.1 == 0))
            || (flags & Self::FIXED_ASPECT != 0 && aspect.0 == 0)
        {
            return None;
        }
        Some(Self {
            version,
            flags,
            min,
            max,
            aspect,
        })
    }

    pub const fn version(&self) -> u16 {
        self.version
    }

    pub const fn flags(&self) -> u16 {
        self.flags
    }

    /// The advisory lower bound, `(0, 0)` when unstated.
    pub const fn min(&self) -> (u16, u16) {
        self.min
    }

    /// The hard upper bound per axis; `0` = unbounded on that axis.
    pub const fn max(&self) -> (u16, u16) {
        self.max
    }

    /// The locked ratio when [`FIXED_ASPECT`](Self::FIXED_ASPECT) is
    /// set; `(0, 0)` otherwise-or-unstated.
    pub const fn aspect(&self) -> (u16, u16) {
        self.aspect
    }

    pub const fn never_maximize(&self) -> bool {
        self.flags & Self::NEVER_MAXIMIZE != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplayRequest {
    /// Controller-only: authorize this admission for desktop preferences.
    PrepareSettings {
        preference: Preference,
        icon: u64,
    },
    Desktop(DesktopRequest),
    Output(OutputRequest),
    Surface(SurfaceRequest),
    AdmitPeer {
        part: PeerBundlePart,
        announced: SlotRef,
    },
    TakeAdmission,
    RetirePeer {
        peer: PeerId,
    },
    RequestClose {
        peer: PeerId,
    },
    SetMode {
        peer: PeerId,
        mode: WindowMode,
    },
    Presentation {
        peer: PeerId,
    },
    PrepareWindow {
        content: Content,
        preference: Preference,
        /// The app icon the window presents under (a `kore_app_icons` id, the
        /// one its dock row publishes; `0` = none). Inert identity, never
        /// authority: it is how a running dock row finds its own windows.
        icon: u64,
    },
    Configure,
    Acknowledge {
        serial: u64,
        surface: u64,
    },
    Focus {
        peer: PeerId,
    },
    CancelAdmission,
    /// Client verb: replace this window's published tab strip.
    SetTabs(TabModel),
    SetLayoutHints(LayoutHints),
    /// Controller verb: replace one row of the dock catalog. Rows of one
    /// `version` share one `count`; a new version starts an empty catalog.
    PublishDockItem {
        version: u16,
        index: u8,
        count: u8,
        item: DockItem,
    },
    /// Controller verb: dequeue the oldest pending dock action, if any.
    TakeShellAction,
    /// Controller verb: the next peer this controller admits is the
    /// capture writer for `ticket` (from [`ShellAction::Captured`]). It
    /// gets no window; its one verb is
    /// [`SurfaceRequest::TakeCapture`], and its teardown retires the
    /// frozen frame.
    PrepareCapture { ticket: u32 },
    /// Controller verb: capture this controller's seat, as its
    /// PrintScreen chord would (the `screenshot` command). A
    /// [`CaptureMode::Window`] capture takes `window`, a peer this
    /// controller launched, or the seat's focused window when `None`; every
    /// other mode names none.
    RequestCapture {
        mode: CaptureMode,
        window: Option<PeerId>,
    },
    /// Controller verb: give back the frozen frame for `ticket` without
    /// writing it (the writer could not be launched). A frame a writer
    /// was admitted for is retired by that writer's teardown instead.
    ReleaseCapture {
        ticket: u32,
    },
    /// Controller verb: read the clipboard text the user pasted into this
    /// controller's terminal, from `offset`, under the ticket its
    /// [`ShellAction::Paste`] carried.
    ReadClipboard { ticket: u64, offset: u32 },
}

/// Clipboard text read for a controller: bytes `offset..offset + len` of a
/// `total`-byte text, the first `len` of `bytes` (the rest zero).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClipboardText {
    pub offset: u32,
    pub len: u8,
    pub total: u32,
    pub bytes: [u8; CLIPBOARD_TEXT_BYTES],
}

impl ClipboardText {
    /// The text this answer carries.
    pub fn text(&self) -> &[u8] {
        &self.bytes[..usize::from(self.len)]
    }

    /// The answer's word: `offset | total << 24 | len << 48`.
    const fn word(&self) -> u64 {
        self.offset as u64 | (self.total as u64) << 24 | (self.len as u64) << 48
    }

    fn from_words(word: u64, data: [u64; 4]) -> Option<Self> {
        let (offset, total, len) = ((word & 0xff_ffff) as u32, (word >> 24 & 0xff_ffff) as u32, (word >> 48) as u16);
        let mut bytes = [0; CLIPBOARD_TEXT_BYTES];
        for (word, chunk) in data.into_iter().zip(bytes.chunks_exact_mut(8)) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        let len = u8::try_from(len).ok()?;
        (len != 0
            && usize::from(len) <= CLIPBOARD_TEXT_BYTES
            && total != 0
            && total as usize <= crate::clipboard::CLIPBOARD_BYTES
            && u64::from(offset) + u64::from(len) <= u64::from(total)
            && bytes[usize::from(len)..].iter().all(|&byte| byte == 0))
        .then_some(Self {
            offset,
            len,
            total,
            bytes,
        })
    }

    fn data(&self) -> [u64; 4] {
        core::array::from_fn(|word| {
            u64::from_le_bytes(self.bytes[word * 8..word * 8 + 8].try_into().unwrap())
        })
    }
}

const _: () = assert!(
    crate::clipboard::CLIPBOARD_BYTES < 1 << 24 && CLIPBOARD_TEXT_BYTES == 4 * 8
);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplayResponse {
    Desktop(DesktopState),
    Output(OutputResponse),
    Surface(SurfaceResponse),
    Accepted,
    Admitted {
        peer: PeerId,
    },
    Presentation {
        peer: Option<PeerId>,
        surface: u64,
        presented: u64,
        mode: WindowMode,
        close_requested: bool,
    },
    Configured(Configure),
    ControlRefused(ControlRefusal),
    ShellAction(Option<ShellAction>),
    ClipboardText(ClipboardText),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ControlRefusal {
    AccessDenied = 1,
    Busy = 2,
    BadBundle = 3,
    Unavailable = 4,
    Pending = 5,
    Invalid = 6,
    Stale = 7,
}

pub fn encode_req(request: &DisplayRequest) -> SurReqWire {
    let mut w = surface::encode_req(&SurfaceRequest::Query);
    match *request {
        DisplayRequest::PrepareSettings { preference, icon } => {
            w.op = PREPARE_SETTINGS;
            let (kind, value) = preference.words();
            w.a = kind.into();
            w.b = value.into();
            w.c = icon.into();
        }
        DisplayRequest::Desktop(request) => {
            w.op = DESKTOP;
            match request {
                DesktopRequest::Query => {}
                DesktopRequest::RestoreComplete => w.subject = 4.into(),
                DesktopRequest::Appearance { mode, density } => {
                    w.subject = 1.into();
                    w.a = (mode as u64).into();
                    w.b = (density.quarters() as u64).into();
                }
                DesktopRequest::CreateWallpaper { width, height } => {
                    w.subject = 2.into();
                    w.a = u64::from(width).into();
                    w.b = u64::from(height).into();
                }
                DesktopRequest::ApplyWallpaper { surface } => {
                    w.subject = 3.into();
                    w.a = surface.into();
                }
            }
        }
        DisplayRequest::Surface(request) => return surface::encode_req(&request),
        DisplayRequest::Output(request) => {
            let out = kore_svc_output::wire::encode_req(&request);
            w.op = OUTPUT | out.op;
            w.subject = u64::from(out.slot.get()).into();
            w.a = out.through;
        }
        DisplayRequest::AdmitPeer { part, announced } => {
            w.op = ADMIT;
            w.subject = u64::from(announced.index()).into();
            w.a = match part {
                PeerBundlePart::Ring => 0,
                PeerBundlePart::Kick => 1,
                PeerBundlePart::ReturnRecv => 2,
                PeerBundlePart::Grant(index) => 256 + u64::from(index),
            }
            .into();
        }
        DisplayRequest::TakeAdmission => w.op = TAKE,
        DisplayRequest::Presentation { peer } => {
            w.op = PRESENTATION;
            w.subject = u64::from(peer.arrival()).into();
        }
        DisplayRequest::PrepareWindow {
            content,
            preference,
            icon,
        } => {
            w.op = PREPARE;
            w.subject = u64::from(content == Content::Application).into();
            let (kind, value) = preference.words();
            w.a = kind.into();
            w.b = value.into();
            w.c = icon.into();
        }
        DisplayRequest::Configure => w.op = CONFIGURE,
        DisplayRequest::Acknowledge { serial, surface } => {
            w.op = ACK_CONFIGURE;
            w.subject = surface.into();
            w.a = serial.into();
        }
        DisplayRequest::Focus { peer } => {
            w.op = FOCUS;
            w.subject = u64::from(peer.arrival()).into();
        }
        DisplayRequest::CancelAdmission => w.op = CANCEL,
        DisplayRequest::RetirePeer { peer } => {
            w.op = RETIRE;
            w.subject = u64::from(peer.arrival()).into();
        }
        DisplayRequest::RequestClose { peer } => {
            w.op = REQUEST_CLOSE;
            w.subject = u64::from(peer.arrival()).into();
        }
        DisplayRequest::SetMode { peer, mode } => {
            w.op = SET_MODE;
            w.subject = u64::from(peer.arrival()).into();
            w.a = u64::from(mode as u8).into();
        }
        DisplayRequest::SetTabs(tabs) => {
            w.op = SET_TABS;
            w.a = (u64::from(tabs.version)
                | (u64::from(tabs.count) << 16)
                | (u64::from(tabs.active) << 24))
                .into();
            w.b = u64::from_le_bytes(tabs.initials).into();
        }
        DisplayRequest::SetLayoutHints(hints) => {
            w.op = SET_LAYOUT_HINTS;
            w.a = (u64::from(hints.version)
                | (u64::from(hints.flags) << 16)
                | (u64::from(hints.min.0) << 32)
                | (u64::from(hints.min.1) << 48))
                .into();
            w.b = (u64::from(hints.max.0)
                | (u64::from(hints.max.1) << 16)
                | (u64::from(hints.aspect.0) << 32)
                | (u64::from(hints.aspect.1) << 48))
                .into();
        }
        DisplayRequest::PublishDockItem {
            version,
            index,
            count,
            item,
        } => {
            w.op = PUBLISH_DOCK_ITEM;
            w.subject = item.icon.into();
            w.a = (u64::from(version)
                | (u64::from(index) << 16)
                | (u64::from(count) << 24)
                | (u64::from(item.kind) << 32)
                | (u64::from(item.flags) << 40)
                | (u64::from(item.opens.bits()) << 48))
                .into();
            let [b, c, d] = label_words(&item.label);
            w.b = b.into();
            w.c = c.into();
            w.d = d.into();
        }
        DisplayRequest::TakeShellAction => w.op = TAKE_SHELL_ACTION,
        DisplayRequest::PrepareCapture { ticket } => {
            w.op = PREPARE_CAPTURE;
            w.subject = u64::from(ticket).into();
        }
        DisplayRequest::RequestCapture { mode, window } => {
            w.op = REQUEST_CAPTURE;
            w.subject = window.map_or(0, |peer| u64::from(peer.arrival())).into();
            w.a = u64::from(mode.code()).into();
        }
        DisplayRequest::ReleaseCapture { ticket } => {
            w.op = RELEASE_CAPTURE;
            w.subject = u64::from(ticket).into();
        }
        DisplayRequest::ReadClipboard { ticket, offset } => {
            w.op = READ_CLIPBOARD;
            w.subject = ticket.into();
            w.a = u64::from(offset).into();
        }
    }
    w
}

pub fn decode_req(w: &SurReqWire) -> Option<DisplayRequest> {
    let request = match w.op {
        PREPARE_SETTINGS => DisplayRequest::PrepareSettings {
            preference: Preference::from_words(w.a.get(), w.b.get())?,
            icon: w.c.get(),
        },
        DESKTOP => DisplayRequest::Desktop(match w.subject.get() {
            0 => DesktopRequest::Query,
            4 => DesktopRequest::RestoreComplete,
            1 => DesktopRequest::Appearance {
                mode: Mode::from_code(u8::try_from(w.a.get()).ok()?)?,
                density: Density::from_quarters(u8::try_from(w.b.get()).ok()?)?,
            },
            2 if w.a.get() != 0 && w.b.get() != 0 => DesktopRequest::CreateWallpaper {
                width: u16::try_from(w.a.get()).ok()?,
                height: u16::try_from(w.b.get()).ok()?,
            },
            3 if w.a.get() != 0 => DesktopRequest::ApplyWallpaper { surface: w.a.get() },
            _ => return None,
        }),
        ADMIT => DisplayRequest::AdmitPeer {
            part: match w.a.get() {
                0 => PeerBundlePart::Ring,
                1 => PeerBundlePart::Kick,
                2 => PeerBundlePart::ReturnRecv,
                256..=511 => PeerBundlePart::Grant((w.a.get() - 256) as u8),
                _ => return None,
            },
            announced: SlotRef::from_wire(u32::try_from(w.subject.get()).ok()?),
        },
        TAKE => DisplayRequest::TakeAdmission,
        PRESENTATION => DisplayRequest::Presentation {
            peer: PeerId::from_arrival(u16::try_from(w.subject.get()).ok()?)?,
        },
        PREPARE => DisplayRequest::PrepareWindow {
            content: match w.subject.get() {
                0 => Content::Terminal,
                1 => Content::Application,
                _ => return None,
            },
            preference: Preference::from_words(w.a.get(), w.b.get())?,
            icon: w.c.get(),
        },
        CONFIGURE => DisplayRequest::Configure,
        ACK_CONFIGURE if w.a.get() != 0 && w.subject.get() != 0 => DisplayRequest::Acknowledge {
            serial: w.a.get(),
            surface: w.subject.get(),
        },
        FOCUS => DisplayRequest::Focus {
            peer: PeerId::from_arrival(u16::try_from(w.subject.get()).ok()?)?,
        },
        CANCEL => DisplayRequest::CancelAdmission,
        RETIRE => DisplayRequest::RetirePeer {
            peer: PeerId::from_arrival(u16::try_from(w.subject.get()).ok()?)?,
        },
        REQUEST_CLOSE => DisplayRequest::RequestClose {
            peer: PeerId::from_arrival(u16::try_from(w.subject.get()).ok()?)?,
        },
        SET_MODE => DisplayRequest::SetMode {
            peer: PeerId::from_arrival(u16::try_from(w.subject.get()).ok()?)?,
            mode: WindowMode::from_wire(w.a.get())?,
        },
        SET_TABS => {
            let a = w.a.get();
            if a >> 32 != 0 {
                return None;
            }
            DisplayRequest::SetTabs(TabModel::new(
                a as u16,
                (a >> 16) as u8,
                (a >> 24) as u8,
                w.b.get().to_le_bytes(),
            )?)
        }
        SET_LAYOUT_HINTS => {
            let (a, b) = (w.a.get(), w.b.get());
            DisplayRequest::SetLayoutHints(LayoutHints::new(
                a as u16,
                (a >> 16) as u16,
                ((a >> 32) as u16, (a >> 48) as u16),
                (b as u16, (b >> 16) as u16),
                ((b >> 32) as u16, (b >> 48) as u16),
            )?)
        }
        PUBLISH_DOCK_ITEM => {
            let a = w.a.get();
            let (version, index, count) = (a as u16, (a >> 16) as u8, (a >> 24) as u8);
            if version == 0
                || count == 0
                || count as usize > MAX_DOCK_ITEMS
                || index >= count
            {
                return None;
            }
            DisplayRequest::PublishDockItem {
                version,
                index,
                count,
                item: DockItem::with_flags(
                    (a >> 32) as u8,
                    (a >> 40) as u8,
                    label_bytes([w.b.get(), w.c.get(), w.d.get()]),
                )?
                .with_icon(w.subject.get())
                .with_opens(KindMask::from_bits((a >> 48) as u16)?),
            }
        }
        TAKE_SHELL_ACTION => DisplayRequest::TakeShellAction,
        PREPARE_CAPTURE => DisplayRequest::PrepareCapture {
            ticket: u32::try_from(w.subject.get())
                .ok()
                .filter(|ticket| *ticket != 0)?,
        },
        REQUEST_CAPTURE => {
            let mode = CaptureMode::from_code(u8::try_from(w.a.get()).ok()?)?;
            let window = match u16::try_from(w.subject.get()).ok()? {
                0 => None,
                arrival => Some(PeerId::from_arrival(arrival)?),
            };
            if window.is_some() && mode != CaptureMode::Window {
                return None;
            }
            DisplayRequest::RequestCapture { mode, window }
        }
        RELEASE_CAPTURE => DisplayRequest::ReleaseCapture {
            ticket: u32::try_from(w.subject.get())
                .ok()
                .filter(|ticket| *ticket != 0)?,
        },
        READ_CLIPBOARD => {
            let offset = u32::try_from(w.a.get()).ok()?;
            if w.subject.get() == 0 || offset as usize >= crate::clipboard::CLIPBOARD_BYTES {
                return None;
            }
            DisplayRequest::ReadClipboard {
                ticket: w.subject.get(),
                offset,
            }
        }
        op if op & OUTPUT != 0 => {
            DisplayRequest::Output(kore_svc_output::wire::decode_req(&OutReqWire {
                op: op & !OUTPUT,
                reserved0: 0,
                reserved1: 0.into(),
                slot: u32::try_from(w.subject.get()).ok()?.into(),
                through: w.a,
                reserved2: 0.into(),
                reserved3: 0.into(),
            })?)
        }
        _ => DisplayRequest::Surface(surface::decode_req(w)?),
    };
    (encode_req(&request) == *w).then_some(request)
}

pub fn encode_rep(response: &DisplayResponse, grants: u8) -> SurRepWire {
    // A blank template carrying only the grant count; every arm below
    // overwrites the tag and its own fields.
    let mut w = surface::encode_rep(&SurfaceResponse::Destroyed { surface: 0 }, grants);
    match *response {
        DisplayResponse::Desktop(state) => {
            w.tag = DESKTOP;
            w.id = (state.mode as u64).into();
            w.a = u64::from(state.density.quarters()).into();
            w.b = (u64::from(state.width) | u64::from(state.height) << 16).into();
        }
        DisplayResponse::Surface(response) => return surface::encode_rep(&response, grants),
        DisplayResponse::Output(response) => {
            let out = kore_svc_output::wire::encode_rep(&response, grants);
            w.tag = OUTPUT | out.tag;
            w.refusal = out.refusal;
            w.id = u64::from(out.attached).into();
            w.a = out.through;
            w.b = out.dropped;
        }
        DisplayResponse::Accepted => w.tag = ACCEPTED,
        DisplayResponse::Admitted { peer } => {
            w.tag = ADMITTED;
            w.id = u64::from(peer.arrival()).into();
        }
        DisplayResponse::Presentation {
            peer,
            surface,
            presented,
            mode,
            close_requested,
        } => {
            w.tag = PRESENTATION;
            w.id = peer.map_or(0, |peer| u64::from(peer.arrival())).into();
            w.a = surface.into();
            w.b = presented.into();
            w.c = (u64::from(mode as u8) | (u64::from(close_requested) << 8)).into();
        }
        DisplayResponse::ControlRefused(reason) => {
            w.tag = REFUSED;
            w.refusal = reason as u8;
        }
        DisplayResponse::Configured(configure) => {
            w.tag = CONFIGURE;
            w.id = configure.window.into();
            w.a = configure.serial.into();
            w.b = configure.bounds.word().into();
            w.c = u64::from(configure.focused).into();
        }
        DisplayResponse::ShellAction(action) => {
            w.tag = SHELL_ACTION;
            let (id, a, b) = ShellAction::words(action);
            w.id = id.into();
            w.a = a.into();
            w.b = b.into();
        }
        DisplayResponse::ClipboardText(text) => {
            w.tag = CLIPBOARD_TEXT;
            w.id = text.word().into();
            let [a, b, c, d] = text.data();
            w.a = a.into();
            w.b = b.into();
            w.c = c.into();
            w.d = d.into();
        }
    }
    w
}

pub fn decode_rep(w: &SurRepWire) -> Option<(DisplayResponse, u8)> {
    let response = match w.tag {
        DESKTOP
            if w.b.get() >> 32 == 0 && w.b.get() as u16 != 0 && (w.b.get() >> 16) as u16 != 0 =>
        {
            DisplayResponse::Desktop(DesktopState {
                mode: Mode::from_code(u8::try_from(w.id.get()).ok()?)?,
                density: Density::from_quarters(u8::try_from(w.a.get()).ok()?)?,
                width: w.b.get() as u16,
                height: (w.b.get() >> 16) as u16,
            })
        }
        ACCEPTED => DisplayResponse::Accepted,
        ADMITTED => DisplayResponse::Admitted {
            peer: PeerId::from_arrival(u16::try_from(w.id.get()).ok()?)?,
        },
        PRESENTATION => {
            let peer = if w.id.get() == 0 {
                if w.a.get() != 0 || w.b.get() != 0 || w.c.get() != 0 {
                    return None;
                }
                None
            } else {
                Some(PeerId::from_arrival(u16::try_from(w.id.get()).ok()?)?)
            };
            let state = w.c.get();
            if state & !0x1ff != 0 {
                return None;
            }
            DisplayResponse::Presentation {
                peer,
                surface: w.a.get(),
                presented: w.b.get(),
                mode: WindowMode::from_wire(state & 0xff)?,
                close_requested: state & 0x100 != 0,
            }
        }
        REFUSED => DisplayResponse::ControlRefused(match w.refusal {
            1 => ControlRefusal::AccessDenied,
            2 => ControlRefusal::Busy,
            3 => ControlRefusal::BadBundle,
            4 => ControlRefusal::Unavailable,
            5 => ControlRefusal::Pending,
            6 => ControlRefusal::Invalid,
            7 => ControlRefusal::Stale,
            _ => return None,
        }),
        CONFIGURE if w.id.get() != 0 && w.a.get() != 0 && w.c.get() <= 1 => {
            let bounds = crate::configure::Bounds::from_word(w.b.get());
            if !bounds.is_valid() {
                return None;
            }
            DisplayResponse::Configured(Configure {
                window: w.id.get(),
                serial: w.a.get(),
                bounds,
                focused: w.c.get() != 0,
            })
        }
        SHELL_ACTION => {
            DisplayResponse::ShellAction(ShellAction::from_wire(w.id.get(), w.a.get(), w.b.get())?)
        }
        CLIPBOARD_TEXT if w.id.get() >> 56 == 0 => DisplayResponse::ClipboardText(
            ClipboardText::from_words(w.id.get(), [w.a.get(), w.b.get(), w.c.get(), w.d.get()])?,
        ),
        tag if tag & OUTPUT != 0 => {
            DisplayResponse::Output(kore_svc_output::wire::decode_rep(&OutRepWire {
                tag: tag & !OUTPUT,
                grants: w.grants,
                refusal: w.refusal,
                attached: u8::try_from(w.id.get()).ok()?,
                reserved: [0; 4],
                through: w.a,
                dropped: w.b,
            })?)
        }
        _ => DisplayResponse::Surface(surface::decode_rep(w)?.0),
    };
    let expected = match response {
        DisplayResponse::Surface(response) => surface::reply_grants(&response),
        _ => 0,
    };
    (w.grants == expected && encode_rep(&response, w.grants) == *w).then_some((response, w.grants))
}

pub fn reply_matches(w: &SurReqWire, response: &DisplayResponse, grants: u8) -> bool {
    let Some(request) = decode_req(w) else {
        return false;
    };
    match (request, *response) {
        (
            DisplayRequest::Desktop(DesktopRequest::CreateWallpaper { .. }),
            DisplayResponse::Surface(SurfaceResponse::SurfaceMade { .. }),
        ) => grants == 1,
        (DisplayRequest::Desktop(_), DisplayResponse::Surface(SurfaceResponse::Refused { .. })) => {
            grants == 0
        }
        (
            DisplayRequest::Desktop(DesktopRequest::Query | DesktopRequest::Appearance { .. }),
            DisplayResponse::Desktop(_),
        ) => grants == 0,
        (
            DisplayRequest::Desktop(DesktopRequest::ApplyWallpaper { .. } | DesktopRequest::RestoreComplete)
            | DisplayRequest::PrepareSettings { .. },
            DisplayResponse::Accepted,
        ) => grants == 0,
        (
            DisplayRequest::Desktop(_) | DisplayRequest::PrepareSettings { .. },
            DisplayResponse::ControlRefused(_),
        ) => grants == 0,
        (DisplayRequest::Surface(_), DisplayResponse::Surface(response)) => {
            surface::reply_matches(w, &response, grants)
        }
        (_, _) if grants != 0 => false,
        (DisplayRequest::Output(_), DisplayResponse::Output(OutputResponse::Refused(_))) => true,
        (DisplayRequest::Output(request), DisplayResponse::Output(response)) => matches!(
            (request, response),
            (OutputRequest::AttachTx { .. }, OutputResponse::Attached)
                | (OutputRequest::Flush { .. }, OutputResponse::Flushed { .. })
                | (OutputRequest::DetachTx, OutputResponse::Detached)
                | (OutputRequest::Status, OutputResponse::Status(_))
        ),
        (
            DisplayRequest::AdmitPeer { .. }
            | DisplayRequest::RetirePeer { .. }
            | DisplayRequest::RequestClose { .. }
            | DisplayRequest::SetMode { .. }
            | DisplayRequest::PrepareWindow { .. }
            | DisplayRequest::Acknowledge { .. }
            | DisplayRequest::Focus { .. }
            | DisplayRequest::SetTabs(_)
            | DisplayRequest::SetLayoutHints(_)
            | DisplayRequest::PublishDockItem { .. }
            | DisplayRequest::PrepareCapture { .. }
            | DisplayRequest::RequestCapture { .. }
            | DisplayRequest::ReleaseCapture { .. },
            DisplayResponse::Accepted,
        ) => true,
        (DisplayRequest::TakeShellAction, DisplayResponse::ShellAction(_)) => true,
        (
            DisplayRequest::ReadClipboard { offset, .. },
            DisplayResponse::ClipboardText(text),
        ) => text.offset == offset,
        (DisplayRequest::TakeAdmission, DisplayResponse::Admitted { .. }) => true,
        (
            DisplayRequest::CancelAdmission,
            DisplayResponse::Admitted { .. } | DisplayResponse::Accepted,
        ) => true,
        (
            DisplayRequest::Presentation { peer: asked },
            DisplayResponse::Presentation { peer, .. },
        ) => peer.is_none() || peer == Some(asked),
        (DisplayRequest::Configure, DisplayResponse::Configured(_)) => true,
        (
            DisplayRequest::AdmitPeer { .. }
            | DisplayRequest::TakeAdmission
            | DisplayRequest::Presentation { .. }
            | DisplayRequest::RetirePeer { .. }
            | DisplayRequest::RequestClose { .. }
            | DisplayRequest::SetMode { .. }
            | DisplayRequest::PrepareWindow { .. }
            | DisplayRequest::Configure
            | DisplayRequest::Acknowledge { .. }
            | DisplayRequest::Focus { .. }
            | DisplayRequest::SetTabs(_)
            | DisplayRequest::SetLayoutHints(_)
            | DisplayRequest::PublishDockItem { .. }
            | DisplayRequest::TakeShellAction
            | DisplayRequest::PrepareCapture { .. }
            | DisplayRequest::RequestCapture { .. }
            | DisplayRequest::ReleaseCapture { .. }
            | DisplayRequest::ReadClipboard { .. },
            DisplayResponse::ControlRefused(_),
        ) => true,
        (DisplayRequest::CancelAdmission, DisplayResponse::ControlRefused(_)) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespaces_roundtrip_and_reject_unused_words() {
        for request in [
            DisplayRequest::Surface(SurfaceRequest::Query),
            DisplayRequest::Output(OutputRequest::AttachTx {
                slot: SlotRef::from_wire(17),
            }),
            DisplayRequest::Output(OutputRequest::Flush { through: u64::MAX }),
            DisplayRequest::Output(OutputRequest::DetachTx),
            DisplayRequest::Output(OutputRequest::Status),
            DisplayRequest::TakeAdmission,
            DisplayRequest::CancelAdmission,
            DisplayRequest::Presentation {
                peer: PeerId::from_arrival(kore_rt::svc::PeerId::SLOT_CAPACITY as u16).unwrap(),
            },
            DisplayRequest::PrepareWindow {
                content: Content::Application,
                preference: Preference::FullHeightAspect {
                    width: 2,
                    height: 3,
                },
                icon: 0x9e37_79b9_7f4a_7c15,
            },
            DisplayRequest::PrepareWindow {
                content: Content::Terminal,
                preference: Preference::Tile,
                icon: 0,
            },
            DisplayRequest::Configure,
            DisplayRequest::Acknowledge {
                serial: 2,
                surface: 5,
            },
            DisplayRequest::Focus {
                peer: PeerId::from_arrival(kore_rt::svc::PeerId::SLOT_CAPACITY as u16).unwrap(),
            },
            DisplayRequest::RequestClose {
                peer: PeerId::from_arrival(kore_rt::svc::PeerId::SLOT_CAPACITY as u16).unwrap(),
            },
            DisplayRequest::SetMode {
                peer: PeerId::from_arrival(kore_rt::svc::PeerId::SLOT_CAPACITY as u16).unwrap(),
                mode: WindowMode::Minimized,
            },
            DisplayRequest::AdmitPeer {
                part: PeerBundlePart::Ring,
                announced: SlotRef::from_wire(19),
            },
            DisplayRequest::AdmitPeer {
                part: PeerBundlePart::Grant(0),
                announced: SlotRef::from_wire(19),
            },
        ] {
            let mut wire = encode_req(&request);
            assert_eq!(decode_req(&wire), Some(request));
            wire.d = 1.into();
            assert_eq!(decode_req(&wire), None);
        }
        for response in [
            DisplayResponse::Output(OutputResponse::Attached),
            DisplayResponse::Output(OutputResponse::Flushed { through: 23 }),
            DisplayResponse::Accepted,
            DisplayResponse::Presentation {
                peer: None,
                surface: 0,
                presented: 0,
                mode: WindowMode::Normal,
                close_requested: false,
            },
            DisplayResponse::Presentation {
                peer: Some(
                    PeerId::from_arrival(kore_rt::svc::PeerId::SLOT_CAPACITY as u16).unwrap(),
                ),
                surface: 9,
                presented: 7,
                mode: WindowMode::Maximized,
                close_requested: true,
            },
            DisplayResponse::ControlRefused(ControlRefusal::Busy),
            DisplayResponse::Configured(Configure {
                window: 0x1_0000_0002,
                serial: 7,
                bounds: crate::configure::Bounds {
                    x: 12,
                    y: 8,
                    width: 533,
                    height: 800,
                },
                focused: true,
            }),
        ] {
            let mut wire = encode_rep(&response, 0);
            assert_eq!(decode_rep(&wire), Some((response, 0)));
            wire.grants = 1;
            assert_eq!(decode_rep(&wire), None);
        }
    }

    #[test]
    fn desktop_requests_and_state_are_strict_and_grants_are_correlated() {
        use kore_rt::svc::SvcProtocol;
        let state = DesktopState { mode: Mode::System, density: Density::Double, width: 1920, height: 1200 };
        let requests = [
            DisplayRequest::PrepareSettings { preference: Preference::Tile, icon: 42 },
            DisplayRequest::Desktop(DesktopRequest::Query),
            DisplayRequest::Desktop(DesktopRequest::RestoreComplete),
            DisplayRequest::Desktop(DesktopRequest::Appearance { mode: state.mode, density: state.density }),
            DisplayRequest::Desktop(DesktopRequest::CreateWallpaper { width: state.width, height: state.height }),
            DisplayRequest::Desktop(DesktopRequest::ApplyWallpaper { surface: 42 }),
        ];
        for request in requests {
            let wire = encode_req(&request);
            assert_eq!(decode_req(&wire), Some(request));
            assert_eq!(WindowWire::request_grants(&wire), usize::from(matches!(request, DisplayRequest::Desktop(DesktopRequest::CreateWallpaper { .. }))));
            let mut invalid = wire; invalid.d = 1.into();
            assert_eq!(decode_req(&invalid), None);
            assert!(reply_matches(&wire, &DisplayResponse::ControlRefused(ControlRefusal::AccessDenied), 0));
            assert!(!reply_matches(&wire, &DisplayResponse::Accepted, 1));
        }
        let reply = DisplayResponse::Desktop(state);
        assert!(reply_matches(&encode_req(&requests[2]), &DisplayResponse::Accepted, 0));
        assert!(!reply_matches(&encode_req(&requests[2]), &reply, 0));
        assert_eq!(decode_rep(&encode_rep(&reply, 0)), Some((reply, 0)));
        assert_eq!(decode_rep(&encode_rep(&reply, 1)), None);
        for invalid in [
            DesktopRequest::CreateWallpaper { width: 0, height: 1 },
            DesktopRequest::ApplyWallpaper { surface: 0 },
        ] { assert_eq!(decode_req(&encode_req(&DisplayRequest::Desktop(invalid))), None); }
        let mut bad = encode_req(&requests[3]); bad.a = 3.into();
        assert_eq!(decode_req(&bad), None);
        let mut bad = encode_req(&requests[3]); bad.b = 11.into();
        assert_eq!(decode_req(&bad), None);
        assert!(!reply_matches(&encode_req(&requests[4]), &reply, 0));
    }

    /// The capture vocabulary rides the controller face byte-exactly: a
    /// zero ticket, an unknown mode or refusal, and stray bits refuse.
    #[test]
    fn capture_verbs_and_actions_round_trip_and_fail_closed() {
        for request in [
            DisplayRequest::PrepareCapture { ticket: 1 },
            DisplayRequest::PrepareCapture { ticket: u32::MAX },
            DisplayRequest::RequestCapture {
                mode: CaptureMode::Screen,
                window: None,
            },
            DisplayRequest::RequestCapture {
                mode: CaptureMode::Region,
                window: None,
            },
            DisplayRequest::RequestCapture {
                mode: CaptureMode::Window,
                window: None,
            },
            DisplayRequest::RequestCapture {
                mode: CaptureMode::Window,
                window: PeerId::from_arrival(0x1_05),
            },
            DisplayRequest::ReleaseCapture { ticket: 3 },
        ] {
            let wire = encode_req(&request);
            assert_eq!(decode_req(&wire), Some(request));
            assert!(reply_matches(&wire, &DisplayResponse::Accepted, 0));
            assert!(reply_matches(
                &wire,
                &DisplayResponse::ControlRefused(ControlRefusal::Busy),
                0
            ));
            assert!(!reply_matches(&wire, &DisplayResponse::Accepted, 1));
            let mut stray = wire;
            stray.d = 1.into();
            assert_eq!(decode_req(&stray), None);
        }
        let mut zero = encode_req(&DisplayRequest::PrepareCapture { ticket: 1 });
        zero.subject = 0.into();
        assert_eq!(decode_req(&zero), None, "a zero ticket names nothing");
        let mut wide = zero;
        wide.subject = (u64::from(u32::MAX) + 1).into();
        assert_eq!(decode_req(&wide), None);
        let mut unknown = encode_req(&DisplayRequest::RequestCapture {
            mode: CaptureMode::Screen,
            window: None,
        });
        unknown.a = 9.into();
        assert_eq!(decode_req(&unknown), None);
        let mut named = encode_req(&DisplayRequest::RequestCapture {
            mode: CaptureMode::Screen,
            window: None,
        });
        named.subject = 0x1_05.into();
        assert_eq!(decode_req(&named), None, "only a window capture names a window");
        named.a = u64::from(CaptureMode::Window.code()).into();
        named.subject = 0x05.into();
        assert_eq!(decode_req(&named), None, "a static pair is no window");
        named.subject = 0x1_0000.into();
        assert_eq!(decode_req(&named), None, "an arrival is 16 bits");

        for action in [
            ShellAction::Captured {
                ticket: 7,
                mode: CaptureMode::Screen,
                width: 2560,
                height: 1600,
                window: None,
            },
            ShellAction::Captured {
                ticket: u32::MAX,
                mode: CaptureMode::Window,
                width: 640,
                height: 400,
                window: PeerId::from_arrival(0xffff),
            },
            ShellAction::CaptureRefused {
                mode: CaptureMode::Window,
                refusal: CaptureRefusal::NoWindow,
            },
        ] {
            let response = DisplayResponse::ShellAction(Some(action));
            let wire = encode_rep(&response, 0);
            assert_eq!(decode_rep(&wire), Some((response, 0)));
            let mut stray = wire;
            stray.c = 1.into();
            assert_eq!(decode_rep(&stray), None);
        }
        for (a, b) in [
            (7 | 1 << 32, 0),               // zero geometry
            (1 << 32, 2560 | 1600 << 16),   // zero ticket
            (7 | 9 << 32, 2560 | 1600 << 16), // unknown mode
            (7 | 1 << 40, 2560 | 1600 << 16), // a bit past the mode
            (7 | 1 << 32, 2560 | 1600 << 16 | 0x1_05 << 32), // a screen names no window
            (7 | 2 << 32, 2560 | 1600 << 16), // a window capture names its window
            (7 | 2 << 32, 2560 | 1600 << 16 | 0x05 << 32), // a static pair is no window
            (7 | 2 << 32, 2560 | 1600 << 16 | 1 << 48), // a bit past the window
        ] {
            assert_eq!(ShellAction::from_wire(4, a, b), None, "{a:#x} {b:#x}");
        }
        assert_eq!(ShellAction::from_wire(5, 1, 0), None, "no refusal code");
        assert_eq!(ShellAction::from_wire(5, 1, 9), None, "unknown refusal");
    }

    #[test]
    fn surface_clients_share_wire_but_cannot_accept_control_replies() {
        let request = surface::encode_req(&SurfaceRequest::Status { surface: 9 });
        let response = DisplayResponse::Surface(SurfaceResponse::Status {
            surface: 9,
            committed: 7,
            presented: 6,
            refused_commits: 0,
            presented_at: 0,
        });
        assert!(reply_matches(&request, &response, 0));
        assert!(!reply_matches(&request, &DisplayResponse::Accepted, 0));
        assert!(surface::decode_rep(&encode_rep(&response, 0)).is_some());
        assert!(surface::decode_rep(&encode_rep(&DisplayResponse::Accepted, 0)).is_none());
    }

    #[test]
    fn presentation_wait_cannot_accept_a_different_peer_generation() {
        let peer = PeerId::from_arrival(kore_rt::svc::PeerId::SLOT_CAPACITY as u16).unwrap();
        let successor =
            PeerId::from_arrival(2 * kore_rt::svc::PeerId::SLOT_CAPACITY as u16).unwrap();
        assert_eq!(peer.slot(), successor.slot());
        let request = encode_req(&DisplayRequest::Presentation { peer });
        for (candidate, accepted) in [(Some(peer), true), (Some(successor), false), (None, true)] {
            assert_eq!(
                reply_matches(
                    &request,
                    &DisplayResponse::Presentation {
                        peer: candidate,
                        surface: 0,
                        presented: 0,
                        mode: WindowMode::Normal,
                        close_requested: false,
                    },
                    0
                ),
                accepted
            );
        }
        let cancel = encode_req(&DisplayRequest::CancelAdmission);
        assert!(reply_matches(&cancel, &DisplayResponse::Accepted, 0));
        assert!(reply_matches(
            &cancel,
            &DisplayResponse::Admitted { peer },
            0
        ));
        assert!(!reply_matches(
            &cancel,
            &DisplayResponse::Admitted { peer },
            1
        ));
    }

    #[test]
    fn close_request_accepts_only_control_replies() {
        let request = encode_req(&DisplayRequest::RequestClose {
            peer: PeerId::from_arrival(kore_rt::svc::PeerId::SLOT_CAPACITY as u16).unwrap(),
        });
        assert!(reply_matches(&request, &DisplayResponse::Accepted, 0));
        assert!(reply_matches(
            &request,
            &DisplayResponse::ControlRefused(ControlRefusal::Busy),
            0
        ));
        assert!(!reply_matches(
            &request,
            &DisplayResponse::Presentation {
                peer: None,
                surface: 0,
                presented: 0,
                mode: WindowMode::Normal,
                close_requested: false,
            },
            0
        ));
        assert!(!reply_matches(&request, &DisplayResponse::Accepted, 1));
    }

    #[test]
    fn set_mode_is_canonical_and_accepts_only_control_replies() {
        let peer = PeerId::from_arrival(kore_rt::svc::PeerId::SLOT_CAPACITY as u16).unwrap();
        for mode in [
            WindowMode::Normal,
            WindowMode::Minimized,
            WindowMode::Maximized,
        ] {
            let request = DisplayRequest::SetMode { peer, mode };
            let wire = encode_req(&request);
            assert_eq!(decode_req(&wire), Some(request));
            assert!(reply_matches(&wire, &DisplayResponse::Accepted, 0));
            assert!(reply_matches(
                &wire,
                &DisplayResponse::ControlRefused(ControlRefusal::Stale),
                0
            ));
        }

        let mut invalid = encode_req(&DisplayRequest::SetMode {
            peer,
            mode: WindowMode::Normal,
        });
        invalid.a = 3.into();
        assert_eq!(decode_req(&invalid), None);

        let mut invalid_state = encode_rep(
            &DisplayResponse::Presentation {
                peer: Some(peer),
                surface: 1,
                presented: 1,
                mode: WindowMode::Normal,
                close_requested: false,
            },
            0,
        );
        invalid_state.c = 3.into();
        assert_eq!(decode_rep(&invalid_state), None);
        invalid_state.c = 0x200.into();
        assert_eq!(decode_rep(&invalid_state), None);
    }

    #[test]
    fn tabs_dock_and_shell_actions_roundtrip_and_fail_closed() {
        let tabs = TabModel::from_labels(3, 1, &["Terminal", "aftershock"]).unwrap();
        assert_eq!(tabs.count(), 2);
        assert_eq!(tabs.active(), 1);
        assert_eq!(&tabs.initials()[..3], b"Ta\0");
        let item = DockItem::from_label(2, "Skylilt").unwrap();
        assert_eq!(item.label_bytes(), b"Skylilt");
        assert!(!item.running());
        assert_eq!(item.flags(), 0);
        let running = item.with_running(true);
        assert!(running.running());
        assert_eq!(running.flags(), DockItem::RUNNING);
        assert_eq!(running.with_running(false), item);
        assert_eq!(running.with_running(true), running);
        assert_eq!(
            DockItem::with_flags(2, DockItem::RUNNING, *item.label()),
            Some(running)
        );
        assert_eq!(DockItem::with_flags(2, 0, *item.label()), Some(item));
        for request in [
            DisplayRequest::SetTabs(tabs),
            DisplayRequest::SetTabs(TabModel::new(1, 8, 7, *b"ABCDEFGH").unwrap()),
            DisplayRequest::SetLayoutHints(
                LayoutHints::new(
                    1,
                    LayoutHints::NEVER_MAXIMIZE | LayoutHints::FIXED_ASPECT,
                    (240, 360),
                    (960, 1440),
                    (2, 3),
                )
                .unwrap(),
            ),
            DisplayRequest::SetLayoutHints(
                LayoutHints::new(u16::MAX, 0, (0, 0), (0, 0), (0, 0)).unwrap(),
            ),
            DisplayRequest::PublishDockItem {
                version: 1,
                index: 0,
                count: 1,
                item,
            },
            DisplayRequest::PublishDockItem {
                version: u16::MAX,
                index: 7,
                count: 8,
                item: DockItem::from_label(u8::MAX, "123456789012345678901234").unwrap(),
            },
            DisplayRequest::PublishDockItem {
                version: 3,
                index: 1,
                count: 2,
                item: DockItem::from_label(DOCK_KIND_TERMINAL, "Terminal")
                    .unwrap()
                    .with_running(true)
                    .with_icon(u64::MAX),
            },
            DisplayRequest::TakeShellAction,
        ] {
            let mut wire = encode_req(&request);
            assert_eq!(decode_req(&wire), Some(request));
            wire.flags = 1.into();
            assert_eq!(decode_req(&wire), None);
            assert!(reply_matches(
                &encode_req(&request),
                &DisplayResponse::ControlRefused(ControlRefusal::Stale),
                0
            ));
            assert!(!reply_matches(
                &encode_req(&request),
                &DisplayResponse::Accepted,
                1
            ));
        }
        let publish = encode_req(&DisplayRequest::PublishDockItem {
            version: 1,
            index: 0,
            count: 1,
            item,
        });
        assert!(reply_matches(&publish, &DisplayResponse::Accepted, 0));
        assert!(!reply_matches(
            &publish,
            &DisplayResponse::ShellAction(None),
            0
        ));
        let take = encode_req(&DisplayRequest::TakeShellAction);
        assert!(reply_matches(&take, &DisplayResponse::ShellAction(None), 0));
        assert!(!reply_matches(&take, &DisplayResponse::Accepted, 0));

        // Structural refusals never reach the model constructors.
        assert_eq!(TabModel::new(0, 1, 0, *b"A\0\0\0\0\0\0\0"), None);
        assert_eq!(TabModel::new(1, 0, 0, [0; 8]), None);
        assert_eq!(TabModel::new(1, 9, 0, *b"ABCDEFGH"), None);
        assert_eq!(TabModel::new(1, 2, 2, *b"AB\0\0\0\0\0\0"), None);
        assert_eq!(TabModel::new(1, 1, 0, *b"\x1f\0\0\0\0\0\0\0"), None);
        assert_eq!(TabModel::new(1, 1, 0, *b"AB\0\0\0\0\0\0"), None);
        assert_eq!(TabModel::from_labels(1, 0, &[]), None);
        assert_eq!(TabModel::from_labels(1, 0, &[""]), None);
        assert_eq!(DockItem::new(0, [0; DOCK_LABEL_BYTES]), None);
        assert_eq!(DockItem::from_label(0, "a\0b"), None);
        assert_eq!(DockItem::from_label(0, "\u{7f}"), None);
        assert_eq!(DockItem::from_label(0, "1234567890123456789012345"), None);
        let mut wire = encode_req(&DisplayRequest::SetTabs(tabs));
        wire.a = (3 | (2 << 16) | (2 << 24)).into();
        assert_eq!(decode_req(&wire), None);
        wire.a = (3 | (2 << 16) | (1 << 24) | (1 << 32)).into();
        assert_eq!(decode_req(&wire), None);
        let mut wire = publish;
        wire.a = (1 | (1 << 16) | (1 << 24) | (2 << 32)).into();
        assert_eq!(decode_req(&wire), None);
        wire.a = (1 | (((MAX_DOCK_ITEMS + 1) as u64) << 24) | (2 << 32)).into();
        assert_eq!(decode_req(&wire), None);
        // Undefined flag bits and handler bits no document class has fail
        // closed; the defined running bit decodes to the flagged row, and
        // the handler mask to the row that opens those classes.
        wire.a = (1 | (1 << 24) | (2 << 32) | (2 << 40)).into();
        assert_eq!(decode_req(&wire), None);
        wire.a = (1 | (1 << 24) | (2 << 32) | (1 << (48 + DocumentKind::ALL.len()))).into();
        assert_eq!(decode_req(&wire), None);
        let editor = KindMask::of(&[DocumentKind::Text, DocumentKind::Code]);
        wire.a = (1 | (1 << 24) | (2 << 32) | (u64::from(editor.bits()) << 48)).into();
        assert_eq!(
            decode_req(&wire),
            Some(DisplayRequest::PublishDockItem {
                version: 1,
                index: 0,
                count: 1,
                item: item.with_opens(editor),
            })
        );
        wire.a = (1 | (1 << 24) | (2 << 32) | (1 << 40)).into();
        assert_eq!(
            decode_req(&wire),
            Some(DisplayRequest::PublishDockItem {
                version: 1,
                index: 0,
                count: 1,
                item: item.with_running(true),
            })
        );
        assert_eq!(DockItem::with_flags(0, 2, *item.label()), None);
        assert_eq!(DockItem::with_flags(0, 0, [0; DOCK_LABEL_BYTES]), None);
        wire.a = (2 << 32).into();
        assert_eq!(decode_req(&wire), None);

        for response in [
            DisplayResponse::ShellAction(None),
            DisplayResponse::ShellAction(Some(ShellAction::Launch {
                version: 9,
                index: 7,
            })),
        ] {
            let mut wire = encode_rep(&response, 0);
            assert_eq!(decode_rep(&wire), Some((response, 0)));
            wire.b = 1.into();
            assert_eq!(decode_rep(&wire), None);
        }
        // Opens carry their object's class in `b`; an "Open With" its dock
        // row there, under the same row law as a launch.
        for response in [
            DisplayResponse::ShellAction(Some(ShellAction::Open {
                object: 0x5eed,
                kind: DocumentKind::Unknown,
            })),
            DisplayResponse::ShellAction(Some(ShellAction::Open {
                object: u64::MAX,
                kind: DocumentKind::Binary,
            })),
            DisplayResponse::ShellAction(Some(ShellAction::OpenWith {
                version: 4,
                index: 7,
                object: 9,
            })),
            DisplayResponse::ShellAction(Some(ShellAction::Pick {
                token: u64::MAX,
                types: 0b100,
            })),
        ] {
            let wire = encode_rep(&response, 0);
            assert_eq!(decode_rep(&wire), Some((response, 0)));
        }
        // A pick names its token and at least one type.
        let pick = encode_rep(&DisplayResponse::ShellAction(Some(ShellAction::Pick { token: 5, types: 1 })), 0);
        for (a, b) in [(0, 1), (5, 0)] {
            let mut bad = pick;
            bad.a = a.into();
            bad.b = b.into();
            assert_eq!(decode_rep(&bad), None, "token {a} types {b}");
        }
        let mut open = encode_rep(
            &DisplayResponse::ShellAction(Some(ShellAction::Open {
                object: 1,
                kind: DocumentKind::Text,
            })),
            0,
        );
        open.b = (DocumentKind::ALL.len() as u64).into();
        assert_eq!(decode_rep(&open), None);
        let mut with = encode_rep(
            &DisplayResponse::ShellAction(Some(ShellAction::OpenWith {
                version: 4,
                index: 0,
                object: 9,
            })),
            0,
        );
        with.b = (4 | ((MAX_DOCK_ITEMS as u64) << 16)).into();
        assert_eq!(decode_rep(&with), None, "a row past the dock");
        with.b = 0.into();
        assert_eq!(decode_rep(&with), None, "no catalog version");
        with.b = 4.into();
        with.a = 0.into();
        assert_eq!(decode_rep(&with), None, "no object");
        let mut wire = encode_rep(&DisplayResponse::ShellAction(None), 0);
        wire.a = 1.into();
        assert_eq!(decode_rep(&wire), None);
        wire.a = 0.into();
        wire.id = 2.into();
        assert_eq!(decode_rep(&wire), None);
        wire.id = 1.into();
        assert_eq!(decode_rep(&wire), None);
        wire.a = (1 | ((MAX_DOCK_ITEMS as u64) << 16)).into();
        assert_eq!(decode_rep(&wire), None);
        wire.a = (1 | (1 << 24)).into();
        assert_eq!(decode_rep(&wire), None);
    }

    /// A shell paste names its ticket; a clipboard read answers its own
    /// offset with a piece inside its text, zero past its length, and
    /// nothing else; and an unticketed or out-of-bound read refuses.
    #[test]
    fn a_shell_paste_reads_its_ticketed_text_and_fails_closed() {
        let paste = DisplayResponse::ShellAction(Some(ShellAction::Paste { ticket: 0xfeed }));
        assert_eq!(decode_rep(&encode_rep(&paste, 0)), Some((paste, 0)));
        let mut bad = encode_rep(&paste, 0);
        bad.a = 0.into();
        assert_eq!(decode_rep(&bad), None, "no ticket");
        bad.a = 1.into();
        bad.b = 1.into();
        assert_eq!(decode_rep(&bad), None, "a stray word");

        let read = DisplayRequest::ReadClipboard {
            ticket: 0xfeed,
            offset: 32,
        };
        let wire = encode_req(&read);
        assert_eq!(decode_req(&wire), Some(read));
        for (ticket, offset) in [(0, 0), (1, crate::clipboard::CLIPBOARD_BYTES as u32)] {
            assert_eq!(
                decode_req(&encode_req(&DisplayRequest::ReadClipboard { ticket, offset })),
                None,
                "ticket {ticket} offset {offset}"
            );
        }
        let mut bytes = [0; CLIPBOARD_TEXT_BYTES];
        bytes[..5].copy_from_slice(b"ls -l");
        let text = ClipboardText {
            offset: 32,
            len: 5,
            total: 37,
            bytes,
        };
        let answer = DisplayResponse::ClipboardText(text);
        let reply = encode_rep(&answer, 0);
        assert_eq!(decode_rep(&reply), Some((answer, 0)));
        assert_eq!(text.text(), b"ls -l");
        assert!(reply_matches(&wire, &answer, 0));
        assert!(!reply_matches(&wire, &answer, 1));
        assert!(reply_matches(&wire, &DisplayResponse::ControlRefused(ControlRefusal::Stale), 0));
        let elsewhere = encode_req(&DisplayRequest::ReadClipboard {
            ticket: 0xfeed,
            offset: 0,
        });
        assert!(!reply_matches(&elsewhere, &answer, 0));
        let mut tail = reply;
        tail.d = (tail.d.get() | 1 << 56).into();
        assert_eq!(decode_rep(&tail), None, "a live byte past the piece");
        for (offset, len, total) in [(0u32, 0u16, 5u32), (0, 33, 64), (33, 5, 37), (0, 1, 0)] {
            let mut bad = reply;
            bad.id = (u64::from(offset) | u64::from(total) << 24 | u64::from(len) << 48).into();
            bad.a = 0.into();
            bad.b = 0.into();
            bad.c = 0.into();
            bad.d = 0.into();
            assert_eq!(decode_rep(&bad), None, "{offset} {len} {total}");
        }
    }

    #[test]
    fn the_ninth_dock_row_round_trips_publication_and_launch() {
        let request = DisplayRequest::PublishDockItem {
            version: 1,
            index: 8,
            count: 9,
            item: DockItem::from_label(DOCK_KIND_APPLICATION, "Skylilt").unwrap(),
        };
        assert_eq!(decode_req(&encode_req(&request)), Some(request));
        let response = DisplayResponse::ShellAction(Some(ShellAction::Launch {
            version: 1,
            index: 8,
        }));
        assert_eq!(decode_rep(&encode_rep(&response, 0)), Some((response, 0)));
    }

    #[test]
    fn configure_refuses_empty_geometry_or_serial_and_ack_refuses_zero_subjects() {
        let configured = DisplayResponse::Configured(Configure {
            window: 0x1_0000_0001,
            serial: 2,
            bounds: crate::configure::Bounds {
                x: 0,
                y: 0,
                width: 1280,
                height: 800,
            },
            focused: false,
        });
        let good = encode_rep(&configured, 0);
        for invalid in [0, 1, 2, 3] {
            let mut wire = good;
            match invalid {
                0 => wire.id = 0.into(),
                1 => wire.a = 0.into(),
                2 => wire.b = 0.into(),
                _ => wire.c = 2.into(),
            }
            assert_eq!(decode_rep(&wire), None);
        }
        for (serial, surface) in [(0, 1), (1, 0)] {
            assert_eq!(
                decode_req(&encode_req(&DisplayRequest::Acknowledge {
                    serial,
                    surface
                })),
                None
            );
        }
    }
    #[test]
    fn layout_hints_validate_structurally() {
        // Version 0 is the unpublished state, never publishable.
        assert!(LayoutHints::new(0, 0, (0, 0), (0, 0), (0, 0)).is_none());
        // Unknown flags refuse (a future revision, not a guess).
        assert!(LayoutHints::new(1, 1 << 15, (0, 0), (0, 0), (0, 0)).is_none());
        // A bounded max below its min refuses; an unbounded axis (0) is fine.
        assert!(LayoutHints::new(1, 0, (100, 0), (99, 0), (0, 0)).is_none());
        assert!(LayoutHints::new(1, 0, (100, 100), (0, 0), (0, 0)).is_some());
        // The aspect is both-or-neither, and FIXED_ASPECT requires it.
        assert!(LayoutHints::new(1, 0, (0, 0), (0, 0), (4, 0)).is_none());
        assert!(LayoutHints::new(1, LayoutHints::FIXED_ASPECT, (0, 0), (0, 0), (0, 0)).is_none());
        let hints = LayoutHints::new(
            2,
            LayoutHints::NEVER_MAXIMIZE | LayoutHints::FIXED_ASPECT,
            (240, 360),
            (960, 1440),
            (2, 3),
        )
        .unwrap();
        assert!(hints.never_maximize());
        assert_eq!(hints.max(), (960, 1440));
        assert_eq!(hints.aspect(), (2, 3));
        assert_eq!(LayoutHints::UNPUBLISHED.version(), 0);
    }
}
