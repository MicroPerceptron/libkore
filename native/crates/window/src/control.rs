//! One display face for founder-owned output/control and admitted surfaces.
//!
//! Surface records keep their canonical ABI, so a `SurfaceWire` client does
//! not depend on the terminal facade. The extra namespaces are disjoint;
//! arrival identity, not an opcode or a terminal number supplied by a client,
//! decides which vocabulary the service permits.

use crate::configure::{Configure, Content, Preference};
use crate::wire::{self as surface, SurfaceRequest, SurfaceResponse};
use kore_abi::{
    session::SlotRef,
    svc::{OutRepWire, OutReqWire, SurRepWire, SurReqWire},
};
use kore_rt::svc::{PeerBundlePart, PeerId};
use kore_svc_output::service::{OutputRequest, OutputResponse};

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
const ACCEPTED: u8 = 0x40;
const ADMITTED: u8 = 0x41;
const REFUSED: u8 = 0x42;
const SHELL_ACTION: u8 = 0x4d;

/// Bound on one window's published tab strip.
pub const MAX_TABS: usize = 8;
/// Bound on the compositor-drawn dock catalog.
pub const MAX_DOCK_ITEMS: usize = 8;
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
}

impl ShellAction {
    const fn from_wire(id: u64, a: u64) -> Option<Option<Self>> {
        match id {
            0 => {
                if a != 0 {
                    return None;
                }
                Some(None)
            }
            1 => {
                let version = a as u16;
                let index = (a >> 16) as u8;
                if a >> 24 != 0 || version == 0 || index as usize >= MAX_DOCK_ITEMS {
                    return None;
                }
                Some(Some(Self::Launch { version, index }))
            }
            _ => None,
        }
    }

    const fn words(action: Option<Self>) -> (u64, u64) {
        match action {
            None => (0, 0),
            Some(Self::Launch { version, index }) => (1, version as u64 | ((index as u64) << 16)),
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
        usize::from(matches!(
            decode_req(request),
            Some(DisplayRequest::Surface(
                SurfaceRequest::CreateSurface { .. }
            ))
        ))
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplayResponse {
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
        } => {
            w.op = PREPARE;
            w.subject = u64::from(content == Content::Application).into();
            let (kind, value) = preference.words();
            w.a = kind.into();
            w.b = value.into();
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
                | (u64::from(item.flags) << 40))
                .into();
            let [b, c, d] = label_words(&item.label);
            w.b = b.into();
            w.c = c.into();
            w.d = d.into();
        }
        DisplayRequest::TakeShellAction => w.op = TAKE_SHELL_ACTION,
    }
    w
}

pub fn decode_req(w: &SurReqWire) -> Option<DisplayRequest> {
    let request = match w.op {
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
            if a >> 48 != 0
                || version == 0
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
                .with_icon(w.subject.get()),
            }
        }
        TAKE_SHELL_ACTION => DisplayRequest::TakeShellAction,
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
            let (id, a) = ShellAction::words(action);
            w.id = id.into();
            w.a = a.into();
        }
    }
    w
}

pub fn decode_rep(w: &SurRepWire) -> Option<(DisplayResponse, u8)> {
    let response = match w.tag {
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
            DisplayResponse::ShellAction(ShellAction::from_wire(w.id.get(), w.a.get())?)
        }
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
            | DisplayRequest::PublishDockItem { .. },
            DisplayResponse::Accepted,
        ) => true,
        (DisplayRequest::TakeShellAction, DisplayResponse::ShellAction(_)) => true,
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
            | DisplayRequest::TakeShellAction,
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
        wire.a = (1 | (9 << 24) | (2 << 32)).into();
        assert_eq!(decode_req(&wire), None);
        // Undefined flag bits and bits above the flag byte fail closed;
        // the defined running bit decodes to the flagged row.
        wire.a = (1 | (1 << 24) | (2 << 32) | (2 << 40)).into();
        assert_eq!(decode_req(&wire), None);
        wire.a = (1 | (1 << 24) | (2 << 32) | (1 << 48)).into();
        assert_eq!(decode_req(&wire), None);
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
        let mut wire = encode_rep(&DisplayResponse::ShellAction(None), 0);
        wire.a = 1.into();
        assert_eq!(decode_rep(&wire), None);
        wire.a = 0.into();
        wire.id = 2.into();
        assert_eq!(decode_rep(&wire), None);
        wire.id = 1.into();
        assert_eq!(decode_rep(&wire), None);
        wire.a = (1 | (8 << 16)).into();
        assert_eq!(decode_rep(&wire), None);
        wire.a = (1 | (1 << 24)).into();
        assert_eq!(decode_rep(&wire), None);
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
