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

/// Window clients share the same four capability auxiliaries as surface
/// clients. The event channel (`crate::channel`, `res/EVENT.md`) occupies
/// the pair's remaining bytes, not a fifth endpoint: the compositor's
/// level state and event records, with one conditional wake.
pub fn split_window_aux(aux: &mut [u8]) -> Option<(&mut [u8], &mut [u8], &mut [u8])> {
    let commits = kore_ring::spsc::region_bytes::<CommitWire>(crate::COMMIT_CAP);
    let output = kore_ring::spsc::region_bytes::<kore_abi::svc::OutChunk>(TERMINAL_OUTPUT_CAP);
    let channel = crate::channel::region_bytes();
    if aux.len() < commits.checked_add(output)?.checked_add(channel)? {
        return None;
    }
    let (commits, tail) = aux.split_at_mut(commits);
    let (output, tail) = tail.split_at_mut(output);
    Some((commits, output, &mut tail[..channel]))
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
        usize::from(matches!(
            decode_req(request),
            Some(SurfaceRequest::CreateSurface { .. })
        ))
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
        let (commits, output, events) =
            split_window_aux(plane).ok_or(kore_ring::spsc::RingError::TooSmall)?;
        kore_ring::spsc::init::<CommitWire>(commits, crate::COMMIT_CAP)?;
        kore_ring::spsc::init::<kore_abi::svc::OutChunk>(output, TERMINAL_OUTPUT_CAP)?;
        crate::channel::init(events)
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
    }
    w
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
        // The backing's detached window view.
        SurfaceResponse::SurfaceMade { .. } => 1,
        _ => 0,
    }
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

    fn every_request() -> [SurfaceRequest; 5] {
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
        ]
    }

    fn every_response() -> [SurfaceResponse; 6] {
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
        ]
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
