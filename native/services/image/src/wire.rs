//! `wire` — the image enum ⇄ wire codecs (`res/USERLAND.md` §5.3).
//!
//! The zerocopy forms live in [`kore_abi::svc`]; decode is fail-closed.
//! The [`ModuleDescriptor`] rides replies whole (it is itself frozen
//! wire); on `Retired`/`Refused` it rides
//! [`empty`](ModuleDescriptor::empty) — its zero `abi` stamp refuses
//! validation, so a data-less reply can never be mistaken for a promise.

use kore_abi::image::{ImageName, ImageRefusal, ImageSourceMetadata, ModuleDescriptor};
use kore_abi::session::SlotRef;
use kore_abi::svc::{ImageRepWire, ImageReqWire};

use kore_rt::svc::SvcProtocol;

use crate::service::{ImageRequest, ImageResponse, ImageService};

/// Lower a request to its wire form (total).
pub fn encode_req(req: &ImageRequest) -> ImageReqWire {
    let (op, name, slot, source) = match *req {
        ImageRequest::Register { name, slot, source } => {
            (ImageReqWire::OP_REGISTER, name, slot.index(), source)
        }
        // No live caller constructs this; it exists so the tombstone family
        // round-trips through one vocabulary. Any frozen value serves.
        ImageRequest::Retired => (
            ImageReqWire::OP_EXECUTABLE_AT,
            ImageName::empty(),
            0,
            ImageSourceMetadata::empty(),
        ),
        ImageRequest::Acquire { name } => (
            ImageReqWire::OP_ACQUIRE,
            name,
            0,
            ImageSourceMetadata::empty(),
        ),
        ImageRequest::AcquireEntry => (
            ImageReqWire::OP_ACQUIRE_ENTRY,
            ImageName::empty(),
            0,
            ImageSourceMetadata::empty(),
        ),
        ImageRequest::Describe { name } => (
            ImageReqWire::OP_DESCRIBE,
            name,
            0,
            ImageSourceMetadata::empty(),
        ),
        ImageRequest::Export { name } => (
            ImageReqWire::OP_EXPORT,
            name,
            0,
            ImageSourceMetadata::empty(),
        ),
        ImageRequest::Retire { name } => (
            ImageReqWire::OP_RETIRE,
            name,
            0,
            ImageSourceMetadata::empty(),
        ),
        // Mint validates the face's first executable while it owns the image
        // lookup, so discovery and face construction cross only once.
        ImageRequest::FaceMint { target } => (
            ImageReqWire::OP_FACE_MINT,
            target,
            0,
            ImageSourceMetadata::empty(),
        ),
        ImageRequest::FaceFork { template } => (
            ImageReqWire::OP_FACE_FORK,
            ImageName::empty(),
            template,
            ImageSourceMetadata::empty(),
        ),
        ImageRequest::FaceBank { face } => (
            ImageReqWire::OP_FACE_BANK,
            ImageName::empty(),
            face,
            ImageSourceMetadata::empty(),
        ),
        ImageRequest::FaceRetire { face } => (
            ImageReqWire::OP_FACE_RETIRE,
            ImageName::empty(),
            face,
            ImageSourceMetadata::empty(),
        ),
        ImageRequest::RetireBound { template } => (
            ImageReqWire::OP_RETIRE_BOUND,
            ImageName::empty(),
            template,
            ImageSourceMetadata::empty(),
        ),
    };
    ImageReqWire {
        op,
        reserved: [0; 7],
        name,
        slot: slot.into(),
        reserved2: 0.into(),
        reserved3: 0.into(),
        source,
    }
}

/// Lift a wire request — fail-closed on the op; the *name* stays raw
/// bytes (the serve step's `as_str` gate is the semantic refusal,
/// `BadName` — the codec refuses shapes, not content).  The face verbs
/// require the name empty: a byte there acquires no accidental meaning.
pub fn decode_req(w: &ImageReqWire) -> Option<ImageRequest> {
    // Tombstones outrank every shape rule (the TaskReqWire discipline): the
    // retired name/ordinal acquisition family answers a typed refusal no
    // matter what payload rides with it, so a stale caller is denied rather
    // than dropped.
    if ImageReqWire::op_retired(w.op) {
        return Some(ImageRequest::Retired);
    }
    if w.reserved != [0; 7] || w.reserved2.get() != 0 || w.reserved3.get() != 0 {
        return None;
    }
    let nameless = w.name == ImageName::empty();
    let sourceless = w.source == ImageSourceMetadata::empty();
    Some(match w.op {
        ImageReqWire::OP_REGISTER => ImageRequest::Register {
            name: w.name,
            slot: SlotRef::from_wire(w.slot.get()),
            source: w.source,
        },
        ImageReqWire::OP_ACQUIRE if w.slot.get() == 0 && sourceless => {
            ImageRequest::Acquire { name: w.name }
        }
        ImageReqWire::OP_ACQUIRE_ENTRY if nameless && w.slot.get() == 0 && sourceless => {
            ImageRequest::AcquireEntry
        }
        ImageReqWire::OP_DESCRIBE if w.slot.get() == 0 && sourceless => {
            ImageRequest::Describe { name: w.name }
        }
        ImageReqWire::OP_EXPORT if w.slot.get() == 0 && sourceless => {
            ImageRequest::Export { name: w.name }
        }
        ImageReqWire::OP_RETIRE if w.slot.get() == 0 && sourceless => {
            ImageRequest::Retire { name: w.name }
        }
        ImageReqWire::OP_FACE_MINT if w.slot.get() == 0 && sourceless => {
            ImageRequest::FaceMint { target: w.name }
        }
        ImageReqWire::OP_FACE_FORK if nameless && sourceless => ImageRequest::FaceFork {
            template: w.slot.get(),
        },
        ImageReqWire::OP_FACE_BANK if nameless && sourceless => {
            ImageRequest::FaceBank { face: w.slot.get() }
        }
        ImageReqWire::OP_FACE_RETIRE if nameless && sourceless => {
            ImageRequest::FaceRetire { face: w.slot.get() }
        }
        ImageReqWire::OP_RETIRE_BOUND if nameless && sourceless => ImageRequest::RetireBound {
            template: w.slot.get(),
        },
        _ => return None,
    })
}

/// Lower a response + the effected grant count (total).
pub fn encode_rep(rep: &ImageResponse, grants: u8) -> ImageRepWire {
    let (tag, refusal, face, generation, module) = match *rep {
        ImageResponse::Registered { module } => (ImageRepWire::TAG_REGISTERED, 0, 0, 0, module),
        ImageResponse::Acquired { module } => (ImageRepWire::TAG_ACQUIRED, 0, 0, 0, module),
        ImageResponse::Described { generation, module } => {
            (ImageRepWire::TAG_DESCRIBED, 0, 0, generation, module)
        }
        ImageResponse::Exported { len } => (
            ImageRepWire::TAG_EXPORTED,
            0,
            0,
            len,
            ModuleDescriptor::empty(),
        ),
        ImageResponse::Retired => (
            ImageRepWire::TAG_RETIRED,
            0,
            0,
            0,
            ModuleDescriptor::empty(),
        ),
        ImageResponse::Face {
            face,
            generation,
            module,
        } => (ImageRepWire::TAG_FACE, 0, face, generation, module),
        ImageResponse::Banked => (ImageRepWire::TAG_BANKED, 0, 0, 0, ModuleDescriptor::empty()),
        ImageResponse::Refused(r) => (
            ImageRepWire::TAG_REFUSED,
            r as u8, // ImageRefusal's own 1-based discriminants ride verbatim
            0,
            0,
            ModuleDescriptor::empty(),
        ),
    };
    ImageRepWire {
        tag,
        refusal,
        grants,
        reserved: 0,
        face: face.into(),
        generation: generation.into(),
        module,
    }
}

/// Lift a wire reply + its grant count — fail-closed (unknown tags and
/// refusal bytes refuse; a data-bearing reply whose descriptor fails
/// [`validate`](ModuleDescriptor::validate) refuses too — a hostile or
/// stale descriptor never reaches a consumer as a promise; a face id on
/// anything but [`TAG_FACE`](ImageRepWire::TAG_FACE) refuses likewise).
pub fn decode_rep(w: &ImageRepWire) -> Option<(ImageResponse, u8)> {
    if w.reserved != 0
        || (w.tag != ImageRepWire::TAG_FACE && w.face.get() != 0)
        || (!matches!(
            w.tag,
            ImageRepWire::TAG_FACE | ImageRepWire::TAG_DESCRIBED | ImageRepWire::TAG_EXPORTED
        ) && w.generation.get() != 0)
    {
        return None;
    }
    let rep = match (w.tag, w.refusal) {
        (ImageRepWire::TAG_REGISTERED, 0) => {
            if w.grants != 0 || !w.module.validate() {
                return None;
            }
            ImageResponse::Registered { module: w.module }
        }
        (ImageRepWire::TAG_ACQUIRED, 0) => {
            if !w.module.validate()
                || usize::from(w.grants) > crate::MAX_IMAGE_SEGS
                || w.grants != w.module.seg_count
            {
                return None;
            }
            ImageResponse::Acquired { module: w.module }
        }
        (ImageRepWire::TAG_DESCRIBED, 0) => {
            if w.grants != 0 || w.generation.get() == 0 || !w.module.validate() {
                return None;
            }
            ImageResponse::Described {
                generation: w.generation.get(),
                module: w.module,
            }
        }
        (ImageRepWire::TAG_EXPORTED, 0) => {
            // Exactly one member-window lease rides an export, and the
            // record carries no descriptor: bytes, never a module promise.
            if w.grants != 1 || w.generation.get() == 0 || w.module != ModuleDescriptor::empty() {
                return None;
            }
            ImageResponse::Exported {
                len: w.generation.get(),
            }
        }
        (ImageRepWire::TAG_RETIRED, 0)
            if w.grants == 0 && w.module == ModuleDescriptor::empty() =>
        {
            ImageResponse::Retired
        }
        (ImageRepWire::TAG_FACE, 0)
            if usize::from(w.grants) == kore_rt::service::MAX_GRANTS
                && w.generation.get() != 0
                && w.module.validate() =>
        {
            ImageResponse::Face {
                face: w.face.get(),
                generation: w.generation.get(),
                module: w.module,
            }
        }
        (ImageRepWire::TAG_BANKED, 0)
            if usize::from(w.grants) == crate::MAX_IMAGE_SEGS
                && w.module == ModuleDescriptor::empty() =>
        {
            ImageResponse::Banked
        }
        (ImageRepWire::TAG_REFUSED, r)
            if w.grants == 0 && w.module == ModuleDescriptor::empty() =>
        {
            ImageResponse::Refused(match r {
                x if x == ImageRefusal::BadImage as u8 => ImageRefusal::BadImage,
                x if x == ImageRefusal::Unknown as u8 => ImageRefusal::Unknown,
                x if x == ImageRefusal::TableFull as u8 => ImageRefusal::TableFull,
                x if x == ImageRefusal::BadName as u8 => ImageRefusal::BadName,
                x if x == ImageRefusal::Denied as u8 => ImageRefusal::Denied,
                x if x == ImageRefusal::Retired as u8 => ImageRefusal::Retired,
                _ => return None,
            })
        }
        _ => return None,
    };
    Some((rep, w.grants))
}

impl SvcProtocol for ImageService {
    type Request = ImageRequest;
    type Response = ImageResponse;
    type ReqWire = ImageReqWire;
    type RepWire = ImageRepWire;

    /// The widest image outcome is one minted face: its ring sub-lease plus
    /// [`MAX_IMAGE_SEGS`](crate::MAX_IMAGE_SEGS) recv endpoints. Ordinary
    /// `Acquire` requests retain their nine-channel shape through
    /// [`request_grants`](Self::request_grants).
    const REPLY_GRANTS: usize = kore_rt::service::MAX_GRANTS;

    fn request_grants(request: &ImageReqWire) -> usize {
        match request.op {
            ImageReqWire::OP_FACE_MINT | ImageReqWire::OP_FACE_FORK => kore_rt::service::MAX_GRANTS,
            ImageReqWire::OP_ACQUIRE
            | ImageReqWire::OP_ACQUIRE_ENTRY
            | ImageReqWire::OP_FACE_BANK => crate::MAX_IMAGE_SEGS,
            // One member-window lease, and the receive for it must stand
            // before the request is visible: this count *is* the client's
            // post, so a zero here would leave the export's only payload
            // with nowhere to land while `reply_matches` still demanded it.
            ImageReqWire::OP_EXPORT => 1,
            _ => 0,
        }
    }

    /// Two serving pairs: pair 0 is the founder-era pair every row gets;
    /// pair 1 is exec's resident **loader face**.  Exec resolves linked
    /// target metadata there and acquires the standing startup stub; spawned
    /// tasks resolve their executable and every `DT_NEEDED` library through
    /// per-child minted faces (`res/IMAGE.md` §4).  Pair 1 is post-less by
    /// construction, so it can *acquire* but structurally cannot `Register`.
    /// The first shell no longer owns a duplicate image face.
    const SERVE_PAIRS: usize = 2;

    /// Untyped: two kinds lawfully cross the same channels — segment
    /// sub-leases (`Acquired`) and fresh grant-channel recv ends
    /// (`Banked`, slice C) — so no single `(kind, rights)` post gate
    /// fits.  Each taker kind-checks at possession instead
    /// ([`try_into_stashed_lease`](kore_rt::cap::ReturnedCapability::try_into_stashed_lease)
    /// / [`try_into_chan_recv`](kore_rt::cap::ReturnedCapability::try_into_chan_recv)),
    /// the mm precedent: the fail-closed moment moves from the post to
    /// the claim, and a mismatched delivery still refuses before use.
    fn grant_policy(_index: usize) -> Option<(kore_abi::cap::ResourceKind, kore_abi::cap::Rights)> {
        None
    }

    fn retain_grant(index: usize) -> bool {
        index < Self::REPLY_GRANTS
    }

    fn reply_matches(request: &ImageReqWire, response: &ImageResponse, grants: u8) -> bool {
        match (request.op, response) {
            (ImageReqWire::OP_REGISTER, ImageResponse::Registered { .. })
            | (ImageReqWire::OP_REGISTER, ImageResponse::Refused(_))
            | (ImageReqWire::OP_RETIRE, ImageResponse::Retired)
            | (ImageReqWire::OP_RETIRE, ImageResponse::Refused(_)) => grants == 0,
            (ImageReqWire::OP_ACQUIRE, ImageResponse::Acquired { module }) => {
                grants == module.seg_count
            }
            (ImageReqWire::OP_ACQUIRE_ENTRY, ImageResponse::Acquired { module }) => {
                grants == module.seg_count
            }
            (ImageReqWire::OP_ACQUIRE, ImageResponse::Refused(_)) => grants == 0,
            (ImageReqWire::OP_ACQUIRE_ENTRY, ImageResponse::Refused(_)) => grants == 0,
            (ImageReqWire::OP_DESCRIBE, ImageResponse::Described { .. })
            | (ImageReqWire::OP_DESCRIBE, ImageResponse::Refused(_))
            | (ImageReqWire::OP_EXPORT, ImageResponse::Refused(_)) => grants == 0,
            // The member-window lease *is* the export; a grantless one
            // conveys nothing and must not correlate.
            (ImageReqWire::OP_EXPORT, ImageResponse::Exported { .. }) => grants == 1,
            (
                ImageReqWire::OP_FACE_MINT | ImageReqWire::OP_FACE_FORK,
                ImageResponse::Face { .. },
            ) => usize::from(grants) == kore_rt::service::MAX_GRANTS,
            (ImageReqWire::OP_FACE_BANK, ImageResponse::Banked) => {
                usize::from(grants) == crate::MAX_IMAGE_SEGS
            }
            (ImageReqWire::OP_FACE_RETIRE, ImageResponse::Retired) => grants == 0,
            (ImageReqWire::OP_RETIRE_BOUND, ImageResponse::Retired) => grants == 0,
            (
                ImageReqWire::OP_FACE_MINT | ImageReqWire::OP_FACE_FORK,
                ImageResponse::Refused(_),
            )
            | (ImageReqWire::OP_FACE_BANK, ImageResponse::Refused(_))
            | (ImageReqWire::OP_FACE_RETIRE, ImageResponse::Refused(_))
            | (ImageReqWire::OP_RETIRE_BOUND, ImageResponse::Refused(_)) => grants == 0,
            _ => false,
        }
    }

    fn encode_req(req: &ImageRequest) -> ImageReqWire {
        encode_req(req)
    }
    fn decode_req(wire: &ImageReqWire) -> Option<ImageRequest> {
        decode_req(wire)
    }
    fn encode_rep(rep: &ImageResponse, grants: u8) -> ImageRepWire {
        encode_rep(rep, grants)
    }
    fn decode_rep(wire: &ImageRepWire) -> Option<(ImageResponse, u8)> {
        decode_rep(wire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kore_abi::image::{IMAGE_ABI, ImageName, ImageSourceMetadata, kind, seg};
    use zerocopy::IntoBytes;

    fn name(s: &str) -> ImageName {
        ImageName::new(s).unwrap()
    }

    fn source() -> ImageSourceMetadata {
        ImageSourceMetadata {
            source: 1.into(),
            version: 2.into(),
            len: 3.into(),
            digest: [4; 32],
            profile: 0.into(),
            reserved: 0.into(),
        }
    }

    /// A minimal descriptor that passes `validate` (the codec's gate).
    fn module() -> ModuleDescriptor {
        let mut m = ModuleDescriptor::empty();
        m.abi = IMAGE_ABI.into();
        m.kind = kind::LIBRARY;
        m.seg_count = 1;
        m.segs[0].kind = seg::KIND_TEXT;
        m.segs[0].access = seg::ACCESS_R | seg::ACCESS_X;
        m.segs[0].memsz = 4096.into();
        m
    }

    /// The op a request lowers to. Exhaustive on purpose: a new request
    /// variant cannot be added without naming it here, which is the
    /// prompt to add its sample to `round_trips` below. `Export` shipped
    /// encodable and undecodable precisely because nothing forced that
    /// pairing — the service dropped the record as unparseable, replied
    /// nothing, and the seeding client waited for an answer that could
    /// never come.
    fn lowered_op(req: &ImageRequest) -> u8 {
        match *req {
            ImageRequest::Register { .. } => ImageReqWire::OP_REGISTER,
            ImageRequest::Retired => ImageReqWire::OP_EXECUTABLE_AT,
            ImageRequest::Acquire { .. } => ImageReqWire::OP_ACQUIRE,
            ImageRequest::AcquireEntry => ImageReqWire::OP_ACQUIRE_ENTRY,
            ImageRequest::Describe { .. } => ImageReqWire::OP_DESCRIBE,
            ImageRequest::Export { .. } => ImageReqWire::OP_EXPORT,
            ImageRequest::Retire { .. } => ImageReqWire::OP_RETIRE,
            ImageRequest::FaceMint { .. } => ImageReqWire::OP_FACE_MINT,
            ImageRequest::FaceFork { .. } => ImageReqWire::OP_FACE_FORK,
            ImageRequest::FaceBank { .. } => ImageReqWire::OP_FACE_BANK,
            ImageRequest::FaceRetire { .. } => ImageReqWire::OP_FACE_RETIRE,
            ImageRequest::RetireBound { .. } => ImageReqWire::OP_RETIRE_BOUND,
        }
    }

    /// Every request/response round-trips through its wire form.
    #[test]
    fn round_trips() {
        for req in [
            ImageRequest::Register {
                name: name("libdemo.so"),
                slot: SlotRef::from_wire(11),
                source: source(),
            },
            ImageRequest::Retired,
            ImageRequest::Acquire {
                name: name("libdemo.so"),
            },
            ImageRequest::AcquireEntry,
            ImageRequest::Describe {
                name: name("libdemo.so"),
            },
            ImageRequest::Export {
                name: name("libdemo.so"),
            },
            ImageRequest::Retire {
                name: name("libdemo.so"),
            },
            ImageRequest::FaceMint {
                target: name("hello"),
            },
            ImageRequest::FaceFork { template: 5 },
            ImageRequest::FaceBank { face: 3 },
            ImageRequest::FaceRetire { face: 7 },
            ImageRequest::RetireBound { template: 9 },
        ] {
            let wire = encode_req(&req);
            assert_eq!(wire.op, lowered_op(&req));
            let back = decode_req(&wire).expect("decodes");
            assert_eq!(encode_req(&back).as_bytes(), wire.as_bytes());
        }
        for (rep, grants) in [
            (ImageResponse::Registered { module: module() }, 0u8),
            (ImageResponse::Acquired { module: module() }, 1),
            (
                ImageResponse::Described {
                    generation: 1,
                    module: module(),
                },
                0,
            ),
            (ImageResponse::Retired, 0),
            (
                ImageResponse::Face {
                    face: 5,
                    generation: 1,
                    module: module(),
                },
                kore_rt::service::MAX_GRANTS as u8,
            ),
            (ImageResponse::Banked, crate::MAX_IMAGE_SEGS as u8),
            (ImageResponse::Refused(ImageRefusal::BadImage), 0),
            (ImageResponse::Refused(ImageRefusal::Unknown), 0),
            (ImageResponse::Refused(ImageRefusal::TableFull), 0),
            (ImageResponse::Refused(ImageRefusal::BadName), 0),
            (ImageResponse::Refused(ImageRefusal::Denied), 0),
        ] {
            let (back, g) = decode_rep(&encode_rep(&rep, grants)).expect("decodes");
            assert_eq!(back, rep);
            assert_eq!(g, grants);
        }
    }

    /// Decode is fail-closed: unknown ops/tags/refusals refuse, and a
    /// data-bearing reply with an invalid descriptor refuses — never a
    /// half-promise.
    #[test]
    fn decode_fails_closed() {
        let mut w = encode_req(&ImageRequest::Acquire { name: name("x") });
        w.op = 0;
        assert!(decode_req(&w).is_none());
        w.op = 200;
        assert!(decode_req(&w).is_none());
        let mut w = encode_req(&ImageRequest::Acquire { name: name("x") });
        w.reserved[0] = 1;
        assert!(decode_req(&w).is_none(), "request reserved bytes refuse");
        let mut w = encode_req(&ImageRequest::Retire { name: name("x") });
        w.slot = 1.into();
        assert!(decode_req(&w).is_none(), "data-only ops carry no slot");
        let mut w = encode_req(&ImageRequest::Describe { name: name("x") });
        w.slot = 1.into();
        assert!(decode_req(&w).is_none(), "Describe carries no slot");
        let mut w = encode_req(&ImageRequest::FaceBank { face: 1 });
        w.name = name("x");
        assert!(decode_req(&w).is_none(), "face verbs are name-less");
        let mut w = encode_req(&ImageRequest::FaceRetire { face: 1 });
        w.name = name("x");
        assert!(decode_req(&w).is_none(), "face verbs are name-less");
        let mut w = encode_req(&ImageRequest::FaceMint {
            target: name("hello"),
        });
        w.slot = 1.into();
        assert!(decode_req(&w).is_none(), "mint names no face");

        let mut r = encode_rep(&ImageResponse::Retired, 0);
        r.tag = 0;
        assert!(decode_rep(&r).is_none());
        let mut r = encode_rep(&ImageResponse::Refused(ImageRefusal::Denied), 0);
        r.refusal = 77;
        assert!(decode_rep(&r).is_none());
        let mut r = encode_rep(&ImageResponse::Retired, 0);
        r.refusal = 1;
        assert!(
            decode_rep(&r).is_none(),
            "a refusal byte on success refuses"
        );
        let mut r = encode_rep(&ImageResponse::Retired, 0);
        r.reserved = 1;
        assert!(decode_rep(&r).is_none(), "reply reserved bytes refuse");
        let mut r = encode_rep(&ImageResponse::Refused(ImageRefusal::Denied), 0);
        r.module = module();
        assert!(
            decode_rep(&r).is_none(),
            "data-less replies carry the empty descriptor"
        );
        let mut r = encode_rep(&ImageResponse::Retired, 0);
        r.face = 3.into();
        assert!(
            decode_rep(&r).is_none(),
            "a face id on anything but TAG_FACE refuses"
        );
        let mut r = encode_rep(&ImageResponse::Acquired { module: module() }, 1);
        r.face = 3.into();
        assert!(
            decode_rep(&r).is_none(),
            "a face id on anything but TAG_FACE refuses"
        );

        // An Acquired reply whose descriptor is unstamped/corrupt refuses.
        let mut r = encode_rep(&ImageResponse::Acquired { module: module() }, 1);
        r.module.abi = 0.into();
        assert!(decode_rep(&r).is_none(), "invalid descriptor refuses");
        let mut r = encode_rep(
            &ImageResponse::Described {
                generation: 1,
                module: module(),
            },
            0,
        );
        r.module.abi = 0.into();
        assert!(decode_rep(&r).is_none(), "invalid description refuses");

        for grants in [0, 2, u8::MAX] {
            let r = encode_rep(&ImageResponse::Acquired { module: module() }, grants);
            assert!(
                decode_rep(&r).is_none(),
                "Acquired must carry exactly seg_count grants"
            );
        }
        for grants in [0, 1, kore_rt::service::MAX_GRANTS as u8 - 1, u8::MAX] {
            let r = encode_rep(
                &ImageResponse::Face {
                    face: 1,
                    generation: 1,
                    module: module(),
                },
                grants,
            );
            assert!(
                decode_rep(&r).is_none(),
                "Face carries the complete client authority"
            );
        }
        for grants in [0, 1, crate::MAX_IMAGE_SEGS as u8 - 1, u8::MAX] {
            let r = encode_rep(&ImageResponse::Banked, grants);
            assert!(
                decode_rep(&r).is_none(),
                "Banked carries exactly MAX_IMAGE_SEGS grants"
            );
        }
        for rep in [
            ImageResponse::Registered { module: module() },
            ImageResponse::Described {
                generation: 1,
                module: module(),
            },
            ImageResponse::Retired,
            ImageResponse::Refused(ImageRefusal::Denied),
        ] {
            let r = encode_rep(&rep, 1);
            assert!(decode_rep(&r).is_none(), "grant-less replies grant none");
        }
    }

    /// An export carries its byte length in `generation`, which the header
    /// guard admits for exactly the tags that use it. This shipped decoding
    /// to `None` — the request encoded, the service answered, and the reply
    /// evaporated at the client — so the round trip is pinned here rather
    /// than left to the far end of a boot.
    #[test]
    fn an_export_reply_survives_the_round_trip_with_its_length() {
        let request = encode_req(&ImageRequest::Export { name: name("ls") });
        // The service must be able to read the request at all. A record the
        // decoder refuses is dropped unanswered by the serve loop, which is
        // indistinguishable from a hung service at the caller.
        assert!(matches!(
            decode_req(&request),
            Some(ImageRequest::Export { name: back }) if back == name("ls")
        ));
        // …and the single lease it answers with needs a receive standing
        // before the request is even visible. `request_grants` is what
        // stands it, so it has to agree with `reply_matches` below.
        assert_eq!(ImageService::request_grants(&request), 1);
        let reply = encode_rep(&ImageResponse::Exported { len: 4096 }, 1);
        let (decoded, grants) = decode_rep(&reply).expect("an export reply decodes");
        assert_eq!(decoded, ImageResponse::Exported { len: 4096 });
        assert_eq!(grants, 1);
        assert!(ImageService::reply_matches(&request, &decoded, 1));

        // The lease is the payload: an export claiming none is not an export.
        let graceless = encode_rep(&ImageResponse::Exported { len: 4096 }, 0);
        assert!(decode_rep(&graceless).is_none());
        // A zero length would name an empty member — refuse rather than
        // seed an empty file over a real executable.
        let empty = encode_rep(&ImageResponse::Exported { len: 0 }, 1);
        assert!(decode_rep(&empty).is_none());
    }

    #[test]
    fn reply_is_correlated_to_the_outstanding_operation() {
        let register = encode_req(&ImageRequest::Register {
            name: name("x"),
            slot: SlotRef::from_wire(7),
            source: source(),
        });
        let acquire = encode_req(&ImageRequest::Acquire { name: name("x") });
        let describe = encode_req(&ImageRequest::Describe { name: name("x") });
        let retire = encode_req(&ImageRequest::Retire { name: name("x") });
        let registered = ImageResponse::Registered { module: module() };
        let acquired = ImageResponse::Acquired { module: module() };
        let described = ImageResponse::Described {
            generation: 1,
            module: module(),
        };

        assert!(ImageService::reply_matches(&register, &registered, 0));
        assert!(ImageService::reply_matches(&acquire, &acquired, 1));
        assert!(ImageService::reply_matches(&describe, &described, 0));
        assert!(ImageService::reply_matches(
            &retire,
            &ImageResponse::Retired,
            0
        ));

        let mint = encode_req(&ImageRequest::FaceMint {
            target: name("hello"),
        });
        let bank = encode_req(&ImageRequest::FaceBank { face: 2 });
        let face_retire = encode_req(&ImageRequest::FaceRetire { face: 2 });
        let face = ImageResponse::Face {
            face: 2,
            generation: 1,
            module: module(),
        };
        let face_grants = kore_rt::service::MAX_GRANTS as u8;
        let bank_grants = crate::MAX_IMAGE_SEGS as u8;
        assert!(ImageService::reply_matches(&mint, &face, face_grants));
        assert!(!ImageService::reply_matches(&mint, &face, 0));
        assert!(ImageService::reply_matches(
            &bank,
            &ImageResponse::Banked,
            bank_grants
        ));
        assert!(!ImageService::reply_matches(
            &bank,
            &ImageResponse::Banked,
            1
        ));
        assert!(ImageService::reply_matches(
            &face_retire,
            &ImageResponse::Retired,
            0
        ));

        for request in [
            &register,
            &acquire,
            &describe,
            &retire,
            &mint,
            &bank,
            &face_retire,
        ] {
            assert!(ImageService::reply_matches(
                request,
                &ImageResponse::Refused(ImageRefusal::Denied),
                0
            ));
        }

        assert!(!ImageService::reply_matches(&register, &acquired, 1));
        assert!(!ImageService::reply_matches(&acquire, &registered, 0));
        assert!(!ImageService::reply_matches(&describe, &acquired, 1));
        assert!(!ImageService::reply_matches(&acquire, &described, 0));
        assert!(!ImageService::reply_matches(&retire, &registered, 0));
        assert!(!ImageService::reply_matches(
            &mint,
            &ImageResponse::Banked,
            bank_grants
        ));
        assert!(!ImageService::reply_matches(&bank, &face, face_grants));
        assert!(!ImageService::reply_matches(&acquire, &face, face_grants));

        assert_eq!(ImageService::request_grants(&mint), face_grants as usize);
        assert_eq!(
            ImageService::request_grants(&acquire),
            crate::MAX_IMAGE_SEGS
        );
        assert_eq!(ImageService::request_grants(&describe), 0);
    }
}
