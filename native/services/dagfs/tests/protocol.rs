//! The plan's §10.2 protocol tests: every semantic variant round-trips through
//! its frozen wire form, and every malformed shape is refused.
//!
//! These test the codec as an adversary would reach it — by writing bytes into
//! a record and asking the service to interpret them. A decode that accepted
//! any of the refusals below would be a hole in the only thing standing between
//! the engine and another process.

mod common;

use common::name;
use kore_abi::session::SlotRef;
use kore_abi::svc::{FS_NAME_BYTES, FsRepWire, FsReqWire};
use kore_rt::svc::{PeerBundlePart, PeerId, SvcProtocol};
use kore_svc_dagfs::{
    ANY_REVISION, AuthorityQuota, FsError, FsRequest, FsResponse, GrantId, Handle, ObjectId,
    ObjectKind, ProviderError, RamService, Rights, SessionId, VersionLease,
    wire::{decode_rep_for, decode_req, encode_rep, encode_req, reply_matches},
};
use zerocopy::IntoBytes;

/// One of every request variant.
fn every_request() -> Vec<FsRequest> {
    let h = Handle::from_bits(0x0000_0003_0000_0007);
    let h2 = Handle::from_bits(0x0000_0009_0000_0002);
    let g = GrantId::from_bits(0x0000_0005_0000_0001);
    let peer = PeerId::from_arrival(0x101).expect("valid peer generation");
    vec![
        FsRequest::AttachWindow {
            slot: 9,
            bytes: 4096,
        },
        FsRequest::OpenRoot {
            rights: Rights::ALL,
        },
        FsRequest::OpenAt {
            parent: h,
            name: name("child"),
            rights: Rights::READ,
        },
        FsRequest::OpenGrant {
            grant: g,
            rights: Rights::READ_ONLY,
        },
        FsRequest::Close { handle: h },
        FsRequest::Stat { handle: h },
        FsRequest::List {
            handle: h,
            cursor: 12,
            window_offset: 0,
        },
        FsRequest::Create {
            parent: h,
            name: name("made"),
            kind: ObjectKind::File,
            subtype: kore_svc_dagfs::ObjectSubtype::PLAIN,
        },
        FsRequest::Create {
            parent: h,
            name: name("dir"),
            kind: ObjectKind::Directory,
            subtype: kore_svc_dagfs::ObjectSubtype::PLAIN,
        },
        FsRequest::Read {
            handle: h,
            offset: 4096,
            window_offset: 64,
            length: 512,
        },
        FsRequest::WriteAt {
            handle: h,
            offset: 8,
            window_offset: 128,
            length: 256,
            expected: 42,
        },
        FsRequest::Append {
            handle: h,
            window_offset: 32,
            length: 16,
            expected: ANY_REVISION,
        },
        FsRequest::Resize {
            handle: h,
            length: 1024,
            expected: 7,
        },
        FsRequest::Rename {
            parent: h,
            old: name("before"),
            new: name("after"),
        },
        FsRequest::Relocate {
            src_parent: h,
            name: name("moving"),
            dst_parent: h2,
            new_name: name("moved"),
        },
        FsRequest::Link {
            parent: h,
            name: name("second"),
            child: h2,
        },
        FsRequest::Unlink {
            parent: h,
            name: name("gone"),
        },
        FsRequest::DetachEdge {
            parent: h,
            name: name("detached"),
        },
        FsRequest::Grant {
            handle: h,
            destination: SessionId::new(2),
            rights: Rights::READ,
            display_name: Some(name("inbox")),
        },
        FsRequest::Grant {
            handle: h,
            destination: SessionId::new(1),
            rights: Rights::READ.union(Rights::WRITE),
            display_name: None,
        },
        FsRequest::GrantScoped {
            handle: h,
            destination: SessionId::new(2),
            rights: Rights::READ.union(Rights::WRITE),
            quota: AuthorityQuota::new(8, 12, 4096, 4),
            display_name: Some(name("data")),
        },
        FsRequest::Revoke { grant: g },
        FsRequest::Sync { handle: h },
        FsRequest::LendVersion {
            handle: h,
            window_offset: 64,
        },
        FsRequest::LendExecutable {
            handle: h,
            window_offset: 64,
        },
        FsRequest::ReadVersion {
            lease: VersionLease::from_bits(0x0000_0002_0000_0003),
            offset: 8,
            window_offset: 96,
            length: 128,
        },
        FsRequest::CommitDerived {
            lease: VersionLease::from_bits(0x0000_0002_0000_0003),
            window_offset: 32,
            length: 256,
            output_window_bytes: 4096,
        },
        FsRequest::ReleaseVersion {
            lease: VersionLease::from_bits(0x0000_0002_0000_0003),
        },
        FsRequest::DerivedInfo {
            handle: h,
            parser: kore_svc_extract::ParserId(1),
        },
        FsRequest::RegisterSystemView {
            rights: Rights::READ_ONLY,
            announced: SlotRef::from_wire(34),
        },
        FsRequest::ApplyView {
            peer,
            rights: Rights::READ_ONLY,
            announced: SlotRef::from_wire(35),
        },
        FsRequest::BindView {
            handle: h,
            rights: Rights::READ_ONLY,
            mode: kore_dagfs::ViewBindingMode::Inherited,
            announced: SlotRef::from_wire(36),
        },
        FsRequest::BindView {
            handle: h,
            rights: Rights::READ_ONLY,
            mode: kore_dagfs::ViewBindingMode::Isolated,
            announced: SlotRef::from_wire(36),
        },
        FsRequest::BindView {
            handle: h,
            rights: Rights::READ_ONLY,
            mode: kore_dagfs::ViewBindingMode::Resources(kore_dagfs::Name::new("input").unwrap()),
            announced: SlotRef::from_wire(36),
        },
        FsRequest::AdmitPeer {
            part: PeerBundlePart::Ring,
            announced: SlotRef::from_wire(37),
        },
        FsRequest::AdmitPeer {
            part: PeerBundlePart::Kick,
            announced: SlotRef::from_wire(38),
        },
        FsRequest::AdmitPeer {
            part: PeerBundlePart::ReturnRecv,
            announced: SlotRef::from_wire(39),
        },
        FsRequest::AdmitPeer {
            part: PeerBundlePart::Grant(2),
            announced: SlotRef::from_wire(40),
        },
        FsRequest::TakeAdmission,
        FsRequest::RetirePeer { peer },
        FsRequest::ConfirmRetired { peer },
        FsRequest::ClosePeer,
    ]
}

#[test]
fn async_protocol_decode_uses_the_outstanding_request() {
    let request = FsRequest::OpenRoot {
        rights: Rights::READ,
    };
    let request_wire = encode_req(&request);
    let response = FsResponse::Opened {
        handle: Handle::from_bits(0x0000_0001_0000_0002),
        stable_object: kore_svc_dagfs::StableObjectId::new(41).unwrap(),
        kind: ObjectKind::Directory,
        subtype: kore_svc_dagfs::ObjectSubtype::PLAIN,
        len: 0,
        revision: 1,
    };
    let response_wire = encode_rep(&response, 0);

    assert!(
        <RamService as SvcProtocol>::decode_rep(&response_wire).is_none(),
        "an Opened record is intentionally ambiguous without its request"
    );
    assert_eq!(
        <RamService as SvcProtocol>::decode_rep_for(&request_wire, &response_wire),
        Some((response, 0)),
        "the async metal client must decode against its outstanding request"
    );
}

#[test]
fn every_request_variant_round_trips() {
    for req in every_request() {
        let wire = encode_req(&req);
        let back = decode_req(&wire).unwrap_or_else(|| panic!("{req:?} must decode"));
        assert_eq!(back, req, "{req:?} must survive the wire unchanged");
    }
}

#[test]
fn system_view_registration_requires_an_explicit_valid_attenuation() {
    let request = FsRequest::RegisterSystemView {
        rights: Rights::READ_ONLY,
        announced: SlotRef::from_wire(34),
    };
    let wire = encode_req(&request);
    assert_eq!(decode_req(&wire), Some(request));

    let mut empty = wire;
    empty.rights = 0;
    assert!(decode_req(&empty).is_none());

    let mut unknown = wire;
    unknown.rights = 1 << 7;
    assert!(decode_req(&unknown).is_none());
}

#[test]
fn every_response_variant_round_trips_against_its_request() {
    let h = Handle::from_bits(0x0000_0003_0000_0007);
    let g = GrantId::from_bits(0x0000_0005_0000_0001);
    let peer = PeerId::from_arrival(0x101).expect("valid peer generation");
    let cases: Vec<(FsRequest, FsResponse)> = vec![
        (
            FsRequest::OpenRoot {
                rights: Rights::ALL,
            },
            FsResponse::Opened {
                handle: h,
                stable_object: kore_svc_dagfs::StableObjectId::new(41).unwrap(),
                kind: ObjectKind::Directory,
                subtype: kore_svc_dagfs::ObjectSubtype::PLAIN,
                len: 0,
                revision: 1,
            },
        ),
        (
            FsRequest::Stat { handle: h },
            FsResponse::Stat {
                object: ObjectId::from_bits(0x0000_0002_0000_0004),
                stable_object: kore_svc_dagfs::StableObjectId::new(42).unwrap(),
                kind: ObjectKind::File,
                subtype: kore_svc_dagfs::ObjectSubtype::PLAIN,
                len: 900,
                revision: 5,
                links: 2,
                grants: 1,
            },
        ),
        (
            FsRequest::List {
                handle: h,
                cursor: 0,
                window_offset: 0,
            },
            FsResponse::Listed {
                entries: 8,
                next: Some(9),
            },
        ),
        (
            FsRequest::List {
                handle: h,
                cursor: 9,
                window_offset: 0,
            },
            FsResponse::Listed {
                entries: 3,
                next: None,
            },
        ),
        (
            FsRequest::Read {
                handle: h,
                offset: 0,
                window_offset: 0,
                length: 64,
            },
            FsResponse::Transferred { count: 64 },
        ),
        (
            FsRequest::WriteAt {
                handle: h,
                offset: 0,
                window_offset: 0,
                length: 4,
                expected: ANY_REVISION,
            },
            FsResponse::Committed { revision: 9 },
        ),
        (
            FsRequest::LendVersion {
                handle: h,
                window_offset: 0,
            },
            FsResponse::VersionLent {
                lease: VersionLease::from_bits(0x0000_0002_0000_0003),
            },
        ),
        (
            FsRequest::LendExecutable {
                handle: h,
                window_offset: 0,
            },
            FsResponse::VersionLent {
                lease: VersionLease::from_bits(0x0000_0002_0000_0003),
            },
        ),
        (
            FsRequest::ReadVersion {
                lease: VersionLease::from_bits(0x0000_0002_0000_0003),
                offset: 0,
                window_offset: 0,
                length: 64,
            },
            FsResponse::Transferred { count: 64 },
        ),
        (
            FsRequest::CommitDerived {
                lease: VersionLease::from_bits(0x0000_0002_0000_0003),
                window_offset: 0,
                length: 128,
                output_window_bytes: 4096,
            },
            FsResponse::DerivedCommitted {
                parser: kore_svc_extract::ParserId(1),
                parser_version: 3,
            },
        ),
        (
            FsRequest::DerivedInfo {
                handle: h,
                parser: kore_svc_extract::ParserId(1),
            },
            FsResponse::DerivedInfo {
                input_version: 9,
                parser_version: 3,
            },
        ),
        (
            FsRequest::Append {
                handle: h,
                window_offset: 0,
                length: 4,
                expected: ANY_REVISION,
            },
            FsResponse::Appended {
                offset: 16,
                revision: 10,
            },
        ),
        (
            FsRequest::Grant {
                handle: h,
                destination: SessionId::new(1),
                rights: Rights::READ,
                display_name: None,
            },
            FsResponse::Granted { grant: g },
        ),
        (
            FsRequest::GrantScoped {
                handle: h,
                destination: SessionId::new(1),
                rights: Rights::READ,
                quota: AuthorityQuota::new(4, 4, 512, 2),
                display_name: Some(name("app")),
            },
            FsResponse::Granted { grant: g },
        ),
        (FsRequest::Close { handle: h }, FsResponse::Done),
        (
            FsRequest::AdmitPeer {
                part: PeerBundlePart::Ring,
                announced: SlotRef::from_wire(37),
            },
            FsResponse::Done,
        ),
        (FsRequest::TakeAdmission, FsResponse::PeerAdmitted { peer }),
        (FsRequest::RetirePeer { peer }, FsResponse::Done),
        (FsRequest::ConfirmRetired { peer }, FsResponse::Done),
        (FsRequest::ClosePeer, FsResponse::Done),
    ];

    for (req, rep) in cases {
        let wire = encode_rep(&rep, 0);
        let (back, grants) =
            decode_rep_for(&req, &wire).unwrap_or_else(|| panic!("{rep:?} must decode"));
        assert_eq!(back, rep, "{rep:?} must survive the wire unchanged");
        assert_eq!(grants, 0, "the filesystem grants no kernel authority");
        assert!(reply_matches(&req, &back));
    }
}

#[test]
fn every_refusal_class_round_trips() {
    // The engine's whole refusal vocabulary must cross intact — a class that
    // collapsed into another would make a client's error handling wrong.
    let errors = [
        FsError::NotFound,
        FsError::AlreadyExists,
        FsError::NotDirectory,
        FsError::IsDirectory,
        FsError::InvalidName,
        FsError::InvalidHandle,
        FsError::AccessDenied,
        FsError::AccessLost,
        FsError::Conflict,
        FsError::Cycle,
        FsError::NotEmpty,
        FsError::OutOfObjects,
        FsError::OutOfEdges,
        FsError::OutOfHandles,
        FsError::OutOfStorage,
        FsError::Range,
        FsError::BudgetExceeded,
        FsError::Unsupported,
        FsError::NoWindow,
        FsError::Busy,
        FsError::IndexIncomplete,
        FsError::ProviderFailure(ProviderError::Media),
    ];
    let req = FsRequest::Stat {
        handle: Handle::from_bits(1),
    };
    for err in errors {
        let wire = encode_rep(&FsResponse::Refused(err), 0);
        let (back, _) = decode_rep_for(&req, &wire).expect("a refusal must decode");
        assert_eq!(back, FsResponse::Refused(err), "{err:?} must cross intact");
    }
    // Distinct classes must not share a status byte, or two different refusals
    // would be indistinguishable to a client.
    let mut statuses: Vec<u8> = errors
        .iter()
        .map(|err| encode_rep(&FsResponse::Refused(*err), 0).status)
        .collect();
    statuses.sort();
    let count = statuses.len();
    statuses.dedup();
    assert_eq!(
        statuses.len(),
        count,
        "every refusal class needs its own status"
    );
}

#[test]
fn access_denied_and_access_lost_are_distinct_on_the_wire() {
    // The distinction is the whole of "immediate loss of access"; collapsing it
    // would make plan §2.3 untestable from a client.
    let denied = encode_rep(&FsResponse::Refused(FsError::AccessDenied), 0);
    let lost = encode_rep(&FsResponse::Refused(FsError::AccessLost), 0);
    assert_ne!(denied.status, lost.status);
}

// ── malformed requests ──────────────────────────────────────────────────────

/// A valid `Stat` record to mutate into invalid shapes.
fn valid_req() -> FsReqWire {
    encode_req(&FsRequest::Stat {
        handle: Handle::from_bits(0x0000_0003_0000_0007),
    })
}

#[test]
fn an_unseeded_or_unknown_op_is_refused() {
    let mut w = valid_req();
    w.op = 0;
    assert!(
        decode_req(&w).is_none(),
        "a zero op is unseeded, not a verb"
    );
    for bad in [FsReqWire::OP_MAX + 1, 100, 255] {
        let mut w = valid_req();
        w.op = bad;
        assert!(decode_req(&w).is_none(), "op {bad} must be refused");
    }
}

#[test]
fn a_non_zero_reserved_field_is_refused() {
    for mutate in [
        (|w: &mut FsReqWire| w.reserved0 = 1) as fn(&mut FsReqWire),
        |w: &mut FsReqWire| w.reserved1 = 1.into(),
        |w: &mut FsReqWire| w.reserved2 = 0xffff_ffff.into(),
    ] {
        let mut w = valid_req();
        mutate(&mut w);
        assert!(
            decode_req(&w).is_none(),
            "a reserve exists to be zero; a set one is a peer we do not understand"
        );
    }
}

#[test]
fn an_undefined_rights_bit_is_refused_not_masked() {
    // Masking would answer the request as though the unknown right had not
    // been asked for — fail-open on the one field that bounds authority.
    let mut w = encode_req(&FsRequest::OpenRoot {
        rights: Rights::ALL,
    });
    w.rights = 0xff;
    assert!(decode_req(&w).is_none());

    // And a verb that takes no rights must refuse a non-zero mask rather than
    // ignore it.
    let mut w = valid_req();
    w.rights = Rights::READ.bits();
    assert!(decode_req(&w).is_none());
}

#[test]
fn a_name_length_past_the_field_is_refused() {
    let mut w = encode_req(&FsRequest::Unlink {
        parent: Handle::from_bits(1),
        name: name("x"),
    });
    w.name_len = (FS_NAME_BYTES + 1) as u8;
    assert!(decode_req(&w).is_none());
    w.name_len = 255;
    assert!(decode_req(&w).is_none());
}

#[test]
fn junk_in_a_names_unused_tail_is_refused() {
    let mut w = encode_req(&FsRequest::Unlink {
        parent: Handle::from_bits(1),
        name: name("short"),
    });
    // The tail past `name_len` is padding this service must not interpret.
    w.name[FS_NAME_BYTES - 1] = 0x41;
    assert!(
        decode_req(&w).is_none(),
        "a peer writing in the padding is a peer whose framing differs"
    );
}

#[test]
fn an_invalid_name_is_refused_by_the_codec() {
    let mut w = encode_req(&FsRequest::Unlink {
        parent: Handle::from_bits(1),
        name: name("ok"),
    });
    // `..` is reserved, and a slash can never be a component.
    w.name[..2].copy_from_slice(b"..");
    w.name_len = 2;
    assert!(decode_req(&w).is_none());

    let mut w = encode_req(&FsRequest::Unlink {
        parent: Handle::from_bits(1),
        name: name("ok"),
    });
    w.name[..3].copy_from_slice(b"a/b");
    w.name_len = 3;
    assert!(decode_req(&w).is_none());

    // A zero-length name is invalid for a verb that needs one.
    let mut w = encode_req(&FsRequest::Unlink {
        parent: Handle::from_bits(1),
        name: name("ok"),
    });
    w.name = [0; FS_NAME_BYTES];
    w.name_len = 0;
    assert!(decode_req(&w).is_none());
}

#[test]
fn an_invalid_or_misplaced_kind_is_refused() {
    let mut w = encode_req(&FsRequest::Create {
        parent: Handle::from_bits(1),
        name: name("x"),
        kind: ObjectKind::File,
        subtype: kore_svc_dagfs::ObjectSubtype::PLAIN,
    });
    for bad in [0u8, 3, 255] {
        w.kind = bad;
        assert!(decode_req(&w).is_none(), "kind {bad} must be refused");
    }
    // A verb that creates nothing must not carry a kind.
    let mut w = valid_req();
    w.kind = FsReqWire::KIND_FILE;
    assert!(decode_req(&w).is_none());
}

#[test]
fn peer_control_fields_are_exact_and_fail_closed() {
    let request = FsRequest::AdmitPeer {
        part: PeerBundlePart::ReturnRecv,
        announced: SlotRef::from_wire(37),
    };
    let wire = encode_req(&request);
    assert_eq!(decode_req(&wire), Some(request));

    for mutate in [
        (|w: &mut FsReqWire| w.handle = 1.into()) as fn(&mut FsReqWire),
        |w: &mut FsReqWire| w.length = 1.into(),
        |w: &mut FsReqWire| w.name_len = 1,
        |w: &mut FsReqWire| w.name[0] = b'x',
    ] {
        let mut malformed = wire;
        mutate(&mut malformed);
        assert!(decode_req(&malformed).is_none());
    }

    let mut close = encode_req(&FsRequest::ClosePeer);
    close.window_offset = 1.into();
    assert!(decode_req(&close).is_none());
    let mut close = encode_req(&FsRequest::ClosePeer);
    close.kind = FsReqWire::KIND_PEER_RING;
    assert!(decode_req(&close).is_none());
}

#[test]
fn a_session_number_outside_grant_is_refused() {
    // Only `Grant` routes to a session; elsewhere the field is a claim the
    // caller has no business making.
    let mut w = valid_req();
    w.session = 1.into();
    assert!(decode_req(&w).is_none());
}

#[test]
fn a_grant_may_omit_its_display_name_but_not_smuggle_one() {
    let req = FsRequest::Grant {
        handle: Handle::from_bits(1),
        destination: SessionId::new(1),
        rights: Rights::READ,
        display_name: None,
    };
    let w = encode_req(&req);
    assert_eq!(decode_req(&w), Some(req), "an unnamed grant is legal");

    // Zero length but non-zero bytes is a contradiction.
    let mut w = encode_req(&req);
    w.name[0] = b'x';
    assert!(decode_req(&w).is_none());
}

#[test]
fn a_request_reads_from_arbitrary_bytes_without_panicking() {
    // The record is `FromBytes`, so any 120 bytes are a *layout*-valid record;
    // it is the codec that must refuse them.  This is the fuzz-shaped case:
    // the decode must terminate with a verdict, never panic.
    use zerocopy::FromBytes;
    for fill in [0x00u8, 0x01, 0x41, 0x7f, 0x80, 0xa5, 0xff] {
        let raw = [fill; core::mem::size_of::<FsReqWire>()];
        let w = FsReqWire::read_from_bytes(&raw).expect("layout admits any bytes");
        let _ = decode_req(&w);
    }
}

// ── malformed replies ───────────────────────────────────────────────────────

fn stat_request() -> FsRequest {
    FsRequest::Stat {
        handle: Handle::from_bits(0x0000_0003_0000_0007),
    }
}

#[test]
fn an_unseeded_or_unknown_status_is_refused() {
    let req = stat_request();
    let mut w = encode_rep(&FsResponse::Done, 0);
    w.status = 0;
    assert!(decode_rep_for(&req, &w).is_none());
    for bad in [FsRepWire::STATUS_MAX + 1, 200, 255] {
        let mut w = encode_rep(&FsResponse::Done, 0);
        w.status = bad;
        assert!(decode_rep_for(&req, &w).is_none(), "status {bad} refused");
    }
}

#[test]
fn a_grant_bearing_reply_is_refused() {
    // Nothing this service answers can carry kernel authority.
    let req = stat_request();
    let mut w = encode_rep(&FsResponse::Done, 0);
    w.grants = 1;
    assert!(decode_rep_for(&req, &w).is_none());
}

#[test]
fn a_reply_with_a_set_reserve_is_refused() {
    let req = stat_request();
    let mut w = encode_rep(&FsResponse::Done, 0);
    w.reserved0 = 1.into();
    assert!(decode_rep_for(&req, &w).is_none());
    let mut w = encode_rep(&FsResponse::Done, 0);
    w.reserved1 = 1.into();
    assert!(decode_rep_for(&req, &w).is_none());
}

#[test]
fn a_refusal_carrying_payload_is_refused() {
    // A refusal is a status and nothing else; payload alongside it is a reply
    // this service did not write.
    let req = stat_request();
    for mutate in [
        (|w: &mut FsRepWire| w.handle = 7.into()) as fn(&mut FsRepWire),
        |w: &mut FsRepWire| w.len = 7.into(),
        |w: &mut FsRepWire| w.revision = 7.into(),
        |w: &mut FsRepWire| w.count = 7.into(),
        |w: &mut FsRepWire| w.cursor = 7.into(),
        |w: &mut FsRepWire| w.kind = 1,
        |w: &mut FsRepWire| w.flags = 1,
    ] {
        let mut w = encode_rep(&FsResponse::Refused(FsError::NotFound), 0);
        mutate(&mut w);
        assert!(decode_rep_for(&req, &w).is_none());
    }
}

#[test]
fn an_undefined_reply_flag_is_refused() {
    let req = FsRequest::List {
        handle: Handle::from_bits(1),
        cursor: 0,
        window_offset: 0,
    };
    let mut w = encode_rep(
        &FsResponse::Listed {
            entries: 1,
            next: None,
        },
        0,
    );
    w.flags = 0x80;
    assert!(decode_rep_for(&req, &w).is_none());
}

#[test]
fn only_a_listing_may_set_flags() {
    let req = stat_request();
    let mut w = encode_rep(
        &FsResponse::Stat {
            object: ObjectId::from_bits(4),
            stable_object: kore_svc_dagfs::StableObjectId::new(42).unwrap(),
            kind: ObjectKind::File,
            subtype: kore_svc_dagfs::ObjectSubtype::PLAIN,
            len: 0,
            revision: 1,
            links: 1,
            grants: 0,
        },
        0,
    );
    w.flags = FsRepWire::FLAG_MORE;
    assert!(decode_rep_for(&req, &w).is_none());
}

#[test]
fn a_complete_listing_must_not_carry_a_resume_cursor() {
    // A stale cursor with no MORE flag is a resume point a client might trust.
    let req = FsRequest::List {
        handle: Handle::from_bits(1),
        cursor: 0,
        window_offset: 0,
    };
    let mut w = encode_rep(
        &FsResponse::Listed {
            entries: 2,
            next: None,
        },
        0,
    );
    w.cursor = 5.into();
    assert!(decode_rep_for(&req, &w).is_none());
}

#[test]
fn an_invalid_kind_in_an_open_reply_is_refused() {
    let req = FsRequest::OpenRoot {
        rights: Rights::ALL,
    };
    let mut w = encode_rep(
        &FsResponse::Opened {
            handle: Handle::from_bits(1),
            stable_object: kore_svc_dagfs::StableObjectId::new(41).unwrap(),
            kind: ObjectKind::File,
            subtype: kore_svc_dagfs::ObjectSubtype::PLAIN,
            len: 0,
            revision: 1,
        },
        0,
    );
    for bad in [0u8, 3, 255] {
        w.kind = bad;
        assert!(decode_rep_for(&req, &w).is_none(), "kind {bad} refused");
    }
}

#[test]
fn a_reply_is_decoded_against_the_request_it_answers() {
    // One fixed record serves every verb, so `count` is a byte count for a
    // Read and a link count for a Stat. Stat now also carries a stable object
    // ID in a request-specific word, so replaying those bytes as a Read is
    // rejected rather than silently reinterpreted.
    let wire = encode_rep(
        &FsResponse::Stat {
            object: ObjectId::from_bits(0x0000_0002_0000_0004),
            stable_object: kore_svc_dagfs::StableObjectId::new(42).unwrap(),
            kind: ObjectKind::File,
            subtype: kore_svc_dagfs::ObjectSubtype::PLAIN,
            len: 900,
            revision: 5,
            links: 3,
            grants: 0,
        },
        0,
    );
    let as_stat = decode_rep_for(&stat_request(), &wire).unwrap().0;
    assert!(matches!(as_stat, FsResponse::Stat { links: 3, .. }));

    let read = FsRequest::Read {
        handle: Handle::from_bits(1),
        offset: 0,
        window_offset: 0,
        length: 8,
    };
    assert!(decode_rep_for(&read, &wire).is_none());
}

#[test]
fn reply_matches_rejects_a_response_for_a_different_verb() {
    let stat = stat_request();
    let done = FsResponse::Done;
    assert!(
        !reply_matches(&stat, &done),
        "a Stat is not answered by Done"
    );

    let close = FsRequest::Close {
        handle: Handle::from_bits(1),
    };
    assert!(reply_matches(&close, &done));
    assert!(
        !reply_matches(&close, &FsResponse::Transferred { count: 0 }),
        "a Close is not answered by a transfer count"
    );
    // Any request may be refused.
    assert!(reply_matches(
        &stat,
        &FsResponse::Refused(FsError::NotFound)
    ));
}

#[test]
fn a_reply_reads_from_arbitrary_bytes_without_panicking() {
    use zerocopy::FromBytes;
    let req = stat_request();
    for fill in [0x00u8, 0x01, 0x41, 0xa5, 0xff] {
        let raw = [fill; core::mem::size_of::<FsRepWire>()];
        let w = FsRepWire::read_from_bytes(&raw).expect("layout admits any bytes");
        let _ = decode_rep_for(&req, &w);
    }
}

// ── framing ─────────────────────────────────────────────────────────────────

#[test]
fn the_records_are_the_frozen_sizes_and_fit_the_pair_window() {
    assert_eq!(core::mem::size_of::<FsReqWire>(), 120);
    assert_eq!(core::mem::size_of::<FsRepWire>(), 48);

    // `align64(192 + 8·rec)` per ring, both inside `[layout] ring_bytes`
    // (0x2000).  Recomputed here from the live sizes so a record that grew is
    // caught by a test as well as by the ABI's const-assert.
    let align64 = |raw: usize| (raw + 63) & !63;
    let req_ring = align64(192 + 8 * core::mem::size_of::<FsReqWire>());
    let rep_ring = align64(192 + 8 * core::mem::size_of::<FsRepWire>());
    assert_eq!(req_ring, 1152);
    assert_eq!(rep_ring, 576);
    assert!(
        req_ring + rep_ring <= 0x2000,
        "both 8-deep rings must fit the pair's window"
    );
}

#[test]
fn an_encoded_request_leaves_every_unused_field_zero() {
    // The encoder starts from a zeroed record, so an unused field is zero
    // rather than stale — which is what makes the decoder's strict zero checks
    // safe to apply.
    let w = encode_req(&FsRequest::Close {
        handle: Handle::from_bits(0x0000_0001_0000_0002),
    });
    assert_eq!(w.reserved0, 0);
    assert_eq!(w.reserved1.get(), 0);
    assert_eq!(w.reserved2.get(), 0);
    assert_eq!(w.kind, 0);
    assert_eq!(w.rights, 0);
    assert_eq!(w.session.get(), 0);
    assert_eq!(w.name, [0u8; FS_NAME_BYTES]);
    assert_eq!(w.name2, [0u8; FS_NAME_BYTES]);
    // And the record is plain bytes end to end — no padding surprises.
    assert_eq!(w.as_bytes().len(), 120);
}
