//! The production shape: a **real founded pair window** with real SPSC rings.
//!
//! `loopback.rs` proves the codec and dispatch over encoded bytes but calls
//! `dispatch` directly. This suite goes through the transport the founder
//! actually builds:
//!
//! ```text
//! client → FsReqWire → spsc request ring (in the pair window)
//!                    → service pops, dispatches
//!                    → FsRepWire → spsc reply ring → client pops
//! ```
//!
//! The window is formatted by `kore_rt::svc::split_window` at `REQ_CAP`/
//! `REP_CAP` — the same call the founder makes, at this protocol's exact record
//! widths — so the ring geometry, the request/reply split offset, the slot
//! strides and the wrap behaviour are the real ones, not a re-derivation.
//!
//! Two windows are in play, and keeping them distinct is the point: the **pair
//! window** carries fixed-size records, and the **data window** carries bulk
//! bytes. Production has exactly this split.
//!
//! What is still not production: the bytes do not cross an address-space
//! boundary, and no founder delivered the windows. That is Phase 2 step 5 and
//! needs the QEMU boot gate.

mod common;

use common::name;
use kore_abi::svc::{FsRepWire, FsReqWire};
use kore_ring::spsc::{Consumer, Producer};
use kore_rt::svc::{REP_CAP, REQ_CAP, rep_offset, split_window, window_bytes};
use kore_svc_dagfs::{
    ANY_REVISION, DataWindow, FsError, FsRequest, FsResponse, ObjectKind, RamService, Rights,
    SessionId, block_on,
    wire::{decode_rep_for, decode_req, encode_rep, encode_req},
};

/// Bytes of bulk data window (separate from the pair's ring window).
const DATA_BYTES: usize = 512;

/// The pair window's size, from the tree's own arithmetic.
const RING_WINDOW_BYTES: usize = window_bytes::<RamService>(REQ_CAP, REP_CAP);

/// A **64-byte aligned** pair window.
///
/// The rings require cache-line alignment (`spsc`'s `ALIGN`), and refuse
/// `Misaligned` without it — adjacent rings packed in one window each have to
/// start on a line so their cursors do not share one. In production that comes
/// free: the founder maps a page-granular window. On a host a plain `Vec<u8>`
/// has no such guarantee, so the alignment is stated here rather than left to
/// the allocator's mood.
#[repr(align(64))]
struct RingWindow([u8; RING_WINDOW_BYTES]);

impl RingWindow {
    fn new() -> Self {
        Self([0; RING_WINDOW_BYTES])
    }
}

/// One founded pair, host-side: the two ring halves over one formatted window,
/// the service, and the client's data window.
struct Pair<'w> {
    client_req: Producer<'w, FsReqWire>,
    client_rep: Consumer<'w, FsRepWire>,
    serve_req: Consumer<'w, FsReqWire>,
    serve_rep: Producer<'w, FsRepWire>,
    service: RamService,
    session: SessionId,
    data: Vec<u8>,
}

impl<'w> Pair<'w> {
    /// The service's serving step: drain every queued request, dispatch it, and
    /// push each reply. This is what `ServeHost`'s loop does per doorbell.
    fn serve_pending(&mut self) -> usize {
        let mut served = 0;
        while let Some(record) = self.serve_req.pop() {
            let response = match decode_req(&record) {
                Some(request) => {
                    let mut window = DataWindow::new(&mut self.data);
                    self.service
                        .dispatch(self.session, &request, Some(&mut window))
                }
                // A record the service cannot read is refused, never guessed.
                None => FsResponse::Refused(FsError::Unsupported),
            };
            self.serve_rep
                .push(encode_rep(&response, 0))
                .expect("the reply ring has a slot per outstanding request");
            served += 1;
        }
        served
    }

    /// One full round trip through both rings.
    fn call(&mut self, request: &FsRequest) -> FsResponse {
        self.client_req
            .push(encode_req(request))
            .expect("request ring has room");
        self.serve_pending();
        let record = self.client_rep.pop().expect("a reply was produced");
        let (response, grants) =
            decode_rep_for(request, &record).expect("the reply decodes against its request");
        assert_eq!(grants, 0, "the filesystem grants no kernel authority");
        response
    }

    /// `call`, folding a refusal into `Err` the way the typed client does.
    fn checked(&mut self, request: &FsRequest) -> Result<FsResponse, FsError> {
        match self.call(request) {
            FsResponse::Refused(err) => Err(err),
            other => Ok(other),
        }
    }

    fn open_root(&mut self) -> kore_svc_dagfs::Handle {
        match self.checked(&FsRequest::OpenRoot {
            rights: Rights::ALL,
        }) {
            Ok(FsResponse::Opened { handle, .. }) => handle,
            other => panic!("open_root: {other:?}"),
        }
    }

    fn create(
        &mut self,
        parent: kore_svc_dagfs::Handle,
        text: &str,
        kind: ObjectKind,
    ) -> kore_svc_dagfs::Handle {
        match self.checked(&FsRequest::Create {
            parent,
            name: name(text),
            kind,
            subtype: kore_svc_dagfs::ObjectSubtype::PLAIN,
        }) {
            Ok(FsResponse::Opened { handle, .. }) => handle,
            other => panic!("create {text}: {other:?}"),
        }
    }
}

/// Build a pair over `window`, with the service attached and a data window
/// declared.
fn pair(window: &mut [u8]) -> Pair<'_> {
    let (client, serve) =
        split_window::<RamService>(window, REQ_CAP, REP_CAP).expect("format the pair window");
    let session = SessionId::new(0);
    let mut service = RamService::default();
    block_on(service.engine().attach_session(session)).expect("attach session");
    let mut pair = Pair {
        client_req: client.req,
        client_rep: client.rep,
        serve_req: serve.req,
        serve_rep: serve.rep,
        service,
        session,
        data: vec![0u8; DATA_BYTES],
    };
    pair.checked(&FsRequest::AttachWindow {
        slot: 7,
        bytes: DATA_BYTES as u32,
    })
    .expect("attach the data window");
    pair
}

// ── geometry ────────────────────────────────────────────────────────────────

#[test]
fn the_pair_window_is_the_geometry_the_founder_would_build() {
    // Computed by `kore_rt::svc`, not restated here — if the record widths or
    // the ring caps move, this moves with them.
    let bytes = window_bytes::<RamService>(REQ_CAP, REP_CAP);
    assert_eq!(bytes, 1728);
    assert_eq!(rep_offset::<RamService>(REQ_CAP), 1152);
    assert!(
        bytes <= 0x2000,
        "the pair must fit `[layout] ring_bytes` (3 pages)"
    );
    // And a window one byte short must refuse to format rather than overlay
    // the reply ring onto the request ring's slots. Sliced off an aligned
    // buffer so the refusal is about size, not alignment.
    let mut short = RingWindow::new();
    assert_eq!(
        split_window::<RamService>(&mut short.0[..bytes - 1], REQ_CAP, REP_CAP).err(),
        Some(kore_ring::spsc::RingError::TooSmall)
    );
}

#[test]
fn the_rings_carry_the_declared_depth() {
    let mut buf = RingWindow::new();
    let p = pair(&mut buf.0);
    assert_eq!(p.client_req.capacity(), REQ_CAP);
    assert_eq!(p.client_rep.capacity(), REP_CAP);
    assert_eq!(p.serve_req.capacity(), REQ_CAP);
    assert_eq!(p.serve_rep.capacity(), REP_CAP);
}

#[test]
fn immutable_extraction_crosses_the_same_production_pair() {
    use kore_svc_extract::{
        InputIdentity, NormalizedResult, ParserProvenance, encode_result,
        registry::{C_JSON, RUST_JSON},
    };
    use kore_svc_extract_worker_rust::parse_json_rust;

    let mut buf = RingWindow::new();
    let mut p = pair(&mut buf.0);
    let root = p.open_root();
    let file = p.create(root, "source.json", ObjectKind::File);
    let bytes = br#"{"a":1,"b":[2,3]}"#;
    p.data[..bytes.len()].copy_from_slice(bytes);
    let revision = match p.checked(&FsRequest::WriteAt {
        handle: file,
        offset: 0,
        window_offset: 0,
        length: bytes.len() as u32,
        expected: ANY_REVISION,
    }) {
        Ok(FsResponse::Committed { revision }) => revision,
        other => panic!("write: {other:?}"),
    };
    let lease = match p.checked(&FsRequest::LendVersion {
        handle: file,
        window_offset: 0,
    }) {
        Ok(FsResponse::VersionLent { lease }) => lease,
        other => panic!("lend: {other:?}"),
    };
    let identity = InputIdentity {
        source: u64::from_le_bytes(p.data[0..8].try_into().unwrap()),
        version: u64::from_le_bytes(p.data[8..16].try_into().unwrap()),
        len: u64::from_le_bytes(p.data[16..24].try_into().unwrap()),
        digest: p.data[24..56].try_into().unwrap(),
    };
    assert_eq!(identity.version, revision);
    p.data.fill(0);
    assert_eq!(
        p.checked(&FsRequest::ReadVersion {
            lease,
            offset: 0,
            window_offset: 0,
            length: bytes.len() as u32,
        }),
        Ok(FsResponse::Transferred {
            count: bytes.len() as u32
        })
    );
    assert_eq!(&p.data[..bytes.len()], bytes);

    let result = parse_json_rust(identity, bytes).unwrap();
    let mut old_result = result.clone();
    old_result.provenance = ParserProvenance::deterministic(identity, RUST_JSON, 1);
    p.data.fill(0);
    let used = encode_result(&old_result, &mut p.data).unwrap();
    assert_eq!(
        p.checked(&FsRequest::CommitDerived {
            lease,
            window_offset: 0,
            length: used as u32,
            output_window_bytes: DATA_BYTES as u32,
        }),
        Ok(FsResponse::DerivedCommitted {
            parser: RUST_JSON,
            parser_version: 1,
        })
    );

    let c_result = NormalizedResult::empty(ParserProvenance::deterministic(identity, C_JSON, 1));
    p.data.fill(0);
    let used = encode_result(&c_result, &mut p.data).unwrap();
    assert_eq!(
        p.checked(&FsRequest::CommitDerived {
            lease,
            window_offset: 0,
            length: used as u32,
            output_window_bytes: DATA_BYTES as u32,
        }),
        Ok(FsResponse::DerivedCommitted {
            parser: C_JSON,
            parser_version: 1,
        })
    );

    p.data.fill(0);
    let used = encode_result(&result, &mut p.data).unwrap();
    assert_eq!(
        p.checked(&FsRequest::CommitDerived {
            lease,
            window_offset: 0,
            length: used as u32,
            output_window_bytes: DATA_BYTES as u32,
        }),
        Ok(FsResponse::DerivedCommitted {
            parser: RUST_JSON,
            parser_version: 2,
        })
    );
    assert_eq!(
        p.checked(&FsRequest::DerivedInfo {
            handle: file,
            parser: RUST_JSON,
        }),
        Ok(FsResponse::DerivedInfo {
            input_version: revision,
            parser_version: 2,
        })
    );
    assert_eq!(
        p.checked(&FsRequest::DerivedInfo {
            handle: file,
            parser: C_JSON,
        }),
        Ok(FsResponse::DerivedInfo {
            input_version: revision,
            parser_version: 1,
        }),
        "upgrading one parser lineage preserves another parser's facts"
    );
    assert_eq!(
        p.checked(&FsRequest::ReleaseVersion { lease }),
        Ok(FsResponse::Done)
    );
    assert_eq!(
        p.checked(&FsRequest::ReadVersion {
            lease,
            offset: 0,
            window_offset: 0,
            length: 1,
        }),
        Err(FsError::InvalidHandle),
        "a released generation cannot read a replacement slot"
    );
}

// ── the workflow, over real rings ───────────────────────────────────────────

#[test]
fn the_full_workflow_crosses_real_rings() {
    let mut buf = RingWindow::new();
    let mut p = pair(&mut buf.0);

    let root = p.open_root();
    let dir = p.create(root, "demo", ObjectKind::Directory);
    let file = p.create(dir, "hello", ObjectKind::File);

    // Write through the data window.
    p.data[..4].copy_from_slice(b"kore");
    let revision = match p.checked(&FsRequest::WriteAt {
        handle: file,
        offset: 0,
        window_offset: 0,
        length: 4,
        expected: ANY_REVISION,
    }) {
        Ok(FsResponse::Committed { revision }) => revision,
        other => panic!("write: {other:?}"),
    };

    // Read it back through the data window.
    p.data.fill(0);
    match p.checked(&FsRequest::Read {
        handle: file,
        offset: 0,
        window_offset: 0,
        length: 4,
    }) {
        Ok(FsResponse::Transferred { count }) => assert_eq!(count, 4),
        other => panic!("read: {other:?}"),
    }
    assert_eq!(&p.data[..4], b"kore");

    // A second name for the same object, then a listing.
    p.checked(&FsRequest::Link {
        parent: dir,
        name: name("hello-again"),
        child: file,
    })
    .unwrap();
    match p.checked(&FsRequest::List {
        handle: dir,
        cursor: 0,
        window_offset: 0,
    }) {
        Ok(FsResponse::Listed { entries, next }) => {
            assert_eq!(entries, 2);
            assert_eq!(next, None);
        }
        other => panic!("list: {other:?}"),
    }

    // And the object's own view of itself.
    match p.checked(&FsRequest::Stat { handle: file }) {
        Ok(FsResponse::Stat {
            len,
            links,
            revision: r,
            ..
        }) => {
            assert_eq!(len, 4);
            assert_eq!(links, 2);
            assert_eq!(r, revision);
        }
        other => panic!("stat: {other:?}"),
    }
}

#[test]
fn a_payload_larger_than_one_record_still_crosses_in_the_data_window() {
    // The point of the two-window split: the ring record stays 120 bytes no
    // matter how much data moves.
    let mut buf = RingWindow::new();
    let mut p = pair(&mut buf.0);
    let root = p.open_root();
    let file = p.create(root, "big", ObjectKind::File);

    let payload: Vec<u8> = (0..400u32).map(|i| (i % 251) as u8).collect();
    p.data[..payload.len()].copy_from_slice(&payload);
    p.checked(&FsRequest::WriteAt {
        handle: file,
        offset: 0,
        window_offset: 0,
        length: payload.len() as u32,
        expected: ANY_REVISION,
    })
    .unwrap();

    p.data.fill(0);
    p.checked(&FsRequest::Read {
        handle: file,
        offset: 0,
        window_offset: 0,
        length: payload.len() as u32,
    })
    .unwrap();
    assert_eq!(&p.data[..payload.len()], &payload[..]);
    assert_eq!(
        size_of::<FsReqWire>(),
        120,
        "the record did not grow with it"
    );
}

// ── ring behaviour ──────────────────────────────────────────────────────────

#[test]
fn the_request_ring_fills_at_its_declared_depth_and_refuses() {
    let mut buf = RingWindow::new();
    let mut p = pair(&mut buf.0);
    let root = p.open_root();

    // Queue without serving: the ring must accept exactly `REQ_CAP` and then
    // refuse, rather than overwrite an unserved request.
    let request = encode_req(&FsRequest::Stat { handle: root });
    let mut queued = 0;
    while p.client_req.push(request).is_ok() {
        queued += 1;
        assert!(queued <= REQ_CAP, "a full ring must refuse, not wrap");
    }
    assert_eq!(queued, REQ_CAP);

    // Draining serves every one, in order, and re-arms the ring.
    assert_eq!(p.serve_pending(), REQ_CAP as usize);
    for _ in 0..REQ_CAP {
        let record = p.client_rep.pop().expect("one reply per request");
        assert_eq!(record.status, FsRepWire::STATUS_OK);
    }
    assert!(p.client_rep.pop().is_none(), "no extra replies");
    assert!(p.client_req.push(request).is_ok(), "the ring re-armed");
}

#[test]
fn requests_are_served_in_order_across_the_ring() {
    // The SPSC ring is ordered, and the dispatch is sequential — so a batch of
    // appends must land in the order they were queued.
    let mut buf = RingWindow::new();
    let mut p = pair(&mut buf.0);
    let root = p.open_root();
    let log = p.create(root, "log", ObjectKind::File);

    p.data[..1].copy_from_slice(b"x");
    for _ in 0..4 {
        p.client_req
            .push(encode_req(&FsRequest::Append {
                handle: log,
                window_offset: 0,
                length: 1,
                expected: ANY_REVISION,
            }))
            .unwrap();
    }
    assert_eq!(p.serve_pending(), 4);

    let mut offsets = Vec::new();
    for _ in 0..4 {
        let record = p.client_rep.pop().unwrap();
        // `len` carries an append's offset.
        offsets.push(record.len.get());
    }
    assert_eq!(offsets, vec![0, 1, 2, 3], "served in queue order");
}

#[test]
fn a_garbage_record_on_the_ring_is_refused_without_disturbing_the_stream() {
    let mut buf = RingWindow::new();
    let mut p = pair(&mut buf.0);
    let root = p.open_root();

    // A hostile peer writes a record with an unknown verb straight into the
    // ring, followed by a good one.
    let mut bogus = encode_req(&FsRequest::Stat { handle: root });
    bogus.op = 200;
    p.client_req.push(bogus).unwrap();
    p.client_req
        .push(encode_req(&FsRequest::Stat { handle: root }))
        .unwrap();

    assert_eq!(p.serve_pending(), 2);
    let first = p.client_rep.pop().unwrap();
    let second = p.client_rep.pop().unwrap();
    assert_ne!(
        first.status,
        FsRepWire::STATUS_OK,
        "garbage must be refused"
    );
    assert_eq!(
        second.status,
        FsRepWire::STATUS_OK,
        "and the next request must still be served"
    );
}

#[test]
fn the_pair_window_holds_only_records_never_payload() {
    // The security property the two-window split exists for: a client's file
    // bytes must never appear in the ring window, because the ring window is
    // shared with the service for control and a payload there would widen what
    // crosses on every call.
    let mut buf = RingWindow::new();
    let secret = b"SECRETPAYLOAD";
    {
        let mut p = pair(&mut buf.0);
        let root = p.open_root();
        let file = p.create(root, "f", ObjectKind::File);
        p.data[..secret.len()].copy_from_slice(secret);
        p.checked(&FsRequest::WriteAt {
            handle: file,
            offset: 0,
            window_offset: 0,
            length: secret.len() as u32,
            expected: ANY_REVISION,
        })
        .unwrap();
        p.data.fill(0);
        p.checked(&FsRequest::Read {
            handle: file,
            offset: 0,
            window_offset: 0,
            length: secret.len() as u32,
        })
        .unwrap();
        assert_eq!(&p.data[..secret.len()], secret, "the bytes did move");
    }
    // The whole ring window, scanned for the payload.
    assert!(
        !buf.0.windows(secret.len()).any(|w| w == secret),
        "file bytes must never appear in the pair's ring window"
    );
}

// ── refusals over the real transport ────────────────────────────────────────

#[test]
fn refusals_cross_the_rings_intact() {
    let mut buf = RingWindow::new();
    let mut p = pair(&mut buf.0);
    let root = p.open_root();

    assert_eq!(
        p.checked(&FsRequest::OpenAt {
            parent: root,
            name: name("absent"),
            rights: Rights::ALL,
        })
        .unwrap_err(),
        FsError::NotFound
    );
    let file = p.create(root, "f", ObjectKind::File);
    assert_eq!(
        p.checked(&FsRequest::Read {
            handle: root,
            offset: 0,
            window_offset: 0,
            length: 4,
        })
        .unwrap_err(),
        FsError::IsDirectory
    );
    // A window-relative range the service will not honour.
    assert_eq!(
        p.checked(&FsRequest::Read {
            handle: file,
            offset: 0,
            window_offset: DATA_BYTES as u32,
            length: 8,
        })
        .unwrap_err(),
        FsError::Range
    );
}

#[test]
fn losing_a_route_reaches_the_client_over_the_rings() {
    let mut buf = RingWindow::new();
    let mut p = pair(&mut buf.0);
    let root = p.open_root();
    let dir = p.create(root, "d", ObjectKind::Directory);
    let file = p.create(dir, "f", ObjectKind::File);

    p.checked(&FsRequest::Unlink {
        parent: dir,
        name: name("f"),
    })
    .unwrap();
    assert_eq!(
        p.checked(&FsRequest::Stat { handle: file }).unwrap_err(),
        FsError::AccessLost
    );
}

// ── the service contract itself ─────────────────────────────────────────────

#[test]
fn the_service_declares_an_empty_endowment() {
    use kore_rt::service::{Bundle, SystemService};
    // The filesystem claims nothing at startup, and the founder reads this.
    assert!(<RamService as SystemService>::Endowment::PROFILE.is_empty());
    assert_eq!(<RamService as SystemService>::NAME, "fs");
}

#[test]
fn the_protocol_declares_no_reply_grants() {
    use kore_rt::svc::SvcProtocol;
    assert_eq!(<RamService as SvcProtocol>::REPLY_GRANTS, 0);
}

#[test]
fn the_protocols_reply_matching_rejects_a_mismatched_answer() {
    use kore_rt::svc::SvcProtocol;
    let stat = encode_req(&FsRequest::Stat {
        handle: kore_svc_dagfs::Handle::from_bits(1),
    });
    // `Done` is not an answer to a `Stat`.
    assert!(!<RamService as SvcProtocol>::reply_matches(
        &stat,
        &FsResponse::Done,
        0
    ));
    // A refusal answers anything.
    assert!(<RamService as SvcProtocol>::reply_matches(
        &stat,
        &FsResponse::Refused(FsError::NotFound),
        0
    ));
    // And a grant-bearing reply is refused outright.
    assert!(!<RamService as SvcProtocol>::reply_matches(
        &stat,
        &FsResponse::Refused(FsError::NotFound),
        1
    ));
}
