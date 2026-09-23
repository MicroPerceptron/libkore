//! End-to-end over the real wire: client → encoded record → codec → dispatch →
//! encoded record → client.
//!
//! Everything here crosses the frozen `FsReqWire`/`FsRepWire` forms as bytes.
//! Nothing is mocked at the enum level, so a codec bug, a per-verb field
//! collision, or a dispatch that mis-reads the data window fails *here* rather
//! than surviving until a founded client exists.
//!
//! This is what Phase 2 can honestly prove on a host. The one thing it cannot
//! is that a real founded pair delivers the bytes — that is Phase 2 step 5, and
//! it needs the QEMU boot gate.

mod common;

use std::{cell::RefCell, rc::Rc};

use common::name;
use kore_abi::svc::{FsDirEntryWire, FsRepWire, FsReqWire};
use kore_svc_dagfs::{
    ANY_REVISION, DataWindow, FsClient, FsError, FsResult, ObjectKind, ProviderError, RamService,
    Rights, SearchQuery, SessionId, Transport, read_search_hit,
    wire::{decode_req, encode_rep},
};

/// Bytes in each client's data window.
const WINDOW_BYTES: usize = 4096;

/// One service, shared by every session in a scenario — the shape a real
/// multi-client host has, without needing one yet.
type SharedService = Rc<RefCell<RamService>>;

/// A client's end of the pair: the shared service, this client's session, and
/// the data window it delivered.
///
/// The window lives here because on metal it is the *client's* memory that the
/// service was given a view of. Holding it on the client side keeps that
/// ownership honest — the service reaches it only through the view handed to
/// `dispatch`.
struct Loopback {
    service: SharedService,
    session: SessionId,
    window: Vec<u8>,
}

impl Transport for Loopback {
    fn exchange(&mut self, request: &FsReqWire) -> FsResult<FsRepWire> {
        // Decode exactly as the serving loop would: a record it cannot read is
        // refused, never guessed at.
        let Some(decoded) = decode_req(request) else {
            return Ok(encode_rep(
                &kore_svc_dagfs::FsResponse::Refused(FsError::Unsupported),
                0,
            ));
        };
        let mut window = DataWindow::new(&mut self.window);
        let response =
            self.service
                .borrow_mut()
                .dispatch(self.session, &decoded, Some(&mut window));
        Ok(encode_rep(&response, 0))
    }
}

/// Build a world with `sessions` attached clients, each with an open root and
/// an attached data window.
fn world(sessions: u16) -> (SharedService, Vec<FsClient<Loopback>>) {
    let service: SharedService = Rc::new(RefCell::new(RamService::default()));
    let mut clients = Vec::new();
    for index in 0..sessions {
        let session = SessionId::new(index);
        kore_svc_dagfs::block_on(service.borrow_mut().engine().attach_session(session))
            .expect("attach session");
        let mut client = FsClient::new(Loopback {
            service: Rc::clone(&service),
            session,
            window: vec![0u8; WINDOW_BYTES],
        });
        client
            .attach_window(index as u32 + 1, WINDOW_BYTES as u32)
            .expect("attach window");
        clients.push(client);
    }
    (service, clients)
}

/// Put `bytes` at the start of a client's window and write them to `handle`.
fn write_all(
    client: &mut FsClient<Loopback>,
    handle: kore_svc_dagfs::Handle,
    bytes: &[u8],
) -> FsResult<u64> {
    client.transport().window[..bytes.len()].copy_from_slice(bytes);
    client.write_at(handle, 0, 0, bytes.len() as u32, ANY_REVISION)
}

/// Read a whole file back through the window.
fn read_all(client: &mut FsClient<Loopback>, handle: kore_svc_dagfs::Handle) -> FsResult<Vec<u8>> {
    let len = client.stat(handle)?.len as u32;
    let count = client.read(handle, 0, 0, len)?;
    Ok(client.transport().window[..count as usize].to_vec())
}

// ── the basic round trip ────────────────────────────────────────────────────

#[test]
fn the_status_record_crosses_the_wire_whole() {
    use kore_abi::svc::FsSystemStatusWire;

    let (_svc, mut clients) = world(1);
    let c = &mut clients[0];

    let len = c.system_status(0).unwrap();
    assert_eq!(len as usize, FsSystemStatusWire::BYTES);
    let wire = kore_svc_dagfs::read_system_status(&c.transport().window, 0).unwrap();
    // The bare RAM harness: volatile posture, one attached window (ours).
    assert_eq!(wire.flags.get() & FsSystemStatusWire::FLAG_DURABLE, 0);
    assert_eq!(wire.windows.get(), 1);
    assert_eq!(wire.commits.get(), 0);
}

#[test]
fn create_write_and_read_back_across_the_wire() {
    let (_svc, mut clients) = world(1);
    let c = &mut clients[0];

    let root = c.open_root(Rights::ALL).unwrap();
    assert_eq!(root.kind, ObjectKind::Directory);

    let file = c
        .create(root.handle, &name("hello"), ObjectKind::File)
        .unwrap();
    assert_eq!(file.kind, ObjectKind::File);
    assert_eq!(file.len, 0);

    let revision = write_all(c, file.handle, b"kore").unwrap();
    assert!(revision > file.revision);

    assert_eq!(read_all(c, file.handle).unwrap(), b"kore");
    let info = c.stat(file.handle).unwrap();
    assert_eq!(info.len, 4);
    assert_eq!(info.links, 1);
    assert_eq!(info.revision, revision);
}

#[test]
fn a_multi_chunk_payload_survives_the_window_round_trip() {
    let (_svc, mut clients) = world(1);
    let c = &mut clients[0];
    let root = c.open_root(Rights::ALL).unwrap();
    let file = c
        .create(root.handle, &name("big"), ObjectKind::File)
        .unwrap();

    // Larger than one provider chunk, so the transfer straddles chunk
    // boundaries on the way in and on the way out.
    let payload: Vec<u8> = (0..600u32).map(|i| (i % 251) as u8).collect();
    write_all(c, file.handle, &payload).unwrap();
    assert_eq!(read_all(c, file.handle).unwrap(), payload);
}

#[test]
fn typed_version_lending_pins_bytes_and_releases_affinely() {
    let (_svc, mut clients) = world(1);
    let c = &mut clients[0];
    let root = c.open_root(Rights::ALL).unwrap();
    let file = c
        .create(root.handle, &name("program"), ObjectKind::File)
        .unwrap();
    let revision = write_all(c, file.handle, b"immutable-v1").unwrap();

    let pending = c.lend_version(file.handle, 64).unwrap();
    let identity_window = c.transport().window.clone();
    let version = pending.identify(&identity_window, 64).unwrap();
    let identity = version.identity();
    assert_eq!(identity.version, revision);
    assert_eq!(identity.len, 12);
    assert_eq!(identity.digest, *blake3::hash(b"immutable-v1").as_bytes());

    // The ordinary file can advance while the lent revision stays frozen.
    write_all(c, file.handle, b"mutable-v2!!").unwrap();
    let count = c.read_version(&version, 0, 0, WINDOW_BYTES as u32).unwrap();
    assert_eq!(&c.transport().window[..count as usize], b"immutable-v1");
    c.release_version(version).unwrap();
}

#[test]
fn refused_identity_landing_does_not_consume_snapshot_capacity() {
    let (_svc, mut clients) = world(1);
    let c = &mut clients[0];
    let root = c.open_root(Rights::ALL).unwrap();
    let file = c
        .create(root.handle, &name("image"), ObjectKind::File)
        .unwrap();
    write_all(c, file.handle, b"elf").unwrap();

    // This used to install the snapshot first and only then discover that its
    // identity did not fit, leaking one unreachable lease per refusal.
    for _ in 0..(kore_svc_dagfs::bounds::MAX_LENT_VERSIONS + 1) {
        assert_eq!(
            c.lend_version(file.handle, WINDOW_BYTES as u32 - 8)
                .unwrap_err(),
            FsError::Range
        );
    }

    let pending = c.lend_version(file.handle, 0).unwrap();
    let identity_window = c.transport().window.clone();
    let version = pending.identify(&identity_window, 0).unwrap();
    c.release_version(version).unwrap();
}

#[test]
fn append_reports_its_offset_across_the_wire() {
    let (_svc, mut clients) = world(1);
    let c = &mut clients[0];
    let root = c.open_root(Rights::ALL).unwrap();
    let log = c
        .create(root.handle, &name("log"), ObjectKind::File)
        .unwrap();

    let mut offsets = Vec::new();
    for _ in 0..3 {
        c.transport().window[..4].copy_from_slice(b"abc\n");
        let (offset, _) = c.append(log.handle, 0, 4, ANY_REVISION).unwrap();
        offsets.push(offset);
    }
    assert_eq!(offsets, vec![0, 4, 8]);
    assert_eq!(read_all(c, log.handle).unwrap(), b"abc\nabc\nabc\n");
}

#[test]
fn a_revision_conflict_crosses_the_wire_intact() {
    let (_svc, mut clients) = world(1);
    let c = &mut clients[0];
    let root = c.open_root(Rights::ALL).unwrap();
    let file = c.create(root.handle, &name("f"), ObjectKind::File).unwrap();

    let stale = c.stat(file.handle).unwrap().revision;
    write_all(c, file.handle, b"first").unwrap();

    c.transport().window[..6].copy_from_slice(b"second");
    assert_eq!(
        c.write_at(file.handle, 0, 0, 6, stale).unwrap_err(),
        FsError::Conflict,
        "the engine's Conflict must reach the client as Conflict"
    );
    assert_eq!(read_all(c, file.handle).unwrap(), b"first");
}

// ── the data window ─────────────────────────────────────────────────────────

#[test]
fn a_bulk_verb_before_attach_is_refused() {
    let service: SharedService = Rc::new(RefCell::new(RamService::default()));
    let session = SessionId::new(0);
    kore_svc_dagfs::block_on(service.borrow_mut().engine().attach_session(session)).unwrap();
    let mut c = FsClient::new(Loopback {
        service: Rc::clone(&service),
        session,
        window: vec![0u8; WINDOW_BYTES],
    });

    // Control-plane verbs work without a window…
    let root = c.open_root(Rights::ALL).unwrap();
    let file = c.create(root.handle, &name("f"), ObjectKind::File).unwrap();
    // …but a read has nowhere to land.
    assert_eq!(c.read(file.handle, 0, 0, 4).unwrap_err(), FsError::NoWindow);
    assert_eq!(
        c.write_at(file.handle, 0, 0, 4, ANY_REVISION).unwrap_err(),
        FsError::NoWindow
    );
    assert_eq!(c.list(root.handle, 0, 0).unwrap_err(), FsError::NoWindow);

    // After attaching, the same calls work.
    c.attach_window(1, WINDOW_BYTES as u32).unwrap();
    assert!(c.list(root.handle, 0, 0).is_ok());
}

#[test]
fn a_transfer_past_the_window_is_refused() {
    let (_svc, mut clients) = world(1);
    let c = &mut clients[0];
    let root = c.open_root(Rights::ALL).unwrap();
    let file = c.create(root.handle, &name("f"), ObjectKind::File).unwrap();
    write_all(c, file.handle, b"payload").unwrap();

    // Landing past the end of the window, and an offset+length that would wrap.
    assert_eq!(
        c.read(file.handle, 0, WINDOW_BYTES as u32, 8).unwrap_err(),
        FsError::Range
    );
    assert_eq!(
        c.read(file.handle, 0, u32::MAX, 8).unwrap_err(),
        FsError::Range
    );
    assert_eq!(
        c.write_at(file.handle, 0, WINDOW_BYTES as u32 - 2, 8, ANY_REVISION)
            .unwrap_err(),
        FsError::Range
    );
    // The file is untouched by every refusal.
    assert_eq!(read_all(c, file.handle).unwrap(), b"payload");
}

#[test]
fn listing_entries_land_in_the_window_and_decode() {
    let (_svc, mut clients) = world(1);
    let c = &mut clients[0];
    let root = c.open_root(Rights::ALL).unwrap();

    for entry in ["alpha", "beta"] {
        let made = c
            .create(root.handle, &name(entry), ObjectKind::File)
            .unwrap();
        c.close(made.handle).unwrap();
    }
    let dir = c
        .create(root.handle, &name("sub"), ObjectKind::Directory)
        .unwrap();
    c.close(dir.handle).unwrap();

    let page = c.list(root.handle, 0, 0).unwrap();
    assert_eq!(page.entries, 3);
    assert_eq!(page.next, None);

    let window = c.transport().window.clone();
    let mut seen: Vec<(String, bool)> = (0..page.entries as usize)
        .map(|i| {
            let entry = FsClient::<Loopback>::read_entry(&window, 0, i).unwrap();
            // `bool` for "is a directory" so the tuple sorts; the kind itself
            // is an enum with no ordering, and inventing one for a test would
            // be a worse trade than this.
            (
                entry.name.as_str().to_string(),
                entry.kind == ObjectKind::Directory,
            )
        })
        .collect();
    seen.sort();
    assert_eq!(
        seen,
        vec![
            ("alpha".to_string(), false),
            ("beta".to_string(), false),
            ("sub".to_string(), true),
        ]
    );
}

#[test]
fn structural_search_crosses_the_real_wire_and_window() {
    let (_svc, mut clients) = world(1);
    let c = &mut clients[0];
    let root = c.open_root(Rights::ALL).unwrap();
    let sub = c
        .create(root.handle, &name("docs"), ObjectKind::Directory)
        .unwrap();
    let report = c
        .create(sub.handle, &name("report.csv"), ObjectKind::File)
        .unwrap();
    write_all(c, report.handle, b"a,b\n1,2\n").unwrap();
    c.create(sub.handle, &name("notes.txt"), ObjectKind::File)
        .unwrap();

    let page = c
        .search(
            root.handle,
            SearchQuery {
                name_contains: Some(name("report")),
                kind: Some(ObjectKind::File),
                min_len: 1,
                max_len: 64,
                content: None,
                freshness: kore_svc_dagfs::FreshnessPolicy::StructuralOnly,
                max_index_lag: 0,
                max_work: 0,
            },
            None,
            0,
        )
        .unwrap();
    assert_eq!(page.hits, 1);
    assert!(page.observation != 0);
    assert_eq!(page.next, None);
    let hit = read_search_hit(&c.transport().window, 0, 0).unwrap();
    let path: Vec<&str> = hit.path.iter().map(|part| part.as_str()).collect();
    assert_eq!(path, ["docs", "report.csv"]);
    assert_eq!(hit.kind, ObjectKind::File);
    assert_eq!(hit.len, 8);
}

/// The deepest legal path crosses the wire whole.
///
/// `MAX_PATH_DEPTH` went 8 → 32 and the frozen hit spelling did not follow
/// it, so the encoder wrote a ninth separator past its 263-byte field — an
/// index past a fixed array, in the service. The field is now pinned to the
/// depth at compile time; this is the boundary case end to end: a
/// 32-component path of 32-byte names is exactly `FS_SEARCH_PATH_BYTES`
/// long, lands in the window, and decodes back to the components.
#[test]
fn the_deepest_legal_path_crosses_the_search_wire_whole() {
    use kore_abi::svc::FS_SEARCH_PATH_BYTES;
    use kore_svc_dagfs::bounds::{MAX_NAME_LEN, MAX_PATH_DEPTH};

    let (_svc, mut clients) = world(1);
    let c = &mut clients[0];
    let root = c.open_root(Rights::ALL).unwrap();
    // Every name fills its whole field: the level number, zero-padded.
    let level_name = |level: usize| name(&format!("{level:0>width$}", width = MAX_NAME_LEN));
    let mut expected: Vec<String> = Vec::new();
    // Directories at every level but the last, then the file: 32 components.
    // A session holds at most `MAX_HANDLES_PER_SESSION` handles, which is the
    // depth bound itself, so each intermediate is released as the chain grows
    // — the walk is the chain, not the handle table.
    let mut parent = root.handle;
    let mut previous: Option<kore_svc_dagfs::Handle> = None;
    for level in 0..MAX_PATH_DEPTH - 1 {
        let made = c
            .create(parent, &level_name(level), ObjectKind::Directory)
            .unwrap();
        if let Some(done) = previous.take() {
            c.close(done).unwrap();
        }
        previous = Some(made.handle);
        parent = made.handle;
        expected.push(level_name(level).as_str().to_string());
    }
    let leaf = name(&format!("{:x>width$}", "leaf", width = MAX_NAME_LEN));
    let made = c.create(parent, &leaf, ObjectKind::File).unwrap();
    c.close(made.handle).unwrap();
    if let Some(done) = previous.take() {
        c.close(done).unwrap();
    }
    expected.push(leaf.as_str().to_string());

    let page = c
        .search(
            root.handle,
            SearchQuery {
                name_contains: Some(name("leaf")),
                kind: Some(ObjectKind::File),
                min_len: 0,
                max_len: 64,
                content: None,
                freshness: kore_svc_dagfs::FreshnessPolicy::StructuralOnly,
                max_index_lag: 0,
                max_work: 0,
            },
            None,
            0,
        )
        .unwrap();
    assert_eq!(page.hits, 1);
    let hit = read_search_hit(&c.transport().window, 0, 0).unwrap();
    assert_eq!(hit.path.len(), MAX_PATH_DEPTH);
    let got: Vec<String> = hit
        .path
        .iter()
        .map(|part| part.as_str().to_string())
        .collect();
    assert_eq!(got, expected);
    let spelled = hit
        .path
        .iter()
        .map(|part| part.as_bytes().len())
        .sum::<usize>()
        + hit.path.len()
        - 1;
    assert_eq!(spelled, FS_SEARCH_PATH_BYTES);
}

#[test]
fn a_listing_page_that_would_not_fit_the_window_is_refused_not_truncated() {
    let (_svc, mut clients) = world(1);
    let c = &mut clients[0];
    let root = c.open_root(Rights::ALL).unwrap();
    for index in 0..4 {
        let made = c
            .create(root.handle, &name(&format!("f{index}")), ObjectKind::File)
            .unwrap();
        c.close(made.handle).unwrap();
    }

    // Land the page so close to the end that fewer entries fit than the
    // directory holds. Silently returning a short page would look identical to
    // a complete one.
    let near_end = (WINDOW_BYTES - FsDirEntryWire::BYTES) as u32;
    assert_eq!(
        c.list(root.handle, 0, near_end).unwrap_err(),
        FsError::Range
    );
    // With room, the same listing succeeds.
    assert_eq!(c.list(root.handle, 0, 0).unwrap().entries, 4);
}

#[test]
fn listing_pages_through_a_large_directory() {
    let (_svc, mut clients) = world(1);
    let c = &mut clients[0];
    let root = c.open_root(Rights::ALL).unwrap();

    let total = 20usize;
    for index in 0..total {
        let made = c
            .create(root.handle, &name(&format!("f{index}")), ObjectKind::File)
            .unwrap();
        c.close(made.handle).unwrap();
    }

    let mut seen = Vec::new();
    let mut cursor = 0u32;
    loop {
        let page = c.list(root.handle, cursor, 0).unwrap();
        let window = c.transport().window.clone();
        for i in 0..page.entries as usize {
            seen.push(
                FsClient::<Loopback>::read_entry(&window, 0, i)
                    .unwrap()
                    .name
                    .as_str()
                    .to_string(),
            );
        }
        match page.next {
            Some(next) => cursor = next,
            None => break,
        }
    }
    seen.sort();
    seen.dedup();
    assert_eq!(seen.len(), total, "every entry appeared exactly once");
}

// ── isolation and sharing, over the wire ────────────────────────────────────

#[test]
fn two_sessions_have_separate_roots_across_the_wire() {
    let (_svc, mut clients) = world(2);

    let a_root = clients[0].open_root(Rights::ALL).unwrap();
    let a_file = clients[0]
        .create(a_root.handle, &name("notes"), ObjectKind::File)
        .unwrap();
    write_all(&mut clients[0], a_file.handle, b"A's bytes").unwrap();

    let b_root = clients[1].open_root(Rights::ALL).unwrap();
    let b_file = clients[1]
        .create(b_root.handle, &name("notes"), ObjectKind::File)
        .unwrap();
    write_all(&mut clients[1], b_file.handle, b"B's bytes").unwrap();

    let a_id = clients[0].stat(a_file.handle).unwrap().object;
    let b_id = clients[1].stat(b_file.handle).unwrap().object;
    assert_ne!(a_id, b_id, "same name, different objects");
    assert_eq!(
        read_all(&mut clients[0], a_file.handle).unwrap(),
        b"A's bytes"
    );
    assert_eq!(
        read_all(&mut clients[1], b_file.handle).unwrap(),
        b"B's bytes"
    );
}

#[test]
fn one_sessions_handle_is_invalid_on_another_pair() {
    // The plan's §10.2 row: a handle minted for one session must be inert when
    // presented over another session's pair.
    let (_svc, mut clients) = world(2);
    let a_root = clients[0].open_root(Rights::ALL).unwrap();
    let a_file = clients[0]
        .create(a_root.handle, &name("mine"), ObjectKind::File)
        .unwrap();

    assert_eq!(
        clients[1].stat(a_file.handle).unwrap_err(),
        FsError::InvalidHandle
    );
    assert_eq!(
        clients[1].read(a_file.handle, 0, 0, 4).unwrap_err(),
        FsError::InvalidHandle
    );
}

#[test]
fn a_grant_and_its_revocation_cross_the_wire() {
    let (_svc, mut clients) = world(2);
    let a_root = clients[0].open_root(Rights::ALL).unwrap();
    let file = clients[0]
        .create(a_root.handle, &name("shared"), ObjectKind::File)
        .unwrap();
    write_all(&mut clients[0], file.handle, b"from A").unwrap();

    let grant = clients[0]
        .grant(
            file.handle,
            SessionId::new(1),
            Rights::READ,
            Some(name("gift")),
        )
        .unwrap();

    let view = clients[1].open_grant(grant, Rights::READ).unwrap();
    assert_eq!(read_all(&mut clients[1], view.handle).unwrap(), b"from A");

    // Read-only really is read-only, through the wire.
    clients[1].transport().window[..2].copy_from_slice(b"no");
    assert_eq!(
        clients[1]
            .write_at(view.handle, 0, 0, 2, ANY_REVISION)
            .unwrap_err(),
        FsError::AccessDenied
    );

    // And revocation is immediate on the handle B already holds.
    clients[0].revoke(grant).unwrap();
    assert_eq!(
        clients[1].stat(view.handle).unwrap_err(),
        FsError::AccessLost
    );
    assert_eq!(read_all(&mut clients[0], file.handle).unwrap(), b"from A");
}

#[test]
fn losing_a_route_reaches_the_client_as_access_lost() {
    let (_svc, mut clients) = world(1);
    let c = &mut clients[0];
    let root = c.open_root(Rights::ALL).unwrap();
    let dir = c
        .create(root.handle, &name("dir"), ObjectKind::Directory)
        .unwrap();
    let file = c.create(dir.handle, &name("f"), ObjectKind::File).unwrap();
    write_all(c, file.handle, b"bytes").unwrap();

    c.unlink(dir.handle, &name("f")).unwrap();
    assert_eq!(c.stat(file.handle).unwrap_err(), FsError::AccessLost);
    assert_eq!(
        c.read(file.handle, 0, 0, 4).unwrap_err(),
        FsError::AccessLost
    );
}

// ── the protocol's own guarantees, observed from a client ───────────────────

#[test]
fn two_names_for_one_object_are_visible_as_one_object() {
    let (_svc, mut clients) = world(1);
    let c = &mut clients[0];
    let root = c.open_root(Rights::ALL).unwrap();
    let file = c
        .create(root.handle, &name("one"), ObjectKind::File)
        .unwrap();
    write_all(c, file.handle, b"shared").unwrap();
    c.link(root.handle, &name("two"), file.handle).unwrap();

    let via_two = c.open_at(root.handle, &name("two"), Rights::ALL).unwrap();
    assert_eq!(
        c.stat(file.handle).unwrap().object,
        c.stat(via_two.handle).unwrap().object
    );
    assert_eq!(c.stat(via_two.handle).unwrap().links, 2);
    assert_eq!(read_all(c, via_two.handle).unwrap(), b"shared");
}

#[test]
fn every_engine_refusal_reaches_the_client_as_itself() {
    let (_svc, mut clients) = world(1);
    let c = &mut clients[0];
    let root = c.open_root(Rights::ALL).unwrap();

    // A representative refusal for each class the client can provoke without
    // reaching into the engine.
    assert_eq!(
        c.open_at(root.handle, &name("absent"), Rights::ALL)
            .unwrap_err(),
        FsError::NotFound
    );
    let file = c.create(root.handle, &name("f"), ObjectKind::File).unwrap();
    assert_eq!(
        c.create(root.handle, &name("f"), ObjectKind::File)
            .unwrap_err(),
        FsError::AlreadyExists
    );
    assert_eq!(
        c.list(file.handle, 0, 0).unwrap_err(),
        FsError::NotDirectory
    );
    assert_eq!(
        c.read(root.handle, 0, 0, 4).unwrap_err(),
        FsError::IsDirectory
    );
    let read_only = c.open_root(Rights::READ).unwrap();
    assert_eq!(
        c.create(read_only.handle, &name("x"), ObjectKind::File)
            .unwrap_err(),
        FsError::AccessDenied
    );
    let dir = c
        .create(root.handle, &name("d"), ObjectKind::Directory)
        .unwrap();
    let inner = c.create(dir.handle, &name("i"), ObjectKind::File).unwrap();
    c.close(inner.handle).unwrap();
    assert_eq!(
        c.unlink(root.handle, &name("d")).unwrap_err(),
        FsError::NotEmpty
    );
    assert_eq!(
        c.resize(file.handle, u64::MAX, ANY_REVISION).unwrap_err(),
        FsError::OutOfStorage
    );
}

#[test]
fn a_provider_failure_crosses_as_a_provider_failure() {
    let (service, mut clients) = world(1);
    let c = &mut clients[0];
    let root = c.open_root(Rights::ALL).unwrap();
    let file = c.create(root.handle, &name("f"), ObjectKind::File).unwrap();
    write_all(c, file.handle, b"original").unwrap();

    service
        .borrow_mut()
        .engine()
        .provider()
        .inject(kore_svc_dagfs::provider::Fault::Write(ProviderError::Media));

    c.transport().window[..8].copy_from_slice(b"replaced");
    let err = c.write_at(file.handle, 0, 0, 8, ANY_REVISION).unwrap_err();
    assert!(
        matches!(err, FsError::ProviderFailure(_)),
        "a provider fault must not arrive as a capacity refusal, got {err:?}"
    );
    // And the request had no visible effect.
    assert_eq!(read_all(c, file.handle).unwrap(), b"original");
}

#[test]
fn a_malformed_record_is_refused_by_the_serving_side() {
    // Bypass the client and hand the transport a record it must refuse — the
    // path a hostile peer would take.
    let (service, _clients) = world(1);
    let mut transport = Loopback {
        service,
        session: SessionId::new(0),
        window: vec![0u8; WINDOW_BYTES],
    };
    let mut bogus = kore_svc_dagfs::wire::encode_req(&kore_svc_dagfs::FsRequest::OpenRoot {
        rights: Rights::ALL,
    });
    bogus.op = 200;
    let reply = transport.exchange(&bogus).unwrap();
    assert_ne!(
        reply.status,
        FsRepWire::STATUS_OK,
        "garbage must not succeed"
    );
}

#[test]
fn a_recipient_discovers_and_opens_a_grant_over_the_wire() {
    // The whole point of `ListGrants`: without it a founded client learns no
    // `GrantId` and the sharing half of the MVP is unreachable from a real
    // client, however well it works inside the engine.
    let (_svc, mut clients) = world(2);
    let a_root = clients[0].open_root(Rights::ALL).unwrap();
    let file = clients[0]
        .create(a_root.handle, &name("shared"), ObjectKind::File)
        .unwrap();
    write_all(&mut clients[0], file.handle, b"from A").unwrap();
    clients[0]
        .grant(
            file.handle,
            SessionId::new(1),
            Rights::READ,
            Some(name("inbox")),
        )
        .unwrap();

    // B knows nothing but its own session — it discovers the rest.
    let count = clients[1].list_grants(0).unwrap();
    assert_eq!(count, 2, "its private root plus the new grant");

    let window = clients[1].transport().window.clone();
    let entries: Vec<_> = (0..count as usize)
        .map(|i| FsClient::<Loopback>::read_grant_entry(&window, 0, i).unwrap())
        .collect();

    let root_entry = entries.iter().find(|e| e.is_private_root).unwrap();
    assert_eq!(root_entry.rights, Rights::ALL);
    assert!(root_entry.display_name.is_none());

    let gift = entries.iter().find(|e| !e.is_private_root).unwrap();
    assert_eq!(gift.rights, Rights::READ);
    assert_eq!(gift.display_name.unwrap().as_str(), "inbox");

    // And the discovered id opens.
    let view = clients[1].open_grant(gift.grant, Rights::READ).unwrap();
    assert_eq!(read_all(&mut clients[1], view.handle).unwrap(), b"from A");
}

#[test]
fn listing_grants_needs_a_window_and_respects_its_bounds() {
    let (_svc, mut clients) = world(1);
    let near_end = (WINDOW_BYTES - 8) as u32;
    assert_eq!(
        clients[0].list_grants(near_end).unwrap_err(),
        FsError::Range,
        "a page that would not fit must be refused, not truncated"
    );
    assert!(clients[0].list_grants(0).is_ok());
}
