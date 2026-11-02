// Each integration suite compiles its own copy of this module and uses the
// subset of helpers it needs, so anything unused *here* is used elsewhere.
#![allow(dead_code)]

//! Shared harness for the filesystem host test suites.
//!
//! These suites drive the engine through its **public API only** — the same
//! surface the Phase 2 service dispatch and the Phase 4 shell built-ins will
//! call.  Nothing here reaches into the engine's internals, so a suite that
//! passes is evidence about behaviour a real client can depend on rather than
//! about the current implementation's shape.

use kore_svc_dagfs::{
    ANY_REVISION, FsResult, Handle, Name, ObjectKind, RamEngine, Rights, SessionId, block_on,
};

/// A validated name, for tests where the name is obviously fine.
pub fn name(text: &str) -> Name {
    Name::new(text).expect("test names are valid")
}

/// An engine with `count` attached sessions, each with an open handle to its
/// own private root at full rights.
///
/// Returns the engine and one `(session, root handle)` pair per session, in
/// order.  Sessions get consecutive ids starting at 0, which is what the
/// founder will do when it provisions pairs.
///
/// The engine is boxed: it is tens of KiB of arenas, and an unoptimized test
/// build does not elide the moves that returning it by value implies. Deref
/// coercion means callers still write `&mut engine`.
pub fn engine_with_sessions(count: u16) -> (Box<RamEngine>, Vec<(SessionId, Handle)>) {
    let mut engine = Box::new(RamEngine::default());
    let mut clients = Vec::new();
    for index in 0..count {
        let session = SessionId::new(index);
        block_on(engine.attach_session(session)).expect("attach");
        let root = engine
            .open_root(session, Rights::ALL)
            .expect("open private root");
        clients.push((session, root.handle));
    }
    (engine, clients)
}

/// One attached session with its root handle — the common single-client case.
pub fn engine_with_one() -> (Box<RamEngine>, SessionId, Handle) {
    let (engine, clients) = engine_with_sessions(1);
    let (session, root) = clients[0];
    (engine, session, root)
}

/// Create a file beneath `parent` and write `bytes` into it, returning the
/// creator's handle to it.
pub fn create_file(
    engine: &mut RamEngine,
    session: SessionId,
    parent: Handle,
    file_name: &str,
    bytes: &[u8],
) -> Handle {
    let made = block_on(engine.create(session, parent, &name(file_name), ObjectKind::File))
        .expect("create file");
    if !bytes.is_empty() {
        block_on(engine.write_at(session, made.handle, 0, bytes, ANY_REVISION))
            .expect("write file");
    }
    made.handle
}

/// Create a file and immediately close the creator's handle.
///
/// For fixture-building loops: `create` hands back a live handle, and a
/// session's handle ceiling is deliberately lower than a directory's entry
/// ceiling, so a loop that keeps every handle it is given runs out of handles
/// long before it runs out of names.  A real client closes what it is done
/// with, and so does this.
pub fn create_file_closed(
    engine: &mut RamEngine,
    session: SessionId,
    parent: Handle,
    file_name: &str,
    bytes: &[u8],
) {
    let handle = create_file(engine, session, parent, file_name, bytes);
    engine.close(session, handle).expect("close");
}

/// Create a directory beneath `parent`, returning the creator's handle.
pub fn create_dir(
    engine: &mut RamEngine,
    session: SessionId,
    parent: Handle,
    dir_name: &str,
) -> Handle {
    block_on(engine.create(session, parent, &name(dir_name), ObjectKind::Directory))
        .expect("create directory")
        .handle
}

/// Read a whole file through `handle` into a `Vec`.
pub fn read_all(engine: &mut RamEngine, session: SessionId, handle: Handle) -> FsResult<Vec<u8>> {
    let len = engine.stat(session, handle)?.len;
    let mut buf = vec![0u8; len as usize];
    let read = block_on(engine.read(session, handle, 0, &mut buf))?;
    buf.truncate(read);
    Ok(buf)
}

/// The names in a directory, across every page of the listing.
///
/// Paging is followed to completion, so a test asserting on directory
/// contents cannot accidentally pass by looking at only the first page.
pub fn list_names(engine: &RamEngine, session: SessionId, dir: Handle) -> FsResult<Vec<String>> {
    let mut names = Vec::new();
    let mut cursor = 0u32;
    loop {
        let page = engine.list(session, dir, cursor)?;
        for entry in page.entries.iter() {
            names.push(entry.name.as_str().to_string());
        }
        match page.next {
            Some(next) => cursor = next,
            None => break,
        }
    }
    names.sort();
    Ok(names)
}
