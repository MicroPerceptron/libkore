//! A host-side driver for the filesystem **protocol** — everything crosses the
//! frozen wire records as bytes.
//!
//! Where `fsdemo` calls the engine directly, this one goes
//! client → `FsReqWire` → codec → dispatch → `FsRepWire` → client for every
//! command, over a real data window. It exists so the protocol path can be
//! driven by hand and inspected, rather than only asserted in tests.
//!
//! ```text
//! cargo run -p kore-svc-dagfs --example fswire          # interactive
//! cargo run -p kore-svc-dagfs --example fswire -- demo  # scripted
//! ```
//!
//! `bytes <command>` shows the 120-byte request record a command encodes to and
//! proves it decodes back unchanged — the quickest way to see that no address
//! and no capability rides the wire.

use std::{
    cell::RefCell,
    io::{self, BufRead, Write},
    rc::Rc,
};

use kore_abi::svc::{FsRepWire, FsReqWire};
use kore_svc_dagfs::{
    ANY_REVISION, DataWindow, FsClient, FsError, FsRequest, FsResponse, FsResult, Handle, Name,
    ObjectKind, RamService, Rights, SessionId, Transport, block_on,
    wire::{decode_req, encode_rep, encode_req},
};

const CLIENTS: u16 = 3;
const WINDOW_BYTES: usize = 4096;

type Shared = Rc<RefCell<RamService>>;

/// The client's end of a pair. `window` is the client's own memory; the service
/// sees it only through the view handed to `dispatch`.
struct Wire {
    service: Shared,
    session: SessionId,
    window: Vec<u8>,
    /// Round trips this client has made — visible in `stats`.
    calls: usize,
}

impl Transport for Wire {
    fn exchange(&mut self, request: &FsReqWire) -> FsResult<FsRepWire> {
        self.calls += 1;
        let Some(decoded) = decode_req(request) else {
            return Ok(encode_rep(&FsResponse::Refused(FsError::Unsupported), 0));
        };
        let mut window = DataWindow::new(&mut self.window);
        let response =
            self.service
                .borrow_mut()
                .dispatch(self.session, &decoded, Some(&mut window));
        Ok(encode_rep(&response, 0))
    }
}

/// A grant this driver watched being issued.
///
/// The service tracks only what a session *holds*, and a grant is held by its
/// recipient — so the issuer's copy of the id is the issuer's own business,
/// exactly as `Revoke(grant_handle)` implies. A real client stores it the same
/// way.
struct Issued {
    by: usize,
    to: u16,
    grant: kore_svc_dagfs::GrantId,
    label: String,
}

fn name(text: &str) -> FsResult<Name> {
    Name::new(text)
}

fn main() {
    let service: Shared = Rc::new(RefCell::new(RamService::default()));
    let mut clients = Vec::new();
    for index in 0..CLIENTS {
        let session = SessionId::new(index);
        block_on(service.borrow_mut().engine().attach_session(session)).expect("attach");
        let mut client = FsClient::new(Wire {
            service: Rc::clone(&service),
            session,
            window: vec![0u8; WINDOW_BYTES],
            calls: 0,
        });
        client
            .attach_window(index as u32 + 1, WINDOW_BYTES as u32)
            .expect("attach window");
        clients.push(client);
    }

    let mut issued: Vec<Issued> = Vec::new();

    if std::env::args().any(|a| a == "demo") {
        demo(&mut clients, &mut issued);
        return;
    }

    println!("kore filesystem — protocol driver (every command crosses the wire)");
    println!("{CLIENTS} clients, each with a {WINDOW_BYTES}-byte data window.");
    println!("`help` for commands, `quit` to exit.\n");

    let mut current = 0usize;
    let stdin = io::stdin();
    loop {
        print!("[wire {current}] $ ");
        let _ = io::stdout().flush();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let line = line.trim_start_matches('\u{feff}').trim();
        if line.is_empty() {
            continue;
        }
        if matches!(line, "quit" | "exit") {
            break;
        }
        if let Some(rest) = line.strip_prefix("use ") {
            match rest.trim().parse::<usize>() {
                Ok(i) if i < clients.len() => {
                    current = i;
                    println!("now session {i}");
                }
                _ => println!("usage: use <0..{}>", clients.len() - 1),
            }
            continue;
        }
        if let Err(err) = run(&mut clients, current, line, &mut issued) {
            println!("refused: {err:?}");
        }
    }
}

fn run(
    clients: &mut [FsClient<Wire>],
    current: usize,
    line: &str,
    issued: &mut Vec<Issued>,
) -> FsResult<()> {
    let (command, tail) = split(line);
    match command {
        "help" => {
            println!(
                "\
  use <n>                 speak as session n
  ls [path]               list a directory (entries cross in the data window)
  stat <path>             metadata, including the object id
  mkdir <path>            create a directory
  touch <path>            create an empty file
  write <path> <text>     overwrite from offset 0
  append <path> <text>    append (offset chosen by the service)
  cat <path>              read the whole file back through the window
  ln <path> <new>         a second name for one object
  mv <path> <new>         move: retires the source edge
  rm <path>               remove one name
  share <path> <n> <ro|rw> [as]   grant to session n
  grants                  list this session's authority edges
  revoke <index>          retire a grant (issuer or holder)
  bytes <command...>      show the encoded request record for a command
  stats                   round trips, storage, arenas
  quit"
            );
            Ok(())
        }
        "bytes" => show_bytes(clients, current, tail),
        "ls" => with_resolved(clients, current, first(tail), |clients, dir| {
            let client = &mut clients[current];
            let mut cursor = 0u32;
            let mut shown = 0;
            loop {
                let page = client.list(dir, cursor, 0)?;
                let window = client.transport().window.clone();
                for i in 0..page.entries as usize {
                    let e = FsClient::<Wire>::read_entry(&window, 0, i)?;
                    let mark = if e.kind == ObjectKind::Directory {
                        "/"
                    } else {
                        ""
                    };
                    println!("  {}{}", e.name.as_str(), mark);
                    shown += 1;
                }
                match page.next {
                    Some(next) => cursor = next,
                    None => break,
                }
            }
            if shown == 0 {
                println!("  (empty)");
            }
            Ok(())
        }),
        "stat" => with_resolved(clients, current, first(tail), |clients, h| {
            let s = clients[current].stat(h)?;
            println!(
                "  {:?}  len={}  revision={}  names={}  grants={}  object={:#x}",
                s.kind,
                s.len,
                s.revision,
                s.links,
                s.grants,
                s.object.to_bits()
            );
            Ok(())
        }),
        "mkdir" => create(clients, current, first(tail), ObjectKind::Directory),
        "touch" => create(clients, current, first(tail), ObjectKind::File),
        "cat" => with_resolved(clients, current, first(tail), |clients, h| {
            let client = &mut clients[current];
            let len = client.stat(h)?.len as u32;
            let count = client.read(h, 0, 0, len)?;
            let bytes = client.transport().window[..count as usize].to_vec();
            println!("{}", String::from_utf8_lossy(&bytes));
            Ok(())
        }),
        "write" | "append" => {
            let (path, text) = split(tail);
            if path.is_empty() {
                println!("usage: {command} <path> <text...>");
                return Ok(());
            }
            let bytes = text.as_bytes();
            if bytes.len() > WINDOW_BYTES {
                return Err(FsError::Range);
            }
            let append = command == "append";
            // `write` on a name that does not exist creates it, then retries.
            // Retry rather than probe-then-write: `put` has already returned
            // its handle by the time this matches, so the fallback costs no
            // handles and the common case stays one walk.
            match put(clients, current, path, bytes, append) {
                Err(FsError::NotFound) if !append => {
                    create(clients, current, path, ObjectKind::File)?;
                    put(clients, current, path, bytes, append)
                }
                other => other,
            }
        }
        "rm" => with_parent(clients, current, first(tail), |clients, parent, leaf| {
            clients[current].unlink(parent, &leaf)
        }),
        "ln" | "mv" => {
            let (src, dst) = split(tail);
            if src.is_empty() || dst.is_empty() {
                println!("usage: {command} <path> <new-path>");
                return Ok(());
            }
            // Nested guards: `mv` holds three handles at once (both parents
            // plus the walk's), and every one is released on the way out.
            with_parent(clients, current, dst, |clients, dst_parent, dst_name| {
                if command == "ln" {
                    with_resolved(clients, current, src, |clients, file| {
                        clients[current].link(dst_parent, &dst_name, file)
                    })
                } else {
                    with_parent(clients, current, src, |clients, src_parent, src_name| {
                        clients[current].relocate(src_parent, &src_name, dst_parent, &dst_name)
                    })
                }
            })
        }
        "share" => {
            let args: Vec<&str> = tail.split_whitespace().collect();
            if args.len() < 3 {
                println!("usage: share <path> <session> <ro|rw> [name]");
                return Ok(());
            }
            let Ok(dest) = args[1].parse::<u16>() else {
                println!("session must be a number");
                return Ok(());
            };
            let rights = match args[2] {
                "ro" => Rights::READ,
                "rw" => Rights::READ
                    .union(Rights::WRITE)
                    .union(Rights::MUTATE_NAMESPACE),
                other => {
                    println!("unknown rights {other:?}");
                    return Ok(());
                }
            };
            let display = match args.get(3) {
                Some(t) => Some(name(t)?),
                None => None,
            };
            with_resolved(clients, current, args[0], |clients, h| {
                let grant = clients[current].grant(h, SessionId::new(dest), rights, display)?;
                let label = display
                    .map(|n| format!("@{}", n.as_str()))
                    .unwrap_or_else(|| "<unnamed>".into());
                println!(
                    "  granted to session {dest} as {label} ({})",
                    describe(rights)
                );
                issued.push(Issued {
                    by: current,
                    to: dest,
                    grant,
                    label,
                });
                Ok(())
            })
        }
        "grants" => {
            let client = &mut clients[current];
            let count = client.list_grants(0)?;
            let window = client.transport().window.clone();
            for i in 0..count as usize {
                let e = FsClient::<Wire>::read_grant_entry(&window, 0, i)?;
                let label = e
                    .display_name
                    .map(|n| format!("@{}", n.as_str()))
                    .unwrap_or_else(|| "<unnamed>".into());
                let kind = if e.is_private_root {
                    "private root"
                } else {
                    "grant"
                };
                println!("    [{i}] {kind:<13} {label:<14} {}", describe(e.rights));
            }
            let held = count as usize;
            for (offset, e) in issued.iter().filter(|e| e.by == current).enumerate() {
                println!(
                    "    [{}] {:<13} {:<14} issued to session {}",
                    held + offset,
                    "grant",
                    e.label,
                    e.to
                );
            }
            println!("  reach an attachment with `@name/file`");
            Ok(())
        }
        "revoke" => {
            let Ok(index) = first(tail).parse::<usize>() else {
                println!("usage: revoke <index from `grants`>");
                return Ok(());
            };
            let client = &mut clients[current];
            let count = client.list_grants(0)?;
            let window = client.transport().window.clone();
            let grant = if index < count as usize {
                FsClient::<Wire>::read_grant_entry(&window, 0, index)?.grant
            } else {
                // Above the held range, it names one this session issued.
                let mine: Vec<&Issued> = issued.iter().filter(|e| e.by == current).collect();
                match mine.get(index - count as usize) {
                    Some(e) => e.grant,
                    None => {
                        println!("no such index");
                        return Ok(());
                    }
                }
            };
            client.revoke(grant)?;
            issued.retain(|e| e.grant != grant);
            Ok(())
        }
        "stats" => {
            for (i, client) in clients.iter_mut().enumerate() {
                println!("  session {i}: {} round trips", client.transport().calls);
            }
            println!("  request record {} bytes", size_of::<FsReqWire>());
            println!("  reply record   {} bytes", size_of::<FsRepWire>());
            Ok(())
        }
        other => {
            println!("unknown command {other:?} — try `help`");
            Ok(())
        }
    }
}

/// Encode a command's request and show the bytes, then decode it back.
fn show_bytes(clients: &mut [FsClient<Wire>], current: usize, tail: &str) -> FsResult<()> {
    use zerocopy::IntoBytes;
    let (command, rest) = split(tail);
    // A representative request per verb, built without touching the service.
    // The handles this representative request borrowed, returned once the
    // encoding has been shown. `bytes` only INSPECTS a record, so it must cost
    // the session nothing — before this it silently spent a handle per call.
    let (request, held): (FsRequest, Vec<Handle>) = match command {
        "ls" => {
            let handle = resolve(clients, current, first(rest))?;
            (
                FsRequest::List {
                    handle,
                    cursor: 0,
                    window_offset: 0,
                },
                vec![handle],
            )
        }
        "stat" => {
            let handle = resolve(clients, current, first(rest))?;
            (FsRequest::Stat { handle }, vec![handle])
        }
        "cat" => {
            let handle = resolve(clients, current, first(rest))?;
            (
                FsRequest::Read {
                    handle,
                    offset: 0,
                    window_offset: 0,
                    length: 64,
                },
                vec![handle],
            )
        }
        "touch" => {
            let (parent, leaf) = resolve_parent(clients, current, first(rest))?;
            (
                FsRequest::Create {
                    parent,
                    name: leaf,
                    kind: ObjectKind::File,
                    subtype: kore_svc_dagfs::ObjectSubtype::PLAIN,
                },
                vec![parent],
            )
        }
        _ => {
            println!("usage: bytes <ls|stat|cat|touch> <path>");
            return Ok(());
        }
    };
    let wire = encode_req(&request);
    let bytes = wire.as_bytes();
    println!("  {request:?}");
    for (row, chunk) in bytes.chunks(16).enumerate() {
        let hex: Vec<String> = chunk.iter().map(|b| format!("{b:02x}")).collect();
        println!("  {:04x}  {}", row * 16, hex.join(" "));
    }
    match decode_req(&wire) {
        Some(back) if back == request => println!("  decodes back unchanged ✓"),
        Some(_) => println!("  DECODED TO SOMETHING ELSE ✗"),
        None => println!("  REFUSED ITS OWN ENCODING ✗"),
    }
    for handle in held {
        shut(clients, current, handle);
    }
    Ok(())
}

// ── client-side path walking (the engine takes one component at a time) ─────

/// Where a resolution starts, and **with what authority** — the second half is
/// what this driver used to throw away.
///
/// Every component was opened with `Rights::ALL`, which made any path inside a
/// read-only grant unreachable: the grant root opens at the grant's own rights,
/// so `stat @g` and `ls @g` worked, but `cat @g/f` demanded full rights one
/// step in and the engine — correctly — refused. Attenuation was
/// indistinguishable from a bug.
///
/// The fix is `fsdemo`'s: carry the anchor's rights and walk with them
/// unchanged. A private root yields `Rights::ALL`; a grant yields exactly what
/// was delegated, so the walk can never out-run the authority it started from,
/// and a refusal is the engine's real answer rather than this driver over-asking.
struct Anchor {
    handle: Handle,
    rights: Rights,
}

fn anchor(clients: &mut [FsClient<Wire>], current: usize, path: &str) -> FsResult<Anchor> {
    if let Some(rest) = path.strip_prefix('@') {
        let wanted = rest.split('/').next().unwrap_or("");
        let client = &mut clients[current];
        let count = client.list_grants(0)?;
        let window = client.transport().window.clone();
        for i in 0..count as usize {
            let e = FsClient::<Wire>::read_grant_entry(&window, 0, i)?;
            if !e.is_private_root
                && e.display_name
                    .map(|n| n.as_str() == wanted)
                    .unwrap_or(false)
            {
                // The grant's OWN rights, which then bound the whole walk.
                return Ok(Anchor {
                    handle: client.open_grant(e.grant, e.rights)?.handle,
                    rights: e.rights,
                });
            }
        }
        return Err(FsError::NotFound);
    }
    Ok(Anchor {
        handle: clients[current].open_root(Rights::ALL)?.handle,
        rights: Rights::ALL,
    })
}

fn components(path: &str) -> Vec<&str> {
    let path = path.strip_prefix('@').map_or(path, |r| {
        r.split_once('/').map(|(_, rest)| rest).unwrap_or("")
    });
    path.split('/').filter(|p| !p.is_empty()).collect()
}

/// Hand a handle back, swallowing the outcome.
///
/// Every handle this driver opens must be returned, including on the error
/// paths — a leak is permanent, and `MAX_HANDLES_PER_SESSION` is 16, so a
/// session that leaks a few per command wedges on `OutOfHandles` after a
/// handful of lines. This driver used to do exactly that: a `Close` request
/// was never sent on any path, so `mkdir d` + `touch d/f` + three appends
/// exhausted session 0 and every later command refused.
///
/// The result is dropped deliberately: a failed close tells a development
/// tool nothing it can act on, and the caller's original error is the one
/// worth reporting.
fn shut(clients: &mut [FsClient<Wire>], current: usize, handle: Handle) {
    let _ = clients[current].close(handle);
}

/// Resolve `path`, lend the handle to `f`, and close it either way.
///
/// A scope guard rather than a close at the end of each arm, because the
/// interesting outcomes here leave early with `?`: `AccessDenied`,
/// `NotFound`, `Conflict`. Those are precisely the paths a protocol driver is
/// used to explore, so they are the paths that must not leak.
fn with_resolved<R>(
    clients: &mut [FsClient<Wire>],
    current: usize,
    path: &str,
    f: impl FnOnce(&mut [FsClient<Wire>], Handle) -> FsResult<R>,
) -> FsResult<R> {
    let handle = resolve(clients, current, path)?;
    let out = f(clients, handle);
    shut(clients, current, handle);
    out
}

/// [`with_resolved`]'s parent form, for the verbs that name a leaf.
fn with_parent<R>(
    clients: &mut [FsClient<Wire>],
    current: usize,
    path: &str,
    f: impl FnOnce(&mut [FsClient<Wire>], Handle, Name) -> FsResult<R>,
) -> FsResult<R> {
    let (parent, leaf) = resolve_parent(clients, current, path)?;
    let out = f(clients, parent, leaf);
    shut(clients, current, parent);
    out
}

/// Walk `parts` from `at`, closing each handle stepped OFF.
///
/// Dropping the parent is safe: a handle is authority over one object, and the
/// engine re-walks a handle's proof on every call, so a child handle does not
/// depend on its parent's staying open. Both exits close the handle in hand,
/// so no path through here leaks.
/// Every component opens at the ANCHOR's rights, unchanged — `fsdemo`'s rule.
fn walk(
    clients: &mut [FsClient<Wire>],
    current: usize,
    at: Anchor,
    parts: &[&str],
) -> FsResult<Handle> {
    let Anchor { handle, rights } = at;
    let mut at = handle;
    for &part in parts {
        let n = match name(part) {
            Ok(n) => n,
            Err(e) => {
                shut(clients, current, at);
                return Err(e);
            }
        };
        match clients[current].open_at(at, &n, rights) {
            Ok(opened) => {
                shut(clients, current, at);
                at = opened.handle;
            }
            Err(e) => {
                shut(clients, current, at);
                return Err(e);
            }
        }
    }
    Ok(at)
}

fn resolve(clients: &mut [FsClient<Wire>], current: usize, path: &str) -> FsResult<Handle> {
    let at = anchor(clients, current, path)?;
    walk(clients, current, at, &components(path))
}

fn resolve_parent(
    clients: &mut [FsClient<Wire>],
    current: usize,
    path: &str,
) -> FsResult<(Handle, Name)> {
    let mut parts = components(path);
    let last = parts.pop().ok_or(FsError::InvalidName)?;
    let leaf = name(last)?;
    let at = anchor(clients, current, path)?;
    let parent = walk(clients, current, at, &parts)?;
    Ok((parent, leaf))
}

fn create(
    clients: &mut [FsClient<Wire>],
    current: usize,
    path: &str,
    kind: ObjectKind,
) -> FsResult<()> {
    with_parent(clients, current, path, |clients, parent, leaf| {
        // `create` HANDS BACK a handle to the new object. Dropping it leaked
        // one per `mkdir`/`touch` on top of the walk's own leaks.
        let opened = clients[current].create(parent, &leaf, kind)?;
        shut(clients, current, opened.handle);
        Ok(())
    })
}

/// `write`/`append`'s shared body — named so the create-then-retry path below
/// reuses it instead of duplicating the request.
fn put(
    clients: &mut [FsClient<Wire>],
    current: usize,
    path: &str,
    bytes: &[u8],
    append: bool,
) -> FsResult<()> {
    with_resolved(clients, current, path, |clients, h| {
        let client = &mut clients[current];
        client.transport().window[..bytes.len()].copy_from_slice(bytes);
        if append {
            let (off, rev) = client.append(h, 0, bytes.len() as u32, ANY_REVISION)?;
            println!("  {} bytes at {off}; revision {rev}", bytes.len());
        } else {
            let rev = client.write_at(h, 0, 0, bytes.len() as u32, ANY_REVISION)?;
            println!("  {} bytes at 0; revision {rev}", bytes.len());
        }
        Ok(())
    })
}

fn describe(rights: Rights) -> String {
    let mut parts = Vec::new();
    for (bit, label) in [
        (Rights::READ, "read"),
        (Rights::WRITE, "write"),
        (Rights::MUTATE_NAMESPACE, "namespace"),
        (Rights::SHARE, "share"),
    ] {
        if rights.contains(bit) {
            parts.push(label);
        }
    }
    if parts.is_empty() {
        "none".into()
    } else {
        parts.join("+")
    }
}

fn first(tail: &str) -> &str {
    tail.split_whitespace().next().unwrap_or("")
}

fn split(line: &str) -> (&str, &str) {
    match line.split_once(char::is_whitespace) {
        Some((head, rest)) => (head, rest.trim_start()),
        None => (line, ""),
    }
}

/// A scripted run showing the protocol path end to end.
fn demo(clients: &mut [FsClient<Wire>], issued: &mut Vec<Issued>) {
    let script = [
        (0usize, "touch hello"),
        (0, "write hello kore over the wire"),
        (0, "cat hello"),
        (0, "stat hello"),
        (0, "bytes stat hello"),
        (0, "ln hello second-name"),
        (0, "ls"),
        (0, "share hello 1 ro inbox"),
        (0, "grants"),
        (1, "grants"),
        (1, "cat @inbox"),
        (1, "write @inbox nope"),
        (0, "revoke 1"),
        (1, "cat @inbox"),
        (0, "stats"),
    ];
    for (session, line) in script {
        println!("\n[session {session}] $ {line}");
        if let Err(err) = run(clients, session, line, issued) {
            println!("refused: {err:?}");
        }
    }
    println!("\nEvery line above crossed the frozen 120-byte request record.\n");
}
