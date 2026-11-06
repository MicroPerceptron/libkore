//! A host-side interactive driver for the filesystem engine.
//!
//! Phase 1 is the engine only — there is no service pair, no founded client,
//! and no shell yet — so this is how a human explores the same behaviour the
//! test suites assert. It is a **development tool**, not part of the service:
//! it lives in `examples/`, uses `std`, and nothing in the crate depends on
//! it. The real thing arrives in Phase 4 as shell built-ins over the same
//! public API this file calls.
//!
//! Run it:
//!
//! ```text
//! cargo run -p kore-svc-dagfs --example fsdemo          # interactive
//! cargo run -p kore-svc-dagfs --example fsdemo -- tour  # scripted walkthrough
//! ```
//!
//! Three sessions are attached at startup, each with its own private root, so
//! isolation and sharing are both demonstrable. `use 1` switches which client
//! you are speaking as — that is the whole point: the same path text means
//! different things to different sessions.
//!
//! Note what this file has to do that the engine deliberately does not:
//! **it walks paths itself**. The engine's vocabulary is single-component
//! (`open_at(parent, name)`), because a path is client-side input and not
//! authority. Every command here re-walks its path from a root each time,
//! which is exactly how a real client behaves.

use std::io::{self, BufRead, Write};

use kore_svc_dagfs::{
    ANY_REVISION, AuthorityKind, FsError, GrantId, Handle, Name, ObjectKind, RamEngine, Rights,
    SessionId, StorageProvider, block_on,
};

/// How many private-root clients the demo attaches.
const CLIENTS: u16 = 3;

/// One client's view: which session it is, and the handle to its own root.
struct Client {
    session: SessionId,
    root: Handle,
}

/// A grant this demo watched being issued.
///
/// The service deliberately does not keep this list. `Engine::grants` reports
/// what a session **holds**, and a grant is held by its *recipient* — so the
/// issuer's copy of the `GrantId` is the issuer's own business, exactly as
/// plan §5's `Revoke(grant_handle)` implies. A real client would store it the
/// same way this demo does.
struct Issued {
    by: u16,
    to: u16,
    grant: GrantId,
    label: String,
}

/// Where a resolution starts, and with what authority.
struct Anchor {
    handle: Handle,
    /// `true` if the caller must close it (an attachment root we opened).
    owned: bool,
    rights: Rights,
}

fn main() {
    let mut engine = Box::new(RamEngine::default());
    let mut clients = Vec::new();
    for index in 0..CLIENTS {
        let session = SessionId::new(index);
        block_on(engine.attach_session(session)).expect("attach session");
        let root = engine
            .open_root(session, Rights::ALL)
            .expect("open private root")
            .handle;
        clients.push(Client { session, root });
    }

    let mut issued: Vec<Issued> = Vec::new();

    if std::env::args().any(|arg| arg == "tour") {
        tour(&mut engine, &clients, &mut issued);
        return;
    }

    println!("kore filesystem engine — interactive demo (RAM provider)");
    println!(
        "{} clients attached, each with its own empty private root.",
        CLIENTS
    );
    println!("`help` for commands, `tour` for a guided walkthrough, `quit` to exit.\n");

    let mut current = 0usize;
    let stdin = io::stdin();
    loop {
        print!("[session {current}] $ ");
        let _ = io::stdout().flush();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(err) => {
                eprintln!("input error: {err}");
                break;
            }
        }
        // Strip a UTF-8 BOM before trimming: PowerShell prepends one when a
        // here-string is piped into a process, and it would otherwise glue
        // itself to the first command word.
        let line = line.trim_start_matches('\u{feff}').trim();
        if line.is_empty() {
            continue;
        }
        if matches!(line, "quit" | "exit") {
            break;
        }
        if let Some(rest) = line.strip_prefix("use") {
            match rest.trim().parse::<usize>() {
                Ok(index) if index < clients.len() => {
                    current = index;
                    println!("now speaking as session {index}");
                }
                _ => println!("usage: use <0..{}>", clients.len() - 1),
            }
            continue;
        }
        if line == "tour" {
            tour(&mut engine, &clients, &mut issued);
            continue;
        }
        if let Err(err) = run(&mut engine, &clients, current, line, &mut issued) {
            println!("refused: {err:?}");
        }
    }
}

/// Dispatch one command line.
fn run(
    engine: &mut RamEngine,
    clients: &[Client],
    current: usize,
    line: &str,
    issued: &mut Vec<Issued>,
) -> Result<(), FsError> {
    let client = &clients[current];
    let mut parts = line.split_whitespace();
    let command = parts.next().unwrap_or("");
    // Everything after the command word, for commands taking free text.
    let tail = line[command.len()..].trim();

    match command {
        "help" => {
            print_help();
            Ok(())
        }
        "ls" => {
            let path = first_arg(tail);
            let (handle, owned) = resolve(engine, client, path)?;
            let result = list(engine, client, handle);
            release(engine, client, handle, owned);
            result
        }
        "stat" => {
            let (handle, owned) = resolve(engine, client, first_arg(tail))?;
            let result = engine.stat(client.session, handle);
            release(engine, client, handle, owned);
            let info = result?;
            println!(
                "  {:?}  len={}  revision={}  names={}  grants={}  object={:#x}",
                info.kind,
                info.len,
                info.revision,
                info.links,
                info.grants,
                info.object.to_bits()
            );
            Ok(())
        }
        "mkdir" => create_at(engine, client, tail, ObjectKind::Directory),
        "touch" => create_at(engine, client, tail, ObjectKind::File),
        "cat" => {
            let (handle, owned) = resolve(engine, client, first_arg(tail))?;
            let result = read_all(engine, client, handle);
            release(engine, client, handle, owned);
            match result {
                Ok(bytes) => {
                    println!("{}", String::from_utf8_lossy(&bytes));
                    Ok(())
                }
                Err(err) => Err(err),
            }
        }
        "write" | "append" => {
            let (path, text) = split_two(tail);
            if path.is_empty() {
                println!("usage: {command} <path> <text...>");
                return Ok(());
            }
            // `write` behaves like a shell redirect and creates the file if it
            // is not there yet — which is what plan §10.3's interactive
            // sequence assumes. `append` requires an existing file.
            let (handle, owned) = match resolve(engine, client, path) {
                Ok(found) => found,
                Err(FsError::NotFound) if command == "write" => {
                    let (parent, parent_owned, name) = resolve_parent(engine, client, path)?;
                    let made =
                        block_on(engine.create(client.session, parent, &name, ObjectKind::File));
                    release(engine, client, parent, parent_owned);
                    (made?.handle, true)
                }
                Err(err) => return Err(err),
            };
            let result = if command == "write" {
                block_on(engine.write_at(client.session, handle, 0, text.as_bytes(), ANY_REVISION))
                    .map(|revision| (0u64, revision))
            } else {
                block_on(engine.append(client.session, handle, text.as_bytes(), ANY_REVISION))
            };
            release(engine, client, handle, owned);
            let (offset, revision) = result?;
            println!(
                "  {} bytes at offset {offset}; revision {revision}",
                text.len()
            );
            Ok(())
        }
        "rm" => {
            let path = first_arg(tail);
            let (parent, owned, name) = resolve_parent(engine, client, path)?;
            let result = block_on(engine.unlink(client.session, parent, &name));
            release(engine, client, parent, owned);
            result
        }
        "ln" => {
            let (src, dst) = split_two(tail);
            if src.is_empty() || dst.is_empty() {
                println!("usage: ln <existing-path> <new-path>");
                return Ok(());
            }
            let (file, file_owned) = resolve(engine, client, src)?;
            let parent = resolve_parent(engine, client, dst);
            let result = match parent {
                Ok((parent, parent_owned, name)) => {
                    let outcome = engine.link(client.session, parent, &name, file);
                    release(engine, client, parent, parent_owned);
                    outcome
                }
                Err(err) => Err(err),
            };
            release(engine, client, file, file_owned);
            result
        }
        "mv" => {
            let (src, dst) = split_two(tail);
            if src.is_empty() || dst.is_empty() {
                println!("usage: mv <path> <new-path>");
                return Ok(());
            }
            let (src_parent, src_owned, src_name) = resolve_parent(engine, client, src)?;
            let result = match resolve_parent(engine, client, dst) {
                Ok((dst_parent, dst_owned, dst_name)) => {
                    let outcome = engine.relocate(
                        client.session,
                        src_parent,
                        &src_name,
                        dst_parent,
                        &dst_name,
                    );
                    release(engine, client, dst_parent, dst_owned);
                    outcome
                }
                Err(err) => Err(err),
            };
            release(engine, client, src_parent, src_owned);
            result
        }
        "share" => {
            // share <path> <session> <ro|rw> [name]
            let args: Vec<&str> = tail.split_whitespace().collect();
            if args.len() < 3 {
                println!("usage: share <path> <session> <ro|rw> [attachment-name]");
                return Ok(());
            }
            let Ok(destination) = args[1].parse::<u16>() else {
                println!("session must be a number");
                return Ok(());
            };
            let rights = match args[2] {
                "ro" => Rights::READ,
                "rw" => Rights::READ
                    .union(Rights::WRITE)
                    .union(Rights::MUTATE_NAMESPACE),
                other => {
                    println!("unknown rights {other:?} — use `ro` or `rw`");
                    return Ok(());
                }
            };
            let display = match args.get(3) {
                Some(text) => Some(Name::new(text)?),
                None => None,
            };
            let (handle, owned) = resolve(engine, client, args[0])?;
            let result = engine.grant(
                client.session,
                handle,
                SessionId::new(destination),
                rights,
                display,
            );
            release(engine, client, handle, owned);
            let grant = result?;
            let label = display
                .map(|n| format!("@{}", n.as_str()))
                .unwrap_or_else(|| "<unnamed>".into());
            println!(
                "  granted to session {destination} as {label} ({})",
                describe_rights(rights)
            );
            // Keep the issuer's copy of the id — this is the only thing that
            // lets it revoke later.
            issued.push(Issued {
                by: client.session.index(),
                to: destination,
                grant,
                label,
            });
            Ok(())
        }
        "grants" => {
            for (index, (_, description)) in routes(engine, client, issued).iter().enumerate() {
                println!("    [{index}] {description}");
            }
            println!("  reach an attachment with a path like `@name/file`");
            Ok(())
        }
        "revoke" => {
            let Ok(index) = first_arg(tail).parse::<usize>() else {
                println!("usage: revoke <index from `grants`>");
                return Ok(());
            };
            let routes = routes(engine, client, issued);
            let Some((grant, description)) = routes.get(index) else {
                println!("no such index — run `grants`");
                return Ok(());
            };
            let grant = *grant;
            println!("  retiring {description}");
            let outcome = block_on(engine.revoke(client.session, grant));
            if outcome.is_ok() {
                issued.retain(|entry| entry.grant != grant);
            }
            outcome
        }
        "stats" => {
            let limits = engine.provider().limits();
            println!("  objects        {}", engine.object_count());
            println!("  directories    {}", engine.directory_count());
            println!("  authority edges{:>3}", engine.authority_edge_count());
            println!(
                "  storage        {} / {} bytes in {} placements",
                engine.provider().bytes_in_use(),
                limits.total_bytes,
                engine.provider().placements()
            );
            println!("  leaked         {}", engine.leaked_placements());
            for client in 0..CLIENTS {
                println!(
                    "  session {client} handles {}",
                    engine.live_handles(SessionId::new(client))
                );
            }
            Ok(())
        }
        other => {
            println!("unknown command {other:?} — try `help`");
            Ok(())
        }
    }
}

fn print_help() {
    println!(
        "\
  use <n>                     speak as session n (each has its own private root)
  ls [path]                   list a directory
  stat <path>                 object metadata, including its identity
  mkdir <path>                create a directory
  touch <path>                create an empty file
  write <path> <text>         overwrite a file from offset 0
  append <path> <text>        append (offset chosen atomically)
  cat <path>                  read a whole file
  ln <path> <new-path>        a second name for the same file object
  mv <path> <new-path>        move: retires the source edge
  rm <path>                   remove one name
  share <path> <n> <ro|rw> [as]   grant a file or subtree to session n
  grants                      list this session's authority edges
  revoke <index>              retire a grant
  stats                       arena and storage accounting
  tour                        scripted walkthrough of the guarantees
  quit

  paths are client-side: `a/b/c` walks from your own root, `@name/x` walks
  from an attachment someone granted you."
    );
}

/// Every authority edge this session can act on, as `(id, description)`.
///
/// Two sources, deliberately: the ones it **holds** come from the service
/// (`Engine::grants`), and the ones it **issued** come from the demo's own
/// registry, because the service does not track an issuer's copy. `revoke`
/// indexes this combined list, so either side of a grant can end it — which is
/// exactly the rule the engine enforces.
fn routes(engine: &RamEngine, client: &Client, issued: &[Issued]) -> Vec<(GrantId, String)> {
    let mut out = Vec::new();
    for info in engine
        .grants(client.session)
        .expect("attached session grant snapshot")
        .iter()
    {
        let label = match info.display_name {
            Some(name) => format!("@{}", name.as_str()),
            None => "<unnamed>".to_string(),
        };
        let kind = match info.kind {
            AuthorityKind::PrivateRoot => "hold: private root",
            AuthorityKind::Grant => "hold: grant",
        };
        out.push((
            info.grant,
            format!("{kind:<19} {label:<14} {}", describe_rights(info.rights)),
        ));
    }
    for entry in issued.iter().filter(|e| e.by == client.session.index()) {
        out.push((
            entry.grant,
            format!(
                "{:<19} {:<14} to session {}",
                "issued: grant", entry.label, entry.to
            ),
        ));
    }
    out
}

// ── path resolution (client-side, by design) ────────────────────────────────

/// Where to start resolving `path`, and the components still to walk.
fn anchor<'a>(
    engine: &mut RamEngine,
    client: &Client,
    path: &'a str,
) -> Result<(Anchor, Vec<&'a str>), FsError> {
    if let Some(stripped) = path.strip_prefix('@') {
        let mut split = stripped.splitn(2, '/');
        let wanted = split.next().unwrap_or("");
        let rest = split.next().unwrap_or("");
        let listed = engine.grants(client.session)?;
        let found = listed.iter().find(|info| {
            info.kind == AuthorityKind::Grant
                && info
                    .display_name
                    .map(|n| n.as_str() == wanted)
                    .unwrap_or(false)
        });
        let Some(info) = found else {
            return Err(FsError::NotFound);
        };
        let opened = engine.open_grant(client.session, info.grant, info.rights)?;
        Ok((
            Anchor {
                handle: opened.handle,
                owned: true,
                rights: info.rights,
            },
            components(rest),
        ))
    } else {
        Ok((
            Anchor {
                handle: client.root,
                owned: false,
                rights: Rights::ALL,
            },
            components(path),
        ))
    }
}

fn components(path: &str) -> Vec<&str> {
    path.split('/').filter(|part| !part.is_empty()).collect()
}

/// Resolve `path` to a handle. The `bool` says whether the caller must close it.
fn resolve(engine: &mut RamEngine, client: &Client, path: &str) -> Result<(Handle, bool), FsError> {
    let (mut at, parts) = anchor(engine, client, path)?;
    for part in parts {
        let name = match Name::new(part) {
            Ok(name) => name,
            Err(err) => {
                release(engine, client, at.handle, at.owned);
                return Err(err);
            }
        };
        match engine.open_at(client.session, at.handle, &name, at.rights) {
            Ok(next) => {
                release(engine, client, at.handle, at.owned);
                at.handle = next.handle;
                at.owned = true;
            }
            Err(err) => {
                release(engine, client, at.handle, at.owned);
                return Err(err);
            }
        }
    }
    Ok((at.handle, at.owned))
}

/// Resolve `path`'s parent directory plus its final component.
fn resolve_parent(
    engine: &mut RamEngine,
    client: &Client,
    path: &str,
) -> Result<(Handle, bool, Name), FsError> {
    let (at, mut parts) = anchor(engine, client, path)?;
    let Some(last) = parts.pop() else {
        release(engine, client, at.handle, at.owned);
        return Err(FsError::InvalidName);
    };
    let name = match Name::new(last) {
        Ok(name) => name,
        Err(err) => {
            release(engine, client, at.handle, at.owned);
            return Err(err);
        }
    };
    // Walk whatever is left of the path to reach the parent.
    let mut current = at;
    for part in parts {
        let component = match Name::new(part) {
            Ok(component) => component,
            Err(err) => {
                release(engine, client, current.handle, current.owned);
                return Err(err);
            }
        };
        match engine.open_at(client.session, current.handle, &component, current.rights) {
            Ok(next) => {
                release(engine, client, current.handle, current.owned);
                current.handle = next.handle;
                current.owned = true;
            }
            Err(err) => {
                release(engine, client, current.handle, current.owned);
                return Err(err);
            }
        }
    }
    Ok((current.handle, current.owned, name))
}

/// Close a handle if we opened it. A session's own root handle is long-lived
/// and must survive; everything we walked through is ours to release.
fn release(engine: &mut RamEngine, client: &Client, handle: Handle, owned: bool) {
    if owned {
        let _ = engine.close(client.session, handle);
    }
}

// ── command bodies ──────────────────────────────────────────────────────────

fn create_at(
    engine: &mut RamEngine,
    client: &Client,
    path: &str,
    kind: ObjectKind,
) -> Result<(), FsError> {
    let path = first_arg(path);
    let (parent, owned, name) = resolve_parent(engine, client, path)?;
    let result = block_on(engine.create(client.session, parent, &name, kind));
    release(engine, client, parent, owned);
    let made = result?;
    // The creator is handed working authority immediately; this demo has no
    // further use for it, so it closes it rather than leaking a handle.
    let _ = engine.close(client.session, made.handle);
    Ok(())
}

fn list(engine: &mut RamEngine, client: &Client, handle: Handle) -> Result<(), FsError> {
    let mut cursor = 0u32;
    let mut shown = 0usize;
    loop {
        let page = engine.list(client.session, handle, cursor)?;
        for entry in page.entries.iter() {
            let marker = if entry.kind == ObjectKind::Directory {
                "/"
            } else {
                ""
            };
            println!("  {}{}", entry.name.as_str(), marker);
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
}

fn read_all(engine: &mut RamEngine, client: &Client, handle: Handle) -> Result<Vec<u8>, FsError> {
    let len = engine.stat(client.session, handle)?.len as usize;
    let mut buf = vec![0u8; len];
    let read = block_on(engine.read(client.session, handle, 0, &mut buf))?;
    buf.truncate(read);
    Ok(buf)
}

fn describe_rights(rights: Rights) -> String {
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
        "none".to_string()
    } else {
        parts.join("+")
    }
}

fn first_arg(tail: &str) -> &str {
    tail.split_whitespace().next().unwrap_or("")
}

/// Split into the first whitespace-delimited word and the rest, verbatim.
fn split_two(tail: &str) -> (&str, &str) {
    match tail.split_once(char::is_whitespace) {
        Some((head, rest)) => (head, rest.trim_start()),
        None => (tail, ""),
    }
}

// ── the guided walkthrough ──────────────────────────────────────────────────

/// Run through the properties the MVP exists to demonstrate, printing what
/// happened at each step.
///
/// This is the same ground the test suites cover, arranged to be *read* rather
/// than asserted — each step says what it did and what the engine answered.
fn tour(engine: &mut RamEngine, clients: &[Client], issued: &mut Vec<Issued>) {
    let a = &clients[0];
    let b = &clients[1];

    heading("1. every session starts with its own private root");
    for client in clients.iter() {
        let info = engine.stat(client.session, client.root).unwrap();
        println!(
            "   session {} root: object {:#x}, {} entries",
            client.session.index(),
            info.object.to_bits(),
            info.links
        );
    }
    println!("   → no machine-wide `/`; three distinct root objects");

    heading("2. the same path text means different objects to different clients");
    run_quiet(engine, clients, issued, 0, "touch notes");
    run_quiet(engine, clients, issued, 1, "touch notes");
    show(engine, clients, issued, 0, "write notes belongs to A");
    show(engine, clients, issued, 1, "write notes belongs to B");
    let (handle_a, owned_a) = resolve(engine, a, "notes").unwrap();
    let (handle_b, owned_b) = resolve(engine, b, "notes").unwrap();
    let id_a = engine.stat(a.session, handle_a).unwrap().object;
    let id_b = engine.stat(b.session, handle_b).unwrap().object;
    println!("   A's `notes` = object {:#x}", id_a.to_bits());
    println!("   B's `notes` = object {:#x}", id_b.to_bits());
    println!(
        "   A reads {:?}",
        String::from_utf8_lossy(&read_all(engine, a, handle_a).unwrap())
    );
    println!(
        "   B reads {:?}",
        String::from_utf8_lossy(&read_all(engine, b, handle_b).unwrap())
    );
    println!("   → same name, different objects: {}", id_a != id_b);
    release(engine, a, handle_a, owned_a);
    release(engine, b, handle_b, owned_b);

    heading("3. one file can carry two names, without being copied");
    run_quiet(engine, clients, issued, 0, "mkdir projects");
    run_quiet(engine, clients, issued, 0, "touch projects/main.rs");
    show(
        engine,
        clients,
        issued,
        0,
        "write projects/main.rs fn main() {}",
    );
    show(
        engine,
        clients,
        issued,
        0,
        "ln projects/main.rs second-name",
    );
    let (via_first, o1) = resolve(engine, a, "projects/main.rs").unwrap();
    let (via_second, o2) = resolve(engine, a, "second-name").unwrap();
    let first = engine.stat(a.session, via_first).unwrap();
    let second = engine.stat(a.session, via_second).unwrap();
    println!(
        "   projects/main.rs -> object {:#x} ({} names)",
        first.object.to_bits(),
        first.links
    );
    println!(
        "   second-name      -> object {:#x} ({} names)",
        second.object.to_bits(),
        second.links
    );
    println!("   → one object, two containment edges");
    release(engine, a, via_first, o1);
    release(engine, a, via_second, o2);

    heading("4. removing one name leaves the other, and the bytes");
    show(engine, clients, issued, 0, "rm projects/main.rs");
    show(engine, clients, issued, 0, "cat second-name");
    println!("   → unlink removed one edge, not the object");

    heading("5. sharing is explicit, and read-only means read-only");
    show(engine, clients, issued, 0, "share second-name 1 ro gift");
    show(engine, clients, issued, 1, "grants");
    show(engine, clients, issued, 1, "cat @gift");
    print!("   B tries to write through the read-only grant: ");
    report(engine, clients, issued, 1, "write @gift overwritten");
    print!("   B tries to re-share it:                       ");
    report(engine, clients, issued, 1, "share @gift 2 ro passed-on");
    print!("   B tries to link it into its own root:         ");
    report(engine, clients, issued, 1, "ln @gift stolen");
    println!("   → the last one is refused because a containment edge carries no");
    println!("     rights: naming it under B's own root would relabel it with B's");
    println!("     full authority, laundering read-only into read-write.");

    heading("6. losing the route ends access immediately");
    let (b_view, b_owned) = resolve(engine, b, "@gift").unwrap();
    println!(
        "   B holds a live handle; reading now gives {:?}",
        String::from_utf8_lossy(&read_all(engine, b, b_view).unwrap())
    );
    show(engine, clients, issued, 0, "revoke 1");
    print!("   after A revokes, B's *existing* handle reads: ");
    match read_all(engine, b, b_view) {
        Ok(bytes) => println!("{:?} (unexpected!)", String::from_utf8_lossy(&bytes)),
        Err(err) => println!("{err:?}"),
    }
    println!("   → no cache to expire: the handle's proof is re-walked every call");
    release(engine, b, b_view, b_owned);

    heading("7. a second live route still works");
    show(engine, clients, issued, 0, "cat second-name");
    println!("   → A never lost anything; only B's route was retired");

    heading("8. writes are revision-checked");
    let (file, owned) = resolve(engine, a, "second-name").unwrap();
    let revision = engine.stat(a.session, file).unwrap().revision;
    println!("   current revision is {revision}");
    let stale = revision;
    let fresh = block_on(engine.write_at(a.session, file, 0, b"first change", ANY_REVISION))
        .expect("unconditional write");
    println!("   an unconditional write committed, revision now {fresh}");
    match block_on(engine.write_at(a.session, file, 0, b"second change", stale)) {
        Ok(_) => println!("   a write against the stale revision succeeded (unexpected!)"),
        Err(err) => println!("   a write against revision {stale} was refused: {err:?}"),
    }
    println!("   → explicit conflict detection, not last-writer-wins by accident");
    release(engine, a, file, owned);

    heading("9. append chooses its offset atomically");
    run_quiet(engine, clients, issued, 0, "touch log");
    for _ in 0..3 {
        let (log, owned) = resolve(engine, a, "log").unwrap();
        let (offset, _) = block_on(engine.append(a.session, log, b"entry ", ANY_REVISION)).unwrap();
        println!("   append landed at offset {offset}");
        release(engine, a, log, owned);
    }
    show(engine, clients, issued, 0, "cat log");
    println!("   → each append got its own space; none overwrote another");

    heading("10. reclamation returns storage when the last route goes");
    let before = engine.provider().bytes_in_use();
    println!("   storage in use: {before} bytes");
    show(engine, clients, issued, 0, "rm second-name");
    show(engine, clients, issued, 0, "rm log");
    show(engine, clients, issued, 0, "rm notes");
    show(engine, clients, issued, 0, "rm projects");
    println!(
        "   after removing A's files: {} bytes in {} placements, {} leaked",
        engine.provider().bytes_in_use(),
        engine.provider().placements(),
        engine.leaked_placements()
    );
    println!("   the remainder is session 1's own `notes`, which nobody removed —");
    println!("   reclamation is per-object and never reached into B's tree.");
    println!("   → last edge and last grant gone ⇒ object and its bytes released");

    println!("\nTour complete. `help` to explore by hand.\n");
}

fn heading(text: &str) {
    println!("\n\x1b[1m── {text}\x1b[0m");
}

/// Run a command, printing the command but discarding ordinary output.
fn run_quiet(
    engine: &mut RamEngine,
    clients: &[Client],
    issued: &mut Vec<Issued>,
    session: usize,
    line: &str,
) {
    if let Err(err) = run(engine, clients, session, line, issued) {
        println!("   [{session}] {line} -> refused: {err:?}");
    }
}

/// Run a command, echoing it first.
fn show(
    engine: &mut RamEngine,
    clients: &[Client],
    issued: &mut Vec<Issued>,
    session: usize,
    line: &str,
) {
    println!("   [session {session}] $ {line}");
    if let Err(err) = run(engine, clients, session, line, issued) {
        println!("   refused: {err:?}");
    }
}

/// Run a command expecting it to be refused, printing only the outcome.
fn report(
    engine: &mut RamEngine,
    clients: &[Client],
    issued: &mut Vec<Issued>,
    session: usize,
    line: &str,
) {
    match run(engine, clients, session, line, issued) {
        Ok(()) => println!("allowed (unexpected!)"),
        Err(err) => println!("{err:?}"),
    }
}
