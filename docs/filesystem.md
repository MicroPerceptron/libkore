# Filesystem contract

`kore::fs` builds on the process-side DagFS owner; it does not implement another
filesystem, cache permissions, create a global root, or clone a view's authority.
The application attaches with the filesystem face supplied at founding. Paths
resolve within the endowed view. Native path walks, listing, stat, creation,
resize, sync, links, grants, revisions and query operations remain on the same
client through `fs::dag`.

The alpha directly preserves native `Fs`/`View` types. More ergonomic file/directory
wrappers can be added above them without hiding graph semantics or manufacturing
POSIX permissions. A global POSIX namespace is not part of this contract.

## Transfer and failure

`Fs::read_at` and `Fs::write_at` transfer at most the actual shared window,
the service-advertised transfer limit, and the wire byte-count limit. The
4 KiB protocol floor is not a ceiling: durable providers can advertise 64 KiB.
The C/Zig callback adapter still uses its explicitly bounded 4 KiB scratch buffer. Reads validate the reply count before copying into caller storage;
a short read need not be EOF. Zero bytes read is EOF. The existing zero-copy
`read_chunk` remains available when the owner can consume its borrowed window.

`read_exact_at` and `write_all_at` loop bounded transfers without allocation.
They check the entire offset range before the first request. Read errors report
an initialized prefix. Write errors report only the acknowledged prefix: the
current unacknowledged chunk may already have committed. This is **not a retry
offset**. Neither helper automatically replays a mutation.

Offset writes preserve the old tail. `resize` is a separate operation; successful
chunks followed by a refused resize leave a partially changed file. Sync and
transaction/revision operations must remain explicit. Index publication is also
separate from authoritative file persistence.

An outstanding request owns the data window. A subsequent write must refuse
before modifying it. Cancellation is not completion: keep the owner alive and
use the native request/reply lifecycle to reconcile an outstanding operation.

## Close and foreign owners

`close_checked` owns the view while pending and returns success only after an
acknowledged close. On refusal it returns the exact view with the error; retry matches an outstanding close and
cannot consume another handle's late reply. Drive that future to completion:
dropping it is not an acknowledged close or a managed cancellation mechanism.

The C/Zig `kore-ffi::Session` retains native views and copied buffers behind
owner-scoped generational handles. It drives one native request at a time,
returns completions through a callback, and quarantines an uncertain/cancelled
owner until task retirement. It never represents `Drop` as acknowledged cleanup.
The callback-only pointer and host integration contracts are specified in
[languages.md](languages.md). Lua/Python adapters still need their own script
lifetime and scheduling integration above this native owner.
