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

The window is not the only transport. A `read_at` whose destination is at
least **256 KiB**, and `read_file` over a file at least that large, may instead
stream through the launch's storage ring from one pinned version. This is a
property of the destination's size, not of the file's: a caller that loops over
a small buffer stays on the window however large the range is, and one whole-file
call holds a single version where a loop of positional reads pins one per call.
Whether a launch holds a ring at all is decided at founding.

`read_exact_at` and `write_all_at` loop bounded transfers without allocation.
They check the entire offset range before the first request. Both report only
the **acknowledged** prefix. For a write, the current unacknowledged chunk may
already have committed. For a read, the destination may hold bytes past the
reported count — a streamed transfer writes into caller storage as it lands,
and a refusal part way leaves that prefix uncounted — so storage past the
acknowledged prefix is clobbered, not retained. Neither count is a **retry
offset**, and neither helper automatically replays a mutation.

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
[languages.md](languages.md). `kore-lua` supplies that integration for Lua: its
open-file table holds the affine views and Lua holds generational handles, so a
use after close is `invalid-handle` rather than a read of a recycled slot. A
Python adapter still needs its own script lifetime and scheduling integration
above this native owner.
