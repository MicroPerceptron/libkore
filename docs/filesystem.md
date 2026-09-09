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

`Fs::read_at` and `Fs::write_at` transfer at most the actual shared window and
wire limit. Reads validate the reply count before copying into caller storage;
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

## Close and future foreign owners

`close_checked` owns the view while pending and returns success only after an
acknowledged close. On refusal it returns the exact view with the error; retry matches an outstanding close and
cannot consume another handle's late reply. Drive that future to completion:
dropping it is not an acknowledged close or a managed cancellation mechanism.

The future C/Lua owner must retain requests and views independently of a script's
stack, use generation-checked handles scoped to one owner, and complete or retire
the owner before releasing windows. Neither raw `Handle` integers nor `Drop`
alone provide this lifecycle. This alpha does not expose an unsafe foreign
handle table in place of that design.
