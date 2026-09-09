# Hash collections

`kore::collections` reexports hashbrown 0.17's `HashMap`, `HashSet`, `HashTable`,
entry modules, `Equivalent`, `DefaultHashBuilder` and `TryReserveError`.
The three collection types are also in `kore::prelude`.

```rust
use kore_std::prelude::*;
let mut counts = HashMap::new();
counts.insert("requests", 1u64);
assert_eq!(counts.get("requests"), Some(&1));
```

Default constructors use the application's installed global allocator. Libkore
installs no allocator. Explicit `new_in` constructors retain upstream Rust
`Allocator` parameters, and `try_reserve` lets callers handle growth failure.
The dependency is `no_std`, with `nightly` and `default-hasher` enabled and
upstream default features disabled. The fast default hasher does not provide
Rust std's HashDoS protection; select a suitable keyed hasher for untrusted keys.

## C and Zig

The existing `kore-ffi` package exports `kore_hash_table_*`; the generated Zig
namespace is `kore.hash.table_*`. The C header and Zig signatures are generated
from the Rust adapter, as for the other foreign surfaces.

A `KoreHashTable *` is opaque and uniquely owned. Creation takes a
`KoreHashAllocator` (context, allocate, deallocate) and initial capacity.
Both the handle and buckets use those callbacks. No implicit global allocator
is selected by this API. Null creation means invalid callbacks or allocation
failure. Destroy accepts null and releases only table storage.

Insert accepts a caller-supplied 64-bit hash, non-null entry pointer, equality
callback, callback context and mandatory previous-entry output. Find/remove
accept a query pointer interpreted by that equality callback. Equal keys must
have equal hashes; hashes are cached for growth, and stored keys must not mutate.
Entries stay caller-owned, including after replacement, removal, clear or destroy.
This represents either a map record or a set element without imposing its layout.

`KORE_HASH_OK`, `MISSING`, `REPLACED`, `INVALID` and `NO_MEMORY` describe immediate
results (each constant has the `KORE_HASH_` prefix). Lookup/removal outputs are
null when absent. Failed growth preserves contents. Clear retains capacity;
visit traverses in unspecified order and stops when its callback returns zero.

The allocator context must outlive the table and callbacks must keep their
allocation/deallocation pairing even if the application's default changes.
They must provide disjoint, correctly aligned storage for the requested layout.
Callbacks must not unwind or reenter the same table. Mutations require exclusive
access; output storage must be valid and disjoint from table and entry storage.
The generated header contains the per-function safety contracts. These are alpha
source interfaces, with no ABI stability promise.

Host C and translated Zig consumers exercise the real exports; Rust tests also
exercise allocator failure and cleanup under Miri. Cross compilation covers all
three Kore targets. This is library validation, not a physical boot claim.
