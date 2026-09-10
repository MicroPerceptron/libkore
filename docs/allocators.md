# C and Zig allocation

`kore-ffi` routes foreign allocations through the Rust application's installed
global allocator. A founded Kore host installs its existing runtime heap once:

```rust,ignore
use kore::runtime::heap::{Heap, HeapSupplySource};
#[global_allocator]
static HEAP: Heap<HeapSupplySource> = Heap::new_in(HeapSupplySource);
```

Enable `kore-std`'s `heap` feature to expose these types. The host needs its
founded heap supply before making allocating calls. Libkore installs no second
heap, grants no memory authority, and does not replace the native allocator.
Host tests use the test executable's allocator; a separate adapter test uses the
actual Kore heap engine with bounded `StaticMemSource` backing. Neither is a
claim of new physical or founded-process validation.

## C

Use the generated `<kore.h>` declarations for `kore_alloc_malloc`,
`kore_alloc_calloc`, `kore_alloc_realloc`, `kore_alloc_free`, and
`kore_alloc_aligned(size, alignment)`. Or opt into familiar source names:

```c
#include <kore/stdlib.h>

void example(void) {
    unsigned char *buffer = calloc(128, 1);
    if (!buffer) return;
    unsigned char *grown = realloc(buffer, 4096);
    if (grown) buffer = grown;
    free(buffer);
}
```

`kore/stdlib.h` provides generated aliases to those same signatures. Include it
instead of a host's `stdlib.h`; it is an allocation surface, not a complete libc.
For compiler-emitted calls or objects already referencing standard allocator
symbols, enable `kore-ffi`'s **`c-stdlib`** feature. It exports `malloc`, `calloc`,
`realloc` and `free` on freestanding Kore targets only. It intentionally does not
interpose the host libc allocator when running tests. The final executable must
retain/link `kore-ffi` and install the native heap; this feature is not a C CRT.

Allocation and layout failures return null; `calloc` checks multiplication
before allocating. No `errno` is set. Successful zero-size allocation produces
a unique freeable block. `free(NULL)` does nothing. `realloc(NULL, n)` allocates;
`realloc(nonnull, 0)` frees and returns null. A failed nonzero realloc leaves the
original block and contents intact. Malloc alignment is at least 16 bytes on
all supported Kore targets. Aligned allocation accepts any nonzero power of
two; its size need not be an alignment multiple. Realloc preserves alignment
and the first `min(old_size, new_size)` bytes.

A private prefix records the layout needed for C's unsized free. It adds storage
and alignment overhead. Realloc currently allocates, copies and frees; it does
not promise in-place growth. Only pass pointers from this API to its free and
realloc functions. Raw Rust allocations and pointers from another allocator do
not carry this prefix and cannot be exchanged directly.

## Zig 0.16

`kore.allocator` is a real `std.mem.Allocator` backed by the same exported calls:

```zig
const std = @import("std");
const kore = @import("kore");

fn example() !void {
    var values: std.ArrayList(u64) = .empty;
    defer values.deinit(kore.allocator);
    try values.append(kore.allocator, 42);
}
```

Aligned allocation, remap and free use the recorded layout. In-place resize
returns false, allowing standard containers to use remap or allocate/copy/free.
Allocation failures become `error.OutOfMemory`. Applications may still pass
another allocator to Zig containers. Live objects must be released through a
compatible allocator; changing a default does not migrate existing objects.

## Default hash tables

C calls `kore_hash_table_create_default(capacity)`; Zig calls
`kore.hash.table_create_default(capacity)`. Entries remain caller-owned. Both
use `kore_alloc_hash_allocator()`, also available when constructing tables
explicitly. Existing callback-based constructors remain available for arenas.
The installed allocator must remain compatible until all its allocations and
tables have been released.

`make cross` also checks the emitted C allocator symbols using `llvm-nm`
(override its path with `LLVM_NM`). CI performs that check for each target.
