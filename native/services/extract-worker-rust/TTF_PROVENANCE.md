# SFNT metadata reader provenance and maintenance

`src/sfnt.rs` is a bounded metadata-only derivative maintained in this worker.
The required-table checks derive from `ttf-parser 0.25.1` (`src/lib.rs` RawFace,
`src/tables/head.rs`, `hhea.rs`, `maxp.rs`), under the preserved MIT license.
Upstream package checksum: `d2df906b07856748fa3f6e0ad0cbaa047052d4a7dd609e231c4f72cee8c36f31`.
Upstream commit: `56c33b910b03ca152f78363ec471c5dfd97c3069`.

The extraction adapter only emits media type and glyph count. The derivative
retains that full output contract and the required head/hhea/maxp checks, but
contains no outline, shaping, bitmap, variation, name, or floating-point parser.
This is not a drop-in fork or whole-font validator. No runtime dependencies,
`unsafe`, global allocator, recursion, or scratch allocation are used.

Parser identity advances to version 2: directory ordering/uniqueness, all table
extents and TTC header versions now fail closed, and the declared member budget
bounds both table and collection directories. Cached version-1 normalized
results must not be silently relabeled. First-face extraction is unchanged.

RUSTSEC-2026-0192 describes the abandoned full upstream crate. Vendoring or
renaming that unchanged crate would not establish maintenance. Instead, this
small reader has an explicit in-tree scope and regression ownership: all edits
must run the required-table/offset/collection tests, worker truncation/mutation
corpus, release benchmark, fuzz harness and founded worker gates. New font facts
or table support require a source review and parser-version decision. Do not
expand this into an unaudited copy of the old full font stack.
