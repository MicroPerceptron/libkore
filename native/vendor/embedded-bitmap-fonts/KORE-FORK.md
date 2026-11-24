# kore fork notes

Vendored from crates.io `embedded-bitmap-fonts-0.1.1`
(sha256 54211cbfdc5c61ccb33c737558194466e9b378f52aa2e79d08c91652520780a7),
the `vendor/acpi` model: ours to patch, provenance recorded here.

Local changes, kept minimal so upstream releases diff cleanly:

1. `Cargo.toml` — appended `[workspace]` (the vendor-tree opt-out).
2. `src/terminus.rs` — the raw atlases exposed as
   `pub static FONT_16x32_RAW` / `pub static FONT_12x24_RAW`: kore's console renders through its own
   row-major SIMD blit (`platform/display/text.rs`), not
   `embedded-graphics` text (whose per-pixel `draw_iter` would regress
   the write-combining aperture pattern), so it needs the raw 1bpp atlas
   bytes the crate otherwise hides behind `ImageRaw`.
3. `FONT-LICENSES.md` / this file — license carriage + provenance (the
   crate ships no license files; its `Cargo.toml` declares Apache-2.0
   and defers font licenses to upstream sources).
