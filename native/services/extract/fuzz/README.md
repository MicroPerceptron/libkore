# Extraction fuzz corpus

`adapters` coverage-fuzzes every accepted Rust/C adapter and immediately sends
every accepted result through the normalized validator. `normalized_validator`
mutates provenance, parser ids, ranges, fact kinds, and output descriptors
independently of parser acceptance.

Run with:

```sh
cargo fuzz run adapters
cargo fuzz run normalized_validator
```

The ordinary host gate also deterministically mutates and truncates the JSON,
XML, TAR, BMP, TTF, and EPUB fixtures, so regressions remain exercised on hosts
without `cargo-fuzz` or libFuzzer.

Upstream coverage-guided fuzz evidence and the exact local commands are
recorded in `res/EXTRACT_VENDOR_AUDIT.md`.
