# libkore alpha

This is the application-facing library. Kore's kernel package is not a dependency.
The initial facade is adapted from Kore #962. `native/` is a byte-exact export of
its committed dependency closure, tracked by `native/provenance.json`.

- Keep one ABI/runtime/filesystem/driver owner graph. Never hand-edit the exported
  owners, add parallel implementations, or cast between equivalent-looking
  capability types from different packages. Change the upstream owner and use
  `tools/sync-native.py` until ownership is deliberately transferred here.
  Use `make fmt`: `cargo fmt --all` also traverses local path dependencies and
  rewrites the upstream snapshot's formatting.
- Preserve affine ownership, outstanding-request custody, bounded allocation and
  error/partial-completion semantics. A lost response does not permit replay.
- Add behavior through the actual native owner/caller path. Empty headers,
  placeholder allocators and mock implementations are not completed language APIs.
- A foreign language's first surface must include filesystem support. C bindings
  must respect Kore's capability model even where POSIX cannot represent it.
- Driver wrappers must account for register side effects and DMA completion/retire
  lifetimes, not just address bounds. Lua probes need a concrete native owner.
- Keep the export coherent, update both Cargo.lock files, and run `make fmt test check docs
  owners cross miri`. Use `tools/check-local.py` to prove composition with local
  Kore clients before claiming a local override works.
- Changes affecting native owners belong upstream until those packages are
  deliberately extracted once. Re-run Kore's full image/gate matrix for such
  changes; this repo's compiler/Miri checks are not boot or hardware evidence.
- Record provenance and distinguish implemented APIs from aspirational design.
  Do not publish a registry release or claim ABI stability as part of a source PR.
