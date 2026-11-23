# kore MMIO-half gating patch

This directory vendors `xhci` 0.9.2 (MIT OR Apache-2.0), the rust-osdev xHCI
crate, at its published contents.

kore uses this crate **à la carte**, and `drivers/xhci/Cargo.toml` has always
said so:

> used à la carte: its typed `ring::trb` + `context` modules, never its
> register accessors (those do their own volatile MMIO, which would bypass the
> `Regs` seam).

The crate is consumed for its **spec-frozen type layouts** — TRB and device /
input context bit fields, which are xHCI specification facts and therefore do
not rot. kore drives the controller's registers itself, through
`kore_driver::seam::Regs`, whose BAR-backed implementation asserts alignment,
checks offset overflow, and refuses an access past the mapped window
(`crates/driver/src/host.rs`, `CapRegs::ptr`). Routing register access through
the crate instead would bypass that seam.

Until now that was a comment. The register half still compiled, and it carried
a non-optional dependency on `accessor` 0.3.0 — which is subject to
**RUSTSEC-2026-0127**: an integer overflow in `array::ReadWrite::new()`
leading to potential memory corruption.

That advisory is *reachable in this crate*, and reachable from device-supplied
data at that. `extended_capabilities/xhci_local_memory.rs` computes

```rust
array::ReadWrite::new(base + 8, (size * 1024).try_into().unwrap(), mapper)
```

where `size` is read straight off a hardware register — so the multiply can
overflow before the crate is even entered, and the `unwrap` is on a
device-controlled value. `xhci_supported_protocol.rs` does the same with a
firmware-reported length. Nothing in kore ever calls either, because nothing
in kore constructs an `accessor` type at all.

## Local changes

Deliberately limited to making the unused half *unbuildable* rather than
merely unused:

1. `src/lib.rs` — `pub mod registers`, `pub mod extended_capabilities`, and
   the three re-exports (`accessor`, `ExtendedCapability`, `Registers`) are
   gated behind a new `mmio` feature.
2. `src/lib.rs` — `mod macros` carries
   `#[cfg_attr(not(feature = "mmio"), allow(unused_macros))]`; several
   register-field macros are reachable only from the gated module, and they
   stay *defined* so the patch is a pure subtraction.
3. `Cargo.toml` — `accessor` becomes `optional = true`, and a `[features]`
   table adds `mmio = ["dep:accessor"]`, `default = []`.

`src/ring` and `src/context` reference neither `accessor` nor the gated
modules, so the half kore actually uses is untouched. `Cargo.toml.orig` is
left as cargo vendored it — the upstream record.

## Effect

`accessor` leaves the dependency graph entirely: it is absent from
`Cargo.lock`, `user/Cargo.lock` and `services/extract/fuzz/Cargo.lock`, and
`make audit` no longer reports RUSTSEC-2026-0127.

The patch is reversible rather than destructive: `--features mmio` still
compiles the register and extended-capability modules exactly as upstream
publishes them. If kore later wants that half, the decision is a feature flag
and a fresh look at `accessor` — not an archaeology exercise.

`paste` 1.0.4 (RUSTSEC-2024-0436, unmaintained) is **not** removed by this and
cannot be: `context/macros.rs`, `ring/trb/mod.rs` and `macros.rs` all use it,
so it is load-bearing for the half kore keeps.

## Upstream status

`xhci` 0.9.2 was published in **July 2023** and is the latest release. No
upstream fix is pending, which is why this is carried here rather than waited
on. The change is small and upstreamable if the crate revives.
