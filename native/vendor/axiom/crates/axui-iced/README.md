# axui-iced

`axui-iced` adapts Iced widgets to Axiom without selecting `iced_wgpu`,
`iced_tiny_skia`, or another Iced graphics backend. It depends only on
`iced_core`, pinned to the MicroPerceptron fork, and records Iced's base
renderer primitives into `axui_core::Frame<A>`.

`Renderer<A>` has no default allocator or `Default` implementation. Tests and
desktop demos inject a `std` allocator explicitly; Kore supplies its own
allocator at the final runtime boundary.

## Boundary of this first integration slice

Implemented:

- clip layers and 4×4 transformations;
- solid and linear-gradient quads;
- per-corner borders, shadows, and pixel-snap intent;
- logical-to-physical scale hints;
- structurally validated, reusable frame recording.

Intentionally still explicit:

- text needs an Iced `Paragraph` and `Editor` implementation in addition to
  glyph painting; it will build on `axrd`'s font work rather than hide a CPU
  fallback here;
- image allocation reports `Unsupported` until Axiom has a resource layer;
- this crate does not produce pixels. A following `axrd` lowering consumes the
  typed display list and compiles it into `.ax` graphs for scheduler dispatch.

The pin in `Cargo.toml` is deliberate: it makes the exact frontend API used by
Axiom reviewable and gives us a controlled place to patch Iced only when the
public renderer contract is insufficient.
