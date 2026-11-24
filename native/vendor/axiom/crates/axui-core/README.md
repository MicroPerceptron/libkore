# axui-core

`axui-core` is Axiom's backend-neutral UI display-list contract. UI frontends
record logical clips, transformations, fills, gradients, borders, and shadows;
render backends consume the resulting `FrameView<A>` and decide how to lower
it.

The crate is `no_std` plus `alloc` and intentionally contains no windowing,
widget, text-shaping, raster, or accelerator dependency. This keeps the
frontend/backend boundary reusable by `axrd` and future Axiom renderers.

`Recorder<A>` and `Frame<A>` require an explicit nightly `Allocator`. There is
no default allocator and no allocator-free constructor; the final runtime
supplies `A`, and its type remains visible through the borrowed graphics seam.
