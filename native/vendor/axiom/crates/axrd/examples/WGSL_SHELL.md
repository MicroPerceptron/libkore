# Kore shell / WGSL study

A native rendering experiment adapted from the supplied `kore-shell-wgsl.html`
mockup. This is not an HTML/WebGPU importer, a production shell, or live system
telemetry: the terminal, inference figures, clock, and capability labels are
static design content.

## Run

```sh
AXRD_WGSL_SCENE=shell AXRD_WGSL_SIZE=960x640 \
  cargo +nightly run --release -p axrd --example wgsl \
  --features window,graph,font
```

For CoreML provider placement on macOS:

```sh
AXRD_WGSL_SCENE=shell AXRD_WGSL_SIZE=960x640 \
AXRD_WGSL_BACKEND=heterogeneous \
  cargo +nightly run --release -p axrd --example wgsl \
  --features window,graph,font,coreml
```

Use `T` to switch tiled/floating arrangements, `N` for day/night, and Escape
to close. These are snapshot switches that rebuild the graph, not animated
window-management interactions. Resizing stretches the last completed frame
until a new, exact-resolution graph is prepared.

The same host runner supports the `openvino` feature on an OpenVINO host; this
study has not been hardware-validated there.

## Capture and export

For a headless night/floating capture through CoreML:

```sh
AXRD_WGSL_SCENE=shell AXRD_WGSL_SIZE=960x640 \
AXRD_WGSL_HEADLESS=1 AXRD_WGSL_THEME=night AXRD_WGSL_LAYOUT=floating \
AXRD_WGSL_BACKEND=heterogeneous AXRD_WGSL_PPM=/tmp/kore-shell.ppm \
  cargo +nightly run --release -p axrd --example wgsl \
  --features window,graph,font,coreml
```

`AXRD_WGSL_AX=/tmp/kore-shell` exports the actual composed graphs. Tiled
exports append `.tile-X-Y.ax`; an untiled export uses the exact supplied path.
Their input ABI is two F32 coordinate axes, X `[1, tile_width]` and Y
`[tile_height, 1]`, containing **global physical pixel centers** (tile origin +
index + 0.5). Shell outputs are three F32 sRGB planes in RGB order. Viewport,
theme, layout, and prepared font coverage are specialized into each artifact.

## What runs where

- `assets/shaders/shell.wgsl` owns the procedural ground and physical-pixel
  sampling convention. `support/wgsl_shell.rs` specializes layout/palette and
  unrolls reusable rounded-box SDF paints into v0-compatible WGSL expressions.
- Standalone Naga and `axshader` compile that source into semantic `.ax`.
- Existing `axrd` font preparation creates supersampled glyph coverage using
  the bundled Fraunces, DM Sans, and DM Mono fonts. They substitute for the
  reference's Caprasimo, Figtree, and JetBrains Mono; no web fonts are required.
- The prepared text graph's RGB inputs are connected directly to the WGSL
  graph outputs. Text source-over and shading share one canonical DAG and
  normal `ax`/`axnn`/`AxHetero` admission. There is no intermediate framebuffer
  readback, per-backend UI renderer, or raw CPU raster-kernel dispatch.
- The host packs completed RGB planes and copies tiles into the presentation
  buffer. Source generation and font preparation are host/build-time work;
  this example does not add a Naga or font-rasterizer dependency to `kore`.

## Tiling and validation

The shell defaults to a 480-pixel tile edge. `AXRD_WGSL_TILE=0` disables tiling;
another integer in 32..=3840 overrides the edge. Tiling does **not** reduce
resolution. All geometry uses the full viewport's coordinates; text is
prepared at the same physical size/phase and clipped to the tile.

The 960×640 monolithic graph failed at execution on the tested CoreML host,
while the same scene at 480×320 succeeded. Four bounded tiles render 960×640
successfully, including the partial bottom row. This is a conservative example
policy, not an inferred universal provider limit or scheduler memory budget.

Local release smoke results (not a benchmark):

- Day/tiled: 4 tiles, all 5,432 scheduled node instances on the CoreML provider;
  approximately 2.44 s preparation and 302 ms total execution.
- Night/floating: all 5,432 node instances on the provider; approximately
  2.46 s preparation and 303 ms execution.
- Untiled native CPU day reference: approximately 7.6 s execution.
- Tiled CoreML versus untiled CPU day output: 89 of 1,843,200 RGB channels
  differ, each by 1/255. Both renders were visually inspected.
- A regression compares native tiled/untiled outputs for both designs at a
  non-divisible 121×83 viewport, including text and partial edge tiles.

The experiment also exposed an exponential-kernel range bug: the SIMD exponent
construction could produce negative infinity for distant shadow samples.
Portable/AVX2 repair lanes now preserve the scalar range contract, and scalar
`exp_f32` preserves finite overflow-edge and subnormal results. Regressions
exercise the automatic planner, SIMD chunks/tails, and a libm boundary oracle.

## Deliberate approximations and next gaps

- Glass uses translucent tint, shadows use an SDF halo, and paint values are
  composed in sRGB. This is **not** the reference's HDR multi-pass sampled
  backdrop blur, analytic shadow quadrature, transfer function, or dither.
- The floating snapshot overlaps chrome but keeps text content clear. It does
  not implement arbitrary z-order text occlusion, dragging, BSP resizing,
  search, or working window controls.
- The v0 compiler cannot directly ingest the reference's helper functions,
  uniform/storage records, atlas sampling, vertex stage, or blur loops.
- Prepared coverage costs roughly 10 MB across this frame's tile artifacts.
  Portable atlas/image sampling would replace much of that static payload.
- This is not real-time yet. Retained compiled tile programs, shared font
  preparation, per-window damage/culling, and fused/tile-local execution are
  the next performance work—not reduced-resolution text or hidden CPU paints.
