# Production welcome surface

`axboot` is the shared production asset, not a second host-only boot design:
the welcome overlay (clock, copy, status chips, the seat pill) over an
animated Kerr black hole, after the lock-screen kit's composition.

```sh
cargo run --release -p axboot --example boot
AXBOOT_SIZE=1920x1200 AXBOOT_PPM=/tmp/boot.ppm cargo run --release -p axboot --example boot
```

The host example takes real UTC from `SystemTime`. `AXBOOT_UNIX_SECONDS` and
`AXBOOT_SECONDS` (the animation's time: the disk's turn and the camera's
swing) are explicit screenshot/test overrides; `AXBOOT_STEP` also times one
animation step. Escape closes the example.

## The scene: traced once, animated per frame

Every pixel traces one null geodesic backwards from the camera through the
Kerr metric (Boyer–Lindquist, RK4 on the Hamiltonian equations; spin 0.9,
inclination 78°). The metric is stationary, so each pixel's path is fixed and
`build/kerr.rs` traces once, at build time, on every build-host core; the
runtime never integrates anything. The camera still moves: it swings ±10°
around the spin axis over four minutes (`orbit_angle`). Metric and disk are
both symmetric about that axis, so the swing is exactly the scene turned by
the swing's angle — the runtime shifts the azimuth it looks sky and disk up
at, and the baked paths stay valid.

- **The lens map** (1920x1200, one texel per pixel on the reference display;
  16 bytes a texel, 37 MB): where the escaping rays meet the sky (a sky-atlas
  point and the fraction that escape) and the two disk images the rays see,
  nearer first (r, φ, redshift g, and each one's coverage), every quantity
  taken over five rays a texel so the photon ring's sub-texel images
  anti-alias. Crossings group into images by where they meet the disk, not
  by their order along each ray. Steps are bounded by how far they turn the
  ray in θ and φ, which resolves the whip in φ of rays passing near the polar
  axis; the last RK4 step into the horizon blows up (Δ → 0 in the
  denominators), and the bake rejects any step that moves r by more than
  half, so no phantom crossing survives it.
- **The sky atlas** (about 2167x1771 UF16 RGB, 23 MB): the sky prefiltered into
  tiers — a fine patch at the source map's full resolution where the camera
  sees the sky nearly unlensed, and whole spheres at 2048, 256 and 32 texels
  a turn for the lensed images. Each pixel reads the tier nearest its
  footprint, so stars do not sparkle as the camera swings; every tier is
  padded by the full swing. It is resampled from two committed plates
  (`assets/sky/`), cut by `tools/sky-prep` from NASA SVS's Deep Star Maps 2020
  — see `assets/sky/README.md` for the source and the credit any surface
  showing this sky must carry.
- **The turbulence table** (2049x512 UF16 RGB, 6.3 MB): the kit's disk noise in
  (φ, ln r). Each octave the table cannot hold (fewer than two samples a
  lattice cell) fades to its mean rather than folding into stripes. The shader
  samples it at `φ − Ω(r)·t`: the pattern co-rotates with the gas.

The bake caches its output by a hash of its source, its parameters and the
sky plates, so a shader edit does not re-trace.

## Runtime contract

- `build.rs` bakes the scene and the text, measures the overlay's layout from
  the baked text, and compiles `assets/boot.wgsl` (with the bake's constants
  prepended) to a 128x32 `.ax` program. The target links no parser, window
  toolkit, or implicit allocator.
- The graph takes position axes and the profile's sixteen field lanes —
  light/dark ink, shape coverage (shadow, dark fill, green), the sky's atlas
  point and coverage, and the near and far crossings — plus the viewport, the
  animation's time and the camera's swing; the turbulence table and the sky
  atlas bind as its two RGB-volume resources. WGSL owns the disk's light, its
  redshift and turbulence, the swing's shift in each sky tier, tone mapping
  and every colour and its order of composition; the runtime samples the lens
  map and prepares coverage — text, and the overlay's few shapes, computed
  only on the tiles that hold them — and packs completed RGBA outputs.
- `TileInputs<A>` keeps explicit allocation throughout. The scheduler/runtime
  admits `GRAPH` once; the same program serves every viewport and frame.
- Fonts are build-time Fraunces (900), DM Sans (450), and DM Mono. Their OFL
  licenses are in `../axrd/assets/fonts`. Two-times-design U16 coverage is
  stored as exact integer summed-area tables; F64 area reconstruction feeds
  F32 anti-aliasing/composition. The clock sets proportional figures (Fraunces
  Black's "1" is two-thirds of its "0") from the left, so a minute never moves
  the hour.
- `Clock` accepts optional Unix seconds, displays `--:--` without calibration,
  and deliberately labels UTC until the runtime provides a timezone policy.
  No fake peer, task count, IOMMU claim, or logged-in identity is displayed.
- `Frame` is the animation's clock: whole 1/15 s frames since the page
  appeared. The overlay stands still. `tile_changed` repaints every tile when
  the frame moves (the swing moves the whole sky) and the clock's tiles when
  the minute changes. The runtime paces frames (`FRAME_NS` is the fastest it
  asks for).

## Kore boundary

`kore-axiom::boot::BootProgram` admits the same asset through `ax` and invokes
the `axnn` catalog. The display runtime reads `kore_rt::clock::wall` and
submits up to eight disjoint graph tiles per
service pass via accelerator-agnostic Kore SPMD. Worker scratch is mapped
before dispatch; cooperative ticket waits retain all borrows. The display
bundle grants `CHANNEL_SPACE` for resident-worker doorbells. A starless
founding is retired and joined before falling back to owner-lane graph
execution, so an idle welcome surface cannot retain spinning workers. Dismissal
retires the region and releases its scratch. Idle waits arm the next minute
deadline, or while the page is up the next animation frame its pacing allows,
rather than polling. No GPU/display driver change is required.

This asset does not add an accelerator provider to Kore. The existing Kore
program adapter currently admits the CPU catalog; the host example also
selects the portable CPU lane explicitly. Neither is evidence of GPU/NPU
dispatch. Provider expansion belongs at that runtime boundary, not in the
scene. On the CPU path one 1920x1200 frame — every animation step, now that
the sky moves — is about 165 ms of one Apple M3 core, so four workers draw
about 24 frames a second at full lane use; Kore paces the scene to at most
half its lane. At one lens texel per pixel the runtime reads each texel
directly instead of blending four; crossings carry ln r, decoded by table, so
nothing takes a logarithm per pixel; and the CPU catalog samples one-slice
tables four samples to a vector. The swing is slow (at its fastest
0.26° a second, about 2.5 px between frames at 4 a second for directly seen
stars), so a lower frame rate draws fewer frames of the same motion without
visible judder.
