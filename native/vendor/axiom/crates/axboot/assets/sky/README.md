# The welcome's sky plates

`near.bin` and `far.bin` are the sky behind the Kerr black hole, cut from an
all-sky map by `tools/sky-prep` (format in `build/sky.rs`):

- `near.bin` — the map at full resolution (8192 texels per turn) over the
  azimuths and latitudes the camera sees nearly unlensed, swing included
  (`SKY_NEAR_*` in `src/kerr_params.rs`);
- `far.bin` — the whole sphere at 2048 texels per turn, for the lensed images.

Both are already turned into the hole's frame (`SKY_BEHIND_GALACTIC_DEG`,
`SKY_ROLL_DEG`), and each carries the orientation it was cut for; the build
refuses a mismatch. The bake resamples them into the runtime's tiered sky atlas.

## Source

NASA Scientific Visualization Studio, *Deep Star Maps 2020*
(<https://svs.gsfc.nasa.gov/4851>), `milkyway_2020_8k_gal.exr`: an 8192x4096
galactic plate carrée in linear half-float, built from 1.7 billion stars of
Hipparcos-2, Tycho-2 and Gaia DR2. The `milkyway` variant omits the bright
Hipparcos and Tycho stars; `starmap_2020_8k_gal.exr` from the same page
includes them and cuts with the same command.

The source map stays out of the tree (it is 119 MB). To cut the plates again:

```sh
cargo run --release -p sky-prep -- milkyway_2020_8k_gal.exr
```

## Credit

The SVS asks that the item be credited to:

> NASA/Goddard Space Flight Center Scientific Visualization Studio

and the map's Gaia DR2 data to ESA/Gaia/DPAC. Any surface that shows this sky
(the welcome, the lock screen) should carry both lines in its credits. See
NASA's media usage guidelines for the terms of use.
