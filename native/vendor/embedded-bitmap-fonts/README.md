# embedded-bitmap-fonts

A comprehensive collection of bitmap fonts for `no_std` environments, designed for use with the [`embedded-graphics`](https://crates.io/crates/embedded-graphics) crate.

## Features

- **27 font families** with 134 font variants
- **Pixel-doubling support** - Scale fonts 2x, 3x, or more with zero additional flash cost
- **No-std compatible** - Works on embedded systems without an allocator
- **Feature flags** - Include only the fonts you need to minimize binary size

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
embedded-bitmap-fonts = { version = "0.1", features = ["terminus"] }
```

Draw text:

```rust
use embedded_bitmap_fonts::{TextStyle, terminus::FONT_8x16};
use embedded_graphics::{pixelcolor::BinaryColor, prelude::*, text::Text};

let style = TextStyle::new(&FONT_8x16, BinaryColor::On);
Text::new("Hello!", Point::zero(), style).draw(&mut display)?;
```

### Pixel Doubling

Scale fonts without additional memory cost:

```rust
use embedded_bitmap_fonts::terminus::FONT_8x16;

// Original 8x16 font
let small = &FONT_8x16;

// Pixel-doubled 16x32 font (same bitmap data!)
let large = FONT_8x16.pixel_double();

// Triple size 24x48
let huge = FONT_8x16.pixel_triple();

// Custom multiplier
let custom = FONT_8x16.pixel_multiply(4);
```

## Available Fonts

Enable fonts using feature flags:

```toml
# Single font
embedded-bitmap-fonts = { version = "0.1", features = ["terminus"] }

# Multiple fonts
embedded-bitmap-fonts = { version = "0.1", features = ["terminus", "tamzen", "cherry"] }

# All fonts (not recommended for embedded - large binary size)
embedded-bitmap-fonts = { version = "0.1", features = ["all-fonts"] }
```

| Feature | Variants | Available Fonts |
|---------|----------|-----------------|
| `artwiz` | 12 | `FONT_5x10`, `FONT_6x10`, `FONT_6x11`, `FONT_6x13`, `FONT_7x14`, `FONT_8x8`, `FONT_8x10`, `FONT_8x16`, `FONT_9x9`, `FONT_9x10`, `FONT_10x10_BOLD`, `FONT_11x12` |
| `bitocra` | 7 | `FONT_4x4`, `FONT_4x5`, `FONT_4x7`, `FONT_6x10`, `FONT_6x11`, `FONT_7x11`, `FONT_7x13` |
| `cherry` | 6 | `FONT_6x10`, `FONT_6x10_BOLD`, `FONT_6x11`, `FONT_6x11_BOLD`, `FONT_7x13`, `FONT_7x13_BOLD` |
| `creep` | 1 | `FONT_7x12` |
| `ctrld` | 6 | `FONT_6x10`, `FONT_6x10_BOLD`, `FONT_7x13`, `FONT_7x13_BOLD`, `FONT_8x16`, `FONT_8x16_BOLD` |
| `dina` | 4 | `FONT_6x10`, `FONT_7x13`, `FONT_7x15`, `FONT_8x16` |
| `dylex` | 9 | `FONT_6x10`, `FONT_6x10_BOLD`, `FONT_7x13`, `FONT_7x13_BOLD`, `FONT_10x20`, `FONT_10x20_BOLD`, `FONT_14x14`, `FONT_14x14_BOLD`, `FONT_14x20` |
| `envypn` | 2 | `FONT_7x13`, `FONT_7x15` |
| `gohufont` | 4 | `FONT_6x11`, `FONT_6x11_BOLD`, `FONT_8x14`, `FONT_8x14_BOLD` |
| `gomme` | 1 | `FONT_10x20` |
| `haxor` | 6 | `FONT_6x11`, `FONT_7x13`, `FONT_8x14`, `FONT_8x17`, `FONT_9x15`, `FONT_9x21` |
| `jmk` | 13 | `FONT_5x10`, `FONT_5x11`, `FONT_5x11_BOLD`, `FONT_6x11`, `FONT_6x11_BOLD`, `FONT_6x13`, `FONT_6x13_BOLD`, `FONT_8x15`, `FONT_8x15_BOLD`, `FONT_10x20`, `FONT_10x20_BOLD`, `FONT_12x24`, `FONT_12x24_BOLD` |
| `kakwa` | 2 | `FONT_6x12`, `FONT_6x12_BOLD` |
| `knxt` | 1 | `FONT_9x20` |
| `lokaltog` | 3 | `FONT_6x10`, `FONT_6x12`, `FONT_11x11` |
| `mplus` | 10 | `FONT_6x13`, `FONT_6x13_BOLD`, `FONT_7x10`, `FONT_7x11`, `FONT_8x11_BOLD`, `FONT_9x11`, `FONT_9x11_BOLD`, `FONT_10x14`, `FONT_11x14_BOLD`, `FONT_27x29` |
| `orp` | 2 | `FONT_6x12`, `FONT_6x12_BOLD` |
| `peep` | 3 | `FONT_7x14`, `FONT_8x16`, `FONT_10x20` |
| `phallus` | 2 | `FONT_10x12`, `FONT_10x15` |
| `progsole` | 1 | `FONT_6x9` |
| `scientifica` | 2 | `FONT_5x12`, `FONT_5x12_BOLD` |
| `spleen` | 6 | `FONT_5x8`, `FONT_6x12`, `FONT_8x16`, `FONT_12x24`, `FONT_16x32`, `FONT_32x64` |
| `tamzen` | 14 | `FONT_5x9`, `FONT_5x9_BOLD`, `FONT_6x12`, `FONT_6x12_BOLD`, `FONT_7x13`, `FONT_7x13_BOLD`, `FONT_7x14`, `FONT_7x14_BOLD`, `FONT_8x15`, `FONT_8x15_BOLD`, `FONT_8x16`, `FONT_8x16_BOLD`, `FONT_10x20`, `FONT_10x20_BOLD` |
| `terminus` | 18 | `FONT_6x12`, `FONT_6x12_BOLD`, `FONT_8x14`, `FONT_8x14_BOLD`, `FONT_8x16`, `FONT_8x16_BOLD`, `FONT_10x18`, `FONT_10x18_BOLD`, `FONT_10x20`, `FONT_10x20_BOLD`, `FONT_11x22`, `FONT_11x22_BOLD`, `FONT_12x24`, `FONT_12x24_BOLD`, `FONT_14x28`, `FONT_14x28_BOLD`, `FONT_16x32`, `FONT_16x32_BOLD` |
| `tewi` | 4 | `FONT_5x5`, `FONT_5x5_BOLD`, `FONT_7x13_BOLD`, `FONT_13x12` |
| `trisk` | 2 | `FONT_8x13`, `FONT_8x14_BOLD` |
| `xbmicons` | 1 | `FONT_9x12` |

**Total: 134 fonts across 27 families**

## Contributing 

Fonts are generated from BDF (Bitmap Distribution Format) files using a custom codegen tool.

### Prerequisites

```bash
git clone https://github.com/ecklf/embedded-bitmap-fonts
cd embedded-bitmap-fonts
cargo build --release -p embedded-bitmap-fonts-codegen
```

### Regenerate All Fonts

```bash
./generate_all.sh
```

### Generate a Single Font

```bash
./target/release/embedded-bitmap-fonts-codegen \
  -i ./fonts/bitmap/terminus \
  -o ./src \
  -f terminus
```

### Codegen Options

```
-i, --input <PATH>     Input directory containing BDF files
-o, --output <PATH>    Output directory for generated Rust files
-f, --family <NAME>    Font family name (used for module name)
-P, --pattern <PAT>    Include only files matching pattern
-X, --exclude <PAT>    Exclude files matching pattern
```

### Adding New Fonts

1. Add BDF files to `fonts/bitmap/<family_name>/`
2. Add entry to `generate_all.sh`
3. Add feature flag to `Cargo.toml`
4. Add module to `src/lib.rs`
5. Run `./generate_all.sh`

## Font Sources

Fonts are sourced from the [Tecate/bitmap-fonts](https://github.com/Tecate/bitmap-fonts) collection.

## License

This crate is licensed under Apache-2.0. Individual fonts may have their own licenses - check the original font sources for details.
