//! One production boot asset shared by hosted examples and freestanding
//! runtimes. Runtime owns time, input, allocation, scheduling and presentation.
//! This crate owns layout, prepared font coverage and the baked Kerr scene,
//! never a CPU renderer.
//!
//! The scene behind the welcome overlay is a Kerr black hole traced once, at
//! build time (`build/kerr.rs`): each pixel's geodesic is fixed, and what
//! moves along it is the disk's turbulence and the camera's slow swing about
//! the spin axis ([`orbit_angle`]), which turns sky and disk alike.
//!
//! A frame is two passes. The hole pass ([`HoleInputs`], [`HOLE_GRAPH`])
//! draws the black hole at [`Viewport::hole_frame`] — the lens map's own
//! resolution on displays taller than it, one texel per pixel — sampling
//! the baked lens map into per-pixel planes, the turbulence table
//! ([`turbulence`]) at the gas's current position and the sky atlas
//! ([`sky_atlas`]) at the swung camera's view. The compose pass
//! ([`ComposeInputs`], [`COMPOSE_GRAPH`]) upsamples that to the display and
//! lays the overlay over it at full resolution, on the tiles the overlay
//! reaches; the rest are the hole alone. The hole's cost is then the same on
//! any panel, and the text is drawn at the panel's own. A display no taller
//! than the lens map draws the hole at its own size, so each tile composes
//! straight over that tile's hole planes ([`Hole::Tile`]), with no frame
//! between the passes.
#![no_std]
#![feature(allocator_api)]
extern crate alloc;

use alloc::vec::Vec;
use core::alloc::Allocator;
#[allow(unused_imports)]
use core_maths::CoreFloat;

pub mod kerr_params;
pub use kerr_params::*;

/// Constants derived by the bake: the disk's inner edge, the packed
/// redshift range, and the sky atlas's grid.
pub mod kerr {
    include!(concat!(env!("OUT_DIR"), "/kerr.rs"));
}

pub const TILE_WIDTH: usize = 128;
pub const TILE_HEIGHT: usize = 32;
pub const TILE_PIXELS: usize = TILE_WIDTH * TILE_HEIGHT;
/// The hole pass: the black hole alone, at [`Viewport::hole_frame`].
pub const HOLE_GRAPH: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/hole.ax"));
/// The hole pass as compiled: the authored asset with the bake's constants.
pub const HOLE_SOURCE: &str = include_str!(concat!(env!("OUT_DIR"), "/hole.wgsl"));
/// The compose pass: the overlay over the hole, at the display's resolution.
pub const COMPOSE_GRAPH: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/compose.ax"));
pub const COMPOSE_SOURCE: &str = include_str!("../assets/compose.wgsl");

/// The animation's frame period while the welcome page is up: the disk turns
/// slowly (an inner orbit takes tens of seconds) and the camera swings slower
/// still, so 15 frames a second is smooth, and a runtime parks between them.
pub const FRAME_NS: u64 = 66_666_667;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidViewport,
    InvalidRegion,
    Allocation,
}

/// Calendar input, not uptime. UTC until the final runtime supplies a zone.
/// Missing calibration is visibly unavailable, never a plausible fake time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Clock {
    minute: Option<u16>,
}
impl Clock {
    pub fn from_unix_seconds(seconds: Option<u64>) -> Self {
        Self {
            minute: seconds.map(|s| ((s / 60) % 1440) as u16),
        }
    }
    pub fn text(self) -> [u8; 5] {
        match self.minute {
            Some(m) => [
                b'0' + (m / 60 / 10) as u8,
                b'0' + (m / 60 % 10) as u8,
                b':',
                b'0' + (m % 60 / 10) as u8,
                b'0' + (m % 10) as u8,
            ],
            None => *b"--:--",
        }
    }
    /// Delay to the next minute boundary. Failed wall calibration is retried
    /// at most once a minute; runtimes should park, not poll between updates.
    pub fn next_update_ns(unix_nanos: Option<u64>) -> u64 {
        const MINUTE: u64 = 60_000_000_000;
        unix_nanos.map_or(MINUTE, |n| MINUTE - n % MINUTE)
    }
}

/// The scene's animation clock: whole frames since the page appeared. Equal
/// frames draw identical scenes, so a tile repaints only when the frame moves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Frame(pub u64);
impl Frame {
    pub fn at_uptime_ns(elapsed_ns: u64) -> Self {
        Self(elapsed_ns / FRAME_NS)
    }
    /// Seconds of animation this frame shows.
    pub fn seconds(self) -> f32 {
        (self.0 as f64 * FRAME_NS as f64 / 1e9) as f32
    }
    /// Delay from `elapsed_ns` to the next frame boundary.
    pub fn next_update_ns(elapsed_ns: u64) -> u64 {
        FRAME_NS - elapsed_ns % FRAME_NS
    }
}

/// One full swing of the camera, there and back, in seconds.
pub const ORBIT_PERIOD_S: f64 = 240.0;

/// The camera's swing around the spin axis at `frame`, in radians, within
/// ±`ORBIT_SWAY_DEG` (the bake padded the sky atlas by exactly that much).
/// A slow sine: at its fastest it turns the sky 0.26° a second — under a
/// pixel per 1/15 s frame for directly seen stars at 1920x1200 — and it eases
/// to rest at each end instead of turning on a dime.
pub fn orbit_angle(frame: Frame) -> f32 {
    let seconds = frame.0 as f64 * FRAME_NS as f64 / 1e9;
    let phase = (seconds / ORBIT_PERIOD_S).fract() * core::f64::consts::TAU;
    (f64::from(ORBIT_SWAY_DEG).to_radians() * phase.sin()) as f32
}

/// Physical viewport with resolution-independent layout. Both axes remain
/// anchored to their edges. The baked scene fits the viewport's height and
/// is centred horizontally.
#[derive(Clone, Copy, Debug)]
pub struct Viewport {
    width: u32,
    height: u32,
    scale: f32,
}
impl Viewport {
    pub fn new(width: u32, height: u32) -> Result<Self, Error> {
        if !(320..=7680).contains(&width) || !(240..=4320).contains(&height) {
            return Err(Error::InvalidViewport);
        }
        Ok(Self {
            width,
            height,
            scale: (width as f32 / 1920.0).min(height as f32 / 1200.0),
        })
    }
    pub fn size(self) -> [u32; 2] {
        [self.width, self.height]
    }
    /// Lens texels per physical pixel: the baked frame fits the height.
    fn lens_step(self) -> f32 {
        LENS_HEIGHT as f32 / self.height as f32
    }
    /// The frame the hole pass draws in. A display taller than the lens map
    /// draws the hole at the lens map's height — one texel per pixel, the
    /// direct read — and upsamples it; the width follows the display's
    /// aspect, rounded up to even so texel centres fall on pixel centres.
    /// Other displays draw the hole at their own size.
    pub fn hole_frame(self) -> Viewport {
        if !self.upsamples_hole() {
            return self;
        }
        let height = LENS_HEIGHT as u32;
        let width = (u64::from(self.width) * u64::from(height)).div_ceil(u64::from(self.height)) as u32;
        Self::new((width + (width & 1)).clamp(320, 7680), height).expect("a lens-height frame of admitted width")
    }

    /// Whether the hole is drawn smaller than the display and upsampled.
    pub fn upsamples_hole(self) -> bool {
        self.height > LENS_HEIGHT as u32
    }

    /// The hole frame's point under display pixel `(x, y)`'s centre, in hole
    /// pixels (centres on integers).
    fn hole_point(self, hole: Viewport, x: f32, y: f32) -> [f32; 2] {
        [
            (x + 0.5) * hole.width as f32 / self.width as f32 - 0.5,
            (y + 0.5) * hole.height as f32 / self.height as f32 - 0.5,
        ]
    }

    /// Whether any of the overlay — its text, chips, the seat pill and the
    /// pill's shadow — can reach the tile at `origin`. Tiles it cannot reach
    /// are the hole alone and skip the compose pass.
    pub fn overlays_tile(self, origin: [u32; 2]) -> bool {
        let s = self.scale;
        let w = self.width as f32 / s;
        let h = self.height as f32 / s;
        let (chips, pill, _) = self.shape_bounds();
        let tile = [
            origin[0] as f32,
            origin[1] as f32,
            (origin[0] as usize + TILE_WIDTH) as f32,
            (origin[1] as usize + TILE_HEIGHT) as f32,
        ];
        // The clock, its zone, the copy, the chips' labels and the hint; the
        // chips and the pill with its shadow; the header's two labels, at
        // their measured widths (as prepare_text places them) with a margin.
        let locked = MASKS[TEXT_LOCKED].width as f32 / 2.0;
        let label = MASKS[TEXT_KERR].width as f32 / 2.0;
        [
            [80.0, h - 450.0, 800.0, h - 50.0],
            chips,
            pill,
            [88.0, 58.0, 96.0 + locked + 8.0, 89.0],
            [w - 96.0 - label - 8.0, 58.0, w - 88.0, 89.0],
        ]
        .iter()
            .any(|b| tile[0] < b[2] * s && tile[2] > b[0] * s && tile[1] < b[3] * s && tile[3] > b[1] * s)
    }

    /// The chips row and the seat pill (with its shadow's reach) in design
    /// pixels: the tiles the shape coverage computes anything for.
    fn shape_bounds(self) -> ([f32; 4], [f32; 4], [f32; 2]) {
        let w = self.width as f32 / self.scale;
        let h = self.height as f32 / self.scale;
        let chip_y = h - CHIP_CENTRE_Y;
        let chips = [
            CHIPS[0][0] - CHIPS[0][1] - CHIP_RADIUS - 2.0,
            chip_y - CHIP_RADIUS - 2.0,
            CHIPS[2][0] + CHIPS[2][1] + CHIP_RADIUS + 2.0,
            chip_y + CHIP_RADIUS + 2.0,
        ];
        let pill = [w - PILL_RIGHT - PILL_HALF - PILL_RADIUS, h - PILL_CENTRE_Y];
        // The shadow sits 12 px low and is cut at 4 sigma.
        let reach = PILL_RADIUS + SHADOW_SIGMA * 4.0 + 2.0;
        let pill_box = [
            pill[0] - PILL_HALF - reach,
            pill[1] - reach,
            pill[0] + PILL_HALF + reach,
            pill[1] + SHADOW_OFFSET + reach,
        ];
        (chips, pill_box, pill)
    }
}

// ─────────────────────────── the baked scene ───────────────────────────

#[repr(C, align(4))]
struct Aligned<T: ?Sized>(T);

static LENS: &Aligned<[u8]> = &Aligned(*include_bytes!(concat!(env!("OUT_DIR"), "/kerr_lens.bin")));
static TURBULENCE: &Aligned<[u8]> =
    &Aligned(*include_bytes!(concat!(env!("OUT_DIR"), "/kerr_turbulence.bin")));
static SKY: &Aligned<[u8]> = &Aligned(*include_bytes!(concat!(env!("OUT_DIR"), "/kerr_sky.bin")));

/// `[u16; 3]` points of a baked UF16 RGB table.
fn rgb_points(bytes: &'static [u8]) -> &'static [[u16; 3]] {
    // SAFETY: the bake writes whole little-endian `[u16; 3]` points; the
    // wrapper aligns the bytes to 4, and kore's graph ABI is little-endian.
    unsafe { core::slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len() / 6) }
}

/// The turbulence table as the shader's `@group(1) @binding(0)` RGB volume:
/// UF16 E5M11 points, x fastest, `TURBULENCE_COLUMNS + 1` wide (the last
/// column repeats the first, so bilinear sampling wraps in φ) by
/// `TURBULENCE_ROWS` rows of ln r, one deep. Red and green are the kit's two
/// noise scales; blue is zero.
pub fn turbulence() -> &'static [[u16; 3]] {
    rgb_points(&TURBULENCE.0)
}

/// The turbulence volume's grid dimensions, as the shader was compiled for.
pub const TURBULENCE_DIMS: [u32; 3] = [TURBULENCE_COLUMNS as u32 + 1, TURBULENCE_ROWS as u32, 1];

/// The shader's binding for [`turbulence`]: group 1, binding 0.
pub const TURBULENCE_BINDING: (u32, u32) = (1, 0);

/// The sky atlas as the shader's `@group(1) @binding(1)` RGB volume: UF16
/// E5M11 points, x fastest, [`SKY_DIMS`] in size — the sky prefiltered into
/// tiers (a fine patch where the camera sees it nearly unlensed, spheres at
/// coarser steps for the lensed images), each padded by the camera's swing.
pub fn sky_atlas() -> &'static [[u16; 3]] {
    rgb_points(&SKY.0)
}

/// The sky atlas's grid dimensions, as the shader was compiled for.
pub const SKY_DIMS: [u32; 3] = kerr::SKY_DIMS;

/// The shader's binding for [`sky_atlas`]: group 1, binding 1.
pub const SKY_BINDING: (u32, u32) = (1, 1);

const _: () = assert!(LENS_WIDTH * LENS_HEIGHT * 16 == include_bytes!(concat!(env!("OUT_DIR"), "/kerr_lens.bin")).len());
const _: () = assert!(
    (TURBULENCE_COLUMNS + 1) * TURBULENCE_ROWS * 6
        == include_bytes!(concat!(env!("OUT_DIR"), "/kerr_turbulence.bin")).len()
);
const _: () = assert!(
    (SKY_DIMS[0] * SKY_DIMS[1]) as usize * 6 == include_bytes!(concat!(env!("OUT_DIR"), "/kerr_sky.bin")).len()
);

/// One lens texel: its sky-atlas point, first and second crossings, and the
/// coverage bytes.
fn lens_texel(x: i32, y: i32) -> [u32; 4] {
    let x = x.clamp(0, LENS_WIDTH as i32 - 1) as usize;
    let y = y.clamp(0, LENS_HEIGHT as i32 - 1) as usize;
    let at = (y * LENS_WIDTH + x) * 16;
    let word = |i: usize| u32::from_le_bytes(LENS.0[at + i * 4..at + i * 4 + 4].try_into().unwrap());
    [word(0), word(1), word(2), word(3)]
}

/// A texel's sky-atlas point (sixteenths of a texel, packed u16 pairs).
fn decode_sky(bits: u32) -> [f32; 2] {
    [(bits & 0xffff) as f32 / 16.0, (bits >> 16) as f32 / 16.0]
}

/// Lens neighbours whose sky points lie further apart than this (in atlas
/// texels) see different images, or different tiers: they are not blended.
const SKY_BLEND_REACH: f32 = 2.0;

/// A crossing's ln r, azimuth and redshift, or `None`.
fn decode_crossing(bits: u32) -> Option<[f32; 3]> {
    let code = bits & 2047;
    if code == 0 {
        return None;
    }
    let r = kerr::LN_R[code as usize];
    let phi = ((bits >> 11) & 2047) as f32 / 2048.0 * core::f32::consts::TAU;
    let g = kerr::G_MIN + (bits >> 22) as f32 / 1023.0 * (kerr::G_MAX - kerr::G_MIN);
    Some([r, phi, g])
}

/// A difference of two azimuths in [0, τ), wrapped into [−π, π). The
/// inputs bound it to (−τ, τ), so one compare-and-shift suffices where
/// `floor` would cost a libm call per sample in this no_std crate.
fn wrap_angle(a: f32) -> f32 {
    use core::f32::consts::{PI, TAU};
    if a >= PI {
        a - TAU
    } else if a < -PI {
        a + TAU
    } else {
        a
    }
}

/// Blend one crossing slot of four neighbours with premultiplied coverage:
/// a neighbour without that crossing contributes zero coverage, so a disk
/// silhouette anti-aliases instead of inventing a radius between the two
/// sides. Neighbours that see a different image of the disk (radius or
/// azimuth far from the strongest one's) are left out, not averaged in.
fn blend_crossing(samples: [(f32, Option<[f32; 3]>); 4]) -> [f32; 4] {
    let Some((_, reference)) = samples
        .iter()
        .filter_map(|&(w, c)| c.map(|c| (w, c)))
        .max_by(|a, b| a.0.total_cmp(&b.0))
    else {
        return [0.0; 4];
    };
    let [r_ref, phi_ref, _] = reference;
    let (mut cover, mut r, mut dphi, mut g) = (0.0, 0.0, 0.0, 0.0);
    for (w, c) in samples {
        let Some([cr, cphi, cg]) = c else { continue };
        let d = wrap_angle(cphi - phi_ref);
        // Radii are ln r: 0.14 is about the 15% the bake groups images by.
        if (cr - r_ref).abs() > 0.14 || d.abs() > 0.35 {
            continue;
        }
        cover += w;
        r += w * cr;
        dphi += w * d;
        g += w * cg;
    }
    // A sample exactly on a texel edge gives its far neighbours no weight:
    // no coverage means no crossing, never a 0/0.
    if cover <= 0.0 {
        return [0.0; 4];
    }
    [cover, r / cover, phi_ref + dphi / cover, g / cover]
}

/// Blend the sky of four neighbours: coverage blends as premultiplied alpha,
/// so the shadow's edge anti-aliases; the sky point is the coverage-weighted
/// mean of the neighbours within reach of the strongest one, which at one
/// texel per pixel is that texel's own point.
fn blend_sky(samples: [(f32, [f32; 2]); 4]) -> [f32; 3] {
    let cover: f32 = samples.iter().map(|s| s.0).sum();
    let (_, reference) = samples.iter().copied().max_by(|a, b| a.0.total_cmp(&b.0)).unwrap();
    let (mut weight, mut x, mut y) = (0.0, 0.0, 0.0);
    for (w, [sx, sy]) in samples {
        if w > 0.0 && (sx - reference[0]).abs() <= SKY_BLEND_REACH && (sy - reference[1]).abs() <= SKY_BLEND_REACH {
            weight += w;
            x += w * sx;
            y += w * sy;
        }
    }
    if weight <= 0.0 {
        return [reference[0], reference[1], 0.0];
    }
    [x / weight, y / weight, cover]
}

/// The hole pass's pixels as the compose pass reads them: packed XRGB words,
/// row major, `width` to a row.
#[derive(Clone, Copy)]
pub struct HoleImage<'a> {
    pub pixels: &'a [u32],
    pub width: usize,
    pub height: usize,
}

impl HoleImage<'_> {
    fn channel(word: u32, shift: u32) -> f32 {
        ((word >> shift) & 0xff) as f32 / 255.0
    }

    /// The pixel `(x, y)`, clamped to the image: RGB in [0, 1].
    fn pixel(&self, x: i32, y: i32) -> [f32; 3] {
        let x = x.clamp(0, self.width as i32 - 1) as usize;
        let y = y.clamp(0, self.height as i32 - 1) as usize;
        let word = self.pixels[y * self.width + x];
        [Self::channel(word, 16), Self::channel(word, 8), Self::channel(word, 0)]
    }

    /// Bilinear RGB at hole point `(x, y)`, centres on integers.
    fn sample(&self, x: f32, y: f32) -> [f32; 3] {
        let (ix, iy) = (x.floor(), y.floor());
        let (fx, fy) = (x - ix, y - iy);
        let (ix, iy) = (ix as i32, iy as i32);
        let [a, b, c, d] = [self.pixel(ix, iy), self.pixel(ix + 1, iy), self.pixel(ix, iy + 1), self.pixel(ix + 1, iy + 1)];
        core::array::from_fn(|k| (a[k] * (1.0 - fx) + b[k] * fx) * (1.0 - fy) + (c[k] * (1.0 - fx) + d[k] * fx) * fy)
    }
}

/// The hole under a compose tile.
#[derive(Clone, Copy)]
pub enum Hole<'a> {
    /// The whole hole frame as the hole pass drew it — how a display taller
    /// than the lens map reads it, upsampled.
    Image(HoleImage<'a>),
    /// The hole pass's own RGB output for this very tile, on a display that
    /// draws the hole at its own size: it composes each tile straight after
    /// its hole, unquantized, with no frame between them.
    Tile([&'a [f32]; 3]),
}

fn plane<A: Allocator>(allocator: A) -> Result<Vec<f32, A>, Error> {
    let mut v = Vec::new_in(allocator);
    v.try_reserve_exact(TILE_PIXELS).map_err(|_| Error::Allocation)?;
    v.resize(TILE_PIXELS, 0.0);
    Ok(v)
}

/// The hole pass's reusable inputs for one tile of [`Viewport::hole_frame`].
/// Allocation and graph admission happen once; a new frame does not rebuild
/// WGSL, .ax or the scene.
pub struct HoleInputs<A: Allocator> {
    pub x: [f32; TILE_WIDTH],
    pub y: [f32; TILE_HEIGHT],
    /// Where the lensed sky behind the hole is read: the sky atlas point (x,
    /// y) and the fraction of the pixel's rays that escape.
    pub sky: [Vec<f32, A>; 3],
    /// The first and second disk crossings: coverage, ln r, φ, g.
    pub near: [Vec<f32, A>; 4],
    pub far: [Vec<f32, A>; 4],
    pub uniforms: [[f32; 1]; 4],
}

/// The compose pass's reusable inputs for one display tile: the hole there
/// and the overlay's coverage.
pub struct ComposeInputs<A: Allocator> {
    pub x: [f32; TILE_WIDTH],
    pub y: [f32; TILE_HEIGHT],
    /// The hole pass's colour at each display pixel, upsampled where it was
    /// drawn smaller.
    pub hole: [Vec<f32, A>; 3],
    pub ink: [Vec<f32, A>; 2],
    /// The overlay's shape coverage: shadow, dark fill, the avatar's green.
    pub shape: [Vec<f32, A>; 3],
}

/// One lens texel, decoded: its sky point and escape coverage byte, and each
/// crossing with its coverage.
#[derive(Clone, Copy)]
struct Texel {
    sky: [f32; 2],
    /// The sky's coverage byte as F32 (0–255): its blend weight rounds as
    /// `w * byte / 255`.
    sky_cover: f32,
    crossing: [Option<[f32; 3]>; 2],
    cover: [f32; 2],
}

impl Texel {
    fn decode(words: [u32; 4]) -> Self {
        let cover = |byte: u32| ((words[3] >> (8 * byte)) & 0xff) as f32 / 255.0;
        Self {
            sky: decode_sky(words[0]),
            sky_cover: ((words[3] >> 16) & 0xff) as f32,
            crossing: [decode_crossing(words[1]), decode_crossing(words[2])],
            cover: [cover(0), cover(1)],
        }
    }
}

/// Pixel centres of the tile at `origin`.
fn positions(origin: [u32; 2]) -> ([f32; TILE_WIDTH], [f32; TILE_HEIGHT]) {
    (
        core::array::from_fn(|i| origin[0] as f32 + i as f32 + 0.5),
        core::array::from_fn(|i| origin[1] as f32 + i as f32 + 0.5),
    )
}

impl<A: Allocator + Clone> HoleInputs<A> {
    pub fn new_in(allocator: A) -> Result<Self, Error> {
        let p = || plane(allocator.clone());
        Ok(Self {
            x: [0.0; TILE_WIDTH],
            y: [0.0; TILE_HEIGHT],
            sky: [p()?, p()?, p()?],
            near: [p()?, p()?, p()?, p()?],
            far: [p()?, p()?, p()?, p()?],
            uniforms: [[0.0]; 4],
        })
    }

    /// Prepare the tile at `origin` of `hole` (a [`Viewport::hole_frame`]).
    pub fn prepare(&mut self, hole: Viewport, origin: [u32; 2], frame: Frame) -> Result<(), Error> {
        if origin[0] >= hole.width || origin[1] >= hole.height {
            return Err(Error::InvalidRegion);
        }
        (self.x, self.y) = positions(origin);
        self.uniforms = [[hole.width as f32], [hole.height as f32], [frame.seconds()], [orbit_angle(frame)]];
        self.prepare_scene(hole, origin);
        Ok(())
    }

    /// Sample the baked lens map under this tile: premultiplied per crossing
    /// slot, and for the sky a coverage blend with the sky point taken from
    /// the neighbours that see the same patch of sky.
    fn prepare_scene(&mut self, viewport: Viewport, origin: [u32; 2]) {
        let k = viewport.lens_step();
        let centre = viewport.width as f32 / 2.0;
        // One texel per pixel with texel centres on pixel centres — the
        // reference 1920x1200 display — reads each texel directly: the blend
        // below would weigh the texel under the pixel by exactly 1 and its
        // neighbours by 0.
        let x0 = (origin[0] as f32 + 0.5 - centre) * k + LENS_WIDTH as f32 / 2.0 - 0.5;
        let y0 = (origin[1] as f32 + 0.5) * k - 0.5;
        if k == 1.0 && x0 == x0.floor() && y0 == y0.floor() {
            self.prepare_scene_exact(x0 as i32, y0 as i32);
            return;
        }
        // Each column's and row's texel and blend fraction, once per tile.
        let columns: [(i32, f32); TILE_WIDTH] = core::array::from_fn(|column| {
            let lx = (origin[0] as f32 + column as f32 + 0.5 - centre) * k + LENS_WIDTH as f32 / 2.0 - 0.5;
            let ix = lx.floor();
            (ix as i32, lx - ix)
        });
        let rows: [(i32, f32); TILE_HEIGHT] = core::array::from_fn(|row| {
            let ly = (origin[1] as f32 + row as f32 + 0.5) * k - 0.5;
            let iy = ly.floor();
            (iy as i32, ly - iy)
        });
        for (row, &(iy, fy)) in rows.iter().enumerate() {
            for (column, &(ix, fx)) in columns.iter().enumerate() {
                let texel = |dx: i32, dy: i32| Texel::decode(lens_texel(ix + dx, iy + dy));
                let corners = [
                    (texel(0, 0), (1.0 - fx) * (1.0 - fy)),
                    (texel(1, 0), fx * (1.0 - fy)),
                    (texel(0, 1), (1.0 - fx) * fy),
                    (texel(1, 1), fx * fy),
                ];
                let at = row * TILE_WIDTH + column;
                let sky = blend_sky(corners.map(|(t, w)| (w * t.sky_cover / 255.0, t.sky)));
                for c in 0..3 {
                    self.sky[c][at] = sky[c];
                }
                for (slot, planes) in [&mut self.near, &mut self.far].into_iter().enumerate() {
                    // A texel's coverage of this crossing scales its weight:
                    // the bake's anti-aliasing carries through the upsample.
                    let blended = blend_crossing(corners.map(|(t, w)| (w * t.cover[slot], t.crossing[slot])));
                    for c in 0..4 {
                        planes[c][at] = blended[c];
                    }
                }
            }
        }
    }

    /// The lens map read one texel per pixel, from texel `(x0, y0)` at the
    /// tile's top left: the sky point and its escape coverage, and each
    /// crossing where the texel's rays see it.
    fn prepare_scene_exact(&mut self, x0: i32, y0: i32) {
        for row in 0..TILE_HEIGHT {
            for column in 0..TILE_WIDTH {
                let texel = lens_texel(x0 + column as i32, y0 + row as i32);
                let at = row * TILE_WIDTH + column;
                let [sx, sy] = decode_sky(texel[0]);
                self.sky[0][at] = sx;
                self.sky[1][at] = sy;
                self.sky[2][at] = ((texel[3] >> 16) & 0xff) as f32 / 255.0;
                for (slot, planes) in [(1, &mut self.near), (2, &mut self.far)] {
                    let coverage = ((texel[3] >> (8 * (slot - 1))) & 0xff) as f32 / 255.0;
                    let crossing = match decode_crossing(texel[slot]) {
                        Some([r, phi, g]) if coverage > 0.0 => [coverage, r, phi, g],
                        _ => [0.0; 4],
                    };
                    for c in 0..4 {
                        planes[c][at] = crossing[c];
                    }
                }
            }
        }
    }

    /// The hole pass's ABI: position x/y, the sky's atlas point and coverage,
    /// the near and far crossings (coverage, ln r, φ, g), the hole frame's
    /// size, the animation's time in seconds and the camera's swing in
    /// radians. All planes are F32. The turbulence table and the sky atlas
    /// bind separately ([`turbulence`], [`sky_atlas`]).
    pub fn planes(&self) -> [&[f32]; 17] {
        [
            &self.x,
            &self.y,
            &self.sky[0],
            &self.sky[1],
            &self.sky[2],
            &self.near[0],
            &self.near[1],
            &self.near[2],
            &self.near[3],
            &self.far[0],
            &self.far[1],
            &self.far[2],
            &self.far[3],
            &self.uniforms[0],
            &self.uniforms[1],
            &self.uniforms[2],
            &self.uniforms[3],
        ]
    }
}

impl<A: Allocator + Clone> ComposeInputs<A> {
    pub fn new_in(allocator: A) -> Result<Self, Error> {
        let p = || plane(allocator.clone());
        Ok(Self {
            x: [0.0; TILE_WIDTH],
            y: [0.0; TILE_HEIGHT],
            hole: [p()?, p()?, p()?],
            ink: [p()?, p()?],
            shape: [p()?, p()?, p()?],
        })
    }

    /// Prepare the display tile at `origin`: the hole under it — from the
    /// frame drawn at `viewport.hole_frame()`, or the tile's own hole planes
    /// on a display that draws it at its own size — and, where the overlay
    /// reaches the tile, its coverage at `clock`. Returns whether it does: a
    /// tile it doesn't reach is the hole alone, and its `hole` planes are the
    /// tile.
    pub fn prepare(&mut self, viewport: Viewport, origin: [u32; 2], clock: Clock, hole: Hole<'_>) -> Result<bool, Error> {
        let frame = viewport.hole_frame();
        let admitted = match hole {
            Hole::Image(image) => {
                image.width == frame.width as usize
                    && image.height == frame.height as usize
                    && image.pixels.len() >= image.width * image.height
            }
            Hole::Tile(planes) => !viewport.upsamples_hole() && planes.iter().all(|p| p.len() == TILE_PIXELS),
        };
        if origin[0] >= viewport.width || origin[1] >= viewport.height || !admitted {
            return Err(Error::InvalidRegion);
        }
        (self.x, self.y) = positions(origin);
        match hole {
            Hole::Image(image) => self.prepare_hole(viewport, frame, origin, image),
            Hole::Tile(planes) => {
                for (plane, source) in self.hole.iter_mut().zip(planes) {
                    plane.copy_from_slice(source);
                }
            }
        }
        let overlaid = viewport.overlays_tile(origin);
        if overlaid {
            self.prepare_text(viewport, origin, clock);
            self.prepare_shapes(viewport, origin);
        } else {
            // Nothing covers the tile: keep a stale tile's coverage out.
            for plane in self.ink.iter_mut().chain(&mut self.shape) {
                plane.fill(0.0);
            }
        }
        Ok(overlaid)
    }

    /// The hole under the tile. Drawn at the display's own size it is read
    /// pixel for pixel — only the tile's own pixels, so tiles composed in
    /// parallel never read each other's — and upsampled bilinearly otherwise.
    fn prepare_hole(&mut self, viewport: Viewport, hole: Viewport, origin: [u32; 2], image: HoleImage<'_>) {
        if !viewport.upsamples_hole() {
            for row in 0..TILE_HEIGHT {
                for column in 0..TILE_WIDTH {
                    let colour = image.pixel((origin[0] as usize + column) as i32, (origin[1] as usize + row) as i32);
                    for c in 0..3 {
                        self.hole[c][row * TILE_WIDTH + column] = colour[c];
                    }
                }
            }
            return;
        }
        // [`HoleImage::sample`] with its source pixels and weights hoisted:
        // a column's taps and weight are the same on every row, and a row's
        // on every column. The arithmetic is the sample's, in its order.
        let taps = |at: f32, extent: usize| {
            let floor = at.floor();
            let first = floor as i32;
            let clamp = |i: i32| i.clamp(0, extent as i32 - 1) as usize;
            (clamp(first), clamp(first + 1), at - floor)
        };
        let columns: [(usize, usize, f32); TILE_WIDTH] = core::array::from_fn(|column| {
            let x = (origin[0] as usize + column) as f32;
            taps(viewport.hole_point(hole, x, 0.0)[0], image.width)
        });
        for row in 0..TILE_HEIGHT {
            let y = (origin[1] as usize + row) as f32;
            let (top, bottom, fy) = taps(viewport.hole_point(hole, 0.0, y)[1], image.height);
            let (top, bottom) = (&image.pixels[top * image.width..], &image.pixels[bottom * image.width..]);
            for (column, &(left, right, fx)) in columns.iter().enumerate() {
                let at = row * TILE_WIDTH + column;
                for (c, shift) in [16, 8, 0].into_iter().enumerate() {
                    let [a, b, cc, d] =
                        [top[left], top[right], bottom[left], bottom[right]].map(|word| HoleImage::channel(word, shift));
                    self.hole[c][at] = (a * (1.0 - fx) + b * fx) * (1.0 - fy) + (cc * (1.0 - fx) + d * fx) * fy;
                }
            }
        }
    }

    fn prepare_text(&mut self, viewport: Viewport, origin: [u32; 2], clock: Clock) {
        for plane in &mut self.ink {
            plane.fill(0.0);
        }
        let s = viewport.scale;
        let w = viewport.width as f32 / s;
        let h = viewport.height as f32 / s;
        // `rect` is [left, top, width, height] in design pixels; text centres
        // horizontally in it when `centered`, and sits on `baseline` (a shared
        // mask top) or centres vertically. `opacity` is the kit's ink alpha.
        let mut paint = |index: usize,
                         rect: [f32; 4],
                         centered: bool,
                         plane: usize,
                         baseline: Option<f32>,
                         opacity: f32| {
            let mask = MASKS[index];
            let left = rect[0]
                + if centered {
                    (rect[2] - mask.width as f32 / 2.0) * 0.5
                } else {
                    0.0
                };
            let top = match baseline {
                Some(top) => rect[1] + (mask.top as f32 - top) / 2.0,
                None => rect[1] + (rect[3] - mask.height as f32 / 2.0) * 0.5,
            };
            mask.paint(
                &mut self.ink[plane],
                origin,
                [left * s, top * s],
                s / 2.0,
                opacity,
            );
        };
        // Proportional figures: Fraunces Black's digits run from 79.5 px
        // ("1") to 119 px ("0"), so a tabular cell wide enough for "00"
        // leaves "1" adrift. Each glyph advances by its own width plus a
        // fixed gap, anchored at the left, so a minute never moves the hour.
        // A shared top reference keeps the colon on the digits' baseline.
        let clock_top = MASKS[..10].iter().map(|m| m.top).min().unwrap() as f32;
        let mut x = 96.0;
        let mut previous_colon = false;
        for c in clock.text() {
            let index = match c {
                b':' => TEXT_COLON,
                b'-' => TEXT_DASH,
                _ => (c - b'0') as usize,
            };
            let colon = index == TEXT_COLON;
            if x > 96.0 {
                x += if colon || previous_colon { CLOCK_COLON_GAP } else { CLOCK_GAP };
            }
            let width = MASKS[index].width as f32 / 2.0;
            paint(index, [x, h - 420.0, width, 177.0], false, 0, Some(clock_top), 1.0);
            x += width;
            previous_colon = colon;
        }
        paint(
            if clock.minute.is_some() { TEXT_UTC } else { TEXT_CLOCK_UNAVAILABLE },
            [99.0, h - 240.0, 300.0, 15.0],
            false,
            0,
            None,
            0.6,
        );
        let body_top = MASKS[TEXT_COPY_1].top.min(MASKS[TEXT_COPY_2].top) as f32;
        paint(TEXT_COPY_1, [99.0, h - 213.0, 700.0, 32.4], false, 0, Some(body_top), 0.92);
        paint(TEXT_COPY_2, [99.0, h - 180.6, 700.0, 32.4], false, 0, Some(body_top), 0.92);
        for (chip, index) in CHIPS.iter().zip([TEXT_ON_DEVICE, TEXT_LOCAL_WORKSPACE, TEXT_READY]) {
            let half = chip[1] + 15.0;
            paint(index, [chip[0] - half, h - 132.0, 2.0 * half, 30.0], true, 0, None, 0.85);
        }
        paint(TEXT_HINT, [99.0, h - 78.0, 700.0, 13.5], false, 0, None, 0.5);
        paint(TEXT_LOCKED, [96.0, 66.0, 400.0, 15.0], false, 0, None, 0.5);
        let label = MASKS[TEXT_KERR].width as f32 / 2.0;
        paint(TEXT_KERR, [w - 96.0 - label, 66.0, label, 15.0], false, 0, None, 0.5);

        // The seat pill, mirroring the shader's geometry.
        let pill_right = w - 96.0;
        let pill_left = pill_right - 2.0 * (PILL_HALF + 34.5);
        let pill_y = h - 124.5;
        paint(TEXT_AVATAR, [pill_left + 15.0, pill_y - 21.0, 42.0, 42.0], true, 1, None, 1.0);
        let label_x = pill_left + 15.0 + 42.0 + 18.0;
        paint(TEXT_NAME, [label_x, pill_y - 20.0, 200.0, 21.0], false, 0, None, 1.0);
        paint(TEXT_SESSION, [label_x, pill_y + 2.0, 200.0, 16.0], false, 0, None, 0.6);
        let button_width = 2.0 * (BUTTON_HALF + 21.0);
        paint(
            TEXT_ENTER,
            [pill_right - 12.0 - button_width, pill_y - 21.0, button_width, 42.0],
            true,
            1,
            None,
            1.0,
        );
    }

    /// The overlay's shapes as coverage, in the order the shader composes
    /// them: the pill's shadow, its dark fill, the avatar's green. The chips'
    /// faint fill and edge, the pill's edge and the Enter button are cream
    /// like the light text, so they join its plane as a union, `1 − (1−a)(1−b)`
    /// — the same composite as mixing each in turn. Only tiles under the chips
    /// row or the pill and its shadow compute anything.
    fn prepare_shapes(&mut self, viewport: Viewport, origin: [u32; 2]) {
        for plane in &mut self.shape {
            plane.fill(0.0);
        }
        let s = viewport.scale;
        let h = viewport.height as f32 / s;
        let aa = s.max(0.5);
        // Shapes are sampled at `front`, the design point under the pixel.
        let tile = [
            origin[0] as f32 / s,
            origin[1] as f32 / s,
            (origin[0] as usize + TILE_WIDTH) as f32 / s,
            (origin[1] as usize + TILE_HEIGHT) as f32 / s,
        ];
        let overlaps = |b: [f32; 4]| tile[0] < b[2] && tile[2] > b[0] && tile[1] < b[3] && tile[3] > b[1];
        let chip_y = h - CHIP_CENTRE_Y;
        let (chip_box, pill_box, pill) = viewport.shape_bounds();
        let (in_chips, in_pill) = (overlaps(chip_box), overlaps(pill_box));
        if !in_chips && !in_pill {
            return;
        }
        let avatar = [pill[0] - (PILL_HALF + PILL_RADIUS - AVATAR_INSET - AVATAR_RADIUS), pill[1]];
        let button = [pill[0] + (PILL_HALF + PILL_RADIUS - BUTTON_INSET - BUTTON_HALF - BUTTON_RADIUS), pill[1]];
        let cover = |d: f32| (0.5 - d * aa).clamp(0.0, 1.0);
        let edge = |d: f32| (0.5 - ((d + 0.75).abs() - 0.75) * aa).clamp(0.0, 1.0);
        for row in 0..TILE_HEIGHT {
            for column in 0..TILE_WIDTH {
                let front = [
                    (origin[0] as f32 + column as f32 + 0.5) / s,
                    (origin[1] as f32 + row as f32 + 0.5) / s,
                ];
                let at = row * TILE_WIDTH + column;
                let mut white = 0.0;
                if in_chips {
                    let d = CHIPS
                        .iter()
                        .map(|c| capsule(front, [c[0], chip_y], c[1], CHIP_RADIUS))
                        .fold(f32::MAX, f32::min);
                    white = 1.0 - (1.0 - 0.12 * cover(d)) * (1.0 - 0.22 * edge(d));
                }
                if in_pill {
                    let d = capsule(front, pill, PILL_HALF, PILL_RADIUS);
                    let shadow_d = capsule(front, [pill[0], pill[1] + SHADOW_OFFSET], PILL_HALF, PILL_RADIUS).max(0.0)
                        / SHADOW_SIGMA;
                    if shadow_d < 4.0 {
                        self.shape[0][at] = (-shadow_d * shadow_d).exp() * 0.40;
                    }
                    self.shape[1][at] = 0.55 * cover(d);
                    white = 1.0 - (1.0 - white) * (1.0 - 0.18 * edge(d));
                    let a = ((front[0] - avatar[0]).powi(2) + (front[1] - avatar[1]).powi(2)).sqrt() - AVATAR_RADIUS;
                    self.shape[2][at] = cover(a);
                    let b = cover(capsule(front, button, BUTTON_HALF, BUTTON_RADIUS));
                    white = 1.0 - (1.0 - white) * (1.0 - b);
                }
                let ink = &mut self.ink[0][at];
                *ink = 1.0 - (1.0 - *ink) * (1.0 - white);
            }
        }
    }

    /// The compose pass's ABI: position x/y, the hole's colour, light/dark
    /// ink coverage and shape coverage (shadow, dark fill, green). All F32.
    pub fn planes(&self) -> [&[f32]; 10] {
        [
            &self.x,
            &self.y,
            &self.hole[0],
            &self.hole[1],
            &self.hole[2],
            &self.ink[0],
            &self.ink[1],
            &self.shape[0],
            &self.shape[1],
            &self.shape[2],
        ]
    }
}

/// The overlay's fixed geometry in design pixels (the kit's composition
/// scaled to 1920x1200); widths measured from the text come from build.rs.
const CHIP_CENTRE_Y: f32 = 117.0;
const CHIP_RADIUS: f32 = 15.0;
const PILL_RIGHT: f32 = 96.0;
const PILL_CENTRE_Y: f32 = 124.5;
const SHADOW_OFFSET: f32 = 12.0;
const SHADOW_SIGMA: f32 = 30.0;

/// Signed distance to a horizontal capsule: centre, half straight length,
/// radius.
fn capsule(p: [f32; 2], centre: [f32; 2], half_length: f32, radius: f32) -> f32 {
    let q = [(p[0] - centre[0]).abs() - half_length, (p[1] - centre[1]).abs()];
    let outside = (q[0].max(0.0).powi(2) + q[1].max(0.0).powi(2)).sqrt();
    outside + q[0].max(q[1]).min(0.0) - radius
}

/// The clock's gap between digits, and either side of its colon.
const CLOCK_GAP: f32 = 6.0;
const CLOCK_COLON_GAP: f32 = 12.0;

// Indices into the baked text runs (build.rs's order).
const TEXT_COLON: usize = 10;
const TEXT_DASH: usize = 11;
const TEXT_COPY_1: usize = 12;
const TEXT_COPY_2: usize = 13;
const TEXT_ON_DEVICE: usize = 14;
const TEXT_LOCAL_WORKSPACE: usize = 15;
const TEXT_READY: usize = 16;
const TEXT_HINT: usize = 17;
const TEXT_AVATAR: usize = 18;
const TEXT_NAME: usize = 19;
const TEXT_SESSION: usize = 20;
const TEXT_ENTER: usize = 21;
const TEXT_UTC: usize = 22;
const TEXT_CLOCK_UNAVAILABLE: usize = 23;
const TEXT_LOCKED: usize = 24;
const TEXT_KERR: usize = 25;

#[derive(Clone, Copy)]
struct Mask {
    width: usize,
    height: usize,
    top: i32,
    offset: usize,
}
include!(concat!(env!("OUT_DIR"), "/masks.rs"));
const COVERAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/coverage.bin"));
impl Mask {
    fn integral(self, x: f64, y: f64) -> f64 {
        let x = x.clamp(0.0, self.width as f64);
        let y = y.clamp(0.0, self.height as f64);
        let ix = x as usize;
        let iy = y as usize;
        let read = |xx: usize, yy: usize| {
            let offset =
                self.offset + (yy.min(self.height) * (self.width + 1) + xx.min(self.width)) * 4;
            u32::from_le_bytes(COVERAGE[offset..offset + 4].try_into().unwrap()) as f64
        };
        let (fx, fy) = (x - ix as f64, y - iy as f64);
        let a = read(ix, iy) * (1.0 - fx) + read(ix + 1, iy) * fx;
        let b = read(ix, iy + 1) * (1.0 - fx) + read(ix + 1, iy + 1) * fx;
        a * (1.0 - fy) + b * fy
    }
    fn paint(self, plane: &mut [f32], tile: [u32; 2], at: [f32; 2], scale: f32, opacity: f32) {
        let left = at[0] - tile[0] as f32;
        let top = at[1] - tile[1] as f32;
        let x0 = left.floor().clamp(0.0, TILE_WIDTH as f32) as usize;
        let y0 = top.floor().clamp(0.0, TILE_HEIGHT as f32) as usize;
        let x1 = (left + self.width as f32 * scale)
            .ceil()
            .clamp(0.0, TILE_WIDTH as f32) as usize;
        let y1 = (top + self.height as f32 * scale)
            .ceil()
            .clamp(0.0, TILE_HEIGHT as f32) as usize;
        let inv = 1.0 / f64::from(scale);
        for y in y0..y1 {
            for x in x0..x1 {
                let sx = (x as f64 - f64::from(left)) * inv;
                let sy = (y as f64 - f64::from(top)) * inv;
                let sum = self.integral(sx + inv, sy + inv)
                    - self.integral(sx, sy + inv)
                    - self.integral(sx + inv, sy)
                    + self.integral(sx, sy);
                let alpha = (sum / (inv * inv * 65535.0)).clamp(0.0, 1.0) as f32 * opacity;
                plane[y * TILE_WIDTH + x] = plane[y * TILE_WIDTH + x].max(alpha);
            }
        }
    }
}

#[cfg(test)]
mod blend_tests {
    use super::*;
    use alloc::alloc::Global;

    /// The blended preparation as it was before its stencils were hoisted out
    /// of the pixel loop and its azimuth wrap lost `floor`.
    fn reference(viewport: Viewport, origin: [u32; 2]) -> [[f32; TILE_PIXELS]; 11] {
        let mut out = [[0.0; TILE_PIXELS]; 11];
        let k = viewport.lens_step();
        let centre = viewport.width as f32 / 2.0;
        for row in 0..TILE_HEIGHT {
            let ly = (origin[1] as f32 + row as f32 + 0.5) * k - 0.5;
            let iy = ly.floor();
            let fy = ly - iy;
            for column in 0..TILE_WIDTH {
                let lx = (origin[0] as f32 + column as f32 + 0.5 - centre) * k + LENS_WIDTH as f32 / 2.0 - 0.5;
                let ix = lx.floor();
                let fx = lx - ix;
                let corners = [
                    (lens_texel(ix as i32, iy as i32), (1.0 - fx) * (1.0 - fy)),
                    (lens_texel(ix as i32 + 1, iy as i32), fx * (1.0 - fy)),
                    (lens_texel(ix as i32, iy as i32 + 1), (1.0 - fx) * fy),
                    (lens_texel(ix as i32 + 1, iy as i32 + 1), fx * fy),
                ];
                let at = row * TILE_WIDTH + column;
                let sky = blend_sky(corners.map(|(t, w)| (w * ((t[3] >> 16) & 0xff) as f32 / 255.0, decode_sky(t[0]))));
                for c in 0..3 {
                    out[c][at] = sky[c];
                }
                for slot in 1..3 {
                    let blended = blend_crossing(corners.map(|(t, w)| {
                        let coverage = ((t[3] >> (8 * (slot - 1))) & 0xff) as f32 / 255.0;
                        (w * coverage, decode_crossing(t[slot]))
                    }));
                    for c in 0..4 {
                        out[3 + (slot - 1) * 4 + c][at] = blended[c];
                    }
                }
            }
        }
        out
    }

    #[test]
    fn the_blended_path_keeps_every_bit_of_the_per_pixel_reference() {
        let mut inputs = HoleInputs::new_in(Global).unwrap();
        // Larger than the lens map, wider and narrower than its 16:10, and
        // tiles at the corners (texels clamped past the map's edges).
        for [w, h] in [[2880, 1800], [2560, 1600], [3840, 2160], [1921, 1201], [2400, 1800]] {
            let viewport = Viewport::new(w, h).unwrap();
            let right = (w - 1) / TILE_WIDTH as u32 * TILE_WIDTH as u32;
            let bottom = (h - 1) / TILE_HEIGHT as u32 * TILE_HEIGHT as u32;
            for origin in [[0, 0], [right, 0], [0, bottom], [right, bottom], [w / 2 / 128 * 128, h / 2 / 32 * 32]] {
                inputs.prepare_scene(viewport, origin);
                let expected = reference(viewport, origin);
                let actual = [
                    &inputs.sky[0], &inputs.sky[1], &inputs.sky[2],
                    &inputs.near[0], &inputs.near[1], &inputs.near[2], &inputs.near[3],
                    &inputs.far[0], &inputs.far[1], &inputs.far[2], &inputs.far[3],
                ];
                for (plane, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
                        assert_eq!(a.to_bits(), e.to_bits(), "{w}x{h} tile {origin:?} plane {plane} pixel {i}");
                    }
                }
            }
        }
    }

    #[test]
    fn tiles_the_overlay_cannot_reach_have_no_coverage() {
        let mut inputs = ComposeInputs::new_in(Global).unwrap();
        let clock = Clock::from_unix_seconds(Some(76740));
        for [w, h] in [[1920, 1200], [2880, 1800], [1437, 897], [3840, 2160], [1200, 1920]] {
            let viewport = Viewport::new(w, h).unwrap();
            let (mut skipped, mut composed) = (0, 0);
            for y in (0..h).step_by(TILE_HEIGHT) {
                for x in (0..w).step_by(TILE_WIDTH) {
                    if viewport.overlays_tile([x, y]) {
                        composed += 1;
                        continue;
                    }
                    skipped += 1;
                    inputs.prepare_text(viewport, [x, y], clock);
                    inputs.prepare_shapes(viewport, [x, y]);
                    for plane in inputs.ink.iter().chain(&inputs.shape) {
                        assert!(plane.iter().all(|&v| v == 0.0), "{w}x{h} tile [{x}, {y}] has coverage");
                    }
                }
            }
            // Most of the panel is the hole alone.
            assert!(skipped > 2 * composed, "{w}x{h}: {skipped} skipped, {composed} composed");
        }
    }

    #[test]
    fn the_upsample_is_exact_at_pixel_centres_and_bilinear_between() {
        let pixels = [0x00_10_20_30, 0x00_30_40_50, 0x00_50_60_70, 0x00_70_80_90];
        let image = HoleImage { pixels: &pixels, width: 2, height: 2 };
        assert_eq!(image.sample(0.0, 0.0), image.pixel(0, 0));
        assert_eq!(image.sample(1.0, 1.0), image.pixel(1, 1));
        let mid = image.sample(0.5, 0.5);
        for c in 0..3 {
            let average = (image.pixel(0, 0)[c] + image.pixel(1, 0)[c] + image.pixel(0, 1)[c] + image.pixel(1, 1)[c]) / 4.0;
            assert!((mid[c] - average).abs() < 1e-6);
        }
        // Past the edges the image clamps.
        assert_eq!(image.sample(-3.0, -3.0), image.pixel(0, 0));
    }

    #[test]
    fn the_hoisted_upsample_is_the_sample_bit_for_bit() {
        // Biggertron's panel, and a 4K one whose hole frame is not a whole
        // ratio of it, over a hole frame of distinct pixels; interior and
        // clipped edge tiles.
        for (size, origins) in [([2880, 1800], [[256, 1184], [2816, 1792]]), ([3840, 2160], [[1408, 640], [3712, 2144]])] {
            let viewport = Viewport::new(size[0], size[1]).unwrap();
            let hole = viewport.hole_frame();
            let [w, h] = hole.size().map(|v| v as usize);
            let pixels: Vec<u32> = (0..w * h).map(|i| (i as u32).wrapping_mul(2_654_435_761) & 0xff_ffff).collect();
            let image = HoleImage { pixels: &pixels, width: w, height: h };
            let mut inputs = ComposeInputs::new_in(Global).unwrap();
            for origin in origins {
                inputs.prepare_hole(viewport, hole, origin, image);
                for row in 0..TILE_HEIGHT {
                    for column in 0..TILE_WIDTH {
                        let (x, y) = ((origin[0] as usize + column) as f32, (origin[1] as usize + row) as f32);
                        let [hx, hy] = viewport.hole_point(hole, x, y);
                        let expected = image.sample(hx, hy);
                        for c in 0..3 {
                            assert_eq!(inputs.hole[c][row * TILE_WIDTH + column].to_bits(), expected[c].to_bits());
                        }
                    }
                }
            }
        }
    }
}
