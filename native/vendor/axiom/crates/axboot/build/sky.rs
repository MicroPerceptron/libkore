//! The sky plates (build host only, std): an all-sky map turned into the
//! hole's frame and cut to what the atlas needs. `tools/sky-prep` cuts
//! them from the source map; the bake reads them back. Both include this
//! file, so the plates' geometry and orientation have one definition.
//!
//! A plate file is a 16-byte header — the magic `KSKY` and the orientation it
//! was cut for (`SKY_BEHIND_GALACTIC_DEG`, `SKY_ROLL_DEG` as f32) — then its
//! grid's texels as little-endian RGB9E5, row-major from the north.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use crate::kerr_params::*;

pub const MAGIC: [u8; 4] = *b"KSKY";
pub const HEADER: usize = 16;

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn normalize(a: [f64; 3]) -> [f64; 3] {
    let n = dot(a, a).sqrt();
    a.map(|c| c / n)
}

pub fn direction(lon: f64, lat: f64) -> [f64; 3] {
    let (sl, cl) = lat.sin_cos();
    let (sn, cs) = lon.sin_cos();
    [cl * cs, cl * sn, sl]
}

pub fn wrap_pi(a: f64) -> f64 {
    (a + PI).rem_euclid(TAU) - PI
}

/// The sky directly behind the hole as the camera sees it: straight through
/// the hole from the camera's position, in the hole's frame (z along the
/// spin, the camera at azimuth 0).
pub fn behind() -> [f64; 3] {
    let (s, c) = f64::from(CAMERA_INCLINATION_DEG).to_radians().sin_cos();
    [-s, 0.0, -c]
}

/// The azimuth of [`behind`]: plates and the fine tier unwrap around it.
pub const BEHIND_AZIMUTH: f64 = PI;

/// The rotation from the hole's frame to the galactic frame: the galactic
/// point `SKY_BEHIND_GALACTIC_DEG` lands on [`behind`], and the galactic
/// north pole leans `SKY_ROLL_DEG` from the hole's spin axis about that line.
pub struct Orientation {
    galactic: [[f64; 3]; 3],
    hole: [[f64; 3]; 3],
}

impl Orientation {
    pub fn new() -> Self {
        let [l, b] = SKY_BEHIND_GALACTIC_DEG.map(|v| f64::from(v).to_radians());
        // A right-handed basis: forward, side, and the pole's direction made
        // perpendicular to forward.
        let basis = |forward: [f64; 3], pole: [f64; 3]| {
            let along = dot(pole, forward);
            let up = normalize(core::array::from_fn(|i| pole[i] - along * forward[i]));
            [forward, cross(up, forward), up]
        };
        let galactic = basis(direction(l, b), [0.0, 0.0, 1.0]);
        let forward = behind();
        let [_, side, up] = basis(forward, [0.0, 0.0, 1.0]);
        let (s, c) = f64::from(SKY_ROLL_DEG).to_radians().sin_cos();
        let up = core::array::from_fn(|i| c * up[i] + s * side[i]);
        Self {
            galactic,
            hole: [forward, cross(up, forward), up],
        }
    }

    /// The galactic direction of a hole-frame direction.
    pub fn to_galactic(&self, h: [f64; 3]) -> [f64; 3] {
        let k = self.hole.map(|axis| dot(h, axis));
        core::array::from_fn(|i| k[0] * self.galactic[0][i] + k[1] * self.galactic[1][i] + k[2] * self.galactic[2][i])
    }
}

/// The header a plate cut for the current orientation carries.
pub fn header() -> [u8; HEADER] {
    let mut out = [0u8; HEADER];
    out[..4].copy_from_slice(&MAGIC);
    for (i, v) in [SKY_BEHIND_GALACTIC_DEG[0], SKY_BEHIND_GALACTIC_DEG[1], SKY_ROLL_DEG].iter().enumerate() {
        out[4 + 4 * i..8 + 4 * i].copy_from_slice(&v.to_le_bytes());
    }
    out
}

/// An equirectangular grid in the hole's frame: texel `(x, y)`'s centre sits
/// at azimuth `lon0 + x·step` (unwrapped around `centre`) and latitude
/// `lat0 − y·step`.
#[derive(Clone, Copy)]
pub struct Grid {
    pub lon0: f64,
    pub lat0: f64,
    pub step: f64,
    pub size: [usize; 2],
    pub centre: f64,
    /// Whether the grid closes in azimuth (a whole sphere).
    pub wraps: bool,
}

impl Grid {
    /// The near plate: full resolution over the azimuths and latitudes in
    /// `SKY_NEAR_*`.
    pub fn near() -> Self {
        let step = TAU / SKY_NEAR_COLUMNS as f64;
        let [a0, a1] = SKY_NEAR_AZIMUTH_DEG.map(|v| f64::from(v).to_radians());
        let [b0, b1] = SKY_NEAR_LATITUDE_DEG.map(|v| f64::from(v).to_radians());
        Self {
            lon0: BEHIND_AZIMUTH + a0,
            lat0: b1,
            step,
            size: [((a1 - a0) / step).round() as usize + 1, ((b1 - b0) / step).round() as usize + 1],
            centre: BEHIND_AZIMUTH,
            wraps: false,
        }
    }

    /// The far plate: the whole sphere, texel centres half a step in from
    /// the seam and the poles.
    pub fn far() -> Self {
        let step = TAU / SKY_FAR_COLUMNS as f64;
        Self {
            lon0: -PI + 0.5 * step,
            lat0: FRAC_PI_2 - 0.5 * step,
            step,
            size: [SKY_FAR_COLUMNS, SKY_FAR_COLUMNS / 2],
            centre: 0.0,
            wraps: true,
        }
    }

    pub fn texels(&self) -> usize {
        self.size[0] * self.size[1]
    }

    /// Azimuth and latitude at fractional texel `(x, y)`.
    pub fn at(&self, x: f64, y: f64) -> (f64, f64) {
        (self.lon0 + x * self.step, self.lat0 - y * self.step)
    }

    /// Fractional texel coordinates of `(lon, lat)`.
    pub fn coord(&self, lon: f64, lat: f64) -> [f64; 2] {
        let lon = self.centre + wrap_pi(lon - self.centre);
        [(lon - self.lon0) / self.step, (self.lat0 - lat) / self.step]
    }

    /// Whether the box `lon × lat` lies inside the grid with a texel to spare
    /// for the bilinear reach.
    pub fn holds(&self, lon: [f64; 2], lat: [f64; 2]) -> bool {
        if self.wraps {
            return true;
        }
        let [x0, y1] = self.coord(lon[0], lat[0]);
        let [x1, y0] = self.coord(lon[1], lat[1]);
        x0 >= 1.0 && y0 >= 1.0 && x1 <= self.size[0] as f64 - 2.0 && y1 <= self.size[1] as f64 - 2.0 && x1 >= x0
    }
}

/// Shared-exponent RGB (9-bit mantissas, 5-bit exponent, bias 15).
pub fn rgb9e5(c: [f64; 3]) -> u32 {
    const MANTISSA: i32 = 9;
    const BIAS: i32 = 15;
    let max_value = (511.0 / 512.0) * 2f64.powi(16);
    let c = c.map(|v| if v.is_finite() { v.clamp(0.0, max_value) } else { 0.0 });
    let max = c[0].max(c[1]).max(c[2]);
    if max <= 0.0 {
        return 0;
    }
    let mut exp = max.log2().floor().max(f64::from(-BIAS - 1)) as i32 + 1 + BIAS;
    let mut denom = 2f64.powi(exp - BIAS - MANTISSA);
    if (max / denom + 0.5).floor() >= 512.0 {
        exp += 1;
        denom *= 2.0;
    }
    let m = c.map(|v| ((v / denom + 0.5).floor() as u32).min(511));
    m[0] | (m[1] << 9) | (m[2] << 18) | ((exp.clamp(0, 31) as u32) << 27)
}

pub fn decode_rgb9e5(bits: u32) -> [f64; 3] {
    let scale = 2f64.powi((bits >> 27) as i32 - 15 - 9);
    [bits & 511, (bits >> 9) & 511, (bits >> 18) & 511].map(|m| f64::from(m) * scale)
}

/// A plate read back: its grid and linear texels.
pub struct Plate {
    pub grid: Grid,
    texels: Vec<[f32; 3]>,
}

impl Plate {
    /// Parse a plate file for `grid`, refusing one cut for another
    /// orientation or another grid.
    pub fn parse(bytes: &[u8], grid: Grid, name: &str) -> Self {
        assert!(
            bytes.len() >= HEADER && bytes[..HEADER] == header(),
            "{name} was cut for another sky orientation: run tools/sky-prep again"
        );
        assert_eq!(bytes.len(), HEADER + grid.texels() * 4, "{name} does not match its grid: run tools/sky-prep again");
        let texels = bytes[HEADER..]
            .chunks_exact(4)
            .map(|w| decode_rgb9e5(u32::from_le_bytes(w.try_into().unwrap())).map(|c| c as f32))
            .collect();
        Self { grid, texels }
    }

    fn texel(&self, x: i64, y: i64) -> [f32; 3] {
        let [w, h] = self.grid.size.map(|v| v as i64);
        let x = if self.grid.wraps { x.rem_euclid(w) } else { x.clamp(0, w - 1) };
        self.texels[(y.clamp(0, h - 1) * w + x) as usize]
    }

    /// Bilinear radiance at `(lon, lat)`.
    pub fn sample(&self, lon: f64, lat: f64) -> [f64; 3] {
        let [x, y] = self.grid.coord(lon, lat);
        let (ix, iy) = (x.floor(), y.floor());
        let (fx, fy) = (x - ix, y - iy);
        let (ix, iy) = (ix as i64, iy as i64);
        let mut out = [0.0; 3];
        for (dx, dy, w) in [(0, 0, (1.0 - fx) * (1.0 - fy)), (1, 0, fx * (1.0 - fy)), (0, 1, (1.0 - fx) * fy), (1, 1, fx * fy)] {
            let t = self.texel(ix + dx, iy + dy);
            for k in 0..3 {
                out[k] += w * f64::from(t[k]);
            }
        }
        out
    }

    /// Mean radiance over the box `lon × lat`: bilinear samples on a grid two
    /// to a plate texel, which is a box filter at the box's own size.
    pub fn box_mean(&self, lon: [f64; 2], lat: [f64; 2]) -> [f64; 3] {
        let n = |span: f64| ((2.0 * span / self.grid.step).ceil() as usize).clamp(2, 128);
        let (nx, ny) = (n(lon[1] - lon[0]), n(lat[1] - lat[0]));
        let mut out = [0.0; 3];
        for j in 0..ny {
            let la = lat[0] + (lat[1] - lat[0]) * (j as f64 + 0.5) / ny as f64;
            for i in 0..nx {
                let lo = lon[0] + (lon[1] - lon[0]) * (i as f64 + 0.5) / nx as f64;
                let s = self.sample(lo, la);
                for k in 0..3 {
                    out[k] += s[k];
                }
            }
        }
        out.map(|c| c / (nx * ny) as f64)
    }
}
