//! Build-time bake of the Kerr welcome scene (build host only, std).
//!
//! Every pixel traces one null geodesic backwards from the camera through
//! the Kerr metric (Boyer–Lindquist, M = 1, E = 1) with RK4 on the
//! Hamiltonian equations, as the lock-screen kit's reference shader does. The
//! metric is stationary and the camera sits at azimuth 0, so each pixel's
//! path is fixed: the trace runs here, once. The runtime animates what moves
//! along those paths — the disk's turbulence, co-rotating with the gas, and
//! the camera's swing about the spin axis, which turns sky and disk alike.
//!
//! Outputs:
//! - the lens map, per texel (`LENS_WIDTH × LENS_HEIGHT`, four little-endian
//!   u32 words), every quantity taken over five rays — the texel's centre and
//!   four quarter offsets — so sub-texel features (the photon ring's thin
//!   higher-order images) anti-alias instead of beading:
//!   - where the escaping rays meet the sky, as a sky-atlas coordinate (u16
//!     column and row, sixteenths of a texel);
//!   - the two disk images the rays see, nearer first, each the mean over
//!     the rays that see it, packed as `r` (11 bits, 0 = none), `φ`
//!     (11 bits) and the redshift factor `g` (10 bits);
//!   - coverage bytes: each image's (the fraction of the five rays that see
//!     it, nearer in the low byte) and the sky's (the fraction that escape,
//!     third byte);
//! - the sky atlas: the sky resampled into tiers prefiltered at different
//!   footprints — a fine patch where the camera sees the sky nearly
//!   unlensed, and whole spheres at coarser steps for the lensed images —
//!   each padded by the camera's swing, so a swinging lookup never leaves its
//!   tier;
//! - the turbulence table, band-limited to its own resolution.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use crate::kerr_params::*;
use crate::sky::{BEHIND_AZIMUTH, Grid, Plate, wrap_pi};

const MAX_STEPS: u32 = 2000;
const STEP_SCALE: f64 = 0.02;
const ESCAPE_RADIUS: f64 = 60.0;

#[derive(Clone, Copy)]
struct State {
    r: f64,
    th: f64,
    pr: f64,
    pth: f64,
    ph: f64,
}

fn rhs(s: State, l: f64, a: f64) -> State {
    let (sn, cs) = s.th.sin_cos();
    let sig = s.r * s.r + a * a * cs * cs;
    let del = s.r * s.r - 2.0 * s.r + a * a;
    let pq = (s.r * s.r + a * a) - a * l;
    let dpd = (4.0 * s.r * pq * del - pq * pq * (2.0 * s.r - 2.0)) / (del * del);
    State {
        r: del * s.pr / sig,
        th: s.pth / sig,
        pr: -0.5 / sig * ((2.0 * s.r - 2.0) * s.pr * s.pr - dpd),
        pth: (cs / sig) * (l / sn - a * sn) * (l / (sn * sn) + a),
        ph: (1.0 / sig) * ((l - a * sn * sn) / (sn * sn) + a * pq / del),
    }
}

fn madd(s: State, d: State, h: f64) -> State {
    State {
        r: s.r + h * d.r,
        th: s.th + h * d.th,
        pr: s.pr + h * d.pr,
        pth: s.pth + h * d.pth,
        ph: s.ph + h * d.ph,
    }
}

fn rk4(s: State, l: f64, a: f64, h: f64) -> State {
    let k1 = rhs(s, l, a);
    let k2 = rhs(madd(s, k1, 0.5 * h), l, a);
    let k3 = rhs(madd(s, k2, 0.5 * h), l, a);
    let k4 = rhs(madd(s, k3, h), l, a);
    let c = h / 6.0;
    State {
        r: s.r + c * (k1.r + 2.0 * k2.r + 2.0 * k3.r + k4.r),
        th: s.th + c * (k1.th + 2.0 * k2.th + 2.0 * k3.th + k4.th),
        pr: s.pr + c * (k1.pr + 2.0 * k2.pr + 2.0 * k3.pr + k4.pr),
        pth: s.pth + c * (k1.pth + 2.0 * k2.pth + 2.0 * k3.pth + k4.pth),
        ph: s.ph + c * (k1.ph + 2.0 * k2.ph + 2.0 * k3.ph + k4.ph),
    }
}

/// The prograde innermost stable circular orbit (Bardeen–Press–Teukolsky).
pub fn isco(a: f64) -> f64 {
    let z1 = 1.0 + (1.0 - a * a).cbrt() * ((1.0 + a).cbrt() + (1.0 - a).cbrt());
    let z2 = (3.0 * a * a + z1 * z1).sqrt();
    3.0 + z2 - (a + 1e-12).signum() * ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)).sqrt()
}

/// One disk crossing: radius, azimuth (unwrapped) and the emitter-to-observer
/// redshift factor `g` of a prograde Keplerian emitter.
#[derive(Clone, Copy, Default)]
struct Crossing {
    r: f64,
    ph: f64,
    g: f64,
}

struct Traced {
    crossings: [Option<Crossing>; 2],
    /// The asymptotic direction a ray that escaped was travelling, as a unit
    /// vector in the hole's frame; `None` when it fell in or ran out of steps.
    escape: Option<[f64; 3]>,
}

/// Trace the ray through the camera's image plane at `(u, v)` ∈ [0,1]², `v`
/// down, for an image of aspect `aspect`.
fn trace(u: f64, v: f64, aspect: f64) -> Traced {
    let a = f64::from(SPIN);
    let rh = 1.0 + (1.0 - a * a).max(0.0).sqrt();
    let r0 = f64::from(CAMERA_RADIUS);
    let th0 = f64::from(CAMERA_INCLINATION_DEG).to_radians();
    let fov = f64::from(FIELD_OF_VIEW_DEG).to_radians();
    let r_in = isco(a);
    let r_out = f64::from(DISK_OUTER);

    // Pixel → direction in the observer's local frame: x +φ, y up (−θ),
    // z toward the hole (−r); then yaw and pitch inside that frame.
    let tanf = (0.5 * fov).tan();
    let mut n = [(u - 0.5) * 2.0 * tanf * aspect, (0.5 - v) * 2.0 * tanf, 1.0];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    n = n.map(|c| c / len);
    let (sy, cy) = f64::from(LOOK_YAW_DEG).to_radians().sin_cos();
    n = [cy * n[0] + sy * n[2], n[1], -sy * n[0] + cy * n[2]];
    let (sp, cp) = f64::from(LOOK_PITCH_DEG).to_radians().sin_cos();
    n = [n[0], cp * n[1] + sp * n[2], -sp * n[1] + cp * n[2]];

    // The camera's ZAMO tetrad → covariant BL momentum, normalised to E = 1,
    // so the field of view is the observer's own.
    let (s0, c0) = th0.sin_cos();
    let sig0 = r0 * r0 + a * a * c0 * c0;
    let del0 = r0 * r0 - 2.0 * r0 + a * a;
    let a0 = (r0 * r0 + a * a) * (r0 * r0 + a * a) - a * a * del0 * s0 * s0;
    let alpha0 = (sig0 * del0 / a0).sqrt();
    let omega0 = 2.0 * a * r0 / a0;
    let (p_r_hat, p_th_hat, p_ph_hat) = (-n[2], -n[1], n[0]);
    let pt_up = 1.0 / alpha0;
    let pr_up = p_r_hat * (del0 / sig0).sqrt();
    let pth_up = p_th_hat / sig0.sqrt();
    let pph_up = omega0 / alpha0 + p_ph_hat * (sig0 / a0).sqrt() / s0;
    let g_tt = -(1.0 - 2.0 * r0 / sig0);
    let g_tp = -2.0 * a * r0 * s0 * s0 / sig0;
    let g_pp = a0 * s0 * s0 / sig0;
    let e = -(g_tt * pt_up + g_tp * pph_up);
    let l = (g_tp * pt_up + g_pp * pph_up) / e;
    let mut s = State {
        r: r0,
        th: th0,
        pr: (sig0 / del0) * pr_up / e,
        pth: sig0 * pth_up / e,
        ph: 0.0,
    };

    let mut crossings = [None; 2];
    let mut found = 0;
    for _ in 0..MAX_STEPS {
        // The kit's adaptive step, ∝ r and shrunk near the horizon, bounded
        // further by how far it turns the ray in θ and φ. Near the polar axis
        // a ray with small L whips round in φ over an arc of width ~|L|/p;
        // bounding each step's turn (θ by a fifth of sin θ, φ by 0.05 rad)
        // resolves that whip in a few dozen steps for any L. A fixed shrink
        // instead either under-resolved it — a wrong sky down the L = 0
        // column — or ran out of steps and drew the ray as captured. A ray
        // with L = 0 exactly passes through the pole: θ runs on past 0, which
        // is the same point as (−θ, φ + π), and everything below reads it
        // that way.
        let rate = rhs(s, l, a);
        let turn_th = (0.2 * s.th.sin().abs()).max(1e-5) / rate.th.abs().max(1e-12);
        let turn_ph = 0.05 / rate.ph.abs().max(1e-12);
        let h = (STEP_SCALE * s.r).min(0.35 * (s.r - rh)).clamp(0.0015, 0.6).min(turn_th).min(turn_ph).max(1e-7);
        let last = s;
        s = rk4(s, l, a, h);
        // Validate the step before reading anything from it. A legitimate
        // step moves r by about STEP_SCALE of itself; the last step into the
        // horizon instead blows up (Δ → 0 sits in the denominators) and can
        // throw r anywhere — even past ESCAPE_RADIUS with a phantom disk
        // crossing on the way, which is what drew dotted rings inside the
        // shadow. A ray that does that was falling in.
        let blown = !s.r.is_finite()
            || !s.th.is_finite()
            || !s.pr.is_finite()
            || (s.r - last.r).abs() > 0.5 * last.r;
        if blown || s.r < rh * 1.02 {
            return Traced { crossings, escape: None };
        }
        // The equatorial plane is cos θ = 0: θ = π/2, and past a pole
        // −π/2 or 3π/2 as well.
        let (f0, f1) = (last.th.cos(), s.th.cos());
        if f0 * f1 < 0.0 {
            let w = f0 / (f0 - f1);
            let rc = last.r + (s.r - last.r) * w;
            if rc > r_in && rc < r_out && found < 2 {
                let om = 1.0 / (rc.powf(1.5) + a);
                let ut = 1.0 / (1.0 - 3.0 / rc + 2.0 * a / rc.powf(1.5)).max(1e-4).sqrt();
                crossings[found] = Some(Crossing {
                    r: rc,
                    ph: last.ph + (s.ph - last.ph) * w,
                    g: 1.0 / (ut * (1.0 - om * l)),
                });
                found += 1;
            }
        }
        if s.r > ESCAPE_RADIUS && s.pr > 0.0 {
            // The asymptotic direction from the BL velocity at exit.
            let d = rhs(s, l, a);
            let (sn, cs) = s.th.sin_cos();
            let (sph, cph) = s.ph.sin_cos();
            let er = [sn * cph, sn * sph, cs];
            let et = [cs * cph, cs * sph, -sn];
            let ep = [-sph, cph, 0.0];
            let v: [f64; 3] = core::array::from_fn(|i| d.r * er[i] + s.r * d.th * et[i] + s.r * sn * d.ph * ep[i]);
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            return Traced {
                crossings,
                escape: Some(v.map(|c| c / n)),
            };
        }
    }
    // Ran out of steps circling the photon ring: dark, as captured.
    Traced { crossings, escape: None }
}

// ───────────────────────────── the sky ─────────────────────────────

/// The sky the atlas resamples: the near plate at full resolution where it
/// holds the box and the box is no wider than one of its texels and a half,
/// else the far sphere.
struct SkyPlates {
    near: Plate,
    far: Plate,
}

impl SkyPlates {
    fn box_mean(&self, lon: [f64; 2], lat: [f64; 2]) -> [f64; 3] {
        if lon[1] - lon[0] <= 1.5 * self.near.grid.step && self.near.grid.holds(lon, lat) {
            self.near.box_mean(lon, lat)
        } else {
            self.far.box_mean(lon, lat)
        }
    }
}

// ───────────────────────────── the atlas ─────────────────────────────

/// One tier's rectangle in the atlas: an equirectangular grid with equal
/// steps in azimuth and latitude. Texel centres sit on integer coordinates,
/// as the volume sampler reads them; azimuth grows with the column, latitude
/// falls with the row.
#[derive(Clone, Copy)]
pub struct Region {
    pub origin: [usize; 2],
    pub size: [usize; 2],
    /// Azimuth and latitude of the region's first texel centre.
    pub lon0: f64,
    pub lat0: f64,
    /// Radians per texel, both axes.
    pub step: f64,
    /// Lookups unwrap azimuth to within half a turn of this.
    centre: f64,
}

impl Region {
    /// A whole sphere at `columns` texels per turn: padded by the swing (and
    /// two texels for the bilinear reach) in azimuth, and by a row past each
    /// pole.
    fn sphere(columns: usize, origin: [usize; 2]) -> Self {
        let step = TAU / columns as f64;
        let pad = sway() + 2.0 * step;
        Self {
            origin,
            size: [((TAU + 2.0 * pad) / step).ceil() as usize + 1, (PI / step).ceil() as usize + 3],
            lon0: -PI - pad,
            lat0: FRAC_PI_2 + step,
            step,
            centre: 0.0,
        }
    }

    /// The fine patch over azimuths `lon` (relative to `centre`) and
    /// latitudes `lat`, padded like a sphere.
    fn patch(centre: f64, lon: [f64; 2], lat: [f64; 2]) -> Self {
        let step = fine_step();
        let (pad_lon, pad_lat) = (sway() + 2.0 * step, 2.0 * step);
        Self {
            origin: [0, 0],
            size: [
                ((lon[1] - lon[0] + 2.0 * pad_lon) / step).ceil() as usize + 1,
                ((lat[1] - lat[0] + 2.0 * pad_lat) / step).ceil() as usize + 1,
            ],
            lon0: centre + lon[0] - pad_lon,
            lat0: lat[1] + pad_lat,
            step,
            centre,
        }
    }

    /// The atlas coordinate of the direction `(lon, lat)`.
    fn coord(&self, lon: f64, lat: f64) -> [f64; 2] {
        let lon = self.centre + wrap_pi(lon - self.centre);
        [
            self.origin[0] as f64 + (lon - self.lon0) / self.step,
            self.origin[1] as f64 + (self.lat0 - lat) / self.step,
        ]
    }

    fn contains(&self, x: usize, y: usize) -> bool {
        (self.origin[0]..self.origin[0] + self.size[0]).contains(&x) && (self.origin[1]..self.origin[1] + self.size[1]).contains(&y)
    }

    /// The sky box atlas texel `(x, y)` of this region covers. A row past a
    /// pole repeats the polar row's box.
    fn texel_box(&self, x: usize, y: usize) -> ([f64; 2], [f64; 2]) {
        let lon = self.lon0 + (x - self.origin[0]) as f64 * self.step;
        let lat = self.lat0 - (y - self.origin[1]) as f64 * self.step;
        let h = 0.5 * self.step;
        let (mut l0, mut l1) = ((lat - h).max(-FRAC_PI_2), (lat + h).min(FRAC_PI_2));
        if l1 - l0 < 0.25 * self.step {
            (l0, l1) = if lat > 0.0 {
                (FRAC_PI_2 - self.step, FRAC_PI_2)
            } else {
                (-FRAC_PI_2, -FRAC_PI_2 + self.step)
            };
        }
        ([lon - h, lon + h], [l0, l1])
    }
}

/// The camera's swing either way, in radians.
fn sway() -> f64 {
    f64::from(ORBIT_SWAY_DEG).to_radians()
}

/// The fine patch's step: the near plate's.
fn fine_step() -> f64 {
    TAU / SKY_NEAR_COLUMNS as f64
}

/// Atlas coordinates are packed as sixteenths of a texel in a u16.
const COORD_SCALE: f64 = 16.0;
const ATLAS_MAX: usize = 4096;

/// The tiers and where they sit, finest first: the fine patch top left, the
/// two small spheres beside it, the medium sphere across the bottom.
pub struct Atlas {
    pub tiers: [Region; 4],
    pub size: [usize; 2],
}

impl Atlas {
    fn new(fine: Region) -> Self {
        let coarse = Region::sphere(SKY_TIER_COLUMNS[1], [fine.size[0], 0]);
        let ultra = Region::sphere(SKY_TIER_COLUMNS[2], [fine.size[0] + coarse.size[0], 0]);
        assert!(coarse.size[1] <= fine.size[1] && ultra.size[1] <= fine.size[1], "the small spheres sit beside the fine patch");
        let medium = Region::sphere(SKY_TIER_COLUMNS[0], [0, fine.size[1]]);
        let width = medium.size[0].max(ultra.origin[0] + ultra.size[0]);
        let height = fine.size[1] + medium.size[1];
        assert!(width < ATLAS_MAX && height < ATLAS_MAX, "the sky atlas must fit u16 sixteenth-texel coordinates");
        Self {
            tiers: [fine, medium, coarse, ultra],
            size: [width, height],
        }
    }

    /// The atlas as the shader's RGB volume: UF16 E5M11, x fastest.
    fn render(&self, sky: &SkyPlates) -> Vec<u8> {
        let [w, h] = self.size;
        let mut out = vec![0u8; w * h * 6];
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let rows_per = h.div_ceil(threads);
        std::thread::scope(|scope| {
            for (chunk, rows) in out.chunks_mut(rows_per * w * 6).enumerate() {
                scope.spawn(move || {
                    for (index, texel) in rows.chunks_exact_mut(6).enumerate() {
                        let (x, y) = (index % w, chunk * rows_per + index / w);
                        let Some(region) = self.tiers.iter().find(|r| r.contains(x, y)) else {
                            continue;
                        };
                        let (lon, lat) = region.texel_box(x, y);
                        let c = sky.box_mean(lon, lat).map(|v| v * f64::from(SKY_GAIN));
                        for k in 0..3 {
                            let bits = axnn_cpu::uf::f32_to_uf16e5m11_rne(c[k] as f32);
                            texel[k * 2..k * 2 + 2].copy_from_slice(&bits.to_le_bytes());
                        }
                    }
                });
            }
        });
        out
    }
}

// ─────────────────────────── packing ───────────────────────────

/// The constants the runtime and the shader read back from the bake.
pub struct Baked {
    pub lens: Vec<u8>,
    pub turbulence: Vec<u8>,
    pub sky: Vec<u8>,
    pub disk_inner: f64,
    pub g_min: f64,
    pub g_max: f64,
    pub atlas: Atlas,
}

pub const R_CODES: f64 = 2046.0;
pub const PHI_CODES: f64 = 2048.0;
pub const G_CODES: f64 = 1023.0;

fn pack(c: Option<Crossing>, r_in: f64, g_min: f64, g_max: f64) -> u32 {
    let Some(c) = c else { return 0 };
    let r_out = f64::from(DISK_OUTER);
    let r = (1.0 + ((c.r - r_in) / (r_out - r_in) * R_CODES).round()).clamp(1.0, 2047.0) as u32;
    let phi = ((c.ph.rem_euclid(TAU) / TAU * PHI_CODES).round() as u32) & 2047;
    let g = (((c.g - g_min) / (g_max - g_min)) * G_CODES).round().clamp(0.0, G_CODES) as u32;
    r | (phi << 11) | (g << 22)
}

// ─────────────────────────── turbulence ───────────────────────────
//
// The kit's disk texture: value noise periodic in its second axis so the
// pattern closes in φ, summed over five octaves, at two scales. The runtime
// rotates it by `Ω(r)·t` per radius, so it co-rotates with the gas.
//
// A table holds an octave only while it samples each lattice cell at least
// twice; below once, the octave folds into false stripes. Each octave fades
// across that range toward its mean, so the table carries exactly what it
// can show.

fn fract(x: f64) -> f64 {
    x - x.floor()
}

fn hash21(p: [f64; 2]) -> f64 {
    let mut q = [fract(p[0] * 123.34), fract(p[1] * 456.21)];
    let d = q[0] * (q[0] + 45.32) + q[1] * (q[1] + 45.32);
    q = [q[0] + d, q[1] + d];
    fract(q[0] * q[1])
}

fn wrap(v: f64, period: f64) -> f64 {
    v - period * (v / period).floor()
}

fn vnoise(p: [f64; 2], period: f64) -> f64 {
    let i = [p[0].floor(), p[1].floor()];
    let f = [p[0] - i[0], p[1] - i[1]];
    let u = f.map(|t| t * t * (3.0 - 2.0 * t));
    let (y0, y1) = (wrap(i[1], period), wrap(i[1] + 1.0, period));
    let mix = |a: f64, b: f64, t: f64| a + (b - a) * t;
    mix(
        mix(hash21([i[0], y0]), hash21([i[0] + 1.0, y0]), u[0]),
        mix(hash21([i[0], y1]), hash21([i[0] + 1.0, y1]), u[0]),
        u[1],
    )
}

/// Five octaves of `vnoise`, each faded by how well a table whose samples sit
/// `spacing` apart (in `p`, the larger component per table axis) holds it.
fn fbm2(p: [f64; 2], period: f64, spacing: f64) -> f64 {
    let (mut v, mut amp, mut q, mut per, mut scale) = (0.0, 0.5, p, period, 1.0);
    for _ in 0..5 {
        let held = (1.0 / (spacing * scale) - 1.0).clamp(0.0, 1.0);
        v += amp * (held * vnoise(q, per) + (1.0 - held) * 0.5);
        q = [q[0] * 2.0 + 17.1, q[1] * 2.0];
        per *= 2.0;
        scale *= 2.0;
        amp *= 0.5;
    }
    v
}

fn turbulence(log_r_min: f64, log_r_max: f64) -> Vec<u8> {
    let columns = TURBULENCE_COLUMNS + 1;
    let row_step = (log_r_max - log_r_min) / (TURBULENCE_ROWS - 1) as f64;
    let column_step = 1.0 / TURBULENCE_COLUMNS as f64;
    // Each scale's sample spacing in its own noise coordinates: the larger of
    // a row step's and a column step's movement.
    let spacing1 = (6.0 * row_step).max(12.0 * column_step);
    let spacing2 = (14.0 * row_step).max(36.0 * column_step);
    let mut out = Vec::with_capacity(columns * TURBULENCE_ROWS * 6);
    for row in 0..TURBULENCE_ROWS {
        let lr = log_r_min + row_step * row as f64;
        for column in 0..columns {
            // The last column is the first again: u = 1 is u = 0.
            let u = (column % TURBULENCE_COLUMNS) as f64 * column_step;
            let n1 = fbm2([lr * 6.0, u * 12.0], 12.0, spacing1);
            let n2 = fbm2([lr * 14.0 + 3.0, u * 36.0 - lr * 2.0], 36.0, spacing2);
            for value in [n1, n2, 0.0] {
                let bits = axnn_cpu::uf::f32_to_uf16e5m11_rne(value as f32);
                out.extend_from_slice(&bits.to_le_bytes());
            }
        }
    }
    out
}

// ───────────────────────────── the bake ─────────────────────────────

/// Where a texel's escaping rays meet the sky: the reference ray's direction
/// (the centre's, else the first to escape), how much sky one texel spans
/// there, and the fraction of rays that escape.
#[derive(Clone, Copy)]
struct Escape {
    lon: f64,
    lat: f64,
    footprint: f64,
    cover: f64,
}

struct Texel {
    escape: Option<Escape>,
    crossings: [Option<Crossing>; 2],
    coverage: [f64; 2],
}

/// Rays per texel: the centre and four quarter offsets.
const RAYS: f64 = 5.0;
/// A quarter-offset ray sits this many texels from the centre.
const CORNER: f64 = 0.353_553_390_593_273_8;

/// A texel's two disk images over its rays. Crossings group into images by
/// where they meet the disk (radius within 15%, azimuth within 0.35 rad of
/// the image's first), not by their order along each ray: where a ray's
/// first crossing slides past the disk's outer edge, its neighbours' first
/// crossing is a different image, and grouping by order docked the image
/// they all see — a dashed dark contour across the disk. The two images
/// most rays see are kept, nearer first (the one the rays meet earlier),
/// each the mean of its rays with coverage the fraction that see it.
fn merge(rays: &[Traced; 5]) -> ([Option<Crossing>; 2], [f64; 2]) {
    struct Image {
        reference: Crossing,
        rays: u8,
        order: f64,
        r: f64,
        dph: f64,
        g: f64,
    }
    let mut images: Vec<Image> = Vec::new();
    for (ray, t) in rays.iter().enumerate() {
        for (order, c) in t.crossings.iter().enumerate() {
            let Some(c) = *c else { continue };
            let same = |i: &Image| {
                let d = (c.ph - i.reference.ph + PI).rem_euclid(TAU) - PI;
                (c.r - i.reference.r).abs() <= 0.15 * i.reference.r && d.abs() <= 0.35 && i.rays & (1 << ray) == 0
            };
            let image = match images.iter().position(same) {
                Some(i) => &mut images[i],
                None => {
                    images.push(Image { reference: c, rays: 0, order: 0.0, r: 0.0, dph: 0.0, g: 0.0 });
                    images.last_mut().unwrap()
                }
            };
            image.rays |= 1 << ray;
            image.order += order as f64;
            image.r += c.r;
            image.dph += (c.ph - image.reference.ph + PI).rem_euclid(TAU) - PI;
            image.g += c.g;
        }
    }
    images.sort_by_key(|i| core::cmp::Reverse(i.rays.count_ones()));
    images.truncate(2);
    let n = |i: &Image| f64::from(i.rays.count_ones());
    images.sort_by(|a, b| (a.order / n(a)).total_cmp(&(b.order / n(b))));
    let mut out = ([None; 2], [0.0; 2]);
    for (slot, i) in images.iter().enumerate() {
        let count = n(i);
        out.0[slot] = Some(Crossing {
            r: i.r / count,
            ph: i.reference.ph + i.dph / count,
            g: i.g / count,
        });
        out.1[slot] = count / RAYS;
    }
    out
}

/// The texel's escape: its reference direction and the widest angle from it
/// to another escaping ray, scaled from a quarter offset to a whole texel. A
/// texel with one escaping ray has no measurable footprint: it sits on the
/// shadow's edge, where the sky is compressed past any tier.
fn escape(rays: &[Traced; 5]) -> Option<Escape> {
    let mut escaping = rays.iter().filter_map(|t| t.escape);
    let reference = escaping.next()?;
    let (mut widest, mut count) = (0.0f64, 1.0);
    for d in escaping {
        let dot = reference[0] * d[0] + reference[1] * d[1] + reference[2] * d[2];
        widest = widest.max(dot.clamp(-1.0, 1.0).acos());
        count += 1.0;
    }
    Some(Escape {
        lon: reference[1].atan2(reference[0]),
        lat: reference[2].clamp(-1.0, 1.0).asin(),
        footprint: if count > 1.0 { widest / CORNER } else { f64::INFINITY },
        cover: count / RAYS,
    })
}

/// The tier whose step is nearest a footprint, on a log scale: the
/// boundaries sit at the geometric means of neighbouring steps.
fn tier(footprint: f64, steps: [f64; 4]) -> usize {
    (1..4).take_while(|&t| footprint >= (steps[t - 1] * steps[t]).sqrt()).last().unwrap_or(0)
}

/// The bounds of the fine tier's directions, trimmed of the outermost
/// `TRIM` on each side: a stray magnified texel far from the rest must not
/// widen the patch for everyone.
const TRIM: f64 = 0.0005;

fn trimmed(mut values: Vec<f64>) -> [f64; 2] {
    values.sort_by(f64::total_cmp);
    let at = |q: f64| values[((values.len() - 1) as f64 * q).round() as usize];
    [at(TRIM), at(1.0 - TRIM)]
}

/// Trace the scene and resample the sky plates (`near`, `far`: the plate
/// files' bytes) into the atlas.
pub fn bake(near: &[u8], far: &[u8]) -> Baked {
    let (w, h) = (LENS_WIDTH, LENS_HEIGHT);
    let aspect = w as f64 / h as f64;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let rows_per = h.div_ceil(threads);
    let mut texels: Vec<Option<Texel>> = (0..w * h).map(|_| None).collect();
    std::thread::scope(|scope| {
        for (chunk, rows) in texels.chunks_mut(rows_per * w).enumerate() {
            scope.spawn(move || {
                for (index, slot) in rows.iter_mut().enumerate() {
                    let (x, y) = (index % w, chunk * rows_per + index / w);
                    let at = |dx: f64, dy: f64| ((x as f64 + dx) / w as f64, (y as f64 + dy) / h as f64);
                    let rays = [(0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
                        .map(|(dx, dy)| {
                            let (u, v) = at(dx, dy);
                            trace(u, v, aspect)
                        });
                    let (crossings, coverage) = merge(&rays);
                    *slot = Some(Texel {
                        escape: escape(&rays),
                        crossings,
                        coverage,
                    });
                }
            });
        }
    });
    let texels: Vec<Texel> = texels.into_iter().map(|t| t.expect("every texel baked")).collect();

    // Tiers: the fine patch takes the texels whose footprint is near one
    // lens texel, and covers their directions — relative to the anti-camera
    // azimuth, where the view looks — padded by the swing.
    let steps = [
        fine_step(),
        TAU / SKY_TIER_COLUMNS[0] as f64,
        TAU / SKY_TIER_COLUMNS[1] as f64,
        TAU / SKY_TIER_COLUMNS[2] as f64,
    ];
    let mut tiers: Vec<usize> = texels.iter().map(|t| t.escape.map_or(0, |e| tier(e.footprint, steps))).collect();
    let fine: Vec<Escape> = texels.iter().zip(&tiers).filter(|&(_, &t)| t == 0).filter_map(|(t, _)| t.escape).collect();
    let lon = trimmed(fine.iter().map(|e| wrap_pi(e.lon - BEHIND_AZIMUTH)).collect());
    let lat = trimmed(fine.iter().map(|e| e.lat).collect());
    for (t, tier) in texels.iter().zip(tiers.iter_mut()) {
        if let (Some(e), 0) = (t.escape, *tier) {
            let rel = wrap_pi(e.lon - BEHIND_AZIMUTH);
            if rel < lon[0] || rel > lon[1] || e.lat < lat[0] || e.lat > lat[1] {
                *tier = 1;
            }
        }
    }
    let patch = Region::patch(BEHIND_AZIMUTH, lon, lat);
    let sky = SkyPlates {
        near: Plate::parse(near, Grid::near(), "assets/sky/near.bin"),
        far: Plate::parse(far, Grid::far(), "assets/sky/far.bin"),
    };
    // The fine patch, swing and all, must lie on the near plate.
    let half = 0.5 * patch.step;
    let edge = [patch.lon0 - half, patch.lon0 + (patch.size[0] as f64 - 0.5) * patch.step];
    let rows = [patch.lat0 - (patch.size[1] as f64 - 0.5) * patch.step, patch.lat0 + half];
    assert!(
        sky.near.grid.holds(edge, rows),
        "the fine sky patch needs azimuths {:.1}°..{:.1}° and latitudes {:.1}°..{:.1}°: widen SKY_NEAR_* and cut the plates again",
        (edge[0] - BEHIND_AZIMUTH).to_degrees(),
        (edge[1] - BEHIND_AZIMUTH).to_degrees(),
        rows[0].to_degrees(),
        rows[1].to_degrees(),
    );
    let atlas = Atlas::new(patch);
    // A texel no ray escapes from still names an in-bounds point (the fine
    // patch's centre) so a swinging lookup stays inside the atlas; its sky
    // coverage of zero hides what it reads.
    let idle = [
        (patch.size[0] as f64 * 0.5 * COORD_SCALE) as u32,
        (patch.size[1] as f64 * 0.5 * COORD_SCALE) as u32,
    ];

    let r_in = isco(f64::from(SPIN));
    let (mut g_min, mut g_max) = (f64::MAX, f64::MIN);
    for c in texels.iter().flat_map(|t| t.crossings.iter().flatten()) {
        g_min = g_min.min(c.g);
        g_max = g_max.max(c.g);
    }
    assert!(g_min < g_max, "the disk must be visible from the camera");
    let mut lens = Vec::with_capacity(w * h * 16);
    for (t, &tier) in texels.iter().zip(&tiers) {
        let (coord, sky_cover) = match t.escape {
            Some(e) => {
                let [x, y] = atlas.tiers[tier].coord(e.lon, e.lat);
                ([(x * COORD_SCALE).round() as u32, (y * COORD_SCALE).round() as u32], e.cover)
            }
            None => (idle, 0.0),
        };
        let coverage = [t.coverage[0], t.coverage[1], sky_cover].map(|c| (c * 255.0).round() as u32);
        for word in [
            coord[0] | (coord[1] << 16),
            pack(t.crossings[0], r_in, g_min, g_max),
            pack(t.crossings[1], r_in, g_min, g_max),
            coverage[0] | (coverage[1] << 8) | (coverage[2] << 16),
        ] {
            lens.extend_from_slice(&word.to_le_bytes());
        }
    }
    Baked {
        lens,
        turbulence: turbulence(r_in.ln(), f64::from(DISK_OUTER).ln()),
        sky: atlas.render(&sky),
        disk_inner: r_in,
        g_min,
        g_max,
        atlas,
    }
}
