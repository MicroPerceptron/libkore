//! Multi-threaded voxel scene — the `voxel` demo scaled up 4×.
//!
//! Run with:
//!   cargo +nightly run --release -p axrd --example voxel_mt --features window
//!
//! Same kernel pipeline as the single-threaded `voxel` example, at a
//! **256³** grid — voxels a quarter the size, so the waving terrain and
//! the three merging metaballs read as smooth surfaces rather than
//! cubes. Everything dynamic is rebuilt every frame and threaded over
//! `std::thread::scope`:
//!
//! * world rebuild (terrain columns + metaball stamping) by z-slab —
//!   slabs own disjoint `u64` words of the bit grid,
//! * brick repack via [`voxel::brick_pack_range`] by brick-z slab,
//! * rendering by scanline band ([`voxel::brick_trace`] primary rays +
//!   [`voxel::march_density_e4m3`] B8 ground mist per pixel).
//!
//! Shading cheats the cubes away: normals come from the analytic
//! gradient of the *continuous* scene fields (terrain heightfield,
//! metaball potential), so lighting is smooth across the voxelized
//! surface.
//!
//! Controls:
//!   Escape — quit
//!
//! Headless hooks:
//!   AXRD_DEMO_FRAMES=N      — exit after N frames, print timing stats
//!   AXRD_DEMO_SNAPSHOT=path — write the final frame as a binary PPM

#![feature(f16)]

use std::time::Instant;

use axnn_cpu::voxel::{self, BRICK};

#[path = "common/mod.rs"]
mod common;
use common::{Fp8Quantizer, cross, dot, lerp3, norm, pack};
use minifb::{Key, Window, WindowOptions};

const W: usize = 640;
const H: usize = 360;

/// World grid edge — 4× the `voxel` example's resolution.
const N: u32 = 256;
const T_FAR: f32 = 900.0;
/// Mist march range cap (world units) — τ saturates far earlier, so
/// this is visually free and bounds the per-pixel march cost.
const MIST_FAR: f32 = 420.0;

// ── Continuous scene fields (build + smooth normals share these) ────

fn terrain_height(x: f32, z: f32, t: f32) -> f32 {
    40.0 + 18.0 * (0.0475 * x + 1.1 * t).sin()
        + 14.0 * (0.0575 * z - 0.8 * t).sin()
        + 10.0 * (0.0275 * (x + z) + 0.5 * t).sin()
}

fn ball_centers(t: f32) -> [[f32; 3]; 3] {
    let c = N as f32 * 0.5;
    [
        [
            c + 52.0 * (0.7 * t).cos(),
            126.0 + 18.0 * (1.3 * t).sin(),
            c + 52.0 * (0.7 * t).sin(),
        ],
        [
            c + 40.0 * (0.9 * t + 2.1).cos(),
            132.0 + 14.0 * (1.1 * t + 1.0).sin(),
            c + 40.0 * (0.9 * t + 2.1).sin(),
        ],
        [
            c + 24.0 * (0.5 * t + 4.2).cos(),
            120.0 + 22.0 * (0.8 * t + 2.5).sin(),
            c + 24.0 * (0.5 * t + 4.2).sin(),
        ],
    ]
}

const BALL_R: f32 = 20.0;

/// Metaball potential − iso: positive inside the merged surface.
fn ball_field(p: [f32; 3], centers: &[[f32; 3]; 3]) -> f32 {
    let mut f = 0.0;
    for c in centers {
        let (dx, dy, dz) = (p[0] - c[0], p[1] - c[1], p[2] - c[2]);
        f += BALL_R * BALL_R / (dx * dx + dy * dy + dz * dz).max(1e-6);
    }
    f - 1.0
}

/// Terrain occupancy field: positive below the height surface.
fn terrain_field(p: [f32; 3], t: f32) -> f32 {
    terrain_height(p[0], p[2], t) - p[1]
}

/// Smooth outward normal at `p` from whichever field owns the surface.
fn scene_normal(p: [f32; 3], t: f32, centers: &[[f32; 3]; 3]) -> ([f32; 3], bool) {
    let is_ball = ball_field(p, centers) > terrain_field(p, t) * 0.04;
    const E: f32 = 0.75;
    let mut g = [0.0f32; 3];
    for a in 0..3 {
        let (mut hi, mut lo) = (p, p);
        hi[a] += E;
        lo[a] -= E;
        g[a] = if is_ball {
            ball_field(hi, centers) - ball_field(lo, centers)
        } else {
            terrain_field(hi, t) - terrain_field(lo, t)
        };
    }
    // Fields decrease outward, so the outward normal is −∇f.
    (norm([-g[0], -g[1], -g[2]]), is_ball)
}

// ── Per-frame world rebuild (one z-slab per call — thread-disjoint) ──

fn rebuild_slab(bits: &mut [u64], z0: u32, z1: u32, t: f32, centers: &[[f32; 3]; 3]) {
    // `bits` here is just this slab's words; index relative to z0.
    bits.fill(0);
    let words_per_z = (N * N / 64) as usize;
    for z in z0..z1 {
        let zoff = (z - z0) as usize * words_per_z;
        for x in 0..N {
            let h = terrain_height(x as f32 + 0.5, z as f32 + 0.5, t).clamp(2.0, 100.0) as u32;
            for y in 0..h {
                let bit = (x + N * y) as usize;
                bits[zoff + bit / 64] |= 1 << (bit % 64);
            }
        }
        // Metaballs: stamp only where a ball's bounding box meets this z.
        for c in centers {
            let pad = BALL_R + 10.0;
            if (z as f32) < c[2] - pad || (z as f32) > c[2] + pad {
                continue;
            }
            let (x0, x1) = (
                ((c[0] - pad) as i32).max(0),
                ((c[0] + pad) as i32).min(N as i32 - 1),
            );
            let (y0, y1) = (
                ((c[1] - pad) as i32).max(0),
                ((c[1] + pad) as i32).min(N as i32 - 1),
            );
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let p = [x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5];
                    if ball_field(p, centers) > 0.0 {
                        let bit = (x as u32 + N * y as u32) as usize;
                        bits[zoff + bit / 64] |= 1 << (bit % 64);
                    }
                }
            }
        }
    }
}

// ── Fog: one continuous field, three storage formats ────────────────
//
// The F8×F32 experiment: the same f32 density field is stored as f32,
// packed F16, and B8, and the mist march dispatches on the selected
// format at identical traversal cost per step. σ is defined in
// normalized fog coordinates, so any resolution samples the same field
// — quality differences are purely the storage format's.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FogFmt {
    F32,
    F16,
    B8,
    F8,
}

impl FogFmt {
    fn name(self) -> &'static str {
        match self {
            FogFmt::F32 => "f32",
            FogFmt::F16 => "f16",
            FogFmt::B8 => "b8",
            FogFmt::F8 => "f8",
        }
    }
    fn next(self) -> Self {
        match self {
            FogFmt::F32 => FogFmt::F16,
            FogFmt::F16 => FogFmt::B8,
            FogFmt::B8 => FogFmt::F8,
            FogFmt::F8 => FogFmt::F32,
        }
    }
}

struct Fog {
    res: u32,
    f32v: Vec<f32>,
    f16v: Vec<u8>,
    e4m3v: Vec<u8>,
    e5m2v: Vec<u8>,
}

fn build_fog(res: u32) -> Fog {
    let n = (res * res * res) as usize;
    let mut f32v = vec![0.0f32; n];
    let inv = 32.0 / res as f32; // frequencies fixed in 32-cell units
    for z in 0..res {
        for y in 0..res {
            for x in 0..res {
                let (xf, yf, zf) = (x as f32 * inv, y as f32 * inv, z as f32 * inv);
                let layer = (-yf * 0.55).exp();
                let turb =
                    0.55 + 0.45 * (0.55 * xf).sin() * (0.38 * zf).sin() * (0.31 * (xf + zf)).cos();
                f32v[(x + res * (y + res * z)) as usize] = (0.30 * layer * turb).clamp(0.0, 0.4);
            }
        }
    }
    let q4 = Fp8Quantizer::e4m3();
    let q5 = Fp8Quantizer::e5m2();
    let e4m3v = f32v.iter().map(|&v| q4.quantize(v)).collect();
    let e5m2v = f32v.iter().map(|&v| q5.quantize(v)).collect();
    let f16v = f32v
        .iter()
        .flat_map(|&v| (v as f16).to_bits().to_le_bytes())
        .collect();
    Fog {
        res,
        f32v,
        f16v,
        e4m3v,
        e5m2v,
    }
}

// ── Shading ─────────────────────────────────────────────────────────

const SKY_ZENITH: [f32; 3] = [0.22, 0.34, 0.55];
const SKY_HORIZON: [f32; 3] = [0.74, 0.66, 0.60];
const MIST_COLOR: [f32; 3] = [0.78, 0.75, 0.74];

fn terrain_albedo(y: f32) -> [f32; 3] {
    if y >= 72.0 {
        [0.72, 0.72, 0.75]
    } else if y >= 40.0 {
        [0.30, 0.55, 0.24]
    } else if y >= 24.0 {
        [0.76, 0.68, 0.44]
    } else {
        [0.25, 0.42, 0.55]
    }
}

fn sky(dir: [f32; 3], light: [f32; 3]) -> [f32; 3] {
    let f = (dir[1] * 1.4 + 0.25).clamp(0.0, 1.0);
    let base = lerp3(SKY_HORIZON, SKY_ZENITH, f);
    let glow = dot(dir, light).max(0.0).powi(32);
    [
        base[0] + 0.9 * glow,
        base[1] + 0.75 * glow,
        base[2] + 0.45 * glow,
    ]
}

#[allow(clippy::too_many_arguments)]
fn render_band(
    buf: &mut [u32],
    y_start: usize,
    rows: usize,
    bricks: &[u64],
    fog: &Fog,
    fmt: FogFmt,
    t: f32,
    eye: [f32; 3],
    basis: ([f32; 3], [f32; 3], [f32; 3]),
    light: [f32; 3],
    centers: &[[f32; 3]; 3],
) {
    let (fwd, right, up) = basis;
    let tanf = 0.577;
    let aspect = W as f32 / H as f32;
    let fog_scale = fog.res as f32 / N as f32;
    let drift = [
        20.0 * (0.13 * t).sin(),
        6.0 * (0.21 * t).sin(),
        20.0 * (0.11 * t).cos(),
    ];
    let mist_strength = 0.85 + 0.40 * (0.4 * t).sin();

    for (row, line) in buf.chunks_exact_mut(W).enumerate() {
        let py = y_start + row;
        if row >= rows {
            break;
        }
        let sy = (1.0 - 2.0 * (py as f32 + 0.5) / H as f32) * tanf;
        for (px, out) in line.iter_mut().enumerate() {
            let sx = (2.0 * (px as f32 + 0.5) / W as f32 - 1.0) * tanf * aspect;
            let dir = norm([
                fwd[0] + sx * right[0] + sy * up[0],
                fwd[1] + sx * right[1] + sy * up[1],
                fwd[2] + sx * right[2] + sy * up[2],
            ]);

            let (t_hit, idx) = voxel::brick_trace(bricks, [N; 3], eye, dir, T_FAR);
            let mut col;
            let t_end;
            if idx != u32::MAX {
                let p = [
                    eye[0] + t_hit * dir[0],
                    eye[1] + t_hit * dir[1],
                    eye[2] + t_hit * dir[2],
                ];
                let (n, is_ball) = scene_normal(p, t, centers);
                let alb = if is_ball {
                    let pulse = 0.75 + 0.25 * (3.0 * t).sin();
                    [1.0 * pulse, 0.42 * pulse, 0.14 * pulse]
                } else {
                    terrain_albedo(p[1])
                };
                let diff = dot(n, light).max(0.0);
                let shade = 0.30 + 0.70 * diff;
                col = [alb[0] * shade, alb[1] * shade, alb[2] * shade];
                let haze = 1.0 - (-0.002 * t_hit).exp();
                col = lerp3(col, SKY_HORIZON, haze);
                t_end = t_hit;
            } else {
                col = sky(dir, light);
                t_end = T_FAR;
            }

            let fog_o = [
                (eye[0] + drift[0]) * fog_scale,
                (eye[1] + drift[1]) * fog_scale,
                (eye[2] + drift[2]) * fog_scale,
            ];
            let fog_d = [dir[0] * fog_scale, dir[1] * fog_scale, dir[2] * fog_scale];
            let dims = [fog.res; 3];
            let range = t_end.min(MIST_FAR);
            let (tau, _) = match fmt {
                FogFmt::F32 => voxel::march_density_f32(&fog.f32v, dims, fog_o, fog_d, range, 30.0),
                FogFmt::F16 => voxel::march_density_f16(&fog.f16v, dims, fog_o, fog_d, range, 30.0),
                FogFmt::B8 => {
                    voxel::march_density_e4m3(&fog.e4m3v, dims, fog_o, fog_d, range, 30.0)
                }
                FogFmt::F8 => {
                    voxel::march_density_e5m2(&fog.e5m2v, dims, fog_o, fog_d, range, 30.0)
                }
            };
            let mist = 1.0 - (-tau * mist_strength).exp();
            col = lerp3(col, MIST_COLOR, mist);

            *out = pack(col);
        }
    }
}

fn main() {
    let frame_limit: Option<u64> = std::env::var("AXRD_DEMO_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok());
    let bench_mode = frame_limit.is_some();
    let snapshot = std::env::var("AXRD_DEMO_SNAPSHOT").ok();
    let fog_res: u32 = std::env::var("AXRD_FOG_RES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let mut fmt = match std::env::var("AXRD_FOG_FMT").as_deref() {
        Ok("f32") => FogFmt::F32,
        Ok("f16") => FogFmt::F16,
        Ok("f8") => FogFmt::F8,
        _ => FogFmt::B8,
    };
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    let mut window = Window::new(
        "axrd voxel_mt — 256³ world, threaded",
        W,
        H,
        WindowOptions {
            scale: minifb::Scale::X2,
            ..WindowOptions::default()
        },
    )
    .expect("window");
    // Bench mode runs uncapped and with a deterministic timestep, so
    // format A/B runs are wall-clock comparable and pixel-identical.
    let uncap = std::env::var("AXRD_UNCAP").is_ok_and(|v| v != "0");
    window.set_target_fps(if bench_mode || uncap { 0 } else { 60 });

    let voxels = (N as usize).pow(3);
    let mut bits = vec![0u64; voxels / 64];
    let bdim = (N / BRICK) as usize;
    let mut bricks = vec![0u64; bdim * bdim * bdim];
    let fog = build_fog(fog_res);

    let mut buf = vec![0u32; W * H];
    let start = Instant::now();
    let mut frames = 0u64;
    let mut last_title = Instant::now();
    let mut frames_at_title = 0u64;

    let words_per_z = (N * N / 64) as usize;
    let bricks_per_bz = bdim * bdim;

    let mut ms_rebuild = 0f64;
    let mut ms_pack = 0f64;
    let mut ms_render = 0f64;
    let mut f_prev = false;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Key F cycles the fog storage format live.
        let f_now = window.is_key_down(Key::F);
        if f_now && !f_prev {
            fmt = fmt.next();
        }
        f_prev = f_now;

        let t = if bench_mode {
            frames as f32 / 60.0
        } else {
            start.elapsed().as_secs_f32()
        };
        let centers = ball_centers(t);
        let t0 = Instant::now();

        // ── World rebuild: z-slabs are word-disjoint in the bit grid ──
        {
            let slab_z = (N as usize)
                .div_ceil(threads)
                .next_multiple_of(BRICK as usize);
            std::thread::scope(|s| {
                for (i, chunk) in bits.chunks_mut(slab_z * words_per_z).enumerate() {
                    let z0 = (i * slab_z) as u32;
                    let z1 = (z0 + (chunk.len() / words_per_z) as u32).min(N);
                    let centers = &centers;
                    s.spawn(move || rebuild_slab(chunk, z0, z1, t, centers));
                }
            });
        }

        let t1 = Instant::now();
        ms_rebuild += (t1 - t0).as_secs_f64() * 1e3;

        // ── Brick repack: threaded by brick-z slab ──
        {
            let bslab = bdim.div_ceil(threads);
            std::thread::scope(|s| {
                for (i, chunk) in bricks.chunks_mut(bslab * bricks_per_bz).enumerate() {
                    let bz0 = (i * bslab) as u32;
                    let bz1 = bz0 + (chunk.len() / bricks_per_bz) as u32;
                    let bits = &bits;
                    s.spawn(move || voxel::brick_pack_range(bits, [N; 3], chunk, bz0, bz1));
                }
            });
        }

        let t2 = Instant::now();
        ms_pack += (t2 - t1).as_secs_f64() * 1e3;

        // ── Camera + sun ──
        let center = [N as f32 * 0.5, 48.0, N as f32 * 0.5];
        let eye = [
            center[0] + 290.0 * (0.30 * t).cos(),
            104.0 + 32.0 * (0.17 * t).sin(),
            center[2] + 290.0 * (0.30 * t).sin(),
        ];
        let fwd = norm([center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]]);
        let right = norm(cross(fwd, [0.0, 1.0, 0.0]));
        let up = cross(right, fwd);
        let light = norm([(0.45 * t).cos(), 0.85, (0.45 * t).sin()]);

        // ── Render: scanline bands across the pool ──
        {
            let band = H.div_ceil(threads);
            std::thread::scope(|s| {
                for (i, chunk) in buf.chunks_mut(band * W).enumerate() {
                    let rows = chunk.len() / W;
                    let (bricks, fog, centers) = (&bricks, &fog, &centers);
                    s.spawn(move || {
                        render_band(
                            chunk,
                            i * band,
                            rows,
                            bricks,
                            fog,
                            fmt,
                            t,
                            eye,
                            (fwd, right, up),
                            light,
                            centers,
                        )
                    });
                }
            });
        }

        ms_render += t2.elapsed().as_secs_f64() * 1e3;
        window.update_with_buffer(&buf, W, H).expect("present");
        frames += 1;

        if last_title.elapsed().as_secs_f32() > 0.5 {
            let fps = (frames - frames_at_title) as f32 / last_title.elapsed().as_secs_f32();
            window.set_title(&format!(
                "axrd voxel_mt — 256³, {threads} threads, fog {}@{fog_res} — {fps:.0} fps (F cycles)",
                fmt.name()
            ));
            last_title = Instant::now();
            frames_at_title = frames;
        }

        if let Some(limit) = frame_limit {
            if frames >= limit {
                break;
            }
        }
    }

    let secs = start.elapsed().as_secs_f32();
    let ff = frames.max(1) as f64;
    println!(
        "{frames} frames in {secs:.2}s — {:.1} fps avg ({W}x{H}, {threads} threads, fog {}@{fog_res})",
        frames as f32 / secs,
        fmt.name(),
    );
    println!(
        "phase avg: rebuild {:.2} ms, pack {:.2} ms, render {:.2} ms",
        ms_rebuild / ff,
        ms_pack / ff,
        ms_render / ff
    );

    if let Some(path) = snapshot {
        let mut ppm = Vec::with_capacity(W * H * 3 + 32);
        ppm.extend_from_slice(format!("P6\n{W} {H}\n255\n").as_bytes());
        for &px in &buf {
            ppm.extend_from_slice(&[(px >> 16) as u8, (px >> 8) as u8, px as u8]);
        }
        std::fs::write(&path, ppm).expect("snapshot write");
        println!("snapshot written to {path}");
    }
}
