//! Gaussian-splat scene — the 3DGS kernels (A + B) as a live renderer.
//!
//! Run with:
//!   cargo +nightly run --release -p axrd --example gsplat --features window
//!
//! The third rendering paradigm over the same scene language: a waving
//! terrain carpet, an orbiting ember ball, and drifting valley mist —
//! all as ~8.5k isotropic Gaussians through the real 3DGS pipeline:
//!
//! 1. **SH eval (Kernel A)** — per-splat view-dependent color from
//!    degree-1 spherical harmonics with **F8 coefficient tables**, one
//!    batched [`sh::sh_eval_e4m3`] call (the F8×F32 fused dots). The
//!    tables are static — albedo plus a fixed-axis sheen, quantized
//!    once at startup like a real 3DGS asset — and the orbiting sun is
//!    a per-splat scalar applied after evaluation.
//! 2. Projection — isotropic conics (`a = c = 1/σ_px²`, `b = 0`).
//! 3. Front-to-back depth sort, then 16×16 **tile binning** (bins are
//!    filled in sorted order, so every tile list is depth-sorted).
//! 4. **Compositing (Kernel B)** — [`splat::composite_strip_f32`] per
//!    8-pixel strip, threaded by tile-row bands, blended over the sky
//!    via the surviving transmittance.
//!
//! Coefficient format is switchable (key G / `AXRD_SH_FMT`):
//! `B8` (default, 12 B per splat), `F8` (faster decode, one less
//! mantissa bit), `f32` (48 B reference).
//!
//! Controls:
//!   Escape — quit;  G — cycle SH coefficient format
//!
//! Headless hooks:
//!   AXRD_DEMO_FRAMES=N      — exit after N frames, deterministic time
//!   AXRD_DEMO_SNAPSHOT=path — final frame as binary PPM

use std::time::Instant;

use axnn_cpu::{sh, splat};
use minifb::{Key, Window, WindowOptions};

#[path = "common/mod.rs"]
mod common;
use common::{Fp8Quantizer, cross, dot, lerp3, norm, pack, unpack, write_ppm};

const W: usize = 640;
const H: usize = 360;
const TS: usize = 16; // tile edge
const TX: usize = W / TS;
const TY: usize = H.div_ceil(TS);

const DEG: usize = 1;
const NC: usize = 4; // (DEG+1)²

const SH_C0: f32 = 0.282_094_79;
const SH_C1: f32 = 0.488_602_5;

const SKY_ZENITH: [f32; 3] = [0.22, 0.34, 0.55];
const SKY_HORIZON: [f32; 3] = [0.74, 0.66, 0.60];

#[derive(Clone, Copy, PartialEq, Eq)]
enum ShFmt {
    F32,
    B8,
    F8,
}

impl ShFmt {
    fn name(self) -> &'static str {
        match self {
            ShFmt::F32 => "f32",
            ShFmt::B8 => "b8",
            ShFmt::F8 => "f8",
        }
    }
    fn next(self) -> Self {
        match self {
            ShFmt::F32 => ShFmt::B8,
            ShFmt::B8 => ShFmt::F8,
            ShFmt::F8 => ShFmt::F32,
        }
    }
}

// ── Scene fields (same language as the other demos) ─────────────────

fn terrain_height(x: f32, z: f32, t: f32) -> f32 {
    10.0 + 4.5 * (0.19 * x + 1.1 * t).sin()
        + 3.5 * (0.23 * z - 0.8 * t).sin()
        + 2.5 * (0.11 * (x + z) + 0.5 * t).sin()
}

fn terrain_normal(x: f32, z: f32, t: f32) -> [f32; 3] {
    const E: f32 = 0.5;
    let dhdx = (terrain_height(x + E, z, t) - terrain_height(x - E, z, t)) / (2.0 * E);
    let dhdz = (terrain_height(x, z + E, t) - terrain_height(x, z - E, t)) / (2.0 * E);
    norm([-dhdx, 1.0, -dhdz])
}

fn terrain_albedo(y: f32) -> [f32; 3] {
    if y >= 18.0 {
        [0.72, 0.72, 0.75]
    } else if y >= 10.0 {
        [0.30, 0.55, 0.24]
    } else if y >= 6.0 {
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

// ── Splat set ───────────────────────────────────────────────────────

const T_GRID: usize = 88; // terrain carpet: T_GRID² splats
const N_EMBER: usize = 400;
const N_MIST: usize = 220;
const N_SPLATS: usize = T_GRID * T_GRID + N_EMBER + N_MIST;

struct Splats {
    pos: Vec<[f32; 3]>,
    sigma: Vec<f32>, // world-space isotropic σ
    opacity: Vec<f32>,
    /// Static albedo — baked into the SH tables once at startup.
    albedo: Vec<[f32; 3]>,
    /// Per-frame sun term, applied to the evaluated color (lighting is
    /// not part of the SH radiance, so the tables stay static and the
    /// F8 quantization is a one-time cost — as in a real 3DGS asset).
    shade: Vec<f32>,
}

fn ember_center(t: f32) -> [f32; 3] {
    [
        32.0 + 13.0 * (0.7 * t).cos(),
        33.0 + 5.0 * (1.3 * t).sin(),
        32.0 + 13.0 * (0.7 * t).sin(),
    ]
}

fn build_splats() -> Splats {
    let mut s = Splats {
        pos: vec![[0.0; 3]; N_SPLATS],
        sigma: vec![0.0; N_SPLATS],
        opacity: vec![0.0; N_SPLATS],
        albedo: vec![[0.0; 3]; N_SPLATS],
        shade: vec![1.0; N_SPLATS],
    };
    let spacing = 64.0 / T_GRID as f32;
    for gz in 0..T_GRID {
        for gx in 0..T_GRID {
            let i = gz * T_GRID + gx;
            let (x, z) = ((gx as f32 + 0.5) * spacing, (gz as f32 + 0.5) * spacing);
            // Albedo painted from the rest-pose height — it rides the
            // splat afterwards, like texture on real geometry.
            s.albedo[i] = terrain_albedo(terrain_height(x, z, 0.0));
            s.sigma[i] = spacing * 0.85;
            s.opacity[i] = 0.92;
        }
    }
    for i in 0..N_EMBER {
        let k = T_GRID * T_GRID + i;
        s.albedo[k] = [1.0, 0.42, 0.14];
        s.sigma[k] = 0.95;
        s.opacity[k] = 0.85;
    }
    for i in 0..N_MIST {
        let k = T_GRID * T_GRID + N_EMBER + i;
        s.albedo[k] = [0.80, 0.78, 0.77];
        s.sigma[k] = 6.0 + 4.0 * ((i * 37 % 100) as f32 / 100.0);
        s.opacity[k] = 0.05 + 0.05 * ((i * 61 % 100) as f32 / 100.0);
    }
    s
}

/// Per-frame animation: positions and the scalar sun term.
fn update_splats(s: &mut Splats, t: f32, light: [f32; 3]) {
    let spacing = 64.0 / T_GRID as f32;
    for gz in 0..T_GRID {
        for gx in 0..T_GRID {
            let i = gz * T_GRID + gx;
            let (x, z) = ((gx as f32 + 0.5) * spacing, (gz as f32 + 0.5) * spacing);
            s.pos[i] = [x, terrain_height(x, z, t), z];
            let n = terrain_normal(x, z, t);
            s.shade[i] = 0.30 + 0.70 * dot(n, light).max(0.0);
        }
    }
    let ec = ember_center(t);
    let pulse = 0.75 + 0.25 * (3.0 * t).sin();
    for i in 0..N_EMBER {
        let k = T_GRID * T_GRID + i;
        // Fibonacci sphere.
        let g = 2.399_963_3_f32;
        let yy = 1.0 - 2.0 * (i as f32 + 0.5) / N_EMBER as f32;
        let r = (1.0 - yy * yy).sqrt();
        let a = g * i as f32;
        let n = [r * a.cos(), yy, r * a.sin()];
        s.pos[k] = [ec[0] + 5.5 * n[0], ec[1] + 5.5 * n[1], ec[2] + 5.5 * n[2]];
        s.shade[k] = pulse * (0.55 + 0.45 * dot(n, light).max(0.0));
    }
    for i in 0..N_MIST {
        let k = T_GRID * T_GRID + N_EMBER + i;
        let (fx, fz) = ((i * 29 % 100) as f32 * 0.64, (i * 53 % 100) as f32 * 0.64);
        let x = fx + 5.0 * (0.13 * t + i as f32).sin();
        let z = fz + 5.0 * (0.11 * t + i as f32 * 0.7).cos();
        let y = terrain_height(x, z, t) * 0.35 + 3.0 + 1.5 * (0.2 * t + i as f32).sin();
        s.pos[k] = [x, y, z];
        s.shade[k] = 1.0;
    }
}

/// One-time bake: albedo + a fixed-axis warm sheen into deg-1 SH
/// coefficients, in all three storage formats.
fn bake_coeffs(
    s: &Splats,
    q4: &Fp8Quantizer,
    q5: &Fp8Quantizer,
    f32v: &mut [f32],
    e4m3v: &mut [u8],
    e5m2v: &mut [u8],
) {
    // color(dir) = C0·dc + C1·(−dir.y·a1 + dir.z·a2 − dir.x·a3) + 0.5
    const SHEEN: f32 = 0.18;
    let axis = norm([0.6, 0.55, 0.58]); // fixed sheen direction
    for i in 0..N_SPLATS {
        let alb = s.albedo[i];
        for ch in 0..3 {
            let base = (i * 3 + ch) * NC;
            let dc = (alb[ch] - 0.5) / SH_C0;
            let k = SHEEN * alb[ch] / SH_C1;
            let c = [dc, -k * axis[1], k * axis[2], -k * axis[0]];
            f32v[base..base + NC].copy_from_slice(&c);
            for (j, &v) in c.iter().enumerate() {
                e4m3v[base + j] = q4.quantize_signed(v);
                e5m2v[base + j] = q5.quantize_signed(v);
            }
        }
    }
}

fn main() {
    let frame_limit: Option<u64> = std::env::var("AXRD_DEMO_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok());
    let bench_mode = frame_limit.is_some();
    let snapshot = std::env::var("AXRD_DEMO_SNAPSHOT").ok();
    let mut fmt = match std::env::var("AXRD_SH_FMT").as_deref() {
        Ok("f32") => ShFmt::F32,
        Ok("f8") => ShFmt::F8,
        _ => ShFmt::B8,
    };
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    let mut window = Window::new(
        "axrd gsplat — 3DGS kernels",
        W,
        H,
        WindowOptions {
            scale: minifb::Scale::X2,
            ..WindowOptions::default()
        },
    )
    .expect("window");
    let uncap = std::env::var("AXRD_UNCAP").is_ok_and(|v| v != "0");
    window.set_target_fps(if bench_mode || uncap { 0 } else { 60 });

    let mut splats = build_splats();
    let q4 = Fp8Quantizer::e4m3();
    let q5 = Fp8Quantizer::e5m2();

    let mut coeff_f32 = vec![0.0f32; N_SPLATS * 3 * NC];
    let mut coeff_e4m3 = vec![0u8; N_SPLATS * 3 * NC];
    let mut coeff_e5m2 = vec![0u8; N_SPLATS * 3 * NC];
    bake_coeffs(
        &splats,
        &q4,
        &q5,
        &mut coeff_f32,
        &mut coeff_e4m3,
        &mut coeff_e5m2,
    );
    let mut dirs = vec![0.0f32; N_SPLATS * 3];
    let mut colors = vec![0.0f32; N_SPLATS * 3];
    let mut center = vec![0.0f32; N_SPLATS * 2];
    let mut conic = vec![0.0f32; N_SPLATS * 3];
    let mut opac = vec![0.0f32; N_SPLATS];
    let mut depth = vec![0.0f32; N_SPLATS];
    let mut order: Vec<u32> = Vec::with_capacity(N_SPLATS);
    let mut tiles: Vec<Vec<u32>> = vec![Vec::new(); TX * TY];

    let mut buf = vec![0u32; W * H];
    let start = Instant::now();
    let mut frames = 0u64;
    let mut last_title = Instant::now();
    let mut frames_at_title = 0u64;
    let mut ms_prep = 0f64;
    let mut ms_bin = 0f64;
    let mut ms_comp = 0f64;
    let mut g_prev = false;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let g_now = window.is_key_down(Key::G);
        if g_now && !g_prev {
            fmt = fmt.next();
        }
        g_prev = g_now;

        let t = if bench_mode {
            frames as f32 / 60.0
        } else {
            start.elapsed().as_secs_f32()
        };
        let t0 = Instant::now();

        // ── Camera + sun (shared rig) ──
        let center_w = [32.0, 12.0, 32.0];
        let eye = [
            center_w[0] + 74.0 * (0.30 * t).cos(),
            26.0 + 8.0 * (0.17 * t).sin(),
            center_w[2] + 74.0 * (0.30 * t).sin(),
        ];
        let fwd = norm([
            center_w[0] - eye[0],
            center_w[1] - eye[1],
            center_w[2] - eye[2],
        ]);
        let right = norm(cross(fwd, [0.0, 1.0, 0.0]));
        let up = cross(right, fwd);
        let light = norm([(0.45 * t).cos(), 0.85, (0.45 * t).sin()]);
        let tanf = 0.577f32;
        let aspect = W as f32 / H as f32;
        let f_px = (H as f32 * 0.5) / tanf;

        // ── Kernel A: animate, evaluate the static SH tables ──
        update_splats(&mut splats, t, light);
        for i in 0..N_SPLATS {
            let d = norm([
                splats.pos[i][0] - eye[0],
                splats.pos[i][1] - eye[1],
                splats.pos[i][2] - eye[2],
            ]);
            dirs[i * 3] = d[0];
            dirs[i * 3 + 1] = d[1];
            dirs[i * 3 + 2] = d[2];
        }
        match fmt {
            ShFmt::F32 => sh::sh_eval_f32(&dirs, &coeff_f32, &mut colors, N_SPLATS, DEG),
            ShFmt::B8 => sh::sh_eval_e4m3(&dirs, &coeff_e4m3, &mut colors, N_SPLATS, DEG),
            ShFmt::F8 => sh::sh_eval_e5m2(&dirs, &coeff_e5m2, &mut colors, N_SPLATS, DEG),
        }
        // 3DGS activation, then the dynamic sun term.
        for i in 0..N_SPLATS {
            let sh_ = splats.shade[i];
            for ch in 0..3 {
                let c = &mut colors[i * 3 + ch];
                *c = (*c + 0.5).max(0.0) * sh_;
            }
        }

        // ── Project to screen, isotropic conics ──
        for i in 0..N_SPLATS {
            let rel = [
                splats.pos[i][0] - eye[0],
                splats.pos[i][1] - eye[1],
                splats.pos[i][2] - eye[2],
            ];
            let (xv, yv, zv) = (dot(rel, right), dot(rel, up), dot(rel, fwd));
            depth[i] = zv;
            if zv < 0.8 {
                opac[i] = 0.0;
                continue;
            }
            let sx = (xv / zv / (tanf * aspect) + 1.0) * 0.5 * W as f32;
            let sy = (1.0 - yv / zv / tanf) * 0.5 * H as f32;
            let sig = (f_px * splats.sigma[i] / zv).clamp(0.6, 64.0);
            center[i * 2] = sx;
            center[i * 2 + 1] = sy;
            let inv = 1.0 / (sig * sig);
            conic[i * 3] = inv;
            conic[i * 3 + 1] = 0.0;
            conic[i * 3 + 2] = inv;
            opac[i] = splats.opacity[i];
        }
        let t1 = Instant::now();
        ms_prep += (t1 - t0).as_secs_f64() * 1e3;

        // ── Sort front-to-back, bin to tiles in sorted order ──
        order.clear();
        order.extend((0..N_SPLATS as u32).filter(|&i| opac[i as usize] > 0.0));
        order.sort_unstable_by(|&a, &b| depth[a as usize].total_cmp(&depth[b as usize]));
        for l in tiles.iter_mut() {
            l.clear();
        }
        for &i in &order {
            let s = i as usize;
            let r = 3.0 * (1.0 / conic[s * 3]).sqrt(); // 3σ_px
            let (cx, cy) = (center[s * 2], center[s * 2 + 1]);
            let tx0 = (((cx - r) / TS as f32) as i32).max(0);
            let tx1 = (((cx + r) / TS as f32) as i32).min(TX as i32 - 1);
            let ty0 = (((cy - r) / TS as f32) as i32).max(0);
            let ty1 = (((cy + r) / TS as f32) as i32).min(TY as i32 - 1);
            for ty in ty0..=ty1 {
                for tx in tx0..=tx1 {
                    tiles[ty as usize * TX + tx as usize].push(i);
                }
            }
        }
        let t2 = Instant::now();
        ms_bin += (t2 - t1).as_secs_f64() * 1e3;

        // ── Kernel B: composite tile bands over the sky, threaded ──
        {
            let rows_per_band = TS * TX.div_ceil(threads).max(1);
            let band_rows = (TY.div_ceil(threads)).max(1) * TS;
            let _ = rows_per_band;
            let (tiles, center, conic, opac, colors) = (&tiles, &center, &conic, &opac, &colors);
            std::thread::scope(|sc| {
                for (bi, band) in buf.chunks_mut(band_rows * W).enumerate() {
                    sc.spawn(move || {
                        let y_base = bi * band_rows;
                        // Sky first.
                        for (r, line) in band.chunks_exact_mut(W).enumerate() {
                            let py = y_base + r;
                            let sy = (1.0 - 2.0 * (py as f32 + 0.5) / H as f32) * tanf;
                            for (px, out) in line.iter_mut().enumerate() {
                                let sx = (2.0 * (px as f32 + 0.5) / W as f32 - 1.0) * tanf * aspect;
                                let dir = norm([
                                    fwd[0] + sx * right[0] + sy * up[0],
                                    fwd[1] + sx * right[1] + sy * up[1],
                                    fwd[2] + sx * right[2] + sy * up[2],
                                ]);
                                *out = pack(sky(dir, light));
                            }
                        }
                        // Splats over it, tile by tile.
                        let band_h = band.len() / W;
                        for ty in (y_base / TS)..((y_base + band_h).div_ceil(TS)) {
                            let rows = (H - ty * TS).min(TS);
                            for tx in 0..TX {
                                let ids = &tiles[ty * TX + tx];
                                if ids.is_empty() {
                                    continue;
                                }
                                for r in 0..rows {
                                    let py = ty * TS + r;
                                    let line_off = (py - y_base) * W;
                                    for strip in 0..TS / splat::PW {
                                        let x0 = tx * TS + strip * splat::PW;
                                        let mut rgb = [0.0f32; 3 * splat::PW];
                                        let mut tt = [1.0f32; splat::PW];
                                        let consumed = splat::composite_strip_f32(
                                            center,
                                            conic,
                                            opac,
                                            colors,
                                            ids,
                                            x0 as f32 + 0.5,
                                            py as f32 + 0.5,
                                            &mut rgb,
                                            &mut tt,
                                        );
                                        let _ = consumed;
                                        for l in 0..splat::PW {
                                            let p = &mut band[line_off + x0 + l];
                                            let s = unpack(*p);
                                            *p = pack([
                                                rgb[l] + tt[l] * s[0],
                                                rgb[splat::PW + l] + tt[l] * s[1],
                                                rgb[2 * splat::PW + l] + tt[l] * s[2],
                                            ]);
                                        }
                                    }
                                }
                            }
                        }
                    });
                }
            });
        }
        ms_comp += t2.elapsed().as_secs_f64() * 1e3;

        window.update_with_buffer(&buf, W, H).expect("present");
        frames += 1;

        if last_title.elapsed().as_secs_f32() > 0.5 {
            let fps = (frames - frames_at_title) as f32 / last_title.elapsed().as_secs_f32();
            window.set_title(&format!(
                "axrd gsplat — {N_SPLATS} splats, SH {} — {fps:.0} fps (G cycles)",
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
        "{frames} frames in {secs:.2}s — {:.1} fps avg ({W}x{H}, {N_SPLATS} splats, {threads} threads, SH {})",
        frames as f32 / secs,
        fmt.name(),
    );
    println!(
        "phase avg: prep+sh {:.2} ms, sort+bin {:.2} ms, composite {:.2} ms",
        ms_prep / ff,
        ms_bin / ff,
        ms_comp / ff
    );

    if let Some(path) = snapshot {
        write_ppm(&path, &buf, W, H);
    }
}
