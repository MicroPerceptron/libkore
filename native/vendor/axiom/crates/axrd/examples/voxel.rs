//! Dynamic voxel scene — live ray-traced demo of the `axnn-cpu` voxel
//! kernels.
//!
//! Run with:
//!   cargo +nightly run --release -p axrd --example voxel --features window
//!
//! A 64³ voxel world is rebuilt every frame — a waving heightfield
//! terrain plus an orbiting, bobbing blob — repacked with
//! [`voxel::brick_pack`] and ray-traced per pixel with
//! [`voxel::brick_trace`] (the 4×4×4-brick two-level DDA). A second,
//! static 32³ **E4M3** density volume supplies ground mist, integrated
//! per pixel with [`voxel::march_density_e4m3`] and animated for free
//! by drifting the fog-space transform and pulsing its strength — no
//! requantization after startup. Shading is face-normal Lambert with
//! an orbiting sun, a height palette, depth haze and a gradient sky.
//!
//! Single-threaded on purpose: what you see is the per-core kernel
//! cost, no parallelism hiding it.
//!
//! Controls:
//!   Escape — quit
//!
//! Headless hooks (for CI / verification):
//!   AXRD_DEMO_FRAMES=N     — exit after N frames, print timing stats
//!   AXRD_DEMO_SNAPSHOT=path — write the final frame as a binary PPM

use std::time::Instant;

use axnn_cpu::voxel::{self, BRICK};
use minifb::{Key, Window, WindowOptions};

const W: usize = 480;
const H: usize = 270;

/// World grid edge (voxels).
const N: u32 = 64;
/// Fog volume edge — coarser than the world, quantized to E4M3.
const FOG_N: u32 = 32;
/// Far clip in voxel units.
const T_FAR: f32 = 300.0;

// ── Small vector helpers ────────────────────────────────────────────

fn norm(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-9);
    [v[0] / l, v[1] / l, v[2] / l]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn lerp3(a: [f32; 3], b: [f32; 3], f: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * f,
        a[1] + (b[1] - a[1]) * f,
        a[2] + (b[2] - a[2]) * f,
    ]
}
fn pack(c: [f32; 3]) -> u32 {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    0xFF00_0000 | (q(c[0]) << 16) | (q(c[1]) << 8) | q(c[2])
}

// ── Scene: animated terrain + blob into a bit grid ──────────────────

fn terrain_height(x: f32, z: f32, t: f32) -> f32 {
    10.0 + 4.5 * (0.19 * x + 1.1 * t).sin()
        + 3.5 * (0.23 * z - 0.8 * t).sin()
        + 2.5 * (0.11 * (x + z) + 0.5 * t).sin()
}

fn rebuild_world(bits: &mut [u64], t: f32) {
    bits.fill(0);
    let set = |bits: &mut [u64], x: u32, y: u32, z: u32| {
        let idx = (x + N * (y + N * z)) as usize;
        bits[idx / 64] |= 1 << (idx % 64);
    };
    for z in 0..N {
        for x in 0..N {
            let h = terrain_height(x as f32, z as f32, t).clamp(1.0, 24.0) as u32;
            for y in 0..h {
                set(bits, x, y, z);
            }
        }
    }
    // Orbiting, bobbing blob (kept above the terrain palette bands).
    let bc = [
        32.0 + 13.0 * (0.7 * t).cos(),
        33.0 + 5.0 * (1.3 * t).sin(),
        32.0 + 13.0 * (0.7 * t).sin(),
    ];
    let r = 5.5f32;
    let (lo, hi) = ((bc[0] - r) as i32, (bc[0] + r) as i32 + 1);
    for z in (bc[2] - r) as i32..(bc[2] + r) as i32 + 1 {
        for y in (bc[1] - r) as i32..(bc[1] + r) as i32 + 1 {
            for x in lo..hi {
                if x < 0 || y < 0 || z < 0 || x >= N as i32 || y >= N as i32 || z >= N as i32 {
                    continue;
                }
                let (dx, dy, dz) = (
                    x as f32 + 0.5 - bc[0],
                    y as f32 + 0.5 - bc[1],
                    z as f32 + 0.5 - bc[2],
                );
                if dx * dx + dy * dy + dz * dz <= r * r {
                    set(bits, x as u32, y as u32, z as u32);
                }
            }
        }
    }
}

// ── Fog: static E4M3 puff field, animated via transform + strength ──

fn build_fog() -> Vec<u8> {
    let mut fog = vec![0u8; (FOG_N * FOG_N * FOG_N) as usize];
    for z in 0..FOG_N {
        for y in 0..FOG_N {
            for x in 0..FOG_N {
                let (xf, yf, zf) = (x as f32, y as f32, z as f32);
                // Ground-hugging layer with sine turbulence.
                let layer = (-yf * 0.22).exp();
                let turb =
                    0.55 + 0.45 * (0.55 * xf).sin() * (0.38 * zf).sin() * (0.31 * (xf + zf)).cos();
                let sigma = (0.35 * layer * turb).clamp(0.0, 0.4);
                fog[(x + FOG_N * (y + FOG_N * z)) as usize] =
                    axnn_cpu::fp8::f32_to_e4m3_nearest(sigma);
            }
        }
    }
    fog
}

// ── Shading ─────────────────────────────────────────────────────────

const SKY_ZENITH: [f32; 3] = [0.22, 0.34, 0.55];
const SKY_HORIZON: [f32; 3] = [0.74, 0.66, 0.60];
const MIST_COLOR: [f32; 3] = [0.78, 0.75, 0.74];

fn albedo(vy: u32, vx: u32, vz: u32, t: f32) -> [f32; 3] {
    let checker = if (vx + vy + vz) & 1 == 0 { 0.92 } else { 1.0 };
    let base = if vy >= 26 {
        // The blob: pulsing ember.
        let p = 0.75 + 0.25 * (3.0 * t).sin();
        [1.0 * p, 0.42 * p, 0.14 * p]
    } else if vy >= 18 {
        [0.72, 0.72, 0.75] // rock
    } else if vy >= 10 {
        [0.30, 0.55, 0.24] // grass
    } else if vy >= 6 {
        [0.76, 0.68, 0.44] // sand
    } else {
        [0.25, 0.42, 0.55] // shallows
    };
    [base[0] * checker, base[1] * checker, base[2] * checker]
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

fn main() {
    let frame_limit: Option<u64> = std::env::var("AXRD_DEMO_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok());
    let snapshot = std::env::var("AXRD_DEMO_SNAPSHOT").ok();

    let mut window = Window::new(
        "axrd voxel demo — brick-traced world + E4M3 mist",
        W,
        H,
        WindowOptions {
            scale: minifb::Scale::X2,
            ..WindowOptions::default()
        },
    )
    .expect("window");
    let uncap = std::env::var("AXRD_UNCAP").is_ok_and(|v| v != "0");
    window.set_target_fps(if frame_limit.is_some() || uncap {
        0
    } else {
        60
    });

    let voxels = (N * N * N) as usize;
    let mut bits = vec![0u64; voxels.div_ceil(64)];
    let bdim = N / BRICK;
    let mut bricks = vec![0u64; (bdim * bdim * bdim) as usize];
    let fog = build_fog();
    let fog_scale = FOG_N as f32 / N as f32;

    let mut buf = vec![0u32; W * H];
    let start = Instant::now();
    let mut frames = 0u64;
    let mut last_title = Instant::now();
    let mut frames_at_title = 0u64;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let t = start.elapsed().as_secs_f32();

        // ── Scene update: refill the grid, repack the bricks ──
        rebuild_world(&mut bits, t);
        voxel::brick_pack(&bits, [N; 3], &mut bricks);

        // ── Camera + sun ──
        let center = [N as f32 * 0.5, 12.0, N as f32 * 0.5];
        let eye = [
            center[0] + 74.0 * (0.30 * t).cos(),
            26.0 + 8.0 * (0.17 * t).sin(),
            center[2] + 74.0 * (0.30 * t).sin(),
        ];
        let fwd = norm([center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]]);
        let right = norm(cross(fwd, [0.0, 1.0, 0.0]));
        let up = cross(right, fwd);
        let light = norm([(0.45 * t).cos(), 0.85, (0.45 * t).sin()]);
        let tanf = 0.577; // ~60° vertical FOV
        let aspect = W as f32 / H as f32;

        // Fog animation: drift the sampling transform, pulse strength.
        let drift = [
            5.0 * (0.13 * t).sin(),
            1.5 * (0.21 * t).sin(),
            5.0 * (0.11 * t).cos(),
        ];
        let mist_strength = 1.15 + 0.55 * (0.4 * t).sin();

        for py in 0..H {
            let sy = (1.0 - 2.0 * (py as f32 + 0.5) / H as f32) * tanf;
            for px in 0..W {
                let sx = (2.0 * (px as f32 + 0.5) / W as f32 - 1.0) * tanf * aspect;
                let dir = norm([
                    fwd[0] + sx * right[0] + sy * up[0],
                    fwd[1] + sx * right[1] + sy * up[1],
                    fwd[2] + sx * right[2] + sy * up[2],
                ]);

                // Primary: brick-traced solid world.
                let (t_hit, idx) = voxel::brick_trace(&bricks, [N; 3], eye, dir, T_FAR);
                let mut col;
                let t_end;
                if idx != u32::MAX {
                    let v = [idx % N, (idx / N) % N, idx / (N * N)];
                    let p = [
                        eye[0] + t_hit * dir[0],
                        eye[1] + t_hit * dir[1],
                        eye[2] + t_hit * dir[2],
                    ];
                    // Entry face = axis where p sits furthest from the
                    // voxel center; normal opposes the ray on that axis.
                    let offs = [
                        p[0] - (v[0] as f32 + 0.5),
                        p[1] - (v[1] as f32 + 0.5),
                        p[2] - (v[2] as f32 + 0.5),
                    ];
                    let axis = if offs[0].abs() >= offs[1].abs() && offs[0].abs() >= offs[2].abs() {
                        0
                    } else if offs[1].abs() >= offs[2].abs() {
                        1
                    } else {
                        2
                    };
                    let mut n = [0.0f32; 3];
                    n[axis] = offs[axis].signum();

                    let alb = albedo(v[1], v[0], v[2], t);
                    let diff = dot(n, light).max(0.0);
                    let shade = 0.30 + 0.70 * diff;
                    col = [alb[0] * shade, alb[1] * shade, alb[2] * shade];
                    // Depth haze toward the horizon color.
                    let haze = 1.0 - (-0.008 * t_hit).exp();
                    col = lerp3(col, SKY_HORIZON, haze);
                    t_end = t_hit;
                } else {
                    col = sky(dir, light);
                    t_end = T_FAR;
                }

                // Mist: march the E4M3 fog volume up to the surface.
                // Same parameter t — scaling origin and direction alike
                // keeps fog-space t identical to world t.
                let fog_o = [
                    (eye[0] + drift[0]) * fog_scale,
                    (eye[1] + drift[1]) * fog_scale,
                    (eye[2] + drift[2]) * fog_scale,
                ];
                let fog_d = [dir[0] * fog_scale, dir[1] * fog_scale, dir[2] * fog_scale];
                let (tau, _) =
                    voxel::march_density_e4m3(&fog, [FOG_N; 3], fog_o, fog_d, t_end, 30.0);
                let mist = 1.0 - (-tau * mist_strength).exp();
                col = lerp3(col, MIST_COLOR, mist);

                buf[py * W + px] = pack(col);
            }
        }

        window.update_with_buffer(&buf, W, H).expect("present");
        frames += 1;

        if last_title.elapsed().as_secs_f32() > 0.5 {
            let fps = (frames - frames_at_title) as f32 / last_title.elapsed().as_secs_f32();
            window.set_title(&format!(
                "axrd voxel demo — brick-traced world + E4M3 mist — {fps:.0} fps"
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
    println!(
        "{frames} frames in {secs:.2}s — {:.1} fps avg ({W}x{H}, {} rays/frame + mist march)",
        frames as f32 / secs,
        W * H
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
