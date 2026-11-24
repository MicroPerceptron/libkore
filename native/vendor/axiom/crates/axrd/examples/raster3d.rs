//! Triangle-rasterized 3D scene — the classic pipeline, as a
//! comparison against the ray-traced `voxel` examples.
//!
//! Run with:
//!   cargo +nightly run --release -p axrd --example raster3d --features window
//!
//! The same scene language as the voxel demos — a waving terrain and an
//! orbiting ember sphere under an orbiting sun — but built from ~9k
//! Gouraud-shaded triangles and drawn by the `axnn-cpu` software
//! rasterization kernels, one bounding-box row at a time:
//!
//! ```text
//! per-row barycentric ramps (scalar fill)
//!   → cmpge_f32 ×3 + and_masks ×2   coverage from the edge functions
//!   → bary_interp                    affine depth across the row
//!   → depth_test                     z-compare + conditional z-write
//!   → bary_interp ×3 + xrgb_pack     Gouraud color, packed to XRGB
//!   → maskstore_u32                  masked framebuffer write
//! ```
//!
//! Triangles are drawn two-sided (winding normalized per triangle) and
//! interpolation is affine — fine at this triangle density. Deliberately
//! single-threaded, like the `voxel` example it mirrors: the fps is the
//! honest per-core cost of the kernel pipeline.
//!
//! Controls:
//!   Escape — quit
//!
//! Headless hooks:
//!   AXRD_DEMO_FRAMES=N      — exit after N frames, print timing stats
//!   AXRD_DEMO_SNAPSHOT=path — write the final frame as a binary PPM

use std::time::Instant;

use axnn_cpu::raster;
use minifb::{Key, Window, WindowOptions};

const W: usize = 480;
const H: usize = 270;

/// Terrain grid resolution (quads per side) over a 64×64 world.
const G: usize = 64;
const WORLD: f32 = 64.0;

// ── Vector helpers ──────────────────────────────────────────────────

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

// ── Scene (same language as the voxel demos) ────────────────────────

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

const SKY_ZENITH: [f32; 3] = [0.22, 0.34, 0.55];
const SKY_HORIZON: [f32; 3] = [0.74, 0.66, 0.60];

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

// ── Meshes ──────────────────────────────────────────────────────────

struct Mesh {
    /// World positions, filled per frame for animated meshes.
    pos: Vec<[f32; 3]>,
    /// World normals, filled per frame.
    nrm: Vec<[f32; 3]>,
    /// Base albedo per vertex, filled per frame.
    alb: Vec<[f32; 3]>,
    /// Vertex index triples.
    tris: Vec<[u32; 3]>,
}

/// Deep-water color for the skirt base and bottom.
const DEEP: [f32; 3] = [0.16, 0.28, 0.40];

fn terrain_mesh() -> Mesh {
    let s = G as u32 + 1;
    // Grid verts, then four y=0 skirt rings (south/north/west/east),
    // then four bottom corners.
    let verts = (s * s + 4 * s + 4) as usize;
    let mut tris = Vec::with_capacity(G * G * 2 + G * 8 + 2);
    for gz in 0..G as u32 {
        for gx in 0..G as u32 {
            let (a, b, c, d) = (
                gz * s + gx,
                gz * s + gx + 1,
                (gz + 1) * s + gx,
                (gz + 1) * s + gx + 1,
            );
            tris.push([a, b, c]);
            tris.push([b, d, c]);
        }
    }
    // Skirt walls: border column down to the matching y=0 ring vert
    // (drawn two-sided, so winding is free).
    let ring = |side: u32, i: u32| s * s + side * s + i;
    for i in 0..G as u32 {
        let quad = |t0: u32, t1: u32, b0: u32, b1: u32, tris: &mut Vec<[u32; 3]>| {
            tris.push([t0, t1, b0]);
            tris.push([t1, b1, b0]);
        };
        quad(i, i + 1, ring(0, i), ring(0, i + 1), &mut tris); // south z=0
        quad(
            G as u32 * s + i,
            G as u32 * s + i + 1,
            ring(1, i),
            ring(1, i + 1),
            &mut tris,
        );
        quad(i * s, (i + 1) * s, ring(2, i), ring(2, i + 1), &mut tris); // west x=0
        quad(
            i * s + G as u32,
            (i + 1) * s + G as u32,
            ring(3, i),
            ring(3, i + 1),
            &mut tris,
        );
    }
    let c0 = s * s + 4 * s;
    tris.push([c0, c0 + 1, c0 + 2]);
    tris.push([c0 + 1, c0 + 3, c0 + 2]);
    Mesh {
        pos: vec![[0.0; 3]; verts],
        nrm: vec![[0.0; 3]; verts],
        alb: vec![[0.0; 3]; verts],
        tris,
    }
}

const SPH_RINGS: usize = 16;
const SPH_SEGS: usize = 24;

fn sphere_mesh() -> Mesh {
    let s = SPH_SEGS as u32 + 1;
    let verts = (SPH_RINGS + 1) * (SPH_SEGS + 1);
    let mut tris = Vec::new();
    for r in 0..SPH_RINGS as u32 {
        for q in 0..SPH_SEGS as u32 {
            let (a, b, c, d) = (
                r * s + q,
                r * s + q + 1,
                (r + 1) * s + q,
                (r + 1) * s + q + 1,
            );
            tris.push([a, b, c]);
            tris.push([b, d, c]);
        }
    }
    Mesh {
        pos: vec![[0.0; 3]; verts],
        nrm: vec![[0.0; 3]; verts],
        alb: vec![[0.0; 3]; verts],
        tris,
    }
}

fn update_terrain(m: &mut Mesh, t: f32) {
    let s = G + 1;
    let step = WORLD / G as f32;
    for gz in 0..s {
        for gx in 0..s {
            let (x, z) = (gx as f32 * step, gz as f32 * step);
            let y = terrain_height(x, z, t);
            let i = gz * s + gx;
            m.pos[i] = [x, y, z];
            m.nrm[i] = terrain_normal(x, z, t);
            m.alb[i] = terrain_albedo(y);
        }
    }
    // Skirt rings at y = 0 with outward side normals.
    let ring_base = s * s;
    let sides: [([f32; 3], Box<dyn Fn(usize) -> (f32, f32)>); 4] = [
        ([0.0, 0.0, -1.0], Box::new(move |i| (i as f32 * step, 0.0))),
        ([0.0, 0.0, 1.0], Box::new(move |i| (i as f32 * step, WORLD))),
        ([-1.0, 0.0, 0.0], Box::new(move |i| (0.0, i as f32 * step))),
        ([1.0, 0.0, 0.0], Box::new(move |i| (WORLD, i as f32 * step))),
    ];
    for (side, (nrm, xz)) in sides.iter().enumerate() {
        for i in 0..s {
            let (x, z) = xz(i);
            let v = ring_base + side * s + i;
            m.pos[v] = [x, 0.0, z];
            m.nrm[v] = *nrm;
            m.alb[v] = DEEP;
        }
    }
    let c0 = ring_base + 4 * s;
    for (k, (x, z)) in [(0.0, 0.0), (WORLD, 0.0), (0.0, WORLD), (WORLD, WORLD)]
        .into_iter()
        .enumerate()
    {
        m.pos[c0 + k] = [x, 0.0, z];
        m.nrm[c0 + k] = [0.0, -1.0, 0.0];
        m.alb[c0 + k] = [DEEP[0] * 0.5, DEEP[1] * 0.5, DEEP[2] * 0.5];
    }
}

fn update_sphere(m: &mut Mesh, t: f32) {
    let center = [
        32.0 + 13.0 * (0.7 * t).cos(),
        33.0 + 5.0 * (1.3 * t).sin(),
        32.0 + 13.0 * (0.7 * t).sin(),
    ];
    let r = 5.5;
    let pulse = 0.75 + 0.25 * (3.0 * t).sin();
    let ember = [1.0 * pulse, 0.42 * pulse, 0.14 * pulse];
    let s = SPH_SEGS + 1;
    for ring in 0..=SPH_RINGS {
        let phi = ring as f32 / SPH_RINGS as f32 * core::f32::consts::PI;
        for seg in 0..=SPH_SEGS {
            let theta = seg as f32 / SPH_SEGS as f32 * core::f32::consts::TAU;
            let n = [phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin()];
            let i = ring * s + seg;
            m.pos[i] = [
                center[0] + r * n[0],
                center[1] + r * n[1],
                center[2] + r * n[2],
            ];
            m.nrm[i] = n;
            m.alb[i] = ember;
        }
    }
}

// ── The kernel-driven rasterizer ────────────────────────────────────

/// Per-frame row scratch, sized to the framebuffer width.
struct Scratch {
    w0: Vec<f32>,
    w1: Vec<f32>,
    w2: Vec<f32>,
    zeros: Vec<f32>,
    zrow: Vec<f32>,
    r: Vec<f32>,
    g: Vec<f32>,
    b: Vec<f32>,
    m0: Vec<u32>,
    m1: Vec<u32>,
    cov: Vec<u32>,
    dmask: Vec<u32>,
    packed: Vec<u32>,
}

impl Scratch {
    fn new() -> Self {
        Scratch {
            w0: vec![0.0; W],
            w1: vec![0.0; W],
            w2: vec![0.0; W],
            zeros: vec![0.0; W],
            zrow: vec![0.0; W],
            r: vec![0.0; W],
            g: vec![0.0; W],
            b: vec![0.0; W],
            m0: vec![0; W],
            m1: vec![0; W],
            cov: vec![0; W],
            dmask: vec![0; W],
            packed: vec![0; W],
        }
    }
}

/// Screen-space vertex: position, affine depth, lit color.
#[derive(Clone, Copy)]
struct SVert {
    x: f32,
    y: f32,
    z: f32,
    rgb: [f32; 3],
}

#[allow(clippy::too_many_arguments)]
fn draw_triangle(
    fb: &mut [u32],
    zbuf: &mut [f32],
    s: &mut Scratch,
    mut v0: SVert,
    mut v1: SVert,
    v2: SVert,
) {
    // Normalize winding so the interior has all edge functions ≥ 0
    // (two-sided shading — no backface cull).
    let area2 = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
    if area2.abs() < 1e-6 {
        return;
    }
    if area2 < 0.0 {
        core::mem::swap(&mut v0, &mut v1);
    }
    let inv_area = 1.0 / area2.abs();

    let min_x = (v0.x.min(v1.x).min(v2.x).floor().max(0.0)) as usize;
    let max_x = (v0.x.max(v1.x).max(v2.x).ceil().min(W as f32 - 1.0)) as usize;
    let min_y = (v0.y.min(v1.y).min(v2.y).floor().max(0.0)) as usize;
    let max_y = (v0.y.max(v1.y).max(v2.y).ceil().min(H as f32 - 1.0)) as usize;
    if min_x > max_x || min_y > max_y {
        return;
    }
    let bw = max_x - min_x + 1;

    // Edge functions (opposite-vertex convention): eᵢ zero on the edge
    // that excludes vertex i, positive inside. λᵢ = eᵢ / 2A.
    let edge = |a: &SVert, b: &SVert, px: f32, py: f32| -> f32 {
        (b.x - a.x) * (py - a.y) - (b.y - a.y) * (px - a.x)
    };
    let (dx0, dy0) = (-(v2.y - v1.y), v2.x - v1.x);
    let (dx1, dy1) = (-(v0.y - v2.y), v0.x - v2.x);
    let (dx2, dy2) = (-(v1.y - v0.y), v1.x - v0.x);

    for py in min_y..=max_y {
        let fy = py as f32 + 0.5;
        let fx0 = min_x as f32 + 0.5;
        // Barycentric ramps for this row (scalar fill; the kernels do
        // the rest of the row).
        let mut e0 = edge(&v1, &v2, fx0, fy) * inv_area;
        let mut e1 = edge(&v2, &v0, fx0, fy) * inv_area;
        let mut e2 = edge(&v0, &v1, fx0, fy) * inv_area;
        let (s0, s1, s2) = (dx0 * inv_area, dx1 * inv_area, dx2 * inv_area);
        let _ = (dy0, dy1, dy2);
        for i in 0..bw {
            s.w0[i] = e0;
            s.w1[i] = e1;
            s.w2[i] = e2;
            e0 += s0;
            e1 += s1;
            e2 += s2;
        }

        let (w0, w1, w2) = (&s.w0[..bw], &s.w1[..bw], &s.w2[..bw]);
        let zeros = &s.zeros[..bw];

        // Coverage: all three barycentrics non-negative.
        raster::cmpge_f32(w0, zeros, &mut s.m0[..bw]);
        raster::cmpge_f32(w1, zeros, &mut s.m1[..bw]);
        raster::and_masks(&s.m0[..bw], &s.m1[..bw], &mut s.cov[..bw]);
        raster::cmpge_f32(w2, zeros, &mut s.m0[..bw]);
        raster::and_masks(&s.cov[..bw], &s.m0[..bw], &mut s.m1[..bw]);

        // Depth: affine interpolation + z-test against the row.
        raster::bary_interp(w0, w1, w2, v0.z, v1.z, v2.z, &mut s.zrow[..bw]);
        let row = py * W + min_x;
        raster::depth_test(
            &s.m1[..bw],
            &s.zrow[..bw],
            &mut zbuf[row..row + bw],
            &mut s.dmask[..bw],
        );

        // Gouraud color for survivors, packed and masked into the fb.
        raster::bary_interp(w0, w1, w2, v0.rgb[0], v1.rgb[0], v2.rgb[0], &mut s.r[..bw]);
        raster::bary_interp(w0, w1, w2, v0.rgb[1], v1.rgb[1], v2.rgb[1], &mut s.g[..bw]);
        raster::bary_interp(w0, w1, w2, v0.rgb[2], v1.rgb[2], v2.rgb[2], &mut s.b[..bw]);
        // xrgb_pack packs its first argument into the low byte; minifb's
        // 0RGB wants blue there, so the channels go in as (b, g, r).
        raster::xrgb_pack(&s.b[..bw], &s.g[..bw], &s.r[..bw], &mut s.packed[..bw]);
        raster::maskstore_u32(&s.dmask[..bw], &s.packed[..bw], &mut fb[row..row + bw]);
    }
}

fn main() {
    let frame_limit: Option<u64> = std::env::var("AXRD_DEMO_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok());
    let snapshot = std::env::var("AXRD_DEMO_SNAPSHOT").ok();

    let mut window = Window::new(
        "axrd raster3d — software raster kernels",
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

    let mut terrain = terrain_mesh();
    let mut sphere = sphere_mesh();
    let tri_count = terrain.tris.len() + sphere.tris.len();

    let mut fb = vec![0u32; W * H];
    let mut zbuf = vec![f32::MAX; W * H];
    let mut scratch = Scratch::new();
    let mut svert = vec![
        SVert {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            rgb: [0.0; 3]
        };
        terrain.pos.len().max(sphere.pos.len())
    ];

    let start = Instant::now();
    let mut frames = 0u64;
    let mut last_title = Instant::now();
    let mut frames_at_title = 0u64;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let t = start.elapsed().as_secs_f32();

        update_terrain(&mut terrain, t);
        update_sphere(&mut sphere, t);

        // ── Camera + sun (identical to the voxel demo) ──
        let center = [32.0, 12.0, 32.0];
        let eye = [
            center[0] + 74.0 * (0.30 * t).cos(),
            26.0 + 8.0 * (0.17 * t).sin(),
            center[2] + 74.0 * (0.30 * t).sin(),
        ];
        let fwd = norm([center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]]);
        let right = norm(cross(fwd, [0.0, 1.0, 0.0]));
        let up = cross(right, fwd);
        let light = norm([(0.45 * t).cos(), 0.85, (0.45 * t).sin()]);
        let tanf = 0.577f32;
        let aspect = W as f32 / H as f32;

        // ── Sky + depth clear ──
        for py in 0..H {
            let sy = (1.0 - 2.0 * (py as f32 + 0.5) / H as f32) * tanf;
            for px in 0..W {
                let sx = (2.0 * (px as f32 + 0.5) / W as f32 - 1.0) * tanf * aspect;
                let dir = norm([
                    fwd[0] + sx * right[0] + sy * up[0],
                    fwd[1] + sx * right[1] + sy * up[1],
                    fwd[2] + sx * right[2] + sy * up[2],
                ]);
                fb[py * W + px] = pack(sky(dir, light));
            }
        }
        zbuf.fill(f32::MAX);

        // ── Transform + light + rasterize both meshes ──
        for mesh in [&terrain, &sphere] {
            for (i, (&p, (&n, &alb))) in mesh
                .pos
                .iter()
                .zip(mesh.nrm.iter().zip(mesh.alb.iter()))
                .enumerate()
            {
                let rel = [p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]];
                let (xv, yv, zv) = (dot(rel, right), dot(rel, up), dot(rel, fwd));
                let (sx, sy) = if zv > 0.5 {
                    (
                        (xv / zv / (tanf * aspect) + 1.0) * 0.5 * W as f32,
                        (1.0 - yv / zv / tanf) * 0.5 * H as f32,
                    )
                } else {
                    (f32::NAN, f32::NAN)
                };
                let diff = dot(n, light).max(0.0);
                let shade = 0.30 + 0.70 * diff;
                let mut rgb = [alb[0] * shade, alb[1] * shade, alb[2] * shade];
                let haze = 1.0 - (-0.008 * zv.max(0.0)).exp();
                rgb = lerp3(rgb, SKY_HORIZON, haze);
                svert[i] = SVert {
                    x: sx,
                    y: sy,
                    z: zv,
                    rgb,
                };
            }
            for tri in &mesh.tris {
                let (a, b, c) = (
                    svert[tri[0] as usize],
                    svert[tri[1] as usize],
                    svert[tri[2] as usize],
                );
                if a.x.is_nan() || b.x.is_nan() || c.x.is_nan() {
                    continue; // behind the near plane
                }
                draw_triangle(&mut fb, &mut zbuf, &mut scratch, a, b, c);
            }
        }

        window.update_with_buffer(&fb, W, H).expect("present");
        frames += 1;

        if last_title.elapsed().as_secs_f32() > 0.5 {
            let fps = (frames - frames_at_title) as f32 / last_title.elapsed().as_secs_f32();
            window.set_title(&format!(
                "axrd raster3d — {tri_count} tris, software raster kernels — {fps:.0} fps"
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
        "{frames} frames in {secs:.2}s — {:.1} fps avg ({W}x{H}, {tri_count} tris, single thread)",
        frames as f32 / secs,
    );

    if let Some(path) = snapshot {
        let mut ppm = Vec::with_capacity(W * H * 3 + 32);
        ppm.extend_from_slice(format!("P6\n{W} {H}\n255\n").as_bytes());
        for &px in &fb {
            ppm.extend_from_slice(&[(px >> 16) as u8, (px >> 8) as u8, px as u8]);
        }
        std::fs::write(&path, ppm).expect("snapshot write");
        println!("snapshot written to {path}");
    }
}
