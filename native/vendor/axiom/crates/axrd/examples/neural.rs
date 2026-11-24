//! Neural pattern renderer — the F8×F8 FMLAL GEMMs as a live demo.
//!
//! Run with:
//!   cargo +nightly run --release -p axrd --example neural --features window
//!
//! Every pixel of every frame is produced by a small coordinate MLP
//! (a CPPN): 8 animated input features → 48 → 48 → RGB. In the F8
//! modes, **both the weights and the per-layer activations are F8**,
//! and each layer is one call to the double-F8 bias GEMM
//! (`gemm_e4m3_e4m3_f16` / `gemm_e5m2_e5m2_f16`) — the kernels the
//! NEON FEAT_FHM tier runs through FMLAL/FMLAL2, engaged **by default**
//! on Apple-silicon builds (no cargo features, no RUSTFLAGS). Biases
//! ride as a constant `1.0` input feature, so the GEMM's bias plane is
//! a shared zero tile.
//!
//! Key **G** cycles B8 → F8 → F32 (the f32 reference runs the same
//! network through `linalg::gemm_f32`), so quantization quality and
//! throughput are directly comparable on the same animation.
//! Activations re-quantize to F8 every layer with a fast bit-level
//! round-to-nearest encoder (via the F16 halfway representation — the
//! inverse of the kernels' unpack trick).
//!
//! Controls:
//!   Escape — quit;  G — cycle network format
//!
//! Headless hooks:
//!   AXRD_DEMO_FRAMES=N      — exit after N frames, deterministic time
//!   AXRD_DEMO_SNAPSHOT=path — final frame as binary PPM

#![feature(f16)]

use std::time::Instant;

use minifb::{Key, Window, WindowOptions};

#[path = "common/mod.rs"]
mod common;
use common::write_ppm;

const W: usize = 480;
const H: usize = 270;

/// Input features (last one is the constant `1.0` carrying the bias).
const NIN: usize = 8;
const HID: usize = 48;
const NOUT: usize = 3;

#[derive(Clone, Copy, PartialEq, Eq)]
enum NetFmt {
    B8,
    F8,
    F32,
}

impl NetFmt {
    fn name(self) -> &'static str {
        match self {
            NetFmt::B8 => "b8",
            NetFmt::F8 => "f8",
            NetFmt::F32 => "f32",
        }
    }
    fn next(self) -> Self {
        match self {
            NetFmt::B8 => NetFmt::F8,
            NetFmt::F8 => NetFmt::F32,
            NetFmt::F32 => NetFmt::B8,
        }
    }
}

// ── Fast F8 encoders ───────────────────────────────────────────────
//
// Round-to-nearest through the F16 halfway point — the exact inverse
// of the kernels' unpack trick. `f16(v/256)`'s magnitude bits are the
// B8 payload shifted left 7 (subnormals included), so encoding is
// one f16 convert, one add for RNE at the cut, one shift. The double
// rounding (f32→f16→fp8) is at worst 1 ulp of f16 — invisible here.

#[inline(always)]
fn enc_e4m3(v: f32) -> u8 {
    let h = ((v / 256.0) as f16).to_bits();
    let sign = ((h >> 8) & 0x80) as u8;
    let mag = h & 0x7FFF;
    let r = (mag + 0x3F + ((mag >> 7) & 1)) >> 7;
    sign | (r.min(0x7E) as u8) // clamp to ±448 (finite max)
}

#[inline(always)]
fn enc_e5m2(v: f32) -> u8 {
    let h = (v as f16).to_bits();
    let sign = ((h >> 8) & 0x80) as u8;
    let mag = h & 0x7FFF;
    let r = (mag + 0x7F + ((mag >> 8) & 1)) >> 8;
    sign | (r.min(0x7B) as u8) // clamp to ±57344 (finite max)
}

/// F8 encodings of the constant `1.0` bias feature.
const ONE_E4M3: u8 = 0x38;
const ONE_E5M2: u8 = 0x3C;

// ── The network ─────────────────────────────────────────────────────

struct Net {
    /// F8 weights, `[n, k]` rows (the bias-GEMM's B layout).
    w_e4m3: [Vec<u8>; 3],
    w_e5m2: [Vec<u8>; 3],
    /// F32 reference weights in `[k, n]` (the `gemm_f32` B layout).
    w_f32: [Vec<f32>; 3],
}

fn lcg(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    (*state >> 8) as f32 / (1 << 24) as f32 * 2.0 - 1.0
}

fn build_net() -> Net {
    let mut seed = 0xC0FFEE42u32;
    let dims = [(NIN, HID), (HID + 1, HID), (HID + 1, NOUT)];
    let mut w_e4m3: [Vec<u8>; 3] = Default::default();
    let mut w_e5m2: [Vec<u8>; 3] = Default::default();
    let mut w_f32: [Vec<f32>; 3] = Default::default();
    for (l, &(k, n)) in dims.iter().enumerate() {
        // CPPN-friendly scale: deep in softsign's fold region, so the
        // layers carve structure instead of washing to gradients.
        let scale = if l == 0 { 3.0 } else { 4.2 } / (k as f32).sqrt();
        let mut rows = vec![0.0f32; n * k]; // [n, k]
        for w in rows.iter_mut() {
            *w = lcg(&mut seed) * scale;
        }
        w_e4m3[l] = rows.iter().map(|&v| enc_e4m3(v)).collect();
        w_e5m2[l] = rows.iter().map(|&v| enc_e5m2(v)).collect();
        // Transpose to [k, n] for the f32 reference path.
        let mut t = vec![0.0f32; k * n];
        for r in 0..n {
            for c in 0..k {
                t[c * n + r] = rows[r * k + c];
            }
        }
        w_f32[l] = t;
    }
    Net {
        w_e4m3,
        w_e5m2,
        w_f32,
    }
}

#[inline(always)]
fn softsign(x: f32) -> f32 {
    x / (1.0 + x.abs())
}

/// One thread's slice: `rows` scanlines starting at `y0`.
#[allow(clippy::too_many_arguments)]
fn render_band(band: &mut [u32], y0: usize, net: &Net, fmt: NetFmt, t: f32, zero_bias: &[u8]) {
    let m = band.len(); // pixels in this band
    let aspect = W as f32 / H as f32;

    // Features: animated coordinate embeddings + the constant 1.
    let mut feats = vec![0.0f32; m * NIN];
    for (i, f) in feats.chunks_exact_mut(NIN).enumerate() {
        let px = (y0 * W + i) % W;
        let py = (y0 * W + i) / W;
        let x = (2.0 * (px as f32 + 0.5) / W as f32 - 1.0) * aspect;
        let y = 2.0 * (py as f32 + 0.5) / H as f32 - 1.0;
        let r = (x * x + y * y).sqrt();
        f[0] = x;
        f[1] = y;
        f[2] = (3.1 * x + 0.80 * t).sin();
        f[3] = (3.7 * y - 0.60 * t).sin();
        f[4] = (5.3 * (x + y) + 0.40 * t).sin();
        f[5] = (7.9 * r - t).sin();
        f[6] = (2.3 * x * y + 0.5 * t).sin();
        f[7] = 1.0;
    }

    let mut h1 = vec![0.0f32; m * HID];
    let mut h2 = vec![0.0f32; m * HID];
    let mut out = vec![0.0f32; m * NOUT];

    match fmt {
        NetFmt::F32 => {
            axnn_cpu::linalg::gemm_f32(&feats, &net.w_f32[0], &mut h1, m, HID, NIN);
            let mut a2 = vec![0.0f32; m * (HID + 1)];
            for (i, row) in a2.chunks_exact_mut(HID + 1).enumerate() {
                for j in 0..HID {
                    row[j] = softsign(h1[i * HID + j]);
                }
                row[HID] = 1.0;
            }
            axnn_cpu::linalg::gemm_f32(&a2, &net.w_f32[1], &mut h2, m, HID, HID + 1);
            let mut a3 = vec![0.0f32; m * (HID + 1)];
            for (i, row) in a3.chunks_exact_mut(HID + 1).enumerate() {
                for j in 0..HID {
                    row[j] = softsign(h2[i * HID + j]);
                }
                row[HID] = 1.0;
            }
            axnn_cpu::linalg::gemm_f32(&a3, &net.w_f32[2], &mut out, m, NOUT, HID + 1);
        }
        NetFmt::B8 | NetFmt::F8 => {
            let (enc, one, w): (fn(f32) -> u8, u8, &[Vec<u8>; 3]) = match fmt {
                NetFmt::B8 => (enc_e4m3, ONE_E4M3, &net.w_e4m3),
                _ => (enc_e5m2, ONE_E5M2, &net.w_e5m2),
            };
            let gemm =
                |a: &[u8], b: &[u8], d: &mut [f32], m: usize, n: usize, k: usize, bias: &[u8]| {
                    match fmt {
                        NetFmt::B8 => axnn_cpu::fp8::gemm_e4m3_e4m3_f16(a, b, bias, d, m, n, k),
                        _ => axnn_cpu::fp8::gemm_e5m2_e5m2_f16(a, b, bias, d, m, n, k),
                    }
                };

            // Layer 1: quantized features.
            let a1: Vec<u8> = feats.iter().map(|&v| enc(v)).collect();
            gemm(&a1, &w[0], &mut h1, m, HID, NIN, &zero_bias[..m * HID * 2]);

            // Layer 2: F8 activations + the constant-1 bias feature.
            let mut a2 = vec![one; m * (HID + 1)];
            for (i, row) in a2.chunks_exact_mut(HID + 1).enumerate() {
                for j in 0..HID {
                    row[j] = enc(softsign(h1[i * HID + j]));
                }
            }
            gemm(
                &a2,
                &w[1],
                &mut h2,
                m,
                HID,
                HID + 1,
                &zero_bias[..m * HID * 2],
            );

            let mut a3 = vec![one; m * (HID + 1)];
            for (i, row) in a3.chunks_exact_mut(HID + 1).enumerate() {
                for j in 0..HID {
                    row[j] = enc(softsign(h2[i * HID + j]));
                }
            }
            gemm(
                &a3,
                &w[2],
                &mut out,
                m,
                NOUT,
                HID + 1,
                &zero_bias[..m * NOUT * 2],
            );
        }
    }

    for (i, px) in band.iter_mut().enumerate() {
        let q = |c: usize| {
            let v = 0.5 + 0.75 * softsign(2.4 * out[i * NOUT + c]);
            (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32
        };
        *px = 0xFF00_0000 | (q(0) << 16) | (q(1) << 8) | q(2);
    }
}

fn main() {
    let frame_limit: Option<u64> = std::env::var("AXRD_DEMO_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok());
    let bench_mode = frame_limit.is_some();
    let snapshot = std::env::var("AXRD_DEMO_SNAPSHOT").ok();
    let mut fmt = match std::env::var("AXRD_NET_FMT").as_deref() {
        Ok("f32") => NetFmt::F32,
        Ok("f8") => NetFmt::F8,
        _ => NetFmt::B8,
    };
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    let mut window = Window::new(
        "axrd neural — F8 CPPN via FMLAL GEMMs",
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

    let net = build_net();
    let band_rows = H.div_ceil(threads);
    // Shared all-zeros f16 bias plane, sized for the largest band layer.
    let zero_bias = vec![0u8; band_rows * W * HID * 2];

    let mut buf = vec![0u32; W * H];
    let start = Instant::now();
    let mut frames = 0u64;
    let mut last_title = Instant::now();
    let mut frames_at_title = 0u64;
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

        std::thread::scope(|s| {
            for (bi, band) in buf.chunks_mut(band_rows * W).enumerate() {
                let (net, zero_bias) = (&net, &zero_bias);
                s.spawn(move || render_band(band, bi * band_rows, net, fmt, t, zero_bias));
            }
        });

        window.update_with_buffer(&buf, W, H).expect("present");
        frames += 1;

        if last_title.elapsed().as_secs_f32() > 0.5 {
            let fps = (frames - frames_at_title) as f32 / last_title.elapsed().as_secs_f32();
            window.set_title(&format!(
                "axrd neural — {NIN}→{HID}→{HID}→{NOUT} CPPN, net {} — {fps:.0} fps (G cycles)",
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
    println!(
        "{frames} frames in {secs:.2}s — {:.1} fps avg ({W}x{H}, {threads} threads, net {}, ~{} MACs/frame)",
        frames as f32 / secs,
        fmt.name(),
        (W * H * (NIN * HID + (HID + 1) * HID + (HID + 1) * NOUT)) / 1_000_000,
    );

    if let Some(path) = snapshot {
        write_ppm(&path, &buf, W, H);
    }
}
