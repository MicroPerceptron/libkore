//! Shared glue for the rendering examples: inline 3-vector math,
//! pixel packing, PPM snapshots, and a fast E4M3 quantizer.
//!
//! These are deliberately *not* kernels: `[f32; 3]` camera/shading math
//! is latency-shaped scalar code where inlining is the optimization —
//! the array-shaped work belongs to the `axnn-cpu` kernels the examples
//! drive.
#![allow(dead_code)]

pub fn norm(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-9);
    [v[0] / l, v[1] / l, v[2] / l]
}

pub fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn lerp3(a: [f32; 3], b: [f32; 3], f: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * f,
        a[1] + (b[1] - a[1]) * f,
        a[2] + (b[2] - a[2]) * f,
    ]
}

/// Pack linear-ish RGB to minifb's `0RGB` u32.
pub fn pack(c: [f32; 3]) -> u32 {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    0xFF00_0000 | (q(c[0]) << 16) | (q(c[1]) << 8) | q(c[2])
}

/// Unpack a `0RGB` u32 back to RGB floats.
pub fn unpack(p: u32) -> [f32; 3] {
    [
        ((p >> 16) & 0xFF) as f32 / 255.0,
        ((p >> 8) & 0xFF) as f32 / 255.0,
        (p & 0xFF) as f32 / 255.0,
    ]
}

/// Nearest-value FP8 encoder via a sorted table of the non-negative
/// finite encodings — O(log n) per value, fast enough to quantize
/// volumes and per-frame SH tables. Works for either OCP format; both
/// carry their sign in bit 7, so signed values encode as
/// sign | quantize(|v|).
pub struct Fp8Quantizer {
    table: Vec<(f32, u8)>,
}

impl Fp8Quantizer {
    fn build(decode: impl Fn(u8) -> f32) -> Self {
        let mut table: Vec<(f32, u8)> = (0..=255u8)
            .filter_map(|b| {
                let v = decode(b);
                (v.is_finite() && v >= 0.0).then_some((v, b))
            })
            .collect();
        table.sort_by(|a, b| a.0.total_cmp(&b.0));
        Fp8Quantizer { table }
    }

    pub fn e4m3() -> Self {
        Self::build(axnn_cpu::fp8::e4m3_to_f32)
    }

    pub fn e5m2() -> Self {
        Self::build(axnn_cpu::fp8::e5m2_to_f32)
    }

    /// Nearest encoding of a signed value (sign bit + magnitude).
    pub fn quantize_signed(&self, v: f32) -> u8 {
        let m = self.quantize(v.abs());
        if v < 0.0 { m | 0x80 } else { m }
    }

    /// Nearest non-negative encoding of `v.max(0)`.
    pub fn quantize(&self, v: f32) -> u8 {
        let i = self.table.partition_point(|&(tv, _)| tv < v);
        let hi = self
            .table
            .get(i)
            .copied()
            .unwrap_or(*self.table.last().unwrap());
        let lo = if i > 0 { self.table[i - 1] } else { hi };
        if (v - lo.0).abs() <= (hi.0 - v).abs() {
            lo.1
        } else {
            hi.1
        }
    }
}

/// Write a binary PPM snapshot of a `0RGB` framebuffer.
pub fn write_ppm(path: &str, buf: &[u32], w: usize, h: usize) {
    let mut ppm = Vec::with_capacity(w * h * 3 + 32);
    ppm.extend_from_slice(format!("P6\n{w} {h}\n255\n").as_bytes());
    for &px in buf {
        ppm.extend_from_slice(&[(px >> 16) as u8, (px >> 8) as u8, px as u8]);
    }
    std::fs::write(path, ppm).expect("snapshot write");
    println!("snapshot written to {path}");
}
