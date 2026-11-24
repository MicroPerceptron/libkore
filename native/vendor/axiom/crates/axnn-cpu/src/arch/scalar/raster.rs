//! Raster primitives — scalar fallback.

/// For each lane: dst[i] = if a[i] >= b[i] { 0xFFFF_FFFF } else { 0 }
pub(crate) fn cmpge_f32(a: &[f32], b: &[f32], dst: &mut [u32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    for ((d, &av), &bv) in dst.iter_mut().zip(a).zip(b) {
        *d = if av >= bv { 0xFFFF_FFFF } else { 0 };
    }
}

/// Bitwise AND of two u32 mask slices.
pub(crate) fn and_masks(a: &[u32], b: &[u32], dst: &mut [u32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    for ((d, &av), &bv) in dst.iter_mut().zip(a).zip(b) {
        *d = av & bv;
    }
}

/// Conditionally write src[i] → dst[i] where mask[i] != 0.
pub(crate) fn maskstore_u32(mask: &[u32], src: &[u32], dst: &mut [u32]) {
    debug_assert_eq!(mask.len(), src.len());
    debug_assert_eq!(mask.len(), dst.len());
    for ((d, &m), &s) in dst.iter_mut().zip(mask).zip(src) {
        if m != 0 {
            *d = s;
        }
    }
}

/// Fused 3-edge coverage test + conditional framebuffer write.
pub(crate) fn fused_edge_coverage(
    e0: &[f32],
    e1: &[f32],
    e2: &[f32],
    thresh: f32,
    color: &[u32],
    fb: &mut [u32],
) {
    let n = fb.len();
    debug_assert_eq!(e0.len(), n);
    debug_assert_eq!(e1.len(), n);
    debug_assert_eq!(e2.len(), n);
    debug_assert_eq!(color.len(), n);
    for i in 0..n {
        if e0[i] >= thresh && e1[i] >= thresh && e2[i] >= thresh {
            fb[i] = color[i];
        }
    }
}

// ── Fragment processing ─────────────────────────────────────────────

pub(crate) fn depth_test(coverage: &[u32], frag_z: &[f32], zbuf: &mut [f32], out_mask: &mut [u32]) {
    let n = coverage.len();
    for i in 0..n {
        if coverage[i] != 0 && frag_z[i] <= zbuf[i] {
            zbuf[i] = frag_z[i];
            out_mask[i] = 0xFFFF_FFFF;
        } else {
            out_mask[i] = 0;
        }
    }
}

pub(crate) fn bary_interp(
    w0: &[f32],
    w1: &[f32],
    w2: &[f32],
    a0: f32,
    a1: f32,
    a2: f32,
    dst: &mut [f32],
) {
    for i in 0..w0.len() {
        dst[i] = w0[i] * a0 + w1[i] * a1 + w2[i] * a2;
    }
}

pub(crate) fn alpha_blend(mask: &[u32], src: &[u32], dst: &mut [u32]) {
    for i in 0..mask.len() {
        if mask[i] == 0 {
            continue;
        }
        let s = src[i];
        let d = dst[i];
        let sa = ((s >> 24) & 0xFF) as f32;
        let inv_a = (255.0 - sa) * (1.0 / 255.0);
        let clamp = |sc: f32, dc: f32| -> u32 {
            let v = sc + dc * inv_a;
            if v < 0.0 {
                0
            } else if v > 255.0 {
                255
            } else {
                v as u32
            }
        };
        dst[i] = clamp((s & 0xFF) as f32, (d & 0xFF) as f32)
            | (clamp(((s >> 8) & 0xFF) as f32, ((d >> 8) & 0xFF) as f32) << 8)
            | (clamp(((s >> 16) & 0xFF) as f32, ((d >> 16) & 0xFF) as f32) << 16)
            | 0xFF00_0000;
    }
}

// ── Color packing ───────────────────────────────────────────────────

pub(crate) fn xrgb_unpack(packed: &[u32], r: &mut [f32], g: &mut [f32], b: &mut [f32]) {
    let scale = 1.0 / 255.0;
    for i in 0..packed.len() {
        let p = packed[i];
        r[i] = (p & 0xFF) as f32 * scale;
        g[i] = ((p >> 8) & 0xFF) as f32 * scale;
        b[i] = ((p >> 16) & 0xFF) as f32 * scale;
    }
}

pub(crate) fn xrgb_pack(r: &[f32], g: &[f32], b: &[f32], packed: &mut [u32]) {
    for i in 0..packed.len() {
        let to_u8 = |v: f32| -> u32 {
            let c = v * 255.0;
            if c < 0.0 {
                0
            } else if c > 255.0 {
                255
            } else {
                c as u32
            }
        };
        packed[i] = to_u8(r[i]) | (to_u8(g[i]) << 8) | (to_u8(b[i]) << 16) | 0xFF00_0000;
    }
}

pub(crate) fn xrgb_pack_rounded(r: &[f32], g: &[f32], b: &[f32], packed: &mut [u32]) {
    for i in 0..packed.len() {
        let to_u8 = |v: f32| -> u32 { (v * 255.0 + 0.5).clamp(0.0, 255.0) as u32 };
        packed[i] = to_u8(r[i]) | (to_u8(g[i]) << 8) | (to_u8(b[i]) << 16) | 0xFF00_0000;
    }
}

/// Pack three biased-INT8 planes (`x ^ 0x80` == wrapping `+128`) into XRGB.
pub(crate) fn xrgb_pack_u8_biased(r: &[u8], g: &[u8], b: &[u8], packed: &mut [u32]) {
    let n = packed.len();
    assert!(r.len() >= n && g.len() >= n && b.len() >= n);
    for i in 0..n {
        packed[i] = u32::from(r[i] ^ 0x80)
            | (u32::from(g[i] ^ 0x80) << 8)
            | (u32::from(b[i] ^ 0x80) << 16)
            | 0xFF00_0000;
    }
}

// ── Shape coverage (scalar reference) ────────────────────────────────
//
// The SIMD tier reproduces these operation for operation; keep the
// expression order identical when editing either side.

const RB_MASK: u32 = 0x00FF_00FF;
const ROUND: u32 = 0x0080_0080;
const OPAQUE: u32 = 0xFF00_0000;

/// Premultiply straight `0xXXRRGGBB` by `opacity` with exact ÷255 rounding.
/// The result carries `opacity` in its top byte.
pub(crate) fn premultiply_u32(color: u32, opacity: u8) -> u32 {
    if opacity == 255 {
        return color | OPAQUE;
    }
    scale_pixel_u8(color | OPAQUE, u32::from(opacity))
}

/// Scale every channel of one premultiplied pixel by `factor` (0..=255),
/// exact `round(c · factor / 255)` per channel.
#[inline(always)]
pub(crate) fn scale_pixel_u8(pixel: u32, factor: u32) -> u32 {
    let rb = (pixel & RB_MASK).wrapping_mul(factor).wrapping_add(ROUND);
    let rb = (rb.wrapping_add((rb >> 8) & RB_MASK) >> 8) & RB_MASK;
    let ag = ((pixel >> 8) & RB_MASK)
        .wrapping_mul(factor)
        .wrapping_add(ROUND);
    let ag = (ag.wrapping_add((ag >> 8) & RB_MASK) >> 8) & RB_MASK;
    rb | (ag << 8)
}

/// Source-over of one premultiplied pixel (integer, exact ÷255).
#[inline(always)]
fn over_one(s: u32, d: u32) -> u32 {
    let inv_a = 255 - (s >> 24);
    let rb = (d & RB_MASK).wrapping_mul(inv_a).wrapping_add(ROUND);
    let rb = (rb.wrapping_add((rb >> 8) & RB_MASK) >> 8) & RB_MASK;
    let xg = ((d >> 8) & RB_MASK).wrapping_mul(inv_a).wrapping_add(ROUND);
    let xg = (xg.wrapping_add((xg >> 8) & RB_MASK) >> 8) & RB_MASK;
    (s.wrapping_add(rb).wrapping_add(xg << 8)) | OPAQUE
}

pub(crate) fn rounded_sdf_row_f32(
    out: &mut [f32],
    px0: f32,
    py: f32,
    half_w: f32,
    half_h: f32,
    radius: f32,
) {
    let qy = py.abs() - (half_h - radius);
    let oy2 = qy.max(0.0) * qy.max(0.0);
    for (i, slot) in out.iter_mut().enumerate() {
        let px = px0 + i as f32;
        let qx = px.abs() - (half_w - radius);
        let ox = qx.max(0.0);
        let len = axmath::sqrt_f32(ox * ox + oy2);
        let inner = qx.max(qy).min(0.0);
        *slot = len + inner - radius;
    }
}

#[inline(always)]
fn coverage_one(d: f32) -> u32 {
    ((0.5 - d).clamp(0.0, 1.0) * 255.0 + 0.5) as u32
}

pub(crate) fn sdf_coverage_u8(d: &[f32], cov: &mut [u8]) {
    assert!(d.len() >= cov.len());
    for (slot, &dist) in cov.iter_mut().zip(d) {
        *slot = coverage_one(dist) as u8;
    }
}

pub(crate) fn sdf_ring_coverage_u8(outer: &[f32], inner: &[f32], cov: &mut [u8]) {
    assert!(outer.len() >= cov.len() && inner.len() >= cov.len());
    for i in 0..cov.len() {
        cov[i] = coverage_one(outer[i]).saturating_sub(coverage_one(inner[i])) as u8;
    }
}

pub(crate) fn sdf_falloff_u8(d: &[f32], falloff: &[u8], cov: &mut [u8]) {
    let n = cov.len();
    let len = falloff.len();
    assert!(d.len() >= n && (1..=32).contains(&len));
    if len == 1 {
        cov[..n].fill(falloff[0]);
        return;
    }
    let half = (len / 2) as f32;
    let top = (len - 1) as f32;
    let last = (len - 2) as u32;
    for k in 0..n {
        let t = (d[k] + half).clamp(0.0, top);
        let i = (t as u32).min(last);
        let f = ((t - i as f32) * 256.0) as u32;
        let a = u32::from(falloff[i as usize]);
        let b = u32::from(falloff[i as usize + 1]);
        cov[k] = ((a * (256 - f) + b * f + 128) >> 8) as u8;
    }
}

#[inline(always)]
fn blend_row_by(dst: &mut [u32], pixel: u32, load_one: impl Fn(usize) -> u32) {
    if pixel >> 24 == 0 {
        return;
    }
    let opaque_src = pixel >> 24 == 255;
    for (i, d) in dst.iter_mut().enumerate() {
        let cov = load_one(i);
        if cov == 0 {
            continue;
        }
        if opaque_src && cov == 255 {
            *d = pixel;
            continue;
        }
        *d = over_one(scale_pixel_u8(pixel, cov), *d);
    }
}

pub(crate) fn blend_row_alpha(dst: &mut [u32], pixel: u32, coverage: &[u8]) {
    assert!(coverage.len() >= dst.len());
    blend_row_by(dst, pixel, |i| u32::from(coverage[i]));
}

pub(crate) fn coverage_blend_u8(coverage: &[u8], color: u32, opacity: u8, dst: &mut [u32]) {
    blend_row_alpha(dst, premultiply_u32(color, opacity), coverage);
}

pub(crate) fn coverage_blend_u16(coverage: &[u16], color: u32, opacity: u8, dst: &mut [u32]) {
    assert!(coverage.len() >= dst.len());
    blend_row_by(dst, premultiply_u32(color, opacity), |i| {
        ((u32::from(coverage[i]) + 128) >> 8).min(255)
    });
}

pub(crate) fn coverage_blend_a32(coverage: &[u32], color: u32, opacity: u8, dst: &mut [u32]) {
    assert!(coverage.len() >= dst.len());
    blend_row_by(dst, premultiply_u32(color, opacity), |i| coverage[i] >> 24);
}

pub(crate) fn blend_row_masked(src: &[u32], mask: &[u8], dst: &mut [u32]) {
    let n = dst.len();
    assert!(src.len() >= n && mask.len() >= n);
    for i in 0..n {
        let m = u32::from(mask[i]);
        if m == 0 {
            continue;
        }
        let s = scale_pixel_u8(src[i], m);
        if s >> 24 == 0 {
            continue;
        }
        dst[i] = over_one(s, dst[i]);
    }
}

// ── Blit operations ─────────────────────────────────────────────────

pub(crate) fn blit_opaque(
    src: &[u32],
    src_pitch: usize,
    dst: &mut [u32],
    dst_pitch: usize,
    width: usize,
    height: usize,
) {
    for row in 0..height {
        let s = row * src_pitch;
        let d = row * dst_pitch;
        dst[d..d + width].copy_from_slice(&src[s..s + width]);
    }
}

pub(crate) fn blit_blend(
    src: &[u32],
    src_pitch: usize,
    dst: &mut [u32],
    dst_pitch: usize,
    width: usize,
    height: usize,
) {
    for row in 0..height {
        let s_off = row * src_pitch;
        let d_off = row * dst_pitch;
        for x in 0..width {
            let s = src[s_off + x];
            let sa = s >> 24;
            if sa == 0 {
                continue;
            }
            if sa == 0xFF {
                dst[d_off + x] = s;
                continue;
            }
            let d = dst[d_off + x];
            let inv_a = 255 - sa;
            let dst_rb = d & 0x00FF_00FF;
            let t_rb = dst_rb.wrapping_mul(inv_a);
            let blended_rb =
                ((t_rb.wrapping_add(t_rb >> 8).wrapping_add(0x0080_0080)) >> 8) & 0x00FF_00FF;
            let dst_xg = (d >> 8) & 0x00FF_00FF;
            let t_xg = dst_xg.wrapping_mul(inv_a);
            let blended_xg =
                ((t_xg.wrapping_add(t_xg >> 8).wrapping_add(0x0080_0080)) >> 8) & 0x00FF_00FF;
            dst[d_off + x] = s.wrapping_add(blended_rb).wrapping_add(blended_xg << 8) | 0xFF00_0000;
        }
    }
}

pub(crate) fn blit_nearest(
    src: &[u32],
    src_pitch: usize,
    src_w: usize,
    src_h: usize,
    dst: &mut [u32],
    dst_pitch: usize,
    dst_w: usize,
    dst_h: usize,
) {
    // Re-use portable implementation (scalar path is identical).
    crate::arch::portable::raster::blit_nearest(
        src, src_pitch, src_w, src_h, dst, dst_pitch, dst_w, dst_h,
    );
}
