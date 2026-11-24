//! Raster primitives — software triangle rasterization pipeline.
//!
//! ## Coverage testing
//!   cmpge_f32 → and_masks → maskstore_u32 (decomposed)
//!   fused_edge_coverage (single-pass, keeps masks in registers)
//!
//! ## Fragment processing
//!   depth_test — compare + conditional z-write + return coverage mask
//!   bary_interp — w0*a0 + w1*a1 + w2*a2 across N pixels
//!   alpha_blend — src×srcα + dst×(1−srcα) on packed RGBA
//!
//! ## Color packing
//!   rgba_pack — 4 × f32 channels → packed u32 RGBA (saturating)
//!   rgba_unpack — packed u32 RGBA → 4 × f32 channels [0.0, 1.0]

use core::simd::Select;
use core::simd::num::{SimdFloat, SimdInt};
use core::simd::prelude::*;

use super::W;

// ── Unchecked SIMD load/store ───────────────────────────────────────
//
// `Simd::from_slice` / `copy_to_slice` insert bounds-check branches
// that LLVM can't eliminate (slice length isn't statically known).
// These helpers bypass the checks — callers guarantee `off + W <= len`
// via the chunk math.

#[inline(always)]
unsafe fn load_u32(s: &[u32], off: usize) -> Simd<u32, W> {
    Simd::from_array(unsafe { *(s.as_ptr().add(off) as *const [u32; W]) })
}

#[inline(always)]
unsafe fn load_f32(s: &[f32], off: usize) -> Simd<f32, W> {
    Simd::from_array(unsafe { *(s.as_ptr().add(off) as *const [f32; W]) })
}

#[inline(always)]
unsafe fn load_u16(s: &[u16], off: usize) -> Simd<u16, W> {
    Simd::from_array(unsafe { *(s.as_ptr().add(off) as *const [u16; W]) })
}

/// Byte-plane load: `4 * W` lanes so one widen feeds four packed stores.
#[inline(always)]
unsafe fn load_u8x4w(s: &[u8], off: usize) -> Simd<u8, { 4 * W }> {
    Simd::from_array(unsafe { *(s.as_ptr().add(off) as *const [u8; 4 * W]) })
}

#[inline(always)]
unsafe fn store_u32(d: &mut [u32], off: usize, v: Simd<u32, W>) {
    unsafe { *(d.as_mut_ptr().add(off) as *mut [u32; W]) = *v.as_array() };
}

#[inline(always)]
unsafe fn store_f32(d: &mut [f32], off: usize, v: Simd<f32, W>) {
    unsafe { *(d.as_mut_ptr().add(off) as *mut [f32; W]) = *v.as_array() };
}

#[inline(always)]
unsafe fn store_u16(d: &mut [u16], off: usize, v: Simd<u16, W>) {
    unsafe { *(d.as_mut_ptr().add(off) as *mut [u16; W]) = *v.as_array() };
}

/// For each lane: dst[i] = 0xFFFF_FFFF if a[i] >= b[i], else 0.
///
/// Uses ordered quiet comparison — NaN produces 0 (not covered).
pub(crate) fn cmpge_f32(a: &[f32], b: &[f32], dst: &mut [u32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    let n = a.len();
    let chunks = n / W;
    let quads = chunks / 4;

    unsafe {
        for i in 0..quads {
            let base = i * 4 * W;
            let o0 = base;
            let o1 = base + W;
            let o2 = base + 2 * W;
            let o3 = base + 3 * W;
            let m0 = load_f32(a, o0).simd_ge(load_f32(b, o0));
            let m1 = load_f32(a, o1).simd_ge(load_f32(b, o1));
            let m2 = load_f32(a, o2).simd_ge(load_f32(b, o2));
            let m3 = load_f32(a, o3).simd_ge(load_f32(b, o3));
            let neg1 = Simd::<i32, W>::splat(-1);
            let zero = Simd::<i32, W>::splat(0);
            store_u32(dst, o0, m0.select(neg1, zero).cast());
            store_u32(dst, o1, m1.select(neg1, zero).cast());
            store_u32(dst, o2, m2.select(neg1, zero).cast());
            store_u32(dst, o3, m3.select(neg1, zero).cast());
        }
        let mut off = quads * 4 * W;
        while off + W <= n {
            let m = load_f32(a, off).simd_ge(load_f32(b, off));
            let neg1 = Simd::<i32, W>::splat(-1);
            let zero = Simd::<i32, W>::splat(0);
            store_u32(dst, off, m.select(neg1, zero).cast());
            off += W;
        }
    }

    for j in chunks * W..n {
        dst[j] = if a[j] >= b[j] { 0xFFFF_FFFF } else { 0 };
    }
}

/// Bitwise AND of two u32 mask slices — intersect two edge coverage masks.
pub(crate) fn and_masks(a: &[u32], b: &[u32], dst: &mut [u32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), dst.len());
    let n = a.len();
    let chunks = n / W;
    let tail = n % W;

    unsafe {
        for i in 0..chunks {
            let off = i * W;
            store_u32(dst, off, load_u32(a, off) & load_u32(b, off));
        }
    }

    let off = chunks * W;
    for j in 0..tail {
        dst[off + j] = a[off + j] & b[off + j];
    }
}

/// Conditionally write src[i] → dst[i] where mask[i] != 0.
///
/// Uses `Mask::select` — the high bit of each u32 mask lane drives
/// selection, matching the all-ones/all-zeros pattern from `cmpge_f32`.
pub(crate) fn maskstore_u32(mask: &[u32], src: &[u32], dst: &mut [u32]) {
    debug_assert_eq!(mask.len(), src.len());
    debug_assert_eq!(mask.len(), dst.len());
    let n = mask.len();
    let chunks = n / W;
    let tail = n % W;

    unsafe {
        for i in 0..chunks {
            let off = i * W;
            let vm = load_u32(mask, off);
            let vs = load_u32(src, off);
            let vd = load_u32(dst, off);
            let cond = vm.cast::<i32>().simd_lt(Simd::splat(0i32));
            let r: Simd<u32, W> = cond.select(vs.cast::<i32>(), vd.cast::<i32>()).cast();
            store_u32(dst, off, r);
        }
    }

    let off = chunks * W;
    for j in 0..tail {
        if mask[off + j] != 0 {
            dst[off + j] = src[off + j];
        }
    }
}

/// Fused 3-edge coverage test + conditional framebuffer write.
///
/// Equivalent to composing cmpge_f32 × 3 + and_masks × 2 + maskstore_u32,
/// but without materialising three intermediate mask buffers.
/// All coverage masks stay in SIMD registers between compare and select.
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
    let chunks = n / W;
    let tail = n % W;

    let vt = Simd::<f32, W>::splat(thresh);
    let pf = 2 * W;

    // Inline helper: test 3 edges + masked write for one W-wide chunk.
    /// Safety: caller guarantees `off + W <= len` for all slices.
    #[inline(always)]
    unsafe fn coverage_chunk(
        e0: &[f32],
        e1: &[f32],
        e2: &[f32],
        color: &[u32],
        fb: &mut [u32],
        off: usize,
        vt: Simd<f32, W>,
    ) {
        unsafe {
            let cov = load_f32(e0, off).simd_ge(vt)
                & load_f32(e1, off).simd_ge(vt)
                & load_f32(e2, off).simd_ge(vt);
            let vc = load_u32(color, off);
            let vfb = load_u32(fb, off);
            store_u32(
                fb,
                off,
                cov.select(vc.cast::<i32>(), vfb.cast::<i32>()).cast(),
            );
        }
    }

    let pairs = chunks / 2;
    let odd = chunks & 1;

    unsafe {
        for i in 0..pairs {
            let a = (i * 2) * W;
            let b = a + W;
            // Prefetch next pair
            prefetch_read(fb, b + pf);
            coverage_chunk(e0, e1, e2, color, fb, a, vt);
            coverage_chunk(e0, e1, e2, color, fb, b, vt);
        }
        if odd != 0 {
            coverage_chunk(e0, e1, e2, color, fb, (chunks - 1) * W, vt);
        }
    }

    let off = chunks * W;
    for j in 0..tail {
        let k = off + j;
        if e0[k] >= thresh && e1[k] >= thresh && e2[k] >= thresh {
            fb[k] = color[k];
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Fragment processing
// ═══════════════════════════════════════════════════════════════════

/// Depth test with conditional z-buffer write.
///
/// For each pixel `i`:
///   - If `coverage[i] != 0` AND `frag_z[i] <= zbuf[i]` (less-or-equal = nearer):
///     write `frag_z[i]` into `zbuf[i]`, set `out_mask[i] = 0xFFFF_FFFF`.
///   - Otherwise: leave `zbuf[i]` untouched, `out_mask[i] = 0`.
///
/// The output mask can then be passed to `maskstore_u32` to write
/// the color buffer only for fragments that passed depth.
pub(crate) fn depth_test(coverage: &[u32], frag_z: &[f32], zbuf: &mut [f32], out_mask: &mut [u32]) {
    let n = coverage.len();
    debug_assert_eq!(frag_z.len(), n);
    debug_assert_eq!(zbuf.len(), n);
    debug_assert_eq!(out_mask.len(), n);
    let chunks = n / W;
    let tail = n % W;
    let pf = 2 * W;

    /// Safety: caller guarantees `off + W <= len` for all slices.
    #[inline(always)]
    unsafe fn depth_chunk(
        coverage: &[u32],
        frag_z: &[f32],
        zbuf: &mut [f32],
        out_mask: &mut [u32],
        off: usize,
    ) {
        unsafe {
            let covered = load_u32(coverage, off)
                .cast::<i32>()
                .simd_lt(Simd::splat(0i32));
            let vz = load_f32(frag_z, off);
            let vzb = load_f32(zbuf, off);
            let pass = covered & vz.simd_le(vzb);

            store_f32(zbuf, off, pass.select(vz, vzb));
            store_u32(
                out_mask,
                off,
                pass.select(Simd::<i32, W>::splat(-1), Simd::<i32, W>::splat(0))
                    .cast(),
            );
        }
    }

    let pairs = chunks / 2;
    let odd = chunks & 1;

    unsafe {
        for i in 0..pairs {
            let a = (i * 2) * W;
            let b = a + W;
            prefetch_read(coverage, b + pf);
            prefetch_write(out_mask, b + pf);
            depth_chunk(coverage, frag_z, zbuf, out_mask, a);
            depth_chunk(coverage, frag_z, zbuf, out_mask, b);
        }
        if odd != 0 {
            depth_chunk(coverage, frag_z, zbuf, out_mask, (chunks - 1) * W);
        }
    }

    let off = chunks * W;
    for j in 0..tail {
        let k = off + j;
        if coverage[k] != 0 && frag_z[k] <= zbuf[k] {
            zbuf[k] = frag_z[k];
            out_mask[k] = 0xFFFF_FFFF;
        } else {
            out_mask[k] = 0;
        }
    }
}

/// Depth test directly in unsigned E5M11 storage.
///
/// Non-negative IEEE-style floats are monotonically ordered by their raw
/// bits, so the comparison needs no decode. NaNs retain ordered-float
/// semantics (either operand NaN fails); positive infinity remains a valid
/// far-plane value. Relative to f32 this halves fragment/z-buffer traffic
/// while retaining one more fraction bit than F16.
pub(crate) fn depth_test_uf16e5m11(
    coverage: &[u32],
    frag_z: &[u16],
    zbuf: &mut [u16],
    out_mask: &mut [u32],
) {
    let n = coverage.len();
    debug_assert_eq!(frag_z.len(), n);
    debug_assert_eq!(zbuf.len(), n);
    debug_assert_eq!(out_mask.len(), n);

    #[inline(always)]
    unsafe fn chunk(
        coverage: &[u32],
        frag_z: &[u16],
        zbuf: &mut [u16],
        out_mask: &mut [u32],
        off: usize,
    ) {
        unsafe {
            let covered: Mask<i16, W> = load_u32(coverage, off).simd_ne(Simd::splat(0)).cast();
            let fragment = load_u16(frag_z, off);
            let old = load_u16(zbuf, off);
            // Every NaN encoding lies above +infinity in this unsigned layout,
            // so one raw bound check per operand implements ordered semantics.
            let infinity = Simd::splat(0xF800);
            let pass = covered
                & fragment.simd_le(old)
                & fragment.simd_le(infinity)
                & old.simd_le(infinity);
            store_u16(zbuf, off, pass.select(fragment, old));
            let pass_i32: Mask<i32, W> = pass.cast();
            let pass_u32: Simd<u32, W> = pass_i32
                .select(Simd::<i32, W>::splat(-1), Simd::splat(0))
                .cast();
            store_u32(out_mask, off, pass_u32);
        }
    }

    let chunks = n / W;
    let pairs = chunks / 2;
    unsafe {
        for i in 0..pairs {
            let a = i * 2 * W;
            chunk(coverage, frag_z, zbuf, out_mask, a);
            chunk(coverage, frag_z, zbuf, out_mask, a + W);
        }
        if chunks & 1 != 0 {
            chunk(coverage, frag_z, zbuf, out_mask, (chunks - 1) * W);
        }
    }

    for i in chunks * W..n {
        let f = frag_z[i];
        let z = zbuf[i];
        if coverage[i] != 0 && f <= z && f <= 0xF800 && z <= 0xF800 {
            zbuf[i] = f;
            out_mask[i] = u32::MAX;
        } else {
            out_mask[i] = 0;
        }
    }
}

/// Barycentric interpolation of a single f32 attribute across N pixels.
///
/// `dst[i] = w0[i]*a0 + w1[i]*a1 + w2[i]*a2`
///
/// where `w0`, `w1`, `w2` are per-pixel barycentric weights and
/// `a0`, `a1`, `a2` are the three vertex attribute values (broadcast).
///
/// Call once per attribute channel (e.g. R, G, B, A, U, V, Z).
pub(crate) fn bary_interp(
    w0: &[f32],
    w1: &[f32],
    w2: &[f32],
    a0: f32,
    a1: f32,
    a2: f32,
    dst: &mut [f32],
) {
    let n = w0.len();
    debug_assert_eq!(w1.len(), n);
    debug_assert_eq!(w2.len(), n);
    debug_assert_eq!(dst.len(), n);

    // Scalar loop — LLVM auto-vectorizes this into 3 FMAs with full
    // unrolling, outperforming explicit 4-lane core::simd (see sub_f32
    // investigation).
    for i in 0..n {
        dst[i] = w0[i] * a0 + w1[i] * a1 + w2[i] * a2;
    }
}

/// Hint to pull the cache line containing `slice[off..]` into L1.
#[inline(always)]
unsafe fn prefetch_read(slice: &[u32], off: usize) {
    if off < slice.len() {
        #[cfg(target_arch = "x86_64")]
        {
            use core::arch::x86_64::_MM_HINT_T0;
            use core::arch::x86_64::_mm_prefetch;
            unsafe { _mm_prefetch(slice.as_ptr().add(off) as *const i8, _MM_HINT_T0) };
        }
        #[cfg(target_arch = "aarch64")]
        {
            use core::arch::aarch64::_PREFETCH_LOCALITY3;
            use core::arch::aarch64::_PREFETCH_READ;
            use core::arch::aarch64::_prefetch;
            unsafe {
                _prefetch(
                    slice.as_ptr().add(off) as *const i8,
                    _PREFETCH_READ,
                    _PREFETCH_LOCALITY3,
                )
            };
        }
    }
}

#[inline(always)]
unsafe fn prefetch_write(slice: &mut [u32], off: usize) {
    if off < slice.len() {
        #[cfg(target_arch = "x86_64")]
        {
            use core::arch::x86_64::_MM_HINT_T0;
            use core::arch::x86_64::_mm_prefetch;
            unsafe { _mm_prefetch(slice.as_ptr().add(off) as *const i8, _MM_HINT_T0) };
        }
        #[cfg(target_arch = "aarch64")]
        {
            use core::arch::aarch64::_PREFETCH_LOCALITY3;
            use core::arch::aarch64::_PREFETCH_WRITE;
            use core::arch::aarch64::_prefetch;
            unsafe {
                _prefetch(
                    slice.as_ptr().add(off) as *const i8,
                    _PREFETCH_WRITE,
                    _PREFETCH_LOCALITY3,
                )
            };
        }
    }
}

// ── Constants (computed once, lives in registers) ───────────────────────────

const RB_MASK: u32 = 0x00FF_00FF;
const ROUND: u32 = 0x0080_0080; // +128 in each 16-bit half-lane
const OPAQUE: u32 = 0xFF00_0000; // force X channel to 0xFF

// ── Core blend of one W-wide chunk ──────────────────────────────────────────

/// Integer-only Porter-Duff "src over" on W XRGB pixels.
///
/// Uses the RB/XG lane-split trick:
///   RB = pixel & 0x00FF00FF          →  R and B in 16-bit slots
///   XG = (pixel >> 8) & 0x00FF00FF   →  X(ignored) and G
///
/// Each slot can hold channel × alpha (max 255×255 = 65025 < 65536).
/// Exact ÷255:  (x + (x >> 8) + 128) >> 8   (error = 0 for all 0..=65025)
///
/// Output X byte is forced to 0xFF (opaque) regardless of blend result.
#[inline(always)]
fn blend_chunk(vs: Simd<u32, W>, vd: Simd<u32, W>) -> Simd<u32, W> {
    let rb_mask = Simd::<u32, W>::splat(RB_MASK);
    let round = Simd::<u32, W>::splat(ROUND);
    let opaque = Simd::<u32, W>::splat(OPAQUE);
    let s8 = Simd::<u32, W>::splat(8);
    let s24 = Simd::<u32, W>::splat(24);

    // inv_alpha = 255 − src_α
    let inv_a = Simd::splat(255u32) - (vs >> s24);

    // Split destination into RB / XG channel pairs
    let dst_rb = vd & rb_mask;
    let dst_xg = (vd >> s8) & rb_mask;

    // Multiply + approx ÷255 for RB pair.
    // Mask (t >> 8) before adding to prevent carry from the lower
    // 16-bit lane (B) overflowing into the upper lane (R).
    let t_rb = dst_rb * inv_a + round;
    let blended_rb = ((t_rb + ((t_rb >> s8) & rb_mask)) >> s8) & rb_mask;

    // Same for XG pair (X channel result is garbage but gets overwritten).
    let t_xg = dst_xg * inv_a + round;
    let blended_xg = ((t_xg + ((t_xg >> s8) & rb_mask)) >> s8) & rb_mask;

    // Recombine + force opaque X channel
    (vs + blended_rb + (blended_xg << s8)) | opaque
}

/// Scalar src-over of one premultiplied pixel (same math as
/// `blend_chunk`, for row tails).
#[inline(always)]
fn blend_one(s: u32, d: u32) -> u32 {
    let inv_a = 255 - (s >> 24);
    let rb = (d & RB_MASK) * inv_a + ROUND;
    let rb = ((rb + ((rb >> 8) & RB_MASK)) >> 8) & RB_MASK;
    let xg = ((d >> 8) & RB_MASK) * inv_a + ROUND;
    let xg = ((xg + ((xg >> 8) & RB_MASK)) >> 8) & RB_MASK;
    (s.wrapping_add(rb).wrapping_add(xg << 8)) | OPAQUE
}

/// Fill a pitched `[height, width]` region with one packed pixel —
/// the direct path for opaque `Bind::Fill` (no intermediate tile).
pub(crate) fn fill_rect_u32(
    dst: &mut [u32],
    pitch: usize,
    value: u32,
    width: usize,
    height: usize,
) {
    for row in 0..height {
        dst[row * pitch..row * pitch + width].fill(value);
    }
}

/// Src-over blend one premultiplied pixel across a pitched region —
/// the direct path for translucent `Bind::Fill`.
pub(crate) fn fill_blend_u32(dst: &mut [u32], pitch: usize, src: u32, width: usize, height: usize) {
    let vs = Simd::<u32, W>::splat(src);
    for row in 0..height {
        let r = &mut dst[row * pitch..row * pitch + width];
        let chunks = width / W;
        for c in 0..chunks {
            let vd = Simd::<u32, W>::from_slice(&r[c * W..c * W + W]);
            blend_chunk(vs, vd).copy_to_slice(&mut r[c * W..c * W + W]);
        }
        for px in r[chunks * W..].iter_mut() {
            *px = blend_one(src, *px);
        }
    }
}

/// Apply mask: choose blended pixel where mask != 0, else keep dst.
#[inline(always)]
fn masked_select(vm: Simd<u32, W>, blended: Simd<u32, W>, original: Simd<u32, W>) -> Simd<u32, W> {
    let zero = Simd::<u32, W>::splat(0);
    let cond = vm.simd_ne(zero);
    cond.select(blended, original)
}

// ── Public entry point ──────────────────────────────────────────────────────

/// Alpha-blend `src` over `dst` (both packed XRGB `0xXXRRGGBB` u32),
/// writing only where `mask[i] != 0`.
///
/// Source alpha is read from bits [31:24] of `src` for the blend factor.
/// Output X byte is always forced to `0xFF` (opaque).
///
/// Assumes `src` is **premultiplied alpha**.  If your source is straight
/// alpha, premultiply first or the colour channels will be wrong.
pub(crate) fn alpha_blend(mask: &[u32], src: &[u32], dst: &mut [u32]) {
    debug_assert_eq!(mask.len(), src.len());
    debug_assert_eq!(mask.len(), dst.len());

    let n = mask.len();
    if n == 0 {
        return;
    }

    let chunks = n / W;
    let tail = n % W;

    // ── SIMD: 2× unrolled main loop ────────────────────────────────────
    //
    // Processing two independent chunks per iteration exposes two full
    // multiply→shift dependency chains to the OoO engine simultaneously,
    // keeping both the integer multiply unit and the shift/logic units fed.

    let pairs = chunks / 2;
    let odd_chunk = chunks & 1;

    let zero_u = Simd::<u32, W>::splat(0);
    let alpha_mask_v = Simd::<u32, W>::splat(OPAQUE);

    // Prefetch distance in u32 elements — two chunks ahead.
    // 2 × W × 4 bytes ≈ 64–128 bytes, roughly one to two cache lines.
    let pf_dist = 2 * W;

    unsafe {
        // ── Paired iterations ──────────────────────────────────────────
        for i in 0..pairs {
            let off_a = (i * 2) * W;
            let off_b = off_a + W;

            // Prefetch next pair's data
            prefetch_read(mask, off_b + pf_dist);
            prefetch_read(src, off_b + pf_dist);
            prefetch_read(dst, off_b + pf_dist);
            prefetch_write(dst, off_b + pf_dist);

            // ── Load chunk A & B masks ─────────────────────────────────
            let vm_a = load_u32(mask, off_a);
            let vm_b = load_u32(mask, off_b);

            // Fast path: both chunks fully masked-out → skip entirely
            let any_a = vm_a.simd_ne(zero_u).any();
            let any_b = vm_b.simd_ne(zero_u).any();
            if !any_a && !any_b {
                continue;
            }

            // ── Chunk A ────────────────────────────────────────────────
            if any_a {
                let vs_a = load_u32(src, off_a);
                let vd_a = load_u32(dst, off_a);

                // Fast path: all src pixels fully opaque → masked move
                let all_opaque_a = (vs_a & alpha_mask_v).simd_eq(alpha_mask_v).all();
                let result_a = if all_opaque_a {
                    masked_select(vm_a, vs_a, vd_a)
                } else {
                    masked_select(vm_a, blend_chunk(vs_a, vd_a), vd_a)
                };
                store_u32(dst, off_a, result_a);
            }

            // ── Chunk B ────────────────────────────────────────────────
            if any_b {
                let vs_b = load_u32(src, off_b);
                let vd_b = load_u32(dst, off_b);

                let all_opaque_b = (vs_b & alpha_mask_v).simd_eq(alpha_mask_v).all();
                let result_b = if all_opaque_b {
                    masked_select(vm_b, vs_b, vd_b)
                } else {
                    masked_select(vm_b, blend_chunk(vs_b, vd_b), vd_b)
                };
                store_u32(dst, off_b, result_b);
            }
        }

        // ── Leftover odd chunk (0 or 1) ────────────────────────────────
        if odd_chunk != 0 {
            let off = (chunks - 1) * W;
            let vm = load_u32(mask, off);

            if vm.simd_ne(zero_u).any() {
                let vs = load_u32(src, off);
                let vd = load_u32(dst, off);

                let all_opaque = (vs & alpha_mask_v).simd_eq(alpha_mask_v).all();
                let result = if all_opaque {
                    masked_select(vm, vs, vd)
                } else {
                    masked_select(vm, blend_chunk(vs, vd), vd)
                };
                store_u32(dst, off, result);
            }
        }
    }

    // ── Scalar tail (< W pixels) ───────────────────────────────────────
    //
    // Same integer RB/AG split, branchless blend, conditional store.
    scalar_tail(
        &mask[chunks * W..],
        &src[chunks * W..],
        &mut dst[chunks * W..],
        tail,
    );
}

/// Scalar fallback for the final 0..W-1 pixels.
#[inline(always)]
fn scalar_tail(mask: &[u32], src: &[u32], dst: &mut [u32], count: usize) {
    for i in 0..count {
        if mask[i] == 0 {
            continue;
        }

        let s = src[i];
        let d = dst[i];

        // Fast path: fully opaque src → just copy
        if s & OPAQUE == OPAQUE {
            dst[i] = s;
            continue;
        }

        let inv_a = 255 - (s >> 24);

        // RB channels — mask intermediate shift to prevent carry overflow.
        let dst_rb = d & RB_MASK as u32;
        let t_rb = dst_rb.wrapping_mul(inv_a).wrapping_add(ROUND);
        let blended_rb = (t_rb.wrapping_add((t_rb >> 8) & RB_MASK) >> 8) & RB_MASK as u32;

        // AG channels
        let dst_ag = (d >> 8) & RB_MASK as u32;
        let t_ag = dst_ag.wrapping_mul(inv_a).wrapping_add(ROUND);
        let blended_ag = (t_ag.wrapping_add((t_ag >> 8) & RB_MASK) >> 8) & RB_MASK as u32;

        dst[i] = s.wrapping_add(blended_rb).wrapping_add(blended_ag << 8) | OPAQUE;
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Color packing utilities
// ═══════════════════════════════════════════════════════════════════

/// Unpack N packed XRGB u32 values into 3 separate f32 channel slices.
///
/// Output channels are normalized to `[0.0, 1.0]`.
/// Channel layout: bits [7:0] = R, [15:8] = G, [23:16] = B, [31:24] = X (ignored).
pub(crate) fn xrgb_unpack(packed: &[u32], r: &mut [f32], g: &mut [f32], b: &mut [f32]) {
    let n = packed.len();
    debug_assert_eq!(r.len(), n);
    debug_assert_eq!(g.len(), n);
    debug_assert_eq!(b.len(), n);
    let chunks = n / W;
    let tail = n % W;

    let inv255 = Simd::<f32, W>::splat(1.0 / 255.0);
    let m0xff = Simd::<u32, W>::splat(0xFF);
    let s8 = Simd::splat(8u32);
    let s16 = Simd::splat(16u32);

    unsafe {
        for i in 0..chunks {
            let off = i * W;
            let vp = load_u32(packed, off);

            store_f32(r, off, (vp & m0xff).cast::<f32>() * inv255);
            store_f32(g, off, ((vp >> s8) & m0xff).cast::<f32>() * inv255);
            store_f32(b, off, ((vp >> s16) & m0xff).cast::<f32>() * inv255);
        }
    }

    let off = chunks * W;
    let scale = 1.0 / 255.0;
    for j in 0..tail {
        let p = packed[off + j];
        r[off + j] = (p & 0xFF) as f32 * scale;
        g[off + j] = ((p >> 8) & 0xFF) as f32 * scale;
        b[off + j] = ((p >> 16) & 0xFF) as f32 * scale;
    }
}

/// Pack 3 separate f32 channel slices into N packed XRGB u32 values.
///
/// Input channels are `[0.0, 1.0]`, clamped and scaled to `[0, 255]`.
/// Output X byte is forced to `0xFF` (opaque).
/// Channel layout: bits [7:0] = R, [15:8] = G, [23:16] = B, [31:24] = 0xFF.
pub(crate) fn xrgb_pack(r: &[f32], g: &[f32], b: &[f32], packed: &mut [u32]) {
    let n = packed.len();
    debug_assert_eq!(r.len(), n);
    debug_assert_eq!(g.len(), n);
    debug_assert_eq!(b.len(), n);
    let chunks = n / W;
    let tail = n % W;

    let v255 = Simd::<f32, W>::splat(255.0);
    let zero = Simd::<f32, W>::splat(0.0);
    let opaque = Simd::<u32, W>::splat(OPAQUE);
    let s8 = Simd::splat(8u32);
    let s16 = Simd::splat(16u32);

    let to_u8 =
        |v: Simd<f32, W>| -> Simd<u32, W> { (v * v255).simd_max(zero).simd_min(v255).cast() };

    unsafe {
        for i in 0..chunks {
            let off = i * W;
            let p = to_u8(load_f32(r, off))
                | (to_u8(load_f32(g, off)) << s8)
                | (to_u8(load_f32(b, off)) << s16)
                | opaque;
            store_u32(packed, off, p);
        }
    }

    let off = chunks * W;
    for j in 0..tail {
        let to_byte = |v: f32| -> u32 {
            let c = v * 255.0;
            if c < 0.0 {
                0
            } else if c > 255.0 {
                255
            } else {
                c as u32
            }
        };
        packed[off + j] = to_byte(r[off + j])
            | (to_byte(g[off + j]) << 8)
            | (to_byte(b[off + j]) << 16)
            | 0xFF00_0000;
    }
}

/// Pack normalized channels with round-to-nearest quantization.
///
/// This matches `(clamp(channel, 0, 1) * 255 + 0.5) as u32` while keeping the
/// conversion and interleave in one SIMD pass.
pub(crate) fn xrgb_pack_rounded(r: &[f32], g: &[f32], b: &[f32], packed: &mut [u32]) {
    let n = packed.len();
    debug_assert_eq!(r.len(), n);
    debug_assert_eq!(g.len(), n);
    debug_assert_eq!(b.len(), n);
    let chunks = n / W;
    let tail = n % W;

    let v255 = Simd::<f32, W>::splat(255.0);
    let half = Simd::<f32, W>::splat(0.5);
    let zero = Simd::<f32, W>::splat(0.0);
    let opaque = Simd::<u32, W>::splat(OPAQUE);
    let s8 = Simd::splat(8u32);
    let s16 = Simd::splat(16u32);
    let to_u8 = |v: Simd<f32, W>| -> Simd<u32, W> {
        (v * v255 + half).simd_max(zero).simd_min(v255).cast()
    };

    unsafe {
        for i in 0..chunks {
            let off = i * W;
            let p = to_u8(load_f32(r, off))
                | (to_u8(load_f32(g, off)) << s8)
                | (to_u8(load_f32(b, off)) << s16)
                | opaque;
            store_u32(packed, off, p);
        }
    }

    let off = chunks * W;
    for j in 0..tail {
        let to_u8 = |v: f32| -> u32 { (v * 255.0 + 0.5).clamp(0.0, 255.0) as u32 };
        packed[off + j] =
            to_u8(r[off + j]) | (to_u8(g[off + j]) << 8) | (to_u8(b[off + j]) << 16) | OPAQUE;
    }
}

/// Pack three biased-INT8 planes into XRGB u32 pixels.
///
/// The planes carry display bytes with the TOSA INT8 `-128` bias
/// (`stored = display.wrapping_sub(128)`), so each channel is widened and
/// wrapping-added 128 — `x ^ 0x80` — then interleaved. X byte is `0xFF`.
///
/// Requires `r.len() >= packed.len()` (and likewise `g`, `b`); asserted.
pub(crate) fn xrgb_pack_u8_biased(r: &[u8], g: &[u8], b: &[u8], packed: &mut [u32]) {
    let n = packed.len();
    assert!(r.len() >= n && g.len() >= n && b.len() >= n);
    const WB: usize = 4 * W;
    let chunks = n / WB;

    let bias = Simd::<u8, WB>::splat(0x80);
    let opaque = Simd::<u32, WB>::splat(OPAQUE);
    let s8 = Simd::<u32, WB>::splat(8);
    let s16 = Simd::<u32, WB>::splat(16);

    unsafe {
        for i in 0..chunks {
            let off = i * WB;
            let vr: Simd<u32, WB> = (load_u8x4w(r, off) ^ bias).cast();
            let vg: Simd<u32, WB> = (load_u8x4w(g, off) ^ bias).cast();
            let vb: Simd<u32, WB> = (load_u8x4w(b, off) ^ bias).cast();
            let p = vr | (vg << s8) | (vb << s16) | opaque;
            *(packed.as_mut_ptr().add(off) as *mut [u32; WB]) = *p.as_array();
        }
    }

    for j in chunks * WB..n {
        packed[j] = u32::from(r[j] ^ 0x80)
            | (u32::from(g[j] ^ 0x80) << 8)
            | (u32::from(b[j] ^ 0x80) << 16)
            | OPAQUE;
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Shape coverage — rounded rectangles, rings, shadows
// ═══════════════════════════════════════════════════════════════════
//
// Every kernel here mirrors the scalar tier operation for operation, so
// the two tiers are bit-exact: the same f32 expression order, IEEE sqrt,
// saturating casts, and the same integer ÷255 trick on the blend side.

use crate::arch::scalar::raster::{premultiply_u32, scale_pixel_u8};

/// Lane offsets `[0, 1, …, W−1]` as f32 — exact integers.
#[inline(always)]
fn lane_offsets() -> Simd<f32, W> {
    Simd::from_array(core::array::from_fn(|i| i as f32))
}

/// Signed distance from a rounded-rectangle edge for one row of pixel
/// centres (negative inside).
///
/// `out[i]` is evaluated at `(px0 + i, py)` in shape-centred coordinates
/// with half extents `half_w`/`half_h`; `radius` is the radius of the
/// corner whose quadrant the whole row segment lies in (callers split a
/// row at the centre column). `q = |p| − (half − r)`;
/// `d = |max(q, 0)| + min(max(q.x, q.y), 0) − r`.
pub(crate) fn rounded_sdf_row_f32(
    out: &mut [f32],
    px0: f32,
    py: f32,
    half_w: f32,
    half_h: f32,
    radius: f32,
) {
    let n = out.len();
    let chunks = n / W;

    let qy = py.abs() - (half_h - radius);
    let oy2 = qy.max(0.0) * qy.max(0.0);
    let vqy = Simd::<f32, W>::splat(qy);
    let voy2 = Simd::<f32, W>::splat(oy2);
    let vhx = Simd::<f32, W>::splat(half_w - radius);
    let vpx0 = Simd::<f32, W>::splat(px0);
    let vrad = Simd::<f32, W>::splat(radius);
    let zero = Simd::<f32, W>::splat(0.0);
    let lanes = lane_offsets();

    unsafe {
        for c in 0..chunks {
            let off = c * W;
            let idx = lanes + Simd::splat(off as f32);
            let px = vpx0 + idx;
            let qx = px.abs() - vhx;
            let ox = qx.simd_max(zero);
            // `core::simd` carries no sqrt (that is `std_float`); the
            // per-lane call is libm's hardware `fsqrt`/`sqrtss`, so the
            // lanes stay correctly rounded and identical to the scalar tier.
            let len = Simd::from_array((ox * ox + voy2).to_array().map(axmath::sqrt_f32));
            let inner = qx.simd_max(vqy).simd_min(zero);
            store_f32(out, off, len + inner - vrad);
        }
    }

    for (i, slot) in out.iter_mut().enumerate().skip(chunks * W) {
        let px = px0 + i as f32;
        let qx = px.abs() - (half_w - radius);
        let ox = qx.max(0.0);
        let len = axmath::sqrt_f32(ox * ox + oy2);
        let inner = qx.max(qy).min(0.0);
        *slot = len + inner - radius;
    }
}

/// `cov = round(255 · clamp(0.5 − d, 0, 1))` for one W-wide chunk.
#[inline(always)]
fn coverage_chunk(d: Simd<f32, W>) -> Simd<u32, W> {
    let half = Simd::<f32, W>::splat(0.5);
    let zero = Simd::<f32, W>::splat(0.0);
    let one = Simd::<f32, W>::splat(1.0);
    let v255 = Simd::<f32, W>::splat(255.0);
    ((half - d).simd_max(zero).simd_min(one) * v255 + half).cast()
}

#[inline(always)]
fn coverage_one(d: f32) -> u32 {
    ((0.5 - d).clamp(0.0, 1.0) * 255.0 + 0.5) as u32
}

/// Coverage of a signed-distance row: `round(255 · clamp(0.5 − d, 0, 1))`.
pub(crate) fn sdf_coverage_u8(d: &[f32], cov: &mut [u8]) {
    let n = cov.len();
    assert!(d.len() >= n);
    let chunks = n / W;
    unsafe {
        for c in 0..chunks {
            let off = c * W;
            let v: Simd<u8, W> = coverage_chunk(load_f32(d, off)).cast();
            *(cov.as_mut_ptr().add(off) as *mut [u8; W]) = *v.as_array();
        }
    }
    for i in chunks * W..n {
        cov[i] = coverage_one(d[i]) as u8;
    }
}

/// Ring coverage: `sat(cov(outer) − cov(inner))`; `inner ⊂ outer` so the
/// subtraction never underflows for well-formed inputs.
pub(crate) fn sdf_ring_coverage_u8(outer: &[f32], inner: &[f32], cov: &mut [u8]) {
    let n = cov.len();
    assert!(outer.len() >= n && inner.len() >= n);
    let chunks = n / W;
    unsafe {
        for c in 0..chunks {
            let off = c * W;
            let o = coverage_chunk(load_f32(outer, off));
            let i = coverage_chunk(load_f32(inner, off));
            let v: Simd<u8, W> = o.saturating_sub(i).cast();
            *(cov.as_mut_ptr().add(off) as *mut [u8; W]) = *v.as_array();
        }
    }
    for i in chunks * W..n {
        cov[i] = coverage_one(outer[i]).saturating_sub(coverage_one(inner[i])) as u8;
    }
}

/// Coverage from a distance table with linear interpolation.
///
/// Entry `i` of `falloff` is the coverage at signed distance `i − len/2`;
/// `d` is clamped to the table's range at both ends. `falloff` holds
/// `1..=32` entries (asserted).
pub(crate) fn sdf_falloff_u8(d: &[f32], falloff: &[u8], cov: &mut [u8]) {
    let n = cov.len();
    let len = falloff.len();
    assert!(d.len() >= n && (1..=32).contains(&len));
    if len == 1 {
        cov[..n].fill(falloff[0]);
        return;
    }
    let mut table = [0u32; 32];
    for (slot, &entry) in table.iter_mut().zip(falloff) {
        *slot = u32::from(entry);
    }
    let half = (len / 2) as f32;
    let top = (len - 1) as f32;
    let last = (len - 2) as u32;
    let chunks = n / W;

    let vhalf = Simd::<f32, W>::splat(half);
    let vtop = Simd::<f32, W>::splat(top);
    let zero = Simd::<f32, W>::splat(0.0);
    let vlast = Simd::<u32, W>::splat(last);
    let v256f = Simd::<f32, W>::splat(256.0);
    let v256 = Simd::<u32, W>::splat(256);
    let round = Simd::<u32, W>::splat(128);
    let s8 = Simd::<u32, W>::splat(8);
    let one = Simd::<usize, W>::splat(1);

    unsafe {
        for c in 0..chunks {
            let off = c * W;
            let t = (load_f32(d, off) + vhalf).simd_max(zero).simd_min(vtop);
            let i: Simd<u32, W> = t.cast::<u32>().simd_min(vlast);
            let f: Simd<u32, W> = ((t - i.cast::<f32>()) * v256f).cast();
            let idx: Simd<usize, W> = i.cast();
            let a = Simd::<u32, W>::gather_or_default(&table[..len], idx);
            let b = Simd::<u32, W>::gather_or_default(&table[..len], idx + one);
            let mixed = (a * (v256 - f) + b * f + round) >> s8;
            let v: Simd<u8, W> = mixed.cast();
            *(cov.as_mut_ptr().add(off) as *mut [u8; W]) = *v.as_array();
        }
    }

    for k in chunks * W..n {
        let t = (d[k] + half).clamp(0.0, top);
        let i = (t as u32).min(last);
        let f = ((t - i as f32) * 256.0) as u32;
        let a = table[i as usize];
        let b = table[i as usize + 1];
        cov[k] = ((a * (256 - f) + b * f + 128) >> 8) as u8;
    }
}

/// Scale every channel of W premultiplied pixels by an 8-bit factor
/// (`0..=255` per lane), exact `round(c · f / 255)` per channel.
#[inline(always)]
fn scale_chunk(v: Simd<u32, W>, f: Simd<u32, W>) -> Simd<u32, W> {
    let rb_mask = Simd::<u32, W>::splat(RB_MASK);
    let round = Simd::<u32, W>::splat(ROUND);
    let s8 = Simd::<u32, W>::splat(8);
    let rb = (v & rb_mask) * f + round;
    let rb = ((rb + ((rb >> s8) & rb_mask)) >> s8) & rb_mask;
    let ag = ((v >> s8) & rb_mask) * f + round;
    let ag = ((ag + ((ag >> s8) & rb_mask)) >> s8) & rb_mask;
    rb | (ag << s8)
}

/// Shared body of the coverage blends: `load_chunk(off)` / `load_one(i)`
/// yield the 8-bit factor per pixel. Lanes with zero coverage keep their
/// destination; fully covered lanes of an opaque pixel are stored directly.
#[inline(always)]
fn blend_row_by(
    dst: &mut [u32],
    pixel: u32,
    load_chunk: impl Fn(usize) -> Simd<u32, W>,
    load_one: impl Fn(usize) -> u32,
) {
    let n = dst.len();
    if pixel >> 24 == 0 {
        return;
    }
    let opaque_src = pixel >> 24 == 255;
    let chunks = n / W;
    let vs = Simd::<u32, W>::splat(pixel);
    let zero = Simd::<u32, W>::splat(0);
    let full = Simd::<u32, W>::splat(255);

    unsafe {
        for c in 0..chunks {
            let off = c * W;
            let cov = load_chunk(off);
            let covered = cov.simd_ne(zero);
            if !covered.any() {
                continue;
            }
            if opaque_src && cov.simd_eq(full).all() {
                store_u32(dst, off, vs);
                continue;
            }
            let vd = load_u32(dst, off);
            let blended = blend_chunk(scale_chunk(vs, cov), vd);
            store_u32(dst, off, covered.select(blended, vd));
        }
    }

    for (i, d) in dst.iter_mut().enumerate().skip(chunks * W) {
        let cov = load_one(i);
        if cov == 0 {
            continue;
        }
        if opaque_src && cov == 255 {
            *d = pixel;
            continue;
        }
        *d = blend_one(scale_pixel_u8(pixel, cov), *d);
    }
}

/// Blend one premultiplied pixel over a row through per-pixel 8-bit
/// coverage. `coverage.len() >= dst.len()` (asserted).
pub(crate) fn blend_row_alpha(dst: &mut [u32], pixel: u32, coverage: &[u8]) {
    let n = dst.len();
    assert!(coverage.len() >= n);
    blend_row_by(
        dst,
        pixel,
        |off| Simd::<u8, W>::from_slice(&coverage[off..off + W]).cast(),
        |i| u32::from(coverage[i]),
    );
}

/// [`blend_row_alpha`] with straight `color` and `opacity` — the glyph /
/// icon-mask tint. `color`'s top byte is ignored.
pub(crate) fn coverage_blend_u8(coverage: &[u8], color: u32, opacity: u8, dst: &mut [u32]) {
    blend_row_alpha(dst, premultiply_u32(color, opacity), coverage);
}

/// 16-bit coverage variant: each lane is quantized as
/// `min((c + 128) >> 8, 255)` before the blend (≤ 1/255 from exact).
pub(crate) fn coverage_blend_u16(coverage: &[u16], color: u32, opacity: u8, dst: &mut [u32]) {
    let n = dst.len();
    assert!(coverage.len() >= n);
    let pixel = premultiply_u32(color, opacity);
    let round = Simd::<u32, W>::splat(128);
    let s8 = Simd::<u32, W>::splat(8);
    let full = Simd::<u32, W>::splat(255);
    blend_row_by(
        dst,
        pixel,
        |off| {
            let c: Simd<u32, W> = Simd::<u16, W>::from_slice(&coverage[off..off + W]).cast();
            ((c + round) >> s8).simd_min(full)
        },
        |i| ((u32::from(coverage[i]) + 128) >> 8).min(255),
    );
}

/// Packed-alpha variant: the top byte of each `coverage` word is the
/// 8-bit factor (an ARGB glyph atlas serves as its own mask).
pub(crate) fn coverage_blend_a32(coverage: &[u32], color: u32, opacity: u8, dst: &mut [u32]) {
    let n = dst.len();
    assert!(coverage.len() >= n);
    let pixel = premultiply_u32(color, opacity);
    let s24 = Simd::<u32, W>::splat(24);
    blend_row_by(
        dst,
        pixel,
        |off| unsafe { load_u32(coverage, off) } >> s24,
        |i| coverage[i] >> 24,
    );
}

/// Blend a premultiplied source row over `dst` through an 8-bit mask:
/// `dst = over(src · mask, dst)`. Lanes with a zero mask keep `dst`.
pub(crate) fn blend_row_masked(src: &[u32], mask: &[u8], dst: &mut [u32]) {
    let n = dst.len();
    assert!(src.len() >= n && mask.len() >= n);
    let chunks = n / W;
    let zero = Simd::<u32, W>::splat(0);
    unsafe {
        for c in 0..chunks {
            let off = c * W;
            let m: Simd<u32, W> = Simd::<u8, W>::from_slice(&mask[off..off + W]).cast();
            let covered = m.simd_ne(zero);
            if !covered.any() {
                continue;
            }
            let vs = scale_chunk(load_u32(src, off), m);
            let vd = load_u32(dst, off);
            let live = covered & (vs >> Simd::splat(24)).simd_ne(zero);
            store_u32(dst, off, live.select(blend_chunk(vs, vd), vd));
        }
    }
    for i in chunks * W..n {
        let m = u32::from(mask[i]);
        if m == 0 {
            continue;
        }
        let s = scale_pixel_u8(src[i], m);
        if s >> 24 == 0 {
            continue;
        }
        dst[i] = blend_one(s, dst[i]);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Blit operations
// ═══════════════════════════════════════════════════════════════════

/// Opaque rectangular blit: copy a `width × height` region of XRGB pixels.
///
/// `src` and `dst` are linear pixel buffers.  `src_pitch` / `dst_pitch` are
/// the number of **pixels** (not bytes) between the start of consecutive
/// scanlines in each buffer.
///
/// The region `[0..width)` of each row is copied; rows are separated by
/// the respective pitch.
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

/// Alpha-blended rectangular blit: composite a `width × height` region
/// of premultiplied-alpha XRGB source over an XRGB destination.
///
/// Every source pixel is tested — if `src_α == 0` the pixel is skipped,
/// if `src_α == 0xFF` it's a direct copy, otherwise a full blend is
/// performed.  Uses the SIMD `alpha_blend` kernel per scanline with an
/// all-ones mask (every pixel is eligible).
pub(crate) fn blit_blend(
    src: &[u32],
    src_pitch: usize,
    dst: &mut [u32],
    dst_pitch: usize,
    width: usize,
    height: usize,
) {
    // All-ones mask: every pixel in the row is eligible for blending.
    // alpha_blend's per-pixel alpha check handles the actual coverage.
    //
    // We allocate the mask once on the stack if it's small enough (≤ 1024),
    // otherwise fall back to a scalar per-pixel loop.
    if width <= 1024 {
        let mask_buf = [0xFFFF_FFFFu32; 1024];
        let mask = &mask_buf[..width];
        for row in 0..height {
            let s = row * src_pitch;
            let d = row * dst_pitch;
            alpha_blend(mask, &src[s..s + width], &mut dst[d..d + width]);
        }
    } else {
        // Large rows: per-pixel scalar blend to avoid heap allocation.
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
                let dst_rb = d & RB_MASK;
                let t_rb = dst_rb.wrapping_mul(inv_a).wrapping_add(ROUND);
                let blended_rb = (t_rb.wrapping_add((t_rb >> 8) & RB_MASK) >> 8) & RB_MASK;
                let dst_xg = (d >> 8) & RB_MASK;
                let t_xg = dst_xg.wrapping_mul(inv_a).wrapping_add(ROUND);
                let blended_xg = (t_xg.wrapping_add((t_xg >> 8) & RB_MASK) >> 8) & RB_MASK;
                dst[d_off + x] = s.wrapping_add(blended_rb).wrapping_add(blended_xg << 8) | OPAQUE;
            }
        }
    }
}

/// Nearest-neighbour scaled blit with alpha blending.
///
/// Samples `src_w × src_h` pixels from the source (at `src_pitch`)
/// and writes them into `dst_w × dst_h` pixels of the destination
/// (at `dst_pitch`), using integer fixed-point UV mapping.
///
/// Every sampled pixel is alpha-blended (premultiplied src-over).
/// Fully transparent pixels (α=0) are skipped; fully opaque (α=0xFF)
/// are direct-copied.
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
    if dst_w == 0 || dst_h == 0 || src_w == 0 || src_h == 0 {
        return;
    }

    // 16.16 fixed-point step for UV mapping.
    let sx = ((src_w as u64) << 16) / dst_w as u64;
    let sy = ((src_h as u64) << 16) / dst_h as u64;

    for dy in 0..dst_h {
        let src_y = (((dy as u64) * sy) >> 16) as usize;
        let s_row = src_y * src_pitch;
        let d_row = dy * dst_pitch;

        for dx in 0..dst_w {
            let src_x = (((dx as u64) * sx) >> 16) as usize;
            let s = src[s_row + src_x];
            let sa = s >> 24;

            if sa == 0 {
                continue;
            }
            if sa == 0xFF {
                dst[d_row + dx] = s;
                continue;
            }

            // Alpha blend (premultiplied src-over)
            let d = dst[d_row + dx];
            let inv_a = 255 - sa;
            let dst_rb = d & RB_MASK;
            let t_rb = dst_rb.wrapping_mul(inv_a).wrapping_add(ROUND);
            let blended_rb = (t_rb.wrapping_add((t_rb >> 8) & RB_MASK) >> 8) & RB_MASK;
            let dst_xg = (d >> 8) & RB_MASK;
            let t_xg = dst_xg.wrapping_mul(inv_a).wrapping_add(ROUND);
            let blended_xg = (t_xg.wrapping_add((t_xg >> 8) & RB_MASK) >> 8) & RB_MASK;
            dst[d_row + dx] = s.wrapping_add(blended_rb).wrapping_add(blended_xg << 8) | OPAQUE;
        }
    }
}
