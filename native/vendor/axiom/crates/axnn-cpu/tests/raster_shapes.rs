//! `RasterBackend` shape primitives against an independent scalar
//! reference: rounded-rectangle SDF coverage at pixel centres, ring =
//! outer − inner, table shadows, coverage tints, and masked blits.

use axnn::{
    Backend, DType, MAX_FALLOFF, RasterBackend, RoundedShape, TensorDesc, gaussian_falloff, shape,
};
use axnn_cpu::{CpuBackend, CpuError};

type Be = CpuBackend<4096>;

/// An owned XRGB framebuffer with a padded pitch, wrapped as an `[H, W]`
/// I32 view over `Vec<u8>` (malloc alignment ≥ 4 on every host).
struct Fb {
    bytes: Vec<u8>,
    w: usize,
    h: usize,
    pitch: usize,
    /// The initial pixel, which the row padding must keep.
    pad: u32,
}

impl Fb {
    fn filled(w: usize, h: usize, pixel: u32) -> Self {
        let pitch = w + 5;
        let bytes = (0..pitch * h).flat_map(|_| pixel.to_le_bytes()).collect();
        Self {
            bytes,
            w,
            h,
            pitch,
            pad: pixel,
        }
    }

    fn desc(&self) -> TensorDesc {
        TensorDesc {
            shape: shape![self.h, self.w],
            dtype: DType::I32,
            layout: axnn::Layout::from_raw(&[self.pitch * 4, 4], 0),
        }
    }

    fn view<'a>(&'a mut self, be: &Be) -> axnn::Tensor<axnn_cpu::CpuStorage<'a>> {
        let desc = self.desc();
        be.wrap_mut(&mut self.bytes, &desc).unwrap()
    }

    fn px(&self, x: usize, y: usize) -> u32 {
        let i = (y * self.pitch + x) * 4;
        u32::from_le_bytes(self.bytes[i..i + 4].try_into().unwrap())
    }

    /// Padding words past each row's width, which no op may touch.
    fn padding_intact(&self) -> bool {
        (0..self.h).all(|y| (self.w..self.pitch).all(|x| self.px(x, y) == self.pad))
    }
}

// ── Independent reference ──────────────────────────────────────────

/// Signed distance at pixel centre `(x, y)` with per-quadrant radii.
fn sdf(shape: RoundedShape, x: usize, y: usize) -> f32 {
    let [tl, tr, br, bl] = shape.clamped_radii().map(f32::from);
    let (w, h) = (f32::from(shape.w), f32::from(shape.h));
    let px = x as f32 + 0.5 - (shape.x as f32 + w * 0.5);
    let py = y as f32 + 0.5 - (shape.y as f32 + h * 0.5);
    let r = match (py < 0.0, px < 0.0) {
        (true, true) => tl,
        (true, false) => tr,
        (false, false) => br,
        (false, true) => bl,
    };
    let qx = px.abs() - (w * 0.5 - r);
    let qy = py.abs() - (h * 0.5 - r);
    let (ox, oy) = (qx.max(0.0), qy.max(0.0));
    (ox * ox + oy * oy).sqrt() + qx.max(qy).min(0.0) - r
}

fn coverage(shape: RoundedShape, x: usize, y: usize) -> f32 {
    (0.5 - sdf(shape, x, y)).clamp(0.0, 1.0)
}

fn ring_coverage(shape: RoundedShape, width: u16, x: usize, y: usize) -> f32 {
    let inner = shape
        .inset(width)
        .map_or(0.0, |inner| coverage(inner, x, y));
    (coverage(shape, x, y) - inner).max(0.0)
}

fn table_coverage(shape: RoundedShape, table: &[u8], x: usize, y: usize) -> f32 {
    let half = (table.len() / 2) as f32;
    let t = (sdf(shape, x, y) + half).clamp(0.0, (table.len() - 1) as f32);
    let i = (t as usize).min(table.len() - 2);
    let f = t - i as f32;
    (f32::from(table[i]) * (1.0 - f) + f32::from(table[i + 1]) * f) / 255.0
}

/// `round(c · f / 255)`, the exact per-channel scale the kernels use.
fn mul255(c: u32, f: u32) -> u32 {
    (c * f + 127) / 255
}

fn premul(color: u32, opacity: u32) -> u32 {
    if opacity == 255 {
        return color | 0xff00_0000;
    }
    (opacity << 24)
        | (mul255((color >> 16) & 0xff, opacity) << 16)
        | (mul255((color >> 8) & 0xff, opacity) << 8)
        | mul255(color & 0xff, opacity)
}

fn scale(pixel: u32, f: u32) -> u32 {
    (mul255(pixel >> 24, f) << 24)
        | (mul255((pixel >> 16) & 0xff, f) << 16)
        | (mul255((pixel >> 8) & 0xff, f) << 8)
        | mul255(pixel & 0xff, f)
}

fn over(s: u32, d: u32) -> u32 {
    let inv = 255 - (s >> 24);
    let ch = |shift: u32| ((s >> shift) & 0xff) + mul255((d >> shift) & 0xff, inv);
    0xff00_0000 | (ch(16) << 16) | (ch(8) << 8) | ch(0)
}

fn channel_close(got: u32, want: u32, tolerance: u32) -> bool {
    (0..3).all(|c| {
        let shift = c * 8;
        ((got >> shift) & 0xff).abs_diff((want >> shift) & 0xff) <= tolerance
    })
}

/// Assert every pixel of `fb` matches `expect(x, y)` within `tolerance`
/// levels per channel and that the padding survived.
fn check(fb: &Fb, tolerance: u32, expect: impl Fn(usize, usize) -> u32, what: &str) {
    for y in 0..fb.h {
        for x in 0..fb.w {
            let (got, want) = (fb.px(x, y), expect(x, y));
            assert!(
                channel_close(got, want, tolerance),
                "{what}: pixel ({x}, {y}) got {got:#010x} want {want:#010x}"
            );
        }
    }
    assert!(fb.padding_intact(), "{what}: row padding was written");
}

// ── rounded_fill ───────────────────────────────────────────────────

#[test]
fn rounded_fill_matches_the_sdf_reference_for_every_radius() {
    let be = Be::new();
    for radius in 0..=16u16 {
        let shape = RoundedShape::uniform(7, 5, 36, 30, radius);
        let mut fb = Fb::filled(52, 44, 0xff00_0000);
        be.rounded_fill(&mut fb.view(&be), 0xffff_ffff, shape)
            .unwrap();
        check(
            &fb,
            1,
            |x, y| {
                let c = (coverage(shape, x, y) * 255.0 + 0.5) as u32;
                0xff00_0000 | c << 16 | c << 8 | c
            },
            &format!("radius {radius}"),
        );
    }
}

#[test]
fn rounded_fill_honours_non_uniform_and_oversized_radii() {
    let be = Be::new();
    for radii in [[0, 4, 9, 16], [16, 0, 0, 16], [40, 1, 3, 7], [6, 6, 6, 6]] {
        let shape = RoundedShape {
            x: 3,
            y: 2,
            w: 33,
            h: 27,
            radii,
        };
        let mut fb = Fb::filled(40, 32, 0xff00_0000);
        be.rounded_fill(&mut fb.view(&be), 0xffff_ffff, shape)
            .unwrap();
        check(
            &fb,
            1,
            |x, y| {
                let c = (coverage(shape, x, y) * 255.0 + 0.5) as u32;
                0xff00_0000 | c << 16 | c << 8 | c
            },
            &format!("radii {radii:?}"),
        );
    }
}

#[test]
fn translucent_rounded_fill_blends_exactly_like_the_integer_reference() {
    let be = Be::new();
    let shape = RoundedShape::uniform(4, 3, 24, 20, 6);
    let pixel = premul(0x0040_c0ff, 160);
    let mut fb = Fb::filled(32, 26, 0xff00_0000);
    for y in 0..fb.h {
        for x in 0..fb.w {
            let i = (y * fb.pitch + x) * 4;
            let v = 0xff00_0000 | ((x * 7 + y * 13) as u32 * 0x0001_0203) & 0x00ff_ffff;
            fb.bytes[i..i + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    let before = fb.bytes.clone();
    be.rounded_fill(&mut fb.view(&be), pixel, shape).unwrap();
    let old = |x: usize, y: usize| {
        let i = (y * fb.pitch + x) * 4;
        u32::from_le_bytes(before[i..i + 4].try_into().unwrap())
    };
    check(
        &fb,
        1,
        |x, y| {
            let c = (coverage(shape, x, y) * 255.0 + 0.5) as u32;
            if c == 0 {
                old(x, y)
            } else {
                over(scale(pixel, c), old(x, y))
            }
        },
        "translucent",
    );
}

#[test]
fn rounded_fill_clips_to_the_view_and_ignores_degenerate_shapes() {
    let be = Be::new();
    // Partially off every edge.
    let shape = RoundedShape::uniform(-9, -7, 40, 36, 8);
    let mut fb = Fb::filled(24, 20, 0xff00_0000);
    be.rounded_fill(&mut fb.view(&be), 0xffff_ffff, shape)
        .unwrap();
    check(
        &fb,
        1,
        |x, y| {
            let c = (coverage(shape, x, y) * 255.0 + 0.5) as u32;
            0xff00_0000 | c << 16 | c << 8 | c
        },
        "clipped",
    );

    // Entirely off-view, empty, and transparent: nothing changes.
    for (shape, pixel) in [
        (RoundedShape::uniform(24, 0, 10, 10, 3), 0xffff_ffffu32),
        (RoundedShape::uniform(-10, 0, 10, 10, 3), 0xffff_ffff),
        (RoundedShape::uniform(0, 20, 10, 10, 3), 0xffff_ffff),
        (
            RoundedShape::uniform(i32::MIN, i32::MAX, 10, 10, 3),
            0xffff_ffff,
        ),
        (RoundedShape::uniform(2, 2, 0, 10, 3), 0xffff_ffff),
        (RoundedShape::uniform(2, 2, 10, 0, 3), 0xffff_ffff),
        (RoundedShape::uniform(2, 2, 10, 10, 3), 0x0000_0000),
    ] {
        let mut fb = Fb::filled(24, 20, 0xff00_0000);
        be.rounded_fill(&mut fb.view(&be), pixel, shape).unwrap();
        check(&fb, 0, |_, _| 0xff00_0000, &format!("no-op {shape:?}"));
    }
}

#[test]
fn square_rounded_fill_is_exactly_fill_rect() {
    let be = Be::new();
    let mut a = Fb::filled(64, 48, 0xff10_2030);
    let mut b = Fb::filled(64, 48, 0xff10_2030);
    be.rounded_fill(
        &mut a.view(&be),
        0xff88_4422,
        RoundedShape::rect(5, 7, 40, 30),
    )
    .unwrap();
    let mut view = b.view(&be);
    let mut sub = view.slice(0, 7, 30).unwrap().slice(1, 5, 40).unwrap();
    be.fill_rect(&mut sub, 0xff88_4422).unwrap();
    drop(view);
    assert_eq!(a.bytes, b.bytes);
}

#[test]
fn rounded_fill_rejects_non_surface_views() {
    let be = Be::new();
    let mut plane = be.zeros(&TensorDesc::new(shape![4, 4], DType::U8)).unwrap();
    assert_eq!(
        be.rounded_fill(&mut plane, 0xffff_ffff, RoundedShape::rect(0, 0, 2, 2)),
        Err(CpuError::UnsupportedOp {
            op: "rounded_fill",
            dtype: DType::U8
        })
    );
}

// ── border_ring ────────────────────────────────────────────────────

#[test]
fn border_ring_coverage_is_outer_minus_inner() {
    let be = Be::new();
    for (radii, width) in [
        ([6, 6, 6, 6], 1),
        ([0, 4, 9, 16], 1),
        ([12, 3, 0, 7], 3),
        ([2, 2, 2, 2], 5),
        ([0, 0, 0, 0], 1),
        ([16, 16, 16, 16], 2),
    ] {
        let shape = RoundedShape {
            x: 4,
            y: 3,
            w: 34,
            h: 28,
            radii,
        };
        let mut fb = Fb::filled(44, 36, 0xff00_0000);
        be.border_ring(&mut fb.view(&be), 0xffff_ffff, shape, width)
            .unwrap();
        check(
            &fb,
            1,
            |x, y| {
                let c = (ring_coverage(shape, width, x, y) * 255.0 + 0.5) as u32;
                0xff00_0000 | c << 16 | c << 8 | c
            },
            &format!("ring radii {radii:?} width {width}"),
        );
    }
}

#[test]
fn border_ring_degenerates_to_a_fill_and_zero_width_is_a_no_op() {
    let be = Be::new();
    let shape = RoundedShape::uniform(3, 2, 20, 14, 5);
    let mut ring = Fb::filled(28, 20, 0xff00_0000);
    let mut fill = Fb::filled(28, 20, 0xff00_0000);
    be.border_ring(&mut ring.view(&be), 0xffff_ffff, shape, 7)
        .unwrap();
    be.rounded_fill(&mut fill.view(&be), 0xffff_ffff, shape)
        .unwrap();
    assert_eq!(ring.bytes, fill.bytes);

    let mut none = Fb::filled(28, 20, 0xff00_0000);
    be.border_ring(&mut none.view(&be), 0xffff_ffff, shape, 0)
        .unwrap();
    check(&none, 0, |_, _| 0xff00_0000, "zero width");

    // Clipped ring: only the visible part of the shape is stroked.
    let clipped = RoundedShape::uniform(-6, -5, 30, 24, 8);
    let mut fb = Fb::filled(20, 16, 0xff00_0000);
    be.border_ring(&mut fb.view(&be), 0xffff_ffff, clipped, 2)
        .unwrap();
    check(
        &fb,
        1,
        |x, y| {
            let c = (ring_coverage(clipped, 2, x, y) * 255.0 + 0.5) as u32;
            0xff00_0000 | c << 16 | c << 8 | c
        },
        "clipped ring",
    );
}

// ── shadow_fill ────────────────────────────────────────────────────

#[test]
fn shadow_fill_samples_the_falloff_table_at_the_signed_distance() {
    let be = Be::new();
    let tables: [&[u8]; 4] = [
        &gaussian_falloff::<32>(8),
        &gaussian_falloff::<16>(2),
        &gaussian_falloff::<8>(0),
        &[255, 200, 150, 100, 50, 0],
    ];
    for (shape, table) in [
        (RoundedShape::uniform(14, 12, 30, 24, 6), tables[0]),
        (
            RoundedShape {
                x: 10,
                y: 9,
                w: 33,
                h: 29,
                radii: [0, 5, 11, 2],
            },
            tables[1],
        ),
        (RoundedShape::uniform(-4, 30, 40, 40, 20), tables[2]),
        (RoundedShape::uniform(20, -3, 12, 50, 3), tables[3]),
    ] {
        let mut fb = Fb::filled(58, 50, 0xff00_0000);
        be.shadow_fill(&mut fb.view(&be), 0xffff_ffff, shape, table)
            .unwrap();
        check(
            &fb,
            1,
            |x, y| {
                let c = (table_coverage(shape, table, x, y) * 255.0 + 0.5) as u32;
                0xff00_0000 | c << 16 | c << 8 | c
            },
            &format!("shadow {shape:?} table {}", table.len()),
        );
    }
}

#[test]
fn shadow_fill_bounds_its_halo_and_validates_the_table() {
    let be = Be::new();
    let table = gaussian_falloff::<16>(4);
    let shape = RoundedShape::uniform(10, 10, 30, 30, 3);
    let mut fb = Fb::filled(50, 50, 0xff00_0000);
    be.shadow_fill(&mut fb.view(&be), 0xffff_ffff, shape, &table)
        .unwrap();
    // Nothing beyond len − len/2 = 8 px outside the box.
    for y in 0..50 {
        for x in 0..50 {
            let outside = x < 2 || x >= 48 || y < 2 || y >= 48;
            assert!(
                !outside || fb.px(x, y) == 0xff00_0000,
                "halo escaped at ({x}, {y})"
            );
        }
    }
    // 15 px inside every edge is past the table's reach: falloff[0].
    assert_eq!(fb.px(25, 25), 0xffff_ffff, "deep interior is falloff[0]");
    assert_eq!(table[0], 255);

    // A shape just off the view still casts its halo in.
    let off = RoundedShape::uniform(-10, 5, 8, 8, 0);
    let mut fb = Fb::filled(20, 20, 0xff00_0000);
    be.shadow_fill(&mut fb.view(&be), 0xffff_ffff, off, &table)
        .unwrap();
    assert_ne!(fb.px(0, 9), 0xff00_0000);

    let mut fb = Fb::filled(8, 8, 0xff00_0000);
    assert_eq!(
        be.shadow_fill(&mut fb.view(&be), 0xffff_ffff, shape, &[]),
        Err(CpuError::ShapeMismatch { op: "shadow_fill" })
    );
    let too_long = [0u8; MAX_FALLOFF + 1];
    assert_eq!(
        be.shadow_fill(&mut fb.view(&be), 0xffff_ffff, shape, &too_long),
        Err(CpuError::ShapeMismatch { op: "shadow_fill" })
    );
}

// ── coverage_blend / blit_masked ───────────────────────────────────

fn glyph(w: usize, h: usize) -> Vec<u8> {
    (0..w * h)
        .map(|i| match i % 7 {
            0 => 0,
            1 => 255,
            2 => 1,
            _ => (i * 53) as u8,
        })
        .collect()
}

#[test]
fn coverage_blend_matches_the_scalar_reference_for_every_mask_dtype() {
    let be = Be::new();
    let (w, h) = (21, 9);
    let cov8 = glyph(w, h);
    let cov16: Vec<u16> = cov8.iter().map(|&c| u16::from(c) * 257).collect();
    let cov32: Vec<u32> = cov8
        .iter()
        .map(|&c| (u32::from(c) << 24) | 0x0012_3456)
        .collect();
    let cov16_bytes: Vec<u8> = cov16.iter().flat_map(|c| c.to_le_bytes()).collect();
    let cov32_bytes: Vec<u8> = cov32.iter().flat_map(|c| c.to_le_bytes()).collect();
    let (color, opacity) = (0x1234_5678u32, 200u8);
    let pixel = premul(color, u32::from(opacity));

    for (dtype, bytes) in [
        (DType::U8, cov8.clone()),
        (DType::I16, cov16_bytes),
        (DType::I32, cov32_bytes),
    ] {
        let mask = be
            .wrap(&bytes, &TensorDesc::new(shape![h, w], dtype))
            .unwrap();
        let mut fb = Fb::filled(w, h, 0xff40_8020);
        be.coverage_blend(&mask, color, opacity, &mut fb.view(&be))
            .unwrap();
        let tolerance = if dtype == DType::I16 { 1 } else { 0 };
        check(
            &fb,
            tolerance,
            |x, y| {
                let c = u32::from(cov8[y * w + x]);
                if c == 0 {
                    0xff40_8020
                } else {
                    over(scale(pixel, c), 0xff40_8020)
                }
            },
            &format!("{dtype:?}"),
        );
    }
}

#[test]
fn coverage_blend_reads_pitched_atlas_views_and_rejects_bad_operands() {
    let be = Be::new();
    // A 40×16 U8 atlas; the glyph is the 13×9 window at (20, 4).
    let atlas: Vec<u8> = (0..40 * 16).map(|i| (i * 29) as u8).collect();
    let atlas_t = be
        .wrap(&atlas, &TensorDesc::new(shape![16, 40], DType::U8))
        .unwrap();
    let glyph_view = atlas_t.slice(0, 4, 9).unwrap().slice(1, 20, 13).unwrap();
    let pixel = premul(0x00ff_ffff, 255);

    let mut fb = Fb::filled(30, 20, 0xff00_0000);
    {
        let mut view = fb.view(&be);
        let mut dst = view.slice(0, 6, 9).unwrap().slice(1, 3, 13).unwrap();
        be.coverage_blend(&glyph_view, 0x00ff_ffff, 255, &mut dst)
            .unwrap();
    }
    check(
        &fb,
        0,
        |x, y| {
            if (3..16).contains(&x) && (6..15).contains(&y) {
                let c = u32::from(atlas[(y - 6 + 4) * 40 + (x - 3 + 20)]);
                if c == 0 {
                    0xff00_0000
                } else {
                    over(scale(pixel, c), 0xff00_0000)
                }
            } else {
                0xff00_0000
            }
        },
        "atlas window",
    );

    let f32_mask = be
        .zeros(&TensorDesc::new(shape![4, 4], DType::F32))
        .unwrap();
    let mut fb = Fb::filled(4, 4, 0);
    assert_eq!(
        be.coverage_blend(&f32_mask, 0, 255, &mut fb.view(&be)),
        Err(CpuError::UnsupportedOp {
            op: "coverage_blend",
            dtype: DType::F32
        })
    );
    let wrong = be.zeros(&TensorDesc::new(shape![4, 5], DType::U8)).unwrap();
    assert_eq!(
        be.coverage_blend(&wrong, 0, 255, &mut fb.view(&be)),
        Err(CpuError::ShapeMismatch {
            op: "coverage_blend"
        })
    );
}

#[test]
fn blit_masked_scales_each_source_pixel_by_its_mask_byte() {
    let be = Be::new();
    let (w, h) = (19, 7);
    let mask = glyph(w, h);
    let src: Vec<u32> = (0..w * h)
        .map(|i| match i % 3 {
            0 => 0xff00_0000 | (i as u32).wrapping_mul(0x0307_0b0d) & 0x00ff_ffff,
            1 => premul(0x0080_4020, 128),
            _ => 0,
        })
        .collect();
    let src_bytes: Vec<u8> = src.iter().flat_map(|p| p.to_le_bytes()).collect();
    let src_t = be
        .wrap(&src_bytes, &TensorDesc::new(shape![h, w], DType::I32))
        .unwrap();
    let mask_t = be
        .wrap(&mask, &TensorDesc::new(shape![h, w], DType::U8))
        .unwrap();
    let mut fb = Fb::filled(w, h, 0xff20_3040);
    be.blit_masked(&src_t, &mask_t, &mut fb.view(&be)).unwrap();
    check(
        &fb,
        0,
        |x, y| {
            let s = scale(src[y * w + x], u32::from(mask[y * w + x]));
            if s >> 24 == 0 {
                0xff20_3040
            } else {
                over(s, 0xff20_3040)
            }
        },
        "blit_masked",
    );

    let mut aliased = be
        .zeros(&TensorDesc::new(shape![h, w], DType::I32))
        .unwrap();
    let alias = aliased;
    assert_eq!(
        be.blit_masked(&alias, &mask_t, &mut aliased),
        Err(CpuError::AliasedStorage { op: "blit_masked" })
    );
}
