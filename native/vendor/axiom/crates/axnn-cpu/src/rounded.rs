//! Rounded-silhouette geometry the raster kernels share.
//!
//! The partition (corner boxes, row bands), the radius clamp and the
//! shadow tables live in [`axnn::raster`] beside the vocabulary every
//! lowering uses, so the scene lowering and the kernels never disagree
//! about a corner; this module adds the two pieces that need arithmetic
//! the kernels own: the backend shape a rect lowers to, and the analytic
//! coverage of one pixel inside a corner box, evaluating the same
//! rounded-rectangle signed-distance rule [`shapes::rounded_fill`](crate::shapes)
//! uses (pixel-centre SDF, `clamp(0.5 − d)`, 8-bit rounding,
//! `axmath::sqrt_f32`). A surface clipped to its rounded silhouette
//! therefore lands on exactly the pixels a rounded fill of the same rect
//! and radii would.

use axnn::{Rect, RoundedShape};

pub use axnn::raster::{
    CornerBox, MASK_SIDE, Radii, bands, clamped, corner_boxes, shadow_falloff, shadow_halo,
};

/// `rect` with `radii` as a backend shape relative to the view origin
/// `(ox, oy)`.
#[inline]
pub fn shape_at(rect: Rect, radii: Radii, ox: i32, oy: i32) -> RoundedShape {
    RoundedShape {
        x: rect.x - ox,
        y: rect.y - oy,
        w: rect.w,
        h: rect.h,
        radii,
    }
}

/// Coverage (0..=255) of the screen pixel `(px, py)` by the silhouette of
/// `rect`, evaluated with the corner arc of `radius` — the pixel must lie in
/// that corner's box, which the callers guarantee.
///
/// Mirrors the backend kernel operand for operand: every intermediate is a
/// half- or quarter-integer well inside `f32`'s exact range, so the value
/// is identical whichever origin the backend measured from.
#[inline]
pub fn corner_coverage(rect: Rect, radius: u16, px: i32, py: i32) -> u8 {
    let half_w = f32::from(rect.w) * 0.5;
    let half_h = f32::from(rect.h) * 0.5;
    let cx = rect.x as f32 + half_w;
    let cy = rect.y as f32 + half_h;
    let r = f32::from(radius);
    let qx = ((px as f32 + 0.5) - cx).abs() - (half_w - r);
    let qy = ((py as f32 + 0.5) - cy).abs() - (half_h - r);
    let ox = qx.max(0.0);
    let oy = qy.max(0.0);
    let len = axmath::sqrt_f32(ox * ox + oy * oy);
    let inner = qx.max(qy).min(0.0);
    let d = len + inner - r;
    ((0.5 - d).clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_is_full_inside_zero_outside_and_partial_on_the_arc() {
        let rect = Rect {
            x: 10,
            y: 20,
            w: 40,
            h: 30,
        };
        assert_eq!(corner_coverage(rect, 8, 17, 27), 255);
        assert_eq!(corner_coverage(rect, 8, 10, 20), 0);
        let edge = corner_coverage(rect, 8, 12, 22);
        assert!(edge > 0 && edge < 255, "{edge}");
        assert_eq!(corner_coverage(rect, 0, 10, 20), 255);
    }
}
