//! Design-space to physical-backing transforms.
//!
//! UI geometry is authored in a stable design space, then transformed once
//! into the actual pixel backing. A contained viewport preserves aspect ratio
//! and centers any letterbox area. Text origins retain an eight-bit subpixel
//! phase so glyph coverage does not snap during this transform.

use crate::Rect;

/// A physical point split into an integer pixel and a `1/256`-pixel phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubpixelPoint {
    pub x: i32,
    pub y: i32,
    pub phase_x: u8,
    pub phase_y: u8,
}

/// A physical rectangle whose top-left origin retains subpixel phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubpixelRect {
    pub rect: Rect,
    pub phase_x: u8,
    pub phase_y: u8,
}

/// Uniform transform from a logical design canvas into a physical backing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    design_width: f32,
    design_height: f32,
    backing_width: u16,
    backing_height: u16,
    scale: f32,
    offset_x: f32,
    offset_y: f32,
}

impl Viewport {
    /// Fit the complete design canvas into the backing without distortion.
    ///
    /// Returns `None` when either canvas has a zero dimension.
    pub fn contain(
        design_width: u16,
        design_height: u16,
        backing_width: u16,
        backing_height: u16,
    ) -> Option<Self> {
        if design_width == 0 || design_height == 0 || backing_width == 0 || backing_height == 0 {
            return None;
        }
        let design_width = f32::from(design_width);
        let design_height = f32::from(design_height);
        let backing_width_f32 = f32::from(backing_width);
        let backing_height_f32 = f32::from(backing_height);
        let scale_x = backing_width_f32 / design_width;
        let scale_y = backing_height_f32 / design_height;
        let scale = if scale_x < scale_y { scale_x } else { scale_y };
        let offset_x = (backing_width_f32 - design_width * scale) * 0.5;
        let offset_y = (backing_height_f32 - design_height * scale) * 0.5;
        Some(Self {
            design_width,
            design_height,
            backing_width,
            backing_height,
            scale,
            offset_x,
            offset_y,
        })
    }

    #[inline]
    pub const fn backing_width(self) -> u16 {
        self.backing_width
    }

    #[inline]
    pub const fn backing_height(self) -> u16 {
        self.backing_height
    }

    #[inline]
    pub const fn scale(self) -> f32 {
        self.scale
    }

    #[inline]
    pub const fn offset(self) -> (f32, f32) {
        (self.offset_x, self.offset_y)
    }

    /// The entire physical backing, suitable as a scene root rectangle.
    #[inline]
    pub const fn root_rect(self) -> Rect {
        Rect {
            x: 0,
            y: 0,
            w: self.backing_width,
            h: self.backing_height,
        }
    }

    /// Transform a design-space length to physical pixels without rounding.
    #[inline]
    pub fn scale_px(self, value: f32) -> f32 {
        value * self.scale
    }

    /// Transform a design-space point while preserving its fractional phase.
    pub fn point(self, x: f32, y: f32) -> SubpixelPoint {
        let (x, phase_x) = split_subpixel(self.offset_x + x * self.scale);
        let (y, phase_y) = split_subpixel(self.offset_y + y * self.scale);
        SubpixelPoint {
            x,
            y,
            phase_x,
            phase_y,
        }
    }

    /// Transform a fill/layout rectangle by snapping its edges, not its size.
    ///
    /// Adjacent design edges therefore resolve to the same physical edge and
    /// cannot acquire one-pixel gaps from independent extent rounding.
    pub fn snapped_rect(self, x: f32, y: f32, width: f32, height: f32) -> Rect {
        let x0 = round_i32(self.offset_x + x * self.scale);
        let y0 = round_i32(self.offset_y + y * self.scale);
        let x1 = round_i32(self.offset_x + (x + width) * self.scale);
        let y1 = round_i32(self.offset_y + (y + height) * self.scale);
        Rect {
            x: x0,
            y: y0,
            w: extent_u16(x0, x1),
            h: extent_u16(y0, y1),
        }
    }

    /// Transform a text box while retaining its top-left subpixel phase.
    pub fn subpixel_rect(self, x: f32, y: f32, width: f32, height: f32) -> SubpixelRect {
        let point = self.point(x, y);
        let x1 = ceil_i32(self.offset_x + (x + width) * self.scale);
        let y1 = ceil_i32(self.offset_y + (y + height) * self.scale);
        SubpixelRect {
            rect: Rect {
                x: point.x,
                y: point.y,
                w: extent_u16(point.x, x1),
                h: extent_u16(point.y, y1),
            },
            phase_x: point.phase_x,
            phase_y: point.phase_y,
        }
    }

    /// Logical dimensions used to construct this transform.
    #[inline]
    pub const fn design_size(self) -> (f32, f32) {
        (self.design_width, self.design_height)
    }
}

fn extent_u16(start: i32, end: i32) -> u16 {
    let extent = end.saturating_sub(start).max(0);
    u16::try_from(extent).unwrap_or(u16::MAX)
}

fn floor_i32(value: f32) -> i32 {
    let truncated = value as i32;
    if (truncated as f32) > value {
        truncated - 1
    } else {
        truncated
    }
}

fn ceil_i32(value: f32) -> i32 {
    let truncated = value as i32;
    if (truncated as f32) < value {
        truncated + 1
    } else {
        truncated
    }
}

fn round_i32(value: f32) -> i32 {
    floor_i32(value + 0.5)
}

fn split_subpixel(value: f32) -> (i32, u8) {
    let mut whole = floor_i32(value);
    let mut phase = ((value - whole as f32) * 256.0 + 0.5) as u16;
    if phase == 256 {
        whole += 1;
        phase = 0;
    }
    (whole, phase as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contain_preserves_aspect_and_centers_letterbox() {
        let viewport = Viewport::contain(100, 100, 200, 100).unwrap();
        assert_eq!(viewport.scale(), 1.0);
        assert_eq!(viewport.offset(), (50.0, 0.0));
        assert_eq!(
            viewport.snapped_rect(0.0, 0.0, 100.0, 100.0),
            Rect {
                x: 50,
                y: 0,
                w: 100,
                h: 100,
            }
        );
    }

    #[test]
    fn default_boot_size_retains_quarter_pixel_origin() {
        let viewport = Viewport::contain(843, 530, 1265, 795).unwrap();
        assert_eq!(viewport.scale(), 1.5);
        assert_eq!(viewport.offset(), (0.25, 0.0));
        assert_eq!(
            viewport.point(64.0, 48.0),
            SubpixelPoint {
                x: 96,
                y: 72,
                phase_x: 64,
                phase_y: 0,
            }
        );
    }

    #[test]
    fn snapped_adjacent_rects_share_the_same_edge() {
        let viewport = Viewport::contain(3, 1, 10, 3).unwrap();
        let left = viewport.snapped_rect(0.0, 0.0, 1.0, 1.0);
        let right = viewport.snapped_rect(1.0, 0.0, 2.0, 1.0);
        assert_eq!(left.x + i32::from(left.w), right.x);
    }
}
