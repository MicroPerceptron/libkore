//! Pointer mapping for a fixed-resolution image presented in a host viewport.
//!
//! A host supplies the destination rectangle in the same coordinate space as
//! its pointer samples. The image resolution remains independent of window
//! size, desktop scaling, and the host's choice of presentation API.

use crate::{Extent2d, PixelRect};

/// Maps an image stretched across an explicit destination rectangle.
///
/// For aspect-preserving presentation, supply the actual image rectangle after
/// letterboxing. Coordinates outside that rectangle are discarded, not clamped
/// to an unrelated edge pixel. The host remains responsible for presenting with
/// the same rectangle and for translating its input into destination coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageViewport {
    image: Extent2d,
    destination: PixelRect,
}

impl ImageViewport {
    /// Creates a mapping from a destination rectangle to the complete image.
    pub const fn new(image: Extent2d, destination: PixelRect) -> Self {
        Self { image, destination }
    }

    /// Maps a complete destination extent to the complete image, stretching
    /// independently along each axis.
    pub const fn stretch(image: Extent2d, destination: Extent2d) -> Self {
        Self::new(image, PixelRect::full(destination))
    }

    /// Returns the fixed image resolution.
    pub const fn image(self) -> Extent2d {
        self.image
    }

    /// Returns the rectangle occupied by the presented image.
    pub const fn destination(self) -> PixelRect {
        self.destination
    }

    /// Maps a finite pointer sample to continuous image coordinates.
    ///
    /// The top and left image edges are zero. The bottom and right edges are
    /// excluded, so truncating a returned coordinate selects an in-bounds pixel.
    /// Subpixel precision is preserved for picking and motion; no extra initial
    /// window scale or device pixel ratio should be applied to the result.
    pub fn map_pointer(self, x: f32, y: f32) -> Option<(f32, f32)> {
        Some((
            map_component(
                x,
                self.destination.x(),
                self.destination.width(),
                self.image.width(),
            )?,
            map_component(
                y,
                self.destination.y(),
                self.destination.height(),
                self.image.height(),
            )?,
        ))
    }

    fn contains_image_position(self, x: f32, y: f32) -> bool {
        x.is_finite()
            && y.is_finite()
            && x >= 0.0
            && y >= 0.0
            && f64::from(x) < f64::from(self.image.width())
            && f64::from(y) < f64::from(self.image.height())
    }
}

fn map_component(value: f32, origin: u32, destination: u32, image: u32) -> Option<f32> {
    if !value.is_finite() {
        return None;
    }
    // Compute the ratio in f64 so even large valid u32 extents retain their
    // exact bounds. A final f32 rounding must not manufacture an edge pixel.
    let local = f64::from(value) - f64::from(origin);
    if !(0.0..f64::from(destination)).contains(&local) {
        return None;
    }
    let mapped = (local * f64::from(image) / f64::from(destination)) as f32;
    Some(if f64::from(mapped) >= f64::from(image) {
        mapped.next_down()
    } else {
        mapped
    })
}

/// Consecutive drag motion in image coordinates, without a resize jump.
///
/// The caller maps the pointer once with [`ImageViewport::map_pointer`] and
/// uses that position for both picking and this state. A press starts a fresh
/// baseline. Release, missing input, a missing viewport, invalid image
/// coordinates, or any change to the image/destination geometry resets it.
#[derive(Debug, Default)]
pub struct PointerDrag {
    previous: Option<(ImageViewport, (f32, f32))>,
}

impl PointerDrag {
    /// Creates a drag tracker without a previous sample.
    pub const fn new() -> Self {
        Self { previous: None }
    }

    /// Clears the baseline, for example when a surface loses input focus.
    pub fn reset(&mut self) {
        self.previous = None;
    }

    /// Returns motion only between consecutive pressed samples using the same
    /// viewport. `position` is already mapped into image coordinates.
    ///
    /// Call this once per input sample, including releases and input loss. The
    /// first valid pressed sample after any reset returns `None` and establishes
    /// the baseline; it never orbits by motion accumulated before the press.
    pub fn update(
        &mut self,
        viewport: Option<ImageViewport>,
        position: Option<(f32, f32)>,
        pressed: bool,
    ) -> Option<(f32, f32)> {
        let previous = self.previous.take();
        if !pressed {
            return None;
        }
        let viewport = viewport?;
        let (x, y) = position?;
        if !viewport.contains_image_position(x, y) {
            return None;
        }
        self.previous = Some((viewport, (x, y)));
        let (prior_viewport, (px, py)) = previous?;
        (prior_viewport == viewport).then_some((x - px, y - py))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extent(width: u32, height: u32) -> Extent2d {
        Extent2d::new(width, height).unwrap()
    }

    #[test]
    fn stretched_mapping_tracks_non_square_resizes_and_initial_scale_two() {
        let image = extent(640, 360);
        for destination in [
            extent(640, 360),
            extent(1280, 720),
            extent(900, 1000),
            extent(320, 90),
        ] {
            let viewport = ImageViewport::stretch(image, destination);
            let width = destination.width() as f32;
            let height = destination.height() as f32;
            assert_eq!(viewport.image(), image);
            assert_eq!(viewport.destination(), PixelRect::full(destination));
            assert_eq!(viewport.map_pointer(0.0, 0.0), Some((0.0, 0.0)));
            assert_eq!(
                viewport.map_pointer(width * 0.5, height * 0.5),
                Some((320.0, 180.0))
            );
            assert_eq!(
                viewport.map_pointer(width * 0.75, height * 0.25),
                Some((480.0, 90.0))
            );
            assert_eq!(viewport.map_pointer(width, 0.0), None);
            assert_eq!(viewport.map_pointer(0.0, height), None);
            let last = viewport
                .map_pointer(width.next_down(), height.next_down())
                .unwrap();
            assert!(last.0 < 640.0 && last.0 as u32 == 639);
            assert!(last.1 < 360.0 && last.1 as u32 == 359);
        }
        let scale_two = ImageViewport::stretch(image, extent(1280, 720));
        assert_eq!(scale_two.map_pointer(1279.0, 719.0), Some((639.5, 359.5)));
    }

    #[test]
    fn offset_image_rect_discards_letterboxing_and_invalid_samples() {
        let viewport =
            ImageViewport::new(extent(640, 360), PixelRect::new(40, 80, 1280, 720).unwrap());
        assert_eq!(viewport.map_pointer(40.0, 80.0), Some((0.0, 0.0)));
        assert_eq!(viewport.map_pointer(680.0, 440.0), Some((320.0, 180.0)));
        assert_eq!(viewport.map_pointer(1319.0, 799.0), Some((639.5, 359.5)));
        for (x, y) in [
            (39.0, 80.0),
            (40.0, 79.0),
            (1320.0, 80.0),
            (40.0, 800.0),
            (-1.0, -1.0),
            (f32::NAN, 100.0),
            (100.0, f32::NAN),
            (f32::INFINITY, 100.0),
            (100.0, f32::NEG_INFINITY),
        ] {
            assert_eq!(viewport.map_pointer(x, y), None);
        }
    }

    #[test]
    fn zero_viewports_are_rejected_and_large_extents_keep_exclusive_bounds() {
        for (width, height) in [(0, 1), (1, 0), (0, 0)] {
            assert!(Extent2d::new(width, height).is_err());
            assert!(PixelRect::new(0, 0, width, height).is_err());
        }
        let viewport = ImageViewport::stretch(extent(u32::MAX, 1), extent(1, 1));
        let (x, y) = viewport.map_pointer(1.0f32.next_down(), 0.5).unwrap();
        assert!(f64::from(x) < f64::from(u32::MAX));
        assert_eq!(y, 0.5);
        assert_eq!(viewport.map_pointer(1.0, 0.5), None);
    }

    #[test]
    fn drag_uses_image_coordinates_and_rebases_on_every_geometry_change() {
        let first = ImageViewport::stretch(extent(640, 360), extent(1280, 720));
        let resized = ImageViewport::stretch(extent(640, 360), extent(900, 1000));
        let moved = ImageViewport::new(extent(640, 360), PixelRect::new(5, 10, 900, 1000).unwrap());
        let new_image = ImageViewport::new(extent(320, 180), moved.destination());
        let mut drag = PointerDrag::new();
        assert_eq!(
            drag.update(Some(first), first.map_pointer(640.0, 360.0), true),
            None
        );
        assert_eq!(
            drag.update(Some(first), first.map_pointer(660.0, 352.0), true),
            Some((10.0, -4.0))
        );
        for viewport in [resized, moved, new_image] {
            assert_eq!(drag.update(Some(viewport), Some((100.0, 60.0)), true), None);
            assert_eq!(
                drag.update(Some(viewport), Some((102.0, 65.0)), true),
                Some((2.0, 5.0))
            );
        }
    }

    #[test]
    fn drag_resets_on_release_focus_loss_zero_extent_and_invalid_input() {
        let viewport = ImageViewport::stretch(extent(640, 360), extent(1280, 720));
        let mut drag = PointerDrag::new();
        for (geometry, position, pressed) in [
            (Some(viewport), Some((40.0, 40.0)), false),
            (Some(viewport), None, true),
            (None, Some((40.0, 40.0)), true),
            (Some(viewport), Some((f32::NAN, 1.0)), true),
            (Some(viewport), Some((1.0, f32::INFINITY)), true),
            (Some(viewport), Some((-1.0, 1.0)), true),
            (Some(viewport), Some((640.0, 1.0)), true),
        ] {
            assert_eq!(drag.update(Some(viewport), Some((10.0, 20.0)), true), None);
            assert_eq!(
                drag.update(Some(viewport), Some((12.0, 25.0)), true),
                Some((2.0, 5.0))
            );
            assert_eq!(drag.update(geometry, position, pressed), None);
            assert_eq!(drag.update(Some(viewport), Some((70.0, 80.0)), true), None);
            drag.reset();
        }
    }
}
