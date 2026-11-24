//! Cursor routing and composited-fallback damage.
//!
//! Cursor pixels remain outside the main `.ax` scene when a runtime exposes an
//! independent plane. Without one, the same resolved visual becomes an Axiom
//! overlay and only the old/new cursor rectangles damage the scene. Switching
//! routes is therefore lossless and does not leak backend policy upward.

use axgfx::{CursorHotspot, Extent2d, PixelRect};
use axinput::{CursorRequest, PointerPosition};

/// Built-in straight-alpha arrow; upload once or use as `.ax` constants.
/// Coverage is supersampled at compile time, never per pointer movement.
/// The silhouette and one-pixel outline are symmetric about the tip-to-stem
/// axis `(1, 2)`. The tip is centred on the hotspot pixel, not its corner.
pub const ARROW_WIDTH: usize = 24;
pub const ARROW_HEIGHT: usize = 32;
pub const ARROW_RGBA: [u8; ARROW_WIDTH * ARROW_HEIGHT * 4] = arrow_rgba();
pub const ARROW_HOTSPOT: CursorHotspot = CursorHotspot { x: 2, y: 1 };

/// Experimental orientation-neutral presentation. Its centre is the selection
/// point, not an enlarged activation area. No motion or hit-test policy changes.
pub const RING_WIDTH: usize = 25;
pub const RING_HEIGHT: usize = 25;
pub const RING_HOTSPOT: CursorHotspot = CursorHotspot { x: 12, y: 12 };
pub const RING_RGBA: [u8; RING_WIDTH * RING_HEIGHT * 4] = ring_rgba();

const fn ring_rgba() -> [u8; RING_WIDTH * RING_HEIGHT * 4] {
    let mut out = [0; RING_WIDTH * RING_HEIGHT * 4];
    let mut p = 0;
    while p < RING_WIDTH * RING_HEIGHT {
        let (mut coverage, mut light) = (0, 0);
        let mut sample = 0;
        while sample < 16 {
            let x = (p % RING_WIDTH) as i32 * 8 + (sample % 4) * 2 + 1 - 100;
            let y = (p / RING_WIDTH) as i32 * 8 + (sample / 4) * 2 + 1 - 100;
            let d = x * x + y * y;
            if d <= 18 * 18 || (d >= 52 * 52 && d <= 76 * 76) {
                coverage += 1;
                light += if d <= 10 * 10 || (d >= 58 * 58 && d <= 70 * 70) {
                    245
                } else {
                    24
                };
            }
            sample += 1;
        }
        if coverage > 0 {
            let color = (light / coverage) as u8;
            out[p * 4] = color;
            out[p * 4 + 1] = color;
            out[p * 4 + 2] = color;
            out[p * 4 + 3] = ((coverage * 255 + 8) / 16) as u8;
        }
        p += 1;
    }
    out
}

const ARROW_SUBPIXELS: i32 = 16;
const ARROW_HEAD: i32 = 40 * ARROW_SUBPIXELS;
const ARROW_TAIL: i32 = 65 * ARROW_SUBPIXELS;
const ARROW_STEM_HALF: i32 = 5 * ARROW_SUBPIXELS;

/// Ink in orthogonal arrow-local coordinates. With `(dx, dy)` measured from
/// the tip, `u = dx + 2*dy` and `v = 2*dx - dy`. Reflection about the major
/// axis is simply `v -> -v`; both the head and shaft use this same frame.
const fn arrow_ink(u: i32, v: i32) -> u8 {
    let v = v.abs();
    let inside = u >= 0
        && u <= ARROW_TAIL
        && ((u <= ARROW_HEAD && 2 * v <= u) || v <= ARROW_STEM_HALF && u >= ARROW_HEAD);
    if !inside {
        return 0;
    }

    // One outline from distance to the actual polygon, rather than a second
    // hand-drawn inset. The local basis scales squared distances by five.
    // Segment endpoints handle the concave head/stem joins consistently.
    const POINTS: [(i32, i32); 7] = [
        (0, 0),
        (ARROW_HEAD, -ARROW_HEAD / 2),
        (ARROW_HEAD, -ARROW_STEM_HALF),
        (ARROW_TAIL, -ARROW_STEM_HALF),
        (ARROW_TAIL, ARROW_STEM_HALF),
        (ARROW_HEAD, ARROW_STEM_HALF),
        (ARROW_HEAD, ARROW_HEAD / 2),
    ];
    const STROKE_SQUARED: i64 = 5 * (ARROW_SUBPIXELS * ARROW_SUBPIXELS) as i64;
    let mut i = 0;
    while i < POINTS.len() {
        let (a, b) = (POINTS[i], POINTS[(i + 1) % POINTS.len()]);
        let (dx, dy) = ((b.0 - a.0) as i64, (b.1 - a.1) as i64);
        let (px, py) = ((u - a.0) as i64, (v - a.1) as i64);
        let length_squared = dx * dx + dy * dy;
        let projection = px * dx + py * dy;
        let near = if projection <= 0 {
            px * px + py * py < STROKE_SQUARED
        } else if projection >= length_squared {
            let (qx, qy) = (px - dx, py - dy);
            qx * qx + qy * qy < STROKE_SQUARED
        } else {
            let cross = px * dy - py * dx;
            cross * cross < STROKE_SQUARED * length_squared
        };
        if near {
            return 24;
        }
        i += 1;
    }
    245
}

const fn arrow_rgba() -> [u8; ARROW_WIDTH * ARROW_HEIGHT * 4] {
    let mut out = [0; ARROW_WIDTH * ARROW_HEIGHT * 4];
    let mut p = 0;
    while p < ARROW_WIDTH * ARROW_HEIGHT {
        let mut coverage = 0;
        let mut light = 0;
        let mut sample = 0;
        // 8x8 regular subpixel coverage, entirely constant-evaluated.
        while sample < 64 {
            let dx = ((p % ARROW_WIDTH) as i32 - ARROW_HOTSPOT.x as i32) * ARROW_SUBPIXELS
                + (sample % 8) * 2
                + 1
                - ARROW_SUBPIXELS / 2;
            let dy = ((p / ARROW_WIDTH) as i32 - ARROW_HOTSPOT.y as i32) * ARROW_SUBPIXELS
                + (sample / 8) * 2
                + 1
                - ARROW_SUBPIXELS / 2;
            let ink = arrow_ink(dx + 2 * dy, 2 * dx - dy);
            if ink != 0 {
                coverage += 1;
                light += ink as i32;
            }
            sample += 1;
        }
        if coverage != 0 {
            let color = (light / coverage) as u8;
            out[p * 4] = color;
            out[p * 4 + 1] = color;
            out[p * 4 + 2] = color;
            out[p * 4 + 3] = ((coverage * 255 + 32) / 64) as u8;
        }
        p += 1;
    }
    out
}

/// Chosen presentation route for the resolved cursor visual.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum CursorRoute {
    /// Render the cursor as a final Axiom compositor overlay.
    #[default]
    Composited,
    /// Send image/position/visibility to an independent platform plane.
    HardwarePlane,
}

/// Resolved cursor theme metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CursorVisual {
    /// Cursor request whose icon the theme resolved.
    pub request: CursorRequest,
    /// Physical image extent at the target scale.
    pub extent: Extent2d,
    /// Hotspot within that image.
    pub hotspot: CursorHotspot,
}

/// Current composited cursor overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompositeCursor {
    /// Target-clipped cursor bounds.
    pub bounds: PixelRect,
    /// Resolved visual to lower as the final scene overlay.
    pub visual: CursorVisual,
}

/// At most two exact cursor-damage rectangles: the old and new bounds.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CursorDamage {
    rects: [Option<PixelRect>; 2],
}

impl CursorDamage {
    /// Returns the number of distinct rectangles.
    pub const fn len(self) -> usize {
        if self.rects[0].is_none() {
            0
        } else if self.rects[1].is_none() {
            1
        } else {
            2
        }
    }

    /// Whether no composited pixels changed.
    pub const fn is_empty(self) -> bool {
        self.rects[0].is_none()
    }

    /// Returns one damage rectangle.
    pub const fn get(self, index: usize) -> Option<PixelRect> {
        if index < self.rects.len() {
            self.rects[index]
        } else {
            None
        }
    }

    fn push(&mut self, rect: Option<PixelRect>) {
        let Some(rect) = rect else {
            return;
        };
        if self.rects[0] == Some(rect) || self.rects[1] == Some(rect) {
            return;
        }
        if self.rects[0].is_none() {
            self.rects[0] = Some(rect);
        } else if self.rects[1].is_none() {
            self.rects[1] = Some(rect);
        }
    }
}

/// Result of one cursor state transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CursorTransition {
    /// Exact damage needed by the composited route.
    pub damage: CursorDamage,
    /// Overlay to include in the next Axiom frame, if software-routed.
    pub composite: Option<CompositeCursor>,
    /// Position/visibility state to submit to the cursor plane, if selected.
    pub plane: Option<(PointerPosition, bool)>,
}

/// Backend-neutral cursor presentation state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CursorController {
    target: Extent2d,
    route: CursorRoute,
    position: PointerPosition,
    visual: CursorVisual,
}

impl CursorController {
    /// Creates cursor state. The default route is composited so absence of a
    /// hardware capability is always functional.
    pub const fn new(target: Extent2d, position: PointerPosition, visual: CursorVisual) -> Self {
        Self {
            target,
            route: CursorRoute::Composited,
            position,
            visual,
        }
    }

    /// Returns the selected route.
    pub const fn route(&self) -> CursorRoute {
        self.route
    }

    /// Returns the current position.
    pub const fn position(&self) -> PointerPosition {
        self.position
    }

    /// Reconfigures target geometry and returns the required transition.
    pub fn set_target(&mut self, target: Extent2d) -> CursorTransition {
        let old = self.composite();
        self.target = target;
        self.transition_from(old)
    }

    /// Selects hardware or composited presentation.
    pub fn set_route(&mut self, route: CursorRoute) -> CursorTransition {
        let old = self.composite();
        self.route = route;
        self.transition_from(old)
    }

    /// Updates position and resolved visual.
    pub fn update(&mut self, position: PointerPosition, visual: CursorVisual) -> CursorTransition {
        let old = self.composite();
        self.position = position;
        self.visual = visual;
        self.transition_from(old)
    }

    fn transition_from(&self, old: Option<CompositeCursor>) -> CursorTransition {
        let composite = self.composite();
        let mut damage = CursorDamage::default();
        damage.push(old.map(|cursor| cursor.bounds));
        damage.push(composite.map(|cursor| cursor.bounds));
        CursorTransition {
            damage,
            composite,
            plane: matches!(self.route, CursorRoute::HardwarePlane)
                .then_some((self.position, self.visual.request.visible)),
        }
    }

    fn composite(&self) -> Option<CompositeCursor> {
        if !matches!(self.route, CursorRoute::Composited) || !self.visual.request.visible {
            return None;
        }
        clipped_cursor_bounds(self.position, self.visual, self.target).map(|bounds| {
            CompositeCursor {
                bounds,
                visual: self.visual,
            }
        })
    }
}

fn clipped_cursor_bounds(
    position: PointerPosition,
    visual: CursorVisual,
    target: Extent2d,
) -> Option<PixelRect> {
    let x0 = i64::from(position.x) - i64::from(visual.hotspot.x);
    let y0 = i64::from(position.y) - i64::from(visual.hotspot.y);
    let x1 = x0 + i64::from(visual.extent.width());
    let y1 = y0 + i64::from(visual.extent.height());
    let clipped_x0 = x0.max(0).min(i64::from(target.width()));
    let clipped_y0 = y0.max(0).min(i64::from(target.height()));
    let clipped_x1 = x1.max(0).min(i64::from(target.width()));
    let clipped_y1 = y1.max(0).min(i64::from(target.height()));
    if clipped_x0 >= clipped_x1 || clipped_y0 >= clipped_y1 {
        return None;
    }
    PixelRect::new(
        clipped_x0 as u32,
        clipped_y0 as u32,
        (clipped_x1 - clipped_x0) as u32,
        (clipped_y1 - clipped_y0) as u32,
    )
    .ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axinput::{CursorIcon, CursorRequest};

    #[test]
    fn ring_is_symmetric_antialiased_and_has_an_unambiguous_hotspot() {
        let centre = (RING_HOTSPOT.y as usize * RING_WIDTH + RING_HOTSPOT.x as usize) * 4;
        assert_eq!(&RING_RGBA[centre..centre + 4], &[245, 245, 245, 255]);
        assert!(RING_RGBA.chunks_exact(4).any(|p| p[3] > 0 && p[3] < 255));
        for y in 0..RING_HEIGHT {
            for x in 0..RING_WIDTH {
                let a = (y * RING_WIDTH + x) * 4;
                let b = (y * RING_WIDTH + (RING_WIDTH - 1 - x)) * 4;
                assert_eq!(&RING_RGBA[a..a + 4], &RING_RGBA[b..b + 4]);
                if x == 0 || y == 0 || x == RING_WIDTH - 1 || y == RING_HEIGHT - 1 {
                    assert_eq!(RING_RGBA[a + 3], 0);
                }
            }
        }
    }

    #[test]
    fn built_in_arrow_has_coverage_and_transparent_padding() {
        assert!(ARROW_RGBA.chunks_exact(4).any(|p| p[3] > 0 && p[3] < 255));
        assert!(
            ARROW_RGBA
                .chunks_exact(4)
                .any(|p| p[0] == 245 && p[3] == 255)
        );
        assert!(
            ARROW_RGBA
                .chunks_exact(4)
                // A one-pixel diagonal outline need not contain an entirely
                // opaque square pixel; its coverage still retains dark ink.
                .any(|p| p[0] == 24 && p[3] > 0)
        );
        for x in 0..ARROW_WIDTH {
            assert_eq!(ARROW_RGBA[x * 4 + 3], 0);
            assert_eq!(
                ARROW_RGBA[((ARROW_HEIGHT - 1) * ARROW_WIDTH + x) * 4 + 3],
                0
            );
        }
    }

    #[test]
    fn arrow_silhouette_and_outline_reflect_about_the_same_major_axis() {
        for u in (-16..=ARROW_TAIL + 16).step_by(4) {
            for v in (0..=ARROW_HEAD / 2 + 16).step_by(4) {
                assert_eq!(arrow_ink(u, v), arrow_ink(u, -v), "u={u}, v={v}");
            }
        }
        // Sample both shaft edges and their one-physical-pixel inset. The
        // arrow-local basis length is sqrt(5), not one display pixel.
        let u = 50 * ARROW_SUBPIXELS;
        assert_eq!(arrow_ink(u, 0), 245);
        for side in [-1, 1] {
            assert_eq!(arrow_ink(u, side * (ARROW_STEM_HALF + 1)), 0);
            assert_eq!(arrow_ink(u, side * (ARROW_STEM_HALF - 32)), 24);
            assert_eq!(arrow_ink(u, side * (ARROW_STEM_HALF - 40)), 245);
        }
    }

    #[test]
    fn rasterized_arrow_coverage_and_ink_are_balanced_about_the_hotspot_axis() {
        let mut mass = [0_i64; 2];
        let mut moment = [0_i64; 2];
        for y in 0..ARROW_HEIGHT {
            for x in 0..ARROW_WIDTH {
                let p = (y * ARROW_WIDTH + x) * 4;
                let alpha = i64::from(ARROW_RGBA[p + 3]);
                let v = 2 * (x as i64 - i64::from(ARROW_HOTSPOT.x))
                    - (y as i64 - i64::from(ARROW_HOTSPOT.y));
                for (i, weight) in [alpha, alpha * i64::from(ARROW_RGBA[p])]
                    .into_iter()
                    .enumerate()
                {
                    mass[i] += weight;
                    moment[i] += weight * v;
                }
                if x == 0 || x == ARROW_WIDTH - 1 || y == 0 || y == ARROW_HEIGHT - 1 {
                    assert_eq!(alpha, 0, "clipped cursor at ({x}, {y})");
                }
            }
        }
        // Reflection at this angle does not map square pixels onto square
        // pixels. Allow raster quantization, but less than 0.05 physical px
        // of lateral centre-of-mass error for both silhouette and light ink.
        for i in 0..2 {
            assert!(mass[i] > 0);
            assert!(
                moment[i].abs() * 10 < mass[i],
                "mass={mass:?}, moment={moment:?}"
            );
        }
        assert_eq!(ARROW_HOTSPOT, CursorHotspot { x: 2, y: 1 });
    }

    fn visual(visible: bool) -> CursorVisual {
        CursorVisual {
            request: CursorRequest {
                icon: CursorIcon::Pointer,
                visible,
                ..CursorRequest::DEFAULT
            },
            extent: Extent2d::new(24, 32).unwrap(),
            hotspot: CursorHotspot { x: 3, y: 4 },
        }
    }

    #[test]
    fn movement_damages_exact_old_and_new_composite_bounds() {
        let mut cursor = CursorController::new(
            Extent2d::new(100, 80).unwrap(),
            PointerPosition::new(10, 10),
            visual(true),
        );
        let change = cursor.update(PointerPosition::new(30, 20), visual(true));
        assert_eq!(change.damage.len(), 2);
        assert_eq!(change.damage.get(0), PixelRect::new(7, 6, 24, 32).ok());
        assert_eq!(change.damage.get(1), PixelRect::new(27, 16, 24, 32).ok());
        assert!(change.composite.is_some());
        assert!(change.plane.is_none());
    }

    #[test]
    fn clipping_keeps_partially_offscreen_cursor_damage_exact() {
        let mut cursor = CursorController::new(
            Extent2d::new(20, 20).unwrap(),
            PointerPosition::new(0, 0),
            visual(true),
        );
        let change = cursor.set_target(Extent2d::new(20, 20).unwrap());
        assert_eq!(
            change.composite.unwrap().bounds,
            PixelRect::new(0, 0, 20, 20).unwrap()
        );
    }

    #[test]
    fn switching_to_plane_erases_software_cursor_once() {
        let mut cursor = CursorController::new(
            Extent2d::new(100, 80).unwrap(),
            PointerPosition::new(10, 10),
            visual(true),
        );
        let change = cursor.set_route(CursorRoute::HardwarePlane);
        assert_eq!(change.damage.len(), 1);
        assert!(change.composite.is_none());
        assert_eq!(change.plane, Some((PointerPosition::new(10, 10), true)));

        let moved = cursor.update(PointerPosition::new(40, 40), visual(true));
        assert!(moved.damage.is_empty());
        assert_eq!(moved.plane, Some((PointerPosition::new(40, 40), true)));
    }

    #[test]
    fn hiding_composited_cursor_only_damages_its_old_bounds() {
        let mut cursor = CursorController::new(
            Extent2d::new(100, 80).unwrap(),
            PointerPosition::new(10, 10),
            visual(true),
        );
        let change = cursor.update(PointerPosition::new(10, 10), visual(false));
        assert_eq!(change.damage.len(), 1);
        assert!(change.composite.is_none());
    }
}
