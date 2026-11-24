//! The standard cursor family: Dart, Text, Grab, Grabbing, Pointer, the four
//! resize arrows, NotAllowed and Loading.
//!
//! Every member holds still. A member is drawn at its designed pose, a shape
//! change is a cut, and a click changes nothing about the cursor: the tip a
//! user aims with is exactly where it was drawn, every frame. The one member
//! whose pose is a function of time is Loading, whose band turns while it is
//! shown. The request comes from UI policy (`CursorShape::from_icon`); this
//! type never decides a shape from button state.
//!
//! All state is fixed-size: no allocation, clock, event loop or backend
//! policy lives here.

use crate::dart::{DART_GLYPH, DEFAULT_DT, DartRasterError, MAX_DT, MIN_DT};
use crate::glyph::{Glyph, LayerPose, OUTLINE_WIDTH, rasterize_layers};
use crate::shapes::{
    BADGE_KEYLINE, BADGE_RING, BADGE_SLASH, DENIED, FINE_KEYLINE, GRAB, GRABBING, LOADING,
    POINTER, RESIZE_EW, RESIZE_NESW, RESIZE_NS, RESIZE_NWSE, SPINNER, TEXT,
};
use axgfx::{CursorHotspot, Extent2d};
use axinput::CursorIcon;

/// Every member is drawn in the family's 36-unit canvas.
pub const FAMILY_WIDTH: usize = 36;
pub const FAMILY_HEIGHT: usize = 36;

/// The spinner turns once every 0.9 s.
const SPIN_RATE: f32 = core::f32::consts::TAU / 0.9;

/// The Dart's tilt to the left: its own half-angle, `atan(4.7 / 12.8)` from
/// the shipping Dart's wing (`dart::PATH`), turned about the tip. That stands
/// its left edge upright under the tip — the lean a desktop arrow has — and
/// leaves the geometry and the hotspot exactly the shipping Dart's.
const DART_TILT: f32 = -0.351_85;

/// A family member. Icons without family artwork resolve to the Dart.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CursorShape {
    /// The shipping Dart, tilted left so its left edge stands upright.
    #[default]
    Dart,
    Text,
    Grab,
    Grabbing,
    Pointer,
    NsResize,
    EwResize,
    NwseResize,
    NeswResize,
    NotAllowed,
    /// `Wait` and `Progress`: a spinning blue band.
    Loading,
}

/// Number of family members.
const MEMBERS: usize = 11;

impl CursorShape {
    pub const ALL: [Self; MEMBERS] = [
        Self::Dart,
        Self::Text,
        Self::Grab,
        Self::Grabbing,
        Self::Pointer,
        Self::NsResize,
        Self::EwResize,
        Self::NwseResize,
        Self::NeswResize,
        Self::NotAllowed,
        Self::Loading,
    ];

    /// Resolve a UI cursor request. `Default` and every icon the family has
    /// no artwork for yet present as the Dart.
    pub const fn from_icon(icon: CursorIcon) -> Self {
        match icon {
            CursorIcon::Text => Self::Text,
            CursorIcon::Grab => Self::Grab,
            CursorIcon::Grabbing => Self::Grabbing,
            CursorIcon::Pointer => Self::Pointer,
            CursorIcon::ResizeVertical | CursorIcon::ResizeRow => Self::NsResize,
            CursorIcon::ResizeHorizontal | CursorIcon::ResizeColumn => Self::EwResize,
            CursorIcon::ResizeDiagonalDown => Self::NwseResize,
            CursorIcon::ResizeDiagonalUp => Self::NeswResize,
            CursorIcon::NotAllowed | CursorIcon::NoDrop => Self::NotAllowed,
            CursorIcon::Wait | CursorIcon::Progress => Self::Loading,
            _ => Self::Dart,
        }
    }

    /// Designed hotspot in the 36-unit canvas: the arrow's tip, the pointing
    /// fingertip, the hands' palm, and the centre of everything else.
    pub const fn hotspot(self) -> CursorHotspot {
        match self {
            Self::Dart | Self::Pointer | Self::NotAllowed => CursorHotspot { x: 18, y: 6 },
            Self::Text
            | Self::NsResize
            | Self::EwResize
            | Self::NwseResize
            | Self::NeswResize
            | Self::Loading => CursorHotspot { x: 18, y: 18 },
            Self::Grab | Self::Grabbing => CursorHotspot { x: 18, y: 20 },
        }
    }

    /// The member's fixed rotation about its hotspot: the Dart's tilt, and
    /// none for every other member (the spinner's turn is added to it).
    const fn tilt(self) -> f32 {
        match self {
            Self::Dart => DART_TILT,
            _ => 0.0,
        }
    }

    /// The glyphs one member is drawn from, sharing one placement.
    const fn glyphs(self) -> &'static [Glyph<'static>] {
        match self {
            Self::Dart => &[DART_GLYPH],
            Self::Text => &[TEXT_GLYPH],
            Self::Grab => &[GRAB_GLYPH],
            Self::Grabbing => &[GRABBING_GLYPH],
            Self::Pointer => &[POINTER_GLYPH],
            Self::NsResize => core::slice::from_ref(&RESIZE_GLYPHS[0]),
            Self::EwResize => core::slice::from_ref(&RESIZE_GLYPHS[1]),
            Self::NwseResize => core::slice::from_ref(&RESIZE_GLYPHS[2]),
            Self::NeswResize => core::slice::from_ref(&RESIZE_GLYPHS[3]),
            Self::NotAllowed => &NOT_ALLOWED_GLYPHS,
            Self::Loading => &[LOADING_GLYPH],
        }
    }
}

/// The cursor the runtime shows: one member, and the spinner's turn.
#[derive(Clone, Copy, Debug)]
pub struct FamilyCursor {
    shape: CursorShape,
    /// Spinner angle; advances only while Loading is shown.
    spin: f32,
}

impl FamilyCursor {
    /// A cursor showing `shape`.
    pub const fn new(shape: CursorShape) -> Self {
        Self { shape, spin: 0.0 }
    }

    pub const fn shape(&self) -> CursorShape {
        self.shape
    }

    /// Whether the pose changes with time alone: only a shown spinner does,
    /// so a caller schedules no wake for any other member.
    pub const fn is_active(&self) -> bool {
        matches!(self.shape, CursorShape::Loading)
    }

    /// Show `shape` from the next pose on.
    pub fn set_shape(&mut self, shape: CursorShape) {
        if self.shape != shape {
            self.shape = shape;
            self.spin = 0.0;
        }
    }

    /// Let `dt` seconds pass. Returns whether the pose may still change
    /// without further input.
    pub fn advance(&mut self, dt: f32) -> bool {
        if !self.is_active() {
            return false;
        }
        let dt = if dt.is_finite() {
            dt.clamp(MIN_DT, MAX_DT)
        } else {
            DEFAULT_DT
        };
        self.spin = (self.spin + SPIN_RATE * dt) % core::f32::consts::TAU;
        true
    }

    pub fn pose(&self) -> FamilyPose {
        FamilyPose {
            shape: self.shape,
            spin: self.spin,
        }
    }
}

/// One resolved family frame: the member shown, and the spinner's turn.
/// Equal poses rasterize to identical pixels, so a caller may skip redundant
/// rasters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FamilyPose {
    shape: CursorShape,
    /// Zero for every member but Loading.
    spin: f32,
}

impl FamilyPose {
    /// A member in its designed pose.
    pub fn rest(shape: CursorShape) -> Self {
        Self { shape, spin: 0.0 }
    }

    /// The member's designed hotspot in physical pixels for a canvas sampled
    /// at `pixels_per_unit`; it always lies inside `extent`.
    pub fn hotspot(&self, pixels_per_unit: f32) -> CursorHotspot {
        let designed = self.shape.hotspot();
        CursorHotspot {
            x: (f32::from(designed.x) * pixels_per_unit + 0.5) as u16,
            y: (f32::from(designed.y) * pixels_per_unit + 0.5) as u16,
        }
    }

    /// The 36-unit canvas at `pixels_per_unit`.
    pub fn extent(pixels_per_unit: f32) -> Result<Extent2d, DartRasterError> {
        if !pixels_per_unit.is_finite() || pixels_per_unit <= 0.0 {
            return Err(DartRasterError::InvalidScale);
        }
        let side = (FAMILY_WIDTH as f32 * pixels_per_unit + 0.5) as u32;
        Extent2d::new(side, side).map_err(|_| DartRasterError::InvalidScale)
    }

    /// Rasterize straight-alpha sRGB RGBA, sampling the member's distance
    /// field at the destination resolution. RGB remains unassociated;
    /// consumers that blend premultiplied surfaces must multiply each colour
    /// channel by the emitted alpha first.
    pub fn rasterize_at_scale(
        self,
        extent: Extent2d,
        hotspot: CursorHotspot,
        pixels_per_unit: f32,
        rgba: &mut [u8],
    ) -> Result<(), DartRasterError> {
        if !pixels_per_unit.is_finite() || pixels_per_unit <= 0.0 {
            return Err(DartRasterError::InvalidScale);
        }
        if u32::from(hotspot.x) >= extent.width() || u32::from(hotspot.y) >= extent.height() {
            return Err(DartRasterError::HotspotOutsideExtent);
        }
        let width = extent.width() as usize;
        let height = extent.height() as usize;
        if width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(4))
            != Some(rgba.len())
        {
            return Err(DartRasterError::BufferLength);
        }
        let glyphs = self.shape.glyphs();
        let mut layers = [REST_LAYER; MEMBER_GLYPHS];
        for (layer, &glyph) in layers.iter_mut().zip(glyphs) {
            *layer = LayerPose {
                glyph,
                angle: self.shape.tilt() + self.spin,
                ..REST_LAYER
            };
        }
        rasterize_layers(
            &layers[..glyphs.len()],
            width,
            height,
            [f32::from(hotspot.x), f32::from(hotspot.y)],
            pixels_per_unit,
            rgba,
        );
        Ok(())
    }
}

/// The most glyphs one member is drawn from (NotAllowed: the hand and its
/// badge).
const MEMBER_GLYPHS: usize = 2;

/// A glyph at its designed placement: about the hotspot, unrotated, full
/// size and fully opaque.
const REST_LAYER: LayerPose<'static> = LayerPose {
    glyph: DART_GLYPH,
    angle: 0.0,
    pivot: [0.0, 0.0],
    scale: [1.0, 1.0],
    offset: [0.0, 0.0],
    opacity: 1.0,
};

const TEXT_GLYPH: Glyph<'static> = Glyph {
    polygons: &[],
    capsules: &TEXT,
    seams: &[],
    keyline: FINE_KEYLINE,
    arcs: &[],
    body: crate::glyph::BODY,
    shadow: true,
};

const GRAB_GLYPH: Glyph<'static> = Glyph {
    polygons: &[&GRAB.palm],
    capsules: &GRAB.strokes,
    seams: &GRAB.seams,
    keyline: OUTLINE_WIDTH,
    arcs: &[],
    body: crate::glyph::BODY,
    shadow: true,
};

const GRABBING_GLYPH: Glyph<'static> = Glyph {
    polygons: &[&GRABBING.palm],
    capsules: &GRABBING.strokes,
    seams: &GRABBING.seams,
    keyline: OUTLINE_WIDTH,
    arcs: &[],
    body: crate::glyph::BODY,
    shadow: true,
};

const POINTER_GLYPH: Glyph<'static> = Glyph {
    polygons: &[&POINTER.palm],
    capsules: &POINTER.strokes,
    seams: &POINTER.seams,
    keyline: OUTLINE_WIDTH,
    arcs: &[],
    body: crate::glyph::BODY,
    shadow: true,
};

/// The pointing hand, then the hollow red ring and slash over it: the hand
/// shows through the ring.
const NOT_ALLOWED_GLYPHS: [Glyph<'static>; MEMBER_GLYPHS] = [
    POINTER_GLYPH,
    Glyph {
        polygons: &[],
        capsules: &BADGE_SLASH,
        seams: &[],
        keyline: BADGE_KEYLINE,
        arcs: &BADGE_RING,
        body: DENIED,
        // Flat on the hand: a shadow would grey the open ring.
        shadow: false,
    },
];

const LOADING_GLYPH: Glyph<'static> = Glyph {
    polygons: &[],
    capsules: &[],
    seams: &[],
    keyline: FINE_KEYLINE,
    arcs: &SPINNER,
    body: LOADING,
    shadow: true,
};

const RESIZE_GLYPHS: [Glyph<'static>; 4] = [
    Glyph {
        polygons: &[&RESIZE_NS.heads[0], &RESIZE_NS.heads[1]],
        capsules: &RESIZE_NS.shaft,
        seams: &[],
        keyline: FINE_KEYLINE,
        arcs: &[],
        body: crate::glyph::BODY,
        shadow: true,
    },
    Glyph {
        polygons: &[&RESIZE_EW.heads[0], &RESIZE_EW.heads[1]],
        capsules: &RESIZE_EW.shaft,
        seams: &[],
        keyline: FINE_KEYLINE,
        arcs: &[],
        body: crate::glyph::BODY,
        shadow: true,
    },
    Glyph {
        polygons: &[&RESIZE_NWSE.heads[0], &RESIZE_NWSE.heads[1]],
        capsules: &RESIZE_NWSE.shaft,
        seams: &[],
        keyline: FINE_KEYLINE,
        arcs: &[],
        body: crate::glyph::BODY,
        shadow: true,
    },
    Glyph {
        polygons: &[&RESIZE_NESW.heads[0], &RESIZE_NESW.heads[1]],
        capsules: &RESIZE_NESW.shaft,
        seams: &[],
        keyline: FINE_KEYLINE,
        arcs: &[],
        body: crate::glyph::BODY,
        shadow: true,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec;
    use std::vec::Vec;

    const DT: f32 = 1.0 / 240.0;
    const SCALE: f32 = 1.5;
    const BODY_RGBA: [u8; 4] = [0x56, 0x63, 0x3f, 0xff];

    fn raster(pose: FamilyPose, ppu: f32) -> (Vec<u8>, usize, CursorHotspot) {
        let extent = FamilyPose::extent(ppu).unwrap();
        let side = extent.width() as usize;
        let hotspot = pose.hotspot(ppu);
        let mut rgba = vec![0; side * side * 4];
        pose.rasterize_at_scale(extent, hotspot, ppu, &mut rgba)
            .unwrap();
        (rgba, side, hotspot)
    }

    fn pixel(rgba: &[u8], side: usize, x: u16, y: u16) -> [u8; 4] {
        let at = (usize::from(y) * side + usize::from(x)) * 4;
        rgba[at..at + 4].try_into().unwrap()
    }

    /// The pixel `units` from the hotspot.
    fn at(rgba: &[u8], side: usize, hotspot: CursorHotspot, ppu: f32, units: [f32; 2]) -> [u8; 4] {
        pixel(
            rgba,
            side,
            (f32::from(hotspot.x) + units[0] * ppu) as u16,
            (f32::from(hotspot.y) + units[1] * ppu) as u16,
        )
    }

    fn border_is_clear(rgba: &[u8], side: usize) -> bool {
        (0..side).all(|i| {
            [(i, 0), (i, side - 1), (0, i), (side - 1, i)]
                .iter()
                .all(|&(x, y)| rgba[(y * side + x) * 4 + 3] == 0)
        })
    }

    #[test]
    fn every_member_rests_on_its_designed_hotspot() {
        for shape in CursorShape::ALL {
            let pose = FamilyPose::rest(shape);
            let designed = shape.hotspot();
            for ppu in [1.0, SCALE, 2.0, 3.0] {
                let (rgba, side, hotspot) = raster(pose, ppu);
                assert_eq!(
                    hotspot,
                    CursorHotspot {
                        x: (f32::from(designed.x) * ppu + 0.5) as u16,
                        y: (f32::from(designed.y) * ppu + 0.5) as u16,
                    },
                    "{shape:?}"
                );
                assert!(border_is_clear(&rgba, side), "{shape:?} at {ppu}");
                let at = pixel(&rgba, side, hotspot.x, hotspot.y);
                match shape {
                    // Tips land on the keyline.
                    CursorShape::Dart | CursorShape::Pointer | CursorShape::NotAllowed => {
                        assert!(at[3] == 0xff && at[0] > 0xe0, "{shape:?} tip {at:?}")
                    }
                    // The spinner turns about an empty centre.
                    CursorShape::Loading => assert_eq!(at[3], 0, "hollow spinner {at:?}"),
                    _ => assert_eq!(at, BODY_RGBA, "{shape:?} hotspot inside the body"),
                }
            }
        }
        // Kore's 1.5x placement of the Dart is unchanged.
        assert_eq!(
            FamilyPose::rest(CursorShape::Dart).hotspot(SCALE),
            CursorHotspot { x: 27, y: 9 }
        );
        assert_eq!(
            FamilyPose::extent(SCALE).unwrap(),
            Extent2d::new(54, 54).unwrap()
        );
    }

    #[test]
    fn the_dart_tilts_left_with_its_left_edge_upright() {
        // The tilt carries the left wing straight below the tip.
        let (sin, cos) = axmath::sin_cos_f32(DART_TILT);
        let wing = crate::dart::PATH[1];
        let left = [cos * wing[0] - sin * wing[1], sin * wing[0] + cos * wing[1]];
        assert!(left[0].abs() < 1.0e-3 && left[1] > 13.0, "left wing at {left:?}");

        let ppu = 3.0;
        let (rgba, side, hotspot) = raster(FamilyPose::rest(CursorShape::Dart), ppu);
        let at = |units: [f32; 2]| at(&rgba, side, hotspot, ppu, units);
        // Open ground: at most the soft shadow's faint fringe.
        let ground = |p: [u8; 4]| p[3] < 0x20;
        // Body just inside the upright left edge, ground outside it, and
        // nothing at all three units out — the shadow's reach ends there.
        for down in [4.0, 8.0] {
            assert_eq!(at([1.6, down]), BODY_RGBA, "inside the left edge at {down}");
        }
        for down in [4.0, 8.0, 11.0] {
            assert!(ground(at([-1.5, down])), "outside the left edge at {down}");
            assert_eq!(at([-3.0, down])[3], 0, "past the shadow at {down}");
        }
        // The right edge leans out: body under it, ground above it.
        assert_eq!(at([4.0, 7.0]), BODY_RGBA, "under the right edge");
        assert!(ground(at([6.0, 3.0])), "above the right edge");
        // The notch between the wings stays open, and nothing is above the
        // tip.
        assert!(ground(at([3.6, 11.4])), "the notch");
        assert!(ground(at([0.0, -2.0])), "above the tip");
    }

    #[test]
    fn members_are_distinct_colored_and_antialiased() {
        let rasters: Vec<_> = CursorShape::ALL
            .iter()
            .map(|&shape| raster(FamilyPose::rest(shape), SCALE).0)
            .collect();
        for (i, a) in rasters.iter().enumerate() {
            assert!(a.chunks_exact(4).any(|p| p[3] > 0 && p[3] < 255));
            assert!(a.chunks_exact(4).any(|p| p[1] > p[0] && p[3] > 200));
            for b in &rasters[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn the_family_holds_still() {
        for shape in CursorShape::ALL {
            let mut cursor = FamilyCursor::new(shape);
            assert_eq!(cursor.pose(), FamilyPose::rest(shape));
            assert_eq!(cursor.is_active(), shape == CursorShape::Loading, "{shape:?}");
            if shape != CursorShape::Loading {
                // Time alone moves nothing, and asks for no wake.
                for _ in 0..240 {
                    assert!(!cursor.advance(DT));
                }
                assert_eq!(cursor.pose(), FamilyPose::rest(shape), "{shape:?}");
            }
            // A change is a cut: the next pose is the new member at rest.
            for next in CursorShape::ALL {
                cursor.set_shape(next);
                assert_eq!(cursor.pose(), FamilyPose::rest(next), "{shape:?}->{next:?}");
                assert_eq!(cursor.pose().hotspot(SCALE), FamilyPose::rest(next).hotspot(SCALE));
            }
        }
    }

    #[test]
    fn invalid_and_extreme_frame_deltas_remain_finite() {
        let mut cursor = FamilyCursor::new(CursorShape::Loading);
        for dt in [f32::NAN, f32::INFINITY, -1.0, 0.0, 10.0, 1.0e-9]
            .into_iter()
            .cycle()
            .take(120)
        {
            assert!(cursor.advance(dt));
            assert!((0.0..core::f32::consts::TAU).contains(&cursor.spin), "{}", cursor.spin);
        }
        let (rgba, side, _) = raster(cursor.pose(), 1.0);
        assert!(border_is_clear(&rgba, side));
    }

    #[test]
    fn icons_resolve_to_members_or_the_dart() {
        assert_eq!(
            CursorShape::from_icon(CursorIcon::Default),
            CursorShape::Dart
        );
        assert_eq!(CursorShape::from_icon(CursorIcon::Text), CursorShape::Text);
        assert_eq!(CursorShape::from_icon(CursorIcon::Grab), CursorShape::Grab);
        assert_eq!(
            CursorShape::from_icon(CursorIcon::Grabbing),
            CursorShape::Grabbing
        );
        assert_eq!(
            CursorShape::from_icon(CursorIcon::Pointer),
            CursorShape::Pointer
        );
        for (icon, shape) in [
            (CursorIcon::ResizeVertical, CursorShape::NsResize),
            (CursorIcon::ResizeRow, CursorShape::NsResize),
            (CursorIcon::ResizeHorizontal, CursorShape::EwResize),
            (CursorIcon::ResizeColumn, CursorShape::EwResize),
            (CursorIcon::ResizeDiagonalDown, CursorShape::NwseResize),
            (CursorIcon::ResizeDiagonalUp, CursorShape::NeswResize),
        ] {
            assert_eq!(CursorShape::from_icon(icon), shape);
        }
        assert_eq!(
            CursorShape::from_icon(CursorIcon::NotAllowed),
            CursorShape::NotAllowed
        );
        assert_eq!(
            CursorShape::from_icon(CursorIcon::NoDrop),
            CursorShape::NotAllowed
        );
        assert_eq!(CursorShape::from_icon(CursorIcon::Help), CursorShape::Dart);
        assert_eq!(
            CursorShape::from_icon(CursorIcon::Wait),
            CursorShape::Loading
        );
        assert_eq!(
            CursorShape::from_icon(CursorIcon::Progress),
            CursorShape::Loading
        );
        assert_eq!(
            CursorShape::from_icon(CursorIcon::Custom(axinput::CursorImageId(7))),
            CursorShape::Dart
        );
    }

    #[test]
    fn raster_contract_refuses_ambiguous_geometry() {
        let pose = FamilyPose::rest(CursorShape::Text);
        let extent = FamilyPose::extent(1.0).unwrap();
        let mut rgba = vec![0; 36 * 36 * 4];
        let hotspot = pose.hotspot(1.0);
        assert_eq!(
            pose.rasterize_at_scale(extent, hotspot, 0.0, &mut rgba),
            Err(DartRasterError::InvalidScale)
        );
        assert_eq!(
            pose.rasterize_at_scale(extent, CursorHotspot { x: 36, y: 0 }, 1.0, &mut rgba),
            Err(DartRasterError::HotspotOutsideExtent)
        );
        assert_eq!(
            pose.rasterize_at_scale(extent, hotspot, 1.0, &mut rgba[4..]),
            Err(DartRasterError::BufferLength)
        );
        assert!(FamilyPose::extent(f32::NAN).is_err());
    }

    #[test]
    fn text_and_resize_use_the_fine_keyline() {
        // The same geometry drawn with the Dart's keyline puts cream farther
        // from the outline than the fine keyline does.
        let cream = |rgba: &[u8]| {
            rgba.chunks_exact(4)
                .filter(|p| p[3] == 0xff && p[0] > 0xe0 && p[1] > 0xe0)
                .count()
        };
        let extent = FamilyPose::extent(3.0).unwrap();
        let hotspot = FamilyPose::rest(CursorShape::Text).hotspot(3.0);
        for glyph in [TEXT_GLYPH, RESIZE_GLYPHS[0], RESIZE_GLYPHS[2]] {
            let mut counts = [0; 2];
            for (count, keyline) in counts.iter_mut().zip([FINE_KEYLINE, OUTLINE_WIDTH]) {
                let mut rgba = vec![0; 108 * 108 * 4];
                rasterize_layers(
                    &[LayerPose {
                        glyph: Glyph { keyline, ..glyph },
                        ..REST_LAYER
                    }],
                    extent.width() as usize,
                    extent.height() as usize,
                    [f32::from(hotspot.x), f32::from(hotspot.y)],
                    3.0,
                    &mut rgba,
                );
                *count = cream(&rgba);
            }
            assert!(counts[0] > 0 && counts[0] < counts[1], "{counts:?}");
        }
    }

    #[test]
    fn not_allowed_is_the_pointing_hand_under_a_hollow_red_badge() {
        let ppu = 3.0;
        let (rgba, side, hotspot) = raster(FamilyPose::rest(CursorShape::NotAllowed), ppu);
        assert_eq!(hotspot, CursorHotspot { x: 54, y: 18 });
        let at = |ux: f32, uy: f32| at(&rgba, side, hotspot, ppu, [ux, uy]);
        let red = |p: [u8; 4]| p[3] == 0xff && p[0] > 0xa0 && p[1] < 0x60 && p[2] < 0x50;
        // Badge centre (8.8, 12.4) lies on the slash; the ring's bottom at
        // radius 3.6.
        assert!(red(at(8.8, 12.4)), "slash {:?}", at(8.8, 12.4));
        assert!(red(at(8.8, 16.1)), "ring {:?}", at(8.8, 16.1));
        // Hollow: inside the ring, off the slash, the ground shows through.
        assert!(
            at(10.6, 12.2)[3] < 0x40,
            "ground through the ring {:?}",
            at(10.6, 12.2)
        );
        // The fingertip is untouched: identical to Pointer above the badge.
        let (pointer, _, _) = raster(FamilyPose::rest(CursorShape::Pointer), ppu);
        let rows = usize::from(hotspot.y) + 3 * ppu as usize;
        assert_eq!(rgba[..rows * side * 4], pointer[..rows * side * 4]);
        // The badge overlaps the palm, slightly: some palm pixels change,
        // but most of the hand is still the Pointer's.
        let palm = |x: usize, y: usize| {
            let (ux, uy) = (
                (x as f32 - f32::from(hotspot.x)) / ppu,
                (y as f32 - f32::from(hotspot.y)) / ppu,
            );
            (-1.2..7.15).contains(&ux) && (6.08..12.8).contains(&uy)
        };
        let (mut covered, mut total) = (0, 0);
        for y in 0..side {
            for x in 0..side {
                if palm(x, y) {
                    total += 1;
                    let at = (y * side + x) * 4;
                    covered += usize::from(rgba[at..at + 4] != pointer[at..at + 4]);
                }
            }
        }
        assert!(
            covered > 0 && covered * 4 < total,
            "{covered} of {total} palm pixels"
        );
    }

    #[test]
    fn loading_spins_blue_while_shown_and_rests_once_gone() {
        let blue = |p: [u8; 4]| p[3] == 0xff && p[2] > 0xa0 && p[2] > p[0] + 0x40;
        let ppu = 3.0;
        let mut cursor = FamilyCursor::new(CursorShape::Loading);
        let (first, side, hotspot) = raster(cursor.pose(), ppu);
        // The band's centre line (radius 4.6) is blue below the hotspot, and
        // its 90° gap faces up.
        let around = |rgba: &[u8], angle: f32| {
            let (s, c) = axmath::sin_cos_f32(angle);
            pixel(
                rgba,
                side,
                (f32::from(hotspot.x) + 4.6 * c * ppu + 0.5) as u16,
                (f32::from(hotspot.y) + 4.6 * s * ppu + 0.5) as u16,
            )
        };
        let quarter = core::f32::consts::FRAC_PI_2;
        assert!(
            blue(around(&first, quarter)),
            "band below {:?}",
            around(&first, quarter)
        );
        assert!(
            around(&first, -quarter)[3] < 0x80,
            "gap above {:?}",
            around(&first, -quarter)
        );

        // It turns with nothing but time, about a still hotspot, at one
        // turn per 0.9 s.
        let mut previous = first;
        for step in 1..=54 {
            assert!(cursor.advance(DT), "a shown spinner never rests");
            if step % 18 == 0 {
                let (next, _, hot) = raster(cursor.pose(), ppu);
                assert_eq!(hot, hotspot);
                assert_ne!(next, previous, "turned by step {step}");
                previous = next;
            }
        }
        let expected = core::f32::consts::TAU * (54.0 * DT) / 0.9;
        assert!(
            (cursor.spin - expected).abs() < 1.0e-3,
            "{} vs {expected}",
            cursor.spin
        );

        cursor.set_shape(CursorShape::Dart);
        assert!(!cursor.is_active());
        assert_eq!(cursor.pose(), FamilyPose::rest(CursorShape::Dart));
        assert_eq!(cursor.spin, 0.0);
    }
}
