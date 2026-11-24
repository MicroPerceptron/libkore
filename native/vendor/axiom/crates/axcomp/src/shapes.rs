//! Standard cursor family geometry, hotspot-relative, in 36-unit cursor space.
//!
//! Text, Grab, Grabbing and Pointer follow Lucide's `text-cursor`, `hand`,
//! `hand-grab` and `pointer` icons (ISC, see `LICENSE-LUCIDE`), filled, fitted to
//! the Dart's 12.8-unit height, and flattened to straight edges at a 0.015-unit
//! tolerance. The four resize arrows are original to the family. Generated
//! from the "Kore cursor family" design (`geometry.json`, `hands.json` variant
//! 4b, and the resize/fine-keyline update); edit the design, not these tables.

use crate::glyph::{Arc, Capsule, full_ring};

/// One hand: a 16-point palm, four fingers and a thumb as rounded strokes, and
/// three seams between the fingers. Grab, Grabbing and Pointer share this
/// topology, so a change between two of them can interpolate point for point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Hand {
    pub palm: [[f32; 2]; 16],
    pub strokes: [Capsule; 5],
    pub seams: [Capsule; 3],
}

/// `4b-grab`, hotspot (18, 20) in the 36-unit canvas.
pub(crate) const GRAB: Hand = Hand {
    palm: [
        [-3.53, -1.62],
        [4.83, -1.62],
        [4.83, 0.0],
        [4.76, 0.89],
        [4.57, 1.75],
        [4.27, 2.55],
        [3.85, 3.28],
        [3.33, 3.91],
        [2.74, 4.42],
        [2.08, 4.79],
        [1.38, 5.02],
        [0.65, 5.1],
        [-0.41, 5.1],
        [-1.69, 4.91],
        [-2.7, 4.39],
        [-3.53, 3.64],
    ],
    strokes: [
        [-2.32, -0.98, -2.32, -5.2, 1.205],
        [-0.34, -0.98, -0.34, -6.5, 1.205],
        [1.64, -0.98, 1.64, -5.2, 1.205],
        [3.62, -0.98, 3.62, -3.9, 1.205],
        [-2.58, 2.86, -5.04, 0.42, 1.205],
    ],
    seams: [
        [-1.33, -6.01, -1.33, -1.0, 0.43],
        [0.65, -6.01, 0.65, -1.0, 0.43],
        [2.63, -4.71, 2.63, -1.0, 0.43],
    ],
};

/// `4b-grabbing`, hotspot (18, 20) in the 36-unit canvas.
pub(crate) const GRABBING: Hand = Hand {
    palm: [
        [-3.53, -1.62],
        [4.83, -1.62],
        [4.83, 0.0],
        [4.76, 0.89],
        [4.57, 1.75],
        [4.27, 2.55],
        [3.85, 3.28],
        [3.33, 3.91],
        [2.74, 4.42],
        [2.08, 4.79],
        [1.38, 5.02],
        [0.65, 5.1],
        [-0.41, 5.1],
        [-1.69, 4.91],
        [-2.7, 4.39],
        [-3.53, 3.64],
    ],
    strokes: [
        [-2.32, -0.98, -2.32, -2.99, 1.205],
        [-0.34, -0.98, -0.34, -3.51, 1.205],
        [1.64, -0.98, 1.64, -2.99, 1.205],
        [3.62, -0.98, 3.62, -1.95, 1.205],
        [-2.58, 2.86, -4.14, 1.3, 1.205],
    ],
    seams: [
        [-1.33, -2.98, -1.33, -0.97, 0.43],
        [0.65, -2.98, 0.65, -0.97, 0.43],
        [2.63, -2.42, 2.63, -1.01, 0.43],
    ],
};

/// `4b-pointer`, hotspot (18, 6) in the 36-unit canvas.
pub(crate) const POINTER: Hand = Hand {
    palm: [
        [-1.2, 6.08],
        [7.15, 6.08],
        [7.15, 7.7],
        [7.09, 8.59],
        [6.9, 9.45],
        [6.59, 10.25],
        [6.17, 10.98],
        [5.66, 11.61],
        [5.06, 12.12],
        [4.4, 12.5],
        [3.7, 12.73],
        [2.97, 12.8],
        [1.91, 12.8],
        [0.63, 12.61],
        [-0.38, 12.09],
        [-1.2, 11.34],
    ],
    strokes: [
        [0.0, 6.73, 0.0, 1.2, 1.205],
        [1.98, 6.73, 1.98, 4.84, 1.205],
        [3.96, 6.73, 3.96, 5.23, 1.205],
        [5.95, 6.73, 5.95, 5.75, 1.205],
        [-0.26, 10.56, -2.72, 8.13, 1.205],
    ],
    seams: [
        [0.99, 2.67, 0.99, 6.68, 0.43],
        [2.97, 5.33, 2.97, 6.64, 0.43],
        [4.96, 5.75, 4.96, 6.66, 0.43],
    ],
};

/// Lucide `text-cursor` as 54 rounded-end segments of width 2.4; hotspot (18, 18).
pub(crate) const TEXT: [Capsule; 54] = [
    [2.38, 4.76, 1.76, 4.75, 1.2],
    [1.76, 4.75, 1.52, 4.73, 1.2],
    [1.52, 4.73, 1.33, 4.68, 1.2],
    [1.33, 4.68, 1.12, 4.59, 1.2],
    [1.12, 4.59, 0.91, 4.48, 1.2],
    [0.91, 4.48, 0.71, 4.34, 1.2],
    [0.71, 4.34, 0.54, 4.18, 1.2],
    [0.54, 4.18, 0.38, 4.0, 1.2],
    [0.38, 4.0, 0.25, 3.8, 1.2],
    [0.25, 3.8, 0.15, 3.59, 1.2],
    [0.15, 3.59, 0.07, 3.37, 1.2],
    [0.07, 3.37, 0.03, 3.18, 1.2],
    [0.03, 3.18, 0.01, 2.99, 1.2],
    [0.01, 2.99, 0.0, 2.33, 1.2],
    [0.0, 2.33, 0.0, -2.91, 1.2],
    [0.0, -2.91, 0.03, -3.2, 1.2],
    [0.03, -3.2, 0.1, -3.47, 1.2],
    [0.1, -3.47, 0.24, -3.78, 1.2],
    [0.24, -3.78, 0.43, -4.05, 1.2],
    [0.43, -4.05, 0.54, -4.2, 1.2],
    [0.54, -4.2, 0.69, -4.32, 1.2],
    [0.69, -4.32, 0.96, -4.51, 1.2],
    [0.96, -4.51, 1.27, -4.65, 1.2],
    [1.27, -4.65, 1.59, -4.73, 1.2],
    [1.59, -4.73, 1.83, -4.76, 1.2],
    [1.83, -4.76, 2.38, -4.76, 1.2],
    [-2.38, 4.76, -1.76, 4.75, 1.2],
    [-1.76, 4.75, -1.44, 4.7, 1.2],
    [-1.44, 4.7, -1.25, 4.65, 1.2],
    [-1.25, 4.65, -1.08, 4.57, 1.2],
    [-1.08, 4.57, -0.91, 4.48, 1.2],
    [-0.91, 4.48, -0.71, 4.34, 1.2],
    [-0.71, 4.34, -0.57, 4.22, 1.2],
    [-0.57, 4.22, -0.41, 4.04, 1.2],
    [-0.41, 4.04, -0.27, 3.84, 1.2],
    [-0.27, 3.84, -0.19, 3.68, 1.2],
    [-0.19, 3.68, -0.09, 3.46, 1.2],
    [-0.09, 3.46, -0.05, 3.27, 1.2],
    [-0.05, 3.27, -0.01, 3.09, 1.2],
    [-0.01, 3.09, 0.0, 2.86, 1.2],
    [-2.38, -4.76, -1.76, -4.75, 1.2],
    [-1.76, -4.75, -1.44, -4.7, 1.2],
    [-1.44, -4.7, -1.25, -4.65, 1.2],
    [-1.25, -4.65, -1.08, -4.57, 1.2],
    [-1.08, -4.57, -0.91, -4.48, 1.2],
    [-0.91, -4.48, -0.71, -4.34, 1.2],
    [-0.71, -4.34, -0.54, -4.18, 1.2],
    [-0.54, -4.18, -0.41, -4.04, 1.2],
    [-0.41, -4.04, -0.27, -3.84, 1.2],
    [-0.27, -3.84, -0.19, -3.68, 1.2],
    [-0.19, -3.68, -0.09, -3.46, 1.2],
    [-0.09, -3.46, -0.05, -3.27, 1.2],
    [-0.05, -3.27, -0.01, -3.09, 1.2],
    [-0.01, -3.09, 0.0, -2.86, 1.2],
];

/// Keyline of Text and the resize arrows (the Dart and hands keep 1.3).
pub(crate) const FINE_KEYLINE: f32 = 0.9;

/// A resize cursor: two Dart-style arrowheads (tip, back corner, notch, back
/// corner) joined by a rounded shaft, 15.2 units tip to tip, hotspot at the
/// centre (18, 18). East-west and the diagonals are the north-south arrow
/// rotated 90° and ∓45°, rounded to 0.01 units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Resize {
    pub heads: [[[f32; 2]; 4]; 2],
    pub shaft: [Capsule; 1],
}

/// `NsResize`.
pub(crate) const RESIZE_NS: Resize = Resize {
    heads: [
        [[0.0, -7.6], [-3.7, -1.4], [0.0, -2.8], [3.7, -1.4]],
        [[0.0, 7.6], [-3.7, 1.4], [0.0, 2.8], [3.7, 1.4]],
    ],
    shaft: [[0.0, -3.1, 0.0, 3.1, 1.1]],
};

/// `EwResize`.
pub(crate) const RESIZE_EW: Resize = Resize {
    heads: [
        [[7.6, 0.0], [1.4, -3.7], [2.8, 0.0], [1.4, 3.7]],
        [[-7.6, 0.0], [-1.4, -3.7], [-2.8, 0.0], [-1.4, 3.7]],
    ],
    shaft: [[3.1, 0.0, -3.1, 0.0, 1.1]],
};

/// `NwseResize`.
pub(crate) const RESIZE_NWSE: Resize = Resize {
    heads: [
        [[-5.37, -5.37], [-3.61, 1.63], [-1.98, -1.98], [1.63, -3.61]],
        [[5.37, 5.37], [-1.63, 3.61], [1.98, 1.98], [3.61, -1.63]],
    ],
    shaft: [[-2.19, -2.19, 2.19, 2.19, 1.1]],
};

/// `NeswResize`.
pub(crate) const RESIZE_NESW: Resize = Resize {
    heads: [
        [[5.37, -5.37], [-1.63, -3.61], [1.98, -1.98], [3.61, 1.63]],
        [[-5.37, 5.37], [-3.61, -1.63], [-1.98, 1.98], [1.63, 3.61]],
    ],
    shaft: [[2.19, -2.19, -2.19, 2.19, 1.1]],
};

/// Not-allowed badge colour: a brick red that holds against both the olive
/// body and the cream keyline.
pub(crate) const DENIED: [f32; 3] = [
    0xc2 as f32 / 255.0,
    0x3b as f32 / 255.0,
    0x2e as f32 / 255.0,
];

/// Centre of the not-allowed badge relative to the pointing hand's hotspot
/// (fingertip, 18, 6): off the lower right of the palm, overlapping it only
/// slightly, and clear of the fingertip so the hotspot still reads.
const BADGE: [f32; 2] = [8.8, 12.4];

/// A hollow red ring with a top-left to bottom-right slash; the cream keyline
/// around the red carries it over the hand beneath. Authored for the family
/// rather than taken from the design export.
pub(crate) const BADGE_RING: [Arc; 1] = [full_ring(BADGE[0], BADGE[1], 3.6, 0.75)];
pub(crate) const BADGE_SLASH: [Capsule; 1] = [[
    BADGE[0] - 2.55,
    BADGE[1] - 2.55,
    BADGE[0] + 2.55,
    BADGE[1] + 2.55,
    0.75,
]];
/// The badge's keyline: thinner than the family's so the hole stays open
/// and the red keeps a 0.9-unit core.
pub(crate) const BADGE_KEYLINE: f32 = 0.6;

/// Loading spinner colour: a clear blue that reads beside the olive family
/// and on both desktop grounds.
pub(crate) const LOADING: [f32; 3] = [
    0x2f as f32 / 255.0,
    0x6f as f32 / 255.0,
    0xc4 as f32 / 255.0,
];

/// The loading spinner: a 270° band centred on the hotspot (18, 18) with
/// rounded ends, spun about its centre. Authored for the family.
pub(crate) const SPINNER: [Arc; 1] = [[
    0.0,
    0.0,
    4.6,
    1.1,
    // Centred on straight down (y grows downward): the 90° gap faces up.
    0.0,
    1.0,
    -core::f32::consts::FRAC_1_SQRT_2,
    core::f32::consts::FRAC_1_SQRT_2,
]];
