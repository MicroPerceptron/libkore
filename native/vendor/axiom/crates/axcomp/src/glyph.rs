//! The one distance-field raster behind every standard cursor.
//!
//! A glyph is straight-edged outlines plus rounded-end strokes in 36-unit
//! cursor space, relative to its hotspot. Every destination pixel samples the
//! signed distance afresh, so scale is presentation density rather than a
//! bitmap multiplier, and the body, keyline and shadow stay one family.

use axmath::{sin_cos_f32, sqrt_f32};

/// The Dart's and the hands' keyline. Text and the resize arrows use a
/// thinner one (`shapes::FINE_KEYLINE`).
pub(crate) const OUTLINE_WIDTH: f32 = 1.3;
/// Widest keyline any glyph may declare; bounds the pixel early-out.
const MAX_KEYLINE: f32 = OUTLINE_WIDTH;
pub(crate) const SHADOW_SOFTNESS: f32 = 2.2;
const SHADOW_ALPHA: f32 = 0.26;
pub(crate) const SHADOW_OFFSET: [f32; 2] = [0.35, 0.75];
pub(crate) const BODY: [f32; 3] = [
    0x56 as f32 / 255.0,
    0x63 as f32 / 255.0,
    0x3f as f32 / 255.0,
];
pub(crate) const KEYLINE: [f32; 3] = [
    0xf9 as f32 / 255.0,
    0xf4 as f32 / 255.0,
    0xed as f32 / 255.0,
];
const SHADOW: [f32; 3] = [
    0x2e as f32 / 255.0,
    0x2b as f32 / 255.0,
    0x25 as f32 / 255.0,
];
/// Smallest and largest axis scale a layer may present. The lower bound keeps
/// a squashed or entering layer legible; the upper bound keeps overshoot
/// inside every fixed extent the family tests.
pub(crate) const MIN_SCALE: f32 = 0.5;
pub(crate) const MAX_SCALE: f32 = 1.25;
/// Beyond this distance a sample has no shadow: the shadow is offset by
/// |SHADOW_OFFSET| < 0.83 and fades out at SHADOW_SOFTNESS.
const SHADOW_REACH: f32 = SHADOW_SOFTNESS + 0.83;

/// Rounded-end stroke: segment `(ax, ay) → (bx, by)` dilated by `radius`.
pub(crate) type Capsule = [f32; 5];
/// Circular band with rounded ends: centre `(cx, cy)`, centre-line
/// `radius`, `half_width`, the unit direction `(mid_x, mid_y)` the band is
/// centred on, and the cosine and sine of half its sweep. A full ring has
/// half-sweep π (cosine −1, sine 0) and so no ends.
pub(crate) type Arc = [f32; 8];

/// A closed ring: every direction lies within the half-sweep π.
pub(crate) const fn full_ring(cx: f32, cy: f32, radius: f32, half_width: f32) -> Arc {
    [cx, cy, radius, half_width, 1.0, 0.0, -1.0, 0.0]
}

/// Borrowed glyph geometry, hotspot-relative, in cursor units.
#[derive(Clone, Copy)]
pub(crate) struct Glyph<'a> {
    /// Even-odd outlines, unioned.
    pub polygons: &'a [&'a [[f32; 2]]],
    /// Strokes unioned with the outline.
    pub capsules: &'a [Capsule],
    /// Keyline-coloured splits drawn only inside the body.
    pub seams: &'a [Capsule],
    /// Circular bands unioned with the outline.
    pub arcs: &'a [Arc],
    /// Keyline width, centred on the outline.
    pub keyline: f32,
    /// Fill colour inside the outline (the family olive for every member
    /// except the not-allowed badge).
    pub body: [f32; 3],
    /// Whether the glyph casts the family drop shadow. A glyph drawn flat on
    /// another (the badge's red on its cream disc) does not.
    pub shadow: bool,
}

/// One glyph placed for a frame. `angle` rotates about `pivot`; `scale`
/// shrinks about the hotspot, which therefore never moves; `offset`
/// translates the whole layer; `opacity` fades body, keyline and shadow
/// together so an outgoing layer never leaves its shadow behind.
#[derive(Clone, Copy)]
pub(crate) struct LayerPose<'a> {
    pub glyph: Glyph<'a>,
    pub angle: f32,
    pub pivot: [f32; 2],
    pub scale: [f32; 2],
    pub offset: [f32; 2],
    pub opacity: f32,
}

/// Capsules are culled in groups of this many against a group bounding box.
const CHUNK: usize = 8;
/// Groups with their own box; any remainder shares the last one.
const MAX_CHUNKS: usize = 8;

/// Axis-aligned box in glyph-local units.
#[derive(Clone, Copy)]
struct Bounds {
    min: [f32; 2],
    max: [f32; 2],
}

impl Bounds {
    const EMPTY: Self = Self {
        min: [f32::INFINITY; 2],
        max: [f32::NEG_INFINITY; 2],
    };

    fn include(&mut self, point: [f32; 2], radius: f32) {
        for axis in 0..2 {
            self.min[axis] = self.min[axis].min(point[axis] - radius);
            self.max[axis] = self.max[axis].max(point[axis] + radius);
        }
    }

    fn of_capsules(capsules: &[Capsule]) -> Self {
        let mut bounds = Self::EMPTY;
        for capsule in capsules {
            bounds.include([capsule[0], capsule[1]], capsule[4]);
            bounds.include([capsule[2], capsule[3]], capsule[4]);
        }
        bounds
    }

    /// Distance from `p` to the box (zero inside): a lower bound on the
    /// distance to anything the box contains.
    fn distance(&self, p: [f32; 2]) -> f32 {
        let dx = (self.min[0] - p[0]).max(p[0] - self.max[0]).max(0.0);
        let dy = (self.min[1] - p[1]).max(p[1] - self.max[1]).max(0.0);
        sqrt_f32(dx * dx + dy * dy)
    }
}

/// Culling structure for one glyph. Skipping only work whose lower bound
/// cannot lower the running minimum leaves every distance bit-identical.
#[derive(Clone, Copy)]
struct Culling {
    /// Box around all ink (radii included): a lower bound outside the glyph.
    whole: Bounds,
    /// Boxes around each group's bare segments, with the group's largest
    /// radius. `box distance − radius` bounds every capsule in the group from
    /// below even inside the glyph, where distances are negative.
    chunks: [Bounds; MAX_CHUNKS],
    radius: [f32; MAX_CHUNKS],
    count: usize,
}

impl Culling {
    fn new(glyph: Glyph<'_>) -> Self {
        let mut whole = Bounds::of_capsules(glyph.capsules);
        for arc in glyph.arcs {
            whole.include([arc[0], arc[1]], arc[2] + arc[3]);
        }
        for polygon in glyph.polygons {
            for &point in *polygon {
                whole.include(point, 0.0);
            }
        }
        let mut chunks = [Bounds::EMPTY; MAX_CHUNKS];
        let mut radius = [0.0; MAX_CHUNKS];
        let count = glyph.capsules.len().div_ceil(CHUNK).min(MAX_CHUNKS);
        for index in 0..count {
            for capsule in &glyph.capsules[chunk_range(index, count, glyph.capsules.len())] {
                chunks[index].include([capsule[0], capsule[1]], 0.0);
                chunks[index].include([capsule[2], capsule[3]], 0.0);
                radius[index] = f32::max(radius[index], capsule[4]);
            }
        }
        Self {
            whole,
            chunks,
            radius,
            count,
        }
    }
}

/// Per-layer constants hoisted out of the pixel loop.
struct Prepared<'a> {
    glyph: Glyph<'a>,
    culling: Culling,
    sin: f32,
    cos: f32,
    translate: [f32; 2],
    offset: [f32; 2],
    scale: [f32; 2],
    min_scale: f32,
    keyline_half: f32,
    opacity: f32,
}

/// Composite `layers` in order (later above earlier) into straight-alpha sRGB
/// RGBA. The caller has validated `rgba.len() == width * height * 4`.
pub(crate) fn rasterize_layers(
    layers: &[LayerPose<'_>],
    width: usize,
    height: usize,
    hotspot: [f32; 2],
    pixels_per_unit: f32,
    rgba: &mut [u8],
) {
    const NONE: Option<Prepared<'static>> = None;
    let mut prepared = [NONE; MAX_LAYERS];
    let mut count = 0;
    for layer in layers.iter().take(MAX_LAYERS) {
        let (sin, cos) = sin_cos_f32(layer.angle);
        let pivot = layer.pivot;
        let sx = layer.scale[0].clamp(MIN_SCALE, MAX_SCALE);
        let sy = layer.scale[1].clamp(MIN_SCALE, MAX_SCALE);
        prepared[count] = Some(Prepared {
            glyph: layer.glyph,
            culling: Culling::new(layer.glyph),
            sin,
            cos,
            // Rotation about `pivot`: world = R·local + pivot − R·pivot.
            translate: [
                pivot[0] * (1.0 - cos) + sin * pivot[1],
                pivot[1] * (1.0 - cos) - sin * pivot[0],
            ],
            offset: layer.offset,
            scale: [sx, sy],
            min_scale: sx.min(sy),
            // Clamped so the pixel early-out's reach always covers it.
            keyline_half: layer.glyph.keyline.clamp(0.0, MAX_KEYLINE) * 0.5,
            opacity: layer.opacity.clamp(0.0, 1.0),
        });
        count += 1;
    }
    let prepared = &prepared[..count];
    // A sample farther than this from a layer's glyph receives exactly zero
    // shadow, body, keyline and seam coverage from it.
    let reach = SHADOW_REACH.max(MAX_KEYLINE * 0.5 + 0.5 / pixels_per_unit) + 0.01;

    for y in 0..height {
        for x in 0..width {
            let sample = [
                (x as f32 - hotspot[0]) / pixels_per_unit,
                (y as f32 - hotspot[1]) / pixels_per_unit,
            ];
            let mut premultiplied = [0.0; 3];
            let mut alpha = 0.0;
            for layer in prepared.iter().flatten() {
                let p = [
                    sample[0] - layer.offset[0] - layer.translate[0],
                    sample[1] - layer.offset[1] - layer.translate[1],
                ];
                // Pixels far from the whole glyph see neither it nor its
                // shadow; most of the canvas takes this exit.
                if layer.culling.whole.distance(layer.local(p)) * layer.min_scale > reach {
                    continue;
                }
                let d = layer.distance(p);
                if d > reach {
                    continue;
                }
                let shadow_alpha = if layer.glyph.shadow {
                    let shadow_d =
                        layer.distance([p[0] - SHADOW_OFFSET[0], p[1] - SHADOW_OFFSET[1]]);
                    (1.0 - smoothstep(-1.0, SHADOW_SOFTNESS, shadow_d)) * SHADOW_ALPHA
                } else {
                    0.0
                };
                let mut own = [
                    SHADOW[0] * shadow_alpha,
                    SHADOW[1] * shadow_alpha,
                    SHADOW[2] * shadow_alpha,
                ];
                let mut own_alpha = shadow_alpha;
                over(
                    &mut own,
                    &mut own_alpha,
                    layer.glyph.body,
                    coverage(d * pixels_per_unit),
                );
                over(
                    &mut own,
                    &mut own_alpha,
                    KEYLINE,
                    coverage((d.abs() - layer.keyline_half) * pixels_per_unit),
                );
                if d < 0.0 && !layer.glyph.seams.is_empty() {
                    let seam = layer.seam_distance(p);
                    over(
                        &mut own,
                        &mut own_alpha,
                        KEYLINE,
                        coverage(seam * pixels_per_unit),
                    );
                }
                let behind = 1.0 - own_alpha * layer.opacity;
                for channel in 0..3 {
                    premultiplied[channel] =
                        own[channel] * layer.opacity + premultiplied[channel] * behind;
                }
                alpha = own_alpha * layer.opacity + alpha * behind;
            }
            let at = (y * width + x) * 4;
            if alpha <= 0.0 {
                rgba[at..at + 4].fill(0);
                continue;
            }
            for channel in 0..3 {
                rgba[at + channel] =
                    ((premultiplied[channel] / alpha).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
            rgba[at + 3] = (alpha.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    }
}

/// Upper bound on glyph layers composited in one raster: every family
/// member visible at once, with the not-allowed member's three glyphs.
pub(crate) const MAX_LAYERS: usize = 12;

impl Prepared<'_> {
    fn local(&self, p: [f32; 2]) -> [f32; 2] {
        [
            (self.cos * p[0] + self.sin * p[1]) / self.scale[0],
            (-self.sin * p[0] + self.cos * p[1]) / self.scale[1],
        ]
    }

    /// Signed distance in world units. With unequal axis scales the local
    /// distance times the smaller scale is a lower bound, so the result stays
    /// 1-Lipschitz and the `reach` early-out remains exact.
    fn distance(&self, p: [f32; 2]) -> f32 {
        culled_distance(self.glyph, &self.culling, self.local(p)) * self.min_scale
    }

    fn seam_distance(&self, p: [f32; 2]) -> f32 {
        let local = self.local(p);
        let mut d = f32::INFINITY;
        for seam in self.glyph.seams {
            d = d.min(capsule_distance(local, seam));
        }
        d * self.min_scale
    }
}

#[cfg(test)]
pub(crate) fn glyph_distance(glyph: Glyph<'_>, p: [f32; 2]) -> f32 {
    let mut d = polygons_distance(glyph, p);
    for capsule in glyph.capsules {
        d = d.min(capsule_distance(p, capsule));
    }
    d
}

/// [`glyph_distance`] with capsule groups skipped when their box cannot beat
/// the running minimum.
fn culled_distance(glyph: Glyph<'_>, culling: &Culling, p: [f32; 2]) -> f32 {
    let mut d = polygons_distance(glyph, p);
    for index in 0..culling.count {
        // The margin absorbs rounding between the box bound and the exact
        // segment distance, so a skipped group can never have held the minimum.
        if culling.chunks[index].distance(p) - culling.radius[index] > d + CULL_MARGIN {
            continue;
        }
        for capsule in &glyph.capsules[chunk_range(index, culling.count, glyph.capsules.len())] {
            d = d.min(capsule_distance(p, capsule));
        }
    }
    d
}

const CULL_MARGIN: f32 = 1.0e-3;

/// Capsules of group `index`; any remainder beyond `MAX_CHUNKS` groups joins
/// the last.
fn chunk_range(index: usize, count: usize, len: usize) -> core::ops::Range<usize> {
    let end = if index + 1 == count {
        len
    } else {
        (index + 1) * CHUNK
    };
    index * CHUNK..end
}

/// Union of the outlines; the first is taken as-is so a single-outline glyph
/// (the Dart) computes exactly what it always has.
fn polygons_distance(glyph: Glyph<'_>, p: [f32; 2]) -> f32 {
    let mut d = match glyph.polygons.split_first() {
        Some((first, rest)) => {
            let mut d = polygon_distance(p, first);
            for polygon in rest {
                d = d.min(polygon_distance(p, polygon));
            }
            d
        }
        None => f32::INFINITY,
    };
    for arc in glyph.arcs {
        d = d.min(arc_distance(p, arc));
    }
    d
}

fn arc_distance(p: [f32; 2], arc: &Arc) -> f32 {
    let [cx, cy, radius, half_width, mx, my, cos_half, sin_half] = *arc;
    let dx = p[0] - cx;
    let dy = p[1] - cy;
    let length = sqrt_f32(dx * dx + dy * dy);
    // Inside the swept sector the band is the ring; beyond it, the nearer
    // rounded end. No angle is ever computed.
    if dx * mx + dy * my >= length * cos_half {
        return (length - radius).abs() - half_width;
    }
    let mut d = f32::INFINITY;
    for side in [1.0, -1.0] {
        let ex = cx + radius * (mx * cos_half - side * my * sin_half);
        let ey = cy + radius * (my * cos_half + side * mx * sin_half);
        d = d.min(sqrt_f32(
            (p[0] - ex) * (p[0] - ex) + (p[1] - ey) * (p[1] - ey),
        ));
    }
    d - half_width
}

fn capsule_distance(p: [f32; 2], capsule: &Capsule) -> f32 {
    let a = [capsule[0], capsule[1]];
    let edge = [capsule[2] - a[0], capsule[3] - a[1]];
    let from = [p[0] - a[0], p[1] - a[1]];
    let length_squared = (edge[0] * edge[0] + edge[1] * edge[1]).max(1.0e-6);
    let t = ((from[0] * edge[0] + from[1] * edge[1]) / length_squared).clamp(0.0, 1.0);
    let nearest = [from[0] - edge[0] * t, from[1] - edge[1] * t];
    sqrt_f32(nearest[0] * nearest[0] + nearest[1] * nearest[1]) - capsule[4]
}

pub(crate) fn polygon_distance(p: [f32; 2], path: &[[f32; 2]]) -> f32 {
    let mut distance_squared = squared_distance(p, path[0]);
    let mut sign = 1.0;
    let mut previous = path.len() - 1;
    for current in 0..path.len() {
        let vi = path[current];
        let vj = path[previous];
        let edge = [vj[0] - vi[0], vj[1] - vi[1]];
        let from_vertex = [p[0] - vi[0], p[1] - vi[1]];
        let length_squared = squared_distance(vj, vi).max(1.0e-6);
        let projection = ((from_vertex[0] * edge[0] + from_vertex[1] * edge[1]) / length_squared)
            .clamp(0.0, 1.0);
        let nearest = [
            from_vertex[0] - edge[0] * projection,
            from_vertex[1] - edge[1] * projection,
        ];
        distance_squared = distance_squared.min(nearest[0] * nearest[0] + nearest[1] * nearest[1]);
        let c0 = p[1] >= vi[1];
        let c1 = p[1] < vj[1];
        let c2 = edge[0] * from_vertex[1] > edge[1] * from_vertex[0];
        if (c0 && c1 && c2) || (!c0 && !c1 && !c2) {
            sign = -sign;
        }
        previous = current;
    }
    sign * sqrt_f32(distance_squared)
}

fn squared_distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy
}

fn coverage(distance: f32) -> f32 {
    (0.5 - distance).clamp(0.0, 1.0)
}

fn smoothstep(low: f32, high: f32, value: f32) -> f32 {
    let t = ((value - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn over(acc: &mut [f32; 3], alpha: &mut f32, color: [f32; 3], layer_alpha: f32) {
    let behind = 1.0 - layer_alpha;
    for channel in 0..3 {
        acc[channel] = color[channel] * layer_alpha + acc[channel] * behind;
    }
    *alpha = layer_alpha + *alpha * behind;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Culling is an optimization only: over a dense grid, inside and outside
    /// every family glyph, the culled distance is bit-identical to the full one.
    #[test]
    fn culled_distance_is_bit_identical_to_the_full_union() {
        let hand = crate::shapes::GRAB;
        let palm: [&[[f32; 2]]; 1] = [&hand.palm];
        let heads = crate::shapes::RESIZE_NWSE.heads;
        let arrow: [&[[f32; 2]]; 2] = [&heads[0], &heads[1]];
        let glyphs = [
            crate::dart::DART_GLYPH,
            Glyph {
                polygons: &[],
                capsules: &crate::shapes::TEXT,
                seams: &[],
                keyline: OUTLINE_WIDTH,
                arcs: &[],
                body: BODY,
                shadow: true,
            },
            Glyph {
                polygons: &palm,
                capsules: &hand.strokes,
                seams: &hand.seams,
                keyline: OUTLINE_WIDTH,
                arcs: &[],
                body: BODY,
                shadow: true,
            },
            Glyph {
                polygons: &arrow,
                capsules: &crate::shapes::RESIZE_NWSE.shaft,
                seams: &[],
                keyline: OUTLINE_WIDTH,
                arcs: &[],
                body: BODY,
                shadow: true,
            },
        ];
        for glyph in glyphs {
            let culling = Culling::new(glyph);
            let mut inside = 0;
            for y in -200..=200 {
                for x in -200..=200 {
                    let p = [x as f32 * 0.05, y as f32 * 0.05];
                    let full = glyph_distance(glyph, p);
                    assert_eq!(
                        culled_distance(glyph, &culling, p).to_bits(),
                        full.to_bits()
                    );
                    inside += usize::from(full < 0.0);
                }
            }
            assert!(inside > 100);
        }
    }
}
