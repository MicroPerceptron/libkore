/// Easing functions and interpolation primitives for animation.
///
/// All easing functions map `t ∈ [0.0, 1.0]` → `[0.0, 1.0]`.
/// They are pure math — no allocations, no state, `no_std`-safe.
///
/// Typical usage: compute `t = elapsed / duration`, apply an easing
/// function, then use `lerp_*` to interpolate between start/end values.

/// Linear (no easing).
#[inline]
pub fn linear(t: f32) -> f32 {
    t
}

/// Quadratic ease-in: starts slow, accelerates.
#[inline]
pub fn ease_in_quad(t: f32) -> f32 {
    t * t
}

/// Quadratic ease-out: starts fast, decelerates.
#[inline]
pub fn ease_out_quad(t: f32) -> f32 {
    t * (2.0 - t)
}

/// Quadratic ease-in-out: slow start and end, fast middle.
#[inline]
pub fn ease_in_out_quad(t: f32) -> f32 {
    if t < 0.5 {
        2.0 * t * t
    } else {
        -1.0 + (4.0 - 2.0 * t) * t
    }
}

/// Cubic ease-in.
#[inline]
pub fn ease_in_cubic(t: f32) -> f32 {
    t * t * t
}

/// Cubic ease-out.
#[inline]
pub fn ease_out_cubic(t: f32) -> f32 {
    let u = t - 1.0;
    u * u * u + 1.0
}

/// Cubic ease-in-out.
#[inline]
pub fn ease_in_out_cubic(t: f32) -> f32 {
    if t < 0.5 {
        4.0 * t * t * t
    } else {
        let u = 2.0 * t - 2.0;
        0.5 * u * u * u + 1.0
    }
}

/// Cubic Bézier easing with control points `(x1, y1)` and `(x2, y2)`.
///
/// Matches CSS `cubic-bezier(x1, y1, x2, y2)`. The curve starts at
/// `(0, 0)` and ends at `(1, 1)`. Given input `t` (time fraction),
/// solves for the `x` parameter of the Bézier, then evaluates `y`.
///
/// Uses Newton's method (5 iterations) — fast and accurate for
/// well-behaved control points.
pub fn cubic_bezier(x1: f32, y1: f32, x2: f32, y2: f32, t: f32) -> f32 {
    if t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }

    // Solve for the Bézier parameter `s` such that bezier_x(s) = t.
    let mut s = t; // initial guess
    for _ in 0..5 {
        let bx = bezier_component(x1, x2, s);
        let dx = bezier_derivative(x1, x2, s);
        if dx.abs() < 1e-7 {
            break;
        }
        s -= (bx - t) / dx;
        s = s.clamp(0.0, 1.0);
    }

    bezier_component(y1, y2, s)
}

/// Evaluate one component of a cubic Bézier: `B(s) = 3(1-s)²·s·p1 + 3(1-s)·s²·p2 + s³`.
#[inline]
fn bezier_component(p1: f32, p2: f32, s: f32) -> f32 {
    let u = 1.0 - s;
    3.0 * u * u * s * p1 + 3.0 * u * s * s * p2 + s * s * s
}

/// Derivative of one Bézier component: `B'(s)`.
#[inline]
fn bezier_derivative(p1: f32, p2: f32, s: f32) -> f32 {
    let u = 1.0 - s;
    3.0 * u * u * p1 + 6.0 * u * s * (p2 - p1) + 3.0 * s * s * (1.0 - p2)
}

// ── Interpolation helpers ──────────────────────────────────────

/// Linearly interpolate between two `f32` values.
#[inline]
pub fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Linearly interpolate between two `i32` values.
#[inline]
pub fn lerp_i32(a: i32, b: i32, t: f32) -> i32 {
    (a as f32 + (b - a) as f32 * t) as i32
}

/// Linearly interpolate between two `u8` values (e.g. opacity).
#[inline]
pub fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t) as u8
}

/// Linearly interpolate between two XRGB colours (`0x00RRGGBB`).
///
/// Interpolates each channel independently. Result is `0x00RRGGBB`.
#[inline]
pub fn lerp_color(a: u32, b: u32, t: f32) -> u32 {
    let r = lerp_u8(((a >> 16) & 0xFF) as u8, ((b >> 16) & 0xFF) as u8, t);
    let g = lerp_u8(((a >> 8) & 0xFF) as u8, ((b >> 8) & 0xFF) as u8, t);
    let bl = lerp_u8((a & 0xFF) as u8, (b & 0xFF) as u8, t);
    (r as u32) << 16 | (g as u32) << 8 | bl as u32
}

// ── Tween state ────────────────────────────────────────────────

/// A lightweight tween tracker.
///
/// Stores start/end times and provides a normalized progress value.
/// No allocations — intended to be stored inline in your animation
/// state struct. The easing function is applied externally so that
/// `Tween` stays generic (one tween can drive position, opacity,
/// color, etc. with different easing curves).
///
/// ```ignore
/// let mut tw = Tween::new(0, 300); // 300ms duration
/// // Each frame:
/// tw.tick(frame_time_ms);
/// let t = ease_out_cubic(tw.progress());
/// scene.set_opacity(node, lerp_u8(0, 255, t));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tween {
    /// Start time (inclusive), in caller-chosen units (ms, frames, etc.).
    pub start: u32,
    /// End time (exclusive).
    pub end: u32,
    /// Current time.
    pub now: u32,
}

impl Tween {
    /// Create a new tween from `start` to `start + duration`.
    #[inline]
    pub const fn new(start: u32, duration: u32) -> Self {
        Self {
            start,
            end: start + duration,
            now: start,
        }
    }

    /// Advance the current time.
    #[inline]
    pub fn tick(&mut self, now: u32) {
        self.now = now;
    }

    /// Normalized progress `[0.0, 1.0]`. Clamped at boundaries.
    #[inline]
    pub fn progress(&self) -> f32 {
        if self.now <= self.start {
            return 0.0;
        }
        if self.now >= self.end {
            return 1.0;
        }
        let elapsed = (self.now - self.start) as f32;
        let duration = (self.end - self.start) as f32;
        elapsed / duration
    }

    /// Whether the tween has finished (`now >= end`).
    #[inline]
    pub fn done(&self) -> bool {
        self.now >= self.end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Easing functions ───────────────────────────────────────

    #[test]
    fn linear_identity() {
        assert_eq!(linear(0.0), 0.0);
        assert_eq!(linear(0.5), 0.5);
        assert_eq!(linear(1.0), 1.0);
    }

    #[test]
    fn ease_in_quad_endpoints() {
        assert_eq!(ease_in_quad(0.0), 0.0);
        assert_eq!(ease_in_quad(1.0), 1.0);
        // Midpoint should be < 0.5 (starts slow).
        assert!(ease_in_quad(0.5) < 0.5);
    }

    #[test]
    fn ease_out_quad_endpoints() {
        assert_eq!(ease_out_quad(0.0), 0.0);
        assert_eq!(ease_out_quad(1.0), 1.0);
        // Midpoint should be > 0.5 (starts fast).
        assert!(ease_out_quad(0.5) > 0.5);
    }

    #[test]
    fn ease_in_out_quad_symmetry() {
        let a = ease_in_out_quad(0.25);
        let b = 1.0 - ease_in_out_quad(0.75);
        assert!((a - b).abs() < 1e-6);
    }

    #[test]
    fn cubic_bezier_linear() {
        // cubic-bezier(0, 0, 1, 1) ≈ linear.
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let y = cubic_bezier(0.0, 0.0, 1.0, 1.0, t);
            assert!((y - t).abs() < 0.01, "t={t}, y={y}");
        }
    }

    #[test]
    fn cubic_bezier_css_ease() {
        // CSS "ease" = cubic-bezier(0.25, 0.1, 0.25, 1.0).
        let y = cubic_bezier(0.25, 0.1, 0.25, 1.0, 0.5);
        // Midpoint of CSS ease is approximately 0.8.
        assert!(y > 0.6 && y < 0.95, "got {y}");
    }

    #[test]
    fn cubic_bezier_boundary() {
        assert_eq!(cubic_bezier(0.42, 0.0, 0.58, 1.0, 0.0), 0.0);
        assert_eq!(cubic_bezier(0.42, 0.0, 0.58, 1.0, 1.0), 1.0);
    }

    // ── Lerp ───────────────────────────────────────────────────

    #[test]
    fn lerp_f32_midpoint() {
        assert_eq!(lerp_f32(0.0, 10.0, 0.5), 5.0);
    }

    #[test]
    fn lerp_i32_works() {
        assert_eq!(lerp_i32(0, 100, 0.0), 0);
        assert_eq!(lerp_i32(0, 100, 1.0), 100);
        assert_eq!(lerp_i32(0, 100, 0.5), 50);
    }

    #[test]
    fn lerp_u8_opacity() {
        assert_eq!(lerp_u8(0, 255, 0.0), 0);
        assert_eq!(lerp_u8(0, 255, 1.0), 255);
        assert_eq!(lerp_u8(0, 255, 0.5), 127);
    }

    #[test]
    fn lerp_color_midpoint() {
        // Red (0xFF0000) to blue (0x0000FF) at midpoint.
        let c = lerp_color(0x00FF0000, 0x000000FF, 0.5);
        let r = (c >> 16) & 0xFF;
        let g = (c >> 8) & 0xFF;
        let b = c & 0xFF;
        assert_eq!(r, 127);
        assert_eq!(g, 0);
        assert_eq!(b, 127);
    }

    // ── Tween ──────────────────────────────────────────────────

    #[test]
    fn tween_progress() {
        let mut tw = Tween::new(100, 200); // 100..300
        assert_eq!(tw.progress(), 0.0);
        assert!(!tw.done());

        tw.tick(200); // midpoint
        assert!((tw.progress() - 0.5).abs() < 1e-6);
        assert!(!tw.done());

        tw.tick(300); // end
        assert_eq!(tw.progress(), 1.0);
        assert!(tw.done());
    }

    #[test]
    fn tween_clamps() {
        let mut tw = Tween::new(10, 5); // 10..15
        tw.tick(5); // before start
        assert_eq!(tw.progress(), 0.0);
        tw.tick(100); // way past end
        assert_eq!(tw.progress(), 1.0);
    }
}
