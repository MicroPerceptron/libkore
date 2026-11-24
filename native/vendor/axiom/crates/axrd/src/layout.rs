/// Fractional anchor point within the parent's rect.
///
/// `(0.0, 0.0)` = top-left, `(1.0, 1.0)` = bottom-right.
/// The anchor determines the reference point from which the
/// pixel [`Offset`] is applied.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Anchor {
    pub x: f32,
    pub y: f32,
}

impl Anchor {
    pub const TOP_LEFT: Self = Self { x: 0.0, y: 0.0 };
    pub const TOP_CENTER: Self = Self { x: 0.5, y: 0.0 };
    pub const TOP_RIGHT: Self = Self { x: 1.0, y: 0.0 };
    pub const CENTER_LEFT: Self = Self { x: 0.0, y: 0.5 };
    pub const CENTER: Self = Self { x: 0.5, y: 0.5 };
    pub const CENTER_RIGHT: Self = Self { x: 1.0, y: 0.5 };
    pub const BOTTOM_LEFT: Self = Self { x: 0.0, y: 1.0 };
    pub const BOTTOM_CENTER: Self = Self { x: 0.5, y: 1.0 };
    pub const BOTTOM_RIGHT: Self = Self { x: 1.0, y: 1.0 };
}

/// Pixel offset from the anchor point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Offset {
    pub x: i32,
    pub y: i32,
}

impl Offset {
    pub const ZERO: Self = Self { x: 0, y: 0 };
}

pub use axnn::raster::Rect;

/// Resolve a child's position relative to `parent`.
///
/// `child_pos = parent_origin + anchor * parent_size + offset`
pub fn resolve_child(parent: &Rect, child: &LayoutParams) -> Rect {
    let ax = parent.x + (parent.w as f32 * child.anchor.x) as i32;
    let ay = parent.y + (parent.h as f32 * child.anchor.y) as i32;
    Rect {
        x: ax + child.offset.x,
        y: ay + child.offset.y,
        w: child.w,
        h: child.h,
    }
}

/// Full layout parameters for a scene node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayoutParams {
    pub anchor: Anchor,
    pub offset: Offset,
    pub w: u16,
    pub h: u16,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Rect intersection / union ───────────────────────────────

    #[test]
    fn intersects_overlap() {
        let a = Rect {
            x: 0,
            y: 0,
            w: 10,
            h: 10,
        };
        let b = Rect {
            x: 5,
            y: 5,
            w: 10,
            h: 10,
        };
        assert!(a.intersects(&b));
        assert!(b.intersects(&a));
    }

    #[test]
    fn intersects_adjacent_no_overlap() {
        let a = Rect {
            x: 0,
            y: 0,
            w: 10,
            h: 10,
        };
        let b = Rect {
            x: 10,
            y: 0,
            w: 10,
            h: 10,
        };
        assert!(!a.intersects(&b));
    }

    #[test]
    fn intersects_zero_area() {
        let a = Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        };
        let b = Rect {
            x: 0,
            y: 0,
            w: 10,
            h: 10,
        };
        assert!(!a.intersects(&b));
    }

    #[test]
    fn union_two_rects() {
        let a = Rect {
            x: 0,
            y: 0,
            w: 10,
            h: 10,
        };
        let b = Rect {
            x: 20,
            y: 5,
            w: 10,
            h: 10,
        };
        let u = a.union(&b);
        assert_eq!(
            u,
            Rect {
                x: 0,
                y: 0,
                w: 30,
                h: 15
            }
        );
    }

    #[test]
    fn union_with_zero() {
        let a = Rect {
            x: 5,
            y: 5,
            w: 10,
            h: 10,
        };
        assert_eq!(a.union(&Rect::ZERO), a);
        assert_eq!(Rect::ZERO.union(&a), a);
    }

    #[test]
    fn intersect_overlap() {
        let a = Rect {
            x: 0,
            y: 0,
            w: 20,
            h: 20,
        };
        let b = Rect {
            x: 10,
            y: 5,
            w: 20,
            h: 20,
        };
        assert_eq!(
            a.intersect(&b),
            Rect {
                x: 10,
                y: 5,
                w: 10,
                h: 15
            }
        );
    }

    #[test]
    fn intersect_no_overlap() {
        let a = Rect {
            x: 0,
            y: 0,
            w: 10,
            h: 10,
        };
        let b = Rect {
            x: 20,
            y: 20,
            w: 10,
            h: 10,
        };
        assert_eq!(a.intersect(&b), Rect::ZERO);
    }

    // ── Layout resolution ───────────────────────────────────────

    #[test]
    fn resolve_top_left_no_offset() {
        let parent = Rect {
            x: 10,
            y: 20,
            w: 100,
            h: 80,
        };
        let child = LayoutParams {
            anchor: Anchor::TOP_LEFT,
            offset: Offset::ZERO,
            w: 30,
            h: 20,
        };
        let r = resolve_child(&parent, &child);
        assert_eq!(
            r,
            Rect {
                x: 10,
                y: 20,
                w: 30,
                h: 20
            }
        );
    }

    #[test]
    fn resolve_center_no_offset() {
        let parent = Rect {
            x: 0,
            y: 0,
            w: 100,
            h: 100,
        };
        let child = LayoutParams {
            anchor: Anchor::CENTER,
            offset: Offset::ZERO,
            w: 20,
            h: 20,
        };
        let r = resolve_child(&parent, &child);
        assert_eq!(
            r,
            Rect {
                x: 50,
                y: 50,
                w: 20,
                h: 20
            }
        );
    }

    #[test]
    fn resolve_with_offset() {
        let parent = Rect {
            x: 0,
            y: 0,
            w: 200,
            h: 100,
        };
        let child = LayoutParams {
            anchor: Anchor::TOP_LEFT,
            offset: Offset { x: 10, y: 5 },
            w: 50,
            h: 30,
        };
        let r = resolve_child(&parent, &child);
        assert_eq!(
            r,
            Rect {
                x: 10,
                y: 5,
                w: 50,
                h: 30
            }
        );
    }

    #[test]
    fn resolve_bottom_right_anchor() {
        let parent = Rect {
            x: 0,
            y: 0,
            w: 100,
            h: 100,
        };
        let child = LayoutParams {
            anchor: Anchor::BOTTOM_RIGHT,
            offset: Offset { x: -20, y: -20 },
            w: 20,
            h: 20,
        };
        let r = resolve_child(&parent, &child);
        assert_eq!(
            r,
            Rect {
                x: 80,
                y: 80,
                w: 20,
                h: 20
            }
        );
    }

    #[test]
    fn resolve_nested_parent_offset() {
        let parent = Rect {
            x: 50,
            y: 50,
            w: 200,
            h: 200,
        };
        let child = LayoutParams {
            anchor: Anchor::CENTER,
            offset: Offset::ZERO,
            w: 40,
            h: 40,
        };
        let r = resolve_child(&parent, &child);
        assert_eq!(
            r,
            Rect {
                x: 150,
                y: 150,
                w: 40,
                h: 40
            }
        );
    }
}
