//! Bounded damage accumulation.
//!
//! A frame's damage is a small set of rects rather than a single union so a
//! flyout opening in one corner and a clock ticking in the other do not
//! repaint the whole diagonal between them. The list is fixed-capacity and
//! `const`-constructible; when more distinct rects arrive than it can hold it
//! collapses to the union and stays collapsed until [`DamageList::clear`].

use crate::layout::Rect;

/// A bounded list of damage rects that degrades to a single union.
///
/// Zero-area rects are ignored. A rect already covered by a listed rect is
/// dropped, and a new rect that covers listed rects replaces them, so the
/// list holds a tight, non-nested set. Once `M` distinct rects are exceeded
/// the list collapses to one union rect and every later push widens it.
#[derive(Debug, Clone, Copy)]
pub struct DamageList<const M: usize> {
    rects: [Rect; M],
    len: u8,
    collapsed: bool,
}

impl<const M: usize> DamageList<M> {
    /// An empty list.
    pub const fn new() -> Self {
        const {
            assert!(M >= 1 && M <= 255, "DamageList capacity must be 1..=255");
        }
        DamageList {
            rects: [Rect::ZERO; M],
            len: 0,
            collapsed: false,
        }
    }

    /// The accumulated rects (a single union rect once collapsed).
    #[inline]
    pub fn rects(&self) -> &[Rect] {
        &self.rects[..self.len as usize]
    }

    /// Whether the list overflowed and now holds only a union.
    #[inline]
    pub const fn is_collapsed(&self) -> bool {
        self.collapsed
    }

    /// Whether nothing has been damaged.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The bounding union of every listed rect (`Rect::ZERO` when empty).
    pub fn union(&self) -> Rect {
        let mut acc = Rect::ZERO;
        for rect in self.rects() {
            acc = acc.union(rect);
        }
        acc
    }

    /// Forget all damage and leave the collapsed state.
    pub fn clear(&mut self) {
        self.len = 0;
        self.collapsed = false;
    }

    /// Record `rect` as damaged.
    pub fn push(&mut self, rect: Rect) {
        if rect.w == 0 || rect.h == 0 {
            return;
        }
        if self.collapsed {
            self.rects[0] = self.rects[0].union(&rect);
            return;
        }
        let len = self.len as usize;
        for held in &self.rects[..len] {
            if held.contains_rect(&rect) {
                return;
            }
        }
        // Drop entries the new rect swallows, compacting in place.
        let mut kept = 0usize;
        for i in 0..len {
            if !rect.contains_rect(&self.rects[i]) {
                self.rects[kept] = self.rects[i];
                kept += 1;
            }
        }
        if kept < M {
            self.rects[kept] = rect;
            self.len = (kept + 1) as u8;
            return;
        }
        let mut acc = rect;
        for held in &self.rects[..kept] {
            acc = acc.union(held);
        }
        self.rects[0] = acc;
        self.len = 1;
        self.collapsed = true;
    }
}

impl<const M: usize> Default for DamageList<M> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(x: i32, y: i32, w: u16, h: u16) -> Rect {
        Rect { x, y, w, h }
    }

    #[test]
    fn distinct_rects_are_listed_exactly() {
        let mut d = DamageList::<4>::new();
        d.push(r(0, 0, 10, 10));
        d.push(r(100, 100, 10, 10));
        assert_eq!(d.rects(), &[r(0, 0, 10, 10), r(100, 100, 10, 10)]);
        assert!(!d.is_collapsed());
        assert_eq!(d.union(), r(0, 0, 110, 110));
    }

    #[test]
    fn zero_area_and_covered_rects_are_dropped() {
        let mut d = DamageList::<4>::new();
        d.push(r(0, 0, 0, 10));
        assert!(d.is_empty());
        d.push(r(0, 0, 10, 10));
        d.push(r(2, 2, 3, 3));
        assert_eq!(d.rects(), &[r(0, 0, 10, 10)]);
        // A covering rect replaces what it swallows.
        d.push(r(0, 0, 20, 20));
        assert_eq!(d.rects(), &[r(0, 0, 20, 20)]);
    }

    #[test]
    fn overflow_collapses_to_union() {
        let mut d = DamageList::<2>::new();
        d.push(r(0, 0, 1, 1));
        d.push(r(10, 0, 1, 1));
        d.push(r(0, 10, 1, 1));
        assert!(d.is_collapsed());
        assert_eq!(d.rects(), &[r(0, 0, 11, 11)]);
        d.push(r(50, 50, 1, 1));
        assert_eq!(d.rects(), &[r(0, 0, 51, 51)]);
        d.clear();
        assert!(!d.is_collapsed());
        assert!(d.is_empty());
    }
}
