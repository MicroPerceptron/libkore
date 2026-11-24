//! Borrowed 4 -> 16 -> 64 voxel occupancy hierarchy with exact coarse skips.

use core::simd::{Mask, Simd};

use super::{BRICK, MISS, brick_trace, brick_trace_packet_fine, sanitize};

/// A borrowed occupancy hierarchy over the leaf words consumed by
/// [`brick_trace`]. Construction validates descriptor sizes once; tracing
/// allocates no storage.
///
/// `level16` contains one word per 16-voxel cube, with ceil-divided dimensions
/// `ceil(dims / 16)`. Its bit `x + 4*y + 16*z` describes whether the corresponding
/// 4-voxel leaf word is nonempty. `level64` has dimensions `ceil(dims / 64)` and
/// uses the same bit layout for its 16-voxel children. All arrays are x-major.
/// Boundary children outside the grid must have zero bits. The caller must
/// keep both levels consistent with the leaf words; construction checks
/// descriptors, not their occupancy contents.
///
/// Empty parents skip occupancy lookups, but retain the scalar tracer's exact
/// per-axis, 4-voxel boundary-addition recurrence and x/y/z tie order. There is
/// no macro-step distance approximation. Arithmetic overflow for valid rays
/// falls back to the scalar tracer. Nonfinite origins/directions and negative
/// or NaN limits return [`MISS`]; positive infinite limits remain supported.
#[derive(Clone, Copy, Debug)]
pub struct BrickHierarchy<'a> {
    bricks: &'a [u64],
    dims: [u32; 3],
    brick_dims: [i32; 3],
    level16: &'a [u64],
    dims16: [usize; 3],
    level64: &'a [u64],
    dims64: [usize; 3],
}

impl<'a> BrickHierarchy<'a> {
    /// Borrow a complete, consistent hierarchy. Dimensions must be positive
    /// multiples of four, their product must fit the scalar tracer's signed
    /// linear-index representation, and every slice must cover its level.
    /// Invalid descriptors panic in both debug and release builds.
    pub fn new(bricks: &'a [u64], dims: [u32; 3], level16: &'a [u64], level64: &'a [u64]) -> Self {
        assert!(dims.iter().all(|&extent| extent > 0 && extent % BRICK == 0));
        let cells = dims
            .iter()
            .try_fold(1_u32, |count, &extent| count.checked_mul(extent))
            .filter(|&count| count <= i32::MAX as u32)
            .expect("voxel hierarchy grid index overflow");
        let dims16 = dims.map(|extent| extent.div_ceil(16) as usize);
        let dims64 = dims.map(|extent| extent.div_ceil(64) as usize);
        assert!(bricks.len() >= cells as usize / 64);
        assert!(level16.len() >= dims16.iter().product());
        assert!(level64.len() >= dims64.iter().product());
        Self {
            bricks,
            dims,
            brick_dims: dims.map(|extent| (extent / BRICK) as i32),
            level16,
            dims16,
            level64,
            dims64,
        }
    }

    /// Trace one valid ray with the same first-hit distance bits and voxel ID
    /// as [`brick_trace`] over the borrowed leaf grid. Coordinates and directions
    /// are in voxel space; directions need not be normalized. Invalid rays use
    /// the miss behavior documented on [`BrickHierarchy`].
    pub fn trace(&self, origin: [f32; 3], dir: [f32; 3], t_max: f32) -> (f32, u32) {
        let fallback = || brick_trace(self.bricks, self.dims, origin, dir, t_max);
        if t_max.is_nan()
            || t_max < 0.0
            || !origin.iter().chain(&dir).all(|value| value.is_finite())
        {
            return MISS;
        }
        let direction = dir.map(sanitize);
        let inverse = direction.map(|value| 1.0 / value);
        let (mut t0, mut t1) = (0.0_f32, t_max);
        for axis in 0..3 {
            let ta = (0.0 - origin[axis]) * inverse[axis];
            let tb = (self.dims[axis] as f32 - origin[axis]) * inverse[axis];
            let (lo, hi) = if ta < tb { (ta, tb) } else { (tb, ta) };
            t0 = t0.max(lo);
            t1 = t1.min(hi);
        }
        if t0 > t1 {
            return MISS;
        }
        if !t0.is_finite() {
            return fallback();
        }
        let mut walk = Walk {
            cell: [0; 3],
            step: [0; 3],
            next: [0.0; 3],
            delta: [0.0; 3],
            time: t0,
        };
        let edge = BRICK as f32;
        for axis in 0..3 {
            let p = origin[axis] + t0 * direction[axis];
            let cell = ((p / edge) as i32).clamp(0, self.brick_dims[axis] - 1);
            walk.cell[axis] = cell;
            walk.step[axis] = if direction[axis] > 0.0 { 1 } else { -1 };
            walk.delta[axis] = edge / direction[axis].abs();
            let boundary = if direction[axis] > 0.0 {
                (cell + 1) as f32 * edge
            } else {
                cell as f32 * edge
            };
            walk.next[axis] = t0 + (boundary - p) / direction[axis];
        }
        if !walk.next.iter().all(|value| value.is_finite()) {
            return fallback();
        }
        // abs(1/d) retains the exact bits of 1/abs(d) for these finite,
        // sanitized directions, including subnormal reciprocal results.
        let fine_delta = inverse.map(f32::abs);
        loop {
            let cell64 = walk.cell.map(|coordinate| coordinate as usize / 16);
            let parent64 = self.level64[index(cell64, self.dims64)];
            let empty_edge = if parent64 == 0 {
                16 // 16 leaf bricks = 64 fine voxels.
            } else {
                let cell16 = walk.cell.map(|coordinate| coordinate as usize / 4);
                if parent64 >> child_bit(cell16) & 1 == 0 {
                    4 // Four leaf bricks = 16 fine voxels.
                } else {
                    let parent16 = self.level16[index(cell16, self.dims16)];
                    if parent16 >> child_bit(walk.cell.map(|coordinate| coordinate as usize)) & 1
                        != 0
                    {
                        let leaf = index(
                            walk.cell.map(|coordinate| coordinate as usize),
                            self.brick_dims.map(|extent| extent as usize),
                        );
                        let word = self.bricks[leaf];
                        if word != 0
                            && let Some(hit) = brick_trace_packet_fine(
                                word,
                                self.dims,
                                walk.cell,
                                origin,
                                direction,
                                fine_delta,
                                [walk.time, t1],
                            )
                        {
                            return hit;
                        }
                    }
                    1
                }
            };
            let advance = if empty_edge > 1 {
                walk.skip_empty(self.brick_dims, empty_edge, t1)
            } else {
                walk.advance(self.brick_dims, t1)
            };
            match advance {
                Advance::Inside => {}
                Advance::Miss => return MISS,
                Advance::Fallback => return fallback(),
            }
        }
    }

    /// Trace xyz-interleaved rays into caller-owned output slices. Only the
    /// first `n` outputs are written; short slices or an overflowing count panic.
    pub fn trace_n(
        &self,
        origins: &[f32],
        dirs: &[f32],
        n: usize,
        t_max: f32,
        out_t: &mut [f32],
        out_idx: &mut [u32],
    ) {
        let coordinates = n
            .checked_mul(3)
            .expect("voxel hierarchy ray count overflow");
        assert!(origins.len() >= coordinates && dirs.len() >= coordinates);
        assert!(out_t.len() >= n && out_idx.len() >= n);
        for ray in 0..n {
            let offset = ray * 3;
            let origin = [origins[offset], origins[offset + 1], origins[offset + 2]];
            let direction = [dirs[offset], dirs[offset + 1], dirs[offset + 2]];
            (out_t[ray], out_idx[ray]) = self.trace(origin, direction, t_max);
        }
    }

    /// Trace four-ray packets, skipping a shared empty 64³ parent when all
    /// four lanes are active in the same leaf brick. Other states retain the
    /// flat packet traversal, including its SIMD occupied-brick fine walk.
    ///
    /// Each skipped lane uses the same exact coarse recurrence as [`Self::trace`].
    /// Empty 16³ children are not skipped by this experimental path. No heap
    /// storage is allocated; only the first `n` output elements are written.
    /// The slice/count and invalid-ray contracts match [`Self::trace_n`].
    pub fn trace_packet_n(
        &self,
        origins: &[f32],
        dirs: &[f32],
        n: usize,
        t_max: f32,
        out_t: &mut [f32],
        out_idx: &mut [u32],
    ) {
        let coordinates = n
            .checked_mul(3)
            .expect("voxel hierarchy ray count overflow");
        assert!(origins.len() >= coordinates && dirs.len() >= coordinates);
        assert!(out_t.len() >= n && out_idx.len() >= n);
        out_t[..n].fill(MISS.0);
        out_idx[..n].fill(MISS.1);
        if t_max.is_nan() || t_max < 0.0 {
            return;
        }
        for start in (0..n).step_by(4) {
            super::brick_trace_packet4::<true>(
                self.bricks,
                self.dims,
                &origins[start * 3..],
                &dirs[start * 3..],
                (n - start).min(4),
                t_max,
                &mut out_t[start..],
                &mut out_idx[start..],
                Some(self),
            );
        }
    }

    /// Try one empty-64-parent skip from the packet's shared active leaf.
    /// `None` leaves every packet value unchanged. On `Some`, all lanes have
    /// either advanced exactly or retired; the returned mask selects lanes
    /// whose original scalar trace must be replayed by the caller.
    #[inline]
    pub(super) fn skip_packet64(
        &self,
        cell: &mut [Simd<i32, 4>; 3],
        step: [Simd<i32, 4>; 3],
        next: &mut [Simd<f32, 4>; 3],
        delta: [Simd<f32, 4>; 3],
        time: &mut Simd<f32, 4>,
        limit: Simd<f32, 4>,
        active: &mut Mask<i32, 4>,
    ) -> Option<Mask<i32, 4>> {
        debug_assert!(active.all());
        debug_assert!(cell.iter().all(|axis| axis.to_array() == [axis[0]; 4]));
        let parent = cell.map(|axis| axis[0] as usize / 16);
        if self.level64[index(parent, self.dims64)] != 0 {
            return None;
        }
        let mut fallback = Mask::splat(false);
        for lane in 0..4 {
            let mut walk = Walk {
                cell: cell.map(|axis| axis[lane]),
                step: step.map(|axis| axis[lane]),
                next: next.map(|axis| axis[lane]),
                delta: delta.map(|axis| axis[lane]),
                time: time[lane],
            };
            match walk.skip_empty(self.brick_dims, 16, limit[lane]) {
                Advance::Inside => {
                    for axis in 0..3 {
                        cell[axis][lane] = walk.cell[axis];
                        next[axis][lane] = walk.next[axis];
                    }
                    time[lane] = walk.time;
                }
                Advance::Miss => active.set(lane, false),
                Advance::Fallback => {
                    fallback.set(lane, true);
                    active.set(lane, false);
                }
            }
        }
        Some(fallback)
    }
}

#[inline]
fn index(cell: [usize; 3], dims: [usize; 3]) -> usize {
    cell[0] + dims[0] * (cell[1] + dims[1] * cell[2])
}

#[inline]
fn child_bit(cell: [usize; 3]) -> usize {
    cell[0] % 4 + 4 * (cell[1] % 4) + 16 * (cell[2] % 4)
}

#[derive(Clone, Copy, Debug)]
struct Walk {
    cell: [i32; 3],
    step: [i32; 3],
    next: [f32; 3],
    delta: [f32; 3],
    time: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Advance {
    Inside,
    Miss,
    Fallback,
}

#[inline]
fn first_axis(next: [f32; 3]) -> usize {
    if next[0] <= next[1] && next[0] <= next[2] {
        0
    } else if next[1] <= next[2] {
        1
    } else {
        2
    }
}

/// The original rounded-addition recurrence, retained as the exact fallback.
#[inline]
fn repeat_add_replay(mut value: f32, delta: f32, count: u32) -> f32 {
    for _ in 0..count {
        value += delta;
    }
    value
}

/// A guarded shortcut on one uniformly spaced F32 exponent region.
///
/// With round-to-nearest/ties-even, adding a fixed delta advances by a fixed
/// integer ULP count on this grid unless it is a half-ULP tie. In the tie case
/// the first sum has even parity, after which the step is fixed as well. The
/// second-minus-first bit difference is therefore the steady step. Keeping
/// the START, both sums, and predicted endpoint in the same exponent region
/// lets induction keep every intervening rounded addition on that same grid.
/// This also covers gradual-underflow subnormals (exponent zero) and step zero.
#[inline]
fn repeat_add_same_binade(value: f32, delta: f32, count: u32) -> Option<f32> {
    debug_assert!(count >= 3);
    const SIGN: u32 = 0x8000_0000;
    const EXPONENT: u32 = 0x7f80_0000;
    let start = value.to_bits();
    let exponent = start & EXPONENT;
    if start & SIGN != 0 || exponent == EXPONENT || !delta.is_finite() || delta <= 0.0 {
        return None;
    }
    let first = value + delta;
    let second = first + delta;
    let first = first.to_bits();
    let second = second.to_bits();
    if (first | second) & SIGN != 0
        || first & EXPONENT != exponent
        || second & EXPONENT != exponent
        || first < start
    {
        return None;
    }
    let step = second.checked_sub(first)?;
    let advance = step.checked_mul(count.checked_sub(1)?)?;
    let candidate = first.checked_add(advance)?;
    if candidate & SIGN != 0 || candidate & EXPONENT != exponent || candidate < second {
        return None;
    }
    Some(f32::from_bits(candidate))
}

#[inline]
fn repeat_add(value: f32, delta: f32, count: u32) -> f32 {
    match count {
        0 => value,
        1 => value + delta,
        2 => {
            let first = value + delta;
            first + delta
        }
        _ => repeat_add_same_binade(value, delta, count)
            .unwrap_or_else(|| repeat_add_replay(value, delta, count)),
    }
}

/// Count a qualifying prefix of one axis's boundary events without changing
/// its rounded-addition recurrence. The first two events retain literal adds;
/// longer prefixes use the same guarded bit progression as repeat_add above.
/// A rejected shortcut leaves the caller's bounded replay unchanged.
#[inline]
fn count_events_before_exit(
    value: f32,
    delta: f32,
    exit: f32,
    inclusive: bool,
    max_steps: u32,
) -> Option<(u32, f32)> {
    const SIGN: u32 = 0x8000_0000;
    const EXPONENT: u32 = 0x7f80_0000;
    let start = value.to_bits();
    let exit_bits = exit.to_bits();
    let exponent = start & EXPONENT;
    if (start | exit_bits) & SIGN != 0
        || exponent == EXPONENT
        || exit_bits & EXPONENT == EXPONENT
        || !delta.is_finite()
        || delta <= 0.0
    {
        return None;
    }
    let before_exit = |time| time < exit || (inclusive && time == exit);
    if !before_exit(value) {
        return Some((0, value));
    }
    if max_steps == 0 {
        return None;
    }
    let first = value + delta;
    if !before_exit(first) {
        return first.is_finite().then_some((1, first));
    }
    if max_steps == 1 {
        return None;
    }
    let second = first + delta;
    if !before_exit(second) {
        return second.is_finite().then_some((2, second));
    }
    if max_steps == 2 {
        return None;
    }
    let first = first.to_bits();
    let second = second.to_bits();
    if (first | second) & SIGN != 0
        || first & EXPONENT != exponent
        || second & EXPONENT != exponent
        || first < start
    {
        return None;
    }
    let step = second.checked_sub(first)?;
    if step == 0 {
        // Keep the replay's defensive exhaustion/fallback behavior for a
        // qualifying sequence that stops progressing at this F32 spacing.
        return None;
    }
    let bound = if inclusive {
        exit_bits
    } else {
        exit_bits.checked_sub(1)?
    };
    // v0 qualifies separately; v1 is the first point on the steady grid.
    // If bound==first, both v0 and v1 count, hence 2 rather than 1 here.
    let count = bound
        .checked_sub(first)?
        .checked_div(step)?
        .checked_add(2)?
        .min(max_steps);
    let candidate = first.checked_add(step.checked_mul(count.checked_sub(1)?)?)?;
    if candidate & SIGN != 0 || candidate & EXPONENT != exponent || candidate < second {
        return None;
    }
    let next = f32::from_bits(candidate);
    if before_exit(next) {
        // Reaching max_steps while another event still qualifies must retain
        // the original replay and its post-loop fallback, not accept a skip.
        return None;
    }
    Some((count, next))
}

impl Walk {
    /// One ordinary coarse step, preserving the scalar operation order.
    #[inline]
    fn advance(&mut self, dims: [i32; 3], limit: f32) -> Advance {
        let axis = first_axis(self.next);
        self.time = self.next[axis];
        if self.time > limit {
            return Advance::Miss;
        }
        self.cell[axis] += self.step[axis];
        if self.cell[axis] < 0 || self.cell[axis] >= dims[axis] {
            return Advance::Miss;
        }
        self.next[axis] += self.delta[axis];
        if self.next[axis].is_finite() {
            Advance::Inside
        } else {
            Advance::Fallback
        }
    }

    /// Skip one known-empty parent without changing the leaf DDA recurrence.
    /// Every loop below is bounded by the parent edge (at most 16 leaf cells).
    #[inline]
    fn skip_empty(&mut self, dims: [i32; 3], edge: i32, limit: f32) -> Advance {
        let mut steps = [0; 3];
        let mut exits = self.next;
        for axis in 0..3 {
            let lower = self.cell[axis] / edge * edge;
            let upper = (lower + edge).min(dims[axis]);
            steps[axis] = if self.step[axis] > 0 {
                upper - self.cell[axis]
            } else {
                self.cell[axis] - lower + 1
            };
            // This preserves each rounded addition, either by a proved bit
            // step within one exponent region or by replaying the recurrence.
            exits[axis] = repeat_add(self.next[axis], self.delta[axis], (steps[axis] - 1) as u32);
        }
        if !exits.iter().all(|value| value.is_finite()) {
            return Advance::Fallback;
        }
        let exit_axis = first_axis(exits);
        let exit_time = exits[exit_axis];
        if exit_time > limit {
            return Advance::Miss;
        }
        let exit_cell = self.cell[exit_axis] + self.step[exit_axis] * steps[exit_axis];
        if exit_cell < 0 || exit_cell >= dims[exit_axis] {
            return Advance::Miss;
        }
        // Each axis is an independent sequence of rounded additions. Before
        // the chosen exit, consume its strictly earlier events and equal-time
        // events only when that axis has the scalar tie rule's higher priority.
        for axis in 0..3 {
            if axis == exit_axis {
                continue;
            }
            if let Some((count, next)) = count_events_before_exit(
                self.next[axis],
                self.delta[axis],
                exit_time,
                axis < exit_axis,
                steps[axis] as u32,
            ) {
                self.cell[axis] += self.step[axis] * count as i32;
                self.next[axis] = next;
            } else {
                for _ in 0..steps[axis] {
                    let before_exit = self.next[axis] < exit_time
                        || (self.next[axis] == exit_time && axis < exit_axis);
                    if !before_exit {
                        break;
                    }
                    self.cell[axis] += self.step[axis];
                    self.next[axis] += self.delta[axis];
                }
            }
            if self.next[axis] < exit_time || (self.next[axis] == exit_time && axis < exit_axis) {
                // Defensive retirement for unexpected stalled arithmetic;
                // never turn an ambiguous skip into an occupancy miss.
                return Advance::Fallback;
            }
        }
        self.cell[exit_axis] = exit_cell;
        self.time = exit_time;
        self.next[exit_axis] = exit_time + self.delta[exit_axis];
        if self.next.iter().all(|value| value.is_finite()) {
            Advance::Inside
        } else {
            Advance::Fallback
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::super::brick_trace_n;
    use super::*;

    fn random(state: &mut u32) -> u32 {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        *state
    }

    #[test]
    fn repeat_add_rejects_missing_start_and_endpoint_binade_counterexamples() {
        for (value, delta, expected) in [
            (0x4143_28ff, 0x40be_5726, 0x41f0_55dc), // Ordinary scene magnitudes.
            (0x6e43_28ff, 0x6dbe_5726, 0x6ef0_55dc),
        ] {
            let value = f32::from_bits(value);
            let delta = f32::from_bits(delta);
            let first = value + delta;
            let second = first + delta;
            let unguarded = first.to_bits() + 2 * (second.to_bits() - first.to_bits());
            assert_eq!(repeat_add_replay(value, delta, 3).to_bits(), expected);
            assert_eq!(
                unguarded,
                expected - 1,
                "omitting the start guard changes one ULP"
            );
            assert!(repeat_add_same_binade(value, delta, 3).is_none());
            assert_eq!(repeat_add(value, delta, 3).to_bits(), expected);
        }
        assert_eq!(repeat_add_same_binade(1.0, 0.25, 3), Some(1.75));
        assert!(repeat_add_same_binade(1.0, 0.25, 4).is_none());
        assert_eq!(repeat_add(1.0, 0.25, 4), 2.0);

        let subnormal = f32::from_bits(0x007f_fffc);
        let one_bit = f32::from_bits(1);
        assert_eq!(
            repeat_add_same_binade(subnormal, one_bit, 3)
                .unwrap()
                .to_bits(),
            0x007f_ffff
        );
        assert!(repeat_add_same_binade(subnormal, one_bit, 4).is_none());
        assert_eq!(repeat_add(subnormal, one_bit, 4).to_bits(), 0x0080_0000);
        // Checked bit arithmetic must also reject counts outside the bounded
        // traversal use; do not execute a billions-of-additions fallback here.
        assert!(repeat_add_same_binade(1.0, f32::EPSILON, u32::MAX).is_none());
        assert!(repeat_add_same_binade(1.0, 4.0 * f32::EPSILON, u32::MAX).is_none());
    }

    #[test]
    fn repeat_add_preserves_ties_subnormals_stagnation_and_excluded_signs() {
        let half_ulp = f32::EPSILON * 0.5;
        for (start, delta, first_bits, step) in [
            (0x3f80_0000, half_ulp, 0x3f80_0000, 0),
            (0x3f80_0001, half_ulp, 0x3f80_0002, 0),
            (0x3f80_0000, half_ulp * 3.0, 0x3f80_0002, 2),
            (0x3f80_0001, half_ulp * 3.0, 0x3f80_0002, 2),
        ] {
            let value = f32::from_bits(start);
            for count in 3..=16 {
                let expected = first_bits + (count - 1) * step;
                assert_eq!(
                    repeat_add_same_binade(value, delta, count)
                        .unwrap()
                        .to_bits(),
                    expected
                );
                assert_eq!(repeat_add(value, delta, count).to_bits(), expected);
                assert_eq!(repeat_add_replay(value, delta, count).to_bits(), expected);
            }
        }
        let one_bit = f32::from_bits(1);
        assert_eq!(
            repeat_add_same_binade(0.0, one_bit, 16).unwrap().to_bits(),
            16
        );
        assert_eq!(repeat_add_same_binade(1e30, 0.25, 16), Some(1e30));
        for (value, delta) in [
            (-0.0, one_bit),
            (-0.0, -0.0),
            (0.0, -0.0),
            (-1.0, 0.125),
            (1.0, -0.5),
            (1.0, 0.0),
            (f32::from_bits(0x8000_0001), one_bit),
            (f32::MAX, f32::MAX),
            (-f32::MAX, f32::MAX),
            (f32::INFINITY, 1.0),
            (f32::NEG_INFINITY, 1.0),
            (1.0, f32::INFINITY),
            (f32::NAN, 1.0),
            (1.0, f32::NAN),
            (f32::from_bits(0x7f00_0000), f32::from_bits(0x7e00_0000)),
        ] {
            assert!(repeat_add_same_binade(value, delta, 16).is_none());
            for count in 0..=16 {
                assert_eq!(
                    repeat_add(value, delta, count).to_bits(),
                    repeat_add_replay(value, delta, count).to_bits(),
                    "excluded value={:08x}, delta={:08x}, count={count}",
                    value.to_bits(),
                    delta.to_bits()
                );
            }
        }
        assert_eq!(repeat_add(-0.0, 1.0, 0).to_bits(), (-0.0_f32).to_bits());
        let payload = f32::from_bits(0x7fc1_2345);
        assert_eq!(repeat_add(payload, 1.0, 0).to_bits(), payload.to_bits());
    }

    #[test]
    fn repeat_add_matches_deterministic_f32_recurrence() {
        let mut seed = 0x8d41_a937;
        let (mut admitted, mut progressing, mut stalled, mut rejected) = (0, 0, 0, 0);
        for case in 0..500_000 {
            let mut value_bits = (random(&mut seed) & 0x7fff_ffff) % 0x7f80_0000;
            let mut delta_bits = (random(&mut seed) & 0x7fff_ffff) % 0x7f80_0000;
            if case % 11 == 0 {
                value_bits |= 0x8000_0000;
            }
            if case % 13 == 0 {
                delta_bits |= 0x8000_0000;
            }
            let value = f32::from_bits(value_bits);
            let delta = f32::from_bits(delta_bits);
            let count = random(&mut seed) % 17;
            let expected = repeat_add_replay(value, delta, count);
            assert_eq!(
                repeat_add(value, delta, count).to_bits(),
                expected.to_bits(),
                "value={value_bits:08x}, delta={delta_bits:08x}, count={count}"
            );
            if count >= 3 {
                if let Some(fast) = repeat_add_same_binade(value, delta, count) {
                    assert_eq!(fast.to_bits(), expected.to_bits());
                    admitted += 1;
                    let first = value + delta;
                    if (first + delta).to_bits() == first.to_bits() {
                        stalled += 1;
                    } else {
                        progressing += 1;
                    }
                } else {
                    rejected += 1;
                }
            }
        }
        assert!(admitted > 10_000 && progressing > 1_000 && stalled > 10_000);
        assert!(rejected > 10_000);
    }

    // Frozen form of the original non-exit-axis loop. Do not route this
    // oracle through the bit progression or production event-count helper.
    fn replay_events_before_exit(
        mut value: f32,
        delta: f32,
        exit: f32,
        inclusive: bool,
        max_steps: u32,
    ) -> (u32, f32, bool) {
        let mut count = 0;
        for _ in 0..max_steps {
            let before_exit = value < exit || (value == exit && inclusive);
            if !before_exit {
                break;
            }
            count += 1;
            value += delta;
        }
        let exhausted = value < exit || (value == exit && inclusive);
        (count, value, exhausted)
    }

    fn assert_event_count_matches_replay(
        value: f32,
        delta: f32,
        exit: f32,
        inclusive: bool,
        max_steps: u32,
    ) -> Option<u32> {
        let expected = replay_events_before_exit(value, delta, exit, inclusive, max_steps);
        let actual = count_events_before_exit(value, delta, exit, inclusive, max_steps)?;
        assert!(!expected.2, "shortcut accepted an exhausted replay");
        assert_eq!(
            (actual.0, actual.1.to_bits()),
            (expected.0, expected.1.to_bits()),
            "value={:08x}, delta={:08x}, exit={:08x}, inclusive={inclusive}, max={max_steps}",
            value.to_bits(),
            delta.to_bits(),
            exit.to_bits(),
        );
        assert!(actual.0 <= max_steps);
        Some(actual.0)
    }

    #[test]
    fn event_counts_preserve_threshold_ties_and_bounded_prefixes() {
        assert_eq!(
            count_events_before_exit(16.0, 1.0, 17.0, true, 16),
            Some((2, 18.0))
        );
        assert_eq!(
            count_events_before_exit(16.0, 1.0, 18.0, true, 16),
            Some((3, 19.0))
        );
        assert_eq!(
            count_events_before_exit(16.0, 1.0, 18.0, false, 16),
            Some((2, 18.0))
        );
        assert_eq!(count_events_before_exit(16.0, 1.0, 18.0, true, 2), None);
        assert_eq!(
            count_events_before_exit(16.0, 1.0, 18.0, true, 3),
            Some((3, 19.0))
        );
        let half_ulp = f32::EPSILON * 0.5;
        let mut admitted = [0_u32; 17];
        let mut rejected = 0;
        for (value, delta) in [
            (16.0, 0.5),
            (1.0, half_ulp),
            (f32::from_bits(0x3f80_0001), half_ulp),
            (1.0, half_ulp * 3.0),
            (f32::from_bits(0x3f80_0001), half_ulp * 3.0),
            (f32::from_bits(0x3f80_0001), half_ulp * 5.0),
            (0.0, f32::from_bits(1)),
            (f32::from_bits(3), f32::from_bits(2)),
            (f32::from_bits(0x007f_fffc), f32::from_bits(1)),
            (1.75, 0.125),
            (1e30, 0.25),
        ] {
            for event in 0..=18 {
                let crossing = repeat_add_replay(value, delta, event).to_bits();
                for bits in [
                    crossing.saturating_sub(1),
                    crossing,
                    crossing.saturating_add(1),
                ] {
                    let exit = f32::from_bits(bits);
                    for inclusive in [false, true] {
                        for max_steps in 0..=16 {
                            if let Some(count) = assert_event_count_matches_replay(
                                value, delta, exit, inclusive, max_steps,
                            ) {
                                admitted[count as usize] += 1;
                            } else {
                                rejected += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(admitted.iter().all(|&count| count != 0));
        assert!(rejected > 100);
    }

    #[test]
    fn event_counts_preserve_rejected_regions_stalls_and_special_values() {
        let value = f32::from_bits(0x4143_28ff);
        let delta = f32::from_bits(0x40be_5726);
        let third = repeat_add_replay(value, delta, 3);
        // The missing-start guard counterexample is at ordinary scene
        // magnitudes. First/second alone cannot establish the steady grid.
        assert!(count_events_before_exit(value, delta, third, false, 16).is_none());
        assert!(count_events_before_exit(1.0, 0.25, 1.99, false, 16).is_none());
        assert!(count_events_before_exit(1e30, 0.25, f32::MAX, false, 16).is_none());
        assert!(count_events_before_exit(16.0, 0.5, 30.0, true, 16).is_none());
        for (value, delta, exit) in [
            (-0.0, f32::from_bits(1), 1.0),
            (0.0, f32::from_bits(1), -0.0),
            (-1.0, 0.125, 1.0),
            (1.0, -0.5, 2.0),
            (1.0, 0.0, 2.0),
            (1.0, -0.0, 2.0),
            (f32::MAX, f32::MAX, f32::MAX),
            (f32::INFINITY, 1.0, 2.0),
            (f32::NEG_INFINITY, 1.0, 2.0),
            (1.0, f32::INFINITY, 2.0),
            (1.0, 1.0, f32::INFINITY),
            (1.0, 1.0, f32::NEG_INFINITY),
            (f32::NAN, 1.0, 2.0),
            (1.0, f32::NAN, 2.0),
            (1.0, 1.0, f32::NAN),
        ] {
            for inclusive in [false, true] {
                for max_steps in 0..=16 {
                    let _ =
                        assert_event_count_matches_replay(value, delta, exit, inclusive, max_steps);
                }
            }
        }
        // These short prefixes use the original one/two additions, including
        // when that literal result crosses an exponent boundary.
        assert_eq!(
            count_events_before_exit(1.75, 0.25, 2.0, false, 16),
            Some((1, 2.0))
        );
        assert_eq!(
            count_events_before_exit(1.5, 0.25, 2.0, false, 16),
            Some((2, 2.0))
        );
        assert_eq!(
            count_events_before_exit(0.0, f32::from_bits(1), 0.0, false, 0),
            Some((0, 0.0))
        );
    }

    #[test]
    fn event_counts_match_deterministic_f32_replay() {
        let mut state = 0x7ab1_9043;
        let (mut short, mut long, mut rejected) = (0, 0, 0);
        for case in 0..50_000 {
            let (mut value, mut delta) = if case % 2 == 0 {
                (
                    16.0 + (random(&mut state) & 0x3ff) as f32 / 256.0,
                    (random(&mut state) % 512 + 1) as f32 / 4096.0,
                )
            } else {
                (
                    f32::from_bits((random(&mut state) & 0x7fff_ffff) % 0x7f80_0000),
                    f32::from_bits((random(&mut state) & 0x7fff_ffff) % 0x7f80_0000),
                )
            };
            if case % 11 == 0 {
                value = -value;
            }
            if case % 13 == 0 {
                delta = -delta;
            }
            let event = random(&mut state) % 19;
            let crossing = repeat_add_replay(value, delta, event).to_bits();
            let exit = f32::from_bits(match random(&mut state) % 3 {
                0 => crossing.saturating_sub(1),
                1 => crossing,
                _ => crossing.saturating_add(1),
            });
            let max_steps = random(&mut state) % 17;
            for inclusive in [false, true] {
                match assert_event_count_matches_replay(value, delta, exit, inclusive, max_steps) {
                    Some(0..=2) => short += 1,
                    Some(_) => long += 1,
                    None => rejected += 1,
                }
            }
        }
        assert!(short > 1_000 && long > 1_000 && rejected > 1_000);
    }

    fn parent_masks(bricks: &[u64], dims: [u32; 3]) -> (Vec<u64>, Vec<u64>) {
        let leaf = dims.map(|v| v as usize / 4);
        let d16 = dims.map(|v| v.div_ceil(16) as usize);
        let d64 = dims.map(|v| v.div_ceil(64) as usize);
        let mut level16 = vec![0; d16.iter().product()];
        let mut level64 = vec![0; d64.iter().product()];
        for z in 0..leaf[2] {
            for y in 0..leaf[1] {
                for x in 0..leaf[0] {
                    if bricks[x + leaf[0] * (y + leaf[1] * z)] != 0 {
                        let parent = x / 4 + d16[0] * (y / 4 + d16[1] * (z / 4));
                        level16[parent] |= 1_u64 << (x % 4 + 4 * (y % 4) + 16 * (z % 4));
                    }
                }
            }
        }
        for z in 0..d16[2] {
            for y in 0..d16[1] {
                for x in 0..d16[0] {
                    if level16[x + d16[0] * (y + d16[1] * z)] != 0 {
                        let parent = x / 4 + d64[0] * (y / 4 + d64[1] * (z / 4));
                        level64[parent] |= 1_u64 << (x % 4 + 4 * (y % 4) + 16 * (z % 4));
                    }
                }
            }
        }
        (level16, level64)
    }

    fn assert_matches(bricks: &[u64], dims: [u32; 3], rays: &[([f32; 3], [f32; 3])], limit: f32) {
        let (level16, level64) = parent_masks(bricks, dims);
        let hierarchy = BrickHierarchy::new(bricks, dims, &level16, &level64);
        let origins: Vec<_> = rays.iter().flat_map(|ray| ray.0).collect();
        let directions: Vec<_> = rays.iter().flat_map(|ray| ray.1).collect();
        let mut expected_t = vec![-123.0; rays.len() + 3];
        let mut expected_id = vec![123; rays.len() + 3];
        let mut actual_t = expected_t.clone();
        let mut actual_id = expected_id.clone();
        let mut packet_t = expected_t.clone();
        let mut packet_id = expected_id.clone();
        brick_trace_n(
            bricks,
            dims,
            &origins,
            &directions,
            rays.len(),
            limit,
            &mut expected_t,
            &mut expected_id,
        );
        hierarchy.trace_n(
            &origins,
            &directions,
            rays.len(),
            limit,
            &mut actual_t,
            &mut actual_id,
        );
        hierarchy.trace_packet_n(
            &origins,
            &directions,
            rays.len(),
            limit,
            &mut packet_t,
            &mut packet_id,
        );
        for (ray, (&actual, &expected)) in actual_t.iter().zip(&expected_t).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "hierarchy distance at ray {ray}, limit {limit}"
            );
        }
        assert_eq!(actual_id, expected_id, "hierarchy hit IDs, limit {limit}");
        for (ray, (&actual, &expected)) in packet_t.iter().zip(&expected_t).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "hierarchy packet distance at ray {ray}, limit {limit}"
            );
        }
        assert_eq!(
            packet_id, expected_id,
            "hierarchy packet IDs, limit {limit}"
        );
    }

    #[test]
    fn hierarchy_packet_skip64_preserves_exact_state_and_mixed_retirement() {
        let dims = [80; 3];
        let mut bricks = vec![0; 20 * 20 * 20];
        // Lane zero advances, lane one leaves the world, lane two clips, and
        // lane three needs the original scalar fallback after overflow.
        let initial: [Walk; 4] = core::array::from_fn(|lane| Walk {
            cell: [1; 3],
            step: if lane == 1 { [-1, 1, 1] } else { [1; 3] },
            next: if lane == 3 { [f32::MAX; 3] } else { [1.0; 3] },
            delta: if lane == 3 { [f32::MAX; 3] } else { [0.25; 3] },
            time: 0.0,
        });
        let limits = [100.0, 100.0, 2.0, f32::INFINITY];
        for occupied in [false, true] {
            bricks[0] = u64::from(occupied);
            let (level16, level64) = parent_masks(&bricks, dims);
            let hierarchy = BrickHierarchy::new(&bricks, dims, &level16, &level64);
            let mut cell =
                core::array::from_fn(|axis| Simd::from_array(initial.map(|walk| walk.cell[axis])));
            let step =
                core::array::from_fn(|axis| Simd::from_array(initial.map(|walk| walk.step[axis])));
            let mut next =
                core::array::from_fn(|axis| Simd::from_array(initial.map(|walk| walk.next[axis])));
            let delta =
                core::array::from_fn(|axis| Simd::from_array(initial.map(|walk| walk.delta[axis])));
            let mut time = Simd::from_array(initial.map(|walk| walk.time));
            let mut active = Mask::splat(true);
            let fallback = hierarchy.skip_packet64(
                &mut cell,
                step,
                &mut next,
                delta,
                &mut time,
                Simd::from_array(limits),
                &mut active,
            );
            if occupied {
                assert!(fallback.is_none());
                assert!(active.all());
            } else {
                assert_eq!(fallback.unwrap().to_bitmask(), 0b1000);
                assert_eq!(active.to_bitmask(), 0b0001);
            }
            for lane in 0..4 {
                let mut expected = initial[lane];
                if !occupied && lane == 0 {
                    // Ordinary single-step replay is independent of the skip.
                    while expected.cell.map(|value| value / 16) == [0; 3] {
                        assert_eq!(expected.advance([20; 3], limits[lane]), Advance::Inside);
                    }
                }
                assert_eq!(cell.map(|axis| axis[lane]), expected.cell);
                assert_eq!(
                    next.map(|axis| axis[lane].to_bits()),
                    expected.next.map(f32::to_bits),
                );
                assert_eq!(time[lane].to_bits(), expected.time.to_bits());
            }
        }
    }

    #[test]
    fn hierarchy_skip_replays_exact_leaf_state_and_ties() {
        let mut seed = 0x73a2_b915;
        for iteration in 0..12_000 {
            let dims = [35, 19, 33];
            let edge = if iteration % 2 == 0 { 4 } else { 16 };
            let cell = dims.map(|extent| (random(&mut seed) % extent as u32) as i32);
            let step = core::array::from_fn(|_| if random(&mut seed) & 1 == 0 { 1 } else { -1 });
            let (next, delta) = if iteration % 5 == 0 {
                // Equal events, signed zero, and additions that stagnate at a
                // large exponent exercise the scalar tie order explicitly.
                let time = [0.0, -0.0, 1.0, 1e30][iteration / 5 % 4];
                ([time; 3], [0.25; 3])
            } else {
                (
                    core::array::from_fn(|_| (random(&mut seed) % 1024) as f32 * 0.125 - 4.0),
                    core::array::from_fn(|_| (random(&mut seed) % 2048 + 1) as f32 * 0.003),
                )
            };
            let limit = if iteration % 7 == 0 {
                64.0
            } else {
                f32::INFINITY
            };
            let mut expected = Walk {
                cell,
                step,
                next,
                delta,
                time: 0.0,
            };
            let mut actual = expected;
            let parent = cell.map(|v| v / edge);
            let expected_result = loop {
                let result = expected.advance(dims, limit);
                if result != Advance::Inside || expected.cell.map(|v| v / edge) != parent {
                    break result;
                }
            };
            let actual_result = actual.skip_empty(dims, edge, limit);
            assert_eq!(actual_result, expected_result, "skip result at {iteration}");
            if actual_result == Advance::Inside {
                assert_eq!(actual.cell, expected.cell, "skip cell at {iteration}");
                assert_eq!(
                    actual.next.map(f32::to_bits),
                    expected.next.map(f32::to_bits),
                    "skip next at {iteration}"
                );
                assert_eq!(
                    actual.time.to_bits(),
                    expected.time.to_bits(),
                    "skip time at {iteration}"
                );
            }
        }
    }

    #[test]
    fn hierarchy_matches_sparse_dense_empty_and_edited_partial_parents() {
        let dims = [132, 20, 68];
        let count = (dims[0] * dims[1] * dims[2] / 64) as usize;
        let mut seed = 0x1826_ca91;
        let mut sparse = vec![0_u64; count];
        for word in &mut sparse {
            if random(&mut seed) % 17 == 0 {
                *word = u64::from(random(&mut seed)) << 32 | u64::from(random(&mut seed));
            }
        }
        let mut rays = Vec::new();
        for y in 0..9 {
            for z in 0..13 {
                rays.push((
                    [-12.0, 10.0, 30.0],
                    [1.0, (y as f32 - 4.0) * 0.025, (z as f32 - 6.0) * 0.035],
                ));
            }
        }
        for _ in 0..513 {
            rays.push((
                core::array::from_fn(|axis| {
                    (random(&mut seed) % 65536) as f32 / 65536.0 * (dims[axis] as f32 + 80.0) - 40.0
                }),
                core::array::from_fn(|_| (random(&mut seed) % 65536) as f32 / 16384.0 - 2.0),
            ));
        }
        for mut bricks in [vec![0; count], vec![u64::MAX; count], sparse] {
            for limit in [0.0, 4.0, 64.0, 700.0, f32::INFINITY] {
                assert_matches(&bricks, dims, &rays, limit);
            }
            for (index, word) in bricks.iter_mut().enumerate() {
                if index % 2 == 0 {
                    *word = 0;
                }
            }
            // The final leaf is in partial parents at both hierarchy levels.
            bricks[count - 1] ^= 1_u64 << 63;
            assert_matches(&bricks, dims, &rays, 700.0);
            assert_matches(&bricks, dims, &rays[..3], 700.0);
            assert_matches(&bricks, dims, &[], 700.0);
        }
    }

    #[test]
    fn hierarchy_hits_after_long_empty_parent_runs_and_corner_ties() {
        let dims = [196, 68, 132];
        let leaf = [49, 17, 33];
        let mut bricks = vec![0_u64; leaf.iter().product()];
        let last = 48 + leaf[0] * (16 + leaf[1] * 32);
        bricks[last] = 1;
        let rays = [
            ([-4.0, 64.5, 128.5], [1.0, 0.0, 0.0]),
            ([200.0, 64.5, 128.5], [-2.0, -0.0, 0.0]),
            ([192.5, -4.0, 128.5], [0.0, 1.0, 0.0]),
            ([192.5, 64.5, -4.0], [0.0, 0.0, 1.0]),
            ([-4.0, -132.0, -68.0], [1.0; 3]),
            ([-4.0, -132.0, -68.0], [1.0, 1.0 + f32::EPSILON, 1.0]),
            ([-4.0, -132.0, -68.0], [1.0, 1.0 - f32::EPSILON, 1.0]),
        ];
        for &(origin, direction) in &rays[..5] {
            assert_ne!(
                brick_trace(&bricks, dims, origin, direction, f32::INFINITY).1,
                u32::MAX
            );
        }
        for word in [1, 0, 1_u64 << 63] {
            bricks[last] = word;
            let coherent: Vec<_> = rays
                .iter()
                .flat_map(|&ray| core::iter::repeat_n(ray, 4))
                .collect();
            for limit in [0.0, 3.5, 68.0, 132.0, 196.0, f32::INFINITY] {
                assert_matches(&bricks, dims, &rays, limit);
                assert_matches(&bricks, dims, &coherent, limit);
            }
        }
    }

    #[test]
    fn hierarchy_packets_split_coherent_parent_exits_and_preserve_edits() {
        let dims = [132; 3];
        let leaf = 33_usize;
        let mut bricks = vec![0; leaf * leaf * leaf];
        let mut seed = 0x269a_b41d;
        // Only the last partial64 parents are occupied. Most complete64
        // parents stay empty, while the final destinations need fine4.
        for z in 0..leaf {
            for y in 0..leaf {
                for x in 0..leaf {
                    if (x == 32 || y == 32 || z == 32) && random(&mut seed) % 3 == 0 {
                        bricks[x + leaf * (y + leaf * z)] =
                            u64::from(random(&mut seed)) << 32 | u64::from(random(&mut seed));
                    }
                }
            }
        }
        let mut rays = Vec::new();
        for packet in 0..128 {
            let origin = if packet % 4 == 0 {
                [64.0; 3]
            } else {
                core::array::from_fn(|_| (random(&mut seed) % 128) as f32 + 0.5)
            };
            let direction: [f32; 3] =
                core::array::from_fn(|_| (random(&mut seed) % 2048) as f32 / 1024.0 - 1.0);
            for lane in 0..4 {
                let mut direction = direction;
                direction[lane % 3] += lane as f32 / 128.0;
                rays.push((origin, direction));
            }
        }
        rays.extend_from_slice(&[
            ([64.0; 3], [1.0; 3]),
            ([64.0; 3], [-1.0; 3]),
            ([64.0; 3], [1.0, -0.0, 0.0]),
        ]);
        for edit in 0..3 {
            for limit in [0.0, 32.0, 64.0, 128.0, f32::INFINITY] {
                assert_matches(&bricks, dims, &rays, limit);
            }
            for (index, word) in bricks.iter_mut().enumerate() {
                if (index + edit) % 5 == 0 {
                    *word = if edit == 0 { 0 } else { u64::MAX };
                }
            }
        }
    }

    #[test]
    fn hierarchy_preserves_boundary_clipping_signed_zero_and_nonunit_rays() {
        let dims = [68, 20, 132];
        let count = (dims[0] * dims[1] * dims[2] / 64) as usize;
        let origins = [
            [0.0; 3],
            [4.0; 3],
            [16.0, 8.0, 16.0],
            [64.0, 16.0, 64.0],
            [68.0, 20.0, 132.0],
            [-4.0, 10.0, 66.0],
            [72.0, 10.0, 66.0],
            [34.0, -4.0, 66.0],
            [34.0, 10.0, 136.0],
        ];
        let directions = [
            [0.0, 0.0, 1.0],
            [-0.0, 0.0, -1.0],
            [0.0; 3],
            [-0.0; 3],
            [1.0; 3],
            [-1.0; 3],
            [0.5, 0.0, 0.0],
            [1.0, -1.0, 0.0],
            [5e-13, -5e-13, 1.0],
            [-1e-12, 1e-12, -0.0],
        ];
        let rays: Vec<_> = origins
            .iter()
            .flat_map(|&origin| directions.iter().map(move |&direction| (origin, direction)))
            .collect();
        for bricks in [
            vec![0; count],
            vec![u64::MAX; count],
            vec![0x8100_0081_8000_0001; count],
        ] {
            for limit in [-1.0, -0.0, 0.0, 0.25, 4.0, 64.0, 132.0, f32::INFINITY] {
                assert_matches(&bricks, dims, &rays, limit);
            }
        }
    }

    #[test]
    fn hierarchy_preserves_finite_overflow_and_subnormal_results() {
        let dims = [68, 4, 4];
        let mut seed = 0x901c_fada;
        let mut rays = vec![
            ([-f32::MAX; 3], [1e-12; 3]),
            ([f32::MAX; 3], [-1e-12; 3]),
            ([-f32::MAX, 2.0, 2.0], [1.0, 0.0, 0.0]),
            ([2.0; 3], [f32::MAX; 3]),
        ];
        for _ in 0..513 {
            let mut finite = || {
                let value = f32::from_bits(random(&mut seed));
                if value.is_finite() { value } else { 0.0 }
            };
            rays.push((
                core::array::from_fn(|_| finite()),
                core::array::from_fn(|_| finite()),
            ));
        }
        for bricks in [[0; 17], [u64::MAX; 17], [0x8100_0081_8000_0001; 17]] {
            for limit in [0.0, 100.0, f32::MAX, f32::INFINITY] {
                assert_matches(&bricks, dims, &rays, limit);
            }
        }
    }

    #[test]
    fn hierarchy_invalid_rays_never_report_false_hits() {
        let bricks = [u64::MAX; 1];
        let level16 = [1];
        let level64 = [1];
        let hierarchy = BrickHierarchy::new(&bricks, [4; 3], &level16, &level64);
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(hierarchy.trace([value, 1.0, 1.0], [1.0; 3], 100.0), MISS);
            assert_eq!(hierarchy.trace([1.0; 3], [value, 1.0, 1.0], 100.0), MISS);
        }
        for limit in [f32::NAN, -1.0, f32::NEG_INFINITY] {
            assert_eq!(hierarchy.trace([1.0; 3], [1.0; 3], limit), MISS);
        }
        assert_eq!(
            hierarchy.trace([1.5; 3], [-0.0; 3], f32::INFINITY),
            (0.0, 21)
        );
        let rays = [
            ([f32::NAN, 1.5, 1.5], [1.0; 3]),
            ([1.5; 3], [f32::INFINITY, 0.0, 0.0]),
            ([1.5; 3], [-0.0; 3]),
            ([f32::NEG_INFINITY, 1.5, 1.5], [1.0; 3]),
            ([1.5; 3], [1.0; 3]),
        ];
        let origins: Vec<_> = rays.iter().flat_map(|ray| ray.0).collect();
        let directions: Vec<_> = rays.iter().flat_map(|ray| ray.1).collect();
        for limit in [f32::NAN, -1.0, f32::NEG_INFINITY, f32::INFINITY] {
            let mut distances = [-123.0_f32; 8];
            let mut ids = [123; 8];
            hierarchy.trace_packet_n(
                &origins,
                &directions,
                rays.len(),
                limit,
                &mut distances,
                &mut ids,
            );
            for (lane, &(origin, direction)) in rays.iter().enumerate() {
                let expected = hierarchy.trace(origin, direction, limit);
                assert_eq!(distances[lane].to_bits(), expected.0.to_bits());
                assert_eq!(ids[lane], expected.1);
            }
            assert_eq!(&distances[rays.len()..], &[-123.0; 3]);
            assert_eq!(&ids[rays.len()..], &[123; 3]);
        }
    }

    #[test]
    #[should_panic]
    fn hierarchy_rejects_short_parent_descriptor() {
        BrickHierarchy::new(&[0; 17], [68, 4, 4], &[0; 5], &[0]);
    }

    #[test]
    #[should_panic]
    fn hierarchy_rejects_overflowing_grid_descriptor() {
        BrickHierarchy::new(&[], [u32::MAX - 3, 4, 4], &[], &[]);
    }

    #[test]
    #[should_panic]
    fn hierarchy_rejects_short_ray_buffers_before_reading() {
        BrickHierarchy::new(&[0], [4; 3], &[0], &[0]).trace_n(
            &[0.0; 2],
            &[1.0; 3],
            1,
            100.0,
            &mut [0.0],
            &mut [0],
        );
    }

    #[test]
    #[should_panic]
    fn hierarchy_packets_reject_short_ray_buffers_before_reading() {
        BrickHierarchy::new(&[0], [4; 3], &[0], &[0]).trace_packet_n(
            &[0.0; 2],
            &[1.0; 3],
            1,
            100.0,
            &mut [0.0],
            &mut [0],
        );
    }

    #[test]
    #[should_panic]
    fn hierarchy_packets_reject_short_output_buffers_before_writing() {
        BrickHierarchy::new(&[0], [4; 3], &[0], &[0]).trace_packet_n(
            &[0.0; 3],
            &[1.0; 3],
            1,
            100.0,
            &mut [0.0],
            &mut [],
        );
    }

    #[test]
    #[should_panic]
    fn hierarchy_packets_reject_overflowing_ray_count() {
        BrickHierarchy::new(&[0], [4; 3], &[0], &[0]).trace_packet_n(
            &[],
            &[],
            usize::MAX,
            100.0,
            &mut [],
            &mut [],
        );
    }
}
