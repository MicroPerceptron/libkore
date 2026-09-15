//! **The lane tier** (`res/SPMD.md` §13, G8) — software SIMT inside ONE
//! worker: W lanes as `core::simd` vector elements, divergence handled
//! by **active masks** with structured reconvergence, never a cross-core
//! gang.
//!
//! The dichotomy that decides every representation here: packed per-lane
//! state is coherent exactly when one core owns it.  A group gangs
//! workers across cores with cache-line-isolated hot words (the barrier,
//! §3); a worker packs its lanes inside one thread with packed words —
//! one XSAVE image, one scheduler record, one founding, one quantum:
//! 1/W of every per-task cost, which no cross-core design can match.
//! The tier is deliberately *not* called a warp: a warp is GPU hardware,
//! and borrowing the name would invite reading CUDA cores into what is
//! software lockstep on one CPU core.
//!
//! # Reconvergence is scope exit
//!
//! The ISPC model, enforced by the closure shape rather than by
//! convention: [`Lockstep::branch`] runs each arm under the tightened
//! mask and RESTORES the parent mask when it returns — the structural
//! guarantee a hand-rolled mask stack forgets exactly once.
//! [`Lockstep::refine`] is the masked while-loop: re-evaluate, tighten,
//! run, until no lane remains; the nesting depth is the caller's own
//! stack, no allocation, no mask array.
//!
//! # Kernels are plain functions
//!
//! A lane kernel is an ordinary `fn` over [`LaneF32`]/[`LaneMask`] —
//! raster's own discipline (`crates/raster`: SIMD is the implementation,
//! the scalar fringe is the parity baseline), recast from integer rows
//! to float FMA.  No macro: the macro layer names launches, not bodies.
//! [`fma`] is the reference kernel the host parity sweep and the terminal
//! demo (`spmd lane`) both run.
//!
//! This module is unconditional — every target compiles it and the host
//! suite proves the mask laws everywhere (`core::simd` legalizes on all
//! three ISAs; riscv64 scalarizes).  The fiber escape for divergence too
//! irreducible to mask is G9's (`res/SPMD.md` §13.3), landing beside
//! this engine, consuming only its mask vocabulary.

use core::simd::prelude::*;

pub mod fiber;
mod switch;

/// Lanes per worker — derived from the enabled target features, the
/// raster `W` precedent recast for `f32`: AVX2 carries 8 single-precision
/// lanes per op; everything else takes the 128-bit floor of 4.
pub const W: usize = if cfg!(target_feature = "avx2") { 8 } else { 4 };

/// One vector of lane values — the tier's working type.
pub type LaneF32 = Simd<f32, W>;

/// One vector of lane indices/integers (tile arithmetic).
pub type LaneU32 = Simd<u32, W>;

/// The execution mask — which lanes are live under the current
/// divergence nest.
pub type LaneMask = Mask<i32, W>;

/// The active-mask engine: the current mask plus the structured
/// reconvergence discipline.  Affine by convention (one per worker
/// body); holding it conveys nothing — it is state, like the barrier.
pub struct Lockstep {
    active: LaneMask,
}

impl Default for Lockstep {
    fn default() -> Self {
        Self::full()
    }
}

impl Lockstep {
    /// Every lane live — a kernel's entry state.
    #[inline]
    pub fn full() -> Self {
        Self {
            active: LaneMask::splat(true),
        }
    }

    /// The current execution mask — what a kernel selects stores by.
    #[inline]
    pub const fn active(&self) -> LaneMask {
        self.active
    }

    /// Whether any lane is live under the current nest.
    #[inline]
    pub fn any(&self) -> bool {
        self.active.any()
    }

    /// Structured divergence: run `then` under `active & cond` and
    /// `otherwise` under `active & !cond`, restoring the parent mask on
    /// return — scope exit IS the reconvergence point.  An arm whose
    /// mask is empty is skipped whole (the ISPC economy: no lane, no
    /// work, no side effects).
    pub fn branch(
        &mut self,
        cond: LaneMask,
        then: impl FnOnce(&mut Self),
        otherwise: impl FnOnce(&mut Self),
    ) {
        let parent = self.active;
        let taken = parent & cond;
        if taken.any() {
            self.active = taken;
            then(self);
        }
        let untaken = parent & !cond;
        if untaken.any() {
            self.active = untaken;
            otherwise(self);
        }
        self.active = parent;
    }

    /// The masked while-loop: re-evaluate `cond` under the current mask,
    /// tighten to the still-true lanes, run `body`, repeat until no lane
    /// remains — then restore the parent mask.  Each lane runs exactly
    /// its own trip count; the loop's divergence lives entirely in the
    /// mask.
    pub fn refine(
        &mut self,
        mut cond: impl FnMut(&Self) -> LaneMask,
        mut body: impl FnMut(&mut Self),
    ) {
        let parent = self.active;
        loop {
            let live = self.active & cond(self);
            if !live.any() {
                break;
            }
            self.active = live;
            body(self);
        }
        self.active = parent;
    }

    /// Visit each live lane's index, in lane order — the G9 escape's
    /// spill walk, and the scalar fringe's iterator.
    #[inline]
    pub fn each_active(&self, mut f: impl FnMut(usize)) {
        for lane in 0..W {
            if self.active.test(lane) {
                f(lane);
            }
        }
    }
}

/// The reference kernel — one masked multiply-add step:
/// `acc = a * b + acc` on the live lanes, the untouched lanes carried
/// through (the §13.5 "raster integer rows recast to float").  Written
/// unfused — `core::simd` reserves the fused form for `std`'s
/// `StdFloat` — and the backend contracts it to FMA where the target
/// spec enables it (the v3 profile carries `+fma`).
#[inline]
pub fn fma(ls: &Lockstep, a: LaneF32, b: LaneF32, acc: LaneF32) -> LaneF32 {
    ls.active().select(a * b + acc, acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask_of(bits: &[bool]) -> LaneMask {
        let mut m = LaneMask::splat(false);
        for (i, &b) in bits.iter().enumerate().take(W) {
            m.set(i, b);
        }
        m
    }

    /// Reconvergence restores the parent across nesting, and each arm
    /// sees exactly `parent & cond` / `parent & !cond`.
    #[test]
    fn branch_masks_are_exact_and_reconverge() {
        let mut ls = Lockstep::full();
        let mut parent = LaneMask::splat(true);
        parent.set(0, false); // lane 0 already retired by an outer nest
        ls.active = parent;

        let mut cond = LaneMask::splat(false);
        cond.set(0, true); // conditioned true, but dead in the parent
        cond.set(1, true);

        let mut then_seen = None;
        let mut else_seen = None;
        ls.branch(
            cond,
            |inner| {
                then_seen = Some(inner.active());
                // Nested divergence must restore THIS arm's mask.
                let sub = LaneMask::splat(false);
                inner.branch(
                    sub,
                    |_| unreachable!("empty arm ran"),
                    |i| {
                        assert_eq!(i.active(), parent & cond, "else-of-empty = the arm");
                    },
                );
                assert_eq!(inner.active(), parent & cond, "nested reconvergence");
            },
            |inner| {
                else_seen = Some(inner.active());
            },
        );
        assert_eq!(then_seen, Some(parent & cond), "then = parent & cond");
        assert_eq!(else_seen, Some(parent & !cond), "else = parent & !cond");
        assert_eq!(ls.active(), parent, "scope exit reconverges");
    }

    /// An arm whose tightened mask is empty never runs.
    #[test]
    fn empty_arms_are_skipped_whole() {
        let mut ls = Lockstep::full();
        ls.branch(
            LaneMask::splat(true),
            |_| {},
            |_| unreachable!("an empty else ran"),
        );
        ls.branch(
            LaneMask::splat(false),
            |_| unreachable!("an empty then ran"),
            |_| {},
        );
        assert!(ls.any());
    }

    /// `refine` gives each lane exactly its own trip count and
    /// reconverges to the parent.
    #[test]
    fn refine_trips_per_lane_and_terminates() {
        let mut ls = Lockstep::full();
        // Lane i wants i trips (lane 0: zero — never live inside).
        let want = LaneU32::from_slice(&core::array::from_fn::<u32, W, _>(|i| i as u32));
        let done = core::cell::Cell::new(LaneU32::splat(0));
        let mut body_runs = 0usize;
        ls.refine(
            |_| done.get().simd_lt(want).into(),
            |inner| {
                body_runs += 1;
                assert!(body_runs <= W, "refine failed to tighten");
                done.set(
                    inner
                        .active()
                        .select(done.get() + LaneU32::splat(1), done.get()),
                );
            },
        );
        let done = done.get();
        assert_eq!(done, want, "each lane ran exactly its own trips");
        assert!(ls.active().all(), "refine reconverges to the parent");
    }

    /// The FMA kernel against the scalar reference, swept over every
    /// ragged edge `0..=W` (raster's parity-sweep shape).
    #[test]
    fn fma_matches_scalar_at_every_ragged_edge() {
        for edge in 0..=W {
            let mut ls = Lockstep::full();
            ls.active = mask_of(&core::array::from_fn::<bool, W, _>(|i| i < edge));
            let a = LaneF32::from_slice(&core::array::from_fn::<f32, W, _>(|i| i as f32 + 0.5));
            let b = LaneF32::splat(2.0);
            let acc = LaneF32::splat(1.0);
            let out = fma(&ls, a, b, acc);
            for lane in 0..W {
                let expect = if lane < edge {
                    (lane as f32 + 0.5) * 2.0 + 1.0
                } else {
                    1.0
                };
                assert_eq!(out[lane], expect, "lane {lane} at edge {edge}");
            }
        }
    }

    /// The width and the mask agree — the one arithmetic every consumer
    /// assumes.
    #[test]
    fn width_is_consistent() {
        assert_eq!(LaneMask::splat(true).to_array().len(), W);
        assert_eq!(LaneF32::LEN, W);
        assert!(W == 4 || W == 8, "the two lawful widths");
    }
}
