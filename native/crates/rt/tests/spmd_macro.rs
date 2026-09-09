//! The `#[kore_rt::spmd]` attribute's full lowering, invoked for
//! real: the geometry consts, the ctx static, and the launcher — with the
//! worker ceiling's `min` proven **in the types** (the `SpawnedSpmd<N>`
//! annotations only compile if the emitted const arithmetic lands on the
//! expected width).  Behavior on the host is the typed refusal, exactly
//! as `thread::spawn`'s.

use kore_rt::spmd::{SpmdCtx, SpawnedSpmd};

/// 2·2·2 = 8 tiles on a MAX_THREADS = 32 plan: under the ceiling, so
/// the grid IS the width — and this annotation is the tripwire that
/// consciously fired when G4c widened the plan (it was `SpawnedSpmd<7>`
/// while the ceiling was 8).  The next widening trips `wall` below
/// instead.
#[kore_rt::spmd(2, 2, 2)]
fn cube(ctx: &'static SpmdCtx, worker: u32) {
    let _ = (ctx, worker);
}

/// 64 tiles: past the 31-worker ceiling, so `min` binds the width.
#[kore_rt::spmd(4, 4, 4)]
fn wall(ctx: &'static SpmdCtx, worker: u32) {
    let _ = (ctx, worker);
}

/// 1·1·4 = 4 tiles: under the ceiling, the grid IS the width.
#[kore_rt::spmd(1, 1, 4)]
fn line(ctx: &'static SpmdCtx, worker: u32) {
    let _ = (ctx, worker);
}

#[test]
fn the_ceiling_binds_in_the_type() {
    // `wall` launches min(64, 31) = 31 workers; a wrong const would
    // make this annotation a compile error, which is the whole test.
    let g: Option<SpawnedSpmd<31>> = wall();
    assert!(g.is_none(), "host builds refuse the founding, typed");
    // `cube` sits under the ceiling: the grid is the width.
    let g: Option<SpawnedSpmd<8>> = cube();
    assert!(g.is_none(), "host builds refuse the founding, typed");
}

#[test]
fn a_grid_under_the_ceiling_keeps_its_width() {
    let g: Option<SpawnedSpmd<4>> = line();
    assert!(g.is_none(), "host builds refuse the founding, typed");
}
