//! `dispatch` — the consumer's geometry bridge: kore's SPMD tile shape
//! lowered to accelerator [`Workgroups`].
//!
//! This is the narrow portable correspondence: a kore tile grid becomes the
//! number of workgroups and the caller supplies each workgroup's logical local
//! size. SIMD/subgroup width is not part of the client contract; each provider
//! chooses it while compiling and scheduling the artifact.
//!
//! # The correspondence (R8, made concrete)
//!
//! | kore SPMD (`res/SPMD.md`) | accel [`Workgroups`] |
//! | --- | --- |
//! | tile grid `[x,y,z]` ([`SpmdCtx::tiles`]) | `group_count` |
//! | explicitly declared local shape | `local_size` |
//!
//! [`for_ctx`] reads the grid straight off the [`SpmdCtx`] a
//! `#[kore_rt::spmd(...)]` region already declared. Only nonzero axes are
//! checked here; device ceilings remain backend admission policy.

use kore_driver::accel::Workgroups;
use kore_rt::spmd::SpmdCtx;
use virtio_accel_core::BackendError;

/// Lower a raw kore tile grid and local work-item shape to [`Workgroups`].
///
/// `grid` is the thread-group grid (kore's tiles); `group` is the
/// intra-group work-item shape (`[1, 1, 1]` for the stride/reduction shape).
/// `Err` on a zero axis — [`Workgroups::new`]'s universal invariant — never a
/// device ceiling, which is the backend's to enforce.
pub fn for_grid(grid: [u32; 3], group: [u32; 3]) -> Result<Workgroups, BackendError> {
    Workgroups::new(grid, group)
}

/// Lower the tile grid an [`SpmdCtx`] already declares — the literal
/// "express the geometry once, target the CPU worker/lane tier **or** an
/// accelerator": a `#[kore_rt::spmd(x, y, z)]` region's ctx carries `[x, y,
/// z]` as its [`tiles`](SpmdCtx::tiles), and this hands that same grid to a
/// submission.  `group` is the caller's intra-group shape, as [`for_grid`].
pub fn for_ctx(ctx: &SpmdCtx, group: [u32; 3]) -> Result<Workgroups, BackendError> {
    for_grid(ctx.tiles(), group)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A raw grid lowers to the group count and local size it named.
    #[test]
    fn for_grid_maps_the_vocabulary_axes() {
        let d = for_grid([16, 8, 2], [64, 1, 1]).expect("nonzero axes lower");
        assert_eq!(d.group_count(), [16, 8, 2]);
        assert_eq!(d.local_size(), [64, 1, 1]);
    }

    /// A ctx's declared tile grid becomes the dispatch grid unchanged — the
    /// "write the geometry once" path, proven against the grid a
    /// macro-shaped region would carry.
    #[test]
    fn for_ctx_reads_the_declared_tile_grid() {
        let ctx = SpmdCtx::with_grid(7, [4, 4, 64]);
        let d = for_ctx(&ctx, [1, 1, 1]).expect("nonzero tiles lower");
        assert_eq!(d.group_count(), ctx.tiles());
        assert_eq!(d.group_count(), [4, 4, 64]);
        assert_eq!(d.local_size(), [1, 1, 1]);
    }

    /// A zero grid axis is refused — the universal invariant, surfaced from
    /// [`Workgroups::new`] rather than guessed around.
    #[test]
    fn a_zero_axis_is_refused() {
        assert_eq!(
            for_grid([0, 1, 1], [1, 1, 1]),
            Err(BackendError::InvalidArgument)
        );
        assert_eq!(
            for_grid([1, 1, 1], [1, 0, 1]),
            Err(BackendError::InvalidArgument)
        );
    }
}
