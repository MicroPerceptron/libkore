//! **Registered kernels** — what a "program" is on the CPU row.
//!
//! The NPU row loads a compiled artifact: an ELF the NPU compiler emitted,
//! placed in device memory and relocated. There is no CPU compiler in this
//! tree, so this row takes the other option the protocol already leaves open.
//! [`ArtifactRef::target`] is twelve **opaque** words whose "schema belongs to
//! the artifact format, not this crate" — so a format whose schema says *the
//! words name a kernel this build already carries* is conformant, not a
//! workaround. `load_program` validates a fixed versioned manifest, then does
//! a lookup rather than loading executable bytes; any other payload is refused
//! rather than ignored.
//!
//! That choice is a v1 posture, not a ceiling: an artifact-loading CPU row can
//! land beside this one under a second [`ArtifactFormat`] without moving a
//! byte of protocol, and the W^X executable-lease path (#660) is what it would
//! stand on.
//!
//! # The tiling contract
//!
//! A kernel declares the size of one **output element**, and that is the unit
//! the region's workers are split over (`res/SPMD.md` R1 — the launch extent
//! is the *work* grid, not a thread count). The backend does the split; a
//! kernel body receives its own element span and a `&mut [u8]` that is exactly
//! those elements' bytes. Two workers therefore cannot alias, and a kernel
//! never reasons about the region at all.
//!
//! Inputs are shared for the whole grid: an elementwise body indexes them by
//! the same element numbers it was given.
//!
//! # What this shape deliberately does not cover
//!
//! Reductions. A sum has one output element, so the tiling above degenerates
//! to a single worker and the parallelism is lost. That is not a gap to paper
//! over here — `spmd::reduce` is a *different launch shape* (map to a partial,
//! then combine), and it wants its own kernel class rather than a reduction
//! squeezed through an elementwise signature. The registry is the place that
//! class will be added.
//!
//! [`ArtifactRef::target`]: virtio_accel_core::ArtifactRef
//! [`ArtifactFormat`]: virtio_accel_core::ArtifactFormat

use core::ops::Range;
use core::simd::{Simd, num::SimdFloat};

use kore_driver::accel::LaunchShape;
use virtio_accel_core::TargetIdentity;

/// Lanes per SIMD step. Eight `f32` is 256 bits — the v3 user profile's
/// natural width, and the widest the tree's baseline profile can assume.
pub(crate) const LANES: usize = 8;

/// The identity word every kernel this row serves begins with — `"KSPM"`.
///
/// A client naming any other first word is naming an artifact for some other
/// format, and gets [`Unsupported`](virtio_accel_core::BackendError::Unsupported)
/// rather than a lookup miss: the two are different failures and a client can
/// act on the difference.
pub const IDENTITY_MAGIC: u32 = u32::from_le_bytes(*b"KSPM");

/// One tile span's view of a kernel launch.
///
/// `out` is **exactly** this span's bytes — the backend has already done the
/// split, so a body writes its whole slice and cannot reach a sibling's.
pub struct TileCtx<'a> {
    /// This worker's element span within the grid.
    pub tiles: Range<u32>,
    /// The whole grid's element count.
    pub total: u32,
    /// The exact launch policy admitted for this execution.
    ///
    /// Explicit workgroup axes remain intact here even though the CPU worker
    /// tier schedules a contiguous linear span. A registered kernel can derive
    /// logical group/local ids from the original shape; equal-product shapes
    /// are never collapsed into the same kernel invocation. No barrier model
    /// is implied by this field (`res/SPMD.md` R8).
    pub launch: LaunchShape,
    /// The bound inputs, whole (not split) — indexed by grid element.
    pub inputs: &'a [&'a [u8]],
    /// This span's output bytes, and only this span's.
    pub out: &'a mut [u8],
}

/// A kernel this build carries.
pub struct Kernel {
    /// The registry key: `[IDENTITY_MAGIC, id, 0, 0, …]`.
    pub id: u32,
    /// Diagnostic name — what a status row prints, never a lookup key.
    pub name: &'static str,
    /// Input bindings this kernel requires, exactly.
    pub inputs: u32,
    /// Bytes in one output element — the tiling unit.
    pub elem_bytes: u32,
    /// Lanes this kernel's body runs in lockstep.
    ///
    /// A property of the compiled body, not a client choice or wire field.
    /// Providers may use it for diagnostics and scheduling decisions.
    pub lanes: u32,
    /// Run one span. Called once per worker, concurrently.
    pub run: fn(TileCtx<'_>),
}

impl Kernel {
    /// The twelve-word identity a client names this kernel by.
    pub const fn identity(&self) -> TargetIdentity {
        let mut words = [0u32; 12];
        words[0] = IDENTITY_MAGIC;
        words[1] = self.id;
        TargetIdentity(words)
    }
}

/// Split a target identity into its kernel id, or `None` when the words are
/// not this format's.
///
/// Fail-closed on the reserved tail: words 2..12 must be zero. A client that
/// set one is asking for something this build does not implement, and
/// accepting the request while ignoring the words would be the dishonest
/// reading — the same posture every sibling wire form in the tree takes.
pub fn kernel_id(target: TargetIdentity) -> Option<u32> {
    let TargetIdentity(words) = target;
    if words[0] != IDENTITY_MAGIC {
        return None;
    }
    if words[2..].iter().any(|&w| w != 0) {
        return None;
    }
    Some(words[1])
}

/// Find a kernel by id.
pub fn lookup(registry: &'static [Kernel], id: u32) -> Option<&'static Kernel> {
    registry.iter().find(|k| k.id == id)
}

// ── the built-in registry ───────────────────────────────────────────────

/// `add.f32` — elementwise `out[i] = in0[i] + in1[i]`.
pub const KERNEL_ADD_F32: u32 = 1;
/// `mul.f32` — elementwise `out[i] = in0[i] * in1[i]`.
pub const KERNEL_MUL_F32: u32 = 2;

/// The kernels this build carries.
///
/// Two, deliberately: enough to prove the seam end to end (binding validation,
/// tiling, the region launch, the serial fallback) without this row pretending
/// to be a compute library. A kernel set is a separate question from a backend,
/// and answering it here would make the backend hard to review.
pub static BUILTIN: &[Kernel] = &[
    Kernel {
        id: KERNEL_ADD_F32,
        name: "add.f32",
        inputs: 2,
        elem_bytes: 4,
        lanes: LANES as u32,
        run: add_f32,
    },
    Kernel {
        id: KERNEL_MUL_F32,
        name: "mul.f32",
        inputs: 2,
        elem_bytes: 4,
        lanes: LANES as u32,
        run: mul_f32,
    },
];

/// Read `count` `f32`s starting at element `first` from a bound input.
///
/// Bindings arrive as bytes: the wire carries offsets into shared windows and
/// says nothing about element types, so a kernel decodes its own operands.
/// Unaligned by construction — a client's offset need not be 4-byte aligned —
/// which is why this reads through `from_le_bytes` rather than casting.
#[inline]
fn load_f32(src: &[u8], index: usize) -> f32 {
    let at = index * 4;
    let mut word = [0u8; 4];
    word.copy_from_slice(&src[at..at + 4]);
    f32::from_le_bytes(word)
}

#[inline]
fn store_f32(dst: &mut [u8], index: usize, value: f32) {
    let at = index * 4;
    dst[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

/// The shared body of both elementwise kernels — `op` applied lane-wise.
///
/// Vectorised through [`core::simd`] rather than a scalar loop
/// (`kore-design-doctrine` §8: byte-at-a-time is a red flag, and the portable
/// API composes with the trait-seam discipline instead of fighting it). The
/// tail below `LANES` runs scalar, which is the ordinary shape and not a
/// fallback.
#[inline]
fn elementwise(ctx: TileCtx<'_>, op: fn(Simd<f32, LANES>, Simd<f32, LANES>) -> Simd<f32, LANES>) {
    let a = ctx.inputs[0];
    let b = ctx.inputs[1];
    let span = (ctx.tiles.end - ctx.tiles.start) as usize;
    let base = ctx.tiles.start as usize;

    let mut i = 0;
    while i + LANES <= span {
        let mut lhs = [0f32; LANES];
        let mut rhs = [0f32; LANES];
        for lane in 0..LANES {
            lhs[lane] = load_f32(a, base + i + lane);
            rhs[lane] = load_f32(b, base + i + lane);
        }
        let out = op(Simd::from_array(lhs), Simd::from_array(rhs)).to_array();
        for lane in 0..LANES {
            store_f32(ctx.out, i + lane, out[lane]);
        }
        i += LANES;
    }
    while i < span {
        let v = op(
            Simd::splat(load_f32(a, base + i)),
            Simd::splat(load_f32(b, base + i)),
        )[0];
        store_f32(ctx.out, i, v);
        i += 1;
    }
}

fn add_f32(ctx: TileCtx<'_>) {
    elementwise(ctx, |a, b| a + b);
}

fn mul_f32(ctx: TileCtx<'_>) {
    elementwise(ctx, |a, b| a * b);
}

/// Keep the `SimdFloat` import load-bearing rather than speculative: the trait
/// is what makes the lane-wise arithmetic above resolve to the portable ops.
const _: fn() = || {
    let _ = Simd::<f32, LANES>::splat(0.0).reduce_sum();
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec;

    fn bytes_of(values: &[f32]) -> vec::Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn floats_of(bytes: &[u8]) -> vec::Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// The identity round-trips, and every non-conforming shape refuses.
    #[test]
    fn identities_round_trip_and_fail_closed() {
        let kernel = &BUILTIN[0];
        assert_eq!(kernel_id(kernel.identity()), Some(kernel.id));

        assert_eq!(
            kernel_id(TargetIdentity([0; 12])),
            None,
            "a zero identity is not this format's"
        );

        let mut wrong_magic = kernel.identity();
        wrong_magic.0[0] ^= 1;
        assert_eq!(kernel_id(wrong_magic), None, "another format's artifact");

        let mut reserved_set = kernel.identity();
        reserved_set.0[7] = 1;
        assert_eq!(
            kernel_id(reserved_set),
            None,
            "a set reserved word must refuse, not be ignored"
        );
    }

    /// A kernel run over one whole span computes the elementwise result —
    /// including the tail below the SIMD width.
    #[test]
    fn add_computes_over_a_full_span_including_the_tail() {
        // 19 elements: two full 8-lane steps plus a 3-element tail.
        let a: vec::Vec<f32> = (0..19).map(|i| i as f32).collect();
        let b: vec::Vec<f32> = (0..19).map(|i| (i * 2) as f32).collect();
        let src_a = bytes_of(&a);
        let src_b = bytes_of(&b);
        let mut out = vec![0u8; 19 * 4];

        add_f32(TileCtx {
            tiles: 0..19,
            total: 19,
            launch: LaunchShape::ArtifactDefined,
            inputs: &[&src_a, &src_b],
            out: &mut out,
        });

        let got = floats_of(&out);
        for i in 0..19 {
            assert_eq!(got[i], a[i] + b[i], "element {i}");
        }
    }

    /// The split is the whole safety argument, so it has to compose: running
    /// two disjoint spans into their own slices must equal one whole run.
    #[test]
    fn disjoint_spans_compose_into_the_whole_result() {
        let a: vec::Vec<f32> = (0..20).map(|i| i as f32).collect();
        let b: vec::Vec<f32> = (0..20).map(|i| (i + 7) as f32).collect();
        let src_a = bytes_of(&a);
        let src_b = bytes_of(&b);

        let mut whole = vec![0u8; 20 * 4];
        mul_f32(TileCtx {
            tiles: 0..20,
            total: 20,
            launch: LaunchShape::ArtifactDefined,
            inputs: &[&src_a, &src_b],
            out: &mut whole,
        });

        let mut split = vec![0u8; 20 * 4];
        {
            let (lo, hi) = split.split_at_mut(9 * 4);
            mul_f32(TileCtx {
                tiles: 0..9,
                total: 20,
                launch: LaunchShape::ArtifactDefined,
                inputs: &[&src_a, &src_b],
                out: lo,
            });
            mul_f32(TileCtx {
                tiles: 9..20,
                total: 20,
                launch: LaunchShape::ArtifactDefined,
                inputs: &[&src_a, &src_b],
                out: hi,
            });
        }
        assert_eq!(whole, split, "a tiled run equals the whole run");
    }

    /// Every registered kernel is reachable by its own identity, and ids are
    /// unique — a duplicate would make `lookup` silently prefer one.
    #[test]
    fn the_registry_is_a_function() {
        for kernel in BUILTIN {
            let found = lookup(BUILTIN, kernel.id).expect("registered");
            assert_eq!(found.name, kernel.name);
        }
        for (i, k) in BUILTIN.iter().enumerate() {
            assert!(
                !BUILTIN[..i].iter().any(|prev| prev.id == k.id),
                "duplicate kernel id {}",
                k.id
            );
            assert!(k.elem_bytes > 0, "{} needs a tiling unit", k.name);
        }
        assert!(lookup(BUILTIN, 0xdead_beef).is_none(), "unknown id misses");
    }
}
