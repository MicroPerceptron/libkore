//! Portable scalar fallback — no SIMD, always compiled.
//!
//! Every kernel here is a plain loop or `core` primitive. Correct
//! first, reasonably efficient second (no naive byte-at-a-time where
//! `copy_nonoverlapping` or slice ops will do). This is the baseline
//! that all SIMD tiers must match in behavior.

// Functions are consumed via glob re-export in arch/mod.rs — the compiler
// can't see through the cfg-gated `use self::scalar::*` from here.
#[allow(dead_code)]
pub(crate) mod cast;
#[allow(dead_code)]
pub(crate) mod conv;
#[allow(dead_code)]
pub(crate) mod gemm;
#[allow(dead_code)]
pub(crate) mod mem;
#[allow(dead_code)]
pub(crate) mod qdot;
#[allow(dead_code)]
pub(crate) mod raster;
#[allow(dead_code)]
pub(crate) mod reduce;
#[allow(dead_code)]
pub(crate) mod rope;
#[allow(dead_code)]
pub(crate) mod vec;
