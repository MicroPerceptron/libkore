//! Base-NEON memory utility tier.
//!
//! Slice primitives already lower to the target's optimized copy routines, so
//! the NEON tier shares their implementation while keeping dispatch rooted in
//! the AArch64 architecture module.

pub(crate) use crate::arch::portable::mem::*;
