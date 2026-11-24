//! SSE2 qdot override — delegates to portable.
//!
//! SSE2 lacks `pmaddubsw` (SSSE3), so there's no hand-tunable
//! instruction for quantized dot products.  The portable tier's
//! scalar loops already auto-vectorize to SSE2 via LLVM.

pub(crate) use super::portable::qdot;
