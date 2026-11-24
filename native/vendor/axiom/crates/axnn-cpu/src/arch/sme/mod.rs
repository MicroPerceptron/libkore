//! SME tier — the Scalable Matrix Extension outer-product engine
//! (Apple M4+, Armv9 SME/SME2 cores).
//!
//! Strictly opt-in (`feature = "sme"`), same doctrine as Intel AMX:
//! streaming-mode and ZA state are runtime contracts the target
//! feature set cannot promise, so the builder asserts the hardware.
//! On Apple Silicon, check `hw.optional.arm.FEAT_SME` before enabling.

pub(crate) mod gemm;
