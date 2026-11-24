//! AVX-VNNI qdot override — fused `vpdpbusd` (u8×i8 → i32) at 256-bit.
//!
//! Available on Intel Alder Lake+ and Zen 4+ without AVX-512.
//! Fuses the maddubs+madd chain into a single instruction.
#[allow(dead_code)]
pub(crate) mod f4;
#[allow(dead_code)]
pub(crate) mod qdot;
