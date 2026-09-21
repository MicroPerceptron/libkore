//! `wasi:random` — libkore's seeded ChaCha generator.
//!
//! One `SystemRng` per guest, as libkore asks for repeated draws: it
//! amortizes hardware entropy and never falls back to a deterministic seed.
//! An entropy refusal is a trap, not a weak answer. The insecure interfaces
//! draw from the same generator — a guest asking for less than
//! cryptographic randomness gets cryptographic randomness anyway.

use alloc::vec;
use alloc::vec::Vec;

use crate::bindings::wasi::random::{insecure, insecure_seed, random};
use crate::{WasiCtx, WasiCtxView};

/// Largest single draw a guest may request: WASI's `len` is a `u64`, and an
/// unbounded one would be an allocation the guest chooses.
const MAX_DRAW: u64 = 1 << 20;

impl WasiCtx {
    fn draw(&mut self, len: u64) -> wasmtime::Result<Vec<u8>> {
        if len > MAX_DRAW {
            wasmtime::bail!("random draw of {len} bytes exceeds {MAX_DRAW}");
        }
        let mut bytes = vec![0; len as usize];
        self.rng
            .fill(&mut bytes)
            .map_err(|e| wasmtime::format_err!("entropy unavailable: {e:?}"))?;
        Ok(bytes)
    }

    fn word(&mut self) -> wasmtime::Result<u64> {
        self.rng
            .random()
            .map_err(|e| wasmtime::format_err!("entropy unavailable: {e:?}"))
    }
}

impl random::Host for WasiCtxView<'_> {
    fn get_random_bytes(&mut self, len: u64) -> wasmtime::Result<Vec<u8>> {
        self.ctx.draw(len)
    }
    fn get_random_u64(&mut self) -> wasmtime::Result<u64> {
        self.ctx.word()
    }
}

impl insecure::Host for WasiCtxView<'_> {
    fn get_insecure_random_bytes(&mut self, len: u64) -> wasmtime::Result<Vec<u8>> {
        self.ctx.draw(len)
    }
    fn get_insecure_random_u64(&mut self) -> wasmtime::Result<u64> {
        self.ctx.word()
    }
}

impl insecure_seed::Host for WasiCtxView<'_> {
    fn insecure_seed(&mut self) -> wasmtime::Result<(u64, u64)> {
        Ok((self.ctx.word()?, self.ctx.word()?))
    }
}
