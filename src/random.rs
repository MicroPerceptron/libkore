//! Automatically seeded random values over Kore's native entropy source.
//!
//! `random::<u64>()?` and `fill(&mut bytes)?` require no seed management.
//! Keep a `SystemRng` for repeated draws: it amortizes hardware entropy without
//! a global lock, TLS, allocation or a shared stream. Errors never fabricate
//! randomness; a failed fill leaves the caller's buffer unchanged.
#![doc = include_str!("../docs/random.md")]

use chacha20::{
    ChaCha20Rng,
    rand_core::{SeedableRng, TryRng},
};
use core::ops::Range;
use zeroize::Zeroizing;

pub use entropy::available;
/// Low-level hardware source; its fill may write a prefix before refusing.
/// Most applications should use this module's higher-level functions instead.
pub use kore_rt::random as entropy;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RandomError {
    EntropyUnavailable,
    InvalidRange,
    /// Bounded rejection sampling could not produce an unbiased draw.
    SamplingExhausted,
}

const RESEED_BYTES: usize = 64 * 1024;
const SAMPLE_ATTEMPTS: usize = 128;

/// A private, automatically seeded cryptographic generator. It is not cloneable
/// or serializable. Keep one per owner; sharing across threads requires the
/// caller's ordinary exclusive synchronization, not a library-global lock.
///
/// Fresh entropy is obtained before a request would exceed the 64 KiB budget.
/// A single larger fill is generated in full and forces reseeding on the next
/// call. Reseeding never occurs halfway through a fill: failure leaves output
/// unchanged. Seed temporaries and retained generator state zeroize on drop.
/// Snapshot/restore users must discard this owner and construct a fresh one.
pub struct SystemRng {
    engine: Option<ChaCha20Rng>,
    remaining: usize,
}

impl Default for SystemRng {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemRng {
    /// Lazy initialization; no entropy is consumed until the first draw.
    pub const fn new() -> Self {
        Self {
            engine: None,
            remaining: 0,
        }
    }

    pub fn fill(&mut self, bytes: &mut [u8]) -> Result<(), RandomError> {
        self.fill_with(bytes, |seed| {
            entropy::fill(seed).map_err(|_| RandomError::EntropyUnavailable)
        })
    }

    fn fill_with(
        &mut self,
        bytes: &mut [u8],
        mut seed_source: impl FnMut(&mut [u8; 32]) -> Result<(), RandomError>,
    ) -> Result<(), RandomError> {
        if bytes.is_empty() {
            return Ok(());
        }
        if self.engine.is_none() || bytes.len() > self.remaining {
            // Erase the old stream even when new entropy refuses. There is no
            // fallback to a stream past its reseed budget.
            self.engine = None;
            self.remaining = 0;
            let mut seed = Zeroizing::new([0u8; 32]);
            seed_source(&mut seed)?;
            self.engine = Some(ChaCha20Rng::from_seed(*seed));
            self.remaining = RESEED_BYTES;
        }
        // ChaCha's RNG has a 64-bit block counter. At most one usize-sized
        // request plus the bounded prior budget uses a key: no counter wrap.
        match self.engine.as_mut().unwrap().try_fill_bytes(bytes) {
            Ok(()) => {}
            Err(never) => match never {},
        }
        self.remaining = self.remaining.saturating_sub(bytes.len());
        Ok(())
    }

    pub fn random<T: Random>(&mut self) -> Result<T, RandomError> {
        T::draw(self)
    }

    /// Uniform half-open range. Rejection sampling avoids modulo bias; empty
    /// ranges refuse before entropy is consumed.
    pub fn random_range(&mut self, range: Range<u64>) -> Result<u64, RandomError> {
        sample_range(range, || self.random())
    }
}

/// Generate a typed value with automatic seeding. Reuse `SystemRng` for loops.
pub fn random<T: Random>() -> Result<T, RandomError> {
    SystemRng::new().random()
}
/// Fill all requested bytes or return an error without modifying the buffer.
pub fn fill(bytes: &mut [u8]) -> Result<(), RandomError> {
    SystemRng::new().fill(bytes)
}
/// Compatibility convenience for a single word.
pub fn u64() -> Result<u64, RandomError> {
    random()
}
pub fn random_range(range: Range<u64>) -> Result<u64, RandomError> {
    SystemRng::new().random_range(range)
}

mod sealed {
    pub trait Sealed {}
}
/// Supported primitive distributions. Integers use all bits; floats are uniform
/// on [0, 1) at their mantissa precision. No representation casting is used.
pub trait Random: sealed::Sealed + Sized {
    #[doc(hidden)]
    fn draw(rng: &mut SystemRng) -> Result<Self, RandomError>;
}
macro_rules! integers {
    ($($t:ty),*) => {$ (
        impl sealed::Sealed for $t {}
        impl Random for $t {
            fn draw(rng: &mut SystemRng) -> Result<Self, RandomError> {
                let mut bytes = [0; core::mem::size_of::<Self>()];
                rng.fill(&mut bytes)?;
                Ok(Self::from_le_bytes(bytes))
            }
        }
    )*};
}
integers!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);
impl sealed::Sealed for bool {}
impl Random for bool {
    fn draw(rng: &mut SystemRng) -> Result<Self, RandomError> {
        Ok(rng.random::<u8>()? & 1 != 0)
    }
}
impl sealed::Sealed for f32 {}
impl Random for f32 {
    fn draw(rng: &mut SystemRng) -> Result<Self, RandomError> {
        Ok((rng.random::<u32>()? >> 8) as f32 * (1.0 / 16777216.0))
    }
}
impl sealed::Sealed for f64 {}
impl Random for f64 {
    fn draw(rng: &mut SystemRng) -> Result<Self, RandomError> {
        Ok((rng.random::<u64>()? >> 11) as f64 * (1.0 / 9007199254740992.0))
    }
}
impl<const N: usize> sealed::Sealed for [u8; N] {}
impl<const N: usize> Random for [u8; N] {
    fn draw(rng: &mut SystemRng) -> Result<Self, RandomError> {
        let mut bytes = [0; N];
        rng.fill(&mut bytes)?;
        Ok(bytes)
    }
}
fn sample_range(
    range: Range<u64>,
    mut draw: impl FnMut() -> Result<u64, RandomError>,
) -> Result<u64, RandomError> {
    let width = range
        .end
        .checked_sub(range.start)
        .filter(|&n| n != 0)
        .ok_or(RandomError::InvalidRange)?;
    let threshold = width.wrapping_neg() % width;
    for _ in 0..SAMPLE_ATTEMPTS {
        let value = draw()?;
        if value >= threshold {
            return Ok(range.start + value % width);
        }
    }
    Err(RandomError::SamplingExhausted)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn failed_initialization_and_reseed_never_publish_a_prefix() {
        let mut rng = SystemRng::new();
        let mut out = [77; 65];
        assert_eq!(
            rng.fill_with(&mut out, |seed| {
                seed[..8].fill(9);
                Err(RandomError::EntropyUnavailable)
            }),
            Err(RandomError::EntropyUnavailable)
        );
        assert_eq!(out, [77; 65]);
        rng.fill_with(&mut out, |_| Ok(())).unwrap();
        rng.remaining = 0;
        out.fill(77);
        assert_eq!(
            rng.fill_with(&mut out, |_| Err(RandomError::EntropyUnavailable)),
            Err(RandomError::EntropyUnavailable)
        );
        assert_eq!(out, [77; 65]);
        assert!(rng.engine.is_none());
    }
    #[test]
    fn chacha_known_answer_and_chunked_stream_agree() {
        let mut rng = SystemRng::new();
        let mut block = [0; 64];
        rng.fill_with(&mut block, |_| Ok(())).unwrap();
        assert_eq!(
            &block[..16],
            &[
                0x76, 0xb8, 0xe0, 0xad, 0xa0, 0xf1, 0x3d, 0x90, 0x40, 0x5d, 0x6a, 0xe5, 0x53, 0x86,
                0xbd, 0x28
            ]
        );
        let mut other = SystemRng::new();
        // The upstream word-oriented RNG may discard a word's unused tail;
        // use word-aligned chunks for the stream-equivalence witness.
        let mut chunks = [0; 64];
        for part in chunks.chunks_mut(8) {
            other.fill_with(part, |_| Ok(())).unwrap();
        }
        assert_eq!(block, chunks);
    }
    #[test]
    fn empty_fill_needs_no_entropy_and_large_fill_reseeds_next_time() {
        let mut rng = SystemRng::new();
        rng.fill_with(&mut [], |_| panic!("empty fill seeded"))
            .unwrap();
        let mut bytes = [0; RESEED_BYTES + 1];
        let mut seeds = 0;
        rng.fill_with(&mut bytes, |_| {
            seeds += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(seeds, 1);
        rng.fill_with(&mut [0], |_| {
            seeds += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(seeds, 2);
    }
    #[test]
    fn rejection_is_unbiased_bounded_and_checks_range_before_drawing() {
        assert_eq!(
            sample_range(4..4, || panic!("invalid range drew")),
            Err(RandomError::InvalidRange)
        );
        let mut draws = [0, 5, 16].into_iter(); // threshold is 6 for width 10
        assert_eq!(sample_range(10..20, || Ok(draws.next().unwrap())), Ok(16));
        assert_eq!(
            sample_range(0..10, || Ok(0)),
            Err(RandomError::SamplingExhausted)
        );
        assert_eq!(
            sample_range(u64::MAX - 1..u64::MAX, || Ok(u64::MAX)),
            Ok(u64::MAX - 1)
        );
    }
    #[test]
    fn typed_draws_cover_primitives_without_another_seed() {
        let mut rng = SystemRng::new();
        rng.fill_with(&mut [0; 4], |_| Ok(())).unwrap();
        let _: u128 = rng.random().unwrap();
        let _: i128 = rng.random().unwrap();
        let _: usize = rng.random().unwrap();
        let _: bool = rng.random().unwrap();
        let _: [u8; 37] = rng.random().unwrap();
        for _ in 0..128 {
            let single: f32 = rng.random().unwrap();
            let double: f64 = rng.random().unwrap();
            assert!((0.0..1.0).contains(&single));
            assert!((0.0..1.0).contains(&double));
            assert!((10..20).contains(&rng.random_range(10..20).unwrap()));
        }
    }
}
