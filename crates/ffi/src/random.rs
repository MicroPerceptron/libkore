use kore::random::{RandomError, SystemRng};

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KoreRandomStatus {
    Ok = 0,
    InvalidArgument = 1,
    EntropyUnavailable = 2,
    InvalidRange = 3,
    SamplingExhausted = 4,
}

/// Rust-owned optional generator for repeated foreign draws. Never copy its
/// bytes or share it concurrently. The free functions need no owner argument.
pub struct KoreRandom {
    rng: SystemRng,
}
impl Default for KoreRandom {
    fn default() -> Self {
        Self::new()
    }
}
impl KoreRandom {
    pub const fn new() -> Self {
        Self {
            rng: SystemRng::new(),
        }
    }
}

fn status(result: Result<(), RandomError>) -> KoreRandomStatus {
    match result {
        Ok(()) => KoreRandomStatus::Ok,
        Err(RandomError::EntropyUnavailable) => KoreRandomStatus::EntropyUnavailable,
        Err(RandomError::InvalidRange) => KoreRandomStatus::InvalidRange,
        Err(RandomError::SamplingExhausted) => KoreRandomStatus::SamplingExhausted,
    }
}
unsafe fn fill(rng: &mut SystemRng, dst: *mut u8, length: usize) -> KoreRandomStatus {
    if length == 0 {
        return KoreRandomStatus::Ok;
    }
    if dst.is_null() || length > isize::MAX as usize {
        return KoreRandomStatus::InvalidArgument;
    }
    // SAFETY: the exported call requires this exclusive writable range.
    status(rng.fill(unsafe { core::slice::from_raw_parts_mut(dst, length) }))
}
/// Automatically seed and fill the whole buffer. Any failure leaves it unchanged.
/// # Safety
/// dst is null for zero length, or names length writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_random_fill(dst: *mut u8, length: usize) -> KoreRandomStatus {
    // SAFETY: forwarded buffer contract.
    unsafe { fill(&mut SystemRng::new(), dst, length) }
}
/// Reuse a private, automatically seeded generator for repeated fills.
/// # Safety
/// rng is a live exclusive Rust-founded generator; dst names length writable
/// bytes disjoint from rng (or is null for zero length).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_random_fill_from(
    rng: *mut KoreRandom,
    dst: *mut u8,
    length: usize,
) -> KoreRandomStatus {
    // SAFETY: caller owns the generator exclusively for this call.
    let Some(rng) = (unsafe { rng.as_mut() }) else {
        return KoreRandomStatus::InvalidArgument;
    };
    // SAFETY: destination is exclusive and disjoint from the generator.
    unsafe { fill(&mut rng.rng, dst, length) }
}
/// A uniformly distributed u64. Failure leaves out unchanged.
/// # Safety
/// out is aligned writable storage for one u64.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_random_u64(out: *mut u64) -> KoreRandomStatus {
    if out.is_null() {
        return KoreRandomStatus::InvalidArgument;
    }
    match kore::random::random() {
        Ok(value) => {
            // SAFETY: caller supplied aligned writable output, validated non-null.
            unsafe {
                out.write(value);
            }
            KoreRandomStatus::Ok
        }
        Err(error) => status(Err(error)),
    }
}
/// A uniformly distributed value in [start, end). Failure leaves out unchanged.
/// # Safety
/// out is aligned writable storage for one u64.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_random_range(
    start: u64,
    end: u64,
    out: *mut u64,
) -> KoreRandomStatus {
    if out.is_null() {
        return KoreRandomStatus::InvalidArgument;
    }
    match kore::random::random_range(start..end) {
        Ok(value) => {
            // SAFETY: caller supplied aligned writable output, validated non-null.
            unsafe {
                out.write(value);
            }
            KoreRandomStatus::Ok
        }
        Err(error) => status(Err(error)),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_requests_and_empty_fills_need_no_entropy_or_output_write() {
        let mut out = 42;
        unsafe {
            assert_eq!(
                kore_random_fill(core::ptr::null_mut(), 0),
                KoreRandomStatus::Ok
            );
            assert_eq!(
                kore_random_fill(core::ptr::null_mut(), 1),
                KoreRandomStatus::InvalidArgument
            );
            assert_eq!(
                kore_random_range(10, 9, &mut out),
                KoreRandomStatus::InvalidRange
            );
        }
        assert_eq!(out, 42);
    }
}
