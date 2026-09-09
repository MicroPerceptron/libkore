//! Task-local execution advice. None of these facts grants ISA authority.
//!
//! A snapshot may be stale immediately after a load (migration). Consumers
//! must stay correct on every CPU in their execution domain. The packed word
//! needs no seqlock retry and carries no capability or cross-task activity.

use core::sync::atomic::{AtomicU64, Ordering};

/// One read-only page follows the founding cap-slot alias (geometry ABI 6).
pub const HINT_PAGE_BYTES: u64 = 4096;

/// Page-aligned hint offset, independent of later arena growth.
pub const fn hint_offset(slots: u32) -> u64 {
    (slots as u64 * 128 + HINT_PAGE_BYTES - 1) & !(HINT_PAGE_BYTES - 1)
}

/// Entire read-only founding alias: records, padding, and execution advice.
pub const fn alias_bytes(slots: u32) -> u64 {
    hint_offset(slots) + HINT_PAGE_BYTES
}

/// Observed CPU facts, not an instruction-set permission or a speed guarantee.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutionHint(u64);

impl ExecutionHint {
    /// Unknown fields are zero. Cache sizes are rounded down to powers of two;
    /// capacity is the existing boot streaming ratio in sixteenths of the BSP.
    pub const fn new(cpu: u8, class: u8, vector_bytes: u8, caches: [u32; 3], capacity: u8) -> Self {
        const fn exponent(bytes: u32) -> u64 {
            if bytes == 0 {
                0
            } else {
                (31 - bytes.leading_zeros()) as u64
            }
        }
        Self(
            0x01_u64 << 56
                | cpu as u64
                | (class as u64) << 8
                | (vector_bytes as u64) << 16
                | exponent(caches[0]) << 24
                | exponent(caches[1]) << 32
                | exponent(caches[2]) << 40
                | (capacity as u64) << 48,
        )
    }

    pub const fn from_bits(bits: u64) -> Option<Self> {
        let class = (bits >> 8) as u8;
        let width = (bits >> 16) as u8;
        if bits >> 56 != 1
            || class > 2
            || !matches!(width, 0 | 16 | 32 | 64)
            || (bits >> 24) as u8 > 31
            || (bits >> 32) as u8 > 31
            || (bits >> 40) as u8 > 31
        {
            None
        } else {
            Some(Self(bits))
        }
    }

    pub const fn bits(self) -> u64 {
        self.0
    }
    pub const fn with_capacity(self, capacity: u8) -> Self {
        Self((self.0 & !(0xff_u64 << 48)) | (capacity as u64) << 48)
    }
    pub const fn cpu(self) -> u8 {
        self.0 as u8
    }
    pub const fn core_class(self) -> u8 {
        (self.0 >> 8) as u8
    }
    pub const fn vector_bytes(self) -> u8 {
        (self.0 >> 16) as u8
    }
    pub const fn capacity_sixteenths(self) -> u8 {
        (self.0 >> 48) as u8
    }
    pub const fn l1_bytes(self) -> Option<u32> {
        let n = (self.0 >> 24) as u8;
        if n == 0 { None } else { Some(1_u32 << n) }
    }
    pub const fn l2_bytes(self) -> Option<u32> {
        let n = (self.0 >> 32) as u8;
        if n == 0 { None } else { Some(1_u32 << n) }
    }
    pub const fn l3_bytes(self) -> Option<u32> {
        let n = (self.0 >> 40) as u8;
        if n == 0 { None } else { Some(1_u32 << n) }
    }
}

/// Kernel is the sole writer; distinct tasks never share this cache line.
#[repr(C, align(64))]
pub struct ExecutionHintCell(AtomicU64);

impl ExecutionHintCell {
    pub const fn new() -> Self {
        Self(AtomicU64::new(0))
    }
    #[inline]
    pub fn read(&self) -> Option<ExecutionHint> {
        ExecutionHint::from_bits(self.0.load(Ordering::Relaxed))
    }
    /// No associated memory is published. The owning scheduler serializes
    /// writers; same-value resumes perform no store or locked RMW.
    #[inline]
    pub fn publish(&self, hint: ExecutionHint) {
        if self.0.load(Ordering::Relaxed) != hint.bits() {
            self.0.store(hint.bits(), Ordering::Relaxed);
        }
    }
}

impl Default for ExecutionHintCell {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    #[test]
    #[ignore = "release-mode advisory-cell overhead probe"]
    fn execution_hint_cost_probe() {
        use std::{hint::black_box, time::Instant};
        let cell = ExecutionHintCell::new();
        let a = ExecutionHint::new(0, 1, 32, [32768, 1048576, 8388608], 16);
        let b = a.with_capacity(8);
        cell.publish(a);
        for mode in 0..3 {
            let mut samples = [0_u128; 31];
            for time in &mut samples {
                let started = Instant::now();
                for i in 0..100_000 {
                    match mode {
                        0 => {
                            black_box(black_box(&cell).read());
                        }
                        1 => black_box(&cell).publish(black_box(a)),
                        _ => black_box(&cell).publish(black_box(if i & 1 == 0 { a } else { b })),
                    }
                }
                *time = started.elapsed().as_nanos();
            }
            samples.sort_unstable();
            std::eprintln!(
                "execution hint mode={mode} 100000 calls ns p50={} p95={} max={}",
                samples[15],
                samples[29],
                samples[30]
            );
        }
    }
    #[test]
    fn snapshots_are_total_and_unknown_versions_fail_closed() {
        let cell = ExecutionHintCell::new();
        assert_eq!(cell.read(), None);
        let a = ExecutionHint::new(0, 1, 32, [48 * 1024, 2 << 20, 8 << 20], 16);
        cell.publish(a);
        cell.publish(a);
        assert_eq!(cell.read(), Some(a));
        assert_eq!(a.l1_bytes(), Some(32768));
        assert_eq!(a.l2_bytes(), Some(2 << 20));
        assert_eq!(a.l3_bytes(), Some(8 << 20));
        let b = ExecutionHint::new(7, 2, 16, [0; 3], 8);
        cell.publish(b);
        assert_eq!(cell.read(), Some(b));
        assert_eq!(a.cpu(), 0); // An old snapshot is still just advice.
        assert_eq!(ExecutionHint::from_bits(u64::MAX), None);
        assert_eq!(alias_bytes(256), 9 * 4096);
        assert_eq!(alias_bytes(32), 2 * 4096);
        assert_eq!(hint_offset(1), 4096);
    }
}
