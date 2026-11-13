//! **The device's view of the driver's memory.**
//!
//! A driver lends a virtual device one window of memory and names places in
//! it by offset: the offset is the *device address*, the only address a
//! driver ever hands a device. The device reaches the window through
//! [`Memory`], which refuses anything outside it, so a driver's mistake, or
//! its malice, stays inside what it lent.
//!
//! The other side of the window is another task that may write any byte at
//! any moment. Nothing here ever forms a Rust reference into it: every
//! access is an atomic load or store, a snapshot the device then validates.
//! A snapshot is not coherent across words; the ring's publication points
//! ([`Memory::load_u16`], [`Memory::store_u16`]) are what order one side's
//! writes before the other's reads.

use core::ptr::NonNull;
use core::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering};

/// Why a device address did not name memory the driver lent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryFault {
    /// Some byte of the range is outside the window.
    Range,
    /// A word access named an address that is not a multiple of its size.
    Alignment,
}

/// A window of driver memory, addressed from zero.
pub trait Memory {
    /// Bytes in the window: device addresses are `0..len`.
    fn len(&self) -> u64;

    /// Whether the window is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether `len` bytes at `at` lie inside the window.
    fn contains(&self, at: u64, len: u64) -> bool {
        at.checked_add(len).is_some_and(|end| end <= self.len())
    }

    /// Snapshot `target.len()` bytes at `at`.
    fn read(&self, at: u64, target: &mut [u8]) -> Result<(), MemoryFault>;

    /// Store `source` at `at`.
    fn write(&self, at: u64, source: &[u8]) -> Result<(), MemoryFault>;

    /// Load the little-endian `u16` at `at` with acquire ordering: every
    /// write the other side made before it published this word is visible
    /// to reads after this one.
    fn load_u16(&self, at: u64) -> Result<u16, MemoryFault>;

    /// Store a little-endian `u16` at `at` with release ordering: every
    /// write this side made before it is visible to the other side once it
    /// has loaded this word.
    fn store_u16(&self, at: u64, value: u16) -> Result<(), MemoryFault>;

    /// Load the little-endian `u32` at `at`, acquire.
    fn load_u32(&self, at: u64) -> Result<u32, MemoryFault>;

    /// Store a little-endian `u32` at `at`, release.
    fn store_u32(&self, at: u64, value: u32) -> Result<(), MemoryFault>;

    /// Load the little-endian `u64` at `at`, acquire.
    fn load_u64(&self, at: u64) -> Result<u64, MemoryFault>;

    /// Store a little-endian `u64` at `at`, release.
    fn store_u64(&self, at: u64, value: u64) -> Result<(), MemoryFault>;
}

/// A mapped window: `len` bytes at `base` in this task, shared with a driver.
///
/// A plain view, copied freely: it owns nothing, and whoever made it keeps
/// the mapping alive ([`Window::new`]).
#[derive(Clone, Copy, Debug)]
pub struct Window {
    base: NonNull<u8>,
    len: usize,
}

impl Window {
    /// A view of `len` bytes at `base`.
    ///
    /// # Safety
    /// `base` is aligned to eight bytes and `len` bytes from it are mapped
    /// readable and writable in this task for as long as this view or any
    /// copy of it is used. The bytes may be written by another task at any
    /// time; nothing else in this task holds a Rust reference into them.
    pub const unsafe fn new(base: NonNull<u8>, len: usize) -> Self {
        Self { base, len }
    }

    /// Where the window is mapped in this task. A host uses it to name a
    /// word of the window to the kernel, never to form a reference.
    pub const fn base(&self) -> NonNull<u8> {
        self.base
    }

    fn range(&self, at: u64, len: usize) -> Result<*mut u8, MemoryFault> {
        if !self.contains(at, len as u64) {
            return Err(MemoryFault::Range);
        }
        // SAFETY: `at + len <= self.len`, so the offset stays inside the
        // mapping `new` was promised.
        Ok(unsafe { self.base.as_ptr().add(at as usize) })
    }

    fn word(&self, at: u64, size: usize) -> Result<*mut u8, MemoryFault> {
        if at % size as u64 != 0 {
            return Err(MemoryFault::Alignment);
        }
        // `base` is eight-byte aligned, so an offset aligned to `size` is an
        // address aligned to it.
        self.range(at, size)
    }
}

/// Bytes moved a word at a time once both sides are word aligned.
const WORD: usize = core::mem::size_of::<u64>();

impl Memory for Window {
    fn len(&self) -> u64 {
        self.len as u64
    }

    fn read(&self, at: u64, target: &mut [u8]) -> Result<(), MemoryFault> {
        let source = self.range(at, target.len())?;
        // Each load is atomic, so a concurrent writer in the driver cannot
        // make this a data race; what it can do is change the bytes, which
        // is why the caller validates the copy and never the window.
        let mut index = 0;
        // SAFETY (all three loops): `source.add(index)` is inside the range
        // checked above for every `index < target.len()`.
        while index < target.len() && (source as usize + index) % WORD != 0 {
            target[index] =
                unsafe { AtomicU8::from_ptr(source.add(index)) }.load(Ordering::Relaxed);
            index += 1;
        }
        while index + WORD <= target.len() {
            let word =
                unsafe { AtomicU64::from_ptr(source.add(index).cast()) }.load(Ordering::Relaxed);
            target[index..index + WORD].copy_from_slice(&word.to_ne_bytes());
            index += WORD;
        }
        while index < target.len() {
            target[index] =
                unsafe { AtomicU8::from_ptr(source.add(index)) }.load(Ordering::Relaxed);
            index += 1;
        }
        Ok(())
    }

    fn write(&self, at: u64, source: &[u8]) -> Result<(), MemoryFault> {
        let target = self.range(at, source.len())?;
        let mut index = 0;
        // SAFETY (all three loops): as in `read`.
        while index < source.len() && (target as usize + index) % WORD != 0 {
            unsafe { AtomicU8::from_ptr(target.add(index)) }
                .store(source[index], Ordering::Relaxed);
            index += 1;
        }
        while index + WORD <= source.len() {
            let mut word = [0u8; WORD];
            word.copy_from_slice(&source[index..index + WORD]);
            unsafe { AtomicU64::from_ptr(target.add(index).cast()) }
                .store(u64::from_ne_bytes(word), Ordering::Relaxed);
            index += WORD;
        }
        while index < source.len() {
            unsafe { AtomicU8::from_ptr(target.add(index)) }
                .store(source[index], Ordering::Relaxed);
            index += 1;
        }
        Ok(())
    }

    fn load_u16(&self, at: u64) -> Result<u16, MemoryFault> {
        let word = self.word(at, core::mem::size_of::<u16>())?;
        // SAFETY: in range and aligned for a `u16`.
        Ok(u16::from_le(
            unsafe { AtomicU16::from_ptr(word.cast()) }.load(Ordering::Acquire),
        ))
    }

    fn store_u16(&self, at: u64, value: u16) -> Result<(), MemoryFault> {
        let word = self.word(at, core::mem::size_of::<u16>())?;
        // SAFETY: in range and aligned for a `u16`.
        unsafe { AtomicU16::from_ptr(word.cast()) }.store(value.to_le(), Ordering::Release);
        Ok(())
    }

    fn load_u32(&self, at: u64) -> Result<u32, MemoryFault> {
        let word = self.word(at, core::mem::size_of::<u32>())?;
        // SAFETY: in range and aligned for a `u32`.
        Ok(u32::from_le(
            unsafe { AtomicU32::from_ptr(word.cast()) }.load(Ordering::Acquire),
        ))
    }

    fn store_u32(&self, at: u64, value: u32) -> Result<(), MemoryFault> {
        let word = self.word(at, core::mem::size_of::<u32>())?;
        // SAFETY: in range and aligned for a `u32`.
        unsafe { AtomicU32::from_ptr(word.cast()) }.store(value.to_le(), Ordering::Release);
        Ok(())
    }

    fn load_u64(&self, at: u64) -> Result<u64, MemoryFault> {
        let word = self.word(at, core::mem::size_of::<u64>())?;
        // SAFETY: in range and aligned for a `u64`.
        Ok(u64::from_le(
            unsafe { AtomicU64::from_ptr(word.cast()) }.load(Ordering::Acquire),
        ))
    }

    fn store_u64(&self, at: u64, value: u64) -> Result<(), MemoryFault> {
        let word = self.word(at, core::mem::size_of::<u64>())?;
        // SAFETY: in range and aligned for a `u64`.
        unsafe { AtomicU64::from_ptr(word.cast()) }.store(value.to_le(), Ordering::Release);
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::boxed::Box;
    use std::vec;

    /// A window over host memory that lives for the test run.
    pub(crate) fn window(bytes: usize) -> Window {
        let words: &'static mut [u64] = Box::leak(vec![0u64; bytes.div_ceil(8)].into_boxed_slice());
        // SAFETY: a leaked, eight-byte aligned allocation of at least
        // `bytes` bytes that nothing else references.
        unsafe { Window::new(NonNull::new(words.as_mut_ptr().cast()).unwrap(), bytes) }
    }

    #[test]
    fn a_window_reaches_exactly_what_was_lent() {
        let memory = window(4096);
        assert_eq!(memory.len(), 4096);
        assert!(memory.contains(4090, 6));
        assert!(!memory.contains(4090, 7));
        assert!(!memory.contains(u64::MAX, 2), "the end does not wrap");

        let mut out = [0u8; 4];
        assert_eq!(memory.read(4093, &mut out), Err(MemoryFault::Range));
        assert_eq!(memory.write(4096, &[1]), Err(MemoryFault::Range));
        assert_eq!(memory.write(4096, &[]), Ok(()), "nothing, at the end");
        assert_eq!(memory.load_u16(4095), Err(MemoryFault::Alignment));
        assert_eq!(memory.load_u16(4096), Err(MemoryFault::Range));
        assert_eq!(memory.store_u32(2, 0), Err(MemoryFault::Alignment));
        assert_eq!(memory.load_u32(4096), Err(MemoryFault::Range));
        assert_eq!(memory.load_u64(4), Err(MemoryFault::Alignment));
        assert_eq!(memory.store_u64(4096, 0), Err(MemoryFault::Range));
    }

    /// Every alignment of a copy takes the byte head, the word body and the
    /// byte tail in some mix, and all of them move the same bytes.
    #[test]
    fn a_copy_moves_the_same_bytes_at_every_alignment() {
        let memory = window(256);
        let pattern: std::vec::Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(37) ^ 0x5a).collect();
        for at in 0..17u64 {
            for len in [0usize, 1, 7, 8, 9, 15, 16, 17, 31, 40] {
                memory.write(0, &[0xee; 96]).unwrap();
                memory.write(at, &pattern[..len]).unwrap();
                let mut back = vec![0u8; len];
                memory.read(at, &mut back).unwrap();
                assert_eq!(back, pattern[..len], "at {at}, {len} bytes");
                // The bytes either side are untouched.
                let mut around = [0u8; 96];
                memory.read(0, &mut around).unwrap();
                assert!(around[..at as usize].iter().all(|b| *b == 0xee));
                assert!(around[at as usize + len..].iter().all(|b| *b == 0xee));
            }
        }
    }

    #[test]
    fn words_are_little_endian_whatever_the_host_is() {
        let memory = window(16);
        memory.store_u16(2, 0x1234).unwrap();
        memory.store_u32(4, 0x89ab_cdef).unwrap();
        memory.store_u64(8, 0x0123_4567_89ab_cdef).unwrap();
        let mut bytes = [0u8; 16];
        memory.read(0, &mut bytes).unwrap();
        assert_eq!(bytes[..8], [0, 0, 0x34, 0x12, 0xef, 0xcd, 0xab, 0x89]);
        assert_eq!(bytes[8..], [0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01]);
        assert_eq!(memory.load_u16(2), Ok(0x1234));
        assert_eq!(memory.load_u32(4), Ok(0x89ab_cdef));
        assert_eq!(memory.load_u64(8), Ok(0x0123_4567_89ab_cdef));
    }
}
