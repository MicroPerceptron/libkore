#![no_std]

//! **One fill, fenced.**  The kernel zeroes every frame it hands out, a
//! driver scrubs every DMA page it re-lends, and the boot probe measures
//! what that costs — and they all did it through `core::ptr::write_bytes`,
//! whose ordinary stores pull each line in before overwriting it.  On a
//! 2026 laptop that read-for-ownership is half the bus traffic of a scrub
//! (measured on Lunar Lake: 46.8 GiB/s of fills was ~94 GiB/s of traffic
//! against a 127 GiB/s bus, `res/ACCEL_MEMORY_SCALE.md` §4 slice 5).
//!
//! [`fill_streaming`] is the same contract as `write_bytes` with streaming
//! (non-temporal) stores where the ISA offers them, so a scrub costs one
//! transfer per line, and a store fence before return, so the bytes are
//! globally visible — to a device about to DMA, to a page walker about to
//! read a table, to another CPU — the moment the call returns.  Where the
//! ISA has no such store the fill is `write_bytes`, and
//! [`streaming_available`] says which the caller got, so a measurement can
//! label its row honestly.

/// Whether [`fill_streaming`] uses streaming stores on this target.
#[inline]
pub const fn streaming_available() -> bool {
    cfg!(target_arch = "x86_64")
}

/// Fill `len` bytes at `dst` with `byte`.  Same contract as
/// [`core::ptr::write_bytes`] plus a store fence: on return every byte is
/// globally visible.
///
/// # Safety
/// `dst..dst+len` must be valid for writes and exclusively the caller's for
/// the duration, exactly as for `write_bytes`.
#[inline]
pub unsafe fn fill_streaming(dst: *mut u8, byte: u8, len: usize) {
    // SAFETY: forwarded from the caller's contract.
    unsafe { imp::fill(dst, byte, len) }
}

#[cfg(target_arch = "x86_64")]
mod imp {
    use core::arch::x86_64::{__m128i, _mm_set1_epi8, _mm_sfence, _mm_stream_si128};

    /// Bytes per streaming store (one SSE2 register — the x86_64
    /// baseline, so the kernel target and every user tier link the same
    /// body; a 64-byte line is four of them and the write-combining
    /// buffer merges them before the line leaves the core).
    const LANE: usize = 16;
    /// Below this the alignment prologue and the fence cost more than the
    /// read-for-ownership they save; plain stores win.
    const STREAM_MIN: usize = 4 * 1024;

    #[inline]
    pub(super) unsafe fn fill(dst: *mut u8, byte: u8, len: usize) {
        if len < STREAM_MIN {
            // SAFETY: the caller's contract.
            unsafe { core::ptr::write_bytes(dst, byte, len) };
            return;
        }
        let head = dst.align_offset(LANE).min(len);
        // SAFETY: `head <= len`, inside the caller's range.
        unsafe { core::ptr::write_bytes(dst, byte, head) };
        let body_start = head;
        let body_len = (len - head) / LANE * LANE;
        let lanes = body_len / LANE;
        let value: __m128i = _mm_set1_epi8(byte as i8);
        // SAFETY: `dst + body_start` is 16-aligned by construction and
        // `body_len` bytes from it lie inside the caller's range.
        unsafe {
            let mut p = dst.add(body_start) as *mut __m128i;
            let mut i = 0;
            while i + 4 <= lanes {
                _mm_stream_si128(p, value);
                _mm_stream_si128(p.add(1), value);
                _mm_stream_si128(p.add(2), value);
                _mm_stream_si128(p.add(3), value);
                p = p.add(4);
                i += 4;
            }
            while i < lanes {
                _mm_stream_si128(p, value);
                p = p.add(1);
                i += 1;
            }
        }
        let tail_start = body_start + body_len;
        // SAFETY: the remaining `len - tail_start < LANE` bytes.
        unsafe { core::ptr::write_bytes(dst.add(tail_start), byte, len - tail_start) };
        // Streaming stores are weakly ordered: drain the write-combining
        // buffers before the caller publishes the memory to anyone.
        _mm_sfence();
    }
}

#[cfg(not(target_arch = "x86_64"))]
mod imp {
    #[inline]
    pub(super) unsafe fn fill(dst: *mut u8, byte: u8, len: usize) {
        // SAFETY: the caller's contract.
        unsafe { core::ptr::write_bytes(dst, byte, len) };
        // Plain stores; the caller's own release ordering already covers
        // them.  (aarch64's `DC ZVA` / `STNP` and a RISC-V NTL variant
        // belong here when a box measures a reason.)
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use std::vec;

    use super::*;

    #[test]
    fn fills_every_byte_at_every_alignment_and_length() {
        // Long enough to take the streaming body, short enough to be quick;
        // every head offset and a tail of every residue.
        for offset in 0..32 {
            for len in [
                0usize,
                1,
                15,
                16,
                17,
                4095,
                4096,
                4097,
                8192 + 37,
                65536 + 63,
            ] {
                let mut buf = vec![0xEEu8; offset + len + 32];
                let dst = unsafe { buf.as_mut_ptr().add(offset) };
                unsafe { fill_streaming(dst, 0x5A, len) };
                assert!(buf[..offset].iter().all(|&b| b == 0xEE), "head clobbered");
                assert!(
                    buf[offset..offset + len].iter().all(|&b| b == 0x5A),
                    "offset {offset} len {len}"
                );
                assert!(
                    buf[offset + len..].iter().all(|&b| b == 0xEE),
                    "tail clobbered"
                );
            }
        }
    }

    #[test]
    fn availability_matches_the_target() {
        assert_eq!(streaming_available(), cfg!(target_arch = "x86_64"));
    }
}
