//! **The host-memory DMA double** — a [`DmaPool`] over leaked, page-aligned
//! process memory with `iova == cpu pointer`, so a driver core's mock device
//! can dereference the addresses the core programmed directly.
//!
//! One copy, here, rather than one per driver crate: `kore-nvme`,
//! `kore-xhci`, `kore-e1000e` and `kore-ivpu` each grew an identical
//! `MockPool` beside their mock device (the `MockNvmeDevice` lesson: prove
//! the semantics on the host, prove the seam on hardware), and the fifth
//! driver is where a seam gets standardized instead of copied. Compiled only
//! for host test builds (`host-double`, which a driver crate enables in its
//! `[dev-dependencies]`), never into a service or the kernel.

use alloc::alloc::{Layout, alloc_zeroed};

use kore_abi::device::IovaAddr;

use crate::seam::{Dma, DmaPool};

/// The host page: the floor every allocation is rounded and aligned to,
/// matching what the metal pool's leaf granularity guarantees.
pub const PAGE: usize = 4096;

/// A DMA buffer over leaked, page-aligned host memory.
pub struct HostBuf {
    ptr: *mut u8,
    len: usize,
}

impl Dma for HostBuf {
    fn cpu(&self) -> *mut u8 {
        self.ptr
    }
    fn iova(&self) -> IovaAddr {
        IovaAddr(self.ptr as u64)
    }
    fn len(&self) -> usize {
        self.len
    }
}

/// Hands out page-aligned zeroed host buffers (leaked — valid for the test
/// process, so a returned [`HostBuf`] never dangles).
#[derive(Clone, Copy, Default)]
pub struct HostPool;

impl DmaPool for HostPool {
    type Buf = HostBuf;
    fn alloc(&self, bytes: usize) -> Option<HostBuf> {
        self.alloc_aligned(bytes, PAGE)
    }
    fn alloc_aligned(&self, bytes: usize, alignment: usize) -> Option<HostBuf> {
        let alignment = alignment.max(PAGE);
        if !alignment.is_power_of_two() {
            return None;
        }
        let len = bytes.next_multiple_of(PAGE).max(PAGE);
        let layout = Layout::from_size_align(len, alignment).ok()?;
        // SAFETY: non-zero, well-formed layout; leaked for the process.
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return None;
        }
        Some(HostBuf { ptr, len })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffers_are_page_aligned_zeroed_and_addressed_by_their_own_pointer() {
        let pool = HostPool;
        let buf = pool.alloc(10).expect("host memory");
        assert_eq!(buf.len(), PAGE);
        assert_eq!(buf.cpu() as usize % PAGE, 0);
        assert_eq!(buf.iova().0, buf.cpu() as u64);
        // SAFETY: a fresh zeroed allocation of `len` bytes.
        assert!(
            unsafe { core::slice::from_raw_parts(buf.cpu(), buf.len()) }
                .iter()
                .all(|&b| b == 0)
        );
        let wide = pool
            .alloc_aligned(3 * PAGE + 1, 2 * PAGE)
            .expect("host memory");
        assert_eq!(wide.len(), 4 * PAGE);
        assert_eq!(wide.cpu() as usize % (2 * PAGE), 0);
        assert!(
            pool.alloc_aligned(PAGE, 3 * PAGE).is_none(),
            "a non-power-of-two alignment"
        );
    }
}
