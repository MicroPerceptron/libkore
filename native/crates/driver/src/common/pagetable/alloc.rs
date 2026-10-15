//! The page-table frame source the walker builds its tree from, plus the shared
//! error vocabulary.
//!
//! A device MMU has **no HHDM**: an entry stores only an output address (an
//! IOVA for an SMMUv3 instance), so to descend into a table built by an earlier
//! map the walker must map that output address back to a CPU pointer — hence
//! [`TableSource::resolve`], the device-side analogue of the kernel
//! `FrameAllocator::phys_to_virt`. The reference impl (a driver's) wraps kdrv's
//! [`DmaPool`](crate::seam::DmaPool): one table frame is one `Dma` buffer, whose
//! `cpu()` / `iova()` are exactly the two halves this trait needs.

use super::addr::DeviceAddr;

/// Errors from a device-MMU map / unmap / command-encode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmuError {
    /// The table source is exhausted — no more table frames.
    OutOfTables,
    /// A VA / output address / length was not aligned to the required granule.
    Misaligned,
    /// An input VA or output address exceeded the format's addressable range.
    OutOfRange,
    /// A mapping already covers the target (leaf present, or a larger block
    /// spans it).
    AlreadyMapped,
    /// The walk hit no mapping where one was required (unmap / a missing table).
    NotMapped,
    /// The requested leaf size is not supported by the format at any level.
    UnsupportedLeafSize,
    /// A command-encode argument (range / granule / address) was invalid.
    InvalidRange,
    /// A command-encode address was misaligned or out of the encodable field.
    InvalidAddress,
    /// A command-encode granule (TTL hint) was unsupported.
    InvalidGranule,
    /// A requested access/attribute set cannot be represented without granting
    /// broader hardware authority than the caller requested.
    UnsupportedAccess,
    /// A command-encode range computation overflowed 64 bits.
    AddressOverflow,
    /// The metadata, entry representation, or supplied table frame cannot
    /// safely represent the declared table geometry.
    InvalidTableGeometry,
    /// A requested range would cover only part of an existing hardware leaf,
    /// or its backing extents do not exactly cover the input range.
    InvalidMappingShape,
    /// A present entry has an encoding that is invalid at its walk level.
    InvalidEntryKind,
    /// The table source could not resolve a table frame's output address back
    /// to a CPU pointer — the tree references a frame this source never
    /// allocated (a corrupted tree or a caller bug), surfaced as a typed
    /// refusal instead of a null-pointer dereference.
    UnknownTableFrame,
}

/// Result of a device-MMU operation.
pub type MmuResult<T = ()> = Result<T, MmuError>;

/// A source of zeroed page-table frames, and the reverse map from a frame's
/// output address to a CPU pointer.
pub trait TableSource {
    /// The output-address type frames are named by (equals the format's
    /// [`PageTableEntry::OutputAddr`](super::entry::PageTableEntry::OutputAddr)).
    type Addr: DeviceAddr;

    /// Allocate one **zeroed**, naturally-aligned table frame and return its
    /// device-visible output address (to store in the parent entry). `Err`
    /// with [`MmuError::OutOfTables`] when exhausted.
    fn alloc_table(&mut self) -> MmuResult<Self::Addr>;

    /// CPU pointer for a frame previously returned by
    /// [`alloc_table`](Self::alloc_table), or `None` when `addr` names no frame
    /// this source allocated (a corrupted tree or a caller bug — the walker
    /// propagates it as [`MmuError::UnknownTableFrame`]; never a null
    /// sentinel). A `Some` pointer is valid for [`table_bytes`] bytes; the
    /// walker reads it as a `[u64]`. Writers use
    /// [`resolve_for_write`](Self::resolve_for_write) before each mutation.
    ///
    /// [`table_bytes`]: Self::table_bytes
    fn resolve(&self, addr: Self::Addr) -> Option<*mut u8>;

    /// Resolve a frame before writing its entries. Sources with explicit
    /// cache publication can retain this write intent; coherent sources use
    /// the default. The pointer must not be retained across publication.
    fn resolve_for_write(&self, addr: Self::Addr) -> Option<*mut u8> {
        self.resolve(addr)
    }

    /// Release a frame during teardown. Best-effort — a source that cannot free
    /// individually may no-op and reclaim in bulk.
    fn release(&mut self, addr: Self::Addr);

    /// Frame size in bytes. Must match the format metadata's table-frame size.
    #[inline]
    fn table_bytes(&self) -> usize {
        0x1000
    }
}

/// The reference [`TableSource`] — one over a kdrv [`DmaPool`](crate::seam::DmaPool).
///
/// Each table frame is a single `Dma` buffer: `iova()` is the output address
/// stored in parent entries, `cpu()` the window the walker writes. Requires the
/// `alloc` feature — it owns a `Vec` of live frame buffers so their DMA windows
/// stay mapped for the tree's life and `resolve` can recover a CPU pointer by
/// IOVA (the device MMU has no HHDM to derive one).
#[cfg(feature = "alloc")]
mod dma_pool {
    use alloc::vec::Vec;
    use core::cell::Cell;

    use kore_abi::device::IovaAddr;

    use super::{MmuError, MmuResult, TableSource};
    use crate::seam::{Dma, DmaPool};

    struct TableFrame<B> {
        buffer: B,
        dirty: Cell<bool>,
    }

    /// [`TableSource`] backed by a [`DmaPool`]. See the module note.
    pub struct DmaPoolTables<P: DmaPool> {
        pool: P,
        // Sorted by the buffer's device address, including after release and
        // reuse. Walks resolve on every table read/write: a linear search here
        // multiplies their cost by the entire resident tree's frame count.
        frames: Vec<TableFrame<P::Buf>>,
        frame_bytes: usize,
    }

    impl<P: DmaPool> DmaPoolTables<P> {
        /// Take ownership of `pool`, handing out `frame_bytes`-sized table
        /// frames (4 KiB for the ARM granule). Owning the supply lets a founded
        /// device binding own this table source without a self-reference.
        pub fn new(pool: P, frame_bytes: usize) -> Self {
            Self {
                pool,
                frames: Vec::new(),
                frame_bytes,
            }
        }

        /// Number of live table frames.
        pub fn frame_count(&self) -> usize {
            self.frames.len()
        }

        /// Scoped access to the owned supply for other class-local DMA needs.
        pub const fn pool(&self) -> &P {
            &self.pool
        }

        fn frame_index(&self, addr: IovaAddr) -> Result<usize, usize> {
            self.frames
                .binary_search_by_key(&addr.0, |frame| frame.buffer.iova().0)
        }

        /// Visit every live table frame's CPU window.
        ///
        /// This is the cache-maintenance enumeration hook: a device whose
        /// table walker does not snoop the CPU cache must write back every
        /// frame the walker may read, and only this owner knows the frames.
        /// The visitor receives each frame's stable CPU base and the uniform
        /// frame length; it must not retain the pointer beyond the call.
        pub fn for_each_frame(&self, mut visit: impl FnMut(*mut u8, usize)) {
            for frame in &self.frames {
                visit(frame.buffer.cpu(), self.frame_bytes);
            }
        }

        /// Publish allocated or written frames, retaining dirtiness if either
        /// cache maintenance or its completion barrier fails. Callbacks must
        /// not mutate this source or retain its pointers. Read-only resolution
        /// never dirties a frame. This is cache publication, not TLB invalidation.
        pub fn publish_dirty_frames<E>(
            &self,
            mut publish: impl FnMut(*mut u8, usize) -> Result<(), E>,
            complete: impl FnOnce() -> Result<(), E>,
        ) -> Result<(), E> {
            for frame in &self.frames {
                if frame.dirty.get() {
                    publish(frame.buffer.cpu(), self.frame_bytes)?;
                }
            }
            complete()?;
            for frame in &self.frames {
                frame.dirty.set(false);
            }
            Ok(())
        }

        /// Recover the affine DMA supply after every table frame was released.
        ///
        /// A non-empty source is returned intact: callers must quiesce the
        /// device and successfully tear down every owned tree before unbinding.
        pub fn into_pool(self) -> Result<P, Self> {
            if self.frames.is_empty() {
                Ok(self.pool)
            } else {
                Err(self)
            }
        }
    }

    impl<P: DmaPool> TableSource for DmaPoolTables<P> {
        type Addr = IovaAddr;

        fn alloc_table(&mut self) -> MmuResult<IovaAddr> {
            let buf = self
                .pool
                .alloc(self.frame_bytes)
                .ok_or(MmuError::OutOfTables)?;
            let iova = buf.iova();
            let index = self
                .frame_index(iova)
                .err()
                .ok_or(MmuError::InvalidTableGeometry)?;
            self.frames.insert(
                index,
                TableFrame {
                    buffer: buf,
                    dirty: Cell::new(true),
                },
            );
            Ok(iova)
        }

        fn resolve(&self, addr: IovaAddr) -> Option<*mut u8> {
            // A miss (a frame this source never allocated) is a typed `None`
            // the walker turns into `MmuError::UnknownTableFrame` — in every
            // build profile, never a null sentinel.
            self.frame_index(addr)
                .ok()
                .map(|index| self.frames[index].buffer.cpu())
        }

        fn resolve_for_write(&self, addr: IovaAddr) -> Option<*mut u8> {
            self.frame_index(addr).ok().map(|index| {
                let frame = &self.frames[index];
                frame.dirty.set(true);
                frame.buffer.cpu()
            })
        }

        fn release(&mut self, addr: IovaAddr) {
            if let Ok(pos) = self.frame_index(addr) {
                // Drop returns the buffer to the pool. Preserve address order
                // so the hot read/write lookup needs no separate index.
                self.frames.remove(pos);
            }
        }

        fn table_bytes(&self) -> usize {
            self.frame_bytes
        }
    }

    #[cfg(test)]
    mod tests {
        use alloc::boxed::Box;
        use core::cell::Cell;

        use super::*;

        struct TestBuf {
            words: Box<[u64; 512]>,
            iova: IovaAddr,
            address_reads: Cell<usize>,
        }

        impl Dma for TestBuf {
            fn cpu(&self) -> *mut u8 {
                self.words.as_ptr() as *mut u8
            }

            fn iova(&self) -> IovaAddr {
                self.address_reads.set(self.address_reads.get() + 1);
                self.iova
            }

            fn len(&self) -> usize {
                core::mem::size_of_val(self.words.as_ref())
            }
        }

        struct TestPool {
            next: Cell<u64>,
        }

        impl DmaPool for TestPool {
            type Buf = TestBuf;

            fn alloc(&self, bytes: usize) -> Option<Self::Buf> {
                if bytes != 0x1000 {
                    return None;
                }
                let index = self.next.get();
                self.next.set(index + 1);
                Some(TestBuf {
                    words: Box::new([0; 512]),
                    iova: IovaAddr(index << 12),
                    address_reads: Cell::new(0),
                })
            }
        }

        #[test]
        fn table_lookup_preserves_identity_across_nonmonotonic_allocation_and_reuse() {
            let mut tables = DmaPoolTables::new(TestPool { next: Cell::new(1) }, 0x1000);
            // Sparse, descending allocations exercise the pool's freedom to
            // recycle addresses; table order is not allocation order.
            for page in [9, 5, 1, 7, 3] {
                tables.pool().next.set(page);
                let addr = tables.alloc_table().unwrap();
                // SAFETY: this test owns the live, zeroed frame; no device
                // accesses it, and the one-byte write is within its backing.
                unsafe { tables.resolve_for_write(addr).unwrap().write(page as u8) };
            }
            tables
                .publish_dirty_frames(|_, _| Ok::<_, ()>(()), || Ok(()))
                .unwrap();
            tables.release(IovaAddr(5 << 12));
            tables.pool().next.set(5);
            let reused = tables.alloc_table().unwrap();
            // SAFETY: resolution returns the owned live test frame's start.
            assert_eq!(unsafe { tables.resolve(reused).unwrap().read() }, 0);
            for page in [1, 3, 7, 9] {
                assert_eq!(
                    // SAFETY: these frames remain owned and have no device.
                    unsafe { tables.resolve(IovaAddr(page << 12)).unwrap().read() },
                    page as u8
                );
            }
            assert!(tables.resolve(IovaAddr(2 << 12)).is_none());
            assert!(tables.resolve(IovaAddr((3 << 12) + 1)).is_none());
            tables.pool().next.set(5);
            assert_eq!(tables.alloc_table(), Err(MmuError::InvalidTableGeometry));
            assert_eq!(tables.frame_count(), 5);
            tables.release(IovaAddr(2 << 12));
            assert_eq!(tables.frame_count(), 5);
            let mut dirty = 0;
            tables
                .publish_dirty_frames(
                    |_, _| {
                        dirty += 1;
                        Ok::<_, ()>(())
                    },
                    || Ok(()),
                )
                .unwrap();
            assert_eq!(dirty, 1, "only the recycled frame needs publication");
        }

        #[test]
        fn table_resolution_cost_grows_logarithmically_with_residency() {
            const FRAMES: usize = 1024;
            let mut tables = DmaPoolTables::new(TestPool { next: Cell::new(1) }, 0x1000);
            for _ in 0..FRAMES {
                tables.alloc_table().unwrap();
            }
            let probes = |tables: &DmaPoolTables<TestPool>| {
                tables
                    .frames
                    .iter()
                    .map(|frame| frame.buffer.address_reads.get())
                    .sum::<usize>()
            };
            // The previous linear scan touched every frame on these lookups.
            // Allow two comparisons beyond ceil(log2(n)) for search endpoints.
            let ceiling = (usize::BITS - (FRAMES - 1).leading_zeros()) as usize + 2;
            let before = probes(&tables);
            assert!(tables.resolve(IovaAddr((FRAMES as u64) << 12)).is_some());
            assert!(probes(&tables) - before <= ceiling);
            let before = probes(&tables);
            assert!(
                tables
                    .resolve_for_write(IovaAddr(((FRAMES + 1) as u64) << 12))
                    .is_none()
            );
            assert!(probes(&tables) - before <= ceiling);
        }

        #[test]
        fn the_table_source_returns_its_affine_supply_only_when_empty() {
            let pool = TestPool { next: Cell::new(1) };
            let mut tables = DmaPoolTables::new(pool, 0x1000);
            let frame = tables.alloc_table().unwrap();
            assert_eq!(tables.frame_count(), 1);
            assert_eq!(tables.pool().next.get(), 2);

            let mut tables = match tables.into_pool() {
                Ok(_) => panic!("a live table frame must retain its DMA supply"),
                Err(tables) => tables,
            };
            tables.release(frame);
            let pool = match tables.into_pool() {
                Ok(pool) => pool,
                Err(_) => panic!("an empty source must return its DMA supply"),
            };
            assert_eq!(pool.next.get(), 2);
        }
    }
}

#[cfg(feature = "alloc")]
pub use dma_pool::DmaPoolTables;
