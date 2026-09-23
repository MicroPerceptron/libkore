//! The per-client bulk **data window** — every checked slice in one place.
//!
//! File bytes and directory listings cross through a region the *client*
//! carved out of its own memory and delivered into this service's arena,
//! following `OutReqWire::OP_ATTACH_TX`'s precedent. A request names an offset
//! and a length **into that window**; no address and no capability ever rides
//! the wire.
//!
//! Which makes every bulk request a bounds problem, and this the module that
//! owns it. The offsets arrive from another process as plain `u32`s, so each
//! one is checked-added and range-tested before a single byte moves — a wrapped
//! `offset + len` that landed back inside the window would be an out-of-bounds
//! read the service performed on its own memory at a peer's request.
//!
//! The window is a `&mut [u8]` rather than a mapped pointer so the whole of it
//! is testable on the host: on metal the serving host hands over its mapping of
//! the delivered view, and in tests a plain buffer stands in. Nothing here
//! knows which it has.

use kore_abi::svc::{FsDirEntryWire, FsSearchHitWire};

use crate::error::{FsError, FsResult};

/// A bounds-checked view of one client's data window.
pub struct DataWindow<'w> {
    bytes: &'w mut [u8],
}

impl<'w> DataWindow<'w> {
    /// Wrap a window's bytes.
    pub fn new(bytes: &'w mut [u8]) -> Self {
        Self { bytes }
    }

    /// The window's length in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// The `offset..offset+len` span, if it lies wholly inside the window.
    ///
    /// Every arithmetic step is checked: a `u32` pair from a peer can overflow
    /// `usize` on a 32-bit target and can wrap into a plausible-looking range
    /// on any target.
    fn span(&self, offset: u32, len: u32) -> FsResult<core::ops::Range<usize>> {
        let offset = usize::try_from(offset).map_err(|_| FsError::Range)?;
        let len = usize::try_from(len).map_err(|_| FsError::Range)?;
        let end = offset.checked_add(len).ok_or(FsError::Range)?;
        if end > self.bytes.len() {
            return Err(FsError::Range);
        }
        Ok(offset..end)
    }

    /// Borrow `len` bytes at `offset` for reading — the source of a write.
    pub fn source(&self, offset: u32, len: u32) -> FsResult<&[u8]> {
        let span = self.span(offset, len)?;
        Ok(&self.bytes[span])
    }

    /// Borrow `len` bytes at `offset` for writing — the destination of a read.
    pub fn destination(&mut self, offset: u32, len: u32) -> FsResult<&mut [u8]> {
        let span = self.span(offset, len)?;
        Ok(&mut self.bytes[span])
    }

    /// How many whole directory entries fit at `offset`.
    ///
    /// A listing writes fixed-size records, so a window with room for two and a
    /// half entries has room for two — a partial trailing entry would be
    /// garbage a client might parse.
    pub fn entry_capacity(&self, offset: u32) -> usize {
        let Ok(offset) = usize::try_from(offset) else {
            return 0;
        };
        self.bytes
            .len()
            .saturating_sub(offset)
            .checked_div(FsDirEntryWire::BYTES)
            .unwrap_or(0)
    }

    /// Place one directory entry at `offset + index * BYTES`.
    pub fn put_entry(&mut self, offset: u32, index: usize, entry: &FsDirEntryWire) -> FsResult<()> {
        use zerocopy::IntoBytes;
        self.put_bytes(offset, index, FsDirEntryWire::BYTES, entry.as_bytes())
    }

    /// Place one authority-edge entry at `offset + index * BYTES`.
    ///
    /// Same granule as [`Self::put_entry`], so a client's window arithmetic is
    /// identical for both listings.
    pub fn put_grant_entry(
        &mut self,
        offset: u32,
        index: usize,
        entry: &kore_abi::svc::FsGrantEntryWire,
    ) -> FsResult<()> {
        use zerocopy::IntoBytes;
        self.put_bytes(
            offset,
            index,
            kore_abi::svc::FsGrantEntryWire::BYTES,
            entry.as_bytes(),
        )
    }

    /// Land the one diagnostic snapshot record at `offset`.
    pub fn put_system_status(
        &mut self,
        offset: u32,
        record: &kore_abi::svc::FsSystemStatusWire,
    ) -> FsResult<()> {
        use zerocopy::IntoBytes;
        self.put_bytes(
            offset,
            0,
            kore_abi::svc::FsSystemStatusWire::BYTES,
            record.as_bytes(),
        )
    }

    /// How many complete structural-search hits fit at `offset`.
    pub fn search_capacity(&self, offset: u32) -> usize {
        let Ok(offset) = usize::try_from(offset) else {
            return 0;
        };
        self.bytes
            .len()
            .saturating_sub(offset)
            .checked_div(FsSearchHitWire::BYTES)
            .unwrap_or(0)
    }

    /// Place one fixed-size search hit at `offset + index * BYTES`.
    pub fn put_search_hit(
        &mut self,
        offset: u32,
        index: usize,
        hit: &FsSearchHitWire,
    ) -> FsResult<()> {
        use zerocopy::IntoBytes;
        self.put_bytes(offset, index, FsSearchHitWire::BYTES, hit.as_bytes())
    }

    /// The shared placement arithmetic behind both entry writers.
    fn put_bytes(
        &mut self,
        offset: u32,
        index: usize,
        stride: usize,
        bytes: &[u8],
    ) -> FsResult<()> {
        let step = index.checked_mul(stride).ok_or(FsError::Range)?;
        let at = usize::try_from(offset)
            .map_err(|_| FsError::Range)?
            .checked_add(step)
            .ok_or(FsError::Range)?;
        let end = at.checked_add(bytes.len()).ok_or(FsError::Range)?;
        if end > self.bytes.len() {
            return Err(FsError::Range);
        }
        self.bytes[at..end].copy_from_slice(bytes);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_span_inside_the_window_is_borrowable_both_ways() {
        let mut buf = [0u8; 64];
        let mut w = DataWindow::new(&mut buf);
        assert_eq!(w.len(), 64);
        assert!(!w.is_empty());

        w.destination(8, 4).unwrap().copy_from_slice(b"kore");
        assert_eq!(w.source(8, 4).unwrap(), b"kore");
        // Untouched bytes stayed zero — the borrow was exactly the span asked for.
        assert_eq!(w.source(0, 8).unwrap(), &[0u8; 8]);
        assert_eq!(w.source(12, 4).unwrap(), &[0u8; 4]);
    }

    #[test]
    fn the_whole_window_and_an_empty_span_are_both_legal() {
        let mut buf = [7u8; 16];
        let mut w = DataWindow::new(&mut buf);
        assert_eq!(w.source(0, 16).unwrap().len(), 16);
        assert_eq!(
            w.source(16, 0).unwrap().len(),
            0,
            "an empty span at the end"
        );
        assert_eq!(w.destination(0, 16).unwrap().len(), 16);
    }

    #[test]
    fn a_span_past_the_end_is_refused() {
        let mut buf = [0u8; 16];
        let mut w = DataWindow::new(&mut buf);
        assert_eq!(w.source(0, 17).unwrap_err(), FsError::Range);
        assert_eq!(w.source(16, 1).unwrap_err(), FsError::Range);
        assert_eq!(w.source(17, 0).unwrap_err(), FsError::Range);
        assert_eq!(w.destination(8, 9).unwrap_err(), FsError::Range);
    }

    #[test]
    fn an_overflowing_span_is_refused_not_wrapped() {
        // The dangerous case: a wrapped `offset + len` that lands back inside
        // the window would be an out-of-bounds access the service performs on
        // its own memory because a peer asked nicely.
        let mut buf = [0u8; 16];
        let mut w = DataWindow::new(&mut buf);
        assert_eq!(w.source(u32::MAX, 1).unwrap_err(), FsError::Range);
        assert_eq!(w.source(1, u32::MAX).unwrap_err(), FsError::Range);
        assert_eq!(
            w.destination(u32::MAX, u32::MAX).unwrap_err(),
            FsError::Range
        );
    }

    #[test]
    fn an_empty_window_admits_nothing() {
        let mut buf: [u8; 0] = [];
        let mut w = DataWindow::new(&mut buf);
        assert!(w.is_empty());
        assert_eq!(w.source(0, 0).unwrap().len(), 0);
        assert_eq!(w.source(0, 1).unwrap_err(), FsError::Range);
        assert_eq!(w.destination(0, 1).unwrap_err(), FsError::Range);
        assert_eq!(w.entry_capacity(0), 0);
    }

    #[test]
    fn entry_capacity_counts_only_whole_entries() {
        let mut buf = [0u8; FsDirEntryWire::BYTES * 2 + 8];
        let w = DataWindow::new(&mut buf);
        assert_eq!(w.entry_capacity(0), 2, "the trailing 8 bytes hold no entry");
        assert_eq!(w.entry_capacity(FsDirEntryWire::BYTES as u32), 1);
        assert_eq!(w.entry_capacity(buf_len_as_u32(&w)), 0);
    }

    fn buf_len_as_u32(w: &DataWindow<'_>) -> u32 {
        w.len() as u32
    }

    #[test]
    fn entry_capacity_saturates_rather_than_underflowing() {
        let mut buf = [0u8; 16];
        let w = DataWindow::new(&mut buf);
        // An offset past the end must report no room, not wrap to a huge count.
        assert_eq!(w.entry_capacity(1_000_000), 0);
        assert_eq!(w.entry_capacity(u32::MAX), 0);
    }

    #[test]
    fn entries_land_at_their_indexed_offsets() {
        use kore_abi::svc::FsReqWire;
        let mut buf = [0u8; FsDirEntryWire::BYTES * 3];
        let mut w = DataWindow::new(&mut buf);
        let mut entry = FsDirEntryWire {
            name_len: 1,
            kind: FsReqWire::KIND_FILE,
            reserved0: 0.into(),
            reserved1: 0.into(),
            object: 0x1234.into(),
            name: [0; kore_abi::svc::FS_NAME_BYTES],
        };
        entry.name[0] = b'a';
        w.put_entry(0, 0, &entry).unwrap();
        entry.name[0] = b'c';
        w.put_entry(0, 2, &entry).unwrap();

        assert_eq!(buf[0], 1, "first entry's name_len");
        assert_eq!(buf[16], b'a', "first entry's name");
        assert_eq!(
            buf[FsDirEntryWire::BYTES],
            0,
            "the skipped slot is untouched"
        );
        assert_eq!(buf[FsDirEntryWire::BYTES * 2 + 16], b'c');
    }

    #[test]
    fn an_entry_past_the_window_is_refused() {
        let mut buf = [0u8; FsDirEntryWire::BYTES];
        let mut w = DataWindow::new(&mut buf);
        let entry = FsDirEntryWire {
            name_len: 1,
            kind: kore_abi::svc::FsReqWire::KIND_FILE,
            reserved0: 0.into(),
            reserved1: 0.into(),
            object: 0.into(),
            name: [0; kore_abi::svc::FS_NAME_BYTES],
        };
        assert!(w.put_entry(0, 0, &entry).is_ok());
        assert_eq!(w.put_entry(0, 1, &entry).unwrap_err(), FsError::Range);
        assert_eq!(w.put_entry(1, 0, &entry).unwrap_err(), FsError::Range);
        // And an index that would overflow the multiply is refused, not wrapped.
        assert_eq!(
            w.put_entry(0, usize::MAX, &entry).unwrap_err(),
            FsError::Range
        );
    }
}
