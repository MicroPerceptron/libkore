//! The bounded RAM [`StorageProvider`].
//!
//! This is not a mock. It is the architecture-neutral fallback composition,
//! and it drives the same engine, namespace, and authority model as the
//! block-backed provider. Its
//! durability promise is exactly what RAM can deliver: a successful mutation
//! is immediately visible to later authorized operations in this boot, and
//! everything is lost at reboot. It never reports more.
//!
//! # Shape
//!
//! A fixed pool of [`RAM_CHUNK_BYTES`]-sized chunks, claimed and released
//! through a per-chunk claim array — the same shape as the kernel's
//! `PhysRegion` bitmap allocation (`kore-design-doctrine` §1), one byte per
//! slot for the same reason `GenIdAllocator` uses one: a simpler claim path
//! and no summary bitmap to keep coherent.  Each placement owns an ordered
//! list of chunk indices, so a file's bytes need not be contiguous and
//! growth does not have to relocate anything.
//!
//! Chunks are zeroed **when claimed**, and a shrink zeroes the tail of the
//! partial chunk it keeps.  Together those two rules are what make the
//! provider's promise — reserved-but-unwritten capacity reads as zeros —
//! hold even across a shrink-then-grow that reuses the same chunk.
//!
//! # Fault injection
//!
//! [`RamProvider::inject`] arms a one-shot failure for a chosen operation.
//! It is deliberately part of the ordinary API rather than a `cfg(test)`
//! seam: provider conformance requires proof that injected failures leave
//! namespace and object metadata consistent, so producing a provider
//! failure is a capability the test suites need, and RAM has no other way to
//! fail.

use core::cell::RefCell;

use crate::{
    bounds::{MAX_CHUNKS_PER_OBJECT, MAX_PLACEMENTS, RAM_CHUNK_BYTES, RAM_CHUNKS},
    error::ProviderError,
    object::ObjectKind,
    provider::{StorageLimits, StorageProvider},
    slot::{SlotKey, SlotTable},
};

/// A RAM placement's identity.  A stale one fails closed with
/// [`ProviderError::UnknownKey`]; it conveys no authority either way.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RamKey(SlotKey);

/// Which operation a one-shot injected failure applies to, and what it
/// reports.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fault {
    Allocate(ProviderError),
    Read(ProviderError),
    Write(ProviderError),
    Resize(ProviderError),
    Release(ProviderError),
    Sync(ProviderError),
}

/// Which operation is being attempted — matched against an armed [`Fault`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Op {
    Allocate,
    Read,
    Write,
    Resize,
    Release,
    Sync,
}

impl Fault {
    fn matches(self, op: Op) -> Option<ProviderError> {
        match (self, op) {
            (Fault::Allocate(e), Op::Allocate)
            | (Fault::Read(e), Op::Read)
            | (Fault::Write(e), Op::Write)
            | (Fault::Resize(e), Op::Resize)
            | (Fault::Release(e), Op::Release)
            | (Fault::Sync(e), Op::Sync) => Some(e),
            _ => None,
        }
    }
}

/// One object's bytes: an ordered chunk list plus the reserved length.
struct Placement {
    chunks: heapless::Vec<u16, MAX_CHUNKS_PER_OBJECT>,
    len: u64,
}

/// Total bytes the pool can hold.
const TOTAL_BYTES: u64 = (RAM_CHUNKS * RAM_CHUNK_BYTES) as u64;

/// Largest object one placement can hold.
const MAX_OBJECT_BYTES: u64 = (MAX_CHUNKS_PER_OBJECT * RAM_CHUNK_BYTES) as u64;

struct RamState {
    pool: [[u8; RAM_CHUNK_BYTES]; RAM_CHUNKS],
    claimed: [bool; RAM_CHUNKS],
    claimed_count: usize,
    placements: SlotTable<Placement, MAX_PLACEMENTS>,
    fault: Option<Fault>,
}

/// Chunks needed to hold `len` bytes, or `None` on overflow.
fn chunks_for(len: u64) -> Option<usize> {
    let granularity = RAM_CHUNK_BYTES as u64;
    let chunks = len.checked_add(granularity - 1)? / granularity;
    usize::try_from(chunks).ok()
}

impl RamState {
    fn new() -> Self {
        Self {
            pool: [[0u8; RAM_CHUNK_BYTES]; RAM_CHUNKS],
            claimed: [false; RAM_CHUNKS],
            claimed_count: 0,
            placements: SlotTable::new(),
            fault: None,
        }
    }

    /// Consume an armed fault if it names `op`.
    fn take_fault(&mut self, op: Op) -> Option<ProviderError> {
        let error = self.fault?.matches(op)?;
        self.fault = None;
        Some(error)
    }

    /// Claim one free chunk, zeroing it so reserved capacity reads as zeros.
    fn claim(&mut self) -> Option<u16> {
        let index = self.claimed.iter().position(|used| !*used)?;
        self.claimed[index] = true;
        self.claimed_count += 1;
        self.pool[index].fill(0);
        u16::try_from(index).ok()
    }

    fn release_chunk(&mut self, index: u16) {
        let index = index as usize;
        if index < RAM_CHUNKS && self.claimed[index] {
            self.claimed[index] = false;
            self.claimed_count -= 1;
        }
    }

    /// Grow or shrink a placement to exactly `len` bytes.  A refusal leaves
    /// the placement byte-for-byte unchanged.
    fn resize_placement(&mut self, key: RamKey, len: u64) -> Result<(), ProviderError> {
        if len > MAX_OBJECT_BYTES {
            return Err(ProviderError::TooLarge);
        }
        let want = chunks_for(len).ok_or(ProviderError::Range)?;
        if want > MAX_CHUNKS_PER_OBJECT {
            return Err(ProviderError::TooLarge);
        }
        let have = self
            .placements
            .get(key.0)
            .ok_or(ProviderError::UnknownKey)?
            .chunks
            .len();

        if want > have {
            // Claim into a scratch list first: if the pool runs out partway,
            // every chunk we took goes back and the placement is untouched.
            let mut fresh: heapless::Vec<u16, MAX_CHUNKS_PER_OBJECT> = heapless::Vec::new();
            for _ in have..want {
                match self.claim() {
                    Some(index) => {
                        if fresh.push(index).is_err() {
                            self.release_chunk(index);
                            for taken in fresh.iter() {
                                self.release_chunk(*taken);
                            }
                            return Err(ProviderError::TooLarge);
                        }
                    }
                    None => {
                        for taken in fresh.iter() {
                            self.release_chunk(*taken);
                        }
                        return Err(ProviderError::Exhausted);
                    }
                }
            }
            let mut placement = self
                .placements
                .get_mut(key.0)
                .ok_or(ProviderError::UnknownKey)?;
            for index in fresh.iter() {
                // Cannot fail: `want <= MAX_CHUNKS_PER_OBJECT` and the list
                // held `have` before this loop.
                if placement.chunks.push(*index).is_err() {
                    return Err(ProviderError::TooLarge);
                }
            }
        } else if want < have {
            let mut dropped: heapless::Vec<u16, MAX_CHUNKS_PER_OBJECT> = heapless::Vec::new();
            {
                let mut placement = self
                    .placements
                    .get_mut(key.0)
                    .ok_or(ProviderError::UnknownKey)?;
                while placement.chunks.len() > want {
                    if let Some(index) = placement.chunks.pop() {
                        let _ = dropped.push(index);
                    }
                }
            }
            for index in dropped.iter() {
                self.release_chunk(*index);
            }
        }

        // Zero the tail of the retained partial chunk, so a later grow back
        // over this region reads as zeros rather than as stale bytes.
        let tail = (len % RAM_CHUNK_BYTES as u64) as usize;
        if tail != 0 {
            let last = {
                let placement = self
                    .placements
                    .get(key.0)
                    .ok_or(ProviderError::UnknownKey)?;
                placement.chunks.last().copied()
            };
            if let Some(index) = last {
                let index = index as usize;
                if index < RAM_CHUNKS {
                    self.pool[index][tail..].fill(0);
                }
            }
        }

        let mut placement = self
            .placements
            .get_mut(key.0)
            .ok_or(ProviderError::UnknownKey)?;
        placement.len = len;
        Ok(())
    }

    /// Copy out of a placement, returning the bytes transferred.
    fn copy_out(&self, key: RamKey, offset: u64, dst: &mut [u8]) -> Result<usize, ProviderError> {
        let placement = self
            .placements
            .get(key.0)
            .ok_or(ProviderError::UnknownKey)?;
        // Every offset/length computation is checked: a wrapping add here
        // would turn an out-of-range read into an in-range one.
        offset
            .checked_add(dst.len() as u64)
            .ok_or(ProviderError::Range)?;
        if offset > placement.len {
            return Err(ProviderError::Range);
        }
        let available = placement.len - offset;
        let want = (dst.len() as u64).min(available) as usize;

        let mut done = 0usize;
        while done < want {
            let at = offset + done as u64;
            let chunk_slot = (at / RAM_CHUNK_BYTES as u64) as usize;
            let within = (at % RAM_CHUNK_BYTES as u64) as usize;
            let index = *placement
                .chunks
                .get(chunk_slot)
                .ok_or(ProviderError::Range)? as usize;
            if index >= RAM_CHUNKS {
                return Err(ProviderError::Range);
            }
            let take = (RAM_CHUNK_BYTES - within).min(want - done);
            // Slice copy, not a byte loop (kore-design-doctrine §8).
            dst[done..done + take].copy_from_slice(&self.pool[index][within..within + take]);
            done += take;
        }
        Ok(done)
    }

    /// Copy into a placement.  Refuses before touching a byte if the range
    /// is not fully reserved, so a write is all-or-nothing.
    fn copy_in(&mut self, key: RamKey, offset: u64, src: &[u8]) -> Result<(), ProviderError> {
        let end = offset
            .checked_add(src.len() as u64)
            .ok_or(ProviderError::Range)?;
        // Split the two disjoint state fields instead of cloning the whole
        // per-object chunk map on every write. The placement is immutable for
        // this operation; only the already-reserved pool bytes are mutated.
        let (placements, pool) = (&self.placements, &mut self.pool);
        let placement = placements.get(key.0).ok_or(ProviderError::UnknownKey)?;
        if end > placement.len {
            // The engine reserves before writing; reaching here means it did
            // not, and writing part of the range would be worse.
            return Err(ProviderError::Range);
        }

        let mut done = 0usize;
        while done < src.len() {
            let at = offset + done as u64;
            let chunk_slot = (at / RAM_CHUNK_BYTES as u64) as usize;
            let within = (at % RAM_CHUNK_BYTES as u64) as usize;
            let index = *placement
                .chunks
                .get(chunk_slot)
                .ok_or(ProviderError::Range)? as usize;
            if index >= RAM_CHUNKS {
                return Err(ProviderError::Range);
            }
            let take = (RAM_CHUNK_BYTES - within).min(src.len() - done);
            pool[index][within..within + take].copy_from_slice(&src[done..done + take]);
            done += take;
        }
        Ok(())
    }
}

/// The bounded in-memory provider.
pub struct RamProvider {
    state: RefCell<RamState>,
}

impl RamProvider {
    pub fn new() -> Self {
        Self {
            state: RefCell::new(RamState::new()),
        }
    }

    /// Arm a one-shot failure for the next matching operation.  See the
    /// module docs on why this is ordinary API.
    pub fn inject(&self, fault: Fault) {
        self.state.borrow_mut().fault = Some(fault);
    }

    /// Disarm any pending injected failure.
    pub fn clear_fault(&self) {
        self.state.borrow_mut().fault = None;
    }

    /// Chunks currently claimed.  The observable that makes "released
    /// storage exactly once" testable.
    pub fn chunks_in_use(&self) -> usize {
        self.state.borrow().claimed_count
    }

    /// Live placements.
    pub fn placements(&self) -> usize {
        self.state.borrow().placements.len()
    }

    /// Bytes currently claimed, rounded to the allocation granularity.
    pub fn bytes_in_use(&self) -> u64 {
        (self.chunks_in_use() * RAM_CHUNK_BYTES) as u64
    }
}

impl Default for RamProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl StorageProvider for RamProvider {
    type Key = RamKey;
    const PERSISTS_ACROSS_RESTART: bool = false;

    fn limits(&self) -> StorageLimits {
        StorageLimits {
            total_bytes: TOTAL_BYTES,
            max_object_bytes: MAX_OBJECT_BYTES,
            max_transfer: crate::bounds::MAX_TRANSFER,
            granularity: RAM_CHUNK_BYTES,
        }
    }

    async fn allocate(&self, _kind: ObjectKind, len: u64) -> Result<RamKey, ProviderError> {
        let mut state = self.state.borrow_mut();
        if let Some(error) = state.take_fault(Op::Allocate) {
            return Err(error);
        }
        if len > MAX_OBJECT_BYTES {
            return Err(ProviderError::TooLarge);
        }
        let key = state
            .placements
            .insert(Placement {
                chunks: heapless::Vec::new(),
                len: 0,
            })
            .map(RamKey)
            .map_err(|_| ProviderError::Exhausted)?;
        // Reserve the requested capacity; on refusal the empty placement goes
        // back so a failed allocate leaves no slot behind.
        if let Err(error) = state.resize_placement(key, len) {
            state.placements.remove(key.0);
            return Err(error);
        }
        Ok(key)
    }

    async fn read(&self, key: RamKey, offset: u64, dst: &mut [u8]) -> Result<usize, ProviderError> {
        let mut state = self.state.borrow_mut();
        if let Some(error) = state.take_fault(Op::Read) {
            return Err(error);
        }
        if dst.len() > crate::bounds::MAX_TRANSFER {
            return Err(ProviderError::TooLarge);
        }
        state.copy_out(key, offset, dst)
    }

    async fn write(&self, key: RamKey, offset: u64, src: &[u8]) -> Result<(), ProviderError> {
        let mut state = self.state.borrow_mut();
        if let Some(error) = state.take_fault(Op::Write) {
            return Err(error);
        }
        if src.len() > crate::bounds::MAX_TRANSFER {
            return Err(ProviderError::TooLarge);
        }
        state.copy_in(key, offset, src)
    }

    async fn resize(&self, key: RamKey, len: u64) -> Result<(), ProviderError> {
        let mut state = self.state.borrow_mut();
        if let Some(error) = state.take_fault(Op::Resize) {
            return Err(error);
        }
        if state.placements.get(key.0).is_none() {
            return Err(ProviderError::UnknownKey);
        }
        state.resize_placement(key, len)
    }

    async fn release(&self, key: RamKey) -> Result<(), ProviderError> {
        let mut state = self.state.borrow_mut();
        if let Some(error) = state.take_fault(Op::Release) {
            return Err(error);
        }
        let placement = state
            .placements
            .remove(key.0)
            .ok_or(ProviderError::UnknownKey)?;
        for index in placement.chunks.iter() {
            state.release_chunk(*index);
        }
        Ok(())
    }

    async fn sync(&self, key: RamKey) -> Result<(), ProviderError> {
        let mut state = self.state.borrow_mut();
        if let Some(error) = state.take_fault(Op::Sync) {
            return Err(error);
        }
        // RAM's durability contract is "visible in this boot", which a
        // successful mutation already satisfied.  Validating the key keeps
        // sync from reporting success about a placement that is gone.
        if state.placements.get(key.0).is_none() {
            return Err(ProviderError::UnknownKey);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_on;

    #[test]
    fn chunks_for_rounds_up_and_refuses_overflow() {
        assert_eq!(chunks_for(0), Some(0));
        assert_eq!(chunks_for(1), Some(1));
        assert_eq!(chunks_for(RAM_CHUNK_BYTES as u64), Some(1));
        assert_eq!(chunks_for(RAM_CHUNK_BYTES as u64 + 1), Some(2));
        assert_eq!(chunks_for(u64::MAX), None);
    }

    #[test]
    fn a_fresh_provider_holds_nothing() {
        let provider = RamProvider::new();
        assert_eq!(provider.chunks_in_use(), 0);
        assert_eq!(provider.placements(), 0);
        assert_eq!(provider.bytes_in_use(), 0);
    }

    #[test]
    fn allocate_reserves_granular_capacity() {
        let provider = RamProvider::new();
        let key = block_on(provider.allocate(ObjectKind::File, 1)).unwrap();
        // One byte still costs one chunk — the declared granularity.
        assert_eq!(provider.chunks_in_use(), 1);
        assert_eq!(provider.placements(), 1);
        block_on(provider.release(key)).unwrap();
        assert_eq!(provider.chunks_in_use(), 0);
        assert_eq!(provider.placements(), 0);
    }

    #[test]
    fn a_zero_length_object_costs_no_chunks() {
        let provider = RamProvider::new();
        let key = block_on(provider.allocate(ObjectKind::File, 0)).unwrap();
        assert_eq!(provider.chunks_in_use(), 0);
        let mut buf = [0u8; 1];
        assert_eq!(block_on(provider.read(key, 0, &mut buf)), Ok(0));
    }

    #[test]
    fn writes_and_reads_cross_chunk_boundaries() {
        let provider = RamProvider::new();
        let len = RAM_CHUNK_BYTES * 2 + 5;
        let key = block_on(provider.allocate(ObjectKind::File, len as u64)).unwrap();

        let mut src = [0u8; 600];
        let src = &mut src[..len];
        for (i, byte) in src.iter_mut().enumerate() {
            *byte = (i % 251) as u8;
        }
        // Start deliberately mid-chunk so the transfer straddles three chunks.
        assert_eq!(block_on(provider.write(key, 7, &src[..len - 7])), Ok(()));

        let mut back = [0u8; 600];
        let back = &mut back[..len];
        assert_eq!(block_on(provider.read(key, 0, back)), Ok(len));
        assert_eq!(&back[..7], &[0u8; 7], "untouched head stayed zero");
        assert_eq!(&back[7..], &src[..len - 7]);
    }

    #[test]
    fn writing_past_reserved_capacity_is_refused_without_touching_bytes() {
        let provider = RamProvider::new();
        let key = block_on(provider.allocate(ObjectKind::File, 4)).unwrap();
        block_on(provider.write(key, 0, b"abcd")).unwrap();

        assert_eq!(
            block_on(provider.write(key, 2, b"xyz")),
            Err(ProviderError::Range)
        );
        // The refused write must not have written its in-range prefix.
        let mut back = [0u8; 4];
        assert_eq!(block_on(provider.read(key, 0, &mut back)), Ok(4));
        assert_eq!(&back, b"abcd");
    }

    #[test]
    fn offset_length_overflow_is_refused() {
        let provider = RamProvider::new();
        let key = block_on(provider.allocate(ObjectKind::File, 4)).unwrap();
        let mut buf = [0u8; 8];
        assert_eq!(
            block_on(provider.read(key, u64::MAX, &mut buf)),
            Err(ProviderError::Range)
        );
        assert_eq!(
            block_on(provider.write(key, u64::MAX, b"x")),
            Err(ProviderError::Range)
        );
    }

    #[test]
    fn shrink_then_grow_reads_as_zeros_not_stale_bytes() {
        let provider = RamProvider::new();
        let key = block_on(provider.allocate(ObjectKind::File, 300)).unwrap();
        let payload = [0xabu8; 300];
        block_on(provider.write(key, 0, &payload)).unwrap();

        // Shrink inside the first chunk, then grow back over the same bytes.
        block_on(provider.resize(key, 100)).unwrap();
        block_on(provider.resize(key, 300)).unwrap();

        let mut back = [0xffu8; 300];
        assert_eq!(block_on(provider.read(key, 0, &mut back)), Ok(300));
        assert_eq!(&back[..100], &payload[..100], "retained bytes survive");
        assert!(
            back[100..].iter().all(|b| *b == 0),
            "regrown capacity must read as zeros, not as the old bytes"
        );
    }

    #[test]
    fn shrink_returns_chunks_to_the_pool() {
        let provider = RamProvider::new();
        let key =
            block_on(provider.allocate(ObjectKind::File, 4 * RAM_CHUNK_BYTES as u64)).unwrap();
        assert_eq!(provider.chunks_in_use(), 4);
        block_on(provider.resize(key, RAM_CHUNK_BYTES as u64)).unwrap();
        assert_eq!(provider.chunks_in_use(), 1);
    }

    #[test]
    fn an_object_larger_than_the_per_object_ceiling_is_refused() {
        let provider = RamProvider::new();
        assert_eq!(
            block_on(provider.allocate(ObjectKind::File, MAX_OBJECT_BYTES + 1)),
            Err(ProviderError::TooLarge)
        );
        // Nothing was consumed by the refusal.
        assert_eq!(provider.chunks_in_use(), 0);
        assert_eq!(provider.placements(), 0);

        let key = block_on(provider.allocate(ObjectKind::File, 1)).unwrap();
        assert_eq!(
            block_on(provider.resize(key, MAX_OBJECT_BYTES + 1)),
            Err(ProviderError::TooLarge)
        );
        // The refused resize left the original capacity intact.
        assert_eq!(provider.chunks_in_use(), 1);
    }

    #[test]
    fn pool_exhaustion_refuses_and_leaves_the_placement_unchanged() {
        let provider = RamProvider::new();
        // Fill the pool with full-size objects.
        let mut keys = std::vec::Vec::new();
        while let Ok(key) = block_on(provider.allocate(ObjectKind::File, MAX_OBJECT_BYTES)) {
            keys.push(key);
        }
        assert_eq!(provider.chunks_in_use(), RAM_CHUNKS);

        // A further allocation is refused, not satisfied partially.
        assert_eq!(
            block_on(provider.allocate(ObjectKind::File, RAM_CHUNK_BYTES as u64)),
            Err(ProviderError::Exhausted)
        );
        assert_eq!(provider.chunks_in_use(), RAM_CHUNKS);

        // Now set up a grow that can claim *some* chunks and then run out,
        // which is the path that has to roll back.  Free most of one object,
        // then hand nearly all of that room to someone else, leaving a few
        // chunks — fewer than the regrow needs.
        let victim = keys[0];
        block_on(provider.resize(victim, RAM_CHUNK_BYTES as u64)).unwrap();
        let spare = RAM_CHUNKS - provider.chunks_in_use();
        let filler_chunks = spare - 3;
        block_on(provider.allocate(ObjectKind::File, (filler_chunks * RAM_CHUNK_BYTES) as u64))
            .unwrap();
        let before = provider.chunks_in_use();
        assert_eq!(RAM_CHUNKS - before, 3, "exactly three chunks left free");

        // The regrow needs far more than three, so it claims what it can and
        // must then put every one of them back.
        assert_eq!(
            block_on(provider.resize(victim, MAX_OBJECT_BYTES)),
            Err(ProviderError::Exhausted)
        );
        assert_eq!(
            provider.chunks_in_use(),
            before,
            "a refused grow must return every chunk it claimed on the way"
        );
        // And the victim still holds its pre-refusal capacity and bytes.
        let mut buf = [0u8; 8];
        assert_eq!(block_on(provider.read(victim, 0, &mut buf)), Ok(8));
        assert_eq!(
            block_on(provider.read(victim, RAM_CHUNK_BYTES as u64, &mut buf)),
            Ok(0),
            "capacity is unchanged at one chunk"
        );
    }

    #[test]
    fn placement_table_exhaustion_is_typed() {
        let provider = RamProvider::new();
        // Zero-length objects consume a placement slot but no chunks, so the
        // table is what runs out first.
        for _ in 0..MAX_PLACEMENTS {
            block_on(provider.allocate(ObjectKind::File, 0)).unwrap();
        }
        assert_eq!(provider.placements(), MAX_PLACEMENTS);
        assert_eq!(
            block_on(provider.allocate(ObjectKind::File, 0)),
            Err(ProviderError::Exhausted)
        );
    }

    #[test]
    fn transfers_larger_than_the_declared_maximum_are_refused() {
        let provider = RamProvider::new();
        let key = block_on(provider.allocate(ObjectKind::File, 16)).unwrap();
        let oversized = [0u8; crate::bounds::MAX_TRANSFER + 1];
        assert_eq!(
            block_on(provider.write(key, 0, &oversized)),
            Err(ProviderError::TooLarge)
        );
        let mut dst = [0u8; crate::bounds::MAX_TRANSFER + 1];
        assert_eq!(
            block_on(provider.read(key, 0, &mut dst)),
            Err(ProviderError::TooLarge)
        );
    }

    #[test]
    fn placements_are_isolated_from_each_other() {
        let provider = RamProvider::new();
        let a = block_on(provider.allocate(ObjectKind::File, 8)).unwrap();
        let b = block_on(provider.allocate(ObjectKind::File, 8)).unwrap();
        assert_ne!(a, b);
        block_on(provider.write(a, 0, b"aaaaaaaa")).unwrap();
        block_on(provider.write(b, 0, b"bbbbbbbb")).unwrap();

        let mut back = [0u8; 8];
        assert_eq!(block_on(provider.read(a, 0, &mut back)), Ok(8));
        assert_eq!(&back, b"aaaaaaaa");
        assert_eq!(block_on(provider.read(b, 0, &mut back)), Ok(8));
        assert_eq!(&back, b"bbbbbbbb");

        // Releasing one leaves the other's bytes intact.
        block_on(provider.release(a)).unwrap();
        assert_eq!(block_on(provider.read(b, 0, &mut back)), Ok(8));
        assert_eq!(&back, b"bbbbbbbb");
    }

    #[test]
    fn a_reused_placement_slot_rejects_the_stale_key() {
        let provider = RamProvider::new();
        let stale = block_on(provider.allocate(ObjectKind::File, 8)).unwrap();
        block_on(provider.write(stale, 0, b"secrets!")).unwrap();
        block_on(provider.release(stale)).unwrap();

        let fresh = block_on(provider.allocate(ObjectKind::File, 8)).unwrap();
        // The generational key is what stops a freed placement from being
        // readable through the old key after the slot is reused.
        assert_ne!(fresh, stale);
        let mut buf = [0u8; 8];
        assert_eq!(
            block_on(provider.read(stale, 0, &mut buf)),
            Err(ProviderError::UnknownKey)
        );
        // And the reused placement does not expose the old bytes.
        assert_eq!(block_on(provider.read(fresh, 0, &mut buf)), Ok(8));
        assert_eq!(&buf, &[0u8; 8]);
    }

    #[test]
    fn injected_faults_are_one_shot_and_op_specific() {
        let provider = RamProvider::new();
        let key = block_on(provider.allocate(ObjectKind::File, 8)).unwrap();

        provider.inject(Fault::Write(ProviderError::Media));
        // A read must not consume a write's fault.
        let mut buf = [0u8; 8];
        assert_eq!(block_on(provider.read(key, 0, &mut buf)), Ok(8));
        assert_eq!(
            block_on(provider.write(key, 0, b"xxxx")),
            Err(ProviderError::Media)
        );
        // Armed once: the next write succeeds.
        assert_eq!(block_on(provider.write(key, 0, b"xxxx")), Ok(()));
    }

    #[test]
    fn an_injected_write_failure_leaves_bytes_untouched() {
        let provider = RamProvider::new();
        let key = block_on(provider.allocate(ObjectKind::File, 8)).unwrap();
        block_on(provider.write(key, 0, b"original")).unwrap();

        provider.inject(Fault::Write(ProviderError::Media));
        assert_eq!(
            block_on(provider.write(key, 0, b"replaced")),
            Err(ProviderError::Media)
        );
        let mut back = [0u8; 8];
        assert_eq!(block_on(provider.read(key, 0, &mut back)), Ok(8));
        assert_eq!(&back, b"original");
    }

    #[test]
    fn an_injected_release_failure_keeps_the_placement() {
        let provider = RamProvider::new();
        let key = block_on(provider.allocate(ObjectKind::File, 8)).unwrap();
        provider.inject(Fault::Release(ProviderError::Media));
        assert_eq!(block_on(provider.release(key)), Err(ProviderError::Media));
        // The chunks are still held — which is exactly the leak the engine
        // has to account for rather than pretend away.
        assert_eq!(provider.placements(), 1);
        assert_eq!(provider.chunks_in_use(), 1);
        provider.clear_fault();
        assert_eq!(block_on(provider.release(key)), Ok(()));
        assert_eq!(provider.chunks_in_use(), 0);
    }

    #[test]
    fn clear_fault_disarms() {
        let provider = RamProvider::new();
        provider.inject(Fault::Allocate(ProviderError::Media));
        provider.clear_fault();
        assert!(block_on(provider.allocate(ObjectKind::File, 4)).is_ok());
    }

    #[test]
    fn declared_limits_describe_the_real_pool() {
        let provider = RamProvider::new();
        let limits = provider.limits();
        assert_eq!(limits.total_bytes, TOTAL_BYTES);
        assert_eq!(limits.max_object_bytes, MAX_OBJECT_BYTES);
        assert_eq!(limits.granularity, RAM_CHUNK_BYTES);
        // A caller sizing work by the declared maximum must actually be able
        // to allocate it.
        assert!(block_on(provider.allocate(ObjectKind::File, limits.max_object_bytes)).is_ok());
    }
}
