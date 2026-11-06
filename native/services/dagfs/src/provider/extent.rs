//! Bounded free-space indexing for block-provider payload extents.
//!
//! Free runs live in a fixed-capacity max-heap, while every physical segment
//! carries neighbour links so release coalesces with at most two nodes. The
//! heap makes both selection and arbitrary neighbour removal O(log N), and
//! selecting its largest run cannot falsely report exhaustion while any run
//! fits. Unlike a memory allocator, metadata never lives in the managed
//! medium — disk blocks remain pure payload and recovery rebuilds this
//! boot-local index from durable extents.
//!
//! The substrate is `kore_collections`' intrusive pair, not an fs-local slot
//! store: nodes live in a [`SlotArena`] (vacancy freelist + generational
//! handles minted per `Used` incarnation) and the free runs are indexed by a
//! [`PriorityIndex`] through the heap back-pointer each node carries inline.
//! What stays here is exactly the block policy — physical-neighbour
//! coalescing, split placement, the recovery containment contract, and the
//! free-block ledger.

use core::alloc::Allocator;

use kore_collections::{PriorityIndex, PriorityLink, SlotArena, SlotId, UNINDEXED};

use crate::{ServiceAlloc, bounds::MAX_BLOCK_EXTENT_NODES, error::ProviderError};

const NONE: u32 = u32::MAX;

/// Handle to one allocated extent — a [`SlotId`] minted by the node arena
/// for the slot's `Used` incarnation, so a handle that outlives its extent
/// is *detectably* stale: resolving it refuses with `Integrity` instead of
/// silently aliasing whatever run later recycles the slot.
pub(super) type ExtentId = SlotId;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum NodeState {
    #[default]
    Vacant,
    Free,
    Used,
}

#[derive(Clone, Copy, Debug)]
struct Node {
    start: u64,
    blocks: u64,
    prev_phys: u32,
    next_phys: u32,
    heap_index: u32,
    state: NodeState,
}

impl Default for Node {
    /// The claimed-but-unwired payload: no physical neighbours, outside the
    /// free index. The arena resets every slot to this on claim and release.
    fn default() -> Self {
        Self {
            start: 0,
            blocks: 0,
            prev_phys: NONE,
            next_phys: NONE,
            heap_index: UNINDEXED,
            state: NodeState::Vacant,
        }
    }
}

impl PriorityLink for Node {
    fn heap_index(&self) -> u32 {
        self.heap_index
    }

    fn set_heap_index(&mut self, position: u32) {
        self.heap_index = position;
    }

    /// Largest run first; equal sizes prefer the lower placement so
    /// selection is deterministic.
    fn outranks(&self, other: &Self) -> bool {
        self.blocks > other.blocks || (self.blocks == other.blocks && self.start < other.start)
    }
}

/// One volume-local free-space index. Its owner already has exclusive access
/// through `BlockProvider`'s task-local state, so it needs no second lock.
pub(super) struct ExtentAllocator {
    nodes: SlotArena<Node, ServiceAlloc>,
    free: PriorityIndex<ServiceAlloc>,
    last_phys: u32,
    free_blocks: u64,
    arena_start: u64,
    arena_end: u64,
}

impl ExtentAllocator {
    pub(super) fn new(start: u64, end: u64) -> Result<Self, ProviderError> {
        let nodes = SlotArena::try_new_in(MAX_BLOCK_EXTENT_NODES, crate::service_alloc())
            .map_err(|_| ProviderError::Backpressure)?;
        let free = PriorityIndex::try_new_in(MAX_BLOCK_EXTENT_NODES, crate::service_alloc())
            .map_err(|_| ProviderError::Backpressure)?;
        let mut allocator = Self {
            nodes,
            free,
            last_phys: NONE,
            free_blocks: 0,
            arena_start: 0,
            arena_end: 0,
        };
        allocator.reset(start, end)?;
        Ok(allocator)
    }

    pub(super) fn reset(&mut self, start: u64, end: u64) -> Result<(), ProviderError> {
        if start >= end {
            return Err(ProviderError::Integrity);
        }
        self.free.clear();
        self.last_phys = NONE;
        self.arena_start = start;
        self.arena_end = end;
        self.free_blocks = end - start;

        // Every slot returns to the vacancy freelist in ascending order.
        // `reset` is reachable only through `configure_arena`'s
        // empty-provider guard (or `new`), so no `ExtentId` survives it —
        // and the arena preserves slot generations across the recycle
        // regardless, so even a handle that wrongly did survive would stay
        // detectably stale instead of aliasing a post-reset extent.
        self.nodes.release_all();
        let whole = self.take_slot()?;
        self.nodes[whole].start = start;
        self.nodes[whole].blocks = end - start;
        self.nodes[whole].state = NodeState::Free;
        self.last_phys = whole as u32;
        self.insert_free(whole)?;
        Ok(())
    }

    #[inline]
    pub(super) const fn free_blocks(&self) -> u64 {
        self.free_blocks
    }

    pub(super) fn range(&self, id: ExtentId) -> Result<(u64, u32), ProviderError> {
        let node = self.node(id)?;
        if node.state != NodeState::Used {
            return Err(ProviderError::Integrity);
        }
        let blocks = u32::try_from(node.blocks).map_err(|_| ProviderError::Integrity)?;
        Ok((node.start, blocks))
    }

    /// Mark `index` as a fresh `Used` incarnation and mint its handle. The
    /// arena's generation bump is what retires every previously minted id
    /// for this slot, whatever recycling path brought the slot here.
    fn mint_used(&mut self, index: usize) -> ExtentId {
        self.nodes[index].state = NodeState::Used;
        self.nodes[index].heap_index = UNINDEXED;
        self.nodes.mint(index)
    }

    /// Allocate from the largest free run, returning the handle and the
    /// run's first block. Heap selection, split, and index maintenance are
    /// O(log N), independent of physical placement order. The returned start
    /// always covers exactly `blocks` — the split carves `blocks` from the
    /// front of the chosen run.
    pub(super) fn allocate(&mut self, blocks: u32) -> Result<(ExtentId, u64), ProviderError> {
        if blocks == 0 {
            return Err(ProviderError::Integrity);
        }
        let need = u64::from(blocks);
        let raw = self.free.peek().ok_or(ProviderError::Exhausted)?;
        let index = self.index(raw)?;
        let original = self.nodes[index];
        if original.state != NodeState::Free {
            return Err(ProviderError::Integrity);
        }
        if original.blocks < need {
            return Err(ProviderError::Exhausted);
        }

        let remainder = if original.blocks > need {
            Some(self.take_slot()?)
        } else {
            None
        };
        self.remove_free(index)?;
        self.nodes[index].blocks = need;
        let id = self.mint_used(index);

        if let Some(tail) = remainder {
            let tail_raw = tail as u32;
            self.nodes[tail] = Node {
                start: original
                    .start
                    .checked_add(need)
                    .ok_or(ProviderError::Integrity)?,
                blocks: original.blocks - need,
                prev_phys: raw,
                next_phys: original.next_phys,
                heap_index: UNINDEXED,
                state: NodeState::Free,
            };
            self.nodes[index].next_phys = tail_raw;
            if original.next_phys == NONE {
                self.last_phys = tail_raw;
            } else {
                let next = self.index(original.next_phys)?;
                self.nodes[next].prev_phys = tail_raw;
            }
            self.insert_free(tail)?;
        }
        self.free_blocks = self
            .free_blocks
            .checked_sub(need)
            .ok_or(ProviderError::Integrity)?;
        Ok((id, original.start))
    }

    /// Reserve an exact recovered extent. Callers pass records in ascending
    /// LBA order, so the containing segment is always the physical tail —
    /// that contract is load-bearing (containment in the tail is what makes
    /// one containment check equivalent to a full overlap scan), which is
    /// why this function resolves the tail itself rather than taking an
    /// arbitrary segment parameter. Only the O(log N) free-heap update
    /// remains after the one recovery sort.
    pub(super) fn reserve_sorted(
        &mut self,
        start: u64,
        blocks: u32,
    ) -> Result<ExtentId, ProviderError> {
        if blocks == 0 {
            return Err(ProviderError::Integrity);
        }
        let tail = self.index(self.last_phys)?;
        let free = self.nodes[tail];
        let need = u64::from(blocks);
        let end = start.checked_add(need).ok_or(ProviderError::Integrity)?;
        let free_end = free
            .start
            .checked_add(free.blocks)
            .ok_or(ProviderError::Integrity)?;
        if free.state != NodeState::Free || start < free.start || end > free_end {
            return Err(ProviderError::Integrity);
        }

        let prefix = start - free.start;
        let suffix = free_end - end;
        let used_slot = if prefix == 0 { tail } else { self.take_slot()? };
        let suffix_slot = if suffix == 0 {
            None
        } else {
            match self.take_slot() {
                Ok(slot) => Some(slot),
                Err(error) => {
                    if prefix != 0 {
                        self.release_slot(used_slot);
                    }
                    return Err(error);
                }
            }
        };

        self.remove_free(tail)?;
        if prefix == 0 {
            // The tail node itself becomes the reservation; its physical
            // links already hold, so only the geometry changes here and
            // `mint_used` below flips the state.
            self.nodes[tail].blocks = need;
        } else {
            self.nodes[tail].blocks = prefix;
            self.nodes[tail].next_phys = used_slot as u32;
            self.insert_free(tail)?;
            self.nodes[used_slot] = Node {
                start,
                blocks: need,
                prev_phys: tail as u32,
                next_phys: NONE,
                heap_index: UNINDEXED,
                state: NodeState::Used,
            };
        }
        let id = self.mint_used(used_slot);

        let used_raw = used_slot as u32;
        if let Some(slot) = suffix_slot {
            let raw = slot as u32;
            self.nodes[used_slot].next_phys = raw;
            self.nodes[slot] = Node {
                start: end,
                blocks: suffix,
                prev_phys: used_raw,
                next_phys: NONE,
                heap_index: UNINDEXED,
                state: NodeState::Free,
            };
            self.insert_free(slot)?;
            self.last_phys = raw;
        } else {
            self.nodes[used_slot].next_phys = NONE;
            self.last_phys = used_raw;
        }
        self.free_blocks = self
            .free_blocks
            .checked_sub(need)
            .ok_or(ProviderError::Integrity)?;
        Ok(id)
    }

    /// Return one allocated extent, merging with its free physical neighbours.
    pub(super) fn release(&mut self, id: ExtentId) -> Result<(), ProviderError> {
        let original = self.slot_of(id)?;
        if self.nodes[original].state != NodeState::Used {
            return Err(ProviderError::Integrity);
        }
        let released = self.nodes[original].blocks;
        self.nodes[original].state = NodeState::Free;
        let mut merged = original;

        let prev_raw = self.nodes[merged].prev_phys;
        if prev_raw != NONE {
            let prev = self.index(prev_raw)?;
            if self.nodes[prev].state == NodeState::Free {
                self.remove_free(prev)?;
                self.absorb(prev, merged)?;
                merged = prev;
            }
        }

        let next_raw = self.nodes[merged].next_phys;
        if next_raw != NONE {
            let next = self.index(next_raw)?;
            if self.nodes[next].state == NodeState::Free {
                self.remove_free(next)?;
                self.absorb(merged, next)?;
            }
        }

        self.insert_free(merged)?;
        self.free_blocks = self
            .free_blocks
            .checked_add(released)
            .filter(|free| *free <= self.arena_end - self.arena_start)
            .ok_or(ProviderError::Integrity)?;
        Ok(())
    }

    /// Merge `victim`'s span onto the tail of `survivor` and retire the
    /// victim's slot — the one link surgery both coalesce directions share.
    /// The caller has already removed whichever of the two was indexed free,
    /// and re-indexes the survivor afterwards.
    fn absorb(&mut self, survivor: usize, victim: usize) -> Result<(), ProviderError> {
        self.nodes[survivor].blocks = self.nodes[survivor]
            .blocks
            .checked_add(self.nodes[victim].blocks)
            .ok_or(ProviderError::Integrity)?;
        let after_raw = self.nodes[victim].next_phys;
        self.nodes[survivor].next_phys = after_raw;
        if after_raw == NONE {
            self.last_phys = survivor as u32;
        } else {
            let after = self.index(after_raw)?;
            self.nodes[after].prev_phys = survivor as u32;
        }
        self.release_slot(victim);
        Ok(())
    }

    fn take_slot(&mut self) -> Result<usize, ProviderError> {
        self.nodes.claim().ok_or(ProviderError::PlacementsFull)
    }

    fn release_slot(&mut self, index: usize) {
        self.nodes.release(index);
    }

    /// Resolve a caller-held handle through the arena: bounds, occupancy,
    /// **and** generation must match, so a handle from a superseded
    /// incarnation refuses instead of aliasing the slot's current occupant.
    /// Internal physical/heap links use raw indices via
    /// [`index`](Self::index) — they are maintained under this module's own
    /// invariants and never outlive an incarnation.
    fn slot_of(&self, id: ExtentId) -> Result<usize, ProviderError> {
        self.nodes.resolve(id).ok_or(ProviderError::Integrity)
    }

    fn node(&self, id: ExtentId) -> Result<&Node, ProviderError> {
        let index = self.slot_of(id)?;
        Ok(&self.nodes[index])
    }

    fn index(&self, raw: u32) -> Result<usize, ProviderError> {
        let index = raw as usize;
        if raw == NONE || index >= self.nodes.capacity() {
            Err(ProviderError::Integrity)
        } else {
            Ok(index)
        }
    }

    fn insert_free(&mut self, index: usize) -> Result<(), ProviderError> {
        debug_assert!(self.nodes[index].blocks != 0);
        self.nodes[index].state = NodeState::Free;
        self.free
            .insert(self.nodes.values_mut(), index)
            .map_err(|_| ProviderError::Integrity)
    }

    fn remove_free(&mut self, index: usize) -> Result<(), ProviderError> {
        if self.nodes[index].state != NodeState::Free || self.nodes[index].blocks == 0 {
            return Err(ProviderError::Integrity);
        }
        self.free
            .remove(self.nodes.values_mut(), index)
            .map_err(|_| ProviderError::Integrity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_allocations_do_not_rescan_the_prefix() {
        let mut allocator = ExtentAllocator::new(100, 10_000).unwrap();
        let mut ids = heapless::Vec::<ExtentId, 128>::new();
        for expected in 100..228 {
            let (id, lba) = allocator.allocate(1).unwrap();
            assert_eq!(lba, expected);
            assert_eq!(allocator.range(id).unwrap(), (expected, 1));
            ids.push(id).unwrap();
        }
        assert_eq!(allocator.free_blocks(), 9_772);
    }

    #[test]
    fn release_coalesces_both_physical_neighbours() {
        let mut allocator = ExtentAllocator::new(72, 100).unwrap();
        let (a, _) = allocator.allocate(4).unwrap();
        let (b, _) = allocator.allocate(5).unwrap();
        let (c, _) = allocator.allocate(6).unwrap();
        allocator.release(a).unwrap();
        allocator.release(c).unwrap();
        allocator.release(b).unwrap();
        assert_eq!(allocator.free_blocks(), 28);
        let (whole, lba) = allocator.allocate(28).unwrap();
        assert_eq!(lba, 72);
        assert_eq!(allocator.range(whole).unwrap(), (72, 28));
    }

    #[test]
    fn sorted_recovery_builds_holes_then_allocates_from_a_free_run() {
        let mut allocator = ExtentAllocator::new(72, 100).unwrap();
        let a = allocator.reserve_sorted(75, 4).unwrap();
        let b = allocator.reserve_sorted(84, 3).unwrap();
        assert_eq!(allocator.range(a).unwrap(), (75, 4));
        assert_eq!(allocator.range(b).unwrap(), (84, 3));
        let (allocation, lba) = allocator.allocate(3).unwrap();
        assert_eq!(lba, 87);
        assert_eq!(allocator.range(allocation).unwrap(), (87, 3));
    }

    #[test]
    fn recovered_extents_must_be_sorted_and_disjoint() {
        let mut allocator = ExtentAllocator::new(72, 100).unwrap();
        allocator.reserve_sorted(80, 4).unwrap();
        assert_eq!(
            allocator.reserve_sorted(78, 1),
            Err(ProviderError::Integrity)
        );
    }

    #[test]
    fn mixed_allocation_and_release_matches_a_block_ownership_model() {
        const START: u64 = 72;
        const BLOCKS: usize = 257;
        let mut allocator = ExtentAllocator::new(START, START + BLOCKS as u64).unwrap();
        let mut used = [false; BLOCKS];
        let mut active = heapless::Vec::<(ExtentId, usize, usize), 128>::new();
        let mut random = 0x815_u64;

        for _ in 0..5_000 {
            random = random
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            if !active.is_empty() && (random & 3) == 0 {
                let index = (random as usize >> 8) % active.len();
                let (id, start, blocks) = active.swap_remove(index);
                assert!(used[start..start + blocks].iter().all(|block| *block));
                used[start..start + blocks].fill(false);
                allocator.release(id).unwrap();
            } else if !active.is_full() {
                let blocks = ((random >> 16) as usize % 16) + 1;
                match allocator.allocate(blocks as u32) {
                    Ok((id, returned)) => {
                        let (lba, actual) = allocator.range(id).unwrap();
                        assert_eq!(lba, returned);
                        assert_eq!(actual as usize, blocks);
                        let start = usize::try_from(lba - START).unwrap();
                        assert!(start + blocks <= BLOCKS);
                        assert!(used[start..start + blocks].iter().all(|block| !*block));
                        used[start..start + blocks].fill(true);
                        active.push((id, start, blocks)).unwrap();
                    }
                    Err(ProviderError::Exhausted) => assert!(
                        !used
                            .windows(blocks)
                            .any(|window| window.iter().all(|block| !*block))
                    ),
                    Err(error) => panic!("unexpected allocation error: {error:?}"),
                }
            }
            assert_eq!(
                allocator.free_blocks(),
                used.iter().filter(|block| !**block).count() as u64
            );
        }

        for (id, _, _) in active {
            allocator.release(id).unwrap();
        }
        let (whole, _) = allocator.allocate(BLOCKS as u32).unwrap();
        assert_eq!(allocator.range(whole).unwrap(), (START, BLOCKS as u32));
    }

    /// A handle that outlives its extent must refuse, even after its slot is
    /// recycled by a new allocation — the silent-alias footgun the
    /// generation tag exists to close.
    #[test]
    fn stale_extent_ids_are_detectably_stale() {
        let mut allocator = ExtentAllocator::new(72, 100).unwrap();
        let (stale, lba) = allocator.allocate(4).unwrap();
        allocator.release(stale).unwrap();
        // Double release fails closed on the dead incarnation.
        assert_eq!(allocator.release(stale), Err(ProviderError::Integrity));
        // The recycled slot serves a fresh incarnation at the same LBA...
        let (fresh, again) = allocator.allocate(4).unwrap();
        assert_eq!(again, lba);
        // ...which the stale handle can neither read nor free.
        assert_eq!(allocator.range(stale), Err(ProviderError::Integrity));
        assert_eq!(allocator.release(stale), Err(ProviderError::Integrity));
        assert_eq!(allocator.range(fresh).unwrap(), (lba, 4));
    }
}
