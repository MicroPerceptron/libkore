//! Removable priority index over intrusively linked nodes.
//!
//! [`PriorityIndex`] is a fixed-capacity binary max-heap of node *indices*
//! whose defining property is **O(log N) removal of an arbitrary element**,
//! not just the top: each indexed node carries its own heap position inline
//! (the [`PriorityLink`] contract), so "where is this node in the heap?" is a
//! field read instead of a scan.  That back-pointer is what a plain
//! `BinaryHeap` cannot offer and what every allocator-shaped consumer needs —
//! a free-space index must un-index an arbitrary run when it coalesces with a
//! released neighbour, a deadline queue must un-index a cancelled timer.
//!
//! ## The seam
//!
//! The engine is deliberately **storage-blind**: every operation takes the
//! node slice (`&mut [T]`) alongside `&mut self`, so the nodes may live in a
//! [`SlotArena`](crate::SlotArena) (the intended pairing — see its docs), a
//! fixed array, or any other stable indexable storage.  The contract trait
//! carries the two per-node concerns the engine cannot know:
//!
//! - **the back-pointer** ([`PriorityLink::heap_index`] /
//!   [`PriorityLink::set_heap_index`]) — where in the node the position is
//!   stored, [`UNINDEXED`] when the node is outside the index;
//! - **the ranking** ([`PriorityLink::outranks`]) — a strict max-heap
//!   priority over whatever fields the node considers load-bearing.
//!
//! New node shape = a new `impl PriorityLink`, never a change here.
//!
//! ## Posture
//!
//! Single-owner (`&mut self`), plain tier: it imposes no locking policy —
//! wrap it in the caller's own exclusivity if shared.  The heap storage is
//! reserved once at construction ([`PriorityIndex::try_new_in`], named arena,
//! never a global) and never grows; exhaustion and any disagreement between a
//! back-pointer and the heap's own record are **typed refusals**
//! ([`PriorityError`]), never a panic or a silent re-site.

use core::alloc::{AllocError, Allocator};

use alloc::vec::Vec;

/// Back-pointer value of a node that is not currently in any index.
///
/// Node indices therefore range over `0..UNINDEXED`; construction refuses a
/// capacity that could issue the sentinel as a real index.
pub const UNINDEXED: u32 = u32::MAX;

/// The intrusive contract an indexed node implements: where its heap
/// position lives, and how it ranks against a peer.
///
/// The back-pointer is maintained exclusively by [`PriorityIndex`] while the
/// node is indexed; the node owns it (initialized to [`UNINDEXED`]) while it
/// is not.  `outranks` must be a strict ordering ("ranks strictly higher"):
/// ties broken inside `outranks` are part of the ranking, and equal elements
/// must answer `false` in both directions or sifting will not terminate
/// deterministically.
pub trait PriorityLink {
    /// This node's current position in the index, or [`UNINDEXED`].
    fn heap_index(&self) -> u32;

    /// Store this node's position; called only by [`PriorityIndex`].
    fn set_heap_index(&mut self, position: u32);

    /// `true` iff `self` ranks strictly above `other` (max-heap order).
    fn outranks(&self, other: &Self) -> bool;
}

/// Why the index refused an operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PriorityError {
    /// The node index is outside the given slice, the node's back-pointer
    /// disagrees with the heap's own record, or an insert named a node that
    /// is already indexed.  The structure refuses rather than re-site an
    /// element it does not verifiably hold.
    Inconsistent,
    /// The index is at its declared capacity.
    Full,
}

/// Fixed-capacity removable binary max-heap of node indices.
///
/// See the module docs for the design; the mechanics are the classic
/// indexed-heap triple — push-back + sift-up on insert, swap-with-last +
/// re-sift on arbitrary removal, back-pointers rewritten on every swap — with
/// every node access bounds-checked into the caller's slice.
pub struct PriorityIndex<A: Allocator + ?Sized + 'static> {
    heap: Vec<u32, &'static A>,
    limit: usize,
}

impl<A: Allocator + ?Sized + 'static> PriorityIndex<A> {
    /// An empty index holding at most `capacity` nodes, backed by `alloc`.
    ///
    /// Fallible by construction (`try_reserve_exact`, never an aborting
    /// grow), and the reservation is the *only* allocation the index ever
    /// performs.  Refuses a capacity that could not distinguish a real
    /// position from [`UNINDEXED`].
    pub fn try_new_in(capacity: usize, alloc: &'static A) -> Result<Self, AllocError> {
        if capacity >= UNINDEXED as usize {
            return Err(AllocError);
        }
        let mut heap = Vec::new_in(alloc);
        heap.try_reserve_exact(capacity).map_err(|_| AllocError)?;
        Ok(Self {
            heap,
            limit: capacity,
        })
    }

    /// The highest-ranked node index, without removing it.
    #[inline]
    pub fn peek(&self) -> Option<u32> {
        self.heap.first().copied()
    }

    /// Number of nodes currently indexed.
    #[inline]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// The declared capacity — the bound `try_new_in` reserved, not whatever
    /// the allocator rounded the reservation up to.
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.limit
    }

    /// Drop every entry without touching node back-pointers.  For callers
    /// that are about to reinitialize the node storage wholesale (a
    /// [`SlotArena::release_all`](crate::SlotArena::release_all) pairing);
    /// anyone keeping the nodes must [`remove`](Self::remove) them instead.
    #[inline]
    pub fn clear(&mut self) {
        self.heap.clear();
    }

    /// Index `nodes[index]`, writing its back-pointer and restoring heap
    /// order in O(log N).  The node must currently be [`UNINDEXED`].
    pub fn insert<T: PriorityLink>(
        &mut self,
        nodes: &mut [T],
        index: usize,
    ) -> Result<(), PriorityError> {
        if self.heap.len() == self.limit {
            return Err(PriorityError::Full);
        }
        let word = Self::index_word(index)?;
        let position = self.heap.len();
        let node = nodes.get_mut(index).ok_or(PriorityError::Inconsistent)?;
        if node.heap_index() != UNINDEXED {
            return Err(PriorityError::Inconsistent);
        }
        node.set_heap_index(position as u32);
        self.heap.push(word);
        self.sift_up(nodes, position)
    }

    /// Un-index `nodes[index]` — *any* indexed node, not just the top — in
    /// O(log N), resetting its back-pointer to [`UNINDEXED`].  The node's
    /// back-pointer and the heap's record must agree exactly.
    pub fn remove<T: PriorityLink>(
        &mut self,
        nodes: &mut [T],
        index: usize,
    ) -> Result<(), PriorityError> {
        let word = Self::index_word(index)?;
        let position = nodes.get(index).ok_or(PriorityError::Inconsistent)?.heap_index() as usize;
        if position >= self.heap.len() || self.heap[position] != word {
            return Err(PriorityError::Inconsistent);
        }
        let last = self.heap.pop().ok_or(PriorityError::Inconsistent)?;
        nodes[index].set_heap_index(UNINDEXED);
        if position < self.heap.len() {
            self.heap[position] = last;
            nodes
                .get_mut(last as usize)
                .ok_or(PriorityError::Inconsistent)?
                .set_heap_index(position as u32);
            if position > 0 && self.outranks_at(nodes, position, (position - 1) / 2)? {
                self.sift_up(nodes, position)
            } else {
                self.sift_down(nodes, position)
            }
        } else {
            Ok(())
        }
    }

    /// A node index as the heap's stored `u32` word.  Refuses an index the
    /// documented `0..UNINDEXED` range cannot represent — a wider slice
    /// index must fail closed here, never truncate onto some other node.
    fn index_word(index: usize) -> Result<u32, PriorityError> {
        u32::try_from(index)
            .ok()
            .filter(|&word| word != UNINDEXED)
            .ok_or(PriorityError::Inconsistent)
    }

    /// Whether the heap entry at `position` outranks the one at `other` —
    /// both are *heap* positions, resolved through the caller's slice.
    fn outranks_at<T: PriorityLink>(
        &self,
        nodes: &[T],
        position: usize,
        other: usize,
    ) -> Result<bool, PriorityError> {
        let left = nodes
            .get(self.heap[position] as usize)
            .ok_or(PriorityError::Inconsistent)?;
        let right = nodes
            .get(self.heap[other] as usize)
            .ok_or(PriorityError::Inconsistent)?;
        Ok(left.outranks(right))
    }

    fn swap<T: PriorityLink>(
        &mut self,
        nodes: &mut [T],
        left: usize,
        right: usize,
    ) -> Result<(), PriorityError> {
        self.heap.swap(left, right);
        let left_node = self.heap[left] as usize;
        let right_node = self.heap[right] as usize;
        nodes
            .get_mut(left_node)
            .ok_or(PriorityError::Inconsistent)?
            .set_heap_index(left as u32);
        nodes
            .get_mut(right_node)
            .ok_or(PriorityError::Inconsistent)?
            .set_heap_index(right as u32);
        Ok(())
    }

    fn sift_up<T: PriorityLink>(
        &mut self,
        nodes: &mut [T],
        mut position: usize,
    ) -> Result<(), PriorityError> {
        while position > 0 {
            let parent = (position - 1) / 2;
            if !self.outranks_at(nodes, position, parent)? {
                break;
            }
            self.swap(nodes, position, parent)?;
            position = parent;
        }
        Ok(())
    }

    fn sift_down<T: PriorityLink>(
        &mut self,
        nodes: &mut [T],
        mut position: usize,
    ) -> Result<(), PriorityError> {
        loop {
            let left = position * 2 + 1;
            if left >= self.heap.len() {
                return Ok(());
            }
            let right = left + 1;
            let mut larger = left;
            if right < self.heap.len() && self.outranks_at(nodes, right, left)? {
                larger = right;
            }
            if !self.outranks_at(nodes, larger, position)? {
                return Ok(());
            }
            self.swap(nodes, position, larger)?;
            position = larger;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestArena;

    static ARENA: TestArena<{ 1 << 16 }> = TestArena::new();

    /// A minimal ranked node: weight descending, id ascending on ties — the
    /// same shape a free-extent node uses (size, then placement).
    #[derive(Clone, Copy)]
    struct Ranked {
        weight: u64,
        id: u64,
        heap_index: u32,
    }

    impl Ranked {
        const fn new(weight: u64, id: u64) -> Self {
            Self {
                weight,
                id,
                heap_index: UNINDEXED,
            }
        }
    }

    impl PriorityLink for Ranked {
        fn heap_index(&self) -> u32 {
            self.heap_index
        }

        fn set_heap_index(&mut self, position: u32) {
            self.heap_index = position;
        }

        fn outranks(&self, other: &Self) -> bool {
            self.weight > other.weight || (self.weight == other.weight && self.id < other.id)
        }
    }

    fn drain_top(index: &mut PriorityIndex<TestArena<{ 1 << 16 }>>, nodes: &mut [Ranked]) -> std::vec::Vec<u64> {
        let mut order = std::vec::Vec::new();
        while let Some(top) = index.peek() {
            order.push(nodes[top as usize].weight);
            index.remove(nodes, top as usize).unwrap();
        }
        order
    }

    #[test]
    fn peek_tracks_the_maximum_with_deterministic_ties() {
        let mut index = PriorityIndex::try_new_in(8, &ARENA).unwrap();
        let mut nodes = [
            Ranked::new(3, 0),
            Ranked::new(9, 1),
            Ranked::new(9, 2),
            Ranked::new(1, 3),
        ];
        for slot in 0..nodes.len() {
            index.insert(&mut nodes, slot).unwrap();
        }
        // Equal weights rank by ascending id: node 1 beats node 2.
        assert_eq!(index.peek(), Some(1));
        assert_eq!(drain_top(&mut index, &mut nodes), [9, 9, 3, 1]);
        assert!(nodes.iter().all(|node| node.heap_index == UNINDEXED));
    }

    #[test]
    fn removing_an_arbitrary_middle_element_keeps_order_and_backpointers() {
        let mut index = PriorityIndex::try_new_in(16, &ARENA).unwrap();
        let mut nodes: [Ranked; 9] = core::array::from_fn(|slot| {
            // Scrambled weights so the victim sits mid-heap, not at an edge.
            Ranked::new([5, 12, 3, 40, 8, 21, 2, 33, 17][slot], slot as u64)
        });
        for slot in 0..nodes.len() {
            index.insert(&mut nodes, slot).unwrap();
        }
        index.remove(&mut nodes, 4).unwrap(); // weight 8, interior
        assert_eq!(nodes[4].heap_index, UNINDEXED);
        // Every survivor's back-pointer still round-trips through the heap.
        for (slot, node) in nodes.iter().enumerate() {
            if slot != 4 {
                assert_ne!(node.heap_index, UNINDEXED);
            }
        }
        assert_eq!(drain_top(&mut index, &mut nodes), [40, 33, 21, 17, 12, 5, 3, 2]);
    }

    #[test]
    fn refusals_are_typed_and_leave_the_index_intact() {
        let mut index = PriorityIndex::try_new_in(1, &ARENA).unwrap();
        let mut nodes = [Ranked::new(1, 0), Ranked::new(2, 1)];
        index.insert(&mut nodes, 0).unwrap();
        // Capacity: the declared bound refuses, it does not grow.
        assert_eq!(index.insert(&mut nodes, 1), Err(PriorityError::Full));
        // Double insert of an already-indexed node, with room to spare so the
        // capacity refusal cannot mask it.
        let mut index = PriorityIndex::try_new_in(4, &ARENA).unwrap();
        nodes[0].heap_index = UNINDEXED;
        index.insert(&mut nodes, 0).unwrap();
        assert_eq!(index.insert(&mut nodes, 0), Err(PriorityError::Inconsistent));
        // Removing a node that is not indexed.
        assert_eq!(index.remove(&mut nodes, 1), Err(PriorityError::Inconsistent));
        // Out-of-slice node index.
        assert_eq!(index.remove(&mut nodes, 99), Err(PriorityError::Inconsistent));
        // A back-pointer that lies about its position refuses instead of
        // unhooking whichever node really sits there.
        nodes[1].heap_index = 0;
        assert_eq!(index.remove(&mut nodes, 1), Err(PriorityError::Inconsistent));
        nodes[1].heap_index = UNINDEXED;
        assert_eq!(index.peek(), Some(0));
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn capacity_that_collides_with_the_sentinel_is_refused() {
        assert!(PriorityIndex::<TestArena<{ 1 << 16 }>>::try_new_in(UNINDEXED as usize, &ARENA).is_err());
    }
}
