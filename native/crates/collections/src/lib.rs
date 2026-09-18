//! `kore-collections` — the **lock-free and arena-parameterized
//! collections** the kernel's concurrency rests on, as a standalone crate.
//!
//! ## Why these live in a crate
//!
//! `res/KRAPH.md`'s standing rule is that no design may reach for a global
//! lock; that rule is only livable if the lock-free primitives are already
//! at hand when a subsystem is written.  Keeping them in `src/collections`
//! made them the *kernel's* — unavailable to a userland service, a driver,
//! or a sibling repo's harness without copying, which is how a tree ends up
//! with two slab allocators.  They are pure data structures over `core`
//! atomics with no kernel concept in them, so nothing was holding them
//! there.
//!
//! **Dependency-free and `no_std`**: `core` is the entire dependency set —
//! not even `bitflags`. Anything that can compile `core` can use this.
//!
//! ## Allocation discipline (`res/ALLOC.md`)
//!
//! There is **no global allocator here and no `alloc::alloc::Global`
//! anywhere**.  Every owning structure is parameterized on an
//! `A: Allocator` the caller names — the kernel passes a
//! tightly-scoped arena, tests pass a test-local one.  The fixed-capacity
//! structures ([`BitSet`], [`AtomicBitSet`]) allocate nothing at all: their
//! bound is a const generic, so they live inline wherever they are declared.
//!
//! ## What is here, and what deliberately is not
//!
//! Lock-free (`&self`, atomics, no lock anywhere):
//! - [`AtomicBitSet`] — fixed-capacity atomic bitmap, CAS claim/release.
//! - [`AtomicBitVec`] — the arena-backed, runtime-sized sibling; the shape
//!   `PhysRegion`'s frame allocator claims against.
//! - [`GenIdAllocator`] — generational ids: a slot index paired with a
//!   generation, so a stale handle is *detectably* stale rather than a
//!   silent alias onto a recycled slot.
//! - [`AtomicSlotMap`] — fixed-capacity stable storage with per-slot
//!   generations, guarded reads, and affine mutation claims that can span an
//!   async operation without excluding unrelated slots.
//!
//! - [`PageStore`] — sparse publication and guarded reclamation of caller-defined
//!   pages, with permanent retirement and transposed pin-counter shards.
//!
//! Plain (single-owner, `&mut self` — wrap in the caller's own lock if
//! shared; the point is that they impose no locking policy):
//! - [`BitSet`] / [`BitVec`] — the non-atomic siblings.
//! - [`BoundedVec`] — a fixed-capacity vector whose storage is in the arena
//!   rather than inline: the bound stays a refusal (`push` hands the value
//!   back, storage never grows) without the worst case landing in a stack
//!   frame.  The shape a `heapless::Vec` has wherever `N` is large enough to
//!   be a stack budget rather than a capacity.
//! - [`BitMat`] — a bit *matrix* plus the row-wise set ops
//!   ([`and_into`], [`or_into`], [`is_disjoint`], …).
//! - [`SlotArena`] — the generational slot store for *intrusively linked*
//!   node graphs: claim/release raw slots on a vacancy freelist, mint a
//!   [`SlotId`] per logical incarnation.  The shape the atomic slot maps
//!   deliberately cannot express (raw cross-slot links, occupancy separate
//!   from handle-worthiness).
//! - [`PriorityIndex`] — removable indexed binary max-heap over nodes that
//!   carry their own heap back-pointer ([`PriorityLink`]); pairs with
//!   [`SlotArena`] to index any subset of an intrusive node graph with
//!   O(log N) arbitrary removal.
//!
//! **Not here, on purpose:**
//! - `AtomicSlab` and the `Slab` family: lock-free, and it belongs here, but
//!   its chunk allocation is bound to the kernel heap singleton
//!   (`Box::new_in(kernel_heap())`, re-supplied on free from `&self`).  It
//!   needs a static-allocator seam before it can move, and its tests need a
//!   real heap; that is its own change, not a rider on this one.
//! - `Tlsf` and `SeqRing`: they compose the kernel's `SpinLock` / `SeqLock`,
//!   so they are neither lock-free nor dependency-free.
//! - `BumpArena` and the `oom` attribution table: the arena's failure path
//!   names *which kernel allocator* ran dry, which is kernel policy, not a
//!   property of a collection.  The crate's own tests carry a minimal
//!   test-local arena instead (`testing`), which is the house rule anyway —
//!   never `Global`, even in a test.

#![no_std]
#![cfg_attr(test, allow(clippy::undocumented_unsafe_blocks))]
#![cfg_attr(feature = "alloc", feature(allocator_api))]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(test)]
extern crate std;

#[cfg(feature = "alloc")]
mod bitmat;
mod bitset;
#[cfg(feature = "alloc")]
mod bitvec;
#[cfg(feature = "alloc")]
mod boundedvec;
mod genid;
#[cfg(feature = "alloc")]
mod pages;
#[cfg(feature = "alloc")]
mod priority;
#[cfg(feature = "alloc")]
mod slotarena;
mod slotmap;
#[cfg(feature = "alloc")]
mod slotvec;

#[cfg(all(test, feature = "alloc"))]
mod testing;

#[cfg(feature = "alloc")]
pub use bitmat::{
    BitMat, and_into, andn_into, for_each_set, is_disjoint, is_zero, or_into, popcount,
};
pub use bitset::{AtomicBitSet, BitSet};
#[cfg(feature = "alloc")]
pub use bitvec::{AtomicBitVec, BitVec};
#[cfg(feature = "alloc")]
pub use boundedvec::BoundedVec;
pub use genid::{GenId, GenIdAllocator};
#[cfg(feature = "alloc")]
pub use priority::{PriorityError, PriorityIndex, PriorityLink, UNINDEXED};
#[cfg(feature = "alloc")]
pub use slotarena::SlotArena;
pub use slotmap::{
    AtomicSlotMap, SlotAccess, SlotId, SlotIter, SlotRead, SlotWrite, SlotWriteIter,
};
#[cfg(feature = "alloc")]
pub use slotvec::{AtomicSlotVec, SlotVecIter, SlotVecRead, SlotVecWrite, SlotVecWriteIter};

#[cfg(feature = "alloc")]
pub use pages::{PageError, PageGuard, PagePresence, PageStore};
