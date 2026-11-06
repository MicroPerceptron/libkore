//! The reverse-edge index: for every object, the containment edges that
//! name it.
//!
//! A forward edge lives in its parent directory's child list; this is the
//! same edge seen from the child, so `Parents` reads upward in time
//! proportional to what it finds rather than walking every directory.  It
//! is derived state: every row is written in the same commit as the forward
//! edge it mirrors, rebuilt at mount by the edge restore that re-links the
//! tree, and never persisted — the durable format is unchanged.
//!
//! Most objects have one parent (`res/FILESYSTEM_SCALE.md` measured 559,796
//! edges for 559,799 objects), so a row keeps its first edge inline and
//! spills only the extra links to the service heap.  A row is claimed on its
//! own, not through the object: a file transaction holds the object's claim
//! across provider I/O, and moving or renaming that file must not turn
//! `Busy` for it.  Claims never wait, like every other engine claim; a
//! contended row is `Busy` and the request retries.
//!
//! An edge is named by its [`EdgeId`] — parent directory and entry slot,
//! both generational — so a rename, which keeps the entry, never touches
//! the index, and a stale entry left by a bug would fail its revalidation
//! rather than name a successor.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, Ordering};

use crate::bounds::MAX_OBJECTS;
use crate::error::{FsError, FsResult};
use crate::namespace::EdgeId;
use crate::object::{ObjectId, ObjectIdExt};

/// One row per object slot.
pub(crate) struct ParentIndex {
    rows: Box<[ParentRow]>,
}

struct ParentRow {
    claimed: AtomicBool,
    edges: UnsafeCell<Edges>,
}

/// A row's edges, in the order they were made.
#[derive(Default)]
struct Edges {
    first: Option<EdgeId>,
    rest: Vec<EdgeId>,
}

// SAFETY: a row's edges are read and written only through a `ParentClaim`,
// which holds the row's `claimed` flag exclusively for its lifetime.
unsafe impl Sync for ParentRow {}

impl ParentIndex {
    /// An empty index with a row for every object slot.
    pub(crate) fn try_new() -> Option<Self> {
        let mut rows = Vec::new();
        rows.try_reserve_exact(MAX_OBJECTS).ok()?;
        rows.resize_with(MAX_OBJECTS, || ParentRow {
            claimed: AtomicBool::new(false),
            edges: UnsafeCell::new(Edges::default()),
        });
        Some(Self {
            rows: rows.into_boxed_slice(),
        })
    }

    /// Claim `object`'s row, or `Busy` while another commit holds it.
    pub(crate) fn claim(&self, object: ObjectId) -> FsResult<ParentClaim<'_>> {
        let row = usize::try_from(object.key().index())
            .ok()
            .and_then(|index| self.rows.get(index))
            .ok_or(FsError::AccessLost)?;
        row.claimed
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .map_err(|_| FsError::Busy)?;
        Ok(ParentClaim { row })
    }
}

/// Exclusive access to one object's row until dropped.
pub(crate) struct ParentClaim<'a> {
    row: &'a ParentRow,
}

impl Drop for ParentClaim<'_> {
    fn drop(&mut self) {
        self.row.claimed.store(false, Ordering::Release);
    }
}

impl ParentClaim<'_> {
    fn edges(&self) -> &Edges {
        // SAFETY: this claim holds the row exclusively (`ParentIndex::claim`).
        unsafe { &*self.row.edges.get() }
    }

    fn edges_mut(&mut self) -> &mut Edges {
        // SAFETY: as `edges`, and `&mut self` makes this the only borrow.
        unsafe { &mut *self.row.edges.get() }
    }

    /// Edges naming the object.
    pub(crate) fn len(&self) -> usize {
        let edges = self.edges();
        usize::from(edges.first.is_some()) + edges.rest.len()
    }

    /// The edges naming the object, in the order they were made.
    pub(crate) fn iter(&self) -> impl Iterator<Item = EdgeId> + '_ {
        let edges = self.edges();
        edges.first.into_iter().chain(edges.rest.iter().copied())
    }

    /// Make room for one more edge, so a commit that then publishes its
    /// forward edge never has to refuse the reverse one.
    pub(crate) fn reserve(&mut self) -> FsResult<()> {
        let edges = self.edges_mut();
        if edges.first.is_some() {
            edges.rest.try_reserve(1).map_err(|_| FsError::OutOfEdges)?;
        }
        Ok(())
    }

    /// Record `edge`, after [`Self::reserve`] made room for it.
    pub(crate) fn add(&mut self, edge: EdgeId) {
        let edges = self.edges_mut();
        match edges.first {
            None => edges.first = Some(edge),
            Some(_) => {
                debug_assert!(edges.rest.len() < edges.rest.capacity(), "reserve first");
                edges.rest.push(edge);
            }
        }
    }

    /// Forget `edge`; an edge the row does not hold is ignored.  A spill
    /// that empties is returned to the heap.
    pub(crate) fn remove(&mut self, edge: EdgeId) {
        let edges = self.edges_mut();
        if edges.first == Some(edge) {
            edges.first = if edges.rest.is_empty() {
                None
            } else {
                Some(edges.rest.remove(0))
            };
        } else if let Some(at) = edges.rest.iter().position(|held| *held == edge) {
            edges.rest.remove(at);
        }
        if edges.rest.is_empty() && edges.rest.capacity() != 0 {
            edges.rest = Vec::new();
        }
    }

    /// The edge `old` became `new` (a move): same object, same place in
    /// the row, no allocation.
    pub(crate) fn replace(&mut self, old: EdgeId, new: EdgeId) {
        let edges = self.edges_mut();
        if let Some(held) = edges
            .first
            .iter_mut()
            .chain(edges.rest.iter_mut())
            .find(|held| **held == old)
        {
            *held = new;
        }
    }

    /// Forget every edge (the object's slot is being freed).
    pub(crate) fn clear(&mut self) {
        *self.edges_mut() = Edges::default();
    }
}
