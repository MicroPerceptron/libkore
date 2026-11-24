use alloc::{collections::TryReserveError, vec::Vec};
use core::alloc::Allocator;

use ax::{AxError, Graph, TensorId};
use axui_core::{Frame, FrameView};

/// Failure to provision bounded compositor state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheError {
    /// A cache must retain at least one entry.
    ZeroCapacity,
    /// The caller-supplied allocator could not reserve the requested storage.
    Allocation,
}

impl From<TryReserveError> for CacheError {
    fn from(_: TryReserveError) -> Self {
        Self::Allocation
    }
}

/// One retained display list and its provider-specific prepared payload.
pub struct PreparedScene<A: Allocator, K, E> {
    source: Frame<A>,
    key: K,
    payload: E,
    last_used: u64,
}

impl<A: Allocator, K: PartialEq, E> PreparedScene<A, K, E> {
    fn matches<B: Allocator>(&self, source: FrameView<'_, B>, key: &K) -> bool {
        self.key == *key && self.source.view() == source
    }
}

impl<A: Allocator, K, E> PreparedScene<A, K, E> {
    /// Returns the retained display-list key.
    pub fn source(&self) -> FrameView<'_, A> {
        self.source.view()
    }

    /// Returns the frontend/runtime-defined structural key.
    pub const fn key(&self) -> &K {
        &self.key
    }

    /// Borrows the provider-specific payload.
    pub const fn payload(&self) -> &E {
        &self.payload
    }

    /// Mutably borrows the provider-specific payload.
    pub const fn payload_mut(&mut self) -> &mut E {
        &mut self.payload
    }

    /// Consumes the entry and returns the provider payload for explicit
    /// release by the runtime that created it.
    pub fn into_payload(self) -> E {
        self.payload
    }
}

/// Bounded least-recently-used prepared-scene cache.
///
/// Entry metadata is reserved at construction. Retaining a new display list
/// uses the allocator explicitly passed to [`PreparedCache::insert`]. Evicted
/// payloads are returned to the caller; provider handles are never silently
/// dropped across a release boundary.
pub struct PreparedCache<A: Allocator, K, E> {
    entries: Vec<PreparedScene<A, K, E>, A>,
    capacity: usize,
    clock: u64,
}

impl<A: Allocator + Clone, K: PartialEq, E> PreparedCache<A, K, E> {
    /// Creates a bounded cache in `allocator`.
    pub fn new_in(capacity: usize, allocator: A) -> Result<Self, CacheError> {
        if capacity == 0 {
            return Err(CacheError::ZeroCapacity);
        }
        let mut entries = Vec::new_in(allocator);
        entries.try_reserve_exact(capacity)?;
        Ok(Self {
            entries,
            capacity,
            clock: 0,
        })
    }

    /// Finds and touches an exact structural scene key.
    pub fn find<B: Allocator>(&mut self, source: FrameView<'_, B>, key: &K) -> Option<usize> {
        let index = self
            .entries
            .iter()
            .position(|entry| entry.matches(source, key))?;
        self.clock = self.clock.wrapping_add(1);
        self.entries[index].last_used = self.clock;
        Some(index)
    }

    /// Inserts a prepared payload and returns an evicted entry, if any.
    pub fn insert<B: Allocator>(
        &mut self,
        source: FrameView<'_, B>,
        key: K,
        payload: E,
        allocator: A,
    ) -> Option<PreparedScene<A, K, E>> {
        self.clock = self.clock.wrapping_add(1);
        let evicted = (self.entries.len() == self.capacity).then(|| {
            let lru = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(index, _)| index)
                .expect("a full nonzero cache has an entry");
            self.entries.swap_remove(lru)
        });
        self.entries.push(PreparedScene {
            source: Frame::copy_from_in(source, allocator),
            key,
            payload,
            last_used: self.clock,
        });
        evicted
    }
}

impl<A: Allocator, K, E> PreparedCache<A, K, E> {
    /// Returns the number of live prepared entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the configured entry bound.
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Borrows an entry by the stable index returned from [`Self::find`] or
    /// the most recent insertion (`len() - 1`).
    pub fn get(&self, index: usize) -> Option<&PreparedScene<A, K, E>> {
        self.entries.get(index)
    }

    /// Removes one retained entry so its payload can be explicitly released.
    pub fn pop(&mut self) -> Option<PreparedScene<A, K, E>> {
        self.entries.pop()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SignatureEntry {
    signature: u64,
    last_used: u64,
}

/// Bounded allocation-explicit memo for structural provider decisions.
pub struct SignatureCache<A: Allocator> {
    entries: Vec<SignatureEntry, A>,
    capacity: usize,
    clock: u64,
}

impl<A: Allocator> SignatureCache<A> {
    /// Creates a cache and reserves all of its storage up front.
    pub fn new_in(capacity: usize, allocator: A) -> Result<Self, CacheError> {
        if capacity == 0 {
            return Err(CacheError::ZeroCapacity);
        }
        let mut entries = Vec::new_in(allocator);
        entries.try_reserve_exact(capacity)?;
        Ok(Self {
            entries,
            capacity,
            clock: 0,
        })
    }

    /// Returns whether a signature is memoized and refreshes its LRU age.
    pub fn contains(&mut self, signature: u64) -> bool {
        let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.signature == signature)
        else {
            return false;
        };
        self.clock = self.clock.wrapping_add(1);
        entry.last_used = self.clock;
        true
    }

    /// Memoizes a signature without allocating after construction.
    pub fn insert(&mut self, signature: u64) {
        if self.contains(signature) {
            return;
        }
        self.clock = self.clock.wrapping_add(1);
        let entry = SignatureEntry {
            signature,
            last_used: self.clock,
        };
        if self.entries.len() < self.capacity {
            self.entries.push(entry);
        } else {
            let lru = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(index, _)| index)
                .expect("a full nonzero cache has an entry");
            self.entries[lru] = entry;
        }
    }
}

/// Hashes operator topology and tensor ABI while excluding concrete tensor
/// dimensions and constant payloads.
///
/// A structural provider rejection can therefore survive live resize without
/// poisoning a different operator graph or output precision.
pub fn structural_graph_signature(graph: Graph<'_>) -> Result<u64, AxError> {
    let mut signature = 0xcbf2_9ce4_8422_2325_u64;
    let mut mix = |value: u64| {
        signature ^= value;
        signature = signature.wrapping_mul(0x0000_0100_0000_01b3);
    };
    mix(graph.tensor_count() as u64);
    mix(graph.node_count() as u64);
    for index in 0..graph.tensor_count() {
        let tensor = graph.tensor(TensorId(index as u32))?;
        mix(tensor.dtype as u8 as u64);
        mix(tensor.rank() as u64);
        mix(tensor.flags.0 as u64);
    }
    for index in 0..graph.node_count() {
        let node = graph.node(index)?;
        mix(node.op.0 as u64);
        mix(node.flags.0 as u64);
        mix(node.input_count() as u64);
        mix(node.output_count() as u64);
        mix(node.attributes().len() as u64);
        for reference in node.inputs().chain(node.outputs()) {
            mix(reference.0 as u64);
        }
    }
    Ok(signature)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axui_core::Rect;
    use std::alloc::System;

    fn frame(width: f32) -> Frame<System> {
        Frame::new_in(Rect::new(0.0, 0.0, width, 40.0), 1.0, System)
    }

    #[test]
    fn prepared_cache_is_bounded_lru_and_returns_provider_payload() {
        let first = frame(10.0);
        let second = frame(20.0);
        let third = frame(30.0);
        let mut cache = PreparedCache::new_in(2, System).unwrap();

        assert!(cache.insert(first.view(), (1, 0), 11, System,).is_none());
        assert!(cache.insert(second.view(), (2, 0), 22, System,).is_none());
        assert!(cache.find(first.view(), &(1, 0),).is_some());

        let evicted = cache.insert(third.view(), (3, 1), 33, System).unwrap();

        assert_eq!(evicted.into_payload(), 22);
        assert_eq!(cache.len(), 2);
        assert!(cache.find(first.view(), &(1, 0),).is_some());
        assert!(cache.find(second.view(), &(2, 0),).is_none());
    }

    #[test]
    fn signature_cache_is_bounded_without_false_hits() {
        let mut cache = SignatureCache::new_in(2, System).unwrap();
        cache.insert(1);
        cache.insert(2);
        assert!(cache.contains(1));
        cache.insert(3);

        assert!(cache.contains(1));
        assert!(!cache.contains(2));
        assert!(cache.contains(3));
    }
}
