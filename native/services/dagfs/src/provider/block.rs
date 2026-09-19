//! Filesystem placement over one affine direct block channel.
//!
//! This adapter is the sole translation point between filesystem placement
//! semantics and block geometry.  The engine above remains generic over
//! [`StorageProvider`]; the channel below remains unaware of files, names,
//! revisions, and authority.

use core::cell::{Cell, RefCell};

use kore_driver::block::{BlockChannel, BlockError, BufferId, Completion, DirectBlockDevice};

use crate::{
    bounds::MAX_BLOCK_PLACEMENTS,
    error::ProviderError,
    object::ObjectKind,
    provider::{
        StorageLimits, StorageProvider,
        extent::{ExtentAllocator, ExtentId},
    },
    slot::{SlotKey, SlotTable},
};

/// Payload floor encoded by versions 1 and 2, and by pre-7b version 3 roots.
pub const LEGACY_DATA_START_LBA: u64 = 72;
/// Exclusive end of the whole-disk GPT discovery window.
///
/// This remains fixed when a mounted volume selects a different payload
/// floor; broadening the filesystem reservation must not broaden which
/// foreign-disk blocks GPT parsing is willing to treat as metadata.
pub const GPT_DISCOVERY_END_LBA: u64 = 72;
/// Per-buffer bulk-I/O ceiling. Four 256 KiB buffers across the whole-disk ->
/// designated-region rebuild, plus any needed private PRP lists and controller
/// queues, fit inside the founded 2 MiB bump DMA pool without relying on
/// reclamation; the pool itself is already mapped by a large IOMMU leaf.
const IO_BUFFER_BYTES: usize = 256 * 1024;
/// Largest byte count one payload read or write may transfer on the block
/// provider — 128 logical blocks at 512 B, one command.
///
/// The engine's shared [`MAX_TRANSFER`] (4 KiB) bounds the RAM provider's
/// chunk model and the engine's own stack buffers; it is not a property of
/// a device that accepts 256 KiB per command.  Measured on metal
/// (`status fs`, 2026-09-09): 17400 commands moved 9 MiB, about 1 KiB
/// each, at 49 µs per command — the per-command cost, not the bytes, was
/// the throughput.  64 KiB keeps a whole read inside the DMA buffer with
/// room for the block-alignment head, needs no stack scratch (bytes move
/// between the registered buffer and the caller directly), and sits at the
/// point past which a single NVMe command gains little.
const BLOCK_MAX_TRANSFER: usize = 64 * 1024;
/// How far a payload read runs ahead of what was asked, when the placement
/// has that much more to give: the next sequential read is then served from
/// the buffer the provider already owns, with no command at all.  Half the
/// DMA buffer, so a 64 KiB request never has to evict what it just fetched;
/// 128 KiB is also where an NVMe read stops getting cheaper per byte.
const READ_AHEAD_BYTES: usize = 128 * 1024;
/// Stack scratch for the one copy that must go through the CPU: mirroring
/// a partial edge block between the two DMA buffers on a write.  One block
/// at most — the provider refuses devices with larger ones at construction.
const COPY_CHUNK_BYTES: usize = 4096;
const BULK_DEPTH: usize = 2;

/// What the `io` buffer currently holds from the device — the read-ahead
/// window.  Cleared by every path that writes, reuses `io`, or changes the
/// bytes it mirrors, so a stale hit is structurally impossible: the only
/// reader is `read_bytes`, and every other operation forgets it first.
#[derive(Clone, Copy, Debug)]
struct HeldRead {
    lba: u64,
    blocks: u32,
}

/// A generational placement key. It is naming data, never authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockKey(SlotKey);

/// One tracked payload placement. `lba`/`blocks` are hot-path caches of the
/// geometry the extent index also holds for `extent` — the IO paths read
/// them without touching the index, so every site that installs or replaces
/// `extent` must refresh them from the same allocation, and `extent` is
/// `None` exactly when `blocks == 0` (nothing allocated). `retire` and the
/// checkpoint drains police that pairing with `Integrity` refusals.
#[derive(Clone, Copy, Debug)]
struct Placement {
    lba: u64,
    blocks: u32,
    len: u64,
    committed: bool,
    extent: Option<ExtentId>,
    /// High-water mark of meaningful bytes, in blocks: every block at or
    /// past it holds the zeros `allocate`/`resize` put there and nothing
    /// has overwritten.  A write that begins at or after it has no old
    /// bytes to fetch or keep — the rollback image is zeros by
    /// construction.  Recovered and copied placements start at `blocks`:
    /// nothing is known about their bytes except that they are all
    /// meaningful.  A `u32` in blocks, not a `u64` in bytes: it rides in
    /// the record's padding, see [`PLACEMENT_BYTES`].
    written_blocks: u32,
}

/// The size of one [`Placement`], pinned.
///
/// Three placement-bound tables of these (`placements`, `retired`,
/// `fallback`) live inside the provider state, which lives inside the
/// service's `serve` future, which is polled on the founder's 4 MiB main
/// stack.  Growing this record by one word grew that future by tens of
/// KiB and overflowed the stack by 744 bytes on metal (2026-09-09, two
/// boots of PR #958): the record's size is part of the stack budget, so a
/// change to it must be deliberate.
const PLACEMENT_BYTES: usize = 40;
const _: () = assert!(core::mem::size_of::<Placement>() == PLACEMENT_BYTES);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlockProviderStats {
    pub reads: u64,
    pub writes: u64,
    pub flushes: u64,
    pub discards: u64,
    pub copied_bytes: u64,
    pub rollbacks: u64,
    pub late_completions: u64,
    pub orphan_reclaims: u64,
    pub reclaim_failures: u64,
    pub bulk_batches: u64,
    /// Payload reads answered from the read-ahead window with no command.
    pub readahead_hits: u64,
    /// Device reads issued on behalf of the read-ahead window.
    pub readahead_reads: u64,
    /// Payload writes that began and ended on block boundaries, so no old
    /// byte had to be fetched into the write image first.
    pub aligned_writes: u64,
    /// Single-block reads of a partial head or tail block, fetched so an
    /// unaligned write into a fresh extent overwrites only its own bytes.
    pub edge_reads: u64,
    /// Writes into an uncommitted extent past its high-water mark: the old
    /// bytes were the allocation's zeros, so no rollback image was read.
    pub pristine_writes: u64,
    /// Whole-object copies moved DMA buffer to DMA buffer for a version
    /// lend, never through the CPU.
    pub snapshot_copies: u64,
    /// Completion waits that ran their budget out without the awaited
    /// completion arriving.
    pub expired_waits: u64,
    /// Of those, the ones where the backend reported itself HEALTHY — so the
    /// device was slow, not dead, and the channel was kept. A rising count is
    /// a device that needs a longer budget, and would previously have been
    /// answered by retiring the volume.
    pub slow_waits: u64,
}

/// Durable coordinates of one file payload. This is allocation metadata,
/// not authority and not the process-local [`BlockKey`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DurableExtent {
    pub lba: u64,
    pub blocks: u32,
    pub len: u64,
}

#[derive(Clone, Copy)]
enum RecoveryOwner {
    Live(usize),
    Fallback,
}

#[derive(Clone, Copy)]
struct RecoveryRecord {
    extent: DurableExtent,
    owner: RecoveryOwner,
}

struct State<D: DirectBlockDevice, const B: usize, const O: usize> {
    channel: BlockChannel<D, B, O>,
    io: BufferId,
    rollback: BufferId,
    buffer_bytes: usize,
    /// The read-ahead window `io` holds, if any.
    held: Option<HeldRead>,
    placements: SlotTable<Placement, MAX_BLOCK_PLACEMENTS>,
    retired: heapless::Vec<Placement, MAX_BLOCK_PLACEMENTS>,
    fallback: heapless::Vec<Placement, MAX_BLOCK_PLACEMENTS>,
    extents: ExtentAllocator,
    /// First block the payload allocator may name. Metadata I/O is confined
    /// below this boundary; payload I/O is confined at or above it.
    data_start: u64,
    /// Exclusive end of the volume-owned payload arena.
    arena_end: u64,
    stats: BlockProviderStats,
}

/// One filesystem-worker provider over one exclusively-owned direct channel.
/// `RefCell` is task-local and deliberately `!Sync`; parallel workers receive
/// distinct provider/channel values rather than locking one qpair.
pub struct BlockProvider<D: DirectBlockDevice, const B: usize = 8, const O: usize = 32> {
    state: RefCell<State<D, B, O>>,
    limits: Cell<StorageLimits>,
}

impl<D: DirectBlockDevice, const B: usize, const O: usize> BlockProvider<D, B, O> {
    pub fn new(channel: BlockChannel<D, B, O>) -> Result<Self, ProviderError> {
        Self::new_with_data_start(channel, LEGACY_DATA_START_LBA)
    }

    /// Construct behind a conservative pre-root metadata boundary.
    ///
    /// Durable mounting passes the largest supported reservation here, reads
    /// only roots/GPT below it, then calls [`configure_arena`](Self::configure_arena)
    /// with the authenticated per-volume bounds before constructing an engine.
    pub(crate) fn new_with_data_start(
        mut channel: BlockChannel<D, B, O>,
        data_start: u64,
    ) -> Result<Self, ProviderError> {
        let geometry = channel.geometry();
        let block = geometry.block_size as usize;
        if block > 4096 || data_start < LEGACY_DATA_START_LBA || data_start >= geometry.block_count
        {
            return Err(ProviderError::Integrity);
        }
        let command_bytes = usize::try_from(
            u64::from(geometry.max_transfer_blocks) * u64::from(geometry.block_size),
        )
        .unwrap_or(usize::MAX);
        let io_bytes = IO_BUFFER_BYTES
            .min(command_bytes)
            .min(channel.max_buffer_bytes());
        if io_bytes < block || B < 2 {
            return Err(ProviderError::Backpressure);
        }
        let io = channel.register_buffer(io_bytes).map_err(map_block)?;
        let rollback = channel.register_buffer(io_bytes).map_err(map_block)?;
        let total = (geometry.block_count - data_start)
            .checked_mul(geometry.block_size as u64)
            .ok_or(ProviderError::Integrity)?;
        let max_object = total.min(u32::MAX as u64 * geometry.block_size as u64);
        // A byte transfer beginning at the last byte of a block can cover
        // one more block than its payload. Keep that head/tail allowance in
        // the declared limit while still admitting a one-byte transfer on a
        // device whose maximum is exactly one block.
        let payload_limit = io_bytes.saturating_sub(block.saturating_sub(1));
        if payload_limit == 0 {
            return Err(ProviderError::Integrity);
        }
        let limits = StorageLimits {
            total_bytes: total,
            max_object_bytes: max_object,
            max_transfer: BLOCK_MAX_TRANSFER.min(payload_limit),
            granularity: block,
        };
        Ok(Self {
            state: RefCell::new(State {
                channel,
                io,
                rollback,
                buffer_bytes: io_bytes,
                held: None,
                placements: SlotTable::new(),
                retired: heapless::Vec::new(),
                fallback: heapless::Vec::new(),
                extents: ExtentAllocator::new(data_start, geometry.block_count)?,
                data_start,
                arena_end: geometry.block_count,
                stats: BlockProviderStats::default(),
            }),
            limits: Cell::new(limits),
        })
    }

    /// Configure the exact payload arena declared by the selected durable
    /// root before any extent is adopted or allocated.
    ///
    /// Recovery begins behind the largest supported reservation so it can
    /// safely inspect roots from every supported profile. Once a root has
    /// authenticated its `data_start` and `arena_blocks`, this call makes that
    /// on-disk fact the provider's one addressability boundary. Refusing a
    /// populated provider prevents a live allocation from changing meaning
    /// underneath its key.
    pub fn configure_arena(&self, data_start: u64, arena_blocks: u64) -> Result<(), ProviderError> {
        let mut state = self.state.borrow_mut();
        if !state.placements.is_empty() || !state.retired.is_empty() || !state.fallback.is_empty() {
            return Err(ProviderError::Integrity);
        }
        let geometry = state.channel.geometry();
        let arena_end = data_start
            .checked_add(arena_blocks)
            .ok_or(ProviderError::Integrity)?;
        if data_start < LEGACY_DATA_START_LBA
            || arena_blocks == 0
            || arena_end > geometry.block_count
        {
            return Err(ProviderError::Integrity);
        }
        let total = arena_blocks
            .checked_mul(u64::from(geometry.block_size))
            .ok_or(ProviderError::Integrity)?;
        let max_object = total.min(u64::from(u32::MAX) * u64::from(geometry.block_size));
        state.extents.reset(data_start, arena_end)?;
        state.data_start = data_start;
        state.arena_end = arena_end;
        self.limits.set(StorageLimits {
            total_bytes: total,
            max_object_bytes: max_object,
            max_transfer: self.limits.get().max_transfer,
            granularity: geometry.block_size as usize,
        });
        Ok(())
    }

    /// The exact payload arena currently enforced by the provider.
    pub fn arena(&self) -> (u64, u64) {
        let state = self.state.borrow();
        (state.data_start, state.arena_end - state.data_start)
    }

    pub fn stats(&self) -> BlockProviderStats {
        let state = self.state.borrow();
        BlockProviderStats {
            late_completions: state.channel.late_completions(),
            ..state.stats
        }
    }

    pub fn placements(&self) -> usize {
        self.state.borrow().placements.len()
    }

    pub fn geometry(&self) -> kore_driver::block::BlockGeometry {
        self.state.borrow().channel.geometry()
    }

    /// Blocks the allocator cannot currently reuse, including the fixed
    /// metadata prefix and every live, retired, or fallback-root extent.
    ///
    /// Retired and fallback extents are intentionally counted even though
    /// they are absent from `placements`: a currently or previously durable
    /// root can still name them, so presenting them as free capacity would be
    /// a false filesystem-pressure metric.
    pub fn occupied_blocks(&self) -> Option<u64> {
        let state = self.state.borrow();
        state.arena_end.checked_sub(state.extents.free_blocks())
    }

    pub fn channel_stats(&self) -> kore_driver::block::ChannelStats {
        self.state.borrow().channel.stats()
    }

    /// Current lifecycle of the mounted block channel.
    ///
    /// Unlike operation counters this is a one-way posture edge: once `Lost`,
    /// status must continue to report the condemned channel even when no
    /// later request is attempted.
    pub fn channel_state(&self) -> kore_driver::block::ChannelState {
        self.state.borrow().channel.state()
    }

    pub fn mark_lost(&self) {
        self.state.borrow_mut().channel.mark_lost();
    }

    pub fn extent(&self, key: BlockKey) -> Result<DurableExtent, ProviderError> {
        let state = self.state.borrow();
        let placement = *state
            .placements
            .get(key.0)
            .ok_or(ProviderError::UnknownKey)?;
        Ok(DurableExtent {
            lba: placement.lba,
            blocks: placement.blocks,
            len: placement.len,
        })
    }

    /// Rebuild the boot-local extent index from both recovery-selectable
    /// roots. `live` keys are returned in input order; extents unique to
    /// `fallback` remain protected until its root slot is replaced.
    ///
    /// The records are sorted once and then admitted in ascending LBA order,
    /// so mount costs O(N log N) rather than performing one growing occupied
    /// scan per record. The rebuild is prepared off to the side and only
    /// installed after all bounds, overlap, and table-capacity checks pass.
    pub(crate) fn recover_extents(
        &self,
        live: &[DurableExtent],
        fallback: &[DurableExtent],
    ) -> Result<heapless::Vec<BlockKey, MAX_BLOCK_PLACEMENTS>, ProviderError> {
        if live.len() > MAX_BLOCK_PLACEMENTS || fallback.len() > MAX_BLOCK_PLACEMENTS {
            return Err(ProviderError::PlacementsFull);
        }

        let mut state = self.state.borrow_mut();
        if !state.placements.is_empty() || !state.retired.is_empty() || !state.fallback.is_empty() {
            return Err(ProviderError::Integrity);
        }
        for extent in live.iter().chain(fallback) {
            state.validate_extent(*extent)?;
        }

        // Design-bounded like every sibling buffer here: the entry guard
        // caps each input at MAX_BLOCK_PLACEMENTS, so a refused push below
        // is index corruption, never volume pressure.
        let mut records = heapless::Vec::<RecoveryRecord, { MAX_BLOCK_PLACEMENTS * 2 }>::new();
        for (index, extent) in live.iter().copied().enumerate() {
            if extent.blocks != 0 {
                records
                    .push(RecoveryRecord {
                        extent,
                        owner: RecoveryOwner::Live(index),
                    })
                    .map_err(|_| ProviderError::Integrity)?;
            }
        }
        for extent in fallback.iter().copied() {
            if extent.blocks != 0 {
                records
                    .push(RecoveryRecord {
                        extent,
                        owner: RecoveryOwner::Fallback,
                    })
                    .map_err(|_| ProviderError::Integrity)?;
            }
        }
        // Intra-group order is immaterial — the group walk below gathers
        // live/fallback from the whole (lba, blocks)-equal group — so the
        // sort key is exactly the group identity.
        records.sort_unstable_by(|left, right| {
            left.extent
                .lba
                .cmp(&right.extent.lba)
                .then(left.extent.blocks.cmp(&right.extent.blocks))
        });

        let mut allocations = heapless::Vec::<Option<ExtentId>, MAX_BLOCK_PLACEMENTS>::new();
        for _ in live {
            allocations
                .push(None)
                .map_err(|_| ProviderError::Integrity)?;
        }
        let mut protected = heapless::Vec::<Placement, MAX_BLOCK_PLACEMENTS>::new();
        let mut rebuilt = ExtentAllocator::new(state.data_start, state.arena_end)?;
        let mut previous_end = state.data_start;
        for group in records.chunk_by(|left, right| {
            left.extent.lba == right.extent.lba && left.extent.blocks == right.extent.blocks
        }) {
            let extent = group[0].extent;
            let end = extent
                .lba
                .checked_add(u64::from(extent.blocks))
                .ok_or(ProviderError::Integrity)?;
            if extent.lba < previous_end {
                return Err(ProviderError::Integrity);
            }

            let mut live_index = None;
            let mut fallback_record = None;
            for record in group {
                match record.owner {
                    RecoveryOwner::Live(index) => {
                        if live_index.replace(index).is_some() {
                            return Err(ProviderError::Integrity);
                        }
                    }
                    RecoveryOwner::Fallback => {
                        if fallback_record.replace(record.extent).is_some() {
                            return Err(ProviderError::Integrity);
                        }
                    }
                }
            }

            let allocation = rebuilt.reserve_sorted(extent.lba, extent.blocks)?;
            if let Some(index) = live_index {
                allocations[index] = Some(allocation);
            } else if let Some(extent) = fallback_record {
                protected
                    .push(Placement {
                        lba: extent.lba,
                        blocks: extent.blocks,
                        len: extent.len,
                        committed: true,
                        extent: Some(allocation),
                        written_blocks: extent.blocks,
                    })
                    .map_err(|_| ProviderError::PlacementsFull)?;
            }
            previous_end = end;
        }

        let mut keys = heapless::Vec::<BlockKey, MAX_BLOCK_PLACEMENTS>::new();
        for (extent, allocation) in live.iter().copied().zip(allocations) {
            // The entry guard bounds `live` by the verified-empty table's own
            // capacity, so neither insert can refuse; a refusal here is
            // index corruption, not a partial mount to roll back.
            let key = state
                .placements
                .insert(Placement {
                    lba: extent.lba,
                    blocks: extent.blocks,
                    len: extent.len,
                    committed: true,
                    extent: allocation,
                    written_blocks: extent.blocks,
                })
                .map(BlockKey)
                .map_err(|_| ProviderError::Integrity)?;
            keys.push(key).map_err(|_| ProviderError::Integrity)?;
        }
        state.extents = rebuilt;
        state.fallback = protected;
        Ok(keys)
    }

    /// Read an aligned span from the metadata reservation.
    pub async fn read_metadata(&self, lba: u64, dst: &mut [u8]) -> Result<(), ProviderError> {
        self.state.borrow_mut().metadata_read(lba, dst).await
    }

    /// Write an aligned span in the inactive metadata arena.
    pub async fn write_metadata(&self, lba: u64, src: &[u8]) -> Result<(), ProviderError> {
        self.state.borrow_mut().metadata_write(lba, src).await
    }

    pub async fn flush_device(&self) -> Result<(), ProviderError> {
        self.state.borrow_mut().flush().await
    }

    /// Retire extents made obsolete by the root that has just become durable.
    /// Reclamation is best-effort and never weakens the durability ack: the
    /// published root no longer references these blocks, so a discard failure
    /// is a leak to report, not a failed commit.
    pub async fn checkpoint(&self) {
        let mut state = self.state.borrow_mut();
        while let Some(placement) = state.fallback.pop() {
            if placement.blocks > 0 && state.channel.geometry().discard {
                match state.discard(placement.lba, placement.blocks).await {
                    Ok(()) => {
                        state.stats.orphan_reclaims = state.stats.orphan_reclaims.saturating_add(1)
                    }
                    Err(_) => {
                        state.stats.reclaim_failures =
                            state.stats.reclaim_failures.saturating_add(1)
                    }
                }
            }
            if let Some(extent) = placement.extent
                && state.extents.release(extent).is_err()
            {
                state.stats.reclaim_failures = state.stats.reclaim_failures.saturating_add(1);
            }
        }
        while let Some(placement) = state.retired.pop() {
            // Capacity is identical and fallback was drained above.
            let _ = state.fallback.push(placement);
        }
        for (_, mut placement) in state.placements.iter_mut() {
            placement.committed = true;
        }
    }

    /// Freeze every live placement before a root record may name it.
    ///
    /// Root publication is a cut: after the root write is submitted, a
    /// failed completion or flush may still leave either the old or the new
    /// root durable. Mark the new root's extents conservatively before that
    /// submission so later mutations use COW even when the publisher cannot
    /// tell which root won. [`checkpoint`](Self::checkpoint) performs the
    /// post-flush reclamation only once the new root is known durable.
    pub(crate) fn prepare_publication(&self) {
        let state = self.state.borrow_mut();
        for (_, mut placement) in state.placements.iter_mut() {
            placement.committed = true;
        }
    }

    /// Simulate a service restart over the same already-provisioned medium.
    /// Production obtains a fresh channel from the provisioner; this helper
    /// retires only boot-local buffers and placement keys.
    pub fn restart(self) -> Result<Self, ProviderError> {
        Self::new(self.into_channel()?)
    }

    /// Simulate a cold durable mount, restoring the conservative boundary
    /// that exists before either root authenticates its exact arena.
    pub(crate) fn restart_with_data_start(self, data_start: u64) -> Result<Self, ProviderError> {
        Self::new_with_data_start(self.into_channel()?, data_start)
    }

    /// Recover the affine channel for re-provisioning — the probing path:
    /// a provider that classified a whole device as foreign hands its
    /// channel back so the composition can narrow onto a discovered
    /// designated region and rebuild.  Boot-local buffers and placement
    /// keys retire; the channel and the backend beneath it ride out intact.
    pub fn into_channel(self) -> Result<BlockChannel<D, B, O>, ProviderError> {
        let mut state = self.state.into_inner();
        state
            .channel
            .unregister_buffer(state.io)
            .map_err(map_block)?;
        state
            .channel
            .unregister_buffer(state.rollback)
            .map_err(map_block)?;
        Ok(state.channel)
    }

    /// Close this provider before abandoning its affine direct channel.
    ///
    /// Registered DMA buffers are children of the device/pool authority held
    /// by the channel backend, so they must leave before the backend is
    /// quiesced and dropped.  This explicit consuming path is used when a
    /// provisioned namespace proves to contain foreign data: the filesystem
    /// must neither keep DMA enabled against an unused disk nor let ordinary
    /// struct field drop order retire the device ahead of its buffers.
    pub async fn retire(self) -> Result<(), ProviderError> {
        let mut state = self.state.into_inner();
        // Evaluate every containment step even if an earlier one refuses; the
        // returned error reports the first broken invariant, while later
        // steps still make the best bounded attempt to close authority.
        let io = state.channel.unregister_buffer(state.io);
        let rollback = state.channel.unregister_buffer(state.rollback);
        let quiesce = state.channel.quiesce_async().await;
        io.and(rollback).and(quiesce).map_err(map_block)
    }

    /// Infrequent test/lifecycle access; refused while the qpair is busy.
    pub fn control<T>(&self, f: impl FnOnce(&mut D) -> T) -> Result<T, ProviderError> {
        self.state
            .borrow_mut()
            .channel
            .control(f)
            .map_err(map_block)
    }
}

impl<D: DirectBlockDevice, const B: usize, const O: usize> State<D, B, O> {
    fn block_size(&self) -> usize {
        self.channel.geometry().block_size as usize
    }

    /// Maximum blocks that fit both the device command and our registered
    /// I/O window. Device MDTS can be much larger than the filesystem's
    /// bounded scratch window; every buffered loop must honor the smaller.
    fn buffered_transfer_blocks(&self) -> u32 {
        self.channel
            .geometry()
            .max_transfer_blocks
            .min((self.buffer_bytes / self.block_size()) as u32)
    }

    /// Clamp one streaming command to the registered-buffer/device ceiling
    /// and the namespace's optional optimal-I/O boundary. The boundary is a
    /// performance hint rather than an addressability rule, but honoring it
    /// here costs one remainder on a slow setup loop and avoids controller
    /// read-modify-split work on the hot transfer path.
    fn streaming_blocks(&self, lba: u64, remaining: u32) -> u32 {
        let mut count = remaining.min(self.buffered_transfer_blocks());
        let boundary = self.channel.geometry().io_hints.optimal_boundary_blocks;
        if boundary != 0 {
            let within = (lba % u64::from(boundary)) as u32;
            count = count.min(boundary - within);
        }
        count
    }

    /// Shape a write so its aligned interior uses the namespace's preferred
    /// granularity and optimal command size. Unaligned head/tail fragments
    /// remain legal, while the bulk of a long extent follows the device's
    /// endurance/performance guidance.
    fn streaming_write_blocks(&self, lba: u64, remaining: u32) -> u32 {
        let mut count = self.streaming_blocks(lba, remaining);
        let hints = self.channel.geometry().io_hints;
        if hints.write_alignment_blocks > 1 {
            let within = (lba % u64::from(hints.write_alignment_blocks)) as u32;
            if within != 0 {
                return count.min(hints.write_alignment_blocks - within);
            }
        }
        if hints.optimal_write_blocks != 0 {
            count = count.min(hints.optimal_write_blocks);
        }
        if hints.preferred_write_blocks > 1 && count > hints.preferred_write_blocks {
            let rounded = count - count % hints.preferred_write_blocks;
            if rounded != 0 {
                count = rounded;
            }
        }
        count
    }

    fn blocks_for(&self, len: u64) -> Result<u32, ProviderError> {
        let block = self.block_size() as u64;
        let blocks = len.checked_add(block - 1).ok_or(ProviderError::Range)? / block;
        u32::try_from(blocks).map_err(|_| ProviderError::TooLarge)
    }

    fn validate_extent(&self, extent: DurableExtent) -> Result<(), ProviderError> {
        if extent.blocks != self.blocks_for(extent.len)? {
            return Err(ProviderError::Integrity);
        }
        if extent.blocks == 0 {
            return if extent.lba == 0 {
                Ok(())
            } else {
                Err(ProviderError::Integrity)
            };
        }
        let end = extent
            .lba
            .checked_add(extent.blocks as u64)
            .ok_or(ProviderError::Integrity)?;
        if extent.lba < self.data_start || end > self.arena_end {
            return Err(ProviderError::Integrity);
        }
        Ok(())
    }

    fn metadata_shape(&self, lba: u64, len: usize) -> Result<(usize, u32), ProviderError> {
        let block = self.block_size();
        if len == 0 || !len.is_multiple_of(block) {
            return Err(ProviderError::Range);
        }
        let blocks = u32::try_from(len / block).map_err(|_| ProviderError::TooLarge)?;
        let end = lba.checked_add(blocks as u64).ok_or(ProviderError::Range)?;
        if end > self.data_start {
            return Err(ProviderError::Range);
        }
        Ok((block, blocks))
    }

    /// Drop the read-ahead window.  Every path that writes, reuses `io`, or
    /// changes bytes the window may mirror calls this first.
    fn forget_held(&mut self) {
        self.held = None;
    }

    async fn metadata_read(&mut self, lba: u64, bytes: &mut [u8]) -> Result<(), ProviderError> {
        self.forget_held();
        let (block, blocks) = self.metadata_shape(lba, bytes.len())?;
        let mut done = 0u32;
        while done < blocks {
            let count = self.streaming_blocks(lba + u64::from(done), blocks - done);
            let at = done as usize * block;
            let len = count as usize * block;
            self.read_blocks(lba + done as u64, count, self.io).await?;
            self.channel
                .read_buffer(self.io, 0, &mut bytes[at..at + len])
                .map_err(map_block)?;
            done += count;
        }
        Ok(())
    }

    async fn metadata_write(&mut self, lba: u64, bytes: &[u8]) -> Result<(), ProviderError> {
        self.forget_held();
        let (block, blocks) = self.metadata_shape(lba, bytes.len())?;
        let mut done = 0u32;
        while done < blocks {
            let count = self.streaming_write_blocks(lba + u64::from(done), blocks - done);
            let at = done as usize * block;
            let len = count as usize * block;
            self.channel
                .write_buffer(self.io, 0, &bytes[at..at + len])
                .map_err(map_block)?;
            self.write_blocks(lba + done as u64, count, self.io).await?;
            done += count;
        }
        Ok(())
    }

    /// Decide what an exhausted wait means, and retire the channel only on
    /// evidence.
    ///
    /// A wait that expired is **not** evidence a device is dead. A completion
    /// can be late because the controller failed, but equally because its
    /// queue is deep, because it is garbage-collecting or thermally
    /// throttled, or because the host descheduled this task. Only the first
    /// justifies `mark_lost`, which is a one-way door: the channel is retired
    /// and the mounted volume goes with it.
    ///
    /// So ask the backend out of band ([`DirectBlockDevice::liveness`], one
    /// `CSTS` read for NVMe). A controller that reports itself healthy makes
    /// this a slow operation, and the request fails without condemning the
    /// channel — the next one may well succeed. `Failed` is the controller's
    /// own verdict and retires it. `Unknown` keeps the old conservative
    /// posture, because a backend with no out-of-band signal has told us
    /// nothing and a timeout is all the evidence that exists.
    fn expired(&mut self) -> ProviderError {
        let liveness = self.channel.liveness();
        self.stats.expired_waits = self.stats.expired_waits.saturating_add(1);
        if liveness == kore_driver::block::Liveness::Healthy {
            self.stats.slow_waits = self.stats.slow_waits.saturating_add(1);
            return ProviderError::Timeout;
        }
        self.channel.mark_lost();
        ProviderError::Lost
    }

    async fn wait(&mut self, request: kore_driver::block::RequestId) -> Result<(), ProviderError> {
        // Each empty wait spends the backend's whole bounded budget — the
        // spin default, or a real park on the task's held MSI-X route
        // ([`DirectBlockDevice::wait`]).  This provider submits
        // synchronously, so a mismatched id is never the answer; the
        // anomalous cross-request drain is bounded by the queue depth
        // rather than a spin count.
        for _ in 0..=O {
            // An executor-aware transport makes its correlated progress
            // here; the direct device's ready default paces in `wait` below.
            self.channel.progress().await.map_err(map_block)?;
            match self.channel.wait() {
                Some(Completion {
                    request: completed,
                    result,
                }) if completed == request => return result.map_err(map_block),
                Some(_) => {
                    self.stats.late_completions = self.stats.late_completions.saturating_add(1);
                }
                None => break,
            }
        }
        Err(self.expired())
    }

    /// Drain an exact bounded request set without assuming completion order.
    /// Every accepted request is consumed even after the first device error,
    /// so its DMA buffer is idle before the error escapes.
    async fn wait_many<const N: usize>(
        &mut self,
        requests: &mut [Option<kore_driver::block::RequestId>; N],
    ) -> Result<(), ProviderError> {
        let mut first_error = None;
        for _ in 0..=O {
            self.channel.progress().await.map_err(map_block)?;
            for completion in self.channel.poll_batch::<N>().into_iter().flatten() {
                self.settle_many(requests, completion, &mut first_error);
            }
            if requests.iter().all(Option::is_none) {
                return first_error.map_or(Ok(()), Err);
            }
            if let Some(completion) = self.channel.wait() {
                self.settle_many(requests, completion, &mut first_error);
                if requests.iter().all(Option::is_none) {
                    return first_error.map_or(Ok(()), Err);
                }
            } else {
                break;
            }
        }
        Err(self.expired())
    }

    fn settle_many<const N: usize>(
        &mut self,
        requests: &mut [Option<kore_driver::block::RequestId>; N],
        completion: Completion,
        first_error: &mut Option<ProviderError>,
    ) {
        let Some(index) = requests
            .iter()
            .position(|request| *request == Some(completion.request))
        else {
            self.stats.late_completions = self.stats.late_completions.saturating_add(1);
            return;
        };
        requests[index] = None;
        if let Err(error) = completion.result
            && first_error.is_none()
        {
            *first_error = Some(map_block(error));
        }
    }

    async fn read_blocks(
        &mut self,
        lba: u64,
        blocks: u32,
        buffer: BufferId,
    ) -> Result<(), ProviderError> {
        self.read_blocks_into(lba, blocks, buffer, 0).await
    }

    async fn read_blocks_into(
        &mut self,
        lba: u64,
        blocks: u32,
        buffer: BufferId,
        buffer_offset: usize,
    ) -> Result<(), ProviderError> {
        let request = self
            .channel
            .submit_read(lba, blocks, buffer, buffer_offset)
            .map_err(map_block)?;
        self.wait(request).await?;
        self.stats.reads = self.stats.reads.saturating_add(1);
        Ok(())
    }

    /// Fetch into `io`, where they will be written back from, the old bytes
    /// of the partial edge blocks of a `covered`-block write that begins
    /// `within` bytes into its first block and carries `len` bytes.  The
    /// interior is overwritten whole and is never read; a write that begins
    /// and ends on block boundaries fetches nothing.  Two distinct edges go
    /// out in one batch — `io` carries one command at a time, so the tail
    /// rides `rollback` and is mirrored across (one block) once both land.
    async fn read_edges(
        &mut self,
        lba: u64,
        within: usize,
        len: usize,
        covered: usize,
    ) -> Result<(), ProviderError> {
        let block = self.block_size();
        let head = within != 0;
        let tail = (within + len) % block != 0;
        let last = covered - 1;
        if !head && !tail {
            self.stats.aligned_writes = self.stats.aligned_writes.saturating_add(1);
            return Ok(());
        }
        if last == 0 || head != tail {
            // One edge — or one block that is both — straight into place.
            let at = if head || last == 0 { 0 } else { last };
            self.read_blocks_into(lba + at as u64, 1, self.io, at * block)
                .await?;
            self.stats.edge_reads = self.stats.edge_reads.saturating_add(1);
            return Ok(());
        }
        // Both edges, distinct blocks.  A queue with one usable slot refuses
        // the second stage with Backpressure; that edge then follows as its
        // own command once `io` is free, straight into place.
        let staged = [(lba, self.io), (lba + last as u64, self.rollback)];
        let mut requests = [None; 2];
        let mut stage_error = None;
        {
            let mut batch = self.channel.batch();
            for (index, (at, buffer)) in staged.into_iter().enumerate() {
                match batch.stage_read(at, 1, buffer, 0) {
                    Ok(request) => requests[index] = Some(request),
                    Err(error) => {
                        stage_error = Some(error);
                        break;
                    }
                }
            }
            batch.publish();
        }
        let accepted = requests.iter().filter(|r| r.is_some()).count();
        let tail_rode_rollback = requests[1].is_some();
        if accepted != 0 {
            self.wait_many(&mut requests).await?;
        }
        self.stats.reads = self.stats.reads.saturating_add(accepted as u64);
        self.stats.edge_reads = self.stats.edge_reads.saturating_add(accepted as u64);
        match stage_error {
            None => {}
            Some(BlockError::Backpressure) if accepted == 1 => {}
            Some(error) => return Err(map_block(error)),
        }
        if tail_rode_rollback {
            let mut chunk = [0u8; COPY_CHUNK_BYTES];
            self.channel
                .read_buffer(self.rollback, 0, &mut chunk[..block])
                .map_err(map_block)?;
            self.channel
                .write_buffer(self.io, last * block, &chunk[..block])
                .map_err(map_block)?;
        } else {
            self.read_blocks_into(lba + last as u64, 1, self.io, last * block)
                .await?;
            self.stats.edge_reads = self.stats.edge_reads.saturating_add(1);
        }
        Ok(())
    }

    async fn write_blocks(
        &mut self,
        lba: u64,
        blocks: u32,
        buffer: BufferId,
    ) -> Result<(), ProviderError> {
        let request = self
            .channel
            .submit_write(lba, blocks, buffer, 0)
            .map_err(map_block)?;
        self.wait(request).await?;
        self.stats.writes = self.stats.writes.saturating_add(1);
        Ok(())
    }

    async fn flush(&mut self) -> Result<(), ProviderError> {
        let request = self.channel.submit_flush().map_err(map_block)?;
        self.wait(request).await?;
        self.stats.flushes = self.stats.flushes.saturating_add(1);
        Ok(())
    }

    async fn discard(&mut self, lba: u64, blocks: u32) -> Result<(), ProviderError> {
        self.forget_held();
        let mut done = 0u32;
        while done < blocks {
            let count = self.streaming_blocks(lba + u64::from(done), blocks - done);
            let request = self
                .channel
                .submit_discard(lba + done as u64, count)
                .map_err(map_block)?;
            self.wait(request).await?;
            self.stats.discards = self.stats.discards.saturating_add(1);
            done += count;
        }
        Ok(())
    }

    async fn zero(&mut self, lba: u64, blocks: u32) -> Result<(), ProviderError> {
        self.forget_held();
        let buffers = [self.io, self.rollback];
        for buffer in buffers {
            self.channel.fill_buffer(buffer, 0).map_err(map_block)?;
        }
        let mut done = 0u32;
        while done < blocks {
            let wave_start = done;
            let mut counts = [0u32; BULK_DEPTH];
            let mut planned = 0usize;
            let mut planned_done = done;
            for count in &mut counts {
                *count = self
                    .streaming_write_blocks(lba + u64::from(planned_done), blocks - planned_done);
                if *count == 0 {
                    break;
                }
                planned += 1;
                planned_done += *count;
            }
            let mut requests = [None; BULK_DEPTH];
            let mut accepted = 0usize;
            let mut stage_error = None;
            let mut at = wave_start;
            {
                let mut batch = self.channel.batch();
                for index in 0..planned {
                    match batch.stage_write(lba + u64::from(at), counts[index], buffers[index], 0) {
                        Ok(request) => {
                            requests[index] = Some(request);
                            accepted += 1;
                        }
                        Err(error) => {
                            stage_error = Some(error);
                            break;
                        }
                    }
                    at += counts[index];
                }
                batch.publish();
            }
            if accepted == 0 {
                return Err(map_block(stage_error.unwrap_or(BlockError::Backpressure)));
            }
            self.wait_many(&mut requests).await?;
            if let Some(error) = stage_error
                && error != BlockError::Backpressure
            {
                return Err(map_block(error));
            }
            done += counts[..accepted].iter().copied().sum::<u32>();
            self.stats.writes = self.stats.writes.saturating_add(accepted as u64);
            self.stats.bulk_batches = self.stats.bulk_batches.saturating_add(1);
        }
        Ok(())
    }

    async fn copy_extent(&mut self, from: u64, to: u64, blocks: u32) -> Result<(), ProviderError> {
        self.forget_held();
        let buffers = [self.io, self.rollback];
        let mut done = 0u32;
        while done < blocks {
            let wave_start = done;
            let mut counts = [0u32; BULK_DEPTH];
            let mut planned = 0usize;
            let mut planned_done = done;
            for count in &mut counts {
                *count = self
                    .streaming_blocks(from + u64::from(planned_done), blocks - planned_done)
                    .min(self.streaming_blocks(to + u64::from(planned_done), blocks - planned_done))
                    .min(self.streaming_write_blocks(
                        to + u64::from(planned_done),
                        blocks - planned_done,
                    ));
                if *count == 0 {
                    break;
                }
                planned += 1;
                planned_done += *count;
            }
            let mut reads = [None; BULK_DEPTH];
            let mut issued = 0usize;
            let mut read_stage_error = None;
            let mut at = wave_start;
            {
                let mut batch = self.channel.batch();
                for index in 0..planned {
                    match batch.stage_read(from + u64::from(at), counts[index], buffers[index], 0) {
                        Ok(request) => {
                            reads[index] = Some(request);
                            issued += 1;
                        }
                        Err(error) => {
                            read_stage_error = Some(error);
                            break;
                        }
                    }
                    at += counts[index];
                }
                batch.publish();
            }
            if issued == 0 {
                return Err(map_block(
                    read_stage_error.unwrap_or(BlockError::Backpressure),
                ));
            }
            self.wait_many(&mut reads).await?;
            if let Some(error) = read_stage_error
                && error != BlockError::Backpressure
            {
                return Err(map_block(error));
            }
            self.stats.reads = self.stats.reads.saturating_add(issued as u64);
            self.stats.bulk_batches = self.stats.bulk_batches.saturating_add(1);

            let mut written = 0usize;
            while written < issued {
                let mut writes = [None; BULK_DEPTH];
                let mut accepted = 0usize;
                let mut write_stage_error = None;
                let mut at = wave_start + counts[..written].iter().copied().sum::<u32>();
                {
                    let mut batch = self.channel.batch();
                    for index in written..issued {
                        let count = counts[index];
                        match batch.stage_write(to + u64::from(at), count, buffers[index], 0) {
                            Ok(request) => {
                                writes[index] = Some(request);
                                accepted += 1;
                            }
                            Err(error) => {
                                write_stage_error = Some(error);
                                break;
                            }
                        }
                        at += count;
                    }
                    batch.publish();
                }
                if accepted == 0 {
                    return Err(map_block(
                        write_stage_error.unwrap_or(BlockError::Backpressure),
                    ));
                }
                self.wait_many(&mut writes).await?;
                if let Some(error) = write_stage_error
                    && error != BlockError::Backpressure
                {
                    return Err(map_block(error));
                }
                written += accepted;
                self.stats.writes = self.stats.writes.saturating_add(accepted as u64);
                self.stats.bulk_batches = self.stats.bulk_batches.saturating_add(1);
            }
            done += counts[..issued].iter().copied().sum::<u32>();
        }
        Ok(())
    }

    async fn retire(&mut self, placement: Placement) -> Result<(), ProviderError> {
        self.forget_held();
        if placement.blocks == 0 {
            if placement.extent.is_some() {
                return Err(ProviderError::Integrity);
            }
            return Ok(());
        }
        if !placement.committed {
            if self.channel.geometry().discard {
                let _ = self.discard(placement.lba, placement.blocks).await;
            }
            self.extents
                .release(placement.extent.ok_or(ProviderError::Integrity)?)?;
            return Ok(());
        }
        self.retired
            .push(placement)
            // A full retired-extent table is a slot ceiling, not a broken
            // provider: `Backpressure` here reached the client as
            // `ProviderFailure`, telling it the storage backend had failed
            // when in fact a bound was reached.
            .map_err(|_| ProviderError::PlacementsFull)
    }

    async fn resize_placement(&mut self, key: BlockKey, len: u64) -> Result<(), ProviderError> {
        self.forget_held();
        let old = *self
            .placements
            .get(key.0)
            .ok_or(ProviderError::UnknownKey)?;
        let wanted = self.blocks_for(len)?;
        if len == old.len {
            return Ok(());
        }
        if wanted == old.blocks && len < old.len {
            self.placements
                .get_mut(key.0)
                .ok_or(ProviderError::UnknownKey)?
                .len = len;
            return Ok(());
        }
        if wanted == 0 {
            self.retire(old).await?;
            *self
                .placements
                .get_mut(key.0)
                .ok_or(ProviderError::UnknownKey)? = Placement {
                lba: 0,
                blocks: 0,
                len,
                committed: false,
                extent: None,
                written_blocks: 0,
            };
            return Ok(());
        }
        if old.blocks > 0 && self.retired.is_full() {
            return Err(ProviderError::Backpressure);
        }
        let (new_extent, new_lba) = self.extents.allocate(wanted)?;
        let replacement = Placement {
            lba: new_lba,
            blocks: wanted,
            len,
            committed: false,
            extent: Some(new_extent),
            // The copy below carries the old meaningful blocks; the rest of
            // the new extent is the zeros written first.
            written_blocks: old.written_blocks.min(wanted),
        };
        let prepared = async {
            self.zero(new_lba, wanted).await?;
            if old.blocks > 0 {
                self.copy_extent(old.lba, new_lba, old.blocks.min(wanted))
                    .await?;
            }
            if len > old.len {
                let block = self.block_size() as u64;
                let tail = ((old.len.div_ceil(block) * block).min(len) - old.len) as usize;
                if tail > 0 {
                    let zeros = [0u8; 4096];
                    self.write_new_bytes(replacement, old.len, &zeros[..tail])
                        .await?;
                }
            }
            Ok::<(), ProviderError>(())
        }
        .await;
        if let Err(error) = prepared {
            self.extents.release(new_extent)?;
            return Err(error);
        }
        if let Err(error) = self.retire(old).await {
            self.extents.release(new_extent)?;
            return Err(error);
        }
        let mut placement = self
            .placements
            .get_mut(key.0)
            .ok_or(ProviderError::UnknownKey)?;
        placement.lba = new_lba;
        placement.blocks = wanted;
        placement.len = len;
        placement.committed = false;
        placement.extent = Some(new_extent);
        Ok(())
    }

    async fn read_bytes(
        &mut self,
        placement: Placement,
        offset: u64,
        dst: &mut [u8],
    ) -> Result<usize, ProviderError> {
        offset
            .checked_add(dst.len() as u64)
            .ok_or(ProviderError::Range)?;
        if offset > placement.len {
            return Err(ProviderError::Range);
        }
        let want = dst.len().min((placement.len - offset) as usize);
        if want == 0 {
            return Ok(0);
        }
        let block = self.block_size();
        let first = offset / block as u64;
        let within = offset as usize % block;
        let covered = (within + want).div_ceil(block) as u32;
        let lba = placement.lba + first;
        // Served from the window `io` already holds, or fetched with room
        // to spare so the next sequential read is.  The window is bounded
        // by the placement (never a neighbour's blocks), by the buffer, and
        // by the device's streaming limits; a boundary that would split the
        // request itself falls back to exactly what was asked, as before.
        let at = match self.held {
            Some(held)
                if held.lba <= lba
                    && lba + u64::from(covered) <= held.lba + u64::from(held.blocks) =>
            {
                self.stats.readahead_hits = self.stats.readahead_hits.saturating_add(1);
                (lba - held.lba) as usize * block
            }
            _ => {
                self.held = None;
                let remaining = placement
                    .blocks
                    .saturating_sub(u32::try_from(first).unwrap_or(u32::MAX))
                    .max(covered);
                let window = u32::try_from(READ_AHEAD_BYTES / block)
                    .unwrap_or(u32::MAX)
                    .max(covered)
                    .min(remaining);
                let mut count = self.streaming_blocks(lba, window);
                if count < covered {
                    count = covered;
                }
                self.read_blocks(lba, count, self.io).await?;
                self.held = Some(HeldRead { lba, blocks: count });
                self.stats.readahead_reads = self.stats.readahead_reads.saturating_add(1);
                0
            }
        };
        self.channel
            .read_buffer(self.io, at + within, &mut dst[..want])
            .map_err(map_block)?;
        self.stats.copied_bytes = self.stats.copied_bytes.saturating_add(want as u64);
        Ok(want)
    }

    async fn write_new_bytes(
        &mut self,
        placement: Placement,
        offset: u64,
        src: &[u8],
    ) -> Result<(), ProviderError> {
        let end = offset
            .checked_add(src.len() as u64)
            .ok_or(ProviderError::Range)?;
        if end > placement.len {
            return Err(ProviderError::Range);
        }
        if src.is_empty() {
            return Ok(());
        }
        let block = self.block_size();
        let first = offset / block as u64;
        let within = offset as usize % block;
        let covered = (within + src.len()).div_ceil(block);

        // Only a partial edge block has old bytes to keep; the interior is
        // overwritten whole.  Edges land in `io` where they will be written
        // from, the payload patches over them in place, `io` goes back: no
        // stack image of the transfer, and an aligned write costs no read.
        self.forget_held();
        self.read_edges(placement.lba + first, within, src.len(), covered)
            .await?;
        self.channel
            .write_buffer(self.io, within, src)
            .map_err(map_block)?;
        if let Err(error) = self
            .write_blocks(placement.lba + first, covered as u32, self.io)
            .await
        {
            self.stats.rollbacks = self.stats.rollbacks.saturating_add(1);
            return Err(error);
        }
        self.stats.copied_bytes = self.stats.copied_bytes.saturating_add(src.len() as u64);
        Ok(())
    }

    /// Overwrite an extent no durable root can name yet.
    ///
    /// A deferred bulk writer publishes one COW replacement on its first
    /// mutation. Re-copying that whole unpublished extent for every later
    /// chunk turns an N-byte import into O(N^2) device traffic. The placement
    /// itself is already the rollback boundary here, so update it in place
    /// while retaining the touched blocks in the dedicated rollback buffer.
    /// An accepted write that fails is restored before the refusal returns;
    /// if restoration itself fails, the channel is lost rather than exposing
    /// bytes whose revision no longer describes them.
    async fn write_uncommitted_bytes(
        &mut self,
        placement: Placement,
        offset: u64,
        src: &[u8],
    ) -> Result<(), ProviderError> {
        debug_assert!(!placement.committed);
        let end = offset
            .checked_add(src.len() as u64)
            .ok_or(ProviderError::Range)?;
        if end > placement.len {
            return Err(ProviderError::Range);
        }
        if src.is_empty() {
            return Ok(());
        }
        let block = self.block_size();
        let first = offset / block as u64;
        let within = offset as usize % block;
        let covered = (within + src.len()).div_ceil(block);

        // The exact old blocks stay in `rollback` until the write's outcome
        // is known.  `io` needs old bytes only where the payload leaves
        // some — a partial head or tail block — and those are mirrored from
        // `rollback`: at most two blocks through the CPU, never the whole
        // transfer.  Past the placement's high-water mark the old bytes are
        // the allocation's zeros: nothing is read, the edges are zeroed in
        // place, and a failed write is restored with zeros.
        self.forget_held();
        let pristine = first >= u64::from(placement.written_blocks);
        if !pristine {
            self.read_blocks(placement.lba + first, covered as u32, self.rollback)
                .await?;
        }
        let head = within != 0;
        let tail = (within + src.len()) % block != 0;
        let last = covered - 1;
        let edges = [
            head.then_some(0),
            (tail && (last != 0 || !head)).then_some(last * block),
        ];
        let mut chunk = [0u8; COPY_CHUNK_BYTES];
        for at in edges.into_iter().flatten() {
            if pristine {
                chunk[..block].fill(0);
            } else {
                self.channel
                    .read_buffer(self.rollback, at, &mut chunk[..block])
                    .map_err(map_block)?;
            }
            self.channel
                .write_buffer(self.io, at, &chunk[..block])
                .map_err(map_block)?;
        }
        if !head && !tail {
            self.stats.aligned_writes = self.stats.aligned_writes.saturating_add(1);
        }
        if pristine {
            self.stats.pristine_writes = self.stats.pristine_writes.saturating_add(1);
        }
        self.channel
            .write_buffer(self.io, within, src)
            .map_err(map_block)?;

        // Split submission from completion so a pre-submit refusal needs no
        // restoration, while every accepted command retains the exact old
        // blocks in `rollback` until its completion is known.
        let request = self
            .channel
            .submit_write(placement.lba + first, covered as u32, self.io, 0)
            .map_err(map_block)?;
        if let Err(error) = self.wait(request).await {
            self.stats.rollbacks = self.stats.rollbacks.saturating_add(1);
            let restored = if pristine {
                self.channel
                    .fill_buffer(self.rollback, 0)
                    .map_err(map_block)
            } else {
                Ok(())
            };
            if restored.is_err()
                || self
                    .write_blocks(placement.lba + first, covered as u32, self.rollback)
                    .await
                    .is_err()
            {
                self.channel.mark_lost();
                return Err(ProviderError::Lost);
            }
            return Err(error);
        }
        self.stats.writes = self.stats.writes.saturating_add(1);
        self.stats.copied_bytes = self.stats.copied_bytes.saturating_add(src.len() as u64);
        Ok(())
    }

    async fn write_bytes(
        &mut self,
        key: BlockKey,
        old: Placement,
        offset: u64,
        src: &[u8],
    ) -> Result<(), ProviderError> {
        let end = offset
            .checked_add(src.len() as u64)
            .ok_or(ProviderError::Range)?;
        if end > old.len {
            return Err(ProviderError::Range);
        }
        if src.is_empty() {
            return Ok(());
        }
        if !old.committed {
            self.write_uncommitted_bytes(old, offset, src).await?;
            self.advance_written(key, end)?;
            return Ok(());
        }
        if self.retired.is_full() {
            return Err(ProviderError::Backpressure);
        }
        let (new_extent, new_lba) = self.extents.allocate(old.blocks)?;
        if let Err(error) = self.copy_extent(old.lba, new_lba, old.blocks).await {
            self.stats.rollbacks = self.stats.rollbacks.saturating_add(1);
            self.extents.release(new_extent)?;
            return Err(error);
        }
        let replacement = Placement {
            lba: new_lba,
            committed: false,
            extent: Some(new_extent),
            ..old
        };
        if let Err(error) = self.write_new_bytes(replacement, offset, src).await {
            self.extents.release(new_extent)?;
            return Err(error);
        }
        if let Err(error) = self.retire(old).await {
            self.extents.release(new_extent)?;
            return Err(error);
        }
        *self
            .placements
            .get_mut(key.0)
            .ok_or(ProviderError::UnknownKey)? = replacement;
        self.advance_written(key, end)
    }

    /// Record that bytes up to `end` are now meaningful.
    fn advance_written(&mut self, key: BlockKey, end: u64) -> Result<(), ProviderError> {
        let blocks = self.blocks_for(end)?;
        let mut placement = self
            .placements
            .get_mut(key.0)
            .ok_or(ProviderError::UnknownKey)?;
        placement.written_blocks = placement.written_blocks.max(blocks);
        Ok(())
    }
}

impl<D: DirectBlockDevice, const B: usize, const O: usize> StorageProvider
    for BlockProvider<D, B, O>
{
    type Key = BlockKey;
    const PERSISTS_ACROSS_RESTART: bool = true;

    fn limits(&self) -> StorageLimits {
        self.limits.get()
    }

    async fn allocate(&self, _kind: ObjectKind, len: u64) -> Result<BlockKey, ProviderError> {
        if len > self.limits.get().max_object_bytes {
            return Err(ProviderError::TooLarge);
        }
        let mut state = self.state.borrow_mut();
        let blocks = state.blocks_for(len)?;
        let (allocation, lba) = if blocks == 0 {
            (None, 0)
        } else {
            let (extent, lba) = state.extents.allocate(blocks)?;
            (Some(extent), lba)
        };
        if let Some(extent) = allocation {
            if let Err(error) = state.zero(lba, blocks).await {
                state.extents.release(extent)?;
                return Err(error);
            }
        }
        let inserted = state
            .placements
            .insert(Placement {
                lba,
                blocks,
                len,
                committed: false,
                extent: allocation,
                written_blocks: 0,
            })
            .map(BlockKey)
            // The table, not the medium: the extent index above already proved the
            // bytes fit, so what ran out is the extent-tracking ceiling.
            .map_err(|_| ProviderError::PlacementsFull);
        if inserted.is_err()
            && let Some(extent) = allocation
        {
            state.extents.release(extent)?;
        }
        inserted
    }

    async fn read(
        &self,
        key: BlockKey,
        offset: u64,
        dst: &mut [u8],
    ) -> Result<usize, ProviderError> {
        if dst.len() > self.limits.get().max_transfer {
            return Err(ProviderError::TooLarge);
        }
        let mut state = self.state.borrow_mut();
        let placement = *state
            .placements
            .get(key.0)
            .ok_or(ProviderError::UnknownKey)?;
        state.read_bytes(placement, offset, dst).await
    }

    /// DMA to DMA: whole extents move between the two registered buffers
    /// in the same two-deep waves the durable COW copy uses, and no byte
    /// visits the CPU.  Block-granular by construction — both placements
    /// begin at offset zero — so the tail block beyond `len` carries the
    /// source's own tail, which nothing below `len` can observe.
    async fn copy_object(
        &self,
        source: BlockKey,
        destination: BlockKey,
        len: u64,
    ) -> Result<(), ProviderError> {
        let mut state = self.state.borrow_mut();
        let from = *state
            .placements
            .get(source.0)
            .ok_or(ProviderError::UnknownKey)?;
        let to = *state
            .placements
            .get(destination.0)
            .ok_or(ProviderError::UnknownKey)?;
        if len > from.len || len > to.len {
            return Err(ProviderError::Range);
        }
        if to.committed {
            return Err(ProviderError::Integrity);
        }
        let blocks = state.blocks_for(len)?;
        if blocks == 0 {
            return Ok(());
        }
        if blocks > from.blocks || blocks > to.blocks {
            return Err(ProviderError::Integrity);
        }
        state.copy_extent(from.lba, to.lba, blocks).await?;
        state.stats.snapshot_copies = state.stats.snapshot_copies.saturating_add(1);
        let mut placement = state
            .placements
            .get_mut(destination.0)
            .ok_or(ProviderError::UnknownKey)?;
        placement.written_blocks = placement.written_blocks.max(blocks);
        Ok(())
    }

    async fn write(&self, key: BlockKey, offset: u64, src: &[u8]) -> Result<(), ProviderError> {
        if src.len() > self.limits.get().max_transfer {
            return Err(ProviderError::TooLarge);
        }
        let mut state = self.state.borrow_mut();
        let placement = *state
            .placements
            .get(key.0)
            .ok_or(ProviderError::UnknownKey)?;
        state.write_bytes(key, placement, offset, src).await
    }

    async fn resize(&self, key: BlockKey, len: u64) -> Result<(), ProviderError> {
        if len > self.limits.get().max_object_bytes {
            return Err(ProviderError::TooLarge);
        }
        self.state.borrow_mut().resize_placement(key, len).await
    }

    async fn release(&self, key: BlockKey) -> Result<(), ProviderError> {
        let mut state = self.state.borrow_mut();
        let placement = *state
            .placements
            .get(key.0)
            .ok_or(ProviderError::UnknownKey)?;
        if placement.blocks > 0 && state.retired.is_full() {
            return Err(ProviderError::Backpressure);
        }
        let placement = state
            .placements
            .remove(key.0)
            .ok_or(ProviderError::UnknownKey)?;
        state.retire(placement).await?;
        Ok(())
    }

    async fn sync(&self, key: BlockKey) -> Result<(), ProviderError> {
        let mut state = self.state.borrow_mut();
        if state.placements.get(key.0).is_none() {
            return Err(ProviderError::UnknownKey);
        }
        state.flush().await
    }
}

fn map_block(error: BlockError) -> ProviderError {
    match error {
        BlockError::Geometry | BlockError::Integrity | BlockError::StaleCompletion => {
            ProviderError::Integrity
        }
        BlockError::Range
        | BlockError::Alignment
        | BlockError::UnknownBuffer
        | BlockError::UnknownRequest => ProviderError::Range,
        BlockError::TooLarge => ProviderError::TooLarge,
        BlockError::ReadOnly => ProviderError::ReadOnly,
        BlockError::Backpressure | BlockError::BufferBusy => ProviderError::Backpressure,
        BlockError::Cancelled => ProviderError::Cancelled,
        BlockError::Revoked => ProviderError::Revoked,
        BlockError::Draining => ProviderError::Draining,
        BlockError::Lost => ProviderError::Lost,
        BlockError::Media => ProviderError::Media,
        BlockError::Transport => ProviderError::Transport,
        BlockError::Unsupported => ProviderError::Integrity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kore_driver::block::{BlockChannel, CompletionMode, memory::MemoryBlockDevice};

    // Keep fixtures compact: provider engines are intentionally heapless and
    // therefore live on the test thread's stack.
    type Device = MemoryBlockDevice<49152, 4096>;
    type Provider = BlockProvider<Device, 3, 8>;

    fn provider() -> Provider {
        let channel =
            BlockChannel::new(Device::new(512, 8).unwrap(), CompletionMode::Poll).unwrap();
        Provider::new(channel).unwrap()
    }

    /// A wait that expired against a HEALTHY controller must not retire the
    /// channel. `mark_lost` is a one-way door — it takes the mounted volume
    /// with it — and a late completion is equally explained by a deep queue,
    /// GC, thermal throttling, or the host descheduling the waiter. Only the
    /// device's own verdict justifies the door.
    /// Sequential payload reads are served from one read-ahead window: the
    /// first read fetches as much of the placement as the buffer holds, the
    /// rest cost no command, and any write to the placement forgets the
    /// window so the next read fetches fresh bytes.
    #[test]
    fn sequential_reads_ride_one_read_ahead_window_until_a_write() {
        let provider = provider();
        // 8 blocks; the test device's 4 KiB buffer makes the window 8 blocks.
        let key = crate::block_on(provider.allocate(ObjectKind::File, 4096)).unwrap();
        for chunk in 0..8u64 {
            let byte = 0x10 + chunk as u8;
            crate::block_on(provider.write(key, chunk * 512, &[byte; 512])).unwrap();
        }
        let before = provider.stats();
        let mut dst = [0u8; 512];
        for chunk in 0..8u64 {
            let got = crate::block_on(provider.read(key, chunk * 512, &mut dst)).unwrap();
            assert_eq!(got, 512);
            assert!(
                dst.iter().all(|b| *b == 0x10 + chunk as u8),
                "chunk {chunk} bytes"
            );
        }
        let after = provider.stats();
        assert_eq!(
            after.readahead_reads - before.readahead_reads,
            1,
            "one window fetch"
        );
        assert_eq!(
            after.readahead_hits - before.readahead_hits,
            7,
            "seven served from it"
        );
        assert_eq!(
            after.reads - before.reads,
            1,
            "one device read for eight requests"
        );

        // A write forgets the window; the next read fetches again and sees
        // the new bytes, never a stale mirror.
        crate::block_on(provider.write(key, 3 * 512, &[0xEE; 512])).unwrap();
        let mid = provider.stats();
        let got = crate::block_on(provider.read(key, 3 * 512, &mut dst)).unwrap();
        assert_eq!(got, 512);
        assert!(dst.iter().all(|b| *b == 0xEE));
        let end = provider.stats();
        assert_eq!(end.readahead_reads - mid.readahead_reads, 1);
        assert_eq!(end.readahead_hits - mid.readahead_hits, 0);
        // And a read past the window's end fetches a new one, unaligned
        // requests included.
        let mut tail = [0u8; 300];
        let got = crate::block_on(provider.read(key, 4096 - 300, &mut tail)).unwrap();
        assert_eq!(got, 300);
        assert!(tail.iter().all(|b| *b == 0x17));
    }

    /// A write that begins and ends on block boundaries has no old bytes to
    /// keep: into an uncommitted extent it costs the rollback read and the
    /// write, into a fresh extent just the write.  A partial edge costs one
    /// block, never the span — and the bytes around every edge survive.
    #[test]
    fn aligned_writes_skip_the_pre_read_and_edges_cost_one_block_each() {
        let provider = provider();
        let key = crate::block_on(provider.allocate(ObjectKind::File, 4096)).unwrap();
        let mut expected = [0u8; 4096];
        let check = |provider: &Provider, expected: &[u8; 4096]| {
            // The test buffer admits ~3.5 KiB per transfer; read in quarters.
            let mut back = [0u8; 4096];
            for piece in 0..4 {
                let at = piece * 1024;
                assert_eq!(
                    crate::block_on(provider.read(key, at as u64, &mut back[at..at + 1024])),
                    Ok(1024)
                );
            }
            assert_eq!(&back[..], &expected[..]);
        };

        // Uncommitted, aligned, past the high-water mark: one write, no
        // read of any kind.
        let before = provider.stats();
        crate::block_on(provider.write(key, 0, &[1; 2048])).unwrap();
        expected[..2048].fill(1);
        let after = provider.stats();
        assert_eq!(after.reads, before.reads);
        assert_eq!(after.writes - before.writes, 1);
        assert_eq!(after.aligned_writes - before.aligned_writes, 1);
        assert_eq!(after.pristine_writes - before.pristine_writes, 1);
        assert_eq!(after.edge_reads, 0);
        check(&provider, &expected);

        // Uncommitted, both edges partial, below the mark: one rollback
        // read — the edges are mirrored from that image, not fetched again.
        let before = provider.stats();
        crate::block_on(provider.write(key, 100, &[2; 1200])).unwrap();
        expected[100..1300].fill(2);
        let after = provider.stats();
        assert_eq!(after.reads - before.reads, 1);
        assert_eq!(after.aligned_writes, before.aligned_writes);
        assert_eq!(after.edge_reads, 0);
        check(&provider, &expected);

        // Committed, aligned: the COW copy reads once, the write itself
        // reads nothing.
        crate::block_on(provider.checkpoint());
        let before = provider.stats();
        crate::block_on(provider.write(key, 512, &[3; 1024])).unwrap();
        expected[512..1536].fill(3);
        let after = provider.stats();
        assert_eq!(after.reads - before.reads, 1, "the extent copy only");
        assert_eq!(after.aligned_writes - before.aligned_writes, 1);
        assert_eq!(after.edge_reads, 0);
        check(&provider, &expected);

        // Committed, two distinct partial edges: the copy plus one block
        // per edge, fetched together.
        crate::block_on(provider.checkpoint());
        let before = provider.stats();
        crate::block_on(provider.write(key, 700, &[4; 1500])).unwrap();
        expected[700..2200].fill(4);
        let after = provider.stats();
        assert_eq!(after.reads - before.reads, 3);
        assert_eq!(after.edge_reads - before.edge_reads, 2);
        check(&provider, &expected);

        // Committed, one block that is both edges.
        crate::block_on(provider.checkpoint());
        let before = provider.stats();
        crate::block_on(provider.write(key, 3000, &[5; 40])).unwrap();
        expected[3000..3040].fill(5);
        let after = provider.stats();
        assert_eq!(after.reads - before.reads, 2);
        assert_eq!(after.edge_reads - before.edge_reads, 1);
        check(&provider, &expected);
    }

    /// A sequential fill of a fresh extent — the store seed — never reads:
    /// each chunk lands past the high-water mark, where the extent holds the
    /// zeros allocation wrote.  A rewrite below the mark reads its rollback
    /// image again, and a failed pristine write restores zeros.
    #[test]
    fn writes_past_the_high_water_mark_read_nothing_and_roll_back_to_zeros() {
        let provider = provider();
        let key = crate::block_on(provider.allocate(ObjectKind::File, 4096)).unwrap();
        let before = provider.stats();
        for chunk in 0..4u64 {
            crate::block_on(provider.write(key, chunk * 1024, &[0x30 + chunk as u8; 1024]))
                .unwrap();
        }
        let after = provider.stats();
        assert_eq!(after.reads, before.reads, "a seed reads nothing");
        assert_eq!(after.writes - before.writes, 4);
        assert_eq!(after.pristine_writes - before.pristine_writes, 4);

        // Below the mark: the old bytes matter again.
        let before = provider.stats();
        crate::block_on(provider.write(key, 1500, &[0xAA; 100])).unwrap();
        let after = provider.stats();
        assert_eq!(after.reads - before.reads, 1);
        assert_eq!(after.pristine_writes, before.pristine_writes);
        let mut back = [0u8; 1024];
        assert_eq!(
            crate::block_on(provider.read(key, 1024, &mut back)),
            Ok(1024)
        );
        assert!(back[..476].iter().all(|b| *b == 0x31));
        assert!(back[476..576].iter().all(|b| *b == 0xAA));
        assert!(back[576..].iter().all(|b| *b == 0x31));

        // A fresh placement whose first, pristine write fails is zeros
        // again afterwards, not a torn image.
        let key = crate::block_on(provider.allocate(ObjectKind::File, 2048)).unwrap();
        provider
            .control(|device| device.inject_after(0, BlockError::Media))
            .unwrap();
        assert_eq!(
            crate::block_on(provider.write(key, 0, &[7; 1024])),
            Err(ProviderError::Media)
        );
        let mut back = [0u8; 1024];
        assert_eq!(crate::block_on(provider.read(key, 0, &mut back)), Ok(1024));
        assert_eq!(back, [0u8; 1024]);
        assert_eq!(provider.stats().rollbacks, 1);
    }

    /// A version lend's snapshot copies whole extents between the DMA
    /// buffers: one read and one write per wave, whatever the transfer
    /// limit, and the destination reads back byte-exact below `len`.
    #[test]
    fn snapshot_copies_move_whole_extents_between_dma_buffers() {
        let provider = provider();
        let source = crate::block_on(provider.allocate(ObjectKind::File, 3000)).unwrap();
        let pattern: [u8; 3000] = core::array::from_fn(|i| (i * 7 % 251) as u8);
        for (index, piece) in pattern.chunks(1000).enumerate() {
            crate::block_on(provider.write(source, (index * 1000) as u64, piece)).unwrap();
        }
        let destination = crate::block_on(provider.allocate(ObjectKind::File, 3000)).unwrap();
        let before = provider.stats();
        crate::block_on(provider.copy_object(source, destination, 3000)).unwrap();
        let after = provider.stats();
        // Six blocks fit one wave of the 4 KiB test buffer: one read, one
        // write, no payload bytes through the CPU.
        assert_eq!(after.reads - before.reads, 1);
        assert_eq!(after.writes - before.writes, 1);
        assert_eq!(after.snapshot_copies - before.snapshot_copies, 1);
        assert_eq!(after.copied_bytes, before.copied_bytes);
        let mut back = [0u8; 3000];
        for (index, piece) in back.chunks_mut(1000).enumerate() {
            assert_eq!(
                crate::block_on(provider.read(destination, (index * 1000) as u64, piece)),
                Ok(1000)
            );
        }
        assert_eq!(back, pattern);
        // A later write to the copy sits below its high-water mark and keeps
        // the copied bytes around it.
        crate::block_on(provider.write(destination, 100, &[0xEE; 50])).unwrap();
        let mut head = [0u8; 200];
        assert_eq!(
            crate::block_on(provider.read(destination, 0, &mut head)),
            Ok(200)
        );
        assert_eq!(&head[..100], &pattern[..100]);
        assert!(head[100..150].iter().all(|b| *b == 0xEE));
        assert_eq!(&head[150..], &pattern[150..200]);
        // Refusals: a committed destination, a length past either object.
        crate::block_on(provider.checkpoint());
        assert_eq!(
            crate::block_on(provider.copy_object(source, destination, 3000)),
            Err(ProviderError::Integrity)
        );
        let fresh = crate::block_on(provider.allocate(ObjectKind::File, 1000)).unwrap();
        assert_eq!(
            crate::block_on(provider.copy_object(source, fresh, 3000)),
            Err(ProviderError::Range)
        );
    }

    /// The provider's transfer limit is its own, not the engine's 4 KiB:
    /// bounded by the DMA buffer, so a device with a large buffer admits a
    /// large single command and a small one refuses cleanly above its own.
    #[test]
    fn transfer_limit_follows_the_buffer_not_the_engine_bound() {
        let provider = provider();
        let limit = provider.limits().max_transfer;
        assert!(
            limit > crate::bounds::MAX_TRANSFER / 2,
            "a 4 KiB buffer admits ~3.5 KiB"
        );
        assert!(limit <= BLOCK_MAX_TRANSFER);
        let key = crate::block_on(provider.allocate(ObjectKind::File, 4096)).unwrap();
        let mut dst = [0u8; BLOCK_MAX_TRANSFER];
        assert_eq!(
            crate::block_on(provider.read(key, 0, &mut dst[..limit + 1])),
            Err(ProviderError::TooLarge)
        );
        assert!(crate::block_on(provider.read(key, 0, &mut dst[..limit])).is_ok());
    }

    #[test]
    fn a_slow_but_healthy_device_is_not_declared_lost() {
        let provider = provider();
        // Allocate while completions still flow — `allocate` zeroes the
        // extent, which is itself a command that must complete.
        let key = crate::block_on(provider.allocate(ObjectKind::File, 512)).unwrap();
        // Write once so the next write sits below the high-water mark and
        // begins with its rollback read: the command whose wait expires is
        // then a read, which leaves nothing to restore.
        crate::block_on(provider.write(key, 0, &[6u8; 512])).unwrap();
        {
            let mut state = provider.state.borrow_mut();
            state
                .channel
                .control(|d| d.set_liveness(kore_driver::block::Liveness::Healthy))
                .unwrap();
            // Never deliver the completion: the wait must run its budget out.
            state.channel.control(|d| d.hold_completions(true)).unwrap();
        }
        let outcome = crate::block_on(provider.write(key, 0, &[7u8; 512]));
        assert_eq!(
            outcome,
            Err(ProviderError::Timeout),
            "a healthy controller means slow, not dead"
        );
        let stats = provider.stats();
        assert_eq!(stats.expired_waits, 1);
        assert_eq!(stats.slow_waits, 1);
        // The door stayed shut: the channel is still usable.
        {
            let state = provider.state.borrow();
            assert_ne!(
                state.channel.state(),
                kore_driver::block::ChannelState::Lost
            );
        }
    }

    /// The controller's own fatal verdict IS evidence, and retires the
    /// channel — the behaviour a timeout alone used to trigger.
    #[test]
    fn a_device_reporting_failure_is_declared_lost() {
        let provider = provider();
        let key = crate::block_on(provider.allocate(ObjectKind::File, 512)).unwrap();
        {
            let mut state = provider.state.borrow_mut();
            state
                .channel
                .control(|d| d.set_liveness(kore_driver::block::Liveness::Failed))
                .unwrap();
            state.channel.control(|d| d.hold_completions(true)).unwrap();
        }
        assert_eq!(
            crate::block_on(provider.write(key, 0, &[7u8; 512])),
            Err(ProviderError::Lost)
        );
        assert_eq!(provider.stats().slow_waits, 0);
        {
            let state = provider.state.borrow();
            assert_eq!(
                state.channel.state(),
                kore_driver::block::ChannelState::Lost
            );
        }
    }

    /// A backend with no out-of-band signal has said nothing, so the
    /// conservative pre-existing posture is kept rather than inventing a
    /// verdict in either direction.
    #[test]
    fn a_backend_with_no_liveness_signal_keeps_the_conservative_posture() {
        let provider = provider();
        let key = crate::block_on(provider.allocate(ObjectKind::File, 512)).unwrap();
        {
            let mut state = provider.state.borrow_mut();
            // `Unknown` is the double's default — asserted so a change to it
            // cannot silently weaken this test.
            assert_eq!(
                state.channel.liveness(),
                kore_driver::block::Liveness::Unknown
            );
            state.channel.control(|d| d.hold_completions(true)).unwrap();
        }
        assert_eq!(
            crate::block_on(provider.write(key, 0, &[7u8; 512])),
            Err(ProviderError::Lost)
        );
    }

    #[test]
    fn explicit_retirement_releases_buffers_before_quiescing_the_device() {
        assert_eq!(crate::block_on(provider().retire()), Ok(()));
    }

    /// The two ways a provider can run out are **different facts**, and each
    /// has to say which one it is.
    ///
    /// A zero-length object takes a placement but no block, so this fills the
    /// extent table without spending a byte of the medium: the refusal must be
    /// `PlacementsFull` (raise the bound / reclaim retired extents), never
    /// `Exhausted` (free bytes) — the volume here is nearly empty. Conflating
    /// them sends whoever hit it looking in the wrong place, which on a
    /// machine with no serial console is most of the cost of the bug.
    #[test]
    fn a_full_extent_table_names_itself_not_the_medium() {
        let provider = provider();
        let mut placed = 0usize;
        let refusal = loop {
            match crate::block_on(provider.allocate(ObjectKind::File, 0)) {
                Ok(_) => {
                    placed += 1;
                    assert!(
                        placed <= MAX_BLOCK_PLACEMENTS,
                        "the extent table admitted more than its ceiling"
                    );
                }
                Err(error) => break error,
            }
        };
        assert_eq!(placed, MAX_BLOCK_PLACEMENTS);
        assert_eq!(refusal, ProviderError::PlacementsFull);
        // The medium was never the constraint: nothing beyond the metadata
        // prefix was ever occupied.
        assert_eq!(provider.occupied_blocks(), Some(LEGACY_DATA_START_LBA));
    }

    #[test]
    fn a_configured_arena_is_the_one_metadata_and_payload_boundary() {
        let provider = provider();
        provider.configure_arena(80, 4).unwrap();
        assert_eq!(provider.arena(), (80, 4));
        assert_eq!(provider.limits().total_bytes, 4 * 512);
        assert_eq!(provider.occupied_blocks(), Some(80));

        let key = crate::block_on(provider.allocate(ObjectKind::File, 4 * 512)).unwrap();
        assert_eq!(provider.extent(key).unwrap().lba, 80);
        assert_eq!(provider.occupied_blocks(), Some(84));
        assert_eq!(
            crate::block_on(provider.allocate(ObjectKind::File, 1)),
            Err(ProviderError::Exhausted)
        );

        let block = [0u8; 512];
        assert_eq!(crate::block_on(provider.write_metadata(79, &block)), Ok(()));
        assert_eq!(
            crate::block_on(provider.write_metadata(80, &block)),
            Err(ProviderError::Range)
        );
        assert_eq!(
            provider.configure_arena(81, 3),
            Err(ProviderError::Integrity),
            "a live placement's address cannot be reinterpreted"
        );
    }

    #[test]
    fn recovery_sorts_once_and_protects_only_fallback_unique_extents() {
        let provider = provider();
        let live = [
            DurableExtent {
                lba: 80,
                blocks: 2,
                len: 1024,
            },
            DurableExtent {
                lba: 75,
                blocks: 2,
                len: 1024,
            },
        ];
        let fallback = [
            live[0],
            DurableExtent {
                lba: 90,
                blocks: 2,
                len: 1024,
            },
        ];

        let keys = provider.recover_extents(&live, &fallback).unwrap();
        assert_eq!(provider.extent(keys[0]).unwrap(), live[0]);
        assert_eq!(provider.extent(keys[1]).unwrap(), live[1]);
        assert_eq!(provider.placements(), 2);
        assert_eq!(provider.occupied_blocks(), Some(LEGACY_DATA_START_LBA + 6));

        let fresh = crate::block_on(provider.allocate(ObjectKind::File, 2 * 512)).unwrap();
        let fresh = provider.extent(fresh).unwrap();
        assert_eq!(fresh.lba, 82);
        assert!(
            live.iter()
                .chain(&fallback[1..])
                .all(|extent| fresh.lba + u64::from(fresh.blocks) <= extent.lba
                    || fresh.lba >= extent.lba + u64::from(extent.blocks))
        );
    }

    #[test]
    fn failed_recovery_rebuild_leaves_the_provider_empty() {
        let provider = provider();
        let overlapping = [
            DurableExtent {
                lba: 75,
                blocks: 4,
                len: 2048,
            },
            DurableExtent {
                lba: 77,
                blocks: 2,
                len: 1024,
            },
        ];
        assert_eq!(
            provider.recover_extents(&overlapping, &[]),
            Err(ProviderError::Integrity)
        );
        assert_eq!(provider.placements(), 0);
        assert_eq!(provider.occupied_blocks(), Some(LEGACY_DATA_START_LBA));

        let valid = [DurableExtent {
            lba: 75,
            blocks: 2,
            len: 1024,
        }];
        let keys = provider.recover_extents(&valid, &[]).unwrap();
        assert_eq!(provider.extent(keys[0]).unwrap(), valid[0]);
    }

    /// The medium's own ceiling keeps reporting itself as the medium's — this
    /// is the tier the split above must not have absorbed.
    #[test]
    fn a_request_beyond_the_medium_still_names_the_medium() {
        let provider = provider();
        let beyond = provider.limits().total_bytes + 1;
        assert_eq!(
            crate::block_on(provider.allocate(ObjectKind::File, beyond)),
            Err(ProviderError::TooLarge)
        );
        assert_eq!(
            crate::FsError::from(ProviderError::TooLarge),
            crate::FsError::OutOfStorage
        );
    }

    #[test]
    fn unaligned_io_resize_release_flush_and_stale_keys() {
        let provider = provider();
        let key = crate::block_on(provider.allocate(ObjectKind::File, 1500)).unwrap();
        crate::block_on(provider.write(key, 17, &[0x5a; 1000])).unwrap();
        let mut back = [0u8; 1000];
        assert_eq!(crate::block_on(provider.read(key, 17, &mut back)), Ok(1000));
        assert_eq!(back, [0x5a; 1000]);
        crate::block_on(provider.resize(key, 6000)).unwrap();
        let mut grown = [0xff; 512];
        assert_eq!(
            crate::block_on(provider.read(key, 5000, &mut grown)),
            Ok(512)
        );
        assert_eq!(grown, [0; 512]);
        crate::block_on(provider.sync(key)).unwrap();
        assert!(provider.stats().flushes > 0);
        crate::block_on(provider.release(key)).unwrap();
        assert_eq!(
            crate::block_on(provider.read(key, 0, &mut back)),
            Err(ProviderError::UnknownKey)
        );
    }

    #[test]
    fn allocation_never_reuses_extents_a_durable_root_still_references() {
        let provider = provider();
        assert_eq!(provider.occupied_blocks(), Some(LEGACY_DATA_START_LBA));
        let key = crate::block_on(provider.allocate(ObjectKind::File, 1024)).unwrap();
        let old = provider.extent(key).unwrap();
        assert_eq!(provider.occupied_blocks(), Some(LEGACY_DATA_START_LBA + 2));
        // The checkpoint makes `old` durable; the grow below must then retire
        // it (the on-disk root still cites those blocks) rather than free it.
        crate::block_on(provider.checkpoint());
        crate::block_on(provider.resize(key, 2048)).unwrap();
        assert_ne!(provider.extent(key).unwrap().lba, old.lba);
        // Four live blocks plus the two-block extent still named by the last
        // durable root: both consume the mounted volume's physical capacity.
        assert_eq!(provider.occupied_blocks(), Some(LEGACY_DATA_START_LBA + 6));
        let old_end = old.lba + u64::from(old.blocks);
        // While `old` waits in `retired`, a fresh allocation must not land on
        // it: a torn commit would recover the durable root over new bytes.
        let fresh = crate::block_on(provider.allocate(ObjectKind::File, 1024)).unwrap();
        let fresh_extent = provider.extent(fresh).unwrap();
        let fresh_end = fresh_extent.lba + u64::from(fresh_extent.blocks);
        assert!(fresh_end <= old.lba || fresh_extent.lba >= old_end);
        // The next checkpoint promotes `old` to the fallback root's
        // protection; it stays occupied until a later checkpoint discards it.
        crate::block_on(provider.checkpoint());
        let next = crate::block_on(provider.allocate(ObjectKind::File, 1024)).unwrap();
        let next_extent = provider.extent(next).unwrap();
        let next_end = next_extent.lba + u64::from(next_extent.blocks);
        assert!(next_end <= old.lba || next_extent.lba >= old_end);
    }

    #[test]
    fn failed_write_rolls_back_visible_bytes() {
        let provider = provider();
        let key = crate::block_on(provider.allocate(ObjectKind::File, 2048)).unwrap();
        let original_extent = provider.extent(key).unwrap();
        crate::block_on(provider.write(key, 0, &[1; 2048])).unwrap();
        assert_eq!(provider.extent(key).unwrap().lba, original_extent.lba);
        provider
            .control(|device| device.inject_after(1, BlockError::Media))
            .unwrap();
        assert_eq!(
            crate::block_on(provider.write(key, 100, &[2; 1200])),
            Err(ProviderError::Media)
        );
        let mut back = [0; 2048];
        assert_eq!(crate::block_on(provider.read(key, 0, &mut back)), Ok(2048));
        assert_eq!(back, [1; 2048]);
        assert_eq!(provider.stats().rollbacks, 1);
    }

    #[test]
    fn deferred_bulk_writes_reuse_the_uncommitted_extent() {
        let provider = provider();
        let key = crate::block_on(provider.allocate(ObjectKind::File, 4096)).unwrap();
        let original = provider.extent(key).unwrap();

        for offset in (0..4096).step_by(1024) {
            crate::block_on(provider.write(key, offset, &[offset as u8; 1024])).unwrap();
            assert_eq!(provider.extent(key).unwrap().lba, original.lba);
        }

        // A checkpoint makes the current extent durable. The next write must
        // return to ordinary COW so the previous root remains recoverable.
        crate::block_on(provider.checkpoint());
        crate::block_on(provider.write(key, 0, &[0xa5; 1024])).unwrap();
        assert_ne!(provider.extent(key).unwrap().lba, original.lba);
    }

    #[test]
    fn a_large_deferred_import_stays_linear_and_exact() {
        std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(|| {
                // Match the current Lua archive member closely enough to pin
                // the workload that first exposed quadratic incremental COW.
                const IMPORT_BYTES: usize = 493_512;
                type ImportDevice = MemoryBlockDevice<1_048_576, 262_144>;
                type ImportChannel = BlockChannel<ImportDevice, 3, 8>;
                let channel =
                    ImportChannel::new(ImportDevice::new(512, 512).unwrap(), CompletionMode::Poll)
                        .unwrap();
                let provider = BlockProvider::new(channel).unwrap();
                let key = crate::block_on(provider.allocate(ObjectKind::File, IMPORT_BYTES as u64))
                    .unwrap();
                let original = provider.extent(key).unwrap();
                let baseline = provider.stats();
                let chunks = IMPORT_BYTES.div_ceil(crate::bounds::MAX_TRANSFER);
                let payload_byte = |global: usize| {
                    (global.wrapping_mul(31) ^ (global >> 8) ^ (global >> 16)) as u8
                };

                for offset in (0..IMPORT_BYTES).step_by(crate::bounds::MAX_TRANSFER) {
                    let count = (IMPORT_BYTES - offset).min(crate::bounds::MAX_TRANSFER);
                    let mut bytes = [0u8; crate::bounds::MAX_TRANSFER];
                    for (index, byte) in bytes[..count].iter_mut().enumerate() {
                        *byte = payload_byte(offset + index);
                    }
                    crate::block_on(provider.write(key, offset as u64, &bytes[..count])).unwrap();
                }

                let after_write = provider.stats();
                assert_eq!(provider.extent(key).unwrap().lba, original.lba);
                assert_eq!(after_write.bulk_batches, baseline.bulk_batches);
                // Every chunk lands past the high-water mark: the import
                // reads nothing at all.
                assert_eq!(after_write.reads, baseline.reads);
                assert_eq!(
                    after_write.pristine_writes - baseline.pristine_writes,
                    chunks as u64
                );
                assert_eq!(after_write.writes - baseline.writes, chunks as u64);

                for offset in (0..IMPORT_BYTES).step_by(crate::bounds::MAX_TRANSFER) {
                    let count = (IMPORT_BYTES - offset).min(crate::bounds::MAX_TRANSFER);
                    let mut bytes = [0u8; crate::bounds::MAX_TRANSFER];
                    assert_eq!(
                        crate::block_on(provider.read(key, offset as u64, &mut bytes[..count])),
                        Ok(count)
                    );
                    for (index, byte) in bytes[..count].iter().copied().enumerate() {
                        assert_eq!(byte, payload_byte(offset + index));
                    }
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn two_providers_own_independent_channels_without_a_shared_lock() {
        let (a, b) = (provider(), provider());
        let ka = crate::block_on(a.allocate(ObjectKind::File, 512)).unwrap();
        let kb = crate::block_on(b.allocate(ObjectKind::File, 512)).unwrap();
        crate::block_on(a.write(ka, 0, &[0xaa; 512])).unwrap();
        crate::block_on(b.write(kb, 0, &[0xbb; 512])).unwrap();
        let mut left = [0; 512];
        let mut right = [0; 512];
        crate::block_on(a.read(ka, 0, &mut left)).unwrap();
        crate::block_on(b.read(kb, 0, &mut right)).unwrap();
        assert_eq!(left, [0xaa; 512]);
        assert_eq!(right, [0xbb; 512]);
    }

    #[test]
    fn provider_retains_a_bounded_multi_request_direct_queue_and_reports_it() {
        let provider = provider();
        {
            let mut state = provider.state.borrow_mut();
            let a = state.channel.submit_flush().unwrap();
            let b = state.channel.submit_flush().unwrap();
            let c = state.channel.submit_flush().unwrap();
            assert_eq!(state.channel.outstanding(), 3);
            assert_eq!(state.channel.poll().unwrap().request, a);
            assert_eq!(state.channel.poll().unwrap().request, b);
            assert_eq!(state.channel.poll().unwrap().request, c);
        }
        let stats = provider.channel_stats();
        assert_eq!(stats.max_outstanding, 3);
        assert_eq!(stats.submissions, 3);
        assert_eq!(stats.completions, 3);
    }

    #[test]
    fn channel_loss_degrades_deterministically_without_wedging() {
        let provider = provider();
        let key = crate::block_on(provider.allocate(ObjectKind::File, 512)).unwrap();
        provider.mark_lost();
        let mut bytes = [0; 16];
        assert_eq!(
            crate::block_on(provider.read(key, 0, &mut bytes)),
            Err(ProviderError::Lost)
        );
        assert_eq!(
            crate::block_on(provider.sync(key)),
            Err(ProviderError::Lost)
        );
    }

    #[test]
    fn advertised_mdts_is_clamped_to_the_registered_window() {
        type WideDevice = MemoryBlockDevice<65_536, 8192>;
        type WideChannel = BlockChannel<WideDevice, 2, 8>;
        let channel =
            WideChannel::new(WideDevice::new(512, 128).unwrap(), CompletionMode::Poll).unwrap();
        let provider = BlockProvider::new(channel).unwrap();

        // One metadata arena is larger than the bounded registered window.
        // It must split at the window, not trust the device's larger MDTS.
        let written = [0x5au8; 16 * 1024];
        crate::block_on(provider.write_metadata(2, &written)).unwrap();
        let mut read = [0u8; 16 * 1024];
        crate::block_on(provider.read_metadata(2, &mut read)).unwrap();
        assert_eq!(read, written);
        assert!(provider.channel_stats().max_outstanding <= 1);
    }

    #[test]
    fn streaming_commands_stop_at_the_advertised_optimal_boundary() {
        let device = Device::new(512, 8)
            .unwrap()
            .with_io_hints(kore_driver::block::BlockIoHints {
                supported_block_sizes: 0,
                preferred_write_blocks: 4,
                write_alignment_blocks: 4,
                optimal_write_blocks: 4,
                optimal_boundary_blocks: 8,
            });
        let channel = BlockChannel::new(device, CompletionMode::Poll).unwrap();
        let provider = Provider::new(channel).unwrap();
        let state = provider.state.borrow();
        assert_eq!(state.streaming_blocks(6, 8), 2);
        assert_eq!(state.streaming_blocks(8, 8), 8);
        assert_eq!(state.streaming_write_blocks(6, 8), 2);
        assert_eq!(state.streaming_write_blocks(8, 8), 4);
    }

    #[test]
    fn bulk_extent_work_uses_two_buffers_and_bounded_queue_batches() {
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                type BulkDevice = MemoryBlockDevice<524_288, 32_768>;
                type BulkChannel = BlockChannel<BulkDevice, 3, 8>;
                let channel =
                    BulkChannel::new(BulkDevice::new(512, 64).unwrap(), CompletionMode::Poll)
                        .unwrap();
                let provider = BlockProvider::new(channel).unwrap();

                let key = crate::block_on(provider.allocate(ObjectKind::File, 64 * 1024)).unwrap();
                // Exercise the durable COW path. Unpublished placements are
                // deliberately updated in place by the deferred-import path.
                crate::block_on(provider.checkpoint());
                crate::block_on(provider.write(key, 17, &[0xa5; 4096])).unwrap();
                let mut back = [0u8; 4096];
                assert_eq!(crate::block_on(provider.read(key, 17, &mut back)), Ok(4096));
                assert_eq!(back, [0xa5; 4096]);

                let channel = provider.channel_stats();
                assert_eq!(channel.peak_registered_bytes, 2 * 32_768);
                assert_eq!(channel.max_outstanding, 2);
                assert!(provider.stats().bulk_batches >= 3);
                assert!(provider.control(|device| device.publishes()).unwrap() >= 3);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn minimum_outstanding_channel_falls_back_to_single_command_waves() {
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                type SerialDevice = MemoryBlockDevice<524_288, 32_768>;
                type SerialChannel = BlockChannel<SerialDevice, 2, 1>;
                let channel =
                    SerialChannel::new(SerialDevice::new(512, 64).unwrap(), CompletionMode::Poll)
                        .unwrap();
                let provider = BlockProvider::new(channel).unwrap();

                // Both zeroing and the durable COW copy plan two-command
                // waves. A depth-two device queue has only one usable slot,
                // so each second stage returns Backpressure and must become
                // the next wave rather than aborting the operation.
                let key = crate::block_on(provider.allocate(ObjectKind::File, 64 * 1024)).unwrap();
                crate::block_on(provider.checkpoint());
                crate::block_on(provider.write(key, 17, &[0xa5; 4096])).unwrap();
                let mut back = [0u8; 4096];
                assert_eq!(crate::block_on(provider.read(key, 17, &mut back)), Ok(4096));
                assert_eq!(back, [0xa5; 4096]);

                assert_eq!(provider.channel_stats().max_outstanding, 1);
                assert!(provider.stats().bulk_batches >= 6);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn one_block_device_still_accepts_bounded_unaligned_payloads() {
        type NarrowDevice = MemoryBlockDevice<65_536, 512>;
        type NarrowChannel = BlockChannel<NarrowDevice, 2, 4>;
        let channel =
            NarrowChannel::new(NarrowDevice::new(512, 1).unwrap(), CompletionMode::Poll).unwrap();
        let provider = BlockProvider::new(channel).unwrap();
        assert_eq!(provider.limits().max_transfer, 1);
        let key = crate::block_on(provider.allocate(ObjectKind::File, 512)).unwrap();
        crate::block_on(provider.write(key, 511, &[0xa5])).unwrap();
        let mut byte = [0u8; 1];
        assert_eq!(crate::block_on(provider.read(key, 511, &mut byte)), Ok(1));
        assert_eq!(byte, [0xa5]);
    }
}
