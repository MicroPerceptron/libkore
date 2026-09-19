//! The storage-provider seam.
//!
//! This is the boundary that makes the same filesystem engine work over RAM
//! over RAM and an NVMe-backed direct block channel without any change to its
//! namespace, authority, or client protocol. It describes
//! **storage semantics, not device commands**: there is no NVMe, SATA, or
//! virtio vocabulary here, and there never will be — a block-backed provider
//! acquires a hardware-neutral affine channel from a control service once,
//! then submits and completes I/O directly without branching on which device
//! is underneath or re-entering that service on the data path.
//!
//! # The boundaries that may not move
//!
//! - [`StorageProvider::Key`] **identifies placement only.** It grants no
//!   authority.  Holding one lets you name bytes, not reach them: every
//!   access check, name, link, and revision belongs to the engine above.
//!   This is why the key type reaches into [`Object`](crate::object) as a
//!   plain associated type rather than being an opaque capability.
//! - **The provider is async** so that a hardware or remote provider cannot
//!   block an executor worker.  The RAM provider never actually parks — its
//!   futures are `Ready` on first poll — but the seam is shaped for the one
//!   that will.
//! - **Providers report their limits and typed errors.**  The engine asks
//!   [`StorageProvider::limits`] rather than assuming a geometry, and
//!   [`ProviderError`] is storage-shaped: a provider cannot say "not found"
//!   about a name it has never seen.
//!
//! # `&self`, not `&mut self`
//!
//! Every method takes `&self`.  A provider whose transfers are genuinely in
//! flight — a direct block channel with queued I/O — must be able to hold
//! several at once, and an `&mut self` seam would forbid that and then have to
//! break when concurrent provider work arrives. [`RamProvider`] pays for this
//! with an internal
//! [`RefCell`](core::cell::RefCell): its borrows are short, taken and
//! dropped inside a single method with no `await` held across one.
//!
//! # What the engine promises the provider
//!
//! The engine reserves capacity with [`StorageProvider::resize`] *before* it
//! writes any bytes.  That is what makes a write request atomic: the only
//! failure that can run out of room happens before anything is visible, and
//! the subsequent [`StorageProvider::write`] into already-reserved space
//! cannot fail for want of space.  A provider must honour that: after a
//! successful `resize` to `n`, a write inside `0..n` fails only for a real
//! medium reason.

mod block;
mod extent;
mod ram;

pub use block::{
    BlockKey, BlockProvider, BlockProviderStats, DurableExtent, GPT_DISCOVERY_END_LBA,
    LEGACY_DATA_START_LBA,
};
pub use ram::{Fault, RamKey, RamProvider};

use crate::{error::ProviderError, object::ObjectKind};

/// What a provider can do, in storage terms.
///
/// The engine consults these instead of assuming a geometry, so a provider
/// with a different granularity or a smaller ceiling changes behaviour
/// without changing engine code.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StorageLimits {
    /// Total bytes the provider can place across all objects.
    pub total_bytes: u64,
    /// Largest single object the provider can hold.
    pub max_object_bytes: u64,
    /// Largest single transfer the provider will accept.
    pub max_transfer: usize,
    /// Allocation granularity in bytes.  Reserving `n` bytes may consume up
    /// to `n` rounded up to this.
    pub granularity: usize,
}

/// Where an object's bytes live, and how to move them.
///
/// A new provider implements this trait and nothing else changes — no engine
/// branch, no new error class, no client-visible difference
/// (`kore-design-doctrine` §3: contract-as-trait, data-as-impl).
#[allow(async_fn_in_trait)]
pub trait StorageProvider {
    /// A placement identifier.  Conveys no authority; see the module docs.
    ///
    /// `Debug` is part of the contract because a placement appears in the
    /// engine's diagnostics and in the assertions of the conformance body
    /// below — a provider key is always a small value type, so requiring it
    /// costs an impl nothing.
    type Key: Copy + Eq + core::fmt::Debug;

    /// Whether placements and their bytes can be reconstructed after this
    /// provider is restarted. Derived batches use the same placement seam as
    /// file payloads, but the RAM provider must not waste its bounded pool on
    /// a second copy that cannot survive a restart anyway.
    const PERSISTS_ACROSS_RESTART: bool;

    /// The provider's declared limits.
    fn limits(&self) -> StorageLimits;

    /// Reserve a placement of `len` bytes, reading as zeros.
    ///
    /// `kind` is informational — a provider may place a directory's metadata
    /// differently from a file's bytes.  The RAM provider treats both the
    /// same, and the engine never allocates for a directory at all.
    async fn allocate(&self, kind: ObjectKind, len: u64) -> Result<Self::Key, ProviderError>;

    /// Read from `offset` into `dst`, returning the bytes transferred.
    ///
    /// A read that starts inside the object but runs past its end is short,
    /// not an error — the count says how much was real.  A read starting at
    /// or past the end transfers zero.  A read starting beyond the placement
    /// is [`ProviderError::Range`].
    async fn read(
        &self,
        key: Self::Key,
        offset: u64,
        dst: &mut [u8],
    ) -> Result<usize, ProviderError>;

    /// Write `src` at `offset`.  All-or-nothing: a refused write leaves every
    /// byte as it was.
    ///
    /// The engine has already reserved `offset + src.len()` bytes via
    /// [`Self::resize`], so this fails only on a real medium error or a
    /// caller bug.
    async fn write(&self, key: Self::Key, offset: u64, src: &[u8]) -> Result<(), ProviderError>;

    /// Copy the first `len` bytes of `source` over the first `len` bytes of
    /// `destination` — the snapshot a version lend freezes.  Both placements
    /// must hold at least `len` bytes; `destination` must not be committed.
    ///
    /// The default streams through one transfer-sized scratch buffer, which
    /// is what the RAM provider needs and all the engine used to do.  A
    /// provider whose bytes never have to visit the CPU overrides it: the
    /// block provider moves whole extents between its DMA buffers, so a
    /// 9 MiB image costs a few hundred commands instead of a few thousand.
    /// A refused copy leaves `destination` in an unspecified state below
    /// `len`; the caller releases it.
    async fn copy_object(
        &self,
        source: Self::Key,
        destination: Self::Key,
        len: u64,
    ) -> Result<(), ProviderError> {
        let mut buffer = [0u8; crate::bounds::MAX_TRANSFER];
        let chunk = self.limits().max_transfer.min(buffer.len());
        let mut offset = 0u64;
        while offset < len {
            let want = usize::try_from((len - offset).min(chunk as u64))
                .map_err(|_| ProviderError::Range)?;
            let read = self.read(source, offset, &mut buffer[..want]).await?;
            if read != want {
                return Err(ProviderError::Media);
            }
            self.write(destination, offset, &buffer[..read]).await?;
            offset += read as u64;
        }
        Ok(())
    }
    /// Reserve or release capacity so the placement holds exactly `len`
    /// bytes.  Growth reads as zeros.  A refused resize leaves the placement
    /// unchanged.
    async fn resize(&self, key: Self::Key, len: u64) -> Result<(), ProviderError>;

    /// Release the placement.  The key is stale afterwards.
    async fn release(&self, key: Self::Key) -> Result<(), ProviderError>;

    /// Cross the provider's durability contract for this placement.
    ///
    /// For RAM this is a no-op that reports success, because "visible to
    /// later authorized operations in this boot" is the whole of RAM's
    /// durability promise. A provider must never report durable
    /// completion it cannot deliver.
    async fn sync(&self, key: Self::Key) -> Result<(), ProviderError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_on;

    /// The conformance expectations every provider must meet, exercised here
    /// against RAM and the direct block provider — which is the point of
    /// having a seam at all, and why this
    /// lives beside the trait rather than inside `ram`.
    fn provider_conformance<P: StorageProvider>(provider: &P) {
        let limits = provider.limits();
        assert!(limits.total_bytes > 0);
        assert!(limits.max_object_bytes > 0);
        assert!(limits.max_transfer > 0);
        assert!(limits.granularity > 0);
        assert!(limits.max_object_bytes <= limits.total_bytes);

        let key = block_on(provider.allocate(ObjectKind::File, 8)).expect("small allocate");

        // Fresh capacity reads as zeros.
        let mut buf = [0xffu8; 8];
        assert_eq!(block_on(provider.read(key, 0, &mut buf)), Ok(8));
        assert_eq!(buf, [0u8; 8]);

        // Write then read back.
        assert_eq!(block_on(provider.write(key, 0, b"kore")), Ok(()));
        let mut back = [0u8; 4];
        assert_eq!(block_on(provider.read(key, 0, &mut back)), Ok(4));
        assert_eq!(&back, b"kore");

        // Reads at and past the end are short, not errors.
        let mut tail = [0u8; 4];
        assert_eq!(block_on(provider.read(key, 6, &mut tail)), Ok(2));
        assert_eq!(block_on(provider.read(key, 8, &mut tail)), Ok(0));

        // Sync must report success or a real reason — never silence.
        assert_eq!(block_on(provider.sync(key)), Ok(()));

        // Grow, then the new region reads as zeros.
        assert_eq!(block_on(provider.resize(key, 600)), Ok(()));
        let mut grown = [0xabu8; 16];
        assert_eq!(block_on(provider.read(key, 584, &mut grown)), Ok(16));
        assert_eq!(grown, [0u8; 16]);
        // The original bytes survived the growth.
        assert_eq!(block_on(provider.read(key, 0, &mut back)), Ok(4));
        assert_eq!(&back, b"kore");

        // Shrink, then the tail is gone.
        // A snapshot copy carries exactly the first `len` bytes of the
        // source, whichever path the provider takes.
        let copy = block_on(provider.allocate(ObjectKind::File, 600)).expect("copy allocate");
        assert_eq!(block_on(provider.copy_object(key, copy, 600)), Ok(()));
        let mut copied = [0u8; 4];
        assert_eq!(block_on(provider.read(copy, 0, &mut copied)), Ok(4));
        assert_eq!(&copied, b"kore");
        assert_eq!(block_on(provider.read(copy, 584, &mut grown)), Ok(16));
        assert_eq!(grown, [0u8; 16]);
        assert_eq!(block_on(provider.release(copy)), Ok(()));

        assert_eq!(block_on(provider.resize(key, 4)), Ok(()));
        assert_eq!(block_on(provider.read(key, 4, &mut tail)), Ok(0));

        // An over-sized object is refused by declared limit, not by crash.
        assert_eq!(
            block_on(provider.allocate(ObjectKind::File, limits.max_object_bytes + 1)),
            Err(ProviderError::TooLarge)
        );

        // Release makes the key stale; every op on it fails closed.
        assert_eq!(block_on(provider.release(key)), Ok(()));
        let mut after = [0u8; 4];
        assert_eq!(
            block_on(provider.read(key, 0, &mut after)),
            Err(ProviderError::UnknownKey)
        );
        assert_eq!(
            block_on(provider.write(key, 0, b"x")),
            Err(ProviderError::UnknownKey)
        );
        assert_eq!(
            block_on(provider.resize(key, 1)),
            Err(ProviderError::UnknownKey)
        );
        assert_eq!(
            block_on(provider.release(key)),
            Err(ProviderError::UnknownKey)
        );
        assert_eq!(block_on(provider.sync(key)), Err(ProviderError::UnknownKey));
    }

    #[test]
    fn ram_provider_meets_the_contract() {
        provider_conformance(&RamProvider::new());
    }

    #[test]
    fn block_provider_meets_the_same_contract() {
        // The in-memory device fixture (128 KiB) and the provider's placement
        // arenas (sized to MAX_BLOCK_PLACEMENTS) are constructed by value, which
        // exceeds a default test-thread stack. This is a TEST-fixture concern
        // only: on metal the durable service is constructed once and moved into
        // its named heap window (the `size_of::<DurableService>()` guard bounds
        // that), never held on a bounded stack. Give the fixture a generous
        // stack rather than shrinking a bound that is a correct worst case
        // (MAX_OBJECTS file extents + MAX_DERIVED_RESULTS durable batches).
        std::thread::Builder::new()
            .stack_size(8 << 20)
            .spawn(|| {
                use kore_driver::block::{BlockChannel, CompletionMode, memory::MemoryBlockDevice};
                type Device = MemoryBlockDevice<131072, 8192>;
                let device = Device::new(512, 16).unwrap();
                let channel: BlockChannel<Device, 3, 8> =
                    BlockChannel::new(device, CompletionMode::Poll).unwrap();
                let provider = BlockProvider::new(channel).unwrap();
                provider_conformance(&provider);
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
