//! Filesystem names for the shared bounded generational claim map.
//!
//! The first filesystem implementation carried a private, single-owner
//! `SlotTable`.  That preserved generations but forced every operation through
//! one mutable engine borrow.  Phase 5 uses the workspace collection directly:
//! each object, directory, authority edge, and handle can now be claimed
//! independently, with explicit contention and no filesystem-global lock.

pub use kore_collections::{
    AtomicBitVec, AtomicSlotMap as SlotTable, AtomicSlotVec, BoundedVec, SlotAccess,
    SlotId as SlotKey, SlotRead, SlotVecRead, SlotVecReservation, SlotVecWrite, SlotWrite,
};

use crate::error::FsError;

/// Preserve the claim map's temporal distinction at the filesystem seam.
///
/// Callers supply the meaning of a stale generation (`InvalidHandle` for an
/// opaque client handle, `AccessLost` for an internal route). Contention has
/// one meaning everywhere: bounded, retryable [`FsError::Busy`].
#[inline]
pub(crate) const fn fs_access(error: SlotAccess, stale: FsError) -> FsError {
    match error {
        SlotAccess::Stale => stale,
        SlotAccess::Busy => FsError::Busy,
    }
}

/// Service-private access to the arena key behind a wire-visible identity.
///
/// [`Handle`], [`GrantId`], and [`VersionLease`] are process-side leaves
/// (`kore_dagfs`): a client holds and encodes them without knowing they name
/// generational arena slots. Only this service reconstructs the slot, and it
/// does so through the public bit forms — so the leaf's contract stays
/// "opaque bits" while the exact stale-generation checks keep working.
pub(crate) trait Keyed: Sized {
    fn key(self) -> SlotKey;
    fn from_key(key: SlotKey) -> Self;
}

macro_rules! keyed {
    ($($id:ty),* $(,)?) => {$(
        impl Keyed for $id {
            #[inline]
            fn key(self) -> SlotKey {
                SlotKey::from_bits(self.to_bits())
            }
            #[inline]
            fn from_key(key: SlotKey) -> Self {
                Self::from_bits(key.to_bits())
            }
        }
    )*};
}

keyed!(
    kore_dagfs::Handle,
    kore_dagfs::GrantId,
    kore_dagfs::VersionLease
);
