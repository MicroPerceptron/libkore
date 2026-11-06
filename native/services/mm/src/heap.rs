//! Async preparation for the ordinary `kore_rt::Heap` engine.
//!
//! A miss records the required extent size and returns `AllocError`. The
//! owning service awaits `refill` over its existing mm face, then retries the
//! allocation once. No allocator callback runs an executor or polls a ring.
//! Each extent stores its own affine backing owner ahead of the heap bytes.

use crate::{MmClient, MmRefusal, MmResponse};
use core::{
    ptr::NonNull,
    sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering},
};
use kore_rt::{
    cap::{MappedLeaseView, StashedLease, verbs},
    mem_source::{ALIGN, MemSource},
};

const _: () = assert!(core::mem::align_of::<Extent>() <= ALIGN);

struct Extent {
    next: *mut Extent,
    backing: StashedLease,
    view: Option<MappedLeaseView>,
    bytes: usize,
    cursor: AtomicUsize,
}

/// Why a cold heap refill could not make a new extent available. These are
/// observations of the existing path, not requests for a different policy.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefillFailure {
    PairBusy = 1,
    SourceBusy = 2,
    NoDemand = 3,
    ExtentOverflow = 4,
    MmBadProps = 5,
    MmExhausted = 6,
    MmBadReturn = 7,
    MmDenied = 8,
    MmQuota = 9,
    MmPending = 10,
    UnexpectedReply = 11,
    MissingLease = 12,
    InstallFailed = 13,
    /// The refill succeeded, but the caller's exact allocation still failed.
    ReserveAfterRefill = 14,
    /// mm answered [`MmRefusal::Full`]. Its supply path never does — that
    /// refusal is a face admission's — but the mapping stays total, so a
    /// reply outside the request's vocabulary is named, not miscounted.
    MmFull = 15,
}

impl RefillFailure {
    pub const fn from_code(code: u8) -> Option<Self> {
        Some(match code {
            1 => Self::PairBusy,
            2 => Self::SourceBusy,
            3 => Self::NoDemand,
            4 => Self::ExtentOverflow,
            5 => Self::MmBadProps,
            6 => Self::MmExhausted,
            7 => Self::MmBadReturn,
            8 => Self::MmDenied,
            9 => Self::MmQuota,
            10 => Self::MmPending,
            11 => Self::UnexpectedReply,
            12 => Self::MissingLease,
            13 => Self::InstallFailed,
            14 => Self::ReserveAfterRefill,
            15 => Self::MmFull,
            _ => return None,
        })
    }

    const fn from_refusal(refusal: MmRefusal) -> Self {
        match refusal {
            MmRefusal::BadProps => Self::MmBadProps,
            MmRefusal::Exhausted => Self::MmExhausted,
            MmRefusal::BadReturn => Self::MmBadReturn,
            MmRefusal::Denied => Self::MmDenied,
            MmRefusal::Quota => Self::MmQuota,
            MmRefusal::Pending => Self::MmPending,
            MmRefusal::Full => Self::MmFull,
        }
    }
}

/// An asynchronously supplied, named heap source. Worker allocation only
/// claims already-published extents or records a miss; it never runs IPC.
/// The owning service awaits refill. Extent headers are immutable after
/// publication except for their atomic, monotone page cursor. Supplied memory
/// remains owned until the heap is destroyed; freed blocks recycle in Heap.
pub struct MmSource {
    extents: AtomicPtr<Extent>,
    wanted: AtomicUsize,
    refilling: AtomicBool,
    mapping: Option<(u64, u64)>,
    mapped_bytes: AtomicUsize,
}

impl MmSource {
    pub const fn new() -> Self {
        Self {
            extents: AtomicPtr::new(core::ptr::null_mut()),
            wanted: AtomicUsize::new(0),
            refilling: AtomicBool::new(false),
            mapping: None,
            mapped_bytes: AtomicUsize::new(0),
        }
    }

    /// Use an exclusively owned, founder-declared virtual address reservation.
    /// Physical pages are still supplied on demand; this is not a memory budget.
    ///
    /// # Safety
    /// No other mapper may install into this span while the source is live.
    pub unsafe fn in_window(base: u64, bytes: u64) -> Self {
        let mut source = Self::new();
        source.mapping = Some((base, bytes));
        source
    }

    /// Satisfy the preceding allocation miss through the caller's real mm
    /// client. A refusal preserves every previously allocated extent.
    pub async fn refill(&self, memory: &mut impl MmClient) -> bool {
        self.refill_detailed(memory).await.is_ok()
    }

    /// The same refill, retaining its terminal reason for a caller's failure
    /// witness. A successful result still needs the caller to retry its own
    /// allocation; the source never promises an allocation on its behalf.
    pub async fn refill_detailed(&self, memory: &mut impl MmClient) -> Result<(), RefillFailure> {
        self.provide(memory, 0).await
    }

    /// Establish an initial mm-supplied extent before advertising readiness.
    /// This verifies and warms the real client path even when a bootstrap
    /// supply still has space. Subsequent misses use `refill` on demand.
    pub async fn reserve_extent(&self, memory: &mut impl MmClient, bytes: usize) -> bool {
        self.provide(memory, bytes).await.is_ok()
    }

    async fn provide(&self, memory: &mut impl MmClient, minimum: usize) -> Result<(), RefillFailure> {
        // A deferred close keeps its reply on this pair. A refill must leave
        // both that continuation and the recorded allocation demand intact.
        if !memory.is_idle() {
            return Err(RefillFailure::PairBusy);
        }
        // This is a non-blocking per-source admission guard on the cold IPC
        // path. Allocation never waits on it; concurrent refill callers retry.
        if self
            .refilling
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return Err(RefillFailure::SourceBusy);
        }
        let wanted = self.wanted.swap(0, Ordering::AcqRel).max(minimum);
        let mut guard = RefillGuard {
            active: &self.refilling,
            wanted: &self.wanted,
            pending: wanted,
        };
        if wanted == 0 {
            return Err(RefillFailure::NoDemand);
        }
        if minimum == 0 && self.has_ready(wanted) {
            guard.pending = 0;
            return Ok(());
        }
        let Some(bytes) = extent_bytes(wanted) else {
            guard.pending = 0;
            return Err(RefillFailure::ExtentOverflow);
        };
        let (reply, backing) = memory.supply(bytes as u64).await;
        // A completed refusal belongs to this request. Retaining its maximum
        // forever would let an oversized surface starve later small requests.
        // Cancellation alone restores demand; concurrent new misses survive.
        guard.pending = 0;
        let backing = match (reply, backing) {
            (MmResponse::Granted, Some(backing)) => backing,
            (MmResponse::Granted, None) => return Err(RefillFailure::MissingLease),
            (MmResponse::Refused(refusal), _) => {
                return Err(RefillFailure::from_refusal(refusal));
            }
            _ => return Err(RefillFailure::UnexpectedReply),
        };
        if !self.install(backing, bytes) {
            return Err(RefillFailure::InstallFailed);
        }
        guard.pending = 0;
        Ok(())
    }

    /// Seed a named heap from a founder-endowed supply before sharing it.
    /// The source retains the real affine lease and maps it receiver-side.
    pub fn seed(&mut self, backing: StashedLease) -> bool {
        self.install(backing, header_bytes() + ALIGN)
    }

    // Caller holds exclusive access or the per-source refill admission guard.
    fn install(&self, mut backing: StashedLease, bytes: usize) -> bool {
        let mut view = None;
        let mapped = (|| {
            let slot = backing.slot()?;
            if let Some((base, capacity)) = self.mapping {
                let (_, len) = verbs::lease_geometry(slot)?;
                let offset = self.mapped_bytes.load(Ordering::Relaxed) as u64;
                let end = offset.checked_add(len)?;
                if base % ALIGN as u64 != 0
                    || len % ALIGN as u64 != 0
                    || len < bytes as u64
                    || end > capacity
                {
                    return None;
                }
                let mapped = verbs::map_owned_lease_at_rw(slot, base.checked_add(offset)?, len)?;
                let ptr = NonNull::slice_from_raw_parts(
                    NonNull::new(mapped.as_mut_ptr())?,
                    usize::try_from(len).ok()?,
                );
                view = Some(mapped);
                self.mapped_bytes
                    .store(usize::try_from(end).ok()?, Ordering::Relaxed);
                Some(ptr)
            } else {
                verbs::map_held_rw(slot)
            }
        })();
        let Some(mapped) = mapped else {
            let _ = backing.try_retire();
            return false;
        };
        let base = mapped.cast::<u8>();
        if base.as_ptr() as usize % ALIGN != 0 || mapped.len() < bytes {
            drop(view);
            let _ = backing.try_retire();
            return false;
        }
        // SAFETY: this fresh, exclusively held mapping covers its header and
        // payload. The release store publishes initialized immutable metadata
        // and the affine owner before any worker can claim the payload.
        unsafe {
            let header = base.cast::<Extent>().as_ptr();
            header.write(Extent {
                next: self.extents.load(Ordering::Relaxed),
                backing,
                view,
                bytes: mapped.len(),
                cursor: AtomicUsize::new(header_bytes()),
            });
            self.extents.store(header, Ordering::Release);
        }
        true
    }

    /// Derive a separately revocable client root for a live heap allocation.
    ///
    /// # Safety
    /// The caller must own the entire live, page-aligned allocation at
    /// `address..address + bytes` from the Heap using this source. It must
    /// revoke this root and its descendants before deallocating that block.
    /// No allocator metadata or adjacent allocation may share its pages.
    pub unsafe fn carve_allocation(&self, address: usize, bytes: usize) -> Option<StashedLease> {
        if bytes == 0 || address % 4096 != 0 || bytes % 4096 != 0 {
            return None;
        }
        let end = address.checked_add(bytes)?;
        let mut header = self.extents.load(Ordering::Acquire);
        while !header.is_null() {
            // SAFETY: immutable source-owned header, published with release.
            let extent = unsafe { &*header };
            let base = header as usize;
            if address >= base.checked_add(header_bytes())?
                && end <= base.checked_add(extent.bytes)?
            {
                return verbs::carve_owned_lease_window(
                    extent.backing.slot()?,
                    (address - base) as u64,
                    bytes as u64,
                );
            }
            header = extent.next;
        }
        None
    }

    fn has_ready(&self, bytes: usize) -> bool {
        let mut header = self.extents.load(Ordering::Acquire);
        while !header.is_null() {
            // SAFETY: source-owned headers persist until exclusive Drop.
            let extent = unsafe { &*header };
            if extent
                .bytes
                .saturating_sub(extent.cursor.load(Ordering::Acquire))
                >= bytes
            {
                return true;
            }
            header = extent.next;
        }
        false
    }
}

struct RefillGuard<'a> {
    active: &'a AtomicBool,
    wanted: &'a AtomicUsize,
    pending: usize,
}
impl Drop for RefillGuard<'_> {
    fn drop(&mut self) {
        // A cancelled IPC must not consume the outstanding miss.
        self.wanted.fetch_max(self.pending, Ordering::Release);
        self.active.store(false, Ordering::Release);
    }
}

// SAFETY: publication is release/acquire, extent claims are one-shot CAS,
// and the cold refill guard permits one publisher. Retained affine owners are
// never mutated while shared; exclusive Drop moves them out before revocation.
unsafe impl Send for MmSource {}
unsafe impl Sync for MmSource {}

impl Default for MmSource {
    fn default() -> Self {
        Self::new()
    }
}

const fn header_bytes() -> usize {
    core::mem::size_of::<Extent>().div_ceil(ALIGN) * ALIGN
}

fn extent_bytes(wanted: usize) -> Option<usize> {
    wanted
        .checked_add(header_bytes())?
        .checked_add(ALIGN - 1)
        .map(|n| n / ALIGN * ALIGN)
}

// SAFETY: each extent's payload is aligned, distinct, source-owned and
// transferred once by CAS. The header remains outside every heap extent.
unsafe impl MemSource for MmSource {
    fn map_extent(&self, min_bytes: usize) -> Option<NonNull<[u8]>> {
        let mut header = self.extents.load(Ordering::Acquire);
        while !header.is_null() {
            // SAFETY: publication precedes traversal; Drop requires exclusive
            // source ownership and cannot overlap an allocation borrow.
            let extent = unsafe { &*header };
            let bytes = min_bytes.max(1).checked_add(ALIGN - 1)? / ALIGN * ALIGN;
            if let Some(offset) = claim_bytes(&extent.cursor, extent.bytes, bytes) {
                // SAFETY: CAS reserves a disjoint, page-aligned subextent;
                // the affine supply owner remains outside all returned bytes.
                let payload = unsafe { NonNull::new_unchecked(header.cast::<u8>().add(offset)) };
                return Some(NonNull::slice_from_raw_parts(payload, bytes));
            }
            header = extent.next;
        }
        self.wanted.fetch_max(min_bytes, Ordering::Release);
        None
    }
}

fn claim_bytes(cursor: &AtomicUsize, total: usize, bytes: usize) -> Option<usize> {
    cursor
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |offset| {
            offset.checked_add(bytes).filter(|end| *end <= total)
        })
        .ok()
}

impl Drop for MmSource {
    fn drop(&mut self) {
        let mut header = *self.extents.get_mut();
        while !header.is_null() {
            // SAFETY: move the only owner and next pointer before revoke can
            // unmap this header. All borrowers/workers have already finished.
            let Extent {
                next,
                mut backing,
                view,
                ..
            } = unsafe { header.read() };
            header = next;
            drop(view);
            let _ = backing.try_retire();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrent_subextents_never_overlap_or_consume_the_header() {
        let cursor = AtomicUsize::new(header_bytes());
        let total = header_bytes() + 64 * ALIGN;
        let claimed: [AtomicUsize; 64] = core::array::from_fn(|_| AtomicUsize::new(0));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    while let Some(offset) = claim_bytes(&cursor, total, ALIGN) {
                        assert!(offset >= header_bytes());
                        assert_eq!(offset % ALIGN, 0);
                        assert_eq!(
                            claimed[(offset - header_bytes()) / ALIGN]
                                .fetch_add(1, Ordering::Relaxed),
                            0
                        );
                    }
                });
            }
        });
        assert!(claimed.iter().all(|n| n.load(Ordering::Relaxed) == 1));
        assert_eq!(cursor.load(Ordering::Relaxed), total);
    }

    #[test]
    fn extent_geometry_accounts_for_owner_and_refuses_overflow() {
        assert_eq!(extent_bytes(ALIGN), Some(ALIGN + header_bytes()));
        assert_eq!(extent_bytes(ALIGN + 1), Some(2 * ALIGN + header_bytes()));
        assert_eq!(extent_bytes(usize::MAX), None);
        assert!(header_bytes() >= core::mem::size_of::<Extent>());
    }

    #[test]
    fn miss_is_an_explicit_refill_request_not_a_poll() {
        let source = MmSource::new();
        assert!(source.map_extent(ALIGN).is_none());
        assert_eq!(source.wanted.load(Ordering::Acquire), ALIGN);
        assert!(source.extents.load(Ordering::Acquire).is_null());
    }
    #[test]
    fn concurrent_misses_preserve_the_largest_request() {
        let source = MmSource::new();
        std::thread::scope(|scope| {
            for pages in 1..=8 {
                let source = &source;
                scope.spawn(move || {
                    assert!(source.map_extent(pages * ALIGN).is_none());
                });
            }
        });
        assert_eq!(source.wanted.load(Ordering::Acquire), 8 * ALIGN);
    }

    #[test]
    fn completed_refusal_does_not_poison_a_smaller_allocation() {
        struct Refusing { idle: bool, requests: usize }
        impl MmClient for Refusing {
            fn is_idle(&self) -> bool {
                self.idle
            }
            async fn request(
                &mut self,
                _: crate::MmRequest,
            ) -> (MmResponse, Option<kore_rt::cap::ReturnedCapability>) {
                (MmResponse::Refused(crate::MmRefusal::Quota), None)
            }
            async fn supply(&mut self, _: u64) -> (MmResponse, Option<StashedLease>) {
                self.requests += 1;
                (MmResponse::Refused(crate::MmRefusal::Quota), None)
            }
            async fn supply_in_class(
                &mut self,
                _: u64,
                _: u64,
            ) -> (MmResponse, Option<StashedLease>) {
                (MmResponse::Refused(crate::MmRefusal::Quota), None)
            }
            async fn stats(&mut self) -> Option<kore_abi::monitor::MemorySnapshot> {
                None
            }
        }
        let source = MmSource::new();
        assert!(source.map_extent(usize::MAX / 2).is_none());
        let mut client = Refusing { idle: false, requests: 0 };
        {
            let mut refill = core::pin::pin!(source.refill_detailed(&mut client));
            let mut context = core::task::Context::from_waker(core::task::Waker::noop());
            assert_eq!(
                core::future::Future::poll(refill.as_mut(), &mut context),
                core::task::Poll::Ready(Err(RefillFailure::PairBusy))
            );
        }
        assert_eq!(client.requests, 0, "an occupied pair publishes nothing");
        assert_eq!(source.wanted.load(Ordering::Acquire), usize::MAX / 2);
        client.idle = true;
        let mut refill = core::pin::pin!(source.refill_detailed(&mut client));
        let mut context = core::task::Context::from_waker(core::task::Waker::noop());
        assert_eq!(
            core::future::Future::poll(refill.as_mut(), &mut context),
            core::task::Poll::Ready(Err(RefillFailure::MmQuota))
        );
        assert_eq!(source.wanted.load(Ordering::Acquire), 0);
        assert!(source.map_extent(ALIGN).is_none());
        assert_eq!(source.wanted.load(Ordering::Acquire), ALIGN);
    }

    #[test]
    fn refill_outcomes_keep_exact_mm_refusals_and_local_admission_causes() {
        for (refusal, expected) in [
            (MmRefusal::BadProps, RefillFailure::MmBadProps),
            (MmRefusal::Exhausted, RefillFailure::MmExhausted),
            (MmRefusal::BadReturn, RefillFailure::MmBadReturn),
            (MmRefusal::Denied, RefillFailure::MmDenied),
            (MmRefusal::Quota, RefillFailure::MmQuota),
            (MmRefusal::Pending, RefillFailure::MmPending),
            (MmRefusal::Full, RefillFailure::MmFull),
        ] {
            assert_eq!(RefillFailure::from_refusal(refusal), expected);
            assert_eq!(RefillFailure::from_code(expected as u8), Some(expected));
        }
        assert_eq!(RefillFailure::from_code(0), None);
        assert_eq!(RefillFailure::from_code(16), None);
    }

    #[test]
    fn abandoned_refill_restores_demand_without_losing_a_newer_miss() {
        let source = MmSource::new();
        source.refilling.store(true, Ordering::Relaxed);
        {
            let _guard = RefillGuard {
                active: &source.refilling,
                wanted: &source.wanted,
                pending: ALIGN,
            };
            assert!(source.map_extent(2 * ALIGN).is_none());
        }
        assert_eq!(source.wanted.load(Ordering::Acquire), 2 * ALIGN);
        assert!(!source.refilling.load(Ordering::Acquire));
    }
}
