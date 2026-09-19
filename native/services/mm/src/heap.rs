//! Async preparation for the ordinary `kore_rt::Heap` engine.
//!
//! A miss records the required extent size and returns `AllocError`. The
//! owning service awaits `refill` over its existing mm face, then retries the
//! allocation once. No allocator callback runs an executor or polls a ring.
//! Each extent stores its own affine backing owner ahead of the heap bytes.

use crate::client::SupplyError;
use crate::{MmClient, MmResponse};
use core::{
    cell::UnsafeCell,
    ptr::NonNull,
    sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering},
};
use kore_rt::svc::CallError;
use kore_rt::{
    cap::{MappedLeaseView, StashedLease, verbs},
    mem_source::{ALIGN, MemSource},
};

const _: () = assert!(core::mem::align_of::<Extent>() <= ALIGN);
const REFILL_OBSERVATION_NS: u64 = 5_000_000;

struct Extent {
    next: *mut Extent,
    backing: StashedLease,
    view: Option<MappedLeaseView>,
    bytes: usize,
    cursor: AtomicUsize,
}

#[derive(Default)]
struct FailedExtent {
    backing: Option<StashedLease>,
    view: Option<MappedLeaseView>,
    uncertain: Option<kore_rt::cap::UncertainLeaseMap>,
}

impl FailedExtent {
    const fn empty() -> Self {
        Self {
            backing: None,
            view: None,
            uncertain: None,
        }
    }

    fn try_retire(&mut self) -> bool {
        if let Some(owner) = self.uncertain.as_mut() {
            if !owner.try_retire() {
                return false;
            }
            self.uncertain = None;
        }
        if let Some(view) = self.view.as_mut() {
            if !view.try_retire_outcome().is_closed() {
                return false;
            }
            self.view = None;
        }
        if let Some(backing) = self.backing.as_mut() {
            if !backing.try_retire_outcome().is_closed() {
                return false;
            }
            self.backing = None;
        }
        true
    }
}

impl Drop for FailedExtent {
    fn drop(&mut self) {
        if !self.try_retire() {
            // No retry owner can outlive this source. Keep the mapping name
            // and its backing reserved until task teardown; ordinary field
            // Drop must not revoke backing beneath an unconfirmed mapping.
            core::mem::forget(self.uncertain.take());
            core::mem::forget(self.view.take());
            core::mem::forget(self.backing.take());
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
    // Zero means no request. Only the holder of refilling may change this;
    // it survives cancellation while the same MM pair retains correlation.
    pending_supply: AtomicUsize,
    // Access only with exclusive self or while holding refilling. Allocation
    // readers never touch this unpublished owner; the atomic is only a hint.
    failed: UnsafeCell<FailedExtent>,
    cleanup_pending: AtomicBool,
    mapping: Option<(u64, u64)>,
    mapped_bytes: AtomicUsize,
}

impl MmSource {
    pub const fn new() -> Self {
        Self {
            extents: AtomicPtr::new(core::ptr::null_mut()),
            wanted: AtomicUsize::new(0),
            refilling: AtomicBool::new(false),
            pending_supply: AtomicUsize::new(0),
            failed: UnsafeCell::new(FailedExtent::empty()),
            cleanup_pending: AtomicBool::new(false),
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

    /// Whether this source retains a supply observation on its MM client.
    /// The service owner must resume it before another user of that same pair
    /// starts an operation. This is scheduling metadata, not a new MM handle.
    pub fn has_pending_refill(&self) -> bool {
        self.pending_supply.load(Ordering::Acquire) != 0
    }

    /// Local cleanup needs observation even after the MM reply completed.
    pub fn needs_maintenance(&self) -> bool {
        self.has_pending_refill() || self.cleanup_pending.load(Ordering::Acquire)
    }

    /// Satisfy the preceding allocation miss through the caller's real mm
    /// client. A refusal preserves every previously allocated extent. Until
    /// a pending refill completes, retries must use that same held MM client.
    pub async fn refill(&self, memory: &mut impl MmClient) -> bool {
        self.provide(memory, 0).await
    }

    /// Establish an initial mm-supplied extent before advertising readiness.
    /// This verifies and warms the real client path even when a bootstrap
    /// supply still has space. Subsequent misses use `refill` on demand.
    pub async fn reserve_extent(&self, memory: &mut impl MmClient, bytes: usize) -> bool {
        self.provide(memory, bytes).await
    }

    async fn provide(&self, memory: &mut impl MmClient, minimum: usize) -> bool {
        // This is a non-blocking per-source admission guard on the cold IPC
        // path. Allocation never waits on it; concurrent refill callers retry.
        if self
            .refilling
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return false;
        }
        let wanted = self.wanted.swap(0, Ordering::AcqRel).max(minimum);
        let mut guard = RefillGuard {
            active: &self.refilling,
            wanted: &self.wanted,
            pending: wanted,
        };
        // SAFETY: the per-source refill guard is held until this future drops.
        let failed = unsafe { &mut *self.failed.get() };
        if !failed.try_retire() {
            return false;
        }
        self.cleanup_pending.store(false, Ordering::Release);
        let pending = self.pending_supply.load(Ordering::Relaxed);
        let bytes;
        let result;
        if pending != 0 {
            bytes = pending;
            result = match memory
                .resume_supply_bounded(bytes as u64, REFILL_OBSERVATION_NS)
                .await
            {
                Ok(reply) => reply,
                Err(CallError::NoOutstanding) => {
                    // Cancellation may precede publication. The original pair
                    // proves there is nothing to resume; restore demand and
                    // permit a fresh request on a later refill call.
                    self.pending_supply.store(0, Ordering::Relaxed);
                    return false;
                }
                Err(_) => return false,
            };
        } else {
            if wanted == 0 {
                return false;
            }
            if minimum == 0 && self.has_ready(wanted) {
                guard.pending = 0;
                return true;
            }
            let Some(requested) = extent_bytes(wanted) else {
                guard.pending = 0;
                return false;
            };
            bytes = requested;
            // Store identity before entering the fallible/cancellable client.
            // A completed typed reply is consumed synchronously below.
            self.pending_supply.store(bytes, Ordering::Relaxed);
            result = match memory.supply_observed(bytes as u64, None).await {
                Ok(reply) => reply,
                Err(SupplyError::NotStarted(_)) => {
                    self.pending_supply.store(0, Ordering::Relaxed);
                    return false;
                }
                Err(SupplyError::Pending(_)) => return false,
            };
        }
        self.pending_supply.store(0, Ordering::Relaxed);
        // Settle only the demand covered by this request. A larger miss may
        // have arrived while its cancelled/failed observation was pending.
        guard.pending = if wanted > bytes - header_bytes() {
            wanted
        } else {
            0
        };
        let (MmResponse::Granted, Some(backing)) = result else {
            return false;
        };
        self.install(failed, backing, bytes)
    }

    /// Seed a named heap from a founder-endowed supply before sharing it.
    /// The source retains the real affine lease and maps it receiver-side.
    /// An occupied cleanup record leaves the caller's input untouched.
    pub fn seed(&mut self, backing: &mut Option<StashedLease>) -> bool {
        let failed = self.failed.get_mut();
        if !failed.try_retire() {
            return false;
        }
        self.cleanup_pending.store(false, Ordering::Release);
        let Some(backing) = backing.take() else {
            return false;
        };
        // Exclusive self makes the cell accessible without taking refilling.
        let mut failed = core::mem::replace(failed, FailedExtent::empty());
        let installed = self.install(&mut failed, backing, header_bytes() + ALIGN);
        *self.failed.get_mut() = failed;
        installed
    }

    // Caller holds exclusive access or the per-source refill admission guard.
    fn install(&self, failed: &mut FailedExtent, backing: StashedLease, bytes: usize) -> bool {
        debug_assert!(
            failed.backing.is_none() && failed.view.is_none() && failed.uncertain.is_none()
        );
        failed.backing = Some(backing);
        self.cleanup_pending.store(true, Ordering::Release);
        let mapped = (|| {
            let slot = failed.backing.as_ref()?.slot()?;
            let (result, end) = if let Some((base, capacity)) = self.mapping {
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
                let end = usize::try_from(end).ok()?;
                (
                    verbs::map_owned_lease_at_rw_outcome(slot, base.checked_add(offset)?, len),
                    Some(end),
                )
            } else {
                (verbs::map_owned_lease_rw_outcome(slot), None)
            };
            let view = match result {
                Ok(view) => view,
                Err(verbs::LeaseMapError::Uncertain(owner)) => {
                    failed.uncertain = Some(owner);
                    return None;
                }
                Err(verbs::LeaseMapError::Refused(_)) => return None,
            };
            // Store the actual view before inspecting its usable heap shape.
            failed.view = Some(view);
            let view = failed.view.as_ref()?;
            let base = NonNull::new(view.as_mut_ptr())?;
            let len = view.len();
            if base.as_ptr() as usize % ALIGN != 0 || len < bytes {
                return None;
            }
            Some((base, len, end))
        })();
        let Some((base, len, end)) = mapped else {
            if failed.try_retire() {
                self.cleanup_pending.store(false, Ordering::Release);
            }
            return false;
        };
        // SAFETY: the complete private mapping covers header and payload.
        // Publish the affine owners before workers can claim any payload.
        unsafe {
            let header = base.cast::<Extent>().as_ptr();
            header.write(Extent {
                next: self.extents.load(Ordering::Relaxed),
                backing: failed.backing.take().unwrap(),
                view: failed.view.take(),
                bytes: len,
                cursor: AtomicUsize::new(header_bytes()),
            });
            if let Some(end) = end {
                self.mapped_bytes.store(end, Ordering::Relaxed);
            }
            self.extents.store(header, Ordering::Release);
        }
        self.cleanup_pending.store(false, Ordering::Release);
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
        unsafe { self.carve_allocation_outcome(address, bytes) }.ok()
    }

    /// Carve a client root while retaining cleanup custody on an unknown
    /// completion. The caller must keep the allocation until either the
    /// successful root or the uncertain owner confirms retirement.
    ///
    /// # Safety
    /// The same exclusive live-allocation requirements as `carve_allocation`
    /// apply, including keeping allocator metadata outside the carved pages.
    pub unsafe fn carve_allocation_outcome(
        &self,
        address: usize,
        bytes: usize,
    ) -> Result<StashedLease, verbs::LeaseCarveError> {
        let bounds = || verbs::LeaseCarveError::Refused(verbs::LeaseCarveRefusal::Bounds);
        if bytes == 0 || address % 4096 != 0 || bytes % 4096 != 0 {
            return Err(bounds());
        }
        let end = address.checked_add(bytes).ok_or_else(bounds)?;
        let mut header = self.extents.load(Ordering::Acquire);
        while !header.is_null() {
            // SAFETY: immutable source-owned header, published with release.
            let extent = unsafe { &*header };
            let base = header as usize;
            if address >= base.checked_add(header_bytes()).ok_or_else(bounds)?
                && end <= base.checked_add(extent.bytes).ok_or_else(bounds)?
            {
                return verbs::carve_owned_lease_window_outcome(
                    extent.backing.slot().ok_or_else(bounds)?,
                    (address - base) as u64,
                    bytes as u64,
                );
            }
            header = extent.next;
        }
        Err(bounds())
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
// and the cold refill guard permits one publisher. Retained affine owners
// in the failed cell are mutated only under the refill guard or exclusive
// self. Allocation readers see only release-published extents. Exclusive Drop
// moves published owners out before revocation.
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
                backing,
                view,
                ..
            } = unsafe { header.read() };
            header = next;
            // Use the same exact-unshare-before-backing order as an extent
            // that failed installation, including on a refused unshare.
            drop(FailedExtent {
                backing: Some(backing),
                view,
                uncertain: None,
            });
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
        struct Refusing;
        impl MmClient for Refusing {
            async fn request(
                &mut self,
                _: crate::MmRequest,
            ) -> (MmResponse, Option<kore_rt::cap::ReturnedCapability>) {
                (MmResponse::Refused(crate::MmRefusal::Quota), None)
            }
            async fn supply(&mut self, _: u64) -> (MmResponse, Option<StashedLease>) {
                (MmResponse::Refused(crate::MmRefusal::Quota), None)
            }
            async fn supply_observed(
                &mut self,
                bytes: u64,
                _: Option<u64>,
            ) -> Result<(MmResponse, Option<StashedLease>), crate::SupplyError> {
                Ok(self.supply(bytes).await)
            }
            async fn resume_supply_bounded(
                &mut self,
                _: u64,
                _: u64,
            ) -> Result<(MmResponse, Option<StashedLease>), kore_rt::svc::CallError> {
                Err(kore_rt::svc::CallError::NoOutstanding)
            }
            async fn stats(&mut self) -> Option<kore_abi::monitor::MemorySnapshot> {
                None
            }
        }
        let source = MmSource::new();
        assert!(source.map_extent(usize::MAX / 2).is_none());
        let mut client = Refusing;
        let mut refill = core::pin::pin!(source.refill(&mut client));
        let mut context = core::task::Context::from_waker(core::task::Waker::noop());
        assert_eq!(
            core::future::Future::poll(refill.as_mut(), &mut context),
            core::task::Poll::Ready(false)
        );
        assert_eq!(source.wanted.load(Ordering::Acquire), 0);
        assert!(source.map_extent(ALIGN).is_none());
        assert_eq!(source.wanted.load(Ordering::Acquire), ALIGN);
    }

    #[derive(Clone, Copy)]
    enum SupplyStep {
        BeforePublication,
        Suspend,
        Pending,
        NotStarted,
        Refuse,
    }

    #[derive(Clone, Copy)]
    enum ResumeStep {
        Timeout,
        Refuse,
    }

    struct RefillClient {
        supply: SupplyStep,
        resume: ResumeStep,
        active: Option<u64>,
        starts: usize,
        resumes: usize,
        requested: u64,
    }

    impl RefillClient {
        fn new(supply: SupplyStep) -> Self {
            Self {
                supply,
                resume: ResumeStep::Refuse,
                active: None,
                starts: 0,
                resumes: 0,
                requested: 0,
            }
        }
    }

    impl MmClient for RefillClient {
        async fn request(
            &mut self,
            _: crate::MmRequest,
        ) -> (MmResponse, Option<kore_rt::cap::ReturnedCapability>) {
            panic!("heap refill must use the observed supply contract")
        }
        async fn supply(&mut self, _: u64) -> (MmResponse, Option<StashedLease>) {
            panic!("heap refill must not use the legacy panicking convenience")
        }
        async fn supply_observed(
            &mut self,
            bytes: u64,
            _: Option<u64>,
        ) -> Result<(MmResponse, Option<StashedLease>), SupplyError> {
            assert!(
                self.active.is_none(),
                "a second supply overtook the original"
            );
            self.starts += 1;
            self.requested = bytes;
            match self.supply {
                SupplyStep::BeforePublication => core::future::pending().await,
                SupplyStep::NotStarted => Err(SupplyError::NotStarted(CallError::Outstanding)),
                SupplyStep::Refuse => Ok((MmResponse::Refused(crate::MmRefusal::Quota), None)),
                SupplyStep::Pending => {
                    self.active = Some(bytes);
                    Err(SupplyError::Pending(CallError::Transport))
                }
                SupplyStep::Suspend => {
                    self.active = Some(bytes);
                    core::future::pending().await
                }
            }
        }
        async fn resume_supply_bounded(
            &mut self,
            bytes: u64,
            budget_ns: u64,
        ) -> Result<(MmResponse, Option<StashedLease>), CallError> {
            self.resumes += 1;
            assert!(budget_ns > 0);
            let Some(active) = self.active else {
                return Err(CallError::NoOutstanding);
            };
            assert_eq!(bytes, active, "resume changed the pending supply size");
            match self.resume {
                ResumeStep::Timeout => Err(CallError::Timeout),
                ResumeStep::Refuse => {
                    self.active = None;
                    Ok((MmResponse::Refused(crate::MmRefusal::Quota), None))
                }
            }
        }
        async fn stats(&mut self) -> Option<kore_abi::monitor::MemorySnapshot> {
            None
        }
    }

    fn poll_refill(future: impl core::future::Future<Output = bool>) -> core::task::Poll<bool> {
        let mut future = core::pin::pin!(future);
        let mut cx = core::task::Context::from_waker(core::task::Waker::noop());
        core::future::Future::poll(future.as_mut(), &mut cx)
    }

    #[test]
    fn cancelled_refill_resumes_original_size_before_serving_a_larger_miss() {
        use core::task::Poll;
        let source = MmSource::new();
        let mut client = RefillClient::new(SupplyStep::Suspend);
        assert!(source.map_extent(ALIGN).is_none());
        assert_eq!(poll_refill(source.refill(&mut client)), Poll::Pending);
        let original = extent_bytes(ALIGN).unwrap();
        assert_eq!(source.pending_supply.load(Ordering::Relaxed), original);
        assert!(!source.refilling.load(Ordering::Acquire));
        assert!(source.map_extent(4 * ALIGN).is_none());
        assert_eq!(poll_refill(source.refill(&mut client)), Poll::Ready(false));
        assert_eq!((client.starts, client.resumes), (1, 1));
        assert_eq!(source.pending_supply.load(Ordering::Relaxed), 0);
        assert_eq!(source.wanted.load(Ordering::Acquire), 4 * ALIGN);
        client.supply = SupplyStep::Refuse;
        assert_eq!(poll_refill(source.refill(&mut client)), Poll::Ready(false));
        assert_eq!((client.starts, client.resumes), (2, 1));
        assert_eq!(client.requested, extent_bytes(4 * ALIGN).unwrap() as u64);
        assert_eq!(source.wanted.load(Ordering::Acquire), 0);
    }

    #[test]
    fn refill_transport_failure_and_timeout_never_republish_the_supply() {
        use core::task::Poll;
        let source = MmSource::new();
        let mut client = RefillClient::new(SupplyStep::Pending);
        assert!(source.map_extent(ALIGN).is_none());
        assert_eq!(poll_refill(source.refill(&mut client)), Poll::Ready(false));
        client.resume = ResumeStep::Timeout;
        assert_eq!(poll_refill(source.refill(&mut client)), Poll::Ready(false));
        assert_eq!((client.starts, client.resumes), (1, 1));
        assert_ne!(source.pending_supply.load(Ordering::Relaxed), 0);
        client.resume = ResumeStep::Refuse;
        assert_eq!(poll_refill(source.refill(&mut client)), Poll::Ready(false));
        assert_eq!((client.starts, client.resumes), (1, 2));
        assert_eq!(source.pending_supply.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn cancellation_before_publication_requires_idle_proof_before_a_new_supply() {
        use core::task::Poll;
        let source = MmSource::new();
        let mut client = RefillClient::new(SupplyStep::BeforePublication);
        assert!(source.map_extent(ALIGN).is_none());
        assert_eq!(poll_refill(source.refill(&mut client)), Poll::Pending);
        assert_eq!(poll_refill(source.refill(&mut client)), Poll::Ready(false));
        assert_eq!((client.starts, client.resumes), (1, 1));
        assert_eq!(source.pending_supply.load(Ordering::Relaxed), 0);
        assert_eq!(source.wanted.load(Ordering::Acquire), ALIGN);
        client.supply = SupplyStep::Refuse;
        assert_eq!(poll_refill(source.refill(&mut client)), Poll::Ready(false));
        assert_eq!((client.starts, client.resumes), (2, 1));
    }

    #[test]
    fn a_foreign_pending_operation_is_not_adopted_as_a_heap_refill() {
        use core::task::Poll;
        let source = MmSource::new();
        let mut client = RefillClient::new(SupplyStep::NotStarted);
        assert!(source.map_extent(ALIGN).is_none());
        assert_eq!(poll_refill(source.refill(&mut client)), Poll::Ready(false));
        assert_eq!(source.pending_supply.load(Ordering::Relaxed), 0);
        assert_eq!(source.wanted.load(Ordering::Acquire), ALIGN);
        client.supply = SupplyStep::Refuse;
        assert_eq!(poll_refill(source.refill(&mut client)), Poll::Ready(false));
        assert_eq!((client.starts, client.resumes), (2, 0));
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
