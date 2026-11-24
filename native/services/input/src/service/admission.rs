//! Boot-frozen, description-only I2C admission snapshot. The input owner uses
//! its existing broker faces before worker creation; steady-state status reads
//! never need device authority, AML evaluation, or a device-register access.

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicU8, Ordering};

use kore_abi::device::{
    I2C_ADMISSION_BUILD_ROW, I2C_ADMISSION_PAGES_PER_CANDIDATE, I2cAdmissionPage,
    I2cAdmissionSnapshot, MAX_I2C_ADMISSION_CANDIDATES, MAX_ROOT_COMPLEXES,
};
use kore_rt::transport::Transport;
use kore_svc_device::client::{DeviceClient, MultiBroker};

struct Published {
    /// 0 untouched, 1 sole writer, 2 immutable complete snapshot, 3 unavailable.
    phase: AtomicU8,
    value: UnsafeCell<MaybeUninit<I2cAdmissionSnapshot>>,
}

// SAFETY: one successful 0->1 writer initializes value before release-publishing
// phase2. Readers acquire phase2, and the value is never changed afterward.
unsafe impl Sync for Published {}

impl Published {
    const fn new() -> Self {
        Self {
            phase: AtomicU8::new(0),
            value: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }
    fn begin(&self) -> bool {
        self.phase
            .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }
    fn finish(&self, snapshot: Option<I2cAdmissionSnapshot>) {
        if let Some(snapshot) = snapshot {
            // SAFETY: only the successful begin caller can reach this write.
            unsafe { (*self.value.get()).write(snapshot) };
            self.phase.store(2, Ordering::Release);
        } else {
            self.phase.store(3, Ordering::Release);
        }
    }
    fn get(&self) -> Option<&I2cAdmissionSnapshot> {
        if self.phase.load(Ordering::Acquire) != 2 {
            return None;
        }
        // SAFETY: the acquire observed initialization, and no future writer exists.
        Some(unsafe { (&*self.value.get()).assume_init_ref() })
    }
}

static ADMISSION: Published = Published::new();

/// Called once by the input owner before claims or worker spawning. A failed
/// read leaves diagnostics unavailable, not a fabricated empty machine; input
/// startup itself proceeds through its existing claim/error path.
pub async fn collect<T: Transport>(broker: &mut MultiBroker<'static, '_, T>) {
    if !ADMISSION.begin() {
        return;
    }
    let mut combined = I2cAdmissionSnapshot::default();
    let mut roots = 0usize;
    for root in 0..MAX_ROOT_COMPLEXES {
        let Some(face) = broker.face(root) else {
            continue;
        };
        let Some(snapshot) = collect_root(face).await else {
            ADMISSION.finish(None);
            return;
        };
        if !merge_root(&mut combined, &snapshot, roots == 0) {
            ADMISSION.finish(None);
            return;
        }
        roots += 1;
    }
    ADMISSION.finish((roots != 0).then_some(combined));
}

async fn collect_root(face: &mut impl DeviceClient) -> Option<I2cAdmissionSnapshot> {
    let mut snapshot = I2cAdmissionSnapshot::default();
    let summary = face.inspect_i2c(0).await.ok()?;
    if summary.row != 0 || !snapshot.insert(summary) {
        return None;
    }
    for row in 1..=snapshot.candidates * I2C_ADMISSION_PAGES_PER_CANDIDATE {
        let page = face.inspect_i2c(row).await.ok()?;
        if page.row != row || !snapshot.insert(page) {
            return None;
        }
    }
    // BuildRefused explicitly advertises the appended page, independently of
    // the candidate count. Never probe an older producer's unknown row: its
    // decoder may silently discard it rather than return a refusal.
    if snapshot.inventory_state == kore_abi::device::I2cInventoryState::BuildRefused as u8 {
        if let Ok(page) = face.inspect_i2c(I2C_ADMISSION_BUILD_ROW).await {
            if page.row != I2C_ADMISSION_BUILD_ROW || !snapshot.insert(page) {
                return None;
            }
        }
    }
    snapshot.validate().then_some(snapshot)
}

/// Global firmware census is repeated by every root, not summed. PCI identities
/// are segment+BDF scoped, and repeated contradictory identity is a failed
/// observation rather than silently replacing an earlier controller.
fn merge_root(
    combined: &mut I2cAdmissionSnapshot,
    root: &I2cAdmissionSnapshot,
    first: bool,
) -> bool {
    if !root.validate() {
        return false;
    }
    if first {
        combined.inventory_state = root.inventory_state;
        combined.not_initialized = root.not_initialized;
        combined.crs_refused = root.crs_refused;
        combined.build = root.build;
    } else if combined.inventory_state != root.inventory_state
        || combined.not_initialized != root.not_initialized
        || combined.crs_refused != root.crs_refused
    {
        return false;
    }
    // A missing optional page is unknown, not contradictory. Two supplied
    // failure records must describe the same global interpreter construction.
    if root.build.is_present() {
        if combined.build.is_present() && combined.build != root.build {
            return false;
        }
        combined.build = root.build;
    }
    for candidate in &root.candidate[..usize::from(root.candidates)] {
        if combined.candidate[..usize::from(combined.candidates)]
            .iter()
            .any(|old| old.segment == candidate.segment && old.locator == candidate.locator)
        {
            // Distinct RootComplex inventories are structurally disjoint.
            return false;
        }
        if usize::from(combined.candidates) < MAX_I2C_ADMISSION_CANDIDATES {
            combined.candidate[usize::from(combined.candidates)] = *candidate;
            combined.candidates += 1;
        }
    }
    combined.total_candidates.set(
        combined
            .total_candidates
            .get()
            .saturating_add(root.total_candidates.get()),
    );
    true
}

pub fn available() -> bool {
    ADMISSION.get().is_some()
}

pub fn diagnostic(row: u8) -> Option<I2cAdmissionPage> {
    ADMISSION.get()?.page(row)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture {
        snapshot: I2cAdmissionSnapshot,
        extra: Option<I2cAdmissionPage>,
        requests: [u8; kore_abi::device::I2C_ADMISSION_ROWS as usize],
        request_count: usize,
    }

    impl DeviceClient for Fixture {
        async fn inspect_i2c(
            &mut self,
            row: u8,
        ) -> Result<I2cAdmissionPage, kore_svc_device::DeviceRefusal> {
            self.requests[self.request_count] = row;
            self.request_count += 1;
            let page = if row == I2C_ADMISSION_BUILD_ROW {
                assert_eq!(
                    self.snapshot.inventory_state,
                    kore_abi::device::I2cInventoryState::BuildRefused as u8,
                    "legacy decoders would silently drop this unknown request"
                );
                self.extra
            } else {
                self.snapshot.page(row)
            };
            page.ok_or(kore_svc_device::DeviceRefusal::Denied)
        }

        async fn claim(
            &mut self,
            _: kore_svc_device::service::DeviceRequest,
        ) -> Result<kore_svc_device::client::Claimed, kore_svc_device::DeviceRefusal> {
            panic!("diagnostics must never claim a device")
        }

        async fn inspect(
            &mut self,
            _: kore_rt::service::DeviceSelector,
        ) -> Result<kore_svc_device::service::DeviceObservation, kore_svc_device::DeviceRefusal>
        {
            panic!("indexed admission only")
        }

        async fn release(&mut self, _: u32) -> bool {
            panic!("diagnostics must never release a device")
        }
        async fn count(&mut self) -> Result<u32, kore_svc_device::DeviceRefusal> {
            panic!("summary already carries the bound")
        }
    }

    fn ready<F: core::future::Future>(future: F) -> F::Output {
        let mut future = core::pin::pin!(future);
        let mut context = core::task::Context::from_waker(core::task::Waker::noop());
        match future.as_mut().poll(&mut context) {
            core::task::Poll::Ready(value) => value,
            core::task::Poll::Pending => panic!("fixture cannot park"),
        }
    }

    fn failed_build(ordinal: u32) -> kore_abi::device::AmlBuildDiagnostic {
        use kore_abi::device::{AmlBuildDiagnostic, AmlBuildErrorClass, AmlBuildStage};
        AmlBuildDiagnostic::failure(
            AmlBuildStage::SsdtLoad,
            AmlBuildErrorClass::Opcode,
            ordinal,
            0x5bff,
        )
        .unwrap()
    }

    #[test]
    fn collection_reads_global_build_evidence_even_with_no_candidates() {
        use kore_abi::device::I2cInventoryState;
        for candidates in [0, 1, MAX_I2C_ADMISSION_CANDIDATES as u8] {
            for (state, supplied) in [
                (I2cInventoryState::BuildRefused, true),
                (I2cInventoryState::BuildRefused, false),
                (I2cInventoryState::TablesUnavailable, false),
                (I2cInventoryState::Initialized, false),
            ] {
                let mut snapshot = root(0, candidates);
                snapshot.inventory_state = state as u8;
                let advertised = state == I2cInventoryState::BuildRefused;
                if advertised {
                    snapshot.build = failed_build(7);
                }
                let mut fixture = Fixture {
                    snapshot,
                    extra: supplied.then(|| snapshot.page(I2C_ADMISSION_BUILD_ROW).unwrap()),
                    requests: [0; kore_abi::device::I2C_ADMISSION_ROWS as usize],
                    request_count: 0,
                };
                let result = ready(collect_root(&mut fixture)).unwrap();
                assert_eq!(result.inventory_state, snapshot.inventory_state);
                assert_eq!(result.candidates, candidates);
                assert_eq!(result.build.is_present(), supplied);
                if supplied {
                    assert_eq!(result.build, snapshot.build);
                }
                assert_eq!(
                    fixture.request_count,
                    usize::from(candidates) * 3 + 1 + usize::from(advertised)
                );
                assert_eq!(
                    fixture.requests[..fixture.request_count].contains(&I2C_ADMISSION_BUILD_ROW),
                    advertised
                );
            }
        }
    }

    #[test]
    fn supplied_wrong_or_malformed_build_page_cannot_be_published() {
        let mut snapshot = root(0, 0);
        snapshot.inventory_state = kore_abi::device::I2cInventoryState::BuildRefused as u8;
        for extra in [
            I2cAdmissionPage {
                row: 0,
                words: [0, 0],
            },
            I2cAdmissionPage {
                row: I2C_ADMISSION_BUILD_ROW,
                words: [0xff, 0],
            },
        ] {
            let mut fixture = Fixture {
                snapshot,
                extra: Some(extra),
                requests: [0; kore_abi::device::I2C_ADMISSION_ROWS as usize],
                request_count: 0,
            };
            assert!(ready(collect_root(&mut fixture)).is_none());
        }
    }

    #[test]
    fn roots_agree_on_one_build_failure_and_optional_absence_is_unknown() {
        let mut first = root(0, 0);
        first.inventory_state = kore_abi::device::I2cInventoryState::BuildRefused as u8;
        let mut second = first;
        second.build = failed_build(7);
        let mut combined = I2cAdmissionSnapshot::default();
        assert!(merge_root(&mut combined, &first, true));
        assert!(merge_root(&mut combined, &second, false));
        assert_eq!(combined.build, second.build);
        assert!(merge_root(&mut combined, &first, false));
        second.build = failed_build(8);
        assert!(!merge_root(&mut combined, &second, false));
    }

    fn root(segment: u16, count: u8) -> I2cAdmissionSnapshot {
        let mut snapshot = I2cAdmissionSnapshot::default();
        snapshot.inventory_state = 1;
        snapshot.candidates = count;
        snapshot.total_candidates.set(u16::from(count));
        snapshot.not_initialized.set(7);
        for (index, candidate) in snapshot
            .candidate
            .iter_mut()
            .take(usize::from(count))
            .enumerate()
        {
            candidate.segment.set(segment);
            candidate.locator.set(index as u16);
        }
        snapshot
    }

    #[test]
    fn roots_preserve_segment_identity_without_multiplying_global_census() {
        let mut combined = I2cAdmissionSnapshot::default();
        assert!(merge_root(&mut combined, &root(0, 2), true));
        assert!(merge_root(&mut combined, &root(1, 2), false));
        assert_eq!(combined.candidates, 4);
        assert_eq!(combined.total_candidates.get(), 4);
        assert_eq!(combined.not_initialized.get(), 7);
        assert_eq!(combined.candidate[2].segment.get(), 1);
    }

    #[test]
    fn truncation_and_unavailable_are_not_empty_success() {
        let mut combined = I2cAdmissionSnapshot::default();
        assert!(merge_root(&mut combined, &root(0, 8), true));
        assert!(merge_root(&mut combined, &root(1, 2), false));
        assert_eq!(combined.candidates, 8);
        assert_eq!(combined.total_candidates.get(), 10);
        let failed = Published::new();
        assert!(failed.begin());
        failed.finish(None);
        assert!(failed.get().is_none());
        assert!(!failed.begin());
    }

    #[test]
    fn publication_is_once_and_conflicting_roots_refuse() {
        let published = Published::new();
        assert!(published.begin());
        published.finish(Some(root(0, 0)));
        assert!(
            published.get().is_some(),
            "an initialized empty root is available"
        );
        assert!(!published.begin());
        let mut combined = root(0, 2);
        assert!(!merge_root(&mut combined, &root(0, 1), false));
        let mut contradictory = root(1, 1);
        contradictory.not_initialized.set(9);
        assert!(!merge_root(&mut combined, &contradictory, false));
    }
}
