//! Exact staging-map custody. A cell is reusable only after both retirement
//! phases close; a failed map still retains its parent until cleanup.

use kore_abi::task::layout::Window;
use kore_rt::cap::{MappedLeaseView, StashedMemory, verbs};

const ALIGN: u64 = 2 * 1024 * 1024;
/// Leave the beginning of TRANSIENT available for recyclable protocol rings.
const RING_RESERVE: u64 = ALIGN;

/// One recyclable peer's protocol-ring window: [`RING_RESERVE`] split evenly
/// across [`PAIRS`](super::PAIRS).
///
/// It therefore **shrinks when the runtime's pair roster grows** — it halved
/// when `MAX_SERVE_PAIRS` went from four to eight, and a client that leased a
/// fixed 512 KiB ring was refused `RingGeometry` from that moment, because the
/// admission requires `ring_bytes <= plan.stride`. Clients lease this, never a
/// literal of their own.
pub const PEER_RING_BYTES: u64 = RING_RESERVE / super::PAIRS as u64;

const _: () = {
    // `PeerMapPlan::address` refuses a stride that is not page aligned.
    assert!(
        PEER_RING_BYTES != 0 && PEER_RING_BYTES % 4096 == 0,
        "a peer ring window must be a non-zero page multiple"
    );
    // And the admission requires `ring_bytes >= window_bytes::<S>(..)`, so a
    // roster that grows past what RING_RESERVE can divide must raise the
    // reserve rather than hand out windows the protocol cannot live in.
    assert!(
        PEER_RING_BYTES as usize
            >= kore_rt::svc::window_bytes::<crate::wire::AccelWire>(
                <crate::wire::AccelWire as kore_rt::svc::SvcProtocol>::REQUEST_CAPACITY,
                <crate::wire::AccelWire as kore_rt::svc::SvcProtocol>::RESPONSE_CAPACITY,
            ),
        "RING_RESERVE / PAIRS no longer holds this protocol's pair window"
    );
};

pub(super) fn cell(window: Window, via: u16) -> Option<Window> {
    if usize::from(via) >= super::PAIRS || window.base() % ALIGN != 0 {
        return None;
    }
    let stride = (window.len().checked_sub(RING_RESERVE)? / super::PAIRS as u64) & !(ALIGN - 1);
    if stride == 0 {
        return None;
    }
    let base = window
        .base()
        .checked_add(RING_RESERVE)?
        .checked_add(stride.checked_mul(u64::from(via))?)?;
    let result = Window::try_new(base, stride)?;
    window.contains(&result).then_some(result)
}

pub(super) trait Retire {
    fn retire(&mut self) -> bool;
}

impl Retire for MappedLeaseView {
    fn retire(&mut self) -> bool {
        self.try_retire_outcome().is_closed()
    }
}

impl Retire for StashedMemory {
    fn retire(&mut self) -> bool {
        self.try_retire_outcome().is_closed()
    }
}

pub(super) struct StagingOwners<M: Retire, P: Retire> {
    mapping: Option<M>,
    parent: Option<P>,
}

pub(super) type OwnedStaging = StagingOwners<MappedLeaseView, StashedMemory>;

impl OwnedStaging {
    pub(super) fn map(parent: StashedMemory, base: u64, len: u64) -> Self {
        let mapping = match &parent {
            StashedMemory::Lease(lease) => lease
                .slot()
                .and_then(|slot| verbs::map_owned_lease_at_rw(slot, base, len)),
            StashedMemory::Route(route) => verbs::map_owned_route_at_rw(route, base, len),
        };
        Self {
            mapping,
            parent: Some(parent),
        }
    }

    pub(super) fn window(&self) -> Option<(*mut u8, usize)> {
        self.mapping
            .as_ref()
            .map(|view| (view.as_mut_ptr(), view.len()))
    }
}

impl<M: Retire, P: Retire> StagingOwners<M, P> {
    /// The caller must discard every borrow of the mapped bytes first.
    pub(super) fn retire(&mut self) -> bool {
        if let Some(mapping) = self.mapping.as_mut() {
            if !mapping.retire() {
                return false;
            }
            self.mapping = None;
        }
        if let Some(parent) = self.parent.as_mut() {
            if !parent.retire() {
                return false;
            }
            self.parent = None;
        }
        true
    }
}

impl<M: Retire, P: Retire> Drop for StagingOwners<M, P> {
    fn drop(&mut self) {
        if !self.retire() {
            // Task teardown is the fallback. Never let an ordinary field Drop
            // retire the parent after exact-unshare refused, or free a dynamic
            // arena name while its mapping remains. Live cells keep this owner
            // for retries instead of relying on Drop.
            core::mem::forget(self.mapping.take());
            core::mem::forget(self.parent.take());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{rc::Rc, vec::Vec};
    use core::cell::RefCell;

    struct Owner {
        phase: u8,
        refusals: u8,
        log: Rc<RefCell<Vec<u8>>>,
    }
    impl Retire for Owner {
        fn retire(&mut self) -> bool {
            self.log.borrow_mut().push(self.phase);
            if self.refusals > 0 {
                self.refusals -= 1;
                false
            } else {
                true
            }
        }
    }

    #[test]
    fn retirement_retries_preserve_order_and_the_same_owners() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut owners = StagingOwners {
            mapping: Some(Owner {
                phase: 1,
                refusals: 1,
                log: log.clone(),
            }),
            parent: Some(Owner {
                phase: 2,
                refusals: 1,
                log: log.clone(),
            }),
        };
        assert!(!owners.retire());
        assert_eq!(&*log.borrow(), &[1]);
        assert!(owners.mapping.is_some() && owners.parent.is_some());
        assert!(!owners.retire());
        assert_eq!(&*log.borrow(), &[1, 1, 2]);
        assert!(owners.mapping.is_none() && owners.parent.is_some());
        assert!(owners.retire());
        assert_eq!(&*log.borrow(), &[1, 1, 2, 2]);
        assert!(owners.retire());
        assert_eq!(&*log.borrow(), &[1, 1, 2, 2]);
    }

    #[test]
    fn failed_mapping_still_requires_parent_retirement() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let mut owners: StagingOwners<Owner, Owner> = StagingOwners {
            mapping: None,
            parent: Some(Owner {
                phase: 2,
                refusals: 1,
                log: log.clone(),
            }),
        };
        assert!(!owners.retire());
        assert!(owners.parent.is_some());
        assert!(owners.retire());
        assert_eq!(&*log.borrow(), &[2, 2]);
    }

    #[test]
    fn cells_are_disjoint_bounded_and_clear_of_ring_prefix() {
        let window = Window::new(8 << 30, 64 << 30);
        let map = peer_map(window);
        assert_eq!(map.base, window.base());
        assert_eq!(map.stride % 4096, 0);
        assert_eq!(map.stride * super::super::PAIRS as u64, RING_RESERVE);
        let mut prior_end = window.base() + RING_RESERVE;
        for via in 0..super::super::PAIRS as u16 {
            let slot = cell(window, via).unwrap();
            assert!(slot.base() >= prior_end);
            assert!(window.contains(&slot));
            assert_eq!(slot.base() % ALIGN, 0);
            assert!(slot.len() >= 2 << 30);
            prior_end = slot.base() + slot.len();
        }
        assert!(cell(window, super::super::PAIRS as u16).is_none());
        assert!(cell(Window::new(ALIGN, ALIGN), 0).is_none());
        assert!(cell(Window::new(1, 64 << 30), 0).is_none());
    }
}

/// Called only after the staging window validates; rings stay in its prefix.
pub(super) fn peer_map(window: Window) -> kore_rt::svc::PeerMapPlan {
    kore_rt::svc::PeerMapPlan {
        base: window.base(),
        stride: RING_RESERVE / super::PAIRS as u64,
    }
}
