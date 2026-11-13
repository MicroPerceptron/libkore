//! `mmap` and `munmap` over a memory capability.
//!
//! `mmap(NULL, len, prot, flags, fd, 0)` on a descriptor for the program's
//! memory supply (`/kore/memory`, [`crate::fd`]) answers `len` bytes of fresh,
//! zeroed memory, as mapping `/dev/zero` does on a POSIX system. Underneath
//! it is three of kore-rt's verbs, the sequence a Wasmtime engine's guest
//! memory already takes (`kore-wasm-platform`):
//!
//! 1. **Carve.** `heap::reserve_from_top` takes a lease off the top of the
//!    supply the allocator grows from the bottom of. The two share cursors, so
//!    a region and `malloc`'s extents never meet, which a raw carve at an
//!    offset of the same capability could not promise.
//! 2. **Place.** The address comes from the free part of the task's link
//!    window (`Rt::link_free_span`), from one process-wide cursor. The caller
//!    never picks it: a hint is ignored and `MAP_FIXED` is refused.
//! 3. **Map.** The whole lease is mapped there.
//!
//! Thread 0 does all three, for whichever thread asked. Every thread that
//! may allocate is founded holding a share of the supply, and `open` and
//! `mmap` hold the caller's access to that share's rights. But a share
//! carries no `REVOKE` (the founder keeps it, to tear the share down), and
//! a region that can be unmapped again is a carve its holder may revoke. The
//! supply's root is thread 0's, so the caller posts its request there and
//! waits, as a filesystem session does. A mapping is therefore the
//! process's: any thread may unmap it.
//!
//! # What `munmap` returns
//!
//! The supply's cursors and the window's cursor only advance: bytes carved
//! and addresses placed are spent for the life of the program. So `munmap`
//! removes the mapping, and a stale access faults, but it *parks* the region:
//! its lease and its address stay with the program, and the next `mmap` of the
//! same length takes a parked region, zeroed again, before anything new is
//! carved. A program that maps and unmaps one size in a loop therefore uses
//! one region. One that maps ever-different sizes spends supply each time,
//! and ends with `ENOMEM`: the table holds [`REGIONS`] regions, and when all
//! are taken a parked one of another length is given up for good.
//!
//! The request rules and the region table are plain data, so the host suite
//! exercises the code the target runs; the verbs are the metal half.

use core::ffi::c_int;

use crate::fd::errno;

/// The mapping granule: kore's user targets pin a 4 KiB page
/// (`max-page-size=4096`), which is also the granule a supply is carved in
/// (`kore_rt::mem_source::ALIGN`), so a region is exactly its pages.
pub(crate) const PAGE: usize = 4096;

/// `<sys/mman.h>`'s protections, musl's values.
pub(crate) const PROT_READ: c_int = 1;
pub(crate) const PROT_WRITE: c_int = 2;
/// `<sys/mman.h>`'s flags, musl's values.
pub(crate) const MAP_SHARED: c_int = 0x01;
pub(crate) const MAP_PRIVATE: c_int = 0x02;
pub(crate) const MAP_FIXED: c_int = 0x10;

/// A request `mmap` can serve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Request {
    /// Whether the mapping is writable.
    pub(crate) write: bool,
    /// The length asked for, in whole pages' bytes.
    pub(crate) asked: usize,
}

/// The whole pages `len` bytes take, or `None` when that overflows.
pub(crate) fn pages(len: usize) -> Option<usize> {
    len.checked_add(PAGE - 1).map(|bytes| bytes & !(PAGE - 1))
}

impl Request {
    /// What `mmap`'s arguments ask for, or the `errno` that refuses them.
    ///
    /// `EINVAL`: no length, an offset (a supply has no positions a caller can
    /// name), neither or both of `MAP_SHARED` and `MAP_PRIVATE`, or a flag
    /// with no meaning here. `ENOTSUP`: `MAP_FIXED` (the kernel's side picks
    /// every address), and any protection but read or read-write: memory a
    /// program wrote is never executable, and a mapping nothing can touch is
    /// not one this API reserves. `ENOMEM`: a length that does not round to
    /// pages.
    ///
    /// `MAP_SHARED` and `MAP_PRIVATE` are both accepted and mean the same:
    /// the region is fresh and this program's alone, so there is no other
    /// mapper for a store to reach or to be kept from. `PROT_WRITE` alone is
    /// read-write, which POSIX permits and every Kore ISA's page tables imply.
    pub(crate) fn of(len: usize, prot: c_int, flags: c_int, offset: i64) -> Result<Self, c_int> {
        if len == 0 || offset != 0 {
            return Err(errno::EINVAL);
        }
        if flags & MAP_FIXED != 0 {
            return Err(errno::ENOTSUP);
        }
        if flags != MAP_SHARED && flags != MAP_PRIVATE {
            return Err(errno::EINVAL);
        }
        let write = match prot {
            PROT_READ => false,
            p if p == PROT_WRITE || p == PROT_READ | PROT_WRITE => true,
            _ => return Err(errno::ENOTSUP),
        };
        let asked = pages(len).ok_or(errno::ENOMEM)?;
        Ok(Self { write, asked })
    }
}

/// How many regions a program may hold, mapped and parked together. A region
/// is a lease and a view in thread 0's arena and a row here; sixteen is the
/// table a Wasmtime engine keeps for the same purpose
/// (`kore_wasm_platform::MAX_MAPPINGS`), small enough to scan, and a program
/// that needs more distinct regions than that at once is better served by
/// one larger mapping it divides itself.
pub(crate) const REGIONS: usize = 16;

/// One region carved for the program: mapped, or parked for the next `mmap`
/// of its length. `L` is its lease and `V` its view, capabilities dropped
/// with the row.
pub(crate) enum Region<L, V> {
    Vacant,
    /// Mapped at `base` for a request of `asked` bytes.
    Mapped {
        lease: L,
        view: V,
        base: usize,
        asked: usize,
    },
    /// Unmapped, keeping its lease and the address it had.
    Parked {
        lease: L,
        base: usize,
        /// The lease's own length, which a map must name exactly.
        len: usize,
        asked: usize,
    },
}

/// A program's regions.
pub(crate) struct Regions<L, V, const N: usize> {
    rows: [Region<L, V>; N],
}

impl<L, V, const N: usize> Regions<L, V, N> {
    pub(crate) const fn new() -> Self {
        Self {
            rows: [const { Region::Vacant }; N],
        }
    }

    /// A row nothing occupies.
    pub(crate) fn vacant(&self) -> Option<usize> {
        self.rows
            .iter()
            .position(|row| matches!(row, Region::Vacant))
    }

    /// A parked region that served a request of exactly `asked` bytes.
    pub(crate) fn parked(&self, asked: usize) -> Option<usize> {
        self.rows
            .iter()
            .position(|row| matches!(row, Region::Parked { asked: was, .. } if *was == asked))
    }

    /// The region mapped at exactly `base`.
    pub(crate) fn mapped(&self, base: usize) -> Option<usize> {
        self.rows
            .iter()
            .position(|row| matches!(row, Region::Mapped { base: at, .. } if *at == base))
    }

    /// Any parked region: what a full table gives up for a new one.
    pub(crate) fn any_parked(&self) -> Option<usize> {
        self.rows
            .iter()
            .position(|row| matches!(row, Region::Parked { .. }))
    }

    /// Take row `at`, leaving it vacant.
    pub(crate) fn take(&mut self, at: usize) -> Region<L, V> {
        core::mem::replace(&mut self.rows[at], Region::Vacant)
    }

    /// Fill row `at`.
    pub(crate) fn put(&mut self, at: usize, region: Region<L, V>) {
        self.rows[at] = region;
    }
}

#[cfg(target_os = "none")]
pub(crate) use metal::{arm, map, map_zeroed, place, posted, serve, unmap};

#[cfg(target_os = "none")]
mod metal {
    use core::cell::UnsafeCell;
    use core::ffi::c_int;
    use core::sync::atomic::{AtomicU32, Ordering};
    use core::time::Duration;

    use kore_ld::place::SharedPlacer;
    use kore_std::runtime::cap::{MappedLeaseView, RetireOutcome, StashedLease, verbs};
    use kore_std::runtime::entry::Rt;
    use kore_std::runtime::futex::{self, WaitOutcome};
    use kore_std::runtime::{pace, thread};

    use super::{PAGE, REGIONS, Region, Regions, Request};
    use crate::fd::{Holding, errno};
    use crate::lock::Lock;

    /// Where regions land: the free part of the task's link window, armed
    /// once by the host before any other thread exists. Unarmed (a program
    /// with no host, or no link window), every `mmap` is `ENOMEM`.
    static WINDOW: SharedPlacer = SharedPlacer::unarmed();

    /// The process's regions, which thread 0 alone touches: the leases and
    /// views in it are capabilities of its arena.
    struct HostOnly<T>(UnsafeCell<T>);
    // SAFETY: only thread 0 forms a reference ([`perform`]'s contract).
    unsafe impl<T> Sync for HostOnly<T> {}
    static REGIONS_HELD: HostOnly<Regions<StashedLease, MappedLeaseView, REGIONS>> =
        HostOnly(UnsafeCell::new(Regions::new()));

    /// Publish the window regions are placed in: `rt`'s free link span
    /// (thread 0's, which is the process's). A task without one leaves the
    /// placer unarmed.
    pub(crate) fn arm(rt: &Rt) {
        if let Ok(span) = rt.link_free_span() {
            let _ = WINDOW.arm(span.free, span.limit - span.free);
        }
    }

    /// An address for `len` bytes in the window regions land in, which is
    /// spent for the life of the program. `None` when the window is unarmed
    /// or has no room.
    pub(crate) fn place(len: usize) -> Option<usize> {
        WINDOW
            .choose(len as u64, PAGE as u64)
            .ok()
            .and_then(|base| usize::try_from(base).ok())
    }

    /// Map `lease` whole at `base`, zeroed, with the request's access.
    ///
    /// The lease is mapped writable to be zeroed whatever was asked: a parked
    /// region holds what its last user left, and a fresh one is zeroed too
    /// rather than trusted to be. A read-only request is then mapped again
    /// without write.
    pub(crate) fn map_zeroed(
        lease: &StashedLease,
        base: usize,
        len: usize,
        write: bool,
    ) -> Option<MappedLeaseView> {
        let slot = lease.slot()?;
        let mut view = verbs::map_owned_lease_at_rw(slot, base as u64, len as u64)?;
        // SAFETY: `view` maps `len` writable bytes at its pointer, which
        // nothing else has been handed yet.
        unsafe { view.as_mut_ptr().write_bytes(0, len) };
        if write {
            return Some(view);
        }
        match view.try_retire_outcome() {
            RetireOutcome::Retired | RetireOutcome::AlreadyDead => {}
            // Still mapped writable: refusing here would strand the view, so
            // the caller gets nothing and the row keeps the lease.
            RetireOutcome::Retryable => return None,
        }
        verbs::map_owned_lease_at(slot, base as u64, len as u64)
    }

    /// One request of thread 0.
    #[derive(Clone, Copy)]
    enum Op {
        Map(Request),
        Unmap { base: usize, len: usize },
    }

    /// Carry `op` out. Thread 0 only: the supply's root, with the `REVOKE`
    /// an unmappable carve needs, is a capability of its arena.
    fn perform(op: Op) -> Result<usize, c_int> {
        // SAFETY: thread 0 is the one caller, and it is not re-entered: no
        // verb below calls back into this module.
        let regions = unsafe { &mut *REGIONS_HELD.0.get() };
        match op {
            Op::Map(request) => map_here(regions, request),
            Op::Unmap { base, len } => unmap_here(regions, base, len).map(|()| 0),
        }
    }

    fn map_here(
        regions: &mut Regions<StashedLease, MappedLeaseView, REGIONS>,
        request: Request,
    ) -> Result<usize, c_int> {
        let (row, lease, base, len) = match regions.parked(request.asked) {
            Some(row) => match regions.take(row) {
                Region::Parked {
                    lease, base, len, ..
                } => (row, lease, base, len),
                // `parked` answered this row.
                _ => return Err(errno::ENOMEM),
            },
            None => {
                // The row first: a carve with nowhere to be recorded would
                // spend supply for nothing. A full table gives up a parked
                // region of another length, whose lease ends with its row.
                let row = match regions.vacant() {
                    Some(row) => row,
                    None => {
                        let row = regions.any_parked().ok_or(errno::ENOMEM)?;
                        drop(regions.take(row));
                        row
                    }
                };
                let lease = kore_std::runtime::heap::reserve_from_top(request.asked as u64)
                    .ok_or(errno::ENOMEM)?;
                let len = lease
                    .slot()
                    .and_then(verbs::lease_geometry)
                    .and_then(|(_, len)| usize::try_from(len).ok())
                    .ok_or(errno::ENOMEM)?;
                let base = place(len).ok_or(errno::ENOMEM)?;
                (row, lease, base, len)
            }
        };
        match map_zeroed(&lease, base, len, request.write) {
            Some(view) => {
                regions.put(
                    row,
                    Region::Mapped {
                        lease,
                        view,
                        base,
                        asked: request.asked,
                    },
                );
                Ok(base)
            }
            None => {
                regions.put(
                    row,
                    Region::Parked {
                        lease,
                        base,
                        len,
                        asked: request.asked,
                    },
                );
                Err(errno::ENOMEM)
            }
        }
    }

    fn unmap_here(
        regions: &mut Regions<StashedLease, MappedLeaseView, REGIONS>,
        base: usize,
        len: usize,
    ) -> Result<(), c_int> {
        let asked = super::pages(len).filter(|&bytes| bytes != 0);
        let row = regions.mapped(base).ok_or(errno::EINVAL)?;
        let Region::Mapped {
            lease,
            mut view,
            base,
            asked: was,
        } = regions.take(row)
        else {
            // `mapped` answered this row.
            return Err(errno::EINVAL);
        };
        if asked != Some(was) {
            regions.put(
                row,
                Region::Mapped {
                    lease,
                    view,
                    base,
                    asked: was,
                },
            );
            return Err(errno::EINVAL);
        }
        let len = view.len();
        match view.try_retire_outcome() {
            RetireOutcome::Retired | RetireOutcome::AlreadyDead => {
                regions.put(
                    row,
                    Region::Parked {
                        lease,
                        base,
                        len,
                        asked: was,
                    },
                );
                Ok(())
            }
            RetireOutcome::Retryable => {
                regions.put(
                    row,
                    Region::Mapped {
                        lease,
                        view,
                        base,
                        asked: was,
                    },
                );
                Err(errno::EAGAIN)
            }
        }
    }

    // ── Reaching thread 0 ───────────────────────────────────────────────

    // The cell's states. A caller owns it at IDLE (under [`CALLERS`]) and at
    // SERVED, thread 0 at POSTED.
    const IDLE: u32 = 0;
    const POSTED: u32 = 1;
    const SERVED: u32 = 2;

    /// One request at a time: the caller holding [`CALLERS`] writes `op` and
    /// posts, thread 0 writes `answer` and serves.
    struct Call {
        state: AtomicU32,
        op: UnsafeCell<Op>,
        answer: UnsafeCell<Result<usize, c_int>>,
    }
    // SAFETY: `op` and `answer` are touched only by the side `state` hands
    // the cell to.
    unsafe impl Sync for Call {}
    static CALL: Call = Call {
        state: AtomicU32::new(IDLE),
        op: UnsafeCell::new(Op::Unmap { base: 0, len: 0 }),
        answer: UnsafeCell::new(Ok(0)),
    };

    /// Callers take turns at the cell.
    static CALLERS: Lock = Lock::new();

    /// How long a caller waits before it re-reads the cell: thread 0 kicks
    /// the main thread when it has answered, and holds no send into a
    /// pthread's doorbell, which rides this slice (the streams' rule).
    const SLICE: Duration = Duration::from_nanos(pace::CEDE_NS);

    /// Whether a request waits for thread 0.
    pub(crate) fn posted() -> bool {
        CALL.state.load(Ordering::Acquire) == POSTED
    }

    /// Thread 0: answer the posted request, if there is one. `true` when one
    /// was answered, and its caller should be woken.
    pub(crate) fn serve() -> bool {
        if !posted() {
            return false;
        }
        // SAFETY: POSTED hands the cell to thread 0 until it stores SERVED.
        let answer = perform(unsafe { *CALL.op.get() });
        // SAFETY: as above.
        unsafe { *CALL.answer.get() = answer };
        // SeqCst, as the futex contract asks of the store before a wake.
        CALL.state.store(SERVED, Ordering::SeqCst);
        futex::wake(&CALL.state);
        true
    }

    /// Have thread 0 carry `op` out, and wait for its answer.
    ///
    /// Thread 0 itself (a filesystem callback, a program with no host) does
    /// it in place. Any other thread of a program without a host has no one
    /// to ask: `ENOMEM`.
    fn ask(op: Op) -> Result<usize, c_int> {
        if crate::program::on_host() {
            return perform(op);
        }
        if !crate::program::hosted() {
            return Err(errno::ENOMEM);
        }
        CALLERS.acquire();
        // SAFETY: IDLE under the lock hands the cell to this caller.
        unsafe { *CALL.op.get() = op };
        CALL.state.store(POSTED, Ordering::SeqCst);
        // Refused only when no post stood, and then thread 0 re-reads the
        // cell before it parks.
        let _ = thread::kick_parent();
        loop {
            match CALL.state.load(Ordering::Acquire) {
                SERVED => break,
                state => {
                    if futex::wait(&CALL.state, state, Some(SLICE)) == WaitOutcome::Unsupported {
                        // No doorbell to park on: the deadline park any
                        // thread can.
                        let _ = pace::cede();
                    }
                }
            }
        }
        // SAFETY: SERVED hands the cell back to this caller.
        let answer = unsafe { *CALL.answer.get() };
        CALL.state.store(IDLE, Ordering::Release);
        CALLERS.release();
        answer
    }

    /// `mmap` on a memory supply the calling thread holds as `held`.
    ///
    /// The caller's own share is what `open` and this call hold the access
    /// to. The region is carved by thread 0, which holds the supply's root:
    /// a share a thread is founded with carries no `REVOKE`, and a carve
    /// that can be unmapped again needs it.
    ///
    /// `EACCES`: a writable mapping of a capability without `MODIFY`.
    /// `ENOMEM`: every row of the table is a live mapping, the supply or the
    /// window has no room, the kernel refused the map, or the program has no
    /// host to carve for this thread. A region whose map is refused is
    /// parked, so its supply bytes and address serve the next request of its
    /// length.
    pub(crate) fn map(held: Holding, request: Request) -> Result<*mut u8, c_int> {
        if request.write && !held.writable {
            return Err(errno::EACCES);
        }
        ask(Op::Map(request)).map(|base| base as *mut u8)
    }

    /// `munmap`: remove the mapping at `base`, which `len` must name whole,
    /// and park its region.
    ///
    /// `EINVAL`: the program has no mapping at exactly `base`, or `len` is
    /// not that mapping's length. `EAGAIN`: the kernel did not confirm the
    /// unmap; the mapping stands and the call may be made again.
    pub(crate) fn unmap(base: usize, len: usize) -> Result<(), c_int> {
        ask(Op::Unmap { base, len }).map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RW: c_int = PROT_READ | PROT_WRITE;

    #[test]
    fn a_length_is_whole_pages() {
        assert_eq!(pages(1), Some(PAGE));
        assert_eq!(pages(PAGE), Some(PAGE));
        assert_eq!(pages(PAGE + 1), Some(2 * PAGE));
        assert_eq!(pages(usize::MAX), None);
        assert_eq!(pages(usize::MAX - PAGE + 2), None);
        assert_eq!(pages(usize::MAX - PAGE + 1), Some(usize::MAX - PAGE + 1));
    }

    #[test]
    fn a_readable_region_of_the_supply_is_served() {
        assert_eq!(
            Request::of(1 << 20, RW, MAP_PRIVATE, 0),
            Ok(Request {
                write: true,
                asked: 1 << 20
            })
        );
        assert_eq!(
            Request::of(1, PROT_READ, MAP_SHARED, 0),
            Ok(Request {
                write: false,
                asked: PAGE
            })
        );
        // Write alone is read-write.
        assert_eq!(
            Request::of(PAGE, PROT_WRITE, MAP_SHARED, 0),
            Ok(Request {
                write: true,
                asked: PAGE
            })
        );
    }

    #[test]
    fn what_has_no_meaning_here_is_refused_by_name() {
        // No length, and an offset into a supply.
        assert_eq!(Request::of(0, RW, MAP_PRIVATE, 0), Err(errno::EINVAL));
        assert_eq!(
            Request::of(PAGE, RW, MAP_PRIVATE, PAGE as i64),
            Err(errno::EINVAL)
        );
        assert_eq!(Request::of(PAGE, RW, MAP_PRIVATE, -1), Err(errno::EINVAL));
        // Neither, both, or a stranger: MAP_ANONYMOUS is 0x20.
        assert_eq!(Request::of(PAGE, RW, 0, 0), Err(errno::EINVAL));
        assert_eq!(
            Request::of(PAGE, RW, MAP_SHARED | MAP_PRIVATE, 0),
            Err(errno::EINVAL)
        );
        assert_eq!(
            Request::of(PAGE, RW, MAP_PRIVATE | 0x20, 0),
            Err(errno::EINVAL)
        );
        // The caller never picks the address.
        assert_eq!(
            Request::of(PAGE, RW, MAP_PRIVATE | MAP_FIXED, 0),
            Err(errno::ENOTSUP)
        );
        // No access, executable (4), and bits past the three.
        for prot in [0, 4, PROT_READ | 4, RW | 4, 8, -1] {
            assert_eq!(
                Request::of(PAGE, prot, MAP_PRIVATE, 0),
                Err(errno::ENOTSUP),
                "{prot}"
            );
        }
        assert_eq!(
            Request::of(usize::MAX, RW, MAP_PRIVATE, 0),
            Err(errno::ENOMEM)
        );
    }

    type Rows = Regions<u32, u32, 3>;

    fn mapped(lease: u32, base: usize, asked: usize) -> Region<u32, u32> {
        Region::Mapped {
            lease,
            view: lease,
            base,
            asked,
        }
    }

    fn park(rows: &mut Rows, base: usize) {
        let row = rows.mapped(base).unwrap();
        let Region::Mapped {
            lease, base, asked, ..
        } = rows.take(row)
        else {
            unreachable!()
        };
        rows.put(
            row,
            Region::Parked {
                lease,
                base,
                len: asked,
                asked,
            },
        );
    }

    #[test]
    fn an_unmapped_region_serves_the_next_request_of_its_length() {
        let mut rows = Rows::new();
        assert_eq!(rows.vacant(), Some(0));
        rows.put(0, mapped(7, 0x1000, PAGE));
        rows.put(1, mapped(8, 0x9000, 2 * PAGE));
        assert_eq!(rows.vacant(), Some(2));
        assert_eq!((rows.parked(PAGE), rows.any_parked()), (None, None));
        park(&mut rows, 0x1000);
        // Parked, it is no longer a mapping, and only its own length takes it.
        assert_eq!(rows.mapped(0x1000), None);
        assert_eq!(rows.parked(2 * PAGE), None);
        assert_eq!(rows.parked(PAGE), Some(0));
        let Region::Parked { lease, base, .. } = rows.take(0) else {
            panic!("the parked row")
        };
        // The same lease at the same address.
        assert_eq!((lease, base), (7, 0x1000));
        assert_eq!(rows.vacant(), Some(0));
    }

    #[test]
    fn a_mapping_is_found_by_its_base_alone() {
        let mut rows = Rows::new();
        rows.put(1, mapped(1, 0x4000, 4 * PAGE));
        assert_eq!(rows.mapped(0x4000), Some(1));
        // An address inside it is not its base.
        assert_eq!(rows.mapped(0x5000), None);
        assert_eq!(rows.mapped(0), None);
    }

    #[test]
    fn a_full_table_has_no_vacant_row() {
        let mut rows = Rows::new();
        for at in 0..3 {
            rows.put(at, mapped(at as u32, 0x1000 * (at + 1), PAGE));
        }
        assert_eq!(rows.vacant(), None);
        park(&mut rows, 0x2000);
        // A parked row is still occupied: it holds a lease. It is what a
        // request of another length takes the place of.
        assert_eq!(rows.vacant(), None);
        assert_eq!(rows.parked(PAGE), Some(1));
        assert_eq!(rows.parked(2 * PAGE), None);
        assert_eq!(rows.any_parked(), Some(1));
        assert!(matches!(rows.take(1), Region::Parked { lease: 1, .. }));
        assert_eq!((rows.vacant(), rows.any_parked()), (Some(1), None));
    }
}
