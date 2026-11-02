//! **Reservations** — address ranges a task reserves in its reservation
//! window (`geom::RESERVE`) and commits page range by page range: a guarded
//! wasm linear memory is one, 4 GiB with a 32 MiB guard on either side, of
//! which only what the guest has grown to is mapped (`res/WASM_PLATFORM.md`
//! §12).
//!
//! A [`Reservation`] is affine and owns its range.  It places the range
//! ([`Reservation::new`]), commits supply carves into it
//! ([`Reservation::commit`]: one session per commit, nothing held after), and
//! on drop releases every commit — one session per supply it drew on — and
//! hands the range back for the next reservation.  The kernel keeps no
//! reservation object: the window's page table is its only state, and a
//! commit over a mapped page refuses there.
//!
//! **One placer per window.**  The placer is this module's, process-wide, so
//! every consumer of the window shares it: two cursors over one window hand
//! out each other's addresses (`kore_ld::place::SharedPlacer`'s warning).
//!
//! **The task's own thread only.**  Commits draw on the supplies the heap
//! holds, which are slots of the task thread's arena (the heap window's
//! refill rule), so a reservation is made, committed and released there and
//! nowhere else; the type is `!Send`.

use core::marker::PhantomData;
#[cfg(target_os = "none")]
use core::sync::atomic::{AtomicPtr, AtomicUsize};
use core::sync::atomic::{AtomicU64, Ordering};

use kore_abi::session::RESERVE_PAGE_BYTES;

use crate::heap::CommitSupply;

/// The placement grain: a reservation's base is aligned to it and its length
/// rounded up to it.  64 MiB divides a guarded wasm memory exactly — two
/// 32 MiB guards and a 4 GiB reservation are 65 of them — and puts each
/// memory's base on a 32 MiB boundary, past any large leaf's alignment.  The
/// 1 TiB window is 16384 granules: a 2 KiB bitmap, allocated by the first
/// reservation, so a program that never reserves pays nothing.
pub const GRANULE: u64 = 64 << 20;

/// Committed ranges one reservation tracks: Wasmtime grows a memory
/// contiguously from its first page, so one range is the steady state, and a
/// commit that would need more than this many refuses rather than allocate.
const COMMITTED_RANGES: usize = 8;

/// Why a reservation could not be made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReserveError {
    /// Off the kore target, or off the task's own thread.
    Unsupported,
    /// The task was stamped no reservation window.
    NoWindow,
    /// No run of free granules is long enough, or the length overflows.
    Full,
}

/// Why a commit did not land.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommitError {
    /// The span is empty, not page-granular, or leaves the reservation.
    OutOfRange,
    /// The committed span would need more ranges than a reservation tracks.
    Fragmented,
    /// No supply this task holds has room for the span.
    NoRoom,
    /// The kernel refused the map; nothing was left mapped.
    Refused,
}

/// **An address range this task reserved**, committed as it grows and
/// released whole on drop.  Thread-bound: see the module docs.
#[must_use = "a dropped reservation releases its commits at once"]
pub struct Reservation {
    /// First byte of the range.
    base: u64,
    /// The range's length, a whole number of [`GRANULE`]s.
    len: u64,
    /// Disjoint committed spans, as offsets from `base`, unordered.
    committed: [Option<(u64, u64)>; COMMITTED_RANGES],
    /// The supplies commits drew on, by [`CommitSupply::index`].
    supplies: [u64; SUPPLY_WORDS],
    _thread_bound: PhantomData<*mut ()>,
}

/// Words of a supply set: one bit per supply a commit may draw on.
const SUPPLY_WORDS: usize = CommitSupply::COUNT.div_ceil(64);

/// Bytes whose release refused: a range still holding a commit is never
/// placed again, so it is counted rather than lost silently.
static LEAKED: AtomicU64 = AtomicU64::new(0);

/// Bytes of the window whose release refused.  Zero on a healthy task.
pub fn leaked_bytes() -> u64 {
    LEAKED.load(Ordering::Relaxed)
}

impl Reservation {
    /// Reserve `len` bytes of this task's reservation window, rounded up to
    /// [`GRANULE`]s and aligned to one.  Nothing is mapped.
    pub fn new(len: u64) -> Result<Self, ReserveError> {
        let (base, window) = placer()?;
        let granules = len.max(1).div_ceil(GRANULE);
        let len = granules.checked_mul(GRANULE).ok_or(ReserveError::Full)?;
        let first = window.claim(usize::try_from(granules).map_err(|_| ReserveError::Full)?);
        let first = first.ok_or(ReserveError::Full)?;
        Ok(Self {
            base: base + first as u64 * GRANULE,
            len,
            committed: [None; COMMITTED_RANGES],
            supplies: [0; SUPPLY_WORDS],
            _thread_bound: PhantomData,
        })
    }

    /// The range's first byte.
    pub fn base(&self) -> u64 {
        self.base
    }

    /// The range's length.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// **Make `[offset, offset + len)` of the range committed**: carve what
    /// of it is not yet off a supply this task holds and map it there,
    /// writable — one [`heap::commit_reserved`](crate::heap::commit_reserved)
    /// per gap, so a span already committed costs nothing.  The span is
    /// page-granular and inside the range.
    ///
    /// All or nothing.  A gap's commit refused partway leaves the prefix the
    /// kernel published, and is released by its own range — which holds no
    /// earlier commit, being a gap — and the gaps this call already filled
    /// are released with it, so a refusal leaves the range as it found it.
    /// A release that itself refuses keeps what it could not remove tracked,
    /// committed and drawn from its supply, so the tracker never claims a
    /// page is free that is mapped, and the drop still releases it.
    pub fn commit(&mut self, offset: u64, len: u64) -> Result<(), CommitError> {
        let end = offset
            .checked_add(len)
            .filter(|&end| len != 0 && end <= self.len)
            .ok_or(CommitError::OutOfRange)?;
        if (offset | len) % RESERVE_PAGE_BYTES != 0 {
            return Err(CommitError::OutOfRange);
        }
        let gaps = gaps(&self.committed, offset, end);
        // Every span the commit leaves must fit the tracker before anything
        // is mapped.
        let mut merged = self.committed;
        for &(at, until) in gaps.iter().flatten() {
            merged = merge(&merged, at, until).ok_or(CommitError::Fragmented)?;
        }
        let mut filled = [None; COMMITTED_RANGES + 1];
        for (i, &(at, until)) in gaps.iter().flatten().enumerate() {
            let outcome = self
                .page(at)
                .zip(u32::try_from((until - at) / RESERVE_PAGE_BYTES).ok())
                .ok_or(CommitError::OutOfRange)
                .and_then(|(page, pages)| {
                    crate::heap::commit_reserved(page, until - at)
                        .map(|supply| (supply, page, pages))
                });
            match outcome {
                Ok((supply, page, pages)) => filled[i] = Some((supply, page, pages, at, until)),
                Err(refusal) => {
                    self.roll_back(&filled);
                    return Err(refusal);
                }
            }
        }
        self.committed = merged;
        for &(supply, ..) in filled.iter().flatten() {
            self.note(supply);
        }
        Ok(())
    }

    /// Release the gaps a refused [`commit`](Self::commit) already filled.
    /// One that will not release stays tracked: mapped, so committed, and
    /// drawn from its supply, which the drop then releases.
    fn roll_back(&mut self, filled: &[Option<(CommitSupply, u32, u32, u64, u64)>]) {
        for &(supply, page, pages, at, until) in filled.iter().flatten() {
            if !crate::heap::release_reserved(supply, page, pages) {
                if let Some(kept) = merge(&self.committed, at, until) {
                    self.committed = kept;
                }
                self.note(supply);
            }
        }
    }

    /// Record that a commit drew on `supply`.
    fn note(&mut self, supply: CommitSupply) {
        self.supplies[supply.index() / 64] |= 1 << (supply.index() % 64);
    }

    /// Whether `[offset, offset + len)` lies wholly inside one committed span.
    pub fn is_committed(&self, offset: u64, len: u64) -> bool {
        offset.checked_add(len).is_some_and(|end| {
            self.committed
                .iter()
                .flatten()
                .any(|&(at, until)| at <= offset && end <= until)
        })
    }

    /// Whether any of `[offset, offset + len)` is committed.
    pub fn meets_committed(&self, offset: u64, len: u64) -> bool {
        let end = offset.saturating_add(len);
        self.committed
            .iter()
            .flatten()
            .any(|&(at, until)| offset < until && at < end)
    }

    /// The window page `offset` falls on.
    fn page(&self, offset: u64) -> Option<u32> {
        let (window, _) = window_span()?;
        u32::try_from((self.base + offset - window) / RESERVE_PAGE_BYTES).ok()
    }
}

impl Drop for Reservation {
    /// Release every commit, once per supply drawn on, then hand the range
    /// back.  A refused release leaves a commit mapped, and the range is
    /// then never placed again (counted in [`leaked_bytes`]): a reservation
    /// placed over a live commit would refuse every commit of its own.
    fn drop(&mut self) {
        let (Some(page), Ok(pages)) = (self.page(0), u32::try_from(self.len / RESERVE_PAGE_BYTES))
        else {
            LEAKED.fetch_add(self.len, Ordering::Relaxed);
            return;
        };
        // Every supply is released even past a refusal: what can go, goes.
        let mut released = true;
        for supply in (0..CommitSupply::COUNT).filter_map(CommitSupply::from_index) {
            let index = supply.index();
            if self.supplies[index / 64] & (1 << (index % 64)) != 0 {
                released &= crate::heap::release_reserved(supply, page, pages);
            }
        }
        if !released {
            LEAKED.fetch_add(self.len, Ordering::Relaxed);
            return;
        }
        if let Ok((base, window)) = placer() {
            window.free(
                ((self.base - base) / GRANULE) as usize,
                (self.len / GRANULE) as usize,
            );
        }
    }
}

/// The parts of `[at, until)` no span of `spans` covers, in address order —
/// at most one more than there are spans.
fn gaps(
    spans: &[Option<(u64, u64)>; COMMITTED_RANGES],
    at: u64,
    until: u64,
) -> [Option<(u64, u64)>; COMMITTED_RANGES + 1] {
    // The spans that meet the range, by start: a handful, sorted in place.
    let mut meeting = [(0u64, 0u64); COMMITTED_RANGES];
    let mut n = 0;
    for &(start, end) in spans.iter().flatten() {
        if start < until && at < end {
            meeting[n] = (start, end);
            n += 1;
        }
    }
    meeting[..n].sort_unstable();
    let mut out = [None; COMMITTED_RANGES + 1];
    let (mut cursor, mut found) = (at, 0);
    for &(start, end) in &meeting[..n] {
        if cursor < start {
            out[found] = Some((cursor, start));
            found += 1;
        }
        cursor = cursor.max(end);
    }
    if cursor < until {
        out[found] = Some((cursor, until));
    }
    out
}

/// `spans` with `[at, until)` merged in, joining the spans it abuts; `None`
/// when no slot is left for it.
fn merge(
    spans: &[Option<(u64, u64)>; COMMITTED_RANGES],
    mut at: u64,
    mut until: u64,
) -> Option<[Option<(u64, u64)>; COMMITTED_RANGES]> {
    let mut out = [None; COMMITTED_RANGES];
    let mut kept = 0;
    for &(start, end) in spans.iter().flatten() {
        if end == at || start == until {
            at = at.min(start);
            until = until.max(end);
        } else {
            out[kept] = Some((start, end));
            kept += 1;
        }
    }
    *out.get_mut(kept)? = Some((at, until));
    Some(out)
}

/// The window's `(base, len)`, from this thread's geometry.
#[cfg(target_os = "none")]
fn window_span() -> Option<(u64, u64)> {
    crate::entry::thread_context()
        .and_then(|rt| rt.geom())
        .and_then(|g| g.window(kore_abi::task::geom::RESERVE))
}

/// Off metal there is no geometry.
#[cfg(not(target_os = "none"))]
fn window_span() -> Option<(u64, u64)> {
    None
}

/// The process's granule bitmap over the window, allocated by the first
/// reservation; null until then.
#[cfg(target_os = "none")]
static BITMAP: AtomicPtr<AtomicU64> = AtomicPtr::new(core::ptr::null_mut());
/// The bitmap's length in granules (zero until allocated).
#[cfg(target_os = "none")]
static BITMAP_GRANULES: AtomicUsize = AtomicUsize::new(0);

/// The window's base and its granule bitmap — this thread must be the
/// task's own, in a task stamped a window.
#[cfg(target_os = "none")]
fn placer() -> Result<(u64, Granules<'static>), ReserveError> {
    if crate::entry::geometry::shard_index() != Some(0) {
        return Err(ReserveError::Unsupported);
    }
    let (base, len) = window_span().ok_or(ReserveError::NoWindow)?;
    let granules = usize::try_from(len / GRANULE).map_err(|_| ReserveError::NoWindow)?;
    if base % GRANULE != 0 || granules == 0 {
        return Err(ReserveError::NoWindow);
    }
    let mut words = BITMAP.load(Ordering::Acquire);
    if words.is_null() {
        let fresh: alloc::boxed::Box<[AtomicU64]> = (0..granules.div_ceil(64))
            .map(|_| AtomicU64::new(0))
            .collect();
        let fresh = alloc::boxed::Box::into_raw(fresh).cast::<AtomicU64>();
        // The task's own thread is the only placer, so the exchange cannot
        // lose; it is one anyway, so a lost race would only free its own.
        match BITMAP.compare_exchange(
            core::ptr::null_mut(),
            fresh,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                BITMAP_GRANULES.store(granules, Ordering::Release);
                words = fresh;
            }
            Err(winner) => {
                // SAFETY: `fresh` came from `Box::into_raw` just above and
                // was never published.
                drop(unsafe {
                    alloc::boxed::Box::from_raw(core::ptr::slice_from_raw_parts_mut(
                        fresh,
                        granules.div_ceil(64),
                    ))
                });
                words = winner;
            }
        }
    }
    let granules = BITMAP_GRANULES.load(Ordering::Acquire);
    // SAFETY: the bitmap is published once, never freed, and spans
    // `granules.div_ceil(64)` words (allocated for this same window).
    let words = unsafe { core::slice::from_raw_parts(words, granules.div_ceil(64)) };
    Ok((base, Granules { words, granules }))
}

/// Off metal there is no window to place in.
#[cfg(not(target_os = "none"))]
fn placer() -> Result<(u64, Granules<'static>), ReserveError> {
    Err(ReserveError::Unsupported)
}

/// **A bitmap of granules, claimed in runs** — lock-free: a run is claimed
/// word by word, each with one compare-and-swap, and a run that meets a bit
/// claimed meanwhile hands back what it took and looks further.  The task's
/// own thread is the only claimant today, so the hand-back never fires; it
/// is what keeps the bitmap sound if it ever is not.
#[derive(Clone, Copy)]
struct Granules<'a> {
    words: &'a [AtomicU64],
    granules: usize,
}

impl Granules<'_> {
    /// Claim `run` free granules in a row, first fit: the first's index.
    fn claim(&self, run: usize) -> Option<usize> {
        if run == 0 || run > self.granules {
            return None;
        }
        let mut start = 0;
        while start + run <= self.granules {
            match self.first_set(start, start + run) {
                // Something in the way: look past it.
                Some(taken) => start = taken + 1,
                None => match self.try_claim(start, run) {
                    Ok(()) => return Some(start),
                    Err(taken) => start = taken + 1,
                },
            }
        }
        None
    }

    /// Release the run `[first, first + run)`.
    fn free(&self, first: usize, run: usize) {
        for (word, mask) in masks(first, run) {
            if let Some(word) = self.words.get(word) {
                word.fetch_and(!mask, Ordering::AcqRel);
            }
        }
    }

    /// The first set granule in `[from, to)`, if any.
    fn first_set(&self, from: usize, to: usize) -> Option<usize> {
        masks(from, to - from).find_map(|(word, mask)| {
            let hit = self.words[word].load(Ordering::Acquire) & mask;
            (hit != 0).then(|| word * 64 + hit.trailing_zeros() as usize)
        })
    }

    /// Claim `[first, first + run)` word by word; on meeting a set granule,
    /// hand back every word already claimed and answer where it was.
    fn try_claim(&self, first: usize, run: usize) -> Result<(), usize> {
        for (claimed, (word, mask)) in masks(first, run).enumerate() {
            let taken = self.words[word].fetch_or(mask, Ordering::AcqRel) & mask;
            if taken != 0 {
                // Keep the bits another claimant holds; hand back ours, in
                // this word and in every word before it.
                self.words[word].fetch_and(!(mask & !taken), Ordering::AcqRel);
                for (word, mask) in masks(first, run).take(claimed) {
                    self.words[word].fetch_and(!mask, Ordering::AcqRel);
                }
                return Err(word * 64 + taken.trailing_zeros() as usize);
            }
        }
        Ok(())
    }
}

/// The `(word, mask)` pairs covering granules `[first, first + run)`.
fn masks(first: usize, run: usize) -> impl Iterator<Item = (usize, u64)> {
    let end = first + run;
    (first / 64..end.div_ceil(64)).map(move |word| {
        let lo = first.max(word * 64) - word * 64;
        let hi = end.min(word * 64 + 64) - word * 64;
        let mask = if hi - lo == 64 {
            u64::MAX
        } else {
            ((1u64 << (hi - lo)) - 1) << lo
        };
        (word, mask)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec::Vec;

    fn bitmap(granules: usize) -> Vec<AtomicU64> {
        (0..granules.div_ceil(64))
            .map(|_| AtomicU64::new(0))
            .collect()
    }

    /// Runs land first fit, span words, refuse when nothing fits, and come
    /// back — coalesced with their neighbours — when freed.
    #[test]
    fn runs_claim_first_fit_and_coalesce_when_freed() {
        let words = bitmap(16384);
        let g = Granules {
            words: &words,
            granules: 16384,
        };
        // A guarded wasm memory: 65 granules, word-straddling.
        let memories: Vec<usize> = (0..252).map(|_| g.claim(65).expect("fits")).collect();
        assert_eq!(memories[0], 0);
        assert_eq!(memories[1], 65);
        assert_eq!(memories[251], 251 * 65);
        assert_eq!(g.claim(65), None, "252 memories fill a 1 TiB window");
        assert_eq!(
            g.claim(4),
            Some(252 * 65),
            "the tail still holds a small run"
        );
        // Two freed neighbours coalesce into a run neither held alone.
        g.free(memories[10], 65);
        g.free(memories[11], 65);
        assert_eq!(g.claim(130), Some(memories[10]));
        assert_eq!(g.claim(1), None);
        assert_eq!(g.claim(0), None);
        assert_eq!(g.claim(16385), None);
    }

    /// A claim that meets granules taken under it hands back what it took.
    #[test]
    fn a_contended_claim_hands_back_its_partial_run() {
        let words = bitmap(256);
        let g = Granules {
            words: &words,
            granules: 256,
        };
        // Granule 100 is someone else's: a run over it refuses, leaves 100
        // set, and leaves nothing else.
        words[1].fetch_or(1 << (100 - 64), Ordering::Relaxed);
        assert_eq!(g.try_claim(60, 65), Err(100));
        let set: u32 = words
            .iter()
            .map(|w| w.load(Ordering::Relaxed).count_ones())
            .sum();
        assert_eq!(set, 1);
        // First fit: the 100 granules below it hold one run; the next
        // starts past it.
        assert_eq!(g.claim(65), Some(0));
        assert_eq!(g.claim(65), Some(101));
    }

    /// Concurrent claimants never share a granule.
    #[test]
    fn concurrent_claims_are_disjoint() {
        let words = bitmap(4096);
        let g = Granules {
            words: &words,
            granules: 4096,
        };
        let runs: Vec<Vec<usize>> = std::thread::scope(|s| {
            let workers: Vec<_> = (0..8)
                .map(|_| {
                    s.spawn(move || {
                        let mut mine = Vec::new();
                        while let Some(first) = g.claim(7) {
                            mine.push(first);
                        }
                        mine
                    })
                })
                .collect();
            workers.into_iter().map(|w| w.join().unwrap()).collect()
        });
        let mut all: Vec<usize> = runs.into_iter().flatten().collect();
        all.sort_unstable();
        for pair in all.windows(2) {
            assert!(pair[0] + 7 <= pair[1], "two runs overlap");
        }
        assert_eq!(all.len(), 4096 / 7);
    }

    /// Committed spans merge where they abut and refuse past capacity.
    #[test]
    fn committed_spans_merge_where_they_abut() {
        let none = [None; COMMITTED_RANGES];
        let one = merge(&none, 0x200_0000, 0x201_0000).unwrap();
        let grown = merge(&one, 0x201_0000, 0x240_0000).unwrap();
        assert_eq!(grown.iter().flatten().count(), 1);
        assert!(grown.contains(&Some((0x200_0000, 0x240_0000))));
        let below = merge(&grown, 0x100_0000, 0x200_0000).unwrap();
        assert!(below.contains(&Some((0x100_0000, 0x240_0000))));
        let mut spans = none;
        for i in 0..COMMITTED_RANGES as u64 {
            spans = merge(&spans, i * 0x2000, i * 0x2000 + 0x1000).unwrap();
        }
        assert_eq!(merge(&spans, 0x10_0000, 0x10_1000), None, "full");
        assert!(
            merge(&spans, 0x1000, 0x2000).is_some(),
            "a join needs no slot"
        );
    }

    /// The gaps of a span are what no committed span covers, in order.
    #[test]
    fn gaps_are_the_uncovered_parts_in_order() {
        let mut spans = [None; COMMITTED_RANGES];
        spans[3] = Some((0x4000, 0x6000));
        spans[0] = Some((0x1000, 0x2000));
        let found: Vec<(u64, u64)> = gaps(&spans, 0, 0x8000).into_iter().flatten().collect();
        assert_eq!(found, [(0, 0x1000), (0x2000, 0x4000), (0x6000, 0x8000)]);
        assert_eq!(gaps(&spans, 0x4000, 0x6000).iter().flatten().count(), 0);
        let inside: Vec<(u64, u64)> = gaps(&spans, 0x1800, 0x5000).into_iter().flatten().collect();
        assert_eq!(inside, [(0x2000, 0x4000)]);
        let none = [None; COMMITTED_RANGES];
        let whole: Vec<(u64, u64)> = gaps(&none, 0x3000, 0x9000).into_iter().flatten().collect();
        assert_eq!(whole, [(0x3000, 0x9000)]);
    }

    /// Off the kore target there is no window: reserving refuses.
    #[test]
    fn off_target_nothing_reserves() {
        assert_eq!(
            Reservation::new(GRANULE).err(),
            Some(ReserveError::Unsupported)
        );
    }
}
