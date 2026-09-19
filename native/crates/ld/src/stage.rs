//! **How a parsed image's `PT_LOAD`s group into leases** — the staging
//! seam's contract half.
//!
//! Turning a parsed image into filled leases happens in three places
//! today, with three different answers to the same question:
//!
//! - `found_service` takes one lease per `PT_LOAD`;
//! - `found_first_shell` folds page-adjacent same-access LOADs into
//!   **runs** and takes one lease per run, because a conveyed lease costs
//!   a convey-table entry and that table is 16 wide;
//! - the U1 boottest hand-builds a single share for the linker stub.
//!
//! Neither policy is wrong — they answer to different budgets — but
//! nothing said which was canonical, and the difference had already
//! drifted into two open-coded loops. Adding a fourth consumer (the stub,
//! staged like a service image) is what forced the question, and
//! `kore-design-doctrine` §2 answers it: standardise the seam at N, not
//! at N+1.
//!
//! **What varies is grouping, and only grouping.** The rest — allocate a
//! backing extent, zero it, copy the span's file bytes in, derive a
//! `MemLease` — is identical, and stays kernel-side because it needs
//! kernel capability types. What lives here is the pure decision, which
//! is also the part that was untested: the run-folding is fiddly (a
//! suffix test over page-rounded spans) and had no coverage at all while
//! it was a loop inside a 700-line founding routine.
//!
//! The seam's three parts, in `kore-trait-architecture`'s terms: the
//! contract is [`Grouping`]; the engine is the kernel-side stager that is
//! generic over it; selection is type instantiation — [`PerLoad`] or
//! [`PerAccessRun`] named at the call site, never a runtime branch.

use heapless::Vec as HVec;

use crate::parse::LoadSegment;

/// Page granule the spans are rounded to. Mirrors `parse::PAGE_SIZE`;
/// a span is always a whole number of pages because a lease's backing is.
const PAGE: u64 = 4096;

/// Access bits of a span — **re-exported from the share verbs' own
/// vocabulary**, never restated.
///
/// These values are an ABI the kernel's share path reads; writing `1 / 2
/// / 4` here would be a second spelling of someone else's constant, and
/// the kind that stays right until it doesn't. Citing them means a change
/// on that side is a compile error here rather than a child mapped with
/// the wrong permissions.
pub mod access {
    pub use kore_abi::session::{SHARE_ACCESS_R as R, SHARE_ACCESS_W as W, SHARE_ACCESS_X as X};
}

/// One lease-sized group of an image: a page-rounded span of link
/// addresses plus the access mode every LOAD inside it shares.
///
/// `[base, end)` is page-aligned at both ends, so it is exactly what a
/// backing extent and its `MemLease` cover.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageSpan {
    /// Page-aligned first link address.
    pub base: u64,
    /// Page-aligned end (exclusive).
    pub end: u64,
    /// The [`access`] bits shared by every LOAD in the span.
    pub access: u8,
}

impl ImageSpan {
    /// Byte length — a page multiple by construction.
    pub fn len(&self) -> u64 {
        self.end - self.base
    }

    /// Never empty (a zero-length LOAD is dropped at parse), but the
    /// convention pair keeps clippy honest.
    pub fn is_empty(&self) -> bool {
        self.end == self.base
    }

    /// Whether this span must be minted exec-blessed. A blessed lease
    /// refuses every WRITE share (the per-lease W^X gate in
    /// `runtime::mm::lease`), which is why the grouping must never put
    /// an X LOAD in the same span as a W one.
    pub fn exec_blessed(&self) -> bool {
        self.access & access::X != 0
    }
}

/// Why a grouping refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupError {
    /// A segment's address or size does not fit the address space.
    Overflow,
    /// The image needs more spans than the caller's budget affords.
    TooManySpans,
    /// The image has no loadable segment.
    Empty,
}

/// **The contract**: how LOADs become lease-sized spans.
///
/// One implementation per budget, selected by type instantiation at the
/// call site. A new policy is a new impl — never a branch in the stager.
pub trait Grouping {
    /// Fold `segments` into spans, in ascending address order.
    ///
    /// `N` is the caller's span budget: the founder's staging stride for
    /// a service, the convey table's width for a wire founding.
    fn group<const N: usize>(
        segments: &[LoadSegment<'_>],
    ) -> Result<HVec<ImageSpan, N>, GroupError>;
}

/// Access bits of one parsed LOAD.
fn access_of(seg: &LoadSegment<'_>) -> u8 {
    let mut a = 0;
    if seg.flags.read {
        a |= access::R;
    }
    if seg.flags.write {
        a |= access::W;
    }
    if seg.flags.exec {
        a |= access::X;
    }
    a
}

/// The page-rounded span one LOAD occupies.
fn span_of(seg: &LoadSegment<'_>) -> Result<(u64, u64), GroupError> {
    let base = seg.vaddr & !(PAGE - 1);
    let end = seg
        .vaddr
        .checked_add(seg.memsz)
        .ok_or(GroupError::Overflow)?
        .next_multiple_of(PAGE);
    Ok((base, end))
}

/// **One lease per `PT_LOAD`** — `found_service`'s policy.
///
/// The founder stages each segment into its own slot of a fixed VA
/// stride, so segments cost founder slots rather than a shared table;
/// there is nothing to gain by folding and a per-LOAD lease keeps each
/// segment's access mode exactly as the linker emitted it.
pub struct PerLoad;

impl Grouping for PerLoad {
    fn group<const N: usize>(
        segments: &[LoadSegment<'_>],
    ) -> Result<HVec<ImageSpan, N>, GroupError> {
        let mut out: HVec<ImageSpan, N> = HVec::new();
        for seg in segments {
            let (base, end) = span_of(seg)?;
            out.push(ImageSpan {
                base,
                end,
                access: access_of(seg),
            })
            .map_err(|_| GroupError::TooManySpans)?;
        }
        if out.is_empty() {
            return Err(GroupError::Empty);
        }
        Ok(out)
    }
}

/// **One lease per maximal run of page-adjacent same-access LOADs** —
/// the wire founding's policy.
///
/// A conveyed lease costs a convey-table entry and that table is 16
/// wide, so folding is what keeps a founding inside its budget: the
/// shell's four LOADs collapse to three runs (its two R|W segments are
/// contiguous).
///
/// The fold is a **suffix test**, which is sound only because `kore-ld`
/// already proved the LOADs page-disjoint and ascending: a segment can
/// only ever extend the run it directly follows. It deliberately never
/// merges across an access change — an X run and a W run in one lease
/// would be unshareable, since the per-lease W^X gate refuses a WRITE
/// share off a blessed lease and an EXECUTE share off an unblessed one.
pub struct PerAccessRun;

impl Grouping for PerAccessRun {
    fn group<const N: usize>(
        segments: &[LoadSegment<'_>],
    ) -> Result<HVec<ImageSpan, N>, GroupError> {
        let mut out: HVec<ImageSpan, N> = HVec::new();
        for seg in segments {
            let (base, end) = span_of(seg)?;
            let acc = access_of(seg);
            match out.last_mut() {
                Some(run) if run.end == base && run.access == acc => run.end = end,
                _ => out
                    .push(ImageSpan {
                        base,
                        end,
                        access: acc,
                    })
                    .map_err(|_| GroupError::TooManySpans)?,
            }
        }
        if out.is_empty() {
            return Err(GroupError::Empty);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::SegFlags;

    fn seg(vaddr: u64, memsz: u64, flags: SegFlags) -> LoadSegment<'static> {
        LoadSegment {
            vaddr,
            memsz,
            filesz: memsz,
            file_off: vaddr,
            align: PAGE,
            flags,
            bytes: &[],
        }
    }

    const R: SegFlags = SegFlags {
        read: true,
        write: false,
        exec: false,
    };
    const RX: SegFlags = SegFlags {
        read: true,
        write: false,
        exec: true,
    };
    const RW: SegFlags = SegFlags {
        read: true,
        write: true,
        exec: false,
    };

    /// The committed shell's real shape: four LOADs, the last two R|W and
    /// page-adjacent. This is the case the convey budget depends on —
    /// four spans would put the founding at 17 of 16 conveys.
    fn shell_like() -> [LoadSegment<'static>; 4] {
        [
            seg(0x4000_0000, 0x123c, R),
            seg(0x4000_2240, 0x9945, RX),
            seg(0x4000_cb88, 0x478, RW),
            seg(0x4000_de68, 0xb38, RW),
        ]
    }

    /// `PerAccessRun` folds the shell to three spans, and the two R|W
    /// LOADs become one lease.
    #[test]
    fn access_runs_fold_the_shell_to_three() {
        let runs = PerAccessRun::group::<8>(&shell_like()).expect("groups");
        assert_eq!(runs.len(), 3, "R, R|X, R|W");
        assert_eq!(
            runs[0],
            ImageSpan {
                base: 0x4000_0000,
                end: 0x4000_2000,
                access: access::R
            }
        );
        assert_eq!(
            runs[1],
            ImageSpan {
                base: 0x4000_2000,
                end: 0x4000_c000,
                access: access::R | access::X
            }
        );
        assert_eq!(
            runs[2],
            ImageSpan {
                base: 0x4000_c000,
                end: 0x4000_f000,
                access: access::R | access::W
            },
            "the two adjacent R|W LOADs share one lease"
        );
    }

    /// The same image under the per-LOAD policy keeps four spans — the
    /// policies genuinely differ, which is why this is a seam and not a
    /// single algorithm someone forgot to share.
    #[test]
    fn per_load_keeps_every_segment_separate() {
        let spans = PerLoad::group::<8>(&shell_like()).expect("groups");
        assert_eq!(spans.len(), 4);
        assert_eq!(spans[2].access, access::R | access::W);
        assert_eq!(spans[3].access, access::R | access::W);
    }

    /// Only *adjacent* same-access LOADs fold. A gap between two R|W
    /// segments must stay two leases — merging across it would hand the
    /// child a mapping over addresses the image never claimed.
    #[test]
    fn a_gap_breaks_a_run() {
        let segs = [seg(0x4000_0000, 0x100, RW), seg(0x4000_4000, 0x100, RW)];
        let runs = PerAccessRun::group::<8>(&segs).expect("groups");
        assert_eq!(runs.len(), 2, "non-adjacent spans do not merge");
    }

    /// An access change breaks a run even when the spans touch. Folding
    /// X and W into one lease would be unshareable at both ends: the
    /// per-lease W^X gate refuses a WRITE share off a blessed lease and
    /// an EXECUTE share off an unblessed one.
    #[test]
    fn an_access_change_breaks_a_run_even_when_adjacent() {
        let segs = [seg(0x4000_0000, 0x1000, RX), seg(0x4000_1000, 0x1000, RW)];
        let runs = PerAccessRun::group::<8>(&segs).expect("groups");
        assert_eq!(runs.len(), 2);
        assert!(runs[0].exec_blessed());
        assert!(!runs[1].exec_blessed());
    }

    /// The budget is the caller's, and overflowing it refuses rather than
    /// truncating — a dropped span is a segment the child never gets.
    #[test]
    fn an_overflowing_budget_refuses() {
        assert_eq!(
            PerLoad::group::<2>(&shell_like()),
            Err(GroupError::TooManySpans)
        );
        // The same image fits under the folding policy, which is the
        // whole reason that policy exists.
        assert!(PerAccessRun::group::<3>(&shell_like()).is_ok());
    }

    /// An image with no loadable segment is refused by both policies —
    /// there is nothing to found from.
    #[test]
    fn no_loadable_segment_refuses() {
        assert_eq!(PerLoad::group::<8>(&[]), Err(GroupError::Empty));
        assert_eq!(PerAccessRun::group::<8>(&[]), Err(GroupError::Empty));
    }

    /// A segment whose end wraps the address space refuses before any
    /// span is produced.
    #[test]
    fn an_overflowing_segment_refuses() {
        let segs = [seg(u64::MAX - 0xfff, 0x2000, R)];
        assert_eq!(PerAccessRun::group::<8>(&segs), Err(GroupError::Overflow));
    }
}
