//! PRP (Physical Region Page) scatter-gather — **spec facts** (NVMe base
//! spec §4.3), the device-address math a data transfer needs.
//!
//! Three generalities, all deliberate:
//! - **Typed addresses**: every device address is a
//!   [`kore_abi::device::IovaAddr`] — "no phys in userland" is a *type*; raw
//!   `u64` appears only inside the wire SQE.
//! - **Page-size parameterized**: the memory page is the controller's
//!   *configured* `CC.MPS` (`2^(12+mps)`, [`MemPage`]), not a hardcoded
//!   4096 — a controller with `MPSMIN > 0` is handled correctly.
//! - **Contiguity is a contract, not an assumption**: [`build`] serves the
//!   seam's contiguous-window buffers ([`Dma`](crate::seam::Dma)'s stated
//!   contract) and computes trailing pages arithmetically;
//!   [`build_scattered`] takes the page list explicitly for any backing
//!   that cannot promise one window.  The two entry points make the
//!   assumption visible at every call site.
//!
//! The list fill is the core's one hot linear pass — vectorized with
//! `core::simd` (8 lanes of page-address iota per step).

use core::simd::Simd;

use kore_abi::device::IovaAddr;

/// A validated NVMe memory page size — `2^(12 + mps)` bytes, the
/// controller's configured `CC.MPS`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemPage {
    bytes: u64,
}

impl MemPage {
    /// The spec minimum (4 KiB, `mps = 0`) — every controller's floor.
    pub const MIN: Self = Self { bytes: 4096 };

    /// From the `CC.MPS` field value (`0..=15`).
    pub const fn from_mps(mps: u8) -> Self {
        Self {
            bytes: 1 << (12 + mps as u32),
        }
    }

    /// Page size in bytes.
    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    /// PRP entries one list page of this size holds (8 bytes each).
    pub const fn prps_per_page(self) -> u64 {
        self.bytes / 8
    }
}

/// The two PRP fields a command carries (still typed; the conversion to
/// the wire's raw `u64` happens inside the SQE builders).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Prp {
    /// PRP1 — the first (possibly page-offset) address.
    pub prp1: IovaAddr,
    /// PRP2 — the second page, a PRP-list pointer, or null.
    pub prp2: IovaAddr,
}

impl Prp {
    const NULL: Self = Self {
        prp1: IovaAddr(0),
        prp2: IovaAddr(0),
    };
}

/// Why a PRP build refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrpError {
    /// The transfer needs a PRP list but no list page was supplied.
    ListNeeded,
    /// The transfer needs more entries than one list page holds
    /// (multi-list chaining is a later tier).
    TooLarge,
    /// A scattered build's page list ran short of the transfer size.
    NotEnoughPages,
}

/// A caller-supplied PRP-**list page** the builder fills when a transfer
/// spans more than two pages — a `DmaPool` buffer the driver keeps for
/// reuse (device-visible: `iova` is what the command's PRP2 points at,
/// `entries` is the CPU-writable view of the same page).
pub struct ListPage<'a> {
    /// The list page's device address (becomes PRP2).
    pub iova: IovaAddr,
    /// CPU-writable entries of the same page.
    pub entries: &'a mut [u64],
}

/// Build the PRP fields for a `size`-byte transfer starting at
/// `buffer_iova`, within **one contiguous device-visible window** (the
/// [`Dma`](crate::seam::Dma) contract — trailing pages are computed
/// arithmetically from the base).
///
/// - `size ≤ first-page-remainder`: single page, PRP2 = 0.
/// - two pages: PRP2 = the second page's address (no list).
/// - more: PRP2 points at `list`, filled with the trailing page addresses
///   (requires `list`; the tail must fit one list page).
pub fn build(
    buffer_iova: IovaAddr,
    size: u64,
    page: MemPage,
    list: Option<ListPage<'_>>,
) -> Result<Prp, PrpError> {
    if size == 0 {
        return Ok(Prp::NULL);
    }
    let pg = page.bytes();
    let base = buffer_iova.0;
    let offset = base & (pg - 1);
    let first_page_bytes = pg - offset;
    let prp1 = buffer_iova;

    if size <= first_page_bytes {
        return Ok(Prp {
            prp1,
            prp2: IovaAddr(0),
        });
    }

    let remaining = size - first_page_bytes;
    let additional = remaining.div_ceil(pg);
    let second_page = (base & !(pg - 1)) + pg;

    if additional == 1 {
        return Ok(Prp {
            prp1,
            prp2: IovaAddr(second_page),
        });
    }

    // A PRP list: the trailing pages, contiguous from the second page —
    // a page-address iota, filled 8 lanes per step (the core's one hot
    // linear pass, vectorized).
    let list = list.ok_or(PrpError::ListNeeded)?;
    let n = additional as usize;
    if n > list.entries.len() || additional > page.prps_per_page() {
        return Err(PrpError::TooLarge);
    }
    fill_page_iota(&mut list.entries[..n], second_page, pg);
    Ok(Prp {
        prp1,
        prp2: list.iova,
    })
}

/// Build the PRP fields for a transfer over an **explicit page list** —
/// the scatter-aware twin of [`build`], for any backing that is not one
/// contiguous device window.  `pages` yields each page's device address
/// in transfer order; the first may be page-offset (its offset rides
/// PRP1, per spec), the rest must be page-aligned.
pub fn build_scattered(
    mut pages: impl Iterator<Item = IovaAddr>,
    size: u64,
    page: MemPage,
    list: Option<ListPage<'_>>,
) -> Result<Prp, PrpError> {
    if size == 0 {
        return Ok(Prp::NULL);
    }
    let pg = page.bytes();
    let first = pages.next().ok_or(PrpError::NotEnoughPages)?;
    let first_page_bytes = pg - (first.0 & (pg - 1));
    if size <= first_page_bytes {
        return Ok(Prp {
            prp1: first,
            prp2: IovaAddr(0),
        });
    }

    let additional = (size - first_page_bytes).div_ceil(pg);
    if additional == 1 {
        let second = pages.next().ok_or(PrpError::NotEnoughPages)?;
        return Ok(Prp {
            prp1: first,
            prp2: second,
        });
    }

    let list = list.ok_or(PrpError::ListNeeded)?;
    let n = additional as usize;
    if n > list.entries.len() || additional > page.prps_per_page() {
        return Err(PrpError::TooLarge);
    }
    for slot in list.entries.iter_mut().take(n) {
        *slot = pages.next().ok_or(PrpError::NotEnoughPages)?.0;
    }
    Ok(Prp {
        prp1: first,
        prp2: list.iova,
    })
}

/// Fill `out` with the page-address iota `start, start+pg, …` — 8 lanes
/// per step, scalar tail.
fn fill_page_iota(out: &mut [u64], start: u64, pg: u64) {
    const LANES: usize = 8;
    // [0,1,…,7]·pg once; each step splats the running base over it.
    let steps = Simd::<u64, LANES>::from_array([0, 1, 2, 3, 4, 5, 6, 7]) * Simd::splat(pg);
    let mut base = start;
    let mut chunks = out.chunks_exact_mut(LANES);
    for chunk in &mut chunks {
        (Simd::splat(base) + steps).copy_to_slice(chunk);
        base += pg * LANES as u64;
    }
    for slot in chunks.into_remainder() {
        *slot = base;
        base += pg;
    }
}

/// The largest single transfer one PRP list page describes at `page`
/// size (the first page plus a full list of trailing pages) — the MDTS
/// the driver caps I/O at before multi-list chaining.
pub const fn max_single_list_transfer(page: MemPage) -> u64 {
    page.bytes() + page.prps_per_page() * page.bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    const PG: MemPage = MemPage::MIN;

    #[test]
    fn mem_page_decodes_mps() {
        assert_eq!(MemPage::from_mps(0).bytes(), 4096);
        assert_eq!(MemPage::from_mps(1).bytes(), 8192);
        assert_eq!(MemPage::from_mps(0).prps_per_page(), 512);
    }

    #[test]
    fn single_page_needs_no_prp2() {
        let p = build(IovaAddr(0x1000), 4096, PG, None).unwrap();
        assert_eq!(p.prp1, IovaAddr(0x1000));
        assert_eq!(p.prp2, IovaAddr(0));
    }

    #[test]
    fn two_pages_use_prp2_directly_no_list() {
        let p = build(IovaAddr(0x1000), 8192, PG, None).unwrap();
        assert_eq!(
            p,
            Prp {
                prp1: IovaAddr(0x1000),
                prp2: IovaAddr(0x2000)
            }
        );

        // Offset into the first page still spills into the second.
        let p = build(IovaAddr(0x1F00), 0x200, PG, None).unwrap();
        assert_eq!(p.prp1, IovaAddr(0x1F00));
        assert_eq!(p.prp2, IovaAddr(0x2000), "spilled 0x100 bytes");
    }

    #[test]
    fn many_pages_fill_the_list_simd_and_point_prp2_at_it() {
        let mut entries = [0u64; 512];
        let list = ListPage {
            iova: IovaAddr(0x9000),
            entries: &mut entries,
        };
        // 21 pages: prp1 + a 20-entry list — crosses the 8-lane SIMD
        // boundary twice with a scalar tail of 4.
        let p = build(IovaAddr(0x1000), 21 * 4096, PG, Some(list)).unwrap();
        assert_eq!(p.prp2, IovaAddr(0x9000));
        for (i, &e) in entries[..20].iter().enumerate() {
            assert_eq!(e, 0x2000 + (i as u64) * 4096, "iota entry {i}");
        }
        assert_eq!(entries[20], 0, "unused entries untouched");
    }

    #[test]
    fn larger_mps_changes_the_geometry() {
        // 8 KiB pages: a 16 KiB transfer is exactly two pages ⇒ PRP2
        // direct, no list — where 4 KiB pages would need a list.
        let pg8 = MemPage::from_mps(1);
        let p = build(IovaAddr(0x4000), 16384, pg8, None).unwrap();
        assert_eq!(p.prp2, IovaAddr(0x6000), "second 8 KiB page");
        assert!(matches!(
            build(IovaAddr(0x4000), 16384, PG, None),
            Err(PrpError::ListNeeded),
        ));
    }

    #[test]
    fn scattered_build_takes_explicit_pages() {
        // Three non-contiguous pages — inexpressible by `build`.
        let mut entries = [0u64; 512];
        let list = ListPage {
            iova: IovaAddr(0xA000),
            entries: &mut entries,
        };
        let pages = [IovaAddr(0x1000), IovaAddr(0x7000), IovaAddr(0x3000)];
        let p = build_scattered(pages.into_iter(), 3 * 4096, PG, Some(list)).unwrap();
        assert_eq!(p.prp1, IovaAddr(0x1000));
        assert_eq!(p.prp2, IovaAddr(0xA000));
        assert_eq!(&entries[..2], &[0x7000, 0x3000], "the scattered tail");
    }

    #[test]
    fn scattered_refuses_a_short_page_list() {
        assert!(matches!(
            build_scattered([IovaAddr(0x1000)].into_iter(), 2 * 4096, PG, None),
            Err(PrpError::NotEnoughPages),
        ));
    }

    #[test]
    fn a_list_is_required_when_the_transfer_spills_past_two_pages() {
        assert_eq!(
            build(IovaAddr(0x1000), 4 * 4096, PG, None),
            Err(PrpError::ListNeeded)
        );
    }

    #[test]
    fn zero_size_is_a_null_prp() {
        assert_eq!(build(IovaAddr(0xdead), 0, PG, None), Ok(Prp::NULL));
    }
}
