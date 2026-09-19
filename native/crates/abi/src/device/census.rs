//! The read-only PCI **function census** — every function the boot-frozen
//! catalog enumerated, as bind-inert identity rows: segment, BDF locator,
//! vendor/device, class triple, revision, the catalog's own flags, and the
//! broker ordinal of the root that owns it.  Descriptions only: a row grants
//! no inventory, claim, mapping, MMIO, or DMA authority, and it carries no
//! address (the locator is `bus << 8 | device << 3 | function`, never a
//! thing to dereference — the `I2cAdmissionCandidate` rule).
//!
//! The distinct observation contract res/USERLAND.md asked for before a
//! bus-wide `lspci`: the KERNEL answers one row per
//! [`SystemMetric::DeviceRow`](crate::session::SystemMetric::DeviceRow)
//! observation on the `SystemObserve` capability only the monitor holds;
//! the monitor assembles this bounded record once (the catalog never
//! changes after boot) and vends it in [`DeviceCensusPage`]s; exec relays
//! the pages to a launcher; the launcher stamps the whole record into the
//! immutable process-context prefix of an executable whose bootstrap row
//! declares it.  No task on that path gains configuration-space access.
//!
//! Bounded by design: a machine with more functions than
//! [`MAX_DEVICE_CENSUS_ROWS`] is reported as truncated (`total > count`,
//! [`DEVICE_CENSUS_TRUNCATED`]) rather than silently shortened.

use zerocopy::byteorder::{LittleEndian, U16};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

/// Rows the assembled census holds.  16 bytes each plus the 16-byte
/// header is 1936 bytes — inside the 2048-byte process-context prefix
/// (`PROCESS_CONTEXT_BYTES`, asserted below) with the rest of that prefix
/// left for nothing else: the census is the whole context of the one
/// executable that declares it.  A laptop enumerates 20–40 functions and
/// a workstation under 100; a dense server can exceed this and is told so.
pub const MAX_DEVICE_CENSUS_ROWS: usize = 120;
/// Rows per vended page — sized so a page rides the monitor and exec reply
/// wires without a bulk window (8 header bytes + 128 = 136 bytes).
pub const DEVICE_CENSUS_PAGE_ROWS: usize = 8;
/// Pages a full census spans.
pub const DEVICE_CENSUS_PAGES: usize = MAX_DEVICE_CENSUS_ROWS.div_ceil(DEVICE_CENSUS_PAGE_ROWS);
/// Layout version.  A zeroed record reads `abi == 0` and is *unavailable*:
/// a launcher that could not obtain the census stamps nothing and the
/// consumer says so rather than rendering an empty machine.
pub const DEVICE_CENSUS_ABI: u16 = 1;
/// `DeviceCensus::flags` / `DeviceCensusPage::flags`: `total > count`.
pub const DEVICE_CENSUS_TRUNCATED: u8 = 1 << 0;
/// `DeviceCensusRow::root` for a function on a root past the broker fleet
/// (`MAX_ROOT_COMPLEXES`): enumerated, but no broker row can hand it out.
pub const DEVICE_CENSUS_ROOT_NONE: u8 = 0xFF;

/// One enumerated PCI function.  `flags` reuses the
/// [`DeviceRecord`](super::DeviceRecord) vocabulary — `HAS_BINDING`,
/// `DRIVER_ENDPOINT` / `NOT_DRIVER_ENDPOINT`, `IS_BRIDGE` — so a census
/// row and a broker inspection describe the same function in the same
/// words.  16 bytes, frozen: additive evolution claims `reserved`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct DeviceCensusRow {
    /// PCI segment group.
    pub segment: U16<LittleEndian>,
    /// `bus << 8 | device << 3 | function`.
    pub locator: U16<LittleEndian>,
    pub vendor: U16<LittleEndian>,
    pub device: U16<LittleEndian>,
    pub class: u8,
    pub subclass: u8,
    pub progif: u8,
    pub revision: u8,
    /// `DeviceRecord::{HAS_BINDING, NOT_DRIVER_ENDPOINT, DRIVER_ENDPOINT,
    /// IS_BRIDGE}` bits; the interrupt/BAR bits are not sampled here.
    pub flags: u8,
    /// Broker ordinal of the owning root (ascending `(segment, bus_start)`),
    /// or [`DEVICE_CENSUS_ROOT_NONE`].
    pub root: u8,
    /// 1 for a populated row; 0 marks the zero row.
    pub present: u8,
    pub reserved: u8,
}

impl DeviceCensusRow {
    pub const EMPTY: Self = Self {
        segment: U16::ZERO,
        locator: U16::ZERO,
        vendor: U16::ZERO,
        device: U16::ZERO,
        class: 0,
        subclass: 0,
        progif: 0,
        revision: 0,
        flags: 0,
        root: 0,
        present: 0,
        reserved: 0,
    };

    /// The flag bits a census row may carry (the sampled subset).
    pub const FLAGS: u8 = super::DeviceRecord::HAS_BINDING
        | super::DeviceRecord::NOT_DRIVER_ENDPOINT
        | super::DeviceRecord::DRIVER_ENDPOINT
        | super::DeviceRecord::IS_BRIDGE;

    pub const fn is_present(&self) -> bool {
        self.present == 1
    }

    pub const fn bus(&self) -> u8 {
        (self.locator.get() >> 8) as u8
    }

    pub const fn slot(&self) -> u8 {
        ((self.locator.get() >> 3) & 0x1F) as u8
    }

    pub const fn function(&self) -> u8 {
        (self.locator.get() & 0x7) as u8
    }

    pub const fn validate(&self) -> bool {
        match self.present {
            0 => {
                let z = Self::EMPTY;
                self.segment.get() == z.segment.get()
                    && self.locator.get() == 0
                    && self.vendor.get() == 0
                    && self.device.get() == 0
                    && self.class == 0
                    && self.subclass == 0
                    && self.progif == 0
                    && self.revision == 0
                    && self.flags == 0
                    && self.root == 0
                    && self.reserved == 0
            }
            1 => self.reserved == 0 && self.flags & !Self::FLAGS == 0,
            _ => false,
        }
    }

    /// The row as one `SystemObserve` report — `(value, aux)` — with the
    /// row's catalog `index` echoed in `aux[63:48]` so a reader can prove
    /// the report answers the row it asked for.
    pub const fn to_observation(&self, index: u16) -> (u64, u64) {
        let value = self.vendor.get() as u64
            | (self.device.get() as u64) << 16
            | (self.class as u64) << 32
            | (self.subclass as u64) << 40
            | (self.progif as u64) << 48
            | (self.revision as u64) << 56;
        let aux = self.segment.get() as u64
            | (self.locator.get() as u64) << 16
            | (self.root as u64) << 32
            | (self.flags as u64) << 40
            | (index as u64) << 48;
        (value, aux)
    }

    /// Inverse of [`to_observation`](Self::to_observation); `None` when the
    /// echoed index is not `index` or the flags leave the sampled subset.
    pub const fn from_observation(index: u16, value: u64, aux: u64) -> Option<Self> {
        if (aux >> 48) as u16 != index {
            return None;
        }
        let row = Self {
            segment: U16::new(aux as u16),
            locator: U16::new((aux >> 16) as u16),
            vendor: U16::new(value as u16),
            device: U16::new((value >> 16) as u16),
            class: (value >> 32) as u8,
            subclass: (value >> 40) as u8,
            progif: (value >> 48) as u8,
            revision: (value >> 56) as u8,
            flags: (aux >> 40) as u8,
            root: (aux >> 32) as u8,
            present: 1,
            reserved: 0,
        };
        if row.validate() { Some(row) } else { None }
    }
}

/// One vended page of the census.  136 bytes, frozen.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct DeviceCensusPage {
    /// Page ordinal, `0..DEVICE_CENSUS_PAGES`.
    pub page: u8,
    /// Populated rows in this page, `0..=DEVICE_CENSUS_PAGE_ROWS`; rows past
    /// it are the zero row.
    pub rows_valid: u8,
    /// Functions the catalog enumerated (may exceed the census bound).
    pub total: U16<LittleEndian>,
    /// Rows the whole census holds (`min(total, MAX_DEVICE_CENSUS_ROWS)`).
    pub count: U16<LittleEndian>,
    /// Root complexes the catalog described.
    pub roots: u8,
    /// [`DEVICE_CENSUS_TRUNCATED`].
    pub flags: u8,
    pub rows: [DeviceCensusRow; DEVICE_CENSUS_PAGE_ROWS],
}

impl DeviceCensusPage {
    pub const EMPTY: Self = Self {
        page: 0,
        rows_valid: 0,
        total: U16::ZERO,
        count: U16::ZERO,
        roots: 0,
        flags: 0,
        rows: [DeviceCensusRow::EMPTY; DEVICE_CENSUS_PAGE_ROWS],
    };

    pub fn validate(&self) -> bool {
        let count = self.count.get() as usize;
        let total = self.total.get() as usize;
        if (self.page as usize) >= DEVICE_CENSUS_PAGES
            || (self.rows_valid as usize) > DEVICE_CENSUS_PAGE_ROWS
            || count > MAX_DEVICE_CENSUS_ROWS
            || count > total
            || self.flags & !DEVICE_CENSUS_TRUNCATED != 0
            || (self.flags & DEVICE_CENSUS_TRUNCATED != 0) != (total > count)
        {
            return false;
        }
        // The page's rows are exactly the census rows at
        // `page * PAGE_ROWS ..`, so `rows_valid` is determined by `count`.
        let first = self.page as usize * DEVICE_CENSUS_PAGE_ROWS;
        let expect = count.saturating_sub(first).min(DEVICE_CENSUS_PAGE_ROWS);
        if self.rows_valid as usize != expect {
            return false;
        }
        self.rows
            .iter()
            .enumerate()
            .all(|(i, row)| row.validate() && row.is_present() == (i < expect))
    }
}

/// The assembled census.  1936 bytes; the whole process-context prefix of
/// the executable that declares it.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct DeviceCensus {
    /// [`DEVICE_CENSUS_ABI`]; 0 = unavailable.
    pub abi: U16<LittleEndian>,
    /// Populated rows, `0..=MAX_DEVICE_CENSUS_ROWS`.
    pub count: U16<LittleEndian>,
    /// Functions the catalog enumerated.
    pub total: U16<LittleEndian>,
    /// Root complexes the catalog described.
    pub roots: u8,
    /// [`DEVICE_CENSUS_TRUNCATED`].
    pub flags: u8,
    pub reserved: [u8; 8],
    pub rows: [DeviceCensusRow; MAX_DEVICE_CENSUS_ROWS],
}

impl DeviceCensus {
    pub const EMPTY: Self = Self {
        abi: U16::ZERO,
        count: U16::ZERO,
        total: U16::ZERO,
        roots: 0,
        flags: 0,
        reserved: [0; 8],
        rows: [DeviceCensusRow::EMPTY; MAX_DEVICE_CENSUS_ROWS],
    };

    /// A fresh, available census with no rows yet — the assembler's start.
    pub const fn new(total: u16, roots: u8) -> Self {
        let count = if (total as usize) > MAX_DEVICE_CENSUS_ROWS {
            MAX_DEVICE_CENSUS_ROWS as u16
        } else {
            total
        };
        Self {
            abi: U16::new(DEVICE_CENSUS_ABI),
            count: U16::new(count),
            total: U16::new(total),
            roots,
            flags: if total > count {
                DEVICE_CENSUS_TRUNCATED
            } else {
                0
            },
            reserved: [0; 8],
            rows: [DeviceCensusRow::EMPTY; MAX_DEVICE_CENSUS_ROWS],
        }
    }

    pub const fn available(&self) -> bool {
        self.abi.get() == DEVICE_CENSUS_ABI
    }

    pub fn validate(&self) -> bool {
        let count = self.count.get() as usize;
        let total = self.total.get() as usize;
        if self.abi.get() != DEVICE_CENSUS_ABI && *self != Self::EMPTY {
            return false;
        }
        if self.abi.get() == 0 {
            return true;
        }
        count <= MAX_DEVICE_CENSUS_ROWS
            && count <= total
            && self.flags & !DEVICE_CENSUS_TRUNCATED == 0
            && (self.flags & DEVICE_CENSUS_TRUNCATED != 0) == (total > count)
            && self.reserved == [0; 8]
            && self
                .rows
                .iter()
                .enumerate()
                .all(|(i, row)| row.validate() && row.is_present() == (i < count))
    }

    /// The populated rows, in catalog (= BDF) order.
    pub fn rows(&self) -> &[DeviceCensusRow] {
        let count = (self.count.get() as usize).min(MAX_DEVICE_CENSUS_ROWS);
        &self.rows[..count]
    }

    /// Pages this census spans (`ceil(count / PAGE_ROWS)`; 0 when empty).
    pub fn page_count(&self) -> usize {
        (self.count.get() as usize)
            .min(MAX_DEVICE_CENSUS_ROWS)
            .div_ceil(DEVICE_CENSUS_PAGE_ROWS)
    }

    /// Set row `index` (`< count`) from one kernel observation; `false`
    /// when the index is out of range or the report does not decode.
    pub fn set_row(&mut self, index: u16, value: u64, aux: u64) -> bool {
        let Some(row) = DeviceCensusRow::from_observation(index, value, aux) else {
            return false;
        };
        let Some(slot) = self
            .rows
            .get_mut(index as usize)
            .filter(|_| index < self.count.get())
        else {
            return false;
        };
        *slot = row;
        true
    }

    /// Page `page` of this census, or `None` past the last page.  An
    /// available census with no rows (a machine that enumerated no PCI
    /// function) still has page 0 — empty — so it reaches its consumer as
    /// "0 enumerated", never as "unavailable".
    pub fn page(&self, page: u8) -> Option<DeviceCensusPage> {
        let first = page as usize * DEVICE_CENSUS_PAGE_ROWS;
        let count = self.count.get() as usize;
        if !self.available() || (page as usize) >= self.page_count().max(1) {
            return None;
        }
        let valid = count.saturating_sub(first).min(DEVICE_CENSUS_PAGE_ROWS);
        let mut out = DeviceCensusPage {
            page,
            rows_valid: valid as u8,
            total: self.total,
            count: self.count,
            roots: self.roots,
            flags: self.flags,
            rows: [DeviceCensusRow::EMPTY; DEVICE_CENSUS_PAGE_ROWS],
        };
        out.rows[..valid].copy_from_slice(&self.rows[first..first + valid]);
        Some(out)
    }

    /// Fold one vended page in.  The first page fixes the header; a later
    /// page whose header disagrees, or a page that does not validate, is
    /// refused and leaves the census unchanged.
    pub fn insert_page(&mut self, page: &DeviceCensusPage) -> bool {
        if !page.validate() {
            return false;
        }
        if !self.available() {
            *self = Self::new(page.total.get(), page.roots);
        }
        if self.total != page.total
            || self.count != page.count
            || self.roots != page.roots
            || self.flags != page.flags
        {
            return false;
        }
        let first = page.page as usize * DEVICE_CENSUS_PAGE_ROWS;
        let valid = page.rows_valid as usize;
        if first + valid > self.count.get() as usize {
            return false;
        }
        self.rows[first..first + valid].copy_from_slice(&page.rows[..valid]);
        true
    }

    /// Whether every row `< count` has been inserted.
    pub fn complete(&self) -> bool {
        self.available() && self.rows().iter().all(DeviceCensusRow::is_present)
    }
}

const _: () = assert!(core::mem::size_of::<DeviceCensusRow>() == 16);
const _: () = assert!(core::mem::size_of::<DeviceCensusPage>() == 8 + 16 * DEVICE_CENSUS_PAGE_ROWS);
const _: () = assert!(core::mem::size_of::<DeviceCensus>() == 16 + 16 * MAX_DEVICE_CENSUS_ROWS);
const _: () =
    assert!(core::mem::size_of::<DeviceCensus>() <= crate::process::PROCESS_CONTEXT_BYTES);
const _: () = assert!(MAX_DEVICE_CENSUS_ROWS <= u16::MAX as usize);
const _: () = assert!(DEVICE_CENSUS_PAGES <= u8::MAX as usize);

#[cfg(test)]
mod tests {
    use super::*;

    fn row(i: u16) -> DeviceCensusRow {
        DeviceCensusRow {
            segment: U16::new(0),
            locator: U16::new(i << 3),
            vendor: U16::new(0x8086),
            device: U16::new(0x2900 + i),
            class: 0x06,
            subclass: 0x00,
            progif: 0,
            revision: 1,
            flags: super::super::DeviceRecord::IS_BRIDGE,
            root: 0,
            present: 1,
            reserved: 0,
        }
    }

    #[test]
    fn a_row_round_trips_through_one_observation_and_echoes_its_index() {
        let r = row(3);
        let (value, aux) = r.to_observation(3);
        assert_eq!(DeviceCensusRow::from_observation(3, value, aux), Some(r));
        assert_eq!(DeviceCensusRow::from_observation(4, value, aux), None);
        // A flag outside the sampled subset does not decode.
        let bad = aux | (u64::from(super::super::DeviceRecord::HAS_BAR0) << 40);
        assert_eq!(DeviceCensusRow::from_observation(3, value, bad), None);
        assert_eq!((r.bus(), r.slot(), r.function()), (0, 3, 0));
    }

    #[test]
    fn pages_reassemble_the_census_and_disagreeing_pages_are_refused() {
        let mut census = DeviceCensus::new(11, 2);
        for i in 0..11u16 {
            let (v, a) = row(i).to_observation(i);
            assert!(census.set_row(i, v, a));
        }
        assert!(census.validate());
        assert!(census.complete());
        assert_eq!(census.page_count(), 2);
        assert!(census.page(2).is_none());

        let mut rebuilt = DeviceCensus::EMPTY;
        assert!(rebuilt.validate());
        assert!(!rebuilt.available());
        for p in 0..2u8 {
            let page = census.page(p).unwrap();
            assert!(page.validate());
            assert!(rebuilt.insert_page(&page));
        }
        assert_eq!(rebuilt, census);

        let mut foreign = census.page(1).unwrap();
        foreign.roots = 3;
        assert!(!rebuilt.insert_page(&foreign));
        assert_eq!(rebuilt, census);
    }

    #[test]
    fn an_empty_census_still_has_its_one_empty_page() {
        let census = DeviceCensus::new(0, 1);
        assert!(census.validate() && census.complete());
        assert_eq!(census.page_count(), 0);
        let page = census.page(0).unwrap();
        assert!(page.validate());
        assert_eq!(page.rows_valid, 0);
        assert!(census.page(1).is_none());
        let mut rebuilt = DeviceCensus::EMPTY;
        assert!(rebuilt.insert_page(&page));
        assert!(rebuilt.available() && rebuilt.complete());
        assert_eq!(rebuilt, census);
    }

    #[test]
    fn truncation_is_reported_not_hidden() {
        let census = DeviceCensus::new(500, 9);
        assert_eq!(census.count.get() as usize, MAX_DEVICE_CENSUS_ROWS);
        assert_eq!(census.total.get(), 500);
        assert_eq!(census.flags, DEVICE_CENSUS_TRUNCATED);
        // Not yet complete (no rows), but structurally a valid record only
        // once its rows are present; the empty rows make it invalid now.
        assert!(!census.validate());
        assert!(!census.complete());
        let mut page = DeviceCensusPage::EMPTY;
        page.total = U16::new(500);
        page.count = U16::new(MAX_DEVICE_CENSUS_ROWS as u16);
        page.rows_valid = DEVICE_CENSUS_PAGE_ROWS as u8;
        // Flag missing while total > count: refused.
        assert!(!page.validate());
    }

    #[test]
    fn the_zero_row_and_a_stray_present_value_validate_as_documented() {
        assert!(DeviceCensusRow::EMPTY.validate());
        let mut r = DeviceCensusRow::EMPTY;
        r.vendor = U16::new(1);
        assert!(!r.validate());
        r = row(1);
        r.present = 2;
        assert!(!r.validate());
    }
}
