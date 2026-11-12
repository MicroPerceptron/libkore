//! The read-only **live state** of the PCI functions the census
//! ([`super::census`]) lists, and the shape of one function's BARs.
//!
//! The census is boot-frozen identity. This is what changes after boot, and
//! what a driver writer asks before touching a device: whether a driver
//! claimed the function, which founded row drives its DMA, which BARs exist,
//! which of them a driver may map and which are mapped now, and how the
//! function can interrupt. A BAR is its index, its size and the length a
//! driver may map; **no address crosses** (res/DEVICE_SERVICE.md §1.1), so
//! nothing here says where a BAR lives.
//!
//! Descriptions only, as a census row is: a state grants no inventory,
//! claim, mapping or route, and reading one changes nothing. It is a
//! snapshot. A claim can land or a route can be released the moment after
//! the kernel answers, so no consumer may treat it as permission.
//!
//! The path is the census's: the kernel answers one
//! [`SystemMetric::DeviceState`](crate::session::SystemMetric::DeviceState)
//! or [`SystemMetric::DeviceBar`](crate::session::SystemMetric::DeviceBar)
//! observation on the `SystemObserve` capability the monitor holds; the
//! monitor samples them on each request (never cached, since they are live)
//! and vends [`DeviceStatePage`]s and one [`DeviceBarTable`]; exec relays
//! them to a launcher; the launcher stamps a [`DeviceInventory`] into the
//! process-context prefix of an executable whose bootstrap row declares it.

use zerocopy::byteorder::{LittleEndian, U16, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use super::census::{DeviceCensus, DeviceCensusPage, MAX_DEVICE_CENSUS_ROWS};

/// BARs a PCI function has (PCI Local Bus Specification 3.0 §6.2.5: six
/// Base Address Registers in a type 0 header).
pub const DEVICE_BARS: usize = 6;
/// Every BAR's bit, for the three per-BAR bitmaps of a [`DeviceState`].
pub const DEVICE_BAR_MASK: u8 = (1 << DEVICE_BARS) - 1;
/// `DeviceState::driver` when no founded row drives the function's DMA.
pub const DEVICE_ROW_NONE: u8 = 0xFF;
/// States per vended page: 8 header bytes + 128 = 136 bytes, the census
/// page's size, so a page rides the monitor and exec reply wires as one.
pub const DEVICE_STATE_PAGE_ROWS: usize = 16;
/// Pages the states of a full census span.
pub const DEVICE_STATE_PAGES: usize = MAX_DEVICE_CENSUS_ROWS.div_ceil(DEVICE_STATE_PAGE_ROWS);
/// Layout version of [`DeviceInventory`]. A zeroed record reads 0 and is
/// *unavailable*.
pub const DEVICE_INVENTORY_ABI: u16 = 1;
/// [`SystemMetric::DeviceBar`](crate::session::SystemMetric::DeviceBar)'s
/// operand is `index | bar << DEVICE_BAR_OPERAND_SHIFT`: the catalog index
/// in the low 13 bits (every index a census can hold fits, asserted below),
/// the BAR number in the top three.
pub const DEVICE_BAR_OPERAND_SHIFT: u32 = 13;

/// The operand naming BAR `bar` of the function at catalog `index`; `None`
/// when either is out of the operand's range.
pub const fn device_bar_operand(index: u16, bar: u8) -> Option<u16> {
    if index >> DEVICE_BAR_OPERAND_SHIFT != 0 || bar as usize >= DEVICE_BARS {
        return None;
    }
    Some(index | (bar as u16) << DEVICE_BAR_OPERAND_SHIFT)
}

/// The `(index, bar)` a [`device_bar_operand`] names; `None` for a BAR
/// number no function has.
pub const fn device_bar_of(operand: u16) -> Option<(u16, u8)> {
    let bar = (operand >> DEVICE_BAR_OPERAND_SHIFT) as u8;
    if bar as usize >= DEVICE_BARS {
        return None;
    }
    Some((operand & ((1 << DEVICE_BAR_OPERAND_SHIFT) - 1), bar))
}

/// One function's live state. 8 bytes, frozen: one observation's value
/// word, byte for byte.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct DeviceState {
    /// [`Self::PRESENT`] and the bits below it.
    pub flags: u8,
    /// The founded row (the founding record's index) whose task activated
    /// DMA on this function, or [`DEVICE_ROW_NONE`]. The kernel's own
    /// shutdown ledger, so it names who drives the device rather than who
    /// claimed it: a function a row claimed and delegated reads the row it
    /// was delegated to, and a driver that never activates DMA reads none.
    pub driver: u8,
    /// Bit `n`: BAR `n` exists as memory the kernel resolved. An I/O BAR is
    /// not one: no driver may reach port space through a BAR.
    pub bars_present: u8,
    /// Bit `n`: a driver may map a prefix of BAR `n`.
    pub bars_mappable: u8,
    /// Bit `n`: BAR `n` is mapped by a driver now.
    pub bars_mapped: u8,
    /// Vectors the function's MSI capability can deliver (1, 2, 4, 8, 16 or
    /// 32); 0 when it has none.
    pub msi_vectors: u8,
    /// Entries in the function's MSI-X table; 0 when it has none.
    pub msix_entries: U16<LittleEndian>,
}

impl DeviceState {
    /// A populated state; clear marks the zero state.
    pub const PRESENT: u8 = 1 << 0;
    /// A driver holds the function: its claim is committed.
    pub const CLAIMED: u8 = 1 << 1;
    /// An interrupt route is established on the function now.
    pub const ROUTED: u8 = 1 << 2;
    /// A driver can establish a remapped MSI-family route (MSI or MSI-X),
    /// the only kind userland may hold.
    pub const ROUTE_MSI: u8 = 1 << 3;
    /// A driver can establish a remapped MSI-X route.
    pub const ROUTE_MSIX: u8 = 1 << 4;
    /// The function has a legacy interrupt pin; [`Self::pin`] says which.
    pub const HAS_PIN: u8 = 1 << 5;
    /// Bits 6 and 7: the legacy pin, 0 for INTA through 3 for INTD.
    pub const PIN_SHIFT: u32 = 6;

    pub const EMPTY: Self = Self {
        flags: 0,
        driver: 0,
        bars_present: 0,
        bars_mappable: 0,
        bars_mapped: 0,
        msi_vectors: 0,
        msix_entries: U16::ZERO,
    };

    pub const fn is_present(&self) -> bool {
        self.flags & Self::PRESENT != 0
    }

    pub const fn claimed(&self) -> bool {
        self.flags & Self::CLAIMED != 0
    }

    /// The legacy interrupt pin, 0 for INTA through 3 for INTD.
    pub const fn pin(&self) -> Option<u8> {
        if self.flags & Self::HAS_PIN == 0 {
            return None;
        }
        Some(self.flags >> Self::PIN_SHIFT)
    }

    /// The founded row driving the function's DMA.
    pub const fn driver(&self) -> Option<u8> {
        if self.driver == DEVICE_ROW_NONE {
            None
        } else {
            Some(self.driver)
        }
    }

    pub const fn validate(&self) -> bool {
        if !self.is_present() {
            return self.flags == 0
                && self.driver == 0
                && self.bars_present == 0
                && self.bars_mappable == 0
                && self.bars_mapped == 0
                && self.msi_vectors == 0
                && self.msix_entries.get() == 0;
        }
        // A pin without its flag would be a second spelling of "no pin".
        (self.flags & Self::HAS_PIN != 0 || self.flags >> Self::PIN_SHIFT == 0)
            && self.bars_present & !DEVICE_BAR_MASK == 0
            // Only a BAR that exists can be mapped, and only one a driver
            // may map is.
            && self.bars_mappable & !self.bars_present == 0
            && self.bars_mapped & !self.bars_mappable == 0
            && (self.msi_vectors == 0
                || (self.msi_vectors.is_power_of_two() && self.msi_vectors <= 32))
            && self.msix_entries.get() <= super::DeviceRecord::MAX_MSIX_TABLE_ENTRIES
    }

    /// The state as one `SystemObserve` report, `(value, aux)`, with the
    /// function's catalog `index` echoed in `aux[63:48]` so a reader can
    /// prove the report answers the function it asked for.
    pub const fn to_observation(&self, index: u16) -> (u64, u64) {
        let value = u64::from_le_bytes([
            self.flags,
            self.driver,
            self.bars_present,
            self.bars_mappable,
            self.bars_mapped,
            self.msi_vectors,
            self.msix_entries.get().to_le_bytes()[0],
            self.msix_entries.get().to_le_bytes()[1],
        ]);
        (value, (index as u64) << 48)
    }

    /// Inverse of [`to_observation`](Self::to_observation); `None` when the
    /// echoed index is not `index`, the rest of `aux` is not zero, or the
    /// state is not a valid populated one.
    pub const fn from_observation(index: u16, value: u64, aux: u64) -> Option<Self> {
        if aux != (index as u64) << 48 {
            return None;
        }
        let bytes = value.to_le_bytes();
        let state = Self {
            flags: bytes[0],
            driver: bytes[1],
            bars_present: bytes[2],
            bars_mappable: bytes[3],
            bars_mapped: bytes[4],
            msi_vectors: bytes[5],
            msix_entries: U16::new(u16::from_le_bytes([bytes[6], bytes[7]])),
        };
        if state.is_present() && state.validate() {
            Some(state)
        } else {
            None
        }
    }
}

/// One vended page of states, row for row beside the census's rows.
/// 136 bytes, frozen.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct DeviceStatePage {
    /// Page ordinal, `0..DEVICE_STATE_PAGES`.
    pub page: u8,
    /// Populated states in this page, `0..=DEVICE_STATE_PAGE_ROWS`.
    pub rows_valid: u8,
    /// States the whole table holds: the census's `count`.
    pub count: U16<LittleEndian>,
    pub reserved: [u8; 4],
    pub states: [DeviceState; DEVICE_STATE_PAGE_ROWS],
}

impl DeviceStatePage {
    pub const EMPTY: Self = Self {
        page: 0,
        rows_valid: 0,
        count: U16::ZERO,
        reserved: [0; 4],
        states: [DeviceState::EMPTY; DEVICE_STATE_PAGE_ROWS],
    };

    /// Page `page` of a table of `count` states, empty: the assembler's
    /// start. `None` past the table's last page, though a table of no states
    /// still has page 0.
    pub const fn new(page: u8, count: u16) -> Option<Self> {
        let count = if count as usize > MAX_DEVICE_CENSUS_ROWS {
            MAX_DEVICE_CENSUS_ROWS
        } else {
            count as usize
        };
        let pages = count.div_ceil(DEVICE_STATE_PAGE_ROWS);
        let last = if pages == 0 { 0 } else { pages - 1 };
        if page as usize > last {
            return None;
        }
        Some(Self {
            page,
            rows_valid: 0,
            count: U16::new(count as u16),
            reserved: [0; 4],
            states: [DeviceState::EMPTY; DEVICE_STATE_PAGE_ROWS],
        })
    }

    /// The catalog index of this page's first state.
    pub const fn first(&self) -> usize {
        self.page as usize * DEVICE_STATE_PAGE_ROWS
    }

    /// States this page holds once complete.
    pub const fn expected(&self) -> usize {
        let left = (self.count.get() as usize).saturating_sub(self.first());
        if left > DEVICE_STATE_PAGE_ROWS {
            DEVICE_STATE_PAGE_ROWS
        } else {
            left
        }
    }

    /// Append the next state, from one kernel observation of the function at
    /// the catalog index this page reaches next; `false` when the page is
    /// full or the report does not decode.
    pub fn push(&mut self, value: u64, aux: u64) -> bool {
        let at = self.rows_valid as usize;
        if at >= self.expected() {
            return false;
        }
        let Some(state) = DeviceState::from_observation((self.first() + at) as u16, value, aux)
        else {
            return false;
        };
        self.states[at] = state;
        self.rows_valid += 1;
        true
    }

    pub fn validate(&self) -> bool {
        let count = self.count.get() as usize;
        if self.page as usize >= DEVICE_STATE_PAGES
            || count > MAX_DEVICE_CENSUS_ROWS
            || self.reserved != [0; 4]
            || self.rows_valid as usize != self.expected()
        {
            return false;
        }
        let valid = self.rows_valid as usize;
        self.states
            .iter()
            .enumerate()
            .all(|(i, state)| state.validate() && state.is_present() == (i < valid))
    }
}

/// One BAR's shape. 16 bytes, frozen. Lengths only: where the BAR lives is
/// not here and never will be.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct DeviceBar {
    /// The BAR's whole size in bytes; 0 for a BAR the function has not.
    pub size: U64<LittleEndian>,
    /// Bytes from the BAR's start a driver may map: the whole BAR, or the
    /// part before the page its MSI-X table or pending bits begin in, which
    /// stay the kernel's. 0 when a driver may map none of it.
    pub mappable: U64<LittleEndian>,
}

impl DeviceBar {
    pub const EMPTY: Self = Self {
        size: U64::ZERO,
        mappable: U64::ZERO,
    };

    /// Bits of `aux` the mappable length takes; the BAR number is echoed
    /// above them.
    const AUX_LENGTH_BITS: u32 = 56;

    pub const fn is_present(&self) -> bool {
        self.size.get() != 0
    }

    pub const fn validate(&self) -> bool {
        self.mappable.get() <= self.size.get()
    }

    /// The BAR as one `SystemObserve` report, `(value, aux)`: its size, and
    /// its mappable length with the BAR number echoed in `aux[58:56]`.
    /// `None` when a length does not leave the echo room, which no BAR of a
    /// machine with a 56-bit physical address space or less can reach.
    pub const fn to_observation(&self, bar: u8) -> Option<(u64, u64)> {
        if self.mappable.get() >> Self::AUX_LENGTH_BITS != 0 || bar as usize >= DEVICE_BARS {
            return None;
        }
        Some((
            self.size.get(),
            self.mappable.get() | (bar as u64) << Self::AUX_LENGTH_BITS,
        ))
    }

    /// Inverse of [`to_observation`](Self::to_observation); `None` when the
    /// echoed BAR is not `bar` or more can be mapped than the BAR holds.
    pub const fn from_observation(bar: u8, value: u64, aux: u64) -> Option<Self> {
        if aux >> Self::AUX_LENGTH_BITS != bar as u64 {
            return None;
        }
        let shape = Self {
            size: U64::new(value),
            mappable: U64::new(aux & ((1 << Self::AUX_LENGTH_BITS) - 1)),
        };
        if shape.validate() { Some(shape) } else { None }
    }
}

/// The BARs of one function. 104 bytes, frozen.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct DeviceBarTable {
    /// The function's catalog index: its row in the census.
    pub index: U16<LittleEndian>,
    /// 1 for a populated table; 0 marks the zero table.
    pub present: u8,
    pub reserved: [u8; 5],
    pub bars: [DeviceBar; DEVICE_BARS],
}

impl DeviceBarTable {
    pub const EMPTY: Self = Self {
        index: U16::ZERO,
        present: 0,
        reserved: [0; 5],
        bars: [DeviceBar::EMPTY; DEVICE_BARS],
    };

    /// The table of the function at catalog `index`, every BAR absent: the
    /// assembler's start.
    pub const fn new(index: u16) -> Self {
        Self {
            index: U16::new(index),
            present: 1,
            reserved: [0; 5],
            bars: [DeviceBar::EMPTY; DEVICE_BARS],
        }
    }

    pub const fn is_present(&self) -> bool {
        self.present == 1
    }

    /// Set BAR `bar` from one kernel observation; `false` when the report
    /// does not decode.
    pub fn set_bar(&mut self, bar: u8, value: u64, aux: u64) -> bool {
        let (Some(slot), Some(shape)) = (
            self.bars.get_mut(bar as usize),
            DeviceBar::from_observation(bar, value, aux),
        ) else {
            return false;
        };
        *slot = shape;
        true
    }

    pub fn validate(&self) -> bool {
        match self.present {
            0 => *self == Self::EMPTY,
            1 => self.reserved == [0; 5] && self.bars.iter().all(DeviceBar::validate),
            _ => false,
        }
    }
}

/// What a device tool is launched with: the census, each of its rows' live
/// state, and the BARs of the one function the command line named, if it
/// named one the census holds. 3008 bytes, inside the process-context
/// prefix (asserted below).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct DeviceInventory {
    /// [`DEVICE_INVENTORY_ABI`]; 0 = unavailable.
    pub abi: U16<LittleEndian>,
    pub reserved: [u8; 6],
    pub census: DeviceCensus,
    /// `states[i]` is the live state of `census.rows[i]`, for every row the
    /// census holds; the zero state past them.
    pub states: [DeviceState; MAX_DEVICE_CENSUS_ROWS],
    /// The named function's BARs, or the zero table.
    pub bars: DeviceBarTable,
}

impl DeviceInventory {
    pub const EMPTY: Self = Self {
        abi: U16::ZERO,
        reserved: [0; 6],
        census: DeviceCensus::EMPTY,
        states: [DeviceState::EMPTY; MAX_DEVICE_CENSUS_ROWS],
        bars: DeviceBarTable::EMPTY,
    };

    /// An available inventory over `census`, with no state and no BARs yet.
    pub const fn new(census: DeviceCensus) -> Self {
        Self {
            abi: U16::new(DEVICE_INVENTORY_ABI),
            reserved: [0; 6],
            census,
            states: [DeviceState::EMPTY; MAX_DEVICE_CENSUS_ROWS],
            bars: DeviceBarTable::EMPTY,
        }
    }

    pub const fn available(&self) -> bool {
        self.abi.get() == DEVICE_INVENTORY_ABI
    }

    /// Fold one vended page of states in; `false`, changing nothing, when
    /// the page does not validate or is not a page of this census's table.
    pub fn insert_states(&mut self, page: &DeviceStatePage) -> bool {
        if !self.available() || !page.validate() || page.count != self.census.count {
            return false;
        }
        let first = page.first();
        let valid = page.rows_valid as usize;
        self.states[first..first + valid].copy_from_slice(&page.states[..valid]);
        true
    }

    /// Whether every census row has its state.
    pub fn complete(&self) -> bool {
        let count = self.census.rows().len();
        self.available() && self.states[..count].iter().all(DeviceState::is_present)
    }

    pub fn validate(&self) -> bool {
        if self.abi.get() == 0 {
            return *self == Self::EMPTY;
        }
        let count = self.census.rows().len();
        self.abi.get() == DEVICE_INVENTORY_ABI
            && self.reserved == [0; 6]
            && self.census.available()
            && self.census.validate()
            && self
                .states
                .iter()
                .enumerate()
                .all(|(i, state)| state.validate() && (i < count || !state.is_present()))
            && self.bars.validate()
            && (!self.bars.is_present() || (self.bars.index.get() as usize) < count)
    }
}

/// Bytes of the device payload a monitor or exec reply carries.
pub const DEVICE_PAGE_BYTES: usize = 136;

/// The device payload slot of the monitor's and exec's reply wires: one
/// census page, one page of states, or one function's BARs, as the reply's
/// tag says, and zero on every other reply. Bytes, so the three records
/// share the slot without one posing as another: each accessor decodes and
/// validates its own.
#[repr(transparent)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct DevicePage(pub [u8; DEVICE_PAGE_BYTES]);

impl DevicePage {
    pub const EMPTY: Self = Self([0; DEVICE_PAGE_BYTES]);

    pub fn of_census(page: &DeviceCensusPage) -> Self {
        let mut out = Self::EMPTY;
        out.0.copy_from_slice(page.as_bytes());
        out
    }

    /// The census page this slot carries, validated.
    pub fn census(&self) -> Option<DeviceCensusPage> {
        DeviceCensusPage::read_from_bytes(&self.0)
            .ok()
            .filter(DeviceCensusPage::validate)
    }

    pub fn of_states(page: &DeviceStatePage) -> Self {
        let mut out = Self::EMPTY;
        out.0.copy_from_slice(page.as_bytes());
        out
    }

    /// The page of states this slot carries, validated.
    pub fn states(&self) -> Option<DeviceStatePage> {
        DeviceStatePage::read_from_bytes(&self.0)
            .ok()
            .filter(DeviceStatePage::validate)
    }

    pub fn of_bars(table: &DeviceBarTable) -> Self {
        let mut out = Self::EMPTY;
        out.0[..core::mem::size_of::<DeviceBarTable>()].copy_from_slice(table.as_bytes());
        out
    }

    /// The BAR table this slot carries: a populated, valid one, with the
    /// slot's remaining bytes zero.
    pub fn bars(&self) -> Option<DeviceBarTable> {
        let (table, rest) = DeviceBarTable::read_from_prefix(&self.0).ok()?;
        (table.is_present() && table.validate() && rest.iter().all(|byte| *byte == 0))
            .then_some(table)
    }
}

/// A PCI function's address as a person writes it: `ssss:bb:dd.f`, or
/// `bb:dd.f` in segment 0, every field hexadecimal but the function, which
/// is one digit `0` to `7`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PciAddress {
    pub segment: u16,
    /// `bus << 8 | device << 3 | function`, a census row's `locator`.
    pub locator: u16,
}

impl PciAddress {
    /// Bytes of the longest spelling, `ssss:bb:dd.f`.
    pub const MAX_TEXT: usize = 12;

    /// Parse `text`, the whole of it. `None` unless it is exactly an
    /// address: a launcher looks for one among a command line's words, so a
    /// word that merely resembles one must not name a device.
    pub fn parse(text: &[u8]) -> Option<Self> {
        let (rest, function) = text.split_last_chunk::<2>()?;
        let [b'.', function @ b'0'..=b'7'] = *function else {
            return None;
        };
        let mut fields = rest.rsplitn(3, |byte| *byte == b':');
        let device = hex_field(fields.next()?, 2)?;
        let bus = hex_field(fields.next()?, 2)?;
        let segment = match fields.next() {
            Some(field) => hex_field(field, 4)?,
            None => 0,
        };
        if device > 0x1F {
            return None;
        }
        Some(Self {
            segment,
            locator: bus << 8 | device << 3 | (function - b'0') as u16,
        })
    }

    /// The census row this address names.
    pub fn row_in(&self, census: &DeviceCensus) -> Option<u16> {
        census
            .rows()
            .iter()
            .position(|row| row.segment.get() == self.segment && row.locator.get() == self.locator)
            .map(|index| index as u16)
    }
}

/// `field` as exactly `digits` hexadecimal digits.
fn hex_field(field: &[u8], digits: usize) -> Option<u16> {
    if field.len() != digits {
        return None;
    }
    field.iter().try_fold(0u16, |value, byte| {
        let digit = (*byte as char).to_digit(16)?;
        Some(value << 4 | digit as u16)
    })
}

const _: () = assert!(core::mem::size_of::<DeviceState>() == 8);
const _: () = assert!(core::mem::size_of::<DeviceStatePage>() == 8 + 8 * DEVICE_STATE_PAGE_ROWS);
const _: () = assert!(
    core::mem::size_of::<DeviceStatePage>() == core::mem::size_of::<super::DeviceCensusPage>()
);
const _: () = assert!(core::mem::size_of::<DeviceCensusPage>() == DEVICE_PAGE_BYTES);
const _: () = assert!(core::mem::size_of::<DeviceStatePage>() == DEVICE_PAGE_BYTES);
const _: () = assert!(core::mem::size_of::<DeviceBarTable>() <= DEVICE_PAGE_BYTES);
const _: () = assert!(core::mem::size_of::<DeviceBar>() == 16);
const _: () = assert!(core::mem::size_of::<DeviceBarTable>() == 8 + 16 * DEVICE_BARS);
const _: () = assert!(core::mem::size_of::<DeviceInventory>() == 3008);
const _: () =
    assert!(core::mem::size_of::<DeviceInventory>() <= crate::process::PROCESS_CONTEXT_BYTES);
const _: () = assert!(DEVICE_STATE_PAGES <= u8::MAX as usize);
// Every index a census row can carry fits the BAR operand beside the BAR.
const _: () = assert!(MAX_DEVICE_CENSUS_ROWS <= 1 << DEVICE_BAR_OPERAND_SHIFT);
const _: () = assert!(DEVICE_BARS <= 1 << (u16::BITS - DEVICE_BAR_OPERAND_SHIFT));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DeviceCensusRow;

    fn state(flags: u8) -> DeviceState {
        DeviceState {
            flags: DeviceState::PRESENT | flags,
            driver: DEVICE_ROW_NONE,
            bars_present: 0b01_0001,
            bars_mappable: 0b00_0001,
            bars_mapped: 0,
            msi_vectors: 1,
            msix_entries: U16::new(0),
        }
    }

    fn census(count: u16) -> DeviceCensus {
        let mut census = DeviceCensus::new(count, 1);
        for index in 0..count {
            census.rows[index as usize] = DeviceCensusRow {
                locator: U16::new(index << 3),
                present: 1,
                ..DeviceCensusRow::EMPTY
            };
        }
        census
    }

    #[test]
    fn a_state_round_trips_one_observation() {
        let mut state = state(DeviceState::CLAIMED | DeviceState::ROUTE_MSI | DeviceState::HAS_PIN);
        state.flags |= 2 << DeviceState::PIN_SHIFT;
        state.driver = 7;
        state.msix_entries = U16::new(0);
        let (value, aux) = state.to_observation(41);
        assert_eq!(DeviceState::from_observation(41, value, aux), Some(state));
        assert_eq!(state.pin(), Some(2));
        assert_eq!(state.driver(), Some(7));
        assert!(state.claimed());
        // Another function's report, or one with stray auxiliary bits, is
        // not this function's state.
        assert_eq!(DeviceState::from_observation(40, value, aux), None);
        assert_eq!(DeviceState::from_observation(41, value, aux | 1), None);
    }

    #[test]
    fn an_impossible_state_is_refused() {
        let ok = state(0);
        assert!(ok.validate());
        // The zero state is the empty row, never a kernel answer.
        assert!(DeviceState::EMPTY.validate());
        assert_eq!(DeviceState::from_observation(0, 0, 0), None);
        for broken in [
            DeviceState {
                bars_present: 0x40,
                ..ok
            },
            DeviceState {
                bars_mappable: 0b10,
                ..ok
            },
            DeviceState {
                bars_mapped: 0b01_0000,
                ..ok
            },
            DeviceState {
                msi_vectors: 3,
                ..ok
            },
            DeviceState {
                msi_vectors: 64,
                ..ok
            },
            DeviceState {
                flags: ok.flags | 1 << DeviceState::PIN_SHIFT,
                ..ok
            },
            DeviceState {
                msix_entries: U16::new(crate::device::DeviceRecord::MAX_MSIX_TABLE_ENTRIES + 1),
                ..ok
            },
        ] {
            assert!(!broken.validate(), "{broken:?}");
        }
    }

    #[test]
    fn a_page_takes_exactly_its_rows_in_order() {
        // 20 states: page 0 holds 16, page 1 the last 4, and there is no
        // page 2.
        let mut first = DeviceStatePage::new(0, 20).unwrap();
        for index in 0..16u16 {
            let (value, aux) = state(0).to_observation(index);
            assert!(first.push(value, aux));
        }
        let (value, aux) = state(0).to_observation(16);
        assert!(!first.push(value, aux), "a full page takes no more");
        assert!(first.validate());

        let mut second = DeviceStatePage::new(1, 20).unwrap();
        // The next state is index 16's; index 17's report is out of order.
        let (value, aux) = state(0).to_observation(17);
        assert!(!second.push(value, aux));
        assert!(!second.validate(), "a short page is not a page");
        for index in 16..20u16 {
            let (value, aux) = state(0).to_observation(index);
            assert!(second.push(value, aux));
        }
        assert!(second.validate());
        assert!(DeviceStatePage::new(2, 20).is_none());
        // A machine with no function still has an empty page 0.
        assert!(DeviceStatePage::new(0, 0).unwrap().validate());
        assert!(DeviceStatePage::new(1, 0).is_none());
    }

    #[test]
    fn a_bar_round_trips_and_never_maps_more_than_it_holds() {
        let bar = DeviceBar {
            size: U64::new(0x10_0000),
            mappable: U64::new(0x8000),
        };
        let (value, aux) = bar.to_observation(4).unwrap();
        assert_eq!(DeviceBar::from_observation(4, value, aux), Some(bar));
        assert_eq!(DeviceBar::from_observation(3, value, aux), None);
        // An absent BAR is the zero shape, and still answers for its number.
        let (value, aux) = DeviceBar::EMPTY.to_observation(5).unwrap();
        assert_eq!(
            DeviceBar::from_observation(5, value, aux),
            Some(DeviceBar::EMPTY)
        );
        assert_eq!(DeviceBar::from_observation(0, 0x1000, 0x2000), None);
        assert!(bar.to_observation(6).is_none());
        let wide = DeviceBar {
            size: U64::new(u64::MAX),
            mappable: U64::new(1 << 56),
        };
        assert!(wide.to_observation(0).is_none());
    }

    #[test]
    fn a_bar_operand_names_a_function_and_a_bar() {
        let operand = device_bar_operand(119, 5).unwrap();
        assert_eq!(device_bar_of(operand), Some((119, 5)));
        assert_eq!(device_bar_operand(0, 0), Some(0));
        assert!(device_bar_operand(1 << DEVICE_BAR_OPERAND_SHIFT, 0).is_none());
        assert!(device_bar_operand(0, 6).is_none());
        assert!(device_bar_of(7 << DEVICE_BAR_OPERAND_SHIFT).is_none());
    }

    #[test]
    fn an_inventory_holds_a_state_per_census_row() {
        let mut inventory = DeviceInventory::new(census(20));
        assert!(inventory.validate());
        assert!(!inventory.complete());
        for page in 0..2u8 {
            let mut states = DeviceStatePage::new(page, 20).unwrap();
            for index in states.first()..states.first() + states.expected() {
                let (value, aux) = state(0).to_observation(index as u16);
                assert!(states.push(value, aux));
            }
            assert!(inventory.insert_states(&states));
        }
        assert!(inventory.complete());
        assert!(inventory.validate());
        // A page of another census's table is refused whole.
        let other = DeviceStatePage::new(0, 21).unwrap();
        assert!(!inventory.insert_states(&other));

        let mut bars = DeviceBarTable::new(19);
        let shape = DeviceBar {
            size: U64::new(0x4000),
            mappable: U64::new(0x4000),
        };
        let (value, aux) = shape.to_observation(0).unwrap();
        assert!(bars.set_bar(0, value, aux));
        assert!(!bars.set_bar(1, value, aux), "the echo names BAR 0");
        inventory.bars = bars;
        assert!(inventory.validate());
        // BARs of a function the census does not hold are not this
        // inventory's.
        inventory.bars.index = U16::new(20);
        assert!(!inventory.validate());
        assert!(DeviceInventory::EMPTY.validate());
        assert!(!DeviceInventory::EMPTY.available());
    }

    #[test]
    fn a_page_slot_decodes_only_the_record_it_carries() {
        let mut states = DeviceStatePage::new(0, 2).unwrap();
        for index in 0..2u16 {
            let (value, aux) = state(0).to_observation(index);
            assert!(states.push(value, aux));
        }
        let slot = DevicePage::of_states(&states);
        assert_eq!(slot.states(), Some(states));

        let mut table = DeviceBarTable::new(3);
        let shape = DeviceBar {
            size: U64::new(0x1000),
            mappable: U64::new(0x1000),
        };
        let (value, aux) = shape.to_observation(2).unwrap();
        assert!(table.set_bar(2, value, aux));
        let slot = DevicePage::of_bars(&table);
        assert_eq!(slot.bars(), Some(table));
        // Bytes past the table are not part of it.
        let mut stray = slot;
        stray.0[DEVICE_PAGE_BYTES - 1] = 1;
        assert_eq!(stray.bars(), None);

        let page = census(3).page(0).unwrap();
        assert_eq!(DevicePage::of_census(&page).census(), Some(page));
        // The zero slot is no record at all.
        assert_eq!(DevicePage::EMPTY.bars(), None);
        assert_eq!(
            DevicePage::EMPTY.census().map(|page| page.rows_valid),
            Some(0)
        );
    }

    #[test]
    fn an_address_is_exactly_an_address() {
        let address = PciAddress::parse(b"0000:00:1f.3").unwrap();
        assert_eq!((address.segment, address.locator), (0, 0x1F << 3 | 3));
        assert_eq!(PciAddress::parse(b"00:1f.3"), Some(address));
        assert_eq!(
            PciAddress::parse(b"0001:ab:02.0"),
            Some(PciAddress {
                segment: 1,
                locator: 0xAB << 8 | 2 << 3
            })
        );
        for not in [
            &b""[..],
            b"1f.3",
            b"00:1f",
            b"00:1f.8",
            b"00:20.0",
            b"0:1f.3",
            b"000:00:1f.3",
            b"00:1f.3x",
            b"x0:1f.3",
            b"0000:00:1f:3",
            b"00000:00:1f.3",
            b"8086:a828",
        ] {
            assert_eq!(
                PciAddress::parse(not),
                None,
                "{:?}",
                core::str::from_utf8(not)
            );
        }
        let census = census(4);
        assert_eq!(
            PciAddress {
                segment: 0,
                locator: 3 << 3
            }
            .row_in(&census),
            Some(3)
        );
        assert_eq!(
            PciAddress {
                segment: 1,
                locator: 3 << 3
            }
            .row_in(&census),
            None
        );
    }
}
