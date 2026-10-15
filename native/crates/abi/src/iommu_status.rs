//! Bounded IOMMU observations. Addresses in fault records are IOVAs, never
//! host physical addresses; these records grant no device authority.
use zerocopy::byteorder::little_endian::{U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

/// One controller plus the complete fault history fits the existing observation wire page.
pub const UNITS: usize = 1;
/// One second to collect immutable controller metadata and bounded fault history.
pub const OBSERVATION_BUDGET_NS: u64 = 1_000_000_000;
pub const FAULTS: usize = crate::session::IOMMU_FAULT_RING as usize;
pub const SEGMENT_KNOWN: u8 = 1;
pub const NUMA_KNOWN: u8 = 2;
pub const INCLUDE_ALL: u8 = 4;

#[repr(C)]
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    FromBytes,
    IntoBytes,
    KnownLayout,
    Immutable,
    Unaligned,
)]
pub struct Unit {
    pub id: U32,
    /// 0 unknown, 1 VT-d, 2 AMD-Vi, 3 SMMUv2, 4 SMMUv3, 5 RISC-V.
    pub kind: u8,
    /// 1 stage 1, 2 stage 2, 3 nested (configured descriptor).
    pub stage: u8,
    /// log2(maximum lawful leaf bytes); zero means format-limited.
    pub leaf_shift: u8,
    pub flags: u8,
    pub segment: U32,
    pub numa: U32,
    /// Translation, PASID, ATS, device TLB, interrupt remapping, nested
    /// translation, page requests, dirty tracking, respectively in bits 0..7.
    /// Hardware capabilities, not proof that each feature is enabled.
    pub capabilities: U64,
    pub pci_scopes: U16,
    pub stream_scopes: U16,
    pub reservations: U32,
}

impl Unit {
    pub fn identity(self) -> (u64, u64) {
        (
            u64::from(self.id.get())
                | u64::from(self.kind) << 32
                | u64::from(self.stage) << 40
                | u64::from(self.leaf_shift) << 48
                | u64::from(self.flags) << 56,
            u64::from(self.segment.get()) | u64::from(self.numa.get()) << 32,
        )
    }
    pub fn features(self) -> (u64, u64) {
        (
            self.capabilities.get(),
            u64::from(self.pci_scopes.get())
                | u64::from(self.stream_scopes.get()) << 16
                | u64::from(self.reservations.get()) << 32,
        )
    }
    pub fn from_observations(identity: (u64, u64), features: (u64, u64)) -> Option<Self> {
        let row = Self {
            id: (identity.0 as u32).into(),
            kind: (identity.0 >> 32) as u8,
            stage: (identity.0 >> 40) as u8,
            leaf_shift: (identity.0 >> 48) as u8,
            flags: (identity.0 >> 56) as u8,
            segment: (identity.1 as u32).into(),
            numa: ((identity.1 >> 32) as u32).into(),
            capabilities: features.0.into(),
            pci_scopes: (features.1 as u16).into(),
            stream_scopes: ((features.1 >> 16) as u16).into(),
            reservations: ((features.1 >> 32) as u32).into(),
        };
        row.validate().then_some(row)
    }
    pub fn validate(&self) -> bool {
        self.kind <= 5
            && (1..=3).contains(&self.stage)
            && (self.leaf_shift == 0 || (12..64).contains(&self.leaf_shift))
            && self.flags & !(SEGMENT_KNOWN | NUMA_KNOWN | INCLUDE_ALL) == 0
            && self.capabilities.get() & !0xff == 0
            && if self.flags & SEGMENT_KNOWN != 0 {
                self.segment.get() <= u16::MAX.into()
            } else {
                self.segment.get() == 0
            }
            && (self.flags & NUMA_KNOWN != 0 || self.numa.get() == 0)
    }
}

#[repr(C)]
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    FromBytes,
    IntoBytes,
    KnownLayout,
    Immutable,
    Unaligned,
)]
pub struct Fault {
    pub packed: U64,
    /// Existing IommuFaultRecord observation: drain sequence and timestamp.
    pub stamp: U64,
}

#[repr(C)]
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    FromBytes,
    IntoBytes,
    KnownLayout,
    Immutable,
    Unaligned,
)]
pub struct Snapshot {
    pub available: u8,
    pub count: u8,
    pub fault_count: u8,
    /// The census changed while reading history; history is best effort.
    pub changed: u8,
    pub first: U16,
    pub reserved: U16,
    pub total: U32,
    pub unresponsive: U32,
    pub captured_ns: U64,
    pub census: U64,
    pub units: [Unit; UNITS],
    pub faults: [Fault; FAULTS],
}

impl Snapshot {
    /// Alternate tagged view of the existing observation payload. Padding is zero.
    pub fn to_wire_page(&self) -> crate::processor_status::ProcessorPage {
        let mut page = crate::processor_status::ProcessorPage::default();
        page.as_mut_bytes()[..core::mem::size_of::<Self>()].copy_from_slice(self.as_bytes());
        page
    }
    pub fn from_wire_page(page: &crate::processor_status::ProcessorPage) -> Option<Self> {
        let (s, tail) = Self::read_from_prefix(page.as_bytes()).ok()?;
        (s.validate() && tail.iter().all(|b| *b == 0)).then_some(s)
    }
    pub fn validate(&self) -> bool {
        self.available == 1
            && self.changed <= 1
            && self.reserved.get() == 0
            && u32::from(self.first.get()) <= self.total.get()
            && usize::from(self.count)
                == (self.total.get() - u32::from(self.first.get())).min(UNITS as u32) as usize
            && usize::from(self.fault_count) <= FAULTS
            && self.units[..usize::from(self.count)]
                .iter()
                .all(Unit::validate)
            && self.units[usize::from(self.count)..]
                .iter()
                .all(|u| *u == Unit::default())
            && self.faults[usize::from(self.fault_count)..]
                .iter()
                .all(|f| *f == Fault::default())
    }
}

/// Shared by the launcher and renderer; malformed pagination never launches a
/// different observation accidentally.
pub fn requested_first(args: &[u8]) -> Option<u16> {
    let text = core::str::from_utf8(args).ok()?;
    let mut words = text.split_ascii_whitespace();
    if words.next()? != "iommu" {
        return None;
    }
    let first = match words.next() {
        Some(s) => s.parse().ok()?,
        None => 0,
    };
    words.next().is_none().then_some(first)
}

const _: () = assert!(
    core::mem::size_of::<Snapshot>()
        <= core::mem::size_of::<crate::processor_status::ProcessorPage>()
);
const _: () = assert!(core::mem::size_of::<Snapshot>() <= crate::process::PROCESS_CONTEXT_BYTES);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unavailable_is_not_an_empty_machine_and_pages_are_canonical() {
        assert!(!Snapshot::default().validate());
        let mut s = Snapshot {
            available: 1,
            ..Default::default()
        };
        assert!(s.validate());
        s.first.set(1);
        assert!(!s.validate());
        s.first.set(0);
        s.fault_count = (FAULTS + 1) as u8;
        assert!(!s.validate());
    }
    #[test]
    fn unit_words_preserve_capabilities_scopes_and_unknown_locality() {
        let u = Unit {
            id: 7.into(),
            kind: 2,
            stage: 2,
            leaf_shift: 21,
            capabilities: 0xad.into(),
            pci_scopes: 12.into(),
            stream_scopes: 3.into(),
            reservations: 9.into(),
            ..Default::default()
        };
        assert_eq!(Unit::from_observations(u.identity(), u.features()), Some(u));
        let mut invalid = u;
        invalid.segment.set(1);
        assert!(!invalid.validate());
    }
    #[test]
    fn selector_is_exact_and_bounded() {
        assert_eq!(requested_first(b"iommu"), Some(0));
        assert_eq!(requested_first(b" iommu 4 "), Some(4));
        for bad in [
            b"iommu 65536".as_slice(),
            b"iommu 4 extra",
            b"sys",
            b"iommu -1",
        ] {
            assert_eq!(requested_first(bad), None);
        }
    }
}
