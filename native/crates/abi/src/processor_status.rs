//! Bounded, read-only processor observations. No field grants ISA authority.
use zerocopy::byteorder::little_endian::{U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

pub const ROWS: usize = 8;

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
pub struct ProcessorRow {
    pub cpu: U16,
    pub class: u8,
    pub vector_bytes: u8,
    pub numa: U32,
    pub hwid: U64,
    /// Socket, cluster, core (16 bits each), SMT thread (8 bits).
    /// Bit 63 marks an explicit NUMA proximity domain; bits 56..62 are zero.
    pub topology: U64,
    pub busy_ns: U64,
    pub l1d: U32,
    pub l2: U32,
    pub l3: U32,
    pub line_bytes: U16,
    pub l3_sharers: U16,
    pub capacity_q16: U32,
    /// bit16: valid hardware feedback; low bytes: performance/efficiency.
    pub feedback: U32,
    pub nominal_mhz: U32,
    pub maximum_mhz: U32,
    /// This CPU's boot DRAM stream, alone (`SystemMetric::ProcessorStream`),
    /// in [`STREAM_UNIT_BPS`] units: fill rate, then copy rate.  Zero =
    /// unmeasured.
    pub stream_write: U16,
    pub stream_copy: U16,
}

/// Bytes per second per unit of [`ProcessorRow::stream_write`] /
/// [`ProcessorRow::stream_copy`] (100 MB/s, so 6.5 TB/s per word).
pub const STREAM_UNIT_BPS: u64 = 100_000_000;

impl ProcessorRow {
    /// Attach the CPU's solo DRAM stream (bytes per second, saturating).
    pub fn set_stream(&mut self, write_bps: u64, copy_bps: u64) {
        let unit = |bps: u64| U16::new((bps / STREAM_UNIT_BPS).min(u64::from(u16::MAX)) as u16);
        self.stream_write = unit(write_bps);
        self.stream_copy = unit(copy_bps);
    }
    /// `(write, copy)` bytes per second; `(0, 0)` when unmeasured.
    pub fn stream_bps(&self) -> (u64, u64) {
        (
            u64::from(self.stream_write.get()) * STREAM_UNIT_BPS,
            u64::from(self.stream_copy.get()) * STREAM_UNIT_BPS,
        )
    }
}

#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct ProcessorSnapshot<const N: usize = ROWS> {
    /// 1 = available; 0 means the entire record is zero/unavailable.
    pub available: U32,
    pub first: U16,
    pub count: U16,
    pub total: U32,
    pub reserved: U32,
    /// Bounds of this sequential observation, not an atomic machine census.
    pub started_ns: U64,
    pub finished_ns: U64,
    pub rows: [ProcessorRow; N],
}

impl<const N: usize> Default for ProcessorSnapshot<N> {
    fn default() -> Self {
        Self {
            available: 0.into(),
            first: 0.into(),
            count: 0.into(),
            total: 0.into(),
            reserved: 0.into(),
            started_ns: 0.into(),
            finished_ns: 0.into(),
            rows: [ProcessorRow::default(); N],
        }
    }
}

/// Small transport chunk preserves the existing exec ring reservation.
pub type ProcessorPage = ProcessorSnapshot<2>;

impl<const N: usize> ProcessorSnapshot<N> {
    pub fn validate(&self) -> bool {
        if self.available.get() == 0 {
            return *self == Self::default();
        }
        if self.available.get() != 1
            || self.reserved.get() != 0
            || self.count.get() as usize > N
            || u32::from(self.first.get()) > self.total.get()
            || self.total.get() > 65536
            || u32::from(self.first.get()) + u32::from(self.count.get()) > self.total.get()
            || self.finished_ns.get() < self.started_ns.get()
        {
            return false;
        }
        if self.count.get() as usize
            != N.min((self.total.get() - u32::from(self.first.get())) as usize)
        {
            return false;
        }
        self.rows.iter().enumerate().all(|(i, row)| {
            if i >= self.count.get() as usize {
                return *row == ProcessorRow::default();
            }
            row.cpu.get() as usize == self.first.get() as usize + i
                && row.class <= 3
                && matches!(row.vector_bytes, 0 | 16 | 32 | 64)
                && row.feedback.get() & !0x1ffff == 0
                && (row.feedback.get() & 0x10000 != 0 || row.feedback.get() == 0)
                && row.topology.get() & 0x7f00_0000_0000_0000 == 0
                && (row.topology.get() >> 63 != 0 || row.numa.get() == 0)
        })
    }
}

/// Status selectors that need a processor page. Input and other scopes retain
/// their existing context. Decimal page starts are explicit CPU ordinals.
pub fn requested_first(args: &[u8]) -> Option<u16> {
    let mut words = args
        .split(|b| b.is_ascii_whitespace())
        .filter(|w| !w.is_empty());
    match words.next() {
        None => Some(0),
        Some(b"sys" | b"cpu") => {
            let first = match words.next() {
                None => 0,
                Some(word) => {
                    if word.is_empty() {
                        return None;
                    }
                    let mut n = 0u16;
                    for b in word {
                        if !b.is_ascii_digit() {
                            return None;
                        }
                        n = n.checked_mul(10)?.checked_add(u16::from(b - b'0'))?;
                    }
                    n
                }
            };
            words.next().is_none().then_some(first)
        }
        _ => None,
    }
}

const _: () =
    assert!(core::mem::size_of::<ProcessorSnapshot>() <= crate::process::PROCESS_CONTEXT_BYTES);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pages_are_complete_ordered_and_canonical() {
        assert_eq!(core::mem::size_of::<ProcessorPage>(), 168);
        assert_eq!(core::mem::size_of::<ProcessorSnapshot>(), 576);
        let mut p = ProcessorPage::default();
        p.available.set(1);
        p.first.set(8);
        p.total.set(11);
        p.count.set(2);
        p.rows[0].cpu.set(8);
        p.rows[1].cpu.set(9);
        assert!(p.validate());
        let good = p;
        p.count.set(0);
        assert!(!p.validate());
        p = good;
        p.rows[1].cpu.set(8);
        assert!(!p.validate());
        p = good;
        p.rows[0].feedback.set(1);
        assert!(!p.validate());
        p = good;
        p.rows[0].topology.set(1 << 56);
        assert!(!p.validate());
        p = good;
        p.started_ns.set(1);
        assert!(!p.validate());
        p = ProcessorPage::default();
        p.available.set(1);
        p.first.set(11);
        p.total.set(11);
        assert!(p.validate());
        p.rows[1].l1d.set(1);
        assert!(!p.validate());
    }
    #[test]
    fn selectors_and_unavailable_are_explicit() {
        for args in [b"".as_slice(), b"cpu", b"sys"] {
            assert_eq!(requested_first(args), Some(0));
        }
        assert_eq!(requested_first(b"cpu 8"), Some(8));
        for args in [b"cpu -1".as_slice(), b"cpu 65536", b"sys 0 extra", b"input"] {
            assert_eq!(requested_first(args), None);
        }
        let mut s: ProcessorSnapshot = ProcessorSnapshot::default();
        assert!(s.validate());
        s.total.set(1);
        assert!(!s.validate());
    }
}
