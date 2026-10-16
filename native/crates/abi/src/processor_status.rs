//! Bounded, read-only processor observations. No field grants ISA authority.
use zerocopy::byteorder::little_endian::{U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

/// Keep status pages at 17 rows. More rows would overflow the bounded text
/// reply even though the process context now has room for a larger catalogue.
/// The header occupies 32 bytes; rows include owner-local counter evidence.
pub const ROWS: usize = 17;
/// Minimum spacing between owner-local readings, even for overlapping requests.
pub const SAMPLE_PERIOD_NS: u64 = 250_000_000;
/// One shared half-second capture for all cores: time for an owner timer to
/// take a baseline and a second reading one period later. Late cores report
/// unavailable; requesting a snapshot does not add sampling interrupts.
pub const CAPTURE_WINDOW_NS: u64 = 2 * SAMPLE_PERIOD_NS;
/// Completed observations remain readable for four nominal periods.
pub const MAX_SAMPLE_AGE_NS: u64 = 4 * SAMPLE_PERIOD_NS;

/// The monitor's processor request. A busy interval never touches the CPU PMU;
/// only an explicit counters selector asks the owner to read it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ProcessorCapture {
    Read = 0,
    Busy = 1,
    Counters = 2,
}

impl ProcessorCapture {
    pub const fn from_wire(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Read),
            1 => Some(Self::Busy),
            2 => Some(Self::Counters),
            _ => None,
        }
    }

    pub const fn captures(self) -> bool {
        !matches!(self, Self::Read)
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
pub struct ProcessorRow {
    pub cpu: U16,
    pub class: u8,
    pub vector_bytes: u8,
    pub numa: U32,
    pub hwid: U64,
    /// Socket, cluster, core (16 bits each), SMT thread (8 bits).
    /// Bit 63 marks an explicit NUMA proximity domain; bits 56..62 are zero.
    pub topology: U64,
    /// Non-idle time during the owner-published window; zero when unavailable.
    pub busy_ns: U64,
    /// Owner-local publication timestamp, potentially preceding this read.
    pub sampled_ns: U64,
    /// Zero means no interval measurement is available for this row.
    pub window_ns: U64,
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
    /// This CPU's solo boot DRAM fill and copy rates, in STREAM_UNIT_BPS.
    pub stream_write: U16,
    pub stream_copy: U16,
    pub counters: CounterSample,
}

/// Bytes per second per unit of the compact boot stream fields.
pub const STREAM_UNIT_BPS: u64 = 100_000_000;

impl ProcessorRow {
    pub fn set_stream(&mut self, write_bps: u64, copy_bps: u64) {
        let unit = |bps: u64| U16::new((bps / STREAM_UNIT_BPS).min(u64::from(u16::MAX)) as u16);
        self.stream_write = unit(write_bps);
        self.stream_copy = unit(copy_bps);
    }

    pub fn stream_bps(&self) -> (u64, u64) {
        (
            u64::from(self.stream_write.get()) * STREAM_UNIT_BPS,
            u64::from(self.stream_copy.get()) * STREAM_UNIT_BPS,
        )
    }

    /// Keep static topology, but discard observations outside this request.
    pub fn retain_window(&mut self, start: u64, end: u64) {
        let baseline = self.sampled_ns.get().checked_sub(self.window_ns.get());
        if self.window_ns.get() == 0
            || baseline.is_none_or(|t| t < start)
            || self.sampled_ns.get() > end
        {
            self.window_ns.set(0);
            self.busy_ns.set(0);
            self.counters = CounterSample::default();
        }
    }
}

/// Raw owner publication, or interval deltas when the enclosing row has a window.
/// `at_ns` is the owner timestamp for raw samples, elapsed time for deltas.
/// Profiles: 1 architectural Intel; 2 PTL P; 3 PTL E/LP-E. Zero = unavailable.
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
pub struct CounterSample {
    pub at_ns: U64,
    /// Profile in byte 0, hardware counter width in byte 1; upper bits reserved.
    pub config: U64,
    /// Instructions, cycles, FP event, L2 misses, extra and traffic (profile-specific).
    pub values: [U64; 6],
}
impl CounterSample {
    pub fn validate(&self) -> bool {
        let config = self.config.get();
        if config == 0 {
            return *self == Self::default();
        }
        let width = (config >> 8) as u32;
        if config >> 16 != 0
            || !(48..=64).contains(&width)
            || !(1..=3).contains(&(config & 255))
            || self.at_ns.get() == 0
        {
            return false;
        }
        let mask = u64::MAX >> (64 - width);
        self.values.iter().all(|v| v.get() <= mask)
            && (config & 255 != 1 || self.values[2..].iter().all(|v| v.get() == 0))
    }
    pub fn interval_since(&self, before: &Self) -> Option<Self> {
        if !self.validate()
            || !before.validate()
            || self.config.get() == 0
            || self.config != before.config
        {
            return None;
        }
        let elapsed = self.at_ns.get().checked_sub(before.at_ns.get())?;
        if elapsed == 0 {
            return None;
        }
        let mask = u64::MAX >> (64 - (self.config.get() >> 8));
        let mut out = *self;
        out.at_ns.set(elapsed);
        for (value, old) in out.values.iter_mut().zip(before.values) {
            let delta = value.get().wrapping_sub(old.get()) & mask;
            // Half-range ordering rejects backwards/reset-like deltas. This
            // assumes less than half a counter range elapses between samples;
            // multiple complete wraps cannot be identified from two readings.
            if delta > mask / 2 {
                return None;
            }
            value.set(delta);
        }
        Some(out)
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
    /// A capture acknowledgement carries a bounded deadline, never CPU rows.
    pub fn validate_capture(&self) -> bool {
        self.available.get() == 1
            && self.reserved.get() == 0
            && self.count.get() == 0
            && u32::from(self.first.get()) <= self.total.get()
            && self.total.get() <= 65536
            && self.started_ns.get() != 0
            && self.finished_ns.get().checked_sub(self.started_ns.get()) == Some(CAPTURE_WINDOW_NS)
            && self.rows.iter().all(|row| *row == ProcessorRow::default())
    }

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
                && (row.window_ns.get() == 0
                    || (row.busy_ns.get() <= row.window_ns.get()
                        && row.window_ns.get() <= row.sampled_ns.get()))
                && row.class <= 3
                && matches!(row.vector_bytes, 0 | 16 | 32 | 64)
                && row.counters.validate()
                && row.topology.get() & 0x7f00_0000_0000_0000 == 0
                && (row.topology.get() >> 63 != 0 || row.numa.get() == 0)
        })
    }
}

/// Status selectors that need a processor page. Input and other scopes retain
/// their existing context. Decimal page starts are explicit CPU ordinals.
pub fn requested_processor(args: &[u8]) -> Option<(u16, ProcessorCapture)> {
    let mut words = args
        .split(|b| b.is_ascii_whitespace())
        .filter(|w| !w.is_empty());
    let scope = words.next()?;
    match scope {
        b"sys" | b"cpu" => {
            let next = words.next();
            let (mode, next) = match (scope, next) {
                (b"cpu", Some(b"counters")) => (ProcessorCapture::Counters, words.next()),
                (b"cpu", Some(b"detail")) => (ProcessorCapture::Read, words.next()),
                (b"sys" | b"cpu", next) => (ProcessorCapture::Busy, next),
                _ => return None,
            };
            let first = match next {
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
            words.next().is_none().then_some((first, mode))
        }
        _ => None,
    }
}

pub fn requested_first(args: &[u8]) -> Option<u16> {
    requested_processor(args).map(|(first, _)| first)
}

const _: () =
    assert!(core::mem::size_of::<ProcessorSnapshot>() <= crate::process::PROCESS_CONTEXT_BYTES);

#[cfg(test)]
mod tests {
    #[test]
    fn capture_header_and_result_windows_are_distinct() {
        let mut page = super::ProcessorPage::default();
        page.available.set(1);
        page.total.set(16);
        page.started_ns.set(1);
        page.finished_ns.set(1 + super::CAPTURE_WINDOW_NS);
        assert!(page.validate_capture());
        assert!(!page.validate());
        page.rows[0].cpu.set(1);
        assert!(!page.validate_capture());
    }

    #[test]
    fn retaining_window_rejects_stale_future_and_underflow_but_keeps_topology() {
        for (sample, duration, kept) in [
            (20, 10, true),
            (19, 10, false),
            (21, 10, false),
            (2, 10, false),
            (20, 0, false),
        ] {
            let mut row = super::ProcessorRow::default();
            row.cpu.set(7);
            row.l2.set(4096);
            row.sampled_ns.set(sample);
            row.window_ns.set(duration);
            row.busy_ns.set(5);
            row.counters.config.set(1);
            row.retain_window(10, 20);
            assert_eq!(row.window_ns.get() != 0, kept);
            assert_eq!(row.busy_ns.get(), if kept { 5 } else { 0 });
            assert_eq!(row.counters.config.get(), if kept { 1 } else { 0 });
            assert_eq!((row.cpu.get(), row.l2.get()), (7, 4096));
        }
    }

    use super::*;
    #[test]
    fn counter_windows_distinguish_wrap_reset_and_missing() {
        let mask = (1u64 << 48) - 1;
        let a = CounterSample {
            at_ns: 100.into(),
            config: (2 | 48 << 8).into(),
            values: [
                0.into(),
                0.into(),
                (mask - 3).into(),
                100.into(),
                0.into(),
                0.into(),
            ],
        };
        let mut b = a;
        b.at_ns.set(200);
        b.values[2].set(5);
        b.values[3].set(104);
        let d = b.interval_since(&a).unwrap();
        assert_eq!(d.at_ns.get(), 100);
        assert_eq!(d.values[2].get(), 9);
        b.values[3].set(0); // reset is not a legitimate small wrapped delta
        assert!(b.interval_since(&a).is_none());
        assert!(a.interval_since(&a).is_none());
        assert!(a.interval_since(&CounterSample::default()).is_none());
        b.config.set(3 | 48 << 8);
        assert!(b.interval_since(&a).is_none());
        b.config.set(2 | 47 << 8);
        assert!(!b.validate());
        b.config.set(2 | 48 << 8);
        b.values[2].set(1 << 48);
        assert!(!b.validate());
    }
    #[test]
    fn pages_are_complete_ordered_and_canonical() {
        assert_eq!(core::mem::size_of::<ProcessorPage>(), 328);
        assert_eq!(core::mem::size_of::<ProcessorSnapshot>(), 2548);
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
        p.rows[0].counters.config.set(1);
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
        p.rows[1].l2.set(1);
        assert!(!p.validate());
    }
    #[test]
    fn selectors_and_unavailable_are_explicit() {
        assert_eq!(
            requested_processor(b"cpu"),
            Some((0, ProcessorCapture::Busy))
        );
        assert_eq!(
            requested_processor(b"sys 8"),
            Some((8, ProcessorCapture::Busy))
        );
        assert_eq!(
            requested_processor(b"cpu counters 8"),
            Some((8, ProcessorCapture::Counters))
        );
        assert_eq!(
            requested_processor(b"cpu detail"),
            Some((0, ProcessorCapture::Read))
        );
        assert_eq!(requested_processor(b"sys counters"), None);
        assert_eq!(ProcessorCapture::from_wire(3), None);
        for args in [b"cpu".as_slice(), b"sys"] {
            assert_eq!(requested_first(args), Some(0));
        }
        assert_eq!(requested_first(b""), None);
        assert_eq!(requested_first(b"cpu 8"), Some(8));
        assert_eq!(requested_first(b"cpu counters"), Some(0));
        assert_eq!(requested_first(b"cpu counters 8"), Some(8));
        assert_eq!(requested_first(b"cpu counters extra"), None);
        for args in [b"cpu -1".as_slice(), b"cpu 65536", b"sys 0 extra", b"input"] {
            assert_eq!(requested_first(args), None);
        }
        let mut s: ProcessorSnapshot = ProcessorSnapshot::default();
        assert!(s.validate());
        s.total.set(1);
        assert!(!s.validate());
    }
}
