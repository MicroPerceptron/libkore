//! Keyboard focus routing in the existing founded terminal echo lease.
//!
//! The OutChunk ring remains first and unchanged. Disjoint tail regions carry
//! one atomic route mailbox and a bounded raw-snapshot ring. The founder alone
//! initializes them; input-stream owns echo/raw production and the compositor
//! owns focus publication and consumption. IDs and epochs convey no authority.
//!
//! The mailbox also carries the seat's **capture intent** (#1414): the
//! PrintScreen chord is a trusted key that input-stream takes out of every
//! route and turns into a request here, so no application can see, swallow
//! or forge it. Like the route, it is a level word, not a queued event: a
//! dropped advisory kick cannot lose it.

use core::sync::atomic::{AtomicU64, Ordering};
use kore_abi::svc::{InputReport, OutChunk};
use kore_ring::spsc::{self, RingError};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned, little_endian::U64};

pub const ECHO_CAP: u32 = 32;
pub const RELAY_CAP: u32 = 64;
pub const ECHO_BYTES: usize = spsc::region_bytes::<OutChunk>(ECHO_CAP);
pub const FOCUS_BYTES: usize = 64;
pub const RELAY_BYTES: usize = spsc::region_bytes::<SnapshotWire>(RELAY_CAP);
pub const WINDOW_BYTES: usize = ECHO_BYTES + FOCUS_BYTES + RELAY_BYTES;
const MAGIC: u64 = u64::from_le_bytes(*b"koreFCS1");
const MAX_EPOCH: u64 = u64::MAX >> 1;
/// The capture word is `seq << 8 | mode`: a 56-bit press sequence.
const MAX_CAPTURE_SEQ: u64 = u64::MAX >> 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FocusTarget {
    Terminal,
    Window,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Focus {
    pub epoch: u64,
    pub target: FocusTarget,
}
impl Focus {
    pub const INITIAL: Self = Self {
        epoch: 1,
        target: FocusTarget::Terminal,
    };
    fn decode(word: u64) -> Option<Self> {
        (word >> 1 != 0).then_some(Self {
            epoch: word >> 1,
            target: if word & 1 == 0 {
                FocusTarget::Terminal
            } else {
                FocusTarget::Window
            },
        })
    }
    fn encode(self) -> Option<u64> {
        (self.epoch != 0 && self.epoch <= MAX_EPOCH)
            .then_some((self.epoch << 1) | u64::from(self.target == FocusTarget::Window))
    }
}

pub use crate::capture::CaptureMode;

/// One capture intent. `seq` counts this seat's requests from 1 and is never
/// reused, so a consumer can tell a new press from one it already took.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureRequest {
    pub seq: u64,
    pub mode: CaptureMode,
}

#[repr(C, align(64))]
struct FocusCell {
    magic: u64,
    route: AtomicU64,
    /// Nonzero means production is stopped until the consumer emits this
    /// epoch's Reset and acknowledges it. Newer snapshots cannot overtake it.
    reset_epoch: AtomicU64,
    dropped: AtomicU64,
    /// The newest capture request, `seq << 8 | mode` (`0` = none yet).
    /// Only input-stream writes it.
    capture: AtomicU64,
    /// The newest request sequence the compositor took. Only the
    /// compositor writes it. Claimed from the reserved tail, which a
    /// founder zeroes, so an older founder's cell reads as no request.
    capture_taken: AtomicU64,
    reserved: [u8; 16],
}

pub struct FocusReader<'a> {
    cell: &'a FocusCell,
}
pub struct FocusWriter<'a> {
    cell: &'a FocusCell,
}
impl FocusReader<'_> {
    pub fn read(&self) -> Option<Focus> {
        Focus::decode(self.cell.route.load(Ordering::Acquire))
    }
}
impl FocusWriter<'_> {
    pub fn read(&self) -> Option<Focus> {
        Focus::decode(self.cell.route.load(Ordering::Acquire))
    }
    /// The compositor is the only writer. Even Window -> Window increments
    /// the epoch so queued reports cannot acquire a different window's focus.
    /// Exhaustion refuses; an old epoch is never reused.
    pub fn advance(&mut self, target: FocusTarget) -> Option<Focus> {
        let current = self.read()?;
        let next = Focus {
            epoch: current.epoch.checked_add(1)?,
            target,
        };
        self.cell.route.store(next.encode()?, Ordering::Release);
        Some(next)
    }
}

#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct SnapshotWire {
    epoch: U64,
    report: InputReport,
}

/// Raw complete keyboard state, or an explicit loss/reset boundary. A Reset
/// is tagged with the route that lost input, never whatever window is current
/// when the consumer eventually wakes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelayEvent {
    Snapshot { epoch: u64, report: InputReport },
    Reset { epoch: u64 },
}

pub struct RelayProducer<'a> {
    ring: spsc::Producer<'a, SnapshotWire>,
    cell: &'a FocusCell,
}
pub struct RelayConsumer<'a> {
    ring: spsc::Consumer<'a, SnapshotWire>,
    cell: &'a FocusCell,
}

impl RelayProducer<'_> {
    /// Only validated keyboard snapshots on a window route are accepted.
    /// Loss is explicit, bounded and nonblocking. While a Reset is pending,
    /// no snapshot is published, even if the consumer has freed ring slots.
    pub fn push(&mut self, focus: Focus, report: InputReport) -> bool {
        if focus.target != FocusTarget::Window || focus.encode().is_none() || !keyboard(report) {
            return false;
        }
        if self.cell.reset_epoch.load(Ordering::Acquire) == 0
            && self
                .ring
                .push(SnapshotWire {
                    epoch: focus.epoch.into(),
                    report,
                })
                .is_ok()
        {
            return true;
        }
        self.cell.dropped.fetch_add(1, Ordering::Relaxed);
        // A later route's loss supersedes an earlier route's pending reset.
        // Focus loss already resets the earlier controller; the consumer's
        // compare-exchange never clears a different, newer route's reset.
        self.cell.reset_epoch.store(focus.epoch, Ordering::Release);
        false
    }

    /// Level state, not a one-shot flag: a failed/dropped advisory kick cannot
    /// clear queued data, an unacknowledged Reset or an untaken capture.
    pub fn needs_wake(&self) -> bool {
        self.cell.reset_epoch.load(Ordering::Acquire) != 0
            || self.ring.capacity_left() != self.ring.capacity()
            || capture_untaken(self.cell)
    }

    /// Publish a capture request for this seat. The newest request replaces
    /// one the compositor has not taken yet: presses closer together than a
    /// compositor pass would capture the same frame. Exhaustion refuses; a
    /// sequence is never reused.
    pub fn request_capture(&mut self, mode: CaptureMode) -> Option<CaptureRequest> {
        // This side is the only writer of the word.
        let seq = (self.cell.capture.load(Ordering::Relaxed) >> 8)
            .checked_add(1)
            .filter(|&seq| seq <= MAX_CAPTURE_SEQ)?;
        self.cell
            .capture
            .store((seq << 8) | mode as u64, Ordering::Release);
        Some(CaptureRequest { seq, mode })
    }
    pub fn dropped(&self) -> u64 {
        self.cell.dropped.load(Ordering::Relaxed)
    }
}

impl RelayConsumer<'_> {
    /// Overflow discards the bounded stale prefix before acknowledging Reset.
    /// The producer is stopped by reset_epoch throughout this drain. Once the
    /// compare-exchange clears it, newer snapshots may queue behind the Reset
    /// returned by this call. The caller must process it before polling again.
    pub fn poll(&mut self) -> Option<RelayEvent> {
        let epoch = self.cell.reset_epoch.load(Ordering::Acquire);
        if epoch != 0 {
            for _ in 0..self.ring.outstanding().min(RELAY_CAP) {
                let _ = self.ring.pop();
            }
            let _ = self.cell.reset_epoch.compare_exchange(
                epoch,
                0,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            return Some(RelayEvent::Reset { epoch });
        }
        // An invalid epoch has no controller to reset. Skip such records
        // within the bounded published prefix; do not let one malformed
        // record hide a valid snapshot already behind it in this batch.
        for _ in 0..self.ring.outstanding().min(RELAY_CAP) {
            let record = self.ring.pop()?;
            let epoch = record.epoch.get();
            if epoch == 0 || epoch > MAX_EPOCH {
                continue;
            }
            if !keyboard(record.report) {
                return Some(RelayEvent::Reset { epoch });
            }
            return Some(RelayEvent::Snapshot {
                epoch,
                report: record.report,
            });
        }
        None
    }
    pub fn pending(&self) -> bool {
        self.cell.reset_epoch.load(Ordering::Acquire) != 0
            || self.ring.outstanding() != 0
            || capture_untaken(self.cell)
    }
    pub fn dropped(&self) -> u64 {
        self.cell.dropped.load(Ordering::Relaxed)
    }

    /// Take the seat's newest capture request, once. Taking acknowledges it,
    /// which settles the producer's wake. A malformed mode is acknowledged
    /// too and captures nothing, so it cannot hold that wake high.
    pub fn take_capture(&mut self) -> Option<CaptureRequest> {
        let word = self.cell.capture.load(Ordering::Acquire);
        let seq = word >> 8;
        // This side is the only writer of `capture_taken`.
        if seq == self.cell.capture_taken.load(Ordering::Relaxed) {
            return None;
        }
        self.cell.capture_taken.store(seq, Ordering::Release);
        Some(CaptureRequest {
            seq,
            mode: CaptureMode::from_code(word as u8)?,
        })
    }
}

/// A request the compositor has not taken: level state for both peers.
fn capture_untaken(cell: &FocusCell) -> bool {
    cell.capture.load(Ordering::Acquire) >> 8 != cell.capture_taken.load(Ordering::Acquire)
}

/// A relayable keyboard snapshot: a keyboard record whose reserved bytes
/// carry nothing but a valid lock state, and which is not a rollover report.
pub fn keyboard(report: InputReport) -> bool {
    report.keyboard_locks().is_some() && !report.data.contains(&1)
}

/// The returned byte regions are disjoint and exact; neither ring's mutable
/// borrow includes the other ring or the atomic mailbox.
pub struct Regions<'a> {
    pub echo: &'a mut [u8],
    pub focus: &'a mut [u8],
    pub relay: &'a mut [u8],
}
pub fn regions(window: &mut [u8]) -> Result<Regions<'_>, RingError> {
    if window.as_ptr().align_offset(64) != 0 {
        return Err(RingError::Misaligned);
    }
    if window.len() < WINDOW_BYTES {
        return Err(RingError::TooSmall);
    }
    let (echo, tail) = window.split_at_mut(ECHO_BYTES);
    let (focus, tail) = tail.split_at_mut(FOCUS_BYTES);
    Ok(Regions {
        echo,
        focus,
        relay: &mut tail[..RELAY_BYTES],
    })
}

/// Exclusive founder initialization, before either peer receives its view.
pub fn init(window: &mut [u8]) -> Result<(), RingError> {
    let parts = regions(window)?;
    spsc::init::<OutChunk>(parts.echo, ECHO_CAP)?;
    spsc::init::<SnapshotWire>(parts.relay, RELAY_CAP)?;
    // SAFETY: regions proved a unique, complete cache-line-aligned mailbox.
    unsafe {
        parts
            .focus
            .as_mut_ptr()
            .cast::<FocusCell>()
            .write(FocusCell {
                magic: MAGIC,
                route: AtomicU64::new(Focus::INITIAL.encode().unwrap()),
                reset_epoch: AtomicU64::new(0),
                dropped: AtomicU64::new(0),
                capture: AtomicU64::new(0),
                capture_taken: AtomicU64::new(0),
                reserved: [0; 16],
            });
    }
    Ok(())
}

fn focus_cell(window: &mut [u8]) -> Result<&FocusCell, RingError> {
    // Callers pass only regions' exact validated mailbox. Header fields are
    // immutable after founder init; all shared mutation uses atomic fields.
    let cell = unsafe { &*window.as_ptr().cast::<FocusCell>() };
    if cell.magic != MAGIC
        || cell.reserved != [0; 16]
        || Focus::decode(cell.route.load(Ordering::Acquire)).is_none()
    {
        return Err(RingError::BadHeader);
    }
    Ok(cell)
}

pub struct InputSide<'a> {
    pub echo: spsc::Producer<'a, OutChunk>,
    pub focus: FocusReader<'a>,
    pub reports: RelayProducer<'a>,
}
pub struct CompositorSide<'a> {
    pub echo: spsc::Consumer<'a, OutChunk>,
    pub focus: FocusWriter<'a>,
    pub reports: RelayConsumer<'a>,
}

/// Attach the sole input-stream side over its receiver-mapped lease view.
pub fn input_side(window: &mut [u8]) -> Result<InputSide<'_>, RingError> {
    let parts = regions(window)?;
    let cell = focus_cell(parts.focus)?;
    let echo = spsc::producer(parts.echo)?;
    let ring = spsc::producer(parts.relay)?;
    if echo.capacity() != ECHO_CAP || ring.capacity() != RELAY_CAP {
        return Err(RingError::BadHeader);
    }
    Ok(InputSide {
        echo,
        focus: FocusReader { cell },
        reports: RelayProducer { ring, cell },
    })
}
/// Attach the sole compositor side over its receiver-mapped sibling view.
pub fn compositor_side(window: &mut [u8]) -> Result<CompositorSide<'_>, RingError> {
    let parts = regions(window)?;
    let cell = focus_cell(parts.focus)?;
    let echo = spsc::consumer(parts.echo)?;
    let ring = spsc::consumer(parts.relay)?;
    if echo.capacity() != ECHO_CAP || ring.capacity() != RELAY_CAP {
        return Err(RingError::BadHeader);
    }
    Ok(CompositorSide {
        echo,
        focus: FocusWriter { cell },
        reports: RelayConsumer { ring, cell },
    })
}

/// Same-address-space form: initialize once and split both rings through the
/// ring's own safe SPSC constructor. Only shared references to the interior-
/// mutable atomic mailbox are duplicated; no mutable byte slices overlap.
pub fn split(window: &mut [u8]) -> Result<(InputSide<'_>, CompositorSide<'_>), RingError> {
    init(window)?;
    let parts = regions(window)?;
    let cell = focus_cell(parts.focus)?;
    let (echo_in, echo_out) = spsc::split(parts.echo, ECHO_CAP)?;
    let (raw_in, raw_out) = spsc::split(parts.relay, RELAY_CAP)?;
    Ok((
        InputSide {
            echo: echo_in,
            focus: FocusReader { cell },
            reports: RelayProducer { ring: raw_in, cell },
        },
        CompositorSide {
            echo: echo_out,
            focus: FocusWriter { cell },
            reports: RelayConsumer {
                ring: raw_out,
                cell,
            },
        },
    ))
}

const _: () = {
    assert!(core::mem::size_of::<FocusCell>() == FOCUS_BYTES);
    assert!(core::mem::size_of::<SnapshotWire>() == 24);
    assert!(WINDOW_BYTES == 3008 && WINDOW_BYTES <= 4096);
};

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    #[repr(align(64))]
    struct Page([u8; 4096]);

    #[test]
    fn page_preserves_echo_and_routes_atomic_epochs() {
        let mut page = Page([0; 4096]);
        let (mut input, mut compositor) = split(&mut page.0).unwrap();
        assert_eq!(input.focus.read(), Some(Focus::INITIAL));
        let mut echo = OutChunk {
            len: 1,
            flags: 0,
            reserved: [0; 6],
            bytes: [0; 24],
        };
        echo.bytes[0] = b'x';
        input.echo.push(echo).unwrap();
        assert_eq!(compositor.echo.pop(), Some(echo));
        let first = compositor.focus.advance(FocusTarget::Window).unwrap();
        let second = compositor.focus.advance(FocusTarget::Window).unwrap();
        assert_eq!(second.epoch, first.epoch + 1);
        assert_eq!(input.focus.read(), Some(second));
        assert!(
            input
                .reports
                .push(second, InputReport::keyboard(1, 0, [4, 0, 0, 0, 0, 0]))
        );
        assert!(input.reports.needs_wake());
        assert!(
            matches!(compositor.reports.poll(), Some(RelayEvent::Snapshot { epoch, .. }) if epoch == second.epoch)
        );
        assert!(!input.reports.needs_wake());
    }

    #[test]
    fn overflow_reset_is_durable_epoch_scoped_and_precedes_new_state() {
        let mut page = Page([0; 4096]);
        let (mut input, mut compositor) = split(&mut page.0).unwrap();
        let focus = compositor.focus.advance(FocusTarget::Window).unwrap();
        let held = InputReport::keyboard(1, 0, [4, 0, 0, 0, 0, 0]);
        for _ in 0..RELAY_CAP {
            assert!(input.reports.push(focus, held));
        }
        assert!(!input.reports.push(focus, held));
        assert!(input.reports.needs_wake());
        let next = compositor.focus.advance(FocusTarget::Window).unwrap();
        assert_eq!(
            compositor.reports.poll(),
            Some(RelayEvent::Reset { epoch: focus.epoch })
        );
        assert_eq!(compositor.reports.poll(), None);
        assert_eq!(input.reports.dropped(), 1);
        assert!(!input.reports.needs_wake());
        assert!(
            input
                .reports
                .push(next, InputReport::keyboard(1, 0, [0; 6]))
        );
        assert!(
            matches!(compositor.reports.poll(), Some(RelayEvent::Snapshot { epoch, report }) if epoch == next.epoch && report.data == [0; 6])
        );
    }

    #[test]
    fn layout_and_focus_exhaustion_fail_closed() {
        let mut page = Page([0; 4096]);
        assert!(matches!(input_side(&mut page.0), Err(RingError::BadHeader)));
        assert!(matches!(
            regions(&mut page.0[..WINDOW_BYTES - 1]),
            Err(RingError::TooSmall)
        ));
        assert!(matches!(
            regions(&mut page.0[1..]),
            Err(RingError::Misaligned)
        ));
        let (input, mut compositor) = split(&mut page.0).unwrap();
        compositor
            .focus
            .cell
            .route
            .store(u64::MAX, Ordering::Release);
        assert_eq!(compositor.focus.advance(FocusTarget::Terminal), None);
        assert_eq!(input.focus.read().unwrap().epoch, MAX_EPOCH);
    }

    #[test]
    fn attachment_checks_exact_layout_and_leaves_unused_tail_alone() {
        let mut page = Page([0xa5; 4096]);
        init(&mut page.0).unwrap();
        assert!(page.0[WINDOW_BYTES..].iter().all(|&byte| byte == 0xa5));
        // The old prefix still attaches as precisely the existing echo ring.
        assert_eq!(
            spsc::producer::<OutChunk>(&mut page.0).unwrap().capacity(),
            ECHO_CAP
        );
        assert!(input_side(&mut page.0).is_ok());
        assert!(compositor_side(&mut page.0).is_ok());

        let parts = regions(&mut page.0).unwrap();
        spsc::init::<SnapshotWire>(parts.relay, RELAY_CAP / 2).unwrap();
        assert!(matches!(input_side(&mut page.0), Err(RingError::BadHeader)));
        assert!(matches!(
            compositor_side(&mut page.0),
            Err(RingError::BadHeader)
        ));
        init(&mut page.0).unwrap();
        let parts = regions(&mut page.0).unwrap();
        spsc::init::<OutChunk>(parts.echo, ECHO_CAP / 2).unwrap();
        assert!(matches!(input_side(&mut page.0), Err(RingError::BadHeader)));
        assert!(matches!(
            compositor_side(&mut page.0),
            Err(RingError::BadHeader)
        ));
    }

    #[test]
    fn terminal_routes_and_malformed_reports_never_enter_raw_or_echo_queues() {
        let mut page = Page([0; 4096]);
        let (mut input, mut compositor) = split(&mut page.0).unwrap();
        let report = InputReport::keyboard(1, 0, [4, 0, 0, 0, 0, 0]);
        assert!(!input.reports.push(Focus::INITIAL, report));
        let focus = compositor.focus.advance(FocusTarget::Window).unwrap();
        let mut malformed = report;
        malformed.kind = 0;
        assert!(!input.reports.push(focus, malformed));
        // The lock byte is the only meaningful reserved byte, and only its
        // three lock bits: anything else is not a keyboard snapshot.
        for (index, value) in [(0, 0x08), (1, 1), (6, 1)] {
            malformed = report;
            malformed.reserved[index] = value;
            assert!(!input.reports.push(focus, malformed));
        }
        malformed = report;
        malformed.data[0] = 1;
        assert!(!input.reports.push(focus, malformed));
        assert!(!input.reports.push(Focus { epoch: 0, ..focus }, report));
        assert_eq!(input.reports.dropped(), 0);
        assert!(!input.reports.needs_wake());
        assert!(!compositor.reports.pending());
        assert_eq!(compositor.reports.poll(), None);
        assert_eq!(compositor.echo.pop(), None);
    }

    /// The seat's lock state rides the snapshot to the window unchanged — a
    /// client resolves Caps Lock exactly as the terminal discipline does.
    #[test]
    fn a_locked_snapshot_reaches_the_window_with_its_locks() {
        use kore_abi::keyboard::Locks;
        let mut page = Page([0; 4096]);
        let (mut input, mut compositor) = split(&mut page.0).unwrap();
        let focus = compositor.focus.advance(FocusTarget::Window).unwrap();
        let report = InputReport::keyboard(1, 0, [4, 0, 0, 0, 0, 0])
            .with_locks(Locks::CAPS.toggled(Locks::SCROLL));
        assert!(input.reports.push(focus, report));
        let Some(RelayEvent::Snapshot { report: got, .. }) = compositor.reports.poll() else {
            panic!("the locked snapshot must be relayed");
        };
        assert_eq!(got.keyboard_locks(), Some(Locks::CAPS.toggled(Locks::SCROLL)));
    }

    #[test]
    fn invalid_wire_epochs_do_not_mask_valid_queued_input() {
        let mut page = Page([0; 4096]);
        let (mut input, mut compositor) = split(&mut page.0).unwrap();
        let focus = compositor.focus.advance(FocusTarget::Window).unwrap();
        let report = InputReport::keyboard(1, 0, [4, 0, 0, 0, 0, 0]);
        for epoch in [0, MAX_EPOCH + 1] {
            input
                .reports
                .ring
                .push(SnapshotWire {
                    epoch: epoch.into(),
                    report,
                })
                .unwrap();
        }
        assert!(input.reports.push(focus, report));
        assert_eq!(
            compositor.reports.poll(),
            Some(RelayEvent::Snapshot {
                epoch: focus.epoch,
                report
            })
        );
        let mut malformed = report;
        malformed.reserved[1] = 1;
        input
            .reports
            .ring
            .push(SnapshotWire {
                epoch: focus.epoch.into(),
                report: malformed,
            })
            .unwrap();
        assert_eq!(
            compositor.reports.poll(),
            Some(RelayEvent::Reset { epoch: focus.epoch })
        );
        assert_eq!(compositor.reports.poll(), None);
    }

    #[test]
    fn two_leases_keep_routes_sources_and_overflow_independent() {
        let mut first_page = Page([0; 4096]);
        let mut second_page = Page([0; 4096]);
        let (mut first_input, mut first_compositor) = split(&mut first_page.0).unwrap();
        let (mut second_input, mut second_compositor) = split(&mut second_page.0).unwrap();
        let first_focus = first_compositor.focus.advance(FocusTarget::Window).unwrap();
        assert_eq!(second_input.focus.read(), Some(Focus::INITIAL));
        let second_focus = second_compositor
            .focus
            .advance(FocusTarget::Window)
            .unwrap();
        let first = InputReport::keyboard(1, 0, [4, 0, 0, 0, 0, 0]);
        let second = InputReport::keyboard(2, 2, [5, 0, 0, 0, 0, 0]);
        for _ in 0..RELAY_CAP {
            assert!(first_input.reports.push(first_focus, first));
        }
        assert!(!first_input.reports.push(first_focus, first));
        assert!(second_input.reports.push(second_focus, second));
        assert_eq!(
            second_compositor.reports.poll(),
            Some(RelayEvent::Snapshot {
                epoch: second_focus.epoch,
                report: second
            })
        );
        assert_eq!(second_input.reports.dropped(), 0);
        assert_eq!(
            first_compositor.reports.poll(),
            Some(RelayEvent::Reset {
                epoch: first_focus.epoch
            })
        );
        assert_eq!(first_compositor.echo.pop(), None);
        assert_eq!(second_compositor.echo.pop(), None);
    }

    #[test]
    fn newer_loss_replaces_old_reset_without_leaking_old_snapshots() {
        let mut page = Page([0; 4096]);
        let (mut input, mut compositor) = split(&mut page.0).unwrap();
        let focus = compositor.focus.advance(FocusTarget::Window).unwrap();
        let held = InputReport::keyboard(1, 0, [4, 0, 0, 0, 0, 0]);
        for _ in 0..RELAY_CAP {
            assert!(input.reports.push(focus, held));
        }
        assert!(!input.reports.push(focus, held));
        let next = compositor.focus.advance(FocusTarget::Window).unwrap();
        assert!(!input.reports.push(next, held));
        assert_eq!(
            compositor.reports.poll(),
            Some(RelayEvent::Reset { epoch: next.epoch })
        );
        assert_eq!(compositor.reports.poll(), None);
        assert_eq!(input.reports.dropped(), 2);
    }

    #[test]
    fn a_capture_request_is_taken_once_and_settles_the_wake() {
        let mut page = Page([0; 4096]);
        let (mut input, mut compositor) = split(&mut page.0).unwrap();
        // A founder's zeroed words read as no request.
        assert_eq!(compositor.reports.take_capture(), None);
        assert!(!input.reports.needs_wake());
        assert!(!compositor.reports.pending());

        let request = input.reports.request_capture(CaptureMode::Screen).unwrap();
        assert_eq!(
            request,
            CaptureRequest {
                seq: 1,
                mode: CaptureMode::Screen
            }
        );
        assert!(
            input.reports.needs_wake(),
            "an untaken capture holds the wake"
        );
        assert!(compositor.reports.pending());
        // The request is not a snapshot: the ring stays empty.
        assert_eq!(compositor.reports.poll(), None);
        assert_eq!(compositor.reports.take_capture(), Some(request));
        assert_eq!(compositor.reports.take_capture(), None, "taken once");
        assert!(!input.reports.needs_wake());
        assert!(!compositor.reports.pending());

        let next = input.reports.request_capture(CaptureMode::Region).unwrap();
        assert_eq!(next.seq, 2, "a sequence is never reused");
        assert_eq!(compositor.reports.take_capture(), Some(next));
    }

    #[test]
    fn a_newer_capture_replaces_an_untaken_one() {
        let mut page = Page([0; 4096]);
        let (mut input, mut compositor) = split(&mut page.0).unwrap();
        input.reports.request_capture(CaptureMode::Screen).unwrap();
        let newest = input.reports.request_capture(CaptureMode::Window).unwrap();
        assert_eq!(compositor.reports.take_capture(), Some(newest));
        assert_eq!(compositor.reports.take_capture(), None);
        assert!(!input.reports.needs_wake());
    }

    /// A capture rides beside the snapshot ring, not in it: a pending Reset
    /// neither blocks nor drops it, and draining the Reset does not take it.
    #[test]
    fn a_capture_is_independent_of_snapshot_loss() {
        let mut page = Page([0; 4096]);
        let (mut input, mut compositor) = split(&mut page.0).unwrap();
        let focus = compositor.focus.advance(FocusTarget::Window).unwrap();
        let held = InputReport::keyboard(1, 0, [4, 0, 0, 0, 0, 0]);
        for _ in 0..=RELAY_CAP {
            let _ = input.reports.push(focus, held);
        }
        let request = input.reports.request_capture(CaptureMode::Window).unwrap();
        assert_eq!(
            compositor.reports.poll(),
            Some(RelayEvent::Reset { epoch: focus.epoch })
        );
        assert_eq!(compositor.reports.poll(), None);
        assert!(input.reports.needs_wake(), "the capture is still owed");
        assert_eq!(compositor.reports.take_capture(), Some(request));
        assert!(!input.reports.needs_wake());
    }

    #[test]
    fn capture_exhaustion_refuses_and_a_malformed_mode_settles() {
        let mut page = Page([0; 4096]);
        let (mut input, mut compositor) = split(&mut page.0).unwrap();
        let last = (MAX_CAPTURE_SEQ << 8) | CaptureMode::Screen as u64;
        input.reports.cell.capture.store(last, Ordering::Release);
        assert_eq!(input.reports.request_capture(CaptureMode::Window), None);
        assert_eq!(input.reports.cell.capture.load(Ordering::Acquire), last);
        assert_eq!(
            compositor.reports.take_capture(),
            Some(CaptureRequest {
                seq: MAX_CAPTURE_SEQ,
                mode: CaptureMode::Screen
            })
        );

        // A mode no chord produces captures nothing, and is still taken.
        for mode in [0u8, 4, 0xff] {
            let seq = 7 + u64::from(mode);
            input
                .reports
                .cell
                .capture
                .store((seq << 8) | u64::from(mode), Ordering::Release);
            assert!(input.reports.needs_wake());
            assert_eq!(compositor.reports.take_capture(), None);
            assert!(!input.reports.needs_wake(), "mode {mode:#x} settled");
        }
    }

    /// The reserved tail still fails closed after the capture claim.
    #[test]
    fn a_dirty_reserved_tail_refuses_attachment() {
        let mut page = Page([0; 4096]);
        init(&mut page.0).unwrap();
        let tail = ECHO_BYTES + FOCUS_BYTES - 1;
        page.0[tail] = 1;
        assert!(matches!(input_side(&mut page.0), Err(RingError::BadHeader)));
        assert!(matches!(
            compositor_side(&mut page.0),
            Err(RingError::BadHeader)
        ));
        page.0[tail] = 0;
        assert!(input_side(&mut page.0).is_ok());
    }

    #[test]
    fn reset_acknowledgment_allows_next_snapshot_across_threads() {
        // Named stack backing and scoped ownership exercise the same SPSC
        // halves as peers without leaked windows or a test global allocator.
        let mut page = Page([0; 4096]);
        let (mut input, mut compositor) = split(&mut page.0).unwrap();
        let focus = compositor.focus.advance(FocusTarget::Window).unwrap();
        let held = InputReport::keyboard(1, 0, [4, 0, 0, 0, 0, 0]);
        let release = InputReport::keyboard(1, 0, [0; 6]);
        let published = std::sync::Barrier::new(2);
        let acknowledged = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                for _ in 0..RELAY_CAP {
                    assert!(input.reports.push(focus, held));
                }
                assert!(!input.reports.push(focus, release));
                published.wait();
                acknowledged.wait();
                assert!(input.reports.push(focus, release));
            });
            published.wait();
            assert_eq!(
                compositor.reports.poll(),
                Some(RelayEvent::Reset { epoch: focus.epoch })
            );
            assert!(!compositor.reports.pending());
            acknowledged.wait();
        });
        assert_eq!(
            compositor.reports.poll(),
            Some(RelayEvent::Snapshot {
                epoch: focus.epoch,
                report: release
            })
        );
        assert!(!compositor.reports.pending());
    }
}
