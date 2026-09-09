//! Clock + timer ABI — ambient time reads, scheduler waits, asynchronous
//! alarms, and privileged clock control: the userland echo of the kernel's
//! `platform::chrono` vocabulary.
//!
//! # The scoping axis: observation is ambient; hardware control is authority
//!
//! Time splits into three structural paths:
//!
//! - **Reading is universal** — no capability. Monotonic uptime (time since
//!   boot) and wall-clock ("real"/calendar) time now are read from a
//!   kernel-published, read-only [`ClockPage`] (the vDSO model): a task maps
//!   the page and projects the current time with **zero syscall and zero
//!   authority**, exactly as reading a clock should be. There is no cap slot,
//!   no session, no ring entry on this path.
//! - **Waiting is scheduler state** — a task-local sleep folds its deadline
//!   into the task's existing park operation. Long-lived or periodic alarms
//!   are multiplexed by the Time service over its ordinary capability-wired
//!   client pair. Neither path grants userland a physical comparator.
//! - **Physical timer and wall-clock control remain capability-gated** — the
//!   low-level [`Timeout`] vocabulary names an affine
//!   [`Timer`](crate::cap::ResourceKind::Timer) as a fixed-file slot, and
//!   disciplining the calendar anchor uses [`SetWallClock`]. Those are control
//!   surfaces, not prerequisites for reading time or asking the scheduler /
//!   Time service to wake a task.
//!
//! This mirrors the kernel, where reading (`WallClock::now`, `ClockSource`) is
//! a public read but the clock *write* (`set_boot_unix`) is closed to
//! bootstrap and ongoing disciplining is a minted capability
//! (`src/platform/chrono/wallclock.rs`).
//!
//! # Monotonic and wall-clock are distinct types
//!
//! Deadlines are **monotonic** ([`Uptime`]); "what time is it" is
//! **wall-clock** ([`UnixTime`]). They are separate types with no coercion, so
//! a calendar timestamp can never be armed as a monotonic deadline (the common
//! clock-API bug) — the compiler rejects it. This echoes the kernel's
//! `Instant` (monotonic, source-pinned) vs. `UnixTime` (calendar) split.
//!
//! # Vocabulary — echoing the kernel, over `core::time::Duration`
//!
//! [`TimerMode`], [`TimerProgram`], [`TimerError`], and [`Frequency`] are the
//! ABI echoes of the identically-named `platform::chrono` types, so the two
//! sides speak one language. All time spans are `core::time::Duration`; the
//! two sides never invent a parallel duration type.

use core::fmt;
use core::ops::{Add, Sub};
use core::time::Duration;

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::session::SlotRef;
use crate::uring::{IoSqeFlags, IoTimeoutFlags, IoUringSqe, op};

/// Saturating `Duration -> u64` nanoseconds — the wire operand is a `u64`
/// nanosecond count (≈ 584 years of range), so any realistic span is exact and
/// only pathological values clamp. Kept private; the public API is
/// `Duration`-typed.
#[inline]
const fn nanos_u64(d: Duration) -> u64 {
    let n = d.as_nanos();
    if n > u64::MAX as u128 {
        u64::MAX
    } else {
        n as u64
    }
}

/// A tick rate — the ABI **hertz projection** of the kernel's richer
/// `platform::chrono::Frequency` (which carries an exact `count / period`
/// pair). The wire only needs the rate, so this is a plain `u64` Hz; the
/// kernel's exact type stays the single source of truth for rate *arithmetic*.
/// Reading the time base's rate (via [`ClockPage::frequency`]) is universal.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct Frequency(u64);

impl Frequency {
    /// A rate of `hz` events per second.
    #[inline]
    pub const fn from_hz(hz: u64) -> Self {
        Self(hz)
    }

    /// The rate in whole hertz.
    #[inline]
    pub const fn as_hz(self) -> u64 {
        self.0
    }

    /// Whether the rate is unknown (`0 Hz` — an uncalibrated time base).
    #[inline]
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// The period of one event, or [`Duration::MAX`] while uncalibrated
    /// (`0 Hz`).
    #[inline]
    pub const fn period(self) -> Duration {
        match 1_000_000_000u64.checked_div(self.0) {
            Some(nanos) => Duration::from_nanos(nanos),
            None => Duration::MAX,
        }
    }
}

/// A **monotonic** point in time — elapsed time since boot, non-decreasing and
/// meaningful only in differences. The userland echo of the kernel's
/// `platform::chrono::Instant`; deadlines for [`Timeout::until`] are `Uptime`,
/// never [`UnixTime`], so a wall-clock stamp cannot be armed as a monotonic
/// deadline. Stored as a `Duration` since the monotonic epoch, so all
/// arithmetic is plain `Duration` math.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Uptime(Duration);

impl Uptime {
    /// The monotonic epoch itself (≈ boot).
    pub const ZERO: Self = Self(Duration::ZERO);

    /// Wrap a raw nanosecond count since the monotonic epoch.
    #[inline]
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(Duration::from_nanos(nanos))
    }

    /// Wrap an already-known elapsed-since-boot duration.
    #[inline]
    pub const fn from_duration(since_boot: Duration) -> Self {
        Self(since_boot)
    }

    /// Nanoseconds since the monotonic epoch (saturating at [`u64::MAX`]).
    #[inline]
    pub const fn as_nanos(self) -> u64 {
        nanos_u64(self.0)
    }

    /// The elapsed duration since the monotonic epoch.
    #[inline]
    pub const fn as_duration(self) -> Duration {
        self.0
    }

    /// Time from `earlier` to `self`, saturating to [`Duration::ZERO`] when
    /// `earlier` is in fact the later instant (mirrors modern
    /// `Instant::duration_since`).
    #[inline]
    pub fn saturating_duration_since(self, earlier: Self) -> Duration {
        self.0.saturating_sub(earlier.0)
    }

    /// Time from `earlier` to `self`, or `None` if `earlier` is later.
    #[inline]
    pub fn checked_duration_since(self, earlier: Self) -> Option<Duration> {
        self.0.checked_sub(earlier.0)
    }
}

/// `Uptime + Duration` — a later monotonic deadline. Saturates (rather than the
/// kernel `Instant`'s panic) so ABI arithmetic never traps across the boundary.
impl Add<Duration> for Uptime {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Duration) -> Self {
        Self(self.0.saturating_add(rhs))
    }
}

/// `Uptime - Duration` — an earlier monotonic deadline (saturating at boot).
impl Sub<Duration> for Uptime {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Duration) -> Self {
        Self(self.0.saturating_sub(rhs))
    }
}

/// `Uptime - Uptime` yields the [`Duration`] between them (saturating).
impl Sub for Uptime {
    type Output = Duration;
    #[inline]
    fn sub(self, earlier: Self) -> Duration {
        self.0.saturating_sub(earlier.0)
    }
}

impl fmt::Debug for Uptime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Uptime").field(&self.0).finish()
    }
}

/// A **wall-clock** (calendar) instant, as elapsed time since the UNIX epoch
/// (1970-01-01 00:00:00 UTC). The userland echo of the kernel's
/// `platform::chrono::UnixTime`, down to the ISO-8601 UTC [`Display`]. Distinct
/// from [`Uptime`]: this is "what time is it", not a monotonic deadline.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct UnixTime(Duration);

impl UnixTime {
    /// Wrap whole seconds since the UNIX epoch.
    #[inline]
    pub const fn from_secs(secs: u64) -> Self {
        Self(Duration::from_secs(secs))
    }

    /// Wrap nanoseconds since the UNIX epoch.
    #[inline]
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(Duration::from_nanos(nanos))
    }

    /// Whole seconds since the UNIX epoch.
    #[inline]
    pub const fn as_secs(self) -> u64 {
        self.0.as_secs()
    }

    /// Nanoseconds since the UNIX epoch (saturating at [`u64::MAX`]).
    #[inline]
    pub const fn as_nanos(self) -> u64 {
        nanos_u64(self.0)
    }

    /// The elapsed duration since the epoch.
    #[inline]
    pub const fn since_epoch(self) -> Duration {
        self.0
    }
}

impl fmt::Display for UnixTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let secs = self.0.as_secs() as i64;
        let (days, tod) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
        let (y, m, d) = civil_from_days(days);
        let (hh, mm, ss) = (tod / 3600, (tod % 3600) / 60, tod % 60);
        write!(f, "{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
    }
}

impl fmt::Debug for UnixTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Gregorian `(year, month, day)` from days since the UNIX epoch — Howard
/// Hinnant's public-domain `civil_from_days`, exact over the whole range. Kept
/// byte-identical to the kernel's copy so both render a `UnixTime` the same.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Failure modes of a fallible clock/timer operation — the ABI echo of the
/// kernel's `platform::chrono::TimerError`, so a completion status maps back
/// 1:1. The boundary-relevant variants are [`Uncalibrated`](Self::Uncalibrated)
/// (time base or wall anchor not yet known — a wrong time is never invented),
/// [`OutOfRange`](Self::OutOfRange) (a delay/interval the hardware cannot
/// represent), and [`Unsupported`](Self::Unsupported); the rest surface a
/// kernel-side arming refusal verbatim.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TimerError {
    /// The timer does not support the requested mode/domain.
    Unsupported,
    /// The timer has not been configured/bound for arming.
    Unconfigured,
    /// The time base (or wall-clock anchor) is not yet calibrated/set.
    Uncalibrated,
    /// Arming was attempted from the wrong CPU for a CPU-local timer.
    WrongCpu,
    /// The backing timer controller is unavailable.
    ControllerUnavailable,
    /// The requested delay or interval cannot be represented by the timer's
    /// counter/comparator width.
    OutOfRange,
}

/// Result of a fallible clock/timer operation.
pub type TimerResult<T> = Result<T, TimerError>;

/// One-shot vs. periodic — the ABI echo of `platform::chrono::TimerMode`.
/// `#[repr(u8)]` pins a stable wire discriminant.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TimerMode {
    /// Fire once, then disarm.
    OneShot = 0,
    /// Re-arm and fire every interval until cancelled (the ring multishot).
    Periodic = 1,
}

/// A timer program: a [`TimerMode`] plus an interval `Duration` — the ABI echo
/// of `platform::chrono::TimerProgram`, and the shape the kernel arms a
/// [`TimerBinding`](../../src/platform/chrono/binding.rs) with. For a relative
/// arm the interval is the delay (one-shot) or the period (periodic).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TimerProgram {
    mode: TimerMode,
    interval: Duration,
}

impl TimerProgram {
    /// A program with an explicit mode and interval.
    #[inline]
    pub const fn new(mode: TimerMode, interval: Duration) -> Self {
        Self { mode, interval }
    }

    /// A one-shot that fires once after `delay`.
    #[inline]
    pub const fn one_shot(delay: Duration) -> Self {
        Self::new(TimerMode::OneShot, delay)
    }

    /// A periodic that fires every `interval`.
    #[inline]
    pub const fn periodic(interval: Duration) -> Self {
        Self::new(TimerMode::Periodic, interval)
    }

    /// The mode.
    #[inline]
    pub const fn mode(self) -> TimerMode {
        self.mode
    }

    /// The interval (delay for one-shot, period for periodic).
    #[inline]
    pub const fn interval(self) -> Duration {
        self.interval
    }
}

/// ABI version of the [`ClockPage`] layout. Bumped only on a layout change;
/// the reserved words keep growth additive.
pub const CLOCK_PAGE_ABI: u16 = 1;

/// The architecture counter named by [`ClockPage::tick_hz`] is readable at
/// user privilege.  When this bit is clear the calibration remains useful to
/// a privileged time service, but the SDK must refuse its zero-trap read
/// rather than execute an instruction the kernel configured to trap.
pub const CLOCK_FLAG_USER_COUNTER: u16 = 1 << 0;

/// Every flag understood by this ABI revision.  Unknown bits fail closed so a
/// newer producer cannot silently be interpreted with older clock semantics.
pub const CLOCK_FLAGS_KNOWN: u16 = CLOCK_FLAG_USER_COUNTER;

/// The kernel-published, **read-only** clock page — kore's vDSO-style time
/// source. A founder publishes an immutable calibrated copy into every task;
/// a task reads the time from it with **no syscall and no capability**, which
/// is what makes *reading* time ambient. A future disciplined shared page uses
/// this same layout and its seqlock fields.
///
/// Time is the standard `CLOCK_REALTIME` construction the kernel wall clock
/// uses (`now = boot_anchor + monotonic_uptime`): userland samples the raw
/// monotonic counter (an unprivileged user-mode read — `RDTSC`, `CNTVCT_EL0`,
/// the RISC-V `time` CSR; done by the SDK, since it needs an arch intrinsic
/// this `#![forbid(unsafe_code)]` crate cannot express) and hands the value to
/// [`uptime`](Self::uptime) / [`wall`](Self::wall), which apply the page's
/// calibrated rate and anchor. This crate owns the **layout and the
/// projection math**; the SDK owns the live [`seq`](Self::seq)-guarded read.
///
/// **Seqlock discipline.** The kernel publishes updates (recalibration, wall
/// discipline) under a seqlock: it bumps [`seq`](Self::seq) to odd, writes the
/// fields, then bumps it to even. A reader samples `seq` (acquire), reads the
/// fields, and re-samples `seq`; if it changed or was odd, it retries. The
/// snapshot this struct models is one iteration of that loop.
///
/// Native-endian, naturally-aligned, read [zero-copy](zerocopy) like the
/// io_uring rings (a shared memory region on kore's LE targets), not the
/// arbitrary-offset arena descriptors.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, FromBytes, IntoBytes, KnownLayout, Immutable)]
pub struct ClockPage {
    /// Seqlock sequence — **odd** while the kernel is mid-update. See the
    /// struct doc's read protocol.
    pub seq: u32,
    /// Layout version ([`CLOCK_PAGE_ABI`]).
    pub abi: u16,
    /// [`CLOCK_FLAG_USER_COUNTER`] and future ABI flags. Unknown bits make
    /// the page unusable rather than changing its meaning silently.
    pub flags: u16,
    /// Calibrated monotonic-counter rate in **hertz**; `0` = uncalibrated
    /// (reads refuse [`TimerError::Uncalibrated`]).
    pub tick_hz: u64,
    /// The raw counter value at the monotonic epoch (≈ boot) — the subtrahend
    /// that turns a live counter sample into elapsed uptime.
    pub epoch_ticks: u64,
    /// Wall-clock anchor: UNIX **nanoseconds** at the monotonic epoch, or `0`
    /// before the anchor is set ([`wall`](Self::wall) refuses until then — a
    /// wrong calendar time is never invented). Ongoing discipline (NTP/PTP)
    /// rebases this via [`SetWallClock`].
    pub boot_unix_nanos: u64,
    /// Precomputed fixed-point ticks-to-nanoseconds reciprocal. Projection is
    /// `(elapsed_ticks * mult) >> shift`, avoiding division in every read.
    pub mult: u64,
    /// Fractional bits in [`mult`](Self::mult). V1 uses 32; the explicit field
    /// lets a later producer select a different precision without ABI drift.
    pub shift: u32,
    /// Natural-layout padding; must be zero.
    pub padding: u32,
    /// Reserved; must be zero. Frozen additive headroom (skew/drift discipline
    /// fields land here without a layout break).
    pub reserved: [u64; 2],
}

impl ClockPage {
    /// Build one calibrated clock block.  The block is immutable after a
    /// founder publishes it into a task's read-only boot page; `seq == 0` is
    /// therefore a stable seqlock snapshot.  A future disciplined shared page
    /// uses the same layout and advances `seq` around updates.
    #[inline]
    pub const fn new(
        tick_hz: u64,
        epoch_ticks: u64,
        boot_unix_nanos: u64,
        user_counter: bool,
    ) -> Self {
        Self {
            seq: 0,
            abi: CLOCK_PAGE_ABI,
            flags: if user_counter {
                CLOCK_FLAG_USER_COUNTER
            } else {
                0
            },
            tick_hz,
            epoch_ticks,
            boot_unix_nanos,
            mult: clock_mult(tick_hz),
            shift: CLOCK_SHIFT,
            padding: 0,
            reserved: [0; 2],
        }
    }

    /// Validate the frozen block before trusting its calibration.
    #[inline]
    pub const fn validate(&self) -> bool {
        self.abi == CLOCK_PAGE_ABI
            && self.flags & !CLOCK_FLAGS_KNOWN == 0
            && self.padding == 0
            && self.reserved[0] == 0
            && self.reserved[1] == 0
            && self.shift < 128
            && ((self.tick_hz == 0 && self.mult == 0) || (self.tick_hz != 0 && self.mult != 0))
    }

    /// The page's layout version.
    #[inline]
    pub const fn abi(&self) -> u16 {
        self.abi
    }

    /// Whether the monotonic time base is calibrated (a nonzero rate).
    #[inline]
    pub const fn is_calibrated(&self) -> bool {
        self.tick_hz != 0
    }

    /// Whether the SDK may sample the raw counter without a privilege trap.
    #[inline]
    pub const fn user_counter_readable(&self) -> bool {
        self.flags & CLOCK_FLAG_USER_COUNTER != 0
    }

    /// Whether the wall-clock anchor has been set.
    #[inline]
    pub const fn has_wall_anchor(&self) -> bool {
        self.boot_unix_nanos != 0
    }

    /// The seqlock sequence — the SDK's tear-free read loops on this (see the
    /// struct doc).
    #[inline]
    pub const fn seq(&self) -> u32 {
        self.seq
    }

    /// The calibrated time-base rate.
    #[inline]
    pub const fn frequency(&self) -> Frequency {
        Frequency::from_hz(self.tick_hz)
    }

    /// Project a raw monotonic-counter sample into **monotonic uptime**. Pure
    /// (`(now_ticks - epoch_ticks) / tick_hz`); refuses
    /// [`TimerError::Uncalibrated`] before the rate is known rather than
    /// returning a bogus zero.
    #[inline]
    pub fn uptime(&self, now_ticks: u64) -> TimerResult<Uptime> {
        if self.tick_hz == 0 || self.mult == 0 || self.shift >= 128 {
            return Err(TimerError::Uncalibrated);
        }
        let elapsed = now_ticks.wrapping_sub(self.epoch_ticks) as u128;
        let nanos = elapsed.saturating_mul(self.mult as u128) >> self.shift;
        let nanos = if nanos > u64::MAX as u128 {
            u64::MAX
        } else {
            nanos as u64
        };
        Ok(Uptime::from_nanos(nanos))
    }

    /// Project a raw monotonic-counter sample into **wall-clock time now** =
    /// anchor + uptime (`CLOCK_REALTIME`). Refuses [`TimerError::Uncalibrated`]
    /// before the anchor is set or the base is calibrated — mirroring the
    /// kernel's `WallClock::now`.
    #[inline]
    pub fn wall(&self, now_ticks: u64) -> TimerResult<UnixTime> {
        if self.boot_unix_nanos == 0 {
            return Err(TimerError::Uncalibrated);
        }
        let uptime = self.uptime(now_ticks)?;
        Ok(UnixTime::from_nanos(
            self.boot_unix_nanos.saturating_add(uptime.as_nanos()),
        ))
    }
}

const CLOCK_SHIFT: u32 = 32;

#[inline]
const fn clock_mult(hz: u64) -> u64 {
    if hz == 0 {
        0
    } else {
        ((1_000_000_000u128 << CLOCK_SHIFT) / hz as u128) as u64
    }
}

/// **When** a [`Timeout`] fires — the type-safe union of a relative delay and
/// the two absolute-deadline domains. This is where monotonic and wall-clock
/// deadlines stay distinct: an absolute monotonic deadline is an [`Uptime`], an
/// absolute calendar deadline is a [`UnixTime`], and the two cannot be
/// interchanged.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Deadline {
    /// Fire `Duration` after the kernel arms the submission (relative,
    /// monotonic) — the ordinary sleep/timeout.
    After(Duration),
    /// Fire at an absolute **monotonic** deadline.
    AtMonotonic(Uptime),
    /// Fire at an absolute **wall-clock** deadline (honors clock discipline).
    AtWall(UnixTime),
}

/// A timeout/timer submission — the io_uring-idiom, [`Timer`]-cap-gated arm
/// path. It maps to an `IORING_OP_TIMEOUT` (or `LINK_TIMEOUT`) SQE via
/// [`encode`](Self::encode); a standalone timeout is "sleep until the
/// deadline", and a [linked](Self::as_link) one is a timeout attached to the
/// preceding op in an `IO_LINK` chain.
///
/// **The capability gate.** [`timer`](Self::timer) is the arena slot of the
/// affine [`Timer`](crate::cap::ResourceKind::Timer) capability (with
/// [`MODIFY`](crate::cap::Rights::MODIFY) rights — arming is a *write*). The
/// SQE names it as a **fixed-file** handle
/// ([`IoSqeFlags::FIXED_FILE`](crate::uring::IoSqeFlags::FIXED_FILE)); the ring
/// carries the *slot index*, never the capability record, so the kernel
/// validates the cap in that slot before arming and the "no cap on the ring"
/// invariant holds. Without a valid `Timer` cap in the slot the arm is refused
/// — which is exactly why sleeping/arming is not ambient.
///
/// **The operand.** Per `res/SYSCALL.md` I5 the deadline rides **inline** in
/// the SQE (no host pointer): [`Deadline`] becomes an unsigned-nanosecond
/// `addr` plus [`IoTimeoutFlags`]; [`count`](Self::count) completions is the
/// alternative wake condition (`0` = pure time, the plain sleep).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Timeout {
    /// Arena slot of the `Timer` capability that authorizes the arm.
    pub timer: SlotRef,
    /// When it fires.
    pub deadline: Deadline,
    /// One-shot or periodic (periodic ⇒ the ring multishot; only meaningful
    /// with [`Deadline::After`]).
    pub mode: TimerMode,
    /// Fire early if this many completions land first (`0` = time only).
    pub count: u32,
    /// Encode as a linked timeout (`LINK_TIMEOUT`) attached to the preceding
    /// `IO_LINK` op, rather than a standalone `TIMEOUT`.
    pub link: bool,
}

impl Timeout {
    /// Sleep for `delay` (relative one-shot) — the `sleep(Duration)`
    /// convenience. The submission form: [`encode`](Self::encode) it onto the
    /// ring; the actual block is the ring `enter` (`submit_and_wait`, riding the
    /// #71 park/wake seam), since this crate carries no syscall of its own.
    #[inline]
    pub const fn sleep(timer: SlotRef, delay: Duration) -> Self {
        Self {
            timer,
            deadline: Deadline::After(delay),
            mode: TimerMode::OneShot,
            count: 0,
            link: false,
        }
    }

    /// Fire at an absolute **monotonic** [`Uptime`] deadline (one-shot). Takes
    /// an `Uptime`, not a `UnixTime`, so a calendar stamp cannot be armed here.
    #[inline]
    pub const fn until(timer: SlotRef, deadline: Uptime) -> Self {
        Self {
            timer,
            deadline: Deadline::AtMonotonic(deadline),
            mode: TimerMode::OneShot,
            count: 0,
            link: false,
        }
    }

    /// Fire at an absolute **wall-clock** [`UnixTime`] deadline (one-shot);
    /// honors clock discipline.
    #[inline]
    pub const fn until_wall(timer: SlotRef, deadline: UnixTime) -> Self {
        Self {
            timer,
            deadline: Deadline::AtWall(deadline),
            mode: TimerMode::OneShot,
            count: 0,
            link: false,
        }
    }

    /// Arm a periodic timer firing every `interval` — a multishot that posts a
    /// completion (with [`IoCqeFlags::MORE`](crate::uring::IoCqeFlags::MORE)
    /// set) each period until [cancelled](Self::remove).
    #[inline]
    pub const fn periodic(timer: SlotRef, interval: Duration) -> Self {
        Self {
            timer,
            deadline: Deadline::After(interval),
            mode: TimerMode::Periodic,
            count: 0,
            link: false,
        }
    }

    /// Arm from a mirrored [`TimerProgram`] (mode + interval) — the relative
    /// form that speaks the kernel's arm vocabulary directly.
    #[inline]
    pub const fn program(timer: SlotRef, program: TimerProgram) -> Self {
        Self {
            timer,
            deadline: Deadline::After(program.interval()),
            mode: program.mode(),
            count: 0,
            link: false,
        }
    }

    /// Set the completion-count wake threshold (fire when this many linked
    /// completions land, even before the deadline). `0` (default) = time only.
    #[inline]
    #[must_use]
    pub const fn with_count(mut self, count: u32) -> Self {
        self.count = count;
        self
    }

    /// Encode as a **linked** timeout (`LINK_TIMEOUT`) bounding the preceding
    /// `IO_LINK` op instead of a standalone sleep.
    #[inline]
    #[must_use]
    pub const fn as_link(mut self) -> Self {
        self.link = true;
        self
    }

    /// The [`IoTimeoutFlags`] this timeout's deadline + mode encode to.
    #[inline]
    fn timeout_flags(&self) -> IoTimeoutFlags {
        let mut f = IoTimeoutFlags::empty();
        match self.deadline {
            Deadline::After(_) => {}
            Deadline::AtMonotonic(_) => f |= IoTimeoutFlags::ABS,
            Deadline::AtWall(_) => f |= IoTimeoutFlags::ABS | IoTimeoutFlags::REALTIME,
        }
        if matches!(self.mode, TimerMode::Periodic) {
            f |= IoTimeoutFlags::MULTISHOT;
        }
        f
    }

    /// The inline nanosecond operand this timeout's deadline encodes to (I5 —
    /// no host pointer).
    #[inline]
    fn operand_nanos(&self) -> u64 {
        match self.deadline {
            Deadline::After(d) => nanos_u64(d),
            Deadline::AtMonotonic(u) => u.as_nanos(),
            Deadline::AtWall(w) => w.as_nanos(),
        }
    }

    /// Encode into an io_uring SQE — `TIMEOUT` (or `LINK_TIMEOUT` if
    /// [`link`](Self::link)), the `Timer` cap named as the fixed-file `fd`, the
    /// deadline inline in `addr` as nanoseconds, `count` in `off`, and the
    /// [`IoTimeoutFlags`] in `op_flags`. Attach a correlation token by setting
    /// [`user_data`](IoUringSqe::user_data) on the returned SQE.
    pub fn encode(&self) -> IoUringSqe {
        let opcode = if self.link {
            op::LINK_TIMEOUT
        } else {
            op::TIMEOUT
        };
        IoUringSqe {
            opcode,
            flags: IoSqeFlags::FIXED_FILE.bits(),
            ioprio: 0,
            fd: self.timer.index() as i32,
            off: self.count as u64,
            addr: self.operand_nanos(),
            len: 1,
            op_flags: self.timeout_flags().bits(),
            user_data: 0,
            buf_index: 0,
            personality: 0,
            splice_fd_in: 0,
            addr3: 0,
            addr3_pad: 0,
        }
    }

    /// Decode a `TIMEOUT` / `LINK_TIMEOUT` SQE back into a [`Timeout`] — the
    /// inverse of [`encode`](Self::encode), used by the kernel ring engine (and
    /// the round-trip tests) to recover the deadline, mode, and `Timer` slot.
    /// Refuses [`TimerError::Unsupported`] for any other opcode.
    pub fn decode(sqe: &IoUringSqe) -> TimerResult<Self> {
        let link = match sqe.opcode {
            op::TIMEOUT => false,
            op::LINK_TIMEOUT => true,
            _ => return Err(TimerError::Unsupported),
        };
        let flags = IoTimeoutFlags::from_bits_truncate(sqe.op_flags);
        let deadline = if flags.contains(IoTimeoutFlags::REALTIME) {
            Deadline::AtWall(UnixTime::from_nanos(sqe.addr))
        } else if flags.contains(IoTimeoutFlags::ABS) {
            Deadline::AtMonotonic(Uptime::from_nanos(sqe.addr))
        } else {
            Deadline::After(Duration::from_nanos(sqe.addr))
        };
        let mode = if flags.contains(IoTimeoutFlags::MULTISHOT) {
            TimerMode::Periodic
        } else {
            TimerMode::OneShot
        };
        Ok(Self {
            timer: SlotRef::from_wire(sqe.fd as u32),
            deadline,
            mode,
            count: sqe.off as u32,
            link,
        })
    }

    /// Build a `TIMEOUT_REMOVE` SQE that cancels the in-flight timeout whose
    /// completion token is `user_data` (Linux carries it in `addr`). No `Timer`
    /// slot: cancellation references the armed op by token, not the resource.
    pub fn remove(user_data: u64) -> IoUringSqe {
        IoUringSqe {
            opcode: op::TIMEOUT_REMOVE,
            flags: 0,
            ioprio: 0,
            fd: -1,
            off: 0,
            addr: user_data,
            len: 0,
            op_flags: 0,
            user_data: 0,
            buf_index: 0,
            personality: 0,
            splice_fd_in: 0,
            addr3: 0,
            addr3_pad: 0,
        }
    }
}

/// Argument payload for **disciplining the wall clock** — rebasing the calendar
/// anchor (NTP/PTP), the one clock *write* that is not an io_uring op.
///
/// This rides the **authority plane** (a [`crate::session`] invocation over the
/// [`Timer`](crate::cap::ResourceKind::Timer) capability), not the ring: it is
/// the userland face of the kernel's disciplining capability — the kernel
/// closes `set_boot_unix` to bootstrap and mints a cap for ongoing rebase
/// (`src/platform/chrono/wallclock.rs`), which the trusted time driver consumes
/// on each update. It carries no ambient authority: without the `Timer` cap the
/// invocation is refused, so a task can *read* wall time freely but never move
/// it.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SetWallClock {
    /// New anchor: UNIX **nanoseconds** at the current monotonic instant. The
    /// kernel republishes it into [`ClockPage::boot_unix_nanos`].
    pub anchor_unix_nanos: u64,
}

// Frozen wire pins (#81; the crate-header wire-freeze discipline) — the clock page userland reads
// zero-copy and the discipline payload; a drift is an ABI break.
const _: () = {
    use core::mem::{align_of, offset_of, size_of};
    assert!(size_of::<ClockPage>() == 64);
    assert!(align_of::<ClockPage>() == 8);
    assert!(offset_of!(ClockPage, seq) == 0);
    assert!(offset_of!(ClockPage, abi) == 4);
    assert!(offset_of!(ClockPage, tick_hz) == 8);
    assert!(offset_of!(ClockPage, epoch_ticks) == 16);
    assert!(offset_of!(ClockPage, boot_unix_nanos) == 24);
    assert!(size_of::<SetWallClock>() == 8);
};

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal `fmt::Write` sink so the ISO-8601 render can be checked
    /// without `alloc` (kore-abi is `no_std`, no allocator).
    struct Sink {
        buf: [u8; 32],
        len: usize,
    }
    impl Sink {
        fn new() -> Self {
            Self {
                buf: [0; 32],
                len: 0,
            }
        }
        fn as_str(&self) -> &str {
            core::str::from_utf8(&self.buf[..self.len]).unwrap()
        }
    }
    impl fmt::Write for Sink {
        fn write_str(&mut self, s: &str) -> fmt::Result {
            let b = s.as_bytes();
            self.buf[self.len..self.len + b.len()].copy_from_slice(b);
            self.len += b.len();
            Ok(())
        }
    }

    fn iso(secs: u64) -> Sink {
        use core::fmt::Write;
        let mut s = Sink::new();
        write!(s, "{}", UnixTime::from_secs(secs)).unwrap();
        s
    }

    /// `UnixTime` renders ISO-8601 UTC identically to the kernel `UnixTime`
    /// (shared vectors, including a leap day) — the two speak one calendar.
    #[test]
    fn unix_time_renders_iso8601_utc() {
        assert_eq!(iso(0).as_str(), "1970-01-01T00:00:00Z");
        assert_eq!(iso(86_400).as_str(), "1970-01-02T00:00:00Z");
        assert_eq!(iso(1_700_000_000).as_str(), "2023-11-14T22:13:20Z");
        assert_eq!(iso(1_582_934_400).as_str(), "2020-02-29T00:00:00Z");
    }

    /// The clock page projects a live counter sample into uptime and wall time
    /// via `now = anchor + uptime`, and refuses before calibration.
    #[test]
    fn clock_page_projects_uptime_and_wall() {
        // Uncalibrated: both reads refuse rather than invent a time.
        let blank = ClockPage::default();
        assert_eq!(blank.uptime(1_000), Err(TimerError::Uncalibrated));
        assert_eq!(blank.wall(1_000), Err(TimerError::Uncalibrated));

        // 1 GHz base, epoch at tick 1_000, anchored at 2023-11-14T22:13:20Z.
        let page = ClockPage::new(1_000_000_000, 1_000, 1_700_000_000 * 1_000_000_000, true);
        // 2 s of ticks past the epoch (1 GHz → 2e9 ticks).
        let now_ticks = 1_000 + 2_000_000_000;
        assert_eq!(
            page.uptime(now_ticks).unwrap().as_duration(),
            Duration::from_secs(2)
        );
        assert_eq!(page.wall(now_ticks).unwrap().as_secs(), 1_700_000_002);
        assert_eq!(page.frequency().as_hz(), 1_000_000_000);

        // Calibrated base but no anchor: uptime works, wall still refuses.
        let no_anchor = ClockPage::new(1_000_000_000, 0, 0, true);
        assert!(no_anchor.uptime(5).is_ok());
        assert_eq!(no_anchor.wall(5), Err(TimerError::Uncalibrated));
    }

    /// A relative sleep round-trips through the wire and names the `Timer` cap
    /// as a fixed-file handle — the capability gate.
    #[test]
    fn sleep_round_trips_and_gates_on_the_timer_cap() {
        let t = Timeout::sleep(SlotRef::from_wire(7), Duration::from_millis(250));
        let sqe = t.encode();

        assert_eq!(sqe.opcode, op::TIMEOUT);
        assert_eq!(
            sqe.flags,
            IoSqeFlags::FIXED_FILE.bits(),
            "Timer cap is a fixed-file handle"
        );
        assert_eq!(sqe.fd, 7, "fd names the Timer capability's arena slot");
        assert_eq!(sqe.addr, 250_000_000, "relative delay is inline nanos (I5)");
        assert_eq!(
            sqe.op_flags, 0,
            "relative monotonic one-shot: no timeout flags"
        );
        assert_eq!(sqe.off, 0, "pure time: no completion threshold");

        assert_eq!(Timeout::decode(&sqe).unwrap(), t);
    }

    /// An absolute monotonic deadline sets `ABS` and decodes back to an
    /// `Uptime` (never a `UnixTime`).
    #[test]
    fn absolute_monotonic_deadline_sets_abs() {
        let t = Timeout::until(SlotRef::from_wire(3), Uptime::from_nanos(9_000_000_000));
        let sqe = t.encode();
        assert_eq!(sqe.op_flags, IoTimeoutFlags::ABS.bits());
        assert_eq!(sqe.addr, 9_000_000_000);
        match Timeout::decode(&sqe).unwrap().deadline {
            Deadline::AtMonotonic(u) => assert_eq!(u.as_nanos(), 9_000_000_000),
            other => panic!("expected monotonic deadline, got {other:?}"),
        }
    }

    /// An absolute wall-clock deadline sets `ABS | REALTIME` and decodes back to
    /// a `UnixTime`.
    #[test]
    fn absolute_wall_deadline_sets_realtime() {
        let t = Timeout::until_wall(SlotRef::from_wire(4), UnixTime::from_secs(1_700_000_000));
        let sqe = t.encode();
        assert_eq!(
            sqe.op_flags,
            (IoTimeoutFlags::ABS | IoTimeoutFlags::REALTIME).bits()
        );
        assert_eq!(Timeout::decode(&sqe).unwrap(), t);
        assert!(matches!(t.deadline, Deadline::AtWall(_)));
    }

    /// A periodic arm sets `MULTISHOT` and decodes back to `Periodic`.
    #[test]
    fn periodic_arm_is_multishot() {
        let t = Timeout::periodic(SlotRef::from_wire(2), Duration::from_millis(10));
        let sqe = t.encode();
        assert_eq!(sqe.op_flags, IoTimeoutFlags::MULTISHOT.bits());
        assert_eq!(sqe.addr, 10_000_000);
        let back = Timeout::decode(&sqe).unwrap();
        assert_eq!(back.mode, TimerMode::Periodic);
        assert_eq!(back, t);
    }

    /// `program(...)` speaks the mirrored `TimerProgram` vocabulary, and
    /// `as_link()` / `with_count()` reshape the submission.
    #[test]
    fn program_link_and_count() {
        let t = Timeout::program(
            SlotRef::from_wire(1),
            TimerProgram::one_shot(Duration::from_secs(1)),
        )
        .with_count(4)
        .as_link();
        let sqe = t.encode();
        assert_eq!(sqe.opcode, op::LINK_TIMEOUT, "as_link => LINK_TIMEOUT");
        assert_eq!(sqe.off, 4, "with_count => completion threshold in off");
        assert_eq!(sqe.addr, 1_000_000_000);
        assert_eq!(Timeout::decode(&sqe).unwrap(), t);
    }

    /// `remove` builds a `TIMEOUT_REMOVE` that references the armed op by its
    /// completion token, carrying no `Timer` slot.
    #[test]
    fn remove_references_by_user_data() {
        let sqe = Timeout::remove(0xdead_beef);
        assert_eq!(sqe.opcode, op::TIMEOUT_REMOVE);
        assert_eq!(sqe.addr, 0xdead_beef);
        assert_eq!(sqe.fd, -1);
    }

    /// Decode refuses a non-timeout opcode rather than misreading it.
    #[test]
    fn decode_rejects_foreign_opcode() {
        let mut sqe = Timeout::sleep(SlotRef::from_wire(0), Duration::ZERO).encode();
        sqe.opcode = op::READ;
        assert_eq!(Timeout::decode(&sqe), Err(TimerError::Unsupported));
    }

    /// Monotonic and wall-clock deadlines are distinct types: `until` takes an
    /// `Uptime`, `until_wall` takes a `UnixTime`, and the arithmetic that builds
    /// a monotonic deadline stays on `Uptime`.
    #[test]
    fn monotonic_and_wall_are_type_distinct() {
        let deadline = Uptime::from_nanos(1_000) + Duration::from_secs(1);
        assert_eq!(deadline.as_nanos(), 1_000_001_000);
        let _ = Timeout::until(SlotRef::from_wire(0), deadline); // Uptime — compiles.
        // Timeout::until(SlotRef::from_wire(0), UnixTime::from_secs(1)) would NOT compile.
    }
}
