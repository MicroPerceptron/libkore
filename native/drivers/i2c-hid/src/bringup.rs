//! The bring-up ladder, as a **sans-IO state machine that demands its own
//! delays**.
//!
//! HID-over-I2C bring-up includes a 60 ms compatibility settle after each
//! power-on and up to a second for a reset acknowledgement. A
//! driver that performed those waits itself would be a driver that spins,
//! and no amount of care further up could undo that — which is why
//! [`Step::Delay`] is a *demand handed to the caller* rather than a sleep
//! taken here. The service folds it into its deadline; a test satisfies it
//! for free.
//!
//! Nothing in this module touches a bus. It emits steps and consumes
//! results, so every ordering rule below — and every race a real device
//! produces — is testable without hardware.

use crate::command::{Command, PowerState, register_address};
use crate::descriptor::{DescriptorError, HidDescriptor};
use crate::framing::{Frame, decode};

/// Compatibility settling time after every successful `SET_POWER(ON)`.
///
/// This is a host compatibility policy, not a mandatory protocol delay.
/// It matches Linux's [`i2c_hid_set_power`] policy, including the second
/// power-on after reset. The caller schedules it without blocking siblings.
///
/// [`i2c_hid_set_power`]: https://github.com/torvalds/linux/blob/v6.12/drivers/hid/i2c-hid/i2c-hid-core.c#L386
pub const POWER_SETTLE_NS: u64 = 60_000_000;
/// How long to wait between polls for the reset acknowledgement.
pub const RESET_POLL_NS: u64 = 1_000_000;
/// The total reset-acknowledgement budget, in **wall clock**.
///
/// The caller reports the time spent on demanded delays, bus transactions,
/// and servicing other children while this reset is outstanding. Counting
/// only nominal [`RESET_POLL_NS`] delays would omit transfer pacing and
/// sibling work, allowing a nominal second to last much longer. Controller
/// completion interrupts and HID GPIO attention are separate paths; absent
/// GPIO routing does not imply an absent controller-completion route.
///
/// Reported time is therefore the budget, and the poll count
/// ([`MAX_RESET_POLLS`]) is only its second ceiling: whichever is reached
/// first ends the wait. Neither is loosened, so a device that is present
/// but slow still gets the full second the protocol allows it.
pub const RESET_TIMEOUT_NS: u64 = 1_000_000_000;
/// The reset-acknowledgement poll ceiling — [`RESET_TIMEOUT_NS`] at the
/// nominal [`RESET_POLL_NS`] cadence, i.e. exactly the bound this ladder
/// enforced before it learned to read a clock.
///
/// Kept beside the wall-clock budget rather than replaced by it, because a
/// caller whose clock has coarse granularity can report zero elapsed for a
/// fast poll — and a budget that only ever sees zero is no budget at all.
/// Two ceilings, either of which ends the wait, is the fail-closed shape.
pub const MAX_RESET_POLLS: u32 = (RESET_TIMEOUT_NS / RESET_POLL_NS) as u32;
/// How long a `NO_IRQ` device is simply slept over instead of polled.
pub const NO_IRQ_RESET_NS: u64 = 100_000_000;

/// The largest scratch this state machine needs for one write.
pub const OUT_LEN: usize = crate::command::MAX_COMMAND_LEN;

/// Device quirks that change the ladder rather than its parameters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Quirks {
    /// The device does not signal its reset acknowledgement, so the
    /// acknowledgement must be waited out rather than polled for.
    ///
    /// Polling such a device reads zero-length frames forever and the
    /// bring-up times out on a device that is in fact ready.
    pub no_irq: bool,
    /// Skip the second `SET_POWER(ON)` after reset for a device that refuses it.
    ///
    /// The default ladder sends it for compatibility. This flag omits both
    /// that command and its associated settling delay.
    pub no_power_after_reset: bool,
}

/// What the caller must do next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    /// Write the first `len` bytes of [`BringUp::out`].
    Write {
        /// Bytes to write.
        len: usize,
    },
    /// Write the first `write_len` bytes of [`BringUp::out`], then read
    /// `read_len` bytes back in the same transaction.
    WriteRead {
        /// Bytes to write.
        write_len: usize,
        /// Bytes to read after the repeated START.
        read_len: usize,
    },
    /// Read `len` bytes with no preceding write.
    Read {
        /// Bytes to read.
        len: usize,
    },
    /// Wait `ns` nanoseconds. **Not a spin** — the caller folds this into
    /// whatever deadline discipline it already has.
    Delay {
        /// Nanoseconds to wait.
        ns: u64,
    },
    /// Bring-up is complete.
    Done,
}

/// Why bring-up failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BringUpError {
    /// The HID descriptor was malformed.
    Descriptor(DescriptorError),
    /// The device never acknowledged its reset within the budget.
    ResetTimeout {
        /// How many polls were made.
        polls: u32,
        /// Wall-clock nanoseconds the caller reported spending on them.
        ///
        /// Carried because the two numbers diagnose different machines:
        /// `polls` at [`MAX_RESET_POLLS`] with `spent` far below
        /// [`RESET_TIMEOUT_NS`] is a device answering nothing on a healthy
        /// fast bus, while a small `polls` with `spent` at the budget is a
        /// bus whose every transaction is timing out underneath the
        /// ladder — the shape a missing interrupt route takes.
        spent_ns: u64,
    },
    /// The report descriptor read back short.
    ReportDescriptorShort {
        /// What the HID descriptor promised.
        want: usize,
        /// What arrived.
        got: usize,
    },
    /// A result was delivered for a step that was not outstanding.
    OutOfOrder,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    FetchDescriptor,
    PowerOn,
    PowerSettle,
    Reset,
    AwaitAck,
    AckPoll,
    PowerAgain,
    PowerAgainSettle,
    FetchReportDescriptor,
    Done,
}

/// Stable diagnostic phase for the protocol ladder. This deliberately
/// coalesces internal wait states: callers need to know which prerequisite
/// failed, not how the state machine represents one bounded delay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Descriptor,
    Power,
    Reset,
    ResetAck,
    ReportDescriptor,
    Done,
}

/// The bring-up ladder.
pub struct BringUp {
    state: State,
    quirks: Quirks,
    /// The descriptor register, from firmware (`_DSM`) — the one address
    /// this protocol does not publish about itself.
    descriptor_register: u16,
    descriptor: Option<HidDescriptor>,
    out: [u8; OUT_LEN],
    /// Reset acknowledgement polls made so far.
    polls: u32,
    /// Wall-clock nanoseconds spent since the reset was written, as the
    /// caller reports them — the reset budget's real denominator (see
    /// [`RESET_TIMEOUT_NS`]).
    reset_spent_ns: u64,
    /// Whether a reset is outstanding, which is what makes a zero-length
    /// frame an *acknowledgement* rather than an empty poll. The bytes
    /// are identical; only this flag tells them apart.
    reset_pending: bool,
    /// How many bytes the last emitted read asked for.
    expect: usize,
}

impl BringUp {
    /// Begin bring-up against a device whose descriptor register firmware
    /// reported.
    pub fn new(descriptor_register: u16, quirks: Quirks) -> Self {
        Self {
            state: State::FetchDescriptor,
            quirks,
            descriptor_register,
            descriptor: None,
            out: [0; OUT_LEN],
            polls: 0,
            reset_spent_ns: 0,
            reset_pending: false,
            expect: 0,
        }
    }

    /// The write buffer the current [`Step`] refers to.
    pub fn out(&self) -> &[u8] {
        &self.out
    }

    /// The validated HID descriptor, once it has been read.
    pub fn descriptor(&self) -> Option<&HidDescriptor> {
        self.descriptor.as_ref()
    }

    /// How many reset-acknowledgement polls were made.
    pub fn polls(&self) -> u32 {
        self.polls
    }

    /// Wall-clock nanoseconds the caller has reported spending since the
    /// reset was written — what [`RESET_TIMEOUT_NS`] is actually measured
    /// against.
    pub fn reset_spent_ns(&self) -> u64 {
        self.reset_spent_ns
    }

    /// Charge one step's wall clock to the outstanding reset, if there is
    /// one.
    ///
    /// Every step between the reset write and its acknowledgement counts,
    /// not only the demanded delays: the poll's bus transaction is where
    /// the time actually goes on a machine whose completion line is dead,
    /// and a budget that ignored it would be the same fiction with an
    /// extra addition in it.
    fn charge(&mut self, elapsed_ns: u64) {
        if self.reset_pending {
            self.reset_spent_ns = self.reset_spent_ns.saturating_add(elapsed_ns);
        }
    }

    /// Current stable diagnostic phase.
    pub const fn phase(&self) -> Phase {
        match self.state {
            State::FetchDescriptor => Phase::Descriptor,
            State::PowerOn | State::PowerSettle | State::PowerAgain | State::PowerAgainSettle => {
                Phase::Power
            }
            State::Reset => Phase::Reset,
            State::AwaitAck | State::AckPoll => Phase::ResetAck,
            State::FetchReportDescriptor => Phase::ReportDescriptor,
            State::Done => Phase::Done,
        }
    }

    /// What to do next.
    pub fn step(&mut self) -> Step {
        match self.state {
            State::FetchDescriptor => {
                self.out[..2].copy_from_slice(&register_address(self.descriptor_register));
                self.expect = 30;
                Step::WriteRead {
                    write_len: 2,
                    read_len: 30,
                }
            }
            State::PowerOn | State::PowerAgain => {
                let reg = self.command_register();
                let c = Command::set_power(reg, PowerState::On);
                let len = c.bytes().len();
                self.out[..len].copy_from_slice(c.bytes());
                Step::Write { len }
            }
            State::PowerSettle | State::PowerAgainSettle => Step::Delay {
                ns: POWER_SETTLE_NS,
            },
            State::Reset => {
                let reg = self.command_register();
                let c = Command::reset(reg);
                let len = c.bytes().len();
                self.out[..len].copy_from_slice(c.bytes());
                Step::Write { len }
            }
            State::AwaitAck => {
                if self.quirks.no_irq {
                    // A device that cannot signal its acknowledgement is
                    // waited out once, not polled: polling it reads
                    // zero-length frames forever and times out a device
                    // that is in fact ready.
                    Step::Delay {
                        ns: NO_IRQ_RESET_NS,
                    }
                } else {
                    Step::Delay { ns: RESET_POLL_NS }
                }
            }
            State::AckPoll => {
                self.expect = 2;
                Step::Read { len: 2 }
            }
            State::FetchReportDescriptor => {
                let (reg, len) = match self.descriptor.as_ref() {
                    Some(d) => (
                        d.w_report_desc_register.get(),
                        d.w_report_desc_length.get() as usize,
                    ),
                    None => (0, 0),
                };
                self.out[..2].copy_from_slice(&register_address(reg));
                self.expect = len;
                Step::WriteRead {
                    write_len: 2,
                    read_len: len,
                }
            }
            State::Done => Step::Done,
        }
    }

    /// Report that a [`Step::Write`] completed.
    pub fn wrote(&mut self) -> Result<(), BringUpError> {
        self.state = match self.state {
            State::PowerOn => State::PowerSettle,
            State::Reset => {
                self.reset_pending = true;
                // The reset budget's clock starts here, not at `new`: what
                // it bounds is the acknowledgement, and the descriptor
                // fetch and power-on that precede it have budgets of their
                // own (the caller's transfer pacing).
                self.reset_spent_ns = 0;
                State::AwaitAck
            }
            State::PowerAgain => State::PowerAgainSettle,
            _ => return Err(BringUpError::OutOfOrder),
        };
        Ok(())
    }

    /// Report that a [`Step::Delay`] elapsed, and the wall-clock
    /// nanoseconds it actually took.
    ///
    /// The demanded figure is in [`Step::Delay`] already; what the ladder
    /// cannot know — and what [`RESET_TIMEOUT_NS`] is denominated in — is
    /// what the caller's clock says the wait really cost. A caller with no
    /// clock at all reports `0` and falls back to the poll ceiling
    /// ([`MAX_RESET_POLLS`]), which is exactly the bound this ladder had
    /// before.
    pub fn delayed(&mut self, elapsed_ns: u64) -> Result<(), BringUpError> {
        self.charge(elapsed_ns);
        self.state = match self.state {
            State::PowerSettle => State::Reset,
            State::PowerAgainSettle => State::FetchReportDescriptor,
            State::AwaitAck => {
                if self.quirks.no_irq {
                    // The wait *is* the acknowledgement for this device.
                    self.reset_pending = false;
                    self.after_reset()
                } else {
                    State::AckPoll
                }
            }
            _ => return Err(BringUpError::OutOfOrder),
        };
        Ok(())
    }

    /// Deliver the bytes a [`Step::Read`] or [`Step::WriteRead`] produced,
    /// and the wall-clock nanoseconds that transaction took.
    ///
    /// The elapsed figure is not bookkeeping garnish: on the
    /// reset-acknowledgement path it is where nearly all the time goes.
    /// See [`RESET_TIMEOUT_NS`].
    pub fn read(&mut self, bytes: &[u8], elapsed_ns: u64) -> Result<(), BringUpError> {
        self.charge(elapsed_ns);
        match self.state {
            State::FetchDescriptor => {
                let d = HidDescriptor::parse(bytes).map_err(BringUpError::Descriptor)?;
                // Linux i2c-hid-core.c (v7.0), i2c_hid_quirks: ELAN
                // 04f3:* refuses the post-reset wakeup. This identity comes
                // from the validated HID descriptor, not an ACPI name (this
                // machine calls its ELAN panel CUST0000). Preserve firmware
                // overrides for other devices and the independent no-IRQ flag.
                self.quirks.no_power_after_reset |= d.w_vendor_id.get() == 0x04f3;
                self.descriptor = Some(d);
                self.state = State::PowerOn;
                Ok(())
            }
            State::AckPoll => {
                let max = self
                    .descriptor
                    .as_ref()
                    .map_or(2, |d| d.w_max_input_length.get());
                match decode(bytes, max) {
                    // The acknowledgement. Identical on the wire to an
                    // empty poll; `reset_pending` is the whole difference.
                    Frame::Empty if self.reset_pending => {
                        self.reset_pending = false;
                        self.state = self.after_reset();
                        Ok(())
                    }
                    // A real report during the ack wait: the device is
                    // demonstrably alive and talking, which satisfies the
                    // reset just as well as the zero-length frame would.
                    // Refusing it here would time out a working device.
                    Frame::Report(_) if self.reset_pending => {
                        self.reset_pending = false;
                        self.state = self.after_reset();
                        Ok(())
                    }
                    _ => {
                        self.polls += 1;
                        // Two ceilings, either of which ends the wait: the
                        // clock the caller reports, and the poll count that
                        // clock's granularity may hide. The old code had
                        // only the count and *called* it the clock.
                        if self.reset_spent_ns >= RESET_TIMEOUT_NS || self.polls >= MAX_RESET_POLLS
                        {
                            return Err(BringUpError::ResetTimeout {
                                polls: self.polls,
                                spent_ns: self.reset_spent_ns,
                            });
                        }
                        self.state = State::AwaitAck;
                        Ok(())
                    }
                }
            }
            State::FetchReportDescriptor => {
                if bytes.len() < self.expect {
                    return Err(BringUpError::ReportDescriptorShort {
                        want: self.expect,
                        got: bytes.len(),
                    });
                }
                self.state = State::Done;
                Ok(())
            }
            _ => Err(BringUpError::OutOfOrder),
        }
    }

    /// Whether the ladder has finished.
    pub fn is_done(&self) -> bool {
        self.state == State::Done
    }

    fn after_reset(&self) -> State {
        if self.quirks.no_power_after_reset {
            State::FetchReportDescriptor
        } else {
            State::PowerAgain
        }
    }

    fn command_register(&self) -> u16 {
        self.descriptor
            .as_ref()
            .map_or(0, |d| d.w_command_register.get())
    }
}
