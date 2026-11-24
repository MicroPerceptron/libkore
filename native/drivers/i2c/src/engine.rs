//! The **generic transfer engine** — written once, over any
//! [`I2cMaster`].
//!
//! Everything that is the same for every I2C controller lives here:
//! validating the request, refusing to start over a latched abort, pumping
//! the sans-IO state machine against a bounded [`Pacing`], turning an
//! exhausted budget into a timeout that carries the controller's
//! registers, and keeping the books.
//!
//! It never names a register, an offset, or a bit. That is the test for
//! whether a change belongs here: if it needs to know which IP it is
//! driving, it belongs in the impl.

use crate::bus::{I2cBus, Xfer};
use crate::error::I2cError;
use crate::master::{BusConfig, I2cMaster, Pacing, RecoveryOutcome, RecoveryStart, XferState};

/// The engine's books — cheap counters the service folds into `status`.
///
/// Aborts and timeouts are counted separately from each other and from
/// address NACKs because they mean different things on a photograph: a
/// NACK count that tracks probe attempts is healthy, a nonzero timeout
/// count never is, and an abort count rising in steady state is a bus
/// problem rather than a driver one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct I2cStats {
    /// Transfers that completed without fault.
    pub transfers: u64,
    /// Bytes written to the bus.
    pub written: u64,
    /// Bytes read from the bus.
    pub read: u64,
    /// Transfers that ended in an abort which was **not** an address NACK.
    pub aborts: u64,
    /// Transfers that ended in an address NACK.
    pub address_nacks: u64,
    /// Transfers whose pacing budget ran out.
    pub timeouts: u64,
    /// Requests rejected by validation or before the transfer could start.
    /// A failed start still recovers any stale controller state.
    pub refused: u64,
    /// `poll` calls that reported no progress — the empty-poll count the
    /// epic's "zero empty polls in steady state" evidence is read from.
    pub idle_polls: u64,
}

/// One configured controller, plus the generic pump.
pub struct I2cEngine<M: I2cMaster> {
    master: M,
    stats: I2cStats,
}

impl<M: I2cMaster> I2cEngine<M> {
    fn recover(&mut self, regs: &M::Regs, pacing: &mut impl Pacing) -> RecoveryOutcome {
        match self.master.begin_cancel(regs) {
            RecoveryStart::Complete(outcome) => return outcome,
            RecoveryStart::WaitCancel => {}
            RecoveryStart::WaitAbort => unreachable!("cancel cannot begin in abort state"),
        }
        pacing.reset_recovery();
        loop {
            if self.master.cancel_complete(regs) {
                self.master.finish_recovery(regs);
                return RecoveryOutcome::Cancelled;
            }
            if !pacing.step() {
                break;
            }
        }

        match self.master.begin_abort(regs) {
            RecoveryStart::Complete(outcome) => {
                self.master.finish_recovery(regs);
                return outcome;
            }
            RecoveryStart::WaitAbort => {}
            RecoveryStart::WaitCancel => unreachable!("abort cannot begin in cancel state"),
        }
        pacing.reset_recovery();
        loop {
            if self.master.abort_complete(regs) {
                self.master.finish_recovery(regs);
                return RecoveryOutcome::Aborted;
            }
            if !pacing.step() {
                let snapshot = self.master.timeout_snapshot(regs);
                self.master.finish_recovery(regs);
                return RecoveryOutcome::Failed(snapshot);
            }
        }
    }

    /// Wrap a controller. Nothing touches hardware until
    /// [`probe`](Self::probe).
    pub fn new(master: M) -> Self {
        Self {
            master,
            stats: I2cStats::default(),
        }
    }

    /// The controller's identity self-check.
    pub fn probe(&mut self, regs: &M::Regs) -> Result<(), I2cError<M::Abort>> {
        self.master.probe(regs)
    }

    /// Configure the bus. May be called repeatedly.
    pub fn configure(&mut self, regs: &M::Regs, cfg: &BusConfig) -> Result<(), I2cError<M::Abort>> {
        self.master.configure(regs, cfg)
    }

    /// Run one transfer to completion, or to its budget.
    ///
    /// On an abort the controller is **recovered before returning**, so
    /// the caller never has to remember to: a latched abort source would
    /// otherwise be read as the cause of the *next* failure, which is a
    /// bug that presents as a stream of identical wrong diagnoses.
    pub fn transfer(
        &mut self,
        regs: &M::Regs,
        xfer: &Xfer<'_>,
        rx: &mut [u8],
        pacing: &mut impl Pacing,
    ) -> Result<usize, I2cError<M::Abort>> {
        if let Err(e) = xfer.validate(rx.len()) {
            self.stats.refused += 1;
            return Err(I2cError::Request(e));
        }

        if let Err(error) = self.master.start(regs, xfer) {
            // A start can fail before poll (e.g. firmware left an abort
            // latched). Preserve the refusal, but give cancel/abort the same
            // bounded completion allowance as a failed running transfer:
            // merely issuing recovery can leave the next child racing it.
            self.stats.refused += 1;
            let _ = self.recover(regs, pacing);
            return Err(error);
        }
        pacing.reset(xfer.write.len().saturating_add(xfer.read_len));

        loop {
            // Poll before pacing: a transfer that the FIFOs already
            // finished must not be charged a wait it did not need. This
            // ordering is also what lets a zero-step budget succeed on a
            // transfer that completes immediately, which is the shape a
            // real interrupt-driven caller has after its wake.
            match self.master.poll(regs, xfer, rx) {
                XferState::Done { read } => {
                    self.stats.transfers += 1;
                    self.stats.written += xfer.write.len() as u64;
                    self.stats.read += read as u64;
                    return Ok(read);
                }
                XferState::Aborted(detail, snap) => {
                    let recovery = self.recover(regs, pacing);
                    if crate::error::AbortDetail::is_address_nack(&detail) {
                        self.stats.address_nacks += 1;
                    } else {
                        self.stats.aborts += 1;
                    }
                    return Err(I2cError::Abort {
                        detail,
                        snapshot: snap,
                        recovery,
                    });
                }
                // Progress means there is more to do *now*: pacing here
                // would sleep on a FIFO that is already asking to be
                // serviced. Only a poll that found nothing is charged a
                // wait, which is also what makes `idle_polls` mean what
                // the acceptance criterion needs it to mean.
                XferState::Busy { progressed: true } => continue,
                XferState::Busy { progressed: false } => {
                    self.stats.idle_polls += 1;
                    if !pacing.step() {
                        self.stats.timeouts += 1;
                        let snap = self.master.timeout_snapshot(regs);
                        // The controller is mid-sequence and must not be
                        // left that way for the next transfer to inherit.
                        let recovery = self.recover(regs, pacing);
                        return Err(I2cError::Timeout {
                            snapshot: snap,
                            recovery,
                        });
                    }
                }
            }
        }
    }

    /// The books.
    pub fn stats(&self) -> &I2cStats {
        &self.stats
    }

    /// The controller, for a witness line's sake.
    pub fn master(&self) -> &M {
        &self.master
    }

    /// Bind this engine to a register window and a pacing budget, giving
    /// the narrow [`I2cBus`] a protocol driver holds.
    pub fn bus<'a, P: Pacing>(&'a mut self, regs: &'a M::Regs, pacing: P) -> Bus<'a, M, P> {
        Bus {
            engine: self,
            regs,
            pacing,
        }
    }
}

/// An [`I2cEngine`] bound to its registers and its pacing — the
/// [`I2cBus`] a protocol driver actually holds.
///
/// The binding is a separate type rather than state on the engine so the
/// borrow is explicit and scoped: a protocol driver holds the bus for the
/// duration of one operation, and cannot outlive the register window it
/// was handed.
pub struct Bus<'a, M: I2cMaster, P: Pacing> {
    engine: &'a mut I2cEngine<M>,
    regs: &'a M::Regs,
    pacing: P,
}

impl<M: I2cMaster, P: Pacing> Bus<'_, M, P> {
    /// The books, through the binding.
    pub fn stats(&self) -> &I2cStats {
        self.engine.stats()
    }
}

impl<M: I2cMaster, P: Pacing> I2cBus for Bus<'_, M, P> {
    type Error = I2cError<M::Abort>;

    fn write(&mut self, addr: u8, bytes: &[u8]) -> Result<(), Self::Error> {
        let xfer = Xfer::write(addr, bytes);
        self.engine
            .transfer(self.regs, &xfer, &mut [], &mut self.pacing)
            .map(|_| ())
    }

    fn write_read(
        &mut self,
        addr: u8,
        bytes: &[u8],
        read: &mut [u8],
    ) -> Result<usize, Self::Error> {
        let xfer = Xfer::write_read(addr, bytes, read.len());
        self.engine
            .transfer(self.regs, &xfer, read, &mut self.pacing)
    }
}
