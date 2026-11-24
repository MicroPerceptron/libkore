//! The controller's two ports, behind the shared [`Regs`] seam.
//!
//! An i8042 is addressed as two byte-wide ports — data and status/command
//! — which is what [`Regs`] describes: *registers with read/write side
//! effects, addressed by offset*. The only mismatch is width, and that is
//! handled honestly below rather than by inventing a second seam.
//!
//! Reusing the seam is the point: it buys this driver
//! [`handshake`](kore_driver::handshake) — the bounded, fault-aware wait
//! every other kore driver already uses — instead of a hand-rolled spin
//! loop that would drift from that discipline the moment either changed.
//!
//! [`Regs`]: kore_driver::Regs

use kore_driver::Regs;

/// Offset of the **data** port in this seam (the controller's `0x60`).
pub const DATA: usize = 0;
/// Offset of the **status / command** port (the controller's `0x64`).
/// Reads return status; writes are commands.
pub const STATUS: usize = 4;

/// Status-register bits (read from [`STATUS`]).
pub mod status {
    /// Output buffer full — a byte is waiting at the data port.
    pub const OBF: u32 = 1 << 0;
    /// Input buffer full — the controller has not consumed the last byte
    /// written, so writing now would lose it.
    pub const IBF: u32 = 1 << 1;
    /// The waiting byte came from the **aux** channel (mouse/touchpad),
    /// not the keyboard.
    pub const AUX: u32 = 1 << 5;
}

/// A **floating bus** reads as all-ones: no controller is decoding these
/// ports at all.
///
/// This is the driver's fault condition in the
/// [`handshake`](kore_driver::handshake) sense — the controller has not
/// merely failed to answer *yet*, it is absent — so a wait abandons
/// immediately instead of burning its whole budget against dead silicon.
/// (`Ack::Faulted` versus `Ack::TimedOut` is exactly this distinction, and
/// it is why the shared seam was worth adopting rather than spinning.)
pub const FLOATING: u32 = 0xff;

/// How the driver reaches its two ports.
///
/// Implemented over an [`IoPortGrant`]-backed accessor in the userland
/// driver task, so every access is one the task's capability admits —
/// there is no path here that reaches a port the grant does not cover
/// (the TSS bitmap refuses it at CPL 3 regardless, but the type says so
/// too).
///
/// [`IoPortGrant`]: kore-abi's `ResourceKind::IoPortGrant`
pub trait PortAccess {
    /// Read the byte-wide port at `off` ([`DATA`] or [`STATUS`]).
    fn inb(&self, off: usize) -> u8;
    /// Write the byte-wide port at `off`.
    fn outb(&self, off: usize, val: u8);
}

/// Adapts a [`PortAccess`] to the shared [`Regs`] seam.
///
/// **Width**: the ports are 8 bits and the seam is 32. Reads
/// zero-extend, writes take the low byte — the widening is total and
/// lossless in the direction that matters (a status byte fits a `u32`),
/// and a write of anything above `0xff` is a caller bug this cannot
/// silently truncate into a different command, so it is asserted in
/// debug builds.
pub struct PortRegs<P: PortAccess>(pub P);

impl<P: PortAccess> Regs for PortRegs<P> {
    #[inline]
    fn read32(&self, off: usize) -> u32 {
        u32::from(self.0.inb(off))
    }

    #[inline]
    fn write32(&self, off: usize, val: u32) {
        debug_assert!(
            val <= 0xff,
            "i8042 ports are byte-wide; a wider write would truncate into a \
             different command"
        );
        self.0.outb(off, val as u8);
    }
}

impl<P: PortAccess> PortRegs<P> {
    /// The status byte.
    #[inline]
    pub fn status(&self) -> u32 {
        self.read32(STATUS)
    }

    /// Whether the controller is absent (the bus floats).
    #[inline]
    pub fn absent(&self) -> bool {
        self.status() == FLOATING
    }

    /// The fault predicate to hand [`handshake`](kore_driver::handshake):
    /// the controller vanished mid-sequence.
    #[inline]
    pub fn vanished(regs: &Self) -> bool {
        regs.status() == FLOATING
    }

    /// A byte is waiting to be read.
    #[inline]
    pub fn output_full(regs: &Self) -> bool {
        regs.status() & status::OBF != 0
    }

    /// The controller has consumed the last byte written — the
    /// precondition for writing another.
    #[inline]
    pub fn input_clear(regs: &Self) -> bool {
        regs.status() & status::IBF == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::RefCell;
    use kore_driver::{Ack, Budget, handshake};

    /// A recorded port transcript — **not** a controller model. It replays
    /// a byte sequence and records writes, so the tests here exercise the
    /// *wait discipline* (does it stop on a fault? does it respect the
    /// budget?) without pretending to be an i8042. The controller's real
    /// behaviour is proven against QEMU's actual chip in the boot gate.
    struct Transcript {
        status: RefCell<alloc::vec::Vec<u8>>,
        writes: RefCell<alloc::vec::Vec<(usize, u8)>>,
    }

    impl Transcript {
        fn new(status: &[u8]) -> Self {
            Self {
                status: RefCell::new(status.iter().rev().copied().collect()),
                writes: RefCell::new(alloc::vec::Vec::new()),
            }
        }
    }

    impl PortAccess for &Transcript {
        fn inb(&self, _off: usize) -> u8 {
            let mut s = self.status.borrow_mut();
            // The last value repeats once the transcript runs out, so a
            // budget can expire against a steady state.
            if s.len() > 1 {
                s.pop().unwrap()
            } else {
                *s.last().unwrap_or(&0)
            }
        }
        fn outb(&self, off: usize, val: u8) {
            self.writes.borrow_mut().push((off, val));
        }
    }

    /// The wait ends as soon as the controller answers.
    #[test]
    fn a_handshake_acks_when_the_bit_appears() {
        let t = Transcript::new(&[0x00, 0x00, status::OBF as u8]);
        let regs = PortRegs(&t);
        let ack = handshake(
            &regs,
            Budget::spins(8),
            || {},
            PortRegs::output_full,
            PortRegs::vanished,
        );
        assert_eq!(ack, Ack::Acked);
    }

    /// **The reason this driver adopted the shared seam**: an absent
    /// controller (floating bus) is reported as `Faulted`, not as a
    /// timeout — so bring-up says "there is no controller here" instead
    /// of "the controller was slow", and does it without spending the
    /// whole budget.
    #[test]
    fn an_absent_controller_faults_rather_than_timing_out() {
        let t = Transcript::new(&[FLOATING as u8]);
        let regs = PortRegs(&t);
        assert!(regs.absent());
        let ack = handshake(
            &regs,
            Budget::spins(1000),
            || {},
            PortRegs::output_full,
            PortRegs::vanished,
        );
        assert_eq!(ack, Ack::Faulted, "absence is a fault, not slowness");
    }

    /// A controller that simply never answers times out — distinct from
    /// the above, and the distinction is what the caller acts on.
    #[test]
    fn a_silent_controller_times_out() {
        let t = Transcript::new(&[0x00]);
        let regs = PortRegs(&t);
        let ack = handshake(
            &regs,
            Budget::spins(4),
            || {},
            PortRegs::output_full,
            PortRegs::vanished,
        );
        assert_eq!(ack, Ack::TimedOut);
    }

    /// Writes reach the port they name, at byte width.
    #[test]
    fn writes_go_to_the_named_port() {
        let t = Transcript::new(&[0x00]);
        let regs = PortRegs(&t);
        regs.write32(STATUS, 0xaa);
        regs.write32(DATA, 0xf4);
        assert_eq!(&*t.writes.borrow(), &[(STATUS, 0xaa), (DATA, 0xf4)]);
    }
}
