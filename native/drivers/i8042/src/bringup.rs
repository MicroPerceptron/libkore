//! The controller **bring-up ladder** (#419) — from an untouched i8042 to
//! one delivering keystrokes on its interrupt line.
//!
//! Every rung is a bounded [`handshake`] with the floating-bus fault, and
//! every rung has its own [`Rung`] identity, so a refusal names *which
//! step* the controller failed rather than leaving "the keyboard didn't
//! work". On a serial-less machine that name is the entire diagnosis.
//!
//! # The order is load-bearing
//!
//! 1. **Disable both ports** before touching anything, so no device can
//!    inject bytes while the controller is being configured.
//! 2. **Flush** whatever is already waiting — firmware leaves bytes here.
//! 3. **Read the command byte** and keep it. The bring-up *modifies*
//!    firmware's configuration; it does not replace it with an assumption.
//! 3b. **Silence the interrupt** for the rest of the ladder — firmware
//!    may leave it enabled, and every device ACK below would otherwise
//!    raise the line before anything is parked to receive it.
//! 4. **Self-test** (`0xAA` → `0x55`). A failure is a refusal carrying the
//!    byte the controller actually returned, never a retry loop.
//! 5. **Port test** (`0xAB` → `0x00`).
//! 6. **Reset the device** (`0xFF` → `0xFA 0xAA`). This comes *before* the
//!    query below: a reset restores the keyboard's default scancode set,
//!    so anything asked earlier is stale by the time bytes flow.
//! 7. **Read the scancode set and the translation bit** — see below.
//! 8. **Enable scanning**, then the keyboard interrupt — last, so a
//!    keystroke cannot arrive with nowhere to go.
//!
//! # Why rung 6 exists
//!
//! The controller can hand up set 1 or set 2, and its translation bit can
//! silently convert one into the other. Assuming either is the single most
//! common way a PS/2 driver produces garbage on hardware that is working
//! perfectly. So the ladder *asks*, and hands the answer to the decoder.

use kore_driver::{Ack, Budget, Regs, handshake};

use crate::port::{self, PortAccess, PortRegs, status};

/// Controller commands (written to the status/command port).
mod cmd {
    /// Read the controller configuration ("command") byte.
    pub const READ_CONFIG: u32 = 0x20;
    /// Write the controller configuration byte.
    pub const WRITE_CONFIG: u32 = 0x60;
    /// Disable the aux (mouse) port.
    pub const DISABLE_AUX: u32 = 0xa7;
    /// Disable the keyboard port.
    pub const DISABLE_KBD: u32 = 0xad;
    /// Enable the keyboard port.
    pub const ENABLE_KBD: u32 = 0xae;
    /// Controller self-test.
    pub const SELF_TEST: u32 = 0xaa;
    /// Keyboard-port test.
    pub const PORT_TEST: u32 = 0xab;
}

/// Configuration-byte bits.
mod config {
    /// Raise an interrupt when the keyboard has a byte.
    pub const KBD_INTERRUPT: u8 = 1 << 0;
    /// Raise an interrupt when the aux port has a byte.
    pub const AUX_INTERRUPT: u8 = 1 << 1;
    /// First-port clock **disable** — set means the keyboard is cut off
    /// from the controller. [`cmd::DISABLE_KBD`] sets this bit and
    /// [`cmd::ENABLE_KBD`] clears it, so the byte this ladder reads at
    /// rung 3 always has it set: the read happens *after* rung 1's
    /// disable, by design.
    ///
    /// [`cmd::DISABLE_KBD`]: super::cmd::DISABLE_KBD
    /// [`cmd::ENABLE_KBD`]: super::cmd::ENABLE_KBD
    pub const KBD_DISABLE: u8 = 1 << 4;
    /// Translate set 2 into set 1 on the way up.
    pub const TRANSLATE: u8 = 1 << 6;
}

/// The byte to write back at the end of the ladder: everything firmware
/// chose, plus the keyboard interrupt, **minus the port-disable bit this
/// ladder set itself**.
///
/// Clearing [`config::KBD_DISABLE`] is load-bearing rather than tidy. The
/// captured byte is read at rung 3, after rung 1 disabled the port, so it
/// carries that disable. Writing it back unmodified re-cuts the keyboard
/// from the controller *after* the device was reset and told to scan —
/// leaving a controller whose interrupt is enabled and whose keyboard can
/// never reach it. Bring-up returns `Ok`, every rung reports healthy, and
/// no keystroke is ever delivered.
///
/// The aux port's disable bit is deliberately **not** cleared: rung 1
/// turned that port off and nothing here services a mouse.
const fn final_config(captured: u8) -> u8 {
    (captured & !config::KBD_DISABLE) | config::KBD_INTERRUPT
}

/// Device commands (written to the data port).
mod dev {
    /// Reset and self-test.
    pub const RESET: u32 = 0xff;
    /// Enable scanning.
    pub const ENABLE_SCAN: u32 = 0xf4;
    /// Disable scanning (so configuration is not raced by keystrokes).
    pub const DISABLE_SCAN: u32 = 0xf5;
    /// Get/set the scancode set.
    pub const SCANCODE_SET: u32 = 0xf0;
    /// Device acknowledgement.
    pub const ACK: u8 = 0xfa;
    /// Self-test passed.
    pub const SELF_TEST_OK: u8 = 0xaa;
}

/// Expected controller self-test reply.
const SELF_TEST_PASS: u8 = 0x55;
/// Expected port-test reply.
const PORT_TEST_PASS: u8 = 0x00;

/// Which rung refused — the whole diagnosis on a serial-less machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rung {
    /// The bus floats: no controller decodes these ports.
    Absent,
    /// The controller never drained its input buffer.
    InputStuck,
    /// The configuration byte could not be read.
    ReadConfig,
    /// Self-test failed, carrying what the controller returned instead of
    /// `0x55`.
    SelfTest(u8),
    /// The keyboard port failed its test.
    PortTest(u8),
    /// The device did not acknowledge a command.
    NoDeviceAck,
    /// The device's reset did not complete.
    Reset(u8),
    /// The configuration byte could not be written back.
    WriteConfig,
}

/// What the ladder learned — handed to the decoder so it never guesses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Brought {
    /// The set the controller is actually handing up, after accounting
    /// for the translation bit.
    pub set: crate::ScancodeSet,
    /// Whether the controller's translation bit was on.
    pub translating: bool,
    /// The configuration byte as read at rung 3 — firmware's choices for
    /// translation, the system flag and the rest.
    ///
    /// **Not** the byte firmware left: the read follows rung 1's port
    /// disables, so bits 4 and 5 report *this ladder's* commands rather
    /// than firmware's. A teardown must therefore restore the port bits
    /// from what it wants, not from here — writing this back verbatim
    /// disables both ports.
    pub firmware_config: u8,
}

/// Wait budget per handshake, in wall-clock nanoseconds. Generous: this
/// runs once at bring-up, and an embedded controller emulating an i8042 can
/// be slow to answer — Linux allows 10 ms for a controller command and more
/// for a device reset.
///
/// A *time* rather than a probe count because the probes here are port I/O
/// to 0x60/0x64, which on a machine with no controller behind them is an
/// unclaimed legacy cycle: tens of microseconds each, non-posted, with the
/// core stalled inside the `in`. At that cost the old 10 000-probe budget
/// was not the ~10 ms it reads as, it was minutes — per rung.
const BUDGET_NS: u64 = 50_000_000;

/// The clockless fallback: what [`BUDGET_NS`] degrades to before any counter
/// is calibrated. Only a probe count is expressible there, and this is the
/// count this ladder shipped with.
const ATTEMPTS: u32 = 10_000;

/// This ladder's per-handshake budget over the caller's clock.
#[inline]
fn budget(clock: fn() -> Option<u64>) -> Budget {
    Budget::new(BUDGET_NS, ATTEMPTS, clock)
}

/// Bring the controller up. `delay` is the caller's inter-probe pause
/// (a short spin on metal) and `clock` its monotonic nanosecond reading,
/// or `None` on a machine with no calibrated counter — the pair every
/// rung's [`Budget`] is built from.
///
/// Every rung is bounded and fault-aware, so this either returns a
/// configured controller or names the rung that refused. It never loops
/// waiting for hardware to change its mind.
pub fn bring_up<P: PortAccess>(
    regs: &PortRegs<P>,
    clock: fn() -> Option<u64>,
    mut delay: impl FnMut(),
) -> Result<Brought, Rung> {
    if regs.absent() {
        return Err(Rung::Absent);
    }

    // 1 · Both ports off before anything is configured.
    write_cmd(regs, cmd::DISABLE_KBD, clock, &mut delay)?;
    write_cmd(regs, cmd::DISABLE_AUX, clock, &mut delay)?;

    // 2 · Drain whatever firmware left behind.  Bounded: a controller
    //     that always reports a byte would otherwise spin here forever.
    for _ in 0..16 {
        if regs.status() & status::OBF == 0 {
            break;
        }
        let _ = regs.read32(port::DATA);
    }

    // 3 · Read the configuration and KEEP it — this ladder edits the
    //     machine's configuration rather than imposing its own.  Note the
    //     read follows rung 1, so the byte carries our own port-disable
    //     bits; `final_config` strips the keyboard's before writing back.
    write_cmd(regs, cmd::READ_CONFIG, clock, &mut delay)?;
    let firmware_config = read_data(regs, clock, &mut delay).map_err(|_| Rung::ReadConfig)? as u8;

    // 3b · **Silence the line for the rest of the ladder.**
    //
    //      Rung 1 disabled the ports but nothing disabled the controller's
    //      *interrupt*, so firmware's setting stood — and on the Yoga
    //      firmware leaves it on (`0x77`, bit 0 set). From rung 7 onward
    //      every device command answers `0xFA`, and each of those raises
    //      the line: roughly a dozen interrupts fired mid-bring-up, before
    //      the relay is parked and while the route is at its most freshly
    //      programmed. "The interrupt, LAST" was only ever true of *setting*
    //      it; it was never cleared first.
    //
    //      A failure here is not fatal: the ladder still works with the
    //      line live, it is merely noisy, and refusing the controller over
    //      a quieting step would trade a working keyboard for a tidy one.
    let quiet = firmware_config & !(config::KBD_INTERRUPT | config::AUX_INTERRUPT);
    if write_cmd(regs, cmd::WRITE_CONFIG, clock, &mut delay).is_ok() {
        let _ = write_data(regs, u32::from(quiet), clock, &mut delay);
    }

    // 4 · Controller self-test.  Note it can reset the configuration on
    //     some silicon, which is exactly why the byte was read first.
    write_cmd(regs, cmd::SELF_TEST, clock, &mut delay)?;
    let got = read_data(regs, clock, &mut delay).map_err(|_| Rung::SelfTest(0))? as u8;
    if got != SELF_TEST_PASS {
        return Err(Rung::SelfTest(got));
    }

    // 5 · Keyboard-port test.
    write_cmd(regs, cmd::PORT_TEST, clock, &mut delay)?;
    let got = read_data(regs, clock, &mut delay).map_err(|_| Rung::PortTest(0))? as u8;
    if got != PORT_TEST_PASS {
        return Err(Rung::PortTest(got));
    }

    // The port must be enabled for the device to answer the queries
    // below; its INTERRUPT stays off until the very end.
    write_cmd(regs, cmd::ENABLE_KBD, clock, &mut delay)?;

    // Quiet the device while it is being interrogated, so a keystroke
    // cannot be mistaken for a reply.
    let _ = device_cmd(regs, dev::DISABLE_SCAN, clock, &mut delay);

    // 6 · Reset the device **before** asking it anything.  `0xFF` restores
    //     the keyboard's power-on default scancode set, so a set read
    //     before this is stale the moment bytes start flowing — the
    //     decoder would be built for one encoding and fed another.
    device_cmd(regs, dev::RESET, clock, &mut delay)?;
    let post = read_data(regs, clock, &mut delay).unwrap_or(0) as u8;
    if post != dev::SELF_TEST_OK && post != dev::ACK {
        return Err(Rung::Reset(post));
    }
    // The reset also re-enables scanning, so quiet the device again: a
    // keystroke arriving now would be read as the query's reply below.
    let _ = device_cmd(regs, dev::DISABLE_SCAN, clock, &mut delay);

    // 7 · Ask which set is live — never assume, and ask the device in the
    //     state it will actually run in.  `0xF0 0x00` returns the current
    //     set; the translation bit says whether the controller is
    //     converting it on the way up.
    device_cmd(regs, dev::SCANCODE_SET, clock, &mut delay)?;
    device_cmd(regs, 0x00, clock, &mut delay)?;
    let reported = read_data(regs, clock, &mut delay).unwrap_or(0) as u8;
    let translating = firmware_config & config::TRANSLATE != 0;
    // What reaches US is set 1 whenever the controller translates,
    // regardless of what the device itself is emitting.
    let set = if translating || reported == 1 {
        crate::ScancodeSet::Set1
    } else {
        crate::ScancodeSet::Set2
    };

    // 8 · Scanning on, now that the encoding is known.
    device_cmd(regs, dev::ENABLE_SCAN, clock, &mut delay)?;

    // 7 · The interrupt, LAST — the caller parks before calling this, so
    //     from here a keystroke has somewhere to go.
    write_cmd(regs, cmd::WRITE_CONFIG, clock, &mut delay)?;
    write_data(
        regs,
        u32::from(final_config(firmware_config)),
        clock,
        &mut delay,
    )
    .map_err(|_| Rung::WriteConfig)?;

    Ok(Brought {
        set,
        translating,
        firmware_config,
    })
}

/// Write a controller command, waiting for the input buffer first.
fn write_cmd<P: PortAccess>(
    regs: &PortRegs<P>,
    command: u32,
    clock: fn() -> Option<u64>,
    delay: &mut impl FnMut(),
) -> Result<(), Rung> {
    wait_input_clear(regs, clock, delay)?;
    regs.write32(port::STATUS, command);
    Ok(())
}

/// Write a data byte, waiting for the input buffer first.
fn write_data<P: PortAccess>(
    regs: &PortRegs<P>,
    byte: u32,
    clock: fn() -> Option<u64>,
    delay: &mut impl FnMut(),
) -> Result<(), Rung> {
    wait_input_clear(regs, clock, delay)?;
    regs.write32(port::DATA, byte);
    Ok(())
}

/// Send a byte to the *device* and consume its acknowledgement.
fn device_cmd<P: PortAccess>(
    regs: &PortRegs<P>,
    byte: u32,
    clock: fn() -> Option<u64>,
    delay: &mut impl FnMut(),
) -> Result<(), Rung> {
    write_data(regs, byte, clock, delay)?;
    match read_data(regs, clock, delay) {
        Ok(v) if v as u8 == dev::ACK => Ok(()),
        // A device that answers something other than ACK has not done what
        // was asked; the ladder refuses rather than proceeding on hope.
        Ok(_) => Err(Rung::NoDeviceAck),
        Err(rung) => Err(rung),
    }
}

/// Wait until the controller has consumed the last byte written.
fn wait_input_clear<P: PortAccess>(
    regs: &PortRegs<P>,
    clock: fn() -> Option<u64>,
    delay: &mut impl FnMut(),
) -> Result<(), Rung> {
    match handshake(
        regs,
        budget(clock),
        delay,
        PortRegs::<P>::input_clear,
        PortRegs::<P>::vanished,
    ) {
        Ack::Acked => Ok(()),
        Ack::Faulted => Err(Rung::Absent),
        Ack::TimedOut => Err(Rung::InputStuck),
    }
}

/// Wait for a byte and read it, skipping anything the aux channel sent.
fn read_data<P: PortAccess>(
    regs: &PortRegs<P>,
    clock: fn() -> Option<u64>,
    delay: &mut impl FnMut(),
) -> Result<u32, Rung> {
    match handshake(
        regs,
        budget(clock),
        delay,
        PortRegs::<P>::output_full,
        PortRegs::<P>::vanished,
    ) {
        Ack::Acked => {
            // An aux byte is not ours: consume and report absence of a
            // keyboard reply rather than decoding a mouse packet as one.
            if regs.status() & status::AUX != 0 {
                let _ = regs.read32(port::DATA);
                return Err(Rung::NoDeviceAck);
            }
            Ok(regs.read32(port::DATA))
        }
        Ack::Faulted => Err(Rung::Absent),
        Ack::TimedOut => Err(Rung::NoDeviceAck),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An absent controller is named as absent at the first rung — no
    /// commands are written into the void.
    #[test]
    fn an_absent_controller_refuses_at_rung_one() {
        struct Floating;
        impl PortAccess for Floating {
            fn inb(&self, _: usize) -> u8 {
                port::FLOATING as u8
            }
            fn outb(&self, _: usize, _: u8) {}
        }
        let regs = PortRegs(Floating);
        assert_eq!(bring_up(&regs, || None, || {}), Err(Rung::Absent));
    }

    /// The byte written back must never re-disable the keyboard port.
    ///
    /// This is the whole bug: the captured byte is read *after* rung 1
    /// disables the port, so it always has [`config::KBD_DISABLE`] set.
    /// Writing it back with only the interrupt OR-ed in leaves a
    /// controller that is configured, reset, scanning, interrupt-enabled
    /// — and cut off from its keyboard, with every rung reporting
    /// success.
    #[test]
    fn the_written_back_config_never_re_disables_the_keyboard() {
        // What rung 3 actually reads on a translating controller whose
        // ports this ladder has just turned off: translation on, both
        // port-disable bits set by our own 0xAD / 0xA7.
        let captured = config::TRANSLATE | config::KBD_DISABLE | (1 << 5);
        let written = final_config(captured);

        assert_eq!(
            written & config::KBD_DISABLE,
            0,
            "the keyboard port must be live in the byte written back"
        );
        assert_ne!(
            written & config::KBD_INTERRUPT,
            0,
            "the keyboard interrupt must be enabled"
        );
        assert_ne!(
            written & config::TRANSLATE,
            0,
            "firmware's translation choice must survive"
        );
        assert_ne!(written & (1 << 5), 0, "the aux port stays disabled");
    }

    /// Every other bit is firmware's, and passes through untouched.
    #[test]
    fn final_config_edits_only_the_two_bits_it_owns() {
        let owned = config::KBD_DISABLE | config::KBD_INTERRUPT;
        for captured in 0u8..=u8::MAX {
            assert_eq!(
                final_config(captured) & !owned,
                captured & !owned,
                "captured {captured:#04x} had a bit changed that the ladder does not own"
            );
        }
    }
}
