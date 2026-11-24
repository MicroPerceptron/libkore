//! Command encoding — the bytes written to the device's command register.
//!
//! A command is the command register's address (little-endian), then a
//! two-byte word packing the report id, the report type, and the opcode.
//! Report ids of 15 or more do not fit the four-bit field and take the
//! spec's sentinel form: the nibble reads `0x0F` and the real id follows
//! as a third byte.

/// HID-over-I2C opcodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Opcode {
    /// Reset the device. Answered by a zero-length input report.
    Reset = 0x01,
    /// Read a report through the data register.
    GetReport = 0x02,
    /// Write a report through the data register.
    SetReport = 0x03,
    /// Read the idle rate.
    GetIdle = 0x04,
    /// Set the idle rate.
    SetIdle = 0x05,
    /// Read the protocol (boot or report).
    GetProtocol = 0x06,
    /// Set the protocol.
    SetProtocol = 0x07,
    /// Change the device's power state.
    SetPower = 0x08,
}

/// Which report collection a command addresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ReportType {
    /// Reports the device sends.
    Input = 1,
    /// Reports the host sends.
    Output = 2,
    /// Configuration reports read and written out of band.
    Feature = 3,
}

/// The power states `SET_POWER` selects.
///
/// The state rides the **report-id nibble**, not a data byte — the one
/// place this protocol reuses a field for something other than its name,
/// and so the one place a driver silently sends the wrong thing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PowerState {
    /// Full power.
    On = 0,
    /// Sleep.
    Sleep = 1,
}

/// The report id above which the sentinel form is required.
pub const REPORT_ID_SENTINEL: u8 = 0x0F;

/// The longest command this module encodes: register (2) + word (2) +
/// sentinel id (1) + data register (2).
pub const MAX_COMMAND_LEN: usize = 7;

/// A command, encoded into a fixed buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Command {
    buf: [u8; MAX_COMMAND_LEN],
    len: usize,
}

impl Command {
    /// The bytes to write.
    pub fn bytes(&self) -> &[u8] {
        &self.buf[..self.len]
    }

    fn build(
        command_register: u16,
        opcode: Opcode,
        report_type: Option<ReportType>,
        report_id: u8,
        data_register: Option<u16>,
    ) -> Self {
        let mut buf = [0u8; MAX_COMMAND_LEN];
        let mut len = 0;

        // The command register address always leads.
        buf[0..2].copy_from_slice(&command_register.to_le_bytes());
        len += 2;

        let ty = report_type.map(|t| t as u8).unwrap_or(0);
        let big_id = report_id >= REPORT_ID_SENTINEL;
        let nibble = if big_id {
            REPORT_ID_SENTINEL
        } else {
            report_id
        };
        buf[len] = (ty << 4) | (nibble & 0x0F);
        buf[len + 1] = opcode as u8;
        len += 2;

        // The sentinel's real id follows the word, not the register.
        if big_id {
            buf[len] = report_id;
            len += 1;
        }

        if let Some(dr) = data_register {
            buf[len..len + 2].copy_from_slice(&dr.to_le_bytes());
            len += 2;
        }

        Self { buf, len }
    }

    /// `SET_POWER`, with the state in the report-id nibble.
    pub fn set_power(command_register: u16, state: PowerState) -> Self {
        Self::build(command_register, Opcode::SetPower, None, state as u8, None)
    }

    /// `RESET`.
    pub fn reset(command_register: u16) -> Self {
        Self::build(command_register, Opcode::Reset, None, 0, None)
    }

    /// `GET_REPORT`, which reads through the data register.
    pub fn get_report(
        command_register: u16,
        data_register: u16,
        report_type: ReportType,
        report_id: u8,
    ) -> Self {
        Self::build(
            command_register,
            Opcode::GetReport,
            Some(report_type),
            report_id,
            Some(data_register),
        )
    }

    /// `SET_REPORT`'s header. The payload follows on the same write.
    pub fn set_report(
        command_register: u16,
        data_register: u16,
        report_type: ReportType,
        report_id: u8,
    ) -> Self {
        Self::build(
            command_register,
            Opcode::SetReport,
            Some(report_type),
            report_id,
            Some(data_register),
        )
    }
}

/// A bare register address, for the two fetches that are plain reads
/// rather than commands: the HID descriptor and the report descriptor.
pub fn register_address(register: u16) -> [u8; 2] {
    register.to_le_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_power_puts_the_state_in_the_report_id_nibble() {
        // The encoding trap: the power state is *not* a data byte.
        assert_eq!(
            Command::set_power(0x0005, PowerState::On).bytes(),
            &[0x05, 0x00, 0x00, 0x08]
        );
        assert_eq!(
            Command::set_power(0x0005, PowerState::Sleep).bytes(),
            &[0x05, 0x00, 0x01, 0x08]
        );
    }

    #[test]
    fn reset_is_the_command_register_then_opcode_one() {
        assert_eq!(Command::reset(0x0005).bytes(), &[0x05, 0x00, 0x00, 0x01]);
        // A different command register moves only the first two bytes.
        assert_eq!(Command::reset(0x1234).bytes(), &[0x34, 0x12, 0x00, 0x01]);
    }

    #[test]
    fn a_small_report_id_rides_the_nibble_and_the_type_the_bits_above_it() {
        let c = Command::get_report(0x0005, 0x0006, ReportType::Feature, 3);
        //  cmd reg | (type 3 << 4) | id 3 | opcode 2 | data reg
        assert_eq!(c.bytes(), &[0x05, 0x00, 0x33, 0x02, 0x06, 0x00]);

        let c = Command::get_report(0x0005, 0x0006, ReportType::Input, 1);
        assert_eq!(c.bytes(), &[0x05, 0x00, 0x11, 0x02, 0x06, 0x00]);
    }

    #[test]
    fn a_report_id_of_fifteen_or_more_takes_the_sentinel_form() {
        // 15 is the first id that does not fit the nibble. Getting the
        // boundary wrong sends a command for report 15 when report 15 was
        // meant, or truncates every id above it to 15.
        let c = Command::get_report(0x0005, 0x0006, ReportType::Input, 15);
        assert_eq!(c.bytes(), &[0x05, 0x00, 0x1F, 0x02, 15, 0x06, 0x00]);

        let c = Command::get_report(0x0005, 0x0006, ReportType::Input, 200);
        assert_eq!(c.bytes(), &[0x05, 0x00, 0x1F, 0x02, 200, 0x06, 0x00]);

        // 14 is the last id that fits inline.
        let c = Command::get_report(0x0005, 0x0006, ReportType::Input, 14);
        assert_eq!(c.bytes(), &[0x05, 0x00, 0x1E, 0x02, 0x06, 0x00]);
    }

    #[test]
    fn set_report_carries_the_data_register_like_get_report() {
        let c = Command::set_report(0x0005, 0x0006, ReportType::Output, 2);
        assert_eq!(c.bytes(), &[0x05, 0x00, 0x22, 0x03, 0x06, 0x00]);
    }

    #[test]
    fn no_command_ever_exceeds_the_buffer() {
        let longest = Command::get_report(0xffff, 0xffff, ReportType::Feature, 255);
        assert_eq!(longest.bytes().len(), MAX_COMMAND_LEN);
    }

    #[test]
    fn a_bare_register_fetch_is_just_the_little_endian_address() {
        assert_eq!(register_address(0x0002), [0x02, 0x00]);
        assert_eq!(register_address(0xabcd), [0xcd, 0xab]);
    }
}
