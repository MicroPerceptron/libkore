//! The **HID descriptor** — the 30-byte block every HID-over-I2C device
//! publishes at its descriptor register, and the map to everything else.
//!
//! Read once at bring-up. Every register the driver later touches comes
//! from here, so a descriptor accepted while malformed is a driver reading
//! and writing addresses the device never claimed.

use zerocopy::byteorder::little_endian::{U16, U32};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

/// The descriptor's own length, in bytes. The spec fixes it at 30, and a
/// device reporting anything else is not speaking this protocol.
pub const HID_DESC_LENGTH: u16 = 30;

/// The only protocol version this driver implements (BCD 1.00).
pub const HID_BCD_VERSION: u16 = 0x0100;

/// The 30-byte HID descriptor, exactly as it appears on the wire.
///
/// Every field is little-endian and the struct is `Unaligned`, so it is
/// reinterpreted from the read buffer with no copy and no hand-written
/// offset arithmetic — the `zerocopy` posture the tree uses wherever bytes
/// become a type.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, KnownLayout, Immutable, Unaligned,
)]
pub struct HidDescriptor {
    /// Length of this descriptor: always [`HID_DESC_LENGTH`].
    pub w_hid_desc_length: U16,
    /// Protocol version in BCD: always [`HID_BCD_VERSION`].
    pub bcd_version: U16,
    /// Length of the report descriptor.
    pub w_report_desc_length: U16,
    /// Register the report descriptor is fetched from.
    pub w_report_desc_register: U16,
    /// Register input reports are read from.
    pub w_input_register: U16,
    /// The largest input report, including its 2-byte length prefix.
    pub w_max_input_length: U16,
    /// Register output reports are written to.
    pub w_output_register: U16,
    /// The largest output report.
    pub w_max_output_length: U16,
    /// Register commands are written to.
    pub w_command_register: U16,
    /// Register command data is exchanged through.
    pub w_data_register: U16,
    /// USB-style vendor id.
    pub w_vendor_id: U16,
    /// USB-style product id.
    pub w_product_id: U16,
    /// Device version.
    pub w_version_id: U16,
    /// Reserved; the spec requires zero but devices are not trusted to
    /// honour it, so it is carried and ignored rather than validated.
    pub reserved: U32,
}

/// Why a descriptor was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescriptorError {
    /// Fewer than 30 bytes were read.
    Short {
        /// How many bytes arrived.
        got: usize,
    },
    /// `wHIDDescLength` was not 30.
    ///
    /// The first field, and the one that catches the common failure:
    /// reading the descriptor register before the device has powered up
    /// returns zeroes or `0xFF`s, and this is what refuses them.
    BadLength(u16),
    /// `bcdVersion` was not 1.00.
    BadVersion(u16),
    /// `wMaxInputLength` was smaller than the 2-byte length prefix every
    /// input report carries, so no report could ever be valid.
    MaxInputTooSmall(u16),
    /// `wReportDescLength` was zero: there is no report descriptor to
    /// parse, so nothing could ever be decoded.
    NoReportDescriptor,
}

impl HidDescriptor {
    /// Reinterpret and validate 30 bytes read from the descriptor
    /// register.
    ///
    /// Validation is not optional and not deferred: every later register
    /// access is taken from these fields, so accepting a malformed
    /// descriptor means writing commands to an address the device never
    /// published.
    pub fn parse(bytes: &[u8]) -> Result<Self, DescriptorError> {
        let d = Self::read_from_prefix(bytes)
            .map(|(d, _)| d)
            .map_err(|_| DescriptorError::Short { got: bytes.len() })?;

        if d.w_hid_desc_length.get() != HID_DESC_LENGTH {
            return Err(DescriptorError::BadLength(d.w_hid_desc_length.get()));
        }
        if d.bcd_version.get() != HID_BCD_VERSION {
            return Err(DescriptorError::BadVersion(d.bcd_version.get()));
        }
        if d.w_max_input_length.get() < 2 {
            return Err(DescriptorError::MaxInputTooSmall(
                d.w_max_input_length.get(),
            ));
        }
        if d.w_report_desc_length.get() == 0 {
            return Err(DescriptorError::NoReportDescriptor);
        }
        Ok(d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A well-formed descriptor, as a real touchpad publishes one.
    pub(crate) fn good() -> [u8; 30] {
        let mut b = [0u8; 30];
        b[0..2].copy_from_slice(&30u16.to_le_bytes()); // wHIDDescLength
        b[2..4].copy_from_slice(&0x0100u16.to_le_bytes()); // bcdVersion
        b[4..6].copy_from_slice(&405u16.to_le_bytes()); // wReportDescLength
        b[6..8].copy_from_slice(&0x0002u16.to_le_bytes()); // wReportDescRegister
        b[8..10].copy_from_slice(&0x0003u16.to_le_bytes()); // wInputRegister
        b[10..12].copy_from_slice(&0x0040u16.to_le_bytes()); // wMaxInputLength
        b[12..14].copy_from_slice(&0x0004u16.to_le_bytes()); // wOutputRegister
        b[14..16].copy_from_slice(&0x0040u16.to_le_bytes()); // wMaxOutputLength
        b[16..18].copy_from_slice(&0x0005u16.to_le_bytes()); // wCommandRegister
        b[18..20].copy_from_slice(&0x0006u16.to_le_bytes()); // wDataRegister
        b[20..22].copy_from_slice(&0x06cbu16.to_le_bytes()); // vendor (Synaptics)
        b[22..24].copy_from_slice(&0x2ba6u16.to_le_bytes()); // product
        b[24..26].copy_from_slice(&0x0001u16.to_le_bytes()); // version
        b
    }

    #[test]
    fn the_wire_layout_is_exactly_thirty_bytes() {
        assert_eq!(core::mem::size_of::<HidDescriptor>(), 30);
        assert_eq!(
            HID_DESC_LENGTH as usize,
            core::mem::size_of::<HidDescriptor>()
        );
    }

    #[test]
    fn a_real_descriptor_decodes_field_for_field() {
        let d = HidDescriptor::parse(&good()).expect("valid");
        assert_eq!(d.w_report_desc_length.get(), 405);
        assert_eq!(d.w_report_desc_register.get(), 0x0002);
        assert_eq!(d.w_input_register.get(), 0x0003);
        assert_eq!(d.w_max_input_length.get(), 0x0040);
        assert_eq!(d.w_command_register.get(), 0x0005);
        assert_eq!(d.w_data_register.get(), 0x0006);
        assert_eq!(d.w_vendor_id.get(), 0x06cb);
        assert_eq!(d.w_product_id.get(), 0x2ba6);
    }

    #[test]
    fn a_short_read_is_refused_rather_than_read_past() {
        for n in 0..30 {
            assert_eq!(
                HidDescriptor::parse(&good()[..n]),
                Err(DescriptorError::Short { got: n }),
                "{n} bytes"
            );
        }
        // Extra bytes are fine: the prefix is what matters.
        let mut long = [0u8; 40];
        long[..30].copy_from_slice(&good());
        assert!(HidDescriptor::parse(&long).is_ok());
    }

    #[test]
    fn the_two_failures_a_powered_down_device_produces_are_both_caught() {
        // Reading before power-up returns all-zeroes...
        assert_eq!(
            HidDescriptor::parse(&[0u8; 30]),
            Err(DescriptorError::BadLength(0))
        );
        // ...or all-ones, when the device is not driving the bus at all.
        assert_eq!(
            HidDescriptor::parse(&[0xffu8; 30]),
            Err(DescriptorError::BadLength(0xffff))
        );
    }

    #[test]
    fn each_malformed_field_is_named_separately() {
        let mut b = good();
        b[2..4].copy_from_slice(&0x0200u16.to_le_bytes());
        assert_eq!(
            HidDescriptor::parse(&b),
            Err(DescriptorError::BadVersion(0x0200))
        );

        let mut b = good();
        b[10..12].copy_from_slice(&1u16.to_le_bytes());
        assert_eq!(
            HidDescriptor::parse(&b),
            Err(DescriptorError::MaxInputTooSmall(1))
        );

        let mut b = good();
        b[4..6].copy_from_slice(&0u16.to_le_bytes());
        assert_eq!(
            HidDescriptor::parse(&b),
            Err(DescriptorError::NoReportDescriptor)
        );
    }
}
