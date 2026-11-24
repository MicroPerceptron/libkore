//! **The one selection point** (#809 §G rule 7): the probed PCI identity
//! becomes a generation here, once, at construction — never `cfg`, never on
//! a datapath. No generation identifier appears outside this module and the
//! generated `regs/<gen>.rs` it names.

/// Realtek's PCI vendor id.
pub const PCI_VENDOR_REALTEK: u16 = 0x10ec;
/// The RTL8922AE — Realtek's BE-generation Wi-Fi 7 PCIe function, read off
/// Gigatron (#388 D0, 2026-09-07): `56:00.0 [10ec:8922] (rev 01)`.
pub const PCI_DEVICE_RTL8922AE: u16 = 0x8922;

/// A part this core drives.
///
/// One variant today. The row's selector names the exact identity, so the
/// broker never grants a function this core would refuse; the enum exists
/// so the second BE part (the RTL8922DE upstream already lists beside it)
/// is a variant and a `regs/` sibling, not a branch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Product {
    /// The RTL8922AE: Wi-Fi 7, 2×2, tri-band, BAR2 registers, one MSI
    /// message, firmware `rtw89/rtw8922a_fw.bin` (format ≤ 4).
    #[default]
    Rtl8922ae,
}

impl Product {
    /// The generation a probed identity selects, or `None` for a part this
    /// core does not drive.
    pub const fn from_pci(vendor: u16, device: u16) -> Option<Self> {
        match (vendor, device) {
            (PCI_VENDOR_REALTEK, PCI_DEVICE_RTL8922AE) => Some(Self::Rtl8922ae),
            _ => None,
        }
    }

    /// The part's name — for a status line, never a lookup key.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Rtl8922ae => "RTL8922AE",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_audited_identity_selects_a_generation() {
        assert_eq!(
            Product::from_pci(PCI_VENDOR_REALTEK, PCI_DEVICE_RTL8922AE),
            Some(Product::Rtl8922ae)
        );
        assert_eq!(
            Product::from_pci(PCI_VENDOR_REALTEK, 0x8852),
            None,
            "an AX-generation sibling"
        );
        assert_eq!(
            Product::from_pci(PCI_VENDOR_REALTEK, 0x8168),
            None,
            "a Realtek wired NIC"
        );
        assert_eq!(Product::from_pci(0x8086, 0x8922), None, "another vendor");
        assert_eq!(Product::default().name(), "RTL8922AE");
    }
}
