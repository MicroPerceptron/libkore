//! **The one selection point** (#809 §G rule 7): the probed PCI identity
//! becomes a generation here, once, at construction — never `cfg`, never on
//! a transmit or receive path. No generation identifier appears outside
//! this module and the generated `regs/<gen>.rs` it names.

/// Intel's PCI vendor id.
pub const PCI_VENDOR_INTEL: u16 = 0x8086;
/// The 82574L — QEMU's `-device e1000e`.
pub const PCI_DEVICE_82574L: u16 = 0x10d3;

/// A part this core drives.
///
/// One variant today. The row's selector names the exact identity, so the
/// broker never grants a function this core would refuse; the enum exists
/// so the second part is a variant and a `regs/` sibling, not a branch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Product {
    /// The 82574L, single-port GbE, PCIe, two queues, five MSI-X vectors.
    #[default]
    I82574L,
}

impl Product {
    /// The generation a probed identity selects, or `None` for a part this
    /// core does not drive.
    pub const fn from_pci(vendor: u16, device: u16) -> Option<Self> {
        match (vendor, device) {
            (PCI_VENDOR_INTEL, PCI_DEVICE_82574L) => Some(Self::I82574L),
            _ => None,
        }
    }

    /// The datasheet's name — for a status line, never a lookup key.
    pub const fn name(self) -> &'static str {
        match self {
            Self::I82574L => "82574L",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_audited_identity_selects_a_generation() {
        assert_eq!(
            Product::from_pci(PCI_VENDOR_INTEL, PCI_DEVICE_82574L),
            Some(Product::I82574L)
        );
        assert_eq!(
            Product::from_pci(PCI_VENDOR_INTEL, 0x100e),
            None,
            "the 82540EM"
        );
        assert_eq!(Product::from_pci(0x10ec, 0x8168), None, "another vendor");
        assert_eq!(Product::default().name(), "82574L");
    }
}
