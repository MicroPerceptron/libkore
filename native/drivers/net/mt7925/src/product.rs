//! **Which part this is**, decided from the claimed function's PCI identity.
//!
//! One variant today. It exists rather than being inlined as an equality
//! check because `res/MT7925_EVIDENCE.md` §1 records the one fact that makes
//! identity awkward on this family: the subsystem id is **MediaTek's own**
//! (`14c3:0717`), not an OEM's, so there is nothing to key a variant on but
//! the device id — and upstream's own `mt7925e` carries three of them
//! (`0x0717`, `0x7925`-era parts and the MT7927/MT7928 siblings it rebrands
//! at `mt7925/pci.c:683,732,737`). A reader reaching for "which part" should
//! find a type, not a literal.

/// The MediaTek vendor id.
pub const VENDOR_MEDIATEK: u16 = 0x14c3;

/// The MT7925's device id.
///
/// The row's selector matches on this **exactly**, as the Realtek and Intel
/// parts' do. A class-and-subclass selector would also admit every other
/// `02:80` function on the bus — which on this machine is nothing today, and
/// is precisely the assumption that should not be baked into a claim.
pub const DEVICE_MT7925: u16 = 0x0717;

/// A part this driver core knows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Product {
    /// MT7925 (RZ717) — `14c3:0717`, the part in the Framework Laptop 13.
    #[default]
    Mt7925,
}

impl Product {
    /// The part a claimed PCI function is, or `None` for a function this
    /// core does not drive.
    ///
    /// Deliberately narrow. The MT7927 and MT7928 share most of this
    /// driver upstream but take a **different fixed map**
    /// (`mt7928_fixed_map`, `mt7925/pci.c:235`), so admitting them here
    /// without transcribing that table would bind a part whose registers
    /// this core resolves to the wrong offsets.
    pub const fn from_pci(vendor: u16, device: u16) -> Option<Self> {
        match (vendor, device) {
            (VENDOR_MEDIATEK, DEVICE_MT7925) => Some(Self::Mt7925),
            _ => None,
        }
    }

    /// The identity [`MT_HW_CHIPID`](crate::regs::mt7925::MT_HW_CHIPID) must
    /// read back for this part.
    pub const fn chip_id(self) -> u32 {
        match self {
            Self::Mt7925 => crate::d0::CHIP_ID_MT7925,
        }
    }
}
