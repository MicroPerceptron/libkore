//! Device ABI — the userland *driver* contract, and the priority surface of
//! this crate.
//!
//! kore runs drivers in userland with the IOMMU mandatory and **no physical
//! address ever crossing the boundary** (res/DEVICE_SERVICE.md §1.1). Every device-exposed
//! address here is an [`IovaAddr`] through the device's own IOMMU domain —
//! the "no phys in userland" rule is a *type* in this module, not a
//! convention. A driver maps a BAR, maps DMA buffers, and programs MSI-X
//! entirely in IOVA terms; the kernel mediates every privileged step and
//! hands back only sealed device-domain addresses.
//!
//! These operations are realized kernel-side as session invocations
//! ([`crate::session`]) over the device capability and its IOMMU binding;
//! each result lands in a caller-named arena slot.  There is no separate
//! device op-code: a mapping is a *derived cap*, so the verbs are
//! `Derive` flavors — [`BarReq`](crate::session::BarReq) (claim a BAR),
//! [`ShareReq`](crate::session::ShareReq) (install it into a space),
//! [`DeviceShareReq`](crate::session::DeviceShareReq) (DMA into the
//! device domain), [`IrqReq`](crate::session::IrqReq) (an MSI/MSI-X
//! route) — and teardown is an ordinary `Revoke` of the leaf.

mod i2c_admission;
mod census;
pub use i2c_admission::*;
pub use census::*;

/// An I/O virtual address — a device-domain address translated by the
/// IOMMU. **Never** a physical address: userland holds only this, so it can
/// neither read nor corrupt host physical memory it was not granted.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct IovaAddr(pub u64);

impl IovaAddr {
    /// Raw device-domain address value.
    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Advance within the same device-domain address space.
    #[inline]
    pub const fn checked_add(self, offset: u64) -> Option<Self> {
        match self.0.checked_add(offset) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }
}

/// A contiguous IOVA window — base + length, the result of a map.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct IovaRange {
    pub base: IovaAddr,
    pub len: u64,
}

/// Which species of device an inventory index names.
///
/// The two are disjoint authorities — a PCI function grants BARs and
/// MSI(-X) routes, a firmware-described one grants I/O ports and a legacy
/// pin — so the broker must know which it is *before* it derives, not
/// after: a claimant handed the wrong species refuses at its possession
/// verb, which is a bind failure twenty lines into hardware bring-up
/// instead of a refusal at the claim.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DeviceSpecies {
    /// A PCI(e) function.
    Pci = 0,
    /// A firmware-described non-PCI device (#417), selected either by a
    /// compressed ACPI id or by a topology-derived platform role.
    Platform = 1,
}

/// How many PCIe root complexes the *static* boot topology provisions for —
/// broker manifest rows, per-driver faces, and founder shelf/seed arrays are
/// all sized by this (#572).
///
/// Sized to the production-PC ceiling, not the test fleet: consumer boards
/// expose one host bridge; HEDT/workstation firmware exposes one per IIO
/// stack / IOD quadrant (Xeon-W `PC0x`, Threadripper PRO) — up to eight.
/// QEMU's NUMA profiles boot two, the Snapdragon X1E three.  **Discovery is
/// not bounded by this**: the catalog records every root firmware describes;
/// roots past this many are enumerated but get no broker row (their devices
/// stay unclaimable, loudly).  Dual-socket servers (16–32 bridges) raise the
/// const — a one-line bump plus the founder slot-bank review documented at
/// its banks.
pub const MAX_ROOT_COMPLEXES: usize = 8;

impl DeviceSpecies {
    /// Decode the wire byte; `None` for an unknown species (fail-closed —
    /// a broker that cannot name the species must not derive).
    #[inline]
    pub const fn from_bits(bits: u8) -> Option<Self> {
        match bits {
            0 => Some(Self::Pci),
            1 => Some(Self::Platform),
            _ => None,
        }
    }
}

/// What the enumerating side already established about one inventory
/// entry — **data, never authority**.
///
/// This is the whole surface a device broker matches selectors against, and
/// it is deliberately the *settled* facts only: what firmware described and
/// what the kernel's own enumeration proved (a live IOMMU binding, an
/// eligible remapped MSI-family/source route, resolved MSI-X table size, and
/// user-mappable BAR membership). It carries nothing that requires
/// touching the device after enumeration.
///
/// **Liveness is not in here on purpose.** Whether a controller's registers
/// decode, and whether anything is plugged into it, are class-specific
/// questions answerable only by reading class-specific registers — which is
/// the driver's job, after it holds the device. Putting a "live" bit here
/// would recreate, one layer out, exactly the kernel-side xHCI probe this
/// inventory exists to delete.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DeviceRecord {
    /// Which species [`index`](Self::index) names.
    pub species: DeviceSpecies,
    /// The entry's index in the inventory — stable for the life of the
    /// boot, and what a claim derives against.
    pub index: u32,
    /// PCI vendor id, or the low half of a platform device's compressed ACPI
    /// id (`0` for topology-selected roles).
    pub vendor: u16,
    /// PCI device id, or the high half of the compressed ACPI id (`0` for a
    /// topology-selected role).
    pub device: u16,
    /// PCI base class, or a `PLATFORM_CLASS_*` role for a platform device.
    pub class: u8,
    /// PCI subclass (`0` for a platform device).
    pub subclass: u8,
    /// PCI programming interface (`0` for a platform device).
    pub progif: u8,
    /// Bus/device/function packed as `bus << 8 | device << 3 | function`,
    /// or the firmware interrupt line (GSI) for a platform device.
    pub locator: u32,
    /// PCI segment group (`0` for a platform device).
    pub segment: u16,
    /// The settled facts a selector may demand — see the `HAS_*` consts.
    pub flags: u8,
    /// Resolved MSI-X table entries. Zero means no valid kernel-mediated table;
    /// claim-time block geometry must fit entirely within this bound.
    pub msix_table_size: u16,
    /// The **diagnostic word** — per-function geometry facts a claimant
    /// can fold into its readiness note when realization refuses, so a
    /// serial-less machine names the mechanism in one shot: `[5:0]`
    /// catalog-present BAR bitmap, `[11:6]` user-mappable BAR bitmap,
    /// `[14:12]` MSI-X table BIR (`7` = no MSI-X), `[26:15]` table
    /// offset >> 12, `[30:27]` reserved for a PM-state sample (`0xF` = not
    /// sampled; inventory reads never touch PMCSR), `[31]` firmware-correlated
    /// PCI I2C-HID controller. Pure description, zero authority; platform
    /// I2C devices carry their controller model here instead.
    pub diag: u32,
    /// PCI revision id (`0` for platform devices) — the `si_stepping`
    /// scalar accelerator firmware validates its silicon against.
    pub revision: u8,
}

impl DeviceRecord {
    /// PCI MSI-X Message Control encodes `table_size - 1` in eleven bits.
    pub const MAX_MSIX_TABLE_ENTRIES: u16 = 2_048;

    /// Platform I2C controller model encoded in [`Self::diag`].
    pub const PLATFORM_I2C_DESIGNWARE: u32 = 1;
    /// Qualcomm QUP/GENI I2C controller model encoded in [`Self::diag`].
    pub const PLATFORM_I2C_GENI: u32 = 2;

    /// The device performs DMA through a live IOMMU binding.
    pub const HAS_BINDING: u8 = 1 << 0;
    /// The device can mint an eligible **remapped** MSI-family route
    /// (MSI-X preferred, ordinary MSI admitted) — the only interrupt
    /// posture a userland driver is offered.
    pub const HAS_REMAPPED_MSI: u8 = 1 << 1;
    /// BAR0 exists and has a user-mappable range.
    pub const HAS_BAR0: u8 = 1 << 2;
    /// Firmware resolved the device's controller interrupt line. For PCI
    /// I2C-HID this is the controller's INTx route, never a child GPIO.
    pub const HAS_LINE: u8 = 1 << 3;
    /// Compatibility guard for brokers predating positive endpoint admission.
    /// It remains set on every withheld PCI function so an older broker also
    /// fails closed.  New brokers require [`DRIVER_ENDPOINT`](Self::DRIVER_ENDPOINT)
    /// instead of inferring authority from this negative fact.
    pub const NOT_DRIVER_ENDPOINT: u8 = 1 << 4;
    /// The kernel's immutable admission policy positively classified this
    /// function as an assignable driver endpoint.  Absence fails closed: a
    /// display function whose firmware ownership is unresolved is not made
    /// claimable merely because it is not a bridge.
    pub const DRIVER_ENDPOINT: u8 = 1 << 5;
    /// The device can mint an eligible remapped **MSI-X** route. This is
    /// separate from [`HAS_REMAPPED_MSI`](Self::HAS_REMAPPED_MSI): the latter
    /// also admits ordinary MSI. Exact table-entry geometry rides
    /// [`msix_table_size`](Self::msix_table_size), not another inferred bit.
    pub const HAS_REMAPPED_MSIX: u8 = 1 << 6;
    /// The function is PCI fabric (a host or PCI-to-PCI bridge), never a
    /// driver's endpoint.  Kept distinct from `NOT_DRIVER_ENDPOINT`: a
    /// firmware-owned or ownership-ambiguous display function is withheld but
    /// is not thereby a bridge.
    pub const IS_BRIDGE: u8 = 1 << 7;

    /// [`Self::diag`] bit 31: the initialized firmware namespace correlated
    /// at least one HID child with this exact enumerated PCI function.
    pub const PCI_DIAG_I2C_HID: u32 = 1 << 31;

    /// An ordinary ACPI-id-selected platform row (i8042 today).
    pub const PLATFORM_CLASS_HID: u8 = 0;
    /// A platform MMIO I2C controller correlated with at least one HID child.
    pub const PLATFORM_CLASS_I2C_HID: u8 = 1;
    /// A kernel-owned GPIO community used for pad interrupt demultiplexing.
    pub const PLATFORM_CLASS_GPIO: u8 = 2;

    /// The compressed ACPI id of a platform entry (`vendor | device << 16`
    /// — the halves this record splits it across).
    #[inline]
    pub const fn hid(&self) -> u32 {
        self.vendor as u32 | ((self.device as u32) << 16)
    }

    /// Whether every flag in `required` is set.
    #[inline]
    pub const fn satisfies(&self, required: u8) -> bool {
        self.flags & required == required
    }
}
