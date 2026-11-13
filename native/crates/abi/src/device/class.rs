//! What a PCI function's **class code** says it is to a driver.
//!
//! A function is offered to drivers on positive evidence only. The class
//! code is the one identification every vendor's function carries, so it is
//! the evidence asked first, and it is asked in one direction: a class named
//! here as a peripheral admits a function; no class, and no absence of one,
//! proves a function safe to hand out. Everything this table does not name
//! is [`Platform`](PciClassRole::Platform), the functions a machine keeps
//! for itself: its management engine, flash and SMBus controllers, power
//! and thermal telemetry, shared memory, and whatever a vendor adds next.
//! Those differ on every platform and their classes are mostly the
//! catch-all "other" subclasses, so they are not enumerated: they are what
//! is left. A new or unknown function is therefore kept, never offered.
//!
//! The kernel decides ([`DeviceRecord::DRIVER_ENDPOINT`]); a reader of a
//! census row uses the same table to say *why* a function is withheld.
//! Codes are the PCI-SIG's (PCI Code and ID Assignment Specification,
//! "Class Codes").
//!
//! [`DeviceRecord::DRIVER_ENDPOINT`]: super::DeviceRecord::DRIVER_ENDPOINT

/// What a class code alone admits a function as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PciClassRole {
    /// Bus fabric: a bridge of any kind (`06`), or an IOMMU (`08/06`). It
    /// carries other functions' traffic and no driver takes it.
    Fabric,
    /// A display controller (`03`), or the legacy VGA-compatible spelling
    /// (`00/01`). A peripheral, but one firmware may still be scanning out
    /// of: the kernel also requires proof of its boot ownership.
    Display,
    /// A device with a standard programming model that a driver is written
    /// against, and that a machine keeps running without.
    Peripheral,
    /// Anything else: a function of the platform itself. Not offered.
    Platform,
}

impl PciClassRole {
    /// The role `class`/`subclass`/`progif` admit.
    pub const fn of(class: u8, subclass: u8, progif: u8) -> Self {
        match (class, subclass) {
            (0x06, _) | (0x08, 0x06) => Self::Fabric,
            (0x03, _) | (0x00, 0x01) => Self::Display,
            // Mass storage, network, multimedia (audio, video, telephony),
            // input, wireless and processing accelerators: every subclass
            // is a device someone attaches or a driver brings up, including
            // the vendor-specific "other" ones (a Wi-Fi radio is `02/80`).
            (0x01 | 0x02 | 0x04 | 0x09 | 0x0d | 0x12, _) => Self::Peripheral,
            // A UART with a named register model: 16450 through 16950.
            // Programming interface 0 ("generic XT") names none, and is
            // what a platform stamps on functions that are not UARTs at
            // all (a sensor hub, a management engine's serial redirection).
            (0x07, 0x00) if matches!(progif, 0x01..=0x06) => Self::Peripheral,
            // An SD host controller: the one base-system peripheral that is
            // a slot for removable media rather than part of the machine.
            (0x08, 0x05) => Self::Peripheral,
            // Host controllers of the external peripheral buses: IEEE 1394,
            // USB (every interface, USB4 included), Fibre Channel and
            // InfiniBand. SMBus (`0c/05`), IPMI (`0c/08`), I3C (`0c/0a`)
            // and "other" (`0c/80`: a platform's I2C, SPI and flash
            // controllers) are the machine's own buses.
            (0x0c, 0x00 | 0x03 | 0x04 | 0x07) => Self::Peripheral,
            _ => Self::Platform,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PciClassRole::{self, Display, Fabric, Peripheral, Platform};

    #[test]
    fn a_class_admits_a_peripheral_and_everything_unnamed_is_the_platforms() {
        for (class, subclass, progif, role, what) in [
            (0x06, 0x00, 0x00, Fabric, "host bridge"),
            (0x06, 0x01, 0x00, Fabric, "ISA/LPC bridge"),
            (0x06, 0x04, 0x00, Fabric, "PCI-to-PCI bridge"),
            (0x08, 0x06, 0x00, Fabric, "IOMMU"),
            (0x03, 0x00, 0x00, Display, "VGA controller"),
            (0x03, 0x80, 0x00, Display, "display, other"),
            (0x00, 0x01, 0x00, Display, "legacy VGA-compatible"),
            (0x01, 0x08, 0x02, Peripheral, "NVMe"),
            (0x01, 0x06, 0x01, Peripheral, "AHCI"),
            (0x02, 0x00, 0x00, Peripheral, "Ethernet"),
            (0x02, 0x80, 0x00, Peripheral, "Wi-Fi radio"),
            (0x04, 0x03, 0x00, Peripheral, "HD Audio"),
            (0x09, 0x00, 0x00, Peripheral, "keyboard"),
            (0x0d, 0x11, 0x00, Peripheral, "Bluetooth"),
            (0x12, 0x00, 0x00, Peripheral, "neural accelerator"),
            (0x07, 0x00, 0x02, Peripheral, "16550 UART"),
            (0x08, 0x05, 0x01, Peripheral, "SD host controller"),
            (0x0c, 0x03, 0x30, Peripheral, "xHCI"),
            (0x0c, 0x03, 0x40, Peripheral, "USB4 host interface"),
            // What a machine keeps. The identities are the ones one Lunar
            // Lake laptop and one AMD laptop enumerate.
            (0x05, 0x00, 0x00, Platform, "PMC shared SRAM"),
            (0x07, 0x80, 0x00, Platform, "management engine HECI"),
            (
                0x07,
                0x00,
                0x00,
                Platform,
                "sensor hub stamped generic serial",
            ),
            (
                0x0c,
                0x80,
                0x00,
                Platform,
                "SPI flash or LPSS I2C controller",
            ),
            (0x0c, 0x05, 0x00, Platform, "SMBus"),
            (0x0c, 0x0a, 0x00, Platform, "I3C"),
            (0x11, 0x80, 0x00, Platform, "thermal and power tuning"),
            (0x10, 0x80, 0x00, Platform, "platform security processor"),
            (0x13, 0x00, 0x00, Platform, "instrumentation"),
            (0x08, 0x80, 0x00, Platform, "system peripheral, other"),
            (0x00, 0x00, 0x00, Platform, "unclassified"),
            (0x00, 0xff, 0x00, Platform, "unclassified, vendor"),
            (0xff, 0x00, 0x00, Platform, "unassigned"),
        ] {
            assert_eq!(
                PciClassRole::of(class, subclass, progif),
                role,
                "{what} ({class:02x}/{subclass:02x}.{progif:02x})"
            );
        }
    }

    /// The table only ever admits by naming: no class outside the ones it
    /// names is a peripheral, whatever its subclass or interface.
    #[test]
    fn no_unnamed_class_is_ever_a_peripheral() {
        for class in 0..=u8::MAX {
            if matches!(
                class,
                0x01 | 0x02 | 0x04 | 0x07 | 0x08 | 0x09 | 0x0c | 0x0d | 0x12
            ) {
                continue;
            }
            for subclass in 0..=u8::MAX {
                for progif in [0x00, 0x01, 0x02, 0x30, 0x80, 0xff] {
                    assert_ne!(
                        PciClassRole::of(class, subclass, progif),
                        Peripheral,
                        "{class:02x}/{subclass:02x}.{progif:02x}"
                    );
                }
            }
        }
    }
}
