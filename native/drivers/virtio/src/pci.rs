//! VirtIO 1.3 modern PCI transport over capability-mapped BAR windows.
//!
//! PCI enumeration is deliberately outside this module.  The kernel-side
//! capability walker may lower the vendor capabilities into [`PciLayout`],
//! but this driver accepts only BAR indices and relative offsets.  It never
//! reads PCI configuration space, probes BAR sizes, or observes a physical
//! address.

use kore_driver::seam::Regs64;
use kore_driver::{Regs, RegsNarrow};

use crate::split::QueueMemory;

const COMMON_MIN: u32 = 56;
const ISR_MIN: u32 = 1;
const NOTIFY_MIN: u32 = 2;
const CONSOLE_CONFIG_MIN: u32 = 8;
const MAX_TRACKED_QUEUES: usize = 64;
const NO_VECTOR: u16 = u16::MAX;

const C_DEVICE_FEATURE_SELECT: usize = 0;
const C_DEVICE_FEATURE: usize = 4;
const C_DRIVER_FEATURE_SELECT: usize = 8;
const C_DRIVER_FEATURE: usize = 12;
const C_MSIX_CONFIG: usize = 16;
const C_DEVICE_STATUS: usize = 20;
const C_CONFIG_GENERATION: usize = 21;
const C_QUEUE_SELECT: usize = 22;
const C_QUEUE_SIZE: usize = 24;
const C_QUEUE_MSIX_VECTOR: usize = 26;
const C_QUEUE_ENABLE: usize = 28;
const C_QUEUE_NOTIFY_OFF: usize = 30;
const C_QUEUE_DESC: usize = 32;
const C_QUEUE_DRIVER: usize = 40;
const C_QUEUE_DEVICE: usize = 48;

const STATUS_ACKNOWLEDGE: u8 = 1;
const STATUS_DRIVER: u8 = 2;
const STATUS_DRIVER_OK: u8 = 4;
const STATUS_FEATURES_OK: u8 = 8;
const STATUS_FAILED: u8 = 128;

pub const F_RING_EVENT_IDX: u64 = 1 << 29;
pub const F_VERSION_1: u64 = 1 << 32;
pub const F_ACCESS_PLATFORM: u64 = 1 << 33;
pub const F_CONSOLE_MULTIPORT: u64 = 1 << 1;

/// One redacted VirtIO PCI vendor-capability region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PciRegion {
    pub bar: u8,
    pub offset: u32,
    pub length: u32,
}

/// Parsed modern VirtIO vendor-capability layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PciLayout {
    pub common: PciRegion,
    pub notify: PciRegion,
    pub isr: PciRegion,
    pub device: PciRegion,
    pub notify_off_multiplier: u32,
}

impl PciLayout {
    /// Assemble the four required modern regions from the kernel's redacted
    /// capability reports. Duplicate required types and malformed multiplier
    /// placement are refused; optional PCI/shared-memory rows are ignored by
    /// this console transport.
    pub fn from_reports(
        reports: &[kore_abi::session::VirtioPciCapReport],
    ) -> Result<Self, TransportError> {
        let mut common = None;
        let mut notify = None;
        let mut isr = None;
        let mut device = None;
        let mut notify_off_multiplier = 0;
        for report in reports {
            if report.bar >= 6
                || report.length == 0
                || report.offset.checked_add(report.length).is_none()
            {
                return Err(TransportError::InvalidRegion);
            }
            let region = PciRegion {
                bar: report.bar,
                offset: report.offset,
                length: report.length,
            };
            match report.cfg_type {
                1 if common.replace(region).is_none() && report.notify_multiplier == 0 => {}
                2 if notify.replace(region).is_none() => {
                    notify_off_multiplier = report.notify_multiplier;
                }
                3 if isr.replace(region).is_none() && report.notify_multiplier == 0 => {}
                4 if device.replace(region).is_none() && report.notify_multiplier == 0 => {}
                5 | 8 if report.notify_multiplier == 0 => {}
                _ => return Err(TransportError::InvalidRegion),
            }
        }
        Ok(Self {
            common: common.ok_or(TransportError::InvalidRegion)?,
            notify: notify.ok_or(TransportError::InvalidRegion)?,
            isr: isr.ok_or(TransportError::InvalidRegion)?,
            device: device.ok_or(TransportError::InvalidRegion)?,
            notify_off_multiplier,
        })
    }
}

/// Affine bounded ownership of the already-mapped BARs used by a transport.
/// Empty slots are explicit; no allocator or global registry participates.
pub struct MappedBars<M> {
    bars: [Option<M>; 6],
    lengths: [usize; 6],
}

impl<M> MappedBars<M> {
    pub const fn new() -> Self {
        Self {
            bars: [const { None }; 6],
            lengths: [0; 6],
        }
    }

    pub fn insert(&mut self, index: u8, regs: M, len: usize) -> Result<(), M> {
        let Some(slot) = self.bars.get_mut(index as usize) else {
            return Err(regs);
        };
        if slot.is_some() {
            return Err(regs);
        }
        *slot = Some(regs);
        self.lengths[index as usize] = len;
        Ok(())
    }

    pub fn get(&self, index: u8) -> Option<&M> {
        self.bars.get(index as usize)?.as_ref()
    }

    pub fn take(&mut self, index: u8) -> Option<M> {
        self.bars.get_mut(index as usize)?.take()
    }
}

impl<M> Default for MappedBars<M> {
    fn default() -> Self {
        Self::new()
    }
}

/// Modern transport refusal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransportError {
    MissingBar(u8),
    InvalidRegion,
    ResetTimeout,
    MissingFeatures {
        required: u64,
        offered: u64,
    },
    FeaturesRejected,
    QueueOutOfRange(u16),
    QueueUnavailable(u16),
    QueueTooSmall {
        queue: u16,
        requested: u16,
        maximum: u16,
    },
    QueueAlreadyEnabled(u16),
    QueueMemoryMisaligned(u16),
    VectorRejected {
        queue: u16,
        vector: u16,
    },
    NotifyOutOfRange(u16),
    ConfigUnstable,
    DeviceFailed,
}

/// Transport operations consumed by a device-class engine.
pub trait VirtioTransport {
    fn reset(&mut self) -> Result<u32, TransportError>;
    fn negotiate(&mut self, supported: u64, required: u64) -> Result<u64, TransportError>;
    fn set_config_vector(&mut self, vector: u16) -> Result<(), TransportError>;
    fn setup_queue(
        &mut self,
        queue: u16,
        memory: QueueMemory,
        vector: u16,
    ) -> Result<(), TransportError>;
    fn finish_init(&mut self) -> Result<(), TransportError>;
    fn notify_queue(&self, queue: u16) -> Result<(), TransportError>;
    fn ack_interrupt(&self) -> Result<u8, TransportError>;
    fn console_max_ports(&self) -> Result<u32, TransportError>;
    fn fail(&mut self) -> Result<(), TransportError>;
    fn epoch(&self) -> u32;
}

/// Modern PCI transport borrowing the owning driver's mapped BAR set.
pub struct ModernPciTransport<M: Regs + RegsNarrow + Regs64> {
    bars: MappedBars<M>,
    layout: PciLayout,
    notify_offsets: [u32; MAX_TRACKED_QUEUES],
    queue_configured: [bool; MAX_TRACKED_QUEUES],
    negotiated: u64,
    epoch: u32,
}

impl<M: Regs + RegsNarrow + Regs64> ModernPciTransport<M> {
    pub fn new(bars: MappedBars<M>, layout: PciLayout) -> Result<Self, TransportError> {
        let transport = Self {
            bars,
            layout,
            notify_offsets: [0; MAX_TRACKED_QUEUES],
            queue_configured: [false; MAX_TRACKED_QUEUES],
            negotiated: 0,
            epoch: 0,
        };
        transport.validate_region(layout.common, COMMON_MIN, 8)?;
        transport.validate_region(layout.notify, NOTIFY_MIN, 2)?;
        transport.validate_region(layout.isr, ISR_MIN, 1)?;
        transport.validate_region(layout.device, CONSOLE_CONFIG_MIN, 4)?;
        if !layout.notify_off_multiplier.is_multiple_of(2) {
            return Err(TransportError::InvalidRegion);
        }
        Ok(transport)
    }

    pub const fn negotiated_features(&self) -> u64 {
        self.negotiated
    }

    /// Return every affine BAR owner after the caller has reset the device and
    /// torn down its queues.
    pub fn into_bars(self) -> MappedBars<M> {
        self.bars
    }

    fn bar(&self, index: u8) -> Result<&M, TransportError> {
        self.bars
            .get(index)
            .ok_or(TransportError::MissingBar(index))
    }

    fn validate_region(
        &self,
        region: PciRegion,
        minimum: u32,
        alignment: u32,
    ) -> Result<(), TransportError> {
        if region.length < minimum || !region.offset.is_multiple_of(alignment) {
            return Err(TransportError::InvalidRegion);
        }
        let end = region
            .offset
            .checked_add(region.length)
            .ok_or(TransportError::InvalidRegion)?;
        self.bar(region.bar)?;
        if end as usize > self.bars.lengths[region.bar as usize] {
            return Err(TransportError::InvalidRegion);
        }
        Ok(())
    }

    fn offset(region: PciRegion, field: usize, width: usize) -> Result<usize, TransportError> {
        let end = field
            .checked_add(width)
            .ok_or(TransportError::InvalidRegion)?;
        if end > region.length as usize {
            return Err(TransportError::InvalidRegion);
        }
        (region.offset as usize)
            .checked_add(field)
            .ok_or(TransportError::InvalidRegion)
    }

    fn common_read8(&self, field: usize) -> Result<u8, TransportError> {
        Ok(self
            .bar(self.layout.common.bar)?
            .read8(Self::offset(self.layout.common, field, 1)?))
    }

    fn common_read16(&self, field: usize) -> Result<u16, TransportError> {
        Ok(self
            .bar(self.layout.common.bar)?
            .read16(Self::offset(self.layout.common, field, 2)?))
    }

    fn common_read32(&self, field: usize) -> Result<u32, TransportError> {
        Ok(self
            .bar(self.layout.common.bar)?
            .read32(Self::offset(self.layout.common, field, 4)?))
    }

    fn common_write8(&self, field: usize, value: u8) -> Result<(), TransportError> {
        Ok(self
            .bar(self.layout.common.bar)?
            .write8(Self::offset(self.layout.common, field, 1)?, value))
    }

    fn common_write16(&self, field: usize, value: u16) -> Result<(), TransportError> {
        Ok(self
            .bar(self.layout.common.bar)?
            .write16(Self::offset(self.layout.common, field, 2)?, value))
    }

    fn common_write32(&self, field: usize, value: u32) -> Result<(), TransportError> {
        Ok(self
            .bar(self.layout.common.bar)?
            .write32(Self::offset(self.layout.common, field, 4)?, value))
    }

    fn common_write64(&self, field: usize, value: u64) -> Result<(), TransportError> {
        Ok(self
            .bar(self.layout.common.bar)?
            .write64_once(Self::offset(self.layout.common, field, 8)?, value))
    }

    fn status(&self) -> Result<u8, TransportError> {
        self.common_read8(C_DEVICE_STATUS)
    }

    fn set_status(&self, status: u8) -> Result<(), TransportError> {
        self.common_write8(C_DEVICE_STATUS, status)
    }

    fn add_status(&self, bit: u8) -> Result<(), TransportError> {
        let current = self.status()?;
        if current & STATUS_FAILED != 0 {
            return Err(TransportError::DeviceFailed);
        }
        self.set_status(current | bit)
    }
}

impl<M: Regs + RegsNarrow + Regs64> VirtioTransport for ModernPciTransport<M> {
    fn reset(&mut self) -> Result<u32, TransportError> {
        self.set_status(0)?;
        for _ in 0..64 {
            if self.status()? == 0 {
                self.epoch = self.epoch.wrapping_add(1);
                if self.epoch == 0 {
                    self.epoch = 1;
                }
                self.negotiated = 0;
                self.notify_offsets = [0; MAX_TRACKED_QUEUES];
                self.queue_configured = [false; MAX_TRACKED_QUEUES];
                return Ok(self.epoch);
            }
            core::hint::spin_loop();
        }
        Err(TransportError::ResetTimeout)
    }

    fn negotiate(&mut self, supported: u64, required: u64) -> Result<u64, TransportError> {
        self.add_status(STATUS_ACKNOWLEDGE)?;
        self.add_status(STATUS_DRIVER)?;

        self.common_write32(C_DEVICE_FEATURE_SELECT, 0)?;
        let low = u64::from(self.common_read32(C_DEVICE_FEATURE)?);
        self.common_write32(C_DEVICE_FEATURE_SELECT, 1)?;
        let high = u64::from(self.common_read32(C_DEVICE_FEATURE)?);
        let offered = low | (high << 32);
        if offered & required != required || supported & required != required {
            return Err(TransportError::MissingFeatures { required, offered });
        }
        let accepted = offered & supported;
        self.common_write32(C_DRIVER_FEATURE_SELECT, 0)?;
        self.common_write32(C_DRIVER_FEATURE, accepted as u32)?;
        self.common_write32(C_DRIVER_FEATURE_SELECT, 1)?;
        self.common_write32(C_DRIVER_FEATURE, (accepted >> 32) as u32)?;
        self.add_status(STATUS_FEATURES_OK)?;
        if self.status()? & STATUS_FEATURES_OK == 0 {
            return Err(TransportError::FeaturesRejected);
        }
        self.negotiated = accepted;
        Ok(accepted)
    }

    fn set_config_vector(&mut self, vector: u16) -> Result<(), TransportError> {
        self.common_write16(C_MSIX_CONFIG, vector)?;
        let accepted = self.common_read16(C_MSIX_CONFIG)?;
        if accepted == NO_VECTOR || accepted != vector {
            return Err(TransportError::VectorRejected {
                queue: u16::MAX,
                vector,
            });
        }
        Ok(())
    }

    fn setup_queue(
        &mut self,
        queue: u16,
        memory: QueueMemory,
        vector: u16,
    ) -> Result<(), TransportError> {
        let at = usize::from(queue);
        if at >= MAX_TRACKED_QUEUES {
            return Err(TransportError::QueueOutOfRange(queue));
        }
        self.common_write16(C_QUEUE_SELECT, queue)?;
        let maximum = self.common_read16(C_QUEUE_SIZE)?;
        if maximum == 0 {
            return Err(TransportError::QueueUnavailable(queue));
        }
        if memory.size == 0 || !memory.size.is_power_of_two() || memory.size > maximum {
            return Err(TransportError::QueueTooSmall {
                queue,
                requested: memory.size,
                maximum,
            });
        }
        if !memory.descriptors.0.is_multiple_of(16)
            || !memory.driver_area.0.is_multiple_of(2)
            || !memory.device_area.0.is_multiple_of(4)
        {
            return Err(TransportError::QueueMemoryMisaligned(queue));
        }
        if self.common_read16(C_QUEUE_ENABLE)? != 0 {
            return Err(TransportError::QueueAlreadyEnabled(queue));
        }

        self.common_write16(C_QUEUE_SIZE, memory.size)?;
        self.common_write16(C_QUEUE_MSIX_VECTOR, vector)?;
        let accepted = self.common_read16(C_QUEUE_MSIX_VECTOR)?;
        if accepted == NO_VECTOR || accepted != vector {
            return Err(TransportError::VectorRejected { queue, vector });
        }
        self.common_write64(C_QUEUE_DESC, memory.descriptors.0)?;
        self.common_write64(C_QUEUE_DRIVER, memory.driver_area.0)?;
        self.common_write64(C_QUEUE_DEVICE, memory.device_area.0)?;

        let queue_notify_off = u32::from(self.common_read16(C_QUEUE_NOTIFY_OFF)?);
        let relative = queue_notify_off
            .checked_mul(self.layout.notify_off_multiplier)
            .ok_or(TransportError::NotifyOutOfRange(queue))?;
        let notify_end = relative
            .checked_add(2)
            .ok_or(TransportError::NotifyOutOfRange(queue))?;
        if notify_end > self.layout.notify.length || !relative.is_multiple_of(2) {
            return Err(TransportError::NotifyOutOfRange(queue));
        }

        self.common_write16(C_QUEUE_ENABLE, 1)?;
        self.notify_offsets[at] = relative;
        self.queue_configured[at] = true;
        Ok(())
    }

    fn finish_init(&mut self) -> Result<(), TransportError> {
        self.add_status(STATUS_DRIVER_OK)
    }

    fn notify_queue(&self, queue: u16) -> Result<(), TransportError> {
        let at = usize::from(queue);
        if at >= MAX_TRACKED_QUEUES || !self.queue_configured[at] {
            return Err(TransportError::QueueOutOfRange(queue));
        }
        let offset = (self.layout.notify.offset as usize)
            .checked_add(self.notify_offsets[at] as usize)
            .ok_or(TransportError::NotifyOutOfRange(queue))?;
        Ok(self.bar(self.layout.notify.bar)?.write16(offset, queue))
    }

    fn ack_interrupt(&self) -> Result<u8, TransportError> {
        let offset = Self::offset(self.layout.isr, 0, 1)?;
        Ok(self.bar(self.layout.isr.bar)?.read8(offset))
    }

    fn console_max_ports(&self) -> Result<u32, TransportError> {
        for _ in 0..8 {
            let before = self.common_read8(C_CONFIG_GENERATION)?;
            let offset = Self::offset(self.layout.device, 4, 4)?;
            let ports = self.bar(self.layout.device.bar)?.read32(offset);
            let after = self.common_read8(C_CONFIG_GENERATION)?;
            if before == after {
                return Ok(ports);
            }
        }
        Err(TransportError::ConfigUnstable)
    }

    fn fail(&mut self) -> Result<(), TransportError> {
        let status = self.status()?;
        self.set_status(status | STATUS_FAILED)
    }

    fn epoch(&self) -> u32 {
        self.epoch
    }
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};
    use core::cell::{Cell, RefCell};

    use super::*;

    #[derive(Debug, PartialEq, Eq)]
    enum BoundsError {
        Misaligned {
            offset: usize,
            width: usize,
        },
        OutOfRange {
            offset: usize,
            width: usize,
            window: usize,
        },
    }

    struct MockMmio {
        bytes: RefCell<Vec<u8>>,
        feature_select: Cell<u32>,
        driver_select: Cell<u32>,
        offered: u64,
        accepted: Cell<u64>,
    }

    impl MockMmio {
        fn new(len: usize, offered: u64) -> Self {
            Self {
                bytes: RefCell::new(vec![0; len]),
                feature_select: Cell::new(0),
                driver_select: Cell::new(0),
                offered,
                accepted: Cell::new(0),
            }
        }

        fn check(&self, offset: usize, width: usize) -> Result<(), BoundsError> {
            if !offset.is_multiple_of(width) {
                return Err(BoundsError::Misaligned { offset, width });
            }
            if offset
                .checked_add(width)
                .is_none_or(|end| end > self.bytes.borrow().len())
            {
                return Err(BoundsError::OutOfRange {
                    offset,
                    width,
                    window: self.bytes.borrow().len(),
                });
            }
            Ok(())
        }

        fn put16(&self, offset: usize, value: u16) {
            self.bytes.borrow_mut()[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
        }

        fn put32(&self, offset: usize, value: u32) {
            self.bytes.borrow_mut()[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
    }

    impl MockMmio {
        fn len(&self) -> usize {
            self.bytes.borrow().len()
        }

        fn read8(&self, offset: usize) -> Result<u8, BoundsError> {
            self.check(offset, 1)?;
            Ok(self.bytes.borrow()[offset])
        }

        fn read16(&self, offset: usize) -> Result<u16, BoundsError> {
            self.check(offset, 2)?;
            Ok(u16::from_le_bytes(
                self.bytes.borrow()[offset..offset + 2].try_into().unwrap(),
            ))
        }

        fn read32(&self, offset: usize) -> Result<u32, BoundsError> {
            self.check(offset, 4)?;
            if offset == C_DEVICE_FEATURE {
                return Ok(if self.feature_select.get() == 0 {
                    self.offered as u32
                } else {
                    (self.offered >> 32) as u32
                });
            }
            Ok(u32::from_le_bytes(
                self.bytes.borrow()[offset..offset + 4].try_into().unwrap(),
            ))
        }

        fn read64(&self, offset: usize) -> Result<u64, BoundsError> {
            self.check(offset, 8)?;
            Ok(u64::from_le_bytes(
                self.bytes.borrow()[offset..offset + 8].try_into().unwrap(),
            ))
        }

        fn write8(&self, offset: usize, value: u8) -> Result<(), BoundsError> {
            self.check(offset, 1)?;
            self.bytes.borrow_mut()[offset] = value;
            Ok(())
        }

        fn write16(&self, offset: usize, value: u16) -> Result<(), BoundsError> {
            self.check(offset, 2)?;
            self.put16(offset, value);
            Ok(())
        }

        fn write32(&self, offset: usize, value: u32) -> Result<(), BoundsError> {
            self.check(offset, 4)?;
            if offset == C_DEVICE_FEATURE_SELECT {
                self.feature_select.set(value);
            } else if offset == C_DRIVER_FEATURE_SELECT {
                self.driver_select.set(value);
            } else if offset == C_DRIVER_FEATURE {
                let shift = self.driver_select.get() * 32;
                let mask = !(u64::from(u32::MAX) << shift);
                self.accepted
                    .set((self.accepted.get() & mask) | (u64::from(value) << shift));
            }
            self.put32(offset, value);
            Ok(())
        }

        fn write64(&self, offset: usize, value: u64) -> Result<(), BoundsError> {
            self.check(offset, 8)?;
            self.bytes.borrow_mut()[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
            Ok(())
        }
    }

    impl Regs for MockMmio {
        fn read32(&self, off: usize) -> u32 {
            MockMmio::read32(self, off).unwrap()
        }
        fn write32(&self, off: usize, value: u32) {
            MockMmio::write32(self, off, value).unwrap()
        }
    }
    impl RegsNarrow for MockMmio {
        fn read8(&self, off: usize) -> u8 {
            MockMmio::read8(self, off).unwrap()
        }
        fn read16(&self, off: usize) -> u16 {
            MockMmio::read16(self, off).unwrap()
        }
        fn write8(&self, off: usize, value: u8) {
            MockMmio::write8(self, off, value).unwrap()
        }
        fn write16(&self, off: usize, value: u16) {
            MockMmio::write16(self, off, value).unwrap()
        }
    }
    impl Regs64 for MockMmio {
        fn read64_once(&self, off: usize) -> u64 {
            MockMmio::read64(self, off).unwrap()
        }
        fn write64_once(&self, off: usize, value: u64) {
            MockMmio::write64(self, off, value).unwrap()
        }
    }

    fn layout() -> PciLayout {
        PciLayout {
            common: PciRegion {
                bar: 4,
                offset: 0,
                length: 56,
            },
            notify: PciRegion {
                bar: 4,
                offset: 128,
                length: 32,
            },
            isr: PciRegion {
                bar: 4,
                offset: 192,
                length: 1,
            },
            device: PciRegion {
                bar: 4,
                offset: 196,
                length: 12,
            },
            notify_off_multiplier: 4,
        }
    }

    fn bars(regs: MockMmio) -> MappedBars<MockMmio> {
        let mut bars = MappedBars::new();
        let len = regs.len();
        bars.insert(4, regs, len).ok().unwrap();
        bars
    }

    #[test]
    fn validates_redacted_regions_against_mapped_bar() {
        assert!(ModernPciTransport::new(bars(MockMmio::new(256, 0)), layout()).is_ok());
        let mut bad = layout();
        bad.notify.length = 200;
        assert!(matches!(
            ModernPciTransport::new(bars(MockMmio::new(256, 0)), bad),
            Err(TransportError::InvalidRegion)
        ));
        bad = layout();
        bad.common.offset = 4;
        assert!(matches!(
            ModernPciTransport::new(bars(MockMmio::new(256, 0)), bad),
            Err(TransportError::InvalidRegion)
        ));
    }

    #[test]
    fn assembles_unique_required_regions_from_kernel_reports() {
        use kore_abi::session::VirtioPciCapReport;

        let reports = [
            VirtioPciCapReport {
                cfg_type: 1,
                bar: 4,
                id: 0,
                offset: 0,
                length: 56,
                notify_multiplier: 0,
            },
            VirtioPciCapReport {
                cfg_type: 2,
                bar: 4,
                id: 0,
                offset: 128,
                length: 32,
                notify_multiplier: 4,
            },
            VirtioPciCapReport {
                cfg_type: 3,
                bar: 4,
                id: 0,
                offset: 192,
                length: 1,
                notify_multiplier: 0,
            },
            VirtioPciCapReport {
                cfg_type: 4,
                bar: 4,
                id: 0,
                offset: 196,
                length: 12,
                notify_multiplier: 0,
            },
        ];
        assert_eq!(PciLayout::from_reports(&reports), Ok(layout()));
        let mut duplicate = reports;
        duplicate[3].cfg_type = 1;
        assert_eq!(
            PciLayout::from_reports(&duplicate),
            Err(TransportError::InvalidRegion)
        );
    }

    #[test]
    fn negotiates_required_modern_iommu_and_multiport_features() {
        let offered = F_VERSION_1 | F_ACCESS_PLATFORM | F_CONSOLE_MULTIPORT | F_RING_EVENT_IDX;
        let mut transport =
            ModernPciTransport::new(bars(MockMmio::new(256, offered)), layout()).unwrap();
        assert_eq!(transport.reset(), Ok(1));
        let required = F_VERSION_1 | F_ACCESS_PLATFORM | F_CONSOLE_MULTIPORT;
        assert_eq!(transport.negotiate(offered, required), Ok(offered));
        assert_eq!(transport.bar(4).unwrap().accepted.get(), offered);

        let mut missing =
            ModernPciTransport::new(bars(MockMmio::new(256, F_VERSION_1)), layout()).unwrap();
        missing.reset().unwrap();
        assert!(matches!(
            missing.negotiate(offered, required),
            Err(TransportError::MissingFeatures { .. })
        ));
    }

    #[test]
    fn configures_queue_vector_addresses_and_bounded_notification() {
        let offered = F_VERSION_1 | F_ACCESS_PLATFORM | F_CONSOLE_MULTIPORT;
        let regs = MockMmio::new(256, offered);
        regs.put16(C_QUEUE_SIZE, 128);
        regs.put16(C_QUEUE_NOTIFY_OFF, 3);
        regs.put32(196 + 4, 3);
        let mut transport = ModernPciTransport::new(bars(regs), layout()).unwrap();
        transport.reset().unwrap();
        transport
            .negotiate(
                offered,
                F_VERSION_1 | F_ACCESS_PLATFORM | F_CONSOLE_MULTIPORT,
            )
            .unwrap();
        transport.set_config_vector(0).unwrap();
        transport
            .setup_queue(
                4,
                QueueMemory {
                    size: 64,
                    descriptors: kore_abi::device::IovaAddr(0x1000),
                    driver_area: kore_abi::device::IovaAddr(0x2000),
                    device_area: kore_abi::device::IovaAddr(0x3000),
                },
                1,
            )
            .unwrap();
        transport.notify_queue(4).unwrap();
        assert_eq!(transport.bar(4).unwrap().read16(128 + 12), Ok(4));
        assert_eq!(transport.console_max_ports(), Ok(3));
        assert_eq!(transport.reset(), Ok(2));
        assert_eq!(
            transport.notify_queue(4),
            Err(TransportError::QueueOutOfRange(4))
        );
    }

    #[test]
    fn queue_setup_refuses_unaligned_device_addresses() {
        let offered = F_VERSION_1 | F_ACCESS_PLATFORM | F_CONSOLE_MULTIPORT;
        let regs = MockMmio::new(256, offered);
        regs.put16(C_QUEUE_SIZE, 8);
        let mut transport = ModernPciTransport::new(bars(regs), layout()).unwrap();
        transport.reset().unwrap();
        assert_eq!(
            transport.setup_queue(
                4,
                QueueMemory {
                    size: 8,
                    descriptors: kore_abi::device::IovaAddr(0x1008),
                    driver_area: kore_abi::device::IovaAddr(0x2000),
                    device_area: kore_abi::device::IovaAddr(0x3000),
                },
                1,
            ),
            Err(TransportError::QueueMemoryMisaligned(4))
        );
    }
}
