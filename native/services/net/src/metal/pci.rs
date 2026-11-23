//! **The PCI composition** — the 82574L class binding over `kore-e1000e`,
//! and what its bound controller offers the row's generic device host.
//!
//! `kore-e1000e` owns every register sequence and `service` owns every
//! frame decision; this module owns neither. It realizes a claimed function
//! into the driver seams (one BAR, one DMA pool, one MSI-family route) and
//! lends the backend one drive cycle at a time. Everything here is
//! exercised without hardware: the bring-up ladder by `kore-e1000e`'s mock,
//! the class binding by the driver host's own tests.

use kore_driver::host::{
    CapPool, CapRegs, ClassQuiesce, DeviceBinding, DeviceResources, DmaActive, DmaPrepared,
    InterruptBlock, InterruptPolicy, PciBindingContract,
};
use kore_e1000e::{E1000e, E1000eNic, Error as E1000eError, Prepared, Product};

use super::{NetDeviceHost, NicBound};

/// The controller reset, identified, its rings programmed, both engines
/// still disabled. The generic host owns the only transition from here.
pub struct E1000eNetPrepared {
    nic: Prepared<CapPool>,
}

/// The bound owner: the running controller and the capability resource set.
///
/// Unlike the USB host there is no "adaptor" to find: a PCI NIC *is* the
/// adaptor, so a bound device is a live datapath by construction and the
/// founding note it earns is always `0`. A link that is down is reported
/// live through [`NicBackend::link`](kore_driver::net::NicBackend::link),
/// never as a bring-up fact.
pub struct E1000eNetBound {
    nic: E1000e<CapPool>,
    resources: DeviceResources<DmaActive>,
}

/// What the claim decided before spending the pool: the generation the
/// probed identity selected.
///
/// `Default` because [`DeviceBinding::Config`] requires one; the default is
/// the only generation today, and the entry ladder always passes the
/// selection it made rather than relying on that.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct E1000eConfig {
    /// The part the claimed function's PCI identity selected.
    pub product: Product,
}

/// The class declaration: one BAR, one interrupt block, DMA.
///
/// `Required(SINGLE)` with the row's `RouteCapable` posture is the only
/// lawful pair for a claim that must realize a route: the controller's
/// completion path is the MSI-family message, and a NIC that could only
/// poll would congest the whole machine with wakes for the sake of a
/// frame that may never come.
pub struct E1000eNetBinding;

impl PciBindingContract for E1000eNetBinding {
    const BARS: kore_rt::service::PciBarMask = kore_rt::service::PciBarMask::BAR0;
    const INTERRUPTS: InterruptPolicy = InterruptPolicy::Required(InterruptBlock::SINGLE);
}

impl DeviceBinding for E1000eNetBinding {
    type Pci = Self;
    type Config = E1000eConfig;
    type Prepared = E1000eNetPrepared;
    type Bound = E1000eNetBound;
    type Error = E1000eError;

    fn prepare(
        resources: &mut DeviceResources<DmaPrepared>,
        config: Self::Config,
    ) -> Result<Self::Prepared, E1000eError> {
        let regs = resources.bar(0).ok_or(E1000eError::NotPresent)?;
        let pool = resources.pool().ok_or(E1000eError::DmaAlloc)?;
        let nic = Prepared::prepare(
            regs,
            pool,
            config.product,
            regs.len(),
            kore_driver::host::clock_now_ns,
            kore_driver::host::relax,
        )?;
        Ok(E1000eNetPrepared { nic })
    }

    fn start(
        prepared: Self::Prepared,
        resources: DeviceResources<DmaActive>,
    ) -> Result<Self::Bound, E1000eError> {
        let regs = resources.bar(0).ok_or(E1000eError::NotPresent)?;
        let nic = prepared.nic.start(
            regs,
            kore_driver::host::clock_now_ns,
            kore_driver::host::relax,
        )?;
        Ok(E1000eNetBound { nic, resources })
    }

    fn quiesce(bound: &mut Self::Bound) -> Result<ClassQuiesce, E1000eError> {
        let regs = bound.resources.bar(0).ok_or(E1000eError::NotPresent)?;
        bound.nic.quiesce(
            regs,
            kore_driver::host::clock_now_ns,
            kore_driver::host::relax,
        )?;
        Ok(ClassQuiesce::Stopped)
    }

    fn unbind(bound: Self::Bound) -> DeviceResources<DmaActive> {
        let E1000eNetBound { nic, resources } = bound;
        drop(nic);
        resources
    }

    /// The core's rung rides
    /// [`IntakeError::ClassBindRefused`](kore_driver::host::IntakeError::ClassBindRefused)
    /// out to the entry ladder, which notes it as
    /// [`PCI_BRING_UP_BASE`](crate::note::PCI_BRING_UP_BASE)` + rung`.
    fn refusal_code(error: &Self::Error) -> u8 {
        error.rung()
    }
}

impl NicBound for E1000eNetBound {
    type Backend<'a> = E1000eNic<'a, CapRegs, CapPool>;

    /// Two disjoint borrows of one owner: the registers out of the resource
    /// set and the controller state. The backend is a view because the BAR
    /// lives in the resource set, not in the core.
    fn backend(&mut self) -> Option<E1000eNic<'_, CapRegs, CapPool>> {
        let regs = self.resources.bar(0)?;
        Some(E1000eNic::new(regs, &mut self.nic))
    }

    fn resources(&self) -> &DeviceResources<DmaActive> {
        &self.resources
    }

    /// A bound PCI NIC is a live datapath: there is no adaptor to be
    /// missing, and the link is a live report, never a rung.
    fn founding_note(&self) -> u64 {
        0
    }

    /// No sub-address: the function is the device.
    fn detail(&self) -> u8 {
        0
    }
}

/// The frame engine's host over the 82574L: the generic device host,
/// instantiated.
pub type E1000eHost = NetDeviceHost<E1000eNetBinding>;

#[cfg(test)]
mod tests {
    use super::*;

    /// The intake code is the only rung map: the entry ladder notes
    /// `PCI_BRING_UP_BASE + code` straight off `ClassBindRefused`, so the
    /// code must be the core's own dense ladder, starting at one.
    #[test]
    fn the_refusal_code_is_the_cores_rung_dense_from_one() {
        let ladder = [
            E1000eError::NotPresent,
            E1000eError::ResetTimeout,
            E1000eError::NvmNotLoaded,
            E1000eError::NoStationAddress,
            E1000eError::DmaAlloc,
            E1000eError::TxFull,
        ];
        for (i, error) in ladder.iter().enumerate() {
            assert_eq!(usize::from(E1000eNetBinding::refusal_code(error)), i + 1);
        }
        let top = crate::note::PCI_BRING_UP_BASE
            + u64::from(E1000eNetBinding::refusal_code(&E1000eError::TxFull));
        assert!(
            top < 64,
            "the whole band stays below the note's detail bits"
        );
        assert_eq!(top, 38, "the decoder ring in kore-abi ends at 38");
    }
}
