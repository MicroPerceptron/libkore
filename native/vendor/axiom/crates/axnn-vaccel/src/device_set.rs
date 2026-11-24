//! Multiple concrete targets over the existing per-provider admission policy.
//! IDs are executor-local metadata, never memory or device authority.
use crate::ax_graph::{ACCEL_BACKEND, AxDeviceModel, CPU_BACKEND};
use ax::{CostModel, Graph, Implementation, NodeView, TensorView};

/// One CPU policy and an ordered set of accelerator policies. Accelerator
/// index zero corresponds to backend 1. Costs share the caller's chosen unit.
/// Actual transfer mechanisms remain owned by the executor and its sessions.
#[derive(Clone, Debug)]
pub struct AxDeviceSet {
    pub cpu: AxDeviceModel,
    pub accelerators: Vec<AxDeviceModel>,
    /// Directional host-staged transfer costs. Missing edges are infeasible.
    /// The current Session path uploads/downloads bytes; these never promise P2P.
    pub routes: Vec<AxTransferCost>,
}
#[derive(Clone, Copy, Debug)]
pub struct AxTransferCost {
    pub source: Option<u16>,
    pub destination: u16,
    pub fixed: u64,
    pub per_byte: u64,
}

pub(crate) trait DevicePolicies: CostModel {
    fn cpu_policy(&self) -> &AxDeviceModel;
    fn accelerator_policy(&self, backend: u16) -> Option<&AxDeviceModel>;
}
impl DevicePolicies for AxDeviceModel {
    fn cpu_policy(&self) -> &AxDeviceModel {
        self
    }
    fn accelerator_policy(&self, backend: u16) -> Option<&AxDeviceModel> {
        (backend == ACCEL_BACKEND && self.accelerator).then_some(self)
    }
}
impl DevicePolicies for AxDeviceSet {
    fn cpu_policy(&self) -> &AxDeviceModel {
        &self.cpu
    }
    fn accelerator_policy(&self, backend: u16) -> Option<&AxDeviceModel> {
        self.accelerators.get(backend.checked_sub(1)? as usize)
    }
}
impl AxDeviceSet {
    fn target(&self, backend: u16) -> Option<(&AxDeviceModel, u16)> {
        if backend == CPU_BACKEND {
            Some((&self.cpu, CPU_BACKEND))
        } else {
            Some((self.accelerator_policy(backend)?, ACCEL_BACKEND))
        }
    }
}
impl CostModel for AxDeviceSet {
    fn backend_count(&self) -> u16 {
        self.accelerators
            .len()
            .checked_add(1)
            .and_then(|n| u16::try_from(n).ok())
            .unwrap_or(0)
    }
    fn supports(&self, b: u16, g: &Graph<'_>, n: NodeView<'_>) -> bool {
        self.target(b).is_some_and(|(m, b)| m.supports(b, g, n))
    }
    fn execute_cost(&self, b: u16, g: &Graph<'_>, n: NodeView<'_>) -> u64 {
        self.target(b)
            .map_or(u64::MAX, |(m, b)| m.execute_cost(b, g, n))
    }
    fn transfer_cost(&self, t: TensorView<'_>, s: Option<u16>, d: u16) -> u64 {
        if self.target(d).is_none() || s.is_some_and(|s| self.target(s).is_none()) {
            return u64::MAX;
        }
        if s == Some(d) {
            return 0;
        }
        let Some(bytes) = ax_exec::native::tensor_storage_bytes(t) else {
            return u64::MAX;
        };
        self.routes
            .iter()
            .filter(|r| r.source == s && r.destination == d)
            .map(|r| {
                r.fixed
                    .saturating_add(r.per_byte.saturating_mul(bytes as u64))
            })
            .min()
            .unwrap_or(u64::MAX)
    }
    fn partition_cost(&self, b: u16) -> u64 {
        self.target(b)
            .map_or(u64::MAX, |(m, b)| m.partition_cost(b))
    }
    fn partition_cost_implementation(&self, b: u16, i: Implementation) -> u64 {
        self.target(b)
            .map_or(u64::MAX, |(m, b)| m.partition_cost_implementation(b, i))
    }
    fn external_output_backend(&self) -> Option<u16> {
        Some(CPU_BACKEND)
    }
    fn implementation_count(&self, b: u16, g: &Graph<'_>, n: NodeView<'_>) -> u16 {
        self.target(b)
            .map_or(0, |(m, b)| m.implementation_count(b, g, n))
    }
    fn supports_implementation(
        &self,
        b: u16,
        g: &Graph<'_>,
        n: NodeView<'_>,
        i: Implementation,
    ) -> bool {
        self.target(b)
            .is_some_and(|(m, b)| m.supports_implementation(b, g, n, i))
    }
    fn execute_cost_implementation(
        &self,
        b: u16,
        g: &Graph<'_>,
        n: NodeView<'_>,
        i: Implementation,
    ) -> u64 {
        self.target(b)
            .map_or(u64::MAX, |(m, b)| m.execute_cost_implementation(b, g, n, i))
    }
    fn region_execute_cost_implementation(
        &self,
        b: u16,
        g: &Graph<'_>,
        n: NodeView<'_>,
        i: Implementation,
    ) -> u64 {
        self.target(b).map_or(u64::MAX, |(m, b)| {
            m.region_execute_cost_implementation(b, g, n, i)
        })
    }
}
