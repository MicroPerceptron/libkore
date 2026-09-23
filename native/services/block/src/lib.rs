//! Block-channel control plane.
//!
//! This crate deliberately contains no per-I/O request protocol.  A
//! [`BlockProvisioner`] authenticates policy, constructs bounded affine
//! [`BlockChannel`](kore_driver::block::BlockChannel)s, observes their lifecycle, and
//! revokes them.  Reads, writes, flushes, discards, and completions stay on the
//! granted channel and never call back through this control object.

#![no_std]

#[cfg(test)]
extern crate std;

pub mod client;
pub mod service;
pub mod wire;

pub use client::{BlockClient, OpenedChannel, RingBlock};
#[cfg(all(target_arch = "x86_64", target_os = "none"))]
pub use service::run;
pub use service::{BlockEndowment, BlockRefusal, BlockRequest, BlockResponse, BlockService};

/// Architecture-neutral service entry selected at this crate boundary.
///
/// x86 currently composes the direct NVMe implementation; other targets keep
/// the same founded control plane and report its honest unavailable posture.
/// Clients and crt0 do not branch on the provider implementation.
#[cfg(not(all(target_arch = "x86_64", target_os = "none")))]
pub async fn run(rt: kore_rt::entry::Rt) -> ! {
    kore_rt::entry::serve::<BlockService>(rt).await
}

use core::ops::{Deref, DerefMut};

pub use kore_driver::block::*;

/// Provisioning policy for one consumer/worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProvisionRequest {
    pub writable: bool,
    pub mode: CompletionMode,
    pub requested_queue_depth: u16,
    pub requested_buffer_bytes: usize,
}

/// Control-plane failures, distinct from data-plane failures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProvisionError {
    Unavailable,
    Policy,
    Exhausted,
    InvalidGeometry,
    Busy,
    Lost,
}

/// Observable provisioner state. No queue or device implementation detail is
/// exposed to callers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProvisionStatus {
    pub available: bool,
    pub channels: u16,
    pub limit: u16,
    pub geometry: Option<BlockGeometry>,
}

/// Maximum independently revocable channel lineages tracked by one control
/// service instance. Policy may choose a smaller runtime limit.
pub const MAX_CHANNELS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ProvisionId {
    slot: u16,
    generation: u32,
}

#[derive(Clone, Copy)]
struct Lineage {
    generation: u32,
    live: bool,
}

/// One affine provisioned channel plus its control-plane lineage.
///
/// This wrapper is deliberately neither `Clone` nor `Copy`. Dereferencing it
/// drives the channel directly; no method calls back into the provisioner.
pub struct Provisioned<C> {
    id: ProvisionId,
    channel: C,
}

impl<C> Deref for Provisioned<C> {
    type Target = C;

    fn deref(&self) -> &Self::Target {
        &self.channel
    }
}

impl<C> DerefMut for Provisioned<C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.channel
    }
}

/// A bounded control-plane allocator. Device-specific construction is a
/// closure so the service can compose the existing driver-host intake without
/// making the generic provisioner name PCI or NVMe.
pub struct BlockProvisioner {
    limit: u16,
    live: u16,
    available: bool,
    geometry: Option<BlockGeometry>,
    lineages: [Lineage; MAX_CHANNELS],
}

impl BlockProvisioner {
    pub const fn unavailable(limit: u16) -> Self {
        Self {
            limit,
            live: 0,
            available: false,
            geometry: None,
            lineages: [Lineage {
                generation: 1,
                live: false,
            }; MAX_CHANNELS],
        }
    }

    pub fn online(limit: u16, geometry: BlockGeometry) -> Result<Self, ProvisionError> {
        if limit == 0 || limit as usize > MAX_CHANNELS {
            return Err(ProvisionError::Exhausted);
        }
        let geometry = geometry
            .validate()
            .map_err(|_| ProvisionError::InvalidGeometry)?;
        Ok(Self {
            limit,
            live: 0,
            available: true,
            geometry: Some(geometry),
            lineages: [Lineage {
                generation: 1,
                live: false,
            }; MAX_CHANNELS],
        })
    }

    pub const fn status(&self) -> ProvisionStatus {
        ProvisionStatus {
            available: self.available,
            channels: self.live,
            limit: self.limit,
            geometry: self.geometry,
        }
    }

    /// Construct and account one affine channel. The returned value has one
    /// owner; dropping/moving it is Rust ownership, not a copyable token.
    pub fn provision<C>(
        &mut self,
        request: ProvisionRequest,
        build: impl FnOnce(ProvisionRequest) -> Result<C, ProvisionError>,
    ) -> Result<Provisioned<C>, ProvisionError> {
        if !self.available {
            return Err(ProvisionError::Unavailable);
        }
        if self.live >= self.limit {
            return Err(ProvisionError::Exhausted);
        }
        if request.requested_queue_depth == 0 || request.requested_buffer_bytes == 0 {
            return Err(ProvisionError::Policy);
        }
        if request.writable && self.geometry.is_some_and(|g| g.read_only) {
            return Err(ProvisionError::Policy);
        }
        let slot = self.lineages[..self.limit as usize]
            .iter()
            .position(|lineage| !lineage.live)
            .ok_or(ProvisionError::Exhausted)?;
        let channel = build(request)?;
        let lineage = &mut self.lineages[slot];
        lineage.live = true;
        self.live += 1;
        Ok(Provisioned {
            id: ProvisionId {
                slot: slot as u16,
                generation: lineage.generation,
            },
            channel,
        })
    }

    /// Drain a returned channel child-before-parent and close its exact
    /// lineage. A failed drain returns the affine channel to its owner.
    pub fn close<C>(
        &mut self,
        mut provisioned: Provisioned<C>,
        drain: impl FnOnce(&mut C) -> Result<(), ProvisionError>,
    ) -> Result<(), (ProvisionError, Provisioned<C>)> {
        if let Err(error) = drain(&mut provisioned.channel) {
            return Err((error, provisioned));
        }
        if !self.retire_id(provisioned.id) {
            return Err((ProvisionError::Lost, provisioned));
        }
        drop(provisioned);
        Ok(())
    }

    /// Force a returned channel through a service-specific revoke action,
    /// then retire its lineage. Used for reset/failure teardown rather than
    /// graceful close.
    pub fn revoke<C>(
        &mut self,
        mut provisioned: Provisioned<C>,
        revoke: impl FnOnce(&mut C),
    ) -> Result<(), ProvisionError> {
        revoke(&mut provisioned.channel);
        if !self.retire_id(provisioned.id) {
            return Err(ProvisionError::Lost);
        }
        drop(provisioned);
        Ok(())
    }

    fn retire_id(&mut self, id: ProvisionId) -> bool {
        let Some(lineage) = self.lineages.get_mut(id.slot as usize) else {
            return false;
        };
        if !lineage.live || lineage.generation != id.generation {
            return false;
        }
        lineage.live = false;
        lineage.generation = lineage.generation.wrapping_add(1).max(1);
        self.live -= 1;
        true
    }

    pub fn mark_lost(&mut self) {
        self.available = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kore_driver::block::memory::MemoryBlockDevice;

    #[test]
    fn provision_is_bounded_non_amplifying_and_data_plane_bypasses_control() {
        type Dev = MemoryBlockDevice<4096, 1024>;
        type Chan = BlockChannel<Dev, 1, 4>;
        let geometry = Dev::new(512, 2).unwrap().geometry();
        let mut provisioner = BlockProvisioner::online(1, geometry).unwrap();
        let request = ProvisionRequest {
            writable: true,
            mode: CompletionMode::Poll,
            requested_queue_depth: 4,
            requested_buffer_bytes: 1024,
        };
        let mut channel: Provisioned<Chan> = provisioner
            .provision(request, |_| {
                BlockChannel::new(Dev::new(512, 2).unwrap(), CompletionMode::Poll)
                    .map_err(|_| ProvisionError::InvalidGeometry)
            })
            .unwrap();
        assert_eq!(provisioner.status().channels, 1);
        assert!(matches!(
            provisioner.provision(request, |_| -> Result<(), ProvisionError> { Ok(()) }),
            Err(ProvisionError::Exhausted)
        ));

        // Direct I/O mutates only the channel; the provisioner's books do not
        // participate in or count the request path.
        let buffer = channel.register_buffer(512).unwrap();
        channel.write_buffer(buffer, 0, &[7; 512]).unwrap();
        let id = channel.submit_write(0, 1, buffer, 0).unwrap();
        assert_eq!(channel.poll().unwrap().request, id);
        assert_eq!(provisioner.status().channels, 1);
        assert!(
            provisioner
                .close(channel, |channel| {
                    channel.quiesce().map_err(|_| ProvisionError::Busy)
                })
                .is_ok()
        );
        assert_eq!(provisioner.status().channels, 0);
    }

    #[test]
    fn diskless_posture_is_honest_and_nonfatal() {
        let mut provisioner = BlockProvisioner::unavailable(4);
        let request = ProvisionRequest {
            writable: false,
            mode: CompletionMode::Poll,
            requested_queue_depth: 1,
            requested_buffer_bytes: 512,
        };
        assert!(matches!(
            provisioner.provision(request, |_| -> Result<(), ProvisionError> { Ok(()) }),
            Err(ProvisionError::Unavailable)
        ));
        assert!(!provisioner.status().available);
    }

    #[test]
    fn failed_close_returns_the_affine_channel_and_keeps_its_lineage_live() {
        type Dev = MemoryBlockDevice<4096, 1024>;
        type Chan = BlockChannel<Dev, 1, 2>;
        let geometry = Dev::new(512, 2).unwrap().geometry();
        let mut provisioner = BlockProvisioner::online(1, geometry).unwrap();
        let request = ProvisionRequest {
            writable: true,
            mode: CompletionMode::Poll,
            requested_queue_depth: 2,
            requested_buffer_bytes: 512,
        };
        let provisioned: Provisioned<Chan> = provisioner
            .provision(request, |_| {
                BlockChannel::new(Dev::new(512, 2).unwrap(), CompletionMode::Poll)
                    .map_err(|_| ProvisionError::InvalidGeometry)
            })
            .unwrap();

        let (_, provisioned) = provisioner
            .close(provisioned, |_| Err(ProvisionError::Busy))
            .err()
            .expect("close refuses with ownership preserved");
        assert_eq!(provisioner.status().channels, 1);
        assert!(matches!(
            provisioner.provision(request, |_| -> Result<(), ProvisionError> { Ok(()) }),
            Err(ProvisionError::Exhausted)
        ));
        assert!(
            provisioner
                .revoke(provisioned, |channel| channel.mark_lost())
                .is_ok()
        );
        assert_eq!(provisioner.status().channels, 0);
    }
}
