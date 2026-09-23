//! `client` — the device-broker client seam: what a driver claims through.
//!
//! [`DeviceClient`] is the `res/USERLAND.md` §4 pinned async shape and
//! [`RingDevice`] is the §5.6 ring + channel proxy over the generic
//! [`SvcClient`](kore_rt::svc::SvcClient). **No sync client is ever
//! written** (§5.6): the caller awaits.
//!
//! A driver holds `impl DeviceClient` and never names the concrete broker,
//! so the claim it issues at `start_on_metal` is written once and does not
//! care whether it is talking to a ring, a proxy, or (once exec conveys
//! client faces at spawn) something wired differently.

use kore_abi::device::DeviceSpecies;
use kore_rt::cap::{DeviceToken, PlatformToken, verbs};
use kore_rt::svc::SvcClient;
use kore_rt::transport::Transport;

use crate::service::{
    DeviceBroker, DeviceObservation, DeviceRefusal, DeviceRequest, DeviceResponse,
};

/// A claim's delivered authority — the device, possessed as the species
/// the broker said it was.
///
/// Two variants rather than one flagged token, mirroring the split one
/// level down: the two species support disjoint derives, and a driver
/// handed the wrong one must fail at the possession verb rather than at
/// its first BAR map.
#[derive(Debug)]
pub enum ClaimedDevice {
    /// A PCI function — BAR windows and MSI(-X) routes derive from it.
    Pci(DeviceToken),
    /// A firmware-described device — fixed I/O/MMIO, a line, and an optional
    /// firmware IOMMU requester binding.
    Platform(PlatformToken),
}

impl ClaimedDevice {
    /// The PCI half, if that is what was claimed.
    pub fn pci(self) -> Option<DeviceToken> {
        match self {
            Self::Pci(d) => Some(d),
            Self::Platform(_) => None,
        }
    }

    /// The firmware-described half, if that is what was claimed.
    pub fn platform(self) -> Option<PlatformToken> {
        match self {
            Self::Platform(d) => Some(d),
            Self::Pci(_) => None,
        }
    }
}

/// What a claim answered: the inventory index to name in a later
/// `Release`, and the possessed device.
#[derive(Debug)]
pub struct Claimed {
    /// The inventory index handed over.
    pub index: u32,
    /// The device, possessed as its species.
    pub device: ClaimedDevice,
    /// The function's diagnostic word
    /// ([`DeviceRecord::diag`](kore_abi::device::DeviceRecord::diag)) —
    /// folded into the claimant's readiness note when realization
    /// refuses, so the note names the mechanism.
    pub diag: u32,
    /// PCI revision id — the `si_stepping` scalar accelerator firmware
    /// validates against.
    pub revision: u8,
    /// PCI vendor id, or zero for a platform device.
    pub vendor: u16,
    /// PCI device id, or zero for a platform device.
    pub device_id: u16,
}

/// The **client** side of the broker protocol (`res/USERLAND.md` §4,
/// pinned): issue one arbitration request, await the data reply beside its
/// delivered device grant.
#[allow(async_fn_in_trait)] // clients are generic over the impl (§4)
pub trait DeviceClient {
    /// Issue one request; a `Granted` reply's device arrives
    /// **possessed** beside it.
    ///
    /// `Err` carries the refusal — including [`DeviceRefusal::Denied`] for
    /// an infrastructure failure of the request path, which is honest
    /// rather than convenient: a claimant that cannot reach the broker and
    /// one the broker refused are both claimants without a device, and
    /// neither may proceed to bring hardware up.
    async fn claim(&mut self, req: DeviceRequest) -> Result<Claimed, DeviceRefusal>;

    /// Observe the first matching identity/class tuple without claiming it.
    async fn inspect(
        &mut self,
        selector: kore_rt::service::DeviceSelector,
    ) -> Result<DeviceObservation, DeviceRefusal>;

    async fn inspect_i2c(
        &mut self,
        row: u8,
    ) -> Result<kore_abi::device::I2cAdmissionPage, DeviceRefusal>;

    /// Request release of a claimed entry.
    ///
    /// This returns `false` with the current broker because an index alone is
    /// not proof that the affine device grant is dead. It remains in the seam
    /// for a future authority-backed return protocol.
    async fn release(&mut self, index: u32) -> bool;

    /// How many entries the inventory holds.
    ///
    /// Failure is explicit: zero is a valid hardware fact and must not also
    /// mean that the broker, transport, or inventory observation failed.
    async fn count(&mut self) -> Result<u32, DeviceRefusal>;
}

/// The born-async ring client over the generic [`SvcClient`].
pub struct RingDevice<'w, 't, T: Transport>(pub SvcClient<'w, 't, T, DeviceBroker>);

impl<T: Transport> RingDevice<'_, '_, T> {
    /// Close admission to the exact previously granted PCI CID. The CID is
    /// observation data from the possessed device; broker face ownership is
    /// checked independently. Retry the same request after uncertainty. No
    /// returned state permits freeing buffers, clearing claims or regranting.
    pub async fn close_claim(
        &mut self,
        index: u32,
        grant: kore_abi::cap::Cid,
    ) -> Result<kore_abi::session::DeviceClaimCloseReport, DeviceRefusal> {
        let request = DeviceRequest::CloseClaim {
            index,
            grant: grant.0,
        };
        let reply = if self.0.is_idle() {
            self.0.call(&request).await
        } else {
            self.0.resume_exact(&request).await
        }
        .map_err(|_| DeviceRefusal::Denied)?;
        match reply.response {
            DeviceResponse::ClaimClosed(report) => Ok(report),
            DeviceResponse::Refused(reason) => Err(reason),
            _ => Err(DeviceRefusal::Denied),
        }
    }
    /// Contain the exact previously granted PCI CID. The CID is
    /// observation data from the possessed device; broker face ownership is
    /// checked independently. Retry the same request after uncertainty. No
    /// returned state permits freeing buffers, clearing claims or regranting.
    pub async fn quiesce_claim(
        &mut self,
        index: u32,
        grant: kore_abi::cap::Cid,
    ) -> Result<kore_abi::session::QuiesceReport, DeviceRefusal> {
        let request = DeviceRequest::QuiesceClaim {
            index,
            grant: grant.0,
        };
        let reply = if self.0.is_idle() {
            self.0.call(&request).await
        } else {
            self.0.resume_exact(&request).await
        }
        .map_err(|_| DeviceRefusal::Denied)?;
        match reply.response {
            DeviceResponse::ClaimQuiesced(report) => Ok(report),
            DeviceResponse::Refused(reason) => Err(reason),
            _ => Err(DeviceRefusal::Denied),
        }
    }
}

/// Progress of one exact claim retirement; containment never releases backing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClaimRetirementProgress {
    Closing,
    Quiescing,
    Contained(kore_abi::session::QuiesceReport),
    DmaPending {
        fabric: kore_abi::session::QuiesceReport,
        report: kore_abi::session::DeviceClaimDmaReport,
    },
    DmaRetired {
        fabric: kore_abi::session::QuiesceReport,
        report: kore_abi::session::DeviceClaimDmaReport,
    },
}

/// Observation failure never discards the original connection or identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClaimRetirementError {
    MissingIdentity,
    Transport(kore_rt::svc::CallError),
    Refused(DeviceRefusal),
    Admission(kore_abi::session::DeviceClaimCloseState),
    UnexpectedReply,
}

impl ClaimRetirementProgress {
    fn request(self, index: u32, grant: kore_abi::cap::Cid) -> Option<DeviceRequest> {
        match self {
            Self::Closing => Some(DeviceRequest::CloseClaim {
                index,
                grant: grant.0,
            }),
            Self::Quiescing => Some(DeviceRequest::QuiesceClaim {
                index,
                grant: grant.0,
            }),
            Self::Contained(_) | Self::DmaPending { .. } => Some(DeviceRequest::RetireClaimDma {
                index,
                grant: grant.0,
            }),
            Self::DmaRetired { .. } => None,
        }
    }

    fn accept(&mut self, response: DeviceResponse) -> Result<(), ClaimRetirementError> {
        use kore_abi::session::DeviceClaimCloseState as State;
        match (*self, response) {
            (Self::Closing, DeviceResponse::ClaimClosed(report)) => match report.state {
                State::AdmissionDrained => {
                    *self = Self::Quiescing;
                    Ok(())
                }
                State::Busy | State::Pending => Ok(()),
                state => Err(ClaimRetirementError::Admission(state)),
            },
            (Self::Quiescing, DeviceResponse::ClaimQuiesced(report)) => {
                *self = Self::Contained(report);
                Ok(())
            }
            (
                Self::Contained(fabric) | Self::DmaPending { fabric, .. },
                DeviceResponse::ClaimDmaRetired(report),
            ) => {
                *self = if report.state == kore_abi::session::DeviceClaimDmaState::Retired {
                    Self::DmaRetired { fabric, report }
                } else {
                    Self::DmaPending { fabric, report }
                };
                Ok(())
            }
            (_, DeviceResponse::Refused(reason)) => Err(ClaimRetirementError::Refused(reason)),
            _ => Err(ClaimRetirementError::UnexpectedReply),
        }
    }
}

/// The original client connection for one issued device claim. Keep this owner
/// beyond probe/bind failure and terminal containment: a freshly opened face
/// cannot stand in for the broker's recorded structural claimant. The grant
/// number is correlation data; this affine connection carries the authority.
/// Dropping this owner never requests release or makes the device reclaimable.
pub struct ClaimBroker<'w, 't, T: Transport> {
    face: RingDevice<'w, 't, T>,
    index: u32,
    grant: Option<kore_abi::cap::Cid>,
    retirement: ClaimRetirementProgress,
}

impl<T: Transport> ClaimBroker<'_, '_, T> {
    /// Observe one bounded reply slice, then return to the owner's serve loop.
    /// Timeout leaves the exact request published; the next step resumes it.
    /// Each call performs at most one protocol exchange. Only an acknowledged
    /// admission drain advances to containment, and no phase releases backing.
    pub async fn retire_step(
        &mut self,
        budget_ns: u64,
    ) -> Result<ClaimRetirementProgress, ClaimRetirementError> {
        let grant = self.grant.ok_or(ClaimRetirementError::MissingIdentity)?;
        let Some(request) = self.retirement.request(self.index, grant) else {
            return Ok(self.retirement);
        };
        let reply = if self.face.0.is_idle() {
            self.face.0.call_bounded(&request, budget_ns).await
        } else {
            self.face.0.resume_exact_bounded(&request, budget_ns).await
        }
        .map_err(ClaimRetirementError::Transport)?;
        self.retirement.accept(reply.response)?;
        Ok(self.retirement)
    }

    pub fn retirement_progress(&self) -> ClaimRetirementProgress {
        self.retirement
    }

    /// Whether the delivered PCI identity was observed before realization.
    /// An unavailable identity leaves the claim quarantined, never guessed.
    pub fn correlated(&self) -> bool {
        self.grant.is_some()
    }

    /// Close only this claim, through the connection which received it. A
    /// cancelled/uncertain crossing resumes the same request; callers cannot
    /// substitute a different index or grant. AdmissionDrained is not proof of
    /// physical quiescence and does not authorize release or buffer reuse.
    pub async fn close(
        &mut self,
    ) -> Result<kore_abi::session::DeviceClaimCloseReport, DeviceRefusal> {
        let grant = self.grant.ok_or(DeviceRefusal::Denied)?;
        self.face.close_claim(self.index, grant).await
    }
    /// Contain this already-closed, drained claim through its original face.
    /// Cancelled/uncertain crossings resume exactly; callers cannot substitute
    /// another identity. The report does not authorize claim release or buffer
    /// reuse, and Busy leaves custody with this owner for an exact retry.
    pub async fn quiesce(&mut self) -> Result<kore_abi::session::QuiesceReport, DeviceRefusal> {
        let grant = self.grant.ok_or(DeviceRefusal::Denied)?;
        self.face.quiesce_claim(self.index, grant).await
    }
}

impl<T: Transport> DeviceClient for RingDevice<'_, '_, T> {
    async fn claim(&mut self, req: DeviceRequest) -> Result<Claimed, DeviceRefusal> {
        let Ok(mut reply) = self.0.call(&req).await else {
            return Err(DeviceRefusal::Denied);
        };
        let DeviceResponse::Granted {
            index,
            species,
            diag,
            revision,
            vendor,
            device: device_id,
        } = reply.response
        else {
            return Err(match reply.response {
                DeviceResponse::Refused(r) => r,
                // A reply out of vocabulary for this request is not a
                // device, so it is a refusal — never an optimistic bind
                // against whatever landed in the slot.
                _ => DeviceRefusal::Denied,
            });
        };
        let Some(returned) = reply.retained[0].take() else {
            self.0.poison_retained(reply.retained);
            return Err(DeviceRefusal::Denied);
        };
        // Possess as the species the broker named. A mismatch refuses
        // here, at the claim, rather than twenty lines into bring-up. The
        // token retains the runtime landing owner for its whole lifetime.
        let device = match species {
            DeviceSpecies::Pci => match verbs::claim_returned_device(returned) {
                Ok(device) => ClaimedDevice::Pci(device),
                Err(returned) => {
                    reply.retained[0] = Some(returned);
                    self.0.poison_retained(reply.retained);
                    return Err(DeviceRefusal::Denied);
                }
            },
            DeviceSpecies::Platform => match verbs::claim_returned_platform_device(returned) {
                Ok(device) => ClaimedDevice::Platform(device),
                Err(returned) => {
                    reply.retained[0] = Some(returned);
                    self.0.poison_retained(reply.retained);
                    return Err(DeviceRefusal::Denied);
                }
            },
        };
        Ok(Claimed {
            index,
            device,
            diag,
            revision,
            vendor,
            device_id,
        })
    }

    async fn inspect(
        &mut self,
        selector: kore_rt::service::DeviceSelector,
    ) -> Result<DeviceObservation, DeviceRefusal> {
        let reply = self
            .0
            .call(&DeviceRequest::Inspect(selector))
            .await
            .map_err(|_| DeviceRefusal::Denied)?;
        match reply.response {
            DeviceResponse::Observed(observed) => Ok(observed),
            DeviceResponse::Refused(refusal) => Err(refusal),
            _ => Err(DeviceRefusal::Denied),
        }
    }

    async fn inspect_i2c(
        &mut self,
        row: u8,
    ) -> Result<kore_abi::device::I2cAdmissionPage, DeviceRefusal> {
        let reply = self
            .0
            .call(&DeviceRequest::InspectI2c { row })
            .await
            .map_err(|_| DeviceRefusal::Denied)?;
        match reply.response {
            DeviceResponse::I2cObserved(page) if page.row == row => Ok(page),
            DeviceResponse::Refused(reason) => Err(reason),
            _ => Err(DeviceRefusal::Denied),
        }
    }

    async fn release(&mut self, index: u32) -> bool {
        matches!(
            self.0.call(&DeviceRequest::Release { index }).await,
            Ok(reply) if reply.response == DeviceResponse::Released
        )
    }

    async fn count(&mut self) -> Result<u32, DeviceRefusal> {
        let reply = self
            .0
            .call(&DeviceRequest::Count)
            .await
            .map_err(|_| DeviceRefusal::Denied)?;
        match reply.response {
            DeviceResponse::Counted { len } => Ok(len),
            DeviceResponse::Refused(refusal) => Err(refusal),
            _ => Err(DeviceRefusal::Denied),
        }
    }
}

/// Where a driver row's boot block carries its client face onto broker
/// `k` — **the one binding between the kernel founder that stamps the
/// bank and the claimant that reads it.**  A row's own serving pairs
/// occupy logical indices `0..serve_pairs` of `SVC_SEND` / `SVC_RING`,
/// `0..2 * serve_pairs` of `SVC_RETURN`, and `0..serve_pairs * grants` of
/// `SVC_GRANT`; the faces start immediately after each run, so a row that
/// grows a second serving pair (the frame row's exec pair, #936) moves its
/// faces up instead of colliding with it.  Face 0 of a one-pair row is
/// bit-identical to the historical layout (`SVC_SEND@(1+k)`,
/// `SVC_GRANT@(1+2k)`, block's `@(2+2k)`), so no existing driver row sees
/// a different bank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaceLayout {
    /// Face 0's kick and ring-view role index (`SVC_SEND` / `SVC_RING`).
    pub face_base: u16,
    /// Face 0's release-send role index (`SVC_RETURN`).
    pub return_base: u16,
    /// Face 0's grant-receive role index.
    pub grant_recv_base: u16,
    /// The per-face stride of the grant lane (2 today: one receive + one
    /// named landing).
    pub grant_stride: u16,
}

impl FaceLayout {
    /// The layout for a row whose own serving pairs are `0..serve_pairs`
    /// with `reply_grants` grant lanes each — cite the row's protocol
    /// (`SERVE_PAIRS`, `BOOT_REPLY_GRANTS`), which is what the manifest
    /// generator hands the founder.  The grant lane never starts at `@0`,
    /// even for a grant-less row: input's historical bank left `@0` free,
    /// and the founder stamps that shape unchanged.
    pub const fn for_row(serve_pairs: usize, reply_grants: usize) -> Self {
        let own_grants = serve_pairs * reply_grants;
        Self {
            face_base: serve_pairs as u16,
            return_base: (2 * serve_pairs) as u16,
            grant_recv_base: if own_grants == 0 {
                1
            } else {
                own_grants as u16
            },
            grant_stride: 2,
        }
    }
}

#[cfg(test)]
mod face_layout_tests {
    use super::FaceLayout;

    /// The bank every existing driver row reads is unchanged, and a
    /// two-pair row's faces sit past its own pairs.
    #[test]
    fn faces_start_past_the_rows_own_pairs() {
        let one_pair_no_grants = FaceLayout::for_row(1, 0); // input
        assert_eq!(one_pair_no_grants.face_base, 1);
        assert_eq!(one_pair_no_grants.return_base, 2);
        assert_eq!(one_pair_no_grants.grant_recv_base, 1);
        let one_pair_one_grant = FaceLayout::for_row(1, 1); // accel, xe
        assert_eq!(one_pair_one_grant.face_base, 1);
        assert_eq!(one_pair_one_grant.grant_recv_base, 1);
        let one_pair_two_grants = FaceLayout::for_row(1, 2); // block
        assert_eq!(one_pair_two_grants.grant_recv_base, 2);
        let two_pairs_one_grant = FaceLayout::for_row(2, 1); // the frame row
        assert_eq!(two_pairs_one_grant.face_base, 2);
        assert_eq!(two_pairs_one_grant.return_base, 4);
        assert_eq!(two_pairs_one_grant.grant_recv_base, 2);
        assert_eq!(two_pairs_one_grant.grant_stride, 2);
    }
}

/// A driver's client faces onto **every live device broker** (#572, one
/// broker per PCIe root complex): face `k` talks to broker `k`, and which
/// faces exist is read off the boot block — an absent handle means that
/// broker was skipped at founding (its root does not exist), which is
/// data, not a wiring fault.
///
/// The refusal fold is fail-closed: the first `Granted` anywhere wins
/// (face order = root ordinal order: ascending (segment, bus_start)); `Taken` on some face with no
/// grant anywhere answers `Taken`; only all-`NoMatch` answers `NoMatch`;
/// and any other refusal — the kernel's own gate, a dead transport —
/// propagates immediately, because a founded-but-broken face is a wiring
/// fault, never evidence about hardware.
pub struct MultiBroker<'w, 't, T: Transport> {
    faces: [Option<RingDevice<'w, 't, T>>; kore_abi::device::MAX_ROOT_COMPLEXES],
}

impl<'t, T: Transport> MultiBroker<'static, 't, T> {
    /// Open every face the founder stamped.  `None` when no face opened
    /// at all (this row was founded off the chain or ahead of every
    /// broker — it can claim nothing, and says so rather than sending
    /// into a dead channel), and also when a STAMPED face fails to open —
    /// a half-wired row must refuse, not quietly claim through fewer
    /// brokers than it was given.
    #[cfg(target_os = "none")]
    pub fn open(
        transport: &'t T,
        boot: &kore_abi::task::Bootstrap<'_>,
        layout: FaceLayout,
    ) -> Option<Self> {
        use kore_abi::task::role;
        use kore_rt::service::MAX_GRANTS;
        use kore_rt::svc::{ClientEndpoints, GrantPin, REQ_CAP, SvcClient, client_rings};

        let mut faces = [const { None }; kore_abi::device::MAX_ROOT_COMPLEXES];
        let mut any = false;
        for k in 0..kore_abi::device::MAX_ROOT_COMPLEXES as u16 {
            let Some(kick) = boot.handle(role::SVC_SEND, layout.face_base + k) else {
                continue;
            };
            let grant_recv = boot.handle(
                role::SVC_GRANT,
                layout.grant_recv_base + layout.grant_stride * k,
            )?;
            let window = boot.handle(role::SVC_RING, layout.face_base + k)?;
            let leased = kore_rt::cap::verbs::take_view(window)?;
            let window = kore_rt::cap::verbs::map_view(leased)?;
            // SAFETY: the `serve_window` contract — `map_view` installed
            // this R/W-USER view over the pair's leased backing in *this*
            // task's space, page-aligned and live for the task's whole
            // life, and this task is its sole client-side attacher.
            let window: &'static mut [u8] = unsafe {
                core::slice::from_raw_parts_mut(window.cast::<u8>().as_ptr(), window.len())
            };
            let rings =
                client_rings::<DeviceBroker>(window, REQ_CAP, kore_rt::svc::REP_CAP).ok()?;
            // The founder still stamps the historical fixed landing at the
            // stride's second role, but device authority outlives a broker
            // call: the runtime rotates an owned landing per delivery and
            // deliberately does not name that fixed slot here.
            let mut grants = [const { None }; MAX_GRANTS];
            grants[0] = Some(GrantPin::retained(grant_recv));
            faces[k as usize] = Some(RingDevice(SvcClient::new(
                transport,
                rings,
                ClientEndpoints { kick, grants },
            )));
            any = true;
        }
        any.then_some(Self { faces })
    }

    /// The live face ordinals, ascending — the catalog root order (ascending (segment, bus_start)).
    pub fn live(&self) -> impl Iterator<Item = usize> + '_ {
        self.faces
            .iter()
            .enumerate()
            .filter_map(|(k, face)| face.is_some().then_some(k))
    }

    /// Face `k`, when broker `k` was founded.  Face 0 is the platform
    /// route: the platform bus rides broker ordinal 0 alone.
    pub fn face(&mut self, k: usize) -> Option<&mut RingDevice<'static, 't, T>> {
        self.faces.get_mut(k)?.as_mut()
    }

    /// Claim the first match across every live broker, in discovery
    /// order, with the fail-closed refusal fold documented on the type.
    pub async fn claim_first(
        &mut self,
        selector: kore_rt::service::DeviceSelector,
    ) -> Result<Claimed, DeviceRefusal> {
        self.claim_first_origin(selector)
            .await
            .map(|(_, claimed)| claimed)
    }

    /// Claim one device and retain its issuing connection as a separate affine
    /// owner. Record the delivered PCI CID before the caller can consume the
    /// token in a probe or binding. The other root faces are no longer needed
    /// by a single-device provider and are dropped without releasing claims.
    pub async fn claim_owned(
        mut self,
        selector: kore_rt::service::DeviceSelector,
    ) -> Result<(Claimed, ClaimBroker<'static, 't, T>), DeviceRefusal> {
        let (origin, claimed) = self.claim_first_origin(selector).await?;
        let grant = match &claimed.device {
            ClaimedDevice::Pci(token) => {
                verbs::capability_cid(token.slot(), kore_abi::cap::ResourceKind::PciDevice)
            }
            ClaimedDevice::Platform(_) => None,
        };
        let face = self.faces[origin]
            .take()
            .expect("successful claim retains its face");
        let owner = ClaimBroker {
            face,
            index: claimed.index,
            grant,
            retirement: ClaimRetirementProgress::Closing,
        };
        Ok((claimed, owner))
    }

    async fn claim_first_origin(
        &mut self,
        selector: kore_rt::service::DeviceSelector,
    ) -> Result<(usize, Claimed), DeviceRefusal> {
        let mut contended = false;
        for (origin, face) in self.faces.iter_mut().enumerate() {
            let Some(face) = face else {
                continue;
            };
            match face.claim(DeviceRequest::Claim(selector)).await {
                Ok(claimed) => return Ok((origin, claimed)),
                Err(DeviceRefusal::Taken) => contended = true,
                Err(DeviceRefusal::NoMatch) => {}
                Err(refusal) => return Err(refusal),
            }
        }
        Err(if contended {
            DeviceRefusal::Taken
        } else {
            DeviceRefusal::NoMatch
        })
    }

    /// Observe the first identity/class match across every live broker without
    /// consulting or mutating any claim ledger.
    pub async fn inspect_first(
        &mut self,
        selector: kore_rt::service::DeviceSelector,
    ) -> Result<DeviceObservation, DeviceRefusal> {
        for face in self.faces.iter_mut().flatten() {
            match face.inspect(selector).await {
                Ok(observed) => return Ok(observed),
                Err(DeviceRefusal::NoMatch) => {}
                Err(refusal) => return Err(refusal),
            }
        }
        Err(DeviceRefusal::NoMatch)
    }
}

#[cfg(test)]
mod retirement_tests {
    use super::*;
    use kore_abi::cap::Cid;
    use kore_abi::session::{
        DeviceClaimCloseReport, DeviceClaimCloseState as Close, QUIESCE_FLAG_IRQ_MASKED,
        QUIESCE_VERDICT_PROVEN, QuiesceReport,
    };

    #[test]
    fn retirement_never_advances_without_its_exact_phase_acknowledgement() {
        let mut phase = ClaimRetirementProgress::Closing;
        let close_request = DeviceRequest::CloseClaim {
            index: 17,
            grant: 0x100000033,
        };
        let quiet = QuiesceReport {
            rid: 22,
            verdict: QUIESCE_VERDICT_PROVEN,
            flags: QUIESCE_FLAG_IRQ_MASKED,
        };
        for state in [
            Close::Busy,
            Close::Pending,
            Close::Mismatch,
            Close::Untracked,
        ] {
            let result = phase.accept(DeviceResponse::ClaimClosed(DeviceClaimCloseReport {
                state,
                readers: if state == Close::Pending { 3 } else { 0 },
            }));
            assert_eq!(
                result.is_ok(),
                matches!(state, Close::Busy | Close::Pending)
            );
            assert_eq!(phase.request(17, Cid(0x100000033)), Some(close_request));
        }
        assert_eq!(
            phase.accept(DeviceResponse::ClaimQuiesced(quiet)),
            Err(ClaimRetirementError::UnexpectedReply)
        );
        assert_eq!(phase, ClaimRetirementProgress::Closing);
        phase
            .accept(DeviceResponse::ClaimClosed(DeviceClaimCloseReport {
                state: Close::AdmissionDrained,
                readers: 0,
            }))
            .unwrap();
        let containment_request = DeviceRequest::QuiesceClaim {
            index: 17,
            grant: 0x100000033,
        };
        assert_eq!(
            phase.request(17, Cid(0x100000033)),
            Some(containment_request)
        );
        for refusal in [DeviceRefusal::Busy, DeviceRefusal::Denied] {
            assert_eq!(
                phase.accept(DeviceResponse::Refused(refusal)),
                Err(ClaimRetirementError::Refused(refusal))
            );
            assert_eq!(
                phase.request(17, Cid(0x100000033)),
                Some(containment_request)
            );
        }
        assert_eq!(
            phase.accept(DeviceResponse::ClaimClosed(DeviceClaimCloseReport {
                state: Close::AdmissionDrained,
                readers: 0
            })),
            Err(ClaimRetirementError::UnexpectedReply)
        );
        phase.accept(DeviceResponse::ClaimQuiesced(quiet)).unwrap();
        assert_eq!(phase, ClaimRetirementProgress::Contained(quiet));
        let dma_request = DeviceRequest::RetireClaimDma {
            index: 17,
            grant: 0x100000033,
        };
        assert_eq!(phase.request(17, Cid(0x100000033)), Some(dma_request));
        use kore_abi::session::{DeviceClaimDmaReport, DeviceClaimDmaState as Dma};
        for state in [
            Dma::WindowRetirementFailed,
            Dma::EntryRetirementFailed,
            Dma::AbortRetained,
            Dma::EntryRetained,
            Dma::ContainmentFailed,
            Dma::Unbound,
        ] {
            let report = DeviceClaimDmaReport { state, removed: 0 };
            phase
                .accept(DeviceResponse::ClaimDmaRetired(report))
                .unwrap();
            assert_eq!(
                phase,
                ClaimRetirementProgress::DmaPending {
                    fabric: quiet,
                    report
                }
            );
            assert_eq!(phase.request(17, Cid(0x100000033)), Some(dma_request));
            assert_eq!(
                phase.accept(DeviceResponse::ClaimQuiesced(quiet)),
                Err(ClaimRetirementError::UnexpectedReply)
            );
            assert_eq!(
                phase.accept(DeviceResponse::Refused(DeviceRefusal::Busy)),
                Err(ClaimRetirementError::Refused(DeviceRefusal::Busy))
            );
            assert_eq!(phase.request(17, Cid(0x100000033)), Some(dma_request));
        }
        let report = DeviceClaimDmaReport {
            state: Dma::Retired,
            removed: 0,
        };
        phase
            .accept(DeviceResponse::ClaimDmaRetired(report))
            .unwrap();
        assert_eq!(
            phase,
            ClaimRetirementProgress::DmaRetired {
                fabric: quiet,
                report
            }
        );
        assert_eq!(phase.request(17, Cid(0x100000033)), None);
    }
}
