//! `verbs` — the **kernel-verb seam** (`res/USERLAND.md` §2): the one
//! face production code reaches capability verbs through.
//!
//! One module, no cfg: every build — host test, userland target, kernel
//! `boot` link — sees the same face.  A verb is either **live on the
//! metal path** (a `SLOTS_VA` possession read or a trap arm, gated
//! per-target inside its body) or a **fail-closed refusal** (`None`/
//! `false`/`Err`) until its kernel arm lands.  A refusal is honest and typed —
//! never a panic, never a simulation.  (The mock world that used to
//! stand in for these verbs under test builds was deleted by the
//! 2026-07-24 ruling: proof of verb behavior lives in the QEMU boot
//! gates, not in a host model.)
//!
//! **Curated, not a blanket alias**: only the verbs production
//! legitimately calls appear here — a caller reaching for a verb this
//! face doesn't carry is a compile error, which is the §2.1 bar's teeth.

mod metal {
    //! The metal verb bodies: live implementations where the kernel arm
    //! exists (each gated on its target inside), typed refusals where it
    //! does not yet.

    use core::ptr::NonNull;

    #[cfg(any(test, target_os = "none"))]
    use kore_abi::cap::Rights;
    use kore_abi::session::SlotRef;

    use crate::cap::{
        BarWindow, ChanRecvToken, ChanSendToken, ChannelMintError, ChannelPair, ChannelSpaceToken,
        DeviceMmuToken, DeviceToken, DisplayToken, DmaWindow, ExecToken, ExecutableCode,
        FinalizeExecError, GuestRegionToken, GuestSpaceToken, HeldChanRecvToken, HeldChanSendToken,
        HypervisorToken, ImageStoreToken, LeaseToken, LeasedToken, MappedLeaseView, ObserveToken,
        PeerDmaAccess, PeerWindow, PlatformToken, PowerToken, QuiesceFault, Quiesced, RegionToken,
        RetireOutcome, RouteToken, StashedLease, SupplyToken, TaskStartOutcome, TaskToken,
        VirtMemoryToken, XMintToken,
    };

    #[cfg(any(test, target_os = "none"))]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum ChannelMintCompletion {
        Committed,
        RolledBack(kore_abi::session::Status),
        Indeterminate,
    }

    /// Result of a non-consuming lineage probe on a receive endpoint.
    ///
    /// `Dead` is positive kernel evidence that the endpoint or an ancestor was
    /// revoked. `Live` means a same-kind descendant was derived and then
    /// retired without changing the probed parent. Everything else preserves
    /// the caller's reconciliation state for a later retry.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum ChanRecvLineage {
        Live,
        Dead,
        Retryable,
    }

    /// Why a direct Found result could not become the caller's exact task
    /// authority.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum DirectTaskClaimError {
        NotLive,
        NotTask,
        Rights,
    }

    #[cfg(any(test, target_os = "none"))]
    fn exact_channel_space_rights(rights: Rights) -> bool {
        rights == crate::cap::CHANNEL_SPACE_RIGHTS
    }

    #[cfg(any(test, target_os = "none"))]
    fn valid_channel_endpoint_derivation_rights(rights: Rights) -> bool {
        !rights.is_empty() && crate::cap::CHANNEL_ENDPOINT_RIGHTS.contains(rights)
    }

    /// Classify only completions whose shape proves the three-result mint's
    /// atomic outcome. Wait-only outcomes require flag or verb shapes this
    /// session never carries, and unknown status bits are not the ABI's
    /// explicit `Malformed`; both are quarantined rather than guessed.
    #[cfg(any(test, target_os = "none"))]
    fn classify_channel_mint(committed: kore_abi::session::Committed) -> ChannelMintCompletion {
        use kore_abi::session::Status;

        let raw_status = committed.raw() as u16;
        let failing = committed.failing_index();
        match (raw_status, failing) {
            (0, 0) => ChannelMintCompletion::Committed,
            (1..=6, 0..=2) => ChannelMintCompletion::RolledBack(committed.status()),
            // Keep this spelling exhaustive against Status without treating
            // its fail-closed unknown decoder as an engine-authored refusal.
            (x, _) if x > Status::AltDelivery as u16 => ChannelMintCompletion::Indeterminate,
            _ => ChannelMintCompletion::Indeterminate,
        }
    }

    /// Why an authoritative device-inventory observation failed.
    ///
    /// There is deliberately no `EndOfInventory` variant: callers fetch the
    /// count first, and every failed record read inside that bound is an
    /// infrastructure error. This prevents a refused session, corrupt report,
    /// or unavailable transport from masquerading as a shorter machine.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum InventoryReadError {
        /// The target has no live report slot/verb transport (including host
        /// builds, which never simulate capability authority).
        Unavailable,
        /// The bounded session could not be staged.
        Build,
        /// The trap transport did not return a committed outcome.
        Transport,
        /// The kernel refused the one-invocation observation.
        Refused(kore_abi::session::Status),
        /// A single-invocation completion named another failing index.
        MalformedCompletion,
        /// The result slot did not contain the exact expected inert report.
        MalformedReport,
    }

    /// Why a Hypervisor grant could not mint one owning guest space.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum GuestSpaceCreateError {
        NoSlot,
        Build,
        Transport,
        Refused(kore_abi::session::Status),
        MalformedCompletion,
        Unavailable,
    }

    /// Why a lease could not be installed into an authenticated GuestSpace.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum GuestShareError {
        InvalidShape,
        NoSlot,
        Build,
        Transport,
        Refused(kore_abi::session::Status),
        MalformedCompletion,
        Unavailable,
    }

    /// The three exact affine capabilities moved into a newly founded vCPU.
    ///
    /// They are bundled so every pre-commit refusal returns the complete
    /// construction set to the caller; no error path can silently drop one
    /// half of the protocol.
    #[derive(Debug)]
    pub struct GuestVcpuInputs {
        pub window: StashedLease,
        pub resume_recv: ChanRecvToken,
        pub exit_send: ChanSendToken,
    }

    /// Why a guest-vCPU founding did not return an authenticated task handle.
    #[derive(Debug)]
    pub enum GuestVcpuFoundError {
        NoSlot(GuestVcpuInputs),
        Build(GuestVcpuInputs),
        Transport(GuestVcpuInputs),
        Refused(kore_abi::session::Status, GuestVcpuInputs),
        MalformedCompletion(GuestVcpuInputs),
        Unavailable(GuestVcpuInputs),
    }

    #[cfg(any(test, target_os = "none"))]
    fn guest_region_rights(access: u8) -> Option<Rights> {
        use kore_abi::session::{SHARE_ACCESS_R, SHARE_ACCESS_W, SHARE_ACCESS_X};
        let allowed = SHARE_ACCESS_R | SHARE_ACCESS_W | SHARE_ACCESS_X;
        if access & !allowed != 0
            || access & SHARE_ACCESS_R == 0
            || access & SHARE_ACCESS_W != 0 && access & SHARE_ACCESS_X != 0
        {
            return None;
        }
        Some(if access & SHARE_ACCESS_W != 0 {
            crate::cap::GUEST_REGION_RW_RIGHTS
        } else {
            crate::cap::GUEST_REGION_RO_RIGHTS
        })
    }

    #[cfg(any(test, target_os = "none"))]
    #[inline]
    fn device_read_committed(
        committed: kore_abi::session::Committed,
    ) -> Result<(), InventoryReadError> {
        if committed.failing_index() != 0 {
            return Err(InventoryReadError::MalformedCompletion);
        }
        match committed.status() {
            kore_abi::session::Status::Ok => Ok(()),
            status => Err(InventoryReadError::Refused(status)),
        }
    }

    #[cfg(any(test, target_os = "none"))]
    #[inline]
    fn classify_owned_retirement(committed: kore_abi::session::Committed) -> RetireOutcome {
        match (committed.status(), committed.failing_index()) {
            (kore_abi::session::Status::Ok, 0) => RetireOutcome::Retired,
            (kore_abi::session::Status::Revoked, 0) => RetireOutcome::AlreadyDead,
            _ => RetireOutcome::Retryable,
        }
    }

    #[cfg(any(test, target_os = "none"))]
    #[inline]
    fn classify_chan_recv_probe(committed: kore_abi::session::Committed) -> ChanRecvLineage {
        match (committed.status(), committed.failing_index()) {
            (kore_abi::session::Status::Ok, 0) => ChanRecvLineage::Live,
            (kore_abi::session::Status::Revoked, 0) => ChanRecvLineage::Dead,
            _ => ChanRecvLineage::Retryable,
        }
    }

    #[cfg(any(test, target_os = "none"))]
    #[inline]
    fn classify_task_start(committed: kore_abi::session::Committed) -> TaskStartOutcome {
        use kore_abi::session::Status;

        if committed.failing_index() != 0 {
            return TaskStartOutcome::Refused(Status::Malformed);
        }
        match committed.status() {
            Status::Ok => TaskStartOutcome::Active,
            Status::Conflict | Status::WouldBlock => TaskStartOutcome::Retryable,
            Status::Revoked | Status::Exited => TaskStartOutcome::Terminal,
            status => TaskStartOutcome::Refused(status),
        }
    }

    /// Validate the kernel's arena-extension geometry before any userland
    /// visibility or allocator state moves.  The producer promises exactly
    /// one dense 256-record chunk at a chunk-indexed offset inside the
    /// founder-stamped structural reservation; re-prove every field here at
    /// the producer→consumer boundary.
    #[cfg(any(test, target_os = "none"))]
    fn arena_extension_span(
        ext_base: usize,
        report: kore_abi::session::GeomReport,
    ) -> Option<usize> {
        let rec_bytes = core::mem::size_of::<kore_abi::cap::CapRecord>();
        let chunk_bytes = (crate::entry::ARENA_SLOT_CEILING as usize).checked_mul(rec_bytes)?;
        let max_bytes = ((crate::entry::ARENA_EXT_SLOT_CEILING - crate::entry::ARENA_SLOT_CEILING)
            as usize)
            .checked_mul(rec_bytes)?;
        if report.len != chunk_bytes as u64 {
            return None;
        }
        let offset = report.base.checked_sub(ext_base as u64)?;
        if offset % chunk_bytes as u64 != 0 {
            return None;
        }
        let new_len = usize::try_from(offset.checked_add(report.len)?).ok()?;
        (new_len <= max_bytes).then_some(new_len)
    }

    /// **The metal claim-supply** (`res/USERLAND.md`; #336 rung 4) — mm/swap's
    /// lessor claims its
    /// [`role::MEM_SUPPLY`](kore_abi::task::role::MEM_SUPPLY) endowment (or
    /// the startup stub's distinct
    /// [`role::LINK_SUPPLY`](kore_abi::task::role::LINK_SUPPLY)) and learns
    /// its founder-seeded backing geometry so
    /// [`MemoryManager`](crate::service) can construct, through the LIVE
    /// session engine: one staged **no-arg `Derive`** naming the supply slot
    /// as the subject (the kernel's claim flavor), committed through the trap
    /// transport.  The answer is a bind-inert
    /// [`GeomReport`](kore_abi::session::GeomReport) — `(base, len)`, the
    /// geometry the frozen `CapRecord` deliberately doesn't carry — in the
    /// derived report slot; the **rights ride the supply record itself**,
    /// read off the R/O slots alias (the retired sentinel's 17-byte scratch
    /// report is dead).  Authentication is the engine's ordinary subject
    /// bind; any refusal is a fail-closed `None`.
    #[cfg(target_os = "none")]
    pub fn claim_supply(slot: SlotRef) -> Option<SupplyToken> {
        use kore_abi::cap::CapRecord;
        use kore_abi::session::GeomReport;

        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let report_slot = crate::entry::geometry::report_slot()?;
        let mut b = SessionBuilder::new();
        b.derive_raw(slot, Rights::empty(), 0, report_slot, None)
            .ok()?;
        let committed = TrapTransport::new().commit_now(&b).ok()?;
        if committed.status() != kore_abi::session::Status::Ok {
            return None;
        }
        // Decode the bind-inert report off the R/O slots alias.  SAFETY: the
        // founder mapped the slots alias R/O over this task's own records; the
        // kernel wrote the report before the trap returned.  Read volatile.
        let rptr = crate::entry::geometry::slot_record_va(report_slot)? as *const CapRecord;
        let g = GeomReport::decode(&unsafe { core::ptr::read_volatile(rptr) })?;
        // The rights ride the supply record itself — same alias, same safety.
        let sptr = crate::entry::geometry::slot_record_va(slot)? as *const CapRecord;
        let rights = unsafe { core::ptr::read_volatile(sptr) }.rights;
        Some(SupplyToken::possess(slot, g.base, g.len, rights))
    }

    /// Off the metal userland targets — host builds — the honest typed
    /// refusal (`res/USERLAND.md` §2.1): there is no trap to issue.
    #[cfg(not(target_os = "none"))]
    pub fn claim_supply(_slot: SlotRef) -> Option<SupplyToken> {
        None
    }

    /// Install the surface-backing arm (`role::SURFACE_POOL`) **whole** at
    /// the task's `geom::SURFACES` window (#578 slice 3) — the
    /// `geom::APERTURE` lesson applied to the pool: one fixed task-level
    /// mapping, never the view cursor, refuse-whole on any geometry
    /// disagreement.  Two gates before the map, both fail-closed:
    ///
    /// - the geometry table must carry a `SURFACES` window at least
    ///   [`SURFACE_POOL_BYTES`] long (a task the founder stamped no pool
    ///   window for — every task but the display row's thread 0 — refuses
    ///   here);
    /// - the arm's kernel-vouched span must BE the contract span
    ///   ([`SURFACE_POOL_BYTES`], the abi's cross-boundary geometry
    ///   contract) — a truncated pool is refused whole, never carved
    ///   short (granule arithmetic over a lying span would hand two
    ///   surfaces one backing).
    ///
    /// The returned view is the mapping's KEEPER: dropping it
    /// exact-unshares the arm, so the service holds it for its whole
    /// life.
    ///
    /// [`SURFACE_POOL_BYTES`]: kore_abi::task::SURFACE_POOL_BYTES
    #[cfg(target_os = "none")]
    pub fn map_surface_pool(pool: &SupplyToken) -> Option<MappedLeaseView> {
        let rt = crate::entry::thread_context()?;
        let (va, len) = rt.geom()?.window(kore_abi::task::geom::SURFACES)?;
        if len < kore_abi::task::SURFACE_POOL_BYTES {
            return None;
        }
        let (_, lease_len) = lease_geometry(pool.slot())?;
        if lease_len != kore_abi::task::SURFACE_POOL_BYTES {
            return None;
        }
        map_owned_lease_at_rw(pool.slot(), va, kore_abi::task::SURFACE_POOL_BYTES)
    }

    /// Off the metal userland targets — the honest typed refusal (no trap,
    /// no geometry table, no space to install into).
    #[cfg(not(target_os = "none"))]
    pub fn map_surface_pool(_pool: &SupplyToken) -> Option<MappedLeaseView> {
        None
    }

    /// Possess a service-granted supply while retaining the reservation that
    /// owns its dynamically selected landing slot.
    ///
    /// This is metal-architecture-neutral: the kernel-authored slot alias,
    /// geometry report, and capability session ABI are shared by x86-64,
    /// AArch64, and RISC-V. Device-specific consumers may remain gated to the
    /// architectures on which their backend exists.
    #[cfg(target_os = "none")]
    pub fn claim_returned_supply(
        returned: crate::cap::ReturnedCapability,
    ) -> Result<SupplyToken, crate::cap::ReturnedCapability> {
        use kore_abi::cap::CapRecord;
        use kore_abi::session::GeomReport;

        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        // A view's no-argument derive maps it and also returns geometry;
        // that report is a virtual address, not a supply's physical extent.
        // Authenticate the lessor species before issuing any such derive.
        if !returned.is_lease_with(Rights::ACCESS) {
            return Err(returned);
        }
        let Some(slot) = returned.slot() else {
            return Err(returned);
        };
        let Some(report_slot) = crate::entry::geometry::report_slot() else {
            return Err(returned);
        };
        let mut b = SessionBuilder::new();
        if b.derive_raw(slot, Rights::empty(), 0, report_slot, None)
            .is_err()
        {
            return Err(returned);
        }
        let Ok(committed) = TrapTransport::new().commit_now(&b) else {
            return Err(returned);
        };
        if committed.status() != kore_abi::session::Status::Ok {
            return Err(returned);
        }
        let Some(rptr) = crate::entry::geometry::slot_record_va(report_slot) else {
            return Err(returned);
        };
        // SAFETY: the task's read-only arena alias is kernel-authored and the
        // commit above completed before either volatile read.
        let Some(geometry) =
            GeomReport::decode(&unsafe { core::ptr::read_volatile(rptr as *const CapRecord) })
        else {
            return Err(returned);
        };
        let Some(sptr) = crate::entry::geometry::slot_record_va(slot) else {
            return Err(returned);
        };
        // SAFETY: same read-only arena-alias contract as the report above.
        let rights = unsafe { core::ptr::read_volatile(sptr as *const CapRecord) }.rights;
        Ok(SupplyToken::possess_returned(
            slot,
            geometry.base,
            geometry.len,
            rights,
            returned,
        ))
    }

    #[cfg(not(target_os = "none"))]
    pub fn claim_returned_supply(
        returned: crate::cap::ReturnedCapability,
    ) -> Result<SupplyToken, crate::cap::ReturnedCapability> {
        Err(returned)
    }

    /// **Grow this task's own cap arena by one chunk** (slice E) — the
    /// runtime half of `DERIVE_KIND_ARENA_EXTEND`, the `grow_pool` shape:
    /// an [`OwnedSlot::reserve`](crate::cap::OwnedSlot) that answers
    /// `None` can extend and retry instead of treating the founding array
    /// as a wall.
    ///
    /// One sole-invocation session on the task's own space self-view
    /// (the kernel takes START's rule), then the bookkeeping the kernel's
    /// [`GeomReport`](kore_abi::session::GeomReport) authorizes: validate
    /// the exact chunk shape and structural bound; open the persistent-name
    /// bitmap over exactly the newly-visible interval; then publish the live
    /// aliased span.  Bitmap-before-length means an allocator can never see
    /// a slot before its reservation bit is ready, and replayed reports fold
    /// monotonically without returning a smaller total.
    ///
    /// Returns the new live slot total, or `None` fail-closed: no stamped
    /// extension window (spawned threads; pre-slice-E foundings), no
    /// report slot, a refused session (a thread's 64-slot arena refuses
    /// structurally; allocation pressure answers retryable), or a report
    /// that decodes to nothing.
    #[cfg(target_os = "none")]
    pub fn extend_arena() -> Option<u32> {
        use kore_abi::cap::CapRecord;
        use kore_abi::session::GeomReport;

        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        // The cheap local gates first — no trap for a task that cannot
        // possibly extend.
        let ext_base = crate::entry::geometry::slots_ext_va()?;
        let self_view = crate::entry::geometry::self_space_slot()?;
        let report_slot = crate::entry::geometry::report_slot()?;

        let mut b = SessionBuilder::new();
        b.extend_arena(self_view, report_slot).ok()?;
        let committed = TrapTransport::new().commit_now(&b).ok()?;
        if committed.status() != kore_abi::session::Status::Ok {
            return None;
        }
        // Decode the bind-inert alias report off the R/O slots alias.
        // SAFETY: the founder mapped the slots alias R/O over this task's
        // own records; the kernel wrote the report before the trap
        // returned.  Read volatile.
        let rptr = crate::entry::geometry::slot_record_va(report_slot)? as *const CapRecord;
        let g = GeomReport::decode(&unsafe { core::ptr::read_volatile(rptr) })?;

        // The aliased span runs dense from the stamped base through the
        // reported chunk.  Re-prove the producer's exact chunk geometry and
        // structural ceiling before making any of it userland-visible.
        let new_len = arena_extension_span(ext_base, g)?;
        let rec_bytes = core::mem::size_of::<CapRecord>();
        let old_len = crate::entry::geometry::slots_ext_len();
        let base = crate::entry::ARENA_SLOT_CEILING as usize;
        if new_len > old_len {
            // Newly-visible interval only: bits below `old` may already be
            // acquired names, and re-opening one would double-allocate it.
            if !crate::entry::geometry::extend_cap_slots(
                base + old_len / rec_bytes,
                base + new_len / rec_bytes,
            ) {
                return None;
            }
            // Publish last: readers admitted by this length now observe both
            // the kernel mappings and the already-open allocator interval.
            crate::entry::geometry::publish_slots_ext(new_len);
        }
        let visible_len = old_len.max(new_len);
        Some((base + visible_len / rec_bytes) as u32)
    }

    /// Off the metal userland targets — the honest typed refusal.
    #[cfg(not(target_os = "none"))]
    pub fn extend_arena() -> Option<u32> {
        None
    }
    /// **Claim the lessor's region grant** (`role::MEM_REGION`, slice D) —
    /// mm's possession of its minting authority: authenticate the
    /// founder-delivered `PhysRegion` record at `slot` and wrap the name
    /// (the [`claim_device`] shape; no geometry report — the region's own
    /// frame bitmap is the ledger every minted lease claims against).
    #[cfg(target_os = "none")]
    pub fn claim_region(slot: SlotRef) -> Option<crate::cap::RegionToken> {
        claim_kind::<crate::cap::RegionKind>(slot)
    }

    /// Off the metal userland targets — the honest typed refusal.
    #[cfg(not(target_os = "none"))]
    pub fn claim_region(_slot: SlotRef) -> Option<crate::cap::RegionToken> {
        None
    }

    #[cfg(target_os = "none")]
    pub fn claim_device(slot: SlotRef) -> Option<DeviceToken> {
        let rights = authority_record(slot, kore_abi::cap::ResourceKind::PciDevice)?;
        Some(DeviceToken::possess(slot, rights))
    }

    /// Possess a broker-delivered PCI device without separating its live arena
    /// name from the runtime reservation that owns that name.
    #[cfg(target_os = "none")]
    pub fn claim_returned_device(
        returned: crate::cap::ReturnedCapability,
    ) -> Result<DeviceToken, crate::cap::ReturnedCapability> {
        let Some(slot) = returned.slot() else {
            return Err(returned);
        };
        let Some(rights) = authority_record(slot, kore_abi::cap::ResourceKind::PciDevice) else {
            return Err(returned);
        };
        Ok(DeviceToken::possess_returned(slot, rights, returned))
    }

    #[cfg(not(target_os = "none"))]
    pub fn claim_returned_device(
        returned: crate::cap::ReturnedCapability,
    ) -> Result<DeviceToken, crate::cap::ReturnedCapability> {
        Err(returned)
    }

    /// Possess an exec-returned task handle while retaining ownership of its
    /// dynamically rotated landing slot.
    #[cfg(target_os = "none")]
    pub fn claim_returned_task(
        returned: crate::cap::ReturnedCapability,
        expected_id: u64,
        expected_rights: Rights,
    ) -> Result<TaskToken, crate::cap::ReturnedCapability> {
        let Some(slot) = returned.slot() else {
            return Err(returned);
        };
        let Some((id, rights)) = task_record(slot) else {
            return Err(returned);
        };
        if id != expected_id || rights != expected_rights {
            return Err(returned);
        }
        Ok(TaskToken::possess_returned(slot, id, rights, returned))
    }

    /// Possess a task handle published directly by the same atomic Found.
    /// Identity comes from the authenticated record, never a numeric reply.
    #[cfg(target_os = "none")]
    pub fn claim_direct_returned_task(
        returned: crate::cap::ReturnedCapability,
        expected_rights: Rights,
    ) -> Result<TaskToken, (crate::cap::ReturnedCapability, DirectTaskClaimError)> {
        let Some(slot) = returned.slot() else {
            return Err((returned, DirectTaskClaimError::NotLive));
        };
        let Some((id, rights)) = task_record(slot) else {
            return Err((returned, DirectTaskClaimError::NotTask));
        };
        if rights != expected_rights {
            return Err((returned, DirectTaskClaimError::Rights));
        }
        Ok(TaskToken::possess_returned(slot, id, rights, returned))
    }

    #[cfg(not(target_os = "none"))]
    pub fn claim_returned_task(
        returned: crate::cap::ReturnedCapability,
        _expected_id: u64,
        _expected_rights: kore_abi::cap::Rights,
    ) -> Result<TaskToken, crate::cap::ReturnedCapability> {
        Err(returned)
    }

    #[cfg(not(target_os = "none"))]
    pub fn claim_direct_returned_task(
        returned: crate::cap::ReturnedCapability,
        _expected_rights: kore_abi::cap::Rights,
    ) -> Result<TaskToken, (crate::cap::ReturnedCapability, DirectTaskClaimError)> {
        Err((returned, DirectTaskClaimError::NotTask))
    }

    #[cfg(not(target_os = "none"))]
    pub fn claim_device(_slot: SlotRef) -> Option<DeviceToken> {
        None
    }

    /// Take possession of a **platform-device** endowment (#417) — the
    /// [`claim_device`] twin for a firmware-described non-PCI device.
    ///
    /// The kind gate is the whole point: a row endowed with a PCI
    /// function refuses here rather than succeeding and then failing at
    /// its first port derive, so a mis-selected manifest row is a bind
    /// failure at bring-up.
    #[cfg(target_os = "none")]
    pub fn claim_platform_device(slot: SlotRef) -> Option<crate::cap::PlatformToken> {
        let rights = authority_record(slot, kore_abi::cap::ResourceKind::PlatformDevice)?;
        Some(crate::cap::PlatformToken::possess(slot, rights))
    }

    /// Possess a broker-delivered platform device while retaining ownership of
    /// its dynamically rotated landing slot.
    #[cfg(target_os = "none")]
    pub fn claim_returned_platform_device(
        returned: crate::cap::ReturnedCapability,
    ) -> Result<crate::cap::PlatformToken, crate::cap::ReturnedCapability> {
        let Some(slot) = returned.slot() else {
            return Err(returned);
        };
        let Some(rights) = authority_record(slot, kore_abi::cap::ResourceKind::PlatformDevice)
        else {
            return Err(returned);
        };
        Ok(crate::cap::PlatformToken::possess_returned(
            slot, rights, returned,
        ))
    }

    #[cfg(not(target_os = "none"))]
    pub fn claim_returned_platform_device(
        returned: crate::cap::ReturnedCapability,
    ) -> Result<crate::cap::PlatformToken, crate::cap::ReturnedCapability> {
        Err(returned)
    }

    #[cfg(not(target_os = "none"))]
    pub fn claim_platform_device(_slot: SlotRef) -> Option<crate::cap::PlatformToken> {
        None
    }

    /// **The rights of an affine owner of a device-tree root** — a broker
    /// row's `RootComplex`/`PlatformBus` endowment and each device it hands
    /// to a driver.
    ///
    /// The owner can use and program the resource, derive its bounded children,
    /// and retire its own subtree. `REVOKE` never reaches an ancestor: giving
    /// it to the broker cannot retire the founder's primordial root, and
    /// giving it to a driver cannot retire the broker's own grant.
    pub const DEVICE_OWNER_RIGHTS: kore_abi::cap::Rights = kore_abi::cap::Rights::ACCESS
        .union(kore_abi::cap::Rights::MODIFY)
        .union(kore_abi::cap::Rights::DERIVE)
        .union(kore_abi::cap::Rights::REVOKE);

    /// **The rights every device-rooted child token carries** — what
    /// [`claim_ports`] and [`establish_line`] ask for.
    ///
    /// `ACCESS` to use the thing, `REVOKE` so the affine token can
    /// retire itself on drop (the `OwnedSlot` teardown leg). Named
    /// rather than repeated at each call site because Σ5 clamps a
    /// derived child to a **subset of its parent's** rights, so anyone
    /// endowing a device to a worker has to grant at least these — and
    /// the first version of the i8042 relay's endowment did not, which
    /// made every port claim `Denied` and left the keyboard dead with
    /// only a wire-level refusal to say so. A constant both sides can
    /// name turns that into something a test can assert.
    pub const DEVICE_CHILD_RIGHTS: kore_abi::cap::Rights =
        kore_abi::cap::Rights::ACCESS.union(kore_abi::cap::Rights::REVOKE);

    /// Claim the device's **I/O ports** (#416) — an arg-free derive, so
    /// the grant covers every range the firmware attributed to this
    /// device and nothing else.
    ///
    /// Arg-free rather than parameterized because the claim is
    /// **claim-once**: a device yields exactly one live port grant, and
    /// an i8042 needs two disjoint ranges (`0x60` and `0x64`), which no
    /// single sub-range request can name.  A caller that asked for a span
    /// bridging the gap would be refused by the kernel's
    /// `declares_ports` gate, so the honest verb takes what firmware
    /// declared.
    ///
    /// The returned grant must live on the **thread that performs the
    /// I/O** — see [`PortGrant`](crate::cap::PortGrant) for why the
    /// bitmap is keyed on the task rather than the address space.
    ///
    /// x86-only among the metal targets: I/O port space does not exist
    /// on the other architectures kore targets, so the claim refuses
    /// there (the `PortGrant` type itself is uninhabitable off x86).
    #[cfg(all(target_os = "none", target_arch = "x86_64"))]
    pub fn claim_ports(dev: &crate::cap::PlatformToken) -> Option<crate::cap::PortGrant> {
        use kore_abi::cap::ResourceKind;

        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let owned = crate::cap::OwnedSlot::reserve()?;
        let mut b = SessionBuilder::new();
        b.derive_raw(
            dev.slot(),
            DEVICE_CHILD_RIGHTS,
            (ResourceKind::IoPortGrant as u16) as u8,
            owned.slot(),
            None,
        )
        .ok()?;
        let committed = TrapTransport::new().commit_now(&b).ok()?;
        if committed.status() != kore_abi::session::Status::Ok {
            return None;
        }
        Some(crate::cap::PortGrant::new(owned))
    }

    #[cfg(not(all(target_os = "none", target_arch = "x86_64")))]
    pub fn claim_ports(_dev: &crate::cap::PlatformToken) -> Option<crate::cap::PortGrant> {
        None
    }

    /// Retire a port grant — the teardown leg.  The ports shut on the
    /// holder's next dispatch without any further act.
    pub fn release_ports(grant: crate::cap::PortGrant) -> bool {
        grant.retire()
    }

    /// **The rights a send end needs to carry a notice** — what
    /// [`post_notice`] asks of the slot it is handed.
    ///
    /// `ACCESS`, and deliberately nothing else. The kernel gates every
    /// `Rendezvous` on `ACCESS` over the bound end (use of the channel),
    /// and the notice-mode arm adds no second demand: a notice derives
    /// nothing, so it makes no `DERIVE` demand and treats a non-empty
    /// rights request as a shape error.
    ///
    /// Named for the same reason [`DEVICE_CHILD_RIGHTS`] is — a task
    /// endowing a worker thread with a send end has to grant at least
    /// this — but here the **ceiling** matters as much as the floor. Σ5
    /// clamps a derived child to a subset of its parent's rights, and a
    /// service's self-kick is carved with
    /// [`SEND_RIGHTS`](crate::svc::SEND_RIGHTS); asking for anything
    /// outside that set refuses the derive rather than the send, which
    /// fails the *spawn* and takes the whole row with it.
    pub const NOTICE_RIGHTS: kore_abi::cap::Rights = kore_abi::cap::Rights::ACCESS;

    /// One **synchronous** notice-mode send — the worker-thread sibling
    /// of [`ChanTx::send`](crate::chan::ChanTx::send).
    ///
    /// The `send` slot must carry [`NOTICE_RIGHTS`].
    ///
    /// A relay thread parks on its interrupt route in a plain loop
    /// rather than under an executor, so it has no `.await` to spend on
    /// a kick. The session is identical; only the transport differs.
    ///
    /// `false` on any refusal, including `WouldBlock` — a droppable
    /// kick, exactly as the async form's contract states: the sender's
    /// queue write happens-before this notice, so a receiver that
    /// consumed a post is awake and will drain every source after it
    /// re-posts.
    #[cfg(target_os = "none")]
    pub fn post_notice(send: SlotRef, cookie: u64, payload: u64) -> bool {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let mut b = SessionBuilder::new();
        if b.notice(send, cookie, payload).is_err() {
            return false;
        }
        TrapTransport::new()
            .commit_now(&b)
            .is_ok_and(|c| c.status() == kore_abi::session::Status::Ok)
    }

    #[cfg(not(target_os = "none"))]
    pub fn post_notice(_send: SlotRef, _cookie: u64, _payload: u64) -> bool {
        false
    }

    /// Establish the device's **legacy interrupt line** (#418) — the
    /// [`establish_interrupt`] twin for a firmware-described device,
    /// yielding the same [`RouteToken`] a driver parks on.
    ///
    /// Arg-free, and that is a design statement rather than an omission:
    /// polarity and trigger come from the firmware description the device
    /// carries (an ACPI interrupt-source override where one exists, else
    /// its own `_CRS`).  A driver cannot ask for a line to be
    /// level-triggered because it would prefer that.
    #[cfg(target_os = "none")]
    pub fn establish_line(dev: &crate::cap::PlatformToken) -> Option<RouteToken> {
        use kore_abi::cap::ResourceKind;

        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let owned = crate::cap::OwnedSlot::reserve()?;
        let mut b = SessionBuilder::new();
        b.derive_raw(
            dev.slot(),
            DEVICE_CHILD_RIGHTS,
            (ResourceKind::IrqRoute as u16) as u8,
            owned.slot(),
            None,
        )
        .ok()?;
        let committed = TrapTransport::new().commit_now(&b).ok()?;
        if committed.status() != kore_abi::session::Status::Ok {
            return None;
        }
        // One line, one vector — a legacy pin has no block to request.
        Some(RouteToken::new(owned, 1))
    }

    #[cfg(not(target_os = "none"))]
    pub fn establish_line(_dev: &crate::cap::PlatformToken) -> Option<RouteToken> {
        None
    }

    /// Establish this PCI controller's firmware-resolved INTx line. No line
    /// number or GPIO identity crosses the capability boundary.
    #[cfg(target_os = "none")]
    pub fn establish_pci_line(dev: &DeviceToken) -> Option<RouteToken> {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let owned = crate::cap::OwnedSlot::reserve()?;
        let mut b = SessionBuilder::new();
        b.derive_pci_line_route(dev.slot(), DEVICE_CHILD_RIGHTS, owned.slot())
            .ok()?;
        let committed = TrapTransport::new().commit_now(&b).ok()?;
        if committed.status() != kore_abi::session::Status::Ok {
            return None;
        }
        Some(RouteToken::new(owned, 1))
    }

    #[cfg(not(target_os = "none"))]
    pub fn establish_pci_line(_dev: &DeviceToken) -> Option<RouteToken> {
        None
    }

    /// Establish the firmware-correlated GPIO-ready route for one initialized
    /// HID child behind this exact controller capability.
    #[cfg(target_os = "none")]
    pub fn establish_i2c_hid_gpio<C: crate::cap::I2cHidSubject>(
        dev: &C,
        child: u32,
    ) -> Option<RouteToken> {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let owned = crate::cap::OwnedSlot::reserve()?;
        let mut b = SessionBuilder::new();
        b.derive_i2c_hid_gpio_route(
            dev.i2c_subject_slot(),
            DEVICE_CHILD_RIGHTS,
            child,
            owned.slot(),
        )
        .ok()?;
        let committed = TrapTransport::new().commit_now(&b).ok()?;
        if committed.status() != kore_abi::session::Status::Ok {
            return None;
        }
        Some(RouteToken::new(owned, 1))
    }

    #[cfg(not(target_os = "none"))]
    pub fn establish_i2c_hid_gpio<C: crate::cap::I2cHidSubject>(
        _dev: &C,
        _child: u32,
    ) -> Option<RouteToken> {
        None
    }
    /// Read and classify an authority-bearing record from the task's read-only
    /// cap-arena alias.  This does not authenticate in userland (which has no
    /// epoch key); the first effect names the slot again and the kernel performs
    /// the authoritative bind.  The read is therefore possession, not minting.
    #[cfg(target_os = "none")]
    fn authority_record_full(
        slot: SlotRef,
        kind: kore_abi::cap::ResourceKind,
    ) -> Option<kore_abi::cap::CapRecord> {
        use kore_abi::cap::CapRecord;

        let ptr = crate::entry::geometry::slot_record_va(slot)? as *const CapRecord;
        // SAFETY: the founder maps this task's arena as a read-only user alias;
        // the kernel is the sole writer and the record is naturally aligned.
        let rec = unsafe { core::ptr::read_volatile(ptr) };
        (rec.kind == kind as u16 && !rec.cid.is_null()).then_some(rec)
    }

    #[cfg(target_os = "none")]
    fn authority_record(slot: SlotRef, kind: kore_abi::cap::ResourceKind) -> Option<Rights> {
        authority_record_full(slot, kind).map(|record| record.rights)
    }

    /// Read the rights carried by a held capability of exactly `kind`.
    ///
    /// This is a possession-side metadata query only.  The read-only arena
    /// alias prevents userland fabrication, and every later effect re-binds
    /// the same slot in the kernel.  It exists for task-local namespaces that
    /// need to present effective rights without inventing a parallel policy
    /// bitmap.
    #[cfg(target_os = "none")]
    pub fn capability_rights(slot: SlotRef, kind: kore_abi::cap::ResourceKind) -> Option<Rights> {
        authority_record(slot, kind)
    }

    /// Observe the identity of a held capability before its affine token moves
    /// into a driver binding. This number carries no authority: a later broker
    /// request must authenticate its original client face and the kernel must
    /// match the original issuer and committed claim generation independently.
    #[cfg(target_os = "none")]
    pub fn capability_cid(
        slot: SlotRef,
        kind: kore_abi::cap::ResourceKind,
    ) -> Option<kore_abi::cap::Cid> {
        authority_record_full(slot, kind).map(|record| record.cid)
    }

    /// Host builds have no task arena whose held record can be observed.
    #[cfg(not(target_os = "none"))]
    pub fn capability_cid(
        _slot: SlotRef,
        _kind: kore_abi::cap::ResourceKind,
    ) -> Option<kore_abi::cap::Cid> {
        None
    }

    /// Read the immutable kernel resource identity carried by a held
    /// capability of exactly `kind`, after checking that it contains
    /// `required` rights.
    ///
    /// This is deliberately not a launch authorization by itself: callers
    /// may use the number only to correlate an already-presented capability
    /// with an immutable service-owned binding. Every effect still names the
    /// original slot and is re-authenticated by the kernel.
    #[cfg(target_os = "none")]
    pub fn capability_scope_rid(
        slot: SlotRef,
        kind: kore_abi::cap::ResourceKind,
        required: Rights,
    ) -> Option<u64> {
        authority_record_full(slot, kind)
            .filter(|record| record.rights.contains(required))
            .map(|record| record.scope_rid)
    }

    /// Resource-local id encoded in the authenticated scope rid. Channel send
    /// and receive faces have different kinds but the same underlying id;
    /// this projection is therefore the stable per-channel binding key.
    #[cfg(target_os = "none")]
    pub fn capability_scope_id(
        slot: SlotRef,
        kind: kore_abi::cap::ResourceKind,
        required: Rights,
    ) -> Option<u32> {
        let rid = capability_scope_rid(slot, kind, required)?;
        let mut compact = 0u32;
        let mut bit = 0u32;
        while bit < 32 {
            compact |= (((rid >> (bit * 2 + 1)) & 1) as u32) << bit;
            bit += 1;
        }
        Some(compact)
    }

    #[cfg(not(target_os = "none"))]
    pub fn capability_rights(
        _slot: SlotRef,
        _kind: kore_abi::cap::ResourceKind,
    ) -> Option<kore_abi::cap::Rights> {
        None
    }

    #[cfg(not(target_os = "none"))]
    pub fn capability_scope_rid(
        _slot: SlotRef,
        _kind: kore_abi::cap::ResourceKind,
        _required: kore_abi::cap::Rights,
    ) -> Option<u64> {
        None
    }

    #[cfg(not(target_os = "none"))]
    pub fn capability_scope_id(
        _slot: SlotRef,
        _kind: kore_abi::cap::ResourceKind,
        _required: kore_abi::cap::Rights,
    ) -> Option<u32> {
        None
    }

    /// **The plain-authority claim** — authenticate the founder-delivered
    /// record at `slot` against its kind, then wrap the name: the whole of
    /// what every `claim_*` verb below does, once, over
    /// [`CapKind`](crate::cap::CapKind).  The named verbs stay (each is a
    /// distinct authority with its own doc and its own token type); what
    /// they no longer each carry is this body.
    #[cfg(target_os = "none")]
    fn claim_kind<K: crate::cap::CapKind>(slot: SlotRef) -> Option<crate::cap::Token<K>> {
        let rights = authority_record(slot, K::RESOURCE)?;
        Some(crate::cap::Token::possess(slot, rights))
    }

    #[cfg(target_os = "none")]
    fn task_record(slot: SlotRef) -> Option<(u64, Rights)> {
        use kore_abi::cap::{CapRecord, ResourceKind};

        let ptr = crate::entry::geometry::slot_record_va(slot)? as *const CapRecord;
        // SAFETY: the founder maps this task's arena as a read-only user alias;
        // the kernel is the sole writer and the record is naturally aligned.
        let rec = unsafe { core::ptr::read_volatile(ptr) };
        (rec.kind == ResourceKind::Task as u16 && !rec.cid.is_null())
            .then_some((rec.cid.resource() as u64, rec.rights))
    }

    #[cfg(target_os = "none")]
    pub fn claim_exec(slot: SlotRef) -> Option<ExecToken> {
        claim_kind::<crate::cap::ExecKind>(slot)
    }

    #[cfg(not(target_os = "none"))]
    pub fn claim_exec(_slot: SlotRef) -> Option<ExecToken> {
        None
    }

    #[cfg(target_os = "none")]
    pub fn claim_virt_memory(slot: SlotRef) -> Option<VirtMemoryToken> {
        claim_kind::<crate::cap::VirtMemoryKind>(slot)
    }

    #[cfg(not(target_os = "none"))]
    pub fn claim_virt_memory(_slot: SlotRef) -> Option<VirtMemoryToken> {
        None
    }

    /// Claim the hypervisor service's machine-virtualization endowment.
    #[cfg(target_os = "none")]
    pub fn claim_hypervisor(slot: SlotRef) -> Option<HypervisorToken> {
        claim_kind::<crate::cap::HypervisorKind>(slot)
    }

    #[cfg(not(target_os = "none"))]
    pub fn claim_hypervisor(_slot: SlotRef) -> Option<HypervisorToken> {
        None
    }

    /// Mint one owning coherent guest-physical address space. The returned
    /// token owns both the live kernel capability and its dynamic arena name.
    #[cfg(target_os = "none")]
    pub fn create_guest_space(
        hypervisor: &HypervisorToken,
    ) -> Result<GuestSpaceToken, GuestSpaceCreateError> {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let mut owned = crate::cap::OwnedSlot::reserve().ok_or(GuestSpaceCreateError::NoSlot)?;
        let mut request = SessionBuilder::new();
        request
            .derive_raw(
                hypervisor.slot(),
                crate::cap::GUEST_SPACE_RIGHTS,
                kore_abi::cap::ResourceKind::GuestSpace as u8,
                owned.slot(),
                None,
            )
            .map_err(|_| GuestSpaceCreateError::Build)?;
        let committed = TrapTransport::new()
            .commit_now(&request)
            .map_err(|_| GuestSpaceCreateError::Transport)?;
        if committed.status() == kore_abi::session::Status::Ok && committed.failing_index() == 0 {
            return Ok(GuestSpaceToken::new(owned));
        }
        if committed.failing_index() == 0
            && committed.raw() as u16 <= kore_abi::session::Status::AltDelivery as u16
        {
            return Err(GuestSpaceCreateError::Refused(committed.status()));
        }
        owned.quarantine_unknown_publish();
        Err(GuestSpaceCreateError::MalformedCompletion)
    }

    #[cfg(not(target_os = "none"))]
    pub fn create_guest_space(
        _hypervisor: &HypervisorToken,
    ) -> Result<GuestSpaceToken, GuestSpaceCreateError> {
        Err(GuestSpaceCreateError::Unavailable)
    }

    /// Install one founder-delivered RAM lease into `space` at a
    /// guest-physical address.
    ///
    /// Both arguments stay borrowed: the new GuestRegion is parented only by
    /// the lease, while the guest space is an authenticated destination and
    /// commit-time read dependency.
    #[cfg(target_os = "none")]
    pub fn share_into_guest(
        lease: &LeaseToken,
        space: &GuestSpaceToken,
        gpa: u64,
        bytes: u64,
        access: u8,
    ) -> Result<GuestRegionToken, GuestShareError> {
        share_guest_slot(lease.slot(), space, gpa, bytes, access)
    }

    /// Install one runtime-carved RAM lease into `space` at a guest-physical
    /// address.
    ///
    /// This is the service-facing sibling of [`share_into_guest`]. A service
    /// normally receives a broad [`SupplyToken`], carves a bounded
    /// [`StashedLease`] for one guest, and shares only that child lease. The
    /// raw arena slot remains sealed inside this module.
    #[cfg(target_os = "none")]
    pub fn share_stashed_into_guest(
        lease: &StashedLease,
        space: &GuestSpaceToken,
        gpa: u64,
        bytes: u64,
        access: u8,
    ) -> Result<GuestRegionToken, GuestShareError> {
        let lease = lease.slot().ok_or(GuestShareError::InvalidShape)?;
        share_guest_slot(lease, space, gpa, bytes, access)
    }

    /// Found one dormant, ordinary scheduler task that runs a guest vCPU.
    ///
    /// The guest-space owner remains with the service. The window and both
    /// notification endpoints move atomically into the vCPU only when the
    /// kernel publishes the lifecycle handle. The caller must then invoke
    /// [`TaskToken::start`]; founding itself never makes a half-assembled
    /// vCPU runnable.
    #[cfg(target_os = "none")]
    pub fn found_guest_vcpu(
        space: &GuestSpaceToken,
        cfg: kore_abi::guest::GuestVcpuConfig,
        inputs: GuestVcpuInputs,
    ) -> Result<TaskToken, GuestVcpuFoundError> {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let mut handle = match crate::cap::OwnedSlot::reserve() {
            Some(handle) => handle,
            None => return Err(GuestVcpuFoundError::NoSlot(inputs)),
        };
        let GuestVcpuInputs {
            window,
            resume_recv,
            exit_send,
        } = inputs;
        let Some(window_slot) = window.slot() else {
            return Err(GuestVcpuFoundError::Build(GuestVcpuInputs {
                window,
                resume_recv,
                exit_send,
            }));
        };
        let mut request = SessionBuilder::new();
        if request
            .found_guest(
                space.slot(),
                cfg.entry,
                cfg.stack,
                cfg.page_table_root,
                window_slot,
                resume_recv.slot(),
                exit_send.slot(),
                handle.slot(),
            )
            .is_err()
        {
            return Err(GuestVcpuFoundError::Build(GuestVcpuInputs {
                window,
                resume_recv,
                exit_send,
            }));
        }
        let committed = match TrapTransport::new().commit_now(&request) {
            Ok(committed) => committed,
            Err(_) => {
                handle.quarantine_unknown_publish();
                return Err(GuestVcpuFoundError::Transport(GuestVcpuInputs {
                    window,
                    resume_recv,
                    exit_send,
                }));
            }
        };
        if committed.status() != kore_abi::session::Status::Ok || committed.failing_index() != 0 {
            if committed.failing_index() == 0
                && committed.raw() as u16 <= kore_abi::session::Status::AltDelivery as u16
            {
                return Err(GuestVcpuFoundError::Refused(
                    committed.status(),
                    GuestVcpuInputs {
                        window,
                        resume_recv,
                        exit_send,
                    },
                ));
            }
            handle.quarantine_unknown_publish();
            return Err(GuestVcpuFoundError::MalformedCompletion(GuestVcpuInputs {
                window,
                resume_recv,
                exit_send,
            }));
        }
        let Some((id, rights)) = task_record(handle.slot()) else {
            handle.quarantine_unknown_publish();
            return Err(GuestVcpuFoundError::MalformedCompletion(GuestVcpuInputs {
                window,
                resume_recv,
                exit_send,
            }));
        };
        if rights != crate::cap::GUEST_VCPU_HANDLE_RIGHTS {
            handle.quarantine_unknown_publish();
            return Err(GuestVcpuFoundError::MalformedCompletion(GuestVcpuInputs {
                window,
                resume_recv,
                exit_send,
            }));
        }

        window.release_after_proven_move();
        resume_recv.release_after_proven_move();
        exit_send.release_after_proven_move();
        Ok(TaskToken::possess_owned(handle, id, rights))
    }

    #[cfg(not(target_os = "none"))]
    pub fn found_guest_vcpu(
        _space: &GuestSpaceToken,
        _cfg: kore_abi::guest::GuestVcpuConfig,
        inputs: GuestVcpuInputs,
    ) -> Result<TaskToken, GuestVcpuFoundError> {
        Err(GuestVcpuFoundError::Unavailable(inputs))
    }

    #[cfg(target_os = "none")]
    fn share_guest_slot(
        lease: SlotRef,
        space: &GuestSpaceToken,
        gpa: u64,
        bytes: u64,
        access: u8,
    ) -> Result<GuestRegionToken, GuestShareError> {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let rights = guest_region_rights(access).ok_or(GuestShareError::InvalidShape)?;
        if bytes == 0 || gpa.checked_add(bytes).is_none() {
            return Err(GuestShareError::InvalidShape);
        }
        let mut owned = crate::cap::OwnedSlot::reserve().ok_or(GuestShareError::NoSlot)?;
        let mut request = SessionBuilder::new();
        request
            .share_guest(
                lease,
                rights,
                space.slot(),
                gpa,
                bytes,
                access,
                owned.slot(),
            )
            .map_err(|_| GuestShareError::Build)?;
        let committed = TrapTransport::new()
            .commit_now(&request)
            .map_err(|_| GuestShareError::Transport)?;
        if committed.status() == kore_abi::session::Status::Ok && committed.failing_index() == 0 {
            return Ok(GuestRegionToken::new(owned, rights));
        }
        if committed.failing_index() == 0
            && committed.raw() as u16 <= kore_abi::session::Status::AltDelivery as u16
        {
            return Err(GuestShareError::Refused(committed.status()));
        }
        owned.quarantine_unknown_publish();
        Err(GuestShareError::MalformedCompletion)
    }

    #[cfg(not(target_os = "none"))]
    pub fn share_into_guest(
        _lease: &LeaseToken,
        _space: &GuestSpaceToken,
        _gpa: u64,
        _bytes: u64,
        _access: u8,
    ) -> Result<GuestRegionToken, GuestShareError> {
        Err(GuestShareError::Unavailable)
    }

    #[cfg(not(target_os = "none"))]
    pub fn share_stashed_into_guest(
        _lease: &StashedLease,
        _space: &GuestSpaceToken,
        _gpa: u64,
        _bytes: u64,
        _access: u8,
    ) -> Result<GuestRegionToken, GuestShareError> {
        Err(GuestShareError::Unavailable)
    }

    /// Claim the launcher's bounded, self-kind `ChannelSpace` grant.
    ///
    /// This is an exact-rights possession boundary. A primordial or otherwise
    /// over-broad record is not silently accepted as the launcher grant, and a
    /// weaker record cannot become a partially working mint client.
    #[cfg(target_os = "none")]
    pub fn claim_channel_space(slot: SlotRef) -> Option<ChannelSpaceToken> {
        let rights = authority_record(slot, kore_abi::cap::ResourceKind::ChannelSpace)?;
        exact_channel_space_rights(rights).then(|| ChannelSpaceToken::possess(slot, rights))
    }

    /// Off metal there is no authenticated cap-arena alias to possess.
    #[cfg(not(target_os = "none"))]
    pub fn claim_channel_space(_slot: SlotRef) -> Option<ChannelSpaceToken> {
        None
    }

    /// Mint one complete rendezvous pair in a single atomic session.
    ///
    /// Three dynamic arena names are reserved first. The staged SSA handles
    /// stay entirely inside this function; on commit each becomes an affine
    /// typed owner coupled to its reservation, on a proved refusal all three
    /// reservations reopen, and an impossible completion quarantines all
    /// three rather than guessing at occupancy.
    #[cfg(target_os = "none")]
    pub fn mint_channel(space: &ChannelSpaceToken) -> Result<ChannelPair, ChannelMintError> {
        mint_channel_with_rights(
            space,
            crate::cap::CHANNEL_ENDPOINT_RIGHTS,
            crate::cap::CHANNEL_ENDPOINT_RIGHTS,
        )
    }

    /// Mint one complete rendezvous pair with independently attenuated
    /// endpoint authority in a single atomic session.
    ///
    /// This is the preferred boundary when one endpoint will be moved into a
    /// less-trusted task: derive the final leaf with its exact rights during
    /// the mint, then move that affine leaf exactly. This avoids retaining an
    /// intermediate endpoint pivot merely to attenuate it after the fact.
    #[cfg(target_os = "none")]
    pub fn mint_channel_with_rights(
        space: &ChannelSpaceToken,
        send_rights: kore_abi::cap::Rights,
        recv_rights: kore_abi::cap::Rights,
    ) -> Result<ChannelPair, ChannelMintError> {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        if !valid_channel_endpoint_derivation_rights(send_rights)
            || !valid_channel_endpoint_derivation_rights(recv_rights)
        {
            return Err(ChannelMintError::InvalidRights);
        }

        let mut root = crate::cap::OwnedSlot::reserve().ok_or(ChannelMintError::NoSlots)?;
        let mut send = crate::cap::OwnedSlot::reserve().ok_or(ChannelMintError::NoSlots)?;
        let mut recv = crate::cap::OwnedSlot::reserve().ok_or(ChannelMintError::NoSlots)?;

        let mut b = SessionBuilder::new();
        let staged_root = b
            .derive_channel(space.slot(), crate::cap::CHANNEL_ROOT_RIGHTS, root.slot())
            .map_err(ChannelMintError::Build)?;
        b.derive_chan_send(staged_root, send_rights, send.slot())
            .map_err(ChannelMintError::Build)?;
        b.derive_chan_recv(staged_root, recv_rights, recv.slot())
            .map_err(ChannelMintError::Build)?;

        let committed = TrapTransport::new()
            .commit_now(&b)
            .map_err(|_| ChannelMintError::Transport)?;
        match classify_channel_mint(committed) {
            ChannelMintCompletion::Committed => Ok(ChannelPair::from_reserved_with_rights(
                root,
                send,
                send_rights,
                recv,
                recv_rights,
            )),
            ChannelMintCompletion::RolledBack(status) => Err(ChannelMintError::Refused(status)),
            ChannelMintCompletion::Indeterminate => {
                root.quarantine_unknown_publish();
                send.quarantine_unknown_publish();
                recv.quarantine_unknown_publish();
                Err(ChannelMintError::MalformedCompletion)
            }
        }
    }

    /// Probe whether a held receive endpoint's complete lineage is still live
    /// without revoking the endpoint itself.
    ///
    /// A successful same-kind derive proves the parent live; the temporary
    /// descendant is immediately retired before returning. A `Revoked`
    /// refusal on that derive proves an ancestor died. This is the cleanup
    /// witness used when a launcher must reconcile remote resources after a
    /// caller-owned `TaskToken` reaps a child, without retaining a second task
    /// handle or consuming a still-running child's canary.
    #[cfg(target_os = "none")]
    pub fn probe_chan_recv_lineage(parent: SlotRef) -> ChanRecvLineage {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let Some(mut child) = crate::cap::OwnedSlot::reserve() else {
            return ChanRecvLineage::Retryable;
        };
        let mut probe = SessionBuilder::new();
        let Ok(staged) =
            probe.derive_chan_recv(parent, crate::cap::CHANNEL_ENDPOINT_RIGHTS, child.slot())
        else {
            return ChanRecvLineage::Retryable;
        };
        // The probe result is a transactional witness only. Kill it in the
        // same session so a live-parent probe cannot accumulate descendants;
        // the local result name is then settled as AlreadyDead below.
        if probe.revoke(staged.into()).is_err() {
            return ChanRecvLineage::Retryable;
        }
        let Ok(committed) = TrapTransport::new().commit_now(&probe) else {
            child.quarantine_unknown_publish();
            return ChanRecvLineage::Retryable;
        };
        match classify_chan_recv_probe(committed) {
            ChanRecvLineage::Live => {
                let owner = ChanRecvToken::new(child);
                match owner.try_retire() {
                    Ok(_) => ChanRecvLineage::Live,
                    Err(_) => ChanRecvLineage::Retryable,
                }
            }
            ChanRecvLineage::Dead => ChanRecvLineage::Dead,
            ChanRecvLineage::Retryable => {
                if committed.failing_index() != 0
                    || committed.raw() as u16 > kore_abi::session::Status::AltDelivery as u16
                {
                    child.quarantine_unknown_publish();
                }
                ChanRecvLineage::Retryable
            }
        }
    }

    #[cfg(not(target_os = "none"))]
    pub fn probe_chan_recv_lineage(_parent: SlotRef) -> ChanRecvLineage {
        ChanRecvLineage::Retryable
    }

    /// Derive one locally retained endpoint below `source`.
    ///
    /// This is the revocation-pivot operation for delegated executable
    /// authority: the caller retains the returned affine child and transfers
    /// only descendants of it. Retiring this pivot therefore cuts the remote
    /// subtree by ordinary T6 lineage without revoking the caller's source.
    #[cfg(target_os = "none")]
    pub fn derive_owned_chan_send(source: SlotRef) -> Option<ChanSendToken> {
        derive_owned_chan_send_with_rights(source, crate::cap::CHANNEL_ENDPOINT_RIGHTS)
    }

    /// Derive one locally retained, explicitly attenuated send endpoint.
    ///
    /// A nonempty subset of the channel endpoint vocabulary is valid. In
    /// particular, a `DERIVE | REVOKE` endpoint is a delegation pivot or
    /// lineage selector but cannot itself rendezvous: the kernel independently
    /// requires `ACCESS` when a send endpoint is used by `Rendezvous`. The
    /// kernel also authenticates that the held source owns `DERIVE`; this
    /// local validation only keeps empty or foreign-rights shapes out of the
    /// session.
    #[cfg(target_os = "none")]
    pub fn derive_owned_chan_send_with_rights(
        source: SlotRef,
        rights: kore_abi::cap::Rights,
    ) -> Option<ChanSendToken> {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        if !valid_channel_endpoint_derivation_rights(rights) {
            return None;
        }

        let mut owned = crate::cap::OwnedSlot::reserve()?;
        let mut session = SessionBuilder::new();
        session
            .derive_chan_send(source, rights, owned.slot())
            .ok()?;
        let committed = TrapTransport::new().commit_now(&session).ok()?;
        if committed.status() == kore_abi::session::Status::Ok && committed.failing_index() == 0 {
            return Some(ChanSendToken::new_with_rights(owned, rights));
        }
        // A normal refusal rolled the only derive back, so the still-reserved
        // name safely reopens on drop. An unfamiliar completion cannot prove
        // rollback and must remain quarantined.
        if committed.failing_index() != 0
            || committed.raw() as u16 > kore_abi::session::Status::AltDelivery as u16
        {
            owned.quarantine_unknown_publish();
        }
        None
    }

    /// Receive-end counterpart of [`derive_owned_chan_send_with_rights`].
    /// The caller retains `source` as the revocation pivot and owns exactly
    /// one fresh attenuated descendant suitable for an exact rendezvous move.
    #[cfg(target_os = "none")]
    pub fn derive_owned_chan_recv_with_rights(
        source: SlotRef,
        rights: kore_abi::cap::Rights,
    ) -> Option<ChanRecvToken> {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        if !rights.contains(kore_abi::cap::Rights::ACCESS)
            || !crate::cap::CHANNEL_ENDPOINT_RIGHTS.contains(rights)
        {
            return None;
        }

        let mut owned = crate::cap::OwnedSlot::reserve()?;
        let mut session = SessionBuilder::new();
        session
            .derive_chan_recv(source, rights, owned.slot())
            .ok()?;
        let committed = TrapTransport::new().commit_now(&session).ok()?;
        if committed.status() == kore_abi::session::Status::Ok && committed.failing_index() == 0 {
            return Some(ChanRecvToken::new_with_rights(owned, rights));
        }
        if committed.failing_index() != 0
            || committed.raw() as u16 > kore_abi::session::Status::AltDelivery as u16
        {
            owned.quarantine_unknown_publish();
        }
        None
    }

    #[cfg(not(target_os = "none"))]
    pub fn derive_owned_chan_send(_source: SlotRef) -> Option<ChanSendToken> {
        None
    }

    #[cfg(not(target_os = "none"))]
    pub fn derive_owned_chan_send_with_rights(
        _source: SlotRef,
        _rights: kore_abi::cap::Rights,
    ) -> Option<ChanSendToken> {
        None
    }

    #[cfg(not(target_os = "none"))]
    pub fn derive_owned_chan_recv_with_rights(
        _source: SlotRef,
        _rights: kore_abi::cap::Rights,
    ) -> Option<ChanRecvToken> {
        None
    }

    /// Host and non-x86 userland builds never simulate capability minting.
    #[cfg(not(target_os = "none"))]
    pub fn mint_channel(_space: &ChannelSpaceToken) -> Result<ChannelPair, ChannelMintError> {
        Err(ChannelMintError::Unavailable)
    }

    #[cfg(not(target_os = "none"))]
    pub fn mint_channel_with_rights(
        _space: &ChannelSpaceToken,
        _send_rights: kore_abi::cap::Rights,
        _recv_rights: kore_abi::cap::Rights,
    ) -> Result<ChannelPair, ChannelMintError> {
        Err(ChannelMintError::Unavailable)
    }

    /// Claim the image store through the same authenticated held-lease exercise
    /// as a memory supply.  The role differentiates the service meaning; the
    /// kernel object is deliberately the common `MemLease` abstraction.
    #[cfg(target_os = "none")]
    pub fn claim_image_store(slot: SlotRef) -> Option<ImageStoreToken> {
        let (base, len, rights) = claim_lease_mapped(slot)?;
        Some(ImageStoreToken::possess(slot, base, len, rights))
    }

    /// Possess + map a **delivered whole-lease byte store** — the mechanism
    /// behind [`claim_image_store`]: one map exercise on the delivered
    /// `MemLease`, the installed `(base, len)` read back from the
    /// geometry report, the kind checked fail-closed.
    #[cfg(target_os = "none")]
    fn claim_lease_mapped(slot: SlotRef) -> Option<(u64, u64, kore_abi::cap::Rights)> {
        use kore_abi::cap::CapRecord;
        use kore_abi::session::GeomReport;

        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let report_slot = crate::entry::geometry::report_slot()?;
        let mut b = SessionBuilder::new();
        b.derive_raw(slot, Rights::empty(), 0, report_slot, None)
            .ok()?;
        let committed = TrapTransport::new().commit_now(&b).ok()?;
        if committed.status() != kore_abi::session::Status::Ok {
            return None;
        }
        let report = crate::entry::geometry::slot_record_va(report_slot)? as *const CapRecord;
        let geom = GeomReport::decode(&unsafe { core::ptr::read_volatile(report) })?;
        let source = crate::entry::geometry::slot_record_va(slot)? as *const CapRecord;
        let rec = unsafe { core::ptr::read_volatile(source) };
        if rec.kind != kore_abi::cap::ResourceKind::MemLease as u16 {
            return None;
        }
        Some((geom.base, geom.len, rec.rights))
    }

    #[cfg(not(target_os = "none"))]
    pub fn claim_image_store(_slot: SlotRef) -> Option<ImageStoreToken> {
        None
    }

    #[cfg(target_os = "none")]
    pub fn claim_xmint(slot: SlotRef) -> Option<XMintToken> {
        claim_kind::<crate::cap::XMintKind>(slot)
    }

    #[cfg(not(target_os = "none"))]
    pub fn claim_xmint(_slot: SlotRef) -> Option<XMintToken> {
        None
    }

    /// Claim one root complex (`res/DEVICE_SERVICE.md` §10) — a broker
    /// row's `role::ROOT_COMPLEX` endowment, through the same
    /// authenticated-record exercise as the X-mint and the exec authority.
    ///
    /// The kind gate is the whole of the check that matters: a task handed
    /// a *device* at this slot claims nothing, so a driver's own endowment
    /// can never bind as a root's inventory.
    #[cfg(target_os = "none")]
    pub fn claim_root_complex(slot: SlotRef) -> Option<crate::cap::RootComplexToken> {
        claim_kind::<crate::cap::RootComplexKind>(slot)
    }

    #[cfg(not(target_os = "none"))]
    pub fn claim_root_complex(_slot: SlotRef) -> Option<crate::cap::RootComplexToken> {
        None
    }

    /// Claim the platform-device inventory — broker row 0's
    /// `role::PLATFORM_BUS` endowment, the [`claim_root_complex`] sibling
    /// for firmware-described (non-PCI) devices (#572).
    #[cfg(target_os = "none")]
    pub fn claim_platform_bus(slot: SlotRef) -> Option<crate::cap::PlatformBusToken> {
        claim_kind::<crate::cap::PlatformBusKind>(slot)
    }

    #[cfg(not(target_os = "none"))]
    pub fn claim_platform_bus(_slot: SlotRef) -> Option<crate::cap::PlatformBusToken> {
        None
    }

    /// Read the subject's inventory entry `index` — what firmware described
    /// and what enumeration proved, as data.  The subject is a
    /// [`RootComplexToken`](crate::cap::RootComplexToken) (root-local PCI
    /// index) or the [`PlatformBusToken`](crate::cap::PlatformBusToken)
    /// (platform index) — one verb, two authorities, disjoint namespaces.
    ///
    /// Mints nothing: the result lands in the caller's report slot as a
    /// bind-inert record, so a broker matching a selector across every
    /// entry causes no capability churn at all.
    ///
    /// The caller first obtains [`inventory_len`], so every error in
    /// `0..len` remains an error. A refusal is never interpreted as EOF.
    #[cfg(target_os = "none")]
    pub fn device_record<K: crate::cap::InventorySubject>(
        store: &crate::cap::Token<K>,
        index: u32,
    ) -> Result<kore_abi::device::DeviceRecord, InventoryReadError> {
        use kore_abi::cap::CapRecord;
        use kore_abi::device::{DeviceRecord, DeviceSpecies};
        use kore_abi::session::DeviceReport;

        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let report_slot =
            crate::entry::geometry::report_slot().ok_or(InventoryReadError::Unavailable)?;
        let mut b = SessionBuilder::new();
        b.read_device_record(store.slot(), index, report_slot)
            .map_err(|_| InventoryReadError::Build)?;
        let committed = TrapTransport::new()
            .commit_now(&b)
            .map_err(|_| InventoryReadError::Transport)?;
        device_read_committed(committed)?;
        let ptr = crate::entry::geometry::slot_record_va(report_slot)
            .ok_or(InventoryReadError::Unavailable)? as *const CapRecord;
        let report = DeviceReport::decode(&unsafe { core::ptr::read_volatile(ptr) })
            .ok_or(InventoryReadError::MalformedReport)?;
        Ok(DeviceRecord {
            // An unknown species is a refusal, not a default: the two
            // grant disjoint authorities, so guessing here would hand the
            // broker a record it would claim the wrong child kind from.
            species: DeviceSpecies::from_bits(report.species)
                .ok_or(InventoryReadError::MalformedReport)?,
            // The caller named the index, so it is the caller that fills
            // it back in — it is deliberately not on the wire.
            index,
            vendor: report.vendor,
            device: report.device,
            class: report.class,
            subclass: report.subclass,
            progif: report.progif,
            locator: report.locator,
            segment: report.segment,
            msix_table_size: report.msix_table_size,
            diag: report.diag,
            revision: report.revision,
            flags: report.flags,
        })
    }

    #[cfg(not(target_os = "none"))]
    pub fn device_record<K: crate::cap::InventorySubject>(
        _store: &crate::cap::Token<K>,
        _index: u32,
    ) -> Result<kore_abi::device::DeviceRecord, InventoryReadError> {
        Err(InventoryReadError::Unavailable)
    }

    /// Read the exact committed claim for this root-local PCI index. A zero
    /// pair is a successful observation without a committed identity. Reports
    /// cannot be bound as capabilities or used as permission to release.
    #[cfg(target_os = "none")]
    pub fn device_claim(
        root: &crate::cap::RootComplexToken,
        index: u32,
    ) -> Result<kore_abi::session::DeviceClaimReport, InventoryReadError> {
        use crate::transport::{Transport, TrapTransport};
        let slot = crate::entry::geometry::report_slot().ok_or(InventoryReadError::Unavailable)?;
        let mut session = crate::session::SessionBuilder::new();
        session
            .read_device_claim(root.slot(), index, slot)
            .map_err(|_| InventoryReadError::Build)?;
        let result = TrapTransport::new()
            .commit_now(&session)
            .map_err(|_| InventoryReadError::Transport)?;
        device_read_committed(result)?;
        let ptr = crate::entry::geometry::slot_record_va(slot)
            .ok_or(InventoryReadError::Unavailable)?
            as *const kore_abi::cap::CapRecord;
        kore_abi::session::DeviceClaimReport::decode(&unsafe { core::ptr::read_volatile(ptr) })
            .ok_or(InventoryReadError::MalformedReport)
    }

    #[cfg(not(target_os = "none"))]
    pub fn device_claim(
        _root: &crate::cap::RootComplexToken,
        _index: u32,
    ) -> Result<kore_abi::session::DeviceClaimReport, InventoryReadError> {
        Err(InventoryReadError::Unavailable)
    }

    /// Idempotently close the expected PCI grant through the selected root's
    /// REVOKE authority. Retry the same identity on uncertainty/Busy/Pending.
    /// AdmissionDrained proves only tracked readers left; no hardware is freed.
    #[cfg(target_os = "none")]
    pub fn close_device_claim(
        root: &crate::cap::RootComplexToken,
        index: u32,
        identity: kore_abi::session::DeviceClaimReport,
    ) -> Result<kore_abi::session::DeviceClaimCloseReport, InventoryReadError> {
        use crate::transport::{Transport, TrapTransport};
        let slot = crate::entry::geometry::report_slot().ok_or(InventoryReadError::Unavailable)?;
        let mut session = crate::session::SessionBuilder::new();
        session
            .close_device_claim(root.slot(), index, identity, slot)
            .map_err(|_| InventoryReadError::Build)?;
        let result = TrapTransport::new()
            .commit_now(&session)
            .map_err(|_| InventoryReadError::Transport)?;
        device_read_committed(result)?;
        let ptr = crate::entry::geometry::slot_record_va(slot)
            .ok_or(InventoryReadError::Unavailable)?
            as *const kore_abi::cap::CapRecord;
        kore_abi::session::DeviceClaimCloseReport::decode(&unsafe { core::ptr::read_volatile(ptr) })
            .ok_or(InventoryReadError::MalformedReport)
    }

    #[cfg(not(target_os = "none"))]
    pub fn close_device_claim(
        _root: &crate::cap::RootComplexToken,
        _index: u32,
        _identity: kore_abi::session::DeviceClaimReport,
    ) -> Result<kore_abi::session::DeviceClaimCloseReport, InventoryReadError> {
        Err(InventoryReadError::Unavailable)
    }

    /// Contain an exact already-closed, drained PCI grant through its issuer.
    /// This does not unmap backing or release the physical claim. WouldBlock
    /// remains a typed refusal; callers retain custody and retry this identity.
    #[cfg(target_os = "none")]
    pub fn quiesce_device_claim(
        root: &crate::cap::RootComplexToken,
        index: u32,
        identity: kore_abi::session::DeviceClaimReport,
    ) -> Result<kore_abi::session::QuiesceReport, InventoryReadError> {
        use crate::transport::{Transport, TrapTransport};
        let slot = crate::entry::geometry::report_slot().ok_or(InventoryReadError::Unavailable)?;
        let mut session = crate::session::SessionBuilder::new();
        session
            .quiesce_device_claim(root.slot(), index, identity, slot)
            .map_err(|_| InventoryReadError::Build)?;
        let result = TrapTransport::new()
            .commit_now(&session)
            .map_err(|_| InventoryReadError::Transport)?;
        device_read_committed(result)?;
        let ptr = crate::entry::geometry::slot_record_va(slot)
            .ok_or(InventoryReadError::Unavailable)?
            as *const kore_abi::cap::CapRecord;
        kore_abi::session::QuiesceReport::decode(&unsafe { core::ptr::read_volatile(ptr) })
            .ok_or(InventoryReadError::MalformedReport)
    }

    #[cfg(not(target_os = "none"))]
    pub fn quiesce_device_claim(
        _root: &crate::cap::RootComplexToken,
        _index: u32,
        _identity: kore_abi::session::DeviceClaimReport,
    ) -> Result<kore_abi::session::QuiesceReport, InventoryReadError> {
        Err(InventoryReadError::Unavailable)
    }

    /// Retire registered DMA windows; all reports retain the physical claim.
    #[cfg(target_os = "none")]
    pub fn retire_device_claim_dma(
        root: &crate::cap::RootComplexToken,
        index: u32,
        identity: kore_abi::session::DeviceClaimReport,
    ) -> Result<kore_abi::session::DeviceClaimDmaReport, InventoryReadError> {
        use crate::transport::{Transport, TrapTransport};
        let slot = crate::entry::geometry::report_slot().ok_or(InventoryReadError::Unavailable)?;
        let mut session = crate::session::SessionBuilder::new();
        session
            .retire_device_claim_dma(root.slot(), index, identity, slot)
            .map_err(|_| InventoryReadError::Build)?;
        let result = TrapTransport::new()
            .commit_now(&session)
            .map_err(|_| InventoryReadError::Transport)?;
        device_read_committed(result)?;
        let ptr = crate::entry::geometry::slot_record_va(slot)
            .ok_or(InventoryReadError::Unavailable)?
            as *const kore_abi::cap::CapRecord;
        kore_abi::session::DeviceClaimDmaReport::decode(&unsafe { core::ptr::read_volatile(ptr) })
            .ok_or(InventoryReadError::MalformedReport)
    }

    #[cfg(not(target_os = "none"))]
    pub fn retire_device_claim_dma(
        _root: &crate::cap::RootComplexToken,
        _index: u32,
        _identity: kore_abi::session::DeviceClaimReport,
    ) -> Result<kore_abi::session::DeviceClaimDmaReport, InventoryReadError> {
        Err(InventoryReadError::Unavailable)
    }

    /// Observe a bounded I2C admission page from this exact held root.
    /// Failed firmware publication is encoded data, not a transport refusal.
    #[cfg(target_os = "none")]
    pub fn i2c_admission_page(
        root: &crate::cap::RootComplexToken,
        row: u8,
    ) -> Result<kore_abi::device::I2cAdmissionPage, InventoryReadError> {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};
        use kore_abi::cap::CapRecord;
        use kore_abi::session::I2cAdmissionReport;

        let report_slot =
            crate::entry::geometry::report_slot().ok_or(InventoryReadError::Unavailable)?;
        let mut builder = SessionBuilder::new();
        builder
            .read_i2c_admission_page(root.slot(), row, report_slot)
            .map_err(|_| InventoryReadError::Build)?;
        let committed = TrapTransport::new()
            .commit_now(&builder)
            .map_err(|_| InventoryReadError::Transport)?;
        device_read_committed(committed)?;
        let pointer = crate::entry::geometry::slot_record_va(report_slot)
            .ok_or(InventoryReadError::Unavailable)? as *const CapRecord;
        // SAFETY: the just-committed observation wrote this bounded report slot.
        let report = I2cAdmissionReport::decode(&unsafe { core::ptr::read_volatile(pointer) })
            .ok_or(InventoryReadError::MalformedReport)?;
        if report.page.row != row {
            return Err(InventoryReadError::MalformedReport);
        }
        Ok(report.page)
    }

    #[cfg(not(target_os = "none"))]
    pub fn i2c_admission_page(
        _root: &crate::cap::RootComplexToken,
        _row: u8,
    ) -> Result<kore_abi::device::I2cAdmissionPage, InventoryReadError> {
        Err(InventoryReadError::Unavailable)
    }

    /// How many entries the subject's inventory holds — the bound on a
    /// broker's walk of that root (or of the platform bus).
    ///
    /// Answered by an explicit, successful count report rather than probing
    /// until a refusal. Entries belong to the boot-frozen catalog; a refused
    /// query is an error, never an empty or shorter inventory.
    #[cfg(target_os = "none")]
    pub fn inventory_len<K: crate::cap::InventorySubject>(
        store: &crate::cap::Token<K>,
    ) -> Result<u32, InventoryReadError> {
        use kore_abi::cap::CapRecord;
        use kore_abi::session::DeviceCountReport;

        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let report_slot =
            crate::entry::geometry::report_slot().ok_or(InventoryReadError::Unavailable)?;
        let mut b = SessionBuilder::new();
        b.read_inventory_len(store.slot(), report_slot)
            .map_err(|_| InventoryReadError::Build)?;
        let committed = TrapTransport::new()
            .commit_now(&b)
            .map_err(|_| InventoryReadError::Transport)?;
        device_read_committed(committed)?;
        let ptr = crate::entry::geometry::slot_record_va(report_slot)
            .ok_or(InventoryReadError::Unavailable)? as *const CapRecord;
        DeviceCountReport::decode(&unsafe { core::ptr::read_volatile(ptr) })
            .map(|report| report.len)
            .ok_or(InventoryReadError::MalformedReport)
    }

    #[cfg(not(target_os = "none"))]
    pub fn inventory_len<K: crate::cap::InventorySubject>(
        _store: &crate::cap::Token<K>,
    ) -> Result<u32, InventoryReadError> {
        Err(InventoryReadError::Unavailable)
    }

    /// Read one HID child correlated with an already-claimed PCI or platform I2C
    /// controller.  The token is the authority and correlation proof; the
    /// caller supplies no AML path or register address.
    #[cfg(target_os = "none")]
    pub fn i2c_hid_record<C: crate::cap::I2cHidSubject>(
        device: &C,
        index: u32,
    ) -> Result<kore_abi::session::I2cHidReport, InventoryReadError> {
        use kore_abi::cap::CapRecord;
        use kore_abi::session::I2cHidReport;

        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let report_slot =
            crate::entry::geometry::report_slot().ok_or(InventoryReadError::Unavailable)?;
        let mut b = SessionBuilder::new();
        b.read_i2c_hid_record(device.i2c_subject_slot(), index, report_slot)
            .map_err(|_| InventoryReadError::Build)?;
        let committed = TrapTransport::new()
            .commit_now(&b)
            .map_err(|_| InventoryReadError::Transport)?;
        device_read_committed(committed)?;
        let ptr = crate::entry::geometry::slot_record_va(report_slot)
            .ok_or(InventoryReadError::Unavailable)? as *const CapRecord;
        I2cHidReport::decode(&unsafe { core::ptr::read_volatile(ptr) })
            .ok_or(InventoryReadError::MalformedReport)
    }

    #[cfg(not(target_os = "none"))]
    pub fn i2c_hid_record<C: crate::cap::I2cHidSubject>(
        _device: &C,
        _index: u32,
    ) -> Result<kore_abi::session::I2cHidReport, InventoryReadError> {
        Err(InventoryReadError::Unavailable)
    }

    /// Authoritative HID-child count behind one claimed PCI or platform controller.
    #[cfg(target_os = "none")]
    pub fn i2c_hid_len<C: crate::cap::I2cHidSubject>(
        device: &C,
    ) -> Result<u32, InventoryReadError> {
        use kore_abi::cap::CapRecord;
        use kore_abi::session::I2cHidCountReport;

        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let report_slot =
            crate::entry::geometry::report_slot().ok_or(InventoryReadError::Unavailable)?;
        let mut b = SessionBuilder::new();
        b.read_i2c_hid_len(device.i2c_subject_slot(), report_slot)
            .map_err(|_| InventoryReadError::Build)?;
        let committed = TrapTransport::new()
            .commit_now(&b)
            .map_err(|_| InventoryReadError::Transport)?;
        device_read_committed(committed)?;
        let ptr = crate::entry::geometry::slot_record_va(report_slot)
            .ok_or(InventoryReadError::Unavailable)? as *const CapRecord;
        I2cHidCountReport::decode(&unsafe { core::ptr::read_volatile(ptr) })
            .map(|report| report.len)
            .ok_or(InventoryReadError::MalformedReport)
    }

    #[cfg(not(target_os = "none"))]
    pub fn i2c_hid_len<C: crate::cap::I2cHidSubject>(
        _device: &C,
    ) -> Result<u32, InventoryReadError> {
        Err(InventoryReadError::Unavailable)
    }

    /// Compatibility spelling for platform-only callers.
    pub fn platform_i2c_hid_record(
        device: &PlatformToken,
        index: u32,
    ) -> Result<kore_abi::session::I2cHidReport, InventoryReadError> {
        i2c_hid_record(device, index)
    }

    /// Compatibility spelling for platform-only callers.
    pub fn platform_i2c_hid_len(device: &PlatformToken) -> Result<u32, InventoryReadError> {
        i2c_hid_len(device)
    }

    /// Hand out the subject's inventory entry `index` as a device
    /// capability — the derive a broker runs to give a driver its device.
    ///
    /// The result is an ordinary device record in `into`, indistinguishable
    /// downstream from a founder-minted one; only its parent differs, so
    /// revoking the subject's grant reclaims every device it ever handed
    /// out — and only its own: a sibling root's grants ride a different
    /// tag lineage.  `false` on any refusal, and a refusal grants nothing
    /// (fail-closed: no authority leaves on an error path).
    #[cfg(target_os = "none")]
    pub fn claim_stored_device<K: crate::cap::InventorySubject>(
        store: &crate::cap::Token<K>,
        index: u32,
        species: kore_abi::device::DeviceSpecies,
        into: SlotRef,
    ) -> bool {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let mut b = SessionBuilder::new();
        if b.claim_stored_device(store.slot(), DEVICE_OWNER_RIGHTS, index, species, into)
            .is_err()
        {
            return false;
        }
        match TrapTransport::new().commit_now(&b) {
            Ok(c) => c.status() == kore_abi::session::Status::Ok,
            Err(_) => false,
        }
    }

    #[cfg(not(target_os = "none"))]
    pub fn claim_stored_device<K: crate::cap::InventorySubject>(
        _store: &crate::cap::Token<K>,
        _index: u32,
        _species: kore_abi::device::DeviceSpecies,
        _into: SlotRef,
    ) -> bool {
        false
    }
    /// Which stage of [`map_bar`] refused — the five metal rounds this
    /// vocabulary was bought with all read "BAR unavailable" while the
    /// true stage ranged over slot exhaustion, the claim derive, and the
    /// map install.  `ClaimCommit`/`MapCommit` carry the kernel's
    /// [`Status`](kore_abi::session::Status) discriminant.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum BarMapFault {
        /// No report slot, or the task's cap arena had no free slot.
        Slot,
        /// The claim session refused to build or transport.
        ClaimBuild,
        /// The claim session committed with a non-Ok status.
        ClaimCommit(u8),
        /// The map session refused to build or transport.
        MapBuild,
        /// The map session committed with a non-Ok status.
        MapCommit(u8),
        /// The geometry report was absent, malformed, or empty.
        Geometry,
    }

    impl BarMapFault {
        /// Pack as `stage[2:0] | status << 3` — the note-detail byte.
        pub fn code(self) -> u8 {
            match self {
                Self::Slot => 1,
                Self::ClaimBuild => 2,
                Self::ClaimCommit(s) => 3 | (s << 3),
                Self::MapBuild => 4,
                Self::MapCommit(s) => 5 | (s << 3),
                Self::Geometry => 6,
            }
        }
    }

    #[cfg(target_os = "none")]
    pub fn map_bar(dev: &DeviceToken, bar: u8) -> Result<BarWindow, BarMapFault> {
        use kore_abi::session::GeomReport;

        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let report = crate::entry::geometry::report_slot().ok_or(BarMapFault::Slot)?;
        // Arena exhaustion is a REAL refusal shape here, not plumbing: a
        // row whose founding spent its slot band answers `None` from this
        // reserve on every retry, identically, no matter what the kernel
        // fixes — the failure mode a stage-less `Option` spent five metal
        // rounds hiding inside "BAR unavailable".
        let mut owned = crate::cap::OwnedSlot::reserve().ok_or(BarMapFault::Slot)?;
        let full = Rights::ACCESS | Rights::MODIFY | Rights::DERIVE | Rights::REVOKE;

        let mut claim = SessionBuilder::new();
        claim
            .claim_bar(dev.slot(), full, bar, owned.slot())
            .map_err(|_| BarMapFault::ClaimBuild)?;
        let committed = TrapTransport::new()
            .commit_now(&claim)
            .map_err(|_| BarMapFault::ClaimBuild)?;
        if committed.status() != kore_abi::session::Status::Ok {
            return Err(BarMapFault::ClaimCommit(committed.status() as u8));
        }
        owned.activate();

        let mut map = SessionBuilder::new();
        map.map_owned(owned.slot(), Rights::ACCESS | Rights::MODIFY, report)
            .map_err(|_| BarMapFault::MapBuild)?;
        let committed = TrapTransport::new()
            .commit_now(&map)
            .map_err(|_| BarMapFault::MapBuild)?;
        if committed.status() != kore_abi::session::Status::Ok {
            return Err(BarMapFault::MapCommit(committed.status() as u8));
        }
        let ptr = crate::entry::geometry::slot_record_va(report).ok_or(BarMapFault::Geometry)?
            as *const kore_abi::cap::CapRecord;
        let geom = GeomReport::decode(&unsafe { core::ptr::read_volatile(ptr) })
            .ok_or(BarMapFault::Geometry)?;
        if geom.len == 0 {
            return Err(BarMapFault::Geometry);
        }
        Ok(BarWindow::new(geom.base as *mut u8, geom.len, owned))
    }

    #[cfg(not(target_os = "none"))]
    pub fn map_bar(_dev: &DeviceToken, _bar: u8) -> Result<BarWindow, BarMapFault> {
        Err(BarMapFault::Slot)
    }

    /// Map a platform device's sole firmware-declared MMIO window.
    #[cfg(target_os = "none")]
    pub fn map_platform_mmio(dev: &crate::cap::PlatformToken) -> Result<BarWindow, BarMapFault> {
        use kore_abi::session::GeomReport;

        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let report = crate::entry::geometry::report_slot().ok_or(BarMapFault::Slot)?;
        let mut owned = crate::cap::OwnedSlot::reserve().ok_or(BarMapFault::Slot)?;
        let full = Rights::ACCESS | Rights::MODIFY | Rights::DERIVE | Rights::REVOKE;

        let mut claim = SessionBuilder::new();
        claim
            .claim_platform_mmio(dev.slot(), full, owned.slot())
            .map_err(|_| BarMapFault::ClaimBuild)?;
        let committed = TrapTransport::new()
            .commit_now(&claim)
            .map_err(|_| BarMapFault::ClaimBuild)?;
        if committed.status() != kore_abi::session::Status::Ok {
            return Err(BarMapFault::ClaimCommit(committed.status() as u8));
        }
        owned.activate();

        let mut map = SessionBuilder::new();
        map.map_owned(owned.slot(), Rights::ACCESS | Rights::MODIFY, report)
            .map_err(|_| BarMapFault::MapBuild)?;
        let committed = TrapTransport::new()
            .commit_now(&map)
            .map_err(|_| BarMapFault::MapBuild)?;
        if committed.status() != kore_abi::session::Status::Ok {
            return Err(BarMapFault::MapCommit(committed.status() as u8));
        }
        let ptr = crate::entry::geometry::slot_record_va(report).ok_or(BarMapFault::Geometry)?
            as *const kore_abi::cap::CapRecord;
        let geom = GeomReport::decode(&unsafe { core::ptr::read_volatile(ptr) })
            .ok_or(BarMapFault::Geometry)?;
        if geom.len == 0 {
            return Err(BarMapFault::Geometry);
        }
        Ok(BarWindow::new(geom.base as *mut u8, geom.len, owned))
    }

    #[cfg(not(target_os = "none"))]
    pub fn map_platform_mmio(_dev: &crate::cap::PlatformToken) -> Result<BarWindow, BarMapFault> {
        Err(BarMapFault::Slot)
    }

    /// Which stage of device stage-1 ownership attestation refused.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum DeviceMmuFault {
        Slot,
        Build,
        Commit(u8),
    }

    /// Convert a live, DMA-active device into the affine stage-1 MMU owner.
    /// Drivers call this only after their class-specific translation and job
    /// path proof succeeds; the kernel additionally revalidates the outer DMA
    /// epoch before minting the facet.
    #[cfg(target_os = "none")]
    pub fn attest_device_mmu(dev: &DeviceToken) -> Result<DeviceMmuToken, DeviceMmuFault> {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let owned = crate::cap::OwnedSlot::reserve().ok_or(DeviceMmuFault::Slot)?;
        let mut b = SessionBuilder::new();
        b.attest_device_mmu(dev.slot(), owned.slot())
            .map_err(|_| DeviceMmuFault::Build)?;
        let committed = TrapTransport::new()
            .commit_now(&b)
            .map_err(|_| DeviceMmuFault::Build)?;
        if committed.status() != kore_abi::session::Status::Ok {
            return Err(DeviceMmuFault::Commit(committed.status() as u8));
        }
        Ok(DeviceMmuToken::new(owned))
    }

    #[cfg(not(target_os = "none"))]
    pub fn attest_device_mmu(_dev: &DeviceToken) -> Result<DeviceMmuToken, DeviceMmuFault> {
        Err(DeviceMmuFault::Slot)
    }

    /// Which stage of registering an owned DMA lease refused.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum FabricRegistrationFault {
        Slot,
        Build,
        Commit(u8),
        MalformedReport,
    }

    #[cfg(target_os = "none")]
    pub(crate) fn describe_fabric(
        slot: kore_abi::session::SlotRef,
    ) -> Result<kore_abi::session::FabricDomainReport, FabricRegistrationFault> {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};
        let report = crate::entry::geometry::report_slot().ok_or(FabricRegistrationFault::Slot)?;
        let mut builder = SessionBuilder::new();
        builder
            .describe_fabric(slot, report)
            .map_err(|_| FabricRegistrationFault::Build)?;
        let committed = TrapTransport::new()
            .commit_now(&builder)
            .map_err(|_| FabricRegistrationFault::Build)?;
        if committed.status() != kore_abi::session::Status::Ok {
            return Err(FabricRegistrationFault::Commit(committed.status() as u8));
        }
        let ptr = crate::entry::geometry::slot_record_va(report)
            .ok_or(FabricRegistrationFault::Slot)?
            as *const kore_abi::cap::CapRecord;
        kore_abi::session::FabricDomainReport::decode(&unsafe { core::ptr::read_volatile(ptr) })
            .ok_or(FabricRegistrationFault::MalformedReport)
    }

    #[cfg(not(target_os = "none"))]
    pub(crate) fn describe_fabric(
        _slot: kore_abi::session::SlotRef,
    ) -> Result<kore_abi::session::FabricDomainReport, FabricRegistrationFault> {
        Err(FabricRegistrationFault::Slot)
    }

    #[cfg(target_os = "none")]
    pub(crate) fn describe_fabric_route(
        slot: kore_abi::session::SlotRef,
    ) -> Result<kore_abi::session::FabricRouteInfo, FabricRegistrationFault> {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};
        let report = crate::entry::geometry::report_slot().ok_or(FabricRegistrationFault::Slot)?;
        let mut builder = SessionBuilder::new();
        builder
            .derive_raw(
                slot,
                Rights::empty(),
                kore_abi::session::DERIVE_KIND_FABRIC_ROUTE_OBSERVE,
                report,
                None,
            )
            .map_err(|_| FabricRegistrationFault::Build)?;
        let committed = TrapTransport::new()
            .commit_now(&builder)
            .map_err(|_| FabricRegistrationFault::Build)?;
        if committed.status() != kore_abi::session::Status::Ok {
            return Err(FabricRegistrationFault::Commit(committed.status() as u8));
        }
        let ptr = crate::entry::geometry::slot_record_va(report)
            .ok_or(FabricRegistrationFault::Slot)?
            as *const kore_abi::cap::CapRecord;
        kore_abi::session::FabricRouteInfo::decode(&unsafe { core::ptr::read_volatile(ptr) })
            .ok_or(FabricRegistrationFault::MalformedReport)
    }

    #[cfg(not(target_os = "none"))]
    pub(crate) fn describe_fabric_route(
        _slot: kore_abi::session::SlotRef,
    ) -> Result<kore_abi::session::FabricRouteInfo, FabricRegistrationFault> {
        Err(FabricRegistrationFault::Slot)
    }

    /// Failure to open or close a possessed mapping route.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum FabricRouteFault {
        Slot,
        Bounds,
        Build,
        Commit(u8),
        Indeterminate,
        MalformedReport,
        Busy,
        Quarantined,
    }

    impl FabricRouteFault {
        /// Stable diagnostic detail; commit status remains distinguishable
        /// from local admission, report, and uncertain publication failures.
        pub const fn code(self) -> u64 {
            match self {
                Self::Slot => 1,
                Self::Bounds => 2,
                Self::Build => 3,
                Self::Commit(status) => 0x100 | status as u64,
                Self::Indeterminate => 4,
                Self::MalformedReport => 5,
                Self::Busy => 6,
                Self::Quarantined => 7,
            }
        }
    }

    #[cfg(target_os = "none")]
    pub(crate) fn open_fabric_route(
        registration: SlotRef,
        request: &kore_abi::session::FabricRouteReq,
    ) -> Result<crate::cap::FabricRouteToken, FabricRouteFault> {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};
        if !request.valid() {
            return Err(FabricRouteFault::Bounds);
        }
        let mut owned = crate::cap::OwnedSlot::reserve().ok_or(FabricRouteFault::Slot)?;
        let mut builder = SessionBuilder::new();
        builder
            .open_fabric_route(
                registration,
                request,
                Rights::ACCESS | Rights::MODIFY | Rights::DERIVE | Rights::REVOKE,
                owned.slot(),
            )
            .map_err(|_| FabricRouteFault::Build)?;
        let completion = TrapTransport::new()
            .commit_now(&builder)
            .map_err(|_| FabricRouteFault::Build)?;
        match (completion.raw() as u16, completion.failing_index()) {
            (0, 0) => Ok(crate::cap::FabricRouteToken::new(owned)),
            (status @ 1..=6, 0) => Err(FabricRouteFault::Commit(status as u8)),
            _ => {
                owned.quarantine_unknown_publish();
                Err(FabricRouteFault::Indeterminate)
            }
        }
    }

    #[cfg(not(target_os = "none"))]
    pub(crate) fn open_fabric_route(
        _registration: SlotRef,
        _request: &kore_abi::session::FabricRouteReq,
    ) -> Result<crate::cap::FabricRouteToken, FabricRouteFault> {
        Err(FabricRouteFault::Slot)
    }

    #[cfg(target_os = "none")]
    pub(crate) fn scope_fabric_route(
        route: SlotRef,
        request: &kore_abi::session::FabricRouteScopeReq,
    ) -> Result<crate::cap::FabricRouteToken, FabricRouteFault> {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};
        if !request.valid() {
            return Err(FabricRouteFault::Bounds);
        }
        let mut owned = crate::cap::OwnedSlot::reserve().ok_or(FabricRouteFault::Slot)?;
        let mut builder = SessionBuilder::new();
        builder
            .scope_fabric_route(
                route,
                request,
                Rights::ACCESS | Rights::MODIFY | Rights::DERIVE | Rights::REVOKE,
                owned.slot(),
            )
            .map_err(|_| FabricRouteFault::Build)?;
        let completion = TrapTransport::new()
            .commit_now(&builder)
            .map_err(|_| FabricRouteFault::Build)?;
        match (completion.raw() as u16, completion.failing_index()) {
            (0, 0) => Ok(crate::cap::FabricRouteToken::new(owned)),
            (status @ 1..=6, 0) => Err(FabricRouteFault::Commit(status as u8)),
            _ => {
                owned.quarantine_unknown_publish();
                Err(FabricRouteFault::Indeterminate)
            }
        }
    }

    #[cfg(not(target_os = "none"))]
    pub(crate) fn scope_fabric_route(
        _route: SlotRef,
        _request: &kore_abi::session::FabricRouteScopeReq,
    ) -> Result<crate::cap::FabricRouteToken, FabricRouteFault> {
        Err(FabricRouteFault::Slot)
    }

    #[cfg(target_os = "none")]
    pub(crate) fn close_fabric_route(route: SlotRef) -> Result<(), FabricRouteFault> {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};
        let route_ptr = crate::entry::geometry::slot_record_va(route)
            .ok_or(FabricRouteFault::Slot)?
            as *const kore_abi::cap::CapRecord;
        let expected = unsafe { core::ptr::read_volatile(route_ptr) }.cid.0;
        let report = crate::entry::geometry::report_slot().ok_or(FabricRouteFault::Slot)?;
        let report_ptr = crate::entry::geometry::slot_record_va(report)
            .ok_or(FabricRouteFault::Slot)?
            as *const kore_abi::cap::CapRecord;
        let mut builder = SessionBuilder::new();
        builder
            .close_fabric_route(route, report)
            .map_err(|_| FabricRouteFault::Build)?;
        let completion = TrapTransport::new()
            .commit_now(&builder)
            .map_err(|_| FabricRouteFault::Build)?;
        finish_fabric_close(
            completion,
            &unsafe { core::ptr::read_volatile(report_ptr) },
            expected,
        )
    }

    #[cfg(any(test, target_os = "none"))]
    fn finish_fabric_close(
        completion: kore_abi::session::Committed,
        record: &kore_abi::cap::CapRecord,
        expected: u64,
    ) -> Result<(), FabricRouteFault> {
        // The high half is a clock stamp, not part of the status/index pair.
        // Unknown outcomes cannot authorize retirement even with a stale report.
        match (completion.raw() as u16, completion.failing_index()) {
            (0, 0) => {}
            (status @ 1..=6, 0) => return Err(FabricRouteFault::Commit(status as u8)),
            _ => return Err(FabricRouteFault::Indeterminate),
        }
        let report = kore_abi::session::FabricRouteCloseReport::decode(record)
            .filter(|report| report.route == expected)
            .ok_or(FabricRouteFault::MalformedReport)?;
        match report.state {
            0 => Ok(()),
            1 => Err(FabricRouteFault::Busy),
            _ => Err(FabricRouteFault::Quarantined),
        }
    }

    #[cfg(not(target_os = "none"))]
    pub(crate) fn close_fabric_route(_route: SlotRef) -> Result<(), FabricRouteFault> {
        Err(FabricRouteFault::Slot)
    }

    #[cfg(test)]
    #[test]
    fn fabric_close_accepts_clock_stamps_but_requires_exact_outcome_and_route() {
        use kore_abi::session::{Committed, FabricRouteCloseReport, Status, stamp_clock};
        for clock in [0, 1, u32::MAX] {
            let completion = Committed::from_raw(stamp_clock(Status::Ok.encode(0), clock));
            for (state, expected) in [
                (0, Ok(())),
                (1, Err(FabricRouteFault::Busy)),
                (2, Err(FabricRouteFault::Quarantined)),
            ] {
                let report = FabricRouteCloseReport {
                    route: 0x100000007,
                    state,
                }
                .encode();
                assert_eq!(
                    finish_fabric_close(completion, &report, 0x100000007),
                    expected
                );
                assert_eq!(
                    finish_fabric_close(completion, &report, 0x100000008),
                    Err(FabricRouteFault::MalformedReport)
                );
            }
            let closed = FabricRouteCloseReport { route: 7, state: 0 }.encode();
            for raw in [Status::Ok.encode(1), Status::Denied.encode(1), 0xffff] {
                assert_eq!(
                    finish_fabric_close(Committed::from_raw(stamp_clock(raw, clock)), &closed, 7),
                    Err(FabricRouteFault::Indeterminate)
                );
            }
            assert_eq!(
                finish_fabric_close(
                    Committed::from_raw(stamp_clock(Status::Denied.encode(0), clock)),
                    &closed,
                    7
                ),
                Err(FabricRouteFault::Commit(Status::Denied as u8))
            );
        }
    }

    /// Register the exact sublease retained by a DMA window. The kernel derives
    /// all geometry and attributes from that lease; there is no public registrar.
    #[cfg(target_os = "none")]
    pub fn register_dma_fabric(
        window: &DmaWindow,
    ) -> Result<crate::cap::FabricRegistrationToken, FabricRegistrationFault> {
        register_fabric_lease(window.dma_lease_slot())
    }

    /// Register an owned supply or sublease before creating any CPU or device
    /// aliases, so exclusive mapping routes can be admitted without conflict.
    #[cfg(target_os = "none")]
    pub fn register_fabric_lease(
        lease: SlotRef,
    ) -> Result<crate::cap::FabricRegistrationToken, FabricRegistrationFault> {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};
        let owned = crate::cap::OwnedSlot::reserve().ok_or(FabricRegistrationFault::Slot)?;
        let mut builder = SessionBuilder::new();
        builder
            .register_fabric_lease(lease, owned.slot())
            .map_err(|_| FabricRegistrationFault::Build)?;
        let committed = TrapTransport::new()
            .commit_now(&builder)
            .map_err(|_| FabricRegistrationFault::Build)?;
        if committed.status() != kore_abi::session::Status::Ok {
            return Err(FabricRegistrationFault::Commit(committed.status() as u8));
        }
        Ok(crate::cap::FabricRegistrationToken::new(owned))
    }

    #[cfg(not(target_os = "none"))]
    pub fn register_dma_fabric(
        _window: &DmaWindow,
    ) -> Result<crate::cap::FabricRegistrationToken, FabricRegistrationFault> {
        Err(FabricRegistrationFault::Slot)
    }

    #[cfg(not(target_os = "none"))]
    pub fn register_fabric_lease(
        _lease: SlotRef,
    ) -> Result<crate::cap::FabricRegistrationToken, FabricRegistrationFault> {
        Err(FabricRegistrationFault::Slot)
    }

    /// Which stage of a peer-aperture publish refused.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum PeerWindowFault {
        Bounds,
        Slot,
        Build,
        Commit(u8),
        /// Publication may have occurred. The result name is quarantined;
        /// the device/BAR owners must contain the mapping before reuse.
        Indeterminate,
    }

    /// A single, non-waiting peer-map invocation can only succeed or roll
    /// back at index zero. Unknown status decoding is not rollback evidence.
    #[cfg(any(test, target_os = "none"))]
    fn settle_peer_publish(
        owned: &mut crate::cap::OwnedSlot,
        completion: Result<kore_abi::session::Committed, crate::transport::TrapUnavailable>,
    ) -> Result<(), PeerWindowFault> {
        // TrapUnavailable is issued only before raw_trap (descriptor encoding
        // or absent scratch). It proves this invocation did not publish.
        let completion = completion.map_err(|_| PeerWindowFault::Build)?;
        match (completion.raw() as u16, completion.failing_index()) {
            (0, 0) => return Ok(()),
            (status @ 1..=6, 0) => return Err(PeerWindowFault::Commit(status as u8)),
            _ => {}
        }
        owned.quarantine_unknown_publish();
        Err(PeerWindowFault::Indeterminate)
    }

    /// Publish one exact BAR subrange into a requester device's IOMMU domain.
    /// The stage-1 token is consumed transiently by the kernel as sponsorship;
    /// the returned affine window owns the public IOVA child and borrows every
    /// wider lifecycle authority for its entire usable lifetime.
    #[cfg(target_os = "none")]
    pub fn peer_window<'a>(
        bar: &'a BarWindow,
        sponsor: &'a DeviceMmuToken,
        requester: &'a DeviceToken,
        offset: u64,
        iova: u64,
        len: u64,
        access: PeerDmaAccess,
    ) -> Result<PeerWindow<'a>, PeerWindowFault> {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        const PAGE: u64 = 4096;
        if len == 0
            || offset & (PAGE - 1) != 0
            || iova & (PAGE - 1) != 0
            || len & (PAGE - 1) != 0
            || offset.checked_add(len).is_none_or(|end| end > bar.len())
            || iova.checked_add(len).is_none()
        {
            return Err(PeerWindowFault::Bounds);
        }
        let mut owned = crate::cap::OwnedSlot::reserve().ok_or(PeerWindowFault::Slot)?;
        let mut b = SessionBuilder::new();
        b.share_peer_device(
            bar.slot(),
            access.rights(),
            requester.slot(),
            sponsor.slot(),
            iova,
            offset,
            len,
            access.wire(),
            owned.slot(),
        )
        .map_err(|_| PeerWindowFault::Build)?;
        let completion = TrapTransport::new().commit_now(&b);
        settle_peer_publish(&mut owned, completion)?;
        Ok(PeerWindow::new(owned, bar.slot(), iova, len))
    }

    #[cfg(not(target_os = "none"))]
    pub fn peer_window<'a>(
        _bar: &'a BarWindow,
        _sponsor: &'a DeviceMmuToken,
        _requester: &'a DeviceToken,
        _offset: u64,
        _iova: u64,
        _len: u64,
        _access: PeerDmaAccess,
    ) -> Result<PeerWindow<'a>, PeerWindowFault> {
        Err(PeerWindowFault::Slot)
    }

    #[cfg(target_os = "none")]
    pub fn establish_interrupt(dev: &DeviceToken) -> Option<RouteToken> {
        establish_interrupts(dev, 0, 1, None)
    }

    /// Establish one consecutive MSI/MSI-X block. Any vector in the block
    /// wakes the returned affine route; the driver drains its bounded device
    /// completion queues before parking again.
    #[cfg(target_os = "none")]
    pub fn establish_interrupts(
        dev: &DeviceToken,
        first_entry: u16,
        count: u16,
        affinity: Option<u32>,
    ) -> Option<RouteToken> {
        establish_interrupts_with_source(dev, first_entry, count, affinity, false)
    }

    /// Establish a consecutive block whose device source is provably MSI-X.
    ///
    /// The kernel still requires interrupt remapping; this additionally
    /// refuses an MSI-only function. Use it when a device protocol assigns a
    /// distinct meaning to the selected MSI-X table entry and an aggregate MSI
    /// message would not be semantically interchangeable.
    #[cfg(target_os = "none")]
    pub fn establish_msix_interrupts(
        dev: &DeviceToken,
        first_entry: u16,
        count: u16,
        affinity: Option<u32>,
    ) -> Option<RouteToken> {
        establish_interrupts_with_source(dev, first_entry, count, affinity, true)
    }

    #[cfg(target_os = "none")]
    fn establish_interrupts_with_source(
        dev: &DeviceToken,
        first_entry: u16,
        count: u16,
        affinity: Option<u32>,
        require_msix: bool,
    ) -> Option<RouteToken> {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let owned = crate::cap::OwnedSlot::reserve()?;
        let mut b = SessionBuilder::new();
        let staged = if require_msix {
            b.derive_msix_route_with(
                dev.slot(),
                Rights::ACCESS | Rights::REVOKE,
                owned.slot(),
                first_entry,
                count,
                affinity,
            )
        } else {
            b.derive_irq_route_with(
                dev.slot(),
                Rights::ACCESS | Rights::REVOKE,
                owned.slot(),
                first_entry,
                count,
                affinity,
            )
        };
        staged.ok()?;
        let committed = TrapTransport::new().commit_now(&b).ok()?;
        if committed.status() != kore_abi::session::Status::Ok {
            return None;
        }
        Some(RouteToken::new(owned, count))
    }

    #[cfg(not(target_os = "none"))]
    pub fn establish_interrupt(_dev: &DeviceToken) -> Option<RouteToken> {
        None
    }

    #[cfg(not(target_os = "none"))]
    pub fn establish_interrupts(
        _dev: &DeviceToken,
        _first_entry: u16,
        _count: u16,
        _affinity: Option<u32>,
    ) -> Option<RouteToken> {
        None
    }

    #[cfg(not(target_os = "none"))]
    pub fn establish_msix_interrupts(
        _dev: &DeviceToken,
        _first_entry: u16,
        _count: u16,
        _affinity: Option<u32>,
    ) -> Option<RouteToken> {
        None
    }

    pub fn release_route(route: RouteToken) -> bool {
        route.retire()
    }

    /// Outcome of one blocking wait on a held interrupt route.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum InterruptWait {
        /// Hardware delivered through the route.
        Fired,
        /// The route was revoked while this task was parked.
        Revoked,
        /// The deadline expired first — [`wait_interrupt_until`] only.
        /// The route stays held and armed; the caller polls its device
        /// once and decides whether to wait again.
        Deadline,
    }

    /// Why [`wait_interrupt`] could not complete one route-backed wait.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum InterruptWaitError {
        /// The target has no live parking trap or descriptor scratch.
        TransportUnavailable,
        /// The kernel could not park the task (bring-up/inactive scheduler).
        WouldBlock,
        /// The kernel refused the wait session with this status.
        Refused(kore_abi::session::Status),
    }

    /// Park this dedicated IRQ relay task on its held affine route.
    ///
    /// The empty session carries `WAIT_IRQ | WAIT_IRQ_SELECTED` and the route's
    /// arena slot. The kernel authenticates that record and parks on its exact
    /// held `IrqRoute`, even when the task holds several routes. The borrow
    /// keeps the affine wait authority alive for the call. There is no polling fallback.
    ///
    /// **Unbounded**: a machine whose delivery path never fires parks
    /// here forever, with no edge left to refuse on.  A caller that
    /// cannot bet its boot on delivery wants
    /// [`wait_interrupt_until`] instead.
    pub fn wait_interrupt(route: &RouteToken) -> Result<InterruptWait, InterruptWaitError> {
        use crate::session::SessionBuilder;
        use crate::transport::{ParkTransport, TrapTransport};
        use kore_abi::session::Status;

        let mut wait = SessionBuilder::new();
        wait.wait_irq_on(route.slot());
        let committed = TrapTransport::new()
            .commit_irq_park(&wait)
            .map_err(|_| InterruptWaitError::TransportUnavailable)?;
        match committed.status() {
            Status::Ok => Ok(InterruptWait::Fired),
            Status::Revoked => Ok(InterruptWait::Revoked),
            Status::WouldBlock => Err(InterruptWaitError::WouldBlock),
            status => Err(InterruptWaitError::Refused(status)),
        }
    }

    /// Park on the held route until it fires **or** `deadline_ns`
    /// (absolute monotonic nanoseconds) passes — the **sleep-or-IRQ
    /// park** (`WAIT_IRQ | WAIT_IRQ_SELECTED | WAIT_SLEEP`, no delivery arm).
    ///
    /// This is the honest shape for the wait every driver actually
    /// wants: *prefer* the route's real delivery, but never bet
    /// liveness on it.  A bounded spin burns a core and calls the answer
    /// "timeout" without ever having waited; [`wait_interrupt`] waits
    /// properly but has no edge to refuse on when delivery is dead — the
    /// wedge that cost the first full metal boot (readiness parked
    /// inside the verb: no NACK, no beacon, every row behind it
    /// starved).  Park-with-deadline-then-poll-once is both: the CPU is
    /// yielded while waiting, and a delivery-dead machine still resolves
    /// [`InterruptWait::Deadline`] with its route intact — poll the
    /// device, then decide whether to wait again.
    ///
    /// A [`Deadline`](InterruptWait::Deadline) outcome is **truthful,
    /// not exclusive**: it means the deadline was due at resolve time,
    /// which an IRQ that landed in the same instant does not change.
    /// The route's pending-fire word coalesces that fire into the *next*
    /// park, so nothing is lost by polling and waiting again.
    ///
    /// `deadline_ns` is on the kernel's monotonic clock — the same one
    /// [`Executor::now`](crate::Executor::now) tracks (it anchors on
    /// every park's status stamp), so `ex.now() + delta` is the correct
    /// way to name one.  An already-past deadline is lawful: it parks,
    /// resolves `Deadline` at once, and costs one trap.
    ///
    /// The `route` borrow is the same wait-authority witness
    /// [`wait_interrupt`] takes.
    pub fn wait_interrupt_until(
        route: &RouteToken,
        deadline_ns: u64,
    ) -> Result<InterruptWait, InterruptWaitError> {
        use crate::session::SessionBuilder;
        use crate::transport::{ParkTransport, TrapTransport};
        use kore_abi::session::Status;

        let mut wait = SessionBuilder::new();
        wait.wait_irq_on(route.slot()).wait_sleep(deadline_ns);
        let committed = TrapTransport::new()
            .commit_irq_park(&wait)
            .map_err(|_| InterruptWaitError::TransportUnavailable)?;
        match committed.status() {
            // The composite park answers `Irq` (the pure-`WAIT_IRQ`
            // park's `Ok` has no counterpart here — this shape resolves
            // through `resolve_irq_sleep`).
            Status::Irq => Ok(InterruptWait::Fired),
            Status::Deadline => Ok(InterruptWait::Deadline),
            Status::Revoked => Ok(InterruptWait::Revoked),
            Status::WouldBlock => Err(InterruptWaitError::WouldBlock),
            status => Err(InterruptWaitError::Refused(status)),
        }
    }

    /// **Park until `deadline_ns`** (absolute monotonic nanoseconds) — the
    /// pure `WAIT_SLEEP` sliver (USERLAND.md §3.4.2): no doorbell, no
    /// route, just the per-CPU timer wheel.  The thread shape (#407): a
    /// spawned thread with no wake edges paces itself on deadline parks.
    /// Returns the committed status word whole so the caller can read the
    /// kernel's clock stamp (`[63:32]`, ms) — an already-past deadline is
    /// lawful (parks, resolves at once) and is exactly how a fresh thread
    /// learns "now".  `None` = the transport refused (host builds).
    pub fn sleep_until(deadline_ns: u64) -> Option<u64> {
        use crate::session::SessionBuilder;
        use crate::transport::{ParkTransport, TrapTransport};

        let mut wait = SessionBuilder::new();
        wait.wait_sleep(deadline_ns);
        TrapTransport::new()
            .commit_irq_park(&wait)
            .ok()
            .map(|c| c.raw())
    }

    /// Issue the authenticated, bind-inert task START effect.  START borrows
    /// its handle and never consumes the arena name, so every retry outcome
    /// leaves the caller's affine token intact.
    #[cfg(target_os = "none")]
    pub fn start_task(slot: SlotRef) -> TaskStartOutcome {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let mut start = SessionBuilder::new();
        if start.start(slot).is_err() {
            return TaskStartOutcome::Unavailable;
        }
        let Ok(committed) = TrapTransport::new().commit_now(&start) else {
            return TaskStartOutcome::Unavailable;
        };
        classify_task_start(committed)
    }

    /// Host/non-metal builds never simulate task authority.
    #[cfg(not(target_os = "none"))]
    pub fn start_task(_slot: SlotRef) -> TaskStartOutcome {
        TaskStartOutcome::Unavailable
    }

    /// Retire one affine token's public arena name.  `Status::Revoked` proves
    /// this record or an ancestor is already dead, which is distinct from a
    /// revoke committed by this call but is equally safe for local-name reuse.
    /// Every other refusal remains retryable and keeps the slot quarantined.
    #[cfg(target_os = "none")]
    pub(crate) fn retire_owned_slot(slot: SlotRef) -> RetireOutcome {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let mut b = SessionBuilder::new();
        if b.revoke(slot).is_err() {
            return RetireOutcome::Retryable;
        }
        let Ok(committed) = TrapTransport::new().commit_now(&b) else {
            return RetireOutcome::Retryable;
        };
        classify_owned_retirement(committed)
    }

    #[cfg(not(target_os = "none"))]
    pub(crate) fn retire_owned_slot(_slot: SlotRef) -> RetireOutcome {
        RetireOutcome::Retryable
    }

    /// Retire one affine mapped-view name through exact unshare. Unlike plain
    /// revoke, this operation validates the direct parent lease and removes
    /// the precise share-list edge and PTE before proving closure.
    #[cfg(target_os = "none")]
    pub(crate) fn retire_owned_view(view: SlotRef, lease: SlotRef) -> RetireOutcome {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let mut b = SessionBuilder::new();
        if b.unshare(view, lease).is_err() {
            return RetireOutcome::Retryable;
        }
        let Ok(committed) = TrapTransport::new().commit_now(&b) else {
            return RetireOutcome::Retryable;
        };
        classify_owned_retirement(committed)
    }

    #[cfg(not(target_os = "none"))]
    pub(crate) fn retire_owned_view(_view: SlotRef, _lease: SlotRef) -> RetireOutcome {
        RetireOutcome::Retryable
    }

    /// Retire one public peer-IOMMU view through exact unshare. Plain revoke
    /// would kill only the capability name because `LeasedIova::teardown` is a
    /// no-op; this verb authenticates the direct BAR parent and removes the
    /// precise generational hardware window before reporting closure.
    #[cfg(target_os = "none")]
    pub(crate) fn retire_owned_iova(view: SlotRef, parent: SlotRef) -> RetireOutcome {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let mut b = SessionBuilder::new();
        if b.unshare_iova(view, parent).is_err() {
            return RetireOutcome::Retryable;
        }
        let Ok(committed) = TrapTransport::new().commit_now(&b) else {
            return RetireOutcome::Retryable;
        };
        classify_owned_retirement(committed)
    }

    #[cfg(not(target_os = "none"))]
    pub(crate) fn retire_owned_iova(_view: SlotRef, _parent: SlotRef) -> RetireOutcome {
        RetireOutcome::Retryable
    }

    /// Retire a fixed landing-slot capability without an allocator
    /// reservation. This is the client grant-bank cleanup face: callers keep
    /// the slot name when the outcome is retryable and may forget it only once
    /// the revoke committed or the lineage was already dead.
    #[inline]
    pub fn retire_slot_outcome(slot: SlotRef) -> RetireOutcome {
        retire_owned_slot(slot)
    }

    /// Retire one fixed-slot mapped view through exact unshare.
    ///
    /// This is the fixed-arena counterpart of [`MappedLeaseView`]'s affine
    /// retirement: `view` must name an installed `LeasedRegion` whose direct
    /// parent is the held `lease`.  The kernel validates that relationship and
    /// removes the precise share-list edge and PTE before reporting closure.
    /// A plain [`retire_slot_outcome`] is not a substitute because revoking a
    /// view name alone does not hard-unshare its mapping.
    ///
    /// [`MappedLeaseView`]: crate::cap::MappedLeaseView
    #[inline]
    pub fn retire_mapped_view(view: SlotRef, lease: SlotRef) -> RetireOutcome {
        retire_owned_view(view, lease)
    }

    /// Carve one affine DMA sub-lease, map its CPU face, zero it, and only then
    /// publish the same frames into the device's IOMMU domain.  The sub-lease
    /// slot is the sole public lifecycle name; its IOVA child is anonymous and
    /// tracked kernel-side beneath it.
    /// Which stage of [`dma_window`] refused — [`BarMapFault`]'s DMA
    /// sibling, bought by the same lesson: the pool's whole CPU view and
    /// the device's IOVA leg both live inside what used to be one `None`.
    /// The commit stages carry the kernel's
    /// [`Status`](kore_abi::session::Status) discriminant.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum DmaMapFault {
        /// The request failed its own bounds (offset/len/iova shape).
        Bounds,
        /// No report slot, or the task's cap arena had no free slot.
        Slot,
        /// The carve session refused to build/transport, or committed
        /// non-Ok (status in the payload; `0` = build).
        Carve(u8),
        /// The CPU-view map session refused (status; `0` = build).
        Map(u8),
        /// The geometry report was absent, malformed, or short.
        Geometry,
        /// The device-publish session refused (status; `0` = build) —
        /// the IOVA leg: domain attach, window registration, the map.
        Publish(u8),
        /// Fabric registration (1), observation (2), or route admission (3) refused.
        Fabric(u8),
    }

    impl DmaMapFault {
        /// Pack as `stage[2:0] | status << 3` — the note-detail byte.
        pub fn code(self) -> u8 {
            match self {
                Self::Bounds => 1,
                Self::Slot => 2,
                Self::Carve(s) => 3 | (s << 3),
                Self::Map(s) => 4 | (s << 3),
                Self::Geometry => 5,
                Self::Publish(s) => 6 | (s << 3),
                Self::Fabric(s) => 7 | (s << 3),
            }
        }
    }

    /// Map a complete held lease R/W at the kernel-selected address while
    /// retaining an exact-unshare owner. The parent lease must remain held
    /// until this view closes; no caller-selected VA or second mapping occurs.
    #[cfg(target_os = "none")]
    pub fn map_owned_lease_rw(lease: SlotRef) -> Option<MappedLeaseView> {
        map_named_cpu_view(lease, crate::entry::geometry::report_slot()?).ok()
    }

    #[cfg(not(target_os = "none"))]
    pub fn map_owned_lease_rw(_lease: SlotRef) -> Option<MappedLeaseView> {
        None
    }

    #[cfg(target_os = "none")]
    fn map_named_cpu_view(lease: SlotRef, report: SlotRef) -> Result<MappedLeaseView, DmaMapFault> {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};
        let mut owner = crate::cap::OwnedSlot::reserve().ok_or(DmaMapFault::Slot)?;
        let mut map = SessionBuilder::new();
        let view = map
            .map_owned_named(
                lease,
                Rights::ACCESS | Rights::MODIFY | Rights::REVOKE,
                owner.slot(),
            )
            .map_err(|_| DmaMapFault::Map(0))?;
        map.cpu_view_geometry(view, report)
            .map_err(|_| DmaMapFault::Map(0))?;
        let committed = match TrapTransport::new().commit_now(&map) {
            Ok(committed) => committed,
            Err(_) => {
                owner.quarantine_unknown_publish();
                return Err(DmaMapFault::Map(0));
            }
        };
        if committed.status() != kore_abi::session::Status::Ok {
            return Err(DmaMapFault::Map(committed.status() as u8));
        }
        let geometry = crate::entry::geometry::slot_record_va(report)
            .and_then(|ptr| {
                kore_abi::session::GeomReport::decode(&unsafe {
                    core::ptr::read_volatile(ptr as *const kore_abi::cap::CapRecord)
                })
            })
            .and_then(|g| {
                let base = NonNull::new(usize::try_from(g.base).ok()? as *mut u8)?;
                let len = usize::try_from(g.len).ok()?;
                (len != 0 && g.base.checked_add(g.len).is_some()).then_some((base, len))
            });
        let Some((base, len)) = geometry else {
            // Publication happened, so a missing report cannot free this name.
            owner.quarantine_unknown_publish();
            return Err(DmaMapFault::Geometry);
        };
        Ok(MappedLeaseView::new(owner, lease, base, len))
    }

    /// Pool storage is initialized before publication; imported storage belongs
    /// to a producer and must survive mapping unchanged.
    #[cfg(target_os = "none")]
    enum DmaContents {
        Zero,
        Preserve,
    }

    #[cfg(target_os = "none")]
    fn map_route_dma_views(
        route: &crate::cap::FabricRouteToken,
        device: SlotRef,
        iova: u64,
        len: u64,
        contents: DmaContents,
        device_write: bool,
    ) -> Result<(MappedLeaseView, crate::cap::OwnedSlot), DmaMapFault> {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};
        let report = crate::entry::geometry::report_slot().ok_or(DmaMapFault::Slot)?;
        let mapping_slot = route.slot();
        let cpu_view = map_named_cpu_view(mapping_slot, report)?;
        if cpu_view.len as u64 != len {
            return Err(DmaMapFault::Geometry);
        }

        if matches!(contents, DmaContents::Zero) {
            // SAFETY: the successful map installed the pool sub-lease RW for
            // exactly `len` bytes. No device leaf exists yet. Imports never
            // enter this branch: their producer already populated the bytes.
            unsafe { core::ptr::write_bytes(cpu_view.as_mut_ptr(), 0, len as usize) };
        }

        // Both mapping names remain independently retireable before their
        // shared route, registration and allocation can close.
        let mut device_owner = crate::cap::OwnedSlot::reserve().ok_or(DmaMapFault::Slot)?;
        let mut publish = SessionBuilder::new();
        // Device MODIFY follows device writes, independently of CPU access.
        let device_rights = if device_write {
            Rights::ACCESS | Rights::MODIFY
        } else {
            Rights::ACCESS
        };
        publish
            .share_device(
                mapping_slot,
                device_rights | Rights::REVOKE,
                device,
                iova,
                len,
                device_write,
                device_owner.slot(),
            )
            .map_err(|_| DmaMapFault::Publish(0))?;
        let committed = match TrapTransport::new().commit_now(&publish) {
            Ok(committed) => committed,
            Err(_) => {
                device_owner.quarantine_unknown_publish();
                return Err(DmaMapFault::Publish(0));
            }
        };
        if committed.status() != kore_abi::session::Status::Ok {
            return Err(DmaMapFault::Publish(committed.status() as u8));
        }
        device_owner.activate();

        Ok((cpu_view, device_owner))
    }

    #[cfg(target_os = "none")]
    fn dma_window_for_slot(
        pool: SupplyToken,
        device: SlotRef,
        offset: u64,
        len: u64,
        iova: u64,
        contents: DmaContents,
        device_write: bool,
    ) -> Result<(DmaWindow, Option<SupplyToken>), DmaMapFault> {
        use kore_abi::session::GeomReport;

        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        const PAGE: u64 = 4096;
        if len == 0
            || offset & (PAGE - 1) != 0
            || len & (PAGE - 1) != 0
            || iova & (PAGE - 1) != 0
            || offset.checked_add(len).is_none_or(|end| end > pool.len())
            || iova.checked_add(len).is_none()
        {
            return Err(DmaMapFault::Bounds);
        }

        let full = Rights::ACCESS | Rights::MODIFY | Rights::DERIVE | Rights::REVOKE;
        let mut derived_owner = None;
        let window_slot = if pool.is_returned() {
            // Preserve the provisioner's exact allocation owner for return;
            // do not replace it with an unnecessary further sublease.
            if offset != 0 || len != pool.len() {
                return Err(DmaMapFault::Bounds);
            }
            pool.slot()
        } else {
            let mut owned = crate::cap::OwnedSlot::reserve().ok_or(DmaMapFault::Slot)?;
            let mut carve = SessionBuilder::new();
            carve
                .carve_sublease(pool.slot(), full, offset, len, owned.slot())
                .map_err(|_| DmaMapFault::Carve(0))?;
            let committed = TrapTransport::new()
                .commit_now(&carve)
                .map_err(|_| DmaMapFault::Carve(0))?;
            if committed.status() != kore_abi::session::Status::Ok {
                return Err(DmaMapFault::Carve(committed.status() as u8));
            }
            owned.activate();
            let slot = owned.slot();
            derived_owner = Some(owned);
            slot
        };

        // Admit the whole window before creating aliases. Pool allocations and
        // imported storage share this custody chain; only initialization differs.
        let registration =
            register_fabric_lease(window_slot).map_err(|_| DmaMapFault::Fabric(1))?;
        let facts = registration
            .describe()
            .map_err(|_| DmaMapFault::Fabric(2))?;
        let request = kore_abi::session::FabricRouteReq {
            destination: facts.domain.into(),
            offset: 0.into(),
            bytes: len.into(),
            source_reset: facts.reset.into(),
            source_power: facts.power.into(),
            source_allocation: facts.allocation.into(),
            destination_reset: facts.reset.into(),
            destination_power: facts.power.into(),
            destination_allocation: facts.allocation.into(),
            access: 7,
            operation: 1,
            // Fresh pool storage or CPU-produced import protocol handoff;
            // this does not attest a foreign device's fence.
            synchronization: 1,
            reserved: [0; 5],
        };
        let route = registration
            .open_route(&request)
            .map_err(|_| DmaMapFault::Fabric(3))?;
        let (cpu_view, device_owner) =
            map_route_dma_views(&route, device, iova, len, contents, device_write)?;
        let geom = GeomReport {
            base: cpu_view.as_ptr() as u64,
            len,
        };

        let (window, retained_parent) = if let Some(owned) = derived_owner {
            (
                DmaWindow::new(geom.base as *mut u8, iova, len, owned),
                Some(pool),
            )
        } else {
            let owned = pool.into_returned_owner().map_err(|_| DmaMapFault::Slot)?;
            (DmaWindow::new(geom.base as *mut u8, iova, len, owned), None)
        };
        let window = window
            .with_device_leaf(device_owner)
            .with_cpu_view(cpu_view)
            .with_fabric_route(registration, route);
        Ok((window, retained_parent))
    }

    /// Carve and publish one DMA window to a PCI function.
    #[cfg(target_os = "none")]
    pub fn dma_window(
        pool: SupplyToken,
        device: &DeviceToken,
        offset: u64,
        len: u64,
        iova: u64,
    ) -> Result<(DmaWindow, Option<SupplyToken>), DmaMapFault> {
        dma_window_for_slot(
            pool,
            device.slot(),
            offset,
            len,
            iova,
            DmaContents::Zero,
            true,
        )
    }

    /// Carve and publish one DMA window to a firmware-described platform
    /// device. The session engine authenticates the platform-device slot and
    /// resolves its firmware StreamID through the same requester seam as PCI.
    #[cfg(target_os = "none")]
    pub fn dma_window_platform(
        pool: SupplyToken,
        device: &PlatformToken,
        offset: u64,
        len: u64,
        iova: u64,
    ) -> Result<(DmaWindow, Option<SupplyToken>), DmaMapFault> {
        dma_window_for_slot(
            pool,
            device.slot(),
            offset,
            len,
            iova,
            DmaContents::Zero,
            true,
        )
    }

    #[cfg(not(target_os = "none"))]
    pub fn dma_window(
        _pool: SupplyToken,
        _device: &DeviceToken,
        _offset: u64,
        _len: u64,
        _iova: u64,
    ) -> Result<(DmaWindow, Option<SupplyToken>), DmaMapFault> {
        Err(DmaMapFault::Slot)
    }

    /// Map and publish a whole returned lease without changing its contents.
    /// The returned window owns the lease, CPU mapping and a named device mapping. The
    /// producer must hand off CPU access before invoking this operation.
    /// `device_write` controls the outer IOMMU permission independently of the
    /// CPU mapping. Input-only imports must pass false even when CPU writes
    /// are allowed by the lease.
    pub fn import_dma_window(
        lease: SupplyToken,
        device: &DeviceToken,
        iova: u64,
        device_write: bool,
    ) -> Result<DmaWindow, DmaMapFault> {
        #[cfg(target_os = "none")]
        {
            if !lease.is_returned() {
                return Err(DmaMapFault::Bounds);
            }
            let len = lease.len();
            let (window, parent) = dma_window_for_slot(
                lease,
                device.slot(),
                0,
                len,
                iova,
                DmaContents::Preserve,
                device_write,
            )?;
            debug_assert!(parent.is_none());
            Ok(window)
        }
        #[cfg(not(target_os = "none"))]
        {
            let _ = (lease, device, iova, device_write);
            Err(DmaMapFault::Slot)
        }
    }

    /// Map a transferred route using its existing admitted custody chain.
    /// The caller has checked immutable route info before passing ownership.
    pub fn import_route_dma_window(
        route: crate::cap::FabricRouteToken,
        device: &DeviceToken,
        iova: u64,
        len: u64,
        device_write: bool,
    ) -> Result<DmaWindow, DmaMapFault> {
        #[cfg(target_os = "none")]
        {
            if len == 0 || len % 4096 != 0 || iova % 4096 != 0 || iova.checked_add(len).is_none() {
                return Err(DmaMapFault::Bounds);
            }
            let info = route.describe().map_err(|_| DmaMapFault::Fabric(2))?;
            if info.bytes != len || info.operation != 1 || info.access & 3 != 3 {
                return Err(DmaMapFault::Bounds);
            }
            let (cpu_view, device_owner) = map_route_dma_views(
                &route,
                device.slot(),
                iova,
                len,
                DmaContents::Preserve,
                device_write,
            )?;
            Ok(
                DmaWindow::from_route(cpu_view.as_mut_ptr(), iova, len, route)
                    .with_cpu_view(cpu_view)
                    .with_device_leaf(device_owner),
            )
        }
        #[cfg(not(target_os = "none"))]
        {
            let _ = (route, device, iova, len, device_write);
            Err(DmaMapFault::Slot)
        }
    }

    /// Which stage of [`share_window`] refused.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum WindowShareFault {
        /// The task has no report slot to receive the (vacant) result.
        Slot,
        /// The session would not build — a malformed request shape.
        Build,
        /// The kernel refused the publish; the payload is its
        /// [`Status`](kore_abi::session::Status) discriminant.
        Commit(u8),
    }

    /// **Install a lease this task did not carve into a device's IOMMU
    /// domain** — the *client-window* half of the DMA axis.
    ///
    /// [`dma_window`] carves, maps, zeroes and publishes a driver-owned pool.
    /// This publishes a lease the driver received: a client's window,
    /// delivered over a return leg, whose bytes the client formatted and
    /// whose CPU half is already mapped. It is what makes a frame service's
    /// transmit path genuinely zero-copy — the device fetches out of the
    /// client's own slot rather than a bounce the service refilled.
    ///
    /// The leaf is **anonymous and parented to the lease** (Σ4: one parent),
    /// so retiring the lease reaps this device mapping with it — which is what
    /// lets a service make "the window is unmapped only once no transfer is
    /// outstanding" a property of one affine owner.
    ///
    /// `write` is the *device's* access. A frame service passes `false`: the
    /// controller reads frames out of the window and must never be able to
    /// write into a client's memory through a leaf the client did not ask for.
    #[cfg(target_os = "none")]
    pub fn share_window(
        lease: SlotRef,
        device: &DeviceToken,
        iova: u64,
        bytes: u64,
        write: bool,
    ) -> Result<(), WindowShareFault> {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let report = crate::entry::geometry::report_slot().ok_or(WindowShareFault::Slot)?;
        let mut publish = SessionBuilder::new();
        publish
            .share_device_anonymous(
                lease,
                Rights::ACCESS,
                device.slot(),
                iova,
                bytes,
                write,
                report,
            )
            .map_err(|_| WindowShareFault::Build)?;
        let committed = TrapTransport::new()
            .commit_now(&publish)
            .map_err(|_| WindowShareFault::Build)?;
        if committed.status() != kore_abi::session::Status::Ok {
            return Err(WindowShareFault::Commit(committed.status() as u8));
        }
        Ok(())
    }

    /// Host builds have no live capability transport.
    #[cfg(not(target_os = "none"))]
    pub fn share_window(
        _lease: SlotRef,
        _device: &DeviceToken,
        _iova: u64,
        _bytes: u64,
        _write: bool,
    ) -> Result<(), WindowShareFault> {
        Err(WindowShareFault::Slot)
    }

    #[cfg(not(target_os = "none"))]
    pub fn dma_window_platform(
        _pool: SupplyToken,
        _device: &PlatformToken,
        _offset: u64,
        _len: u64,
        _iova: u64,
    ) -> Result<(DmaWindow, Option<SupplyToken>), DmaMapFault> {
        Err(DmaMapFault::Slot)
    }

    /// Open the device-side DMA latch for one fully mapped, prepared window.
    /// The raw sub-lease slot never leaves `kore-rt`; the kernel authenticates
    /// both this device and the window owner again at the session boundary.
    #[cfg(target_os = "none")]
    fn activate_window(device: SlotRef, window: &DmaWindow) -> bool {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let mut activation = SessionBuilder::new();
        if activation
            .activate_device(device, window.dma_mapping_slot())
            .is_err()
        {
            return false;
        }
        TrapTransport::new()
            .commit_now(&activation)
            .is_ok_and(|committed| committed.status() == kore_abi::session::Status::Ok)
    }

    #[cfg(target_os = "none")]
    pub fn activate_device_window(device: &DeviceToken, window: &DmaWindow) -> bool {
        activate_window(device.slot(), window)
    }

    /// **Prove a PCI function quiet at the fabric** (#809 §L) — the terminal
    /// counterpart of [`activate_device_window`]: the kernel masks the
    /// function's interrupts, closes and reads back Bus Master Enable, binds
    /// its requester to Abort and quarantines its DMA epoch, then writes a
    /// `QuiesceReport` into this task's report slot.  The decoded report is
    /// the one and only source of a [`Quiesced`] witness.
    ///
    /// `MODIFY` on the device is the gate, re-checked by the kernel through
    /// the same projection every other device mutation uses.  A refusal
    /// leaves the epoch quarantined fail-closed on the kernel side; the
    /// class host records it and containment runs regardless.  The class
    /// driver must have stopped its own engine *first*: after this the
    /// function cannot reach memory and its interrupts are masked, so a
    /// controller shutdown that needs either would wedge.
    #[cfg(target_os = "none")]
    pub fn quiesce_device(device: &DeviceToken) -> Result<Quiesced, QuiesceFault> {
        use kore_abi::cap::CapRecord;
        use kore_abi::session::{QuiesceReport, Status};

        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let report_slot = crate::entry::geometry::report_slot().ok_or(QuiesceFault::Slot)?;
        let mut b = SessionBuilder::new();
        b.quiesce_device(device.slot(), report_slot)
            .map_err(|_| QuiesceFault::Build)?;
        let committed = TrapTransport::new()
            .commit_now(&b)
            .map_err(|_| QuiesceFault::Transport)?;
        if committed.status() != Status::Ok {
            return Err(QuiesceFault::Refused(committed.status() as u8));
        }
        // Decode the bind-inert verdict off the R/O slots alias.
        // SAFETY: the founder mapped the slots alias R/O over this task's
        // own records; the kernel wrote the report before the trap
        // returned.  Read volatile.
        let rptr = crate::entry::geometry::slot_record_va(report_slot).ok_or(QuiesceFault::Slot)?
            as *const CapRecord;
        let rec = unsafe { core::ptr::read_volatile(rptr) };
        QuiesceReport::decode(&rec)
            .and_then(Quiesced::minted)
            .ok_or(QuiesceFault::Report)
    }

    /// Off the metal there is no fabric to prove quiet.
    #[cfg(not(target_os = "none"))]
    pub fn quiesce_device(_device: &DeviceToken) -> Result<Quiesced, QuiesceFault> {
        Err(QuiesceFault::Unenforced)
    }

    /// Open a firmware-described platform device's DMA latch after every
    /// prepared IOVA leaf has been published.
    #[cfg(target_os = "none")]
    pub fn activate_platform_window(device: &PlatformToken, window: &DmaWindow) -> bool {
        activate_window(device.slot(), window)
    }

    /// Host builds have no live capability transport.
    #[cfg(not(target_os = "none"))]
    pub fn activate_device_window(_device: &DeviceToken, _window: &DmaWindow) -> bool {
        false
    }

    #[cfg(not(target_os = "none"))]
    pub fn activate_platform_window(_device: &PlatformToken, _window: &DmaWindow) -> bool {
        false
    }

    /// **The metal grant return** (the lessor's take-back — mm's
    /// `Release`) — stage one `Revoke × REVOKE_KIND_RETURN` naming the
    /// delivered-back `view` slot as the subject and `supply`'s slot as
    /// the owning lease (MODIFY-gated), committed through the LIVE session
    /// engine via the trap transport.  The kernel retires the view's tag,
    /// unmaps **every leaf its receivers installed over the window** (any
    /// task, any space — the window-scoped clear), and answers with the
    /// window's kernel-vouched `(physical base, len)` as a bind-inert
    /// [`GeomReport`](kore_abi::session::GeomReport) in this task's report
    /// slot — a delivered record carries no offset/len, and a
    /// client-echoed range would let a tenant poison the lessor's free
    /// list.  Any refusal — no report slot, a non-`Ok` commit, a
    /// non-report record — is a fail-closed `None`.
    #[cfg(target_os = "none")]
    pub fn reclaim_extent(view: SlotRef, supply: &SupplyToken) -> Option<(u64, u64)> {
        use kore_abi::cap::CapRecord;
        use kore_abi::session::GeomReport;

        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let report_slot = crate::entry::geometry::report_slot()?;
        let mut b = SessionBuilder::new();
        b.return_grant(view, supply.slot(), report_slot).ok()?;
        let committed = TrapTransport::new().commit_now(&b).ok()?;
        if committed.status() != kore_abi::session::Status::Ok {
            return None;
        }
        // Decode the bind-inert report off the R/O slots alias.  SAFETY:
        // the founder mapped the slots alias R/O over this task's own
        // records; the kernel wrote the report before the trap returned.
        // Read volatile.
        let ptr = crate::entry::geometry::slot_record_va(report_slot)? as *const CapRecord;
        let g = GeomReport::decode(&unsafe { core::ptr::read_volatile(ptr) })?;
        if g.len == 0 {
            return None;
        }
        Some((g.base, g.len))
    }

    /// Off the metal userland targets — the honest typed refusal
    /// (`res/USERLAND.md` §2.1): there is no trap to issue.
    #[cfg(not(target_os = "none"))]
    pub fn reclaim_extent(_view: SlotRef, _supply: &SupplyToken) -> Option<(u64, u64)> {
        None
    }
    /// Possess a delivered **`MemLease`** — the shape an image-service
    /// `Acquire` grant arrives in, and the one [`take_view`] refuses.
    ///
    /// A lease grant is an attenuation of the lessor's `MemLease`, not a
    /// `LeasedRegion` view, so every image grant read back through
    /// `take_view` answered `None` on metal: the client had no way to
    /// possess what the service had lawfully sent.  Same discipline as
    /// [`take_view`] otherwise — the record is read as *data* off the R/O
    /// slots alias, no `bind`, and the real authority is re-earned by
    /// slot at whatever session names it next.
    ///
    /// `None` for anything that is not a delivered lease.
    #[cfg(target_os = "none")]
    pub fn take_lease(slot: SlotRef) -> Option<LeaseToken> {
        use kore_abi::cap::{CapRecord, ResourceKind};

        // SAFETY: the R/O `SLOTS_VA` alias over this task's own
        // `slots_phys` (see `take_view`).
        let ptr = crate::entry::geometry::slot_record_va(slot)? as *const CapRecord;
        let rec = unsafe { core::ptr::read_volatile(ptr) };
        if rec.kind != ResourceKind::MemLease as u16 {
            return None;
        }
        // A delivered record carries no span — the geometry rides the
        // service's own descriptor; the slot is the authority.
        Some(LeaseToken::possess(slot, 0, rec.rights))
    }

    /// Off the metal userland targets — the honest typed refusal.
    #[cfg(not(target_os = "none"))]
    pub fn take_lease(_slot: SlotRef) -> Option<LeaseToken> {
        None
    }

    /// Possess a delivered fixed-slot `ChanSend` without manufacturing a
    /// dynamic arena owner. The returned affine witness can move the exact
    /// endpoint through the typed service return protocol; a successful move
    /// scrubs this task's source holding.
    #[cfg(target_os = "none")]
    pub fn take_chan_send(
        slot: SlotRef,
        required: kore_abi::cap::Rights,
    ) -> Option<HeldChanSendToken> {
        use kore_abi::cap::{CapRecord, ResourceKind};

        let ptr = crate::entry::geometry::slot_record_va(slot)? as *const CapRecord;
        // SAFETY: `slot_record_va` names this task's read-only cap-record
        // alias. The subsequent send re-authenticates the live record.
        let rec = unsafe { core::ptr::read_volatile(ptr) };
        if rec.kind != ResourceKind::ChanSend as u16 || !rec.rights.contains(required) {
            return None;
        }
        Some(HeldChanSendToken::possess(slot, rec.rights))
    }

    #[cfg(not(target_os = "none"))]
    pub fn take_chan_send(
        _slot: SlotRef,
        _required: kore_abi::cap::Rights,
    ) -> Option<HeldChanSendToken> {
        None
    }

    /// Possess a delivered fixed-slot `ChanRecv` without manufacturing a
    /// dynamic arena owner. The returned witness is affine; a copied slot
    /// coordinate alone is not receive authority.
    #[cfg(target_os = "none")]
    pub fn take_chan_recv(
        slot: SlotRef,
        required: kore_abi::cap::Rights,
    ) -> Option<HeldChanRecvToken> {
        use kore_abi::cap::{CapRecord, ResourceKind};

        let ptr = crate::entry::geometry::slot_record_va(slot)? as *const CapRecord;
        // SAFETY: `slot_record_va` names this task's read-only cap-record
        // alias. Every later post re-authenticates the live record by slot.
        let rec = unsafe { core::ptr::read_volatile(ptr) };
        if rec.kind != ResourceKind::ChanRecv as u16 || !rec.rights.contains(required) {
            return None;
        }
        Some(HeldChanRecvToken::possess(slot, rec.rights))
    }

    #[cfg(not(target_os = "none"))]
    pub fn take_chan_recv(
        _slot: SlotRef,
        _required: kore_abi::cap::Rights,
    ) -> Option<HeldChanRecvToken> {
        None
    }

    /// **The metal take-view** (`res/USERLAND.md` §5.6; Phase-A checkpoint
    /// C) — read a delivered `LeasedRegion` view record off the R/O
    /// cap-slot alias at [`SLOTS`](kore_abi::task::geom::SLOTS) alias and return
    /// a [`LeasedToken`] naming the slot.  Like [`recv_pin`] it takes
    /// **no** `bind`: the record is read as *data*, and `map_view`
    /// re-authenticates the real record *by slot* (CMAC ‖ holder ‖ tag
    /// liveness) at the map trap.  `None` for anything that is not a
    /// delivered view.  Reads `rec.kind` (the `ResourceKind::LeasedRegion`
    /// discriminant) and `rec.rights` (carried into the token).
    #[cfg(target_os = "none")]
    pub fn take_view(slot: SlotRef) -> Option<LeasedToken> {
        use kore_abi::cap::{CapRecord, ResourceKind};

        // SAFETY: the founder mapped `[SLOTS_VA, +ARENA_SLOTS·128)` R/O
        // (`READ|USER`, no `WRITE`) over THIS task's own `slots_phys`;
        // `slot` names a slot in range and 128-byte aligned within the span,
        // and the kernel is the sole writer of that frame.  Read it volatile.
        let ptr = crate::entry::geometry::slot_record_va(slot)? as *const CapRecord;
        let rec = unsafe { core::ptr::read_volatile(ptr) };
        if rec.kind != ResourceKind::LeasedRegion as u16 {
            return None;
        }
        // window_size has no metal home (the delivered `CapRecord` carries
        // no offset/len); the client tracks the founder-layout window and
        // `map_view` re-binds authority by slot.  Rights ride the record.
        Some(LeasedToken::possess(slot, 0, rec.rights))
    }

    /// Off the metal userland targets — the honest typed refusal
    /// (`res/USERLAND.md` §2.1): there is no fixed-VA alias to read.
    #[cfg(not(target_os = "none"))]
    pub fn take_view(_slot: SlotRef) -> Option<LeasedToken> {
        None
    }
    /// **The metal map-view** (`res/USERLAND.md` §5.6; #336 rung 2) — consume a
    /// possessed [`LeasedToken`] (a delivered `Leased` view, read off the R/O
    /// slots alias by [`take_view`]) and **install its backing into this task's
    /// own address space** through the LIVE session engine: one staged
    /// `Derive`-encoded map exercise naming the view's arena slot as the
    /// subject (the kernel's
    /// `Derive × LeasedRegion` map flavor), committed through the trap
    /// transport.  The kernel installs a leaf at this task's founder-installed
    /// map-window cursor — so a *second* map lands at the next span instead of
    /// refusing a fixed VA — and answers with a bind-inert
    /// [`GeomReport`](kore_abi::session::GeomReport) in the derived
    /// [`report_slot`](crate::entry): the verb decodes `(base, len)` off
    /// the R/O slots alias and returns a `NonNull` slice over it.  Any refusal
    /// — no report slot (a handle-less task), no geometry, a non-`Ok` commit,
    /// a non-report record — is a fail-closed `None`.
    ///
    /// The delivered view must carry `ACCESS`; `MODIFY` adds write permission.
    /// It deliberately carries no `DERIVE`: this exact held-view/no-arg
    /// operation is an exercise of a leaf, not authority to mint another view.
    /// The subject's tag is **not** consumed: the token is returned to the
    /// caller in spirit — the
    /// record stays live and may be mapped again (each map burns cursor span).
    /// Long-lived ring/queue users intentionally keep that mapping. A
    /// transient caller that knows the view's held immediate-parent lease must
    /// instead close it with [`retire_mapped_view`]; plain slot retirement does
    /// not remove the PTE or its typed share-list edge.
    #[cfg(target_os = "none")]
    pub fn map_view(view: LeasedToken) -> Option<NonNull<[u8]>> {
        use kore_abi::session::GeomReport;

        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let report_slot = crate::entry::geometry::report_slot()?;
        let mut b = SessionBuilder::new();
        b.derive_raw(view.slot(), Rights::empty(), 0, report_slot, None)
            .ok()?;
        let committed = TrapTransport::new().commit_now(&b).ok()?;
        if committed.status() != kore_abi::session::Status::Ok {
            return None;
        }
        // Decode the bind-inert report off the R/O slots alias.  SAFETY: the
        // founder mapped the slots alias R/O over this task's own records; the
        // kernel wrote the report before the trap returned.  Read volatile.
        let ptr =
            crate::entry::geometry::slot_record_va(report_slot)? as *const kore_abi::cap::CapRecord;
        let rec = unsafe { core::ptr::read_volatile(ptr) };
        let g = GeomReport::decode(&rec)?;
        if g.len == 0 {
            return None;
        }
        Some(NonNull::slice_from_raw_parts(
            NonNull::new(g.base as *mut u8)?,
            g.len as usize,
        ))
    }

    /// Map a runtime-owned detached view while retaining the affine arena
    /// reservation in its [`StashedView`](crate::cap::StashedView) owner.
    /// This is the long-lived service-ring form of [`map_view`].
    #[cfg(target_os = "none")]
    pub fn map_stashed_view(view: &crate::cap::StashedView) -> Option<NonNull<[u8]>> {
        map_stashed_view_in(view, kore_abi::session::ViewMapWindow::Map)
    }

    /// Map a detached view into a separate receiver-owned window while its
    /// affine reservation stays in `view`. Large surfaces belong in
    /// `Surfaces`, not in the per-thread cursor also used by heap growth.
    /// Missing stamps, exhausted windows and invalid views all fail closed;
    /// this grants no access to a compositor pool or framebuffer aperture.
    /// The `Surfaces` compatibility spelling selects the first lane. Clients
    /// retaining current + previous backings should use
    /// [`map_stashed_view_surface_lane`] with their own affine lane ledger.
    #[cfg(target_os = "none")]
    pub fn map_stashed_view_in(
        view: &crate::cap::StashedView,
        window: kore_abi::session::ViewMapWindow,
    ) -> Option<NonNull<[u8]>> {
        match window {
            kore_abi::session::ViewMapWindow::Map => map_stashed_view_request(view, None),
            kore_abi::session::ViewMapWindow::Surfaces => {
                map_stashed_view_request(view, Some(kore_abi::session::SurfaceMapLane::First))
            }
        }
    }

    /// Map a detached surface view into one fixed reusable lane. Before reuse,
    /// the caller must first obtain protocol proof that the lessor root was
    /// destroyed (which removes the PTE), then retire this exact view name to a
    /// closed outcome. Plain view retirement alone is not an unmap operation.
    #[cfg(target_os = "none")]
    pub fn map_stashed_view_surface_lane(
        view: &crate::cap::StashedView,
        lane: kore_abi::session::SurfaceMapLane,
    ) -> Option<NonNull<[u8]>> {
        map_stashed_view_request(view, Some(lane))
    }

    #[cfg(target_os = "none")]
    fn map_stashed_view_request(
        view: &crate::cap::StashedView,
        surface_lane: Option<kore_abi::session::SurfaceMapLane>,
    ) -> Option<NonNull<[u8]>> {
        use kore_abi::session::GeomReport;

        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let report_slot = crate::entry::geometry::report_slot()?;
        let mut b = SessionBuilder::new();
        match surface_lane {
            Some(lane) => b
                .map_view_surface_lane(view.slot()?, lane, report_slot)
                .ok()?,
            None => b
                .map_view_in(
                    view.slot()?,
                    kore_abi::session::ViewMapWindow::Map,
                    report_slot,
                )
                .ok()?,
        };
        let committed = TrapTransport::new().commit_now(&b).ok()?;
        if committed.status() != kore_abi::session::Status::Ok {
            return None;
        }
        let ptr =
            crate::entry::geometry::slot_record_va(report_slot)? as *const kore_abi::cap::CapRecord;
        // SAFETY: the slots alias is read-only and the kernel published this
        // report before the successful commit returned.
        let rec = unsafe { core::ptr::read_volatile(ptr) };
        let geom = GeomReport::decode(&rec)?;
        if geom.len == 0 {
            return None;
        }
        Some(NonNull::slice_from_raw_parts(
            NonNull::new(geom.base as *mut u8)?,
            geom.len as usize,
        ))
    }

    #[cfg(not(target_os = "none"))]
    pub fn map_stashed_view(_view: &crate::cap::StashedView) -> Option<NonNull<[u8]>> {
        None
    }

    #[cfg(not(target_os = "none"))]
    pub fn map_stashed_view_in(
        _view: &crate::cap::StashedView,
        _window: kore_abi::session::ViewMapWindow,
    ) -> Option<NonNull<[u8]>> {
        None
    }

    #[cfg(not(target_os = "none"))]
    pub fn map_stashed_view_surface_lane(
        _view: &crate::cap::StashedView,
        _lane: kore_abi::session::SurfaceMapLane,
    ) -> Option<NonNull<[u8]>> {
        None
    }

    /// Off the metal userland targets — the honest typed refusal: there is no trap
    /// to issue (and no space to install into).
    #[cfg(not(target_os = "none"))]
    pub fn map_view(_view: LeasedToken) -> Option<NonNull<[u8]>> {
        None
    }
    /// **Map a held lease for reading** (`Register`'s validation window):
    /// the complete lease/sub-lease at this task's map-window cursor (the
    /// builder's `map_owned` — the map flavor, #336 rung 2), returning the
    /// reported span.  Any refusal — no report slot, a non-`Ok` commit, a
    /// non-report record — is a fail-closed `None`.
    #[cfg(target_os = "none")]
    pub fn map_held(lease: SlotRef) -> Option<NonNull<[u8]>> {
        use kore_abi::cap::CapRecord;
        use kore_abi::session::GeomReport;

        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let report_slot = crate::entry::geometry::report_slot()?;
        let mut b = SessionBuilder::new();
        b.map_owned(lease, Rights::ACCESS, report_slot).ok()?;
        let committed = TrapTransport::new().commit_now(&b).ok()?;
        if committed.status() != kore_abi::session::Status::Ok {
            return None;
        }
        // SAFETY: the founder mapped the slots alias R/O over this task's
        // own records; the kernel wrote the report before the trap
        // returned.  Read volatile.
        let ptr = crate::entry::geometry::slot_record_va(report_slot)? as *const CapRecord;
        let g = GeomReport::decode(&unsafe { core::ptr::read_volatile(ptr) })?;
        if g.len == 0 {
            return None;
        }
        Some(NonNull::slice_from_raw_parts(
            NonNull::new(g.base as *mut u8)?,
            g.len as usize,
        ))
    }

    /// Off the metal userland targets — the honest typed refusal (there is no
    /// trap to issue and no space to install into).
    #[cfg(not(target_os = "none"))]
    pub fn map_held(_lease: SlotRef) -> Option<NonNull<[u8]>> {
        None
    }

    /// [`map_held`]'s **writable** sibling: map the complete held lease
    /// `R|W` at this task's map-window cursor (the serving-side shape —
    /// the image service formats and serves its minted loader-face rings
    /// through one whole-supply mapping held for the task's life, slice
    /// C).  `MODIFY` in the derive rights is what the kernel's install
    /// turns into a writable leaf, so the subject must carry it; an
    /// executable lease refuses `W` at the record layer regardless.
    #[cfg(target_os = "none")]
    pub fn map_held_rw(lease: SlotRef) -> Option<NonNull<[u8]>> {
        use kore_abi::cap::CapRecord;
        use kore_abi::session::GeomReport;

        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let report_slot = crate::entry::geometry::report_slot()?;
        let mut b = SessionBuilder::new();
        b.map_owned(lease, Rights::ACCESS | Rights::MODIFY, report_slot)
            .ok()?;
        let committed = TrapTransport::new().commit_now(&b).ok()?;
        if committed.status() != kore_abi::session::Status::Ok {
            return None;
        }
        // SAFETY: the founder mapped the slots alias R/O over this task's
        // own records; the kernel wrote the report before the trap
        // returned.  Read volatile.
        let ptr = crate::entry::geometry::slot_record_va(report_slot)? as *const CapRecord;
        let g = GeomReport::decode(&unsafe { core::ptr::read_volatile(ptr) })?;
        if g.len == 0 {
            return None;
        }
        Some(NonNull::slice_from_raw_parts(
            NonNull::new(g.base as *mut u8)?,
            g.len as usize,
        ))
    }

    /// Off the metal userland targets — the honest typed refusal (there is no
    /// trap to issue and no space to install into).
    #[cfg(not(target_os = "none"))]
    pub fn map_held_rw(_lease: SlotRef) -> Option<NonNull<[u8]>> {
        None
    }

    /// Query a held memory lease's exact window without mapping it. The
    /// kernel-vouched geometry is returned through the task's bind-inert
    /// report slot; no capability or map-cursor span is consumed.
    #[cfg(target_os = "none")]
    pub fn lease_geometry(lease: SlotRef) -> Option<(u64, u64)> {
        use kore_abi::cap::CapRecord;
        use kore_abi::session::GeomReport;

        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let report_slot = crate::entry::geometry::report_slot()?;
        let mut b = SessionBuilder::new();
        b.derive_raw(lease, Rights::empty(), 0, report_slot, None)
            .ok()?;
        let committed = TrapTransport::new().commit_now(&b).ok()?;
        if committed.status() != kore_abi::session::Status::Ok {
            return None;
        }
        let ptr = crate::entry::geometry::slot_record_va(report_slot)? as *const CapRecord;
        let geom = GeomReport::decode(&unsafe { core::ptr::read_volatile(ptr) })?;
        (geom.len != 0).then_some((geom.base, geom.len))
    }

    #[cfg(not(target_os = "none"))]
    pub fn lease_geometry(_lease: SlotRef) -> Option<(u64, u64)> {
        None
    }

    /// Map one complete held lease read-only at a caller-selected VA. `len`
    /// must equal the lease's kernel-reported length; subrange callers must
    /// first carve an exact lease. The affine result exact-unshares its PTE
    /// before its slot may be reused. Unlike [`map_held`], this does not touch
    /// the monotone receiver-map cursor.
    #[cfg(target_os = "none")]
    pub fn map_owned_lease_at(lease: SlotRef, va: u64, len: u64) -> Option<MappedLeaseView> {
        map_owned_whole_lease_at(lease, va, len, kore_abi::session::SHARE_ACCESS_R)
    }

    #[cfg(not(target_os = "none"))]
    pub fn map_owned_lease_at(_lease: SlotRef, _va: u64, _len: u64) -> Option<MappedLeaseView> {
        None
    }

    /// [`map_owned_lease_at`]'s **writable** whole-lease sibling. This is the
    /// shell's stamp-before-convey shape after it carves a page lease: write
    /// `.kore.ld`, exact-unshare the view, retire the page lease, then convey
    /// the childless root. Σ5 still gates `W` through `MODIFY`.
    #[cfg(target_os = "none")]
    pub fn map_owned_lease_at_rw(lease: SlotRef, va: u64, len: u64) -> Option<MappedLeaseView> {
        map_owned_whole_lease_at(
            lease,
            va,
            len,
            kore_abi::session::SHARE_ACCESS_R | kore_abi::session::SHARE_ACCESS_W,
        )
    }

    /// Map one complete finalized lease R/X at a caller-selected VA. The
    /// kernel accepts this only for an executable lease and returns an affine
    /// view whose drop exact-unshares the mapping.
    #[cfg(target_os = "none")]
    pub fn map_owned_lease_at_rx(lease: SlotRef, va: u64, len: u64) -> Option<MappedLeaseView> {
        map_owned_whole_lease_at(
            lease,
            va,
            len,
            kore_abi::session::SHARE_ACCESS_R | kore_abi::session::SHARE_ACCESS_X,
        )
    }

    #[cfg(not(target_os = "none"))]
    pub fn map_owned_lease_at_rx(_lease: SlotRef, _va: u64, _len: u64) -> Option<MappedLeaseView> {
        None
    }

    /// Map one complete held lease R/W at `va`, publishing the revocable view
    /// into a caller-owned fixed `result` slot.
    ///
    /// This is the fixed-SSA counterpart of [`map_owned_lease_at_rw`]. It does
    /// not reserve a dynamic arena name; the caller must already own `result`,
    /// must keep it live while touching the mapping, and must close it with
    /// [`retire_mapped_view`] before reusing either the VA or the slot.
    #[cfg(target_os = "none")]
    pub fn map_fixed_lease_at_rw(lease: SlotRef, va: u64, len: u64, result: SlotRef) -> bool {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let Some((_, lease_len)) = lease_geometry(lease) else {
            return false;
        };
        if len == 0 || len != lease_len {
            return false;
        }
        let mut b = SessionBuilder::new();
        if b.share_cpu_access(
            lease,
            Rights::ACCESS | Rights::MODIFY | Rights::REVOKE,
            va,
            len,
            kore_abi::session::SHARE_ACCESS_R | kore_abi::session::SHARE_ACCESS_W,
            result,
        )
        .is_err()
        {
            return false;
        }
        TrapTransport::new()
            .commit_now(&b)
            .is_ok_and(|committed| committed.status() == kore_abi::session::Status::Ok)
    }

    #[cfg(not(target_os = "none"))]
    pub fn map_fixed_lease_at_rw(_lease: SlotRef, _va: u64, _len: u64, _result: SlotRef) -> bool {
        false
    }

    /// Shared whole-lease implementation for the fixed-address view helpers.
    #[cfg(target_os = "none")]
    fn map_owned_whole_lease_at(
        lease: SlotRef,
        va: u64,
        len: u64,
        access: u8,
    ) -> Option<MappedLeaseView> {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let (_, lease_len) = lease_geometry(lease)?;
        if len == 0 || len != lease_len {
            return None;
        }
        let len_usize = usize::try_from(len).ok()?;
        let base = NonNull::new(usize::try_from(va).ok()? as *mut u8)?;
        let owned = crate::cap::OwnedSlot::reserve()?;
        let mut b = SessionBuilder::new();
        if access
            & !(kore_abi::session::SHARE_ACCESS_R
                | kore_abi::session::SHARE_ACCESS_W
                | kore_abi::session::SHARE_ACCESS_X)
            != 0
            || access & kore_abi::session::SHARE_ACCESS_R == 0
            || access & kore_abi::session::SHARE_ACCESS_W != 0
                && access & kore_abi::session::SHARE_ACCESS_X != 0
        {
            return None;
        }
        let mut rights = Rights::ACCESS | Rights::REVOKE;
        if access & kore_abi::session::SHARE_ACCESS_W != 0 {
            rights |= Rights::MODIFY;
        }
        b.share_cpu_access(lease, rights, va, len, access, owned.slot())
            .ok()?;
        let committed = TrapTransport::new().commit_now(&b).ok()?;
        if committed.status() != kore_abi::session::Status::Ok {
            return None;
        }
        Some(MappedLeaseView::new(owned, lease, base, len_usize))
    }

    #[cfg(not(target_os = "none"))]
    pub fn map_owned_lease_at_rw(_lease: SlotRef, _va: u64, _len: u64) -> Option<MappedLeaseView> {
        None
    }

    /// **Carve one segment lease** off a held supply (`Register`'s
    /// plain-segment step): one session — a sub-lease window
    /// `[offset, offset+len)` of `subject`, landing at `into`.  The slot
    /// IS the stashed lease (`Acquire` grants name it; `Retire` revokes
    /// it); full rights so downstream shares + revokes are unblocked
    /// (Σ5 re-gates every child regardless).
    #[cfg(target_os = "none")]
    pub fn carve_lease_window(subject: SlotRef, offset: u64, len: u64, into: SlotRef) -> bool {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let mut b = SessionBuilder::new();
        if b.carve_sublease(
            subject,
            Rights::ACCESS | Rights::MODIFY | Rights::DERIVE | Rights::REVOKE,
            offset,
            len,
            into,
        )
        .is_err()
        {
            return false;
        }
        let Ok(committed) = TrapTransport::new().commit_now(&b) else {
            return false;
        };
        committed.status() == kore_abi::session::Status::Ok
    }

    /// Off the metal userland targets — the honest typed refusal.
    #[cfg(not(target_os = "none"))]
    pub fn carve_lease_window(_subject: SlotRef, _offset: u64, _len: u64, _into: SlotRef) -> bool {
        false
    }

    /// Carve one segment lease into a runtime-reserved persistent slot.
    ///
    /// This is the ownership-safe service form of [`carve_lease_window`]: the
    /// caller never chooses an arena index and receives an affine value that
    /// keeps the allocator reservation paired with the live capability.
    #[cfg(target_os = "none")]
    pub fn carve_owned_lease_window(
        subject: SlotRef,
        offset: u64,
        len: u64,
    ) -> Option<StashedLease> {
        let owned = crate::cap::OwnedSlot::reserve()?;
        if !carve_lease_window(subject, offset, len, owned.slot()) {
            return None;
        }
        Some(StashedLease::new(owned))
    }

    /// Off metal there is no session engine and therefore no owned lease.
    #[cfg(not(target_os = "none"))]
    pub fn carve_owned_lease_window(
        _subject: SlotRef,
        _offset: u64,
        _len: u64,
    ) -> Option<StashedLease> {
        None
    }

    /// Snapshot the runtime allocator's remaining persistent slot capacity.
    /// The image archive uses this only as an all-members preflight; each
    /// subsequent owned carve still performs the authoritative atomic claim.
    #[cfg(target_os = "none")]
    pub fn available_owned_slots() -> usize {
        crate::entry::geometry::available_cap_slots()
    }

    /// No owned capability slots can be minted off the metal target.
    #[cfg(not(target_os = "none"))]
    pub fn available_owned_slots() -> usize {
        0
    }

    /// **Narrow an already-mapped range's protection** to `access`
    /// ([`SHARE_ACCESS_R`](kore_abi::session::SHARE_ACCESS_R)`/W/X`) —
    /// the RELRO seal and the passive-page seal (`res/IMAGE.md` §4 phase
    /// 2), and `kore_ld`'s `LinkWorld::seal_ro` on metal.
    ///
    /// `space` is this task's own address space (its
    /// `role::ADDRESS_SPACE` self-view); the kernel verifies that by
    /// page-table root and demands `MODIFY`.  `scratch` is an arena slot
    /// the session names as its result — the verb mints nothing, so the
    /// slot is left vacant, but a session still has to name one.
    ///
    /// **Clear-only.**  Naming an access bit the range does not already
    /// carry is refused, not clamped, and nothing is written when it is:
    /// the kernel proves the whole range before touching a leaf, so a
    /// refusal never leaves a sealed prefix behind.  That monotonicity is
    /// what makes this safe to hand a linker — it can drop `W` from its
    /// own data and cannot add `W` back to its own text.
    ///
    /// `va`/`len` are page-granular; the range must be exactly tiled by
    /// installed leaves.
    #[cfg(target_os = "none")]
    pub fn seal_range(space: SlotRef, va: u64, len: u64, access: u8, scratch: SlotRef) -> bool {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        // A linker states a segment's access in `image::seg::ACCESS_*`
        // and this verb takes `session::SHARE_ACCESS_*`.  They are two
        // vocabularies that happen to agree bit for bit — so a caller
        // passing one where the other belongs is *correct today and
        // silently wrong the day either moves*.  Nothing else in the tree
        // ties them, and a divergence would mis-seal rather than refuse:
        // an `X` read as `W` is precisely the escalation the verb's
        // monotonicity exists to prevent.  Assert the agreement instead
        // of trusting it.
        const _: () = {
            use kore_abi::image::seg;
            use kore_abi::session::{SHARE_ACCESS_R, SHARE_ACCESS_W, SHARE_ACCESS_X};
            assert!(seg::ACCESS_R == SHARE_ACCESS_R);
            assert!(seg::ACCESS_W == SHARE_ACCESS_W);
            assert!(seg::ACCESS_X == SHARE_ACCESS_X);
        };

        // `ProtectReq`: va u64 ‖ len u64 ‖ access u8 ‖ reserved [u8; 7],
        // LE and unaligned-safe — built by hand so the path needs no
        // allocation, since it runs in the stub before any heap exists.
        let mut arg = [0u8; kore_abi::session::ProtectReq::WIRE];
        arg[0..8].copy_from_slice(&va.to_le_bytes());
        arg[8..16].copy_from_slice(&len.to_le_bytes());
        arg[16] = access;

        let mut b = SessionBuilder::new();
        if b.derive_raw(
            space,
            // Mints no child, so it requests no rights on one — the
            // posture of `SET_TP`, the other self-effect on this subject.
            Rights::empty(),
            kore_abi::session::DERIVE_KIND_PROTECT,
            scratch,
            Some(&arg),
        )
        .is_err()
        {
            return false;
        }
        let Ok(committed) = TrapTransport::new().commit_now(&b) else {
            return false;
        };
        committed.status() == kore_abi::session::Status::Ok
    }

    /// **Install a held lease into this task's OWN space at an exact VA**,
    /// with explicit access bits — `map_image`'s receiver-maps verb, and
    /// the counterpart to [`map_held`] (which lets the kernel choose the
    /// address). A linker needs the exact one: a module's segments must land
    /// at the base its relocations were computed against.
    ///
    /// This deliberately uses the ordinary held-subject `share_cpu_access`
    /// path. `share_to` is the bootstrap exception for a parent mutating a
    /// not-yet-founded child's space; requiring that explicit-destination
    /// path after first instruction would preserve the founder's injection
    /// mechanism instead of completing the receiver-maps flip.
    ///
    /// `access` is `SHARE_ACCESS_R/W/X`; the kernel's W^X backstop refuses
    /// `W|X` in one request, and `X` additionally requires an executable
    /// lease.
    #[cfg(target_os = "none")]
    pub fn share_into_self(lease: SlotRef, va: u64, len: u64, access: u8, result: SlotRef) -> bool {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        // Σ5 carries the requested grant into the minted view: a writable
        // mapping must request MODIFY from its lease, while immutable image
        // mappings deliberately request ACCESS only. The kernel still owns
        // the W^X and executable-lease checks; this is the least authority that
        // can express the requested leaf.
        let mut rights = Rights::ACCESS;
        if access & kore_abi::session::SHARE_ACCESS_W != 0 {
            rights |= Rights::MODIFY;
        }
        let mut b = SessionBuilder::new();
        if b.share_cpu_access(lease, rights, va, len, access, result)
            .is_err()
        {
            return false;
        }
        let Ok(committed) = TrapTransport::new().commit_now(&b) else {
            return false;
        };
        committed.status() == kore_abi::session::Status::Ok
    }

    /// Off the metal userland targets — the honest typed refusal.
    #[cfg(not(target_os = "none"))]
    pub fn share_into_self(
        _lease: SlotRef,
        _va: u64,
        _len: u64,
        _access: u8,
        _result: SlotRef,
    ) -> bool {
        false
    }

    /// **Install this task's thread pointer** (`DERIVE_KIND_SET_TP`) —
    /// the TLS half of a link, once the module's TLS block is built.
    ///
    /// the startup stub issues the same verb through
    /// `entry::anchor_boot_thread`, which
    /// also stamps the thread block it points at; this is the bare
    /// effect, for a caller that has already built what `tp` must
    /// dereference to.
    #[cfg(target_os = "none")]
    pub fn set_thread_pointer(space: SlotRef, tp: u64, scratch: SlotRef) -> bool {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let mut b = SessionBuilder::new();
        if b.derive_raw(
            space,
            Rights::empty(),
            kore_abi::session::DERIVE_KIND_SET_TP,
            scratch,
            Some(&tp.to_le_bytes()),
        )
        .is_err()
        {
            return false;
        }
        let Ok(committed) = TrapTransport::new().commit_now(&b) else {
            return false;
        };
        committed.status() == kore_abi::session::Status::Ok
    }

    /// Off the metal userland targets — the honest typed refusal.
    #[cfg(not(target_os = "none"))]
    pub fn set_thread_pointer(_space: SlotRef, _tp: u64, _scratch: SlotRef) -> bool {
        false
    }

    /// Off the metal userland targets — the honest typed refusal.
    #[cfg(not(target_os = "none"))]
    pub fn seal_range(
        _space: SlotRef,
        _va: u64,
        _len: u64,
        _access: u8,
        _scratch: SlotRef,
    ) -> bool {
        false
    }

    /// **Carve + finalize a text segment in ONE session** (`Register`'s text
    /// step, res/IMAGE.md §6): a writable sub-lease lands temporarily at
    /// `into_sub`, then is consumed into the executable-only lease at
    /// `into_exec`. I3: a pre-commit refusal unwinds the whole session; on
    /// success `into_sub` is vacant and no writable lineage survives.
    #[cfg(target_os = "none")]
    pub fn carve_exec_window(
        subject: SlotRef,
        offset: u64,
        len: u64,
        into_sub: SlotRef,
        into_exec: SlotRef,
    ) -> bool {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let mut b = SessionBuilder::new();
        if b.carve_sublease(
            subject,
            Rights::ACCESS | Rights::MODIFY | Rights::DERIVE | Rights::REVOKE,
            offset,
            len,
            into_sub,
        )
        .is_err()
            || b.finalize_exec(into_sub, into_exec).is_err()
        {
            return false;
        }
        let Ok(committed) = TrapTransport::new().commit_now(&b) else {
            return false;
        };
        committed.status() == kore_abi::session::Status::Ok
            && exec_result_state(into_exec) == ExecResultState::Replaced
    }

    /// Off the metal userland targets — the honest typed refusal.
    #[cfg(not(target_os = "none"))]
    pub fn carve_exec_window(
        _subject: SlotRef,
        _offset: u64,
        _len: u64,
        _into_sub: SlotRef,
        _into_exec: SlotRef,
    ) -> bool {
        false
    }

    /// Carve a writable sub-lease and finalize it into one executable-only
    /// runtime-owned lease in the same atomic session.  The temporary source
    /// slot is cleared after its tag retires and is immediately reusable; only
    /// the returned executable capability remains live.
    #[cfg(target_os = "none")]
    pub fn carve_owned_exec_window(
        subject: SlotRef,
        offset: u64,
        len: u64,
    ) -> Option<ExecutableCode> {
        let writable = crate::cap::OwnedSlot::reserve()?;
        let mut executable = crate::cap::OwnedSlot::reserve()?;
        if !carve_exec_window(subject, offset, len, writable.slot(), executable.slot()) {
            if exec_result_state(executable.slot()) == ExecResultState::Malformed {
                executable.quarantine_unknown_publish();
            }
            return None;
        }
        // `writable` never activates: the committed transition cleared its
        // now-dead record, so Reserved-drop releases exactly that scratch name.
        drop(writable);
        Some(ExecutableCode::new(executable))
    }

    /// Off metal there is no executable-finalization session to issue.
    #[cfg(not(target_os = "none"))]
    pub fn carve_owned_exec_window(
        _subject: SlotRef,
        _offset: u64,
        _len: u64,
    ) -> Option<ExecutableCode> {
        None
    }

    /// Consume a fixed-slot writable lease token and return a dynamically
    /// owned executable-only lease.  On refusal the exact input token is
    /// returned and no source authority changed.
    #[cfg(target_os = "none")]
    pub fn finalize_exec(
        lease: LeaseToken,
    ) -> Result<ExecutableCode, FinalizeExecError<LeaseToken>> {
        let mut output = match crate::cap::OwnedSlot::reserve() {
            Some(output) => output,
            None => return Err(FinalizeExecError::Refused(lease)),
        };
        match finalize_exec_slot(lease.slot(), &mut output) {
            ExecFinalizeCompletion::Replaced => Ok(ExecutableCode::new(output)),
            ExecFinalizeCompletion::Refused => Err(FinalizeExecError::Refused(lease)),
            ExecFinalizeCompletion::Consumed => Err(FinalizeExecError::Consumed),
        }
    }

    #[cfg(not(target_os = "none"))]
    pub fn finalize_exec(
        lease: LeaseToken,
    ) -> Result<ExecutableCode, FinalizeExecError<LeaseToken>> {
        Err(FinalizeExecError::Refused(lease))
    }

    /// Consume a runtime-owned writable lease and return its executable-only
    /// successor while preserving affine slot ownership on either outcome.
    #[cfg(target_os = "none")]
    pub fn finalize_stashed_exec(
        mut lease: StashedLease,
    ) -> Result<ExecutableCode, FinalizeExecError<StashedLease>> {
        let mut output = match crate::cap::OwnedSlot::reserve() {
            Some(output) => output,
            None => return Err(FinalizeExecError::Refused(lease)),
        };
        let Some(source) = lease.slot() else {
            return Err(FinalizeExecError::Refused(lease));
        };
        match finalize_exec_slot(source, &mut output) {
            ExecFinalizeCompletion::Refused => Err(FinalizeExecError::Refused(lease)),
            ExecFinalizeCompletion::Consumed => {
                // The source name is gone even though the successor was not
                // published; release only its local allocator reservation.
                lease.owned.release_after_proven_move();
                Err(FinalizeExecError::Consumed)
            }
            ExecFinalizeCompletion::Replaced => {
                // The committed kernel transition cleared `source`; only its
                // dynamic allocator reservation remains to release.
                lease.owned.release_after_proven_move();
                Ok(ExecutableCode::new(output))
            }
        }
    }

    #[cfg(not(target_os = "none"))]
    pub fn finalize_stashed_exec(
        lease: StashedLease,
    ) -> Result<ExecutableCode, FinalizeExecError<StashedLease>> {
        Err(FinalizeExecError::Refused(lease))
    }

    /// Return an executable window to writable, preserving affine slot
    /// ownership on either outcome — [`finalize_stashed_exec`]'s mirror.
    ///
    /// Takes the [`ExecutableCode`] **by value**, which is the local half of
    /// the safety proof: `ExecutableMapping` borrows the authority, so no
    /// R/X mapping of this task's can outlive the move.
    #[cfg(target_os = "none")]
    pub fn return_writable(
        mut code: ExecutableCode,
    ) -> Result<StashedLease, FinalizeExecError<ExecutableCode>> {
        let mut output = match crate::cap::OwnedSlot::reserve() {
            Some(output) => output,
            None => return Err(FinalizeExecError::Refused(code)),
        };
        let Some(source) = code.slot() else {
            return Err(FinalizeExecError::Refused(code));
        };
        match return_writable_slot(source, &mut output) {
            ExecFinalizeCompletion::Refused => Err(FinalizeExecError::Refused(code)),
            ExecFinalizeCompletion::Consumed => {
                code.release_after_proven_move();
                Err(FinalizeExecError::Consumed)
            }
            ExecFinalizeCompletion::Replaced => {
                code.release_after_proven_move();
                Ok(StashedLease::from_owned(output))
            }
        }
    }

    #[cfg(not(target_os = "none"))]
    pub fn return_writable(
        code: ExecutableCode,
    ) -> Result<StashedLease, FinalizeExecError<ExecutableCode>> {
        Err(FinalizeExecError::Refused(code))
    }

    #[cfg(target_os = "none")]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum ExecFinalizeCompletion {
        Refused,
        Replaced,
        Consumed,
    }

    #[cfg(target_os = "none")]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum ExecResultState {
        Replaced,
        Vacant,
        Malformed,
    }

    #[cfg(target_os = "none")]
    fn exec_result_state(output: SlotRef) -> ExecResultState {
        use kore_abi::cap::{CapRecord, ResourceKind};

        let Some(ptr) = crate::entry::geometry::slot_record_va(output) else {
            return ExecResultState::Malformed;
        };
        // SAFETY: the kernel is the sole writer of this thread's read-only
        // arena alias, and the committing trap completed before this read.
        let record = unsafe { core::ptr::read_volatile(ptr as *const CapRecord) };
        let rights = Rights::ACCESS | Rights::DERIVE | Rights::REVOKE;
        if record.cid.is_null() {
            ExecResultState::Vacant
        } else if record.kind == ResourceKind::MemLease as u16 && record.rights == rights {
            ExecResultState::Replaced
        } else {
            ExecResultState::Malformed
        }
    }

    /// The writable successor's shape proof — [`exec_result_state`]'s twin.
    /// The rights differ (`MODIFY` is back), and checking the exact set is
    /// what makes a wrong-shaped record `Malformed` rather than silently
    /// accepted.
    #[cfg(target_os = "none")]
    fn writable_result_state(output: SlotRef) -> ExecResultState {
        use kore_abi::cap::{CapRecord, ResourceKind};

        let Some(ptr) = crate::entry::geometry::slot_record_va(output) else {
            return ExecResultState::Malformed;
        };
        // SAFETY: as `exec_result_state` — the kernel is the sole writer of
        // this thread's read-only arena alias, and the committing trap
        // completed before this read.
        let record = unsafe { core::ptr::read_volatile(ptr as *const CapRecord) };
        let rights = Rights::ACCESS | Rights::MODIFY | Rights::DERIVE | Rights::REVOKE;
        if record.cid.is_null() {
            ExecResultState::Vacant
        } else if record.kind == ResourceKind::MemLease as u16 && record.rights == rights {
            ExecResultState::Replaced
        } else {
            ExecResultState::Malformed
        }
    }

    /// Return an executable window to writable, mirroring
    /// [`finalize_exec_slot`]. `Consumed` means the source is gone and no
    /// authority came back — the caller has lost the arena and must not
    /// retry.
    #[cfg(target_os = "none")]
    fn return_writable_slot(
        source: SlotRef,
        output: &mut crate::cap::OwnedSlot,
    ) -> ExecFinalizeCompletion {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let mut b = SessionBuilder::new();
        if b.return_writable(source, output.slot()).is_err() {
            return ExecFinalizeCompletion::Refused;
        }
        let Ok(committed) = TrapTransport::new().commit_now(&b) else {
            return ExecFinalizeCompletion::Refused;
        };
        if committed.status() != kore_abi::session::Status::Ok || committed.failing_index() != 0 {
            return ExecFinalizeCompletion::Refused;
        }
        match writable_result_state(output.slot()) {
            ExecResultState::Replaced => ExecFinalizeCompletion::Replaced,
            ExecResultState::Vacant => ExecFinalizeCompletion::Consumed,
            ExecResultState::Malformed => {
                output.quarantine_unknown_publish();
                ExecFinalizeCompletion::Consumed
            }
        }
    }

    #[cfg(target_os = "none")]
    fn finalize_exec_slot(
        source: SlotRef,
        output: &mut crate::cap::OwnedSlot,
    ) -> ExecFinalizeCompletion {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let mut b = SessionBuilder::new();
        if b.finalize_exec(source, output.slot()).is_err() {
            return ExecFinalizeCompletion::Refused;
        }
        let Ok(committed) = TrapTransport::new().commit_now(&b) else {
            return ExecFinalizeCompletion::Refused;
        };
        if committed.status() != kore_abi::session::Status::Ok || committed.failing_index() != 0 {
            return ExecFinalizeCompletion::Refused;
        }
        match exec_result_state(output.slot()) {
            ExecResultState::Replaced => ExecFinalizeCompletion::Replaced,
            ExecResultState::Vacant => ExecFinalizeCompletion::Consumed,
            ExecResultState::Malformed => {
                // A malformed result is quarantined so its allocator name
                // cannot alias authority whose shape userland cannot prove.
                output.quarantine_unknown_publish();
                ExecFinalizeCompletion::Consumed
            }
        }
    }

    /// **Revoke one held slot**: one revoke session. For a shared CPU view,
    /// this unmaps its leaf; for a lease, T6 reaps the whole view subtree.
    #[cfg(target_os = "none")]
    pub fn revoke_slot(slot: SlotRef) -> bool {
        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let mut b = SessionBuilder::new();
        if b.revoke(slot).is_err() {
            return false;
        }
        let Ok(committed) = TrapTransport::new().commit_now(&b) else {
            return false;
        };
        committed.status() == kore_abi::session::Status::Ok
    }

    /// Off the metal userland targets — the honest typed refusal.
    #[cfg(not(target_os = "none"))]
    pub fn revoke_slot(_slot: SlotRef) -> bool {
        false
    }

    /// **Retire a stashed lease** (`Retire`'s per-lease step): the named
    /// spelling used by the image service. See [`revoke_slot`].
    #[inline]
    pub fn retire_lease(slot: SlotRef) -> bool {
        revoke_slot(slot)
    }
    /// **The metal recv-pin** (`res/USERLAND.md` §3.4.3; Phase-A checkpoint B) —
    /// the first real possession verb: read the doorbell `ChanRecv` record the
    /// founder aliased **read-only** at [`SLOTS`](kore_abi::task::geom::SLOTS) alias
    /// and return its `(chan, recv_tag)` pin — the two plaintext lineage fields a
    /// park-on-standing-post `ParkReq` names.  It takes **no** `bind`:
    /// userland holds no epoch key, so the record is read as *data*, and
    /// the kernel re-authenticates it
    /// (holder ‖ tag liveness ‖ CMAC) at the park trap
    /// (`src/runtime/session/mod.rs`, `park_on_standing`).  `None` for anything
    /// that is not a `ChanRecv`.
    ///
    /// The `(chan, recv_tag)` extraction is the kernel park path's, field-for-
    /// field (`src/runtime/session/prepare.rs`: the `OpCode::Rendezvous` post
    /// arm derives `chan = b.cid.resource()`, `recv_tag = b.cid.tag().index()`
    /// off the bound record's `cid`).  So this returns exactly `cid.resource()`
    /// and `cid.tag()` off the same `CapRecord.cid`, and the kernel's park
    /// re-bind matches by construction; a wrong field would bind the wrong
    /// channel/tag and the park would refuse instead of standing.
    #[cfg(target_os = "none")]
    pub fn recv_pin(recv_slot: SlotRef) -> Option<(u32, u32)> {
        use kore_abi::cap::{CapRecord, ResourceKind};

        // SAFETY: the founder mapped `[SLOTS_VA, +ARENA_SLOTS·128)` as a
        // read-only (`READ|USER`, no `WRITE`) alias of THIS task's own
        // `slots_phys`; `recv_slot` names a slot the founding stamped
        // (`role::DOORBELL[0]`), in range and 128-byte aligned within the span.
        // The kernel is the sole writer of that frame, so read it volatile.
        let ptr = crate::entry::geometry::slot_record_va(recv_slot)? as *const CapRecord;
        let rec = unsafe { core::ptr::read_volatile(ptr) };
        if rec.kind != ResourceKind::ChanRecv as u16 {
            return None;
        }
        // chan = the channel id (the `cid`'s resource half); recv_tag = the
        // record's own revocation tag (the `cid`'s tag half) — the exact pair the
        // kernel's rendezvous-post binds (`b.cid.resource()`/`b.cid.tag()`).
        Some((rec.cid.resource(), rec.cid.tag().0))
    }

    /// Off the metal userland targets — host builds — the honest typed
    /// refusal (`res/USERLAND.md` §2.1): there is no slot alias to read.
    #[cfg(not(target_os = "none"))]
    pub fn recv_pin(_recv_slot: SlotRef) -> Option<(u32, u32)> {
        None
    }
    /// **The metal take-notice** (`res/USERLAND.md` §3.1; Phase-A
    /// checkpoint C) — decode a delivered doorbell `(cookie, payload)` off
    /// the R/O cap-slot alias at [`SLOTS`](kore_abi::task::geom::SLOTS) alias.
    /// The discriminator is the ABI's own [`Notice::decode`] (kernel
    /// encodes, userland decodes — one impl, no drift): a notice is
    /// **bind-inert** data (`cid == `[`Cid::NULL`](kore_abi::cap::Cid::NULL),
    /// `kind == NOTICE_KIND`, magic in `parent`), reading `scope_rid` as the
    /// cookie and `holder` as the payload.  It does **not** clear the slot —
    /// the R/O alias forbids a userland write; the clear folds into the
    /// executor's next `post_recv` repost, which scrubs the landing slot
    /// kernel-side (`session::dispatch::post`), so a re-read after consume
    /// cannot double-report.
    #[cfg(target_os = "none")]
    pub fn take_notice(slot: SlotRef) -> Option<(u64, u64)> {
        use kore_abi::cap::CapRecord;
        use kore_abi::session::Notice;

        // SAFETY: the R/O `SLOTS_VA` alias over this task's own `slots_phys`
        // (see `take_view`); `slot` in range, 128-byte aligned, kernel the
        // sole writer — read the frame volatile.
        let ptr = crate::entry::geometry::slot_record_va(slot)? as *const CapRecord;
        let rec = unsafe { core::ptr::read_volatile(ptr) };
        Notice::decode(&rec).map(|n| (n.cookie, n.payload))
    }

    /// Off the metal userland targets — the honest typed refusal.
    #[cfg(not(target_os = "none"))]
    pub fn take_notice(_slot: SlotRef) -> Option<(u64, u64)> {
        None
    }

    /// **The metal take-delivery** (`res/USERLAND.md` §5.2; Phase-A
    /// checkpoint C) — read a recv landing slot off the R/O cap-slot alias
    /// at [`SLOTS`](kore_abi::task::geom::SLOTS) alias and classify it, the
    /// probe `ChanRx::recv` and the executor consume drive.  The metal
    /// discriminator (the `CapRecord` has no `RecKind`):
    ///
    /// - a **notice** — [`Notice::decode`] succeeds (`cid == NULL`, the
    ///   notice magic + kind) — returns [`RawDelivery::Notice`]`(cookie,
    ///   payload)`, consumed via the folded repost-clear (as [`take_notice`]);
    /// - a **vacant/reserved** slot — `cid == `[`Cid::NULL`](kore_abi::cap::Cid::NULL)
    ///   but *not* a deliberate notice — returns `None`;
    /// - a **delivered capability** — `cid != NULL` (sealed) — returns
    ///   [`RawDelivery::Cap`], **left in place** for the typed take
    ///   (`take_view`/`take_lease`/`take_handle`), which the kernel
    ///   re-binds by slot at its use-trap.
    ///
    /// The discrimination is keyed on `cid` alone: `Notice → Notice`,
    /// `Reserved → None`, `else → Cap`.
    #[cfg(target_os = "none")]
    pub fn take_delivery(slot: SlotRef) -> Option<crate::cap::RawDelivery> {
        use kore_abi::cap::CapRecord;
        use kore_abi::session::Notice;

        use crate::cap::RawDelivery;

        // SAFETY: the R/O `SLOTS_VA` alias over this task's own `slots_phys`
        // (see `take_view`); `slot` in range, 128-byte aligned, kernel the
        // sole writer — read the frame volatile.
        let ptr = crate::entry::geometry::slot_record_va(slot)? as *const CapRecord;
        let rec = unsafe { core::ptr::read_volatile(ptr) };
        if let Some(n) = Notice::decode(&rec) {
            return Some(RawDelivery::Notice(n.cookie, n.payload));
        }
        // Vacant / scrubbed / a scratch reservation — bind-inert, no cap.
        if rec.cid.is_null() {
            return None;
        }
        // A sealed capability record — reported, left in place for the take.
        Some(RawDelivery::Cap)
    }

    /// Off the metal userland targets — the honest typed refusal.
    #[cfg(not(target_os = "none"))]
    pub fn take_delivery(_slot: SlotRef) -> Option<crate::cap::RawDelivery> {
        None
    }

    /// Non-owning ingress check that a delivered slot holds a **channel
    /// send end** carrying at least `required` — the
    /// [`ReturnedCapability::is_lease_with`](crate::cap::ReturnedCapability::is_lease_with)
    /// discipline for a walking-return landing, whose slot the service
    /// owns raw (no receiver-owned wrapper exists on that path).  Reads
    /// the authenticated delivery's public record as metadata; the later
    /// kick/grant verbs still perform the authoritative bind.
    #[cfg(target_os = "none")]
    pub fn slot_is_chan_send(slot: SlotRef, required: Rights) -> bool {
        use kore_abi::cap::{CapRecord, ResourceKind};

        let Some(ptr) = crate::entry::geometry::slot_record_va(slot) else {
            return false;
        };
        // SAFETY: the R/O `SLOTS_VA` alias over this task's own slot records
        // (the `take_delivery` argument above); `slot` was range-checked by
        // `slot_record_va`, and the kernel is the sole writer.
        let record = unsafe { core::ptr::read_volatile(ptr as *const CapRecord) };
        record.kind == ResourceKind::ChanSend as u16 && record.rights.contains(required)
    }

    /// Off the metal userland targets — no live record to inspect.
    #[cfg(not(target_os = "none"))]
    pub fn slot_is_chan_send(_slot: SlotRef, _required: kore_abi::cap::Rights) -> bool {
        false
    }

    /// **The metal take-handle** (Phase S; Phase-A checkpoint C) — read a
    /// delivered task handle off the R/O cap-slot alias at
    /// [`SLOTS`](kore_abi::task::geom::SLOTS) alias and return a [`TaskToken`]
    /// naming the slot (the exec ring client's possession of the spawn
    /// reply's transferred handle).  The tag stays **live**: this is exec
    /// keeping its own authority, re-bound by slot in later sessions. Reads
    /// `rec.kind` (the
    /// `ResourceKind::Task` discriminant), `rec.cid.resource()` (the task
    /// id), and `rec.rights`.
    #[cfg(target_os = "none")]
    pub fn take_handle(slot: SlotRef) -> Option<TaskToken> {
        let (id, rights) = task_record(slot)?;
        // The tag stays live: this possession is a shadow of the arena record,
        // and every later effect re-binds the kernel twin by exact cid.
        Some(TaskToken::possess(slot, id, rights))
    }

    /// Off the metal userland targets — the honest typed refusal.
    #[cfg(not(target_os = "none"))]
    pub fn take_handle(_slot: SlotRef) -> Option<TaskToken> {
        None
    }

    /// **The metal claim-display** (the display handoff's endowment claim,
    /// A.2) — read the display-owned `MmioRegion` record off the R/O
    /// slots alias and return a [`DisplayToken`] naming the slot.  The
    /// record is read as *data* (authority is re-authenticated by slot at
    /// the map trap, the [`take_view`] discipline); the map + mode reports
    /// carry the geometry.  `None` for anything that is not a display
    /// region record.
    #[cfg(target_os = "none")]
    pub fn claim_display(slot: SlotRef) -> Option<DisplayToken> {
        // One gate, not a re-spelling of it: `authority_record` carries the
        // kind check AND the cid-liveness half this site's hand-rolled copy
        // had silently dropped (the audit's gate-drift finding).  Rights are
        // deliberately not mirrored — the MODIFY gate is kernel-side.
        let _rights = authority_record(slot, kore_abi::cap::ResourceKind::MmioRegion)?;
        Some(DisplayToken::possess(slot))
    }

    /// Off the metal userland targets — the honest typed refusal.
    #[cfg(not(target_os = "none"))]
    pub fn claim_display(_slot: SlotRef) -> Option<DisplayToken> {
        None
    }

    /// **Claim the `role::POWER` endowment** (#392): validate that the boot
    /// block's slot really holds a `SystemPower` record before taking
    /// possession.  `None` for anything else — a fail-closed claim, the
    /// [`claim_display`] discipline.
    #[cfg(target_os = "none")]
    pub fn claim_power(slot: SlotRef) -> Option<PowerToken> {
        // `authority_record`, not a hand-rolled copy of it (the audit's
        // gate-drift finding: the copies had dropped the cid-liveness
        // half).  Rights stay unmirrored — the MODIFY gate is kernel-side.
        let _rights = authority_record(slot, kore_abi::cap::ResourceKind::SystemPower)?;
        Some(PowerToken::possess(slot))
    }

    /// Off the metal userland targets — the honest typed refusal.
    #[cfg(not(target_os = "none"))]
    pub fn claim_power(_slot: SlotRef) -> Option<PowerToken> {
        None
    }

    /// Claim the monitor service's `SYSTEM_OBSERVE` endowment.
    #[cfg(target_os = "none")]
    pub fn claim_observe(slot: SlotRef) -> Option<ObserveToken> {
        let _rights = authority_record(slot, kore_abi::cap::ResourceKind::SystemObserve)?;
        Some(ObserveToken::possess(slot))
    }

    #[cfg(not(target_os = "none"))]
    pub fn claim_observe(_slot: SlotRef) -> Option<ObserveToken> {
        None
    }

    /// Sample one closed system metric through the read-only observation cap.
    #[cfg(target_os = "none")]
    pub fn observe_metric(
        token: &ObserveToken,
        metric: kore_abi::session::SystemMetric,
    ) -> Option<kore_abi::session::SystemObserveReport> {
        observe_metric_at(token, metric, 0)
    }

    /// Sample an indexed device-census or processor metric at `operand`;
    /// every other metric must pass 0, which the kernel enforces.
    #[cfg(target_os = "none")]
    pub fn observe_metric_at(
        token: &ObserveToken,
        metric: kore_abi::session::SystemMetric,
        operand: u16,
    ) -> Option<kore_abi::session::SystemObserveReport> {
        use kore_abi::cap::{CapRecord, Rights};
        use kore_abi::session::{
            DERIVE_KIND_SYSTEM_OBSERVE, Status, SystemObserveReport, SystemObserveReq,
        };

        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let report_slot = crate::entry::geometry::report_slot()?;
        let arg = SystemObserveReq::encode(metric, operand);
        let mut req = SessionBuilder::new();
        req.derive_raw(
            token.slot(),
            Rights::empty(),
            DERIVE_KIND_SYSTEM_OBSERVE,
            report_slot,
            Some(&arg),
        )
        .ok()?;
        let committed = TrapTransport::new().commit_now(&req).ok()?;
        if committed.status() != Status::Ok {
            return None;
        }
        let ptr = crate::entry::geometry::slot_record_va(report_slot)? as *const CapRecord;
        let report = SystemObserveReport::decode(&unsafe { core::ptr::read_volatile(ptr) })?;
        (report.metric == metric).then_some(report)
    }

    /// Consume one kernel-vouched completion for an MM-issued supply rooted
    /// in `region`, on the `pair` arrival lane. Absence is a normal poll miss.
    #[cfg(target_os = "none")]
    pub fn take_supply_release(
        region: &RegionToken,
        pair: u16,
    ) -> Option<kore_abi::session::SupplyReleaseReport> {
        use kore_abi::cap::{CapRecord, Rights};
        use kore_abi::session::{
            DERIVE_KIND_SUPPLY_RELEASE, Status, SupplyReleaseReport, SupplyReleaseReq,
        };

        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let report_slot = crate::entry::geometry::report_slot()?;
        let mut req = [0u8; SupplyReleaseReq::WIRE];
        req[0..2].copy_from_slice(&pair.to_le_bytes());
        let mut request = SessionBuilder::new();
        request
            .derive_raw(
                region.slot(),
                Rights::empty(),
                DERIVE_KIND_SUPPLY_RELEASE,
                report_slot,
                Some(&req),
            )
            .ok()?;
        let committed = TrapTransport::new().commit_now(&request).ok()?;
        if committed.status() != Status::Ok {
            return None;
        }
        let ptr = crate::entry::geometry::slot_record_va(report_slot)? as *const CapRecord;
        SupplyReleaseReport::decode(&unsafe { core::ptr::read_volatile(ptr) })
    }

    #[cfg(not(target_os = "none"))]
    pub fn take_supply_release(
        _region: &RegionToken,
        _pair: u16,
    ) -> Option<kore_abi::session::SupplyReleaseReport> {
        None
    }

    #[cfg(not(target_os = "none"))]
    pub fn observe_metric(
        _token: &ObserveToken,
        _metric: kore_abi::session::SystemMetric,
    ) -> Option<kore_abi::session::SystemObserveReport> {
        None
    }

    #[cfg(not(target_os = "none"))]
    pub fn observe_metric_at(
        _token: &ObserveToken,
        _metric: kore_abi::session::SystemMetric,
        _operand: u16,
    ) -> Option<kore_abi::session::SystemObserveReport> {
        None
    }

    /// **Request a machine power transition** through the held grant (#392).
    ///
    /// Synchronous by necessity: [`SystemService::serve`] is sync and cannot
    /// await, so this drives `commit_now` directly — the
    /// [`map_display`] discipline, not the async client.
    ///
    /// The commit **accepts** a transition; it does not perform one. The
    /// kernel checks authority and admission inline, then latches the
    /// transition for the run-loop bottom half. `PowerOff` and `Reboot` drive
    /// graceful teardown plus a terminal firmware trigger;
    /// `SuspendToIdle` latches a reversible attempt whose preparation may
    /// still abort. Thus `true` means only *accepted*. A caller may execute
    /// more instructions before the bottom half starts; if idle is entered,
    /// wake is observed out of band when the existing machine resumes.
    ///
    /// `kind` is deliberately `0`, **not** `SystemPower`: the self-kind derive
    /// is the founding endowment, and this is the transition verb. Passing the
    /// self kind here would mint a second grant instead of asking for a
    /// transition (#399).
    ///
    /// [`SystemService::serve`]: crate::service::SystemService::serve
    #[cfg(target_os = "none")]
    pub fn request_power(
        token: &PowerToken,
        transition: kore_abi::session::PowerTransition,
    ) -> bool {
        use kore_abi::session::Status;

        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let Some(report_slot) = crate::entry::geometry::report_slot() else {
            return false;
        };
        // `PowerReq` is `transition: u8 ‖ reserved: [u8; 7]`; the kernel
        // validates the discriminant through `PowerTransition`'s generated
        // check and refuses a non-zero reserve.
        let mut arg = [0u8; 8];
        arg[0] = transition as u8;

        let mut req = SessionBuilder::new();
        if req
            .derive_raw(token.slot(), Rights::empty(), 0, report_slot, Some(&arg))
            .is_err()
        {
            return false;
        }
        match TrapTransport::new().commit_now(&req) {
            Ok(committed) => committed.status() == Status::Ok,
            Err(_) => false,
        }
    }

    /// Off the metal userland targets — no trap to issue, so no transition.
    #[cfg(not(target_os = "none"))]
    pub fn request_power(
        _token: &PowerToken,
        _transition: kore_abi::session::PowerTransition,
    ) -> bool {
        false
    }

    /// **The metal display map + describe** (A.2) — install the full
    /// framebuffer aperture in the task's **aperture window**
    /// (`geom::APERTURE`; **write-combining**, the region's own attrs)
    /// through the LIVE session engine, and learn
    /// the mode with the claim flavor's no-arg derive: the
    /// [`GeomReport`](kore_abi::session::GeomReport) names the installed
    /// `(va, len)`, the [`DisplayReport`](kore_abi::session::DisplayReport)
    /// packs base + pitch/width/height/format — both kernel-vouched, both
    /// decoded off the R/O slots alias.  Any refusal is a fail-closed
    /// `None`.
    #[cfg(target_os = "none")]
    pub fn map_display(token: &DisplayToken) -> Option<DisplayWindow> {
        use kore_abi::session::{DisplayReport, GeomReport};

        use crate::session::SessionBuilder;
        use crate::transport::{Transport, TrapTransport};

        let report_slot = crate::entry::geometry::report_slot()?;

        // The same `map_owned` every held-BAR install uses: the KERNEL
        // discriminates on the subject's owner (`MmioOwner::Display`) and
        // reserves from the task's aperture window rather than its view
        // cursor, so nothing here names a window or a VA.  A caller that
        // could pick would be a caller that could put a framebuffer back on
        // the cursor.
        let mut map = SessionBuilder::new();
        map.map_owned(token.slot(), Rights::ACCESS | Rights::MODIFY, report_slot)
            .ok()?;
        let committed = TrapTransport::new().commit_now(&map).ok()?;
        if committed.status() != kore_abi::session::Status::Ok {
            return None;
        }
        let ptr =
            crate::entry::geometry::slot_record_va(report_slot)? as *const kore_abi::cap::CapRecord;
        // SAFETY: the founder mapped the slots alias R/O; the kernel
        // wrote the report before the trap returned.  Read volatile.
        let geom = GeomReport::decode(&unsafe { core::ptr::read_volatile(ptr) })?;
        if geom.len == 0 {
            return None;
        }

        let mut claim = SessionBuilder::new();
        claim
            .derive_raw(
                token.slot(),
                Rights::empty(),
                kore_abi::cap::ResourceKind::Display as u8,
                report_slot,
                None,
            )
            .ok()?;
        let committed = TrapTransport::new().commit_now(&claim).ok()?;
        if committed.status() != kore_abi::session::Status::Ok {
            return None;
        }
        let rec = unsafe { core::ptr::read_volatile(ptr) };
        let mode = DisplayReport::decode(&rec)?;
        Some(DisplayWindow {
            base: geom.base as *mut u8,
            mode,
        })
    }

    /// Off the metal userland targets — the honest typed refusal: there is no
    /// trap to issue (and no aperture to install into).
    #[cfg(not(target_os = "none"))]
    pub fn map_display(_token: &DisplayToken) -> Option<DisplayWindow> {
        None
    }

    /// The mapped framebuffer aperture + its kernel-vouched mode — the
    /// display service's render target.
    ///
    /// Deliberately **unowned**, unlike [`BarWindow`]/[`DmaWindow`]
    /// (crate::cap): the aperture install is an owner-rooted map of the
    /// held `MmioRegion` — no derived view record lands in any arena
    /// slot, so there is no `OwnedSlot` for a retire path to ride.  The
    /// mapping's life IS the service's life; the teardown story (and an
    /// owner, if one is then needed) arrives with mode-switch/hotplug,
    /// which will remap the aperture and must own that design whole.
    #[derive(Debug)]
    pub struct DisplayWindow {
        /// Base VA of the WC mapping.
        pub base: *mut u8,
        /// The mode (pitch / width / height / format).
        pub mode: kore_abi::session::DisplayReport,
    }

    #[cfg(test)]
    mod retirement_tests {
        use super::*;
        use kore_abi::session::{Committed, Status};

        fn committed(status: Status, index: u16) -> Committed {
            Committed::from_raw(status.encode(index))
        }

        #[test]
        fn uncertain_peer_publication_never_reopens_or_revokes_the_result_name() {
            use crate::cap::{OwnedSlot, OwnedSlotState};
            for completion in [
                committed(Status::Ok, 1),
                committed(Status::Denied, 1),
                committed(Status::Deadline, 0),
                committed(Status::Irq, 0),
                committed(Status::Unfounded, 0),
                Committed::from_raw(0xffff),
            ] {
                let mut owned = OwnedSlot {
                    slot: SlotRef::from_wire(42),
                    state: OwnedSlotState::Reserved,
                    _thread_bound: core::marker::PhantomData,
                };
                assert_eq!(
                    settle_peer_publish(&mut owned, Ok(completion)),
                    Err(PeerWindowFault::Indeterminate)
                );
                assert_eq!(owned.state, OwnedSlotState::Indeterminate);
                assert_eq!(
                    owned.retire_via(|_| panic!(
                        "unknown publication cannot be revoked by guessing"
                    )),
                    RetireOutcome::Retryable
                );
                assert_eq!(owned.state, OwnedSlotState::Indeterminate);
            }
            for status in [
                Status::Ok,
                Status::Denied,
                Status::Forged,
                Status::Revoked,
                Status::Conflict,
                Status::Malformed,
                Status::WouldBlock,
            ] {
                let mut owned = OwnedSlot {
                    slot: SlotRef::from_wire(42),
                    state: OwnedSlotState::Reserved,
                    _thread_bound: core::marker::PhantomData,
                };
                let result = settle_peer_publish(&mut owned, Ok(committed(status, 0)));
                if status == Status::Ok {
                    assert_eq!(result, Ok(()));
                } else {
                    assert_eq!(result, Err(PeerWindowFault::Commit(status as u8)));
                }
                assert_eq!(owned.state, OwnedSlotState::Reserved);
                owned.release_reservation();
            }
            let mut owned = OwnedSlot {
                slot: SlotRef::from_wire(42),
                state: OwnedSlotState::Reserved,
                _thread_bound: core::marker::PhantomData,
            };
            assert_eq!(
                settle_peer_publish(&mut owned, Err(crate::transport::TrapUnavailable)),
                Err(PeerWindowFault::Build)
            );
            assert_eq!(owned.state, OwnedSlotState::Reserved);
            owned.release_reservation();
        }

        #[test]
        fn guest_region_rights_require_read_and_preserve_w_x() {
            use kore_abi::session::{SHARE_ACCESS_R, SHARE_ACCESS_W, SHARE_ACCESS_X};

            assert_eq!(
                guest_region_rights(SHARE_ACCESS_R),
                Some(crate::cap::GUEST_REGION_RO_RIGHTS)
            );
            assert_eq!(
                guest_region_rights(SHARE_ACCESS_R | SHARE_ACCESS_X),
                Some(crate::cap::GUEST_REGION_RO_RIGHTS)
            );
            assert_eq!(
                guest_region_rights(SHARE_ACCESS_R | SHARE_ACCESS_W),
                Some(crate::cap::GUEST_REGION_RW_RIGHTS)
            );
            assert_eq!(guest_region_rights(0), None);
            assert_eq!(guest_region_rights(SHARE_ACCESS_W), None);
            assert_eq!(
                guest_region_rights(SHARE_ACCESS_R | SHARE_ACCESS_W | SHARE_ACCESS_X),
                None
            );
            assert_eq!(guest_region_rights(SHARE_ACCESS_R | 0x80), None);
        }

        #[test]
        fn single_retirement_classifies_success_dead_and_retryable() {
            assert_eq!(
                classify_owned_retirement(committed(Status::Ok, 0)),
                RetireOutcome::Retired
            );
            assert_eq!(
                classify_owned_retirement(committed(Status::Revoked, 0)),
                RetireOutcome::AlreadyDead
            );
            assert_eq!(
                classify_owned_retirement(committed(Status::Conflict, 0)),
                RetireOutcome::Retryable
            );
            assert_eq!(
                classify_owned_retirement(committed(Status::Denied, 0)),
                RetireOutcome::Retryable
            );
            assert_eq!(
                classify_owned_retirement(committed(Status::Ok, 1)),
                RetireOutcome::Retryable
            );
            assert_eq!(
                classify_owned_retirement(committed(Status::Revoked, 1)),
                RetireOutcome::Retryable
            );
        }

        #[test]
        fn receive_lineage_probe_requires_positive_live_or_dead_evidence() {
            assert_eq!(
                classify_chan_recv_probe(committed(Status::Ok, 0)),
                ChanRecvLineage::Live
            );
            assert_eq!(
                classify_chan_recv_probe(committed(Status::Revoked, 0)),
                ChanRecvLineage::Dead
            );
            assert_eq!(
                classify_chan_recv_probe(committed(Status::WouldBlock, 0)),
                ChanRecvLineage::Retryable
            );
            assert_eq!(
                classify_chan_recv_probe(committed(Status::Ok, 1)),
                ChanRecvLineage::Retryable
            );
        }

        #[test]
        fn task_start_classifies_active_retry_terminal_and_denial() {
            assert_eq!(
                classify_task_start(committed(Status::Ok, 0)),
                TaskStartOutcome::Active
            );
            assert_eq!(
                classify_task_start(committed(Status::WouldBlock, 0)),
                TaskStartOutcome::Retryable
            );
            assert_eq!(
                classify_task_start(committed(Status::Conflict, 0)),
                TaskStartOutcome::Retryable
            );
            assert_eq!(
                classify_task_start(committed(Status::Revoked, 0)),
                TaskStartOutcome::Terminal
            );
            assert_eq!(
                classify_task_start(committed(Status::Denied, 0)),
                TaskStartOutcome::Refused(Status::Denied)
            );
            assert_eq!(
                classify_task_start(committed(Status::Ok, 1)),
                TaskStartOutcome::Refused(Status::Malformed)
            );
        }

        #[test]
        fn device_reads_accept_only_a_single_successful_invocation() {
            assert_eq!(device_read_committed(committed(Status::Ok, 0)), Ok(()));
            assert_eq!(
                device_read_committed(committed(Status::Denied, 0)),
                Err(InventoryReadError::Refused(Status::Denied))
            );
            assert_eq!(
                device_read_committed(committed(Status::Ok, 1)),
                Err(InventoryReadError::MalformedCompletion)
            );
        }

        #[test]
        fn channel_space_claim_accepts_only_the_bounded_grant_contract() {
            assert!(exact_channel_space_rights(crate::cap::CHANNEL_SPACE_RIGHTS));
            assert!(!exact_channel_space_rights(Rights::DERIVE));
            assert!(!exact_channel_space_rights(
                crate::cap::CHANNEL_SPACE_RIGHTS | Rights::MODIFY
            ));
        }

        #[test]
        fn channel_endpoint_derivation_allows_a_non_invocable_delegation_pivot() {
            assert!(valid_channel_endpoint_derivation_rights(
                Rights::DERIVE | Rights::REVOKE
            ));
            assert!(valid_channel_endpoint_derivation_rights(
                Rights::ACCESS | Rights::REVOKE
            ));
            assert!(!valid_channel_endpoint_derivation_rights(Rights::empty()));
            assert!(!valid_channel_endpoint_derivation_rights(
                Rights::ACCESS | Rights::MODIFY
            ));
        }

        #[test]
        fn channel_mint_completion_proves_commit_or_atomic_rollback() {
            assert_eq!(
                classify_channel_mint(committed(Status::Ok, 0)),
                ChannelMintCompletion::Committed
            );
            assert_eq!(
                classify_channel_mint(committed(Status::Denied, 0)),
                ChannelMintCompletion::RolledBack(Status::Denied)
            );
            assert_eq!(
                classify_channel_mint(committed(Status::WouldBlock, 2)),
                ChannelMintCompletion::RolledBack(Status::WouldBlock)
            );

            assert_eq!(
                classify_channel_mint(committed(Status::Ok, 1)),
                ChannelMintCompletion::Indeterminate
            );
            assert_eq!(
                classify_channel_mint(committed(Status::Revoked, 3)),
                ChannelMintCompletion::Indeterminate
            );
            assert_eq!(
                classify_channel_mint(committed(Status::Deadline, 0)),
                ChannelMintCompletion::Indeterminate
            );
            assert_eq!(
                classify_channel_mint(Committed::from_raw(0xffff)),
                ChannelMintCompletion::Indeterminate
            );
        }
    }

    #[cfg(test)]
    mod arena_extend_tests {
        use super::*;
        use kore_abi::session::GeomReport;

        const BASE: usize = 0x7600_0000;
        const REC: usize = core::mem::size_of::<kore_abi::cap::CapRecord>();
        const CHUNK: usize = crate::entry::ARENA_SLOT_CEILING as usize * REC;

        #[test]
        fn arena_extension_report_accepts_only_dense_whole_chunks() {
            assert_eq!(
                arena_extension_span(
                    BASE,
                    GeomReport {
                        base: BASE as u64,
                        len: CHUNK as u64,
                    },
                ),
                Some(CHUNK)
            );
            assert_eq!(
                arena_extension_span(
                    BASE,
                    GeomReport {
                        base: (BASE + CHUNK) as u64,
                        len: CHUNK as u64,
                    },
                ),
                Some(2 * CHUNK)
            );

            for report in [
                GeomReport {
                    base: BASE as u64,
                    len: 0,
                },
                GeomReport {
                    base: BASE as u64,
                    len: (CHUNK / 2) as u64,
                },
                GeomReport {
                    base: (BASE + 1) as u64,
                    len: CHUNK as u64,
                },
                GeomReport {
                    base: (BASE - CHUNK) as u64,
                    len: CHUNK as u64,
                },
                GeomReport {
                    base: (BASE
                        + (crate::entry::ARENA_EXT_SLOT_CEILING as usize
                            - crate::entry::ARENA_SLOT_CEILING as usize)
                            * REC) as u64,
                    len: CHUNK as u64,
                },
            ] {
                assert_eq!(arena_extension_span(BASE, report), None);
            }
        }
    }
}

pub use metal::*;
