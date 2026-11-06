//! The inventory endowment, the arbitration vocabulary, the claim ledger,
//! the claimant-face mint, and the serve step.

use kore_abi::device::{DeviceRecord, DeviceSpecies};
use kore_abi::session::SlotRef;
use kore_abi::task::{FACE_RING_STRIDE, FACE_SUPPLY_BYTES, role};
use kore_rt::cap::{PlatformBusToken, RootComplexToken, SupplyToken, verbs};
use kore_rt::entry::FixedSlotBand;
use kore_rt::service::{
    DeviceSelector, GrantArg, MintedPairSpec, PciBarMask, PciInterruptRequirement, ReturnMode,
    Serve, StartFailure, SystemService,
};

use kore_abi::cap::Rights;

/// The most devices this broker tracks as claimed at once.
///
/// A ceiling, not a guess: the founder's own per-row device ceiling is
/// four and a machine's driver rows are a manifest fact, so a claim table
/// this size covers every row the boot chain can found several times over.
/// It is heapless because the broker runs in the control plane of every
/// driver's startup — an allocation there would make bring-up depend on a
/// heap the broker has no reason to hold.
pub const MAX_CLAIMS: usize = 16;

/// Rights carried by a broker-delivered device root.
///
/// `REVOKE` belongs on the delivered **child**, not only on the broker's
/// inventory ancestor: it lets the affine receiver retire that child (and its
/// BAR/route/port descendants) when the token dies.  Revoking the child cannot
/// reach upward into the broker's [`RootComplexToken`], so this preserves the
/// broker's reclamation authority while making receiver-side teardown lawful.
pub const DEVICE_GRANT_RIGHTS: Rights = verbs::DEVICE_OWNER_RIGHTS;

/// Claimant faces the broker's [`role::FACE_SUPPLY`] band strides into —
/// the geometry contract's own quotient, so the count can never disagree
/// with the span the founder mints.
pub const FACE_COUNT: usize = (FACE_SUPPLY_BYTES / FACE_RING_STRIDE) as usize;

/// The first minted face's [`Serve::via`] identity.  Minted vias start at
/// [`MAX_SERVE_PAIRS`](kore_rt::svc::MAX_SERVE_PAIRS) — the host's rule
/// that a minted pair must never collide with a fixed pair — so face `k`
/// arrives stamped `FACE_VIA_BASE + k` and the claim ledger's owner
/// column tells claimants apart without a wire field.
pub const FACE_VIA_BASE: u16 = kore_rt::svc::MAX_SERVE_PAIRS as u16;

/// The face legs' landing band: two base-arena slots per face (the
/// grant lane at `+2k`, the reply kick at `+2k+1`), claimed at start —
/// the exec convey-table shape, and derived from the same frozen runtime
/// layout (past the boot-image band, the serve scratch, and the
/// report + heap-carve tail).  BASE arena on purpose: a rendezvous
/// landing must be a base-arena slot — the kernel denies an extension
/// landing, which is how this band's first draft (at
/// `ARENA_SLOT_CEILING`) refused every walking post.  Fixed arithmetic,
/// not an allocator: a landed leg IS the slot [`MintedPairSpec`] then
/// cites for the face's life, so the slot map must be derivable from
/// the face index alone.
const FACE_LAND_BASE: u32 =
    kore_abi::task::MAX_BOOT_HANDLES as u32 + kore_rt::entry::SERVE_SCRATCH_SLOTS as u32 + 2;
const FACE_LAND_END: u32 = FACE_LAND_BASE + 2 * FACE_COUNT as u32;

/// A conveyed face leg must at least be postable by the broker.
const FACE_LEG_RIGHTS: Rights = Rights::ACCESS;

const _: () = assert!(
    kore_rt::svc::window_bytes::<DeviceBroker>(kore_rt::svc::REQ_CAP, kore_rt::svc::REP_CAP)
        <= FACE_RING_STRIDE as usize,
    "a device pair must fit one face stride, or the mint would format past its slot"
);

/// The client→service vocabulary — **ownership arbitration**.
///
/// Deliberately no I/O verb of any kind: see the crate docs for why a generic
/// device request would be the god-object `res/DEVICE_SERVICE.md` §1 rules
/// out. `Inspect` is description-only and grants no authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceRequest {
    /// Hand me the first unclaimed entry matching `selector`.
    ///
    /// Claiming twice with the same selector yields two *different*
    /// entries, which is how a row that drives two controllers of one
    /// class gets both without a nth-match field to get wrong.
    Claim(DeviceSelector),
    /// Describe the first entry with the selector's species and identity/class
    /// tuple without applying its resource requirements or claiming it.
    ///
    /// This is the diagnostic twin of `Claim`: it lets a driver distinguish
    /// absent hardware from an inventory row that lacks BAR, IOMMU, interrupt,
    /// or endpoint-admission facts. The reply is data only.
    Inspect(DeviceSelector),
    /// Read one bounded firmware/admission page from this broker's held
    /// root. Description only: no claim-ledger mutation or grant.
    InspectI2c { row: u8 },
    /// Request that entry `index` be put back in the pool.
    ///
    /// The current broker refuses this index-only form: an integer does not
    /// prove the granted capability (and all device activity beneath it) is
    /// dead. The wire verb is reserved for an authority-backed return.
    Release {
        /// The inventory index a previous `Claim` answered with.
        index: u32,
    },
    /// Close admission to the exact PCI grant on this structural serving
    /// face. This does not release the claim or free any physical resource.
    CloseClaim { index: u32, grant: u64 },
    /// Issuer-owned fabric containment, only after exact admission drain.
    QuiesceClaim { index: u32, grant: u64 },
    /// Retire registered DMA while retaining this exact claim.
    RetireClaimDma { index: u32, grant: u64 },
    /// How many entries the inventory holds.
    ///
    /// Exists so a claimant can tell *"this machine has no such device"*
    /// from *"somebody already has it"* — on a serial-less machine that
    /// distinction is the difference between a hardware report and a
    /// wiring bug.
    Count,
    /// Open a claimant-face provisioning window over face slot `face` of
    /// the [`role::FACE_SUPPLY`] band — the founding parent's half of the
    /// minted-pair handshake (`res/DEVICE_SERVICE.md` §10).  The reply
    /// reserves the slot; the face's two channel sends then arrive as
    /// return-plane conveys, each an implicit [`Provision`](Self::Provision)
    /// step, and the second completes the mint.
    MintFace {
        /// Which stride of the band to format and attach.
        face: u32,
    },
    /// **Implicit only** — synthesized by the serve host when a convey
    /// lands on the broker's walking return post
    /// ([`SvcProtocol::implicit_return_request`](kore_rt::svc::SvcProtocol)).
    /// Never decoded from the request ring: a claimant writing it would be
    /// forging the broker's own internal vocabulary, so the wire codec
    /// refuses it fail-closed.
    Provision,
}

/// The reply vocabulary — **data only**. A granted device arrives as a
/// delivered token; the reply says which entry it was.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceResponse {
    /// The device is yours. The token rides the grant channel.
    Granted {
        /// The function's diagnostic word
        /// ([`DeviceRecord::diag`](kore_abi::device::DeviceRecord::diag)).
        diag: u32,
        /// PCI revision id — the claimant's `si_stepping` scalar.
        revision: u8,
        /// PCI vendor id, or zero for a platform device.
        vendor: u16,
        /// PCI device id, or zero for a platform device.
        device: u16,
        /// The inventory index handed over — name it to `Release`.
        index: u32,
        /// Which species the delivered token must be possessed as
        /// (`claim_device` vs `claim_platform_device`).
        species: DeviceSpecies,
    },
    /// Only tracked admission drain; never physical retirement or release.
    ClaimClosed(kore_abi::session::DeviceClaimCloseReport),
    /// Fabric stop evidence only; no physical claim or backing is released.
    ClaimQuiesced(kore_abi::session::QuiesceReport),
    ClaimDmaRetired(kore_abi::session::DeviceClaimDmaReport),
    /// Read-only inventory description; no token rides the reply.
    Observed(DeviceObservation),
    I2cObserved(kore_abi::device::I2cAdmissionPage),
    /// The entry is back in the pool.
    ///
    /// Reserved for an authority-backed release protocol; the current broker
    /// never emits this response to an index-only request.
    Released,
    /// The inventory's size.
    Counted {
        /// How many entries it holds.
        len: u32,
    },
    /// `MintFace` accepted: the face slot is reserved and the walking
    /// return post now expects the face's two channel sends, grant lane
    /// first.
    Provisioning {
        /// The reserved face slot, echoed.
        face: u32,
    },
    /// The first provisioning convey (the grant lane) is banked; one
    /// more (the reply kick) completes the face.
    Banked {
        /// The face being provisioned, echoed.
        face: u32,
    },
    /// The face is live: its stride is formatted, the pair is attached,
    /// and requests over it arrive stamped with this identity.
    Minted {
        /// The pair's [`Serve::via`] stamp (`FACE_VIA_BASE + face`).
        via: u16,
    },
    /// Refused — and a refusal emits **no** grant (fail-closed: no
    /// authority leaves on an error path).
    Refused(DeviceRefusal),
}

/// The bounded facts returned by [`DeviceRequest::Inspect`].
///
/// The selector already carries identity and class, so the observation returns
/// only the facts needed to explain a failed claim. A granted wildcard claim
/// separately reports the selected PCI identity because the claimant did not
/// already know it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceObservation {
    pub index: u32,
    pub species: DeviceSpecies,
    pub flags: u8,
    pub diag: u32,
    pub revision: u8,
}

impl From<DeviceRecord> for DeviceObservation {
    fn from(record: DeviceRecord) -> Self {
        Self {
            index: record.index,
            species: record.species,
            flags: record.flags,
            diag: record.diag,
            revision: record.revision,
        }
    }
}

/// Why the broker refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceRefusal {
    /// No entry on this machine matches the selector — a fact about the
    /// hardware, and distinct from [`Taken`](Self::Taken) on purpose.
    NoMatch,
    /// Every match is already claimed. Also what a re-claim after a
    /// claimant *crash* looks like, which is honest: the broker cannot
    /// prove the holder is gone, so it will not hand out a device
    /// somebody may still be driving (see the crate docs).
    Taken,
    /// An inventory observation or hand-out session refused — the kernel's
    /// own gate or the report transport said no.
    Denied,
    /// A `Release` named an index nobody holds.
    NotHeld,
    /// This broker was founded without a [`role::FACE_SUPPLY`] band, so
    /// it can mint no claimant face at all.
    NoSupply,
    /// `MintFace` named a face slot out of range or not vacant, or a
    /// provisioning window is already open (one mint in flight — the
    /// founding parent drives them in lockstep).
    BadFace,
    /// A return-plane convey arrived that no open provisioning window
    /// expects, or carried something other than a channel send.  The
    /// half-built face was unwound with the refusal — its banked leg
    /// retired, its slot vacant again (fail-closed: no face exists by
    /// half).
    BadConvey,
    /// Claim admission has not drained, or containment currently holds its writer.
    Busy,
}

/// The `role::ROOT_COMPLEX` endowment: the sole inventory grant, claimed
/// out of the boot block.
///
/// Deliberately **not** `role::DEVICE` beside it: a task that both
/// arbitrates devices and drives one is the god-object the design rules
/// out, and the profile is where that is made unrepresentable rather than
/// merely discouraged.
#[derive(Debug, kore_rt::Bundle)]
#[bundle(name = "device")]
pub struct DeviceEndowment {
    /// **This row's root complex** (#572): every broker row endows
    /// `ROOT_COMPLEX@0` = "my root", and the founder maps broker ordinal
    /// k → the k-th discovered root — one endowment type, N instances.
    /// A row whose root the machine did not discover is skipped at
    /// founding, so a founded broker always holds a live root.
    #[bundle(role = role::ROOT_COMPLEX)]
    pub root: RootComplexToken,
    /// The platform-device inventory — the roots' non-PCI twin (#572):
    /// PS/2-class devices sit on no PCIe root, so their arbiter rides
    /// broker row 0 alone; every other ordinal stamps vacant and this
    /// resolves `None`.
    #[bundle(role = role::PLATFORM_BUS, optional)]
    pub platform: Option<PlatformBusToken>,
    /// The claimed claimant-face band ([`role::FACE_SUPPLY`]), when this
    /// row was founded with one — the broker's full-span serving sub of
    /// the shelf-held band whose strides become minted claimant pairs.
    ///
    /// **Optional by design, not leniency**: the role is absent until the
    /// founder grows its band arm, and a broker without it still
    /// arbitrates claims — it just refuses every `MintFace` with
    /// [`DeviceRefusal::NoSupply`].  A role that is *present* but does
    /// not claim stays a hard [`kore_rt::service::BindError`]: a half-usable endowment must
    /// refuse the row, never dilute into the absent case.
    #[bundle(
        role = role::FACE_SUPPLY,
        optional
    )]
    /// Declaring the band beside the store is also what arms the founder's
    /// claimant-band mint; the derive emits both coordinates into one profile.
    pub faces: Option<SupplyToken>,
}

/// One handed-out entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Claim {
    /// The store-local inventory index.  Which of this row's two
    /// inventories (its root, or — ordinal 0 — the platform bus) it is
    /// local to is the `species` beside it: the two stores vend disjoint
    /// species, so no third discriminator exists to get wrong (#572).
    index: u32,
    /// Which species it was handed out as.
    species: DeviceSpecies,
    /// **Which serving pair claimed it** — the §10 endpoint stamp
    /// ([`Serve::via`]), recorded so the books attribute every held
    /// function to the driver row it went to.  Structural: the founding
    /// wired each claimant its own pair, so this is which face the claim
    /// arrived on, never a wire field.  Quarantine stays safe across
    /// owners; a future authority-backed release will check it.
    owner: u16,
    /// Exact kernel publication observed after delivery. Missing correlation
    /// retains the claim and withholds future retirement; it is not a rollback.
    identity: Option<kore_abi::session::DeviceClaimReport>,
}

/// One claimant-face slot's lifecycle — the mint handshake's state
/// machine, kept beside the slot arithmetic that names its two landing
/// slots ([`FACE_LAND_BASE`]`+ 2k / + 2k + 1`).
///
/// At most one face is ever off `Vacant`-or-`Live` (the open provisioning
/// window, [`DeviceBroker::minting`]): the founding parent drives mints
/// in lockstep, and refusing a second window keeps the walking landing
/// offer unambiguous.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum FaceState {
    /// Nothing minted here.
    #[default]
    Vacant,
    /// `MintFace` reserved the slot; no leg has landed yet.
    Reserved,
    /// The grant lane landed and is banked; the reply kick completes it.
    Granted,
    /// The face serves: stride formatted, pair attached, legs cited by
    /// the live [`MintedPairSpec`].
    Live,
}

/// Occupancy snapshot (the sibling services' `*Stats` shape).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceStats {
    /// Devices handed out since bind.
    pub granted: u32,
    /// Devices given back.
    pub released: u32,
    /// Claims refused for want of a match.
    pub unmatched: u32,
    /// Claims refused because every match was already held.
    pub contended: u32,
    /// Inventory observations or hand-outs the kernel's own gate refused —
    /// **visible, never silent**: a failed read or grant is a wiring fault,
    /// and it must not read as "no such device".
    pub denied: u32,
    /// Claimant faces currently live.
    pub faces: u32,
    /// Face mints that failed after opening — a bad convey, a dropped
    /// attach — each one a wiring fault somebody should read.
    pub face_faults: u32,
}

/// Preserve the reason an authoritative inventory-count query failed in the
/// founder-visible service-start NACK. These are local diagnostics, not wire
/// refusal codes.
const fn inventory_start_detail(error: verbs::InventoryReadError) -> u64 {
    match error {
        verbs::InventoryReadError::Unavailable => 1,
        verbs::InventoryReadError::Build => 2,
        verbs::InventoryReadError::Transport => 3,
        verbs::InventoryReadError::MalformedCompletion => 4,
        verbs::InventoryReadError::MalformedReport => 5,
        verbs::InventoryReadError::Refused(status) => 0x100 | status as u64,
    }
}

/// The device broker — the inventory and a claim ledger, and nothing else.
///
/// It holds no device: the whole point is that it can only *hand one out*.
/// Its state is which entries are spoken for, which is the smallest thing
/// that makes "one device, one owner" true.
#[derive(Debug)]
pub struct DeviceBroker {
    /// This row's root complex (#572 — one broker instance per root).
    root: RootComplexToken,
    /// The platform-device inventory, when this row was founded with one
    /// (broker ordinal 0 alone).
    platform: Option<PlatformBusToken>,
    claims: heapless::Vec<Claim, MAX_CLAIMS>,
    stats: DeviceStats,
    /// The root's authoritative, boot-frozen inventory size; the platform
    /// bus's rides beside it.
    ///
    /// Populated by the fallible pre-readiness hook rather than `bind`, whose
    /// trait cannot report failure. `None` is retained as a defensive state:
    /// neither `Count` nor `Claim` may turn a failed startup observation into
    /// an empty inventory.
    root_len: Option<u32>,
    platform_len: Option<u32>,
    /// The claimant-face band's possessed supply, when this row was
    /// founded with one ([`DeviceEndowment::faces`]).
    supply: Option<SupplyToken>,
    /// The whole-band serving view — mapped R/W **once** at start for the
    /// task's life (the image face table's map-once contract, which is
    /// also what makes [`MintedPairSpec::new`]'s warrant citable).
    face_window: Option<core::ptr::NonNull<u8>>,
    /// The face legs' claimed landing band (`FACE_LAND_BASE..FACE_LAND_END`)
    /// — held for the task's life so nothing else can claim a slot a
    /// live pair cites.
    _face_land: Option<FixedSlotBand>,
    /// Per-face lifecycle; see [`FaceState`].
    faces: [FaceState; FACE_COUNT],
    /// The face with the open provisioning window, if any — at most one,
    /// which is what keeps [`Self::pending_return_post`]'s offer
    /// unambiguous.
    minting: Option<u32>,
    /// A completed mint's pair, parked for the serve host's attach pass
    /// ([`SystemService::pending_pair_attach`]).
    pending_attach: Option<MintedPairSpec>,
}

impl DeviceBroker {
    /// Occupancy snapshot.
    pub fn stats(&self) -> DeviceStats {
        self.stats
    }

    /// Does `record` carry the selector's species and identity/class tuple?
    ///
    /// This deliberately excludes claimability and resource requirements. It
    /// is the read-only inventory predicate used by `Inspect`; `matches`
    /// layers the authority-bearing requirements over it.
    fn describes(record: &DeviceRecord, selector: &DeviceSelector) -> bool {
        match selector {
            DeviceSelector::Pci(sel) => {
                record.species == DeviceSpecies::Pci
                    && record.class == sel.class
                    && record.subclass == sel.subclass
                    && record.progif == sel.progif
                    && sel.identity.matches(record.vendor, record.device)
            }
            DeviceSelector::Platform { hid, .. } => {
                record.species == DeviceSpecies::Platform
                    && record.class == DeviceRecord::PLATFORM_CLASS_HID
                    && record.hid() == *hid
            }
            DeviceSelector::PlatformI2cHid { .. } => {
                record.species == DeviceSpecies::Platform
                    && record.class == DeviceRecord::PLATFORM_CLASS_I2C_HID
            }
            DeviceSelector::PciI2cHid { .. } => {
                // The native input profile currently uses Intel LPSS
                // reset/clock semantics, not arbitrary serial-bus IP.
                // Linux's intel-lpss-pci.c LNL-M / PTL-P tables map these
                // I2C functions to its 100 MHz ehl_i2c_info; a852/e452 are UART.
                // https://github.com/torvalds/linux/blob/master/drivers/mfd/intel-lpss-pci.c
                record.is_native_i2c_hid_candidate()
            }
        }
    }

    /// Does `record` satisfy `selector`?
    ///
    /// This is the founding loop's `claimable` predicate, lifted into ring
    /// 3 — minus the two xHCI register probes, which went where they
    /// belong: the driver reads its own controller after it holds it.
    fn matches(record: &DeviceRecord, selector: &DeviceSelector) -> bool {
        match selector {
            DeviceSelector::Pci(sel) => {
                if !Self::describes(record, selector) {
                    return false;
                }
                // Endpoint admission is positive.  In particular, absence
                // of the bridge bit is not proof that firmware released a
                // display controller to us.
                if !record.satisfies(DeviceRecord::DRIVER_ENDPOINT)
                    || record.satisfies(DeviceRecord::NOT_DRIVER_ENDPOINT)
                    || record.flags & DeviceRecord::IS_BRIDGE != 0
                {
                    return false;
                }
                let present_bars = (record.diag & 0x3f) as u8;
                let user_bars = ((record.diag >> 6) & 0x3f) as u8;
                let has_user_bar0 = user_bars & PciBarMask::BAR0.bits() != 0;
                if record.msix_table_size > DeviceRecord::MAX_MSIX_TABLE_ENTRIES
                    || user_bars & !present_bars != 0
                    || record.satisfies(DeviceRecord::HAS_BAR0) != has_user_bar0
                    || (record.satisfies(DeviceRecord::HAS_REMAPPED_MSIX)
                        && (!record.satisfies(DeviceRecord::HAS_REMAPPED_MSI)
                            || record.msix_table_size == 0))
                {
                    // Inventory is kernel-produced, so any impossible PCI
                    // geometry is corruption or producer/decoder drift. Do
                    // not consume the affine row under contradictory facts.
                    return false;
                }
                let required_bars = sel.bars.bits();
                if user_bars & required_bars != required_bars {
                    return false;
                }
                let mut required = DeviceRecord::DRIVER_ENDPOINT;
                if required_bars & PciBarMask::BAR0.bits() != 0 {
                    // Keep the legacy flag and the exact diagnostic bitmap
                    // mutually consistent for the common BAR0 case.
                    required |= DeviceRecord::HAS_BAR0;
                }
                if sel.dma {
                    required |= DeviceRecord::HAS_BINDING;
                }
                required |= match sel.interrupts {
                    PciInterruptRequirement::None => 0,
                    PciInterruptRequirement::MsiFamily => DeviceRecord::HAS_REMAPPED_MSI,
                    PciInterruptRequirement::Msix(block) => {
                        let Some(end) = block.first_entry().checked_add(block.count()) else {
                            return false;
                        };
                        if end > record.msix_table_size {
                            return false;
                        }
                        DeviceRecord::HAS_REMAPPED_MSIX
                    }
                };
                if matches!(sel.interrupts, PciInterruptRequirement::Msix(_)) {
                    // Every strict MSI-X block is also an MSI-family route;
                    // demanding both catches an internally inconsistent
                    // inventory instead of laundering it into a grant.
                    required |= DeviceRecord::HAS_REMAPPED_MSI;
                }
                record.satisfies(required)
            }
            DeviceSelector::Platform { irq, dma, .. } => {
                let mut required = 0;
                if *irq {
                    required |= DeviceRecord::HAS_LINE;
                }
                if *dma {
                    required |= DeviceRecord::HAS_BINDING;
                }
                Self::describes(record, selector) && record.satisfies(required)
            }
            DeviceSelector::PlatformI2cHid { irq } => {
                Self::describes(record, selector)
                    && (!*irq || record.satisfies(DeviceRecord::HAS_LINE))
            }
            DeviceSelector::PciI2cHid { irq } => {
                let resources = DeviceSelector::Pci(kore_rt::service::PciSelector {
                    class: 0x0c,
                    subclass: 0x80,
                    progif: 0,
                    identity: kore_rt::service::PciIdentitySelector::ANY,
                    bars: PciBarMask::BAR0,
                    dma: false,
                    interrupts: PciInterruptRequirement::None,
                });
                Self::describes(record, selector)
                    && record.diag & DeviceRecord::PCI_DIAG_I2C_HID != 0
                    && Self::matches(record, &resources)
                    && (!*irq || record.satisfies(DeviceRecord::HAS_LINE))
            }
        }
    }

    /// Scan one inventory for a description-only match.
    fn inspect_with<E>(
        len: u32,
        selector: &DeviceSelector,
        mut read: impl FnMut(u32) -> Result<DeviceRecord, E>,
    ) -> Result<DeviceRecord, DeviceRefusal> {
        for index in 0..len {
            let record = read(index).map_err(|_| DeviceRefusal::Denied)?;
            if Self::describes(&record, selector) {
                return Ok(record);
            }
        }
        Err(DeviceRefusal::NoMatch)
    }

    /// Scan one inventory's authoritative bound through a supplied record
    /// reader.
    ///
    /// The closure seams keep host tests honest without fabricating an
    /// affine token. A read error wins over `NoMatch`/`Taken`: those
    /// hardware facts are valid only after every required entry was
    /// observed.
    fn scan_with<E>(
        len: u32,
        is_claimed: impl Fn(u32) -> bool,
        selector: &DeviceSelector,
        mut read: impl FnMut(u32) -> Result<DeviceRecord, E>,
    ) -> Result<DeviceRecord, DeviceRefusal> {
        let mut any = false;
        for index in 0..len {
            let record = read(index).map_err(|_| DeviceRefusal::Denied)?;
            if !Self::matches(&record, selector) {
                continue;
            }
            any = true;
            if !is_claimed(record.index) {
                return Ok(record);
            }
        }
        Err(if any {
            DeviceRefusal::Taken
        } else {
            DeviceRefusal::NoMatch
        })
    }

    /// Is `(species, index)` already spoken for?  The ledger is
    /// per-store, and the species IS the store here: this row's root
    /// vends PCI functions, its (ordinal-0-only) platform bus the rest.
    fn claimed(&self, species: DeviceSpecies, index: u32) -> bool {
        self.claims
            .iter()
            .any(|claim| claim.species == species && claim.index == index)
    }

    /// The arena slot holding the store that vends `species` — the grant
    /// subject a claim derives from.
    fn store_slot(&self, species: DeviceSpecies) -> Option<kore_abi::session::SlotRef> {
        match species {
            DeviceSpecies::Pci => Some(self.root.slot()),
            DeviceSpecies::Platform => self.platform.as_ref().map(|token| token.slot()),
        }
    }

    /// Find the first unclaimed entry matching `selector` — this row's
    /// root for a PCI selector, its platform bus for a platform one.  The
    /// species split routes wholesale: a store that cannot vend the
    /// selector's species is never scanned, and a broker holding no
    /// platform bus answers a platform selector `NoMatch` (drivers probe
    /// every broker; ordinal 0 answers).
    fn find(&self, selector: &DeviceSelector) -> Result<DeviceRecord, DeviceRefusal> {
        match selector {
            DeviceSelector::Pci(_) | DeviceSelector::PciI2cHid { .. } => {
                let len = self.root_len.ok_or(DeviceRefusal::Denied)?;
                Self::scan_with(
                    len,
                    |index| self.claimed(DeviceSpecies::Pci, index),
                    selector,
                    |index| verbs::device_record(&self.root, index),
                )
            }
            DeviceSelector::Platform { .. } | DeviceSelector::PlatformI2cHid { .. } => {
                let Some(bus) = self.platform.as_ref() else {
                    return Err(DeviceRefusal::NoMatch);
                };
                let len = self.platform_len.ok_or(DeviceRefusal::Denied)?;
                Self::scan_with(
                    len,
                    |index| self.claimed(DeviceSpecies::Platform, index),
                    selector,
                    |index| verbs::device_record(bus, index),
                )
            }
        }
    }

    /// Observe a matching record without consulting or mutating the claim
    /// ledger. Observation is metadata, never evidence that authority is free.
    fn inspect(&self, selector: &DeviceSelector) -> Result<DeviceRecord, DeviceRefusal> {
        match selector {
            DeviceSelector::Pci(_) | DeviceSelector::PciI2cHid { .. } => {
                let len = self.root_len.ok_or(DeviceRefusal::Denied)?;
                Self::inspect_with(len, selector, |index| {
                    verbs::device_record(&self.root, index)
                })
            }
            DeviceSelector::Platform { .. } | DeviceSelector::PlatformI2cHid { .. } => {
                let Some(bus) = self.platform.as_ref() else {
                    return Err(DeviceRefusal::NoMatch);
                };
                let len = self.platform_len.ok_or(DeviceRefusal::Denied)?;
                Self::inspect_with(len, selector, |index| verbs::device_record(bus, index))
            }
        }
    }

    fn inspect_i2c(&self, row: u8) -> Result<kore_abi::device::I2cAdmissionPage, DeviceRefusal> {
        // Use the already-held root; no temporary device capability is
        // manufactured merely to diagnose why a device could not be claimed.
        Self::inspect_i2c_with(
            row,
            |row| verbs::i2c_admission_page(&self.root, row).map_err(|_| DeviceRefusal::Denied),
            |index| self.claimed(DeviceSpecies::Pci, index),
        )
    }

    fn inspect_i2c_with(
        row: u8,
        mut read: impl FnMut(u8) -> Result<kore_abi::device::I2cAdmissionPage, DeviceRefusal>,
        claimed: impl FnOnce(u32) -> bool,
    ) -> Result<kore_abi::device::I2cAdmissionPage, DeviceRefusal> {
        use kore_abi::device::{
            I2C_ADMISSION_BUILD_ROW, I2C_ADMISSION_CLAIMED, I2C_ADMISSION_PAGES_PER_CANDIDATE,
        };
        let mut page = read(row)?;
        if page.row != row || !page.valid() {
            return Err(DeviceRefusal::Denied);
        }
        if row != 0
            && row < I2C_ADMISSION_BUILD_ROW
            && (row - 1) % I2C_ADMISSION_PAGES_PER_CANDIDATE == 0
        {
            let identity = read(row + 2)?;
            if identity.row != row + 2 || !identity.valid() {
                return Err(DeviceRefusal::Denied);
            }
            if claimed(identity.words[0]) {
                page.words[1] |= u32::from(I2C_ADMISSION_CLAIMED);
            }
        }
        Ok(page)
    }

    /// Undo the reservation made immediately before a device grant was
    /// emitted.  `ServeHost` is single-step: no later request can interleave
    /// between that push and the grant-session outcome, so the reservation is
    /// exactly the ledger tail.
    fn rollback_dropped_grant(
        claims: &mut heapless::Vec<Claim, MAX_CLAIMS>,
        stats: &mut DeviceStats,
        response: &mut DeviceResponse,
    ) {
        let DeviceResponse::Granted { index, species, .. } = *response else {
            debug_assert!(false, "device grant drop without a Granted reply");
            return;
        };

        // The owner stamp is not re-derivable from the reply, so the tail
        // match is on the grant's own identity (index + species); the
        // single-step lifecycle makes the tail the only candidate either way.
        let expected = Some((index, species));
        if claims.last().map(|c| (c.index, c.species)) != expected {
            // Never remove unrelated authority in a release build if the
            // single-step lifecycle is violated.  The reply still becomes a
            // refusal below, while the unexpected row stays conservatively
            // reserved instead of becoming double-claimable.
            debug_assert!(
                false,
                "the dropped device grant must be the ledger's newest reservation"
            );
        } else {
            let _ = claims.pop();
            stats.granted = stats.granted.saturating_sub(1);
        }
        stats.denied = stats.denied.saturating_add(1);
        *response = DeviceResponse::Refused(DeviceRefusal::Denied);
    }

    fn owned_claim_identity(
        claims: &heapless::Vec<Claim, MAX_CLAIMS>,
        index: u32,
        owner: u16,
        grant: u64,
    ) -> Option<kore_abi::session::DeviceClaimReport> {
        claims
            .iter()
            .find(|claim| {
                claim.index == index && claim.species == DeviceSpecies::Pci && claim.owner == owner
            })?
            .identity
            .filter(|identity| identity.grant.0 == grant)
    }

    fn retain_claim_identity(
        claims: &mut heapless::Vec<Claim, MAX_CLAIMS>,
        index: u32,
        owner: u16,
        identity: kore_abi::session::DeviceClaimReport,
    ) {
        if identity.generation == 0 {
            return;
        }
        if let Some(claim) = claims.iter_mut().find(|claim| {
            claim.index == index && claim.species == DeviceSpecies::Pci && claim.owner == owner
        }) {
            // The committed callback runs once. Never replace existing
            // correlation with an observation of a later generation.
            if claim.identity.is_none() {
                claim.identity = Some(identity);
            }
        }
    }

    /// Classify an index-only release without mutating the ledger.
    ///
    /// A known index is `Denied`, not released: possession of the integer says
    /// nothing about whether the live device capability was revoked. An
    /// unknown index retains the more specific `NotHeld` answer.
    fn unproven_release_refusal(
        claims: &heapless::Vec<Claim, MAX_CLAIMS>,
        index: u32,
    ) -> DeviceRefusal {
        if claims.iter().any(|claim| claim.index == index) {
            DeviceRefusal::Denied
        } else {
            DeviceRefusal::NotHeld
        }
    }

    /// Face `k`'s two leg landing slots: the grant lane, then the reply
    /// kick — fixed arithmetic over the claimed band, because a landed
    /// leg IS the slot the live [`MintedPairSpec`] cites for the face's
    /// whole life.
    fn face_leg_slots(face: u32) -> (SlotRef, SlotRef) {
        let base = FACE_LAND_BASE + 2 * face;
        (SlotRef::from_wire(base), SlotRef::from_wire(base + 1))
    }

    /// The pure `MintFace` admission — the [`scan_with`](Self::scan_with)
    /// discipline: all of the policy, none of the metal, so the host
    /// tests pin the refusal matrix.
    fn face_open_check(
        has_supply: bool,
        faces: &[FaceState],
        minting: Option<u32>,
        face: u32,
    ) -> Result<(), DeviceRefusal> {
        if !has_supply {
            return Err(DeviceRefusal::NoSupply);
        }
        if minting.is_some() {
            return Err(DeviceRefusal::BadFace);
        }
        match faces.get(face as usize) {
            Some(FaceState::Vacant) => Ok(()),
            _ => Err(DeviceRefusal::BadFace),
        }
    }

    /// Tear a half-built (or attach-refused) face back to `Vacant`:
    /// retire whatever landed in its leg slots, close any open window,
    /// and count the fault where the shell's status poll can see it.
    /// Retiring a never-landed slot is a no-op refusal, so this is
    /// lawful from every phase.
    fn unwind_face(&mut self, face: u32) {
        let (grant, kick) = Self::face_leg_slots(face);
        let _ = verbs::retire_slot_outcome(grant);
        let _ = verbs::retire_slot_outcome(kick);
        if let Some(state) = self.faces.get_mut(face as usize) {
            *state = FaceState::Vacant;
        }
        if self.minting == Some(face) {
            self.minting = None;
        }
        self.stats.face_faults = self.stats.face_faults.saturating_add(1);
    }
}

impl SystemService for DeviceBroker {
    const NAME: &'static str = "device";
    type Endowment = DeviceEndowment;

    /// **Walking** — the exec convey-table mode: the return plane's
    /// landings rotate through service-owned slots, because what lands
    /// there (a mint's channel legs today; a returned device capability
    /// when the authority-backed `Release` arrives) becomes long-lived
    /// state at slots this service must cite.  Lawful exactly since the
    /// single-pair flip: the serve host refuses a Retained/Walking
    /// service founded with a POSTED extra pair (the rotation lives on
    /// pair 0 alone), and the static second pair is gone.
    ///
    /// The walking offer stream ([`Self::pending_return_post`]) is
    /// host-global while return posts are per-fixed-pair, so a
    /// supply-founded broker must serve exactly ONE fixed pair — two
    /// hosts would race one offered landing.  `start_on_metal` refuses
    /// that shape closed.
    const RETURN_MODE: ReturnMode = ReturnMode::Walking;

    fn bind(endowment: DeviceEndowment) -> Self {
        Self {
            root: endowment.root,
            platform: endowment.platform,
            claims: heapless::Vec::new(),
            stats: DeviceStats::default(),
            root_len: None,
            platform_len: None,
            supply: endowment.faces,
            face_window: None,
            _face_land: None,
            faces: [FaceState::Vacant; FACE_COUNT],
            minting: None,
            pending_attach: None,
        }
    }

    /// Fetch each held inventory's count once through the explicit
    /// successful count report.  A refused/garbled observation on either
    /// held store prevents readiness and carries its exact bounded
    /// category in the founder NACK instead of advertising a truncated
    /// machine.
    ///
    /// A founded face band is readied here too — geometry gate, the
    /// map-once serving view, the leg landing band — and fail-closed at
    /// start like the image face table: a vocabulary the service cannot
    /// serve refuses readiness, not first mint.  Details `0x200..` in the
    /// founder NACK, clear of the inventory's `1..=5`/`0x1xx` band.
    fn start_on_metal(&mut self, rt: &kore_rt::entry::Rt) -> Result<(), StartFailure> {
        let len = verbs::inventory_len(&self.root)
            .map_err(|error| StartFailure::new(inventory_start_detail(error)))?;
        self.root_len = Some(len);
        if let Some(bus) = self.platform.as_ref() {
            let len = verbs::inventory_len(bus)
                .map_err(|error| StartFailure::new(inventory_start_detail(error)))?;
            self.platform_len = Some(len);
        }
        if let Some(supply) = self.supply.as_ref() {
            // The walking-offer ambiguity guard (see `RETURN_MODE`): a
            // face-minting broker serves exactly one fixed pair.
            if <Self as kore_rt::svc::SvcProtocol>::SERVE_PAIRS > 1 {
                return Err(StartFailure::new(0x205));
            }
            if supply.len() != FACE_SUPPLY_BYTES {
                return Err(StartFailure::new(0x200));
            }
            let win = verbs::map_held_rw(supply.slot()).ok_or_else(|| StartFailure::new(0x201))?;
            if win.len() < FACE_SUPPLY_BYTES as usize {
                return Err(StartFailure::new(0x202));
            }
            let band = rt
                .claim_fixed_slot_band(FACE_LAND_BASE, FACE_LAND_END)
                .ok_or_else(|| StartFailure::new(0x204))?;
            self.face_window = Some(win.cast::<u8>());
            self._face_land = Some(band);
        }
        Ok(())
    }

    /// Answer one arbitration request.
    ///
    /// The only verb that emits authority is `Claim`, it emits exactly one
    /// grant, and every refusal path emits none — the D4 bounded-emission
    /// discipline, and the fail-closed rule that no authority leaves on an
    /// error path.
    fn serve(&mut self, req: DeviceRequest, sv: &mut Serve<'_, DeviceResponse>) {
        match req {
            // The machine total across every held inventory.  A held store
            // whose count never resolved refuses rather than summing a
            // silently truncated machine.
            // This ROW's total: its root's functions plus — ordinal 0 —
            // the platform devices.  A held store whose count never
            // resolved refuses rather than summing a silently truncated
            // inventory.
            DeviceRequest::Count => {
                let mut total = 0u32;
                let mut whole = true;
                match self.root_len {
                    Some(len) => total = total.saturating_add(len),
                    None => whole = false,
                }
                if self.platform.is_some() {
                    match self.platform_len {
                        Some(len) => total = total.saturating_add(len),
                        None => whole = false,
                    }
                }
                if whole {
                    sv.reply(DeviceResponse::Counted { len: total });
                } else {
                    self.stats.denied = self.stats.denied.saturating_add(1);
                    sv.reply(DeviceResponse::Refused(DeviceRefusal::Denied));
                }
            }
            DeviceRequest::InspectI2c { row } => match self.inspect_i2c(row) {
                Ok(page) => sv.reply(DeviceResponse::I2cObserved(page)),
                Err(reason) => sv.reply(DeviceResponse::Refused(reason)),
            },
            DeviceRequest::Inspect(selector) => match self.inspect(&selector) {
                Ok(record) => sv.reply(DeviceResponse::Observed(record.into())),
                // A negative observation is not a refused claim, so it must
                // not contaminate the claim-only `unmatched` counter.
                Err(DeviceRefusal::NoMatch) => {
                    sv.reply(DeviceResponse::Refused(DeviceRefusal::NoMatch));
                }
                Err(DeviceRefusal::Denied) => {
                    self.stats.denied = self.stats.denied.saturating_add(1);
                    sv.reply(DeviceResponse::Refused(DeviceRefusal::Denied));
                }
                // `inspect` cannot consult the claim ledger or speak the
                // face/release vocabulary.  Preserve fail-closed behavior if
                // that invariant is ever broken.
                Err(_) => {
                    debug_assert!(false, "read-only inspection returned an invalid refusal");
                    self.stats.denied = self.stats.denied.saturating_add(1);
                    sv.reply(DeviceResponse::Refused(DeviceRefusal::Denied));
                }
            },
            DeviceRequest::Claim(selector) => {
                let record = match self.find(&selector) {
                    Ok(hit) => hit,
                    Err(refusal) => {
                        match refusal {
                            DeviceRefusal::NoMatch => self.stats.unmatched += 1,
                            DeviceRefusal::Taken => self.stats.contended += 1,
                            // `find` never speaks the face vocabulary; a
                            // claim refused for any other reason is the
                            // kernel's own gate saying no.
                            DeviceRefusal::Denied
                            | DeviceRefusal::NotHeld
                            | DeviceRefusal::NoSupply
                            | DeviceRefusal::BadFace
                            | DeviceRefusal::BadConvey
                            | DeviceRefusal::Busy => {
                                self.stats.denied = self.stats.denied.saturating_add(1)
                            }
                        }
                        sv.reply(DeviceResponse::Refused(refusal));
                        return;
                    }
                };
                // Record the claim BEFORE emitting the grant. The ledger
                // is what makes "one device, one owner" true, and a
                // successful grant this service then failed to record
                // would let the next claimant be handed the same
                // controller — two drivers programming one device's
                // registers. Refusing a device that was never handed out
                // is recoverable; handing one out twice is not.
                // The grant subject is the MATCHED store's token — the
                // claim derives below the root (or bus) that actually owns
                // the function, so its lineage revoke is per-root (#572).
                let Some(subject) = self.store_slot(record.species) else {
                    self.stats.denied = self.stats.denied.saturating_add(1);
                    sv.reply(DeviceResponse::Refused(DeviceRefusal::Denied));
                    return;
                };
                if self
                    .claims
                    .push(Claim {
                        index: record.index,
                        species: record.species,
                        owner: sv.via(),
                        identity: None,
                    })
                    .is_err()
                {
                    self.stats.denied += 1;
                    sv.reply(DeviceResponse::Refused(DeviceRefusal::Denied));
                    return;
                }
                // The device's own authority: use and program it, derive its
                // BARs/routes/IOVA windows, and retire this delivered child
                // when the affine token dies. `REVOKE` reaches downward from
                // this child; it cannot retire the broker's store ancestor.
                sv.grant(
                    subject,
                    DEVICE_GRANT_RIGHTS,
                    GrantArg::Device {
                        index: record.index,
                        species: record.species,
                    },
                );
                self.stats.granted += 1;
                let (vendor, device) = match record.species {
                    DeviceSpecies::Pci => (record.vendor, record.device),
                    DeviceSpecies::Platform => (0, 0),
                };
                sv.reply(DeviceResponse::Granted {
                    index: record.index,
                    species: record.species,
                    diag: record.diag,
                    revision: record.revision,
                    vendor,
                    device,
                });
            }
            DeviceRequest::CloseClaim { index, grant } => {
                let Some(identity) =
                    Self::owned_claim_identity(&self.claims, index, sv.via(), grant)
                else {
                    sv.reply(DeviceResponse::Refused(DeviceRefusal::Denied));
                    return;
                };
                let response = match verbs::close_device_claim(&self.root, index, identity) {
                    Ok(report) => DeviceResponse::ClaimClosed(report),
                    Err(_) => DeviceResponse::Refused(DeviceRefusal::Denied),
                };
                // Even Drained keeps the ledger and backing reserved. Loss of
                // this reply permits only an exact retry of the same grant.
                sv.reply(response);
            }
            DeviceRequest::QuiesceClaim { index, grant } => {
                let Some(identity) =
                    Self::owned_claim_identity(&self.claims, index, sv.via(), grant)
                else {
                    sv.reply(DeviceResponse::Refused(DeviceRefusal::Denied));
                    return;
                };
                let response = match verbs::quiesce_device_claim(&self.root, index, identity) {
                    Ok(report) => DeviceResponse::ClaimQuiesced(report),
                    Err(verbs::InventoryReadError::Refused(
                        kore_abi::session::Status::WouldBlock,
                    )) => DeviceResponse::Refused(DeviceRefusal::Busy),
                    Err(_) => DeviceResponse::Refused(DeviceRefusal::Denied),
                };
                // Even Drained keeps the ledger and backing reserved. Loss of
                // this reply permits only an exact retry of the same grant.
                sv.reply(response);
            }
            DeviceRequest::RetireClaimDma { index, grant } => {
                let Some(identity) =
                    Self::owned_claim_identity(&self.claims, index, sv.via(), grant)
                else {
                    sv.reply(DeviceResponse::Refused(DeviceRefusal::Denied));
                    return;
                };
                let response = match verbs::retire_device_claim_dma(&self.root, index, identity) {
                    Ok(report) => DeviceResponse::ClaimDmaRetired(report),
                    Err(verbs::InventoryReadError::Refused(
                        kore_abi::session::Status::WouldBlock,
                    )) => DeviceResponse::Refused(DeviceRefusal::Busy),
                    Err(_) => DeviceResponse::Refused(DeviceRefusal::Denied),
                };
                // Even Drained keeps the ledger and backing reserved. Loss of
                // this reply permits only an exact retry of the same grant.
                sv.reply(response);
            }
            DeviceRequest::Release { index } => {
                let refusal = Self::unproven_release_refusal(&self.claims, index);
                if refusal == DeviceRefusal::Denied {
                    self.stats.denied = self.stats.denied.saturating_add(1);
                }
                sv.reply(DeviceResponse::Refused(refusal));
            }
            DeviceRequest::MintFace { face } => {
                if let Err(refusal) = Self::face_open_check(
                    self.face_window.is_some(),
                    &self.faces,
                    self.minting,
                    face,
                ) {
                    sv.reply(DeviceResponse::Refused(refusal));
                    return;
                }
                self.faces[face as usize] = FaceState::Reserved;
                self.minting = Some(face);
                sv.reply(DeviceResponse::Provisioning { face });
            }
            DeviceRequest::Provision => {
                // The host synthesizes this step only after a convey landed
                // on a landing THIS service offered, and an offer exists
                // only while a window is open — so the arm below is the
                // defensive shape, not a reachable protocol state.
                let Some(face) = self.minting else {
                    sv.reply(DeviceResponse::Refused(DeviceRefusal::BadConvey));
                    return;
                };
                let (grant_slot, kick_slot) = Self::face_leg_slots(face);
                match self.faces.get(face as usize).copied() {
                    // The grant lane landed.
                    Some(FaceState::Reserved) => {
                        if !verbs::slot_is_chan_send(grant_slot, FACE_LEG_RIGHTS) {
                            self.unwind_face(face);
                            sv.reply(DeviceResponse::Refused(DeviceRefusal::BadConvey));
                            return;
                        }
                        self.faces[face as usize] = FaceState::Granted;
                        sv.reply(DeviceResponse::Banked { face });
                    }
                    // The reply kick landed — complete the face.
                    Some(FaceState::Granted) => {
                        if !verbs::slot_is_chan_send(kick_slot, FACE_LEG_RIGHTS) {
                            self.unwind_face(face);
                            sv.reply(DeviceResponse::Refused(DeviceRefusal::BadConvey));
                            return;
                        }
                        let Some(win) = self.face_window else {
                            self.unwind_face(face);
                            sv.reply(DeviceResponse::Refused(DeviceRefusal::NoSupply));
                            return;
                        };
                        // Format face `k`'s ring stride through the
                        // whole-band view: zero, then pair-format — the
                        // image face table's recycled-slot hygiene.
                        // SAFETY: the view is live for the task's life
                        // (mapped once at start, length gated ≥
                        // FACE_SUPPLY_BYTES there); this face is not Live,
                        // so no attached pair aliases this stride
                        // (occupancy is the exclusion discipline); the
                        // `&mut` ends before the spec is exposed.
                        let stride = FACE_RING_STRIDE as usize;
                        let pair_bytes = kore_rt::svc::window_bytes::<Self>(
                            kore_rt::svc::REQ_CAP,
                            kore_rt::svc::REP_CAP,
                        );
                        let base = unsafe { win.as_ptr().add(face as usize * stride) };
                        let region = unsafe { core::slice::from_raw_parts_mut(base, stride) };
                        region.fill(0);
                        if kore_rt::svc::init_window::<Self>(
                            &mut region[..pair_bytes],
                            kore_rt::svc::REQ_CAP,
                            kore_rt::svc::REP_CAP,
                        )
                        .is_err()
                        {
                            self.unwind_face(face);
                            sv.reply(DeviceResponse::Refused(DeviceRefusal::Denied));
                            return;
                        }
                        let mut grant_tx = [None; kore_rt::service::MAX_GRANTS];
                        grant_tx[0] = Some(grant_slot);
                        let via = FACE_VIA_BASE + face as u16;
                        // SAFETY: `region` above proved the mapping live
                        // and in-bounds for this stride; the Live claim
                        // below makes it exclusive to this pair until an
                        // unwind detaches it; the view outlives the host
                        // (task-life map); `init_window` just formatted it.
                        let spec = unsafe {
                            MintedPairSpec::new(
                                core::ptr::NonNull::new_unchecked(base),
                                pair_bytes,
                                grant_tx,
                                via,
                                Some(kick_slot),
                            )
                        };
                        self.pending_attach = Some(spec);
                        self.faces[face as usize] = FaceState::Live;
                        self.minting = None;
                        self.stats.faces = self.stats.faces.saturating_add(1);
                        sv.reply(DeviceResponse::Minted { via });
                    }
                    _ => {
                        self.unwind_face(face);
                        sv.reply(DeviceResponse::Refused(DeviceRefusal::BadConvey));
                    }
                }
            }
        }
    }

    fn pending_return_post(&self) -> Option<SlotRef> {
        let face = self.minting?;
        let (grant, kick) = Self::face_leg_slots(face);
        match self.faces.get(face as usize)? {
            FaceState::Reserved => Some(grant),
            FaceState::Granted => Some(kick),
            _ => None,
        }
    }

    fn pending_pair_attach(&mut self) -> Option<MintedPairSpec> {
        self.pending_attach.take()
    }

    fn pair_attached(&mut self, via: u16, attached: bool) {
        if attached {
            return;
        }
        // A refused attach on a face this service minted: unwind it whole
        // — legs retired, slot vacant — so a later `MintFace` can rebuild
        // it instead of leasing against a dead spec.
        let Some(face) = via.checked_sub(FACE_VIA_BASE).map(u32::from) else {
            return;
        };
        if self.faces.get(face as usize) == Some(&FaceState::Live) {
            self.unwind_face(face);
            self.stats.faces = self.stats.faces.saturating_sub(1);
        }
    }

    fn grants_committed(&mut self, response: &DeviceResponse, emitted: usize, via: u16) {
        debug_assert_eq!(emitted, 1, "a device Claim emits exactly one grant");
        let DeviceResponse::Granted {
            index,
            species: DeviceSpecies::Pci,
            ..
        } = *response
        else {
            return;
        };
        // Observation failure cannot turn a delivered grant into a refusal or
        // clear its reservation. A future retire must refuse missing identity.
        if let Ok(identity) = verbs::device_claim(&self.root, index) {
            Self::retain_claim_identity(&mut self.claims, index, via, identity);
        }
    }

    fn grants_dropped(&mut self, response: &mut DeviceResponse, emitted: usize) {
        debug_assert_eq!(emitted, 1, "a device Claim emits exactly one grant");
        Self::rollback_dropped_grant(&mut self.claims, &mut self.stats, response);
    }
}

/// The service-arena slot a claimed device is delivered into is the
/// claimant's business, not the broker's — this alias exists only so the
/// client seam and the driver host name the same thing.
pub type ClaimSlot = SlotRef;

#[cfg(test)]
mod face_tests {
    use super::*;

    /// The mint admission refuses everything but a vacant slot on a
    /// supply-founded broker with no window already open — each arm a
    /// distinct wiring fault the founding parent must see loudly.
    #[test]
    fn the_mint_admission_matrix_is_fail_closed() {
        let vacant = [FaceState::Vacant; FACE_COUNT];
        assert_eq!(
            DeviceBroker::face_open_check(true, &vacant, None, 0),
            Ok(())
        );
        assert_eq!(
            DeviceBroker::face_open_check(true, &vacant, None, FACE_COUNT as u32 - 1),
            Ok(())
        );
        // No supply beats every other refusal: the broker cannot mint at
        // all, and saying "bad face" would send the founder hunting the
        // wrong bug.
        assert_eq!(
            DeviceBroker::face_open_check(false, &vacant, None, 0),
            Err(DeviceRefusal::NoSupply)
        );
        // Out of range.
        assert_eq!(
            DeviceBroker::face_open_check(true, &vacant, None, FACE_COUNT as u32),
            Err(DeviceRefusal::BadFace)
        );
        // A window already open — mints are lockstep.
        assert_eq!(
            DeviceBroker::face_open_check(true, &vacant, Some(1), 0),
            Err(DeviceRefusal::BadFace)
        );
        // Occupied, in any non-vacant phase.
        for state in [FaceState::Reserved, FaceState::Granted, FaceState::Live] {
            let mut faces = vacant;
            faces[3] = state;
            assert_eq!(
                DeviceBroker::face_open_check(true, &faces, None, 3),
                Err(DeviceRefusal::BadFace)
            );
        }
    }

    /// The leg landing slots are pure arithmetic over the claimed band —
    /// derivable from the face index alone, disjoint across faces, and
    /// inside the claimed interval (the slot a live `MintedPairSpec`
    /// cites must be one the band actually owns).
    #[test]
    fn the_leg_slot_map_is_disjoint_and_in_band() {
        let mut seen = [false; 2 * FACE_COUNT];
        for face in 0..FACE_COUNT as u32 {
            let (grant, kick) = DeviceBroker::face_leg_slots(face);
            for slot in [grant, kick] {
                let index = slot.index();
                assert!(index >= FACE_LAND_BASE && index < FACE_LAND_END);
                let offset = (index - FACE_LAND_BASE) as usize;
                assert!(!seen[offset], "two faces share landing slot {index}");
                seen[offset] = true;
            }
        }
        assert!(seen.iter().all(|s| *s), "the band has no unused slot");
    }

    /// Minted vias start past every fixed pair and stay unique per face —
    /// the host's collision rule, pinned where the numbers are chosen.
    #[test]
    fn minted_vias_clear_the_fixed_pairs() {
        assert!(FACE_VIA_BASE as usize >= kore_rt::svc::MAX_SERVE_PAIRS);
        for face in 0..FACE_COUNT as u16 {
            let via = FACE_VIA_BASE + face;
            assert!((via as usize) >= kore_rt::svc::MAX_SERVE_PAIRS);
            assert_eq!(via - FACE_VIA_BASE, face);
        }
    }

    /// The band geometry quotient: the face count times the stride is
    /// exactly the supply span the founder mints (`FACE_SUPPLY_BYTES`),
    /// and the landing band holds exactly two slots per face.
    #[test]
    fn the_face_geometry_is_the_contracts_quotient() {
        assert_eq!(FACE_COUNT as u64 * FACE_RING_STRIDE, FACE_SUPPLY_BYTES);
        assert_eq!(FACE_LAND_END - FACE_LAND_BASE, 2 * FACE_COUNT as u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kore_rt::service::{PciBarMask, PciIdentitySelector, PciSelector};

    #[test]
    fn global_i2c_build_page_never_queries_or_marks_a_candidate_claim() {
        use kore_abi::device::{
            AmlBuildDiagnostic, AmlBuildErrorClass, AmlBuildStage, I2C_ADMISSION_BUILD_ROW,
            I2cAdmissionPage,
        };
        let page = I2cAdmissionPage {
            row: I2C_ADMISSION_BUILD_ROW,
            words: AmlBuildDiagnostic::failure(
                AmlBuildStage::SsdtLoad,
                AmlBuildErrorClass::Opcode,
                u32::MAX,
                0x5bff,
            )
            .unwrap()
            .words(),
        };
        let mut reads = 0;
        let observed = DeviceBroker::inspect_i2c_with(
            page.row,
            |row| {
                assert_eq!(row, I2C_ADMISSION_BUILD_ROW);
                reads += 1;
                Ok(page)
            },
            |_| panic!("a firmware failure has no claim-ledger identity"),
        );
        assert_eq!(observed, Ok(page));
        assert_eq!(reads, 1);
    }

    #[test]
    fn last_i2c_candidate_still_uses_its_own_inventory_index_for_claim_state() {
        use kore_abi::device::{I2C_ADMISSION_BUILD_ROW, I2C_ADMISSION_CLAIMED, I2cAdmissionPage};
        let row = I2C_ADMISSION_BUILD_ROW - 3;
        let mut reads = 0;
        let observed = DeviceBroker::inspect_i2c_with(
            row,
            |requested| {
                let page = match reads {
                    0 => {
                        assert_eq!(requested, row);
                        I2cAdmissionPage {
                            row,
                            words: [0x00c9_0000, 0],
                        }
                    }
                    1 => {
                        assert_eq!(requested, row + 2);
                        I2cAdmissionPage {
                            row: row + 2,
                            words: [19, 0xa851_8086],
                        }
                    }
                    _ => panic!("only candidate and identity reads are needed"),
                };
                reads += 1;
                Ok(page)
            },
            |index| {
                assert_eq!(index, 19);
                true
            },
        )
        .unwrap();
        assert_eq!(reads, 2);
        assert_eq!(observed.words[1], u32::from(I2C_ADMISSION_CLAIMED));
    }

    fn pci(class: u8, subclass: u8, progif: u8, flags: u8, index: u32) -> DeviceRecord {
        DeviceRecord {
            species: DeviceSpecies::Pci,
            index,
            vendor: 0x8086,
            device: 0x9a13,
            class,
            subclass,
            progif,
            locator: 0xa0,
            segment: 0,
            flags,
            msix_table_size: 0,
            diag: if flags & DeviceRecord::HAS_BAR0 != 0 {
                (1 << 0) | (1 << 6)
            } else {
                0
            },
            revision: 0,
        }
    }

    fn xhci_selector(dma: bool, msi: bool) -> DeviceSelector {
        DeviceSelector::Pci(PciSelector {
            class: 0x0c,
            subclass: 0x03,
            progif: 0x30,
            identity: PciIdentitySelector::ANY,
            bars: PciBarMask::BAR0,
            dma,
            interrupts: if msi {
                PciInterruptRequirement::MsiFamily
            } else {
                PciInterruptRequirement::None
            },
        })
    }

    fn pci_i2c(flags: u8, index: u32) -> DeviceRecord {
        let mut record = pci(0x0c, 0x80, 0, flags, index);
        record.vendor = 0x8086;
        record.device = 0xa878;
        record.diag |= DeviceRecord::PCI_DIAG_I2C_HID;
        record
    }

    const LIVE: u8 = DeviceRecord::DRIVER_ENDPOINT
        | DeviceRecord::HAS_BAR0
        | DeviceRecord::HAS_BINDING
        | DeviceRecord::HAS_REMAPPED_MSI;

    /// The class triple must match on all three bytes — a selector that
    /// matched on class alone would hand a USB driver an IDE controller.
    #[test]
    fn the_whole_class_triple_is_load_bearing() {
        let sel = xhci_selector(true, true);
        assert!(DeviceBroker::matches(&pci(0x0c, 0x03, 0x30, LIVE, 0), &sel));
        for wrong in [
            pci(0x01, 0x03, 0x30, LIVE, 0),
            pci(0x0c, 0x06, 0x30, LIVE, 0),
            pci(0x0c, 0x03, 0x20, LIVE, 0),
        ] {
            assert!(
                !DeviceBroker::matches(&wrong, &sel),
                "a near-miss on any of the three bytes is not a match"
            );
        }
    }

    /// A product-specific driver must reject a class-compatible function it
    /// cannot program. This comparison happens before the affine grant leaves
    /// the broker; rejecting it after claim would strand the ledger because
    /// index-only release is intentionally unproven.
    #[test]
    fn exact_pci_identity_filters_before_claim() {
        let mut exact = xhci_selector(true, true);
        let DeviceSelector::Pci(selector) = &mut exact else {
            unreachable!();
        };
        selector.identity = PciIdentitySelector::exact(0x8086, 0x9a13);

        assert!(DeviceBroker::matches(
            &pci(0x0c, 0x03, 0x30, LIVE, 0),
            &exact,
        ));
        let mut wrong_vendor = pci(0x0c, 0x03, 0x30, LIVE, 0);
        wrong_vendor.vendor = 0x10de;
        assert!(!DeviceBroker::matches(&wrong_vendor, &exact));
        let mut wrong_device = pci(0x0c, 0x03, 0x30, LIVE, 0);
        wrong_device.device ^= 1;
        assert!(!DeviceBroker::matches(&wrong_device, &exact));

        assert!(
            DeviceBroker::matches(&wrong_vendor, &xhci_selector(true, true)),
            "explicit ANY preserves class-driver matching"
        );
    }

    #[test]
    fn inspection_reports_the_identity_even_when_claim_requirements_fail() {
        let mut selector = xhci_selector(true, true);
        let DeviceSelector::Pci(pci_selector) = &mut selector else {
            unreachable!();
        };
        pci_selector.identity = PciIdentitySelector::exact(0x8086, 0x9a13);

        let wrong = pci(0x0c, 0x03, 0x30, LIVE, 0);
        let mut withheld = pci(0x0c, 0x03, 0x30, DeviceRecord::NOT_DRIVER_ENDPOINT, 1);
        withheld.device = 0x9a13;
        let records = [
            DeviceRecord {
                device: 0xffff,
                ..wrong
            },
            withheld,
        ];

        assert!(!DeviceBroker::matches(&withheld, &selector));
        let observed = DeviceBroker::inspect_with(records.len() as u32, &selector, |index| {
            Ok::<_, ()>(records[index as usize])
        })
        .expect("identity-only inspection sees the withheld record");
        assert_eq!(observed.index, 1);
        assert_eq!(observed.flags, DeviceRecord::NOT_DRIVER_ENDPOINT);
    }

    /// A driver that DMAs demands a live IOMMU binding, one that parks on
    /// interrupts demands a remapped MSI-family route, and **every** PCI
    /// claimant demands a usable BAR0.
    ///
    /// The BAR0 requirement is unconditional on purpose: it is not in the
    /// selector, so nothing a claimant writes can opt out of it, and a
    /// device with no user-mappable registers is one no userland driver
    /// can drive at all.
    #[test]
    fn the_required_facts_are_demanded_not_hoped_for() {
        let sel = xhci_selector(true, true);
        assert!(!DeviceBroker::matches(
            &pci(0x0c, 0x03, 0x30, LIVE & !DeviceRecord::HAS_BINDING, 0),
            &sel,
        ));
        assert!(!DeviceBroker::matches(
            &pci(0x0c, 0x03, 0x30, LIVE & !DeviceRecord::HAS_REMAPPED_MSI, 0),
            &sel,
        ));
        // A selector that asks for neither still needs BAR0.
        let bare = xhci_selector(false, false);
        assert!(DeviceBroker::matches(
            &pci(
                0x0c,
                0x03,
                0x30,
                DeviceRecord::DRIVER_ENDPOINT | DeviceRecord::HAS_BAR0,
                0,
            ),
            &bare,
        ));
        assert!(
            !DeviceBroker::matches(&pci(0x0c, 0x03, 0x30, 0, 0), &bare),
            "no user-mappable BAR0 means no userland driver, whatever the selector says"
        );
    }

    /// A PCI-to-PCI bridge is never anybody's device, even when it
    /// class-matches and carries every flag.
    #[test]
    fn a_bridge_is_never_a_drivers_device() {
        assert!(!DeviceBroker::matches(
            &pci(0x0c, 0x03, 0x30, LIVE | DeviceRecord::IS_BRIDGE, 0),
            &xhci_selector(true, true),
        ));
    }

    #[test]
    fn endpoint_admission_is_positive_and_msix_pair_is_exact() {
        let mut xe = xhci_selector(true, true);
        let DeviceSelector::Pci(selector) = &mut xe else {
            unreachable!();
        };
        selector.class = 0x03;
        selector.subclass = 0x00;
        selector.progif = 0x00;
        selector.identity = PciIdentitySelector::exact(0x8086, 0x56a0);
        selector.interrupts =
            PciInterruptRequirement::Msix(kore_rt::service::PciMsixBlock::ENTRIES_0_1);

        let strict = LIVE | DeviceRecord::HAS_REMAPPED_MSIX;
        let dg2 = |flags| {
            let mut record = pci(0x03, 0x00, 0x00, flags, 0);
            record.device = 0x56a0;
            record.msix_table_size = if flags & DeviceRecord::HAS_REMAPPED_MSIX != 0 {
                2
            } else {
                0
            };
            record
        };
        assert!(DeviceBroker::matches(&dg2(strict), &xe));
        assert!(
            !DeviceBroker::matches(&dg2(LIVE), &xe),
            "ordinary MSI-family eligibility cannot satisfy a strict MSI-X-pair owner"
        );
        assert!(
            !DeviceBroker::matches(
                &dg2(strict & !DeviceRecord::DRIVER_ENDPOINT | DeviceRecord::NOT_DRIVER_ENDPOINT,),
                &xe,
            ),
            "a matching DG2 identity is not proof that firmware released it"
        );
        assert!(
            !DeviceBroker::matches(&dg2(strict | DeviceRecord::NOT_DRIVER_ENDPOINT), &xe),
            "contradictory positive and negative endpoint facts fail closed"
        );

        let mut one_entry = dg2(strict);
        one_entry.msix_table_size = 1;
        let mut second = dg2(strict);
        second.index = 1;
        assert!(
            !DeviceBroker::matches(&one_entry, &xe),
            "a strict two-entry owner cannot consume a one-entry table"
        );
        let records = [one_entry, second];
        let selected = DeviceBroker::scan_with(
            records.len() as u32,
            |_| false,
            &xe,
            |index| Ok::<_, ()>(records[index as usize]),
        )
        .expect("the one-entry record is skipped before the pair-capable row is selected");
        assert_eq!(selected.index, 1);
    }

    #[test]
    fn exact_bar_mask_filters_before_the_claim_ledger() {
        let mut ivpu = xhci_selector(true, true);
        let DeviceSelector::Pci(selector) = &mut ivpu else {
            unreachable!();
        };
        selector.class = 0x12;
        selector.subclass = 0;
        selector.progif = 0;
        selector.bars = PciBarMask::BAR0.union(PciBarMask::BAR4);

        let bar0_only = pci(0x12, 0, 0, LIVE, 0);
        let mut complete = pci(0x12, 0, 0, LIVE, 1);
        complete.diag |= (1 << 4) | (1 << (6 + 4));
        assert!(!DeviceBroker::matches(&bar0_only, &ivpu));
        assert!(DeviceBroker::matches(&complete, &ivpu));

        let records = [bar0_only, complete];
        let selected = DeviceBroker::scan_with(
            records.len() as u32,
            |_| false,
            &ivpu,
            |index| Ok::<_, ()>(records[index as usize]),
        )
        .expect("the incomplete BAR geometry is rejected before selection");
        assert_eq!(selected.index, 1);
    }

    #[test]
    fn impossible_inventory_geometry_fails_closed() {
        let selector = xhci_selector(true, true);

        let mut oversized_msix = pci(0x0c, 0x03, 0x30, LIVE, 0);
        oversized_msix.msix_table_size = DeviceRecord::MAX_MSIX_TABLE_ENTRIES + 1;
        assert!(!DeviceBroker::matches(&oversized_msix, &selector));

        let mut user_bar_without_present_bar = pci(0x0c, 0x03, 0x30, LIVE, 0);
        user_bar_without_present_bar.diag &= !(1 << 0);
        assert!(!DeviceBroker::matches(
            &user_bar_without_present_bar,
            &selector
        ));

        let mut stale_bar0_flag = pci(0x0c, 0x03, 0x30, LIVE, 0);
        stale_bar0_flag.diag &= !(1 << 6);
        assert!(!DeviceBroker::matches(&stale_bar0_flag, &selector));

        let mut orphan_msix_fact = pci(0x0c, 0x03, 0x30, LIVE | DeviceRecord::HAS_REMAPPED_MSIX, 0);
        assert!(!DeviceBroker::matches(&orphan_msix_fact, &selector));
        orphan_msix_fact.msix_table_size = 1;
        orphan_msix_fact.flags &= !DeviceRecord::HAS_REMAPPED_MSI;
        assert!(!DeviceBroker::matches(&orphan_msix_fact, &selector));
    }

    /// The two species never cross: a PCI selector does not match a
    /// firmware-described entry, and a platform selector does not match a
    /// PCI function that happens to share its id halves.
    #[test]
    fn the_species_never_cross() {
        let mut plat = pci(0, 0, 0, DeviceRecord::HAS_LINE, 3);
        plat.species = DeviceSpecies::Platform;
        // `vendor | device << 16` — PNP0303, the i8042 keyboard id.
        plat.vendor = 0xd041;
        plat.device = 0x0303;

        let by_hid = DeviceSelector::Platform {
            hid: 0x0303_d041,
            irq: true,
            dma: false,
        };
        assert!(DeviceBroker::matches(&plat, &by_hid));
        assert!(!DeviceBroker::matches(&plat, &xhci_selector(true, true)));

        let mut pci_twin = plat;
        pci_twin.species = DeviceSpecies::Pci;
        assert!(
            !DeviceBroker::matches(&pci_twin, &by_hid),
            "a PCI function with the same id halves is not a platform device"
        );
    }

    /// A platform selector that demands an interrupt line refuses a device
    /// firmware described without one — the row would otherwise found a
    /// driver that can only poll, and this arc ships no polled mode.
    #[test]
    fn a_line_less_platform_device_refuses_a_row_that_needs_one() {
        let mut plat = pci(0, 0, 0, 0, 3);
        plat.species = DeviceSpecies::Platform;
        plat.vendor = 0xd041;
        plat.device = 0x0303;
        assert!(!DeviceBroker::matches(
            &plat,
            &DeviceSelector::Platform {
                hid: 0x0303_d041,
                irq: true,
                dma: false,
            },
        ));
        assert!(DeviceBroker::matches(
            &plat,
            &DeviceSelector::Platform {
                hid: 0x0303_d041,
                irq: false,
                dma: false,
            },
        ));
    }

    #[test]
    fn a_dma_platform_selector_requires_a_live_binding() {
        let mut plat = pci(0, 0, 0, DeviceRecord::HAS_LINE, 3);
        plat.species = DeviceSpecies::Platform;
        plat.vendor = 0xd041;
        plat.device = 0x0d15;
        let selector = DeviceSelector::Platform {
            hid: 0x0d15_d041,
            irq: true,
            dma: true,
        };
        assert!(!DeviceBroker::matches(&plat, &selector));
        plat.flags |= DeviceRecord::HAS_BINDING;
        assert!(DeviceBroker::matches(&plat, &selector));
    }

    #[test]
    fn topology_selected_i2c_never_aliases_an_acpi_id_row() {
        let mut i2c = pci(0, 0, 0, DeviceRecord::HAS_LINE, 4);
        i2c.species = DeviceSpecies::Platform;
        i2c.class = DeviceRecord::PLATFORM_CLASS_I2C_HID;

        let selector = DeviceSelector::PlatformI2cHid { irq: true };
        assert!(DeviceBroker::matches(&i2c, &selector));

        let mut legacy = i2c;
        legacy.class = DeviceRecord::PLATFORM_CLASS_HID;
        legacy.vendor = 0xd041;
        legacy.device = 0x0303;
        assert!(!DeviceBroker::matches(&legacy, &selector));
        assert!(DeviceBroker::matches(
            &legacy,
            &DeviceSelector::Platform {
                hid: 0x0303_d041,
                irq: true,
                dma: false,
            },
        ));

        i2c.flags &= !DeviceRecord::HAS_LINE;
        assert!(!DeviceBroker::matches(&i2c, &selector));
        assert!(DeviceBroker::matches(
            &i2c,
            &DeviceSelector::PlatformI2cHid { irq: false },
        ));
    }

    #[test]
    fn pci_i2c_selection_requires_correlation_for_the_exact_function() {
        let selector = DeviceSelector::PciI2cHid { irq: false };
        let pio = DeviceRecord::DRIVER_ENDPOINT | DeviceRecord::HAS_BAR0;
        let mut unrelated = pci(0x0c, 0x80, 0, pio, 0);
        // Other nonzero diagnostic bits (including the PM sample) are not
        // initialized firmware correlation for this function.
        unrelated.diag |= 0x7800_0000;
        unrelated.locator = 0x78;
        let mut withheld = pci_i2c(DeviceRecord::NOT_DRIVER_ENDPOINT, 1);
        withheld.locator = 0x79;
        let mut available = pci_i2c(pio, 2);
        available.locator = 0x7a;
        let records = [unrelated, withheld, available];

        assert!(!DeviceBroker::describes(&unrelated, &selector));
        assert!(!DeviceBroker::matches(&withheld, &selector));
        assert!(DeviceBroker::matches(&available, &selector));
        let selected = DeviceBroker::scan_with(
            records.len() as u32,
            |_| false,
            &selector,
            |index| Ok::<_, ()>(records[index as usize]),
        )
        .expect("a correlated PIO endpoint needs neither DMA nor MSI");
        assert_eq!(selected.index, available.index);

        let observed = DeviceBroker::inspect_with(records.len() as u32, &selector, |index| {
            Ok::<_, ()>(records[index as usize])
        })
        .expect("inspection reports the first correlated function even when withheld");
        assert_eq!(observed.index, withheld.index);
        assert_eq!(observed.flags, DeviceRecord::NOT_DRIVER_ENDPOINT);

        assert_eq!(
            DeviceBroker::scan_with(
                records.len() as u32,
                |index| index == available.index,
                &selector,
                |index| Ok::<_, ()>(records[index as usize]),
            ),
            Err(DeviceRefusal::Taken),
            "an unclaimed but uncorrelated sibling is never a fallback",
        );
    }

    #[test]
    fn an_uncorrelated_supported_i2c_controller_is_observable_but_not_claimable() {
        let mut record = pci_i2c(
            DeviceRecord::DRIVER_ENDPOINT | DeviceRecord::HAS_BAR0 | DeviceRecord::HAS_LINE,
            0,
        );
        record.diag &= !(1 << 31);
        for irq in [false, true] {
            let selector = DeviceSelector::PciI2cHid { irq };
            assert!(DeviceBroker::describes(&record, &selector));
            assert!(!DeviceBroker::matches(&record, &selector));
            assert_eq!(
                DeviceBroker::inspect_with(1, &selector, |_| Ok::<_, ()>(record))
                    .unwrap()
                    .index,
                record.index,
            );
            assert_eq!(
                DeviceBroker::scan_with(1, |_| false, &selector, |_| Ok::<_, ()>(record)),
                Err(DeviceRefusal::NoMatch),
            );
        }
    }

    #[test]
    fn pci_i2c_line_requirement_uses_the_controller_line_fact() {
        let mut record = pci_i2c(
            DeviceRecord::DRIVER_ENDPOINT | DeviceRecord::HAS_BAR0 | DeviceRecord::HAS_REMAPPED_MSI,
            0,
        );
        let optional = DeviceSelector::PciI2cHid { irq: false };
        let required = DeviceSelector::PciI2cHid { irq: true };
        assert!(DeviceBroker::matches(&record, &optional));
        assert!(
            !DeviceBroker::matches(&record, &required),
            "MSI eligibility cannot substitute for a firmware controller line",
        );
        record.flags &= !DeviceRecord::HAS_REMAPPED_MSI;
        record.flags |= DeviceRecord::HAS_LINE;
        assert!(DeviceBroker::matches(&record, &required));

        let mut platform = record;
        platform.species = DeviceSpecies::Platform;
        assert!(!DeviceBroker::matches(&platform, &optional));
        platform.class = DeviceRecord::PLATFORM_CLASS_I2C_HID;
        platform.subclass = 0;
        platform.diag = DeviceRecord::PLATFORM_I2C_DESIGNWARE;
        assert!(DeviceBroker::matches(
            &platform,
            &DeviceSelector::PlatformI2cHid { irq: true },
        ));
        assert!(!DeviceBroker::matches(&platform, &required));
    }

    #[test]
    fn pci_i2c_correlation_does_not_bypass_endpoint_or_bar_admission() {
        let selector = DeviceSelector::PciI2cHid { irq: false };
        let base = pci_i2c(DeviceRecord::DRIVER_ENDPOINT | DeviceRecord::HAS_BAR0, 0);
        for corrupt in [
            |record: &mut DeviceRecord| record.flags &= !DeviceRecord::DRIVER_ENDPOINT,
            |record: &mut DeviceRecord| record.flags |= DeviceRecord::NOT_DRIVER_ENDPOINT,
            |record: &mut DeviceRecord| record.flags |= DeviceRecord::IS_BRIDGE,
            |record: &mut DeviceRecord| record.flags &= !DeviceRecord::HAS_BAR0,
            |record: &mut DeviceRecord| record.diag &= !(1 << 6),
            |record: &mut DeviceRecord| record.diag &= !(1 << 0),
            |record: &mut DeviceRecord| record.diag = DeviceRecord::PCI_DIAG_I2C_HID,
            |record: &mut DeviceRecord| record.class ^= 1,
            |record: &mut DeviceRecord| record.subclass ^= 1,
            |record: &mut DeviceRecord| record.progif = 1,
            |record: &mut DeviceRecord| {
                record.msix_table_size = DeviceRecord::MAX_MSIX_TABLE_ENTRIES + 1;
            },
        ] {
            let mut record = base;
            corrupt(&mut record);
            assert!(
                !DeviceBroker::matches(&record, &selector),
                "record {record:?}"
            );
        }
    }

    #[test]
    fn native_i2c_profile_does_not_reset_unknown_controller_wrappers() {
        let selector = DeviceSelector::PciI2cHid { irq: false };
        let mut record = pci_i2c(DeviceRecord::DRIVER_ENDPOINT | DeviceRecord::HAS_BAR0, 0);
        for device in [
            0xa850, 0xa851, 0xa878, 0xa879, 0xa87a, 0xa87b, 0xe450, 0xe451, 0xe478, 0xe479, 0xe47a,
            0xe47b,
        ] {
            record.device = device;
            assert!(DeviceBroker::matches(&record, &selector));
        }
        for (vendor, device) in [
            (0x8086, 0xa852),
            (0x8086, 0x9d60),
            (0x1022, 0xa878),
            (0x8086, 0xe452),
            (0x8086, 0xe425),
            (0x8086, 0xe427),
            (0x8086, 0xe430),
            (0x8086, 0xe446),
            (0x8086, 0xe477),
            (0x8086, 0xe47c),
            (0x1022, 0xe450),
        ] {
            record.vendor = vendor;
            record.device = device;
            assert!(!DeviceBroker::describes(&record, &selector));
            assert!(!DeviceBroker::matches(&record, &selector));
        }
    }

    #[test]
    fn panther_lake_i2c_inspection_preserves_claim_requirements() {
        let optional = DeviceSelector::PciI2cHid { irq: false };
        let routed = DeviceSelector::PciI2cHid { irq: true };
        // Biggertron's two PCI controllers, as captured by Ubuntu. Candidate
        // evidence must survive a refusal; it does not confer claim authority.
        for device in [0xe450, 0xe451] {
            let mut record = pci_i2c(DeviceRecord::DRIVER_ENDPOINT | DeviceRecord::HAS_BAR0, 0);
            record.device = device;
            assert!(record.is_native_i2c_hid_candidate());
            assert!(DeviceBroker::describes(&record, &routed));
            assert!(DeviceBroker::matches(&record, &optional));
            assert!(!DeviceBroker::matches(&record, &routed));
            record.flags |= DeviceRecord::HAS_LINE;
            assert!(DeviceBroker::matches(&record, &routed));
            for corrupt in [
                |r: &mut DeviceRecord| r.flags &= !DeviceRecord::DRIVER_ENDPOINT,
                |r: &mut DeviceRecord| r.flags |= DeviceRecord::NOT_DRIVER_ENDPOINT,
                |r: &mut DeviceRecord| r.flags |= DeviceRecord::IS_BRIDGE,
                |r: &mut DeviceRecord| r.flags &= !DeviceRecord::HAS_BAR0,
                |r: &mut DeviceRecord| r.diag = 0,
            ] {
                let mut refused = record;
                corrupt(&mut refused);
                assert!(DeviceBroker::describes(&refused, &routed));
                assert!(!DeviceBroker::matches(&refused, &optional));
            }
            for corrupt in [
                |r: &mut DeviceRecord| r.species = DeviceSpecies::Platform,
                |r: &mut DeviceRecord| r.class ^= 1,
                |r: &mut DeviceRecord| r.subclass ^= 1,
                |r: &mut DeviceRecord| r.progif = 1,
            ] {
                let mut wrong_kind = record;
                corrupt(&mut wrong_kind);
                assert!(!wrong_kind.is_native_i2c_hid_candidate());
                assert!(!DeviceBroker::describes(&wrong_kind, &optional));
            }
        }
    }

    /// The claim ledger is speculative until the grant session commits.  A
    /// refusal must make the same entry immediately claimable again and turn
    /// the reply into a grant-less denial rather than leaking `Granted`.
    #[test]
    fn a_dropped_grant_rolls_back_its_claim_and_reply() {
        let claim = Claim {
            index: 7,
            species: DeviceSpecies::Pci,
            owner: 1,
            identity: None,
        };
        let mut claims = heapless::Vec::<Claim, MAX_CLAIMS>::new();
        assert!(claims.push(claim).is_ok());
        let mut stats = DeviceStats {
            granted: 1,
            ..DeviceStats::default()
        };
        let mut response = DeviceResponse::Granted {
            index: claim.index,
            species: claim.species,
            diag: 0,
            revision: 0,
            vendor: 0,
            device: 0,
        };

        DeviceBroker::rollback_dropped_grant(&mut claims, &mut stats, &mut response);

        assert!(
            claims.is_empty(),
            "the un-effected device is claimable again"
        );
        assert_eq!(stats.granted, 0, "no device was handed out");
        assert_eq!(stats.denied, 1, "the transport refusal is visible");
        assert_eq!(response, DeviceResponse::Refused(DeviceRefusal::Denied));
    }

    #[test]
    fn committed_correlation_preserves_owner_species_and_first_generation() {
        use kore_abi::{cap::Cid, session::DeviceClaimReport};
        let mut claims = heapless::Vec::<Claim, MAX_CLAIMS>::new();
        for species in [DeviceSpecies::Pci, DeviceSpecies::Platform] {
            claims
                .push(Claim {
                    index: 7,
                    species,
                    owner: 19,
                    identity: None,
                })
                .unwrap();
        }
        let identity = DeviceClaimReport {
            generation: 3,
            grant: Cid(0x1234_0000_0007),
        };
        DeviceBroker::retain_claim_identity(&mut claims, 7, 18, identity);
        DeviceBroker::retain_claim_identity(&mut claims, 8, 19, identity);
        DeviceBroker::retain_claim_identity(
            &mut claims,
            7,
            19,
            DeviceClaimReport {
                generation: 0,
                grant: Cid::NULL,
            },
        );
        assert!(claims.iter().all(|claim| claim.identity.is_none()));
        assert_eq!(claims.len(), 2, "failed observation never undoes delivery");
        DeviceBroker::retain_claim_identity(&mut claims, 7, 19, identity);
        assert_eq!(claims[0].identity, Some(identity));
        assert_eq!(
            claims[1].identity, None,
            "platform index is a different namespace"
        );
        DeviceBroker::retain_claim_identity(
            &mut claims,
            7,
            19,
            DeviceClaimReport {
                generation: 5,
                grant: Cid(9),
            },
        );
        assert_eq!(
            claims[0].identity,
            Some(identity),
            "correlation is never silently replaced"
        );
        assert_eq!(
            DeviceBroker::owned_claim_identity(&claims, 7, 19, identity.grant.0),
            Some(identity)
        );
        assert_eq!(
            DeviceBroker::owned_claim_identity(&claims, 7, 18, identity.grant.0),
            None
        );
        assert_eq!(
            DeviceBroker::owned_claim_identity(&claims, 8, 19, identity.grant.0),
            None
        );
        assert_eq!(
            DeviceBroker::owned_claim_identity(&claims, 7, 19, identity.grant.0 + 1),
            None
        );
        assert_eq!(claims.len(), 2);
        assert_eq!(
            DeviceBroker::unproven_release_refusal(&claims, 7),
            DeviceRefusal::Denied
        );
    }

    /// A returned device is an affine owner, not a borrowed view: its root
    /// must be able to retire itself, and every child verb's requested rights
    /// must be a subset of the root grant under Σ5 monotonicity.
    #[test]
    fn a_device_grant_can_retire_itself_and_derive_owned_children() {
        assert!(DEVICE_GRANT_RIGHTS.contains(Rights::REVOKE));
        assert!(DEVICE_GRANT_RIGHTS.contains(verbs::DEVICE_CHILD_RIGHTS));
        assert!(DEVICE_GRANT_RIGHTS.contains(Rights::DERIVE));
    }

    /// An index is replayable data, not affine authority. Until release carries
    /// proof of revocation, it must leave the reservation intact rather than
    /// make the same hardware claimable beside a still-live sibling grant.
    #[test]
    fn an_index_only_release_never_clears_a_claim() {
        let claim = Claim {
            index: 7,
            species: DeviceSpecies::Pci,
            owner: 1,
            identity: None,
        };
        let mut claims = heapless::Vec::<Claim, MAX_CLAIMS>::new();
        assert!(claims.push(claim).is_ok());

        assert_eq!(
            DeviceBroker::unproven_release_refusal(&claims, claim.index),
            DeviceRefusal::Denied,
        );
        assert_eq!(claims.as_slice(), &[claim], "the ledger remains reserved");
        assert_eq!(
            DeviceBroker::unproven_release_refusal(&claims, 99),
            DeviceRefusal::NotHeld,
        );
    }

    /// A failed first read is infrastructure denial, never evidence that the
    /// machine lacks a matching controller.
    #[test]
    fn a_record_failure_never_becomes_no_match() {
        let claims = heapless::Vec::<Claim, MAX_CLAIMS>::new();
        let result = DeviceBroker::scan_with(
            1,
            |index| claims.iter().any(|c| c.index == index),
            &xhci_selector(true, true),
            |_| Err::<DeviceRecord, _>(()),
        );
        assert_eq!(result, Err(DeviceRefusal::Denied));
    }

    /// A failure after an already-held match is still denial, not proof that
    /// every matching device was taken; a later unclaimed entry may exist.
    #[test]
    fn a_record_failure_never_becomes_taken() {
        let mut claims = heapless::Vec::<Claim, MAX_CLAIMS>::new();
        assert!(
            claims
                .push(Claim {
                    index: 0,
                    species: DeviceSpecies::Pci,
                    owner: 0,
                    identity: None,
                })
                .is_ok()
        );
        let result = DeviceBroker::scan_with(
            2,
            |index| claims.iter().any(|c| c.index == index),
            &xhci_selector(true, true),
            |index| match index {
                0 => Ok(pci(0x0c, 0x03, 0x30, LIVE, 0)),
                _ => Err(()),
            },
        );
        assert_eq!(result, Err(DeviceRefusal::Denied));
    }

    /// `NoMatch` and `Taken` remain meaningful after a complete successful
    /// scan; only incomplete observation loses the right to assert them.
    #[test]
    fn a_complete_scan_preserves_hardware_and_contention_refusals() {
        let empty = heapless::Vec::<Claim, MAX_CLAIMS>::new();
        let no_match = DeviceBroker::scan_with(
            1,
            |index| empty.iter().any(|c| c.index == index),
            &xhci_selector(true, true),
            |_| Ok::<_, ()>(pci(0x01, 0x08, 0x02, LIVE, 0)),
        );
        assert_eq!(no_match, Err(DeviceRefusal::NoMatch));

        let mut held = heapless::Vec::<Claim, MAX_CLAIMS>::new();
        assert!(
            held.push(Claim {
                index: 0,
                species: DeviceSpecies::Pci,
                owner: 0,
                identity: None,
            })
            .is_ok()
        );
        let taken = DeviceBroker::scan_with(
            1,
            |index| held.iter().any(|c| c.index == index),
            &xhci_selector(true, true),
            |_| Ok::<_, ()>(pci(0x0c, 0x03, 0x30, LIVE, 0)),
        );
        assert_eq!(taken, Err(DeviceRefusal::Taken));
    }
}
