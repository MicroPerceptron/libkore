//! Fail-closed codecs over [`kore_abi::svc`]'s frozen device-broker forms
//! (`res/USERLAND.md` §5.3).
//!
//! Every decode is total and refuses anything out of vocabulary: an
//! unknown op, an unknown tag, a refusal byte on a success tag, a grant
//! count that does not match the tag, or an invalid reserved/geometry field. The peer wrote
//! garbage (or a future revision), and in *this* protocol a wrong guess
//! would hand a device to the wrong task — so the codec is deliberately as
//! strict as the power service's rather than as lenient as swap's.

use kore_abi::device::DeviceSpecies;
use kore_abi::svc::{DeviceRepWire, DeviceReqWire};
use kore_rt::service::{
    DeviceSelector, PciBarMask, PciIdentitySelector, PciInterruptRequirement, PciMsixBlock,
    PciSelector,
};
use kore_rt::svc::SvcProtocol;

use crate::service::{
    DeviceBroker, DeviceObservation, DeviceRefusal, DeviceRequest, DeviceResponse,
};

/// A selector's `dma` bit, as it rides [`DeviceReqWire::flags`].
///
/// The wire carries the *requirement* the claimant states, not the record
/// flags it will be matched against — the broker translates. Keeping the
/// two spaces separate is what stops a claimant from asserting a fact
/// about the hardware (`HAS_BINDING`) by writing a bit.
pub const REQ_DMA: u8 = 1 << 0;
/// A PCI selector's remapped MSI-family requirement bit.
pub const REQ_MSI: u8 = 1 << 1;
/// A platform selector's `irq` bit.
pub const REQ_LINE: u8 = 1 << 2;
/// Tighten [`REQ_MSI`] to the exact MSI-X block encoded in `reserved2`. This
/// bit is only valid when `REQ_MSI` is also set, so an older peer sees unknown
/// vocabulary and refuses rather than weakening the request to ordinary MSI.
pub const REQ_MSIX: u8 = 1 << 3;
/// Require initialized firmware HID children behind this PCI controller.
/// Older brokers reject this new bit rather than claiming an arbitrary
/// SerialIO-class function.
pub const REQ_I2C_HID: u8 = 1 << 4;
/// Every defined requirement bit — anything else set is a refusal.
const REQ_ALL: u8 = REQ_DMA | REQ_MSI | REQ_LINE | REQ_MSIX | REQ_I2C_HID;

/// Lower a request to its wire form (total).
pub fn encode_req(req: &DeviceRequest) -> DeviceReqWire {
    let mut w = DeviceReqWire {
        op: 0,
        species: 0,
        class: 0,
        subclass: 0,
        progif: 0,
        flags: 0,
        reserved: [0; 2],
        word: 0.into(),
        reserved2: [0; 4],
    };
    match *req {
        DeviceRequest::CloseClaim { index, grant } => {
            return kore_abi::svc::DeviceClaimCloseWire {
                op: DeviceReqWire::OP_CLOSE_CLAIM,
                reserved: [0; 3],
                index: index.into(),
                grant: grant.into(),
            }
            .encode();
        }
        DeviceRequest::QuiesceClaim { index, grant } => {
            return kore_abi::svc::DeviceClaimCloseWire {
                op: DeviceReqWire::OP_QUIESCE_CLAIM,
                reserved: [0; 3],
                index: index.into(),
                grant: grant.into(),
            }
            .encode();
        }
        DeviceRequest::RetireClaimDma { index, grant } => {
            return kore_abi::svc::DeviceClaimCloseWire {
                op: DeviceReqWire::OP_RETIRE_CLAIM_DMA,
                reserved: [0; 3],
                index: index.into(),
                grant: grant.into(),
            }
            .encode();
        }
        DeviceRequest::Claim(DeviceSelector::Pci(sel))
        | DeviceRequest::Inspect(DeviceSelector::Pci(sel)) => {
            w.op = if matches!(*req, DeviceRequest::Claim(_)) {
                DeviceReqWire::OP_CLAIM
            } else {
                DeviceReqWire::OP_INSPECT
            };
            w.species = DeviceSpecies::Pci as u8;
            w.class = sel.class;
            w.subclass = sel.subclass;
            w.progif = sel.progif;
            // Zero is the frozen pre-mask spelling for the previously
            // unconditional BAR0 requirement. Preserve it so BAR0-only
            // clients remain intelligible to an older broker; a multi-BAR
            // claimant makes the formerly-reserved byte nonzero and an old
            // broker therefore refuses before claiming.
            w.reserved[0] = if sel.bars == PciBarMask::BAR0 {
                0
            } else {
                sel.bars.bits()
            };
            let interrupt_flags = match sel.interrupts {
                PciInterruptRequirement::None => 0,
                PciInterruptRequirement::MsiFamily => REQ_MSI,
                PciInterruptRequirement::Msix(block) => {
                    let first = block.first_entry().to_le_bytes();
                    let count = block.count().to_le_bytes();
                    w.reserved2 = [first[0], first[1], count[0], count[1]];
                    REQ_MSI | REQ_MSIX
                }
            };
            w.flags = if sel.dma { REQ_DMA } else { 0 } | interrupt_flags;
            w.word = sel.identity.packed().into();
        }
        DeviceRequest::Claim(DeviceSelector::Platform { hid, irq, dma })
        | DeviceRequest::Inspect(DeviceSelector::Platform { hid, irq, dma }) => {
            w.op = if matches!(*req, DeviceRequest::Claim(_)) {
                DeviceReqWire::OP_CLAIM
            } else {
                DeviceReqWire::OP_INSPECT
            };
            w.species = DeviceSpecies::Platform as u8;
            w.class = kore_abi::device::DeviceRecord::PLATFORM_CLASS_HID;
            w.flags = if irq { REQ_LINE } else { 0 } | if dma { REQ_DMA } else { 0 };
            w.word = hid.into();
        }
        DeviceRequest::Claim(DeviceSelector::PlatformI2cHid { irq })
        | DeviceRequest::Inspect(DeviceSelector::PlatformI2cHid { irq }) => {
            w.op = if matches!(*req, DeviceRequest::Claim(_)) {
                DeviceReqWire::OP_CLAIM
            } else {
                DeviceReqWire::OP_INSPECT
            };
            w.species = DeviceSpecies::Platform as u8;
            w.class = kore_abi::device::DeviceRecord::PLATFORM_CLASS_I2C_HID;
            w.flags = if irq { REQ_LINE } else { 0 };
        }
        DeviceRequest::Claim(DeviceSelector::PciI2cHid { irq })
        | DeviceRequest::Inspect(DeviceSelector::PciI2cHid { irq }) => {
            w.op = if matches!(*req, DeviceRequest::Claim(_)) {
                DeviceReqWire::OP_CLAIM
            } else {
                DeviceReqWire::OP_INSPECT
            };
            w.species = DeviceSpecies::Pci as u8;
            w.class = 0x0c;
            w.subclass = 0x80;
            w.flags = REQ_I2C_HID | if irq { REQ_LINE } else { 0 };
        }
        DeviceRequest::Release { index } => {
            w.op = DeviceReqWire::OP_RELEASE;
            w.word = index.into();
        }
        DeviceRequest::InspectI2c { row } => {
            w.op = DeviceReqWire::OP_INSPECT_I2C;
            w.word = u32::from(row).into();
        }
        DeviceRequest::Count => w.op = DeviceReqWire::OP_COUNT,
        DeviceRequest::MintFace { face } => {
            w.op = DeviceReqWire::OP_MINT_FACE;
            w.word = face.into();
        }
        // Encodable for the codec's totality; `decode_req` refuses it —
        // see the vocabulary's own doc.
        DeviceRequest::Provision => w.op = DeviceReqWire::OP_PROVISION,
    }
    w
}

/// Lift a wire request — fail-closed.
///
/// The unused fields of each op are required to be **zero**, not merely
/// ignored: a `Count` carrying a class triple, or a PCI claim carrying a
/// platform-line requirement, is a peer this service does not understand,
/// and reading past the confusion is how one op quietly becomes another.
pub fn decode_req(w: &DeviceReqWire) -> Option<DeviceRequest> {
    if w.op == DeviceReqWire::OP_CLOSE_CLAIM {
        let request = kore_abi::svc::DeviceClaimCloseWire::view(w);
        let tag = request.grant.get() as u32;
        return (request.reserved == [0; 3] && tag != 0 && tag != u32::MAX).then_some(
            DeviceRequest::CloseClaim {
                index: request.index.get(),
                grant: request.grant.get(),
            },
        );
    }
    if w.op == DeviceReqWire::OP_QUIESCE_CLAIM {
        let request = kore_abi::svc::DeviceClaimCloseWire::view(w);
        let tag = request.grant.get() as u32;
        return (request.reserved == [0; 3] && tag != 0 && tag != u32::MAX).then_some(
            DeviceRequest::QuiesceClaim {
                index: request.index.get(),
                grant: request.grant.get(),
            },
        );
    }
    if w.op == DeviceReqWire::OP_RETIRE_CLAIM_DMA {
        let request = kore_abi::svc::DeviceClaimCloseWire::view(w);
        let tag = request.grant.get() as u32;
        return (request.reserved == [0; 3] && tag != 0 && tag != u32::MAX).then_some(
            DeviceRequest::RetireClaimDma {
                index: request.index.get(),
                grant: request.grant.get(),
            },
        );
    }
    if w.reserved[1] != 0 {
        return None;
    }
    if w.flags & !REQ_ALL != 0 {
        return None;
    }
    Some(match w.op {
        DeviceReqWire::OP_CLAIM => match DeviceSpecies::from_bits(w.species)? {
            DeviceSpecies::Pci => {
                if w.flags & REQ_I2C_HID != 0 {
                    if (w.class, w.subclass, w.progif) != (0x0c, 0x80, 0)
                        || w.flags & !(REQ_I2C_HID | REQ_LINE) != 0
                        || w.reserved != [0; 2]
                        || w.reserved2 != [0; 4]
                        || w.word.get() != 0
                    {
                        return None;
                    }
                    return Some(DeviceRequest::Claim(DeviceSelector::PciI2cHid {
                        irq: w.flags & REQ_LINE != 0,
                    }));
                }
                // A PCI claim names no ACPI id or line requirement. Its word
                // is instead the complete exact-identity selector vocabulary:
                // zero for ANY, `vendor | device << 16` otherwise.
                if w.flags & REQ_LINE != 0 || w.flags & REQ_MSIX != 0 && w.flags & REQ_MSI == 0 {
                    return None;
                }
                let interrupts = if w.flags & REQ_MSIX != 0 {
                    let first = u16::from_le_bytes([w.reserved2[0], w.reserved2[1]]);
                    let count = u16::from_le_bytes([w.reserved2[2], w.reserved2[3]]);
                    PciInterruptRequirement::Msix(PciMsixBlock::new(first, count)?)
                } else if w.flags & REQ_MSI != 0 {
                    if w.reserved2 != [0; 4] {
                        return None;
                    }
                    PciInterruptRequirement::MsiFamily
                } else {
                    if w.reserved2 != [0; 4] {
                        return None;
                    }
                    PciInterruptRequirement::None
                };
                let bars = if w.reserved[0] == 0 {
                    PciBarMask::BAR0
                } else {
                    PciBarMask::from_bits(w.reserved[0])?
                };
                DeviceRequest::Claim(DeviceSelector::Pci(PciSelector {
                    class: w.class,
                    subclass: w.subclass,
                    progif: w.progif,
                    identity: PciIdentitySelector::from_packed(w.word.get()),
                    bars,
                    dma: w.flags & REQ_DMA != 0,
                    interrupts,
                }))
            }
            DeviceSpecies::Platform => {
                // A platform claim names either a compressed ACPI id or one
                // topology-derived platform role. It never carries PCI
                // MSI/BAR vocabulary; DMA is explicit for bus masters.
                if (w.subclass, w.progif) != (0, 0)
                    || w.flags & (REQ_MSI | REQ_MSIX | REQ_I2C_HID) != 0
                    || w.reserved[0] != 0
                    || w.reserved2 != [0; 4]
                {
                    return None;
                }
                match (w.class, w.word.get()) {
                    (kore_abi::device::DeviceRecord::PLATFORM_CLASS_HID, hid) if hid != 0 => {
                        DeviceRequest::Claim(DeviceSelector::Platform {
                            hid,
                            irq: w.flags & REQ_LINE != 0,
                            dma: w.flags & REQ_DMA != 0,
                        })
                    }
                    (kore_abi::device::DeviceRecord::PLATFORM_CLASS_I2C_HID, 0) => {
                        DeviceRequest::Claim(DeviceSelector::PlatformI2cHid {
                            irq: w.flags & REQ_LINE != 0,
                        })
                    }
                    _ => return None,
                }
            }
        },
        DeviceReqWire::OP_INSPECT => {
            // Inspection carries exactly the claim selector vocabulary; only
            // the operation and the absence of an authority-bearing reply
            // differ. Reuse the claim decoder so the two cannot drift.
            let mut claim = *w;
            claim.op = DeviceReqWire::OP_CLAIM;
            let DeviceRequest::Claim(selector) = decode_req(&claim)? else {
                return None;
            };
            DeviceRequest::Inspect(selector)
        }
        DeviceReqWire::OP_INSPECT_I2C => {
            if (w.species, w.class, w.subclass, w.progif, w.flags) != (0, 0, 0, 0, 0)
                || w.reserved != [0; 2]
                || w.reserved2 != [0; 4]
                || w.word.get() >= u32::from(kore_abi::device::I2C_ADMISSION_ROWS)
            {
                return None;
            }
            DeviceRequest::InspectI2c {
                row: w.word.get() as u8,
            }
        }
        DeviceReqWire::OP_RELEASE => {
            if (w.species, w.class, w.subclass, w.progif, w.flags) != (0, 0, 0, 0, 0)
                || w.reserved[0] != 0
                || w.reserved2 != [0; 4]
            {
                return None;
            }
            DeviceRequest::Release {
                index: w.word.get(),
            }
        }
        DeviceReqWire::OP_COUNT => {
            if (w.species, w.class, w.subclass, w.progif, w.flags) != (0, 0, 0, 0, 0)
                || w.word.get() != 0
                || w.reserved[0] != 0
                || w.reserved2 != [0; 4]
            {
                return None;
            }
            DeviceRequest::Count
        }
        DeviceReqWire::OP_MINT_FACE => {
            if (w.species, w.class, w.subclass, w.progif, w.flags) != (0, 0, 0, 0, 0)
                || w.reserved[0] != 0
                || w.reserved2 != [0; 4]
            {
                return None;
            }
            DeviceRequest::MintFace { face: w.word.get() }
        }
        // `Provision` is the serve host's own synthetic step
        // (`implicit_return_request`); a peer writing its op onto the
        // request ring is forging internal vocabulary, and reading past
        // that would let a claimant consume another face's banked leg.
        DeviceReqWire::OP_PROVISION => return None,
        _ => return None,
    })
}

/// Lower a response (total).
///
/// `grants` is the count the serve host actually effected, never a count
/// reconstructed from the response tag.  A mismatch remains an invalid wire
/// record and [`decode_rep`] refuses it; silently changing zero back to one
/// would make a client wait forever for a device that never landed.
pub fn encode_rep(rep: &DeviceResponse, grants: u8) -> DeviceRepWire {
    let mut w = DeviceRepWire {
        tag: 0,
        refusal: 0,
        grants,
        species: 0,
        reserved: [0; 4],
        word: 0.into(),
        diag: 0.into(),
        vendor: 0.into(),
        device: 0.into(),
    };
    match *rep {
        DeviceResponse::ClaimQuiesced(report) => {
            w.tag = DeviceRepWire::TAG_CLAIM_QUIESCED;
            w.reserved[0] = report.verdict;
            w.reserved[1] = report.flags;
            w.word = report.rid.into();
        }
        DeviceResponse::ClaimClosed(report) => {
            w.tag = DeviceRepWire::TAG_CLAIM_CLOSED;
            w.reserved[0] = report.state as u8;
            w.word = report.readers.into();
        }
        DeviceResponse::ClaimDmaRetired(report) => {
            w.tag = DeviceRepWire::TAG_CLAIM_DMA_RETIRED;
            w.reserved[0] = report.state as u8;
            w.word = report.removed.into();
        }
        DeviceResponse::Granted {
            index,
            species,
            diag,
            revision,
            vendor,
            device,
        } => {
            w.tag = DeviceRepWire::TAG_GRANTED;
            w.species = species as u8;
            w.word = index.into();
            w.diag = diag.into();
            w.reserved[0] = revision;
            w.vendor = vendor.into();
            w.device = device.into();
        }
        DeviceResponse::Observed(observed) => {
            w.tag = DeviceRepWire::TAG_OBSERVED;
            w.species = observed.species as u8;
            w.word = observed.index.into();
            w.diag = observed.diag.into();
            w.reserved[0] = observed.revision;
            w.reserved[1] = observed.flags;
        }
        DeviceResponse::I2cObserved(page) => {
            w.tag = DeviceRepWire::TAG_I2C_OBSERVED;
            w.reserved[0] = page.row;
            w.word = page.words[0].into();
            w.diag = page.words[1].into();
        }
        DeviceResponse::Released => w.tag = DeviceRepWire::TAG_RELEASED,
        DeviceResponse::Counted { len } => {
            w.tag = DeviceRepWire::TAG_COUNTED;
            w.word = len.into();
        }
        DeviceResponse::Provisioning { face } => {
            w.tag = DeviceRepWire::TAG_PROVISIONING;
            w.word = face.into();
        }
        DeviceResponse::Banked { face } => {
            w.tag = DeviceRepWire::TAG_BANKED;
            w.word = face.into();
        }
        DeviceResponse::Minted { via } => {
            w.tag = DeviceRepWire::TAG_MINTED;
            w.word = u32::from(via).into();
        }
        DeviceResponse::Refused(r) => {
            w.tag = DeviceRepWire::TAG_REFUSED;
            w.refusal = match r {
                DeviceRefusal::NoMatch => DeviceRepWire::REFUSE_NO_MATCH,
                DeviceRefusal::Taken => DeviceRepWire::REFUSE_TAKEN,
                DeviceRefusal::Denied => DeviceRepWire::REFUSE_DENIED,
                DeviceRefusal::NotHeld => DeviceRepWire::REFUSE_NOT_HELD,
                DeviceRefusal::NoSupply => DeviceRepWire::REFUSE_NO_SUPPLY,
                DeviceRefusal::BadFace => DeviceRepWire::REFUSE_BAD_FACE,
                DeviceRefusal::BadConvey => DeviceRepWire::REFUSE_BAD_CONVEY,
                DeviceRefusal::Busy => DeviceRepWire::REFUSE_BUSY,
            };
        }
    }
    w
}

/// Lift a wire reply — fail-closed.
///
/// The grant count is checked **against the tag**: a refusal claiming a
/// grant, or a `Granted` claiming none, is a protocol break and not a
/// reply to read. That check is the one that matters here — a client that
/// believed a grant-less `Granted` would go looking for a device token
/// that never arrived and bind against whatever was in the slot.
pub fn decode_rep(w: &DeviceRepWire) -> Option<(DeviceResponse, u8)> {
    if w.tag == DeviceRepWire::TAG_CLAIM_QUIESCED {
        if w.refusal != 0
            || w.grants != 0
            || w.species != 0
            || w.reserved[2..] != [0; 2]
            || w.diag.get() != 0
            || w.vendor.get() != 0
            || w.device.get() != 0
        {
            return None;
        }
        let report = kore_abi::session::QuiesceReport {
            rid: w.word.get(),
            verdict: w.reserved[0],
            flags: w.reserved[1],
        };
        return kore_abi::session::QuiesceReport::decode(&report.encode())
            .map(|report| (DeviceResponse::ClaimQuiesced(report), 0));
    }
    if w.tag == DeviceRepWire::TAG_CLAIM_CLOSED {
        if w.refusal != 0
            || w.grants != 0
            || w.species != 0
            || w.reserved[1..] != [0; 3]
            || w.diag.get() != 0
            || w.vendor.get() != 0
            || w.device.get() != 0
        {
            return None;
        }
        // Reuse the canonical kernel report decoder for state/count validity.
        let mut record = kore_abi::session::DeviceClaimCloseReport {
            state: kore_abi::session::DeviceClaimCloseState::Mismatch,
            readers: 0,
        }
        .encode();
        record.scope_rid = u64::from(w.reserved[0]) | u64::from(w.word.get()) << 8;
        return kore_abi::session::DeviceClaimCloseReport::decode(&record)
            .map(|report| (DeviceResponse::ClaimClosed(report), 0));
    }
    if w.tag == DeviceRepWire::TAG_CLAIM_DMA_RETIRED {
        if w.refusal != 0
            || w.grants != 0
            || w.species != 0
            || w.reserved[1..] != [0; 3]
            || w.diag.get() != 0
            || w.vendor.get() != 0
            || w.device.get() != 0
        {
            return None;
        }
        // Reuse the canonical kernel report decoder for state/count validity.
        let mut record = kore_abi::session::DeviceClaimDmaReport {
            state: kore_abi::session::DeviceClaimDmaState::Retired,
            removed: 0,
        }
        .encode();
        record.scope_rid = u64::from(w.reserved[0]) | u64::from(w.word.get()) << 8;
        return kore_abi::session::DeviceClaimDmaReport::decode(&record)
            .map(|report| (DeviceResponse::ClaimDmaRetired(report), 0));
    }
    // PCI identity belongs only to an actual grant, never an observation
    // (including I2C admission pages) or a refusal.
    if w.tag != DeviceRepWire::TAG_GRANTED && (w.vendor.get() != 0 || w.device.get() != 0) {
        return None;
    }
    match w.tag {
        DeviceRepWire::TAG_GRANTED if w.reserved[1..] != [0u8; 3] => return None,
        DeviceRepWire::TAG_OBSERVED if w.reserved[2..] != [0u8; 2] => return None,
        DeviceRepWire::TAG_I2C_OBSERVED if w.reserved[1..] != [0u8; 3] => return None,
        DeviceRepWire::TAG_GRANTED
        | DeviceRepWire::TAG_OBSERVED
        | DeviceRepWire::TAG_I2C_OBSERVED => {}
        _ if w.diag.get() != 0 || w.reserved != [0u8; 4] => return None,
        _ => {}
    }
    let rep = match (w.tag, w.refusal) {
        (DeviceRepWire::TAG_I2C_OBSERVED, 0) => {
            if w.grants != 0 || w.species != 0 {
                return None;
            }
            let page = kore_abi::device::I2cAdmissionPage {
                row: w.reserved[0],
                words: [w.word.get(), w.diag.get()],
            };
            if !page.valid() {
                return None;
            }
            DeviceResponse::I2cObserved(page)
        }
        (DeviceRepWire::TAG_GRANTED, 0) => {
            if w.grants != 1 {
                return None;
            }
            let species = DeviceSpecies::from_bits(w.species)?;
            if species == DeviceSpecies::Platform && (w.vendor.get() != 0 || w.device.get() != 0) {
                return None;
            }
            DeviceResponse::Granted {
                index: w.word.get(),
                species,
                diag: w.diag.get(),
                revision: w.reserved[0],
                vendor: w.vendor.get(),
                device: w.device.get(),
            }
        }
        (DeviceRepWire::TAG_OBSERVED, 0) => {
            if w.grants != 0 {
                return None;
            }
            DeviceResponse::Observed(DeviceObservation {
                index: w.word.get(),
                species: DeviceSpecies::from_bits(w.species)?,
                flags: w.reserved[1],
                diag: w.diag.get(),
                revision: w.reserved[0],
            })
        }
        (DeviceRepWire::TAG_RELEASED, 0) => {
            if w.grants != 0 || w.species != 0 || w.word.get() != 0 {
                return None;
            }
            DeviceResponse::Released
        }
        (DeviceRepWire::TAG_COUNTED, 0) => {
            if w.grants != 0 || w.species != 0 {
                return None;
            }
            DeviceResponse::Counted { len: w.word.get() }
        }
        (DeviceRepWire::TAG_PROVISIONING, 0) => {
            if w.grants != 0 || w.species != 0 {
                return None;
            }
            DeviceResponse::Provisioning { face: w.word.get() }
        }
        (DeviceRepWire::TAG_BANKED, 0) => {
            if w.grants != 0 || w.species != 0 {
                return None;
            }
            DeviceResponse::Banked { face: w.word.get() }
        }
        (DeviceRepWire::TAG_MINTED, 0) => {
            if w.grants != 0 || w.species != 0 {
                return None;
            }
            let via = u16::try_from(w.word.get()).ok()?;
            DeviceResponse::Minted { via }
        }
        (DeviceRepWire::TAG_REFUSED, r) => {
            if w.grants != 0 || w.species != 0 || w.word.get() != 0 {
                return None;
            }
            DeviceResponse::Refused(match r {
                DeviceRepWire::REFUSE_NO_MATCH => DeviceRefusal::NoMatch,
                DeviceRepWire::REFUSE_TAKEN => DeviceRefusal::Taken,
                DeviceRepWire::REFUSE_DENIED => DeviceRefusal::Denied,
                DeviceRepWire::REFUSE_NOT_HELD => DeviceRefusal::NotHeld,
                DeviceRepWire::REFUSE_NO_SUPPLY => DeviceRefusal::NoSupply,
                DeviceRepWire::REFUSE_BAD_FACE => DeviceRefusal::BadFace,
                DeviceRepWire::REFUSE_BAD_CONVEY => DeviceRefusal::BadConvey,
                DeviceRepWire::REFUSE_BUSY => DeviceRefusal::Busy,
                _ => return None,
            })
        }
        _ => return None,
    };
    Some((rep, w.grants))
}

impl SvcProtocol for DeviceBroker {
    type Request = DeviceRequest;
    type Response = DeviceResponse;
    type ReqWire = DeviceReqWire;
    type RepWire = DeviceRepWire;

    /// One — a `Claim` hands over exactly one device, and no verb here
    /// hands over more.
    const REPLY_GRANTS: usize = 1;

    /// **One** — the control pair (`res/DEVICE_SERVICE.md` §10, the
    /// minted-face broker): its client is the kernel chain founder, which
    /// drives `MintFace` through it at each driver row's founding round.
    /// Every claimant rides a runtime-minted pair instead (via
    /// `FACE_VIA_BASE + n`), so the driver-row count never touches this
    /// constant again — the wall this replaces was three static pairs
    /// against [`MAX_SERVE_PAIRS`](kore_rt::svc::MAX_SERVE_PAIRS).
    /// Requests are stamped with their arrival pair either way
    /// ([`Serve::via`](kore_rt::service::Serve::via)) — the ledger's
    /// owner column.
    const SERVE_PAIRS: usize = 1;

    /// A claimed device outlives the request and may coexist with a later
    /// platform-device claim. Rotate the receiver-owned landing instead of
    /// reposting over the first live device record.
    fn request_grants(request: &DeviceReqWire) -> usize {
        // These operations are always grantless. In particular a retiring
        // owner need not reserve another device landing to close its old one.
        if matches!(
            request.op,
            DeviceReqWire::OP_CLOSE_CLAIM
                | DeviceReqWire::OP_QUIESCE_CLAIM
                | DeviceReqWire::OP_RETIRE_CLAIM_DMA
        ) {
            0
        } else {
            Self::REPLY_GRANTS
        }
    }

    fn retain_grant(index: usize) -> bool {
        index == 0
    }

    fn reply_matches(request: &DeviceReqWire, response: &DeviceResponse, grants: u8) -> bool {
        match (request.op, response) {
            (
                DeviceReqWire::OP_CLOSE_CLAIM,
                DeviceResponse::ClaimClosed(_) | DeviceResponse::Refused(DeviceRefusal::Denied),
            )
            | (
                DeviceReqWire::OP_RETIRE_CLAIM_DMA,
                DeviceResponse::ClaimDmaRetired(_)
                | DeviceResponse::Refused(DeviceRefusal::Denied | DeviceRefusal::Busy),
            )
            | (
                DeviceReqWire::OP_QUIESCE_CLAIM,
                DeviceResponse::ClaimQuiesced(_)
                | DeviceResponse::Refused(DeviceRefusal::Denied | DeviceRefusal::Busy),
            ) => grants == 0,
            (DeviceReqWire::OP_CLAIM, DeviceResponse::Granted { species, .. }) => {
                grants == 1 && request.species == *species as u8
            }
            (DeviceReqWire::OP_INSPECT, DeviceResponse::Observed(observed)) => {
                grants == 0 && request.species == observed.species as u8
            }
            (DeviceReqWire::OP_INSPECT_I2C, DeviceResponse::I2cObserved(page)) => {
                grants == 0 && request.word.get() == u32::from(page.row) && page.valid()
            }
            (
                DeviceReqWire::OP_CLAIM,
                DeviceResponse::Refused(
                    DeviceRefusal::NoMatch | DeviceRefusal::Taken | DeviceRefusal::Denied,
                ),
            )
            | (
                DeviceReqWire::OP_RELEASE,
                DeviceResponse::Refused(DeviceRefusal::Denied | DeviceRefusal::NotHeld),
            )
            | (
                DeviceReqWire::OP_INSPECT,
                DeviceResponse::Refused(DeviceRefusal::NoMatch | DeviceRefusal::Denied),
            )
            | (DeviceReqWire::OP_INSPECT_I2C, DeviceResponse::Refused(DeviceRefusal::Denied))
            | (DeviceReqWire::OP_COUNT, DeviceResponse::Counted { .. })
            | (DeviceReqWire::OP_COUNT, DeviceResponse::Refused(DeviceRefusal::Denied))
            | (DeviceReqWire::OP_MINT_FACE, DeviceResponse::Provisioning { .. })
            | (
                DeviceReqWire::OP_MINT_FACE,
                DeviceResponse::Refused(DeviceRefusal::NoSupply | DeviceRefusal::BadFace),
            )
            | (
                DeviceReqWire::OP_PROVISION,
                DeviceResponse::Banked { .. } | DeviceResponse::Minted { .. },
            )
            | (
                DeviceReqWire::OP_PROVISION,
                DeviceResponse::Refused(
                    DeviceRefusal::BadConvey | DeviceRefusal::NoSupply | DeviceRefusal::Denied,
                ),
            ) => grants == 0,
            _ => false,
        }
    }

    /// The walking return plane's synthetic step: a convey landed on the
    /// landing this broker offered, so the host drives one `Provision`
    /// through the ordinary pipeline — the exec convey-table shape.
    fn implicit_return_request() -> Option<DeviceRequest> {
        Some(DeviceRequest::Provision)
    }

    fn encode_req(req: &DeviceRequest) -> DeviceReqWire {
        encode_req(req)
    }
    fn decode_req(wire: &DeviceReqWire) -> Option<DeviceRequest> {
        decode_req(wire)
    }
    fn encode_rep(rep: &DeviceResponse, grants: u8) -> DeviceRepWire {
        encode_rep(rep, grants)
    }
    fn decode_rep(wire: &DeviceRepWire) -> Option<(DeviceResponse, u8)> {
        decode_rep(wire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xhci() -> DeviceRequest {
        DeviceRequest::Claim(DeviceSelector::Pci(PciSelector {
            class: 0x0c,
            subclass: 0x03,
            progif: 0x30,
            identity: PciIdentitySelector::ANY,
            bars: PciBarMask::BAR0,
            dma: true,
            interrupts: PciInterruptRequirement::MsiFamily,
        }))
    }

    #[test]
    fn claim_retirement_replies_are_reachable_and_operation_correlated() {
        use kore_abi::session::{
            DeviceClaimCloseReport, DeviceClaimCloseState, QUIESCE_FLAG_IRQ_MASKED,
            QUIESCE_VERDICT_PROVEN, QuiesceReport,
        };
        let close = DeviceRequest::CloseClaim {
            index: 7,
            grant: 0x1234_00000077,
        };
        let quiesce = DeviceRequest::QuiesceClaim {
            index: 7,
            grant: 0x1234_00000077,
        };
        let closed = DeviceResponse::ClaimClosed(DeviceClaimCloseReport {
            state: DeviceClaimCloseState::AdmissionDrained,
            readers: 0,
        });
        let quiet = DeviceResponse::ClaimQuiesced(QuiesceReport {
            rid: 42,
            verdict: QUIESCE_VERDICT_PROVEN,
            flags: QUIESCE_FLAG_IRQ_MASKED,
        });
        let dma = DeviceRequest::RetireClaimDma {
            index: 7,
            grant: 0x1234_00000077,
        };
        let retired = DeviceResponse::ClaimDmaRetired(kore_abi::session::DeviceClaimDmaReport {
            state: kore_abi::session::DeviceClaimDmaState::Retired,
            removed: 5,
        });
        for (request, response) in [(close, closed), (quiesce, quiet), (dma, retired)] {
            let wire = encode_req(&request);
            assert_eq!(
                DeviceBroker::request_grants(&wire),
                0,
                "retirement never reserves a fresh device landing"
            );
            assert_eq!(decode_req(&wire), Some(request));
            let (decoded, grants) = decode_rep(&encode_rep(&response, 0)).unwrap();
            assert_eq!(decoded, response);
            assert!(DeviceBroker::reply_matches(&wire, &decoded, grants));
            assert!(!DeviceBroker::reply_matches(&wire, &decoded, 1));
            assert!(DeviceBroker::reply_matches(
                &wire,
                &DeviceResponse::Refused(DeviceRefusal::Denied),
                0
            ));
        }
        assert!(!DeviceBroker::reply_matches(&encode_req(&close), &quiet, 0));
        assert!(!DeviceBroker::reply_matches(
            &encode_req(&quiesce),
            &closed,
            0
        ));
        for response in [closed, quiet] {
            assert!(!DeviceBroker::reply_matches(
                &encode_req(&dma),
                &response,
                0
            ));
        }
        for request in [close, quiesce] {
            assert!(!DeviceBroker::reply_matches(
                &encode_req(&request),
                &retired,
                0
            ));
        }
        for mutate in [
            |w: &mut DeviceRepWire| w.reserved[0] = 0xff,
            |w: &mut DeviceRepWire| w.reserved[1] = 1,
            |w: &mut DeviceRepWire| w.grants = 1,
            |w: &mut DeviceRepWire| w.species = 1,
            |w: &mut DeviceRepWire| w.refusal = 1,
            |w: &mut DeviceRepWire| w.diag = 1.into(),
        ] {
            let mut encoded = encode_rep(&retired, 0);
            mutate(&mut encoded);
            assert_eq!(decode_rep(&encoded), None);
        }
        let busy = DeviceResponse::Refused(DeviceRefusal::Busy);
        assert_eq!(decode_rep(&encode_rep(&busy, 0)), Some((busy, 0)));
        assert!(DeviceBroker::reply_matches(&encode_req(&quiesce), &busy, 0));
        assert!(!DeviceBroker::reply_matches(&encode_req(&close), &busy, 0));
        for mutate in [
            |w: &mut DeviceRepWire| w.reserved[0] = 0xff,
            |w: &mut DeviceRepWire| w.reserved[1] = 0x80,
            |w: &mut DeviceRepWire| w.reserved[2] = 1,
            |w: &mut DeviceRepWire| w.diag = 1.into(),
            |w: &mut DeviceRepWire| w.grants = 1,
            |w: &mut DeviceRepWire| w.species = 1,
        ] {
            let mut dirty = encode_rep(&quiet, 0);
            mutate(&mut dirty);
            assert_eq!(decode_rep(&dirty), None);
        }
    }

    #[test]
    fn exact_claim_close_wire_is_shape_tight_and_grantless() {
        use kore_abi::session::{DeviceClaimCloseReport, DeviceClaimCloseState};
        let request = DeviceRequest::CloseClaim {
            index: 0x12345678,
            grant: 0x11223344_55667788,
        };
        assert_eq!(decode_req(&encode_req(&request)), Some(request));
        for grant in [0, u32::MAX as u64] {
            assert_eq!(
                decode_req(&encode_req(&DeviceRequest::CloseClaim { index: 7, grant })),
                None
            );
        }
        let mut malformed = kore_abi::svc::DeviceClaimCloseWire::view(&encode_req(&request));
        malformed.reserved[1] = 1;
        assert_eq!(decode_req(&malformed.encode()), None);
        let reply = DeviceResponse::ClaimClosed(DeviceClaimCloseReport {
            state: DeviceClaimCloseState::Pending,
            readers: 3,
        });
        assert_eq!(decode_rep(&encode_rep(&reply, 0)), Some((reply, 0)));
        assert_eq!(decode_rep(&encode_rep(&reply, 1)), None);
        let mut malformed = encode_rep(&reply, 0);
        malformed.word = 0.into();
        assert_eq!(decode_rep(&malformed), None);
    }

    #[test]
    fn i2c_admission_inspection_has_no_grants_or_selector_side_channels() {
        use kore_abi::device::{
            AmlBuildDiagnostic, AmlBuildErrorClass, AmlBuildStage, I2C_ADMISSION_BUILD_ROW,
            I2C_ADMISSION_ROWS, I2cAdmissionPage, I2cAdmissionSnapshot,
        };
        for row in [0, 1, 24, I2C_ADMISSION_BUILD_ROW] {
            let request = DeviceRequest::InspectI2c { row };
            let wire = encode_req(&request);
            assert_eq!(decode_req(&wire), Some(request));
            for change in [
                |w: &mut DeviceReqWire| w.species = 1,
                |w: &mut DeviceReqWire| w.flags = 1,
                |w: &mut DeviceReqWire| w.class = 1,
                |w: &mut DeviceReqWire| w.reserved[0] = 1,
                |w: &mut DeviceReqWire| w.reserved2[3] = 1,
                |w: &mut DeviceReqWire| w.word = u32::from(I2C_ADMISSION_ROWS).into(),
            ] {
                let mut dirty = wire;
                change(&mut dirty);
                assert_eq!(decode_req(&dirty), None);
            }
        }
        for page in [
            I2cAdmissionSnapshot::default().page(0).unwrap(),
            I2cAdmissionPage {
                row: 24,
                words: [0x55, 0xa851_8086],
            },
            I2cAdmissionPage {
                row: I2C_ADMISSION_BUILD_ROW,
                words: AmlBuildDiagnostic::failure(
                    AmlBuildStage::SsdtLoad,
                    AmlBuildErrorClass::Opcode,
                    u32::MAX,
                    0x5bff,
                )
                .unwrap()
                .words(),
            },
        ] {
            let response = DeviceResponse::I2cObserved(page);
            let wire = encode_rep(&response, 0);
            assert_eq!(decode_rep(&wire), Some((response, 0)));
            assert_eq!(core::mem::size_of_val(&wire), 20);
            assert_eq!((wire.vendor.get(), wire.device.get()), (0, 0));
            for change in [
                |w: &mut DeviceRepWire| w.vendor = 0x8086u16.into(),
                |w: &mut DeviceRepWire| w.device = 0xa851u16.into(),
            ] {
                let mut dirty = wire;
                change(&mut dirty);
                assert_eq!(decode_rep(&dirty), None, "admission is not a PCI grant");
            }
            assert_eq!(decode_rep(&encode_rep(&response, 1)), None);
            let request = encode_req(&DeviceRequest::InspectI2c { row: page.row });
            assert!(DeviceBroker::reply_matches(&request, &response, 0));
            assert!(!DeviceBroker::reply_matches(&request, &response, 1));
            let other = encode_req(&DeviceRequest::InspectI2c {
                row: (page.row + 1) % I2C_ADMISSION_ROWS,
            });
            assert!(!DeviceBroker::reply_matches(&other, &response, 0));
            assert!(DeviceBroker::reply_matches(
                &request,
                &DeviceResponse::Refused(DeviceRefusal::Denied),
                0,
            ));
        }
    }

    /// Every vocabulary member survives a round trip in both directions.
    #[test]
    fn round_trips() {
        for req in [
            xhci(),
            DeviceRequest::Claim(DeviceSelector::Pci(PciSelector {
                class: 1,
                subclass: 8,
                progif: 2,
                identity: PciIdentitySelector::ANY,
                bars: PciBarMask::BAR0,
                dma: false,
                interrupts: PciInterruptRequirement::None,
            })),
            DeviceRequest::Claim(DeviceSelector::Pci(PciSelector {
                class: 0x03,
                subclass: 0x00,
                progif: 0x00,
                identity: PciIdentitySelector::exact(0x8086, 0x56a0),
                bars: PciBarMask::BAR0,
                dma: true,
                interrupts: PciInterruptRequirement::Msix(PciMsixBlock::ENTRIES_0_1),
            })),
            DeviceRequest::Claim(DeviceSelector::Platform {
                hid: 0x0303_d041,
                irq: true,
                dma: false,
            }),
            DeviceRequest::Claim(DeviceSelector::Platform {
                hid: 7,
                irq: false,
                dma: false,
            }),
            DeviceRequest::Claim(DeviceSelector::Platform {
                hid: 0x0d15_d041,
                irq: true,
                dma: true,
            }),
            DeviceRequest::Inspect(DeviceSelector::Pci(PciSelector {
                class: 0x03,
                subclass: 0x00,
                progif: 0x00,
                identity: PciIdentitySelector::exact(0x8086, 0x64a0),
                bars: PciBarMask::BAR0,
                dma: true,
                interrupts: PciInterruptRequirement::None,
            })),
            DeviceRequest::Claim(DeviceSelector::PlatformI2cHid { irq: true }),
            DeviceRequest::Inspect(DeviceSelector::PlatformI2cHid { irq: false }),
            DeviceRequest::Claim(DeviceSelector::PciI2cHid { irq: false }),
            DeviceRequest::Claim(DeviceSelector::PciI2cHid { irq: true }),
            DeviceRequest::Inspect(DeviceSelector::PciI2cHid { irq: false }),
            DeviceRequest::Inspect(DeviceSelector::PciI2cHid { irq: true }),
            DeviceRequest::Release { index: 0 },
            DeviceRequest::Release { index: u32::MAX },
            DeviceRequest::Count,
            DeviceRequest::MintFace { face: 0 },
            DeviceRequest::MintFace { face: 7 },
        ] {
            assert_eq!(decode_req(&encode_req(&req)), Some(req));
        }
        for (rep, grants) in [
            (
                DeviceResponse::Granted {
                    index: 4,
                    species: DeviceSpecies::Pci,
                    diag: 0,
                    revision: 0,
                    vendor: 0x1022,
                    device: 0x1118,
                },
                1,
            ),
            (
                DeviceResponse::Observed(DeviceObservation {
                    index: 7,
                    species: DeviceSpecies::Pci,
                    flags: 0xa5,
                    diag: 0xf001_0041,
                    revision: 4,
                }),
                0,
            ),
            (
                DeviceResponse::Granted {
                    index: u32::MAX,
                    species: DeviceSpecies::Platform,
                    diag: 0,
                    revision: 0,
                    vendor: 0,
                    device: 0,
                },
                1,
            ),
            (DeviceResponse::Released, 0),
            (DeviceResponse::Counted { len: 31 }, 0),
            (DeviceResponse::Provisioning { face: 2 }, 0),
            (DeviceResponse::Banked { face: 2 }, 0),
            (DeviceResponse::Minted { via: 5 }, 0),
            (DeviceResponse::Refused(DeviceRefusal::NoMatch), 0),
            (DeviceResponse::Refused(DeviceRefusal::Taken), 0),
            (DeviceResponse::Refused(DeviceRefusal::Denied), 0),
            (DeviceResponse::Refused(DeviceRefusal::NotHeld), 0),
            (DeviceResponse::Refused(DeviceRefusal::NoSupply), 0),
            (DeviceResponse::Refused(DeviceRefusal::BadFace), 0),
            (DeviceResponse::Refused(DeviceRefusal::BadConvey), 0),
        ] {
            assert_eq!(decode_rep(&encode_rep(&rep, grants)), Some((rep, grants)));
        }
    }

    #[test]
    fn pci_identity_uses_the_existing_claim_word_without_widening_the_wire() {
        let any = encode_req(&xhci());
        assert_eq!(any.word.get(), 0, "the old wildcard encoding is frozen");

        let exact = DeviceRequest::Claim(DeviceSelector::Pci(PciSelector {
            class: 0x03,
            subclass: 0x00,
            progif: 0x00,
            identity: PciIdentitySelector::exact(0x8086, 0x56a0),
            bars: PciBarMask::BAR0,
            dma: true,
            interrupts: PciInterruptRequirement::Msix(PciMsixBlock::ENTRIES_0_1),
        }));
        let wire = encode_req(&exact);
        assert_eq!(wire.word.get(), 0x56a0_8086);
        assert_eq!(wire.flags & (REQ_MSI | REQ_MSIX), REQ_MSI | REQ_MSIX);
        assert_eq!(wire.reserved2, [0, 0, 2, 0]);
        assert_eq!(decode_req(&wire), Some(exact));
        assert_eq!(core::mem::size_of::<DeviceReqWire>(), 16);
    }

    #[test]
    fn strict_msix_block_cannot_be_weakened_or_malformed_on_the_wire() {
        let mut wire = encode_req(&xhci());
        wire.flags = (wire.flags & !REQ_MSI) | REQ_MSIX;
        assert_eq!(decode_req(&wire), None);

        let mut invalid = encode_req(&DeviceRequest::Claim(DeviceSelector::Pci(PciSelector {
            class: 0x03,
            subclass: 0,
            progif: 0,
            identity: PciIdentitySelector::exact(0x8086, 0x56a0),
            bars: PciBarMask::BAR0,
            dma: true,
            interrupts: PciInterruptRequirement::Msix(PciMsixBlock::ENTRIES_0_1),
        })));
        invalid.reserved2 = [0, 0, 3, 0];
        assert_eq!(decode_req(&invalid), None, "count must be a power of two");
        invalid.reserved2 = [0xff, 0xff, 2, 0];
        assert_eq!(decode_req(&invalid), None, "entry range must not overflow");
    }

    #[test]
    fn pci_i2c_claim_and_inspect_reject_mixed_selector_vocabulary() {
        for irq in [false, true] {
            for request in [
                DeviceRequest::Claim(DeviceSelector::PciI2cHid { irq }),
                DeviceRequest::Inspect(DeviceSelector::PciI2cHid { irq }),
            ] {
                let base = encode_req(&request);
                assert_eq!(base.species, DeviceSpecies::Pci as u8);
                assert_eq!(base.flags, REQ_I2C_HID | if irq { REQ_LINE } else { 0 });
                for extra in [
                    REQ_DMA,
                    REQ_MSI,
                    REQ_MSIX,
                    REQ_MSI | REQ_MSIX,
                    1 << 5,
                    1 << 6,
                    1 << 7,
                ] {
                    let mut dirty = base;
                    dirty.flags |= extra;
                    assert_eq!(decode_req(&dirty), None, "mixed requirement {extra:#x}");
                }
                for dirty_field in [
                    |w: &mut DeviceReqWire| w.class ^= 1,
                    |w: &mut DeviceReqWire| w.subclass ^= 1,
                    |w: &mut DeviceReqWire| w.progif = 1,
                    |w: &mut DeviceReqWire| w.species = DeviceSpecies::Platform as u8,
                    |w: &mut DeviceReqWire| w.reserved[0] = 1,
                    |w: &mut DeviceReqWire| w.reserved[1] = 1,
                    |w: &mut DeviceReqWire| w.word = 0xa878_8086u32.into(),
                ] {
                    let mut dirty = base;
                    dirty_field(&mut dirty);
                    assert_eq!(decode_req(&dirty), None);
                }
                for index in 0..base.reserved2.len() {
                    let mut dirty = base;
                    dirty.reserved2[index] = 1;
                    assert_eq!(decode_req(&dirty), None, "reserved2 byte {index}");
                }
            }

            for request in [
                DeviceRequest::Claim(DeviceSelector::PlatformI2cHid { irq }),
                DeviceRequest::Inspect(DeviceSelector::PlatformI2cHid { irq }),
            ] {
                let mut platform = encode_req(&request);
                platform.flags |= REQ_I2C_HID;
                assert_eq!(
                    decode_req(&platform),
                    None,
                    "PCI topology bit cannot alias a platform row"
                );
            }
        }
    }

    #[test]
    fn multi_bar_mask_uses_the_old_reserve_and_round_trips_exactly() {
        let mut req = xhci();
        let DeviceRequest::Claim(DeviceSelector::Pci(selector)) = &mut req else {
            unreachable!();
        };
        selector.bars = PciBarMask::BAR0.union(PciBarMask::BAR4);
        let wire = encode_req(&req);
        assert_eq!(wire.reserved, [0x11, 0]);
        assert_eq!(decode_req(&wire), Some(req));

        let bar0 = encode_req(&xhci());
        assert_eq!(bar0.reserved, [0, 0], "BAR0 keeps the frozen wire spelling");
    }

    /// `Provision` is the host's own synthetic step: its op encodes (the
    /// codec is total) but never decodes — a claimant that writes it onto
    /// the request ring is refused, not interpreted.
    #[test]
    fn the_implicit_op_never_decodes_from_the_wire() {
        let w = encode_req(&DeviceRequest::Provision);
        assert_eq!(w.op, DeviceReqWire::OP_PROVISION);
        assert_eq!(decode_req(&w), None);
    }

    /// A `MintFace` carrying selector junk in the unused fields is a peer
    /// this service does not understand — same rule as every other op.
    #[test]
    fn a_dirty_mint_face_is_refused() {
        let base = encode_req(&DeviceRequest::MintFace { face: 1 });
        for dirt in [
            |w: &mut DeviceReqWire| w.species = 1,
            |w: &mut DeviceReqWire| w.class = 0x0c,
            |w: &mut DeviceReqWire| w.flags = REQ_DMA,
        ] {
            let mut dirty = base;
            dirt(&mut dirty);
            assert_eq!(decode_req(&dirty), None);
        }
        // A minted via past u16 in the reply is a protocol break.
        let mut wide = encode_rep(&DeviceResponse::Minted { via: 3 }, 0);
        wide.word = 0x1_0000u32.into();
        assert_eq!(decode_rep(&wide), None);
    }

    /// The fail-closed obligations, each one a distinct way a peer can lie.
    #[test]
    fn decode_fails_closed() {
        let base = encode_req(&xhci());

        // An all-zero record is not a claim — the 1-based discriminant
        // property. It matters here for the same reason it matters in the
        // power protocol: an unseeded record must never *acquire* anything.
        let mut zero = base;
        zero.op = 0;
        assert_eq!(decode_req(&zero), None);

        let mut unknown = base;
        unknown.op = 200;
        assert_eq!(decode_req(&unknown), None);

        // Byte one and the trailing reserve remain frozen zero. Byte zero is
        // now the PCI BAR mask, so an out-of-range bit is the fail-closed case.
        let mut dirty = base;
        dirty.reserved = [0, 1];
        assert_eq!(decode_req(&dirty), None);
        let mut dirty = base;
        dirty.reserved = [1 << 6, 0];
        assert_eq!(decode_req(&dirty), None);
        let mut dirty = base;
        dirty.reserved2 = [0, 0, 1, 0];
        assert_eq!(decode_req(&dirty), None);

        // An undefined requirement bit is refused: the bit is where a
        // later requirement goes, and honouring it as "no requirement"
        // now would make adding one a silent break.
        let mut extra = base;
        extra.flags |= 1 << 7;
        assert_eq!(decode_req(&extra), None);

        // Cross-species contamination: a PCI claim carrying the platform
        // line requirement, and a platform claim carrying a class triple.
        // The PCI word is no longer contamination: it is the exact identity
        // selector tested above.
        let mut hybrid = base;
        hybrid.flags |= REQ_LINE;
        assert_eq!(
            decode_req(&hybrid),
            None,
            "a PCI claim cannot request a platform interrupt line"
        );
        let mut hybrid = encode_req(&DeviceRequest::Claim(DeviceSelector::Platform {
            hid: 0x0303_d041,
            irq: true,
            dma: false,
        }));
        hybrid.class = 0x0c;
        assert_eq!(decode_req(&hybrid), None);

        // An unknown species.
        let mut alien = base;
        alien.species = 9;
        assert_eq!(decode_req(&alien), None);

        // A `Count` is not a smuggled claim.
        let mut counted = encode_req(&DeviceRequest::Count);
        counted.class = 0x0c;
        assert_eq!(decode_req(&counted), None);
    }

    /// The grant count must agree with the tag, in both directions.
    ///
    /// This is the reply-side check that actually protects a claimant: a
    /// `Granted` with no grant would send it looking for a device token
    /// that never arrived, and a refusal *with* one would leave authority
    /// it never accounted for sitting in its arena.
    #[test]
    fn a_grant_count_that_disagrees_with_its_tag_is_a_break() {
        let mut granted = encode_rep(
            &DeviceResponse::Granted {
                index: 2,
                species: DeviceSpecies::Pci,
                diag: 0,
                revision: 0,
                vendor: 0,
                device: 0,
            },
            1,
        );
        granted.grants = 0;
        assert_eq!(
            decode_rep(&granted),
            None,
            "a Granted with no grant is not a device"
        );

        let mut refused = encode_rep(&DeviceResponse::Refused(DeviceRefusal::Taken), 0);
        refused.grants = 1;
        assert_eq!(
            decode_rep(&refused),
            None,
            "a refusal must never carry authority"
        );

        // A refusal byte on a success tag, and an unknown refusal.
        let mut mixed = encode_rep(&DeviceResponse::Released, 0);
        mixed.refusal = DeviceRepWire::REFUSE_TAKEN;
        assert_eq!(decode_rep(&mixed), None);
        let mut alien = encode_rep(&DeviceResponse::Refused(DeviceRefusal::Taken), 0);
        alien.refusal = 99;
        assert_eq!(decode_rep(&alien), None);

        // An unknown species on a grant.
        let mut species = encode_rep(
            &DeviceResponse::Granted {
                index: 2,
                species: DeviceSpecies::Pci,
                diag: 0,
                revision: 0,
                vendor: 0,
                device: 0,
            },
            1,
        );
        species.species = 7;
        assert_eq!(decode_rep(&species), None);

        // A set reserve, on the reply side too.
        let mut dirty = encode_rep(&DeviceResponse::Released, 0);
        dirty.reserved = [0, 1, 0, 0];
        assert_eq!(decode_rep(&dirty), None);
    }

    /// The encoder preserves the host's effected count even when a buggy
    /// response still claims success.  The decoder then refuses the mismatch;
    /// it must never manufacture the missing transfer from the tag.
    #[test]
    fn encode_never_reconstructs_a_missing_grant_from_the_tag() {
        let wire = encode_rep(
            &DeviceResponse::Granted {
                index: 2,
                species: DeviceSpecies::Pci,
                diag: 0,
                revision: 0,
                vendor: 0,
                device: 0,
            },
            0,
        );
        assert_eq!(wire.grants, 0);
        assert_eq!(decode_rep(&wire), None);
    }

    /// The ring is ordered but carries no transaction id, so an otherwise
    /// valid reply for another verb must not complete the outstanding call.
    #[test]
    fn a_reply_must_answer_the_outstanding_operation() {
        let claim = encode_req(&xhci());
        let release = encode_req(&DeviceRequest::Release { index: 2 });
        let count = encode_req(&DeviceRequest::Count);
        let granted = DeviceResponse::Granted {
            index: 2,
            species: DeviceSpecies::Pci,
            diag: 0,
            revision: 0,
            vendor: 0,
            device: 0,
        };

        assert!(<DeviceBroker as SvcProtocol>::reply_matches(
            &claim, &granted, 1
        ));
        assert!(<DeviceBroker as SvcProtocol>::reply_matches(
            &release,
            &DeviceResponse::Refused(DeviceRefusal::Denied),
            0,
        ));
        assert!(<DeviceBroker as SvcProtocol>::reply_matches(
            &count,
            &DeviceResponse::Counted { len: 4 },
            0,
        ));
        assert!(<DeviceBroker as SvcProtocol>::reply_matches(
            &count,
            &DeviceResponse::Refused(DeviceRefusal::Denied),
            0,
        ));

        assert!(!<DeviceBroker as SvcProtocol>::reply_matches(
            &claim,
            &DeviceResponse::Released,
            0,
        ));
        assert!(!<DeviceBroker as SvcProtocol>::reply_matches(
            &release, &granted, 1,
        ));
        assert!(!<DeviceBroker as SvcProtocol>::reply_matches(
            &count, &granted, 1,
        ));

        let wrong_species = DeviceResponse::Granted {
            index: 2,
            species: DeviceSpecies::Platform,
            diag: 0,
            revision: 0,
            vendor: 0,
            device: 0,
        };
        assert!(!<DeviceBroker as SvcProtocol>::reply_matches(
            &claim,
            &wrong_species,
            1,
        ));
        assert!(!<DeviceBroker as SvcProtocol>::reply_matches(
            &release,
            &DeviceResponse::Released,
            0,
        ));
    }
}
