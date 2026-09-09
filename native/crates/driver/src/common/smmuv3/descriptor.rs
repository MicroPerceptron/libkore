//! SMMUv3 stream/context descriptors, command descriptors, and log records.
//!
//! Pure, spec-dictated (Arm IHI 0070) `const fn` value encoders with no MMIO /
//! barrier / allocation, over the neutral types in [`crate::common`], serving
//! any driver whose device embeds an SMMUv3.
//!
//! Seeded from the kernel's proven
//! `platform::arch::arm64::iommu::smmuv3::descriptor`, but **independently
//! owned by the driver tree** — deliberately not a mirror (#316, closed
//! not-planned): this copy may take driver-specific changes the kernel side
//! wouldn't, and neither side tracks the other. The SMMUv3 spec, not the
//! kernel module, is the reference for both.

use super::PageResponse;
use crate::common::flags::{AccessFlags, PageSize};
use crate::common::pagetable::{MmuError, MmuResult};

pub const STREAM_ENTRY_BYTES: usize = 64;
pub const CONTEXT_ENTRY_BYTES: usize = 64;
pub const COMMAND_ENTRY_BYTES: usize = 16;
pub const EVENT_ENTRY_BYTES: usize = 32;
pub const PRI_ENTRY_BYTES: usize = 16;
/// Policy ceiling on substream-id bits for a **linear** CD table (2^10 × 64 B =
/// one 64 KiB table). Enforced by
/// [`Smmuv3DeviceContext::stream_entry`](super::Smmuv3DeviceContext::stream_entry),
/// which clamps `cd_bits` to it; the raw [`StreamEntry`] encoders accept any
/// architected S1CDMax value (masked to its 5-bit field).
pub const MAX_LINEAR_CD_BITS: u8 = 10;

const STE_VALID: u64 = 1;
const STE_CFG_ABORT: u64 = 0 << 1;
const STE_CFG_BYPASS: u64 = 4 << 1;
const STE_CFG_S1: u64 = 5 << 1;
const STE_S1CTXPTR_MASK: u64 = 0x000f_ffff_ffff_ffc0;
const STE_S1CDMAX_SHIFT: u32 = 59;
const STE_S1DSS_TERMINATE: u64 = 0;
const STE_S1DSS_BYPASS: u64 = 1;
const STE_S1C_WBRA: u64 = (1 << 2) | (1 << 4);
const STE_S1CSH_INNER: u64 = 3 << 6;
const STE_S1STALLD: u64 = 1 << 27;
const STE_EATS_TRANS: u64 = 1 << 28;
const STE_SHCFG_INCOMING: u64 = 1 << 44;

const CD_VALID: u64 = 1 << 31;
const CD_EPD1: u64 = 1 << 30;
const CD_AA64: u64 = 1 << 41;
const CD_STALL: u64 = 1 << 44;
const CD_R: u64 = 1 << 45;
const CD_A: u64 = 1 << 46;
const CD_ASET: u64 = 1 << 47;
const CD_TTB0_MASK: u64 = 0x000f_ffff_ffff_fff0;

/// The architected SubstreamID width: SSID fields are 20 bits ([31:12] in
/// CFGI_CD / ATC_INV / PRI_RESP word 0), directly below StreamID [63:32] — an
/// unmasked over-wide `ssid` would spill into the StreamID field and mis-target
/// a different stream, so every encoder bounds `ssid` to this mask.
const CMD_SSID_MASK: u64 = 0xf_ffff;

/// The `ADDR` field of a stage-1 TLBI command — bits [63:12] of word 1.
const TLBI_VA_MASK: u64 = 0xffff_ffff_ffff_f000;

/// The `TTL` hint for a leaf page size: the translation level whose entries
/// the invalidation may restrict itself to (3 = last level).
const fn ttl_of(leaf_size: PageSize) -> Option<u8> {
    match leaf_size {
        PageSize::Size4K => Some(3),
        PageSize::Size2M => Some(2),
        PageSize::Size1G => Some(1),
        _ => None,
    }
}
/// The architected S1CDMax width: 5 bits ([63:59] of STE word 0). `cd_bits`
/// beyond it would be silently shifted out of the word, so the STE encoders
/// mask it (the linear-CD-table policy ceiling is [`MAX_LINEAR_CD_BITS`],
/// enforced by [`Smmuv3DeviceContext::stream_entry`](super::Smmuv3DeviceContext::stream_entry)).
const STE_S1CDMAX_MASK: u64 = 0x1f;

const OP_CFGI_STE: u8 = 0x03;
const OP_CFGI_ALL: u8 = 0x04;
const OP_CFGI_CD: u8 = 0x05;
const OP_TLBI_NH_ASID: u8 = 0x11;
const OP_TLBI_NH_VA: u8 = 0x12;
const OP_TLBI_NSNH_ALL: u8 = 0x30;
const OP_ATC_INV: u8 = 0x40;
const OP_PRI_RESP: u8 = 0x41;
const OP_RESUME: u8 = 0x44;
const OP_CMD_SYNC: u8 = 0x46;

/// A 64-byte SMMUv3 Stream Table Entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamEntry(pub [u64; 8]);

impl StreamEntry {
    #[inline]
    pub const fn invalid() -> Self {
        Self([0; 8])
    }

    #[inline]
    pub const fn abort() -> Self {
        let mut words = [0; 8];
        words[0] = STE_VALID | STE_CFG_ABORT;
        Self(words)
    }

    #[inline]
    pub const fn bypass(attr_types_override: bool) -> Self {
        let mut words = [0; 8];
        words[0] = STE_VALID | STE_CFG_BYPASS;
        words[1] = if attr_types_override {
            STE_SHCFG_INCOMING
        } else {
            0
        };
        Self(words)
    }

    pub const fn stage1(
        context_root: u64,
        cd_bits: u8,
        ats: bool,
        stall: bool,
        default_bypass: bool,
        attr_types_override: bool,
    ) -> Self {
        let mut words = [0; 8];
        // `cd_bits` is masked to the architected 5-bit S1CDMax field — an
        // over-wide value can never corrupt the shift out of word 0.
        words[0] = STE_VALID
            | STE_CFG_S1
            | (context_root & STE_S1CTXPTR_MASK)
            | ((cd_bits as u64 & STE_S1CDMAX_MASK) << STE_S1CDMAX_SHIFT);
        words[1] = if default_bypass {
            STE_S1DSS_BYPASS
        } else {
            STE_S1DSS_TERMINATE
        } | STE_S1C_WBRA
            | STE_S1CSH_INNER
            | if ats { STE_EATS_TRANS } else { 0 }
            | if stall { 0 } else { STE_S1STALLD }
            | if attr_types_override {
                STE_SHCFG_INCOMING
            } else {
                0
            };
        Self(words)
    }
}

/// A 64-byte SMMUv3 Context Descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextDescriptor(pub [u64; 8]);

impl ContextDescriptor {
    #[inline]
    pub const fn invalid() -> Self {
        Self([0; 8])
    }

    pub const fn stage1(
        root: u64,
        asid: u16,
        stall: bool,
        mair: u64,
        t0sz: u8,
        tg0: u8,
        ips: u8,
    ) -> Self {
        let mut words = [0; 8];
        // Active CPU T0SZ/TG0/IPS, WBWA inner/outer, inner-shareable, EPD1,
        // AArch64 long descriptor, and ASID tagging.
        words[0] = t0sz as u64
            | ((tg0 as u64) << 6)
            | (1 << 8)
            | (1 << 10)
            | (3 << 12)
            | CD_EPD1
            | CD_VALID
            | ((ips as u64) << 32)
            | CD_AA64
            | if stall { CD_STALL } else { 0 }
            | CD_R
            | CD_A
            | CD_ASET
            | ((asid as u64) << 48);
        words[1] = root & CD_TTB0_MASK;
        words[3] = mair;
        Self(words)
    }
}

/// A 16-byte SMMUv3 command-queue descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Command([u64; 2]);

impl Command {
    #[inline]
    const fn op(op: u8) -> Self {
        Self([op as u64, 0])
    }

    #[inline]
    pub const fn cfgi_all() -> Self {
        let mut cmd = Self::op(OP_CFGI_ALL);
        cmd.0[1] = 31;
        cmd
    }

    #[inline]
    pub const fn cfgi_ste(sid: u32) -> Self {
        let mut cmd = Self::op(OP_CFGI_STE);
        cmd.0[0] |= (sid as u64) << 32;
        cmd.0[1] = 1;
        cmd
    }

    /// `ssid` is truncated to the architected 20-bit SSID field
    /// ([`CMD_SSID_MASK`]) so it can never spill into StreamID.
    #[inline]
    pub const fn cfgi_cd(sid: u32, ssid: u32) -> Self {
        let mut cmd = Self::op(OP_CFGI_CD);
        cmd.0[0] |= ((ssid as u64 & CMD_SSID_MASK) << 12) | ((sid as u64) << 32);
        cmd.0[1] = 1;
        cmd
    }

    #[inline]
    pub const fn tlbi_global() -> Self {
        Self::op(OP_TLBI_NSNH_ALL)
    }

    #[inline]
    pub const fn tlbi_asid(asid: u16) -> Self {
        let mut cmd = Self::op(OP_TLBI_NH_ASID);
        cmd.0[0] |= (asid as u64) << 48;
        cmd
    }

    #[inline]
    pub const fn tlbi_va(asid: u16, va: u64) -> Self {
        let mut cmd = Self::op(OP_TLBI_NH_VA);
        cmd.0[0] |= (asid as u64) << 48;
        cmd.0[1] = 1 | (va & TLBI_VA_MASK);
        cmd
    }

    /// **Stage-1 range invalidation** — one command for a whole span of device
    /// virtual addresses, instead of one [`tlbi_va`](Self::tlbi_va) per page.
    ///
    /// This is the invalidation a device MMU actually needs. A device maps and
    /// unmaps *buffers*, not pages: a device firmware runtime region can be
    /// 64 MiB, a job's batch buffer is megabytes. Retiring one at 4 KiB
    /// granularity means 16384 commands through the CMDQ before the `CMD_SYNC`
    /// can even be posted; this covers the same span in one, and
    /// [`tlbi_va_sweep`](Self::tlbi_va_sweep) covers an arbitrary length in a
    /// handful.
    ///
    /// The invalidated span is `multiplier * 2^scale * 4 KiB`. `leaf_size`
    /// supplies the TTL hint, and its alignment is checked because the Arm
    /// architecture makes a mismatched TTL/address combination *unpredictable*
    /// — a silently wrong invalidation, which on a device MMU means the engine
    /// keeps translating through a page the host has already reused.
    pub fn tlbi_va_range(
        asid: u16,
        va: u64,
        leaf_size: PageSize,
        multiplier: u8,
        scale: u8,
    ) -> MmuResult<Self> {
        if !(1..=32).contains(&multiplier) || scale > 31 {
            return Err(MmuError::InvalidRange);
        }
        let ttl = ttl_of(leaf_size).ok_or(MmuError::InvalidGranule)?;
        if va & (leaf_size.bytes() as u64 - 1) != 0 {
            return Err(MmuError::InvalidAddress);
        }
        let span = u64::from(multiplier)
            .checked_shl(u32::from(scale))
            .and_then(|granules| granules.checked_mul(PageSize::Size4K.bytes() as u64))
            .ok_or(MmuError::AddressOverflow)?;
        if va.checked_add(span).is_none() {
            return Err(MmuError::AddressOverflow);
        }

        let mut cmd = Self::op(OP_TLBI_NH_VA);
        // NUM is encoded biased-by-one in [16:12]; SCALE in [24:20]; ASID in
        // [63:48] (where the stage-2 form carries VMID in [47:32] instead).
        cmd.0[0] |=
            ((u64::from(multiplier) - 1) << 12) | (u64::from(scale) << 20) | ((asid as u64) << 48);
        // Leaf=1, TTL in [9:8], and TG=01 for the 4 KiB translation granule.
        cmd.0[1] = 1 | (u64::from(ttl) << 8) | (1 << 10) | (va & TLBI_VA_MASK);
        Ok(cmd)
    }

    /// Cover `bytes` from `va` with the **fewest** stage-1 range invalidations,
    /// writing them into `out` and returning how many were produced.
    ///
    /// A single [`tlbi_va_range`](Self::tlbi_va_range) can only express
    /// `multiplier * 2^scale` granules with `multiplier <= 32`, so an arbitrary
    /// length needs a short sequence. Choosing `(multiplier, scale)` by hand is
    /// exactly the kind of arithmetic that goes quietly wrong — an under-covered
    /// range leaves live translations behind — so it belongs here once rather
    /// than in each driver.
    ///
    /// The sequence is **exact**: it never invalidates outside
    /// `va..va + bytes`. Both `va` and `bytes` must be `leaf_size`-aligned.
    /// [`MmuError::InvalidRange`] if `out` is too short — the caller learns the
    /// range was not fully covered rather than silently retiring part of it.
    ///
    /// Four commands are enough for any span a device will unmap (each command
    /// covers up to 32 × 2³¹ granules), so a `[Command; 8]` on the stack is a
    /// generous ceiling.
    pub fn tlbi_va_sweep(
        asid: u16,
        va: u64,
        bytes: u64,
        leaf_size: PageSize,
        out: &mut [Self],
    ) -> MmuResult<usize> {
        let leaf = leaf_size.bytes() as u64;
        if ttl_of(leaf_size).is_none() {
            return Err(MmuError::InvalidGranule);
        }
        if va & (leaf - 1) != 0 || bytes & (leaf - 1) != 0 {
            return Err(MmuError::InvalidAddress);
        }
        if bytes == 0 {
            return Ok(0);
        }
        va.checked_add(bytes).ok_or(MmuError::AddressOverflow)?;

        // The hardware counts 4 KiB translation granules (TG=01), so a leaf
        // page is `gpl` of them — a power of two, which is what lets the scale
        // for a leaf-granule count simply shift up by its log.
        let gpl = leaf / PageSize::Size4K.bytes() as u64;
        let shift = gpl.trailing_zeros();

        let mut remaining = bytes / leaf; // in leaf granules
        let mut at = va;
        let mut n = 0usize;
        while remaining > 0 {
            // Smallest scale that brings the count within the 32-entry
            // multiplier field, so each command covers as much as it can.
            let mut scale = 0u32;
            while (remaining >> scale) > 32 {
                scale += 1;
            }
            let multiplier = (remaining >> scale) as u8; // 1..=32
            let total_scale = scale + shift;
            if total_scale > 31 {
                return Err(MmuError::InvalidRange);
            }
            if n >= out.len() {
                return Err(MmuError::InvalidRange);
            }
            out[n] = Self::tlbi_va_range(asid, at, leaf_size, multiplier, total_scale as u8)?;
            n += 1;

            let covered = u64::from(multiplier) << scale; // leaf granules
            remaining -= covered;
            at += covered * leaf;
        }
        Ok(n)
    }

    /// `ssid` is truncated to the architected 20-bit SSID field
    /// ([`CMD_SSID_MASK`]) so it can never spill into StreamID.
    #[inline]
    pub const fn atc_all(sid: u32, ssid: Option<u32>) -> Self {
        let mut cmd = Self::op(OP_ATC_INV);
        cmd.0[0] |= (sid as u64) << 32;
        if let Some(ssid) = ssid {
            cmd.0[0] |= (1 << 11) | ((ssid as u64 & CMD_SSID_MASK) << 12);
        }
        cmd.0[1] = 52;
        cmd
    }

    /// Invalidate the smallest power-of-two 4 KiB ATC span covering the
    /// requested byte range. PCIe ATS requires both power-of-two sizing and size
    /// alignment, so a non-aligned range can conservatively cover a few
    /// neighbouring entries without degenerating to a whole-device flush.
    ///
    /// An `ssid` beyond the architected 20-bit SSID field is refused with
    /// [`MmuError::InvalidRange`] (this encoder already validates and returns
    /// [`MmuResult`], so a bad argument is an error, not a silent truncation).
    pub fn atc_range(sid: u32, ssid: Option<u32>, address: u64, bytes: usize) -> MmuResult<Self> {
        if let Some(ssid) = ssid
            && u64::from(ssid) > CMD_SSID_MASK
        {
            return Err(MmuError::InvalidRange);
        }
        let last = address
            .checked_add(
                u64::try_from(bytes)
                    .map_err(|_| MmuError::AddressOverflow)?
                    .checked_sub(1)
                    .ok_or(MmuError::InvalidRange)?,
            )
            .ok_or(MmuError::AddressOverflow)?;
        let first_page = address >> 12;
        let last_page = last >> 12;
        let differing = first_page ^ last_page;
        let log2_span = if differing == 0 {
            0
        } else {
            u64::BITS - differing.leading_zeros()
        };
        if log2_span >= 52 {
            return Ok(Self::atc_all(sid, ssid));
        }
        let span_mask = (1u64 << log2_span) - 1;
        let aligned = (first_page & !span_mask) << 12;

        let mut cmd = Self::op(OP_ATC_INV);
        cmd.0[0] |= (sid as u64) << 32;
        if let Some(ssid) = ssid {
            cmd.0[0] |= (1 << 11) | ((ssid as u64 & CMD_SSID_MASK) << 12);
        }
        cmd.0[1] = u64::from(log2_span) | (aligned & 0xffff_ffff_ffff_f000);
        Ok(cmd)
    }

    /// `ssid` is truncated to the architected 20-bit SSID field
    /// ([`CMD_SSID_MASK`]), matching the 9-bit mask already applied to `group`.
    #[inline]
    pub const fn pri_response(
        sid: u32,
        ssid: Option<u32>,
        group: u16,
        response: PageResponse,
    ) -> Self {
        let mut cmd = Self::op(OP_PRI_RESP);
        cmd.0[0] |= (sid as u64) << 32;
        if let Some(ssid) = ssid {
            cmd.0[0] |= (1 << 11) | ((ssid as u64 & CMD_SSID_MASK) << 12);
        }
        let response = match response {
            PageResponse::Invalid => 0,
            PageResponse::Failure => 1,
            PageResponse::Success => 2,
        };
        cmd.0[1] = (group as u64 & 0x1ff) | (response << 12);
        cmd
    }

    #[inline]
    pub const fn resume(sid: u32, stall_tag: u16, response: PageResponse) -> Self {
        let mut cmd = Self::op(OP_RESUME);
        let response = match response {
            PageResponse::Success => 1,
            PageResponse::Invalid | PageResponse::Failure => 2,
        };
        cmd.0[0] |= ((response as u64) << 12) | ((sid as u64) << 32);
        cmd.0[1] = stall_tag as u64;
        cmd
    }

    #[inline]
    pub const fn sync() -> Self {
        Self::op(OP_CMD_SYNC)
    }

    /// Little-endian 16-byte serialization for the command queue.
    #[inline]
    pub fn bytes(self) -> [u8; COMMAND_ENTRY_BYTES] {
        let mut out = [0; COMMAND_ENTRY_BYTES];
        out[..8].copy_from_slice(&self.0[0].to_le_bytes());
        out[8..].copy_from_slice(&self.0[1].to_le_bytes());
        out
    }
}

/// Decoded reason of an SMMUv3 fault event record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmmuV3FaultReason {
    BadStreamId,
    StreamEntryFetch,
    BadStreamEntry,
    StreamDisabled,
    BadSubstreamId,
    ContextDescriptorFetch,
    BadContextDescriptor,
    WalkExternalAbort,
    Translation,
    AddressSize,
    Access,
    Permission,
    VirtualMemoryStructureFetch,
    Unknown(u8),
}

/// A decoded SMMUv3 fault event record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmmuV3Fault {
    pub reason: SmmuV3FaultReason,
    pub sid: u32,
    pub ssid: Option<u32>,
    pub address: u64,
    /// `FetchAddr` (word 3, bits [51:3]) — for configuration and
    /// walk faults, the bus address of the FAILED FETCH itself (the
    /// STE/CD/PTE read), as opposed to `address`, the transaction's
    /// input address.  Zero when the event type does not provide one.
    pub fetch_addr: u64,
    /// `RnW` (word 1, bit 35): `true` for a read, `false` for a write.
    /// Meaningful for event classes whose architected record carries the
    /// transaction access type, including translation and permission faults.
    pub read: bool,
    /// `InD` (word 1, bit 34): `true` for an instruction fetch, `false` for
    /// data. The architecture defines writes as data accesses regardless of
    /// this bit, so consumers must inspect `read` first.
    pub instruction: bool,
    pub stage2: bool,
    pub stalled: bool,
    pub stall_tag: u16,
}

impl SmmuV3Fault {
    pub const fn decode(words: [u64; 4]) -> Self {
        let reason = match (words[0] & 0xff) as u8 {
            0x02 => SmmuV3FaultReason::BadStreamId,
            0x03 => SmmuV3FaultReason::StreamEntryFetch,
            0x04 => SmmuV3FaultReason::BadStreamEntry,
            0x06 => SmmuV3FaultReason::StreamDisabled,
            0x08 => SmmuV3FaultReason::BadSubstreamId,
            0x09 => SmmuV3FaultReason::ContextDescriptorFetch,
            0x0a => SmmuV3FaultReason::BadContextDescriptor,
            0x0b => SmmuV3FaultReason::WalkExternalAbort,
            0x10 => SmmuV3FaultReason::Translation,
            0x11 => SmmuV3FaultReason::AddressSize,
            0x12 => SmmuV3FaultReason::Access,
            0x13 => SmmuV3FaultReason::Permission,
            0x25 => SmmuV3FaultReason::VirtualMemoryStructureFetch,
            other => SmmuV3FaultReason::Unknown(other),
        };
        let ssid = if (words[0] & (1 << 11)) != 0 {
            Some(((words[0] >> 12) & 0xfffff) as u32)
        } else {
            None
        };
        Self {
            reason,
            sid: (words[0] >> 32) as u32,
            ssid,
            address: words[2],
            fetch_addr: words[3] & 0x000f_ffff_ffff_fff8,
            read: (words[1] & (1 << 35)) != 0,
            instruction: (words[1] & (1 << 34)) != 0,
            stage2: (words[1] & (1 << 39)) != 0,
            stalled: (words[1] & (1 << 31)) != 0,
            stall_tag: (words[1] & 0xffff) as u16,
        }
    }
}

/// A decoded SMMUv3 Page Request Interface record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmmuV3PriEntry {
    pub sid: u32,
    pub ssid: Option<u32>,
    pub address: u64,
    pub access: AccessFlags,
    pub group: u16,
    pub last: bool,
}

impl SmmuV3PriEntry {
    pub const fn decode(words: [u64; 2]) -> Self {
        let mut access = AccessFlags::empty();
        if (words[0] & (1 << 60)) != 0 {
            access = access.union(AccessFlags::READ);
        }
        if (words[0] & (1 << 61)) != 0 {
            access = access.union(AccessFlags::WRITE);
        }
        if (words[0] & (1 << 59)) != 0 {
            access = access.union(AccessFlags::EXECUTE);
        }
        Self {
            sid: words[0] as u32,
            ssid: if (words[0] & (1 << 63)) != 0 {
                Some(((words[0] >> 32) & 0xfffff) as u32)
            } else {
                None
            },
            address: words[1] & 0xffff_ffff_ffff_f000,
            access,
            group: (words[1] & 0x1ff) as u16,
            last: (words[0] & (1 << 62)) != 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cfgi_ste_places_sid_in_architected_field() {
        let bytes = Command::cfgi_ste(0x1234_5678).bytes();
        assert_eq!(bytes[0], OP_CFGI_STE);
        assert_eq!(
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            0x1234_5678
        );
        assert_eq!(bytes[8], 1);
    }

    #[test]
    fn atc_range_rounds_to_smallest_aligned_power_of_two_span() {
        let command = Command::atc_range(0x30, Some(7), 0x7000, 0x4000).unwrap();
        assert_eq!(command.0[0] >> 32, 0x30);
        assert_eq!((command.0[0] >> 12) & 0xfffff, 7);
        // Pages 7..10 need the aligned 16-page span starting at page 0.
        assert_eq!(command.0[1] & 0x3f, 4);
        assert_eq!(command.0[1] & 0xffff_ffff_ffff_f000, 0);
    }

    #[test]
    fn stage1_range_tlbi_encodes_multiplier_scale_asid_and_leaf_hint() {
        // 3 × 2⁴ granules from a 1 GiB-aligned base, hinted at the 1 GiB level.
        let command = Command::tlbi_va_range(7, 0x4000_0000, PageSize::Size1G, 3, 4).unwrap();
        assert_eq!((command.0[0] >> 12) & 0x1f, 2, "NUM is biased by one");
        assert_eq!((command.0[0] >> 20) & 0x1f, 4, "SCALE");
        // The ASID rides [63:48] — where the stage-2 form carried VMID [47:32].
        assert_eq!(command.0[0] >> 48, 7);
        assert_eq!((command.0[1] >> 8) & 0x3, 1, "TTL = 1 for a 1 GiB leaf");
        assert_eq!((command.0[1] >> 10) & 0x3, 1, "TG = 01, the 4 KiB granule");
        assert_eq!(command.0[1] & 1, 1, "leaf");
        assert_eq!(command.0[1] & TLBI_VA_MASK, 0x4000_0000);
    }

    #[test]
    fn stage1_range_tlbi_refuses_out_of_range_arguments() {
        // A TTL/address mismatch is *unpredictable* per the architecture, so a
        // misaligned base must be refused rather than encoded.
        assert_eq!(
            Command::tlbi_va_range(1, 0x20_1000, PageSize::Size2M, 1, 0),
            Err(MmuError::InvalidAddress)
        );
        // NUM is a 5-bit biased field: 0 and 33 are both unencodable.
        assert_eq!(
            Command::tlbi_va_range(1, 0x1000, PageSize::Size4K, 0, 0),
            Err(MmuError::InvalidRange)
        );
        assert_eq!(
            Command::tlbi_va_range(1, 0x1000, PageSize::Size4K, 33, 0),
            Err(MmuError::InvalidRange)
        );
        assert_eq!(
            Command::tlbi_va_range(1, 0x1000, PageSize::Size4K, 1, 32),
            Err(MmuError::InvalidRange)
        );
    }

    /// The sweep must cover the requested span **exactly** — an under-covered
    /// range leaves live device translations behind, which is the failure this
    /// helper exists to prevent.
    #[test]
    fn the_sweep_covers_a_span_exactly_and_compactly() {
        /// Total 4 KiB granules a command claims: (NUM + 1) << SCALE.
        fn span_granules(c: &Command) -> u64 {
            let num = ((c.0[0] >> 12) & 0x1f) + 1;
            let scale = (c.0[0] >> 20) & 0x1f;
            num << scale
        }
        let mut out = [Command::sync(); 8];

        // A 64 MiB firmware runtime region — 16384 pages, which as single-page
        // invalidations would be 16384 CMDQ entries.
        let n = Command::tlbi_va_sweep(3, 0x8000_0000, 64 << 20, PageSize::Size4K, &mut out)
            .expect("sweep");
        assert!(n <= 2, "64 MiB should take one or two commands, took {n}");
        let covered: u64 = out[..n].iter().map(span_granules).sum();
        assert_eq!(covered, (64 << 20) / 4096, "span not covered exactly");

        // A deliberately awkward length: 101 pages is not a power of two and
        // exceeds the 32-entry multiplier, so it must decompose.
        let n = Command::tlbi_va_sweep(3, 0x9000_0000, 101 * 4096, PageSize::Size4K, &mut out)
            .expect("sweep");
        let covered: u64 = out[..n].iter().map(span_granules).sum();
        assert_eq!(covered, 101, "awkward span not covered exactly");
        // ...and the commands tile the range contiguously, in order.
        let mut at = 0x9000_0000u64;
        for c in &out[..n] {
            assert_eq!(c.0[1] & TLBI_VA_MASK, at, "commands must tile in order");
            at += span_granules(c) * 4096;
        }
        assert_eq!(at, 0x9000_0000 + 101 * 4096);
    }

    #[test]
    fn the_sweep_refuses_rather_than_partially_covering() {
        let mut one = [Command::sync(); 1];
        // 101 pages needs more than one command; a short buffer must error,
        // never silently retire only the first chunk.
        assert_eq!(
            Command::tlbi_va_sweep(3, 0x9000_0000, 101 * 4096, PageSize::Size4K, &mut one),
            Err(MmuError::InvalidRange)
        );
        // Misaligned base or length is refused outright.
        let mut out = [Command::sync(); 8];
        assert_eq!(
            Command::tlbi_va_sweep(3, 0x1000, 0x1000, PageSize::Size2M, &mut out),
            Err(MmuError::InvalidAddress)
        );
        // An empty span is a no-op, not an error.
        assert_eq!(
            Command::tlbi_va_sweep(3, 0x8000_0000, 0, PageSize::Size4K, &mut out),
            Ok(0)
        );
    }

    /// A 2 MiB leaf still counts in 4 KiB granules on the wire (TG = 01), so
    /// the scale must absorb the 512× — getting this wrong under-invalidates
    /// by a factor of 512.
    #[test]
    fn a_large_leaf_size_scales_into_four_kib_granules() {
        let mut out = [Command::sync(); 8];
        let n = Command::tlbi_va_sweep(1, 0x4000_0000, 4 << 20, PageSize::Size2M, &mut out)
            .expect("sweep");
        assert_eq!(n, 1, "two 2 MiB leaves fit one command");
        let num = ((out[0].0[0] >> 12) & 0x1f) + 1;
        let scale = (out[0].0[0] >> 20) & 0x1f;
        assert_eq!(num << scale, 1024, "4 MiB is 1024 four-KiB granules");
        assert_eq!((out[0].0[1] >> 8) & 0x3, 2, "TTL = 2 for a 2 MiB leaf");
    }

    #[test]
    fn ssid_encoders_bound_the_20_bit_field() {
        // Masked sites: an over-wide ssid must not spill into StreamID [63:32].
        let over = (1 << 20) | 5;
        for cmd in [
            Command::cfgi_cd(0x77, over),
            Command::atc_all(0x77, Some(over)),
            Command::pri_response(0x77, Some(over), 3, PageResponse::Success),
        ] {
            assert_eq!((cmd.0[0] >> 12) & 0xfffff, 5, "SSID field truncated");
            assert_eq!((cmd.0[0] >> 32) as u32, 0x77, "StreamID intact");
        }
        // Validated site: atc_range already returns MmuResult, so it refuses.
        assert_eq!(
            Command::atc_range(0x77, Some(over), 0x1000, 0x1000),
            Err(MmuError::InvalidRange)
        );
    }

    #[test]
    fn stream_entry_masks_s1cdmax_to_its_five_bit_field() {
        let ste = StreamEntry::stage1(0x4_0000, 0x3f, false, false, false, false);
        // 0x3f masked to 0x1f — nothing shifted out of word 0, VALID/CFG intact.
        assert_eq!(ste.0[0] >> STE_S1CDMAX_SHIFT, 0x1f);
        assert_eq!(ste.0[0] & 0xf, STE_VALID | STE_CFG_S1);
    }

    #[test]
    fn pri_record_preserves_group_and_access() {
        let words = [
            0x2345 | (0xabc_u64 << 32) | (1 << 63) | (1 << 62) | (1 << 60),
            0xdead_beef_1000 | 0x1a,
        ];
        let entry = SmmuV3PriEntry::decode(words);
        assert_eq!(entry.sid, 0x2345);
        assert_eq!(entry.ssid, Some(0xabc));
        assert_eq!(entry.group, 0x1a);
        assert!(entry.last);
        assert!(entry.access.contains(AccessFlags::READ));
    }

    #[test]
    fn fault_record_preserves_the_architected_access_class() {
        let data_write = SmmuV3Fault::decode([0x13, 0, 0x1000, 0]);
        assert!(!data_write.read);
        assert!(!data_write.instruction);

        let data_read = SmmuV3Fault::decode([0x13, 1 << 35, 0x2000, 0]);
        assert!(data_read.read);
        assert!(!data_read.instruction);

        let instruction_read = SmmuV3Fault::decode([0x13, (1 << 35) | (1 << 34), 0x3000, 0]);
        assert!(instruction_read.read);
        assert!(instruction_read.instruction);
    }

    #[test]
    fn walk_external_abort_preserves_fetch_address_and_access_class() {
        let fault = SmmuV3Fault::decode([0x0b, (1 << 35) | (1 << 34), 0x1234_5000, 0xabcde_fff8]);

        assert_eq!(fault.reason, SmmuV3FaultReason::WalkExternalAbort);
        assert_eq!(fault.address, 0x1234_5000);
        assert_eq!(fault.fetch_addr, 0xabcde_fff8);
        assert!(fault.read);
        assert!(fault.instruction);
    }
}
