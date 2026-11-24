//! The **xHCI register surface**, hand-built on the [`Regs`](kore_driver::seam::Regs)
//! seam.  Small and flat: capability registers decode the geometry once
//! ([`Caps`]); operational / runtime / doorbell / port registers are reached
//! by computed byte offset.  Layouts are xHCI 1.2 §5 (cross-checked against
//! the vendored crate's register module, which we do not link — see the
//! crate-level note on why the accessor half is unused).

use kore_driver::seam::Regs;

// ── Capability registers (BAR0 + 0) ─────────────────────────────────────
/// `CAPLENGTH` (u8 @ 0x00) — the operational-register offset; `HCIVERSION`
/// (u16 @ 0x02) shares the dword.
pub const CAPLENGTH: usize = 0x00;
/// `HCSPARAMS1` (u32 @ 0x04): MaxSlots[7:0], MaxIntrs[18:8], MaxPorts[31:24].
pub const HCSPARAMS1: usize = 0x04;
/// `HCSPARAMS2` (u32 @ 0x08): ERST Max[7:4], Max Scratchpad Hi[25:21],
/// Max Scratchpad Lo[31:27].
pub const HCSPARAMS2: usize = 0x08;
/// `HCCPARAMS1` (u32 @ 0x10): AC64[0], CSZ[2], xECP[31:16].
pub const HCCPARAMS1: usize = 0x10;
/// `DBOFF` (u32 @ 0x14): doorbell array offset [31:2].
pub const DBOFF: usize = 0x14;
/// `RTSOFF` (u32 @ 0x18): runtime register space offset [31:5].
pub const RTSOFF: usize = 0x18;

// ── Operational registers (BAR0 + CAPLENGTH) ────────────────────────────
/// `USBCMD` (u32 @ op+0x00): RS[0], HCRST[1], INTE[2], HSEE[3].
pub const USBCMD: usize = 0x00;
/// `USBSTS` (u32 @ op+0x04): HCH[0], HSE[2], EINT[3], PCD[4], CNR[11], HCE[12].
pub const USBSTS: usize = 0x04;
/// `PAGESIZE` (u32 @ op+0x08): supported page sizes (bit n ⇒ 2^(n+12)).
pub const PAGESIZE: usize = 0x08;
/// `CRCR` (u64 @ op+0x18): RCS[0], CS[1], CA[2], CRR[3], pointer[63:6].
pub const CRCR: usize = 0x18;
/// `DCBAAP` (u64 @ op+0x30): Device Context Base Address Array Pointer.
pub const DCBAAP: usize = 0x30;
/// `CONFIG` (u32 @ op+0x38): MaxSlotsEn[7:0].
pub const CONFIG: usize = 0x38;
/// The Port Register Set base (op+0x400); PORTSC[n] = op + 0x400 + n*0x10.
pub const PORT_BASE: usize = 0x400;
/// `PORTSC` offset within a port register set.
pub const PORTSC: usize = 0x00;

// ── USBCMD / USBSTS / CRCR bits ─────────────────────────────────────────
/// USBCMD Run/Stop.
pub const USBCMD_RS: u32 = 1 << 0;
/// USBCMD Host Controller Reset.
pub const USBCMD_HCRST: u32 = 1 << 1;
/// USBCMD Interrupter Enable.
pub const USBCMD_INTE: u32 = 1 << 2;
/// USBSTS Host Controller Halted.
pub const USBSTS_HCH: u32 = 1 << 0;
/// USBSTS Controller Not Ready.
pub const USBSTS_CNR: u32 = 1 << 11;
/// USBSTS Host Controller Error.
pub const USBSTS_HCE: u32 = 1 << 12;
/// USBSTS Event Interrupt (RW1C).
pub const USBSTS_EINT: u32 = 1 << 3;
/// CRCR Ring Cycle State (the producer cycle the ring starts at).
pub const CRCR_RCS: u64 = 1 << 0;
/// CRCR Command Abort.
pub const CRCR_CA: u64 = 1 << 2;
/// CRCR Command Ring Running.
pub const CRCR_CRR: u64 = 1 << 3;

// ── Interrupter register set (runtime + 0x20 + i*0x20) ───────────────────
/// First interrupter register set offset within the runtime block.
pub const INTR_BASE: usize = 0x20;
/// One interrupter register set is 0x20 bytes.
pub const INTR_STRIDE: usize = 0x20;
/// `IMAN` (u32 @ intr+0x00): IP[0] (RW1C), IE[1].
pub const IMAN: usize = 0x00;
/// `IMOD` (u32 @ intr+0x04): interrupt moderation interval (250 ns units).
pub const IMOD: usize = 0x04;
/// `ERSTSZ` (u32 @ intr+0x08): Event Ring Segment Table size (entries).
pub const ERSTSZ: usize = 0x08;
/// `ERSTBA` (u64 @ intr+0x10): Event Ring Segment Table Base Address.
pub const ERSTBA: usize = 0x10;
/// `ERDP` (u64 @ intr+0x18): Event Ring Dequeue Pointer; EHB[3] (RW1C).
pub const ERDP: usize = 0x18;
/// IMAN Interrupt Pending (RW1C).
pub const IMAN_IP: u32 = 1 << 0;
/// IMAN Interrupt Enable.
pub const IMAN_IE: u32 = 1 << 1;
/// ERDP Event Handler Busy (RW1C) — written set to acknowledge a drain.
pub const ERDP_EHB: u64 = 1 << 3;

/// The controller's decoded geometry — read once at bring-up from the
/// capability registers.
#[derive(Clone, Copy, Debug)]
pub struct Caps {
    /// Operational register base = `CAPLENGTH`.
    pub op: usize,
    /// Runtime register base = `RTSOFF & !0x1f`.
    pub rt: usize,
    /// Doorbell array base = `DBOFF & !0x3`.
    pub db: usize,
    /// Number of device slots the controller supports (`MaxSlots`).
    pub max_slots: u8,
    /// Number of root-hub ports (`MaxPorts`).
    pub max_ports: u8,
    /// Number of interrupters (`MaxIntrs`).
    pub max_intrs: u16,
    /// Context size: `true` ⇒ 64-byte contexts (HCCPARAMS1.CSZ), else 32.
    pub csz64: bool,
    /// 64-bit addressing capable (HCCPARAMS1.AC64).
    pub ac64: bool,
    /// Max scratchpad buffers the controller wants (split hi/lo field).
    pub max_scratchpads: u32,
    /// Event Ring Segment Table max size (log2 entries, HCSPARAMS2).
    pub erst_max: u8,
    /// xHCI Extended Capabilities pointer (dword offset from BAR0), 0 = none.
    pub xecp: u16,
}

/// Decode `HCSPARAMS2`'s Max Scratchpad Buffers (split hi/lo field:
/// Hi[25:21] `<< 5 |` Lo[31:27]).  Public and freestanding so a caller
/// holding only the raw dword shares this decode instead of re-deriving
/// the split (the register-bits-from-the-spec rule: one decode, cited).
pub fn max_scratchpads(hcsparams2: u32) -> u32 {
    let sp_hi = (hcsparams2 >> 21) & 0x1f;
    let sp_lo = (hcsparams2 >> 27) & 0x1f;
    (sp_hi << 5) | sp_lo
}

/// `PORTSC` Current Connect Status — something is plugged into this port.
pub const PORTSC_CCS: u32 = 1 << 0;

/// Whether this controller's registers actually **decode**.
///
/// A function can be present in config space yet unreadable through its
/// BAR — Lunar Lake's Type-C xHC at 00:0d.0 sleeps in D3 until PD
/// negotiation, and it enumerates *before* the standalone PCH controller
/// at 00:14.0 that actually hosts the ports.  A driver bound to the
/// sleeping one burns its whole bring-up budget polling all-ones: a
/// two-minute mute stall on metal, while the machine's keyboard sits on
/// the live controller one match later.
///
/// `CAPLENGTH`'s low byte covers the complete capability-register header and
/// is dword-aligned on a live controller; an all-ones read is invalid too.
/// Those facts make this answerable at all — and why it
/// lives **here** rather than in whatever picks the device.  The test is
/// meaningful only where the register semantics are known, so it belongs
/// to the class that knows them; a chooser that ran it would have to
/// learn xHCI to do so.
fn cap_length(cap0: u32) -> Option<usize> {
    let cap_len = (cap0 & 0xff) as usize;
    (cap0 != u32::MAX
        && cap_len >= RTSOFF + core::mem::size_of::<u32>()
        && cap_len.is_multiple_of(core::mem::align_of::<u32>()))
    .then_some(cap_len)
}

pub fn registers_decode(regs: &impl Regs) -> bool {
    cap_length(regs.read32(CAPLENGTH)).is_some()
}

/// How many root-hub ports report a **connected** device, out of
/// `MaxPorts` — a read-only sweep of `PORTSC.CCS` (the register's
/// write-1-to-clear bits are unaffected by reads).
///
/// The third tier of choosing a controller, and the one liveness alone
/// cannot supply: once anything has powered Lunar Lake's Type-C xHC it
/// decodes perfectly while hosting zero devices, and the keyboard and the
/// boot stick sit on the PCH controller one match later.  KVM never
/// exposes it — QEMU models exactly one controller, with the emulated
/// keyboard on it — so this is a metal-only distinction that a test rig
/// will always report as unnecessary.
///
/// Returns `(connected, max_ports)` so a caller can log both: on a
/// serial-less machine that one line answers *"is the keyboard even on
/// the controller kore drives?"* without a flash cycle of guessing. `None`
/// rejects device-reported geometry whose register walk would leave BAR0.
pub fn connected_ports(regs: &impl Regs, bar_len: usize) -> Option<(usize, usize)> {
    let cap_end = RTSOFF.checked_add(core::mem::size_of::<u32>())?;
    if bar_len < cap_end {
        return None;
    }
    // Decode CAPLENGTH exactly once. A power-managed controller can change
    // decode state between MMIO reads; validating one sample and computing an
    // offset from another could otherwise turn all-ones into an unaligned
    // register access.
    let cap0 = regs.read32(CAPLENGTH);
    let op = cap_length(cap0)?;
    let hcs1 = regs.read32(HCSPARAMS1);
    let max_ports = (hcs1 >> 24) as usize;
    let ports_end = op
        .checked_add(PORT_BASE)?
        .checked_add(max_ports.checked_mul(0x10)?)?;
    if ports_end > bar_len {
        return None;
    }
    let mut connected = 0usize;
    for i in 0..max_ports {
        let portsc = regs.read32(op + PORT_BASE + i * 0x10 + PORTSC);
        if portsc & PORTSC_CCS != 0 {
            connected += 1;
        }
    }
    Some((connected, max_ports))
}

impl Caps {
    /// Decode the capability registers over `regs`.
    pub fn read(regs: &impl Regs) -> Self {
        let cap0 = regs.read32(CAPLENGTH);
        let op = (cap0 & 0xff) as usize;

        let hcs1 = regs.read32(HCSPARAMS1);
        let max_slots = (hcs1 & 0xff) as u8;
        let max_intrs = ((hcs1 >> 8) & 0x7ff) as u16;
        let max_ports = ((hcs1 >> 24) & 0xff) as u8;

        let hcs2 = regs.read32(HCSPARAMS2);
        let erst_max = ((hcs2 >> 4) & 0xf) as u8;
        let max_scratchpads = max_scratchpads(hcs2);

        let hcc1 = regs.read32(HCCPARAMS1);
        let ac64 = hcc1 & 1 != 0;
        let csz64 = hcc1 & (1 << 2) != 0;
        let xecp = ((hcc1 >> 16) & 0xffff) as u16;

        let rt = (regs.read32(RTSOFF) & !0x1f) as usize;
        let db = (regs.read32(DBOFF) & !0x3) as usize;

        Self {
            op,
            rt,
            db,
            max_slots,
            max_ports,
            max_intrs,
            csz64,
            ac64,
            max_scratchpads,
            erst_max,
            xecp,
        }
    }

    /// The byte size of one context entry (32 or 64).
    pub fn ctx_bytes(&self) -> usize {
        if self.csz64 { 64 } else { 32 }
    }

    /// Absolute offset of an operational register.
    pub fn op(&self, off: usize) -> usize {
        self.op + off
    }

    /// Absolute offset of `PORTSC` for root-hub port `n` (0-based here;
    /// port 1 in spec terms is `n = 0`).
    pub fn portsc(&self, n: u8) -> usize {
        self.op + PORT_BASE + (n as usize) * 0x10 + PORTSC
    }

    /// Absolute offset of register `off` in interrupter `i`'s register set.
    pub fn intr(&self, i: u16, off: usize) -> usize {
        self.rt + INTR_BASE + (i as usize) * INTR_STRIDE + off
    }

    /// Absolute offset of doorbell `n` (0 = command ring / host; n = slot id).
    pub fn doorbell(&self, n: u8) -> usize {
        self.db + (n as usize) * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::{Cell, RefCell};

    /// A flat register file for decode tests.
    struct Flat(RefCell<[u32; 16]>);
    impl Regs for Flat {
        fn read32(&self, off: usize) -> u32 {
            self.0.borrow()[off / 4]
        }
        fn write32(&self, off: usize, val: u32) {
            self.0.borrow_mut()[off / 4] = val;
        }
    }

    /// A BAR-sized register file for root-hub port discovery tests.
    struct PortFlat(RefCell<[u32; 512]>);
    impl Regs for PortFlat {
        fn read32(&self, off: usize) -> u32 {
            self.0.borrow()[off / 4]
        }
        fn write32(&self, off: usize, val: u32) {
            self.0.borrow_mut()[off / 4] = val;
        }
    }

    /// A controller that stops decoding after the first CAPLENGTH sample.
    struct PowerGatingCap {
        file: RefCell<[u32; 512]>,
        cap_reads: Cell<usize>,
    }
    impl Regs for PowerGatingCap {
        fn read32(&self, off: usize) -> u32 {
            if off == CAPLENGTH {
                let reads = self.cap_reads.get();
                self.cap_reads.set(reads + 1);
                if reads != 0 {
                    return u32::MAX;
                }
            }
            self.file.borrow()[off / 4]
        }
        fn write32(&self, off: usize, val: u32) {
            self.file.borrow_mut()[off / 4] = val;
        }
    }

    #[test]
    fn decodes_geometry_and_split_scratchpad_field() {
        let mut file = [0u32; 16];
        // CAPLENGTH = 0x40, HCIVERSION 0x0110 in the high half.
        file[CAPLENGTH / 4] = 0x0110_0040;
        // HCSPARAMS1: MaxSlots 64, MaxIntrs 16, MaxPorts 8.
        file[HCSPARAMS1 / 4] = (8 << 24) | (16 << 8) | 64;
        // HCSPARAMS2: ERST Max 4; scratchpad hi=0b00001, lo=0b00010 ⇒ 34.
        file[HCSPARAMS2 / 4] = (4 << 4) | (1 << 21) | (2 << 27);
        // HCCPARAMS1: AC64=1, CSZ=1, xECP=0x0040.
        file[HCCPARAMS1 / 4] = 1 | (1 << 2) | (0x0040 << 16);
        file[DBOFF / 4] = 0x0000_2000;
        file[RTSOFF / 4] = 0x0000_1000;

        let caps = Caps::read(&Flat(RefCell::new(file)));
        assert_eq!(caps.op, 0x40);
        assert_eq!(caps.max_slots, 64);
        assert_eq!(caps.max_intrs, 16);
        assert_eq!(caps.max_ports, 8);
        assert_eq!(caps.erst_max, 4);
        assert_eq!(caps.max_scratchpads, (1 << 5) | 2, "split hi/lo field");
        assert!(caps.csz64, "64-byte contexts");
        assert_eq!(caps.ctx_bytes(), 64);
        assert!(caps.ac64);
        assert_eq!(caps.xecp, 0x0040);
        assert_eq!(caps.rt, 0x1000);
        assert_eq!(caps.db, 0x2000);
        // Interrupter 0's ERDP, doorbell 3, portsc for port 1 (n=0).
        assert_eq!(caps.intr(0, ERDP), 0x1000 + 0x20 + 0x18);
        assert_eq!(caps.doorbell(3), 0x2000 + 12);
        assert_eq!(caps.portsc(0), 0x40 + 0x400);
    }

    #[test]
    fn register_liveness_rejects_power_gated_reads() {
        let regs = Flat(RefCell::new([0; 16]));
        assert!(!registers_decode(&regs));
        regs.write32(CAPLENGTH, 0x0110_0000);
        assert!(!registers_decode(&regs), "CAPLENGTH is the low byte");
        regs.write32(CAPLENGTH, u32::MAX);
        assert!(!registers_decode(&regs));
        regs.write32(CAPLENGTH, 0x0110_0040);
        assert!(registers_decode(&regs));
    }

    #[test]
    fn connected_ports_walks_only_valid_reported_geometry() {
        let regs = PortFlat(RefCell::new([0; 512]));
        let op = 0x40;
        regs.write32(CAPLENGTH, 0x0110_0000 | op as u32);
        regs.write32(HCSPARAMS1, 2 << 24);
        regs.write32(op + PORT_BASE, 0);
        regs.write32(op + PORT_BASE + 0x10, PORTSC_CCS);

        let exact_len = op + PORT_BASE + 2 * 0x10;
        assert_eq!(connected_ports(&regs, exact_len), Some((1, 2)));
        assert_eq!(
            connected_ports(&regs, exact_len - 1),
            None,
            "device-reported ports outside BAR0 must be refused before the walk"
        );
    }

    #[test]
    fn connected_ports_uses_the_caplength_sample_it_validates() {
        let mut file = [0; 512];
        let op = 0x40;
        file[CAPLENGTH / 4] = 0x0110_0000 | op as u32;
        file[HCSPARAMS1 / 4] = 1 << 24;
        file[(op + PORT_BASE) / 4] = PORTSC_CCS;
        let regs = PowerGatingCap {
            file: RefCell::new(file),
            cap_reads: Cell::new(0),
        };

        assert_eq!(connected_ports(&regs, op + PORT_BASE + 0x10), Some((1, 1)),);
        assert_eq!(
            regs.cap_reads.get(),
            1,
            "a second sample could change decode state before offset use",
        );
    }
}
