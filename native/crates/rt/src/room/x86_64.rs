//! The x86-64 room — the dormant x87/MMX bank as the zeroth storage
//! tier (`res/SPMD.md` §13.6).
//!
//! Sound because the build makes it so: every x86 target spec carries
//! `-mmx` (the crate-wide reservation), Rust emits no x87 arithmetic
//! (no `f80` exists), and the one C build carries `-mno-x87` — so the
//! tag-word/EMMS hazard, which only x87 *arithmetic* awakens, is
//! structurally retired, and these eight registers belong to the room
//! alone.  The accessors name the registers EXPLICITLY in `asm!` — a
//! fixed-register `movq` is baseline x86-64 encoding and needs no
//! feature re-admission (rustc's `target_feature` whitelist has no
//! `mmx` entry at all; the crate-wide spec reservation only bars the
//! ALLOCATOR from the bank, which is exactly the tenancy wanted).
//! MMX↔AVX carries no transition penalty
//! (separate physical file; `vzeroupper` concerns XMM/YMM only), so v3
//! userland is unaffected.
//!
//! Preemption-survival is the landed FPU capability model's: the MM
//! bank is XSAVE component 0, round-tripped per task at every
//! kernel→user return — proven on metal by
//! [`validate::register_room`](crate::validate).

use super::{RoomBank, RoomSlot};

/// The MM bank — zero-sized: the state IS the registers.
pub(crate) struct MmBank;

impl RoomBank for MmBank {
    fn name() -> &'static str {
        "mmx"
    }

    fn new() -> Self {
        // A fresh room must read vacant: zero the bank (the thread may
        // inherit whatever the last tenant of component 0 left).
        let mut bank = Self;
        for slot in [
            RoomSlot::Sched,
            RoomSlot::Mask,
            RoomSlot::Reconverge,
            RoomSlot::Mail0,
            RoomSlot::Mail1,
            RoomSlot::Mail2,
            RoomSlot::Mail3,
            RoomSlot::Broadcast,
        ] {
            bank.store(slot, 0);
        }
        bank
    }

    #[inline]
    fn store(&mut self, slot: RoomSlot, value: u64) {
        store_mm(slot as u8, value);
    }

    #[inline]
    fn load(&self, slot: RoomSlot) -> u64 {
        load_mm(slot as u8)
    }
}

/// `movq mmN, value` — one register store, selected by the slot.  The
/// register name must be a bare literal in both the template and the
/// clobber (`asm!` accepts no interpolation there), hence the eight
/// spelled-out arms.
fn store_mm(slot: u8, value: u64) {
    use core::arch::asm;
    // SAFETY (each arm): the MM bank is architecturally present on
    // every x86-64 part; the crate-wide `-mmx` reservation plus
    // unemittable x87 arithmetic make it exclusively the room's, so a
    // fixed-register `movq` touches state nothing else observes.
    unsafe {
        match slot {
            0 => asm!("movq mm0, {v}", v = in(reg) value, out("mm0") _, options(nomem, nostack, preserves_flags)),
            1 => asm!("movq mm1, {v}", v = in(reg) value, out("mm1") _, options(nomem, nostack, preserves_flags)),
            2 => asm!("movq mm2, {v}", v = in(reg) value, out("mm2") _, options(nomem, nostack, preserves_flags)),
            3 => asm!("movq mm3, {v}", v = in(reg) value, out("mm3") _, options(nomem, nostack, preserves_flags)),
            4 => asm!("movq mm4, {v}", v = in(reg) value, out("mm4") _, options(nomem, nostack, preserves_flags)),
            5 => asm!("movq mm5, {v}", v = in(reg) value, out("mm5") _, options(nomem, nostack, preserves_flags)),
            6 => asm!("movq mm6, {v}", v = in(reg) value, out("mm6") _, options(nomem, nostack, preserves_flags)),
            _ => asm!("movq mm7, {v}", v = in(reg) value, out("mm7") _, options(nomem, nostack, preserves_flags)),
        }
    }
}

/// `movq value, mmN` — one register load, selected by the slot.
fn load_mm(slot: u8) -> u64 {
    use core::arch::asm;
    let value: u64;
    // SAFETY (each arm): as `store_mm`; a read clobbers no `mmN` state.
    unsafe {
        match slot {
            0 => asm!("movq {v}, mm0", v = out(reg) value, options(nomem, nostack, preserves_flags)),
            1 => asm!("movq {v}, mm1", v = out(reg) value, options(nomem, nostack, preserves_flags)),
            2 => asm!("movq {v}, mm2", v = out(reg) value, options(nomem, nostack, preserves_flags)),
            3 => asm!("movq {v}, mm3", v = out(reg) value, options(nomem, nostack, preserves_flags)),
            4 => asm!("movq {v}, mm4", v = out(reg) value, options(nomem, nostack, preserves_flags)),
            5 => asm!("movq {v}, mm5", v = out(reg) value, options(nomem, nostack, preserves_flags)),
            6 => asm!("movq {v}, mm6", v = out(reg) value, options(nomem, nostack, preserves_flags)),
            _ => asm!("movq {v}, mm7", v = out(reg) value, options(nomem, nostack, preserves_flags)),
        }
    }
    value
}
