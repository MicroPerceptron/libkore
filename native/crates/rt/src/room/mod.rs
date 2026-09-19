//! **The register room** (`res/SPMD.md` §13.6, R7) — one worker's
//! private eight-slot bank for same-core inter-fiber state, and the quad
//! per-lane mail (the ITC) that rides its spare slots.
//!
//! On x86-64 the dormant x87/MMX bank backs it: the build forbids `-mmx`
//! crate-wide and Rust emits no x87 arithmetic, so `MM0..MM7` are eight
//! 64-bit registers nothing else touches — storage that tile traffic can
//! never evict, which is the whole point (**latency stability, not
//! average latency**).  Everywhere else the room is a plain word array
//! in the owning deck: other architectures have no dormant bank, so the
//! floor IS their implementation (§13.6 rule 4), same API, no pretence
//! of the eviction-immunity the bank alone provides.
//!
//! # Never inter-worker
//!
//! Register files have no coherence.  [`Room`] is affine and `!Sync` by
//! construction (rule 1 enforced by the type, not a comment): it lives
//! in one worker's fiber deck, only the RUNNING fiber touches it, and
//! the deck's cooperative switch is the one sequence point — mail posted
//! by lane A is readable by lane B from B's next resume onward.  Zero
//! ordering semantics exist and none are claimed.
//!
//! # Tenancy (the ruled quad per-lane mail)
//!
//! | slot | tenant |
//! |------|--------|
//! | `MM0` | the quad scheduler word's working copy (G9; the MEMORY word stays authoritative until the §13.6-rule-5 microbench rules the flip) |
//! | `MM1` | the active mask |
//! | `MM2` | reconvergence state |
//! | `MM3..MM6` | lane 0..3 **mailboxes** — one per quad lane |
//! | `MM7` | broadcast/scratch — last-writer-wins, never cleared |
//!
//! A mailbox is a single word, not a queue: `0` is the vacancy sentinel
//! (made honest by [`NonZeroU64`] in the API), the **receiver clears**
//! (read-and-clear its own box), and a displaced message comes back to
//! the sender — last-writer-wins made visible, never silent loss.
//! Octet mail is deferred by ruling: five spare slots cannot accommodate eight
//! lanes, so mail is quad-scoped until a consumer forces channel
//! semantics.

use core::marker::PhantomData;
use core::num::NonZeroU64;

/// The eight slots and their fixed tenancy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RoomSlot {
    /// The quad scheduler word's working copy (G9).
    Sched = 0,
    /// The active mask.
    Mask = 1,
    /// Reconvergence state.
    Reconverge = 2,
    /// Lane 0's mailbox.
    Mail0 = 3,
    /// Lane 1's mailbox.
    Mail1 = 4,
    /// Lane 2's mailbox.
    Mail2 = 5,
    /// Lane 3's mailbox.
    Mail3 = 6,
    /// Broadcast/scratch — last-writer-wins, never cleared.
    Broadcast = 7,
}

/// One quad lane — the mail addressing space (0..4 by construction, so
/// a mailbox index cannot be out of range).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum QuadLane {
    /// Lane 0.
    L0 = 0,
    /// Lane 1.
    L1 = 1,
    /// Lane 2.
    L2 = 2,
    /// Lane 3.
    L3 = 3,
}

impl QuadLane {
    /// The lane for a deck index, or `None` past the quad.
    pub const fn of(lane: usize) -> Option<Self> {
        match lane {
            0 => Some(Self::L0),
            1 => Some(Self::L1),
            2 => Some(Self::L2),
            3 => Some(Self::L3),
            _ => None,
        }
    }

    /// This lane's mailbox slot — the one arithmetic of the tenancy.
    pub const fn mailbox(self) -> RoomSlot {
        match self {
            Self::L0 => RoomSlot::Mail0,
            Self::L1 => RoomSlot::Mail1,
            Self::L2 => RoomSlot::Mail2,
            Self::L3 => RoomSlot::Mail3,
        }
    }
}

/// The bank contract — eight 64-bit slots, stored and loaded by the one
/// owner.  Deliberately not `Sync`-shaped: there is no ordering to
/// offer, and an implementation over registers could not honour one.
pub(crate) trait RoomBank {
    /// The bank's name for the terminal reports — `"mmx"` or `"deck"`.
    fn name() -> &'static str;
    /// A fresh bank, every slot zero (vacant).
    fn new() -> Self;
    /// Store `value` into `slot`.
    fn store(&mut self, slot: RoomSlot, value: u64);
    /// Load `slot`.
    fn load(&self, slot: RoomSlot) -> u64;
}

/// **The floor** — a deck-resident word array.  Chosen over
/// strip-resident storage deliberately: strip words would survive a
/// quad steal and hand floor targets STRONGER semantics than the MM
/// bank (registers cannot migrate cores), so deck residency keeps the
/// law uniform — room state is same-worker-episode only, and a steal
/// requires a quiescent room.
pub(crate) struct FloorBank {
    slots: [u64; 8],
}

impl RoomBank for FloorBank {
    fn name() -> &'static str {
        "deck"
    }

    fn new() -> Self {
        Self { slots: [0; 8] }
    }

    #[inline]
    fn store(&mut self, slot: RoomSlot, value: u64) {
        self.slots[slot as usize] = value;
    }

    #[inline]
    fn load(&self, slot: RoomSlot) -> u64 {
        self.slots[slot as usize]
    }
}

// ── Selection — the hint pattern: per-target module, one cfg'd alias,
//    the floor everywhere a real bank is not proven. ───────────────────

#[cfg(all(target_os = "none", target_arch = "x86_64"))]
mod x86_64;

/// This target's bank: the MM registers on metal x86-64 (gated on
/// `target_os = "none"` like `hint::Monitor` — the host suite keeps the
/// floor even on x86 silicon, so it cannot pass for a reason the metal
/// target never reproduces), the deck array elsewhere.
#[cfg(all(target_os = "none", target_arch = "x86_64"))]
pub(crate) use x86_64::MmBank as Bank;

/// This target's bank — the floor.
#[cfg(not(all(target_os = "none", target_arch = "x86_64")))]
pub(crate) use FloorBank as Bank;

/// One worker's room — affine, `!Sync`/`!Send` by construction (the
/// raw-pointer marker): a register file cannot cross threads, so the
/// type cannot either.
pub struct Room {
    bank: Bank,
    /// Rule 1 at the type level: never `Send`, never `Sync`.
    _pinned: PhantomData<*mut ()>,
}

impl Room {
    /// A fresh room, every slot vacant.
    pub(crate) fn new() -> Self {
        Self {
            bank: Bank::new(),
            _pinned: PhantomData,
        }
    }

    /// The bank's name — the terminal reports' `bank=` field.
    pub fn bank_name() -> &'static str {
        <Bank as RoomBank>::name()
    }

    /// Raw store — the deck's tenancy slots (`Sched`/`Mask`/
    /// `Reconverge`); the mail API below is the lane-facing surface.
    #[inline]
    pub(crate) fn store(&mut self, slot: RoomSlot, value: u64) {
        self.bank.store(slot, value);
    }

    /// Raw load.
    #[inline]
    pub(crate) fn load(&self, slot: RoomSlot) -> u64 {
        self.bank.load(slot)
    }

    /// Post `msg` into `dst`'s mailbox.  The displaced message comes
    /// BACK — last-writer-wins made visible, never silent loss.  The
    /// receiver (and only the receiver) clears, via [`take`](Self::take).
    pub fn post(&mut self, dst: QuadLane, msg: NonZeroU64) -> Option<NonZeroU64> {
        let slot = dst.mailbox();
        let displaced = NonZeroU64::new(self.bank.load(slot));
        self.bank.store(slot, msg.get());
        displaced
    }

    /// Read-and-clear OWN mailbox — `None` when vacant.  A lane may
    /// re-read its box within its slice by simply not taking it; taking
    /// is the receipt.
    pub fn take(&mut self, me: QuadLane) -> Option<NonZeroU64> {
        let slot = me.mailbox();
        let msg = NonZeroU64::new(self.bank.load(slot));
        if msg.is_some() {
            self.bank.store(slot, 0);
        }
        msg
    }

    /// Publish to the broadcast slot — last-writer-wins, never cleared.
    pub fn publish(&mut self, value: u64) {
        self.bank.store(RoomSlot::Broadcast, value);
    }

    /// Read the broadcast slot.
    pub fn snoop(&self) -> u64 {
        self.bank.load(RoomSlot::Broadcast)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(v: u64) -> NonZeroU64 {
        NonZeroU64::new(v).expect("test message is non-zero")
    }

    /// The mail laws: vacancy is honest, the receiver clears, a
    /// displaced message returns to the sender.
    #[test]
    fn mail_posts_takes_and_returns_displacement() {
        let mut room = Room::new();
        assert_eq!(room.take(QuadLane::L1), None, "vacant reads vacant");
        assert_eq!(
            room.post(QuadLane::L1, msg(7)),
            None,
            "first post displaces nothing"
        );
        assert_eq!(
            room.post(QuadLane::L1, msg(9)),
            Some(msg(7)),
            "the displaced message returns"
        );
        assert_eq!(
            room.take(QuadLane::L1),
            Some(msg(9)),
            "the receiver reads the last write"
        );
        assert_eq!(room.take(QuadLane::L1), None, "taking clears");
        // The neighbour's box was never touched.
        assert_eq!(room.take(QuadLane::L2), None);
    }

    /// Broadcast is last-writer-wins and never cleared by reads.
    #[test]
    fn broadcast_is_sticky() {
        let mut room = Room::new();
        assert_eq!(room.snoop(), 0);
        room.publish(3);
        room.publish(5);
        assert_eq!(room.snoop(), 5);
        assert_eq!(room.snoop(), 5, "snooping never clears");
    }

    /// The tenancy arithmetic: each lane's mailbox is its own slot, all
    /// four inside the spare band, and the deck tenants stay untouched
    /// by mail traffic.
    #[test]
    fn tenancy_is_disjoint() {
        assert_eq!(QuadLane::L0.mailbox() as u8, 3);
        assert_eq!(QuadLane::L3.mailbox() as u8, 6);
        assert_eq!(QuadLane::of(4), None, "the quad has four lanes");
        let mut room = Room::new();
        room.store(RoomSlot::Sched, 0xDEAD);
        let _ = room.post(QuadLane::L0, msg(1));
        room.publish(2);
        assert_eq!(
            room.load(RoomSlot::Sched),
            0xDEAD,
            "mail never clobbers a tenant"
        );
    }

    /// The host floor names itself honestly.
    #[test]
    fn the_floor_is_named() {
        assert_eq!(Room::bank_name(), "deck");
    }
}
