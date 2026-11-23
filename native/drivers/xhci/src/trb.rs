//! The **TRB adapter** — the vendored [`xhci`] crate's typed TRB builders on
//! one side, the [`kore_driver`] ring entry traits on the other.
//!
//! Every xHCI TRB is 16 bytes (`[u32; 4]`); the **Cycle** bit is bit 0 of
//! dword 3 for *all* TRB types (xHCI 1.2 §4.11.1 — the vendored crate places
//! it there too, `rw_bit!([3](0), cycle_bit)`).  So one raw [`Trb`] newtype
//! serves the command, transfer, and event rings, and the substrate's
//! producer cycle stamp / consumer validity poll are both that one bit.
//!
//! The driver builds a TRB with the crate's typed API (`command::EnableSlot`,
//! `transfer::SetupStage`, …) **without** setting the cycle bit, calls
//! [`into`](Trb::from) to get the raw form, and stages it — [`SubRing`] then
//! stamps the current producer cycle at publish time (only the ring knows
//! the PCS).  Events are read raw and decoded with the crate's `TryFrom`.

use kore_driver::ring::{CqEntry, SqEntry};

/// The Cycle bit — bit 0 of the last (control) dword, for every TRB type.
const CYCLE: u32 = 1 << 0;

/// A raw 16-byte TRB — the common currency of all three xHCI rings.  Built
/// from the vendored crate's typed TRBs (`.into()`), decoded back with its
/// `TryFrom<[u32; 4]>` per ring.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct Trb(pub [u32; 4]);

impl Trb {
    /// The raw dwords (to write into a ring slot / feed the crate's decode).
    pub fn raw(self) -> [u32; 4] {
        self.0
    }

    /// The cycle bit currently stamped in this TRB.
    pub fn cycle(self) -> bool {
        self.0[3] & CYCLE != 0
    }

    /// The TRB Type field (dword 3, bits 10..=15) — routes an event TRB to
    /// its typed decode without constructing every variant.
    pub fn trb_type(self) -> u8 {
        ((self.0[3] >> 10) & 0x3f) as u8
    }
}

/// Any typed command/transfer/event TRB converts to the raw form via its
/// `into_raw()` — one blanket bridge so call sites read `trb.into()`.
impl<T: Into<[u32; 4]>> From<T> for Trb {
    fn from(t: T) -> Self {
        Trb(t.into())
    }
}

impl SqEntry for Trb {
    /// Stamp the producer cycle bit (the ring owns the PCS; the typed
    /// builder left it clear).
    fn stamp_cycle(&mut self, cycle: bool) {
        if cycle {
            self.0[3] |= CYCLE;
        } else {
            self.0[3] &= !CYCLE;
        }
    }
}

impl CqEntry for Trb {
    /// An event TRB is published for the reader when its Cycle bit matches
    /// the consumer's expected phase (the xHCI CCS handshake).
    fn valid(&self, phase: bool) -> bool {
        (self.0[3] & CYCLE != 0) == phase
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xhci::ring::trb::command;

    #[test]
    fn typed_command_converts_to_raw_with_correct_type_field() {
        // Enable Slot = TRB type 9; the builder leaves cycle clear.
        let raw: Trb = command::EnableSlot::new().into();
        assert_eq!(raw.trb_type(), 9, "Enable Slot Command = type 9");
        assert!(
            !raw.cycle(),
            "typed builder leaves cycle clear for the ring to stamp"
        );
    }

    #[test]
    fn stamp_and_validity_are_the_same_bit() {
        let mut t = Trb::default();
        assert!(!t.cycle());
        t.stamp_cycle(true);
        assert!(t.cycle());
        assert!(t.valid(true), "cycle 1 is valid for expected phase 1");
        assert!(!t.valid(false), "cycle 1 is stale for expected phase 0");
        t.stamp_cycle(false);
        assert!(t.valid(false));
    }
}
