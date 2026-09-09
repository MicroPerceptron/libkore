//! **Vif identity and the vif table** — the structure that makes a closed vif
//! unreachable rather than merely absent.

/// Vifs one device may hold open at once.
///
/// A design bound.  Each open vif costs a mapped client window and a place in
/// the classifier's fan-out, both of which are bounded resources.
pub const MAX_VIFS: usize = 8;

/// A vif's identity: which slot, and **which tenant of that slot**.
///
/// The generation is the whole point.  Slots are reused, and a client that
/// holds an identifier across a close would otherwise address whoever took
/// the slot next — reading its frames, or injecting into its ring.  Bumping
/// the generation on close makes every previously-issued identifier for that
/// slot permanently unresolvable, so a stale producer is refused by
/// construction rather than by a check someone has to remember to write.
///
/// Generation `0` is never issued, so a zeroed identifier — the value a
/// shared window starts at, and the value a confused peer is most likely to
/// send — never resolves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VifId {
    /// Slot index.
    pub index: u16,
    /// Which tenant of that slot.
    pub generation: u16,
}

impl VifId {
    /// The identifier that never resolves.
    pub const NONE: Self = Self {
        index: 0,
        generation: 0,
    };

    /// Whether this identifier could name a live vif at all.
    pub const fn is_some(self) -> bool {
        self.generation != 0
    }
}

/// Why a vif lookup refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VifRefusal {
    /// The slot index is outside the table.
    OutOfRange,
    /// The slot is empty.
    Closed,
    /// The slot is live, but belongs to a **later** tenant: the caller is
    /// holding an identifier from before a close.  Distinguished from
    /// [`Closed`](Self::Closed) because it is the one that indicates a real
    /// use-after-close, and a service should say so rather than reporting a
    /// generic miss.
    Stale,
}

/// A fixed table of open vifs, keyed by [`VifId`].
///
/// Owns slot allocation and the generation discipline together, so a service
/// cannot open a slot without recording a tenant or close one without
/// invalidating its identifiers.
pub struct VifTable<T> {
    slots: [Option<T>; MAX_VIFS],
    generation: [u16; MAX_VIFS],
}

impl<T> Default for VifTable<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> VifTable<T> {
    /// An empty table.  Generations start at 1 so that a zeroed identifier
    /// never resolves.
    pub const fn new() -> Self {
        Self {
            slots: [const { None }; MAX_VIFS],
            generation: [1; MAX_VIFS],
        }
    }

    /// Open a vif, returning its identity, or `None` when every slot is taken.
    pub fn open(&mut self, value: T) -> Option<VifId> {
        let index = self.slots.iter().position(|s| s.is_none())?;
        self.slots[index] = Some(value);
        Some(VifId {
            index: index as u16,
            generation: self.generation[index],
        })
    }

    /// Resolve `id` to its slot index, or say why not.
    fn resolve(&self, id: VifId) -> Result<usize, VifRefusal> {
        let index = id.index as usize;
        if index >= MAX_VIFS {
            return Err(VifRefusal::OutOfRange);
        }
        if self.generation[index] != id.generation {
            return Err(VifRefusal::Stale);
        }
        if self.slots[index].is_none() {
            return Err(VifRefusal::Closed);
        }
        Ok(index)
    }

    /// The vif `id` names, or why it does not resolve.
    pub fn get(&self, id: VifId) -> Result<&T, VifRefusal> {
        let index = self.resolve(id)?;
        Ok(self.slots[index]
            .as_ref()
            .expect("resolve checked occupancy"))
    }

    /// The vif `id` names, mutably.
    pub fn get_mut(&mut self, id: VifId) -> Result<&mut T, VifRefusal> {
        let index = self.resolve(id)?;
        Ok(self.slots[index]
            .as_mut()
            .expect("resolve checked occupancy"))
    }

    /// Close the vif `id` names, returning what it held.
    ///
    /// Bumps the slot's generation, so **every** identifier previously issued
    /// for it — including this one — stops resolving. Skips generation 0 on
    /// wrap so the never-issued value stays never-issued.
    pub fn close(&mut self, id: VifId) -> Result<T, VifRefusal> {
        let index = self.resolve(id)?;
        let value = self.slots[index].take().expect("resolve checked occupancy");
        let next = self.generation[index].wrapping_add(1);
        self.generation[index] = if next == 0 { 1 } else { next };
        Ok(value)
    }

    /// Open vifs, with their current identities.
    pub fn iter(&self) -> impl Iterator<Item = (VifId, &T)> {
        self.slots.iter().enumerate().filter_map(|(i, slot)| {
            slot.as_ref().map(|v| {
                (
                    VifId {
                        index: i as u16,
                        generation: self.generation[i],
                    },
                    v,
                )
            })
        })
    }

    /// Open vifs, mutably, with their current identities.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (VifId, &mut T)> {
        let generation = self.generation;
        self.slots
            .iter_mut()
            .enumerate()
            .filter_map(move |(i, slot)| {
                slot.as_mut().map(|v| {
                    (
                        VifId {
                            index: i as u16,
                            generation: generation[i],
                        },
                        v,
                    )
                })
            })
    }

    /// How many vifs are open.
    pub fn len(&self) -> usize {
        self.slots.iter().flatten().count()
    }

    /// Whether no vif is open.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reason identity carries a generation: an identifier held across a
    /// close must stop resolving, not address whoever took the slot next.
    /// Without this, a stale producer writes into a stranger's ring.
    #[test]
    fn a_stale_identifier_cannot_reach_the_slots_next_tenant() {
        let mut table: VifTable<&str> = VifTable::new();
        let first = table.open("peer").expect("a free slot");
        assert_eq!(table.get(first), Ok(&"peer"));

        assert_eq!(table.close(first), Ok("peer"));

        // The very same identifier, reused after the close.
        assert_eq!(table.get(first), Err(VifRefusal::Stale));

        // A new tenant lands in the same slot and gets a different identity.
        let second = table.open("infra").expect("the slot is free again");
        assert_eq!(second.index, first.index, "the slot really was reused");
        assert_ne!(second.generation, first.generation);

        assert_eq!(
            table.get(first),
            Err(VifRefusal::Stale),
            "the old identifier still does not resolve"
        );
        assert_eq!(table.get(second), Ok(&"infra"));
    }

    /// A zeroed identifier — what a fresh shared window contains, and what a
    /// confused peer is most likely to send — never resolves.
    #[test]
    fn the_zero_identifier_never_resolves() {
        let mut table: VifTable<u8> = VifTable::new();
        assert_eq!(table.get(VifId::NONE), Err(VifRefusal::Stale));
        assert!(!VifId::NONE.is_some());

        let live = table.open(7).unwrap();
        assert!(live.is_some());
        assert_ne!(live.generation, 0);
        assert_eq!(table.get(VifId::NONE), Err(VifRefusal::Stale));
    }

    /// An index outside the table and an index inside but empty are
    /// distinguishable refusals — a service should be able to say which.
    #[test]
    fn refusals_name_their_reason() {
        let mut table: VifTable<u8> = VifTable::new();
        let live = table.open(1).unwrap();

        let far = VifId {
            index: MAX_VIFS as u16,
            generation: 1,
        };
        assert_eq!(table.get(far), Err(VifRefusal::OutOfRange));

        // A slot that was never opened is Closed, not Stale: generation 1 is
        // the initial value, so the identifier is current for an empty slot.
        let never = VifId {
            index: (MAX_VIFS - 1) as u16,
            generation: 1,
        };
        assert_eq!(table.get(never), Err(VifRefusal::Closed));

        assert_eq!(table.close(live), Ok(1));
        assert_eq!(
            table.close(live),
            Err(VifRefusal::Stale),
            "closing twice is a stale reference, not a double free"
        );
    }

    /// The table is bounded, and a full table refuses rather than evicting.
    #[test]
    fn the_table_is_bounded_and_refuses_when_full() {
        let mut table: VifTable<usize> = VifTable::new();
        let ids: alloc::vec::Vec<_> = (0..MAX_VIFS)
            .map(|i| table.open(i).expect("within the bound"))
            .collect();
        assert_eq!(table.len(), MAX_VIFS);
        assert!(table.open(999).is_none(), "a full table refuses");

        // Every identifier is distinct and still resolves.
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(table.get(*id), Ok(&i));
        }

        table.close(ids[0]).unwrap();
        assert_eq!(table.len(), MAX_VIFS - 1);
        assert!(table.open(1234).is_some(), "a freed slot is reusable");
    }

    /// Iteration yields the *current* identity of each open vif, so a
    /// classifier fanning a frame out cannot pick up a stale one.
    #[test]
    fn iteration_yields_current_identities() {
        let mut table: VifTable<u8> = VifTable::new();
        let a = table.open(10).unwrap();
        let b = table.open(20).unwrap();
        table.close(a).unwrap();
        let c = table.open(30).unwrap();

        let seen: alloc::vec::Vec<_> = table.iter().map(|(id, v)| (id, *v)).collect();
        assert_eq!(seen.len(), 2);
        assert!(seen.contains(&(b, 20)));
        assert!(seen.contains(&(c, 30)));
        assert!(
            !seen.iter().any(|(id, _)| *id == a),
            "the closed identity is gone"
        );

        for (_, v) in table.iter_mut() {
            *v += 1;
        }
        assert_eq!(table.get(b), Ok(&21));
        assert_eq!(table.get(c), Ok(&31));
    }

    /// Generation 0 is skipped on wrap, so the never-issued value stays
    /// never-issued even after 65535 closes.
    #[test]
    fn generation_skips_zero_on_wrap() {
        let mut table: VifTable<u8> = VifTable::new();
        for _ in 0..=u16::MAX as u32 + 2 {
            let id = table.open(0).expect("slot 0 is free each round");
            assert_ne!(id.generation, 0, "0 is never issued");
            table.close(id).unwrap();
        }
    }
}
