//! The **cookie table** — correlate a device completion back to the `u64`
//! its submission carried (the kore-ring-shaped idiom: submit with a cookie,
//! poll, look it up by the id the device echoes).  The one storage choice
//! that differs by posture, behind a single type.

/// The **heapless** per-ring slot ceiling — the fixed inline table's size
/// when the `alloc` feature is off (the kernel boot-test posture).  With
/// `alloc` (userland — the global allocator is the mm-backed heap), the
/// table is sized **exactly** to the negotiated depth instead and this
/// ceiling does not apply.
pub const MAX_SLOTS: usize = 256;

/// `id → cookie`, in two allocation forms.
///
/// **Indexed** ([`put`](Self::put)/[`get`](Self::get)): the id is a ring
/// slot the caller already owns, indexed `id % len` — for a device whose
/// slots are reused in ring order (a NIC's descriptor ring), where a slot
/// is free again by the time the ring comes back around to it.
///
/// **Claimed** ([`claim`](Self::claim)/[`release`](Self::release)): the
/// table hands out an id that no command in flight holds, and takes it
/// back only when that command's completion is reaped — for a device that
/// echoes the id and completes out of order (NVMe, whose CID must be unique
/// among a queue's outstanding commands).  Round-robin reuse is NOT safe
/// there: an in-flight count below the depth proves that *some* command
/// completed, not that the one holding the next id did, and a straggling
/// read still in flight when its id is reissued has its cookie overwritten
/// — its completion then answers the newer command before that one's data
/// landed, and the newer command's completion answers nobody, so the
/// straggler's request never completes.
///
/// Under `alloc` the backing is sized to the ring's depth; otherwise a
/// fixed `MAX_SLOTS`.  Both index identically — the house `Neutral`
/// pattern.
pub struct CookieTable {
    #[cfg(feature = "alloc")]
    inner: alloc::boxed::Box<[u64]>,
    #[cfg(not(feature = "alloc"))]
    inner: [u64; MAX_SLOTS],
    /// Ids currently claimed and not yet released.
    #[cfg(feature = "alloc")]
    live: alloc::boxed::Box<[bool]>,
    #[cfg(not(feature = "alloc"))]
    live: [bool; MAX_SLOTS],
    /// The free ids, as a stack: `free[..free_len]`.
    #[cfg(feature = "alloc")]
    free: alloc::boxed::Box<[u16]>,
    #[cfg(not(feature = "alloc"))]
    free: [u16; MAX_SLOTS],
    free_len: usize,
    /// Ids the table hands out (`0..depth`). The heapless posture's arrays
    /// are the fixed ceiling, so their length is not the depth — counting
    /// live ids from the array length once reported 253 of a 3-deep table
    /// as in flight.
    depth: usize,
}

impl CookieTable {
    /// A zeroed table for `depth` entries, posture-appropriate.  Under
    /// `alloc`, sized exactly to `depth`; otherwise the fixed ceiling, with
    /// only ids below `depth` claimable.
    #[cfg(feature = "alloc")]
    pub fn new(depth: u16) -> Self {
        let depth = depth as usize;
        let mut table = Self {
            inner: alloc::vec![0u64; depth].into_boxed_slice(),
            live: alloc::vec![false; depth].into_boxed_slice(),
            free: alloc::vec![0u16; depth].into_boxed_slice(),
            free_len: 0,
            depth: 0,
        };
        table.fill_free(depth);
        table
    }

    /// A zeroed table for `depth` entries, posture-appropriate.
    #[cfg(not(feature = "alloc"))]
    pub fn new(depth: u16) -> Self {
        let mut table = Self {
            inner: [0u64; MAX_SLOTS],
            live: [false; MAX_SLOTS],
            free: [0u16; MAX_SLOTS],
            free_len: 0,
            depth: 0,
        };
        table.fill_free((depth as usize).min(MAX_SLOTS));
        table
    }

    /// Stack the ids `0..depth` so the first claim takes 0, the next 1,
    /// and so on — the order a fresh queue would have issued them.
    fn fill_free(&mut self, depth: usize) {
        for (at, id) in (0..depth).rev().enumerate() {
            self.free[at] = id as u16;
        }
        self.free_len = depth;
        self.depth = depth;
    }

    /// **Claim** an id no command in flight holds, recording `cookie` for
    /// it.  `None` when every id is claimed: the caller's ring is full.
    pub fn claim(&mut self, cookie: u64) -> Option<u16> {
        if self.free_len == 0 {
            return None;
        }
        self.free_len -= 1;
        let id = self.free[self.free_len];
        self.live[id as usize] = true;
        self.inner[id as usize] = cookie;
        Some(id)
    }

    /// **Release** id `id` on its completion, returning the cookie it was
    /// claimed with.  `None` when `id` is not in flight — a completion the
    /// device owes nobody, which must not free an id twice.
    pub fn release(&mut self, id: u16) -> Option<u64> {
        let at = id as usize;
        if at >= self.live.len() || !self.live[at] {
            return None;
        }
        self.live[at] = false;
        self.free[self.free_len] = id;
        self.free_len += 1;
        Some(self.inner[at])
    }

    /// Release every id at once, as though each command in flight completed
    /// — for a queue the device has dropped whole (a controller reset).
    /// Answers how many were in flight.
    pub fn clear(&mut self) -> u16 {
        let lost = self.depth.saturating_sub(self.free_len) as u16;
        *self = Self::new(self.depth as u16);
        lost
    }

    /// Whether `id` is claimed and not yet released.
    pub fn is_live(&self, id: u16) -> bool {
        self.live.get(id as usize).is_some_and(|live| *live)
    }

    /// Ids claimed and not yet released. Derived from `depth`, not the
    /// backing length: without `alloc` the live array is the fixed ceiling,
    /// so `len - free_len` would report the unclaimable ids as live.
    pub fn live_count(&self) -> usize {
        self.depth - self.free_len
    }

    /// The live id carrying `cookie`, if any — the reverse lookup an abort
    /// needs. A scan of the depth: control plane only.
    pub fn find(&self, cookie: u64) -> Option<u16> {
        (0..self.depth as u16).find(|&id| self.is_live(id) && self.get(id) == cookie)
    }

    /// Record `cookie` for command id `id`.
    pub fn put(&mut self, id: u16, cookie: u64) {
        let n = self.inner.len();
        self.inner[id as usize % n] = cookie;
    }

    /// The cookie recorded for id `id`.
    pub fn get(&self, id: u16) -> u64 {
        let n = self.inner.len();
        self.inner[id as usize % n]
    }

    /// The table's capacity (the modulus).
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the table has zero slots (never, in practice — the clippy
    /// convention pair).
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A claimed id is never handed out again until its completion releases
    /// it, whatever order completions arrive in.
    #[test]
    fn a_claimed_id_stays_claimed_until_released() {
        let mut table = CookieTable::new(4);
        let a = table.claim(10).expect("first");
        let b = table.claim(11).expect("second");
        let c = table.claim(12).expect("third");
        let d = table.claim(13).expect("fourth");
        assert_eq!([a, b, c, d], [0, 1, 2, 3]);
        assert_eq!(table.claim(14), None, "every id is in flight");
        // The middle one completes first — out of order, as random reads do.
        assert_eq!(table.release(b), Some(11));
        let again = table.claim(15).expect("the released id is free again");
        assert_eq!(again, b);
        assert!(table.is_live(a) && table.is_live(c) && table.is_live(d));
        assert_eq!(table.claim(16), None, "nothing else was released");
    }

    /// Releasing an id nobody holds frees nothing: a duplicate completion
    /// cannot make two commands share an id.
    #[test]
    fn a_release_of_an_unclaimed_id_is_refused() {
        let mut table = CookieTable::new(2);
        assert_eq!(table.release(0), None, "never claimed");
        let a = table.claim(1).expect("claim");
        assert_eq!(table.release(a), Some(1));
        assert_eq!(table.release(a), None, "already released");
        assert_eq!(table.release(7), None, "out of range");
        let x = table.claim(2).expect("one");
        let y = table.claim(3).expect("two");
        assert_ne!(x, y);
        assert_eq!(table.claim(4), None);
    }

    /// The indexed form still answers by slot, untouched by claims.
    /// The live count is claims minus releases at every depth — including
    /// a table shallower than the heapless ceiling, whose backing arrays are
    /// longer than its depth.
    #[test]
    fn live_count_tracks_claims_at_a_shallow_depth() {
        let mut table = CookieTable::new(3);
        assert_eq!(table.live_count(), 0);
        let a = table.claim(1).unwrap();
        let b = table.claim(2).unwrap();
        assert_eq!(table.live_count(), 2);
        assert_eq!(table.release(a), Some(1));
        assert_eq!(table.live_count(), 1);
        table.claim(3).unwrap();
        table.claim(4).unwrap();
        assert_eq!(table.live_count(), 3);
        assert_eq!(table.claim(5), None, "a 3-deep table is full at 3");
        assert_eq!(table.release(b), Some(2));
        assert_eq!(table.live_count(), 2);
    }

    #[test]
    fn put_and_get_index_by_slot() {
        let mut table = CookieTable::new(3);
        table.put(2, 99);
        assert_eq!(table.get(2), 99);
        assert_eq!(table.live_count(), 0);
    }
}
