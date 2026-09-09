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

/// `id → cookie`, indexed `id % len` (ids are reused round-robin over the
/// table, so the modulus is exact, never lossy, as long as in-flight count
/// stays `< len` — which the producer ring enforces).
///
/// Under `alloc` the backing is a `Box<[u64]>` sized to the ring's depth;
/// otherwise a fixed `[u64; MAX_SLOTS]`.  Both index identically — the
/// house `Neutral` pattern.
pub struct CookieTable {
    #[cfg(feature = "alloc")]
    inner: alloc::boxed::Box<[u64]>,
    #[cfg(not(feature = "alloc"))]
    inner: [u64; MAX_SLOTS],
}

impl CookieTable {
    /// A zeroed table for `depth` entries, posture-appropriate.  Under
    /// `alloc`, sized exactly to `depth`; otherwise the fixed ceiling.
    #[cfg(feature = "alloc")]
    pub fn new(depth: u16) -> Self {
        Self {
            inner: alloc::vec![0u64; depth as usize].into_boxed_slice(),
        }
    }

    /// A zeroed table for `depth` entries, posture-appropriate.
    #[cfg(not(feature = "alloc"))]
    pub fn new(_depth: u16) -> Self {
        Self {
            inner: [0u64; MAX_SLOTS],
        }
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
