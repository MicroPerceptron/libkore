//! The **ring books** — the honest-counters doctrine, shared: a driver that
//! sits behind a service face reports these in its `summarize` row, and a
//! first-light boot-test prints them to prove the ring actually cycled.

/// A ring's visible books.  Field names are the vocabulary both drivers
/// speak (`kore_nvme::QpairStats` is this type; xHCI's rings report the same).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RingStats {
    /// Entries submitted (staged + doorbell-published).
    pub submitted: u64,
    /// Completions reaped.
    pub completed: u64,
    /// Completions with a non-success status.
    pub errors: u64,
    /// Completions naming an id with no command in flight — a device
    /// answering twice, or answering something never submitted.  Dropped,
    /// never delivered, and counted so the books show it.
    pub stray: u64,
}
