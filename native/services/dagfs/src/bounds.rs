//! Capacity and shape constants — the engine's memory budget.
//!
//! The bounds themselves live in the process-side leaf
//! ([`kore_dagfs::bounds`]): a client sizes its window, its page, and its
//! walk from the same constants the engine sizes its arenas from, so the two
//! cannot disagree about what fits. The bounds that stay here are the
//! session ones, because they are derived from the runtime's serving-slot
//! space, which the leaf deliberately does not link.

pub use kore_dagfs::bounds::*;

/// Sessions keyed by a serving slot: every slot the serve host can name. A
/// pair's session is its slot (pair *n* → session *n*) — the founded pairs
/// first, then the per-launch faces the host grows past its bootstrap table
/// — so this is the host's whole slot space, [`PeerId::SLOT_CAPACITY`].
/// Sizing it by the bootstrap table (`MAX_SERVE_PAIRS`) once left six faces
/// beside the two founded pairs: a seventh program that reads files was
/// refused however much memory was free.
///
/// [`PeerId::SLOT_CAPACITY`]: kore_rt::svc::PeerId::SLOT_CAPACITY
pub const PAIR_SESSIONS: usize = kore_rt::svc::PeerId::SLOT_CAPACITY;

/// The first stable task-view session: the band directly above every
/// pair's, so no slot the host grows into can name a task view.
pub const FIRST_TASK_VIEW_SESSION: usize = PAIR_SESSIONS;

/// Maximum authority sessions: one per serving slot, then the stable task
/// views. A command pair is its own session, but its root is a descendant
/// of exactly one task view and disappears when the command exits.
///
/// A ceiling, not a reservation: a session's storage — its
/// [`MAX_HANDLES_PER_SESSION`] handles and [`MAX_EDGES_PER_SESSION`] edges —
/// is allocated when its slot first attaches, and the host reuses the lowest
/// free slot, so the footprint follows the most sessions ever attached at
/// once.
pub const MAX_SESSIONS: usize = PAIR_SESSIONS + MAX_TASK_VIEWS;

const _: () = assert!(
    MAX_SESSIONS <= u16::MAX as usize + 1,
    "every session must be nameable by a SessionId"
);

#[cfg(test)]
mod tests {
    use super::*;

    /// Every session holds at least its own root, and every chain the
    /// system builds fits the lineage bound with room to delegate.
    #[test]
    fn a_session_holds_its_root_and_the_system_chain_fits() {
        assert!(MAX_EDGES_PER_SESSION >= 1);
        assert!(MAX_LINEAGE_DEPTH > 8);
    }

    /// The task views sit above every slot a pair can occupy.
    #[test]
    fn task_views_sit_above_every_pair_slot() {
        assert_eq!(FIRST_TASK_VIEW_SESSION, kore_rt::svc::PeerId::SLOT_CAPACITY);
        assert!(FIRST_TASK_VIEW_SESSION >= kore_rt::svc::MAX_SERVE_PAIRS);
        assert_eq!(MAX_SESSIONS - FIRST_TASK_VIEW_SESSION, MAX_TASK_VIEWS);
    }

    /// The reachability walk's visited row is keyed by directory slot, so it
    /// must cover the arena exactly and stay a stack-sized row.
    #[test]
    fn reach_set_covers_the_directory_arena() {
        assert_eq!(crate::reach::ReachSet::CAPACITY, MAX_DIRECTORIES);
        assert!(crate::reach::ReachSet::BYTES <= 512);
    }
}
