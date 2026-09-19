//! Capacity and shape constants — the engine's memory budget.
//!
//! The bounds themselves live in the process-side leaf
//! ([`kore_dagfs::bounds`]): a client sizes its window, its page, and its
//! walk from the same constants the engine sizes its arenas from, so the two
//! cannot disagree about what fits. The one bound that stays here is the
//! session count, because it is derived from the runtime's serving-pair
//! width, which the leaf deliberately does not link.

pub use kore_dagfs::bounds::*;

/// Maximum authority sessions. The low band exactly follows the runtime's
/// fixed/recyclable serving-pair bound; the high four are stable task-view
/// owners. A command pair is still its own session, but its root is a
/// descendant of exactly one high-band task view and disappears when the
/// command exits. Deriving this instead of respelling the serving width keeps
/// the authority table and the host's recyclable cells in lockstep.
pub const MAX_SESSIONS: usize = kore_rt::svc::MAX_SERVE_PAIRS + MAX_TASK_VIEWS;

#[cfg(test)]
mod tests {
    use super::*;

    /// Every session needs at least its private root, and the service shares one
    /// authority arena across sessions.
    #[test]
    fn authority_arena_covers_one_root_per_session() {
        assert!(MAX_AUTHORITY_EDGES >= MAX_SESSIONS);
    }

    /// The reachability walk's visited row is keyed by directory slot, so it
    /// must cover the arena exactly and stay a stack-sized row.
    #[test]
    fn reach_set_covers_the_directory_arena() {
        assert_eq!(crate::reach::ReachSet::CAPACITY, MAX_DIRECTORIES);
        assert!(crate::reach::ReachSet::BYTES <= 512);
    }
}
