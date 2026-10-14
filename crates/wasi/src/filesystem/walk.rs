//! The guest's paths, resolved one component at a time from a descriptor's
//! own view.
//!
//! # Only beneath the descriptor
//!
//! [`Fs::open`] walks from the task root and, when the first component is
//! not a containment edge there, falls back to the task's **named grants**
//! (`libs`, `apps`, an embedder's private `data` grant). That is right for a
//! shell verb and wrong for a guest: a guest opening `data/x` would reach
//! authority its embedder holds and never handed it. So a guest path is
//! resolved by [`walk`] over [`Steps`], which offers exactly the owner's
//! [`Fs::open_at`] — one containment step, a missing child is `NotFound` —
//! and a close. There is no `open` in it to call. The only other entry is
//! [`Fs::open_root`], the task's own root, which is what the preopen *is*.
//!
//! `..` has no meaning (`Name` reserves it), and an absolute path would name
//! the task root regardless of the descriptor: both are refused before any
//! request ([`components`]). A descriptor cannot reach above its preopen.
//!
//! A named grant the embedder offers (`Filesystem::with_named_grants`) is a
//! preopen of its own, under the grant's name: the guest walks beneath it
//! exactly as beneath `.`, and still never reaches a grant *through* `.`.
//!
//! [`Fs::open`]: kore::fs::Fs::open
//! [`Fs::open_at`]: kore::fs::Fs::open_at
//! [`Fs::open_root`]: kore::fs::Fs::open_root

use alloc::vec::Vec;

use kore::fs::dag::bounds::{MAX_NAME_LEN, MAX_PATH_DEPTH};
use kore::fs::{FsError, Name, Refusal, Rights};

use crate::bindings::wasi::filesystem::types::ErrorCode;

/// A guest path as the components it walks, or the WASI refusal.
///
/// Empty and `.` segments are skipped (`a//./b` is `a/b`; `.` alone is no
/// component — the descriptor itself). An absolute path and any `..`
/// component are `not-permitted`: the first would name the task root past
/// the descriptor, the second would climb above it.
pub(crate) fn components(path: &str) -> Result<Vec<Name>, ErrorCode> {
    if path.starts_with('/') {
        return Err(ErrorCode::NotPermitted);
    }
    let mut names = Vec::new();
    for segment in path.split('/') {
        match segment {
            "" | "." => continue,
            ".." => return Err(ErrorCode::NotPermitted),
            _ if segment.len() > MAX_NAME_LEN => return Err(ErrorCode::NameTooLong),
            _ => {}
        }
        if names.len() == MAX_PATH_DEPTH {
            return Err(ErrorCode::NameTooLong);
        }
        names.push(Name::from_bytes(segment.as_bytes()).map_err(|_| ErrorCode::Invalid)?);
    }
    Ok(names)
}

/// What resolving a guest path may do: one containment step, and give a
/// view back. Deliberately nothing else — no whole-path `open`, no grant.
///
/// The host implements it on the owner's `Fs` (`open_at`, `close_checked`);
/// this module's tests implement it on a script, which is how the request
/// sequence is pinned on the host.
pub(crate) trait Steps {
    type View;
    async fn open_at(
        &mut self,
        parent: &Self::View,
        name: Name,
        rights: Rights,
    ) -> Result<Self::View, Refusal>;
    async fn close(&mut self, view: Self::View);
}

/// Walk `names` beneath `base`: one step per component, each from the view
/// the previous step opened, every intermediate closed on success and on
/// refusal. `base` is borrowed, never closed. A missing component is the
/// service's `NotFound` — there is no fallback (see the module docs).
pub(crate) async fn walk<S: Steps>(
    steps: &mut S,
    base: &S::View,
    names: &[Name],
    rights: Rights,
) -> Result<S::View, Refusal> {
    let Some((first, rest)) = names.split_first() else {
        return Err(Refusal::Fs(FsError::InvalidName));
    };
    let mut current = steps.open_at(base, *first, rights).await?;
    for name in rest {
        let next = steps.open_at(&current, *name, rights).await;
        steps.close(current).await;
        current = next?;
    }
    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::VecDeque;
    use alloc::string::String;
    use core::pin::pin;
    use core::task::{Context, Poll, Waker};

    /// Every future here is ready on its first poll: the script answers
    /// synchronously.
    fn now<F: Future>(future: F) -> F::Output {
        match pin!(future).poll(&mut Context::from_waker(Waker::noop())) {
            Poll::Ready(output) => output,
            Poll::Pending => panic!("a scripted walk never parks"),
        }
    }

    #[derive(Debug, PartialEq)]
    enum Step {
        OpenAt { parent: u32, name: String },
        Close(u32),
    }

    /// Views are numbers; each step is recorded and answered from a script.
    struct Script {
        steps: Vec<Step>,
        answers: VecDeque<Result<u32, Refusal>>,
    }

    impl Steps for Script {
        type View = u32;
        async fn open_at(&mut self, parent: &u32, name: Name, _: Rights) -> Result<u32, Refusal> {
            self.steps.push(Step::OpenAt {
                parent: *parent,
                name: name.as_str().into(),
            });
            self.answers.pop_front().unwrap_or(Err(Refusal::Pair))
        }
        async fn close(&mut self, view: u32) {
            self.steps.push(Step::Close(view));
        }
    }

    fn names(path: &str) -> Vec<Name> {
        components(path).unwrap()
    }

    fn open_at(parent: u32, name: &str) -> Step {
        Step::OpenAt {
            parent,
            name: name.into(),
        }
    }

    #[test]
    fn paths_cannot_climb_or_start_at_the_root() {
        assert_eq!(components("..").unwrap_err(), ErrorCode::NotPermitted);
        assert_eq!(components("a/../b").unwrap_err(), ErrorCode::NotPermitted);
        assert_eq!(components("/etc").unwrap_err(), ErrorCode::NotPermitted);
        assert!(components(".").unwrap().is_empty());
        assert!(components("").unwrap().is_empty());
        let walked = names("./a//b/.");
        assert_eq!(walked.len(), 2);
        assert_eq!(walked[0].as_bytes(), b"a");
        assert_eq!(walked[1].as_bytes(), b"b");
    }

    #[test]
    fn paths_are_bounded_by_the_engines_proof() {
        let long = "x".repeat(MAX_NAME_LEN + 1);
        assert_eq!(components(&long).unwrap_err(), ErrorCode::NameTooLong);
        let deep = ["d"; MAX_PATH_DEPTH + 1].join("/");
        assert_eq!(components(&deep).unwrap_err(), ErrorCode::NameTooLong);
        assert_eq!(
            names(&["d"; MAX_PATH_DEPTH].join("/")).len(),
            MAX_PATH_DEPTH
        );
        assert_eq!(components("a\0b").unwrap_err(), ErrorCode::Invalid);
    }

    /// The rule this module exists for: a guest's `data/x` is resolved by
    /// `open_at` from the descriptor's own view, and a missing `data` is
    /// `NotFound`. Resolution has no whole-path `open` to fall back
    /// through, so the task's named grants — where an embedder's private
    /// `data` grant would answer — are never consulted.
    #[test]
    fn a_missing_component_is_not_found_never_a_grant() {
        let mut script = Script {
            steps: Vec::new(),
            answers: [Err(Refusal::Fs(FsError::NotFound))].into(),
        };
        let walked = now(walk(&mut script, &7, &names("data/x"), Rights::READ));
        assert!(matches!(walked, Err(Refusal::Fs(FsError::NotFound))));
        assert_eq!(script.steps, [open_at(7, "data")]);
    }

    /// Each step opens beneath the previous one, and every intermediate is
    /// closed — on success and when a later step is refused.
    #[test]
    fn a_walk_chains_from_the_base_and_closes_what_it_passed() {
        let mut script = Script {
            steps: Vec::new(),
            answers: [Ok(8), Ok(9)].into(),
        };
        let leaf = now(walk(&mut script, &7, &names("a/b"), Rights::READ)).unwrap();
        assert_eq!(leaf, 9);
        assert_eq!(
            script.steps,
            [open_at(7, "a"), open_at(8, "b"), Step::Close(8)]
        );

        let mut script = Script {
            steps: Vec::new(),
            answers: [Ok(8), Err(Refusal::Fs(FsError::NotFound))].into(),
        };
        let refused = now(walk(&mut script, &7, &names("a/b"), Rights::READ));
        assert!(matches!(refused, Err(Refusal::Fs(FsError::NotFound))));
        assert_eq!(
            script.steps,
            [open_at(7, "a"), open_at(8, "b"), Step::Close(8)]
        );
    }
}
