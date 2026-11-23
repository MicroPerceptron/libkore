//! The program's end request: one word any of the program's threads posts and
//! the host reads. The first request wins; later ones only end their thread.

use core::ffi::c_int;
use core::sync::atomic::{AtomicU64, Ordering};

/// One end request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Exit {
    /// The C status: the wire's `i32`, committed as given.
    pub(crate) status: c_int,
    /// `exit` rather than `_Exit` or a panic: finalizers may run.
    pub(crate) finalize: bool,
    /// The main thread asked, and is ending itself; the host waits for it
    /// before the barrier, so a program that ran on it alone stops cleanly.
    pub(crate) from_main: bool,
}

const POSTED: u64 = 1 << 63;
const FINALIZE: u64 = 1 << 33;
const FROM_MAIN: u64 = 1 << 32;

impl Exit {
    fn word(self) -> u64 {
        POSTED
            | (u64::from(self.finalize) * FINALIZE)
            | (u64::from(self.from_main) * FROM_MAIN)
            | u64::from(self.status as u32)
    }

    fn from_word(word: u64) -> Option<Self> {
        (word & POSTED != 0).then_some(Self {
            status: word as u32 as c_int,
            finalize: word & FINALIZE != 0,
            from_main: word & FROM_MAIN != 0,
        })
    }
}

/// Zero until the first request, then that request, for the program's life.
pub(crate) struct ExitWord(AtomicU64);

impl ExitWord {
    pub(crate) const fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    /// The request, once one was posted.
    pub(crate) fn posted(&self) -> Option<Exit> {
        Exit::from_word(self.0.load(Ordering::Acquire))
    }

    /// Ask for `exit`. `false` when another request was first; the word keeps
    /// that one.
    pub(crate) fn post(&self, exit: Exit) -> bool {
        self.0
            .compare_exchange(0, exit.word(), Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_round_trips_every_status_and_flag() {
        for status in [0, 1, 42, -1, c_int::MIN, c_int::MAX] {
            for finalize in [false, true] {
                for from_main in [false, true] {
                    let exit = Exit {
                        status,
                        finalize,
                        from_main,
                    };
                    let word = ExitWord::new();
                    assert_eq!(word.posted(), None);
                    assert!(word.post(exit));
                    assert_eq!(word.posted(), Some(exit));
                }
            }
        }
    }

    #[test]
    fn the_first_request_wins() {
        let word = ExitWord::new();
        let first = Exit {
            status: 3,
            finalize: true,
            from_main: false,
        };
        let second = Exit {
            status: 7,
            finalize: false,
            from_main: true,
        };
        assert!(word.post(first));
        assert!(!word.post(second));
        assert_eq!(word.posted(), Some(first));
    }

    #[test]
    fn a_status_of_zero_is_still_a_request() {
        let word = ExitWord::new();
        let exit = Exit {
            status: 0,
            finalize: false,
            from_main: false,
        };
        assert!(word.post(exit));
        assert_eq!(word.posted(), Some(exit));
    }
}
