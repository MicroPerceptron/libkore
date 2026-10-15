//! **Timelines** — the monotone completion record a client waits on without
//! a call (`res/STORAGE_TRANSPORT.md` §4).
//!
//! Sequences are implicit: every submission naming a timeline takes the
//! next sequence on both sides. Completions arrive in any order; the record
//! publishes `done` as the highest sequence whose predecessors have all
//! completed, and `fault` as the lowest sequence that failed. The pack lane
//! published exactly this; here it is one fixed-size value with no heap.

use kore_abi::stor::FenceRecord;

/// Sequences a timeline can hold completed ahead of `done`. Nothing else
/// bounds the gap — completions run out of order, and once `done + 1` is
/// slow the engine keeps popping behind it — so the engine holds a
/// submission whose timeline already has this many open
/// ([`Timeline::open`]). A completion past the window could never be
/// recorded, and the fence would stop at it for good.
pub const WINDOW: u64 = 256;

const WORDS: usize = (WINDOW / 64) as usize;

/// One timeline's books.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Timeline {
    issued: u64,
    done: u64,
    fault: u64,
    /// Bit `i` set: sequence `done + 1 + i` has completed.
    ahead: [u64; WORDS],
}

/// A completion the timeline could not accept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimelineError {
    /// The sequence was never issued, is already done, or already
    /// completed.
    Unknown,
    /// The sequence is further than [`WINDOW`] ahead of `done`.
    TooFar,
}

impl Timeline {
    /// Take the next sequence for a new submission.
    pub fn issue(&mut self) -> u64 {
        self.issued += 1;
        self.issued
    }

    /// Record `seq`'s completion; `ok` false marks it failed. Returns the
    /// record to publish.
    pub fn complete(&mut self, seq: u64, ok: bool) -> Result<FenceRecord, TimelineError> {
        if seq <= self.done || seq > self.issued {
            return Err(TimelineError::Unknown);
        }
        let offset = seq - self.done - 1;
        if offset >= WINDOW {
            return Err(TimelineError::TooFar);
        }
        let (word, bit) = ((offset / 64) as usize, offset % 64);
        if self.ahead[word] & (1 << bit) != 0 {
            return Err(TimelineError::Unknown);
        }
        self.ahead[word] |= 1 << bit;
        if !ok && (self.fault == 0 || seq < self.fault) {
            self.fault = seq;
        }
        let run = self.leading_run();
        if run > 0 {
            self.shift(run);
            self.done += run;
        }
        Ok(self.record())
    }

    /// The record as it stands.
    pub const fn record(&self) -> FenceRecord {
        FenceRecord {
            done: self.done,
            fault: self.fault,
        }
    }

    /// Sequences issued and not yet within `done`.
    pub const fn open(&self) -> u64 {
        self.issued - self.done
    }

    /// Consecutive completed sequences starting at `done + 1`.
    fn leading_run(&self) -> u64 {
        let mut run = 0;
        for word in self.ahead {
            let ones = word.trailing_ones() as u64;
            run += ones;
            if ones < 64 {
                break;
            }
        }
        run
    }

    /// Drop the lowest `n` bits of the window.
    fn shift(&mut self, n: u64) {
        let words = (n / 64) as usize;
        let bits = (n % 64) as u32;
        for index in 0..WORDS {
            let low = self.ahead.get(index + words).copied().unwrap_or(0);
            let high = self.ahead.get(index + words + 1).copied().unwrap_or(0);
            self.ahead[index] = if bits == 0 {
                low
            } else {
                (low >> bits) | (high << (64 - bits))
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn done_is_the_contiguous_prefix_whatever_the_completion_order() {
        let mut t = Timeline::default();
        let seqs: [u64; 5] = core::array::from_fn(|_| t.issue());
        assert_eq!(seqs, [1, 2, 3, 4, 5]);
        assert_eq!(t.complete(3, true).unwrap().done, 0);
        assert_eq!(t.complete(2, true).unwrap().done, 0);
        assert_eq!(t.complete(1, true).unwrap().done, 3, "1..=3 now contiguous");
        assert_eq!(t.complete(5, true).unwrap().done, 3);
        assert_eq!(t.complete(4, true).unwrap(), FenceRecord { done: 5, fault: 0 });
        assert_eq!(t.open(), 0);
    }

    #[test]
    fn the_fault_is_the_lowest_failed_sequence_and_sticks() {
        let mut t = Timeline::default();
        for _ in 0..4 {
            t.issue();
        }
        assert_eq!(t.complete(3, false).unwrap().fault, 3);
        assert_eq!(t.complete(2, false).unwrap().fault, 2, "a lower failure wins");
        t.complete(1, true).unwrap();
        let record = t.complete(4, true).unwrap();
        assert_eq!(record, FenceRecord { done: 4, fault: 2 });
        t.issue();
        assert_eq!(t.complete(5, true).unwrap().fault, 2, "sticky past done");
    }

    #[test]
    fn unknown_duplicate_and_far_completions_are_refused() {
        let mut t = Timeline::default();
        assert_eq!(t.complete(1, true), Err(TimelineError::Unknown), "never issued");
        t.issue();
        t.issue();
        t.complete(2, true).unwrap();
        assert_eq!(t.complete(2, true), Err(TimelineError::Unknown), "twice");
        t.complete(1, true).unwrap();
        assert_eq!(t.complete(1, true), Err(TimelineError::Unknown), "already done");
        for _ in 0..=WINDOW {
            t.issue();
        }
        assert_eq!(t.complete(2 + WINDOW + 1, true), Err(TimelineError::TooFar));
    }

    /// The window shifts across word boundaries without losing a bit.
    #[test]
    fn a_long_out_of_order_run_shifts_across_words() {
        let mut t = Timeline::default();
        for _ in 0..200 {
            t.issue();
        }
        for seq in (2..=200).rev() {
            assert_eq!(t.complete(seq, true).unwrap().done, 0);
        }
        assert_eq!(t.complete(1, true).unwrap().done, 200);
        assert_eq!(t.ahead, [0; WORDS]);
    }
}
