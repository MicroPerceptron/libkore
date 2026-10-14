//! One aggregate stdout/stderr budget for Lua print and error paths.

use alloc::vec::Vec;

#[derive(Default)]
pub(crate) struct Capture {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub limit: usize,
    pub exhausted: bool,
}

impl Capture {
    fn remaining(&self) -> usize {
        self.limit
            .saturating_sub(self.stdout.len().saturating_add(self.stderr.len()))
    }

    pub fn append(&mut self, bytes: &[u8], stderr: bool) -> bool {
        if bytes.len() > self.remaining() {
            self.exhausted = true;
            return false;
        }
        let output = if stderr {
            &mut self.stderr
        } else {
            &mut self.stdout
        };
        if output.try_reserve(bytes.len()).is_err() {
            self.exhausted = true;
            return false;
        }
        output.extend_from_slice(bytes);
        true
    }

    /// Errors have priority within the same aggregate limit as `print`.
    /// Reclaim the tail of stdout if it filled the budget before a failure;
    /// otherwise a near-limit print could erase the only diagnosis. Borrow
    /// the Lua string and never allocate proportionally to its length.
    pub fn append_error(&mut self, error: &[u8]) {
        let body = error.strip_suffix(b"\n").unwrap_or(error);
        let wanted = body.len().saturating_add(1).min(self.limit);
        let reclaim = wanted
            .saturating_sub(self.remaining())
            .min(self.stdout.len());
        self.stdout.truncate(self.stdout.len() - reclaim);
        let remaining = self.remaining();
        if remaining == 0 {
            return;
        }
        let copied = body.len().min(remaining - 1);
        if self.stderr.try_reserve(copied + 1).is_ok() {
            self.stderr.extend_from_slice(&body[..copied]);
            self.stderr.push(b'\n');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_error_is_truncated_at_policy_limit() {
        let mut capture = Capture {
            limit: 32,
            ..Capture::default()
        };
        capture.append_error(&[b'x'; 4096]);
        assert_eq!(capture.stderr.len(), 32);
        assert_eq!(capture.stderr[31], b'\n');
        assert_eq!(capture.remaining(), 0);
    }

    #[test]
    fn print_quota_error_reclaims_stdout_within_shared_budget() {
        let mut capture = Capture {
            limit: 8,
            ..Capture::default()
        };
        assert!(capture.append(b"hello", false));
        assert!(!capture.append(b"more", false));
        assert!(capture.exhausted);
        capture.append_error(b"quota exceeded: a deliberately long error");
        assert!(capture.stdout.is_empty());
        assert_eq!(capture.stderr, b"quota e\n");
        assert_eq!(capture.stdout.len() + capture.stderr.len(), 8);
        assert_eq!(capture.stderr[7], b'\n');
        assert_eq!(capture.remaining(), 0);
    }

    #[test]
    fn nearly_full_stdout_preserves_later_error() {
        let mut capture = Capture {
            limit: 2048,
            ..Capture::default()
        };
        assert!(capture.append(&[b'x'; 2047], false));
        capture.append_error(b"=qwen-tool:1: sentinel\nstack traceback:");
        assert!(capture.stderr.windows(8).any(|part| part == b"sentinel"));
        assert!(capture.stderr.ends_with(b"\n"));
        assert!(capture.stdout.len() + capture.stderr.len() <= 2048);
    }

    #[test]
    fn zero_budget_has_no_error_escape() {
        let mut capture = Capture::default();
        capture.append_error(b"long error");
        assert!(capture.stderr.is_empty());
        assert!(!capture.append(b"x", false));
        assert!(capture.exhausted);
    }
}
