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

    /// Errors are reported through the same aggregate limit as `print`.
    /// Reserve a byte for one line terminator and borrow the Lua string;
    /// never allocate proportionally to an untrusted error message.
    pub fn append_error(&mut self, error: &[u8]) {
        let remaining = self.remaining();
        if remaining == 0 {
            return;
        }
        let body = error.strip_suffix(b"\n").unwrap_or(error);
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
    fn print_quota_error_uses_only_remaining_shared_budget() {
        let mut capture = Capture {
            limit: 8,
            ..Capture::default()
        };
        assert!(capture.append(b"hello", false));
        assert!(!capture.append(b"more", false));
        assert!(capture.exhausted);
        capture.append_error(b"quota exceeded: a deliberately long error");
        assert_eq!(capture.stdout, b"hello");
        assert_eq!(capture.stderr.len(), 3);
        assert_eq!(capture.stderr[2], b'\n');
        assert_eq!(capture.remaining(), 0);
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
