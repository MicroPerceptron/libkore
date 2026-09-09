use crate::{FsError, FsResult};

/// The frozen filesystem wire's maximum component width.
const NAME_BYTES: usize = kore_abi::svc::FS_NAME_BYTES;

/// One validated name component.
///
/// Validity is established once, at construction, and the type carries it
/// thereafter — an engine operation cannot be handed an unchecked name. The
/// rules:
///
/// - non-empty, and at most 32 bytes;
/// - valid UTF-8;
/// - no embedded NUL and no `/`; and
/// - `.` and `..` are reserved.
///
/// `.` and `..` are refused outright rather than given traversal meaning:
/// with no ambient root and a per-client view, `..` would either escape a
/// view or need a silently-clamping rule. The native graph contract refuses
/// to inherit either behaviour from POSIX by accident. A client that wants
/// "the parent" holds the parent's handle, which is authority it already had.
#[derive(Clone, Copy)]
pub struct Name {
    len: u8,
    bytes: [u8; NAME_BYTES],
}

impl Name {
    /// Validate a `str` into a name.
    pub fn new(name: &str) -> FsResult<Self> {
        Self::from_bytes(name.as_bytes())
    }

    /// Validate raw bytes into a name — the wire path, where UTF-8 has not
    /// been established yet.
    pub fn from_bytes(bytes: &[u8]) -> FsResult<Self> {
        if bytes.is_empty() || bytes.len() > NAME_BYTES {
            return Err(FsError::InvalidName);
        }
        if core::str::from_utf8(bytes).is_err() {
            return Err(FsError::InvalidName);
        }
        if bytes.contains(&0) || bytes.contains(&b'/') {
            return Err(FsError::InvalidName);
        }
        if bytes == b"." || bytes == b".." {
            return Err(FsError::InvalidName);
        }
        let mut stored = [0u8; NAME_BYTES];
        stored[..bytes.len()].copy_from_slice(bytes);
        Ok(Self {
            len: bytes.len() as u8,
            bytes: stored,
        })
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len as usize]
    }

    /// The name as a `str`. Infallible: [`Self::from_bytes`] established
    /// UTF-8 before the bytes were stored.
    #[inline]
    pub fn as_str(&self) -> &str {
        core::str::from_utf8(self.as_bytes()).unwrap_or("")
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl PartialEq for Name {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl Eq for Name {}

impl core::fmt::Debug for Name {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Name({:?})", self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_match_the_frozen_wire_width_and_refuse_traversal_syntax() {
        let exact = [b'x'; NAME_BYTES];
        let over = [b'x'; NAME_BYTES + 1];
        assert!(Name::from_bytes(&exact).is_ok());
        assert_eq!(Name::from_bytes(&over), Err(FsError::InvalidName));
        for invalid in [
            b"".as_slice(),
            b".".as_slice(),
            b"..".as_slice(),
            b"a/b".as_slice(),
        ] {
            assert_eq!(Name::from_bytes(invalid), Err(FsError::InvalidName));
        }
    }
}
