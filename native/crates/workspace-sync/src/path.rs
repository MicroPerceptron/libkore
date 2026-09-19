//! Canonical, portable relative workspace paths.

use core::{cmp::Ordering, fmt};

use heapless::Vec;
use unicode_normalization::UnicodeNormalization;

use crate::bounds::{MAX_COMPONENT_BYTES, MAX_PATH_BYTES, MAX_PATH_DEPTH};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PathError {
    Empty,
    Absolute,
    EmptyComponent,
    DotComponent,
    TooDeep,
    ComponentTooLong,
    PathTooLong,
    InvalidUtf8,
    NotNfc,
    NonPortable,
    ReservedName,
}

#[derive(Clone, Eq, Hash, PartialEq)]
pub struct CanonicalPath(Vec<u8, MAX_PATH_BYTES>);

impl CanonicalPath {
    pub fn new(bytes: &[u8]) -> Result<Self, PathError> {
        if bytes.is_empty() {
            return Err(PathError::Empty);
        }
        if bytes.first() == Some(&b'/') || bytes.first() == Some(&b'\\') {
            return Err(PathError::Absolute);
        }
        if bytes.len() > MAX_PATH_BYTES {
            return Err(PathError::PathTooLong);
        }
        let text = core::str::from_utf8(bytes).map_err(|_| PathError::InvalidUtf8)?;
        if !text.nfc().eq(text.chars()) {
            return Err(PathError::NotNfc);
        }
        let mut depth = 0usize;
        for component in bytes.split(|byte| *byte == b'/') {
            if component.is_empty() {
                return Err(PathError::EmptyComponent);
            }
            depth = depth.checked_add(1).ok_or(PathError::TooDeep)?;
            if depth > MAX_PATH_DEPTH {
                return Err(PathError::TooDeep);
            }
            validate_component(component)?;
        }
        Vec::from_slice(bytes)
            .map(Self)
            .map_err(|_| PathError::PathTooLong)
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        self.0.as_slice()
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        // Construction establishes UTF-8.
        core::str::from_utf8(self.as_bytes()).unwrap_or("")
    }

    pub fn depth(&self) -> usize {
        self.as_bytes().split(|byte| *byte == b'/').count()
    }

    pub fn file_name(&self) -> &[u8] {
        self.as_bytes()
            .rsplit(|byte| *byte == b'/')
            .next()
            .unwrap_or_default()
    }

    pub fn parent(&self) -> Option<Self> {
        let at = self.as_bytes().iter().rposition(|byte| *byte == b'/')?;
        Self::new(&self.as_bytes()[..at]).ok()
    }

    /// Whether `self` is the same path as, or a component-delimited child of,
    /// `ancestor`. Byte-prefix alone is insufficient (`a` must not match `ab`).
    pub fn is_within(&self, ancestor: &Self) -> bool {
        self == ancestor
            || (self.as_bytes().starts_with(ancestor.as_bytes())
                && self.as_bytes().get(ancestor.as_bytes().len()) == Some(&b'/'))
    }

    /// Portable collision key equality. V1 rejects Unicode lowercase
    /// collisions even though kore itself is byte-exact, so a snapshot has
    /// one meaning on case-sensitive and case-insensitive host filesystems.
    pub fn portable_eq(&self, other: &Self) -> bool {
        self.as_str()
            .chars()
            .flat_map(char::to_lowercase)
            .eq(other.as_str().chars().flat_map(char::to_lowercase))
    }
}

impl Ord for CanonicalPath {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_bytes().cmp(other.as_bytes())
    }
}

impl PartialOrd for CanonicalPath {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Debug for CanonicalPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("CanonicalPath")
            .field(&self.as_str())
            .finish()
    }
}

fn validate_component(component: &[u8]) -> Result<(), PathError> {
    if component == b"." || component == b".." {
        return Err(PathError::DotComponent);
    }
    if component.len() > MAX_COMPONENT_BYTES {
        return Err(PathError::ComponentTooLong);
    }
    if component
        .last()
        .is_some_and(|byte| matches!(byte, b'.' | b' '))
    {
        return Err(PathError::NonPortable);
    }
    if component.iter().any(|byte| {
        *byte < 0x20
            || matches!(
                byte,
                0x7f | b'\\' | b':' | b'*' | b'?' | b'"' | b'<' | b'>' | b'|'
            )
    }) {
        return Err(PathError::NonPortable);
    }
    let stem = component
        .split(|byte| *byte == b'.')
        .next()
        .unwrap_or(component);
    if is_windows_reserved(stem) {
        return Err(PathError::ReservedName);
    }
    Ok(())
}

fn is_windows_reserved(stem: &[u8]) -> bool {
    const FIXED: [&[u8]; 4] = [b"con", b"prn", b"aux", b"nul"];
    if FIXED.iter().any(|name| stem.eq_ignore_ascii_case(name)) {
        return true;
    }
    if stem.len() == 4
        && (stem[..3].eq_ignore_ascii_case(b"com") || stem[..3].eq_ignore_ascii_case(b"lpt"))
        && matches!(stem[3], b'1'..=b'9')
    {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn accepts_portable_relative_nfc_paths() {
        let path = CanonicalPath::new("src/café.rs".as_bytes()).unwrap();
        assert_eq!(path.depth(), 2);
        assert_eq!(path.file_name(), "café.rs".as_bytes());
        assert_eq!(path.parent().unwrap().as_str(), "src");
    }

    #[test]
    fn rejects_escape_ambiguity_and_nonportable_names() {
        for bad in [
            "",
            "/a",
            "a//b",
            "a/../b",
            "a/.",
            "a\\b",
            "CON",
            "lpt9.txt",
            "a. ",
            "e\u{301}.txt",
        ] {
            assert!(CanonicalPath::new(bad.as_bytes()).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn portable_equality_catches_case_collisions() {
        let a = CanonicalPath::new("Src/File.rs".as_bytes()).unwrap();
        let b = CanonicalPath::new("src/file.rs".as_bytes()).unwrap();
        assert!(a.portable_eq(&b));
        assert_ne!(a, b);
    }

    #[test]
    fn subtree_test_is_component_delimited() {
        let root = CanonicalPath::new(b"src").unwrap();
        assert!(CanonicalPath::new(b"src/lib.rs").unwrap().is_within(&root));
        assert!(root.is_within(&root));
        assert!(!CanonicalPath::new(b"src2/lib.rs").unwrap().is_within(&root));
    }

    proptest! {
        #[test]
        fn arbitrary_untrusted_path_bytes_never_panic_or_reencode_differently(
            bytes in proptest::collection::vec(any::<u8>(), 0..=(MAX_PATH_BYTES + 16)),
        ) {
            if let Ok(path) = CanonicalPath::new(&bytes) {
                prop_assert_eq!(path.as_bytes(), bytes.as_slice());
                prop_assert_eq!(CanonicalPath::new(path.as_bytes()), Ok(path));
            }
        }
    }
}
