//! Shared client-side path syntax.
//!
//! The service resolves one validated [`crate::Name`] at a time. A path is
//! therefore presentation-side iteration, never ambient authority and never
//! a string interpreted by the filesystem service.

/// The components of a path, empty segments skipped.
#[derive(Clone, Copy, Debug)]
pub struct Components<'a> {
    rest: &'a [u8],
}

/// Iterate `path`'s components.
#[inline]
pub const fn components(path: &[u8]) -> Components<'_> {
    Components { rest: path }
}

impl<'a> Iterator for Components<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        while let [b'/', tail @ ..] = self.rest {
            self.rest = tail;
        }
        if self.rest.is_empty() {
            return None;
        }
        let end = self
            .rest
            .iter()
            .position(|byte| *byte == b'/')
            .unwrap_or(self.rest.len());
        let component = &self.rest[..end];
        self.rest = &self.rest[end..];
        Some(component)
    }
}

/// Split a path into its parent prefix and final component.
pub fn split_parent(path: &[u8]) -> Option<(&[u8], &[u8])> {
    let mut end = path.len();
    while end > 0 && path[end - 1] == b'/' {
        end -= 1;
    }
    if end == 0 {
        return None;
    }
    let mut at = end;
    while at > 0 && path[at - 1] != b'/' {
        at -= 1;
    }
    Some((&path[..at], &path[at..end]))
}

/// Whether two paths walk the same sequence of directory components.
pub fn same_directory(left: &[u8], right: &[u8]) -> bool {
    let mut left = components(left);
    let mut right = components(right);
    loop {
        match (left.next(), right.next()) {
            (None, None) => return true,
            (Some(a), Some(b)) if a == b => {}
            _ => return false,
        }
    }
}

/// How many components a path walk would follow.
pub fn depth(path: &[u8]) -> usize {
    components(path).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec::Vec;

    fn split(path: &str) -> Vec<&str> {
        components(path.as_bytes())
            .map(|component| core::str::from_utf8(component).unwrap())
            .collect()
    }

    #[test]
    fn empty_segments_are_skipped_everywhere() {
        assert_eq!(split("a/b"), ["a", "b"]);
        assert_eq!(split("/a/b"), ["a", "b"]);
        assert_eq!(split("a/b/"), ["a", "b"]);
        assert_eq!(split("//a//b//"), ["a", "b"]);
        assert!(split("").is_empty());
        assert!(split("///").is_empty());
    }

    #[test]
    fn dot_components_remain_for_name_validation() {
        assert_eq!(split("a/../b"), ["a", "..", "b"]);
        assert!(crate::Name::from_bytes(b"..").is_err());
        assert!(crate::Name::from_bytes(b".").is_err());
    }

    #[test]
    fn split_and_directory_comparison_follow_component_semantics() {
        assert!(split_parent(b"").is_none());
        assert!(split_parent(b"///").is_none());
        assert_eq!(
            split_parent(b"file"),
            Some((b"".as_slice(), b"file".as_slice()))
        );
        assert_eq!(
            split_parent(b"a/b//"),
            Some((b"a/".as_slice(), b"b".as_slice()))
        );
        assert!(same_directory(b"a/b", b"/a//b/"));
        assert!(same_directory(b"", b"/"));
        assert!(!same_directory(b"a/b", b"a"));
    }

    #[test]
    fn depth_counts_the_walk() {
        assert_eq!(depth(b""), 0);
        assert_eq!(depth(b"/"), 0);
        assert_eq!(depth(b"a"), 1);
        assert_eq!(depth(b"//a//b//c//"), 3);
    }
}
