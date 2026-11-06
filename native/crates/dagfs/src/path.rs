//! Shared client-side path syntax.
//!
//! The service resolves one validated [`crate::Name`] at a time. A path is
//! therefore presentation-side iteration, never ambient authority and never
//! a string interpreted by the filesystem service.
//!
//! Two dot spellings, two treatments. `.` names the directory it stands in,
//! so it adds no step, and no edge can carry it ([`crate::Name`] refuses
//! it): [`components`] skips it, which can only turn a refused path into the
//! one it plainly meant. `..` is kept: the containment DAG stores no parent
//! edges, so a walk cannot honour it and name validation refuses it. Only
//! [`resolve`] folds `..`, against a position the caller tracks.

/// The components of a path, empty and `.` segments skipped.
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
        loop {
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
            if component != b"." {
                return Some(component);
            }
        }
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

/// Why a path has no canonical root-relative form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unresolved {
    /// A `..` would climb past the view root, which has no parent.
    AboveRoot,
    /// The canonical form does not fit the caller's buffer.
    TooLong,
}

/// Whether `target` reaches outside the subtree of the position it is
/// resolved against: it is rooted (`/…`) or climbs (a `..` component).
/// Only such a target needs [`resolve`] and the view root to be named.
pub fn leaves_position(target: &[u8]) -> bool {
    target.first() == Some(&b'/') || target.split(|byte| *byte == b'/').any(|part| part == b"..")
}

/// Fold `target` onto `position` — both paths within one view — into the
/// canonical path from that view's root, written to `out`; returns its
/// length.
///
/// A rooted `target` (`/…`) starts at the view root rather than at
/// `position`. `.` and empty components are inert, `..` pops one component,
/// and climbing past the root is refused rather than clamped: the root has
/// no parent, and pretending otherwise would silently name a different
/// object. The result has no empty, `.` or `..` component and no leading or
/// trailing `/`; the root itself is the empty path.
pub fn resolve(position: &[u8], target: &[u8], out: &mut [u8]) -> Result<usize, Unresolved> {
    let mut len = 0;
    if target.first() != Some(&b'/') {
        len = fold(out, len, position)?;
    }
    fold(out, len, target)
}

/// Append `path`'s components to the canonical prefix `out[..len]`.
fn fold(out: &mut [u8], mut len: usize, path: &[u8]) -> Result<usize, Unresolved> {
    for component in path.split(|byte| *byte == b'/') {
        match component {
            b"" | b"." => {}
            b".." => {
                if len == 0 {
                    return Err(Unresolved::AboveRoot);
                }
                len = out[..len]
                    .iter()
                    .rposition(|byte| *byte == b'/')
                    .unwrap_or(0);
            }
            name => {
                let sep = usize::from(len > 0);
                let end = len + sep + name.len();
                if end > out.len() {
                    return Err(Unresolved::TooLong);
                }
                if sep == 1 {
                    out[len] = b'/';
                }
                out[len + sep..end].copy_from_slice(name);
                len = end;
            }
        }
    }
    Ok(len)
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
    fn a_dot_adds_no_step_and_a_parent_remains_for_name_validation() {
        assert_eq!(split("./a/./b/."), ["a", "b"]);
        assert!(split(".").is_empty());
        assert!(split("./").is_empty());
        // `.` names where it stands; no edge can carry it, so skipping it
        // never changes which object a path names.
        assert!(crate::Name::from_bytes(b".").is_err());
        // `..` stays for the walk to refuse: the DAG stores no parent.
        assert_eq!(split("a/../b"), ["a", "..", "b"]);
        assert!(crate::Name::from_bytes(b"..").is_err());
        // A name that merely starts with a dot is an ordinary name.
        assert_eq!(split("./.config/..x"), [".config", "..x"]);
    }

    #[test]
    fn a_dotted_prefix_is_the_same_directory() {
        assert_eq!(
            split_parent(b"./file"),
            Some((b"./".as_slice(), b"file".as_slice()))
        );
        assert!(same_directory(b"./", b""));
        assert!(same_directory(b"a/./b", b"a/b"));
        assert_eq!(depth(b"./a/."), 1);
    }

    fn resolved(position: &str, target: &str) -> Result<std::string::String, Unresolved> {
        let mut out = [0u8; 64];
        let len = resolve(position.as_bytes(), target.as_bytes(), &mut out)?;
        Ok(std::string::String::from_utf8(out[..len].to_vec()).unwrap())
    }

    #[test]
    fn resolution_folds_onto_the_position_or_the_root() {
        assert_eq!(resolved("", "a/b").unwrap(), "a/b");
        assert_eq!(resolved("docs", "a").unwrap(), "docs/a");
        assert_eq!(resolved("docs/a", "..").unwrap(), "docs");
        assert_eq!(resolved("docs/a", "../../x").unwrap(), "x");
        assert_eq!(resolved("docs/a", "./b/../c/").unwrap(), "docs/a/c");
        // A rooted target ignores the position.
        assert_eq!(resolved("docs/a", "/x/y").unwrap(), "x/y");
        assert_eq!(resolved("docs/a", "/").unwrap(), "");
        assert_eq!(resolved("docs", "").unwrap(), "docs");
        // `..` at the root is refused, never clamped.
        assert_eq!(resolved("", ".."), Err(Unresolved::AboveRoot));
        assert_eq!(resolved("docs", "/.."), Err(Unresolved::AboveRoot));
        assert_eq!(resolved("docs", "../.."), Err(Unresolved::AboveRoot));
        // Names that only look like dots are names.
        assert_eq!(resolved("", ".config/..x").unwrap(), ".config/..x");
    }

    #[test]
    fn resolution_is_bounded_by_the_buffer() {
        let mut out = [0u8; 5];
        assert_eq!(resolve(b"", b"abcde", &mut out), Ok(5));
        assert_eq!(resolve(b"", b"abc/de", &mut out), Err(Unresolved::TooLong));
        // Popping can bring a long walk back within bounds.
        assert_eq!(resolve(b"", b"ab/../cde", &mut out), Ok(3));
        assert_eq!(&out[..3], b"cde");
    }

    #[test]
    fn only_a_rooted_or_climbing_target_leaves_its_position() {
        assert!(leaves_position(b"/"));
        assert!(leaves_position(b"/a"));
        assert!(leaves_position(b".."));
        assert!(leaves_position(b"a/../b"));
        assert!(!leaves_position(b""));
        assert!(!leaves_position(b"a/b"));
        assert!(!leaves_position(b"./a"));
        assert!(!leaves_position(b"..x/a"));
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
