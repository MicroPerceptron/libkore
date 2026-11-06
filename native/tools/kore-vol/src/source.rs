//! The host side of a `put`: the tree to copy, read and checked whole before
//! the volume is opened, so a name the volume cannot hold or a directory too
//! wide for it refuses the install before any of it is written.

use std::{
    fs,
    path::{Path, PathBuf},
};

use kore_svc_dagfs::{Name, bounds::MAX_DIR_ENTRIES};

/// One object to create.
pub enum Entry {
    File { name: Name, path: PathBuf, len: u64 },
    Directory { name: Name, entries: Vec<Entry> },
}

impl Entry {
    pub fn name(&self) -> &Name {
        match self {
            Self::File { name, .. } | Self::Directory { name, .. } => name,
        }
    }
}

/// The tree under its destination name, and what it will cost the volume.
pub struct Plan {
    pub root: Entry,
    pub files: usize,
    pub directories: usize,
    pub bytes: u64,
    /// Levels from the destination down to its deepest object, the
    /// destination itself being one.
    pub depth: usize,
}

impl Plan {
    pub fn objects(&self) -> usize {
        self.files + self.directories
    }
}

#[derive(Default)]
struct Tally {
    files: usize,
    directories: usize,
    bytes: u64,
    depth: usize,
}

/// Read `source` (a regular file or a directory) as the tree to create
/// under `name`. Symbolic links and special files are refused, not
/// followed: an install copies what the build wrote, and a link would copy
/// whatever it points at, from anywhere on the host.
pub fn plan(source: &Path, name: Name) -> Result<Plan, String> {
    let mut tally = Tally::default();
    let root = entry(source, name, 1, &mut tally)?;
    Ok(Plan {
        root,
        files: tally.files,
        directories: tally.directories,
        bytes: tally.bytes,
        depth: tally.depth,
    })
}

fn entry(path: &Path, name: Name, depth: usize, tally: &mut Tally) -> Result<Entry, String> {
    let metadata =
        fs::symlink_metadata(path).map_err(|error| format!("{}: {error}", path.display()))?;
    tally.depth = tally.depth.max(depth);
    let kind = metadata.file_type();
    if kind.is_file() {
        tally.files += 1;
        tally.bytes += metadata.len();
        return Ok(Entry::File {
            name,
            path: path.to_path_buf(),
            len: metadata.len(),
        });
    }
    if !kind.is_dir() {
        return Err(format!(
            "{}: neither a file nor a directory (links and special files are not copied)",
            path.display()
        ));
    }
    tally.directories += 1;
    let mut children = Vec::new();
    for child in fs::read_dir(path).map_err(|error| format!("{}: {error}", path.display()))? {
        let child = child.map_err(|error| format!("{}: {error}", path.display()))?;
        let child_path = child.path();
        let child_name = child
            .file_name()
            .to_str()
            .and_then(|text| Name::new(text).ok())
            .ok_or_else(|| {
                format!(
                    "{}: not a kore name (UTF-8, at most 32 bytes, no NUL)",
                    child_path.display()
                )
            })?;
        children.push((child_name, child_path));
    }
    if children.len() > MAX_DIR_ENTRIES {
        return Err(format!(
            "{}: {} entries, and a kore directory holds at most {MAX_DIR_ENTRIES}",
            path.display(),
            children.len()
        ));
    }
    // Created in name order, so two installs of one tree make the same
    // volume.
    children.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    let mut entries = Vec::with_capacity(children.len());
    for (child_name, child_path) in children {
        entries.push(entry(&child_path, child_name, depth + 1, tally)?);
    }
    Ok(Entry::Directory { name, entries })
}
