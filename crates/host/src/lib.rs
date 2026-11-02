//! Host-side Kore tooling: what a program on a Linux or macOS host needs to
//! work with a Kore machine it keeps.
//!
//! [`kore-std`](https://github.com/MicroPerceptron/libkore) is the facade for
//! programs running *under* Kore. This crate is its counterpart on the other
//! side of the machine boundary — a VM supervisor such as vKore, a volume
//! browser, a packager — and it is built from the same exported native
//! owners (`native/provenance.json`), never from a second copy of them.
//!
//! # A machine's volume, offline
//!
//! A Kore machine keeps its files in DagFS, which no host kernel mounts.
//! [`volume`] drives the filesystem row's own engine — the exported
//! `kore-vol` procedure over `kore-svc-dagfs` and `kore-driver`'s image
//! device — so what it reads is what a booted Kore reads, and what it writes
//! is what the next boot mounts:
//!
//! - **Found as the boot finds it:** the whole disk when it is Kore's, else
//!   the GPT partition designated `KORE_FS`. A raw disk image, or on Linux a
//!   machine's disk itself (its block device, with read and, to write, write
//!   permission on its node).
//! - **Offline only.** A running machine keeps its free extents and its next
//!   root slot in memory, so a second writer would corrupt the volume
//!   silently. The image device takes a whole-file record lock and refuses an
//!   image another process holds — QEMU holds the images it runs — so a
//!   machine must be shut down before its volume is opened here.
//! - **All or nothing.** [`volume::put`] and [`volume::rm`] are one root
//!   publication each, read back through a fresh recovery before they
//!   report success; a failure part way leaves the volume as it was.
//! - **One format version.** A commit writes [`volume::FORMAT_VERSION`]. A
//!   volume a newer Kore wrote is refused as `Incompatible`: use the libkore
//!   revision whose export matches the Kore build the machine boots.
//! - **Kore's bounds,** checked before anything is written:
//!   [`volume::MAX_OBJECTS`] objects on a volume, [`volume::MAX_DIR_ENTRIES`]
//!   entries in a directory, [`volume::MAX_PATH_DEPTH`] levels, names of at
//!   most 32 bytes ([`volume::Name`]).
//!
//! Paths are a terminal's: relative to terminal `face`'s home (`work/face-N`
//! on the volume), or from the volume's root when they start with `/`
//! (reads only). Every call runs the engine on a thread of its own with the
//! stack recovery needs, so callers need none.
//!
//! ```no_run
//! use kore_host::volume::{self, Kind};
//! use std::path::Path;
//!
//! # fn main() -> Result<(), volume::VolError> {
//! let disk = Path::new("machine.raw");
//! // What terminal 0 saved with PrintScreen, as the machine left it.
//! let shots = volume::read(disk, None, |mount| {
//!     let mut shots = Vec::new();
//!     mount.walk(0, "pictures/screenshots", &mut |entry| {
//!         if entry.kind == Kind::File {
//!             shots.push((entry.path.to_owned(), entry.len));
//!         }
//!         Ok(())
//!     })?;
//!     Ok(shots)
//! })?;
//! for (name, len) in shots {
//!     println!("{name}: {len} bytes");
//! }
//! # Ok(())
//! # }
//! ```
#![forbid(unsafe_code)]

/// A Kore machine's volume, offline, through the filesystem row's own engine
/// (the crate docs).
pub mod volume {
    pub use kore_vol::{
        DEFAULT_BLOCK_SIZE, FORMAT_VERSION, Formatted, Installed, Kind, Listed, Listing,
        MAX_DIR_ENTRIES, MAX_OBJECTS, MAX_PATH_DEPTH, Mount, MountStatus, Name, Removed,
        VOLUME_STACK_BYTES, VolError, Walked, format, is_face, put, read, rm,
    };
}
