//! `kore-vol` — a kore volume, from the host.
//!
//! What a terminal's home holds, and files from the host copied into it: what
//! the host built (an application's bundle, a dataset) put where a terminal
//! reads it, with no system-level bootstrapping in between
//! (`res/DAGFS.md`, "From the host").
//!
//! This library is the tool's whole procedure; the `kore-vol` binary is a
//! printer over it, and a host program that keeps machines (a VM supervisor
//! staging a project for a terminal, a packager) drives the same calls
//! instead of a second copy of them:
//!
//! - [`format()`] formats a blank image as the machine's first boot would.
//! - [`put`] installs a host file or tree under a terminal's home; [`rm`]
//!   removes one. Each is one root publication, read back through a fresh
//!   recovery before it reports success.
//! - [`read`] mounts the volume read-only and hands a [`Mount`] to a
//!   closure: [`Mount::list`], [`Mount::read_to`] and [`Mount::walk`] answer
//!   what a terminal's home holds, as data.
//!
//! - **Paths are a terminal's.** `path` is what terminal `N` (`face`, 0 for
//!   the first) types, relative to its home (`work/face-N` on the user
//!   volume): after `put dist js/hello`, terminal 0 reads the build as
//!   `js/hello`. Reads also take a path from the volume's root when it starts
//!   with `/`; nothing writes there.
//! - **The image** is a raw disk image, offline: the file a QEMU run boots
//!   from after `qemu-img convert -O raw`, or a copy of a machine's disk. The
//!   user volume on it is found exactly as the filesystem row finds it at
//!   boot — the same composition and the same `locate`
//!   (`kore_svc_dagfs::durable`): the whole disk when it is kore's, else the
//!   GPT partition designated `KORE_FS`. Its block size is `block_size`,
//!   [`DEFAULT_BLOCK_SIZE`] unless named.
//! - **A machine's disk** (Linux, as root): the disk's block device node, or
//!   the node of the partition that holds its user volume, with the machine
//!   shut down and the disk attached to the host running the tool (a live
//!   system on that machine, or the disk in an enclosure). The volume is
//!   found the same way, over the whole disk, at the device's own logical
//!   block size; it is written unbuffered, with the disk held and claimed as
//!   `device` describes, so never beside anything that mounts, claims or
//!   runs a machine on it.
//! - **All or nothing.** Every check runs before anything is written, and
//!   the install becomes visible in one root publication: a failure part
//!   way, of the tool or of the host, leaves the volume as it was. Each
//!   change is then read back through a fresh recovery and compared with its
//!   source.
//! - **`format`** formats a blank volume — both root blocks zero — exactly
//!   as the next boot would, so an image made with `qemu-img create` takes
//!   an install before it first boots. Nothing else is ever formatted, and
//!   a disk never is: its machine formats it at its first boot.
//! - **One format version.** A commit writes the format version this crate
//!   was built with, and a volume another kore version wrote is refused as
//!   [`VolError::Persistence`] with `Incompatible`: build the tool from the
//!   tree the image boots.
//!
//! **One open volume per disk per process.** The image device's lock is a
//! POSIX record lock, which belongs to the process and is dropped when any
//! of its descriptors of the file closes. Two calls on one disk from one
//! process therefore never refuse each other — a [`put`] while a [`read`]
//! session holds the same image would write under the reader — and the
//! first to finish drops the other's lock, so a machine could then start
//! on a disk the process still has open. A host program serializes its own
//! calls per disk ([`VolError::is_in_use`] answers only for other processes).
//!
//! Every call runs the engine on a thread of its own with
//! [`VOLUME_STACK_BYTES`] of stack — the engine builds its tables on the
//! calling thread's stack while it recovers a volume — so a caller needs no
//! stack of its own beyond its usual one.

#[cfg(target_os = "linux")]
mod device;
mod source;
mod volume;

pub use kore_svc_dagfs::{
    Name,
    bounds::{MAX_DIR_ENTRIES, MAX_OBJECTS, MAX_PATH_DEPTH},
};
pub use volume::{
    Formatted, Installed, Kind, Listed, Listing, Mount, MountStatus, Removed, VolError, Walked,
    format, put, read, rm,
};

/// The on-disk format version this build's commits write. The engine mounts
/// a volume of this version or an older one it still reads, and refuses a
/// newer one as `Incompatible`: build the tool from the tree the image boots.
pub const FORMAT_VERSION: u32 = kore_svc_dagfs::persistence::FORMAT_VERSION;

/// The logical block size an image is read with unless told otherwise:
/// QEMU's, for every disk image it creates, and so what kore formats a
/// QEMU disk with. A disk is read at its own, which a named block size must
/// agree with.
pub const DEFAULT_BLOCK_SIZE: u32 = 512;

/// The engine builds its tables on the calling thread's stack while it
/// recovers a volume; the filesystem row places them on its heap for the
/// same reason (`DurableService::initialize`). A host thread's default 2 MiB
/// is not enough, 16 MiB is ample.
pub const VOLUME_STACK_BYTES: usize = 16 << 20;

/// Whether `face` names a terminal: one whose home (`work/face-N`) the
/// filesystem row roots it at.
pub fn is_face(face: u8) -> bool {
    kore_svc_dagfs::face_home_label(face).is_some()
}

/// Run `work` on a thread with [`VOLUME_STACK_BYTES`] of stack, borrowing
/// what it needs from the caller, and hand back what it answered.
fn on_volume_stack<T: Send>(
    work: impl FnOnce() -> Result<T, VolError> + Send,
) -> Result<T, VolError> {
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .name("kore-vol".into())
            .stack_size(VOLUME_STACK_BYTES)
            .spawn_scoped(scope, work)
            .map_err(|error| VolError::Host(format!("cannot start the volume thread: {error}")))?
            .join()
            .unwrap_or_else(|_| Err(VolError::Host("the volume thread panicked".into())))
    })
}
