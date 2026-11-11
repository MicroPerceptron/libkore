//! The C entry points `kore/program.h` declares: what an independent C program
//! on Kore adds to `kore.h` by linking `libkore.so`.
//!
//! Compiled on every target so the header has one source of declarations;
//! off Kore there is no program to host, and each entry refuses.

use core::ffi::c_void;

use kore_ffi::{KoreFsError, KoreStatus};

/// `kore.h`'s `KoreFsStep` as a C caller may pass it: possibly null.
pub type KoreFsStep = Option<kore_ffi::KoreFsStep>;

/// The stack `main` runs on: a guarded strip of the task's stack window,
/// committed whole when the program starts. A frame deeper than this faults
/// on the guard page below it, as a stack overflow does.
///
/// The size of the strip `tcc`, a large C program whose depth its input
/// decides, measured its need under Kore's runtime at and runs in; ISO C's
/// own guarantee of 127 nested blocks (C11 §5.2.4.1) needs a small part of
/// it. Larger strips come out of the 16 MiB window the program's SPMD gangs
/// share, and would be committed at every launch.
pub const KORE_MAIN_STACK_BYTES: usize = 1 << 20;

/// Run `step` as this program's filesystem session, through the filesystem
/// face its launch carried, rooted where it was launched (its working
/// directory). The callbacks and their pointers obey `KoreFsStep`'s contract:
/// each callback runs on the program's host thread, as on any native
/// executor, while the caller waits, so a callback must not depend on the
/// caller's thread-local state.
///
/// Returns `KORE_STATUS_OK` once the callback program ended and every view it
/// left open closed. `KORE_STATUS_QUARANTINED`: a reply was lost or
/// uncertain, and no session can run again in this program.
/// `KORE_STATUS_REFUSED`: a close was refused (`*error` names why, and the
/// view stays open), the launch carried no filesystem face
/// (`KORE_FS_ERROR_UNSUPPORTED`), or the caller is not the thread `main` runs
/// on, which alone waits for the host. `KORE_STATUS_CAPACITY`: the native
/// owner could not be formed. `KORE_STATUS_INVALID_ARGUMENT`: `step` is null.
/// `*error` is written whenever `error` is not null.
///
/// # Safety
/// `step` obeys `KoreFsStep`'s pointer, serialization and no-unwind contract;
/// `user` remains valid, and the callbacks' alone, until this returns. `error`
/// is null or valid for one aligned write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_fs_run(
    step: KoreFsStep,
    user: *mut c_void,
    error: *mut KoreFsError,
) -> KoreStatus {
    let (status, cause) = match step {
        None => (KoreStatus::InvalidArgument, KoreFsError::None),
        // SAFETY: the caller's contract, forwarded whole.
        #[cfg(target_os = "none")]
        Some(step) => unsafe { crate::program::session(step, user) },
        // No program to host off Kore.
        #[cfg(not(target_os = "none"))]
        Some(_) => {
            let _ = user;
            (KoreStatus::Refused, KoreFsError::None)
        }
    };
    if !error.is_null() {
        // SAFETY: the caller's contract: null or valid for one aligned write.
        unsafe { error.write(cause) };
    }
    status
}

/// [`KoreFdInfo::kind`]: the memory supply the program's allocator grows
/// on. `mmap` answers a fresh, zeroed region of it.
pub const KORE_FD_MEMORY: u32 = 1;
/// [`KoreFdInfo::access`]: the capability may be read.
pub const KORE_FD_READ: u32 = 1 << 0;
/// [`KoreFdInfo::access`]: the capability may be written.
pub const KORE_FD_WRITE: u32 = 1 << 1;

/// What a descriptor names, as the calling thread holds it
/// ([`kore_fd_describe`]).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KoreFdInfo {
    /// What kind of capability it is: `KORE_FD_MEMORY`.
    pub kind: u32,
    /// What the capability's own rights permit this thread, as
    /// `KORE_FD_READ | KORE_FD_WRITE` bits: the rights it was founded with,
    /// which the access `open` asked for can only narrow.
    pub access: u32,
    /// For `KORE_FD_MEMORY`, the supply's whole length in bytes: what the
    /// allocator and every mapping share, not what is still free.
    pub length: u64,
}

/// Describe the capability behind `fd`, a descriptor `open` answered
/// (`<fcntl.h>`): `open` finds a capability this program was founded holding
/// by its name, and this reads what was found. Nothing is granted or changed.
///
/// Capabilities are held per thread, so the answer is the calling thread's:
/// the same descriptor may describe a capability on the thread that opened
/// it and be refused on one founded without it.
///
/// Returns `KORE_STATUS_OK` with `*out` written.
/// `KORE_STATUS_INVALID_ARGUMENT`: `out` is null. `KORE_STATUS_STALE_HANDLE`:
/// `fd` is not an open descriptor. `KORE_STATUS_REFUSED`: the calling thread
/// holds no such capability, or this is not a Kore program. A refusal leaves
/// `*out` unchanged.
///
/// # Safety
/// `out` is null or valid for one aligned write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_fd_describe(
    fd: core::ffi::c_int,
    out: *mut KoreFdInfo,
) -> KoreStatus {
    if out.is_null() {
        return KoreStatus::InvalidArgument;
    }
    let Ok((row, _)) = crate::fd::TABLE.get(fd) else {
        return KoreStatus::StaleHandle;
    };
    let Some(info) = describe(&crate::fd::NAMES[row]) else {
        return KoreStatus::Refused;
    };
    // SAFETY: the caller's contract: valid for one aligned write.
    unsafe { out.write(info) };
    KoreStatus::Ok
}

/// What the calling thread holds at `name`.
#[cfg(target_os = "none")]
fn describe(name: &crate::fd::Name) -> Option<KoreFdInfo> {
    use crate::fd::Kind;
    let held = crate::fd::holding(name)?;
    let (kind, length) = match name.kind {
        Kind::Memory => (
            KORE_FD_MEMORY,
            kore_std::runtime::cap::verbs::lease_geometry(held.slot)?.1,
        ),
    };
    Some(KoreFdInfo {
        kind,
        access: if held.readable { KORE_FD_READ } else { 0 }
            | if held.writable { KORE_FD_WRITE } else { 0 },
        length,
    })
}

/// Off Kore no thread holds a capability.
#[cfg(not(target_os = "none"))]
fn describe(_name: &crate::fd::Name) -> Option<KoreFdInfo> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn never(
        _: *mut kore_ffi::KoreFsContext,
        _: *const kore_ffi::KoreFsCompletion,
        _: *mut c_void,
    ) {
        panic!("a session ran with no program to host");
    }

    #[test]
    fn a_null_step_is_an_invalid_argument_and_the_error_is_written() {
        let mut error = KoreFsError::Busy;
        // SAFETY: a null step names no callback; `error` is valid.
        let status = unsafe { kore_fs_run(None, core::ptr::null_mut(), &mut error) };
        assert_eq!(status, KoreStatus::InvalidArgument);
        assert_eq!(error, KoreFsError::None);
    }

    #[test]
    fn with_no_program_to_host_a_session_is_refused_and_never_runs() {
        let mut error = KoreFsError::Busy;
        // SAFETY: `never` is never called; `error` is valid.
        let status = unsafe { kore_fs_run(Some(never), core::ptr::null_mut(), &mut error) };
        assert_eq!(status, KoreStatus::Refused);
        assert_eq!(error, KoreFsError::None);
        // A null error pointer is permitted.
        // SAFETY: as above.
        let status =
            unsafe { kore_fs_run(Some(never), core::ptr::null_mut(), core::ptr::null_mut()) };
        assert_eq!(status, KoreStatus::Refused);
    }

    #[test]
    fn describing_refuses_by_cause_and_leaves_the_answer_alone() {
        let untouched = KoreFdInfo {
            kind: 7,
            access: 7,
            length: 7,
        };
        let mut info = untouched;
        // SAFETY: null is permitted; `info` is valid.
        unsafe {
            assert_eq!(
                kore_fd_describe(3, core::ptr::null_mut()),
                KoreStatus::InvalidArgument
            );
            // The streams' numbers and an unopened one are not descriptors.
            for fd in [0, 1, 2, -1, 63, 1 << 20] {
                assert_eq!(kore_fd_describe(fd, &mut info), KoreStatus::StaleHandle);
            }
        }
        // An open descriptor, off Kore: no thread holds a capability.
        let access = crate::fd::Access::of(crate::fd::O_RDWR).unwrap();
        let fd = crate::fd::TABLE.claim(0, access).unwrap();
        // SAFETY: `info` is valid.
        assert_eq!(
            unsafe { kore_fd_describe(fd, &mut info) },
            KoreStatus::Refused
        );
        crate::fd::TABLE.release(fd).unwrap();
        assert_eq!(info, untouched);
    }
}
