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
}
