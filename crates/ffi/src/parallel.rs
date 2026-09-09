use crate::KoreStatus;
use core::ffi::c_void;

/// Called on native worker threads. Its range is a disjoint partition of the
/// requested tile span, not a worker identifier or a CPU number.
pub type KoreSpmdStep = Option<unsafe extern "C" fn(*mut c_void, u32, u32)>;
pub type KoreThreadStep = Option<unsafe extern "C" fn(*mut c_void)>;
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct KoreDispatch {
    pub workers: u32,
    pub tiles: u32,
}

struct SharedCallback {
    step: unsafe extern "C" fn(*mut c_void, u32, u32),
    user: *mut c_void,
}
// SAFETY: the exported unsafe entry requires concurrent calls on user to be
// valid. The native scoped dispatch joins all workers before this owner returns.
unsafe impl Sync for SharedCallback {}
impl SharedCallback {
    fn call(&self, range: core::ops::Range<u32>) {
        // SAFETY: forwarded foreign thread-safety and lifetime contract.
        unsafe {
            (self.step)(self.user, range.start, range.end);
        }
    }
}
/// Run a joined SPMD region through the existing native founding and width policy.
/// Refused means no callback was launched. A post-launch join refusal retains
/// the borrowed frame in the native owner; it never returns for serial replay.
/// This blocks the calling thread: use a worker, not an executor callback.
/// # Safety
/// step and user remain valid until return; calls may run concurrently over
/// disjoint tile ranges. Shared data must be synchronized. No unwind/longjmp,
/// nested native dispatch, or retained user pointers. out is writable, aligned
/// and disjoint from user storage. No per-worker capabilities are added.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_spmd_dispatch(
    tiles: u32,
    step: KoreSpmdStep,
    user: *mut c_void,
    out: *mut KoreDispatch,
) -> KoreStatus {
    let Some(step) = step else {
        return KoreStatus::InvalidArgument;
    };
    if out.is_null() {
        return KoreStatus::InvalidArgument;
    }
    let callback = SharedCallback { step, user };
    let Some(result) = kore::spmd::dispatch_slices(tiles, |range| callback.call(range)) else {
        return KoreStatus::Refused;
    };
    // SAFETY: caller provides disjoint aligned output; native join completed.
    unsafe {
        out.write(KoreDispatch {
            workers: result.width,
            tiles: result.tiles,
        });
    }
    KoreStatus::Ok
}
struct SingleCallback {
    step: unsafe extern "C" fn(*mut c_void),
    user: *mut c_void,
}
// SAFETY: thread_run requires the user storage to be valid on its worker until
// native join; width one guarantees the callback never runs concurrently itself.
unsafe impl Sync for SingleCallback {}
impl SingleCallback {
    fn call(&self) {
        // SAFETY: the exported scope holds user storage until native join.
        unsafe {
            (self.step)(self.user);
        }
    }
}
/// Execute one callback on a native worker and join it before returning.
/// No detached owner, native ticket fabrication, or callback lifetime extension.
/// # Safety
/// step/user are valid on the worker until return, with no unwind/longjmp or
/// retained pointers. The caller must not concurrently access unsynchronized
/// user storage. This blocks; do not call from a native executor callback.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_thread_run(step: KoreThreadStep, user: *mut c_void) -> KoreStatus {
    let Some(step) = step else {
        return KoreStatus::InvalidArgument;
    };
    let callback = SingleCallback { step, user };
    match kore::spmd::dispatch_slices_at(1, 1, |_| callback.call()) {
        Some(_) => KoreStatus::Ok,
        None => KoreStatus::Refused,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn host_refusal_does_not_invoke_a_worker_or_publish_output() {
        unsafe extern "C" fn no_worker(_: *mut c_void, _: u32, _: u32) {
            panic!("host worker executed");
        }
        unsafe extern "C" fn no_thread(_: *mut c_void) {
            panic!("host thread executed");
        }
        let mut result = KoreDispatch {
            workers: 99,
            tiles: 88,
        };
        // SAFETY: callbacks never dereference user and host native dispatch refuses.
        unsafe {
            assert_eq!(
                kore_spmd_dispatch(4, Some(no_worker), core::ptr::null_mut(), &mut result),
                KoreStatus::Refused
            );
            assert_eq!(
                kore_thread_run(Some(no_thread), core::ptr::null_mut()),
                KoreStatus::Refused
            );
        }
        assert_eq!((result.workers, result.tiles), (99, 88));
    }
}
