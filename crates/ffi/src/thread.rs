//! Pthread-shaped lifecycle over the native process-lifetime controller.
extern crate alloc as rust_alloc;
use core::sync::atomic::{AtomicPtr, Ordering};
use core::{
    alloc::{Allocator, Layout},
    ffi::c_void,
    ptr::{self, NonNull},
};
use kore::runtime::thread_controller::{Controller, Error, Id};
use rust_alloc::{alloc::Global, sync::Arc};

/// Opaque join handle. Transfer between native workers; do not copy ownership.
/// Successful join/detach consumes it. Concurrent operations on the same handle
/// or use after consumption are invalid, including through a copied pointer.
pub struct KoreThread {
    _private: [u8; 0],
}
/// Reserved attributes. Only a null attributes pointer is currently supported.
pub struct KoreThreadAttr {
    _private: [u8; 0],
}
pub type KoreThreadStart = Option<unsafe extern "C" fn(*mut c_void) -> *mut c_void>;
pub const KORE_THREAD_OK: i32 = 0;
pub const KORE_THREAD_ESRCH: i32 = 3;
pub const KORE_THREAD_EAGAIN: i32 = 11;
pub const KORE_THREAD_ENOMEM: i32 = 12;
pub const KORE_THREAD_EBUSY: i32 = 16;
pub const KORE_THREAD_EINVAL: i32 = 22;
pub const KORE_THREAD_EDEADLK: i32 = 35;
pub const KORE_THREAD_ECANCELED: i32 = 125;

static THREADS: Controller = Controller::new();
struct Ticket {
    id: Id,
    result: Arc<AtomicPtr<c_void>>,
}
struct StartContext {
    call: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    argument: *mut c_void,
    result: Arc<AtomicPtr<c_void>>,
}
fn status(error: Error) -> i32 {
    match error {
        Error::Unavailable | Error::Full => KORE_THREAD_EAGAIN,
        Error::Invalid => KORE_THREAD_ESRCH,
        Error::Busy => KORE_THREAD_EBUSY,
        Error::Deadlock => KORE_THREAD_EDEADLK,
        Error::Terminated => KORE_THREAD_ECANCELED,
    }
}
fn allocate<T>() -> Option<NonNull<T>> {
    Global.allocate(Layout::new::<T>()).ok().map(NonNull::cast)
}
unsafe fn release<T>(pointer: NonNull<T>) {
    // SAFETY: caller owns this initialized allocation after every user stopped.
    unsafe {
        ptr::drop_in_place(pointer.as_ptr());
        Global.deallocate(pointer.cast(), Layout::new::<T>());
    }
}
fn invoke(argument: usize) -> usize {
    // SAFETY: create transfers this initialized context to native custody;
    // cleanup cannot run until this callback's thread has been reaped.
    let context = unsafe { &*(argument as *const StartContext) };
    call_context(context)
}
fn call_context(context: &StartContext) -> usize {
    // SAFETY: forwarded foreign callback and argument lifetime contract.
    let value = unsafe { (context.call)(context.argument) };
    context.result.store(value, Ordering::Release);
    0
}
fn retire(argument: usize) {
    // SAFETY: called exactly once by native owner after successful reclamation.
    unsafe {
        release(NonNull::new_unchecked(argument as *mut StartContext));
    }
}

/// Start an independent native thread; return zero or a KORE_THREAD_E* code.
/// Default attributes only. Lazily starts the process-lifetime controller.
/// Uses the installed process allocator. A failed create leaves out unchanged
/// and never retains the caller's argument. Calls block on native workers;
/// they must not run inside an executor callback that owes sibling polls.
/// # Safety
/// out is aligned, writable and disjoint from callback storage. start/argument
/// remain valid through callback termination, including after detach. The
/// callback synchronizes shared data and must not unwind or longjmp across Rust.
/// Its returned pointer must remain valid for the joining caller. No TLS
/// destructors, cancellation, explicit pthread_exit or custom stacks are supplied.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_thread_create(
    out: *mut *mut KoreThread,
    attributes: *const KoreThreadAttr,
    start: KoreThreadStart,
    argument: *mut c_void,
) -> i32 {
    let Some(call) = start else {
        return KORE_THREAD_EINVAL;
    };
    if out.is_null() || !attributes.is_null() {
        return KORE_THREAD_EINVAL;
    }
    match THREADS.start() {
        Ok(()) | Err(Error::Busy) => {}
        Err(error) => return status(error),
    }
    let Ok(result) = Arc::try_new(AtomicPtr::new(ptr::null_mut())) else {
        return KORE_THREAD_ENOMEM;
    };
    let Some(ticket) = allocate::<Ticket>() else {
        return KORE_THREAD_ENOMEM;
    };
    let Some(context) = allocate::<StartContext>() else {
        // SAFETY: ticket is uninitialized; release allocation without dropping.
        unsafe {
            Global.deallocate(ticket.cast(), Layout::new::<Ticket>());
        }
        return KORE_THREAD_ENOMEM;
    };
    // SAFETY: fresh allocation, no alias is exposed yet.
    unsafe {
        context.as_ptr().write(StartContext {
            call,
            argument,
            result: result.clone(),
        });
    }
    let id = match THREADS.create_with_cleanup(invoke, context.as_ptr() as usize, Some(retire)) {
        Ok(id) => id,
        Err(error) => {
            // SAFETY: failed create retained context custody and launched no callback.
            unsafe {
                release(context);
                Global.deallocate(ticket.cast(), Layout::new::<Ticket>());
            }
            return status(error);
        }
    };
    // SAFETY: ticket allocation remains exclusively ours; caller provided out.
    unsafe {
        ticket.as_ptr().write(Ticket { id, result });
        out.write(ticket.as_ptr().cast());
    }
    KORE_THREAD_OK
}

/// Join from any native worker. Success and ECANCELED consume the handle; other
/// errors retain it for retry. A null result pointer discards the return value.
/// # Safety
/// thread is a live uniquely owned handle from create, with no concurrent
/// join/detach. result, when nonnull, is aligned, writable and disjoint from the
/// handle. See create's callback and returned-pointer lifetime contracts.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_thread_join(
    thread: *mut KoreThread,
    result: *mut *mut c_void,
) -> i32 {
    let Some(ticket) = NonNull::new(thread.cast::<Ticket>()) else {
        return KORE_THREAD_EINVAL;
    };
    // SAFETY: caller supplies a live exclusive handle; native Id is an identifier.
    let joined = THREADS.join(unsafe { ticket.as_ref().id });
    match joined {
        Ok(_) => {
            // SAFETY: native reclamation completed; result is caller-owned output.
            unsafe {
                if !result.is_null() {
                    result.write(ticket.as_ref().result.load(Ordering::Acquire));
                }
                release(ticket);
            }
            KORE_THREAD_OK
        }
        Err(Error::Terminated) => {
            // SAFETY: native thread ended without returning, but was reclaimed.
            unsafe {
                release(ticket);
            }
            KORE_THREAD_ECANCELED
        }
        Err(error) => status(error),
    }
}

/// Relinquish joining. Success consumes the handle; the controller retains and
/// reaps the native thread and its callback context after termination.
/// # Safety
/// thread is a live uniquely owned handle from create, with no concurrent
/// join/detach. Caller-owned callback storage remains valid through termination.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_thread_detach(thread: *mut KoreThread) -> i32 {
    let Some(ticket) = NonNull::new(thread.cast::<Ticket>()) else {
        return KORE_THREAD_EINVAL;
    };
    // SAFETY: caller supplies a live exclusive handle.
    match THREADS.detach(unsafe { ticket.as_ref().id }) {
        Ok(()) => {
            // SAFETY: only this handle is freed; native owner retains the context.
            unsafe {
                release(ticket);
            }
            KORE_THREAD_OK
        }
        Err(error) => status(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn returned_pointer_outlives_callback_context_with_provenance() {
        unsafe extern "C" fn identity(pointer: *mut c_void) -> *mut c_void {
            pointer
        }
        let mut value = 17usize;
        let result = Arc::try_new(AtomicPtr::new(ptr::null_mut())).unwrap();
        let context = StartContext {
            call: identity,
            argument: (&raw mut value).cast(),
            result: result.clone(),
        };
        assert_eq!(call_context(&context), 0);
        drop(context);
        // SAFETY: result retains the original pointer to live, exclusive value.
        unsafe {
            *result.load(Ordering::Acquire).cast::<usize>() = 23;
        }
        assert_eq!(value, 23);
        assert_eq!(Arc::strong_count(&result), 1);
    }
    #[test]
    fn host_refusal_preserves_output_and_never_calls_foreign_code() {
        unsafe extern "C" fn never(_: *mut c_void) -> *mut c_void {
            panic!("host ran native callback")
        }
        let sentinel = ptr::without_provenance_mut(16);
        let mut output = sentinel;
        // SAFETY: valid disjoint output; callbacks cannot run on the host.
        unsafe {
            assert_eq!(
                kore_thread_create(&mut output, ptr::null(), Some(never), ptr::null_mut()),
                KORE_THREAD_EAGAIN
            );
            assert_eq!(output, sentinel);
            assert_eq!(
                kore_thread_create(&mut output, ptr::null(), None, ptr::null_mut()),
                KORE_THREAD_EINVAL
            );
            assert_eq!(
                kore_thread_join(ptr::null_mut(), ptr::null_mut()),
                KORE_THREAD_EINVAL
            );
            assert_eq!(kore_thread_detach(ptr::null_mut()), KORE_THREAD_EINVAL);
        }
    }
}

/// Optional freestanding linker spelling.
/// # Safety
/// Same contract as [`kore_thread_create`].
#[cfg(all(feature = "c-stdlib", target_os = "none"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pthread_create(
    out: *mut *mut KoreThread,
    attr: *const KoreThreadAttr,
    start: KoreThreadStart,
    arg: *mut c_void,
) -> i32 {
    unsafe { kore_thread_create(out, attr, start, arg) }
}
/// Optional freestanding linker spelling.
/// # Safety
/// Same contract as [`kore_thread_join`].
#[cfg(all(feature = "c-stdlib", target_os = "none"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pthread_join(thread: *mut KoreThread, result: *mut *mut c_void) -> i32 {
    unsafe { kore_thread_join(thread, result) }
}
/// Optional freestanding linker spelling.
/// # Safety
/// Same contract as [`kore_thread_detach`].
#[cfg(all(feature = "c-stdlib", target_os = "none"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pthread_detach(thread: *mut KoreThread) -> i32 {
    unsafe { kore_thread_detach(thread) }
}
