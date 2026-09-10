//! Foreign managed-thread teardown. Native capabilities stay with Controller.
use crate::thread::Control;
use crate::{KORE_THREAD_EAGAIN, KORE_THREAD_EINVAL, KORE_THREAD_ENOTSUP, KORE_THREAD_OK};
use core::{
    cell::Cell,
    ffi::c_void,
    ptr,
    sync::atomic::{AtomicPtr, AtomicU64, Ordering},
};

pub const KORE_THREAD_KEYS_MAX: usize = 128;
pub const KORE_THREAD_DESTRUCTOR_ITERATIONS: usize = 4;
pub const KORE_THREAD_TLS_DESTRUCTORS_MAX: usize = 128;
pub const KORE_THREAD_CANCEL_ENABLE: i32 = 0;
pub const KORE_THREAD_CANCEL_DISABLE: i32 = 1;
pub const KORE_THREAD_CANCEL_DEFERRED: i32 = 0;
pub const KORE_THREAD_CANCEL_ASYNCHRONOUS: i32 = 1;
pub type KoreThreadKey = u64;
pub type KoreThreadDestructor = Option<unsafe extern "C" fn(*mut c_void)>;

struct KeySlot {
    state: AtomicU64,
    destructor: AtomicPtr<c_void>,
}
impl KeySlot {
    const fn new() -> Self {
        Self {
            state: AtomicU64::new(0),
            destructor: AtomicPtr::new(ptr::null_mut()),
        }
    }
}
static KEYS: [KeySlot; KORE_THREAD_KEYS_MAX] = [const { KeySlot::new() }; KORE_THREAD_KEYS_MAX];
#[derive(Clone, Copy)]
struct Destructor {
    call: KoreThreadDestructor,
    argument: *mut c_void,
}
const EMPTY: Destructor = Destructor {
    call: None,
    argument: ptr::null_mut(),
};

/// Caller-owned cleanup node. Fields are private implementation state: initialize
/// through cleanup_push, keep at a stable address, and pop in LIFO order.
#[repr(C)]
pub struct KoreThreadCleanup {
    previous: *mut KoreThreadCleanup,
    call: KoreThreadDestructor,
    argument: *mut c_void,
}

#[thread_local]
static CURRENT: Cell<*const Control> = Cell::new(ptr::null());
#[thread_local]
static DISABLED: Cell<bool> = Cell::new(false);
#[thread_local]
static FINISHING: Cell<bool> = Cell::new(false);
#[thread_local]
static CLEANUP: Cell<*mut KoreThreadCleanup> = Cell::new(ptr::null_mut());
#[thread_local]
static VALUES: [Cell<(u64, *mut c_void)>; KORE_THREAD_KEYS_MAX] =
    [const { Cell::new((0, ptr::null_mut())) }; KORE_THREAD_KEYS_MAX];
#[thread_local]
static DESTRUCTORS: [Cell<Destructor>; KORE_THREAD_TLS_DESTRUCTORS_MAX] =
    [const { Cell::new(EMPTY) }; KORE_THREAD_TLS_DESTRUCTORS_MAX];
#[thread_local]
static DTOR_COUNT: Cell<usize> = Cell::new(0);

pub(crate) fn enter(control: &Control) {
    assert!(CURRENT.get().is_null());
    CURRENT.set(control);
    DISABLED.set(false);
    FINISHING.set(false);
}
pub(crate) fn leave() {
    CURRENT.set(ptr::null());
}
pub(crate) fn cancellable() -> bool {
    !CURRENT.get().is_null() && !DISABLED.get() && !FINISHING.get()
}
fn active() -> bool {
    !CURRENT.get().is_null()
}
fn key_slot(key: u64) -> Option<(usize, &'static KeySlot)> {
    let index = (key % KORE_THREAD_KEYS_MAX as u64) as usize;
    let state = key / KORE_THREAD_KEYS_MAX as u64;
    let slot = &KEYS[index];
    (state & 3 == 2 && slot.state.load(Ordering::Acquire) == state).then_some((index, slot))
}

/// Create a process-local TLS key. No allocator; bounded to KEYS_MAX live keys.
/// Deleted keys are generation-checked; reuse cannot expose an old TLS value.
/// # Safety
/// out is aligned writable storage. Destructor code remains valid until every
/// possible invocation has finished, including concurrent delete/exit races.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_thread_key_create(
    out: *mut KoreThreadKey,
    destructor: KoreThreadDestructor,
) -> i32 {
    if out.is_null() {
        return KORE_THREAD_EINVAL;
    }
    for (index, slot) in KEYS.iter().enumerate() {
        let old = slot.state.load(Ordering::Acquire);
        if old & 3 != 0 || old > u64::MAX / KORE_THREAD_KEYS_MAX as u64 - 4 {
            continue;
        }
        if slot
            .state
            .compare_exchange(old, old | 1, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            continue;
        }
        // A reader that observes a recycled callback must also observe this
        // generation's BUILD state when it rechecks the key after the load.
        slot.destructor.store(
            destructor.map_or(ptr::null_mut(), |call| call as *mut c_void),
            Ordering::Release,
        );
        slot.state.store(old | 2, Ordering::Release);
        unsafe {
            out.write((old | 2) * KORE_THREAD_KEYS_MAX as u64 + index as u64);
        }
        return KORE_THREAD_OK;
    }
    KORE_THREAD_EAGAIN
}
/// Delete a key without invoking destructors. Values in other threads become
/// inaccessible; callers retain responsibility for any associated allocations.
#[unsafe(no_mangle)]
pub extern "C" fn kore_thread_key_delete(key: KoreThreadKey) -> i32 {
    let Some((_, slot)) = key_slot(key) else {
        return KORE_THREAD_EINVAL;
    };
    let old = key / KORE_THREAD_KEYS_MAX as u64;
    match slot
        .state
        .compare_exchange(old, old + 2, Ordering::AcqRel, Ordering::Acquire)
    {
        Ok(_) => KORE_THREAD_OK,
        Err(_) => KORE_THREAD_EINVAL,
    }
}
/// Set this managed native thread's value. Null suppresses its destructor.
/// # Safety
/// value and registered destructor obey the application's lifetime contract.
/// The function retains this pointer until replacement, deletion or thread exit.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_thread_setspecific(key: KoreThreadKey, value: *const c_void) -> i32 {
    if !active() {
        return KORE_THREAD_EAGAIN;
    }
    let Some((index, _)) = key_slot(key) else {
        return KORE_THREAD_EINVAL;
    };
    VALUES[index].set((key, value.cast_mut()));
    KORE_THREAD_OK
}
/// Return this managed thread's value, or null for unset/stale/unmanaged keys.
#[unsafe(no_mangle)]
pub extern "C" fn kore_thread_getspecific(key: KoreThreadKey) -> *mut c_void {
    if !active() {
        return ptr::null_mut();
    }
    let Some((index, _)) = key_slot(key) else {
        return ptr::null_mut();
    };
    let (stored, value) = VALUES[index].get();
    if stored == key {
        value
    } else {
        ptr::null_mut()
    }
}
/// Register a compiler TLS destructor in reverse-registration exit order.
/// Returns EAGAIN for unmanaged threads or exhausted bounded registration space.
/// # Safety
/// argument and destructor code must remain valid until invoked on this thread.
/// Destructors must return normally; they must not exit, unwind or longjmp.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_thread_atexit(
    destructor: KoreThreadDestructor,
    argument: *mut c_void,
) -> i32 {
    if !active() {
        return KORE_THREAD_EAGAIN;
    }
    if destructor.is_none() {
        return KORE_THREAD_EINVAL;
    }
    let count = DTOR_COUNT.get();
    if count == KORE_THREAD_TLS_DESTRUCTORS_MAX {
        return KORE_THREAD_EAGAIN;
    }
    DESTRUCTORS[count].set(Destructor {
        call: destructor,
        argument,
    });
    DTOR_COUNT.set(count + 1);
    KORE_THREAD_OK
}
/// Push a caller-owned cleanup handler. It runs on cancellation/explicit exit,
/// or an executing pop; ordinary callback return does not invoke handlers.
/// # Safety
/// node is exclusive writable storage, not already linked, remaining at a stable
/// address until popped or invoked. Handler and argument remain live throughout.
/// Handlers must return normally and may not exit, unwind or longjmp.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_thread_cleanup_push(
    node: *mut KoreThreadCleanup,
    call: KoreThreadDestructor,
    argument: *mut c_void,
) -> i32 {
    if node.is_null() || call.is_none() {
        return KORE_THREAD_EINVAL;
    }
    if !active() || FINISHING.get() {
        return KORE_THREAD_EAGAIN;
    }
    unsafe {
        node.write(KoreThreadCleanup {
            previous: CLEANUP.get(),
            call,
            argument,
        });
    }
    CLEANUP.set(node);
    KORE_THREAD_OK
}
/// Pop the most recent cleanup handler, optionally invoking it.
/// # Safety
/// node is the live top node returned by cleanup_push. Handler contract as above.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_thread_cleanup_pop(
    node: *mut KoreThreadCleanup,
    execute: i32,
) -> i32 {
    if node.is_null() || CLEANUP.get() != node {
        return KORE_THREAD_EINVAL;
    }
    let (previous, call, argument) = unsafe { ((*node).previous, (*node).call, (*node).argument) };
    CLEANUP.set(previous);
    if execute != 0 {
        unsafe {
            call.unwrap()(argument);
        }
    }
    KORE_THREAD_OK
}

pub(crate) fn finish(cleanup: bool) {
    FINISHING.set(true);
    DISABLED.set(true);
    if cleanup {
        while !CLEANUP.get().is_null() {
            unsafe {
                kore_thread_cleanup_pop(CLEANUP.get(), 1);
            }
        }
    } else {
        CLEANUP.set(ptr::null_mut());
    }
    run_dtors();
    // pthread keys clear BEFORE invoking, so repopulation is observable.
    for _ in 0..KORE_THREAD_DESTRUCTOR_ITERATIONS {
        let mut invoked = false;
        for (index, value) in VALUES.iter().enumerate() {
            let (key, pointer) = value.replace((0, ptr::null_mut()));
            if pointer.is_null() {
                continue;
            }
            let Some((_, slot)) = key_slot(key) else {
                continue;
            };
            let call = slot.destructor.load(Ordering::Acquire);
            if key_slot(key).is_none() || call.is_null() {
                continue;
            }
            debug_assert_eq!(index, (key % KORE_THREAD_KEYS_MAX as u64) as usize);
            let call: unsafe extern "C" fn(*mut c_void) = unsafe { core::mem::transmute(call) };
            unsafe {
                call(pointer);
            }
            invoked = true;
        }
        // Key destructors can construct compiler TLS objects. Honor them without
        // restarting the bounded number of key-destructor passes.
        run_dtors();
        if !invoked {
            break;
        }
    }
    for value in &VALUES {
        value.set((0, ptr::null_mut()));
    }
}
fn run_dtors() {
    while DTOR_COUNT.get() != 0 {
        let index = DTOR_COUNT.get() - 1;
        DTOR_COUNT.set(index);
        let destructor = DESTRUCTORS[index].replace(EMPTY);
        unsafe {
            destructor.call.unwrap()(destructor.argument);
        }
    }
}

/// Sentinel returned by joining a cooperatively canceled managed thread.
#[unsafe(no_mangle)]
pub extern "C" fn kore_thread_canceled() -> *mut c_void {
    ptr::without_provenance_mut(usize::MAX)
}
/// Enable/disable deferred cancellation. Pending requests survive disable.
/// Re-enabling does not itself cancel; the next cancellation point checks it.
/// # Safety
/// old_state is null or aligned writable output. Managed native threads only.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_thread_setcancelstate(state: i32, old_state: *mut i32) -> i32 {
    if state != KORE_THREAD_CANCEL_ENABLE && state != KORE_THREAD_CANCEL_DISABLE {
        return KORE_THREAD_EINVAL;
    }
    if !active() || FINISHING.get() {
        return KORE_THREAD_EAGAIN;
    }
    let old = DISABLED.replace(state == KORE_THREAD_CANCEL_DISABLE);
    if !old_state.is_null() {
        unsafe {
            old_state.write(i32::from(old));
        }
    }
    KORE_THREAD_OK
}
/// Only deferred cancellation is supported. Asynchronous requests return ENOTSUP
/// without changing state or output; arbitrary interruption would strand owners.
/// # Safety
/// old_type is null or aligned writable storage. Managed native threads only.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_thread_setcanceltype(kind: i32, old_type: *mut i32) -> i32 {
    if kind == KORE_THREAD_CANCEL_ASYNCHRONOUS {
        return KORE_THREAD_ENOTSUP;
    }
    if kind != KORE_THREAD_CANCEL_DEFERRED {
        return KORE_THREAD_EINVAL;
    }
    if !active() || FINISHING.get() {
        return KORE_THREAD_EAGAIN;
    }
    if !old_type.is_null() {
        unsafe {
            old_type.write(KORE_THREAD_CANCEL_DEFERRED);
        }
    }
    KORE_THREAD_OK
}
/// Act on a pending enabled cancellation request. No effect on unmanaged threads.
/// # Safety
/// This may terminate without unwinding caller stack frames. Only use from a
/// foreign managed thread with explicit cleanup handlers for stack-owned resources;
/// never cross live Rust RAII guards or executor callbacks at a cancellation point.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_thread_testcancel() {
    if cancellable() && unsafe { &*CURRENT.get() }.cancel.load(Ordering::Acquire) {
        unsafe {
            kore_thread_exit(kore_thread_canceled());
        }
    }
}
/// Run cleanup handlers and TLS destructors, publish value, then exit this native
/// thread. Joining reports success with value (including the cancellation sentinel).
/// # Safety
/// Must be called on a managed foreign thread, outside teardown. Does not unwind
/// Rust/C++ stack objects. Cleanups and destructors must return normally. value
/// remains valid for the joining caller. Never call from an executor callback.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kore_thread_exit(value: *mut c_void) -> ! {
    assert!(
        active() && !FINISHING.get(),
        "thread_exit outside managed callback"
    );
    finish(true);
    unsafe { &*CURRENT.get() }.finish(value, 2);
    leave();
    kore::thread::exit(kore::ExitCode::OK)
}

/// Compiler ABI TLS registration; DSOs must remain loaded until all registered
/// destructors finish. Kore does not currently unload these images.
/// # Safety
/// Same contract as kore_thread_atexit; dso is an identity only, not owned here.
#[cfg(all(feature = "c-stdlib", target_os = "none"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __cxa_thread_atexit_impl(
    call: KoreThreadDestructor,
    argument: *mut c_void,
    _dso: *mut c_void,
) -> i32 {
    if unsafe { kore_thread_atexit(call, argument) } == 0 {
        0
    } else {
        -1
    }
}
/// Compiler ABI alias; same lifetime contract as __cxa_thread_atexit_impl.
/// # Safety
/// Same contract as kore_thread_atexit; DSO unloading is not supported.
#[cfg(all(feature = "c-stdlib", target_os = "none"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __cxa_thread_atexit(
    call: KoreThreadDestructor,
    argument: *mut c_void,
    dso: *mut c_void,
) -> i32 {
    unsafe { __cxa_thread_atexit_impl(call, argument, dso) }
}

/// Freestanding pthread spelling.
/// # Safety
/// Same contract as kore_thread_cancel.
#[cfg(all(feature = "c-stdlib", target_os = "none"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pthread_cancel(thread: *mut crate::KoreThread) -> i32 {
    unsafe { crate::kore_thread_cancel(thread) }
}

/// Freestanding pthread spelling.
/// # Safety
/// Same contract as kore_thread_key_create.
#[cfg(all(feature = "c-stdlib", target_os = "none"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pthread_key_create(
    out: *mut KoreThreadKey,
    destructor: KoreThreadDestructor,
) -> i32 {
    unsafe { crate::kore_thread_key_create(out, destructor) }
}

/// Freestanding pthread spelling.
/// # Safety
/// Same contract as kore_thread_key_delete.
#[cfg(all(feature = "c-stdlib", target_os = "none"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pthread_key_delete(key: KoreThreadKey) -> i32 {
    crate::kore_thread_key_delete(key)
}

/// Freestanding pthread spelling.
/// # Safety
/// Same contract as kore_thread_setspecific.
#[cfg(all(feature = "c-stdlib", target_os = "none"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pthread_setspecific(key: KoreThreadKey, value: *const c_void) -> i32 {
    unsafe { crate::kore_thread_setspecific(key, value) }
}

/// Freestanding pthread spelling.
/// # Safety
/// Same contract as kore_thread_getspecific.
#[cfg(all(feature = "c-stdlib", target_os = "none"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pthread_getspecific(key: KoreThreadKey) -> *mut c_void {
    crate::kore_thread_getspecific(key)
}

/// Freestanding pthread spelling.
/// # Safety
/// Same contract as kore_thread_setcancelstate.
#[cfg(all(feature = "c-stdlib", target_os = "none"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pthread_setcancelstate(state: i32, old: *mut i32) -> i32 {
    unsafe { crate::kore_thread_setcancelstate(state, old) }
}

/// Freestanding pthread spelling.
/// # Safety
/// Same contract as kore_thread_setcanceltype.
#[cfg(all(feature = "c-stdlib", target_os = "none"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pthread_setcanceltype(kind: i32, old: *mut i32) -> i32 {
    unsafe { crate::kore_thread_setcanceltype(kind, old) }
}

/// Freestanding pthread spelling.
/// # Safety
/// Same contract as kore_thread_testcancel.
#[cfg(all(feature = "c-stdlib", target_os = "none"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pthread_testcancel() -> () {
    unsafe { crate::kore_thread_testcancel() }
}

/// Freestanding pthread spelling.
/// # Safety
/// Same contract as kore_thread_exit.
#[cfg(all(feature = "c-stdlib", target_os = "none"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pthread_exit(value: *mut c_void) -> ! {
    unsafe { crate::kore_thread_exit(value) }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use core::mem::MaybeUninit;
    use std::vec::Vec;

    struct Context(Control);
    impl Context {
        fn new() -> Self {
            Self(Control::new())
        }
        fn enter(&self) {
            enter(&self.0);
        }
    }
    impl Drop for Context {
        fn drop(&mut self) {
            if active() {
                finish(false);
                leave();
            }
        }
    }
    unsafe extern "C" fn record(argument: *mut c_void) {
        let (events, number) = unsafe { &mut *argument.cast::<(Vec<usize>, usize)>() };
        events.push(*number);
        *number += 1;
    }
    #[test]
    fn cleanup_then_compiler_tls_then_key_and_deleted_generation_stays_dead() {
        let context = Context::new();
        context.enter();
        let mut trace = (Vec::<usize>::new(), 0usize);
        let argument = (&raw mut trace).cast();
        let mut key = 0;
        let mut deleted = 0;
        let mut cleanup = MaybeUninit::uninit();
        unsafe {
            assert_eq!(kore_thread_key_create(&mut key, Some(record)), 0);
            assert_eq!(kore_thread_key_create(&mut deleted, Some(record)), 0);
            assert_eq!(kore_thread_setspecific(key, argument), 0);
            assert_eq!(kore_thread_setspecific(deleted, argument), 0);
            assert_eq!(kore_thread_key_delete(deleted), 0);
            let mut replacement = 0;
            assert_eq!(kore_thread_key_create(&mut replacement, Some(record)), 0);
            assert_ne!(replacement, deleted);
            assert!(kore_thread_getspecific(replacement).is_null());
            assert_eq!(
                kore_thread_setspecific(deleted, argument),
                KORE_THREAD_EINVAL
            );
            assert_eq!(
                kore_thread_cleanup_push(cleanup.as_mut_ptr(), Some(record), argument),
                0
            );
            assert_eq!(kore_thread_atexit(Some(record), argument), 0);
            finish(true);
            assert_eq!(trace.0, [0, 1, 2]);
            assert!(kore_thread_getspecific(key).is_null());
            assert_eq!(kore_thread_key_delete(key), 0);
            assert_eq!(kore_thread_key_delete(replacement), 0);
        }
    }
    #[test]
    fn key_repopulation_is_cleared_before_call_and_stops_at_four_passes() {
        struct Value {
            key: u64,
            calls: usize,
        }
        unsafe extern "C" fn again(argument: *mut c_void) {
            let value = unsafe { &mut *argument.cast::<Value>() };
            assert!(kore_thread_getspecific(value.key).is_null());
            value.calls += 1;
            assert_eq!(unsafe { kore_thread_setspecific(value.key, argument) }, 0);
        }
        let context = Context::new();
        context.enter();
        let mut value = Value { key: 0, calls: 0 };
        unsafe {
            assert_eq!(kore_thread_key_create(&mut value.key, Some(again)), 0);
            assert_eq!(
                kore_thread_setspecific(value.key, (&raw mut value).cast()),
                0
            );
        }
        finish(false);
        assert_eq!(value.calls, KORE_THREAD_DESTRUCTOR_ITERATIONS);
        assert!(kore_thread_getspecific(value.key).is_null());
        assert_eq!(kore_thread_key_delete(value.key), 0);
    }
    #[test]
    fn registrations_are_lifo_bounded_and_can_register_during_teardown() {
        struct Value {
            count: usize,
            rearm: bool,
        }
        unsafe extern "C" fn destructor(argument: *mut c_void) {
            let value = unsafe { &mut *argument.cast::<Value>() };
            value.count += 1;
            if value.rearm {
                value.rearm = false;
                assert_eq!(unsafe { kore_thread_atexit(Some(destructor), argument) }, 0);
            }
        }
        let context = Context::new();
        context.enter();
        let mut value = Value {
            count: 0,
            rearm: true,
        };
        for _ in 0..KORE_THREAD_TLS_DESTRUCTORS_MAX {
            assert_eq!(
                unsafe { kore_thread_atexit(Some(destructor), (&raw mut value).cast()) },
                0
            );
        }
        assert_eq!(
            unsafe { kore_thread_atexit(Some(destructor), (&raw mut value).cast()) },
            KORE_THREAD_EAGAIN
        );
        finish(false);
        assert_eq!(value.count, KORE_THREAD_TLS_DESTRUCTORS_MAX + 1);
    }
    #[test]
    fn tls_values_are_native_thread_local_and_cancellation_disable_retains_request() {
        let context = Context::new();
        context.enter();
        let mut key = 0;
        let mut value = 42usize;
        unsafe {
            assert_eq!(kore_thread_key_create(&mut key, None), 0);
            assert_eq!(kore_thread_setspecific(key, (&raw mut value).cast()), 0);
        }
        std::thread::spawn(move || {
            let context = Context::new();
            context.enter();
            assert!(kore_thread_getspecific(key).is_null());
        })
        .join()
        .unwrap();
        assert_eq!(kore_thread_getspecific(key), (&raw mut value).cast());
        context.0.cancel.store(true, Ordering::Release);
        let mut old = -1;
        unsafe {
            assert_eq!(
                kore_thread_setcancelstate(KORE_THREAD_CANCEL_DISABLE, &mut old),
                0
            );
            assert_eq!(old, KORE_THREAD_CANCEL_ENABLE);
            assert!(!cancellable());
            kore_thread_testcancel(); // Disabled cancellation must not terminate.
            assert!(context.0.cancel.load(Ordering::Acquire));
            assert_eq!(
                kore_thread_setcanceltype(KORE_THREAD_CANCEL_ASYNCHRONOUS, &mut old),
                KORE_THREAD_ENOTSUP
            );
            assert_eq!(old, KORE_THREAD_CANCEL_ENABLE); // failure preserves output
            assert_eq!(
                kore_thread_setcancelstate(KORE_THREAD_CANCEL_ENABLE, &mut old),
                0
            );
            assert!(cancellable());
        }
        assert_eq!(kore_thread_key_delete(key), 0);
    }
    #[test]
    fn cleanup_pop_order_and_normal_return_do_not_run_unpopped_handlers() {
        let context = Context::new();
        context.enter();
        let mut trace = (Vec::<usize>::new(), 0usize);
        let argument = (&raw mut trace).cast();
        let mut first = MaybeUninit::uninit();
        let mut second = MaybeUninit::uninit();
        unsafe {
            assert_eq!(
                kore_thread_cleanup_push(first.as_mut_ptr(), Some(record), argument),
                0
            );
            assert_eq!(
                kore_thread_cleanup_push(second.as_mut_ptr(), Some(record), argument),
                0
            );
            assert_eq!(
                kore_thread_cleanup_pop(first.as_mut_ptr(), 1),
                KORE_THREAD_EINVAL
            );
            assert_eq!(kore_thread_cleanup_pop(second.as_mut_ptr(), 1), 0);
        }
        finish(false);
        assert_eq!(trace.0, [0]);
    }
    #[test]
    fn concurrent_key_reuse_never_pairs_a_new_destructor_with_an_old_value() {
        use core::sync::atomic::AtomicBool;
        use std::sync::Arc;
        unsafe extern "C" fn old(pointer: *mut c_void) {
            assert_eq!(unsafe { *pointer.cast::<usize>() }, 0);
        }
        unsafe extern "C" fn replacement(pointer: *mut c_void) {
            assert_eq!(unsafe { *pointer.cast::<usize>() }, 1);
        }
        let context = Context::new();
        context.enter();
        for _ in 0..16 {
            let mut key = 0;
            let mut value = 0usize;
            unsafe {
                assert_eq!(kore_thread_key_create(&mut key, Some(old)), 0);
                assert_eq!(kore_thread_setspecific(key, (&raw mut value).cast()), 0);
            }
            let go = Arc::new(AtomicBool::new(false));
            let worker_go = go.clone();
            let worker = std::thread::spawn(move || {
                while !worker_go.load(Ordering::Acquire) {
                    std::thread::yield_now();
                }
                assert_eq!(kore_thread_key_delete(key), 0);
                let mut new_key = 0;
                assert_eq!(
                    unsafe { kore_thread_key_create(&mut new_key, Some(replacement)) },
                    0
                );
                assert_eq!(kore_thread_key_delete(new_key), 0);
            });
            go.store(true, Ordering::Release);
            finish(false);
            worker.join().unwrap();
        }
    }
}
